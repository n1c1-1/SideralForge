"""The blocks of the graph and how the workflows share them.

A workflow is a chain of blocks, named as in the standard tools (Calibration, Registration, Selection, Stacking,
Astrometry, Color, Denoise, Stars, Gradient, Output). The engine's internal steps (the half-stacks) are not blocks:
they are shown inside the block that makes them and as a wire to the block that uses them.

The Stars block gives two images, the starless image and the stars alone: two wires that go to Post as layers, or into
a Stars and background block of the Composition section (between the outputs and Post), which recombines them. Those
blocks are Post's layers of part "combined" (``Graph.compositions``): the graph and Post show the same composition.

The order is the engine's real one: Gradient comes right after Astrometry, before Color: it works on the stack, and
every later block reads the stack with the gradient removed. Two
workflows share a block as long as they compute it, and everything before it, the same way (the keys of the engine,
``api.blocks``). The graph is therefore a tree: one root (the frames), a branch where two workflows start to differ, one
leaf (the output) per workflow, all leaves going to Post.

The views use their own block keys (calibration, alignement, selection, empilement, astrometrie, couleur,
deconv, debruitage, etoiles, fond, post; composition for the Stars and background blocks): ``ENGINE_BLOCK`` maps the
engine's names to them. The stage names are the
engine's (``gradient``, ``render_simple``...).

Issue #43: a project whose sessions are calibrated each on its own starts with a Session block and a Calibration
block per session (``Graph.sessions``), whose wires join into one bar before Registration: the only place where the
graph joins. A project with one session (or its nights calibrated together, as before) has one Session block in
place of Frames. A workflow that starts from an image has an Image block on its own lane, wired straight to its first
block. A block the user cut is drawn hatched, its wire across it (``Node.cut``); a block that cannot run because of a
cut is greyed with its reason (``Node.grey``): the image passes through both.

This module also holds the block table (``STEPS``, ``BY_KEY``) that the Inspect strip and the panel share, and the
block-level answers (stages, state, duration, summary) over a ``ProjectModel``. ``Graph`` builds the tree of
``Node`` objects the graph and the Inspect strip draw.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .. import texts
from .i18n import _, hm, language, number, plural

# the columns of the graph, left to right; optional columns only appear when a workflow uses them
COLUMNS = ["sources", "calibration", "alignement", "selection", "crop", "empilement", "astrometrie", "fond", "couleur",
           "deconv", "debruitage", "etoiles", "synth", "post", "fusion", "composition", "final"]
OPTIONAL = {"crop", "debruitage", "deconv", "etoiles", "synth", "post", "fusion", "composition"}
PHASES = [("Pre-processing", "calibration", "empilement"), ("Linear processing", "astrometrie", "etoiles"),
          ("Composition", "fusion", "composition")]
TOOL_COLUMN = {"debruitage": "debruitage", "etoiles": "etoiles", "deconv": "deconv",
               "auto": "fond", "rbf": "fond", "simple": "fond"}
TITLES = {"align_images": "Align images", "sources": "Frames", "final": "Post", "image": "Image", "session": "Session", "fusion": "Combine channels",
          "calibration": "Calibration",
          "alignement": "Registration",
          "selection": "Selection", "crop": "Crop", "empilement": "Stacking", "astrometrie": "Astrometry", "fond": "Gradient",
          "couleur": "Color", "debruitage": "Denoise", "deconv": "Deconvolution", "etoiles": "Stars/background", "synth": "Synthetic stars",
          "post": "Output", "demi_piles": "Half-stacks", "composition": "Combine layers"}
# the optional blocks, in the toolbar's order
TOOLBOX = ["debruitage", "deconv", "etoiles"]
WARN_MARK = "⚠ "                       # a summary that opens with it: the block draws a warning icon (issue #58)
#: the two images of the Stars block (the parts of Post's layers) and their words on the wires
STAR_OUTPUTS = (("starless", "Starless image"), ("stars", "Stars alone"))
# the engine's block names (core/workflow.BLOCKS) and the views' keys
ENGINE_BLOCK = {"fusion": "fusion", "image": "image", "calibration": "calibration", "registration": "alignement",
                "selection": "selection", "crop": "crop",
                "stacking": "empilement", "astrometry": "astrometrie", "color": "couleur", "denoise": "debruitage",
                "deconv": "deconv", "synth": "synth", "stars": "etoiles", "gradient": "fond", "output": "post"}
KEY_ENGINE_BLOCK = {v: k for k, v in ENGINE_BLOCK.items()}
# the wires that are not the chain: what a block takes from a block further back, shown when it is selected
SIDE_INPUTS = {
    "debruitage": [("empilement", "half-stacks")],
    "couleur": [("astrometrie", "Gaia spectra")],
}
TRUNK = ["inventory", "calibrate", "register", "stack"]
#: the image blocks: placed in any order and more than once (issue #47), drawn in columns by their rank
IMAGE_KEYS = {"align_images", "astrometrie", "fond", "couleur", "debruitage", "deconv"}
#: the frame blocks, in their columns
FRAME_KEYS = ["calibration", "alignement", "selection", "crop", "empilement"]

# every state a block can be in: up to date, pending, outdated, and what a run adds
STATE_WORD = {"done": "Up to date", "pending": "Not computed yet", "stale": "Outdated", "running": "Running",
              "failed": "Failed", "interrupted": "Interrupted", "skipped": "Skipped"}
# worst first: a block is as bad as its worst stage in any of its workflows
STATE_ORDER = ["running", "failed", "interrupted", "stale", "skipped", "pending", "done"]

# the review points ("Review pause"): a dot like a debugger's, red while the run will stop there, yellow with
# the pause bars while the run is stopped there for the user, green once the user has checked these results
BREAK_COL, PAUSE_COL, REVIEWED_COL = "#e5484d", "#f5c542", "#5fbf7f"
# the most urgent first: what a block shows when its workflows are not all in the same case
REVIEW_ORDER = ["paused", "pending", "reviewed"]

# the block table: key, phase, title, kind (fixed | tool | auto | slot), engine stages, what it does
STEPS = [
    dict(key="session", phase=0, title="Session", kind="fixed", stages=[],
         what="One night: its light frames and the dark, flat and bias frames chosen for it, and the filter."),
    dict(key="image", phase=0, title="Image", kind="fixed", stages=["image"],
         what="An image file (a linear stack) this workflow starts from: the blocks after it read it as a stack."),
    dict(key="align_images", phase=0, title="Align images", kind="slot", stages=["image"],
         what="Brings several images onto one of them, through their plate solutions or their stars: each image keeps its "
              "own wire, the others follow the reference's frame, what overflows is cut."),
    dict(key="calibration", phase=1, title="Calibration", kind="fixed", stages=["inventory", "calibrate"],
         what="Dark and flat masters, then each light calibrated: thermal signal, vignetting and defective pixels "
              "removed."),
    dict(key="alignement", phase=2, title="Registration", kind="fixed", stages=["sessions", "register"],
         what="Measures the stars of each frame (size, transparency) and aligns all frames on the sharpest one."),
    dict(key="selection", phase=2, title="Selection", kind="fixed", stages=[], run=["register"],
         what="Which frames go into the stack: by star size and transparency thresholds, or one by one."),
    dict(key="crop", phase=2, title="Crop", kind="fixed", stages=[], run=["register"],
         what="The part of the field that is stacked and processed: a small target no longer costs the whole frame. "
              "Calibration and Registration keep the whole field."),
    dict(key="empilement", phase=3, title="Stacking", kind="fixed", stages=["stack"],
         what="Stacks the frames straight from the sensor's color pixels (drizzle without debayering) and rejects "
              "trails and outlier pixels."),
    dict(key="demi_piles", phase=3, title="Half-stacks", kind="auto", stages=[], run=["stack"],
         what="Two independent stacks of about 360 frames each: their difference measures the noise, which lets "
              "Denoise work without inventing anything. Computed during stacking, at no extra time."),
    dict(key="astrometrie", phase=4, title="Astrometry", kind="fixed",
         stages=["astrometry", "reference", "catalogs"],
         what="Finds where the image points on the sky by matching its stars to Gaia, then fetches the catalogue "
              "stars."),
    dict(key="couleur", phase=4, title="Color", kind="fixed", stages=["color"],
         what="Measures the sensor's response on the Gaia spectra of the field stars, then the matrix to the true "
              "sky colors."),
    dict(key="synth", phase=4, title="Synthetic stars", kind="fixed", stages=["synth_stars"],
         what="Draws the detected stars again, with the shape of this image or of the instrument."),
    dict(key="deconv", phase=4, title="Deconvolution", kind="tool", tool="deconv", stages=["deconv"],
         what="Sharpens the image with the point spread function measured on the stars."),
    dict(key="debruitage", phase=4, title="Denoise", kind="tool", tool="debruitage", stages=["denoise"],
         what="A network learns on the two half-stacks to predict one from the other: it removes the noise without "
              "inventing anything."),
    dict(key="etoiles", phase=4, title="Stars/background", kind="tool", tool="etoiles", stages=["star_removal"],
         what="A network trained on our stacks takes the stars off the image: two images, the starless image and the "
              "stars alone, sent to Post as they are or combined by a Combine layers block."),
    dict(key="fond", phase=4, title="Gradient", kind="slot", stages=["gradient"],
         what="Models the sky gradient (light pollution, the Moon, vignetting) on the stack and removes it from what "
              "every later block reads. Polynomial: a smooth surface fitted on the sky between the objects. "
              "Automatic: a more flexible surface that follows complex gradients."),
    dict(key="post", phase=5, title="Output", kind="fixed", stages=["render_simple"],
         what="Prepares the render: what this workflow gives in Post, its image and its sliders."),
    dict(key="fusion", phase=6, title="Combine channels", kind="slot", stages=["fusion", "render_simple"],
         what="Combines the outputs of several workflows (one per filter, or a dual-band image's channels) into one "
              "colour image with a palette (SHO, HSO, HOO, RGB, LRGB), aligned through their plate solutions."),
    dict(key="composition", phase=6, title="Combine layers", kind="slot", stages=[],
         what="Combines the Stars/background block's two images: the starless image and the stars, each stretched, the stars "
              "put back on top (Screen, Linear dodge or Lighten). Its settings are those of its layer in Post."),
]
BY_KEY = {b["key"]: b for b in STEPS}
# what the optional blocks say they do (the toolbar cards' tooltips)
TOOLS = {
    "debruitage": dict(title="Denoise", phase=4,
                       what="Removes the noise without inventing anything: a network learns on the two half-stacks "
                            "to predict one from the other."),
    "etoiles": dict(title="Stars/background", phase=4,
                    what="Takes the stars off the image with a trained network: the starless image and the stars "
                         "alone, two images for Post (combined by a Combine layers block, or as they are)."),
    "synth": dict(title="Synthetic stars", phase=4,
                  what="Draws the detected stars again, with the shape of this image or of the instrument."),
    "deconv": dict(title="Deconvolution", phase=4,
                   what="Sharpens the image with the point spread function measured on the stars."),
}


def kbase(key: str) -> str:
    """A view's block key without its session or its number: "calibration@2" → "calibration", "debruitage~2" →
    "debruitage"."""
    return key.partition("@")[0].partition("~")[0]


def knum(key: str) -> str:
    """The number of a graph's later node of a block ("2" for "debruitage~2", issue #47), "" otherwise."""
    return key.partition("@")[0].partition("~")[2]


def block_title(p, key: str) -> str:
    """The name of a block, translated. The output of a workflow carries the workflow's name; a block placed twice
    in a graph, its number ("Denoise 2")."""
    if key == "post" and p is not None:
        return p.name
    n = knum(key)
    base = kbase(key)                                 # a session's block: its block's name
    t = _(TITLES.get(base) or BY_KEY[base]["title"])
    if p is not None and getattr(p, "model", None) is not None:   # numbered left to right in its workflow, the
        try:                                                       # same name everywhere (GUI exploration 4.1)
            same = [k for k in chain(p) if kbase(k) == base]
        except (KeyError, ValueError, AttributeError):
            same = []
        if key in same:
            return f"{t} {same.index(key) + 1}" if len(same) > 1 else t
    return f"{t} {n}" if n else t


def calibration_counts(darks, flats, bias=None, gone=()) -> str:
    """"3 darks · 1 flat (· 0 bias)": each count with its noun in the right form, as the lights' (GUI exploration 2).
    A kind in ``gone`` (``gone_kinds``) without a frame: its folder is not found, not empty (GUI exploration 4)."""
    def one(kind, n, counted, missing):
        return _(missing) if kind in gone and not int(n or 0) else counted
    parts = [one("darks", darks, plural(int(darks or 0), "{n} dark", "{n} darks"), "darks not found"),
             one("flats", flats, plural(int(flats or 0), "{n} flat", "{n} flats"), "flats not found")]
    if bias is not None:
        parts.append(one("bias", bias, plural(int(bias or 0), "{n} bias", "{n} bias"), "bias not found"))
    return " · ".join(parts)


def gone_kinds(s, sids=None) -> set[str]:
    """The kinds of frames (lights, darks...) of these sessions (all when None) one of whose folders is not there:
    moved or renamed outside, its disk unplugged (GUI exploration 4). Each folder is looked at once, never each file."""
    if not hasattr(s, "session") or getattr(s, "p", None) is None:
        return set()
    out: set[str] = set()
    for se in s.p.sessions:
        if sids is not None and se.id not in sids:
            continue
        for kind in ("lights", "darks", "flats", "bias", "flat_darks"):
            folders = {(p if not p.suffix else p.parent) for p in (Path(s.p.abspath(e)) for e in getattr(se, kind))}
            if any(not f.exists() for f in folders):
                out.add(kind)
    return out


def own_calibration_summary(s, p, sid: int) -> str:
    """Under a Calibration placed in parallel: its counts, and whether they are its own or still its Session's."""
    n = {k: len(s.calibration_files(p, sid, k)) for k in ("darks", "flats")}
    follows = all(s.calibration_follows(p, sid, k) for k in ("darks", "flats", "bias", "flat_darks"))
    return (_("as Session") if follows else _("own files")) + " · " + calibration_counts(n["darks"], n["flats"])


def session_summary(s, sid: int) -> str:
    """Under a Session block's name: its lights, its calibration frames, its filter; a kind whose folder is not
    found says so (GUI exploration 4)."""
    ses = next((x for x in s.sessions if x["id"] == sid), {})
    gone = gone_kinds(s, {sid})
    if not int(ses.get("lights") or 0):                # issue #48: a Session placed empty says what it needs
        se = s.session(sid) if hasattr(s, "session") else None
        flt = getattr(se, "filter", "")
        return (_("lights folder not found") if "lights" in gone else _("lights to add")) + (f" · {flt}" if flt else "")
    text = plural(int(ses.get("lights") or 0), "{n} light", "{n} lights")
    text += " · " + calibration_counts(ses.get("darks", 0), ses.get("flats", 0), gone=gone)
    flt = getattr(s.session(sid), "filter", "") if hasattr(s, "session") else ""
    warn = WARN_MARK if hasattr(s, "file_warnings") and s.file_warnings(sid) else ""   # issue #58: see its panel
    return warn + text + (f" · {flt}" if flt else "")


def node_title(s, n) -> str:
    """The name a block shows: a session's name for its Session block, the workflow's for its Output, the frames'
    block named after its one session (or "All nights")."""
    if n.key == "session":
        return s.session(n.sid).name or _("Session {n}").format(n=n.sid)
    if n.key == "post" and not n.handle:              # issue #47: the block's name; the workflow's under it
        return _("Output")
    if n.key == "composition" and n.num:              # several Combine layers blocks: all numbered, top to bottom, as
        return f"{block_title(None, 'composition')} {n.num}"     # Gradient 1, Gradient 2 (control round)
    if n.num and n.key in FRAME_KEYS:                 # frame blocks placed in parallel, the same way
        return f"{block_title(None, n.key)} {n.num}"
    if n.key in ("fusion", "image") and n.paths:      # several Fusions, several Images: numbered in the project's
        src = "fusion" if n.key == "fusion" else "path"            # order (exploration 2, control round); a branch
        bk = getattr(s, "block_key", None)                         # shares its source's block: counted once (round 4:
        ident = (lambda q: bk(q, n.key)) if bk is not None else (lambda q: q.pid)   # one Image became "Image 1")
        same = []
        for q in getattr(s, "paths", []):
            if isinstance(q.source, dict) and src in q.source and ident(q) not in same:
                same.append(ident(q))
        mine = ident(n.paths[0])
        if len(same) > 1 and mine in same:
            return f"{block_title(None, n.key)} {same.index(mine) + 1}"
    if n.paths and "~" in n.key or (n.paths and n.key in ("fond", "couleur", "debruitage")):
        # issue #47: a block placed twice, numbered left to right; shared by a branch that has it twice: numbered
        # as in that branch ("Gradient 1" before its "Gradient 2", control round)
        base = kbase(n.key)
        p = max(n.paths, key=lambda q: sum(kbase(k) == base for k in _safe_chain(q)))
        return block_title(p, n.key)
    if n.key == "sources" and hasattr(s, "session") and s.sessions:
        if len(s.sessions) == 1:
            return s.sessions[0]["name"] or _("Session")
        if getattr(s, "stack_mode", "together") == "together":
            return _("All nights")
    return block_title(n.paths[0] if n.paths else None, n.key)


def review_title(key: str) -> str:
    """The name of a block in the texts about a pause (the output of a workflow is just "Output" there)."""
    return _("Output") if key == "post" else block_title(None, key)


def _safe_chain(p) -> list[str]:
    try:
        return chain(p) if getattr(p, "model", None) is not None else []
    except (KeyError, ValueError, AttributeError):
        return []


def chain(p) -> list[str]:
    """The block keys of a workflow, in order (the frames excluded, its Output last); a workflow pulled to the post
    from a block stops there, then has its output."""
    m = p.model
    return list(m.display_keys(p) if hasattr(m, "display_keys") else m.block_keys(p))


# ------------------------------------------------------------------ block-level answers over a model
def stages_of(s, p, key: str) -> list[str]:
    """The engine stages of this block in the workflow's chain; a session's block ("calibration@2"), that session's
    instances (issue #43)."""
    allst = s.all_stages(p)
    k, _at, sid = key.partition("@")
    k, _t, num = k.partition("~")                     # a graph's later node (issue #47): its own stage
    return [x for x in (f"{st}@{sid}" if sid else f"{st}~{num}" if num else st for st in BY_KEY[k]["stages"])
            if x in allst]


def run_stages(s, p, key: str) -> list[str]:
    """What Run to here runs for this block (Selection: Registration)."""
    return BY_KEY[kbase(key)].get("run") or stages_of(s, p, key)


def block_state(s, p, key: str) -> str:
    """The state of a block in one workflow: the worst of its stages (Selection follows Registration)."""
    st = run_stages(s, p, key)
    if not st:
        return "done"
    states = {s.state(p, x) for x in st}
    return next((w for w in STATE_ORDER if w in states), "pending")


def block_seconds(s, p, key: str) -> float:
    return sum(s.seconds(x, p) for x in stages_of(s, p, key))


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", " ")


def drizzle_auto(s, p) -> bool:
    """Whether the Stacking's drizzle scale is the engine's automatic one and not known yet (the frames are not aligned):
    the block says "auto", never the default 1.5 the engine would then replace (GUI exploration 2: "×1,5" before the
    run, "×1" after, without any action)."""
    if "scale" in (p.params.get("stack") or {}) or not hasattr(s, "param_info"):
        return False
    info = s.param_info(p, "stack", "scale")
    return info is None or (info.origin != "user" and not info.suggestion_known)


def method_summary(s, p, key: str) -> str:
    from .model import method_of, methods_of
    m = method_of(p, key)                       # "fond~2": its own node's method (round 3)
    return _(next(x["short"] for x in methods_of(kbase(key)) if x["key"] == m))


@dataclass
class Node:
    key: str                                   # step key, or "sources" / "final"
    paths: list = field(default_factory=list)  # the workflows going through this block
    parent: Node | None = None
    children: list = field(default_factory=list)
    col: int = 0
    y: float = 0.0
    w: float = 0.0                             # its width in the graph (the column's)
    lid: int | None = None                     # a Stars and background block: its layer in Post
    handle: bool = False                       # the Stars block of a workflow whose Output block it holds (#17)
    sid: int | None = None                     # a Session block, or a session's Calibration block: its session
    cut: bool = False                          # cut by the user in its workflows (issue #43)
    grey: str = ""                             # why it cannot run (a cut before it), translated; "" when it can
    broken: str = ""                           # why it cannot run at all (a fusion's channel gone...): circled in red
    missing: str = ""                          # why it cannot run on this computer: the data of the sky it needs is
    #                                            not on the disk (core/sky_data.py); left out of the runs, the rest runs
    slot: int = -1                             # an image block: its rank among its workflow's image blocks (#47)
    num: int = 0                               # a Combine layers block among several, or a frame block among those
    #                                            placed in parallel: its number (control round)
    _bk: object = None

    @property
    def ref(self) -> str:
        """The key the views select the block by: its key, with its session ("calibration@2")."""
        return f"{self.key}@{self.sid}" if self.sid is not None else self.key

    @property
    def shared(self) -> bool:
        return len(self.paths) > 1


class Graph:
    """The tree of blocks for the current state of the project. A second argument (tests pass None) is not read.

    Beside the tree, what goes to Post (``Post``'s layers, ``project.post_layers``; a workflow that has none yet
    counts with the one Post will give it: its Stars and background block with Stars, else its Output): the Stars and
    background blocks (``compositions``: their nodes, after the outputs), the Stars block's images sent to Post as
    they are (``direct``: the Stars node and the part), and the workflows whose Output image is a layer (``to_post``).
    """

    def __init__(self, project, _unused=None):
        self.s = project
        frames = [p for p in project.paths if getattr(p, "source", None) is None]
        self.root = Node("sources", frames)
        self.final = Node("final", list(project.paths))
        self.leaves: dict[int, Node] = {}
        self.nodes: list[Node] = [self.root]
        self.by_path: dict[int, dict[str, Node]] = {}
        #: issue #43: (Session block, its Calibration block) per session calibrated on its own
        self.sessions: list[tuple[Node, Node]] = []
        #: the Calibration blocks placed in parallel on a night, drawn under its own (session -> blocks)
        self.calib_forks: dict[int, list[Node]] = {}
        sids = project.separate_sessions() if hasattr(project, "separate_sessions") else []
        uses = getattr(project, "sessions_of", None)
        for sid in sids:
            mine = [p for p in frames if uses is None or sid in uses(p)]       # the workflows that stack it
            model_ = getattr(project, "model", None) or project
            calib = [q for q in mine if not getattr(model_, "is_graph", lambda x: False)(q)
                     or any(n["block"] == "calibration" for n in model_.graph(q))]      # #47: placed, or not drawn
            # switched off: one Calibration block for every session of the workflow, each drawn so (GUI exploration 2)
            off = bool(calib) and all(getattr(model_, "is_cut", lambda q, k: False)(q, "calibration") for q in calib)
            fork = getattr(model_, "calibration_fork", lambda q, n: None)
            sn, cn = Node("session", mine, sid=sid), Node("calibration", [q for q in calib if fork(q, sid) is None],
                                                          sid=sid, cut=off)
            self.sessions.append((sn, cn))
            self.nodes += [sn, cn]
            apart: dict[int, list] = {}                # night's Calibrations placed in parallel, each with its workflows
            for q in calib:
                if fork(q, sid) is not None:
                    apart.setdefault(fork(q, sid), []).append(q)
            for mark in sorted(apart):
                n = Node("calibration", apart[mark], sid=sid, cut=all(getattr(model_, "is_cut", lambda q, k: False)(
                    q, "calibration") for q in apart[mark]))
                self.calib_forks.setdefault(sid, []).append(n)
                self.nodes.append(n)
        model = getattr(project, "model", None) or project
        for p in project.paths:
            cur = self.root
            self.by_path[p.pid] = {"sources": self.root, "final": self.final}
            for sn, cn in self.sessions:
                if p in sn.paths:
                    self.by_path[p.pid][sn.ref], self.by_path[p.pid][cn.ref] = sn, cn
                    self.by_path[p.pid][cn.ref] = next((n for n in self.calib_forks.get(sn.sid, []) if p in n.paths), cn)
            rank = 0
            for k in chain(p):
                cut = bool(getattr(model, "is_cut", None)) and model.is_cut(p, k)
                grey = model.greyed_reason(p, k) if hasattr(model, "greyed_reason") else ""
                if cut or grey:                         # not in the engine's chain (or a pass-through): its own branch
                    bk = ("cut" if cut else "grey", k, project.block_key(p, k), cur._bk)
                else:
                    bk = project.block_key(p, k)
                nxt = next((c for c in cur.children if c._bk == bk), None)
                if nxt is None:
                    nxt = Node(k, [], cur, cut=cut, grey=grey)
                    nxt._bk = bk
                    cur.children.append(nxt)
                    self.nodes.append(nxt)
                nxt.paths.append(p)
                self.by_path[p.pid][k] = nxt
                if kbase(k) in IMAGE_KEYS:              # issue #47: its column, by its rank in the workflow
                    nxt.slot = rank
                    rank += 1
                cur = nxt
            self.leaves[p.pid] = cur
        # issue #43: a fusion's block holds its workflow (its Output: no block of its own); its inputs, the outputs of
        # the workflows it combines
        self.fusion_inputs: dict[int, list[Node]] = {}
        # a session whose data cannot be computed (a folder not there, a frame that cannot be read, darks of another
        # size): its block circled in red with the reason, Run refused (round 2)
        why = getattr(project, "session_problem", None)
        if why is not None:
            for sn, _cn in self.sessions:
                sn.broken = why(sn.sid)
            if not self.sessions and frames:
                self.root.broken = next((w for s in project.p.sessions if (w := why(s.id))), "")
        for p in project.paths:
            src = getattr(p, "source", None)
            if isinstance(src, dict) and "fusion" not in src and self.by_path[p.pid].get("image") is not None:
                # its file moved, unplugged or damaged: circled in red with the reason, Run refused (round 2)
                self.by_path[p.pid]["image"].broken = getattr(project, "image_problem", lambda q: "")(p)
            if isinstance(src, dict) and "fusion" in src:
                n = self.by_path[p.pid].get("fusion")
                if n is not None:
                    n.handle = True
                    n.broken = getattr(project, "fusion_problem", lambda q: "")(p)   # circled in red (1.1)
                    self.by_path[p.pid]["post"] = n
                    if src["fusion"].get("stars"):       # option A: its starless image and its star layer
                        self.by_path[p.pid]["etoiles"] = n
        # the data of the sky a block needs and this computer has not (the offline catalogue not installed): marked
        # with the reason; the runs leave it out with what follows it, and run the rest
        sky = getattr(project, "sky_missing", None)
        if sky is not None:
            for n in self.nodes:
                if n.paths and n.key not in ("sources", "final", "session") and not n.cut and not n.grey:
                    n.missing = next((w for q in n.paths if (w := sky(q, n.key))), "")
        self.compositions: list[Node] = []
        self.direct: list[tuple[Node, str]] = []
        self.to_post: set[int] = set()
        #: the layers of Post, top first: (name, workflow id or None for an image sent from Inspect)
        self.post_layers: list[tuple[str, int | None]] = []
        layers = project.post_layers() if hasattr(project, "post_layers") else []
        have = {lay.get("pid") for lay in layers if lay.get("kind") == "workflow"}
        from .model import POST_PART_NAMES
        feeds = getattr(project, "feeds_fusion", None)
        removed = getattr(project, "post_removed", lambda: set())()     # as Post fills them (post3): every image
        for p in project.paths:                       # not taken out by the user (#52), an output sent from a block
            default = "combined" if p.opts.get("etoiles") == "retirer" else None
            if p.pid not in have and ((p.pid, default) not in removed or p.end) and \
                    not (feeds is not None and feeds(p)) and \
                    getattr(project, "gives_post", lambda q: True)(p):   # #49: no image yet, no link to Post
                stars = p.opts.get("etoiles") == "retirer" and self.node_of(p, "etoiles") is not None
                layers.insert(0, {"kind": "workflow", "pid": p.pid, "part": "combined" if stars else None, "id": None,
                                  "name": p.name + (" · " + POST_PART_NAMES["combined"] if stars else "")})
        for lay in layers:
            if lay.get("kind") == "image":
                self.post_layers.append((str(lay.get("name") or "?"), None))
            if lay.get("kind") != "workflow":
                continue
            p = next((q for q in project.paths if q.pid == lay.get("pid")), None)
            if p is None:
                continue
            self.post_layers.append((str(lay.get("name") or p.name), p.pid))
            part = lay.get("part")
            if part is None:
                self.to_post.add(p.pid)
                continue
            src = self.node_of(p, "etoiles")
            if src is None:
                continue
            if part == "combined":
                n = Node("composition", [p], src, lid=lay.get("id"))
                n._bk = ("composition", lay.get("id"), p.pid)
                self.nodes.append(n)
                self.compositions.append(n)
            elif (src, part) not in self.direct:
                self.direct.append((src, part))
        #: the workflows whose image comes to Post with their lights: no layer yet, their wire drawn dashed
        waits = getattr(project, "awaits_post", lambda q: False)
        self.awaits_post = {p.pid for p in project.paths if waits(p) and not (feeds is not None and feeds(p)) and
                            (p.pid, "combined" if p.opts.get("etoiles") == "retirer" else None) not in removed}
        # GUI exploration 4, Q10: a workflow with Stars but no lights yet has no layer, so no Combine layers block; it is
        # shown greyed from the start (no wire, never chosen), the one Post will give it once the lights are added
        gives = getattr(project, "gives_post", lambda q: True)
        for p in project.paths:
            if p.opts.get("etoiles") == "retirer" and p.pid not in have and getattr(p, "source", None) is None \
                    and (p.pid, "combined") not in removed and not gives(p):
                src = self.node_of(p, "etoiles")
                if src is not None:
                    self.nodes.append(Node("composition", [p], src, grey=_("after the lights")))
        self._fold_outputs(project)
        self._number_parallel(model)
        for p in project.paths:                       # several Combine layers blocks of a workflow: each its number, its
            mine = [n for n in self.compositions if p in n.paths]   # layer's ("Full · Combine layers 2"), the first 1
            if len(mine) > 1:                         # (control round: as Gradient 1, Gradient 2)
                for n in mine:
                    name = str((project.post_layer(n.lid) or {}).get("name") or "") \
                        if n.lid is not None and hasattr(project, "post_layer") else ""
                    tail = name.rpartition(" ")[2]
                    n.num = int(tail) if tail.isdigit() else 1
        for p in project.paths:                       # a fusion's inputs: its channels' last blocks (after the fold)
            src = getattr(p, "source", None)
            n = self.by_path[p.pid].get("fusion") if isinstance(src, dict) and "fusion" in src else None
            if n is not None:
                self.fusion_inputs[id(n)] = list({id(self.leaves[i["pid"]]): self.leaves[i["pid"]]
                                                  for i in src["fusion"]["inputs"]
                                                  if i["pid"] in self.leaves and self.leaves[i["pid"]] is not n}.values())
        self.nodes.append(self.final)

    def _number_parallel(self, model):
        """Frame blocks placed in parallel: all of a kind numbered, as a block placed twice in a chain, so that two
        alike are never two of the same name. A night's Calibrations, among themselves."""
        fork = getattr(model, "node_fork", lambda q, k: None)
        night = getattr(model, "own_calibration_night", lambda q, s=None: None)

        def number(group):
            for i, n in enumerate(group):
                n.num = i + 1
        for k in FRAME_KEYS:
            shown = [n for n in self.nodes if n.key == k and n.paths and n.sid is None]
            if len(shown) > 1 and any(fork(n.paths[0], k) is not None or
                                      (k == "calibration" and night(n.paths[0]) is not None) for n in shown):
                number(shown)
        for sn, cn in self.sessions:
            if sn.sid in self.calib_forks:
                number([n for n in [cn] + self.calib_forks[sn.sid] if n.paths])

    def _fold_outputs(self, project):
        """Issues #17, #49: a workflow has no Output block of its own in the graph: its last block holds it
        (``node_of(p, "post")`` answers it; the workflow's handle, its name and menu, when the block is its own; the
        wire to Post carries that menu otherwise)."""
        for p in project.paths:                           # the Output is never a block: hidden in the wire of its
            po = self.by_path[p.pid].get("post")          # last block, always (#49; GUI exploration 2: a shared last
            last = po.parent if po is not None else None  # block, a branch sent to Post, showed it again)
            if last is None or po.key != "post":
                continue
            if po in last.children:                       # two workflows without blocks (images already stretched)
                # share theirs: taken out once. A block drawn after it (Synthetic stars, after the Stars/background that
                # holds the Output) takes its place under the block before: the Output is never a block, whatever
                # follows it (it came back each time a block was added after it: tests/client/test_no_output_block.py)
                at = last.children.index(po)
                last.children[at:at + 1] = po.children
                for c in po.children:
                    c.parent = last
                self.nodes.remove(po)
            self.by_path[p.pid]["post"] = last
            if self.leaves.get(p.pid) is po:
                self.leaves[p.pid] = last
            if len(last.paths) == 1 and last.key not in ("sources", "session"):
                last.handle = True                        # its workflow's menu on it; a shared one: on the wire

    def node_of(self, p, key: str, lid: int | None = None) -> Node | None:
        """The node of a block of a workflow; a Stars and background block by its Post layer (``lid``; else the
        workflow's first one)."""
        if key == "sources":
            return self.root
        if key == "final":
            return self.final
        if "@" in key:                                  # a session's block (issue #43)
            k, sid = key.split("@", 1)
            mine = self.by_path.get(p.pid, {}).get(key) if p is not None else None   # beside its own Calibration
            return mine or next((n for pair in self.sessions for n in pair if n.key == k and str(n.sid) == sid), None)
        if p is None:
            return None
        if key == "composition":
            mine = [n for n in getattr(self, "compositions", []) if p in n.paths]
            return next((n for n in mine if n.lid == lid), mine[0] if mine else None)
        return self.by_path.get(p.pid, {}).get(key)

    # ------------------------------------------------------------ state of a block
    def state(self, node: Node) -> str:
        """done | pending | stale | running | failed | interrupted | skipped, for the workflows going through it."""
        if node.key == "sources":
            return "done" if self.s.sources_loaded else "pending"
        if node.key == "session":
            return "done" if self.s.session(node.sid).lights else "pending"
        if node.sid is not None:                      # a session's Calibration: its own stages in every workflow
            states = {self.s.state(p, x) for p in node.paths for x in self.session_stages(p, node.sid)}
            return next((w for w in STATE_ORDER if w in states), "pending")
        if node.grey or (node.cut and node.key not in ("calibration", "alignement")):
            return "skipped" if node.grey else "done"
        if node.key == "final":
            return "done" if self.s.paths and all(p.rendered for p in self.s.paths) else "pending"
        if node.key == "composition":                 # computed in Post from the Stars block's images
            return block_state(self.s, node.paths[0], "etoiles")
        states = {block_state(self.s, p, node.key) for p in node.paths}
        # the Output hidden in its wire (#49): its image to Post is this block's too, not done until it is
        states |= {st for st in (block_state(self.s, p, "post") for p in node.paths
                                 if node.key != "post" and p.pid in self.to_post and
                                 self.by_path.get(p.pid, {}).get("post") is node)
                   if st != "skipped"}                    # skipped for a failure elsewhere: the block itself is done
        return next((w for w in STATE_ORDER if w in states), "pending")

    def session_stages(self, p, sid) -> list[str]:
        """The stages of a session's Calibration in a workflow: its instances, or the workflow's own when it stacks
        that session alone (one filter's)."""
        inst = [f"inventory@{sid}", f"calibrate@{sid}"]
        allst = self.s.all_stages(p) if hasattr(self.s, "all_stages") else []
        return inst if any(x in allst for x in inst) else ["inventory", "calibrate"]

    def breakpoint(self, node: Node) -> tuple[str, int, int] | None:
        """The review point of a block: (state, how many of its workflows have it, how many go through it), the state
        being the most urgent of theirs (paused, pending, reviewed); None when no workflow pauses after it."""
        review = getattr(self.s, "review_state", None)
        if review is None or node.key in ("sources", "final", "composition", "session") or not node.paths:
            return None
        have = [st for st in (review(p, node.ref) for p in node.paths) if st]     # a session's Calibration: its own
        if not have:
            return None
        return next(w for w in REVIEW_ORDER if w in have), len(have), len(node.paths)

    def seconds(self, node: Node) -> float:
        if node.key in ("sources", "final", "composition", "session") or not node.paths or node.grey:
            return 0.0
        if node.sid is not None:
            return sum(self.s.seconds(x, node.paths[0]) for x in self.session_stages(node.paths[0], node.sid))
        return block_seconds(self.s, node.paths[0], node.key)

    def summary(self, node: Node) -> str:
        """Under a block's title; the Stars block that holds its workflow's Output says the workflow's name first."""
        text = self._summary(node)
        if node.key == "fusion":
            return text
        return f"{node.paths[0].name} · {text}" if node.handle and node.paths else text

    def _summary(self, node: Node) -> str:
        s = self.s
        if node.grey:
            return node.grey
        title = node_title(s, node) if node.cut else ""      # "désactivé" agrees with the block's name
        if node.cut and node.key not in ("calibration", "alignement"):
            return _("off: image passes through").format(block=title)
        if node.key == "session":
            return session_summary(s, node.sid)
        if node.sid is not None and node.cut:
            return _("off: frames as they are").format(block=title)
        if node.key == "calibration" and node.paths and not node.cut and hasattr(s, "own_calibration_night"):
            night = node.sid if node.sid is not None else s.own_calibration_night(node.paths[0])
            if night is not None and s.own_calibration_night(node.paths[0], night) is not None:
                return own_calibration_summary(s, node.paths[0], night)
        if node.sid is not None:
            ses = next((x for x in s.sessions if x["id"] == node.sid), {})
            return calibration_counts(ses.get("darks", 0), ses.get("flats", 0), gone=gone_kinds(s, {node.sid}))
        if node.cut:
            return (_("off: frames as they are") if node.key == "calibration" else
                    _("off: frames as aligned")).format(block=title)
        if node.key == "image":
            src = node.paths[0].source or {}
            if not src.get("path"):
                return _("file to choose")
            return Path(str(src.get("path") or "")).name + ("  · " + _("stretched") if src.get("stretched") else "")
        if node.key == "fusion":
            f = (node.paths[0].source or {}).get("fusion") or {}
            text = f"{f.get('palette', '')} · " + ", ".join(i["role"] for i in f.get("inputs") or [])
            name = node.paths[0].name
            return text if name == f.get("palette") else f"{name} · {text}"
        if node.key == "sources":
            if not s.sources_loaded:
                return session_summary(s, s.sessions[0]["id"]) if len(s.sessions) == 1 else _("to add")
            if len(s.sessions) == 1 or getattr(s, "stack_mode", "together") == "together":
                if len(s.sessions) > 1:
                    return plural(len(s.sessions), "{n} night · {lights} lights", "{n} nights · {lights} lights",
                                  lights=fmt_int(s.n_lights()))
                return session_summary(s, s.sessions[0]["id"])
            return plural(len(s.sessions), "{n} session · {lights} lights", "{n} sessions · {lights} lights",
                          lights=fmt_int(s.n_lights()))
        if node.key == "final":
            return plural(len(self.post_layers), "{n} layer", "{n} layers")
        p = node.paths[0]
        k = kbase(node.key)
        if k == "post":
            if p.end:
                return _("from {block}").format(block=block_title(p, p.end))
            # with Stars, the image goes to the Stars block (and to Post only when a layer shows it as it is)
            # with Stars its images go to Post from the Stars block: this block is only the workflow (issue #8)
            return p.name + " · " + (_("to Post") if p.pid in self.to_post or self.node_of(p, "etoiles") is None
                                     else _("workflow"))
        if k == "composition":
            lay = self.s.post_layer(node.lid) if node.lid is not None and hasattr(self.s, "post_layer") else None
            return _("stars in {mode}").format(mode=_((lay or {}).get("blend") or "Screen"))
        if k == "calibration":
            inv = s.info()
            return calibration_counts(inv.get("n_darks", 0), inv.get("n_flats", 0), inv.get("n_bias", 0),
                                      gone=gone_kinds(s))
        if k == "alignement":
            left = getattr(getattr(s, "_facts", None), "register", {}).get("not_aligned")
            return _("{n} not aligned").format(n=left) if left else _("ref: sharpest frame")
        if k == "crop":
            z = s.param_value(p, "stack", "zone")
            return _("{w} x {h} px").format(w=z[2] - z[0], h=z[3] - z[1]) if z else _("whole field")
        if k == "selection":
            rule = s.selection_rule(p)
            if rule == "min_weight":
                return _("all frames · weight ≥ 5 %")
            if rule == "sota":
                return _("k × σ filters · weight ≥ 5 %")
            t, f = s.thresholds_now()
            unit = s.thresholds_unit()
            return (_("transparency ≥ {t} % · stars ≤ {f} px").format(t=f"{t:.0f}", f=number(f)) if unit == "px" else
                    _("transparency ≥ {t} % · stars ≤ {f} × median").format(t=f"{t:.0f}", f=number(f)))
        if k == "empilement":
            if getattr(s, "colour_frames", lambda _p: False)(p):      # issue #57: no demosaicing, no drizzle
                txt = _("Classic stack · colour frames")
            elif s.param_value(p, "stack", "method") == "classic":
                txt = _("classic stack · {how}").format(how=texts.choice_label(
                    str(s.param_value(p, "stack", "classic_debayer") or "rcd"), language()))
            else:
                kappa = number(s.param_value(p, "stack", "kappa") or 5, 0)
                if drizzle_auto(s, p):          # the engine sets it from the frames once they are aligned
                    txt = _("drizzle auto · rejection at {kappa} σ").format(kappa=kappa)
                else:
                    sc = number(s.param_value(p, "stack", "scale") or 1.0, 2).rstrip("0").rstrip(",.")
                    txt = _("drizzle ×{scale} · rejection at {kappa} σ").format(scale=sc, kappa=kappa)
            if any(q.opts["debruitage"] for q in node.paths):
                txt += " · " + _("+ half-stacks")
            return txt
        if k == "debruitage":                   # a block placed twice: its own method (round 3)
            from .model import engine_name, method_of
            stage = engine_name(node.key) if "~" in node.key else "denoise"
            fast = " · " + _("fast pass") if s.param_value(p, stage, "fast") else ""
            if method_of(p, node.key) == "affine":
                return _("pretrained + adapted · {length}").format(length=texts.choice_label(
                    str(s.param_value(p, stage, "adaptation") or "short"), language())) + fast
            return _("pretrained network") + fast
        if k in ("fond", "etoiles", "deconv", "synth"):
            return method_summary(s, p, node.key)
        if k == "astrometrie":                  # a result once computed; before, what it will do (round 3)
            return _("solved · Gaia DR3") if block_state(s, p, k) == "done" else _("solves the field · Gaia DR3")
        if k == "couleur":
            return _("Gaia stars · white balance")
        return ""

    def side_inputs(self, node: Node):
        """[(source node, label)] of the wires that come from further back, for the selected block."""
        if not node.paths or node.key in ("sources", "final"):
            return []
        p = node.paths[0]
        pairs = list(SIDE_INPUTS.get(kbase(node.key), []))
        out = []
        for k, lab in pairs:
            src = self.node_of(p, k)
            if src is not None and src is not node.parent:
                out.append((src, _(lab)))
        return out


__all__ = ["session_summary", "node_title", "COLUMNS", "OPTIONAL", "PHASES", "TOOL_COLUMN", "TITLES", "TOOLBOX", "STEPS",
           "BY_KEY", "TOOLS",
           "STATE_WORD", "STATE_ORDER", "ENGINE_BLOCK", "Graph", "Node", "block_title", "chain", "stages_of",
           "run_stages", "block_state", "block_seconds", "hm", "BREAK_COL", "PAUSE_COL", "REVIEWED_COL", "review_title",
           "IMAGE_KEYS", "FRAME_KEYS", "kbase", "knum"]
