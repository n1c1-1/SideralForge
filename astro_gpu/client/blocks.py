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

This module also holds the block table (``STEPS``, ``BY_KEY``) that the Inspect strip and the panel share, and the
block-level answers (stages, state, duration, summary) over a ``ProjectModel``. ``Graph`` builds the tree of
``Node`` objects the graph and the Inspect strip draw.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import texts
from .i18n import _, hm, language, number, plural

# the columns of the graph, left to right; optional columns only appear when a workflow uses them
COLUMNS = ["sources", "calibration", "alignement", "selection", "empilement", "astrometrie", "fond", "couleur",
           "deconv", "debruitage", "etoiles", "post", "composition", "final"]
OPTIONAL = {"debruitage", "deconv", "etoiles", "post", "composition"}
PHASES = [("Pre-processing", "calibration", "empilement"), ("Linear processing", "astrometrie", "etoiles"),
          ("Composition", "composition", "composition")]
TOOL_COLUMN = {"debruitage": "debruitage", "etoiles": "etoiles", "deconv": "deconv",
               "auto": "fond", "rbf": "fond", "simple": "fond"}
TITLES = {"sources": "Frames", "final": "Post", "calibration": "Calibration", "alignement": "Registration",
          "selection": "Selection", "empilement": "Stacking", "astrometrie": "Astrometry", "fond": "Gradient",
          "couleur": "Color", "debruitage": "Denoise", "deconv": "Deconvolution", "etoiles": "Split stars & background",
          "post": "Output", "demi_piles": "Half-stacks", "composition": "Combine stars & background layers"}
# the optional blocks, in the toolbar's order
TOOLBOX = ["debruitage", "deconv", "etoiles"]
#: the two images of the Stars block (the parts of Post's layers) and their words on the wires
STAR_OUTPUTS = (("starless", "Starless image"), ("stars", "Stars alone"))
# the engine's block names (core/workflow.BLOCKS) and the views' keys
ENGINE_BLOCK = {"calibration": "calibration", "registration": "alignement", "selection": "selection",
                "stacking": "empilement", "astrometry": "astrometrie", "color": "couleur", "denoise": "debruitage",
                "stars": "etoiles", "gradient": "fond", "output": "post"}
KEY_ENGINE_BLOCK = {v: k for k, v in ENGINE_BLOCK.items()}
# the wires that are not the chain: what a block takes from a block further back, shown when it is selected
SIDE_INPUTS = {
    "debruitage": [("empilement", "half-stacks")],
    "couleur": [("astrometrie", "Gaia spectra")],
}
TRUNK = ["inventory", "calibrate", "register", "stack"]

# every state a block can be in: up to date, pending, outdated, and what a run adds
STATE_WORD = {"done": "Up to date", "pending": "Not computed yet", "stale": "Outdated", "running": "Running",
              "failed": "Failed", "interrupted": "Interrupted", "skipped": "Skipped"}
# worst first: a block is as bad as its worst stage in any of its workflows
STATE_ORDER = ["running", "failed", "interrupted", "stale", "skipped", "pending", "done"]

# the review points ("Pause here for review"): a dot like a debugger's, red while the run will stop there, yellow with
# the pause bars while the run is stopped there for the user, green once the user has checked these results
BREAK_COL, PAUSE_COL, REVIEWED_COL = "#e5484d", "#f5c542", "#5fbf7f"
# the most urgent first: what a block shows when its workflows are not all in the same case
REVIEW_ORDER = ["paused", "pending", "reviewed"]

# the block table: key, phase, title, kind (fixed | tool | auto | slot), engine stages, what it does
STEPS = [
    dict(key="calibration", phase=1, title="Calibration", kind="fixed", stages=["inventory", "calibrate"],
         what="Dark and flat masters, then each light calibrated: thermal signal, vignetting and defective pixels "
              "removed."),
    dict(key="alignement", phase=2, title="Registration", kind="fixed", stages=["register"],
         what="Measures the stars of each frame (size, transparency) and aligns all frames on the sharpest one."),
    dict(key="selection", phase=2, title="Selection", kind="fixed", stages=[], run=["register"],
         what="Which frames go into the stack: by star size and transparency thresholds, or one by one."),
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
    dict(key="deconv", phase=4, title="Deconvolution", kind="tool", tool="deconv", stages=["deconv"],
         what="Sharpens the image with the point spread function measured on the stars."),
    dict(key="debruitage", phase=4, title="Denoise", kind="tool", tool="debruitage", stages=["denoise"],
         what="A network learns on the two half-stacks to predict one from the other: it removes the noise without "
              "inventing anything."),
    dict(key="etoiles", phase=4, title="Split stars & background", kind="tool", tool="etoiles", stages=["star_removal"],
         what="A network trained on our stacks takes the stars off the image: two images, the starless image and the "
              "stars alone, sent to Post as they are or combined by a Combine stars & background layers block."),
    dict(key="fond", phase=4, title="Gradient", kind="slot", stages=["gradient"],
         what="Models the sky gradient (light pollution, the Moon, vignetting) on the stack and removes it from what "
              "every later block reads. Polynomial: a smooth surface fitted on the sky between the objects. "
              "Automatic: a more flexible surface that follows complex gradients."),
    dict(key="post", phase=5, title="Output", kind="fixed", stages=["render_simple"],
         what="Prepares the render: what this workflow gives in Post, its image and its sliders."),
    dict(key="composition", phase=6, title="Combine stars & background layers", kind="slot", stages=[],
         what="Combines the split block's two images: the starless image and the stars, each stretched, the stars "
              "put back on top (Screen, Linear dodge or Lighten). Its settings are those of its layer in Post."),
]
BY_KEY = {b["key"]: b for b in STEPS}
# what the optional blocks say they do (the toolbar cards' tooltips)
TOOLS = {
    "debruitage": dict(title="Denoise", phase=4,
                       what="Removes the noise without inventing anything: a network learns on the two half-stacks "
                            "to predict one from the other."),
    "etoiles": dict(title="Split stars & background", phase=4,
                    what="Takes the stars off the image with a trained network: the starless image and the stars "
                         "alone, two images for Post (combined by a Combine stars & background layers block, or as they are)."),
    "deconv": dict(title="Deconvolution", phase=4,
                   what="Sharpens the image with the point spread function measured on the stars."),
}


def block_title(p, key: str) -> str:
    """The name of a block, translated. The output of a workflow carries the workflow's name."""
    if key == "post" and p is not None:
        return p.name
    return _(TITLES.get(key) or BY_KEY[key]["title"])


def review_title(key: str) -> str:
    """The name of a block in the texts about a pause (the output of a workflow is just "Output" there)."""
    return _("Output") if key == "post" else block_title(None, key)


def chain(p) -> list[str]:
    """The block keys of a workflow, in order (the frames excluded, its Output last); a workflow pulled to the post
    from a block stops there, then has its output."""
    return list(p.model.block_keys(p))


# ------------------------------------------------------------------ block-level answers over a model
def stages_of(s, p, key: str) -> list[str]:
    """The engine stages of this block in the workflow's chain."""
    allst = s.all_stages(p)
    return [x for x in BY_KEY[key]["stages"] if x in allst]


def run_stages(s, p, key: str) -> list[str]:
    """What Run to here runs for this block (Selection: Registration)."""
    return BY_KEY[key].get("run") or stages_of(s, p, key)


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


def method_summary(s, p, key: str) -> str:
    from .model import method_of, methods_of
    m = method_of(p, key)
    return _(next(x["short"] for x in methods_of(key) if x["key"] == m))


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
    _bk: object = None

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
        self.root = Node("sources", list(project.paths))
        self.final = Node("final", list(project.paths))
        self.leaves: dict[int, Node] = {}
        self.nodes: list[Node] = [self.root]
        self.by_path: dict[int, dict[str, Node]] = {}
        for p in project.paths:
            cur = self.root
            self.by_path[p.pid] = {"sources": self.root, "final": self.final}
            for k in chain(p):
                bk = project.block_key(p, k)
                nxt = next((c for c in cur.children if c._bk == bk), None)
                if nxt is None:
                    nxt = Node(k, [], cur)
                    nxt._bk = bk
                    cur.children.append(nxt)
                    self.nodes.append(nxt)
                nxt.paths.append(p)
                self.by_path[p.pid][k] = nxt
                cur = nxt
            self.leaves[p.pid] = cur
        self.compositions: list[Node] = []
        self.direct: list[tuple[Node, str]] = []
        self.to_post: set[int] = set()
        #: the layers of Post, top first: (name, workflow id or None for an image sent from Inspect)
        self.post_layers: list[tuple[str, int | None]] = []
        layers = project.post_layers() if hasattr(project, "post_layers") else []
        have = {lay.get("pid") for lay in layers if lay.get("kind") == "workflow"}
        from .model import POST_PART_NAMES
        for p in project.paths:                       # the layer Post gives a workflow it does not have yet
            if p.pid not in have:
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
        self._fold_outputs(project)
        self.nodes.append(self.final)

    def _fold_outputs(self, project):
        """Issue #17: a workflow with the Stars block has no Output block of its own in the graph (it sat between the
        Stars block and what follows it, its arrows around it): the Stars block holds it (the workflow's handle: its
        name, its menu; ``node_of(p, "post")`` answers it). Kept when the Stars block is shared by other workflows, or
        when the Output image itself is a layer of Post."""
        for p in project.paths:
            st, po = self.by_path[p.pid].get("etoiles"), self.by_path[p.pid].get("post")
            if st is None or po is None or po.parent is not st or po.children or len(st.paths) != 1 or \
                    p.pid in self.to_post:
                continue
            st.children.remove(po)
            self.nodes.remove(po)
            self.by_path[p.pid]["post"] = st
            if self.leaves.get(p.pid) is po:
                self.leaves[p.pid] = st
            st.handle = True

    def node_of(self, p, key: str, lid: int | None = None) -> Node | None:
        """The node of a block of a workflow; a Stars and background block by its Post layer (``lid``; else the
        workflow's first one)."""
        if key == "sources":
            return self.root
        if key == "final":
            return self.final
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
        if node.key == "final":
            return "done" if self.s.paths and all(p.rendered for p in self.s.paths) else "pending"
        if node.key == "composition":                 # computed in Post from the Stars block's images
            return block_state(self.s, node.paths[0], "etoiles")
        states = {block_state(self.s, p, node.key) for p in node.paths}
        return next((w for w in STATE_ORDER if w in states), "pending")

    def breakpoint(self, node: Node) -> tuple[str, int, int] | None:
        """The review point of a block: (state, how many of its workflows have it, how many go through it), the state
        being the most urgent of theirs (paused, pending, reviewed); None when no workflow pauses after it."""
        review = getattr(self.s, "review_state", None)
        if review is None or node.key in ("sources", "final", "composition") or not node.paths:
            return None
        have = [st for st in (review(p, node.key) for p in node.paths) if st]
        if not have:
            return None
        return next(w for w in REVIEW_ORDER if w in have), len(have), len(node.paths)

    def seconds(self, node: Node) -> float:
        if node.key in ("sources", "final", "composition") or not node.paths:
            return 0.0
        return block_seconds(self.s, node.paths[0], node.key)

    def summary(self, node: Node) -> str:
        """Under a block's title; the Stars block that holds its workflow's Output says the workflow's name first."""
        text = self._summary(node)
        return f"{node.paths[0].name} · {text}" if node.handle and node.paths else text

    def _summary(self, node: Node) -> str:
        s = self.s
        if node.key == "sources":
            if not s.sources_loaded:
                return _("to add")
            return plural(len(s.sessions), "{n} session · {lights} lights", "{n} sessions · {lights} lights",
                          lights=fmt_int(s.n_lights()))
        if node.key == "final":
            return plural(len(self.post_layers), "{n} layer", "{n} layers")
        p = node.paths[0]
        k = node.key
        if k == "post":
            if p.end:
                return _("from {block}").format(block=block_title(p, p.end))
            # with Stars, the image goes to the Stars block (and to Post only when a layer shows it as it is)
            # with Stars its images go to Post from the Stars block: this block is only the workflow (issue #8)
            return _("to Post") if p.pid in self.to_post or self.node_of(p, "etoiles") is None else _("workflow")
        if k == "composition":
            lay = self.s.post_layer(node.lid) if node.lid is not None and hasattr(self.s, "post_layer") else None
            return _("stars in {mode}").format(mode=_((lay or {}).get("blend") or "Screen"))
        if k == "calibration":
            inv = s.info()
            return (_("{darks} darks, {flats} flats, no bias") if not inv.get("n_bias") else
                    _("{darks} darks, {flats} flats, {bias} bias")).format(
                darks=inv.get("n_darks", 0), flats=inv.get("n_flats", 0), bias=inv.get("n_bias", 0))
        if k == "alignement":
            return _("on the sharpest frame")
        if k == "selection":
            rule = s.selection_rule(p)
            if rule == "min_weight":
                return _("every frame, weight ≥ 5 % of the best")
            if rule == "sota":
                return _("k × σ filters · weight ≥ 5 % of the best")
            t, f = s.thresholds_now()
            unit = s.thresholds_unit()
            return (_("transparency ≥ {t} % · stars ≤ {f} px").format(t=f"{t:.0f}", f=number(f)) if unit == "px" else
                    _("transparency ≥ {t} % · stars ≤ {f} × median").format(t=f"{t:.0f}", f=number(f)))
        if k == "empilement":
            if s.param_value(p, "stack", "method") == "classic":
                txt = _("classic stack · {how}").format(how=texts.choice_label(
                    str(s.param_value(p, "stack", "classic_debayer") or "rcd"), language()))
            else:
                sc = number(s.param_value(p, "stack", "scale") or 1.0, 2).rstrip("0").rstrip(",.")
                txt = _("drizzle ×{scale} · rejection at {kappa} σ").format(
                    scale=sc, kappa=number(s.param_value(p, "stack", "kappa") or 5, 0))
            if any(q.opts["debruitage"] for q in node.paths):
                txt += " · " + _("+ half-stacks")
            return txt
        if k == "debruitage":
            if p.opts.get("dn") == "affine":
                return _("pretrained + adapted · {length}").format(length=texts.choice_label(
                    str(s.param_value(p, "denoise", "adaptation") or "short"), language()))
            return _("pretrained network")
        if k in ("fond", "etoiles", "deconv"):
            return method_summary(s, p, k)
        if k == "astrometrie":
            return _("plate solved · Gaia DR3")
        if k == "couleur":
            return _("Gaia stars · white balance")
        return ""

    def side_inputs(self, node: Node):
        """[(source node, label)] of the wires that come from further back, for the selected block."""
        if not node.paths or node.key in ("sources", "final"):
            return []
        p = node.paths[0]
        pairs = list(SIDE_INPUTS.get(node.key, []))
        out = []
        for k, lab in pairs:
            src = self.node_of(p, k)
            if src is not None and src is not node.parent:
                out.append((src, _(lab)))
        return out


__all__ = ["COLUMNS", "OPTIONAL", "PHASES", "TOOL_COLUMN", "TITLES", "TOOLBOX", "STEPS", "BY_KEY", "TOOLS",
           "STATE_WORD", "STATE_ORDER", "ENGINE_BLOCK", "Graph", "Node", "block_title", "chain", "stages_of",
           "run_stages", "block_state", "block_seconds", "hm", "BREAK_COL", "PAUSE_COL", "REVIEWED_COL", "review_title"]
