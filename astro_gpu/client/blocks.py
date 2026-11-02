"""The blocks of the graph and how the workflows share them.

A workflow is a chain of blocks, named as in the standard tools (Calibration, Registration, Selection, Stacking,
Astrometry, Color, Denoise, Stars, Gradient, Output). The engine's internal steps (the half-stacks, the starless image)
are not blocks: they are shown inside the block that makes them and as a wire to the block that uses them.

The order is the engine's real one (DESIGN §2.4): Gradient comes after Stars, whose starless image it works on. Two
workflows share a block as long as they compute it, and everything before it, the same way (the keys of the engine,
``api.blocks``). The graph is therefore a tree: one root (the frames), a branch where two workflows start to differ, one
leaf (the output) per workflow, all leaves going to Post.

The views keep the mockup's internal block keys (calibration, alignement, selection, empilement, astrometrie, couleur,
deconv, debruitage, etoiles, fond, post): ``ENGINE_BLOCK`` maps the engine's names to them. The stage names are the
engine's (``background_poly``, ``render_simple``...).

This module also holds the block table (``STEPS``, ``BY_KEY``) that the Inspect strip and the panel share, and the
block-level answers (stages, state, duration, summary) over a ``ProjectModel``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import texts
from .i18n import _, hm, language, plural

# the columns of the graph, left to right; optional columns only appear when a workflow uses them
COLUMNS = ["sources", "calibration", "alignement", "selection", "empilement", "astrometrie", "couleur", "deconv",
           "debruitage", "etoiles", "fond", "post", "final"]
OPTIONAL = {"debruitage", "deconv", "etoiles"}
PHASES = [("Pre-processing", "calibration", "empilement"), ("Linear processing", "astrometrie", "fond")]
TOOL_COLUMN = {"debruitage": "debruitage", "etoiles": "etoiles", "appris": "etoiles", "deconv": "deconv",
               "halpha": "fond", "simple": "fond"}
TITLES = {"sources": "Frames", "final": "Post", "calibration": "Calibration", "alignement": "Registration",
          "selection": "Selection", "empilement": "Stacking", "astrometrie": "Astrometry", "fond": "Gradient",
          "couleur": "Color", "debruitage": "Denoise", "deconv": "Deconvolution", "etoiles": "Stars",
          "appris": "Stars", "post": "Output", "demi_piles": "Half-stacks", "sans_etoiles": "Starless image"}
# the optional blocks, in the toolbar's order
TOOLBOX = ["debruitage", "deconv", "etoiles"]
HIDDEN_BLOCKS = {"appris"}                      # the neural removal is a method of the Stars block
# the engine's block names (core/workflow.BLOCKS) and the views' keys
ENGINE_BLOCK = {"calibration": "calibration", "registration": "alignement", "selection": "selection",
                "stacking": "empilement", "astrometry": "astrometrie", "color": "couleur", "denoise": "debruitage",
                "stars": "etoiles", "gradient": "fond", "output": "post"}
KEY_ENGINE_BLOCK = {v: k for k, v in ENGINE_BLOCK.items()}
# the wires that are not the chain: what a block takes from a block further back, shown when it is selected
SIDE_INPUTS = {
    "debruitage": [("empilement", "half-stacks")],
    "etoiles": [("astrometrie", "Gaia stars")],
    "couleur": [("astrometrie", "Gaia spectra")],
}
SIDE_INPUTS_HALPHA = [("astrometrie", "H-alpha survey")]
TRUNK = ["inventory", "calibrate", "register", "stack"]

# every state a block can be in: the three of the spec, and what a run adds
STATE_WORD = {"done": "Up to date", "pending": "Pending", "stale": "Outdated", "running": "Running",
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
         stages=["astrometry", "reference", "catalogs", "surveys"],
         what="Finds where the image points on the sky by matching its stars to Gaia, then fetches the catalogue "
              "stars (and the H-alpha survey if the background uses it)."),
    dict(key="couleur", phase=4, title="Color", kind="fixed", stages=["color"],
         what="Measures the sensor's response on the Gaia spectra of the field stars, then the matrix to the true "
              "sky colors."),
    dict(key="deconv", phase=4, title="Deconvolution", kind="tool", tool="deconv", stages=["deconv"],
         what="Sharpens the image with the point spread function measured on the stars."),
    dict(key="debruitage", phase=4, title="Denoise", kind="tool", tool="debruitage", stages=["denoise"],
         what="A network learns on the two half-stacks to predict one from the other: it removes the noise without "
              "inventing anything."),
    dict(key="etoiles", phase=4, title="Stars", kind="tool", tool="etoiles",
         stages=["optics", "stars", "star_net", "starless"],
         what="Each star measured on Gaia with the real PSF of the optics, subtracted, then redrawn at output: this "
              "gives the star sliders in Post."),
    dict(key="appris", phase=4, title="Neural star removal", kind="tool", tool="appris", stages=["star_net"],
         what="In addition to the star measurement: a star-removal network trained on this stack. Gives a second "
              "starless image."),
    dict(key="sans_etoiles", phase=4, title="Starless image", kind="auto", stages=["starless"],
         what="Subtracts the star model (the nebula under the stars is kept), then denoises the starless image. "
              "Comes with Stars."),
    dict(key="fond", phase=4, title="Gradient", kind="slot", stages=["background", "background_poly"],
         what="Polynomial: a smooth surface fitted on the dark areas. H-alpha survey: separates the true sky "
              "background from instrument defects using a professional survey."),
    dict(key="post", phase=5, title="Output", kind="fixed",
         stages=["render_prep", "render_prep_learned", "render_simple"],
         what="Prepares the render: what this workflow gives in Post, its image and its sliders."),
]
BY_KEY = {b["key"]: b for b in STEPS}
# what the optional blocks say they do (the toolbar cards' tooltips)
TOOLS = {
    "debruitage": dict(title="Denoise", phase=4,
                       what="Removes the noise without inventing anything: a network learns on the two half-stacks "
                            "to predict one from the other."),
    "etoiles": dict(title="Stars", phase=4,
                    what="Measures each star on Gaia with the real PSF of the optics, removes it, then redraws it at "
                         "output: gives the star sliders in Post."),
    "appris": dict(title="Neural star removal", phase=4,
                   what="A star-removal network trained on this stack, in addition to the star measurement: a "
                        "second starless image."),
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


def chain(steps, p) -> list[str]:
    """The block keys of a workflow, in order (frames and post excluded); a workflow pulled to the post from a block
    stops there, then has its output. ``steps`` is unused (the mockup's Steps object): the model knows the chain."""
    return [k for k in p_model(p).block_keys(p) if k not in HIDDEN_BLOCKS]


def p_model(p):
    """The model a workflow view belongs to."""
    return p.model


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


def fnum(x, d=1) -> str:
    from .i18n import number
    return number(x, d)


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
    _bk: object = None

    @property
    def shared(self) -> bool:
        return len(self.paths) > 1


class Graph:
    """The tree of blocks for the current state of the project."""

    def __init__(self, project, steps=None):
        self.s = project
        self.steps = steps
        self.root = Node("sources", list(project.paths))
        self.final = Node("final", list(project.paths))
        self.leaves: dict[int, Node] = {}
        self.nodes: list[Node] = [self.root]
        self.by_path: dict[int, dict[str, Node]] = {}
        for p in project.paths:
            cur = self.root
            self.by_path[p.pid] = {"sources": self.root, "final": self.final}
            for k in chain(steps, p):
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
        self.nodes.append(self.final)

    def node_of(self, p, key: str) -> Node | None:
        if key == "sources":
            return self.root
        if key == "final":
            return self.final
        if p is None:
            return None
        return self.by_path.get(p.pid, {}).get(key)

    # ------------------------------------------------------------ state of a block
    def state(self, node: Node) -> str:
        """done | pending | stale | running | failed | interrupted | skipped, for the workflows going through it."""
        if node.key == "sources":
            return "done" if self.s.sources_loaded else "pending"
        if node.key == "final":
            return "done" if self.s.paths and all(p.rendered for p in self.s.paths) else "pending"
        states = {block_state(self.s, p, node.key) for p in node.paths}
        return next((w for w in STATE_ORDER if w in states), "pending")

    def breakpoint(self, node: Node) -> tuple[str, int, int] | None:
        """The review point of a block: (state, how many of its workflows have it, how many go through it), the state
        being the most urgent of theirs (paused, pending, reviewed); None when no workflow pauses after it."""
        review = getattr(self.s, "review_state", None)
        if review is None or node.key in ("sources", "final") or not node.paths:
            return None
        have = [st for st in (review(p, node.key) for p in node.paths) if st]
        if not have:
            return None
        return next(w for w in REVIEW_ORDER if w in have), len(have), len(node.paths)

    def seconds(self, node: Node) -> float:
        if node.key in ("sources", "final") or not node.paths:
            return 0.0
        return block_seconds(self.s, node.paths[0], node.key)

    def summary(self, node: Node) -> str:
        s = self.s
        if node.key == "sources":
            if not s.sources_loaded:
                return _("to add")
            return plural(len(s.sessions), "{n} session · {lights} lights", "{n} sessions · {lights} lights",
                          lights=fmt_int(s.n_lights()))
        if node.key == "final":
            return plural(len(s.paths), "{n} layer", "{n} layers")
        p = node.paths[0]
        k = node.key
        if k == "post":
            return _("from {block}").format(block=block_title(p, p.end)) if p.end else _("to Post")
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
            return (_("transparency ≥ {t} % · stars ≤ {f} px").format(t=f"{t:.0f}", f=fnum(f)) if unit == "px" else
                    _("transparency ≥ {t} % · stars ≤ {f} × median").format(t=f"{t:.0f}", f=fnum(f)))
        if k == "empilement":
            if s.param_value(p, "stack", "method") == "classic":
                txt = _("classic stack · {how}").format(how=texts.choice_label(
                    str(s.param_value(p, "stack", "classic_debayer") or "rcd"), language()))
            else:
                sc = fnum(s.param_value(p, "stack", "scale") or 1.0, 2).rstrip("0").rstrip(",.")
                txt = _("drizzle ×{scale} · rejection at {kappa} σ").format(
                    scale=sc, kappa=fnum(s.param_value(p, "stack", "kappa") or 5, 0))
            if any(q.opts["debruitage"] for q in node.paths):
                txt += " · " + _("+ half-stacks")
            return txt
        if k == "debruitage":
            from .model import DN_SUMMARY
            dn = p.opts.get("dn", "appris")
            if dn in DN_SUMMARY:
                return _(DN_SUMMARY[dn])
            return _("Noise2Noise · {n} it.").format(n=fmt_int(s.param_value(p, "denoise", "iterations") or 0))
        if k in ("fond", "etoiles", "deconv"):
            return method_summary(s, p, k)
        if k == "astrometrie":
            return _("plate solved · Gaia DR3")
        if k == "couleur":
            if s.param_value(p, "color", "correction") == "per_colour":
                return _("Gaia spectra · one factor per colour")
            return _("Gaia spectra · colour matrix")
        return ""

    def side_inputs(self, node: Node):
        """[(source node, label)] of the wires that come from further back, for the selected block."""
        if not node.paths or node.key in ("sources", "final"):
            return []
        p = node.paths[0]
        pairs = list(SIDE_INPUTS.get(node.key, []))
        if node.key == "fond" and p.opts["fond"] == "halpha":
            pairs = SIDE_INPUTS_HALPHA
        out = []
        for k, lab in pairs:
            src = self.node_of(p, k)
            if src is not None and src is not node.parent:
                out.append((src, _(lab)))
        return out


__all__ = ["COLUMNS", "OPTIONAL", "PHASES", "TOOL_COLUMN", "TITLES", "TOOLBOX", "STEPS", "BY_KEY", "TOOLS",
           "STATE_WORD", "STATE_ORDER", "ENGINE_BLOCK", "Graph", "Node", "block_title", "chain", "stages_of",
           "run_stages", "block_state", "block_seconds", "hm", "BREAK_COL", "PAUSE_COL", "REVIEWED_COL", "review_title"]
