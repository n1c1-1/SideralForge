"""The blocks of the graph and how the workflows share them.

A workflow is a chain of blocks, named as in the standard tools (Calibration, Registration, Stacking, Astrometry,
Background, Color, Denoise, Deconvolution, Stars). The engine's internal steps (the half-stacks, the starless image)
are not blocks: they are shown inside the block that makes them and as a wire to the block that uses them.

Two workflows share a block as long as they compute it, and everything before it, the same way. The graph is
therefore a tree: one root (the frames), a branch where two workflows start to differ, one leaf (the output) per
workflow, all leaves going to Post.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from i18n import _
from model import PathDef
from steps import BY_KEY, STEPS

# the columns of the graph, left to right; optional columns only appear when a workflow uses them
COLUMNS = ["sources", "calibration", "alignement", "selection", "empilement", "astrometrie", "fond", "couleur",
           "deconv", "debruitage", "etoiles", "post", "final"]
OPTIONAL = {"debruitage", "deconv", "etoiles"}
PHASES = [("Pre-processing", "calibration", "empilement"), ("Linear processing", "astrometrie", "etoiles")]
TOOL_COLUMN = {"debruitage": "debruitage", "etoiles": "etoiles", "appris": "etoiles", "deconv": "deconv",
               "halpha": "fond", "simple": "fond"}
TITLES = {"sources": "Frames", "final": "Post", "calibration": "Calibration", "alignement": "Registration",
          "selection": "Selection", "empilement": "Stacking", "astrometrie": "Astrometry", "fond": "Gradient",
          "couleur": "Color", "debruitage": "Denoise", "deconv": "Deconvolution", "etoiles": "Stars",
          "appris": "Stars", "post": "Output"}
# the wires that are not the chain: what a block takes from a block further back, shown when it is selected
SIDE_INPUTS = {
    "debruitage": [("empilement", "half-stacks")],
    "etoiles": [("astrometrie", "Gaia stars")],
    "couleur": [("astrometrie", "Gaia spectra")],
}
SIDE_INPUTS_HALPHA = [("astrometrie", "H-alpha survey")]


def block_title(p: PathDef | None, key: str) -> str:
    """The name of a block, translated. The output of a workflow carries the workflow's name."""
    if key == "post" and p is not None:
        return p.name
    return _(TITLES.get(key) or BY_KEY[key]["title"])


def chain(steps, p: PathDef) -> list[str]:
    """The block keys of a workflow, in order (frames and post excluded); a workflow pulled to the post from a block
    stops there, then has its output."""
    from project import HIDDEN_BLOCKS
    keys = [b["key"] for b in STEPS if b["kind"] != "auto" and b["key"] not in HIDDEN_BLOCKS and steps.included(p, b)]
    end = getattr(p, "end", None)
    if end is not None and end in keys:
        keys = keys[: keys.index(end) + 1] + ["post"]
    return keys


@dataclass
class Node:
    key: str                                   # step key, or "sources" / "final"
    paths: list = field(default_factory=list)  # the workflows going through this block
    parent: "Node | None" = None
    children: list = field(default_factory=list)
    col: int = 0
    y: float = 0.0

    @property
    def shared(self) -> bool:
        return len(self.paths) > 1


class Graph:
    """The tree of blocks for the current state of the project."""

    def __init__(self, project, steps):
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
                bk = self.block_key(p, k)
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

    def block_key(self, p: PathDef, k: str):
        if k == "post":
            return ("post", p.pid)
        if k == "selection":
            return ("selection", p.params.get("stack", {}).get("selection"))
        stages = [x for x in self.steps.stages_of(p, BY_KEY[k]) if x != "surveys"]   # the survey goes with the fond
        return (k, p.opts["fond"] if k == "fond" else None, tuple(self.s.stage_key(p, x) for x in stages))

    def node_of(self, p: PathDef | None, key: str) -> Node | None:
        if key == "sources":
            return self.root
        if key == "final":
            return self.final
        if p is None:
            return None
        return self.by_path.get(p.pid, {}).get(key)

    # ------------------------------------------------------------ state of a block
    def state(self, node: Node) -> str:
        """done | pending | stale, for the workflows going through it."""
        if node.key == "sources":
            return "done" if self.s.sources_loaded else "pending"
        if node.key == "final":
            return "done" if self.s.paths and all(p.rendered for p in self.s.paths) else "pending"
        states = {self.steps.state(p, BY_KEY[node.key]) for p in node.paths}
        if "stale" in states:
            return "stale"
        if "pending" in states:
            return "pending"
        return "done"

    def seconds(self, node: Node) -> float:
        if node.key in ("sources", "final") or not node.paths:
            return 0.0
        p = node.paths[0]
        sec = self.steps.seconds(p, BY_KEY[node.key])
        if node.key == "etoiles":
            sec += self.s.seconds("starless")
        return sec

    def summary(self, node: Node) -> str:
        s = self.s
        if node.key == "sources":
            if not s.sources_loaded:
                return _("to add")
            n = len(s.sessions)
            return _("{n} session(s) · {lights} lights").format(n=n, lights=s.n_lights())
        if node.key == "final":
            return _("{n} layer(s)").format(n=len(s.paths))
        p = node.paths[0]
        k = node.key
        from project import DN_SUMMARY, METHODS, method_of
        if k == "empilement" and any(q.opts["debruitage"] for q in node.paths):
            return self.steps.summary(p, BY_KEY[k]) + " · " + _("+ half-stacks")
        if k == "post":
            end = getattr(p, "end", None)
            return _("from {block}").format(block=block_title(p, end)) if end else _("to Post")
        if k == "debruitage":
            dn = p.opts.get("dn", "appris")
            return _(DN_SUMMARY[dn]) if dn in DN_SUMMARY else _("Noise2Noise · {n} it.").format(
                n=f"{self.steps.param(p, 'denoise', 'iterations'):,}".replace(",", " "))
        if k in ("fond", "etoiles", "deconv"):
            m = method_of(p, k)
            return _(next(x["short"] for x in METHODS[k] if x["key"] == m))
        if k == "astrometrie":
            return _("plate solved · Gaia DR3")
        if k == "couleur":
            return _("SPCC · Gaia spectra")
        return self.steps.summary(p, BY_KEY[k])

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
