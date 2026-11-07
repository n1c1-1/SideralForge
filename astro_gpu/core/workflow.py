"""Workflows of a v2 project: options → the chain of stages, the blocks, the option rules.

Entry points: ``check_options``, ``set_option`` (the rules, with their notes), ``chain`` / ``chains()`` (the stage
graph, ``Chains``), ``method_of``, ``implied``, ``TEMPLATES``.

A workflow is a set of options (which optional blocks it has and the method of each) plus the settings the user
stored. The options give its chain: the stages it runs, in the global order. Two workflows share a stage result
when the stage key is the same (core/store.py); this module only says which stages a workflow has and which of them
each stage reads.

    options   {"denoise": None | "pretrained" | "adapted", "stars": None | "neural", "gradient": "polynomial" | "rbf" |
               "automatic", "deconvolution": None, "cut": [the blocks the user cut, CUTTABLE]}
    end       None, or "stacking" for the Raw stack exit (inventory → stack, then render_simple)

Issue #43, the blocks redesign:

- Sessions calibrated each on its own: when the project calibrates its sessions separately (``sessions``: their ids,
  two or more), the chain holds an instance of inventory and calibrate per session, named ``<stage>@<session id>``,
  then the ``sessions`` stage, which joins their frames into one calibrated set: what register and stack read as
  "inventory" and "calibrate" (``PROVIDES``). An instance's key is that of a single-session project with its frames.
- Cut blocks (``CUTTABLE``): Calibration and Registration stay in the chain and run in a pass-through mode (no master
  frames; no alignment: their settings imply it); Astrometry, Gradient and Color leave the chain. A block that needs a
  cut one is greyed (``greyed``: its reason): it leaves the chain too, and the image passes on.
- Image source: a workflow may start from an image file (``image``): the ``image`` stage stands for the stack
  (``PROVIDES``); the blocks before it are absent, and those that need a real stack (the half-stacks of Denoise, then
  Stars) are greyed.
- Mono is planned, not built: a session has a filter; sessions of one filter will stack together.

The Gradient block comes right after stacking: its method models the sky gradient, which every later stage reads
removed (stages/gradient.py). The Stars block takes the stars off the Output block's image (stages/star_removal.py):
it comes after it. Rule: Stars needs Denoise (the network was trained on denoised images). Other option values are not
in the engine and raise. A stored gradient "halpha" (the background anchored on the H-alpha survey, which worked on the
starless image) is read as the method that replaces it, a stored denoise "n2n" (a network trained from scratch on each
image) as the pretrained network, a stored stars "psf" or "psf_neural" (the star model subtracted, then redrawn) as the
network that replaced them.

The stage list comes from ``stages/v2.py`` (``V2_STAGES``, loaded lazily); tests build toy ``Chains``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

# ---------------------------------------------------------------------------------------------- options

#: the allowed values of each option
OPTION_VALUES: dict[str, tuple[Any, ...]] = {
    "denoise": (None, "pretrained", "adapted"),
    "stars": (None, "neural"),
    "gradient": ("polynomial", "rbf", "automatic"),
    "deconvolution": (None,),
}
#: a missing option takes this value (the Classic stack). "cut", the blocks the user cut (a list, check_cut), is there
#: only when one is: the options of a workflow without a cut stay as they were before issue #43
DEFAULT_OPTIONS: dict[str, Any] = {"denoise": None, "stars": None, "gradient": "polynomial", "deconvolution": None}
#: the blocks the user can cut (issue #43). Stacking and Output cannot be cut; Denoise and Stars are added or removed
CUTTABLE = ("calibration", "registration", "astrometry", "gradient", "color")
#: early exits: the block a workflow's output comes from when it does not go to the end
EXITS = ("stacking",)

#: the workflow templates; a template is only used at creation
TEMPLATES: dict[str, dict[str, Any]] = {
    "Classic stack": {"denoise": None, "stars": None, "gradient": "polynomial", "deconvolution": None},
    "Classic + denoise": {"denoise": "pretrained", "stars": None, "gradient": "polynomial", "deconvolution": None},
    "Full": {"denoise": "pretrained", "stars": "neural", "gradient": "polynomial", "deconvolution": None},
}

# the notes of the option rules (shown in the client, translated by its catalogues)
NOTE_DENOISE_ADDED = "Denoise added: needed by Stars/background"
REFUSE_DENOISE = "needed by Stars/background"
REFUSE_GRADIENT = "the sky gradient is always removed: choose another method instead"
#: stored option values of earlier versions, read as the current ones
OPTION_ALIASES: dict[str, dict[Any, Any]] = {"gradient": {"halpha": "polynomial"}, "denoise": {"n2n": "pretrained"},
                                              "stars": {"psf": "neural", "psf_neural": "neural"}}
REFUSE_AFTER_EXIT = "No block can be added after an early output"


class OptionRefused(ValueError):
    """An option change the rules do not allow (removing Denoise under Stars, removing the Gradient...). The message
    says why, for a toast."""


def check_options(options: dict[str, Any]) -> dict[str, Any]:
    """The options, complete and in the fixed key order. Raises ValueError on an unknown option, a value that is
    not in the engine, or options that break a rule (Stars without Denoise, H-alpha without Stars)."""
    unknown = set(options) - set(OPTION_VALUES) - {"cut"}
    if unknown:
        raise ValueError(f"unknown workflow option(s): {sorted(unknown)}")
    o = {k: options.get(k, DEFAULT_OPTIONS[k]) for k in OPTION_VALUES}
    cut = check_cut(options.get("cut") or [])
    if cut:
        o["cut"] = cut
    for k, alias in OPTION_ALIASES.items():
        o[k] = alias.get(o[k], o[k])
    for k, allowed in OPTION_VALUES.items():
        if o[k] not in allowed:
            raise ValueError(f"workflow option {k}={o[k]!r} is not in the engine (allowed: {list(allowed)})")
    if o["stars"] and not o["denoise"]:
        raise ValueError(f"workflow option stars={o['stars']!r} needs denoise: the star removal works on the "
                         "denoised stack")
    return o


def check_cut(cut: Any) -> list[str]:
    """The cut blocks as they are stored: distinct names of CUTTABLE, in the graph's order."""
    if not isinstance(cut, (list, tuple)) or not all(isinstance(b, str) for b in cut):
        raise ValueError(f"the cut blocks must be a list of block names, not {cut!r}")
    bad = [b for b in cut if b not in CUTTABLE]
    if bad:
        raise ValueError(f"these blocks cannot be cut: {bad} (allowed: {list(CUTTABLE)})")
    return [b for b in CUTTABLE if b in cut]


def set_cut(options: dict[str, Any], block: str, on: bool) -> dict[str, Any]:
    """The options with ``block`` cut (``on``) or restored."""
    o = check_options(options)
    cut = set(cuts(o)) | {block} if on else set(cuts(o)) - {block}
    o.pop("cut", None)
    return check_options({**o, "cut": sorted(cut)})


def cuts(options: dict[str, Any]) -> list[str]:
    """The blocks a workflow's options cut."""
    return list(options.get("cut") or [])


#: why a block cannot run (the user reads it on the greyed block; English, translated by the client)
NEEDS_ASTROMETRY = "needs Astrometry"
NEEDS_HALF_STACKS = "needs half-stacks"
NEEDS_DENOISE = "needs Denoise"
MONO_NO_COLOUR = "monochrome: colours from filters"


def greyed(options: dict[str, Any], image: bool = False, mono: bool = False, stretched: bool = False
           ) -> dict[str, str]:
    """{block: why it cannot run} for the blocks of a workflow that need a block it cut, or a stack when it starts
    from an image, or colours its monochrome frames do not have (issue #43: greyed, the image passes on)."""
    o = check_options(options)
    cut = set(cuts(o))
    out: dict[str, str] = {}
    if mono and "color" not in cut:
        out["color"] = MONO_NO_COLOUR
    if "astrometry" in cut and "color" not in cut:
        out["color"] = NEEDS_ASTROMETRY
    if image and o["denoise"]:
        out["denoise"] = NEEDS_HALF_STACKS
    if image and o["stars"] and not stretched:
        out["stars"] = NEEDS_DENOISE
    return out


def check_end(end: str | None) -> str | None:
    if end is not None and end not in EXITS:
        raise ValueError(f"workflow end {end!r} is not an exit (allowed: None, {', '.join(map(repr, EXITS))})")
    return end


def set_option(options: dict[str, Any], key: str, value: Any, end: str | None = None) -> tuple[dict, list[str]]:
    """The options with ``key`` set to ``value``, and the notes on what came or went with it. Raises OptionRefused
    when the rules forbid the change, ValueError on an invalid option or value."""
    o = check_options(options)
    if key not in OPTION_VALUES:
        raise ValueError(f"unknown workflow option {key!r}")
    if key == "gradient" and value is None:
        raise OptionRefused(REFUSE_GRADIENT)
    if value not in OPTION_VALUES[key]:
        raise ValueError(f"workflow option {key}={value!r} is not in the engine (allowed: {list(OPTION_VALUES[key])})")
    if check_end(end) is not None:
        raise OptionRefused(REFUSE_AFTER_EXIT)
    if o[key] == value:
        return o, []
    notes: list[str] = []
    if key == "denoise":
        if value is None and o["stars"]:
            raise OptionRefused(REFUSE_DENOISE)
    o[key] = value
    if o["stars"] and not o["denoise"]:
        o["denoise"] = "pretrained"
        notes.append(NOTE_DENOISE_ADDED)
    return check_options(o), notes


def method_of(options: dict[str, Any], block: str) -> str | None:
    """The method of a block: denoise 'pretrained' / 'adapted', stars 'neural', gradient 'polynomial' / 'rbf' /
    'automatic'; None when the workflow does not have the block."""
    o = check_options(options)
    if block in ("gradient", "denoise", "stars", "deconvolution"):
        return o[block]
    raise ValueError(f"block {block!r} has no method")


def implied(options: dict[str, Any], stage: str) -> dict[str, Any]:
    """The parameter values the options impose on a stage (they win over the stored ones): the method of the
    Gradient and Denoise blocks."""
    o = check_options(options)
    if stage == "gradient":
        return {"method": o["gradient"]}
    if stage == "denoise" and o["denoise"]:
        return {"method": o["denoise"]}
    if stage == "calibrate" and "calibration" in cuts(o):
        return {"apply_masters": False}
    if stage == "register" and "registration" in cuts(o):
        return {"align": False}
    return {}


# ---------------------------------------------------------------------------------------------- the chain table

#: the global order of the v2 stages (each stage only reads stages before it)
ORDER = ["fusion", "image", "inventory", "calibrate", "sessions", "register", "stack", "astrometry", "reference", "catalogs",
         "surveys", "gradient", "denoise", "color", "render_simple", "star_removal"]
#: the stages that read the image the blocks before them made (issue #47), and which of those blocks they apply: all
#: of them, or only the sky gradients (Color measures its stars on the stack, its sky removed, and the Color blocks
#: before it applied: a second one measures what the first left; the star removal takes its noise from the
#: half-stacks so)
READERS: dict[str, str] = {"gradient": "all", "denoise": "all", "render_simple": "all", "color": "colours",
                           "star_removal": "skies"}
#: the image blocks each rule of READERS applies
READ_KINDS: dict[str, tuple[str, ...]] = {"all": ("gradient", "denoise", "color"), "skies": ("gradient",),
                                         "colours": ("gradient", "color")}
#: stages that stand for others in a chain (issue #43): the sessions' join for each session's frames and calibration,
#: the image of an image source for the stack
PROVIDES: dict[str, tuple[str, ...]] = {"sessions": ("inventory", "calibrate"), "image": ("stack", "render_simple"),
                                         "fusion": ("stack",)}
#: dependencies a stage has beyond its declared ``deps``
EXTRA: dict[str, tuple[str, ...]] = {}
#: dependencies that may be absent from a chain (the stage then runs without them); the stages that read the stack
#: read it with the sky gradient removed when the chain has the Gradient block (gradient.stack_image)
OPTIONAL: dict[str, tuple[str, ...]] = {
    # an image source has no frame list: the camera, the pointing and the date come from the target, or are unknown
    "astrometry": ("inventory",),
    "catalogs": ("inventory",),
    "gradient": ("surveys",),                       # where the H-alpha survey sees nebulosity: no sky sample there
    "color": ("gradient", "register"),             # without Registration (an image source): a default star size
    "denoise": ("gradient",),
    "render_simple": ("color", "denoise", "gradient"),
    "star_removal": ("gradient",),
}
#: the blocks of the graph and their stages; Selection has no stage of its own (it follows register)
BLOCKS: dict[str, tuple[str, ...]] = {
    "fusion": ("fusion",),
    "image": ("image",),
    "calibration": ("inventory", "calibrate"),
    "registration": ("sessions", "register"),       # the sessions join where they are registered together
    "selection": (),
    "stacking": ("stack",),
    "astrometry": ("astrometry", "reference", "catalogs", "surveys"),
    "gradient": ("gradient",),
    "denoise": ("denoise",),                        # Color applies its matrix after Denoise: the engine's order
    "color": ("color",),
    "stars": ("star_removal",),                     # drawn before the Output, which every workflow ends with
    "output": ("render_simple",),
}
BLOCK_TITLES = {"fusion": "Fusion", "image": "Image", "calibration": "Calibration", "registration": "Registration",
                "selection": "Selection",
                "stacking": "Stacking", "astrometry": "Astrometry", "color": "Color", "denoise": "Denoise",
                "stars": "Stars/background", "gradient": "Gradient", "output": "Output"}
#: what Run to here runs for a block without stages of its own
RUN_TO: dict[str, tuple[str, ...]] = {"selection": ("register",)}
#: stages a block's key leaves out: the survey follows the Gradient method, not the Astrometry block
KEY_LEAVES_OUT: dict[str, tuple[str, ...]] = {"astrometry": ("surveys", "catalogs")}   # catalogs: fetched for a
#                                                 Color block after it; the plate solution is the same (issue #47)


def base(name: str) -> str:
    """The stage of a chain entry: "calibrate" for the instance "calibrate@2", "denoise" for the second Denoise of a
    graph, "denoise~2"."""
    return name.split("@", 1)[0].split("~", 1)[0]


def block_order(name: str) -> tuple[int, int]:
    """Where an engine block stands in the usual order: its block's rank, then a graph's later node after the first
    ("gradient~2" after "gradient")."""
    order = list(BLOCKS)
    b, _sep, n = name.partition("~")
    return (order.index(b) if b in order else len(order), int(n) if n.isdigit() else 0)


def session_of(name: str) -> str | None:
    """The session id (as text) of an instance ("2" for "calibrate@2"), None for a plain stage."""
    return name.split("@", 1)[1] if "@" in name else None


def chain_stages(options: dict[str, Any], end: str | None = None, reference: bool = False,
                 sessions: Sequence[int] = (), image: bool = False, mono: bool = False,
                 fusion: bool = False, stretched: bool = False) -> list[str]:
    """The stages of a workflow, in the global order. ``reference``: the project has a reference image (the stage sits
    in the Astrometry block). ``sessions``: the ids of the sessions calibrated each on its own (two or more; else the
    sessions are one frame list, as before issue #43). ``image``: the workflow starts from an image file."""
    o = check_options(options)
    cut = set(cuts(o))
    if fusion:                      # other workflows' outputs combined (stages/fusion.py): then its Output block,
        # and with stars (option A: the channels' starless images fused) the Split block that gives the star layer back
        return _ordered(["fusion", "render_simple"] + (["star_removal"] if o["stars"] else []))
    if stretched:                   # an image already stretched: its stars taken off, or Post at once
        return _ordered(["image"] + (["star_removal"] if o["stars"] else []))
    if image:
        out = ["image"]
    elif len(sessions) >= 2:
        out = [f"{s}@{sid}" for s in ("inventory", "calibrate") for sid in sessions] + ["sessions", "register", "stack"]
    else:
        out = ["inventory", "calibrate", "register", "stack"]
    if not image and check_end(end) == "stacking":
        return _ordered(out + ["render_simple"])
    grey = greyed(o, image, mono)
    colour = "color" not in cut and "color" not in grey
    if "astrometry" not in cut:
        # the Gaia catalogue is the colour calibration's (without Color: not fetched)
        out += ["astrometry"] + (["reference"] if reference else []) + (["catalogs"] if colour else []) + ["surveys"]
    if "gradient" not in cut:
        out.append("gradient")
    if colour:
        out.append("color")
    if o["denoise"] and "denoise" not in grey:
        out.append("denoise")
    out.append("render_simple")
    if o["stars"] and "stars" not in grey:
        out.append("star_removal")
    return _ordered(out)


def _ordered(names: Sequence[str]) -> list[str]:
    """Chain entries in the global order; the instances of a stage in the order given (their sessions')."""
    rank = {n: i for i, n in enumerate(names)}
    return sorted(names, key=lambda n: (ORDER.index(base(n)), rank[n]))


Rule = Callable[[dict, "str | None", bool], Iterable[str]]


class Chains:
    """The stage graph of v2 workflows: the stages, their order and dependencies (declared ∪ extra ∪ optional),
    the blocks, and the chain of a workflow (``rule``: options, end, reference → stage names)."""

    def __init__(self, stages: Sequence[Any], order: Sequence[str] | None = None,
                 extra: dict[str, tuple[str, ...]] | None = None, optional: dict[str, tuple[str, ...]] | None = None,
                 blocks: dict[str, tuple[str, ...]] | None = None, rule: Rule | None = None):
        self.stages: dict[str, Any] = {s.name: s for s in stages}
        self.order: list[str] = list(order) if order is not None else [s.name for s in stages]
        self.extra = dict(extra or {})
        self.optional = dict(optional or {})
        self.blocks = dict(blocks or {})
        self.rule = rule
        missing = [n for n in self.order if n not in self.stages] + [n for n in self.stages if n not in self.order]
        if missing:
            raise ValueError(f"stages and order disagree on: {sorted(set(missing))}")
        self.deps2: dict[str, tuple[str, ...]] = {}
        for n in self.order:
            ds: list[str] = []
            for d in (*self.stages[n].deps, *self.extra.get(n, ()), *self.optional.get(n, ())):
                if d not in ds:
                    ds.append(d)
            late = [d for d in ds if d not in self.order or self.order.index(d) >= self.order.index(n)]
            if late:
                raise ValueError(f"stage '{n}' reads {late}, which do not come before it")
            self.deps2[n] = tuple(ds)
        self._block_of = {s: b for b, ss in reversed(list(self.blocks.items())) for s in ss}

    # ------------------------------------------------------------------ chains
    def chain(self, options: dict[str, Any], end: str | None = None, *, reference: bool = False,
              sessions: Sequence[int] = (), image: bool = False, mono: bool = False,
              fusion: bool = False, stretched: bool = False) -> list[str]:
        """The stages of a workflow, ordered and checked."""
        if self.rule is None:
            raise TypeError("these Chains have no rule: use ordered() with the stage names")
        if len(sessions) >= 2 or image or mono or fusion:
            return self.ordered(self.rule(options, end, reference, sessions=sessions, image=image, mono=mono,
                                          fusion=fusion, stretched=stretched))
        return self.ordered(self.rule(options, end, reference))

    def sequence(self, names: Iterable[str]) -> list[str]:
        """Stage names as a chain in the order given (a graph's, issue #47), every required dependency present."""
        out = list(dict.fromkeys(names))
        unknown = [n for n in out if base(n) not in self.stages]
        if unknown:
            raise ValueError(f"unknown stage(s): {sorted(unknown)}")
        for n in out:
            self.deps_in_chain(out, n)
        return out

    def ordered(self, names: Iterable[str]) -> list[str]:
        """Stage names as a chain: in the global order (the instances of a stage in the order given), every required
        dependency present (ValueError otherwise)."""
        names = list(dict.fromkeys(names))
        unknown = [n for n in names if base(n) not in self.stages]
        if unknown:
            raise ValueError(f"unknown stage(s): {sorted(unknown)}")
        rank = {n: i for i, n in enumerate(names)}
        out = sorted(names, key=lambda n: (self.order.index(base(n)), rank[n]))
        for n in out:
            self.deps_in_chain(out, n)
        return out

    def resolve(self, chain: Sequence[str], reader: str, dep: str) -> str | None:
        """The chain entry a stage reads as ``dep`` (issue #43): ``dep`` itself; for an instance, the instance of
        ``dep`` of its own session; else the stage that stands for ``dep`` (``PROVIDES``); None when absent."""
        if dep in chain:
            return dep
        sid = session_of(reader)
        if sid is not None and f"{dep}@{sid}" in chain:
            return f"{dep}@{sid}"
        return next((x for x in chain if dep in PROVIDES.get(base(x), ())), None)

    def deps_in_chain(self, chain: Sequence[str], stage: str) -> tuple[str, ...]:
        """What a stage reads in this chain: its dependencies present in it (``resolve``). A missing required
        dependency is a ValueError naming its block. The sessions' join reads every instance of the chain."""
        if stage not in chain:
            raise ValueError(f"stage '{stage}' is not in the chain")
        b = base(stage)
        if b == "sessions":
            return tuple(x for x in chain if session_of(x) is not None)
        out: list[str] = []
        prior: list[str] = []
        reads = b in READERS and b in self.stages and "stack" in self.stages
        if reads:
            # issue #47: the image blocks before it in the graph, in order (an image block placed after it is not
            # read; a chain in the global order has them where they always were)
            ops = READ_KINDS[READERS[b]]
            prior = [x for x in chain[:list(chain).index(stage)] if base(x) in ops]
        for d in self.deps2[b]:
            if reads and d in IMAGE_OPS:                 # where it was declared: the graph's blocks of that stage
                out += [x for x in prior if base(x) == d and x not in out]
                continue
            r = self.resolve(chain, stage, d)
            if r is not None:
                if r not in out:
                    out.append(r)
            elif d not in self.optional.get(b, ()):
                raise ValueError(f"stage '{stage}' needs the {self.block_title(self.block_of(d))} block "
                                 f"(stage '{d}')")
        out += [x for x in prior if x not in out]
        return tuple(out)

    def ancestors(self, chain: Sequence[str], stage: str) -> list[str]:
        """Every stage this one reads, directly or not, in the chain, in the chain's order."""
        seen: set[str] = set()
        todo = list(self.deps_in_chain(chain, stage))
        while todo:
            d = todo.pop()
            if d not in seen:
                seen.add(d)
                todo += self.deps_in_chain(chain, d)
        return [s for s in chain if s in seen]

    def downstream(self, chain: Sequence[str], stage: str) -> list[str]:
        """The stages of the chain that read this one, directly or not, in order (the stage excluded)."""
        return [s for s in chain if s != stage and stage in self.ancestors(chain, s)]

    # ------------------------------------------------------------------ blocks
    def block_of(self, stage: str) -> str:
        """The block a stage belongs to (the stage name itself when no block lists it); an instance's block is the
        block of its session: "calibration@2"; a graph's later node of a block, that node: "denoise~2" (issue #47)."""
        b = self._block_of.get(base(stage), base(stage))
        sid = session_of(stage)
        if "~" in stage.split("@", 1)[0]:
            return f"{b}~{stage.split('@', 1)[0].split('~', 1)[1]}"
        return f"{b}@{sid}" if sid is not None else b

    @staticmethod
    def block_title(block: str) -> str:
        return BLOCK_TITLES.get(base(block), base(block))

    def blocks_in(self, chain: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
        """(block, its stages in the chain) for the blocks of a chain, in graph order; Selection comes with
        register. The instances of a session make the block of that session ("calibration@2"), in session order."""
        inside = {s for s in chain if "~" not in s}
        out = []
        for b, ss in self.blocks.items():
            mine = tuple(s for s in ss if s in inside)
            if mine or (b in RUN_TO and all(s in inside for s in RUN_TO[b])):
                out.append((b, mine))
            out += [(self.block_of(s), (s,)) for s in chain if "~" in s and base(s) in ss]     # a graph's later nodes
            sids = list(dict.fromkeys(session_of(s) for s in chain if session_of(s) is not None and base(s) in ss))
            out += [(f"{b}@{sid}", tuple(s for s in chain if base(s) in ss and session_of(s) == sid)) for sid in sids]
        return out

    def run_to(self, block: str, chain: Sequence[str]) -> list[str]:
        """The stages Run to here targets for a block (their missing ancestors are the runner's business)."""
        b, sid = base(block), session_of(block)
        if b not in self.blocks:
            raise ValueError(f"unknown block {block!r}")
        if "~" in block:
            return [s for s in chain if self.block_of(s) == block]
        ss = RUN_TO.get(b) or self.blocks[b]
        return [s for s in chain if base(s) in ss and session_of(s) == sid and "~" not in s]

    def key_stages(self, block: str, chain: Sequence[str]) -> list[str]:
        """The stages whose keys make a block's key (Output adds the workflow id: never shared)."""
        b, sid = base(block), session_of(block)
        if "~" in block:
            return [s for s in chain if self.block_of(s) == block]
        return [s for s in chain if base(s) in self.blocks.get(b, ()) and session_of(s) == sid and "~" not in s
                and base(s) not in KEY_LEAVES_OUT.get(b, ())]


# ---------------------------------------------------------------------------------------------- free graphs (#47)
#
# A workflow is a graph the user builds from a palette (issue #47): its source (the sessions' frames, an image file, a
# stretched image, or a fusion of other workflows) then a sequence of blocks, ``blocks``: [{"block", "name", "method",
# "off"}]. Each block takes and gives a kind of data (``IO``); a block is placed only where its input is what the block
# before it gives, and where the blocks it needs from further up are (``NEEDS``). The image blocks that change the image
# (``IMAGE_OPS``) can be placed in any order and more than once: every later block reads the image they make, in the
# graph's order (stages/gradient.image_in). An image block's entry in the chain is its node's ``name``: the block's
# stage for the first one, "<stage>~<n>" for the others (stable: a node keeps its name, its settings with it).
#
# The workflows made before keep their options (``blocks`` None); ``from_options`` gives their graph, whose chain and
# keys are the same (Gradient, Denoise, Color: the order the engine always ran them in).

FRAMES, IMAGE, PAIR = "frames", "image", "pair"
#: what each block takes and gives
IO: dict[str, tuple[str, str]] = {
    "calibration": (FRAMES, FRAMES), "registration": (FRAMES, FRAMES), "selection": (FRAMES, FRAMES),
    "stacking": (FRAMES, IMAGE),
    "astrometry": (IMAGE, IMAGE), "gradient": (IMAGE, IMAGE), "color": (IMAGE, IMAGE), "denoise": (IMAGE, IMAGE),
    "output": (IMAGE, IMAGE), "stars": (IMAGE, PAIR)}
#: the palette's blocks after the sources, in their usual order
PALETTE = ("calibration", "registration", "selection", "stacking", "astrometry", "gradient", "denoise", "color",
           "output", "stars")
#: the frame blocks keep this order among themselves (a calibration after the registration means nothing)
FRAME_BLOCKS = ("calibration", "registration", "selection", "stacking")
#: the image blocks that change the image: any order, any number
IMAGE_OPS = ("gradient", "color", "denoise")
#: the blocks with a method, and their methods
METHODS: dict[str, tuple[str, ...]] = {"gradient": OPTION_VALUES["gradient"], "denoise": ("pretrained", "adapted"),
                                       "stars": ("neural",)}
DEFAULT_METHOD = {"gradient": "polynomial", "denoise": "pretrained", "stars": "neural"}
#: the sources of a workflow
SOURCES = ("frames", "image", "stretched", "fusion")
#: what a source gives
SOURCE_GIVES = {"frames": FRAMES, "image": IMAGE, "stretched": IMAGE, "fusion": IMAGE}

#: why a block cannot be placed (the client translates them)
WHY_INPUT = "takes {takes}; {before} gives {gives}"
WHY_STACK_FIRST = "{block} needs Stacking first"
WHY_ONCE = "one {block} per workflow"
WHY_ONCE_OFF = "{block} already here, off: switch on (D)"
WHY_FRAME_ORDER = "{block} comes {where} {other}"
WHY_NEEDS = "needs {need} first"
WHY_HALVES = "needs half-stacks (from frames)"
WHY_SOURCE = "no {block} {source}"
WHY_AFTER_OUTPUT = "only Stars/background after Output"
WHY_OUTPUT_AUTO = "Output is automatic"
WHY_OUTPUT_LAST = "Output ends the chain"
WHY_REMOVE_NEEDED = "needed by {other}"
WHY_OFF_NEEDED = "needed by {other}: switch {other} off first"      # the same, said for a switch-off
WHY_REMOVE_BRIDGE = "the next blocks need its {takes}"
WHY_OFF_KIND = "{block}: {takes} → {gives}, needed"
KIND_WORDS = {FRAMES: "frames", IMAGE: "image", PAIR: "starless + stars"}
#: "from ..." as a whole: the French joins the preposition to the word ("de brutes", "d'un fichier image")
SOURCE_WORDS = {"frames": "from frames", "image": "from image file", "stretched": "from stretched image",
                "fusion": "from fusion"}


class BlockRefused(ValueError):
    """A block that cannot be placed, removed or switched off there: the message says why (translated by the
    client, its fields filled)."""

    def __init__(self, template: str, **args: str):
        self.template, self.fields = template, args
        super().__init__(template.format(**args))


def _title(b: str) -> str:
    return BLOCK_TITLES.get(b, b)


def node(block: str, method: str | None = None, off: bool = False, name: str | None = None) -> dict[str, Any]:
    """A node of a workflow's graph."""
    if block not in IO:
        raise ValueError(f"unknown block {block!r} (allowed: {list(IO)})")
    m = method if method is not None else DEFAULT_METHOD.get(block)
    if block in METHODS and m not in METHODS[block]:
        raise ValueError(f"block {block!r}: method {m!r} is not in the engine (allowed: {list(METHODS[block])})")
    return {"block": block, "name": name or block, "method": m if block in METHODS else None, "off": bool(off)}


def new_name(blocks: Sequence[dict], block: str) -> str:
    """The name of a new node of ``block``: its stage's name, or "<stage>~<n>" when the graph has one (never reused
    while the graph has it)."""
    used = {n["name"] for n in blocks}
    if block not in used:
        return block
    k = 2
    while f"{block}~{k}" in used:
        k += 1
    return f"{block}~{k}"


def check_blocks(blocks: Any, source: str = "frames") -> list[dict[str, Any]]:
    """A graph's blocks as they are stored: nodes checked one by one, then the graph (``graph_problem``). Raises
    ValueError (BlockRefused for a rule of the graph)."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    if not isinstance(blocks, (list, tuple)):
        raise ValueError(f"a workflow's blocks are a list, not {blocks!r}")
    out = []
    for n in blocks:
        if not isinstance(n, dict) or "block" not in n:
            raise ValueError(f"a block is {{'block', 'name', 'method', 'off'}}, not {n!r}")
        nn = node(n["block"], n.get("method"), n.get("off", False), n.get("name"))
        if nn["block"] not in IMAGE_OPS and nn["name"] != nn["block"] or \
                nn["block"] in IMAGE_OPS and base(nn["name"]) != nn["block"]:
            raise ValueError(f"block {nn['block']!r} cannot be named {nn['name']!r}")
        out.append(nn)
    names = [n["name"] for n in out]
    if len(set(names)) != len(names):
        raise ValueError(f"two blocks of a workflow share a name: {names}")
    graph_problem(out, source, raise_it=True)
    return out


def _active(blocks: Sequence[dict]) -> list[dict]:
    return [n for n in blocks if not n.get("off")]


def graph_problem(blocks: Sequence[dict], source: str, raise_it: bool = False) -> BlockRefused | None:
    """The first rule a graph breaks (its switched-off blocks count as absent), or None."""
    cur = [""]

    def bad(template: str, **a: str):
        e = BlockRefused(template, **a)
        e.block = cur[0]                              # the block whose rule fails
        if raise_it:
            raise e
        return e
    act = _active(blocks)
    seen: set[str] = set()
    kind = SOURCE_GIVES[source]
    before = "the source"
    after_output = False
    frame_rank = -1
    for n in act:
        b = n["block"]
        cur[0] = b
        takes, gives = IO[b]
        if b not in IMAGE_OPS and b in seen:
            return bad(WHY_ONCE, block=_title(b))
        if source in ("image", "stretched", "fusion") and b in FRAME_BLOCKS:
            return bad(WHY_SOURCE, source=SOURCE_WORDS[source], block=_title(b))
        if source == "stretched" and b != "stars":
            return bad(WHY_SOURCE, source=SOURCE_WORDS[source], block=_title(b))
        if source == "fusion" and b not in ("output", "stars"):
            return bad(WHY_SOURCE, source=SOURCE_WORDS[source], block=_title(b))
        if after_output and b != "stars":
            return bad(WHY_AFTER_OUTPUT)
        if takes != kind and takes == IMAGE and kind == FRAMES:      # what to do, not only why (8.5)
            return bad(WHY_STACK_FIRST, block=_title(b))
        if takes != kind:
            return bad(WHY_INPUT, takes=KIND_WORDS[takes], before=before, gives=KIND_WORDS[kind])
        if b in FRAME_BLOCKS:
            r = FRAME_BLOCKS.index(b)
            if r < frame_rank:
                other = next(x for x in FRAME_BLOCKS[r + 1:] if x in seen)
                return bad(WHY_FRAME_ORDER, block=_title(b), where="before", other=_title(other))
            frame_rank = r
        if b == "color" and "astrometry" not in seen:
            return bad(WHY_NEEDS, need=_title("astrometry"))
        if b == "denoise" and (source != "frames" or "stacking" not in seen):
            return bad(WHY_HALVES)
        if b == "stars" and source in ("frames", "image") and "denoise" not in seen:      # trained on denoised images
            return bad(WHY_NEEDS, need=_title("denoise"))
        if b == "stars" and source not in ("stretched",) and "output" not in seen:
            return bad(WHY_NEEDS, need=_title("output"))
        seen.add(b)
        kind, before = gives, _title(b)
        if b == "output":
            after_output = True
    return None


def place_problem(blocks: Sequence[dict], source: str, block: str, index: int) -> BlockRefused | None:
    """Why ``block`` cannot be inserted at ``index`` of the graph (None: it can)."""
    if block == "output":
        return BlockRefused(WHY_OUTPUT_AUTO)
    if block not in IMAGE_OPS and any(n["block"] == block and n.get("off") for n in blocks):
        return BlockRefused(WHY_ONCE_OFF, block=_title(block))      # GUI exploration 4.4: no second one
    trial = list(blocks[:index]) + [node(block, name=new_name(blocks, block))] + list(blocks[index:])
    e = graph_problem(with_output(trial, source), source)
    if e is not None and e.template == WHY_STACK_FIRST and getattr(e, "block", "") == "output":
        e = BlockRefused(WHY_STACK_FIRST, block=_title(block))     # the block placed needs it, not the Output (never
        e.block = block                                             # a block on screen)
    return e


def places(blocks: Sequence[dict], source: str, block: str) -> list[int]:
    """The indices where ``block`` can be inserted."""
    return [i for i in range(len(blocks) + 1) if place_problem(blocks, source, block, i) is None]


def palette_problem(blocks: Sequence[dict], source: str, block: str) -> BlockRefused | None:
    """Why ``block`` cannot be placed anywhere in the graph (the palette greys it and says this), None when it can.
    Among the reasons at each place, the one of the place where the block would go by default."""
    if places(blocks, source, block):
        return None
    return place_problem(blocks, source, block, default_place(blocks, block))


def default_place(blocks: Sequence[dict], block: str) -> int:
    """Where a double-clicked block goes: after the last block that comes before it in the palette's usual order, or
    at the first valid place."""
    order = PALETTE
    r = order.index(block)
    idx = 0
    for i, n in enumerate(blocks):
        if order.index(n["block"]) <= r:
            idx = i + 1
    return idx


def insert_place(blocks: Sequence[dict], source: str, block: str) -> int | None:
    """The place a double-click inserts ``block`` at: the default one when valid, else the nearest valid place; None
    when there is none."""
    ok = places(blocks, source, block)
    if not ok:
        return None
    d = default_place(blocks, block)
    return min(ok, key=lambda i: (abs(i - d), -i))


def remove_problem(blocks: Sequence[dict], source: str, index: int) -> BlockRefused | None:
    """Why the block at ``index`` cannot be removed (None: it can)."""
    if blocks[index]["block"] == "output":
        return BlockRefused(WHY_OUTPUT_AUTO)
    trial = list(blocks[:index]) + list(blocks[index + 1:])
    e = graph_problem(with_output(trial, source), source)
    if e is None:
        return None
    return _explain_removal(blocks, trial, source, index, e)


def _explain_removal(blocks, trial, source, index, e):
    b = blocks[index]["block"]
    if e.template in (WHY_NEEDS, WHY_HALVES) and getattr(e, "block", ""):
        return BlockRefused(WHY_REMOVE_NEEDED, other=_title(e.block))
    if e.template in (WHY_INPUT, WHY_HALVES) or b == "stacking":
        takes, gives = IO[b]
        if takes != gives:
            return BlockRefused(WHY_REMOVE_BRIDGE, takes=KIND_WORDS[gives], block=_title(b))
        return BlockRefused(WHY_REMOVE_NEEDED, other=_title(next(n["block"] for n in _active(trial[index:]))))
    return e


def off_problem(blocks: Sequence[dict], source: str, index: int) -> BlockRefused | None:
    """Why the block at ``index`` cannot be switched off (None: it can; switching on is always checked too)."""
    n = blocks[index]
    takes, gives = IO[n["block"]]
    if n["block"] in ("output", "stacking", "stars") or takes != gives:
        return BlockRefused(WHY_OFF_KIND, block=_title(n["block"]), takes=KIND_WORDS[takes], gives=KIND_WORDS[gives])
    trial = [dict(x, off=True) if i == index else x for i, x in enumerate(blocks)]
    e = graph_problem(with_output(trial, source), source)
    if e is None:
        return None
    e = _explain_removal(blocks, [x for i, x in enumerate(blocks) if i != index], source, index, e)
    if e.template == WHY_REMOVE_NEEDED:                  # a switch-off is not a removal: say what to switch off
        return BlockRefused(WHY_OFF_NEEDED, **e.fields)
    return e


def from_options(options: dict[str, Any], end: str | None = None, source: str = "frames") -> list[dict[str, Any]]:
    """The graph of a workflow made before issue #47 (its options): the same chain, the same keys. Its cut blocks
    (Calibration, Registration: run as pass-through) are switched off; Astrometry, Gradient and Color, cut, are
    absent."""
    o = check_options(options)
    cut = set(cuts(o))
    out: list[dict] = []
    if source == "stretched":
        return [node("stars")] if o["stars"] else []
    if source == "fusion":
        return [node("output")] + ([node("stars")] if o["stars"] else [])
    if source == "frames":
        out += [node("calibration", off="calibration" in cut), node("registration", off="registration" in cut),
                node("selection"), node("stacking")]
        if check_end(end) == "stacking":
            return out + [node("output")]
    if "astrometry" not in cut:
        out.append(node("astrometry"))
    if "gradient" not in cut:
        out.append(node("gradient", o["gradient"]))
    if o["denoise"] and source == "frames":
        out.append(node("denoise", o["denoise"]))
    if "color" not in cut and "astrometry" not in cut:
        out.append(node("color"))
    out.append(node("output"))
    if o["stars"] and (source == "frames" and o["denoise"]):
        out.append(node("stars"))
    return out


def options_of(blocks: Sequence[dict], source: str = "frames") -> dict[str, Any]:
    """The options a graph stands for (what the code that reads options sees): each block's first active node's
    method, the absent or switched-off blocks cut."""
    act = _active(blocks)
    first = {}
    for n in act:
        first.setdefault(n["block"], n)
    o: dict[str, Any] = {"denoise": first["denoise"]["method"] if "denoise" in first else None,
                         "stars": "neural" if "stars" in first else None,
                         "gradient": first["gradient"]["method"] if "gradient" in first else "polynomial",
                         "deconvolution": None}
    if o["stars"] and not o["denoise"]:
        o["denoise"] = "pretrained"              # a stretched image or a fusion: their Split block needs no Denoise
    cut = []
    if source == "frames":
        cut += [b for b in ("calibration", "registration") if b not in first]
    cut += [b for b in ("astrometry", "gradient", "color") if b not in first]
    if cut:
        o["cut"] = cut
    return check_options(o)


def sequence_chain(blocks: Sequence[dict], source: str = "frames", *, reference: bool = False,
                   sessions: Sequence[int] = ()) -> list[str]:
    """The chain of a graph, in the graph's order: the source's stages, then each active node's (an image block's
    entry is its node's name). Calibration and Registration absent or switched off still run, as a pass-through."""
    act = _active(blocks)
    has = {n["block"] for n in act}
    out: list[str] = []
    if source == "fusion":
        out.append("fusion")
    elif source in ("image", "stretched"):
        out.append("image")
    elif not act:
        return out                                    # an empty graph from frames: nothing to compute (issue #49)
    else:
        # a graph being built stops where its blocks do: the frames are aligned when a block from Registration on is
        # placed (without Registration, as a pass-through), stacked when Stacking is
        later = {"registration", "selection", "stacking"} | {n["block"] for n in act if IO[n["block"]][0] == IMAGE}
        aligned = bool(has & later)
        if len(sessions) >= 2:
            out += [f"{s}@{sid}" for s in ("inventory", "calibrate") for sid in sessions]
            out += ["sessions", "register"] if aligned else ["sessions"]
        else:
            out += ["inventory", "calibrate"] + (["register"] if aligned else [])
        if "stacking" in has:
            out.append("stack")
    for n in act:
        b = n["block"]
        if b in FRAME_BLOCKS:
            continue                                  # in the source's stages above
        if b == "astrometry":
            out += ["astrometry"] + (["reference"] if reference else []) + (["catalogs"] if "color" in has else []) \
                + ["surveys"]
        elif b in IMAGE_OPS:
            out.append(n["name"])
        elif b == "output":
            out.append("render_simple")
        elif b == "stars":
            out.append("star_removal")
    return out


def node_of(blocks: Sequence[dict], entry: str) -> dict | None:
    """The node a chain entry belongs to (None for the source's stages)."""
    b = base(entry)
    for n in blocks:
        if n["name"] == entry or (n["block"] not in IMAGE_OPS and BLOCKS.get(n["block"]) and b in BLOCKS[n["block"]]):
            return n
    return None


def node_implied(blocks: Sequence[dict], entry: str, source: str = "frames") -> dict[str, Any]:
    """The parameter values a graph imposes on one of its chain's entries: the methods of its node, the
    pass-through of an absent or switched-off Calibration or Registration, and a Denoise that keeps its denoised
    half-stacks for a later Denoise."""
    b = base(entry)
    act = _active(blocks)
    has = {n["block"] for n in act}
    if b == "calibrate" and "calibration" not in has:
        return {"apply_masters": False}
    if b == "register" and "registration" not in has:
        return {"align": False}
    n = next((x for x in act if x["name"] == entry), None) if b in IMAGE_OPS else None
    if b == "gradient" and n is not None:
        return {"method": n["method"]}
    if b == "denoise" and n is not None:
        out = {"method": n["method"]}
        i = act.index(n)
        if any(x["block"] == "denoise" for x in act[i + 1:]):
            out["keep_halves"] = True
        return out
    return {}


def sync_blocks(blocks: Sequence[dict], source: str, options: dict[str, Any]) -> list[dict[str, Any]]:
    """A graph brought in line with options set by code that speaks options (a method chosen, Denoise or Stars added
    or taken off, a block cut or restored): the first node of each block takes the method; a block the options add
    goes to its usual place; one they take off goes (Calibration and Registration: switched off)."""
    o = check_options(options)
    g = [dict(n) for n in blocks]
    if options_of(g, source) == o:
        return g
    cut = set(cuts(o))

    def first(b):
        return next((n for n in g if n["block"] == b and not n["off"]), None)

    def add(b, method=None):
        i = insert_place(g, source, b)
        if i is not None:
            g.insert(i, node(b, method, name=new_name(g, b)))

    def drop(b):
        g[:] = [n for n in g if n["block"] != b]
    if not o["stars"]:
        drop("stars")
    if o["denoise"] is None:
        drop("denoise")
    for b in ("calibration", "registration"):
        if source == "frames":
            for n in g:
                if n["block"] == b:
                    n["off"] = b in cut
    for b in ("color", "gradient", "astrometry"):
        if b in cut:
            drop(b)
    for b in ("astrometry", "gradient", "color"):
        if b not in cut and first(b) is None and source in ("frames", "image") and any(n["block"] == "output"
                                                                                      for n in g):
            add(b, o["gradient"] if b == "gradient" else None)
    if o["denoise"] and first("denoise") is None and source == "frames":
        add("denoise", o["denoise"])
    if o["stars"] and first("stars") is None:
        add("stars")
    for b in ("gradient", "denoise"):
        n = first(b)
        if n is not None and o.get(b):
            n["method"] = o[b]
    return g


def with_output(blocks: Sequence[dict], source: str = "frames") -> list[dict[str, Any]]:
    """A graph with its Output block where it belongs (issue #49: the user does not place it): right after the last
    block that gives an image, before Stars/background; none while the graph gives no image (it ends on
    frames). A stretched image has none (it is its own output)."""
    g = [dict(n) for n in blocks if n["block"] != "output"]
    if source == "stretched":
        return g
    act = [n for n in g if not n.get("off")]
    image = source in ("image", "fusion") or any(IO[n["block"]][1] in (IMAGE, PAIR) for n in act)
    if not image:
        return g
    stars = next((i for i, n in enumerate(g) if n["block"] == "stars"), len(g))
    return g[:stars] + [node("output")] + g[stars:]


def source_of(w: Any) -> str:
    """A workflow's source kind."""
    src = getattr(w, "source", None)
    if not src:
        return "frames"
    if "fusion" in src:
        return "fusion"
    return "stretched" if src.get("stretched") else "image"


def graph_of(w: Any) -> list[dict[str, Any]]:
    """A workflow's graph: its blocks, or the graph of its options (a workflow made before issue #47)."""
    if getattr(w, "blocks", None) is not None:
        return [dict(n) for n in w.blocks]
    return from_options(w.options, w.end, source_of(w))


_CHAINS: list[Chains] = []


def chains() -> Chains:
    """The v2 stage graph (built on first use: it imports the stages)."""
    if not _CHAINS:
        from ..stages.v2 import V2_STAGES
        _CHAINS.append(Chains(V2_STAGES(), order=ORDER, extra=EXTRA, optional=OPTIONAL, blocks=BLOCKS,
                              rule=chain_stages))
    return _CHAINS[0]


def chain(options: dict[str, Any], end: str | None = None, *, reference: bool = False,
          sessions: Sequence[int] = (), image: bool = False, mono: bool = False, fusion: bool = False,
          stretched: bool = False) -> list[str]:
    """The checked chain of a workflow on the v2 stages."""
    return chains().chain(options, end, reference=reference, sessions=sessions, image=image, mono=mono,
                          fusion=fusion, stretched=stretched)
