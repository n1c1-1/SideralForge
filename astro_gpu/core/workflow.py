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
NOTE_DENOISE_ADDED = "Denoise added: the star removal needs it"
REFUSE_DENOISE = "the star removal needs it: remove Stars first"
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
NEEDS_HALF_STACKS = "needs the half-stacks of a stack"
NEEDS_DENOISE = "needs Denoise"
MONO_NO_COLOUR = "a monochrome image has no colours: they come from combining filters"


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
         "surveys", "gradient", "color", "denoise", "render_simple", "star_removal"]
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
    "color": ("color",),
    "denoise": ("denoise",),
    "stars": ("star_removal",),                     # drawn before the Output, which every workflow ends with
    "output": ("render_simple",),
}
BLOCK_TITLES = {"fusion": "Fusion", "image": "Image", "calibration": "Calibration", "registration": "Registration",
                "selection": "Selection",
                "stacking": "Stacking", "astrometry": "Astrometry", "color": "Color", "denoise": "Denoise",
                "stars": "Stars", "gradient": "Gradient", "output": "Output"}
#: what Run to here runs for a block without stages of its own
RUN_TO: dict[str, tuple[str, ...]] = {"selection": ("register",)}
#: stages a block's key leaves out: the survey follows the Gradient method, not the Astrometry block
KEY_LEAVES_OUT: dict[str, tuple[str, ...]] = {"astrometry": ("surveys",)}


def base(name: str) -> str:
    """The stage of a chain entry: "calibrate" for the instance "calibrate@2"."""
    return name.split("@", 1)[0]


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
        for d in self.deps2[b]:
            r = self.resolve(chain, stage, d)
            if r is not None:
                if r not in out:
                    out.append(r)
            elif d not in self.optional.get(b, ()):
                raise ValueError(f"stage '{stage}' needs the {self.block_title(self.block_of(d))} block "
                                 f"(stage '{d}')")
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
        block of its session: "calibration@2"."""
        b = self._block_of.get(base(stage), base(stage))
        sid = session_of(stage)
        return f"{b}@{sid}" if sid is not None else b

    @staticmethod
    def block_title(block: str) -> str:
        return BLOCK_TITLES.get(base(block), base(block))

    def blocks_in(self, chain: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
        """(block, its stages in the chain) for the blocks of a chain, in graph order; Selection comes with
        register. The instances of a session make the block of that session ("calibration@2"), in session order."""
        inside = set(chain)
        out = []
        for b, ss in self.blocks.items():
            mine = tuple(s for s in ss if s in inside)
            if mine or (b in RUN_TO and all(s in inside for s in RUN_TO[b])):
                out.append((b, mine))
            sids = list(dict.fromkeys(session_of(s) for s in chain if session_of(s) is not None and base(s) in ss))
            out += [(f"{b}@{sid}", tuple(s for s in chain if base(s) in ss and session_of(s) == sid)) for sid in sids]
        return out

    def run_to(self, block: str, chain: Sequence[str]) -> list[str]:
        """The stages Run to here targets for a block (their missing ancestors are the runner's business)."""
        b, sid = base(block), session_of(block)
        if b not in self.blocks:
            raise ValueError(f"unknown block {block!r}")
        ss = RUN_TO.get(b) or self.blocks[b]
        return [s for s in chain if base(s) in ss and session_of(s) == sid]

    def key_stages(self, block: str, chain: Sequence[str]) -> list[str]:
        """The stages whose keys make a block's key (Output adds the workflow id: never shared)."""
        b, sid = base(block), session_of(block)
        return [s for s in chain if base(s) in self.blocks.get(b, ()) and session_of(s) == sid
                and base(s) not in KEY_LEAVES_OUT.get(b, ())]


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
