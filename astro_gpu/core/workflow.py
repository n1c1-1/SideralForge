"""Workflows of a v2 project: options → the chain of stages, the blocks, the option rules.

Entry points: ``check_options``, ``set_option`` (the rules, with their notes), ``chain`` / ``chains()`` (the stage
graph, ``Chains``), ``method_of``, ``implied``, ``TEMPLATES``.

A workflow is a set of options (which optional blocks it has and the method of each) plus the settings the user
stored. The options give its chain: the stages it runs, in the global order. Two workflows share a stage result
when the stage key is the same (core/store.py); this module only says which stages a workflow has and which of them
each stage reads.

    options   {"denoise": None | "pretrained" | "adapted", "stars": None | "neural", "gradient": "polynomial" | "rbf" |
               "automatic", "deconvolution": None}
    end       None, or "stacking" for the Raw stack exit (inventory → stack, then render_simple)

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
#: a missing option takes this value (the Classic stack)
DEFAULT_OPTIONS: dict[str, Any] = {"denoise": None, "stars": None, "gradient": "polynomial", "deconvolution": None}
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
    unknown = set(options) - set(OPTION_VALUES)
    if unknown:
        raise ValueError(f"unknown workflow option(s): {sorted(unknown)}")
    o = {k: options.get(k, DEFAULT_OPTIONS[k]) for k in OPTION_VALUES}
    for k, alias in OPTION_ALIASES.items():
        o[k] = alias.get(o[k], o[k])
    for k, allowed in OPTION_VALUES.items():
        if o[k] not in allowed:
            raise ValueError(f"workflow option {k}={o[k]!r} is not in the engine (allowed: {list(allowed)})")
    if o["stars"] and not o["denoise"]:
        raise ValueError(f"workflow option stars={o['stars']!r} needs denoise: the star removal works on the "
                         "denoised stack")
    return o


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
    return {}


# ---------------------------------------------------------------------------------------------- the chain table

#: the global order of the v2 stages (each stage only reads stages before it)
ORDER = ["inventory", "calibrate", "register", "stack", "astrometry", "reference", "catalogs", "surveys", "gradient",
         "color", "denoise", "render_simple", "star_removal"]
#: dependencies a stage has beyond its declared ``deps``
EXTRA: dict[str, tuple[str, ...]] = {}
#: dependencies that may be absent from a chain (the stage then runs without them); the stages that read the stack
#: read it with the sky gradient removed when the chain has the Gradient block (gradient.stack_image)
OPTIONAL: dict[str, tuple[str, ...]] = {
    "color": ("gradient",),
    "denoise": ("gradient",),
    "render_simple": ("color", "denoise", "gradient"),
    "star_removal": ("gradient",),
}
#: the blocks of the graph and their stages; Selection has no stage of its own (it follows register)
BLOCKS: dict[str, tuple[str, ...]] = {
    "calibration": ("inventory", "calibrate"),
    "registration": ("register",),
    "selection": (),
    "stacking": ("stack",),
    "astrometry": ("astrometry", "reference", "catalogs", "surveys"),
    "gradient": ("gradient",),
    "color": ("color",),
    "denoise": ("denoise",),
    "stars": ("star_removal",),                     # drawn before the Output, which every workflow ends with
    "output": ("render_simple",),
}
BLOCK_TITLES = {"calibration": "Calibration", "registration": "Registration", "selection": "Selection",
                "stacking": "Stacking", "astrometry": "Astrometry", "color": "Color", "denoise": "Denoise",
                "stars": "Stars", "gradient": "Gradient", "output": "Output"}
#: what Run to here runs for a block without stages of its own
RUN_TO: dict[str, tuple[str, ...]] = {"selection": ("register",)}
#: stages a block's key leaves out: the survey follows the Gradient method, not the Astrometry block
KEY_LEAVES_OUT: dict[str, tuple[str, ...]] = {"astrometry": ("surveys",)}


def chain_stages(options: dict[str, Any], end: str | None = None, reference: bool = False) -> list[str]:
    """The stages of a workflow, in the global order. ``reference``: the project has a reference image (the stage sits
    in the Astrometry block)."""
    o = check_options(options)
    out = ["inventory", "calibrate", "register", "stack"]
    if check_end(end) == "stacking":
        return out + ["render_simple"]
    out += ["astrometry"] + (["reference"] if reference else []) + ["catalogs", "gradient", "color"]
    if o["denoise"]:
        out.append("denoise")
    out.append("render_simple")
    if o["stars"]:
        out.append("star_removal")
    return [s for s in ORDER if s in out]


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
    def chain(self, options: dict[str, Any], end: str | None = None, *, reference: bool = False) -> list[str]:
        """The stages of a workflow, ordered and checked."""
        if self.rule is None:
            raise TypeError("these Chains have no rule: use ordered() with the stage names")
        return self.ordered(self.rule(options, end, reference))

    def ordered(self, names: Iterable[str]) -> list[str]:
        """Stage names as a chain: in the global order, every required dependency present (ValueError otherwise)."""
        names = set(names)
        unknown = names - set(self.stages)
        if unknown:
            raise ValueError(f"unknown stage(s): {sorted(unknown)}")
        out = [n for n in self.order if n in names]
        for n in out:
            self.deps_in_chain(out, n)
        return out

    def deps_in_chain(self, chain: Sequence[str], stage: str) -> tuple[str, ...]:
        """What a stage reads in this chain: its dependencies present in it. A missing required dependency is a
        ValueError naming its block."""
        inside = set(chain)
        if stage not in inside:
            raise ValueError(f"stage '{stage}' is not in the chain")
        out = []
        for d in self.deps2[stage]:
            if d in inside:
                out.append(d)
            elif d not in self.optional.get(stage, ()):
                raise ValueError(f"stage '{stage}' needs the {self.block_title(self.block_of(d))} block "
                                 f"(stage '{d}')")
        return tuple(out)

    def ancestors(self, chain: Sequence[str], stage: str) -> list[str]:
        """Every stage this one reads, directly or not, in the chain, in order."""
        seen: set[str] = set()
        todo = list(self.deps_in_chain(chain, stage))
        while todo:
            d = todo.pop()
            if d not in seen:
                seen.add(d)
                todo += self.deps_in_chain(chain, d)
        return [s for s in self.order if s in seen]

    def downstream(self, chain: Sequence[str], stage: str) -> list[str]:
        """The stages of the chain that read this one, directly or not, in order (the stage excluded)."""
        return [s for s in chain if s != stage and stage in self.ancestors(chain, s)]

    # ------------------------------------------------------------------ blocks
    def block_of(self, stage: str) -> str:
        """The block a stage belongs to (the stage name itself when no block lists it)."""
        return self._block_of.get(stage, stage)

    @staticmethod
    def block_title(block: str) -> str:
        return BLOCK_TITLES.get(block, block)

    def blocks_in(self, chain: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
        """(block, its stages in the chain) for the blocks of a chain, in graph order; Selection comes with
        register."""
        inside = set(chain)
        out = []
        for b, ss in self.blocks.items():
            mine = tuple(s for s in ss if s in inside)
            if mine or (b in RUN_TO and all(s in inside for s in RUN_TO[b])):
                out.append((b, mine))
        return out

    def run_to(self, block: str, chain: Sequence[str]) -> list[str]:
        """The stages Run to here targets for a block (their missing ancestors are the runner's business)."""
        if block not in self.blocks:
            raise ValueError(f"unknown block {block!r}")
        return [s for s in chain if s in (RUN_TO.get(block) or self.blocks[block])]

    def key_stages(self, block: str, chain: Sequence[str]) -> list[str]:
        """The stages whose keys make a block's key (Output adds the workflow id: never shared)."""
        return [s for s in chain if s in self.blocks.get(block, ()) and s not in KEY_LEAVES_OUT.get(block, ())]


_CHAINS: list[Chains] = []


def chains() -> Chains:
    """The v2 stage graph (built on first use: it imports the stages)."""
    if not _CHAINS:
        from ..stages.v2 import V2_STAGES
        _CHAINS.append(Chains(V2_STAGES(), order=ORDER, extra=EXTRA, optional=OPTIONAL, blocks=BLOCKS,
                              rule=chain_stages))
    return _CHAINS[0]


def chain(options: dict[str, Any], end: str | None = None, *, reference: bool = False) -> list[str]:
    """The checked chain of a workflow on the v2 stages."""
    return chains().chain(options, end, reference=reference)
