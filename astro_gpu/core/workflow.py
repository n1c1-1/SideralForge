"""Workflows of a v2 project: options → the chain of stages, the blocks, the option rules.

Entry points: ``check_options``, ``set_option`` (the rules, with their notes), ``chain`` / ``chains()`` (the stage
graph, ``Chains``), ``method_of``, ``implied``, ``TEMPLATES``.

A workflow is a set of options (which optional blocks it has and the method of each) plus the settings the user
stored. The options give its chain: the stages it runs, in the global order. Two workflows share a stage result
when the stage key is the same (core/store.py); this module only says which stages a workflow has and which of them
each stage reads.

    options   {"denoise": None | "n2n", "stars": None | "psf" | "psf_neural",
               "gradient": "halpha" | "polynomial", "deconvolution": None}
    end       None, or "stacking" for the Raw stack exit (inventory → stack, then render_simple)

Rules: Stars needs Denoise; the H-alpha gradient needs Stars (the background stage works on the starless image);
psf_neural with a stored ``starless.learned_stars`` file omits star_net (an earlier neural layer). Other option
values are not in the engine and raise.

The stage list comes from ``stages/v2.py`` (``V2_STAGES``, loaded lazily); tests build toy ``Chains``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

# ---------------------------------------------------------------------------------------------- options

#: the allowed values of each option
OPTION_VALUES: dict[str, tuple[Any, ...]] = {
    "denoise": (None, "n2n"),
    "stars": (None, "psf", "psf_neural"),
    "gradient": ("halpha", "polynomial"),
    "deconvolution": (None,),
}
#: a missing option takes this value (the Classic stack)
DEFAULT_OPTIONS: dict[str, Any] = {"denoise": None, "stars": None, "gradient": "polynomial", "deconvolution": None}
#: early exits: the block a workflow's output comes from when it does not go to the end
EXITS = ("stacking",)

#: the workflow templates; a template is only used at creation
TEMPLATES: dict[str, dict[str, Any]] = {
    "Classic stack": {"denoise": None, "stars": None, "gradient": "polynomial", "deconvolution": None},
    "Classic + denoise": {"denoise": "n2n", "stars": None, "gradient": "polynomial", "deconvolution": None},
    "Full": {"denoise": "n2n", "stars": "psf", "gradient": "halpha", "deconvolution": None},
    "Full + neural stars": {"denoise": "n2n", "stars": "psf_neural", "gradient": "halpha", "deconvolution": None},
}

# the notes of the option rules (shown in the client, translated by its catalogues)
NOTE_DENOISE_ADDED = "Denoise added: the star measurement needs it"
NOTE_STARS_ADDED_NEURAL = "Stars added: neural star removal starts from the star model"
NOTE_STARS_ADDED_HALPHA = "Stars added: the anchored background is computed on the starless image"
NOTE_REPLACES_POLYNOMIAL = "Replaces the simple background"
NOTE_REPLACES_HALPHA = "Replaces the H-alpha anchored background"
NOTE_NEURAL_REMOVED = "Neural star removal removed too: it starts from the star model"
NOTE_POLYNOMIAL_INSTEAD = ("Simple background instead of the anchored one: the anchored background is computed on the "
                           "starless image")
REFUSE_DENOISE = "the star measurement needs it: remove Stars first"
REFUSE_GRADIENT = "a background step is required: drop the H-alpha anchored background on it to replace it"
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
    for k, allowed in OPTION_VALUES.items():
        if o[k] not in allowed:
            raise ValueError(f"workflow option {k}={o[k]!r} is not in the engine (allowed: {list(allowed)})")
    if o["stars"] and not o["denoise"]:
        raise ValueError(f"workflow option stars={o['stars']!r} needs denoise: the star measurement works on the "
                         "denoised stack")
    if o["gradient"] == "halpha" and not o["stars"]:
        raise ValueError("workflow option gradient='halpha' needs stars: the H-alpha gradient is computed on the "
                         "starless image")
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
    elif key == "stars":
        if value is None:
            if o["stars"] == "psf_neural":
                notes.append(NOTE_NEURAL_REMOVED)
            if o["gradient"] == "halpha":
                o["gradient"] = "polynomial"
                notes.append(NOTE_POLYNOMIAL_INSTEAD)
        elif value == "psf_neural" and o["stars"] is None:
            notes.append(NOTE_STARS_ADDED_NEURAL)
    elif key == "gradient":
        if value == "halpha":
            notes.append(NOTE_REPLACES_POLYNOMIAL)
            if not o["stars"]:
                o["stars"] = "psf"
                notes.append(NOTE_STARS_ADDED_HALPHA)
        else:
            notes.append(NOTE_REPLACES_HALPHA)
    o[key] = value
    if o["stars"] and not o["denoise"]:
        o["denoise"] = "n2n"
        notes.append(NOTE_DENOISE_ADDED)
    return check_options(o), notes


def method_of(options: dict[str, Any], block: str) -> str | None:
    """The internal method of a block: denoise 'n2n', stars 'psf' / 'psf_neural', gradient 'halpha' /
    'poly_simple' (background_poly, without Stars) / 'poly_physical' (background with surveys.survey 'none', with
    Stars); None when the workflow does not have the block. The definition keeps gradient 'polynomial' for both."""
    o = check_options(options)
    if block == "gradient":
        if o["gradient"] == "halpha":
            return "halpha"
        return "poly_physical" if o["stars"] else "poly_simple"
    if block in ("denoise", "stars", "deconvolution"):
        return o[block]
    raise ValueError(f"block {block!r} has no method")


def implied(options: dict[str, Any], stage: str) -> dict[str, Any]:
    """The parameter values the options impose on a stage (they win over the stored ones): star_net enabled with
    psf_neural; no H-alpha survey for the polynomial gradient with Stars."""
    o = check_options(options)
    if stage == "star_net" and o["stars"] == "psf_neural":
        return {"enabled": True}
    if stage == "surveys" and o["gradient"] == "polynomial" and o["stars"]:
        return {"survey": "none"}
    return {}


def names_file(value: Any) -> bool:
    """Whether a 'Learned star layer' value names a file ('', 'none' and 'off' mean no layer; star_net.layer_file)."""
    v = str(value or "").strip()
    return bool(v) and v.lower() not in ("none", "off")


def has_learned_layer(params: dict[str, dict[str, Any]]) -> bool:
    """Whether a workflow's stored settings name an earlier learned star layer (``starless.learned_stars``)."""
    return names_file((params.get("starless") or {}).get("learned_stars"))


# ---------------------------------------------------------------------------------------------- the chain table

#: the global order of the v2 stages (each stage only reads stages before it)
ORDER = ["inventory", "calibrate", "register", "stack", "astrometry", "reference", "catalogs", "surveys", "color",
         "denoise", "optics", "stars", "star_net", "starless", "background", "background_poly", "render_prep",
         "render_prep_learned", "render_simple"]
#: dependencies a stage has beyond its declared ``deps`` (an implicit read through starless' auto value)
EXTRA: dict[str, tuple[str, ...]] = {"starless": ("star_net",)}
#: dependencies that may be absent from a chain (the stage then runs without them)
OPTIONAL: dict[str, tuple[str, ...]] = {
    "starless": ("star_net",),
    "background_poly": ("denoise",),
    "render_simple": ("color", "denoise", "background_poly"),
}
#: the blocks of the graph and their stages; Selection has no stage of its own (it follows register)
BLOCKS: dict[str, tuple[str, ...]] = {
    "calibration": ("inventory", "calibrate"),
    "registration": ("register",),
    "selection": (),
    "stacking": ("stack",),
    "astrometry": ("astrometry", "reference", "catalogs", "surveys"),
    "color": ("color",),
    "denoise": ("denoise",),
    "stars": ("optics", "stars", "star_net", "starless"),
    "gradient": ("background", "background_poly"),
    "output": ("render_prep", "render_prep_learned", "render_simple"),
}
BLOCK_TITLES = {"calibration": "Calibration", "registration": "Registration", "selection": "Selection",
                "stacking": "Stacking", "astrometry": "Astrometry", "color": "Color", "denoise": "Denoise",
                "stars": "Stars", "gradient": "Gradient", "output": "Output"}
#: what Run to here runs for a block without stages of its own
RUN_TO: dict[str, tuple[str, ...]] = {"selection": ("register",)}
#: stages a block's key leaves out: the survey follows the Gradient method, not the Astrometry block
KEY_LEAVES_OUT: dict[str, tuple[str, ...]] = {"astrometry": ("surveys",)}


def chain_stages(options: dict[str, Any], end: str | None = None, reference: bool = False,
                 learned_layer: bool = False) -> list[str]:
    """The stages of a workflow, in the global order. ``reference``: the project has a reference
    image (the stage sits in the Astrometry block); ``learned_layer``: the workflow stores an earlier learned star
    layer, so psf_neural runs without star_net."""
    o = check_options(options)
    out = ["inventory", "calibrate", "register", "stack"]
    if check_end(end) == "stacking":
        return out + ["render_simple"]
    out += ["astrometry"] + (["reference"] if reference else []) + ["catalogs"]
    if o["stars"]:
        out.append("surveys")                       # background reads it (survey 'none' for the polynomial)
    out.append("color")
    if o["denoise"]:
        out.append("denoise")
    if o["stars"]:
        out += ["optics", "stars"]
        if o["stars"] == "psf_neural" and not learned_layer:
            out.append("star_net")
        out += ["starless", "background", "render_prep_learned" if o["stars"] == "psf_neural" else "render_prep"]
    else:
        out += ["background_poly", "render_simple"]
    return [s for s in ORDER if s in out]


Rule = Callable[[dict, "str | None", bool, bool], Iterable[str]]


class Chains:
    """The stage graph of v2 workflows: the stages, their order and dependencies (declared ∪ extra ∪ optional),
    the blocks, and the chain of a workflow (``rule``: options, end, reference, learned_layer → stage names)."""

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
              learned_layer: bool = False) -> list[str]:
        """The stages of a workflow, ordered and checked."""
        if self.rule is None:
            raise TypeError("these Chains have no rule: use ordered() with the stage names")
        return self.ordered(self.rule(options, end, reference, learned_layer))

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


def chain(options: dict[str, Any], end: str | None = None, *, reference: bool = False,
          learned_layer: bool = False) -> list[str]:
    """The checked chain of a workflow on the v2 stages."""
    return chains().chain(options, end, reference=reference, learned_layer=learned_layer)
