"""Workflows v2: options → chain (DESIGN §2.1), dependencies in a chain (§2.2), the option rules of the mockup."""
from __future__ import annotations

import re

import pytest

from astro_gpu.core import workflow as wfl
from astro_gpu.core.workflow import TEMPLATES, Chains, OptionRefused, chain, chain_stages, chains, set_option

BASE = ["inventory", "calibrate", "register", "stack"]
TABLE = {      # §2.1: the stages after stack, without the reference image
    "Classic stack": ["astrometry", "catalogs", "surveys", "gradient", "color", "render_simple"],
    "Classic + denoise": ["astrometry", "catalogs", "surveys", "gradient", "denoise", "color", "render_simple"],
    "Full": ["astrometry", "catalogs", "surveys", "gradient", "denoise", "color", "render_simple", "star_removal"],
}
CLASSIC, CLASSIC_DN, FULL = (TEMPLATES[k] for k in TABLE)


# ---------------------------------------------------------------------------------------------- the chains

@pytest.mark.parametrize("name", list(TABLE))
def test_template_chains_equal_the_table(name):
    assert chain_stages(TEMPLATES[name]) == BASE + TABLE[name]
    assert chain(TEMPLATES[name]) == BASE + TABLE[name]           # checked on the real stage graph
    with_ref = chain(TEMPLATES[name], reference=True)
    assert with_ref == BASE + ["astrometry", "reference"] + TABLE[name][1:]


@pytest.mark.parametrize("name", list(TABLE))
def test_raw_stack_exit(name):
    """The Raw stack exit is the same chain whatever the source workflow's options."""
    assert chain(TEMPLATES[name], "stacking") == BASE + ["render_simple"]
    assert chain(TEMPLATES[name], "stacking", reference=True) == BASE + ["render_simple"]


def test_the_earlier_star_methods_read_as_the_network():
    """A stored stars "psf" or "psf_neural" (the star model, then redrawn) is read as the network that replaced them."""
    for old in ("psf", "psf_neural"):
        assert wfl.check_options({**FULL, "stars": old}) == FULL
        assert chain({**FULL, "stars": old}) == BASE + TABLE["Full"]


def test_gradient_method():
    for o in (CLASSIC, FULL):
        assert wfl.method_of(o, "gradient") == "polynomial"
        assert wfl.implied(o, "gradient") == {"method": "polynomial"}
        assert "surveys" in chain(o)                     # a guide for the samples, read if there
    # the H-alpha anchored background of earlier versions is read as the method that replaces it
    assert wfl.check_options({**FULL, "gradient": "halpha"}) == FULL


def test_implied_values():
    """The options impose the method of the Gradient and Denoise blocks, nothing else (the catalogue is the same with
    or without Stars)."""
    assert wfl.implied(FULL, "stack") == {}
    assert wfl.implied(FULL, "star_removal") == {}
    assert wfl.implied(FULL, "catalogs") == wfl.implied(CLASSIC, "catalogs") == {}
    assert wfl.implied(FULL, "denoise") == {"method": "pretrained"}


@pytest.mark.parametrize("options,end", [
    ({"denoise": "nlbayes"}, None),                                   # ◇ methods
    ({"denoise": "pretrained", "stars": "neural", "gradient": "polynomial", "deconvolution": "rl"}, None),
    ({"stars": "learned"}, None),
    ({"gradient": None}, None),
    ({"gradient": "mars"}, None),
    ({"stars": "neural", "gradient": "polynomial"}, None),            # Stars needs Denoise
    ({"denoise": "pretrained", "colour": True}, None),                       # unknown option
    (CLASSIC, "astrometry"),                                          # not an exit
])
def test_invalid_options_raise(options, end):
    with pytest.raises(ValueError):
        chain(options, end)
    with pytest.raises(ValueError):
        chain_stages(options, end)


def test_missing_options_take_the_classic_defaults():
    assert wfl.check_options({}) == CLASSIC
    assert list(wfl.check_options({"deconvolution": None, "denoise": "pretrained"})) == list(wfl.OPTION_VALUES)


# ---------------------------------------------------------------------------------------------- the option rules

# the mockup's with_tool / without_tool (model.py:279-332), in the v2 options
TOOL_CASES = [
    # with Stars: Denoise comes with it
    (CLASSIC, "stars", "neural", {**CLASSIC, "denoise": "pretrained", "stars": "neural"}, [wfl.NOTE_DENOISE_ADDED]),
    (CLASSIC_DN, "stars", "neural", FULL, []),
    # with Denoise
    (CLASSIC, "denoise", "pretrained", CLASSIC_DN, []),
    # without Stars
    (FULL, "stars", None, CLASSIC_DN, []),
    # without Denoise (no Stars)
    (CLASSIC_DN, "denoise", None, CLASSIC, []),
    # no change, no note
    (FULL, "gradient", "polynomial", FULL, []),
]


@pytest.mark.parametrize("before,key,value,after,notes", TOOL_CASES)
def test_option_rules(before, key, value, after, notes):
    new, got = set_option(before, key, value)
    assert new == after
    assert got == notes
    chain(new)                                      # every result is a valid workflow


@pytest.mark.parametrize("before,key,value,why", [
    (FULL, "denoise", None, wfl.REFUSE_DENOISE),           # Denoise cannot go while Stars is there
    (FULL, "gradient", None, wfl.REFUSE_GRADIENT),         # the Gradient cannot be removed, only its method changes
    (CLASSIC, "gradient", None, wfl.REFUSE_GRADIENT),
])
def test_option_rules_refuse(before, key, value, why):
    with pytest.raises(OptionRefused, match="^" + re.escape(why) + "$"):
        set_option(before, key, value)


def test_no_block_after_an_early_output():
    with pytest.raises(OptionRefused):
        set_option(CLASSIC, "denoise", "pretrained", end="stacking")


def test_set_option_rejects_invalid_values():
    for key, value in (("denoise", "nlbayes"), ("stars", "starnet"), ("stars", "psf"), ("deconvolution", "rl"),
                       ("colour", 1)):
        with pytest.raises(ValueError) as e:
            set_option(FULL, key, value)
        assert not isinstance(e.value, OptionRefused)


def test_set_option_does_not_change_its_input():
    before = dict(FULL)
    set_option(FULL, "stars", None)
    assert FULL == before


# ---------------------------------------------------------------------------------------------- dependencies

def test_optional_and_extra_dependencies():
    ch = chains()
    full = chain(FULL)
    assert ch.deps_in_chain(full, "star_removal") == ("render_simple", "stack", "gradient")
    assert set(ch.deps_in_chain(chain(CLASSIC), "color")) == {"stack", "catalogs", "register", "gradient"}
    assert ch.deps_in_chain(chain(CLASSIC_DN), "denoise") == ("stack", "gradient")
    assert ch.deps_in_chain(chain(CLASSIC_DN), "render_simple") == ("stack", "color", "denoise", "gradient")
    assert ch.deps_in_chain(chain(CLASSIC, "stacking"), "render_simple") == ("stack",)
    assert ch.ancestors(chain(CLASSIC, "stacking"), "render_simple") == BASE
    assert ch.downstream(full, "denoise") == ["render_simple", "star_removal"]


def test_missing_required_dependency_names_the_block():
    ch = chains()
    stages = [s for s in chain(FULL) if s != "render_simple"]
    with pytest.raises(ValueError, match="Output block"):
        ch.ordered(stages)
    with pytest.raises(ValueError, match="Astrometry block"):
        ch.deps_in_chain(BASE + ["color"], "color")


def test_v2_stage_graph():
    ch = chains()
    assert ch.order == wfl.ORDER and set(ch.stages) == set(wfl.ORDER)
    new_stages = {"gradient": ("stack",),                                  # the dependencies of the new stages
                  "render_simple": ("stack", "color", "denoise", "gradient"),
                  "star_removal": ("render_simple", "stack", "gradient")}
    for name, deps in new_stages.items():
        assert set(deps) <= set(ch.deps2[name])
    for name in wfl.ORDER:                                     # every stage sits in exactly one block
        assert sum(name in ss for ss in wfl.BLOCKS.values()) == 1


def test_blocks():
    ch = chains()
    c = chain(FULL, reference=True)
    blocks = ch.blocks_in(c)
    assert [b for b, _ in blocks] == ["calibration", "registration", "selection", "stacking", "astrometry",
                                      "gradient", "denoise", "color", "stars", "output"]
    assert dict(blocks)["stars"] == ("star_removal",)
    assert ch.run_to("selection", c) == ["register"]
    assert ch.run_to("gradient", c) == ["gradient"]
    assert ch.run_to("stars", c) == ["star_removal"]
    assert ch.key_stages("astrometry", c) == ["astrometry", "reference"]   # the survey follows Gradient; the
    # catalogs follow a Color block after it (issue #47: a branch without Color shares the plate solution)
    assert ch.block_of("star_removal") == "stars" and ch.block_of("register") == "registration"
    assert [b for b, _ in ch.blocks_in(chain(CLASSIC, "stacking"))] == ["calibration", "registration", "selection",
                                                                        "stacking", "output"]


# ---------------------------------------------------------------------------------------------- toy chains

class _Toy:
    def __init__(self, name, deps=()):
        self.name, self.deps = name, tuple(deps)


def test_toy_chains():
    """A → B (optional for C) → C, with an EXTRA edge A → C: what P2's tests build."""
    ch = Chains([_Toy("A"), _Toy("B", ["A"]), _Toy("C", ["B"])], extra={"C": ("A",)}, optional={"C": ("B",)},
                blocks={"first": ("A",), "second": ("B", "C")})
    assert ch.deps2["C"] == ("B", "A")
    assert ch.deps_in_chain(ch.ordered(["C", "A"]), "C") == ("A",)
    assert ch.deps_in_chain(ch.ordered(["A", "B", "C"]), "C") == ("B", "A")
    assert ch.ordered(["C", "B", "A"]) == ["A", "B", "C"]
    with pytest.raises(ValueError, match="first block"):
        ch.ordered(["B", "C"])
    with pytest.raises(TypeError):
        ch.chain({})
    toy = Chains([_Toy("A"), _Toy("B", ["A"])], rule=lambda o, end, reference: o["stages"])
    assert toy.chain({"stages": ["B", "A"]}) == ["A", "B"]


def test_chains_refuse_a_wrong_order():
    with pytest.raises(ValueError, match="before it"):
        Chains([_Toy("B", ["A"]), _Toy("A")])
    with pytest.raises(ValueError, match="disagree"):
        Chains([_Toy("A")], order=["A", "Z"])
