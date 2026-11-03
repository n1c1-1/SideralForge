"""Workflows v2: options → chain (DESIGN §2.1), dependencies in a chain (§2.2), the option rules of the mockup."""
from __future__ import annotations

import re

import pytest

from astro_gpu.core import workflow as wfl
from astro_gpu.core.workflow import TEMPLATES, Chains, OptionRefused, chain, chain_stages, chains, set_option

BASE = ["inventory", "calibrate", "register", "stack"]
TABLE = {      # §2.1: the stages after stack, without the reference image
    "Classic stack": ["astrometry", "catalogs", "gradient", "color", "render_simple"],
    "Classic + denoise": ["astrometry", "catalogs", "gradient", "color", "denoise", "render_simple"],
    "Full": ["astrometry", "catalogs", "gradient", "color", "denoise", "optics", "stars", "starless", "background",
             "render_prep"],
    "Full + neural stars": ["astrometry", "catalogs", "gradient", "color", "denoise", "optics", "stars", "star_net",
                            "starless", "background", "render_prep_learned"],
}
CLASSIC, CLASSIC_DN, FULL, NEURAL = (TEMPLATES[k] for k in TABLE)


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


def test_earlier_neural_layer_omits_star_net():
    c = chain(NEURAL, learned_layer=True)
    assert "star_net" not in c and c[-1] == "render_prep_learned"
    assert c == [s for s in BASE + TABLE["Full + neural stars"] if s != "star_net"]
    assert chain(FULL, learned_layer=True) == BASE + TABLE["Full"]          # psf: the layer changes nothing here
    assert chains().deps_in_chain(c, "starless") == ("stars", "optics", "denoise", "stack", "color", "catalogs",
                                                     "gradient")


def test_gradient_method():
    for o in (CLASSIC, FULL):
        assert wfl.method_of(o, "gradient") == "polynomial"
        assert wfl.implied(o, "gradient") == {"method": "polynomial"}
        assert "surveys" not in chain(o)
    # the H-alpha anchored background of earlier versions is read as the method that replaces it
    assert wfl.check_options({**FULL, "gradient": "halpha"}) == FULL


def test_implied_star_net():
    assert wfl.implied(NEURAL, "star_net") == {"enabled": True}
    assert wfl.implied(FULL, "star_net") == {}
    assert wfl.implied(NEURAL, "stack") == {}


@pytest.mark.parametrize("options,end", [
    ({"denoise": "nlbayes"}, None),                                   # ◇ methods
    ({"denoise": "n2n", "stars": "psf", "gradient": "polynomial", "deconvolution": "rl"}, None),
    ({"stars": "learned"}, None),
    ({"gradient": None}, None),
    ({"gradient": "mars"}, None),
    ({"stars": "psf", "gradient": "polynomial"}, None),               # Stars needs Denoise
    ({"denoise": "n2n", "colour": True}, None),                       # unknown option
    (CLASSIC, "astrometry"),                                          # not an exit
])
def test_invalid_options_raise(options, end):
    with pytest.raises(ValueError):
        chain(options, end)
    with pytest.raises(ValueError):
        chain_stages(options, end)


def test_missing_options_take_the_classic_defaults():
    assert wfl.check_options({}) == CLASSIC
    assert list(wfl.check_options({"deconvolution": None, "denoise": "n2n"})) == list(wfl.OPTION_VALUES)


# ---------------------------------------------------------------------------------------------- the option rules

# the mockup's with_tool / without_tool (model.py:279-332), in the v2 options
TOOL_CASES = [
    # with Stars: Denoise comes with it
    (CLASSIC, "stars", "psf", {**CLASSIC, "denoise": "n2n", "stars": "psf"}, [wfl.NOTE_DENOISE_ADDED]),
    # with the neural removal: Stars, then Denoise, come with it
    (CLASSIC, "stars", "psf_neural", {**CLASSIC, "denoise": "n2n", "stars": "psf_neural"},
     [wfl.NOTE_STARS_ADDED_NEURAL, wfl.NOTE_DENOISE_ADDED]),
    (FULL, "stars", "psf_neural", NEURAL, []),
    # with Denoise
    (CLASSIC, "denoise", "n2n", CLASSIC_DN, []),
    # without Stars: the neural removal goes
    (NEURAL, "stars", None, CLASSIC_DN, [wfl.NOTE_NEURAL_REMOVED]),
    (FULL, "stars", None, CLASSIC_DN, []),
    # without the neural removal
    (NEURAL, "stars", "psf", FULL, []),
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
    (NEURAL, "denoise", None, wfl.REFUSE_DENOISE),
    (FULL, "gradient", None, wfl.REFUSE_GRADIENT),         # the Gradient cannot be removed, only its method changes
    (CLASSIC, "gradient", None, wfl.REFUSE_GRADIENT),
])
def test_option_rules_refuse(before, key, value, why):
    with pytest.raises(OptionRefused, match="^" + re.escape(why) + "$"):
        set_option(before, key, value)


def test_no_block_after_an_early_output():
    with pytest.raises(OptionRefused):
        set_option(CLASSIC, "denoise", "n2n", end="stacking")


def test_set_option_rejects_invalid_values():
    for key, value in (("denoise", "nlbayes"), ("stars", "starnet"), ("deconvolution", "rl"), ("colour", 1)):
        with pytest.raises(ValueError) as e:
            set_option(FULL, key, value)
        assert not isinstance(e.value, OptionRefused)


def test_set_option_does_not_change_its_input():
    before = dict(NEURAL)
    set_option(NEURAL, "stars", None)
    assert NEURAL == before


# ---------------------------------------------------------------------------------------------- dependencies

def test_optional_and_extra_dependencies():
    ch = chains()
    full, neural = chain(FULL), chain(NEURAL)
    assert "star_net" not in ch.deps_in_chain(full, "starless")
    assert "star_net" in ch.deps_in_chain(neural, "starless")                 # the EXTRA edge, when present
    assert set(ch.deps_in_chain(chain(CLASSIC), "color")) == {"stack", "catalogs", "register", "gradient"}
    assert ch.deps_in_chain(chain(CLASSIC_DN), "denoise") == ("stack", "gradient")
    assert ch.deps_in_chain(chain(CLASSIC_DN), "render_simple") == ("stack", "color", "denoise", "gradient")
    assert ch.deps_in_chain(chain(CLASSIC, "stacking"), "render_simple") == ("stack",)
    assert ch.ancestors(chain(CLASSIC, "stacking"), "render_simple") == BASE
    assert ch.downstream(full, "denoise") == ["optics", "stars", "starless", "background", "render_prep"]


def test_missing_required_dependency_names_the_block():
    ch = chains()
    stages = [s for s in chain(FULL) if s != "denoise"]
    with pytest.raises(ValueError, match="Denoise block"):
        ch.ordered(stages)
    with pytest.raises(ValueError, match="Astrometry block"):
        ch.deps_in_chain(BASE + ["color"], "color")


def test_v2_stage_graph():
    ch = chains()
    assert ch.order == wfl.ORDER and set(ch.stages) == set(wfl.ORDER)
    new_stages = {"gradient": ("stack",),                                  # the dependencies of the new stages
                  "render_simple": ("stack", "color", "denoise", "gradient")}
    for name, deps in new_stages.items():
        assert set(deps) <= set(ch.deps2[name])
    for name in wfl.ORDER:                                     # every stage sits in exactly one block
        assert sum(name in ss for ss in wfl.BLOCKS.values()) == 1


def test_blocks():
    ch = chains()
    c = chain(NEURAL, reference=True)
    blocks = ch.blocks_in(c)
    assert [b for b, _ in blocks] == ["calibration", "registration", "selection", "stacking", "astrometry",
                                      "gradient", "color", "denoise", "stars", "output"]
    assert dict(blocks)["stars"] == ("optics", "stars", "star_net", "starless", "background")
    assert ch.run_to("selection", c) == ["register"]
    assert ch.run_to("gradient", c) == ["gradient"]
    assert ch.run_to("stars", c) == ["optics", "stars", "star_net", "starless", "background"]
    assert ch.key_stages("astrometry", c) == ["astrometry", "reference", "catalogs"]   # the survey follows Gradient
    assert ch.block_of("star_net") == "stars" and ch.block_of("register") == "registration"
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
    toy = Chains([_Toy("A"), _Toy("B", ["A"])], rule=lambda o, end, reference, learned: o["stages"])
    assert toy.chain({"stages": ["B", "A"]}) == ["A", "B"]


def test_chains_refuse_a_wrong_order():
    with pytest.raises(ValueError, match="before it"):
        Chains([_Toy("B", ["A"]), _Toy("A")])
    with pytest.raises(ValueError, match="disagree"):
        Chains([_Toy("A")], order=["A", "Z"])
