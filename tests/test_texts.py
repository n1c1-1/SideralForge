"""P16: the texts that come from the engine (astro_gpu/texts.py, client/locale/fr/engine.json).

Checks that: every automatic value's reason is a template plus arguments with a French; the stages give the same
English text as before; numbers get a decimal comma in French; every metric a stage writes has a catalogue entry; every
parameter label, help, tip and unit, and every stage title and description, has a French.
"""
from __future__ import annotations

import ast
import json
import os
import string
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astro_gpu import texts
from astro_gpu.core.params import Auto

STAGES = Path(texts.STAGES_DIR)
ENGINE_JSON = texts.LOCALE / "fr" / "engine.json"


# ---------------------------------------------------------------------------------------------------- Auto
def test_auto_keeps_the_old_calls():
    a = Auto(2.0, "because")
    assert (a.value, a.reason, a.template, a.args) == (2.0, "because", "because", None)
    assert Auto(2.0, reason="old keyword").reason == "old keyword"
    assert Auto(2.0, "because") == Auto(2.0, "because")
    # a fixed text is used as it is: braces in it are not fields
    assert Auto(1, "a {literal} brace").reason == "a {literal} brace"
    assert a.text("fr") == "because"                       # no French for a text nobody declared: the English


def test_auto_template_and_arguments():
    a = Auto(1.5, "FWHM {fwhm:.1f} px with {n} frames", {"fwhm": 4.25, "n": 80})
    assert a.reason == "FWHM 4.2 px with 80 frames"
    assert a.template == "FWHM {fwhm:.1f} px with {n} frames" and a.args == {"fwhm": 4.25, "n": 80}
    assert Auto(1.5, "{x}", {"x": 1}) != Auto(1.5, "{x}", {"x": 2})


def test_decimal_comma_in_french_only():
    t = "{x:.1f} px, {y} s, {n} frames, camera {c}"
    args = {"x": 1.5, "y": 2.0, "n": 1234, "c": "Sony v1.2"}
    assert texts.format_reason(t, args, "en") == "1.5 px, 2.0 s, 1234 frames, camera Sony v1.2"
    # no French for this template: the English words, but the numbers written the French way
    assert texts.format_reason(t, args, "fr") == "1,5 px, 2,0 s, 1234 frames, camera Sony v1.2"
    # with a French (the real ones)
    a = Auto(1.5, "FWHM {fwhm:.1f} px with {n} frames: under-sampled, enough frames to drizzle", {"fwhm": 4.5, "n": 80})
    assert "4.5" in a.text("en") and "4,5" in a.text("fr") and "4.5" not in a.text("fr")
    b = Auto(2.0, "star FWHM {fwhm:.1f} px x drizzle {scale}", {"fwhm": 3.04, "scale": 1.5})
    assert b.text("en") == "star FWHM 3.0 px x drizzle 1.5"
    assert b.text("fr") == "FWHM des étoiles 3,0 px × drizzle 1,5"
    assert texts.number(1.5, 1, "fr") == "1,5" and texts.number(1.5, 1, "en") == "1.5"
    assert texts.number(1.5, 1) == "1.5"


def test_numpy_numbers_and_text_arguments():
    f = texts._FORMATTERS[True]
    assert f.format("{a:.2f} {b:.1f} {c}", a=np.float32(1.25), b=np.float64(2.5), c=np.int64(3)) == "1,25 2,5 3"
    assert f.format("{c}", c="v1.2") == "v1.2"               # a text keeps its dots
    assert f.format("{n}", n=True) == "True"                 # a flag is not a number
    assert f.format("{n:d} {k:.0f}", n=12, k=7) == "12 7"
    assert f.format("{k:.1f}", k=3) == "3,0"                 # an int written as a decimal number


def test_a_bad_translation_falls_back_on_the_english(monkeypatch):
    monkeypatch.setitem(texts.catalogue("fr"), "{n} frames", "{nombre} images")      # a wrong field name
    assert texts.format_reason("{n} frames", {"n": 5}, "fr") == "5 frames"
    assert texts.format_reason("never translated {n}", {"n": 2.5}, "fr") == "never translated 2,5"


# ---------------------------------------------------------------------------------------------------- reasons
def test_every_reason_is_a_template_with_a_french():
    sites = texts.reason_sites()                  # raises for an f-string or a concatenation
    assert len(sites) >= 22
    seen = set()
    for site in sites:
        assert site.templates, site.where
        for t in site.templates:
            seen.add(t)
            fields = texts.template_fields(t)
            if fields:
                assert site.arg_names is not None, f"{site.where}: fields {sorted(fields)} but no arguments"
                assert fields == site.arg_names, f"{site.where}: fields {sorted(fields)} against arguments {site.arg_names}"
            else:
                assert not site.arg_names, f"{site.where}: arguments for a text without fields"
            assert texts.has_translation(t, "fr"), f"{site.where}: no French for {t!r}"
            fr = texts.translate(t, "fr")
            assert fr != t, f"{site.where}: the French of {t!r} is the English"
            # the French writes the same numbers the same way
            spec_en = {(n, s) for _, n, s, _ in string.Formatter().parse(t) if n}
            spec_fr = {(n, s) for _, n, s, _ in string.Formatter().parse(fr) if n}
            assert spec_en == spec_fr, f"{site.where}: fields of the French differ: {sorted(spec_en)} / {sorted(spec_fr)}"
    assert seen == set(texts.reason_templates())


def test_no_reason_is_an_f_string():
    for path in sorted(STAGES.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Auto":
                reason = node.args[1]
                assert not any(isinstance(n, (ast.JoinedStr, ast.BinOp)) for n in ast.walk(reason)), \
                    f"{path.name}:{node.lineno}: the reason of an Auto is an f-string"


def test_reason_sites_refuse_an_f_string(tmp_path, monkeypatch):
    (tmp_path / "bad.py").write_text('def auto():\n    return Auto(1, f"{n} frames")\n', encoding="utf-8")
    monkeypatch.setattr(texts, "STAGES_DIR", tmp_path)
    with pytest.raises(ValueError):
        texts.reason_sites()


# a stand-in for the context of a stage: just what the auto() functions read
class FakeCtx:
    def __init__(self, tmp_path, metrics=None, jsons=None, project=None):
        self.tmp, self.metrics, self.jsons, self.project = tmp_path, metrics or {}, jsons or {}, project

    def input(self, stage, name):
        return self.tmp / f"{stage}_{name}"

    def metric_of(self, stage, key, default=None):
        return self.metrics.get((stage, key), default)

    def load_json(self, stage, name):
        return self.jsons[(stage, name)]


def _stack_ctx(tmp_path, n, fwhm, low):
    """n frames of star size fwhm, the first ``low`` of them with a transparency of 0.2 (the others 1)."""
    transp = np.ones(n)
    transp[:low] = 0.2
    np.savez(tmp_path / "register_reg.npz", fwhm=np.full(n, fwhm), rms=np.full(n, 0.2), transp=transp)
    rng = np.random.default_rng(1)
    np.save(tmp_path / "calibrate_frame_stats.npy", rng.uniform(50, 100, (n, 4, 2)))
    return FakeCtx(tmp_path, metrics={("calibrate", "dark_frame_noise"): 2.0, ("calibrate", "cache_scale"): 1.0})


def _old_stack_reasons(n, f, frac, hb):
    """The reasons of stages/stack.py as its f-strings wrote them before P16."""
    out = {"half_block": f"{n} frames: blocks of {hb} give two half-stacks of about {n // 2} frames"
                         + ("" if hb == 20 else " (blocks of 20 need 80 frames)")}
    if f < 3.0 and n >= 150:
        out["scale"] = f"small stars (FWHM {f:.1f} px) and {n} frames"
    elif f < 5.5 and n >= 40:
        out["scale"] = f"FWHM {f:.1f} px with {n} frames: under-sampled, enough frames to drizzle"
    else:
        out["scale"] = f"FWHM {f:.1f} px / {n} frames: drizzling would add noise without detail"
    if frac > 0.5:
        out["min_transparency"] = (f"{100 * frac:.0f} % of frames are below 60 % transparency: only the "
                                   "worst (below 30 %) are excluded")
    elif frac > 0:
        out["min_transparency"] = (f"{100 * frac:.0f} % of frames lost more than 40 % of the signal "
                                   "(clouds or haze): excluded")
    else:
        out["min_transparency"] = "transparency is stable: no frame falls below 60 %"
    return out


@pytest.mark.parametrize("n, fwhm, low", [
    (100, 4.5, 0),           # blocks of 20, scale 1.5, stable
    (60, 4.5, 6),            # blocks of 15, 10 % lost
    (200, 2.5, 120),         # small stars, 60 % lost
    (30, 6.5, 3),            # few frames, big stars
])
def test_stack_gives_the_same_english_and_the_same_values(tmp_path, n, fwhm, low):
    from astro_gpu.stages.stack import Stack
    auto = Stack().auto(_stack_ctx(tmp_path, n, fwhm, low))
    want = _old_stack_reasons(n, fwhm, low / n, int(max(1, min(20, n // 4))))
    for name, text in want.items():
        assert auto[name].reason == text, name
        assert auto[name].text("en") == text
        assert auto[name].args is not None or auto[name].template == text
    assert auto["half_block"].value == int(max(1, min(20, n // 4)))
    assert auto["scale"].value == (2.0 if fwhm < 3.0 and n >= 150 else 1.5 if fwhm < 5.5 and n >= 40 else 1.0)
    assert auto["min_transparency"].value == (0.3 if low / n > 0.5 else 0.6)
    assert {"read_noise", "gain"} <= set(auto)
    assert auto["read_noise"].reason == "noise of a single dark frame"
    assert auto["gain"].reason == "sky level vs sky noise of the light frames, read noise removed"
    # and in French, with the decimal comma
    fr = {k: a.text("fr") for k, a in auto.items()}
    assert fr["half_block"].startswith(f"{n} images : des blocs de ")
    if fwhm < 5.5 and n >= 40 and fwhm == 4.5:
        assert fr["scale"] == f"FWHM 4,5 px avec {n} images : image sous-échantillonnée, assez d'images pour le drizzle"
    for k, a in auto.items():
        assert a.text("fr") != a.reason or not any(c.isalpha() for c in a.reason), k


def test_stack_without_darks_reason(tmp_path):
    from astro_gpu.stages.stack import Stack
    ctx = _stack_ctx(tmp_path, 100, 4.5, 0)
    ctx.metrics.clear()
    auto = Stack().auto(ctx)
    assert "read_noise" not in auto
    assert auto["gain"].reason == "sky level vs sky noise of the light frames (no darks: read noise ignored)"
    assert texts.has_translation(auto["gain"].template, "fr")


def test_the_other_stages(tmp_path):
    from astro_gpu.stages.astrometry import Astrometry
    from astro_gpu.stages.background import Background
    from astro_gpu.stages.calibrate import Calibrate
    from astro_gpu.stages.color import Color
    from astro_gpu.stages.denoise import Denoise
    from astro_gpu.stages.register import Register

    cores = os.cpu_count()
    for stage in (Calibrate(), Register()):
        a = stage.auto(None)["workers"]
        assert a.reason == f"{cores} CPU cores detected" and a.text("fr") == f"{cores} cœurs de processeur détectés"

    d = Denoise().auto(FakeCtx(tmp_path, jsons={("stack", "drz_meta.json"): {"width": 6000, "height": 4500}}))
    assert d["iterations"].reason == "27 Mpx stack" and d["iterations"].text("fr") == "empilement de 27 Mpx"

    c = Color().auto(FakeCtx(tmp_path, jsons={("stack", "drz_meta.json"): {"scale": 1.5}},
                             metrics={("register", "fwhm_px"): 3.04}))
    assert c["aperture"].reason == "star FWHM 3.0 px x drizzle 1.5"
    assert c["aperture"].text("fr") == "FWHM des étoiles 3,0 px × drizzle 1,5"
    assert c["aperture"].value == 4.0

    b = Background().auto(FakeCtx(tmp_path, metrics={("astrometry", "scale_arcsec"): 2.0}))
    assert b["large_scale_px"].value == 420.0
    assert b["large_scale_px"].reason == '14 arcmin at 2.00"/px'
    assert b["large_scale_px"].text("fr") == '14 arcmin à 2,00"/px'

    inv = {("inventory", "pointing_ra"): 270.9, ("inventory", "pointing_dec"): -23.5, ("inventory", "object"): "IC 1805",
           ("inventory", "pixel_um"): 3.76, ("inventory", "focal_mm"): 388.0, ("inventory", "camera"): "ILCE-7S"}
    tgt = SimpleNamespace(ra=None, dec=None, name="", focal_mm=None)
    project = SimpleNamespace(config=SimpleNamespace(target=tgt, name="heart"))
    a = Astrometry().auto(FakeCtx(tmp_path, metrics=inv, project=project))
    assert a["ra"].reason == "telescope pointing in the FITS headers"
    assert a["target"].reason == "project / FITS object name"
    assert a["pixel_um"].reason == "camera ILCE-7S" and a["pixel_um"].text("fr") == "appareil ILCE-7S"
    assert a["focal_mm"].reason == "recorded in the files"
    tgt.ra, tgt.dec, tgt.focal_mm = 10.0, 41.0, 250.0
    a = Astrometry().auto(FakeCtx(tmp_path, metrics=inv, project=project))
    assert a["ra"].reason == "project target" and a["ra"].text("fr") == "cible du projet"
    assert a["focal_mm"].reason == "entered when the project was created"


def test_manifest_keeps_the_english_text():
    """The runner stores ``a.reason`` (v1 manifests): still a plain English string."""
    a = Auto(3, "{cores} CPU cores detected", {"cores": 8})
    assert {k: x.reason for k, x in {"workers": a}.items()} == {"workers": "8 CPU cores detected"}
    json.dumps({"workers": a.reason})


# ---------------------------------------------------------------------------------------------------- metrics
#: f-string keys the stages write, and what they stand for
EXPANSIONS = {
    "n_{}": ["n_darks", "n_flats", "n_bias", "n_flat_darks"],
    "replaced_{}": ["replaced_G", "replaced_B"],
    "collapsed_{}": ["collapsed_green", "collapsed_blue"],
    "excess_{}": ["excess_green", "excess_blue"],
}
#: the functions that build the dictionaries of metrics the stages pass on (``ctx.metric(k, v)`` in a loop)
HELPERS = {"stars.py": ["forced_photometry", "star_qc", "local_psf_correction"], "far_halo.py": ["far_halo"],
           "star_shape.py": ["measure"]}


def _key(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
    return None


def _expand(key):
    if "{" not in key:
        return [key]
    assert key in EXPANSIONS, f"a metric key written with an f-string, {key!r}: say what it stands for in EXPANSIONS"
    return EXPANSIONS[key]


def written_metric_keys() -> dict[str, str]:
    """Every metric key the stages write: {key: where}, read in the source (``ctx.metric("key", ...)`` and the
    dictionaries of metrics that ``stars`` builds)."""
    out: dict[str, str] = {}
    for path in sorted(STAGES.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "metric"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "ctx" and node.args):
                k = _key(node.args[0])
                for key in (_expand(k) if k else []):                 # a name (k_): the helpers below
                    out.setdefault(key, f"{path.name}:{node.lineno}")
        for fn in HELPERS.get(path.name, []):
            func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn)
            for sub in ast.walk(func):
                keys = []
                if isinstance(sub, ast.Dict):
                    keys = [_key(k) for k in sub.keys if k is not None]
                elif isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store):
                    keys = [_key(sub.slice)]
                for k in keys:
                    if k and not k.startswith("_"):
                        for key in _expand(k):
                            out.setdefault(key, f"{path.name}:{fn}")
    return out


def test_every_metric_the_stages_write_has_a_catalogue_entry():
    written = written_metric_keys()
    assert len(written) > 100
    missing = sorted(k for k in written if k not in texts.METRICS)
    assert not missing, f"metrics with no catalogue entry (astro_gpu/texts.py): {missing}"
    # and no entry for a key nobody writes (a typo)
    dead = sorted(k for k in texts.METRICS if k not in written)
    assert not dead, f"catalogue entries no stage writes: {dead}"


def test_every_metric_of_the_real_manifests_has_a_catalogue_entry():
    manifests = sorted(Path("F:/astro_essais/projets").glob("*/work/*/manifest.json"))
    if not manifests:
        pytest.skip("the test projects of F:/astro_essais are not here")
    keys = set()
    for f in manifests:
        keys |= set(json.loads(f.read_text(encoding="utf-8")).get("metrics") or {})
    assert len(keys) > 100
    assert not sorted(k for k in keys if k not in texts.METRICS)


def test_metric_entries_are_complete_and_in_french():
    for key, m in texts.METRICS.items():
        assert m.key == key
        assert m.label and m.label[0].isupper() and len(m.label) <= 40, key
        assert m.meaning.endswith("."), key
        assert m.sep
        if m.fmt:
            format(3 if m.fmt == "d" else 1.5, m.fmt)                 # a valid number format
        for t in (m.label, m.meaning, m.unit):
            assert texts.has_translation(t, "fr"), f"{key}: no French for {t!r}"
        assert texts.metric_label(key, "fr") and texts.metric_meaning(key, "fr") != m.meaning


def test_format_metric():
    f = texts.format_metric
    assert f("scale_arcsec", 1.3493, "en") == "1.349 arcsec/px" and f("scale_arcsec", 1.3493, "fr") == "1,349 arcsec/px"
    assert f("field_deg", [3.47, 2.319], "en") == "3.47 × 2.32 deg"
    assert f("field_deg", [3.47, 2.319], "fr") == "3,47 × 2,32°"
    assert f("n_lights", 15, "fr") == "15" and f("stars_used", 4742, "en") == "4742"
    assert f("integration_h", 0.75, "fr") == "0,75 h" and f("exposure_s", 180.0, "fr") == "180 s"
    assert f("pixel_um", 3.75999999046326, "fr") == "3,76 µm"
    assert f("cache_gb", 0.8, "en") == "0.8 GB" and f("cache_gb", 0.8, "fr") == "0,8 Go"
    assert f("parity_flipped", True, "en") == "Yes" and f("parity_flipped", False, "fr") == "Non"
    assert f("focal_mm", None, "fr") == "–"
    assert f("coverage", 1.0, "en") == "100.0 %" and f("coverage", 0.955, "fr") == "95,5 %"
    assert f("crop", [1, 3731, 2, 2544], "fr") == "1, 3731, 2, 2544 px"
    assert f("local_psf_stars_per_cell", [24, 59], "fr") == "24–59"
    assert f("noise_reduction", 13.84, "fr") == "13,84 ×"
    assert f("far_halo_profile", {"78": 3.0, "107": 2.2}, "fr") == "2 valeurs"
    assert f("matrix", [[2.1, -0.8, 0.38], [0.5, 0.3, 0.2]], "fr") == "2,100 / -0,800 / 0,380 ; 0,500 / 0,300 / 0,200"
    assert f("no_such_metric", 2.5, "en") == "2.5" and f("no_such_metric", 2.5, "fr") == "2,5"
    assert texts.metric_label("no_such_metric") == "No such metric"
    assert texts.metric_unit("fwhm_px", "fr") == "px" and texts.metric_unit("field_deg", "fr") == "°"


def test_fixed_words_of_the_metrics():
    f = texts.format_metric
    assert f("survey", "nsns", "en") == "NSNS" and f("survey", "none", "fr") == "aucun"
    assert f("star_flat_photometry", "aperture", "fr") == "grandes ouvertures"
    assert f("response_source", "fit", "fr") == "ajustées sur ce champ"
    assert f("response_source", "camera profile (heart_a7s)", "fr") == "profil de l'appareil (heart_a7s)"
    assert f("star_profile", "focused, FWHM 2.50 px (the measured stars are defocused rings)", "fr").startswith(
        "nette, FWHM 2,50 px")
    assert f("angular_shape", "too few stars", "fr") == "trop peu d'étoiles"
    assert f("far_halo", "fewer than 3 bright stars", "fr") == "moins de 3 étoiles brillantes"
    assert f("ha_fit", "fitted", "en") == "fitted"
    kept = ("camera HII signature imposed (the survey correlates with other structures in this field); the rejection on "
            "all channels left 5 % of the H-alpha amplitude of the whole field: objects rejected on green and blue only")
    assert "le rejet sur tous les canaux a laissé 5 %" in f("ha_fit", kept, "fr")
    assert f("camera", "ZWO ASI2600MC Pro", "fr") == "ZWO ASI2600MC Pro"           # a name is never translated
    for v in texts.VALUE_TEXTS.values():
        assert texts.has_translation(v, "fr"), v


# ---------------------------------------------------------------------------------------------------- parameters
def test_every_parameter_text_has_a_french():
    cov = texts.coverage("fr")
    for kind, missing in cov.items():
        assert not missing, f"{kind}: no French for {[m[:70] for m in missing]}"


def test_every_help_text_has_a_french_and_is_not_the_english():
    n_help = n_tip = 0
    for stage in texts.stage_names():
        assert texts.stage_title(stage, "fr") and texts.stage_description(stage, "fr") != texts.stage_description(stage)
        for name in texts.param_names(stage):
            h, h_fr = texts.param_help(stage, name), texts.param_help(stage, name, "fr")
            assert h and h_fr and h_fr != h, f"{stage}.{name}: help"
            n_help += 1
            t, t_fr = texts.param_tip(stage, name), texts.param_tip(stage, name, "fr")
            if t:
                assert t_fr != t, f"{stage}.{name}: tip"
                n_tip += 1
            assert texts.param_label(stage, name, "fr"), f"{stage}.{name}: label"
    assert n_help > 100 and n_tip > 30


def test_parameter_texts_read_well():
    assert texts.param_help("stack", "scale", "fr").startswith("Résolution de sortie")
    assert "1,5 pour la plupart" in texts.param_tip("stack", "scale", "fr")           # decimal commas inside the text
    assert texts.param_help("stack", "scale", "en").startswith("Output resolution")
    assert texts.param_unit("astrometry", "ra", "fr") == "°"
    assert texts.param_texts("stack", "scale", "fr") == {
        "label": texts.param_label("stack", "scale", "fr"), "help": texts.param_help("stack", "scale", "fr"),
        "tip": texts.param_tip("stack", "scale", "fr"), "unit": "×"}
    assert texts.param_texts("stack", "pixfrac", "en")["unit"] == ""
    assert texts.stage_title("stack", "fr") == "Empilement (drizzle)"
    assert texts.choice_label("nsns", "fr") == "NSNS" and texts.choice_label("aux", "en") == "Second GPU"
    assert texts.choice_label("aux", "fr") == "GPU secondaire"
    assert texts.choice_label("something else", "fr") == "something else"
    # the vocabulary of the glossary
    fr = " ".join(texts.catalogue("fr")[k] for k in texts.catalogue("fr") if "frames" in k.lower())
    assert "frames" not in fr


# ---------------------------------------------------------------------------------------------------- the catalogue
def test_engine_json_is_clean():
    raw = ENGINE_JSON.read_text(encoding="utf-8")
    pairs = json.loads(raw, object_pairs_hook=lambda p: p)
    keys = [k for k, _ in pairs]
    assert len(keys) == len(set(keys)), "a key appears twice"
    assert len(keys) > 400
    for k, v in pairs:
        assert isinstance(v, str) and v.strip() == v and v, k
        f_en = {n for _, n, s, _ in string.Formatter().parse(k) if n}
        f_fr = {n for _, n, s, _ in string.Formatter().parse(v) if n}
        assert f_en == f_fr, f"fields differ in {k!r}"
        assert "  " not in v, k
    # each entry is something the engine says: a template, a metric text, a stage text or a value
    wanted = set(texts.reason_templates())
    wanted |= {t for m in texts.METRICS.values() for t in (m.label, m.meaning, m.unit)}
    wanted |= set(texts.VALUE_TEXTS.values()) | {p[1] for p in texts.VALUE_PATTERNS} | set(texts.UI_TEXTS)
    wanted |= set(texts.CHOICE_TEXTS.values())
    for s in texts._stages().values():
        wanted |= {s["title"], s["description"]}
        for d in s["params"].values():
            wanted |= {d.get(f) for f in ("label", "help", "tip", "unit")}
    assert [k for k in keys if k not in wanted] == [], "entries that no engine text uses (a typo in the English)"


def test_the_client_dictionary_serves_the_same_french():
    from astro_gpu.client import i18n
    try:
        i18n.set_language("fr")
        assert i18n._("camera {camera}") == "appareil {camera}"
        assert i18n._(texts.METRICS["fwhm_px"].meaning) == texts.metric_meaning("fwhm_px", "fr")
        assert i18n._(texts._stages()["stack"]["params"]["scale"]["help"]) == texts.param_help("stack", "scale", "fr")
    finally:
        i18n.set_language("en")


def test_module_stays_light():
    """texts.py is used by the API and the tests: it must not need Qt or the client package."""
    import subprocess
    import sys
    code = "import sys, astro_gpu.texts; assert not [m for m in sys.modules if m.startswith(('PySide', 'PyQt', 'torch'))]"
    subprocess.run([sys.executable, "-c", code], check=True)
