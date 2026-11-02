"""Post's tools (docs/pilot/POST_TOOLS.md): the Curves tool and the filters added after the PixInsight / Siril inventory.

Each method against its reference: the interpolations against scipy, Photoshop's presets against the .acv files
Photoshop ships (when installed), GHS, asinh, the histogram transformation and SCNR against siril-cli (when installed),
CLAHE against OpenCV, the starlet transform against its definition. Every filter is the identity at its defaults, then
has a known effect on synthetic data. The Curves editor in the page: points added, moved and deleted with the mouse, a
draft while a point moves and the exact image on release (one undo step), the live histogram, the presets (built-in, the
user's own, Photoshop files), the French. The real project (slow): screenshots EN and FR and the time of a filter drag.

ASTRO_GPU_POST_TOOLS_SHOTS names the folder of the screenshots (default: %TEMP%/post_tools_shots).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from test_post import (  # noqa: E402, F401 - the fixtures and helpers of Post's tests
    REAL,
    USER_BEFORE,
    USER_SETTINGS,
    RealSetup,
    Setup,
    _fingerprint,
    _settings,
    named,
    post,
    preps,
    qapp,
    reference,
    shown,
)

from astro_gpu.client import i18n, postengine  # noqa: E402
from astro_gpu.client.curveseditor import CurvesEditor, FunctionGraph  # noqa: E402
from astro_gpu.client.postengine import FILTERS, Composer, new_filter_layer, new_workflow_layer, normalize  # noqa: E402
from astro_gpu.client.postfilters import colour as pcol  # noqa: E402
from astro_gpu.client.postfilters import curves as pc  # noqa: E402
from astro_gpu.client.postfilters import detail as pd  # noqa: E402
from astro_gpu.client.postfilters import tone as pt  # noqa: E402

SIRIL = Path("C:/Program Files/Siril/bin/siril-cli.exe")
PS_CURVES = Path("C:/Program Files/Adobe/Adobe Photoshop 2026/Presets/Curves")
SHOTS = Path(os.environ.get("ASTRO_GPU_POST_TOOLS_SHOTS") or
             Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "post_tools_shots")
NEW = ("ghs", "asinh", "histogram", "curves", "clahe", "ml_sharpen", "ml_denoise", "scnr")
USER_PRESETS = USER_SETTINGS.with_name("post_curves_presets.json")      # where the user's own presets would be
PRESETS_BEFORE = _fingerprint(USER_PRESETS)


class Ctx:
    def __init__(self, level=0, draft=False):
        self.level, self.f, self.draft = level, 2 ** level, draft


def _img(h=96, w=128, seed=0):
    rng = np.random.default_rng(seed)
    a = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 4)
    return np.clip((a - a.min()) / (a.max() - a.min()) * 0.9 + 0.03, 0, 1).astype(np.float32)


def _params(key, **kw):
    p = {q["key"]: q["default"] for q in FILTERS[key]["params"]}
    p.update(kw)
    return p


# ---------------------------------------------------------------------------------------------- neutral defaults
def test_every_new_filter_is_the_identity_at_its_defaults_and_does_something_when_moved():
    x = _img()
    for key in NEW:
        f = FILTERS[key]
        for level, draft in ((0, False), (1, False), (2, True)):
            got = f["fn"](x, _params(key), Ctx(level, draft))
            assert got is x or np.array_equal(got, x), (key, level, draft)
    moved = {"ghs": dict(stretch=2.0, symmetry=0.2), "asinh": dict(stretch=30.0), "histogram": dict(midtones=0.3),
             "curves": dict(curves=pc.preset_curves("Medium Contrast")), "clahe": dict(amount=1.0),
             "ml_sharpen": dict(l2=1.0, l3=1.0), "ml_denoise": dict(l1=3.0, l2=3.0, target="RGB"),
             "scnr": dict(strength=1.0)}
    for key, kw in moved.items():
        got = FILTERS[key]["fn"](x, _params(key, **kw), Ctx())
        assert got.shape == x.shape and got.dtype == np.float32 and np.isfinite(got).all(), key
        assert np.abs(got - x).max() * 255 > 1, key


# ---------------------------------------------------------------------------------------------- the curves
def test_the_interpolations_are_akimas_the_natural_cubic_and_linear():
    from scipy.interpolate import Akima1DInterpolator, CubicSpline
    pts = [[0, 0], [0.1, 0.05], [0.3, 0.42], [0.55, 0.5], [0.8, 0.9], [1, 1]]
    xs = np.linspace(0, 1, 2001)
    xp, yp = np.array(pts).T
    ak = pc.Spline(pts, "Akima")(xs)
    ref = np.clip(Akima1DInterpolator(xp, yp)(xs), 0, 1)
    assert np.abs(ak - ref).max() < 1e-9                                   # Akima 1970, as scipy's (and Siril's)
    nat = pc.Spline(pts, "Cubic spline")(xs)
    assert np.abs(nat - np.clip(CubicSpline(xp, yp, bc_type="natural")(xs), 0, 1)).max() < 1e-9
    assert np.abs(pc.Spline(pts, "Linear")(xs) - np.interp(xs, xp, yp)).max() < 1e-12
    for interp in pc.INTERPS:                                              # through every point; collinear: a line
        sp = pc.Spline(pts, interp)
        assert np.allclose(sp(xp), yp, atol=1e-12)
        assert np.allclose(pc.Spline([[0, 0], [0.4, 0.4], [1, 1]], interp)(xs), xs, atol=1e-12)
    two = pc.Spline([[0, 0.1], [1, 0.9]], "Akima")
    assert np.allclose(two(xs), 0.1 + 0.8 * xs)
    # outside its end points a curve is flat (Photoshop's moved black and white points)
    sp = pc.Spline([[0.2, 0.0], [0.8, 1.0]], "Linear")
    assert sp(0.1) == 0 and sp(0.95) == 1 and abs(sp(0.5) - 0.5) < 1e-12
    # the image: the table of a draft within a tenth of an 8-bit level of the exact curve
    x = _img(64, 80)
    s = pc.Spline(pts, "Akima")
    exact = s.map(x)
    assert np.abs(exact - np.clip(Akima1DInterpolator(xp, yp)(x.astype(np.float64)), 0, 1)).max() < 2e-6
    assert np.abs(s.map(x, draft=True) - exact).max() * 255 < 0.15


def test_the_channels_act_on_what_they_name():
    x = _img(48, 64, 3)
    lift = {"pts": [[0, 0], [0.5, 0.65], [1, 1]], "interp": "Akima"}
    s = pc.Spline(lift["pts"], "Akima")
    rgb = pc.apply(x, {"RGB/K": lift})
    assert np.abs(rgb - s.map(x)).max() < 1e-6
    r = pc.apply(x, {"R": lift})
    assert np.abs(r[..., 0] - s.map(np.ascontiguousarray(x[..., 0]))).max() < 1e-6
    assert np.array_equal(r[..., 1:], x[..., 1:])
    # L*: the lightness follows the curve, a* and b* stay (colours kept)
    lab0, lab1 = pcol.to_lab(x), pcol.to_lab(pc.apply(x, {"L*": lift}))
    inside = (pcol.from_lab(lab1) > 0.002).all(-1) & (pcol.from_lab(lab1) < 0.998).all(-1)
    assert np.abs(lab1[..., 0] - 100 * s(lab0[..., 0] / 100))[inside].max() < 0.1
    assert np.abs(lab1[..., 1:] - lab0[..., 1:])[inside].max() < 0.2
    # S: the HSV saturation follows the curve, the lightness L* is kept; a grey pixel stays grey
    grey = np.full((4, 4, 3), 0.4, np.float32)
    assert np.abs(pc.apply(grey, {"S": lift}) - grey).max() < 1e-3
    y = pc.apply(x, {"S": lift})
    S0 = (x.max(-1) - x.min(-1)) / x.max(-1)
    S1 = (y.max(-1) - y.min(-1)) / np.maximum(y.max(-1), 1e-6)
    assert (S1 > S0).mean() > 0.95
    assert np.abs(pcol.lightness(y) - pcol.lightness(x)).max() < 0.5
    # c*: chroma scaled, hue and lightness kept
    c = pc.apply(x, {"c*": lift})
    lab2 = pcol.to_lab(c)
    ok = (c > 0.002).all(-1) & (c < 0.998).all(-1)                         # in gamut (else clipped)
    assert ok.mean() > 0.5 and np.abs(lab2[..., 0] - lab0[..., 0])[ok].max() < 0.3
    h0, h2 = np.arctan2(lab0[..., 2], lab0[..., 1]), np.arctan2(lab2[..., 2], lab2[..., 1])
    assert np.abs(np.angle(np.exp(1j * (h2 - h0))))[ok].max() < 0.03
    assert (np.hypot(lab2[..., 1], lab2[..., 2]) >= np.hypot(lab0[..., 1], lab0[..., 2]) - 0.2)[ok].all()
    # H: a hue curve moved by 30 degrees turns every hue by 30 degrees
    turn = {"pts": [[0, 30 / 360], [330 / 360, 1.0], [1, 1]], "interp": "Linear"}
    sat = np.zeros((1, 3, 3), np.float32)
    sat[0] = [[0.55, 0.42, 0.42], [0.42, 0.55, 0.44], [0.42, 0.44, 0.56]]          # turned, still in gamut
    lab_a, lab_b = pcol.to_lab(sat), pcol.to_lab(pc.apply(sat, {"H": turn}))
    dh = np.degrees(np.arctan2(lab_b[..., 2], lab_b[..., 1]) - np.arctan2(lab_a[..., 2], lab_a[..., 1])) % 360
    assert np.abs(dh - 30).max() < 2.0
    # the order is Siril's: RGB/K first, then the channels, then L*, S, c*, H
    assert pc.CHANNELS == ["RGB/K", "R", "G", "B", "L*", "S", "c*", "H"]


def test_the_presets_are_the_pro_tools_ones():
    names = [p["name"] for p in pc.builtin_presets()]
    assert names[0] == "Linear" and pc.preset_curves("Linear") == {}
    for n in ("Medium Contrast", "Strong Contrast", "Lighter", "Darker", "Increase Contrast", "Linear Contrast",
              "Negative", "Color Negative", "Cross Process"):
        assert n in names
    astro = [p for p in pc.builtin_presets() if p["source"] not in ("Photoshop",) and p["name"] != "Linear"]
    assert len(astro) == 4 and all(p["source"] for p in astro)
    # Photoshop's: the very points of the .acv files Photoshop ships
    if PS_CURVES.is_dir():
        for n in ("Medium Contrast", "Strong Contrast", "Lighter", "Darker", "Increase Contrast", "Linear Contrast",
                  "Negative", "Color Negative", "Cross Process"):
            got = pc.read_acv((PS_CURVES / f"{n} (RGB).acv").read_bytes())
            want = pc.preset_curves(n)
            assert set(got) == set(want), n
            for ch in got:
                assert np.allclose(got[ch]["pts"], want[ch]["pts"], atol=1e-12), (n, ch)
    # what they do: contrast presets darken the shadows and lift the highlights, Lighter lifts, Negative inverts
    x = np.linspace(0, 1, 256, dtype=np.float32)[None, :, None].repeat(3, -1)
    mc = pc.apply(x, pc.preset_curves("Medium Contrast"))[0, :, 0]
    assert mc[60] < x[0, 60, 0] and mc[200] > x[0, 200, 0]
    assert (pc.apply(x, pc.preset_curves("Lighter"))[0, 1:-1, 0] > x[0, 1:-1, 0]).all()
    assert np.allclose(pc.apply(x, pc.preset_curves("Negative"))[0, :, 0], 1 - x[0, :, 0], atol=1e-6)
    lift = pc.apply(x, pc.preset_curves("Lift the midtones"))[0, :, 0]
    assert abs(lift[89] - 0.5) < 0.02                                     # MTF(0.35, 0.35) = 0.5
    prot = pc.apply(x, pc.preset_curves("Midtone contrast, highlights protected"))[0, :, 0]
    assert np.all(np.diff(prot) >= -1e-6) and prot[-1] == pytest.approx(1.0)


def test_the_users_presets_and_photoshop_files(tmp_path):
    f = tmp_path / "presets.json"
    assert pc.user_presets(f) == []
    curves = {"RGB/K": {"pts": [[0, 0], [0.3, 0.4], [1, 1]], "interp": "Akima"},
              "B": {"pts": [[0, 0.05], [1, 0.95]], "interp": "Linear"}}
    pc.save_user_preset("Heart, soft", curves, f)
    pc.save_user_preset("Other", {}, f)
    got = pc.user_presets(f)
    assert [p["name"] for p in got] == ["Heart, soft", "Other"] and got[0]["curves"] == curves
    pc.save_user_preset("Heart, soft", {"G": curves["B"]}, f)                  # the same name replaces it
    assert [p["name"] for p in pc.user_presets(f)] == ["Other", "Heart, soft"]
    pc.delete_user_preset("Other", f)
    assert [p["name"] for p in pc.user_presets(f)] == ["Heart, soft"]
    # the default file is beside the settings (here the test's temporary settings), never the user's
    assert pc.presets_file().parent == Path(os.environ["ASTRO_GPU_SETTINGS"]).parent
    # Photoshop .acv: written and read back (to the 8-bit point); .json: exact
    pc.write_preset_file(tmp_path / "c.acv", curves)
    back = pc.read_preset_file(tmp_path / "c.acv")
    assert set(back) == {"RGB/K", "B"}
    assert np.allclose(back["RGB/K"]["pts"], [[0, 0], [77 / 255, 102 / 255], [1, 1]], atol=1e-9)
    pc.write_preset_file(tmp_path / "c.json", curves, "x")
    assert pc.read_preset_file(tmp_path / "c.json") == curves
    with pytest.raises(ValueError):
        pc.read_acv(b"\x00\x04\x00\x01\x00\x09")
    # damaged curves are made safe: sorted, in range, x apart, 2 to 100 points
    junk = {"RGB/K": {"pts": [[1.5, 2], [0.5, 0.2], [0.5004, 0.9], ["x", 1], [-1, 0]], "interp": "Wiggly"},
            "Q": {"pts": [[0, 0], [1, 1]]}, "R": {"pts": [[0.3, 0.3]]}}
    assert pc.clean_curves(junk) == {"RGB/K": {"pts": [[0, 0], [0.5, 0.2], [1, 1]], "interp": "Akima"}}


# ---------------------------------------------------------------------------------------------- the stretches
def _ramp(n=1024):
    x = np.linspace(0, 1, n)
    rgb = np.stack([x, x ** 0.7 * 0.8, 0.5 + 0.45 * np.sin(x * 9)], -1).astype(np.float32)
    return np.ascontiguousarray(np.repeat(rgb[None], 8, 0))


def test_the_generalised_hyperbolic_stretch_follows_its_equations():
    xs = np.linspace(0, 1, 20001)
    for b in (-3.0, -1.0, -0.5, 0.0, 0.5, 2.0, 8.0):
        for sp, lp, hp in ((0.1, 0.0, 1.0), (0.3, 0.1, 0.8)):
            p = dict(stretch=2.0, local=b, symmetry=sp, shadows=lp, highlights=hp)
            y = pt.ghs_curve(p, xs)
            assert y[0] == pytest.approx(0, abs=1e-9) and y[-1] == pytest.approx(1, abs=1e-9)
            assert np.all(np.diff(y) >= -1e-12)                             # monotone
            d = np.gradient(y, xs)
            assert abs(xs[np.argmax(d)] - sp) < 2e-3                        # the most contrast at SP
            if lp > 0:
                assert np.ptp(d[(xs > 0.005) & (xs < lp - 0.005)]) < 1e-6 * d.max() + 1e-6   # linear below LP
            if hp < 1:
                assert np.ptp(d[(xs > hp + 0.005) & (xs < 0.995)]) < 1e-6 * d.max() + 1e-6   # and above HP
            # its slope: D (1 + b D |x - SP|)^(-(1+b)/b) near SP (the published derivative), before normalisation
            D = math.expm1(2.0)
            k = (xs > sp + 0.01) & (xs < min(hp, 1) - 0.01)
            yk = xs[k] - sp
            if b > 0:
                want = (1 + b * D * yk) ** (-(1 + b) / b)
            elif b == 0:
                want = np.exp(-D * yk)
            else:
                want = (1 - b * D * yk) ** (1 / b)
            ratio = d[k] / want
            assert np.ptp(ratio) < 2e-3 * ratio.mean(), b
    # continuous in b through 0 and -1; no stretch at D = 0; the black point first
    a, z, c = (pt.ghs_curve(dict(stretch=2.0, local=b, symmetry=0.2), xs) for b in (-1e-4, 0.0, 1e-4))
    assert np.abs(a - z).max() < 1e-3 and np.abs(c - z).max() < 1e-3
    a, z = (pt.ghs_curve(dict(stretch=2.0, local=b, symmetry=0.2), xs) for b in (-1.0001, -1.0))
    assert np.abs(a - z).max() < 1e-3
    assert np.array_equal(pt.ghs_curve(dict(stretch=0.0, symmetry=0.3), xs), xs)
    bp = pt.ghs_curve(dict(stretch=0.0, black=0.1), xs)
    assert np.allclose(bp, np.clip((xs - 0.1) / 0.9, 0, 1))
    # the luminance models keep the colour ratios (below the clipping)
    x = _ramp()
    y = pt.ghs(x, dict(stretch=1.5, local=1.0, symmetry=0.05, model="Human-weighted luminance"))
    ok = (y.max(-1) < 0.999) & (x.min(-1) > 0.02)
    r0, r1 = x[..., 0] / np.maximum(x[..., 1], 1e-6), y[..., 0] / np.maximum(y[..., 1], 1e-6)
    assert np.abs(r1 - r0)[ok].max() < 1e-4


def test_asinh_and_the_histogram_transformation_follow_their_formulas():
    x = _ramp()
    t = x.astype(np.float64)
    beta, bp = 25.0, 0.02
    tp = np.clip((t - bp) / (1 - bp), 0, None)
    L = tp.mean(-1, keepdims=True)
    want = np.clip(tp * np.arcsinh(beta * L) / (L * np.arcsinh(beta)), 0, 1)
    got = pt.asinh(x, dict(stretch=beta, black=bp, model="Even-weighted luminance"))
    assert np.abs(got - want)[L[..., 0] > 1e-3].max() < 1e-5
    assert np.array_equal(pt.asinh(x, dict(stretch=0.0, black=0.0)), x)
    xs = np.linspace(0, 1, 1001)
    m = pt.ht_curve(dict(shadows=0.0, midtones=0.2, highlights=1.0), xs)
    assert m[200] == pytest.approx(0.5, abs=1e-9) and m[0] == 0 and m[-1] == pytest.approx(1)
    s = pt.ht_curve(dict(shadows=0.1, midtones=0.5, highlights=0.9), xs)
    assert np.allclose(s, np.clip((xs - 0.1) / 0.8, 0, 1))
    one = pt.histogram_transform(x, dict(shadows=0.0, midtones=0.3, highlights=1.0, channel="B"))
    assert np.array_equal(one[..., :2], x[..., :2]) and not np.allclose(one[..., 2], x[..., 2])


def test_scnr_is_sirils_formulas():
    x = _ramp()
    r, g, b = (x[..., k].astype(np.float64) for k in range(3))
    for method, a, want in (("Average neutral", 1.0, np.minimum(g, (r + b) / 2)),
                            ("Maximum neutral", 1.0, np.minimum(g, np.maximum(r, b))),
                            ("Maximum mask", 0.7, g * 0.3 * (1 - np.maximum(r, b)) + np.maximum(r, b) * g),
                            ("Additive mask", 0.7, g * 0.3 * (1 - np.minimum(1, r + b)) + np.minimum(1, r + b) * g)):
        y = pcol.scnr(x, method, a, False)
        assert np.abs(y[..., 1] - want).max() < 1e-6, method
        assert np.array_equal(y[..., [0, 2]], x[..., [0, 2]])
    y = pcol.scnr(x, "Average neutral", 1.0, True)                         # the lightness put back (Siril's L*)
    assert np.abs(pcol.lab_linear(y)[..., 0] - pcol.lab_linear(x)[..., 0]).max() < 0.05


@pytest.mark.skipif(not SIRIL.exists(), reason="siril-cli not installed")
def test_the_stretches_and_scnr_equal_siril(tmp_path):
    """The same image through siril-cli 1.4 and through Post: equal to the 8-bit level."""
    from astropy.io import fits
    x = _ramp()
    fits.writeto(tmp_path / "cin.fits", np.ascontiguousarray(np.moveaxis(x, -1, 0)), overwrite=True)
    cases = [
        ("ght -D=2 -B=0 -SP=0.1 -independent", lambda a: pt.ghs(a, dict(stretch=2, local=0, symmetry=0.1))),
        ("ght -D=1.5 -B=3 -SP=0.2 -LP=0.05 -HP=0.8 -independent",
         lambda a: pt.ghs(a, dict(stretch=1.5, local=3, symmetry=0.2, shadows=0.05, highlights=0.8))),
        ("ght -D=2 -B=2 -SP=0.15 -human -clipmode=rescale",
         lambda a: pt.ghs(a, dict(stretch=2, local=2, symmetry=0.15, model="Human-weighted luminance"))),
        ("ght -D=2 -B=2 -SP=0.15 -even -clipmode=rescale",
         lambda a: pt.ghs(a, dict(stretch=2, local=2, symmetry=0.15, model="Even-weighted luminance"))),
        ("asinh 20 0.02", lambda a: pt.asinh(a, dict(stretch=20, black=0.02, model="Even-weighted luminance"))),
        ("asinh -human 20 0.05", lambda a: pt.asinh(a, dict(stretch=20, black=0.05, model="Human-weighted luminance"))),
        ("mtf 0.05 0.3 0.95", lambda a: pt.histogram_transform(a, dict(shadows=0.05, midtones=0.3, highlights=0.95))),
        ("rmgreen -nopreserve 0", lambda a: pcol.scnr(a, "Average neutral", 1.0, False)),
        ("rmgreen -nopreserve 1", lambda a: pcol.scnr(a, "Maximum neutral", 1.0, False)),
        ("rmgreen -nopreserve 2 0.7", lambda a: pcol.scnr(a, "Maximum mask", 0.7, False)),
        ("rmgreen -nopreserve 3 0.7", lambda a: pcol.scnr(a, "Additive mask", 0.7, False)),
        ("rmgreen 0", lambda a: pcol.scnr(a, "Average neutral", 1.0, True)),
    ]
    worst = {}
    for k, (cmd, fn) in enumerate(cases):
        (tmp_path / "c.ssf").write_text(f"requires 1.2.0\nsetext fits\nset32bits\nload cin\n{cmd}\nsave o{k}\n")
        subprocess.run([str(SIRIL), "-d", str(tmp_path), "-s", str(tmp_path / "c.ssf")], capture_output=True,
                       timeout=60, check=False)
        s = np.moveaxis(fits.getdata(tmp_path / f"o{k}.fits").astype(np.float64), 0, -1)
        worst[cmd] = float(np.abs(fn(x) - s).max() * 255)
    assert max(worst.values()) < 0.5, worst


# ---------------------------------------------------------------------------------------------- detail
def test_the_starlet_transform_and_the_multiscale_tools():
    rng = np.random.default_rng(5)
    x = _img(128, 160)
    layers, res = pd.starlet(x, 5)
    back = res.clone()
    for w in layers:
        back += w
    assert np.abs(back.permute(1, 2, 0).numpy() - x).max() < 1e-5          # the layers and the residual: the image
    # the noise of each layer for a unit white noise is e(j) (Starck and Murtagh), and the MAD finds sigma
    noise = rng.normal(0, 1, (512, 512)).astype(np.float32)
    nl, _r = pd.starlet(noise, 5)
    for j, w in enumerate(nl):
        assert abs(float(w.std()) - pd.NOISE_E[j]) < 0.06 * pd.NOISE_E[j] + 0.002, j
    assert abs(float(pd.noise_sigma(nl[0])[0]) - 1.0) < 0.03
    # sharpening: a 4 px structure is raised by the bias of the 4 px layer, a 16 px one much less
    yy, xx = np.mgrid[0:128, 0:160].astype(np.float32)
    small = np.exp(-((xx - 50) ** 2 + (yy - 60) ** 2) / (2 * 2.0 ** 2))
    big = np.exp(-((xx - 115) ** 2 + (yy - 60) ** 2) / (2 * 12.0 ** 2))
    img = np.repeat((0.2 + 0.3 * small + 0.3 * big)[..., None], 3, -1).astype(np.float32)
    y = pd.sharpen(img, [0, 0, 1.0, 0, 0], "RGB")
    gain_small, gain_big = (y[60, 50, 0] - 0.2) / 0.3, (y[60, 115, 0] - 0.2) / 0.3
    assert gain_small > 1.25 and gain_big < 1.12 and gain_small - gain_big > 0.15
    assert pd.sharpen(img, [0] * 5, "RGB") is img
    # at a reduced resolution the layers finer than a pixel are left out (draft = the full image reduced)
    assert pd.sharpen(img, [1.0, 0, 0, 0, 0], "RGB", level=1) is img
    # noise reduction: pure noise much lower, a smooth signal kept
    sig = np.repeat((0.3 + 0.2 * big)[..., None], 3, -1).astype(np.float32)
    noisy = (sig + rng.normal(0, 0.03, sig.shape)).astype(np.float32)
    den = pd.denoise(noisy, [3.0, 3.0, 2.0, 0.0], 1.0, "RGB")
    assert np.std(den - sig) < 0.5 * np.std(noisy - sig)
    assert abs(float(den[60, 115, 0] - sig[60, 115, 0])) < 0.02
    lum = pd.denoise(noisy, [3.0, 3.0, 2.0, 0.0], 1.0, "Lightness (CIE L*)")
    ab0, ab1 = pcol.to_lab(noisy)[..., 1:], pcol.to_lab(lum)[..., 1:]
    assert np.abs(pcol.to_lab(lum)[..., 0] - pcol.to_lab(sig)[..., 0]).std() < \
        0.6 * np.abs(pcol.to_lab(noisy)[..., 0] - pcol.to_lab(sig)[..., 0]).std()
    assert np.abs(ab1 - ab0).mean() < 0.6                                  # the colour left as it was


def test_clahe_is_zuiderveld_as_opencv_computes_it():
    rng = np.random.default_rng(2)
    base = cv2.GaussianBlur(rng.random((180, 260)).astype(np.float32), (0, 0), 20)
    u8 = np.clip((base - base.min()) / np.ptp(base) * 120 + 60, 0, 255).astype(np.uint8)
    for clip, grid in ((2.0, (4, 3)), (4.0, (8, 6)), (1.5, (10, 7))):
        want = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(u8).astype(np.float64)
        got = pd.clahe_lightness(u8.astype(np.float32) / 255, clip, 260 / grid[0]) * 255
        assert np.abs(got - want).max() <= 1.01, (clip, grid, np.abs(got - want).max())
    # a region of the image (the 100 % view, rendered with margins) is that place of the whole image where its four
    # nearest tiles are inside the region: the grid is the whole image's
    big = cv2.GaussianBlur(rng.random((300, 400)).astype(np.float32), (0, 0), 12)
    L = (big - big.min()) / np.ptp(big) * 0.5 + 0.25
    whole = pd.clahe_lightness(L, 2.0, 40.0)
    y0, y1, x0, x1 = 60, 260, 80, 330
    reg = pd.clahe_lightness(np.ascontiguousarray(L[y0:y1, x0:x1]), 2.0, 40.0, (x0, y0), (400, 300))
    _gx, _gy, tw, th = pd._grid(400, 300, 40.0)
    ys, xs = np.arange(y0, y1), np.arange(x0, x1)

    def inside(c, t, lo, hi):                     # both neighbouring tiles of each pixel lie within [lo, hi)
        a = np.floor(c / t - 0.5).astype(int)
        return (a * t >= lo) & ((a + 2) * t <= hi)
    ok = inside(ys, th, y0, y1)[:, None] & inside(xs, tw, x0, x1)[None, :]
    assert ok.sum() > 5000 and np.isfinite(reg).all()
    assert np.abs(reg - whole[y0:y1, x0:x1])[ok].max() * 255 < 0.01
    # on an image: more local contrast, the colours' a* b* kept, amount 0 the identity
    x = np.repeat((u8.astype(np.float32) / 255)[..., None], 3, -1) * np.array([1.0, 0.8, 0.7], np.float32)
    y = pd.clahe(x, 1.0, 2.0, 32.0)
    assert np.std(pcol.lightness(y)) > 1.3 * np.std(pcol.lightness(x))
    assert pd.clahe(x, 0.0, 2.0, 32.0) is x


# ---------------------------------------------------------------------------------------------- in the composition
class Flat(postengine.Provider):
    def __init__(self, img):
        self.img = img

    def available(self, pid):
        return True

    def canvas(self, pids):
        return postengine.Canvas(self.img.shape[1], self.img.shape[0])

    def layer(self, lay, level, box, draft=False, stop=None):
        a = self.img
        f = 2 ** level
        if f > 1:
            a = cv2.resize(a, (a.shape[1] // f, a.shape[0] // f), interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = box
        return np.ascontiguousarray(a[y0:y1, x0:x1])


def test_the_composition_returns_the_histograms_and_drafts_use_tables():
    img = _img(160, 240)
    cp = Composer(Flat(img))
    base = new_workflow_layer(1, "A", True)
    cur = new_filter_layer("curves", base["id"])
    cur["params"]["curves"] = pc.preset_curves("Strong Contrast")
    layers = normalize([cur, base])
    out = cp.compose(layers, 1, hists=(cur["id"],))
    h = out.hists[cur["id"]]
    assert set(h) == {"R", "G", "B", "L*", "S", "c*", "H"} and all(len(v) == 256 for v in h.values())
    small = cv2.resize(img, (120, 80), interpolation=cv2.INTER_AREA)       # the histogram of the curve's input
    assert np.array_equal(h["R"], pc.histograms(small)["R"])
    draft = cp.compose(layers, 1, draft=True)
    assert not draft.hists and np.abs(draft.rgb - out.rgb).max() * 255 < 0.5
    assert np.abs(out.rgb - np.clip(pc.apply(small, cur["params"]["curves"]), 0, 1)).max() < 1e-6


# ---------------------------------------------------------------------------------------------- the editor in the page
def _wait(s, cond, timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s.qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_the_curves_editor_edits_by_hand_with_drafts_then_the_exact_image(post):
    s, v = post, post.view
    full = named(v, "Full")
    v.select(full["id"], "layer")
    fl = v.add_filter("curves")
    assert s.settle()
    ed = v.curves_editor
    assert isinstance(ed, CurvesEditor) and ed.channel == "RGB/K"
    assert ed.graph.hist is not None and ed.graph.hist["R"].sum() > 0       # the live histogram of its input
    before = shown(v).copy()
    n_undo = len(v.undo)
    g = ed.graph
    g.resize(360, 290)
    drafts = []
    v.rendered.connect(lambda r: drafts.append(r.kind))
    # a click adds a point there; dragging it upwards: drafts, then on release the exact image, one undo step
    p0 = g.to_w(0.5, 0.5).toPoint()
    QTest.mousePress(g, Qt.LeftButton, Qt.NoModifier, p0)
    for k in range(1, 6):
        QTest.mouseMove(g, g.to_w(0.5, 0.5 + 0.03 * k).toPoint())
        _wait(s, lambda: "draft" in drafts, 5)
    QTest.mouseRelease(g, Qt.LeftButton, Qt.NoModifier, g.to_w(0.5, 0.65).toPoint())
    assert s.settle()
    pts = fl["params"]["curves"]["RGB/K"]["pts"]
    assert len(pts) == 3 and abs(pts[1][0] - 0.5) < 0.01 and abs(pts[1][1] - 0.65) < 0.01
    assert "draft" in drafts and len(v.undo) == n_undo + 1
    assert not np.array_equal(shown(v), before) and np.array_equal(shown(v), reference(s).rgb)
    assert abs(ed.inp.value() - pts[1][0]) < 1e-3 and abs(ed.out.value() - pts[1][1]) < 1e-3
    # Ctrl+click deletes the point; the end points stay; Undo brings the point back
    QTest.mouseClick(g, Qt.LeftButton, Qt.ControlModifier, g.to_w(*pts[1]).toPoint())
    assert len(fl["params"]["curves"]["RGB/K"]["pts"]) == 2
    assert not g.delete(0) and not g.delete(1)
    assert s.settle() and np.array_equal(shown(v), before)
    v.do_undo()
    assert s.settle()
    fl = v.layer(fl["id"])
    assert len(fl["params"]["curves"]["RGB/K"]["pts"]) == 3
    # a drag out of the graph deletes; the arrow keys nudge by one 8-bit level
    ed = v.curves_editor
    g = ed.graph
    g.resize(360, 290)
    g.sel = 1
    y0 = fl["params"]["curves"]["RGB/K"]["pts"][1][1]
    QTest.keyClick(g, Qt.Key_Up)
    assert abs(fl["params"]["curves"]["RGB/K"]["pts"][1][1] - (y0 + 1 / 255)) < 1e-9
    a = g.to_w(*fl["params"]["curves"]["RGB/K"]["pts"][1]).toPoint()
    QTest.mousePress(g, Qt.LeftButton, Qt.NoModifier, a)
    QTest.mouseMove(g, QPoint(a.x(), g.height() + 80))
    QTest.mouseRelease(g, Qt.LeftButton, Qt.NoModifier, QPoint(a.x(), g.height() + 80))
    assert len(fl["params"]["curves"]["RGB/K"]["pts"]) == 2
    # a preset, then a channel of its own; the composition is saved with the curves
    ed.apply_preset("Medium Contrast")
    assert fl["params"]["curves"]["RGB/K"]["pts"] == pc.preset_curves("Medium Contrast")["RGB/K"]["pts"]
    ed.set_channel("L*")
    assert ed.graph.curve is fl["params"]["curves"]["L*"] and "•" in ed.ch_buttons["RGB/K"].text()
    assert s.settle() and np.array_equal(shown(v), reference(s).rgb)
    v.flush()
    saved = next(x for x in s.model.p.post["layers"] if x["id"] == fl["id"])
    assert saved["params"]["curves"]["RGB/K"]["pts"] == pc.preset_curves("Medium Contrast")["RGB/K"]["pts"]
    back = postengine.load_layers(json.loads(json.dumps(s.model.p.post["layers"])), {1, 2})
    assert next(x for x in back if x["id"] == fl["id"])["params"]["curves"] == pc.clean_curves(fl["params"]["curves"])


def test_the_users_presets_in_the_editor(post, tmp_path):
    v = post.view
    v.select(named(v, "Full")["id"], "layer")
    fl = v.add_filter("curves")
    ed = v.curves_editor
    ed.apply_preset("Strong Contrast")
    assert ed.save_preset("Heart S") == "Heart S"
    assert ed.preset.currentData() == "user:Heart S" and ed.a_delete.isEnabled()
    assert [p["name"] for p in pc.user_presets()] == ["Heart S"]
    ed.reset_all()
    assert pc.active(fl["params"]["curves"]) == []
    ed.apply_preset("user:Heart S")
    assert fl["params"]["curves"]["RGB/K"] == pc.preset_curves("Strong Contrast")["RGB/K"]
    ed.delete_preset()
    assert pc.user_presets() == []
    # a Photoshop file
    pc.write_preset_file(tmp_path / "x.acv", pc.preset_curves("Cross Process"))
    assert ed.import_file(str(tmp_path / "x.acv"))
    assert pc.active(fl["params"]["curves"]) and "R" in fl["params"]["curves"]
    assert ed.export_file(str(tmp_path / "y.json"))
    assert pc.read_preset_file(tmp_path / "y.json") == pc.clean_curves(fl["params"]["curves"])


def test_the_stretches_show_their_transfer_function_and_the_menu_says_the_method(post):
    v = post.view
    v.select(named(v, "Full")["id"], "layer")
    fl = v.add_filter("ghs")
    assert post.settle()
    assert isinstance(v.function_graph, FunctionGraph) and v.function_graph.hist is not None
    fl["params"].update(stretch=2.0, symmetry=0.2)
    xs, ys = v.function_graph.values(11)
    assert ys[2] > xs[2]
    v.build_filter_menu()
    tips = {a.text(): a.toolTip() for m in v.filter_menu.actions() for a in m.menu().actions()}
    assert "Payne" in tips["Generalised hyperbolic stretch"] and "Lupton" in tips["Arcsinh stretch"]
    assert "Arcsinh stretch (symmetry point)" not in tips                   # the first Post's: no longer offered
    sc = v.add_filter("scnr")
    from PySide6.QtWidgets import QCheckBox
    box = next(w for w in v.findChildren(QCheckBox) if w.text() == "Preserve lightness")
    assert box.isChecked() and sc["params"]["lightness"] is True
    box.setChecked(False)
    assert sc["params"]["lightness"] is False


def test_the_tools_in_french(preps, qapp):
    i18n.set_language("fr")
    s = Setup(preps, qapp)
    try:
        assert s.settle()
        v = s.view
        v.select(named(v, "Full")["id"], "layer")
        for key in NEW:
            v.select(named(v, "Full")["id"], "layer")
            v.add_filter(key)
        ed = None
        for lay in v.layers:
            if lay["kind"] == "filter" and lay["filter"] == "curves":
                v.select(lay["id"], "layer")
                ed = v.curves_editor
        for ch in pc.CHANNELS:
            ed.set_channel(ch)
        ed.fill_presets()
        texts = [ed.preset.itemText(i) for i in range(ed.preset.count())]
        assert "   Contraste moyen" in texts and "Personnalisé" in texts and "   Remonter les tons moyens" in texts
        assert ed.b_reset.text() == "Réinitialiser le canal"
        v.build_filter_menu()
        assert [a.text() for a in v.filter_menu.actions()][:2] == ["Étirement", "Contraste et courbes"]
        assert not i18n.missing(), sorted(i18n.missing())
    finally:
        s.close()
        i18n.set_language("en")


# ---------------------------------------------------------------------------------------------- the real project
needs_real = pytest.mark.skipif(not (REAL / "project.astroproj").exists(), reason="test copy not there")


def take_tool_shots(s, lang: str, out: Path):
    """The Curves editor with a preset, the filter list, and the new filters applied, on the real project."""
    from PySide6.QtGui import QPainter, QPixmap
    out.mkdir(parents=True, exist_ok=True)
    v, app = s.view, s.qapp
    s.page.resize(1600, 980)
    full = named(v, "Full")

    def grab(w):
        for _i in range(6):
            app.processEvents()
            time.sleep(0.03)
        return w.grab()

    def save(name, pm=None):
        assert (pm or grab(s.page)).save(str(out / f"tools_{name}_{lang}.png")), name

    def only(lay):                                       # the one filter shown, the others hidden
        for x in v.layers:
            if x["kind"] == "filter":
                x["visible"] = x["id"] == lay["id"]
        v.select(lay["id"], "layer")
        v.rebuild()
        v.request(final_only=True)
        assert s.settle(600)

    v.select(full["id"], "layer")
    cur = v.add_filter("curves")
    v.curves_editor.apply_preset("Midtone contrast, highlights protected")
    v.curves_editor.set_channel("RGB/K")
    only(cur)
    save("curves_preset")
    v.curves_editor.preset.showPopup()
    pop = v.curves_editor.preset.view().window()
    shot = QPixmap(grab(s.page))
    qp = QPainter(shot)
    at = v.curves_editor.preset.mapTo(s.page, QPoint(0, v.curves_editor.preset.height()))
    qp.drawPixmap(QPoint(max(0, min(at.x(), shot.width() - pop.width())),
                         max(0, min(at.y(), shot.height() - pop.height()))), grab(pop))
    qp.end()
    v.curves_editor.preset.hidePopup()
    save("curves_presets_list", shot)
    cur["params"]["curves"].clear()
    cur["params"]["curves"].update({"S": {"pts": [[0, 0], [0.3, 0.42], [1, 1]], "interp": "Akima"},
                                    "L*": {"pts": [[0, 0], [0.25, 0.21], [0.7, 0.76], [1, 1]], "interp": "Akima"}})
    v.rebuild()
    v.curves_editor.set_channel("S")
    only(cur)
    save("curves_saturation")
    # the filter list (the offscreen screen is small: the popups are drawn where the user would see them)
    v.build_filter_menu()
    menu = v.filter_menu
    menu.popup(v.b_filter.mapToGlobal(QPoint(0, v.b_filter.height())))
    fam = next(a for a in menu.actions() if a.text() in ("Stretch", "Étirement"))
    sub = fam.menu()
    sub.popup(menu.mapToGlobal(menu.actionGeometry(fam).topRight()))
    shot = QPixmap(grab(s.page))
    qp = QPainter(shot)
    at = v.b_filter.mapTo(s.page, QPoint(0, v.b_filter.height()))
    at = QPoint(min(at.x(), shot.width() - menu.width() - sub.width()), at.y())
    qp.drawPixmap(at, grab(menu))
    qp.drawPixmap(QPoint(at.x() + menu.width(), at.y() + menu.actionGeometry(fam).y()), grab(sub))
    qp.end()
    sub.hide()
    menu.hide()
    save("filter_list", shot)
    # the new filters, one at a time (the detail ones at 100 %, with the same view without the filter)
    cw, ch = v.src.canvas
    for key, kw, zoom in (("ghs", dict(stretch=0.8, local=2.0, symmetry=0.17, highlights=0.85,
                                       model="Human-weighted luminance"), None),
                          ("clahe", dict(amount=0.4, clip=1.6, radius=128), None),
                          ("ml_sharpen", dict(l2=0.5, l3=0.7, l4=0.4), (0.37, 0.36)),
                          ("ml_denoise", dict(l1=3.0, l2=2.0, l3=1.0, target="Chrominance (CIE a* b*)"),
                           (0.37, 0.36)),
                          ("scnr", dict(strength=0.8, method="Maximum neutral"), None),
                          ("histogram", dict(shadows=0.03, midtones=0.42), None)):
        v.select(full["id"], "layer")
        f = v.add_filter(key)
        f["params"].update(kw)
        only(f)
        if zoom:
            v.viewer.zoom_to(1.0, cw * zoom[0], ch * zoom[1])
            assert s.settle(600)
        save(key)
        if zoom:
            f["visible"] = False
            v.rebuild()
            v.request(final_only=True)
            assert s.settle(600)
            save(key + "_before")
            v.viewer.fit()
            assert s.settle(600)


def test_tool_shots_on_the_synthetic_project(preps, qapp):
    for lang in ("en", "fr"):
        i18n.set_language(lang)
        s = Setup(preps, qapp)
        try:
            assert s.settle()
            take_tool_shots(s, lang, SHOTS / "synthetic")
        finally:
            s.close()
            i18n.set_language("en")
    assert len(list((SHOTS / "synthetic").glob("tools_*.png"))) == 2 * 12


@needs_real
@pytest.mark.slow
def test_real_post_tools_screenshots_en_and_fr(qapp):
    for lang in ("en", "fr"):
        i18n.set_language(lang)
        s = RealSetup(qapp)
        try:
            assert s.settle(600)
            take_tool_shots(s, lang, SHOTS)
        finally:
            s.close()
            i18n.set_language("en")
    names = sorted(p.name for p in SHOTS.glob("tools_*.png"))
    assert len(names) == 2 * 12, names


def drag_times(s, key: str, moves: int = 8) -> list[float]:
    """The drafts' times (ms, the whole request in the worker) while a filter's first slider (or a curve point) is
    dragged on the real project."""
    v = s.view
    full = named(v, "Full")
    v.select(full["id"], "layer")
    f = v.add_filter(key)
    assert s.settle(600)
    times = []
    v.rendered.connect(lambda r: times.append(r.ms) if r.kind == "draft" else None)
    for k in range(moves):
        if key == "curves":
            f["params"]["curves"]["RGB/K"] = {"pts": [[0, 0], [0.4, 0.4 + 0.02 * (k + 1)], [1, 1]], "interp": "Akima"}
        else:
            q = FILTERS[key]["params"][0]
            f["params"][q["key"]] = q["lo"] + (q["hi"] - q["lo"]) * 0.05 * (k + 1)
        n = len(times)
        v.request()
        v._final_t.stop()
        _wait(s, lambda n=n: len(times) > n, 30)
    v.request(final_only=True)
    assert s.settle(600)
    return times


@needs_real
@pytest.mark.slow
def test_real_a_filter_drag_is_quick(qapp):
    s = RealSetup(qapp)
    try:
        assert s.settle(600)
        out = {}
        for key in ("curves", "ghs", "clahe", "ml_sharpen"):
            t = drag_times(s, key)
            out[key] = round(float(np.median(t)), 1)
        print(f"drafts on {s.worker.device} (level {s.view.draft_level}): {out}")
        assert max(out.values()) < 1500
    finally:
        s.close()


def test_nothing_here_wrote_the_users_presets_or_settings():
    assert _fingerprint(USER_SETTINGS) == USER_BEFORE and _fingerprint(USER_PRESETS) == PRESETS_BEFORE
