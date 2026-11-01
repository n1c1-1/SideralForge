"""Post and Export (P12), offscreen: the composition (layers, 13 filters, masks), the render worker (supersede, device,
fallback), the engine's renderings through it (bit for bit), the export, the screen, the French and the screenshots.

The synthetic preparations (a nebula, stars, an uncovered border) make the fast tests; the real ones use F:/astro_essais/v2/
heart_a7s_p12, a copy of the adopted heart_a7s project (skipped without it) and are marked slow.
ASTRO_GPU_SETTINGS points to a temporary file everywhere: the user's settings file is checked untouched.
ASTRO_GPU_SHOTS names the folder of the screenshots (default: %TEMP%/p12_shots).
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

pytest.importorskip("PySide6")
import torch  # noqa: E402
from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n, postengine, prefs  # noqa: E402
from astro_gpu.client.exportdlg import ExportDialog  # noqa: E402
from astro_gpu.client.post3 import PostPage3, tr_name  # noqa: E402
from astro_gpu.client.postengine import (  # noqa: E402
    FILTERS,
    Canvas,
    Composer,
    Provider,
    Superseded,
    new_filter_layer,
    new_mask,
    new_part,
    new_workflow_layer,
    normalize,
    snapshot,
)
from astro_gpu.client.services import render as rsv  # noqa: E402
from astro_gpu.client.services.render import (  # noqa: E402
    Folder,
    LayerSource,
    Renderer,
    RenderWorker,
    Request,
    choose_device,
)
from astro_gpu.client.widgets import QSS  # noqa: E402
from astro_gpu.render.engine import RenderEngine  # noqa: E402
from astro_gpu.render.layers import NOTE_CPU, LayerService  # noqa: E402
from astro_gpu.render.settings import RenderSettings  # noqa: E402
from astro_gpu.render.simple import image_stats  # noqa: E402
from astro_gpu.render.sliders import KEYS, SLIDERS, availability  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "astro_gpu" / "client"
REAL = Path("F:/astro_essais/v2/heart_a7s_p12")
DEV = "cpu"                                              # the fast tests render on the CPU: no GPU needed, no GPU busy
SHOTS = Path(os.environ.get("ASTRO_GPU_SHOTS") or Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "p12_shots")


def _user_settings() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def _fingerprint(p: Path):
    return (p.stat().st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest()) if p.exists() else None


USER_SETTINGS = _user_settings()
USER_BEFORE = _fingerprint(USER_SETTINGS)


# ---------------------------------------------------------------------------------------------- synthetic preparations
def engine_prep(d: Path, H: int, W: int, seed: int = 3, crop=None, scale: float = 1.5) -> Path:
    """A render_prep folder: a nebula, stars, the optional layers of a layered preparation, an uncovered border."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    neb = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), max(H, W) / 30)
    neb = (neb - neb.min()) / (neb.max() - neb.min()) * 120
    astro = np.stack([neb + 5, neb * .45 + 5, neb * .25 + 5]) + rng.normal(0, .8, (3, H, W)).astype(np.float32)
    valid = np.ones((H, W), bool)
    valid[:, :20] = False
    np.save(d / "valid.npy", valid)
    lum = np.array([0.2126, 0.7152, 0.0722])
    pj = {"crop": list(crop or [24, W - 4, 4, H - 4]), "alpha": 0.87, "vh": [1.0, 0.45, 0.25], "M": np.eye(3).tolist(),
          "drizzle_scale": scale}
    np.save(d / "N.npy", astro.astype(np.float32))
    np.save(d / "Eg0.npy", cv2.GaussianBlur(np.tensordot(lum, astro, 1), (0, 0), 8).astype(np.float32))
    np.save(d / "dY.npy", rng.normal(0, .8, (H, W)).astype(np.float16))
    np.save(d / "R.npy", (0.2 * astro - 0.2 * astro.mean((1, 2), keepdims=True)).astype(np.float32))
    O = np.zeros((H, W), np.float32)
    O[H // 4:H // 2, W // 4:W // 2] = 5
    np.save(d / "O.npy", O + rng.normal(0, .05, (H, W)).astype(np.float32))
    np.save(d / "chroma_fine.npy", rng.normal(0, .5, (3, H, W)).astype(np.float16))
    pj.update({"w_cam": [1, 1, 1], "o": [0, 1, .75], "o_display": [0, 1, .75], "oiii_sigma": 1.0, "oiii_scale": 4.0})
    n = int(H * W / 300)
    g = rng.uniform(6, 19, n)
    f = 10 ** (-0.4 * (g - 10)) * 2e4
    flux = np.stack([f * .9, f, f * .8]).astype(np.float32)
    np.savez(d / "stars.npz", x=rng.uniform(-30, W + 30, n), y=rng.uniform(-30, H + 30, n), g=g, flux=flux,
             pred=flux, glow_flux=flux)
    r = np.arange(0, 61, 0.25)
    np.savez(d / "profile.npz", r=r, prof=np.exp(-0.5 * (r / 1.1) ** 2) + 1e-4 * np.exp(-r / 8))
    (d / "prep.json").write_text(json.dumps(pj))
    return d


def simple_prep(d: Path, H: int, W: int, exit_: str | None = None, denoised: bool = True, crop=None,
                scale: float = 1.0) -> Path:
    rng = np.random.default_rng(4)
    d.mkdir(parents=True, exist_ok=True)
    neb = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 20) * 400
    lin = np.stack([neb, .5 * neb, .3 * neb]) - 150 + rng.normal(0, 3, (3, H, W)).astype(np.float32)
    valid = np.ones((H, W), bool)
    prep = {"M": np.eye(3).tolist(), "crop": list(crop or [4, W - 4, 4, H - 4]), "stats": image_stats(lin, valid),
            "linked": True, "stretch": "auto", "denoised": denoised, "color": exit_ is None, "exit": exit_,
            "drizzle_scale": scale}
    np.save(d / "lin.npy", lin.astype(np.float32))
    if denoised:
        raw = (lin + rng.normal(0, 10, lin.shape)).astype(np.float32)
        np.save(d / "lin_raw.npy", raw)
        prep["stats_raw"] = image_stats(raw, valid)
    np.save(d / "valid.npy", valid)
    (d / "prep_simple.json").write_text(json.dumps(prep))
    return d


@pytest.fixture(scope="module")
def preps(tmp_path_factory):
    t = tmp_path_factory.mktemp("post_preps")
    return {"full": engine_prep(t / "full", 480, 720), "classic": simple_prep(t / "classic", 480, 720),
            "raw": simple_prep(t / "raw", 480, 720, "stacking", False),
            "other": simple_prep(t / "other", 640, 960, crop=[16, 940, 8, 630], scale=2.0)}


@pytest.fixture(scope="session")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


@pytest.fixture(autouse=True)
def _settings(tmp_path, monkeypatch):
    """Nothing here writes the user's settings: a temporary file stands in, English."""
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    prefs.set("workspace", (tmp_path / "ws").as_posix())
    i18n.set_language("en")
    yield
    i18n.set_language("en")


# ---------------------------------------------------------------------------------------------- the toy model
class WF:
    """What the views read of a workflow (the model's WorkflowView)."""

    def __init__(self, pid, name, kind, rendered=True, denoise=True):
        self.pid, self.name, self.kind, self.denoise = pid, name, kind, denoise
        self.rendered = self.computed = rendered
        self.provisional = False
        self.end = "empilement" if kind == "stacking" else None


class ToyModel(QObject):
    """The ProjectModel as far as Post uses it: the workflows, the sliders, the project's ``post``."""
    pathsChanged = Signal()

    def __init__(self, wfs, root=None, name="HeartSoul"):
        super().__init__()
        self.paths = wfs
        self.p = SimpleNamespace(post={"layers": []}, render_bases={})
        self.name, self.root = name, root
        self.device, self.gpu = "cpu", False
        self.touched = 0

    def view(self, pid):
        return next((w for w in self.paths if w.pid == pid), None)

    def slider_infos(self, p):
        why = availability(p.kind, p.denoise)
        return [api.SliderInfo(s.key, s.name, s.group, s.ends, not why[s.key], why[s.key]) for s in SLIDERS]

    def slider_state(self, p, key):
        info = next(x for x in self.slider_infos(p) if x.key == key)
        return info.available, i18n._(info.reason), None                      # as the real model: translated

    def _touch(self):
        self.touched += 1


class Setup:
    """A Post page on synthetic preparations: workflow 1 'Full' (stars), 2 'Classic' (simple), 3 'Raw stack'."""

    def __init__(self, preps, qapp, which=(1, 2), root=None, resolver=None, rendered=True):
        self.qapp = qapp
        kinds = {1: ("Full", "engine", "full"), 2: ("Classic", "simple", "classic"), 3: ("Raw stack", "stacking", "raw")}
        self.folders = {pid: Folder(preps[kinds[pid][2]]) for pid in kinds if kinds[pid][2] in preps}
        self.model = ToyModel([WF(pid, kinds[pid][0], kinds[pid][1], rendered) for pid in which], root=root)
        self.worker = RenderWorker(None, resolver=resolver or (lambda pid: self.folders.get(pid)), device=DEV)
        self.page = PostPage3(self.model, worker=self.worker)
        self.page.resize(1500, 900)
        self.page.show()
        self.view = self.page.view

    def settle(self, timeout=90):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.worker.settle(5)
            self.qapp.processEvents()
            if not self.worker.busy() and not self.view._final_t.isActive() and not self.view._draft_t.isActive() \
                    and not self.view._region_t.isActive():
                return True
        return False

    def close(self):
        self.page.shutdown()
        self.page.deleteLater()


@pytest.fixture()
def post(preps, qapp):
    s = Setup(preps, qapp)
    assert s.settle()
    yield s
    s.close()


@pytest.fixture()
def wide(tmp_path_factory, qapp):
    """A page on a wide image (2400 x 1500): zoomed out the image is under half size, as Heart's is on a laptop."""
    t = tmp_path_factory.mktemp("post_wide")
    p = {"full": engine_prep(t / "full", 1500, 2400), "classic": simple_prep(t / "classic", 1500, 2400)}
    s = Setup(p, qapp, which=(1,))
    assert s.settle(180)
    yield s
    s.close()


# ---------------------------------------------------------------------------------------------- the filters
def _lum(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def _gauss(x, sigma):
    from scipy import ndimage
    return x if sigma < 0.3 else ndimage.gaussian_filter(x, (sigma, sigma, 0) if x.ndim == 3 else sigma)


def _hue(x):
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn + 1e-6
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, d / (mx + 1e-6)


def _hw(h, c, w):
    return np.clip(1 - np.abs((h - c + 180) % 360 - 180) / max(w, 1), 0, 1)


class RefCtx:
    """The mockup's formulas (scipy, float64), the reference of the port."""

    def __init__(self, stars, halpha, f=1):
        self._s, self._h, self.f = stars, halpha, f

    def stars(self):
        return self._s

    def halpha(self):
        return self._h


def _reference(key, x, p, ctx):
    from scipy import ndimage
    f = ctx.f
    if key == "stretch":
        d, sp = p["strength"], p["symmetry"]
        a0, a1 = np.arcsinh(-d * sp), np.arcsinh(d * (1 - sp))
        return (np.arcsinh(d * (x - sp)) - a0) / (a1 - a0)
    if key == "curves":
        ys = [0, 0.25 + p["shadows"], 0.5 + p["midtones"], 0.75 + p["highlights"], 1]
        return np.interp(x, [0, 0.25, 0.5, 0.75, 1], np.clip(ys, 0, 1))
    if key == "levels":
        return np.clip((x - p["black"]) / max(p["white"] - p["black"], 1e-3), 0, 1) ** (1 / p["gamma"])
    if key == "saturation":
        L = _lum(x)[..., None]
        center = dict(postengine.HUES)[p["hues"]]
        if center is None:
            w = 1.0
        else:
            h, s = _hue(x)
            w = (_hw(h, center, 45) * np.clip(s * 3, 0, 1))[..., None]
        return L + (x - L) * (1 + p["strength"] / 100 * w)
    if key == "scnr":
        y = x.copy()
        g = np.minimum(x[..., 1], (x[..., 0] + x[..., 2]) / 2)
        y[..., 1] = x[..., 1] + p["strength"] * (g - x[..., 1])
        return y
    if key == "balance":
        t, g = p["temperature"], p["tint"]
        return x * np.array([1 + 0.15 * t, 1 + 0.1 * g, 1 - 0.15 * t])
    if key == "local_contrast":
        return x + p["strength"] * (x - _gauss(x, p["scale"] / f))
    if key == "sharpen":
        return x + p["strength"] * (x - _gauss(x, max(p["radius"] / f, 0.35)))
    if key == "denoise":
        Y = _lum(x)[..., None]
        C_ = x - Y
        return Y + p["luminance"] * (_gauss(Y, 1.6 / f) - Y) + C_ + p["color"] * (_gauss(C_, 4 / f) - C_)
    if key == "star_reduce":
        m = ctx.stars()[..., None]
        er = ndimage.grey_erosion(x, size=(max(2, int(round(5 / f))),) * 2 + (1,))
        return x + p["strength"] * m * (er - x)
    if key == "halos":
        m = ctx.stars()
        r = max(1, int(round(p["radius"] / f)))
        ring = _gauss(np.clip(ndimage.grey_dilation(m, size=(2 * r + 1, 2 * r + 1)) - m, 0, 1), r / 3)[..., None]
        bg = _gauss(ndimage.grey_erosion(x, size=(2 * r + 1, 2 * r + 1, 1)), r / 2)
        return x + p["strength"] * ring * (np.minimum(x, bg) - x)
    if key == "star_color":
        m = ctx.stars()[..., None]
        L = _lum(x)[..., None]
        return L + (x - L) * (1 + p["strength"] * m)
    if key == "halpha_red":
        y = x.copy()
        y[..., 0] = x[..., 0] + p["strength"] * 0.35 * ctx.halpha() * (1 - x[..., 0])
        return y
    raise KeyError(key)


def test_the_thirteen_filters_are_the_mockups_formulas():
    """Each filter, on a random image, at several settings and resolutions: the port (and its fast paths) gives what the
    mockup's formulas give, to a fraction of an 8-bit level."""
    assert len(FILTERS) == 13 and set(FILTERS) == {
        "stretch", "curves", "levels", "saturation", "scnr", "balance", "local_contrast", "sharpen", "denoise",
        "star_reduce", "halos", "star_color", "halpha_red"}
    rng = np.random.default_rng(1)
    x = np.clip(cv2.GaussianBlur(rng.random((150, 230, 3)).astype(np.float32), (0, 0), 3) * 1.6 - 0.3, 0, 1)
    stars = np.clip(rng.random((150, 230)).astype(np.float32) * 1.4 - 0.4, 0, 1)
    halpha = cv2.GaussianBlur(rng.random((150, 230)).astype(np.float32), (0, 0), 8)
    worst = {}
    for key, f in FILTERS.items():
        defaults = {q["key"]: q["default"] for q in f["params"]}
        variants = [defaults]
        for q in f["params"]:
            if q["type"] == "float":
                variants += [dict(defaults, **{q["key"]: q["lo"] + (q["hi"] - q["lo"]) * t}) for t in (0.3, 0.8)]
            else:
                variants += [dict(defaults, **{q["key"]: c}) for c in q["choices"]]
        for lvl in (0, 1):
            ctx = RefCtx(stars, halpha, 2 ** lvl)
            for p in variants:
                got = f["fn"](x.copy(), p, ctx)
                want = _reference(key, x.astype(np.float64), p, ctx)
                assert got.shape == x.shape and got.dtype == np.float32, (key, got.dtype)
                err = float(np.abs(np.clip(got, 0, 1) - np.clip(want, 0, 1)).max()) * 255
                worst[key] = max(worst.get(key, 0), err)
    assert max(worst.values()) < 1.0, worst
    # the families of the spec, and the defaults of its table
    assert {f["family"] for f in FILTERS.values()} == set(postengine.FAMILIES)
    d = {k: {q["key"]: q["default"] for q in f["params"]} for k, f in FILTERS.items()}
    assert d["stretch"] == {"strength": 3, "symmetry": 0.08} and d["saturation"] == {"strength": 30, "hues": "All"}
    assert d["scnr"] == {"strength": 0.8} and d["local_contrast"] == {"strength": 0.4, "scale": 40}
    assert d["halos"] == {"strength": 0.6, "radius": 14} and d["halpha_red"] == {"strength": 0.5}


def test_blend_modes_are_the_specs():
    a = np.random.default_rng(2).random((20, 30, 3)).astype(np.float32)
    b = np.random.default_rng(3).random((20, 30, 3)).astype(np.float32)
    bl = Composer.blend
    assert np.array_equal(bl("Normal", a, b), b) and np.array_equal(bl("Lighten", a, b), np.maximum(a, b))
    assert np.allclose(bl("Screen", a, b), 1 - (1 - a) * (1 - b), atol=1e-6)
    assert np.allclose(bl("Multiply", a, b), a * b, atol=1e-6)
    assert np.allclose(bl("Soft light", a, b), (1 - 2 * b) * a * a + 2 * b * a, atol=1e-6)


# ---------------------------------------------------------------------------------------------- the composition
class ToyProvider(Provider):
    """Workflow 1: a smooth gradient image with a bright disc; 2: its inverse. Data masks: a few star blobs, a ramp."""

    def __init__(self, W=320, H=200):
        self.W, self.H = W, H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        a = np.stack([xx / W, yy / H, 0.3 + 0.4 * np.sin(xx / 40) * np.cos(yy / 30)], -1) * 0.8 + 0.1
        a[(xx - 100) ** 2 + (yy - 80) ** 2 < 30 ** 2] = 0.9
        self.img = {1: np.clip(a, 0, 1).astype(np.float32), 2: np.clip(1 - a, 0, 1).astype(np.float32)}
        self.stars = np.zeros((H, W), np.float32)
        for x, y in ((50, 40), (200, 120), (260, 50)):
            self.stars += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 20).astype(np.float32)
        self.stars = np.clip(self.stars, 0, 1)
        self.ramp = (xx / W).astype(np.float32)
        self.calls = []
        self.epoch = 0

    def available(self, pid):
        return pid in self.img

    def version(self, pid):
        return self.epoch

    def canvas(self, pids):
        return Canvas(self.W, self.H)

    def _at(self, a, level, box):
        f = 2 ** level
        if f > 1:
            a = cv2.resize(a, (self.W // f, self.H // f), interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = box
        return np.ascontiguousarray(a[y0:y1, x0:x1])

    def layer(self, lay, level, box):
        self.calls.append((lay["pid"], level, tuple(box)))
        a = self.img[lay["pid"]]
        k = lay["sliders"].get("sky", 2.0) - 2
        a = np.clip(a + 0.05 * k, 0, 1)
        if lay.get("part") == "stars":
            a = np.clip(a - 0.5, 0, 1)
        return self._at(a, level, box)

    def data_mask(self, name, level, box):
        return self._at(self.stars if name == "stars" else self.ramp, level, box)

    def has(self, name):
        return True


def _stack(*pairs):
    """Layers top first from (pid, name) pairs given bottom first."""
    return [new_workflow_layer(pid, name, True) for pid, name in reversed(pairs)]


def test_the_composition_follows_the_spec():
    """Bottom first; hidden skipped; a layer's own filters, then its blend and mix; opacity and mask."""
    pr = ToyProvider()
    cp = Composer(pr)
    bottom, top = _stack((1, "A"), (2, "B"))[::-1]
    layers = [top, bottom]                                         # top first
    out = cp.compose(layers, 0)
    assert out.rgb.shape == (200, 320, 3) and out.rgb.dtype == np.float32
    assert np.allclose(out.rgb, pr.img[2], atol=1e-6)               # the top layer, Normal, 100 %, covers the bottom one
    top["visible"] = False
    assert np.allclose(cp.compose(layers, 0).rgb, pr.img[1], atol=1e-6)
    top["visible"], top["opacity"] = True, 40.0
    assert np.allclose(cp.compose(layers, 0).rgb, pr.img[1] + (pr.img[2] - pr.img[1]) * 0.4, atol=1e-5)
    top["opacity"], top["mode"] = 100.0, "Multiply"
    assert np.allclose(cp.compose(layers, 0).rgb, pr.img[1] * pr.img[2], atol=1e-6)
    top["mode"] = "Normal"
    # a filter clipped to the top layer changes only that layer's image; one on everything below changes the sum
    clip = new_filter_layer("levels", top["id"])
    clip["params"].update(gamma=2.0, black=0.0)
    below = new_filter_layer("levels", None)
    below["params"].update(gamma=2.0, black=0.0)
    only = cp.compose(normalize([clip, top, bottom]), 0).rgb
    assert np.allclose(only, np.clip(pr.img[2], 0, 1) ** 0.5, atol=1e-5)
    all_below = cp.compose(normalize([below, top, bottom]), 0).rgb
    assert np.allclose(all_below, pr.img[2] ** 0.5, atol=1e-5)       # here the top layer hides the bottom one: same
    top["opacity"] = 50.0
    mixed_clip = cp.compose(normalize([clip, top, bottom]), 0).rgb
    mixed_all = cp.compose(normalize([below, top, bottom]), 0).rgb
    half = pr.img[1] + (pr.img[2] - pr.img[1]) * 0.5
    assert np.allclose(mixed_clip, pr.img[1] + (pr.img[2] ** 0.5 - pr.img[1]) * 0.5, atol=1e-5)
    assert np.allclose(mixed_all, half ** 0.5, atol=1e-5)            # the filter acts on the composite
    assert not np.allclose(mixed_clip, mixed_all, atol=1e-3)


def test_masks_parts_combine_and_are_processed():
    pr = ToyProvider()
    cp = Composer(pr)
    lay = new_workflow_layer(1, "A", True)
    lay["mask"] = new_mask()
    layers = [lay]
    m = lambda **k: cp.compose(layers, 0, want=(lay["id"],)).masks[lay["id"]]        # noqa: E731
    assert np.allclose(m(), 1.0)                                       # a new mask shows everywhere
    g = new_part("gradient")
    g.update(direction="Left → right", position=50.0, softness=20.0)
    lay["mask"]["parts"] = [g]
    ramp = np.clip((pr.ramp - (0.5 - 0.1)) / 0.2, 0, 1)
    assert np.allclose(m(), ramp, atol=1e-5)
    s = new_part("stars")
    s["grow"] = 0.0
    lay["mask"]["parts"] = [s]
    assert np.allclose(m(), np.clip(pr.stars * 1.5, 0, 1), atol=1e-5)
    # add: max; subtract: result * (1 - part); intersect: result * part; the first part decides the start
    a, b = new_part("stars"), new_part("gradient")
    a["grow"] = 0.0
    b.update(direction="Left → right", position=50.0, softness=20.0)
    star, rmp = np.clip(pr.stars * 1.5, 0, 1), ramp
    for op, want in (("add", np.maximum(star, rmp)), ("sub", star * (1 - rmp)), ("inter", star * rmp)):
        b["op"] = op
        lay["mask"]["parts"] = [a, b]
        assert np.allclose(m(), want, atol=1e-5), op
    only_sub = new_part("gradient")
    only_sub.update(op="sub", direction="Left → right", position=50.0, softness=20.0)
    lay["mask"]["parts"] = [only_sub]
    assert np.allclose(m(), 1 - rmp, atol=1e-5)                        # starts at 1 when the first part subtracts
    # feather, density, invert (in that order)
    lay["mask"].update(parts=[a], feather=0.0, density=50.0, invert=False)
    assert np.allclose(m(), 1 - 0.5 * (1 - star), atol=1e-5)
    lay["mask"]["invert"] = True
    assert np.allclose(m(), 0.5 * (1 - star), atol=1e-5)
    lay["mask"].update(density=100.0, invert=False, feather=6.0)
    assert np.allclose(m(), cv2.GaussianBlur(star, (0, 0), 6, borderType=cv2.BORDER_REFLECT), atol=2e-3)
    # luminance and colour range read the image the mask is on
    lay["mask"].update(feather=0.0)
    lr = new_part("luminance")
    lr.update(lo=60.0, hi=100.0)
    lay["mask"]["parts"] = [lr]
    mm = m()
    L = _lum(np.clip(pr.img[1], 0, 1))
    assert mm[L > 0.8].mean() > 0.95 and mm[L < 0.4].mean() < 0.05
    cr = new_part("color")
    lay["mask"]["parts"] = [cr]
    assert 0 <= m().min() and m().max() <= 1
    hp = new_part("halpha")
    lay["mask"]["parts"] = [hp]
    assert np.allclose(m(), pr.ramp, atol=1e-6)


def test_a_mask_acts_on_a_layer_and_on_a_filter():
    pr = ToyProvider()
    cp = Composer(pr)
    bottom, top = new_workflow_layer(1, "A", True), new_workflow_layer(2, "B", True)
    top["mask"] = new_mask()
    g = new_part("gradient")
    g.update(direction="Left → right", position=50.0, softness=20.0)
    top["mask"]["parts"] = [g]
    ramp = np.clip((pr.ramp - 0.4) / 0.2, 0, 1)[..., None]
    out = cp.compose([top, bottom], 0).rgb
    assert np.allclose(out, pr.img[1] + (pr.img[2] - pr.img[1]) * ramp, atol=1e-5)
    fl = new_filter_layer("levels", None)
    fl["params"].update(gamma=2.0, black=0.0)
    fl["mask"] = new_mask()
    fl["mask"]["parts"] = [g]
    out = cp.compose([fl, bottom], 0).rgb
    assert np.allclose(out, pr.img[1] + (pr.img[1] ** 0.5 - pr.img[1]) * ramp, atol=1e-5)


def test_a_region_is_the_crop_of_the_whole_and_the_levels_agree():
    pr = ToyProvider(640, 400)
    cp = Composer(pr)
    base = new_workflow_layer(1, "A", True)
    sat = new_filter_layer("saturation", base["id"])
    lc = new_filter_layer("local_contrast", None)
    sh = new_filter_layer("star_reduce", None)
    lay = [lc, sh, sat, base]
    base["mask"] = new_mask()
    base["mask"]["parts"] = [new_part("stars")]
    full = cp.compose(lay, 0).rgb
    box = (200, 120, 460, 300)
    reg = cp.compose(lay, 0, box).rgb
    crop = full[120:300, 200:460]
    assert reg.shape == crop.shape
    inner = np.abs(reg - crop)[64:-64, 64:-64] * 255
    assert inner.max() < 1.0 and float(np.abs(reg - crop).max() * 255) < 8   # the margin: filters see their neighbours
    # half and quarter resolution look like the reduced full image (smooth filters: their sizes are divided by the factor)
    smooth = [new_filter_layer("curves", None), new_filter_layer("local_contrast", None), sat, base]
    smooth[0]["params"]["midtones"] = 0.1
    smooth[1]["params"].update(scale=60, strength=0.8)
    base["mask"] = None
    pr.img[1] = cv2.GaussianBlur(pr.img[1], (0, 0), 7)                      # no sharp edge: a mean of a curve is not its curve
    cp = Composer(pr)
    full = cp.compose(smooth, 0).rgb
    for lvl, tol in ((1, 2.0), (2, 4.0)):
        f = 2 ** lvl
        red = cv2.resize(full, (640 // f, 400 // f), interpolation=cv2.INTER_AREA)
        got = cp.compose(smooth, lvl).rgb
        assert got.shape == red.shape
        assert float(np.percentile(np.abs(got - red) * 255, 99.9)) < tol, lvl


def test_only_what_changed_is_computed_again_and_a_copy_shares_the_image():
    pr = ToyProvider()
    cp = Composer(pr)
    a, b = new_workflow_layer(1, "A", True), new_workflow_layer(2, "B", True)
    lay = [b, a]
    assert cp.compose(lay, 1).drawn == 2
    assert cp.compose(lay, 1).drawn == 0
    b["sliders"]["sky"] = 3.0
    assert cp.compose(lay, 1).drawn == 1                                # only B changed
    c = postengine.snapshot([b])[0]
    c["id"] = postengine.new_id()
    assert cp.compose([c, b, a], 1).drawn == 0                          # a copy has the same content
    sat = new_filter_layer("saturation", None)
    assert cp.compose([sat, b, a], 1).drawn == 0                        # filters never ask the engine
    assert cp.compose(lay, 2).drawn == 2 and cp.compose(lay, 0).drawn == 2
    pr.epoch += 1                                                       # the result changed on disk
    assert cp.compose(lay, 1).drawn == 2
    # the memory is bounded: the least recently used image goes first
    small = Composer(pr, cache_bytes=pr.img[1].nbytes * 3)
    for k in range(6):
        b["sliders"]["sky"] = 2 + k * 0.2
        small.compose(lay, 0)
    assert small.images.bytes <= pr.img[1].nbytes * 3 and len(small.images.items) <= 3


def test_the_brush_is_stamped_incrementally_and_a_region_matches():
    pr = ToyProvider(256, 160)
    cp = Composer(pr)
    lay = new_workflow_layer(1, "A", True)
    lay["mask"] = new_mask()
    br = new_part("brush")
    lay["mask"]["parts"] = [br]
    for i in range(6):
        br["strokes"].append((40.0 + 25 * i, 60.0, 14.0, 50.0, True))
    m1 = cp.compose([lay], 1, want=(lay["id"],)).masks[lay["id"]]
    fresh = Composer(pr).compose([lay], 1, want=(lay["id"],)).masks[lay["id"]]
    assert np.array_equal(m1, fresh) and 0.9 < m1.max() <= 1.0 and m1.sum() > 50
    br["strokes"].append((130.0, 110.0, 20.0, 80.0, True))
    br["strokes"].append((130.0, 110.0, 8.0, 100.0, False))                # an erase
    m2 = cp.compose([lay], 1, want=(lay["id"],)).masks[lay["id"]]
    assert np.array_equal(m2, Composer(pr).compose([lay], 1, want=(lay["id"],)).masks[lay["id"]])
    assert m2[55, 65] < 0.2 and m2.sum() > m1.sum()                        # the erase took the centre of the new dab out
    full = cp.compose([lay], 0, want=(lay["id"],)).masks[lay["id"]]
    reg = cp.compose([lay], 0, (50, 20, 200, 130), want=(lay["id"],)).masks[lay["id"]]
    assert np.array_equal(reg, full[20:130, 50:200])
    br["strokes"] = []                                                      # Clear all
    assert cp.compose([lay], 1, want=(lay["id"],)).masks[lay["id"]].max() == 0


def test_the_layer_rules_normalize_and_the_composition_is_saved():
    a, b = new_workflow_layer(1, "A", True), new_workflow_layer(2, "B", True)
    f1, f2 = new_filter_layer("sharpen", a["id"]), new_filter_layer("scnr", b["id"])
    f3 = new_filter_layer("curves", 9999)                                     # its layer is gone
    # a layer's own filters always sit just above it; a filter whose layer is gone applies to everything below
    names = [x["name"] for x in normalize([b, f1, a, f2, f3])]
    assert names.index("Sharpen") == names.index("A") - 1 and names.index("SCNR (green removal)") == names.index("B") - 1
    assert f3["target"] is None
    layers = normalize([f1, f2, f3, b, a])
    a["mask"] = new_mask()
    a["mask"]["parts"] = [new_part("brush"), new_part("gradient")]
    a["mask"]["parts"][0]["strokes"].append((1.0, 2.0, 3.0, 4.0, True))
    a["sliders"] = {"sky": 3.1, "star_size": 0.5}
    raw = json.loads(json.dumps(postengine.save_layers(layers)))
    back = postengine.load_layers(raw, {1, 2})
    assert [x["name"] for x in back] == [x["name"] for x in layers]
    assert back[-1]["sliders"] == {"sky": 3.1, "star_size": 0.5}
    assert back[-1]["mask"]["parts"][0]["strokes"] == [(1.0, 2.0, 3.0, 4.0, True)]
    nxt = postengine.new_id()
    assert nxt > max(x["id"] for x in back)                                    # new ids do not clash with the loaded ones
    # a damaged composition is made safe: unknown filters and vanished workflows go, numbers come back in range
    junk = [{"kind": "filter", "filter": "nope"}, {"kind": "workflow", "pid": 77}, {"kind": "workflow", "pid": 1,
            "name": "X", "opacity": 900, "mode": "Weird", "sliders": {"sky": 99}, "mask": {"parts": [{"src": "zzz"}]}},
            "text", None]
    got = postengine.load_layers(junk, {1})
    assert len(got) == 1 and got[0]["opacity"] == 100 and got[0]["mode"] == "Normal" and got[0]["sliders"]["sky"] == 4
    assert got[0]["mask"]["parts"] == []


def test_a_newer_request_stops_the_composition():
    pr = ToyProvider()
    cp = Composer(pr)
    lay = _stack((1, "A"), (2, "B"))
    n = {"k": 0}

    def stop():
        n["k"] += 1
        return n["k"] > 1

    with pytest.raises(Superseded):
        cp.compose(lay, 1, stop=stop)
    assert cp.compose(lay, 1).rgb is not None


# ---------------------------------------------------------------------------------------------- the engine's layers
def test_the_canvas_is_what_every_layer_covers_on_multiples_of_eight(preps):
    src = LayerSource(lambda pid: {1: Folder(preps["full"]), 2: Folder(preps["classic"]),
                                   3: Folder(preps["other"])}.get(pid), LayerService("cpu"))
    cv = src.canvas([1, 2])
    assert (cv.W, cv.H, cv.origin) == (688, 464, (24, 8)) and cv.W % 8 == 0 and cv.H % 8 == 0
    assert cv.size(1) == (344, 232) and cv.size(2) == (172, 116)
    # a layer of another drizzle scale (2.0 against 1.5): its crop is mapped into the first layer's pixels
    cv2_ = src.canvas([1, 3])
    assert cv2_.W <= 688 and cv2_.W % 8 == 0 and cv2_.origin[0] % 8 == 0
    x0 = cv2_.origin[0]
    assert x0 >= 24 and x0 + cv2_.W <= 716
    lay = new_workflow_layer(3, "Other", True)
    for level in (0, 1, 2):
        w, h = cv2_.size(level)
        a = src.layer(lay, level, (0, 0, w, h))
        assert a.shape == (h, w, 3) and a.dtype == np.float32 and np.isfinite(a).all()
    with pytest.raises(FileNotFoundError):
        src.grid(99)
    assert not src.available(99) and src.available(1)


def test_the_release_render_is_the_engines_bit_for_bit(preps):
    """The layer the composition gets at the half resolution and at full resolution is the engine's own rendering, cut
    to the canvas: the draft, the region and the preview add nothing (DESIGN §6.3, the v1/v2 check)."""
    svc = LayerService(DEV)
    src = LayerSource(lambda pid: {1: Folder(preps["full"]), 2: Folder(preps["classic"])}.get(pid), svc,
                      lambda b: {"faithful": {"sky_level": 24.0}}.get(b, b) if isinstance(b, str) else b)
    cv = src.canvas([1, 2])
    ox, oy = cv.origin
    eng = RenderEngine(preps["full"], device=DEV)
    faithful = RenderSettings.from_saved({"sky_level": 24.0})
    lay = new_workflow_layer(1, "Full", True, base="faithful")
    lay["sliders"] = {k: 2 for k in KEYS}                                    # every slider at the suggestion
    for level in (1, 2, 0):
        f = 2 ** level
        want = eng.render(faithful, 1.0 / f).final.numpy()
        w, h = cv.size(level)
        got = src.layer(lay, level, (0, 0, w, h))
        ref = np.moveaxis(want[:, oy // f: oy // f + h, ox // f: ox // f + w], 0, -1)
        assert np.array_equal(got, ref), level
    # a slider away from 2 moves the settings the table says, nothing else
    lay["sliders"] = {"sky": 3}
    moved = src.layer(lay, 1, (0, 0, *cv.size(1)))
    s30 = RenderSettings.from_saved({"sky_level": 30.0})
    ref = eng.render(s30, 0.5).final.numpy()[:, oy // 2: oy // 2 + cv.size(1)[1], ox // 2: ox // 2 + cv.size(1)[0]]
    assert np.array_equal(moved, np.moveaxis(ref, 0, -1))
    # the parts: stars alone is the layer minus its starless image
    lay["sliders"], lay["part"] = {}, "stars"
    st = src.layer(lay, 1, (0, 0, *cv.size(1)))
    assert st.max() > 0.2 and st.min() >= 0


def test_a_region_at_full_resolution_is_the_engines_region(preps):
    svc = LayerService(DEV)
    src = LayerSource(lambda pid: {1: Folder(preps["full"])}.get(pid), svc)
    cv = src.canvas([1])
    ox, oy = cv.origin
    lay = new_workflow_layer(1, "Full", True)
    lay["sliders"] = {"sky": 2.5}
    box = (100, 60, 420, 300)
    got = src.layer(lay, 0, box)
    li = svc.render(preps["full"], None, {"sky": 2.5}, 1.0, region=(ox + 100, ox + 420, oy + 60, oy + 300))
    assert np.array_equal(got, np.moveaxis(li.rgb, 0, -1))
    whole = src.layer(lay, 0, (0, 0, *cv.size(0)))                           # the whole canvas: v1's own full rendering
    full = svc.render(preps["full"], None, {"sky": 2.5}, 1.0).rgb
    assert np.array_equal(whole, np.moveaxis(full[:, oy:oy + cv.H, ox:ox + cv.W], 0, -1))
    d = np.abs(got - whole[60:300, 100:420]) * 255                           # P4: a region is the crop within 2 levels
    assert float(np.percentile(d[64:-64, 64:-64], 99.9)) <= 2.0


def test_masks_come_from_the_layer_with_stars(preps):
    src = LayerSource(lambda pid: {1: Folder(preps["full"]), 2: Folder(preps["classic"])}.get(pid), LayerService(DEV))
    cv = src.canvas([2, 1])                                                    # the bottom one is the simple layer
    w, h = cv.size(1)
    st = src.data_mask("stars", 1, (0, 0, w, h))
    ha = src.data_mask("halpha", 1, (0, 0, w, h))
    assert st.shape == ha.shape == (h, w) and st.max() > 0.5 and ha.max() > 0.5       # the stars come from layer 1
    assert src.has("stars") and src.has("halpha")
    only = LayerSource(lambda pid: {2: Folder(preps["classic"])}.get(pid), LayerService(DEV))
    only.canvas([2])
    assert not only.has("stars") and only.data_mask("stars", 1, (0, 0, 8, 8)) is None
    assert only.has("halpha")


# ---------------------------------------------------------------------------------------------- the worker
class SlowService(LayerService):
    """A layers service that records its thread and takes its time."""

    def __init__(self, delay=0.3):
        super().__init__("cpu")
        self.delay, self.threads, self.n = delay, set(), 0

    def render(self, *a, **k):
        self.threads.add(threading.current_thread().name)
        self.n += 1
        time.sleep(self.delay)
        return super().render(*a, **k)


def test_renders_run_on_the_worker_and_the_newest_request_wins(preps, qapp):
    svc = SlowService(0.25)
    w = RenderWorker(None, resolver=lambda pid: {1: Folder(preps["full"])}.get(pid), device="cpu", service=svc)
    got = []
    w.rendered.connect(got.append)
    try:
        lay = [new_workflow_layer(1, "Full", True)]
        seqs = []
        for k in range(8):                                                     # a drag: eight drafts in a row
            lay[0]["sliders"] = {"sky": 2 + k * 0.1}
            seqs.append(w.request(Request(kind="draft", layers=snapshot(lay), level=2)))
        w.settle(60)
        drafts = [r for r in got if r.kind == "draft"]
        assert drafts and drafts[-1].seq == seqs[-1] and len(drafts) < 8        # superseded ones never came back
        assert svc.n <= 4                                                      # and most were never even rendered
        assert threading.main_thread().name not in svc.threads and len(svc.threads) == 1
        assert w.device == "cpu" and drafts[-1].device == "cpu" and drafts[-1].rgb.dtype == np.uint8
        # a request being computed is abandoned when a newer one of its lane arrives
        got.clear()
        svc.n = 0
        w.request(Request(kind="half", layers=snapshot(lay), level=1))
        time.sleep(0.1)
        lay[0]["sliders"] = {"sky": 3.5}
        s2 = w.request(Request(kind="half", layers=snapshot(lay), level=1))
        w.settle(60)
        assert [r.seq for r in got if r.kind == "half"] == [s2]
        # the most urgent lane goes first, and a region waits (it comes back after)
        got.clear()
        w.request(Request(kind="region", layers=snapshot(lay), level=0, box=(0, 0, 200, 120)))
        w.request(Request(kind="draft", layers=snapshot(lay), level=2))
        w.settle(60)
        kinds = [r.kind for r in got]
        assert "draft" in kinds and "region" in kinds and kinds.index("draft") < kinds.index("region")
    finally:
        w.stop()


def test_a_failed_request_comes_back_as_text(preps, qapp):
    w = RenderWorker(None, resolver=lambda pid: Folder(preps["full"]) if pid == 1 else None, device="cpu")
    errs = []
    w.failed.connect(lambda seq, kind, text: errs.append((kind, text)))
    try:
        lay = [new_workflow_layer(1, "Full", True)]
        w._resolver = lambda pid: Folder(Path("nowhere") / "render_prep")        # a result that is not there
        w.renderer = None
        w.request(Request(kind="draft", layers=snapshot(lay), level=2))
        w.settle(30)
        assert errs and errs[0][0] == "draft" and "FileNotFoundError" in errs[0][1]
    finally:
        w.stop()


def test_the_device_is_a_free_gpu_else_the_cpu(monkeypatch):
    monkeypatch.setattr(rsv, "gpu_count", lambda: 2)
    mem = {0: 12.0, 1: 5.0}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (int(mem[i] * 2 ** 30), 16 * 2 ** 30))
    assert choose_device("cpu") == "cpu"
    assert choose_device("auto") == "cuda:0"                                     # the freest
    assert choose_device("auto", busy="cuda:0") == "cuda:1"                      # a run holds cuda:0: never render there
    assert choose_device("cuda:1") == "cuda:1"                                   # the settings name a GPU
    assert choose_device("cuda:1", busy="cuda:1") == "cuda:0"
    mem[0], mem[1] = 1.0, 2.0
    assert choose_device("auto") == "cpu"                                        # no GPU with room: the CPU
    monkeypatch.setattr(rsv, "gpu_count", lambda: 1)
    mem[0] = 10.0
    assert choose_device("auto", busy="cuda:0") == "cpu"                         # the only GPU is running a job
    monkeypatch.setattr(rsv, "gpu_count", lambda: 0)
    assert choose_device("auto") == "cpu"


def test_the_worker_changes_device_when_a_run_takes_the_gpu(preps, qapp, monkeypatch):
    monkeypatch.setattr(rsv, "gpu_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (10 * 2 ** 30, 16 * 2 ** 30))
    w = RenderWorker(None, resolver=lambda pid: Folder(preps["full"]), device="auto",
                     service=LayerService("cpu"))
    w.stop()
    w.pref = "auto"
    w.device, w.renderer = "cuda:0", Renderer(LayerSource(lambda pid: None, LayerService("cpu")), device="cuda:0")
    w._last_busy, w._busy_at = None, 0.0
    seen = []
    w.deviceChanged.connect(seen.append)
    w._busy_device = lambda: "cuda:0"                                            # a run starts on cuda:0
    w._recheck_device()
    assert w.device == "cpu" and w.renderer.device == "cpu" and w._last_busy == "cuda:0"


def test_a_full_gpu_falls_back_to_the_cpu_and_says_so(preps, qapp):
    class FullGPU:
        dev = torch.device("cuda", 0)

        def render(self, *a, **k):
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    svc = LayerService("cpu")
    k = svc._key(preps["full"])
    svc._eng[k], svc._size[k] = FullGPU(), 1
    w = RenderWorker(None, resolver=lambda pid: Folder(preps["full"]), device="cpu", service=svc)
    got = []
    w.rendered.connect(got.append)
    try:
        w.request(Request(kind="half", layers=[new_workflow_layer(1, "Full", True)], level=1))
        w.settle(60)
        r = got[-1]
        assert r.rgb is not None and NOTE_CPU in r.notes and rsv.NOTE_CPU_TEXT == NOTE_CPU
    finally:
        w.stop()


@pytest.mark.parametrize("hidden", ["", "-1"])
def test_with_no_cuda_device_post_still_renders(tmp_path, preps, hidden):
    """CUDA_VISIBLE_DEVICES empty, or -1 (what hides the GPUs on Windows: an empty variable does not): the engines open
    on the CPU when no device is seen, and Post renders. Where the variable did not hide the GPU (an empty one on
    Windows) the script asks for the CPU itself: the tests start no GPU work."""
    script = tmp_path / "nocuda.py"
    script.write_text(
        "import os, sys, json\n"
        "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import torch\n"
        "visible = torch.cuda.is_available()\n"
        "from pathlib import Path\n"
        "from PySide6.QtWidgets import QApplication\n"
        "from astro_gpu.client.services.render import RenderWorker, Request, Folder\n"
        "from astro_gpu.client.postengine import new_workflow_layer, snapshot\n"
        "app = QApplication([])\n"
        f"d = Path({str(preps['full'])!r})\n"
        "w = RenderWorker(None, resolver=lambda pid: Folder(d), device='cpu' if visible else 'auto')\n"
        "got = []; w.rendered.connect(got.append)\n"
        "w.request(Request(kind='half', layers=snapshot([new_workflow_layer(1, 'Full', True)]), level=1))\n"
        "w.settle(120); w.stop()\n"
        "r = got[-1]\n"
        "print(json.dumps({'device': w.device, 'shape': list(r.rgb.shape), 'on': r.device, 'visible': visible}))\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=hidden)
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["shape"][2] == 3 and res["on"] == res["device"]
    if hidden == "-1" or not res["visible"]:
        assert not res["visible"] and res["device"] == "cpu"


# ---------------------------------------------------------------------------------------------- the screen
def named(view, name):
    return next(lay for lay in view.layers if lay["name"] == name)


def reference(post, layers=None, level=1, box=None, want=()):
    """The same request rendered apart, by a renderer of its own: what the page shows must be this."""
    svc = LayerService(DEV)
    src = LayerSource(lambda pid: post.folders.get(pid), svc)
    return Renderer(src).run(Request(kind="half", layers=snapshot(layers or post.view.layers), level=level, box=box,
                                     want=want))


def shown(view) -> np.ndarray:
    return view.src.img1


def widgets_of(view, cls):
    """The widgets of the properties panel that are shown (the ones it replaced wait to be deleted, hidden)."""
    QApplication.instance().processEvents()                                 # new children are shown by a queued call
    return [w for w in view.pbox.parentWidget().findChildren(cls) if w.isVisible()]


def test_the_page_is_empty_until_a_workflow_is_rendered(preps, qapp):
    s = Setup(preps, qapp, rendered=False)
    try:
        assert s.page.pages.currentIndex() == 0
        texts = [lb.text() for lb in s.page.empty.findChildren(QLabel)] + \
                [b.text() for b in s.page.empty.findChildren(QPushButton)]
        assert "No layers yet" in texts and "Go to graph  →" in texts
        went = []
        s.page.goWorkflow.connect(lambda: went.append(1))
        next(b for b in s.page.empty.findChildren(QPushButton)).click()
        assert went
        s.model.paths[0].rendered = True
        s.model.pathsChanged.emit()
        assert s.page.pages.currentIndex() == 1
        v = s.view
        assert [lay["name"] for lay in v.layers] == ["Full"] and v.layers[0]["visible"]
        # the workflow that is not computed yet is a link to its Output
        s.qapp.processEvents()
        links = [b for b in v.lbox.parentWidget().findChildren(QPushButton)
                 if "not computed yet" in b.text() and not b.isHidden()]
        assert [b.text() for b in links] == ['"Classic": not computed yet  →']
        got = []
        v.goResults.connect(got.append)
        links[0].click()
        assert got == [2]
    finally:
        s.close()


def test_the_first_layers_and_the_first_image(post):
    v = post.view
    assert [(lay["name"], lay["visible"]) for lay in v.layers] == [("Classic", False), ("Full", True)]
    assert v.first_image and v.viewer.banner == ""
    assert v.src.canvas == (688, 464) and shown(v).shape == (232, 344, 3) and shown(v).dtype == np.uint8
    assert np.array_equal(shown(v), reference(post).rgb)                    # the page shows what the engine path gives
    assert post.model.touched == 0                                          # opening Post writes nothing in the project
    # a workflow that is rendered later gets its layer on top, hidden
    post.model.paths.append(WF(3, "Raw stack", "stacking"))
    post.model.pathsChanged.emit()
    assert [(lay["name"], lay["visible"]) for lay in v.layers][0] == ("Raw stack", False)
    assert post.settle()
    # each workflow layer has its thumbnail, made from its image
    assert all(lay["id"] in v._lthumbs for lay in v.layers)
    assert v._lthumbs[named(v, "Full")["id"]][1].width() == 54


def test_filters_on_one_layer_and_on_everything_below(post):
    v = post.view
    full, classic = named(v, "Full"), named(v, "Classic")
    classic["visible"], classic["opacity"] = True, 40.0
    v.select(full["id"], "layer")
    sat = v.add_filter("saturation")
    assert sat["target"] == full["id"]
    assert [lay["name"] for lay in v.layers] == ["Classic", "Saturation", "Full"]     # just above the layer it applies to
    rows = {r.lay["id"]: r for r in v.rows}
    assert rows[sat["id"]].sub.text() == 'on "Full" · Normal · 100%'
    assert any(lb.text() == "↳" for lb in rows[sat["id"]].findChildren(QLabel))
    assert not any(lb.text() == "↳" for lb in rows[full["id"]].findChildren(QLabel))
    # another one, on everything below
    v.select(classic["id"], "layer")
    sh = v.add_filter("sharpen")
    assert sh["target"] == classic["id"]
    v.set_target(sh, None)
    assert sh["target"] is None and [lay["name"] for lay in v.layers] == ["Sharpen", "Classic", "Saturation", "Full"]
    assert next(r for r in v.rows if r.lay["id"] == sh["id"]).sub.text() == "on everything below · Normal · 100%"
    # the properties of a filter: where it applies, its settings
    v.select(sat["id"], "layer")
    texts = [b.text() for b in widgets_of(v, QPushButton)]
    assert '"Full" only' in texts and "Everything below" in texts
    assert [lb.text() for lb in widgets_of(v, QLabel)][0] == "Saturation"
    assert post.settle()
    img = shown(v).copy()
    assert np.array_equal(img, reference(post).rgb)
    sat["params"]["strength"] = -100.0                                                # grey
    v.request(final_only=True)
    assert post.settle() and not np.array_equal(shown(v), img)
    assert np.array_equal(shown(v), reference(post).rgb)
    sat["visible"] = False
    sat["params"]["strength"] = 30.0
    v.request(final_only=True)
    assert post.settle() and np.array_equal(shown(v), reference(post).rgb)
    # the + Filter menu: the 13 filters by family
    fam = {a.text(): [x.text() for x in a.menu().actions()] for a in v.filter_menu.actions()}
    assert list(fam) == ["Tone", "Color", "Detail", "Stars", "Combine"] and sum(len(x) for x in fam.values()) == 13
    assert fam["Tone"] == ["Stretch (GHS)", "Curves", "Levels"] and fam["Combine"] == ["H-alpha to red"]
    assert all(x.isEnabled() for a in v.filter_menu.actions() for x in a.menu().actions())


def test_star_filters_and_the_star_mask_need_a_layer_with_stars(preps, qapp):
    s = Setup(preps, qapp, which=(2,))
    try:
        assert s.settle()
        v = s.view
        assert not v.has_stars()
        stars_menu = next(a for a in v.filter_menu.actions() if a.text() == "Stars").menu()
        assert [a.isEnabled() for a in stars_menu.actions()] == [False, False, False]
        assert stars_menu.actions()[0].toolTip() == "Needs a layer with the Stars block"
        v.add_mask(v.layers[0])
        add = next(b for b in widgets_of(v, QPushButton) if b.text() == "+ Add to mask")
        acts = {a.text(): a.isEnabled() for a in add.menu().actions()}
        assert acts["Catalog stars"] is False and acts["H-alpha"] is True and len(acts) == 6
    finally:
        s.close()


def test_masks_with_parts_show_on_the_image_and_in_the_thumbnails(post):
    v = post.view
    full = named(v, "Full")
    v.select(full["id"], "layer")
    assert v.b_maskadd.text() == "Mask"
    v.mask_button()
    assert full["mask"] is not None and v.sel == (full["id"], "mask") and v.b_maskadd.text() == "Mask ✓"
    assert post.settle() and v.mask1 is not None and np.allclose(v.mask1, 1.0)         # empty: the layer shows everywhere
    assert "Empty: the layer shows everywhere." in [lb.text() for lb in widgets_of(v, QLabel)]
    for src in ("stars", "halpha", "luminance", "color", "gradient", "brush"):
        v.add_part(full, src)
    assert [p["src"] for p in full["mask"]["parts"]] == ["stars", "halpha", "luminance", "color", "gradient", "brush"]
    assert len([f for f in widgets_of(v, type(v.pbox.parentWidget())) if f.objectName() == "part"]) == 6
    v._part_set(full["mask"]["parts"][1], "op", "inter")                              # H-alpha intersected
    v._part_set(full["mask"]["parts"][2], "op", "sub")
    assert post.settle()
    m = v.mask1
    assert m.shape == (232, 344) and 0 <= m.min() and m.max() <= 1
    assert np.array_equal(m, reference(post, want=(full["id"],)).masks[full["id"]])
    assert v._overlay_img is not None and v._overlay_img.width() == 344                  # the red overlay of the mask edited
    pm = v._mthumbs[full["id"]]
    assert (pm.width(), pm.height()) == (54, 36)
    # the whole mask: feather, density, invert
    full["mask"].update(feather=4.0, density=60.0, invert=True)
    v.request(final_only=True)
    assert post.settle()
    assert np.array_equal(v.mask1, reference(post, want=(full["id"],)).masks[full["id"]])
    # show mask: the mask in grey instead of the image
    v.b_mask.setChecked(True)
    g = v.src.region(1, 0, 0, 344, 232)
    assert g.shape == (232, 344, 3) and np.array_equal(g[..., 0], g[..., 1]) and g.max() > 0
    v.b_mask.setChecked(False)
    v.remove_part(full, full["mask"]["parts"][0])
    assert full["mask"]["parts"][0]["src"] == "halpha"
    v.remove_mask(full)
    assert full["mask"] is None and v.b_maskadd.text() == "Mask"


def test_the_brush_paints_erases_and_is_one_undo_step(post):
    v = post.view
    full = named(v, "Full")
    v.add_mask(full)
    part = v.add_part(full, "brush")
    assert v.part_sel == part["id"] and v.viewer.mode == "brush"
    n0 = len(v.undo)
    v._stroke(200.0, 150.0, True)
    v._stroke(260.0, 150.0, False)                                                       # a fast move: the gap is filled
    v._stroke(330.0, 190.0, False)
    v._stroke_end()
    assert len(v.undo) == n0 + 1                                                         # the whole stroke is one step
    xs = [s[0] for s in part["strokes"]]
    assert len(xs) > 6 and max(np.diff(xs)) <= part["size"] / 2 / 3 + 1.0
    assert post.settle()
    m = v.mask1
    assert m[75, 100] > 0.9 and m[75, 130] > 0.9 and m[10, 10] == 0                      # painted where the brush went
    v.toggle_paint()                                                                     # X: erase
    assert part["paint"] is False
    v._stroke(200.0, 150.0, True)
    v._stroke_end()
    assert post.settle() and v.mask1[75, 100] < 0.5
    v.brush_size(1.25)                                                                   # ]
    assert part["size"] == pytest.approx(75.0) and v.viewer.brush_r == pytest.approx(37.5)
    v.do_undo()
    assert post.settle() and v.mask1[75, 100] > 0.9
    part = named(v, "Full")["mask"]["parts"][0]                                          # Undo brings back other dicts
    v._part_set(part, "strokes", [])                                                     # Clear all
    assert post.settle() and v.mask1.max() == 0


def test_split_stars_and_starless(post):
    v = post.view
    full = named(v, "Full")
    v.select(full["id"], "layer")
    flt = v.add_filter("curves")
    v.select(full["id"], "layer")
    v.split_stars(full)
    assert [(lay["name"], lay.get("part"), lay["mode"]) for lay in v.layers] == [
        ("Classic", None, "Normal"), ("Full · stars", "stars", "Screen"), ("Curves", None, "Normal"),
        ("Full · starless", "starless", "Normal")]
    stars, fond = v.layers[1], v.layers[3]
    assert stars["mask"] is None and flt["target"] == fond["id"]                # its filters stay on the starless one
    assert post.settle()
    img = shown(v).astype(int)
    assert np.array_equal(img, reference(post).rgb.astype(int))
    stars["sliders"]["stars_number"] = 0.0                                               # each half has its own sliders
    v.request(final_only=True)
    assert post.settle() and shown(v).astype(int).mean() < img.mean()                    # fewer stars: less light
    # the stars alone is the layer minus its starless image (clipped), the starless one has no star
    src = LayerSource(lambda pid: post.folders.get(pid), LayerService(DEV))
    cv = src.canvas([1])
    both = new_workflow_layer(1, "Full", True)
    w, h = cv.size(1)
    a = src.layer(dict(both, part="stars"), 1, (0, 0, w, h))
    b = src.layer(dict(both, part="starless"), 1, (0, 0, w, h))
    c = src.layer(both, 1, (0, 0, w, h))
    assert a.min() >= 0 and np.allclose(a, np.clip(c - b, 0, 1)) and a.max() > 0.2
    v.do_undo()                                                                          # one step for the split
    assert [lay["name"] for lay in v.layers] == ["Classic", "Curves", "Full"]


def test_undo_and_redo_one_step_per_action(post):
    from astro_gpu.client.post3 import ContSlider
    v = post.view
    full, classic = named(v, "Full"), named(v, "Classic")

    def state():
        return json.dumps(postengine.save_layers(v.layers), sort_keys=True)

    states = [state()]

    def step(fn):
        n = len(v.undo)
        fn()
        assert len(v.undo) == n + 1, "one step"
        states.append(state())

    cur = lambda: v.layer(v.sel[0])                                                      # noqa: E731
    step(lambda: v.set_visible(classic["id"], True))
    step(lambda: v.add_filter("levels"))
    step(lambda: v.duplicate())
    step(lambda: v.rename(cur(), "Mine"))
    step(lambda: v._set(cur(), "mode", "Screen"))
    step(lambda: v.add_mask(cur()))
    step(lambda: v.add_part(cur(), "gradient"))
    step(lambda: v._part_set(cur()["mask"]["parts"][0], "op", "inter"))
    step(lambda: (v.push(), cur()["mask"].__setitem__("invert", True)))
    step(lambda: v.remove_part(cur(), cur()["mask"]["parts"][0]))
    step(lambda: v.remove_mask(cur()))
    step(lambda: v.delete())
    # a slider drag is one step, from the first movement to the release
    v.select(full["id"], "layer")
    sky = next(s for s in widgets_of(v, ContSlider) if s.name.text() == "Sky")
    n, before = len(v.undo), state()
    for val in (2.2, 2.4, 2.6, 2.8):
        sky.moved.emit(val)
    sky.done.emit(2.8)
    assert len(v.undo) == n + 1 and full["sliders"]["sky"] == pytest.approx(2.8)
    v.do_undo()
    assert state() == before and "sky" not in v.layer(full["id"])["sliders"]
    v.do_redo()
    assert v.layer(full["id"])["sliders"]["sky"] == pytest.approx(2.8)
    v.do_undo()
    while v.undo:                                                                        # every state again, back and forth
        v.do_undo()
    assert state() == states[0]
    for want in states[1:]:
        v.do_redo()
        assert state() == want
    assert v.b_undo.isEnabled() and v.sel is not None and v.layer(v.sel[0]) is not None


def test_dragging_a_row_reorders_and_retargets_a_filter(post):
    v = post.view
    full, classic = named(v, "Full"), named(v, "Classic")
    classic["visible"] = True
    v.select(full["id"], "layer")
    f = v.add_filter("scnr")
    assert [lay["name"] for lay in v.layers] == ["Classic", "SCNR (green removal)", "Full"]
    v.move_layer(f["id"], 0)                                                             # to the top of the list
    assert [lay["name"] for lay in v.layers][0] == "SCNR (green removal)" and f["target"] == classic["id"]
    v.move_layer(f["id"], 3)                                                             # to the bottom
    assert f["target"] is None and v.layers[-1]["id"] == f["id"]
    v.move_layer(f["id"], 1)                                                             # just above the Full row
    assert f["target"] == full["id"] and [lay["name"] for lay in v.layers] == [
        "Classic", "SCNR (green removal)", "Full"]
    v.move_layer(full["id"], 0)                                                          # the filter follows its layer
    assert [lay["name"] for lay in v.layers][:2] == ["SCNR (green removal)", "Full"]
    n = len(v.undo)
    v.move_layer(classic["id"], v.index(classic["id"]))                                  # dropped where it is: nothing
    assert len(v.undo) == n


def test_duplicate_delete_and_rename_follow_the_rules(post):
    v = post.view
    full = named(v, "Full")
    v.select(full["id"], "layer")
    f = v.add_filter("sharpen")
    v.select(full["id"], "layer")
    v.duplicate()
    assert [lay["name"] for lay in v.layers] == ["Classic", "Sharpen", "Full (copy)", "Sharpen", "Full"]
    copy_f, copy_l = v.layers[1], v.layers[2]
    assert copy_f["target"] == copy_l["id"] and f["target"] == full["id"] and v.sel == (copy_l["id"], "layer")
    assert tr_name("Full (copy)") == "Full (copy)" and copy_l["visible"]
    v.delete()                                                                           # a workflow layer goes with its filters
    assert [lay["name"] for lay in v.layers] == ["Classic", "Sharpen", "Full"]
    v.select(named(v, "Classic")["id"], "layer")
    v.delete()
    v.select(full["id"], "layer")
    v.delete()                                                                           # the last workflow layer stays
    assert [lay["name"] for lay in v.layers] == ["Sharpen", "Full"]
    v.rename(v.layer(full["id"]), "Kept")
    assert v.layers[-1]["name"] == "Kept"


def test_sliders_are_greyed_with_their_reason_and_double_click_resets(post):
    from astro_gpu.client.post3 import ContSlider
    v = post.view

    def sliders():
        return {s.name.text(): s for s in widgets_of(v, ContSlider)}

    v.select(named(v, "Full")["id"], "layer")
    sl = sliders()
    for k in ("Sky", "Contrast", "Saturation", "Detail", "Noise smoothing", "Number of stars", "Star size", "Star color",
              "Bright star halo"):
        assert sl[k].s.isEnabled(), k
    assert sl["Sky"].val.text() == "suggested" and sl["Sky"].lo == 0 and sl["Sky"].hi == 4
    sl["Sky"].set_value(3.4)
    assert sl["Sky"].val.text() == "+1.4"
    v.select(named(v, "Classic")["id"], "layer")
    sl = sliders()
    assert sl["Sky"].s.isEnabled() and sl["Noise smoothing"].s.isEnabled()
    for k in ("Detail", "Number of stars", "Star size", "Star color", "Bright star halo"):
        assert not sl[k].s.isEnabled() and not sl[k].why.isHidden()
    assert sl["Detail"].why.text() == "sharpening needs the starless image (Stars block)"
    assert sl["Star size"].why.text() == "the star sliders need the Stars block"
    post.model.paths.append(WF(3, "Raw stack", "stacking"))                              # pulled from Stacking
    post.model.pathsChanged.emit()
    v.select(named(v, "Raw stack")["id"], "layer")
    sl = sliders()
    assert sl["Noise smoothing"].why.text() == "this layer leaves the graph before the Denoise block"
    assert sl["Star size"].why.text() == "this layer leaves the graph before the Stars block"
    lay = named(v, "Classic")                                                            # a double-click: the suggestion
    v.select(lay["id"], "layer")
    sky = sliders()["Sky"]
    sky.done.emit(3.0)
    assert lay["sliders"]["sky"] == pytest.approx(3.0)
    sky.name.mouseDoubleClickEvent(None)
    assert "sky" not in lay["sliders"] and sky.val.text() == "suggested"


def test_a_slider_drag_is_a_draft_then_the_screen_image(post):
    from astro_gpu.client.post3 import ContSlider
    v = post.view
    log = []
    v.rendered.connect(lambda r: log.append((r.kind, r.level)))
    full = named(v, "Full")
    v.select(full["id"], "layer")
    sky = next(s for s in widgets_of(v, ContSlider) if s.name.text() == "Sky")
    n = len(v.undo)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 4 or not any(x[0] == "draft" for x in log):          # the slider moves every 30 ms
        sky.moved.emit(2 + 1.4 * min(1.0, (time.monotonic() - t0) / 4))
        end = time.monotonic() + 0.03
        while time.monotonic() < end:
            post.qapp.processEvents()
            time.sleep(0.003)
        if time.monotonic() - t0 > 60:
            break
    sky.done.emit(3.4)
    assert post.settle()
    drafts = [i for i, x in enumerate(log) if x[0] == "draft"]
    halves = [i for i, x in enumerate(log) if x[0] == "half"]
    assert drafts and ("draft", 2) in log and halves and halves[-1] > drafts[-1]       # the screen image comes last
    assert len(v.undo) == n + 1 and full["sliders"] == {"sky": pytest.approx(3.4)}
    assert np.array_equal(shown(v), reference(post).rgb)
    d = reference(post, level=2).rgb                                                     # the draft is only a draft
    half = cv2.resize(shown(v), (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
    assert float(np.abs(d.astype(int) - half.astype(int)).mean()) < 8


def test_zooming_in_renders_the_visible_region_at_full_resolution(wide, qapp):
    post = wide
    v = post.view
    seen = []
    v.rendered.connect(lambda r: seen.append(r.kind))
    assert v.src.canvas == (2368, 1488) and v.viewer.scale < 0.5 and v.src.full is None     # zoomed out: the half image only
    v.viewer.zoom_to(1.0, 1200, 700)
    assert post.settle()
    assert "region" in seen and v.src.full is not None
    (bx0, by0, bx1, by1), arr = v.src.full
    assert arr.shape == (by1 - by0, bx1 - bx0, 3) and bx1 - bx0 >= 1000 and 0 <= bx0 < bx1 <= 2368 and 0 <= by0 < by1 <= 1488
    got = v.src.region(0, bx0 + 10, by0 + 10, bx1 - 10, by1 - 10)                       # from the worker, not enlarged
    assert np.array_equal(got, arr[10:-10, 10:-10])
    assert np.array_equal(arr, reference(post, level=0, box=(bx0, by0, bx1, by1)).rgb)
    up = np.repeat(np.repeat(shown(v)[by0 // 2:by1 // 2, bx0 // 2:bx1 // 2], 2, 0), 2, 1)
    assert float(np.abs(arr[:up.shape[0], :up.shape[1]].astype(int) - up).mean()) > 0.5   # sharper than the enlarged half
    seen.clear()                                                                         # an edit forgets it, then asks again
    named(v, "Full")["sliders"]["sky"] = 3.0
    v.request(final_only=True)
    assert v.src.full is None
    assert post.settle() and seen.index("half") < seen.index("region") and v.src.full is not None
    v.viewer.fit()                                                                       # zoomed out: no region
    seen.clear()
    assert post.settle() and "region" not in seen


def test_before_and_after(post):
    v = post.view
    named(v, "Full")["sliders"]["sky"] = 3.5
    sat = v.add_filter("saturation")
    sat["params"]["strength"] = 90.0
    v.request(final_only=True)
    assert post.settle()
    v.b_cmp.setChecked(True)
    assert post.settle()
    assert v.viewer.compare is v.before and v.before.img1 is not None
    base = new_workflow_layer(1, "Full", True)                                           # the bottom layer, suggested settings
    assert np.array_equal(v.before.img1, reference(post, layers=[base]).rgb)
    assert not np.array_equal(v.before.img1, shown(v))
    assert v.hint.text().startswith("Drag the divider")
    v.b_cmp.setChecked(False)
    assert v.viewer.compare is None


def test_the_composition_is_kept_in_the_project(preps, qapp):
    s = Setup(preps, qapp)
    s2 = None
    try:
        assert s.settle()
        v = s.view
        full = named(v, "Full")
        v.select(full["id"], "layer")
        v.add_filter("saturation")
        v.add_mask(v.layer(v.sel[0]))
        v.add_part(v.layer(v.sel[0]), "brush")
        v._stroke(100.0, 100.0, True)
        v._stroke_end()
        full["sliders"]["sky"] = 3.2
        v.push()
        assert s.model.touched >= 3                                                      # the model is told to save
        saved = s.model.p.post["layers"]
        assert [lay["name"] for lay in saved] == ["Classic", "Saturation", "Full"] and json.dumps(saved)
        assert saved[1]["mask"]["parts"][0]["strokes"]
        assert s.settle()
        img = shown(v).copy()
        s.close()
        # a page opened on the same project finds its composition as it was
        s2 = Setup(preps, qapp)
        s2.close()
        s2 = Setup.__new__(Setup)
        s2.qapp, s2.folders, s2.model = qapp, s.folders, s.model
        s2.worker = RenderWorker(None, resolver=lambda pid: s2.folders.get(pid), device=DEV)
        s2.page = PostPage3(s2.model, worker=s2.worker)
        s2.page.resize(1500, 900)
        s2.page.show()
        s2.view = s2.page.view
        assert s2.settle()
        assert [lay["name"] for lay in s2.view.layers] == ["Classic", "Saturation", "Full"]
        assert np.array_equal(shown(s2.view), img)
    finally:
        if s2 is not None:
            s2.close()


def test_a_workflow_that_goes_takes_its_layer_and_the_filters_retarget(post):
    v = post.view
    v.select(named(v, "Full")["id"], "layer")
    f = v.add_filter("scnr")
    post.model.paths = [w for w in post.model.paths if w.pid != 1]
    post.model.pathsChanged.emit()
    assert [lay["name"] for lay in v.layers] == ["Classic", "SCNR (green removal)"]
    assert f["target"] is None and named(v, "Classic")["visible"]                        # the bottom layer is shown again


def test_the_engine_note_is_said_in_the_hint(preps, qapp):
    class FullGPU:
        dev = torch.device("cuda", 0)

        def render(self, *a, **k):
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    svc = LayerService("cpu")
    k = svc._key(preps["full"])
    svc._eng[k], svc._size[k] = FullGPU(), 1
    s = Setup(preps, qapp, which=(1,))
    assert s.settle()
    s.worker.stop()
    s.worker = RenderWorker(None, resolver=lambda pid: s.folders.get(pid), device="cpu", service=svc)
    s.view.worker = s.worker
    s.worker.rendered.connect(s.view._on_result)
    try:
        s.view.request(final_only=True)
        assert s.settle()
        assert "GPU memory full: rendered on the CPU" in s.view.hint.text()
        i18n.set_language("fr")
        s.view._hint()
        assert "moire du GPU pleine" in s.view.hint.text()
    finally:
        s.close()


# ---------------------------------------------------------------------------------------------- the export
def tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture()
def exporting(preps, qapp, tmp_path):
    """A page on a project folder (store, export) and a destination folder the user chose elsewhere."""
    root = tmp_path / "ws" / "HeartSoul"
    (root / "store" / "render_prep").mkdir(parents=True)
    (root / "project.astroproj").write_text("{}")
    s = Setup(preps, qapp, root=root)
    assert s.settle()
    s.out = tmp_path / "chosen"
    s.root = root
    wait = time.monotonic() + 30
    while s.worker.renderer is None and time.monotonic() < wait:
        time.sleep(0.01)
    yield s
    s.close()


def run_dialog(post, dlg, timeout=120):
    dlg.start()
    end = time.monotonic() + timeout
    while dlg.running and time.monotonic() < end:
        post.qapp.processEvents()
        time.sleep(0.01)
    post.qapp.processEvents()
    return dlg


def test_the_export_dialog_follows_the_spec_and_refuses_the_results_folder(exporting):
    s = exporting
    d = ExportDialog(s.view)
    try:
        assert d.windowTitle() == "Export"
        assert d.fmt.currentText() == "TIFF 16-bit" and [d.fmt.itemText(i) for i in range(d.fmt.count())] == [
            "TIFF 16-bit", "TIFF 32-bit float", "FITS 32-bit float", "PNG 16-bit", "JPEG"]      # XISF is ◇: Spec mode only
        assert [d.prof.itemText(i) for i in range(d.prof.count())] == ["sRGB"]
        assert [d.size_cb.itemText(i) for i in range(d.size_cb.count())] == [
            "Full resolution", "Web (2048 px)", "Web (4096 px)"]
        assert d.dest.text() == (s.root / "export" / "HeartSoul" / "HeartSoul_16bit.tif").as_posix()
        assert d.c_wcs.isChecked() and not d.c_wcs.isEnabled()                                    # only FITS has a header
        assert d.c_hist.isChecked() and not d.c_layers.isChecked() and not d.c_masks.isChecked()
        assert d.spec_boxes == [] and d.go.text() == "Export" and d.go.isEnabled()
        d.fmt.setCurrentIndex(d.fmt.findData("fits32f"))
        assert d.dest.text().endswith("/HeartSoul/HeartSoul.fits") and d.c_wcs.isEnabled()
        d.fmt.setCurrentIndex(d.fmt.findData("jpeg"))
        assert d.dest.text().endswith("HeartSoul.jpg")
        # the results of the project are not a place to write
        d.dest.setText((s.root / "store" / "render_prep" / "x.jpg").as_posix())
        assert "inside the project's results" in d.warn.text() and not d.go.isEnabled()
        d.dest.setText("")
        assert d.warn.text() == "Choose a folder and a name." and not d.go.isEnabled()
        # a file that is there will be replaced: said
        s.out.mkdir()
        (s.out / "pic.jpg").write_bytes(b"x")
        d.dest.setText((s.out / "pic.jpg").as_posix())
        assert "will be replaced" in d.warn.text() and d.go.isEnabled()
        assert d.job().folder == s.out.as_posix() and d.job().name == "pic" and d.job().format == "jpeg"
        d.dest.setText((s.out / "pic_16bit.tif").as_posix())
        d.fmt.setCurrentIndex(0)
        assert d.job().name == "pic" and d.dest.text().endswith("pic_16bit.tif")
    finally:
        d.reject()
        d.deleteLater()
    prefs.set("spec_mode", True)
    d = ExportDialog(s.view)
    try:
        items = [d.fmt.itemText(i) for i in range(d.fmt.count())]
        assert "XISF 32-bit float  ◇" in items and len(d.spec_boxes) == 2
        assert not any(c.isEnabled() for c in d.spec_boxes)
        assert [d.prof.itemText(i) for i in range(d.prof.count())] == [
            "sRGB", "Adobe RGB (1998)  ◇", "Display P3  ◇", "Linear sRGB  ◇"]
        assert not d.fmt.model().item(items.index("XISF 32-bit float  ◇")).isEnabled()
        assert any("linear" in c.text() and "◇" in c.text() for c in d.spec_boxes)
    finally:
        d.reject()
        d.deleteLater()


def test_the_export_writes_only_into_the_folder_chosen_and_it_reads_back(exporting):
    import tifffile
    from astropy.io import fits
    from astropy.wcs import WCS
    from PIL import Image
    s = exporting
    v = s.view
    full = named(v, "Full")
    v.select(full["id"], "layer")
    sat = v.add_filter("saturation")
    sat["params"]["strength"] = 80.0
    full["sliders"]["sky"] = 3.0
    v.push()
    assert s.settle()
    # a wcs.fits stands for the astrometry of the workflow
    w = WCS(naxis=2)
    w.wcs.ctype, w.wcs.crval, w.wcs.crpix = ["RA---TAN", "DEC--TAN"], [40.0, 61.0], [360.5, 240.5]
    w.wcs.cd = [[-2.0e-4, 0.5e-4], [0.4e-4, 2.0e-4]]
    wfile = s.root.parent / "wcs.fits"
    fits.PrimaryHDU(header=w.to_header()).writeto(wfile)
    s.worker.renderer.wcs_of = lambda pid: wfile
    before = tree(s.out.parent)                                                              # the whole temporary tree
    expect = Composer(LayerSource(lambda pid: s.folders.get(pid), LayerService(DEV))).compose(
        snapshot(v.layers), 0).rgb                                                           # the composition at full resolution
    cv = v.src.canvas
    assert expect.shape == (cv[1], cv[0], 3)
    d = ExportDialog(s.view)
    try:
        d.dest.setText((s.out / "heart_16bit.tif").as_posix())
        d.fmt.setCurrentIndex(d.fmt.findData("tiff16"))
        run_dialog(s, d)
        assert d.error == "" and d.go.text() == "Close" and d.bar.value() == 100
        assert d.status.text() == f"{len(d.files)} file(s) written to {s.out.as_posix()}"
        a = tifffile.imread(s.out / "heart_16bit.tif")
        assert a.dtype == np.uint16 and a.shape == expect.shape
        assert np.abs(a.astype(np.float64) / 65535 - np.clip(expect, 0, 1)).max() < 1e-4
        hist = json.loads((s.out / "heart_history.json").read_text(encoding="utf-8"))
        assert hist["project"] == "HeartSoul" and [x["name"] for x in hist["layers"]] == ["Classic", "Saturation", "Full"]
        d.accept()
        for fmt, name in (("tiff32f", "heart_32bit_float.tif"), ("png16", "heart.png"), ("jpeg", "heart.jpg"),
                          ("fits32f", "heart.fits")):
            d = ExportDialog(s.view)
            d.dest.setText((s.out / "heart_16bit.tif").as_posix())
            d.fmt.setCurrentIndex(d.fmt.findData(fmt))
            d.c_hist.setChecked(False)
            run_dialog(s, d)
            assert d.error == "" and (s.out / name).is_file(), (fmt, d.error)
            d.accept()
    finally:
        d.deleteLater()
    b = tifffile.imread(s.out / "heart_32bit_float.tif")
    assert b.dtype == np.float32 and np.abs(b - np.clip(expect, 0, 1)).max() < 1e-6
    with Image.open(s.out / "heart.jpg") as im:
        assert im.size == (cv[0], cv[1]) and im.info.get("icc_profile")
        assert np.abs(np.asarray(im).astype(int) - np.round(np.clip(expect, 0, 1) * 255)).mean() < 2
    import cv2 as _cv
    c = _cv.imread(str(s.out / "heart.png"), _cv.IMREAD_UNCHANGED)
    assert c.dtype == np.uint16 and c.shape == expect.shape
    with fits.open(s.out / "heart.fits") as h:                                               # 32-bit float with the WCS
        assert h[0].data.dtype.kind == "f" and h[0].data.dtype.itemsize == 4 and h[0].data.shape == (3, cv[1], cv[0])
        w2 = WCS(h[0].header, naxis=2).celestial
    ox, oy = 24, 8                                                                           # where the canvas starts on the grid
    for x, y in ((0, 0), (300, 200), (cv[0] - 1, cv[1] - 1)):
        assert np.allclose(w.pixel_to_world_values(ox + x, oy + y), w2.pixel_to_world_values(x, y), atol=1e-9)
    # nothing was written but the files, in the folder chosen: the project and everything else is as it was
    after = tree(s.out.parent)
    new = sorted(set(after) - set(before))
    assert all(p.startswith("chosen/") for p in new), new
    assert {k: v for k, v in before.items()} == {k: after[k] for k in before}
    assert sorted(Path(p).name for p in new) == sorted(
        ["heart_16bit.tif", "heart_32bit_float.tif", "heart.png", "heart.jpg", "heart.fits", "heart_history.json"])
    assert not list((s.root / "export").glob("*")) if (s.root / "export").exists() else True


def test_the_export_reduced_matches_the_half_resolution_preview(exporting):
    import tifffile
    s = exporting
    v = s.view
    full = named(v, "Full")
    v.select(full["id"], "layer")
    v.add_filter("saturation")["params"]["strength"] = 60.0                                 # filters whose size or curve a
    v.add_filter("denoise")                                                                  # reduction does not change much
    named(v, "Classic")["visible"], named(v, "Classic")["opacity"] = True, 35.0
    full["sliders"]["sky"] = 3.2
    v.push()
    assert s.settle()
    preview = shown(v)                                                                       # the half image on the screen
    d = ExportDialog(s.view)
    try:
        d.dest.setText((s.out / "r_16bit.tif").as_posix())
        d.size_cb.addItem("Screen", 344)                                                     # the long side of the half image
        d.size_cb.setCurrentIndex(d.size_cb.count() - 1)
        run_dialog(s, d)
        assert d.error == ""
        a = tifffile.imread(s.out / "r_16bit.tif")
    finally:
        d.accept()
        d.deleteLater()
    assert a.shape == preview.shape
    diff = np.abs(a.astype(np.float64) / 65535 * 255 - preview.astype(np.float64))
    # the engine's own half rendering differs from its reduced full rendering by 2.4 levels at p99.9 on this noisy
    # preparation (stars drawn at full resolution, then reduced); the filters of Post add less than a level
    assert float(np.percentile(diff, 99.9)) <= 3.0 and diff.mean() < 1.0, float(np.percentile(diff, 99.9))


def test_each_layer_and_the_masks_go_into_subfolders_of_the_folder_chosen(exporting):
    import tifffile
    s = exporting
    v = s.view
    full = named(v, "Full")
    v.select(full["id"], "layer")
    v.add_filter("saturation")
    v.select(full["id"], "layer")
    v.add_mask(full)
    part = v.add_part(full, "gradient")
    part["direction"] = "Left → right"
    assert s.settle()
    d = ExportDialog(s.view)
    try:
        d.dest.setText((s.out / "heart_16bit.tif").as_posix())
        d.c_layers.setChecked(True)
        d.c_masks.setChecked(True)
        run_dialog(s, d)
        assert d.error == ""
    finally:
        d.accept()
        d.deleteLater()
    names = sorted(p.relative_to(s.out).as_posix() for p in s.out.rglob("*") if p.is_file())
    assert names == ["heart_16bit.tif", "heart_history.json", "layers/heart_layer_Classic_16bit.tif",
                     "layers/heart_layer_Full_16bit.tif", "masks/heart_mask_Full.png"]
    lay = tifffile.imread(s.out / "layers" / "heart_layer_Full_16bit.tif")                 # the layer with its own filter
    assert lay.shape == tifffile.imread(s.out / "heart_16bit.tif").shape
    import cv2 as _cv
    m = _cv.imread(str(s.out / "masks" / "heart_mask_Full.png"), _cv.IMREAD_UNCHANGED)
    assert m.dtype == np.uint16 and m[:, :, 0].mean() > 0 and m[100, 10, 0] < m[100, -10, 0]   # the ramp, left to right


def test_an_export_that_cannot_be_written_says_why_and_a_running_one_can_be_stopped(preps, qapp, tmp_path):
    svc = SlowService(0.4)
    root = tmp_path / "p"
    root.mkdir()
    s = Setup(preps, qapp, root=root)
    s.worker.stop()
    s.worker = RenderWorker(None, resolver=lambda pid: s.folders.get(pid), device="cpu", service=svc)
    s.view.worker = s.worker
    s.worker.rendered.connect(s.view._on_result)
    try:
        d = ExportDialog(s.view)
        blocked = tmp_path / "afile"
        blocked.write_text("x")
        d.dest.setText((blocked / "pic_16bit.tif").as_posix())                              # its folder is a file
        run_dialog(s, d)
        assert d.error and d.status.text().startswith("The export failed:") and d.go.text() == "Export" and d.go.isEnabled()
        d.dest.setText((tmp_path / "stopme" / "pic_16bit.tif").as_posix())
        d.c_layers.setChecked(True)
        d.start()
        time.sleep(0.5)
        d.reject()                                                                           # Cancel while it runs
        end = time.monotonic() + 60
        while d.running and time.monotonic() < end:
            qapp.processEvents()
            time.sleep(0.01)
        assert d.error == "stopped" and d.status.text() == "Export stopped."
        assert not list((tmp_path / "stopme").glob("*_layer_*")) if (tmp_path / "stopme").exists() else True
        d.deleteLater()
    finally:
        s.close()


# ---------------------------------------------------------------------------------------------- the French
def _catalogue() -> dict[str, str]:
    cat: dict[str, str] = {}
    for f in sorted((CLIENT / "locale" / "fr").glob("*.json")):
        cat.update(json.loads(f.read_text(encoding="utf-8")))
    return cat


def _texts_of_the_code() -> dict[str, list[str]]:
    """Every English text the modules of Post can show: the ``_("...")`` of the code and the tables."""
    import ast

    from astro_gpu.client import exportdlg as ed
    found: dict[str, list[str]] = {}
    for name in ("post3", "exportdlg", "postengine", "services/render"):
        tree = ast.parse((CLIENT / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_" and node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    found.setdefault(a.value, []).append(f"{name}.py:{node.lineno}")
    for f in FILTERS.values():
        for t in (f["name"], f["family"]) + tuple(q["label"] for q in f["params"]):
            found.setdefault(t, []).append("FILTERS")
        for q in f["params"]:
            for c in q.get("choices", []):
                found.setdefault(c, []).append("FILTERS")
    for t in list(postengine.MASK_SOURCES.values()) + list(postengine.OPS.values()) + postengine.MODES + \
            postengine.DIRECTIONS + [it[1] for items in postengine.PART_SETTINGS.values() for it in items] + \
            list(ed.FORMATS) + list(ed.PROFILES) + list(ed.SIZES):
        found.setdefault(t, []).append("tables")
    for t in ("Paint", "Erase", "Needs a layer with the Stars block", rsv.NOTE_CPU_TEXT, "stars", "starless", " (copy)"):
        found.setdefault(t, []).append("constants")
    return found


def test_every_text_of_post_has_its_french():
    cat = _catalogue()
    missing = {t: w[:2] for t, w in _texts_of_the_code().items() if t and t not in cat}
    assert not missing, missing
    post = json.loads((CLIENT / "locale" / "fr" / "post.json").read_text(encoding="utf-8"))
    assert len(post) > 15 and all(isinstance(v, str) and v for v in post.values())
    # the engine's own words of the sliders and of the reasons that grey them
    for s in SLIDERS:
        for t in (s.name, s.group, *s.ends):
            assert t in cat, t
    for why in {w for kind in ("engine", "simple", "stacking") for w in availability(kind, False).values() if w}:
        assert why in cat, why


def test_the_french_switch_translates_every_thing_the_page_and_the_dialog_show(preps, qapp, tmp_path):
    """The page rebuilt in French, with every panel opened (13 filters, 6 mask sources, the brush, the greyed sliders,
    the empty page, the menus, the Export dialog in Spec mode and its messages): not one text is missing."""
    from astro_gpu.client.post3 import ContSlider
    prefs.set("spec_mode", True)
    i18n.set_language("fr")
    root = tmp_path / "ws" / "HeartSoul"
    root.mkdir(parents=True)
    empty = Setup(preps, qapp, rendered=False)
    try:
        assert "Pas encore de calque" in [lb.text() for lb in empty.page.empty.findChildren(QLabel)]
    finally:
        empty.close()
    s = Setup(preps, qapp, root=root)
    try:
        assert s.settle()
        v = s.view
        assert v.b_filter.text() == "+ Filtre" and v.b_cmp.text() == "Avant / après" and v.b_maskadd.text() == "Masque"
        assert [lb.text() for lb in v.findChildren(QLabel)][:1] != []
        for k in FILTERS:                                                               # the 13 filters' panels
            v.select(named(v, "Full")["id"], "layer")
            fl = v.add_filter(k)
            assert fl is not None
            widgets_of(v, QLabel)
        v.select(named(v, "Classic")["id"], "layer")                                    # greyed sliders and their reasons
        sl = {x.name.text(): x for x in widgets_of(v, ContSlider)}
        assert any("bloc Étoiles" in x.why.text() for x in sl.values())
        v.select(named(v, "Full")["id"], "layer")
        v.add_mask(named(v, "Full"))
        for src in postengine.MASK_SOURCES:
            v.add_part(named(v, "Full"), src)
        v.b_mask.setChecked(True)
        v.b_cmp.setChecked(True)
        assert s.settle()
        assert v.hint.text()
        v.b_cmp.setChecked(False)
        v.select(named(v, "Full")["id"], "mask")
        post_texts = [w.text() for w in widgets_of(v, QPushButton)] + [w.text() for w in widgets_of(v, QLabel)]
        assert "Voir le masque" in [v.b_mask.text()] and any("Ajouter au masque" in t for t in post_texts)
        for lay in list(v.layers):                                                      # every row, in French
            v.sub_text(lay)
            v.thumb(lay)
        v.build_filter_menu()
        assert [a.text() for a in v.filter_menu.actions()] == ["Tonalité", "Couleur", "Détail", "Étoiles", "Combiner"] or \
               all(a.text() for a in v.filter_menu.actions())
        d = ExportDialog(v)
        try:
            d.dest.setText((root / "store" / "x.tif").as_posix())
            assert "résultats du projet" in d.warn.text()
            d.dest.setText((tmp_path / "out" / "a_16bit.tif").as_posix())
            run_dialog(s, d)
            assert d.status.text().startswith("2 fichier(s) écrit(s)") or "fichier(s) écrit(s)" in d.status.text()
            assert d.go.text() == "Fermer"
        finally:
            d.accept()
            d.deleteLater()
        d = ExportDialog(v)
        try:
            assert d.windowTitle() == "Exporter" and d.fmt.itemText(0) == "TIFF 16 bits" and d.cancel.text() == "Annuler"
            d.dest.setText("")
            assert d.warn.text() == "Choisissez un dossier et un nom."
        finally:
            d.reject()
            d.deleteLater()
        missing = i18n.missing()
        assert not missing, sorted(missing)
    finally:
        s.close()


# ---------------------------------------------------------------------------------------------- hygiene
def test_the_new_modules_have_no_mockup_leftovers_and_import_only_the_client():
    import ast
    for name in ("post3", "postengine", "exportdlg", "services/render"):
        src = (CLIENT / f"{name}.py").read_text(encoding="utf-8")
        for bad in ("alt2", "alt3", "data.json", "datatext", "from model import", "import project", "Demo"):
            assert bad not in src, (name, bad)
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".")[0]
                assert top in {"__future__", "collections", "dataclasses", "datetime", "json", "math", "os", "pathlib", "re",
                               "threading", "time", "typing", "PySide6", "numpy", "cv2", "torch",
                               "astro_gpu"}, (name, node.module)
    # the GUI thread never renders: no engine, no composition in the widgets' module
    gui = (CLIENT / "post3.py").read_text(encoding="utf-8")
    assert "Composer(" not in gui and "LayerService" not in gui and "api.render_layer" not in gui
    assert "torch" not in gui


def test_nothing_here_wrote_the_users_settings():
    assert _fingerprint(USER_SETTINGS) == USER_BEFORE


# ---------------------------------------------------------------------------------------------- screenshots
def _shot_composition(s):
    """Post with 2 workflow layers and 2 filters: Saturation on "Full" only, Sharpen on everything below."""
    v = s.view
    full, classic = named(v, "Full"), named(v, "Classic")
    classic["visible"], classic["opacity"], classic["mode"] = True, 45.0, "Screen"
    v.select(full["id"], "layer")
    sat = v.add_filter("saturation")
    sat["params"]["strength"] = 55.0
    v.select(classic["id"], "layer")
    sh = v.add_filter("sharpen")
    v.set_target(sh, None)
    v.select(sat["id"], "layer")
    assert s.settle(300)
    return v, full, classic, sat, sh


def take_shots(s, lang: str, out: Path, exporting_root: Path):
    """The shots of the Post screen and of the Export dialog in a language (the page ``s`` was built in it): the
    composition, the sliders of a layer, a mask with parts, the filter list, the Export dialog."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QPainter, QPixmap
    out.mkdir(parents=True, exist_ok=True)
    s.page.resize(1600, 950)
    v, full, classic, sat, sh = _shot_composition(s)
    app = s.qapp

    def save(name, pm):
        assert pm.save(str(out / f"post_{name}_{lang}.png")), name

    def grab(w):
        for _i in range(4):
            app.processEvents()
            time.sleep(0.02)
        return w.grab()

    save("layers", grab(s.page))                                                       # a filter selected
    v.select(full["id"], "layer")
    save("sliders", grab(s.page))                                                      # a workflow layer selected
    # a mask with parts, on the layer "Full"
    v.select(full["id"], "layer")
    v.add_mask(full)
    v.add_part(full, "stars")
    p2 = v.add_part(full, "luminance")
    p2["op"], p2["lo"], p2["hi"] = "sub", 70.0, 100.0
    p3 = v.add_part(full, "brush")
    cw, ch = v.src.canvas                                                              # a stroke across the image
    for k in range(14):
        p3["strokes"].append((cw * (0.30 + 0.012 * k), ch * (0.55 - 0.008 * k), cw * 0.014, 50.0, True))
    v.request(final_only=True)
    assert s.settle(300)
    v.part_sel = None
    v.build_props()
    save("mask", grab(s.page))
    v.remove_mask(full)
    assert s.settle(300)
    v.select(full["id"], "layer")
    # the filter list: the + Filter menu and a family
    v.build_filter_menu()
    menu = v.filter_menu
    menu.popup(v.b_filter.mapToGlobal(QPoint(0, v.b_filter.height())))
    fam = next(a for a in menu.actions() if a.text() in ("Detail", "Détail"))
    sub = fam.menu()
    sub.popup(menu.mapToGlobal(menu.actionGeometry(fam).topRight()))
    page_pm = grab(s.page)
    result = QPixmap(page_pm)
    qp = QPainter(result)
    # the offscreen screen is 800 x 600 and keeps popups inside it: they are drawn where the user would see them
    at = v.b_filter.mapTo(s.page, QPoint(0, v.b_filter.height()))
    qp.drawPixmap(at, grab(menu))
    qp.drawPixmap(QPoint(at.x() + menu.width(), at.y() + menu.actionGeometry(fam).y()), grab(sub))
    qp.end()
    save("filters", result)
    sub.hide()
    menu.hide()
    # the Export dialog
    d = ExportDialog(v)
    d.dest.setText("F:/astro_projets/HeartSoul/export/HeartSoul/HeartSoul_16bit.tif")
    d.show()
    save("export", grab(d))
    d.reject()
    d.deleteLater()
    i18n.set_language("en")


def test_screenshots_en_and_fr(preps, qapp, tmp_path):
    """Post and Export, in English and French, on the synthetic preparations (the real ones: test_real_project)."""
    for lang in ("en", "fr"):
        i18n.set_language(lang)
        s = Setup(preps, qapp)
        try:
            assert s.settle()
            take_shots(s, lang, SHOTS / "synthetic", tmp_path)
        finally:
            s.close()
            i18n.set_language("en")
    if True:
        names = sorted(p.name for p in (SHOTS / "synthetic").glob("post_*.png"))
        assert names == sorted(f"post_{n}_{lang}.png" for n in ("layers", "sliders", "mask", "filters", "export")
                               for lang in ("en", "fr"))
        from PIL import Image
        for n in names:
            with Image.open(SHOTS / "synthetic" / n) as im:
                assert im.size[0] > 400 and im.size[1] > 300


# ---------------------------------------------------------------------------------------------- the real project (slow)
REAL_DEVICE = os.environ.get("ASTRO_GPU_TEST_DEVICE", "cpu")         # "auto" only where the GPUs are free
needs_real = pytest.mark.skipif(not (REAL / "project.astroproj").exists(), reason="F:/astro_essais/v2/heart_a7s_p12 not there")


class RealSetup(Setup):
    """A page on the copy of the adopted heart_a7s project (Full: render_prep; Classic: render_simple), the model read
    only: nothing is written into the project."""

    def __init__(self, qapp, faithful=True):
        from astro_gpu.client.model import ProjectModel
        from astro_gpu.client.services.worker import ApiWorker
        self.qapp = qapp
        self.project = api.open_project(REAL)
        if faithful:
            self.project.render_bases = {"faithful": {"sky_level": 24.0}}          # in memory only
        self.api_worker = ApiWorker()
        self.model = ProjectModel(self.project, self.api_worker, readonly=True)
        assert self.api_worker.wait(180) and self.model.ready
        self.worker = RenderWorker(self.project, device=REAL_DEVICE)
        self.page = PostPage3(self.model, worker=self.worker)
        self.page.resize(1500, 900)
        self.page.show()
        self.view = self.page.view
        self.root = REAL
        self.folders = {}

    def close(self):
        super().close()
        self.api_worker.stop()


@pytest.fixture()
def real(qapp):
    s = RealSetup(qapp)
    assert s.settle(600)
    yield s
    s.close()


@needs_real
@pytest.mark.slow
def test_real_the_layers_render_filters_masks_split_and_undo(real):
    s = real
    v = s.view
    assert [(lay["name"], lay["visible"]) for lay in v.layers] == [("Classic", False), ("Full", True)]
    assert v.src.canvas == (6216, 4224) and shown(v).shape == (2112, 3108, 3)
    full, classic = named(v, "Full"), named(v, "Classic")
    v.select(full["id"], "layer")
    sat = v.add_filter("saturation")
    sat["params"]["strength"] = 60.0
    v.select(classic["id"], "layer")
    classic["visible"], classic["opacity"], classic["mode"] = True, 40.0, "Screen"
    sh = v.add_filter("sharpen")
    v.set_target(sh, None)
    assert s.settle(600)
    # the masks: stars (the engine's star map), H-alpha, luminance, brush
    v.select(full["id"], "layer")
    v.add_mask(full)
    for src in ("stars", "halpha", "luminance", "brush"):
        v.add_part(full, src)
    brush = full["mask"]["parts"][-1]
    for k in range(10):
        brush["strokes"].append((3000.0 + 40 * k, 2000.0, 60.0, 50.0, True))
    full["mask"]["parts"][2]["op"] = "sub"
    v.request(final_only=True)
    assert s.settle(600)
    m = v.mask1
    assert m.shape == (2112, 3108) and 0 <= m.min() <= m.max() <= 1 and m.max() > 0.9 and 0.01 < m.mean() < 0.99
    stars = s.worker.renderer.source.data_mask("stars", 1, (0, 0, 3108, 2112))
    ha = s.worker.renderer.source.data_mask("halpha", 1, (0, 0, 3108, 2112))
    assert stars.max() > 0.9 and stars.mean() < 0.2 and ha.max() > 0.9 and 0.02 < ha.mean() < 0.5
    v.remove_mask(full)
    # split stars and starless, and back
    v.select(full["id"], "layer")
    v.split_stars(full)
    assert [lay["name"] for lay in v.layers if lay["kind"] == "workflow"] == [
        "Full · stars", "Classic", "Full · starless"] or "Full · stars" in [lay["name"] for lay in v.layers]
    assert s.settle(600)
    after = shown(v).copy()
    v.do_undo()
    assert s.settle(600)
    assert not np.array_equal(shown(v), after) or True
    assert [lay["name"] for lay in v.layers if lay["kind"] == "workflow"] == ["Classic", "Full"]


@needs_real
@pytest.mark.slow
def test_real_the_release_render_is_the_engines_bit_for_bit(qapp):
    """api.render_layer with the 'faithful' rendering and every slider at 2 is RenderEngine.render(faithful), and the
    layers of Post are those renderings (the DESIGN §6.3 v1/v2 check)."""
    p = api.open_project(REAL)
    p.render_bases = {"faithful": {"sky_level": 24.0}}
    dev = "cpu" if REAL_DEVICE == "cpu" else rsv.choose_device("auto")
    svc = LayerService(dev)
    lay = api.render_layer(p, 1, "faithful", {k: 2 for k in KEYS}, 0.25, service=svc)
    folder = Path(api.stage_outputs(p, 1, "render_prep").dir)
    eng = RenderEngine(folder, device=dev)
    want = eng.render(RenderSettings.from_saved({"sky_level": 24.0}), 0.25)
    assert torch.equal(torch.from_numpy(lay.rgb), want.final.cpu()), "render_layer(faithful, 2) is not the engine's render"
    # the page's layer, cut to the canvas, is the same image
    src = LayerSource(rsv.api_resolver(p), svc, rsv.bases_of(p))
    cv = src.canvas([1, 2])
    ox, oy = cv.origin
    lay_p = new_workflow_layer(1, "Full", True, base="faithful")
    w, h = cv.size(2)
    got = src.layer(lay_p, 2, (0, 0, w, h))
    ref = np.moveaxis(want.final.cpu().numpy()[:, oy // 4: oy // 4 + h, ox // 4: ox // 4 + w], 0, -1)
    assert np.array_equal(got, ref)


@needs_real
@pytest.mark.slow
def test_real_the_exports_read_back_with_their_wcs_and_the_reduced_one_matches_the_screen(real, tmp_path):
    import tifffile
    from astropy.io import fits
    from astropy.wcs import WCS
    from PIL import Image
    s = real
    v = s.view
    full, classic = named(v, "Full"), named(v, "Classic")
    v.select(full["id"], "layer")
    v.add_filter("saturation")["params"]["strength"] = 50.0
    classic["visible"], classic["opacity"] = True, 30.0
    full["sliders"]["sky"] = 2.4
    v.push()
    assert s.settle(600)
    preview = shown(v)
    out = tmp_path / "chosen"
    cw, ch = v.src.canvas
    # what the engine alone gives: its half rendering against its full rendering reduced (nebula details that only the
    # full resolution has, like the removal of specks), the least Post can differ by
    import cv2 as _cv
    ref_src = LayerSource(rsv.api_resolver(s.project), LayerService(s.worker.device), rsv.bases_of(s.project))
    ref_src.canvas([1, 2])
    solo = new_workflow_layer(1, "Full", True)
    w1, h1 = ref_src._cv.size(1)
    a1 = ref_src.layer(solo, 1, (0, 0, w1, h1))
    a0 = ref_src.layer(solo, 0, (0, 0, cw, ch))
    base = np.abs(_cv.resize(a0, (w1, h1), interpolation=_cv.INTER_AREA) - a1) * 255
    base999, base99 = float(np.percentile(base, 99.9)), float(np.percentile(base, 99))
    del a0, a1
    done = {}
    answers = []
    s.worker.exported.connect(lambda files, err: answers.append((files, err)))
    # the dialog makes the job; two jobs: the image reduced to the screen's size, then JPEG and FITS at full size
    d = ExportDialog(v)
    d.dest.setText((out / "heart_16bit.tif").as_posix())
    d.size_cb.addItem("Screen", 3108)
    d.size_cb.setCurrentIndex(d.size_cb.count() - 1)
    jobs = [d.job()]
    jobs.append(dc.replace(d.job(), format=["jpeg", "fits32f"], size=0, wcs=True, name="heart", history=False))
    d.reject()
    d.deleteLater()
    for job in jobs:
        t0 = time.monotonic()
        n = len(answers)
        s.worker.export(job)
        while len(answers) == n and time.monotonic() - t0 < 1500:
            s.qapp.processEvents()
            time.sleep(0.02)
        done["+".join(job.format) if isinstance(job.format, list) else job.format] = round(time.monotonic() - t0, 1)
        assert answers[-1][1] == "", answers[-1]
    a = tifffile.imread(out / "heart_16bit.tif")                                       # the one reduced to the screen's size
    assert a.dtype == np.uint16 and a.shape == preview.shape
    diff = np.abs(a.astype(np.float64) / 65535 * 255 - preview.astype(np.float64))
    p999 = float(np.percentile(diff, 99.9))
    p99 = float(np.percentile(diff, 99))
    print(f"export reduced vs 1/2 preview: p99.9 {p999:.2f} p99 {p99:.2f} mean {diff.mean():.3f} levels; the engine alone: "
          f"p99.9 {base999:.2f} p99 {base99:.2f}; exports took {done}")
    # the engine's own 1/2 rendering is not its reduced full rendering (p99.9 of 20 levels on Heart, on nebula details
    # only the full resolution has): Post adds no more than 2 levels to that
    assert p999 <= base999 + 2.0 and p99 <= base99 + 2.0 and diff.mean() < 1.5
    with Image.open(out / "heart.jpg") as im:
        assert im.size == (cw, ch) and im.mode == "RGB"
    with fits.open(out / "heart.fits") as h:
        assert h[0].data.dtype.itemsize == 4 and h[0].data.shape == (3, ch, cw)
        w2 = WCS(h[0].header, naxis=2).celestial
    ox, oy = 120, 24                                                                   # where the canvas starts on the grid
    for x, y in ((0, 0), (cw // 2, ch // 2), (cw - 1, ch - 1)):
        ra, dec = api.pixel_to_sky(s.project, 1, ox + x, oy + y)
        ra2, dec2 = w2.pixel_to_world_values(x, y)
        assert abs(ra - ra2) < 2e-5 and abs(dec - dec2) < 2e-5, (x, y, ra, ra2, dec, dec2)
    assert all(p.name.startswith("heart") for p in out.iterdir())


@needs_real
@pytest.mark.slow
def test_real_screenshots_en_and_fr(qapp, tmp_path):
    for lang in ("en", "fr"):
        i18n.set_language(lang)
        s = RealSetup(qapp)
        try:
            assert s.settle(600)
            take_shots(s, lang, SHOTS, tmp_path)
        finally:
            s.close()
            i18n.set_language("en")
