"""Post and Export, offscreen: layers, filters, masks, the render worker, the engine's renderings, the export.

The composition (layers, filters, masks), the render worker (supersede, device, fallback), the engine's renderings
through it (bit for bit), the export, the screen, the French and the screenshots.

The synthetic preparations (a nebula, stars, an uncovered border) make the fast tests; the real ones read the test
project (tests/conftest.py: Full computed, its computed Classic workflow added in memory) and are marked slow.
ASTRO_GPU_SETTINGS points to a temporary file everywhere: the user's settings file is checked untouched.
ASTRO_GPU_SHOTS names the folder of the screenshots (default: %TEMP%/post_shots).
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
from conftest import TEST_PROJECT, TEST_PROJECT_CLASSIC, needs_test_project  # noqa: E402
from PySide6.QtCore import QObject, Qt, Signal  # noqa: E402
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
from astro_gpu.render.layers import NOTE_CPU, LayerService  # noqa: E402
from astro_gpu.render.simple import image_stats  # noqa: E402
from astro_gpu.render.sliders import KEYS, KINDS, SLIDERS, availability  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "astro_gpu" / "client"
#: the modules of Post (their texts are translated, their imports checked)
POST_MODULES = ("post3", "exportdlg", "postengine", "services/render", "curveseditor", "postfilters/__init__",
                "postfilters/colour", "postfilters/curves", "postfilters/tone", "postfilters/detail")
REAL = TEST_PROJECT
DEV = "cpu"                                              # the fast tests render on the CPU: no GPU needed, no GPU busy
SHOTS = Path(os.environ.get("ASTRO_GPU_SHOTS") or Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "post_shots")


def _user_settings() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def _fingerprint(p: Path):
    return (p.stat().st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest()) if p.exists() else None


USER_SETTINGS = _user_settings()
USER_BEFORE = _fingerprint(USER_SETTINGS)


# ---------------------------------------------------------------------------------------------- synthetic preparations
def stars_prep(d: Path, H: int, W: int, seed: int = 3, crop=None, scale: float = 1.5) -> tuple[Path, Path]:
    """A Stars workflow's two folders: its Output block's (render_simple: a nebula and its stars, an uncovered border)
    and its Stars block's (star_removal: the starless image and the stars alone, the same grid)."""
    rng = np.random.default_rng(seed)
    out, st = d / "output", d / "stars"
    out.mkdir(parents=True, exist_ok=True)
    st.mkdir(parents=True, exist_ok=True)
    neb = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), max(H, W) / 30)
    neb = (neb - neb.min()) / (neb.max() - neb.min()) * 120
    starless = np.stack([neb, neb * .45, neb * .25]) + rng.normal(0, .8, (3, H, W)).astype(np.float32)
    stars = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[:H, :W]
    for x, y, a in zip(rng.uniform(0, W, 60), rng.uniform(0, H, 60), rng.uniform(50, 3000, 60)):
        stars += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))
    stars = np.stack([stars * .9, stars, stars * .8]).astype(np.float32)
    lin = (starless + stars).astype(np.float32)
    valid = np.ones((H, W), bool)
    valid[:, :20] = False
    lin[:, ~valid] = 0
    crop = list(crop or [24, W - 4, 4, H - 4])
    stats = image_stats(lin, valid)
    np.save(out / "lin.npy", lin)
    np.save(out / "valid.npy", valid)
    (out / "prep_simple.json").write_text(json.dumps({"M": np.eye(3).tolist(), "crop": crop, "stats": stats,
                                                      "linked": True, "stretch": "auto", "denoised": False,
                                                      "color": True, "exit": None, "drizzle_scale": scale}))
    np.save(st / "starless.npy", starless.astype(np.float32))
    np.save(st / "stars.npy", stars)
    np.save(st / "valid.npy", valid)
    (st / "prep_stars.json").write_text(json.dumps({"crop": crop, "M": np.eye(3).tolist(), "drizzle_scale": scale,
                                                    "unit": 1.0, "background": [0, 0, 0], "stats_full": stats}))
    return out, st


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
    full, full_stars = stars_prep(t / "full", 480, 720)
    return {"full": full, "full_stars": full_stars, "classic": simple_prep(t / "classic", 480, 720),
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
    prefs.set("last_folder", tmp_path.as_posix())
    i18n.set_language("en")
    yield
    i18n.set_language("en")


# ---------------------------------------------------------------------------------------------- the toy model
class WF:
    """What the views read of a workflow (the model's WorkflowView)."""

    def __init__(self, pid, name, kind, rendered=True, denoise=True, stars=False):
        self.pid, self.name, self.kind, self.denoise = pid, name, kind, denoise
        self.rendered = self.computed = rendered
        self.provisional = False
        self.end = "empilement" if kind == "stacking" else None
        self.opts = {"etoiles": "retirer" if stars else "garder"}


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

    def slider_infos(self, p, part=None):
        why = availability(part or p.kind, p.denoise)
        return [api.SliderInfo(s.key, s.name, s.group, s.ends, not why[s.key], why[s.key]) for s in SLIDERS]

    def slider_state(self, p, key, part=None):
        info = next(x for x in self.slider_infos(p, part) if x.key == key)
        return info.available, i18n._(info.reason), None                      # as the real model: translated

    def edited(self):
        self.touched += 1


class Setup:
    """A Post page on synthetic preparations: workflow 1 'Full', 2 'Classic' (simple), 3 'Raw stack'. The folders by
    source id: a workflow's id for its Output block, ("stars", id) for its Stars block (workflow 1's). ``stars``: Full
    has the Stars block, so its layer is its Stars and background block (else its Output image)."""

    def __init__(self, preps, qapp, which=(1, 2), root=None, resolver=None, rendered=True, stars=False):
        self.qapp = qapp
        kinds = {1: ("Full", "simple", "full"), 2: ("Classic", "simple", "classic"), 3: ("Raw stack", "stacking", "raw")}
        self.folders = {pid: Folder(preps[kinds[pid][2]]) for pid in kinds if kinds[pid][2] in preps}
        if "full_stars" in preps:
            self.folders[("stars", 1)] = Folder(preps["full_stars"])
        self.model = ToyModel([WF(pid, kinds[pid][0], kinds[pid][1], rendered, stars=stars and pid == 1)
                               for pid in which], root=root)
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
    full, full_stars = stars_prep(t / "full", 1500, 2400)
    p = {"full": full, "full_stars": full_stars, "classic": simple_prep(t / "classic", 1500, 2400)}
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
    """The reference formulas of the filters (scipy, float64)."""

    def __init__(self, stars, halpha, f=1):
        self._s, self._h, self.f = stars, halpha, f
        self.level, self.draft = int(round(np.log2(f))), False

    def stars(self):
        return self._s

    def halpha(self):
        return self._h


def _reference(key, x, p, ctx):
    f = ctx.f
    if key == "stretch":
        d, sp = p["strength"], p["symmetry"]
        if d == 0:                                       # no stretch: the limit of the formula
            return x
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
    if key == "halpha_red":
        y = x.copy()
        y[..., 0] = x[..., 0] + p["strength"] * 0.35 * ctx.halpha() * (1 - x[..., 0])
        return y
    raise KeyError(key)


#: the original filters of Post; the settings some have since, at the values that keep the original behaviour
FIRST = {"stretch", "levels", "saturation", "scnr", "balance", "local_contrast", "sharpen", "denoise", "halpha_red"}
#: the filters that acted through the catalogue's star mask, gone with the star model (the Stars block's stars are a
#: layer of their own now)
GONE = {"star_reduce", "halos", "star_color"}
AS_FIRST = {"scnr": {"method": "Average neutral", "lightness": False}}


def test_the_original_filters_match_their_reference_formulas():
    """Each filter of the first Post, on a random image, at several settings and resolutions: the port (and its fast
    paths) gives what the reference formulas give, to a fraction of an 8-bit level; its three-slider Curves opens as the
    same curve in the Curves tool (tests/client/test_post_tools.py tests the new filters)."""
    assert FIRST < set(FILTERS) and "curves" in FILTERS and not GONE & set(FILTERS)
    rng = np.random.default_rng(1)
    x = np.clip(cv2.GaussianBlur(rng.random((150, 230, 3)).astype(np.float32), (0, 0), 3) * 1.6 - 0.3, 0, 1)
    stars = np.clip(rng.random((150, 230)).astype(np.float32) * 1.4 - 0.4, 0, 1)
    halpha = cv2.GaussianBlur(rng.random((150, 230)).astype(np.float32), (0, 0), 8)
    worst = {}
    for key in sorted(FIRST):
        f = FILTERS[key]
        defaults = {q["key"]: q["default"] for q in f["params"]}
        defaults.update(AS_FIRST.get(key, {}))
        variants = [defaults]
        for q in f["params"]:
            if q["key"] in AS_FIRST.get(key, {}):
                continue
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
    # the first Curves (shadows, midtones, highlights): saved compositions open with the same curve, exact
    for sh, mid, hi in ((0.1, -0.05, 0.08), (-0.2, 0.2, 0.0), (0.05, 0.0, -0.2)):
        ctx = RefCtx(stars, halpha, 1)
        old = {"shadows": sh, "midtones": mid, "highlights": hi}
        saved = {"id": 5, "kind": "filter", "filter": "curves", "name": "Curves", "visible": True, "opacity": 100.0,
                 "mode": "Normal", "params": old, "mask": None, "target": None}
        lay = postengine.load_layers([saved], set())[0]
        ctx.draft = False
        got = FILTERS["curves"]["fn"](x.copy(), lay["params"], ctx)
        err = float(np.abs(got - _reference("curves", x.astype(np.float64), old, ctx)).max()) * 255
        worst["curves (first)"] = max(worst.get("curves (first)", 0), err)
    assert max(worst.values()) < 1.0, worst
    # the families; the defaults are neutral (see the next test), the spec's old ones are the settings to try
    assert {f["family"] for f in FILTERS.values()} == set(postengine.FAMILIES)
    d = {k: {q["key"]: q["default"] for q in f["params"]} for k, f in FILTERS.items()}
    assert d["stretch"] == {"strength": 0, "symmetry": 0.08} and d["saturation"] == {"strength": 0, "hues": "All", "protect": 0}
    assert d["scnr"] == {"strength": 0, "method": "Average neutral", "lightness": True}
    assert d["local_contrast"] == {"strength": 0, "scale": 40}
    assert d["halpha_red"] == {"strength": 0}


def test_a_filter_added_with_its_defaults_leaves_the_image_unchanged():
    """Adding a filter must not move the image before the user touches a slider: every one of the 13, on a layer and on
    everything below, at full resolution and in a reduced draft, with a mask and a half opacity, and through a blend mode
    (the composite is the same to the float, so surely within 8-bit rounding)."""
    pr = ToyProvider()
    for level in (0, 2):
        for key in FILTERS:
            cp = Composer(pr)
            base = _stack((1, "A"), (2, "B"))                       # top first: B over A
            want = cp.compose(base, level).rgb
            for target in (base[0]["id"], None):
                f = new_filter_layer(key, target)
                assert {q["key"]: q["default"] for q in FILTERS[key]["params"]} == f["params"]
                got = cp.compose(normalize([f] + base), level).rgb
                assert np.abs(got - want).max() * 255 < 0.5, (key, level, target)
                assert np.array_equal(got, want), (key, level, target)           # in fact exactly
    # the same through a mask, a lower opacity and a blend mode: still nothing moves
    cp = Composer(pr)
    base = _stack((1, "A"), (2, "B"))
    want = cp.compose(base, 0).rgb
    for key in FILTERS:
        f = new_filter_layer(key, None)
        f["opacity"], f["mask"] = 50.0, new_mask()
        f["mask"]["parts"].append(new_part("luminance"))
        assert np.abs(cp.compose(normalize([f] + base), 0).rgb - want).max() * 255 < 0.5, key
        f["mask"], f["opacity"], f["mode"] = None, 100.0, "Lighten"
        assert np.abs(cp.compose(normalize([f] + base), 0).rgb - want).max() * 255 < 0.5, key
    # and a filter does change the image once a slider moves (the test above is not an empty one)
    sat = new_filter_layer("saturation", None)
    sat["params"]["strength"] = 60.0
    assert np.abs(cp.compose(normalize([sat] + base), 0).rgb - want).max() * 255 > 1


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
    """Workflow 1: a smooth gradient image with a bright disc; 2: its inverse. The H-alpha data mask: a few blobs."""

    def __init__(self, W=320, H=200):
        self.W, self.H = W, H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        a = np.stack([xx / W, yy / H, 0.3 + 0.4 * np.sin(xx / 40) * np.cos(yy / 30)], -1) * 0.8 + 0.1
        a[(xx - 100) ** 2 + (yy - 80) ** 2 < 30 ** 2] = 0.9
        self.img = {1: np.clip(a, 0, 1).astype(np.float32), 2: np.clip(1 - a, 0, 1).astype(np.float32)}
        self.blobs = np.zeros((H, W), np.float32)
        for x, y in ((50, 40), (200, 120), (260, 50)):
            self.blobs += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 20).astype(np.float32)
        self.blobs = np.clip(self.blobs, 0, 1)
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

    def layer(self, lay, level, box, draft=False, stop=None):
        self.calls.append((lay["pid"], level, tuple(box)))
        a = self.img[lay["pid"]]
        k = lay["sliders"].get("sky", 2.0) - 2
        a = np.clip(a + 0.05 * k, 0, 1)
        if lay.get("part") == "stars":
            a = np.clip(a - 0.5, 0, 1)
        return self._at(a, level, box)

    def data_mask(self, name, level, box):
        return self._at(self.blobs if name == "halpha" else self.ramp, level, box)

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
    s = new_part("halpha")
    lay["mask"]["parts"] = [s]
    assert np.allclose(m(), pr.blobs, atol=1e-5)
    with pytest.raises(ValueError):
        new_part("stars")                                              # the catalogue's star mask is gone
    # add: max; subtract: result * (1 - part); intersect: result * part; the first part decides the start
    a, b = new_part("halpha"), new_part("gradient")
    b.update(direction="Left → right", position=50.0, softness=20.0)
    star, rmp = pr.blobs, ramp
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
    assert np.allclose(m(), pr.blobs, atol=1e-6)


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
    sh = new_filter_layer("denoise", None)
    sh["params"].update(luminance=0.6, color=0.4)
    lay = [lc, sh, sat, base]
    base["mask"] = new_mask()
    base["mask"]["parts"] = [new_part("halpha")]
    full = cp.compose(lay, 0).rgb
    box = (200, 120, 460, 300)
    reg = cp.compose(lay, 0, box).rgb
    crop = full[120:300, 200:460]
    assert reg.shape == crop.shape
    inner = np.abs(reg - crop)[64:-64, 64:-64] * 255
    assert inner.max() < 1.0 and float(np.abs(reg - crop).max() * 255) < 8   # the margin: filters see their neighbours
    # half and quarter resolution look like the reduced full image (smooth filters: their sizes are divided by the factor)
    smooth = [new_filter_layer("curves", None), new_filter_layer("local_contrast", None), sat, base]
    smooth[0]["params"]["curves"] = postengine.curves_of_old(0.0, 0.1, 0.0)
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
    assert names.index("Sharpen") == names.index("A") - 1 and names.index("Green removal") == names.index("B") - 1
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


def test_the_layer_tree_and_its_moves_follow_affinity_photo():
    """A filter applying to one layer is its child (just above it in the list, under it in the panel); one applying to
    everything below is a top-level row. postengine.move nests, un-nests and reorders as a drop does, a layer moving
    with its children; reading the panel from the bottom up is the order of the composition."""
    pr = ToyProvider()
    cp = Composer(pr)
    a, b = new_workflow_layer(1, "A", True), new_workflow_layer(2, "B", False)
    lo, top = new_filter_layer("levels", a["id"]), new_filter_layer("curves", a["id"])
    lo["params"].update(black=0.2)
    top["params"]["curves"] = postengine.curves_of_old(0.0, 0.15, 0.0)
    g = new_filter_layer("saturation", None)
    layers = normalize([g, b, top, lo, a])
    assert [(x["name"], d) for x, d in postengine.rows(layers)] == [
        ("Saturation", 0), ("B", 0), ("A", 0), ("Curves", 1), ("Levels", 1)]
    assert [x["name"] for x, _k in postengine.tree(layers)] == ["Saturation", "B", "A"]
    assert [x["name"] for x, d in postengine.rows(layers, collapsed={a["id"]})] == ["Saturation", "B", "A"]
    # the bottom-up reading: A, then Levels (its lower child), then Curves, then Saturation on everything below
    out = cp.compose(layers, 0).rgb
    ctx = postengine.Ctx(cp, 0, (0, 0, 320, 200))
    want = postengine.f_curves(postengine.f_levels(pr.img[1], lo["params"], ctx), top["params"], ctx)
    assert np.allclose(out, np.clip(want, 0, 1), atol=1e-6)
    # moves: nest, reorder among the children, un-nest to a row of its own, a layer with its children
    L = postengine.move(layers, g["id"], a["id"], 2)                    # Saturation becomes A's lowest child
    assert [(x["name"], d) for x, d in postengine.rows(L)] == [("B", 0), ("A", 0), ("Curves", 1), ("Levels", 1),
                                                             ("Saturation", 1)] and g["target"] == a["id"]
    L = postengine.move(L, lo["id"], None, 0)                           # Levels out, at the top: everything below
    assert [(x["name"], d) for x, d in postengine.rows(L)] == [("Levels", 0), ("B", 0), ("A", 0), ("Curves", 1),
                                                             ("Saturation", 1)] and lo["target"] is None
    L = postengine.move(L, a["id"], None, 0)                            # A to the top, with its children
    assert [x["name"] for x in L] == ["Curves", "Saturation", "A", "Levels", "B"]
    assert L == normalize(list(L))                                      # always a valid list
    with pytest.raises(ValueError):
        postengine.move(L, a["id"], b["id"], 0)                         # a pixel layer is never nested
    with pytest.raises(ValueError):
        postengine.move(L, lo["id"], g["id"], 0)                        # nor in a filter


def test_a_composition_saved_by_the_previous_version_opens_as_it_was():
    """The saved format is the same list (a filter's target), with a new kind (image): a project saved before the
    Affinity panel opens with its filters nested where they applied, and composes the same image."""
    old = [{"id": 7, "kind": "filter", "filter": "sharpen", "name": "Sharpen", "visible": True, "opacity": 100.0,
            "mode": "Normal", "params": {"strength": 0.6, "radius": 1.2}, "mask": None, "target": None},
           {"id": 3, "kind": "workflow", "pid": 2, "part": None, "name": "Classic", "visible": True, "opacity": 40.0,
            "mode": "Screen", "base": None, "sliders": {}, "mask": None},
           {"id": 5, "kind": "filter", "filter": "saturation", "name": "Saturation", "visible": True, "opacity": 100.0,
            "mode": "Normal", "params": {"strength": 55.0, "hues": "All"}, "mask": None, "target": 4},
           {"id": 4, "kind": "workflow", "pid": 1, "part": None, "name": "Full", "visible": True, "opacity": 100.0,
            "mode": "Normal", "base": None, "sliders": {"sky": 3.0}, "mask": None}]
    back = postengine.load_layers(json.loads(json.dumps(old)), {1, 2})
    assert [(x["name"], d) for x, d in postengine.rows(back)] == [("Sharpen", 0), ("Classic", 0), ("Full", 0),
                                                                ("Saturation", 1)]
    again = postengine.save_layers(back)                                # saved again: the same file, the settings
    for d in again:                                                     # added since at their defaults (#23)
        if d.get("filter") == "saturation":
            assert d["params"].pop("protect") == 0
    assert again == old
    pr = ToyProvider()
    a = Composer(pr).compose(back, 1).rgb
    b = Composer(pr).compose(postengine.snapshot(old), 1).rgb
    assert np.array_equal(a, b)


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
    to the canvas: the Output block's image by the simple rendering, the Stars block's images by theirs."""
    from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings
    from astro_gpu.render.stars import StarSettings, StarsRenderEngine
    svc = LayerService(DEV)
    folders = {1: Folder(preps["full"]), ("stars", 1): Folder(preps["full_stars"]), 2: Folder(preps["classic"])}
    src = LayerSource(folders.get, svc, lambda b: {"faithful": {"sky": 0.24}}.get(b, b) if isinstance(b, str) else b)
    cv = src.canvas([1, 2])
    ox, oy = cv.origin
    out, stars = SimpleRenderEngine(preps["full"], device=DEV), StarsRenderEngine(preps["full_stars"], device=DEV)
    lay = new_workflow_layer(1, "Full", True, base="faithful")
    lay["sliders"] = {k: 2 for k in KEYS}                                    # every slider at the suggestion
    for level in (1, 2, 0):
        f = 2 ** level
        want = out.render(SimpleSettings(sky=0.24), 1.0 / f).final.numpy()
        w, h = cv.size(level)
        got = src.layer(lay, level, (0, 0, w, h))
        ref = np.moveaxis(want[:, oy // f: oy // f + h, ox // f: ox // f + w], 0, -1)
        assert np.array_equal(got, ref), level
    # a slider away from 2 moves the settings the table says, nothing else
    lay["sliders"] = {"sky": 3}
    moved = src.layer(lay, 1, (0, 0, *cv.size(1)))
    ref = out.render(SimpleSettings(sky=0.30), 0.5).final.numpy()
    ref = ref[:, oy // 2: oy // 2 + cv.size(1)[1], ox // 2: ox // 2 + cv.size(1)[0]]
    assert np.array_equal(moved, np.moveaxis(ref, 0, -1))
    # the Stars block's images: from its folder, each part its own rendering
    w, h = cv.size(1)
    for part in ("starless", "stars", "combined"):
        pl = new_workflow_layer(1, "Full", True, part=part)
        got = src.layer(pl, 1, (0, 0, w, h))
        ref = stars.render(StarSettings(), 0.5, part=part).final.numpy()[:, oy // 2: oy // 2 + h, ox // 2: ox // 2 + w]
        assert np.array_equal(got, np.moveaxis(ref, 0, -1)), part
    st = src.layer(new_workflow_layer(1, "Full", True, part="stars"), 1, (0, 0, w, h))
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
    d = np.abs(got - whole[60:300, 100:420]) * 255                           # a region is the crop within 2 levels
    assert float(np.percentile(d[64:-64, 64:-64], 99.9)) <= 2.0


def test_the_halpha_mask_comes_from_a_workflow_layer(preps):
    """The H-alpha map of the masks: from the bottom workflow layer, its Output block's image or its starless image;
    an image layer has none."""
    folders = {1: Folder(preps["full"]), ("stars", 1): Folder(preps["full_stars"]), 2: Folder(preps["classic"])}
    src = LayerSource(folders.get, LayerService(DEV))
    cv = src.canvas([2, 1])                                                    # the bottom one is Classic
    w, h = cv.size(1)
    ha = src.data_mask("halpha", 1, (0, 0, w, h))
    assert ha.shape == (h, w) and ha.max() > 0.5 and src.has("halpha")
    st = LayerSource(folders.get, LayerService(DEV))
    st.canvas([("stars", 1)])                                                  # the Stars block's images alone
    m = st.data_mask("halpha", 1, (0, 0, w, h))
    assert m is not None and m.max() > 0.5


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
    if post.worker.renderer is not None:                                    # the image layers' sources
        src.image_sources = dict(post.worker.renderer.source.image_sources)
    return Renderer(src).run(Request(kind="half", layers=snapshot(layers or post.view.layers), level=level, box=box,
                                     want=want))


def near(a: np.ndarray, b: np.ndarray) -> bool:
    """The same image to one grey level (the page's worker and the reference's renderer may round apart under a busy
    machine: two image tests failed once each by one level in a full run)."""
    return a.shape == b.shape and int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()) <= 1


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
        # GUI exploration 2: the layers it will have, named as the graph's Post block counts them, not computed yet
        assert "Nothing computed yet" in texts and "Graph  →" in texts
        assert "Full  ·  not computed yet" in texts and "Classic  ·  not computed yet" in texts
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
                 if "to compute" in b.text() and not b.isHidden()]
        assert [b.text() for b in links] == ['"Classic": to compute  →']
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
    # a workflow rendered later comes too (GUI exploration 1.3); only an image the user took out stays out (issues
    # #10, #52: model.post_removed, tests/client/test_graph47_client.py); a block's output sent from the graph comes
    post.model.paths.append(WF(4, "Later", None))
    post.model.pathsChanged.emit()
    assert "Later" in [lay["name"] for lay in v.layers]
    post.model.paths.append(WF(3, "Raw stack", "stacking"))
    post.model.pathsChanged.emit()
    # an image sent from a block arrives shown, as the Split images do (GUI exploration 2)
    assert [(lay["name"], lay["visible"]) for lay in v.layers][0] == ("Raw stack", True)
    assert [lay["visible"] for lay in v.layers if lay["name"] == "Later"] == [False]
    assert post.settle()
    # each workflow layer has its thumbnail, made from its image
    assert all(lay["id"] in v._lthumbs for lay in v.layers if lay["name"] != "Later")   # (no image of its own here)
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
    # the panel (Affinity Photo's): the filter is a child of its layer, indented under it
    assert [(r.lay["name"], r.depth) for r in v.rows] == [("Classic", 0), ("Full", 0), ("Saturation", 1)]
    assert rows[sat["id"]].sub.text() == "Normal · 100%" and rows[sat["id"]].toolTip() == 'Applies to "Full" only'
    assert rows[full["id"]].toggle is not None and rows[full["id"]].toggle.arrowType() == Qt.DownArrow
    assert rows[classic["id"]].toggle is None
    # another one, on everything below
    v.select(classic["id"], "layer")
    sh = v.add_filter("sharpen")
    assert sh["target"] == classic["id"]
    v.set_target(sh, None)
    assert sh["target"] is None and [lay["name"] for lay in v.layers] == ["Sharpen", "Classic", "Saturation", "Full"]
    assert next(r for r in v.rows if r.lay["id"] == sh["id"]).sub.text() == "on everything below · Normal · 100%"
    assert [(r.lay["name"], r.depth) for r in v.rows] == [("Sharpen", 0), ("Classic", 0), ("Full", 0),
                                                         ("Saturation", 1)]
    # the properties of a filter: its settings; where it applies is its place in the layers (issue #26)
    v.select(sat["id"], "layer")
    texts = [b.text() for b in widgets_of(v, QPushButton)]
    assert '"Full" only' not in texts and "All below" not in texts
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
    # the + Filter menu: the filters by family (the first Post's arcsinh stretch is no longer offered)
    fam = {a.text(): [x.text() for x in a.menu().actions()] for a in v.filter_menu.actions()}
    assert list(fam) == ["Stretch", "Contrast", "Color", "Detail", "Noise", "Combine"]
    assert sum(len(x) for x in fam.values()) == len([f for f in FILTERS.values() if not f.get("hidden")]) == 20  # #22, #23
    # Curves and Levels are stretches (2 November: where the user looked for them first)
    assert fam["Stretch"] == ["Generalised hyperbolic stretch", "Arcsinh stretch", "Histogram transformation", "Curves",
                              "Levels"]
    assert "Curves" not in fam["Contrast"] and fam["Combine"] == ["H-alpha to red"]
    assert "Highlight compression" in fam["Contrast"] and "Faint area smoothing" in fam["Noise"]
    assert "Spot healing" in fam["Detail"]
    assert all(x.isEnabled() for a in v.filter_menu.actions() for x in a.menu().actions())


def test_a_filter_added_in_the_page_does_not_move_the_image_until_a_slider_does(preps, qapp):
    """What the user did: + Filter > Saturation changed the picture before any slider was touched. Each filter,
    added with its defaults, leaves what the page shows as it was; the shots (English, French) show the new filter."""
    out = SHOTS / "new_filter"
    out.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "fr"):
        i18n.set_language(lang)
        s = Setup(preps, qapp)
        try:
            assert s.settle()
            v = s.view
            full = named(v, "Full")
            v.select(full["id"], "layer")
            assert s.settle()
            before = shown(v).copy()
            added = [v.add_filter(key) for key in FILTERS]          # each at its defaults (the composer test checks
            assert s.settle()                                         # them one by one): the page shows the same
            assert np.array_equal(shown(v), before), [x["filter"] for x in added]
            assert np.array_equal(shown(v), reference(s).rgb)
            for lay in added:                                       # hide them all, Saturation comes back for the shot
                lay["visible"] = False
            sat = next(x for x in added if x["filter"] == "saturation")
            sat["visible"] = True
            v.select(sat["id"], "layer")
            v.request(final_only=True)
            assert s.settle() and np.array_equal(shown(v), before)
            s.page.resize(1600, 950)
            for _i in range(4):
                qapp.processEvents()
                time.sleep(0.02)
            assert s.page.grab().save(str(out / f"post_new_filter_{lang}.png"))
            sat["params"]["strength"] = 60.0                        # and a slider does move it
            v.request(final_only=True)
            assert s.settle() and not np.array_equal(shown(v), before)
        finally:
            s.close()
            i18n.set_language("en")


def test_every_filter_and_every_mask_source_is_offered_without_the_stars_block(preps, qapp):
    """No filter and no mask part needs the Stars block any more (its stars are a layer of their own): on a page
    without it, the + Filter menu has no Stars family and the mask offers its five sources, all enabled."""
    s = Setup(preps, qapp, which=(2,))
    try:
        assert s.settle()
        v = s.view
        assert "Stars" not in [a.text() for a in v.filter_menu.actions()]
        assert all(x.isEnabled() for a in v.filter_menu.actions() for x in a.menu().actions())
        v.add_mask(v.layers[0])
        add = next(b for b in widgets_of(v, QPushButton) if b.text() == "+ Add to mask")
        acts = {a.text(): a.isEnabled() for a in add.menu().actions()}
        assert acts == {"H-alpha": True, "Luminance range": True, "Color range": True, "Linear gradient": True,
                        "Brush": True}
    finally:
        s.close()


def test_masks_with_parts_show_on_the_image_and_in_the_thumbnails(post):
    v = post.view
    full = named(v, "Full")
    v.select(full["id"], "layer")
    assert v.b_maskadd.text() == "+ Mask"
    v.mask_button()
    assert full["mask"] is not None and v.sel == (full["id"], "mask") and v.b_maskadd.text() == "Mask ✓"
    assert post.settle() and v.mask1 is not None and np.allclose(v.mask1, 1.0)         # empty: the layer shows everywhere
    assert "Empty: the layer shows everywhere." in [lb.text() for lb in widgets_of(v, QLabel)]
    for src in ("halpha", "luminance", "color", "gradient", "brush"):
        v.add_part(full, src)
    assert [p["src"] for p in full["mask"]["parts"]] == ["halpha", "luminance", "color", "gradient", "brush"]
    assert len([f for f in widgets_of(v, type(v.pbox.parentWidget())) if f.objectName() == "part"]) == 5
    v._part_set(full["mask"]["parts"][1], "op", "inter")                              # the luminance intersected
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
    assert full["mask"]["parts"][0]["src"] == "luminance"
    v.remove_mask(full)
    assert full["mask"] is None and v.b_maskadd.text() == "+ Mask"


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


def test_spot_healing_is_a_brush_healed_on_release_with_nothing_left_on_the_image(post):
    """Issue #20, Photoshop's Spot Healing Brush: a click or a stroke heals when let go (one Undo step); a stroke is
    spots every half radius, none healed from the stroke itself; Alt+click sets the source by hand for the strokes
    that follow (Photoshop's Healing Brush); nothing is drawn on the image afterwards."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPainter
    v = post.view
    f = v.add_filter("heal")
    v.select(f["id"], "layer")
    assert v.viewer.mode == "heal" and v._heal_layer() is f
    assert post.settle()
    before = shown(v).copy()
    n0 = len(v.undo)
    left = int(Qt.LeftButton.value)
    v._heal_press(200.0, 150.0, left, False)
    assert f["params"]["spots"] == [] and len(v.undo) == n0                            # healed when let go
    v._heal_release()
    spots = f["params"]["spots"]
    assert len(spots) == 1 and len(v.undo) == n0 + 1 and spots[0][2] == pytest.approx(v.heal_size / 2)
    assert np.hypot(spots[0][3], spots[0][4]) > spots[0][2]                            # the source beside the spot
    assert post.settle() and not np.array_equal(shown(v), before)
    # a stroke along a trail: spots every half radius, their sources off the stroke, one Undo step
    r = v.heal_size / 2
    v._heal_press(100.0, 300.0, left, False)
    v._heal_move(100.0 + 4 * r, 300.0)
    v._heal_release()
    trail = f["params"]["spots"][1:]
    assert len(trail) == 9 and len(v.undo) == n0 + 2
    assert all(abs(y + dy - 300.0) >= 2 * r - 1 or not (100.0 - r <= x + dx <= 100.0 + 5 * r)
               for x, y, _r, dx, dy in trail)
    # Alt+click: the source by hand; the next stroke keeps the offset (Photoshop's Aligned)
    v._heal_press(400.0, 100.0, left, True)
    assert len(f["params"]["spots"]) == 10                                              # sets the source only
    v._heal_press(300.0, 100.0, left, False)
    v._heal_release()
    assert f["params"]["spots"][-1][3:] == [100.0, 0.0]
    v.do_undo()
    assert len(v._heal_layer()["params"]["spots"]) == 10
    # nothing on the image when no stroke is painted
    img = QImage(64, 64, QImage.Format_ARGB32)
    img.fill(0)
    qp = QPainter(img)
    v._heal_src = v._heal_off = None
    v._heal_overlay(qp, v.viewer)
    qp.end()
    assert img.constBits().tobytes() == bytes(64 * 64 * 4)
    assert post.settle()


def test_a_stars_workflow_comes_as_its_stars_and_background_image(preps, qapp):
    """A workflow with the Stars block: its layer is its Stars block's recombined image, the stars put back in Screen;
    it shows only the sliders it has, the stars first, and the Stars blend under their title. Its starless image and
    its stars alone are layers of the same folder: the stars over the starless image in Screen are the combined one."""
    from PySide6.QtWidgets import QComboBox

    from astro_gpu.client.post3 import ContSlider
    s = Setup(preps, qapp, stars=True)
    try:
        assert s.settle()
        v = s.view
        assert [(lay["name"], lay.get("part")) for lay in v.layers] == [
            ("Classic", None), ("Full · Recombine", "combined")]
        full = v.layers[1]
        assert full["blend"] == "Screen" and full["mode"] == "Normal" and full["visible"]
        assert np.array_equal(shown(v), reference(s).rgb)
        v.select(full["id"], "layer")
        names = [x.name.text() for x in widgets_of(v, ContSlider)]
        assert names == ["Opacity", "Strength", "Small stars", "Star size", "Star color", "Sky", "Contrast",
                         "Faint tones",
                         "Saturation", "Colours", "H-alpha"]
        assert all(x.s.isEnabled() for x in widgets_of(v, ContSlider))
        combo = next(c for c in widgets_of(v, QComboBox) if c.currentData() == "Screen")
        before = shown(v).copy()
        combo.setCurrentIndex(combo.findData("Linear dodge (add)"))
        assert full["blend"] == "Linear dodge (add)"
        v.request(final_only=True)
        assert s.settle() and np.array_equal(shown(v), reference(s).rgb) and not np.array_equal(shown(v), before)
        # the Stars block's three images, from its folder
        src = LayerSource(lambda sid: s.folders.get(sid), LayerService(DEV))
        cv = src.canvas([("stars", 1)])
        w, h = cv.size(1)
        a = src.layer(new_workflow_layer(1, "Full", True, part="stars"), 1, (0, 0, w, h))
        b = src.layer(new_workflow_layer(1, "Full", True, part="starless"), 1, (0, 0, w, h))
        c = src.layer(new_workflow_layer(1, "Full", True, part="combined"), 1, (0, 0, w, h))
        assert a.min() >= 0 and a.max() > 0.2
        assert np.allclose(c, 1 - (1 - a) * (1 - b), atol=1e-4)
    finally:
        s.close()


def test_a_stars_and_background_layer_has_its_two_images_as_sub_layers_with_their_own_filters(preps, qapp):
    """Issue #3: under a Stars and background layer, its stars alone and its starless image are sub-layers; a filter
    added on one of them applies to that image only, before the two are put back together, and the eye of a
    sub-layer leaves its image out."""
    s = Setup(preps, qapp, stars=True)
    try:
        assert s.settle()
        v = s.view
        full = v.layers[1]
        assert [(r.lay["kind"], r.lay.get("sub"), r.depth) for r in v.rows][1:] == [
            ("workflow", None, 0), ("sub", "stars", 1), ("sub", "starless", 1)]
        before = shown(v).copy()
        v.select(full["id"], "stars")
        lv = v.add_filter("levels")
        lv["params"].update(black=0.0, white=0.5)
        assert lv["target"] == full["id"] and lv["sub"] == "stars"
        assert [(r.lay.get("sub"), r.depth) for r in v.rows][2:] == [("stars", 1), ("stars", 2), ("starless", 1)]
        # issue #21: an arrow on a row with rows under it only, down expanded, right collapsed
        subs = {r.lay["sub"]: r for r in v.rows if r.lay["kind"] == "sub"}
        assert subs["stars"].toggle.arrowType() == Qt.DownArrow and subs["starless"].toggle is None
        v.toggle_children((full["id"], "stars"))
        assert [(r.lay.get("sub"), r.depth) for r in v.rows][2:] == [("stars", 1), ("starless", 1)]
        assert next(r for r in v.rows if r.lay.get("sub") == "stars").toggle.arrowType() == Qt.RightArrow
        v.toggle_children((full["id"], "stars"))
        v.request(final_only=True)
        assert s.settle()
        # the stars alone brightened, the starless image untouched: the stars brighter, the sky as it was
        src = LayerSource(lambda sid: s.folders.get(sid), LayerService(DEV))
        w, h = src.canvas([("stars", 1)]).size(1)
        a = src.layer(new_workflow_layer(1, "Full", True, part="stars"), 1, (0, 0, w, h))
        b = src.layer(new_workflow_layer(1, "Full", True, part="starless"), 1, (0, 0, w, h))
        a2 = postengine.FILTERS["levels"]["fn"](a, lv["params"], None)
        want = postengine.Composer.blend("Screen", b, a2)
        alone = reference(s, postengine.only_layer(v.layers, full["id"])).rgb
        assert np.abs(alone.astype(np.float32) - want * 255).max() <= 2
        assert not np.array_equal(shown(v), before)
        # the stars sub-layer hidden: the starless image alone
        v.set_shown(full["id"], "stars", False)
        alone = reference(s, postengine.only_layer(v.layers, full["id"])).rgb
        assert np.abs(alone.astype(np.float32) - np.clip(b, 0, 1) * 255).max() <= 2
        # saved and loaded as it is
        kept = postengine.load_layers(postengine.save_layers(v.layers), {1, 2})
        assert next(x for x in kept if x.get("part") == "combined")["shown"] == {"stars": False, "starless": True}
        assert next(x for x in kept if x["kind"] == "filter")["sub"] == "stars"
        # taken out of the layer, a filter is no longer on one of its images
        v.set_target(lv, None)
        assert lv["target"] is None and lv["sub"] is None
    finally:
        s.close()


def test_a_starless_slider_leaves_the_stars_as_they_are(preps, qapp):
    """Issue #19: a Stars and background layer's two images are rendered and kept apart, each keyed on its own sliders:
    moving a slider of the starless image renders the starless image only, the stars stay (no blink); the two put
    together are the engine's recombined image."""
    s = Setup(preps, qapp, stars=True)
    try:
        assert s.settle()
        v = s.view
        full = v.layers[1]
        src = LayerSource(lambda sid: s.folders.get(sid), LayerService(DEV))
        w, h = src.canvas([("stars", 1)]).size(1)
        whole = src.layer(new_workflow_layer(1, "Full", True, part="combined"), 1, (0, 0, w, h))
        alone = reference(s, postengine.only_layer(v.layers, full["id"])).rgb
        assert np.abs(alone.astype(np.float32) - np.clip(whole, 0, 1) * 255).max() <= 1
        asked = []
        real = s.worker.renderer.source.layer
        s.worker.renderer.source.layer = lambda lay, *a, **k: (asked.append(lay.get("part")), real(lay, *a, **k))[1]
        full["sliders"]["contrast"] = 3.17                                        # values no other test uses (a cache)
        v.request(final_only=True)
        assert s.settle()
        assert asked and "stars" not in asked and "starless" in asked
        asked.clear()
        full["sliders"]["star_strength"] = 3.31
        v.request(final_only=True)
        assert s.settle()
        assert "stars" in asked and "starless" not in asked
    finally:
        s.close()


def test_post_undo_takes_back_a_graph_gesture_when_post_has_none(post):
    """GUI exploration 4: on the Post page, Ctrl+Z after a gesture made in the graph (Post having no step of its
    own) undoes that gesture, then Ctrl+Y redoes it: one Undo for the whole window."""
    v = post.view
    s = v.s
    v.undo.clear()
    v.redo.clear()
    assert v.do_undo() is False                                  # nothing anywhere: nothing happens
    p = s.paths[0]
    old = p.name
    names = {"undo": [old], "redo": []}                          # the workflows' Undo, as the model keeps it

    def back(src, dst):
        if not names[src]:
            return False
        names[dst].append(p.name)
        p.name = names[src].pop()
        return True
    s.undo, s.redo = (lambda: back("undo", "redo")), (lambda: back("redo", "undo"))
    s.can_undo, s.can_redo = (lambda: bool(names["undo"])), (lambda: bool(names["redo"]))
    p.name = "Renamed in the graph"                              # a gesture in the graph, before Post is shown
    v._undo_buttons()
    assert v.b_undo.isEnabled()
    assert v.do_undo() is True and s.paths[0].name == old
    assert v.do_redo() is True and s.paths[0].name == "Renamed in the graph"


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


def _drag(v, row, target_row, rel=0.5, x=None):
    """Drag ``row`` and drop it over ``target_row`` at ``rel`` of its height (and x in the panel, default: left)."""
    from PySide6.QtCore import QPoint
    holder = v.lbox.parentWidget()
    v.lbox.activate()                                                                    # the rows laid out, as on screen
    QApplication.processEvents()
    g = target_row.geometry()
    pt = QPoint(int(x if x is not None else g.left() + 20), int(g.top() + rel * g.height()))
    v.drag_start(row)
    v.drag_move(row, holder.mapToGlobal(pt))
    framed = [r for r in v.rows if r.property("drop")]                                  # a nest: the row says it
    text = v.drop_text.text() if v.drop_text.isVisible() else (framed[0].sub.text() if framed else "")
    seen = dict(v._drop or {}), v.drop_line.isVisible(), text, [bool(r.property("drop")) for r in v.rows]
    v.drag_end(row, holder.mapToGlobal(pt))
    return seen


def test_dragging_a_row_nests_or_unnests_a_filter_with_clear_hints(post):
    """Affinity Photo's drops: onto a layer nests a filter (the row is framed), between top-level rows it applies to
    everything below (a full-width line), between a layer's children it joins them (an indented line); a short label
    says which. A layer moves with its children and is never nested."""
    v = post.view
    full, classic = named(v, "Full"), named(v, "Classic")
    classic["visible"] = True
    v.select(full["id"], "layer")
    f = v.add_filter("scnr")
    v.add_filter("curves")
    for _i in range(3):
        post.qapp.processEvents()
    order = lambda: [(r.lay["name"], r.depth) for r in v.rows]                          # noqa: E731
    assert order() == [("Classic", 0), ("Full", 0), ("Curves", 1), ("Green removal", 1)]
    row = lambda lid: next(r for r in v.rows if r.lay["id"] == lid)                     # noqa: E731
    # to the top, between rows: a row of its own, on everything below
    d, line, text, frames = _drag(v, row(f["id"]), row(classic["id"]), 0.1)
    assert d["parent"] is None and d["index"] == 0 and line and text == "On all below"
    assert not any(frames) and f["target"] is None
    assert order() == [("Green removal", 0), ("Classic", 0), ("Full", 0), ("Curves", 1)]
    # onto the Classic row: nested in it, the row framed
    d, line, text, frames = _drag(v, row(f["id"]), row(classic["id"]), 0.5)
    assert d["parent"] == classic["id"] and not line and text == 'Applies to "Classic" only' and any(frames)
    assert f["target"] == classic["id"] and order() == [("Classic", 0), ("Green removal", 1), ("Full", 0),
                                                       ("Curves", 1)]
    # between Full's children (the indented line): it joins them, under Curves (applied before it)
    rows_now = v.rows
    d, line, text, _f = _drag(v, row(f["id"]), rows_now[-1], 0.9, x=200)
    assert d["parent"] == full["id"] and d["index"] == 1 and text == 'Applies to "Full" only'
    assert d["x0"] > v.rows[0].geometry().left() + 40                                     # the line is indented
    assert order() == [("Classic", 0), ("Full", 0), ("Curves", 1), ("Green removal", 1)]
    assert [x["name"] for x in v.layers] == ["Classic", "Curves", "Green removal", "Full"]
    # a layer moves with its children; dropped among another layer's children it goes after that group
    _drag(v, row(full["id"]), row(classic["id"]), 0.1)
    assert order() == [("Full", 0), ("Curves", 1), ("Green removal", 1), ("Classic", 0)]
    assert full["visible"] and classic["visible"]
    # one undo step per drop, nothing when it lands where it was
    n = len(v.undo)
    _drag(v, row(classic["id"]), row(classic["id"]), 0.2)
    assert len(v.undo) == n
    v.do_undo()
    assert [(r.lay["name"], r.depth) for r in v.rows][0] == ("Classic", 0)
    # the composition follows the panel: read from the bottom up
    assert post.settle() and np.array_equal(shown(v), reference(post).rgb)
    # ▾ / ▸ hides a layer's children in the panel only
    v.toggle_children(full["id"])
    assert [r.lay["name"] for r in v.rows] == ["Classic", "Full"] and row(full["id"]).toggle.arrowType() == Qt.RightArrow
    assert "2 filters" in row(full["id"]).sub.text()
    v.toggle_children(full["id"])
    assert len(v.rows) == 4


def test_the_context_menu_and_the_properties_nest_and_unnest(post):
    v = post.view
    full, classic = named(v, "Full"), named(v, "Classic")
    v.select(classic["id"], "layer")
    f = v.add_filter("levels")
    assert v.scope_choices(f) == [('"Classic" only', classic["id"]), ("All below", None)]
    v.set_target(f, None)                                                                # out: just above its layer
    assert [(r.lay["name"], r.depth) for r in v.rows][:2] == [("Levels", 0), ("Classic", 0)]
    assert v.scope_choices(f)[0] == ('"Classic" only', classic["id"])                    # the layer just under it
    v.set_target(f, full["id"])
    assert f["target"] == full["id"] and [(r.lay["name"], r.depth) for r in v.rows][-1] == ("Levels", 1)


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
    v.delete()                                                                           # the last pixel layer stays
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
    for k in ("Sky", "Contrast", "Saturation", "Noise smoothing"):                    # the Output block's image
        assert sl[k].s.isEnabled(), k
    for k in ("Strength", "Star size", "Star color"):
        assert not sl[k].s.isEnabled(), k
    assert sl["Sky"].val.text() == "0" and sl["Sky"].lo == 0 and sl["Sky"].hi == 4
    sl["Sky"].set_value(3.4)
    assert sl["Sky"].val.text() == "+1.4"
    v.select(named(v, "Classic")["id"], "layer")
    sl = sliders()
    assert sl["Sky"].s.isEnabled() and sl["Noise smoothing"].s.isEnabled()
    for k in ("Strength", "Star size", "Star color"):
        assert not sl[k].s.isEnabled() and not sl[k].why.isHidden()
    assert sl["Star size"].why.text() == "the star sliders work on the stars of the Stars/background block"
    post.model.paths.append(WF(3, "Raw stack", "stacking"))                              # pulled from Stacking
    post.model.pathsChanged.emit()
    v.select(named(v, "Raw stack")["id"], "layer")
    sl = sliders()
    assert sl["Noise smoothing"].why.text() == "this layer leaves the graph before the Denoise block"
    assert sl["Star size"].why.text() == "this layer leaves the graph before the Stars/background block"
    lay = named(v, "Classic")                                                            # a double-click: the suggestion
    v.select(lay["id"], "layer")
    sky = sliders()["Sky"]
    sky.done.emit(3.0)
    assert lay["sliders"]["sky"] == pytest.approx(3.0)
    sky.name.mouseDoubleClickEvent(None)
    assert "sky" not in lay["sliders"] and sky.val.text() == "0"
    # issue #16: a double-click on the slider itself too (Lightroom's)
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    sky = sliders()["Sky"]
    sky.done.emit(3.5)
    assert lay["sliders"]["sky"] == pytest.approx(3.5)
    sky = sliders()["Sky"]
    QTest.mouseDClick(sky.s, Qt.LeftButton, Qt.NoModifier, QPoint(sky.s.width() // 2, sky.s.height() // 2))
    assert "sky" not in lay["sliders"]


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
    refines = [i for i, x in enumerate(log) if x[0] == "refine"]
    # on the CPU: drafts at an eighth while it moves, then the exact quarter, then the screen image (the half)
    assert v.draft_level == 3 and drafts and ("draft", 3) in log and halves and halves[-1] > drafts[-1]
    assert refines and ("refine", 2) in log and drafts[-1] < refines[-1] < halves[-1]
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


def _stage_image(preps):
    """A stage's linear image on the grid of the Full layer (3, 480, 720): its calibrated nebula, as a stack is."""
    return np.load(preps["full"] / "lin.npy")


def test_an_image_sent_from_a_stage_is_a_normal_layer(post, preps, tmp_path):
    """add_image_layer: Inspect sends a stage's image (an array, a file or a function); it is a layer on top, stretched
    automatically and moved by Sky, Contrast and Saturation (linked or each channel), with a blend mode, a mask and
    filters of its own; the project keeps its key, never its pixels, and finds the image again from the store."""
    from astro_gpu.client.post3 import ContSlider
    from astro_gpu.render.simple import SimpleRenderEngine
    from astro_gpu.render.sliders import settings_at
    v = post.view
    arr = _stage_image(preps)
    lay = post.page.add_image_layer("Stack", arr, "stack/0123abcd")
    assert v.layers[0] is lay and lay["kind"] == "image" and lay["visible"] and v.sel == (lay["id"], "layer")
    assert post.settle()
    assert near(shown(v), reference(post).rgb)
    # its pixels: the simple rendering's automatic stretch of the array, cut to the canvas
    eng = SimpleRenderEngine.from_array(arr)
    want = eng.render(settings_at("simple", None, {}, False), 0.5).final.numpy()
    ox, oy = 24, 8
    W2, H2 = v.src.canvas[0] // 2, v.src.canvas[1] // 2
    ref = np.moveaxis(want[:, oy // 2:oy // 2 + H2, ox // 2:ox // 2 + W2], 0, -1)
    got = post.worker.renderer.source.layer(lay, 1, (0, 0, W2, H2))
    assert np.array_equal(got, ref)
    # its settings: the channels and three sliders, nothing that needs a workflow
    names = [x.name.text() for x in widgets_of(v, ContSlider) if x.name.text() not in ("Opacity",)]
    assert names == ["Sky", "Contrast", "Saturation"]
    from PySide6.QtWidgets import QCheckBox
    # one name and one kind of control with the Output block's in the graph (GUI exploration 2)
    boxes = [c for c in widgets_of(v, QCheckBox) if c.text() == "Linked channels"]
    assert len(boxes) == 1 and boxes[0].isChecked()
    boxes[0].setChecked(False)
    assert lay["linked"] is False and v.undo
    v.do_undo()
    lay = v.layer(lay["id"])
    assert lay["linked"] is True
    before = shown(v).copy()
    lay["sliders"]["sky"] = 3.4
    v.request(final_only=True)
    assert post.settle() and not np.array_equal(shown(v), before)
    assert near(shown(v), reference(post).rgb)
    v._set(lay, "linked", False)
    assert post.settle() and near(shown(v), reference(post).rgb)
    # blendable, maskable, its own filters
    lay["mode"], lay["opacity"] = "Screen", 60.0
    v.add_mask(lay)
    v.add_part(lay, "gradient")
    v.select(lay["id"], "layer")
    f = v.add_filter("curves")
    assert f["target"] == lay["id"] and [(r.lay["name"], r.depth) for r in v.rows][:2] == [("Stack", 0),
                                                                                        ("Curves", 1)]
    assert post.settle() and near(shown(v), reference(post).rgb)
    assert "image" in next(r for r in v.rows if r.lay["id"] == lay["id"]).sub.text()
    # saved by its key, no pixels
    saved = post.model.p.post["layers"]
    img = next(x for x in saved if x["kind"] == "image")
    assert img["key"] == "stack/0123abcd" and img["linked"] is False and len(json.dumps(saved)) < 20_000
    # a file, a function
    np.save(tmp_path / "den.npy", arr[:, :, ::-1].copy())
    lay2 = post.page.add_image_layer("Denoised", tmp_path / "den.npy", "denoise/feed")
    lay3 = post.page.add_image_layer("Mono", lambda: arr[1] * 2, "fn:mono")
    assert post.settle() and not v._missing_images
    assert near(shown(v), reference(post).rgb)
    assert [x["name"] for x in v.layers][:2] == ["Mono", "Denoised"] and lay2["visible"] and lay3["visible"]


def test_an_image_layer_is_found_again_from_the_store_or_said_missing(preps, qapp, tmp_path):
    root = tmp_path / "proj"
    d = root / "store" / "stack" / "0123abcd"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}")
    arr = _stage_image(preps)
    np.save(d / "drz_full.npy", arr)
    assert rsv.store_image_resolver(root)("stack/0123abcd") == d / "drz_full.npy"
    assert rsv.store_image_resolver(root)("stack/0123abcd/drz_full") == d / "drz_full.npy"
    assert rsv.store_image_resolver(root)("stack/nothere") is None
    assert rsv.store_image_resolver(root)("../stack/0123abcd") is None
    s = Setup(preps, qapp)
    s2 = s3 = None
    try:
        assert s.settle()
        lay = s.page.add_image_layer("Stack", arr, "stack/0123abcd")
        lay["sliders"]["contrast"] = 3.0
        s.view.push()
        assert s.settle()
        img = shown(s.view).copy()
        model = s.model
        s.close()
        # opened again: the key is read from the project's store
        s2 = Setup.__new__(Setup)
        s2.qapp, s2.folders, s2.model = qapp, s.folders, model
        s2.worker = RenderWorker(None, resolver=lambda pid: s2.folders.get(pid), device=DEV)
        s2.page = PostPage3(model, worker=s2.worker)
        s2.page.set_image_resolver(rsv.store_image_resolver(root))
        s2.page.resize(1500, 900)
        s2.page.show()
        s2.view = s2.page.view
        assert s2.settle()
        assert s2.view.layers[0]["kind"] == "image" and near(shown(s2.view), img)
        s2.close()
        s2 = None
        # without its result: the layer stays, said not found; the others are composed
        s3 = Setup.__new__(Setup)
        s3.qapp, s3.folders, s3.model = qapp, s.folders, model
        s3.worker = RenderWorker(None, resolver=lambda pid: s3.folders.get(pid), device=DEV)
        s3.page = PostPage3(model, worker=s3.worker)
        s3.page.resize(1500, 900)
        s3.page.show()
        s3.view = s3.page.view
        assert s3.settle()
        row = next(r for r in s3.view.rows if r.lay["kind"] == "image")
        assert "not found" in row.sub.text() and s3.view.src.img1 is not None
    finally:
        for x in (s2, s3):
            if x is not None:
                x.close()


def test_post_opens_on_an_image_layer_alone(preps, qapp):
    s = Setup(preps, qapp, rendered=False)
    try:
        assert s.page.pages.currentIndex() == 0
        s.page.add_image_layer("Stack", _stage_image(preps), "stack/0123abcd")
        assert s.page.pages.currentIndex() == 1
        assert s.settle()
        assert s.view.src.canvas == (720, 480) and shown(s.view).shape == (240, 360, 3)
    finally:
        s.close()


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
        v.push()                                 # (a slider let go: the change, then the undo step)
        qapp.processEvents()                     # the composition is saved just after the edit, with it
        assert s.model.touched >= 1                                                      # the model is told to save
        saved = s.model.p.post["layers"]
        assert [lay["name"] for lay in saved] == ["Classic", "Saturation", "Full"] and json.dumps(saved)
        assert saved[1]["mask"]["parts"][0]["strokes"]
        assert saved[2]["sliders"]["sky"] == 3.2
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


def test_an_edit_is_saved_with_its_change_not_the_state_before_it(post, qapp):
    """push() takes the undo step before a change; the composition is saved after it (it used to be saved before it,
    so that the last edit was lost when the project closed or the language changed)."""
    v = post.view

    def saved():
        qapp.processEvents()
        return [lay["name"] for lay in post.model.p.post["layers"]]
    v.select(named(v, "Full")["id"], "layer")
    f = v.add_filter("saturation")
    assert saved() == ["Classic", "Saturation", "Full"]
    v.select(f["id"], "layer")
    v.delete()
    assert saved() == ["Classic", "Full"]
    v.rename(named(v, "Classic"), "Mine")
    assert saved() == ["Mine", "Full"]
    v.do_undo()
    assert saved() == ["Classic", "Full"]


def test_the_layer_list_is_as_tall_as_its_rows_and_opens_unscrolled(post, qapp):
    v = post.view
    full = named(v, "Full")
    for key in ("curves", "levels", "saturation"):
        v.select(full["id"], "layer")
        v.add_filter(key)
    v.select(full["id"], "layer")                              # the bottom layer selected: the list is made to show it
    for _i in range(3):
        qapp.processEvents()
    sb = v.lsc.verticalScrollBar()
    assert len(v.rows) == 5 and sb.maximum() == 0 and sb.value() == 0     # 5 rows fit: nothing hidden, nothing cut
    assert v.rows[0].geometry().top() >= 0
    assert v.lsc.height() >= v.lsc.widget().sizeHint().height()


def test_an_image_from_inspect_keeps_its_neutral_sky_and_its_name_follows_the_language(post, preps, qapp):
    from astro_gpu.client import i18n

    def src():
        return np.ones((3, 64, 96), np.float32)
    src.neutral = True                                          # shown with a neutral sky in Inspect (services/inspect)
    lay = post.page.add_image_layer("Stack", src, "stack/feed")             # Inspect names it in English (tune._send)
    assert lay["linked"] is False                                           # its channels stretched apart
    i18n.set_language("fr")
    try:
        post.view.rebuild()
        row = next(r for r in post.view.rows if r.lay["id"] == lay["id"])
        assert row.n.text() == i18n._("Stack") != "Stack"                  # shown in French
        assert not {t for t in i18n.missing() if "DSC" in t}
        named_file = post.page.add_image_layer("DSC0001.ARW", src, "file:DSC0001.ARW")
        post.view.rebuild()
        assert next(r for r in post.view.rows if r.lay["id"] == named_file["id"]).n.text() == "DSC0001.ARW"
        assert "DSC0001.ARW" not in i18n.missing()                         # a file name is not asked for
    finally:
        i18n.set_language("en")
    plain = post.page.add_image_layer("Denoised stack", lambda: np.ones((3, 64, 96), np.float32), "denoise/feed")
    assert plain["linked"] is True


def test_a_workflow_that_goes_takes_its_layer_and_the_filters_retarget(post):
    v = post.view
    v.select(named(v, "Full")["id"], "layer")
    f = v.add_filter("scnr")
    post.model.paths = [w for w in post.model.paths if w.pid != 1]
    post.model.pathsChanged.emit()
    assert [lay["name"] for lay in v.layers] == ["Classic", "Green removal"]
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
        assert not d.c_wcs.isChecked() and not d.c_wcs.isEnabled()                                    # only FITS has a header
        assert d.c_hist.isChecked() and not d.c_layers.isChecked() and not d.c_masks.isChecked()
        assert d.spec_boxes == [] and d.go.text() == "Export" and d.go.isEnabled()
        d.fmt.setCurrentIndex(d.fmt.findData("fits32f"))
        assert d.dest.text().endswith("/HeartSoul/HeartSoul.fits") and d.c_wcs.isEnabled() and d.c_wcs.isChecked()
        assert not d.c_grid.isEnabled() and not d.c_grid.isChecked()            # drawn into a picture, not a FITS
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
        assert "XISF 32-bit float  ◇" in items and len(d.spec_boxes) == 1                  # (the history is in the header now)
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
        n = len(d.files)
        assert d.status.text() == f"{n} file{'' if n == 1 else 's'} written to {s.out.as_posix()}"
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
            d.c_hist.setChecked(fmt == "fits32f")                                            # (FITS: in its header too)
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
        # the rows bottom up (FITS, Siril), the history of the composition in the header
        assert h[0].header["ROWORDER"] == "BOTTOM-UP"
        assert np.abs(np.moveaxis(h[0].data[:, ::-1], 0, -1) - np.clip(expect, 0, 1)).max() < 1e-6
        hist_lines = [str(x) for x in h[0].header["HISTORY"]]
    assert any(x.startswith("layer Full") for x in hist_lines) and any("its filter Saturation" in x for x in hist_lines)
    assert "sky=1" in "".join(hist_lines), hist_lines                                 # (a line may wrap in it)
    ox, oy = 24, 8                                                                           # where the canvas starts on the grid
    for x, y in ((0, 0), (300, 200), (cv[0] - 1, cv[1] - 1)):
        assert np.allclose(w.pixel_to_world_values(ox + x, oy + y), w2.pixel_to_world_values(x, cv[1] - 1 - y),
                           atol=1e-9)
    # nothing was written but the files, in the folder chosen: the project and everything else is as it was
    after = tree(s.out.parent)
    new = sorted(set(after) - set(before))
    assert all(p.startswith("chosen/") for p in new), new
    assert {k: v for k, v in before.items()} == {k: after[k] for k in before}
    assert sorted(Path(p).name for p in new) == sorted(
        ["heart_16bit.tif", "heart_32bit_float.tif", "heart.png", "heart.jpg", "heart.fits", "heart_history.json"])
    assert not list((s.root / "export").glob("*")) if (s.root / "export").exists() else True


def test_the_export_reduced_matches_the_half_resolution_preview(exporting, preps):
    import tifffile

    from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings
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
    # the engine's own half rendering differs from its reduced full rendering on the stars (a point star reduced
    # then stretched is not a stretched star reduced: about 15 levels at p99.9 on this preparation); Post adds
    # nothing beyond that
    e = SimpleRenderEngine(preps["full"], device=DEV)
    half = e.render(SimpleSettings(), 0.5).final.numpy()
    red = cv2.resize(np.moveaxis(e.render(SimpleSettings(), 1.0).final.numpy(), 0, -1), (half.shape[2], half.shape[1]),
                     interpolation=cv2.INTER_AREA)
    own = float(np.percentile(np.abs(np.moveaxis(half, 0, -1) - red) * 255, 99.9))
    assert float(np.percentile(diff, 99.9)) <= own + 1.0 and diff.mean() < 1.0, (float(np.percentile(diff, 99.9)), own)


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
        d.fmt.setCurrentIndex(d.fmt.findData("tiff16"))
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
    from astro_gpu.client.postfilters import curves as pc
    found: dict[str, list[str]] = {}
    for name in POST_MODULES:
        tree = ast.parse((CLIENT / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_" and node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    found.setdefault(a.value, []).append(f"{name}.py:{node.lineno}")
    for f in FILTERS.values():
        for t in (f["name"], f["family"], f["source"]) + tuple(q["label"] for q in f["params"]):
            found.setdefault(t, []).append("FILTERS")
        for q in f["params"]:
            for c in q.get("choices", []):
                found.setdefault(c, []).append("FILTERS")
    for t in list(postengine.MASK_SOURCES.values()) + list(postengine.OPS.values()) + postengine.MODES + \
            postengine.DIRECTIONS + [it[1] for items in postengine.PART_SETTINGS.values() for it in items] + \
            list(ed.FORMATS) + list(ed.PROFILES) + list(ed.SIZES):
        found.setdefault(t, []).append("tables")
    from astro_gpu.client.model import POST_PART_NAMES
    for t in ("Paint", "Erase", rsv.NOTE_CPU_TEXT, " (copy)") + tuple(POST_PART_NAMES.values()) + postengine.BLENDS:
        found.setdefault(t, []).append("constants")
    for t in list(pc.CHANNEL_TIPS.values()) + pc.INTERPS + ["Classic tone curves", "Astrophotography"] + \
            [x for p in pc.builtin_presets() for x in (p["name"], p["source"])] + list(postengine.OLD_NAMES):
        found.setdefault(t, []).append("curves")
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
    for why in {w for kind in KINDS for w in availability(kind, False).values() if w}:
        assert why in cat, why


def test_the_french_switch_translates_every_thing_the_page_and_the_dialog_show(preps, qapp, tmp_path):
    """The page rebuilt in French, with every panel opened (every filter, 6 mask sources, the brush, the greyed sliders,
    the empty page, the menus, the Export dialog in Spec mode and its messages): not one text is missing."""
    from astro_gpu.client.post3 import ContSlider
    prefs.set("spec_mode", True)
    i18n.set_language("fr")
    root = tmp_path / "ws" / "HeartSoul"
    root.mkdir(parents=True)
    empty = Setup(preps, qapp, rendered=False)
    try:
        shown = [lb.text() for lb in empty.page.empty.findChildren(QLabel)]   # 2nd round: the layers to come, named
        assert "Rien de calculé" in shown and any("pas encore calculé" in t for t in shown), shown
    finally:
        empty.close()
    s = Setup(preps, qapp, root=root)
    try:
        assert s.settle()
        v = s.view
        assert v.b_filter.text() == "+ Filtre" and v.b_cmp.text() == "Avant / après" and v.b_maskadd.text() == "+ Masque"
        assert [lb.text() for lb in v.findChildren(QLabel)][:1] != []
        for k in FILTERS:                                                               # every filter's panel
            v.select(named(v, "Full")["id"], "layer")
            fl = v.add_filter(k)
            assert fl is not None
            widgets_of(v, QLabel)
        v.select(named(v, "Classic")["id"], "layer")                                    # greyed sliders and their reasons
        sl = {x.name.text(): x for x in widgets_of(v, ContSlider)}
        assert any("bloc Étoiles/fond" in x.why.text() for x in sl.values())
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
        assert "Voir masque" in [v.b_mask.text()] and any("Ajouter au masque" in t for t in post_texts)
        for lay in list(v.layers):                                                      # every row, in French
            v.sub_text(lay)
            v.thumb(lay)
        v.build_filter_menu()
        assert [a.text() for a in v.filter_menu.actions()] == ["Étirement", "Contraste", "Couleur", "Détail",
                                                               "Bruit", "Combinaison"]
        d = ExportDialog(v)
        try:
            d.dest.setText((root / "store" / "x.tif").as_posix())
            assert "résultats du projet" in d.warn.text()
            d.dest.setText((tmp_path / "out" / "a_16bit.tif").as_posix())
            run_dialog(s, d)
            assert d.status.text().startswith(("2 fichiers écrits dans", "1 fichier écrit dans")), d.status.text()
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
def test_the_post_modules_import_only_the_client_libraries():
    import ast
    for name in POST_MODULES:
        src = (CLIENT / f"{name}.py").read_text(encoding="utf-8")
        for bad in ("alt2", "alt3", "data.json", "datatext", "from model import", "import project", "Demo"):
            assert bad not in src, (name, bad)
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".")[0]
                assert top in {"__future__", "collections", "dataclasses", "datetime", "json", "math", "os", "pathlib", "re",
                               "threading", "time", "typing", "PySide6", "numpy", "cv2", "torch", "astropy",
                               "astro_gpu"}, (name, node.module)
    # the GUI thread never renders: no engine, no composition in the widgets' module
    for name in ("post3", "curveseditor"):
        gui = (CLIENT / f"{name}.py").read_text(encoding="utf-8")
        assert "Composer(" not in gui and "LayerService" not in gui and "api.render_layer" not in gui
        assert "torch" not in gui, name


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
    v.add_part(full, "halpha")
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
needs_real = needs_test_project


def real_project():
    """The test project with its computed Classic workflow added in memory (TEST_PROJECT_CLASSIC): nothing is saved."""
    p = api.open_project(REAL)
    api.add_workflow(p, *TEST_PROJECT_CLASSIC)
    return p


class RealSetup(Setup):
    """A page on the test project (Full: render_prep; Classic: render_simple), the model read only: nothing is
    written into the project."""

    def __init__(self, qapp, faithful=True):
        from astro_gpu.client.model import ProjectModel
        from astro_gpu.client.services.worker import ApiWorker
        self.qapp = qapp
        self.project = real_project()
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
def test_real_the_layers_render_filters_masks_and_undo(real):
    s = real
    v = s.view
    # Full's layer: its Output block's image, or its Stars and background image when it has the Stars block
    assert [(lay["name"].split(" · ")[0], lay["visible"]) for lay in v.layers] == [("Classic", False), ("Full", True)]
    assert v.src.canvas == (6216, 4224) and shown(v).shape == (2112, 3108, 3)
    full, classic = v.layers[1], named(v, "Classic")
    v.select(full["id"], "layer")
    sat = v.add_filter("saturation")
    sat["params"]["strength"] = 60.0
    v.select(classic["id"], "layer")
    classic["visible"], classic["opacity"], classic["mode"] = True, 40.0, "Screen"
    sh = v.add_filter("sharpen")
    v.set_target(sh, None)
    assert s.settle(600)
    # the masks: H-alpha, luminance, brush
    v.select(full["id"], "layer")
    v.add_mask(full)
    for src in ("halpha", "luminance", "brush"):
        v.add_part(full, src)
    brush = full["mask"]["parts"][-1]
    for k in range(10):
        brush["strokes"].append((3000.0 + 40 * k, 2000.0, 60.0, 50.0, True))
    full["mask"]["parts"][1]["op"] = "sub"
    v.request(final_only=True)
    assert s.settle(600)
    m = v.mask1
    assert m.shape == (2112, 3108) and 0 <= m.min() <= m.max() <= 1 and m.max() > 0.9 and 0.01 < m.mean() < 0.99
    ha = s.worker.renderer.source.data_mask("halpha", 1, (0, 0, 3108, 2112))
    assert ha.max() > 0.9 and 0.02 < ha.mean() < 0.5
    v.remove_mask(full)
    assert full["mask"] is None
    v.do_undo()                                                                          # the mask comes back
    assert v.layer(full["id"])["mask"] is not None


@needs_real
@pytest.mark.slow
def test_real_the_release_render_is_the_engines_bit_for_bit(qapp):
    """api.render_layer with the 'faithful' rendering and every slider at 2 is the simple rendering of the Output block
    (SimpleRenderEngine.render(faithful)), and the layers of Post are those renderings."""
    from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings
    p = real_project()
    p.render_bases = {"faithful": {"sky": 0.24}}
    dev = "cpu" if REAL_DEVICE == "cpu" else rsv.choose_device("auto")
    svc = LayerService(dev)
    lay = api.render_layer(p, 1, "faithful", {k: 2 for k in KEYS}, 0.25, service=svc)
    folder = Path(api.stage_outputs(p, 1, rsv.OUTPUT_STAGE).dir)
    eng = SimpleRenderEngine(folder, device=dev)
    want = eng.render(SimpleSettings(sky=0.24), 0.25)
    assert torch.equal(torch.from_numpy(lay.rgb), want.final.cpu()), "render_layer(faithful, 2) is not the engine's render"
    # the page's layer, cut to the canvas, is the same image
    src = LayerSource(rsv.api_resolver(p), svc, rsv.bases_of(p))
    cv = src.canvas([w.id for w in p.workflows])
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
    # the reduced export against the screen, on one layer: the engine's own resolution differences are measured on
    # that layer (below), so what Post adds is what is left
    v.select(full["id"], "layer")
    classic["visible"] = False
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
    ref_src.canvas([w.id for w in s.project.workflows])
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
    def export(job):
        t0 = time.monotonic()
        n = len(answers)
        s.worker.export(job)
        while len(answers) == n and time.monotonic() - t0 < 1500:
            s.qapp.processEvents()
            time.sleep(0.02)
        done["+".join(job.format) if isinstance(job.format, list) else job.format] = round(time.monotonic() - t0, 1)
        assert answers[-1][1] == "", answers[-1]

    # the dialog makes the job: the image reduced to the screen's size
    d = ExportDialog(v)
    d.dest.setText((out / "heart_16bit.tif").as_posix())
    d.size_cb.addItem("Screen", 3108)
    d.size_cb.setCurrentIndex(d.size_cb.count() - 1)
    reduced = d.job()
    d.reject()
    d.deleteLater()
    export(reduced)
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
    # JPEG and FITS at full size of a composition (two layers, a filter), the FITS with its astrometric solution
    v.add_filter("saturation")["params"]["strength"] = 50.0
    classic["visible"], classic["opacity"] = True, 30.0
    v.push()
    assert s.settle(600)
    cw, ch = v.src.canvas
    export(dc.replace(reduced, format=["jpeg", "fits32f"], size=0, wcs=True, name="heart", history=False))
    with Image.open(out / "heart.jpg") as im:
        assert im.size == (cw, ch) and im.mode == "RGB"
    with fits.open(out / "heart.fits") as h:
        assert h[0].data.dtype.itemsize == 4 and h[0].data.shape == (3, ch, cw)
        w2 = WCS(h[0].header, naxis=2).celestial
    ox, oy = ref_src.canvas([w.id for w in s.project.workflows]).origin               # where the canvas starts on the grid
    for x, y in ((0, 0), (cw // 2, ch // 2), (cw - 1, ch - 1)):
        ra, dec = api.pixel_to_sky(s.project, 1, ox + x, oy + y)
        ra2, dec2 = w2.pixel_to_world_values(x, ch - 1 - y)                                   # (FITS rows bottom up)
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
