"""Issue #43, the blocks redesign, in the engine: sessions calibrated each with its own frames then joined, cut blocks
and the blocks greyed by a cut, a workflow that starts from an image. On the real v2 stage graph; the frames are
empty files (the keys read their names and sizes only), the joined frames and the image synthetic arrays."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from astro_gpu.core import project2
from astro_gpu.core import workflow as wfl
from astro_gpu.core.journal import Journal
from astro_gpu.core.project2 import Session
from astro_gpu.core.stage import Context, StageError
from astro_gpu.core.store import Keys, Store, StoreProject
from astro_gpu.core.workflow import TEMPLATES

FULL = TEMPLATES["Full"]
CLASSIC = TEMPLATES["Classic stack"]


def frames(folder: Path, names: dict[str, int]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for n, size in names.items():
        (folder / n).write_bytes(b"\0" * size)
    return folder


@pytest.fixture
def nights(tmp_path):
    """Two nights of lights, the same darks and flats for both (the split of one session, the user's test D)."""
    raw = tmp_path / "raw"
    return {"n1": frames(raw / "n1", {"a1.ARW": 100, "a2.ARW": 110}), "n2": frames(raw / "n2", {"b1.ARW": 120}),
            "darks": frames(raw / "darks", {"d1.ARW": 90}), "flats": frames(raw / "flats", {"f1.ARW": 80})}


def project(root: Path, nights: dict, mode: str, sessions=("n1", "n2"), options=FULL) -> project2.Project2:
    p = project2.create(root, "p")
    p.stack_mode = mode
    for i, n in enumerate(sessions, 1):
        p.sessions.append(Session(i, n, lights=[str(nights[n])], darks=[str(nights["darks"])],
                                  flats=[str(nights["flats"])]))
    p.add_workflow("Full", options)
    project2.save(p)
    return p


# ---------------------------------------------------------------------------------------------- sessions

def test_two_sessions_calibrated_each_on_its_own_then_joined(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session")
    assert p.separate_sessions() == [1, 2]
    k = Keys(p)
    ch = k.chain(1)
    assert ch[:7] == ["inventory@1", "inventory@2", "calibrate@1", "calibrate@2", "sessions", "register", "stack"]
    assert k.deps(1, "calibrate@2") == ("inventory@2",)
    assert k.deps(1, "sessions") == ("inventory@1", "inventory@2", "calibrate@1", "calibrate@2")
    assert k.deps(1, "register") == ("sessions",)                         # it reads the join as "calibrate"
    assert k.deps(1, "stack") == ("sessions", "register")
    assert k.deps(1, "astrometry")[0] == "sessions"                       # and as "inventory"
    blocks = [b for b, _ in k.rules.chains.blocks_in(ch)]
    assert blocks[:4] == ["calibration@1", "calibration@2", "registration", "selection"]
    assert k.rules.chains.run_to("calibration@2", ch) == ["inventory@2", "calibrate@2"]
    # a session's calibration is that of a project holding that session alone: the same key, the same result
    alone = project(tmp_path / "q", nights, "together", sessions=("n1",))
    assert k.key(1, "calibrate@1") == Keys(alone).key(1, "calibrate")
    assert k.key(1, "calibrate@2") != k.key(1, "calibrate@1")
    # the results live in the stage's folder
    assert Store(p.root).dir("calibrate@2", k.key(1, "calibrate@2")).parent.name == "calibrate"


def test_one_session_or_the_old_mode_keep_the_chain_and_keys_of_before(tmp_path, nights):
    together = project(tmp_path / "a", nights, "together")
    per = project(tmp_path / "b", nights, "per_session", sessions=("n1",))
    old = project(tmp_path / "c", nights, "together", sessions=("n1",))
    assert Keys(together).chain(1) == wfl.chain(FULL)
    assert Keys(per).keys(1) == Keys(old).keys(1)


def test_a_session_job_reads_its_own_session(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session")
    p.workflow(1).params["calibrate@2"] = {"flat_scale_px": 40.0}         # this session's own setting
    k = Keys(p)
    sp = StoreProject(k, 1, "calibrate@2")
    assert sp.stage_dir("inventory") == Store(p.root).dir("inventory", k.key(1, "inventory@2"))
    assert sp.stage_dir("calibrate") == Store(p.root).tmp_dir("calibrate", k.key(1, "calibrate@2"))
    assert sp.overrides("calibrate")["flat_scale_px"] == 40.0
    assert "flat_scale_px" not in k.overrides(1, "calibrate@1")
    inv = StoreProject(k, 1, "inventory@2")
    assert [Path(f).name for f in inv.config.inputs.lights] == ["n2"]
    reg = StoreProject(k, 1, "register")
    assert reg.stage_dir("calibrate") == reg.stage_dir("inventory") == Store(p.root).dir("sessions", k.key(1, "sessions"))


def _calibrated(d: Path, cache: Path, n: int, level: float, scale: float, dark: float, white: float = 16000.0):
    d.mkdir(parents=True)
    H, W = 8, 10
    mm = np.memmap(cache, dtype=np.float16, mode="w+", shape=(n, H, W))
    mm[:] = level
    mm[0, 0, 0] = 65000.0                                              # a saturated sample
    mm.flush()
    del mm
    np.save(d / "master_dark.npy", np.full((H, W), dark, np.float32))
    np.save(d / "master_flat.npy", np.ones((H, W), np.float32))
    np.save(d / "badmask.npy", np.zeros((H, W), bool))
    np.save(d / "frame_stats.npy", np.full((n, 4, 2), level, np.float32))
    (d / "cache.json").write_text(json.dumps({"path": str(cache), "n": n, "height": H, "width": W, "scale": scale,
                                              "cfa": "RGGB", "white_level": white, "dark_level": dark}))
    (d / "manifest.json").write_text(json.dumps({"metrics": {"dark_frame_noise": 3.0, "bad_pixels_pct": 0.1}}))


class _Project:
    """What the join's job sees: the instances' folders, its own, its chain."""

    def __init__(self, root: Path, dirs: dict[str, Path], chain: list[str]):
        self.dirs, self.chain, self.root = dirs, chain, root
        self.logs = root / "logs"
        self.scratch = root / "scratch"
        self.logs.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, s):
        return self.dirs.get(s, self.root / "_absent" / s)


def test_the_join_puts_the_sessions_frames_one_after_the_other_on_one_scale(tmp_path):
    from astro_gpu.stages.sessions import Sessions, SessionsParams
    dirs = {}
    for i, (n, lvl, sc, dark) in enumerate(((3, 1000.0, 1.0, 500.0), (2, 1000.0, 0.5, 600.0)), 1):
        inv = tmp_path / f"inv{i}"
        inv.mkdir()
        (inv / "frames.json").write_text(json.dumps({"lights": [{"path": f"s{i}_{j}"} for j in range(n)],
                                                     "darks": [{"path": "d"}], "flats": [], "bias": [],
                                                     "flat_darks": []}))
        (inv / "manifest.json").write_text(json.dumps({"metrics": {
            "width": 10, "height": 8, "cfa": "RGGB", "n_lights": n, "n_darks": 1, "integration_h": 0.5 * i,
            "exposure_s": 120, "white_level": 16000.0, "first_time": f"2026-01-0{i}T22:00", "camera": "X"}}))
        dirs[f"inventory@{i}"] = inv
        _calibrated(tmp_path / f"cal{i}", tmp_path / f"c{i}.f16", n, lvl, sc, dark)
        dirs[f"calibrate@{i}"] = tmp_path / f"cal{i}"
    dirs["sessions"] = tmp_path / "out"
    chain = ["inventory@1", "inventory@2", "calibrate@1", "calibrate@2", "sessions", "register"]
    pr = _Project(tmp_path, dirs, chain)
    st = Sessions()
    ctx = Context(pr, st, Journal(pr.logs), "cpu")
    st.run(ctx, SessionsParams())
    out = tmp_path / "out"
    info = json.loads((out / "cache.json").read_text())
    assert info["n"] == 5 and info["scale"] == 0.5 and [s["n"] for s in info["sessions"]] == [3, 2]
    mm = np.memmap(info["path"], dtype=np.float16, mode="r", shape=(5, 8, 10))
    assert float(mm[0, 1, 1]) == 500.0 and float(mm[3, 1, 1]) == 1000.0   # the first session brought to 0.5
    assert mm[0, 0, 0] == np.float16(65000.0)                             # a saturated mark stays a mark
    st_ = np.load(out / "frame_stats.npy")
    assert st_.shape == (5, 4, 2) and st_[0, 0, 0] == 500.0
    fr = json.loads((out / "frames.json").read_text())
    assert [f["path"] for f in fr["lights"]] == ["s1_0", "s1_1", "s1_2", "s2_0", "s2_1"]
    # the stack's saturation level, (white - dark) / flat x scale: the lowest of the sessions'
    sat = (info["white_level"] - np.load(out / "master_dark.npy")) / np.load(out / "master_flat.npy") * info["scale"]
    assert np.allclose(sat, min((16000 - 500) * 1.0, (16000 - 600) * 0.5))
    m = ctx.metrics
    assert m["n_lights"] == 5 and m["integration_h"] == 1.5 and m["sessions"] == 2 and m["cache_scale"] == 0.5
    assert m["first_time"] == "2026-01-01T22:00"


def test_sessions_of_different_sensors_are_refused():
    from astro_gpu.stages.sessions import joined_inventory
    a = {"width": 10, "height": 8, "cfa": "RGGB"}
    with pytest.raises(StageError, match="same sensor"):
        joined_inventory([a, {**a, "cfa": "BGGR"}], [{}, {}])


# ---------------------------------------------------------------------------------------------- cut blocks

def test_cut_blocks_leave_or_pass_through_and_grey_what_needs_them(tmp_path, nights):
    base_chain = wfl.chain(FULL)
    o = wfl.set_cut(FULL, "astrometry", True)
    assert o["cut"] == ["astrometry"] and wfl.cuts(FULL) == [] and "cut" not in wfl.check_options(FULL)
    ch = wfl.chain(o)
    assert not {"astrometry", "catalogs", "surveys", "color"} & set(ch) and "gradient" in ch
    assert wfl.greyed(o) == {"color": wfl.NEEDS_ASTROMETRY}
    assert wfl.chain(wfl.set_cut(FULL, "gradient", True)) == [s for s in base_chain if s != "gradient"]
    assert "color" not in wfl.chain(wfl.set_cut(FULL, "color", True))
    assert wfl.greyed(wfl.set_cut(o, "color", True)) == {}                 # cut by the user, not greyed
    # Calibration and Registration stay, in their pass-through mode; only their keys (and what follows) change
    p = project(tmp_path / "p", nights, "together")
    k0 = Keys(p).keys(1)
    p.workflow(1).options = wfl.set_cut(FULL, "calibration", True)
    k1 = Keys(p)
    assert k1.chain(1) == base_chain and k1.key_params(1, "calibrate") == {"apply_masters": False}
    assert k1.key(1, "inventory") == k0["inventory"] and k1.key(1, "calibrate") != k0["calibrate"]
    p.workflow(1).options = wfl.set_cut(FULL, "registration", True)
    k2 = Keys(p)
    assert k2.key_params(1, "register")["align"] is False and k2.key(1, "calibrate") == k0["calibrate"]
    assert wfl.set_cut(wfl.set_cut(FULL, "color", True), "color", False) == wfl.check_options(FULL)
    with pytest.raises(ValueError):
        wfl.set_cut(FULL, "stacking", True)                                 # Stacking cannot be cut


# ---------------------------------------------------------------------------------------------- image source

def test_a_workflow_from_an_image_reads_it_as_the_stack(tmp_path, nights):
    img = tmp_path / "m42.fits"
    from astropy.io import fits
    fits.PrimaryHDU(np.random.default_rng(0).random((3, 20, 30)).astype(np.float32)).writeto(img)
    p = project(tmp_path / "p", nights, "per_session")
    w = p.add_workflow("From an image", FULL)
    w.source = {"path": str(img), "stretched": False}
    project2.save(p)
    q = project2.load(p.root)
    assert q.workflow(w.id).source == {"path": str(img), "stretched": False}
    k = Keys(q)
    ch = k.chain(w.id)
    assert ch[0] == "image" and not {"inventory", "calibrate", "register", "stack", "denoise"} & set(ch)
    assert ch[-2:] == ["render_simple", "star_removal"]                     # Stars without Denoise (issue #55)
    assert wfl.greyed(FULL, image=True) == {"denoise": wfl.NEEDS_HALF_STACKS}
    assert k.deps(w.id, "star_removal")[:2] == ("render_simple", "image")
    assert [n["block"] for n in wfl.from_options(FULL, source="image")][-2:] == ["output", "stars"]
    assert k.deps(w.id, "astrometry") == ("image",) and k.deps(w.id, "color") == ("catalogs", "image", "gradient")
    sp = StoreProject(k, w.id, "astrometry")
    assert sp.stage_dir("stack") == Store(q.root).dir("image", k.key(w.id, "image"))
    assert not sp.stage_dir("inventory").exists()
    assert k.overrides(w.id, "image")["path"].endswith("m42.fits") and k.key_params(w.id, "image") == {}
    before = k.key(w.id, "image")
    fits.PrimaryHDU(np.zeros((3, 200, 300), np.float32)).writeto(img, overwrite=True)  # another file: another key
    assert Keys(q).key(w.id, "image") != before
    # the sessions' workflow keeps its chain
    assert Keys(q).chain(1)[0] == "inventory@1"


def test_the_image_stage_writes_a_stack(tmp_path):
    from astropy.io import fits

    from astro_gpu.stages.image import ImageParams, ImageSource
    img = tmp_path / "mono.fits"
    fits.PrimaryHDU(np.linspace(0, 1, 600, dtype=np.float32).reshape(20, 30)).writeto(img)
    pr = _Project(tmp_path, {"image": tmp_path / "out"}, ["image"])
    st = ImageSource()
    ctx = Context(pr, st, Journal(pr.logs), "cpu")
    st.run(ctx, ImageParams(path=str(img)))
    a = np.load(tmp_path / "out" / "drz_full.npy")
    assert a.shape == (3, 20, 30) and np.allclose(a[0], a[2]) and a.max() == pytest.approx(65535, rel=1e-4)
    assert json.loads((tmp_path / "out" / "drz_meta.json").read_text())["scale"] == 1.0
    assert np.load(tmp_path / "out" / "drz_weight.npy").shape == (3, 20, 30)
    with pytest.raises(StageError):
        st.run(ctx, ImageParams(path=str(tmp_path / "gone.fits")))


def test_the_definition_checks_the_new_fields(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session")
    p.sessions[0].filter = "Ha"
    project2.save(p)
    assert project2.load(p.root).sessions[0].filter == "Ha"
    for bad in (lambda q: setattr(q.sessions[0], "filter", "Z"),
                lambda q: setattr(q.workflow(1), "source", {"path": 3}),
                lambda q: setattr(q.workflow(1), "source", {"fusion": {"palette": "XYZ", "inputs": []}}),
                lambda q: q.workflow(1).options.update(cut=["output"])):
        q = project2.load(p.root)
        bad(q)
        with pytest.raises(ValueError):
            q.validate()


# ---------------------------------------------------------------------------------------------- monochrome, fusion

def test_a_monochrome_workflow_has_no_colour_and_stacks_its_filter_only(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session")
    p.sessions[0].filter, p.sessions[1].filter = "Ha", "OIII"
    p.workflow(1).sessions = [1]
    project2.save(p)
    k = Keys(project2.load(p.root))
    ch = k.chain(1)
    assert ch[:4] == ["inventory", "calibrate", "register", "stack"]          # its one session: a single chain
    assert "color" not in ch and "catalogs" not in ch and "astrometry" in ch
    assert wfl.greyed(FULL, mono=True) == {"color": wfl.MONO_NO_COLOUR}
    alone = project(tmp_path / "q", nights, "together", sessions=("n1",))
    alone.sessions[0].filter = "Ha"
    project2.save(alone)
    assert k.key(1, "calibrate") == Keys(project2.load(alone.root)).key(1, "calibrate")
    assert [Path(f).name for f in StoreProject(k, 1, "inventory").config.inputs.lights] == ["n1"]


def test_cfa_mono_is_one_channel_given_to_the_three():
    from astro_gpu.core import cfa
    assert cfa.is_supported(cfa.MONO) and cfa.period(cfa.MONO) == 2 and cfa.is_mono("GGGG")
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    sp = cfa.superpixel(a, cfa.MONO)
    assert sp.shape == (2, 2, 3) and np.allclose(sp[..., 0], sp[..., 2]) and sp[0, 0, 1] == 2.5
    img = np.zeros((3, 2, 2), np.float32)
    img[1] = 5
    assert np.all(cfa.mono_rgb(img) == 5)


class _Ctx:
    def __init__(self):
        self.metrics = {}

    def log(self, *a):
        pass

    def metric(self, k, v):
        self.metrics[k] = v


def test_darks_of_another_exposure_are_scaled():
    from astro_gpu.stages.calibrate import scaled_dark
    bias = np.full((2, 2), 100, np.float32)
    dark = np.full((2, 2), 150, np.float32)                  # 300 s: 50 ADU of thermal signal
    frames = {"lights": [{"exposure": 60.0}], "darks": [{"exposure": 300.0}]}
    c = _Ctx()
    assert np.allclose(scaled_dark(c, dark, bias, 0.0, frames), 110.0) and c.metrics["dark_scale"] == 0.2
    frames["darks"] = [{"exposure": 60.0}]
    assert np.allclose(scaled_dark(c, dark, bias, 0.0, frames), 150.0)


def _output(folder: Path, img: np.ndarray, wcs=None):
    folder.mkdir(parents=True)
    np.save(folder / "lin.npy", img.astype(np.float32))
    np.save(folder / "valid.npy", np.ones(img.shape[1:], bool))
    (folder / "prep_simple.json").write_text(json.dumps({"drizzle_scale": 1.0}))
    if wcs is not None:
        wcs.to_fits().writeto(folder / "wcs.fits")


def _wcs(crpix):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [337.4, -20.8]
    w.wcs.crpix = crpix
    w.wcs.cdelt = [-0.0002, 0.0002]
    return w


def _star(H, W, x0, y0):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / 4.0)


def test_the_fusion_aligns_its_channels_through_their_plate_solutions(tmp_path):
    from astro_gpu.stages.fusion import Fusion, FusionParams, fuse
    H, W = 60, 80
    ha = np.repeat((10 * _star(H, W, 40, 30))[None], 3, 0) + 0.01
    oiii = np.repeat((5 * _star(H, W, 43, 28))[None], 3, 0) + 0.01     # the same star, 3 px right, 2 px up
    _output(tmp_path / "ha", ha, _wcs([41.0, 31.0]))
    _output(tmp_path / "oiii", oiii, _wcs([44.0, 29.0]))
    pr = _Project(tmp_path, {"fusion": tmp_path / "out"}, ["fusion"])
    st = Fusion()
    ctx = Context(pr, st, Journal(pr.logs), "cpu")
    ins = [{"role": "Ha", "channel": "L", "dir": str(tmp_path / "ha"), "wcs": str(tmp_path / "ha" / "wcs.fits")},
           {"role": "OIII", "channel": "L", "dir": str(tmp_path / "oiii"),
            "wcs": str(tmp_path / "oiii" / "wcs.fits")}]
    st.run(ctx, FusionParams(palette="HOO", inputs=ins))
    rgb = np.load(tmp_path / "out" / "drz_full.npy")
    assert rgb.shape == (3, H, W)
    for c in range(3):                                    # every channel peaks on the first one's star
        assert np.unravel_index(np.argmax(rgb[c]), (H, W)) == (30, 40)
    assert np.allclose(rgb[1], rgb[2])                    # HOO: O III green and blue
    assert json.loads((tmp_path / "out" / "drz_meta.json").read_text())["source"] == "fusion"
    ch = {"Ha": np.ones((2, 2), np.float32), "SII": np.zeros((2, 2), np.float32),
          "OIII": np.full((2, 2), 0.5, np.float32)}
    assert np.allclose(fuse(ch, "SHO")[:, 0, 0], [0, 1, 0.5]) and np.allclose(fuse(ch, "HSO")[:, 0, 0], [1, 0, 0.5])
    one = np.ones((1, 1), np.float32)
    lrgb = fuse({"L": 2 * one, "R": one, "G": one, "B": one}, "LRGB")
    assert np.allclose(lrgb, 2.0)                         # the colour of RGB, the luminance of L
    with pytest.raises(StageError, match="SII"):
        fuse({"Ha": ch["Ha"]}, "SHO")


def test_a_fusion_workflow_reads_its_channels_outputs(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session")
    p.sessions[0].filter, p.sessions[1].filter = "Ha", "OIII"
    p.workflow(1).sessions = [1]
    o = p.add_workflow("OIII", CLASSIC)
    o.sessions = [2]
    f = p.add_workflow("HOO", CLASSIC)
    f.source = {"fusion": {"palette": "HOO", "inputs": [{"pid": 1, "role": "Ha", "channel": "L"},
                                                        {"pid": o.id, "role": "OIII", "channel": "L"}]}}
    project2.save(p)
    q = project2.load(p.root)
    k = Keys(q)
    assert k.chain(f.id) == ["fusion", "render_simple"] and k.deps(f.id, "render_simple") == ("fusion",)
    assert k.fusion_inputs(f.id) == [(1, "render_simple"), (1, "astrometry"), (o.id, "render_simple"),
                                     (o.id, "astrometry")]
    ov = k.overrides(f.id, "fusion")
    assert ov["palette"] == "HOO" and ov["inputs"][1]["dir"].endswith(k.key(o.id, "render_simple"))
    before = k.key(f.id, "fusion")
    q.workflow(o.id).params["stack"] = {"kappa": 6}                         # a channel's workflow changes
    assert Keys(q).key(f.id, "fusion") != before
    from astro_gpu.core import runner2
    stages = [j.stage for j in runner2.plan_workflows(q, [f.id]).jobs]
    assert stages[-2:] == ["fusion", "render_simple"] and stages.count("render_simple") == 3
    q.workflow(f.id).source = {"fusion": {"palette": "HOO", "inputs": [{"pid": 99, "role": "Ha"}]}}
    with pytest.raises(ValueError):                                          # a fusion of a workflow that is not there
        q.validate()


def test_an_already_stretched_image_goes_to_post_as_it_is(tmp_path, nights):
    import cv2

    from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings
    from astro_gpu.stages.image import ImageParams, ImageSource
    png = tmp_path / "done.png"
    img = np.zeros((20, 30, 3), np.uint8)
    img[5:15, 10:20] = (40, 120, 200)                                       # BGR
    cv2.imwrite(str(png), img)
    pr = _Project(tmp_path, {"image": tmp_path / "out"}, ["image"])
    st = ImageSource()
    st.run(Context(pr, st, Journal(pr.logs), "cpu"), ImageParams(path=str(png), stretched=True))
    prep = json.loads((tmp_path / "out" / "prep_simple.json").read_text())
    assert prep["stretched"] and prep["stats"]["stretched"]
    y = SimpleRenderEngine(tmp_path / "out", device="cpu").render(SimpleSettings(contrast=0, saturation=1), 1.0)
    out = y.final.numpy()
    assert np.allclose(out[:, 10, 15], np.array([200, 120, 40]) / 255, atol=0.02)   # RGB, as it was
    p = project(tmp_path / "p", nights, "per_session")
    w = p.add_workflow("Done elsewhere", FULL)
    w.source = {"path": str(png), "stretched": True}
    project2.save(p)
    k = Keys(project2.load(p.root))
    assert k.chain(w.id) == ["image", "star_removal"]                         # Stars without Denoise
    assert k.overrides(w.id, "image")["stretched"] is True
    before = k.key(w.id, "image")
    q = project2.load(p.root)
    q.workflow(w.id).source = {"path": str(png), "stretched": False}
    assert Keys(q).key(w.id, "image") != before                             # linear or stretched: two results


def test_a_linear_image_has_its_stars_taken_off_and_gives_them_to_a_fusion(tmp_path):
    """Issue #55, a real run on the CPU: an Image (a FITS file, no half-stack, no Denoise), its Output, its stars taken
    off by the network, its noise unit read on its own sky; the fusion aligns those stars through the plate solutions,
    and refuses them without one."""
    from astropy.io import fits

    from astro_gpu.stages.fusion import STARS_FILE, Fusion, FusionParams
    from astro_gpu.stages.image import ImageParams, ImageSource
    from astro_gpu.stages.render_simple import RenderSimple, RenderSimpleParams
    from astro_gpu.stages.star_removal import StarRemoval, StarRemovalParams
    H, W = 128, 128
    sigma = 30.0
    rng = np.random.default_rng(3)
    a = 1000.0 + sigma * rng.standard_normal((3, H, W)) + 30000.0 * _star(H, W, 67, 66)
    img = tmp_path / "linear.fits"
    fits.PrimaryHDU(a.astype(np.float32)).writeto(img)
    dirs = {"image": tmp_path / "image", "stack": tmp_path / "image", "render_simple": tmp_path / "out",
            "star_removal": tmp_path / "split"}
    pr = _Project(tmp_path, dirs, ["image", "render_simple", "star_removal"])
    for st, par in ((ImageSource(), ImageParams(path=str(img))), (RenderSimple(), RenderSimpleParams()),
                    (StarRemoval(), StarRemovalParams())):
        st.run(Context(pr, st, Journal(pr.logs), "cpu"), par)
    assert not (tmp_path / "image" / "drz_A.npy").exists()
    prep = json.loads((tmp_path / "split" / "prep_stars.json").read_text())
    assert prep["unit"] == pytest.approx(sigma, rel=0.25)
    lin = np.load(tmp_path / "out" / "lin.npy")
    starless, stars = np.load(tmp_path / "split" / "starless.npy"), np.load(tmp_path / "split" / "stars.npy")
    assert np.allclose(starless + stars, lin, atol=1e-2)                    # the two images add up to the Output's
    assert starless[:, 66, 67].max() < 0.2 * lin[:, 66, 67].min()           # the star is gone
    assert stars[:, 66, 67].min() > 0.8 * lin[:, 66, 67].min()

    # the fusion's channels on their own grid: the same star 3 px left and 2 px up there
    _wcs([68.0, 67.0]).to_fits().writeto(tmp_path / "image_wcs.fits")
    neb = np.repeat((500 * _star(H, W, 30, 30))[None], 3, 0) + 1.0
    _output(tmp_path / "ch_out", neb, _wcs([65.0, 65.0]))
    _split(tmp_path / "ch_split", neb, np.zeros((3, H, W)))
    ins = [{"role": r, "channel": "L", "pid": 1, "dir": str(tmp_path / "ch_out"), "stars_dir": str(tmp_path / "ch_split"),
            "wcs": str(tmp_path / "ch_out" / "wcs.fits")} for r in ("Ha", "OIII")]
    source = {"mode": "workflow", "pid": 2, "dir": str(tmp_path / "split")}
    out = tmp_path / "fused"
    pr = _Project(tmp_path, {"fusion": out}, ["fusion"])
    st = Fusion()
    st.run(Context(pr, st, Journal(pr.logs), "cpu"),
           FusionParams(palette="HOO", inputs=ins, stars={**source, "wcs": str(tmp_path / "image_wcs.fits")}))
    layer = np.load(out / STARS_FILE)
    y, x = np.unravel_index(np.argmax(layer.sum(0)), (H, W))
    assert abs(int(y) - 64) <= 1 and abs(int(x) - 64) <= 1                  # aligned on the channels' grid
    with pytest.raises(StageError, match="Astrometry"):
        st.run(Context(pr, st, Journal(pr.logs), "cpu"), FusionParams(palette="HOO", inputs=ins, stars=source))


# ---------------------------------------------------------------------------------------------- option A: stars

def _split(folder: Path, starless: np.ndarray, stars: np.ndarray):
    folder.mkdir(parents=True)
    np.save(folder / "starless.npy", starless.astype(np.float32))
    np.save(folder / "stars.npy", stars.astype(np.float32))
    np.save(folder / "valid.npy", np.ones(starless.shape[1:], bool))


def test_option_a_fuses_the_starless_channels_and_makes_a_star_layer(tmp_path):
    from astro_gpu.stages.fusion import STARS_FILE, Fusion, FusionParams, star_support
    H, W = 40, 50
    rng = np.random.default_rng(0)
    neb = np.repeat((3 * _star(H, W, 25, 20) * 20)[None], 3, 0)
    star = np.repeat((8 * _star(H, W, 10, 10))[None], 3, 0)
    noise = lambda: rng.normal(0, 0.01, (3, H, W))                       # noqa: E731
    for role, k in (("Ha", 1.0), ("OIII", 0.5)):
        _output(tmp_path / f"{role}_out", neb * k + star)
        _split(tmp_path / f"{role}_split", neb * k + 0.001, star * k + noise())
    ins = [{"role": r, "channel": "L", "pid": i, "dir": str(tmp_path / f"{r}_out"),
            "stars_dir": str(tmp_path / f"{r}_split"), "wcs": ""} for i, r in ((1, "Ha"), (2, "OIII"))]
    for mode, extra in (("palette", {}), ("channel", {"role": "Ha"})):
        out = tmp_path / f"out_{mode}"
        pr = _Project(tmp_path, {"fusion": out}, ["fusion"])
        st = Fusion()
        ctx = Context(pr, st, Journal(pr.logs), "cpu")
        ins2 = [dict(i, pid=1) for i in ins]                               # one grid: no plate solution needed
        st.run(ctx, FusionParams(palette="HOO", inputs=ins2, stars={"mode": mode, **extra}))
        starless = np.load(out / "drz_full.npy")
        stars = np.load(out / STARS_FILE)
        assert starless[:, 10, 10].max() < 0.05 * starless.max()          # no star in the starless fusion
        assert stars[:, 10, 10].min() > 0 and stars[:, 30, 45].max() == 0  # the star kept, the noise left out
        assert json.loads((out / "drz_meta.json").read_text())["starless"]
        if mode == "channel":
            assert np.allclose(stars[0], stars[1])                        # neutral stars
    s = np.zeros((20, 20), np.float32) + np.random.default_rng(1).normal(0, 1, (20, 20)).astype(np.float32)
    s[10, 10] = 50
    kept = star_support(s)
    assert kept[10, 10] > 40 and np.count_nonzero(kept) < 40


def test_a_starless_fusion_chain_ends_with_its_split_block(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session")
    p.sessions[0].filter, p.sessions[1].filter = "Ha", "OIII"
    p.workflow(1).sessions = [1]
    o = p.add_workflow("OIII", FULL)
    o.sessions = [2]
    f = p.add_workflow("HOO", FULL)
    f.source = {"fusion": {"palette": "HOO", "stars": {"mode": "channel", "role": "Ha"}, "weights": {"OIII": 0.5},
                           "inputs": [{"pid": 1, "role": "Ha", "channel": "L"},
                                      {"pid": o.id, "role": "OIII", "channel": "L"}]}}
    project2.save(p)
    q = project2.load(p.root)
    k = Keys(q)
    assert k.chain(f.id) == ["fusion", "render_simple", "star_removal"]
    assert (1, "star_removal") in k.fusion_inputs(f.id)
    ov = k.overrides(f.id, "fusion")
    assert ov["stars"] == {"mode": "channel", "role": "Ha"} and ov["weights"] == {"OIII": 0.5}
    assert ov["inputs"][0]["stars_dir"].endswith(k.key(1, "star_removal"))
    before = k.key(f.id, "fusion")
    q.workflow(f.id).source["fusion"]["weights"] = {"OIII": 0.6}
    assert Keys(q).key(f.id, "fusion") != before
    q.workflow(f.id).source["fusion"]["stars"] = {"mode": "workflow"}                   # no pid
    with pytest.raises(ValueError):
        q.validate()
