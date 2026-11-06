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
    assert ch[0] == "image" and not {"inventory", "calibrate", "register", "stack", "denoise", "star_removal"} & set(ch)
    assert wfl.greyed(FULL, image=True) == {"denoise": wfl.NEEDS_HALF_STACKS, "stars": wfl.NEEDS_DENOISE}
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
                lambda q: setattr(q.workflow(1), "source", {"path": ""}),
                lambda q: setattr(q.workflow(1), "source", {"path": "x.fits", "stretched": True}),
                lambda q: q.workflow(1).options.update(cut=["output"])):
        q = project2.load(p.root)
        bad(q)
        with pytest.raises(ValueError):
            q.validate()
