"""A Calibration placed in parallel: its own darks, flats, bias and dark flats, in the engine. Its files are part of what
its result depends on; a series that does not change computes once with the Session's. The frames are empty files (the
keys read their names and sizes only), then a real run on the CPU over small FITS frames."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from astro_gpu import api
from astro_gpu.core import project2, runner2
from astro_gpu.core import workflow as wfl
from astro_gpu.core.project2 import Session
from astro_gpu.core.store import Keys, StoreProject


def frames(folder: Path, names: dict[str, int]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for n, size in names.items():
        (folder / n).write_bytes(b"\0" * size)
    return folder


@pytest.fixture
def nights(tmp_path):
    raw = tmp_path / "raw"
    return {"n1": frames(raw / "n1", {"a1.ARW": 100, "a2.ARW": 110}), "n2": frames(raw / "n2", {"b1.ARW": 120}),
            "darks": frames(raw / "darks", {"d1.ARW": 90}), "flats": frames(raw / "flats", {"f1.ARW": 80}),
            "other": frames(raw / "other", {"d2.ARW": 95, "d3.ARW": 96})}


GRAPH = [wfl.node("calibration"), wfl.node("registration"), wfl.node("stacking")]


def project(root: Path, nights: dict, mode: str, sessions=("n1",)) -> project2.Project2:
    p = project2.create(root, "p")
    p.stack_mode = mode
    for i, n in enumerate(sessions, 1):
        p.sessions.append(Session(i, n, lights=[str(nights[n])], darks=[str(nights["darks"])],
                                  flats=[str(nights["flats"])]))
    p.add_workflow("Main", wfl.DEFAULT_OPTIONS)
    p.set_blocks(1, GRAPH)
    project2.save(p)
    return p


def branch(p, own: dict) -> int:
    w = p.add_workflow("Branch", wfl.DEFAULT_OPTIONS)
    p.set_blocks(w.id, GRAPH)
    w.calibration = own
    p.validate()
    return w.id


def test_a_calibration_that_changed_nothing_computes_once_with_the_session_s(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session")
    b = branch(p, {"1": {"fork": 2}})
    k = Keys(p)
    assert k.keys(b) == k.keys(1)
    assert b in k.sharing(1, "calibrate")


def test_its_own_files_change_its_result_and_what_follows_only_for_it(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session")
    b = branch(p, {"1": {"fork": 2, "darks": [str(nights["other"] / "d2.ARW"), str(nights["other"] / "d3.ARW")]}})
    k = Keys(p)
    for st in ("inventory", "calibrate", "register", "stack"):
        assert k.key(b, st) != k.key(1, st), st
    two = branch(p, copy.deepcopy(p.workflow(b).calibration))
    assert Keys(p).keys(two) == Keys(p).keys(b)      # two identical series: one result


def test_its_job_reads_its_own_files_and_the_session_s_for_the_other_kinds(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session")
    d2 = str(nights["other"] / "d2.ARW")
    b = branch(p, {"1": {"fork": 2, "darks": [d2]}})
    inp = StoreProject(Keys(p), b, "inventory").config.inputs
    assert [Path(f).name for f in inp.darks] == ["d2.ARW"]
    assert [Path(f).name for f in inp.flats] == ["flats"]            # follows the Session
    assert [Path(f).name for f in inp.lights] == ["n1"]
    assert [Path(f).name for f in StoreProject(Keys(p), 1, "inventory").config.inputs.darks] == ["darks"]


def test_on_several_nights_it_replaces_only_the_night_it_is_placed_on(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session", ("n1", "n2"))
    b = branch(p, {"2": {"fork": 2, "flats": [str(nights["other"] / "d2.ARW")]}})
    k = Keys(p)
    assert k.key(b, "calibrate@1") == k.key(1, "calibrate@1")
    assert k.key(b, "calibrate@2") != k.key(1, "calibrate@2")
    assert k.key(b, "sessions") != k.key(1, "sessions")
    assert [Path(f).name for f in StoreProject(k, b, "inventory@2").config.inputs.flats] == ["d2.ARW"]
    assert [Path(f).name for f in StoreProject(k, b, "inventory@1").config.inputs.flats] == ["flats"]


def test_it_is_saved_and_an_older_project_has_none(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session")
    b = branch(p, {"1": {"fork": 2, "bias": []}})
    project2.save(p)
    q = project2.load(p.root)
    assert q.workflow(b).calibration == {"1": {"fork": 2, "bias": []}}
    d = p.to_dict()
    for w in d["workflows"]:
        w.pop("calibration")
    assert project2.Project2.from_dict(d, p.root).workflow(1).calibration is None


@pytest.mark.parametrize("own", [{}, {"1": {}}, {"7": {"fork": 1}}, {"1": {"fork": 1, "lights": []}},
                                 {"1": {"fork": 1, "darks": [""]}}, {"1": {"fork": "a"}}])
def test_a_definition_that_makes_no_sense_is_refused(tmp_path, nights, own):
    p = project(tmp_path / "p", nights, "per_session")
    with pytest.raises(ValueError):
        branch(p, own)


def test_a_mark_is_kept_and_read_by_no_key(tmp_path, nights):
    p = project(tmp_path / "p", nights, "per_session")
    w = p.add_workflow("Branch", wfl.DEFAULT_OPTIONS)
    p.set_blocks(w.id, [wfl.node("calibration"), wfl.node("registration"), wfl.node("stacking", fork=2)])
    assert p.workflow(w.id).blocks[2]["fork"] == 2
    assert "fork" not in p.workflow(1).blocks[2]         # none on a block of before
    assert Keys(p).keys(w.id) == Keys(p).keys(1)
    project2.save(p)
    assert project2.load(p.root).workflow(w.id).blocks[2]["fork"] == 2


def test_a_real_run_gives_each_series_its_master(tmp_path):
    """On the CPU, small FITS frames: two series of darks at different levels, one master each."""
    from astropy.io import fits

    from astro_gpu.core.store import Store

    def write(path: Path, level: float, seed: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        a = np.clip(rng.normal(level, 4, (24, 32)), 0, 65535).astype(np.uint16)
        h = fits.Header()
        for k, v in {"BAYERPAT": "RGGB", "EXPTIME": 15.0, "GAIN": 100.0, "DATE-OBS": "2026-08-16T22:00:00",
                     "INSTRUME": "TESTCAM"}.items():
            h[k] = v
        fits.PrimaryHDU(a, h).writeto(path, overwrite=True)
        return path
    raw = tmp_path / "raw"
    for i in range(3):
        write(raw / "L" / f"L{i}.fits", 400, i)
        write(raw / "A" / f"D{i}.fits", 50, 10 + i)
        write(raw / "B" / f"E{i}.fits", 90, 20 + i)
    p = project2.create(tmp_path / "p", "p")
    p.stack_mode = "per_session"
    p.devices = {"main": "cpu", "aux": ""}
    p.sessions.append(Session(1, "n", lights=[str(raw / "L")], darks=[str(raw / "A")]))
    p.add_workflow("Main", wfl.DEFAULT_OPTIONS)
    p.set_blocks(1, [wfl.node("calibration")])
    w = p.add_workflow("Other", wfl.DEFAULT_OPTIONS)
    p.set_blocks(w.id, [wfl.node("calibration")])
    w.calibration = {"1": {"fork": 2, "darks": [str(f) for f in sorted((raw / "B").glob("*.fits"))]}}
    alike = p.add_workflow("Alike", wfl.DEFAULT_OPTIONS)
    p.set_blocks(alike.id, [wfl.node("calibration")])
    alike.calibration = {"1": {"fork": 3}}
    for x in p.workflows:
        x.params["calibrate"] = {"workers": 1}
    project2.save(p)
    plan = api.plan_workflows(p, [1, w.id, alike.id], device="cpu")
    jobs = [j for j in plan.jobs if j.stage == "calibrate"]
    assert len(jobs) == 2                                    # the series that follows its Session is computed once
    k = Keys(project2.load(p.root))
    assert k.key(alike.id, "calibrate") == k.key(1, "calibrate") != k.key(w.id, "calibrate")
    assert runner2.run_plan(p.root, runner2.prepare_run(p, plan, "cpu").plan_file, echo=False) == "done"
    s = Store(p.root)
    a = np.load(s.dir("calibrate", k.key(1, "calibrate")) / "master_dark.npy")
    b = np.load(s.dir("calibrate", k.key(w.id, "calibrate")) / "master_dark.npy")
    assert 30 < float(np.median(b - a)) < 50
