"""The file-type warning before a run: master FITS files made by another program beside raw lights are the normal case
(their scale is read as the lights', io/frames.fits_adu), not a mistake; and a same warning is said once, with the
sessions it concerns (a project of three nights said "the darks are FITS files, the lights ARW" nine times)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from astro_gpu import api
from astro_gpu.core import project2
from astro_gpu.core.project import Target, type_warnings
from astro_gpu.io.frames import master_kinds


def _fits(path: Path, **hdr) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.Header()
    for k, v in hdr.items():
        h[k] = v
    fits.PrimaryHDU(np.zeros((4, 6), np.float32), h).writeto(path, overwrite=True)
    return str(path)


def test_a_master_beside_raw_lights_is_not_a_file_of_the_wrong_type(tmp_path):
    dark = _fits(tmp_path / "dark.fit", STACKCNT=16)
    flat = _fits(tmp_path / "flat.fit", IMAGETYP="Master Flat")
    single = _fits(tmp_path / "d1.fits")
    files = {"lights": ["a.ARW", "b.ARW"], "darks": [dark], "flats": [flat], "bias": [single]}
    assert master_kinds(files) == {"darks", "flats"}
    assert [(w["key"], w["kind"]) for w in type_warnings(files)] == [("other_type", "darks"), ("other_type", "flats"),
                                                                     ("other_type", "bias")]
    assert [(w["key"], w["kind"]) for w in type_warnings(files, master_kinds(files))] == [("other_type", "bias")]
    assert master_kinds({"lights": ["a.ARW"], "darks": [dark, single]}) == set()        # a single frame among them


def _project(root: Path, masters: bool) -> project2.Project2:
    p = project2.create(root / "P", "P", Target(name="NGC 7000"))
    dark = _fits(root / "cal" / "dark.fit", **({"STACKCNT": 16} if masters else {}))
    bias = _fits(root / "cal" / "bias.fit", **({"STACKCNT": 50} if masters else {}))
    for i in range(3):
        lights = root / f"L{i}"
        lights.mkdir(parents=True)
        for j in range(2):
            (lights / f"DSC{j}.ARW").write_bytes(b"\0" * 64)
        flat = _fits(root / f"F{i}" / "flat.fit", **({"STACKCNT": 30} if masters else {}))
        p.sessions.append(project2.Session(i + 1, f"night {i + 1}", lights=[str(lights)], darks=[dark], flats=[flat],
                                           bias=[bias]))
    project2.save(p)
    return p


def test_the_checks_say_a_same_warning_once_with_its_sessions_and_nothing_for_masters(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_disk_check", lambda p: api.Check("ok", "disk", ""))      # fake raw files: no size
    checks = [c for c in api.health_checks(_project(tmp_path / "a", masters=False)) if c.key == "file_types"]
    assert [c.text for c in checks] == [f"night 1, night 2, night 3: the {k} are FITS, the lights ARW"
                                        for k in ("darks", "flats", "bias")]
    assert checks[0].args["sessions"] == ["night 1", "night 2", "night 3"]
    assert not [c for c in api.health_checks(_project(tmp_path / "b", masters=True)) if c.key == "file_types"]


def test_the_run_window_says_a_same_warning_once_with_its_sessions():
    from astro_gpu.client.runs import RunDialog
    warn = [{"key": "other_type", "kind": "darks", "type": "FITS", "main": "ARW", "n": 1}]
    ses = {1: "night 1", 2: "night 2", 3: "night 3"}
    s = SimpleNamespace(file_warnings=lambda sid: warn if sid != 3 else [], colour_note=lambda sid: None,
                        view=lambda w: SimpleNamespace(source=None))
    s.p = SimpleNamespace(sessions=[SimpleNamespace(id=i) for i in ses], session=lambda sid: SimpleNamespace(name=ses[sid]),
                          sessions_of=lambda w: s.p.sessions)
    me = SimpleNamespace(plan=SimpleNamespace(jobs=[SimpleNamespace(wf=[1])]), s=s)
    assert RunDialog._file_warnings(me) == ["night 1, night 2: The darks are FITS files, the lights ARW"]
