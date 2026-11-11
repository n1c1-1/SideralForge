"""Calibration masters made by another program (a master bias, dark and flat, one file each, FITS in floats on the 0-1
scale): recognised from their headers and used as they are, never calibrated nor stacked again; what their data
contradict is said in the log."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from astro_gpu.core.project import Inputs, Project
from astro_gpu.core.runner import Pipeline
from astro_gpu.io.frames import master_count, read_info
from astro_gpu.stages.calibrate import Calibrate
from astro_gpu.stages.inventory import Inventory

H, W = 40, 48
BLACK = 512.0


def _write(path: Path, data, history=(), **hdr) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.Header()
    for k, v in {"BAYERPAT": "RGGB", **hdr}.items():
        h[k] = v
    for line in history:
        h.add_history(line)
    fits.PrimaryHDU(data=data, header=h).writeto(path, overwrite=True)
    return path


def _vignetting() -> np.ndarray:
    """1 at the centre, 0.3 in the corners."""
    y, x = np.mgrid[:H, :W]
    return (1 - 0.7 * ((x - W / 2) ** 2 + (y - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)).astype(np.float32)


def test_a_master_is_known_by_its_frame_count_its_image_type_or_the_history_of_its_stacking(tmp_path):
    z = np.zeros((4, 4), np.float32)
    assert master_count(fits.Header({"STACKCNT": 30})) == 30
    assert master_count(fits.Header({"NCOMBINE": 12})) == 12
    assert master_count(fits.Header({"IMAGETYP": "Master Flat"})) == 0          # a master, its count not said
    h = fits.Header()
    h.add_history("mean stacking with winsorized sigma clipping rejection (low=3.000 high=3")
    assert master_count(h) == 0
    for single in ({}, {"IMAGETYP": "Dark Frame"}, {"STACKCNT": 1}, {"DARK_CNT": 20, "CALSTAT": "BD"}):
        assert master_count(fits.Header(single)) is None                       # a calibrated light counts its darks too
    assert read_info(_write(tmp_path / "f.fit", z, STACKCNT=30)).stacked == 30
    assert read_info(_write(tmp_path / "g.fit", z)).stacked is None


def _evenness(fr: np.ndarray) -> float:
    """How far the calibrated sky of the corners is from the centre's (0: the vignetting taken out exactly)."""
    return abs(float(np.nanmedian(fr[3:7, 3:7]) / np.nanmedian(fr[16:24, 20:28])) - 1)


def _session(tmp_path: Path, flat, flat_hdr: dict, dark_level: float = BLACK, bias: bool = True,
             single_darks: int = 0) -> tuple[dict, dict, list[str], np.ndarray]:
    """Lights of a black level of 512 ADU, a sky of 1000 ADU under the vignetting and a hot pixel; a master dark and a
    master bias in floats on the 0-1 scale; the flat given. (manifest, cache.json, log lines, first calibrated frame)."""
    L, D, F, B = (tmp_path / k for k in "LDFB")
    v = _vignetting()
    for i in range(3):
        a = BLACK + 1000 * v + np.random.default_rng(i).normal(0, 2, (H, W))
        a[7, 9] += 3000
        _write(L / f"l{i}.fits", np.round(a).astype(np.uint16), EXPTIME=300.0, **{"DATE-OBS": f"2026-08-16T22:0{i}:00"})
    dark = np.full((H, W), dark_level, np.float32)
    dark[7, 9] += 3000
    _write(D / "dark.fit", dark / np.float32(65535), EXPTIME=300.0, STACKCNT=16,
           history=["mean stacking with winsorized sigma clipping rejection"])
    for i in range(single_darks):
        _write(D / f"d{i}.fits", np.round(dark).astype(np.uint16), EXPTIME=300.0)
    if bias:
        _write(B / "bias.fit", np.full((H, W), BLACK / 65535, np.float32), STACKCNT=50)
    _write(F / "flat.fit", flat, EXPTIME=0.05, **flat_hdr)
    pr = Project.create(tmp_path / "p", "p", Inputs(lights=[str(L)], darks=[str(D)], flats=[str(F)],
                                                    bias=[str(B)] if bias else []))
    pr.config.devices = {"main": "cpu"}
    pr.config.params = {"calibrate": {"workers": 1, "hot_from_lights": False}}
    pr.save()
    Pipeline(pr, [Inventory(), Calibrate()]).run()
    d = Path(pr.stage_dir("calibrate"))
    m = json.loads((d / "manifest.json").read_text())
    assert m["status"] == "done", m.get("error")
    cache = json.loads((d / "cache.json").read_text())
    fr = np.asarray(np.memmap(cache["path"], dtype=np.float16, mode="r", shape=(3, H, W))[0], np.float32) / cache["scale"]
    logs = [json.loads(x) for x in (pr.logs / "events.jsonl").read_text().splitlines()]
    return m, cache, [(e.get("level"), e["msg"]) for e in logs if e.get("type") == "log" and e.get("stage") == "calibrate"], fr


def test_a_master_flat_already_calibrated_is_used_as_it_is(tmp_path):
    """The master flat of another program holds no bias (its flats were calibrated before they were stacked): the bias
    taken off it again bent the vignetting (here the corners came out 20 % above the centre)."""
    flat = 3000 * _vignetting() / np.float32(65535)
    m, _cache, logs, fr = _session(tmp_path, flat.astype(np.float32), {"STACKCNT": 30})
    assert _evenness(fr) < 0.05
    said = [t for _lv, t in logs]
    assert "the flats are a master of 30 frames, already calibrated: used as it is" in said
    assert "the darks are a master of 16 frames: used as it is" in said
    assert "the bias are a master of 50 frames: used as it is" in said
    assert not [t for lv, t in logs if lv == "warning" and ("flats" in t or "dark" in t)]
    assert "dark_frame_noise" not in m["metrics"]          # the noise of one dark is not known from a master


def test_single_frames_beside_a_master_are_left_out_and_said(tmp_path):
    flat = 3000 * _vignetting() / np.float32(65535)
    _m, _c, logs, fr = _session(tmp_path, flat.astype(np.float32), {"NCOMBINE": 25}, single_darks=2)
    assert ("warning", "2 single darks given beside the master are left out (it holds its frames already)") in logs
    assert "the flats are a master of 25 frames, already calibrated: used as it is" in [t for _lv, t in logs]
    assert _evenness(fr) < 0.05


def test_what_the_data_contradict_is_said_a_dark_without_its_bias_and_single_flats_under_the_bias(tmp_path):
    """A master dark whose bias was taken off cannot take the lights' off; flats given as single frames that go under
    the bias had theirs taken off already (the calibration then takes it off again): warnings, not guesses."""
    flat = np.round(3000 * _vignetting() - 1000).clip(0).astype(np.uint16)      # one frame, its bias taken off
    _m, _c, logs, _fr = _session(tmp_path, flat, {}, dark_level=4.0)
    warns = [t for lv, t in logs if lv == "warning"]
    assert any(t.startswith("the master dark's level (4 ADU) is far under the bias (512 ADU): its bias was taken off")
               for t in warns), warns
    assert any(t.startswith("the flats go under the bias they are corrected with") for t in warns), warns
