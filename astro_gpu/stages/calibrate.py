"""Calibration of the raw Bayer frames (no debayering): master dark / bias / flat, bad pixel map, calibrated cache."""
from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from ..core import cfa as cfa_mod
from ..core import devices
from ..core.imgutil import robust_sigma
from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage
from ..io.frames import read_cfa, read_colour

SAT_MARK = 65000.0      # value written for saturated pixels in the cache
VALID_MAX = 60000.0     # anything above is treated as saturated downstream
MIN_USABLE = 0.5        # the share of a frame's samples below which it has no statistics (issue #64)


class CalibrateParams(StageParams):
    hot_sigma: float = P(0.6, label="Hot pixel threshold", level=ADVANCED, ge=0.1, le=10, step=0.1, unit="x sky noise",
                         help="A pixel whose dark current exceeds this fraction of the sky noise of a light frame is "
                              "excluded from the stack.",
                         tip="Low values (0.5-1) exclude more pixels; with many frames that costs nothing because "
                             "the drizzle fills the gaps. Raise it if more than ~5 % of the pixels get excluded.")
    flat_defect: float = P(0.3, label="Flat defect threshold", level=ADVANCED, ge=0.05, le=0.9, step=0.05,
                           help="Pixels whose flat response differs from their neighbourhood by more than this fraction "
                                "(dust specks, dead pixels) are excluded.",
                           tip="0.3 = 30 %. Lower it if dust donuts remain visible after calibration.")
    hot_from_lights: bool = P(True, label="Hot pixels (lights)", level=ADVANCED,
                              help="Also flag the pixels that stand out from their neighbours in many light frames "
                                   "taken at different sky positions (the sky drifts across the sensor, a defect "
                                   "does not). Darks taken at another temperature miss pixels that are hot only "
                                   "when the sensor is warmer; in the stack they draw the drift path of the mount.",
                              tip="Skipped automatically when the sky did not move on the sensor during the night.")
    flat_scale_px: float = P(0.0, label="Flat scale", level=ADVANCED, ge=0, le=500, step=5, unit="px",
                             help="Keep only the scales of the master flat larger than this (Gaussian sigma, sensor "
                                  "pixels). 0 = the whole flat. For flat frames borrowed from another night of the "
                                  "same optics: they hold the vignetting of the instrument, but not this night's dust.",
                             tip="About 40 px removes dust donuts and keeps the vignetting.")
    saturation: float = P(0.976, label="Saturation level", level=ADVANCED, ge=0.5, le=1.0, step=0.001, unit="x white level",
                          help="Raw values above this fraction of the sensor white level are flagged as saturated; the "
                               "stacker then treats them specially instead of averaging clipped values.",
                          tip="Keep the default unless star cores look flat-topped or rings appear around bright stars.")
    apply_masters: bool = P(True, label="Apply calibration", level=ADVANCED,
                            help="Off when the Calibration block is cut (issue #43): the frames are taken as they are "
                                 "(already calibrated), without master dark, flat or defective pixels.")
    workers: int = P(8, label="Parallel workers", level=ADVANCED, ge=1, le=64,
                     help="Number of CPU processes used to calibrate frames in parallel.",
                     tip="Defaults to the number of CPU cores minus two (max 12).")


class Calibrate(Stage):
    name = "calibrate"
    title = "Calibration"
    description = ("Builds the master dark, bias and flat, finds hot, cold and dusty pixels, then subtracts the dark "
                   "and divides by the flat for every light frame. Frames stay as raw Bayer mosaics (no debayering): "
                   "colour is reconstructed later by the drizzle, without interpolation. The calibrated frames are "
                   "cached on the fast scratch disk; an interrupted calibration resumes frame by frame.")
    deps = ("inventory",)
    Params = CalibrateParams
    # 2: hot pixels also seen in the lights; 3: a bad pixel stays excluded when saturated; 4: hot pixels seen in the
    # lights when the sky did not move too (a lone pixel above its neighbours of every colour: m8 kept 45-57 coloured
    # dots per square degree), and then filled from their same-colour neighbours (left out, they were holes). The
    # master bias and flat dark (EXTRA_MASTERS, for the Inspect mode) are extra files: an older result lacks them
    version = "4"

    def auto(self, ctx: Context) -> dict[str, Auto]:
        n = max(1, min(12, (os.cpu_count() or 4) - 2))
        return {"workers": Auto(n, "{cores} CPU cores detected", {"cores": os.cpu_count()})}

    def run(self, ctx: Context, p: CalibrateParams) -> None:
        frames = ctx.load_json("inventory", "frames.json")
        if cfa_mod.is_colour(ctx.metric_of("inventory", "cfa")):       # frames already in colour (issue #57)
            (ctx.dir / MASTERS).unlink(missing_ok=True)
            calibrate_colour(ctx, p, frames)
            return
        given = given_masters(ctx, p)
        if given is not None:                # the masters of an earlier result: rebuild the calibrated cache only
            rebuild_cache(ctx, p, frames["lights"], given, ctx.metric_of("inventory", "cfa"),
                          float(ctx.metric_of("inventory", "white_level")))
            return
        H, W = int(ctx.metric_of("inventory", "height")), int(ctx.metric_of("inventory", "width"))
        black = float(ctx.metric_of("inventory", "black_level"))
        if not p.apply_masters:
            pass_through(ctx, p, frames["lights"], H, W, black)
            return
        # issue #50: the masters' phase moves the bar too, file by file (its share: its files against the lights)
        n_hot = min(40, len(frames["lights"])) + 2 if p.hot_from_lights and len(frames["lights"]) >= 15 else 0
        nm = sum(len(frames[k]) for k in ("bias", "darks", "flats", "flat_darks")) + (1 if frames["darks"] else 0) \
            + n_hot
        share = nm / max(nm + len(frames["lights"]), 1)
        ctx._files_total, ctx._files_read, ctx._lights_from = nm, 0, share
        ctx.phase(0.0, share)
        bias = master_bias([f["path"] for f in frames["bias"]], ctx)
        if bias is not None:
            ctx.save("master_bias.npy", bias)
        dark, dark_noise = master_dark([f["path"] for f in frames["darks"]], bias, H, W, black, ctx)
        dark = scaled_dark(ctx, dark, bias, black, frames)
        kept: dict = {}
        flat = master_flat([f["path"] for f in frames["flats"]], [f["path"] for f in frames["flat_darks"]], bias,
                           float(np.median(dark)), H, W, p.flat_scale_px, ctx, keep=kept,
                           cfa=ctx.metric_of("inventory", "cfa") or "RGGB")
        if "flat_dark" in kept:
            ctx.save("master_flat_dark.npy", kept.pop("flat_dark"))
        calibrate_lights(ctx, p, frames["lights"], dark, flat, bool(frames["darks"]), dark_noise,
                         ctx.metric_of("inventory", "cfa"), float(ctx.metric_of("inventory", "white_level")))


def pass_through(ctx: Context, p: CalibrateParams, lights: list[dict], H: int, W: int, black: float) -> None:
    """The Calibration block cut (issue #43): the frames as they are (already calibrated), the black level taken off,
    no flat, no defective pixel; the calibrated cache the later stages read is written all the same."""
    ctx.log("Calibration cut: the frames are taken as they are (no master dark, flat or defective pixels)")
    dark = np.full((H, W), black, np.float32)
    flat = np.ones((H, W), np.float32)
    ctx.save("master_dark.npy", dark)
    ctx.save("master_flat.npy", flat)
    bad = np.zeros((H, W), bool)
    ctx.save("badmask.npy", bad)
    ctx.metric("dark_level", round(black, 1))
    ctx.metric("bad_pixels_pct", 0.0)
    ctx.metric("calibration_cut", True)
    _calibrated_cache(ctx, p, lights, dark, flat, bad, ctx.metric_of("inventory", "cfa"),
                      float(ctx.metric_of("inventory", "white_level")), time.time())


def master_bias(paths: list[str], ctx: Context) -> np.ndarray | None:
    """Master bias (offset): median of the bias frames, or None without any."""
    return _median_stack(paths, ctx, "bias") if paths else None


def master_dark(paths: list[str], bias: np.ndarray | None, H: int, W: int, black: float,
                ctx: Context) -> tuple[np.ndarray, float | None]:
    """Master dark (median of the darks) and the noise of one dark frame; without darks, the master bias or the black
    level stands for it."""
    if paths:
        dark = _median_stack(paths, ctx, "darks")
        one = read_cfa(paths[0])
        _tick(ctx, "the noise of one dark")
        return dark, robust_sigma(one[::3, ::3])[1]
    dark = bias.copy() if bias is not None else np.full((H, W), black, np.float32)
    ctx.log("no dark frames: using " + ("the master bias" if bias is not None else "the black level") +
            " as dark (hot pixels will not be removed by calibration)", "warning")
    return dark, None


def _exposure(frames: list[dict]) -> float | None:
    t = [float(f["exposure"]) for f in frames if f.get("exposure")]
    return float(np.median(t)) if t else None


def scaled_dark(ctx: Context, dark: np.ndarray, bias: np.ndarray | None, black: float, frames: dict) -> np.ndarray:
    """The master dark brought to the lights' exposure when the darks were taken at another one: its thermal signal
    (the dark minus the bias, or minus the black level without bias frames) scaled by the ratio of the exposures, the
    bias put back (dark current grows in proportion to the exposure; the classic dark scaling of the reduction
    guides, e.g. the astropy CCD reduction guide, "scaling dark frames"). Darks of the lights' exposure: as they are."""
    t_l, t_d = _exposure(frames.get("lights") or []), _exposure(frames.get("darks") or [])
    if not frames.get("darks") or not t_l or not t_d or abs(t_d - t_l) <= 0.05 * t_l:
        return dark
    base = bias if bias is not None else np.float32(black)
    r = t_l / t_d
    ctx.log(f"darks of {t_d:g} s for lights of {t_l:g} s: their thermal signal scaled by {r:.3f}"
            + ("" if bias is not None else " (no bias frames: above the black level)"))
    ctx.metric("dark_scale", round(r, 4))
    return (base + (dark - base) * np.float32(r)).astype(np.float32)


def master_flat(paths: list[str], flat_dark_paths: list[str], bias: np.ndarray | None, dark_level: float, H: int,
                W: int, flat_scale_px: float, ctx: Context, keep: dict | None = None, cfa: str = "RGGB") -> np.ndarray:
    """Master flat normalised to 1 at the centre of each position of the colour pattern (``cfa``: its 4 Bayer
    positions, its 36 X-Trans ones), all ones without flats. ``keep``: receives the master flat dark under
    "flat_dark" when there are flat darks."""
    P = cfa_mod.period(cfa)
    if paths:
        if flat_dark_paths:
            fdark = _median_stack(flat_dark_paths, ctx, "flat darks")
            if keep is not None:
                keep["flat_dark"] = fdark
        elif bias is not None:
            fdark = bias
        else:
            fdark = np.full((H, W), dark_level, np.float32)   # flats are short: dark current negligible
        flat = _master_flat(paths, fdark, ctx)
        if flat_scale_px > 0:
            # borrowed flat: vignetting only (each colour plane smoothed; its dust is not this night's dust)
            for k in range(P * P):
                v = _plane(flat, k, P)
                v[...] = cv2.GaussianBlur(np.ascontiguousarray(v), (0, 0), flat_scale_px / P)
            ctx.log(f"master flat kept above {flat_scale_px:.0f} px only (vignetting, not dust)")
    else:
        flat = np.ones((H, W), np.float32)
        ctx.log("no flat frames: vignetting and dust will not be corrected (the sky model will absorb part of "
                "the vignetting)", "warning")
    r = 200 // P                                        # the centre's 400 x 400 sensor pixels, in plane pixels
    for k in range(P * P):
        v = _plane(flat, k, P)
        c = v[v.shape[0] // 2 - r:v.shape[0] // 2 + r, v.shape[1] // 2 - r:v.shape[1] // 2 + r]
        v /= np.median(c)
    return flat


def calibrate_lights(ctx: Context, p: CalibrateParams, lights: list[dict], dark: np.ndarray, flat: np.ndarray,
                     have_darks: bool, dark_noise: float | None, cfa: str, white: float) -> None:
    """Bad pixel map and calibrated cache of the light frames from the given masters (saved with them in the
    stage's folder: master_dark, master_flat, badmask, frame_stats, cache.json)."""
    H, W = dark.shape
    t0 = time.time()
    level = float(np.median(dark))
    ctx.save("master_dark.npy", dark)
    ctx.metric("dark_level", round(level, 1))
    if dark_noise:
        ctx.metric("dark_frame_noise", round(dark_noise, 2))
    ctx.save("master_flat.npy", flat)
    ctx.metric("vignetting_min", round(float(flat.min()), 3))

    # ---------------- bad pixels: hot (dark), cold / dust (flat)
    P = cfa_mod.period(cfa)
    sky_noise = _sky_noise(lights[len(lights) // 2]["path"], dark, flat, P)
    bad = np.zeros((H, W), bool)
    if have_darks:
        bad |= (dark - level) > p.hot_sigma * sky_noise
    from scipy.ndimage import median_filter
    for k in range(P * P):
        v = _plane(flat, k, P)
        loc = median_filter(v, size=9)
        _plane(bad, k, P)[...] |= (v < (1 - p.flat_defect) * loc) | (v > (1 + p.flat_defect) * loc)
    # a pixel the flat's light hardly reached (a sensor's overscan, a dead column: a band the local test above cannot
    # see) would be multiplied by ten or more
    bad |= flat < FLAT_UNLIT * float(np.median(flat))
    moved = True
    if p.hot_from_lights and len(lights) >= 15:
        n0 = int(bad.sum())
        msg, moved = _hot_from_lights([f["path"] for f in lights], dark, flat, bad, median_filter, cfa=cfa,
                                      tick=lambda what: _tick(ctx, what))
        ctx.metric("hot_pixels_from_lights", int(bad.sum()) - n0)
        ctx.log(msg)
    # the sky did not move: a pixel left out would be a hole in the stack, it is filled from its neighbours instead
    ctx.metric("bad_pixels_filled", not moved)
    if not moved:
        ctx.log("the sky did not move on the sensor: the excluded pixels are filled from their same-colour neighbours")
    ctx.save("badmask.npy", bad)
    ctx.metric("bad_pixels_pct", round(100 * float(bad.mean()), 2))
    ctx.metric("sky_noise_adu", round(sky_noise, 2))
    ctx.log(f"master dark level {level:.1f} ADU, flat min {flat.min():.3f}, {100 * bad.mean():.2f} % pixels "
            f"excluded, sky noise {sky_noise:.1f} ADU")
    _calibrated_cache(ctx, p, lights, dark, flat, bad, cfa, white, t0, not moved)


#: what a v2 runner copies from an earlier result whose calibrated cache was lost, with MASTERS (their settings
#: and metrics): calibrate then rebuilds the cache only (``given_masters``)
MASTER_FILES = ("master_dark.npy", "master_flat.npy", "badmask.npy")
MASTERS = "masters.json"
#: masters kept for the Inspect mode only (there when the project has bias or flat dark frames): the cache-only
#: rebuild copies them from the earlier result itself (``from`` of MASTERS)
EXTRA_MASTERS = ("master_bias.npy", "master_flat_dark.npy")


def result_values(values: dict) -> dict:
    """Of a stage's settings (a dict, its manifest's or ``model_dump``'s), those that decide the result (the worker
    count does not)."""
    d = dict(values)
    d.pop("workers", None)
    if d.get("apply_masters", True):                 # the default: as the results made before the setting existed
        d.pop("apply_masters", None)
    return d


def result_params(p: CalibrateParams) -> dict:
    return result_values(p.model_dump(mode="json"))


def given_masters(ctx: Context, p: CalibrateParams) -> dict | None:
    """{"params", "metrics"} of masters placed in the stage's folder with MASTERS, when they were made with the same
    settings; None otherwise (a full calibration, which replaces them)."""
    f = ctx.dir / MASTERS
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    if d.get("params") == result_params(p) and all((ctx.dir / n).is_file() for n in MASTER_FILES):
        return d
    f.unlink(missing_ok=True)
    ctx.log("the masters of the earlier result were made with other settings: full calibration", "warning")
    return None


def rebuild_cache(ctx: Context, p: CalibrateParams, lights: list[dict], given: dict, cfa: str, white: float) -> None:
    """The cache-only path: the calibrated frames from the masters already in the stage's folder (the same files,
    so the same cache as a full run), their metrics restored."""
    t0 = time.time()
    dark, flat, bad = (np.load(ctx.out(n)) for n in MASTER_FILES)
    for k, v in (given.get("metrics") or {}).items():
        if k not in ("cache_scale", "cache_gb"):
            ctx.metric(k, v)
    ctx.log("calibrated frames rebuilt with the masters of the earlier result (same settings)")
    fill = bool((given.get("metrics") or {}).get("bad_pixels_filled", False))
    src = Path(str(given.get("from") or ""))
    for n in EXTRA_MASTERS:
        if given.get("from") and (src / n).is_file() and not ctx.out(n).is_file():
            shutil.copy2(src / n, ctx.out(n))
    _calibrated_cache(ctx, p, lights, dark, flat, bad, cfa, white, t0, fill)
    (ctx.dir / MASTERS).unlink(missing_ok=True)


def _calibrated_cache(ctx: Context, p: CalibrateParams, lights: list[dict], dark: np.ndarray, flat: np.ndarray,
                      bad: np.ndarray, cfa: str, white: float, t0: float, fill: bool = False) -> None:
    """The calibrated cache of the light frames (float16 memmap on the scratch disk), its frame statistics and
    cache.json, from the masters and the bad pixel map; the bad pixels left out (NaN), or with ``fill`` (the sky did
    not move on the sensor) filled with the median of their same-colour neighbours that are not bad."""
    H, W = dark.shape
    level = float(np.median(dark))
    # values are scaled so that the brightest valid value stays below the float16 range
    scale = float(min(1.0, 16000.0 / max(white - level, 1.0)))
    n = len(lights)
    cache = ctx.scratch / "lights.f16"
    done_p = ctx.ckpt / "done.npy"
    stats_p = ctx.ckpt / "stats.npy"
    resume = cache.exists() and done_p.exists() and cache.stat().st_size == n * H * W * 2
    if resume:
        done = np.load(done_p)
        stats = np.load(stats_p)
        ctx.log(f"resuming calibration: {int(done.sum())}/{n} frames already done")
    else:
        done = np.zeros(n, bool)
        stats = np.zeros((n, 4, 2), np.float32)
        mm = np.memmap(cache, dtype=np.float16, mode="w+", shape=(n, H, W))
        del mm
    paths_dir = ctx.ckpt
    np.save(paths_dir / "dark.npy", dark)
    np.save(paths_dir / "flat.npy", flat)
    np.save(paths_dir / "bad.npy", bad)
    todo = [(i, lights[i]["path"]) for i in range(n) if not done[i]]
    sat_raw = p.saturation * white
    init = (str(cache), n, H, W, str(paths_dir), sat_raw, scale, fill, cfa)
    ctx.phase(getattr(ctx, "_lights_from", 0.0), 1.0)          # issue #50: the lights' share of the bar
    ctx.progress(done.mean(), f"calibrating {len(todo)} frames ({p.workers} processes)", force=True)
    with devices.one_maths_thread():                           # issue #125: 1 GB less reserved a worker
        pool = get_context("spawn").Pool(p.workers, initializer=_init, initargs=init)
    with pool:
        for c, (idx, st) in enumerate(pool.imap_unordered(_calib_one, todo, chunksize=2)):
            stats[idx] = st
            done[idx] = True
            if (c + 1) % 20 == 0 or c + 1 == len(todo):
                np.save(done_p, done)
                np.save(stats_p, stats)
            ctx.progress(done.mean(), f"{int(done.sum())}/{n} frames calibrated")
    for f in ("dark.npy", "flat.npy", "bad.npy"):
        (paths_dir / f).unlink(missing_ok=True)
    ctx.save("frame_stats.npy", stats)
    ctx.save_json("cache.json", {"path": str(cache), "n": n, "height": H, "width": W, "scale": scale,
                                 "cfa": cfa, "sat_mark": SAT_MARK, "valid_max": VALID_MAX,
                                 "white_level": white, "dark_level": level})
    ctx.metric("cache_scale", scale)
    ctx.metric("cache_gb", round(n * H * W * 2 / 1e9, 1))
    ctx.log(f"{n} frames calibrated in {(time.time() - t0) / 60:.1f} min (cache {n * H * W * 2 / 1e9:.1f} GB)")
    from .inventory import _superpixel_rgb
    ctx.preview("master_flat", _flat_view(flat, cfa))
    mm = np.memmap(cache, dtype=np.float16, mode="r", shape=(n, H, W))
    fr = np.nan_to_num(np.asarray(mm[n // 2], np.float32))
    ctx.preview("calibrated_frame", _superpixel_rgb(np.where(fr > VALID_MAX, 0, fr), cfa))


def _plane(a: np.ndarray, k: int, P: int = 2) -> np.ndarray:
    """The samples of position k of the colour pattern of period P (core/cfa.plane)."""
    return cfa_mod.plane(a, k, P)


#: the files of the masters read at a time (their reading and decoding are what these phases wait for)
READERS = 4


def _read_frames(paths: list[str], put, start: int = 0, tick=None, read=None) -> None:
    """The files from rank ``start`` read on several threads: ``put(i, frame)`` stores each one in its place (so the
    result does not depend on which is read first); ``tick(c)``: c files of ``paths`` are done."""
    with ThreadPoolExecutor(READERS) as pool:
        jobs = [pool.submit(lambda i=i, q=q: put(i, (read or read_cfa)(q))) for i, q in enumerate(paths) if i >= start]
        for c, job in enumerate(as_completed(jobs)):
            job.result()
            if tick is not None:
                tick(start + c + 1)


def _median_stack(paths: list[str], ctx: Context, label: str) -> np.ndarray:
    """The pixel-wise median of the frames. They are read into one array (no list then stack), and the median is taken
    in bands of rows, in place: the memory is the frames once, not three times (28 darks of 12 Mpx: 1.3 GB, not 4)."""
    first = read_cfa(paths[0])
    stack = np.empty((len(paths), *first.shape), np.float32)
    stack[0] = first
    _tick(ctx, f"reading {label} 1/{len(paths)}")

    def put(i, a):
        stack[i] = a

    _read_frames(paths, put, 1, lambda c: _tick(ctx, f"reading {label} {c}/{len(paths)}"))
    out = np.empty(stack.shape[1:], np.float32)
    for r in range(0, out.shape[0], BAND):
        out[r:r + BAND] = np.median(stack[:, r:r + BAND], axis=0, overwrite_input=True)
    return out


#: the rows a pixel-wise statistic of a stack of frames is taken on at a time (its temporary arrays stay small)
BAND = 256
#: a flat pixel under this share of the flat's median received no light: a bad pixel
FLAT_UNLIT = 0.1


def _master_flat(paths: list[str], fdark: np.ndarray, ctx: Context) -> np.ndarray:
    """The flats, each over its central level, averaged without the values over 3 MAD from the pixel's median. Read
    into one array and reduced in bands of rows: the memory is the frames once (46 flats of 12 Mpx: 2 GB, not 10)."""
    H, W = fdark.shape
    fl = np.empty((len(paths), H, W), np.float32)

    def put(i, a):
        fl[i] = a - fdark

    _read_frames(paths, put, 0, lambda c: _tick(ctx, f"reading flats {c}/{len(paths)}"))
    fl /= np.median(fl[:, H // 2 - 200:H // 2 + 200, W // 2 - 200:W // 2 + 200], axis=(1, 2), keepdims=True)
    return _flats_mean(fl)


def _flats_mean(fl: np.ndarray) -> np.ndarray:
    """The flats (n, H, W), each over its central level, averaged without the values over 3 MAD from the pixel's
    median (by bands of rows)."""
    H, W = fl.shape[1:]
    out = np.empty((H, W), np.float32)
    for r in range(0, H, BAND):
        b = fl[:, r:r + BAND]
        med = np.median(b, axis=0)
        dev = np.abs(b - med)
        keep = dev < 3 * (1.4826 * np.median(dev, axis=0) + 1e-6)
        out[r:r + BAND] = np.where(keep, b, 0).sum(0) / np.maximum(keep.sum(0), 1)
    return out


#: a pixel whose four nearest neighbours (the other colours) all stay within this many sigmas of their sky is a lone
#: pixel: a star or a knot of nebula lights its neighbours too, even a red one in green (a test on the share of its
#: rise flagged the red knots of M8's nebula)
LONE_SIGMA = 3.0


def _tick(ctx: Context, what: str) -> None:
    """One more file of the masters' phase read (issue #50): the bar moves by one file."""
    ctx._files_read = getattr(ctx, "_files_read", 0) + 1
    tot = getattr(ctx, "_files_total", 0)
    ctx.progress(min(1.0, ctx._files_read / tot) if tot else 0.0, what, force=ctx._files_read == 1)


def _hot_from_lights(paths, dark, flat, bad, median_filter, n=40, k=5.0, cfa: str = "RGGB", tick=None) -> str:
    """Flag in ``bad`` the pixels that stand out from their same-colour neighbours in the lower quartile of ``n``
    light frames spread over the session, in sensor coordinates: a defect is bright in every frame. When the sky
    drifted by three star widths or more between the first and the last frame, a star covers a given pixel in a
    small part of the frames only, so that is enough. When it did not, the stars stand still too: a defect is then
    also a lone pixel: its four nearest neighbours (the other colours) within LONE_SIGMA of their sky, where a star
    or a nebula lights them too. Returns (message, whether the sky moved). Each position of the pattern (``cfa``)
    is a plane of its own: its neighbours in the plane are of its colour."""
    P = cfa_mod.period(cfa)
    sub = [paths[i] for i in np.linspace(0, len(paths) - 1, min(n, len(paths))).astype(int)]
    # the pixels already flagged (an unlit edge of the flat, nearly 0: values of 1e8 once divided) make the
    # correlation's peak fall anywhere (a still sky read as moved by 1000 px, its stars then flagged as hot pixels)
    a, b = (np.where(bad, 0.0, np.nan_to_num((read_cfa(q) - dark) / flat, nan=0.0, posinf=0.0, neginf=0.0))
            for q in (sub[0], sub[-1]))
    g0 = cfa_mod.green_position(cfa)
    ga, gb = (cv2.resize(np.ascontiguousarray(_plane(x, g0, P)), None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA).astype(np.float32) for x in (a, b))
    hp = [g - cv2.GaussianBlur(g, (0, 0), 8) for g in (ga, gb)]
    (dx, dy), _ = cv2.phaseCorrelate(hp[0], hp[1])
    drift = 2.0 * P * float(np.hypot(dx, dy))                 # sensor pixels (a green plane at half its resolution)
    moved = drift >= 12.0
    if tick is not None:
        tick("hot pixels: the sky's drift")
        tick("hot pixels: the sky's drift")
    st = np.empty((len(sub), *dark.shape), np.float32)          # one array, the quartile in bands: the frames once

    def put(i, a, st=st):
        st[i] = (a - dark) / flat

    _read_frames(sub, put, 0, (lambda c: tick(f"hot pixels: frame {c}/{len(sub)}")) if tick is not None else None)
    q = np.empty(dark.shape, np.float32)
    for r in range(0, q.shape[0], BAND):
        q[r:r + BAND] = np.percentile(st[:, r:r + BAND], 25, axis=0, overwrite_input=True)
    del st
    lone = None
    if not moved:
        # each pixel's rise above its local sky, in sigmas of its plane; the four nearest are the other colours
        z = np.zeros_like(q)
        for c in range(P * P):
            v = _plane(q, c, P)
            r = v - cv2.blur(median_filter(v, size=9), (15, 15))
            _plane(z, c, P)[...] = r / max(robust_sigma(r[::2, ::2])[1], 1e-6)
        nb = np.stack([np.roll(z, sh, axis=ax) for ax in (0, 1) for sh in (1, -1)])
        lone = np.max(nb, axis=0) < LONE_SIGMA
    n_new = 0
    for c in range(P * P):
        v = _plane(q, c, P)
        h = v - median_filter(v, size=5)
        s = robust_sigma(h[::2, ::2])[1]
        hot = (h > k * s) & ~_plane(bad, c, P)
        if lone is not None:
            hot &= _plane(lone, c, P)
        _plane(bad, c, P)[...] |= hot
        n_new += int(hot.sum())
    how = (f"the sky moved by {drift:.0f} px on the sensor" if moved else
           f"the sky moved by {drift:.0f} px only: lone pixels, not stars")
    return (f"hot pixels from the lights: {n_new} more pixels flagged (lower quartile of {len(sub)} frames; {how})",
            moved)


def _sky_noise(path: str, dark: np.ndarray, flat: np.ndarray, P: int = 2) -> float:
    x = (read_cfa(path) - dark) / flat
    step = max(1, 4 * 2 // P)                         # about the same samples per plane as on a Bayer sensor
    return float(np.median([robust_sigma(_plane(x, k, P)[::step, ::step])[1] for k in range(P * P)]))


def _flat_view(flat: np.ndarray, cfa: str) -> np.ndarray:
    """Flat field shown as a colour image normalised to its maximum (vignetting and dust)."""
    rgb = np.moveaxis(cfa_mod.superpixel(flat, cfa), -1, 0)   # frames already in colour: their three planes (issue #57)
    return np.clip(rgb / np.percentile(rgb, 99.9), 0, 1)


# ---------------------------------------------------------------------- worker processes
_G: dict = {}


def _init(cache, n, H, W, paths_dir, sat_raw, scale, fill=False, cfa="RGGB"):
    d = Path(paths_dir)
    bad = np.load(d / "bad.npy")
    _G.update(dark=np.load(d / "dark.npy"), flat=np.load(d / "flat.npy"), bad=bad,
              mm=np.memmap(cache, dtype=np.float16, mode="r+", shape=(n, H, W)), sat=sat_raw, scale=scale,
              fill=_neighbours(bad, cfa) if fill else None, cfa=cfa)


def _neighbours(bad: np.ndarray, cfa: str = "RGGB"):
    """(rows, columns) of the bad pixels and of their 8 same-colour neighbours (2 px away; clipped at the edges). On
    an X-Trans sensor, whose same-colour neighbours are not on a lattice of step 2: the 8 nearest of its colour within
    2 px (each 3 x 3 window holds every colour, so a 5 x 5 one holds 2 red and 2 blue at least, besides the centre)."""
    H, W = bad.shape
    ys, xs = np.nonzero(bad)
    if cfa_mod.is_xtrans(cfa):
        g = cfa_mod.channels(cfa)
        near = sorted(((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if dy or dx),
                      key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
        table = np.zeros((6, 6, 8, 2), int)
        for i in range(6):
            for j in range(6):
                same = [o for o in near if g[(i + o[0]) % 6, (j + o[1]) % 6] == g[i, j]][:8]
                same += [same[-1]] * (8 - len(same))           # the nearest repeated: a median of the same samples
                table[i, j] = same
        off = table[ys % 6, xs % 6]                            # (n, 8, 2)
        ny, nx = ys[:, None] + off[..., 0], xs[:, None] + off[..., 1]
    else:
        off = [(dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2) if dy or dx]
        ny = np.stack([ys + dy for dy, _ in off], 1)
        nx = np.stack([xs + dx for _, dx in off], 1)
    ny = np.where((ny < 0) | (ny >= H), ys[:, None], ny)
    nx = np.where((nx < 0) | (nx >= W), xs[:, None], nx)
    return ys, xs, ny, nx


def _calib_one(args):
    idx, path = args
    raw = read_cfa(path)
    sat = raw >= _G["sat"]
    x = (raw - _G["dark"]) / _G["flat"] * _G["scale"]
    x = np.minimum(x, VALID_MAX - 1)
    # a bad pixel stays excluded even when it saturates: marked saturated, the stack took a hot pixel for the core
    # of a bright star and drew the drift of the mount with it (green dotted trails on the Flaming Star)
    x[sat & ~_G["bad"]] = SAT_MARK
    x[_G["bad"]] = np.nan
    if _G.get("fill") is not None:
        ys, xs, ny, nx = _G["fill"]
        nb = x[ny, nx]
        nb = np.where(nb > VALID_MAX, np.nan, nb)
        with np.errstate(all="ignore"):
            x[ys, xs] = np.nanmedian(nb, axis=1)          # NaN again when every neighbour is bad
    _G["mm"][idx] = x.astype(np.float16)
    stats = []
    for v in _stat_samples(x, _G.get("cfa") or "RGGB"):
        stats.append(_frame_stat(v))
    return idx, stats


def _frame_stat(v: np.ndarray) -> tuple[float, float]:
    """(level, noise) of a frame's samples; NaN for a frame with too few usable ones (a white frame, taken on purpose
    to mark where a series starts): Selection leaves it out."""
    ok = np.isfinite(v) & (v < VALID_MAX)
    return robust_sigma(v[ok]) if ok.mean() >= MIN_USABLE else (np.nan, np.nan)


def _stat_samples(x: np.ndarray, cfa: str) -> list[np.ndarray]:
    """The four sets of samples frame_stats measures (level, noise): the four Bayer positions (one in four of their
    samples each way); on an X-Trans sensor its red, green (twice: the same four entries as a Bayer sensor's, so that
    their mean weighs the colours alike) and blue samples, about as many."""
    if not cfa_mod.is_xtrans(cfa):
        return [_plane(x, k)[::4, ::4] for k in range(4)]
    sub = x[::5, ::5]                                  # a step prime to 6: every position of the pattern
    col = cfa_mod.colour_map(cfa, *x.shape)[::5, ::5]
    r, g, b = (sub[col == c] for c in range(3))
    return [r, g, g, b]


def open_cache(ctx: Context, mode: str = "r"):
    """(memmap (N, H, W) float16, cache info) of the calibrated frames."""
    info = ctx.load_json("calibrate", "cache.json")
    mm = np.memmap(info["path"], dtype=np.float16, mode=mode, shape=(info["n"], info["height"], info["width"]))
    return mm, info


# ---------------------------------------------------------------------- frames already in colour (issue #57)
#: what a colour frame's masters are, colour by colour, besides the luminance (H, W) kept under the usual names (the
#: stack's saturation level, Inspect)
MASTER_RGB = ("master_dark_rgb.npy", "master_flat_rgb.npy")


def open_colour(info: dict) -> np.memmap:
    """(memmap (N, 3, H, W) float16) of the three planes of the calibrated frames already in colour (``info``:
    cache.json)."""
    return np.memmap(info["rgb_path"], dtype=np.float16, mode="r", shape=(info["n"], 3, info["height"], info["width"]))


class CachedFrames:
    """The calibrated frames of a cache (cache.json ``info``; ``colour``: the three planes of frames already in
    colour), frame i as float32 (a new array) by ``frames[i]``, one thread at a time.

    Each frame is one explicit read of its bytes into a float16 buffer reused from frame to frame (issue #125). Through
    a memory map of the cache, the stack's three passes brought every frame in by page faults, and its pages stayed in
    the process's working set until the system trimmed them: on a night of 754 frames (a 17 GB cache), the stack's
    working set reached 19.9 GB, only 8.2 GB of it private, on a machine of 27.7 GB whose free memory stayed under
    1 GB, about 4,000 pages read back from the disk a second. Read this way, the cache's pages are only in the
    system's file cache, which it gives up first; the file is open only during a read. The values are the same to the
    bit: the same float16, converted to float32 the same way. ``mapped``: through the memory map as before (the tests
    compare both). ``seconds``: the time spent reading the frames, for the passes' lines in the log."""

    def __init__(self, info: dict, colour: bool = False, mapped: bool = False):
        self.path = info["rgb_path"] if colour else info["path"]
        self.n = int(info["n"])
        self.shape = (3, info["height"], info["width"]) if colour else (info["height"], info["width"])
        self.mm = np.memmap(self.path, dtype=np.float16, mode="r", shape=(self.n, *self.shape)) if mapped else None
        self.buf = None if mapped else np.empty(self.shape, np.float16)
        self.seconds = 0.0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i) -> np.ndarray:
        t = time.perf_counter()
        i = int(i)
        if not 0 <= i < self.n:
            raise IndexError(f"frame {i} of a cache of {self.n}")
        if self.mm is not None:
            out = np.asarray(self.mm[i], np.float32)
        else:
            view = memoryview(self.buf).cast("B")
            with open(self.path, "rb", buffering=0) as f:
                f.seek(i * view.nbytes)
                got = 0
                while got < view.nbytes:                        # a read may return less than asked: the rest after
                    k = f.readinto(view[got:])
                    if not k:
                        raise OSError(f"{self.path}: frame {i} cut short (a cache of {self.n} frames expected)")
                    got += k
            out = self.buf.astype(np.float32)
        self.seconds += time.perf_counter() - t
        return out


def calibrate_colour(ctx: Context, p: CalibrateParams, frames: dict) -> None:
    """The calibration of frames already in colour, channel by channel (issue #57): the master dark, bias and flat of
    their own three planes, each frame's dark taken off and its flat divided by colour by colour; written as a
    luminance cache (N, H, W) (what the registration reads) and the planes beside it (``_colour_cache``). The frames
    are not linear when they were stretched (every JPEG is): the correction is then approximate (said by the session
    and the inventory). Hot pixels are not looked for in the lights: a frame already demosaiced has spread them over
    their neighbours."""
    H, W = int(ctx.metric_of("inventory", "height")), int(ctx.metric_of("inventory", "width"))
    black = float(ctx.metric_of("inventory", "black_level"))
    white = float(ctx.metric_of("inventory", "white_level"))
    if not p.apply_masters:
        ctx.log("Calibration cut: the frames are taken as they are (no master dark, flat or defective pixels)")
        ctx.metric("calibration_cut", True)
        _colour_cache(ctx, p, frames["lights"], np.full((3, H, W), black, np.float32), np.ones((3, H, W), np.float32),
                      np.zeros((H, W), bool), white, time.time())
        return
    nm = sum(len(frames[k]) for k in ("bias", "darks", "flats", "flat_darks"))
    ctx._files_total, ctx._files_read, ctx._lights_from = nm, 0, nm / max(nm + len(frames["lights"]), 1)
    ctx.phase(0.0, ctx._lights_from)
    bias = _colour_median([f["path"] for f in frames["bias"]], ctx, "bias") if frames["bias"] else None
    if bias is not None:
        ctx.save("master_bias.npy", cfa_mod.luminance(bias))
    if frames["darks"]:
        dark = _colour_median([f["path"] for f in frames["darks"]], ctx, "darks")
        one = cfa_mod.luminance(read_colour(frames["darks"][0]["path"]))
        ctx.metric("dark_frame_noise", round(robust_sigma(one[::3, ::3])[1], 2))
    else:
        dark = bias.copy() if bias is not None else np.full((3, H, W), black, np.float32)
        ctx.log("no dark frames: using " + ("the master bias" if bias is not None else "the black level") +
                " as dark (hot pixels will not be removed by calibration)", "warning")
    dark = scaled_dark(ctx, dark, bias, black, frames)
    flat = _colour_flat([f["path"] for f in frames["flats"]], [f["path"] for f in frames["flat_darks"]], bias, dark, H, W,
                        p.flat_scale_px, ctx)
    ctx.log("hot pixels seen in the lights: not looked for in frames already in colour (demosaicing has spread them)")
    level = np.median(dark, axis=(1, 2), keepdims=True)
    t0 = time.time()
    sky = read_colour(frames["lights"][len(frames["lights"]) // 2]["path"])
    noise = float(np.median([robust_sigma(((sky - dark) / flat)[c, ::4, ::4])[1] for c in range(3)]))
    bad = np.zeros((H, W), bool)
    if frames["darks"]:
        bad |= ((dark - level) > p.hot_sigma * noise).any(0)
    from scipy.ndimage import median_filter
    for c in range(3):
        loc = median_filter(flat[c], size=9)
        bad |= (flat[c] < (1 - p.flat_defect) * loc) | (flat[c] > (1 + p.flat_defect) * loc)
        bad |= flat[c] < FLAT_UNLIT * float(np.median(flat[c]))
    ctx.metric("bad_pixels_pct", round(100 * float(bad.mean()), 2))
    ctx.metric("sky_noise_adu", round(noise, 2))
    ctx.metric("dark_level", round(float(np.median(dark)), 1))
    ctx.metric("vignetting_min", round(float(flat.min()), 3))
    _colour_cache(ctx, p, frames["lights"], dark, flat, bad, white, t0)


def _colour_median(paths: list[str], ctx: Context, label: str) -> np.ndarray:
    """The pixel-wise median (3, H, W) of frames already in colour, in bands of rows (as ``_median_stack``)."""
    first = read_colour(paths[0])
    stack = np.empty((len(paths), *first.shape), np.float32)
    stack[0] = first
    _tick(ctx, f"reading {label} 1/{len(paths)}")

    def put(i, a):
        stack[i] = a

    _read_frames(paths, put, 1, lambda c: _tick(ctx, f"reading {label} {c}/{len(paths)}"), read_colour)
    out = np.empty(stack.shape[1:], np.float32)
    for r in range(0, out.shape[1], BAND):
        out[:, r:r + BAND] = np.median(stack[:, :, r:r + BAND], axis=0, overwrite_input=True)
    return out


def _colour_flat(paths: list[str], fd_paths: list[str], bias: np.ndarray | None, dark: np.ndarray, H: int, W: int,
                 flat_scale_px: float, ctx: Context) -> np.ndarray:
    """The master flat (3, H, W) of frames already in colour, each colour normalised to 1 at its centre; all ones
    without flats. Each flat over its own central level, per colour, then ``_flats_mean``."""
    if not paths:
        ctx.log("no flat frames: vignetting and dust will not be corrected (the sky model will absorb part of "
                "the vignetting)", "warning")
        return np.ones((3, H, W), np.float32)
    if fd_paths:
        fdark = _colour_median(fd_paths, ctx, "flat darks")
    elif bias is not None:
        fdark = bias
    else:
        fdark = np.broadcast_to(np.median(dark, axis=(1, 2), keepdims=True), (3, H, W)).astype(np.float32)
    fl = np.empty((len(paths), 3, H, W), np.float32)

    def put(i, a):
        fl[i] = a - fdark

    _read_frames(paths, put, 0, lambda c: _tick(ctx, f"reading flats {c}/{len(paths)}"), read_colour)
    y0, y1, x0, x1 = H // 2 - 200, H // 2 + 200, W // 2 - 200, W // 2 + 200
    fl /= np.median(fl[:, :, max(y0, 0):y1, max(x0, 0):x1], axis=(2, 3), keepdims=True)
    flat = np.stack([_flats_mean(fl[:, c]) for c in range(3)])
    for c in range(3):
        if flat_scale_px > 0:
            flat[c] = cv2.GaussianBlur(flat[c], (0, 0), flat_scale_px)
        flat[c] /= np.median(flat[c, max(y0, 0):y1, max(x0, 0):x1])
    if flat_scale_px > 0:
        ctx.log(f"master flat kept above {flat_scale_px:.0f} px only (vignetting, not dust)")
    return flat


def _colour_cache(ctx: Context, p: CalibrateParams, lights: list[dict], dark: np.ndarray, flat: np.ndarray,
                  bad: np.ndarray, white: float, t0: float) -> None:
    """The calibrated frames already in colour: their luminance in ``lights.f16`` (N, H, W), what the registration
    reads as a monochrome mosaic, their three planes in ``lights_rgb.f16`` (N, 3, H, W), read by the stack; the masters
    colour by colour (MASTER_RGB) and their luminance under the usual names."""
    _, H, W = dark.shape
    level = float(np.median(dark))
    scale = float(min(1.0, 16000.0 / max(white - level, 1.0)))
    n = len(lights)
    cache, rgb = ctx.scratch / "lights.f16", ctx.scratch / "lights_rgb.f16"
    ctx.save("master_dark.npy", cfa_mod.luminance(dark))
    ctx.save("master_flat.npy", cfa_mod.luminance(flat))
    ctx.save("badmask.npy", bad)
    for name, a in zip(MASTER_RGB, (dark, flat)):
        ctx.save(name, a)
    mm = np.memmap(cache, dtype=np.float16, mode="w+", shape=(n, H, W))
    mr = np.memmap(rgb, dtype=np.float16, mode="w+", shape=(n, 3, H, W))
    del mm, mr
    d = ctx.ckpt
    np.save(d / "dark.npy", dark)
    np.save(d / "flat.npy", flat)
    np.save(d / "bad.npy", bad)
    stats = np.zeros((n, 4, 2), np.float32)
    init = (str(cache), str(rgb), n, H, W, str(d), p.saturation * white, scale)
    ctx.phase(getattr(ctx, "_lights_from", 0.0), 1.0)
    ctx.progress(0.0, f"calibrating {n} frames ({p.workers} processes)", force=True)
    with devices.one_maths_thread():                           # issue #125: 1 GB less reserved a worker
        pool = get_context("spawn").Pool(p.workers, initializer=_init_colour, initargs=init)
    with pool:
        for c, (idx, st) in enumerate(pool.imap_unordered(_calib_colour, list(enumerate(f["path"] for f in lights)),
                                                          chunksize=2)):
            stats[idx] = st
            ctx.progress((c + 1) / n, f"{c + 1}/{n} frames calibrated")
    for f in ("dark.npy", "flat.npy", "bad.npy"):
        (d / f).unlink(missing_ok=True)
    ctx.save("frame_stats.npy", stats)
    # the cache's pattern is its luminance's: the stages that walk it read one monochrome mosaic (core/cfa.MONO)
    ctx.save_json("cache.json", {"path": str(cache), "rgb_path": str(rgb), "n": n, "height": H, "width": W, "scale": scale,
                                 "cfa": cfa_mod.MONO, "colour": True, "sat_mark": SAT_MARK, "valid_max": VALID_MAX,
                                 "white_level": white, "dark_level": level})
    ctx.metric("cache_scale", scale)
    ctx.metric("cache_gb", round(n * H * W * 8 / 1e9, 1))
    ctx.log(f"{n} frames already in colour calibrated in {(time.time() - t0) / 60:.1f} min")
    from .inventory import _superpixel_rgb
    ctx.preview("master_flat", _flat_view(flat, cfa_mod.RGB))
    fr = np.nan_to_num(np.asarray(np.memmap(rgb, dtype=np.float16, mode="r", shape=(n, 3, H, W))[n // 2], np.float32))
    ctx.preview("calibrated_frame", _superpixel_rgb(np.where(fr > VALID_MAX, 0, fr), cfa_mod.RGB))


def _init_colour(cache, rgb, n, H, W, paths_dir, sat_raw, scale):
    d = Path(paths_dir)
    _G.update(dark=np.load(d / "dark.npy"), flat=np.load(d / "flat.npy"), bad=np.load(d / "bad.npy"),
              mm=np.memmap(cache, dtype=np.float16, mode="r+", shape=(n, H, W)),
              rgb=np.memmap(rgb, dtype=np.float16, mode="r+", shape=(n, 3, H, W)), sat=sat_raw, scale=scale)


def _calib_colour(args):
    """One frame already in colour, calibrated colour by colour: its three planes and its luminance written, a
    saturated sample marked (the luminance too: a star clipped in one colour), the bad pixels left out."""
    idx, path = args
    raw = read_colour(path)
    sat = raw >= _G["sat"]
    x = (raw - _G["dark"]) / _G["flat"] * _G["scale"]
    x = np.minimum(x, VALID_MAX - 1)
    x[sat] = SAT_MARK
    y = cfa_mod.luminance(np.where(sat, 0.0, x))
    y[sat.any(0)] = SAT_MARK
    x[:, _G["bad"]] = np.nan
    y[_G["bad"]] = np.nan
    _G["rgb"][idx] = x.astype(np.float16)
    _G["mm"][idx] = y.astype(np.float16)
    stats = []
    for c in (0, 1, 1, 2):                                  # the same four entries as a Bayer sensor's (their mean)
        stats.append(_frame_stat(x[c, ::4, ::4]))
    return idx, stats

