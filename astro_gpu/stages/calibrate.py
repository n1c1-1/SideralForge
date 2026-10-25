"""Calibration of the raw Bayer frames (no debayering): master dark / bias / flat, bad pixel map, calibrated cache."""
from __future__ import annotations

import os
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from ..core.imgutil import robust_sigma
from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage
from ..io.frames import read_cfa

SAT_MARK = 65000.0      # value written for saturated pixels in the cache
VALID_MAX = 60000.0     # anything above is treated as saturated downstream


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
    hot_from_lights: bool = P(True, label="Hot pixels seen in the lights", level=ADVANCED,
                              help="Also flag the pixels that stand out from their neighbours in many light frames "
                                   "taken at different sky positions (the sky drifts across the sensor, a defect "
                                   "does not). Darks taken at another temperature miss pixels that are hot only "
                                   "when the sensor is warmer; in the stack they draw the drift path of the mount.",
                              tip="Skipped automatically when the sky did not move on the sensor during the night.")
    flat_scale_px: float = P(0.0, label="Flat: large scales only", level=ADVANCED, ge=0, le=500, step=5, unit="px",
                             help="Keep only the scales of the master flat larger than this (Gaussian sigma, sensor "
                                  "pixels). 0 = the whole flat. For flat frames borrowed from another night of the "
                                  "same optics: they hold the vignetting of the instrument, but not this night's dust.",
                             tip="About 40 px removes dust donuts and keeps the vignetting.")
    saturation: float = P(0.976, label="Saturation level", level=ADVANCED, ge=0.5, le=1.0, step=0.001, unit="x white level",
                          help="Raw values above this fraction of the sensor white level are flagged as saturated; the "
                               "stacker then treats them specially instead of averaging clipped values.",
                          tip="Keep the default unless star cores look flat-topped or rings appear around bright stars.")
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
    version = "3"          # 2: hot pixels also seen in the lights; 3: a bad pixel stays excluded when saturated

    def auto(self, ctx: Context) -> dict[str, Auto]:
        n = max(1, min(12, (os.cpu_count() or 4) - 2))
        return {"workers": Auto(n, f"{os.cpu_count()} CPU cores detected")}

    def run(self, ctx: Context, p: CalibrateParams) -> None:
        frames = ctx.load_json("inventory", "frames.json")
        H, W = int(ctx.metric_of("inventory", "height")), int(ctx.metric_of("inventory", "width"))
        black = float(ctx.metric_of("inventory", "black_level"))
        bias = master_bias([f["path"] for f in frames["bias"]], ctx)
        dark, dark_noise = master_dark([f["path"] for f in frames["darks"]], bias, H, W, black, ctx)
        flat = master_flat([f["path"] for f in frames["flats"]], [f["path"] for f in frames["flat_darks"]], bias,
                           float(np.median(dark)), H, W, p.flat_scale_px, ctx)
        calibrate_lights(ctx, p, frames["lights"], dark, flat, bool(frames["darks"]), dark_noise,
                         ctx.metric_of("inventory", "cfa"), float(ctx.metric_of("inventory", "white_level")))


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
        return dark, robust_sigma(one[::3, ::3])[1]
    dark = bias.copy() if bias is not None else np.full((H, W), black, np.float32)
    ctx.log("no dark frames: using " + ("the master bias" if bias is not None else "the black level") +
            " as dark (hot pixels will not be removed by calibration)", "warning")
    return dark, None


def master_flat(paths: list[str], flat_dark_paths: list[str], bias: np.ndarray | None, dark_level: float, H: int,
                W: int, flat_scale_px: float, ctx: Context) -> np.ndarray:
    """Master flat normalised to 1 at the centre of each Bayer plane (all ones without flats)."""
    if paths:
        if flat_dark_paths:
            fdark = _median_stack(flat_dark_paths, ctx, "flat darks")
        elif bias is not None:
            fdark = bias
        else:
            fdark = np.full((H, W), dark_level, np.float32)   # flats are short: dark current negligible
        flat = _master_flat(paths, fdark, ctx)
        if flat_scale_px > 0:
            # borrowed flat: vignetting only (each colour plane smoothed; its dust is not this night's dust)
            for k in range(4):
                i, j = divmod(k, 2)
                flat[i::2, j::2] = cv2.GaussianBlur(flat[i::2, j::2], (0, 0), flat_scale_px / 2.0)
            ctx.log(f"master flat kept above {flat_scale_px:.0f} px only (vignetting, not dust)")
    else:
        flat = np.ones((H, W), np.float32)
        ctx.log("no flat frames: vignetting and dust will not be corrected (the sky model will absorb part of "
                "the vignetting)", "warning")
    for k in range(4):
        v = _plane(flat, k)
        c = v[v.shape[0] // 2 - 100:v.shape[0] // 2 + 100, v.shape[1] // 2 - 100:v.shape[1] // 2 + 100]
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
    sky_noise = _sky_noise(lights[len(lights) // 2]["path"], dark, flat)
    bad = np.zeros((H, W), bool)
    if have_darks:
        bad |= (dark - level) > p.hot_sigma * sky_noise
    from scipy.ndimage import median_filter
    for k in range(4):
        v = _plane(flat, k)
        loc = median_filter(v, size=9)
        _plane(bad, k)[...] |= (v < (1 - p.flat_defect) * loc) | (v > (1 + p.flat_defect) * loc)
    if p.hot_from_lights and len(lights) >= 15:
        n0 = int(bad.sum())
        msg = _hot_from_lights([f["path"] for f in lights], dark, flat, bad, median_filter)
        ctx.metric("hot_pixels_from_lights", int(bad.sum()) - n0)
        ctx.log(msg)
    ctx.save("badmask.npy", bad)
    ctx.metric("bad_pixels_pct", round(100 * float(bad.mean()), 2))
    ctx.metric("sky_noise_adu", round(sky_noise, 2))
    ctx.log(f"master dark level {level:.1f} ADU, flat min {flat.min():.3f}, {100 * bad.mean():.2f} % pixels "
            f"excluded, sky noise {sky_noise:.1f} ADU")

    # ---------------- calibrated cache (float16 memmap on the scratch disk)
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
    init = (str(cache), n, H, W, str(paths_dir), sat_raw, scale)
    with get_context("spawn").Pool(p.workers, initializer=_init, initargs=init) as pool:
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


def _plane(a: np.ndarray, k: int) -> np.ndarray:
    i, j = divmod(k, 2)
    return a[..., i::2, j::2]


def _median_stack(paths: list[str], ctx: Context, label: str) -> np.ndarray:
    stack = []
    for i, p in enumerate(paths):
        stack.append(read_cfa(p))
        ctx.progress(0.0, f"reading {label} {i + 1}/{len(paths)}", force=(i == 0))
    return np.median(np.stack(stack), axis=0).astype(np.float32)


def _master_flat(paths: list[str], fdark: np.ndarray, ctx: Context) -> np.ndarray:
    H, W = fdark.shape
    fl = np.stack([read_cfa(p) - fdark for p in paths])
    fl /= np.median(fl[:, H // 2 - 200:H // 2 + 200, W // 2 - 200:W // 2 + 200], axis=(1, 2), keepdims=True)
    med = np.median(fl, axis=0)
    mad = 1.4826 * np.median(np.abs(fl - med), axis=0) + 1e-6
    keep = np.abs(fl - med) < 3 * mad
    return (np.where(keep, fl, 0).sum(0) / np.maximum(keep.sum(0), 1)).astype(np.float32)


def _hot_from_lights(paths, dark, flat, bad, median_filter, n=40, k=5.0) -> str:
    """Flag in ``bad`` the pixels that stand out from their same-colour neighbours in the lower quartile of ``n``
    light frames spread over the session, in sensor coordinates: a defect is bright in every frame, the sky has
    drifted (a star covers a given pixel in a small part of the frames only). Skipped when the sky moved by less
    than three star widths between the first and the last frame (stars would then look like defects)."""
    sub = [paths[i] for i in np.linspace(0, len(paths) - 1, min(n, len(paths))).astype(int)]
    a, b = ((read_cfa(q) - dark) / flat for q in (sub[0], sub[-1]))
    ga, gb = (cv2.resize(_plane(x, 1), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
              for x in (a, b))
    hp = [g - cv2.GaussianBlur(g, (0, 0), 8) for g in (ga, gb)]
    (dx, dy), _ = cv2.phaseCorrelate(hp[0], hp[1])
    drift = 4.0 * float(np.hypot(dx, dy))                     # sensor pixels (colour plane at half resolution)
    if drift < 12.0:
        return f"hot pixels from the lights: skipped, the sky moved by {drift:.0f} px only on the sensor"
    q = np.percentile(np.stack([(read_cfa(x) - dark) / flat for x in sub]), 25, axis=0)
    n_new = 0
    for c in range(4):
        v = _plane(q, c)
        h = v - median_filter(v, size=5)
        s = robust_sigma(h[::2, ::2])[1]
        hot = (h > k * s) & ~_plane(bad, c)
        _plane(bad, c)[...] |= hot
        n_new += int(hot.sum())
    return (f"hot pixels from the lights: {n_new} more pixels flagged (lower quartile of {len(sub)} frames; the sky "
            f"moved by {drift:.0f} px on the sensor)")


def _sky_noise(path: str, dark: np.ndarray, flat: np.ndarray) -> float:
    x = (read_cfa(path) - dark) / flat
    return float(np.median([robust_sigma(_plane(x, k)[::4, ::4])[1] for k in range(4)]))


def _flat_view(flat: np.ndarray, cfa: str) -> np.ndarray:
    """Flat field shown as a colour image normalised to its maximum (vignetting and dust)."""
    h, w = flat.shape
    planes = {c: [] for c in "RGB"}
    for k, c in enumerate(cfa):
        i, j = divmod(k, 2)
        planes[c].append(flat[i:h - h % 2:2, j:w - w % 2:2])
    rgb = np.stack([np.mean(planes[c], axis=0) for c in "RGB"])
    return np.clip(rgb / np.percentile(rgb, 99.9), 0, 1)


# ---------------------------------------------------------------------- worker processes
_G: dict = {}


def _init(cache, n, H, W, paths_dir, sat_raw, scale):
    d = Path(paths_dir)
    _G.update(dark=np.load(d / "dark.npy"), flat=np.load(d / "flat.npy"), bad=np.load(d / "bad.npy"),
              mm=np.memmap(cache, dtype=np.float16, mode="r+", shape=(n, H, W)), sat=sat_raw, scale=scale)


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
    _G["mm"][idx] = x.astype(np.float16)
    stats = []
    for k in range(4):
        v = _plane(x, k)[::4, ::4]
        v = v[np.isfinite(v) & (v < VALID_MAX)]
        stats.append(robust_sigma(v))
    return idx, stats


def open_cache(ctx: Context, mode: str = "r"):
    """(memmap (N, H, W) float16, cache info) of the calibrated frames."""
    info = ctx.load_json("calibrate", "cache.json")
    mm = np.memmap(info["path"], dtype=np.float16, mode=mode, shape=(info["n"], info["height"], info["width"]))
    return mm, info

