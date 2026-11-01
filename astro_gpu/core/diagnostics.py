"""Diagnostics of the basic stages for the Inspect mode: what ``api.calibration_frames``, ``registration_table``,
``stack_diagnostics`` and ``color_fit`` compute on demand, from the raw calibration frames and the stage outputs.

Numpy and OpenCV only (no Qt, no torch): any thread may call these, but they read files and compute, so the client
calls them from its worker thread. The measures of a raw frame are cached in memory by file (path, size, date).
"""
from __future__ import annotations

import numbers
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

#: raw values at or above this fraction of the white level count as saturated (CalibrateParams.saturation)
SATURATION = 0.976
#: rings of the radial profile, from the centre (0) to the corners (1) of the frame
RINGS = 10
#: a hot pixel stands out from its same-colour neighbours by this many times the frame's noise
HOT_SIGMA = 10.0
#: an outlier is this many robust sigmas away from the other frames of its kind (and past the tolerance of the measure)
OUTLIER_SIGMA = 5.0
#: a star: a compact source of STAR_MIN_AREA binned pixels at least, STAR_SIGMA x the noise above the local background,
#: in the 2x2-binned frame after a 3x3 median (a hot pixel or a small cosmic-ray hit does not survive the median)
STAR_SIGMA = 6.0
STAR_MIN_AREA = 6
#: the local background of the star count: medians of STAR_TILE x STAR_TILE binned pixels, interpolated; the frame's
#: border (two tiles) is left out
STAR_TILE = 32
#: a dark, bias, flat dark or flat with this many stars or more is a frame of the sky, whatever the other frames say.
#: Real ones hold 0 to 8 (hot pixel clusters, cosmic rays, dust: darks, bias and flats of the Sony A7S, Canon 6D / 40D /
#: Ra, ASI2600MC); the 100 "darks" of 28.12.24 (the Rosette field, shot with the telescope open) hold 26 to 410, lights
#: hundreds to thousands (docs/pilot/DARKS_28_12_24.md)
STARS_NOT_CALIBRATION = 20
#: a dark, bias or flat dark of a camera raw this far above the camera's black level (both: x its noise and % of the
#: sensor's range) holds light, not only the sensor's offset and dark current
SKY_LEVEL_NOISE = 20.0
SKY_LEVEL_RANGE_PCT = 3.0

KIND_ONE = {"darks": "dark", "flats": "flat", "bias": "bias frame", "flat_darks": "flat dark", "lights": "light"}

KIND_PLURAL = {"darks": "darks", "flats": "flats", "bias": "bias frames", "flat_darks": "flat darks",
               "lights": "lights"}

_LOCK = threading.Lock()
_STATS: dict[tuple, dict[str, Any]] = {}


# ---------------------------------------------------------------------------------------------- calibration frames

def _file_key(path: str | Path, white: float) -> tuple:
    p = Path(path)
    st = p.stat()
    return str(p), st.st_size, st.st_mtime_ns, float(white)


def raw_stats(path: str | Path, white: float) -> dict[str, Any]:
    """The measures of one raw frame, in raw ADU (black level not subtracted), cached by file:

    * ``median``: level of the frame; ``planes``: level of each Bayer position (row-major 2x2);
    * ``noise``: robust pixel noise, from the differences of same-colour neighbours (a gradient or vignetting does not
      count), median of the four positions;
    * ``hot_pixels``: pixels above their same-colour 5x5 neighbourhood by HOT_SIGMA x the noise;
    * ``rings``: median level in RINGS rings from the centre to the corners; ``centre`` (inner fifth) and ``edge``
      (outer 15 %, the corners) and ``gradient`` = centre - edge (positive: the centre is brighter);
    * ``saturated_pct``: pixels at or above SATURATION x ``white``; ``max``;
    * ``stars``: compact sources of the frame (``count_stars``): about none in a dark, bias or flat, many in a frame
      of the sky; ``camera_raw``: the file is a camera raw (its black level is the camera's, not a FITS keyword).
    """
    key = _file_key(path, white)
    with _LOCK:
        hit = _STATS.get(key)
    if hit is not None:
        return dict(hit)
    from ..io.frames import frame_kind, read_cfa
    raw = read_cfa(path)
    out = _measure(raw, white)
    out["camera_raw"] = frame_kind(path) == "raw"
    with _LOCK:
        _STATS[key] = out
    return dict(out)


def _measure(raw: np.ndarray, white: float) -> dict[str, Any]:
    raw = np.asarray(raw, np.float32)
    H, W = raw.shape
    planes, noises, hot = [], [], 0
    for k in range(4):
        i, j = divmod(k, 2)
        v = np.ascontiguousarray(raw[i::2, j::2])
        planes.append(float(np.median(v[::2, ::2])))
        d = (v[::2, 1:] - v[::2, :-1]).ravel()
        s = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))
        noises.append(s)
        if s > 0:
            hot += int(np.count_nonzero(v - cv2.medianBlur(v, 5) > HOT_SIGMA * s))
    # radial profile on a binned image (2x2 cells, then about 300 px on the long side)
    sp = raw[:H - H % 2, :W - W % 2].reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    f = max(1, int(round(max(sp.shape) / 300)))
    small = cv2.resize(sp, (max(1, sp.shape[1] // f), max(1, sp.shape[0] // f)), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx + 0.5) / w - 0.5, ((yy + 0.5) / h - 0.5) * h / w) / np.hypot(0.5, 0.5 * h / w)
    ring = np.minimum((r * RINGS).astype(int), RINGS - 1)
    rings = [float(np.median(small[ring == q])) if np.any(ring == q) else float("nan") for q in range(RINGS)]
    centre = float(np.median(small[r < 0.2]))
    edge = float(np.median(small[r > 0.85])) if np.any(r > 0.85) else rings[-1]
    return {"median": float(np.median(raw[::3, ::3])), "planes": planes, "noise": float(np.median(noises)),
            "hot_pixels": hot, "rings": rings, "centre": centre, "edge": edge, "gradient": centre - edge,
            "saturated_pct": float(100.0 * np.count_nonzero(raw >= SATURATION * white) / raw.size),
            "max": float(raw.max()), "stars": count_stars(sp)}


def count_stars(binned: np.ndarray) -> int:
    """The stars of a 2x2-binned frame: compact sources STAR_SIGMA x the noise above the local background and
    STAR_MIN_AREA binned pixels at least, after a 3x3 median. A dark, bias or flat holds none: hot pixels and small
    cosmic-ray hits are one or two pixels and do not survive the median; a smooth glow (amp glow, the growing central
    light, vignetting) is in the background. Left out: the border (two STAR_TILE: the sharp edge of the vignetting of a
    full-frame sensor lies there), long or hollow shapes (a dust ring, a column), and the bright rim of a dust shadow
    or of a sharp vignetting edge (something much darker than the background lies next to it). Numbers on real
    frames: STARS_NOT_CALIBRATION."""
    a = np.asarray(binned, np.float32)
    t = STAR_TILE
    th, tw = a.shape[0] // t, a.shape[1] // t
    if th < 5 or tw < 5:
        return 0
    a = a[:th * t, :tw * t]
    bg = np.median(a.reshape(th, t, tw, t).transpose(0, 2, 1, 3).reshape(th, tw, t * t), axis=2).astype(np.float32)
    res = np.ascontiguousarray(a - cv2.resize(bg, (tw * t, th * t), interpolation=cv2.INTER_LINEAR))
    res = cv2.medianBlur(res, 3)[2 * t:-2 * t, 2 * t:-2 * t]
    v = res[::2, ::2]
    s = float(1.4826 * np.median(np.abs(v - np.median(v))))
    s = s if s > 0 else float(np.std(v))
    if not s > 0:
        return 0
    n, lab, st, cen = cv2.connectedComponentsWithStats((res > STAR_SIGMA * s).astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0
    area, bw, bh = st[1:, cv2.CC_STAT_AREA], st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]
    big, small = np.maximum(bw, bh), np.minimum(bw, bh)
    ok = (area >= STAR_MIN_AREA) & (big <= 3 * small + 2) & (area >= 0.4 * bw * bh) & (big <= 2 * t)
    if not ok.any():
        return 0
    # the darkest of the smoothed surroundings: noise stays above -1 x s; a dust shadow or a vignetting edge does not
    low = cv2.erode(cv2.GaussianBlur(res, (0, 0), 2.0), np.ones((21, 21), np.uint8))
    x = np.clip(np.round(cen[1:, 0]).astype(int), 0, res.shape[1] - 1)
    y = np.clip(np.round(cen[1:, 1]).astype(int), 0, res.shape[0] - 1)
    ok &= low[y, x] > -3.0 * s
    return int(np.count_nonzero(ok))


def raw_temperature(path: str | Path) -> float | None:
    """The sensor or camera temperature (°C) a RAW file's maker notes give (Canon: CameraTemperature), else None."""
    try:
        import exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=True)
    except Exception:  # noqa: BLE001 - metadata is optional
        return None
    for k, v in tags.items():
        name = k.split(" ", 1)[-1]
        if name in ("CameraTemperature", "SensorTemperature", "AmbientTemperature"):
            try:
                return float(str(v).split()[0])
            except ValueError:
                continue
    return None


def flat_measures(st: dict[str, Any], zero: float, white: float) -> dict[str, Any]:
    """A flat's level as % of the sensor's range above ``zero`` (its offset: the flat dark, bias or dark level), and its
    vignetting: each ring's level relative to the centre (1 = as bright as the centre)."""
    span = max(white - zero, 1.0)
    c = st["rings"][0] - zero
    vig = [float((v - zero) / c) if c > 0 else float("nan") for v in st["rings"]]
    return {"level_pct": float(100.0 * (st["median"] - zero) / span), "vignetting": vig,
            "corner_pct": float(100.0 * vig[-1]) if np.isfinite(vig[-1]) else float("nan")}


def _robust(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return float("nan"), float("nan")
    c = float(np.median(v))
    return c, float(1.4826 * np.median(np.abs(v - c)))


def _off(v: float | None, c: float, s: float, tol: float) -> bool:
    return v is not None and np.isfinite(v) and np.isfinite(c) and abs(v - c) > max(OUTLIER_SIGMA * s, tol)


def _finite(*v: Any) -> bool:
    return all(isinstance(x, numbers.Real) and not isinstance(x, bool) and np.isfinite(x) for x in v)


def flag_frames(kind: str, stats: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """(flag, reason, checks) of each frame of one kind, the reason in plain words:

    * "bad": the frame is not of its kind, whatever the other frames say (a whole set of wrong frames looks normal
      against itself): a dark, bias, flat dark or flat with stars (``stars`` >= STARS_NOT_CALIBRATION: a frame of the
      sky), or a dark, bias or flat dark of a camera raw whose level is far above the camera's black level (light
      reached the sensor);
    * "warn": an outlier against the other frames of its kind (three at least: exposure, temperature, level, noise, hot
      pixels, centre-to-corner gradient; for lights the calibrated sky level and noise only) or, for a flat, badly
      exposed;
    * "ok" otherwise.

    Nothing is excluded: the stage still uses every frame. ``checks``: [{"key", "text", numbers}] for translation."""
    plural = KIND_PLURAL.get(kind, kind)
    one = KIND_ONE.get(kind, kind)
    out = []
    many = len(stats) >= 3
    cen = {}
    for m in ("median", "noise", "hot_pixels", "gradient", "temperature", "exposure", "level_pct"):
        cen[m] = _robust([s.get(m) for s in stats])
    noise_c = cen["noise"][0] if np.isfinite(cen["noise"][0]) else 0.0
    for s in stats:
        checks: list[dict[str, Any]] = []

        def add(key: str, text: str, **args: Any) -> None:
            checks.append({"key": key, "text": text, **args})

        # ---- absolute: is it a calibration frame at all?
        n = s.get("stars")
        if kind in ("darks", "bias", "flat_darks", "flats") and _finite(n) and n >= STARS_NOT_CALIBRATION:
            add("stars", f"contains stars ({n:.0f} found): a frame of the sky, not a {one}"
                + (" (was the cap on?)" if kind != "flats" else ""), value=int(n), limit=STARS_NOT_CALIBRATION)
        black, white, med, noise = s.get("black"), s.get("white"), s.get("median"), s.get("noise")
        if kind in ("darks", "bias", "flat_darks") and s.get("camera_raw") and _finite(black, white, med, noise) \
                and noise > 0 and white > black:
            above = med - black
            if above > max(SKY_LEVEL_NOISE * noise, SKY_LEVEL_RANGE_PCT / 100.0 * (white - black)):
                add("sky_level", f"its level is {above:.0f} ADU above the camera's black level ({black:.0f} ADU), "
                                 f"{above / noise:.0f}x its noise: light reached the sensor, not a {one}",
                    value=above, black=black, noise=noise)
        absolute = len(checks)

        if many:
            v, (c, sg) = s.get("exposure"), cen["exposure"]
            if _off(v, c, sg, 0.01 * abs(c)):
                add("exposure", f"exposure {v:g} s, the other {plural} {c:g} s", value=v, others=c)
            v, (c, sg) = s.get("temperature"), cen["temperature"]
            if _off(v, c, sg, 3.0):
                add("temperature", f"taken at {v:.0f} °C, the other {plural} at {c:.0f} °C", value=v, others=c)
            if kind == "flats":
                v, (c, sg) = s.get("level_pct"), cen["level_pct"]
                if _off(v, c, sg, 0.1 * abs(c)):
                    add("level", f"its level ({v:.0f} % of full scale) differs from the other flats ({c:.0f} %): "
                                 "the light changed?", value=v, others=c)
            elif kind == "lights":
                v, (c, sg) = s.get("median"), cen["median"]
                if _off(v, c, sg, 0.25 * abs(c)):
                    add("sky", f"its sky level ({v:.0f} ADU) differs from the other lights ({c:.0f} ADU): moon, "
                               "twilight, clouds or dew?", value=v, others=c)
            else:
                v, (c, sg) = s.get("median"), cen["median"]
                if _off(v, c, sg, max(2.0, 0.5 * noise_c, 0.002 * abs(c))):
                    add("median", f"its level ({v:.1f} ADU) differs from the other {plural} ({c:.1f} ADU)"
                        + (": another exposure, ISO or temperature?" if kind in ("darks", "flat_darks") else ""),
                        value=v, others=c)
                v, (c, sg) = s.get("hot_pixels"), cen["hot_pixels"]
                if v is not None and np.isfinite(c) and v > max(2.0 * c, c + max(50.0, OUTLIER_SIGMA * sg)):
                    add("hot_pixels", f"{v} hot pixels, the other {plural} about {c:.0f}: a warmer sensor?",
                        value=v, others=c)
                v, (c, sg) = s.get("gradient"), cen["gradient"]
                if _off(v, c, sg, max(2.0, 0.5 * noise_c)):
                    add("gradient", f"the centre is {abs(v):.1f} ADU {'brighter' if v > 0 else 'darker'} than the "
                                    f"corners, the other {plural} {c:+.1f} ADU: heat or light reaching the sensor?",
                        value=v, others=c)
            v, (c, sg) = s.get("noise"), cen["noise"]
            if _off(v, c, sg, (0.25 if kind == "lights" else 0.2) * abs(c)):
                add("noise", f"its noise ({v:.1f} ADU) is {v / c:.1f}x that of the other {plural} ({c:.1f} ADU)"
                    if c > 0 else f"its noise is {v:.1f} ADU", value=v, others=c)
        if kind == "flats":
            sat, lvl = s.get("saturated_pct") or 0.0, s.get("level_pct")
            if sat > 0.1:
                add("saturated", f"{sat:.2f} % of its pixels are saturated", value=sat)
            if lvl is not None and np.isfinite(lvl):
                if lvl < 10:
                    add("underexposed", f"its level is only {lvl:.0f} % of full scale (aim for 30-60 %)", value=lvl)
                elif lvl > 85:
                    add("overexposed", f"its level is {lvl:.0f} % of full scale: close to saturation, where the "
                                       "sensor is no longer linear", value=lvl)
        flag = "bad" if absolute else "warn" if checks else "ok"
        reason = "; ".join(c["text"] for c in checks)
        out.append((flag, (reason[0].upper() + reason[1:]) if reason else "", checks))
    return out


# ---------------------------------------------------------------------------------------------- registration

def frame_geometry(coef: np.ndarray, width: float, height: float,
                   aligned: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(dx, dy, rotation, scale) of each frame from its registration polynomial (n, 2, 10) onto the reference
    (stages/register.py: coordinates normalised to [-1, 1] on a ``width`` x ``height`` sensor):

    * dx, dy: how far the frame's stars sit from the reference's, at the centre of the frame (frame minus reference,
      sensor pixels; the client's drift chart uses the same sign);
    * rotation: the frame's field turned relative to the reference, degrees, positive clockwise as the image is
      shown (y down);
    * scale: the star spacing of the frame over the reference's (1 = same scale).

    NaN for the frames that are not ``aligned``."""
    c = np.asarray(coef, float)
    n = len(c)
    ok = np.ones(n, bool) if aligned is None else np.asarray(aligned, bool)
    hw, hh = width / 2.0, height / 2.0
    J = np.zeros((n, 2, 2))
    J[:, 0, 0], J[:, 0, 1] = c[:, 0, 1] / hw, c[:, 0, 2] / hh
    J[:, 1, 0], J[:, 1, 1] = c[:, 1, 1] / hw, c[:, 1, 2] / hh
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    ok = ok & np.isfinite(det) & (np.abs(det) > 1e-9)
    nan = np.full(n, np.nan)
    dx, dy, rot, scale = nan.copy(), nan.copy(), nan.copy(), nan.copy()
    dx[ok], dy[ok] = hw - c[ok, 0, 0], hh - c[ok, 1, 0]
    Ji = np.linalg.inv(J[ok])                                   # reference -> frame
    rot[ok] = np.degrees(np.arctan2(Ji[:, 1, 0] - Ji[:, 0, 1], Ji[:, 0, 0] + Ji[:, 1, 1]))
    scale[ok] = np.sqrt(np.abs(np.linalg.det(Ji)))
    return dx, dy, rot, scale


# ---------------------------------------------------------------------------------------------- stacking

_NOISE: dict[tuple, dict[str, float]] = {}


def stack_noise(stack_dir: Path, weights: Sequence[float], half_block: int) -> dict[str, float]:
    """The noise of each channel of a done stack from its half-stacks (``stack.halves_noise``), for a result made
    before the stack measured it: the halves are read memory-mapped, sampled and closed; the half weights come from
    the frames' weights and the half-stack block, as the stack dealt them. Cached by folder and file date."""
    from ..stages.stack import halves_noise, noise_step
    fa, fb = stack_dir / "drz_A.npy", stack_dir / "drz_B.npy"
    key = (str(stack_dir), fa.stat().st_mtime_ns, fb.stat().st_mtime_ns)
    with _LOCK:
        hit = _NOISE.get(key)
    if hit is not None:
        return dict(hit)
    sample = []
    for f in (fa, fb):
        a = np.load(f, mmap_mode="r")
        step = noise_step(a.shape[1], a.shape[2])
        sample.append(np.array(a[:, ::step, ::step]))
        del a
    w = np.asarray(weights, float)
    idx = np.arange(len(w))
    in_a = (idx // max(1, int(half_block))) % 2 == 0
    out = halves_noise(sample[0], sample[1], float(w[in_a & (w > 0)].sum()), float(w[~in_a & (w > 0)].sum()))
    with _LOCK:
        _NOISE[key] = out
    return dict(out)


# ---------------------------------------------------------------------------------------------- photometric colour

_BPRP: dict[tuple, dict[int, float]] = {}


def bp_rp(catalog_dir: Path | None, sources: np.ndarray) -> np.ndarray:
    """Gaia BP - RP of the given source ids from the catalogue result's gaia_deep.npz (NaN when unknown)."""
    out = np.full(len(sources), np.nan)
    f = catalog_dir / "gaia_deep.npz" if catalog_dir is not None else None
    if f is None or not f.is_file():
        return out
    key = (str(f), f.stat().st_mtime_ns)
    with _LOCK:
        table = _BPRP.get(key)
    if table is None:
        with np.load(f) as z:
            if not {"source", "bp", "rp"} <= set(z.files):
                return out
            src, bp, rp = z["source"], z["bp"], z["rp"]
        order = np.argsort(src)
        table = {"src": src[order], "c": (bp - rp)[order]}
        with _LOCK:
            _BPRP[key] = table
    src, col = table["src"], table["c"]
    if not len(src):
        return out
    s = np.asarray(sources, src.dtype)
    j = np.clip(np.searchsorted(src, s), 0, len(src) - 1)
    hit = src[j] == s
    out[hit] = col[j[hit]]
    return out


def colour_fit_again(img: np.ndarray, xp: Any, aperture: float, white_temp: float, Mtot: np.ndarray,
                     S: np.ndarray, from_fit: bool) -> dict[str, np.ndarray]:
    """The per-star fit of a colour result made before the stage kept it (``color.star_fit``), made again from what
    it kept: the same photometry on its stack (the same code, so the same stars and fluxes), the outliers of the
    sensitivity-curve fit (``from_fit``: when the free fit gave the curves), and the camera -> XYZ matrix taken back
    from M.npy (its scale, lost there, does not change a chromaticity)."""
    import colour

    from ..stages.color import XYZ_TO_SRGB, bradford, colour_star_table, fit_response, star_fit, white_xyz
    table, why = colour_star_table(img, np.asarray(xp["x"]), np.asarray(xp["y"]), aperture)
    idx = np.where(why == "")[0]
    kept: list = []
    if from_fit and len(idx):
        fit_response(np.asarray(xp["flux"])[idx], np.asarray(xp["lam"]), table[idx], kept=kept)
    w_d65 = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    M = np.linalg.solve(XYZ_TO_SRGB @ bradford(white_xyz(white_temp), w_d65), np.asarray(Mtot, float))
    return star_fit(xp, table, why, idx, kept, S, M)
