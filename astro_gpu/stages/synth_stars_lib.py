"""Stars redrawn on a stars-alone image (issue #77): detected in the image, each with its own flux and colour, and one
star shape for all of them: the shape measured at the centre of the field, or the instrument's theoretical one.

The theoretical shape is computed from the aperture (Fraunhofer: the squared modulus of the Fourier transform of the
pupil, with its central obstruction and spider), per colour and over the width of each colour band, then blurred by
the seeing. The spikes come out of the computation, and their strength follows each star's flux.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import fftconvolve

#: the colour bands of the three channels, nm
BANDS = ((590.0, 680.0), (500.0, 570.0), (420.0, 500.0))
#: the wavelengths sampled in each band (the fringes of one wavelength do not show in a broad band)
BAND_SAMPLES = 5
#: the half-size of the stamps of the measured shape
R_MEASURED = 32
#: stars whose peak is above this share of the brightest are taken as saturated
SATURATED = 0.6
#: a star is detected this many noise units above the sky of the stars-alone image
DETECT_SIGMAS = 8.0


@dataclass
class Instrument:
    """The optics of the theoretical shape. ``obstruction``: the central obstruction's diameter, a share of the
    aperture; ``spider_arms``: 0 for none, 4 gives a cross; ``seeing_px``: the width at half height of the final star."""

    diameter_mm: float
    focal_mm: float
    pixel_um: float
    obstruction: float = 0.0
    spider_arms: int = 0
    spider_mm: float = 1.0
    angle_deg: float = 0.0
    seeing_px: float = 2.5


def pupil(N: int, step: float, ins: Instrument) -> np.ndarray:
    """The aperture on an N x N grid of ``step`` mm: 1 where light passes, its edges softened over one grid step."""
    c = (np.arange(N) - N // 2) * step
    yy, xx = np.meshgrid(c, c, indexing="ij")
    r = np.hypot(yy, xx)
    a = np.clip((ins.diameter_mm / 2 - r) / step + 0.5, 0, 1)
    if ins.obstruction > 0:
        a *= np.clip((r - ins.obstruction * ins.diameter_mm / 2) / step + 0.5, 0, 1)
    for k in range(ins.spider_arms):
        t = math.radians(ins.angle_deg) + 2 * math.pi * k / ins.spider_arms
        along = xx * math.cos(t) + yy * math.sin(t)
        across = np.abs(-xx * math.sin(t) + yy * math.cos(t))
        a *= 1 - np.clip((ins.spider_mm / 2 - across) / step + 0.5, 0, 1) * (along > 0)
    return a.astype(np.float32)


def moffat(fwhm: float, r: int, beta: float = 3.0) -> np.ndarray:
    alpha = fwhm / (2 * math.sqrt(2 ** (1 / beta) - 1))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (1 + (yy ** 2 + xx ** 2) / alpha ** 2) ** -beta
    return k / k.sum()


def theoretical_shape(ins: Instrument, R: int = 300) -> np.ndarray:
    """The star of this instrument (3, 2R+1, 2R+1), each colour summing to 1: the diffraction figure of the aperture,
    averaged over the colour band, on the pixel grid, blurred by the seeing."""
    lam_max = max(b[1] for b in BANDS) * 1e-6                     # mm
    lam_min = min(b[0] for b in BANDS) * 1e-6
    # the image's sample small enough for the pupil to fit twice in the grid, at the shortest wavelength
    os_ = max(3, int(math.ceil(ins.pixel_um * 1e-3 * 2 * ins.diameter_mm / (lam_min * ins.focal_mm))) | 1)
    N = min(8192, 2 * (R + 2) * os_)
    N += N % 2
    R = min(R, N // (2 * os_) - 2)
    delta = ins.pixel_um * 1e-3 / os_                             # mm on the sensor
    out = []
    for lo, hi in BANDS:
        total = 0.0
        for lam_nm in np.linspace(lo, hi, BAND_SAMPLES):
            lam = lam_nm * 1e-6
            step = lam * ins.focal_mm / (N * delta)               # mm of pupil per grid step
            p = pupil(N, step, ins)
            I = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(p)))) ** 2
            total = total + I / I.sum()
        m = N // 2
        half = os_ // 2
        cut = total[m - os_ * R - half:m + os_ * R + half + 1, m - os_ * R - half:m + os_ * R + half + 1]
        binned = cut.reshape(2 * R + 1, os_, 2 * R + 1, os_).sum((1, 3))
        out.append(binned)
    del lam_max
    k = moffat(ins.seeing_px, int(6 * ins.seeing_px) | 1)
    shapes = [np.clip(fftconvolve(c, k, mode="same"), 0, None) for c in out]
    return np.stack([s / s.sum() for s in shapes]).astype(np.float32)


def _noise(a: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(a - np.median(a))))


def _maxima(L: np.ndarray, sigma: float, window: int, T: float) -> tuple[np.ndarray, np.ndarray]:
    s = cv2.GaussianBlur(L, (0, 0), sigma)
    ys, xs = np.nonzero((s == cv2.dilate(s, np.ones((window, window)))) & (s > T))
    return ys, xs


def star_size(L: np.ndarray, ys: np.ndarray, xs: np.ndarray, r: int = 20) -> float:
    """The light's radius of the brightest stars below saturation (the rms radius of their median stamp, pixels): a
    defocused ring has a large one, whatever its hollow centre does to a measure at half height."""
    H, W = L.shape
    sat = SATURATED * float(L.max())
    keep = (ys > r) & (ys < H - r - 1) & (xs > r) & (xs < W - r - 1) & (L[ys, xs] < sat)
    order = np.argsort(-L[ys[keep], xs[keep]])[:300]
    st = np.array([L[y - r:y + r + 1, x - r:x + r + 1] for y, x in zip(ys[keep][order], xs[keep][order])], np.float32)
    if not len(st):
        return 1.5
    st = st / np.maximum(st.sum((1, 2), keepdims=True), 1e-9)
    m = np.clip(np.median(st, 0), 0, None)
    m = m - np.median(np.r_[m[0], m[-1], m[:, 0], m[:, -1]])
    m = np.clip(m, 0, None)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return float(np.sqrt((m * (yy ** 2 + xx ** 2)).sum() / max(m.sum(), 1e-9) / 2))


def detect(L: np.ndarray, valid: np.ndarray, margin: int = 34):
    """The stars of a stars-alone luminance: (ys, xs, peaks, threshold, size). Peaks are looked for at the scale of the
    stars, so that a defocused ring is one star and not several."""
    sub = L[::4, ::4]
    noise = _noise(sub)
    T = max(DETECT_SIGMAS * noise, float(np.quantile(sub, 0.99999)) * 4e-4)
    ys, xs = _maxima(L, 1.0, 5, T)
    wide_y, wide_x = _maxima(L, 6.0, 31, T * 0.1)                 # centred on a whole star, even a hollow one
    size = star_size(L, wide_y, wide_x)
    if size > 3.0:
        ys, xs = _maxima(L, 1.2 * size, int(2.5 * size) | 1, T * 0.1)
    good = cv2.erode(valid.astype(np.uint8), np.ones((81, 81))) > 0
    H, W = L.shape
    k = (ys > margin) & (ys < H - margin) & (xs > margin) & (xs < W - margin) & good[ys, xs]
    return ys[k], xs[k], L[ys[k], xs[k]], T, size


def _centre(p: np.ndarray, rad: float) -> tuple[float, float]:
    n = p.shape[0] // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    q = np.clip(p - np.median(np.r_[p[0], p[-1], p[:, 0], p[:, -1]]), 0, None) * (np.hypot(yy, xx) < rad)
    t = max(float(q.sum()), 1e-9)
    return float((q * yy).sum() / t), float((q * xx).sum() / t)


def moved(p: np.ndarray, dy: float, dx: float) -> np.ndarray:
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(p, M, (p.shape[1], p.shape[0]), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def measured_shape(L: np.ndarray, ys, xs, peaks, R: int = R_MEASURED, centre_share: float = 0.35) -> np.ndarray | None:
    """The median of the isolated unsaturated stars near the centre of the field, (2R+1, 2R+1), summing to 1."""
    H, W = L.shape
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(yy, xx)
    rc = np.hypot(ys - H / 2, xs - W / 2) / np.hypot(H / 2, W / 2)
    sat = SATURATED * float(L.max())
    done = np.zeros((H, W), bool)
    stamps = []
    for i in np.argsort(-peaks):
        y, x = int(ys[i]), int(xs[i])
        if rc[i] > centre_share or peaks[i] > sat or done[y, x]:
            continue
        done[max(0, y - 40):y + 40, max(0, x - 40):x + 40] = True
        p = np.asarray(L[y - R:y + R + 1, x - R:x + R + 1], np.float32)
        if p[rr > 20].max() > 0.03 * p.max():
            continue
        dy, dx = _centre(p, 6.0)
        st = moved(p, -dy, -dx)
        st = st - np.median(st[rr > 28])
        stamps.append(st / max(float(st[rr < 24].sum()), 1e-9))
        if len(stamps) >= 300:
            break
    if len(stamps) < 5:
        return None
    m = np.median(np.array(stamps), 0)
    m = np.clip(m - np.median(m[rr > 28]), 0, None)
    return (m / m.sum()).astype(np.float32)


def flux_of(s: np.ndarray, y: int, x: int, r: int) -> np.ndarray:
    """The star's light per colour in an aperture of radius ~r, its sky taken on the stamp's edge."""
    st = s[:, y - r:y + r + 1, x - r:x + r + 1]
    bg = np.median(np.c_[st[:, 0], st[:, -1]], axis=1)[:, None, None]
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return np.maximum(((st - bg) * (np.hypot(yy, xx) < r * 0.85)).sum((1, 2)), 0)


def saturated_fluxes(ys, xs, sat_mask, fluxes, gaia, reach: float = 4.0) -> tuple[np.ndarray, int]:
    """The saturated stars' light from the catalogue: the image's zero point on the unsaturated stars matched to
    Gaia, the colour of the unsaturated stars of the nearest BP-RP. ``gaia``: {x, y, g, bp, rp}. Returns the fluxes,
    and how many saturated stars the catalogue gave."""
    from scipy.spatial import cKDTree
    out = fluxes.copy()
    ok = np.isfinite(gaia["g"])                                  # (the offline extract gives no colour)
    gx, gy, g, col = gaia["x"][ok], gaia["y"][ok], gaia["g"][ok], (gaia["bp"] - gaia["rp"])[ok]
    tree = cKDTree(np.c_[gx, gy])
    d, j = tree.query(np.c_[xs, ys], distance_upper_bound=reach)
    hit = np.isfinite(d)
    ref = hit & ~sat_mask & (fluxes[1] > 0)
    if ref.sum() < 10:
        return out, 0
    zp = float(np.median(2.5 * np.log10(fluxes[1][ref]) + g[j[ref]]))
    cols = col[j[ref]]
    ratios = np.stack([fluxes[0][ref] / fluxes[1][ref], fluxes[2][ref] / fluxes[1][ref]])
    n = 0
    for i in np.nonzero(sat_mask & hit)[0]:
        near = np.argsort(np.abs(cols - col[j[i]]))[:30] if np.isfinite(col[j[i]]) else np.arange(len(cols))
        fg = 10 ** ((zp - g[j[i]]) / 2.5)
        out[:, i] = fg * float(np.median(ratios[0, near])), fg, fg * float(np.median(ratios[1, near]))
        n += 1
    return out, n


def _reach(shape: np.ndarray, flux: float, floor: float) -> int:
    """The radius a star of this flux needs: past it, the shape's light is under ``floor`` at every angle."""
    R = shape.shape[-1] // 2
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rings = np.rint(np.hypot(yy, xx)).astype(int).ravel()
    env = np.zeros(R + 2, np.float32)
    np.maximum.at(env, np.minimum(rings, R + 1), shape.ravel())
    over = np.nonzero(env[:R + 1] * flux > floor)[0]
    return int(np.clip(over.max() if over.size else 2, 4, R))


def _centres(L, ys, xs, size):
    r = max(6, int(2.5 * size))
    H, W = L.shape
    return [_centre(L[y - r:y + r + 1, x - r:x + r + 1], r) if r < y < H - r and r < x < W - r else (0.0, 0.0)
            for y, x in zip(ys.tolist(), xs.tolist())]


def _apertures(s, ys, xs, peaks, T, size):
    out = np.zeros((3, len(ys)), np.float32)
    H, W = s.shape[1:]
    for i, (y, x, p) in enumerate(zip(ys.tolist(), xs.tolist(), peaks.tolist())):
        r = min(max(int(3 * size), 14 if p > T * 20 else 7), y, x, H - y - 1, W - x - 1)
        out[:, i] = flux_of(s, y, x, r)
    return out


def from_the_image(s: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """The stars redrawn with the shape measured at the centre of the field; the brightest and the saturated stay
    real, with all their light above the threshold and a very soft edge (their spikes are not in the shape's reach)."""
    L = s.mean(0)
    ys, xs, peaks, T, size = detect(L, valid)
    shape = measured_shape(L, ys, xs, peaks) if len(ys) else None
    if shape is None:
        return None, {"stars_drawn": 0}
    real = (peaks >= np.quantile(peaks, 0.998)) | (peaks > SATURATED * float(L.max()))
    _n, lab = cv2.connectedComponents((L > T * 0.35).astype(np.uint8), connectivity=8)
    ids = np.unique(lab[ys[real], xs[real]])
    region = np.isin(lab, ids[ids > 0]).astype(np.uint8)
    mask = cv2.GaussianBlur(cv2.dilate(region, np.ones((31, 31))).astype(np.float32), (0, 0), 12)
    mask = np.maximum(mask, region.astype(np.float32))
    k = ~real
    fl = _apertures(s, ys[k], xs[k], peaks[k], T, size)
    drawn = draw(*L.shape, ys[k], xs[k], _centres(L, ys[k], xs[k], size), fl, shape, T * 0.05)
    return drawn * (1 - mask) + s * mask, {"stars_drawn": int(k.sum()), "stars_kept": int(real.sum())}


def from_the_instrument(s: np.ndarray, valid: np.ndarray, ins: Instrument, gaia: dict | None) -> tuple[np.ndarray | None, dict]:
    """Every detected star redrawn with the instrument's theoretical shape, the saturated ones from the catalogue."""
    L = s.mean(0)
    ys, xs, peaks, T, size = detect(L, valid)
    if not len(ys):
        return None, {"stars_drawn": 0}
    fl = _apertures(s, ys, xs, peaks, T, size)
    sat = peaks > SATURATED * float(L.max())
    n = 0
    if gaia is not None:
        fl, n = saturated_fluxes(ys, xs, sat, fl, gaia, max(4.0, 2.5 * size))
    drawn = draw(*L.shape, ys, xs, _centres(L, ys, xs, size), fl, theoretical_shape(ins), T * 0.05)
    return drawn, {"stars_drawn": len(ys), "stars_saturated": int(sat.sum()), "stars_from_catalogue": n}


def draw(H: int, W: int, ys, xs, centres, fluxes, shapes, floor: float) -> np.ndarray:
    """The stars on an empty image (3, H, W). ``shapes``: (3, k, k), or (k, k) for one shape in every colour."""
    out = np.zeros((3, H, W), np.float32)
    three = shapes if shapes.ndim == 3 else np.stack([shapes] * 3)
    R = three.shape[-1] // 2
    env = three.max(0)
    for i in np.argsort(fluxes.sum(0)):
        fl = fluxes[:, i]
        y, x = int(ys[i]), int(xs[i])
        r = _reach(env, float(fl.max()), floor)
        r = min(r, y, x, H - y - 1, W - x - 1)
        if r < 2:
            continue
        dy, dx = centres[i]
        for c in range(3):
            st = three[c, R - r:R + r + 1, R - r:R + r + 1]
            st = moved(st, dy, dx) if r <= 64 else st
            out[c, y - r:y + r + 1, x - r:x + r + 1] += fl[c] * st
    return out
