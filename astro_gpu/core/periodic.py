"""The colour mosaic's periodic pattern taken off a stack (issue #28).

A one-shot-colour stack made by a Bayer drizzle, from frames that move little on the sensor, keeps a faint pattern of
the colour mosaic: each colour was sampled on its own lattice of the sensor, every 2 sensor pixels, and the small
differences between the photosites of a colour (flat and dark residuals, the two greens) repeat on that lattice. In
the stack it is a set of sharp peaks of the spectrum at the lattice's frequencies (Rosette: 220 to 350 times the
spectrum around them), invisible under the noise until the Denoise block takes the noise off. The star-removal
network adds a grid of its own (the checkerboard of the networks that reduce then enlarge an image: A. Odena, V.
Dumoulin, C. Olah, "Deconvolution and Checkerboard Artifacts", Distill, 2016), at the multiples of 1/16 px^-1 (2, 4,
2.7 px...), whatever its input.

It is taken off as periodic noise is: by notch filters in the frequency domain (R. C. Gonzalez, R. E. Woods, "Digital
Image Processing", 4th ed., 2018, sect. 5.4, notch reject filters), placed only at the mosaic's frequencies (the
multiples of 1 / (2 x the drizzle scale) per output pixel) and only where a peak stands MIN_RATIO times above the
spectrum around it, each a Gaussian of a few frequency bins: the stars and the nebula have no peak there, a broad
spectrum, of which the notches take a negligible share.
"""
from __future__ import annotations

import itertools

import numpy as np

#: a peak is notched when its power stands this many times above the median of the spectrum around it
MIN_RATIO = 10.0
#: the notch: a Gaussian of this sigma (cycles per pixel: the pattern's strength varies over the field, over about
#: 1 / NOTCH_WIDTH pixels, which spreads its peak), and the ring that measures the spectrum around a peak (cycles per
#: pixel)
NOTCH_WIDTH, RING = 0.004, (0.008, 0.02)


def lattice_frequencies(period: float) -> list[tuple[float, float]]:
    """The frequencies (cycles per pixel, fx, fy, folded in -0.5..0.5) of a pattern repeating every ``period`` pixels
    along both axes: the multiples of 1 / period, without (0, 0)."""
    f0 = 1.0 / max(period, 1e-6)
    ks = sorted({round(((k * f0 + 0.5) % 1.0) - 0.5, 6) for k in range(0, int(np.ceil(1.0 / f0)) + 1)})
    return [(fx, fy) for fx, fy in itertools.product(ks, ks) if (fx, fy) != (0.0, 0.0)]


def mosaic_frequencies(scale: float, period: int = 2) -> list[tuple[float, float]]:
    """The frequencies of a colour mosaic's lattice in a drizzle at ``scale``: every ``period`` x scale output pixels
    (period 2: a Bayer mosaic; 6: an X-Trans one)."""
    return lattice_frequencies(float(period) * scale)


def remove_mosaic_pattern(img: np.ndarray, scale: float, valid: np.ndarray | None = None,
                          period: int = 2) -> tuple[np.ndarray, list]:
    """The colour mosaic's pattern of a drizzle at ``scale`` notched (``remove_periodic``)."""
    return remove_periodic(img, mosaic_frequencies(scale, period), valid)


def remove_periodic(img: np.ndarray, freqs: list[tuple[float, float]], valid: np.ndarray | None = None
                    ) -> tuple[np.ndarray, list]:
    """``img`` (c, h, w) with its peaks at ``freqs`` notched, each channel apart; returns (image, [(channel, fx, fy,
    ratio)] of the peaks taken). The peaks are found on the image with its stars clipped (the 99th percentile) and
    its uncovered pixels at the median, the notches applied to the image itself."""
    from scipy import fft
    c, h, w = img.shape
    fy = fft.fftfreq(h)[:, None]
    fx = fft.rfftfreq(w)[None, :]
    out = np.empty_like(img, dtype=np.float32)
    taken = []
    for k in range(c):
        done: set[tuple[int, int]] = set()                        # a frequency and its mirror are one point here
        a = img[k].astype(np.float32)
        ok = valid if valid is not None else np.isfinite(a)
        med = float(np.median(a[ok][::17])) if ok.any() else 0.0
        hi = float(np.percentile(a[ok][::17], 99)) if ok.any() else 1.0
        probe = np.where(ok, np.clip(a, None, hi), med) - med
        P = np.abs(fft.rfft2(probe, workers=-1)) ** 2
        notch = np.ones(P.shape, np.float32)
        for gx, gy in freqs:
            for sx in ((gx,) if gx >= 0 else (-gx,)):            # the real FFT keeps fx >= 0 (its mirror is implied)
                sy = gy if gx >= 0 else -gy
                iy, ix = int(round(sy * h)) % h, int(round(sx * w))
                if ix >= P.shape[1] or (iy, ix) in done:
                    continue
                done.add((iy, ix))
                ry, rx = int(np.ceil(RING[1] * h)), int(np.ceil(RING[1] * w))
                y0, y1 = max(0, iy - ry), min(h, iy + ry + 1)
                x0, x1 = max(0, ix - rx), min(P.shape[1], ix + rx + 1)
                box = P[y0:y1, x0:x1]
                yy, xx = np.mgrid[y0:y1, x0:x1]
                rr = np.hypot((yy - iy) / h, (xx - ix) / w)
                ring = box[(rr >= RING[0]) & (rr <= RING[1])]
                core = box[rr <= RING[0] / 2]
                peak = float(core.sum() / max(core.size, 1)) if core.size else float(P[iy, ix])
                ratio = peak / max(float(np.median(ring)), 1e-30) if ring.size else 0.0
                if ratio < MIN_RATIO:
                    continue
                dy = (fy - sy + 0.5) % 1.0 - 0.5                    # the distance in frequency, the lattice wrapped
                g = np.exp(-0.5 * ((dy / NOTCH_WIDTH) ** 2 + ((fx - sx) / NOTCH_WIDTH) ** 2))
                notch *= (1.0 - g).astype(np.float32)
                taken.append((k, float(sx), float(sy), round(ratio, 1)))
        if (notch < 1.0).any():
            spec = fft.rfft2(np.where(ok, a, med) - med, workers=-1)
            a = (fft.irfft2(spec * notch, s=(h, w), workers=-1) + med).astype(np.float32)
            a[~ok] = img[k][~ok]
        out[k] = a
    return out, taken
