"""A small sky the chain can solve without the network: a synthetic image of a field and the offline Gaia extracts of
that field (star positions and their sampled XP spectra), in the format core/sirilcat.py reads. The test chains that
need the catalogue (tests/test_no_download_in_run.py) use it: with it, a run computes Astrometry, Catalogues and Color
from the disk alone; without it, the run's pre-check says what is missing.

``field(folder)`` writes ``sky.fits`` (3 x 700 x 700, linear, 3"/px around RA 38, Dec 61) and the extracts in
``folder/extracts`` (``ASTRO_GPU_GAIA_EXTRACTS``), and returns their paths and the target."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from astro_gpu.core import sirilcat

RA0, DEC0, SCALE = 38.0, 61.0, 3.0          # the field's centre (deg) and its scale ("/px)
W = H = 700
#: the target that makes the scale known (3 um pixels at 206.265 mm: 3"/px), as a project entered with its focal length
TARGET = {"name": "synthetic field", "ra": RA0, "dec": DEC0, "focal_mm": 206.265, "pixel_um": 3.0}


def _write(path: Path, kind: int, level: int, stars: dict, chunk: tuple[int, int] | None = None) -> None:
    """An extract file (HEALpixel Catalog Format 1.0.0) of the given stars (ra, dec, g; flux and e for spectra)."""
    import astropy.units as u
    from astropy_healpix import HEALPix
    hp = HEALPix(nside=2 ** level, order="nested")
    pix = np.asarray(hp.lonlat_to_healpix(stars["ra"] * u.deg, stars["dec"] * u.deg), np.int64)
    if chunk:
        first, n = chunk[1] * 4 ** (level - chunk[0]), 4 ** (level - chunk[0])
    else:
        first, n = 0, 12 * 4 ** level
    o = np.argsort(pix, kind="stable")
    rec = np.zeros(len(o), sirilcat.ASTRO_DT if kind == 1 else sirilcat.XP_DT)
    rec["ra"] = np.round(stars["ra"][o] / sirilcat.SCALE).astype(np.int32)
    rec["dec"] = np.round(stars["dec"][o] / sirilcat.SCALE).astype(np.int32)
    rec["g"] = np.round(stars["g"][o] * 1000).astype(np.int16)
    if kind == 2:
        rec["e"] = stars["e"][o]
        rec["flux"] = (stars["flux"][o] * 10.0 ** stars["e"][o][:, None]).astype(np.float16)
    idx = np.cumsum(np.bincount(pix[o] - first, minlength=n)).astype("<u4")
    h = bytearray(128)
    h[:48] = b"Siril test extract".ljust(48, b"\0")
    h[48:53] = bytes([3, level, kind, 1 if chunk else 0, chunk[0] if chunk else 0])
    if chunk:
        h[53:57] = int(chunk[1]).to_bytes(4, "little")
        h[57:61] = int(first).to_bytes(4, "little")
        h[61:65] = int(first + n - 1).to_bytes(4, "little")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(h) + idx.tobytes() + rec.tobytes())


def field(folder: Path, spectra: bool = True) -> dict:
    """The image and the extracts of the field (``spectra``: the spectra extract too). Returns {"image", "extracts",
    "target"}."""
    import astropy.units as u
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy_healpix import HEALPix
    rng = np.random.default_rng(61)
    n = 4000
    r = 1.3 * np.sqrt(rng.uniform(0, 1, n))                        # stars over 1.3 deg around the centre
    a = rng.uniform(0, 2 * np.pi, n)
    dec = DEC0 + r * np.sin(a)
    ra = RA0 + r * np.cos(a) / np.cos(np.radians(dec))
    g = 7.0 + 9.0 * rng.uniform(0, 1, n) ** 0.8                     # more faint stars than bright ones, G 7 to 16
    # the spectra extract is cut in level-1 chunks: the field's chunk only
    chunk = int(HEALPix(nside=2, order="nested").lonlat_to_healpix(RA0 * u.deg, DEC0 * u.deg))
    inside = np.asarray(HEALPix(nside=2, order="nested").lonlat_to_healpix(ra * u.deg, dec * u.deg)) == chunk
    ra, dec, g = ra[inside], dec[inside], g[inside]
    stars = {"ra": ra, "dec": dec, "g": g}
    ext = folder / "extracts"
    _write(ext / "siril_cat_healpix8_astro.dat", 1, 8, stars)
    if spectra:
        lam = sirilcat.XP_WAVELENGTHS
        temp = rng.uniform(3500, 9000, len(ra))                     # black bodies: colours from red to blue
        e = np.full(len(ra), 16, np.int8)
        with np.errstate(over="ignore"):
            bb = 1.0 / (lam[None, :] ** 5 * (np.exp(1.4388e7 / (lam[None, :] * temp[:, None])) - 1.0))
        bb /= bb[:, (lam > 540) & (lam < 560)].mean(1, keepdims=True)
        flux = bb * (10.0 ** (-0.4 * (g[:, None] - 10.0))) * 1e-16
        _write(ext / "siril_cat1_healpix8_xpsamp" / f"siril_cat1_healpix8_xpsamp_{chunk}.dat", 2, 8,
               {**stars, "flux": flux, "e": e}, chunk=(1, chunk))
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0 + 0.01, DEC0 - 0.005]                        # a little off the target, as a pointing
    w.wcs.crpix = [W / 2, H / 2]
    w.wcs.cdelt = [-SCALE / 3600, SCALE / 3600]
    w.wcs.pc = [[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]
    x, y = w.world_to_pixel_values(ra, dec)
    keep = (x > -10) & (x < W + 10) & (y > -10) & (y < H + 10)
    img = np.full((3, H, W), 0.05, np.float32) + rng.normal(0, 0.0002, (3, H, W)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for xi, yi, gi in zip(x[keep], y[keep], g[keep]):
        x0, y0 = int(round(xi)), int(round(yi))
        sl = np.s_[max(0, y0 - 9):y0 + 10, max(0, x0 - 9):x0 + 10]
        peak = 0.004 * 10 ** (-0.4 * (gi - 14.0))                 # G 8 just under the top of the scale
        prof = peak * np.exp(-((xx[sl] - xi) ** 2 + (yy[sl] - yi) ** 2) / (2 * 1.4 ** 2))
        for c, k in enumerate((0.9, 1.0, 0.8)):
            img[c][sl] += (k * prof).astype(np.float32)
    img = np.minimum(img, 1.0)[:, ::-1]                             # top row first, as a stack: the parity the
    #                                                                 solver tries first
    image = folder / "sky.fits"
    fits.writeto(image, img, overwrite=True)
    return {"image": image, "extracts": ext, "target": dict(TARGET)}
