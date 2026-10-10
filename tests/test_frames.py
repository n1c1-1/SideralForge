"""Frame readers: FITS Bayer pattern handling (offsets, row order) and metadata."""
from __future__ import annotations

import numpy as np
from astropy.io import fits

from astro_gpu.io.frames import cfa_channel_map, read_cfa, read_info


def _write(path, data, **hdr):
    h = fits.Header()
    for k, v in hdr.items():
        h[k] = v
    fits.PrimaryHDU(data=data, header=h).writeto(path, overwrite=True)


def test_fits_bayer_offsets_and_roworder(tmp_path):
    a = np.arange(24, dtype=np.uint16).reshape(4, 6)
    p = tmp_path / "f.fits"
    _write(p, a, BAYERPAT="RGGB", XBAYROFF=1, YBAYROFF=0, ROWORDER="BOTTOM-UP", EXPTIME=120.0, GAIN=100,
           XPIXSZ=3.76, FOCALLEN=530.0, INSTRUME="ZWO ASI2600MC Pro", OBJCTRA="00 03 30", OBJCTDEC="+67 15 00",
           **{"DATE-OBS": "2026-08-17T22:10:00"})
    info = read_info(p)
    assert info.cfa == "GRBG"                          # RGGB shifted by one column
    assert info.exposure == 120.0 and info.pixel_um == 3.76 and info.focal_mm == 530.0
    assert abs(info.ra - 0.875) < 1e-3 and abs(info.dec - 67.25) < 1e-3
    img = read_cfa(p)
    assert img.shape == (4, 6) and img[0, 0] == a[-1, 0]   # bottom-up rows flipped to top-down


def test_cfa_channel_map():
    assert cfa_channel_map("RGGB") == [0, 1, 1, 2]
    assert cfa_channel_map("BGGR") == [2, 1, 1, 0]
    assert cfa_channel_map("GRBG") == [1, 0, 2, 1]


def test_non_bayer_fits_has_no_pattern(tmp_path):
    p = tmp_path / "rgb.fits"
    _write(p, np.zeros((3, 8, 8), np.float32))
    assert read_info(p).cfa == ""
