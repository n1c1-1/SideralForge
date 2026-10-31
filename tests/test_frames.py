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


def test_compressed_fits_are_frames(tmp_path):
    from astro_gpu.core.project import frame_kind, list_frames
    a = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 7)
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(a, fits.Header({"EXPTIME": 300.0, "BAYERPAT": "RGGB"}))]
                 ).writeto(tmp_path / "a.fits.fz")
    fits.PrimaryHDU(a, fits.Header({"EXPTIME": 12.0})).writeto(tmp_path / "b.fit")
    import gzip
    (tmp_path / "c.fit.gz").write_bytes(gzip.compress((tmp_path / "b.fit").read_bytes()))
    (tmp_path / "notes.txt.gz").write_bytes(b"")
    assert [p.name for p in list_frames([str(tmp_path)])] == ["a.fits.fz", "b.fit", "c.fit.gz"]
    assert frame_kind("x.NEF") == "raw" and frame_kind("x.tif") == ""
    i = read_info(tmp_path / "a.fits.fz")            # the image and its header are in the first extension
    assert (i.width, i.height, i.exposure, i.cfa) == (8, 8, 300.0, "RGGB")
    assert np.array_equal(read_cfa(tmp_path / "a.fits.fz"), a) and np.array_equal(read_cfa(tmp_path / "c.fit.gz"), a)
    assert read_info(tmp_path / "c.fit.gz").exposure == 12.0


def test_exposure_in_milliseconds_and_other_key_names(tmp_path):
    p = tmp_path / "stellina.fits"
    h = fits.Header()
    h["BAYERPAT"] = "BGGR"
    h["EXPOSURE"] = (10000, "[ms] Total Exposure Time")      # as a Vaonis Stellina writes it
    h["PIXSZ"], h["FOCAL"], h["TEMP"] = 2.4, 400, 20.6
    fits.PrimaryHDU(np.zeros((4, 4), np.uint16), h).writeto(p)
    i = read_info(p)
    assert (i.exposure, i.pixel_um, i.focal_mm, i.temperature) == (10.0, 2.4, 400.0, 20.6)


def _tiff(path, tags):
    from PIL import Image
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    ifd = ImageFileDirectory_v2()
    for k, v in tags.items():
        ifd[k] = v
    Image.new("L", (1, 1)).save(path, tiffinfo=ifd)
    return path.read_bytes()


def test_cr3_and_raf_metadata(tmp_path):
    """Canon CR3: the main directory in an ISO-media box CMT1, the EXIF directory in CMT2; Fujifilm RAF: the EXIF in
    the JPEG preview, whose offset and length are at bytes 84-91."""
    from astro_gpu.io.frames import exif_tags
    cmt1 = _tiff(tmp_path / "1.tif", {271: "Canon", 272: "Canon EOS Ra"})
    from PIL.TiffImagePlugin import IFDRational
    cmt2 = _tiff(tmp_path / "2.tif", {33434: IFDRational(901, 10), 34855: 1600, 36867: "2022:09:19 20:56:40"})
    box = b"".join((8 + len(b)).to_bytes(4, "big") + name + b for name, b in ((b"CMT1", cmt1), (b"CMT2", cmt2)))
    p = tmp_path / "x.cr3"
    p.write_bytes(b"\0\0\0\x18ftypcrx " + b"\0" * 12 + box)
    t = exif_tags(p)
    assert str(t["Image Model"]) == "Canon EOS Ra" and str(t["EXIF ExposureTime"]) == "901/10"
    assert str(t["EXIF ISOSpeedRatings"]) == "1600" and str(t["EXIF DateTimeOriginal"]) == "2022:09:19 20:56:40"
    from PIL import Image
    exif = Image.Exif()
    exif[272] = "GFX 50R"
    exif.get_ifd(0x8769)[33434] = IFDRational(300, 1)
    jpg = tmp_path / "prev.jpg"
    Image.new("RGB", (8, 8)).save(jpg, exif=exif)
    raf = bytearray(b"FUJIFILMCCD-RAW " + b"\0" * 76)
    raf[84:88], raf[88:92] = (100).to_bytes(4, "big"), jpg.stat().st_size.to_bytes(4, "big")
    raf += b"\0" * (100 - len(raf)) + jpg.read_bytes()
    (tmp_path / "x.raf").write_bytes(bytes(raf))
    t = exif_tags(tmp_path / "x.raf")
    assert str(t["Image Model"]) == "GFX 50R" and str(t["EXIF ExposureTime"]) == "300"
