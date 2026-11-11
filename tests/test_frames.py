"""Frame readers: FITS Bayer pattern handling (offsets, row order) and metadata."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
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


def test_a_header_without_coordinates_does_not_load_the_sky_coordinates(tmp_path):
    """#115: the reader of a FITS header imported astropy's coordinates for any frame (half a second at the first opening
    of a project of FITS frames); only a header with sexagesimal coordinates needs them (a process of its own: counted
    from a clean start)."""
    import subprocess
    import sys
    p = tmp_path / "f.fits"
    _write(p, np.zeros((4, 6), np.uint16), BAYERPAT="RGGB", EXPTIME=60.0)
    code = ("import sys; from astro_gpu.io.frames import read_info; i = read_info(sys.argv[1]); "
            "print(i.cfa, i.ra, 'astropy.coordinates' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code, str(p)], capture_output=True, text=True, timeout=120,
                         cwd=str(Path(__file__).resolve().parents[1]))
    assert out.stdout.split() == ["RGGB", "None", "False"], out.stderr[-2000:]


def test_the_useful_area_of_the_header_is_the_frame(tmp_path):
    """Issue #46: a FITS frame read through its TRIMSEC: its size, its pixels, its Bayer pattern from the area's first
    pixel. TRIMSEC decides over DATASEC, which a frame can give as the whole image."""
    a = np.arange(48, dtype=np.uint16).reshape(6, 8)
    p = tmp_path / "f.fits"
    _write(p, a, BAYERPAT="RGGB", TRIMSEC="[2:7,2:5]", DATASEC="[1:8,1:6]", BIASSEC="[8:8,1:6]")
    info = read_info(p)
    assert (info.width, info.height) == (6, 4)
    assert info.cfa == "BGGR"                          # one column and one row in: the pattern's other corner
    img = read_cfa(p)
    assert img.shape == (4, 6) and np.array_equal(img, a[1:5, 1:7])


def test_datasec_counts_without_trimsec_and_the_rows_stay_the_files(tmp_path):
    """The section is in the file's own rows (1 is its first row): a bottom-up frame is cut first, then turned; the
    pattern follows the row the image then starts at."""
    a = np.arange(48, dtype=np.uint16).reshape(6, 8)
    p = tmp_path / "f.fits"
    _write(p, a, BAYERPAT="RGGB", DATASEC="[1:8,1:3]", ROWORDER="BOTTOM-UP")
    info = read_info(p)
    assert (info.width, info.height) == (8, 3) and info.cfa == "GBRG"      # 3 rows left out above, once turned
    img = read_cfa(p)
    assert img.shape == (3, 8) and np.array_equal(img, a[:3][::-1])


@pytest.mark.parametrize("hdr", [{}, {"TRIMSEC": "[1:8,1:6]"}, {"TRIMSEC": "N/A"}, {"DATASEC": "[0:8,1:6]"},
                                 {"TRIMSEC": "[1:9,1:6]"}, {"TRIMSEC": "[5:2,1:6]"}])
def test_a_frame_without_a_usable_section_is_read_whole(tmp_path, hdr):
    a = np.arange(48, dtype=np.uint16).reshape(6, 8)
    p = tmp_path / "f.fits"
    _write(p, a, BAYERPAT="RGGB", **hdr)
    info = read_info(p)
    assert (info.width, info.height, info.cfa) == (8, 6, "RGGB")
    assert np.array_equal(read_cfa(p), a)


def test_the_inventory_works_on_the_useful_area_and_says_so(tmp_path):
    from types import SimpleNamespace

    from astro_gpu.core.project import Inputs
    from astro_gpu.stages.inventory import Inventory, InventoryParams
    d = tmp_path / "lights"
    d.mkdir()
    for i in range(2):
        _write(d / f"l{i}.fits", np.full((6, 8), 100, np.uint16), EXPTIME=20.0, TRIMSEC="[2:7,2:5]")
    logs, saved = [], {}
    ctx = SimpleNamespace(project=SimpleNamespace(config=SimpleNamespace(inputs=Inputs(lights=[str(d)]))),
                          progress=lambda *a, **k: None, log=lambda m, level="info": logs.append(m),
                          preview=lambda *a, **k: None, save_json=lambda n, o: saved.update({n: o}), metrics={})
    ctx.metric = lambda k, v: ctx.metrics.__setitem__(k, v)
    Inventory().run(ctx, InventoryParams())
    assert (ctx.metrics["width"], ctx.metrics["height"]) == (6, 4)
    assert "useful area from the frames' header: 6x4 of 8x6 pixels, the overscan left out" in logs


def _inventory_of(d):
    from types import SimpleNamespace

    from astro_gpu.core.project import Inputs
    from astro_gpu.stages.inventory import Inventory, InventoryParams
    logs, saved = [], {}
    ctx = SimpleNamespace(project=SimpleNamespace(config=SimpleNamespace(inputs=Inputs(lights=[str(d)]))),
                          progress=lambda *a, **k: None, log=lambda m, level="info": logs.append(m),
                          preview=lambda *a, **k: None, save_json=lambda n, o: saved.update({n: o}), metrics={})
    ctx.metric = lambda k, v: ctx.metrics.__setitem__(k, v)
    Inventory().run(ctx, InventoryParams())
    return ctx.metrics, logs


def test_an_empty_or_damaged_frame_is_left_out_by_its_name(tmp_path):
    """The night test pass: one empty or damaged file among the lights stopped the whole run, without its name."""
    from astro_gpu.core.stage import StageError
    d = tmp_path / "lights"
    d.mkdir()
    (d / "a_vide.ARW").write_bytes(b"")
    (d / "b_abime.fits").write_bytes(b"" * 5000)
    for i in range(2):
        _write(d / f"l{i}.fits", np.full((6, 8), 100, np.uint16), EXPTIME=20.0)
    metrics, logs = _inventory_of(d)
    assert metrics["n_lights"] == 2
    assert "2 lights cannot be read and are left out (damaged or empty files?): a_vide.ARW, b_abime.fits" in logs
    for f in d.glob("l*.fits"):
        f.unlink()
    with pytest.raises(StageError, match="no light frame can be read .*a_vide.ARW, b_abime.fits"):
        _inventory_of(d)


def test_cfa_channel_map():
    assert cfa_channel_map("RGGB") == [0, 1, 1, 2]
    assert cfa_channel_map("BGGR") == [2, 1, 1, 0]
    assert cfa_channel_map("GRBG") == [1, 0, 2, 1]


def test_non_bayer_fits_has_no_pattern(tmp_path):
    p = tmp_path / "mono.fits"
    _write(p, np.zeros((8, 8), np.float32))
    assert read_info(p).cfa == ""


def test_an_rgb_fits_has_the_pattern_of_a_frame_already_in_colour(tmp_path):
    p = tmp_path / "rgb.fits"                          # issue #57: three planes, not a monochrome frame
    _write(p, np.zeros((3, 8, 8), np.float32))
    assert read_info(p).cfa == "RGB"


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
    assert frame_kind("x.NEF") == "raw" and frame_kind("x.mov") == ""  # (a .tif is a colour frame, issue #57)
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
