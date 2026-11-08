"""Issue #57: light frames already in colour (JPEG, PNG, TIFF, RGB FITS): read as three planes, calibrated colour by
colour, registered on their luminance, stacked by the Classic method; what cannot go with them is refused."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from astro_gpu.core import cfa as cfa_mod
from astro_gpu.core import stack_classic as sc
from astro_gpu.core.project import Inputs, Project, colour_lights, frame_kind, list_frames, type_warnings
from astro_gpu.core.runner import Pipeline
from astro_gpu.core.stage import StageError
from astro_gpu.io.frames import colour_session, is_colour_frame, read_cfa, read_colour, read_info
from astro_gpu.stages.calibrate import Calibrate
from astro_gpu.stages.inventory import Inventory, same_kind
from astro_gpu.stages.register import Register
from astro_gpu.stages.stack import COLOUR_NO_DRIZZLE, Stack


def jpeg(path: Path, value=(30, 60, 120), size=(40, 60)) -> Path:
    """A JPEG of one colour (R, G, B), (height, width)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.empty((*size, 3), np.uint8)
    a[...] = value[::-1]
    assert cv2.imwrite(str(path), a, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return path


def field(H: int, W: int, seed: int = 3, n: int = 150) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(15, W - 15, n), rng.uniform(15, H - 15, n), 10 ** rng.uniform(1.8, 2.4, n)]


def light(stars, H: int, W: int, dx: float, dy: float, seed: int) -> np.ndarray:
    """A sky of a slightly coloured level with its stars shifted by (dx, dy), Gaussian noise: (H, W, 3), RGB order."""
    y, x = np.mgrid[:H, :W]
    img = np.zeros((H, W), np.float32)
    for sx, sy, f in stars:
        px, py = sx + dx, sy + dy
        ix, iy = int(px), int(py)
        if 8 <= ix < W - 8 and 8 <= iy < H - 8:
            win = (x[iy - 8:iy + 9, ix - 8:ix + 9] - px) ** 2 + (y[iy - 8:iy + 9, ix - 8:ix + 9] - py) ** 2
            img[iy - 8:iy + 9, ix - 8:ix + 9] += f * np.exp(-win / (2 * 1.8 ** 2))
    tint = np.array([1.1, 1.0, 0.8], np.float32)
    rng = np.random.default_rng(seed)
    return np.stack([np.clip((20 + img) * tint[c] + rng.normal(0, 2.5, img.shape), 0, 255) for c in range(3)], 2)


def jpeg_frames(folder: Path, n: int = 6, H: int = 220, W: int = 300) -> list[tuple[float, float]]:
    stars = field(H, W)
    rng = np.random.default_rng(5)
    shifts = []
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        dx, dy = float(rng.uniform(-8, 8)), float(rng.uniform(-8, 8))
        shifts.append((dx, dy))
        a = light(stars, H, W, dx, dy, 100 + i)
        assert cv2.imwrite(str(folder / f"f_{i:03d}.jpg"), a.round().astype(np.uint8)[..., ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    return shifts


# ---------------------------------------------------------------------------------------------- what is a frame
def test_jpeg_png_tiff_are_frames_and_stay_aside_in_a_folder_of_raw_frames(tmp_path):
    assert (frame_kind("a.JPG"), frame_kind("a.png"), frame_kind("a.TIFF")) == ("colour", "colour", "colour")
    assert frame_kind("a.ARW") == "raw" and frame_kind("a.fits") == "fits" and frame_kind("a.mov") == ""
    both = tmp_path / "both"
    both.mkdir()
    (both / "DSC_1.ARW").write_bytes(b"\0" * 8)
    (both / "DSC_1.JPG").write_bytes(b"\0" * 8)
    assert [f.name for f in list_frames([str(both)])] == ["DSC_1.ARW"]    # a camera's JPEGs beside its raws: not frames
    phone = tmp_path / "phone"
    phone.mkdir()
    (phone / "a.jpg").write_bytes(b"\0" * 8)
    (phone / "b.tiff").write_bytes(b"\0" * 8)
    assert [f.name for f in list_frames([str(phone)])] == ["a.jpg", "b.tiff"]
    assert colour_lights(["a.JPG", "b.jpg"]) and not colour_lights(["a.ARW", "b.ARW", "c.JPG"])


def test_a_folder_of_raw_frames_keeps_its_keys_with_the_camera_s_jpegs_beside_them(tmp_path):
    """Their keys do not change: the JPEGs beside the raw files of a folder are not frames of the session."""
    from astro_gpu.core import project2
    from astro_gpu.core import workflow as wfl
    from astro_gpu.core.store import Keys
    raws = tmp_path / "raws"
    raws.mkdir()
    for i in range(3):
        (raws / f"DSC_{i}.ARW").write_bytes(b"\0" * 100)
    p = project2.create(tmp_path / "p", "p")
    p.sessions.append(project2.Session(1, "night", lights=[str(raws)]))
    w = p.add_workflow("W", dict(wfl.TEMPLATES["Full"]))
    before = Keys(p).keys(w.id)
    for i in range(3):
        (raws / f"DSC_{i}.JPG").write_bytes(b"\0" * 50)
    assert Keys(p).keys(w.id) == before


def test_warnings_jpegs_among_raws_and_calibration_of_another_kind():
    assert [w["key"] for w in type_warnings({"lights": ["a.ARW", "b.ARW", "c.JPG"]})] == ["unread"]
    # all JPEG: nothing to say about the lights; a raw among them is mixed (the inventory leaves it out)
    assert type_warnings({"lights": ["a.JPG", "b.JPG"]}) == []
    assert [w["key"] for w in type_warnings({"lights": ["a.JPG", "b.JPG", "c.ARW"]})] == ["mixed"]
    assert [w["key"] for w in type_warnings({"lights": ["a.JPG", "b.JPG"], "darks": ["d.ARW"]})] == ["wrong_kind"]
    assert [w["key"] for w in type_warnings({"lights": ["a.ARW"], "darks": ["d.JPG"]})] == ["wrong_kind"]
    assert [w["key"] for w in type_warnings({"lights": ["a.JPG"], "flats": ["f.tif"]})] == ["other_type"]


# ---------------------------------------------------------------------------------------------- reading
def test_a_jpeg_is_read_as_three_planes_with_its_luminance_and_size(tmp_path):
    f = jpeg(tmp_path / "a.jpg")
    info = read_info(f)
    assert (info.kind, info.cfa, info.width, info.height, info.white, info.black) == ("colour", "RGB", 60, 40, 255.0, 0.0)
    a = read_colour(f)
    assert a.shape == (3, 40, 60) and a.dtype == np.float32
    assert np.allclose(a.mean((1, 2)), (30, 60, 120), atol=3)                  # R, G, B order
    assert np.allclose(read_cfa(f), cfa_mod.luminance(a))
    assert is_colour_frame(f)


def test_16_bit_png_and_tiff_keep_their_depth(tmp_path):
    import tifffile
    a = np.empty((30, 50, 3), np.uint16)
    a[...] = (40000, 20000, 10000)
    png = tmp_path / "a.png"
    assert cv2.imwrite(str(png), a[..., ::-1])
    tif = tmp_path / "b.tif"
    tifffile.imwrite(tif, a)
    for f in (png, tif):
        info = read_info(f)
        assert (info.white, info.width, info.height, info.cfa) == (65535.0, 50, 30, "RGB")
        assert np.allclose(read_colour(f).mean((1, 2)), (40000, 20000, 10000))


def test_an_rgb_fits_frame_is_a_colour_frame_and_a_float_one_is_brought_to_16_bits(tmp_path):
    from astropy.io import fits
    d = np.zeros((3, 30, 50), np.float32)
    d[0], d[1], d[2] = 0.5, 0.25, 0.125
    f = tmp_path / "a.fits"
    fits.writeto(f, d)
    info = read_info(f)
    assert (info.kind, info.cfa, info.width, info.height, info.white) == ("fits", "RGB", 50, 30, 65535.0)
    assert is_colour_frame(f)
    assert np.allclose(read_colour(f).mean((1, 2)), (32767.5, 16383.75, 8191.875))
    bayer = tmp_path / "b.fits"
    h = fits.Header()
    h["BAYERPAT"] = "RGGB"
    fits.writeto(bayer, np.full((30, 50), 1000, np.uint16), h)
    assert not is_colour_frame(bayer) and read_info(bayer).cfa == "RGGB"


def test_a_grey_image_is_refused_with_its_reason(tmp_path):
    f = tmp_path / "grey.png"
    assert cv2.imwrite(str(f), np.zeros((20, 30), np.uint8))
    with pytest.raises(StageError, match="grey.png cannot be used as a colour frame: it has one channel"):
        read_info(f)


def test_a_jpeg_turned_by_its_orientation_tag_keeps_its_real_size(tmp_path):
    from PIL import Image
    f = tmp_path / "a.jpg"
    im = Image.fromarray(np.zeros((40, 60, 3), np.uint8))
    exif = Image.Exif()
    exif[274] = 6
    im.save(f, exif=exif)
    info = read_info(f)
    assert (info.width, info.height) == (40, 60) and read_colour(f).shape == (3, 60, 40)


def test_the_preview_of_a_colour_frame_keeps_its_colours(tmp_path):
    from astro_gpu import api
    a = read_colour(jpeg(tmp_path / "a.jpg", size=(40, 60)))
    v = api._superpixel(a, "RGB", 0.0)
    assert v.shape == (20, 30, 3) and np.allclose(v.mean((0, 1)), (30, 60, 120), atol=3)
    assert cfa_mod.superpixel(a, cfa_mod.RGB).shape == (20, 30, 3)
    assert cfa_mod.is_supported(cfa_mod.RGB) and cfa_mod.is_colour("RGB") and not cfa_mod.is_mono("RGB")


def test_colour_session_says_what_is_stretched(tmp_path):
    from astropy.io import fits
    assert colour_session([jpeg(tmp_path / "a.jpg")]) == {"type": "JPG", "stretched": True}
    d = tmp_path / "f.fits"
    fits.writeto(d, np.zeros((3, 10, 10), np.uint16))
    assert colour_session([d]) == {"type": "FITS", "stretched": False}
    assert colour_session(["x.ARW"]) is None


# ---------------------------------------------------------------------------------------------- refused
def test_calibration_frames_of_the_wrong_kind_are_refused():
    light_f = {"cfa": "RGB", "white": 255.0, "path": "l.jpg"}
    out = {"darks": [{"cfa": "RGGB", "white": 65535.0, "path": "d.ARW"}], "flats": [], "bias": [], "flat_darks": []}
    with pytest.raises(StageError, match=r"the darks \(d.ARW\) are a raw mosaic or monochrome"):
        same_kind(out, "RGB", light_f)
    with pytest.raises(StageError, match=r"the darks \(d.jpg\) are already in colour"):
        same_kind({**out, "darks": [{"cfa": "RGB", "white": 255.0, "path": "d.jpg"}]}, "RGGB",
                  {"cfa": "RGGB", "white": 16383.0, "path": "l.ARW"})
    with pytest.raises(StageError, match="reach 65535 and the lights 255"):
        same_kind({**out, "darks": [{"cfa": "RGB", "white": 65535.0, "path": "d.tif"}]}, "RGB", light_f)
    same_kind({**out, "darks": [{"cfa": "RGB", "white": 255.0, "path": "d.jpg"}]}, "RGB", light_f)      # of the same kind
    same_kind({**out, "darks": [{"cfa": "RGGB", "white": 16383.0, "path": "d.ARW"}]}, "RGGB",
              {"cfa": "RGGB", "white": 16383.0, "path": "l.ARW"})


def test_the_inventory_refuses_a_raw_mosaic_dark_for_jpeg_lights(tmp_path):
    from astropy.io import fits
    (tmp_path / "L").mkdir()
    for i in range(2):
        jpeg(tmp_path / "L" / f"l{i}.jpg")
    h = fits.Header()
    h["BAYERPAT"] = "RGGB"
    (tmp_path / "D").mkdir()
    fits.writeto(tmp_path / "D" / "d.fits", np.zeros((40, 60), np.uint16), h)
    pr = Project.create(tmp_path / "p", "p", Inputs(lights=[str(tmp_path / "L")], darks=[str(tmp_path / "D")]))
    pipe = Pipeline(pr, [Inventory()])
    pipe.run()
    m = json.loads((Path(pr.stage_dir("inventory")) / "manifest.json").read_text())
    assert m["status"] == "failed"
    assert "the darks (d.fits) are a raw mosaic or monochrome, the lights already in colour" in m["error"]


def test_bayer_drizzle_is_refused_with_its_reason():
    assert "Bayer drizzle does not apply to frames already in colour" in COLOUR_NO_DRIZZLE


# ---------------------------------------------------------------------------------------------- the stack
def test_the_classic_stack_of_colour_frames_takes_their_three_planes():
    """Frames already in colour: no debayering; the three planes resampled, rejected and averaged as a debayered
    frame's; a bad sample filled in its own plane; a saturated pixel flagged once."""
    H, W = 48, 64
    frames = [np.stack([np.full((H, W), 30.0 + i), np.full((H, W), 60.0), np.full((H, W), 90.0)]).astype(np.float32)
              for i in range(5)]
    frames[0][1, 10, 10] = np.nan                                         # a bad sample
    ident = np.zeros((2, 10))
    ident[0, 0], ident[0, 1], ident[1, 0], ident[1, 2] = W / 2, W / 2, H / 2, H / 2
    res = sc.stack_classic(lambda i: frames[i], np.repeat(ident[None], 5, 0), used=range(5), ch_of_k=[0, 1, 2], H=H, W=W,
                           device="cpu", colour=True, sat_level=np.full((H, W), 250.0, np.float32))
    assert res.full.shape == (3, H, W)
    assert np.allclose(res.full[:, 24, 32], (32.0, 60.0, 90.0), atol=0.5)
    assert np.allclose(res.full[1, 10, 10], 60.0, atol=0.5)
    assert res.frames[0]["bad_filled"] == 1 and res.frames[0]["cfa_balance"] is None


def test_the_whole_chain_on_jpeg_frames(tmp_path):
    """A real run on the CPU: 6 JPEG frames with shifted stars, through inventory, calibration, registration and the
    stack: each star where it is in the reference frame, the stack colour by colour, the noise down."""
    lights = tmp_path / "L"
    shifts = jpeg_frames(lights)
    pr = Project.create(tmp_path / "p", "p", Inputs(lights=[str(lights)]))
    pr.config.devices = {"main": "cpu"}
    pr.config.params = {"calibrate": {"workers": 1}, "register": {"workers": 1}}
    pr.save()
    Pipeline(pr, [Inventory(), Calibrate(), Register(), Stack()]).run()
    d = {s: Path(pr.stage_dir(s)) for s in ("inventory", "calibrate", "register", "stack")}
    for k, q in d.items():
        m = json.loads((q / "manifest.json").read_text())
        assert m["status"] == "done", (k, m.get("error"))
    assert json.loads((d["inventory"] / "manifest.json").read_text())["metrics"]["colour_frames"] is True
    cache = json.loads((d["calibrate"] / "cache.json").read_text())
    assert cache["cfa"] == cfa_mod.MONO and cache["colour"] and Path(cache["rgb_path"]).is_file()
    assert np.load(d["calibrate"] / "frame_stats.npy").shape == (6, 4, 2)
    reg = np.load(d["register"] / "reg.npz")
    assert np.isfinite(reg["rms"]).all() and float(np.median(reg["rms"])) < 0.6
    ref = int(reg["ref"])
    meta = json.loads((d["stack"] / "drz_meta.json").read_text())
    assert meta["method"] == "classic" and meta["scale"] == 1.0
    drz = np.load(d["stack"] / "drz_full.npy")
    assert drz.shape == (3, 220, 300)
    stars = field(220, 300)
    errs = []
    for sx, sy, _flux in stars[stars[:, 2] > 120]:
        ex, ey = sx + shifts[ref][0], sy + shifts[ref][1]
        ix, iy = int(round(ex)), int(round(ey))
        if not (12 <= ix < 288 and 12 <= iy < 208):
            continue
        win = drz[1][iy - 6:iy + 7, ix - 6:ix + 7]
        w = np.clip(win - np.median(drz[1][iy - 11:iy + 12, ix - 11:ix + 12]), 0, None)
        if w.sum() <= 0:
            continue
        yy, xx = np.mgrid[iy - 6:iy + 7, ix - 6:ix + 7]
        errs.append(((w * xx).sum() / w.sum() - ex, (w * yy).sum() / w.sum() - ey))
    errs = np.array(errs)
    assert len(errs) > 10
    # the centroids of a crowded field are noisy one by one (a neighbour's wing in the window): their median is the
    # registration's error (the 10-frame, 1000 x 700 bench gave 0.006 and -0.05 px of mean error over 230 stars)
    assert np.abs(np.median(errs, axis=0)).max() < 0.2
    sky = np.median(drz, axis=(1, 2))
    assert 1.0 < sky[0] / sky[1] < 1.2 and 0.7 < sky[2] / sky[1] < 0.9      # the colours kept as the frames have them
    one = read_colour(sorted(lights.glob("*.jpg"))[0])[1]
    sample = (slice(20, 200, 2), slice(20, 280, 2))
    from astro_gpu.core.imgutil import robust_sigma
    assert robust_sigma(drz[1][sample])[1] < 0.6 * robust_sigma(one[sample])[1]
    m = json.loads((d["stack"] / "manifest.json").read_text())
    assert m["params"]["method"] == "classic"


def test_a_drizzle_chosen_by_hand_on_colour_frames_fails_with_its_reason(tmp_path):
    lights = tmp_path / "L"
    jpeg_frames(lights, n=4, H=200, W=260)
    pr = Project.create(tmp_path / "p", "p", Inputs(lights=[str(lights)]))
    pr.config.devices = {"main": "cpu"}
    pr.config.params = {"stack": {"method": "drizzle"}, "calibrate": {"workers": 1}, "register": {"workers": 1}}
    pr.save()
    Pipeline(pr, [Inventory(), Calibrate(), Register(), Stack()]).run()
    m = json.loads((Path(pr.stage_dir("stack")) / "manifest.json").read_text())
    assert m["status"] == "failed" and COLOUR_NO_DRIZZLE in m["error"]


def test_calibrate_divides_the_flat_colour_by_colour(tmp_path):
    """A colour flat with its own vignetting per colour: each colour of the light frames back to a flat level."""
    L, F, D = tmp_path / "L", tmp_path / "F", tmp_path / "D"
    for q in (L, F, D):
        q.mkdir()
    H, W = 100, 120
    y, x = np.mgrid[:H, :W]
    v = 1 - 0.3 * ((x - W / 2) ** 2 + (y - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    tint = np.array([1.0, 0.8, 0.6], np.float32)
    rng = np.random.default_rng(7)
    import tifffile
    for i in range(4):
        tifffile.imwrite(L / f"l{i}.tif", np.stack([np.clip(1000 * tint[c] * v + rng.normal(0, 2, v.shape) + 100, 0, 65535)
                                                    for c in range(3)], 2).astype(np.uint16))
        tifffile.imwrite(F / f"f{i}.tif", np.stack([np.clip(30000 * tint[c] * v + 100, 0, 65535)
                                                    for c in range(3)], 2).astype(np.uint16))
        tifffile.imwrite(D / f"d{i}.tif", np.full((H, W, 3), 100, np.uint16))
    pr = Project.create(tmp_path / "p", "p", Inputs(lights=[str(L)], flats=[str(F)], darks=[str(D)]))
    pr.config.devices = {"main": "cpu"}
    pr.config.params = {"calibrate": {"workers": 1}}
    pr.save()
    Pipeline(pr, [Inventory(), Calibrate()]).run()
    d = Path(pr.stage_dir("calibrate"))
    m = json.loads((d / "manifest.json").read_text())
    assert m["status"] == "done", m.get("error")
    cache = json.loads((d / "cache.json").read_text())
    rgb = np.memmap(cache["rgb_path"], dtype=np.float16, mode="r", shape=(4, 3, H, W))
    fr = np.asarray(rgb[0], np.float32) / cache["scale"]
    for c in range(3):
        centre, corner = np.median(fr[c, 40:60, 50:70]), np.median(fr[c, 2:12, 2:12])
        assert abs(centre / corner - 1) < 0.02, c                              # the vignetting taken out
    assert 0.55 < np.median(fr[2]) / np.median(fr[0]) < 0.65                   # the colours kept
    assert np.load(d / "master_flat_rgb.npy").shape == (3, H, W) and np.load(d / "master_dark.npy").shape == (H, W)
