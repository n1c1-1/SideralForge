"""Reading raw colour (Bayer and X-Trans) frames from camera RAW files and astro-camera FITS files, and the frames that
are already in colour (issue #57: JPEG, PNG, TIFF, RGB FITS).

Everything downstream works on the undebayered mosaic in ADU, top row first, with a colour pattern string describing
its top-left block: ``"RGGB"`` (2 x 2 Bayer), or 36 letters for a 6 x 6 X-Trans sensor (core/cfa.py).

A frame already in colour has no mosaic: the pattern is ``cfa.RGB``, ``read_colour`` gives its three planes, and
``read_cfa`` its luminance, so that whatever only needs a frame's level or stars reads it as any other. The method is
DeepSkyStacker's for RGB pictures, as its documentation gives it: three planes throughout, the master flat, dark and
offset included (FAQ, http://deepskystacker.free.fr/english/faq.htm : its raw modes work "with only one monochrome 16
bits channel instead of 3 RGB channels [...] flat, dark and offset frames are computed and internally stored with only
one plane instead of 3"), the stars found on the luminance (technical details,
http://deepskystacker.free.fr/english/technical.htm : "a Gaussian curve to the luminance"). Its values are taken as
the file holds them, with no gamma curve taken off: neither that documentation nor Siril's preprocessing guide
describes such a step for JPEG lights, and the curve a camera applied is not known (not the sRGB curve alone: its tone
curve, contrast and white balance are in it). So the frames stay non-linear, which the session says plainly
(client/frames.py), and a dark or a flat only approximately corrects them.
"""
from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..core import cfa as cfa_mod
from ..core.project import frame_kind

#: pixel pitch (micrometres) of common sensors, used to guess the plate scale when the file does not say
PIXEL_PITCH_UM = {
    "ILCE-7S": 8.40, "ILCE-7SM2": 8.40, "ILCE-7SM3": 8.40, "ILCE-7M3": 5.95, "ILCE-7M4": 5.12, "ILCE-7RM3": 4.51,
    "ILCE-7RM4": 3.76, "ILCE-6000": 3.92, "ILCE-6400": 3.92, "Canon EOS 6D": 6.55, "Canon EOS 6D Mark II": 5.67,
    "Canon EOS R": 5.36, "Canon EOS Ra": 5.36, "Canon EOS R6": 6.56, "Canon EOS 5D Mark III": 6.25,
    "Canon EOS 5D Mark IV": 5.36, "Canon EOS 600D": 4.30, "Canon EOS 2000D": 3.72, "Canon EOS 77D": 3.72,
    "NIKON D810A": 4.88, "NIKON D810": 4.88, "NIKON D750": 5.95, "NIKON Z 6": 5.94, "NIKON Z 6_2": 5.94,
    "NIKON D5300": 3.89, "ZWO ASI2600MC Pro": 3.76, "ZWO ASI533MC Pro": 3.76, "ZWO ASI294MC Pro": 4.63,
    "ZWO ASI071MC Pro": 4.78, "ZWO ASI183MC Pro": 2.40, "ZWO ASI585MC": 2.90, "ZWO ASI2400MC Pro": 5.94,
    "QHY268C": 3.76, "QHY600C": 3.76, "Canon EOS 40D": 5.70, "Canon EOS 5D Mark II": 6.41, "Canon EOS 700D": 4.30,
    "PENTAX 645Z": 5.32, "GFX 50R": 5.31, "GFX 50S": 5.31, "GFX100S": 3.76, "ILCE-7RM2": 4.51,
    "X-T2": 3.91, "X-T3": 3.76, "X-T4": 3.76, "X-T5": 3.04, "X-T20": 3.91, "X-T30": 3.76, "X-T30 II": 3.76,
    "X-H1": 3.91, "X-H2": 3.04, "X-H2S": 3.76, "X-S10": 3.76, "X-E3": 3.91, "X-E4": 3.76, "X-Pro2": 3.91,
    "X-Pro3": 3.76,
}


@dataclass
class FrameInfo:
    path: str
    kind: str                       # "raw" | "fits" | "colour"
    width: int
    height: int
    cfa: str                        # e.g. "RGGB", 36 letters for X-Trans (core/cfa.py); "" for monochrome data; "RGB"
                                    # for a frame already in colour
    black: float                    # black level (ADU) - from the file, refined later by darks/bias
    white: float                    # saturation level (ADU)
    exposure: float | None = None   # seconds
    iso: float | None = None        # ISO or camera gain
    time: str | None = None         # ISO timestamp
    camera: str = ""
    focal_mm: float | None = None
    pixel_um: float | None = None
    temperature: float | None = None
    ra: float | None = None         # pointing hints from FITS headers (degrees)
    dec: float | None = None
    object: str = ""
    pixel_scale: float | None = None  # arcsec per pixel, when the FITS headers give it (PIXSCALE, SECPIX)
    filter: str = ""                # the session filter its FITS header names (FILTER: Ha, OIII...); "" for none
    stacked: int | None = None      # a master: the frames it was stacked from (0: not said); None: one frame
                                    # (``master_count``)

    def to_dict(self) -> dict:
        return asdict(self)


def _cfa_from_rawpy(r) -> str:
    """The pattern of the visible image's top-left block: 2 x 2 (Bayer) or 6 x 6 (X-Trans, issue #35); "" for any
    other. Read from the colour LibRaw gives each visible pixel (``raw_colors_visible``), which starts where the
    image read by ``read_cfa`` starts (``raw_pattern`` is anchored on the full sensor, margins included: taken for
    the visible image's, it shifts the colours when a margin is not a multiple of the period)."""
    pat = r.raw_pattern
    if pat is None or pat.shape not in ((2, 2), (6, 6)):
        return ""
    P = pat.shape[0]
    desc = r.color_desc.decode()
    cols = r.raw_colors_visible[:P, :P]
    return "".join(desc[int(cols[i, j])] for i in range(P) for j in range(P)).replace("g", "G").upper()


def read_info(path: str | Path, like: FrameInfo | None = None) -> FrameInfo:
    """Metadata of a frame. With ``like`` (a frame of the same camera already read), RAW files are not decoded:
    only their EXIF header is read and the sensor properties are copied (much faster on slow disks)."""
    path = Path(path)
    ext = path.suffix.lower()
    kind = frame_kind(path)
    if kind == "raw":
        if like is not None and like.kind == "raw" and Path(like.path).suffix.lower() == ext:
            info = FrameInfo(str(path), "raw", like.width, like.height, like.cfa, like.black, like.white)
            _exif_into(info, path)
            if info.camera and info.camera != like.camera:
                return _raw_info(path)
            return info
        return _raw_info(path)
    if kind == "fits":
        return _fits_info(path)
    if kind == "colour":
        return _colour_info(path)
    raise ValueError(f"unsupported file type: {path.name}")


def read_cfa(path: str | Path) -> np.ndarray:
    """Raw mosaic (H, W) float32 in ADU, top row first, black level NOT subtracted. A frame already in colour: its
    luminance (``read_colour``, core/cfa.luminance)."""
    path = Path(path)
    kind = frame_kind(path)
    if kind == "raw":
        import rawpy
        with rawpy.imread(str(path)) as r:
            return r.raw_image_visible.astype(np.float32)
    if kind == "fits":
        a, hd = _fits_data(path)
        if a.ndim == 3 and a.shape[0] == 3:
            return cfa_mod.luminance(_rows_top_down(a, hd))
        if a.ndim != 2:
            raise ValueError(f"{path.name}: expected a 2-D Bayer frame, got shape {a.shape}")
        area = useful_area(hd)
        if area is not None:                              # in the file's own rows: before they are put top first
            x, y, w, h = area
            a = a[y:y + h, x:x + w]
        return np.ascontiguousarray(_rows_top_down(a, hd))
    if kind == "colour":
        return cfa_mod.luminance(read_colour(path))
    raise ValueError(f"unsupported file type: {path.name}")


def _fits_data(path: Path):
    """(the image in ADU (``fits_adu``), its header) of a FITS file, read as float32 (compressed FITS: the image is in
    the first extension)."""
    from astropy.io import fits
    try:
        with fits.open(path, memmap=False) as h:
            hdu = next(x for x in h if x.data is not None)
            return fits_adu(np.asarray(hdu.data, np.float32), hdu.header), hdu.header
    except (OSError, StopIteration, TypeError, ValueError) as e:
        raise unreadable(path) from e


#: the highest value of a FITS image in floats that is still on the 0-1 scale (a little over 1: a calibrated or
#: normalised frame can go past it); a frame in ADU, even a dark, has its black level and its hot pixels far above it
UNIT_MAX = 1.5
#: the ADU of 1.0 in a FITS image in floats on the 0-1 scale: the stacking programs write the raw ADU divided by 65535
#: (a 14-bit camera's 512 black level is 0.0078 in their masters), whatever the camera's bit depth
UNIT_ADU = 65535.0


def fits_adu(a: np.ndarray, hd) -> np.ndarray:
    """The one rule for the values of a FITS frame (lights, darks, bias, flats; mosaic, monochrome or in colour): an image
    in floats (BITPIX < 0) whose values all stay within ``UNIT_MAX`` is on the 0-1 scale and is brought to ADU (times
    ``UNIT_ADU``); one in floats already in ADU, and every integer image, as the file holds it (astropy has applied
    BZERO and BSCALE: a 16-bit file with BZERO 32768 is 0-65535). Read on the whole image, so that every frame of a
    file is on the scale its header levels are (``fits_level``)."""
    if int(hd.get("BITPIX", 16)) >= 0 or not a.size or not float(np.nanmax(a)) <= UNIT_MAX:
        return a
    return a * np.float32(UNIT_ADU)


def fits_level(v: float, hd) -> float:
    """A level a FITS header gives in the image's own units (DATAMAX, SATURATE, PEDESTAL) in ADU, by ``fits_adu``'s rule:
    a level within ``UNIT_MAX`` in a file of floats is on the 0-1 scale."""
    return v * UNIT_ADU if int(hd.get("BITPIX", 16)) < 0 and v <= UNIT_MAX else v


def _rows_top_down(a: np.ndarray, hd) -> np.ndarray:
    """An image (..., H, W) with its top row first, as a FITS header's ROWORDER says."""
    return a[..., ::-1, :] if str(hd.get("ROWORDER", "TOP-DOWN")).upper() == "BOTTOM-UP" else a


def is_colour_frame(path: str | Path) -> bool:
    """Whether a frame is already in colour: a JPEG, PNG or TIFF, or a FITS frame of three planes (its header only)."""
    p = Path(path)
    kind = frame_kind(p)
    if kind == "colour":
        return True
    if kind != "fits":
        return False
    try:
        return _is_rgb_fits(fits_header(p))
    except Exception:  # noqa: BLE001 - an unreadable header is said elsewhere (it cannot be read)
        return False


def _is_rgb_fits(hd) -> bool:
    return int(hd.get("NAXIS", 0)) == 3 and int(hd.get("NAXIS3", 0)) == 3


def colour_session(lights: list, is_colour=None) -> dict | None:
    """What a list of light frames is when they are already in colour (issue #57): {"type": the file type,
    "stretched": True when it can only be a stretched picture (a JPEG, PNG or 8-bit TIFF, stages/image.py), so not
    linear data}; None for raw or FITS mosaics or monochrome frames. By the names, the first FITS frame's header."""
    from ..core.project import colour_lights, file_type
    from ..stages.image import stretched_by_nature
    if not lights:
        return None
    named = colour_lights(lights)
    if not named and not (frame_kind(lights[0]) == "fits" and (is_colour or is_colour_frame)(lights[0])):
        return None
    first = next((f for f in lights if frame_kind(f) == "colour"), lights[0]) if named else lights[0]
    return {"type": file_type(first), "stretched": stretched_by_nature(first)}


def read_colour(path: str | Path) -> np.ndarray:
    """A frame already in colour: (3, H, W) float32, planes R, G, B, top row first, as the file holds it (0-255,
    0-65535; float images in 0-1 brought to 0-65535 so that the calibrated cache, float16, keeps their faint values:
    ``fits_adu`` for a FITS file)."""
    path = Path(path)
    kind = frame_kind(path)
    if kind == "fits":
        a, hd = _fits_data(path)
        if a.ndim != 3 or a.shape[0] != 3:
            raise colour_refused(path, "it is not a colour image of three planes")
        return np.ascontiguousarray(_rows_top_down(a, hd))
    if kind != "colour":
        raise ValueError(f"unsupported file type: {path.name}")
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        try:
            a = np.asarray(tifffile.imread(path))
        except Exception as e:  # noqa: BLE001 - a damaged or foreign file
            raise colour_refused(path, "the file is damaged or not a TIFF") from e
        if a.ndim == 3 and a.shape[-1] in (3, 4):
            a = np.moveaxis(a[..., :3], -1, 0)
        elif not (a.ndim == 3 and a.shape[0] == 3):
            raise colour_refused(path, "it has one channel: it is not a colour image")
        return _fits_scale(a.astype(np.float32)) if a.dtype.kind == "f" else a.astype(np.float32)
    import cv2
    # IMREAD_COLOR applies a JPEG's orientation tag and takes off an alpha channel; a single-channel picture is said
    a = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if a is None:
        raise colour_refused(path, "the file is damaged or wrongly named")
    if _image_header(path)[3] < 3:
        raise colour_refused(path, "it has one channel: it is not a colour image")
    return np.moveaxis(a[..., ::-1], -1, 0).astype(np.float32)


def _fits_scale(a: np.ndarray) -> np.ndarray:
    """A float image in 0-1 brought to 0-65535; one already on that scale as it is (a TIFF of floats: ``fits_adu``'s
    rule)."""
    return a * np.float32(UNIT_ADU) if a.size and float(np.nanmax(a)) <= UNIT_MAX else a


def colour_refused(path: str | Path, why: str) -> Exception:
    """The error of a frame already in colour that cannot be used, in plain words with its name."""
    from ..core.stage import StageError
    return StageError(f"{Path(path).name} cannot be used as a colour frame: {why}; take it out of the session")


def _image_header(path: Path) -> tuple[int, int, int, int]:
    """(width, height, bits per channel, negative for floating point, channels) of a JPEG, PNG or TIFF, from its header
    alone; a JPEG's orientation tag counted (the size of the picture as ``read_colour`` gives it)."""
    suf = path.suffix.lower()
    if suf in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(path) as t:
            q = t.pages[0]
            bits = int(q.bitspersample)
            return int(q.imagewidth), int(q.imagelength), -bits if int(q.sampleformat) == 3 else bits, \
                int(q.samplesperpixel)
    if suf == ".png":
        with open(path, "rb") as f:
            head = f.read(26)
        if head[:4] != b"\x89PNG":
            raise ValueError("not a PNG")
        # the IHDR chunk: width, height, bit depth, colour type (0 grey, 2 RGB, 3 palette, 4 grey + alpha, 6 RGB + alpha)
        kind = head[25]
        return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"), head[24] if kind != 3 else 8, \
            {0: 1, 2: 3, 3: 3, 4: 1, 6: 3}.get(kind, 0)
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(274, 1) in (5, 6, 7, 8):           # the EXIF orientation: a quarter turn
            w, h = h, w
        return w, h, 8, len(im.getbands())


def _colour_info(path: Path) -> FrameInfo:
    """A JPEG, PNG or TIFF frame: its size and depth from its header, its capture from its EXIF (issue #57)."""
    try:
        w, h, bits, bands = _image_header(path)
    except Exception as e:  # noqa: BLE001 - a damaged or foreign file
        raise colour_refused(path, "the file is damaged or wrongly named") from e
    if bands < 3:
        raise colour_refused(path, "it has one channel: it is not a colour image")
    info = FrameInfo(str(path), "colour", w, h, cfa_mod.RGB, 0.0, 65535.0 if bits < 0 else 2.0 ** bits - 1)
    _exif_into(info, path)
    return info


def daylight_balance(path: str | Path) -> np.ndarray | None:
    """The camera's daylight white balance of a raw file, as (red, 1, blue) multipliers of the green: the balance the
    camera and the raw converters apply to a daylight scene (LibRaw's pre_mul, ``daylight_whitebalance``), the one
    DeepSkyStacker's "Use camera white balance" applies to raw files; the balance the camera recorded when the
    daylight one is missing. None for a FITS frame (no such balance) or when the file holds none."""
    if frame_kind(Path(path)) != "raw":
        return None
    import rawpy
    with rawpy.imread(str(path)) as r:
        for m in (r.daylight_whitebalance, r.camera_whitebalance):
            m = [float(v) for v in (m or [])][:3]
            if len(m) == 3 and all(np.isfinite(m)) and min(m) > 0:
                return np.array([m[0] / m[1], 1.0, m[2] / m[1]], np.float32)
    return None


def _raw_info(path: Path) -> FrameInfo:
    import rawpy
    with rawpy.imread(str(path)) as r:
        h, w = r.raw_image_visible.shape
        cfa = _cfa_from_rawpy(r)
        black = float(np.mean(r.black_level_per_channel))
        white = float(r.white_level)
    info = FrameInfo(str(path), "raw", w, h, cfa, black, white)
    _exif_into(info, path)
    return info


def exif_tags(path: Path) -> dict:
    """The EXIF tags of a RAW file, named as exifread names them ('Image Model', 'EXIF ExposureTime'...). exifread reads
    the TIFF-based files (ARW, CR2, NEF, PEF, DNG...) itself; a Canon CR3 keeps them in two ISO-media boxes (CMT1: the
    main directory, CMT2: the EXIF directory), a Fujifilm RAF in its JPEG preview."""
    import exifread
    ext = path.suffix.lower()
    with open(path, "rb") as f:
        if ext == ".cr3":
            head = f.read(1 << 20)
            tags = {}
            for box, group in ((b"CMT1", "Image"), (b"CMT2", "EXIF")):
                k = head.find(box)
                if k < 4:
                    continue
                size = int.from_bytes(head[k - 4:k], "big")
                t = exifread.process_file(io.BytesIO(head[k + 4:k - 4 + size]), details=False)
                tags.update({f"{group} {key.split(' ', 1)[1]}": v for key, v in t.items() if " " in key})
            return tags
        if ext == ".raf":
            head = f.read(92)
            off, n = int.from_bytes(head[84:88], "big"), int.from_bytes(head[88:92], "big")
            f.seek(off)
            return exifread.process_file(io.BytesIO(f.read(n)), details=False)
        return exifread.process_file(f, details=False)


def _exif_into(info: FrameInfo, path: Path) -> None:
    try:
        t = exif_tags(path)
        dt = t.get("EXIF DateTimeOriginal")
        if dt:
            info.time = datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S").isoformat()
        info.exposure = _ratio(t.get("EXIF ExposureTime"))
        iso = t.get("EXIF ISOSpeedRatings")
        info.iso = float(str(iso).split(",")[0]) if iso else None
        fl = _ratio(t.get("EXIF FocalLength"))
        info.focal_mm = fl if fl and fl > 0 else None
        info.camera = str(t.get("Image Model", "")).strip()
    except Exception:  # noqa: BLE001 - metadata is optional
        pass
    if info.time is None:          # no capture time in the file: its date on disk (a copy's is the copy's date)
        info.time = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    info.pixel_um = PIXEL_PITCH_UM.get(info.camera)


#: a date a capture program writes in its file names: YYYY-MM-DD or YYYYMMDD, not inside a longer number, then maybe a
#: time (HH-MM-SS, HHMMSS, HH:MM, after "_", "-", "T" or a space)
_NAME_DATE = re.compile(r"(?<!\d)((?:19|20)\d\d)(-?)(\d\d)\2(\d\d)(?:[_T -](\d\d)[-:h]?(\d\d)(?:[-:m]?(\d\d))?)?(?!\d)")


def name_date(name: str) -> datetime | None:
    """The date a frame's file name says (a capture program's "…_2025-06-17_23-46-58.arw"), with its time when it has
    one (else midnight), or None: no date, a number that is not a valid date between 1990 and a year from now (a frame
    counter, an exposure), or two different dates in one name (which one is the night's?). Never the file's date on
    disk: a copy changes it."""
    stem = Path(name).name
    found: set[datetime] = set()
    for m in _NAME_DATE.finditer(stem):
        y, mo, d, hh, mi, ss = (int(x) if x else None for x in m.group(1, 3, 4, 5, 6, 7))
        try:
            day = datetime(y, mo, d)
        except ValueError:
            continue
        if not datetime(1990, 1, 1) <= day <= datetime.now() + timedelta(days=366):
            continue
        if hh is not None and hh < 24 and mi < 60 and (ss or 0) < 60:
            day = day.replace(hour=hh, minute=mi, second=ss or 0)
        found.add(day)
    if len({t.date() for t in found}) != 1:
        return None
    return max(found)                                   # the one with a time, when the date is written twice


def clock_disagreement(frames) -> dict | None:
    """Whether the frames' own date (EXIF, FITS header: their ``time`` as ``read_info`` gives it) contradicts the date
    their file names say: a camera whose clock was never set writes 2014-01-01 in frames a capture program named
    2025-06-17. ``frames``: (file name or path, ISO time) pairs. The first pair that differs by more than two days (a
    night crosses midnight, a time zone moves a few hours: not a disagreement) gives {"camera": ISO date, "names": ISO
    date}; None when they agree or when no name holds a date (nothing is guessed). Only said: the engine keeps the
    frames' own date."""
    for name, time in frames:
        said = name_date(str(name)) if time else None
        if said is None:
            continue
        try:
            own = datetime.fromisoformat(str(time).strip().rstrip("Z")).replace(tzinfo=None)
        except ValueError:
            continue
        if abs(own - said) > timedelta(days=2):
            return {"camera": own.date().isoformat(), "names": said.date().isoformat()}
    return None


def _ratio(v) -> float | None:
    if v is None:
        return None
    s = str(v)
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except ValueError:
        return None


def fits_header(path: str | Path):
    """The header of the image of a FITS file (of its first extension for a compressed one, whose primary part is
    empty)."""
    from astropy.io import fits
    try:
        with fits.open(path, memmap=False) as h:
            for hdu in h:
                if hdu.header.get("NAXIS", 0) >= 2:
                    return hdu.header.copy()
            return h[0].header.copy()
    except (OSError, IndexError, ValueError) as e:     # not a FITS file (astropy: "No SIMPLE card found…")
        raise unreadable(path) from e


#: a FITS image section: [first column:last column,first row:last row], counted from 1, both ends included
_SECTION = re.compile(r"^\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]$")


def useful_area(hd) -> tuple[int, int, int, int] | None:
    """The useful area of a FITS image, (first column, first row, width, height) from 0 in the file's own rows; None
    when it is the whole image (no such keyword, or one that cannot be read or does not fit the image). TRIMSEC, the
    section to keep once the frame is reduced (IRAF's ccdproc, Massey's "A User's Guide to CCD Reductions with IRAF"),
    decides over DATASEC: observatory frames can give the whole image as DATASEC and the real area as TRIMSEC only."""
    W, H = int(hd.get("NAXIS1", 0)), int(hd.get("NAXIS2", 0))
    for key in ("TRIMSEC", "DATASEC"):
        m = _SECTION.match(str(hd.get(key, "")).strip())
        if m is None:
            continue
        x1, x2, y1, y2 = (int(v) for v in m.groups())
        if 1 <= x1 <= x2 <= W and 1 <= y1 <= y2 <= H:
            return None if (x2 - x1 + 1, y2 - y1 + 1) == (W, H) else (x1 - 1, y1 - 1, x2 - x1 + 1, y2 - y1 + 1)
    return None


def unreadable(path: str | Path) -> Exception:
    """The error of a frame that cannot be read, in plain words with its name and what to do (GUI exploration 2: the
    run said astropy's "OSError: No SIMPLE card found…")."""
    from ..core.stage import StageError
    return StageError(f"{Path(path).name} cannot be read: it is not a valid FITS file (damaged, or wrongly named); "
                      f"take it out of the session")


def _exposure_s(hd) -> float | None:
    """Exposure in seconds: EXPTIME or EXPOSURE, converted when the header gives it in milliseconds (Stellina:
    'EXPOSURE = 10000 / [ms] Total Exposure Time')."""
    for key in ("EXPTIME", "EXPOSURE"):
        v = _num(hd.get(key))
        if v is not None:
            unit = (hd.comments[key] or "").lower()
            return v / 1000.0 if "[ms]" in unit or "millisec" in unit or unit.startswith("ms") else v
    return None


#: filter names written by capture programs and observatories → the session filters (core/project2.FILTERS)
FILTER_NAMES = {"ha": "Ha", "h-alpha": "Ha", "halpha": "Ha", "h_alpha": "Ha", "oiii": "OIII", "o3": "OIII",
                "sii": "SII", "s2": "SII", "l": "L", "lum": "L", "luminance": "L", "clear": "L", "r": "R", "red": "R",
                "rp": "R", "g": "G", "green": "G", "gp": "G", "v": "G", "b": "B", "blue": "B"}


def frame_filter(path: str | Path) -> str:
    """The session filter a FITS frame names (its FILTER keyword; "" for a raw file, an unknown or no filter)."""
    p = Path(path)
    if frame_kind(p) != "fits":
        return ""
    try:
        return _filter_of(fits_header(p))
    except Exception:  # noqa: BLE001 - an unreadable header names no filter
        return ""


def _filter_of(hd) -> str:
    return FILTER_NAMES.get(str(hd.get("FILTER", "")).strip().lower(), "")


#: the header keywords of the number of frames a master was stacked from: STACKCNT (written by the stacking programs
#: of amateur astronomy, with a HISTORY line of the stacking), NCOMBINE (the combining task of the professional
#: reduction packages, and the programs that follow it). Keywords that count the darks or flats used to calibrate a
#: frame (DARK_CNT, CALSTAT...) are left aside: a calibrated light carries them too
STACK_COUNT = ("STACKCNT", "NCOMBINE")
#: what else says a file is a master: its image type ('Master Flat', 'MASTERDARK', the archives' 'masterflat'), or a
#: HISTORY line of its stacking ('mean stacking with winsorized sigma clipping rejection...')
MASTER_TYPE = re.compile(r"master", re.I)
MASTER_HISTORY = re.compile(r"\bstacking\b", re.I)


def master_count(hd) -> int | None:
    """Whether a calibration file is a master, and of how many frames, from its header alone: the frame count of
    ``STACK_COUNT`` (above 1), else 0 for an image type or a HISTORY line of a stacking (``MASTER_TYPE``,
    ``MASTER_HISTORY``); None for one frame. The only rule of the program for it (the Calibration uses a master as it
    is, stages/calibrate.py)."""
    for key in STACK_COUNT:
        n = _num(hd.get(key))
        if n is not None and n > 1:
            return int(n)
    if MASTER_TYPE.search(str(hd.get("IMAGETYP", ""))) or any(MASTER_HISTORY.search(str(h)) for h in hd.get("HISTORY", [])):
        return 0
    return None


def master_kinds(files: dict[str, list]) -> set[str]:
    """The calibration kinds of a session ({kind: [path]}) given as masters only (``master_count``, their headers
    alone; the first single frame stops the reading of its kind): a master FITS file beside raw lights is the normal
    case, not a file of the wrong type (core/project.type_warnings)."""
    out = set()
    for kind, paths in files.items():
        if kind == "lights" or not paths:
            continue
        try:
            if all(frame_kind(Path(f)) == "fits" and master_count(fits_header(f)) is not None for f in paths):
                out.add(kind)
        except Exception:  # noqa: BLE001 - a header that cannot be read: not a master, said elsewhere
            continue
    return out


def _fits_info(path: Path) -> FrameInfo:
    hd = fits_header(path)
    naxis = int(hd.get("NAXIS", 0))
    w, h = int(hd.get("NAXIS1", 0)), int(hd.get("NAXIS2", 0))
    pat = str(hd.get("BAYERPAT", "")).strip().upper() if naxis == 2 else ""
    area = useful_area(hd) if naxis == 2 else None
    if pat:
        xo, yo = int(hd.get("XBAYROFF", 0)) % 2, int(hd.get("YBAYROFF", 0)) % 2
        if area is not None:                          # the image now starts at the area's first pixel, top row first
            xo += area[0]
            yo += h - area[1] - area[3] if str(hd.get("ROWORDER", "")).upper() == "BOTTOM-UP" else area[1]
        grid = [[pat[0], pat[1]], [pat[2], pat[3]]]
        pat = "".join(grid[(i + yo) % 2][(j + xo) % 2] for i in range(2) for j in range(2))
    if area is not None:
        w, h = area[2], area[3]
    bitpix = int(hd.get("BITPIX", 16))
    # the levels in ADU, as fits_adu reads the pixels: a file of floats reaches 65535 (on the 0-1 scale, brought to
    # ADU; already in ADU, the 16 bits of a camera), and its header levels on the 0-1 scale are brought to ADU too
    # (white 1.0 made a float master's level of 0.13 read as 13 % of saturation, where it is 8400 ADU)
    white = fits_level(float(hd.get("DATAMAX", hd.get("SATURATE", 2 ** 16 - 1 if bitpix in (16, 32) or bitpix < 0
                                                                else 1.0))), hd)
    if _is_rgb_fits(hd):                       # a frame already in colour (issue #57): read_colour's range
        pat, white = cfa_mod.RGB, 255.0 if bitpix == 8 else 65535.0 if bitpix < 0 else float(hd.get("DATAMAX", 65535.0))
    info = FrameInfo(str(path), "fits", w, h, pat, fits_level(float(hd.get("PEDESTAL", hd.get("OFFSET", 0)) or 0), hd),
                     white)
    info.exposure = _exposure_s(hd)
    info.iso = _num(hd.get("GAIN", hd.get("ISOSPEED")))
    info.time = str(hd.get("DATE-OBS", "")) or datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    info.camera = str(hd.get("INSTRUME", "")).strip()
    info.focal_mm = _num(hd.get("FOCALLEN", hd.get("FOCAL")))
    info.pixel_um = _num(hd.get("XPIXSZ", hd.get("PIXSZ"))) or PIXEL_PITCH_UM.get(info.camera)
    info.temperature = _num(hd.get("CCD-TEMP", hd.get("SET-TEMP", hd.get("TEMP"))))
    info.object = str(hd.get("OBJECT", "")).strip()
    info.pixel_scale = _num(hd.get("PIXSCALE", hd.get("SECPIX")))
    info.filter = _filter_of(hd)
    info.stacked = master_count(hd)
    try:
        if "RA" in hd and "DEC" in hd and isinstance(hd["RA"], (int, float)):
            info.ra, info.dec = float(hd["RA"]), float(hd["DEC"])
        elif "RA" in hd and "DEC" in hd and ":" in str(hd["RA"]):         # sexagesimal text (observatory archives)
            # imported only then (#115: half a second at the first opening of a project of FITS frames, for nothing when
            # their headers have no coordinates)
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            c = SkyCoord(str(hd["RA"]), str(hd["DEC"]), unit=(u.hourangle, u.deg))
            info.ra, info.dec = float(c.ra.deg), float(c.dec.deg)
        elif "OBJCTRA" in hd and "OBJCTDEC" in hd:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            c = SkyCoord(hd["OBJCTRA"], hd["OBJCTDEC"], unit=(u.hourangle, u.deg))
            info.ra, info.dec = float(c.ra.deg), float(c.dec.deg)
    except Exception:  # noqa: BLE001
        pass
    return info


def _num(v) -> float | None:
    try:
        return float(v) if v is not None and str(v).strip() != "" else None
    except (TypeError, ValueError):
        return None


def cfa_channel_map(cfa: str) -> list[int]:
    """Output channel (0=R, 1=G, 2=B) of each position k = P row + col of the pattern (P: 2, or 6 for X-Trans)."""
    return ["RGB".index(c) for c in cfa]
