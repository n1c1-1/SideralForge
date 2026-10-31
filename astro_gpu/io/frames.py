"""Reading raw colour (Bayer) frames from camera RAW files and astro-camera FITS files.

Everything downstream works on the undebayered mosaic in ADU, top row first, with a 2x2 colour pattern string
such as ``"RGGB"`` describing the top-left 2x2 block.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

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
}


@dataclass
class FrameInfo:
    path: str
    kind: str                       # "raw" | "fits"
    width: int
    height: int
    cfa: str                        # e.g. "RGGB"; "" for non-Bayer data (unsupported in v1)
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

    def to_dict(self) -> dict:
        return asdict(self)


def _cfa_from_rawpy(r) -> str:
    desc = r.color_desc.decode()
    pat = r.raw_pattern
    s = "".join(desc[int(pat[i, j])] for i in range(2) for j in range(2))
    return s.replace("g", "G")


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
    raise ValueError(f"unsupported file type: {path.name}")


def read_cfa(path: str | Path) -> np.ndarray:
    """Raw mosaic (H, W) float32 in ADU, top row first, black level NOT subtracted."""
    path = Path(path)
    kind = frame_kind(path)
    if kind == "raw":
        import rawpy
        with rawpy.imread(str(path)) as r:
            return r.raw_image_visible.astype(np.float32)
    if kind == "fits":
        from astropy.io import fits
        with fits.open(path, memmap=False) as h:          # compressed FITS: the image is in the first extension
            hdu = next(x for x in h if x.data is not None)
            a = np.asarray(hdu.data, np.float32)
            hd = hdu.header
        if a.ndim != 2:
            raise ValueError(f"{path.name}: expected a 2-D Bayer frame, got shape {a.shape}")
        if str(hd.get("ROWORDER", "TOP-DOWN")).upper() == "BOTTOM-UP":
            a = a[::-1]
        return np.ascontiguousarray(a)
    raise ValueError(f"unsupported file type: {path.name}")


def _raw_info(path: Path) -> FrameInfo:
    import rawpy
    with rawpy.imread(str(path)) as r:
        h, w = r.raw_image_visible.shape
        cfa = _cfa_from_rawpy(r) if r.raw_pattern is not None and r.raw_pattern.shape == (2, 2) else ""
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
    with fits.open(path, memmap=False) as h:
        for hdu in h:
            if hdu.header.get("NAXIS", 0) >= 2:
                return hdu.header.copy()
        return h[0].header.copy()


def _exposure_s(hd) -> float | None:
    """Exposure in seconds: EXPTIME or EXPOSURE, converted when the header gives it in milliseconds (Stellina:
    'EXPOSURE = 10000 / [ms] Total Exposure Time')."""
    for key in ("EXPTIME", "EXPOSURE"):
        v = _num(hd.get(key))
        if v is not None:
            unit = (hd.comments[key] or "").lower()
            return v / 1000.0 if "[ms]" in unit or "millisec" in unit or unit.startswith("ms") else v
    return None


def _fits_info(path: Path) -> FrameInfo:
    hd = fits_header(path)
    naxis = int(hd.get("NAXIS", 0))
    w, h = int(hd.get("NAXIS1", 0)), int(hd.get("NAXIS2", 0))
    pat = str(hd.get("BAYERPAT", "")).strip().upper() if naxis == 2 else ""
    if pat:
        xo, yo = int(hd.get("XBAYROFF", 0)) % 2, int(hd.get("YBAYROFF", 0)) % 2
        grid = [[pat[0], pat[1]], [pat[2], pat[3]]]
        pat = "".join(grid[(i + yo) % 2][(j + xo) % 2] for i in range(2) for j in range(2))
    bitpix = int(hd.get("BITPIX", 16))
    white = float(hd.get("DATAMAX", 2 ** 16 - 1 if bitpix in (16, 32) else 1.0))
    info = FrameInfo(str(path), "fits", w, h, pat, float(hd.get("PEDESTAL", hd.get("OFFSET", 0)) or 0), white)
    info.exposure = _exposure_s(hd)
    info.iso = _num(hd.get("GAIN", hd.get("ISOSPEED")))
    info.time = str(hd.get("DATE-OBS", "")) or datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    info.camera = str(hd.get("INSTRUME", "")).strip()
    info.focal_mm = _num(hd.get("FOCALLEN", hd.get("FOCAL")))
    info.pixel_um = _num(hd.get("XPIXSZ", hd.get("PIXSZ"))) or PIXEL_PITCH_UM.get(info.camera)
    info.temperature = _num(hd.get("CCD-TEMP", hd.get("SET-TEMP", hd.get("TEMP"))))
    info.object = str(hd.get("OBJECT", "")).strip()
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        if "RA" in hd and "DEC" in hd and isinstance(hd["RA"], (int, float)):
            info.ra, info.dec = float(hd["RA"]), float(hd["DEC"])
        elif "OBJCTRA" in hd and "OBJCTDEC" in hd:
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
    """Output channel (0=R, 1=G, 2=B) of each 2x2 position k = 2*row + col."""
    return ["RGB".index(c) for c in cfa]
