"""An image source (issue #43): a workflow that starts from an image file instead of the sessions' frames.

The image (FITS or TIFF, linear: straight out of a stack) is written as a stack result, which the later stages read
as "stack" (core/workflow.PROVIDES): drz_full.npy (3, H, W), drz_weight.npy (all ones: every pixel covered),
drz_meta.json (scale 1: the image is at its own scale). It has no half-stacks: Denoise is greyed, and Stars takes its
noise unit from the image itself (stages/star_removal.py).

A monochrome image is given its one channel three times. Values in [0, 1] are brought to the range of a 16-bit
sensor's (x 65535), so that the levels the later stages measure are of the size they meet in a stack.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.params import P, StageParams
from ..core.stage import Context, Stage, StageError

FORMATS = (".fits", ".fit", ".fts", ".tif", ".tiff", ".png", ".jpg", ".jpeg")
#: the first bytes of each format (GUI exploration 7.3: a text file or a missing one is refused with its reason)
MAGIC = {".fits": (b"SIMPLE",), ".fit": (b"SIMPLE",), ".fts": (b"SIMPLE",),
         ".tif": (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+"), ".tiff": (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+"),
         ".png": (b"\x89PNG",), ".jpg": (b"\xff\xd8\xff",), ".jpeg": (b"\xff\xd8\xff",)}
#: an 8 or 16-bit picture: stretched by nature (GUI exploration 1.7)
STRETCHED_FORMATS = (".png", ".jpg", ".jpeg")


def stretched_by_nature(path: str | Path) -> bool:
    """Whether a file can only be a stretched picture: a PNG or JPEG, or an 8-bit TIFF (a stack is never stored on 8
    bits; GUI exploration round 3: an 8-bit TIFF stayed "Linear" where the same photo in PNG was "Stretched"). Reads
    the TIFF's first page header only."""
    p = Path(str(path or ""))
    suf = p.suffix.lower()
    if suf in STRETCHED_FORMATS:
        return True
    if suf not in (".tif", ".tiff"):
        return False
    try:
        import tifffile
        with tifffile.TiffFile(p) as t:
            return t.pages[0].bitspersample <= 8
    except Exception:  # noqa: BLE001 - a TIFF that cannot be read: its problem is said elsewhere (image_problem)
        return False


def image_problem(path: str | Path) -> str:
    """Why a file cannot be an Image block's image, in plain English ("{name}" to fill in), or "" when it can: none
    chosen, not found, not a format read here, or its content not of its format (a text file renamed)."""
    if not str(path or "").strip():
        return "No image file chosen yet."
    p = Path(str(path).strip())
    if not p.is_file():
        return "File not found: {name}"
    suf = p.suffix.lower()
    if suf not in FORMATS:
        return "{name} is not an image this version reads (FITS, TIFF, PNG or JPEG)."
    try:
        with open(p, "rb") as f:
            head = f.read(8)
        if any(head.startswith(m) for m in MAGIC[suf]) and suf in (".fits", ".fit", ".fts") and _fits_damaged(p):
            return "{name} cannot be read: the file is damaged."
    except OSError:
        return "{name} cannot be read."
    if not any(head.startswith(m) for m in MAGIC[suf]):
        return "{name} is not an image: its content is not that of its format."
    return ""


def _fits_damaged(p: Path) -> bool:
    """Whether a FITS file is cut or damaged (GUI exploration round 2: said before the run, as a missing file is): its
    header has no END card, or the file is shorter than the data its header declares. Reads only the header."""
    size = p.stat().st_size
    cards: dict[str, str] = {}
    n = 0
    with open(p, "rb") as f:
        while True:
            block = f.read(2880)
            if len(block) < 2880 or n >= 1000:           # no END in what there is (or an absurd header)
                return True
            n += 1
            for i in range(0, 2880, 80):
                card = block[i:i + 80].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    try:
                        bitpix = abs(int(cards.get("BITPIX", "0")))
                        dims = [int(cards.get(f"NAXIS{k}", "0")) for k in range(1, int(cards.get("NAXIS", "0")) + 1)]
                    except ValueError:
                        return True
                    data = bitpix // 8 * int(np.prod(dims)) if dims else 0
                    return size < n * 2880 + data
                if card[8:10] == "= ":
                    cards[key] = card[10:].split("/")[0].strip()


class ImageParams(StageParams):
    path: str = P("", label="Image file", help="The image this workflow starts from (FITS or TIFF).")
    stretched: bool = P(False, label="Already stretched", level="internal",
                        help="The image is already stretched (not linear): it goes to Post, or to Split stars, as "
                             "it is.")
    align_to: str = P("", label="Aligned on", level="internal",
                      help="The image file this one is aligned on (Align images); empty: as it is.")


def read_image(path: str | Path) -> np.ndarray:
    """(3, H, W) float32 of an image file: FITS (its first image) or TIFF; a monochrome image on three channels."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        with fits.open(path, memmap=False) as h:
            hdu = next((x for x in h if x.data is not None and x.data.ndim >= 2), None)
            if hdu is None:
                raise StageError(f"no image in {path.name}")
            a = np.asarray(hdu.data, np.float32)
        if a.ndim == 3 and a.shape[0] not in (1, 3) and a.shape[-1] in (1, 3):
            a = np.moveaxis(a, -1, 0)
    elif suf in (".tif", ".tiff"):
        import tifffile
        a = np.asarray(tifffile.imread(path), np.float32)
        if a.ndim == 3 and a.shape[-1] in (1, 3, 4):
            a = np.moveaxis(a[..., :3], -1, 0)
    elif suf in (".png", ".jpg", ".jpeg"):            # a stretched image (8 or 16 bits)
        import cv2
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise StageError(f"{path.name}: the image cannot be read")
        a = a.astype(np.float32)
        a = np.moveaxis(a[..., :3][..., ::-1], -1, 0) if a.ndim == 3 else a
    else:
        raise StageError(f"{path.name}: not an image this version reads (FITS, TIFF, PNG or JPEG)")
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3 or a.shape[0] not in (1, 3):
        raise StageError(f"{path.name}: an image of shape {a.shape} is neither monochrome nor colour")
    if a.shape[0] == 1:
        a = np.repeat(a, 3, axis=0)
    return a


class ImageSource(Stage):
    name = "image"
    title = "Image"
    description = ("Starts the workflow from an image file (a linear stack): the blocks after it read it as they read "
                   "a stack.")
    Params = ImageParams
    version = "1"

    def run(self, ctx: Context, p: ImageParams) -> None:
        if not p.path:
            raise StageError("no image file: choose the image this workflow starts from")
        if not Path(p.path).is_file():
            raise StageError(f"the image file is not there: {p.path}")
        a = read_image(p.path)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        valid = None
        if p.align_to:
            from .align_images import align_onto
            ref = np.nan_to_num(read_image(p.align_to), nan=0.0, posinf=0.0, neginf=0.0)
            a, valid = align_onto(a, p.path, ref, p.align_to, ctx.log)
        if p.stretched:
            self.stretched(ctx, p, a, valid)
            return
        top = float(np.percentile(a, 99.99)) if a.size else 0.0
        factor = 65535.0 if 0 < top <= 1.5 else 1.0
        if factor != 1.0:
            a = a * np.float32(factor)
        _c, H, W = a.shape
        ctx.save("drz_full.npy", a.astype(np.float32))
        ctx.save("drz_weight.npy", np.ones((3, H, W), np.float32) if valid is None else
                 np.repeat(valid[None].astype(np.float32), 3, axis=0))
        ctx.save_json("drz_meta.json", {"scale": 1.0, "source": "image", "file": Path(p.path).name,
                                        "factor": factor, "height": H, "width": W})
        ctx.metric("width", W)
        ctx.metric("height", H)
        ctx.metric("factor", factor)
        ctx.log(f"image {Path(p.path).name}: {W} x {H}" + (" (values x 65535)" if factor != 1.0 else ""))
        v = np.moveaxis(a, 0, -1)
        lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
        ctx.preview("image", np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.5)

    def stretched(self, ctx: Context, p: ImageParams, a: np.ndarray, valid: np.ndarray | None = None) -> None:
        """An image already stretched: its values in 0-1 (an integer file by its type's range, a float one by its
        maximum when above 1), written as the Output block writes (lin.npy, valid.npy, prep_simple.json, flagged
        stretched: Post shows it as it is) and as a stack (for Split stars, which reads its noise there)."""
        from ..render.simple import image_stats
        top = float(a.max()) if a.size else 1.0
        scale = 65535.0 if top > 255.5 else (255.0 if top > 1.5 else 1.0)
        a = np.clip(a / np.float32(scale), 0, 1).astype(np.float32)
        _c, H, W = a.shape
        valid = np.ones((H, W), bool) if valid is None else valid
        stats = image_stats(a, valid)
        stats["stretched"] = True
        ctx.save("lin.npy", a)
        np.save(ctx.out("valid.npy"), valid)
        for name in ("drz_full.npy", "drz_A.npy", "drz_B.npy"):
            ctx.save(name, a)
        ctx.save("drz_weight.npy", np.repeat(valid[None].astype(np.float32), 3, axis=0))
        ctx.save_json("drz_meta.json", {"scale": 1.0, "source": "image", "stretched": True, "width": W, "height": H})
        ctx.save_json("prep_simple.json", {
            "M": np.eye(3).tolist(), "crop": [0, W, 0, H], "median": stats["lum"]["median"], "mad": stats["lum"]["mad"],
            "p99995": stats["lum"]["p99995"], "stats": stats, "stretch": "auto", "linked": True, "denoised": False,
            "color": True, "sky": "as stretched", "source": Path(p.path).name, "gains": [1.0, 1.0, 1.0], "exit": None,
            "width": W, "height": H, "drizzle_scale": 1.0, "stretched": True})
        ctx.metric("width", W)
        ctx.metric("height", H)
        ctx.metric("factor", scale)
        ctx.log(f"image {Path(p.path).name}: {W} x {H}, already stretched (values / {scale:g})")
        ctx.preview("image", np.moveaxis(a, 0, -1))
