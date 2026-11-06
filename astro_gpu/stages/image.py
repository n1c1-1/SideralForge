"""An image source (issue #43): a workflow that starts from an image file instead of the sessions' frames.

The image (FITS or TIFF, linear: straight out of a stack) is written as a stack result, which the later stages read
as "stack" (core/workflow.PROVIDES): drz_full.npy (3, H, W), drz_weight.npy (all ones: every pixel covered),
drz_meta.json (scale 1: the image is at its own scale). It has no half-stacks: Denoise, then Stars, are greyed.

A monochrome image is given its one channel three times. Values in [0, 1] are brought to the range of a 16-bit
sensor's (x 65535), so that the levels the later stages measure are of the size they meet in a stack.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.params import P, StageParams
from ..core.stage import Context, Stage, StageError

FORMATS = (".fits", ".fit", ".fts", ".tif", ".tiff", ".png", ".jpg", ".jpeg")


class ImageParams(StageParams):
    path: str = P("", label="Image file", help="The image this workflow starts from (FITS or TIFF).")
    stretched: bool = P(False, label="Already stretched", level="internal",
                        help="The image is already stretched (not linear): it goes to Post, or to Split stars, as "
                             "it is.")


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
        if p.stretched:
            self.stretched(ctx, p, a)
            return
        top = float(np.percentile(a, 99.99)) if a.size else 0.0
        factor = 65535.0 if 0 < top <= 1.5 else 1.0
        if factor != 1.0:
            a = a * np.float32(factor)
        _c, H, W = a.shape
        ctx.save("drz_full.npy", a.astype(np.float32))
        ctx.save("drz_weight.npy", np.ones((3, H, W), np.float32))
        ctx.save_json("drz_meta.json", {"scale": 1.0, "source": "image", "file": Path(p.path).name,
                                        "factor": factor, "height": H, "width": W})
        ctx.metric("width", W)
        ctx.metric("height", H)
        ctx.metric("factor", factor)
        ctx.log(f"image {Path(p.path).name}: {W} x {H}" + (" (values x 65535)" if factor != 1.0 else ""))
        v = np.moveaxis(a, 0, -1)
        lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
        ctx.preview("image", np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.5)

    def stretched(self, ctx: Context, p: ImageParams, a: np.ndarray) -> None:
        """An image already stretched: its values in 0-1 (an integer file by its type's range, a float one by its
        maximum when above 1), written as the Output block writes (lin.npy, valid.npy, prep_simple.json, flagged
        stretched: Post shows it as it is) and as a stack (for Split stars, which reads its noise there)."""
        from ..render.simple import image_stats
        top = float(a.max()) if a.size else 1.0
        scale = 65535.0 if top > 255.5 else (255.0 if top > 1.5 else 1.0)
        a = np.clip(a / np.float32(scale), 0, 1).astype(np.float32)
        _c, H, W = a.shape
        valid = np.ones((H, W), bool)
        stats = image_stats(a, valid)
        stats["stretched"] = True
        ctx.save("lin.npy", a)
        np.save(ctx.out("valid.npy"), valid)
        for name in ("drz_full.npy", "drz_A.npy", "drz_B.npy"):
            ctx.save(name, a)
        ctx.save("drz_weight.npy", np.ones((3, H, W), np.float32))
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
