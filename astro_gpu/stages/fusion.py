"""The channels' fusion (issue #43): the outputs of several workflows (one per filter of a monochrome camera, or the
channels of one colour workflow shot through a dual-band filter) combined into one colour image, with a palette.

A fusion workflow starts from other workflows (``Workflow.source = {"fusion": {...}}``): its ``fusion`` stage reads
their Output blocks' linear images (render_simple: lin.npy, the sky taken off; valid.npy) and writes a stack, which its
Output block renders as any stack (core/workflow.PROVIDES).

1. Alignment: every input is resampled onto the first one's grid through their plate solutions (each workflow's
   astrometry: wcs.fits): a pixel of the first grid goes to the sky, then into the other image (the reprojection of
   the FITS World Coordinate System, Calabretta & Greisen 2002; the map computed on a coarse grid and interpolated,
   then a bicubic resampling). Inputs of the same workflow (a dual-band image's channels) need none.
2. Each channel is its input's luminance (a monochrome image's grey), or one of its colours (``R``, ``G``, ``B``), or
   ``GB`` (the mean of green and blue: the O III of a dual-band filter on a colour sensor, whose green and blue pixels
   both see the 501 nm line).
3. Each channel is brought to the same scale: divided by a high percentile of its signal (99.5 %, above its sky at
   0), so that a faint O III weighs as much as a strong H-alpha in the palette (the channels' balance of narrowband
   palettes, which every reference guide sets by eye or by such a normalisation).
4. The palette maps the channels to red, green and blue: SHO (S II red, H-alpha green, O III blue: the "Hubble
   palette"), HSO, HOO (H-alpha red, O III green and blue), RGB, and LRGB (the colour of RGB, the luminance of L:
   each pixel's RGB scaled so that its luminance is L's).

The result's valid area is where every input has data. Its scale: the first input's (its level kept).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..core.params import P, StageParams
from ..core.stage import Context, Stage, StageError

#: palettes: the roles a palette reads, and its matrix (rows R, G, B; columns the roles, in that order)
PALETTES: dict[str, tuple[tuple[str, ...], list[list[float]]]] = {
    "SHO": (("SII", "Ha", "OIII"), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "HSO": (("Ha", "SII", "OIII"), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "HOO": (("Ha", "OIII"), [[1, 0], [0, 1], [0, 1]]),
    "RGB": (("R", "G", "B"), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "LRGB": (("L", "R", "G", "B"), [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
}
CHANNELS = ("L", "R", "G", "B", "GB")
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


class FusionParams(StageParams):
    palette: str = P("SHO", label="Palette", choices=list(PALETTES),
                     help="How the filters become red, green and blue: SHO (S II, H-alpha, O III: the Hubble "
                          "palette), HSO, HOO (H-alpha red, O III green and blue), RGB, LRGB (the luminance of L).")
    #: the inputs, set by the project (not keyed: their workflows' keys are): [{"role", "channel", "dir", "wcs"}]
    inputs: list[dict] = P([], label="Inputs", level="internal",
                           help="The Output folders of the workflows combined, their roles and channels.")
    percentile: float = P(99.5, label="Channel balance", level="advanced", ge=90, le=99.99, step=0.1, unit="%",
                          help="Each channel is divided by this percentile of its signal before the palette: the "
                               "filters weigh alike.")


def channel_of(lin: np.ndarray, channel: str) -> np.ndarray:
    """One channel of an input's linear image (3, H, W): its luminance (L), a colour, or the mean of green and blue."""
    if channel == "L":
        return np.tensordot(LUM, lin, 1).astype(np.float32)
    if channel == "GB":
        return (0.5 * (lin[1] + lin[2])).astype(np.float32)
    return lin["RGB".index(channel)].astype(np.float32)


def reprojection_map(wcs_ref, wcs_in, shape: tuple[int, int], step: int = 32):
    """(map_x, map_y) float32 of the reference grid's pixels in the input image (cv2.remap): through the sky, on a
    coarse grid spanning the image, interpolated by a bicubic spline."""
    from scipy.interpolate import RectBivariateSpline
    H, W = shape
    ys = np.linspace(0, H - 1, max(4, int(np.ceil(H / step)) + 1))
    xs = np.linspace(0, W - 1, max(4, int(np.ceil(W / step)) + 1))
    gx, gy = np.meshgrid(xs, ys)
    ra, dec = wcs_ref.all_pix2world(gx, gy, 0)
    ix, iy = wcs_in.all_world2pix(ra, dec, 0)
    yy, xx = np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64)
    mx = RectBivariateSpline(ys, xs, ix, kx=3, ky=3)(yy, xx).astype(np.float32)
    my = RectBivariateSpline(ys, xs, iy, kx=3, ky=3)(yy, xx).astype(np.float32)
    return mx, my


def balance(ch: np.ndarray, valid: np.ndarray, pct: float) -> tuple[np.ndarray, float]:
    """A channel divided by the given percentile of its valid pixels above 0 (its sky is at 0)."""
    v = ch[valid]
    v = v[np.isfinite(v)]
    top = float(np.percentile(v, pct)) if v.size else 1.0
    top = top if top > 0 else 1.0
    return ch / np.float32(top), top


def fuse(chans: dict[str, np.ndarray], palette: str) -> np.ndarray:
    """(3, H, W) of the balanced channels by role through a palette."""
    roles, mat = PALETTES[palette]
    missing = [r for r in roles if r not in chans]
    if missing:
        raise StageError(f"the {palette} palette needs the {', '.join(missing)} channel(s)")
    if palette == "LRGB":
        rgb = np.stack([chans["R"], chans["G"], chans["B"]])
        lum = np.tensordot(LUM, rgb, 1)
        k = np.where(lum > 1e-6, chans["L"] / np.maximum(lum, 1e-6), 1.0).astype(np.float32)
        return (rgb * k[None]).astype(np.float32)
    m = np.asarray(mat, np.float32)
    stack = np.stack([chans[r] for r in roles])
    return np.tensordot(m, stack, 1).astype(np.float32)


class Fusion(Stage):
    name = "fusion"
    title = "Channels' fusion"
    description = ("Combines the outputs of several workflows (one per filter, or a dual-band image's channels) into "
                   "one colour image with a palette (SHO, HSO, HOO, RGB, LRGB), aligned through their plate "
                   "solutions.")
    Params = FusionParams
    version = "1"

    def run(self, ctx: Context, p: FusionParams) -> None:
        from astropy.io import fits
        from astropy.wcs import WCS
        if not p.inputs:
            raise StageError("no workflow to combine: choose the channels of the fusion")
        ref = None
        chans: dict[str, np.ndarray] = {}
        valid_all = None
        tops = {}
        for n, inp in enumerate(p.inputs):
            d = Path(inp["dir"])
            if not (d / "lin.npy").is_file():
                raise StageError(f"the output of the {inp['role']} channel is not computed")
            lin = np.load(d / "lin.npy", mmap_mode="r")
            valid = np.load(d / "valid.npy")
            ch = channel_of(np.asarray(lin), inp.get("channel", "L"))
            del lin
            wcs_f = Path(inp.get("wcs") or "")
            if ref is None:
                ref = {"shape": ch.shape, "wcs": WCS(fits.getheader(wcs_f)) if wcs_f.is_file() else None,
                       "dir": d, "src": str(d)}
                out_ch, out_valid = ch, valid
            elif str(d) == ref["src"] or ch.shape == ref["shape"] and inp.get("same_grid"):
                out_ch, out_valid = ch, valid
            else:
                if ref["wcs"] is None or not wcs_f.is_file():
                    raise StageError("the channels come from different stacks: each needs its Astrometry (its plate "
                                     "solution aligns it on the first channel)")
                mx, my = reprojection_map(ref["wcs"], WCS(fits.getheader(wcs_f)), ref["shape"])
                out_ch = cv2.remap(ch, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                out_valid = cv2.remap(valid.astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
                ctx.log(f"{inp['role']}: aligned on the first channel through the plate solutions")
            valid_all = out_valid if valid_all is None else (valid_all & out_valid)
            chans[inp["role"]] = out_ch
            ctx.progress(0.8 * (n + 1) / len(p.inputs), f"channel {inp['role']}")
        level = None
        for role in list(chans):
            chans[role], tops[role] = balance(chans[role], valid_all, p.percentile)
            if level is None:
                level = tops[role]
        rgb = fuse(chans, p.palette) * np.float32(level or 1.0)
        rgb[:, ~valid_all] = 0
        H, W = ref["shape"]
        ctx.save("drz_full.npy", rgb.astype(np.float32))
        ctx.save("drz_weight.npy", np.repeat(valid_all[None].astype(np.float32), 3, axis=0))
        meta = json.loads((ref["dir"] / "prep_simple.json").read_text(encoding="utf-8")) \
            if (ref["dir"] / "prep_simple.json").is_file() else {}
        ctx.save_json("drz_meta.json", {"scale": float(meta.get("drizzle_scale", 1.0)), "source": "fusion",
                                        "palette": p.palette, "width": W, "height": H,
                                        "channels": {r: round(t, 4) for r, t in tops.items()}})
        if ref["wcs"] is not None:
            ref["wcs"].to_fits().writeto(ctx.out("wcs.fits"), overwrite=True)
        ctx.metric("palette", p.palette)
        ctx.metric("channels", sorted(chans))
        ctx.log(f"{p.palette} from {', '.join(chans)}; channel scales " +
                ", ".join(f"{r} {t:.4g}" for r, t in tops.items()))
        v = np.moveaxis(rgb, 0, -1)
        lo, hi = np.percentile(v[valid_all], 1), np.percentile(v[valid_all], 99.7)
        ctx.preview("fusion", np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.5)
