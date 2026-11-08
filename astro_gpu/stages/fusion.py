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

Stars (option A, the user's choice of 6 November; the published practice: the palette made on starless images, a star
layer put back at the end - T. Jones, astrobackyard.com/rgb-stars-narrowband-images; P. Cosgrove, cosgrovescosmos.com,
"my NB workflow"): each channel's workflow has its Split stars block; the fusion combines their starless images and
writes a star layer (``STARS_FILE``) from a source: the channels' stars with the same palette and scales, one
channel's stars (neutral), or another workflow's stars (a short RGB or colour session, in their own colours). The
fusion's own Split stars block then gives Post the starless fusion and that star layer, which the Combine stars &
background block recombines. Taking the stars off before combining avoids the coloured rings of stars whose size
differs from one filter to another.
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
#: the star layer of a starless fusion (option A), in the fusion's folder: what its Split stars block reads
STARS_FILE = "fusion_stars.npy"
#: where a starless fusion's stars come from
STAR_SOURCES = ("palette", "channel", "workflow")


class FusionParams(StageParams):
    palette: str = P("SHO", label="Palette", choices=list(PALETTES),
                     help="How the filters become red, green and blue: SHO (S II, H-alpha, O III: the Hubble "
                          "palette), HSO, HOO (H-alpha red, O III green and blue), RGB, LRGB (the luminance of L).")
    #: the inputs, set by the project (not keyed: their workflows' keys are): [{"role", "channel", "dir",
    #: "stars_dir", "wcs", "pid"}]
    inputs: list[dict] = P([], label="Inputs", level="internal",
                           help="The Output folders of the workflows combined, their roles and channels.")
    #: option A, set by the project: {} (the channels with their stars), or {"mode": "palette" | "channel" ("role") |
    #: "workflow" ("dir", "wcs", "pid")}
    stars: dict = P({}, label="Stars", level="internal",
                    help="Where the stars put back on the starless fusion come from.")
    #: the user's weight of each channel ({role: weight}, 1 by default), set by the project: a channel too faint for
    #: the balance (its noise raised with it) is lowered by hand
    weights: dict = P({}, label="Channel weights", level="internal",
                      help="Each channel's weight in the palette, after the balance (1: as balanced).")
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


def warp(a: np.ndarray, wp, nearest: bool = False) -> np.ndarray:
    """An image resampled through a reprojection map (reprojection_map)."""
    mx, my = wp
    return cv2.remap(np.ascontiguousarray(a, np.float32), mx, my, cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def balance(ch: np.ndarray, valid: np.ndarray, pct: float) -> tuple[np.ndarray, float]:
    """A channel divided by the given percentile of its valid pixels above 0 (its sky is at 0)."""
    v = ch[valid]
    v = v[np.isfinite(v)]
    top = float(np.percentile(v, pct)) if v.size else 1.0
    top = top if top > 0 else 1.0
    return ch / np.float32(top), top


def star_support(s: np.ndarray, k: float = 3.0, grow: int = 2) -> np.ndarray:
    """A stars-only image kept on its stars: the pixels above k times its noise (1.4826 x its median absolute
    deviation), the mask grown by ``grow`` pixels to keep the stars' wings; zero elsewhere (the noise the subtraction
    left). The classic star mask by threshold."""
    v = s[::3, ::3]
    med = float(np.median(v))
    sig = 1.4826 * float(np.median(np.abs(v - med))) or 1e-12
    m = (s - med > k * sig).astype(np.uint8)
    if grow:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return np.where(m > 0, np.maximum(s - med, 0), 0).astype(np.float32)


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
    title = "Combine channels"
    description = ("Combines the outputs of several workflows (one per filter, or a dual-band image's channels) into "
                   "one colour image with a palette (SHO, HSO, HOO, RGB, LRGB), aligned through their plate "
                   "solutions.")
    Params = FusionParams
    version = "3"                    # 2: option A, the starless channels and a star layer; 3: the stars kept on
    #                                  their support (the noise of the subtraction left out); the channels' weights

    def run(self, ctx: Context, p: FusionParams) -> None:
        from astropy.io import fits
        from astropy.wcs import WCS
        if not p.inputs:
            raise StageError("no workflow to combine: choose the channels of the fusion")
        starless = bool(p.stars)                     # option A: the channels' starless images, then a star layer
        ref: dict = {}
        chans: dict[str, np.ndarray] = {}
        warps: dict[str, tuple | None] = {}
        valid_all = None
        tops: dict[str, float] = {}

        def align(a: np.ndarray, wcs_f: Path, src: str):
            """(the image on the first channel's grid, its warp or None)."""
            if src == ref["src"]:
                return a, None
            if ref["wcs"] is None or not wcs_f.is_file():
                raise StageError("the channels come from different stacks: each needs its Astrometry (its plate "
                                 "solution aligns it on the first channel)")
            wp = reprojection_map(ref["wcs"], WCS(fits.getheader(wcs_f)), ref["shape"])
            return warp(a, wp), wp

        for n, inp in enumerate(p.inputs):
            d = Path(inp["stars_dir"] if starless else inp["dir"])
            name = "starless.npy" if starless else "lin.npy"
            if not (d / name).is_file():
                raise StageError(f"the {inp['role']} channel has no starless image: add Stars/background to "
                                 "its workflow, and run it" if starless else
                                 f"the output of the {inp['role']} channel is not computed")
            valid = np.load(d / "valid.npy")
            ch = channel_of(np.asarray(np.load(d / name, mmap_mode="r")), inp.get("channel", "L"))
            wcs_f = Path(inp.get("wcs") or "")
            src = str(inp.get("pid", d))
            if not ref:
                ref = {"shape": ch.shape, "wcs": WCS(fits.getheader(wcs_f)) if wcs_f.is_file() else None,
                       "dir": Path(inp["dir"]), "src": src}
            out_ch, wp = align(ch, wcs_f, src)
            out_valid = valid if wp is None else warp(valid.astype(np.float32), wp, nearest=True) > 0.5
            if wp is not None:
                ctx.log(f"{inp['role']}: aligned on the first channel through the plate solutions")
            warps[inp["role"]] = wp
            valid_all = out_valid if valid_all is None else (valid_all & out_valid)
            chans[inp["role"]] = out_ch
            ctx.progress(0.6 * (n + 1) / len(p.inputs), f"channel {inp['role']}")
        level = None
        for role in list(chans):
            chans[role], tops[role] = balance(chans[role], valid_all, p.percentile)
            if level is None:
                level = tops[role]
            wgt = float(p.weights.get(role, 1.0))
            if wgt != 1.0:                           # the user's weight: the balance's divisor scaled with it
                chans[role] *= np.float32(wgt)
                tops[role] /= wgt
                ctx.log(f"{role}: weight {wgt:g}")
        level = np.float32(level or 1.0)
        rgb = fuse(chans, p.palette) * level
        rgb[:, ~valid_all] = 0
        H, W = ref["shape"]
        ctx.save("drz_full.npy", rgb.astype(np.float32))
        ctx.save("drz_weight.npy", np.repeat(valid_all[None].astype(np.float32), 3, axis=0))
        if starless:
            ctx.progress(0.8, "the stars")
            stars = self.star_layer(ctx, p, tops, warps, level, align)
            stars[:, ~valid_all] = 0
            ctx.save(STARS_FILE, stars.astype(np.float32))
            ctx.metric("star_source", p.stars.get("mode"))
        meta = json.loads((ref["dir"] / "prep_simple.json").read_text(encoding="utf-8")) \
            if (ref["dir"] / "prep_simple.json").is_file() else {}
        ctx.save_json("drz_meta.json", {"scale": float(meta.get("drizzle_scale", 1.0)), "source": "fusion",
                                        "palette": p.palette, "width": W, "height": H, "starless": starless,
                                        "channels": {r: round(t, 4) for r, t in tops.items()}})
        if ref["wcs"] is not None:
            ref["wcs"].to_fits().writeto(ctx.out("wcs.fits"), overwrite=True)
        ctx.metric("palette", p.palette)
        ctx.metric("channels", sorted(chans))
        ctx.log(f"{p.palette} from {', '.join(chans)}" + (" (starless)" if starless else "") + "; channel scales "
                + ", ".join(f"{r} {t:.4g}" for r, t in tops.items()))
        v = np.moveaxis(rgb, 0, -1)
        lo, hi = np.percentile(v[valid_all], 1), np.percentile(v[valid_all], 99.7)
        ctx.preview("fusion", np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.5)

    @staticmethod
    def star_layer(ctx: Context, p: FusionParams, tops, warps, level, align) -> np.ndarray:
        """The stars put back on the starless fusion (option A): the channels' stars combined with the same palette
        and scales; one channel's stars, neutral; or another workflow's stars (a short RGB or colour session: their
        own colours), aligned through its plate solution and scaled on its starless image."""
        mode = p.stars.get("mode")
        by_role = {i["role"]: i for i in p.inputs}

        def stars_of(inp) -> np.ndarray:
            d = Path(inp["stars_dir"])
            if not (d / "stars.npy").is_file():
                raise StageError(f"the {inp['role']} channel's stars are not computed")
            a = star_support(channel_of(np.asarray(np.load(d / "stars.npy", mmap_mode="r")), inp.get("channel", "L")))
            wp = warps.get(inp["role"])
            return (a if wp is None else warp(a, wp)) / np.float32(tops[inp["role"]])

        if mode == "palette":
            return fuse({r: stars_of(i) for r, i in by_role.items()}, p.palette) * level
        if mode == "channel":
            r = p.stars.get("role")
            if r not in by_role:
                raise StageError(f"no {r} channel to take the stars from")
            ctx.log(f"the stars of the {r} channel, neutral")
            return np.repeat((stars_of(by_role[r]) * level)[None], 3, axis=0)
        if mode == "workflow":
            d = Path(p.stars.get("dir") or "")
            if not (d / "stars.npy").is_file() or not (d / "starless.npy").is_file():
                raise StageError("the stars' workflow has no Split stars result: add Stars/background to it, "
                                 "and run it")
            s = np.stack([star_support(c) for c in np.asarray(np.load(d / "stars.npy"), np.float32)])
            base = np.tensordot(LUM, np.asarray(np.load(d / "starless.npy", mmap_mode="r"), np.float32), 1)
            v = np.load(d / "valid.npy")
            top = float(np.percentile(base[v], p.percentile)) if v.any() else 1.0
            wcs_f = Path(p.stars.get("wcs") or "")
            src = str(p.stars.get("pid", d))
            out = np.stack([align(s[c], wcs_f, src)[0] for c in range(3)])
            ctx.log("the stars of another workflow, in their own colours")
            return out / np.float32(top if top > 0 else 1.0) * level
        raise StageError(f"unknown star source {mode!r} (one of {list(STAR_SOURCES)})")
