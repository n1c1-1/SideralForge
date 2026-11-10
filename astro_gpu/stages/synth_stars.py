"""The Synthetic stars block (issue #77), after Stars/background: the detected stars of the stars-alone image redrawn.

Methods (``method``): ``measured`` the shape measured at the centre of the field, the brightest stars left real, for
Post's brush to replace the stars where the optics are poor; ``instrument`` the instrument's theoretical star, every
star replaced, for a missed focus or a bad tracking. It writes a stars-alone image in the Stars block's layout
(stages/synth_stars_lib.py): adding stars can only put light on the starless image, never take any from it.
"""
from __future__ import annotations

import json

import numpy as np
from pydantic import field_validator

from ..core.params import P, StageParams, check_choice
from ..core.stage import Context, Stage, StageError
from . import synth_stars_lib as lib
from .denoise import before_after

METHODS = ("measured", "instrument")
KINDS = ("newton", "refractor", "sct", "lens")
#: a camera of unknown pixel: the size does not matter, only focal length / pixel (the scale) does
DEFAULT_PIXEL_UM = 3.76


class SynthStarsParams(StageParams):
    method: str = P("measured", label="Method", choices=list(METHODS),
                    help="Image shape: the stars take the shape measured at the centre of the field. Instrument PSF: "
                         "every star takes the theoretical shape of the instrument.")
    instrument: str = P("newton", label="Type", choices=list(KINDS),
                        help="The kind of optics: it says what stands in the light's way.")
    diameter_mm: float = P(150.0, label="Diameter", unit="mm", gt=10, le=2000, step=1.0,
                           help="The diameter of the aperture.")
    obstruction: float = P(33.0, label="Obstruction", unit="%", ge=0, le=70, step=1.0,
                           help="The diameter of the central obstruction, as a share of the aperture's.")
    spider_arms: int = P(4, label="Spider arms", ge=0, le=6, help="The arms of the spider: 4 give a cross of spikes.")
    spider_mm: float = P(1.5, label="Spider width", unit="mm", ge=0.1, le=20, step=0.1,
                         help="The width of the spider's arms: wider arms make brighter spikes.")
    angle_deg: float = P(0.0, label="Spider angle", unit="°", ge=-180, le=180, step=1.0,
                         help="The direction of the spikes on the image.")
    seeing_px: float = P(2.5, label="Star width", unit="px", ge=1.0, le=8.0, step=0.1,
                         help="The width at half height of the stars drawn: the turbulence added to the diffraction.")

    @field_validator("method")
    @classmethod
    def _method(cls, v: str) -> str:
        return check_choice("method", v, METHODS)

    @field_validator("instrument")
    @classmethod
    def _kind(cls, v: str) -> str:
        return check_choice("instrument", v, KINDS)


class SynthStars(Stage):
    name = "synth_stars"
    title = "Synthetic stars"
    description = "Draws the detected stars again, with the shape measured in the image or the instrument's."
    deps = ("star_removal",)
    Params = SynthStarsParams
    version = "1"

    def run(self, ctx: Context, p: SynthStarsParams) -> None:
        s = ctx.load("star_removal", "stars.npy").astype(np.float32)
        v = ctx.load("star_removal", "valid.npy")
        valid = (v if v.ndim == 2 else v.min(0)) > 0
        ctx.progress(0.1, "drawing the stars")
        if p.method == "instrument":
            drawn, info = lib.from_the_instrument(s, valid, self.instrument(ctx, p), self.catalogue(ctx))
        else:
            drawn, info = lib.from_the_image(s, valid)
        if drawn is None:
            ctx.log("no star to measure: the stars are left as they are", "warning")
            drawn = s
        ctx.metric("stars_drawn", info.get("stars_drawn", 0))
        if "stars_kept" in info:
            ctx.metric("stars_kept", info["stars_kept"])
        if "stars_saturated" in info:
            ctx.metric("stars_saturated", info["stars_saturated"])
            ctx.metric("stars_from_catalogue", info["stars_from_catalogue"])
        ctx.save("stars.npy", np.asarray(drawn, np.float32))
        ctx.save("valid.npy", v)
        (ctx.out("prep_stars.json")).write_text(json.dumps(ctx.load_json("star_removal", "prep_stars.json")), encoding="utf-8")
        ctx.preview("before_after", before_after(s, drawn))

    @staticmethod
    def instrument(ctx: Context, p: SynthStarsParams) -> lib.Instrument:
        """The optics; the focal length and the pixel only count by their ratio, the plate scale of the astrometry."""
        scale = ctx.metric_of("astrometry", "scale_arcsec") if ctx.has("astrometry") else None
        tgt = getattr(getattr(ctx.project, "config", None), "target", None)
        pixel = getattr(tgt, "pixel_um", None) or ctx.metric_of("inventory", "pixel_um") or DEFAULT_PIXEL_UM
        focal = getattr(tgt, "focal_mm", None) or ctx.metric_of("inventory", "focal_mm")
        if scale:
            focal = float(pixel) * 206.265 / float(scale)
        if not focal:
            raise StageError("the instrument's star needs the scale of the image: place the Astrometry block")
        newton = p.instrument == "newton"
        return lib.Instrument(diameter_mm=p.diameter_mm, focal_mm=float(focal), pixel_um=float(pixel),
                              obstruction=p.obstruction / 100 if p.instrument in ("newton", "sct") else 0.0,
                              spider_arms=p.spider_arms if newton else 0, spider_mm=p.spider_mm,
                              angle_deg=p.angle_deg, seeing_px=p.seeing_px)

    @staticmethod
    def catalogue(ctx: Context) -> dict | None:
        if not ctx.has("catalogs"):
            return None
        try:
            with np.load(ctx.input("catalogs", "gaia_deep.npz")) as z:
                return {k: z[k] for k in ("x", "y", "g", "bp", "rp")}
        except Exception:  # noqa: BLE001 - only the saturated stars' light depends on it
            return None
