"""Professional H-alpha sky survey reprojected onto the stack: the absolute reference of the faint background."""
from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request

import numpy as np

from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage, StageError

SURVEYS = {
    "nsns": ("simg.de/P/NSNS/DR0_2/halpha", "NSNS DR0.2 H-alpha (northern sky, ~1')"),
    "finkbeiner": ("CDS/P/Finkbeiner", "Finkbeiner 2003 H-alpha composite (all sky, 6')"),
}


class SurveysParams(StageParams):
    survey: str = P("auto", label="H-alpha reference survey", choices=["auto", "nsns", "finkbeiner", "none"],
                    help="Professional H-alpha map used as absolute reference for the faint background: what is "
                         "faint emission (kept) and what is sky gradient or light pollution (removed).",
                    tip="'auto' uses NSNS (sharper) when it covers the field, otherwise the all-sky Finkbeiner map. "
                        "'none' for fields without emission (galaxies, clusters): the background is then modelled "
                        "with a smooth polynomial only.")
    min_coverage: float = P(0.95, label="Minimum NSNS coverage", level=ADVANCED, ge=0.0, le=1.0, step=0.05,
                            help="Below this fraction of the field covered by NSNS, 'auto' switches to Finkbeiner.")


class Surveys(Stage):
    name = "surveys"
    title = "H-alpha reference"
    description = ("Downloads a professional H-alpha survey of the field (NSNS in the north, Finkbeiner elsewhere) "
                   "and reprojects it pixel by pixel onto the stack. Your camera sees the same H-alpha plus sky "
                   "gradients; comparing both lets the background step separate real faint nebulosity from "
                   "light pollution with a physical reference instead of guessing.")
    deps = ("astrometry", "stack")
    Params = SurveysParams
    version = "1"
    network = True

    def run(self, ctx: Context, p: SurveysParams) -> None:
        from astropy.io import fits
        from astropy.wcs import WCS
        if p.survey == "none":
            ctx.save_json("survey.json", {"survey": "none"})
            ctx.metric("survey", "none")
            ctx.log("no H-alpha reference: the background will be modelled without anchoring")
            return
        wd = WCS(fits.getheader(ctx.input("astrometry", "wcs.fits")))
        meta = ctx.load_json("stack", "drz_meta.json")
        W, H = meta["width"], meta["height"]
        scale = float(ctx.metric_of("astrometry", "scale_arcsec"))
        order = ["nsns", "finkbeiner"] if p.survey == "auto" else [p.survey]
        out = None
        for k, name in enumerate(order):
            ctx.progress(0.1 + 0.4 * k, f"downloading {SURVEYS[name][1]}")
            img = fetch_reprojected(SURVEYS[name][0], wd, W, H, scale)
            cov = float(np.isfinite(img).mean())
            ctx.log(f"{SURVEYS[name][1]}: covers {100 * cov:.0f} % of the field")
            if cov >= p.min_coverage or name == order[-1]:
                out, used = img, name
                break
        if out is None or np.isfinite(out).mean() < 0.3:
            raise StageError("no H-alpha survey covers this field: set the survey to 'none'")
        ctx.save("ref_ha.npy", out.astype(np.float32))
        ctx.save_json("survey.json", {"survey": used, "title": SURVEYS[used][1]})
        ctx.metric("survey", used)
        ctx.metric("coverage", round(float(np.isfinite(out).mean()), 3))
        v = out[np.isfinite(out)]
        ctx.metric("ha_p50_p99", [round(float(np.percentile(v, 50)), 2), round(float(np.percentile(v, 99)), 2)])
        sm = np.nan_to_num(out[::4, ::4])
        lo, hi = np.percentile(sm, [1, 99.5])
        ctx.preview("halpha_reference", np.clip((sm - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.5)


def fetch_reprojected(hips: str, wd, W: int, H: int, scale: float) -> np.ndarray:
    """hips2fits cut-out on a TAN grid 3x coarser than the stack, then bilinear reprojection onto the stack."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from reproject import reproject_interp
    c = wd.pixel_to_world(W / 2, H / 2)
    cs = 3 * scale / 3600.0
    side = float(np.hypot(W, H) * scale / 3600.0 * 1.05)
    n = int(np.ceil(side / cs))
    hdr = {"NAXIS": 2, "NAXIS1": n, "NAXIS2": n, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
           "CRVAL1": float(c.ra.deg), "CRVAL2": float(c.dec.deg), "CRPIX1": n / 2 + 0.5, "CRPIX2": n / 2 + 0.5,
           "CDELT1": -cs, "CDELT2": cs}
    url = ("https://alasky.cds.unistra.fr/hips-image-services/hips2fits?hips=" + urllib.parse.quote(hips, safe="")
           + f"&width={n}&height={n}&wcs=" + urllib.parse.quote(json.dumps(hdr)) + "&format=fits")
    with urllib.request.urlopen(url, timeout=900) as r:
        data = r.read()
    img = fits.open(io.BytesIO(data))[0].data.astype(np.float32)
    out, foot = reproject_interp((img, WCS(hdr)), wd, shape_out=(H, W), order="bilinear")
    return np.where(foot > 0, out, np.nan).astype(np.float32)
