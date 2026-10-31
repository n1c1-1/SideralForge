"""Professional H-alpha sky survey reprojected onto the stack: the absolute reference of the faint background."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage, StageError

SURVEYS = {
    "nsns": ("simg.de/P/NSNS/DR0_2/halpha", "NSNS DR0.2 H-alpha (northern sky, ~1')"),
    "finkbeiner": ("CDS/P/Finkbeiner", "Finkbeiner 2003 H-alpha composite (all sky, 6')"),
}
#: the survey's own HiPS server, read tile by tile when the CDS cut-out service does not answer (2026-10-30)
HIPS_SERVERS = {"nsns": "https://simg.de/nebulae3/dr0_2/halpha"}


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
            img = fetch_survey(name, wd, W, H, scale, ctx.log)
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


def fetch_survey(name: str, wd, W: int, H: int, scale: float, log=print) -> np.ndarray:
    """The survey on the stack's grid: from the CDS cut-out service, else from the survey's own HiPS server."""
    try:
        return fetch_reprojected(SURVEYS[name][0], wd, W, H, scale)
    except Exception as e:  # noqa: BLE001 - CDS unreachable: the survey's own server when it has one
        if name not in HIPS_SERVERS:
            raise
        log(f"the CDS cut-out service does not answer ({str(e)[:120]}): tiles read from {HIPS_SERVERS[name]}",
            "warning")
        return fetch_hips_tiles(HIPS_SERVERS[name], wd, W, H, scale)


def fetch_hips_tiles(url: str, wd, W: int, H: int, scale: float) -> np.ndarray:
    """The HiPS read tile by tile from its server, at the coarsest level still finer than 3 stack pixels (as the
    cut-out), sampled every 3 stack pixels then interpolated onto the stack. Within a tile, the HEALPix nested index
    of a pixel splits into x (even bits) and y (odd bits), and the value sits at row tile-1-x, column y: checked on the
    NSNS maps the CDS cut-out gave the Iris and the Heart (correlation 0.994 and 0.998, levels within 1 %)."""
    import cv2
    from astropy.coordinates import ICRS
    from astropy.io import fits
    from astropy.utils.data import download_file
    from astropy_healpix import HEALPix
    props = _hips_properties(url)
    T, deepest = int(props["hips_tile_width"]), int(props["hips_order"])
    bits = int(round(np.log2(T)))
    pix_deg = 58.6323 / (T * 2 ** deepest)                     # mean HEALPix pixel size at the deepest level
    level = max(0, deepest - int(max(0, np.floor(np.log2(3 * scale / 3600.0 / pix_deg)))))
    yy, xx = np.mgrid[0:H:3, 0:W:3]
    hp = HEALPix(nside=T * 2 ** level, order="nested", frame=ICRS())
    ip = hp.skycoord_to_healpix(wd.pixel_to_world(xx.ravel(), yy.ravel())).astype(np.int64)
    tile, sub = ip >> (2 * bits), ip & (T * T - 1)
    x, y = _compact(sub), _compact(sub >> 1)
    out = np.full(ip.shape, np.nan, np.float32)
    for t in np.unique(tile):
        u = f"{url}/Norder{level}/Dir{(t // 10000) * 10000}/Npix{t}.fits"
        try:
            d = fits.getdata(download_file(u, cache=True)).astype(np.float32)
        except urllib.error.HTTPError:                       # no tile there: outside the survey
            continue
        m = tile == t
        out[m] = d[T - 1 - x[m], y[m]]
    small = out.reshape(yy.shape)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) if small.shape != (H, W) else small


def _hips_properties(url: str) -> dict:
    with urllib.request.urlopen(f"{url}/properties", timeout=60) as r:
        lines = r.read().decode("utf-8", "replace").splitlines()
    return {k.strip(): v.strip() for k, _, v in (ln.partition("=") for ln in lines if "=" in ln and not
                                                  ln.lstrip().startswith("#"))}


def _compact(v: np.ndarray) -> np.ndarray:
    """Every other bit of v (bits 0, 2, 4...) packed together: one coordinate of a nested HEALPix index."""
    v = v & 0x5555555555555555
    v = (v | (v >> 1)) & 0x3333333333333333
    v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FF
    v = (v | (v >> 8)) & 0x0000FFFF0000FFFF
    return (v | (v >> 16)) & 0x00000000FFFFFFFF


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
