"""Professional H-alpha sky survey reprojected onto the stack: the absolute reference of the faint background."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from ..core import online, skycache
from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage, StageError

SURVEYS = {
    "nsns": ("simg.de/P/NSNS/DR0_2/halpha", "NSNS DR0.2 H-alpha (northern sky, ~1')"),
    "finkbeiner": ("CDS/P/Finkbeiner", "Finkbeiner 2003 H-alpha composite (all sky, 6')"),
}
#: the survey's own HiPS server, read tile by tile when the CDS cut-out service does not answer
HIPS_SERVERS = {"nsns": "https://simg.de/nebulae3/dr0_2/halpha"}


class SurveysParams(StageParams):
    survey: str = P("auto", label="H-alpha survey", choices=["auto", "nsns", "finkbeiner", "none"],
                    help="Professional H-alpha map of the field: it shows where the faint nebulosity is, so that "
                         "the Gradient block takes no sky sample there. It never enters the sky model.",
                    tip="'Automatic' uses NSNS (sharper) when it covers the field, otherwise the all-sky Finkbeiner map. "
                        "'None': no survey (the Gradient block then chooses its samples on the image alone).")
    min_coverage: float = P(0.95, label="NSNS coverage", level=ADVANCED, ge=0.0, le=1.0, step=0.05,
                            help="Below this fraction of the field covered by NSNS, 'auto' switches to Finkbeiner.")


class Surveys(Stage):
    name = "surveys"
    title = "H-alpha reference"
    description = ("Downloads once a professional H-alpha survey of the field (NSNS in the north, Finkbeiner "
                   "elsewhere), kept on disk, and reprojects it onto the stack. It shows where the faint nebulosity "
                   "is: the Gradient block takes no sky sample there, so the faint extensions of a nebula are not "
                   "removed as sky. No survey (no network, or none covers the field): the run goes on without it.")
    deps = ("astrometry", "stack")
    Params = SurveysParams
    version = "2"          # 2: never stops the run: no survey, or no network, gives 'none' and the reason
    network = True

    def run(self, ctx: Context, p: SurveysParams) -> None:
        from astropy.io import fits
        from astropy.wcs import WCS
        if p.survey == "none":
            ctx.save_json("survey.json", {"survey": "none"})
            ctx.metric("survey", "none")
            ctx.log("no H-alpha survey (set to none): the Gradient block chooses its samples on the image alone")
            return
        wd = WCS(fits.getheader(ctx.input("astrometry", "wcs.fits")))
        meta = ctx.load_json("stack", "drz_meta.json")
        W, H = meta["width"], meta["height"]
        scale = float(ctx.metric_of("astrometry", "scale_arcsec"))
        try:
            out, used = survey_on_grid(wd, W, H, scale, p.survey, p.min_coverage, ctx.log,
                                       lambda f, m: ctx.progress(0.1 + 0.8 * f, m))
        except Exception as e:  # noqa: BLE001 - a guide only: the run goes on without it, the reason said
            why = (e.why if isinstance(e, online.ServiceDown) else
                   "no H-alpha survey covers this field" if isinstance(e, StageError) else f"not available ({e})")
            ctx.save_json("survey.json", {"survey": "none", "why": str(why)[:300]})
            ctx.metric("survey", "none")
            ctx.log(f"no H-alpha survey: {why}; the Gradient block chooses its samples on the image alone", "warning")
            return
        ctx.save("ref_ha.npy", out.astype(np.float32))
        ctx.save_json("survey.json", {"survey": used, "title": SURVEYS[used][1]})
        ctx.metric("survey", used)
        ctx.metric("coverage", round(float(np.isfinite(out).mean()), 3))
        v = out[np.isfinite(out)]
        ctx.metric("ha_p50_p99", [round(float(np.percentile(v, 50)), 2), round(float(np.percentile(v, 99)), 2)])
        sm = np.nan_to_num(out[::4, ::4])
        lo, hi = np.percentile(sm, [1, 99.5])
        ctx.preview("halpha_reference", np.clip((sm - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.5)


def survey_on_grid(wd, W: int, H: int, scale: float, which: str = "auto", min_coverage: float = 0.95, log=print,
                   progress=None) -> tuple[np.ndarray, str]:
    """The H-alpha survey of the field on a W x H grid of WCS ``wd`` (``scale`` arcsec per pixel), and the survey's
    name: with ``auto``, NSNS when it covers ``min_coverage`` of the field, else Finkbeiner. NaN outside the survey."""
    order = ["nsns", "finkbeiner"] if which == "auto" else [which]
    for k, name in enumerate(order):
        if progress is not None:
            progress(k / len(order), f"downloading {SURVEYS[name][1]}")
        img = fetch_survey(name, wd, W, H, scale, log)
        cov = float(np.isfinite(img).mean())
        log(f"{SURVEYS[name][1]}: covers {100 * cov:.0f} % of the field")
        if cov >= min_coverage or name == order[-1]:
            if cov < 0.3:
                break
            return img, name
    raise StageError("no H-alpha survey covers this field: set the survey to 'none'")


#: the survey's levels taken as its sky and as its bright emission (percentiles of the field)
LEVEL_PERCENTILES = (1.0, 99.5)


def emission_level(img: np.ndarray) -> np.ndarray:
    """The survey scaled so that its sky is 0 and its bright emission 1 (``LEVEL_PERCENTILES`` of the field), NaN
    outside it: the same reading on every survey and every field."""
    v = img[np.isfinite(img)]
    if v.size == 0:
        return np.full(img.shape, np.nan, np.float32)
    lo, hi = np.percentile(v, LEVEL_PERCENTILES)
    return ((img - lo) / max(float(hi - lo), 1e-9)).astype(np.float32)


#: a survey sees nebulosity when its range (1st to 99.5th percentile) is this many times its noise (a field without
#: nebulosity: the survey's noise alone, about 5 times)
EMISSION_CONTRAST = 12.0


def has_emission(img: np.ndarray) -> bool:
    """Whether the survey (any scale: raw or emission_level) shows nebulosity rather than its own noise. The noise is
    measured between pixels a few apart (the survey is smooth at its own resolution, so neighbours would understate
    it): the robust spread of the differences, over the square root of 2."""
    a = np.asarray(img, np.float32)
    v = a[np.isfinite(a)]
    if v.size < 100:
        return False
    s = max(2, min(a.shape) // 128)
    d = (a[:, s:] - a[:, :-s]).ravel()
    d = d[np.isfinite(d)]
    if d.size < 100:
        return False
    noise = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2)
    lo, hi = np.percentile(v, LEVEL_PERCENTILES)
    return float(hi - lo) > EMISSION_CONTRAST * max(noise, 1e-12)


def fetch_survey(name: str, wd, W: int, H: int, scale: float, log=print) -> np.ndarray:
    """The survey on the stack's grid. A survey with its own HiPS server is read tile by tile, each tile cached under
    ASTRO_GPU_HOME (issue #39: any field of a sky already seen, by any project, then works offline; the cut-out of the
    CDS service was cached for its exact grid only, so that a new stack of the same field downloaded again); the
    cut-out service when the tiles cannot be had (and for the surveys without a server of their own)."""
    if name in HIPS_SERVERS:
        try:
            return fetch_hips_tiles(HIPS_SERVERS[name], wd, W, H, scale)
        except Exception as e:  # noqa: BLE001 - the survey's server unreachable: the CDS cut-out (cached by grid)
            why = e.why if isinstance(e, online.ServiceDown) else f"does not answer ({str(e)[:120]})"
            log(f"the survey's own server {why}: the CDS cut-out service is asked", "warning")
    return fetch_reprojected(SURVEYS[name][0], wd, W, H, scale)


def prefetch_field(roots, which: str = "nsns", progress=None) -> dict:
    """The survey's tiles of the solved fields of these projects fetched into the cache (Settings: Download for
    offline use, issue #39); returns {"fields": n solved, "tiles": n now in the cache, "failed": [why...]}."""
    import json as _json

    from astropy.io import fits
    from astropy.wcs import WCS
    out = {"fields": 0, "tiles": 0, "failed": []}
    url = HIPS_SERVERS.get(which)
    if url is None:
        out["failed"].append(f"{which} has no tile server")
        return out
    roots = list(roots)
    for k, root in enumerate(roots):
        if progress is not None:
            progress(k / max(len(roots), 1), str(root))
        store = Path(root) / "store"
        sizes = []                                       # the stacks' grids (the solution's header holds none)
        for m in store.glob("stack/*/drz_meta.json"):
            try:
                d = _json.loads(m.read_text(encoding="utf-8"))
                sizes.append((int(d["width"]), int(d["height"])))
            except (OSError, ValueError, KeyError):
                continue
        for wcsf in sorted(store.glob("astrometry/*/wcs.fits")):
            try:
                wd = WCS(fits.getheader(wcsf))
                man = _json.loads((wcsf.parent / "manifest.json").read_text(encoding="utf-8"))
                scale = float((man.get("metrics") or {}).get("scale_arcsec") or 0) or None
                if scale is None or not sizes:
                    continue
                W, H = max(sizes, key=lambda wh: wh[0] * wh[1])   # the largest: the tiles cover the others too
                fetch_hips_tiles(url, wd, W, H, scale)
                out["fields"] += 1
            except Exception as e:  # noqa: BLE001 - one field that cannot be read: the others still come
                out["failed"].append(f"{Path(root).name}: {str(e)[:120]}")
    out["tiles"] = len(list(skycache.cache_dir().glob("hips_tiles/*.fits")))
    return out


def _get(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _cached_url(url: str, sub: str, service: str, budget: float) -> bytes | None:
    """The bytes at ``url``, from the cache (``cache/<sub>/``; astropy's download cache, where earlier versions put
    the tiles, is read too) else downloaded and cached; None when the server says the file does not exist (a tile
    outside the survey, remembered as such)."""
    import hashlib
    rel = f"{sub}/{hashlib.sha1(url.encode()).hexdigest()}"
    p = skycache.find(rel + ".fits")
    if p is not None:
        return p.read_bytes()
    if skycache.find(rel + ".absent") is not None:
        return None
    try:
        from astropy.utils.data import download_file, is_url_in_cache
        if is_url_in_cache(url):
            return open(download_file(url, cache=True), "rb").read()
    except Exception:  # noqa: BLE001 - astropy's cache unreadable: download
        pass
    try:
        data = online.call(service, lambda: _get(url), budget)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            dst = skycache.cache_dir() / (rel + ".absent")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            return None
        raise
    dst = skycache.cache_dir() / (rel + ".fits")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(dst)
    return data


def fetch_hips_tiles(url: str, wd, W: int, H: int, scale: float) -> np.ndarray:
    """The HiPS read tile by tile from its server, at the coarsest level still finer than 3 stack pixels (as the
    cut-out), sampled every 3 stack pixels then interpolated onto the stack. Within a tile, the HEALPix nested index
    of a pixel splits into x (even bits) and y (odd bits), and the value sits at row tile-1-x, column y: checked on the
    NSNS maps the CDS cut-out gave the Iris and the Heart (correlation 0.994 and 0.998, levels within 1 %)."""
    import cv2
    from astropy.coordinates import ICRS
    from astropy.io import fits
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
        raw = _cached_url(u, "hips_tiles", "hips", 120.0)
        if raw is None:                                      # no tile there: outside the survey
            continue
        d = fits.getdata(io.BytesIO(raw)).astype(np.float32)
        m = tile == t
        out[m] = d[T - 1 - x[m], y[m]]
    small = out.reshape(yy.shape)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) if small.shape != (H, W) else small


def _hips_properties(url: str) -> dict:
    raw = _cached_url(f"{url}/properties", "hips_tiles", "hips", 60.0)
    if raw is None:
        raise StageError(f"{url} has no HiPS properties")
    lines = raw.decode("utf-8", "replace").splitlines()
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
    data = _cached_url(url, "hips2fits", "hips2fits", 300.0)
    if data is None:
        raise StageError("the CDS cut-out service has no image for this field")
    img = fits.open(io.BytesIO(data))[0].data.astype(np.float32)
    out, foot = reproject_interp((img, WCS(hdr)), wd, shape_out=(H, W), order="bilinear")
    return np.where(foot > 0, out, np.nan).astype(np.float32)
