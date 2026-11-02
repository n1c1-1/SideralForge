"""Where the bench measures the target's signal-to-noise: regions chosen on an image independent of the stacks compared.

No side effect on import (numpy and scipy only), so that the tests can import it alone (tests/test_bench_regions.py).

The defect it replaces: the target (the brightest extended region) and its nearby sky used to be chosen box by box on
Siril's own image, then every stack was measured on those boxes. A choice made on a noisy image keeps the boxes where
that image's noise happens to fall well (a brighter nebula, a darker sky), so the image that chose is favoured and the
others are not: circular analysis. On Iris it put our classic stack 9 % under Siril's, while the same measure with
the choice made on our stack put it 13 % over, and with a neutral choice the two were equal.

The published method:
- SExtractor's dual-image mode (Bertin & Arnouts 1996, A&AS 117, 393): detect on one image, measure on another;
- the rule against circular ("double dipping") analysis (Kriegeskorte et al. 2009, Nature Neuroscience 12, 535): a
  region selected on some data must not be used to measure those same data; select on independent data.
So the regions are chosen on an independent image (the case's Base stack of 100 frames, other frames than the Express
stacks'), or, when there is none, on the mean of all the stacks compared (symmetric: no stack is favoured, and the
order of the list does not matter); then every stack is measured on those same regions. The nearby sky is the whole
band around the target (not its darker half, which is again a choice made on the noise).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from aperture import rstd
from scipy import ndimage as ndi

GOOD = 0.6          # a box counts when at least this share of its pixels is in the common sky and not masked


def boxes(L: np.ndarray, ok: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of each b x b box over its usable pixels, and the share of usable pixels."""
    H, W = L.shape
    hb, wb = H // b, W // b
    Lc = L[: hb * b, : wb * b].reshape(hb, b, wb, b)
    oc = ok[: hb * b, : wb * b].reshape(hb, b, wb, b)
    n = oc.sum((1, 3))
    s = np.where(oc, Lc, 0).sum((1, 3), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / n).astype(np.float32), n / float(b * b)


@dataclass
class BoxMap:
    """The box means of one image (boxes of ``bo`` pixels of its own grid) and where they are on the sky.

    ``v`` box means, ``f`` share of usable pixels, ``wcs`` the image's (all_pix2world / all_world2pix, origin 0),
    ``dx``/``dy`` its measured shift against the catalogue, ``pixarea`` arcsec2 of a pixel, ``bo_exact`` the exact
    box size in pixels (``bo`` is its rounding), ``zp`` the photometric zero point (for surface brightnesses)."""
    v: np.ndarray
    f: np.ndarray
    wcs: object
    dx: float
    dy: float
    pixarea: float
    bo: int
    bo_exact: float
    zp: float = float("nan")

    def centres_sky(self):
        """Sky position (ra, dec) of the centre of every box, flattened."""
        hb, wb = self.v.shape
        cy, cx = np.mgrid[0:hb, 0:wb] * self.bo + (self.bo - 1) / 2.0
        ra, dec = self.wcs.all_pix2world((cx - self.dx).ravel(), (cy - self.dy).ravel(), 0)
        return ra, dec

    def box_coords(self, ra, dec):
        """Fractional box coordinates (row, column; a box centre is an integer) of sky positions."""
        x, y = self.wcs.all_world2pix(np.asarray(ra, float), np.asarray(dec, float), 0)
        x, y = x + self.dx, y + self.dy
        return (y + 0.5) / self.bo - 0.5, (x + 0.5) / self.bo - 0.5


def local_noise(v: np.ndarray, t: np.ndarray) -> float:
    """Spread of the boxes of the region ``t`` around the median of their 3 x 3 neighbours (the correction is the
    share of the neighbours' own noise)."""
    med = ndi.median_filter(np.nan_to_num(v), 3, mode="nearest")
    return rstd((v - med)[t]) / np.sqrt(1 + 1.5 / 9)


def poly2_removed(v: np.ndarray, good: np.ndarray, iters: int = 4, dark: float = 0.7) -> np.ndarray:
    """The map minus a degree-2 polynomial fitted to the darker ``dark`` share of the good boxes (the nebula and the
    target are the bright ones; refitted ``iters`` times on the darker share of the residuals): the usual robust
    background of a gradient-removed image, as the bench's Siril reference (a degree-2 polynomial) does."""
    h, w = v.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x, y = xx / max(w, 1) - 0.5, yy / max(h, 1) - 0.5
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)
    ok = good & np.isfinite(v)
    if ok.sum() < 50:
        return v
    use = ok & (v <= np.percentile(v[ok], 100 * dark))
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A[use], v[use], rcond=None)
        res = v - A @ c
        use = ok & (res <= np.percentile(res[ok], 100 * dark))
    return v - A @ c


def sky_plane(v: np.ndarray, sky: np.ndarray, iters: int = 3) -> np.ndarray:
    """A plane (a + b x + c y) fitted to the values of the sky boxes (3-sigma clipped over a few iterations), over the
    whole map."""
    h, w = v.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.stack([np.ones((h, w)), xx / max(w, 1) - 0.5, yy / max(h, 1) - 0.5], -1)
    use = sky & np.isfinite(v)
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A[use], v[use], rcond=None)
        res = v - A @ c
        s = rstd(res[sky & np.isfinite(v)])
        use = sky & np.isfinite(v) & (np.abs(res) < 3 * s)
    return A @ c


def select_regions(v: np.ndarray, f: np.ndarray) -> dict | None:
    """Target and sky boxes from the box means ``v`` of ONE image (its gradient removed): the target is the brightest
    extended region (the largest connected parts of the top 5 % of the smoothed box map, at most 5 % of the field),
    the sky the whole band of boxes around it (the darkest tenth of the field if the band is too small)."""
    good = (f > GOOD) & np.isfinite(v)
    if good.sum() < 200:
        return None
    vg = np.where(good, v, np.nanmedian(v[good]))
    noise = rstd((v - ndi.median_filter(np.nan_to_num(v), 3, mode="nearest"))[good])
    sm = ndi.median_filter(vg, 9, mode="nearest")
    thr = np.percentile(sm[good], 95)
    lab, n = ndi.label((sm >= thr) & good)
    if n == 0:
        return None
    sizes = ndi.sum_labels(np.ones_like(sm), lab, np.arange(1, n + 1))
    tgt = np.zeros_like(good)
    for k in np.argsort(sizes)[::-1]:
        if tgt.sum() >= 0.6 * sizes.sum() or tgt.sum() + sizes[k] > 0.05 * good.sum() and tgt.any():
            break
        tgt |= lab == k + 1
    tgt &= good
    if tgt.sum() < 20:
        return None
    r2 = max(8, int(0.04 * max(v.shape)))
    band = ndi.binary_dilation(tgt, iterations=r2) & ~ndi.binary_dilation(tgt, iterations=3) & good
    sky = band if band.sum() >= 50 else good & (v <= np.nanpercentile(v[good], 10))
    sky_level = np.nanmedian(v[sky])
    return {"target": tgt, "sky": sky, "fraction": float(tgt.sum() / good.sum()),
            "level_sigma": float(np.median((v[tgt] - sky_level) / noise))}


def resampled(m: BoxMap, ref: BoxMap) -> tuple[np.ndarray, np.ndarray]:
    """The box means of ``m`` seen on the boxes of ``ref`` (bilinear), in ref's pixel units (the images are on one
    photometric scale: a pixel holds a flux in proportion to its area). Also the mask of valid boxes."""
    ra, dec = ref.centres_sky()
    r, c = m.box_coords(ra, dec)
    ok = (m.f > GOOD) & np.isfinite(m.v)
    val = np.where(ok, m.v, 0.0).astype(np.float64)
    w = ok.astype(np.float64)
    vs = ndi.map_coordinates(val, [r, c], order=1, mode="constant", cval=0.0)
    ws = ndi.map_coordinates(w, [r, c], order=1, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (vs / ws) * (ref.pixarea / m.pixarea)
    return out.reshape(ref.v.shape), (ws > 0.999).reshape(ref.v.shape)


def mean_map(maps: dict[str, BoxMap], ref: str) -> BoxMap:
    """The mean of all the stacks compared, on ref's boxes, where all have data, gradient removed. Symmetric: the
    order of the list does not change it (up to the choice of the grid ``ref``)."""
    base = maps[ref]
    acc, ok = [], np.ones(base.v.shape, bool)
    for m in maps.values():
        if m is base:
            a, valid = base.v.astype(np.float64), (base.f > GOOD) & np.isfinite(base.v)
        else:
            a, valid = resampled(m, base)
        acc.append(a)
        ok &= valid
    mean = np.mean(acc, 0)
    v = poly2_removed(np.where(ok, mean, np.nan), ok).astype(np.float32)
    return BoxMap(v, np.where(ok, 1.0, 0.0), base.wcs, base.dx, base.dy, base.pixarea, base.bo, base.bo_exact, base.zp)


def choose(maps: dict[str, BoxMap], ref: str, independent: BoxMap | None = None) -> tuple[dict | None, BoxMap, str]:
    """The regions, the map they were chosen on and where it came from ("independent" or "mean")."""
    if independent is not None:
        sel, src = independent, "independent"
    else:
        sel, src = mean_map(maps, ref), "mean"
    return select_regions(sel.v, sel.f), sel, src


def on_regions(reg: dict, sel: BoxMap, m: BoxMap, plane: bool = True) -> dict:
    """Signal (median of the target boxes minus median of the sky boxes), noise (spread of the target boxes around
    their neighbours) and their ratio for the stack ``m``, on the regions chosen on ``sel``. Each box of m takes the
    status of the box of sel its centre falls in. ``plane``: the sky under the target is a plane fitted to the
    sky boxes (else their median)."""
    ra, dec = m.centres_sky()
    r, c = sel.box_coords(ra, dec)
    ri, ci = np.floor(r + 0.5).astype(int), np.floor(c + 0.5).astype(int)
    hs, ws = sel.v.shape
    inb = (ri >= 0) & (ci >= 0) & (ri < hs) & (ci < ws)
    to, ko = np.zeros(ri.shape, bool), np.zeros(ri.shape, bool)
    to[inb], ko[inb] = reg["target"][ri[inb], ci[inb]], reg["sky"][ri[inb], ci[inb]]
    ok = (m.f > GOOD) & np.isfinite(m.v)
    to, ko = to.reshape(m.v.shape) & ok, ko.reshape(m.v.shape) & ok
    out = {"target_boxes": int(to.sum()), "sky_boxes": int(ko.sum())}
    if to.sum() < 20 or ko.sum() < 20:
        return out
    # the sky under the target is a plane fitted to the nearby sky boxes of this stack (a raw stack keeps a smooth
    # gradient of its own: flats, sky glow; with a plain median of the band, a different gradient in each stack moves
    # the signal with the exact region, Heart: 0.98 or 0.92 for the same pair). A local model of the background around
    # an extended source, as SExtractor's background map and the sky annulus of aperture photometry (DAOPHOT) do; a
    # global polynomial would eat the extended nebula itself (M51)
    sig = float(np.median((m.v - sky_plane(m.v, ko))[to])) if plane else float(np.median(m.v[to]) - np.median(m.v[ko]))
    noi = local_noise(m.v, to) * (m.bo / m.bo_exact)         # same angular box: white-noise scaling of the rounding
    out.update({"snr": sig / noi, "signal": sig, "noise": noi})
    if np.isfinite(m.zp) and sig > 0:
        out["signal_sb"] = float(m.zp - 2.5 * np.log10(sig / m.pixarea))
        out["noise_sb"] = float(m.zp - 2.5 * np.log10(noi / m.pixarea))
    return out


# ---------------------------------------------------------------------------------------------- saved maps

def save_maps(path, maps: dict[str, BoxMap], sel: BoxMap | None, source: str) -> None:
    """The box maps of a measure (small arrays), so that the target step can be redone without reading the images."""
    import json
    arrays, meta = {}, {"source": source, "names": list(maps), "maps": {}}
    items = dict(maps)
    if sel is not None:
        items["__selection__"] = sel
    for n, m in items.items():
        arrays[f"{n}.v"], arrays[f"{n}.f"] = m.v, m.f
        meta["maps"][n] = {"hdr": m.wcs.to_header(relax=True).tostring(), "dx": m.dx, "dy": m.dy,
                           "pixarea": m.pixarea, "bo": m.bo, "bo_exact": m.bo_exact, "zp": m.zp}
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_maps(path) -> tuple[dict[str, BoxMap], BoxMap | None, str]:
    import json

    from astropy.wcs import WCS
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    out = {}
    for n, d in meta["maps"].items():
        out[n] = BoxMap(z[f"{n}.v"], z[f"{n}.f"], WCS(d["hdr"], naxis=2), d["dx"], d["dy"], d["pixarea"], d["bo"],
                        d["bo_exact"], d["zp"])
    sel = out.pop("__selection__", None)
    return out, sel, meta["source"]
