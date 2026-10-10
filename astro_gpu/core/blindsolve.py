"""All-sky blind plate solving with geometric hashing of star quads (the principle of astrometry.net).

Index: every Hipparcos star brighter than V 9.5 with its 8 nearest neighbours gives C(8,3) = 56 quads; a quad of
4 stars is coded by the positions of its two inner stars in the frame where its two most distant stars are (0,0)
and (1,1) - a 4-number code invariant to translation, rotation and scale. The image does the same with its
brightest stars; matching codes give hypotheses (4 star pairs -> similarity transform) that are verified by
counting how many other image stars land on catalogue stars. Built for wide fields (> ~2 deg) where enough
bright stars are present; the index is built once and cached.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CACHE = Path.home() / ".astro_gpu" / "cache"


def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def _tangent(xyz, center):
    """Gnomonic projection of unit vectors around ``center`` (unit vector): (..., 2) in radians."""
    c = center / np.linalg.norm(center)
    east = np.cross([0.0, 0.0, 1.0], c)
    if np.linalg.norm(east) < 1e-9:
        east = np.array([1.0, 0.0, 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(c, east)
    d = xyz @ c
    return np.stack([(xyz @ east) / d, (xyz @ north) / d], -1)


def quad_codes(P: np.ndarray):
    """P (M, 4, 2) points -> codes (M, 4), the canonical order of the 4 points (M, 4), validity mask (M,)."""
    M = len(P)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    d = np.stack([np.hypot(*(P[:, a] - P[:, b]).T) for a, b in pairs], 1)
    best = np.argmax(d, 1)
    order = np.zeros((M, 4), int)
    for k, (a, b) in enumerate(pairs):
        rest = [i for i in range(4) if i not in (a, b)]
        sel = best == k
        order[sel] = [a, b, rest[0], rest[1]]
    Q = np.take_along_axis(P, order[:, :, None].repeat(2, 2), 1)
    A, B = Q[:, 0], Q[:, 1]
    # similarity mapping A -> (0,0), B -> (1,1)
    v = B - A
    zA = A[:, 0] + 1j * A[:, 1]
    zv = v[:, 0] + 1j * v[:, 1]
    f = (1 + 1j) / zv

    def tr(p):
        z = (p[:, 0] + 1j * p[:, 1] - zA) * f
        return np.stack([z.real, z.imag], 1)
    C, D = tr(Q[:, 2]), tr(Q[:, 3])
    ok = (np.hypot(C[:, 0] - 0.5, C[:, 1] - 0.5) < 0.7072) & (np.hypot(D[:, 0] - 0.5, D[:, 1] - 0.5) < 0.7072)
    # canonical symmetries: xC <= xD (swap C, D), then xC + xD <= 1 (swap A, B: (x, y) -> (1 - x, 1 - y))
    sw = C[:, 0] > D[:, 0]
    C[sw], D[sw] = D[sw].copy(), C[sw].copy()
    order[sw, 2], order[sw, 3] = order[sw, 3].copy(), order[sw, 2].copy()
    fl = (C[:, 0] + D[:, 0]) > 1
    C[fl], D[fl] = 1 - C[fl], 1 - D[fl]
    order[fl, 0], order[fl, 1] = order[fl, 1].copy(), order[fl, 0].copy()
    sw2 = fl & (C[:, 0] > D[:, 0])
    C[sw2], D[sw2] = D[sw2].copy(), C[sw2].copy()
    order[sw2, 2], order[sw2, 3] = order[sw2, 3].copy(), order[sw2, 2].copy()
    return np.concatenate([C, D], 1), order, ok


class SkyIndex:
    """Quad index of a bright-star catalogue: 'hip' (Hipparcos V < 9.5, fields > ~4 deg) or 'tyc' (Tycho-2
    VT < 11, fields of ~1-6 deg)."""

    def __init__(self, catalog: str = "hip", k: int | None = None):
        self.k = k or (8 if catalog == "hip" else 7)
        self.combos = np.array(list(combinations(range(self.k), 3)))
        p = CACHE / f"quads_{catalog}_k{self.k}.npz"
        if p.exists():
            d = np.load(p)
            self.ra, self.dec, self.codes, self.quads = d["ra"], d["dec"], d["codes"], d["quads"]
        else:
            self.ra, self.dec = _hipparcos() if catalog == "hip" else _tycho()
            self.codes, self.quads = self._build()
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, ra=self.ra, dec=self.dec, codes=self.codes, quads=self.quads)
        self.catalog = catalog
        self.xyz = _unit(self.ra, self.dec)
        self.star_tree = cKDTree(self.xyz)
        self.code_tree = cKDTree(self.codes)

    def _build(self):
        xyz = _unit(self.ra, self.dec)
        _, nb = cKDTree(xyz).query(xyz, self.k + 1)
        codes, quads = [], []
        for c3 in self.combos:
            idx = np.c_[nb[:, 0], nb[:, 1 + c3]]                      # (N, 4) star indices
            P = np.stack([_tangent_rows(xyz[idx[:, j]], xyz[idx[:, 0]]) for j in range(4)], 1)
            code, order, ok = quad_codes(P)
            codes.append(code[ok].astype(np.float32))
            quads.append(np.take_along_axis(idx, order, 1)[ok].astype(np.int32))
        return np.concatenate(codes), np.concatenate(quads)


def _tangent_rows(xyz, centers):
    """Row-wise gnomonic projection of xyz[i] around centers[i]."""
    c = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    z = np.array([0.0, 0.0, 1.0])
    east = np.cross(z, c)
    bad = np.linalg.norm(east, axis=1) < 1e-9
    east[bad] = [1.0, 0.0, 0.0]
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(c, east)
    d = (xyz * c).sum(1)
    return np.stack([(xyz * east).sum(1) / d, (xyz * north).sum(1) / d], 1)


def _hipparcos(vmax=9.5):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["RAICRS", "DEICRS"], row_limit=-1, timeout=600)
    t = v.query_constraints(catalog="I/239/hip_main", Vmag=f"<{vmax}")[0]
    ra, dec = np.asarray(t["RAICRS"], float), np.asarray(t["DEICRS"], float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return ra[ok], dec[ok]


def _tycho(vmax=11.0):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["RAmdeg", "DEmdeg"], row_limit=-1, timeout=900)
    t = v.query_constraints(catalog="I/259/tyc2", VTmag=f"<{vmax}")[0]
    ra, dec = np.asarray(t["RAmdeg"], float), np.asarray(t["DEmdeg"], float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return ra[ok], dec[ok]


def solve(xy: np.ndarray, width: int, height: int, index: SkyIndex | None = None, n_stars: int = 60,
          tol: float = 0.012, min_matches: int = 10, progress=None) -> dict | None:
    """Blind solve. ``xy`` (N, 2): detected stars, brightest first, in image pixels.

    Returns {"ra", "dec" (field centre), "scale" (arcsec/px), "rotation" (deg), "mirror", "matches"} or None.
    """
    index = index or SkyIndex("hip")
    best = None
    for mirror in (False, True):
        pts = xy[:n_stars].astype(np.float64).copy()
        if mirror:
            pts[:, 1] = height - 1 - pts[:, 1]
        tree = cKDTree(pts)
        _, nb = tree.query(pts, min(index.k + 1, len(pts)))
        if nb.shape[1] < index.k + 1:
            return None
        res = []
        for c3 in index.combos:
            idx = np.c_[nb[:, 0], nb[:, 1 + c3]]
            code, order, ok = quad_codes(pts[idx])
            res.append((code[ok], np.take_along_axis(idx, order, 1)[ok]))
        codes = np.concatenate([r[0] for r in res])
        iquads = np.concatenate([r[1] for r in res])
        hits = index.code_tree.query_ball_point(codes, tol)
        cand = [(i, j) for i, js in enumerate(hits) for j in js]
        if progress:
            progress(f"{len(cand)} candidate matches ({'mirrored' if mirror else 'direct'})")
        for i, j in cand:
            r = _verify(pts, iquads[i], index.quads[j], index, min_matches, width, height)
            if r is not None and (best is None or r["matches"] > best["matches"]):
                r["mirror"] = mirror
                best = r
        if best is not None:
            break
    if best is None:
        return None
    # field centre
    T = best.pop("_T")
    centre = _pix_to_sky(np.array([[(width - 1) / 2, (height - 1) / 2 if not best["mirror"] else (height - 1) / 2]]),
                         T, best.pop("_c"))
    best["ra"], best["dec"] = float(centre[0, 0]), float(centre[0, 1])
    return best


def _fit_similarity(src, dst):
    """Least-squares similarity dst ~ a*src + b (complex form). Returns (a, b)."""
    zs = src[:, 0] + 1j * src[:, 1]
    zd = dst[:, 0] + 1j * dst[:, 1]
    A = np.c_[zs, np.ones_like(zs)]
    sol = np.linalg.lstsq(A, zd, rcond=None)[0]
    return sol[0], sol[1]


def _pix_to_sky(p, T, c):
    a, b = T
    z = (p[:, 0] + 1j * p[:, 1]) * a + b
    xi, eta = z.real, z.imag
    cvec = c / np.linalg.norm(c)
    east = np.cross([0.0, 0.0, 1.0], cvec)
    east /= max(np.linalg.norm(east), 1e-12)
    north = np.cross(cvec, east)
    v = cvec[None] + xi[:, None] * east[None] + eta[:, None] * north[None]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    dec = np.degrees(np.arcsin(v[:, 2]))
    ra = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
    return np.c_[ra, dec]


def _verify(pts, iq, cq, index, min_matches, width, height):
    c = index.xyz[cq[0]]
    cat = _tangent(index.xyz[cq], c)
    a, b = _fit_similarity(pts[iq], cat)
    scale = abs(a)                                           # radians per pixel
    if not (np.radians(0.2 / 3600) < scale < np.radians(300 / 3600)):
        return None
    if np.abs((pts[iq, 0] + 1j * pts[iq, 1]) * a + b - (cat[:, 0] + 1j * cat[:, 1])).max() > 3 * scale:
        return None
    sky = _pix_to_sky(pts, (a, b), c)
    xyz = _unit(sky[:, 0], sky[:, 1])
    rt = min(max(4 * scale, np.radians(20 / 3600)), np.radians(120 / 3600))
    d, _ = index.star_tree.query(xyz, distance_upper_bound=rt)
    m = int(np.isfinite(d).sum())
    if m < min_matches:
        return None
    # chance level: catalogue density in the field x matching area x number of image stars
    rf = 0.5 * np.hypot(width, height) * scale
    centre = _unit(*_pix_to_sky(np.array([[width / 2, height / 2]]), (a, b), c)[0])
    n_cat = len(index.star_tree.query_ball_point(centre, rf))
    lam = len(pts) * n_cat / (np.pi * rf * rf) * np.pi * rt * rt
    if m < lam + 4 * np.sqrt(lam) + 6:
        return None
    return {"matches": m, "chance": round(float(lam), 2), "scale": float(np.degrees(scale) * 3600),
            "rotation": float(np.degrees(np.angle(a))), "_T": (a, b), "_c": c}


def solve_any(xy, width, height, progress=None) -> dict | None:
    """Try the wide-field (Hipparcos) index, then the small-field (Tycho-2) index."""
    for cat in ("hip", "tyc"):
        r = solve(xy, width, height, SkyIndex(cat), progress=progress)
        if r is not None:
            r["index"] = cat
            return r
    return None
