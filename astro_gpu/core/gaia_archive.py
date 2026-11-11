"""Star catalogues from VizieR (CDS), or from the GAVO data centre (Heidelberg) and the ESA Gaia archive when VizieR
does not answer.

The same data under the same (VizieR) column names whatever the source: Gaia DR3 stars in a cone, the sampled XP
spectra of Gaia DR3 stars and the Hipparcos stars. VizieR is asked first; a service that does not answer is set aside
for 10 minutes and the next one is asked at once (``core/online.py``: short timeouts, a budget per query, one line in
the log per fallback). The CDS can be unreachable for days, and the ESA archive can answer its asynchronous queries
with errors or only after minutes; GAVO is the fast fallback.

XP spectra: VizieR and the ESA archive give 343 points, 336-1020 nm every 2 nm; GAVO gives the same spectra on 41
points, 400-800 nm every 10 nm (equal to the ESA values within 0.3 %). Spectra of both kinds are brought to the
common 400-800 nm grid (``common_grid``).
"""
from __future__ import annotations

import numpy as np

from . import online

#: seconds one query may take before its service is set aside: a Gaia cone (a sky tile, up to ~300 000 stars in the
#: Milky Way, GAVO's asynchronous jobs included), a Hipparcos / XP request, the last resort (nothing comes after it)
CONE_BUDGET = 300.0
SMALL_BUDGET = 60.0
LAST_BUDGET = 600.0


def _vizier(columns, row_limit=-1, host: str = "vizier", **kw):
    from astroquery.vizier import Vizier
    v = Vizier(columns=columns, row_limit=row_limit, timeout=(online.CONNECT_S, online.READ_S), **kw)
    v.VIZIER_SERVER = online.VIZIER_HOSTS[host]
    return v


# VizieR I/355/gaiadr3 column -> gaiadr3.gaia_source column (positions are the same epoch 2016.0 ICRS ones)
GAIA_COLS = {"Source": "source_id", "RA_ICRS": "ra", "DE_ICRS": "dec", "Gmag": "phot_g_mean_mag",
             "BPmag": "phot_bp_mean_mag", "RPmag": "phot_rp_mean_mag", "pmRA": "pmra", "pmDE": "pmdec",
             "XPsamp": "has_xp_sampled"}
# VizieR I/239/hip_main column -> public.hipparcos column (positions ICRS at epoch 1991.25 in both)
HIP_COLS = {"HIP": "hip", "RAICRS": "ra", "DEICRS": "de", "Vmag": "vmag", "B-V": "b_v", "pmRA": "pmra",
            "pmDE": "pmde"}
XP_WAVELENGTHS = np.arange(336.0, 1020.0 + 1e-6, 2.0)
GAVO_TAP = "https://dc.zah.uni-heidelberg.de/tap"
GAVO_XP_WAVELENGTHS = np.arange(400.0, 800.0 + 1e-6, 10.0)
XP_SAMPLED_G = 17.65            # Gaia DR3 published sampled XP spectra for the sources brighter than this


def _gavo(query: str):
    """GAVO TAP: a synchronous query, or an asynchronous job when it takes more than the ~20 s the service allows a
    synchronous one (dense Milky Way fields)."""
    import pyvo
    svc = pyvo.dal.TAPService(GAVO_TAP, session=online.requests_session((online.CONNECT_S, 120.0)))
    try:
        return svc.run_sync(query, maxrec=50_000_000).to_table()
    except pyvo.dal.DALQueryError as e:
        if "timed out" not in str(e).lower() and "too long" not in str(e).lower():
            raise
        return svc.run_async(query, maxrec=50_000_000).to_table()


def _column(t, k) -> np.ndarray:
    a = np.ma.asarray(t[k])
    if a.dtype == bool or a.dtype.kind == "b":
        return np.ma.filled(a.astype(np.float64), np.nan)
    if k in ("Source", "source_id", "HIP", "hip"):
        return np.ma.filled(a, -1).astype(np.int64)
    return np.ma.filled(a.astype(np.float64), np.nan)


def _range_sql(col: str, flt: str) -> str:
    """A VizieR column filter ('<18.5', '18.5..20.5', '>3') as ADQL."""
    f = flt.strip()
    if ".." in f:
        lo, hi = f.split("..")
        return f"{col} >= {float(lo)} AND {col} < {float(hi)}"
    if f.startswith("<"):
        return f"{col} < {float(f[1:])}"
    if f.startswith(">"):
        return f"{col} > {float(f[1:])}"
    return f"{col} = {float(f)}"


def _esa(query: str):
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    return Gaia.launch_job_async(query).get_results()


def gaia_cone(ra: float, dec: float, radius: float, cols: list[str], gfilter: str, say=None, polygon=None) -> dict:
    """Gaia DR3 stars within ``radius`` deg of (ra, dec) whose G magnitude passes ``gfilter`` (VizieR syntax):
    {VizieR column: array}. VizieR, else GAVO, else the ESA archive; raises ``online.ServiceDown`` when none
    answers. ``say``: where the fallbacks are told (``online.sayer``). ``polygon``: [(ra, dec)] of a region inside the
    cone that holds every star the caller keeps (a sky tile with a margin, ``skycache.tile_polygon``): GAVO is asked for
    it rather than the whole cone, so that the stars returned may be those of the polygon only."""

    def vizier():
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        v = _vizier(cols, column_filters={"Gmag": gfilter})
        r = v.query_region(SkyCoord(ra, dec, unit="deg"), radius=radius * u.deg, catalog="I/355/gaiadr3",
                           cache=False)
        return {k: (_column(r[0], k) if len(r) else np.zeros(0, np.int64 if k == "Source" else np.float64))
                for k in cols}

    def gavo():             # GAVO's gaia.dr3lite: the same columns but has_xp_sampled, which follows from G
        need = [k for k in cols if k != "XPsamp"] + (["Gmag"] if "XPsamp" in cols and "Gmag" not in cols else [])
        t = _gavo_cone(need, ra, dec, radius, gfilter, polygon=polygon)
        out = {k: _column(t, GAIA_COLS[k]) for k in need}
        if "XPsamp" in cols:
            out["XPsamp"] = (out["Gmag"] < XP_SAMPLED_G).astype(np.float64)
        return {k: out[k] for k in cols}

    def esa():
        where = (f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius:.6f})) AND "
                 f"{_range_sql('phot_g_mean_mag', gfilter)}")
        t = _esa(f"SELECT {', '.join(GAIA_COLS[k] for k in cols)} FROM gaiadr3.gaia_source {where}")
        return {k: _column(t, GAIA_COLS[k]) for k in cols}

    return online.first_answer([("vizier", vizier), ("gavo", gavo), ("esa", esa)], "Gaia stars", say,
                               budget=CONE_BUDGET)


def _gavo_cone(need: list[str], ra: float, dec: float, radius: float, gfilter: str, depth: int = 0, polygon=None):
    """A Gaia cone from GAVO; when the service fails on it (dense Milky Way fields), the cone is covered by 7 cones of
    half its radius (one at the centre, six around at 0.866 of the radius: the smallest cover), twice at most, and the
    stars found twice are kept once. ``polygon``: asked first in place of the cone (see ``gaia_cone``), the cone when
    the service fails on it."""
    sel = ", ".join(GAIA_COLS[k] for k in dict.fromkeys(need + ["Source", "RA_ICRS", "DE_ICRS"]))
    if polygon:
        # issue #129: GAVO took most of a field's first run on the cones around the sky tiles, twice the tile's area;
        # the polygon around the tile came 2 to 5 times faster on small tiles near the Rosette, with the same stars in
        # the tile (the caller keeps only those)
        pts = ", ".join(f"{a:.8f}, {b:.8f}" for a, b in polygon)
        try:
            return _gavo(f"SELECT {sel} FROM gaia.dr3lite WHERE 1=CONTAINS(POINT('ICRS', ra, dec), POLYGON('ICRS', "
                         f"{pts})) AND {_range_sql('phot_g_mean_mag', gfilter)}")
        except Exception as e:  # noqa: BLE001 - the cone below
            if online.unreachable(e):
                raise
    try:                    # (two GAVO queries at most at once: online.SLOTS, held by the caller's online.call)
        return _gavo(f"SELECT {sel} FROM gaia.dr3lite WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', "
                     f"{ra:.8f}, {dec:.8f}, {radius:.6f})) AND {_range_sql('phot_g_mean_mag', gfilter)}")
    except Exception as e:  # noqa: BLE001
        # unreachable: not 7 x 7 smaller queries that all wait for a connection
        if depth >= 2 or online.unreachable(e):
            raise
    from astropy.table import unique, vstack
    d = 0.866 * radius
    parts = [_gavo_cone(need, ra, dec, 0.52 * radius, gfilter, depth + 1)]
    for a in np.radians(np.arange(0, 360, 60)):
        dd = dec + d * np.sin(a)
        parts.append(_gavo_cone(need, ra + d * np.cos(a) / max(np.cos(np.radians(dd)), 1e-3), dd, 0.52 * radius,
                                gfilter, depth + 1))
    t = unique(vstack([p for p in parts if len(p)]), keys="source_id") if any(len(p) for p in parts) else parts[0]
    c = np.radians
    sep = np.degrees(np.arccos(np.clip(np.sin(c(dec)) * np.sin(c(np.asarray(t["dec"]))) + np.cos(c(dec)) *
                                       np.cos(c(np.asarray(t["dec"]))) * np.cos(c(np.asarray(t["ra"]) - ra)), -1, 1)))
    return t[sep <= radius]


def hipparcos_all(cols: list[str], vmax: float, say=None) -> dict:
    """Every Hipparcos star brighter than V = vmax on the whole sky (V < 9: about 22 000 stars): {VizieR column:
    array}. VizieR, its mirror at Harvard, else the ESA archive."""

    def vizier(host):
        def q():
            v = _vizier(cols, host=host, column_filters={"Vmag": f"<{vmax}"})
            r = v.query_constraints(catalog="I/239/hip_main", cache=False)
            return {k: (_column(r[0], k) if len(r) else np.zeros(0)) for k in cols}
        return q

    def esa():
        t = _esa(f"SELECT {', '.join(HIP_COLS[k] for k in cols)} FROM public.hipparcos WHERE vmag < {vmax}")
        return {k: _column(t, HIP_COLS[k]) for k in cols}

    out = online.first_answer([("vizier", vizier("vizier")), ("vizier_cfa", vizier("vizier_cfa")), ("esa", esa)],
                              "the Hipparcos stars", say, budget=SMALL_BUDGET * 3)
    if "HIP" in out:
        out["HIP"] = np.asarray(out["HIP"], np.float64)
    return out


def hipparcos_cone(ra: float, dec: float, radius: float, cols: list[str], vmax: float, say=None) -> dict:
    """Hipparcos stars brighter than V = vmax within ``radius`` deg: {VizieR column: array} (from the whole-sky list,
    cached)."""
    from .skycache import _sep, hipparcos
    t = hipparcos(list(dict.fromkeys(list(cols) + ["RAICRS", "DEICRS"])), vmax, log=say)
    m = _sep(ra, dec, t["RAICRS"], t["DEICRS"]) <= radius
    return {k: t[k][m] for k in cols}


def xp_spectrum_vizier(sid: int):
    """(wavelength nm, flux) of one star from VizieR, None when it has none; raises when VizieR fails."""
    v = _vizier(["Source", "lambda", "Flux"])
    rr = v.query_constraints(catalog="I/355/xpsample", Source=f"={sid}", cache=False)
    if len(rr) and len(rr[0]):
        lam = np.asarray(rr[0]["lambda"], float)
        o = np.argsort(lam)
        return lam[o], np.asarray(rr[0]["Flux"], float)[o]
    return None


def xp_spectra_gavo(sids: list[int], batch: int = 500) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sampled XP spectra of Gaia DR3 stars from GAVO (400-800 nm every 10 nm), by batches: {source id: (wavelength,
    flux)}."""
    out: dict = {}
    for i in range(0, len(sids), batch):
        part = ",".join(str(int(s)) for s in sids[i:i + batch])
        t = _gavo(f"SELECT source_id, flux FROM gdr3spec.spectra WHERE source_id IN ({part})")
        for row in t:
            f = np.asarray(row["flux"], float)
            if f.size == GAVO_XP_WAVELENGTHS.size:
                out[int(row["source_id"])] = (GAVO_XP_WAVELENGTHS.copy(), f)
    return out


def common_grid(rows: list) -> list:
    """XP spectra (wavelength, flux) or None, all on one wavelength grid: unchanged when they share one, else every
    spectrum resampled on the GAVO grid (400-800 nm every 10 nm), which the finer ESA grid covers."""
    grids = {len(r[0]) for r in rows if r is not None}
    if len(grids) <= 1:
        return rows
    g = GAVO_XP_WAVELENGTHS
    return [None if r is None else (g.copy(), np.interp(g, r[0], r[1])) for r in rows]


def xp_spectra_esa(sids: list[int], batch: int = 500) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Sampled XP spectra of Gaia DR3 stars from the ESA archive, by batches: {source id: (wavelength, flux)}."""
    from astroquery.gaia import Gaia
    out: dict = {}
    for i in range(0, len(sids), batch):
        part = [int(s) for s in sids[i:i + batch]]
        d = Gaia.load_data(ids=part, retrieval_type="XP_SAMPLED", data_release="Gaia DR3",
                           data_structure="DATAMODEL_GAIA", format="votable")
        for v in d.values():
            t = v[0].to_table() if hasattr(v[0], "to_table") else v[0]
            for row in t:
                f = np.asarray(row["flux"], float)
                if f.size == XP_WAVELENGTHS.size:
                    out[int(row["source_id"])] = (XP_WAVELENGTHS.copy(), f)
    return out
