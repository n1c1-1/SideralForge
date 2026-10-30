"""Star catalogues from VizieR (CDS), or from the GAVO data centre (Heidelberg) and the ESA Gaia archive when VizieR
does not answer.

The same data under the same (VizieR) column names whatever the source: Gaia DR3 stars in a cone, the sampled XP
spectra of Gaia DR3 stars and the Hipparcos stars. VizieR is asked first; after one failure the rest of the session
goes to the other services (CDS was unreachable for days on 2026-10-28; on 2026-10-30 the ESA archive answered its
asynchronous queries with errors 500 or in minutes, while GAVO gave 10 000 stars in 6 s).

XP spectra: VizieR and the ESA archive give 343 points, 336-1020 nm every 2 nm; GAVO gives the same spectra on 41
points, 400-800 nm every 10 nm (equal to the ESA values within 0.3 %). Spectra of both kinds are brought to the
common 400-800 nm grid (``common_grid``).
"""
from __future__ import annotations

import threading

import numpy as np

_VIZIER_DOWN = threading.Event()

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
    import pyvo
    return pyvo.dal.TAPService(GAVO_TAP).run_sync(query, maxrec=50_000_000).to_table()


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


def gaia_cone(ra: float, dec: float, radius: float, cols: list[str], gfilter: str, timeout: int = 900) -> dict:
    """Gaia DR3 stars within ``radius`` deg of (ra, dec) whose G magnitude passes ``gfilter`` (VizieR syntax):
    {VizieR column: array}."""
    if not _VIZIER_DOWN.is_set():
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            from astroquery.vizier import Vizier
            v = Vizier(columns=cols, column_filters={"Gmag": gfilter}, row_limit=-1, timeout=timeout)
            r = v.query_region(SkyCoord(ra, dec, unit="deg"), radius=radius * u.deg, catalog="I/355/gaiadr3")
            return {k: (_column(r[0], k) if len(r) else np.zeros(0, np.int64 if k == "Source" else np.float64))
                    for k in cols}
        except Exception:  # noqa: BLE001 - VizieR unreachable: the ESA archive
            _VIZIER_DOWN.set()
    where = (f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius:.6f})) AND "
             f"{_range_sql('phot_g_mean_mag', gfilter)}")
    try:                    # GAVO's gaia.dr3lite: the same columns but has_xp_sampled, which follows from G
        need = [k for k in cols if k != "XPsamp"] + (["Gmag"] if "XPsamp" in cols and "Gmag" not in cols else [])
        t = _gavo(f"SELECT {', '.join(GAIA_COLS[k] for k in need)} FROM gaia.dr3lite {where}")
        out = {k: _column(t, GAIA_COLS[k]) for k in need}
        if "XPsamp" in cols:
            out["XPsamp"] = (out["Gmag"] < XP_SAMPLED_G).astype(np.float64)
        return {k: out[k] for k in cols}
    except Exception:  # noqa: BLE001 - GAVO unreachable: the ESA archive
        pass
    t = _esa(f"SELECT {', '.join(GAIA_COLS[k] for k in cols)} FROM gaiadr3.gaia_source {where}")
    return {k: _column(t, GAIA_COLS[k]) for k in cols}


def hipparcos_cone(ra: float, dec: float, radius: float, cols: list[str], vmax: float) -> dict:
    """Hipparcos stars brighter than V = vmax within ``radius`` deg: {VizieR column: array}."""
    if not _VIZIER_DOWN.is_set():
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            from astroquery.vizier import Vizier
            v = Vizier(columns=cols, column_filters={"Vmag": f"<{vmax}"}, row_limit=-1, timeout=300)
            r = v.query_region(SkyCoord(ra, dec, unit="deg"), radius=radius * u.deg, catalog="I/239/hip_main")
            return {k: (_column(r[0], k) if len(r) else np.zeros(0)) for k in cols}
        except Exception:  # noqa: BLE001
            _VIZIER_DOWN.set()
    sel = ", ".join(HIP_COLS[k] for k in cols)
    t = _esa(f"SELECT {sel} FROM public.hipparcos WHERE 1=CONTAINS(POINT('ICRS', ra, de), "
             f"CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius:.6f})) AND vmag < {vmax}")
    return {k: _column(t, HIP_COLS[k]) for k in cols}


def xp_spectrum_vizier(sid: int):
    """(wavelength nm, flux) of one star from VizieR, None when it has none; raises when VizieR fails."""
    from astroquery.vizier import Vizier
    v = Vizier(columns=["Source", "lambda", "Flux"], row_limit=-1, timeout=120)
    rr = v.query_constraints(catalog="I/355/xpsample", Source=f"={sid}")
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


def vizier_down() -> bool:
    return _VIZIER_DOWN.is_set()


def mark_vizier_down() -> None:
    _VIZIER_DOWN.set()
