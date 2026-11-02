"""The sky catalogue cache (core/skycache.py), the online services' time limits (core/online.py) and the offline
Gaia extracts (core/sirilcat.py; their download: test_gaia_extracts.py). No network: the services are replaced by a
synthetic sky or by a local server that never answers."""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from astro_gpu.core import gaia_archive, online, sirilcat, skycache


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """A private ASTRO_GPU_HOME and user home (the earlier versions' cache is ~/.astro_gpu/cache), no Siril found,
    offline unless a test says otherwise, fresh verdicts on the services."""
    home = tmp_path / "userhome"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "aghome"))
    (tmp_path / "nowhere").mkdir()
    for v in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(v, str(tmp_path / "nowhere"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")
    online.reset()
    sirilcat.reset()
    yield
    online.reset()
    sirilcat.reset()


# ------------------------------------------------------------------ a synthetic sky in place of the Gaia services

COLS = ["Source", "RA_ICRS", "DE_ICRS", "Gmag", "BPmag", "RPmag", "pmRA", "pmDE", "XPsamp"]


def _sky(n=60000, ra0=150.0, dec0=30.0, half=6.0, seed=1):
    rng = np.random.default_rng(seed)
    dec = rng.uniform(dec0 - half, dec0 + half, n)
    ra = ra0 + rng.uniform(-half, half, n) / np.cos(np.radians(dec))
    g = rng.uniform(6, 21, n)
    return {"Source": np.arange(1, n + 1, dtype=np.int64), "RA_ICRS": ra, "DE_ICRS": dec, "Gmag": g,
            "BPmag": g + 0.4, "RPmag": g - 0.5, "pmRA": rng.normal(0, 5, n), "pmDE": rng.normal(0, 5, n),
            "XPsamp": (g < 17.65).astype(float)}


class FakeGaia:
    """gaia_cone over the synthetic sky; counts the calls."""

    def __init__(self, sky):
        self.sky, self.calls = sky, 0
        self.lock = threading.Lock()

    def __call__(self, ra, dec, radius, cols, gfilter, say=None):
        if online.offline():
            raise online.ServiceDown("vizier", "is not asked (offline mode)")
        with self.lock:
            self.calls += 1
        lo, hi = skycache.parse_filter(gfilter)
        s = self.sky
        m = (skycache._sep(ra, dec, s["RA_ICRS"], s["DE_ICRS"]) <= radius) & (s["Gmag"] < hi)
        if lo is not None:
            m &= s["Gmag"] >= lo
        return {k: s[k][m] for k in cols}


def _brute(sky, cones, cols, lo, hi):
    m = np.zeros(len(sky["Gmag"]), bool)
    for ra, dec, r in cones:
        m |= skycache._sep(ra, dec, sky["RA_ICRS"], sky["DE_ICRS"]) <= r
    m &= sky["Gmag"] < hi
    if lo is not None:
        m &= sky["Gmag"] >= lo
    return {k: sky[k][m] for k in cols}


def _same_set(a, b):
    return np.array_equal(np.sort(a["Source"]), np.sort(b["Source"]))


# ------------------------------------------------------------------ folders

def test_the_cache_is_under_astro_gpu_home_and_the_old_cache_is_read_never_written(tmp_path):
    """Before, the catalogue cache was ~/.astro_gpu/cache whatever ASTRO_GPU_HOME said: tests and benches wrote the
    user's cache and a pilot home could not carry its own. Now it is ASTRO_GPU_HOME/cache; the old one is still read
    (the user's downloads are not lost) and never touched."""
    assert skycache.cache_dir() == tmp_path / "aghome" / "cache"
    old = Path.home() / ".astro_gpu" / "cache"
    (old / "xp").mkdir(parents=True)
    np.save(old / "xp" / "42.npy", np.ones((2, 3)))
    before = sorted(p.relative_to(old) for p in old.rglob("*"))
    assert skycache.find("xp/42.npy") == old / "xp" / "42.npy"
    skycache.save_npy(skycache.cache_dir() / "xp" / "43.npy", np.zeros(0))
    assert skycache.find("xp/43.npy") == skycache.cache_dir() / "xp" / "43.npy"
    assert sorted(p.relative_to(old) for p in old.rglob("*")) == before
    skycache.save_npy(skycache.cache_dir() / "xp" / "42.npy", np.zeros(3))      # the new cache comes first
    assert skycache.find("xp/42.npy").parent.parent == skycache.cache_dir()


def test_tiles_in_cone_finds_every_tile_the_cone_touches():
    rng = np.random.default_rng(5)
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    for level in (3, 5, 6, 8):
        res = 58.6 / 2 ** level
        for _ in range(40):
            ra0, dec0 = rng.uniform(0, 360), rng.uniform(-88, 88)
            r = rng.uniform(0.02, 3.0) * res
            got = set(skycache.tiles_in_cone(level, ra0, dec0, r).tolist())
            n = 3000
            d = r * np.sqrt(rng.uniform(0, 1, n))
            c = SkyCoord(ra0 * u.deg, dec0 * u.deg).directional_offset_by(rng.uniform(0, 2 * np.pi, n) * u.rad,
                                                                           d * u.deg)
            need = set(skycache.tile_of(level, c.ra.deg, c.dec.deg).tolist())
            assert need <= got, (level, ra0, dec0, r)
            assert len(got) <= len(need) + 12             # and not much more


# ------------------------------------------------------------------ sky tiles

def test_a_field_inside_fetched_tiles_needs_no_network(monkeypatch):
    """The catalogue was cached by exact cone centre: a re-stack of the same field (its solution moves the centres a
    little) downloaded everything again. Tiles: the first field fetches its tiles, the moved field reads them, offline
    too; the stars are exactly those of the cones."""
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "")
    sky = _sky()
    fake = FakeGaia(sky)
    monkeypatch.setattr(gaia_archive, "gaia_cone", fake)
    cones = [(150.0 + dx, 30.0 + dy, 0.75) for dx in (-0.6, 0.6) for dy in (-0.6, 0.6)]
    got = skycache.gaia_stars(cones, COLS, "<18.5")
    assert fake.calls > 0 and _same_set(got, _brute(sky, cones, COLS, None, 18.5))
    i = np.argsort(got["Source"])
    ref = _brute(sky, cones, COLS, None, 18.5)
    j = np.argsort(ref["Source"])
    assert all(np.array_equal(got[k][i], ref[k][j]) for k in COLS)
    n = fake.calls
    moved = [(ra + 0.03, dec - 0.02, r * 0.95) for ra, dec, r in cones]
    assert _same_set(skycache.gaia_stars(moved, COLS, "<18.5"), _brute(sky, moved, COLS, None, 18.5))
    assert fake.calls == n
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")
    assert _same_set(skycache.gaia_stars(moved, COLS, "<18.5"), _brute(sky, moved, COLS, None, 18.5))
    # a brighter query with fewer columns (the astrometry's) is served by the same tiles, offline
    few = ["RA_ICRS", "DE_ICRS", "Gmag"]
    small = [(150.1, 29.9, 0.4)]
    b = skycache.gaia_stars(small, few, "<13.25")
    ref = _brute(sky, small, few, None, 13.25)
    assert np.array_equal(np.sort(b["Gmag"]), np.sort(ref["Gmag"]))
    # a field elsewhere, offline: a clear error at once, not a wait
    t = time.monotonic()
    with pytest.raises(skycache.SkyUnavailable):
        skycache.gaia_stars([(160.0, 35.0, 0.5)], COLS, "<18.5")
    assert time.monotonic() - t < 5


def test_a_magnitude_range_layer_and_its_ladder(monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "")
    sky = _sky(seed=2)
    fake = FakeGaia(sky)
    monkeypatch.setattr(gaia_archive, "gaia_cone", fake)
    few = ["RA_ICRS", "DE_ICRS", "Gmag"]
    cones = [(150.0, 30.0, 1.2)]
    f = skycache.gaia_stars(cones, few, "18.5..20.5")
    ref = _brute(sky, cones, few, 18.5, 20.5)
    assert np.array_equal(np.sort(f["Gmag"]), np.sort(ref["Gmag"]))
    # the astrometry's G < 13.3 fetches the G < 13.5 layer (level 5); G < 13.45 then reads it
    a = skycache.gaia_stars(cones, few, "<13.3")
    n = fake.calls
    b = skycache.gaia_stars(cones, few, "<13.45")
    assert fake.calls == n
    assert np.array_equal(np.sort(b["Gmag"]), np.sort(_brute(sky, cones, few, None, 13.45)["Gmag"]))
    assert (a["Gmag"] < 13.3).all()
    names = {lay.dir.name.split("_")[0] for lay in skycache.layers()}
    assert names == {"G18.5-20.5", "G-13.5"}


# ------------------------------------------------------------------ online services

@pytest.fixture
def silent_server():
    """A server that accepts connections and never answers."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(16)
    conns, stop = [], threading.Event()

    def accept():
        s.settimeout(0.2)
        while not stop.is_set():
            try:
                conns.append(s.accept()[0])
            except OSError:
                continue
    t = threading.Thread(target=accept, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{s.getsockname()[1]}/"
    stop.set()
    for c in conns:
        c.close()
    s.close()


def test_a_silent_service_costs_seconds_and_the_next_one_answers(monkeypatch, silent_server):
    """VizieR accepting connections and never answering made the first query wait its whole 600-900 s timeout.
    Now the service fails its probe within the probe's timeout, is set aside, and the next source is asked at once,
    with one plain line in the log (one, not one per tile)."""
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "")
    monkeypatch.setattr(online, "PROBES", {"vizier": silent_server})
    monkeypatch.setattr(online, "PROBE_S", (1.0, 1.0))
    sky = _sky(seed=3)
    asked = []

    def gavo_cone(need, ra, dec, radius, gfilter, depth=0):
        asked.append(radius)
        from astropy.table import Table
        r = FakeGaia(sky)(ra, dec, radius, need + ["Source"], gfilter)
        return Table({gaia_archive.GAIA_COLS[k]: v for k, v in r.items()})
    monkeypatch.setattr(gaia_archive, "_gavo_cone", gavo_cone)
    logs = []
    t = time.monotonic()
    got = skycache.gaia_stars([(150.0, 30.0, 0.8)], ["RA_ICRS", "DE_ICRS", "Gmag"], "<16",
                              log=lambda m, level="info": logs.append((level, m)))
    assert time.monotonic() - t < 8
    assert asked and len(got["Gmag"]) == len(_brute(sky, [(150.0, 30.0, 0.8)], ["Gmag"], None, 16)["Gmag"])
    assert online.is_down("vizier")
    assert [m for lv, m in logs if lv == "warning"] == [
        "VizieR (CDS, Strasbourg) does not answer (silent): Gaia stars from GAVO (Heidelberg)"]


def test_a_query_over_its_budget_is_abandoned_and_the_service_set_aside(monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "")
    monkeypatch.setattr(online, "PROBES", {})
    t = time.monotonic()
    with pytest.raises(online.ServiceDown) as e:
        online.call("gavo", lambda: time.sleep(5), 0.3)
    assert time.monotonic() - t < 2 and "took more than" in e.value.why
    assert online.is_down("gavo")
    with pytest.raises(online.ServiceDown):                       # set aside: not asked again
        online.call("gavo", lambda: 1, 5)
    said = []
    r = online.first_answer([("gavo", lambda: 1), ("esa", lambda: 2)], "the things", lambda m, level="w": said.append(m))
    assert r == 2 and said == ["GAVO (Heidelberg) took more than 0 s: the things from the ESA Gaia archive"]


# ------------------------------------------------------------------ the offline extracts

def _write_siril(path: Path, kind: int, level: int, stars: dict, chunk: tuple[int, int] | None = None):
    """A Siril catalogue file (format 1.0.0) of the given stars (ra, dec, g, and flux / e for xp_sampled)."""
    import astropy.units as u
    from astropy_healpix import HEALPix
    hp = HEALPix(nside=2 ** level, order="nested")
    pix = np.asarray(hp.lonlat_to_healpix(stars["ra"] * u.deg, stars["dec"] * u.deg), np.int64)
    if chunk:
        clevel, c = chunk
        first = c * 4 ** (level - clevel)
        n = 4 ** (level - clevel)
    else:
        first, n = 0, 12 * 4 ** level
    o = np.argsort(pix, kind="stable")
    dt = sirilcat.ASTRO_DT if kind == 1 else sirilcat.XP_DT
    rec = np.zeros(len(o), dt)
    rec["ra"] = np.round(stars["ra"][o] / sirilcat.SCALE).astype(np.int32)
    rec["dec"] = np.round(stars["dec"][o] / sirilcat.SCALE).astype(np.int32)
    rec["g"] = np.round(stars["g"][o] * 1000).astype(np.int16)
    rec["pmra"], rec["pmdec"] = 3, -4
    if kind == 2:
        rec["e"] = stars["e"][o]
        rec["flux"] = (stars["flux"][o] * 10.0 ** stars["e"][o][:, None]).astype(np.float16)
    idx = np.cumsum(np.bincount(pix[o] - first, minlength=n)).astype("<u4")
    h = bytearray(128)
    h[:48] = b"Siril test extract".ljust(48, b"\0")
    h[48:53] = bytes([3, level, kind, 1 if chunk else 0, chunk[0] if chunk else 0])
    if chunk:
        h[53:57] = int(chunk[1]).to_bytes(4, "little")
        h[57:61] = int(first).to_bytes(4, "little")
        h[61:65] = int(first + n - 1).to_bytes(4, "little")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(h))
        f.write(idx.tobytes())
        f.write(rec.tobytes())


def test_siril_astrometric_extract_is_found_and_read(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    n = 4000
    st = {"ra": 40.0 + rng.uniform(-1, 1, n), "dec": 61.0 + rng.uniform(-0.5, 0.5, n), "g": rng.uniform(6, 17, n)}
    cat = tmp_path / "sirildata" / "siril_cat_healpix8_astro.dat"
    _write_siril(cat, 1, 8, st)
    assert sirilcat.locate()["astro"] is None
    # an extra place only: where another program's settings say (its catalogue installer writes the key)
    cfg = tmp_path / "nowhere" / "siril" / "config.1.4.ini"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("[core]\ncatalogue_gaia_astro=" + str(cat).replace("\\", "\\\\") + "\n", encoding="utf-8")
    sirilcat.reset()
    assert sirilcat.locate()["astro"].path == cat
    r = sirilcat.astro_cone(40.2, 61.1, 0.3, 15.0)
    sel = (skycache._sep(40.2, 61.1, st["ra"], st["dec"]) <= 0.3) & (st["g"] < 15.0)
    assert len(r["Gmag"]) == sel.sum()
    assert np.allclose(np.sort(r["RA_ICRS"]), np.sort(st["ra"][sel]), atol=1e-6)
    assert np.allclose(np.sort(r["Gmag"]), np.sort(np.round(st["g"][sel] * 1000) / 1000))
    assert (r["pmRA"] == 3).all()
    # the extract keeps the N brightest stars of a pixel: where a pixel holds N, it is complete only to its faintest
    whole = sirilcat.astro_cone(40.0, 61.0, 1.5, 30.0)
    assert len(whole["Gmag"]) == n and np.isfinite(whole["complete_to"]) and whole["complete_to"] <= 17.0
    # truncated / foreign files are ignored
    bad = tmp_path / "bad" / "siril_cat_healpix8_astro.dat"
    bad.parent.mkdir()
    bad.write_bytes(cat.read_bytes()[:-16])
    assert sirilcat.read_header(bad) is None


def test_siril_xp_extract_gives_the_spectra_by_position(tmp_path, monkeypatch):
    import astropy.units as u
    from astropy_healpix import HEALPix
    rng = np.random.default_rng(8)
    n = 300
    st = {"ra": 38.0 + rng.uniform(-0.5, 0.5, n), "dec": 61.0 + rng.uniform(-0.3, 0.3, n), "g": rng.uniform(8, 14, n),
          "e": rng.integers(14, 17, n).astype(np.int8)}
    st["flux"] = rng.uniform(0.5, 2.0, (n, 343)) * 10.0 ** (-st["e"][:, None].astype(float))
    h1 = HEALPix(nside=2, order="nested")
    chunk = int(h1.lonlat_to_healpix(38.0 * u.deg, 61.0 * u.deg))
    inside = np.asarray(h1.lonlat_to_healpix(st["ra"] * u.deg, st["dec"] * u.deg)) == chunk
    st = {k: v[inside] for k, v in st.items()}
    n = int(inside.sum())
    d = tmp_path / "cats" / "siril_cat1_healpix8_xpsamp"
    _write_siril(d / f"siril_cat1_healpix8_xpsamp_{chunk}.dat", 2, 8, st, chunk=(1, chunk))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(tmp_path / "cats"))
    sirilcat.reset()
    assert list(sirilcat.locate()["xp"]) == [chunk]
    q = np.r_[np.arange(0, n, 3), [0]]
    ra, dec, g = st["ra"][q].copy(), st["dec"][q].copy(), st["g"][q].copy()
    ra[-1] += 1.0 / 3600                       # 1" away: not that star
    rows = sirilcat.xp_spectra(ra, dec, g)
    assert rows[-1] is None and all(r is not None for r in rows[:-1])
    for k, i in enumerate(q[:-1]):
        lam, f = rows[k]
        assert len(lam) == 343 and lam[0] == 336 and lam[-1] == 1020
        assert np.allclose(f, st["flux"][i], rtol=1e-3)


# ------------------------------------------------------------------ names, catalogs stage

def test_target_names_are_resolved_from_the_shipped_catalogues_offline(tmp_path, monkeypatch):
    from astro_gpu.stages.astrometry import resolve_name
    for k in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)", "LOCALAPPDATA", "APPDATA"):
        monkeypatch.setenv(k, str(tmp_path / "empty"))                # no other program's lists: ours only
    m51 = (202.469625, 47.195167)                                     # OpenNGC's NGC 5194
    t = time.monotonic()
    assert resolve_name("M 51") == pytest.approx(m51)
    assert resolve_name("NGC 5194") == pytest.approx(m51)
    assert resolve_name("whirlpool_galaxy") == pytest.approx(m51)
    assert resolve_name("LDN 1622") == pytest.approx((88.6503, 2.0081))       # the Lynds dark nebulae (VII/7A)
    assert resolve_name("Sh2-155") == pytest.approx((344.1826, 62.6176), abs=1e-3)
    assert time.monotonic() - t < 3
    from astro_gpu.core.stage import StageError
    with pytest.raises(StageError):                          # unknown and offline: at once
        resolve_name("Nowhere 12")
    skycache.save_json(skycache.cache_dir() / "names.json", {"nowhere12": [1.0, 2.0, "simbad"]})
    assert resolve_name("Nowhere 12") == (1.0, 2.0)


def test_catalogs_reads_an_earlier_versions_cones_else_the_tiles(monkeypatch):
    """A re-run of a project whose cones an earlier version downloaded reads them (no network, the same stars); a
    missing cell comes from the tiles of the fine cones that reach it."""
    from astro_gpu.stages import catalogs
    sky = _sky(seed=4)
    centres = np.array([[150.0, 30.0], [151.0, 30.0]])
    fine = (np.array([[150.0, 30.0], [150.5, 30.0], [151.0, 30.0]]), 0.6)
    radius, gf = 0.8, "<18.5"
    old = Path.home() / ".astro_gpu" / "cache" / "gaia"
    old.mkdir(parents=True)
    for c in centres[:1]:
        np.savez(old / f"{catalogs._cone_key(COLS, gf, c, radius)}.npz", **_brute(sky, [(*c, radius)], COLS, None, 18.5))
    asked = []

    def fake_stars(cones, cols, gfilter, log=None, progress=None, label=""):
        asked.append(cones)
        return _brute(sky, cones, cols, None, 18.5)
    monkeypatch.setattr(skycache, "gaia_stars", fake_stars)

    class Ctx:
        def log(self, m, level="info"):
            pass

        def progress(self, f, m=""):
            pass
    got = catalogs._query_cells(Ctx(), centres, radius, COLS, gf, "deep", 0, 0.5, fine)
    assert asked == [[(150.0, 30.0, 0.6), (150.5, 30.0, 0.6), (151.0, 30.0, 0.6)]]
    assert _same_set(got, _brute(sky, asked[0] + [(150.0, 30.0, radius)], COLS, None, 18.5))
    np.savez(old / f"{catalogs._cone_key(COLS, gf, centres[1], radius)}.npz",
             **_brute(sky, [(*centres[1], radius)], COLS, None, 18.5))
    got = catalogs._query_cells(Ctx(), centres, radius, COLS, gf, "deep", 0, 0.5, fine)
    assert len(asked) == 1                                   # every cell in the old cache: no tiles, no network
    want = _brute(sky, [(150.0, 30.0, radius), (151.0, 30.0, radius)], COLS, None, 18.5)
    assert _same_set(got, want) and len(got["Source"]) == len(want["Source"])


def test_in_cones_is_the_union_of_the_cones():
    rng = np.random.default_rng(9)
    ra, dec = rng.uniform(140, 160, 20000), rng.uniform(20, 40, 20000)
    cones = [(150.0, 30.0, 1.0), (152.0, 31.0, 0.5), (149.0, 29.0, 0.5)]
    want = np.zeros(len(ra), bool)
    for c in cones:
        want |= skycache._sep(c[0], c[1], ra, dec) <= c[2]
    assert np.array_equal(skycache.in_cones(ra, dec, cones), want) and want.sum() > 100


def test_survey_downloads_are_cached_and_a_missing_tile_is_remembered(monkeypatch):
    """The H-alpha cut-out and HiPS tiles: cached under ASTRO_GPU_HOME (a re-run works offline); a tile the server
    does not have (outside the survey) is remembered as absent, not asked again."""
    import urllib.error

    from astro_gpu.stages import surveys
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "")
    monkeypatch.setattr(online, "PROBES", {})
    got = []

    def get(url, timeout=60.0):
        got.append(url)
        if "Npix9" in url:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        return b"SIMPLE  = T" + url.encode()
    monkeypatch.setattr(surveys, "_get", get)
    a = surveys._cached_url("https://example.org/hips/Norder3/Dir0/Npix1.fits", "hips_tiles", "hips", 10)
    assert surveys._cached_url("https://example.org/hips/Norder3/Dir0/Npix9.fits", "hips_tiles", "hips", 10) is None
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")
    assert surveys._cached_url("https://example.org/hips/Norder3/Dir0/Npix1.fits", "hips_tiles", "hips", 10) == a
    assert surveys._cached_url("https://example.org/hips/Norder3/Dir0/Npix9.fits", "hips_tiles", "hips", 10) is None
    assert len(got) == 2 and (skycache.cache_dir() / "hips_tiles").is_dir()
