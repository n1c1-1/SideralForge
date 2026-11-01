"""The named deep-sky objects (core/skyobjects.py, api.sky_objects): designations, the merge of the catalogues into one
object per physical object, sizes and angles, the objects of a field on a synthetic TAN solution, the cache.

No network: tiny catalogues written in tmp_path in the formats of OpenNGC, VizieR's VII/20 answer and Siril's lists.
The real download is ``network``; the real test project (F:/astro_essais) is level 3."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from astro_gpu import api
from astro_gpu.core import skycache
from astro_gpu.core import skyobjects as so

COLS = ("Name;Type;RA;Dec;Const;MajAx;MinAx;PosAng;B-Mag;V-Mag;J-Mag;H-Mag;K-Mag;SurfBr;Hubble;Pax;Pm-RA;Pm-Dec;RadVel;"
        "Redshift;Cstar U-Mag;Cstar B-Mag;Cstar V-Mag;M;NGC;IC;Cstar Names;Identifiers;Common names;NED notes;"
        "OpenNGC notes;Sources").split(";")


def _row(**kw) -> str:
    kw = {k.replace("_", "-").replace("Common-names", "Common names"): v for k, v in kw.items()}
    return ";".join(str(kw.get(c, "")) for c in COLS)


NGC_ROWS = [
    _row(Name="NGC0224", Type="G", RA="00:42:44.35", Dec="+41:16:08.6", MajAx="177.83", MinAx="69.66", PosAng="35",
         B_Mag="4.29", V_Mag="3.44", M="031", Identifiers="PGC 002557,UGC 00454", Common_names="Andromeda Galaxy"),
    _row(Name="NGC6523", Type="Neb", RA="18:03:41.27", Dec="-24:22:48.6", MajAx="45.00", MinAx="30.00", V_Mag="5.80",
         M="008", NGC="6533", Identifiers="LBN 25", Common_names="Lagoon Nebula"),
    _row(Name="NGC6533", Type="Dup", RA="18:03:41.27", Dec="-24:22:48.6", NGC="6523"),
    _row(Name="NGC2239", Type="Cl+N", RA="06:31:55.56", Dec="+04:56:34.6", MajAx="9.30", NGC="2244",
         Identifiers="C 050,MWSC 0896"),
    _row(Name="NGC2244", Type="Dup", RA="06:31:55.56", Dec="+04:56:34.6", NGC="2239"),
    _row(Name="IC1805", Type="Cl+N", RA="02:32:41.51", Dec="+61:27:24.8", MajAx="60.00", MinAx="60.00", V_Mag="6.50",
         Identifiers="LBN 654,MWSC 0209"),
    _row(Name="IC0001", Type="**", RA="00:08:27.05", Dec="+27:43:03.6"),
    _row(Name="IC0002", Type="NonEx", RA="00:11:00.88", Dec="-12:49:22.3"),
    _row(Name="NGC7000", Type="HII", RA="20:59:17.14", Dec="+44:31:43.6", MajAx="120.00", MinAx="30.00", V_Mag="4.00",
         Identifiers="C 020,LBN 373", Common_names="North America Nebula"),
    _row(Name="NGC2024", Type="Neb", RA="05:41:43.00", Dec="-01:51:22.0", MajAx="30.00", MinAx="30.00",
         Common_names="Flame Nebula"),
    _row(Name="NGC1909", Type="RfN", RA="05:04:55.44", Dec="-07:15:56.3", MajAx="180.00", MinAx="60.00", IC="2118",
         Common_names="the Witch Head Nebula"),
    _row(Name="IC2118", Type="Dup", RA="05:04:55.44", Dec="-07:15:56.3", NGC="1909"),
]
ADDENDUM_ROWS = [
    _row(Name="B033", Type="DrkN", RA="05:40:59.0", Dec="-02:27:30", MajAx="6.00", MinAx="4.00", PosAng="90",
         Common_names="Horsehead Nebula"),
    _row(Name="C009", Type="HII", RA="22:57:54.0", Dec="+62:31:06", MajAx="50.00", MinAx="30.00", PosAng="90",
         Identifiers="LBN 529,SH 2-155", Common_names="Cave Nebula"),
    _row(Name="M040", Type="**", RA="12:22:16.1", Dec="+58:05:04", M="040", Identifiers="WDS J12222+5805AB"),
]
VII20 = """#
# VizieR Astronomical Server vizier.cfa.harvard.edu
#RESOURCE=yCat_7020
#Name: VII/20

_RAJ2000\t_DEJ2000\tSh2\tDiam
deg\tdeg\t \tarcmin
----------\t----------\t----\t----
271.358495\t-24.394253\t  25\t  90
308.080382\t+47.356933\t 116\t   2
314.694146\t+44.321772\t 117\t 240
344.182555\t+62.617601\t 155\t  60
038.346410\t+61.442231\t 190\t 150
093.727650\t+12.353946\t 271\t   8
"""
SIRIL = {
    "messier.csv": "name,ra,dec,diameter,mag,alias\n"
                   "M8,270.92208,-24.38,90,5.8,Lagoon nebula/NGC6523/NGC6533/Sh2-25\n"
                   "M31,10.684575,41.268903,178,3.5,Andromeda galaxy/NGC224\n",
    "ngc.csv": "name,ra,dec,diameter,mag,alias\n"
               "NGC224,10.68459,41.268889,189.1,3.4,M31\n"
               "NGC6523,270.925005,-24.38,45,,M8/NGC6533\n"
               "NGC7000,314.8,44.52,120,4,\n",
    "ic.csv": "name,ra,dec,diameter,mag,alias\nIC1805,38.200005,61.461667,20,6.5,\n",
    "sh2.csv": "name,ra,dec,alias\n"
               "Sh2-25,271.35846,-24.394283,Lagoon nebula/M8/NGC6523/NGC6533\n"
               "Sh2-116,308.0802,47.357054,NGC7000\n"
               "Sh2-117,314.694146,44.321772,\n"
               "Sh2-155,344.18,62.62,\n"
               "Sh2-190,38.34645,61.442323,\n"
               "Sh2-271,93.72765,12.353946,NGC2024\n",
}


@pytest.fixture
def sources(tmp_path) -> dict:
    """The tiny sources: OpenNGC's two files, VizieR's VII/20 answer, a Siril catalogue folder."""
    d = tmp_path / "src"
    (d / "siril").mkdir(parents=True)
    head = ";".join(COLS) + "\n"
    (d / "NGC.csv").write_text(head + "\n".join(NGC_ROWS) + "\n", encoding="utf-8")
    (d / "addendum.csv").write_text(head + "\n".join(ADDENDUM_ROWS) + "\n", encoding="utf-8")
    (d / "vii20.tsv").write_text(VII20, encoding="utf-8")
    for name, txt in SIRIL.items():
        (d / "siril" / name).write_text(txt, encoding="utf-8")
    return {"openngc": [d / "NGC.csv", d / "addendum.csv"], "sharpless": d / "vii20.tsv", "siril": d / "siril"}


@pytest.fixture
def full(sources) -> so.Catalogue:
    return so.build_catalogue(sources["openngc"], sources["sharpless"], sources["siril"])


@pytest.fixture(autouse=True)
def _fresh():
    so.reset()
    yield
    so.reset()


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """A private, empty ASTRO_GPU_HOME, offline, without earlier versions' cache nor Siril (a test sets them)."""
    h = tmp_path / "home"
    monkeypatch.setenv("ASTRO_GPU_HOME", str(h))
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")
    monkeypatch.setattr(skycache, "legacy_dirs", lambda: [])
    monkeypatch.setattr(so, "siril_catalogue_dir", lambda: None)
    return h


def _designations(o: so.SkyObject) -> set[str]:
    return {o.name, *o.aliases}


# ---------------------------------------------------------------------------------------------- designations

def test_designations_are_written_one_way():
    n = so.normalize
    assert n("NGC0224") == n("ngc 224") == n("NGC 224") == "NGC 224"
    assert n("M031") == n("m31") == "M 31"
    assert n("SH 2-155") == n("Sh2-155") == n("sh2 155") == "Sh2-155"
    assert n("IC0011") == "IC 11" and n("Mel022") == "Mel 22" and n("Cl399") == "Cr 399" and n("C 049") == "C 49"
    assert n("NGC85A") == "NGC 85A" and n("ESO056-115") == "ESO 56-115"
    assert n("Crab nebula") is None and n("30 Dor Cluster") is None


# ---------------------------------------------------------------------------------------------- the merge

def test_a_messier_object_and_its_ngc_duplicates_are_one_object(full):
    m8 = full.find("NGC 6533")
    assert m8 is not None and m8.name == "M 8"
    assert {"NGC 6523", "NGC 6533", "Sh2-25"} <= set(m8.aliases)
    assert m8.common_name == "Lagoon Nebula" and m8.kind == "nebula"
    assert sum(1 for o in full.objects() if _designations(o) & {"M 8", "NGC 6523", "NGC 6533", "Sh2-25"}) == 1
    # OpenNGC's main row is NGC 2239, everybody says NGC 2244
    rosette_cl = full.find("NGC 2239")
    assert rosette_cl.name == "NGC 2244" and {"NGC 2239", "C 50"} <= set(rosette_cl.aliases)
    # a curated name and links the catalogues lack
    heart = full.find("Sh2-190")
    assert heart.name == "IC 1805" and heart.common_name == "Heart Nebula" and "Mel 15" in heart.aliases
    witch = full.find("NGC 1909")
    assert witch.name == "IC 2118" and witch.common_name == "Witch Head Nebula"
    cave = full.find("C 9")
    assert cave.name == "Sh2-155" and cave.common_name == "Cave Nebula"
    # stars and non-existent objects are left out, a Messier double star is kept
    assert full.find("IC 1") is None and full.find("IC 2") is None
    assert full.find("M 40").kind == "double star"
    assert full.find("m31").name == "M 31"


def test_a_cross_identification_between_far_objects_is_dropped(full):
    # Siril's list says Sh2-116 (2') is NGC 7000 (4 degrees away), and Sh2-271 is NGC 2024 (16 degrees away)
    assert full.find("Sh2-116").name == "Sh2-116" and full.find("Sh2-116").aliases == ()
    assert full.find("Sh2-117").name == "NGC 7000"
    assert full.find("Sh2-271").name == "Sh2-271"
    assert "Sh2-271" not in full.find("NGC 2024").aliases


def test_sizes_angles_positions_and_magnitudes(full):
    m31 = full.find("M 31")
    assert (m31.major, m31.minor, m31.angle, m31.mag) == (177.83, 69.66, 35.0, 3.44)      # V before B
    assert m31.ra == pytest.approx((42 + 44.35 / 60) / 60 * 15) and m31.dec == pytest.approx(41 + 16 / 60 + 8.6 / 3600)
    assert full.find("M 8").dec == pytest.approx(-(24 + 22 / 60 + 48.6 / 3600))
    cl = full.find("NGC 2244")
    assert cl.major == cl.minor == 9.3 and cl.angle == 0.0                 # one diameter: round
    b33 = full.find("B 33")
    assert (b33.major, b33.minor, b33.angle, b33.kind, b33.common_name) == (6.0, 4.0, 90.0, "dark nebula",
                                                                            "Horsehead Nebula")
    sh116 = full.find("Sh2-116")
    assert (sh116.major, sh116.minor, sh116.kind, sh116.mag) == (2.0, 2.0, "HII region", None)
    assert full.find("NGC 7000").major == 120.0                          # OpenNGC's axes before the Sharpless diameter
    assert "OpenNGC" in full.source and "VII/20" in full.source and "Siril" in full.source


def test_siril_lists_alone_name_the_objects_without_types(sources):
    cat = so.build_catalogue(None, None, sources["siril"])
    m31 = cat.find("NGC 224")
    assert m31.name == "M 31" and m31.common_name == "Andromeda Galaxy" and m31.kind == "other"
    assert m31.major == m31.minor == 178.0
    assert cat.find("Sh2-190").name == "IC 1805" and cat.find("IC 1805").common_name == "Heart Nebula"
    assert cat.find("Sh2-25").name == "M 8" and cat.find("Sh2-116").aliases == ()
    assert cat.find("Sh2-117").kind == "HII region"
    assert cat.source.startswith("Siril (")


# ---------------------------------------------------------------------------------------------- the objects of a field

def _tan(ra0: float, dec0: float, w: int, h: int, scale_deg: float):
    from astropy.wcs import WCS
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra0, dec0]
    wcs.wcs.crpix = [(w + 1) / 2, (h + 1) / 2]
    wcs.wcs.cdelt = [-scale_deg, scale_deg]
    return wcs


def test_objects_in_field_by_centre_or_outline_biggest_first():
    W, H = 1000, 800
    wcs = _tan(150.0, 30.0, W, H, 0.001)                                # 1 x 0.8 deg, 3.6"/px, north up

    def at(x, y, name, major, minor=None, angle=0.0):
        ra, dec = wcs.all_pix2world(x, y, 0)
        return so.SkyObject(name, "", "nebula", float(ra), float(dec), major, major if minor is None else minor, angle)
    objs = [
        at(500, 400, "NGC 1", 1.0),                         # inside
        at(5000, 400, "NGC 2", 10.0),                       # 4.5 deg away
        at(1100, 400, "NGC 3", 30.0),                       # centre 6' right of the picture, 15' radius: overlaps
        at(1100, 300, "NGC 4", 4.0),                        # centre 6' out, 2' radius: does not
        at(500, 900, "NGC 5", 30.0, 1.0, 0.0),              # 6' above, long north-south: reaches in
        at(500, 900, "NGC 6", 30.0, 1.0, 90.0),             # same, east-west: does not
        at(-1500, 400, "NGC 7", 360.0),                     # 2 deg away, 3 deg radius: the picture is inside it
        at(1010, 400, "NGC 8", 0.0),                        # size unknown, 10 px out
    ]
    got = [o.name for o in so.objects_in_field(wcs, W, H, objs)]
    assert got == ["NGC 7", "NGC 3", "NGC 5", "NGC 1"]
    assert [o.name for o in so.objects_in_field(wcs, W, H, so.Catalogue.from_objects(objs), margin=20)] == \
        ["NGC 7", "NGC 3", "NGC 5", "NGC 1", "NGC 8"]
    assert so.objects_in_field(wcs, W, H, []) == []
    # the returned objects are the catalogue's, whole
    assert so.objects_in_field(wcs, W, H, objs)[1] == objs[2]


def test_objects_of_a_field_from_the_merged_catalogue(full):
    m31 = full.find("M 31")
    wcs = _tan(m31.ra + 1.6, m31.dec, 1000, 1000, 0.002)                # 2 x 2 deg, M 31's centre 1.2 deg outside
    got = so.objects_in_field(wcs, 1000, 1000, full)
    assert [o.name for o in got] == ["M 31"]                            # its 3 deg outline reaches in


# ---------------------------------------------------------------------------------------------- the cache

def _cache(home: Path, sources: dict) -> Path:
    d = home / "cache" / so.SUB
    d.mkdir(parents=True)
    shutil.copy(sources["openngc"][0], d / so.OPENNGC_FILES["NGC.csv"])
    shutil.copy(sources["openngc"][1], d / so.OPENNGC_FILES["addendum.csv"])
    shutil.copy(sources["sharpless"], d / so.VII20_FILE)
    return d


def test_the_merged_catalogue_is_cached_and_made_again_when_a_source_changes(home, sources):
    d = _cache(home, sources)
    first = so.load_catalogue()
    assert "OpenNGC" in first.source and "VII/20" in first.source and first.notes == ()
    pk = list(d.glob("catalogue_*.pkl"))
    assert len(pk) == 1
    assert so.load_catalogue() is first                                # in memory
    so.reset()
    t = time.perf_counter()
    again = so.load_catalogue()
    assert time.perf_counter() - t < 0.2
    assert again is not first and again.names == first.names and again.source == first.source
    assert np.array_equal(again.major, first.major)
    # a source that changes: made again, the old merged file goes
    (d / so.VII20_FILE).write_text(VII20.replace("  25\t  90", "  25\t  95"), encoding="utf-8")
    os.utime(d / so.VII20_FILE, (time.time() + 5, time.time() + 5))
    so.reset()
    assert so.load_catalogue().find("Sh2-116").major == 2.0
    pk2 = list(d.glob("catalogue_*.pkl"))
    assert len(pk2) == 1 and pk2 != pk
    assert so.catalogue_source() == first.source


def test_without_any_catalogue_a_plain_message_says_why(home):
    with pytest.raises(so.CatalogueUnavailable, match="offline mode"):
        so.load_catalogue()
    with pytest.raises(so.CatalogueUnavailable, match="install Siril"):
        so.objects_in_field(_tan(10, 10, 100, 100, 0.01), 100, 100)
    assert so.catalogue_source() == ""
    assert api.CatalogueUnavailable is so.CatalogueUnavailable and api.SkyObject is so.SkyObject


def test_a_missing_source_is_noted_and_siril_fills_in(home, sources, monkeypatch):
    monkeypatch.setattr(so, "siril_catalogue_dir", lambda: sources["siril"])
    cat = so.load_catalogue()
    assert cat.source.startswith("Siril (") and cat.find("IC 1805").common_name == "Heart Nebula"
    assert any("OpenNGC could not be had (offline mode)" in n for n in cat.notes)
    assert any("Sharpless" in n and "without their sizes" in n for n in cat.notes)


def test_a_failed_download_is_not_tried_again_for_a_while(home, monkeypatch):
    monkeypatch.delenv("ASTRO_GPU_OFFLINE")
    calls = []

    def down():
        calls.append(1)
        raise OSError("no network")
    monkeypatch.setattr(so, "_download_openngc", down)
    monkeypatch.setattr(so, "_download_vii20", lambda say: down())
    with pytest.raises(so.CatalogueUnavailable, match="no network"):
        so.load_catalogue()
    so.reset()                                                           # as another process: the file remembers
    with pytest.raises(so.CatalogueUnavailable, match="failed less than 30 minutes ago"):
        so.load_catalogue()
    assert len(calls) == 2                                              # once each source, not once per call
    assert (home / "cache" / so.SUB / so.FAILED_FILE).is_file()


def test_siril_installed_here_gives_a_fast_catalogue(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))       # the real Siril folder, offline
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")
    monkeypatch.setattr(skycache, "legacy_dirs", lambda: [])
    if so.siril_catalogue_dir() is None:
        pytest.skip("Siril is not installed")
    cat = so.load_catalogue()
    assert len(cat) > 10000 and cat.find("NGC 224").name == "M 31"
    so.reset()
    t = time.perf_counter()
    assert len(so.load_catalogue()) == len(cat)
    assert time.perf_counter() - t < 0.2


# ---------------------------------------------------------------------------------------------- the API

def test_api_sky_objects_reads_the_solution_and_the_stack_grid(home, sources, tmp_path, monkeypatch):
    """The store is stubbed (``_dir`` / ``_upstream``): the solution of an astrometry folder, the grid of its stack."""
    from astropy.io import fits
    _cache(home, sources)
    astro, stack = tmp_path / "store" / "astrometry", tmp_path / "store" / "stack"
    astro.mkdir(parents=True)
    stack.mkdir(parents=True)
    heart = so.build_catalogue(sources["openngc"], sources["sharpless"]).find("IC 1805")
    fits.PrimaryHDU(header=_tan(heart.ra, heart.dec, 1000, 800, 0.002).to_header()).writeto(astro / "wcs.fits")
    (stack / "drz_meta.json").write_text('{"width": 1000, "height": 800}', encoding="utf-8")
    done = {"astrometry": astro}
    monkeypatch.setattr(api, "_dir", lambda p, wf, stage, which="current": done.get(stage))
    monkeypatch.setattr(api, "_upstream", lambda p, wf, d, up, which: stack if (stack / "drz_meta.json").exists() else None)
    objs = api.sky_objects(None, 1)
    assert [o.name for o in objs] == ["IC 1805"] and objs[0].common_name == "Heart Nebula"
    # an older stack without the size in its metadata: the shape of drz_full (10 x 8 px: a corner 1.3 deg from the
    # Heart's centre, outside its 30' radius)
    (stack / "drz_meta.json").write_text('{"scale": 1.5}', encoding="utf-8")
    np.save(stack / "drz_full.npy", np.zeros((3, 8, 10), np.uint8))
    assert api.sky_objects(None, 1) == []
    (stack / "drz_meta.json").unlink()
    with pytest.raises(api.NotDone, match="stack"):
        api.sky_objects(None, 1)
    (astro / "wcs.fits").unlink()
    with pytest.raises(api.NotDone, match="no solution"):
        api.sky_objects(None, 1)
    done.clear()
    with pytest.raises(api.NotDone, match="astrometry result of workflow 1 is not computed"):
        api.sky_objects(None, 1)


HEART = Path("F:/astro_essais/v2/heart_a7s_tour")


@pytest.mark.level3                                       # real data (tests/README.md)
@pytest.mark.skipif(not (HEART / "project.astroproj").is_file(), reason="the heart_a7s_tour test project is not here")
def test_real_heart_field_names_the_heart_nebula(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")                      # no network in a test: Siril or a cache
    try:
        so.load_catalogue()
    except so.CatalogueUnavailable:
        pytest.skip("no catalogue offline here (Siril not installed, nothing cached)")
    p = api.open_project(HEART)                                         # read only
    wid = next(w.id for w in api.workflows(p) if api.stage_outputs(p, w.id, "astrometry") is not None)
    objs = api.sky_objects(p, wid)
    by = {o.name: o for o in objs}
    assert by["IC 1805"].common_name == "Heart Nebula" and "Sh2-190" in by["IC 1805"].aliases
    assert {"IC 1848", "NGC 896", "NGC 1027"} <= set(by)
    assert objs[0].major >= objs[-1].major


@pytest.mark.network
def test_real_download_of_the_catalogues(home, monkeypatch):
    monkeypatch.delenv("ASTRO_GPU_OFFLINE")
    cat = so.load_catalogue(log=print)
    assert "OpenNGC" in cat.source and len(cat) > 11000
    assert cat.find("IC 1805").kind == "cluster with nebula" and cat.find("M 31").angle == 35.0
    if "VII/20" in cat.source:
        assert cat.find("Sh2-190").name == "IC 1805" and cat.find("Sh2-129").major > 0
