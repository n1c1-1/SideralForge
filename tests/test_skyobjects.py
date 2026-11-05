"""The named deep-sky objects (core/skyobjects.py, api.sky_objects): designations, the merge of the catalogues into one
object per physical object, sizes and angles, the objects of a field on a synthetic TAN solution, the cache.

No network: tiny catalogues written in tmp_path in the formats of OpenNGC, VizieR's VII/20 answer and Steinicke's
Revised NGC/IC, and the catalogues shipped with the package (astro_gpu/resources/catalogues), read with no other
program's folders; the real test project (F:/astro_essais) is level 3."""
from __future__ import annotations

import os
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
    kw = {k.replace("_", "-").replace("Common-names", "Common names").replace("NED-notes", "NED notes"): v
          for k, v in kw.items()}
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
    _row(Name="NGC5457", Type="G", RA="14:03:12.58", Dec="+54:20:55.5", MajAx="24.00", M="101"),
    _row(Name="NGC5866", Type="G", RA="15:06:29.50", Dec="+55:45:47.6", MajAx="6.31", MinAx="2.72"),
    _row(Name="NGC5194", Type="G", RA="13:29:52.71", Dec="+47:11:42.9", MajAx="13.71", M="051",
         Common_names="Whirlpool Galaxy"),
    _row(Name="NGC5195", Type="G", RA="13:29:59.59", Dec="+47:15:58.1", MajAx="5.48"),
    _row(Name="NGC1055", Type="G", RA="02:41:45.23", Dec="+00:26:35.4", MajAx="7.24"),
]
ADDENDUM_ROWS = [
    _row(Name="B033", Type="DrkN", RA="05:40:59.0", Dec="-02:27:30", MajAx="6.00", MinAx="4.00", PosAng="90",
         Common_names="Horsehead Nebula"),
    _row(Name="C009", Type="HII", RA="22:57:54.0", Dec="+62:31:06", MajAx="50.00", MinAx="30.00", PosAng="90",
         Identifiers="LBN 529,SH 2-155", Common_names="Cave Nebula"),
    _row(Name="M040", Type="**", RA="12:22:16.1", Dec="+58:05:04", M="040", Identifiers="WDS J12222+5805AB"),
    _row(Name="M102", Type="Dup", RA="14:03:12.54", Dec="+54:20:56.2", M="101",
         NED_notes="Identification is controversial, here we take NED assumption."),
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
ST_COLS = ("N,NI,A,C,D,S,P,CON,RH,RM,RS,V,DG,DM,DS,BMAG,VMAG,B-V,SB,X,Y,PA,TYPE,z,D(z),Dist,PGC,"
           + ",".join(f"ID{k}" for k in range(1, 12)))


def _st(cat, num, ra, dec, *ids, size="", comp="", dreyer="*", status=1) -> str:
    """A row of Steinicke's sheet as CSV: ra / dec in degrees, written as h m s / d m s."""
    h = ra / 15.0
    rh, rm = int(h), int((h - int(h)) * 60)
    rs = round(((h - rh) * 60 - rm) * 60, 1)
    sign, d = ("-" if dec < 0 else "+"), abs(dec)
    dg, dm = int(d), int((d - int(d)) * 60)
    ds = round(((d - dg) * 60 - dm) * 60)
    row = [cat, str(num), "", comp, dreyer, str(status), "", "", str(rh), str(rm), str(rs), sign, str(dg), str(dm),
           str(ds), "", "", "", "", size, "", "", "", "", "", "", ""]
    row += ['"' + i.replace('"', '""') + '"' for i in ids]
    return ",".join(row + [""] * (38 - len(row)))


STEINICKE = "\n".join([
    ST_COLS,
    _st("N", 224, 10.6846, 41.2690, "M 31", "UGC 454", "Andromeda nebula", size="190"),
    _st("N", 6523, 270.925, -24.38, "M 8", "Sh2-25", "Lagoon nebula", size="45", status=2),
    _st("N", 6533, 270.925, -24.38, "NGC 6523", "M 8", status=7, dreyer=""),
    _st("N", 7000, 314.8, 44.52, "Sh2-116", "North America nebula", size="120", status=2),   # Sh2-116: 4 deg away
    _st("N", 2024, 85.42, -1.85, "Sh2-271", "Flame nebula", size="30", status=2),            # Sh2-271: 16 deg away
    _st("N", 5866, 226.6229, 55.7633, "M 102", "UGC 9723", size="6"),
    _st("N", 5194, 202.4696, 47.1952, "M 51", "Whirlpool galaxy", size="11"),
    _st("N", 5195, 202.4983, 47.2661, "UGC 8494", "Whirlpool galaxy", size="6"),           # the pair's name: not its own
    _st("N", 1055, 40.4383, 0.4431, "UGC 2173", '"Sombrero"', size="7"),                    # a likeness, in quotes
    _st("N", 2238, 97.7, 5.0, "LBN 948", "knot in Rosette nebula", status=2),
    _st("N", 2239, 97.98, 4.94, "NGC 2244", "OCL 515", "in Rosette nebula", status=7, dreyer=""),
    _st("N", 2239, 97.98, 4.94, "M 99", comp="2", dreyer=""),                               # a component: no link
]) + "\n"


@pytest.fixture
def sources(tmp_path) -> dict:
    """The tiny sources: OpenNGC's two files, VizieR's VII/20 answer, Steinicke's sheet as CSV."""
    d = tmp_path / "src"
    d.mkdir(parents=True)
    head = ";".join(COLS) + "\n"
    (d / "NGC.csv").write_text(head + "\n".join(NGC_ROWS) + "\n", encoding="utf-8")
    (d / "addendum.csv").write_text(head + "\n".join(ADDENDUM_ROWS) + "\n", encoding="utf-8")
    (d / "vii20.tsv").write_text(VII20, encoding="utf-8")
    (d / "ngcic.csv").write_text(STEINICKE, encoding="utf-8")
    return {"openngc": [d / "NGC.csv", d / "addendum.csv"], "sharpless": d / "vii20.tsv", "steinicke": d / "ngcic.csv"}


@pytest.fixture
def full(sources) -> so.Catalogue:
    return so.build_catalogue(sources["openngc"], sources["sharpless"], sources["steinicke"])


@pytest.fixture
def shipped_tiny(sources, monkeypatch) -> dict:
    """The tiny sources in place of the shipped files (load_catalogue reads them)."""
    monkeypatch.setattr(so, "OPENNGC_FILES", tuple(sources["openngc"]))
    monkeypatch.setattr(so, "VII20_FILE", sources["sharpless"])
    monkeypatch.setattr(so, "STEINICKE_FILE", sources["steinicke"])
    return sources


@pytest.fixture(autouse=True)
def _fresh():
    so.reset()
    yield
    so.reset()


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """A private, empty ASTRO_GPU_HOME, offline, without earlier versions' cache, and no other program's folders
    (Program Files, LOCALAPPDATA, APPDATA, XDG: an empty folder): the catalogue is the package's alone."""
    h = tmp_path / "home"
    monkeypatch.setenv("ASTRO_GPU_HOME", str(h))
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")
    monkeypatch.setattr(skycache, "legacy_dirs", lambda: [])
    empty = tmp_path / "empty"
    empty.mkdir()
    for k in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)", "LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME",
              "XDG_CONFIG_HOME"):
        monkeypatch.setenv(k, str(empty))
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
    # the tiny Steinicke file says Sh2-116 (2') is NGC 7000 (4 degrees away), and Sh2-271 is NGC 2024 (16 degrees)
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
    assert "OpenNGC" in full.source and "VII/20" in full.source and "Steinicke" in full.source
    assert "Siril" not in full.source


def test_steinickes_links_and_names(full, sources):
    # M 102: OpenNGC makes it a duplicate of M 101, Steinicke says NGC 5866; Steinicke's wins, M 101 stays alone
    assert full.find("NGC 5866").name == "M 102" and full.find("M 101").aliases == ("NGC 5457",)
    without = so.build_catalogue(sources["openngc"], sources["sharpless"], None)
    assert without.find("M 102").name == "M 101" and without.find("NGC 5866").name == "NGC 5866"
    # a common name of one object is taken; a pair's ("Whirlpool galaxy" for NGC 5195 too) and a likeness in quotes
    # ("Sombrero" for NGC 1055) are not; OpenNGC's names come first
    assert full.find("NGC 2024").common_name == "Flame Nebula"
    assert full.find("NGC 5195").common_name == "" and full.find("NGC 1055").common_name == ""
    assert full.find("M 51").common_name == "Whirlpool Galaxy"
    # "in Rosette nebula", "knot in ...", and a component's identifications are other objects: no link
    assert "M 99" not in full.find("NGC 2244").aliases and full.find("M 99") is None
    recs, links, common = so.read_steinicke(sources["steinicke"])
    assert ("NGC 6533", "M 8", True) in links and common["NGC 6523"] == "Lagoon Nebula"
    assert "NGC 5195" not in common and "NGC 1055" not in common
    assert len(recs) == 11 and recs[0].ra == pytest.approx(10.6846, abs=1e-3)


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

def test_the_merged_catalogue_is_cached_and_made_again_when_a_source_changes(home, shipped_tiny):
    d = home / "cache" / so.SUB
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
    vii = shipped_tiny["sharpless"]
    vii.write_text(VII20.replace("  25\t  90", "  25\t  95"), encoding="utf-8")
    os.utime(vii, (time.time() + 5, time.time() + 5))
    so.reset()
    assert so.load_catalogue().find("Sh2-116").major == 2.0
    pk2 = list(d.glob("catalogue_*.pkl"))
    assert len(pk2) == 1 and pk2 != pk
    assert so.catalogue_source() == first.source


def test_a_missing_shipped_file_says_so_plainly(home, shipped_tiny):
    shipped_tiny["steinicke"].unlink()
    with pytest.raises(so.CatalogueUnavailable, match="install the program again"):
        so.load_catalogue()
    with pytest.raises(so.CatalogueUnavailable, match="ngcic"):
        so.objects_in_field(_tan(10, 10, 100, 100, 0.01), 100, 100)
    assert so.catalogue_source() == ""
    assert api.CatalogueUnavailable is so.CatalogueUnavailable and api.SkyObject is so.SkyObject


# ---------------------------------------------------------------------------------------------- the shipped catalogue

#: the bench fields (F:/astro_essais/bench, from their solutions: centre, size, the CD matrix x 1e4 deg/px) and objects
#: the client names there (2 November 2026, shipped catalogue: the same names as with the earlier sources)
BENCH = {
    "iris": ((316.0233, 68.0490, 6616, 4355, (3.53240, -0.48732, 0.48602, 3.53332)),
             {"NGC 7023": "Iris Nebula", "Sh2-136": ""}),
    "rosette": ((97.9957, 5.0007, 4252, 2762, (2.16876, -4.89042, 4.89150, 2.16893)),
                {"NGC 2237": "Rosette Nebula", "NGC 2244": "", "NGC 2246": "Rosette B"}),
    "heart": ((41.2947, 61.3245, 4182, 3213, (7.75017, 15.90311, -15.90210, 7.74472)),
              {"IC 1805": "Heart Nebula", "IC 1848": "Soul Nebula", "NGC 896": "Fish Head Nebula", "NGC 1027": "",
               "IC 1831": ""}),
    "m51": ((202.6005, 47.0171, 5609, 3577, (-3.02008, -2.22000, 2.21961, -3.02025)),
            {"M 51": "Whirlpool Galaxy", "NGC 5195": ""}),
    "m8": ((270.9573, -23.4964, 6285, 4252, (-1.70445, -5.27904, 5.28608, -1.70921)),
           {"M 8": "Lagoon Nebula", "M 20": "Trifid Nebula", "M 21": "", "NGC 6530": ""}),
    "seestar": ((10.6886, 41.1648, 987, 1904, (-3.16169, -5.78428, 5.78995, -3.16266)),
                {"M 31": "Andromeda Galaxy", "M 32": "", "NGC 206": ""}),
}


def _tan_cd(ra0, dec0, w, h, cd):
    wcs = _tan(ra0, dec0, w, h, 1.0)
    wcs.wcs.cdelt = [1.0, 1.0]
    wcs.wcs.pc = np.reshape(cd, (2, 2)) * 1e-4
    return wcs


@pytest.fixture(scope="module")
def shipped():
    """The catalogue made from the shipped files alone (build_catalogue's defaults: no cache, no network)."""
    return so.build_catalogue()


def test_the_shipped_catalogue_needs_nothing_else(home):
    root = Path(so.__file__).resolve().parents[1] / "resources" / "catalogues"
    assert so.LYNDS_FILE.is_file()
    for f in so._files():
        assert f.is_file() and f.is_relative_to(root)
    cat = so.load_catalogue()
    assert len(cat) > 12000 and cat.notes == ()
    src = so.catalogue_source()
    assert "OpenNGC" in src and "Steinicke" in src and "VII/20" in src and "Siril" not in src
    for folder in ("openngc", "steinicke", "sharpless", "lynds"):          # each with its origin and terms
        readme = (root / folder / "README.md").read_text(encoding="utf-8")
        assert "Origin" in readme and "Version" in readme and ("Licence" in readme or "Terms" in readme)
    assert "steinicke-zehnle@t-online.de" in (root / "steinicke" / "README.md").read_text(encoding="utf-8")
    assert "Attribution-ShareAlike 4.0" in (root / "openngc" / "LICENSE.txt").read_text(encoding="utf-8")


def test_the_110_messier_objects(shipped):
    seen = set()
    for n in range(1, 111):
        o = shipped.find(f"M {n}")
        assert o is not None and o.name == f"M {n}", n
        seen.add((round(o.ra, 4), round(o.dec, 4)))
    assert len(seen) == 110                                             # 110 different objects
    assert shipped.find("NGC 5866").name == "M 102" and shipped.find("NGC 5457").name == "M 101"
    assert shipped.find("Mel 22").name == "M 45" and shipped.find("M 40").kind == "double star"
    assert shipped.find("IC 4715").name == "M 24" and shipped.find("IC 4725").name == "M 25"
    # Steinicke and OpenNGC agree on every Messier number Steinicke gives (all but M 40 and M 45)
    _recs, links, _common = so.read_steinicke(so.STEINICKE_FILE)
    st = {b: a for a, b, _c in links if b.startswith("M ")}
    assert len(st) == 108 and not {"M 40", "M 45"} & set(st)
    for m, d in st.items():
        assert shipped.find(d).name == m, (m, d)


@pytest.mark.parametrize("field", list(BENCH))
def test_the_bench_fields_are_named(shipped, field):
    (ra, dec, w, h, cd), expected = BENCH[field]
    got = {o.name: o for o in so.objects_in_field(_tan_cd(ra, dec, w, h, cd), w, h, shipped)}
    for name, common in expected.items():
        assert name in got, (field, name, sorted(got))
        assert got[name].common_name == common, (field, name, got[name].common_name)


def test_the_target_search_index(home):
    idx = so.names_index()
    assert idx["Whirlpool Galaxy"][2] == "M 51" and idx["Heart Nebula"][2] == "IC 1805"
    assert idx["LDN 1622"][:2] == pytest.approx((88.6503, 2.0081))
    assert len([k for k in idx if k.startswith("LDN ")]) > 1700
    assert idx["Sh2-155"][2] == "Sh2-155" and idx["NGC 5866"][2] == "M 102"


# ---------------------------------------------------------------------------------------------- the API

def test_api_sky_objects_reads_the_solution_and_the_stack_grid(home, shipped_tiny, tmp_path, monkeypatch):
    """The store is stubbed (``_dir`` / ``_upstream``): the solution of an astrometry folder, the grid of its stack."""
    from astropy.io import fits
    sources = shipped_tiny
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


from conftest import TEST_PROJECT as HEART  # noqa: E402  (the maintained test project, read only)


@pytest.mark.level3                                       # real data (tests/README.md)
@pytest.mark.skipif(not (HEART / "project.astroproj").is_file(),
                    reason="the test project is not here: run tools/make_test_project.py")
def test_real_heart_field_names_the_heart_nebula(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")                      # no network: the shipped catalogue
    p = api.open_project(HEART)                                         # read only
    wid = next(w.id for w in api.workflows(p) if api.stage_outputs(p, w.id, "astrometry") is not None)
    objs = api.sky_objects(p, wid)
    by = {o.name: o for o in objs}
    assert by["IC 1805"].common_name == "Heart Nebula" and "Sh2-190" in by["IC 1805"].aliases
    assert {"IC 1848", "NGC 896", "NGC 1027"} <= set(by)
    assert objs[0].major >= objs[-1].major
