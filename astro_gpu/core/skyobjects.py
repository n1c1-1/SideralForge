"""The named deep-sky objects of a field (galaxies, nebulae, clusters: Messier, NGC, IC, Sharpless), offline.

The catalogue is made of these sources, each optional, merged into one object per physical object:

- OpenNGC (github.com/mattiaverga/OpenNGC, CC-BY-SA 4.0): ``NGC.csv`` + ``addendum.csv``, every NGC and IC object with
  its type, axes, position angle, Messier number and common names. Downloaded once (4 MB) into
  ``$ASTRO_GPU_HOME/cache/skyobjects``.
- The Sharpless catalogue of HII regions (Sharpless 1959, VizieR VII/20): 313 regions with their diameter. Downloaded
  once (35 kB) from VizieR (Strasbourg, else its Harvard mirror) into the same folder.
- Siril's deep-sky lists when Siril is installed (``share/siril/catalogue``: messier, ngc, ic, sh2 .csv: a position, one
  diameter, cross-identifications): the NGC / IC / Messier objects when OpenNGC cannot be had (no types then, round
  outlines), the Sharpless positions when VII/20 cannot be had, and their cross-identifications always.
- ``CURATED``: well-known English names the catalogues lack (the Heart Nebula is plain "IC 1805" in all of them) and the
  Sharpless / NGC links they miss.

Earlier versions' cache (``~/.astro_gpu/cache``) is read too (skycache). ``ASTRO_GPU_OFFLINE=1``: nothing is downloaded.
A download that failed is not tried again for half an hour, by any process (``downloads.json`` in the cache folder): the
catalogue is then made of what there is, and its ``notes`` say what is missing.

A cross-identification between two objects whose centres are further apart than about their size is dropped (Siril's
list says Sh2-116, 2' wide, is NGC 7000, 4 degrees away).

Names: Messier > NGC > IC > Sharpless > the rest ("M 42", "NGC 2244", "IC 1805", "Sh2-275"); the other designations of
the object are its ``aliases``. Sizes in arcminutes, position angle in degrees east of north, RA / Dec in degrees J2000.

The merged catalogue is kept as columns in ``cache/skyobjects/catalogue_<key>.pkl`` (loaded in a few tens of ms) and in
memory; it is made again when a source file changes. No source at all: ``CatalogueUnavailable``, with a plain message.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BUILD_VERSION = "2"                 # the merged catalogue's format and rules: bump to make it again
SUB = "skyobjects"                  # the cache sub-folder
OPENNGC_URL = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/{}"
OPENNGC_FILES = {"NGC.csv": "openngc_NGC.csv", "addendum.csv": "openngc_addendum.csv"}     # remote -> cached name
VII20_FILE = "sharpless_vii20.tsv"
VII20_QUERY = ("https://{host}/viz-bin/asu-tsv?-source=VII/20/catalog&-out=Sh2,Diam&-out.add=_RAJ2000,_DEJ2000"
               "&-oc.form=d&-out.max=unlimited")
DOWNLOAD_BUDGET_S = 120.0           # seconds a download may take
RETRY_S = 1800.0                    # a download that failed is tried again after this long (noted in downloads.json)
LINK_MIN_ARCMIN = 20.0              # two cross-identified objects may be this far apart, or 3/4 of the larger one's size
OUTLINE_POINTS = 48

#: OpenNGC's object types, in plain words (``SkyObject.kind``); the Sharpless regions are "HII region", an object of
#: Siril's lists (no type) "other"
KINDS = {"G": "galaxy", "GPair": "galaxy pair", "GTrpl": "galaxy triplet", "GGroup": "galaxy group",
         "OCl": "open cluster", "GCl": "globular cluster", "Cl+N": "cluster with nebula", "*Ass": "star association",
         "PN": "planetary nebula", "HII": "HII region", "EmN": "emission nebula", "Neb": "nebula",
         "RfN": "reflection nebula", "DrkN": "dark nebula", "SNR": "supernova remnant", "*": "star",
         "**": "double star", "Other": "other"}
#: OpenNGC types left out (stars, nothing found at the position...) unless the object has a Messier number
LEFT_OUT = frozenset({"*", "**", "Other", "NonEx", "Nova"})

#: (designation, English common name or "", other designations of the same object): names the catalogues lack or give
#: worse, and links they miss. A common name here wins over the catalogues'. Links are checked on the positions too.
CURATED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("IC 1805", "Heart Nebula", ("Sh2-190", "Mel 15")),
    ("IC 1848", "Soul Nebula", ("Sh2-199",)),
    ("NGC 896", "Fish Head Nebula", ("IC 1795",)),
    ("NGC 281", "Pacman Nebula", ("Sh2-184",)),
    ("IC 1396", "Elephant's Trunk Nebula", ("Sh2-131",)),
    ("NGC 7380", "Wizard Nebula", ("Sh2-142",)),
    ("IC 443", "Jellyfish Nebula", ("Sh2-248",)),
    ("IC 410", "Tadpoles Nebula", ("Sh2-236",)),
    ("IC 2177", "Seagull Nebula", ("Sh2-296",)),
    ("NGC 2359", "Thor's Helmet", ("Sh2-298",)),
    ("IC 4628", "Prawn Nebula", ()),
    ("IC 63", "Ghost of Cassiopeia", ()),
    ("IC 2118", "Witch Head Nebula", ()),
    ("NGC 6334", "Cat's Paw Nebula", ("Sh2-8",)),
    ("NGC 6357", "War and Peace Nebula", ("Sh2-11",)),
    ("NGC 6302", "Bug Nebula", ("Sh2-6",)),
    ("M 1", "Crab Nebula", ("Sh2-244",)),
    ("M 8", "Lagoon Nebula", ("Sh2-25",)),
    ("M 16", "Eagle Nebula", ("Sh2-49",)),
    ("M 17", "Omega Nebula", ("Sh2-45",)),
    ("M 20", "Trifid Nebula", ("Sh2-30",)),
    ("M 42", "Orion Nebula", ("Sh2-281",)),
    ("M 101", "Pinwheel Galaxy", ()),
    ("NGC 6888", "Crescent Nebula", ("Sh2-105",)),
    ("NGC 7000", "North America Nebula", ("Sh2-117",)),
    ("NGC 7635", "Bubble Nebula", ("Sh2-162",)),
    ("NGC 1499", "California Nebula", ("Sh2-220",)),
    ("IC 405", "Flaming Star Nebula", ("Sh2-229",)),
    ("NGC 1555", "Hind's Variable Nebula", ("Sh2-238",)),
    ("NGC 2024", "Flame Nebula", ("Sh2-277",)),
    ("NGC 1977", "Running Man Nebula", ("Sh2-279",)),
    ("IC 5146", "Cocoon Nebula", ("Sh2-125",)),
    ("NGC 2174", "Monkey Head Nebula", ("Sh2-252",)),
    ("NGC 2467", "Skull and Crossbones Nebula", ("Sh2-311",)),
    ("NGC 2237", "Rosette Nebula", ("NGC 2238", "Sh2-275")),
    ("Sh2-155", "Cave Nebula", ("C 9",)),
    ("Sh2-276", "Barnard's Loop", ()),
    ("Sh2-264", "Angelfish Nebula", ()),
    ("Sh2-240", "Spaghetti Nebula", ()),
    ("Sh2-274", "Medusa Nebula", ()),
    ("Sh2-129", "Flying Bat Nebula", ()),
    ("Sh2-132", "Lion Nebula", ()),
    ("Sh2-101", "Tulip Nebula", ()),
    ("Sh2-157", "Lobster Claw Nebula", ()),
    ("Sh2-261", "Lower's Nebula", ()),
    ("Sh2-308", "Dolphin Head Nebula", ()),
    ("Sh2-119", "Clamshell Nebula", ()),
    ("Sh2-106", "Snow Angel Nebula", ()),
)
#: the designation shown when an object has several of the same catalogue (OpenNGC keeps NGC 2239 for the Rosette
#: cluster, everybody says NGC 2244)
PREFERRED = frozenset({"NGC 2244", "IC 2118"})


class CatalogueUnavailable(Exception):
    """No catalogue of named objects can be had (offline, nothing cached, Siril not installed): the message says why
    and what to do, in plain words."""


@dataclass(frozen=True)
class SkyObject:
    """A named deep-sky object. ``name``: the designation shown (Messier > NGC > IC > Sharpless: "M 42", "NGC 2244",
    "IC 1805", "Sh2-275"); ``common_name``: English, or ""; ``kind``: "galaxy", "HII region", "open cluster"... (``KINDS``;
    "other" when unknown); ``ra``, ``dec``: degrees J2000; ``major``, ``minor``: axes in arcminutes (0: size unknown;
    ``minor`` = ``major`` for a round object or when only one diameter is known); ``angle``: position angle of the major
    axis, degrees east of north (0 when unknown); ``aliases``: the object's other designations; ``mag``: visual (else
    blue) magnitude, None when unknown."""
    name: str
    common_name: str
    kind: str
    ra: float
    dec: float
    major: float
    minor: float
    angle: float
    aliases: tuple[str, ...] = ()
    mag: float | None = None


# ------------------------------------------------------------------ designations

_SH2_RE = re.compile(r"^\s*sh\s*2\s*[-\s]?\s*0*(\d+)\s*$", re.I)
_DES_RE = re.compile(r"^\s*([A-Za-z]{1,6})\s*[-\s]?\s*0*(\d+)\s*([A-Za-z]{0,2}(?:-\d+)?)\s*$")
_PREFIX = {"M": "M", "NGC": "NGC", "IC": "IC", "MEL": "Mel", "C": "C", "B": "B", "CL": "Cr", "CR": "Cr", "COL": "Cr",
           "H": "H"}
_CAT_RANK = {"M": 0, "NGC": 1, "IC": 2, "Sh2": 3, "C": 5}


def normalize(designation: str) -> str | None:
    """The designation as shown ("NGC0224", "ngc 224" -> "NGC 224"; "M031" -> "M 31"; "SH 2-155" -> "Sh2-155";
    "Mel022" -> "Mel 22"; "Cl399" -> "Cr 399"), or None when the text is not one (a common name)."""
    m = _SH2_RE.match(designation)
    if m:
        return f"Sh2-{int(m.group(1))}"
    m = _DES_RE.match(designation)
    if not m:
        return None
    pre = m.group(1).upper()
    return f"{_PREFIX.get(pre, pre)} {int(m.group(2))}{m.group(3).upper()}"


def _rank(d: str) -> tuple:
    """Sort key of designations: the preferred one first, then Messier, NGC, IC, Sharpless, the rest; by number."""
    if d.startswith("Sh2-"):
        pre, rest = "Sh2", d[4:]
    else:
        pre, _, rest = d.partition(" ")
    m = re.match(r"(\d+)(.*)", rest)
    num, tail = (int(m.group(1)), m.group(2)) if m else (0, rest)
    return (0 if d in PREFERRED else 1, _CAT_RANK.get(pre, 4), pre, num, tail)


# ------------------------------------------------------------------ the sources

@dataclass
class _Rec:
    """One object of one source."""
    desig: str
    ra: float
    dec: float
    major: float = 0.0
    minor: float = 0.0
    angle: float = 0.0
    kind: str = "other"
    mag: float | None = None
    common: str = ""


def _num(v: str | None) -> float | None:
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _sexa(v: str, hours: bool) -> float | None:
    """'02:32:41.51' (hours) / '+61:27:24.8' (degrees) -> degrees."""
    v = (v or "").strip()
    if not v:
        return None
    sign = -1.0 if v.startswith("-") else 1.0
    try:
        parts = [abs(float(x)) for x in v.lstrip("+-").split(":")]
    except ValueError:
        return None
    x = sum(p / 60.0 ** i for i, p in enumerate(parts))
    return sign * x * (15.0 if hours else 1.0)


def _title(s: str) -> str:
    """'Crab nebula' -> 'Crab Nebula' (Siril's names); a leading 'the ' goes (OpenNGC's 'the Witch Head Nebula')."""
    s = re.sub(r"^the\s+", "", s.strip(), flags=re.I)
    return " ".join(w[:1].upper() + w[1:] for w in s.split())


def _first_name(names: str) -> str:
    """The first of OpenNGC's comma-separated common names, without a leading 'the '."""
    n = (names or "").split(",")[0].strip()
    return re.sub(r"^the\s+", "", n, flags=re.I)


Link = tuple[str, str, bool]        # (designation, designation of the same object, check the positions)


def read_openngc(files: Iterable[Path]) -> tuple[list[_Rec], list[Link]]:
    """OpenNGC's NGC.csv / addendum.csv: the objects (stars and non-existent objects left out) and their
    cross-identifications (Messier, the other NGC / IC numbers, Sharpless, Caldwell; a "Dup" row links its designation
    to the object it duplicates)."""
    recs: list[_Rec] = []
    links: list[Link] = []
    for f in files:
        with open(f, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter=";"):
                name = normalize(row.get("Name") or "")
                if not name:
                    continue
                typ = (row.get("Type") or "").strip()
                m = (row.get("M") or "").strip()
                ids = [f"M {int(m)}"] if m.isdigit() else []
                for col in ("NGC", "IC"):
                    for v in (row.get(col) or "").split(","):
                        d = normalize(col + v.strip()) if v.strip() else None
                        if d and d != name:
                            ids.append(d)
                for ident in (row.get("Identifiers") or "").split(","):
                    ident = ident.strip()
                    if re.match(r"^(SH\s*2-|C\s+\d)", ident, re.I):
                        d = normalize(ident)
                        if d:
                            ids.append(d)
                links += [(name, d, False) for d in ids]
                if typ in ("Dup", "NonEx") or (typ in LEFT_OUT and not m.isdigit()):
                    continue
                ra, dec = _sexa(row.get("RA") or "", True), _sexa(row.get("Dec") or "", False)
                if ra is None or dec is None:
                    continue
                major, minor = _num(row.get("MajAx")) or 0.0, _num(row.get("MinAx")) or 0.0
                major, minor = major or minor, minor or major
                mag = _num(row.get("V-Mag"))
                recs.append(_Rec(name, ra, dec, major, minor, _num(row.get("PosAng")) or 0.0, KINDS.get(typ, "other"),
                                 mag if mag is not None else _num(row.get("B-Mag")),
                                 _first_name(row.get("Common names") or "")))
    return recs, links


def read_vii20(path: Path) -> list[_Rec]:
    """VizieR's tab-separated answer for VII/20 (columns Sh2, Diam, _RAJ2000, _DEJ2000 in degrees)."""
    rows = [ln for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip() and not ln.startswith("#")]
    if len(rows) < 3:
        return []
    cols = [c.strip() for c in rows[0].split("\t")]
    try:
        i_sh, i_d, i_ra, i_de = (cols.index(c) for c in ("Sh2", "Diam", "_RAJ2000", "_DEJ2000"))
    except ValueError:
        return []
    out = []
    for ln in rows[1:]:
        p = ln.split("\t")
        if len(p) < len(cols) or not p[i_sh].strip().isdigit():
            continue                            # the units and the dashes under the header
        ra, dec, diam = _num(p[i_ra]), _num(p[i_de]), _num(p[i_d]) or 0.0
        if ra is None or dec is None:
            continue
        out.append(_Rec(f"Sh2-{int(p[i_sh])}", ra, dec, diam, diam, 0.0, "HII region"))
    return out


def read_siril(folder: Path, objects: bool = True, sharpless: bool = True
               ) -> tuple[list[_Rec], list[Link], dict[str, str]]:
    """Siril's lists (``messier.csv``, ``ngc.csv``, ``ic.csv``, ``sh2.csv``: name, ra, dec, [diameter, mag], alias): the
    objects of the lists asked for (``objects``: Messier / NGC / IC; ``sharpless``: Sh2), the cross-identifications and
    the common names its aliases hold, of every list."""
    recs: list[_Rec] = []
    links: list[Link] = []
    common: dict[str, str] = {}
    for fname, keep in (("messier.csv", objects), ("ngc.csv", objects), ("ic.csv", objects), ("sh2.csv", sharpless)):
        f = Path(folder) / fname
        if not f.is_file():
            continue
        with open(f, encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                name = normalize(row.get("name") or "")
                ra, dec = _num(row.get("ra")), _num(row.get("dec"))
                if not name or ra is None or dec is None:
                    continue
                for tok in (row.get("alias") or "").split("/"):
                    if not tok.strip():
                        continue
                    d = normalize(tok)
                    if d is None:
                        common.setdefault(name, _title(tok))
                    elif d != name:
                        links.append((name, d, True))
                if keep:
                    diam = _num(row.get("diameter")) or 0.0
                    recs.append(_Rec(name, ra, dec, diam, diam, 0.0,
                                     "HII region" if name.startswith("Sh2-") else "other", _num(row.get("mag"))))
    return recs, links, common


def _sep(ra1, dec1, ra2, dec2):
    """Angular distance in degrees (haversine; arrays welcome)."""
    r1, d1, r2, d2 = (np.radians(np.asarray(a, np.float64)) for a in (ra1, dec1, ra2, dec2))
    s = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0.0, 1.0))))


def _close(a: _Rec | None, b: _Rec | None) -> bool:
    """Whether two records can be one object: their centres closer than about their size (True when one is unknown)."""
    if a is None or b is None:
        return True
    return float(_sep(a.ra, a.dec, b.ra, b.dec)) * 60.0 <= max(LINK_MIN_ARCMIN, 0.75 * max(a.major, b.major))


def merge(recs: Sequence[_Rec], links: Sequence[Link], common: dict[str, str] | None = None,
          curated: Sequence[tuple[str, str, Sequence[str]]] = CURATED) -> dict:
    """One object per physical object: the records and the designations linked to them in groups (a checked link
    between two records too far apart is dropped); the name, the position, size and angle are the group's first
    designation's (``_rank``) that has them, except for a cluster in a larger nebula of the group (IC 1805, 60', and
    Sh2-190, 150': the outline is the nebula's, the Heart Nebula); the kind from its first record that has one; the
    common name from
    ``curated``, else the records', else Siril's aliases (``common``). The catalogue as columns."""
    rec_of: dict[str, _Rec] = {}
    for r in recs:
        rec_of.setdefault(r.desig, r)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for d in rec_of:
        find(d)
    for a, b, check in links:
        if a != b and (not check or _close(rec_of.get(a), rec_of.get(b))):
            union(a, b)
    named: dict[str, str] = {}
    for d, cn, others in curated:
        if cn:
            named[d] = cn
        for o in others:
            if _close(rec_of.get(d), rec_of.get(o)):
                union(d, o)
    groups: dict[str, list[str]] = {}
    for d in list(parent):
        groups.setdefault(find(d), []).append(d)
    common = common or {}
    rows = []
    for members in groups.values():
        members.sort(key=_rank)
        rs = [rec_of[d] for d in members if d in rec_of]
        if not rs:
            continue                            # designations of objects no source has
        geo = rs[0]
        sized = next((r for r in rs if r.major > 0), geo)
        if "cluster" in (sized.kind or ""):            # a cluster and the nebula around it: the nebula's outline
            sized = max((r for r in rs if r.major > sized.major and "nebula" in (r.kind or "").lower()
                         or r.major > sized.major and r.kind == "HII region"), key=lambda r: r.major, default=sized)
        cn = (next((named[d] for d in members if named.get(d)), "")
              or next((r.common for r in rs if r.common), "")
              or next((common[d] for d in members if common.get(d)), ""))
        kind = next((r.kind for r in rs if r.kind and r.kind != "other"), "other")
        mag = next((r.mag for r in rs if r.mag is not None), None)
        rows.append((members[0], cn, kind, tuple(members[1:]), geo.ra, geo.dec, sized.major, sized.minor, sized.angle,
                     np.nan if mag is None else mag))
    rows.sort(key=lambda t: _rank(t[0]))
    return {"names": [r[0] for r in rows], "common": [r[1] for r in rows], "kinds": [r[2] for r in rows],
            "aliases": [r[3] for r in rows],
            **{k: np.array([r[i] for r in rows], np.float64)
               for i, k in enumerate(("ra", "dec", "major", "minor", "angle", "mag"), start=4)}}


# ------------------------------------------------------------------ the catalogue

@dataclass(eq=False)
class Catalogue:
    """The objects as columns (fast to load and to filter); ``catalogue[i]`` / ``objects()`` give ``SkyObject``s.
    ``source``: where it comes from (``catalogue_source``); ``notes``: plain lines on what is missing and why."""
    cols: dict
    source: str = ""
    notes: tuple[str, ...] = ()
    _index: dict[str, int] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        c = self.cols
        self.names: list[str] = list(c["names"])
        self.common: list[str] = list(c["common"])
        self.kinds: list[str] = list(c["kinds"])
        self.aliases: list[tuple[str, ...]] = [tuple(a) for a in c["aliases"]]
        for k in ("ra", "dec", "major", "minor", "angle", "mag"):
            setattr(self, k, np.asarray(c[k], np.float64).reshape(-1))

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> SkyObject:
        mag = float(self.mag[i])
        return SkyObject(self.names[i], self.common[i], self.kinds[i], float(self.ra[i]), float(self.dec[i]),
                         float(self.major[i]), float(self.minor[i]), float(self.angle[i]), self.aliases[i],
                         None if math.isnan(mag) else mag)

    def objects(self) -> list[SkyObject]:
        return [self[i] for i in range(len(self))]

    def find(self, designation: str) -> SkyObject | None:
        """The object of a designation, its name or one of its aliases, in any spelling ("NGC0224", "m31")."""
        if self._index is None:
            idx: dict[str, int] = {}
            for i, (n, al) in enumerate(zip(self.names, self.aliases)):
                for d in (n, *al):
                    idx.setdefault(d, i)
            self._index = idx
        d = normalize(designation) or designation.strip()
        i = self._index.get(d)
        return None if i is None else self[i]

    @classmethod
    def from_objects(cls, objects: Iterable[SkyObject], source: str = "") -> Catalogue:
        objs = list(objects)
        return cls({"names": [o.name for o in objs], "common": [o.common_name for o in objs],
                    "kinds": [o.kind for o in objs], "aliases": [tuple(o.aliases) for o in objs],
                    **{k: np.array([getattr(o, k) for o in objs], np.float64)
                       for k in ("ra", "dec", "major", "minor", "angle")},
                    "mag": np.array([np.nan if o.mag is None else o.mag for o in objs], np.float64)}, source)


def build_catalogue(openngc: Sequence[Path] | None = None, sharpless: Path | None = None,
                    siril: Path | None = None) -> Catalogue:
    """The merged catalogue of the sources given (no network, no cache): OpenNGC's files, VizieR's VII/20 answer,
    Siril's catalogue folder (its Messier / NGC / IC lists used only without OpenNGC, its Sharpless list only without
    VII/20, its cross-identifications always)."""
    recs: list[_Rec] = []
    links: list[Link] = []
    parts = []
    if openngc:
        r, li = read_openngc(openngc)
        recs += r
        links += li
        parts.append(f"OpenNGC ({openngc[0]})")
    if sharpless is not None:
        recs += read_vii20(sharpless)
        parts.append(f"Sharpless, VizieR VII/20 ({sharpless})")
    common: dict[str, str] = {}
    if siril is not None:
        r, li, common = read_siril(siril, objects=not openngc, sharpless=sharpless is None)
        recs += r
        links += li
        parts.append(f"Siril ({siril})")
    return Catalogue(merge(recs, links, common), " + ".join(parts))


# ------------------------------------------------------------------ finding, downloading, caching

def _cache_dir() -> Path:
    from .skycache import cache_dir
    return cache_dir() / SUB


def _cached(name: str) -> Path | None:
    """A file of the cache (or of an earlier version's cache, read only)."""
    from .skycache import listing
    return listing(SUB).get(name)


def siril_catalogue_dir() -> Path | None:
    """Siril's deep-sky lists (``share/siril/catalogue``) when Siril is installed (sirilcat.siril_share_dirs)."""
    from .sirilcat import siril_share_dirs
    for d in siril_share_dirs():
        c = d / "catalogue"
        if any((c / f).is_file() for f in ("ngc.csv", "sh2.csv", "messier.csv")):
            return c
    return None


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _download_openngc() -> Path:
    """OpenNGC's two files from GitHub into the cache; the path of NGC.csv. ``online.ServiceDown`` / ValueError."""
    from . import online

    def fetch() -> dict[str, bytes]:
        s = online.requests_session()
        out = {}
        for remote in OPENNGC_FILES:
            r = s.get(OPENNGC_URL.format(remote))
            r.raise_for_status()
            out[remote] = r.content
        return out
    data = online.call("github", fetch, DOWNLOAD_BUDGET_S)
    if not data["NGC.csv"].startswith(b"Name;Type;RA;Dec") or data["NGC.csv"].count(b"\n") < 10000:
        raise ValueError("the file GitHub sent is not OpenNGC's catalogue")
    for remote, local in OPENNGC_FILES.items():
        _write(_cache_dir() / local, data[remote])
    return _cache_dir() / OPENNGC_FILES["NGC.csv"]


def _download_vii20(say: Callable[[str], None]) -> Path:
    """VizieR's VII/20 (Strasbourg, else Harvard) into the cache. ``online.ServiceDown`` / ValueError."""
    from . import online

    def fetch(host: str) -> bytes:
        r = online.requests_session().get(VII20_QUERY.format(host=host))
        r.raise_for_status()
        return r.content
    chain = [(name, (lambda h=host: fetch(h))) for name, host in online.VIZIER_HOSTS.items()]
    data = online.first_answer(chain, "the Sharpless catalogue", say, budget=DOWNLOAD_BUDGET_S)
    path = _cache_dir() / VII20_FILE
    tmp = path.with_name(path.name + ".check")
    _write(tmp, data)
    try:
        n = len(read_vii20(tmp))
        if n < 300:
            raise ValueError(f"VizieR's Sharpless catalogue has {n} regions instead of 313")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


_LOCK = threading.RLock()
_TRIED: dict[str, float] = {}           # source -> when a download was last tried (time.time)
_MEM: dict = {"cat": None, "key": None, "complete": False, "t": 0.0}
FAILED_FILE = "downloads.json"          # source -> when its download last failed: the next processes wait too


def reset() -> None:
    """Forget the catalogue in memory and the downloads tried in this process (tests; a new session)."""
    with _LOCK:
        _TRIED.clear()
        _MEM.update(cat=None, key=None, complete=False, t=0.0)


def _failures() -> dict:
    try:
        d = json.loads((_cache_dir() / FAILED_FILE).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _may_try(name: str) -> bool:
    """Whether a download may be tried now: not tried in this process, nor failed in any, for RETRY_S."""
    now = time.time()
    t = _TRIED.get(name)
    if t is None:
        t = _failures().get(name)
    if isinstance(t, (int, float)) and 0 <= now - t < RETRY_S:
        return False
    _TRIED[name] = now
    return True


def _failed(name: str) -> None:
    d = _failures()
    d[name] = time.time()
    try:
        _write(_cache_dir() / FAILED_FILE, json.dumps(d).encode())
    except OSError:
        pass


def _why(e: BaseException) -> str:
    """A failed download in plain words ("GitHub does not answer (no connection)")."""
    from .online import ServiceDown
    if isinstance(e, ServiceDown):
        return f"GitHub {e.why}" if e.name == "github" else str(e)
    return str(e)[:160] or type(e).__name__


def _source(name: str, cached_name: str, fetch: Callable[[], Path], download: bool) -> tuple[Path | None, str]:
    """(the cached file, "") or (None, why it is missing), downloading it when it is not cached (once per RETRY_S)."""
    from . import online
    p = _cached(cached_name)
    if p is not None:
        return p, ""
    if online.offline():
        return None, "offline mode"
    if not download:
        return None, "not in the cache"
    if not _may_try(name):
        return None, f"its download failed less than {RETRY_S / 60:.0f} minutes ago"
    try:
        return fetch(), ""
    except Exception as e:  # noqa: BLE001 - any failure: the other sources
        _failed(name)
        return None, _why(e)


def _sources(say: Callable[[str], None], download: bool) -> tuple[Path | None, Path | None, Path | None, dict]:
    """(OpenNGC's NGC.csv, VII/20, Siril's folder, {source: why it is missing})."""
    ngc, why_ngc = _source("openngc", OPENNGC_FILES["NGC.csv"], _download_openngc, download)
    sh2, why_sh2 = _source("sharpless", VII20_FILE, lambda: _download_vii20(say), download)
    missing = {k: v for k, v in (("openngc", why_ngc), ("sharpless", why_sh2)) if v}
    return ngc, sh2, siril_catalogue_dir(), missing


SIRIL_FILES = ("messier.csv", "ngc.csv", "ic.csv", "sh2.csv")


def _key(files: list[tuple[str, Path | None]]) -> str:
    parts: list = [BUILD_VERSION, repr(CURATED), sorted(PREFERRED)]
    for label, p in files:
        if p is None:
            parts.append([label, None])
            continue
        targets = [p] if p.is_file() else [p / n for n in SIRIL_FILES if (p / n).is_file()]
        for f in targets:
            try:
                st = f.stat()
                parts.append([label, str(f), st.st_size, st.st_mtime_ns])
            except OSError:
                parts.append([label, str(f), None])
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:16]


def _load_pickle(path: Path) -> Catalogue | None:
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d.get("version") != BUILD_VERSION:
            return None
        return Catalogue(d["cols"], d["source"])
    except Exception:  # noqa: BLE001 - unreadable / older: made again
        return None


def _save_pickle(path: Path, cat: Catalogue) -> None:
    try:
        _write(path, pickle.dumps({"version": BUILD_VERSION, "cols": cat.cols, "source": cat.source},
                                  protocol=pickle.HIGHEST_PROTOCOL))
        for old in path.parent.glob("catalogue_*.pkl"):
            if old != path:
                old.unlink(missing_ok=True)
    except OSError:
        pass                                    # a read-only cache: made again next time


def load_catalogue(log: Callable | None = None, download: bool = True) -> Catalogue:
    """The catalogue of named objects: in memory, else the merged cache, else made from the sources (OpenNGC and
    VII/20 downloaded once when missing and ``download``; Siril's lists when installed). ``log(msg)`` gets one plain
    line per source that is missing. ``CatalogueUnavailable`` when there is no source at all."""
    from . import online
    with _LOCK:
        cat = _MEM["cat"]
        if cat is not None and (_MEM["complete"] or time.monotonic() - _MEM["t"] < RETRY_S):
            return cat
        notes: list[str] = []
        say0 = online.sayer(log)

        def say(msg: str, level: str = "warning") -> None:
            notes.append(msg)
            say0(msg, level)
        ngc, sh2, siril, missing = _sources(say, download)
        if ngc is None and sh2 is None and siril is None:
            _MEM.update(cat=None, key=None, complete=False, t=time.monotonic())
            why = missing.get("openngc", "")
            raise CatalogueUnavailable(
                "No catalogue of named objects: OpenNGC is not in the cache and could not be downloaded"
                + (f" ({why})" if why else "") + ", and Siril, whose lists would do, is not installed. "
                "Connect to the internet once (the catalogue, 4 MB, is then kept) or install Siril.")
        if ngc is None and "openngc" in missing:
            say(f"OpenNGC could not be had ({missing['openngc']}): "
                + ("the objects come from Siril's lists, without their types and with round outlines" if siril
                   else "only the Sharpless regions are named"))
        if sh2 is None and "sharpless" in missing:
            say(f"the Sharpless catalogue (VizieR VII/20) could not be had ({missing['sharpless']}): "
                + ("its regions come from Siril's list, without their sizes" if siril else "no Sharpless regions"))
        add = (ngc.with_name(OPENNGC_FILES["addendum.csv"]) if ngc is not None else None)
        if add is not None and not add.is_file():
            add = _cached(OPENNGC_FILES["addendum.csv"])
        key = _key([("openngc", ngc), ("addendum", add), ("sharpless", sh2), ("siril", siril)])
        complete = ngc is not None and sh2 is not None
        if cat is None or _MEM["key"] != key:
            path = _cache_dir() / f"catalogue_{key}.pkl"
            cat = _load_pickle(path) if path.is_file() else None
            if cat is None:
                cat = build_catalogue([f for f in (ngc, add) if f is not None] or None, sh2, siril)
                _save_pickle(path, cat)
        cat.notes = tuple(notes)
        _MEM.update(cat=cat, key=key, complete=complete, t=time.monotonic())
        return cat


def catalogue_source() -> str:
    """Where the objects' names come from, e.g. "OpenNGC (C:/.../cache/skyobjects/openngc_NGC.csv) + Sharpless,
    VizieR VII/20 (...) + Siril (C:/Program Files/Siril/share/siril/catalogue)"; "" when there is no catalogue. Loads
    the catalogue when it is not in memory yet (which may download it)."""
    try:
        return load_catalogue().source
    except CatalogueUnavailable:
        return ""


# ------------------------------------------------------------------ the objects of a field

def _outline(ra: np.ndarray, dec: np.ndarray, major: np.ndarray, minor: np.ndarray, angle: np.ndarray,
             n: int = OUTLINE_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """(RA, Dec) in degrees of ``n`` points on each object's ellipse (axes in arcminutes, position angle east of
    north), through the gnomonic projection centred on the object: arrays (objects, n)."""
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)[None, :]
    a = np.radians(major / 120.0)[:, None]
    b = np.radians(np.where(minor > 0, minor, major) / 120.0)[:, None]
    th = np.radians(angle)[:, None]
    east = a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th)
    north = a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th)
    r0, d0 = np.radians(ra)[:, None], np.radians(dec)[:, None]
    den = np.cos(d0) - north * np.sin(d0)
    out_ra = np.degrees(r0 + np.arctan2(east, den)) % 360.0
    out_dec = np.degrees(np.arctan2(np.sin(d0) + north * np.cos(d0), np.hypot(east, den)))
    return out_ra, out_dec


def _polygon_meets_rect(px: np.ndarray, py: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Whether each closed polygon (rows of ``px``, ``py``) meets the rectangle: one of its edges crosses or enters it,
    or the rectangle is inside it."""
    ax, ay = px, py
    bx, by = np.roll(px, -1, axis=1), np.roll(py, -1, axis=1)
    finite = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(bx) & np.isfinite(by)
    box = ((np.minimum(ax, bx) <= x1) & (np.maximum(ax, bx) >= x0)
           & (np.minimum(ay, by) <= y1) & (np.maximum(ay, by) >= y0))
    dx, dy = bx - ax, by - ay
    sides = [np.sign(dx * (cy - ay) - dy * (cx - ax)) for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    s = np.stack(sides)
    straddle = ~((s > 0).all(axis=0) | (s < 0).all(axis=0))
    edge = (finite & box & straddle).any(axis=1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2         # the rectangle inside the polygon: its centre is
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = finite & ((ay > cy) != (by > cy)) & (cx < (bx - ax) * (cy - ay) / (by - ay) + ax)
    inside = (cross.sum(axis=1) % 2) == 1
    return edge | inside


def objects_in_field(wcs, width: int, height: int, catalogue: Catalogue | Iterable[SkyObject] | None = None,
                     margin: float = 0.0) -> list[SkyObject]:
    """The objects whose centre or outline (the ellipse of their axes and angle) falls in a picture of ``width`` x
    ``height`` pixels solved by the astropy ``wcs`` (0-based pixels, the picture spans -0.5 .. width - 0.5), or within
    ``margin`` pixels of it; biggest first, then by name. ``catalogue``: a ``Catalogue``, ``SkyObject``s, or None for
    ``load_catalogue()`` (which may raise ``CatalogueUnavailable``)."""
    if catalogue is None:
        cat = load_catalogue()
    elif isinstance(catalogue, Catalogue):
        cat = catalogue
    else:
        cat = Catalogue.from_objects(catalogue)
    if not len(cat):
        return []
    W, H, m = float(width), float(height), float(margin)
    x0, x1, y0, y1 = -0.5 - m, W - 0.5 + m, -0.5 - m, H - 0.5 + m
    # the field: its centre and the distance to its farthest edge point
    xs = np.array([(W - 1) / 2, x0, x1, x0, x1, (W - 1) / 2, (W - 1) / 2, x0, x1])
    ys = np.array([(H - 1) / 2, y0, y0, y1, y1, y0, y1, (H - 1) / 2, (H - 1) / 2])
    fra, fdec = wcs.all_pix2world(xs, ys, 0)
    radius = float(np.nanmax(_sep(fra[0], fdec[0], fra[1:], fdec[1:])))
    d = _sep(fra[0], fdec[0], cat.ra, cat.dec)
    cand = np.nonzero((d <= radius + cat.major / 120.0 + 1e-6) & (d < 80.0))[0]
    if not len(cand):
        return []
    ra, dec, major = cat.ra[cand], cat.dec[cand], cat.major[cand]
    # centres: the plain projection, refined with the distortions near the picture (far out they may diverge)
    lx, ly = wcs.wcs_world2pix(ra, dec, 0)
    near = (lx >= x0 - 0.1 * W) & (lx <= x1 + 0.1 * W) & (ly >= y0 - 0.1 * H) & (ly <= y1 + 0.1 * H)
    cx, cy = lx.astype(np.float64), ly.astype(np.float64)
    if near.any():
        ax_, ay_ = wcs.all_world2pix(ra[near], dec[near], 0, quiet=True)
        cx[near], cy[near] = ax_, ay_
    hit = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    # outlines of the sized objects whose centre is outside (the plain projection: the distortions are a few pixels)
    todo = np.nonzero(~hit & (major > 0))[0]
    if len(todo):
        ora, odec = _outline(ra[todo], dec[todo], major[todo], cat.minor[cand][todo], cat.angle[cand][todo])
        px, py = wcs.wcs_world2pix(ora.ravel(), odec.ravel(), 0)
        hit[todo] = _polygon_meets_rect(np.asarray(px).reshape(ora.shape), np.asarray(py).reshape(ora.shape),
                                        x0, x1, y0, y1)
    keep = cand[hit]
    order = sorted(keep.tolist(), key=lambda i: (-cat.major[i], _rank(cat.names[i])))
    return [cat[i] for i in order]
