"""Siril's offline Gaia DR3 extracts, read in place (never copied, never written).

Siril users often have them: Siril's catalogue installer puts them in Siril's data folder and Siril reads them for its
plate solving and its spectrophotometric colour calibration. Read here, they make our astrometry and the XP spectra of
the catalogues stage local.

Format (Siril HEALpixel Catalog Format 1.0.0, Zenodo 14697486): a 128-byte header (title 48 bytes, Gaia release,
HEALPix level, type 1 astrometric / 2 xp_sampled, chunked flag, chunk level, then three packed uint32: the chunk's
pixel, its first and last pixel at the index level), one uint32 per pixel of the index level (nested scheme) giving
the cumulative number of records up to and including it, then the records grouped by pixel in pixel order.

- astrometric extract (``siril_cat_healpix8_astro.dat``, 1.5 GB, one file), 16 bytes a star: RA and Dec int32 x
  360/(2^31-1) deg (ICRS, epoch 2016.0), pmRA and pmDec int16 mas/yr, Teff uint16 K, G int16 in mmag. It holds the 127
  brightest stars of each level-8 pixel (14' wide): complete to G ~ 17 far from the Milky Way, to G ~ 14-16 in it.
- xp_sampled extract (``siril_cat1_healpix8_xpsamp_<N>.dat``, one file per level-1 pixel N), 701 bytes a star: the
  same first 14 bytes without Teff, an int8 exponent e and 343 float16 values: the sampled XP spectrum, 336-1020 nm
  every 2 nm, in W m-2 nm-1 = value x 10^-e (checked on Gaia archive spectra of the user's cache: equal within 0.03 %).

Where (the first found): ``ASTRO_GPU_SIRIL_CATALOGUES`` (folders separated by ``os.pathsep``), the files Siril's own
settings name (``catalogue_gaia_astro`` / ``catalogue_gaia_photo`` in its config*.ini), Siril's data folder
(Windows ``%LOCALAPPDATA%/siril``, Linux ``~/.local/share/siril``, macOS ``~/Library/Application Support/siril``).
"""
from __future__ import annotations

import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCALE = 360.0 / (2 ** 31 - 1)
HEADER = 128
ASTRO_DT = np.dtype([("ra", "<i4"), ("dec", "<i4"), ("pmra", "<i2"), ("pmdec", "<i2"), ("teff", "<u2"),
                     ("g", "<i2")])
XP_DT = np.dtype([("ra", "<i4"), ("dec", "<i4"), ("pmra", "<i2"), ("pmdec", "<i2"), ("g", "<i2"), ("e", "i1"),
                  ("flux", "<f2", (343,))])
XP_WAVELENGTHS = np.arange(336.0, 1020.0 + 1e-6, 2.0)
TYPE_ASTRO, TYPE_XPSAMP = 1, 2


@dataclass
class CatFile:
    path: Path
    level: int
    kind: int
    chunk_level: int = 0
    chunk: int = 0
    first: int = 0
    last: int = 0
    _index: np.ndarray | None = field(default=None, repr=False)
    _cap: int | None = field(default=None, repr=False)

    @property
    def dtype(self) -> np.dtype:
        return ASTRO_DT if self.kind == TYPE_ASTRO else XP_DT

    @property
    def n_index(self) -> int:
        return (self.last - self.first + 1) if self.chunk_level or self.first or self.last else 12 * 4 ** self.level

    def index(self) -> np.ndarray:
        if self._index is None:
            self._index = np.fromfile(self.path, dtype="<u4", count=self.n_index, offset=HEADER).astype(np.int64)
        return self._index

    def cap(self) -> int:
        """The most records a pixel holds (the extract keeps that many brightest stars per pixel)."""
        if self._cap is None:
            self._cap = int(np.diff(np.r_[0, self.index()]).max())
        return self._cap

    def read(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The records of the given pixels (index level, nested): (records, the pixel of each record)."""
        idx = self.index()
        pixels = np.unique(np.asarray(pixels, np.int64))
        pixels = pixels[(pixels >= self.first) & (pixels < self.first + self.n_index)]
        if not len(pixels):
            return np.zeros(0, self.dtype), np.zeros(0, np.int64)
        k = pixels - self.first
        # runs of consecutive pixels: one read each
        cut = np.where(np.diff(k) != 1)[0] + 1
        recs, pix = [], []
        data0 = HEADER + 4 * self.n_index
        with open(self.path, "rb") as f:
            for run in np.split(k, cut):
                a = int(idx[run[0] - 1]) if run[0] > 0 else 0
                b = int(idx[run[-1]])
                if b <= a:
                    continue
                f.seek(data0 + a * self.dtype.itemsize)
                r = np.frombuffer(f.read((b - a) * self.dtype.itemsize), dtype=self.dtype)
                recs.append(r)
                cnt = np.diff(np.r_[a, idx[run]])
                pix.append(np.repeat(run + self.first, cnt))
        if not recs:
            return np.zeros(0, self.dtype), np.zeros(0, np.int64)
        return np.concatenate(recs), np.concatenate(pix)


def read_header(path: Path) -> CatFile | None:
    """The catalogue described by a Siril catalogue file, or None when it is not one (or is truncated)."""
    try:
        with open(path, "rb") as f:
            h = f.read(HEADER)
        size = path.stat().st_size
    except OSError:
        return None
    if len(h) < HEADER or not h[:48].startswith(b"Siril"):
        return None
    level, kind, chunked, clevel = h[49], h[50], h[51], h[52]
    if not (1 <= level <= 12) or kind not in (TYPE_ASTRO, TYPE_XPSAMP):
        return None
    chunk, first, last = (int.from_bytes(h[o:o + 4], "little") for o in (53, 57, 61))
    cf = CatFile(Path(path), level, kind, clevel if chunked else 0, chunk if chunked else 0,
                 first if chunked else 0, last if chunked else 0)
    if cf.chunk_level == 0:
        cf.first = cf.last = 0
    n = cf.n_index
    if size < HEADER + 4 * n:
        return None
    try:
        total = int(np.fromfile(path, dtype="<u4", count=1, offset=HEADER + 4 * (n - 1))[0])
    except (OSError, IndexError):
        return None
    if size != HEADER + 4 * n + total * cf.dtype.itemsize:      # incomplete download / other record layout
        return None
    return cf


# ------------------------------------------------------------------ where the files are

def _siril_config_files() -> list[Path]:
    homes = []
    if sys.platform == "win32":
        if os.environ.get("LOCALAPPDATA"):
            homes.append(Path(os.environ["LOCALAPPDATA"]) / "siril")
        if os.environ.get("APPDATA"):
            homes.append(Path(os.environ["APPDATA"]) / "siril")
    homes += [Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "siril",
              Path.home() / "Library" / "Application Support" / "siril"]
    out = []
    for h in homes:
        if h.is_dir():
            # newest settings first: config.1.4.ini before config.ini
            out += sorted(h.glob("config*.ini"), key=lambda p: (p.name != "config.ini", p.name), reverse=True)
    return out


def _from_config() -> list[Path]:
    paths: list[Path] = []
    for cfg in _siril_config_files():
        try:
            txt = cfg.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for key in ("catalogue_gaia_astro", "catalogue_gaia_photo"):
            m = re.search(rf"(?m)^{key}=(.*)$", txt)
            if m and m.group(1).strip():
                v = m.group(1).strip().strip('"').replace("\\\\", "\\")
                paths.append(Path(os.path.expanduser(v)))
    return paths


def _data_dirs() -> list[Path]:
    out = []
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        out.append(Path(os.environ["LOCALAPPDATA"]) / "siril")
    out += [Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "siril",
            Path.home() / "Library" / "Application Support" / "siril"]
    return out


def _candidates() -> list[Path]:
    env = [Path(p) for p in os.environ.get("ASTRO_GPU_SIRIL_CATALOGUES", "").split(os.pathsep) if p.strip()]
    return env + _from_config() + _data_dirs()


def _scan(p: Path, found: dict) -> None:
    """A file, or a folder holding Siril catalogue files (or their xpsamp folder)."""
    if p.is_file():
        cf = read_header(p)
        if cf is None:
            return
        if cf.kind == TYPE_ASTRO and found["astro"] is None:
            found["astro"] = cf
        elif cf.kind == TYPE_XPSAMP:
            found["xp"].setdefault(cf.chunk if cf.chunk_level else -1, cf)
        return
    if not p.is_dir():
        return
    for q in sorted(p.glob("siril_cat*_healpix*_astro*.dat")):
        _scan(q, found)
    for q in sorted(p.glob("siril_cat*_healpix*_xpsamp*.dat")):
        _scan(q, found)
    for d in sorted(p.glob("siril_cat*_healpix*_xpsamp")):
        if d.is_dir():
            for q in sorted(d.glob("siril_cat*_healpix*_xpsamp*.dat")):
                _scan(q, found)


_FOUND: dict | None = None
_FOUND_LOCK = threading.Lock()


def locate() -> dict:
    """{"astro": CatFile | None, "xp": {level-1 chunk: CatFile}} (looked up once per process; ``reset`` forgets)."""
    global _FOUND
    with _FOUND_LOCK:
        if _FOUND is None:
            found: dict = {"astro": None, "xp": {}}
            for c in _candidates():
                try:
                    _scan(c, found)
                except OSError:
                    continue
            _FOUND = found
        return _FOUND


def reset() -> None:
    global _FOUND
    with _FOUND_LOCK:
        _FOUND = None


def describe() -> str:
    f = locate()
    parts = []
    if f["astro"] is not None:
        parts.append(f"astrometric extract {f['astro'].path}")
    if f["xp"]:
        parts.append(f"{len(f['xp'])} chunks of the spectrophotometric extract in {next(iter(f['xp'].values())).path.parent}")
    return "; ".join(parts) or "none"


def siril_share_dirs() -> list[Path]:
    """Siril's installed data folders (``share/siril``: its deep-sky lists in ``catalogue/``), the first existing
    first: ASTRO_GPU_SIRIL_DIR, next to a ``siril`` program on the PATH, the usual install places."""
    import shutil
    c: list[Path] = []
    if os.environ.get("ASTRO_GPU_SIRIL_DIR"):
        c.append(Path(os.environ["ASTRO_GPU_SIRIL_DIR"]))
    for exe in ("siril", "siril-cli"):
        w = shutil.which(exe)
        if w:
            c.append(Path(w).resolve().parent.parent / "share" / "siril")
    for env in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        if os.environ.get(env):
            c.append(Path(os.environ[env]) / "Siril" / "share" / "siril")
    if os.environ.get("LOCALAPPDATA"):
        c.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Siril" / "share" / "siril")
    c += [Path("/usr/share/siril"), Path("/usr/local/share/siril"), Path("/app/share/siril"),
          Path("/var/lib/flatpak/app/org.siril.Siril/current/active/files/share/siril"),
          Path.home() / ".local/share/flatpak/app/org.siril.Siril/current/active/files/share/siril",
          Path("/Applications/Siril.app/Contents/Resources/share/siril")]
    out = []
    for p in c:
        try:
            if (p / "catalogue").is_dir() and p not in out:
                out.append(p)
        except OSError:
            continue
    return out


# ------------------------------------------------------------------ queries

def _sep_deg(ra1, dec1, ra2, dec2):
    r = np.radians
    v = np.sin(r(dec1)) * np.sin(r(dec2)) + np.cos(r(dec1)) * np.cos(r(dec2)) * np.cos(r(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(v, -1.0, 1.0)))


def astro_cone(ra: float, dec: float, radius: float, gmax: float) -> dict | None:
    """Stars of the astrometric extract within ``radius`` deg of (ra, dec) with G < gmax: {"RA_ICRS", "DE_ICRS" (epoch
    2016.0), "Gmag", "pmRA", "pmDE", "complete_to": the G down to which every Gaia star is there (inf: all)}; None when
    the extract is not installed."""
    from .skycache import tiles_in_cone
    cf = locate()["astro"]
    if cf is None:
        return None
    pixels = tiles_in_cone(cf.level, ra, dec, radius)
    rec, pix = cf.read(pixels)
    r_ra = rec["ra"].astype(np.float64) * SCALE
    r_dec = rec["dec"].astype(np.float64) * SCALE
    g = rec["g"].astype(np.float64) / 1000.0
    # where a pixel is full, it holds only its brightest stars: complete down to its faintest one
    complete_to = np.inf
    if len(rec):
        up, inv, cnt = np.unique(pix, return_inverse=True, return_counts=True)
        full = cnt >= cf.cap()
        if full.any():
            gmaxpix = np.full(len(up), -np.inf)
            np.maximum.at(gmaxpix, inv, g)
            complete_to = float(gmaxpix[full].min())
    keep = (g < gmax) & (_sep_deg(ra, dec, r_ra, r_dec) <= radius)
    return {"RA_ICRS": r_ra[keep], "DE_ICRS": r_dec[keep], "Gmag": g[keep],
            "pmRA": rec["pmra"][keep].astype(np.float64), "pmDE": rec["pmdec"][keep].astype(np.float64),
            "complete_to": complete_to}


def xp_spectra(ra: np.ndarray, dec: np.ndarray, g: np.ndarray, sources: np.ndarray | None = None,
               tol_arcsec: float = 0.05, tol_mag: float = 0.01) -> list:
    """Sampled XP spectra (wavelength nm, flux W m-2 nm-1) of Gaia stars given by their epoch-2016.0 position and G,
    from the installed chunks of Siril's spectrophotometric extract; None for a star not found there."""
    out: list = [None] * len(ra)
    chunks = locate()["xp"]
    if not chunks or not len(ra):
        return out
    import astropy.units as u
    from astropy_healpix import HEALPix
    any_cf = next(iter(chunks.values()))
    level = any_cf.level
    hp = HEALPix(nside=2 ** level, order="nested")
    ra, dec, g = (np.asarray(a, np.float64) for a in (ra, dec, g))
    pix = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg).astype(np.int64)
    pix2 = pix.copy()
    if sources is not None:                     # the pixel coded in the Gaia source id (level 12), when it differs
        s = np.asarray(sources, np.int64)
        ok = s > 0
        pix2[ok] = (s[ok] // 34359738368) >> (2 * (12 - level))
    tol = tol_arcsec / 3600.0
    for which in (pix, pix2):
        todo = np.array([i for i in range(len(ra)) if out[i] is None])
        if not len(todo):
            break
        p = which[todo]
        for cf in chunks.values():
            if cf.chunk_level:
                mine = (p >> (2 * (level - cf.chunk_level))) == cf.chunk
            else:
                mine = np.ones(len(p), bool)
            if not mine.any():
                continue
            rec, rpix = cf.read(np.unique(p[mine]))
            if not len(rec):
                continue
            r_ra = rec["ra"].astype(np.float64) * SCALE
            r_dec = rec["dec"].astype(np.float64) * SCALE
            r_g = rec["g"].astype(np.float64) / 1000.0
            order = np.argsort(rpix, kind="stable")
            rp_sorted = rpix[order]
            for i in todo[mine]:
                a, b = np.searchsorted(rp_sorted, [which[i], which[i] + 1])
                if b <= a:
                    continue
                j = order[a:b]
                d = _sep_deg(ra[i], dec[i], r_ra[j], r_dec[j])
                k = int(np.argmin(d))
                if d[k] <= tol and abs(r_g[j[k]] - g[i]) <= tol_mag:
                    r = rec[j[k]]
                    flux = r["flux"].astype(np.float64) * 10.0 ** (-float(r["e"]))
                    out[i] = (XP_WAVELENGTHS.copy(), flux)
    return out
