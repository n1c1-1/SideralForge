"""The offline Gaia DR3 extracts: where they come from, whether they are installed, and the download.

Never automatic: the download runs only when the user asks for it (Settings › Catalogs & astrometry). The files are
read by ``core/sirilcat.py`` from our own folder, ``gaia_extracts`` of the cache (``sirilcat.own_dir()``):

    gaia_extracts/siril_cat_healpix8_astro.dat                                  star positions (1.5 GB)
    gaia_extracts/siril_cat1_healpix8_xpsamp/siril_cat1_healpix8_xpsamp_<N>.dat  star spectra, N = 0..47

Source: two Zenodo records (CC BY 4.0), HEALPix extracts of Gaia DR3 made by the Siril project. The file lists (names,
sizes, md5, links) are read from the Zenodo API when the download starts. For each file:

1. ``<name>.bz2.part`` is filled by a streamed GET with ``Range: bytes=<have>-`` (the answer 206 is appended; 200 starts
   again from 0): a cancel, a crash or a restart of the program loses nothing, the next run goes on from there;
2. its md5 is checked (a mismatch deletes it and says so) and it is renamed ``<name>.bz2``;
3. it is unpacked, streaming, into ``<name>.part`` which is checked as an extract and renamed to its final name, then
   the ``.bz2`` is deleted. A file already there, complete, is skipped.

``download(kind, progress, cancel)`` raises ``Cancelled`` or ``DownloadError`` (plain English) and always makes the
engine look again for the files (``sirilcat.reset``), so that what is complete is used at once. The texts it produces
(progress, errors) are English templates passed through ``translate`` and their numbers through ``number``: the
client sets both to its language.
"""
from __future__ import annotations

import bz2
import contextlib
import hashlib
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import online, sirilcat

ZENODO_API = "https://zenodo.org/api/records/{record}"
LICENCE = "CC BY 4.0"
ASTRO = "astro"                                                 # star positions
XP = "xp"                                                       # star spectra
RECORDS = {ASTRO: 14692304, XP: 14738271}
VERSIONS = {ASTRO: "1.0.0", XP: "1.0.1"}
TITLES = {ASTRO: "Siril Astrometry Catalogue extracted from Gaia DR3",
          XP: "Siril Spectrophotometric Catalog extracted from Gaia DR3"}
NAME_RE = {ASTRO: re.compile(r"^siril_cat_healpix8_astro\.dat\.bz2$"),
           XP: re.compile(r"^siril_cat1_healpix8_xpsamp_(\d+)\.dat\.bz2$")}
SUBDIR = {ASTRO: "", XP: "siril_cat1_healpix8_xpsamp"}
FILES = {ASTRO: 1, XP: 48}                                      # files a complete extract has
DOWNLOAD_BYTES = {ASTRO: 1_142_276_749, XP: 10_550_000_000}     # compressed (XP: 10.55 GB in all)
#: what a file takes once unpacked, compared with the compressed file (the astrometric one: 1.5 GB for 1.14 GB)
UNPACKED_RATIO = 1.5
CHUNK = 1 << 20
REPORT_EVERY_S = 0.2                                            # progress is handed on five times a second at most
MAX_RETRIES = 6                                                 # connection failures in a row without any progress


class DownloadError(Exception):
    """The download stopped; the message says why in plain words."""


class Cancelled(Exception):
    """The user cancelled; what was downloaded is kept for the next time."""


@dataclass
class Remote:
    name: str
    size: int
    md5: str | None
    url: str


@dataclass
class Status:
    kind: str
    state: str                  # "installed" | "partial" | "missing"
    files: int                  # complete files the engine finds
    total: int                  # files a complete extract has
    part_bytes: int = 0         # bytes of unfinished downloads in our folder
    outside: Path | None = None  # folder of a copy found elsewhere (not ours), when the engine uses one
    own: bool = False           # every file the engine finds is in our folder


# ------------------------------------------------------------------ texts

def translate(s: str) -> str:
    """The language of the texts (a hook: the client sets it to its translation function)."""
    return s


def number(x: float, digits: int = 1) -> str:
    """A number as text (a hook: the client sets it to write a decimal comma in French)."""
    return f"{x:.{digits}f}"


def _t(template: str, **fields) -> str:
    return translate(template).format(**fields)


def human(n: float) -> str:
    return f"{number(n / 1e9)} {translate('GB')}" if n >= 1e9 else f"{number(n / 1e6, 0)} {translate('MB')}"


# ------------------------------------------------------------------ what is installed

def folder() -> Path:
    return sirilcat.own_dir()


def target_dir(kind: str) -> Path:
    return folder() / SUBDIR[kind] if SUBDIR[kind] else folder()


def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def status(kind: str) -> Status:
    """What the engine finds for this extract (``sirilcat.locate``), and what an unfinished download left."""
    found = sirilcat.locate()
    if kind == ASTRO:
        paths = [found["astro"].path] if found["astro"] is not None else []
    else:
        paths = [cf.path for cf in found["xp"].values()]
    mine = [p for p in paths if _inside(p, folder())]
    outside = next((p.parent for p in paths if p not in mine), None)
    part = 0
    d = target_dir(kind)
    if d.is_dir():
        for e in d.iterdir():
            if e.name.endswith((".part", ".bz2")) and e.is_file() and (kind == XP) == ("xpsamp" in e.name):
                with contextlib.suppress(OSError):
                    part += e.stat().st_size
    total = FILES[kind]
    state = "installed" if len(paths) >= total else "partial" if paths or part else "missing"
    return Status(kind, state, len(paths), total, part, outside, len(mine) == len(paths))


# ------------------------------------------------------------------ the file lists

def _plain(e: BaseException) -> str:
    """The failure of a request in a few plain words."""
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code:
        return _t("the server answered HTTP {code}", code=code)
    return online._plain(e)


def remote_files(kind: str, session=None, api: str | None = None) -> list[Remote]:
    """The files of the extract's Zenodo record (names, sizes, md5, download links), in the order N = 0, 1, ..."""
    session = session or online.requests_session()
    url = (api or ZENODO_API).format(record=RECORDS[kind])
    try:
        r = session.get(url)
        r.raise_for_status()
        listing = r.json()["files"]
    except Exception as e:  # noqa: BLE001 - any failure of the list is one plain message
        raise DownloadError(_t("Cannot read the file list of the data set on Zenodo ({why}). "
                               "Check the internet connection and try again.", why=_plain(e))) from e
    out = []
    for f in listing:
        m = NAME_RE[kind].match(str(f.get("key", "")))
        link = (f.get("links") or {}).get("self")
        if not m or not link:
            continue
        md5 = str(f.get("checksum") or "")
        out.append(Remote(f["key"], int(f["size"]), md5.split(":", 1)[1].lower() if md5.startswith("md5:") else None,
                          link))
    if not out:
        raise DownloadError(_t("The Zenodo record has no file of this extract any more; the program has to be updated."))
    out.sort(key=lambda x: int((NAME_RE[kind].match(x.name).groups() or ("0",))[0]))
    return out


# ------------------------------------------------------------------ the download

_RUN = threading.Lock()


class _Report:
    """Progress of the whole download: units of work done (download + unpacking, both counted in compressed bytes) of
    the total, and what is being done, handed on at most five times a second."""

    def __init__(self, cb: Callable | None, total: int):
        self.cb, self.total, self.base, self.last = cb, total, 0, 0.0

    def __call__(self, within: int, text: str, force: bool = False):
        if self.cb is None:
            return
        now = time.monotonic()
        if force or now - self.last >= REPORT_EVERY_S:
            self.last = now
            self.cb(min(self.base + within, self.total), self.total, text)


def _check(cancel: threading.Event | None):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _md5(path: Path, cancel) -> str:
    h = hashlib.md5()  # noqa: S324 - an integrity check against Zenodo's listed md5, not security
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            _check(cancel)
            h.update(chunk)
    return h.hexdigest()


def _fetch(rf: Remote, part: Path, session, cancel, report: _Report, label: str) -> None:
    """Fill ``part`` with the file (going on from what it holds) until it has ``rf.size`` bytes."""
    import requests
    fails = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have > rf.size:
            part.unlink()
            have = 0
        if have == rf.size:
            return
        _check(cancel)
        headers = {"Accept-Encoding": "identity"}        # the bytes of the file, as they are (a range counts them)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with session.get(rf.url, headers=headers, stream=True) as r:
                if r.status_code == 416:                 # the part is not a prefix of the file: start again
                    part.unlink(missing_ok=True)
                    continue
                if r.status_code not in (200, 206):
                    raise DownloadError(_t("Zenodo refused the download of {name} (HTTP {code}). Try again later.",
                                           name=rf.name, code=r.status_code))
                if r.status_code == 206:
                    m = re.match(r"bytes (\d+)-", r.headers.get("Content-Range", ""))
                    if not m or int(m.group(1)) != have:     # not the part we asked for: start again
                        part.unlink(missing_ok=True)
                        continue
                    mode = "ab"
                else:                                    # the server ignores ranges (or it is the first request)
                    mode, have = "wb", 0
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        _check(cancel)
                        f.write(chunk)
                        have += len(chunk)
                        fails = 0
                        report(have, label + _t(": downloading {done} of {size}", done=human(have),
                                                size=human(rf.size)))
                        if have > rf.size:
                            break
        except requests.RequestException as e:
            fails += 1
            if fails > MAX_RETRIES:
                raise DownloadError(_t("The connection to Zenodo failed while downloading {name} ({why}). What was "
                                       "downloaded is kept: start again to go on.", name=rf.name, why=_plain(e))) from e
            for _i in range(2 * fails):                  # a short pause, cancellable
                _check(cancel)
                time.sleep(0.5)
            continue
        if (part.stat().st_size if part.exists() else 0) < rf.size:      # the server closed early: go on from there
            fails += 1
            if fails > MAX_RETRIES:
                raise DownloadError(_t("The download of {name} stops early. What was downloaded is kept: start again "
                                       "to go on.", name=rf.name))


def _unpack(src: Path, dst_part: Path, cancel, report: _Report, label: str) -> None:
    """bz2 (possibly several streams) to ``dst_part``, streaming."""
    size = src.stat().st_size
    done = 0
    dec = bz2.BZ2Decompressor()
    open_stream = False
    with open(src, "rb") as f, open(dst_part, "wb") as out:
        while chunk := f.read(CHUNK):
            _check(cancel)
            done += len(chunk)
            while chunk:
                try:
                    out.write(dec.decompress(chunk))
                except (OSError, EOFError, ValueError) as e:
                    raise DownloadError(_t("{name} is damaged and cannot be unpacked. It was deleted; start again.",
                                           name=src.name)) from e
                open_stream = True
                if dec.eof:
                    chunk = dec.unused_data
                    dec, open_stream = bz2.BZ2Decompressor(), False
                else:
                    chunk = b""
            report(size + done, label + _t(": unpacking {done} of {size}", done=human(done), size=human(size)))
        if open_stream:
            raise DownloadError(_t("{name} is incomplete and cannot be unpacked. It was deleted; start again.",
                                   name=src.name))
        out.flush()
        os.fsync(out.fileno())


def _one(kind: str, rf: Remote, session, cancel, report: _Report, label: str) -> None:
    d = target_dir(kind)
    final = d / rf.name[:-4]
    bz, part = d / rf.name, d / (rf.name + ".part")
    if final.exists() and sirilcat.read_header(final) is not None:
        report.base += 2 * rf.size                       # already there
        return
    d.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(d).free
    need = (0 if bz.exists() else rf.size - (part.stat().st_size if part.exists() else 0)) + UNPACKED_RATIO * rf.size
    if free < need:
        raise DownloadError(_t("Not enough free disk space in {folder}: {need} needed for {name}, {free} free.",
                               folder=d, need=human(need), name=rf.name, free=human(free)))
    if not bz.exists():
        _fetch(rf, part, session, cancel, report, label)
        report(rf.size, label + _t(": checking"), force=True)
        if rf.md5 is not None and _md5(part, cancel) != rf.md5:
            part.unlink(missing_ok=True)
            raise DownloadError(_t("{name} was damaged during the download (its checksum does not match Zenodo's). "
                                   "The file was deleted; start again.", name=rf.name))
        os.replace(part, bz)                             # from here the .bz2 is a verified file
    dst_part = d / (final.name + ".part")
    try:
        _unpack(bz, dst_part, cancel, report, label)
        if sirilcat.read_header(dst_part) is None:
            raise DownloadError(_t("{name} unpacked into something that is not an extract. It was deleted; "
                                   "start again.", name=rf.name))
    except DownloadError:
        dst_part.unlink(missing_ok=True)
        bz.unlink(missing_ok=True)
        raise
    except BaseException:
        dst_part.unlink(missing_ok=True)                 # a cancel: the .bz2 stays, only the unpacking restarts
        raise
    os.replace(dst_part, final)
    bz.unlink(missing_ok=True)
    report.base += 2 * rf.size


def download(kind: str, progress: Callable[[int, int, str], None] | None = None,
             cancel: threading.Event | None = None, session=None, api: str | None = None) -> None:
    """Download and unpack the extract ``kind`` (``ASTRO`` or ``XP``) into our folder, going on from what a stopped
    download left. ``progress(done, total, text)``: units of work (download plus unpacking, in compressed bytes) and
    the plain text of the phase. Raises ``Cancelled`` or ``DownloadError``."""
    if kind not in RECORDS:
        raise ValueError(kind)
    if online.offline():
        raise DownloadError(_t("The program is in offline mode (ASTRO_GPU_OFFLINE): nothing is downloaded."))
    if not _RUN.acquire(blocking=False):
        raise DownloadError(_t("A download is already running."))
    try:
        session = session or online.requests_session()
        if progress is not None:
            progress(0, 1, _t("Reading the file list from Zenodo"))
        files = remote_files(kind, session, api)
        report = _Report(progress, 2 * sum(f.size for f in files))
        n = len(files)
        for i, rf in enumerate(files, 1):
            _check(cancel)
            label = _t("File {i} of {n}", i=i, n=n) if n > 1 else _t("Star positions")
            _one(kind, rf, session, cancel, report, label)
        report(0, _t("Done"), force=True)
    except OSError as e:
        raise DownloadError(_t("Cannot write in {folder}: {why}", folder=folder(), why=e.strerror or e)) from e
    finally:
        _RUN.release()
        sirilcat.reset()
