"""The bright stars of the whole sky: the Hipparcos list and the blind solver's two star indexes, whether they are
installed, and their installation.

Never automatic: the installation runs only when the user asks for it (Settings › Catalogs & astrometry); a run reads
what is in the cache (``core/skycache.py``) and fetches nothing (``core/online.py``). Three files of the cache:

    hipparcos/hip_main_v9.npz   Hipparcos stars V < 9 (22 000, 1 MB): the Catalogues step adds those missing from Gaia
                                (its detector saturates on the brightest), stages/catalogs.py
    quads_hip_k8.npz            the blind solver's index of wide fields (Hipparcos V < 9.5, 110 MB)
    quads_tyc_k7.npz            the blind solver's index of fields of 1-6 deg (Tycho-2 VT < 11, 620 MB)

No service gives them as files: the lists are asked of VizieR (CDS), its Harvard mirror, else the ESA archive
(``gaia_archive.hipparcos_all``, ``blindsolve.star_list``: a few MB, about 20 MB for the 550 000 Tycho-2 stars, short
timeouts and one line per fallback), and the indexes are built here from them (a few minutes each). Resumable step by
step: a file already there is skipped, and a star list downloaded for an index is kept beside it until its index is
written (``stars_<catalog>.npz``), so that a cancel or a crash during a build does not download it again. A query or a
build in progress cannot be cut short: a cancel is waited for at the end of the query, at the next neighbour
combination of a build (a second at most).

``download(progress, cancel)`` has the form of ``gaia_extracts.download`` (its ``Cancelled``, ``DownloadError`` and
texts, translated through its hooks), so that the one download job of the client (``client/services/extracts.py``)
runs either.
"""
from __future__ import annotations

import contextlib
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import blindsolve, online, skycache
from .gaia_extracts import Cancelled, DownloadError, _plain, _t, human

KIND = "bright"
HIP_VMAX = 9.0                                                  # the Catalogues step's supplement (stages/catalogs.py)
HIP_COLS = ["HIP", "RAICRS", "DEICRS", "Vmag", "B-V", "pmRA", "pmDE"]
FILES = [skycache.HIP_FILE.format(vmax=HIP_VMAX), blindsolve.INDEX_FILES["hip"], blindsolve.INDEX_FILES["tyc"]]
#: what the three files take on the disk once built, and what is downloaded to build them (measured on a complete cache)
DISK_BYTES = 735_000_000
DOWNLOAD_BYTES = 25_000_000
#: the share of the whole installation each step takes (the Tycho-2 list and its index are most of the time)
WEIGHTS = {"hip_list": 1, "hip_stars": 1, "hip_index": 2, "tyc_stars": 4, "tyc_index": 6}


@dataclass
class Status:
    state: str                  # "installed" | "partial" | "missing"
    files: int                  # of the three, found in the cache (or an earlier version's)
    total: int
    size: int                   # bytes of the files found


def status() -> Status:
    found = [p for p in (skycache.find(f) for f in FILES) if p is not None]
    size = 0
    for p in found:
        with contextlib.suppress(OSError):
            size += p.stat().st_size
    state = "installed" if len(found) == len(FILES) else "partial" if found else "missing"
    return Status(state, len(found), len(FILES), size)


def index_here() -> bool:
    """Whether the blind solver can run: its first index (wide fields) is on the disk, the one it asks first."""
    return skycache.find(blindsolve.INDEX_FILES["hip"]) is not None


# ------------------------------------------------------------------ the installation

_RUN = threading.Lock()


def _check(cancel: threading.Event | None):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _waiting(fn: Callable, cancel: threading.Event | None):
    """``fn()`` in a thread of its own, so that a cancel is answered when it comes: the query is left to end by itself
    (its answer dropped), the files written so far stay."""
    box: dict = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - handed to the waiting thread
            box["error"] = e
    t = threading.Thread(target=run, daemon=True, name="bright-stars-query")
    t.start()
    while t.is_alive():
        t.join(0.1)
        _check(cancel)
    if "error" in box:
        raise box["error"]
    return box["value"]


def download(progress: Callable[[int, int, str], None] | None = None, cancel: threading.Event | None = None) -> None:
    """Download the Hipparcos and Tycho-2 lists and build the files that are not in the cache yet. ``progress(done,
    total, text)``: the steps' weights (``WEIGHTS``) and the plain text of the step. Raises ``Cancelled`` or
    ``DownloadError``."""
    if online.offline():
        raise DownloadError(_t("The program is in offline mode: nothing is downloaded."))
    if not _RUN.acquire(blocking=False):
        raise DownloadError(_t("A download is already running."))
    total = sum(WEIGHTS.values())
    done = [0]

    def say(step: str, text: str, within: float = 0.0):
        if progress is not None:
            progress(int(done[0] + within * WEIGHTS[step]), total, text)

    def finished(step: str):
        done[0] += WEIGHTS[step]

    try:
        cache = skycache.cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(cache).free < 1.2 * DISK_BYTES:
            raise DownloadError(_t("Not enough free disk space in {folder}: {need} needed for {name}, {free} free.",
                                   folder=cache, need=human(1.2 * DISK_BYTES), name=_t("Bright stars"),
                                   free=human(shutil.disk_usage(cache).free)))
        # 1. the Hipparcos list of the Catalogues step, as skycache.hipparcos would keep it
        if skycache.find(FILES[0]) is None:
            _check(cancel)
            say("hip_list", _t("Hipparcos stars: downloading"))
            _query(lambda: skycache.hipparcos(HIP_COLS, HIP_VMAX, fetch=True), cancel)
        finished("hip_list")
        # 2. the two indexes of the blind solver, each from its list (kept until its index is written)
        for cat, name in (("hip", "Hipparcos"), ("tyc", "Tycho-2")):
            if skycache.find(blindsolve.INDEX_FILES[cat]) is not None:
                finished(f"{cat}_stars")
                finished(f"{cat}_index")
                continue
            keep = cache / f"stars_{cat}.npz"
            if keep.exists():
                with np.load(keep) as z:
                    stars = (z["ra"], z["dec"])
            else:
                _check(cancel)
                say(f"{cat}_stars", _t("{name} stars: downloading", name=name))
                stars = _query(lambda c=cat: blindsolve.star_list(c), cancel)
                skycache.save_npz(keep, ra=stars[0], dec=stars[1])
            finished(f"{cat}_stars")
            _check(cancel)

            def step(i, n, c=cat, label=name):
                _check(cancel)
                say(f"{c}_index", _t("{name} index: building, {done} of {n}", name=label, done=i, n=n), i / n)
            say(f"{cat}_index", _t("{name} index: building", name=name))
            blindsolve.SkyIndex(cat, stars=stars, step=step)
            keep.unlink(missing_ok=True)                # in the index now
            finished(f"{cat}_index")
        if progress is not None:
            progress(total, total, _t("Done"))
    except OSError as e:
        raise DownloadError(_t("Cannot write in {folder}: {why}", folder=skycache.cache_dir(),
                               why=e.strerror or e)) from e
    finally:
        _RUN.release()


def _query(fn: Callable, cancel: threading.Event | None):
    """A list asked of the services, its failure in plain words."""
    try:
        return _waiting(fn, cancel)
    except (Cancelled, DownloadError):
        raise
    except Exception as e:  # noqa: BLE001 - any failure of the services is one plain message
        raise DownloadError(_t("No service gave the star list ({why}). Check the internet connection and try again: "
                               "what was installed is kept.", why=_plain(e))) from e
