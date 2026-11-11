"""What a run needs of the sky and finds only on the disk, said before it starts.

A run downloads nothing (core/online.py): the steps that read data of the sky read the offline Gaia extracts
(core/sirilcat.py) or what earlier versions downloaded into the cache (core/skycache.py). The planner asks ``missing``
for each step it would run; a step whose data is not on this computer is left out of the run with its reason, with the
steps that need its result, and every other step and workflow still runs (``runner2.Planner``, ``Plan.left_out``); the
graph marks the block that needs the data, its reason on hover, and its panel opens the place to get it.

    step         what it reads                                     not on the disk
    astrometry   the stars of the field: the offline star          not runnable (no fallback: the solution IS the
                 extract, or the cached sky tiles of the field     match on the catalogue)
    catalogs     the same stars, and their sampled spectra: the    not runnable, nor the Color block that reads it
                 offline spectra extract (its chunk of the field),  (it has no fallback; Color is the block named)
                 or, without the star extract, the cached spectra
    surveys      the H-alpha survey of the field (its tiles)        runnable: 'none' and the reason, its documented
                                                                    way; the Gradient block then samples on the image
    (catalogs)   the Hipparcos stars (bright stars missing          runnable: the supplement is left out with a
                 from Gaia), from the cache                         warning, its documented way

Cheap: the extracts are found once per process (``sirilcat.locate``: a few headers, never the 21 GB); the sky tiles
of a field are a few file tests, kept for a minute.
"""
from __future__ import annotations

import threading
import time

#: the reasons, in English (the client translates them); ACTION is where the user gets the data
NO_STARS = "the star catalogue is not installed on this computer"
NO_SPECTRA = "the star spectra of the catalogue are not installed on this computer"
ACTION = "Settings › Catalogs & astrometry"
#: the block that needs a step's data (the Catalogues step sits in the Astrometry block; Color is what needs it)
NEEDED_BY = {"astrometry": "astrometry", "catalogs": "color"}
#: the stages whose data is checked
STEPS = frozenset(NEEDED_BY)

_MEMO: dict = {}
_LOCK = threading.Lock()
MEMO_S = 60.0


def reset() -> None:
    """Forget what was found (after an install, in tests)."""
    with _LOCK:
        _MEMO.clear()


def _memo(key, fn):
    now = time.monotonic()
    with _LOCK:
        hit = _MEMO.get(key)
    if hit is not None and now - hit[0] < MEMO_S:
        return hit[1]
    v = fn()
    with _LOCK:
        _MEMO[key] = (now, v)
    return v


def _tiles_hold(ra: float, dec: float) -> bool:
    """Whether the cache holds a sky tile of the field's centre in a layer of the astrometry's columns (an earlier
    version's downloads; a field near it was run before)."""
    from . import skycache

    def look() -> bool:
        need = {"RA_ICRS", "DE_ICRS", "Gmag"}
        for lay in skycache.layers():
            if need <= set(lay.cols) and lay.hi >= 10.5 and lay.lo is None:
                t = int(skycache.tile_of(lay.level, [ra], [dec])[0])
                if lay.path(t).exists():
                    return True
        return False
    return _memo(("tiles", str(skycache.cache_dir()), round(ra, 1), round(dec, 1)), look)


def stars_here(centre: tuple[float, float] | None) -> bool:
    """Whether the stars of the field are on this computer: the offline star extract, or the cached tiles of the
    field's centre (when it is known)."""
    from . import sirilcat
    if sirilcat.locate()["astro"] is not None:
        return True
    return centre is not None and _tiles_hold(*centre)


def spectra_here(centre: tuple[float, float] | None) -> bool:
    """Whether the stars' spectra are on this computer: the spectra extract (its chunk of the field when the centre is
    known), or, when the stars come from the tiles (ids of Gaia: the cache keeps spectra by id), cached spectra."""
    from . import sirilcat, skycache
    found = sirilcat.locate()
    if found["xp"]:
        if centre is None:
            return True
        chunk = _chunk_of(*centre)
        return any(not cf.chunk_level or cf.chunk == chunk for cf in found["xp"].values())
    if found["astro"] is not None:              # its stars have no Gaia id: no cached spectrum can be theirs
        return False
    return _memo(("xp", str(skycache.cache_dir())), lambda: bool(skycache.listing("xp")))


def _chunk_of(ra: float, dec: float) -> int:
    import astropy.units as u
    from astropy_healpix import HEALPix
    return int(HEALPix(nside=2, order="nested").lonlat_to_healpix(ra * u.deg, dec * u.deg))


def missing(stage: str, centre: tuple[float, float] | None = None) -> str:
    """Why the step cannot run on this computer (English; the client translates it), "" when its data is here or it
    needs none. ``centre``: (ra, dec) of the field when known (the target, the solved field)."""
    if stage == "astrometry":
        return "" if stars_here(centre) else NO_STARS
    if stage == "catalogs":
        if not stars_here(centre):
            return NO_STARS
        return "" if spectra_here(centre) else NO_SPECTRA
    return ""
