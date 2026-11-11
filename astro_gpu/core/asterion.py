"""Asterion, the plate solver FLOAT calls for the first solution of a stack (position, scale, orientation, mirror).

Why: FLOAT's own search tries trial scales one after the other, each with its catalogue and a triangle match (22 of
them when the focal length is unknown); Asterion matches shapes of four stars that depend neither on the scale nor on
the orientation, against an index of the sky, and answers in a fraction of a second, a failure within a second or
two. FLOAT's fine fit on the Gaia stars starts from its solution; FLOAT's own search stays as the fallback.

How it is called: the command line program on a FITS file of the stack's luminance written in the stage's scratch
folder, its answer read from its JSON (``--json``) and its matched stars from the CSV it writes beside the image
(``--matches``). Not the native library: the program is the one file of the released archive (the library is another
archive, and a source build makes the same program), it holds no state between two stacks worth keeping (a stack is
solved once per run, and starting the program costs 0.06 s), a crash of it cannot take FLOAT down, and a stop kills it
at once. Not a list of FLOAT's stars either: the command line takes images only, and its own detection sorts the
detections (too wide to be stars left out, a pass on the detections whose light is concentrated like a star's) which
is what solves the nebulous fields.

Conventions (checked on 103 stacks of the Heart, the Rosette, the Pleiades, NGC 7000 and a Seestar field against
FLOAT's own solutions, tools/bench/astrometry_compare.py): the stack is written as it is stored, row 0 first (astropy
writes ``data[0]`` as the first FITS row, and Asterion reads a FITS file in its row order: there is no ROWORDER keyword
to honour), so the WCS Asterion gives, read by astropy with pixel origin 0, is in FLOAT's pixel frame as it is (x the
column, y the row of the stored array, the centre of the first pixel at (0, 0)), the frame of ``wcs.fits``: the centres
agree within 0.4", the scales within 0.05 %, the orientations within 0.03 deg, and the corners as near as two
distortion fits on different catalogues come (2 px at most, 3 to 9 px on two fields where FLOAT's corners are the ones
off: the Tycho-2 stars measured there fall 1.5 to 3 px from FLOAT's solution, 0.2 px from Asterion's). Its matched
stars' x, y are 0-based in the same frame. Asterion calls "flipped" an image whose determinant of the CD matrix is
positive, which is FLOAT's ``parity_flipped`` and the ``mirror`` of the triangle match (every stack stored top row
first: not a real mirror). Its "rotation" is the position angle of the image's +y axis (east of north), so FLOAT's
``up_position_angle`` (the angle of -y, the top of the picture) is rotation + 180.

Where (the first found): the program, the ``exe`` given (the setting), ``ASTRO_GPU_ASTERION`` (its folder, or the
program itself), FLOAT's own folder ``<ASTRO_GPU_HOME>/asterion`` (where Settings would put it), and, in an installed
FLOAT, the ``asterion`` folder of the installation (beside ``app``); the index, the ``catalogs`` given, Asterion's own
``ASTERION_CATALOGS``, the ``catalogs`` folder beside the program found, then beside FLOAT's own and the installation's,
and, as an extra place only, never required, Asterion's default ``~/.asterion/catalogs`` (an Asterion the user installed
himself). One layout whichever way the program came (the released archive or a build from its sources, which makes the
same single file)::

    asterion/asterion.exe                 the program (``asterion`` elsewhere), with LICENSE.txt and third-party/
    asterion/catalogs/tycho2.astx         the index (378 MB), and any other <name>.astx installed
    asterion/catalogs/tycho2.astx.wide    the quads of wide fields, which Asterion writes beside an index the first time
                                          it solves a wide field: shipped with it, or the folder must be writable

Nothing is ever downloaded here.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .stage import StageError

ENV_PROGRAM = "ASTRO_GPU_ASTERION"
ENV_CATALOGS = "ASTERION_CATALOGS"
#: the oldest version whose answers this module reads (its JSON keys, the matches file, the exit codes)
MIN_VERSION = (0, 4, 2)
#: the relative error the scale hint tolerates, both ways (Asterion's own for a scale read from a file header)
SCALE_TOLERANCE = 0.25
#: seconds the process is given beyond its own ``--timeout`` (its start, the image read) before it is killed
GRACE_S = 5.0
#: how often the process is watched for the end, the stop and the timeout
POLL_S = 0.05
#: Asterion's exit codes: solved, no solution, an error (no catalogue, bad options), cancelled
SOLVED, UNSOLVED, ERROR, CANCELLED = 0, 1, 2, 130


class AsterionError(StageError):
    """The program cannot be run (missing, not a program, too old, no index, a crash): never a field it cannot solve."""


@dataclass
class Solution:
    """A first solution in FLOAT's conventions (module docstring). ``wcs``: TAN(-SIP) of astropy in FLOAT's pixel frame;
    ``ra``, ``dec``: the sky at pixel (W/2, H/2), where the stage measures its own centre; ``scale``: arcsec per pixel
    of the stack; ``rotation``: Asterion's (the +y axis, east of north); ``up_position_angle``: the stage's (-y);
    ``flipped``: the stage's ``parity_flipped`` and its triangle match's mirror; ``matches``: the matched stars, (n, 5)
    x, y (FLOAT's pixels), ra, dec (deg), residual (arcsec); ``seconds``: the whole call, the FITS file written
    included."""
    wcs: object
    ra: float
    dec: float
    scale: float
    rotation: float
    up_position_angle: float
    flipped: bool
    matched: int
    detected: int
    rms_arcsec: float
    log_odds: float
    index: str
    seconds: float
    matches: np.ndarray = field(repr=False)
    timing: dict = field(default_factory=dict, repr=False)


@dataclass
class Status:
    """Whether Asterion can solve here: ``ok``; ``program``, ``version``, ``catalogs`` and ``indexes`` found; ``why``
    a plain English sentence when not; ``missing``: not ok because the program or its index is not there (not
    installed), rather than there and unusable (too old, not the program, which the stage says as a warning)."""
    ok: bool
    program: Path | None = None
    version: str = ""
    catalogs: Path | None = None
    indexes: list[str] = field(default_factory=list)
    why: str = ""
    missing: bool = False


# ------------------------------------------------------------------ where the program and its index are

def program_name() -> str:
    return "asterion.exe" if sys.platform == "win32" else "asterion"


def own_dir() -> Path:
    """FLOAT's own folder of the program: ``<ASTRO_GPU_HOME>/asterion``."""
    from .skycache import home
    return home() / "asterion"


def own_catalogs() -> Path:
    """FLOAT's own folder of the index: ``catalogs`` beside its program (``<ASTRO_GPU_HOME>/asterion/catalogs``)."""
    return own_dir() / "catalogs"


def install_dir(here: Path | None = None) -> Path | None:
    """The ``asterion`` folder of an installed FLOAT: beside ``app``, the folder the installer puts the code in (none
    in a source checkout, whose parent folder is nobody's). ``here``: this file."""
    app = Path(here or __file__).resolve().parents[2]
    return app.parent / "asterion" if app.name == "app" else None


def _as_program(p: Path) -> Path | None:
    p = Path(p)
    if p.is_dir():
        p = p / program_name()
    return p if p.is_file() else None


def find_program(exe: str | os.PathLike | None = None) -> Path | None:
    for c in [exe, os.environ.get(ENV_PROGRAM), own_dir(), install_dir()]:
        if c and str(c).strip():
            found = _as_program(Path(str(c).strip()))
            if found is not None:
                return found
    return None


def indexes(folder: Path) -> list[str]:
    """The names of the index files of a folder (``tycho2`` for ``tycho2.astx``)."""
    try:
        return sorted(p.stem for p in folder.glob("*.astx") if p.is_file() and p.stat().st_size > 0)
    except OSError:
        return []


def find_catalogs(catalogs: str | os.PathLike | None = None, program: Path | None = None) -> Path | None:
    """The first folder holding an index (``*.astx``); ``program``: the one found, whose ``catalogs`` folder comes
    before FLOAT's own (a program and its index given together, in one folder, stay together)."""
    inst = install_dir()
    for c in [catalogs, os.environ.get(ENV_CATALOGS), program.parent / "catalogs" if program else None, own_catalogs(),
              inst / "catalogs" if inst else None, Path.home() / ".asterion" / "catalogs"]:
        if c and str(c).strip() and indexes(Path(str(c).strip())):
            return Path(str(c).strip())
    return None


def _hidden() -> dict:
    """No console window on Windows (FLOAT's workers have one, hidden; a program they start would open its own), and a
    priority under FLOAT's own work."""
    if sys.platform != "win32":
        return {}
    return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS}


_VERSIONS: dict[tuple[str, float, int], str] = {}
_VERSIONS_LOCK = threading.Lock()


def program_version(exe: Path) -> str:
    """``0.4.2`` from ``asterion --version`` (asked once per program file: its path, date and size), "" when it does not
    answer as Asterion."""
    try:
        st = exe.stat()
    except OSError:
        return ""
    key = (str(exe), st.st_mtime, st.st_size)
    with _VERSIONS_LOCK:
        if key in _VERSIONS:
            return _VERSIONS[key]
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=30, stdin=subprocess.DEVNULL, **_hidden())
        m = re.search(r"Asterion\D*(\d+)\.(\d+)\.(\d+)", r.stdout)
        v = ".".join(m.groups()) if m else ""
    except (OSError, subprocess.SubprocessError):
        v = ""
    with _VERSIONS_LOCK:
        _VERSIONS[key] = v
    return v


def status(exe: str | os.PathLike | None = None, catalogs: str | os.PathLike | None = None) -> Status:
    """Whether Asterion can solve on this computer, and why not (cheap: file lookups, and the version asked once)."""
    prog = find_program(exe)
    if prog is None:
        return Status(False, why="the Asterion plate solver is not installed", missing=True)
    v = program_version(prog)
    if not v:
        return Status(False, prog, why=f"{prog} does not run as the Asterion plate solver")
    if tuple(int(x) for x in v.split(".")) < MIN_VERSION:
        return Status(False, prog, v, why=f"Asterion {v} is too old, {'.'.join(map(str, MIN_VERSION))} at least is "
                                          f"needed")
    cat = find_catalogs(catalogs, prog)
    if cat is None:
        return Status(False, prog, v, why="the star index of the Asterion plate solver is not installed",
                      missing=True)
    return Status(True, prog, v, cat, indexes(cat))


def installed(exe: str | os.PathLike | None = None, catalogs: str | os.PathLike | None = None) -> bool:
    return status(exe, catalogs).ok


# ------------------------------------------------------------------ solving

def _luminance(image: np.ndarray) -> np.ndarray:
    """One plane: the sum of the channels of a (C, H, W) stack (Asterion's own reading of a colour image), its pixels
    without data (NaN, inf: the edges a drizzle leaves) at the median, not 0: a step at the edge is not a star."""
    a = np.asarray(image, dtype=np.float32)
    if a.ndim == 3:
        a = a.sum(0, dtype=np.float32)
    elif a.ndim != 2:
        raise ValueError(f"an image of 2 or 3 dimensions is needed, not {a.shape}")
    else:
        a = a.copy()
    bad = ~np.isfinite(a)
    if bad.any():
        # the median of one pixel in 64 (the whole of a 27 Mpx stack's takes a second for the same level)
        sub = a[::8, ::8]
        sub = sub[np.isfinite(sub)]
        a[bad] = float(np.median(sub)) if sub.size else 0.0
    return a


def write_fits(image: np.ndarray, path: Path) -> Path:
    """The image as Asterion reads it: one float32 plane, rows in the stored order, no keyword that would hint a
    position or a scale (``--no-hints`` too)."""
    from astropy.io import fits
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(_luminance(image)).writeto(path, overwrite=True)
    return path


def command(exe: Path, image: Path, catalogs: Path, ra=None, dec=None, radius=None, scale=None,
            scale_tol=SCALE_TOLERANCE, timeout=10.0, sigma=None, binning=None) -> list[str]:
    cmd = [str(exe), str(image), "--json", "--plain", "--matches", "--no-hints", "--catalog-dir", str(catalogs),
           "--timeout", f"{max(1.0, float(timeout)):g}"]
    if ra is not None and dec is not None:
        # a limit, not a hint: searched from the position outwards and never beyond the radius
        cmd += ["--ra", f"{float(ra) % 360:.6f}", "--dec", f"{float(dec):.6f}"]
        if radius is not None:
            cmd += ["--radius", f"{float(radius):g}"]
    if scale:
        cmd += ["--scale-low", f"{scale / (1 + scale_tol):.6g}", "--scale-high", f"{scale * (1 + scale_tol):.6g}"]
    if sigma is not None:
        cmd += ["--sigma", f"{float(sigma):g}"]
    if binning is not None:
        cmd += ["--binning", str(int(binning))]
    return cmd


def _answer(stdout: str) -> dict:
    """The JSON object of one image (the whole stdout; a list or a ``results`` wrapper taken too)."""
    i = stdout.find("{")
    j = json.loads(stdout[i:]) if i >= 0 else {}
    if isinstance(j, dict) and isinstance(j.get("results"), list):
        j = j["results"][0] if j["results"] else {}
    if isinstance(j, list):
        j = j[0] if j else {}
    return j if isinstance(j, dict) else {}


def read_matches(path: Path) -> np.ndarray:
    """(n, 5): x, y, ra, dec, residual (arcsec) of the matched stars; (0, 5) when the file is missing."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [[float(r[k]) for k in ("x", "y", "ra", "dec", "residual_arcsec")] for r in csv.DictReader(f)]
    except (OSError, KeyError, ValueError):
        return np.zeros((0, 5))
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def solution_from(answer: dict, matches: np.ndarray, shape: tuple[int, int], seconds: float) -> Solution | None:
    """Asterion's answer in FLOAT's conventions, None when it is not a solution."""
    import warnings

    from astropy.io import fits
    from astropy.wcs import WCS
    if not answer.get("solved") or not isinstance(answer.get("wcs"), dict):
        return None
    hdr = fits.Header()
    for k, v in answer["wcs"].items():
        hdr[k] = v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # astropy's notes on the SIP keywords of a TAN-SIP header
        w = WCS(hdr)
    H, W = shape
    c = w.pixel_to_world(W / 2, H / 2)
    rot = float(answer.get("rotation", np.nan))
    return Solution(wcs=w, ra=float(c.ra.deg), dec=float(c.dec.deg), scale=float(answer["scale"]), rotation=rot,
                    up_position_angle=float((rot + 180.0) % 360.0), flipped=bool(answer.get("flipped")),
                    matched=int(answer.get("matchedStars", len(matches))), detected=int(answer.get("stars", 0)),
                    rms_arcsec=float(answer.get("rmsArcsec", np.nan)), log_odds=float(answer.get("logOdds", np.nan)),
                    index=str(answer.get("index", "")), seconds=seconds, matches=matches,
                    timing=dict(answer.get("timing") or {}))


def run(cmd: list[str], timeout: float, check: Callable[[], None] | None = None) -> tuple[int | None, str, str]:
    """(exit code, stdout, stderr) of the program; None as the code when it was killed at ``timeout`` seconds. ``check``
    is called while it runs (the stage's stop): whatever it raises kills the process first, then goes on."""
    import tempfile
    # the answers go to files, not pipes: a pipe nobody reads while waiting would stop a program that writes much
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            p = subprocess.Popen(cmd, stdout=out, stderr=err, stdin=subprocess.DEVNULL, **_hidden())
        except OSError as e:
            raise AsterionError(f"the Asterion plate solver cannot be started ({e})") from e
        end = time.monotonic() + timeout
        try:
            while p.poll() is None:
                if check is not None:
                    check()
                if time.monotonic() > end:
                    p.kill()
                    p.wait()
                    return None, "", ""
                time.sleep(POLL_S)
        except BaseException:
            p.kill()
            p.wait()
            raise
        out.seek(0)
        err.seek(0)
        return (p.returncode, out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace"))


def solve(image: np.ndarray, work: str | os.PathLike, ra: float | None = None, dec: float | None = None,
          radius: float | None = None, scale: float | None = None, scale_tol: float = SCALE_TOLERANCE,
          timeout: float = 10.0, check: Callable[[], None] | None = None, exe=None, catalogs=None,
          sigma: float | None = None, binning: int | None = None, info: dict | None = None) -> Solution | None:
    """The first solution of a stack, or None when Asterion finds none (or none within ``timeout`` seconds: it is
    then killed). ``image``: the stack as FLOAT stores it, (C, H, W) or (H, W); ``work``: a folder for the FITS file and
    the matches (the stage's scratch; both removed after); ``ra``, ``dec``, ``radius`` (deg): where the field is, a limit
    of the search (none: the whole sky); ``scale`` (arcsec/px of the stack): a limit too, within ``scale_tol``; ``check``:
    called while the program runs (``ctx.check_stop``: a stop kills it and is raised); ``sigma``, ``binning``: Asterion's
    detection threshold and binning, its own defaults when None; ``info``: filled with what happened (seconds,
    detected stars, exit code, the program's message). Raises AsterionError only when the program cannot be run."""
    t0 = time.monotonic()
    st = status(exe, catalogs)
    if not st.ok:
        raise AsterionError(st.why)
    work = Path(work)
    img = work / "asterion_stack.fits"
    csv_path = img.with_name(img.stem + ".matches.csv")
    shape = tuple(np.shape(image)[-2:])
    try:
        write_fits(image, img)
        cmd = command(st.program, img, st.catalogs, ra, dec, radius, scale, scale_tol, timeout, sigma, binning)
        code, out, err = run(cmd, timeout + GRACE_S, check)
        seconds = time.monotonic() - t0
        if info is not None:
            info.update(seconds=seconds, exit=code, version=st.version, index=",".join(st.indexes))
        if code is None:
            if info is not None:
                info["why"] = f"no answer within {timeout + GRACE_S:g} s"
            return None
        try:
            answer = _answer(out)
        except ValueError:
            answer = {}
        if info is not None:
            info.update(detected=answer.get("stars"), timing=answer.get("timing"),
                        why=answer.get("error") or answer.get("message") or err.strip()[-300:])
        if code == CANCELLED:
            return None
        if code not in (SOLVED, UNSOLVED) or (code == SOLVED and not answer):
            raise AsterionError(f"the Asterion plate solver failed (exit {code}): "
                                f"{(err.strip() or out.strip())[-300:] or 'no message'}")
        return solution_from(answer, read_matches(csv_path), shape, seconds) if code == SOLVED else None
    finally:
        for p in (img, csv_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
