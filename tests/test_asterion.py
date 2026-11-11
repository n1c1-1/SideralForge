"""The Asterion wrapper (core/asterion.py): its answers read in FLOAT's conventions, where the program and its index are
found, a field without a solution, a program that cannot run, the timeout and the stop. Without the program: recorded
answers of asterion.exe 0.4.2 (tests/data/asterion: the Heart and the Rosette stacks, a noise image, a folder without
an index) and a runner that gives them back. The level 3 tests run the real program on a rendered star field; they
are skipped where it or its index is not installed (``ASTRO_GPU_ASTERION``, ``ASTERION_CATALOGS``)."""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import psutil
import pytest

from astro_gpu.core import asterion as A
from astro_gpu.core.stage import Interrupted

DATA = Path(__file__).with_name("data") / "asterion"
#: the two recorded stacks: (width, height), and where FLOAT's own solution put their centres (its metrics)
#: where the level 3 tests find the real program and its index (the other tests see neither)
REAL = {k: os.environ.get(k) for k in (A.ENV_PROGRAM, A.ENV_CATALOGS)}
INSTALL_DIR = A.install_dir                     # the real one: every test but its own sees none
FIELDS = {"heart": ((4256, 2848), (40.75553, 61.12286)), "rosette": ((4256, 2848), (97.86539, 4.86855))}


def _float_wcs(name):
    from astropy.io import fits
    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(fits.Header.fromstring((DATA / f"{name}_float_wcs.txt").read_text(encoding="ascii"), sep="\n"))


def _recorded(name):
    answer = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    return A.solution_from(answer, A.read_matches(DATA / f"{name}.matches.csv"), FIELDS[name][0][::-1], 0.1)


@pytest.fixture(autouse=True)
def _nowhere(tmp_path, monkeypatch):
    """No program and no index but the ones a test puts somewhere (never the user's)."""
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "aghome"))
    for k in (A.ENV_PROGRAM, A.ENV_CATALOGS):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "userhome"))
    monkeypatch.setattr(A, "install_dir", lambda here=None: None)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """An installed Asterion whose runs give back a recorded answer: ``fake.answer = (exit code, stdout, stderr, the
    matches file or None)``; ``fake.cmds`` the command lines it got."""
    prog = tmp_path / "prog" / A.program_name()
    prog.parent.mkdir()
    prog.write_bytes(b"")
    cat = tmp_path / "cat"
    cat.mkdir()
    (cat / "tycho2.astx").write_bytes(b"x")
    monkeypatch.setenv(A.ENV_PROGRAM, str(prog.parent))
    monkeypatch.setenv(A.ENV_CATALOGS, str(cat))
    monkeypatch.setattr(A, "program_version", lambda exe: "0.4.2")

    class Fake:
        answer = (1, (DATA / "unsolved.json").read_text(), "", None)
        cmds: list = []

    def run(cmd, timeout, check=None):
        Fake.cmds.append(cmd)
        code, out, err, matches = Fake.answer
        image = Path(cmd[1])
        assert image.is_file()                       # the stack was written before the program ran
        if matches is not None:
            image.with_name(image.stem + ".matches.csv").write_text(matches.read_text())
        return code, out, err
    monkeypatch.setattr(A, "run", run)
    return Fake


# ------------------------------------------------------------------ conventions, on the two recorded stacks

@pytest.mark.parametrize("name", list(FIELDS))
def test_a_recorded_answer_is_in_floats_pixel_frame_as_it_is(name):
    """Asterion's WCS, read with pixel origin 0, is FLOAT's frame (rows as the stack stores them, top first): its
    matched stars fall on their pixels, its corners within 1.5 px of FLOAT's own solution of the same stack, its centre
    where the stage measures its own (pixel W/2, H/2), its mirror FLOAT's parity, its up angle FLOAT's."""
    from astropy.coordinates import SkyCoord
    sol = _recorded(name)
    wf = _float_wcs(name)
    (W, H), centre = FIELDS[name]
    m = sol.matches
    assert len(m) >= 100
    x, y = sol.wcs.world_to_pixel(SkyCoord(m[:, 2], m[:, 3], unit="deg"))
    # no shift (a pixel origin of 1, or half a pixel, would show here) and each star at its own fit residual
    assert abs(np.mean(x - m[:, 0])) < 0.03 and abs(np.mean(y - m[:, 1])) < 0.03
    d = np.hypot(x - m[:, 0], y - m[:, 1])
    assert d.max() < 0.5 and np.median(np.abs(d * sol.scale - m[:, 4])) < 0.05
    pts = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [W / 2, H / 2]], float)
    sep = sol.wcs.pixel_to_world(pts[:, 0], pts[:, 1]).separation(wf.pixel_to_world(pts[:, 0], pts[:, 1])).arcsec
    # the corners as near as two distortion fits on different catalogues come (Tycho-2, Gaia): within the 3 px the
    # stage's fine fit starts from
    assert sep[:4].max() / sol.scale < 1.5 and sep[4] < 0.5, sep
    assert SkyCoord(sol.ra, sol.dec, unit="deg").separation(SkyCoord(*centre, unit="deg")).arcsec < 1.0
    assert np.linalg.det(wf.pixel_scale_matrix) > 0 and sol.flipped           # FLOAT's parity_flipped, stored top first
    c, north = wf.pixel_to_world(W / 2, H / 2), wf.pixel_to_world(W / 2, H / 2 - 100)
    assert abs((sol.up_position_angle - c.position_angle(north).deg + 180) % 360 - 180) < 0.05
    assert sol.scale == pytest.approx(abs(wf.proj_plane_pixel_scales()[0].to_value("arcsec")), rel=2e-3)
    assert sol.matched >= len(m) and sol.detected == 1000 and sol.index == "tycho2"


def test_the_stack_is_written_as_it_is_stored(tmp_path):
    """Row 0 of the stack is the first row of the FITS file (no flip: the conventions above hang on it), the channels
    summed, the pixels without data at the median rather than 0 (a step at a drizzle's edge is not a star), no header
    keyword that would hint a position or a scale."""
    from astropy.io import fits
    img = np.zeros((3, 20, 25), np.float32)
    img[:, 0, :] = 1.0
    img[1, 3, 4] = np.nan
    A.write_fits(img, tmp_path / "s.fits")
    with fits.open(tmp_path / "s.fits") as f:
        data, hdr = f[0].data, f[0].header
    assert data.dtype.kind == "f" and data.dtype.itemsize == 4 and data.shape == (20, 25)
    assert (data[0] == 3).all() and data[3, 4] == 0.0 and np.isfinite(data).all()
    assert not {"RA", "DEC", "OBJCTRA", "PIXSCALE", "FOCALLEN", "XPIXSZ", "ROWORDER"} & set(hdr)


def test_the_command_line_limits_the_search_to_what_is_known(tmp_path):
    exe, img, cat = Path("a.exe"), Path("s.fits"), Path("c")
    blind = A.command(exe, img, cat, timeout=3)
    assert "--no-hints" in blind and "--json" in blind and "--matches" in blind and "--ra" not in blind
    assert blind[blind.index("--timeout") + 1] == "3" and blind[blind.index("--catalog-dir") + 1] == "c"
    c = A.command(exe, img, cat, ra=370.0, dec=-5.5, radius=4, scale=2.0, sigma=15, binning=2)
    assert c[c.index("--ra") + 1] == "10.000000" and c[c.index("--radius") + 1] == "4"
    lo, hi = float(c[c.index("--scale-low") + 1]), float(c[c.index("--scale-high") + 1])
    assert lo == pytest.approx(2.0 / 1.25) and hi == pytest.approx(2.5)
    assert c[c.index("--sigma") + 1] == "15" and c[c.index("--binning") + 1] == "2"


# ------------------------------------------------------------------ solve, through the recorded answers

def test_a_solution_comes_back_and_its_files_are_removed(tmp_path, fake):
    fake.answer = (0, (DATA / "heart.json").read_text(), "", DATA / "heart.matches.csv")
    info: dict = {}
    sol = A.solve(np.ones((3, 2848, 4256), np.float32), tmp_path / "work", ra=40.76, dec=61.11, radius=10, info=info)
    assert sol is not None and sol.matched == 892 and len(sol.matches) == 300
    assert sol.scale == pytest.approx(6.36828, rel=1e-5) and sol.flipped
    assert info["exit"] == 0 and info["detected"] == 1000 and info["seconds"] > 0
    assert list((tmp_path / "work").iterdir()) == []                # the FITS file and the matches are gone
    assert "--ra" in fake.cmds[-1]


def test_no_solution_is_none_with_what_happened(tmp_path, fake):
    fake.answer = (1, (DATA / "unsolved.json").read_text(), "", None)
    info: dict = {}
    assert A.solve(np.ones((60, 80), np.float32), tmp_path, info=info) is None
    assert info["exit"] == 1 and info["detected"] == 0


def test_no_answer_within_the_timeout_is_none(tmp_path, fake):
    fake.answer = (None, "", "", None)
    info: dict = {}
    assert A.solve(np.ones((60, 80), np.float32), tmp_path, timeout=2, info=info) is None
    assert "within" in info["why"] and list(tmp_path.glob("asterion_stack*")) == []


def test_a_program_that_fails_is_a_clear_error(tmp_path, fake):
    """Exit 2 (no index in its folder, bad options): an error with the program's own words, never a "no solution"."""
    fake.answer = (2, "", (DATA / "no_catalog.stderr").read_text(), None)
    with pytest.raises(A.AsterionError, match="No catalog is installed"):
        A.solve(np.ones((60, 80), np.float32), tmp_path)
    fake.answer = (-1073741819, "", "", None)                        # a crash: no answer at all
    with pytest.raises(A.AsterionError, match="exit -1073741819"):
        A.solve(np.ones((60, 80), np.float32), tmp_path)


# ------------------------------------------------------------------ installed or not

def test_nothing_installed_says_so_and_solve_raises(tmp_path):
    st = A.status()
    assert not st.ok and "not installed" in st.why and st.program is None and st.missing
    assert not A.installed()
    with pytest.raises(A.AsterionError, match="not installed"):
        A.solve(np.ones((60, 80), np.float32), tmp_path)


def test_a_program_without_its_index_or_too_old_says_why(tmp_path, monkeypatch):
    prog = A.own_dir() / A.program_name()
    prog.parent.mkdir(parents=True)
    prog.write_bytes(b"")
    monkeypatch.setattr(A, "program_version", lambda exe: "0.4.2")
    st = A.status()
    assert not st.ok and st.program == prog and "star index" in st.why and st.missing
    monkeypatch.setattr(A, "program_version", lambda exe: "0.3.9")
    assert "too old" in A.status().why and not A.status().missing
    monkeypatch.setattr(A, "program_version", lambda exe: "")
    assert "does not run" in A.status().why


def test_where_the_program_and_the_index_are_looked_for(tmp_path, monkeypatch):
    """The setting first, then the environment, then FLOAT's own folders; the index of an Asterion the user installed
    himself (~/.asterion/catalogs) last; a folder without an .astx is no index."""
    monkeypatch.setattr(A, "program_version", lambda exe: "0.4.2")

    def put(folder: Path, name: str) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"x")
        return folder
    own = put(A.own_dir(), A.program_name())
    assert A.find_program() == own / A.program_name()
    env = put(tmp_path / "env", A.program_name())
    monkeypatch.setenv(A.ENV_PROGRAM, str(env))
    assert A.find_program() == env / A.program_name()
    monkeypatch.setenv(A.ENV_PROGRAM, str(env / A.program_name()))       # the program itself
    assert A.find_program() == env / A.program_name()
    setting = put(tmp_path / "setting", A.program_name())
    assert A.find_program(setting) == setting / A.program_name()
    theirs = put(Path.home() / ".asterion" / "catalogs", "tycho2.astx")
    assert A.find_catalogs() == theirs
    ours = put(A.own_catalogs(), "gaia-500.astx")
    assert ours == own / "catalogs" and A.find_catalogs() == ours
    beside = put(setting / "catalogs", "tycho2.astx")                     # a program and its index in one folder
    assert A.find_catalogs(program=A.find_program(setting)) == beside
    assert A.status(setting).catalogs == beside and A.status().catalogs == ours
    put(tmp_path / "notes", "readme.txt")
    monkeypatch.setenv(A.ENV_CATALOGS, str(tmp_path / "notes"))
    assert A.find_catalogs() == ours
    monkeypatch.setenv(A.ENV_CATALOGS, str(theirs))
    assert A.find_catalogs() == theirs and A.status().indexes == ["tycho2"]


def test_an_installed_float_finds_the_program_beside_its_code(tmp_path):
    """The installer puts the code in <install>/app: the program is looked for in <install>/asterion; a source
    checkout has no such folder (its parent is nobody's)."""
    installed = tmp_path / "FLOAT" / "app" / "astro_gpu" / "core" / "asterion.py"
    assert INSTALL_DIR(installed) == tmp_path / "FLOAT" / "asterion"
    assert INSTALL_DIR(tmp_path / "src" / "astro" / "astro_gpu" / "core" / "asterion.py") is None


# ------------------------------------------------------------------ the process: timeout and stop

def _sleeper(tmp_path) -> tuple[list[str], Path]:
    """A program that says its process id and waits: the base Python, not a virtual environment's launcher (it runs the
    interpreter as a child of its own, which killing the launcher would leave behind)."""
    pid = tmp_path / "pid.txt"
    code = f"import os, time; open(r'{pid}', 'w').write(str(os.getpid())); time.sleep(60)"
    return [getattr(sys, "_base_executable", sys.executable), "-c", code], pid


def _wait_pid(pid: Path) -> int:
    end = time.monotonic() + 20
    while not (pid.is_file() and pid.read_text()) and time.monotonic() < end:
        time.sleep(0.05)
    return int(pid.read_text())


def test_the_program_is_killed_at_the_timeout(tmp_path):
    cmd, pid = _sleeper(tmp_path)
    t = time.monotonic()
    assert A.run(cmd, 1.0) == (None, "", "")
    assert time.monotonic() - t < 10
    assert not psutil.pid_exists(_wait_pid(pid))


def test_a_stop_kills_the_program_and_is_raised(tmp_path):
    cmd, pid = _sleeper(tmp_path)

    def check():
        if pid.is_file() and pid.read_text():
            raise Interrupted("stop requested")
    t = time.monotonic()
    with pytest.raises(Interrupted):
        A.run(cmd, 60.0, check)
    assert time.monotonic() - t < 15
    assert not psutil.pid_exists(_wait_pid(pid))


def test_a_program_that_cannot_start_is_a_clear_error(tmp_path):
    with pytest.raises(A.AsterionError, match="cannot be started"):
        A.run([str(tmp_path / "missing.exe")], 5.0)


# ------------------------------------------------------------------ level 3: the real program

def _real(monkeypatch):
    for k, v in REAL.items():
        if v:
            monkeypatch.setenv(k, v)
    st = A.status()
    if not st.ok:
        pytest.skip(f"Asterion is not installed here: {st.why}")


def _field(flipped: bool, W=1500, H=1000, scale=20.0, rot=30.0):
    """A star field rendered through a WCS of FLOAT's frame: the matched stars of the recorded Heart (Tycho-2 stars),
    brightest first as Asterion listed them. (image, wcs, the stars' pixels)."""
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    m = A.read_matches(DATA / "heart.matches.csv")
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [40.75, 61.12]
    w.wcs.crpix = [W / 2 + 1, H / 2 + 1]                 # astropy's own: 1-based
    t = np.radians(rot)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    w.wcs.cd = scale / 3600 * (R if flipped else R @ np.diag([-1.0, 1.0]))
    x, y = w.world_to_pixel(SkyCoord(m[:, 2], m[:, 3], unit="deg"))
    rng = np.random.default_rng(3)
    img = rng.normal(100, 2, (H, W)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    keep = (x > 3) & (x < W - 4) & (y > 3) & (y < H - 4)
    for k, (sx, sy) in enumerate(zip(x[keep], y[keep])):
        f = 4000.0 * 0.985 ** k
        sl = np.s_[int(sy) - 6:int(sy) + 7, int(sx) - 6:int(sx) + 7]
        img[sl] += f * np.exp(-((xx[sl] - sx) ** 2 + (yy[sl] - sy) ** 2) / (2 * 1.3 ** 2))
    return img, w, np.c_[x[keep], y[keep]], m[keep]


@pytest.mark.level3
@pytest.mark.parametrize("flipped", [True, False])
def test_real_the_program_solves_a_rendered_field_in_floats_frame(tmp_path, monkeypatch, flipped):
    from astropy.coordinates import SkyCoord
    _real(monkeypatch)
    img, w, xy, m = _field(flipped)
    assert len(xy) > 60
    info: dict = {}
    sol = A.solve(img, tmp_path, ra=41.0, dec=61.0, radius=5, timeout=20, info=info)
    assert sol is not None, info
    assert sol.flipped is flipped
    assert sol.scale == pytest.approx(20.0, rel=5e-3)
    x, y = sol.wcs.world_to_pixel(SkyCoord(m[:, 2], m[:, 3], unit="deg"))
    assert np.median(np.hypot(x - xy[:, 0], y - xy[:, 1])) < 0.3
    assert sol.seconds < 10 and list(tmp_path.iterdir()) == []


@pytest.mark.level3
def test_real_noise_has_no_solution_and_a_stop_ends_the_program(tmp_path, monkeypatch):
    _real(monkeypatch)
    rng = np.random.default_rng(0)
    noise = rng.normal(100, 5, (600, 800)).astype(np.float32)
    info: dict = {}
    t = time.monotonic()
    assert A.solve(noise, tmp_path, timeout=10, info=info) is None
    assert info["exit"] == A.UNSOLVED and time.monotonic() - t < 15

    def stop():
        raise Interrupted("stop requested")
    with pytest.raises(Interrupted):
        A.solve(noise, tmp_path, timeout=10, check=stop)
    assert list(tmp_path.iterdir()) == []
