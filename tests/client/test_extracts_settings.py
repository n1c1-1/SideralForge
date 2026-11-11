"""Settings › Catalogs & astrometry: the "Offline Gaia extracts" section, offscreen.

Folder, state of each extract, the two downloads run off the GUI thread, progress, cancel. The download itself is
replaced by a function that reports progress (its real behaviour: tests/test_gaia_extracts.py); nothing is written
outside a scratch folder.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QProgressBar, QPushButton  # noqa: E402

from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.services import extracts  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402
from astro_gpu.client.settingsdlg import SettingsDialog  # noqa: E402
from astro_gpu.core import gaia_extracts as ge  # noqa: E402
from astro_gpu.core import sirilcat  # noqa: E402

MACHINE = Machine([Gpu("cuda:0", "NVIDIA RTX 4060 Ti", 16.0)], 16, 64.0)
ASTRO_NAME = "siril_cat_healpix8_astro.dat"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    (tmp_path / "nowhere").mkdir()
    for v in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(v, str(tmp_path / "nowhere"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "userhome"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    prefs.load()
    prefs.set("spec_mode", False)
    i18n.set_language("en")
    sirilcat.reset()
    yield tmp_path
    i18n.set_language("en")
    sirilcat.reset()


def _texts(root) -> list[str]:
    return [w.text() for w in root.findChildren(QLabel) if w.isVisibleTo(root) and w.text()] + \
           [w.text() for w in root.findChildren(QPushButton) if w.isVisibleTo(root)]


def _gaia(shown: list[str]) -> list[str]:
    """The texts of this section only: the Asterion section after it says its own "Not installed"."""
    end = [i for i, t in enumerate(shown) if t in ("Asterion plate solver", "Solveur astrométrique Asterion")]
    return shown[:end[0]] if end else shown


def _page(qapp):
    d = SettingsDialog(None, MACHINE)
    d.show()
    d.show_page("catalogs")
    qapp.processEvents()
    return d, d.pages.currentWidget()


def _button(page, start):
    return next(b for b in page.findChildren(QPushButton) if b.text().startswith(start))


def _wait(qapp, cond, seconds=10.0):
    end = time.monotonic() + seconds
    while not cond() and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    assert cond()


def _tiny_astro(path: Path):
    h = bytearray(128)
    h[:48] = b"Siril test extract".ljust(48, b"\0")
    h[48:53] = bytes([3, 1, 1, 0, 0])
    rec = np.zeros(4, sirilcat.ASTRO_DT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(h) + np.full(48, 4, "<u4").tobytes() + rec.tobytes())


def test_the_section_shows_the_folder_the_state_and_the_two_downloads(env, qapp):
    d, page = _page(qapp)
    shown = _texts(page)
    assert "Offline Gaia extracts" in shown and "Without them" in " ".join(shown)
    assert "Download positions (1.1 GB)" in shown and "Download spectra (10.6 GB)" in shown
    assert _gaia(shown).count("Not installed") == 3 and "Open folder" in shown        # the bright stars too
    assert not [t for t in shown if "Gaia DR3 folder" in t or "12 GB" in t or "◇" in t]      # the old, unbuilt widgets
    edit = next(e for e in page.findChildren(QLineEdit) if e.text() == str(ge.folder()))
    assert edit.isReadOnly() and ge.folder() == env / "home" / "cache" / "gaia_extracts"
    assert not [b for b in page.findChildren(QProgressBar) if b.isVisibleTo(page)]            # idle: no bar
    assert _button(page, "Download positions").isEnabled()
    assert not ge.folder().exists()                                  # looking at the page writes nothing
    d.close()


def test_each_extract_says_which_block_needs_it(env, qapp):
    """The positions alone leave the Color block out of the runs (it calibrates on the spectra): the spectra's row says
    so next to their download, and the positions' row says what needs them."""
    d, page = _page(qapp)
    shown = _gaia(_texts(page))
    assert "Needed by Astrometry" in shown and "Needed by the Color block" in shown
    assert shown.index("Star positions") < shown.index("Needed by Astrometry") < shown.index("Star spectra") < \
        shown.index("Needed by the Color block")
    d.close()
    i18n.set_language("fr")
    d, page = _page(qapp)
    shown = _texts(page)
    assert "Nécessaires au bloc Couleur" in shown and "Nécessaires à l'astrométrie" in shown
    d.close()


def test_installed_and_found_elsewhere_are_told_apart(env, qapp):
    other = env / "nowhere" / "siril"                               # another program's data folder
    _tiny_astro(other / ASTRO_NAME)
    sirilcat.reset()
    d, page = _page(qapp)
    shown = _texts(page)
    assert f"Found in another program's data folder: {other}" in shown
    assert _button(page, "Download positions").isEnabled()      # an own copy can still be downloaded
    _tiny_astro(ge.folder() / ASTRO_NAME)
    sirilcat.reset()
    d.rebuild()
    qapp.processEvents()
    page = d.pages.currentWidget()
    shown = _texts(page)
    assert "Installed" in shown and _gaia(shown).count("Not installed") == 2
    b = _button(page, "Download positions")
    assert not b.isEnabled() and b.toolTip() == "Already installed."
    # a download stopped half-way
    (ge.folder() / "siril_cat1_healpix8_xpsamp").mkdir()
    (ge.folder() / "siril_cat1_healpix8_xpsamp" / "siril_cat1_healpix8_xpsamp_3.dat.bz2.part").write_bytes(b"x" * 3_000_000)
    d.rebuild()
    qapp.processEvents()
    assert "Partly downloaded: 3 MB" in _texts(d.pages.currentWidget())
    d.close()


def test_a_download_runs_off_the_gui_thread_and_reports(env, qapp, monkeypatch):
    main = threading.get_ident()
    ran_in = []

    def fake(kind, progress=None, cancel=None, **kw):
        ran_in.append(threading.get_ident())
        progress(0, 1000, "Reading the file list from Zenodo")
        progress(400, 1000, "File 1 of 2: downloading 0.4 GB of 1.0 GB")
        progress(1000, 1000, "Done")
        _tiny_astro(ge.folder() / ASTRO_NAME)
        sirilcat.reset()
    monkeypatch.setattr(ge, "download", fake)
    d, page = _page(qapp)
    _button(page, "Download positions").click()
    _wait(qapp, lambda: not extracts.job().running() and "Done." in _texts(d.pages.currentWidget()))
    assert ran_in and ran_in[0] != main
    page = d.pages.currentWidget()
    assert not [b for b in page.findChildren(QProgressBar) if b.isVisibleTo(page)]
    assert "Installed" in _texts(page) and not _button(page, "Download positions").isEnabled()
    d.close()


def test_cancel_stops_the_download_and_closing_the_window_does_not(env, qapp, monkeypatch):
    started = threading.Event()
    stopped = []

    def slow(kind, progress=None, cancel=None, **kw):
        progress(100, 1000, "File 1 of 48: downloading 0.1 GB of 0.2 GB")
        started.set()
        while not cancel.wait(0.01):
            pass
        stopped.append(kind)
        raise ge.Cancelled()
    monkeypatch.setattr(ge, "download", slow)
    d, page = _page(qapp)
    _button(page, "Download spectra").click()
    _wait(qapp, started.is_set)
    _wait(qapp, lambda: any("downloading 0.1 GB" in t for t in _texts(d.pages.currentWidget())))
    page = d.pages.currentWidget()
    bar = next(b for b in page.findChildren(QProgressBar))
    assert bar.isVisibleTo(page) and bar.value() == 100
    assert not _button(page, "Download positions").isEnabled()          # one download at a time
    d.close()                                                                # the download goes on
    d.deleteLater()
    qapp.processEvents()
    assert extracts.job().running() and not stopped
    d2, page = _page(qapp)                                                   # a window opened later shows where it is
    assert any("downloading 0.1 GB" in t for t in _texts(page))
    next(b for b in page.findChildren(QPushButton) if b.text() == "Cancel").click()
    _wait(qapp, lambda: not extracts.job().running()
          and any(t.startswith("Cancelled.") for t in _texts(d2.pages.currentWidget())))
    assert stopped == ["xp"]
    assert _button(d2.pages.currentWidget(), "Download spectra").isEnabled()
    d2.close()


def test_an_error_is_shown_in_plain_words(env, qapp, monkeypatch):
    def refuse(kind, progress=None, cancel=None, **kw):
        raise ge.DownloadError("Not enough free disk space in X: 1.5 GB needed")
    monkeypatch.setattr(ge, "download", refuse)
    d, page = _page(qapp)
    _button(page, "Download positions").click()
    _wait(qapp, lambda: not extracts.job().running() and
          "Not enough free disk space in X: 1.5 GB needed" in _texts(d.pages.currentWidget()))
    d.close()


def test_the_bright_stars_have_their_row_and_install_from_it(env, qapp, monkeypatch):
    """The third row: the Hipparcos list and the blind solver's index, what they are for, installed or not with their
    size, Install run by the one download job off the GUI thread (core/bright_stars.py, replaced here)."""
    from astro_gpu.core import bright_stars as bs
    from astro_gpu.core import skycache
    d, page = _page(qapp)
    shown = _gaia(_texts(page))
    assert "Bright stars" in shown and "Find an unknown field; add the brightest stars Gaia lacks" in shown
    assert shown.count("Not installed") == 3
    b = _button(page, "Install bright stars (0.7 GB)")
    assert b.isEnabled() and "Hipparcos and Tycho-2" in b.toolTip() and "25 MB" in b.toolTip()
    main, ran_in = threading.get_ident(), []

    def fake(progress=None, cancel=None):
        ran_in.append(threading.get_ident())
        progress(5, 14, "Tycho-2 index: building, 3 of 35")
        for f in bs.FILES:
            skycache.save_npz(skycache.cache_dir() / f, x=np.zeros(1000))
        progress(14, 14, "Done")
    monkeypatch.setattr(bs, "download", fake)
    monkeypatch.setattr(ge, "download", lambda *a, **k: pytest.fail("the Gaia download ran"))
    b.click()
    _wait(qapp, lambda: not extracts.job().running() and "Done." in _texts(d.pages.currentWidget()))
    assert ran_in and ran_in[0] != main
    page = d.pages.currentWidget()
    assert any(t.startswith("Installed · ") for t in _texts(page))
    b = _button(page, "Install bright stars")
    assert not b.isEnabled() and b.toolTip() == "Already installed."
    (skycache.cache_dir() / bs.FILES[2]).unlink()
    d.rebuild()
    qapp.processEvents()
    assert "Partly installed: 2 of 3 files" in _texts(d.pages.currentWidget())
    d.close()


def test_the_section_has_its_french(env, qapp):
    i18n.set_language("fr")
    d, page = _page(qapp)
    shown = _texts(page)
    assert "Extraits Gaia hors ligne" in shown and "Télécharger positions (1,1 Go)" in shown
    assert "Télécharger spectres (10,6 Go)" in shown and _gaia(shown).count("Non installé") == 3
    assert "Étoiles brillantes" in shown and "Installer étoiles brillantes (0,7 Go)" in shown
    from astro_gpu.core import bright_stars as bs
    assert d._bright_status_text(bs.Status("installed", 3, 3, 737_000_000)) == "Installé · 737 Mo"
    _tiny_astro(env / "nowhere" / "siril" / ASTRO_NAME)
    sirilcat.reset()
    d.rebuild()
    qapp.processEvents()
    assert any(t.startswith("Trouvé dans le dossier de données d'un autre programme") for t in _texts(d.pages.currentWidget()))
    assert not i18n.missing(), sorted(i18n.missing())
    # the download's own texts
    ge.translate, ge.number = i18n._, i18n.number
    try:
        assert ge.human(10.55e9) == "10,6 Go" and ge.human(5e8) == "500 Mo"
        for t in ("Reading the file list from Zenodo", ": checking", "A download is already running.", "Done"):
            assert i18n._(t) != t
        assert "Fichier 3 sur 48" in ge._t("File {i} of {n}", i=3, n=48)
    finally:
        ge.translate, ge.number = (lambda s: s), (lambda x, d=1: f"{x:.{d}f}")
    d.close()
