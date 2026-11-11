"""The bright stars of the whole sky (core/bright_stars.py): the Hipparcos list of the Catalogues step and the blind
solver's two indexes, installed by an explicit action, resumable, cancellable; and the pre-check of a run that would
need the blind solver (core/sky_data.py). The services are replaced by small fake lists: nothing reaches the network."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from astro_gpu.core import blindsolve, bright_stars, gaia_archive, sirilcat, sky_data, skycache
from tests import skyfield
from tests.test_no_download_in_run import no_socket, offline_sky  # noqa: F401 - fixtures


def _sky(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    return r.uniform(0, 360, n), np.degrees(np.arcsin(r.uniform(-1, 1, n)))


@pytest.fixture
def services(offline_sky, monkeypatch):  # noqa: F811
    """The three lists the installation asks for, counted; the disk check made small."""
    asked: list[str] = []

    def hip_all(cols, vmax, say=None):
        asked.append(f"hip_main<{vmax:g}")
        ra, dec = _sky(300, 1)
        out = {"HIP": np.arange(300, dtype=np.float64), "RAICRS": ra, "DEICRS": dec, "Vmag": np.linspace(1, vmax, 300),
               "B-V": np.full(300, 0.6), "pmRA": np.zeros(300), "pmDE": np.zeros(300)}
        return {k: out[k] for k in cols}

    def lists(name, n, seed):
        def f(vmax=None, say=None):
            asked.append(name)
            return _sky(n, seed)
        return f
    monkeypatch.setattr(gaia_archive, "hipparcos_all", hip_all)
    monkeypatch.setattr(blindsolve, "_hipparcos", lists("hip_quads", 400, 2))
    monkeypatch.setattr(blindsolve, "_tycho", lists("tycho", 600, 3))
    monkeypatch.setattr(bright_stars, "DISK_BYTES", 1000)
    sky_data.reset()
    yield asked
    sky_data.reset()


def test_the_three_files_are_built_from_the_lists_and_read_by_what_needs_them(services, no_socket):  # noqa: F811
    assert bright_stars.status().state == "missing" and not bright_stars.index_here()
    with pytest.raises(skycache.SkyUnavailable):
        blindsolve.SkyIndex("hip")
    seen = []
    bright_stars.download(lambda d, t, text: seen.append((d, t, text)))
    st = bright_stars.status()
    assert st.state == "installed" and st.files == 3 and st.size > 0
    assert sorted(services) == ["hip_main<9", "hip_quads", "tycho"] and no_socket == []
    assert seen[-1] == (seen[-1][1], seen[-1][1], "Done") and [d for d, _t, _x in seen] == sorted(d for d, _t, _x in seen)
    assert any("Tycho-2 index: building" in x for _d, _t, x in seen)
    # what the run reads: the supplement of the Catalogues step and the two indexes, from the cache only
    t = skycache.hipparcos(["HIP", "Vmag"], 9.0)
    assert len(t["HIP"]) == 300
    assert len(blindsolve.SkyIndex("hip").ra) == 400 and len(blindsolve.SkyIndex("tyc").ra) == 600
    assert not list(skycache.cache_dir().glob("stars_*.npz"))            # the lists kept only until their index
    bright_stars.download()                                              # again: nothing asked, nothing built
    assert len(services) == 3


def test_a_cancel_keeps_what_is_done_and_the_next_install_goes_on_from_there(services):
    cancel = threading.Event()

    def progress(done, total, text):
        if "Tycho-2 index: building, 3 of" in text:
            cancel.set()
    with pytest.raises(bright_stars.Cancelled):
        bright_stars.download(progress, cancel)
    st = bright_stars.status()
    assert st.state == "partial" and st.files == 2 and bright_stars.index_here()
    assert (skycache.cache_dir() / "stars_tyc.npz").exists()            # the Tycho-2 list is not asked again
    bright_stars.download()
    assert bright_stars.status().state == "installed" and services.count("tycho") == 1


def test_a_failing_service_is_said_in_plain_words(services, monkeypatch):
    def down(*a, **k):
        raise ConnectionError("no connection")
    monkeypatch.setattr(blindsolve, "_tycho", down)
    with pytest.raises(bright_stars.DownloadError, match="No service gave the star list"):
        bright_stars.download()
    assert bright_stars.status().files == 2


def test_the_offline_mode_installs_nothing(services, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")
    with pytest.raises(bright_stars.DownloadError, match="offline mode"):
        bright_stars.download()
    assert services == []


def test_a_field_of_unknown_position_needs_the_blind_solver_s_index(services, monkeypatch):
    """The stars installed, the target's position unknown: Astrometry would stop at once without the blind solver's
    index (or Asterion, which solves at any scale); a known position needs neither."""
    from astro_gpu.core import asterion
    monkeypatch.setattr(asterion, "status", lambda *a, **k: asterion.Status(False, why="not installed", missing=True))
    sky = skyfield.field(skycache.cache_dir().parent / "sky", spectra=False)
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(sky["extracts"]))
    sirilcat.reset()
    sky_data.reset()
    assert sky_data.missing("astrometry", None) == sky_data.NO_INDEX
    assert sky_data.missing("astrometry", (38.0, 61.0)) == ""
    monkeypatch.setattr(asterion, "status", lambda *a, **k: asterion.Status(True))
    sky_data.reset()
    assert sky_data.missing("astrometry", None) == ""                   # Asterion finds the field
    monkeypatch.setattr(asterion, "status", lambda *a, **k: asterion.Status(False, missing=True))
    bright_stars.download()
    sky_data.reset()
    assert sky_data.missing("astrometry", None) == ""
    assert sky_data.NEEDED_BY["astrometry"] == "astrometry"


def test_the_planner_leaves_out_an_astrometry_that_could_not_find_its_field(services, monkeypatch):
    """Every Gaia extract installed but no index and no target: the Astrometry and what follows are left out with the
    reason (the block is marked), the rest runs; the reason is said in French too."""
    from astro_gpu.core import asterion, project2, runner2
    from astro_gpu.core.project import Target
    from astro_gpu.core.workflow import TEMPLATES, node
    monkeypatch.setattr(asterion, "status", lambda *a, **k: asterion.Status(False, missing=True))
    root = skycache.cache_dir().parent
    sky = skyfield.field(root / "sky")
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(sky["extracts"]))
    sirilcat.reset()
    sky_data.reset()
    p = project2.create(root / "P", "P", Target())
    w = p.add_workflow("Image", TEMPLATES["Classic stack"])
    w.source = {"path": str(sky["image"]), "stretched": False}
    p.set_blocks(w.id, [node("astrometry"), node("color")])
    project2.save(p)
    assert runner2.plan_all(p).left_out == {w.id: {"astrometry": sky_data.NO_INDEX}}
    bright_stars.download()
    sky_data.reset()
    assert runner2.plan_all(p).left_out == {}
    from astro_gpu.client import i18n
    try:
        i18n.set_language("fr")
        assert i18n._(sky_data.NO_INDEX).startswith("la position de la cible est inconnue")
    finally:
        i18n.set_language("en")
