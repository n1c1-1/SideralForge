"""A block whose data of the sky is not on this computer (the offline star catalogue not installed: core/sky_data.py)
is marked on the graph with the reason on hover, says it in its panel, and the run window lists it before the run;
the other blocks and workflows still run. Nothing is downloaded during a run (tests/test_no_download_in_run.py)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.graph import BlockItem  # noqa: E402
from astro_gpu.core import sirilcat, sky_data  # noqa: E402
from tests.client.test_fusion_sessions_client import done, open_page, qapp, settle  # noqa: E402, F401

NO_STARS = "the star catalogue is not installed on this computer"


@pytest.fixture
def no_catalogue(tmp_path, monkeypatch):
    """No offline extract anywhere, no cache of earlier downloads."""
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "userhome"))
    for v in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(v, str(tmp_path / "nowhere"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    sirilcat.reset()
    sky_data.reset()
    yield
    i18n.set_language("en")
    sirilcat.reset()
    sky_data.reset()


def _texts(root) -> list[str]:
    return [w.text() for w in root.findChildren(QLabel) if w.isVisibleTo(root) and w.text()]


def test_the_block_without_its_catalogue_is_marked_and_says_why(qapp, tmp_path, monkeypatch, no_catalogue):  # noqa: F811
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        q = m.paths[0]
        g = Graph(m)
        ast = g.node_of(q, "astrometrie")
        assert ast.missing == NO_STARS
        assert all(not n.missing for n in g.nodes if n is not ast)          # Stacking, Gradient...: they run
        item = next(it for it in page.graph.scene().items() if isinstance(it, BlockItem) and it.node.key == "astrometrie")
        tip = item.toolTip()
        assert f"Cannot run: {NO_STARS}. Install it: Settings › Catalogs & astrometry" in tip
        page.select(step="astrometrie")
        settle(w, page)
        assert f"Cannot run: {NO_STARS}" in _texts(page.panel)
        # the run window says it before the run, and the plan runs the rest
        plan = m.plan_all()
        assert plan.left_out == {q.pid: {"astrometry": NO_STARS}} and {"inventory", "stack"} <= {j.stage for j in
                                                                                                  plan.jobs}
        from astro_gpu.client.runs import RunDialog
        d = RunDialog(m, plan, "Run")
        assert f'"{q.name}": Astrometry and the blocks after it not run: {NO_STARS}. Install it: Settings › Catalogs ' \
               "& astrometry" in d.warnings()
        d.close()
        i18n.set_language("fr")
        assert m.sky_missing(q, "astrometrie") == "le catalogue d'étoiles n'est pas installé sur cet ordinateur"
        i18n.set_language("en")
        # installed meanwhile (Settings): the block can run, at once
        from tests import skyfield
        sky = skyfield.field(tmp_path / "sky", spectra=True)
        monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(sky["extracts"]))
        sirilcat.reset()
        m.sky_changed()
        QApplication.processEvents()
        assert Graph(m).node_of(q, "astrometrie").missing == "" and m.plan_all().left_out == {}
    finally:
        done(w, m, page)


def test_the_positions_without_the_spectra_mark_the_color_block(qapp, tmp_path, monkeypatch, no_catalogue):  # noqa: F811
    """The star positions installed alone (the spectra are the 10 GB part): Astrometry runs, the Color block, which
    calibrates on the spectra, is marked with its reason and offers the way to Settings."""
    from tests import skyfield
    sky = skyfield.field(tmp_path / "sky", spectra=False)
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(sky["extracts"]))
    sirilcat.reset()
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        q = m.paths[0]
        g = Graph(m)
        assert g.node_of(q, "couleur").missing == sky_data.NO_SPECTRA
        assert g.node_of(q, "astrometrie").missing == ""
        page.select(step="couleur")
        settle(w, page)
        assert f"Cannot run: {sky_data.NO_SPECTRA}" in _texts(page.panel)
        assert _button(page.panel, "Install catalogue").isEnabled()
    finally:
        done(w, m, page)


def _button(root, text):
    from PySide6.QtWidgets import QPushButton
    return next(b for b in root.findChildren(QPushButton) if b.text() == text and b.isVisibleTo(root))


def test_the_blocked_block_opens_the_catalogue_s_settings_page(qapp, tmp_path, monkeypatch, no_catalogue):  # noqa: F811
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        page.select(step="astrometrie")
        settle(w, page)
        b = _button(page.panel, "Install catalogue")
        assert "Settings › Catalogs & astrometry" in b.toolTip()
        asked = []
        page.open_settings = lambda pg=None: asked.append(pg)      # the main window's (never a dialog in a test)
        b.click()
        assert asked == ["catalogs"]
        page.select(step="empilement")                              # a block that needs no catalogue: no button
        settle(w, page)
        assert not [x for x in page.panel.findChildren(type(b)) if x.text() == "Install catalogue" and
                    x.isVisibleTo(page.panel)]
    finally:
        done(w, m, page)


def test_home_says_once_when_the_star_catalogue_is_not_installed(qapp, tmp_path, monkeypatch, no_catalogue):  # noqa: F811
    from astro_gpu.client import prefs
    from astro_gpu.client.home import HomeView
    from astro_gpu.client.services.worker import ApiWorker
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    worker = ApiWorker()
    home = HomeView(worker)
    try:
        home.resize(1200, 700)
        assert home.sky_box.isVisibleTo(home)
        assert home.sky_text.text() == "Star catalogue not installed: Astrometry and Color need it"
        asked = []
        home.open_settings = lambda pg=None: asked.append(pg)
        _button(home, "Install catalogue").click()
        assert asked == ["catalogs"]
        from tests import skyfield
        sky = skyfield.field(tmp_path / "sky", spectra=False)
        monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(sky["extracts"]))
        sirilcat.reset()
        home.sky_changed()
        assert not home.sky_box.isVisibleTo(home)
    finally:
        home.close()
        worker.stop()


def test_get_survey_fetches_it_outside_the_run_and_the_next_run_uses_it(qapp, tmp_path, monkeypatch,  # noqa: F811
                                                                         no_catalogue):
    """The survey of the field not on the disk: the run gave 'none'. The Gradient block's panel says it and Get survey
    fetches it (an action of the user, its progress shown), then the results made without it are computed again."""
    import numpy as np

    from astro_gpu.client.graph import GraphPage
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services import survey as survey_job
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2, runner2
    from astro_gpu.core.project import Target
    from astro_gpu.core.workflow import TEMPLATES, node
    from astro_gpu.stages import surveys
    from tests import skyfield
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    sky = skyfield.field(tmp_path / "sky", spectra=False)
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(sky["extracts"]))
    sirilcat.reset()
    p = project2.create(tmp_path / "P", "P", Target(**sky["target"]))
    wf = p.add_workflow("Image", TEMPLATES["Classic stack"])
    wf.source = {"path": str(sky["image"]), "stretched": False}
    p.set_blocks(wf.id, [node("astrometry"), node("gradient")])
    project2.save(p)
    plan = runner2.plan_all(p)
    assert runner2.run_plan(p.root, runner2.prepare_run(p, plan, "cpu").plan_file, echo=False) == "done"
    fetched = []

    def fake(wd, W, H, scale, which="auto", min_coverage=0.95, log=print, progress=None, fetch=False):
        assert fetch                                              # the explicit action: it may fetch
        for k in range(3):
            progress(k / 3, f"tile {k + 1} of 3")
        fetched.append((W, H))
        return np.zeros((H, W), np.float32), "nsns"
    monkeypatch.setattr(surveys, "survey_on_grid", fake)
    worker = ApiWorker()
    m = ProjectModel(project2.load(p.root), worker)
    assert worker.wait(120)
    m._sync()
    page = GraphPage(m)
    try:
        page.resize(1500, 800)
        page.select(step="fond")
        settle(worker, page)
        assert "H-alpha survey: not on this computer" in _texts(page.panel)
        get = _button(page.panel, "Get survey")
        assert "Nothing is downloaded during a run" in get.toolTip()
        j = survey_job.job()
        ended = []
        j.finished.connect(lambda pid, why: ended.append((pid, why)))
        get.click()
        import time
        t0 = time.monotonic()
        while not ended and time.monotonic() - t0 < 20:
            QApplication.processEvents()
        QApplication.processEvents()
        assert fetched == [(skyfield.W, skyfield.H)] and ended == [(wf.id, "")]
        again = runner2.plan_all(project2.load(p.root))
        assert {"surveys", "gradient", "render_simple"} <= {x.stage for x in again.jobs}       # computed again
        assert {"image", "astrometry"}.isdisjoint({x.stage for x in again.jobs})                # kept
        assert any((p.root / "store" / "_trash").glob("surveys-*-survey-*"))                    # never deleted
    finally:
        page.close()
        m.close(save=False)
        worker.stop()
