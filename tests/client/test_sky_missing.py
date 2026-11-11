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
        assert f"Cannot run on this computer: {NO_STARS}. Install it: Settings › Catalogs & astrometry" in tip
        page.select(step="astrometrie")
        settle(w, page)
        assert f"Cannot run on this computer: {NO_STARS}" in _texts(page.panel)
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

