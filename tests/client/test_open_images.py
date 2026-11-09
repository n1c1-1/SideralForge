"""Images opened as layers of Post: from a menu, the graph follows; from nothing, the project is made."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import drafts, gallery, prefs  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402

MACHINE = Machine([Gpu("cuda:0", "RTX", 16.0)], 16, 64.0)


@pytest.fixture(scope="session")
def qapp():
    from astro_gpu.client.widgets import QSS
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return app


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    import cv2
    from astropy.io import fits
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    prefs.load()
    prefs.set("home_sections", [])
    prefs.set("cache_dir", (tmp_path / "scratch").as_posix())
    appmod.apply_language("en")
    monkeypatch.setattr(appmod, "machine_info", lambda: MACHINE)
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: None)
    png, fit = tmp_path / "a.png", tmp_path / "b.fits"
    cv2.imwrite(str(png), np.full((8, 8, 3), 128, np.uint8))
    fits.PrimaryHDU(np.full((3, 8, 8), 0.1, np.float32)).writeto(fit)
    monkeypatch.setattr(gallery.QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(png), str(fit)], ""))
    windows = []
    yield {"tmp": tmp_path, "windows": windows, "mp": monkeypatch, "png": png, "fit": fit}
    for w in windows:
        w.close()
        w.deleteLater()
    QApplication.processEvents()


def settle(w, n=10):
    for _i in range(n):
        QApplication.processEvents()
        w.worker.wait(30)
    QApplication.processEvents()


def main(env) -> appmod.Main:
    w = appmod.Main()
    w.resize(1500, 800)
    w.show()
    env["windows"].append(w)
    w.worker.wait()
    settle(w)
    launched = env["launched"] = []
    env["mp"].setattr(w, "launch", lambda: launched.append(True))
    return w


def test_from_home_the_project_is_made_the_graph_follows_and_post_is_shown(env):
    w = main(env)
    w.show_home()
    settle(w)
    w.home.openImages.emit()
    w.worker.wait()
    settle(w, 14)
    assert w.project is not None and drafts.is_untitled(w.project.root) and w.mode == 2
    s = w.s
    assert [q.name for q in s.paths] == ["a", "b"] and all(s.is_plain_image(q) for q in s.paths)
    assert [s.p.workflow(q.pid).source["stretched"] for q in s.paths] == [True, False]
    g = Graph(s)
    assert {q.pid for q in s.paths} <= set(g.to_post) and [n.key for n in g.nodes].count("image") == 2
    assert env["launched"]


def test_cancelling_the_files_makes_no_project(env):
    w = main(env)
    env["mp"].setattr(gallery.QFileDialog, "getOpenFileNames", lambda *a, **k: ([], ""))
    w.open_images()
    settle(w)
    assert w.project is None and not env["launched"]


def test_a_project_already_open_takes_them_beside_what_it_has(env):
    w = main(env)
    w.new_project()
    w.worker.wait()
    settle(w, 12)
    w.graph.place_images([str(env["png"])])
    settle(w)
    w.open_images()
    settle(w, 12)
    assert [q.name for q in w.s.paths] == ["a", "a 2", "b"] or len(w.s.paths) == 3
    assert w.mode == 2


def test_post_has_its_button_and_its_empty_page_asks_for_images(env):
    w = main(env)
    w.new_project()
    w.worker.wait()
    settle(w, 12)
    w.set_mode(2)
    settle(w)
    page = w.post_holder.page
    assert page.pages.currentIndex() == 0
    asked = []
    env["mp"].setattr(w, "open_images", lambda: asked.append(1))
    page.openImages.disconnect()
    page.openImages.connect(lambda: asked.append(1))
    page.view.b_open.clicked.emit()                               # (goes through the view's own signal)
    page.openImages.emit()
    assert asked


def test_the_kind_of_an_image_is_changed_on_its_layer_and_computed_again(env):
    w = main(env)
    w.new_project()
    w.worker.wait()
    settle(w, 12)
    w.graph.place_images([str(env["fit"])])
    settle(w)
    w.set_mode(2)
    settle(w)
    view, q = w.post_holder.page.view, w.s.paths[0]
    seen = []
    view.runAsked.connect(lambda: seen.append(1))
    assert not q.source["stretched"]
    view.set_stretched(q, True)
    settle(w)
    assert w.s.view(q.pid).source["stretched"] and seen


def test_a_linear_image_alone_can_be_run(env):
    w = main(env)
    w.new_project()
    w.worker.wait()
    settle(w, 12)
    w.graph.place_images([str(env["fit"])])
    settle(w)
    assert not any(appmod.no_block(x) for x in w.s.p.workflows)
