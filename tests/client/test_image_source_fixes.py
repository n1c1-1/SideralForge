"""GUI exploration round 2, the Image block's file: the "Already stretched image" template has something to run, the
kind follows the file chosen, a damaged FITS is said before the run, a file gone circles its block in red and Run is
refused. Offscreen on scratch projects; nothing is computed."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.stages.image import image_problem  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def fits_file(path: Path) -> Path:
    from astropy.io import fits
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.zeros((40, 40), np.float32)).writeto(path, overwrite=True)
    return path


def picture(path: Path) -> Path:
    import cv2
    cv2.imwrite(str(path), np.full((8, 8, 3), 128, np.uint8))
    return path


@pytest.fixture
def model(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    project2.save(p)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(120)
    m._sync()
    yield m
    m.close(save=False)
    w.stop()


def test_a_stretched_image_alone_has_something_to_run(model, tmp_path):
    """The "Already stretched image" template is Image -> Post: its Image block is computed, so it is not "no block:
    place one"; an Image without its file says what is missing (its file), not "add lights or image"."""
    png = picture(tmp_path / "done.png")
    q = model.add_image_workflow("Stretched", str(png), stretched=True)
    model.set_graph(q, [])
    w = model.p.workflow(q.pid)
    assert not appmod.no_block(w)
    lin = SimpleNamespace(blocks=[], source={"path": "x.fits", "stretched": False})
    assert not appmod.no_block(lin)                   # a linear image too: it goes to Post as it is
    assert appmod.images_only(model)
    model.add_graph_workflow("Frames", [])
    assert not appmod.images_only(model)              # a frames workflow: "add lights or image" stays right


def test_the_kind_follows_the_file_chosen(model, tmp_path):
    """A PNG or JPEG makes the Image stretched; a FITS or TIFF chosen after it makes it linear again; a FITS chosen
    first keeps the kind the user (or the template) set."""
    png, jpg = picture(tmp_path / "a.png"), picture(tmp_path / "b.jpg")
    fits = fits_file(tmp_path / "pile.fits")
    tif = tmp_path / "pile.tif"
    import tifffile
    tifffile.imwrite(tif, np.zeros((8, 8, 3), np.uint16))
    q = model.add_image_workflow("Image", "")
    model.set_graph(q, [])
    src = lambda: model.p.workflow(q.pid).source      # noqa: E731
    model.set_image_file(q, str(png))
    assert src()["stretched"] is True
    model.set_image_file(q, str(fits))
    assert src() == {"path": fits.as_posix(), "stretched": False}
    model.set_image_file(q, str(jpg))
    assert src()["stretched"] is True
    model.set_image_file(q, str(tif))
    assert src()["stretched"] is False
    model.set_image_file(q, stretched=True)            # chosen by the user: a FITS or TIFF after it keeps it
    model.set_image_file(q, str(fits))
    assert src()["stretched"] is True


def test_an_8_bit_tiff_is_a_stretched_picture(model, tmp_path):
    """Control round: an 8-bit TIFF (an already stretched photo) stayed "Linear" where the same PNG or JPEG was
    "Stretched": a stack is never stored on 8 bits. A 16 or 32-bit TIFF stays linear."""
    import tifffile

    from astro_gpu.stages.image import stretched_by_nature
    t8, t16 = tmp_path / "photo8.tif", tmp_path / "stack16.tiff"
    tifffile.imwrite(t8, np.zeros((8, 8, 3), np.uint8))
    tifffile.imwrite(t16, np.zeros((8, 8, 3), np.uint16))
    assert stretched_by_nature(t8) and not stretched_by_nature(t16) and stretched_by_nature(tmp_path / "x.png")
    q = model.add_image_workflow("Image", "")
    model.set_graph(q, [])
    src = lambda: model.p.workflow(q.pid).source      # noqa: E731
    model.set_image_file(q, str(t8))
    assert src()["stretched"] is True
    model.set_image_file(q, str(t16))                  # the kind the 8-bit TIFF gave goes with it
    assert src()["stretched"] is False


def test_a_damaged_file_s_tooltip_has_one_full_stop(qapp):
    """Control round: "the file is damaged.. Run is refused" (the reason's own full stop, then the sentence's)."""
    from astro_gpu.client.blocks import Node
    from astro_gpu.client.graph import BlockItem
    n = Node("image", [], broken="bad.fits cannot be read: the file is damaged.")
    tip = BlockItem(SimpleNamespace(), n, "Image", "", "pending", False, False, True).toolTip()
    assert "damaged. Run is refused" in tip and ".." not in tip


def test_a_damaged_fits_is_said(tmp_path):
    """A FITS whose header has no end, or shorter than its data, is said unreadable before the run (as a missing file
    is said "File not found")."""
    good = fits_file(tmp_path / "good.fits")
    assert image_problem(good) == ""
    bad = tmp_path / "bad.fits"
    bad.write_bytes(b"SIMPLE  =                    T" + b"\0" * 3000)
    assert image_problem(bad) == "{name} cannot be read: the file is damaged."
    cut = tmp_path / "cut.fits"
    cut.write_bytes(good.read_bytes()[:2880 + 100])     # its header whole, its data cut
    assert "damaged" in image_problem(cut)
    i18n.set_language("fr")
    try:
        assert i18n._("{name} cannot be read: the file is damaged.") != "{name} cannot be read: the file is damaged."
    finally:
        i18n.set_language("en")


def test_an_image_file_gone_circles_its_block_and_refuses_run(model, tmp_path):
    """The file moved after Save: the Image block is circled in red with the reason, and Run is refused with it."""
    fits = fits_file(tmp_path / "soon_gone.fits")
    q = model.add_image_workflow("Image", str(fits))
    model.set_graph(q, [])
    model.insert_block(model.view(q.pid), "gradient")
    q = model.view(q.pid)
    assert model.image_problem(q) == "" and not Graph(model).node_of(q, "image").broken
    assert not model.run_problems()
    model.worker.wait(30)                     # the estimates read the image's size meanwhile (Windows: no rename then)
    fits.rename(tmp_path / "elsewhere.fits")
    assert model.image_problem(q) == "File not found: soon_gone.fits"
    assert Graph(model).node_of(q, "image").broken == "File not found: soon_gone.fits"
    assert model.run_problems() == [(q.name, "File not found: soon_gone.fits")]
    (tmp_path / "damaged.fits").write_bytes(b"SIMPLE  =                    T" + b"\0" * 3000)
    model.set_image_file(q, str(tmp_path / "damaged.fits"))
    assert "damaged" in Graph(model).node_of(model.view(q.pid), "image").broken


def test_a_second_window_never_replaces_what_the_first_saved(model, tmp_path):
    """The night test pass: two windows on one project, the last to save replaced the other's work, silently. A
    window whose file changed on the disk since it read it writes nothing (Save says so)."""
    other = ProjectModel(project2.load(model.root), model.worker)
    try:
        model.set_description("from A")
        assert model.flush() and not model.changed_on_disk() and other.changed_on_disk()
        other.set_description("from B")
        assert other.changed_on_disk() and other.flush() is False and other.dirty
        assert project2.load(model.root).description == "from A"
        assert other.flush(force=True) and project2.load(model.root).description == "from B"
        model.set_description("A again")
        assert model.flush() is False                     # now A is the one behind
    finally:
        other.close(save=False)
