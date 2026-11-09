"""Small points of the client: the Post slider's rail, its zero, the viewer's banner, the annotations (issues #69, #72,
#74)."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from astro_gpu.client.annotate import Annotations, Frame, Named, paint  # noqa: E402
from astro_gpu.client.post3 import ContSlider, offset_text  # noqa: E402
from astro_gpu.client.viewer import Viewer  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def test_a_click_on_the_rail_takes_the_slider_there_and_says_when_done(qapp):
    s = ContSlider("Sky", 0.0, 4.0, 2.0, 2.0)
    s.resize(400, 50)
    QWidget.show(s)
    QApplication.processEvents()
    done = []
    s.done.connect(done.append)
    QTest.mouseClick(s.s, Qt.LeftButton, Qt.NoModifier, QPoint(s.s.width() // 4, s.s.height() // 2))
    assert abs(s.value() - 1.0) < 0.15 and len(done) == 1
    s.close()


def test_a_click_on_the_handle_does_not_move_the_slider(qapp):
    s = ContSlider("Sky", 0.0, 4.0, 2.0, 2.0)
    s.resize(400, 50)
    QWidget.show(s)
    QApplication.processEvents()
    QTest.mouseClick(s.s, Qt.LeftButton, Qt.NoModifier, QPoint(s.s.width() // 2, s.s.height() // 2))
    assert abs(s.value() - 2.0) < 0.02
    s.close()


def test_a_value_that_rounds_to_zero_reads_zero(qapp):
    assert offset_text(2.04) == "0" and offset_text(1.96) == "0" and offset_text(2.0) == "0"
    assert offset_text(2.5).endswith(("0.5", "0,5"))


def test_the_viewer_says_its_banner_without_an_image(qapp):
    v = Viewer()
    v.resize(400, 300)
    empty = v.grab().toImage()
    v.banner = "Rendering…"
    assert v.grab().toImage() != empty


def test_annotations_stay_inside_the_picture(qapp):
    img = QImage(400, 300, QImage.Format_RGB32)
    img.fill(QColor(0, 0, 0))
    t = np.linspace(0, 2 * np.pi, 60)
    outline = np.c_[100 + 120 * np.cos(t), 150 + 120 * np.sin(t)]              # a circle across the picture's left edge
    sky = Annotations(objects=[Named(100.0, 150.0, outline, "IC 354")])
    p = QPainter(img)
    paint(p, sky, Frame(400, 300), {"grid": False, "objects": True}, QRectF(100, 50, 200, 200))
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(300, 400, 4)[:, :, :3].sum(2)
    inside = np.zeros((300, 400), bool)
    inside[50:251, 100:301] = True
    assert a[inside].max() > 0 and a[~inside].max() == 0


def test_a_crop_handle_keeps_the_box_inside_and_at_its_proportion(qapp):
    from types import SimpleNamespace
    v = Viewer()
    v.source = SimpleNamespace(size=lambda: (1000, 800))
    assert v._crop_moved("se", (100, 100, 300, 300), None, 2000, 500) == (100, 100, 1000, 500)
    assert v._crop_moved("move", (100, 100, 300, 300), (0, 0), -500, 20) == (0, 120, 200, 320)
    v.crop_ratio = 2.0
    x0, y0, x1, y1 = v._crop_moved("se", (100, 100, 300, 200), None, 500, 150)
    assert (x0, y0) == (100, 100) and abs((x1 - x0) / (y1 - y0) - 2.0) < 1e-6 and x1 == 500
    x0, y0, x1, y1 = v._crop_moved("se", (900, 700, 950, 725), None, 5000, 5000)
    assert x1 <= 1000 and y1 <= 800 and abs((x1 - x0) / (y1 - y0) - 2.0) < 1e-6


def test_the_count_says_the_frames_are_being_read_rather_than_zero(qapp):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QLabel

    from astro_gpu.client import sources
    cls = next(c for c in vars(sources).values() if isinstance(c, type) and "update_count" in vars(c))
    canvas = SimpleNamespace(message="", update=lambda: None)
    me = SimpleNamespace(_reading=True, items=[], sid=1, group="lights", held=lambda k: 3, count=QLabel(),
                         viewer=SimpleNamespace(canvas=canvas), excluded=lambda i: False)
    cls.update_count(me)
    assert me.count.text() == "Reading the frames…" and canvas.message == "Reading the frames…"
    me._reading = False
    cls.update_count(me)
    assert me.count.text().startswith("0 ") and canvas.message == ""
