"""The Crop block (issue #79): a zone of the field, set before Stacking."""
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from astro_gpu.client import cropedit
from astro_gpu.client.cropedit import CropEdit
from astro_gpu.core import workflow as wf


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def frames(*names):
    return [wf.node(n) for n in names]


def test_the_crop_block_goes_between_selection_and_stacking():
    blocks = frames("calibration", "registration", "selection", "stacking")
    i = wf.insert_place(blocks, "frames", "crop")
    assert i == 3
    placed = blocks[:i] + [wf.node("crop")] + blocks[i:]
    assert wf.graph_problem(wf.with_output(placed, "frames"), "frames") is None
    assert wf.node_implied(blocks, "stack") == {"zone": None}            # the block taken off: the whole field
    assert "zone" not in wf.node_implied(placed, "stack")


class Session:
    def __init__(self, zone=None):
        self.zone, self.changes = zone, []

    def param_value(self, p, stage, name):
        return self.zone

    def apply_change(self, paths, change):
        self.changes.append(change)


def got(sky=True):
    return {"rgb": np.zeros((200, 300, 3), np.uint8), "step": 2, "size": (600, 400), "sky": np.full((6, 9), sky)}


def edit(qapp, session, data=None):
    e = CropEdit(session, object(), data or got())
    e.resize(600, 400)
    return e


def test_the_zone_drawn_is_stored_in_pixels_of_the_frame(qapp):
    s = Session()
    e = edit(qapp, s)
    assert e.zone() is None and not e.whole.isEnabled()
    e.viewer.crop_box = (30.0, 40.0, 130.0, 140.0)
    e.drawn()
    assert e.zone() == [60, 80, 260, 280] and e.whole.isEnabled()
    assert s.changes == [{"kind": "params", "items": [("stack", "zone", [60, 80, 260, 280])]}]
    e.clear()
    assert s.changes[-1]["items"] == [("stack", "zone", None)] and e.viewer.crop_box is None


def test_a_zone_stored_comes_back_and_the_whole_frame_is_no_zone(qapp):
    e = edit(qapp, Session(zone=[60, 80, 260, 280]))
    assert e.viewer.crop_box == (30.0, 40.0, 130.0, 140.0) and e.zone() == [60, 80, 260, 280]
    e.viewer.crop_box = (0.0, 0.0, 300.0, 200.0)
    assert e.zone() is None


def test_the_block_says_when_little_sky_is_left(qapp):
    e = edit(qapp, Session(zone=[60, 80, 260, 280]), got(sky=False))
    assert e.sky_share() == 0.0 and "Gradient" in e.note.text() and "e0a050" in e.note.styleSheet()
    e = edit(qapp, Session(zone=[60, 80, 260, 280]), got(sky=True))
    assert e.sky_share() == 1.0 and "Gradient" not in e.note.text() and cropedit.SKY_MIN > 0
