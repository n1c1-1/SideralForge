"""The Crop block's page (issue #79): the reference frame, the zone to stack drawn on it, the sky it leaves."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .i18n import _
from .services.inspect import SKY_BLOCK
from .viewer import StaticSource, Viewer

#: under this share of sky in the zone, the block says that the Gradient has little to work on
SKY_MIN = 0.25


class CropEdit(QWidget):
    def __init__(self, s, p, got: dict, parent=None):
        super().__init__(parent)
        self.s, self.p, self.got = s, p, got
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.viewer = Viewer(self)
        self.viewer.set_source(StaticSource(got["rgb"]))
        self.viewer.mode = "crop"
        self.viewer.crop_edit = True
        lay.addWidget(self.viewer, 1)
        row = QHBoxLayout()
        self.note = QLabel()
        self.note.setObjectName("dim")
        row.addWidget(self.note, 1)
        self.whole = QPushButton(_("Whole field"))
        self.whole.setToolTip(_("Stacks the whole field again."))
        self.whole.clicked.connect(self.clear)
        row.addWidget(self.whole)
        lay.addLayout(row)
        zone = s.param_value(p, "stack", "zone")
        if zone:
            self.viewer.crop_box = tuple(c / got["step"] for c in zone)
        self.viewer.cropChanged.connect(self.drawn)
        self.refresh()
        self.viewer.fit()

    def zone(self):
        """The zone drawn, in pixels of the reference frame; None: the whole field."""
        box = self.viewer.crop_box
        if not box:
            return None
        step, (W, H) = self.got["step"], self.got["size"]
        x0, y0, x1, y1 = (int(round(c * step)) for c in box)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        return None if (x0, y0, x1, y1) == (0, 0, W, H) or x1 <= x0 or y1 <= y0 else [x0, y0, x1, y1]

    def sky_share(self) -> float:
        """The share of the zone that is sky (the blocks of the reference frame near its darkest ones)."""
        sky = self.got["sky"]
        box = self.viewer.crop_box or (0, 0, *self.got["rgb"].shape[1::-1])
        x0, y0, x1, y1 = (int(c / SKY_BLOCK) for c in box)
        part = sky[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
        return float(np.mean(part)) if part.size else 0.0

    def refresh(self):
        z = self.zone()
        self.whole.setEnabled(z is not None)
        if z is None:
            self.note.setText(_("Draw the zone to stack on the frame."))
            self.note.setStyleSheet("")
            return
        share = self.sky_share()
        text = _("{w} x {h} px, {pct} % of the field, sky {sky} %").format(
            w=z[2] - z[0], h=z[3] - z[1], sky=round(100 * share),
            pct=round(100 * (z[2] - z[0]) * (z[3] - z[1]) / (self.got["size"][0] * self.got["size"][1])))
        if share < SKY_MIN:
            text += " · " + _("little sky left for the Gradient")
        self.note.setText(text)
        self.note.setStyleSheet("color: #e0a050;" if share < SKY_MIN else "")

    def store(self, zone):
        self.s.apply_change([self.p], {"kind": "params", "items": [("stack", "zone", zone)]})

    def drawn(self):
        self.refresh()
        self.store(self.zone())

    def clear(self):
        self.viewer.crop_box = None
        self.viewer.update()
        self.refresh()
        self.store(None)
