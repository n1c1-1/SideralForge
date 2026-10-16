"""Methods tab: the processing methods in plain words, tried side by side as variants and compared at 100 %.

No single processing suits every instrument and object. Each decision lists its options with what they do and what
was seen on real data; choosing another option either changes the processing shown (main or the active variant) or
creates a new variant, computed in its own folders next to main. 'Compare' renders the same zone of every processing
with the current rendering settings.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.methods import DECISIONS, Decision, Option, current_option, downstream, merged, option_params
from ..core.project import Project, stage_graph
from . import style
from .runner import stage_history_seconds
from .widgets import scroll


def _fmt_size(b: float) -> str:
    return f"{b / 2 ** 30:.1f} GB" if b >= 2 ** 30 else f"{b / 2 ** 20:.0f} MB"


def _fmt_dur(s: float) -> str:
    return f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


def _dir_size(d) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0


def _estimate(project: Project, stages: list[str]) -> tuple[float, int]:
    """Time (from this machine's history) and disk space (main's folders of the same stages) of computing them."""
    t = sum(stage_history_seconds(s) or 60.0 for s in stages)
    b = sum(_dir_size(project.work / s) for s in stages)
    return t, b


def _small(text: str, colour: str = style.FG2, italic: bool = False) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(f"color:{colour};" + ("font-style:italic;" if italic else ""))
    return lab


class DecisionCard(QGroupBox):
    """One decision: its question, its options, and what choosing another one would cost."""

    tryVariant = Signal(object, object, str)        # decision, option, file value
    applyHere = Signal(object, object, str)

    def __init__(self, decision: Decision, parent=None):
        super().__init__(decision.title, parent)
        self.decision = decision
        self.project: Project | None = None
        self.current: Option | None = None
        self.file_value = ""
        v = QVBoxLayout(self)
        v.addWidget(_small(decision.question, style.FG))
        self.group = QButtonGroup(self)
        self.radios: dict[str, QRadioButton] = {}
        for o in decision.options:
            rb = QRadioButton(o.label)
            self.radios[o.key] = rb
            self.group.addButton(rb)
            v.addWidget(rb)
            box = QVBoxLayout()
            box.setContentsMargins(24, 0, 0, 4)
            box.addWidget(_small(o.description))
            if o.evidence:
                box.addWidget(_small(o.evidence, italic=True))
            v.addLayout(box)
            rb.toggled.connect(lambda on, oo=o: on and self._picked(oo))
        if decision.note:
            v.addWidget(_small(decision.note, italic=True))
        self.custom = _small("Its parameters were set by hand in the Processing tab: none of these exactly.", style.WARN)
        v.addWidget(self.custom)
        self.action = QWidget()
        a = QHBoxLayout(self.action)
        a.setContentsMargins(0, 4, 0, 0)
        self.cost = _small("")
        a.addWidget(self.cost, 1)
        self.try_b = QPushButton("Try as a new variant")
        self.try_b.setToolTip("Compute this method in its own folders, next to the current one, to compare them")
        self.apply_b = QPushButton("Apply here")
        self.cancel_b = QPushButton("Cancel")
        for b in (self.try_b, self.apply_b, self.cancel_b):
            a.addWidget(b)
        v.addWidget(self.action)
        self.try_b.clicked.connect(lambda: self.tryVariant.emit(self.decision, self._chosen(), self.file_value))
        self.apply_b.clicked.connect(lambda: self.applyHere.emit(self.decision, self._chosen(), self.file_value))
        self.cancel_b.clicked.connect(self.reset)
        self.action.hide()

    def _chosen(self) -> Option | None:
        for o in self.decision.options:
            if self.radios[o.key].isChecked():
                return o
        return None

    def set_project(self, project: Project, shown: str):
        self.project = project
        self.apply_b.setText(f"Apply to '{shown}'")
        self.reset()

    def reset(self):
        self.current = current_option(self.decision, self.project.overrides) if self.project else None
        self.file_value = ""
        self.group.setExclusive(False)
        for o in self.decision.options:
            rb = self.radios[o.key]
            rb.blockSignals(True)
            rb.setChecked(self.current is not None and o.key == self.current.key)
            rb.setText(o.label + ("   (current)" if self.current is not None and o.key == self.current.key else ""))
            rb.blockSignals(False)
        self.group.setExclusive(True)
        self.custom.setVisible(self.current is None)
        self.action.hide()

    def _picked(self, o: Option):
        if self.current is not None and o.key == self.current.key:
            self.action.hide()
            return
        if o.kind == "file":
            f, _ = QFileDialog.getOpenFileName(self, "Star layer (.npy)", "", "NumPy array (*.npy)")
            if not f:
                self.reset()
                return
            self.file_value = f
        stages = downstream(self.decision.touched(), stage_graph())
        t, b = _estimate(self.project, stages)
        self.cost.setText(f"Computes again {len(stages)} stages (about {_fmt_dur(t)}); a variant takes about "
                          f"{_fmt_size(b)} on disk.")
        self.action.show()


class CompareWorker(QObject):
    """Renders the same zone of several processings, one at a time (one engine on the GPU at a time)."""

    overview = Signal(object)                   # uint8 (h, w, 3)
    zone = Signal(str, object)                  # name, uint8 (h, w, 3)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, root, device):
        super().__init__()
        self.root, self.device = root, device

    def _engine(self, variant: str):
        from ..render.engine import StarRemovalEngines
        p = Project(self.root)
        p.config.active_variant = variant
        col = p.stage_dir("color") / "manifest.json"
        white = json.loads(col.read_text())["params"].get("white_temp", 5778.0) if col.exists() else 5778.0
        return StarRemovalEngines(p.stage_dir("render_prep"), device=self.device, calibrated_white=white)

    @staticmethod
    def _free():
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @Slot(str, dict)
    def render_overview(self, variant, values):
        from ..render.settings import RenderSettings
        try:
            eng = self._engine(variant)
            lay = eng.render(RenderSettings(**values), 0.2)
            x0, x1, y0, y1 = eng.crop
            f = int(round(1 / lay.scale))
            img = (lay.final[:, y0 // f:y1 // f, x0 // f:x1 // f].clamp(0, 1) * 255 + 0.5).byte()
            self.overview.emit(img.permute(1, 2, 0).contiguous().cpu().numpy())
            del eng, lay
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"overview: {e}")
        self._free()

    @Slot(list, float, float, int)
    def render_zone(self, entries, fx, fy, size):
        """entries: (label, variant, rendering settings) - the same zone of each, at full resolution."""
        from ..render.settings import RenderSettings
        for label, v, values in entries:
            try:
                eng = self._engine(v)
                lay = eng.render(RenderSettings(**values), 1.0)
                x0, x1, y0, y1 = eng.crop
                cx, cy = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
                ys, xs = int(max(y0, min(cy - size / 2, y1 - size))), int(max(x0, min(cx - size * 0.75, x1 - 1.5 * size)))
                img = (lay.final[:, ys:ys + size, xs:xs + int(1.5 * size)].clamp(0, 1) * 255 + 0.5).byte()
                self.zone.emit(label, img.permute(1, 2, 0).contiguous().cpu().numpy())
                del eng, lay
            except Exception as e:  # noqa: BLE001
                self.failed.emit(f"{label}: {e}")
            self._free()
        self.finished.emit()


class ClickImage(QLabel):
    """The overview: a click chooses the zone to compare."""

    clicked = Signal(float, float)

    def mousePressEvent(self, e):
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return
        ox, oy = (self.width() - pm.width()) / 2, (self.height() - pm.height()) / 2
        fx, fy = (e.position().x() - ox) / pm.width(), (e.position().y() - oy) / pm.height()
        if 0 <= fx <= 1 and 0 <= fy <= 1:
            self.clicked.emit(fx, fy)


def _qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    return QImage(a.data, a.shape[1], a.shape[0], 3 * a.shape[1], QImage.Format_RGB888).copy()


class CompareDialog(QDialog):
    """The same zone of several views at 100 %: entries (label, processing variant, rendering settings)."""

    _overview = Signal(str, dict)
    _zone = Signal(list, float, float, int)

    def __init__(self, project: Project, entries: list[tuple[str, str, dict]], parent=None,
                 title: str = "Compare the processings"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1500, 950)
        self.entries, self.fx, self.fy = entries, 0.5, 0.5
        v = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(_small("To compare:", style.FG))
        self.checks: list[tuple[QCheckBox, tuple]] = []
        for e in entries:
            cb = QCheckBox(e[0])
            cb.setChecked(True)
            self.checks.append((cb, e))
            left.addWidget(cb)
        left.addWidget(_small("Click the overview to choose the zone; the crops are at 100 % (one image pixel per "
                              "screen pixel).", style.FG2))
        self.ov = ClickImage()
        self.ov.setAlignment(Qt.AlignCenter)
        self.ov.setMinimumSize(360, 260)
        self.ov.clicked.connect(self._pick)
        left.addWidget(self.ov)
        row = QHBoxLayout()
        row.addWidget(QLabel("Zone height (px):"))
        self.size = QSpinBox()
        self.size.setRange(200, 1600)
        self.size.setSingleStep(100)
        self.size.setValue(500)
        row.addWidget(self.size)
        left.addLayout(row)
        self.go = QPushButton("Render the zone")
        self.go.clicked.connect(self._render)
        left.addWidget(self.go)
        self.state = _small("")
        left.addWidget(self.state)
        left.addStretch(1)
        v.addLayout(left, 1)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        v.addWidget(scroll(self.grid_host), 4)
        self.n = 0
        self.thread = QThread()
        self.worker = CompareWorker(project.root, project.config.devices.get("main", "cuda:0"))
        self.worker.moveToThread(self.thread)
        self._overview.connect(self.worker.render_overview)
        self._zone.connect(self.worker.render_zone)
        self.worker.overview.connect(self._show_overview)
        self.worker.zone.connect(self._show_zone)
        self.worker.failed.connect(lambda m: self.state.setText(f"<span style='color:{style.ERR}'>{m}</span>"))
        self.worker.finished.connect(lambda: (self.go.setEnabled(True), self.state.setText("done")))
        self.thread.start()
        self.state.setText("rendering the overview...")
        self._overview.emit(entries[0][1], entries[0][2])

    def _show_overview(self, img):
        self._ov_img = img
        self._draw_overview()
        self.state.setText("")

    def _draw_overview(self):
        img = getattr(self, "_ov_img", None)
        if img is None:
            return
        pm = QPixmap.fromImage(_qimage(img)).scaled(self.ov.width(), self.ov.height(), Qt.KeepAspectRatio,
                                                    Qt.SmoothTransformation)
        from PySide6.QtGui import QColor, QPainter, QPen
        p = QPainter(pm)
        p.setPen(QPen(QColor(style.ACCENT), 2))
        p.drawEllipse(int(self.fx * pm.width()) - 6, int(self.fy * pm.height()) - 6, 12, 12)
        p.end()
        self.ov.setPixmap(pm)

    def _pick(self, fx, fy):
        self.fx, self.fy = fx, fy
        self._draw_overview()

    def _render(self):
        chosen = [e for cb, e in self.checks if cb.isChecked()]
        if not chosen:
            return
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.n = 0
        self.go.setEnabled(False)
        self.state.setText(f"rendering {len(chosen)} views at full resolution, one after the other...")
        self._zone.emit(chosen, self.fx, self.fy, self.size.value())

    def _show_zone(self, name, img):
        box = QVBoxLayout()
        w = QWidget()
        w.setLayout(box)
        box.addWidget(QLabel(f"<b>{name}</b>"))
        lab = QLabel()
        lab.setPixmap(QPixmap.fromImage(_qimage(img)))
        box.addWidget(lab)
        self.grid.addWidget(w, self.n // 2, self.n % 2)
        self.n += 1

    def closeEvent(self, e):
        self.thread.quit()
        self.thread.wait(10000)
        super().closeEvent(e)


class MethodsTab(QWidget):
    activeChanged = Signal()          # the processing shown changed (or its parameters)
    runRequested = Signal()           # compute the processing shown
    compareRequested = Signal()
    showRendering = Signal()

    def __init__(self, is_running, parent=None):
        super().__init__(parent)
        self.is_running = is_running
        self.project: Project | None = None
        self.pipeline = None
        v = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(QLabel("Processing shown:"))
        self.combo = QComboBox()
        self.combo.setMinimumWidth(260)
        self.combo.setToolTip("Main = the project's own processing; the others are variants computed next to it. "
                              "The processing shown is the one the Processing tab, Final adjustment and the exports "
                              "use.")
        self.combo.activated.connect(self._switch)
        head.addWidget(self.combo)
        self.status = QLabel("")
        head.addWidget(self.status, 1)
        self.compute_b = QPushButton("Compute")
        self.compute_b.setToolTip("Compute the stages of the processing shown that are not up to date")
        self.compare_b = QPushButton("Compare...")
        self.compare_b.setToolTip("Render the same zone of every computed processing at 100 %")
        self.promote_b = QPushButton("Make it main")
        self.promote_b.setToolTip("This variant becomes the project's processing; the former main is kept as a variant")
        self.delete_b = QPushButton("Delete variant")
        for b in (self.compute_b, self.compare_b, self.promote_b, self.delete_b):
            head.addWidget(b)
        v.addLayout(head)
        self.note = _small("")
        v.addWidget(self.note)
        body = QWidget()
        self.cards_box = QVBoxLayout(body)
        self.cards: list[DecisionCard] = []
        for d in DECISIONS:
            c = DecisionCard(d)
            c.tryVariant.connect(self._try_variant)
            c.applyHere.connect(self._apply_here)
            self.cards.append(c)
            self.cards_box.addWidget(c)
        render = QGroupBox("Chosen at rendering (instant, in Final adjustment)")
        rv = QVBoxLayout(render)
        rv.addWidget(_small("These switch instantly, without computing anything again: the star removal used for "
                            "the image (star model or learned network), the diffraction spikes as photographed or "
                            "drawn, the stars as photographed, the sharpening of the nebula, the glare of the "
                            "bright stars and how much of it stays with the nebula.", style.FG))
        go = QPushButton("Open Final adjustment")
        go.clicked.connect(self.showRendering.emit)
        rv.addWidget(go, 0, Qt.AlignLeft)
        self.cards_box.addWidget(render)
        self.cards_box.addStretch(1)
        v.addWidget(scroll(body), 1)
        self.compute_b.clicked.connect(self.runRequested.emit)
        self.compare_b.clicked.connect(self.compareRequested.emit)
        self.promote_b.clicked.connect(self._promote)
        self.delete_b.clicked.connect(self._delete)

    # ------------------------------------------------------------------ state
    def set_project(self, project: Project, pipeline):
        self.project, self.pipeline = project, pipeline
        self.refresh()

    def shown_name(self) -> str:
        return (self.project.config.active_variant or "main") if self.project else "main"

    def ready_processings(self) -> list[str]:
        """Main and the variants whose rendering preparation is done ('' = main)."""
        out = []
        cur = self.project.config.active_variant
        for v in [""] + list(self.project.config.variants):
            self.project.config.active_variant = v
            m = self.project.stage_dir("render_prep") / "manifest.json"
            if m.exists() and json.loads(m.read_text()).get("status") == "done":
                out.append(v)
        self.project.config.active_variant = cur
        return out

    def refresh(self):
        if not self.project:
            return
        p = self.project
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("main", "")
        for n in p.config.variants:
            self.combo.addItem(n, n)
        self.combo.setCurrentIndex(max(self.combo.findData(p.config.active_variant), 0))
        self.combo.blockSignals(False)
        running = self.is_running()
        self.combo.setEnabled(not running)
        v = p.config.active_variant
        sts = self.pipeline.status()
        own = p.variant_stages(v) if v else {s.name for s in sts}
        todo = [s.name for s in sts if s.name in own and s.state != "done"]
        if running:
            txt = f"<span style='color:{style.RUN}'>computing...</span>"
        elif todo:
            t = sum(stage_history_seconds(s) or 60.0 for s in todo)
            txt = f"<span style='color:{style.WARN}'>{len(todo)} stages to compute (about {_fmt_dur(t)})</span>"
        else:
            txt = f"<span style='color:{style.OK}'>up to date</span>"
        if v:
            txt += f" &middot; {_fmt_size(p.variant_size(v))} on disk"
        self.status.setText(txt)
        self.compute_b.setEnabled(bool(todo) and not running)
        self.promote_b.setVisible(bool(v))
        self.delete_b.setVisible(bool(v))
        self.promote_b.setEnabled(not running and not todo)
        self.delete_b.setEnabled(not running)
        self.compare_b.setEnabled(not running)
        if v:
            rec = p.config.variants[v]
            ch = "; ".join(f"{st}: " + ", ".join(f"{k} = {'automatic' if val is None else val}" for k, val in kv.items())
                           for st, kv in rec.get("params", {}).items())
            self.note.setText(f"Variant made {rec.get('created', '')[:16].replace('T', ' ')}. Differs from main by: "
                              f"{ch or 'nothing'}." + (f" {rec['note']}" if rec.get("note") else ""))
        else:
            self.note.setText("Main: the project's own processing. A method tried below can be computed as a variant "
                              "next to it, then compared and kept or deleted.")
        for c in self.cards:
            c.set_project(p, self.shown_name())
            c.setEnabled(not running)

    def _switch(self, i):
        name = self.combo.itemData(i)
        if self.is_running():
            self.refresh()
            return
        self.project.set_active_variant(name)
        self.activeChanged.emit()

    # ------------------------------------------------------------------ actions
    def _try_variant(self, d: Decision, o: Option | None, file_value: str):
        if o is None:
            return
        p = self.project
        default = f"{d.title}: {o.label}"
        name, ok = QInputDialog.getText(self, "New variant", "Name of the variant:", text=default)
        if not ok or not name.strip():
            return
        base = p.config.variants[p.config.active_variant]["params"] if p.config.active_variant else {}
        try:
            p.create_variant(name.strip(), merged(base, option_params(d, o, file_value)),
                             note=f"From '{self.shown_name()}' with '{d.title}' set to '{o.label}'.")
        except ValueError as e:
            QMessageBox.warning(self, "New variant", str(e))
            return
        # the new processing gets a view of its own (the current rendering settings), next to the others
        views = p.config.render.setdefault("views", {})
        if name.strip() not in views:
            views[name.strip()] = {"variant": name.strip(), "settings": dict(p.config.render.get("settings", {})),
                                   "created": datetime.now().isoformat(timespec="seconds"),
                                   "note": f"'{d.title}' set to '{o.label}'"}
        p.set_active_variant(name.strip())
        self.activeChanged.emit()
        self._offer_run()

    def _apply_here(self, d: Decision, o: Option | None, file_value: str):
        if o is None:
            return
        if QMessageBox.question(self, "Apply", f"Set '{d.title}' to '{o.label}' in '{self.shown_name()}'? Its stages "
                                "after the change are computed again (the previous result of this processing is "
                                "replaced; use 'Try as a new variant' to keep it).") != QMessageBox.Yes:
            return
        for st, kv in option_params(d, o, file_value).items():
            for k, val in kv.items():
                if val is None:
                    self.project.clear_override(st, k)
                else:
                    self.project.set_override(st, k, val)
        self.activeChanged.emit()
        self._offer_run()

    def _offer_run(self):
        sts = self.pipeline.status()
        todo = [s.name for s in sts if s.state != "done"]
        if not todo:
            return
        t = sum(stage_history_seconds(s) or 60.0 for s in todo)
        if QMessageBox.question(self, "Compute", f"Compute '{self.shown_name()}' now? {len(todo)} stages, about "
                                f"{_fmt_dur(t)}.") == QMessageBox.Yes:
            self.runRequested.emit()

    def _promote(self):
        v = self.project.config.active_variant
        if not v:
            return
        keep = f"Former main {datetime.now().strftime('%Y-%m-%d %H%M')}"
        if QMessageBox.question(self, "Make it main", f"'{v}' becomes the project's processing. The former main is "
                                f"kept as the variant '{keep}' (delete it later to free the disk).") != QMessageBox.Yes:
            return
        self.project.promote_variant(v, keep_main_as=keep)
        self.activeChanged.emit()

    def _delete(self):
        v = self.project.config.active_variant
        if not v:
            return
        if QMessageBox.question(self, "Delete variant", f"Delete '{v}' and everything it computed "
                                f"({_fmt_size(self.project.variant_size(v))})? Main is not touched.") != QMessageBox.Yes:
            return
        self.project.delete_variant(v)
        self.activeChanged.emit()
