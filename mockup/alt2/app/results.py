"""The result of a step: large at the centre (the frames chart, the stack, before / after...), its measurements and
the settings it ran with on the right. Used as the Result face of the Workflow view and in the detached window.
The decisions taken on the data, like the frame selection thresholds, are made here. A second workflow can be shown
side by side at the same step.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QTabBar,
                               QVBoxLayout, QWidget)

from charts import FramesChart, SpectralChart
from dialogs import run
from model import DATA, Session, hm
from settings import fmt, step_settings
from steps import BY_KEY, DIM, PHASES, Steps, fnum, kv
from viewer import StaticSource, Viewer
from widgets import OK, SHARED, WARN, clear, label, para

STATE_WORD = {"done": ("calculée", OK), "shared": ("calculée · partagée", SHARED), "pending": ("à calculer", DIM),
              "stale": ("à recalculer", WARN)}


def state_chip(state) -> QLabel:
    word, col = STATE_WORD[state]
    chip = QLabel(word)
    chip.setStyleSheet(f"color: {col}; border: 1px solid {col}; border-radius: 9px; padding: 1px 8px; "
                       "font-size: 9pt;")
    return chip


class ResultPane(QWidget):
    editStep = Signal()                                  # "Modifier les réglages"
    openSources = Signal(int)
    openPost = Signal()

    def __init__(self, session: Session, show_title=False, listen=False, parent=None):
        super().__init__(parent)
        self.s = session
        self.steps = Steps(session)
        self.show_title = show_title
        self.img_tab = 0
        self.cmp_pid = None
        self.picked = next((i for i, n in enumerate(session.data["frames"]["name"]) if "DSC02333" in n), None)
        self._syncing = False
        self._dirty = False
        self.sel_info = None
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(16, 8, 16, 10)
        mv.setSpacing(6)
        self.chead = QVBoxLayout()
        mv.addLayout(self.chead)
        self.center = QVBoxLayout()
        mv.addLayout(self.center, 1)
        h.addWidget(mid, 1)
        right = QWidget()
        right.setObjectName("panel")
        right.setFixedWidth(330)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(12, 12, 12, 10)
        rsc = QScrollArea()
        rsc.setWidgetResizable(True)
        rsc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        rh = QWidget()
        self.rl = QVBoxLayout(rh)
        self.rl.setContentsMargins(0, 0, 6, 0)
        rsc.setWidget(rh)
        rv.addWidget(rsc, 1)
        h.addWidget(right)
        if listen:                                       # the detached window follows the session by itself
            for sig in (session.pathsChanged, session.currentChanged, session.sourcesChanged):
                sig.connect(self.request)
            session.costChanged.connect(self.cost_changed)

    # ------------------------------------------------------------ refresh
    def request(self):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def showEvent(self, e):
        if self._dirty:
            self.refresh()
        super().showEvent(e)

    def cost_changed(self):
        if not self.isVisible():
            self._dirty = True
            return
        if self.s.sel_step == "selection" and self.sel_info is not None:
            self.build_right()
            self._update_selection_info()
        else:
            self.refresh()

    @property
    def path(self):
        return self.s.path

    def other(self):
        p = self.path
        return next((q for q in self.s.paths if q.pid == self.cmp_pid and p is not None and q.pid != p.pid), None)

    def refresh(self):
        self._dirty = False
        if self.path is None or BY_KEY.get(self.s.sel_step) is None:
            clear(self.chead)
            clear(self.center)
            clear(self.rl)
            return
        self.build_center()
        self.build_right()

    # ------------------------------------------------------------ centre
    def build_center(self):
        clear(self.chead)
        clear(self.center)
        self.sel_info = None
        p = self.path
        b = BY_KEY[self.s.sel_step]
        if not self.steps.included(p, b):
            return
        state = self.steps.state(p, b)
        if self.show_title:
            top = QHBoxLayout()
            top.addWidget(label(f"{self.steps.title(p, b)}", "h1"))
            top.addWidget(state_chip(state))
            top.addStretch(1)
            top.addWidget(label(f"« {p.name} » · {PHASES[b['phase']]}", "small"))
            self.chead.addLayout(top)
        if state == "pending":
            self.pending_state(p, b)
            return
        if state == "stale":
            self.chead.addWidget(self.stale_strip(p, b))
        if b["key"] == "selection":
            self.selection_center(p)
            return
        vis = self.steps.visuals(p, b["key"])
        if not vis:
            self.center.addWidget(label("Pas d'image pour cette étape.", "dim"))
            self.center.addStretch(1)
            return
        bar = QHBoxLayout()
        i = min(self.img_tab, len(vis) - 1)
        if len(vis) > 1:
            tb = QTabBar()
            tb.setDrawBase(False)
            tb.setExpanding(False)
            for _, cap in vis:
                tb.addTab(cap)
            tb.setCurrentIndex(i)
            tb.currentChanged.connect(self._img_tab)
            bar.addWidget(tb)
        else:
            bar.addWidget(label(vis[0][1], "small"))
        bar.addStretch(1)
        bar.addWidget(label("Comparer avec", "dim"))
        cmp_ = QComboBox()
        cmp_.setMinimumWidth(190)
        cmp_.addItem("—", None)
        for q in self.s.paths:
            if q.pid != p.pid:
                cmp_.addItem(q.name, q.pid)
        k = cmp_.findData(self.cmp_pid)
        cmp_.setCurrentIndex(max(k, 0))
        if k < 0:
            self.cmp_pid = None
        cmp_.setEnabled(len(self.s.paths) > 1)
        cmp_.currentIndexChanged.connect(lambda j, c=cmp_: self._cmp_changed(c.itemData(j)))
        self.cmp_box = cmp_
        bar.addWidget(cmp_)
        self.chead.addLayout(bar)
        q = self.other()
        if q is None:
            self.center.addWidget(self.visual(p, b, vis[i]), 1)
            return
        why = self.compare_note(p, q, b)
        if why:
            self.center.addWidget(label(why, "small", wrap=True))
            self.center.addWidget(self.visual(p, b, vis[i]), 1)
            return
        row = QHBoxLayout()
        cols = []
        for w in (p, q):
            v = QVBoxLayout()
            v.addWidget(label(f"« {w.name} »", "h2"))
            wv = self.steps.visuals(w, b["key"])
            if self.steps.state(w, b) == "pending":
                v.addWidget(label(f"Pas encore calculée dans « {w.name} ».", "dim"), 1, Qt.AlignCenter)
            else:
                wid = self.visual(w, b, wv[min(i, len(wv) - 1)])
                v.addWidget(wid, 1)
                cols.append(wid)
            row.addLayout(v, 1)
        self.center.addLayout(row, 1)
        vws = [c for c in cols if isinstance(c, Viewer)]
        if len(vws) == 2:
            vws[0].viewChanged.connect(lambda: self._sync(vws[0], vws[1]))
            vws[1].viewChanged.connect(lambda: self._sync(vws[1], vws[0]))

    def _cmp_changed(self, pid):
        self.cmp_pid = pid
        self.build_center()

    def compare_note(self, p, q, b) -> str:
        if b["phase"] < 4:
            return f"Étape commune à tous les workflows : même résultat dans « {q.name} »."
        if not self.steps.included(q, b):
            return f"Pas d'étape « {self.steps.title(p, b)} » dans « {q.name} »."
        st = self.steps.run_stages(p, b)
        mine = set(self.s.path_stages(q))
        if st and all(x in mine and self.s.stage_key(p, x) == self.s.stage_key(q, x) for x in st):
            return f"Même résultat dans « {q.name} » : étape calculée une seule fois pour les deux workflows."
        return ""

    def _sync(self, a: Viewer, b: Viewer):
        if self._syncing:
            return
        self._syncing = True
        try:
            if a.source and b.source and a.source.size() == b.source.size():
                b.zoom_to(a.scale, a.cx, a.cy)
        finally:
            self._syncing = False

    def _img_tab(self, i):
        self.img_tab = i
        self.build_center()

    def visual(self, p, b, vis):
        kind, cap = vis
        if kind == "chart:spectral":
            return SpectralChart(self.s.data["spectral"]["lam"], self.s.data["spectral"]["S"])
        vw = Viewer()
        vw.setMinimumHeight(300)
        vw.set_source(self.steps.source(kind))
        ov = self.steps.overlay(b["key"], kind)
        if ov:
            vw.overlays.append(ov)
        return vw

    def pending_state(self, p, b):
        items = self.s.pending_upto(p, self.steps.run_stages(p, b))
        sec = sum(self.s.seconds(x) for _, x in items)
        box = QVBoxLayout()
        box.setAlignment(Qt.AlignCenter)
        box.addWidget(label("Pas encore calculée", "h2"), 0, Qt.AlignHCenter)
        what = ("Le graphe des poses vient des mesures de l'alignement." if b["key"] == "selection" else
                "Le résultat de cette étape s'affichera ici, en grand.")
        names = self.steps.step_titles(p, [x for _, x in items])
        box.addWidget(para(what + f" Il faut calculer : {', '.join(names)}, soit {hm(sec)}."), 0, Qt.AlignHCenter)
        box.addSpacing(8)
        rb = QPushButton(f"Lancer jusqu'ici  ·  {hm(sec)}")
        rb.setObjectName("primary")
        rb.setEnabled(self.s.sources_loaded and bool(items))
        if not self.s.sources_loaded:
            rb.setToolTip("Ajoute d'abord des sources")
        rb.clicked.connect(lambda: run(self.s, items, self.window(), f"Lancer jusqu'à : {self.steps.title(p, b)}"))
        box.addWidget(rb, 0, Qt.AlignHCenter)
        box.addWidget(label("Ou essaie d'abord sur une zone de l'image : bouton « Mode test », en haut.", "small"),
                      0, Qt.AlignHCenter)
        self.center.addStretch(1)
        self.center.addLayout(box)
        self.center.addStretch(2)

    def stale_strip(self, p, b):
        f = QFrame()
        f.setObjectName("stale")
        f.setStyleSheet(f"QFrame#stale {{ background: #2a2316; border: 1px solid {WARN}; border-radius: 5px; }}"
                        " QLabel { background: transparent; }")
        h = QHBoxLayout(f)
        h.setContentsMargins(10, 5, 10, 5)
        h.addWidget(label("Résultat de l'ancien calcul : un réglage a changé depuis.", ""), 1)
        items = self.s.pending_upto(p, self.steps.run_stages(p, b))
        sec = sum(self.s.seconds(x) for _, x in items)
        rb = QPushButton(f"Recalculer jusqu'ici  ·  {hm(sec)}")
        rb.clicked.connect(lambda: run(self.s, items, self.window(), f"Recalculer jusqu'à : {self.steps.title(p, b)}"))
        h.addWidget(rb)
        return f

    # ---- the frames chart, large, and the frame under the cursor
    def selection_center(self, p):
        ch = FramesChart(self.s.data["frames"], 0.6, 2.0)
        ch.tmin, ch.fmax = self.s.thresholds_now()
        ch.sugg = self.s.suggested_thresholds()
        ch.hand = set(self.s.frames_excluded)
        ch.current = self.picked
        ch.setMinimumHeight(300)
        ch.changed.connect(lambda c=ch: self.s.set_selection(c.tmin, c.fmax, c.hand))
        ch.picked.connect(self._pick)
        self.chart = ch
        ch.setToolTip("Seuils : glisser · pose : clic · plusieurs poses : tracer un cadre")
        self.center.addWidget(ch, 3)
        self.sel_info = QHBoxLayout()
        self.center.addLayout(self.sel_info)
        bottom = QHBoxLayout()
        self.pick_view = Viewer()
        self.pick_view.setMinimumHeight(220)
        bottom.addWidget(self.pick_view, 1)
        self.pick_side = QVBoxLayout()
        side = QWidget()
        side.setFixedWidth(280)
        side.setLayout(self.pick_side)
        bottom.addWidget(side)
        self.center.addLayout(bottom, 2)
        self._update_selection_info()
        self._pick(self.picked)

    def _update_selection_info(self):
        if self.sel_info is None:
            return
        try:
            clear(self.sel_info)
        except RuntimeError:
            return
        n = len(self.s.data["frames"]["fwhm"])
        kept = self.s.kept_frames()
        t = label(f"<b>{kept} poses gardées sur {n}</b>, soit "
                  f"{fnum(kept * self.s.data['frames']['exposure_s'] / 3600)} h d'intégration.", "dim")
        t.setTextFormat(Qt.RichText)
        self.sel_info.addWidget(t)
        self.sel_info.addStretch(1)
        p = self.path
        if self.steps.state(p, BY_KEY["empilement"]) == "stale":
            items = self.s.pending_upto(p, ["stack"])
            sec = sum(self.s.seconds(x) for _, x in items)
            self.sel_info.addWidget(label("L'empilement est à recalculer pour tous les workflows.", "warn"))
            rb = QPushButton(f"Recalculer l'empilement  ·  {hm(sec)}")
            rb.clicked.connect(lambda: run(self.s, items, self.window(), "Recalculer l'empilement"))
            self.sel_info.addWidget(rb)

    def _pick(self, i):
        self.picked = i
        if hasattr(self, "chart"):
            try:
                self.chart.current = i
                self.chart.update()
            except RuntimeError:
                pass
        try:
            clear(self.pick_side)
        except (RuntimeError, AttributeError):
            return
        if i is None:
            self.pick_side.addWidget(label("Clique un point du graphe pour voir la pose brute.", "dim", wrap=True))
            self.pick_side.addStretch(1)
            return
        f = self.s.data["frames"]
        name = f["name"][i].rsplit(".", 1)[0]
        path = DATA / "raws" / "lights" / f"{name}.jpg"
        tr = 100 * f["transp"][i] / float(np.percentile(f["transp"], 90))
        ex = bool(self.s.excluded_mask()[i])
        self.pick_side.addWidget(label(f["name"][i], "h2"))
        self.pick_side.addWidget(kv([("Heure", f["time"][i]), ("Taille des étoiles", f"{fnum(f['fwhm'][i], 2)} px"),
                                     ("Transparence", f"{tr:.0f} % des meilleures"),
                                     ("État", "exclue" if ex else "gardée")], pad=10))
        b1 = QPushButton("Garder cette pose" if i in self.s.frames_excluded else "Exclure cette pose")
        b1.clicked.connect(lambda: (self.s.set_selection(hand=set(self.s.frames_excluded) ^ {i}), self._pick(i),
                                    self._redraw_chart()))
        self.pick_side.addWidget(b1, 0, Qt.AlignLeft)
        b2 = QPushButton("Ouvrir dans les sources  →")
        b2.setObjectName("link")
        b2.clicked.connect(lambda: self.openSources.emit(i))
        self.pick_side.addWidget(b2, 0, Qt.AlignLeft)
        self.pick_side.addStretch(1)
        if path.exists():
            self.pick_view.set_source(StaticSource(np.asarray(Image.open(path).convert("RGB"))))

    def _redraw_chart(self):
        try:
            self.chart.hand = set(self.s.frames_excluded)
            self.chart.update()
        except RuntimeError:
            pass

    # ------------------------------------------------------------ right
    def build_right(self):
        clear(self.rl)
        p = self.path
        b = BY_KEY[self.s.sel_step]
        if not self.steps.included(p, b):
            return
        state = self.steps.state(p, b)
        self.rl.addWidget(label(self.steps.state_sentence(p, b), "small", wrap=True))
        self.rl.addSpacing(6)
        self.rl.addWidget(label("Mesures", "h2"))
        facts = self.steps.facts(p, b["key"]) if state != "pending" or b["key"] == "post" else []
        if b["key"] == "post" and state == "pending":
            self.rl.addWidget(label("Curseurs que ce calque aura en post :", "small"))
        if facts:
            self.rl.addWidget(kv(facts, pad=10))
        else:
            self.rl.addWidget(label("Elles apparaîtront après le calcul.", "dim"))
        rows = []
        for _, rs in step_settings(self.s, p, b["key"]):
            for r in rs:
                if r["level"] != "normal":
                    continue
                txt = fmt(r["value"], r["type"], r.get("choices"))
                if r.get("unit") and r["type"] in ("int", "float"):
                    txt += f" {r['unit']}"
                if r["changed"]:
                    txt += f" <span style='color:{WARN}'>· modifié</span>"
                rows.append((r["label"], txt))
        if rows:
            self.rl.addSpacing(8)
            self.rl.addWidget(label("Réglages prévus" if state == "pending" else
                                    ("Réglages actuels" if state == "stale" else "Réglages utilisés"), "h2"))
            if state == "stale":
                self.rl.addWidget(label("Le résultat affiché vient des réglages d'avant.", "small", wrap=True))
            self.rl.addWidget(kv(rows, pad=10))
        self.rl.addSpacing(6)
        e = QPushButton("Voir les réglages  →")
        e.setObjectName("link")
        e.clicked.connect(self.editStep.emit)
        self.rl.addWidget(e, 0, Qt.AlignLeft)
        if b["key"] == "post" and p.rendered:
            o = QPushButton("Ouvrir en post  →")
            o.setObjectName("primary")
            o.clicked.connect(self.openPost.emit)
            self.rl.addWidget(o, 0, Qt.AlignLeft)
        self.rl.addStretch(1)


class DetachedResult(QWidget):
    """The result in its own window (second screen), following the chosen step."""
    closed = Signal()
    editStep = Signal()

    def __init__(self, session: Session, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Résultat · maquette alt2")
        self.setObjectName("root")
        self.setStyleSheet("QWidget#root { background: #15171c; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 6, 0, 0)
        self.pane = ResultPane(session, show_title=True, listen=True)
        self.pane.editStep.connect(self.editStep.emit)
        v.addWidget(self.pane)
        self.resize(1400, 900)
        self.pane.refresh()

    def closeEvent(self, e):
        self.closed.emit()
        super().closeEvent(e)
