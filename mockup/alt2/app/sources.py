"""Sources view: the signal and calibration frames of the project, their correspondences, and a raw frames browser
(grid, viewer with filmstrip and blink, sortable list) where frames are kept or excluded.

The thumbnails and viewer images are prepared from the raw files by build/prepare_raws.py (one stretch shared by
all the lights, so that a hazy frame looks darker when blinking).
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image
from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
                               QListView, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from model import DATA, Session
from viewer import StaticSource, Viewer
from widgets import OK, WARN, clear, hline, label, para

GROUPS = [("lights", "Poses", "Signal"), ("darks", "Darks", "Calibration"), ("flats", "Flats", "Calibration"),
          ("bias", "Offsets", "Calibration"), ("flat_darks", "Darks de flat", "Calibration")]


def fnum(x, d=1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def fexp(t) -> str:
    if t is None:
        return "?"
    return f"1/{round(1 / t)} s" if t < 1 else f"{t:g} s"


class NumItem(QTableWidgetItem):
    """A table cell sorted by its number, shown with a French decimal comma."""

    def __init__(self, value, text=None):
        super().__init__(text if text is not None else str(value))
        self.v = value

    def __lt__(self, other):
        return self.v < getattr(other, "v", 0)


class GroupRow(QFrame):
    clicked = Signal(str)

    def __init__(self, key, title, sub, selected, empty, parent=None):
        super().__init__(parent)
        self.key = key
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(("QFrame { background: #2b3446; border-radius: 4px; }" if selected else
                            "QFrame { background: transparent; border-radius: 4px; } QFrame:hover "
                            "{ background: #232831; }") + " QLabel { background: transparent; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 5, 8, 5)
        v.setSpacing(0)
        a = QLabel(title)
        a.setStyleSheet("font-weight: 600;" + (" color: #6b7280;" if empty else ""))
        b = QLabel(sub)
        b.setObjectName("small")
        b.setWordWrap(True)
        v.addWidget(a)
        v.addWidget(b)

    def mousePressEvent(self, e):
        self.clicked.emit(self.key)


class ImportDialog(QDialog):
    """Simulated reading of a folder: the real file names of the project go by, then what was found."""

    def __init__(self, raws: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajouter un dossier")
        self.setMinimumWidth(560)
        g = raws["groups"]
        self.names = [(k, f["file"]) for k in ("lights", "darks", "flats") for f in g.get(k, {}).get("frames", [])]
        self.folder = g.get("lights", {}).get("folder", "")
        v = QVBoxLayout(self)
        v.addWidget(label("Ajouter un dossier", "h1"))
        v.addWidget(label(f"Dossier choisi : {self.folder}", "dim", wrap=True))
        self.bar = QProgressBar()
        self.bar.setRange(0, len(self.names))
        self.bar.setTextVisible(False)
        v.addWidget(self.bar)
        self.cur = label("", "small")
        v.addWidget(self.cur)
        self.found = label("", "", wrap=True)
        self.found.setTextFormat(Qt.RichText)
        v.addWidget(self.found)
        v.addWidget(label("Maquette : la lecture est simulée ; ce sont les fichiers du projet Heart.", "small"))
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Annuler")
        cancel.clicked.connect(self.reject)
        self.ok = QPushButton("Ajouter au projet")
        self.ok.setObjectName("primary")
        self.ok.setEnabled(False)
        self.ok.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(self.ok)
        v.addLayout(row)
        self.i = 0
        self.t = QTimer(self)
        self.t.timeout.connect(self._tick)
        self.t.start(15)

    def _tick(self):
        self.i = min(self.i + 12, len(self.names))
        self.bar.setValue(self.i)
        if self.i < len(self.names):
            k, f = self.names[self.i]
            self.cur.setText(f"Lecture : {f}")
            return
        self.t.stop()
        n = {k: sum(1 for kk, _ in self.names if kk == k) for k in ("lights", "darks", "flats")}
        self.cur.setText(f"{len(self.names)} fichiers lus.")
        self.found.setText(f"<b>Trouvé</b> : {n['lights']} poses, {n['darks']} darks et {n['flats']} flats, rangés "
                           "par type d'après la durée de pose, l'ISO et le niveau. Ni offsets ni darks de flat.")
        self.ok.setEnabled(True)

    def finish_now(self):
        self.i = len(self.names)
        self._tick()


class SourcesView(QWidget):
    goChain = Signal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        p = DATA / "raws.json"
        self.raws = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"groups": {}}
        self.inv = session.data["inventory"]
        self.group = "lights"
        self.mode = 1                                      # 0 grid, 1 viewer, 2 list
        self.cur = 0
        self.sort = "heure"
        self.excl = {"darks": set(), "flats": set()}       # calibration frames excluded by hand
        self.pick_zone = False
        tr = np.array(session.data["frames"]["transp"])
        self.p90 = float(np.percentile(tr, 90))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages)
        self.pages.addWidget(self._empty_page())
        main = QWidget()
        self.pages.addWidget(main)
        root = QHBoxLayout(main)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        # ---- left: groups and correspondences
        left = QWidget()
        left.setObjectName("panel")
        left.setFixedWidth(300)
        self.lv = QVBoxLayout(left)
        self.lv.setContentsMargins(10, 12, 10, 10)
        root.addWidget(left)
        # ---- centre: browser
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 6, 10, 6)
        self.mode_btns = []
        g = QButtonGroup(self)
        for i, t in enumerate(("Grille", "Visionneuse", "Liste")):
            b = QPushButton(t)
            b.setCheckable(True)
            b.setObjectName("toggle")
            b.clicked.connect(lambda _=False, k=i: self.set_mode(k))
            g.addButton(b)
            bar.addWidget(b)
            self.mode_btns.append(b)
        bar.addSpacing(12)
        bar.addWidget(label("Trier par", "dim"))
        self.sort_cb = QComboBox()
        self.sort_cb.currentIndexChanged.connect(self._sort_changed)
        bar.addWidget(self.sort_cb)
        self.blink = QPushButton("▶  Clignoter")
        self.blink.setCheckable(True)
        self.blink.setObjectName("toggle")
        self.blink.setToolTip("Fait défiler les poses à 5 images par seconde dans la visionneuse (barre d'espace)")
        self.blink.toggled.connect(self._blink)
        bar.addWidget(self.blink)
        bar.addStretch(1)
        self.count = label("", "dim")
        bar.addWidget(self.count)
        mv.addLayout(bar)
        self.info = label("", "small")
        self.info.setContentsMargins(12, 0, 12, 4)
        mv.addWidget(self.info)
        self.stack = QStackedWidget()
        # grid
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setIconSize(QSize(176, 118))
        self.grid.setGridSize(QSize(196, 160))
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grid.itemDoubleClicked.connect(lambda it: (self._goto(it.data(Qt.UserRole)), self.set_mode(1)))
        self.grid.itemClicked.connect(lambda it: self._goto(it.data(Qt.UserRole)))
        self.stack.addWidget(self.grid)
        # viewer + filmstrip
        vw = QWidget()
        vv = QVBoxLayout(vw)
        vv.setContentsMargins(0, 0, 0, 0)
        vv.setSpacing(0)
        self.viewer = Viewer()
        self.viewer.overlays.append(self._overlay)
        self.viewer.rectDone.connect(self._zone_done)
        vv.addWidget(self.viewer, 1)
        nav = QHBoxLayout()
        nav.setContentsMargins(10, 4, 10, 4)
        self.prev_b, self.next_b = QPushButton("◀"), QPushButton("▶")
        self.prev_b.setToolTip("Pose précédente (←)")
        self.next_b.setToolTip("Pose suivante (→)")
        self.prev_b.clicked.connect(lambda: self.step(-1))
        self.next_b.clicked.connect(lambda: self.step(1))
        self.keep_b = QPushButton()
        self.keep_b.setToolTip("Garder ou exclure cette pose (touche X)")
        self.keep_b.clicked.connect(self.toggle_keep)
        self.zone_b = QPushButton("Tracer la zone de test")
        self.zone_b.setToolTip("Le mode test ne calculera que cette zone")
        self.zone_b.clicked.connect(self.start_zone)
        nav.addWidget(self.prev_b)
        nav.addWidget(self.next_b)
        nav.addWidget(self.keep_b)
        nav.addStretch(1)
        nav.addWidget(self.zone_b)
        vv.addLayout(nav)
        self.strip = QListWidget()
        self.strip.setViewMode(QListView.IconMode)
        self.strip.setFlow(QListView.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setIconSize(QSize(120, 80))
        self.strip.setGridSize(QSize(128, 100))
        self.strip.setFixedHeight(118)
        self.strip.setMovement(QListView.Static)
        self.strip.setUniformItemSizes(True)
        self.strip.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.strip.itemClicked.connect(lambda it: self._goto(it.data(Qt.UserRole)))
        vv.addWidget(self.strip)
        self.stack.addWidget(vw)
        # list
        self.table = QTableWidget()
        self.table.setStyleSheet("QTableWidget { background: #1c1f26; gridline-color: #2a2f38; } "
                                 "QHeaderView::section { background: #23272f; border: none; padding: 4px; }")
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.cellDoubleClicked.connect(lambda r, c: (self._goto(self.table.item(r, 0).data(Qt.UserRole)),
                                                           self.set_mode(1)))
        self.table.cellClicked.connect(lambda r, c: self._table_click(r, c))
        self.stack.addWidget(self.table)
        mv.addWidget(self.stack, 1)
        root.addWidget(mid, 1)
        # ---- right: properties
        right = QWidget()
        right.setObjectName("panel")
        right.setFixedWidth(320)
        self.rv = QVBoxLayout(right)
        self.rv.setContentsMargins(12, 12, 12, 12)
        root.addWidget(right)

        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.step(1))
        for seq, fn in (("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1)), ("X", self.toggle_keep),
                        ("Space", self.blink.toggle)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        session.testChanged.connect(self.viewer.update)
        session.sourcesChanged.connect(self._sources_changed)
        self.select_group("lights")
        self.pages.setCurrentIndex(1 if session.sources_loaded else 0)

    def _empty_page(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setAlignment(Qt.AlignCenter)
        v.addWidget(label("Pas encore de sources", "h1"), 0, Qt.AlignHCenter)
        v.addWidget(para("Ajoute le dossier de tes poses. Les darks, flats et offsets sont cherchés à côté et rangés "
                         "par type ; leurs correspondances avec les poses (durée, ISO, niveaux) sont vérifiées et dites "
                         "en clair. Tu pourras ensuite parcourir chaque brute et l'exclure."), 0, Qt.AlignHCenter)
        v.addSpacing(10)
        b = QPushButton("Ajouter un dossier…")
        b.setObjectName("primary")
        b.clicked.connect(self.import_folder)
        v.addWidget(b, 0, Qt.AlignHCenter)
        return w

    def import_folder(self):
        d = ImportDialog(self.raws, self.window())
        self._import = d
        if d.exec():
            self.s.load_sources()

    def _sources_changed(self):
        self.pages.setCurrentIndex(1 if self.s.sources_loaded else 0)
        self.select_group("lights")

    # ------------------------------------------------------------ data
    def frames(self, key=None):
        return self.raws["groups"].get(key or self.group, {}).get("frames", [])

    def order(self):
        fr = self.frames()
        idx = list(range(len(fr)))
        if self.sort == "etoiles":
            idx.sort(key=lambda i: fr[i].get("fwhm", 0))
        elif self.sort == "transp":
            idx.sort(key=lambda i: fr[i].get("transp", 0))
        elif self.sort == "niveau":
            idx.sort(key=lambda i: fr[i].get("level", 0))
        return idx

    def black(self) -> float:
        return float(self.raws["groups"].get(self.group, {}).get("black", 0))

    def excluded(self, i) -> bool:
        f = self.frames()[i]
        if self.group == "lights":
            return f.get("idx") in self.s.frames_excluded
        return i in self.excl.get(self.group, set())

    def file(self, i, thumb=False) -> str:
        f = self.frames()[i]
        return str(DATA / "raws" / self.group / (f["name"] + ("_t" if thumb else "") + ".jpg"))

    # ------------------------------------------------------------ left panel
    def build_left(self):
        clear(self.lv)
        section = None
        for key, title, sec in GROUPS:
            if sec != section:
                section = sec
                if sec == "Calibration":
                    self.lv.addSpacing(8)
                self.lv.addWidget(label(sec, "h2"))
            grp = self.raws["groups"].get(key)
            n = grp["count"] if grp else 0
            if key == "lights":
                sub = (f"{n} poses de {fnum(self.inv['exposure_s'], 0)} s · ISO {int(self.inv['iso_gain'])} · nuit "
                       f"du {self.inv['first_time'][8:10]}/{self.inv['first_time'][5:7]}")
            elif n:
                sub = f"{n} images · {grp['folder']}"
            else:
                sub = "aucun · facultatif"
            row = GroupRow(key, f"{title} · {n}" if n else title, sub, key == self.group, not n)
            row.clicked.connect(self.select_group)
            self.lv.addWidget(row)
            if key == "lights":
                b = QPushButton("+ Ajouter des poses (autre nuit)…")
                b.setObjectName("link")
                b.clicked.connect(self._add)
                self.lv.addWidget(b)
        self.lv.addSpacing(10)
        self.lv.addWidget(hline())
        self.lv.addWidget(label("Correspondances", "h2"))
        for ok, text in self.checks():
            row = QHBoxLayout()
            ic = QLabel("✓" if ok is True else ("!" if ok is False else "i"))
            ic.setFixedWidth(14)
            ic.setStyleSheet(f"color: {OK if ok is True else (WARN if ok is False else '#8b93a1')}; font-weight: 700;")
            row.addWidget(ic, 0, Qt.AlignTop)
            t = label(text, "small", wrap=True)
            row.addWidget(t, 1)
            self.lv.addLayout(row)
        self.lv.addStretch(1)

    def checks(self):
        g = self.raws["groups"]
        out = []
        lg = g.get("lights", {})
        if "darks" in g:
            d = g["darks"]
            same = d.get("exposure") == lg.get("exposure") and d.get("iso") == lg.get("iso")
            out.append((same, f"Darks : pose {fexp(d.get('exposure'))} et ISO {d.get('iso', '?')}, "
                              + ("comme les poses." if same else "différents des poses.")))
            lv = float(np.median([f["level"] for f in d["frames"]]))
            out.append((True, f"Darks : niveau médian {fnum(d['black'] + lv, 0)} ADU, pour un noir nominal de "
                              f"{fnum(d['black'], 0)} ADU."))
        if "flats" in g:
            pct = float(np.median([f["level_pct"] for f in g["flats"]["frames"]]))
            out.append((None, f"Flats : {g['flats']['count']} images, exposées à {fnum(pct, 0)} % de la saturation, "
                              f"pose {fexp(g['flats'].get('exposure'))}."))
        out.append((None, f"Ni offsets ni darks de flat : les flats sont corrigés du niveau noir de l'appareil "
                          f"({int(g.get('lights', {}).get('black', 512))})."))
        n_ex = len(self.s.frames_excluded) + sum(len(v) for v in self.excl.values())
        if n_ex:
            out.append((False, f"{n_ex} image(s) exclue(s) à la main : l'empilement est à recalculer."))
        return out

    def _add(self):
        QMessageBox.information(self, "Maquette", "Dans le vrai client : choisir un dossier ; ses poses sont lues, "
                                "regroupées par nuit et appareil, et les darks et flats correspondants sont proposés.")

    def select_group(self, key):
        grp = self.raws["groups"].get(key)
        if not grp:
            QMessageBox.information(self, "Maquette", "Pas d'images dans ce groupe pour le projet Heart. Dans le vrai "
                                    "client : « Ajouter un dossier… ».")
            return
        self.group = key
        self.cur = 0
        self.blink.setChecked(False)
        self.sort_cb.blockSignals(True)
        self.sort_cb.clear()
        opts = [("heure", "heure")]
        if key == "lights":
            opts += [("etoiles", "taille des étoiles"), ("transp", "transparence")]
        opts.append(("niveau", "niveau"))
        for k, t in opts:
            self.sort_cb.addItem(t, k)
        self.sort = "heure"
        self.sort_cb.blockSignals(False)
        self.zone_b.setVisible(key == "lights")
        self.build_left()
        self.fill()
        self.set_mode(self.mode)

    # ------------------------------------------------------------ browser
    def _item_text(self, i):
        f = self.frames()[i]
        if self.group == "lights":
            t = f"{f['name']}\n{fnum(f.get('fwhm', 0))} px · {100 * f.get('transp', 0) / self.p90:.0f} %"
        elif self.group == "flats":
            t = f"{f['name']}\n{fnum(f['level_pct'], 0)} % de la saturation"
        else:
            t = f"{f['name']}\n{fnum(self.black() + f['level'], 0)} ADU"
        return ("✕ " if self.excluded(i) else "") + t

    def fill(self):
        idx = self.order()
        for w, thumb in ((self.grid, True), (self.strip, True)):
            w.clear()
            for i in idx:
                it = QListWidgetItem(QIcon(self.file(i, thumb)), self._item_text(i) if w is self.grid else "")
                it.setData(Qt.UserRole, i)
                it.setToolTip(self._item_text(i).replace("\n", " · "))
                if self.excluded(i):
                    it.setForeground(QColor("#e06c6c"))
                w.addItem(it)
        self.fill_table(idx)
        self.update_count()

    def fill_table(self, idx):
        fr = self.frames()
        if self.group == "lights":
            cols = ["Fichier", "Heure", "Étoiles (px)", "Transparence (%)", "Poids", "Niveau (ADU)", "Gardée"]
        elif self.group == "flats":
            cols = ["Fichier", "Date", "Niveau (% saturation)", "Gardé"]
        else:
            cols = ["Fichier", "Date", "Niveau (ADU)", "Bruit (ADU)", "Gardé"]
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(idx))
        for r, i in enumerate(idx):
            f = fr[i]
            keep = "non" if self.excluded(i) else "oui"
            if self.group == "lights":
                cells = [QTableWidgetItem(f["file"]), QTableWidgetItem(f.get("time", "")),
                         NumItem(f.get("fwhm", 0), fnum(f.get("fwhm", 0), 2)),
                         NumItem(f.get("transp", 0), f"{100 * f.get('transp', 0) / self.p90:.0f}"),
                         NumItem(f.get("weight", 0), fnum(f.get("weight", 0), 2)),
                         NumItem(f["level"], fnum(f["level"])), QTableWidgetItem(keep)]
            elif self.group == "flats":
                cells = [QTableWidgetItem(f["file"]), QTableWidgetItem(f["mtime"]),
                         NumItem(f["level_pct"], fnum(f["level_pct"])), QTableWidgetItem(keep)]
            else:
                cells = [QTableWidgetItem(f["file"]), QTableWidgetItem(f["mtime"]),
                         NumItem(f["level"], fnum(self.black() + f["level"], 0)),
                         NumItem(f["noise"], fnum(f["noise"], 2)), QTableWidgetItem(keep)]
            cells[0].setData(Qt.UserRole, i)
            for c, it in enumerate(cells):
                if keep == "non":
                    it.setForeground(QColor("#e06c6c"))
                self.table.setItem(r, c, it)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def _table_click(self, r, c):
        i = self.table.item(r, 0).data(Qt.UserRole)
        self._goto(i)
        if c == self.table.columnCount() - 1:
            self.toggle_keep()

    def update_count(self):
        fr = self.frames()
        n_ex = sum(self.excluded(i) for i in range(len(fr)))
        noun = {"lights": "poses", "darks": "darks", "flats": "flats"}.get(self.group, "images")
        self.count.setText(f"{len(fr)} {noun} · {n_ex} exclue{'s' if n_ex > 1 else ''}")

    def _sort_changed(self, k):
        self.sort = self.sort_cb.itemData(k) or "heure"
        self.fill()
        self._goto(self.cur)

    def set_mode(self, k):
        self.mode = k
        self.mode_btns[k].setChecked(True)
        self.stack.setCurrentIndex(k)
        self._goto(self.cur)

    def step(self, d):
        idx = self.order()
        if not idx:
            return
        pos = idx.index(self.cur) if self.cur in idx else 0
        self._goto(idx[(pos + d) % len(idx)])

    def _goto(self, i):
        fr = self.frames()
        if not fr:
            return
        self.cur = int(i)
        f = fr[self.cur]
        if self.mode == 1:
            src = StaticSource(np.asarray(Image.open(self.file(self.cur)).convert("RGB")))
            self.viewer.set_source(src, keep_view=True)
        for w in (self.grid, self.strip):
            for r in range(w.count()):
                if w.item(r).data(Qt.UserRole) == self.cur:
                    w.setCurrentRow(r)
                    w.scrollToItem(w.item(r))
                    break
        if self.group == "lights":
            t = (f"{f['file']} · {f.get('time', '')} · {fnum(self.inv['exposure_s'], 0)} s · ISO "
                 f"{int(self.inv['iso_gain'])} · étoiles {fnum(f.get('fwhm', 0), 2)} px · transparence "
                 f"{100 * f.get('transp', 0) / self.p90:.0f} % des meilleures · poids {fnum(f.get('weight', 0), 2)}")
        elif self.group == "flats":
            t = f"{f['file']} · {f['mtime']} · niveau {fnum(f['level_pct'])} % de la saturation"
        else:
            t = (f"{f['file']} · {f['mtime']} · niveau {fnum(self.black() + f['level'], 0)} ADU · bruit "
                 f"{fnum(f['noise'], 2)} ADU")
        self.info.setText(t + (" · EXCLUE" if self.excluded(self.cur) else ""))
        self.keep_b.setText("Garder cette pose" if self.excluded(self.cur) else "Exclure cette pose  (X)")
        self.build_right()

    def toggle_keep(self):
        fr = self.frames()
        if not fr:
            return
        if self.group == "lights":
            k = fr[self.cur].get("idx")
            if k is None:
                return
            hand = set(self.s.frames_excluded)
            hand ^= {k}
            self.s.set_selection(None, None, hand)
        else:
            self.excl.setdefault(self.group, set()).symmetric_difference_update({self.cur})
            ex = {g: sorted(v) for g, v in self.excl.items() if v}
            self.s.set_trunk_param("calibrate", "excluded", tuple(sorted((g, tuple(v)) for g, v in ex.items())))
        r = self.grid.currentRow()
        for w in (self.grid, self.strip):
            for j in range(w.count()):
                if w.item(j).data(Qt.UserRole) == self.cur:
                    it = w.item(j)
                    it.setForeground(QColor("#e06c6c") if self.excluded(self.cur) else QColor("#d8dce4"))
                    if w is self.grid:
                        it.setText(self._item_text(self.cur))
        self.grid.setCurrentRow(r)
        self.fill_table(self.order())
        self.update_count()
        self.build_left()
        self._goto(self.cur)

    def _blink(self, on):
        if on:
            self.set_mode(1)
            self.timer.start(200)
        else:
            self.timer.stop()

    # ------------------------------------------------------------ right panel
    def build_right(self):
        clear(self.rv)
        g = self.raws["groups"].get(self.group, {})
        inv = self.inv
        self.rv.addWidget(label("Propriétés", "h2"))
        rows = []
        if self.group == "lights":
            rows = [("Appareil", inv["camera"]), ("Capteur", f"{inv['width']} × {inv['height']} px, matrice "
                                                             f"{inv['cfa']}"),
                    ("Pose", f"{fnum(inv['exposure_s'], 0)} s"), ("ISO", f"{int(inv['iso_gain'])}"),
                    ("Première pose", inv["first_time"].replace("T", " ")[:16]),
                    ("Dernière pose", inv["last_time"].replace("T", " ")[:16]),
                    ("Intégration", f"{fnum(inv['integration_h'])} h ({g.get('count')} × "
                                    f"{fnum(inv['exposure_s'], 0)} s)"),
                    ("Noir / blanc", f"{int(inv['black_level'])} / {int(inv['white_level'])} ADU")]
        elif g:
            rows = [("Images", str(g["count"])), ("Pose", fexp(g.get("exposure"))), ("ISO", str(g.get("iso", "?"))),
                    ("Prises le", g["frames"][0]["mtime"][:10])]
        rows.append(("Dossier", g.get("folder", "")))
        self.rv.addWidget(self._kv(rows))
        self.rv.addSpacing(8)
        self.rv.addWidget(hline())
        fr = self.frames()
        if fr:
            f = fr[self.cur]
            self.rv.addWidget(label(f"Image affichée : {f['file']}", "h2"))
            rows = [("Niveau médian", f"{fnum(f['level'])} ADU au-dessus du noir" if self.group != "darks" else
                     f"{fnum(self.black() + f['level'], 0)} ADU (noir nominal {fnum(self.black(), 0)})")]
            if self.group == "lights" and "fwhm" in f:
                rows += [("Heure", f.get("time", "")), ("Taille des étoiles", f"{fnum(f['fwhm'], 2)} px"),
                         ("Transparence", f"{100 * f['transp'] / self.p90:.0f} % des meilleures poses"),
                         ("Poids dans la pile", fnum(f["weight"], 2))]
            elif self.group == "flats":
                rows.append(("Exposition", f"{fnum(f['level_pct'])} % de la saturation"))
            else:
                rows.append(("Bruit", f"{fnum(f['noise'], 2)} ADU"))
            rows.append(("État", "exclue" if self.excluded(self.cur) else "gardée"))
            self.rv.addWidget(self._kv(rows))
        if self.group == "lights":
            self.rv.addSpacing(8)
            self.rv.addWidget(label("La taille des étoiles et la transparence sont mesurées par l'alignement. Pour "
                                    "exclure par seuils plutôt qu'une à une :", "small", wrap=True))
            b = QPushButton("Sélection des poses par seuils  →")
            b.setObjectName("link")
            b.clicked.connect(lambda: self.goChain.emit("selection"))
            self.rv.addWidget(b)
        self.rv.addStretch(1)

    @staticmethod
    def _kv(rows) -> QLabel:
        html = "<table cellspacing='0' cellpadding='2'>" + "".join(
            f"<tr><td style='color:#8b93a1; padding-right:10px'>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>"
        lb = QLabel(html)
        lb.setTextFormat(Qt.RichText)
        lb.setWordWrap(True)
        return lb

    # ------------------------------------------------------------ test zone
    def start_zone(self):
        self.select_group("lights") if self.group != "lights" else None
        self.set_mode(1)
        self.pick_zone = True
        self.viewer.mode = "rect"
        self.viewer.banner = "Trace la zone de test sur la pose"
        self.viewer.update()

    def _scale(self):
        g = self.raws["groups"]["lights"]
        return self.s.data["grid"][0] / g["view"][0]

    def _zone_done(self, x0, y0, x1, y1):
        if not self.pick_zone:
            return
        k = self._scale()
        self.s.test_zone = (x0 * k, y0 * k, x1 * k, y1 * k)
        self.pick_zone = False
        self.viewer.mode = "pan"
        self.viewer.banner = ""
        if not self.s.test_on:
            self.s.test_on = True
        self.s.testChanged.emit()

    def _overlay(self, p: QPainter, v: Viewer):
        if not (self.s.test_on and self.group == "lights") or self.pick_zone:
            return
        k = self._scale()
        zx0, zy0, zx1, zy1 = self.s.test_zone
        r = QRectF(v.img_to_w(zx0 / k, zy0 / k), v.img_to_w(zx1 / k, zy1 / k))
        p.setPen(QPen(QColor(WARN), 2, Qt.DashLine))
        p.setBrush(QColor(230, 162, 60, 30))
        p.drawRect(r)
        p.setPen(QColor(WARN))
        p.drawText(r.topLeft() + type(r.topLeft())(6, -6), f"Zone de test · {100 * self.s.zone_fraction():.0f} % de "
                                                           "l'image")
