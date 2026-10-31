"""Mockup alt2 of the astro_gpu heavy client: three views (Sources, Workflow with its Settings and Result faces,
Post); each computed workflow becomes a layer in Post. A demo menu switches between a blank, a half run and a run project.

    C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe mockup/alt2/app/main.py

Nothing is computed for real: see LISEZMOI.md for what is real and what is simulated.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtCore import QLocale, Qt  # noqa: E402
from PySide6.QtGui import QActionGroup, QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFrame,  # noqa: E402
                               QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from model import DATA, SCENARIOS, Session, hm  # noqa: E402
from post import PostPage  # noqa: E402
from sources import SourcesView  # noqa: E402
from steps import fnum  # noqa: E402
from widgets import QSS, WARN, label  # noqa: E402
from workflow import WorkflowView  # noqa: E402

HELP = """
<h2>Maquette alt2 v5 · client lourd</h2>
<p>Une maquette, pas le logiciel : <b>aucun calcul n'est lancé</b>. Les images, mesures, durées et réglages sont
ceux de ton projet Heart, lu sans rien modifier ; ce que ferait chaque action est montré par des aperçus préparés à
l'avance avec le vrai moteur.</p>
<p>Le <b>menu Démo</b>, en haut à droite, passe d'un projet vierge à un projet à moitié calculé, puis à un projet
calculé. Il n'existe que dans la maquette.</p>
<h3>Réel ou simulé</h3>
<p><b>Réel</b> : les brutes et leurs mesures, les images et mesures de chaque étape, les durées mesurées sur ton PC,
les 101 réglages du moteur avec la raison des valeurs suggérées ; les calques Complet et Complet + retrait appris et
chaque position de leurs curseurs ; les masques.<br>
<b>Approximé</b> : les calques Pile classique et Classique + débruitage, plusieurs curseurs bougés ensemble, le
mélange des calques.<br>
<b>Simulé</b> : les calculs, la lecture des dossiers, le mode test, l'export ; les réglages « à créer dans le
moteur ».</p>
<p>Raccourcis : Ctrl+1 à Ctrl+3 les vues · Tab réglages / résultat · sources ← → X espace · post C avant/après,
M masque, F ajuster, 1 zoom 100 %, Ctrl+Z / Ctrl+Y.</p>
"""


class TestBar(QFrame):
    def __init__(self, session: Session, main, parent=None):
        super().__init__(parent)
        self.s = session
        self.main = main
        self.setStyleSheet(f"QFrame {{ background: #2a2316; border-bottom: 1px solid {WARN}; }} "
                           "QLabel { background: transparent; }")
        h = QHBoxLayout(self)
        h.setContentsMargins(12, 6, 12, 6)
        t = QLabel("MODE TEST")
        t.setStyleSheet(f"color: {WARN}; font-weight: 700;")
        h.addWidget(t)
        self.zone = QLabel()
        h.addWidget(self.zone)
        b = QPushButton("Tracer la zone…")
        b.clicked.connect(main.pick_zone)
        h.addWidget(b)
        h.addSpacing(12)
        h.addWidget(label("Poses", "dim"))
        self.frames = QComboBox()
        self.frames.addItems(["toutes", "une sur 2", "une sur 4"])
        self.frames.currentIndexChanged.connect(lambda i: self._set(frames=[1.0, 0.5, 0.25][i]))
        h.addWidget(self.frames)
        self.short = QCheckBox("Raccourcir le retrait appris (÷4, aperçu moins fidèle)")
        self.short.toggled.connect(lambda v: self._set(short=v))
        h.addWidget(self.short)
        h.addStretch(1)
        self.est = QLabel()
        self.est.setTextFormat(Qt.RichText)
        h.addWidget(self.est)
        self.d = QPushButton("Détail")
        self.d.clicked.connect(self.detail)
        h.addWidget(self.d)
        for sig in (session.testChanged, session.currentChanged, session.pathsChanged):
            sig.connect(self.update_text)
        self.update_text()

    def _set(self, frames=None, short=None):
        if frames is not None:
            self.s.test_frames = frames
        if short is not None:
            self.s.test_short_learned = short
        self.s.testChanged.emit()

    def update_text(self):
        a = self.s.zone_fraction()
        self.zone.setText(f"Zone : {fnum(100 * a, 0)} % de l'image")
        p = self.s.path
        self.d.setEnabled(p is not None)
        if p is None:
            self.est.setText("Crée d'abord un workflow.")
            return
        rows = self.s.test_estimate(p)
        full, test = sum(r[1] for r in rows), sum(r[2] for r in rows)
        self.est.setText(f"« {p.name} » depuis les poses brutes : <b>{hm(test)}</b> au lieu de {hm(full)}")

    def detail(self):
        p = self.s.path
        rows = self.s.test_estimate(p)
        d = QDialog(self)
        d.setWindowTitle("Mode test : détail de l'estimation")
        v = QVBoxLayout(d)
        v.addWidget(label(f"Workflow « {p.name} », calculé depuis les poses brutes en mode test", "h2"))
        tb = QTableWidget(len(rows), 4)
        tb.setHorizontalHeaderLabels(["Étape", "Complet", "Test", "Comment"])
        tb.verticalHeader().setVisible(False)
        tb.setStyleSheet("QTableWidget { background: #1c1f26; gridline-color: #2a2f38; } QHeaderView::section "
                         "{ background: #23272f; border: none; padding: 4px; }")
        for i, (s, full, t, how) in enumerate(rows):
            for j, val in enumerate((self.s.title(s), hm(full), hm(t), how)):
                tb.setItem(i, j, QTableWidgetItem(val))
        tb.resizeColumnsToContents()
        tb.horizontalHeader().setStretchLastSection(True)
        v.addWidget(tb)
        v.addWidget(label("Estimation de la maquette. Le résultat est un aperçu : proche du résultat final, pas "
                          "identique (les réglages suggérés dépendent du bruit mesuré, donc du nombre de poses).",
                          "small", wrap=True))
        d.resize(900, 560)
        d.exec()


class Main(QMainWindow):
    def __init__(self, scenario="vierge"):
        super().__init__()
        self.s = None
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QFrame()
        bar.setObjectName("panel")
        bar.setStyleSheet("QFrame#panel { border-bottom: 1px solid #2a2f38; }")
        h = QHBoxLayout(bar)
        h.setContentsMargins(12, 0, 12, 0)
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 11pt;")
        h.addWidget(self.title)
        h.addSpacing(30)
        self.view_btns = []
        g = QButtonGroup(self)
        for i, name in enumerate(("1  Sources", "2  Workflow", "3  Post")):
            b = QPushButton(name)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=i: self.show_view(k))
            g.addButton(b)
            h.addWidget(b)
            self.view_btns.append(b)
        h.addStretch(1)
        self.demo = QPushButton()
        self.demo.setToolTip("Maquette seulement : passer d'un projet vierge à un projet calculé")
        self.demo.setStyleSheet("QPushButton { border: 1px dashed #6b7280; color: #aab2bf; }")
        m = QMenu(self.demo)
        ag = QActionGroup(self)
        self.demo_actions = {}
        for key, name in SCENARIOS.items():
            a = m.addAction(name)
            a.setCheckable(True)
            ag.addAction(a)
            a.triggered.connect(lambda _=False, k=key: self.set_scenario(k))
            self.demo_actions[key] = a
        self.demo.setMenu(m)
        h.addWidget(self.demo)
        h.addSpacing(10)
        self.bt_test = QPushButton("Mode test")
        self.bt_test.setObjectName("toggle")
        self.bt_test.setCheckable(True)
        self.bt_test.setToolTip("Essayer le workflow sur une zone de l'image avant le calcul complet")
        self.bt_run = QPushButton("Lancer")
        self.bt_run.setObjectName("primary")
        hb = QPushButton("?")
        hb.setFixedWidth(30)
        hb.setToolTip("Aide de la maquette")
        for w in (self.bt_test, self.bt_run, hb):
            h.addWidget(w)
        bar.setFixedHeight(46)
        v.addWidget(bar)
        self.test_host = QVBoxLayout()
        v.addLayout(self.test_host)
        self.views = QStackedWidget()
        v.addWidget(self.views, 1)
        self.status = QLabel()
        self.status.setObjectName("small")
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(label("Maquette : aucun calcul réel  ", "small"))
        self.statusBar().setStyleSheet("QStatusBar { background: #1c1f26; border-top: 1px solid #2a2f38; }")
        self.bt_test.toggled.connect(self.toggle_test)
        self.bt_run.clicked.connect(self.run)
        hb.clicked.connect(self.help)
        for seq, i in (("Ctrl+1", 0), ("Ctrl+2", 1), ("Ctrl+3", 2)):
            QShortcut(QKeySequence(seq), self).activated.connect(lambda i=i: self.show_view(i))
        self.set_scenario(scenario)

    # ------------------------------------------------------------ demo projects
    def set_scenario(self, key):
        """Builds the views again on a new session: blank, half run or run project."""
        cur = self.views.currentIndex() if self.s is not None else -1
        if getattr(self, "wf", None) is not None and self.wf.detached is not None:
            self.wf.detached.close()
        while self.views.count():
            w = self.views.widget(0)
            self.views.removeWidget(w)
            w.deleteLater()
        if hasattr(self, "testbar"):
            self.testbar.deleteLater()
        self.s = Session(key)
        s = self.s
        self.demo_actions[key].setChecked(True)
        self.demo.setText(f"Démo : {SCENARIOS[key].lower()}  ▾ ")
        self.title.setText(f"<b>{s.data['project']['name']}</b>  ·  {s.data['project']['target']}")
        self.setWindowTitle(f"astro_gpu · maquette alt2 · {s.data['project']['name']}")
        self.testbar = TestBar(s, self)
        self.testbar.setVisible(False)
        self.test_host.addWidget(self.testbar)
        self.bt_test.blockSignals(True)
        self.bt_test.setChecked(False)
        self.bt_test.blockSignals(False)
        self.src = SourcesView(s)
        self.wf = WorkflowView(s)
        self.post = PostPage(s)
        for w in (self.src, self.wf, self.post):
            self.views.addWidget(w)
        self.src.goChain.connect(self.go_selection)
        self.wf.openSources.connect(self.open_sources)
        self.wf.openPost.connect(lambda: self.show_view(2))
        self.post.goWorkflow.connect(lambda: self.show_view(1))
        self.post.view.goProcessing.connect(self.go_workflow_step)
        self.post.view.goResults.connect(self.go_results)
        for sig in (s.pathsChanged, s.costChanged, s.currentChanged, s.sourcesChanged):
            sig.connect(self.update_status)
        self.show_view(0 if key == "vierge" else (min(cur, 2) if cur >= 0 else 1))
        self.update_status()

    # ------------------------------------------------------------ navigation
    def show_view(self, i):
        self.views.setCurrentIndex(i)
        self.view_btns[i].setChecked(True)
        if i == 2:
            self.post.refresh()

    def go_workflow_step(self, pid, step):
        self.s.select(index=next(i for i, p in enumerate(self.s.paths) if p.pid == pid), step=step)
        self.show_view(1)
        self.wf.set_face(0)

    def go_results(self, pid):
        self.s.select(index=next(i for i, p in enumerate(self.s.paths) if p.pid == pid), step="post")
        self.show_view(1)

    def go_selection(self, key):
        if self.s.path is not None:
            self.s.select(step=key)
        self.show_view(1)

    def open_sources(self, frame):
        self.show_view(0)
        if frame >= 0 and self.s.sources_loaded:
            if self.src.group != "lights":
                self.src.select_group("lights")
            i = next((j for j, f in enumerate(self.src.frames()) if f.get("idx") == frame), None)
            if i is not None:
                self.src.set_mode(1)
                self.src._goto(i)

    def toggle_test(self, on):
        self.s.test_on = on
        self.testbar.setVisible(on)
        self.s.testChanged.emit()

    def pick_zone(self):
        if not self.s.sources_loaded:
            self.show_view(0)
            return
        self.show_view(0)
        self.src.start_zone()

    def update_status(self):
        s = self.s
        pend = s.all_pending() if s.paths else []
        sec = sum(s.seconds(st) for _, st in pend)
        n = len(s.paths)
        what = f"{n} workflow{'s' if n > 1 else ''}"
        self.bt_run.setText(f"Lancer le projet : {what} · {hm(sec)}" if pend else f"Lancer le projet : {what}")
        ok = s.sources_loaded and bool(s.paths)
        self.bt_run.setEnabled(ok)
        self.bt_run.setToolTip("Ajoute d'abord des sources" if not s.sources_loaded else
                               ("Crée d'abord un workflow" if not s.paths else
                                "Tous les workflows du projet : un récapitulatif s'affiche avant de lancer"))
        txt = f"{SCENARIOS[s.scenario]}  ·  {n} workflow{'s' if n > 1 else ''}"
        if s.path is not None:
            txt += f"  ·  affiché : {s.path.name}"
        if n:
            txt += f"  ·  {'à calculer : ' + hm(sec) if pend else 'tout est calculé'}"
        self.status.setText(txt)

    def run(self):
        self.wf.launch(None, "Lancer le projet")
        self.update_status()

    def help(self):
        d = QDialog(self)
        d.setWindowTitle("Aide de la maquette")
        lv = QVBoxLayout(d)
        t = QLabel(HELP)
        t.setWordWrap(True)
        t.setTextFormat(Qt.RichText)
        lv.addWidget(t)
        b = QPushButton("Commencer")
        b.setObjectName("primary")
        b.clicked.connect(d.accept)
        lv.addWidget(b, 0, Qt.AlignRight)
        d.resize(780, 720)
        d.exec()


def main():
    import traceback

    from PySide6.QtWidgets import QMessageBox
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    QLocale.setDefault(QLocale(QLocale.French, QLocale.France))     # decimal comma in the number fields

    def show_error(t, v, tb):                     # launched without a console: errors go to a dialog
        QMessageBox.critical(None, "Maquette alt2 : erreur", "".join(traceback.format_exception(t, v, tb))[-3000:])

    sys.excepthook = show_error
    if not (DATA / "data.json").exists():
        QMessageBox.warning(None, "Maquette alt2", "Les images de la maquette ne sont pas préparées.\n\nLancer une "
                            "fois :\nC:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe "
                            "mockup/alt2/build/prepare.py")
        return
    w = Main()
    w.resize(1680, 1000)
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
