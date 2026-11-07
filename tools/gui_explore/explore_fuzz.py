"""GUI exploration, area "fuzz": long random sequences of real gestures, invariants checked after every step.

Seeded and reproducible: a sequence is a start (a project kind, or a new project with a template) and a list of
abstract actions ``[name, r1, r2, r3]`` (the r's pick among what the window shows at that moment). The same list
replayed on the same start does the same gestures.

    python explore_fuzz.py run 17            # one sequence (seed 17): runs/seed_17.json
    python explore_fuzz.py drive 0 300 4     # seeds 0..299, 4 processes at a time, each under a hard timeout
    python explore_fuzz.py minimize <sig>    # the shortest failing sequence of a signature, minimised (ddmin)
    python explore_fuzz.py note <case.json>  # replayed twice, then noted (with its minimal steps)

Invariants after each step: no Python exception in the window's slots, no hang (watchdog), the definition validates,
plans / keys / health checks compute, the drawn graph matches the model (every block drawn, no overlap, no orphan
wire), Undo then Redo gives back exactly the same definition (and Undo gives back the one before the edit), no
template window or left-open dialog, no blocking pop-up for ordinary gestures, no internal word shown. At the end of a
sequence the whole Undo chain goes back to the start, and the whole Redo chain to the end.
"""
from __future__ import annotations

import faulthandler
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
if MODE in ("drive", "summary", "show", "minimize_all", "note_all"):
    Explorer = None                                   # the driver only spawns processes: no window
else:
    from harness import TMP, Explorer, RunsBlocked  # noqa: E402

OUT = Path("F:/astro_essais/A_REGARDER/exploration2/fuzz")
RUNS = OUT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
PY = sys.executable

KINDS = ["blank", "session", "session_empty_wf", "image", "session+image", "two_sessions", "two_flows", "mono",
         "legacy", "jpg_mistake"]
TEMPLATES = ["blank", "full", "classic_dn", "classic", "nights", "calibrated", "aligned", "raw", "image",
             "image_colours", "image_plain", "image_stretched", "mono_each", "sho", "hso", "hoo", "rgb", "lrgb",
             "dualband"]
ACTIONS = [("dbl_card", 10), ("drag_card", 14), ("click_block", 7), ("menu", 14), ("cross", 5), ("del", 5),
           ("d", 4), ("to_palette", 4), ("pull_port", 4), ("wire", 4), ("undo", 8), ("redo", 5), ("add_session", 3),
           ("panel", 9), ("files", 4), ("image_file", 3), ("mode", 4), ("lang", 1), ("enter", 1), ("esc", 1)]
NEEDS_GRAPH = {"dbl_card", "drag_card", "click_block", "menu", "cross", "del", "d", "to_palette", "pull_port", "wire",
               "add_session", "panel", "files", "image_file", "enter", "esc"}
ORDINARY = NEEDS_GRAPH - {"panel", "files", "image_file", "add_session", "enter"}
SEV = {"exc": "bloquant", "hang": "bloquant", "never_ready": "bloquant", "validate": "bloquant", "plan": "bloquant",
       "keys": "bloquant", "health": "bloquant", "undo_redo": "gênant", "undo_back": "gênant", "not_undoable": "gênant",
       "chain_start": "gênant", "chain_end": "gênant", "graph_stale": "gênant", "graph_missing": "gênant",
       "overlap": "gênant", "orphan_wire": "mineur", "template_window": "gênant", "dialog_open": "gênant",
       "popup": "gênant", "text": "mineur", "multi_undo": "mineur", "qt": "mineur", "harness": "harness"}


# ---------------------------------------------------------------------------------------------- the sequence
def make_sequence(seed: int) -> dict:
    rng = random.Random(seed)
    if rng.random() < 0.6:
        start = {"kind": rng.choice(KINDS)}
    else:
        start = {"template": rng.choice(TEMPLATES)}
    start["lang"] = "fr" if rng.random() < 0.6 else "en"
    names, weights = zip(*ACTIONS)
    n = rng.randint(20, 60)
    acts = [[rng.choices(names, weights)[0], rng.random(), rng.random(), rng.random()] for _ in range(n)]
    return {"seed": seed, "start": start, "actions": acts}


def pick(lst, r):
    """An element by its r in [0, 1); a hand-written case may give a text instead: the element it names (a palette
    block's key or title, a block's title, a menu item)."""
    if not lst:
        return None
    if isinstance(r, str):
        def name(x):
            if isinstance(x, tuple):
                return " ".join(str(getattr(y, "title", y)) for y in x[:2])
            return str(getattr(x, "title", x))
        hit = [x for x in lst if r == name(x) or r in name(x).split(" ")] or [x for x in lst if r in name(x)]
        return hit[0] if hit else None
    return lst[min(int(r * len(lst)), len(lst) - 1)]


def lt(r, v) -> bool:
    return not isinstance(r, str) and r < v


def norm(text: str) -> str:
    t = re.sub(r"[A-Za-z]:[\\/][^\s'\"]*", "<path>", str(text))
    t = re.sub(r"\d+", "N", t)
    return t[:160]


class Fail(Exception):
    pass


class Fuzz:
    """One process: the window, the gestures, the checks."""

    def __init__(self, lang: str):
        from PySide6.QtCore import qInstallMessageHandler
        self.ex = Explorer("fuzz", lang=lang)
        self.errs: list[str] = []
        self.qt: list[str] = []
        sys.excepthook = self._hook
        qInstallMessageHandler(self._qt)
        self.dialogs: list[str] = []
        self.fileans = 0.0                             # the r that decides the file dialogs' answers of this step
        self._data = 0
        self._menus0 = self._popups0 = 0
        self.step_fails: list[dict] = []
        self._patch_files()
        self.ex.dialog_handler = self._dialog

    # ------------------------------------------------------------ capture
    def _hook(self, t, v, tb):
        if t is RunsBlocked or "RunsBlocked" in str(t):
            return
        self.errs.append("".join(traceback.format_exception(t, v, tb)))

    def _qt(self, mode, ctx, msg):
        if "propagateSizeHints" in msg or "QFont::setPointSize" in msg:
            return
        self.qt.append(msg)

    def _patch_files(self):
        from PySide6.QtWidgets import QFileDialog
        fz = self

        def answer(kind):
            r = fz.fileans
            ex = fz.ex
            fz._data += 1
            k = fz._data
            if kind == "folder":
                opts = ["raws", "raws", "mono", "empty", "", "jpgdir"]
                o = pick(opts, r)
                if o == "raws":
                    return str(ex.raws(f"add{k}", 6 + k % 5, start=7000 + 20 * k))
                if o == "mono":
                    return str(ex.fits_frames(f"mono{k}_{('Ha', 'OIII', 'SII')[k % 3]}", 4, ("Ha", "OIII", "SII")[k % 3]))
                if o == "empty":
                    d = TMP / "data" / f"empty{k}"
                    d.mkdir(parents=True, exist_ok=True)
                    return str(d)
                if o == "jpgdir":
                    d = TMP / "data" / f"jpg{k}"
                    d.mkdir(parents=True, exist_ok=True)
                    from PIL import Image
                    import numpy as np
                    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(d / "IMG_0001.JPG")
                    return str(d)
                return ""
            if kind == "files":
                o = pick(["raws", "jpg", "mixed", "none"], r)
                if o == "none":
                    return []
                d = ex.raws(f"files{k}", 3, start=8000 + 10 * k)
                fs = sorted(str(x) for x in d.iterdir())
                if o in ("jpg", "mixed"):
                    from PIL import Image
                    import numpy as np
                    j = d / "PHONE_0001.JPG"
                    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(j)
                    fs = [str(j)] if o == "jpg" else fs + [str(j)]
                return fs
            if kind == "file":
                o = pick(["fits", "fits", "png", "missing", ""], r)
                if o == "fits":
                    return str(ex.image_file(f"img{k}.fits"))
                if o == "png":
                    return str(ex.image_file(f"img{k}.png"))
                if o == "missing":
                    return str(TMP / "data" / "nowhere" / "gone.fits")
                return ""
            return ""

        def fd(kind):
            def f(*a, **kw):
                fz.ex.file_asks.append(f"{kind}: {a[1] if len(a) > 1 else ''}")
                ans = answer(kind)
                if kind in ("file", "files", "save"):
                    return (ans, "")
                return ans
            return f
        QFileDialog.getOpenFileName = staticmethod(fd("file"))
        QFileDialog.getOpenFileNames = staticmethod(fd("files"))
        QFileDialog.getExistingDirectory = staticmethod(fd("folder"))
        QFileDialog.getSaveFileName = staticmethod(fd("save"))

    def _dialog(self, d):
        """Every modal dialog: recorded; a name typed in a name dialog, the import accepted when it has read the
        folder, the language switched in Settings; the template window (it must not come) cancelled."""
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QComboBox, QLineEdit
        ex = self.ex
        ex.dialog_handler = self._dialog                  # armed again for the next one
        name = d.__class__.__name__
        self.dialogs.append(name if name != "QDialog" else "QDialog:" + d.windowTitle())
        if name == "TemplateGallery":
            d.reject()
            return
        if name == "ImportDialog":
            t0 = time.time()

            def poll():
                if not d.isVisible():
                    return
                if getattr(d, "ok", None) is not None and d.ok.isEnabled():
                    d.ok.click()
                elif time.time() - t0 > 8:
                    d.reject()
                else:
                    QTimer.singleShot(100, poll)
            QTimer.singleShot(100, poll)
            return
        if name == "SettingsDialog":
            for cb in d.findChildren(QComboBox):
                if cb.findData("en") >= 0 and cb.findData("fr") >= 0:
                    cur = cb.currentData()
                    cb.setCurrentIndex(cb.findData("fr" if cur == "en" else "en"))
                    break
            QTimer.singleShot(50, d.reject)
            return
        eds = [e for e in d.findChildren(QLineEdit) if e.isVisible()]
        if eds:
            names = ["Nuit du 12", "", "x" * 90, "Nuit 2 bis", "A/B:C*?", "Étoiles"]
            eds[0].setText(pick(names, self.fileans))
            QTimer.singleShot(0, d.accept)
            return
        QTimer.singleShot(0, d.reject)

    # ------------------------------------------------------------ the start
    def start(self, start: dict, tag: str):
        ex = self.ex
        from astro_gpu.client import app as appmod
        from astro_gpu.client import prefs
        from astro_gpu.client.i18n import language
        if ex.w is not None and ex.w.s is not None:      # the last sequence's project (maybe broken): never saved
            ex.w.s.readonly = True
            ex.w.s._dirty = False
        if language() != start["lang"]:
            prefs.set("language", start["lang"])
            if ex.w is not None:
                ex.w.change_language(start["lang"])
            else:
                appmod.apply_language(start["lang"])
        if "kind" in start:
            root = ex.project(start["kind"], name=f"{start['kind'].replace('+', '_')}_{tag}")
            ex.open(root)
        else:
            ex.open()
            key = start["template"]

            def h(d, key=key):
                self.dialogs.append("TemplateGallery(start)")
                ex.dialog_handler = self._dialog
                if d.__class__.__name__ != "TemplateGallery":
                    d.reject()
                    return
                i = next((i for i, k in enumerate(d.tab_keys) for t in d.templates if t.get("key") == key and
                          t["tab"] == k), 0)
                d.tabs.setCurrentIndex(i)
                d._tab(i)
                t = next((c.t for c in d.cards if c.t.get("key") == key), None)
                if t is not None:
                    d.pick(t)
                if key.startswith("image"):
                    d.file.setText(str(ex.image_file("start.png" if key == "image_stretched" else "start.fits")))
                if d.ok.isEnabled():
                    d.ok.click()
                else:                                   # Create greyed (a filter missing): Cancel, a blank page
                    d.reject()
            ex.dialog_handler = h
            ex.w.create_project(str(ex.ws / f"np_{tag}"), f"np_{tag}", "M 31")
            ex.settle(20)
            ex.dialog_handler = self._dialog
        self.wait_ready()
        self.dialogs.clear()

    def activate(self):
        """The window active, as the user's is (offscreen, no window is: the shortcuts Ctrl+Z, Del, D would not
        fire)."""
        from PySide6.QtWidgets import QApplication
        if QApplication.activeWindow() is not self.ex.w:
            self.ex.w.activateWindow()
            self.ex.settle(1)

    def undo(self):
        self.activate()
        self.ex.undo()

    def redo(self):
        self.activate()
        self.ex.redo()

    def wait_ready(self, t: float = 25.0) -> bool:
        ex = self.ex
        end = time.time() + t
        while time.time() < end:
            ex.settle(2)
            s = ex.w.s if ex.w is not None else None
            if s is None:
                return True
            page = ex.w.graph
            if s.ready and (page is None or not page._pending):
                ex.settle(2)
                return True
        return False

    # ------------------------------------------------------------ what the window shows
    def items(self):
        its = self.ex.blocks()
        return sorted(its, key=lambda it: (round(it.sceneBoundingRect().x()), round(it.sceneBoundingRect().y()),
                                           it.node.key))

    def title(self, it) -> str:
        return (it.title or it.node.key).replace("\n", " ")

    def ensure_graph(self, desc: list):
        w = self.ex.w
        if w.mode != 0:
            self.ex.click(w.mode_btns[0])
            desc.append("clic sur « " + w.mode_btns[0].text() + " »")
            self.wait_ready()

    # ------------------------------------------------------------ the gestures
    def act(self, a: list) -> str:
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QLineEdit, QListWidget,
                                       QPushButton, QSlider)
        from PySide6.QtTest import QTest
        ex = self.ex
        name, r1, r2, r3 = a
        self.fileans = r3
        desc: list[str] = []
        if ex.w.project is None:
            return "(pas de projet ouvert)"
        self.activate()
        if name in NEEDS_GRAPH:
            self.ensure_graph(desc)
        vp = ex.view.viewport()
        if name == "dbl_card":
            cards = list(ex.page.cards.items())
            ok = [kv for kv in cards if not kv[1].reason]
            b, c = pick(ok if (ok and lt(r2, 0.8) and not isinstance(r1, str)) else cards, r1)
            desc.append(f"double-clic sur la carte « {c.title} » de la palette" + (" (grisée)" if c.reason else ""))
            ex.double_click_card(b)
        elif name == "drag_card":
            cards = list(ex.page.cards.items())
            ok = [kv for kv in cards if not kv[1].reason]
            b, c = pick(ok if (ok and lt(r3, 0.85) and not isinstance(r1, str)) else cards, r1)
            targets = []
            if not c.reason:
                if b not in ("session", "image", "fusion"):
                    ex.drag_palette_hover(b, QPoint(3, 3))
                    targets = [("la place proposée « " + (g.toolTip() or getattr(g, "text", "")) + " »",
                                ex.ghost_center(g)) for g in sorted(ex.ghosts(), key=lambda g: (
                                    g.sceneBoundingRect().x(), g.sceneBoundingRect().y()))]
                    ex.drag_leave()
                z = ex.new_flow_zone()
                if z is not None and z.scene() is ex.view.scene():
                    targets.append(("la zone du bas (nouveau flux)", ex.ghost_center(z)))
            targets.append(("un endroit vide du graphe", QPoint(int(vp.width() * (0.1 + 0.8 * r2)),
                                                                 int(vp.height() * (0.1 + 0.8 * r3)))))
            where, pt = pick(targets, r2)
            desc.append(f"glisser la carte « {c.title} » sur {where}" + (" (carte grisée)" if c.reason else ""))
            rec = ex.drag_palette(b, pt)
            if c.reason and rec is not None:
                self.fail("text", "drag_greyed", f"la carte grisée « {c.title} » lance quand même un glisser")
        elif name in ("click_block", "menu", "cross", "to_palette", "pull_port"):
            its = self.items()
            if name == "cross":
                rem = [it for it in its if it.removable]
                its = rem if (rem and lt(r2, 0.9) and not isinstance(r1, str)) else its
            if name == "to_palette":
                rem = [it for it in its if it.removable]
                its = rem if (rem and lt(r2, 0.9) and not isinstance(r1, str)) else its
            if name == "pull_port":
                its = [it for it in its if it.ports]
            it = pick(its, r1)
            if it is None:
                return "; ".join(desc + [f"({name}: aucun bloc)"])
            t = self.title(it)
            if name == "click_block":
                desc.append(f"clic sur le bloc « {t} »")
                ex.click_block(it)
            elif name == "cross":
                desc.append(f"clic sur la croix du bloc « {t} »" + ("" if it.removable else " (sans croix)"))
                ex.click_block(it, "cross")
            elif name == "to_palette":
                desc.append(f"glisser le bloc « {t} » sur la palette")
                ex.drag_block_to_palette(it)
            elif name == "pull_port":
                what, y = pick(list(it.ports), r2)
                a_ = ex.view.mapFromScene(it.mapToScene(QPointF(it.w, y)))
                post = ex.view.final_item
                if post is None:
                    return "; ".join(desc + ["(pas de bloc Post)"])
                to_post = not lt(0.8, r3) if not isinstance(r3, str) else True
                b_ = ex.view.mapFromScene(post.sceneBoundingRect().center()) if to_post else \
                    QPoint(int(vp.width() * r3), int(vp.height() * r2))
                desc.append(f"tirer le port {'' if what is None else '« ' + what + ' » '}du bloc « {t} » sur " +
                            ("Post" if to_post else "un endroit vide"))
                ex._mouse(vp, _ev("press"), a_)
                for k in (0.3, 0.6, 1.0):
                    ex._mouse(vp, _ev("move"), QPoint(int(a_.x() + (b_.x() - a_.x()) * k),
                                                      int(a_.y() + (b_.y() - a_.y()) * k)), Qt.NoButton, Qt.LeftButton)
                ex._mouse(vp, _ev("release"), b_, Qt.LeftButton, Qt.NoButton)
                ex.settle(6)
            else:                                       # menu
                items = ex.right_click_block(it)
                leaves = _leaves(items)
                if not leaves:
                    desc.append(f"clic droit sur le bloc « {t} » (menu vide ou tout grisé)")
                else:
                    txt = pick(leaves, r2)
                    desc.append(f"clic droit sur le bloc « {t} » → « {txt} »")
                    it2 = next((x for x in self.items() if x.node.key == it.node.key and x.node.sid == it.node.sid and
                                [q.pid for q in x.node.paths] == [q.pid for q in it.node.paths]), None)
                    if it2 is not None:
                        ex.right_click_block(it2, pick=txt)
        elif name == "del":
            desc.append("touche Suppr")
            ex.key(Qt.Key_Delete)
        elif name == "d":
            desc.append("touche D")
            ex.key(Qt.Key_D)
        elif name == "enter":
            desc.append("touche Entrée")
            ex.key(Qt.Key_Return)
        elif name == "esc":
            desc.append("touche Échap")
            ex.key(Qt.Key_Escape)
        elif name == "wire":
            from astro_gpu.client.graph import WireItem
            ws = sorted([x for x in ex.view.scene().items() if isinstance(x, WireItem)],
                        key=lambda w: (w.path().pointAtPercent(0.5).x(), w.path().pointAtPercent(0.5).y()))
            w = pick(ws, r1)
            if w is None:
                return "; ".join(desc + ["(aucun fil supprimable)"])
            p = ex.view.mapFromScene(w.mapToScene(w.path().pointAtPercent(0.35 + 0.3 * r3)))
            tip = (w.toolTip() or "").split("\n")[0]
            if lt(r2, 0.6):
                desc.append(f"clic sur le fil « {tip} » puis Suppr")
                ex.click(vp, p)
                ex.key(Qt.Key_Delete)
            else:
                from PySide6.QtGui import QContextMenuEvent
                from PySide6.QtWidgets import QApplication
                desc.append(f"clic droit sur le fil « {tip} » → Retirer")
                from astro_gpu.client.i18n import _ as tr
                ex.menu_pick = tr("Remove")
                QApplication.sendEvent(vp, QContextMenuEvent(QContextMenuEvent.Mouse, p, vp.mapToGlobal(p)))
                ex.settle()
        elif name == "undo":
            desc.append("Ctrl+Z")
            self.undo()
        elif name == "redo":
            desc.append("Ctrl+Y")
            self.redo()
        elif name == "add_session":
            from astro_gpu.client.graph import GhostItem
            gs = [g for g in ex.view.scene().items() if isinstance(g, GhostItem) and g.on_click is not None]
            if gs and lt(r1, 0.7):
                g = gs[0]
                desc.append("clic sur « + Session » (réponse au choix de dossier: " + _ans_word("folder", r3) + ")")
                ex.click(vp, ex.ghost_center(g))
            else:
                its = [it for it in self.items() if it.node.key in ("session", "sources")]
                it = pick(its, r2)
                if it is None:
                    return "; ".join(desc + ["(pas de bloc Session)"])
                from astro_gpu.client.i18n import _ as tr
                desc.append(f"clic droit sur « {self.title(it)} » → « {tr("Add session…")} » (dossier: "
                            f"{_ans_word('folder', r3)})")
                ex.right_click_block(it, pick=tr("Add session…"))
        elif name in ("panel", "files", "image_file"):
            if name in ("files", "image_file"):
                keys = ("session", "sources") if name == "files" else ("image",)
                its = [it for it in self.items() if it.node.key in keys]
                it = pick(its, r1)
                if it is not None:
                    desc.append(f"clic sur le bloc « {self.title(it)} »")
                    ex.click_block(it)
            pn = ex.page.panel
            ws = [w for w in pn.findChildren(QPushButton) + pn.findChildren(QComboBox) + pn.findChildren(QLineEdit) +
                  pn.findChildren(QCheckBox) + pn.findChildren(QAbstractSpinBox) + pn.findChildren(QSlider) +
                  pn.findChildren(QListWidget)
                  if w.isVisible() and w.isEnabled() and not isinstance(w.parent(), QAbstractSpinBox)
                  and not isinstance(w.parent(), QComboBox)]
            if name == "files":
                ws = [w for w in ws if isinstance(w, QListWidget) or (isinstance(w, QPushButton) and (
                    w.text().startswith(("×", "▸", "▾")) or w.menu() is not None))]
            if name == "image_file":
                ws = [w for w in ws if isinstance(w, QPushButton) and "…" in w.text()] or ws
            ws.sort(key=lambda w: (w.mapTo(pn, QPoint(0, 0)).y(), w.mapTo(pn, QPoint(0, 0)).x()))
            w = pick(ws, r2)
            if w is None:
                return "; ".join(desc + ["(panneau: rien à cliquer)"])
            label = getattr(w, "text", lambda: "")() if not isinstance(w, QListWidget) else "liste de fichiers"
            label = label if isinstance(label, str) else ""
            tip = (w.toolTip() or "").split("\n")[0]
            what = f"« {label or tip or w.__class__.__name__} »"
            if isinstance(w, QPushButton):
                if w.menu() is not None:
                    acts = [x.text() for x in w.menu().actions() if not x.isSeparator() and x.isEnabled()]
                    txt = pick(acts, r3)
                    ex.menu_pick = txt
                    desc.append(f"panneau: {what} → « {txt} » (réponse: {_ans_word('folder' if txt and 'old' in txt else 'files', r3)})")
                else:
                    desc.append(f"panneau: clic sur {what}")
                ex.click(w)
            elif isinstance(w, QComboBox):
                i = min(int(r3 * w.count()), w.count() - 1)
                desc.append(f"panneau: liste {what} → « {w.itemText(i)} »")
                w.setCurrentIndex(i)
                w.activated.emit(i)
            elif isinstance(w, QLineEdit):
                names = ["Nuit du 12", "", "x" * 90, "Nuit 2 bis", "A/B:C*?", "Étoiles"]
                t = pick(names, r3)
                desc.append(f"panneau: saisir « {t[:20]} » dans {what} puis Entrée")
                w.setFocus()
                w.selectAll()
                QTest.keyClick(w, Qt.Key_Delete)
                if t:
                    QTest.keyClicks(w, t)
                QTest.keyClick(w, Qt.Key_Return)
                w.editingFinished.emit()
            elif isinstance(w, QCheckBox):
                desc.append(f"panneau: case {what}")
                ex.click(w)
            elif isinstance(w, QAbstractSpinBox):
                lo, hi = w.minimum(), w.maximum()
                v = lo + (hi - lo) * r3
                v = int(v) if isinstance(lo, int) else v
                desc.append(f"panneau: valeur {what} → {v}")
                w.setValue(v)
                w.editingFinished.emit()
            elif isinstance(w, QSlider):
                v = int(w.minimum() + (w.maximum() - w.minimum()) * r3)
                desc.append(f"panneau: curseur {what} → {v}")
                w.setValue(v)
                w.sliderReleased.emit()
            elif isinstance(w, QListWidget):
                n = w.count()
                if n:
                    w.setFocus()
                    w.clearSelection()
                    for i in range(n):
                        if (i * 0.37 + r3) % 1 < 0.4:
                            w.item(i).setSelected(True)
                    desc.append(f"panneau: choisir {len(w.selectedItems())} fichier(s) de la liste puis Suppr")
                    QTest.keyClick(w, Qt.Key_Delete)
            ex.settle(6)
        elif name == "mode":
            i = min(int(r1 * 3), 2)
            b = ex.w.mode_btns[i]
            desc.append(f"clic sur le mode « {b.text()} »")
            ex.click(b)
        elif name == "lang":
            desc.append("Réglages (Ctrl+,): changer de langue")
            from PySide6.QtWidgets import QApplication
            ex.key(Qt.Key_Comma, ex.w, Qt.ControlModifier)
        return "; ".join(desc)

    def fail(self, kind, sig, detail):
        self.step_fails.append({"kind": kind, "sig": f"{kind}:{sig}", "detail": str(detail)[:3000]})

    # ------------------------------------------------------------ the state, for Undo
    def state(self) -> dict:
        s = self.ex.w.s
        d = s.p.to_dict()
        wfs = json.loads(json.dumps(d["workflows"], sort_keys=True, default=str))
        return {"workflows": wfs, "stack_mode": d["stack_mode"],
                "session_ids": [x["id"] for x in d["sessions"]],
                "session_meta": [(x["id"], x["name"], x["filter"]) for x in d["sessions"]],
                "post": json.loads(json.dumps(d.get("post"), sort_keys=True, default=str)),
                "current": d["current"]}

    @staticmethod
    def diff(a: dict, b: dict, skip=("current",)) -> list[str]:
        return [k for k in a if k not in skip and a[k] != b.get(k)]

    @staticmethod
    def explain(a: dict, b: dict, comp: str) -> str:
        x, y = a.get(comp), b.get(comp)
        if comp == "workflows":
            ia = {w["id"]: w for w in x}
            ib = {w["id"]: w for w in y}
            out = []
            for i in sorted(set(ia) | set(ib)):
                if i not in ia or i not in ib:
                    out.append(f"workflow {i}: {'absent avant' if i not in ia else 'absent après'}")
                    continue
                for f in ia[i]:
                    if ia[i][f] != ib[i].get(f):
                        out.append(f"workflow {i} « {ia[i]['name']} », champ {f}: {json.dumps(ia[i][f])[:300]} → "
                                   f"{json.dumps(ib[i].get(f))[:300]}")
            return " | ".join(out)[:1500]
        return f"{json.dumps(x, ensure_ascii=False)[:600]} → {json.dumps(y, ensure_ascii=False)[:600]}"

    def undo_len_new(self, last) -> int:
        """How many Undo entries were added since ``last`` (the top entry before)."""
        u = self.ex.w.s._undo
        if last is None:
            return len(u)
        for i in range(len(u) - 1, -1, -1):
            if u[i] is last:
                return len(u) - 1 - i
        return len(u)

    # ------------------------------------------------------------ the checks
    def check(self, name: str, before: dict | None, last_undo, popups0: int, modals0: int):
        ex = self.ex
        from astro_gpu import api
        w = ex.w
        for e in self.errs:
            loc = _where(e)
            self.fail("exc", f"{_exc_type(e)}@{loc}", e)
        self.errs.clear()
        for m in self.qt:
            if "Traceback" in m or "Error" in m:
                self.fail("qt", norm(m)[:80], m)
        self.qt.clear()
        for d in self.dialogs:
            if d.startswith("TemplateGallery"):
                self.fail("template_window", name, "la fenêtre des modèles s'ouvre après le début du projet")
            elif name in ORDINARY and not d.startswith(("QDialog", "SettingsDialog", "ImportDialog")):
                self.fail("dialog_open", f"{name}:{d}", f"une fenêtre {d} s'ouvre")
        self.dialogs.clear()
        left = [x.__class__.__name__ for x in ex.dialogs()]
        if left:
            self.fail("dialog_open", "left:" + ",".join(sorted(left)), "fenêtre restée ouverte: " + ", ".join(left))
            ex.close_dialogs()
        for kind, title, text in ex.popups[popups0:]:
            if name in ORDINARY:
                self.fail("popup", f"{name}:{norm(title)}", f"{kind} « {title} »: {text}")
        if w.project is None or w.s is None:
            return
        s = w.s
        try:
            s.p.validate()
        except Exception as e:  # noqa: BLE001
            self.fail("validate", norm(f"{type(e).__name__}: {e}")[:90], traceback.format_exc()[-1500:])
        if s.ready:
            for what, fn in (("plan", lambda: api.plan_all(s.p)), ("health", lambda: api.health_checks(s.p))):
                try:
                    fn()
                except Exception as e:  # noqa: BLE001
                    self.fail(what, f"{type(e).__name__}@{_where(traceback.format_exc())}",
                              f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
            for wf in s.p.workflows:
                try:
                    api.blocks(s.p, wf.id)
                except Exception as e:  # noqa: BLE001
                    self.fail("keys", f"{type(e).__name__}@{_where(traceback.format_exc())}",
                              f"workflow {wf.id} « {wf.name} »: {type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}")
        self.check_graph()
        self.check_texts()
        # Undo: an edit adds one entry; Undo gives back the definition before it, Redo the one after
        if before is None or w.mode == 2 or name in ("undo", "redo", "lang"):
            return
        after = self.state()
        n_new = self.undo_len_new(last_undo)
        changed = self.diff(before, after)
        if changed and n_new == 0:
            comps = ",".join(changed)
            self.fail("not_undoable", f"{name}:{comps}", f"{comps} changé sans entrée d'annulation: " +
                      " ; ".join(self.explain(before, after, c) for c in changed))
        if n_new > 1:
            self.fail("multi_undo", name, f"un seul geste ajoute {n_new} entrées d'annulation")
        if n_new >= 1:
            for _ in range(n_new):
                self.undo()
                self.wait_ready()
            back = self.state()
            d1 = self.diff(before, back)
            if d1:
                self.fail("undo_back", f"{name}:{','.join(d1)}", "Ctrl+Z ne rend pas l'état d'avant le geste: " +
                          " ; ".join(self.explain(before, back, c) for c in d1))
            for _ in range(n_new):
                self.redo()
                self.wait_ready()
            again = self.state()
            d2 = self.diff(after, again)
            if d2:
                self.fail("undo_redo", f"{name}:{','.join(d2)}", "Ctrl+Z puis Ctrl+Y ne rend pas le même état: " +
                          " ; ".join(self.explain(after, again, c) for c in d2))
            for e in self.errs:
                self.fail("exc", f"{_exc_type(e)}@{_where(e)}", "(pendant Ctrl+Z/Ctrl+Y de contrôle)\n" + e)
            self.errs.clear()

    def check_graph(self):
        from PySide6.QtCore import QRectF
        from PySide6.QtWidgets import QGraphicsPathItem
        from astro_gpu.client.blocks import Graph
        from astro_gpu.client.graph import BlockItem
        ex = self.ex
        s = ex.w.s
        if not s.ready or ex.view.drag_block is not None:
            return
        view = ex.view
        g_now = Graph(s)

        def sig(g):
            return sorted((n.key, n.sid or 0, tuple(sorted(q.pid for q in n.paths)), bool(n.cut), n.lid or 0)
                          for n in g.nodes if n.paths or n.key in ("sources", "final"))
        if view.graph is None or sig(view.graph) != sig(g_now):
            self.fail("graph_stale", "", "le graphe dessiné n'est pas celui du modèle (pas reconstruit)")
        its = [x for x in view.scene().items() if isinstance(x, BlockItem)]
        for p in s.paths:
            try:
                keys = s.display_keys(p)
            except Exception as e:  # noqa: BLE001
                self.fail("exc", f"display_keys:{type(e).__name__}", traceback.format_exc()[-1500:])
                continue
            for k in keys:
                if k == "post":
                    continue
                if not any(it.node.key == k and (p in it.node.paths or it.node.sid is not None) for it in its):
                    self.fail("graph_missing", re.sub(r"~\d+", "~N", k),
                              f"le bloc {k} du workflow « {p.name} » n'est pas dessiné")
        ns = len(s.p.sessions)
        drawn_s = [it for it in its if it.node.key == "session"]
        if ns >= 2 and len(drawn_s) != ns and s.separate_sessions():
            self.fail("graph_missing", "session", f"{ns} sessions, {len(drawn_s)} blocs Session dessinés")
        rects = [(it, it.mapToScene(QRectF(0, 0, it.w, 64)).boundingRect().adjusted(3, 3, -3, -3)) for it in its]
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, ra = rects[i]
                b, rb = rects[j]
                inter = ra.intersected(rb)
                if inter.width() > 2 and inter.height() > 2:
                    ka, kb = sorted([re.sub(r"~\d+", "~N", a.node.key), re.sub(r"~\d+", "~N", b.node.key)])
                    self.fail("overlap", f"{ka}/{kb}", f"les blocs « {self.title(a)} » et « {self.title(b)} » se "
                              f"chevauchent ({inter.width():.0f}×{inter.height():.0f})")
        anchors = [it.sceneBoundingRect().adjusted(-16, -16, 16, 16) for it in its]
        paths = [x for x in view.scene().items() if isinstance(x, QGraphicsPathItem)]
        for x in paths:
            if x.pen().widthF() >= 2.9:                  # the sessions' join bar
                anchors.append(x.sceneBoundingRect().adjusted(-6, -6, 6, 6))
        pull = getattr(view, "_pull_path", None)
        for x in paths:
            if x is pull or x.pen().widthF() >= 2.9:
                continue
            pa = x.path()
            if pa.elementCount() < 2:
                continue
            e0, e1 = pa.elementAt(0), pa.elementAt(pa.elementCount() - 1)
            for e in (e0, e1):
                pt = x.mapToScene(e.x, e.y)
                if not any(r.contains(pt) for r in anchors):
                    self.fail("orphan_wire", "", f"un fil finit dans le vide en ({pt.x():.0f}, {pt.y():.0f}): "
                              f"{(x.toolTip() or '').splitlines()[0] if x.toolTip() else ''}")
                    break

    def check_texts(self):
        from astro_gpu.client.i18n import language
        ex = self.ex
        lang = language()
        texts = list(ex.texts())
        try:
            texts.append(ex.w.status.text())
        except RuntimeError:
            pass
        for m in ex.menus[self._menus0:]:
            texts += [x.get("text", "") for x in _flat(m)] + [x.get("tip", "") for x in _flat(m)]
        texts += [p[1] + " " + p[2] for p in ex.popups[self._popups0:]]
        pats = [r"\b[a-z]+~\d+\b", r"\b[a-z]+@\d+\b", r"\brender_simple\b", r"\bstar_removal\b",
                r"\b(astrometrie|debruitage|etoiles)\b", r"Traceback", r"\b(KeyError|ValueError|TypeError|"
                r"AttributeError|IndexError|NoneType)\b", r"\bNone\b"]
        if lang == "en":
            pats.append(r"\b(fond|empilement|alignement|couleur)\b")
        seen = set()
        for t in texts:
            if not isinstance(t, str) or not t:
                continue
            for line in t.splitlines():
                if re.search(r"[A-Za-z]:[\\/]", line):     # a path (tooltips of folders): not checked
                    continue
                for pat in pats:
                    m = re.search(pat, line)
                    if m and (pat, m.group(0)) not in seen:
                        seen.add((pat, m.group(0)))
                        tok = re.sub(r"\d+", "N", m.group(0))
                        self.fail("text", f"{lang}:{tok}", f"« {line[:200]} »")

    # ------------------------------------------------------------ one sequence
    def run(self, seq: dict, tag: str, stop_sig: str | None = None, progress: Path | None = None,
            watchdog: float = 90.0, hangfile=None) -> dict:
        ex = self.ex
        t0 = time.time()
        res = {"seed": seq.get("seed"), "start": seq["start"], "actions": seq["actions"], "steps": [], "failures": [],
               "done": 0}
        faulthandler.dump_traceback_later(watchdog * 2, exit=True, file=hangfile or sys.stderr)
        self.step_fails = []
        try:
            self.start(seq["start"], tag)
        except Exception:  # noqa: BLE001
            res["failures"].append({"step": -1, "kind": "harness", "sig": "harness:start",
                                    "detail": traceback.format_exc()[-2000:]})
            return res
        for e in self.errs:
            res["failures"].append({"step": -1, "kind": "exc", "sig": f"exc:{_exc_type(e)}@{_where(e)}",
                                    "detail": "(à l'ouverture)\n" + e})
        self.errs.clear()
        self.qt.clear()
        s0 = self.state() if ex.w.s is not None else None
        u0 = ex.w.s._undo[-1] if ex.w.s is not None and ex.w.s._undo else None
        n_u0 = len(ex.w.s._undo) if ex.w.s is not None else 0
        seen: dict[str, dict] = {}
        for i, a in enumerate(seq["actions"]):
            if progress is not None:
                progress.write_text(json.dumps({"seed": seq.get("seed"), "step": i, "action": a,
                                                "steps": res["steps"]}, ensure_ascii=False), encoding="utf-8")
            faulthandler.dump_traceback_later(watchdog, exit=True, file=hangfile or sys.stderr)
            self.step_fails = []
            s = ex.w.s
            before = self.state() if s is not None else None
            last = s._undo[-1] if s is not None and s._undo else None
            p0, m0 = len(ex.popups), len(ex.modals)
            self._menus0, self._popups0 = len(ex.menus), len(ex.popups)
            try:
                d = self.act(a)
            except RunsBlocked:
                d = f"{a[0]} (lancement bloqué)"
            except Exception:  # noqa: BLE001
                d = f"{a[0]} (erreur du banc)"
                self.fail("harness", a[0], traceback.format_exc()[-2000:])
            res["steps"].append(d)
            if not self.wait_ready():
                self.fail("never_ready", a[0], "le projet reste « en lecture » plus de 25 s")
            try:
                self.check(a[0], before, last, p0, m0)
            except Exception:  # noqa: BLE001
                self.fail("harness", "check:" + a[0], traceback.format_exc()[-2000:])
            for f in self.step_fails:
                f["step"] = i
                if f["sig"] in seen:                   # once per sequence (its first step), counted
                    seen[f["sig"]]["count"] = seen[f["sig"]].get("count", 1) + 1
                    continue
                seen[f["sig"]] = f
                res["failures"].append(f)
            res["done"] = i + 1
            if stop_sig and any(f["sig"] == stop_sig for f in self.step_fails):
                res["stopped"] = i
                faulthandler.cancel_dump_traceback_later()
                return res
            if any(f["kind"] == "validate" for f in self.step_fails):
                res["broken"] = i                       # the definition is broken: what follows is its echo
                faulthandler.cancel_dump_traceback_later()
                res["time"] = round(time.time() - t0, 1)
                return res
        # the whole Undo chain back to the start, then the whole Redo chain to the end
        faulthandler.dump_traceback_later(watchdog * 3, exit=True, file=hangfile or sys.stderr)
        self.step_fails = []
        if ex.w.s is not None and s0 is not None and (not stop_sig or stop_sig.startswith(("chain", "exc"))):
            try:
                self._u0 = u0
                self._chains(s0, n_u0, len(seq["actions"]))
            except Exception:  # noqa: BLE001
                self.fail("harness", "chains", traceback.format_exc()[-2000:])
            for f in self.step_fails:
                f["step"] = len(seq["actions"])
                res["failures"].append(f)
        faulthandler.cancel_dump_traceback_later()
        res["time"] = round(time.time() - t0, 1)
        return res

    def _chains(self, s0: dict, n_u0: int, n: int):
        ex = self.ex
        w = ex.w
        if w.mode != 0:
            ex.click(w.mode_btns[0])
            self.wait_ready()
        end = self.state()
        undone = extra = 0
        while len(w.s._undo) > n_u0 and undone < 260:
            self.undo()
            self.wait_ready()
            undone += 1
        while len(w.s._undo) < n_u0 and w.s.can_redo() and extra < 260:   # the user undid past the start
            self.redo()
            self.wait_ready()
            extra += 1
        st = self.state()
        d = self.diff(s0, st)
        u = w.s._undo
        reachable = n_u0 == 0 or (len(u) == n_u0 and u[-1] is self._u0)   # else the user undid past the start
        if d and reachable:                                             # and edited since: the start is gone
            self.fail("chain_start", ",".join(d), "toute la chaîne Ctrl+Z ne revient pas au début: " +
                      " ; ".join(self.explain(s0, st, c) for c in d))
        for _ in range(extra):                         # back to the end: what was undone, redone (a redo the user
            self.undo()                                # left pending stays pending)
            self.wait_ready()
        for _ in range(undone):
            self.redo()
            self.wait_ready()
        st = self.state()
        d = self.diff(end, st)
        if d:
            self.fail("chain_end", ",".join(d), "toute la chaîne Ctrl+Y ne revient pas à la fin: " +
                      " ; ".join(self.explain(end, st, c) for c in d))
        for e in self.errs:
            self.fail("exc", f"{_exc_type(e)}@{_where(e)}", "(pendant la chaîne Ctrl+Z / Ctrl+Y)\n" + e)
        self.errs.clear()


# ---------------------------------------------------------------------------------------------- helpers
def _ev(kind):
    from PySide6.QtCore import QEvent
    return {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove, "release": QEvent.MouseButtonRelease}[kind]


def _flat(items):
    for x in items:
        yield x
        for y in _flat(x.get("sub") or []):
            yield y


def _leaves(items) -> list[str]:
    out = []
    for x in _flat(items):
        if x.get("text") in (None, "---") or x.get("sub") or not x.get("enabled"):
            continue
        out.append(x["text"].split("\t")[0])
    return out


def _ans_word(kind, r):
    if kind == "folder":
        return {"raws": "un dossier de bruts", "mono": "un dossier FITS avec filtre", "empty": "un dossier vide",
                "": "Annuler", "jpgdir": "un dossier avec un JPEG"}[pick(["raws", "raws", "mono", "empty", "",
                                                                          "jpgdir"], r)]
    if kind == "files":
        return {"raws": "3 bruts", "jpg": "un JPEG", "mixed": "3 bruts et un JPEG", "none": "Annuler"}[
            pick(["raws", "jpg", "mixed", "none"], r)]
    return {"fits": "une image FITS", "png": "une image PNG", "missing": "un fichier absent", "": "Annuler"}[
        pick(["fits", "fits", "png", "missing", ""], r)]


def _exc_type(tb: str) -> str:
    last = [x for x in tb.strip().splitlines() if x and not x.startswith(" ")]
    return (last[-1].split(":")[0] if last else "?")[:60]


def _where(tb: str) -> str:
    fr = re.findall(r'File "([^"]+)", line (\d+), in (\S+)', tb)
    app = [f for f in fr if "astro_gpu" in f[0].replace("\\", "/")]
    f = (app or fr or [("?", "0", "?")])[-1]
    return f"{Path(f[0]).name}:{f[2]}"


# ---------------------------------------------------------------------------------------------- modes
def cmd_run(seed: int):
    seq = make_sequence(seed)
    fz = Fuzz(seq["start"]["lang"])
    prog = RUNS / f"seed_{seed}.progress.json"
    hang = open(RUNS / f"seed_{seed}.hang.txt", "w", encoding="utf-8")
    res = fz.run(seq, f"s{seed}", progress=prog, hangfile=hang)
    (RUNS / f"seed_{seed}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    hang.close()
    print("SEED", seed, "steps", res["done"], "fails", len(res["failures"]), "sigs",
          sorted({f["sig"] for f in res["failures"]}), "t", res.get("time"), flush=True)
    os._exit(0)                                       # no slow teardown of the window and its threads


def cmd_drive(a: int, b: int, par: int, timeout: float = 1500.0):
    """Seeds a..b-1, ``par`` processes at a time, each under a hard timeout; a process that dies (hang, crash)
    leaves its progress: the seed is recorded as a hang at that step."""
    todo = [s for s in range(a, b) if not (RUNS / f"seed_{s}.json").exists()]
    running: list[tuple[int, subprocess.Popen, float]] = []
    log = open(OUT / "drive.log", "a", encoding="utf-8")
    while todo or running:
        while todo and len(running) < par:
            s = todo.pop(0)
            env = dict(os.environ)
            p = subprocess.Popen([PY, str(Path(__file__)), "run", str(s)], cwd=str(HERE.parents[1]),
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((s, p, time.time()))
        time.sleep(2)
        for s, p, t in list(running):
            rc = p.poll()
            if rc is None and time.time() - t < timeout:
                continue
            if rc is None:
                p.kill()
                rc = "timeout"
            running.remove((s, p, t))
            if not (RUNS / f"seed_{s}.json").exists():
                prog = RUNS / f"seed_{s}.progress.json"
                info = json.loads(prog.read_text(encoding="utf-8")) if prog.exists() else {}
                hang = (RUNS / f"seed_{s}.hang.txt")
                ht = hang.read_text(encoding="utf-8", errors="replace") if hang.exists() else ""
                seq = make_sequence(s)
                res = {"seed": s, "start": seq["start"], "actions": seq["actions"], "steps": info.get("steps", []),
                       "done": info.get("step", 0), "failures": [{
                           "step": info.get("step", 0), "kind": "hang",
                           "sig": f"hang:{(info.get('action') or ['?'])[0]}@{_where(ht) if ht else rc}",
                           "detail": f"le processus ne répond plus (code {rc})\n" + ht[-2500:]}]}
                (RUNS / f"seed_{s}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print("done", s, rc, flush=True)
    log.close()


def cmd_minimize_all(sigs: list[str], par: int = 4, timeout: float = 2400.0):
    """``minimize`` of each signature in its own process (a hang kills one, not all), ``par`` at a time."""
    todo = [x.strip() for x in sigs if x.strip()]
    running = []
    log = open(OUT / "minimize.log", "a", encoding="utf-8")
    while todo or running:
        while todo and len(running) < par:
            sig = todo.pop(0)
            p = subprocess.Popen([PY, str(Path(__file__)), "minimize", sig], cwd=str(HERE.parents[1]), stdout=log,
                                 stderr=subprocess.STDOUT)
            running.append((sig, p, time.time()))
        time.sleep(2)
        for sig, p, t in list(running):
            if p.poll() is None and time.time() - t < timeout:
                continue
            if p.poll() is None:
                p.kill()
            running.remove((sig, p, t))
            print("minimised", sig, p.returncode, flush=True)
    log.close()


def cmd_note_all(plan_file: str):
    """Each planned note (its case, its title, the signatures of its root cause): the count of sequences hitting it and
    their seeds written into its case, the case replayed twice and noted (``confirm``), the seeds in seeds.txt."""
    plan = json.loads(Path(plan_file).read_text(encoding="utf-8"))
    runs = load_runs()
    lines = []
    log = OUT / "anomalies.jsonl"
    done = {json.loads(x)["title"] for x in log.read_text(encoding="utf-8").splitlines() if x.strip()} \
        if log.exists() else set()
    for it in plan:
        seeds = sorted({r["seed"] for r in runs for f in r["failures"] if f["sig"] in it["sigs"]})
        cf = OUT / "cases" / it["case"]
        case = json.loads(cf.read_text(encoding="utf-8"))
        case.update(title=it["title"], severity=it["severity"], detail=it["detail"], seeds=seeds, n_seq=len(seeds),
                    n_total=len(runs))
        cf.write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8")
        lines.append(f"{it['severity']}\t{it['title']}\t{len(seeds)} séquences\tgraines: {' '.join(map(str, seeds))}")
        if it["title"] in done:                         # noted by an earlier run of this mode
            continue
        r = subprocess.run([PY, str(Path(__file__)), "confirm", str(cf)], cwd=str(HERE.parents[1]), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=900)
        print(it["title"][:60], [x for x in r.stdout.splitlines() if x.startswith("CONFIRM")], flush=True)
    (OUT / "seeds.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_runs() -> list[dict]:
    out = []
    for f in sorted(RUNS.glob("seed_*.json")):
        if f.name.endswith(".progress.json"):
            continue
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except ValueError:
            pass
    return out


def cmd_summary():
    runs = load_runs()
    by: dict[str, list] = {}
    n_act = 0
    for r in runs:
        n_act += r.get("done", 0)
        for f in r["failures"]:
            by.setdefault(f["sig"], []).append((r["seed"], f["step"]))
    print(f"{len(runs)} sequences, {n_act} actions")
    for sig, hits in sorted(by.items(), key=lambda kv: -len({h[0] for h in kv[1]})):
        print(f"{len({h[0] for h in hits}):4d} seq  {sig}   e.g. seed {hits[0][0]} step {hits[0][1]}")


def cmd_show(f: str, width: int = 400):
    r = json.loads(Path(f).read_text(encoding="utf-8"))
    print(r["start"], "seed", r.get("seed"))
    fs: dict[int, list] = {}
    for x in r["failures"]:
        fs.setdefault(x["step"], []).append(x)
    for f_ in fs.get(-1, []):
        print("START !!", f_["sig"], "|", f_["detail"][:width].replace("\n", " / "))
    for i, s in enumerate(r["steps"]):
        print(i, r["actions"][i][0], "|", s)
        for f_ in fs.get(i, []):
            print("    !!", f_["sig"], "|", f_["detail"][:width].replace("\n", " / "))
    for f_ in fs.get(len(r["steps"]), []):
        print("END !!", f_["sig"], "|", f_["detail"][:width * 2].replace("\n", " / "))


def replay(fz: "Fuzz", start: dict, actions: list, sig: str, tag: str) -> dict:
    return fz.run({"seed": None, "start": start, "actions": actions}, tag, stop_sig=sig, watchdog=60)


def cmd_minimize(sig: str, max_trials: int = 70):
    """The shortest failing sequence of ``sig`` (the earliest failing step cuts it), minimised by ddmin: actions
    dropped while it still fails. Writes cases/<slug>.json."""
    runs = load_runs()
    cands = []
    for r in runs:
        st = [f["step"] for f in r["failures"] if f["sig"] == sig]
        if st:
            cands.append((min(st) + 1, r))
    if not cands:
        print("no run with", sig)
        return
    cands.sort(key=lambda x: x[0])
    n, r = cands[0]
    acts = r["actions"][:n]
    fz = Fuzz(r["start"]["lang"])
    k = [0]

    def fails(lst) -> bool:
        k[0] += 1
        res = replay(fz, r["start"], lst, sig, f"m{k[0]}_{abs(hash(sig)) % 10000}")
        return any(f["sig"] == sig for f in res["failures"])
    if r["failures"] and any(f["kind"] == "hang" for f in r["failures"] if f["sig"] == sig):
        print("hang: not minimised in-process")
        best = acts
    elif not fails(acts):
        print("NOT REPRODUCED", sig, "seed", r["seed"])
        best = None
    else:
        best = acts
        k_ = 1                                          # first: only the last actions (the failing one and before)
        while k_ < len(best):
            if fails(best[-k_:]):
                best = best[-k_:]
                break
            k_ *= 2
        gran = 2
        while len(best) >= 2 and k[0] < max_trials:
            chunk = max(1, len(best) // gran)
            reduced = False
            for i in range(0, len(best), chunk):
                cand = best[:i] + best[i + chunk:]
                if cand and fails(cand):
                    best = cand
                    gran = max(gran - 1, 2)
                    reduced = True
                    break
                if k[0] >= max_trials:
                    break
            if not reduced:
                if chunk == 1:
                    break
                gran = min(len(best), gran * 2)
        if len(best) == 1 and k[0] < max_trials and fails([]):
            best = []
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", sig)[:80]
    (OUT / "cases").mkdir(exist_ok=True)
    case = {"sig": sig, "seed": r["seed"], "start": r["start"], "actions": best, "trials": k[0],
            "n_seq": len({c[1]["seed"] for c in cands}), "orig_len": n}
    (OUT / "cases" / f"{slug}.json").write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8")
    print("MIN", sig, "from", n, "to", None if best is None else len(best), "trials", k[0], flush=True)
    os._exit(0)


def cmd_confirm(case_file: str):
    """Replays a minimal case twice (fresh projects): the steps and whether each run reproduced it."""
    case = json.loads(Path(case_file).read_text(encoding="utf-8"))
    fz = Fuzz(case["start"]["lang"])
    outs = []
    for k in range(2):
        res = replay(fz, case["start"], case["actions"] or [], case["sig"], f"c{k}_{abs(hash(case['sig'])) % 10000}")
        hit = [f for f in res["failures"] if f["sig"] == case["sig"]]
        outs.append({"ok": bool(hit), "steps": res["steps"], "detail": hit[0]["detail"] if hit else ""})
        if k == 1 and hit:
            case["shot"] = fz.ex.shot("case_" + re.sub(r"[^a-zA-Z0-9]+", "_", case["sig"])[:60])
    case["confirm"] = outs
    Path(case_file).write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8")
    print("CONFIRM", case["sig"], [o["ok"] for o in outs], flush=True)
    if case.get("title"):                             # annotated by hand: noted, with its minimal steps
        start = case["start"]
        st = ("projet de départ: " + (f"type « {start['kind']} »" if "kind" in start else
                                       f"nouveau projet, modèle « {start['template']} »") + f", langue {start['lang']}")
        both = all(o["ok"] for o in outs)
        steps = [st] + (outs[-1]["steps"] if outs[-1]["ok"] else outs[0]["steps"])
        detail = case["detail"] + f"\n\nReproduit {sum(o['ok'] for o in outs)}/2 fois sur la séquence minimale. " + \
            f"Vu dans {case.get('n_seq', '?')} séquence(s) sur {case.get('n_total', '?')} (graines: " + \
            f"{', '.join(map(str, case.get('seeds', [])[:12]))}). Signature: {case['sig']}." + \
            ("" if both else " (reproduction partielle : à confirmer)")
        if outs[-1]["detail"]:
            detail += "\n\nCe que le contrôle a vu: " + outs[-1]["detail"][:900]
        fz.ex.note(case["severity"], case["title"], detail, shot=True, steps=steps)
    os._exit(0)


if __name__ == "__main__":
    if MODE == "run":
        cmd_run(int(sys.argv[2]))
    elif MODE == "drive":
        cmd_drive(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 4)
    elif MODE == "summary":
        cmd_summary()
    elif MODE == "show":
        cmd_show(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 400)
    elif MODE == "minimize":
        cmd_minimize(sys.argv[2])
    elif MODE == "minimize_all":
        cmd_minimize_all(Path(sys.argv[2]).read_text(encoding="utf-8").split("\n") if Path(sys.argv[2]).exists()
                         else sys.argv[2].split("|"), int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    elif MODE == "note_all":
        cmd_note_all(sys.argv[2])
    elif MODE == "confirm":
        cmd_confirm(sys.argv[2])
