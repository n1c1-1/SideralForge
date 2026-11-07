"""Round 4, area clavier_fenetre, axis 3: French <-> English in the middle of work (a selection, an open panel, Inspect,
Post, the Log, the Settings dialog open), through the Settings dialog as the user does it. Prints.

    python explore5_clavier_lang.py [graph|inspect|post|back]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from explore_post_common import P, start  # noqa: E402
from explore2_blocks_common import G, Explorer, drawn  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QWidget, QTableWidget, QAbstractItemView  # noqa: E402

start(1200)
ex = Explorer("clavier_fenetre", lang="fr", size=(1600, 950))
from astro_gpu.client import prefs  # noqa: E402
prefs.set("language", "fr")             # as a French user has it (the harness applies the language only)
g = G(ex)
pp = P(ex)
WHERE = sys.argv[1:] or ["graph"]
CHAIN = ["calibration", "registration", "stacking", "astrometry", "gradient", "denoise", "color", "stars"]
FR_WORDS = re.compile(r"\b(Empilement|Astrométrie|Débruitage|Couleur|Étoiles|Alignement|brutes|calque|Calque|"
                      r"Supprimer|Désactiver|désactivé|réactivé|Annulé|Rétabli|Projet|Graphe|Inspection|Journal|"
                      r"Exécuter|Réglages|avancés|Méthode|Pas encore|calculé|Ouvrir|Vers Post|Renommer|séance|"
                      r"Fichier|Édition|Affichage|Aide|Session ajoutée|à exécuter|gardé|supprimée?|ajoutée?|"
                      r"Recombiner|Sans étoiles|Fond|fond|Paramètres|Langue|Enregistrer|Branche)\b")
EN_WORDS = re.compile(r"\b(Stacking|Registration|Denoise|Color|Stars|Delete|Switch off|Undone|Redone|Graph|Inspect|"
                      r"Log|Run|Settings|advanced|Method|Not computed|Open|To Post|Rename|File|Edit|View|Help|"
                      r"added|removed|deleted|kept|Raw stack|Combine|Starless|Background|layer|Layer|Branch|"
                      r"Language|Save|the|to run)\b")
NAMES = ("Nuit A", "Mon flux")


def lang_via_settings(code):
    res = {}

    def h(d):
        cb = next((c for c in d.findChildren(QComboBox) if c.findData(code) >= 0), None)
        if cb is None:
            res["err"] = "no language combo"
            d.reject()
            return
        cb.setCurrentIndex(cb.findData(code))
        for _ in range(10):
            QApplication.processEvents()
        res["dialog_texts"] = ex.texts(d) if False else [t for t in _texts(d)]
        d.grab().save(str(ex.out / f"lang_settings_after_{code}.png"))
        d.reject()
    ex.dialog_handler = h
    ex.w.open_settings()
    ex.settle(15)
    ex.lang = code
    return res


def _texts(root: QWidget) -> list[str]:
    from PySide6.QtWidgets import QAbstractButton, QLabel, QGroupBox
    out = []
    for x in [root] + root.findChildren(QWidget):
        if not x.isVisible():
            continue
        if isinstance(x, (QLabel, QAbstractButton)) and x.text():
            out.append(x.text())
        if isinstance(x, QGroupBox) and x.title():
            out.append(x.title())
        if isinstance(x, QComboBox):
            out += [x.itemText(i) for i in range(x.count())]
        if isinstance(x, QLineEdit) and x.placeholderText():
            out.append(x.placeholderText())
        if x.toolTip():
            out.append(x.toolTip())
        if isinstance(x, QTableWidget):
            for r in range(min(x.rowCount(), 30)):
                for c in range(x.columnCount()):
                    it = x.item(r, c)
                    if it is not None and it.text():
                        out.append("[table] " + it.text())
            for c in range(x.columnCount()):
                hi = x.horizontalHeaderItem(c)
                if hi is not None:
                    out.append("[header] " + hi.text())
    return out


def all_texts():
    out = _texts(ex.w)
    for it in ex.view.scene().items() if ex.w.graph is not None else []:
        for attr in ("title", "summary", "text"):
            v = getattr(it, attr, None)
            if isinstance(v, str) and v:
                out.append("[graph] " + v)
        if it.toolTip():
            out.append("[graph tip] " + it.toolTip())
    for m in ex.w.menuBar().actions():
        out.append("[menu] " + m.text())
        if m.menu():
            for a in m.menu().actions():
                if a.text():
                    out.append("[menu] " + a.text())
    out.append("[title] " + ex.w.windowTitle())
    out.append("[status] " + ex.w.status.text())
    return out


def other_lang(texts, code):
    rx = FR_WORDS if code == "en" else EN_WORDS
    hits = []
    for t in dict.fromkeys(texts):
        if any(n in t for n in NAMES):
            continue
        for m in rx.finditer(t):
            hits.append(f"{m.group(0)!r} in {t[:140]!r}")
            break
    return hits


def state():
    return {"sel": ex.s.sel_step, "cur": ex.s.current, "mode": ex.w.mode, "undo": len(ex.s._undo),
            "redo": len(ex.s._redo), "dirty": getattr(ex.s, "dirty", None),
            "paths": [q.name for q in ex.s.paths], "sessions": [x.name for x in ex.s.p.sessions],
            "blocks": [[n["block"] + ("/off" if n.get("off") else "") for n in (w.blocks or [])]
                       for w in ex.s.p.workflows],
            "log": ex.w.b_log.isChecked()}


def prepare():
    g.new("session_empty_wf", CHAIN, clear=False)
    # unsaved edits: a name typed, a workflow renamed, a block switched off, an image sent to Post
    ex.click_block(g.blk("sources"))
    ed = next((e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisibleTo(ex.page.panel)), None)
    ex.click(ed)
    ed.selectAll()
    QTest.keyClicks(ed, "Nuit A")
    QTest.keyClick(ed, Qt.Key_Return)
    ex.settle(6)

    def ren(d):
        e = d.findChildren(QLineEdit)
        e[0].setText("Mon flux")
        d.accept()
    ex.dialog_handler = ren
    g.menu(g.blk("couleur"), "Renommer")
    ex.settle(8)
    print("renamed:", [q.name for q in ex.s.paths], ex.modals[-1:], [x.get("text") for x in ex.menus[-1]][:12])
    ex.click_block(g.blk("fond"))
    pp.key(Qt.Key_D)
    g.menu(g.blk("empilement"), "Vers Post")
    ex.click_block(g.blk("debruitage"))
    ex.settle(6)


for where in WHERE:
    print("=====", where, flush=True)
    prepare()
    if where == "inspect":
        pp.key(Qt.Key_Return)
    elif where == "post":
        ex.w.set_mode(2)
    elif where == "log":
        ex.w.b_log.setChecked(True)
    ex.settle(8)
    st0 = state()
    ex.shot(f"lang_{where}_before_fr")
    res = lang_via_settings("en")
    ex.settle(10)
    pp.check_slot_errors(f"langue -> en ({where})")
    st1 = state()
    print("state before:", json.dumps(st0, ensure_ascii=False))
    print("state after: ", json.dumps(st1, ensure_ascii=False))
    print("settings dialog after switch, French left:", other_lang(res.get("dialog_texts", []), "en")[:10], res.get("err"))
    tx = all_texts()
    hits = other_lang(tx, "en")
    print(f"window after switch to en: {len(hits)} French hits")
    for h in hits[:40]:
        print("   FR:", h)
    ex.shot(f"lang_{where}_after_en")
    # the panel shown, the menus of a block, the toast of a gesture, Undo's words
    m = g.menu(g.blk("couleur"), None) if g.blk("couleur") else []
    mh = other_lang(pp.menu_texts(m), "en")
    print("block menu French hits:", mh[:8])
    pp.undo()
    print("Ctrl+Z after switch: status", repr(ex.w.status.text()), "toast", repr(pp.toast()), "state", state()["blocks"])
    ex.shot(f"lang_{where}_undo_en")
    pp.redo()
    print("Ctrl+Y: status", repr(ex.w.status.text()))
    # back to French
    lang_via_settings("fr")
    ex.settle(10)
    pp.check_slot_errors(f"langue -> fr ({where})")
    st2 = state()
    print("state back fr:", json.dumps(st2, ensure_ascii=False))
    hits = other_lang(all_texts(), "fr")
    print(f"window after switch back to fr: {len(hits)} English hits")
    for h in hits[:40]:
        print("   EN:", h)
    ex.shot(f"lang_{where}_back_fr")
    pp.undo()
    print("Ctrl+Z back in fr: status", repr(ex.w.status.text()))
ex.close()
