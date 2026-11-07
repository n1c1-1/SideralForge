"""Round 4, area clavier_fenetre: each anomaly reproduced twice, then written (ex.note) with its screenshot.

    python explore5_clavier_notes.py [ctrl_s | port_escape | undo_words | auto_names | post_undo | keys | tab] [--write]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from explore_post_common import P, start  # noqa: E402
from explore2_blocks_common import G, Explorer, drawn  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit  # noqa: E402

args = sys.argv[1:]
WRITE = "--write" in args
ONLY = [a for a in args if a != "--write"]
start(1500)
ex = Explorer("clavier_fenetre")
prefs.set("language", "fr")
g = G(ex)
pp = P(ex)
CHAIN = ["calibration", "registration", "stacking", "astrometry", "gradient", "denoise", "color"]
ST = ["Nouveau projet, une session de 15 brutes, un workflow vide",
      "Double-clic dans la palette : Calibration, Alignement, Empilement, Astrométrie, Gradient, Débruitage, Couleur"]
found: dict[str, list] = {}


def hit(key, ok: bool, shot=None):
    found.setdefault(key, []).append((ok, shot))


def note(key, severity, title, detail, steps):
    reps = found.get(key, [])
    ok = [s for o, s in reps if o]
    print(f"{key}: reproduced {len(ok)}/{len(reps)}", flush=True)
    if len(ok) >= 2 and WRITE:
        ex.note(severity, title, detail, shot=ok[-1] or True, steps=steps)


def lang(code):
    def h(d):
        cb = next(c for c in d.findChildren(QComboBox) if c.findData(code) >= 0)
        cb.setCurrentIndex(cb.findData(code))
        for _ in range(10):
            QApplication.processEvents()
        d.reject()
    ex.dialog_handler = h
    ex.w.open_settings()
    ex.settle(12)


def name_field():
    pn = ex.page.panel
    return next((e for e in pn.findChildren(QLineEdit) if e.isVisibleTo(pn)), None)


def ctrl_s():
    for rep in range(2):
        g.new("session_empty_wf", CHAIN)
        ex.click_block(g.blk("sources"))
        ed = name_field()
        ex.click(ed)
        ed.selectAll()
        QTest.keyClicks(ed, "Nuit du 4")
        pp.activate()
        ed.setFocus()
        QTest.keyClick(ed, Qt.Key_S, Qt.ControlModifier)
        ex.settle(8)
        disk = None
        for f in Path(ex.s.p.root).glob("*.astroproj"):
            disk = json.loads(f.read_text(encoding="utf-8"))
        saved = [x["name"] for x in disk["sessions"]] if disk else None
        print(rep, "field", ed.text(), "model", ex.s.p.sessions[0].name, "disk", saved, "title", ex.w.windowTitle(),
              "status", ex.w.status.text())
        sh = ex.shot(f"note_ctrl_s_{rep}")
        hit("ctrl_s", ed.text() == "Nuit du 4" and saved == ["Session 1"], sh)
    note("ctrl_s", "gênant", "Ctrl+S pendant la saisie d'un nom : le nom tapé n'est pas enregistré",
         "Nom de session tapé dans son panneau sans Entrée, puis Ctrl+S : la barre dit « Projet enregistré », le champ "
         "montre « Nuit du 4 », mais le bloc et le fichier du projet gardent « Session 1 ». Les logiciels de référence "
         "valident le champ en cours avant d'enregistrer ; ici l'utilisateur croit avoir enregistré son nom.",
         ST + ["Clic sur le bloc Session 1", "Panneau : clic dans le champ du nom, taper « Nuit du 4 » (sans Entrée)",
               "Ctrl+S"])


def port_escape():
    from astro_gpu.client.graph import BH
    for rep in range(2):
        g.new("session_empty_wf", CHAIN)
        it = g.blk("fond")
        vp = ex.view.viewport()
        a = ex.view.mapFromScene(it.mapToScene(QPointF(it.w - 1, BH / 2)))
        b = ex.view.mapFromScene(ex.view.final_item.sceneBoundingRect().center())
        n0 = len(pp.graph_layers())
        ex._mouse(vp, QEvent.MouseButtonPress, a)
        for t in (0.3, 0.6, 0.9):
            ex._mouse(vp, QEvent.MouseMove, QPoint(int(a.x() + (b.x() - a.x()) * t), int(a.y() + (b.y() - a.y()) * t)),
                      Qt.NoButton, Qt.LeftButton)
        ex.settle(2)
        pp.activate()
        QTest.keyClick(vp, Qt.Key_Escape)
        ex.settle(3)
        sh = ex.shot(f"note_port_escape_{rep}")       # the pulled wire still drawn after Escape?
        ex._mouse(vp, QEvent.MouseButtonRelease, b, Qt.LeftButton, Qt.NoButton)
        ex.settle(8)
        n1 = len(pp.graph_layers())
        print(rep, "layers", n0, "->", n1, "toast", pp.toast())
        sh2 = ex.shot(f"note_port_escape_after_{rep}")
        hit("port_escape", n1 == n0 + 1, sh2)
    note("port_escape", "mineur", "Échap pendant qu'on tire le port d'un bloc vers Post n'annule pas le geste",
         "Port de Gradient tiré vers Post, Échap appuyé avant de lâcher : le fil suit toujours la souris et, lâché sur "
         "Post, le calque « Workflow · Gradient » est créé. Dans les logiciels de référence, Échap annule un glisser en "
         "cours (rien ne change).",
         ST + ["Tirer le port (●) de Gradient vers le bloc Post sans lâcher", "Échap", "Lâcher sur Post"])


def undo_words():
    for rep in range(2):
        g.new("session_empty_wf", CHAIN, clear=False)
        ex.click_block(g.blk("fond"))
        pp.key(Qt.Key_D)
        lang("en")
        pp.undo()
        st = ex.w.status.text()
        print(rep, "status after Ctrl+Z in English:", st)
        sh = ex.shot(f"note_undo_words_{rep}")
        hit("undo_words", "désactivé" in st, sh)
        lang("fr")
    note("undo_words", "mineur", "Après passage en anglais, Ctrl+Z annonce l'étape en français",
         "Gradient désactivé (D) en français, langue passée en anglais (Paramètres), Ctrl+Z : la barre d'état dit "
         "« Undone: Gradient désactivé · Workflow » — moitié anglais, moitié français. Les mots de l'étape sont gardés "
         "dans la langue du moment où elle a été faite ; le Journal, lui, est retraduit.",
         ST + ["Clic sur Gradient, D", "Édition > Paramètres > Langue : English", "Ctrl+Z"])


def auto_names():
    for rep in range(2):
        g.new("session_empty_wf", CHAIN, clear=False)
        g.menu(g.blk("empilement"), "Vers Post")
        lang("en")
        names = [q.name for q in ex.s.paths]
        ex.click_block(g.blk("final"))
        print(rep, "workflows in English:", names, "panel", pp.panel_texts()[:6])
        sh = ex.shot(f"note_auto_names_{rep}")
        hit("auto_names", any("Empilement brut" in n for n in names), sh)
        lang("fr")
    note("auto_names", "question", "Noms faits par l'application gardés dans la langue d'origine",
         "« Vers Post » sur Empilement en français crée le workflow « Workflow · Empilement brut ». Passé en anglais, "
         "il garde ce nom (fil vers Post, panneau Post, Journal : « Layer \"Workflow · Empilement brut\" from "
         "Stacking »). Les noms tapés par l'utilisateur sont bien gardés ; faut-il que les noms donnés par "
         "l'application (tant que l'utilisateur ne les a pas changés) suivent la langue ?",
         ST + ["Clic droit sur Empilement > Vers Post", "Édition > Paramètres > Langue : English", "Clic sur Post"])


def post_undo():
    for rep in range(2):
        g.new("session_empty_wf", CHAIN)
        g.menu(g.blk("empilement"), "Vers Post")
        n0 = len(ex.s.paths)
        ex.w.set_mode(2)
        ex.settle(8)
        en = ex.w.a_undo.isEnabled()
        pp.undo()
        n1 = len(ex.s.paths)
        st = ex.w.status.text()
        print(rep, "Ctrl+Z in Post: workflows", n0, "->", n1, "Undo enabled", en, "status", st)
        sh = ex.shot(f"note_post_undo_{rep}")
        lang("en")
        lang("fr")
        ex.w.set_mode(2)
        ex.settle(6)
        pp.undo()
        n2 = len(ex.s.paths)
        print(rep, "same after a language switch: workflows", n1, "->", n2, "status", ex.w.status.text())
        hit("post_undo", n1 == n0 and en, sh)
    note("post_undo", "mineur", "Ctrl+Z sur la page Post ne fait rien et ne dit rien après un geste du graphe",
         "« Vers Post » sur Empilement (un calque de plus), puis page Post, Ctrl+Z (Édition > Annuler est actif) : rien "
         "ne change, aucun message. Revenu au graphe, le même Ctrl+Z retire le calque. Sur Post, Ctrl+Z n'annule que les "
         "retouches de Post ; quand il n'y en a pas, rien ne dit pourquoi le geste ne fait rien. Proposer : annuler la "
         "dernière étape du projet, ou griser Annuler avec la raison.",
         ST + ["Clic droit sur Empilement > Vers Post", "Onglet Post", "Ctrl+Z"])


def keys():
    for rep in range(2):
        g.new("session_empty_wf", CHAIN)
        ex.click_block(g.blk("fond"))
        pp.del_key()
        pp.undo()
        has = g.blk("fond") is not None
        pp.key(Qt.Key_Z, ex.w, Qt.ControlModifier | Qt.ShiftModifier)
        redo_shift = g.blk("fond") is None
        ex.click_block(g.blk("debruitage"))
        s0 = ex.s.sel_step
        pp.key(Qt.Key_Right)
        right = ex.s.sel_step
        pp.key(Qt.Key_F2)
        f2 = ex.modals[-1:] if ex.modals else []
        print(rep, "after undo present", has, "Ctrl+Shift+Z redid", redo_shift, "Right", s0, "->", right, "F2 modals", f2)
        sh = ex.shot(f"note_keys_{rep}")
        hit("keys", has and not redo_shift and right == s0, sh)
    note("keys", "question", "Raccourcis attendus absents : Ctrl+Maj+Z, flèches entre blocs, F2",
         "Ctrl+Maj+Z (rétablir dans Photoshop et Lightroom) ne fait rien : seul Ctrl+Y rétablit. Avec un bloc "
         "sélectionné, les flèches ne passent pas au bloc voisin et F2 ne renomme rien (Windows). Ctrl+C / Ctrl+V / "
         "Ctrl+D ne font rien non plus (aucun copier-coller ou dupliquer de bloc). À décider : lesquels ajouter.",
         ST + ["Clic sur Gradient, Suppr, Ctrl+Z", "Ctrl+Maj+Z : Gradient reste là",
               "Clic sur Débruitage, flèche droite : la sélection ne bouge pas ; F2 : rien"])


def tab():
    for rep in range(2):
        g.new("session_empty_wf", CHAIN)
        ex.click_block(g.blk("fond"))
        seen = []
        for _ in range(16):
            pp.activate()
            QTest.keyClick(QApplication.focusWidget() or ex.w, Qt.Key_Tab)
            ex.settle(2)
            f = QApplication.focusWidget()
            seen.append((type(f).__name__, f.text() if hasattr(f, "text") and callable(f.text) else ""))
        print(rep, seen)
        cards = any(t == "Card" or "Card" in t for t, _ in seen)
        sh = ex.shot(f"note_tab_{rep}")
        hit("tab", not cards, sh)
    note("tab", "question", "Clavier seul : les blocs de la palette ne sont pas atteignables avec Tab",
         "Tab depuis le graphe parcourt : « Tout afficher », le panneau (Supprimer, Désactiver, réglages), Journal, "
         "l'onglet Graphe, le choix GPU/CPU, Exécuter, la palette entière (un seul arrêt, sans pouvoir choisir un "
         "bloc), le graphe, puis recommence. Aucun bloc de la palette ni du graphe ne se choisit au clavier, et l'ordre "
         "n'est pas celui de l'écran (le panneau de droite avant la barre du haut et la palette).",
         ST + ["Clic sur Gradient", "Tab répété"])


CASES = {"ctrl_s": ctrl_s, "port_escape": port_escape, "undo_words": undo_words, "auto_names": auto_names,
         "post_undo": post_undo, "keys": keys, "tab": tab}
for name, fn in CASES.items():
    if ONLY and name not in ONLY:
        continue
    print("=====", name, flush=True)
    try:
        fn()
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
    pp.check_slot_errors(name, ST)
ex.close()
