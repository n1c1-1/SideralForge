"""run_status round 2: the new anomalies, reproduced a second time (the first: explore2_run_explore.py) and noted
when their condition holds. Notes in plain French for the user.

    python explore2_run_confirm.py [n01 ...] [--dry]
"""
from __future__ import annotations

import json
import sys
import time

from explore2_run_explore import menu_action, run_menu, truncated, win
from explore2_run_recheck import (  # noqa: F401
    EXC, P, QPushButton, Qt, back, blk, dlg, dump, ex, f5, fresh, obs, place, refs, run_button, run_checks, shift_f5,
    shot, texts)
from harness import _slug
from PySide6.QtCore import QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

DONE: set[str] = set()


def note(sev, title, detail, steps, widget=None):
    if title in DONE:
        return
    DONE.add(title)
    if "--dry" in sys.argv:
        print("WOULD NOTE", sev, "|", title, "\n   ", detail[:700], flush=True)
        return
    if widget is None or isinstance(widget, str):
        ex.note(sev, title, detail, shot=widget or True, steps=steps)
        return
    f = ex.shot(f"{len(ex.notes()) + 1:03d}_{_slug(title)}", widget)
    rec = {"area": ex.area, "severity": sev, "title": title, "detail": detail, "steps": steps, "shot": f,
           "lang": ex.lang, "t": time.strftime("%H:%M:%S")}
    with ex.log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("ANOMALY", sev, title, flush=True)


def n01_run_all_skips_warnings():
    fresh("jpg_mistake")
    place("stacking")
    w = ex.w
    tip = w.bt_run.toolTip()
    d = shift_f5("empilement")
    plan_warn = [x for x in texts(d) if x.startswith("⚠")] if d else []
    ex.close_dialogs()
    d = run_button()
    run_warn = [x for x in texts(d) if x.startswith("⚠") or "JPG" in x] if d else []
    if plan_warn and not run_warn:
        note("gênant", "Exécuter part tout de suite : l'avertissement sur le fichier d'un autre type n'est jamais montré "
             "avant le calcul",
             f"Session de 15 brutes avec un JPEG ajouté par erreur, Empilement posé. Maj+F5 ouvre une fenêtre qui "
             f"avertit : « {plan_warn[0]} ». Le bouton Exécuter (et F5, « Tout exécuter ») lance le calcul sans "
             f"fenêtre préalable : le récapitulatif ne dit rien du JPEG ({[x for x in texts(d)][:6]}), et l'info-bulle "
             f"d'Exécuter est vide ({tip!r}). Même chose pour un flux Image sans fichier (rien ne dit qu'il ne sera "
             "pas exécuté) et pour l'avertissement de place disque : ces avertissements n'existent que dans la "
             "fenêtre de Maj+F5.",
             ["Projet : une session avec un JPEG parmi les brutes", "Placer Empilement",
              "Maj+F5 sur Empilement : l'avertissement est là ; Annuler", "Clic sur Exécuter (barre du haut)"], d)
    ex.close_dialogs()


def n03_undo_with_plan_open():
    fresh("session_empty_wf")
    w = ex.w
    place("calibration", "registration", "stacking", "gradient")
    d = shift_f5("fond")
    ex.undo()
    ex.settle(8)
    d2 = ex.w.run.dialog
    rows = [r.name.text() for r in d2.rows] if d2 is not None else []
    gone = blk("fond") is None
    if d2 is not None and d2.isVisible() and gone and "Gradient" in rows and d2.b_run.isEnabled():
        shot_f = ex.shot(f"{len(ex.notes()) + 1:03d}_plan_after_undo", d2)
        ex.click(d2.b_run)
        for _ in range(10):
            ex.settle(10)
        after = dump(ex.w.run.dialog)
        note("gênant", "Ctrl+Z avec la fenêtre d'exécution ouverte : la fenêtre garde le bloc annulé et l'exécute",
             f"Maj+F5 sur Gradient ouvre « Exécuter « Workflow » jusqu'à Gradient » ({', '.join(rows)}). Ctrl+Z "
             "pendant que la fenêtre est ouverte retire le Gradient du graphe, mais la fenêtre n'a pas changé : "
             "elle liste toujours Gradient, et son bouton Exécuter lance le calcul de l'ancien graphe (après le "
             f"clic : « {after.get('summary')} », lignes {after.get('rows')}). Même chose en changeant GPU/CPU "
             "fenêtre ouverte : la durée reste celle d'avant (« ≈ 1 min » quand le bouton du haut dit « ≈ 5 min » "
             "sur le CPU).",
             ["Session → Calibration, Alignement, Empilement, Gradient", "Clic sur Gradient, Maj+F5",
              "Ctrl+Z (la fenêtre reste ouverte)", "Clic sur Exécuter dans la fenêtre"], shot_f)
    ex.close_dialogs()


def n05_fusion_refusal_scope():
    fresh("mono")
    place("fusion")
    ha = ex.s.paths[0].pid
    d = shift_f5("empilement", ha)
    if d is not None and not d.b_run.isEnabled() and "Fusion" not in [r.name.text() for r in d.rows]:
        note("question", "Fusion impossible : Maj+F5 sur l'Empilement de Ha est refusé aussi, alors qu'il ne calcule "
             "pas la Fusion",
             f"Projet mono Ha/OIII, Fusion posée sans Astrométrie (refusée, entourée de rouge). Maj+F5 sur "
             f"l'Empilement du workflow Ha ouvre « {d.windowTitle()} » avec seulement "
             f"{[r.name.text() for r in d.rows]} ; Exécuter y est grisé et la fenêtre dit en rouge « "
             f"{d.summary.text()} ». Le refus vaut pour tout le projet, même pour un calcul qui ne passe pas par la "
             "Fusion (empiler Ha pour le vérifier avant de régler la Fusion).",
             ["Projet mono Ha/OIII", "Double-clic Fusion dans la palette", "Clic sur l'Empilement de Ha, Maj+F5"], d)
    ex.close_dialogs()


def n06_fusion_refused_still_quotes_time():
    fresh("mono")
    place("fusion")
    w = ex.w
    run_txt, run_en, line = w.bt_run.text(), w.bt_run.isEnabled(), w.status.text()
    ex.close_dialogs()
    ex.double_click_block(blk("fusion"))
    ex.settle(15)
    page = ex.w.tune.pages.currentWidget()
    btns = [b for b in page.findChildren(QPushButton) if b.isVisible() and b.text().startswith("Jusqu")]
    b = btns[0] if btns else None
    info = (b.text(), b.isEnabled(), b.toolTip()) if b else None
    shot_f = ex.shot(f"{len(ex.notes()) + 1:03d}_inspect_fusion_refused")
    back()
    if not run_en and "min" in run_txt and b is not None and b.isEnabled():
        note("mineur", "Fusion refusée : « Jusqu'ici · ≤ 1 min » actif en Inspection, durée sur Exécuter grisé et "
             "« à exécuter » dans la barre d'état",
             f"Fusion sans Astrométrie (Exécuter refusé). Le bouton du haut, grisé, affiche quand même « {run_txt} » ; "
             f"la barre d'état dit « {line} » sans parler du blocage. En Inspection de la Fusion, le bouton "
             f"« {info[0]} » est actif, son info-bulle liste des blocs ({info[2][:80]}…) sans dire que l'exécution "
             "est refusée : un clic ouvre une fenêtre qui refuse. Le panneau de la Fusion ne dit pas non plus "
             "pourquoi elle est entourée de rouge (seule l'info-bulle du bloc le dit).",
             ["Projet mono Ha/OIII", "Double-clic Fusion", "Lire Exécuter et la barre d'état",
              "Double-clic sur Fusion (Inspection)"], shot_f)


def n07_reason_first_channel_only():
    fresh("mono")
    place("fusion")
    tip = ex.w.bt_run.toolTip()
    names = [q.name for q in ex.s.paths]
    has_astro = [bool(blk("astrometrie", q.pid)) for q in ex.s.paths[:2]]
    if "Ha" in tip and "OIII" not in tip and not any(has_astro):
        note("mineur", "Fusion refusée : la raison ne nomme que Ha alors qu'OIII n'a pas d'Astrométrie non plus",
             f"Workflows {names}, aucun n'a d'Astrométrie. Info-bulle d'Exécuter et fenêtre d'exécution : « {tip} ». "
             "Après avoir posé l'Astrométrie dans Ha, la raison devient « OIII : requiert Astrométrie » : on découvre "
             "le second manque seulement après avoir corrigé le premier.",
             ["Projet mono Ha/OIII", "Double-clic Fusion", "Survoler Exécuter"])


def n08_mono_calibration_rows():
    fresh("mono")
    d = f5()
    rows = [r.name.text() for r in d.rows] if d else []
    cal_drawn = [x for x in ex.blocks() if x.node.ref.startswith("calibration")]
    ex.close_dialogs()
    fresh("two_sessions")
    place("calibration", "registration", "stacking")
    d2 = f5()
    rows2 = [r.name.text() for r in d2.rows] if d2 else []
    ex.close_dialogs()
    if cal_drawn and not any(r.startswith("Calibration") for r in rows) and any(r.startswith("Calibration") for r in rows2):
        fresh("mono")
        d = f5()
        note("mineur", "Projet mono : les Calibration de Ha et d'OIII, dessinées au graphe, n'ont pas de ligne dans la "
             "fenêtre d'exécution",
             f"Projet mono (Ha, OIII), chacun Calibration → Alignement → Empilement au graphe. Lignes de la fenêtre : "
             f"{rows} : pas de Calibration (calculée sous « Alignement »). Dans un projet à deux sessions et un "
             f"workflow, la fenêtre a bien ses lignes : {rows2}. La même Calibration posée est montrée ou non selon "
             "le projet.",
             ["Projet mono Ha/OIII (Calibration, Alignement, Empilement)", "F5 (Tout exécuter)"], d)
        ex.close_dialogs()


def n09_off_block_up_to_date():
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "gradient")
    ex.right_click_block(blk("empilement"), "Pause vérif")
    ex.settle(5)
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    ex.settle(8)
    tip = blk("fond").toolTip()
    d = f5()
    r = [(x.name.text(), x.dur.text()) for x in d.rows] if d else []
    P("  rows", r, "tip", tip)
    if ("Gradient", "à jour") in r and "À jour" in tip:
        note("mineur", "Bloc désactivé : « à jour » au graphe et dans la fenêtre d'exécution alors que rien n'est "
             "calculé",
             f"Gradient désactivé (touche D) derrière un Empilement jamais calculé, Pause vérif après Empilement. "
             f"Info-bulle du Gradient au graphe : {tip!r}. Fenêtre d'exécution (F5) : {r} : le Gradient a sa ligne "
             "cochée en vert « à jour ». Un bloc désactivé laisse passer l'image ; dire « à jour » laisse croire qu'il "
             "a été calculé.",
             ["Session → Calibration, Alignement, Empilement, Gradient", "Clic droit sur Empilement > Pause vérif",
              "Clic sur Gradient, D", "Survoler Gradient ; F5"], d)
    ex.close_dialogs()


def n10_run_menu_greyed():
    fresh("session")
    m = run_menu()
    bad = [(t, tip) for t, en, tip, _k in m if not en and (tip == t or not tip)]
    if bad:
        note("mineur", "Menu Exécuter : « Tout exécuter » et « Arrêter » grisés sans dire pourquoi",
             f"Une session de brutes, pas encore de bloc. Menu Exécuter : {[(t, en, tip) for t, en, tip, _k in m]}. "
             "« Jusqu'au bloc choisi » grisé dit pourquoi ; « Tout exécuter » et « Arrêter » grisés n'ont que leur "
             "nom en info-bulle (même chose sans brutes, ou avec des workflows sans bloc). Le bouton Exécuter, lui, "
             "dit « Aucun workflow : poser un bloc ».",
             ["Projet avec une session de brutes, sans bloc", "Ouvrir le menu Exécuter, survoler les entrées"])


def n11_keys_in_plan():
    fresh("session")
    place("stacking")
    d = shift_f5("empilement")
    fw = d.focusWidget() or d
    QApplication.sendEvent(fw, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    QApplication.sendEvent(fw, QKeyEvent(QEvent.KeyRelease, Qt.Key_Escape, Qt.NoModifier))
    ex.settle(6)
    esc_vis = d.isVisible()
    ex.close_dialogs()
    d = shift_f5("empilement")
    default = [b.text() for b in d.findChildren(QPushButton) if b.isDefault()]
    fw = d.focusWidget() or d
    QApplication.sendEvent(fw, QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
    QApplication.sendEvent(fw, QKeyEvent(QEvent.KeyRelease, Qt.Key_Return, Qt.NoModifier))
    ex.settle(15)
    started = ex.w.run.track is not None
    P("  esc visible", esc_vis, "default", default, "started", started, "focus", fw.__class__.__name__,
      getattr(fw, "text", lambda: "")())
    if not started:
        note("mineur", "Fenêtre d'exécution : Entrée ne lance pas l'exécution" + (", Échap ne la ferme pas" if esc_vis
                                                                                  else ""),
             f"Fenêtre « Exécuter « Workflow » jusqu'à Empilement » : le bouton bleu est Exécuter, mais le bouton par "
             f"défaut (celui d'Entrée) est {default} (caché à ce moment) : Entrée ne fait rien."
             + (" Échap ne ferme pas la fenêtre non plus." if esc_vis else " Échap la ferme."),
             ["Session → Empilement", "Clic sur Empilement, Maj+F5", "Appuyer sur Entrée"], d)
    ex.close_dialogs()


def n12_long_title_cut():
    fresh("two_sessions")
    w = ex.w
    w.resize(1366, 768)
    ex.settle(5)
    s = ex.s
    s.p.sessions[0].name = "Nuit du 12 au 13 octobre, Bortle 4, lune 30 %"
    s.p.workflows[0].name = "Nébuleuse du Cœur, champ large avec la caméra principale"
    s.flush()
    ex.w.open_project(s.p.root)
    ex.settle(15)
    place("calibration", "registration", "stacking")
    d = shift_f5("calibration@1")
    tr = truncated(d) if d else []
    if tr and not tr[0][3]:
        note("mineur", "Fenêtre d'exécution : titre coupé sans « … » ni info-bulle avec des noms longs",
             f"Workflow et session aux noms longs, écran 1366 × 768. Maj+F5 sur la Calibration de la session : le "
             f"titre « {d.windowTitle()} » est coupé net à droite (largeur {tr[0][2]} px pour {tr[0][1]} px de texte), "
             "sans points de suspension ni info-bulle. Les lignes, elles, sont abrégées avec « … ».",
             ["Session nommée « Nuit du 12 au 13 octobre, Bortle 4, lune 30 % », workflow au nom long",
              "Placer Calibration, Alignement, Empilement", "Clic sur la Calibration, Maj+F5"], d)
    ex.close_dialogs()
    w.resize(*ex.size)


def n13_settings_cache_bar():
    fresh("session")
    seen = {}

    def look(d):
        d.show_page("folders")
        ex.settle(20)
        seen["t"] = texts(d)
        seen["shot"] = ex.shot(f"{len(ex.notes()) + 1:03d}_settings_cache_bar", d)
        seen["long"] = [x for x in texts(d)]
        d.reject()
    ex.dialog_handler = look
    menu_action("Édition", "Param").trigger()
    ex.settle(25)
    t = seen.get("t", [])
    use = next((x for x in t if " Go sur " in x), None)
    if use:
        note("mineur", "Paramètres : la barre « Cache » montre l'occupation de tout le disque",
             f"Paramètres > Dossiers et disque, « Utilisation du disque » : une ligne « Cache » avec une barre presque "
             f"pleine et « {use} ». C'est le disque qui porte le cache qui est plein à ce point, pas le cache "
             "(projets de test vides) : on lit que le cache prend presque 2 To.",
             ["Édition > Paramètres…", "Page « Dossiers et disque »"], seen["shot"])


def n14_settings_long_names():
    seen = {}

    def look(d):
        out = []
        from PySide6.QtWidgets import QAbstractButton, QListWidget
        nav = d.findChild(QListWidget)
        for i in range(nav.count()):
            nav.setCurrentRow(i)
            ex.settle(3)
            for b in d.pages.currentWidget().findChildren(QAbstractButton):
                if b.isVisible() and b.text():
                    out.append(b.text())
        seen["b"] = out
        nav.setCurrentRow(3)
        ex.settle(3)
        seen["shot"] = ex.shot(f"{len(ex.notes()) + 1:03d}_settings_long_names", d)
        d.reject()
    fresh("session")
    ex.dialog_handler = look
    menu_action("Édition", "Param").trigger()
    ex.settle(25)
    longs = [b for b in seen.get("b", []) if len(b.split()) > 3]
    if longs:
        note("mineur", "Paramètres : boutons et cases à phrases entières, avec articles",
             f"Noms de plus de deux mots dans Paramètres : {longs}. La règle des noms courts (deux mots, sans article, "
             "la phrase dans l'info-bulle) n'y est pas appliquée ; ces contrôles n'ont pas d'info-bulle.",
             ["Édition > Paramètres…", "Parcourir les pages"], seen["shot"])


def n15_off_block_menu():
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "gradient")
    m_on = [a.get("text") for a in ex.right_click_block(blk("fond"))]
    ex.right_click_block(blk("fond"), "Pause vérif")
    ex.settle(5)
    bp_on = [x.breakpoints for x in ex.s.p.workflows]
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)
    ex.settle(8)
    bp_off = [x.breakpoints for x in ex.s.p.workflows]
    m_off = ex.right_click_block(blk("fond"))
    ex.undo()                                        # Ctrl+Z of the D: the block on again, its pause?
    bp_undo = [x.breakpoints for x in ex.s.p.workflows]
    on_again = "off" not in blk("fond").toolTip()
    ex.redo()
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_D)                                 # D again: on
    ex.settle(8)
    bp_back = [x.breakpoints for x in ex.s.p.workflows]
    P("  on_again after undo", on_again)
    P("  back", bp_back, "undo", bp_undo)
    P("  on", m_on, bp_on, "off", [(a.get("text"), a.get("checked")) for a in m_off], bp_off)
    texts_off = [a.get("text") for a in m_off]
    if any(t and "workflow" in t.lower() for t in texts_off) and not any(t and "workflow" in t.lower() for t in m_on):
        note("gênant", "Bloc désactivé : son menu devient celui du workflow (Renommer, Dupliquer, Supprimer workflow)",
             f"Menu du Gradient actif : {m_on}. Le même Gradient désactivé (D) : {texts_off}. Le clic droit sur un "
             "bloc désactivé ouvre les commandes du workflow (Renommer, Dupliquer, Enregistrer comme modèle…, "
             "Supprimer workflow) mêlées à celles du bloc ; « Vers Post » disparaît.",
             ["Session → Calibration, Alignement, Empilement, Gradient", "Clic sur Gradient, D",
              "Clic droit sur Gradient"])
    if bp_on != bp_off:
        note("mineur", "Désactiver un bloc retire sa pause sans le dire",
             f"Pause vérif posée sur Gradient (pauses des workflows : {bp_on}), puis D (désactiver) : {bp_off}. "
             f"Ctrl+Z du D : {bp_undo} ; réactiver avec D : {bp_back}. Rien ne prévient que la pause est perdue, et "
             "le menu du bloc désactivé propose toujours « Pause vérif ».",
             ["Session → Calibration, Alignement, Empilement, Gradient", "Clic droit sur Gradient > Pause vérif",
              "D sur Gradient", "Ctrl+Z, ou D à nouveau"])


def n16_branch_output_block():
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "astrometry")
    ex.drag_palette_hover("gradient", ex.view.viewport().rect().center())
    gs = [(g.branch_key, ex.ghost_center(g)) for g in ex.ghosts() if getattr(g, "branch_key", None)]
    ex.drag_leave()
    tgt = next((c for k, c in gs if k == "astrometrie"), None)
    if tgt is None:
        return
    ex.drag_palette("gradient", tgt)
    ex.settle(15)
    out = blk("post")
    if out is None:
        return
    out_title, out_tip = out.title, out.toolTip()
    d = shift_f5("post")
    info = dump(d)
    note("question", "Après une branche, un bloc « Sortie » apparaît au graphe ; Maj+F5 dessus ouvre « Exécuter … "
         "jusqu'à Sortie »",
         f"Workflow jusqu'à Astrométrie, Gradient déposé sous Astrométrie (branche). Le premier workflow se termine "
         f"par un bloc « {out_title} » ({out_tip!r}). La règle : la Sortie n'est pas un bloc à l'écran. Maj+F5 "
         f"dessus : « {info.get('title')} », lignes {info.get('rows')}.",
         ["Workflow jusqu'à Astrométrie", "Glisser Gradient sous Astrométrie", "Clic sur « Sortie », Maj+F5"])
    ex.close_dialogs()


def n17_log_names():
    fresh("mono")
    ex.right_click_block(blk("calibration@2"), "Pause vérif")
    ex.settle(5)
    ex.undo()
    ex.redo()
    rows = [x for _t, x in ex.w.logp.rows()][:4]
    ex.w.b_log.setChecked(True)
    ex.settle(5)
    if "Annulé" in rows and any(r == "Pause définie après Calibration" for r in rows):
        note("mineur", "Journal : « Annulé », « Rétabli » sans dire quoi ; « Pause définie après Calibration » sans "
             "dire de quelle session",
             f"Projet mono Ha/OIII : Pause vérif sur la Calibration d'OIII, Ctrl+Z, Ctrl+Y. Journal : {rows}. "
             "L'historique ne dit pas ce qui a été annulé ou rétabli, ni de quelle Calibration il s'agit (deux au "
             "graphe ; la fenêtre d'exécution dit « Calibration · Session 1 » dans un projet à deux sessions).",
             ["Projet mono Ha/OIII", "Clic droit sur la Calibration d'OIII > Pause vérif", "Ctrl+Z, Ctrl+Y",
              "Ouvrir le Journal"])
    ex.w.b_log.setChecked(False)


def n18_failure_detail_lost():
    from explore2_run_explore import _fail_with
    fresh("session")
    w = ex.w
    place("stacking")
    d = _fail_with(RuntimeError("worker crashed: CUDA error 999 (unknown error)"))
    b = next((x for x in d.findChildren(QPushButton) if x.isVisible() and x.text() == "Voir journal"), None)
    if b is not None:
        ex.click(b)
        ex.settle(8)
    con = w.logp.console.toPlainText()
    hist = [x for _t, x in w.logp.rows()][:2]
    ex.close_dialogs()
    d2 = _fail_with(FileNotFoundError(2, "No such file or directory", "G:/Astro/M31/lights"))
    nf = [x for x in texts(d2) if "introuvable" in x]
    ex.close_dialogs()
    if "999" not in con and b is not None:
        note("gênant", "Démarrage impossible : la cause réelle n'est écrite nulle part, « Voir journal » redit la même "
             "phrase",
             "Un démarrage qui échoue sur une erreur imprévue (ici « worker crashed: CUDA error 999 », simulée) : la "
             "fenêtre dit « Exécution non démarrée : une erreur inattendue l'en a empêchée » et propose « Voir "
             f"journal ». Le Journal ouvert : console « {con} », historique {hist} ; l'export du journal non plus "
             "ne contient pas l'erreur. Le détail technique est perdu : impossible de savoir ce qui s'est passé. "
             f"Autre cas : un dossier de brutes absent (G:/Astro/M31/lights) donne « {nf[0] if nf else '?'} » sans "
             "dire quel dossier, alors que ce n'est pas un dossier du projet.",
             ["Session → Empilement", "Exécuter (le démarrage échoue)", "« Voir journal »"])


def n19_stop_while_starting():
    import time as _t
    from astro_gpu.core import runner2
    from PySide6.QtWidgets import QFrame
    fresh("session")
    w = ex.w
    place("stacking")

    def slow(*a, **k):
        _t.sleep(4)
        raise OSError(28, "No space left on device")
    old = runner2._spawn_worker
    runner2._spawn_worker = slow
    try:
        ex.click(w.bt_run)
        ex.settle(5)
        rs = next((x for x in w.findChildren(QFrame) if x.__class__.__name__ == "RunStatus" and x.isVisible()), None)
        d = ex.w.run.dialog
        info = {"bar": (rs.b_stop.isEnabled(), rs.b_stop.toolTip()) if rs else None,
                "window": (d.b_stop.isEnabled(), d.b_stop.toolTip()) if d else None,
                "menu": w.a_stop.isEnabled(), "hint": [x for x in texts(d) if "Fenêtre" in x or "fenêtre" in x]}
        shot_f = ex.shot(f"{len(ex.notes()) + 1:03d}_stop_while_starting")
        shot_d = ex.shot(f"{len(ex.notes()) + 2:03d}_hint_window_closed", d) if d else None
        w.a_stop.trigger()
        ex.settle(5)
        phase = ex.w.run.phase
        t0 = _t.time()
        while ex.w.run.running and _t.time() - t0 < 20:
            ex.settle(5)
    finally:
        runner2._spawn_worker = old
    P("  info", info, phase)
    if info["bar"] and not info["bar"][0] and info["menu"] and phase == "starting":
        note("mineur", "Pendant le démarrage : « Arrêter » grisé sans raison dans la fenêtre et en bas, actif mais sans "
             "effet dans le menu Exécuter",
             f"Démarrage lent (simulé : 4 s). Bouton Arrêter de la barre d'état : {info['bar']} ; de la fenêtre "
             f"d'exécution : {info['window']} (grisés, sans info-bulle). Menu Exécuter > Arrêter : actif ; le choisir "
             "ne fait rien (toujours « Démarrage… »). Si le démarrage traîne, on ne peut ni arrêter ni savoir pourquoi.",
             ["Session → Empilement", "Exécuter (démarrage lent)", "Regarder Arrêter en bas, dans la fenêtre, dans le "
              "menu Exécuter"], shot_f)
    if info["hint"] and shot_d:
        note("mineur", "Fenêtre d'exécution : « Fenêtre fermée : l'exécution continue » écrit dans la fenêtre ouverte",
             f"Pendant une exécution, la fenêtre (ouverte) affiche « {info['hint'][0]} ». Raccourcie, la phrase "
             "affirme que la fenêtre est fermée ; elle voulait dire « fermer cette fenêtre n'arrête pas l'exécution ».",
             ["Session → Empilement", "Exécuter", "Lire la fenêtre pendant le démarrage"], shot_d)


def n20_pauses_two_flows():
    fresh("two_flows")
    for sid in (1, 2):
        ex.click_block(next(x for x in ex.blocks() if x.node.key == "session" and x.node.sid == sid))
        place("calibration", "registration", "stacking", "gradient")
    p1, p2 = ex.s.paths[0].pid, ex.s.paths[1].pid
    n1, n2 = ex.s.paths[0].name, ex.s.paths[1].name
    ex.right_click_block(blk("empilement", p1), "Pause vérif")
    ex.right_click_block(blk("alignement", p2), "Pause vérif")
    ex.settle(5)
    ex.close_dialogs()
    ex.w.run.dialog = None
    d = shift_f5("fond", p1)              # its own plan, for comparison
    own = [r.name.text() for r in d.rows] if d else []
    ex.close_dialogs()
    ex.w.run.dialog = None
    d = f5()
    rows = [(r.name.text(), r.who.text()) for r in d.rows] if d else []
    notes_ = (d.pause_note.text(), d.pause_more.text()) if d else ("", "")
    if d is not None and not any(w == n1 for _r, w in rows):
        note("gênant", "Tout exécuter avec une pause dans chacun de deux flux : seul le flux à la pause la plus en "
             "amont avance",
             f"Deux flux : « {n1} » (pause après Empilement) et « {n2} » (pause après Alignement), chacun Calibration → "
             f"Alignement → Empilement → Gradient. F5 (Tout exécuter) ne prévoit que {rows} : rien de « {n1} », "
             f"alors que Maj+F5 dans « {n1} » prévoit bien {own} jusqu'à sa pause. La fenêtre dit « {notes_[0]} » "
             f"et « {notes_[1]} » : la pause « ensuite » est celle de l'autre flux (« {n1} »), alors que la phrase "
             f"la range sous « {n2} ».",
             ["Projet à deux flux (Session 1, Session 2)", "Dans chacun : Calibration, Alignement, Empilement, Gradient",
              "Pause vérif sur l'Empilement du premier, sur l'Alignement du second", "F5"], d)
    ex.close_dialogs()


def n21_log_gender():
    fresh("session_empty_wf")
    place("calibration", "registration", "selection", "stacking", "astrometry", "color")
    rows = [x for _t, x in ex.w.logp.rows()]
    bad = [r for r in rows if r.split(" ajouté ")[0] in ("Calibration", "Sélection", "Astrométrie", "Couleur")]
    ex.w.b_log.setChecked(True)
    ex.settle(5)
    if bad:
        note("mineur", "Journal : « Astrométrie ajouté », « Calibration ajouté » : accord au masculin",
             f"Lignes du Journal : {rows[:6]}. Les blocs féminins (Calibration, Sélection, Astrométrie, Couleur, "
             "Fusion…) sont « ajouté ». Une tournure sans accord éviterait la faute (« Ajout : Astrométrie », "
             "« Astrométrie : ajout à « Workflow » »).",
             ["Session + workflow vide", "Placer Calibration, Alignement, Sélection, Empilement, Astrométrie, "
              "Couleur", "Ouvrir le Journal"])
    ex.w.b_log.setChecked(False)


def n22_fusion_broken_with_plan_open():
    fresh("mono")
    w = ex.w
    place("fusion")
    ha, oiii = ex.s.paths[0].pid, ex.s.paths[1].pid
    for pid in (ha, oiii):
        ex.click_block(blk("empilement", pid))
        place("astrometry")
    d = shift_f5("fusion")
    ex.click_block(blk("astrometrie", ha))
    ex.key(Qt.Key_Delete)
    ex.settle(10)
    d2 = ex.w.run.dialog
    ok = d2 is not None and d2.isVisible() and d2.b_run.isEnabled()
    if ok:
        shot_f = ex.shot(f"{len(ex.notes()) + 1:03d}_fusion_broken_plan_open", d2)
        ex.click(d2.b_run)
        ex.settle(25)
        d3 = ex.w.run.dialog
        if ex.w.run.track is None and d3 is not None and d3.summary.text() == d2.summary.text():
            note("gênant", "Fusion rendue impossible fenêtre ouverte : Exécuter reste actif et le clic ne fait rien",
                 "Projet mono avec Astrométrie dans Ha et OIII, Fusion posée. Maj+F5 sur la Fusion ouvre la fenêtre "
                 f"(« {d3.summary.text()} », Exécuter actif). Pendant qu'elle est ouverte, Suppr sur l'Astrométrie de Ha : "
                 "le bouton Exécuter du haut devient grisé (« HOO bloquée : Ha : requiert Astrométrie ») mais la fenêtre "
                 "ne change pas ; son Exécuter reste actif et un clic ne lance rien et ne dit rien.",
                 ["Projet mono Ha/OIII, Astrométrie dans les deux, Fusion posée", "Clic sur Fusion, Maj+F5",
                  "Fenêtre ouverte : clic sur l'Astrométrie de Ha, Suppr", "Clic sur Exécuter dans la fenêtre"], shot_f)
    ex.close_dialogs()


if __name__ == "__main__":
    run_checks(dict(globals()), prefix="n", dump_name="confirm_dump.json")
