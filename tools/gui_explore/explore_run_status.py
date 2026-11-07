"""run_status: running, without ever computing. The Run button and its device combo, the run window (plan, Run to here,
a start that fails), pauses for review, the status line, the health check, Inspect of blocks never computed, the Log.

Each anomaly is reproduced here (a second time: the dumps of explore_run_observe*.py were the first) and noted only when
its condition holds. Notes in French, for the user to judge.

    python explore_run_status.py [check ...]
"""
from __future__ import annotations

import faulthandler
import json
import sys
import time
import traceback

from harness import TMP, Explorer, _slug, stray_qt_errors
from PySide6.QtWidgets import QAbstractButton, QLabel, QPushButton

faulthandler.dump_traceback_later(1500, exit=True)
stray_qt_errors()
ex = Explorer("run_status", lang="fr", size=(1600, 950))
EXC: list[str] = []
DONE: set[str] = set()


def hook(t, v, tb):
    EXC.append("".join(traceback.format_exception(t, v, tb)))
    print("!! UNCAUGHT", EXC[-1][-800:], flush=True)


sys.excepthook = hook


def P(*a):
    print(*a, flush=True)


def note(sev, title, detail, steps, widget=None):
    """ex.note, with the screenshot of ``widget`` (a dialog) when given."""
    if title in DONE:
        return
    DONE.add(title)
    if "--dry" in sys.argv:
        print("WOULD NOTE", sev, "|", title, "\n   ", detail[:600], flush=True)
        return
    if widget is None:
        ex.note(sev, title, detail, shot=True, steps=steps)
        return
    f = ex.shot(f"{len(ex.notes()) + 1:03d}_{_slug(title)}", widget)
    rec = {"area": ex.area, "severity": sev, "title": title, "detail": detail, "steps": steps, "shot": f,
           "lang": ex.lang, "t": time.strftime("%H:%M:%S")}
    with ex.log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("ANOMALY", sev, title, flush=True)


def texts(w):
    out = []
    for x in w.findChildren(QLabel) + w.findChildren(QAbstractButton):
        try:
            if x.isVisible() and x.text():
                out.append(x.text())
        except RuntimeError:
            pass
    return out


def blk(ref, pid=None):
    return next((x for x in ex.blocks() if x.node.ref == ref and
                 (pid is None or (x.node.paths and x.node.paths[0].pid == pid))), None)


def place(*blocks):
    for b in blocks:
        ex.double_click_card(b)
        ex.settle(12)


def fresh(kind):
    root = ex.project(kind)
    ex.open(root)
    ex.settle(15)
    ex.close_dialogs()


def here(ref, pid=None):
    """Select a block with a click and Run to here (what Shift+F5 calls)."""
    it = blk(ref, pid)
    ex.click_block(it)
    d = ex.w.run.run_to_here()
    ex.settle(8)
    return d


def run_all_window():
    d = ex.w.run.launch()
    ex.settle(8)
    return d


def rows(d):
    return [r.name.text() for r in d.rows] if d is not None else []


def inspect(ref, pid=None):
    ex.double_click_block(blk(ref, pid))
    ex.settle(15)
    page = ex.w.tune.pages.currentWidget()
    btns = [b for b in page.findChildren(QPushButton) if b.isVisible() and b.text().startswith("Exécuter jusqu")]
    return page, btns


def back():
    ex.w.set_mode(0)
    ex.settle(6)


PAUSE = "Pause ici"
MSG_NO_DATA = "pas de raison"


# ---------------------------------------------------------------------------------------------- checks
def c_stack_only_known():
    fresh("session")
    place("stacking")
    d = run_all_window()
    r = rows(d)
    if "Calibration" in r and "Alignement" in r and r.index("Empilement") < r.index("Calibration"):
        note("mineur", "Déjà connu : Calibration et Alignement non posés listés dans la fenêtre d'exécution, après "
             "Empilement",
             f"Graphe Session 1 → Empilement seulement. La fenêtre « Exécuter le projet » liste : {', '.join(r)}. "
             "Les blocs que le moteur calcule sans qu'ils soient posés apparaissent sous leur nom, et dans le "
             "désordre (Empilement en premier). Même chose dans le récapitulatif d'échec (« Non démarrés : "
             "Empilement, Calibration, Alignement. ») et dans l'info-bulle du bouton « Exécuter jusqu'ici » "
             "d'Inspection (« Calibration, Alignement, Empilement »). Déjà connu.",
             ["Projet avec une session de brutes", "Double-clic sur Empilement dans la palette",
              "Exécuter (ou Maj+F5 sur Empilement)"], d)
    ex.close_dialogs()


def c_failed_start():
    fresh("session")
    place("stacking")
    ex.click(ex.w.bt_run)
    ex.settle(30)
    d = ex.w.run.dialog
    t = texts(d) if d is not None else []
    hist = [x for _t, x in ex.w.logp.rows()]
    no_hist = not any("exécution" in x.lower() or "échec" in x.lower() for x in hist)
    raw = any("runs are blocked" in x for x in t)
    if d is not None and ("En échec" in t) and raw:
        note("gênant", "Démarrage impossible : la fenêtre dit « En échec » avec le message brut en anglais, rien "
             "dans l'historique du Journal",
             "Quand le lancement ne peut pas démarrer (ici bloqué exprès par l'exploration ; en vrai un disque plein, "
             "un droit refusé…), la fenêtre affiche « En échec » en rouge, « Non démarrés : Empilement, Calibration, "
             "Alignement. » puis la raison telle que l'erreur Python la donne, en anglais (« exploration: runs are "
             "blocked (no GPU, no computation) »). Elle ne dit pas en clair « L'exécution n'a pas pu démarrer » "
             "(seule la console du Journal le dit). L'historique du Journal n'a aucune ligne pour cet échec "
             f"(lignes : {hist[:3]}), la barre d'état ne dit rien. Seule la présentation est jugée ici.",
             ["Projet avec une session, Empilement posé", "Clic sur Exécuter (barre du haut)",
              "Regarder la fenêtre, la barre d'état, le Journal"], d)
    con = ex.w.logp.console.toPlainText()
    if "5 étapes" in con and d is not None and len(d.rows) == 3:
        note("mineur", "La console compte des « étapes » (5) quand la fenêtre montre 3 blocs",
             f"Console du Journal : « {con.splitlines()[0][10:]} ». La fenêtre d'exécution montre 3 lignes (Empilement, "
             "Calibration, Alignement). Le mot « étapes » et ce nombre sont internes (étapes du moteur : inventaire, "
             "préparation du rendu…).",
             ["Session → Empilement", "Exécuter", "Ouvrir le Journal (bouton en bas à droite)"])
    ex.close_dialogs()


def c_repeated_run_to_here():
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "gradient", "denoise", "gradient", "denoise")
    d = here("debruitage~2")
    empty = d is not None and not d.plan.jobs
    state = blk("debruitage~2").toolTip()
    if empty:
        note("gênant", "Exécuter jusqu'ici sur « Débruitage 2 » / « Gradient 2 » : « Rien à exécuter » alors que rien "
             "n'est calculé",
             "Graphe avec deux Gradient et deux Débruitage (numérotés 1 et 2). Maj+F5 (Exécuter jusqu'au bloc "
             "sélectionné) sur « Débruitage 2 » ouvre « Exécuter « Workflow » jusqu'à Débruitage 2 » avec « Rien à "
             "exécuter » et Exécuter grisé, alors que le bloc est « Pas encore calculé » et qu'« Exécuter » (tout) le "
             f"calcule. Même chose sur « Gradient 2 ». Info-bulle du bloc : {state!r}.",
             ["Session + workflow vide", "Placer Calibration, Alignement, Empilement, Gradient, Débruitage, Gradient, "
              "Débruitage (double-clic palette)", "Clic sur « Débruitage 2 », Maj+F5"], d)
    ex.close_dialogs()
    page, btns = inspect("debruitage~2")
    if btns:
        b = btns[0]
        txt, tip = b.text(), b.toolTip()
        ex.click(b)
        ex.settle(8)
        d2 = ex.w.run.dialog
        if d2 is not None and not d2.plan.jobs and "min" in txt:
            note("gênant", "Inspection de « Débruitage 2 » : « Exécuter jusqu'ici · ≈ 7 min » ouvre une fenêtre "
                 "« Rien à exécuter »",
                 f"Le bouton d'Inspection de Débruitage 2 annonce « {txt} » (info-bulle : « {tip} » : il vise en "
                 "fait Débruitage 1, sans Débruitage 2). Un clic ouvre « Exécuter « Workflow » jusqu'à Débruitage 2 » "
                 "avec « Rien à exécuter ». Les deux ne disent pas la même chose, et aucun des deux ne calcule "
                 "Débruitage 2.",
                 ["Graphe avec deux Débruitage", "Double-clic sur « Débruitage 2 » (Inspection)",
                  "Clic sur « Exécuter jusqu'ici »"], d2)
        if "gradient~2" in tip:
            note("mineur", "Clé interne « gradient~2 » dans l'info-bulle d'« Exécuter jusqu'ici »",
                 f"Inspection de Débruitage (1 ou 2) : l'info-bulle du bouton « Exécuter jusqu'ici » liste « {tip} ». "
                 "« gradient~2 » est une clé interne ; le graphe dit « Gradient 2 », et « Gradient » y désigne "
                 "« Gradient 1 ».",
                 ["Graphe avec deux Gradient puis un Débruitage", "Double-clic sur Débruitage",
                  "Survoler « Exécuter jusqu'ici »"])
        ex.close_dialogs()
    back()
    # the names of the first copy in Inspect
    page, _b = inspect("fond")
    head = texts(page)[0] if texts(page) else ""
    strip = [x for x in texts(ex.w.tune.strip) if x.startswith("Gradient")]
    if head.startswith("Gradient  ·") and "Gradient 1" not in strip:
        note("gênant", "Bloc répété : « Gradient 1 » au graphe, « Gradient » dans Inspection, les pauses et le Journal",
             f"Le graphe, le panneau et la fenêtre d'exécution disent « Gradient 1 » et « Gradient 2 ». Inspection : "
             f"en-tête « {head} », bande « {', '.join(strip)} ». La pause posée sur Gradient 1 dit « L'exécution "
             "s'arrêtera après Gradient… », le Journal « Pause définie après Gradient », la fenêtre « S'arrête après "
             "Gradient… » : on ne sait pas lequel des deux. Même chose pour Débruitage 1.",
             ["Graphe avec deux Gradient", "Double-clic sur « Gradient 1 »", "Regarder l'en-tête et la bande"])
    back()


def c_repeated_pause_crash():
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "gradient", "gradient")
    n = len(EXC)
    m = ex.right_click_block(blk("fond~2"), PAUSE)
    ex.settle(10)
    if len(EXC) > n:
        note("bloquant", "« Pause ici pour vérification » sur « Gradient 2 » : erreur Python (ValueError)",
             "Le menu du bloc « Gradient 2 » (ou « Débruitage 2 ») propose « Pause ici pour vérification ». Le "
             "choisir lève une erreur non rattrapée ; dans l'application, elle s'affiche en boîte d'erreur brute. "
             "Aucune pause n'est posée, et le message d'avant reste affiché.\n" + EXC[-1][-700:],
             ["Graphe avec deux Gradient", "Clic droit sur « Gradient 2 »", "« Pause ici pour vérification »"])


def c_session_calibration():
    fresh("two_sessions")
    place("calibration", "registration", "stacking")
    d = run_all_window()
    r = rows(d)
    if any("@" in x for x in r):
        tips = [x.toolTip() for x in d.rows if "@" in x.toolTip() or "sessions:" in x.toolTip()]
        note("gênant", "Projet à deux sessions : la fenêtre d'exécution montre « inventory@2 », « inventory@1 »",
             f"Calibration posée avec deux sessions (une Calibration par session au graphe). Lignes de la fenêtre : "
             f"{', '.join(r)}. Info-bulles : {tips}. Les deux Calibration s'appellent des clés internes ; "
             "« sessions » est un nom d'étape. Même chose en anglais, dans le récapitulatif d'échec (« Non démarrés : "
             "…, inventory@2, inventory@1. ») et dans l'info-bulle « Exécuter jusqu'ici » d'Inspection "
             "(« calibration@2, calibration@1, Alignement, Empilement »).",
             ["Projet avec deux sessions de brutes et un workflow vide", "Placer Calibration, Alignement, Empilement",
              "Exécuter"], d)
    if r.count("Session") >= 2:
        note("mineur", "Les blocs Session apparaissent « à jour » dans la fenêtre d'exécution, sous le nom « Session »",
             f"Lignes : {', '.join(r)}. Les deux Session (au graphe « Session 1 », « Session 2 » ; « Ha », « OIII » "
             "dans un projet mono) sont des lignes cochées « à jour » nommées « Session ». Une Session ne se calcule "
             "pas ; ici elle prend deux lignes et pas son nom.",
             ["Projet à deux sessions (ou mono Ha/OIII)", "Placer Calibration, Alignement, Empilement", "Exécuter"], d)
    ex.close_dialogs()
    d = here("calibration@1")
    if d is not None and not d.plan.jobs:
        dd = d
        page, btns = inspect("calibration@1")
        t = texts(page)
        st = next((x for x in t if x in ("À jour", "Pas encore calculé")), "?")
        note("gênant", "Calibration d'une session : Maj+F5 dit « Rien à exécuter », Inspection dit « " + st + " »",
             f"Projet à deux sessions, Calibration posée : Maj+F5 sur la Calibration de Session 1 ouvre « Exécuter "
             "« Workflow » jusqu'à Calibration » avec « Rien à exécuter ». Rien n'a été calculé (la fenêtre "
             "« Exécuter » tout liste bien les calibrations). En Inspection, la même Calibration affiche "
             f"« {st} » et un bouton « Exécuter jusqu'ici » grisé sans raison ({[b.isEnabled() for b in btns]}). Dans "
             "un projet mono (Ha, OIII), Inspection dit « Calibration n'est pas calculé pour ce workflow. »",
             ["Projet à deux sessions", "Placer Calibration, Alignement, Empilement",
              "Clic sur la Calibration de Session 1, Maj+F5", "Double-clic dessus"], dd)
        back()
    ex.close_dialogs()
    # a pause on a session's Calibration
    it = blk("calibration@1")
    paths = [q.name for q in it.node.paths]
    m = ex.right_click_block(it, PAUSE)
    ex.settle(8)
    has = any(PAUSE.lower() in (a.get("text") or "").lower() for a in m)
    bps = [w.breakpoints for w in ex.s.p.workflows]
    if has and not any(bps) and getattr(blk("calibration@1"), "bp", None) is None:
        note("gênant", "« Pause ici pour vérification » sur la Calibration d'une session ne fait rien",
             f"Le menu de la Calibration de Session 1 propose « Pause ici pour vérification ». Le choisir ne pose "
             f"aucune pause (pauses des workflows : {bps}), aucune marque sur le bloc, aucun message. Workflows du "
             f"bloc vus par le menu : {paths}.",
             ["Projet à deux sessions, Calibration posée", "Clic droit sur la Calibration de Session 1",
              "« Pause ici pour vérification »"])


def c_empty_workflows():
    fresh("two_flows")
    w = ex.w
    line, tip = w.status.text(), w.bt_run.toolTip()
    if "à jour" in line:
        note("gênant", "Workflows sans bloc : « tout est à jour » alors que rien n'est calculé",
             f"Deux flux, chacun une Session de brutes sans aucun bloc. Barre d'état : « {line} ». Bouton Exécuter "
             f"grisé, info-bulle « {tip} ». Rien n'a jamais été calculé : il faudrait dire qu'il n'y a pas encore "
             "de bloc à calculer (même chose avec une session et un workflow vide).",
             ["Projet avec deux flux (Session 1, Session 2) sans bloc", "Regarder Exécuter et la barre d'état"])
    if w.a_here.isEnabled() and ex.s.sel_step not in ("sources", "final") and blk(ex.s.sel_step) is None:
        d = w.run.run_to_here()
        ex.settle(6)
        note("gênant", "Sélection fantôme : « Exécuter jusqu'au bloc sélectionné » vise un Empilement qui n'existe pas",
             f"À l'ouverture du projet (deux flux sans bloc), le menu Exécuter garde « Exécuter jusqu'au bloc "
             f"sélectionné » actif ; le bloc sélectionné est « {ex.s.sel_step} », absent du graphe. Maj+F5 ouvre "
             f"« {d.windowTitle() if d else '?'} » avec « Rien à exécuter ».",
             ["Ouvrir un projet à deux flux sans bloc", "Maj+F5"], d)
        ex.close_dialogs()


def c_session_no_wf():
    fresh("session")
    w = ex.w
    if not w.bt_run.isEnabled() and not w.bt_run.toolTip():
        note("mineur", "Exécuter grisé sans info-bulle : une session de brutes, pas encore de workflow",
             f"Une Session avec 15 brutes, aucun bloc. Exécuter est grisé, sans info-bulle ; la barre d'état dit "
             f"« {w.status.text()} ». Un bouton grisé devrait dire pourquoi (placer un bloc).",
             ["Projet avec une session de brutes, sans workflow", "Survoler Exécuter"])


def c_no_lights():
    ex.new_project("SansBrutes")
    ex.settle(10)
    place("session", "stacking")
    d = here("empilement")
    if d is not None and d.plan.jobs and not d.b_run.isEnabled():
        t = texts(d)
        note("gênant", "Sans brutes, Maj+F5 montre « ≈ 3 h 02 · disque 18,9 Go » et Exécuter grisé sans dire pourquoi",
             f"Projet vide, Session placée (sans brutes), Empilement placé. Maj+F5 sur Empilement : {t}. Les durées "
             "et la place disque sont inventées (aucune brute), Exécuter est grisé et la fenêtre ne dit pas qu'il "
             "faut ajouter les brutes. En Inspection, le bouton « Exécuter jusqu'ici · ≈ 3 h 02 » est grisé (la page "
             "dit « Pas encore d'images. »).",
             ["Nouveau projet vide", "Double-clic Session puis Empilement dans la palette", "Maj+F5"], d)
    ex.close_dialogs()


def c_image_without_file():
    fresh("session")
    place("stacking")
    z = ex.new_flow_zone()
    ex.drag_palette("image", ex.ghost_center(z))
    ex.settle(15)
    d = run_all_window()
    t = texts(d) if d is not None else []
    eng = [x for x in t if "has no file yet" in x]
    if eng:
        note("gênant", "Fenêtre d'exécution : la raison « its Image block has no file yet » en anglais",
             f"Une Session → Empilement et une Image sans fichier (nouveau flux). La fenêtre « Exécuter le projet » "
             f"dit : « {eng[0]} ». La raison n'est pas traduite.",
             ["Projet avec une session, Empilement posé", "Glisser une Image sur « Nouveau flux »", "Exécuter"], d)
    ex.close_dialogs()
    ex.new_project("ImageSansFichier")
    ex.settle(10)
    place("image", "astrometry")
    ex.close_dialogs()
    page, btns = inspect("astrometrie")
    if btns and not btns[0].isEnabled() and not btns[0].toolTip() and "min" in btns[0].text():
        note("mineur", "Inspection, Image sans fichier : « Exécuter jusqu'ici · ≤ 1 min » grisé sans info-bulle",
             f"Image placée sans fichier, Astrométrie après. Inspection d'Astrométrie : bouton « {btns[0].text()} » "
             "grisé, sans info-bulle, avec une durée alors que rien ne peut être calculé. La raison est écrite plus "
             "bas (« Pas encore de fichier image : choisissez-le dans le bloc Image. »), pas sur le bouton.",
             ["Nouveau projet vide", "Double-clic Image puis Astrométrie dans la palette",
              "Double-clic sur Astrométrie"])
    back()


def c_split_named_output():
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "astrometry", "gradient", "color", "denoise", "stars")
    d = here("etoiles")
    r = rows(d)
    if any(x.startswith("Sortie") for x in r) and not any(x.startswith("Split") for x in r):
        tip = next((x.toolTip() for x in d.rows if x.row.title.startswith("Sortie")), "")
        note("gênant", "Le bloc « Split étoiles et fond » s'appelle « Sortie « Workflow » » dans la fenêtre d'exécution",
             f"Titre : « {d.windowTitle()} », lignes : {', '.join(r)}. La dernière ligne regroupe « {tip} » sous le "
             "nom « Sortie « Workflow » », alors que le graphe montre le bloc Split étoiles et fond (la sortie n'est "
             "pas un bloc à l'écran). Même nom dans le récapitulatif d'échec.",
             ["Workflow complet jusqu'à Split étoiles et fond", "Clic sur Split, Maj+F5"], d)
    ex.close_dialogs()
    if blk("etoiles").title.startswith("Split"):
        note("gênant", "Mot anglais « Split » dans l'interface française",
             "Le bloc s'appelle « Split étoiles et fond » au graphe, dans la palette (« VERS LE POST »), la fenêtre "
             "d'exécution (« Exécuter « Workflow » jusqu'à Split étoiles et fond »), les messages de pause et le "
             "Journal (« Split étoiles et fond ajouté à « Workflow » »).",
             ["Placer le bloc depuis la palette", "Regarder son nom"])
    d = here("composition")
    if d is not None and not d.plan.jobs:
        note("gênant", "Maj+F5 sur « Combiner les calques étoiles et fond » : « Rien à exécuter » sans rien calculé",
             f"Rien n'est calculé. Maj+F5 sur le bloc de Composition ouvre « {d.windowTitle()} » avec « Rien à "
             "exécuter ». Il faudrait calculer jusqu'au Split dont il dépend, ou dire pourquoi ce n'est pas possible.",
             ["Workflow avec Split étoiles et fond (le bloc Combiner est créé)", "Clic sur Combiner…, Maj+F5"], d)
    ex.close_dialogs()


def c_branch_log():
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
    eng = [x for _t, x in ex.w.logp.rows() if x.startswith("Branch")]
    if eng:
        ex.w.b_log.setChecked(True)
        ex.settle(5)
        note("gênant", "Journal : « Branch \"Workflow · Gradient\" made » en anglais",
             f"Une branche faite en déposant Gradient sous Astrométrie : le Journal (interface en français) écrit "
             f"« {eng[0]} ».", ["Workflow jusqu'à Astrométrie", "Glisser Gradient sous Astrométrie (nouvelle branche)",
                                "Ouvrir le Journal"])
        ex.w.b_log.setChecked(False)
    d = here("post")
    if d is not None:
        stages = [j.stage for j in d.plan.jobs]
        if "astrometry" not in stages and blk("astrometrie") is not None:
            note("question", "Branche : « Exécuter jusqu'à Sortie » du premier workflow ne calcule pas l'Astrométrie "
                 "posée avant",
                 f"Après la branche, le premier workflow finit par Astrométrie puis sa Sortie (bloc « Sortie » au "
                 f"graphe). Maj+F5 sur cette Sortie : lignes {rows(d)} ; l'Astrométrie placée juste avant n'en fait "
                 "pas partie (« Exécuter » tout la calcule).", ["Faire la branche ci-dessus",
                                                               "Clic sur la Sortie du premier workflow, Maj+F5"], d)
    ex.close_dialogs()


def c_device():
    from astro_gpu.client import prefs
    fresh("session")
    place("stacking")
    w = ex.w
    prefs.set("device", "auto")
    w.set_device("auto")
    ex.settle(6)
    auto = w.bt_run.text()
    shown = w.device_cb.currentText()
    i = w.device_cb.findData("cpu")
    w.device_cb.setCurrentIndex(i)
    w.device_cb.activated.emit(i)
    ex.settle(8)
    cpu = w.bt_run.text()
    i = w.device_cb.findData("cuda:0")
    w.device_cb.setCurrentIndex(i)
    w.device_cb.activated.emit(i)
    ex.settle(8)
    g0 = w.bt_run.text()
    if auto == cpu and auto != g0:
        note("gênant", f"Durée : « {auto} » en Automatique comme sur le CPU, « {g0} » sur la même {shown}",
             f"Session → Empilement. Avec le réglage Automatique (par défaut), la liste à gauche de Exécuter montre "
             f"« {shown} » et le bouton dit « {auto} ». En choisissant le CPU : « {cpu} ». En choisissant {shown} "
             f"elle-même : « {g0} ». L'estimation en Automatique est celle du CPU, alors que le calcul se fera sur "
             "la carte graphique.", ["Session → Empilement", "Lire Exécuter", "Choisir CPU puis la RTX 4090 dans la "
                                     "liste à gauche de Exécuter"])
    items = [w.device_cb.itemText(k) for k in range(w.device_cb.count())]
    seen = {}

    def look(d):
        from PySide6.QtWidgets import QComboBox
        seen["c"] = [[c.itemText(k) for k in range(c.count())] for c in d.findChildren(QComboBox) if c.isVisible()]
        d.reject()
    ex.dialog_handler = look
    w.open_settings("compute")
    ex.settle(8)
    sett = (seen.get("c") or [[]])[0]
    if "Automatique" in sett and "Automatique" not in items:
        note("mineur", "Liste GPU/CPU et Paramètres > Calcul ne proposent pas les mêmes choix",
             f"Liste à gauche de Exécuter : {items}. Paramètres > Calcul : {sett}. « Automatique » n'est pas dans la "
             "liste du haut (elle montre la plus grosse carte, sans dire que c'est automatique) ; y choisir une "
             "carte quitte Automatique pour de bon. « CPU » d'un côté, « CPU seul » de l'autre.",
             ["Ouvrir un projet", "Ouvrir la liste à gauche de Exécuter", "Édition > Paramètres… > Calcul"])
    w.set_device("auto")


def c_health():
    """FITS lights and one JPEG: the warning in the panel, not before the run, and the health check never read."""
    import numpy as np
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    from PIL import Image
    name = "fits_jpg_note"
    root = ex.ws / name
    p = project2.create(root, name, Target(name="M 31"))
    p.stack_mode = "per_session"
    dd = ex.fits_frames(f"{name}/lights", 8)
    j = TMP / "data" / name / "phone" / "IMG_0001.JPG"
    j.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(j)
    p.sessions.append(project2.Session(1, "Session 1", lights=[str(dd), str(j)]))
    project2.save(p)
    ex.open(root)
    ex.settle(20)
    place("stacking")
    warned = blk("sources").summary.startswith("⚠")
    d = run_all_window()
    t = texts(d) if d is not None else []
    if warned and d is not None and not any("JPG" in x or "erreur" in x or "type" in x for x in t):
        note("gênant", "Fichier d'un autre type : pas d'avertissement dans la fenêtre d'exécution",
             f"La Session contient 8 FITS et un JPEG. Le bloc Session et son panneau le signalent (⚠, « IMG_0001.JPG "
             f"n'est pas une pose que le moteur lit… »). La fenêtre « Exécuter le projet » n'en dit rien : {t}. La "
             "règle : averti sur la Session, dans son panneau, avant une exécution.",
             ["Session de FITS avec un JPEG ajouté par erreur", "Placer Empilement", "Exécuter"], d)
    ex.close_dialogs()
    ex.double_click_block(blk("sources"))
    for _ in range(30):
        ex.settle(5)
    t = texts(ex.w.tune.pages.currentWidget())
    if "Lecture des images…" in t and ex.w.frames.data is None:
        note("gênant", "Vérifications bloquées sur « Lecture des images… » quand une brute ne se lit pas",
             "Session de FITS avec un JPEG (ou des brutes illisibles, cas « 14 brutes et un JPEG »). Inspection de la "
             "Session (ou « Revoir les images… » du panneau) : la section Vérifications reste « Lecture des "
             "images… » (attendu plus de 15 s) ; la lecture échoue sans rien dire. L'avertissement sur le fichier "
             "d'un autre type n'apparaît donc jamais dans les Vérifications.",
             ["Session de FITS avec un JPEG", "Double-clic sur la Session (Inspection)", "Attendre"])
    back()
    ex.click_block(blk("sources"))
    ex.settle(8)
    used = [x for x in texts(ex.page.panel) if "utilisées" in x]
    if used and "9 / 9" in used[0]:
        note("mineur", "Panneau Session : « Brutes · 9 / 9 utilisées » compte le JPEG",
             f"8 FITS et un JPEG signalé comme ajouté par erreur : le panneau dit « {used[0]} ». Le JPEG n'est pas "
             "une pose que le moteur lit ; il est compté comme utilisé.", ["Session de FITS avec un JPEG",
                                                                           "Clic sur la Session", "Lire le panneau"])


def c_strip():
    fresh("image")
    ex.double_click_block(blk("image"))
    ex.settle(12)
    st = texts(ex.w.tune.strip)
    if "Images" in st and "Sortie" in st:
        b = next(x for x in ex.w.tune.strip.findChildren(QPushButton) if x.text() == "Images")
        ex.click(b)
        ex.settle(10)
        page = texts(ex.w.tune.pages.currentWidget())
        note("mineur", "Bande d'Inspection : « Images » et « Sortie » qui ne sont pas des blocs du graphe",
             f"Workflow d'une Image (fichier choisi), sans Session. Bande d'Inspection : {[x for x in st if x != '›']}. "
             f"« Images » (le bloc des poses) y est en tête ; un clic montre {page[:3]} et « Importer un dossier… ». "
             "« Sortie » y est aussi, alors que la sortie n'est pas un bloc à l'écran. Dans un projet à Session, la "
             "bande dit « Images » là où le graphe dit « Session 1 ».",
             ["Projet avec une Image", "Double-clic sur l'Image", "Clic sur « Images » dans la bande"])
    back()


def c_menu_names():
    fresh("two_sessions")
    place("calibration", "registration", "stacking")
    m = ex.w.a_here.text()
    d = here("session@1")
    if d is not None and not d.plan.jobs and "Session" in d.windowTitle():
        note("mineur", "Maj+F5 actif sur un bloc Session : « Exécuter jusqu'à Session » puis « Rien à exécuter »",
             f"Le menu Exécuter a « {m} » (Maj+F5) ; Inspection a « Exécuter jusqu'ici » : deux noms pour la même "
             f"action. Sur un bloc Session, l'action reste active et ouvre « {d.windowTitle()} » avec « Rien à "
             "exécuter » (une Session ne se calcule pas : à griser, comme sur Post).",
             ["Projet à deux sessions, blocs posés", "Clic sur Session 1", "Maj+F5"], d)
    ex.close_dialogs()


def c_tooltips_stage_names():
    fresh("image")
    ex.click_block(blk("image"))
    place("astrometry")
    d = run_all_window()
    tips = [x.toolTip() for x in d.rows] if d is not None else []
    if any(t.startswith("image:") for t in tips):
        note("mineur", "Info-bulles de la fenêtre d'exécution : noms d'étapes, « image: » en minuscule, deux-points "
             "sans espace",
             f"Info-bulles des lignes : {tips}. On y lit des étapes du moteur (« Inventaire », « Préparation du "
             "rendu », « Catalogues Gaia », « Relevé H-alpha », « Gradient du ciel 2 ») et des noms bruts "
             "(« image: », « fusion: », « sessions: »). En français il manque l'espace avant les deux-points "
             "(« Empilement: ≈ 4 min »).", ["Projet Image, Astrométrie posée", "Exécuter",
                                            "Survoler les lignes"], d)
    ex.close_dialogs()


def c_log_language():
    fresh("session")
    place("stacking", "astrometry")
    w = ex.w
    w.change_language("en")
    ex.settle(10)
    rows_en = [x for _t, x in w.logp.rows()]
    w.b_log.setChecked(True)
    ex.settle(5)
    bad = [x for x in rows_en if "Empilement" in x or "Astrométrie" in x]
    if bad:
        note("mineur", "Journal : après passage en anglais, les noms de blocs restent en français",
             f"Lignes du Journal en anglais : {rows_en[:3]}. Le nom du bloc a été écrit dans la langue d'alors.",
             ["Interface en français, placer Empilement et Astrométrie", "Édition > Paramètres > langue anglaise",
              "Ouvrir le Journal"])
    w.change_language("fr")
    ex.settle(10)
    w.b_log.setChecked(False)


def c_fusion_without_astrometry():
    fresh("mono")
    place("fusion")
    d = run_all_window()
    st = [j.stage for j in d.plan.jobs] if d is not None else []
    if d is not None and "fusion" in st and "astrometry" not in st and d.b_run.isEnabled():
        note("question", "Fusion lancée alors que les workflows Ha et OIII n'ont pas d'Astrométrie",
             "Projet mono (Ha, OIII : Calibration, Alignement, Empilement), Fusion posée (HOO). Son panneau dit : "
             "« le workflow de chaque canal a besoin de son Astrométrie ». La fenêtre d'exécution propose pourtant "
             f"le lancement sans avertissement ({rows(d)}).", ["Projet mono Ha/OIII", "Double-clic Fusion",
                                                               "Exécuter"], d)
    ex.close_dialogs()
    page, btns = inspect("calibration@1", None)
    head = texts(page)[0] if texts(page) else ""
    if head.count("Ha") >= 2:
        note("mineur", f"En-tête d'Inspection « {head} » : le nom de la session répété",
             "Calibration de la session Ha dans un projet mono : l'en-tête répète « Ha » (nom de la session, puis du "
             "workflow).", ["Projet mono Ha/OIII", "Double-clic sur la Calibration de Ha"])
    back()


def c_recap_two_flows():
    fresh("two_flows")
    for sid in (1, 2):
        ex.click_block(next(x for x in ex.blocks() if x.node.key == "session" and x.node.sid == sid))
        place("stacking")
    ex.click(ex.w.bt_run)
    ex.settle(30)
    d = ex.w.run.dialog
    t = [x for x in texts(d) if x.startswith("Non démarrés")] if d else []
    if t and "Empilement, Empilement" in t[0]:
        note("mineur", "Récapitulatif : « Non démarrés : Empilement, Empilement, … » sans dire de quel flux",
             f"Deux flux (Workflow, Flow), chacun Session → Empilement. Après un démarrage impossible : « {t[0]} ». "
             "Les noms se répètent sans le workflow.", ["Deux flux avec chacun un Empilement", "Exécuter"], d)
    ex.close_dialogs()


def c_recap_language():
    fresh("session")
    place("stacking")
    ex.click(ex.w.bt_run)
    ex.settle(30)
    ex.w.change_language("en")
    ex.settle(10)
    d = ex.w.run.dialog
    t = [x for x in texts(d) if x.startswith("Not started")] if d else []
    if len(t) >= 2:
        note("mineur", "Récapitulatif d'échec écrit deux fois après un changement de langue",
             f"Après un démarrage impossible, passer l'interface en anglais : la fenêtre d'exécution montre deux "
             f"fois le récapitulatif ({t}, et deux fois la raison).",
             ["Session → Empilement", "Exécuter (démarrage impossible)", "Édition > Paramètres > English"], d)
    ex.close_dialogs()
    ex.w.change_language("fr")
    ex.settle(10)


def c_words_no_frames():
    ex.new_project("MotsSansBrutes")
    ex.settle(10)
    place("session", "stacking")
    line = ex.w.status.text()
    page, btns = inspect("empilement")
    t = texts(page)
    msg = next((x for x in t if x.startswith("Pas encore d")), "")
    back()
    if "brutes" in line and msg == "Pas encore d'images.":
        note("mineur", "Sans brutes : « ajoutez les brutes » dans la barre d'état, « Pas encore d'images. » en "
             "Inspection",
             f"Session vide puis Empilement. Barre d'état : « {line} ». Inspection d'Empilement : « {msg} ». La page "
             "Session dit « Pas encore d'images », le bloc « brutes à ajouter ». Le même manque est dit avec deux "
             "mots (brutes, images).", ["Nouveau projet vide", "Double-clic Session puis Empilement",
                                        "Lire la barre d'état", "Double-clic sur Empilement"])


def c_top_bar_small():
    """1366 × 768: the top bar and the run window."""
    fresh("session_empty_wf")
    place("calibration", "registration", "stacking", "astrometry", "gradient", "color", "denoise", "stars")
    ex.w.resize(1366, 768)
    ex.settle(10)
    bar = ex.w.bt_run.parentWidget()
    over = ex.w.device_cb.geometry().intersects(ex.w.bt_run.geometry()) or \
        ex.w.bt_run.geometry().right() > bar.width()
    d = run_all_window()
    ex.shot("small_1366_window")
    if d is not None:
        ex.shot("small_1366_run_dialog", d)
        if d.height() > 768 or d.width() > 1366:
            note("gênant", "Fenêtre d'exécution plus grande que l'écran 1366 × 768",
                 f"Taille {d.width()} × {d.height()}.", ["Workflow complet", "Exécuter, écran 1366 × 768"], d)
    if over:
        note("gênant", "Barre du haut à 1366 × 768 : la liste GPU et Exécuter se chevauchent",
             "", ["Fenêtre 1366 × 768", "Regarder la barre du haut"])
    ex.close_dialogs()
    ex.w.resize(*ex.size)
    ex.settle(5)


CHECKS = {k[2:]: v for k, v in dict(globals()).items() if k.startswith("c_")}
for name in [a for a in sys.argv[1:] if not a.startswith("--")] or list(CHECKS):
    P(f"\n######## {name}")
    n0 = len(EXC)
    try:
        CHECKS[name]()
    except Exception:  # noqa: BLE001
        P("!! CHECK EXC")
        traceback.print_exc(file=sys.stdout)
    P("  popups", ex.popups[-4:], "uncaught", len(EXC) - n0)
    ex.popups.clear()
    ex.close_dialogs()
ex.close()
P("END", len(ex.notes()), "notes")
