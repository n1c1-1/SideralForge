"""block_actions: each suspected anomaly reproduced twice from scratch through the GUI, then noted with its screenshot.

    python explore_blocks_confirm.py [check names...]
"""
from __future__ import annotations

import faulthandler
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(int(__import__("os").environ.get("BA_TIMEOUT", "2000")), exit=True)

from explore_blocks_lib import EXC, TOASTS, drawn, overlaps, panel_info, state  # noqa: E402
from harness import Explorer  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402

ex = Explorer("block_actions")
ONLY = sys.argv[1:]
CHECKS = []
DONE: list[str] = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ------------------------------------------------------------------ gestures
def act():
    ex.w.activateWindow()
    ex.settle(3)


def place(*blocks):
    for b in blocks:
        ex.double_click_card(b)
        ex.settle(4)


def new(kind, chain=(), name=None):
    if ex.w is not None and ex.w.s is not None:
        ex.w.s._dirty = False              # a throw-away project: closed without saving (a broken one cannot be)
    ex.open(ex.project(kind, name))
    act()
    place(*chain)
    act()
    ex.s._undo.clear()                 # what the check does starts here (the building is not what is judged)
    ex.s._redo.clear()


def blk(key, pid=None, sid=None, title=None):
    if title is not None:
        return next((it for it in ex.blocks() if it.title == title), None)
    return ex.block(key, pid=pid, sid=sid)


def menu(it, pick=None):
    m = ex.right_click_block(it, pick=pick)
    ex.settle(6)
    act()
    return m


def texts(m):
    return [x.get("text") for x in m]


def pid_of(name):
    return next(q.pid for q in ex.s.paths if q.name == name)


def graph_of(name="Workflow"):
    q = next(q for q in ex.s.paths if q.name == name)
    return [n["block"] + ("/off" if n["off"] else "") for n in ex.s.graph(q)]


def toasts_since(n):
    return TOASTS[n:]


CHAIN = ["calibration", "registration", "selection", "stacking", "astrometry", "gradient", "denoise", "color", "stars"]
SHORT = ["calibration", "registration", "stacking", "astrometry", "gradient", "color"]


def run(fn):
    """Twice from scratch: noted only when both runs show it."""
    got = []
    for i in range(2):
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            r = None
            print("  script error", fn.__name__, e, traceback.format_exc()[-800:], flush=True)
        got.append(r)
        if not r:
            break
    if len(got) == 2 and all(got):
        sev, title, detail, steps = got[1]
        if __import__("os").environ.get("BA_DRY"):
            print("WOULD NOTE", sev, title, "\n   ", detail, flush=True)
            ex.shot("dry_" + fn.__name__)
            DONE.append(fn.__name__)
            return
        ex.note(sev, title, detail + "\n(reproduit 2 fois)", shot=True, steps=steps)
        DONE.append(fn.__name__)
    else:
        print("  not reproduced twice:", fn.__name__, flush=True)


# ------------------------------------------------------------------ the checks
@check
def fusion_session_delete():
    new("mono")
    for name in ("Ha", "OIII"):
        ex.click_block(blk("empilement", pid=pid_of(name)))
        place("astrometry")
    place("fusion")
    act()
    ex.s._undo.clear()
    ex.s._redo.clear()
    if blk("fusion") is None:
        return None
    n, e = len(TOASTS), len(EXC)
    ex.click_block(blk("session", sid=1), "cross")
    s1 = state(ex)
    gone = "Ha" not in [w["name"] for w in s1["workflows"]]
    fus = blk("fusion")
    t = toasts_since(n)
    act()
    ex.undo()
    back = "Ha" in [w["name"] for w in state(ex)["workflows"]]
    ex.redo()
    exc = EXC[e:]
    ex.settle(4)
    e2 = len(EXC)
    title0 = ex.w.windowTitle()
    ex.key(Qt.Key_S, ex.w, Qt.ControlModifier)
    save_exc = EXC[e2:]
    saved = "Ctrl+S lève à son tour une erreur (« " + save_exc[-1].strip().splitlines()[-1] + " ») : le projet ne "         "peut plus être enregistré, ni fermé proprement (la fermeture enregistre et échoue de même)." if save_exc else         "Ctrl+S ne lève pas d'erreur."
    if gone and fus is not None and exc:
        return ("bloquant", "Supprimer la Session d'un canal de Fusion : accepté sans raison, la Fusion reste orpheline, "
                "puis Ctrl+Y lève une erreur Python",
                f"Projet mono (sessions Ha et OIII, chacune avec Calibration, Alignement, Empilement, Astrométrie) et un "
                f"bloc Fusion HOO. La croix de la Session « Ha » supprime la session ET tout le workflow Ha (calibration "
                f"à astrométrie), sans aucun message (messages : {t}). La Fusion reste dessinée avec « {fus.summary} » "
                f"alors que son canal Ha n'existe plus : la suppression aurait dû être refusée avec la raison (la Fusion "
                f"en a besoin). Ctrl+Z ramène Ha ({'oui' if back else 'non'}), mais Ctrl+Y lève une exception Python : "
                f"« {exc[-1].strip().splitlines()[-1]} ». Même chose avec Suppr, glisser la Session vers la palette, "
                f"« Supprimer le bloc » du menu, et « Supprimer le workflow » du dernier bloc de Ha. Après cette erreur, "
                f"{saved}",
                ["Projet mono Ha + OIII, Astrométrie dans chaque, double-clic Fusion",
                 "Croix de la Session « Ha »", "Ctrl+Z puis Ctrl+Y", "Ctrl+S"])
    return None


@check
def pause_repeated_block():
    new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
    p = ex.s.paths[0]
    ex.page.place_block("gradient", p.pid, 3)          # a second Gradient dropped in front of Astrometry
    ex.settle(6)
    act()
    first = blk(None, title="Gradient 1")
    if first is None:
        return None
    e, n = len(EXC), len(TOASTS)
    m = menu(first, pick="Pause vérif")
    exc = EXC[e:]
    it = blk(None, title="Gradient 1")
    if exc:
        return ("bloquant", "« Pause ici pour vérification » sur un bloc placé deux fois : erreur Python, rien n'est posé",
                f"Chaîne Calibration → Alignement → Empilement → Astrométrie → Gradient, puis un second Gradient déposé "
                f"devant Astrométrie : le graphe les nomme « Gradient 1 » (le nouveau) et « Gradient 2 ». Menu du clic "
                f"droit de « Gradient 1 » → « Pause ici pour vérification » : exception Python « "
                f"{exc[-1].strip().splitlines()[-1]} » (une clé interne « fond~2 »/« fond~3 » apparaît), aucun point de "
                f"pause n'est dessiné (bp={getattr(it, 'bp', None)}), aucun message (messages : {toasts_since(n)}).",
                ["Chaîne jusqu'à Gradient", "Déposer un second Gradient devant Astrométrie",
                 "Clic droit sur « Gradient 1 » → Pause ici pour vérification"])
    return None


@check
def repeated_numbering():
    new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
    p = ex.s.paths[0]
    ex.page.place_block("gradient", p.pid, 3)
    ex.settle(6)
    act()
    first = blk(None, title="Gradient 1")
    if first is None:
        return None
    ex.click_block(first)
    pan = panel_info(ex)["titles"]
    n = len(TOASTS)
    ex.key(Qt.Key_D)
    t = toasts_since(n)
    act()
    ex.undo()
    place("color", "color")
    n2 = len(TOASTS)
    menu(blk("astrometrie"), pick="Supprimer")
    t2 = toasts_since(n2)
    words = " ".join(t)
    bad = [x for x in pan if x.startswith("Gradient") and x != "Gradient 1"] or ("Gradient 2" in words or "Gradient 3" in words
                                                                               or ("Gradient 1" not in words))
    if bad:
        return ("gênant", "Numéro d'un bloc placé deux fois : « Gradient 1 » dans le graphe, autre nom dans le panneau "
                "ou les messages",
                f"Un second Gradient déposé devant Astrométrie s'appelle « Gradient 1 » dans le graphe (rangé à gauche). "
                f"Panneau : {pan[:1]}. Touche D sur ce bloc, message : {t}. La règle : le même nom partout (graphe, "
                f"panneau, messages). Avec deux Couleur (« Couleur 1 », « Couleur 2 » dans le graphe), le refus de supprimer "
                f"Astrométrie dit : {t2} (sans numéro). Les calques faits en tirant le port ● vers le Post portent le numéro interne : "
                f"avec « Gradient 1 », « Gradient 2 », « Gradient 3 » dans le graphe, on obtient « Sortie de Gradient 3 », "
                f"« Sortie de Gradient », « Sortie de Gradient 2 » ; « Couleur 1 » donne « Sortie de Couleur ».",
                ["Chaîne jusqu'à Gradient", "Déposer un second Gradient devant Astrométrie",
                 "Cliquer « Gradient 1 », touche D", "Placer deux Couleur, supprimer Astrométrie"])
    return None


def repeated_delete_redo():
    new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
    p = ex.s.paths[0]
    ex.page.place_block("gradient", p.pid, 3)
    ex.settle(6)
    act()
    ex.s._undo.clear()
    ex.s._redo.clear()
    first = blk(None, title="Gradient 1")
    if first is None:
        return None
    g0 = graph_of()
    ex.click_block(first)
    ex.key(Qt.Key_Delete)
    g1 = graph_of()
    titles1 = [d["title"] for d in drawn(ex)]
    act()
    ex.undo()
    g2 = graph_of()
    act()
    ex.redo()
    g3 = graph_of()
    titles3 = [d["title"] for d in drawn(ex)]
    if g1 != g0 and (g2 != g0 or g3 != g1 or titles1 != titles3):
        return ("gênant", "Bloc placé deux fois supprimé : Ctrl+Z / Ctrl+Y ne redonnent pas le même graphe",
                f"Graphe avant : {g0}. Après Suppr sur « Gradient 1 » : {g1} ; dessin : {titles1}. Après Ctrl+Z : {g2}. "
                f"Après Ctrl+Y : {g3} ; dessin : {titles3}.",
                ["Chaîne jusqu'à Gradient, second Gradient devant Astrométrie", "Cliquer « Gradient 1 », Suppr",
                 "Ctrl+Z puis Ctrl+Y"])
    return None


@check
def post_undo():
    new("session", CHAIN)
    it = blk("etoiles")
    if it is None:
        return None
    before = state(ex)["post"]
    m = menu(it, pick="sans étoiles")
    after = state(ex)["post"]
    ex.undo()
    undone = state(ex)["post"]
    can = ex.s.can_undo()
    if after != before and undone == after:
        return ("gênant", "Envoyer une image au Post depuis le graphe : Ctrl+Z ne l'annule pas",
                f"Chaîne complète jusqu'à « Split étoiles et fond ». Clic droit → « Envoyer l'image sans étoiles à "
                f"Post-traitement » : un calque est ajouté ({len(before)} → {len(after)} calques). Ctrl+Z dans le graphe "
                f"ne le retire pas (calques après Ctrl+Z : {len(undone)} ; il reste à annuler : {can}). Même chose pour "
                f"« Envoyer les étoiles seules… », « Nouveau bloc Combiner les calques étoiles et fond », « Envoyer son "
                f"image au Post » du dernier bloc et les boutons « Vers le Post » du panneau. La règle : toute "
                f"modification s'annule par Ctrl+Z.",
                ["Chaîne complète", "Clic droit sur Split étoiles et fond → Envoyer l'image sans étoiles…", "Ctrl+Z"])
    return None


@check
def output_drawn_after_send():
    new("session", SHORT)
    it = blk("fond")
    menu(it, pick="Vers Post")
    titles = [d["title"] for d in drawn(ex)]
    if any(t == "Sortie" for t in titles):
        pan = panel_info(ex)
        return ("gênant", "Après « Envoyer son image au Post », un bloc « Sortie » apparaît dans le graphe",
                f"Chaîne jusqu'à Couleur, clic droit sur Gradient → « Envoyer son image au Post ». Le graphe dessine "
                f"alors un bloc « Sortie » (sélectionné, panneau « {pan['titles'][:1]} » avec « Enregistrer comme "
                f"modèle… »), et Couleur est déplacé au-dessus. La règle : la Sortie n'est pas un bloc à l'écran, elle "
                f"vit dans le fil vers le Post. Blocs dessinés : {titles}. Même chose depuis Empilement, Astrométrie, "
                f"Débruitage, Couleur, et en tirant le port ● sur Post-traitement.",
                ["Chaîne jusqu'à Couleur", "Clic droit sur Gradient → Envoyer son image au Post"])
    return None


@check
def send_creates_workflow_in_menus():
    new("session", SHORT)
    menu(blk("fond"), pick="Vers Post")
    m = menu(blk("calibration"))
    t = texts(m)
    if any("seulement" in (x or "") for x in t):
        return ("question", "Après un envoi au Post, chaque bloc en amont propose « Retirer de « Sortie de Gradient » "
                "seulement »",
                f"Chaîne jusqu'à Couleur, Gradient envoyé au Post. Le menu de Calibration devient : {t}. L'utilisateur "
                f"n'a créé aucun workflow « Sortie de Gradient » : l'envoi au Post en a fait un, qu'on peut maintenant "
                f"désactiver ou retirer bloc par bloc. Est-ce voulu ?",
                ["Chaîne jusqu'à Couleur", "Clic droit Gradient → Envoyer son image au Post", "Clic droit Calibration"])
    return None


@check
def duplicate():
    new("session", SHORT)
    n = len(TOASTS)
    menu(blk("couleur"), pick="Dupliquer")
    st = state(ex)
    titles = [d["title"] for d in drawn(ex)]
    copy = next((w for w in st["workflows"] if "copie" in w["name"]), None)
    if copy is not None:
        return ("gênant", "Dupliquer : aucun message, deux blocs « Sortie » apparaissent, la copie n'est plus un graphe",
                f"Chaîne jusqu'à Couleur, clic droit sur Couleur (dernier bloc) → « Dupliquer ». Messages : "
                f"{toasts_since(n)} (rien ne dit ce qui a été fait ni Ctrl+Z). Blocs dessinés : {titles} : deux "
                f"« Sortie » apparaissent alors que la Sortie n'est pas un bloc à l'écran. La copie « {copy['name']} » "
                f"est enregistrée comme un workflow à options (sans graphe de blocs : {copy['blocks']}), pas comme le "
                f"graphe dupliqué.",
                ["Chaîne jusqu'à Couleur", "Clic droit sur Couleur → Dupliquer"])
    return None


@check
def delete_workflow_popup():
    new("session", SHORT)
    n = len(ex.popups)
    menu(blk("couleur"), pick="Supprimer workflow")
    pops = ex.popups[n:]
    if pops:
        return ("gênant", "« Supprimer le workflow » ouvre une fenêtre de confirmation bloquante",
                f"(Le mot « workflow » dans ce menu est déjà connu ; ici c'est la fenêtre.) Clic droit sur le dernier "
                f"bloc → « Supprimer le workflow » : une question bloquante s'ouvre {pops[-1][1:]} au lieu d'agir tout de "
                f"suite avec un court message et Ctrl+Z. Ce même menu propose aussi « Supprimer le bloc » juste en "
                f"dessous : deux suppressions de portée différente côte à côte.",
                ["Chaîne jusqu'à Couleur", "Clic droit Couleur → Supprimer le workflow"])
    return None


@check
def delete_no_message():
    new("session", SHORT)
    n = len(TOASTS)
    ex.click_block(blk("fond"))
    ex.key(Qt.Key_Delete)
    t = toasts_since(n)
    ok = any(("retir" in x or "supprim" in x) and "Ctrl+Z" in x for x in t)
    if not ok and "fond" not in [k for k in [n["block"] for n in ex.s.graph(ex.s.paths[0])]]:
        return ("mineur", "Supprimer un bloc : pas de message court ni de rappel Ctrl+Z",
                f"Gradient supprimé (Suppr, croix, menu, panneau ou glissé sur la palette) : seul message « {t} » (le "
                f"coût), rien ne dit « Gradient supprimé · Ctrl+Z l'annule ». Désactiver, retirer une Session ou une "
                f"Image donnent pourtant ce message.",
                ["Chaîne jusqu'à Couleur", "Cliquer Gradient, Suppr"])
    return None


@check
def del_key_word():
    new("session", SHORT)
    m = menu(blk("fond"))
    ex.click_block(blk("fond"))
    tip = next((b["tip"] for b in panel_info(ex)["buttons"] if b["text"] == "Supprimer"), "")
    if any("\tDel" in (x or "") for x in texts(m)):
        return ("mineur", "Menus en français : raccourci écrit « Del » (le panneau dit « Suppr »)",
                f"Menu du clic droit : « Supprimer le bloc\\tDel » ; info-bulle du bouton du panneau : « {tip} ». En "
                f"français la touche s'appelle Suppr.",
                ["Clic droit sur un bloc"])
    return None


@check
def french_agreement():
    new("session", SHORT)
    n = len(TOASTS)
    ex.click_block(blk("calibration"))
    ex.key(Qt.Key_D)
    t1 = toasts_since(n)
    ex.undo()
    n = len(TOASTS)
    menu(blk("astrometrie"), pick="Vers Post")
    t2 = toasts_since(n)
    words = " ".join(t1 + t2)
    if "Calibration désactivé " in words or "de Astrométrie" in words:
        return ("mineur", "Accords et élisions dans les messages : « Calibration désactivé », « Sortie de Astrométrie », "
                "« Image retiré »",
                f"Messages vus : {t1 + t2}. Aussi « Sélection désactivé », « Couleur désactivé », « Image retiré, avec "
                f"les blocs qui le suivent », « sortie de Empilement ». Le nom du bloc est collé dans une phrase "
                f"masculine sans élision.",
                ["Cliquer Calibration, D", "Clic droit Astrométrie → Envoyer son image au Post"])
    return None


@check
def post_word():
    new("session", CHAIN)
    m = texts(menu(blk("etoiles")))
    if "Vers Post" in m and any("à Post-traitement" in (x or "") for x in m):
        return ("mineur", "Le même menu dit « au Post » et « à Post-traitement »",
                f"Menu de Split étoiles et fond : {[x for x in m if x and 'Post' in x]}. Le bloc s'appelle "
                f"« Post-traitement » dans le graphe ; les messages disent « envoyée en Post-traitement ».",
                ["Chaîne complète", "Clic droit sur Split étoiles et fond"])
    return None


@check
def right_click_no_select():
    new("session", SHORT)
    ex.click_block(blk("calibration"))
    menu(blk("fond"))
    sel = ex.s.sel_step
    if sel != "fond":
        return ("question", "Le clic droit ne sélectionne pas le bloc : le panneau montre un autre bloc que celui du menu",
                f"Calibration sélectionné, clic droit sur Gradient : la sélection reste « {sel} » et le panneau montre "
                f"toujours Calibration pendant que le menu agit sur Gradient. Dans les logiciels de référence, le clic "
                f"droit sélectionne l'objet.",
                ["Cliquer Calibration", "Clic droit sur Gradient"])
    return None


@check
def no_context_menu():
    new("session", SHORT)
    a = menu(blk("sources"))
    b = menu(blk("final"))
    if not a and not b:
        return ("mineur", "Pas de menu au clic droit sur la Session (seule) ni sur Post-traitement",
                "Un projet à une session : le clic droit sur le bloc « Session 1 » n'ouvre rien (avec deux sessions il "
                "propose « Renommer la session », « Supprimer le bloc », « Ajouter une session… »). Le clic droit sur "
                "« Post-traitement » n'ouvre rien non plus (pas d'« Ouvrir le Post-traitement »).",
                ["Clic droit sur Session 1", "Clic droit sur Post-traitement"])
    return None


@check
def post_open_grey():
    new("session", SHORT)
    ex.click_block(blk("final"))
    b = next((x for x in panel_info(ex)["buttons"] if "Ouvrir" in x["text"]), None)
    if b is not None and not b["enabled"] and not b["tip"]:
        return ("mineur", "« Ouvrir le Post-traitement » grisé sans dire pourquoi",
                "Panneau du bloc Post-traitement : le bouton « Ouvrir le Post-traitement » est grisé, sans info-bulle "
                "ni phrase (il faut d'abord calculer un workflow). Le double-clic sur le même bloc ouvre pourtant le "
                "Post-traitement.",
                ["Cliquer Post-traitement"])
    return None


@check
def overlaps_new_flow():
    new("session_empty_wf")
    o1 = overlaps(ex)
    ex.shot("overlap_session_empty_wf")
    new("blank")
    place("session", "image")
    o2 = overlaps(ex)
    s2 = ex.shot("overlap_blank_session_image")
    new("two_sessions", ["calibration", "registration", "stacking"])
    o3 = overlaps(ex)
    if o1 or o2 or o3:
        return ("mineur", "La zone « + Nouveau flux » chevauche des blocs",
                f"Projet session + workflow vide : {o1}. Page vierge + Session + Image : {o2} (capture {s2}). Deux "
                f"sessions + chaîne : {o3}. Les rectangles se recouvrent de quelques pixels à une demi-hauteur de bloc ; "
                f"l'étiquette « spectres Gaia » tombe aussi dans la zone quand Couleur est sélectionné.",
                ["Projet session + workflow vide", "Page vierge, Session puis Image", "Deux sessions + chaîne"])
    return None


@check
def empilement_keyboard():
    new("session", ["calibration", "registration", "stacking", "gradient"])
    act()
    ex.click_block(blk("alignement"))
    f1 = QApplication.focusWidget()
    act()
    ex.click_block(blk("empilement"))
    f2 = QApplication.focusWidget()
    aw = QApplication.activeWindow()
    tip = next((b["tip"] for b in panel_info(ex)["buttons"] if b["text"] == "↺"), None)
    n = len(TOASTS)
    from PySide6.QtTest import QTest
    QTest.keyClick(ex.view.viewport(), Qt.Key_D)
    QTest.keyClick(ex.view.viewport(), Qt.Key_Delete)
    ex.settle(4)
    t = toasts_since(n)
    if f1 is not None and f2 is None and not t:
        return ("question", "Après un clic sur Empilement, la fenêtre perd l'activation : le clavier ne répond plus (D, Suppr) jusqu'au clic suivant",
                f"Clic sur Alignement : le focus clavier est sur le graphe ({type(f1).__name__}). Clic sur Empilement : "
                f"la fenêtre principale n'est plus la fenêtre active ({aw}), plus aucun widget n'a le focus ; le panneau "
                f"d'Empilement montre alors le bouton « ↺ » ({tip!r}) — sans ce bouton (Empilement dernier bloc, ou deux "
                f"sessions) le clavier marche. D et Suppr ne font rien et ne disent rien (messages : {t}), alors que "
                f"sur Astrométrie ils donnent la raison du refus. Observé hors écran (à vérifier dans la vraie "
                f"fenêtre). Conséquence observée : après la suppression d'un bloc qui suit Empilement (la sélection passe sur "
                f"Empilement), Ctrl+Z et Ctrl+Y ne répondent plus.",
                ["Chaîne Calibration → Alignement → Empilement → Gradient", "Cliquer Empilement", "D, puis Suppr"])
    return None


@check
def session_rename_undo():
    new("two_sessions", ["calibration", "registration", "stacking"])

    def h(d):
        for e in d.findChildren(QLineEdit):
            e.setText("Nuit renommée")
        d.accept()
    ex.dialog_handler = h
    menu(blk("session", sid=1), pick="Renommer session")
    renamed = any(x["name"] == "Nuit renommée" for x in state(ex)["sessions"])
    can = ex.s.can_undo()
    ex.undo()
    still = any(x["name"] == "Nuit renommée" for x in state(ex)["sessions"])
    if renamed and still:
        return ("mineur", "Renommer une session ne s'annule pas par Ctrl+Z",
                f"Deux sessions ; clic droit sur « Session 1 » → « Renommer la session » → « Nuit renommée ». Ctrl+Z "
                f"laisse « Nuit renommée » (Annuler disponible juste après : {'oui' if can else 'non'}) ; renommer un workflow, lui, s'annule. "
                f"Aucun message après le renommage.",
                ["Deux sessions", "Clic droit Session 1 → Renommer la session", "Ctrl+Z"])
    return None


@check
def session_calibration_no_delete():
    new("two_sessions", ["calibration", "registration", "stacking"])
    it = blk("calibration", sid=1)
    if it is None:
        return None
    m = texts(menu(it))
    ex.click_block(blk("calibration", sid=1))
    btn = [b["text"] for b in panel_info(ex)["buttons"]]
    n = len(TOASTS)
    ex.drag_block_to_palette(blk("calibration", sid=1))
    t = toasts_since(n)
    if not any("Supprimer" in (x or "") for x in m) and not any("Désactiver" in (x or "") for x in m):
        return ("gênant", "Calibration d'une session (deux sessions) : ni Supprimer ni Désactiver, et aucune raison",
                f"Avec deux sessions, chaque session a son bloc Calibration. Son menu : {m} ; son panneau : {btn} ; pas "
                f"de croix ; glissé sur la palette : rien, aucun message ({t}). Avec une seule session, Calibration se "
                f"supprime et se désactive. La règle : un bloc qui ne peut pas partir le dit (raison), un bloc qui peut "
                f"partir a sa croix, Suppr et « Supprimer le bloc ».",
                ["Deux sessions + Calibration, Alignement, Empilement", "Clic droit sur une Calibration de session",
                 "Glisser cette Calibration sur la palette"])
    return None


@check
def fusion_menu():
    new("mono")
    for name in ("Ha", "OIII"):
        ex.click_block(blk("empilement", pid=pid_of(name)))
        place("astrometry")
    place("fusion")
    act()
    it = blk("fusion")
    if it is None:
        return None
    m = texts(menu(it))
    if "Inspection" not in m:
        return ("mineur", "Menu de la Fusion réduit à « Supprimer le bloc »",
                f"Clic droit sur Fusion : {m}. Pas d'« Inspection » (le double-clic l'ouvre pourtant), pas de « Pause ici "
                f"pour vérification », pas de « Renommer » (son nom se change dans le panneau), alors que les autres "
                f"blocs ont ces entrées.",
                ["Projet mono, Astrométrie dans Ha et OIII, Fusion", "Clic droit sur Fusion"])
    return None


@check
def d_on_source_silent():
    new("image", ["astrometry"])
    n = len(TOASTS)
    ex.click_block(blk("image"))
    ex.key(Qt.Key_D)
    t = toasts_since(n)
    m = texts(menu(blk("image")))
    if not t:
        return ("mineur", "D sur un bloc Image (ou Session, Calibration de session) : rien, sans un mot",
                f"Touche D sur le bloc Image : aucun changement, aucun message. Son menu ({m}) ne dit pas non plus "
                f"pourquoi il ne se désactive pas, alors que pour Empilement le menu affiche « Désactiver — raison ». "
                f"Même silence pour D sur une Session et sur la Calibration d'une session.",
                ["Projet image + Astrométrie", "Cliquer Image, D"])
    return None


@check
def switch_off_hidden_in_panel():
    new("session", SHORT)
    m = texts(menu(blk("astrometrie")))
    ex.click_block(blk("astrometrie"))
    btn = [b["text"] for b in panel_info(ex)["buttons"]]
    if any(x and x.startswith("Désactiver  —") for x in m) and not any("Désactiver" in b for b in btn):
        return ("question", "Désactiver : le menu dit pourquoi c'est impossible, le panneau cache simplement le bouton",
                f"Astrométrie (Couleur en a besoin) : menu « {[x for x in m if x and x.startswith('Désactiver')][0]} » "
                f"(grisé avec la raison) ; le panneau n'a que {btn}, sans bouton ni raison. Même chose pour Empilement, "
                f"Débruitage, Split étoiles et fond.",
                ["Chaîne jusqu'à Couleur", "Clic droit Astrométrie", "Cliquer Astrométrie"])
    return None


@check
def split_english_word():
    new("session", CHAIN)
    m = texts(menu(blk("debruitage")))
    if any("Split" in (x or "") for x in m):
        return ("mineur", "Le mot anglais « Split » dans les menus et messages en français",
                f"Menu de Débruitage : « {[x for x in m if x and 'Split' in x][0]} ». Le bloc s'appelle « Split étoiles et "
                f"fond » partout (graphe, palette, menus, messages). En français : « Séparer étoiles et fond ».",
                ["Chaîne complète", "Clic droit sur Débruitage"])
    return None


@check
def shared_partial_delete():
    new("session", SHORT)
    menu(blk("fond"), pick="Vers Post")
    act()
    ex.s._undo.clear()
    ex.s._redo.clear()
    n = len(TOASTS)
    it = blk("astrometrie")
    ex.hover(ex.view.viewport(), ex.view.mapFromScene(it.mapToScene(it.w - 12, 12)))
    ex.click(ex.view.viewport(), ex.view.mapFromScene(it.mapToScene(it.w - 12, 12)))
    t = toasts_since(n)
    gs = {w["name"]: [b[0] for b in w["blocks"]] for w in state(ex)["workflows"]}
    part = any("astrometry" in v for v in gs.values()) and any("astrometry" not in v for v in gs.values())
    if part and any("ne peut pas" in x for x in t):
        return ("gênant", "La croix d'un bloc partagé le retire d'une partie des workflows, le message ne parle que du refus",
                f"Chaîne jusqu'à Couleur, Gradient envoyé au Post (cela crée « Sortie de Gradient », qui partage Calibration "
                f"→ Gradient). Croix d'Astrométrie : elle est retirée de « Sortie de Gradient » mais gardée dans "
                f"« Workflow » (Couleur en a besoin). Message : {t}. Rien ne dit qu'elle a quand même été retirée "
                f"ailleurs ; le bloc reste affiché. Workflows après : {gs}.",
                ["Chaîne jusqu'à Couleur", "Clic droit Gradient → Envoyer son image au Post", "Croix d'Astrométrie"])
    return None


@check
def output_block_actions():
    new("session", SHORT)
    menu(blk("fond"), pick="Vers Post")
    it = blk(None, title="Sortie")
    if it is None:
        return None
    m = texts(menu(it))
    n = len(TOASTS)
    menu(blk(None, title="Sortie"), pick="Supprimer")
    t = toasts_since(n)
    if any("absent du workflow" in (x or "") for x in m + t):
        return ("gênant", "Le bloc « Sortie » d'un envoi au Post : « Supprimer le bloc » refuse avec une phrase absurde",
                f"Après « Envoyer son image au Post » sur Gradient, le bloc « Sortie » a ce menu : {m}. « Supprimer le "
                f"bloc » répond : {t}. « Désactiver » est grisé avec « ce bloc n'est pas dans le workflow ». Il n'a pas "
                f"de croix. Le menu propose aussi « Supprimer le workflow » (fenêtre de confirmation).",
                ["Chaîne jusqu'à Couleur", "Clic droit Gradient → Envoyer son image au Post",
                 "Clic droit Sortie → Supprimer le bloc"])
    return None


@check
def fusion_astrometry_delete():
    new("session", ["calibration", "registration", "stacking", "astrometry"])
    place("fusion")
    act()
    if blk("fusion") is None:
        return None
    ex.s._undo.clear()
    n = len(TOASTS)
    ex.click_block(blk("astrometrie"))
    ex.key(Qt.Key_Delete)
    t = toasts_since(n)
    g = graph_of()
    fus = blk("fusion")
    if "astrometry" not in g and fus is not None:
        ex.click_block(fus)
        txt = " ".join(panel_info(ex)["texts"])
        need = "Astrométrie" in txt or "Astrometry" in txt
        return ("gênant", "Astrométrie d'un workflow lu par une Fusion : supprimée sans refus",
                f"Chaîne Calibration → Alignement → Empilement → Astrométrie, puis Fusion (HOO bicolore). Suppr sur "
                f"Astrométrie : supprimée ({g}), message : {t}. Le panneau de la Fusion dit pourtant que chaque canal "
                f"a besoin de son Astrométrie ({'oui' if need else 'texte non trouvé'}). La règle des dépendances "
                f"lointaines : refusé avec la raison.",
                ["Chaîne jusqu'à Astrométrie", "Double-clic Fusion", "Cliquer Astrométrie, Suppr"])
    return None


def drag_branch(block, under_key, pid):
    ex.drag_palette_hover(block, QPoint(5, 5))
    gh = [g for g in ex.ghosts() if getattr(g, "branch_key", None) == under_key and g.pid == pid]
    pt = ex.ghost_center(gh[0]) if gh else None
    ex.drag_leave()
    if pt is not None:
        ex.drag_palette(block, pt)
        ex.settle(6)
    act()


@check
def shared_switch_offer():
    new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
    w = pid_of("Workflow")
    ex.click_block(blk("astrometrie"))
    drag_branch("color", "astrometrie", w)
    it = blk("astrometrie", pid=w)
    if it is None or len(it.node.paths) < 2:
        return None
    m = menu(it)
    offer = [x for x in m if (x.get("text") or "").startswith("Désactiver dans les")]
    n = len(TOASTS)
    menu(blk("astrometrie", pid=w), pick=offer[0]["text"]) if offer else None
    t = toasts_since(n)
    ex.click_block(blk("astrometrie", pid=w))
    btn = [b["text"] for b in panel_info(ex)["buttons"]]
    if offer and offer[0]["enabled"] and any("besoin" in x for x in t):
        return ("mineur", "Bloc partagé par une branche : « Désactiver dans les 2 workflows » est proposé, puis refusé",
                f"Chaîne jusqu'à Gradient, puis une branche Couleur déposée sous Astrométrie. Menu d'Astrométrie : "
                f"{[x.get('text') for x in m if 'Désactiver' in (x.get('text') or '')]} (tous actifs). « "
                f"{offer[0]['text']} » répond : {t}. Sur une chaîne simple la même entrée est grisée avec la raison. Le "
                f"panneau, lui, n'a pas de bouton Désactiver ({btn}).",
                ["Chaîne jusqu'à Gradient", "Glisser Couleur sur « nouvelle branche » sous Astrométrie",
                 "Clic droit Astrométrie → Désactiver dans les 2 workflows"])
    return None


def _comp():
    return next((it for it in ex.blocks() if it.node.key == "composition"), None)


@check
def composition_default():
    new("session", CHAIN)
    it = _comp()
    if it is None or it.node.lid is not None:
        return None
    n = len(TOASTS)
    vp = ex.view.viewport()
    pt = ex.view.mapFromScene(it.mapToScene(it.w - 12, 12))
    ex.hover(vp, pt)
    ex.click(vp, pt)
    still = _comp() is not None
    post_after_cross = len(state(ex)["post"])
    t1 = toasts_since(n)
    new("session", CHAIN)
    p0 = len(state(ex)["post"])
    ex.click_block(_comp())
    p1 = len(state(ex)["post"])
    dirty = ex.s.dirty
    act()
    n = len(TOASTS)
    ex.key(Qt.Key_Delete)
    gone = _comp() is None
    t2 = toasts_since(n)
    act()
    ex.undo()
    back = _comp() is not None
    if still and p1 > p0 and gone and not back:
        return ("gênant", "Bloc « Combiner les calques étoiles et fond » donné par défaut : la croix ne fait rien, un clic "
                "crée un calque, Suppr le retire sans message et Ctrl+Z ne le rend pas",
                f"Chaîne complète jusqu'à Split étoiles et fond : le graphe montre d'office un bloc « Combiner les calques "
                f"étoiles et fond ». Sa croix (et « Supprimer le bloc » de son menu) : rien ne se passe, aucun message "
                f"({t1}), le bloc reste. Un simple clic dessus crée un calque dans le Post ({p0} → {p1} calque) et "
                f"ce qui ne s'annule pas : sélectionner modifie le projet. Ensuite Suppr le retire, sans "
                f"message ({t2}), et Ctrl+Z ne le fait pas revenir.",
                ["Chaîne complète", "Croix du bloc Combiner (rien)", "Clic sur le bloc, Suppr, Ctrl+Z"])
    return None


def main():
    for fn in CHECKS:
        if ONLY and fn.__name__ not in ONLY:
            continue
        print("check", fn.__name__, flush=True)
        run(fn)
    print("NOTED", DONE, flush=True)
    ex.close()


main()
