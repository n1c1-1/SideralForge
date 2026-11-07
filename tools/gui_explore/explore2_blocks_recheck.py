"""block_actions round 2, step 1: each round-1 anomaly replayed with real gestures; one line per anomaly in
recheck.jsonl (status: corrigé | toujours là | changé).

    python explore2_blocks_recheck.py [check numbers...]
"""
from __future__ import annotations

import faulthandler
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(900, exit=True)

from explore2_blocks_common import CHAIN, EXC, SHORT, TOASTS, G, Explorer, overlaps, panel_info, state  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402

ex = Explorer("block_actions")
g = G(ex)
OUT = ex.out / "recheck_raw.jsonl"                    # raw runs; recheck.jsonl is written from them, once checked
R1 = [json.loads(x) for x in Path("F:/astro_essais/A_REGARDER/exploration/block_actions/anomalies.jsonl")
      .read_text(encoding="utf-8").splitlines() if x.strip()]
ONLY = [int(x) for x in sys.argv[1:]]
CHECKS: dict[int, object] = {}


def check(n):
    def deco(fn):
        CHECKS[n] = fn
        return fn
    return deco


def last_exc(e0):
    return EXC[e0][-400:].strip().splitlines()[-1] if len(EXC) > e0 else ""


@check(1)
def fusion_session_delete():
    g.mono_fusion()
    if g.blk("fusion") is None:
        return "changé", "La Fusion n'a pas pu être posée par double-clic sur ce projet mono : cas non rejoué.", None
    n, e = len(TOASTS), len(EXC)
    g.cross(g.blk("session", sid=1))
    st = state(ex)
    gone = "Ha" not in [x["name"] for x in st["sessions"]]
    t = g.toasts(n)
    shot = ex.shot("recheck_01_after_cross")
    if not gone:
        return ("corrigé" if t and not EXC[e:] else "changé",
                f"La croix de la Session Ha est refusée ; message : {t}.", shot)
    g.undo()
    back = "Ha" in [x["name"] for x in state(ex)["sessions"]]
    g.redo()
    e2 = len(EXC)
    ex.key(Qt.Key_S, ex.w, Qt.ControlModifier)
    fus = g.blk("fusion")
    exc = EXC[e:]
    if exc:
        return ("toujours là", f"La croix de la Session Ha est acceptée (messages {t}) ; Ctrl+Z la ramène "
                f"({'oui' if back else 'non'}) ; erreur Python : « {last_exc(e)} ».", shot)
    return ("changé", f"La Session Ha est retirée (messages : {t}) ; la Fusion "
            f"{'reste dessinée : « ' + fus.summary + ' »' if fus else 'disparaît aussi'} ; Ctrl+Z la ramène "
            f"({'oui' if back else 'non'}), Ctrl+Y et Ctrl+S sans erreur.", shot)


def two_gradients():
    g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
    pid = ex.s.paths[0].pid
    ok = g.drop_idx("gradient", pid, 3)
    if not ok:
        ex.page.place_block("gradient", pid, 3)
        ex.settle(6)
    g.act()
    return ok


@check(2)
def pause_repeated_block():
    two_gradients()
    first = g.blk(title="Gradient 1")
    if first is None:
        return "changé", f"Pas de « Gradient 1 » après un second Gradient : blocs {g.titles()}.", None
    e, n = len(EXC), len(TOASTS)
    g.menu(first, pick="Pause vérif")
    bp = state(ex)["workflows"][0]["bp"]
    shot = ex.shot("recheck_02_pause")
    if EXC[e:]:
        return "toujours là", f"« Pause vérif » sur Gradient 1 : erreur Python « {last_exc(e)} ».", shot
    return ("corrigé" if bp else "changé", f"« Pause vérif » sur Gradient 1 : sans erreur, pause posée : {bp} ; "
            f"messages : {g.toasts(n)}.", shot)


@check(3)
def repeated_numbering():
    two_gradients()
    first = g.blk(title="Gradient 1")
    if first is None:
        return "changé", f"Pas de « Gradient 1 » : {g.titles()}.", None
    ex.click_block(first)
    pan = [x for x in panel_info(ex)["titles"] if "Gradient" in x]
    n = len(TOASTS)
    g.act()
    ex.key(Qt.Key_D)
    t = g.toasts(n)
    g.undo()
    n2 = len(TOASTS)
    m = g.menu(g.blk(title="Gradient 1"), pick="Vers Post")
    layers = [x.get("name") for x in state(ex)["post"]]
    t2 = g.toasts(n2)
    shot = ex.shot("recheck_03_numbering")
    words = " ".join(t + t2 + [str(x) for x in layers])
    bad = [x for x in pan if x.strip() not in ("Gradient 1",) and "Gradient" in x] or "Gradient 2" in words or \
        "Gradient 3" in words or "Gradient 1" not in words
    return ("toujours là" if bad else "corrigé",
            f"Graphe « Gradient 1 » ; panneau {pan[:2]} ; D : {t} ; Vers Post : {t2} ; calques {layers}.", shot)


@check(4)
def post_undo():
    g.new("session", CHAIN)
    it = g.blk("etoiles")
    before = state(ex)["post"]
    g.menu(it, pick="Sans étoiles vers Post")
    after = state(ex)["post"]
    g.undo()
    undone = state(ex)["post"]
    g.redo()
    redone = state(ex)["post"]
    shot = ex.shot("recheck_04_post_undo")
    if after == before:
        return "changé", f"« Sans étoiles vers Post » ne change pas les calques ({len(before)}).", shot
    ok = undone == before and redone == after
    return ("corrigé" if ok else "toujours là",
            f"Calques : avant {len(before)}, après l'envoi {len(after)}, après Ctrl+Z {len(undone)}, après Ctrl+Y "
            f"{len(redone)}.", shot)


@check(5)
def output_drawn_after_send():
    g.new("session", SHORT)
    g.menu(g.blk("fond"), pick="Vers Post")
    t = g.titles()
    shot = ex.shot("recheck_05_output")
    return ("toujours là" if "Sortie" in t else "corrigé", f"Blocs dessinés après « Vers Post » sur Gradient : {t}.", shot)


@check(6)
def send_creates_workflow_in_menus():
    g.new("session", SHORT)
    g.menu(g.blk("fond"), pick="Vers Post")
    m = g.texts(g.menu(g.blk("calibration")))
    shot = ex.shot("recheck_06_menus")
    bad = [x for x in m if x and "seulement" in x]
    return ("toujours là" if bad else "corrigé", f"Menu de Calibration après l'envoi de Gradient : {m}.", shot)


@check(7)
def duplicate():
    g.new("session", SHORT)
    n = len(TOASTS)
    g.menu(g.blk("couleur"), pick="Dupliquer")
    st = state(ex)
    t = g.titles()
    copy = next((w for w in st["workflows"] if "copie" in w["name"] or "copy" in w["name"]), None)
    shot = ex.shot("recheck_07_duplicate")
    if copy is None:
        return "changé", f"Dupliquer ne crée pas de copie : workflows {[w['name'] for w in st['workflows']]}.", shot
    bad = "Sortie" in t or copy["blocks"] is None or not g.toasts(n)
    return ("toujours là" if bad and copy["blocks"] is None else ("changé" if bad else "corrigé"),
            f"Copie « {copy['name']} » : graphe {None if copy['blocks'] is None else [b[0] for b in copy['blocks']]} ; "
            f"messages {g.toasts(n)} ; blocs dessinés {t}.", shot)


@check(8)
def delete_workflow_popup():
    g.new("session", SHORT)
    p, md, n = len(ex.popups), len(ex.modals), len(TOASTS)
    m = g.texts(g.menu(g.blk("couleur"), pick="Supprimer workflow"))
    pops = ex.popups[p:] + ex.modals[md:]
    shot = ex.shot("recheck_08_delete_wf")
    left = [q.name for q in ex.s.paths]
    if pops:
        return "toujours là", f"« Supprimer workflow » ouvre encore une fenêtre : {pops}.", shot
    g.undo()
    back = [q.name for q in ex.s.paths]
    return ("corrigé", f"« Supprimer workflow » agit sans fenêtre ; message {g.toasts(n)} ; workflows après {left}, "
            f"après Ctrl+Z {back}.", shot)


@check(9)
def delete_no_message():
    g.new("session", SHORT)
    n = len(TOASTS)
    ex.click_block(g.blk("fond"))
    g.act()
    ex.key(Qt.Key_Delete)
    t = g.toasts(n)
    shot = ex.shot("recheck_09_delete_msg")
    ok = any("Ctrl+Z" in x for x in t)
    return ("corrigé" if ok else "toujours là", f"Suppr sur Gradient : messages {t}.", shot)


@check(10)
def del_key_word():
    g.new("session", SHORT)
    m = g.texts(g.menu(g.blk("fond")))
    shot = ex.shot("recheck_10_del")
    bad = any("\tDel" in (x or "") for x in m)
    return ("toujours là" if bad else "corrigé", f"Menu de Gradient : {[x for x in m if x and 'Supprimer' in x]}.", shot)


@check(11)
def french_agreement():
    g.new("session", SHORT)
    n = len(TOASTS)
    ex.click_block(g.blk("calibration"))
    g.act()
    ex.key(Qt.Key_D)
    t1 = g.toasts(n)
    g.undo()
    n = len(TOASTS)
    g.menu(g.blk("astrometrie"), pick="Vers Post")
    t2 = g.toasts(n) + [str(x.get("name")) for x in state(ex)["post"]]
    g.new("image", [])
    n = len(TOASTS)
    ex.click_block(g.blk("image"))
    g.act()
    ex.key(Qt.Key_Delete)
    t3 = g.toasts(n)
    shot = ex.shot("recheck_11_agreement")
    words = " ".join(t1 + t2 + t3)
    bad = [w for w in ("Calibration désactivé ", "de Astrométrie", "Image retiré ", "de Empilement") if w in words]
    return ("toujours là" if bad else "corrigé", f"Messages : D sur Calibration {t1} ; Vers Post sur Astrométrie {t2} ; "
            f"Suppr sur Image {t3}.", shot)


@check(12)
def post_word():
    g.new("session", CHAIN)
    m = g.texts(g.menu(g.blk("etoiles")))
    shot = ex.shot("recheck_12_post_word")
    p = [x for x in m if x and "Post" in x]
    bad = any("Post-traitement" in x for x in p)
    return ("toujours là" if bad else "corrigé", f"Entrées Post du menu Étoiles/fond : {p}.", shot)


@check(13)
def right_click_no_select():
    g.new("session", SHORT)
    ex.click_block(g.blk("calibration"))
    g.menu(g.blk("fond"))
    sel = ex.s.sel_step
    shot = ex.shot("recheck_13_select")
    return ("corrigé" if sel == "fond" else "toujours là", f"Après clic sur Calibration puis clic droit sur Gradient, "
            f"le bloc choisi est « {sel} ».", shot)


@check(14)
def no_context_menu():
    g.new("session", SHORT)
    a = g.texts(g.menu(g.blk("sources")))
    b = g.texts(g.menu(g.blk("final")))
    shot = ex.shot("recheck_14_menus")
    if a and b:
        return "corrigé", f"Menu Session : {a} ; menu Post : {b}.", shot
    if a or b:
        return "changé", f"Menu de la Session seule : {a or 'aucun'} ; menu du bloc Post : {b or 'aucun'}.", shot
    return "toujours là", "Ni la Session seule ni Post n'ont de menu.", shot


@check(15)
def post_open_grey():
    g.new("session", SHORT)
    ex.click_block(g.blk("final"))
    b = next((x for x in panel_info(ex)["buttons"] if "Ouvrir" in x["text"]), None)
    shot = ex.shot("recheck_15_open_post")
    if b is None:
        return "changé", "Pas de bouton « Ouvrir » dans le panneau de Post.", shot
    ok = b["enabled"] or b["tip"]
    return ("corrigé" if ok else "toujours là", f"Bouton « {b['text']} » grisé={not b['enabled']}, infobulle : "
            f"« {b['tip']} ».", shot)


@check(16)
def overlaps_new_flow():
    g.new("session_empty_wf")
    o1 = overlaps(ex)
    g.new("blank")
    g.place("session", "image")
    o2 = overlaps(ex)
    s2 = ex.shot("recheck_16_overlap_blank")
    g.new("two_sessions", ["calibration", "registration", "stacking"])
    o3 = overlaps(ex)
    if not (o1 or o2 or o3):
        return "corrigé", "Plus aucun chevauchement des rectangles dans les trois cas.", s2
    return "toujours là", f"Chevauchements : session+workflow vide {o1} ; Session puis Image {o2} ; deux sessions {o3}.", s2


@check(17)
def empilement_keyboard():
    g.new("session", ["calibration", "registration", "stacking", "gradient"])
    ex.click_block(g.blk("alignement"))
    ex.click_block(g.blk("empilement"))
    f2 = QApplication.focusWidget()
    aw = QApplication.activeWindow()
    from PySide6.QtTest import QTest
    n = len(TOASTS)
    QTest.keyClick(ex.view.viewport(), Qt.Key_D)
    QTest.keyClick(ex.view.viewport(), Qt.Key_Delete)
    ex.settle(4)
    t = g.toasts(n)
    g_after = g.graph_of()
    shot = ex.shot("recheck_17_keyboard")
    ok = f2 is not None and aw is not None
    return ("corrigé" if ok and (t or "stacking" not in g_after) else "toujours là",
            f"Après un clic sur Empilement : focus {type(f2).__name__ if f2 else None}, fenêtre active "
            f"{'oui' if aw else 'non'} ; D puis Suppr donnent {t}, graphe {g_after}.", shot)


@check(18)
def session_rename_undo():
    g.new("two_sessions", ["calibration", "registration", "stacking"])

    def h(d):
        for e in d.findChildren(QLineEdit):
            e.setText("Nuit renommée")
        d.accept()
    ex.dialog_handler = h
    n = len(TOASTS)
    g.menu(g.blk("session", sid=1), pick="Renommer session")
    renamed = any(x["name"] == "Nuit renommée" for x in state(ex)["sessions"])
    t = g.toasts(n)
    g.undo()
    still = any(x["name"] == "Nuit renommée" for x in state(ex)["sessions"])
    g.redo()
    again = any(x["name"] == "Nuit renommée" for x in state(ex)["sessions"])
    shot = ex.shot("recheck_18_rename")
    if not renamed:
        return "changé", "Le renommage par le menu n'a pas pris (champ de saisie non trouvé ?).", shot
    return ("corrigé" if not still and again else "toujours là",
            f"Renommée ; message {t} ; après Ctrl+Z encore renommée : {'oui' if still else 'non'} ; après Ctrl+Y : "
            f"{'oui' if again else 'non'}.", shot)


@check(19)
def session_calibration_no_delete():
    g.new("two_sessions", ["calibration", "registration", "stacking"])
    it = g.blk("calibration", sid=1)
    m = g.texts(g.menu(it))
    ex.click_block(g.blk("calibration", sid=1))
    btn = [b["text"] for b in panel_info(ex)["buttons"]]
    shot = ex.shot("recheck_19_session_calib")
    n = len(TOASTS)
    before = g.snap()
    g.menu(g.blk("calibration", sid=1), pick="Supprimer")
    t = g.toasts(n)
    changed = g.snap() != before
    has_menu = any((x or "").startswith("Supprimer") for x in m)
    has_panel = any(b in ("Supprimer", "Désactiver") for b in btn)
    st = "corrigé" if has_menu and has_panel else ("changé" if has_menu else "toujours là")
    return (st, f"Menu de la Calibration d'une session : {m} ; panneau : {btn} ; « Supprimer » du menu : "
            f"{'fait' if changed else 'rien'}, message {t}.", shot)


@check(20)
def fusion_menu():
    g.mono_fusion()
    it = g.blk("fusion")
    if it is None:
        return "changé", "Fusion non posée.", None
    m = g.texts(g.menu(it))
    shot = ex.shot("recheck_20_fusion_menu")
    return ("corrigé" if "Inspection" in m else ("changé" if len([x for x in m if x != "---"]) > 1 else "toujours là"),
            f"Menu de la Fusion : {m}.", shot)


@check(21)
def d_on_source_silent():
    g.new("image", ["astrometry"])
    n = len(TOASTS)
    ex.click_block(g.blk("image"))
    g.act()
    ex.key(Qt.Key_D)
    t = g.toasts(n)
    m = g.texts(g.menu(g.blk("image")))
    shot = ex.shot("recheck_21_d_image")
    return ("corrigé" if t else "toujours là", f"D sur le bloc Image : messages {t} ; menu {m}.", shot)


@check(22)
def switch_off_hidden_in_panel():
    g.new("session", SHORT)
    ex.click_block(g.blk("astrometrie"))
    b = next((x for x in panel_info(ex)["buttons"] if x["text"].startswith("Désactiver")), None)
    shot = ex.shot("recheck_22_panel_off")
    return ("corrigé" if b and b["tip"] else "toujours là", f"Panneau d'Astrométrie (Couleur en a besoin) : bouton "
            f"{b['text'] if b else 'absent'}, grisé={not b['enabled'] if b else '-'}, infobulle « {b['tip'] if b else ''} ».",
            shot)


@check(23)
def split_english_word():
    g.new("session", CHAIN)
    m = g.texts(g.menu(g.blk("debruitage")))
    t = g.titles()
    shot = ex.shot("recheck_23_split")
    bad = any("Split" in (x or "") for x in m + t)
    return ("toujours là" if bad else "corrigé", f"Menu de Débruitage : {[x for x in m if x and 'Désactiver' in x]} ; "
            f"bloc nommé « {[x for x in t if 'toiles' in x]} ».", shot)


@check(24)
def shared_partial_delete():
    g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
    w = g.pid_of("Workflow") if any(q.name == "Workflow" for q in ex.s.paths) else ex.s.paths[0].pid
    ok = g.drag_branch("color", "astrometrie", w)
    ex.s._undo.clear()
    ex.s._redo.clear()
    names = [q.name for q in ex.s.paths]
    if not ok or len(names) < 2:
        return "changé", f"Branche Couleur sous Astrométrie non créée : workflows {names}.", None
    n = len(TOASTS)
    before = {q.name: g.graph_of(q.name) for q in ex.s.paths}
    g.cross(g.blk("astrometrie", pid=w))
    after = {q.name: g.graph_of(q.name) for q in ex.s.paths}
    t = g.toasts(n)
    shot = ex.shot("recheck_24_shared_cross")
    part = any("astrometry" in v for v in after.values()) and any(
        "astrometry" in before[k] and "astrometry" not in v for k, v in after.items() if k in before)
    if part and not any("retir" in x and "ne peut" not in x for x in t):
        return "toujours là", f"Croix d'Astrométrie partagée : avant {before}, après {after} ; message {t}.", shot
    return ("corrigé" if not part else "changé", f"Croix d'Astrométrie partagée (Couleur en branche) : avant {before}, "
            f"après {after} ; message {t}.", shot)


@check(25)
def output_block_actions():
    g.new("session", SHORT)
    g.menu(g.blk("fond"), pick="Vers Post")
    it = g.blk(title="Sortie")
    shot = ex.shot("recheck_25_output_menu")
    if it is None:
        return "corrigé", "Plus de bloc « Sortie » dans le graphe après « Vers Post » : son menu n'existe plus.", shot
    m = g.texts(g.menu(it))
    return "toujours là", f"Bloc Sortie toujours là, menu {m}.", shot


@check(26)
def fusion_astrometry_delete():
    g.mono_fusion()
    if g.blk("fusion") is None:
        return "changé", "Fusion non posée.", None
    ha = g.pid_of("Ha")
    n = len(TOASTS)
    ex.click_block(g.blk("astrometrie", pid=ha))
    g.act()
    ex.key(Qt.Key_Delete)
    t = g.toasts(n)
    gr = g.graph_of("Ha")
    shot = ex.shot("recheck_26_fusion_astro")
    return ("toujours là" if "astrometry" not in gr else "corrigé",
            f"Suppr sur l'Astrométrie du canal Ha (lu par la Fusion) : graphe Ha {gr} ; message {t}.", shot)


@check(27)
def shared_switch_offer():
    g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
    w = ex.s.paths[0].pid
    g.drag_branch("color", "astrometrie", w)
    it = g.blk("astrometrie", pid=w)
    if it is None or len(it.node.paths) < 2:
        return "changé", "Branche non créée.", None
    m = g.menu(it)
    offs = [(x["text"], x["enabled"]) for x in m if "Désactiver" in (x.get("text") or "")]
    sub = [x for x in m if x.get("sub")]
    n = len(TOASTS)
    first = next((x for x in m if (x.get("text") or "").startswith("Désactiver") and x["enabled"]), None)
    if first:
        g.menu(g.blk("astrometrie", pid=w), pick=first["text"])
    t = g.toasts(n)
    shot = ex.shot("recheck_27_shared_off")
    refused_after_offer = first is not None and any("besoin" in x or "requis" in x for x in t)
    return ("toujours là" if refused_after_offer else "corrigé",
            f"Menu d'Astrométrie partagée : {offs} ; sous-menus {[(x['text'], [y.get('text') for y in x['sub']]) for x in sub]} ; "
            f"choix « {first['text'] if first else '-'} » : {t}.", shot)


@check(28)
def composition_default():
    g.new("session", CHAIN)
    comp = lambda: next((it for it in ex.blocks() if it.node.key == "composition"), None)  # noqa: E731
    it = comp()
    if it is None:
        return "corrigé", "Plus de bloc Recombiner donné d'office après Étoiles/fond.", ex.shot("recheck_28_comp")
    lid0 = it.node.lid
    p0 = len(state(ex)["post"])
    n = len(TOASTS)
    g.cross(it)
    gone1 = comp() is None
    t1 = g.toasts(n)
    p1 = len(state(ex)["post"])
    g.undo()
    back1 = comp() is not None
    g.new("session", CHAIN)
    p2 = len(state(ex)["post"])
    ex.click_block(comp())
    p3 = len(state(ex)["post"])
    dirty = ex.s.can_undo()
    shot = ex.shot("recheck_28_comp")
    bad = (not gone1) or p3 != p2 or not back1
    return ("toujours là" if bad else "corrigé",
            f"Recombiner donné d'office (calque déjà fait : {'oui' if lid0 is not None else 'non'}) ; croix : retiré "
            f"{'oui' if gone1 else 'non'}, message {t1}, calques {p0}→{p1}, Ctrl+Z le rend {'oui' if back1 else 'non'} ; "
            f"un simple clic change les calques {p2}→{p3}, rend Annuler disponible : {'oui' if dirty else 'non'}.", shot)


def main():
    for k, fn in CHECKS.items():
        if ONLY and k not in ONLY:
            continue
        title = R1[k - 1]["title"]
        print(f"check {k} {fn.__name__}", flush=True)
        e0 = len(EXC)
        try:
            status, note, shot = fn()
        except Exception as e:  # noqa: BLE001
            status, note, shot = "changé", f"(script : {type(e).__name__}: {e})", None
            print(traceback.format_exc()[-1500:], flush=True)
        if len(EXC) > e0 and "erreur Python" not in note:
            note += f" Erreur Python pendant le rejeu : « {last_exc(e0)} »."
        rec = {"title": title, "status": status, "note": note, "shot": shot}
        print("  ->", status, note[:400], flush=True)
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ex.close()


main()
