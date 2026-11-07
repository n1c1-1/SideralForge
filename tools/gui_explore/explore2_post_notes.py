"""post_wires round 2: the new anomalies, each reproduced twice from a fresh project before it is noted (with the
screenshot of the second time). Args: the candidates to run (default all)."""
import sys
import time

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QComboBox

start(1200)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n11"]
ST = ["Nouveau projet avec une session de 15 brutes", "Double-clic dans la palette : Calibration, Alignement, "
      "Empilement, Astrométrie, Gradient, Débruitage, Couleur"]


def wait(sec):
    t = time.time()
    while time.time() - t < sec:
        ex.settle(5)


def simulate():
    s = ex.s
    orig = s._sync

    def fake(*a, **k):
        orig(*a, **k)
        for v in s.paths:
            v.rendered = True
            v.computed = False
    s._sync = fake
    s._sync()


def twice(fn):
    """fn() -> (bool, detail, shot name). Noted when both runs say True."""
    a = fn()
    b = fn()
    print(fn.__name__, "run1", a[0], "run2", b[0], "|", b[1], flush=True)
    return (a[0] and b[0]), b


def n1():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    v = ex.w.post_holder.page.view
    tgt = next(lay for lay in v.layers if "Gradient" in lay["name"])
    v.select(tgt["id"], "layer")
    ex.settle(3)
    ex.click(v.b_del)
    ex.settle(10)
    ex.click(ex.w.mode_btns[0])
    ex.settle(10)
    br = [q.name for q in ex.s.paths if "Gradient" in q.name]
    outs = [(b.title, b.summary) for b in pp.blocks_of("post")]
    wires = [w.toolTip().splitlines()[0] for w in pp.wires()]
    ok = bool(br) and not any("Gradient" in w for w in wires)
    f = ex.shot("n1_layer_deleted_in_post")
    return ok, (f"Après « Supprimer » du calque « Sortie Gradient » dans la page Post, le graphe garde la branche "
                f"« {br[0] if br else ''} » et son bloc {outs} sans fil vers Post ; la barre d'état compte toujours "
                f"{len(ex.s.paths)} workflows (elle sera calculée). Retirer le fil dans le graphe, lui, supprime la "
                "branche."), f


def n2():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    v = ex.w.post_holder.page.view
    lay = next(lay for lay in v.layers if "Gradient" in lay["name"])
    v.select(lay["id"], "layer")
    ex.settle(4)
    c = next(c for c in v.right.findChildren(QComboBox) if c.isVisible() and c.count() > 2)
    c.setCurrentIndex(2)
    ex.settle(6)
    before = [(x["name"], x.get("mode")) for x in v.layers]
    ex.click(ex.w.mode_btns[0])
    ex.settle(8)
    w = next(w for w in pp.wires() if "Sortie Gradient" in w.toolTip())
    pp.click_wire(w)
    pp.del_key()
    pp.undo()
    ex.click(ex.w.mode_btns[2])
    ex.settle(10)
    after = [(x["name"], x.get("mode")) for x in v.layers]
    f = ex.shot("n2_layer_settings_lost")
    ok = ("Sortie Gradient", "Screen") in before and ("Sortie Gradient", "Screen") not in after
    return ok, (f"Calque « Sortie Gradient » mis en mode Écran dans Post ; dans le graphe, Suppr sur son fil (la "
                f"branche s'en va) puis Ctrl+Z : la branche et le calque reviennent, mais le calque est revenu en mode "
                f"Normal (avant : {before} ; après Ctrl+Z : {after}). Pour le fil du workflow principal, Ctrl+Z garde "
                "bien les réglages."), f


def n3():
    ex.new_project(f"DeuxImages{time.time():.0f}")
    ex.w.activateWindow()
    ex.double_click_card("image")
    ex.settle(8)
    ex.double_click_card("image")
    ex.settle(8)
    outs = [(b.title, b.summary) for b in pp.blocks_of("post")]
    imgs = pp.blocks_of("image")
    f = ex.shot("n3_two_outputs_no_layer")
    ok = len(outs) >= 2 and pp.final_summary().startswith("0")
    return ok, (f"Projet vierge, double-clic sur Image deux fois (aucun fichier) : un seul bloc Image ({len(imgs)}), "
                f"mais deux workflows ({[q.name for q in ex.s.paths]}) et deux blocs {outs}, alors que le bloc Post "
                f"dit « {pp.final_summary()} ». Les blocs disent « vers Post » sans que rien n'y aille, et le second "
                "double-clic n'a pas fait une deuxième image."), f


def n4():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    ex.click_block(pp.blocks_of("post")[0])
    ex.double_click_card("denoise")
    ex.settle(6)
    br = next((q for q in ex.s.paths if "Gradient" in q.name), None)
    blocks = [n["block"] for n in ex.s.p.workflow(br.pid).blocks] if br else []
    f = ex.shot("n4_branch_grown_name")
    ok = br is not None and "denoise" in blocks
    return ok, (f"Gradient envoyé au Post (branche « Sortie Gradient »), puis la branche sélectionnée (clic sur son bloc "
                f"Sortie) et double-clic sur Débruitage dans la palette : la branche finit maintenant par Débruitage "
                f"({blocks}) mais le calque et le bloc Sortie s'appellent toujours « {br.name if br else ''} »."), f


def n5():
    root = ex.project("mono")
    ex.open(root)
    ex.w.activateWindow()
    ex.double_click_card("fusion")
    ex.settle(8)
    for fl in ("Ha", "OIII"):
        q = next(q for q in ex.s.paths if q.name == fl)
        ex.right_click_block(pp.block("debruitage", pid=q.pid), "Vers Post")
        ex.settle(6)
    ex.click_block(pp.final())
    names = [n for n, _ in pp.graph_layers()]
    f = ex.shot("n5_channel_layer_names")
    ok = not any("Ha" in n for n in names if n.startswith("Sortie"))
    return ok, (f"Projet mono avec Fusion : « Vers Post » sur le Débruitage de Ha puis sur celui d'OIII. Les calques "
                f"s'appellent {names} : rien ne dit lequel est Ha et lequel est OIII (les calques d'Étoiles/fond, eux, "
                "portent le nom du workflow : « Workflow · Image sans étoiles »)."), f


def n6():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("empilement"), "Vers Post")
    ex.settle(6)
    m = ex.right_click_block(pp.block("empilement", pid=ex.s.paths[0].pid))
    texts = [d.get("text") for d in m]
    f = ex.shot("n6_menu_elision")
    ok = any(t and t.startswith("Retirer de E") for t in texts)
    return ok, f"Menu : {texts}. En français : « Retirer d'Empilement brut »." , f


def n7():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    pp.pull_port(pp.block("fond"))
    n = len(ex.s.paths)
    wait(6)                                         # the first message has gone
    pp.pull_port(pp.block("fond", pid=ex.s.paths[0].pid))
    t = pp.toast()
    m = [d.get("text") for d in ex.right_click_block(pp.block("fond", pid=ex.s.paths[0].pid))]
    f = ex.shot("n7_port_twice_silent")
    ok = len(ex.s.paths) == n and not t
    return ok, (f"Le port ● de Gradient tiré une deuxième fois sur Post : rien ne se passe et aucun message ne dit que "
                f"l'image est déjà un calque (message : « {t} »). Le menu du bloc n'a plus « Vers Post » : {m}."), f


def n8():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.click_block(pp.final())
    btn = [b for b in pp.panel_buttons() if "Post" in b[0]]
    ex.double_click_block(pp.final())
    ex.settle(8)
    mode = ex.w.mode
    f = ex.shot("n8_dblclick_post_open")
    ex.w.set_mode(0)
    ex.settle(4)
    ok = bool(btn) and not btn[0][1] and mode == 2
    return ok, (f"Le bouton du panneau {btn} est grisé (« rien de calculé »), mais un double-clic sur le bloc Post (ou "
                "l'onglet Post) ouvre quand même la page Post, qui dit « Pas encore de calque ». Les deux chemins ne "
                "disent pas la même chose."), f


def n9():
    ex2 = ex
    ex2.w.resize(1366, 768)
    ex2.settle(4)
    session_project(ex2)
    build_chain(ex2, CHAINS["stars"])
    vr = ex2.view.mapToScene(ex2.view.viewport().rect()).boundingRect()
    vis = vr.contains(pp.final().sceneBoundingRect())
    f = ex2.shot("n9_1366_post_hidden")
    ex2.w.resize(1600, 950)
    ex2.settle(4)
    return (not vis), ("Fenêtre 1366×768, chaîne jusqu'à Étoiles/fond : le bloc Post et le bloc Recombiner sont hors de "
                       "vue à droite (barre de défilement). Pour tirer un port sur Post il faut d'abord défiler ou "
                       "« Tout afficher », qui rend les textes des blocs minuscules."), f


def n11():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    pp.click_wire(pp.wires()[0])
    pp.del_key()
    wait(6)
    gone = pp.toast()
    ex.right_click_block(pp.block("couleur"), "Vers Post")
    ex.settle(6)
    t = pp.toast()
    f = ex.shot("n11_send_back_silent")
    ok = bool(pp.wires()) and not t
    return ok, (f"Fil du workflow retiré, puis (le message parti : « {gone} ») « Vers Post » dans le menu de Couleur : le "
                f"fil revient sans aucun message (message : « {t} »), alors que « Vers Post » sur un bloc du milieu dit "
                "« Calque … vers Post »."), f


CANDS = {
    "n1": ("gênant", "Calque supprimé dans la page Post : sa branche reste dans le graphe, sans fil",
           ST + ["(état simulé : workflows calculés)", "Clic droit sur Gradient > Vers Post", "Onglet Post",
                 "Sélectionner le calque « Sortie Gradient » > Supprimer", "Onglet Graphe"], n1),
    "n2": ("gênant", "Fil d'une branche retiré puis Ctrl+Z : le calque revient sans ses réglages de Post",
           ST + ["(état simulé : workflows calculés)", "Clic droit sur Gradient > Vers Post",
                 "Onglet Post : calque « Sortie Gradient », Mode de fusion Écran", "Onglet Graphe",
                 "Clic sur le fil de « Sortie Gradient », Suppr", "Ctrl+Z", "Onglet Post"], n2),
    "n3": ("gênant", "Deux blocs « Sortie … vers Post » alors que Post dit « 0 calque »",
           ["Nouveau projet vierge", "Double-clic sur Image dans la palette", "Double-clic sur Image encore"], n3),
    "n4": ("gênant", "Branche envoyée au Post prolongée par un bloc : le calque garde l'ancien nom",
           ST + ["Clic droit sur Gradient > Vers Post", "Clic sur le bloc Sortie de la branche",
                 "Double-clic sur Débruitage dans la palette"], n4),
    "n5": ("mineur", "Calques envoyés depuis deux canaux : rien ne dit lequel est Ha",
           ["Projet mono : sessions Ha et OIII, leurs workflows (Calibration, Alignement, Empilement)",
            "Double-clic sur Fusion", "Clic droit sur Débruitage de Ha > Vers Post",
            "Clic droit sur Débruitage d'OIII > Vers Post", "Clic sur Post"], n5),
    "n6": ("mineur", "Français : « Retirer de Empilement brut » au lieu de « Retirer d'Empilement brut »",
           ST + ["Clic droit sur Empilement > Vers Post", "Clic droit sur Empilement"], n6),
    "n7": ("mineur", "Port tiré une deuxième fois sur Post : rien ne se passe, sans message",
           ST + ["Tirer le port ● de Gradient sur Post", "Attendre que le message parte",
                 "Tirer encore le port ● de Gradient sur Post"], n7),
    "n8": ("mineur", "« Ouvrir Post » grisé, mais le double-clic sur le bloc Post ouvre la page",
           ST + ["Clic sur Post (bouton grisé)", "Double-clic sur Post"], n8),
    "n9": ("mineur", "1366×768 : le bloc Post est hors de vue avec une chaîne complète",
           ["Fenêtre 1366×768", "Nouveau projet avec une session de 15 brutes",
            "Double-clic dans la palette : Calibration … Couleur, Étoiles/fond"], n9),
    "n11": ("mineur", "« Vers Post » qui rend son fil au workflow : aucun message",
            ST + ["Clic sur le fil vers Post, Suppr", "Attendre que le message parte",
                  "Clic droit sur Couleur > Vers Post"], n11),
}
for k in which:
    sev, title, steps, fn = CANDS[k]
    try:
        ok, (_, detail, f) = twice(fn)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("FAILED", k, e)
        continue
    if ok:
        ex.note(sev, title, detail, shot=f, steps=steps)
    else:
        print("NOT REPRODUCED", k)
print("slot", pp.take_slot_errors())
ex.close()
