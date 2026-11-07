"""post_wires round 2, second batch: each candidate reproduced twice before it is noted."""
import sys
import time

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start

start(900)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["n12", "n13"]
ST = ["Nouveau projet avec une session de 15 brutes", "Double-clic dans la palette : Calibration, Alignement, "
      "Empilement, Astrométrie, Gradient, Débruitage, Couleur"]


def wait(sec):
    t = time.time()
    while time.time() - t < sec:
        ex.settle(5)


def run_text():
    return [b.text() for b in ex.w.findChildren(type(ex.w.mode_btns[0])) if b.isVisible() and
            ("Exécuter" in b.text())]


def n12():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    pp.click_wire(pp.wires()[0])
    pp.del_key()
    ex.click_block(pp.final())
    panel = pp.panel_texts()
    run = run_text()
    status = [t for t in ex.texts() if "exécuter" in t.lower()]
    f = ex.shot("n12_zero_layer_run")
    ok = pp.final_summary().startswith("0") and len(panel) <= 1 and any("min" in r for r in run)
    return ok, (f"Fil du workflow retiré : le bloc Post dit « {pp.final_summary()} », son panneau ne montre que "
                f"{panel} (rien ne dit qu'aucun calque n'y va, ni comment en renvoyer un), et le bouton dit encore "
                f"{run} : l'exécution calculerait un workflow dont l'image ne va nulle part."), f


def n13():
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    ex.click_block(pp.blocks_of("composition")[0])
    pp.del_key()
    wait(6)
    ex.right_click_block(pp.block("etoiles"), "+ Recombiner")
    ex.settle(6)
    t = pp.toast()
    f = ex.shot("n13_recombine_silent")
    ok = len(pp.blocks_of("composition")) == 1 and not t
    return ok, (f"Recombiner supprimé, puis (le message parti) « + Recombiner » dans le menu d'Étoiles/fond : le bloc "
                f"et son calque reviennent sans aucun message (message : « {t} »), alors que sa suppression dit "
                "« Recombiner retiré · Ctrl+Z »."), f


CANDS = {
    "n12": ("question", "Plus aucun calque dans Post : le panneau est vide et Exécuter calcule quand même",
            ST + ["Clic sur le fil vers Post, Suppr", "Clic sur Post"], n12),
    "n13": ("mineur", "« + Recombiner » : aucun message", ["Nouveau projet avec une session de 15 brutes",
            "Double-clic dans la palette : Calibration … Couleur, Étoiles/fond", "Clic sur Recombiner, Suppr",
            "Attendre que le message parte", "Clic droit sur Étoiles/fond > + Recombiner"], n13),
}
for k in which:
    sev, title, steps, fn = CANDS[k]
    a = fn()
    b = fn()
    print(k, a[0], b[0], "|", b[1], flush=True)
    if a[0] and b[0]:
        ex.note(sev, title, b[1], shot=b[2], steps=steps)
    else:
        print("NOT REPRODUCED", k)
ex.close()
