"""post_wires round 2, third batch: each candidate reproduced twice before it is noted."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start

start(900)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["n14", "n15"]


def chains():
    return {q.name: [n['block'] for n in (ex.s.p.workflow(q.pid).blocks or [])] for q in ex.s.paths}


FR = {"calibration": "Calibration", "registration": "Alignement", "stacking": "Empilement", "denoise": "Débruitage",
      "stars": "Étoiles/fond", "output": None}


def fr(blocks):
    return ", ".join(FR[b] for b in blocks if FR.get(b))


def n14():
    root = ex.project("mono")
    ex.open(root)
    ex.w.activateWindow()
    before = chains()
    lay0 = [n for n, _ in pp.graph_layers()]
    ex.double_click_card("fusion")
    ex.settle(8)
    ex.right_click_block(pp.block("fusion"), "Supprimer")
    ex.settle(6)
    after = chains()
    lay1 = [n for n, _ in pp.graph_layers()]
    f = ex.shot("n14_fusion_deleted_channels")
    ok = after.get("Ha") != before.get("Ha")
    return ok, (f"Avant la Fusion, Ha = {fr(before['Ha'])} et les calques de Post sont {lay0}. Fusion placée puis "
                f"supprimée (menu Supprimer, « Fusion et suivants retirés ») : Ha = {fr(after.get('Ha', []))}, et les "
                f"calques sont maintenant {[x.replace('Recombine', 'Recombiner') for x in lay1]} avec deux blocs "
                "Recombiner. La Fusion a ajouté Débruitage et Étoiles/fond à chaque canal ; sa suppression les laisse, "
                "et les images des canaux ne vont plus au Post telles quelles."), f


def n15():
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    m = ex.right_click_block(pp.block("etoiles"))
    items = [d.get("text") for d in m if "Post" in (d.get("text") or "")]
    ex.right_click_block(pp.block("etoiles"), "Vers Post")
    ex.settle(6)
    lay = [n.replace("Recombine", "Recombiner") for n, _ in pp.graph_layers()]
    f = ex.shot("n15_split_own_send")
    ok = "Workflow" in lay
    return ok, (f"Le menu d'Étoiles/fond offre {items}. « Vers Post » envoie l'image d'avant le retrait des étoiles "
                f"(celle de Couleur) : le nouveau calque s'appelle « Workflow » ({lay}). Ni le menu ni le calque ne "
                "disent de quelle image il s'agit, à côté de « Sans étoiles vers Post » et « Étoiles vers Post »."), f


CANDS = {
    "n14": ("gênant", "Fusion supprimée : les canaux gardent ce que la Fusion leur a ajouté",
            ["Projet mono : sessions Ha et OIII, leurs workflows (Calibration, Alignement, Empilement)",
             "Double-clic sur Fusion", "Clic droit sur Fusion > Supprimer"], n14),
    "n15": ("mineur", "« Vers Post » d'Étoiles/fond : on ne sait pas quelle image part",
            ["Nouveau projet avec une session de 15 brutes", "Double-clic dans la palette : Calibration … Couleur, "
             "Étoiles/fond", "Clic droit sur Étoiles/fond > Vers Post"], n15),
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
