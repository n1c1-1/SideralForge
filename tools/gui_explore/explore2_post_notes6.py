"""post_wires round 2, sixth batch: the Post page's Supprimer greyed on the only layer (twice, fresh projects)."""
from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start

start(600)
ex = Explorer("post_wires")
pp = P(ex)


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


res = []
for run in (1, 2):
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    v = ex.w.post_holder.page.view
    v.select(v.layers[0]["id"], "layer")
    ex.settle(4)
    f = ex.shot(f"n18_delete_greyed_run{run}")
    res.append((not v.b_del.isEnabled()) and not v.b_del.toolTip())
    print(run, "del enabled", v.b_del.isEnabled(), "tip", repr(v.b_del.toolTip()))
if all(res):
    ex.note("mineur", "Page Post : « Supprimer » grisé sur le seul calque, sans dire pourquoi",
            "Un seul calque (« Workflow ») dans la page Post, sélectionné : le bouton « Supprimer » est grisé, sans "
            "infobulle ni message. Le fil vers Post, lui, se retire dans le graphe.", shot=f,
            steps=["Nouveau projet avec une session de 15 brutes", "Double-clic dans la palette : Calibration … Couleur",
                   "(état simulé : workflow calculé)", "Onglet Post", "Sélectionner le calque « Workflow »"])
ex.close()
