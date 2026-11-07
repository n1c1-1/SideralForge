"""post_wires F: what goes to Post with image sources (no file, linear, stretched), empty sessions, two flows, mono
channels and fusions (a channel never a layer of its own while it feeds a Fusion)."""
from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton

start(2400)
ex = Explorer("post_wires")
pp = P(ex)


def summary(tag):
    gl = pp.graph_layers()
    print(f"  [{tag}] post={pp.final_summary()!r} layers={gl} wires={[w.toolTip().splitlines()[0] for w in pp.wires()]}")
    print("     paths:", [(q.pid, q.name, (q.source or {}).get('path', '-') if q.source else None,
                           [n['block'] for n in (ex.s.p.workflow(q.pid).blocks or [])]) for q in ex.s.paths])
    return gl


# ---------------------------------------------------------------- 1. blank project: an Image block without a file
print("=" * 20, "image without file")
root = ex.new_project("Vide")
ex.w.activateWindow()
summary("blank")
ex.double_click_card("image")
ex.settle(8)
gl = summary("image placed, no file")
st = ["Nouveau projet vierge", "Double-clic sur Image dans la palette (aucun fichier choisi)"]
if gl:
    pp.note_once("image_nofile_post", "gênant", "Un bloc Image sans fichier compte comme un calque du Post-traitement",
                 f"Le bloc Post-traitement dit « {pp.final_summary()} » et un fil part du bloc Image, alors qu'aucun "
                 "fichier n'est choisi (la règle : rien ne va au Post depuis un workflow vide).", st)
ex.click_block(pp.final())
print("post panel:", pp.panel_texts())
# the Image block's menu
im = pp.block("image")
m = ex.right_click_block(im)
print("image menu:", [d.get("text") for d in m])
# remove its wire, then the menu
if pp.wires():
    pp.click_wire(pp.wires()[0])
    pp.del_key()
    summary("image wire removed")
    m = ex.right_click_block(pp.block("image"))
    print("image menu after:", [d.get("text") for d in m])
    if not any("nvoyer" in (d.get("text") or "") for d in m):
        pp.note_once("image_no_sendback", "gênant", "Bloc Image : fil retiré, pas de « Envoyer son image au Post »",
                     f"Menu : {[d.get('text') for d in m]}", st + ["Retirer son fil vers Post-traitement"])
    ex.right_click_block(pp.block("image"), "Vers Post")
    summary("image sent back")

# ---------------------------------------------------------------- 2. a blank project: an empty Session (no frames)
print("=" * 20, "empty session")
ex.new_project("VideSession")
ex.w.activateWindow()
ex.double_click_card("session")
ex.settle(8)
summary("empty session")
build_chain(ex, ["calibration", "registration", "stacking"])
gl = summary("empty session chain")
if gl:
    pp.note_once("empty_session_post", "question", "Une session sans brute envoie un calque au Post-traitement",
                 f"Session placée vide, puis Calibration, Alignement, Empilement : le bloc Post-traitement dit "
                 f"« {pp.final_summary()} » et un fil part d'Empilement alors qu'il n'y a aucune brute.",
                 ["Nouveau projet vierge", "Double-clic sur Session (vide)", "Double-clic : Calibration, Alignement, "
                                                                            "Empilement"])

# ---------------------------------------------------------------- 3. linear image file, then stretched
for stretched in (False, True):
    print("=" * 20, "image file stretched" if stretched else "image file linear")
    root = ex.project("image")
    ex.open(root)
    ex.w.activateWindow()
    st = ["Projet avec un bloc Image (fichier FITS linéaire)"]
    im = pp.block("image")
    if stretched:
        ex.click_block(im)
        b = [x for x in ex.page.panel.findChildren(QAbstractButton) if x.isVisible() and x.text() in ("Étirée", "Stretched")]
        print("kind buttons:", [(x.text(), x.isEnabled(), x.isChecked()) for x in ex.page.panel.findChildren(QAbstractButton)
                                if x.isVisible()])
        if b:
            ex.click(b[0])
            ex.settle(6)
        st.append("Panneau du bloc Image : Type « Étirée »")
    summary("image source")
    print("cards:", {k: c.reason for k, c in ex.page.cards.items()})
    blocks = ["stars"] if stretched else ["astrometry", "gradient", "color"]
    for b in blocks:
        ex.double_click_card(b)
        ex.settle(6)
        gl = summary(f"after {b}")
    pp.check_slot_errors("chaîne sur une image", st)
    if stretched:
        sp = pp.block("etoiles")
        print("stretched split:", sp is not None, sp.ports if sp else None, [c.title for c in pp.blocks_of("composition")])
        if sp is not None:
            pp.pull_port(sp, "stars")
            summary("stretched: stars to Post")
            m = ex.right_click_block(pp.block("etoiles"))
            print("stretched split menu:", [d.get("text") for d in m])
    else:
        # the middle blocks' images: Gradient
        pp.pull_port(pp.block("fond"))
        summary("image: gradient sent")
        it = pp.block("image")
        m = ex.right_click_block(it)
        print("image menu:", [d.get("text") for d in m])
        # the source's image itself: can it go to Post? (it has no port)
        print("image block ports:", pp.block("image").ports)
        if not pp.block("image").ports and not any("nvoyer" in (d.get("text") or "") for d in m):
            pp.note_once("image_source_no_send", "question", "L'image d'un bloc Image ne peut pas être envoyée au "
                         "Post-traitement telle quelle", "Le bloc Image donne une image mais n'a ni port ● ni entrée "
                         "« Envoyer son image au Post » quand des blocs le suivent (la règle : tout bloc qui donne une "
                         "image l'envoie au Post).", st + ["Double-clic : Astrométrie, Gradient, Couleur",
                                                            "Clic droit sur le bloc Image"])

# ---------------------------------------------------------------- 4. two flows
print("=" * 20, "two flows")
root = ex.project("two_flows")
ex.open(root)
ex.w.activateWindow()
summary("two flows empty")
for pid in [q.pid for q in ex.s.paths]:
    for b in CHAINS["gradient"]:
        ex.page.place_block(b, pid)
        ex.settle(4)
gl = summary("two flows built")
st = ["Projet à deux flux parallèles (deux sessions)", "Chaque flux : Calibration … Gradient"]
if len(gl) != 2 or len(pp.wires()) != 2:
    pp.note_once("two_flows_count", "gênant", "Deux flux : le Post-traitement n'a pas un calque par flux",
                 f"Calques : {gl} ; fils : {len(pp.wires())}", st)
# remove flow 2's wire, Ctrl+Z
w2 = [w for w in pp.wires() if "Flow" in w.toolTip() or "Flux" in w.toolTip()]
if w2:
    pp.click_wire(w2[0])
    pp.del_key()
    summary("flow 2 wire removed")
    # a middle block of flow 1 sent: does it go to Post now?
    p1 = ex.s.paths[0]
    pp.pull_port(pp.block("empilement", pid=p1.pid))
    gl = summary("flow1 stacking sent after flow2 wire removed")
    if len(ex.s.paths) == 3 and len(gl) == 1:
        pp.note_once("two_flows_send_lost", "bloquant", "Deux flux : après le retrait du fil d'un flux, l'image envoyée "
                     "depuis l'autre flux ne va pas au Post-traitement",
                     f"La branche « {ex.s.paths[-1].name} » est créée, le message l'annonce comme calque, mais le bloc "
                     f"Post-traitement reste à « {pp.final_summary()} » sans fil pour elle.",
                     st + ["Retirer le fil vers Post-traitement du flux 2", "Tirer le port ● d'Empilement du flux 1 sur "
                                                                           "Post-traitement"])
# ---------------------------------------------------------------- 5. mono channels and Fusion
print("=" * 20, "mono + fusion")
root = ex.project("mono")
ex.open(root)
ex.w.activateWindow()
gl = summary("mono")
st = ["Projet mono : sessions Ha et OIII, leurs workflows (Calibration, Alignement, Empilement)"]
ex.double_click_card("fusion")
ex.settle(8)
pp.check_slot_errors("Fusion placée", st)
gl = summary("fusion placed")
names = [n for n, _ in gl]
st2 = st + ["Double-clic sur Fusion dans la palette"]
chan = [n for n in names if n in ("Ha", "OIII")]
if chan:
    pp.note_once("mono_channel_layer", "gênant", "Un canal d'une Fusion est aussi un calque du Post-traitement",
                 f"Calques : {names}", st2)
fu = pp.block("fusion")
print("fusion block:", fu.title if fu else None, fu.summary if fu else None, fu.ports if fu else None)
m = ex.right_click_block(fu) if fu else []
print("fusion menu:", [d.get("text") for d in m])
# a channel's last block: its menu, its port
for f in ("Ha", "OIII"):
    q = next(q for q in ex.s.paths if q.name == f)
    it = pp.block("empilement", pid=q.pid)
    m = ex.right_click_block(it)
    texts = [d.get("text") for d in m]
    print(f"channel {f} stacking menu:", texts, "ports", it.ports)
    pp.check_texts("menu d'un canal", pp.menu_texts(m), st2)
    if any("Vers Post" in (t or "") for t in texts):
        ex.right_click_block(it, "Vers Post")
        g2 = summary(f"channel {f} sent")
        if any(n == f for n, _ in g2) or len(g2) > len(gl):
            print("channel became a layer")
            pp.note_once("channel_send_menu", "question", "Le menu d'un canal de Fusion offre « Envoyer son image au "
                         "Post » : le canal devient alors un calque à part",
                         f"Projet mono, Fusion HOO placée : le dernier bloc du canal {f} (Empilement) propose « Envoyer "
                         f"son image au Post ». Choisi, {f} devient un calque à côté de HOO ({[n for n, _ in g2]}). Voulu, "
                         "ou un canal ne doit jamais être un calque tant qu'il alimente une Fusion ?",
                         st2 + [f"Clic droit sur Empilement de {f} > Envoyer son image au Post"])
        break
# the fusion's wire: removable? removed, Ctrl+Z
fw = [w for w in pp.wires() if fu is not None]
print("wires now:", [w.toolTip() for w in pp.wires()])
# delete the Fusion: do the channels come back as layers?
fu = pp.block("fusion")
if fu is not None:
    ex.right_click_block(fu, "Supprimer")
    pp.check_slot_errors("Supprimer la Fusion", st2)
    g3 = summary("fusion deleted")
    pp.undo()
    g4 = summary("fusion delete undone")
    if len(g4) != len(gl) and not pp.block("fusion"):
        pp.note_once("fusion_undo", "gênant", "Ctrl+Z après suppression de la Fusion ne la remet pas", "", st2)
# dual band: a colour chain, then Fusion
print("=" * 20, "dual band")
session_project(ex)
build_chain(ex, CHAINS["color"])
summary("colour chain")
ex.double_click_card("fusion")
ex.settle(8)
g = summary("fusion of colour channels")
st3 = ["Projet couleur, chaîne jusqu'à Couleur", "Double-clic sur Fusion"]
if any(n == "Workflow" for n, _ in g):
    pp.note_once("dual_channel_layer", "question", "Fusion bi-bande : le workflow couleur reste aussi un calque",
                 f"Calques : {g}", st3)
ex.shot("fusion_dualband")
ex.close()
