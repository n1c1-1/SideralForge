"""post_wires C: Stars/background: its two ports pulled to Post and to Composition, taken out of Post, the
Recombine blocks (palette double-click and drag, menu, panel button, delete, Ctrl+Z / Ctrl+Y)."""
from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start, steps_chain
from PySide6.QtCore import Qt

start(2400)
ex = Explorer("post_wires")
pp = P(ex)
ST = steps_chain("stars")


def state(tag):
    comps = pp.blocks_of("composition")
    direct = [w.toolTip().split("\n")[0] for w in pp.wires()]
    summ = pp.final_summary()
    gl = pp.graph_layers()
    print(f"  [{tag}] summary={summ!r} comps={[(c.node.lid, c.title) for c in comps]} direct={direct} graph={gl}")
    print(f"        saved={[(x.get('kind'), x.get('part'), x.get('name')) for x in pp.saved_layers()]} "
          f"edited={(getattr(ex.s.p, 'post', None) or {}).get('edited')} toast={pp.toast()!r}")
    num = int(summ.split()[0]) if summ.split()[0].isdigit() else -1
    if num != len(gl):
        pp.note_once("stars_count_" + tag, "gênant", "Nombre de calques du bloc Post-traitement faux (étoiles)",
                     f"Le bloc dit « {summ} », la liste en a {len(gl)} : {gl}", ST + [tag])
    return comps, direct, gl


def fresh():
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    return pp.block("etoiles")


# ---------------------------------------------------------------- 1. the ports pulled onto Post
print("=" * 20, "ports to Post")
sp = fresh()
print("ports:", sp.ports, "tip:", sp.toolTip())
ex.click_block(sp)
print("split panel:", pp.panel_texts())
pp.check_texts("panneau Split étoiles", pp.panel_texts(), ST)
c0, d0, g0 = state("start")
pp.check_texts("liste des calques (graphe)", [n for n, _ in g0], ST)
ex.click_block(pp.final())
pp.check_texts("panneau Post-traitement", pp.panel_texts(), ST)
print("post panel:", pp.panel_texts())
sp = pp.block("etoiles")
pp.pull_port(sp, "starless")
pp.check_slot_errors("port « sans étoiles » tiré sur Post-traitement", ST)
c1, d1, g1 = state("starless to Post")
if len(d1) != 1:
    pp.note_once("starless_port", "gênant", "Port « image sans étoiles » tiré sur le Post-traitement : pas de fil",
                 f"Fils amovibles : {d1}", ST + ["Tirer le ● du haut de Split étoiles sur Post-traitement"])
if len(c1) != len(c0):
    pp.note_once("starless_port_comp", "question", "Envoyer l'image sans étoiles au Post fait disparaître/apparaître "
                 "un bloc Combiner", f"Avant {len(c0)} bloc(s) Combiner, après {len(c1)}.", ST)
ex.shot("stars_starless_sent")
pp.pull_port(pp.block("etoiles"), "stars")
pp.check_slot_errors("port « étoiles seules » tiré sur Post-traitement", ST)
c2, d2, g2 = state("stars to Post")
pp.pull_port(pp.block("etoiles"), "starless")             # twice
c3, d3, g3 = state("starless twice")
print("toast twice:", pp.toast())
if len(g3) != len(g2):
    pp.note_once("starless_twice", "mineur", "Image sans étoiles envoyée deux fois : deux calques", f"{g3}", ST)
pp.check_texts("message", [pp.toast()], ST)
# panel of the Stars/background block now
ex.click_block(pp.block("etoiles"))
print("split panel after:", pp.panel_texts(), pp.panel_buttons())
# the Post block panel: names
ex.click_block(pp.final())
pt = pp.panel_texts()
print("post panel:", pt)
pp.check_texts("panneau Post-traitement", pt, ST + ["Envoyer les deux images de Split au Post-traitement"])
# hover / click / Del on the dashed wire of the stars alone
ws = pp.wires()
w = next((x for x in ws if "toiles seules" in x.toolTip() or "stars" in x.toolTip().lower()), ws[0] if ws else None)
if w is not None:
    tip = w.toolTip()
    if pp.free_point_on(w) is None:
        pp.note_once("direct_hidden", "gênant", "Fil pointillé d'une image de Split inatteignable",
                     f"Aucun point libre du fil « {tip.splitlines()[0]} » : il passe sous les blocs.", ST)
    else:
        pp.click_wire(w)
        pp.del_key()
        pp.check_slot_errors("Suppr sur le fil des étoiles seules", ST)
        c4, d4, g4 = state("Del on stars wire")
        if len(d4) != len(d3) - 1:
            pp.note_once("direct_del", "gênant", "Suppr sur un fil pointillé de Split ne le retire pas", tip, ST)
        pp.undo()
        c5, d5, g5 = state("undo of Del on stars wire")
        if len(d5) != len(d3):
            pp.note_once("direct_undo", "gênant", "Ctrl+Z ne remet pas l'image de Split retirée du Post-traitement",
                         f"Après Suppr sur le fil « {tip.splitlines()[0]} » puis Ctrl+Z, le fil ne revient pas. "
                         f"Workflows après Ctrl+Z : {[(q.name, [n['block'] for n in ex.s.p.workflow(q.pid).blocks]) for q in ex.s.paths]}",
                         ST + ["Envoyer les étoiles seules au Post (port)", "Clic sur son fil pointillé", "Suppr",
                               "Ctrl+Z"])
        if pp.block("etoiles") is None and g5:
            pp.note_once("stale_layers", "gênant", "Split étoiles disparu, mais le Post-traitement compte encore ses "
                         "calques", f"Le bloc Post-traitement dit « {pp.final_summary()} » et son panneau liste "
                         f"{[n for n, _ in g5]}, alors qu'aucun bloc Split ni fil ne les fournit plus.",
                         ST + ["Envoyer l'image sans étoiles au Post", "Ctrl+Z jusqu'à retirer Split"])
        if pp.block("etoiles") is None:
            pp.redo()
            state("redo after undo removed Split")
# the wire's menu on the starless one
ws = pp.wires()
if ws:
    m = pp.right_click_wire(ws[0], "Retirer")
    pp.check_slot_errors("menu du fil pointillé > Retirer", ST)
    print("direct wire menu:", m)
    state("wire menu remove")
# menu "Take ... out of Post"
sp = pp.block("etoiles")
m = ex.right_click_block(sp)
texts = [d.get("text") for d in m]
print("split menu:", texts)
pp.check_texts("menu Split", pp.menu_texts(m), ST)
out = next((t for t in texts if t and "Retirer" in t and "Post" in t), None)
if out:
    ex.right_click_block(sp, out)
    pp.check_slot_errors(out, ST)
    state("menu take out")

# ---------------------------------------------------------------- 2. Combine blocks
print("=" * 20, "Combine blocks")
sp = fresh()
c0, d0, g0 = state("start")
# the auto Combine block: delete it (its cross)
if c0:
    cb = c0[0]
    ex.click_block(cb)
    print("combine panel:", pp.panel_texts()[:8], pp.panel_buttons())
    pp.check_texts("panneau Combiner", pp.panel_texts(), ST)
    pp.del_key()
    pp.check_slot_errors("Suppr sur le bloc Combiner", ST)
    c1, d1, g1 = state("auto combine deleted (Del)")
    if len(c1) == len(c0):
        pp.note_once("comb_del", "gênant", "Suppr sur le bloc Combiner ne le retire pas", "", ST + ["Clic sur Combiner",
                                                                                                    "Suppr"])
    if not g1:
        print("  Post block has no layer now; split menu:", [d.get("text") for d in ex.right_click_block(pp.block("etoiles"))])
    pp.undo()
    c2, d2, g2 = state("undo combine delete")
    if len(c2) != len(c0):
        pp.note_once("comb_undo", "gênant", "Ctrl+Z ne remet pas le bloc Combiner supprimé",
                     f"Après Suppr sur le bloc Combiner étoiles et fond puis Ctrl+Z, il ne revient pas. Workflows : "
                     f"{[(q.name, [n['block'] for n in ex.s.p.workflow(q.pid).blocks]) for q in ex.s.paths]}",
                     ST + ["Clic sur le bloc Combiner les calques étoiles et fond", "Suppr", "Ctrl+Z"])
    pp.redo()
    state("redo")
# palette double-click: a new Combine
ex.click_block(pp.block("etoiles"))
card = ex.card("composition")
print("combine card reason:", card.reason if card else None)
ex.double_click_card("composition")
pp.check_slot_errors("double-clic sur Combiner dans la palette", ST)
c3, d3, g3 = state("palette double-click")
# palette drag onto its ghost
ex.drag_palette_hover("composition", ex.view.viewport().rect().center())
gh = ex.ghosts()
print("ghosts while dragging Combine:", [(g.text, g.rect) for g in gh])
centres = [ex.ghost_center(g) for g in gh]
ex.drag_leave()
if gh:
    rec = ex.drag_palette("composition", centres[0])
    print("drop:", rec)
    pp.check_slot_errors("glisser Combiner depuis la palette", ST)
    c4, d4, g4 = state("palette drag")
    # overlapping Combine blocks?
    rects = [c.sceneBoundingRect() for c in pp.blocks_of("composition")]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if rects[i].intersects(rects[j]) and rects[i].intersected(rects[j]).height() > 4:
                pp.note_once("comb_overlap", "gênant", "Blocs Combiner superposés",
                             f"{len(rects)} blocs Combiner : deux se chevauchent.", ST + ["Ajouter plusieurs blocs Combiner"])
    ex.shot("stars_many_combine")
else:
    pp.note_once("comb_noghost", "gênant", "Glisser Combiner depuis la palette : aucune place ne s'allume", "", ST)
# menu New Combine
m = ex.right_click_block(pp.block("etoiles"), "+ Recombiner")
pp.check_slot_errors("menu Nouveau bloc Combiner", ST)
c5, d5, g5 = state("menu new combine")
# port to composition
r = pp.pull_port(pp.block("etoiles"), "stars", to="composition")
print("pull to composition:", r)
pp.check_slot_errors("port tiré sur Composition", ST)
c6, d6, g6 = state("port to Composition")
if r == "no ghost":
    pp.note_once("comp_ghost", "gênant", "Port de Split tiré : pas de place « Combiner » en Composition", "", ST)
# names of the Combine blocks: numbered?
titles = [c.title for c in pp.blocks_of("composition")]
print("combine titles:", titles)
if len(titles) > 1 and len(set(titles)) == 1:
    pp.note_once("comb_numbers", "mineur", "Plusieurs blocs Combiner portent le même nom",
                 f"{len(titles)} blocs Combiner, tous « {titles[0]} » : on ne sait pas lequel est quel calque du "
                 "Post-traitement (les blocs placés deux fois sont numérotés ailleurs).", ST + ["Ajouter 3 blocs Combiner"])
ex.click_block(pp.final())
print("post panel with many combine:", pp.panel_texts())
# Combine block menu
cb = pp.blocks_of("composition")[-1]
m = ex.right_click_block(cb)
print("combine menu:", m)
pp.check_texts("menu Combiner", pp.menu_texts(m), ST)
if m and not any("\t" in (d.get("text") or "") for d in m):
    pp.note_once("comb_menu_del", "mineur", "Menu du bloc Combiner : « Supprimer le bloc » sans le raccourci Suppr",
                 "Les autres blocs affichent « Supprimer le bloc  Suppr » ; celui-ci non.",
                 ST + ["Clic droit sur Combiner les calques étoiles et fond"])
n_before = len(pp.blocks_of("composition"))
ex.right_click_block(cb, "Supprimer")
pp.check_slot_errors("menu Combiner > Supprimer le bloc", ST)
state("combine deleted by menu")
# cross of a Combine
cb = pp.blocks_of("composition")[0]
ex.hover(ex.view.viewport(), ex.point_of(cb))
ex.click(ex.view.viewport(), ex.point_of(cb, "cross"))
pp.check_slot_errors("croix du bloc Combiner", ST)
state("combine deleted by its cross")
# undo once: the Combine deleted by its cross comes back?
pp.undo()
cu, du, gu = state("undo after cross")
if pp.block("etoiles") is None:
    pp.note_once("undo_removes_split", "gênant", "Ctrl+Z après la suppression d'un bloc Combiner retire Split étoiles",
                 "Le bloc Combiner supprimé (croix) ne revient pas ; Ctrl+Z défait à la place l'ajout de Split étoiles et "
                 f"fond. Le bloc Post-traitement dit encore « {pp.final_summary()} » pour des calques qui n'ont plus de "
                 "source.", ST + ["Ajouter des blocs Combiner", "Croix d'un bloc Combiner", "Ctrl+Z"])
while ex.s.can_redo():
    pp.redo()
state("redo all")
# delete the Stars/background block while Combine blocks hang on it
sp = pp.block("etoiles")
m = ex.right_click_block(sp)
ex.right_click_block(sp, "Supprimer")
pp.check_slot_errors("Supprimer Split avec des blocs Combiner", ST)
c7, d7, g7 = state("split deleted")
print("paths:", [(q.name, [n['block'] for n in ex.s.p.workflow(q.pid).blocks]) for q in ex.s.paths])
if c7:
    pp.note_once("comb_orphan", "gênant", "Split supprimé : des blocs Combiner restent", "", ST)
if pp.block("etoiles") is None and g7 and not pp.wires():
    pp.note_once("split_deleted_stale", "gênant", "Split supprimé : le Post-traitement compte encore ses calques, et "
                 "l'image du workflow ne va plus au Post", f"Après « Supprimer le bloc » sur Split étoiles et fond (avec "
                 f"des blocs Combiner), le bloc Post-traitement dit « {pp.final_summary()} » ({[n for n, _ in g7]}), sans "
                 "aucun fil : ni les calques Combiner (plus de source), ni l'image de Couleur ne vont au Post.",
                 ST + ["Ajouter des blocs Combiner", "Clic droit sur Split > Supprimer le bloc"])
pp.undo()
c8, d8, g8 = state("undo split delete")
if len(c8) == 0:
    pp.note_once("split_undo_comb", "gênant", "Ctrl+Z après suppression de Split : les blocs Combiner ne reviennent pas",
                 f"Split revient ({pp.block('etoiles') is not None}) mais ses blocs Combiner et ses fils vers le Post sont "
                 "perdus.", ST + ["Ajouter des blocs Combiner", "Supprimer Split étoiles et fond", "Ctrl+Z"])
ex.close()
