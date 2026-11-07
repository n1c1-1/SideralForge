"""sources_and_files, part 4: the Fusion block. Placed with mono filter workflows of various sets (Ha+OIII,
SII+Ha+OIII, R+G+B, L+R+G+B, one filter only) and with a colour workflow (dual band); its panel: palette buttons,
channels, weights, stars, rename; a channel's workflow deleted afterwards; deleting the Fusion; Ctrl+Z."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_sources_common import (  # noqa: E402
    Explorer, Notes, _, button, graph_texts, look, missing_fr, panel, ptexts, start, undo, visible, wait,
)
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QPushButton  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

start(1800)
ex = Explorer("sources_files")
N = Notes(ex)
LOG: list[str] = []


def log(*a):
    t = " ".join(str(x) for x in a)
    LOG.append(t)
    print(t, flush=True)


CHAIN = ("calibration", "registration", "stacking", "astrometry", "gradient")


def project_of(name: str, filters=(), colour=False, chain=CHAIN) -> Path:
    """Sessions of these filters (FITS with FILTER), each with its own workflow; ``colour``: a colour session and its
    workflow."""
    root = ex.ws / name
    p = project2.create(root, name, Target(name="M 31"))
    p.stack_mode = "per_session"
    for i, f in enumerate(filters):
        p.sessions.append(project2.Session(i + 1, f, lights=[str(ex.fits_frames(f"{name}/{f}", 4, f))], filter=f))
    if colour:
        p.sessions.append(project2.Session(len(filters) + 1, "Couleur", lights=[str(ex.raws(f"{name}/osc", 6))]))
    project2.save(p)
    for i, f in enumerate(filters):
        w = p.add_workflow(f, wfl.TEMPLATES["Classic stack"])
        w.sessions = [i + 1]
        p.set_blocks(w.id, [wfl.node(b) for b in chain])
    if colour:
        w = p.add_workflow("Couleur", wfl.TEMPLATES["Classic stack"])
        w.sessions = [len(filters) + 1]
        p.set_blocks(w.id, [wfl.node(b) for b in chain + ("color",)])
    project2.save(p)
    return root


def fusion_paths():
    return [q for q in ex.s.paths if q.source is not None and "fusion" in q.source]


def fusion_block(i=0):
    bs = [b for b in ex.blocks() if b.node.key == "fusion"]
    return bs[i] if len(bs) > i else None


def sel_fusion(i=0):
    it = fusion_block(i)
    if it is not None:
        ex.click_block(it)
        wait(ex, 0.3)
    return it


def palette_buttons():
    from astro_gpu.stages.fusion import PALETTES
    return {b.text(): b for b in visible(panel(ex), QPushButton) if b.text() in PALETTES}


def fusion_state(q=None):
    q = q or (fusion_paths()[0] if fusion_paths() else None)
    if q is None:
        return None
    q = ex.s.view(q.pid) or q
    return {"name": q.name, "fusion": (q.source or {}).get("fusion")}


def open_project(root):
    if ex.w is not None and ex.w.project is not None:
        ex.w.close_project()
        wait(ex, 0.4)
    ex.open(root)
    wait(ex, 1.0)
    ex.w.activateWindow()


before_missing = missing_fr()
SETS = [("HaOIII", ("Ha", "OIII"), "HOO"), ("SHO3", ("SII", "Ha", "OIII"), "SHO"), ("RGB3", ("R", "G", "B"), "RGB"),
        ("LRGB4", ("L", "R", "G", "B"), "LRGB"), ("HaOnly", ("Ha",), None)]
toasts: list[str] = []
for name, filters, expect in SETS:
    root = project_of(f"fus_{name}", filters)
    open_project(root)
    ex.page.toasted.connect(lambda t: toasts.append(t))
    card = ex.card("fusion")
    reason = card.reason if card is not None else "?"
    log("SET", name, filters, "fusion card reason:", repr(reason))
    steps = [f"projet de {len(filters)} filtre(s) {'+'.join(filters)}, un workflow par filtre",
             "double-clic sur Fusion dans la palette"]
    n_modals = len(ex.modals)
    ex.double_click_card("fusion")
    wait(ex, 1.0)
    fs = fusion_state()
    log("SET", name, "placed:", fs, "modals:", ex.modals[n_modals:], "toast:", toasts[-1:])
    ex.shot(f"fus_{name}_placed")
    if ex.modals[n_modals:]:
        N.note("bloquant", "Placer une Fusion ouvre une fenêtre", str(ex.modals[n_modals:]), steps=steps)
    if expect is None:
        if fs is not None:
            N.note("gênant", "Fusion placée avec un seul filtre", str(fs), steps=steps)
        if not reason:
            N.note("mineur", "Un seul filtre : la carte Fusion n'est pas grisée", f"toast : {toasts[-1:]}", steps=steps)
        else:
            # the hint shown over the graph
            ex.page.show_hint(card, reason)
            ex.settle(2)
            ex.shot(f"fus_{name}_hint")
            log("SET hint", ex.page.hint.text())
            ex.page.hide_hint()
        continue
    if fs is None:
        N.note("gênant", f"Fusion non placée avec {'+'.join(filters)}", f"toast : {toasts[-1:]}", steps=steps)
        continue
    if fs["fusion"]["palette"] != expect:
        N.note("question", f"Fusion placée avec {'+'.join(filters)} : palette {fs['fusion']['palette']} (attendu "
                           f"{expect})", str(fs), steps=steps)
    sel_fusion()
    pt = ptexts(ex)
    log("SET panel", pt)
    look(ex, N, f"panneau Fusion ({'+'.join(filters)})", steps=steps)
    ex.shot(f"fus_{name}_panel")
    pb = palette_buttons()
    log("SET palettes", {k: (b.isEnabled(), b.isChecked(), b.toolTip()) for k, b in pb.items()})
    # role and text glued: "SIISII, son gris"
    from PySide6.QtWidgets import QLabel
    roles = [lb for lb in visible(panel(ex), QLabel) if lb.text().startswith("<b>")]
    if roles:
        r0 = roles[0]
        nxt = [lb for lb in visible(panel(ex), QLabel) if lb is not r0 and abs(lb.y() - r0.y()) < 6 and lb.x() > r0.x()]
        if nxt and nxt[0].x() - (r0.x() + r0.width()) < 4:
            N.note("mineur", "Panneau Fusion : le nom du canal collé à son texte (« SIISII, son gris »)",
                   f"« {r0.text()} » finit à x={r0.x() + r0.width()}, le texte suivant commence à x={nxt[0].x()}.",
                   steps=steps)
    # the block's graph: the fusion's wires and its place
    log("SET graph", graph_texts(ex))
    # the channels' workflows need Astrometry: here each has it (CHAIN)

# ------------------------------------------------------------------ the panel of a SHO fusion: palettes, weights, stars, rename
root = ex.ws / "fus_SHO3"
open_project(root)
ex.page.toasted.connect(lambda t: toasts.append(t))
if not fusion_paths():                       # the one placed above, saved with the project
    ex.double_click_card("fusion")
    wait(ex, 1.0)
q = fusion_paths()[0]
log("PANEL fusion", q.name, len(fusion_paths()))
sel_fusion()
steps = ["projet SII+Ha+OIII", "Fusion placée", "clic sur la Fusion"]
for pal in ("HSO", "HOO", "SHO"):
    pb = palette_buttons()
    b = pb.get(pal)
    if b is None or not b.isEnabled():
        log("PAL", pal, "not enabled")
        continue
    ex.click(b)
    wait(ex, 0.6)
    sel_fusion()
    fs = fusion_state(q)
    log("PAL", pal, fs["fusion"]["palette"], ptexts(ex)[:14], graph_texts(ex))
    ex.shot(f"fus_palette_{pal}")
    N.check(fs["fusion"]["palette"] == pal, "gênant", f"Bouton de palette {pal} sans effet", str(fs), steps=steps)
    if pal == "HOO":
        rows = [t for t in ptexts(ex) if "SII" in t]
        if rows:
            N.note("question", "Fusion en HOO : le canal SII reste listé (avec son poids) alors que la palette ne "
                               "le lit pas", f"Lignes : {rows}", steps=steps + ["Palette : HOO"])
        summ = [s for k, t, s in graph_texts(ex) if k == "fusion"]
        log("PAL HOO summary", summ)
undo(ex)
wait(ex, 0.6)
log("PAL undo", fusion_state(q)["fusion"]["palette"])
N.check(fusion_state(q)["fusion"]["palette"] == "HOO", "gênant", "Ctrl+Z ne rend pas la palette précédente de la Fusion",
        str(fusion_state(q)), steps=steps + ["HSO, HOO, SHO", "Ctrl+Z"])
undo(ex, redo=True)
wait(ex, 0.6)
# weights
sel_fusion()
sps = visible(panel(ex), QDoubleSpinBox)
log("W spins", [(s.value(), s.toolTip()[:40]) for s in sps])
if sps:
    sp = sps[0]
    sp.setFocus()
    sp.selectAll()
    QTest.keyClicks(sp, "0,6")
    QTest.keyClick(sp, Qt.Key_Return)
    wait(ex, 0.6)
    log("W after", fusion_state(q)["fusion"].get("weights"))
    N.check(bool(fusion_state(q)["fusion"].get("weights")), "gênant", "Poids d'un canal de la Fusion : sans effet",
            str(fusion_state(q)), steps=steps + ["premier canal : poids 0,6, Entrée"])
    sel_fusion()
    sps = visible(panel(ex), QDoubleSpinBox)
    log("W shown", [s.value() for s in sps])
    undo(ex)
    wait(ex, 0.6)
    log("W undo", fusion_state(q)["fusion"].get("weights"))
    N.check(not fusion_state(q)["fusion"].get("weights"), "gênant", "Ctrl+Z ne rend pas le poids d'un canal",
            str(fusion_state(q)), steps=steps + ["poids 0,6", "Ctrl+Z"])
    # a weight of 0, 5 (the limits)
    sel_fusion()
    sps = visible(panel(ex), QDoubleSpinBox)
    sp = sps[1]
    sp.setFocus()
    sp.selectAll()
    QTest.keyClicks(sp, "9")
    QTest.keyClick(sp, Qt.Key_Return)
    wait(ex, 0.6)
    log("W 9 ->", fusion_state(q)["fusion"].get("weights"))
# stars
sel_fusion()
cbs = visible(panel(ex), QComboBox)
log("S combos", [[c.itemText(i) for i in range(c.count())] for c in cbs])
if cbs:
    cb = cbs[0]
    others = {x.pid: (x.opts.get("etoiles"), x.opts.get("debruitage")) for x in ex.s.paths if x.source is None}
    log("S before", cb.currentText(), others)
    cb.setCurrentIndex(0)
    cb.activated.emit(0)
    wait(ex, 0.8)
    log("S none", fusion_state(q)["fusion"].get("stars"), graph_texts(ex))
    ex.shot("fus_stars_none")
    sel_fusion()
    cb = visible(panel(ex), QComboBox)[0]
    cb.setCurrentIndex(1)
    cb.activated.emit(1)
    wait(ex, 0.8)
    log("S palette", fusion_state(q)["fusion"].get("stars"))
    undo(ex)
    wait(ex, 0.6)
    undo(ex)
    wait(ex, 0.6)
    log("S after undos", fusion_state(q)["fusion"].get("stars"),
        {x.pid: (x.opts.get("etoiles"), x.opts.get("debruitage")) for x in ex.s.paths if x.source is None})
# rename
sel_fusion()
ed = next((e for e in visible(panel(ex), QLineEdit)), None)
if ed is not None:
    ed.setFocus()
    ed.selectAll()
    ed.setText("Nébuleuse SHO")              # typed (QTest cannot type "é": harness)
    ed.editingFinished.emit()
    wait(ex, 0.6)
    log("R renamed", fusion_state(q)["name"], [(b.node.key, b.title, b.summary) for b in ex.blocks() if b.node.key == "fusion"])
    N.check(fusion_state(q)["name"] == "Nébuleuse SHO", "gênant", "Renommer la Fusion dans son panneau : sans effet",
            str(fusion_state(q)), steps=steps + ["taper un nom dans le titre du panneau", "Entrée"])
    ex.shot("fus_renamed")
    undo(ex)
    wait(ex, 0.6)
    log("R undo", fusion_state(q)["name"])
    N.check(fusion_state(q)["name"] != "Nébuleuse SHO", "mineur", "Ctrl+Z ne rend pas l'ancien nom de la Fusion",
            str(fusion_state(q)), steps=steps + ["renommer", "Ctrl+Z"])
    # the name everywhere: Post's layers (the Post block's panel)
    undo(ex, redo=True)
    wait(ex, 0.5)
    post = ex.block("final")
    if post is not None:
        ex.click_block(post)
        wait(ex, 0.3)
        log("R post panel", ptexts(ex))
if not button(panel(ex), _("Delete"), QPushButton):
    pass
sel_fusion()
if not button(panel(ex), _("Delete"), QPushButton):
    N.note("mineur", "Panneau Fusion : pas de bouton « Supprimer le bloc » (le panneau Image en a un)",
           f"Panneau : {ptexts(ex)[:10]}", steps=steps)
# the fusion's menu
items = ex.right_click_block(fusion_block())
log("M fusion menu", items)

# ------------------------------------------------------------------ a channel's workflow deleted afterwards
n_pop = len(ex.popups)
ha = next(x for x in ex.s.paths if x.name == "Ha")
out = next((b for b in ex.blocks() if b.node.key == "post" and ha in b.node.paths), None)
items = ex.right_click_block(out, _("Delete workflow")) if out is not None else []
wait(ex, 1.0)
log("D Ha workflow deleted", [x.get("text") for x in items], ex.popups[n_pop:], fusion_state(q), graph_texts(ex))
ex.shot("fus_channel_deleted")
st = ["projet SII+Ha+OIII avec sa Fusion", "clic droit sur la Sortie du workflow Ha", "Supprimer le workflow"]
if ex.popups[n_pop:]:
    N.note("gênant", "Supprimer un workflow passe par une fenêtre de confirmation bloquante",
           f"Message : {ex.popups[n_pop:]}. Les autres suppressions (bloc, session) se font tout de suite avec "
           "Ctrl+Z.", steps=st)
fs = fusion_state(q)
if fs is not None and any(i["pid"] == ha.pid for i in fs["fusion"]["inputs"]) and ex.s.view(ha.pid) is None:
    sel_fusion()
    pt = ptexts(ex)
    log("D fusion panel after", pt)
    ex.shot("fus_channel_deleted_panel")
    N.note("gênant", "Le workflow d'un canal supprimé : la Fusion le garde comme canal « ? » sans avertissement",
           f"Panneau : {[t for t in pt if '?' in t or 'Ha' in t]}; graphe : {graph_texts(ex)}", steps=st)
undo(ex)
wait(ex, 1.0)
log("D undo", [x.name for x in ex.s.paths], fusion_state(q))
N.check(any(x.name == "Ha" for x in ex.s.paths), "gênant", "Ctrl+Z ne rend pas le workflow d'un canal supprimé",
        "", steps=st + ["Ctrl+Z"])
# the Ha session removed: its flow goes with it
n_pop = len(ex.popups)
it = next((b for b in ex.blocks() if b.node.key == "session" and b.node.sid == 2), None)
if it is not None:
    ex.right_click_block(it, _("Delete"))
    wait(ex, 1.0)
    log("D Ha session removed", [x.name for x in ex.s.paths], fusion_state(q), graph_texts(ex), toasts[-1:])
    N.exceptions(["projet SII+Ha+OIII avec sa Fusion SHO", "clic droit sur la Session Ha", "Supprimer le bloc"])
    ex.shot("fus_session_removed")
    fs = fusion_state(q)
    if fs is not None and any(ex.s.view(i["pid"]) is None for i in fs["fusion"]["inputs"]):
        sel_fusion()
        N.note("gênant", "Session d'un canal supprimée : la Fusion garde un canal sans workflow, sans le dire",
               f"Fusion : {fs}; panneau : {ptexts(ex)[:16]}", steps=["projet SII+Ha+OIII avec sa Fusion",
                                                                      "Supprimer la Session Ha"])
    undo(ex)
    wait(ex, 1.5)
    log("D undo session", [x.name for x in ex.s.paths], [x.name for x in ex.s.p.sessions])

# ------------------------------------------------------------------ delete the fusion: cross, menu, Del; Ctrl+Z
n0 = len(ex.s.paths)
for how in ("menu", "cross", "del", "palette"):
    it = fusion_block()
    if it is None:
        break
    if how == "menu":
        ex.right_click_block(it, _("Delete"))
    elif how == "cross":
        ex.click_block(it, "cross")
    elif how == "del":
        ex.click_block(it)
        ex.w.activateWindow()
        ex.key(Qt.Key_Delete)
    else:
        ex.drag_block_to_palette(it)
    wait(ex, 0.8)
    gone = fusion_block() is None
    log("X", how, gone, len(ex.s.paths), toasts[-1:], graph_texts(ex))
    N.check(gone, "gênant", f"Supprimer la Fusion ({how}) : sans effet", "", steps=["Fusion", how])
    if gone:
        comp = [k for k, t, s in graph_texts(ex) if k == "composition"]
        if comp:
            N.note("mineur", "Fusion supprimée : son bloc « Combiner les calques étoiles et fond » reste", str(comp),
                   steps=["Fusion SHO placée", "Supprimer la Fusion"])
    undo(ex)
    wait(ex, 1.0)
    N.check(fusion_block() is not None and len(ex.s.paths) == n0, "gênant", "Ctrl+Z ne rend pas la Fusion supprimée",
            f"{len(ex.s.paths)} / {n0}", steps=["Supprimer la Fusion", "Ctrl+Z"])
# a second fusion
ex.double_click_card("fusion")
wait(ex, 1.0)
names = [x.name for x in fusion_paths()]
log("2nd fusion", names, [(b.title, b.summary) for b in ex.blocks() if b.node.key == "fusion"])
ex.shot("fus_two_fusions")
bt = [b.title for b in ex.blocks() if b.node.key == "fusion"]
if len(bt) == 2 and len(set(bt)) == 1:
    N.note("mineur", "Deux Fusions portent le même titre « Fusion » sur le graphe",
           f"Workflows : {names}; blocs : {[(b.title, b.summary) for b in ex.blocks() if b.node.key == 'fusion']}",
           steps=["Fusion placée", "double-clic sur Fusion de nouveau"])

# ------------------------------------------------------------------ dual band: a colour workflow
root = project_of("fus_colour", (), colour=True)
open_project(root)
ex.page.toasted.connect(lambda t: toasts.append(t))
card = ex.card("fusion")
log("C card reason", repr(card.reason if card else None))
ex.double_click_card("fusion")
wait(ex, 1.0)
fs = fusion_state()
log("C placed", fs, toasts[-1:])
ex.shot("fus_colour_placed")
steps = ["projet d'une session couleur et son workflow", "double-clic sur Fusion"]
if fs is None:
    N.note("gênant", "Fusion non placée sur un workflow couleur (bi-bande)", str(toasts[-1:]), steps=steps)
else:
    sel_fusion()
    pt = ptexts(ex)
    log("C panel", pt)
    ex.shot("fus_colour_panel")
    look(ex, N, "panneau Fusion bi-bande", steps=steps)
    if fs["fusion"].get("stars") is None:
        N.note("question", "Fusion bi-bande placée depuis la palette : sans retrait des étoiles, contrairement au modèle "
                           "« Bi-bande »", f"Placée : {fs}. Le modèle de la galerie met les étoiles du workflow couleur.",
               steps=steps)
# colour + mono: what fusion is placed?
root = project_of("fus_mixed", ("Ha", "OIII"), colour=True)
open_project(root)
ex.double_click_card("fusion")
wait(ex, 1.0)
log("MIX placed", fusion_state())
sel_fusion()
log("MIX panel", ptexts(ex)[:30])
ex.shot("fus_mixed_panel")

missing_t = missing_fr() - before_missing
log("MISSING fusion", sorted(missing_t))
if missing_t:
    N.note("mineur", "Fusion : textes restés en anglais", " | ".join(sorted(missing_t))[:1500], shot=False,
           steps=["Fusion placée, son panneau"])
bad = [t for t in toasts if "Fusion retiré," in t or "Image retiré," in t]
if bad:
    N.note("mineur", "Message après suppression : « Fusion retiré » / « Image retiré » (accord féminin manquant)",
           f"Messages : {sorted(set(bad))}", shot=False, steps=["supprimer une Fusion (ou une Image)"])
N.exceptions(["voir fusion_log.txt"])
(ex.out / "fusion_log.txt").write_text("\n".join(LOG), encoding="utf-8")
ex.close()
print("DONE")
