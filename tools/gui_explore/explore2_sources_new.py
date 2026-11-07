"""sources_files, round 2, new checks: the project reopens on the block selected (Save, close, open), a Session or an
Image placed (on a flow, on the new-flow zone) is one Ctrl+Z and one Ctrl+Y, the Image's file and kind undone, the
calibration frames picked as files, a lone Session's menu."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore2_sources_common import (  # noqa: E402
    TMP, Explorer, Notes, _, button, graph_texts, look, panel, ptexts, start, undo, visible, wait,
)
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QWidget  # noqa: E402

start(900)
import os  # noqa: E402

ex = Explorer("sources_files", lang=sys.argv[1] if len(sys.argv) > 1 else "fr",
              size=tuple(int(x) for x in os.environ.get("EXPLORE_SIZE", "1600x950").split("x")))
N = Notes(ex)
LOG: list[str] = []
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None


def log(*a):
    t = " ".join(str(x) for x in a)
    LOG.append(t)
    print(t, flush=True)


def want(sec):
    return ONLY is None or sec in ONLY


def where():
    """What the window shows as selected: the flow, the block, the panel's title."""
    s = ex.s
    p = s.path
    title = next((w.text() for w in visible(panel(ex), QLabel) if w.text()), "")
    ed = next((w.text() for w in visible(panel(ex), QLineEdit)), "")
    return {"flow": p.name if p else None, "block": s.sel_step, "panel": ed or title}


def save():
    ex.key(Qt.Key_S, ex.w, Qt.ControlModifier)
    wait(ex, 0.5)


def reopen(root):
    ex.w.close_project()
    wait(ex, 0.5)
    ex.w.open_project(root)
    wait(ex, 1.5)


def blk(key, pid=None, sid=None):
    return ex.block(key, pid=pid, sid=sid)


def stacks():
    return len(ex.s._undo), len(ex.s._redo)


def shape():
    s = ex.s
    return {"sessions": [(x.id, x.name) for x in s.p.sessions],
            "flows": [(q.name, q.source.get("path", "")[-12:] if q.source else None,
                       tuple(s.p.workflow(q.pid).sessions or [])) for q in s.paths],
            "blocks": sorted(f"{b.node.key}@{b.node.sid}" if b.node.sid else b.node.key for b in ex.blocks())}


# ================================================================== A. reopen on the block selected
if want("A"):
    root = ex.project("mono", "reopenA")
    ex.open(root)
    wait(ex, 1.0)
    log("A open", where(), [(b.node.key, [q.pid for q in b.node.paths], b.node.sid) for b in ex.blocks()])
    pid2 = ex.s.paths[1].pid if len(ex.s.paths) > 1 else None
    it = blk("alignement", pid=pid2) or blk("registration", pid=pid2)
    if it is None:
        it = next((b for b in ex.blocks() if b.node.key not in ("session", "sources", "final") and pid2 in
                   [q.pid for q in b.node.paths]), None)
    ex.click_block(it)
    wait(ex, 0.4)
    chosen = where()
    log("A chosen", chosen, "dirty", ex.s.dirty)
    save()
    reopen(root)
    got = where()
    log("A reopened (selection only, Ctrl+S)", got)
    ex.shot("A_reopen_selection_only")
    if (got["flow"], got["block"]) != (chosen["flow"], chosen["block"]):
        N.note("gênant", "Projet rouvert : il ne revient pas sur le bloc choisi quand rien d'autre n'a changé",
               f"Bloc choisi : {chosen['block']} du workflow « {chosen['flow']} », puis Ctrl+S, fermer, rouvrir : "
               f"le projet s'ouvre sur {got['block']} du workflow « {got['flow']} ». Ctrl+S n'écrit rien quand il n'y "
               "a pas de modification : le choix du bloc n'est pas gardé.",
               steps=["projet à deux workflows (Ha, OIII)", "clic sur l'Alignement du workflow OIII", "Ctrl+S",
                      "fermer le projet", "le rouvrir"])
    # with an edit: the filter of session 1 changed, then the block chosen, Save
    ex.click_block(blk("session", sid=1) or blk("session"))
    wait(ex, 0.3)
    ex.s.rename_session(1, "Ha nuit")
    wait(ex, 0.3)
    it = blk("alignement", pid=pid2)
    ex.click_block(it)
    wait(ex, 0.3)
    chosen = where()
    save()
    reopen(root)
    got = where()
    log("A reopened (after an edit)", chosen, got)
    ex.shot("A_reopen_after_edit")
    if (got["flow"], got["block"]) != (chosen["flow"], chosen["block"]):
        N.note("gênant", "Projet rouvert après une modification enregistrée : pas sur le bloc choisi",
               f"Choisi : {chosen}; rouvert : {got}",
               steps=["projet Ha + OIII", "renommer la session 1", "clic sur l'Alignement du workflow OIII", "Ctrl+S",
                      "fermer, rouvrir"])
    # the second session's block chosen
    s2 = blk("session", sid=2)
    if s2 is not None:
        ex.click_block(s2)
        wait(ex, 0.3)
        ex.s.rename_session(2, "OIII nuit")
        chosen = where()
        save()
        reopen(root)
        got = where()
        log("A session 2", chosen, got)
        if (got["flow"], got["block"]) != (chosen["flow"], chosen["block"]):
            N.note("gênant", "Projet rouvert : la deuxième Session choisie n'est pas celle montrée",
                   f"Choisi : {chosen}; rouvert : {got}", steps=["projet Ha + OIII", "clic sur la Session OIII",
                                                                 "renommer", "Ctrl+S", "fermer, rouvrir"])
    # Post chosen
    post = blk("final")
    if post is not None:
        ex.click_block(post)
        wait(ex, 0.3)
        ex.s.rename_session(2, "OIII")
        chosen = where()
        save()
        reopen(root)
        got = where()
        log("A post", chosen, got)
        if got["block"] != chosen["block"]:
            N.note("mineur", "Projet rouvert : Post était choisi, il s'ouvre ailleurs", f"Choisi : {chosen}; rouvert : "
                   f"{got}", steps=["clic sur Post", "une modification", "Ctrl+S", "fermer, rouvrir"])
    ex.w.close_project()
    wait(ex, 0.3)

    # an image flow: the Image chosen
    root = ex.project("session+image", "reopenB")
    ex.open(root)
    wait(ex, 1.0)
    im = blk("image")
    log("B open", where(), shape())
    if im is not None:
        ex.click_block(im)
        wait(ex, 0.3)
        b = next((x for x in visible(panel(ex), QPushButton) if x.text() == _("Stretched")), None)
        if b is not None and b.isEnabled():
            ex.click(b)
            wait(ex, 0.4)
        ex.click_block(blk("image"))
        wait(ex, 0.3)
        chosen = where()
        save()
        reopen(root)
        got = where()
        log("B image", chosen, got)
        ex.shot("B_reopen_image")
        if (got["flow"], got["block"]) != (chosen["flow"], chosen["block"]):
            N.note("gênant", "Projet rouvert : l'Image choisie n'est pas reprise, il s'ouvre sur la Session",
                   f"Choisi : bloc {chosen['block']} du workflow « {chosen['flow']} » ; rouvert : bloc {got['block']} "
                   f"du workflow « {got['flow']} » (panneau « {got['panel']} »).",
                   steps=["projet d'une Session et d'une Image", "clic sur l'Image", "Nature : Étirée", "Ctrl+S",
                          "fermer, rouvrir"])
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== C. a Session / an Image placed: one Ctrl+Z, one Ctrl+Y
if want("C"):
    root = ex.project("session", "placeC")
    ex.open(root)
    wait(ex, 1.0)
    log("C start", shape(), stacks())
    for card in ("session", "image"):
        for target in ("new_flow", "ghosts"):
            before = shape()
            u0 = stacks()
            if target == "new_flow":
                z = ex.new_flow_zone()
                if z is None:
                    log("C no new-flow zone")
                    continue
                rec = ex.drag_palette(card, ex.ghost_center(z))
            else:
                ex.drag_palette_hover(card, ex.ghost_center(ex.new_flow_zone()) if ex.new_flow_zone() else None)
                gh = ex.ghosts()
                log("C ghosts for", card, [(getattr(g, "text", ""), ) for g in gh])
                ex.drag_leave()
                if not gh:
                    continue
                rec = ex.drag_palette(card, ex.ghost_center(gh[0]))
            wait(ex, 0.8)
            after = shape()
            u1 = stacks()
            log("C drop", card, target, rec, after, u0, u1)
            st = [f"projet d'une Session", f"glisser {card} de la palette sur "
                  + ("la zone Nouveau flux" if target == "new_flow" else "la première place proposée")]
            ex.shot(f"C_{card}_{target}_placed")
            if after == before:
                continue
            if u1[0] - u0[0] != 1:
                N.note("gênant", f"{card.capitalize()} posée : {u1[0] - u0[0]} pas d'annulation au lieu d'un",
                       f"Avant {before}; après {after}", steps=st)
            undo(ex)
            back = shape()
            log("C undo", back)
            if back != before:
                N.note("gênant", f"Ctrl+Z après avoir posé {card} : le projet ne revient pas à l'état d'avant",
                       f"Avant : {before}\nAprès Ctrl+Z : {back}", steps=st + ["Ctrl+Z"])
            undo(ex, redo=True)
            again = shape()
            log("C redo", again)
            if again != after:
                N.note("gênant", f"Ctrl+Y après avoir annulé la pose de {card} : pas le même graphe",
                       f"Posé : {after}\nAprès Ctrl+Y : {again}", steps=st + ["Ctrl+Z", "Ctrl+Y"])
            undo(ex)
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== D. the Image's file and kind, undone
if want("D"):
    png = ex.image_file("photo.png")
    fits_img = ex.image_file("pile.fits")
    txt = TMP / "data" / "images" / "readme.txt"
    txt.write_text("x", encoding="utf-8")
    ex.new_project("ImageD", None)
    ex.close_dialogs()
    wait(ex, 0.5)
    toasts = []
    ex.page.toasted.connect(lambda t: toasts.append(t))
    ex.double_click_card("image")
    wait(ex, 0.8)

    def src():
        q = next((q for q in ex.s.paths if q.source is not None), None)
        return dict(q.source) if q else None

    def choose(f):
        ex.click_block(blk("image"))
        wait(ex, 0.3)
        b = button(panel(ex), _("Choose image…"), QPushButton) or button(panel(ex), _("Change image…"), QPushButton)
        ex.file_queue = [str(f)]
        ex.click(b)
        wait(ex, 0.6)

    s0 = src()
    choose(png)
    s1 = src()
    log("D png", s0, s1, toasts[-1:])
    ex.click_block(blk("image"))
    ex.shot("D_png_chosen")
    if not s1 or not s1.get("stretched"):
        N.note("gênant", "Image PNG choisie : toujours « Linéaire »", f"Source : {s1}",
               steps=["Projet Vierge", "Image posée", "Choisir image… > photo.png"])
    u = undo(ex)
    log("D undo png", u, src())
    if src() != s0:
        N.note("gênant", "Ctrl+Z après avoir choisi le fichier de l'Image : le fichier reste",
               f"Avant : {s0}; après Ctrl+Z : {src()} ({u})", steps=["Image posée", "Choisir image… > photo.png",
                                                                      "Ctrl+Z"])
    undo(ex, redo=True)
    log("D redo png", src())
    if src() != s1:
        N.note("gênant", "Ctrl+Y après avoir annulé le choix du fichier de l'Image : pas le même fichier",
               f"Attendu {s1}; obtenu {src()}", steps=["Choisir image… > photo.png", "Ctrl+Z", "Ctrl+Y"])
    choose(fits_img)
    s2 = src()
    log("D fits after png", s2)
    if s2 and s2.get("stretched"):
        N.note("question", "Une pile FITS choisie après un PNG reste « Étirée »",
               f"Source : {s2}. Le PNG avait mis l'Image en Étirée ; la pile FITS qui le remplace garde cette nature.",
               steps=["Image posée", "Choisir image… > photo.png", "Changer image… > pile.fits"])
    n_toasts = len(toasts)
    choose(txt)
    log("D txt", src(), toasts[n_toasts:])
    if src() != s2:
        N.note("gênant", "Image : un fichier texte remplace l'image", f"{src()}", steps=["Changer image… > readme.txt"])
    if len(toasts) == n_toasts:
        N.note("mineur", "Image : un fichier texte refusé sans un mot", "", steps=["Changer image… > readme.txt"])
    # the kind toggled, undone
    ex.click_block(blk("image"))
    wait(ex, 0.3)
    k0 = src()
    b = next((x for x in visible(panel(ex), QPushButton) if x.text() == (_("Linear") if k0.get("stretched") else
                                                                         _("Stretched"))), None)
    if b is not None and b.isEnabled():
        ex.click(b)
        wait(ex, 0.4)
        k1 = src()
        undo(ex)
        log("D kind", k0, k1, src())
        if src() != k0:
            N.note("gênant", "Ctrl+Z après avoir changé la nature de l'Image : elle reste changée",
                   f"Avant {k0}, après le clic {k1}, après Ctrl+Z {src()}",
                   steps=["Image posée avec un fichier", "Nature : l'autre bouton", "Ctrl+Z"])
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== E. calibration frames picked as files
if want("E"):
    lights = ex.raws("calE/lights", 6)
    darks = ex.raws("calE/darks", 5, start=8000)
    darks2 = ex.raws("calE/darks_b", 3, start=8800)
    flats = ex.raws("calE/flats", 4, start=8100)
    ex.new_project("CalE", None)
    ex.close_dialogs()
    wait(ex, 0.5)
    toasts = []
    ex.page.toasted.connect(lambda t: toasts.append(t))
    ex.double_click_card("session")
    wait(ex, 0.8)
    from explore2_sources_common import buttons, menu_button  # noqa: E402

    def sel():
        ex.click_block(blk("session") or blk("sources"))
        wait(ex, 0.3)

    def add(kind_i, what, ans):
        sel()
        bs = buttons(panel(ex), _("Add") + " ▾", QPushButton)
        ex.file_queue = [ans]
        menu_button(ex, bs[kind_i], what)
        wait(ex, 0.6)

    def titles():
        sel()
        return [t for t in ptexts(ex) if "  ·  " in t and not t.startswith("▸") and not t.startswith("▾")]

    add(0, _("Add folder…"), str(lights))
    u0 = stacks()
    dk = sorted(str(x) for x in darks.iterdir())
    add(1, _("Add files…"), dk[:2])
    add(1, _("Add files…"), [str(next(darks2.iterdir()))])
    add(2, _("Add files…"), sorted(str(x) for x in flats.iterdir())[:3])
    log("E titles", titles(), "block", [(b.node.key, b.summary) for b in ex.blocks()], "undo", u0, stacks(), toasts)
    ex.shot("E_files_picked")
    look(ex, N, "panneau Session, darks et flats choisis en fichiers", steps=["Session posée", "Darks > Ajouter "
                                                                               "fichiers… (2 + 1 d'un autre dossier)"])
    t = titles()
    if not any(x.startswith(_("Darks") + "  ·  3") for x in t):
        N.note("gênant", "Darks choisis en fichiers : le compte n'est pas 3", f"{t}",
               steps=["Darks > Ajouter fichiers… > 2 darks", "Darks > Ajouter fichiers… > 1 dark d'un autre dossier"])
    rows = [b.text() for b in visible(panel(ex), QPushButton) if "📁" in b.text()]
    log("E rows", rows)
    # the same dark file again, then its folder whole
    n = len(toasts)
    add(1, _("Add files…"), dk[:1])
    log("E same file again", toasts[n:], titles())
    add(1, _("Add folder…"), str(darks))
    sel()
    rows = [b.text() for b in visible(panel(ex), QPushButton) if "📁" in b.text()]
    log("E folder after files", rows, titles(), toasts[n:])
    ex.shot("E_folder_after_files")
    dk_row = next((r for r in rows if "📁 darks " in r or r.endswith("📁 darks") or "darks  ·" in r), "")
    if dk_row and "5" not in dk_row:
        N.note("gênant", "Dossier de darks ajouté après deux de ses fichiers : la ligne ne compte pas ses 5 fichiers",
               f"Lignes : {rows}", steps=["Darks > Ajouter fichiers… > 2 darks du dossier", "Darks > Ajouter dossier… "
                                                                                          "> ce dossier (5 darks)"])
    t = titles()
    dt = next((x for x in t if x.startswith(_("Darks"))), "")
    if dt and not dt.endswith("6"):
        N.note("gênant", "Darks : compte faux après l'ajout du dossier de fichiers déjà choisis",
               f"Titre : {dt!r} (attendu 6 : 5 du dossier et 1 d'un autre)", steps=["Darks > 2 fichiers, 1 fichier "
                                                                                     "d'un autre dossier, puis le "
                                                                                     "dossier des 2 premiers"])
    # take out the lone dark of the other folder by its folder cross
    sel()
    xs = [b for b in visible(panel(ex), QPushButton) if b.text() == "×"]
    rows_b = [b for b in visible(panel(ex), QPushButton) if "📁" in b.text() and "darks_b" in b.text()]
    if rows_b:
        y = rows_b[0].mapTo(panel(ex), rows_b[0].rect().center()).y()
        x = next((b for b in xs if abs(b.mapTo(panel(ex), b.rect().center()).y() - y) < 6), None)
        if x is not None:
            ex.click(x)
            wait(ex, 0.6)
            log("E darks_b out", titles())
    # flats: the 4th flat added alone, then open the group, take one out by its cross
    add(2, _("Add files…"), [sorted(str(x) for x in flats.iterdir())[3]])
    sel()
    rows = [b for b in visible(panel(ex), QPushButton) if "📁" in b.text() and "flats" in b.text()]
    log("E flats rows", [r.text() for r in rows])
    if rows:
        ex.click(rows[0])
        wait(ex, 0.5)
        lists = [x for x in visible(panel(ex), __import__("PySide6.QtWidgets", fromlist=["QListWidget"]).QListWidget)]
        log("E flats list", [x.count() for x in lists])
        ex.shot("E_flats_open")
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== F. a lone Session: menu, panel; two sessions
if want("F"):
    root = ex.project("two_sessions", "sessF")
    ex.open(root)
    wait(ex, 1.0)
    for sid in (1, 2):
        it = blk("session", sid=sid)
        if it is None:
            log("F no session block", sid, shape())
            continue
        items = ex.right_click_block(it)
        ex.click_block(blk("session", sid=sid))
        wait(ex, 0.3)
        log("F menu", sid, [(x.get("text"), x.get("enabled")) for x in items])
        log("F panel", sid, ptexts(ex)[:6])
    ex.shot("F_two_sessions_panel")
    has_del = button(panel(ex), _("Delete"), QPushButton)
    log("F panel delete button (two sessions)", bool(has_del))
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== G. the Fusion's cross; what stays after it is removed
if want("G"):
    from astro_gpu.core import project2, workflow as wfl  # noqa: E401,E402
    from astro_gpu.core.project import Target  # noqa: E402
    name = "fusG"
    root = ex.ws / name
    p = project2.create(root, name, Target(name="M 31"))
    p.stack_mode = "per_session"
    for i, f in enumerate(("SII", "Ha", "OIII")):
        p.sessions.append(project2.Session(i + 1, f, lights=[str(ex.fits_frames(f"{name}/{f}", 4, f))], filter=f))
    project2.save(p)
    for i, f in enumerate(("SII", "Ha", "OIII")):
        w = p.add_workflow(f, wfl.TEMPLATES["Classic stack"])
        w.sessions = [i + 1]
        p.set_blocks(w.id, [wfl.node(b) for b in ("calibration", "registration", "stacking", "astrometry", "gradient")])
    project2.save(p)
    ex.open(root)
    wait(ex, 1.0)
    before = shape()
    log("G before", before)
    ex.double_click_card("fusion")
    wait(ex, 1.0)
    placed = shape()
    log("G placed", placed)
    ex.shot("G_fusion_placed")
    fu = blk("fusion")
    for attempt in (1, 2):
        fu = blk("fusion")
        ex.click_block(fu)
        wait(ex, 0.3)
        fu = blk("fusion")
        r = fu.sceneBoundingRect()
        ex.hover(ex.view.viewport(), ex.point_of(fu, "cross"))
        ex.shot(f"G_fusion_hover_cross_{attempt}")
        under = ex.view.itemAt(ex.point_of(fu, "cross"))
        log("G cross point", attempt, r, ex.point_of(fu, "cross"), type(under).__name__,
            getattr(under, "title", None), getattr(under, "_cross", None))
        # the cross where it is drawn (the harness's "cross" point counts the port margin of the bounding rect)
        xp = ex.view.mapFromScene(fu.mapToScene(fu._x_rect().center()))
        ex.hover(ex.view.viewport(), xp)
        ex.click(ex.view.viewport(), xp)
        wait(ex, 0.8)
        log("G after cross", attempt, blk("fusion") is not None, shape())
        if blk("fusion") is None:
            undo(ex)
            wait(ex, 0.6)
    # by the menu: what stays
    fu = blk("fusion")
    ex.right_click_block(fu, _("Delete"))
    wait(ex, 0.8)
    after = shape()
    log("G after menu delete", after)
    ex.shot("G_fusion_deleted")
    nb = {k: after["blocks"].count(k) for k in set(after["blocks"])}
    n0 = {k: before["blocks"].count(k) for k in set(before["blocks"])}
    extra = {k: v - n0.get(k, 0) for k, v in nb.items() if v != n0.get(k, 0)}
    if extra.get("composition", 0) > 1:
        N.note("gênant", "Supprimer la Fusion ajoute deux blocs Recombiner (un par canal) au lieu d'en retirer",
               f"Avant la Fusion : pas de Débruitage, d'Étoiles/fond ni de Recombiner. Fusion posée : elle ajoute un "
               f"Débruitage et un Étoiles/fond à chaque canal, et un Recombiner. Fusion supprimée (« Fusion et suivants "
               f"retirés ») : il reste en plus {extra} — trois Recombiner au lieu d'un, et les Débruitage et "
               "Étoiles/fond qu'elle avait ajoutés.",
               steps=["projet SII, Ha, OIII, un workflow par filtre (Calibration … Gradient)", "double-clic sur Fusion",
                      "clic droit sur la Fusion > Supprimer"])
    undo(ex)
    log("G undo", shape() == placed)
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== H. "+ Session": does the new night join the flow?
if want("H"):
    lights = ex.raws("plusH/l1", 5)
    l2 = ex.raws("plusH/l2", 4, start=7000)
    ex.new_project("PlusH", None)
    ex.close_dialogs()
    wait(ex, 0.5)
    ex.double_click_card("session")
    wait(ex, 0.8)
    from explore2_sources_common import buttons, menu_button  # noqa: E402
    ex.click_block(blk("session") or blk("sources"))
    bs = buttons(panel(ex), _("Add") + " ▾", QPushButton)
    ex.file_queue = [str(lights)]
    menu_button(ex, bs[0], _("Add folder…"))
    wait(ex, 0.6)
    log("H one", shape())
    from astro_gpu.client.graph import GhostItem  # noqa: E402
    gh = [g for g in ex.view.scene().items() if isinstance(g, GhostItem) and "Session" in (g.text or "")
          and not getattr(g, "new_flow", False)]
    log("H ghosts", [g.text for g in gh])
    u0 = stacks()
    if gh:
        ex.file_queue = [str(l2)]
        from PySide6.QtCore import QPointF
        r = gh[0].sceneBoundingRect()
        ex.click(ex.view.viewport(), ex.view.mapFromScene(QPointF(r.center().x(), r.center().y())))
        wait(ex, 1.0)
    two = shape()
    log("H two", two, u0, stacks(), [(q.name, ex.s.p.workflow(q.pid).sessions) for q in ex.s.paths])
    ex.shot("H_plus_session")
    ex.double_click_card("calibration")
    wait(ex, 0.8)
    log("H calibration", shape())
    ex.shot("H_calibration_after_plus_session")
    ws = [ex.s.p.workflow(q.pid).sessions for q in ex.s.paths]
    if len(ex.s.p.sessions) == 2 and not any(2 in (x or []) for x in ws):
        N.note("gênant", "« + Session » : la deuxième nuit ne rejoint pas le workflow, la Calibration n'est posée que "
                         "pour la première",
               f"La session ajoutée par « + Session » (sous la Session 1) n'est reliée à rien : sessions des workflows "
               f"{ws}. Calibration posée ensuite (double-clic) : blocs {shape()['blocks']}. Rien ne dit que cette "
               "nuit ne sera pas empilée ; la barre d'état dit seulement « Session ajoutée · darks/flats dans son "
               "bloc ».",
               steps=["projet Vierge", "Session posée (double-clic), Brutes > Ajouter > un dossier",
                      "clic sur « + Session » sous elle, un deuxième dossier", "double-clic sur Calibration"])
    undo(ex)
    undo(ex)
    log("H two undos", shape())
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== I. a channel's Session removed: rows and wires
if want("I"):
    from astro_gpu.core import project2, workflow as wfl  # noqa: E401,E402
    from astro_gpu.core.project import Target  # noqa: E402
    for rep in (1, 2):
        name = f"fusI{rep}"
        root = ex.ws / name
        p = project2.create(root, name, Target(name="M 31"))
        p.stack_mode = "per_session"
        for i, f in enumerate(("SII", "Ha", "OIII")):
            p.sessions.append(project2.Session(i + 1, f, lights=[str(ex.fits_frames(f"{name}/{f}", 4, f))], filter=f))
        project2.save(p)
        for i, f in enumerate(("SII", "Ha", "OIII")):
            w = p.add_workflow(f, wfl.TEMPLATES["Classic stack"])
            w.sessions = [i + 1]
            p.set_blocks(w.id, [wfl.node(b) for b in ("calibration", "registration", "stacking", "astrometry")])
        project2.save(p)
        ex.open(root)
        wait(ex, 1.0)
        if rep == 2:
            ex.double_click_card("fusion")
            wait(ex, 1.0)

        def rows():
            out = {}
            for b in ex.blocks():
                if b.node.key in ("session", "calibration", "alignement"):
                    pid = b.node.paths[0].pid if b.node.paths else None
                    k = "alignement" if b.node.key == "alignement" else b.node.key
                    out[f"{k}@{pid}"] = round(b.sceneBoundingRect().center().y())
            return out
        r0 = rows()
        it = blk("session", sid=2)
        ex.right_click_block(it, _("Delete"))
        wait(ex, 1.0)
        r1 = rows()
        log("I rows", rep, r0, r1, [(x.id, x.name, x.lights) for x in ex.s.p.sessions])
        ex.shot(f"I_session_removed_{rep}")
        bad = r1.get("session@2") is not None and r1.get("alignement@2") is not None and \
            abs(r1["session@2"] - r1["alignement@2"]) > 5
        if bad:
            N.note("gênant", "Session d'un canal retirée : sa ligne descend en bas, les fils se croisent",
                   f"Hauteurs des blocs (y) avant : {r0} ; après : {r1}. La Session Ha vidée est redessinée sous OIII, "
                   "mais sa chaîne (Alignement, Empilement…) reste au milieu : les fils Calibration → Alignement des "
                   "lignes Ha et OIII se croisent.",
                   steps=["projet SII, Ha, OIII, un workflow par filtre" + (" et leur Fusion" if rep == 2 else ""),
                          "clic droit sur la Session Ha > Supprimer"])
        undo(ex)
        log("I undo", rep, rows() == r0)
        ex.w.close_project()
        wait(ex, 0.3)

# ================================================================== J. gallery: one workflow per filter (filters chosen); stretched image
if want("J"):
    def gal(tab_key, key, chips=(), image="", cap=None):
        def h(d):
            if d.__class__.__name__ != "TemplateGallery":
                d.reject()
                return
            d.tabs.setCurrentIndex(d.tab_keys.index(tab_key))
            ex.settle(2)
            card = next(c for c in d.cards if c.t.get("key") == key)
            ex.click(card)
            ok0 = d.ok.isEnabled()
            for c in chips:
                b = next((x for x in d.findChildren(QPushButton) if x.isVisible() and x.text() == c), None)
                if b is not None:
                    ex.click(b)
            if image:
                d.file.setText(image)
            ex.settle(3)
            cap.update(ok0=ok0, ok=d.ok.isEnabled(), ok_tip=d.ok.toolTip(),
                       hint=[x.text() for x in d.findChildren(QLabel) if x.isVisible() and "ltre" in x.text()][:3])
            ex.shot(f"J_gallery_{key}_{len(chips)}", d)
            if d.ok.isEnabled():
                d.ok.click()
            else:
                d.reject()
        return h
    for chips in ((), ("Ha", "OIII")):
        cap = {}
        ex.open()
        ex.dialog_handler = gal("mono", "mono_each", chips, cap=cap)
        ex.w.create_project(str(ex.ws / f"JMono{len(chips)}"), f"JMono{len(chips)}", "M 31")
        wait(ex, 1.0)
        log("J mono_each", chips, cap, shape() if ex.s else None)
        ex.shot(f"J_created_mono_each_{len(chips)}")
        if chips and ex.s is not None and shape()["blocks"].count("alignement") < 2:
            N.note("gênant", "Modèle « Un workflow par filtre » : juste créé, Ha et OIII dessinés en une seule chaîne "
                             "avec deux blocs « Sortie »",
                   f"Après Créer (filtres Ha et OIII), les deux Sessions et leurs Calibrations entrent dans un seul "
                   f"Alignement, un seul Empilement… jusqu'au Débruitage, puis deux blocs « Sortie » (« Ha · vers "
                   f"Post », « OIII · vers Post ») vont à Post : on lit que les deux filtres seront empilés ensemble. "
                   f"L'aperçu de la galerie montrait deux lignes. Dès que chaque Session a ses brutes, le graphe se "
                   f"sépare en deux chaînes et les « Sortie » disparaissent. Blocs : {shape()['blocks']}",
                   steps=["Nouveau projet", "galerie : onglet Filtres, « Un workflow par filtre »", "Filtres : Ha, OIII",
                          "Créer"])
        if ex.w.project is not None:
            ex.w.close_project()
            wait(ex, 0.3)
    # the same, then each session's lights added: do the filters get their own chain?
    for rep in (1, 2):
        cap = {}
        ex.dialog_handler = gal("mono", "mono_each", ("Ha", "OIII"), cap=cap)
        ex.w.create_project(str(ex.ws / f"JMonoL{rep}"), f"JMonoL{rep}", "M 31")
        wait(ex, 1.0)
        from explore2_sources_common import buttons, menu_button  # noqa: E402
        for sid, f in ((1, "Ha"), (2, "OIII")):
            d = ex.fits_frames(f"JMonoL{rep}/{f}", 4, f)
            ex.click_block(blk("session", sid=sid))
            wait(ex, 0.3)
            bs = buttons(panel(ex), _("Add") + " ▾", QPushButton)
            ex.file_queue = [str(d)]
            menu_button(ex, bs[0], _("Add folder…"))
            wait(ex, 0.8)
        sh = shape()
        log("J mono_each with lights", rep, sh, [(b.node.key, b.title, b.summary) for b in ex.blocks()
                                                 if b.node.key == "post"])
        ex.shot(f"J_mono_each_with_lights_{rep}")
        n_al = sh["blocks"].count("alignement")
        if n_al < 2:
            N.note("gênant", "Modèle « Un workflow par filtre » : Ha et OIII dessinés dans une seule chaîne",
                   f"Filtres Ha et OIII choisis dans la galerie, puis les brutes de chaque filtre ajoutées dans sa "
                   f"Session : le graphe montre les deux Sessions et leurs Calibrations entrant dans un seul "
                   f"Alignement, un seul Empilement… jusqu'au Débruitage, puis deux blocs « Sortie » ({n_al} "
                   f"Alignement). L'aperçu de la galerie montrait deux lignes, une par filtre. Blocs : {sh['blocks']}",
                   steps=["Nouveau projet", "galerie : onglet Filtres, « Un workflow par filtre »", "Filtres : Ha, OIII",
                          "Créer", "Session Ha > Brutes > Ajouter dossier (FITS Ha)",
                          "Session OIII > Brutes > Ajouter dossier (FITS OIII)"])
        ex.w.close_project()
        wait(ex, 0.3)
    png = ex.image_file("fini.png")
    cap = {}
    ex.dialog_handler = gal("image", "image_stretched", (), str(png), cap=cap)
    ex.w.create_project(str(ex.ws / "JStretched"), "JStretched", "M 31")
    wait(ex, 1.0)
    log("J stretched", cap, shape(), ex.w.status.text(), ex.w.bt_run.text(), ex.w.bt_run.isEnabled(),
        ex.w.bt_run.toolTip(), ex.s.sel_step)
    ex.shot("J_created_stretched")
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== K. a Session dragged onto a flow: joins it?
if want("K"):
    from astro_gpu.client.graph import GhostItem  # noqa: E402
    root = ex.project("two_flows", "flowsK")
    ex.open(root)
    wait(ex, 1.0)
    vp = ex.view.viewport()
    z = ex.new_flow_zone()
    ex.drag_palette_hover("session", ex.ghost_center(z))
    ghs = [g for g in ex.view.scene().items() if isinstance(g, GhostItem)]
    log("K ghosts while dragging", [(getattr(g, "text", ""), getattr(g, "new_flow", False),
                                     round(g.sceneBoundingRect().center().y())) for g in ghs], ex.ghosts())
    ex.shot("K_drag_session_places")
    ex.drag_leave()
    cand = [g for g in (ex.ghosts() or ghs) if not getattr(g, "new_flow", False)]
    log("K start", shape(), [(q.name, ex.s.p.workflow(q.pid).sessions) for q in ex.s.paths])
    for target in ("+ Session", "Flux"):
        before = shape()
        wf0 = [(q.name, tuple(ex.s.p.workflow(q.pid).sessions or [])) for q in ex.s.paths]
        g = next((x for x in ex.view.scene().items() if isinstance(x, GhostItem) and target in (x.text or "").replace("  ", " ")), None)
        if g is None:
            log("K no ghost", target)
            continue
        rec = ex.drag_palette("session", ex.ghost_center(g))
        wait(ex, 0.8)
        wf1 = [(q.name, tuple(ex.s.p.workflow(q.pid).sessions or [])) for q in ex.s.paths]
        log("K dropped on", target, rec, wf0, "->", wf1, shape()["sessions"])
        ex.shot(f"K_session_dropped_{target[-4:].strip()}")
        if target == "+ Session" and dict(wf1).get("Workflow") != dict(wf0).get("Workflow") and \
                dict(wf1).get("Flow") == dict(wf0).get("Flow"):
            N.note("question", "Deux flux vides : une Session déposée sur « + Session » (sous la Session 2) rejoint le "
                               "flux de la Session 1, sans que rien ne le montre",
                   f"Flux avant : {wf0} ; après : {wf1}. « + Session » est dessiné sous la Session 2 ; la nouvelle "
                   "Session est dessinée sous elle, sans fil ; elle rejoint le workflow choisi dans la palette "
                   "(« pour « Workflow » »), celui de la Session 1. Rien à l'écran ne dit à quel flux elle appartient.",
                   steps=["projet à deux flux vides (Session 1 → Workflow, Session 2 → Flow)",
                          "glisser Session de la palette sur « + Session »"])
        u = undo(ex)
        log("K undo", u, shape() == before)
        undo(ex, redo=True)
        log("K redo", [(q.name, tuple(ex.s.p.workflow(q.pid).sessions or [])) for q in ex.s.paths] == wf1)
        undo(ex)
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== L. language changed with a Session / Image panel open
if want("L"):
    root = ex.project("session+image", "langL")
    ex.open(root)
    wait(ex, 1.0)
    for key in ("sources", "image"):
        ex.click_block(blk(key))
        wait(ex, 0.3)
        t_fr = ptexts(ex)[:14]
        ex.w.change_language("en" if ex.lang == "fr" else "fr")
        wait(ex, 1.0)
        t_other = ptexts(ex)[:14]
        g_other = [(b.title, b.summary) for b in ex.blocks()]
        log("L", key, t_fr, "->", t_other, g_other, "selected", ex.s.sel_step)
        ex.shot(f"L_language_{key}")
        ex.w.change_language(ex.lang)
        wait(ex, 1.0)
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== N. darks shared from a session, that session removed
if want("N"):
    root = ex.project("two_sessions", "sharedN")
    ex.open(root)
    wait(ex, 1.0)
    from explore2_sources_common import buttons, menu_button  # noqa: E402
    dk = ex.raws("sharedN/darks", 4, start=8000)
    ex.click_block(blk("session", sid=1))
    wait(ex, 0.3)
    bs = buttons(panel(ex), _("Add") + " ▾", QPushButton)
    ex.file_queue = [str(dk)]
    menu_button(ex, bs[1], _("Add folder…"))
    wait(ex, 0.6)
    ex.click_block(blk("session", sid=2))
    wait(ex, 0.3)
    bs = buttons(panel(ex), _("Add") + " ▾", QPushButton)
    items = menu_button(ex, bs[1], "Session 1")
    wait(ex, 0.6)
    s2 = ex.s.p.session(2)
    log("N shared", [x.get("text") for x in items], s2.darks, ptexts(ex)[12:20])
    ex.click_block(blk("session", sid=1))
    ex.right_click_block(blk("session", sid=1), _("Delete"))
    wait(ex, 0.8)
    s2 = ex.s.p.session(2)
    log("N session 1 removed", [(x.id, x.name, x.darks) for x in ex.s.p.sessions], shape())
    ex.click_block(blk("session", sid=2) or blk("sources"))
    wait(ex, 0.3)
    log("N panel s2", [t for t in ptexts(ex) if "Darks" in t or "⇄" in t or "partag" in t or "shared" in t])
    ex.shot("N_shared_darks_after_removal")
    undo(ex)
    log("N undo", [(x.id, x.name, x.darks) for x in ex.s.p.sessions])
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== P. frames added, Save, reopen: still there
if want("P"):
    root = ex.project("session", "saveP")
    ex.open(root)
    wait(ex, 1.0)
    from explore2_sources_common import buttons, menu_button  # noqa: E402
    fl = ex.raws("saveP/flats", 3, start=8100)
    ex.click_block(blk("session") or blk("sources"))
    wait(ex, 0.3)
    bs = buttons(panel(ex), _("Add") + " ▾", QPushButton)
    ex.file_queue = [sorted(str(x) for x in fl.iterdir())]
    menu_button(ex, bs[2], _("Add files…"))
    wait(ex, 0.6)
    # one light taken out by its cross
    ex.click_block(blk("session") or blk("sources"))
    rows = [b for b in visible(panel(ex), QPushButton) if "📁" in b.text()]
    ex.click(rows[0])
    wait(ex, 0.4)
    xs = [b for b in visible(panel(ex), QPushButton) if b.text() == "×" and b.width() == 26 and b.height() == 20]
    if xs:
        ex.click(xs[0])
        wait(ex, 0.6)
    before = (len(ex.s.session_files(1, "lights")), len(ex.s.session_files(1, "flats")), ex.s.dirty)
    save()
    reopen(root)
    after = (len(ex.s.session_files(1, "lights")), len(ex.s.session_files(1, "flats")), ex.s.dirty)
    log("P saved/reopened", before, after, where())
    if before[:2] != after[:2]:
        N.note("gênant", "Brutes et flats modifiés, enregistrés, rouverts : pas les mêmes", f"{before} -> {after}",
               steps=["Session de 15 brutes", "Flats > Ajouter fichiers… 3", "× d'une brute", "Ctrl+S",
                      "fermer, rouvrir"])
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== Q. words seen on the way; notes of what was seen twice
if want("Q"):
    ex.new_project("WordsQ", None)
    ex.close_dialogs()
    wait(ex, 0.5)
    ex.double_click_card("image")
    wait(ex, 0.8)
    reasons = {k: c.reason for k, c in ex.page.cards.items() if c.reason}
    log("Q reasons", reasons)
    bad = [r for r in reasons.values() if " de A" in r or " de E" in r or " de É" in r or " de I" in r]
    if bad:
        ex.page.show_hint(ex.card("registration"), reasons.get("registration", ""))
        ex.settle(2)
        N.note("mineur", "Palette, Image posée : « pas de Alignement », « pas de Empilement » (élision manquante)",
               f"Raisons des blocs grisés : {bad}. Attendu « pas d'Alignement », « pas d'Empilement ».",
               steps=["projet Vierge", "Image posée (double-clic)", "survol des blocs grisés de la palette"])
    ex.w.close_project()
    wait(ex, 0.3)
    O = ex.out
    shot = lambda n: str(O / n) if (O / n).is_file() else False  # noqa: E731
    N.note("question", "Projet créé depuis un modèle : le bloc choisi est le dernier (Étoiles/fond), pas la Session à "
                       "remplir",
           "Après Créer avec « Complet » ou « Plusieurs nuits », le panneau ouvert est celui d'Étoiles/fond, alors que la "
           "barre d'état dit « rien à calculer : ajouter brutes ou image » : la première chose à faire est dans la "
           "Session, qu'il faut aller chercher (blocs petits, graphe dézoomé).",
           shot=shot("created_full.png"), steps=["Nouveau projet", "galerie : « Complet » (ou « Plusieurs nuits »)",
                                                 "Créer"])
    N.note("mineur", "Résumé du bloc Session : « 3 dark · 3 flat » sans pluriel (« 3 brutes » l'a)",
           "Bloc Session : « 6 brutes · 3 dark · 3 flat » ; en anglais « 6 lights · 3 dark · 3 flat ». Les brutes "
           "prennent le pluriel, pas les darks ni les flats. Calibration : « 5 dark · 0 flat ».",
           shot=shot("E_files_picked.png"), steps=["Session posée", "Darks > Ajouter fichiers… (3)",
                                                   "Flats > Ajouter fichiers… (3)"])
    N.note("question", "Modèle « Image déjà étirée » : « aucun bloc : en poser un » et Exécuter grisé",
           "Le modèle dit que l'image « va à Post telle quelle ». Projet créé : Image → Post, la barre d'état dit "
           "« 1 workflow · aucun bloc : en poser un » et Exécuter est grisé avec la même raison. Choisir un PNG dans "
           "une Image posée affiche au contraire « + ≈ 1 min à exécuter ». On ne sait pas si l'image est déjà dans "
           "Post ou s'il faut faire quelque chose.",
           shot=shot("J_created_stretched.png"), steps=["Nouveau projet", "galerie : onglet À partir d'une image, "
                                                        "« Image déjà étirée », un PNG", "Créer"])

# ================================================================== S. a lone Session: Del, dragged to the palette; rename by menu undone/redone
if want("S"):
    root = ex.project("session_empty_wf", "loneS")
    ex.open(root)
    wait(ex, 1.0)
    toasts = []
    ex.page.toasted.connect(lambda t: toasts.append(t))
    it = blk("session") or blk("sources")
    ex.click_block(it)
    ex.key(Qt.Key_Delete)
    wait(ex, 0.5)
    log("S Del on lone session", shape(), toasts[-1:], ex.w.status.text(), ex.popups[-1:])
    it = blk("session") or blk("sources")
    ex.drag_block_to_palette(it)
    wait(ex, 0.5)
    log("S dragged to palette", shape(), toasts[-1:], ex.w.status.text(), ex.drops[-1:])
    ex.shot("S_lone_session_dragged")
    if len(ex.s.p.sessions) == 1 and not toasts:
        N.note("mineur", "Session seule : Suppr ou glisser vers la palette ne fait rien, sans dire pourquoi",
               f"Le menu dit « Supprimer — seule session » (grisé) ; Suppr et le glisser vers la palette ne font rien et "
               f"rien ne s'affiche. Barre d'état : {ex.w.status.text()!r}",
               steps=["projet d'une Session", "clic sur la Session, Suppr", "glisser la Session sur la palette"])

    def name_dialog(newname):
        def h(d):
            eds = d.findChildren(QLineEdit)
            if eds:
                eds[0].setText(newname)
            d.accept()
        return h
    ex.dialog_handler = name_dialog("Nuit B")
    ex.right_click_block(blk("session") or blk("sources"), _("Rename session"))
    wait(ex, 0.5)
    n1 = ex.s.p.session(1).name
    undo(ex)
    n0 = ex.s.p.session(1).name
    undo(ex, redo=True)
    n2 = ex.s.p.session(1).name
    log("S rename by menu", n1, n0, n2)
    if not (n1 == "Nuit B" and n0 != "Nuit B" and n2 == "Nuit B"):
        N.note("gênant", "Renommer la session par son menu : Ctrl+Z / Ctrl+Y ne suivent pas",
               f"Après renommage {n1!r}, Ctrl+Z {n0!r}, Ctrl+Y {n2!r}",
               steps=["clic droit sur la Session > Renommer session > « Nuit B »", "Ctrl+Z", "Ctrl+Y"])
    # filter: redo
    ex.click_block(blk("session") or blk("sources"))
    from PySide6.QtWidgets import QComboBox
    cb = next((c for c in visible(panel(ex), QComboBox)), None)
    cb.setCurrentIndex(cb.findData("OIII"))
    wait(ex, 0.4)
    f1 = ex.s.p.session(1).filter
    undo(ex)
    f0 = ex.s.p.session(1).filter
    undo(ex, redo=True)
    f2 = ex.s.p.session(1).filter
    ex.click_block(blk("session") or blk("sources"))
    cb = next((c for c in visible(panel(ex), QComboBox)), None)
    log("S filter", f1, f0, f2, cb.currentData() if cb else None, (blk("session") or blk("sources")).summary)
    if not (f1 == "OIII" and f0 != "OIII" and f2 == "OIII") or (cb and cb.currentData() != f2):
        N.note("gênant", "Filtre de la session : Ctrl+Z / Ctrl+Y ne suivent pas",
               f"{f1!r} → Ctrl+Z {f0!r} → Ctrl+Y {f2!r}; panneau {cb.currentData() if cb else None!r}",
               steps=["Session : Filtre = OIII", "Ctrl+Z", "Ctrl+Y"])
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== T. gallery, image template: Create greyed, why?
if want("T"):
    for path in ("", "C:/nulle/part/absente.fits"):
        cap = {}

        def h(d, path=path, cap=cap):
            d.tabs.setCurrentIndex(d.tab_keys.index("image"))
            ex.settle(2)
            ex.click(next(c for c in d.cards if c.t.get("key") == "image"))
            d.file.setText(path)
            ex.settle(3)
            cap.update(ok=d.ok.isEnabled(), tip=d.ok.toolTip(), file_tip=d.file.toolTip(),
                       warn=[x.text() for x in d.findChildren(QLabel) if x.isVisible() and x.text()
                             and ("fichier" in x.text().lower() or "image" in x.text().lower())][:6])
            ex.shot(f"T_gallery_image_{'missing' if path else 'empty'}", d)
            d.reject()
        ex.open()
        ex.dialog_handler = h
        ex.w.create_project(str(ex.ws / f"T{len(path)}"), f"T{len(path)}", "M 31")
        wait(ex, 0.8)
        log("T", repr(path), cap)
        if ex.w.project is not None:
            ex.w.close_project()
            wait(ex, 0.3)

# ================================================================== V. Home cards of projects without frames; Session summary cut
if want("V"):
    ex.new_project("VideNuits", "hoo")
    wait(ex, 0.5)
    it = blk("session", sid=1) or blk("sources")
    log("V session tooltip", repr(it.toolTip()) if it else None, it.summary if it else None)
    ex.w.close_project()
    wait(ex, 0.5)
    from astro_gpu.client import home as homemod  # noqa: E402
    cards = [w for w in ex.w.findChildren(QWidget) if w.isVisible() and w.__class__.__name__.endswith("Card")]
    info = []
    for c in cards:
        t = [x.text() for x in c.findChildren(QLabel) if x.isVisible() and x.text()]
        info.append(t)
    log("V home cards", info)
    ex.shot("V_home_cards")
    vc = next((t for t in info if "VideNuits" in t), None)
    if vc and any("nuit" in x for x in vc):
        N.note("mineur", "Accueil : un projet sans aucune brute affiche « 2 nuits »",
               f"Carte : {vc}. Le projet vient du modèle HOO : deux Sessions vides (Ha, OIII), aucune brute. Un projet "
               "Vierge dit « Pas encore d'images » ; celui-ci compte ses Sessions vides comme des nuits.",
               steps=["Nouveau projet", "galerie : onglet Filtres, HOO", "Créer", "fermer : Accueil"])

# ================================================================== W. reopen on: a session's Calibration, a Fusion, Gradient 2
if want("W"):
    root = ex.project("mono", "reopenW")
    ex.open(root)
    wait(ex, 1.0)
    res = []

    def trial(label, item_fn, edit):
        it = item_fn()
        if it is None:
            res.append((label, "no block"))
            return
        ex.click_block(it)
        wait(ex, 0.3)
        edit()
        it = item_fn()
        ex.click_block(it)
        wait(ex, 0.3)
        chosen = where()
        save()
        reopen(root)
        got = where()
        res.append((label, chosen, got))
        log("W", label, chosen, got)
        ex.shot(f"W_reopen_{label}")
        if (got["flow"], got["block"]) != (chosen["flow"], chosen["block"]):
            N.note("gênant", f"Projet rouvert : {label} choisi, il s'ouvre ailleurs",
                   f"Choisi : bloc {chosen['block']} du workflow « {chosen['flow']} » ; rouvert : bloc {got['block']} "
                   f"du workflow « {got['flow']} ».",
                   steps=[f"clic sur {label}", "une modification", "Ctrl+S", "fermer, rouvrir"])
    trial("la Calibration de la session OIII", lambda: blk("calibration", sid=2),
          lambda: ex.s.rename_session(1, "Ha a"))
    ex.double_click_card("fusion")
    wait(ex, 1.0)
    trial("la Fusion", lambda: blk("fusion"), lambda: ex.s.rename_session(1, "Ha b"))
    ex.w.close_project()
    wait(ex, 0.3)
    # Gradient twice in an image flow
    root = ex.project("image", "reopenW2")
    ex.open(root)
    wait(ex, 1.0)
    ex.click_block(blk("image"))
    ex.double_click_card("gradient")
    wait(ex, 0.6)
    g1 = next((b for b in ex.blocks() if b.node.key.startswith("fond")), None)
    if g1 is not None:
        ex.click_block(g1)
        ex.double_click_card("gradient")
        wait(ex, 0.6)
    gs = [(b.node.key, b.title) for b in ex.blocks() if b.node.key.startswith("fond")]
    log("W gradients", gs)
    g2 = lambda: next((b for b in ex.blocks() if b.node.key.startswith("fond") and b.title.endswith("2")), None)  # noqa
    trial("Gradient 2", g2, lambda: ex.s.rename_session(1, "x") if ex.s.p.sessions else None)
    ex.w.close_project()
    wait(ex, 0.3)
    log("W results", res)

# ================================================================== X. import a mono folder as a new project, then a second filter
if want("X"):
    from explore2_sources_common import Dispatch  # noqa: E402
    ha = ex.fits_frames("NGC7000 2026-08-01/Ha", 5, "Ha")
    oiii = ex.fits_frames("NGC7000 2026-08-01/OIII", 5, "OIII")
    seen = {}

    def npd(d):
        seen["npd"] = (d.name.text(), d.target.text(), d.location.as_posix())
        d.ok.click() if d.ok.isEnabled() else d.reject()

    def imp(d):
        try:
            d.finish_now()
        except Exception as e:  # noqa: BLE001
            seen["imp_err"] = str(e)
        ex.settle(10)
        seen.setdefault("imp", []).append((d.found.text(), d.groups.text(), d.warn.text(),
                                           {k: (r.what.text()[-20:], r.detail.text()) for k, r in d.rows.items()}))
        ex.shot(f"X_import_dialog_{len(seen['imp'])}", d)
        d.ok.click() if d.ok.isEnabled() else d.reject()

    def gal(d):
        seen["gal"] = (d.tab_keys[d.tabs.currentIndex()], [c.t.get("key") for c in d.cards],
                       d.ok.isEnabled(), d.name.text())
        ex.shot("X_gallery", d)
        d.reject()

    ex.open()
    disp = Dispatch(ex, NewProjectDialog=npd, ImportDialog=imp, TemplateGallery=gal)
    ex.file_queue = [str(ha)]
    ex.w.import_folder_as_new()
    wait(ex, 2.0)
    log("X import Ha", seen, shape() if ex.s else None,
        [(x.id, x.name, x.filter) for x in ex.s.p.sessions] if ex.s else None)
    ex.shot("X_after_import_ha")
    if ex.s is not None:
        f = [x.filter for x in ex.s.p.sessions]
        if f and f[0] != "Ha":
            N.note("gênant", "Importer un dossier de FITS Ha comme nouveau projet : le filtre de la session reste vide",
                   f"Sessions : {[(x.name, x.filter) for x in ex.s.p.sessions]} ; les FITS ont FILTER=Ha.",
                   steps=["Accueil", "Importer un dossier comme nouveau projet…", "un dossier de 5 FITS FILTER=Ha"])
        u0 = stacks()
        ex.file_queue = [str(oiii)]
        ex.w.add_sources()
        wait(ex, 2.0)
        log("X import OIII (Ctrl+I)", seen.get("imp"), shape(), [(x.id, x.name, x.filter) for x in ex.s.p.sessions],
            [(q.name, ex.s.p.workflow(q.pid).sessions) for q in ex.s.paths], u0, stacks())
        ex.shot("X_after_import_oiii")
        wsx = [ex.s.p.workflow(q.pid).sessions for q in ex.s.paths]
        if any(set(x or []) >= {1, 2} for x in wsx):
            N.note("gênant", "Fichier > Importer dossier : des FITS OIII ajoutés à un projet Ha rejoignent le même "
                             "workflow que Ha",
                   f"Projet importé depuis un dossier de FITS Ha, puis Ctrl+I sur un dossier de FITS OIII : nouvelle "
                   f"session (filtre {ex.s.p.sessions[-1].filter!r}), dans le même workflow que Ha (sessions des "
                   f"workflows {wsx}) : Ha et OIII seraient alignés et empilés ensemble, sans avertissement. Les deux "
                   f"sessions portent le même nom « {ex.s.p.sessions[0].name} ».",
                   steps=["Accueil", "Importer un dossier comme nouveau projet… > dossier de FITS FILTER=Ha",
                          "Fichier > Importer dossier… (Ctrl+I) > dossier de FITS FILTER=OIII"])
    if seen.get("npd") and seen["npd"][1] == "Ha":
        N.note("mineur", "Importer un dossier nommé « Ha » : nom et cible proposés « Ha »",
               f"Dossier « NGC7000 2026-08-01/Ha » : la fenêtre Nouveau projet propose le nom {seen['npd'][0]!r} et la "
               f"cible {seen['npd'][1]!r}. Pour un dossier « lights », le dossier au-dessus est pris (« M81 2026-03-01 », "
               "cible M81) ; un dossier nommé d'après le filtre ne l'est pas.",
               shot=str(ex.out / "X_after_import_ha.png"),
               steps=["Accueil", "Importer un dossier comme nouveau projet…", "dossier NGC7000 2026-08-01/Ha"])
    disp.stop()
    if ex.w.project is not None:
        ex.w.close_project()
        wait(ex, 0.3)

# ================================================================== Y. TIFF kinds; a PNG in the linear template; the image file gone
if want("Y"):
    import numpy as np
    from PIL import Image as PI
    d = TMP / "data" / "tiffs"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    t8 = d / "stretched8.tif"
    PI.fromarray((rng.random((80, 100, 3)) * 255).astype(np.uint8)).save(t8)
    t16 = d / "linear16.tif"
    try:
        import tifffile
        tifffile.imwrite(t16, (rng.random((80, 100, 3)) * 3000 + 500).astype(np.uint16))
        t32 = d / "linear32.tif"
        tifffile.imwrite(t32, (rng.random((80, 100, 3)) * 0.05 + 0.1).astype(np.float32))
    except ImportError:
        PI.fromarray((rng.random((80, 100)) * 3000 + 500).astype(np.uint16)).save(t16)
        t32 = None
    ex.new_project("TiffY", None)
    ex.close_dialogs()
    wait(ex, 0.5)
    ex.double_click_card("image")
    wait(ex, 0.8)
    kinds = {}
    for f in [t8, t16, t32] if "skip_tiff" not in sys.argv else []:
        if f is None:
            continue
        ex.click_block(blk("image"))
        wait(ex, 0.3)
        b = button(panel(ex), _("Choose image…"), QPushButton) or button(panel(ex), _("Change image…"), QPushButton)
        ex.file_queue = [str(f)]
        ex.click(b)
        wait(ex, 0.6)
        q = next(q for q in ex.s.paths if q.source is not None)
        kinds[f.name] = dict(q.source).get("stretched")
        # back to linear before the next file, as a user would
        ex.click_block(blk("image"))
        lb = next((x for x in visible(panel(ex), QPushButton) if x.text() == _("Linear")), None)
        if lb is not None and lb.isEnabled():
            ex.click(lb)
            wait(ex, 0.3)
    log("Y tiff kinds", kinds)
    if any(v for k, v in kinds.items() if k.startswith("linear")):
        N.note("gênant", "Image : un TIFF 16 bits ou 32 bits linéaire choisi est marqué « Étirée »",
               f"Nature mise par le choix du fichier : {kinds}. Le modèle « Traiter une image empilée » et le champ de la "
               "galerie disent « un fichier FITS ou TIFF, sorti directement d'un empilement » : un TIFF de pile est "
               "linéaire. Ici tout TIFF passe en Étirée, et Gradient, Astrométrie, Couleur ne se posent plus derrière.",
               steps=["projet Vierge", "Image posée", "Choisir image… > un TIFF 16 bits (ou 32 bits flottant) linéaire"])
    ex.w.close_project()
    wait(ex, 0.3)
    # the image file moved away after Save
    img = ex.image_file("bientot_parti.fits")
    root = ex.ws / "GoneY"
    ex.new_project("GoneY", None)
    ex.close_dialogs()
    wait(ex, 0.5)
    ex.double_click_card("image")
    wait(ex, 0.6)
    ex.click_block(blk("image"))
    b = button(panel(ex), _("Choose image…"), QPushButton)
    ex.file_queue = [str(img)]
    ex.click(b)
    wait(ex, 0.6)
    ex.click_block(blk("image"))
    ex.double_click_card("gradient")
    wait(ex, 0.6)
    save()
    ex.w.close_project()
    wait(ex, 0.3)
    trash = Path("F:/astro_essais/_trash")
    trash.mkdir(parents=True, exist_ok=True)
    moved = trash / f"bientot_parti_{int(__import__('time').time())}.fits"
    __import__('shutil').move(str(img), str(moved))
    ex.w.open_project(root)
    wait(ex, 1.5)
    ex.click_block(blk("image"))
    wait(ex, 0.3)
    warn = [t for t in ptexts(ex) if "introuv" in t.lower() or "missing" in t.lower() or "not found" in t.lower()
            or "déplac" in t.lower()]
    log("Y gone", warn, (blk("image").summary, blk("image").node.broken if blk("image") else None),
        ex.w.bt_run.isEnabled(), ex.w.bt_run.toolTip(), ex.w.status.text())
    ex.shot("Y_image_file_gone")
    n_pop, n_mod = len(ex.popups), len(ex.modals)
    try:
        ex.click(ex.w.bt_run)
    except Exception as e:  # noqa: BLE001
        log("Y run raised", type(e).__name__, e)
    wait(ex, 2.0)
    run_dlgs = [(x.__class__.__name__, x.windowTitle(), [t for t in __import__("explore2_sources_common").texts(x)
                                                          if t.strip()][:40])
                for x in ex.dialogs()]
    log("Y run with the file gone", ex.popups[n_pop:], ex.modals[n_mod:], run_dlgs, ex.w.status.text())
    ex.shot("Y_run_with_file_gone")
    [ex.shot("Y_run_dialog", x) for x in ex.dialogs()]
    ex.close_dialogs()
    if blk("image") is not None and not blk("image").node.broken and ex.w.bt_run.isEnabled():
        N.note("gênant", "Image dont le fichier a disparu : le panneau le dit, mais ni le bloc ni Exécuter",
               f"Panneau : « {warn[0] if warn else ''} ». Le bloc Image n'est ni entouré ni marqué ⚠ (résumé "
               f"« {blk('image').summary} ») ; Exécuter reste actif, la barre d'état dit « {ex.w.status.text()} ». "
               f"Après Exécuter : fenêtres {ex.modals[n_mod:]}, messages {ex.popups[n_pop:]}.",
               steps=["Image posée, un FITS choisi, Gradient posé", "Ctrl+S, fermer", "déplacer le FITS ailleurs",
                      "rouvrir le projet", "clic sur l'Image", "Exécuter"])
    if not warn:
        N.note("gênant", "Image dont le fichier a disparu : rien ne le dit à la réouverture",
               f"Panneau : {ptexts(ex)[:8]}; Exécuter actif : {ex.w.bt_run.isEnabled()}",
               steps=["Image posée, un FITS choisi, Gradient", "Ctrl+S, fermer", "déplacer le FITS", "rouvrir"])
    ex.w.close_project()
    wait(ex, 0.3)

# ================================================================== Z. Ctrl+Z right after a template; after Ctrl+I
if want("Z"):
    from explore2_sources_common import Dispatch  # noqa: E402

    def galz(d):
        d.tabs.setCurrentIndex(d.tab_keys.index("frames"))
        ex.settle(2)
        t = next(c.t for c in d.cards if c.t.get("key") == "classic")
        d.pick(t)
        d.ok.click()
    ex.open()
    ex.dialog_handler = galz
    ex.w.create_project(str(ex.ws / "UndoZ"), "UndoZ", "M 31")
    wait(ex, 1.0)
    s0 = shape()
    st0 = stacks()
    u = undo(ex)
    s1 = shape()
    log("Z undo after template", u, st0, s0["blocks"], "->", s1["blocks"])
    ex.shot("Z_undo_after_template")
    if s1 != s0:
        undo(ex, redo=True)
        log("Z redo", shape() == s0)
    # Ctrl+I: a second folder imported, then Ctrl+Z / Ctrl+Y
    l1 = ex.fits_frames("undoZ/l1", 4)
    l2 = ex.fits_frames("undoZ/l2b", 3)

    def imp(d):
        try:
            d.finish_now()
        except Exception:  # noqa: BLE001
            pass
        ex.settle(8)
        d.ok.click() if d.ok.isEnabled() else d.reject()
    disp = Dispatch(ex, ImportDialog=imp)
    for folder in (l1, l2):
        ex.file_queue = [str(folder)]
        ex.w.add_sources()
        wait(ex, 1.5)
        log('Z import', folder.name, disp.seen[-1:], [(x.id, x.name, x.lights) for x in ex.s.p.sessions], ex.w.status.text(), ex.popups[-1:], ex.file_asks[-1:])
    disp.stop()
    a = shape()
    u = undo(ex)
    b = shape()
    undo(ex, redo=True)
    c = shape()
    lc = lambda: [(x.id, len(ex.s.session_files(x.id, "lights"))) for x in ex.s.p.sessions]  # noqa: E731
    log("Z Ctrl+I", a, u, b, c, lc())
    ex.shot("Z_after_ctrl_i_undo_redo")
    if any(n == 0 for _i, n in lc()) and len(a["sessions"]) >= 3:
        N.note("gênant", "Modèle puis Fichier > Importer dossier : la Session vide du modèle reste à part, les brutes "
                         "vont dans de nouvelles Sessions",
               f"Projet « Empilement classique » (une Session 1 vide, « brutes à ajouter »), puis Ctrl+I sur deux "
               f"dossiers : brutes par session {lc()}, sessions {a['sessions']}, sessions du workflow "
               f"{a['flows']}. La Session 1 reste dessinée au-dessus, vide et reliée à rien ; les deux nouvelles "
               "portent le même nom (la date). Le panneau affiché ensuite s'appelle « Images ».",
               steps=["Nouveau projet", "galerie : « Empilement classique »", "Créer",
                      "Fichier > Importer dossier… (Ctrl+I) > un dossier de FITS", "Ctrl+I > un autre dossier"])
    if b["sessions"] == a["sessions"]:
        N.note("gênant", "Fichier > Importer dossier (Ctrl+I) : Ctrl+Z ne retire pas la session ajoutée",
               f"Sessions après deux imports : {a['sessions']} ; après Ctrl+Z : {b['sessions']}",
               steps=["projet « Empilement classique »", "Ctrl+I > un dossier", "Ctrl+I > un autre dossier", "Ctrl+Z"])
    ex.w.close_project()
    wait(ex, 0.3)

N.exceptions(["voir new_log.txt"])
(ex.out / f"new_log_{ex.lang}.txt").write_text("\n".join(LOG), encoding="utf-8")
print("DONE", TMP)
ex.close()
