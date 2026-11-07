"""palette_matrix, second pass: each distinct anomaly the matrix found is reproduced once more, from a fresh project, and
noted (in French) with its screenshot.  python explore_palette_confirm.py [name...]"""
from __future__ import annotations

import json
import sys

from explore_palette_lib import Matrix

from PySide6.QtCore import QPoint

from astro_gpu.client import i18n

m = Matrix("confirm")
ex = m.ex
_note = ex.note


def note(severity, title, detail="", shot=True, steps=None):
    """harness.note, except that a screenshot named here and already taken (at the moment that shows the anomaly) is
    kept as it is, not taken again at the time of the note."""
    import time as _t
    if isinstance(shot, str) and (ex.out / f"{shot}.png").exists():
        rec = {"area": ex.area, "severity": severity, "title": title, "detail": detail, "steps": steps or [],
               "shot": str(ex.out / f"{shot}.png"), "lang": ex.lang, "t": _t.strftime("%H:%M:%S")}
        with ex.log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("ANOMALY", severity, title, flush=True)
        return
    _note(severity, title, detail, shot, steps)


ex.note = note
ONLY = sys.argv[1:]
DONE = []


def case(fn):
    if not ONLY or fn.__name__ in ONLY:
        DONE.append(fn)
    return fn


def fresh(kind, name=None):
    if kind.startswith("new_"):
        tpl = {"new_blank": None, "new_full": "full", "new_nights": "nights", "new_sho": "sho",
               "new_stretched": "image_stretched", "new_image": "image", "new_raw": "raw"}[kind]
        m._k = getattr(m, "_k", 0) + 1
        root, _got = m.new_project(f"C{m._k}", tpl)
    else:
        root = ex.project(kind)
        ex.open(root)
    ex.settle(8)
    m.hook_page()
    ex.popups.clear()
    ex.modals.clear()
    return root


def paths():
    s = ex.s
    return [(q.name, s.graph(q)) for q in s.paths]


def item(key, nth=0):
    its = [it for it in ex.blocks() if it.node.key == key]
    its.sort(key=lambda it: (it.sceneBoundingRect().y(), it.sceneBoundingRect().x()))
    return its[nth] if len(its) > nth else None


# ------------------------------------------------------------ the cases
@case
def image_undo_two_steps():
    """An Image placed, one Ctrl+Z: an Image workflow stays, with blocks the user never placed."""
    for attempt in (1, 2):
        fresh("session+image")
        before = paths()
        ex.double_click_card("image")
        placed = paths()
        ex.undo()
        one = paths()
        if attempt == 2 and one != before:
            extra = [p for p in one if p not in before]
            ex.note("gênant", "Image posée puis Ctrl+Z : il reste un workflow « Image 2 » avec des blocs jamais posés",
                    "Double-clic sur la carte Image de la palette (projet avec une Session et une Image), puis un seul "
                    "Ctrl+Z. Attendu (PRINCIPLES : chaque modification s'annule, retour exact à l'état d'avant) : le "
                    "bloc Image disparaît. Vu : un workflow reste, avec les blocs " +
                    ", ".join(n['block'] for p in extra for n in p[1]) + " que personne n'a posés ; il faut un "
                    "second Ctrl+Z. Même chose pour une Image déposée par glisser (zone du bas, vide, sur un bloc, "
                    "sur Post). Avant : " + repr([p[0] for p in before]) + " ; après pose : " +
                    repr([p[0] for p in placed]) + " ; après un Ctrl+Z : " + repr([(n, [b['block'] for b in g])
                                                                                  for n, g in one]),
                    steps=["projet Session + Image", "double-clic carte Image", "Ctrl+Z"])
        print("image_undo", attempt, one != before, flush=True)


@case
def zone_not_lit():
    """Dragging a Session or an Image: the bottom zone (new flow) does not light up."""
    res = []
    for attempt in (1, 2):
        fresh("session+image")
        z = ex.new_flow_zone()
        pt = ex.view.mapFromScene(z.sceneBoundingRect().center())
        ex.drag_palette_hover("session", pt)
        z = ex.new_flow_zone()
        res.append((z.hot, [g.hot for g in ex.ghosts()]))
        if attempt == 2 and not z.hot:
            ex.note("gênant", "En glissant une Session ou une Image, la zone du bas ne s'allume pas",
                    "Carte Session (ou Image, ou Fusion) tenue au-dessus de la zone « + Nouveau flux : déposez une "
                    "Session ou une Image ici ». Attendu (PRINCIPLES : pendant le glisser, seules les places valides "
                    "s'allument ; une Session ou une Image déposée sur la zone du bas démarre un flux parallèle) : la "
                    "zone s'allume. Vu : rien ne s'allume, la zone reste grise, et la Session est acceptée partout "
                    "dans le graphe (sur Post, sur un bloc, dans le vide) sans indication de ce qui va se passer.",
                    steps=["projet Session + Image", "glisser la carte Session au-dessus de la zone du bas"])
        ex.drag_leave()
    print("zone", res, flush=True)


@case
def session_state_no_place():
    """Sessions without a workflow: Calibration is available in the palette, but dragging it shows no place."""
    for attempt in (1, 2):
        fresh("session")
        c = ex.card("calibration")
        reason = c.reason
        vp = ex.view.viewport()
        ex.drag_palette_hover("calibration", QPoint(vp.width() // 2, vp.height() // 2))
        gh = ex.ghosts()
        if attempt == 2:
            ex.shot("session_sans_workflow_glisser_calibration")
        ex.drag_leave()
        si = item("sources") or item("session")
        rec = ex.drag_palette("calibration", ex.point_of(si) if si else QPoint(vp.width() // 2, vp.height() // 2))
        after = paths()
        print("session_state", attempt, repr(reason), len(gh), rec, after, flush=True)
        if attempt == 2 and not reason and not gh:
            ex.note("gênant", "Session sans workflow : Calibration est disponible mais le glisser ne montre aucune place",
                    "Projet avec une Session (15 poses) et aucun workflow. Dans la palette, Calibration, Alignement, "
                    "Sélection, Empilement ne sont pas grisés (le double-clic les pose). Mais en glissant la carte sur "
                    "le graphe, aucune place ne s'allume et le dépôt est refusé (même sur le bloc Session). Attendu "
                    "(PRINCIPLES : un bloc se pose par glisser-déposer ou double-clic ; pendant le glisser, les places "
                    "valides s'allument) : une place « après les sessions ». Dépôt accepté : " + str(rec) +
                    " ; workflows après : " + repr(after), shot="session_sans_workflow_glisser_calibration",
                    steps=["projet Session seule", "glisser la carte Calibration sur le graphe"])


def comp_items():
    return [(it.node.lid, it.title) for it in ex.blocks() if it.node.key == "composition"]


@case
def composition_undo():
    """A Recombine block placed from the palette: the first double-click shows nothing new, Ctrl+Z
    never takes it away, the two blocks have the same name, the layer's name is English."""
    for attempt in (1, 2):
        fresh("legacy")
        c0 = comp_items()
        ex.double_click_card("composition")
        c1 = comp_items()
        if attempt == 2:
            ex.shot("combiner_premier_double_clic")
        ex.double_click_card("composition")
        c2 = comp_items()
        undo_on = ex.w.a_undo.isEnabled()
        ex.undo()
        c3 = comp_items()
        names = [x.get("name") for x in ex.s.post_layers()]
        print("composition", attempt, c0, c1, c2, c3, undo_on, names, flush=True)
        if attempt != 2:
            continue
        if len(c1) == len(c0):
            ex.note("mineur", "Premier double-clic sur « Combiner les calques étoiles et fond » : rien ne change à "
                    "l'écran", "Workflow complet (avec Split étoiles et fond) : la section Composition montre déjà un "
                    "bloc « Combiner les calques étoiles et fond ». Double-clic sur la carte du même nom : toujours un "
                    f"seul bloc ({c0} → {c1}), aucun message ; le second double-clic en ajoute un. Attendu : un bloc de "
                    "plus à chaque pose, ou la carte grisée avec la raison (« déjà là »).",
                    shot="combiner_premier_double_clic",
                    steps=["projet workflow complet", "double-clic carte Combiner"])
        if len(c3) == len(c2):
            ex.note("gênant", "Bloc « Combiner les calques étoiles et fond » posé depuis la palette : Ctrl+Z ne le "
                    "retire pas", "Workflow complet, deux double-clics sur la carte « Combiner les calques étoiles et "
                    f"fond » : {len(c2)} blocs dans la section Composition. Ctrl+Z (Édition > Annuler "
                    f"{'actif' if undo_on else 'grisé'}) : il en reste {len(c3)}. Attendu (PRINCIPLES : chaque "
                    "modification s'annule par Ctrl+Z, retour exact) : le dernier bloc posé disparaît.",
                    steps=["projet workflow complet", "double-clic carte Combiner (×2)", "Ctrl+Z"])
        if len(c2) >= 2 and len({t for _l, t in c2}) == 1:
            ex.note("mineur", "Deux blocs « Combiner les calques étoiles et fond » portent exactement le même nom",
                    f"Après deux poses, la section Composition montre {len(c2)} blocs nommés "
                    f"« {c2[0][1]} », sans numéro. Attendu (PRINCIPLES : les copies sont numérotées de gauche à droite, "
                    "« Gradient 1 », « Gradient 2 ») : des noms qui les distinguent.",
                    steps=["projet workflow complet", "double-clic carte Combiner (×2)"])
        eng = [n for n in names if n and "Combine" in n]
        if eng:
            ex.note("mineur", "Calque de Post créé par « Combiner » nommé en anglais",
                    f"Le calque de Post que crée la pose du bloc s'appelle « {eng[0]} » (anglais) alors que "
                    "l'interface est en français. Attendu (PRINCIPLES : en français, aucun mot anglais) : "
                    "« … · Combiner les calques étoiles et fond ». (Nom vu dans les calques du projet ; à vérifier "
                    "dans la liste des calques de Post.)", shot=False,
                    steps=["projet workflow complet", "double-clic carte Combiner", "ouvrir Post-traitement"])


@case
def fusion_on_empty_colour():
    """Fusion placed when the colour workflow is empty (no stacking)."""
    for attempt in (1, 2):
        fresh("session+image")
        reason = ex.card("fusion").reason
        ex.double_click_card("fusion")
        ps = paths()
        fus = [q for q in ex.s.paths if q.source and "fusion" in q.source]
        print("fusion", attempt, repr(reason), [q.name for q in fus], flush=True)
        if attempt == 2 and not reason and fus:
            ex.note("question", "Fusion disponible et posée sur un workflow couleur encore vide",
                    "Projet : une Session dont le workflow est vide (aucun bloc), plus une Image. La carte Fusion "
                    "n'est pas grisée ; un double-clic pose une Fusion « " + fus[0].name + " » qui prend les canaux "
                    "R et GB du workflow vide (pas encore d'Empilement, donc pas d'image). Attendu ? Griser Fusion "
                    "tant que le workflow couleur ne donne pas d'image, avec la raison, comme les autres blocs.",
                    steps=["projet Session (workflow vide) + Image", "double-clic carte Fusion"])


@case
def session_dropped_anywhere():
    """A Session dropped on Post, on a block or in the void: a session added to the existing flow."""
    for attempt in (1, 2):
        fresh("session+image")
        post = ex.view.final_item
        n0 = len(ex.s.p.sessions)
        w0 = [(q.name, ex.s.sessions_of(q)) for q in ex.s.paths]
        rec = ex.drag_palette("session", ex.view.mapFromScene(post.sceneBoundingRect().center()))
        w1 = [(q.name, ex.s.sessions_of(q)) for q in ex.s.paths]
        print("session_on_post", attempt, rec, n0, len(ex.s.p.sessions), w0, w1, flush=True)
        if attempt == 2 and rec and rec["accepted"]:
            ex.note("question", "Une Session déposée sur Post (ou sur un bloc, ou dans le vide) est acceptée",
                    "Carte Session glissée et lâchée sur le bloc Post. Vu : le dépôt est accepté, une session est "
                    f"ajoutée ({n0} → {len(ex.s.p.sessions)} sessions), workflows {w0} → {w1}. Rien ne s'allumait "
                    "pendant le glisser. Attendu (PRINCIPLES : seules les places valides s'allument ; une Session "
                    "déposée sur la zone du bas démarre un flux parallèle) : un dépôt sur Post ou sur un bloc qui n'a "
                    "rien à voir est refusé, ou la place visée s'allume. À trancher.",
                    steps=["projet Session + Image", "glisser la carte Session sur Post"])


@case
def multi_step_undo():
    """Session placed on a blank page; a frame block double-clicked when there are sessions and no workflow: one Ctrl+Z
    leaves part of it."""
    out = {}
    for attempt in (1, 2):
        fresh("blank")
        b0 = m.state()
        ex.double_click_card("session")
        ex.undo()
        out["session"] = (m.state() == b0, m.diff(b0, m.state()))
        fresh("session")
        b0 = m.state()
        ex.double_click_card("calibration")
        a1 = paths()
        ex.undo()
        out["calibration"] = (m.state() == b0, m.diff(b0, m.state()), a1)
        if attempt == 2:
            ex.shot("calibration_puis_ctrlz_session_seule")
        print("multi", attempt, {k: v[0] for k, v in out.items()}, flush=True)
    if not out["session"][0] or not out["calibration"][0]:
        ex.note("gênant", "Une seule pose, plusieurs Ctrl+Z : Session sur page vierge, bloc de poses sur Session seule",
                "Même famille que l'Image (note précédente). (1) Projet vierge, double-clic (ou dépôt) de la carte "
                "Session : la session et un workflow vide sont créés ; un Ctrl+Z retire le workflow mais laisse la "
                f"session ({'ok' if out['session'][0] else out['session'][1]}). (2) Projet avec une Session sans "
                "workflow, double-clic sur Calibration (ou Alignement, Sélection, Empilement) : un workflow est créé "
                f"avec le bloc ({out['calibration'][2]}) ; un Ctrl+Z retire le bloc mais laisse un workflow vide "
                f"({'ok' if out['calibration'][0] else out['calibration'][1]}). Attendu (PRINCIPLES : chaque "
                "modification s'annule, retour exact à l'état d'avant) : un Ctrl+Z par geste.",
                shot="calibration_puis_ctrlz_session_seule",
                steps=["projet vierge", "double-clic Session", "Ctrl+Z", "projet Session seule",
                       "double-clic Calibration", "Ctrl+Z"])


def new_window():
    """A stuck project (it cannot be saved, so it cannot be closed): a new main window for what follows (the stuck
    one is left hidden)."""
    if ex.w is not None:
        ex.w.hide()
    ex.w = None
    ex.open()


@case
def template_session_zone():
    """A new project made from the Full template: a Session dropped on the bottom zone; then Undo, Save, close."""
    res = []
    for attempt in (1, 2):
        new_window()
        fresh("new_full")
        e0 = len(m.exc)
        w0 = [(q.name, ex.s.sessions_of(q)) for q in ex.s.paths]
        z = ex.new_flow_zone()
        rec = ex.drag_palette("session", ex.view.mapFromScene(z.sceneBoundingRect().center()))
        ex.settle(6)
        w1 = [(q.name, ex.s.sessions_of(q)) for q in ex.s.paths]
        exc = [x.splitlines()[0] for x in m.exc[e0:]]
        sess = [x.name for x in ex.s.p.sessions]
        if attempt == 2:
            ex.shot("modele_complet_session_zone")
        ex.undo()
        undo_ok = [x.name for x in ex.s.p.sessions] == [] and [(q.name, ex.s.sessions_of(q)) for q in ex.s.paths] == w0
        e1 = len(m.exc)
        save_err = close_err = None
        try:
            ex.s.flush()                                   # what Save writes
        except Exception as e:  # noqa: BLE001
            save_err = f"{type(e).__name__}: {e}"
        try:
            r = ex.w.close_project(go_home=True)
            ex.settle(6)
            close_err = None if r is not False else "close refused"
        except Exception as e:  # noqa: BLE001
            close_err = f"{type(e).__name__}: {e}"
        res.append(dict(rec=rec, w0=w0, w1=w1, exc=exc, sess=sess, undo_ok=undo_ok, save=save_err, close=close_err,
                        later=[x.splitlines()[0] for x in m.exc[e1:]]))
        print("tpl_zone", attempt, res[-1], flush=True)
    r = res[-1]
    if r["exc"]:
        ex.note("bloquant", "Projet neuf (modèle Complet) : une Session déposée sur la zone du bas lève une erreur "
                "Python et laisse un projet qu'on ne peut plus enregistrer", "Nouveau projet, modèle « Complet » "
                "choisi dans la fenêtre des modèles (pas encore de poses). Carte Session glissée sur la zone « + "
                "Nouveau flux ». Vu : " + r["exc"][0] + " (dans l'application réelle : une boîte d'erreur avec la "
                f"trace Python). La Session est quand même ajoutée ({r['sess']}), workflows {r['w0']} → {r['w1']}. "
                f"Ensuite, Ctrl+Z {'remet' if r['undo_ok'] else 'ne remet pas'} l'état d'avant ; enregistrer : "
                f"{r['save'] or 'ok'} ; fermer le projet : {r['close'] or 'ok'}. (Sans Ctrl+Z, ouvrir un autre "
                "projet échoue sur la même erreur : le projet ne peut plus être enregistré ni fermé.) Même chose avec "
                "les modèles « Plusieurs nuits » et « Empilement brut » (tout modèle de poses, projet sans poses). Attendu (PRINCIPLES : une Session déposée sur la zone du bas démarre "
                "un flux parallèle ; pas d'erreur Python) : un second flux avec sa Session vide.",
                shot="modele_complet_session_zone",
                steps=["nouveau projet, modèle Complet", "glisser la carte Session sur la zone du bas",
                       "ouvrir un autre projet"])
    new_window()


@case
def calibration_second_flow():
    """Two parallel flows: Calibration dropped on the second flow's place."""
    res = []
    for attempt in (1, 2):
        fresh("two_flows")
        e0 = len(m.exc)
        vp = ex.view.viewport()
        ex.drag_palette_hover("calibration", QPoint(vp.width() // 2, 4))
        gh = [(g.pid, ex.ghost_center(g), g.toolTip()) for g in ex.ghosts()]
        ex.drag_leave()
        tgt = next((g for g in gh if g[0] == 2), None)
        if tgt is None:
            print("no ghost for flow 2", gh)
            return
        ex.drag_palette("calibration", tgt[1])
        ex.settle(6)
        exc = [x.splitlines()[0] for x in m.exc[e0:]]
        res.append((exc, paths(), m.heading(), ex.s.sel_step, ex.s.path.name if ex.s.path else None))
        if attempt == 2:
            ex.shot("deux_flux_calibration_flux2")
        print("cal_flow2", attempt, res[-1], flush=True)
    if res[-1][0]:
        ex.note("bloquant", "Deux flux parallèles : Calibration déposée sur le second flux lève une erreur Python",
                "Projet à deux flux (Session 1 → « Workflow », Session 2 → « Flow », chacun vide). Carte Calibration "
                "glissée sur la place allumée du second flux (« Calibration ici, dans \"Flow\" »). Vu : le bloc est "
                "posé dans « Flow » mais le panneau lève " + res[-1][0][0] + " (panel.block : node.paths[0]) — dans "
                "l'application réelle une boîte d'erreur avec la trace Python ; l'en-tête de la palette reste « " +
                res[-1][2] + " », le bloc choisi est " + repr(res[-1][3]) + " du workflow « " + str(res[-1][4]) +
                " ». Attendu : la Calibration de la Session 2 posée et choisie, sans erreur, la palette « pour "
                "Flow ».", shot="deux_flux_calibration_flux2",
                steps=["projet à deux flux", "glisser Calibration sur la place du second flux"])


@case
def mono_branch_ghost_overlap():
    for attempt in (1, 2):
        fresh("mono")
        vp = ex.view.viewport()
        ex.drag_palette_hover("astrometry", QPoint(vp.width() // 2, 4))
        gh = [(f"ghost:{g.text}[{g.pid}]", g.sceneBoundingRect()) for g in ex.ghosts()]
        over = [o for o in m.overlaps(gh) if "ghost" in o]
        if attempt == 2:
            ex.shot("mono_branche_chevauche")
        ex.drag_leave()
        print("mono_overlap", attempt, over, flush=True)
    if over:
        ex.note("mineur", "Pendant le glisser, la place « nouvelle branche » chevauche le bloc du flux d'en dessous",
                "Projet mono (Ha et OIII, chacun Calibration → Alignement → Empilement). Carte Astrométrie (ou "
                "Gradient, Débruitage) tenue au-dessus du graphe : sous l'Empilement de Ha apparaît la place "
                "« ⑂ nouvelle branche », qui recouvre en partie l'Empilement de OIII : " + "; ".join(over) +
                ". Attendu (PRINCIPLES : rien ne se chevauche) : la place de branche ne cache aucun bloc.",
                shot="mono_branche_chevauche",
                steps=["projet mono Ha + OIII", "glisser la carte Astrométrie sans lâcher"])


@case
def zone_over_blocks():
    """The bottom new-flow zone drawn over the blocks of the sources column (an empty workflow)."""
    found = {}
    for attempt in (1, 2):
        for kind in ("session+image", "two_sessions"):
            fresh(kind)
            o = [x for x in m.overlaps() if "Nouveau flux" in x]
            if attempt == 2 and o:
                ex.shot(f"zone_chevauche_{kind.replace('+', '_')}")
            found.setdefault(kind, []).append(o)
        fresh("session_empty_wf")
        ex.double_click_card("image")
        o = [x for x in m.overlaps() if "Nouveau flux" in x]
        if attempt == 2 and o:
            ex.shot("zone_chevauche_image_posee")
        found.setdefault("session_empty_wf+image", []).append(o)
    print("zone_over", found, flush=True)
    if found["session+image"][1]:
        ex.note("gênant", "La zone « + Nouveau flux » est dessinée par-dessus les blocs de la colonne des sources",
                "Dès que le workflow des poses est vide (aucun bloc encore), la zone du bas « + Nouveau flux : déposez "
                "une Session ou une Image ici » démarre dans la colonne des sources et recouvre les blocs qui y sont "
                "empilés : l'Image (projet Session + Image, à l'ouverture), la Session 2 et « + Session » (projet à "
                "deux sessions), une Image ou une Fusion posée depuis la palette, une seconde Session. Chevauchements "
                "mesurés : " + "; ".join(found["session+image"][1] + found["two_sessions"][1]) + ". Attendu "
                "(PRINCIPLES : rien ne se chevauche) : la zone sous le dernier bloc.",
                shot="zone_chevauche_session_image",
                steps=["projet Session (workflow vide) + Image, à l'ouverture"])


@case
def zone_over_calibration2():
    res = []
    for attempt in (1, 2):
        fresh("legacy")
        ex.double_click_card("session")
        o = [x for x in m.overlaps() if "Nouveau flux" in x]
        res.append(o)
        if attempt == 2 and o:
            ex.shot("zone_chevauche_calibration_session2")
    print("zone_cal2", res, flush=True)
    if res[1]:
        ex.note("mineur", "Workflow complet : après une seconde Session, sa Calibration touche la zone « + Nouveau flux »",
                "Projet au workflow complet (Session 1 → … → Split étoiles et fond). Double-clic sur la carte Session : "
                "la Session 2 et sa Calibration apparaissent ; la Calibration de la Session 2 chevauche la zone du bas "
                "(" + "; ".join(res[1]) + "). Attendu (PRINCIPLES : rien ne se chevauche).",
                shot="zone_chevauche_calibration_session2",
                steps=["projet workflow complet", "double-clic carte Session"])


@case
def stretched_template():
    res = []
    for attempt in (1, 2):
        fresh("new_stretched")
        s = ex.s
        p = s.paths[0] if s.paths else None
        if p is None:
            print("no workflow")
            return
        res.append((p.name, (p.source or {}).get("stretched"), [n["block"] for n in s.graph(p)],
                    [b for b, c in ex.page.cards.items() if not c.reason]))
        if attempt == 2:
            ex.shot("modele_image_etiree")
        print("stretched", attempt, res[-1], flush=True)
    if res[-1][1] is False:
        ex.note("gênant", "Modèle « Image déjà étirée » : l'image est prise comme linéaire, avec Astrométrie, Gradient "
                "et Couleur", "Nouveau projet, modèle « Image déjà étirée » (un PNG choisi). Le texte du modèle dit que "
                "l'image va au Post telle quelle. Vu : le workflow « " + res[-1][0] + " » a les blocs " +
                ", ".join(res[-1][2]) + " sous « Traitement linéaire », la source est marquée non étirée, et la "
                "palette propose " + ", ".join(res[-1][3]) + " (Gradient et Couleur sur une image étirée). Attendu "
                "(PRINCIPLES : la nature de la source, linéaire ou étirée, est respectée) : Image → Post, la palette "
                "grisant les blocs linéaires avec la raison.", shot="modele_image_etiree",
                steps=["nouveau projet", "fenêtre des modèles : « Image déjà étirée », un PNG"])


@case
def empty_panel_at_open():
    res = []
    for attempt in (1, 2):
        fresh("image")
        from PySide6.QtWidgets import QLabel
        pan = [x.text() for x in ex.page.panel.findChildren(QLabel) if x.isVisible() and x.text()]
        res.append((ex.s.sel_step, ex.page.current_node(), pan, m.heading()))
        if attempt == 2:
            ex.shot("panneau_vide_ouverture")
        print("empty_panel", attempt, res[-1], flush=True)
    if res[-1][1] is None and not res[-1][2]:
        ex.note("question", "À l'ouverture, aucun bloc choisi : panneau de droite vide, la palette vise pourtant un "
                "workflow", "Projets dont le workflow n'a pas de bloc Empilement (une Image seule, une Session au "
                "workflow vide, deux sessions...) : à l'ouverture le panneau de droite est entièrement vide, aucun bloc "
                "n'est surligné, alors que l'en-tête de la palette dit « " + res[-1][3] + " ». Attendu : un bloc "
                "choisi (la source, ou le dernier bloc) et son panneau, ou un panneau qui dit quoi faire.",
                shot="panneau_vide_ouverture", steps=["ouvrir un projet Image seule"])


@case
def session_orphan():
    """Two flows (or mono, each session its own workflow): a Session placed by a double-click belongs to none."""
    res = {}
    for attempt in (1, 2):
        for kind in ("two_flows", "mono"):
            fresh(kind)
            ex.double_click_card("session")
            s = ex.s
            new = max(x.id for x in s.p.sessions)
            drawn = [it for it in ex.blocks() if it.node.sid == new and it.node.key in ("session", "sources")]
            res[kind] = (new, [(q.name, s.sessions_of(q)) for q in s.paths], len(drawn), s.sel_step, m.heading())
            if attempt == 2:
                ex.shot(f"session_invisible_{kind}")
        print("orphan", attempt, res, flush=True)
    r = res["two_flows"]
    if r[2] == 0:
        note("gênant", "Deux flux : une Session posée par double-clic n'apparaît nulle part dans le graphe",
             "Projet à deux flux parallèles (Session 1 → « Workflow », Session 2 → « Flow »). Double-clic sur la carte "
             f"Session : la Session {r[0]} est créée et son panneau s'ouvre à droite, mais aucun bloc ne la montre dans "
             f"le graphe ; elle n'appartient à aucun workflow ({r[1]}). Même chose dans un projet mono (Ha, OIII : "
             f"{res['mono'][1]}, bloc dessiné : {res['mono'][2]}) et dans un projet fait avec le modèle SHO "
             "(la matrice : 60 poses de Session sur 70 dans ce cas). Le dépôt sur la zone du bas, lui, fait bien un "
             "flux. Attendu (PRINCIPLES : un bloc se pose et se relie "
             "seul) : la Session rejoint le flux choisi (en-tête « " + r[4] + " ») ou démarre un flux à elle, et se "
             "voit.", shot="session_invisible_two_flows",
             steps=["projet à deux flux", "double-clic carte Session"])


@case
def composition_stale_redo():
    res = []
    for attempt in (1, 2):
        fresh("legacy")
        ex.double_click_card("gradient")
        n_g = lambda: len([n for n in ex.s.graph(ex.s.paths[0]) if n["block"] == "gradient"])  # noqa: E731
        a = n_g()
        ex.undo()
        b = n_g()
        ex.double_click_card("composition")
        ex.double_click_card("composition")
        c = len(comp_items())
        ex.redo()
        d = n_g()
        e = len(comp_items())
        if attempt == 2:
            ex.shot("combiner_puis_ctrly")
        res.append((a, b, c, d, e))
        print("stale_redo", attempt, res[-1], flush=True)
    a, b, c, d, e = res[-1]
    if d > b:
        note("gênant", "Après la pose d'un bloc « Combiner », Ctrl+Y refait une modification plus ancienne",
             "Workflow complet : double-clic Gradient (Gradient 2 posé), Ctrl+Z (retiré), puis deux double-clics sur la "
             f"carte « Combiner les calques étoiles et fond » ({c} blocs Combiner). Ctrl+Y : Gradient 2 revient "
             f"({b} → {d} blocs Gradient) alors que la dernière action était la pose de Combiner ({e} blocs Combiner "
             "après). Attendu (PRINCIPLES : Ctrl+Z / Ctrl+Y défont et refont exactement) : après une nouvelle action, "
             "plus rien à refaire.", shot="combiner_puis_ctrly",
             steps=["projet workflow complet", "double-clic Gradient", "Ctrl+Z", "double-clic Combiner ×2", "Ctrl+Y"])


@case
def drag_zoom():
    res = []
    for attempt in (1, 2):
        fresh("legacy")
        k0 = ex.view.transform().m11()
        if attempt == 2:
            ex.shot("legacy_avant_glisser")
        vp = ex.view.viewport()
        ex.drag_palette_hover("gradient", QPoint(vp.width() // 2, 4))
        k1 = ex.view.transform().m11()
        n = len(ex.ghosts())
        if attempt == 2:
            ex.shot("legacy_pendant_glisser")
        ex.drag_leave()
        res.append((round(k0, 3), round(k1, 3), n))
        print("zoom", attempt, res[-1], flush=True)
    k0, k1, n = res[-1]
    if k1 < k0 * 0.9:
        note("mineur", "Pendant le glisser d'un bloc, le graphe rapetisse : les textes deviennent illisibles",
             f"Workflow complet, carte Gradient tenue au-dessus du graphe : {n} places s'ouvrent (une colonne pour "
             f"chacune) et le graphe est réduit pour tout montrer (échelle {k0} → {k1}). Les titres des blocs et des "
             "places font alors quelques pixels (voir les deux captures : avant / pendant). Attendu (PRINCIPLES : "
             "lisible en 1366×768 et 1920×1080) : des places lisibles pendant le glisser, quitte à faire défiler.",
             shot="legacy_pendant_glisser", steps=["projet workflow complet", "glisser la carte Gradient sans lâcher"])


def hover_card(block):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QEnterEvent
    from PySide6.QtWidgets import QApplication
    c = ex.card(block)
    p = QPointF(c.width() / 2, c.height() / 2)
    QApplication.sendEvent(c, QEnterEvent(p, c.mapToGlobal(p), c.mapToGlobal(p)))
    ex.settle(2)
    return ex.page.hint.text() if ex.page.hint.isVisible() else None


@case
def reasons():
    got = {}
    for attempt in (1, 2):
        fresh("new_image")                                 # Image → Astrometry → Gradient → Color
        got["stars"] = (ex.card("stars").reason, ex.card("denoise").reason, hover_card("stars"))
        if attempt == 2:
            ex.shot("raison_split_impasse")
        fresh("session_empty_wf")
        got["astro"] = (ex.card("astrometry").reason, hover_card("astrometry"))
        if attempt == 2:
            ex.shot("raison_astrometrie_poses")
        fresh("image")
        got["fusion"] = (ex.card("fusion").reason, hover_card("fusion"))
        if attempt == 2:
            ex.shot("raison_fusion")
        print("reasons", attempt, got, flush=True)
    st = got["stars"]
    if st[0] and "Débruitage" in st[0] and st[1]:
        ex.note("gênant", "Workflow Image : Split étoiles et fond renvoie vers Débruitage, qui ne peut pas y être posé",
                "Workflow « Traiter une image empilée » (Image → Astrométrie → Gradient → Couleur). Carte « Split "
                f"étoiles et fond » grisée, message : « {st[2]} ». Mais la carte Débruitage est elle-même grisée : "
                f"« {st[1]} ». Le message envoie donc vers une impasse : Split étoiles et fond ne peut jamais être "
                "posé sur une image empilée. Attendu (PRINCIPLES : un bloc grisé dit pourquoi, et la raison est "
                "juste) : soit Split possible sur une image, soit une raison qui dit que ce n'est pas possible ici.",
                shot="raison_split_impasse", steps=["nouveau projet, modèle « Traiter une image empilée »",
                                                    "survoler la carte Split étoiles et fond"])
    a = got["astro"]
    if a[0]:
        ex.note("mineur", "Raison de la carte Astrométrie (workflow de poses sans Empilement) : ne dit pas quoi faire",
                f"Workflow de poses encore vide : Astrométrie, Gradient, Débruitage, Couleur, Split sont grisés avec « "
                f"{a[1]} ». C'est juste mais abstrait (« la source donne une série de poses ») ; ailleurs la palette "
                "dit quoi poser avant (« il lui faut le bloc Astrométrie avant lui »). Attendu (PRINCIPLES : "
                "dépendances lointaines refusées avec la raison, en mots simples) : « il lui faut un Empilement avant "
                "lui ».", shot="raison_astrometrie_poses",
                steps=["projet Session avec un workflow vide", "survoler la carte Astrométrie"])
    f = got["fusion"]
    if f[0]:
        ex.note("mineur", "Raison de la carte Fusion grisée : une description, pas le pourquoi",
                f"Projet avec une seule Image : Fusion grisée, message « {f[1]} ». Le message décrit ce que fait "
                "Fusion mais ne dit pas ce qui manque ici. Attendu : « il faut au moins deux workflows de filtres, ou "
                "un workflow couleur à partir de poses ».", shot="raison_fusion",
                steps=["projet Image seule", "survoler la carte Fusion"])


for fn in DONE:
    ex.guard(fn.__name__, fn)
print("missing:", sorted(i18n.missing())[:20])
ex.close()
