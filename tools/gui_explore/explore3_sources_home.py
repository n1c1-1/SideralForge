"""sources_and_files, part 1: Home and new projects (every template of the start window, from Home's button and the
File menu), Import folder as new project, Open project, recent projects."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore3_sources_common import (  # noqa: E402
    TMP, Explorer, Notes, _, button, graph_texts, guard, look, missing_fr, start, texts, wait,
)
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

start(1500)
ex = Explorer("sources_files")
N = Notes(ex)
w = ex.open()
wait(ex, 1.0)
from astro_gpu.client.gallery import GALLERY, TABS  # noqa: E402

LOG: list[str] = []


def log(*a):
    t = " ".join(str(x) for x in a)
    LOG.append(t)
    print(t, flush=True)


# ------------------------------------------------------------------ Home
home = w.home
log("home texts:", texts(home)[:20])
look(ex, N, "Accueil", home, steps=["Ouvrir l'application (Accueil)"])
ex.shot("home_start")


def home_button(part):
    return button(home, part, QPushButton)


def new_project_handler(name, target="", then=None, location=None, capture=None):
    """The New project dialog: its name and target typed, Create; ``then``: the handler of the next modal."""
    def h(d):
        info = {"title": d.windowTitle(), "location": d.location.as_posix(), "name0": d.name.text(),
                "target0": d.target.text(), "texts": texts(d)}
        if capture is not None:
            capture.update(info)
        if location:
            d.location = Path(location)
        d.name.setText(name)
        d.target.setText(target)
        d._check_target()
        d._update()
        if capture is not None:
            capture["ok_enabled"] = d.ok.isEnabled()
            capture["exists_shown"] = d.exists.isVisible()
            capture["found"] = d.found.text()
            ex.shot(f"newproject_dialog_{_slug(name)}", d)
        ex.dialog_handler = then
        if d.ok.isEnabled():
            d.ok.click()
        else:
            d.reject()
    return h


def _slug(t):
    return "".join(c if c.isalnum() else "_" for c in t.lower())[:40]


def gallery_handler(tab_key, key, image="", capture=None, shot=None, name=None):
    """The template gallery: the tab, the card, its image file, then Create."""
    def h(d):
        info = {"class": d.__class__.__name__, "title": d.windowTitle()}
        if d.__class__.__name__ != "TemplateGallery":
            info["unexpected"] = True
            if capture is not None:
                capture.update(info)
            d.reject()
            return
        info["tabs"] = [d.tabs.tabText(i) for i in range(d.tabs.count())]
        info["tab_keys"] = list(d.tab_keys)
        if tab_key in d.tab_keys:
            d.tabs.setCurrentIndex(d.tab_keys.index(tab_key))
            ex.settle(2)
        card = next((c for c in d.cards if c.t.get("key") == key), None)
        info["cards"] = [c.t.get("key") for c in d.cards]
        if card is None:
            info["no_card"] = True
            if capture is not None:
                capture.update(info)
            d.reject()
            return
        ex.click(card)
        if image:
            d.file.setText(image)
        if name is not None:
            d.name.setText(name)
        ex.settle(2)
        info.update(ok=d.ok.isEnabled(), duration=d.duration.text(), text=d.text.text(), name=d.name.text(),
                    file_visible=d.file_row.isVisible(), placeholder=d.file.placeholderText(),
                    texts=texts(d))
        if shot:
            ex.shot(shot, d)
        if capture is not None:
            capture.update(info)
        if d.ok.isEnabled():
            d.ok.click()
        else:
            d.reject()
    return h


def project_state():
    s = ex.s
    if s is None:
        return {}
    return {"paths": [(q.name, q.source, q.sessions, [n.get("block") for n in (s.p.workflow(q.pid).blocks or [])])
                      for q in s.paths],
            "sessions": [(x.id, x.name, x.filter, list(x.lights)) for x in s.p.sessions],
            "blocks": graph_texts(ex)}


def close_project():
    if ex.w.project is not None:
        ex.w.close_project()
        wait(ex, 0.3)


def modal_count():
    return len(ex.modals)


# ------------------------------------------------------------------ every template, from Home's button
before_missing = missing_fr()
fr_image = ex.image_file("stack.fits")
png = ex.image_file("finished.png")
results = {}
for t in GALLERY:
    key, tab = t["key"], t["tab"]
    name = f"T {t['name']}"[:40].replace("/", "_")
    cap1, cap2 = {}, {}
    image = ""
    if t.get("source") == "image":
        image = str(png if t.get("stretched") else fr_image)
    n_modals = modal_count()
    n_pop = len(ex.popups)
    ex.dialog_handler = new_project_handler(name, "M 42", gallery_handler(tab, key, image, cap2,
                                                                         shot=f"gallery_{key}"), capture=cap1)
    b = home_button(_("New project…"))
    if b is None:
        ex.show_home() if hasattr(ex, "show_home") else ex.w.show_home()
        wait(ex, 0.3)
        b = home_button(_("New project…"))
    guard(ex, N, f"Nouveau projet ({key})", ex.click, b)
    wait(ex, 1.0)
    ex.dialog_handler = None
    st = project_state()
    modals = ex.modals[n_modals:]
    pops = ex.popups[n_pop:]
    root = ex.ws / name
    results[key] = {"cap1": {k: v for k, v in cap1.items() if k != "texts"}, "cap2": {k: v for k, v in cap2.items()
                                                                                      if k != "texts"},
                    "modals": modals, "popups": pops, "state": st, "exists": (root / "project.astroproj").exists(),
                    "dialogs_left": [d.__class__.__name__ for d in ex.dialogs()]}
    log("TEMPLATE", key, results[key])
    ex.shot(f"created_{key}")
    steps = ["Accueil", "Nouveau projet…", f"nom « {name} », cible M 42, Créer",
             f"galerie : onglet {_(dict(TABS)[tab])}, modèle « {_(t['name'])} »", "Créer"]
    if not results[key]["exists"]:
        N.note("bloquant", f"Projet non créé avec le modèle {_(t['name'])}", str(results[key]), steps=steps)
    if not cap2:
        N.note("gênant", f"Fenêtre des modèles absente après Nouveau projet ({_(t['name'])})",
               f"Fenêtres modales vues : {modals}", steps=steps)
    elif cap2.get("ok") is False:
        N.note("gênant", f"Modèle « {_(t['name'])} » : Créer grisé dans la galerie d'un projet neuf",
               f"Texte sous le modèle : {cap2.get('duration')!r}; image : {image!r}", steps=steps)
    if pops:
        N.note("gênant", f"Message bloquant à la création d'un projet ({_(t['name'])})", str(pops), steps=steps)
    if results[key]["dialogs_left"]:
        N.note("mineur", f"Fenêtre restée ouverte après création ({_(t['name'])})",
               str(results[key]["dialogs_left"]), steps=steps)
    if ex.w.project is not None and key == "image_stretched":
        src = ex.s.paths[0].source if ex.s.paths else {}
        blocks = [n.get("block") for n in (ex.s.p.workflow(ex.s.paths[0].pid).blocks or [])] if ex.s.paths else []
        if not (src or {}).get("stretched") or any(b in blocks for b in ("astrometry", "gradient", "color")):
            N.note("bloquant", "Modèle « Image déjà étirée » : crée une chaîne linéaire, l'image marquée Linéaire",
                   f"Le modèle dit que l'image « va au Post telle quelle » ; le workflow créé a la source {src} et les "
                   f"blocs {blocks} (Astrométrie, Gradient, Couleur sur une image déjà étirée). L'aperçu du modèle "
                   "dans la galerie montre déjà cette chaîne.", steps=steps)
    if ex.w.project is not None and tab == "frames" and not t.get("blank"):
        keys = [k for k, _t, _s in st["blocks"]]
        first = next(((k, tt, ss) for k, tt, ss in st["blocks"] if k == "sources"), None)
        if "session" not in keys and first is not None:
            N.note("gênant", "Modèle à partir de brutes : le premier bloc s'appelle « Images » et non Session",
                   f"Après le modèle « {_(t['name'])} », la chaîne commence par un bloc pointillé « {first[1]} · "
                   f"{first[2]} », pas par un bloc Session comme la palette et les autres modèles (Filtres). Son "
                   "panneau dit « Images » et propose « Importer un dossier… ».", steps=steps)
        if key == "nights" and len(st["sessions"]) < 2:
            N.note("gênant", "Modèle « Plusieurs nuits » : même graphe que « Complet », aucune deuxième nuit",
                   f"L'aperçu montre deux nuits, chacune avec sa Session et sa Calibration ; le projet créé n'a "
                   f"que {len(st['sessions'])} session(s) et la même chaîne que « Complet » : {keys}.", steps=steps)
    if key == "mono_each" and "revenez" in (cap2.get("duration") or ""):
        N.note("gênant", "Modèle « Un workflow par filtre » inutilisable dans un projet neuf : « puis revenez »",
               f"Créer est grisé ; la galerie dit : {cap2.get('duration')!r}. Or la fenêtre des modèles n'apparaît "
               "qu'à la création du projet : on ne peut pas « revenir ».", steps=steps)
    if ex.w.project is not None:
        look(ex, N, f"panneau après création ({_(t['name'])})", steps=steps)
        # what was made: a workflow for a template, nothing for Blank
        n_paths = len(ex.s.paths)
        if t.get("blank") and n_paths:
            N.note("question", "Modèle Vierge : un workflow est créé", str(st["paths"]), steps=steps)
        if not t.get("blank") and cap2.get("ok") and not n_paths:
            N.note("gênant", f"Modèle « {_(t['name'])} » : aucun workflow créé", str(st), steps=steps)
    close_project()

# the texts of the gallery: in French
missing = missing_fr() - before_missing
log("MISSING after gallery:", sorted(missing))
if missing:
    N.note("mineur", "Galerie des modèles : textes restés en anglais",
           "Textes sans traduction française affichés par la galerie ou le graphe créé : " +
           " | ".join(sorted(missing))[:1500], shot=False,
           steps=["Nouveau projet…", "galerie : parcourir chaque onglet et chaque modèle"])

# ------------------------------------------------------------------ the File menu's New project (Ctrl+N), Blank
cap1, cap2 = {}, {}
ex.dialog_handler = new_project_handler("Menu blank", "", gallery_handler("frames", "blank", capture=cap2), capture=cap1)
act = next(a for m in ex.w.menuBar().actions() for a in m.menu().actions() if a.text() == _("New project…"))
guard(ex, N, "Fichier > Nouveau projet", act.trigger)
wait(ex, 1.0)
log("MENU NEW", cap1.get("location"), cap2.get("tabs"), project_state())
ex.shot("menu_new_blank")
if ex.w.project is not None:
    # Blank: the empty page says how to start; the palette is there
    gt = graph_texts(ex)
    scene_txt = [getattr(it, "text", "") for it in ex.view.scene().items() if isinstance(getattr(it, "text", None), str)]
    log("BLANK scene texts", scene_txt, gt)
close_project()

# Ctrl+N from a project
from PySide6.QtCore import Qt  # noqa: E402
ex.w.open_project(ex.ws / "T Full")
wait(ex, 0.8)
ex.w.activateWindow()
ex.dialog_handler = new_project_handler("CtrlN", "", gallery_handler("mono", "hoo", capture=cap2))
n_modals = modal_count()
ex.key(Qt.Key_N, ex.w, Qt.ControlModifier)
wait(ex, 1.0)
log("CTRL+N", ex.modals[n_modals:], project_state())
if not ex.modals[n_modals:]:
    log("Ctrl+N did not reach the action offscreen (harness): the menu action is used")
    act = next(a for m in ex.w.menuBar().actions() for a in m.menu().actions() if a.text() == _("New project…"))
    act.trigger()
    wait(ex, 1.0)
    log("CTRL+N via action", ex.modals[n_modals:], project_state())
ex.dialog_handler = None
ex.shot("ctrl_n_hoo")
close_project()

# a name with characters Windows refuses, an existing name, an empty name
for nm in ("M31: «test» / *?", "T Full", "   "):
    cap = {}
    ex.dialog_handler = new_project_handler(nm, "", None, capture=cap)
    ex.click(home_button(_("New project…")))
    wait(ex, 0.6)
    log("NAME", repr(nm), {k: v for k, v in cap.items() if k != "texts"}, "project:",
        ex.w.project.root if ex.w.project else None)
    if ex.w.project is not None:
        close_project()
ex.close_dialogs()

# ------------------------------------------------------------------ Import folder as new project
# real FITS frames (the fake raw files are never decoded: the import reads the headers)
lights = ex.fits_frames("M81 2026-03-01/lights", 8)
ex.fits_frames("M81 2026-03-01/darks", 4)
ex.fits_frames("M81 2026-03-01/flats", 3)
imp = {}
cap1 = {}
gal = {}
from explore3_sources_common import Dispatch  # noqa: E402


def import_handler(d):
    imp["title"] = d.windowTitle()
    try:
        d.finish_now()
    except Exception as e:  # noqa: BLE001
        imp["wait_error"] = str(e)
    ex.settle(10)
    imp.update(found=d.found.text(), groups=d.groups.text(), warn=d.warn.text(), cur=d.cur.text(), ok=d.ok.isEnabled(),
               rows={k: (r.what.text(), r.detail.text()) for k, r in d.rows.items()}, texts=texts(d))
    ex.shot("import_new_dialog", d)
    if d.ok.isEnabled():
        d.ok.click()
    else:
        d.reject()


def new_from_folder_handler(d):
    """The New project dialog of the import: its proposed name kept."""
    cap1.update(title=d.windowTitle(), location=d.location.as_posix(), name0=d.name.text(), target0=d.target.text(),
                found=d.found.text())
    ex.shot("import_new_project_dialog", d)
    if d.ok.isEnabled():
        d.ok.click()
    else:
        cap1["ok_disabled"] = True
        d.reject()


def gallery_pick_full(d):
    gal["seen"] = True
    gal["tabs"] = list(d.tab_keys)
    ex.shot("import_new_gallery", d)
    t = next((c.t for c in d.cards if c.t.get("key") == "full"), None)
    if t is not None:
        d.pick(t)
    d.ok.click()


n_modals = modal_count()
ex.file_queue = [str(lights)]
disp = Dispatch(ex, NewProjectDialog=new_from_folder_handler, ImportDialog=import_handler,
                TemplateGallery=gallery_pick_full)
b = home_button(_("Import folder as new project…"))
guard(ex, N, "Importer un dossier comme nouveau projet", ex.click, b)
wait(ex, 2.5)
disp.stop()
log("IMPORT NEW", cap1, {k: v for k, v in imp.items() if k != "texts"}, gal, disp.seen, ex.popups[-3:])
ex.shot("import_new_result")
st = project_state()
log("IMPORT NEW state", st)
steps = ["Accueil", "Importer un dossier comme nouveau projet…", f"dossier {lights.name} (8 FITS, darks/ et flats/ à côté)",
         "Nouveau projet : garder le nom proposé, Créer", "Fenêtre d'import : Ajouter au projet"]
if "de offsets" in (imp.get("found") or "") or "Pas de offsets, darks de flat" in (imp.get("found") or ""):
    N.note("mineur", "Fenêtre d'import : « Pas de offsets, darks de flat. » (français fautif)",
           f"Texte : {imp.get('found')!r}. Attendu par exemple « Ni offsets ni darks de flat. » ; les lignes vides "
           "disent « Aucune » (Darks, Offsets…).", steps=["Accueil", "Importer un dossier comme nouveau projet…",
                                                       "un dossier de brutes FITS avec darks/ et flats/ à côté"])
both = [x for x in disp.seen if x[2]]
if both:
    N.note("gênant", "Importer un dossier comme nouveau projet : la galerie des modèles et la fenêtre d'import s'ouvrent ensemble",
           f"Fenêtres ouvertes l'une sur l'autre : {both}. L'ordre vu : {[x[0] for x in disp.seen]}.", steps=steps)
if cap1.get("location") and Path(cap1["location"]).resolve() == lights.resolve():
    N.note("gênant", "Importer un dossier : le projet est proposé à l'intérieur du dossier des brutes",
           f"L'emplacement proposé par « Nouveau projet » est le dossier des brutes lui-même ({cap1['location']}) : "
           "le projet serait créé dans le dossier de prise de vue.", steps=steps)
if ex.w.project is not None:
    proj_root = ex.w.project.root
    if str(proj_root).lower().startswith(str(lights).lower()):
        N.note("gênant", "Le projet importé est créé dans le dossier des brutes",
               f"Projet : {proj_root}; brutes : {lights}", steps=steps)
    look(ex, N, "après Importer un dossier comme nouveau projet", steps=steps)
    ex.shot("import_new_graph")
    log("IMPORT NEW graph", graph_texts(ex), ex.w.mode, [d.__class__.__name__ for d in ex.dialogs()])
    if not st.get("sessions"):
        N.note("bloquant", "Importer un dossier comme nouveau projet : aucune session dans le projet créé",
               f"État : {st}", steps=steps)
    # the same folder imported again (File > Import folder…)
    n_pop = len(ex.popups)
    ex.file_queue = [str(lights)]
    ex.w.add_sources()
    wait(ex, 0.8)
    log("IMPORT again", ex.popups[n_pop:], [d.__class__.__name__ for d in ex.dialogs()])
    if ex.popups[n_pop:]:
        N.note("mineur", "Importer de nouveau le même dossier : fenêtre de message bloquante",
               f"Message : {ex.popups[n_pop:]}. Un simple message dans la barre d'état suffirait.",
               shot=False, steps=steps + ["Fichier > Importer un dossier…", "le même dossier"])
close_project()

# Import folder as new project, cancelled at each step
ex.file_queue = [""]
ex.click(home_button(_("Import folder as new project…")))
wait(ex, 0.3)
log("IMPORT cancel folder: project", ex.w.project, "popups", ex.popups[-2:])
ex.file_queue = [str(lights)]
ex.dialog_handler = None                       # New project dialog cancelled
ex.click(home_button(_("Import folder as new project…")))
wait(ex, 0.3)
log("IMPORT cancel name: project", ex.w.project)
# a folder without frames
empty = TMP / "data" / "empty_folder"
empty.mkdir(parents=True, exist_ok=True)
imp.clear()
ex.file_queue = [str(empty)]
cap1.clear()
disp = Dispatch(ex, NewProjectDialog=new_from_folder_handler, ImportDialog=import_handler,
                TemplateGallery=lambda d: d.reject())
ex.click(home_button(_("Import folder as new project…")))
wait(ex, 1.5)
disp.stop()
log("IMPORT empty", cap1, {k: v for k, v in imp.items() if k != "texts"}, "project:",
    ex.w.project.root if ex.w.project else None, project_state())
ex.shot("import_empty_folder")
if ex.w.project is not None and imp.get("ok") is False:
    st = project_state()
    N.note("question", "Importer un dossier vide : le projet est créé quand même, sans images",
           f"Le dossier n'a aucune image ; l'import dit {imp.get('warn')!r} ; le projet reste créé : {st}",
           steps=["Accueil", "Importer un dossier comme nouveau projet…", "un dossier vide", "Créer",
                  "la fenêtre d'import : Ajouter grisé, Annuler"])
close_project()

# a folder of phone pictures (JPEG) imported as a new project
jpgs = TMP / "data" / "phone_pics"
jpgs.mkdir(parents=True, exist_ok=True)
from PIL import Image as _Im  # noqa: E402
import numpy as _np  # noqa: E402
for i in range(4):
    _Im.fromarray(_np.zeros((20, 30, 3), _np.uint8)).save(jpgs / f"IMG_{i:04d}.JPG")
imp.clear()
cap1.clear()
ex.file_queue = [str(jpgs)]
disp = Dispatch(ex, NewProjectDialog=new_from_folder_handler, ImportDialog=import_handler,
                TemplateGallery=lambda d: d.reject())
ex.click(home_button(_("Import folder as new project…")))
wait(ex, 1.5)
disp.stop()
log("IMPORT jpgs", {k: v for k, v in imp.items() if k != "texts"}, project_state())
ex.shot("import_jpg_folder")
if imp and not imp.get("ok"):
    N.note("mineur", "Importer un dossier de JPEG : « Aucune image brute choisie » alors que le dossier est choisi",
           f"La fenêtre d'import dit {imp.get('warn')!r} / {imp.get('cur')!r} sans dire que les 4 fichiers JPEG ne "
           f"sont pas des poses lues (RAW ou FITS). Lignes : {imp.get('rows')}",
           steps=["Accueil", "Importer un dossier comme nouveau projet…", "un dossier de 4 JPEG", "Créer"])
close_project()

# ------------------------------------------------------------------ Open project, recent projects
some = ex.ws / "T Full"
ex.file_queue = [str(some / "project.astroproj")]
ex.click(home_button(_("Open project…")))
wait(ex, 1.0)
log("OPEN", ex.w.project.root if ex.w.project else None, "dialogs", [d.__class__.__name__ for d in ex.dialogs()],
    ex.modals[-2:])
if ex.w.project is not None:
    if any(d.__class__.__name__ == "TemplateGallery" for d in ex.dialogs()):
        N.note("gênant", "Ouvrir un projet existant ouvre la fenêtre des modèles", "", steps=["Ouvrir un projet…"])
    ex.shot("open_existing")
close_project()
# a file that is not a project, a folder without project
bad = TMP / "data" / "notaproject" / "project.astroproj"
bad.parent.mkdir(parents=True, exist_ok=True)
bad.write_text("{ not json", encoding="utf-8")
n_pop = len(ex.popups)
ex.file_queue = [str(bad)]
ex.click(home_button(_("Open project…")))
wait(ex, 0.6)
log("OPEN bad", ex.popups[n_pop:], ex.w.project)
for k, title, text in ex.popups[n_pop:]:
    if any(x in text for x in ("JSONDecodeError", "Expecting", "line 1 column", "Errno", "Traceback")):
        N.note("mineur", "Ouvrir un projet illisible : erreur Python brute dans le message",
               f"{title} : {text}", shot=False, steps=["Accueil", "Ouvrir un projet…", "un project.astroproj abîmé"])
# the File menu's recent projects
fm = ex.w.menuBar().actions()[0].menu()
rec = next(a for a in fm.actions() if a.menu() is not None)
items = [(a.text(), a.toolTip()) for a in rec.menu().actions()]
log("RECENT", items)
# the Home cards
ex.w.show_home()
wait(ex, 1.0)
cards = home.all_cards()
log("HOME cards", [(c.info.name, c.info.root) for c in cards][:30])
ex.shot("home_after")
look(ex, N, "Accueil après création de projets", home, steps=["Créer plusieurs projets", "Revenir à l'Accueil"])
# a recent project whose folder went away (renamed, not deleted)
gone = ex.ws / "T HOO"
moved = ex.ws / "T HOO moved"
try:
    gone.rename(moved)
except OSError as e:
    log("rename failed", e)
ex.w.build_menus()
fm = ex.w.menuBar().actions()[0].menu()
rec = next(a for a in fm.actions() if a.menu() is not None)
n_pop = len(ex.popups)
a = next((x for x in rec.menu().actions() if x.toolTip() and Path(x.toolTip()) == gone), None)
log("RECENT gone action", a.text() if a is not None else None)
if a is not None:
    a.trigger()
    wait(ex, 0.6)
log("RECENT gone", ex.popups[n_pop:])
ex.w.build_menus()
fm = ex.w.menuBar().actions()[0].menu()
rec = next(a for a in fm.actions() if a.menu() is not None)
log("RECENT after gone", [x.text() for x in rec.menu().actions()])
ex.w.show_home()
wait(ex, 1.0)
cards = home.all_cards()
still = [c for c in cards if Path(c.info.root) == gone]
log("HOME after gone", len(still))
if still:
    ex.shot("home_gone_card")
try:
    moved.rename(gone)
except OSError:
    pass
# a click on a card opens its project
c = next((c for c in home.all_cards() if Path(c.info.root) == ex.ws / "T Full"), None)
if c is not None:
    ex.click(c)
    wait(ex, 1.0)
    log("CARD click ->", ex.w.project.root if ex.w.project else None)
    close_project()

# ------------------------------------------------------------------ my templates: save, then the gallery's tab, export, import
ex.w.open_project(ex.ws / "T Full")
wait(ex, 1.0)
ex.w.activateWindow()
out_b = next((b for b in ex.blocks() if b.node.key == "post" or b.node.handle), None)
tdlg = {}


def template_dlg(d):
    from PySide6.QtWidgets import QComboBox, QLineEdit
    tdlg["title"] = d.windowTitle()
    tdlg["texts"] = texts(d)
    ex.shot("tmpl_save_dialog", d)
    eds = d.findChildren(QLineEdit)
    if eds:
        eds[0].setText("Mon complet")
    d.accept()


items = []
if out_b is not None:
    ex.dialog_handler = template_dlg
    items = ex.right_click_block(out_b, _("Save as template…"))
    wait(ex, 0.8)
log("TMPL save", [x.get("text") for x in items], {k: v for k, v in tdlg.items() if k != "texts"})
close_project()
gal_t = {}


def gallery_mine(d):
    gal_t["tabs"] = [d.tabs.tabText(i) for i in range(d.tabs.count())]
    if "mine" in d.tab_keys:
        d.tabs.setCurrentIndex(d.tab_keys.index("mine"))
        ex.settle(3)
        gal_t["cards"] = [c.t.get("name") for c in d.cards]
        mine = next((c.t for c in d.cards if c.t.get("user")), None)
        if mine is not None:
            d.pick(mine)
            ex.settle(2)
            gal_t["buttons"] = [(b.text(), b.isVisible()) for b in d.findChildren(QPushButton)]
            ex.shot("tmpl_gallery_mine", d)
            # export it
            exp = TMP / "data" / "mon_complet.sftemplate"
            ex.file_queue = [str(exp)]
            d.export_b.click()
            ex.settle(3)
            gal_t["exported"] = exp.exists()
            # import a file that is not a template
            bad = TMP / "data" / "bad.sftemplate"
            bad.write_text('{"format": "other"}', encoding="utf-8")
            ex.file_queue = [str(bad)]
            next(b for b in d.findChildren(QPushButton) if b.text() == _("Import…")).click()
            ex.settle(3)
            gal_t["bad_import"] = d.duration.text()
            # import the good one back
            ex.file_queue = [str(exp)]
            next(b for b in d.findChildren(QPushButton) if b.text() == _("Import…")).click()
            ex.settle(3)
            gal_t["good_import_cards"] = [c.t.get("name") for c in d.cards]
            d.pick(mine)
            ex.settle(2)
    # an image template without its file: why is Create greyed?
    if "image" in d.tab_keys:
        d.tabs.setCurrentIndex(d.tab_keys.index("image"))
        ex.settle(3)
        t = next((c.t for c in d.cards if c.t.get("key") == "image"), None)
        if t is not None:
            d.pick(t)
            d.file.setText("")
            ex.settle(2)
            gal_t["image_nofile"] = (d.ok.isEnabled(), d.duration.text(), d.file.placeholderText())
            d.file.setText(str(TMP / "data" / "nowhere.fits"))
            ex.settle(2)
            gal_t["image_missing"] = (d.ok.isEnabled(), d.duration.text())
            ex.shot("tmpl_gallery_image_missing_file", d)
    d.reject()


disp = Dispatch(ex, NewProjectDialog=lambda d: (d.name.setText("Tmpl test"), d.ok.click()), TemplateGallery=gallery_mine)
ex.click(home_button(_("New project…")))
wait(ex, 1.5)
disp.stop()
log("TMPL gallery", gal_t)
steps = ["projet « Complet »", "clic droit sur la Sortie > Enregistrer comme modèle… « Mon complet »", "Nouveau projet…",
         "galerie : onglet Mes modèles"]
if "cards" in gal_t and "Mon complet" not in gal_t["cards"]:
    N.note("gênant", "Modèle enregistré absent de l'onglet « Mes modèles »", str(gal_t), steps=steps)
if not gal_t.get("cards") and "tabs" in gal_t:
    N.note("gênant", "Enregistrer comme modèle : pas d'onglet « Mes modèles » ensuite", str(gal_t), steps=steps)
if gal_t.get("exported") is False:
    N.note("gênant", "Exporter un modèle : aucun fichier écrit", str(gal_t), steps=steps + ["Exporter…"])
nf = gal_t.get("image_nofile")
if nf and not nf[0] and not nf[1]:
    N.note("mineur", "Galerie : modèle à partir d'une image sans fichier, Créer grisé sans dire pourquoi",
           f"Créer grisé ; texte sous le modèle vide ; seul le champ dit {nf[2]!r}.",
           steps=["Nouveau projet…", "onglet À partir d'une image", "Traiter une image empilée", "champ Image vide"])
mf = gal_t.get("image_missing")
if mf and not mf[0] and not mf[1]:
    N.note("mineur", "Galerie : fichier image introuvable, Créer grisé sans dire pourquoi",
           f"Chemin d'un fichier absent tapé : Créer grisé, rien ne dit que le fichier n'existe pas.",
           steps=["Nouveau projet…", "onglet À partir d'une image", "taper un chemin inexistant"])
close_project()

missing = missing_fr() - before_missing
log("MISSING total:", sorted(missing))
(ex.out / "home_log.txt").write_text("\n".join(LOG), encoding="utf-8")
N.exceptions(["voir le journal du script explore_sources_home.py"])
ex.close()
print("DONE", TMP)
