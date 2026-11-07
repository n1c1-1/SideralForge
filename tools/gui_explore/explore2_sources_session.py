"""sources_and_files, part 2: the Session block and its panel. Folders and files of every kind (lights, darks, flats,
bias, flat darks), mixtures of types (raw ARW / CR2, FITS mono with filters, FITS colour, JPG, PNG, TIFF), empty
folders, folders without frames, a file twice, a folder in two sessions; the warnings (block, panel, health check);
the groups opened, files taken out one by one, several (Del), a whole folder, the last light; the filter; rename;
"Use the frames of…"; "+ Session"; a session removed (menu, cross, Del) and Ctrl+Z; "Review frames…"; two sessions,
two flows, mono filters, per-session Calibration blocks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from explore2_sources_common import (  # noqa: E402
    TMP, Dispatch, Explorer, Notes, _, button, buttons, graph_texts, guard, look, menu_button, missing_fr, panel,
    ptexts, start, texts, undo, visible, wait,
)
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QListWidget, QPushButton  # noqa: E402

start(2400)
ex = Explorer("sources_files")
N = Notes(ex)
LOG: list[str] = []


def log(*a):
    t = " ".join(str(x) for x in a)
    LOG.append(t)
    print(t, flush=True)


# ------------------------------------------------------------------ data
D = TMP / "data" / "sess"
arw = ex.raws("sess/arw_lights", 15)
cr2 = ex.raws("sess/cr2_lights", 10, ".CR2", start=7000)
darks = ex.raws("sess/darks", 5, start=8000)
flats = ex.raws("sess/flats", 6, start=8100)
bias_cr2 = ex.raws("sess/bias_cr2", 4, ".CR2", start=8200)
fdarks = ex.raws("sess/flat_darks", 3, start=8300)
ha = ex.fits_frames("sess/Ha", 6, "Ha")
oiii = ex.fits_frames("sess/OIII", 6, "OIII")
sii = ex.fits_frames("sess/SII", 5, "SII")
mono_l = ex.fits_frames("sess/L", 4)
empty = D / "empty"
empty.mkdir(parents=True, exist_ok=True)
nofr = D / "no_frames"
nofr.mkdir(parents=True, exist_ok=True)
(nofr / "notes.txt").write_text("not a frame", encoding="utf-8")
(nofr / "readme.md").write_text("x", encoding="utf-8")
misc = D / "misc"
misc.mkdir(parents=True, exist_ok=True)
from PIL import Image  # noqa: E402

Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(misc / "phone.JPG")
Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(misc / "picture.png")
Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(misc / "scan.tif")
from astropy.io import fits  # noqa: E402

colour_fits = D / "colour_fits"
colour_fits.mkdir(parents=True, exist_ok=True)
for i in range(3):
    fits.writeto(colour_fits / f"rgb_{i}.fits", (np.random.default_rng(i).random((3, 40, 60)) * 1000).astype(np.uint16),
                 overwrite=True)
(misc / "notes.txt").write_text("x", encoding="utf-8")


def sel_session(sid=None):
    """Click the Session block (the sources block when there is one session)."""
    it = None
    if sid is not None:
        it = next((b for b in ex.blocks() if b.node.key == "session" and b.node.sid == sid), None)
    if it is None:
        it = ex.block("session") or ex.block("sources")
    if it is None:
        return None
    ex.click_block(it)
    wait(ex, 0.3)
    return it


def group_title(kind_title: str) -> str:
    """The panel's title line of a kind ("Brutes  ·  15 / 15 utilisées")."""
    for t in ptexts(ex):
        if t.startswith(_(kind_title) + "  ·  "):
            return t
    return ""


def add_via_menu(kind_title: str, what: str, answer):
    """The Add ▾ button under a kind: its menu's action (folder / files / another session's), the file dialog
    answered with ``answer``."""
    # the Add buttons in the order of the kinds
    order = ["Lights", "Darks", "Flats", "Bias", "Flat darks"]
    bs = buttons(panel(ex), _("Add") + " ▾", QPushButton)
    b = bs[order.index(kind_title)]
    if answer is not None:
        ex.file_queue = [answer]
    items = menu_button(ex, b, what)
    wait(ex, 1.0)
    return items


def summary():
    it = ex.block("session") or ex.block("sources")
    if it is None:
        return ""
    s_items = [b for b in ex.blocks() if b.node.key in ("session", "sources")]
    return " | ".join(f"{b.title}: {b.summary}" for b in s_items)


def folder_rows():
    """The folder rows of the panel: (button, its text)."""
    return [(b, b.text()) for b in visible(panel(ex), QPushButton) if "📁" in b.text()]


def file_lists():
    return [x for x in visible(panel(ex), QListWidget) if x.__class__.__name__ == "FileList"]


def health_lines():
    """The health check of the frames (Inspect, the Frames block): its lines."""
    from astro_gpu.client.frames import FramesView
    ex.w.set_mode(1)
    wait(ex, 1.0)
    fv = [x for x in ex.w.findChildren(FramesView) if x.isVisible()]
    out = []
    if fv:
        fv[0].show_page(0)
        wait(ex, 1.5)
        out = [t for t in texts(fv[0])]
    ex.shot(f"health_{len(LOG)}")
    ex.w.set_mode(0)
    wait(ex, 0.5)
    return out


# ================================================================== A. one session of 15 raws
root = ex.project("session", "sessA")
w = ex.open(root)
w.activateWindow()
wait(ex, 1.0)
before_missing = missing_fr()
sel_session()
log("A panel", ptexts(ex))
log("A block", summary())
ex.shot("A_session_panel")
look(ex, N, "panneau Session (15 ARW)", steps=["Projet d'une session de 15 ARW", "clic sur le bloc Session"])
N.check("15 / 15" in group_title("Lights"), "gênant", "Compte des brutes faux dans le panneau Session",
        f"Titre : {group_title('Lights')!r}", steps=["clic sur le bloc Session"])
if not button(panel(ex), _("Delete"), QPushButton):
    N.note("mineur", "Panneau Session : pas de bouton « Supprimer le bloc » (les panneaux Image et des autres blocs "
                     "en ont un)", "Le panneau d'une Session n'offre ni Supprimer ni Ajouter une session ; on ne "
           "les trouve que dans le menu du bloc.", steps=["clic sur le bloc Session"])
tip = next((t for t in ptexts(ex) if "monochromes sont prévues" in t), "")
if tip:
    N.note("question", "Infobulle du Filtre : « Les caméras monochromes sont prévues » alors que les modèles Filtres existent",
           f"Infobulle : {tip!r}. Les modèles SHO/HOO/LRVB et la Fusion utilisent déjà le filtre des sessions.",
           steps=["clic sur le bloc Session", "survol du combo Filtre"])

# the menu of the Session block, one session
it = ex.block("sources") or ex.block("session")
items = ex.right_click_block(it)
log("A menu", items)
if not items:
    N.note("gênant", "Clic droit sur une Session seule : aucun menu (ni Renommer, ni Ajouter une session, ni Supprimer)",
           "Le bloc Session d'un projet d'une seule session n'a pas de menu contextuel ; avec deux sessions, chaque "
           "Session a « Renommer la session », « Supprimer le bloc » et « Ajouter une session… ».",
           steps=["projet d'une session", "clic droit sur le bloc Session"])
dele = next((x for x in items if x.get("text", "").startswith(_("Delete"))), None)
if dele is not None and not dele["enabled"] and not dele.get("tip"):
    N.note("mineur", "Menu d'une Session seule : « Supprimer le bloc » grisé sans dire pourquoi",
           f"Menu : {[x.get('text') for x in items]}", steps=["clic droit sur le bloc Session (seule session)"])
if dele is None:
    log("A: no Delete block in the menu of a lone session")
if any("\tDel" in (x.get("text") or "") for x in items):
    N.note("mineur", "Menus en français : raccourci affiché « Del » au lieu de « Suppr »",
           f"Menu : {[x.get('text') for x in items]}", shot=False, steps=["clic droit sur un bloc"])

# ------------------------------------------------------------------ B. darks folder, flats files, bias of another type
add_via_menu("Darks", _("Add folder…"), str(darks))
sel_session()
log("B darks", group_title("Darks"), summary())
N.check(group_title("Darks").endswith("5"), "gênant", "Darks ajoutés par dossier : compte faux",
        f"{group_title('Darks')!r}", steps=["Session > Darks > Ajouter ▾ > Ajouter un dossier…", "dossier de 5 ARW"])
fl = sorted(str(x) for x in flats.iterdir())[:3]
add_via_menu("Flats", _("Add files…"), fl)
sel_session()
log("B flats", group_title("Flats"), summary())
N.check(group_title("Flats").endswith("3"), "gênant", "Flats ajoutés par fichiers : compte faux",
        f"{group_title('Flats')!r}", steps=["Session > Flats > Ajouter ▾ > Ajouter des fichiers…", "3 ARW"])
add_via_menu("Bias", _("Add folder…"), str(bias_cr2))
sel_session()
log("B bias CR2", group_title("Bias"), summary(), [t for t in ptexts(ex) if t.startswith("⚠")])
ex.shot("B_bias_other_type")
warn_panel = [t for t in ptexts(ex) if t.startswith("⚠")]
st = ["Session de 15 ARW", "Offsets > Ajouter ▾ > Ajouter un dossier…", "dossier de 4 CR2"]
N.check(bool(warn_panel), "gênant", "Offsets CR2 avec des brutes ARW : aucun avertissement dans le panneau",
        f"Panneau : {ptexts(ex)}", steps=st)
N.check("⚠" in summary(), "gênant", "Offsets CR2 avec des brutes ARW : pas de ⚠ sur le bloc Session",
        f"Résumé du bloc : {summary()!r}", steps=st)
add_via_menu("Flat darks", _("Add folder…"), str(fdarks))
sel_session()
log("B flat darks", group_title("Flat darks"), summary())
ex.shot("B_all_kinds")
look(ex, N, "panneau Session (tous les types ajoutés)", steps=["Ajouter darks, flats, offsets, darks de flats"])

# ------------------------------------------------------------------ C. a JPG, a PNG, a TIFF among the lights
add_via_menu("Lights", _("Add files…"), [str(misc / "phone.JPG")])
sel_session()
wp = [t for t in ptexts(ex) if t.startswith("⚠")]
log("C jpg", group_title("Lights"), summary(), wp)
ex.shot("C_jpg_in_lights")
st = ["Session de 15 ARW", "Brutes > Ajouter ▾ > Ajouter des fichiers…", "phone.JPG"]
N.check(any("phone.JPG" in t for t in wp), "gênant", "Un JPEG parmi les brutes : pas d'avertissement nommant le fichier",
        f"Avertissements : {wp}", steps=st)
cnt = group_title("Lights")
log("C count", cnt)
if "16 / 16" in cnt:
    N.note("mineur", "Un JPEG non lu compté parmi les brutes « utilisées »",
           f"Titre : {cnt!r} alors que le JPEG n'est pas lu par le moteur (#57) : 16 / 16 utilisées.", steps=st)
add_via_menu("Lights", _("Add files…"), [str(misc / "picture.png"), str(misc / "scan.tif")])
sel_session()
wp = [t for t in ptexts(ex) if t.startswith("⚠")]
log("C png tif", group_title("Lights"), summary(), wp)
ex.shot("C_png_tif_in_lights")
# the file dialog's filter: frames first, all files offered
log("C file asks", ex.file_asks[-3:])
# a text file among the flats (a file of no image type at all)
add_via_menu("Flats", _("Add files…"), [str(misc / "notes.txt")])
sel_session()
wp = [t for t in ptexts(ex) if t.startswith("⚠")]
log("C txt in flats", group_title("Flats"), wp)
ex.shot("C_txt_in_flats")
N.check(any("notes.txt" in t or "TXT" in t for t in wp), "gênant", "Un fichier texte parmi les flats : aucun avertissement",
        f"Avertissements : {wp}; titre {group_title('Flats')!r}",
        steps=["Flats > Ajouter ▾ > Ajouter des fichiers…", "notes.txt (Tous les fichiers)"])
# ------------------------------------------------------------------ D. the same file twice, the same folder twice
n1 = Path(ex.s.p.abspath(ex.s.p.session(1).lights[0]))
n_before = ex.s.p.session(1).lights[:]
toasts = []
ex.page.toasted.connect(lambda t: toasts.append(t))
add_via_menu("Lights", _("Add files…"), [str(misc / "phone.JPG")])
add_via_menu("Lights", _("Add folder…"), str(n1))
add_via_menu("Lights", _("Add folder…"), str(n1).replace("\\", "/").upper())
sel_session()
log("D twice", ex.s.p.session(1).lights == n_before, toasts, group_title("Lights"), ex.w.status.text())
if ex.s.p.session(1).lights == n_before and not toasts:
    N.note("mineur", "Un fichier ou un dossier ajouté deux fois : ignoré sans un mot",
           "Ajouter de nouveau phone.JPG, puis le dossier des brutes déjà présent : rien ne change et rien n'est dit "
           "(pas de message « déjà dans la session »).", steps=["Brutes > Ajouter des fichiers… > phone.JPG (déjà là)",
                                                               "Brutes > Ajouter un dossier… > le dossier déjà là"])
# a new folder, then one of its files added on its own
add_via_menu("Lights", _("Add folder…"), str(arw))
one = sorted(str(x) for x in arw.iterdir())[0]
add_via_menu("Lights", _("Add files…"), [one])
sel_session()
rows = [t for _b, t in folder_rows()]
log("D file inside folder", ex.s.p.session(1).lights, group_title("Lights"), rows)
ex.shot("D_file_of_folder_added_alone")
arw_row = next((t for t in rows if "arw_lights" in t), "")
if arw_row and "15" not in arw_row:
    N.note("gênant", "Un fichier d'un dossier déjà présent, ajouté seul : la ligne du dossier dit « 2 fichiers » et "
                     "n'en montre plus que deux",
           f"Ligne : {arw_row!r} ; le dossier a 15 ARW. Ouverte, elle liste le dossier lui-même comme un fichier "
           "(« arw_lights ») et le fichier ajouté ; les 14 autres brutes n'y sont plus visibles ni retirables une à une. "
           f"Titre : {group_title('Lights')!r}.",
           steps=["Brutes > Ajouter un dossier… > arw_lights (15 ARW)", "Brutes > Ajouter des fichiers… > DSC05850.ARW "
                  "du même dossier", "ouvrir la ligne 📁 arw_lights"])
lf = ex.s.session_files(1, "lights")
if len(lf) != len(set(x.lower() for x in lf)):
    N.note("gênant", "Un fichier d'un dossier déjà présent, ajouté seul : compté deux fois",
           f"Fichiers de la session : {len(lf)} pour {len(set(x.lower() for x in lf))} distincts",
           steps=["Brutes > Ajouter un dossier… > arw_lights", "Brutes > Ajouter des fichiers… > un de ses fichiers"])

# ------------------------------------------------------------------ E. empty folder, folder without frames
add_via_menu("Flats", _("Add folder…"), str(empty))
add_via_menu("Darks", _("Add folder…"), str(nofr))
sel_session()
rows = [t for _b, t in folder_rows()]
log("E rows", rows, [t for t in ptexts(ex) if t.startswith("⚠")])
ex.shot("E_empty_folders")
st = ["Flats > Ajouter un dossier… > un dossier vide", "Darks > Ajouter un dossier… > un dossier de .txt/.md"]
if not any(t.startswith("⚠") and ("vide" in t or "empty" in t or "aucune" in t.lower()) for t in ptexts(ex)):
    N.note("mineur", "Un dossier vide ou sans poses ajouté : listé « 0 fichier » sans avertissement",
           f"Lignes de dossiers : {rows}", steps=st)

# ------------------------------------------------------------------ F. open a group, take out one file, several, a folder
sel_session()
frs = folder_rows()
log("F folder rows", [t for _b, t in frs])
arw_row = next((b for b, t in frs if "n1" in t), None)
if arw_row is not None:
    ex.click(arw_row)
    wait(ex, 0.4)
lists = file_lists()
log("F lists", [x.count() for x in lists])
ex.shot("F_group_open")
look(ex, N, "panneau Session, groupe de fichiers ouvert", steps=["clic sur la ligne du dossier des brutes"])
lst = next((x for x in lists if x.count() >= 10), None)
if lst is None:
    N.note("gênant", "Le dossier des brutes ne s'ouvre pas sur ses fichiers", f"Listes : {[x.count() for x in lists]}",
           steps=["clic sur la ligne 📁 du dossier des brutes"])
else:
    n0 = len(ex.s.session_files(1, "lights"))
    # one file, its cross
    row_w = lst.itemWidget(lst.item(2))
    xb = [b for b in row_w.findChildren(QPushButton) if b.text() == "×"][0]
    gone_name = lst.item(2).data(Qt.UserRole)
    ex.click(xb)
    wait(ex, 1.0)
    n1 = len(ex.s.session_files(1, "lights"))
    log("F one out", gone_name, n0, n1, group_title("Lights"))
    N.check(n1 == n0 - 1, "gênant", "Retirer un fichier par sa croix : le compte ne baisse pas d'un",
            f"Avant {n0}, après {n1}", steps=["ouvrir le dossier des brutes", "× d'un fichier"])
    sel_session()
    lists = file_lists()
    lst = next((x for x in lists if x.count() >= 10), None)
    log("F list still open after removal", [x.count() for x in lists])
    if lst is None:
        N.note("mineur", "Après avoir retiré un fichier, le groupe se referme",
               "Le dossier ouvert pour retirer un fichier se referme : pour en retirer un autre il faut le rouvrir.",
               steps=["ouvrir le dossier des brutes", "× d'un fichier"])
        frs = folder_rows()
        b = next((b for b, t in frs if "n1" in t), None)
        if b is not None:
            ex.click(b)
            wait(ex, 0.4)
        lst = next((x for x in file_lists() if x.count() >= 5), None)
    if lst is not None:
        # several: select three, Del
        lst.setFocus()
        for r in (0, 1, 3):
            lst.item(r).setSelected(True)
        chosen = lst.chosen()
        ex.key(Qt.Key_Delete, lst)
        wait(ex, 1.0)
        n2 = len(ex.s.session_files(1, "lights"))
        log("F three out (Del)", chosen, n1, n2, group_title("Lights"))
        N.check(n2 == n1 - 3, "gênant", "Retirer plusieurs fichiers (sélection + Suppr) : le compte est faux",
                f"Avant {n1}, après {n2}, choisis {len(chosen)}", steps=["ouvrir le dossier", "Ctrl+clic sur 3 fichiers",
                                                                         "Suppr"])
    sel_session()
    ex.shot("F_after_removals")
    # Ctrl+Z after a file taken out: the principle says the frames are saved by Save, not undone
    can = ex.s.can_undo()
    undo(ex)
    wait(ex, 0.6)
    log("F undo after file removal: can_undo", can, len(ex.s.session_files(1, "lights")))

# a whole folder: the CR2 bias
sel_session()
frs = folder_rows()
bias_row = next((b for b, t in frs if "bias_cr2" in t), None)
if bias_row is not None:
    r = bias_row.parentWidget()
    xs = [b for b in visible(panel(ex), QPushButton) if b.text() == "×" and abs(b.mapTo(panel(ex), QPoint(0, 0)).y()
                                                                                  - bias_row.mapTo(panel(ex), QPoint(0, 0)).y()) < 6]
    log("F bias row cross", len(xs), xs[0].width() if xs else None, xs[0].height() if xs else None)
    if xs:
        ex.shot("F_folder_cross")
        ex.click(xs[0])
        wait(ex, 1.0)
        sel_session()
        log("F bias out", group_title("Bias"), summary(), [t for t in ptexts(ex) if t.startswith("⚠")])
        N.check(_("none") in group_title("Bias"), "gênant", "Retirer tout un dossier par sa croix : il reste",
                f"{group_title('Bias')!r}", steps=["Offsets : × de la ligne 📁 bias_cr2"])
        N.check(not any("CR2" in t for t in ptexts(ex) if t.startswith("⚠")), "mineur",
                "L'avertissement des offsets CR2 reste après leur retrait", f"{ptexts(ex)}",
                steps=["Offsets : × de la ligne 📁 bias_cr2"])

# the misc files one by one (JPG, PNG, TIFF): their folder row holds them
sel_session()
frs = folder_rows()
misc_row = next((b for b, t in frs if "misc" in t), None)
log("F misc row", misc_row.text() if misc_row else None, [t for t in ptexts(ex) if t.startswith("⚠")])
lists = file_lists()
log("F lists open by themselves (a warned file)", [x.count() for x in lists])

# ------------------------------------------------------------------ G. the last light
# take the lights out by the panel: every folder's cross, then the files
for _round in range(6):
    sel_session()
    frs = folder_rows()
    lights_rows = []
    # the rows above the Darks title belong to the lights
    y_darks = None
    for lb in visible(panel(ex), __import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel):
        if lb.text().startswith(_("Darks") + "  ·  "):
            y_darks = lb.mapTo(panel(ex), QPoint(0, 0)).y()
    for b, t in frs:
        if y_darks is None or b.mapTo(panel(ex), QPoint(0, 0)).y() < y_darks:
            lights_rows.append(b)
    if not lights_rows:
        break
    b = lights_rows[0]
    xs = [x for x in visible(panel(ex), QPushButton) if x.text() == "×" and abs(x.mapTo(panel(ex), QPoint(0, 0)).y()
                                                                                - b.mapTo(panel(ex), QPoint(0, 0)).y()) < 6]
    if not xs:
        break
    ex.click(xs[0])
    wait(ex, 1.0)
sel_session()
log("G no lights", group_title("Lights"), summary(), ex.s.p.session(1).lights, ex.popups[-2:])
ex.shot("G_last_light_removed")
st = ["Session : × de chaque dossier des brutes, jusqu'à la dernière"]
N.check(not ex.s.p.session(1).lights, "gênant", "La dernière brute ne peut pas être retirée",
        f"Brutes restantes : {ex.s.p.session(1).lights}", steps=st)
look(ex, N, "panneau Session sans brutes", steps=st)
# the graph and the run: nothing to run
log("G graph", graph_texts(ex), "run button", ex.w.bt_run.text(), ex.w.bt_run.isEnabled(), ex.w.status.text())
# the lights back, by a folder
add_via_menu("Lights", _("Add folder…"), str(arw))
sel_session()
log("G back", group_title("Lights"), summary())

# ------------------------------------------------------------------ H. filter, rename, Ctrl+Z
sel_session()
cb = next((c for c in visible(panel(ex), QComboBox)), None)
items = [cb.itemText(i) for i in range(cb.count())] if cb else []
log("H filter items", items)
if cb is not None:
    cb.setCurrentIndex(cb.findData("Ha"))
    wait(ex, 0.8)
    log("H filter Ha", ex.s.p.session(1).filter, summary(), ex.s.can_undo())
    sel_session()
    ex.shot("H_filter_Ha")
    can = ex.s.can_undo()
    undo(ex)
    wait(ex, 0.6)
    log("H undo filter", ex.s.p.session(1).filter)
    N.check(ex.s.p.session(1).filter != "Ha", "gênant", "Changer le filtre d'une session ne s'annule pas (Ctrl+Z)",
            f"Après Ctrl+Z, filtre : {ex.s.p.session(1).filter!r} (annulable : {can})",
            steps=["Session : Filtre = Ha", "Ctrl+Z"])
    sel_session()
    cb = next((c for c in visible(panel(ex), QComboBox)), None)
    if cb is not None and cb.currentData():
        cb.setCurrentIndex(0)
        wait(ex, 0.5)
# rename in the panel
sel_session()
ed = next((e for e in visible(panel(ex), QLineEdit)), None)
if ed is not None:
    ed.setFocus()
    ed.selectAll()
    from PySide6.QtTest import QTest
    QTest.keyClicks(ed, "Nuit du 3 mars")
    QTest.keyClick(ed, Qt.Key_Return)
    wait(ex, 0.6)
    log("H rename", ex.s.p.session(1).name, summary())
    N.check(ex.s.p.session(1).name == "Nuit du 3 mars", "gênant", "Renommer la session dans son panneau : sans effet",
            f"Nom : {ex.s.p.session(1).name!r}", steps=["Session : taper un nom dans le titre du panneau", "Entrée"])
    can = ex.s.can_undo()
    undo(ex)
    wait(ex, 0.6)
    log("H undo rename", ex.s.p.session(1).name, can)
    N.check(ex.s.p.session(1).name != "Nuit du 3 mars", "gênant", "Renommer une session ne s'annule pas (Ctrl+Z)",
            f"Après Ctrl+Z le nom reste {ex.s.p.session(1).name!r} (annulable : {can}). Le principe : toute "
            "modification s'annule, seules les images d'une session sont exclues.",
            steps=["Session : renommer dans le panneau", "Ctrl+Z"])
    # an empty name
    sel_session()
    ed = next((e for e in visible(panel(ex), QLineEdit)), None)
    ed.setFocus()
    ed.selectAll()
    QTest.keyClick(ed, Qt.Key_Delete)
    QTest.keyClick(ed, Qt.Key_Return)
    wait(ex, 0.6)
    sel_session()
    ed = next((e for e in visible(panel(ex), QLineEdit)), None)
    log("H empty name", repr(ex.s.p.session(1).name), repr(ed.text() if ed else None))
    if ed is not None and not ed.text().strip():
        N.note("mineur", "Nom de session vidé : le titre du panneau reste vide",
               f"Le nom gardé est {ex.s.p.session(1).name!r} mais le champ montre un titre vide.",
               steps=["Session : effacer le titre du panneau", "Entrée"])
# rename by the menu: the name dialog
ren = {}


def name_dialog(d):
    ren["class"] = d.__class__.__name__
    ren["title"] = d.windowTitle()
    ren["texts"] = texts(d)
    ex.shot("H_rename_dialog", d)
    eds = d.findChildren(QLineEdit)
    if eds:
        eds[0].setText("Nuit B")
    d.accept()


it = ex.block("sources") or ex.block("session")
ex.dialog_handler = name_dialog
items = ex.right_click_block(it, _("Rename session"))
wait(ex, 0.6)
log("H rename by menu", ren, ex.s.p.session(1).name)

# ------------------------------------------------------------------ I. "+ Session": a second night
from astro_gpu.client.graph import GhostItem  # noqa: E402
gh = [g for g in ex.view.scene().items() if isinstance(g, GhostItem) and not getattr(g, "new_flow", False)
      and "Session" in (g.text or "")]
log("I ghosts", [g.text for g in gh])
if gh:
    ex.file_queue = [str(cr2)]
    vp = ex.view.viewport()
    pt = ex.view.mapFromScene(gh[0].sceneBoundingRect().center())
    ex.click(vp, pt)
    wait(ex, 1.5)
log("I sessions", [(s.id, s.name, s.lights) for s in ex.s.p.sessions], ex.file_asks[-1:], graph_texts(ex))
ex.shot("I_two_sessions_no_workflow")
two = len(ex.s.p.sessions) == 2
N.check(two, "gênant", "« + Session » n'ajoute pas de session", f"Sessions : {ex.s.p.sessions}",
        steps=["clic sur « + Session » sous le bloc Session", "dossier de 10 CR2"])
if two and not any(b.node.key in ("session", "sources") for b in ex.blocks()):
    N.note("bloquant", "Deux sessions sans workflow : les blocs Session disparaissent du graphe",
           f"Après « + Session » dans un projet d'une session sans workflow (le seul workflow supprimé, ou un projet "
           f"importé ainsi), le graphe ne montre plus que {graph_texts(ex)} : aucune des deux Sessions n'est dessinée, "
           "on ne peut plus les choisir ni les supprimer ; « + Session » chevauche la zone « Nouveau flux ».",
           steps=["projet d'une session (15 ARW) sans workflow", "clic sur « + Session »", "dossier de 10 CR2"])

# the realistic way: a Blank project, a Session placed from the palette, its lights, then "+ Session"
ex.w.close_project()
wait(ex, 0.5)
ex.new_project("Deux nuits", None)
ex.close_dialogs()
wait(ex, 0.8)
ex.w.activateWindow()
ex.page.toasted.connect(lambda t: toasts.append(t))
ex.double_click_card("session")
wait(ex, 1.0)
log("I placed", graph_texts(ex), [(x.id, x.name) for x in ex.s.p.sessions], [q.name for q in ex.s.paths])
ex.shot("I_session_placed")
add_via_menu("Lights", _("Add folder…"), str(arw))
add_via_menu("Darks", _("Add folder…"), str(darks))
sel_session()
log("I first session", summary(), graph_texts(ex))
gh = [g for g in ex.view.scene().items() if isinstance(g, GhostItem) and not getattr(g, "new_flow", False)
      and "Session" in (g.text or "")]
if gh:
    ex.file_queue = [str(cr2)]
    r = gh[0].sceneBoundingRect()
    under = ex.view.itemAt(ex.view.mapFromScene(r.center()))
    if under is not gh[0] and getattr(under, "new_flow", False):
        N.note("gênant", "« + Session » est en partie sous la zone « Nouveau flux » : un clic en son milieu ne fait rien",
               f"« + Session » ({r.height():.0f} px de haut) est recouvert par la zone « Nouveau flux » à partir de "
               f"{under.sceneBoundingRect().top() - r.top():.0f} px : seul son haut répond au clic.",
               steps=["projet Vierge", "Session placée, ses brutes ajoutées", "clic au milieu de « + Session »"])
    from PySide6.QtCore import QPointF
    ex.click(ex.view.viewport(), ex.view.mapFromScene(QPointF(r.center().x(), r.top() + 5)))
    wait(ex, 1.5)
log("I sessions (blank project)", [(x.id, x.name, x.lights) for x in ex.s.p.sessions], graph_texts(ex))
ex.shot("I_two_sessions")
two = len(ex.s.p.sessions) == 2
# Calibration placed: one per session
ex.double_click_card("calibration")
wait(ex, 1.0)
cal = [(b.node.key, b.node.sid, b.title, b.summary) for b in ex.blocks() if b.node.key == "calibration"]
log("I calibration per session", cal)
ex.shot("I_calibration_per_session")
if two and len(cal) != 2:
    N.note("gênant", "Deux sessions : la Calibration placée n'est pas dessinée une fois par session",
           f"Blocs Calibration : {cal}", steps=["projet Vierge", "Session placée, ses brutes", "+ Session",
                                                 "double-clic sur Calibration dans la palette"])
# a file taken out after a block placed, then Ctrl+Z: what is undone?
if two:
    sel_session(1)
    frs = folder_rows()
    b = next((b for b, t in frs if "arw_lights" in t), None)
    if b is not None:
        ex.click(b)
        wait(ex, 0.4)
    lst = next((x for x in file_lists() if x.count() >= 5), None)
    if lst is not None:
        n_cal = len([x for x in ex.blocks() if x.node.key == "calibration"])
        n_files = len(ex.s.session_files(1, "lights"))
        xb = [b for b in lst.itemWidget(lst.item(0)).findChildren(QPushButton) if b.text() == "×"][0]
        ex.click(xb)
        wait(ex, 1.0)
        how = undo(ex)
        n_cal2 = len([x for x in ex.blocks() if x.node.key == "calibration"])
        n_files2 = len(ex.s.session_files(1, "lights"))
        log("I file out then Ctrl+Z", how, n_cal, "->", n_cal2, "files", n_files, "->", n_files2)
        ex.shot("I_file_out_then_undo")
        if n_cal2 < n_cal and n_files2 < n_files:
            N.note("question", "Ctrl+Z après avoir retiré un fichier : le fichier reste retiré, c'est le bloc placé avant "
                               "qui disparaît",
                   f"Calibration : {n_cal} → {n_cal2} bloc(s) ; brutes : {n_files} → {n_files2}. Les images d'une "
                   "session ne s'annulent pas (voulu), mais Ctrl+Z défait sans prévenir une action plus ancienne.",
                   steps=["Session placée, ses brutes, + Session", "Calibration placée", "× d'une brute", "Ctrl+Z"])
        undo(ex, redo=True)
if two:
    # per-session Session and Calibration blocks?
    keys = [(b.node.key, b.node.sid, b.title, b.summary) for b in ex.blocks()]
    log("I blocks", keys)
    # its panel: the other session's darks offered
    sel_session(2)
    log("I panel s2", ptexts(ex))
    ex.shot("I_session2_panel")
    items = add_via_menu("Darks", None, None)
    log("I darks menu s2", items)
    pre = _("Use the frames of “{name}”").split("{")[0].strip()
    use = next((x for x in items if pre and pre in (x.get("text") or "")), None)
    if use is None:
        N.note("gênant", "« Prendre celles de … » absent du menu Ajouter d'une deuxième session",
               f"Menu : {items}", steps=["deuxième session", "Darks > Ajouter ▾"])
    else:
        add_via_menu("Darks", use["text"], None)
        sel_session(2)
        log("I shared", group_title("Darks"), [t for t in ptexts(ex) if "⇄" in t])
        ex.shot("I_shared_darks")
        N.check(any("⇄" in t for t in ptexts(ex)), "mineur", "Darks partagés : le panneau ne dit pas « partagé avec »",
                f"{ptexts(ex)}", steps=["deuxième session", "Darks > Ajouter ▾ > Prendre celles de « … »"])
        # the type mismatch: ARW darks taken by a CR2 session
        wp = [t for t in ptexts(ex) if t.startswith("⚠")]
        log("I warn s2", wp, summary())
        N.check(any("ARW" in t for t in wp), "gênant", "Session CR2 qui prend les darks ARW d'une autre : pas d'avertissement",
                f"Avertissements : {wp}", steps=["deuxième session (CR2)", "Darks > Prendre celles de la première (ARW)"])
    look(ex, N, "panneau de la deuxième session", steps=["+ Session", "clic sur la deuxième Session"])
    # the same folder in two sessions
    sel_session(2)
    add_via_menu("Lights", _("Add folder…"), str(arw))
    sel_session(2)
    log("I same folder in two sessions", group_title("Lights"), [t for t in ptexts(ex) if t.startswith("⚠")],
        ex.popups[-1:], toasts[-2:])
    ex.shot("I_same_folder_two_sessions")
    s1 = set(x.lower() for x in ex.s.session_files(1, "lights"))
    s2 = set(x.lower() for x in ex.s.session_files(2, "lights"))
    if s1 & s2 and not any(t.startswith("⚠") for t in ptexts(ex)):
        N.note("gênant", "Le même dossier de brutes dans deux sessions : accepté sans avertissement",
               f"{len(s1 & s2)} brutes dans les deux sessions ; elles seraient empilées deux fois.",
               steps=["deuxième session", "Brutes > Ajouter un dossier… > le dossier de la première session"])
    # ------------------------------------------------------------------ J. remove a session: menu, cross, Del, Ctrl+Z
    n_pop = len(ex.popups)
    it2 = next((b for b in ex.blocks() if b.node.key == "session" and b.node.sid == 2), None)
    lights2 = list(ex.s.p.session(2).lights)
    items = ex.right_click_block(it2, _("Delete")) if it2 is not None else []
    log("J menu", items)
    wait(ex, 1.0)
    log("J removed by menu", [s.id for s in ex.s.p.sessions], ex.popups[n_pop:], toasts[-1:])
    ex.shot("J_removed_menu")
    N.check(len(ex.s.p.sessions) == 1, "gênant", "Supprimer une session par son menu : sans effet",
            str(ex.s.p.sessions), steps=["clic droit sur la deuxième Session", "Supprimer le bloc"])
    if ex.popups[n_pop:]:
        N.note("gênant", "Supprimer une session ouvre une fenêtre bloquante", str(ex.popups[n_pop:]),
               steps=["clic droit sur la deuxième Session", "Supprimer le bloc"])
    undo(ex)
    wait(ex, 1.5)
    log("J undo", [(s.id, s.name, len(s.darks), s.lights) for s in ex.s.p.sessions])
    if len(ex.s.p.sessions) == 2 and list(ex.s.p.session(2).lights) != lights2:
        N.note("gênant", "Ctrl+Z rend la session supprimée sans ses brutes", f"Avant : {lights2}; après : "
               f"{ex.s.p.session(2).lights}", steps=["Supprimer la deuxième Session", "Ctrl+Z"])
    N.check(len(ex.s.p.sessions) == 2, "gênant", "Ctrl+Z ne rend pas la session supprimée",
            str(ex.s.p.sessions), steps=["Supprimer la deuxième Session (menu)", "Ctrl+Z"])
    undo(ex, redo=True)
    wait(ex, 1.0)
    log("J redo", [s.id for s in ex.s.p.sessions])
    undo(ex)
    wait(ex, 1.5)
    # the cross
    it2 = next((b for b in ex.blocks() if b.node.key == "session" and b.node.sid == 2), None)
    if it2 is not None:
        ex.click_block(it2, "cross")
        wait(ex, 1.0)
        log("J removed by cross", [s.id for s in ex.s.p.sessions])
        N.check(len(ex.s.p.sessions) == 1, "gênant", "La croix d'une Session ne la supprime pas", "",
                steps=["survol de la deuxième Session", "clic sur sa croix"])
        undo(ex)
        wait(ex, 1.5)
    # Del
    it2 = next((b for b in ex.blocks() if b.node.key == "session" and b.node.sid == 2), None)
    if it2 is not None:
        ex.click_block(it2)
        wait(ex, 0.3)
        ex.w.activateWindow()
        ex.key(Qt.Key_Delete)
        wait(ex, 1.0)
        log("J removed by Del", [s.id for s in ex.s.p.sessions])
        N.check(len(ex.s.p.sessions) == 1, "gênant", "Suppr sur une Session sélectionnée ne la supprime pas", "",
                steps=["clic sur la deuxième Session", "Suppr"])
        undo(ex)
        wait(ex, 1.5)
    log("J back", [(s.id, s.name) for s in ex.s.p.sessions], graph_texts(ex))
    ex.shot("J_after_undos")

# ------------------------------------------------------------------ K. Review frames…
sel_session()
b = button(panel(ex), _("Review frames…"), QPushButton)
mode0 = ex.w.mode
if b is not None:
    ex.click(b)
    wait(ex, 1.5)
log("K review", mode0, "->", ex.w.mode, ex.s.sel_step)
ex.shot("K_review_frames")
N.check(ex.w.mode == 1, "gênant", "« Revoir les images… » n'ouvre pas Inspection", f"mode {ex.w.mode}",
        steps=["clic sur le bloc Session", "Revoir les images…"])
look(ex, N, "Inspection après « Revoir les images… »", ex.w, steps=["Session", "Revoir les images…"])
ex.w.set_mode(0)
wait(ex, 0.5)

# ------------------------------------------------------------------ L. the health check (FITS lights: they are read)
ex.w.close_project()
wait(ex, 0.5)
ex.new_project("Vérification", None)
ex.close_dialogs()
wait(ex, 0.8)
ex.w.activateWindow()
ex.double_click_card("session")
wait(ex, 1.0)
add_via_menu("Lights", _("Add folder…"), str(mono_l))


def health_wait(tag):
    from astro_gpu.client.frames import FramesView
    ex.w.set_mode(1)
    wait(ex, 1.0)
    fv = [x for x in ex.w.findChildren(FramesView) if x.isVisible()]
    lines = []
    for _i in range(24):
        if not fv:
            break
        lines = texts(fv[0])
        if _("Reading the frames…") not in lines:
            break
        wait(ex, 0.5)
    ex.shot(f"L_health_{tag}")
    ex.w.set_mode(0)
    wait(ex, 0.4)
    return fv, lines


fv, hl = health_wait("1_fits")
log("L health FITS only", hl)
st = ["Projet Vierge, Session placée", "Brutes : un dossier de 4 FITS", "Inspection (Sessions)"]
if not fv:
    N.note("gênant", "Inspection d'une Session : pas de page Sessions avec la vérification", f"mode {ex.w.mode}",
           steps=st)
add_via_menu("Lights", _("Add files…"), [str(misc / "phone.JPG"), str(misc / "picture.png")])
sel_session()
log("L panel", [t for t in ptexts(ex) if t.startswith("⚠")], summary())
ex.shot("L_fits_session")
fv, hl = health_wait("2_jpg")
log("L health + JPG/PNG", hl)
st += ["Brutes : Ajouter des fichiers… > phone.JPG, picture.png", "Inspection (Sessions)"]
if _("Reading the frames…") in hl:
    N.note("gênant", "Vérification des images : reste « Lecture des images… » après un JPEG parmi les brutes",
           f"Textes : {hl}", steps=st)
else:
    N.check(any("JPG" in t or "phone" in t or "PNG" in t for t in hl), "gênant",
            "Vérification des images : le JPEG et le PNG des brutes n'y sont pas signalés",
            f"Le principe : un fichier d'un autre type est signalé sur le bloc, dans son panneau et avant l'exécution. "
            f"Le bloc et le panneau le disent ; la vérification d'Inspection dit : {hl[-6:]}", steps=st)
    if fv:
        look(ex, N, "Inspection, page Sessions et vérification", fv[0], steps=st)
add_via_menu("Flats", _("Add folder…"), str(colour_fits))
fv, hl = health_wait("3_colour_flats")
log("L health + colour FITS flats", hl)
if _("Reading the frames…") in hl:
    N.note("gênant", "Vérification des images : reste « Lecture des images… » avec des flats FITS couleur",
           f"Textes : {hl}", steps=st + ["Flats : un dossier de 3 FITS couleur (3 plans)", "Inspection (Sessions)"])
add_via_menu("Bias", _("Add folder…"), str(bias_cr2))
fv, hl = health_wait("4_cr2_bias")
log("L health + CR2 bias (fake raw files: never decoded)", hl)
if _("Reading the frames…") not in hl:
    N.check(any("CR2" in t for t in hl), "gênant", "Vérification des images : les offsets CR2 (brutes FITS) n'y sont "
            "pas signalés", f"Lignes : {hl[-6:]}", steps=st + ["Offsets : un dossier de 4 CR2", "Inspection"])
ex.w.set_mode(0)
wait(ex, 0.5)
sel_session()
cb = next((c for c in visible(panel(ex), QComboBox)), None)
if cb is not None:
    cb.setCurrentIndex(cb.findData("OIII"))
    wait(ex, 0.8)
ex.w.set_mode(1)
wait(ex, 1.5)
from astro_gpu.client.frames import FramesView
fv = [x for x in ex.w.findChildren(FramesView) if x.isVisible()]
log("L health with filter OIII", texts(fv[0]) if fv else None)
if fv and fv[0].table is not None:
    t = fv[0].table
    hdr = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    row = [t.item(0, c).text() if t.item(0, c) else "" for c in range(t.columnCount())]
    log("L table", hdr, row)
    if _("Filter") in hdr and "OIII" not in row:
        N.note("gênant", "Inspection > Sessions : la colonne Filtre ne montre pas le filtre de la session",
               f"Colonnes {hdr} ; ligne {row} (filtre choisi : OIII).",
               steps=["Session : Filtre = OIII", "Inspection > Sessions"])
ex.w.set_mode(0)
wait(ex, 0.5)

# ------------------------------------------------------------------ M. mono sessions, two flows
ex.w.close_project()
wait(ex, 0.5)
root = ex.project("mono", "monoM")
ex.open(root)
wait(ex, 1.0)
ex.w.activateWindow()
bl = [(b.node.key, b.node.sid, b.title, b.summary) for b in ex.blocks()]
log("M blocks", bl)
ex.shot("M_mono_graph")
sb = [x for x in bl if x[0] == "session"]
cb_ = [x for x in bl if x[0] == "calibration"]
if len(sb) != 2:
    N.note("gênant", "Projet de deux filtres (Ha, OIII) : les deux Sessions ne sont pas dessinées", str(bl),
           steps=["ouvrir un projet de deux sessions Ha et OIII, un workflow par filtre"])
if len(cb_) != 2:
    N.note("mineur", "Projet de deux filtres : une Calibration par session attendue", str(cb_),
           steps=["ouvrir un projet de deux sessions Ha et OIII"])
sel_session(1)
log("M panel Ha", ptexts(ex)[:20])
ex.shot("M_panel_Ha")
look(ex, N, "panneau d'une session Ha", steps=["projet mono", "clic sur la Session Ha"])
cb = next((c for c in visible(panel(ex), QComboBox)), None)
if cb is not None:
    cb.setCurrentIndex(0)
    wait(ex, 1.0)
    log("M Ha -> none", [(b.node.key, b.title, b.summary) for b in ex.blocks()], ex.s.filters())
    ex.shot("M_filter_removed")
    sel_session(1)
    cb = next((c for c in visible(panel(ex), QComboBox)), None)
    if cb is not None:
        cb.setCurrentIndex(cb.findData("Ha"))
        wait(ex, 0.8)
z = ex.new_flow_zone()
rec = ex.drag_palette("session", ex.ghost_center(z)) if z is not None else None
wait(ex, 1.2)
log("M new flow", rec, [(q.name, q.sessions) for q in ex.s.paths], [(x.id, x.name, x.filter) for x in ex.s.p.sessions])
ex.shot("M_new_flow")
new_sid = max(x.id for x in ex.s.p.sessions)
log("M new session panel", ptexts(ex)[:12], ex.s.sel_step)
if len(ex.s.p.sessions) != 3:
    N.note("gênant", "Session déposée sur la zone « Nouveau flux » : pas de nouvelle session", str(rec),
           steps=["projet mono", "glisser Session de la palette sur « Nouveau flux »"])
else:
    sel_session(new_sid)
    add_via_menu("Lights", _("Add folder…"), str(sii))
    sel_session(new_sid)
    log("M SII lights", group_title("Lights"), summary(), ex.s.p.session(new_sid).filter)
    ex.shot("M_SII_added")
    if ex.s.p.session(new_sid).filter != "SII":
        N.note("question", "Brutes FITS d'un filtre (FILTER=SII) ajoutées dans le panneau : le filtre de la session "
                           "n'est pas repris de leur en-tête",
               f"Filtre de la session : {ex.s.p.session(new_sid).filter!r} ; « + Session » le lit pourtant de "
               "l'en-tête.", steps=["Nouveau flux : Session déposée", "Brutes > Ajouter un dossier… > 5 FITS SII"])
    n_paths = len(ex.s.paths)
    it3 = next((b for b in ex.blocks() if b.node.key == "session" and b.node.sid == new_sid), None)
    if it3 is not None:
        ex.right_click_block(it3, _("Delete"))
        wait(ex, 1.0)
        log("M flow session removed", len(ex.s.paths), n_paths, [x.id for x in ex.s.p.sessions])
        undo(ex)
        wait(ex, 1.5)
        log("M undo", len(ex.s.paths), [x.id for x in ex.s.p.sessions])
        N.check(len(ex.s.paths) == n_paths and any(x.id == new_sid for x in ex.s.p.sessions), "gênant",
                "Ctrl+Z après la suppression de la Session d'un flux : le flux ne revient pas",
                f"workflows {len(ex.s.paths)} / {n_paths}", steps=["Supprimer la Session du nouveau flux", "Ctrl+Z"])
ex.shot("M_end")

missing = missing_fr() - before_missing
log("MISSING A-M", sorted(missing))
if missing:
    N.note("mineur", "Session et vérification : textes restés en anglais", " | ".join(sorted(missing))[:1500],
           shot=False, steps=["Session : ajouter des fichiers de tous types, Inspection"])
(ex.out / "session_log.txt").write_text("\n".join(LOG), encoding="utf-8")
N.exceptions(["voir le journal du script explore_sources_session.py"])
ex.close()
print("DONE", TMP)
