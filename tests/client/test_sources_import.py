"""GUI exploration of 6-7 November, part 7 (sources, import, templates, files) and 1.7: what it found stays fixed.
Offscreen on scratch projects; nothing is computed (the Denoise preview and the inventory on tiny synthetic data)."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.gallery import GALLERY, chain_of, template_blocks, template_source  # noqa: E402
from astro_gpu.client.graph import GraphPage  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

FITS_HEAD = b"SIMPLE  =                    T" + b" " * 50


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def fits_file(path: Path, flt: str = "") -> Path:
    from astropy.io import fits
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.PrimaryHDU(np.zeros((4, 4), np.float32))
    if flt:
        h.header["FILTER"] = flt
    h.writeto(path, overwrite=True)
    return path


def open_page(tmp_path, monkeypatch, sessions=0, workflows=()):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    p.stack_mode = "per_session"
    for i in range(sessions):
        raw = tmp_path / "raw" / f"n{i + 1}"
        for k in range(3):
            fits_file(raw / f"L_{k:03d}.fits")
        p.sessions.append(project2.Session(i + 1, f"night {i + 1}", lights=[raw.as_posix()]))
    for name in workflows:
        p.add_workflow(name, wfl.TEMPLATES[name])
    project2.save(p)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(120)
    m._sync()
    page = GraphPage(m)
    page.resize(1500, 800)
    return w, m, page


def settle(w, page):
    w.wait(60)
    QApplication.processEvents()
    page.refresh()


def close(w, m, page):
    """As the window closes a project: the import's frame service stopped too (its threads left running crashed the
    test process at its exit)."""
    from astro_gpu.client.services.inspect import _services
    sv = _services.get(m)
    page.close()
    m.close(save=False)
    if sv is not None:
        sv.stop()
    w.stop()


def texts(widget) -> list[str]:
    return [x.text() for x in widget.findChildren(QLabel)] + [x.text() for x in widget.findChildren(QPushButton)]


def test_the_already_stretched_template_makes_a_stretched_image(qapp, tmp_path, monkeypatch):
    """1.7: its chain goes to Post as it is, its image marked stretched; its preview shows the same."""
    t = next(x for x in GALLERY if x["key"] == "image_stretched")
    assert template_source(t) == "stretched" and template_blocks(t) == [] and chain_of(t) == [("post", "")]
    png = tmp_path / "done.png"
    import cv2
    cv2.imwrite(str(png), np.full((8, 8, 3), 128, np.uint8))
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        d = page.new_workflow(tab="image")
        d.pick(next(c.t for c in d.cards if c.t["key"] == "image_stretched"))
        assert not d.ok.isEnabled() and d.why.text() == "No image file chosen yet."      # 7.5: says why
        d.file.setText(str(tmp_path / "gone.png"))
        assert not d.ok.isEnabled() and "File not found" in d.ok.toolTip()
        d.file.setText(str(png))
        assert d.ok.isEnabled() and not d.why.isVisibleTo(d)
        d.ok.click()
        settle(w, page)
        q = m.paths[-1]
        assert m.p.workflow(q.pid).source == {"path": png.as_posix(), "stretched": True}
        assert m.graph(q) == [] and m.source_kind(q) == "stretched"
    finally:
        close(w, m, page)


def test_an_image_block_takes_a_png_as_stretched_and_refuses_what_is_not_an_image(qapp, tmp_path, monkeypatch):
    """1.7, 7.3: a PNG or JPEG chosen is stretched (still changeable); a missing or a text file is refused, said why."""
    from astro_gpu.client.gallery import image_filter
    from astro_gpu.stages.image import image_problem
    assert "*.png" in image_filter() and "*.jpg" in image_filter() and "(*)" in image_filter()
    jpg, txt = tmp_path / "a.jpg", tmp_path / "notes.txt"
    import cv2
    cv2.imwrite(str(jpg), np.full((8, 8, 3), 128, np.uint8))
    txt.write_text("hello")
    fake = tmp_path / "fake.fits"
    fake.write_text("not a FITS file at all")
    assert image_problem(jpg) == "" and image_problem(fits_file(tmp_path / "s.fits")) == ""
    assert image_problem(tmp_path / "gone.fits").startswith("File not found")
    assert "not an image this version reads" in image_problem(txt)
    assert "its content is not that of its format" in image_problem(fake)
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.palette_action("image")                                       # placed from the palette: no file yet
        settle(w, page)
        q = m.paths[-1]
        assert m.p.workflow(q.pid).source == {"path": "", "stretched": False}
        page.select(index=m.paths.index(m.view(q.pid)), step="image")
        page.refresh()
        toasts = []
        page.panel.toast.connect(toasts.append)
        for bad in (txt, tmp_path / "gone.fits"):
            assert not page.panel.set_image(m.view(q.pid), str(bad)) and toasts
        assert m.p.workflow(q.pid).source["path"] == ""
        assert page.panel.set_image(m.view(q.pid), str(jpg))
        assert m.p.workflow(q.pid).source == {"path": jpg.as_posix(), "stretched": True}
        m.set_image_file(m.view(q.pid), stretched=False)                  # the user can still change it
        assert m.p.workflow(q.pid).source["stretched"] is False
    finally:
        close(w, m, page)


def test_frames_templates_start_from_session_blocks(qapp, tmp_path, monkeypatch):
    """7.2: Full on a project without frames starts from an (empty) Session block; Several nights from two."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        d = page.new_workflow(tab="frames")
        d.pick(next(c.t for c in d.cards if c.t["key"] == "full"))
        d.ok.click()
        settle(w, page)
        assert len(m.p.sessions) == 1 and not m.p.sessions[0].lights
        d = page.new_workflow(tab="frames")
        d.pick(next(c.t for c in d.cards if c.t["key"] == "nights"))
        d.ok.click()
        settle(w, page)
        assert len(m.p.sessions) == 2 and len(m.separate_sessions()) == 2
    finally:
        close(w, m, page)


def test_one_workflow_per_filter_works_on_a_new_project(qapp, tmp_path, monkeypatch):
    """7.2: its filters are chosen in the gallery; each gets its empty Session block and its workflow."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        d = page.new_workflow(tab="mono")
        d.pick(next(c.t for c in d.cards if c.t["key"] == "mono_each"))
        assert d.filter_row.isVisibleTo(d) and not d.ok.isEnabled() and "Choose the filters" in d.why.text()
        d.filter_b["Ha"].setChecked(True)
        d.filter_b["OIII"].setChecked(True)
        assert d.ok.isEnabled() and "Ha, OIII" in d.duration.text()
        d.ok.click()
        settle(w, page)
        assert sorted(s.filter for s in m.p.sessions) == ["Ha", "OIII"]
        assert sorted(q.name for q in m.paths) == ["Ha", "OIII"]
    finally:
        close(w, m, page)


def test_files_added_to_a_session_stay_listed_and_removable(qapp, tmp_path, monkeypatch):
    """7.4, 7.5: a folder gives its files, once for all; a file already there, a folder twice, an empty folder: said,
    not added; the lights' filter from their FITS header; one file taken out alone, the others kept."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        sid = m.add_filter_session("")
        folder = tmp_path / "sii"
        files = [fits_file(folder / f"S_{k}.fits", "SII") for k in range(4)]
        empty = tmp_path / "empty"
        empty.mkdir()
        notes: list = []
        assert m.add_session_entries(sid, "lights", [str(folder)], notes) == 4 and not notes
        assert sorted(Path(e).name for e in m.p.session(sid).lights) == ["S_0.fits", "S_1.fits", "S_2.fits", "S_3.fits"]
        assert m.p.session(sid).filter == "SII"                            # from the header
        assert m.add_session_entries(sid, "lights", [str(files[1]), str(folder)], notes) == 0
        assert [t for t, _n in notes] == ["“{name}” is already in the session."] * 2
        notes.clear()
        assert m.add_session_entries(sid, "flats", [str(empty)], notes) == 0 and "holds no frames" in notes[0][0]
        settle(w, page)
        assert len(m.session_files(sid, "lights")) == 4
        assert m.remove_session_files(sid, "lights", [str(files[1])])
        assert sorted(Path(e).name for e in m.p.session(sid).lights) == ["S_0.fits", "S_2.fits", "S_3.fits"]
    finally:
        close(w, m, page)


def test_a_folder_after_some_of_its_files_adds_the_others(qapp, tmp_path, monkeypatch):
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        sid = m.add_filter_session("")
        folder = tmp_path / "lights"
        files = [fits_file(folder / f"L_{k}.fits") for k in range(3)]
        assert m.add_session_entries(sid, "lights", [str(files[0])]) == 1
        assert m.add_session_entries(sid, "lights", [str(folder)]) == 2
        assert [Path(e).name for e in m.p.session(sid).lights] == ["L_0.fits", "L_1.fits", "L_2.fits"]
    finally:
        close(w, m, page)


def test_a_project_listing_a_folder_still_reads_it_and_its_files_are_known(qapp, tmp_path, monkeypatch):
    """A project made before the files were listed one by one: its folder entry is read, a file of it is already in."""
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    try:
        assert [Path(e).name for e in m.p.session(1).lights] == ["n1"] and len(m.session_files(1, "lights")) == 3
        notes: list = []
        assert m.add_session_entries(1, "lights", [m.session_files(1, "lights")[0]], notes) == 0 and notes
    finally:
        close(w, m, page)


def test_the_session_files_are_added_taken_out_and_said_kind_by_kind(qapp, tmp_path, monkeypatch):
    """The Session's Inspect is where its files go: an empty kind shows where, with no window in the way; the files of
    a kind are listed under it; the same files in two kinds, and lights at the exposure of flats, are said beside it."""
    from PySide6.QtCore import QMimeData, QUrl

    from astro_gpu.client.services.inspect import SessionRow
    from astro_gpu.client.sources import GroupRow, SourcesView, local_paths
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    view = SourcesView(m)
    said: list = []
    view.toast.connect(said.append)
    try:
        view.refresh()
        view.select_group("darks")
        assert view.stack.currentIndex() == 3 and "Drop your files here" in texts(view)
        assert not view.mode_btns[0].isEnabled()
        one, other = fits_file(tmp_path / "a" / "D_1.fits"), fits_file(tmp_path / "b" / "D_2.fits")
        assert view.add_files("darks", [str(one), str(other)]) == 2                # from two places
        settle(w, page)
        view.refresh()
        assert "2 files added" in said[-1] and view.stack.currentIndex() == view.mode
        lst = view.file_list
        assert sorted(lst.item(i).text() for i in range(lst.count())) == ["D_1.fits", "D_2.fits", "a · 1", "b · 1"]
        assert not lst.item(0).flags() & Qt.ItemIsSelectable                       # the folder's name: only a title
        lst.item(1).setSelected(True)
        view.files_out("darks", lst.chosen())
        settle(w, page)
        view.refresh()
        assert [Path(x).name for x in m.p.session(1).darks] == ["D_2.fits"] and "Ctrl+Z" in said[-1]
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(one))])
        assert local_paths(mime) == [Path(one).as_posix()]
        row = next(r for r in view.findChildren(GroupRow) if r.key == "flats")
        row.dropped.emit("flats", [m.session_files(1, "lights")[0]])               # a row takes what is dropped on it
        settle(w, page)
        view.refresh()
        assert view.group == "flats" and any("also in the lights" in t for t in view.anomalies("flats"))
        assert any("also in the lights" in t for t in texts(view))
        view.data = SimpleNamespace(sessions=[SessionRow(1, "night 1", lights=3, exposure=1 / 8000)])
        assert any("exposure of flats" in t for t in view.anomalies("lights"))
        view.data = SimpleNamespace(sessions=[SessionRow(1, "night 1", lights=3, exposure=15.0)])
        assert not any("exposure of flats" in t for t in view.anomalies("lights"))
    finally:
        view.close()
        close(w, m, page)


def test_the_fusion_panel_lists_its_palette_s_channels_and_can_delete(qapp, tmp_path, monkeypatch):
    """7.6, #27, #34: in HOO, SII is not listed with a weight; a Delete block button as the Image block's."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        for f in ("SII", "Ha", "OIII"):
            m.add_filter_session(f)
        made = {f: m.filter_workflow(f) for f in ("SII", "Ha", "OIII")}
        q = m.add_fusion("Nebula", "HOO", [{"pid": made[f].pid, "role": f, "channel": "L"} for f in made])
        settle(w, page)
        page.select(index=m.paths.index(m.view(q.pid)), step="fusion")
        page.refresh()
        from PySide6.QtWidgets import QComboBox
        tx = texts(page.panel) + [c.currentText() for c in page.panel.findChildren(QComboBox)]    # #55: a channel's list
        assert "Delete" in tx
        assert not any(t.startswith("SII ·") for t in tx) and any(t.startswith("Ha ·") for t in tx)
        assert "Unused by HOO: SII" in tx
    finally:
        close(w, m, page)


def test_two_images_alike_by_name_size_and_date_have_two_keys(tmp_path):
    """7.7: the image source's key holds a sample of its content; a copy elsewhere keeps its key."""
    from astro_gpu.core.store import Keys
    a, b, c = tmp_path / "A" / "stack.fits", tmp_path / "B" / "stack.fits", tmp_path / "C" / "stack.fits"
    for f, v in ((a, 1.0), (b, 2.0), (c, 1.0)):
        from astropy.io import fits
        f.parent.mkdir()
        fits.PrimaryHDU(np.full((3, 20, 30), v, np.float32)).writeto(f)
        os.utime(f, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    ws = []
    for f in (a, b, c):
        x = p.add_workflow(f.parent.name, wfl.TEMPLATES["Full"])
        x.source = {"path": f.as_posix(), "stretched": False}
        ws.append(x)
    project2.save(p)
    k = Keys(project2.load(p.root))
    ka, kb, kc = (k.key(x.id, "image") for x in ws)
    assert ka != kb and ka == kc


def test_denoise_preview_of_a_small_stack(tmp_path):
    """7.8: the before/after preview of a stack under 800 px is a smaller square, not a Python error."""
    from astro_gpu.stages.denoise import before_after
    a = np.random.default_rng(0).random((3, 200, 300)).astype(np.float32)
    out = before_after(a, a)
    assert out.shape == (3, 200, 406)
    assert before_after(np.ones((3, 900, 1000), np.float32), np.ones((3, 900, 1000), np.float32)).shape == \
        (3, 800, 1606)


def test_a_frames_folder_not_found_is_said(tmp_path):
    """7.9: the inventory names the folder that is not there, rather than asking to add frames."""
    from astro_gpu.core.project import Inputs
    from astro_gpu.core.stage import StageError
    from astro_gpu.stages.inventory import Inventory, InventoryParams
    gone = (tmp_path / "unplugged" / "lights").as_posix()
    ctx = SimpleNamespace(project=SimpleNamespace(config=SimpleNamespace(inputs=Inputs(lights=[gone]))))
    with pytest.raises(StageError, match="the folder of the lights is not found: .*unplugged/lights"):
        Inventory().run(ctx, InventoryParams())
    from astro_gpu.client.services.run import engine_text
    i18n.set_language("fr")
    try:
        assert engine_text(f"the folder of the flat darks is not found: {gone} (is its disk plugged in?)") == \
            f"dossier des darks de flat introuvable : {gone} (son disque est-il branché ?)"
    finally:
        i18n.set_language("en")


def test_a_frame_with_a_useful_area_is_sized_and_said_as_the_engine_reads_it(tmp_path):
    """Issue #46: the size the sessions' check compares is the useful area's, and the engine's two new lines have
    their French."""
    from astropy.io import fits

    from astro_gpu.client.model import _frame_read
    from astro_gpu.client.services.run import engine_text
    p = tmp_path / "a.fits"
    fits.PrimaryHDU(np.zeros((6, 8), np.uint16), fits.Header({"TRIMSEC": "[2:7,2:5]"})).writeto(p)
    assert _frame_read(str(p), True) == (True, (6, 4))
    i18n.set_language("fr")
    try:
        assert engine_text("useful area from the frames' header: 3054x2042 of 3136x2112 pixels, the overscan left out") \
            == "zone utile lue dans l'en-tête des images : 3054x2042 sur 3136x2112 pixels, l'overscan laissé de côté"
        assert engine_text("sparse field: 12 frame(s) aligned from a few stars (rotation, shift and scale only)") == \
            "champ peu étoilé : 12 image(s) alignée(s) sur quelques étoiles (rotation, décalage et échelle seulement)"
        assert engine_text("the 37 stars of this sparse field do not identify it on the sky beyond doubt, so no solution "
                           "is given rather than a wrong one: take Astrometry out of this workflow") == \
            "les 37 étoiles de ce champ peu étoilé ne le reconnaissent pas sur le ciel avec certitude : pas de solution " \
            "plutôt qu'une solution fausse ; retirez l'Astrométrie de ce workflow"
        assert engine_text("sparse field: the solution holds 117 powers of ten over chance, 9 needed (31 of 37 stars)") \
            == "champ peu étoilé : la solution tient à 117 puissances de dix au-dessus du hasard, 9 exigées (31 étoiles sur 37)"
        assert engine_text("reference frame: a.fits aligned too few of the frames (another field?), b.fits is used") == \
            "image de référence : a.fits alignait trop peu d'images (un autre champ ?), b.fits est prise"
        assert engine_text("only 5 stars found in the stack, at least 6 are needed to identify the field: add light "
                           "frames, or take Astrometry out of this workflow") == \
            "5 étoiles seulement dans l'empilement, il en faut au moins 6 pour reconnaître le champ : ajoutez des " \
            "brutes, ou retirez l'Astrométrie de ce workflow"
    finally:
        i18n.set_language("en")


def test_import_words_and_where_a_new_project_goes(tmp_path, monkeypatch):
    """7.1, #28: the project is never proposed inside the frames' folder; "Pas d'offsets"."""
    from astro_gpu.client.home import projects_location
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    raw = tmp_path / "M81" / "lights"
    raw.mkdir(parents=True)
    prefs.remember_folder(raw)
    where = Path(projects_location(avoid=str(raw)))
    assert where.resolve() != raw.resolve() and raw.resolve() not in where.resolve().parents
    prefs.add_recent((tmp_path / "projects" / "Old").as_posix(), "Old")
    (tmp_path / "projects").mkdir()
    assert Path(projects_location(avoid=str(raw))) == tmp_path / "projects"
    i18n.set_language("fr")
    try:
        assert i18n._("No bias.") == "Pas d'offsets." and i18n._("No bias, no dark flats.") == \
            "Ni offsets ni darks de flat."
        assert i18n._("Dark flats") == "Darks de flat" and i18n._("Flat darks") == "Darks de flat"
    finally:
        i18n.set_language("en")


def test_new_project_is_never_proposed_in_the_frames_last_added(tmp_path, monkeypatch):
    """GUI exploration 4: darks just added to a session (their folder the last one used), Home > New project proposes
    where the projects are, never inside the darks' folder; with no project yet, not in a folder of frames either."""
    from astro_gpu.client import home
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    darks = tmp_path / "data" / "darks"
    fits_file(darks / "D_001.fits")
    prefs.remember_folder(darks)
    assert Path(home.projects_location()) != darks                          # no project yet: not the frames' folder
    (tmp_path / "projects").mkdir()
    prefs.add_recent((tmp_path / "projects" / "First").as_posix(), "First")
    seen = []

    class Dialog:
        def __init__(self, location, *a):
            seen.append(Path(location))

        def exec(self):
            return False
    monkeypatch.setattr(home, "NewProjectDialog", Dialog)
    home.HomeView.ask_new_project(SimpleNamespace(window=lambda: None))
    assert seen == [tmp_path / "projects"]
