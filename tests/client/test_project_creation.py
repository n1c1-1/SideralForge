"""A new project: the graph at once, its name asked once; the Project block; the recovery after a crash."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QMessageBox, QPushButton  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import drafts, prefs  # noqa: E402
from astro_gpu.client.graph import ProjectItem  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.skyobjects import SkyObject  # noqa: E402
from astro_gpu.core.workflow import TEMPLATES  # noqa: E402

MACHINE = Machine([Gpu("cuda:0", "RTX", 16.0)], 16, 64.0)


@pytest.fixture(scope="session")
def qapp():
    from astro_gpu.client.widgets import QSS
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return app


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    prefs.load()
    prefs.set("home_sections", [])
    prefs.set("cache_dir", (tmp_path / "scratch").as_posix())
    appmod.apply_language("en")
    monkeypatch.setattr(appmod, "machine_info", lambda: MACHINE)
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: None)
    windows = []
    yield {"tmp": tmp_path, "windows": windows, "mp": monkeypatch}
    for w in windows:
        w.close()
        w.deleteLater()
    QApplication.processEvents()


def settle(w, n=8):
    for _i in range(n):
        QApplication.processEvents()
        w.worker.wait(30)
    QApplication.processEvents()


def main(env, size=(1500, 800)) -> appmod.Main:
    w = appmod.Main()
    w.resize(*size)
    w.show()
    env["windows"].append(w)
    w.worker.wait()
    settle(w)
    return w


def untitled(env) -> appmod.Main:
    w = main(env)
    w.new_project()
    w.worker.wait()
    settle(w, 12)
    return w


def named(env, name="M 42"):
    return lambda parent, suggested="": (env["tmp"] / "ws" / name, name)


# ---------------------------------------------------------------------------------------------- the graph at once
def test_new_project_opens_the_graph_with_no_window_and_leaves_no_trace_when_closed(env):
    w = untitled(env)
    root = w.project.root
    assert w.screens.currentWidget() is w.proj_screen and drafts.is_untitled(root) and (root / project2.FILE).is_file()
    assert w.project.name == "Untitled" and not prefs.recent() and not [d for d in w.findChildren(QDialog) if d.isVisible()]
    assert not w.s.dirty
    w.close_project()
    settle(w)
    assert not root.exists() and not drafts.recoveries()


def test_the_name_is_asked_at_the_first_save_and_the_project_moves_there(env):
    w = untitled(env)
    old = w.project.root
    w.graph.palette_action("session")
    w.worker.wait()
    settle(w)
    asked = []
    env["mp"].setattr(appmod, "ask_project_name", lambda parent, name="": asked.append(name))    # cancelled
    w.save()
    assert drafts.is_untitled(w.project.root) and asked == [""] and w.s.dirty
    env["mp"].setattr(appmod, "ask_project_name", named(env))
    w.save()
    settle(w)
    root = env["tmp"] / "ws" / "M 42"
    assert w.project.root == root == w.s.root and (root / project2.FILE).is_file() and not w.s.dirty
    assert project2.load(root).name == "M 42" and len(project2.load(root).sessions) == 1
    assert not old.exists() and prefs.recent()[0]["path"] == root.as_posix()
    assert w.windowTitle() == "FLOAT · M 42"
    asked.clear()
    env["mp"].setattr(appmod, "ask_project_name", lambda parent, name="": asked.append(name))
    w.save()
    assert not asked                                              # once


def test_the_name_is_asked_before_a_run_and_a_refusal_runs_nothing(env):
    w = untitled(env)
    env["mp"].setattr(appmod, "ask_project_name", lambda parent, name="": None)
    assert w.run.before_run() is False
    assert w.run.launch() is None
    env["mp"].setattr(appmod, "ask_project_name", named(env, "Run"))
    assert w.run.before_run() is True and not drafts.is_untitled(w.project.root)
    assert w.run.before_run() is True


def test_the_name_proposed_is_the_target_else_the_folder_of_the_images(env):
    w = untitled(env)
    assert w._suggested_name() == ""
    w.s.add_filter_session("")
    sid = w.s.p.sessions[0].id
    w.s.p.session(sid).lights = [(env["tmp"] / "Rosette" / "a.fits").as_posix()]
    assert w._suggested_name() == "Rosette"
    w.s.set_target_name("M 42")
    assert w._suggested_name() == "M 42"
    w.s.set_project_name("Orion in the wind")                    # typed in the Project block's form (issue #89)
    assert w._suggested_name() == "Orion in the wind"


def test_closing_a_project_without_a_name_with_edits_asks_its_name_or_drops_it(env):
    w = untitled(env)
    w.graph.palette_action("session")
    w.worker.wait()
    settle(w)
    root = w.project.root
    env["mp"].setattr(w, "ask_save", lambda: "save")
    env["mp"].setattr(appmod, "ask_project_name", lambda parent, name="": None)
    assert w.close_project() is False and w.project is not None   # cancelled: it stays
    env["mp"].setattr(w, "ask_save", lambda: "discard")
    assert w.close_project() is not False and not root.exists()


# ---------------------------------------------------------------------------------------------- the Project block
def project_with_a_chain(env) -> appmod.Main:
    root = env["tmp"] / "ws" / "Chain"
    p = project2.create(root, "Chain", Target())
    p.sessions.append(project2.Session(1, "night", lights=["F:/raw/none"]))
    p.add_workflow("Full", TEMPLATES["Full"])
    project2.save(p)
    w = main(env)
    w.open_project(root)
    w.worker.wait()
    settle(w, 12)
    return w


def item(w) -> ProjectItem:
    return next(x for x in w.graph.graph.scene().items() if isinstance(x, ProjectItem))


def panel_field(w, placeholder_text=None) -> QLineEdit:
    return next(e for e in w.graph.panel.findChildren(QLineEdit) if e.isVisibleTo(w.graph.panel))


def click_project(w):
    gv = w.graph.graph
    w.graph.select_project()
    settle(w)
    return gv


def test_the_project_block_is_above_the_first_object_from_the_blank_page_on(env):
    w = untitled(env)
    it = item(w)
    assert it.state == "none" and it.title == "Project" and it.summary == "Untitled"
    zone = w.graph.graph.new_flow_ghost.rect
    assert it.rect.bottom() < zone.top()
    assert w.graph.current_node() is w.graph.project_node         # a blank page: its panel
    w2 = project_with_a_chain(env)
    first = w2.graph.graph.graph.root
    assert item(w2).rect.bottom() < first.y - 27 and item(w2).rect.left() == first.col
    assert not item(w2).selected


def test_its_wire_to_astrometry_shows_only_while_selected(env):
    w = project_with_a_chain(env)
    calls = []
    orig = w.graph.graph._project_wire
    env["mp"].setattr(w.graph.graph, "_project_wire", lambda it, n: (calls.append(n.key), orig(it, n)))
    w.graph.graph.rebuild()
    assert not calls
    click_project(w)
    w.graph.graph.rebuild()
    assert item(w).selected and "astrometrie" in calls
    calls.clear()
    w.graph.select(step="alignement")
    settle(w)
    assert not calls and not item(w).selected
    w.graph.select(step="astrometrie")
    settle(w)
    assert calls == ["astrometrie"]


def test_the_target_typed_goes_green_or_orange_and_undo_takes_it_back(env):
    w = project_with_a_chain(env)
    click_project(w)
    ed = panel_field(w)
    ed.setText("M 42")
    ed.editingFinished.emit()
    settle(w)
    t = w.s.p.target
    assert t.name == "M 42" and t.ra is not None and w.s.target_state().dot == "known" and item(w).title == "M 42"
    assert w.s.dirty
    w.s.set_target_name("Region of mine")
    settle(w)
    assert t.ra is None and w.s.target_state().dot == "unknown" and item(w).state == "partial"
    assert w.s.undo()
    settle(w)
    assert w.s.p.target.name == "M 42" and w.s.p.target.ra is not None
    assert w.s.undo()
    assert w.s.p.target.name == "" and w.s.target_state().dot == "empty"


def test_a_del_on_the_project_block_does_nothing(env):
    w = project_with_a_chain(env)
    click_project(w)
    n = len(w.s.paths)
    w.graph._delete_selected()
    settle(w)
    assert len(w.s.paths) == n and len(w.s.p.sessions) == 1


def test_capture_data_and_comments_are_kept_with_the_project(env):
    w = project_with_a_chain(env)
    w.s.set_target_values(focal_mm=135.0, pixel_um=5.36)
    w.s.set_description("Second night in the wind")
    w.save()
    d = project2.load(w.project.root)
    assert d.target.focal_mm == 135.0 and d.target.pixel_um == 5.36 and d.description == "Second night in the wind"


ROSETTE = SkyObject("NGC 2237", "Rosette Nebula", "HII region", 98.0, 5.0, 80.0, 60.0, 0.0, ("Caldwell 49",), None)


def test_an_empty_target_shows_what_astrometry_found_without_changing_what_was_typed(env):
    w = project_with_a_chain(env)
    w.s._facts.in_field = [ROSETTE]
    st = w.s.target_state()
    assert st.dot == "known" and st.name == "NGC 2237" and not st.replace
    assert w.s.p.target.name == "" and not w.s.dirty              # never written into the input Astrometry reads


def test_a_target_the_field_does_not_show_is_proposed_for_replacement_never_replaced(env):
    w = project_with_a_chain(env)
    w.s.set_target_name("M 42")
    w.s._facts.in_field = [ROSETTE]
    st = w.s.target_state()
    assert st.dot == "unknown" and st.replace == "NGC 2237" and w.s.p.target.name == "M 42"
    click_project(w)
    w.graph.panel.refresh(w.graph.project_node)
    b = next(x for x in w.graph.panel.findChildren(QPushButton) if x.text() == "Replace by NGC 2237")
    b.click()
    settle(w)
    assert w.s.p.target.name == "NGC 2237" and w.s.target_state().dot == "known"
    # its common name and aliases count; a name of one's own is never contradicted
    w.s.set_target_name("Rosette Nebula")
    assert not w.s.target_state().replace
    w.s.set_target_name("Region of mine")
    assert not w.s.target_state().replace


def test_a_name_with_coordinates_does_not_touch_the_astrometry_settings():
    from types import SimpleNamespace

    from astro_gpu.stages.astrometry import Astrometry
    ctx = SimpleNamespace(project=SimpleNamespace(config=SimpleNamespace(target=Target(name="M 42", ra=83.8, dec=-5.4))),
                          metric_of=lambda *a: None, load_json=lambda *a: {})
    auto = Astrometry().auto(ctx)
    assert "target" not in auto and auto["ra"].value == 83.8
    ctx.project.config.target = Target(name="Region of mine")
    assert Astrometry().auto(ctx)["target"].value == "Region of mine"


# ---------------------------------------------------------------------------------------------- the recovery
def crash(w):
    """The window goes with no close: what a crash leaves on the disk stays."""
    w._recovery_timer.stop()
    w._recovery_mine = False


def test_a_crash_is_offered_on_home_and_reopens_the_project_unsaved(env):
    w = project_with_a_chain(env)
    root = w.project.root
    w.s.set_target_name("M 42")
    w._write_recovery()
    before = project2.load(root).target.name
    assert before == "" and len(drafts.recoveries()) == 1         # the project's own file is not touched
    crash(w)
    w2 = main(env)
    w2.show_home()
    settle(w2)
    assert not w2.banner.isHidden() and "Chain" in w2.banner.text.text()
    assert not w2.banner.close_b.isVisibleTo(w2.banner)           # kept until the user chooses
    b = next(x for x in w2.banner.findChildren(QPushButton) if x.text() == "Reopen")
    b.click()
    w2.worker.wait()
    settle(w2, 12)
    assert w2.project is not None and w2.project.root == root and w2.s.p.target.name == "M 42" and w2.s.dirty
    w2.save()
    assert project2.load(root).target.name == "M 42" and not drafts.recoveries()


def test_an_untitled_project_stays_untitled_after_a_crash(env):
    w = untitled(env)
    root = w.project.root
    w.graph.palette_action("session")
    w.worker.wait()
    settle(w)
    w._write_recovery()
    crash(w)
    w2 = main(env)
    assert root.exists()                                          # it has a recovery: not an orphan
    w2.show_home()
    settle(w2)
    next(x for x in w2.banner.findChildren(QPushButton) if x.text() == "Reopen").click()
    w2.worker.wait()
    settle(w2, 12)
    assert w2.project.root == root and drafts.is_untitled(root) and len(w2.s.p.sessions) == 1 and w2.s.dirty


def test_a_project_opened_from_home_and_saved_leaves_no_copy_of_before(env):
    """The night test pass: after a crash, the project opened from Home (not through the banner), edited and saved:
    the older copy was still offered, and Reopen would have taken the saved work back."""
    w = project_with_a_chain(env)
    root = w.project.root
    w.s.set_target_name("M 42")
    w._write_recovery()
    crash(w)
    w2 = main(env)
    w2.open_project(root)
    w2.worker.wait()
    settle(w2, 12)
    w2.s.set_target_name("M 31")
    w2.save()
    assert project2.load(root).target.name == "M 31" and not drafts.recoveries()


def test_ignore_drops_the_copy_and_the_folder_of_an_untitled_project(env):
    w = untitled(env)
    root = w.project.root
    w.graph.palette_action("session")
    w.worker.wait()
    settle(w)
    w._write_recovery()
    crash(w)
    w2 = main(env)
    w2.show_home()
    settle(w2)
    next(x for x in w2.banner.findChildren(QPushButton) if x.text() == "Ignore").click()
    settle(w2)
    assert not drafts.recoveries() and not root.exists() and w2.banner.isHidden()


def test_a_folder_left_before_any_edit_is_cleaned_at_the_next_start(env):
    w = untitled(env)
    root = w.project.root
    crash(w)
    main(env)
    assert not root.exists()


def test_saving_and_closing_normally_leave_no_copy(env):
    w = project_with_a_chain(env)
    w.s.set_target_name("M 42")
    w._write_recovery()
    assert drafts.recoveries()
    w.save()
    assert not drafts.recoveries()
    w.s.set_target_name("M 31")
    w._write_recovery()
    assert drafts.recoveries()
    w.close_project()
    assert not drafts.recoveries()


def test_an_edit_writes_the_copy_some_seconds_after(env):
    w = project_with_a_chain(env)
    env["mp"].setattr(appmod, "RECOVERY_MS", 50)
    w._recovery_timer.setInterval(50)
    w.s.set_target_name("M 42")
    assert w._recovery_timer.isActive() and not drafts.recoveries()
    import time
    t0 = time.time()
    while not drafts.recoveries() and time.time() - t0 < 3:
        QApplication.processEvents()
    assert drafts.recoveries()[0]["definition"]["target"]["name"] == "M 42"


def test_the_banner_of_a_failure_can_still_be_closed(env):
    w = main(env)
    w.banner.say("Something failed")
    assert w.banner.close_b.isVisibleTo(w.banner) and not w.banner.findChildren(QPushButton)


def test_a_real_click_selects_the_project_block(env):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    w = project_with_a_chain(env)
    gv = w.graph.graph
    w.show()
    settle(w)
    assert not item(w).selected
    QTest.mouseClick(gv.viewport(), Qt.LeftButton, pos=gv.mapFromScene(item(w).sceneBoundingRect().center()))
    settle(w)
    assert item(w).selected


FULL = dict(instrument="Refractor 80/480", mount="German mount", filter="Dual band", location="Garden",
            date="2026-11-09")


def form(w):
    """The Project block's form shown in the centre of Inspect (issue #89)."""
    from astro_gpu.client.panel import ProjectForm
    f = w.tune.project_form
    assert isinstance(f, ProjectForm) and f.isVisible() and w.tune.isVisible()
    return f


def test_the_project_block_is_red_orange_or_green_and_its_tooltip_says_what_is_missing(env):
    """Issue #89: red with no target, orange with a target but shooting details missing, green when complete."""
    w = project_with_a_chain(env)
    assert item(w).state == "none"
    assert "Missing: target, instrument, mount, filter, location, date" in item(w).toolTip()
    w.s.set_target_name("M 42")
    settle(w)
    assert item(w).state == "partial" and "Missing: instrument, mount, filter, location, date" in item(w).toolTip()
    for k, v in FULL.items():
        w.s.set_shooting_detail(k, v)
    settle(w)
    assert item(w).state == "complete" and "Missing" not in item(w).toolTip() and "Complete" in item(w).toolTip()
    w.s.set_shooting_detail("date", "  ")                         # emptied: forgotten, missing again
    settle(w)
    assert "date" not in w.s.p.acquisition and item(w).state == "partial" and "Missing: date" in item(w).toolTip()
    w.s.set_target_name("")
    w.s._facts.in_field = [ROSETTE]                               # what Astrometry found counts as a target
    w.graph.graph.rebuild()
    assert item(w).state == "partial"
    from astro_gpu.client.widgets import BAD, OK, WARN
    assert ProjectItem.COLOURS == {"none": BAD, "partial": WARN, "complete": OK}


def test_a_double_click_on_the_project_block_opens_its_form_in_inspect(env):
    """Issue #89: a double-click on the Project block opens it in Inspect, as a double-click opens the other blocks, with
    one form in the centre: name, target, shooting details; what is typed there is the project's."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    w = project_with_a_chain(env)
    w.s.set_target_name("M 42")
    gv = w.graph.graph
    settle(w)
    QTest.mouseDClick(gv.viewport(), Qt.LeftButton, pos=gv.mapFromScene(item(w).sceneBoundingRect().center()))
    settle(w)
    assert w.mode == 1 and w.graph.project_open()
    f = form(w)
    assert w.tune.center.indexOf(f) >= 0 and not w.tune.panel.findChildren(QLineEdit)   # one form, in the centre
    assert f.name.text() == "Chain" and f.target.text() == "M 42" and "Missing: instrument" in f.state.text()
    assert all(e.toolTip() for e in [f.name, f.target, *f.capture.values(), *f.shooting.values(), f.comment])
    f.name.setText("Orion")
    f.name.editingFinished.emit()
    f.target.setText("NGC 7000")
    f.target.editingFinished.emit()
    settle(w)
    assert w.s.p.name == "Orion" and w.s.p.target.name == "NGC 7000" and w.s.dirty
    assert form(w) is f and "Orion" in w.title.text()                      # kept, not built again under the fingers
    f.name.setText("  ")                                                  # a name emptied: the one it had comes back
    f.name.editingFinished.emit()
    settle(w)
    assert w.s.p.name == "Orion" and f.name.text() == "Orion"
    assert w.s.undo()
    settle(w)
    assert w.s.p.target.name == "M 42" and f.target.text() == "M 42"
    assert w.s.undo()
    settle(w)
    assert w.s.p.name == "Chain" and f.name.text() == "Chain"
    w.tune.back.emit()                                                    # Esc: the graph, the block still chosen
    settle(w)
    assert w.mode == 0 and item(w).selected


def test_a_single_click_keeps_the_graph_and_enter_opens_the_form(env):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    w = project_with_a_chain(env)
    gv = w.graph.graph
    QTest.mouseClick(gv.viewport(), Qt.LeftButton, pos=gv.mapFromScene(item(w).sceneBoundingRect().center()))
    settle(w)
    assert w.mode == 0 and item(w).selected and w.graph.panel.findChildren(QLineEdit)
    w.activateWindow()                                    # offscreen the window is not active by itself: its shortcuts
    QApplication.setActiveWindow(w)                       # need it to be
    gv.viewport().setFocus()
    QTest.keyClick(gv.viewport(), Qt.Key_Return)
    settle(w)
    assert w.mode == 1 and form(w).target.isVisible()
    w.resize(1100, 520)                                   # a small window: the form scrolls, never wider than its room
    settle(w)
    f = form(w)
    assert f.verticalScrollBar().maximum() > 0 and f.widget().width() <= f.viewport().width()
    w.tune.select(step="alignement")                                     # another block: the form goes
    w.tune.refresh()
    settle(w)
    assert w.tune.project_form is None and not w.graph.project_open()


def test_a_shooting_detail_of_the_form_reaches_the_report_and_back(env):
    """Issue #89: the form and the Report edit one place in the project: what is typed in one is in the other, in the
    Report's text, saved with the project, taken back by Undo."""
    from astro_gpu.client import report as rep
    w = project_with_a_chain(env)
    w.graph.open_project()
    settle(w)
    f = form(w)
    f.shooting["mount"].setText("German mount")
    f.shooting["mount"].editingFinished.emit()
    settle(w)
    assert w.s.p.acquisition == {"mount": "German mount"} and w.s.dirty
    w.set_mode(3)
    settle(w)
    assert w.report.fields["mount"].text() == "German mount"
    assert ("mount", "Mount", "German mount") in rep.shot_rows(rep.ReportData(), {}, w.report.entered(), "en")
    w.report.fields["date"].setText("2026-11-09")
    w.report.set_field("date")
    assert w.s.undo()                                                   # Undo while the Report is shown: its field too
    settle(w)
    assert "date" not in w.s.p.acquisition and w.report.fields["date"].text() == ""
    assert w.s.redo()
    settle(w)
    assert w.report.fields["date"].text() == "2026-11-09"
    w.set_mode(0)
    w.graph.open_project()
    settle(w)
    f = form(w)
    assert f.shooting["date"].text() == "2026-11-09" and "Missing: target, instrument, filter, location" in f.state.text()
    w.save()
    assert project2.load(w.project.root).acquisition == {"mount": "German mount", "date": "2026-11-09"}
    assert w.s.undo()
    settle(w)
    assert w.s.p.acquisition == {"mount": "German mount"} and form(w).shooting["date"].text() == ""


def test_a_field_the_cursor_sits_in_takes_the_value_changed_elsewhere_and_enter_writes_only_a_change(env):
    """Issue #131: Location typed in the form, Enter, the cursor left there; the Report changes it: the form shows the
    Report's value and Enter there does not write the old one back. A field being typed in keeps what is typed. The
    same both ways: the Report's field the cursor sits in takes the value the form changed."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    w = project_with_a_chain(env)
    w.graph.open_project()
    settle(w)
    w.activateWindow()                                    # offscreen the window is not active by itself: no focus
    QApplication.setActiveWindow(w)
    f = form(w)
    e = f.shooting["location"]
    e.setFocus()
    QTest.keyClicks(e, "Backyard")
    QTest.keyClick(e, Qt.Key_Return)
    settle(w)
    assert e.hasFocus() and w.s.p.acquisition == {"location": "Backyard"}
    w.s.set_shooting_detail("location", "Mountain site")                  # as the Report does
    f.load()                                                              # the block shown again
    assert e.hasFocus() and e.text() == "Mountain site"
    QTest.keyClick(e, Qt.Key_Return)
    settle(w)
    assert w.s.p.acquisition == {"location": "Mountain site"}
    w.s.set_shooting_detail("location", "Peak")                           # the form not shown again meanwhile
    QTest.keyClick(e, Qt.Key_Return)
    assert w.s.p.acquisition == {"location": "Peak"}
    f.load()
    QTest.keyClicks(e, " 2")                                              # being typed in: what is typed stays
    w.s.set_shooting_detail("filter", "Dual band")
    f.load()
    assert e.text() == "Peak 2"
    QTest.keyClick(e, Qt.Key_Return)
    settle(w)
    assert w.s.p.acquisition["location"] == "Peak 2"
    w.set_mode(3)                                                         # the other way: the Report's field
    settle(w)
    r = w.report.fields["location"]
    r.setFocus()
    assert r.hasFocus() and r.text() == "Peak 2"
    w.s.set_shooting_detail("location", "Garden")                         # as the form does
    settle(w)
    w.report._fill_entered()
    assert r.text() == "Garden"
    QTest.keyClick(r, Qt.Key_Return)
    assert w.s.p.acquisition["location"] == "Garden"


def test_the_top_bar_shows_the_name_given_at_the_first_save(env):
    w = untitled(env)
    env["mp"].setattr(appmod, "ask_project_name", named(env, "Orion"))
    w.save()
    assert "Orion" in w.title.text() and "Orion" in w.windowTitle()


def test_the_status_bar_gives_no_duration_during_a_run(env):
    w = project_with_a_chain(env)
    env["mp"].setattr(type(w.run), "running", property(lambda s: True))
    env["mp"].setattr(type(w.run), "progress", lambda s: 0.5)
    w.refresh()
    assert "to run" not in w._line


@pytest.mark.parametrize("has,word", [(False, "Add lights"), (True, "Open frames")])
def test_the_session_panel_says_add_lights_while_it_has_none(env, has, word):
    from PySide6.QtWidgets import QPushButton
    root = env["tmp"] / "ws" / "Empty"
    frames = env["tmp"] / "frames"
    frames.mkdir()
    (frames / "a.fits").write_bytes(b"x")
    files = [str(frames)] if has else []
    p = project2.create(root, "Empty", Target())
    p.sessions.append(project2.Session(1, "night", lights=files))
    p.add_workflow("Full", TEMPLATES["Full"])
    project2.save(p)
    w = main(env)
    w.open_project(root)
    w.worker.wait()
    settle(w, 12)
    panel = w.graph.panel
    panel.session_panel(1)
    settle(w)
    assert any(b.text() == word for b in panel.findChildren(QPushButton) if b.isVisibleTo(panel))
