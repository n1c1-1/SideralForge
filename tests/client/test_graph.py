"""P9b: the graph, the panel and the model, offscreen, on F:/astro_essais/v2/heart_a7s with four workflows.

The test project is a v2 copy with adopted results (all 15 stages of Full done). The tests add Classic stack, Classic +
denoise and Full + neural stars in memory; the model is read-only (nothing is saved into the project), except the
autosave test, which works on a copy of the definition in a temporary folder. ASTRO_GPU_SETTINGS points to a temporary
file in every test; the user's settings file is checked untouched. ASTRO_GPU_SHOTS names the folder of the screenshots
(default: a temporary folder).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")      # the offscreen platform has no fonts else

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QFont, QFontMetricsF  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGraphicsPathItem,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from astro_gpu import api  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.blocks import COLUMNS, Graph, block_state  # noqa: E402
from astro_gpu.client.graph import BH, BW, MIN_SCALE, BlockItem, GraphPage  # noqa: E402
from astro_gpu.client.i18n import hm  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.panel import CompactRow, MethodCard  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

F_PROJECT = Path("F:/astro_essais/v2/heart_a7s")
pytestmark = pytest.mark.skipif(not (F_PROJECT / "project.astroproj").is_file(),
                                reason="the heart_a7s v2 test project is not on this machine")
FOUR = (("Classic stack", "Classic stack", None),
        ("Classic + denoise", "Classic + denoise", {"denoise": {"iterations": 6000}}),
        ("Full + neural stars", "Full + neural stars", {"denoise": {"iterations": 6000}}))


def _user_settings() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def _fingerprint(p: Path):
    return (p.stat().st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest()) if p.exists() else None


def _tree_snapshot(folder: Path):
    """Every file of the test project, its store included (the scratch copies read it through a junction)."""
    return sorted((p.relative_to(folder).as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
                  for p in folder.rglob("*") if p.is_file())


USER_SETTINGS = _user_settings()
USER_BEFORE = _fingerprint(USER_SETTINGS)
PROJECT_BEFORE = _tree_snapshot(F_PROJECT) if F_PROJECT.is_dir() else None


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    appmod.apply_language("en")
    shown = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, _k=kind, **k: shown.append((_k, a[1:3])))
    yield {"tmp": tmp_path, "shown": shown}
    appmod.apply_language("en")
    assert _fingerprint(USER_SETTINGS) == USER_BEFORE, "the user's settings file was touched"


def _scratch_project(folder: Path) -> Path:
    """A copy of the test project's definition whose store is a junction onto the real one: the results are read
    (done states, previews), and what the model writes (state.json, the definition) lands in the temporary folder."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(F_PROJECT / "project.astroproj", folder / "project.astroproj")
    for name in ("store", "inputs"):
        if (F_PROJECT / name).is_dir() and not (folder / name).exists():
            try:
                r = subprocess.run(["cmd", "/c", "mklink", "/J", str(folder / name), str(F_PROJECT / name)],
                                   capture_output=True, text=True)
            except OSError as e:
                pytest.skip(f"no directory junction here: {e}")
            if r.returncode:
                pytest.skip(f"no directory junction here: {r.stderr.strip()}")
    return folder


def _open_four(folder: Path | None = None):
    p = api.open_project(folder or F_PROJECT)
    for name, template, params in FOUR:
        api.add_workflow(p, name, template, params)
    return p


@pytest.fixture(scope="module")
def world(qapp, tmp_path_factory):
    folder = _scratch_project(tmp_path_factory.mktemp("graph") / "heart")
    p = _open_four(folder)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert not m.ready                                # the GUI thread did not wait for the disk
    assert w.wait(180) and m.ready, m.error
    yield {"p": p, "w": w, "m": m, "folder": folder}
    m.flush()
    w.stop()


def _events():
    """Process the events, and delete the widgets a rebuild let go of (no event loop runs in the tests, so the
    deferred deletions would otherwise pile up, hidden, under the panel)."""
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def settle(w, page=None, n=12):
    """Process events until the worker is idle and the page has nothing left to build."""
    for _i in range(n):
        w.wait(60)
        _events()
        if page is not None and page._pending:
            _events()
        time.sleep(0.01)
    w.wait(60)
    _events()


def _reset(m: ProjectModel):
    while m.can_undo():
        m.undo()
    m._redo.clear()
    m.scope, m.pending_change, m.current, m.sel_step = None, None, 0, "empilement"
    m.p.stack_mode = "together"
    m.flush()


@pytest.fixture()
def model(world):
    m = world["m"]
    _reset(m)
    yield m
    _reset(m)


@pytest.fixture()
def page(world, model, qapp):
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    v.setContentsMargins(0, 0, 0, 0)
    pg = GraphPage(model)
    v.addWidget(pg)
    root.resize(1680, 934)
    root.show()
    settle(world["w"], pg)
    pg.root = root
    yield pg
    root.close()
    root.deleteLater()
    qapp.processEvents()


def node(page, key, pid=None):
    g = page.graph.graph
    return g.node_of(page.s.view(pid) if pid else page.s.path, key)


def texts_of(root: QWidget) -> list[str]:
    out = []
    for w in [root] + root.findChildren(QWidget):
        if not w.isVisibleTo(root) and w is not root:
            continue
        if isinstance(w, QLabel):
            out.append(re.sub(r"<[^>]+>", "", w.text()))
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
        if w.toolTip():
            out.append(w.toolTip())
    return [t.strip() for t in out if t and t.strip()]


NAMES = {"Full", "Classic stack", "Classic + denoise", "Full + neural stars", "Raw stack"}


def untranslated(texts) -> list[str]:
    """Texts shown in English although a French translation exists (the names the user typed are never translated)."""
    cat = i18n._cat
    return [t for t in texts if t in cat and cat[t] != t and t not in NAMES]


def cut_texts(root: QWidget) -> list[str]:
    """Labels, buttons and check boxes narrower than their text, and scroll areas whose content is wider than them
    (their horizontal bar is off: what passes the right edge is cut)."""
    out = []
    for sa in root.findChildren(QScrollArea):
        inner = sa.widget()
        if sa.isVisibleTo(root) and inner is not None and sa.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff:
            if inner.minimumSizeHint().width() > sa.viewport().width() + 1:
                out.append(f"{type(sa.parent()).__name__}: content {inner.minimumSizeHint().width()} px > "
                           f"{sa.viewport().width()} px")
    for w in root.findChildren(QWidget):
        if not w.isVisibleTo(root):
            continue
        if isinstance(w, QLabel) and not w.wordWrap() and w.text() and (w.pixmap() is None or w.pixmap().isNull()):
            if w.width() + 1 < w.sizeHint().width():
                out.append(w.text())
        elif isinstance(w, (QPushButton, QCheckBox)) and w.text():
            if w.width() + 1 < w.sizeHint().width():
                out.append(w.text())
    return out


def graph_faults(page) -> list[str]:
    """What is wrong in the graph as drawn: a block outside the view or drawn smaller than 80 % (off Show all), a
    title that does not fit its block, a wire that runs behind a block it does not join."""
    gv = page.graph
    out = []
    items = [it for it in gv.scene().items() if isinstance(it, BlockItem)]
    if not gv.b_all.isChecked():
        if gv.transform().m11() < MIN_SCALE - 1e-6:
            out.append(f"scale {gv.transform().m11():.3f}")
        vp = QRectF(gv.viewport().rect())
        for it in items:
            r = QRectF(gv.mapFromScene(it.sceneBoundingRect()).boundingRect())
            if not vp.contains(r):
                out.append(f"{it.title} outside the view")
    f = QFont("Segoe UI", 10)
    f.setBold(True)
    for it in items:
        if it.node.key != "post" and QFontMetricsF(f).horizontalAdvance(it.title) > it.w - 18:
            out.append(f"title {it.title!r} wider than its block")
    rects = [QRectF(it.pos().x(), it.pos().y(), it.w, BH).adjusted(1, 1, -1, -1) for it in items]
    for wire in gv.scene().items():
        if isinstance(wire, QGraphicsPathItem) and wire.zValue() == 1:
            path = wire.path()
            for i in range(1, 200):
                pt = path.pointAtPercent(i / 200)
                hit = next((r for r in rects if r.contains(pt)), None)
                if hit is not None:
                    out.append(f"a wire runs behind the block at x={hit.x():.0f}, y={hit.y():.0f}")
                    break
    return out


def panel_faults(page) -> list[str]:
    """The editors of the panel's setting rows do not all start at the same x."""
    panel = page.panel
    host = panel.findChildren(QScrollArea)[0].widget()
    eds = [w for w in host.findChildren(QWidget) if w.isVisibleTo(panel) and (
        isinstance(w, QAbstractSpinBox) or (isinstance(w, QCheckBox) and not w.text()) or
        (isinstance(w, (QLineEdit, QPushButton)) and isinstance(w.parent(), CompactRow) and w.text() != "↺"))]
    xs = sorted({w.mapTo(panel, QPoint(0, 0)).x() for w in eds})
    return [f"editors at x = {xs}"] if len(xs) > 1 else []


def button(root: QWidget, start: str) -> QPushButton:
    return next(b for b in root.findChildren(QPushButton) if b.text().startswith(start) and b.isVisibleTo(root))


# ---------------------------------------------------------------------------------------------- the tree
def test_the_model_does_not_wait_for_the_disk(qapp):
    """Opening a project is instant: the worker warms the API; keys, states and costs come after readyChanged."""
    p = _open_four()
    w = ApiWorker()
    t0 = time.perf_counter()
    m = ProjectModel(p, w, readonly=True)
    assert time.perf_counter() - t0 < 1.0 and not m.ready
    assert [v.name for v in m.paths] == ["Full", "Classic stack", "Classic + denoise", "Full + neural stars"]
    assert m.all_stages() == [] and m.total_seconds() == 0.0 and m.state(None, "stack") == "pending"
    got = []
    m.readyChanged.connect(lambda: got.append(m.ready))
    assert w.wait(180)
    assert m.ready and got and got[-1]
    assert m.all_stages(m.paths[0])[:4] == ["inventory", "calibrate", "register", "stack"]
    w.stop()


def test_the_tree_shares_blocks_as_the_engine_does(world, model, page):
    """Calibration to Color shared by the 4 workflows, Denoise by 3, a split at Stars, 4 outputs."""
    g = Graph(model)
    by_key = {}
    for n in g.nodes:
        by_key.setdefault(n.key, []).append(n)
    ids = {v.name: v.pid for v in model.paths}
    for key in ("calibration", "alignement", "selection", "empilement", "astrometrie", "couleur"):
        assert len(by_key[key]) == 1 and len(by_key[key][0].paths) == 4, key
    den = by_key["debruitage"]
    assert len(den) == 1 and {q.name for q in den[0].paths} == {"Full", "Classic + denoise", "Full + neural stars"}
    stars = by_key["etoiles"]
    assert len(stars) == 2 and all(len(n.paths) == 1 for n in stars)           # the methods differ
    assert {q.name for n in stars for q in n.paths} == {"Full", "Full + neural stars"}
    assert len(by_key["post"]) == 4 and all(len(n.paths) == 1 for n in by_key["post"])
    assert len(by_key["fond"]) == 4                                            # after the split: one each
    assert g.final.paths and len(g.final.paths) == 4
    # the leaves go to Post, one per workflow
    assert {g.leaves[pid].key for pid in ids.values()} == {"post"}
    assert by_key["debruitage"][0].shared and not stars[0].shared


def test_the_graph_shows_the_real_engine_order(world, model, page):
    """Frames, Calibration, Registration, Selection, Stacking, Astrometry, Color, Denoise, Stars, Gradient, Output,
    Post (DESIGN §2.4); the phase bands; Deconvolution only when something uses it."""
    assert COLUMNS == ["sources", "calibration", "alignement", "selection", "empilement", "astrometrie", "couleur",
                       "deconv", "debruitage", "etoiles", "fond", "post", "final"]
    assert model.block_keys(model.paths[0]) == ["calibration", "alignement", "selection", "empilement", "astrometrie",
                                                "couleur", "debruitage", "etoiles", "fond", "post"]
    items = {it.node.key + str(id(it.node)): it for it in page.graph.scene().items() if isinstance(it, BlockItem)}
    xs = sorted({it.pos().x() for it in items.values()})
    assert len(xs) == 12                                                        # no Deconvolution column
    gradient = [it for it in items.values() if it.node.key == "fond"]
    stars = [it for it in items.values() if it.node.key == "etoiles"]
    assert min(g.pos().x() for g in gradient) > max(s.pos().x() for s in stars)  # Gradient after Stars
    bands = [it for it in page.graph.scene().items() if type(it).__name__ == "BandItem"]
    assert sorted(b.text for b in bands) == ["Linear processing", "Pre-processing"]


def test_states_come_from_the_api(world, model, page):
    api_states = api.states(model.p, update=False)
    for (wid, stage), st in api_states.items():
        assert model.state(model.view(wid), stage) == st.state
    g = Graph(model)
    full = model.paths[0]
    assert full.computed and full.rendered and not model.paths[1].computed
    states = {(n.key, n.paths[0].name): g.state(n) for n in g.nodes if n.paths and n.key not in ("sources", "final")}
    assert states[("empilement", "Full")] == "done"
    assert states[("fond", "Classic stack")] == "pending" and states[("post", "Classic stack")] == "pending"
    assert states[("etoiles", "Full + neural stars")] == "pending" and states[("etoiles", "Full")] == "done"
    assert g.state(g.final) == "pending"                                       # not every layer is rendered
    # a block is as bad as its worst stage in any of its workflows
    model._states[(2, "background_poly")] = api.StageState("failed", "boom", "k")
    assert block_state(model, model.view(2), "fond") == "failed"
    model._states[(2, "background_poly")] = api.StageState("running", "", "k")
    assert block_state(model, model.view(2), "fond") == "running"
    model._sync()
    assert block_state(model, model.view(2), "fond") == "pending"
    assert g.summary(g.root).startswith("1 session(s)") and "40 lights" in g.summary(g.root)


def test_a_shared_stage_is_one_key(world, model):
    full, classic, cd, neural = model.paths
    assert [q.name for q in model.sharing(full, "stack")] == [q.name for q in model.paths]
    assert model.owner(neural, "stack") is full
    assert {q.name for q in model.sharing(full, "denoise")} == {"Full", "Classic + denoise", "Full + neural stars"}
    assert model.stage_key(full, "stack") == api.stage_key(model.p, full.pid, "stack")
    assert model.upstream(full, "color") >= {"inventory", "calibrate", "register", "stack", "astrometry"}
    assert "render_prep" in model.downstream(full, "stars")
    assert [x for _p, x in model.pending_upto(classic, ["render_simple"])] == ["background_poly", "render_simple"]
    assert model.pending(full) == []
    assert model.seconds("stack", full) > 0 and model.seconds("denoise_cl") == 300.0


# ---------------------------------------------------------------------------------------------- option rules
def test_adding_and_removing_blocks_follows_the_rules_with_toasts(world, model, page):
    classic = model.view(2)
    toasts = []
    page.toasted.connect(toasts.append)
    page.add_tool("etoiles", classic.pid)                                      # Stars needs Denoise
    settle(world["w"], page)
    assert classic.opts["etoiles"] == "mesurer" and classic.opts["debruitage"]
    assert any("Denoise added" in t for t in toasts)
    assert model.sel_step == "etoiles" and model.path is classic
    # Denoise cannot go while Stars is there
    toasts.clear()
    page.remove_tool(node(page, "debruitage", classic.pid), [classic.pid])
    assert classic.opts["debruitage"] and toasts and "remove Stars first" in toasts[0]
    # removing Stars turns the H-alpha gradient into the polynomial one, and the neural removal goes too
    model.set_opts(classic, {**classic.opts, "fond": "halpha", "appris": True})
    toasts.clear()
    page.remove_tool(node(page, "etoiles", classic.pid), [classic.pid])
    assert classic.opts["etoiles"] == "garder" and classic.opts["fond"] == "simple" and not classic.opts["appris"]
    assert len(toasts) == 1 and toasts[0].count(" · ") == 1
    # Gradient cannot be removed, and the card after the output says so
    new, why = model.without_tool(classic.opts, "simple")
    assert new is None and why


def test_the_toolbar_cards_say_what_each_block_costs(world, model, page):
    from astro_gpu.client.graph import ToolCard
    cards = {c.key: c for c in page.findChildren(ToolCard)}
    assert set(cards) == {"debruitage", "etoiles"}                             # Deconvolution is ◇: Spec mode only
    assert all(c.state == "in" for c in cards.values())                        # Full has both
    model.select(index=1)                                                      # Classic stack
    page.refresh()
    cards = {c.key: c for c in page.findChildren(ToolCard)}
    assert cards["debruitage"].state == "add" and cards["etoiles"].state == "add"
    texts = [lb.text() for c in cards.values() for lb in c.findChildren(QLabel)]
    assert any(t.startswith("+") and "h" in t or t.startswith("+") for t in texts)
    assert hm(model.tool_cost(model.path.opts, "etoiles")) in " ".join(texts) or "same duration" in " ".join(texts)
    assert page.findChildren(QLabel, "h2")[0].text() == "Add step"


# ---------------------------------------------------------------------------------------------- replace / variant
def _edit_kappa(page, value):
    """Edit Stacking's rejection threshold the way the user does: in the row of the panel."""
    settle(page.s.worker, page, 6)
    page.panel.expert = True
    page.panel.refresh(page.panel.node)
    _events()
    row = next(r for r in live_rows(page) if r.r["key"] == "stack.kappa")
    page.panel.set_setting(row.r, value)
    for _i in range(3):
        _events()                                     # the page builds itself again, with the choice box
    return row.r


def live_rows(page) -> list[CompactRow]:
    """The setting rows the panel shows now (not the ones a rebuild hid before deleting them)."""
    return [r for r in page.panel.findChildren(CompactRow)
            if not (r.isHidden() and r.testAttribute(Qt.WA_WState_ExplicitShowHide))]


def method_cards(page):
    return [c for c in page.panel.findChildren(MethodCard) if c.isVisibleTo(page.panel)]


def test_a_change_on_a_shared_done_block_offers_replace_or_variant(world, model, page):
    page.select_node(node(page, "empilement"))
    settle(world["w"], page)
    assert model.scope == [v.pid for v in model.paths]
    r = _edit_kappa(page, 4.0)
    settle(world["w"], page)
    ch = model.pending_change
    assert ch and ch["step"] == "empilement" and ch["value"] == 4.0 and len(ch["pids"]) == 4
    rep = button(page.panel, "Replace")
    var = button(page.panel, "Try as variant")
    paths = [q for q in model.paths if q.pid in ch["pids"]]
    c_all = model.cost_of_change(paths, ch)
    c_one = model.cost_of_change([model.path], ch)
    assert c_all.seconds > c_one.seconds > 0                                     # the scope of 4 costs more than Full
    assert hm(c_all.seconds) in rep.text() and hm(c_one.seconds) in var.text()
    assert c_all.seconds == api.cost(model.p, [api.with_change(model.p, q.pid, {"kind": "param", "stage": "stack",
                                             "name": "kappa", "value": 4.0}) for q in paths]).seconds
    # nothing was changed yet
    assert all("kappa" not in (q.params.get("stack") or {}) for q in model.paths)
    button(page.panel, "Cancel").click()
    assert model.pending_change is None and r["value"] == 5.0
    # Replace: the change goes to the scope, their results downstream are Outdated
    _edit_kappa(page, 4.0)
    button(page.panel, "Replace").click()
    settle(world["w"], page)
    assert all(q.params["stack"]["kappa"] == 4.0 for q in model.paths)
    assert model.state(model.paths[0], "stack") == "stale" and model.state(model.paths[0], "render_prep") == "stale"
    assert model.state(model.paths[0], "register") == "done"
    assert model.paths[0].computed is False and model.paths[0].rendered                       # a previous result stays
    g = Graph(model)
    assert g.state(node(page, "empilement")) == "stale"
    # back to the suggestion: the old results are Up to date again, nothing runs (5.1)
    model.undo()
    settle(world["w"], page)
    assert model.paths[0].computed and model.state(model.paths[0], "stack") == "done"
    assert "kappa" not in (model.paths[0].params.get("stack") or {})


def test_try_as_variant_keeps_the_original_and_shares_the_blocks_before(world, model, page):
    page.select_node(node(page, "empilement", 1))
    _edit_kappa(page, 3.5)
    n_before = len(model.paths)
    button(page.panel, "Try as variant").click()
    settle(world["w"], page)
    assert len(model.paths) == n_before + 1
    q = model.paths[-1]
    full = model.view(1)
    assert q.variant_of == full.pid and q.name.startswith("Full · ") and "3,5" not in q.name
    assert q.params["stack"]["kappa"] == 3.5 and "stack" not in full.params or "kappa" not in full.params.get("stack", {})
    assert model.path is q and model.p.current == q.pid
    g = Graph(model)
    for key in ("calibration", "alignement", "selection"):                      # shared up to the changed block
        assert g.node_of(q, key) is g.node_of(full, key)
    assert g.node_of(q, "empilement") is not g.node_of(full, "empilement")
    # the panel says what it is, with Keep and Discard
    page.select_node(g.node_of(q, "empilement"))
    settle(world["w"], page)
    assert any("Variant of" in t for t in texts_of(page.panel))
    assert button(page.panel, "Keep") and button(page.panel, "Discard")
    button(page.panel, "Discard").click()
    settle(world["w"], page)
    assert len(model.paths) == n_before and model.path.pid == full.pid
    # Keep: the variant replaces its original and takes its name
    page.select_node(node(page, "empilement", 1))
    _edit_kappa(page, 3.5)
    button(page.panel, "Try as variant").click()
    settle(world["w"], page)
    q = model.paths[-1]
    page.select_node(Graph(model).node_of(q, "empilement"))
    settle(world["w"], page)
    button(page.panel, "Keep").click()
    settle(world["w"], page)
    assert len(model.paths) == n_before and model.path.name == "Full" and model.path.params["stack"]["kappa"] == 3.5
    assert model.path.variant_of is None and model.view(1) is None


def test_a_method_card_asks_the_same_question(world, model, page):
    """Gradient's method card follows the rule too (SPEC §5.2): on a computed Gradient it offers Replace / variant."""
    page.select_node(node(page, "fond", 1))
    settle(world["w"], page)
    cards = method_cards(page)
    assert [c.ok for c in cards] == [True, True]
    other = next(c for c in cards if "Polynomial" in c.findChildren(QLabel)[0].text())
    other.clicked.emit()
    settle(world["w"], page)
    assert model.pending_change and model.pending_change["key"] == "fond"
    assert "Method: Polynomial" in model.pending_change["what"]
    var = button(page.panel, "Try as variant")
    var.click()
    settle(world["w"], page)
    q = model.paths[-1]
    assert q.opts["fond"] == "simple" and q.opts["etoiles"] == "mesurer" and q.variant_of == 1
    assert q.name == "Full · polynomial"


def test_undo_and_redo_restore_the_workflows(world, model, page):
    n0 = len(model.paths)
    p = model.add_path("Extra", {"debruitage": False, "etoiles": "garder", "appris": False, "fond": "simple"})
    assert len(model.paths) == n0 + 1 and model.can_undo()
    model.rename(p.pid, "Renamed")
    model.set_param(model.view(1), "stack", "kappa", 4.0)
    settle(world["w"], page)
    assert model.view(1).params["stack"]["kappa"] == 4.0
    assert model.undo() and "stack" not in model.view(1).params
    assert model.undo() and model.view(p.pid).name == "Extra"
    assert model.undo() and len(model.paths) == n0
    assert model.redo() and len(model.paths) == n0 + 1
    assert model.redo() and model.view(p.pid).name == "Renamed"
    assert model.redo() and model.view(1).params["stack"]["kappa"] == 4.0
    assert not model.redo()
    assert model.undo() and model.can_redo()
    # one batch is one undo step
    with model.batch():
        model.rename(p.pid, "A")
        model.rename(p.pid, "B")
    assert model.view(p.pid).name == "B" and not model.can_redo()
    assert model.undo() and model.view(p.pid).name == "Renamed"
    # an edit that changes nothing is not an undo step
    depth = len(model._undo)
    model.rename(p.pid, "Renamed")
    assert len(model._undo) == depth
    # the computed results are never removed: a deleted workflow comes back with its states
    full = model.view(1)
    model.remove_path(model.index_of(1))
    assert model.view(1) is None
    model.undo()
    assert model.view(1) is not None and model.view(1).computed and full.pid == 1


def test_outputs_pulled_to_post_make_a_raw_stack_workflow(world, model, page):
    n0 = len(model.paths)
    stacking = node(page, "empilement")
    item = next(it for it in page.graph.scene().items() if isinstance(it, BlockItem) and it.node is stacking)
    assert item.port and not any(it.port for it in page.graph.scene().items()
                                 if isinstance(it, BlockItem) and it.node.key != "empilement")
    # drag the port onto Post, as the user does
    page.graph.start_wire(item)
    post = page.graph.final_item.sceneBoundingRect().center()
    page.graph.move_wire(post)
    assert page.graph.final_item.target
    page.graph.end_wire(post)
    settle(world["w"], page)
    assert len(model.paths) == n0 + 1
    q = model.paths[-1]
    assert q.name == "Raw stack" and q.end == "empilement" and model.path is q and model.sel_step == "post"
    assert model.block_keys(q) == ["calibration", "alignement", "selection", "empilement", "post"]
    assert model.all_stages(q)[-1] == "render_simple"
    g = Graph(model)
    assert g.summary(g.node_of(q, "post")) == "from Stacking"
    assert g.node_of(q, "empilement") is g.node_of(model.view(1), "empilement")      # shares everything up to Stacking
    assert any("Raw stack" in t for t in texts_of(page.graph))
    # no block can be added after an early output
    from astro_gpu.client.graph import ToolCard
    assert all(c.state == "in" for c in page.findChildren(ToolCard))
    assert [c.text() for c in page.findChildren(QLabel) if c.text() == "after the output"]
    ok, why, need = model.slider_state(q, "stars_number")
    assert ok is False and "before the Stars block" in why and need is None             # nothing to add after the exit
    assert model.slider_state(model.view(1), "stars_number")[0] is True


def test_new_workflow_duplicate_rename_delete(world, model, page, monkeypatch):
    from astro_gpu.client import graph as graphmod

    class FakeDialog:
        def __init__(self, session, start, parent):
            self.name = type("N", (), {"text": lambda s: "Mine"})()
            self.cur, self.opts, self.params = 0, dict(session.paths[0].opts), {}

        def exec(self):
            return True
    monkeypatch.setattr(graphmod, "NewWorkflowDialog", FakeDialog)
    n0 = len(model.paths)
    page.new_workflow()
    settle(world["w"], page)
    assert len(model.paths) == n0 + 1 and model.path.name == "Mine" and model.sel_step == "post"
    # Duplicate keeps the options, the settings and the exit
    src = model.view(1)
    FakeDialog.__init__ = lambda s, session, start, parent: (setattr(s, "name", type("N", (), {"text": lambda z: "Full copy"})()),
                                                              setattr(s, "cur", 3 + session.paths.index(start)),
                                                              setattr(s, "opts", dict(start.opts)), setattr(s, "params", {}))[0]
    page.workflow_action("duplicate", src.pid)
    settle(world["w"], page)
    cp = model.path
    assert cp.name == "Full copy" and cp.opts == src.opts and cp.params == src.params
    # Rename and Delete (confirmed)
    page.rename(cp.pid, "Other")
    assert model.view(cp.pid).name == "Other"
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    page.workflow_action("delete", cp.pid)
    settle(world["w"], page)
    assert model.view(cp.pid) is None and len(model.paths) == n0 + 1
    assert model.sel_step == "sources"
    assert any("deleted" in h[1] for h in model.history)


def test_clicking_selects_and_sets_the_scope(world, model, page):
    n = node(page, "debruitage")
    page.select_node(n)
    settle(world["w"], page)
    assert model.sel_step == "debruitage" and model.scope == [v.pid for v in n.paths] and len(n.paths) == 3
    assert model.path in n.paths
    n = node(page, "etoiles", 4)
    page.select_node(n)
    assert model.path.pid == 4 and model.scope == [4]
    page.select_node(page.graph.graph.final)
    assert model.sel_step == "final" and model.scope is None
    settle(world["w"], page)
    assert "Layers" in texts_of(page.panel) and "Pending" in " ".join(texts_of(page.panel))
    opened = []
    page.openBlock.connect(lambda: opened.append(1))
    page.graph.blockOpened.emit(n)
    assert opened


# ---------------------------------------------------------------------------------------------- the panel
def test_the_panel_lists_the_settings_of_the_engine(world, model, page):
    page.select_node(node(page, "empilement"))
    settle(world["w"], page)
    rows = {r.r["key"]: r for r in live_rows(page)}
    assert {"stack.scale", "stack.pixfrac"} <= set(rows)
    sc = rows["stack.scale"].r
    assert sc["suggested"] == 1.5 and "FWHM" in sc["reason"] and sc["min"] == 1.0 and sc["max"] == 3.0
    assert sc["unit"] == "×" and not sc["changed"] and sc["type"] == "float"
    # advanced settings: the others, from the engine, nothing left out
    page.panel.expert = True
    page.panel.refresh(page.panel.node)
    keys = {r.r["key"] for r in live_rows(page)}
    assert {"stack.kappa", "stack.gain", "stack.half_block", "stack.read_noise"} <= keys
    assert "stack.exclude" not in keys and "stack.min_transparency" not in keys    # set by the project / Selection
    # the Details of a clicked setting: range bar, "Suggested: v · reason", help
    rows = {r.r["key"]: r for r in live_rows(page)}
    rows["stack.scale"].focused.emit(rows["stack.scale"].r)                          # a click on the row
    dt = texts_of(page.panel.details)
    assert any(t.startswith("Suggested: 1.5 · FWHM 4.0 px with 40 frames") for t in dt)
    page.panel.expert = False


def test_a_settings_row_changes_marks_and_goes_back_to_the_suggestion(world, model, page):
    page.select_node(node(page, "debruitage"))
    settle(world["w"], page)
    rows = {r.r["key"]: r for r in live_rows(page)}
    it = rows["denoise.iterations"]
    assert it.r["changed"] and it.r["value"] == 6000 and it.r["suggested"] == 24000      # Full has 6000 stored
    back = [b for b in it.findChildren(QPushButton) if b.text() == "↺"][0]
    assert not back.isHidden()
    assert "24000" in back.toolTip() or "24 000" in back.toolTip() or "24,000" in back.toolTip()
    name = it.findChildren(QLabel)[0]
    assert "e6a23c" in name.styleSheet().lower()                                          # the name of a changed row
    back.click()                                                                          # ↺: back to the suggestion
    settle(world["w"], page)
    ch = model.pending_change                       # the denoise is done and shared by 3: Replace / variant is asked
    assert ch and ch["kind"] == "param" and ch["value"] is None and len(ch["pids"]) == 3
    button(page.panel, "Replace").click()
    settle(world["w"], page)
    assert "iterations" not in (model.view(1).params.get("denoise") or {})              # the stored value is gone
    rows = {r.r["key"]: r for r in live_rows(page)}
    assert not rows["denoise.iterations"].r["changed"] and rows["denoise.iterations"].r["value"] == 24000
    assert model.state(model.view(1), "denoise") == "stale"                               # 24000 steps: to compute again
    model.undo()
    settle(world["w"], page)
    assert model.state(model.view(1), "denoise") == "done"


def test_selection_thresholds_come_from_the_stack_settings(world, model, page):
    model.scope = None
    page.select_node(node(page, "selection"))
    settle(world["w"], page)
    t, f = model.thresholds_now()
    assert t == 60 and abs(f - 2.0 * 3.95) < 1e-6 and model.thresholds_unit() == "px"
    st, sf = model.suggested_thresholds()
    assert st == 60 and abs(sf - f) < 1e-6
    assert model.kept_frames() == 40 and any("40 of 40 frames kept" in t for t in texts_of(page.panel))
    # a change goes through the same choice (the stack is done)
    page.panel.propose_selection(70, f)
    settle(world["w"], page)
    assert model.pending_change and model.pending_change["kind"] == "params"
    assert model.pending_change["items"][0] == ("stack", "min_transparency", 0.7)
    button(page.panel, "Cancel").click()
    model.request_selection(70, f)
    settle(world["w"], page)
    assert model.kept_frames() <= 40 and model.selection_total() == 40
    assert "transparency ≥ 60 % · stars ≤ 7.9 px" in Graph(model).summary(node(page, "selection"))


# ---------------------------------------------------------------------------------------------- Spec mode
def test_spec_mode_shows_what_the_engine_does_not_have_disabled(world, model, page):
    from astro_gpu.client.graph import ToolCard
    page.select_node(node(page, "debruitage"))
    settle(world["w"], page)
    plain = texts_of(page.panel)
    assert not any("◇" in t for t in plain)
    assert len(method_cards(page)) == 1                                        # Noise2Noise only
    prefs.set("spec_mode", True)
    page.refresh()
    settle(world["w"], page)
    cards = method_cards(page)
    assert len(cards) == 4 and [c.ok for c in cards] == [True, False, False, False]
    assert any("◇" in "".join(lb.text() for lb in c.findChildren(QLabel)) for c in cards[1:])
    page.panel.expert = False
    rows = live_rows(page)
    new_rows = [r for r in rows if r.r["kind"] == "new"]
    assert new_rows and not [b for r in new_rows for b in r.findChildren(QPushButton)
                             if b.text() == "↺" and b.isVisibleTo(r)]           # never changed: no ↺
    for r in new_rows:
        editors = r.findChildren(QComboBox) + r.findChildren(QCheckBox) + r.findChildren(QDoubleSpinBox) + \
            r.findChildren(QSpinBox)
        assert editors and not any(e.isEnabled() for e in editors), r.r["key"]
    tc = {c.key: c for c in page.findChildren(ToolCard)}
    assert "deconv" in tc and not tc["deconv"].isEnabled() and tc["deconv"].state == "off"
    assert "◇" in " ".join(lb.text() for lb in tc["deconv"].findChildren(QLabel))
    # a click on a ◇ method changes nothing
    cards[1].clicked.emit() if cards[1].ok else None
    assert model.pending_change is None
    # the Gradient block says the reorder is for later
    page.select_node(node(page, "fond"))
    settle(world["w"], page)
    assert any("Gradient before Color" in t for t in texts_of(page.panel))
    # a ◇ value is never stored and never changes a key
    keys_before = {s: model.stage_key(model.paths[0], s) for s in model.all_stages(model.paths[0])}
    page.panel.set_setting(next(r for r in new_rows).r, 1)
    assert keys_before == {s: model.stage_key(model.paths[0], s) for s in model.all_stages(model.paths[0])}
    assert model.pending_change is None and not model.can_undo()


# ---------------------------------------------------------------------------------------------- language
def _visit_everything(page, w):
    g = page.graph.graph
    for n in list(g.nodes):
        page.select_node(n)
        settle(w, page, 6)


def test_the_french_switch_translates_everything_visible(world, model, page):
    prefs.set("spec_mode", True)
    appmod.apply_language("fr")
    i18n.missing()
    page.graph.b_all.setText(i18n._("Show all"))
    page.refresh()
    settle(world["w"], page)
    _visit_everything(page, world["w"])
    page.graph.show_toast(i18n._("+ {t} to run").format(t="3 min"))
    page.select_node(node(page, "empilement"))
    _edit_kappa(page, 4.0)                                                    # the choice box, in French
    settle(world["w"], page)
    t = texts_of(page.root)
    assert not untranslated(t), untranslated(t)
    assert not i18n.missing(), sorted(i18n.missing())
    assert "Remplacer" in " ".join(t) and "Essayer en variante" in " ".join(t)
    assert any(x.startswith("Suggéré : ") for x in t)                        # the Details of the setting
    button(page.panel, "Annuler").click()
    # block names, states, summaries and the toolbar in French
    g = page.graph.graph
    tip = " ".join(it.toolTip() for it in page.graph.scene().items() if isinstance(it, BlockItem))
    assert "Empilement" in tip and "À jour" in tip and "À calculer" in tip
    assert i18n._("Gradient") == "Gradient"
    assert g.summary(next(n for n in g.nodes if n.key == "alignement")) == "sur l'image la plus nette"
    assert "Ajouter une étape" in " ".join(t)
    # back to English at runtime: the page is built again, in English
    appmod.apply_language("en")
    page.refresh()
    settle(world["w"], page)
    assert "Add step" in texts_of(page.root)


def test_decimal_commas_in_french_reasons(world, model, page):
    appmod.apply_language("fr")
    page.select_node(node(page, "empilement"))
    settle(world["w"], page)
    r = next(x for x in live_rows(page) if x.r["key"] == "stack.scale").r
    assert "4,0 px" in r["reason"] and "." not in r["reason"].split("px")[0].split("FWHM")[-1]


# ---------------------------------------------------------------------------------------------- GUI thread
def test_nothing_slow_runs_in_the_gui_thread(qapp, monkeypatch):
    """Disk scans, describe, outputs, previews and warm run in the worker thread; the keys, states and costs the GUI
    asks for afterwards read caches."""
    from astro_gpu.core import store
    calls = []

    def watch(mod, name):
        orig = getattr(mod, name)

        def f(*a, **k):
            calls.append((name, threading.current_thread() is threading.main_thread()))
            return orig(*a, **k)
        monkeypatch.setattr(mod, name, f)
    for name in ("warm", "describe", "stage_outputs", "selection_preview", "frame_table", "scan_folder"):
        watch(api, name)
    for name in ("list_frames", "file_id"):
        watch(store, name)
    p = _open_four()
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    pg = GraphPage(m)
    v.addWidget(pg)
    root.resize(1680, 934)
    root.show()
    assert w.wait(180)
    settle(w, pg)
    assert api._MACHINE and any(c == "warm" and not main for c, main in calls)   # torch was imported by the worker
    calls.clear()                                                            # now: what the GUI thread does alone
    for n in list(pg.graph.graph.nodes):
        pg.select_node(n)
        QApplication.processEvents()
    pg.select_node(node(pg, "empilement"))
    settle(w, pg)
    _edit_kappa(pg, 4.0)
    button(pg.panel, "Try as variant").click()
    m.add_path("X", {"debruitage": True, "etoiles": "mesurer", "appris": True, "fond": "halpha"})
    m.undo()
    m.cost()
    m.all_pending()
    m.states_probe = None
    settle(w, pg)
    main_calls = [c for c, main in calls if main]
    assert not main_calls, main_calls
    assert any(c in ("describe",) for c, main in calls if not main)           # the worker answered the panel
    assert not any(main for _c, main in calls)
    w.stop()
    root.close()


def test_autosave_writes_only_the_definition_and_flushes_on_close(qapp, tmp_path):
    root = tmp_path / "copy"
    root.mkdir()
    shutil.copy(F_PROJECT / "project.astroproj", root / "project.astroproj")
    p = api.open_project(root)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(180)
    before = (root / "project.astroproj").read_bytes()
    m.add_path("Autosaved", {"debruitage": True, "etoiles": "garder", "appris": False, "fond": "simple"})
    assert (root / "project.astroproj").read_bytes() == before                  # debounced: not yet
    time.sleep(0.45)
    qapp.processEvents()
    assert "Autosaved" in (root / "project.astroproj").read_text(encoding="utf-8")
    m.rename(m.paths[-1].pid, "Last one")
    m.flush()                                                                    # Save flushes at once
    assert [x.name for x in api.open_project(root).workflows][-1] == "Last one"
    m.select(index=0)
    m.close()
    names = {f.name for f in root.iterdir()}
    assert names <= {"project.astroproj", "state.json", "logs", "cache", "store", "inputs", "scratch"}, names
    w.stop()


# ---------------------------------------------------------------------------------------------- screenshots
def _shots_dir(tmp_path) -> Path:
    d = Path(os.environ.get("ASTRO_GPU_SHOTS") or tmp_path / "shots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_shots(world, model, page, tmp_path):
    """graph, graph_all, panel_shared (Stacking, Denoise), panel_stars, variant and spec, in English and in French:
    no text cut, no English left in French, no overlap of the blocks."""
    out = _shots_dir(tmp_path)
    w = world["w"]
    cut = {}

    def shot(name):
        settle(w, page, 8)
        page.root.grab().save(str(out / f"{name}.png"))
        assert (out / f"{name}.png").stat().st_size > 5000
        c = cut_texts(page.root) + graph_faults(page) + panel_faults(page)
        if c:
            cut[name] = c
        if i18n.language() == "fr":
            assert not untranslated(texts_of(page.root)), (name, untranslated(texts_of(page.root)))

    for lang in ("en", "fr"):
        appmod.apply_language(lang)
        _reset(model)
        prefs.set("spec_mode", False)
        page.graph.toast.hide()                          # a word said in the other language
        page.panel.expert, page.panel.focus_key = False, None
        page.refresh()
        page.graph.b_all.setText(i18n._("Show all"))
        shot(f"graph_{lang}")
        page.graph.b_all.setChecked(True)
        shot(f"graph_all_{lang}")
        page.graph.b_all.setChecked(False)
        for key, name in (("empilement", "panel_shared"), ("debruitage", "panel_denoise"), ("etoiles", "panel_stars"),
                          ("selection", "panel_selection"), ("fond", "panel_gradient")):
            page.select_node(node(page, key))
            shot(f"{name}_{lang}")
        # the variant choice
        page.select_node(node(page, "empilement"))
        settle(w, page)
        _edit_kappa(page, 4.0)
        shot(f"variant_{lang}")
        button(page.panel, i18n._("Try as variant")).click()
        assert model.paths[-1].name == f"Full · {i18n._('Rejection threshold').lower()} 4"   # named in the language
        page.select_node(Graph(model).node_of(model.paths[-1], "empilement"))
        shot(f"variant_made_{lang}")
        _reset(model)
        page.graph.toast.hide()
        page.panel.expert, page.panel.focus_key = False, None
        # Spec mode
        prefs.set("spec_mode", True)
        page.panel.expert = False
        page.select_node(node(page, "debruitage"))
        page.refresh()
        shot(f"spec_denoise_{lang}")
        page.select_node(node(page, "calibration"))
        shot(f"spec_calibration_{lang}")
        prefs.set("spec_mode", False)
        page.refresh()
    assert not cut, cut
    # no block overlaps another one
    items = [it for it in page.graph.scene().items() if isinstance(it, BlockItem)]
    rects = [it.sceneBoundingRect().adjusted(2, 2, -2, -2) for it in items]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not a.intersects(b)
    assert BW > 0 and BH > 0 and QPointF


# ---------------------------------------------------------------------------------------------- more of the page
def test_the_page_says_it_is_reading_while_the_worker_works(qapp):
    p = _open_four()
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    pg = GraphPage(m)
    v.addWidget(pg)
    root.resize(1680, 934)
    root.show()
    qapp.processEvents()
    assert pg.graph.wait.isVisible() and pg.graph.wait.text() == "Reading the project…"
    assert not any(isinstance(it, BlockItem) and it.node.key == "calibration" for it in pg.graph.scene().items())
    assert w.wait(180)
    settle(w, pg)
    assert not pg.graph.wait.isVisible()
    assert any(isinstance(it, BlockItem) and it.node.key == "calibration" for it in pg.graph.scene().items())
    appmod.apply_language("fr")
    pg.graph.show_wait()
    m.error = "boom"
    pg.graph.show_wait()
    assert pg.graph.wait.isVisible() and "boom" in pg.graph.wait.text() and "ne peut pas" in pg.graph.wait.text()
    m.error = ""
    w.stop()
    root.close()


def test_dropping_a_tool_on_a_lane_adds_it_where_the_rules_say(world, model, page):
    graph = page.graph
    graph.drag_tool = "etoiles"                                  # a card of the toolbar is being dragged
    graph.rebuild()
    assert {g.pid for g in graph.ghosts} == {2, 3}               # the lanes that do not have Stars yet
    graph.drag_tool = None
    graph.rebuild()
    toasts = []
    page.toasted.connect(toasts.append)
    graph.toolDropped.emit("etoiles", 3)                         # on Classic + denoise
    settle(world["w"], page)
    assert model.view(3).opts["etoiles"] == "mesurer" and model.sel_step == "etoiles" and model.path.pid == 3
    assert model.scope == [1, 3]                                 # its Stars are the same as Full's: one block, two workflows
    assert not toasts or "Denoise added" not in toasts[0]        # it had Denoise
    graph.toolDropped.emit("etoiles", 2)                         # on Classic stack: Denoise comes with it
    settle(world["w"], page)
    assert model.view(2).opts["debruitage"] and any("Denoise added" in t for t in toasts)
    graph.toolDropped.emit("deconv", 2)                          # ◇: nothing happens, and it says so
    assert any(t.startswith("◇") for t in toasts)
    assert model.undo() and model.undo()


def test_a_project_without_workflows_has_a_graph_and_an_import_button(qapp, tmp_path):
    p = project2.create(tmp_path / "P", "P", Target(name="M 1"))
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(120)
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    pg = GraphPage(m)
    v.addWidget(pg)
    root.resize(1680, 934)
    root.show()
    settle(w, pg)
    items = [it for it in pg.graph.scene().items() if isinstance(it, BlockItem)]
    assert [it.node.key for it in items] == ["sources", "final"] or {it.node.key for it in items} == {"sources", "final"}
    assert "+  New workflow" in [getattr(g, "text", "") for g in pg.graph.scene().items()]
    got = []
    pg.addSources.connect(lambda: got.append(1))
    pg.select_node(pg.graph.graph.root)
    settle(w, pg)
    button(pg.panel, "Import folder").click()
    assert got
    assert [lb.text() for lb in pg.findChildren(QLabel) if lb.text() == "Add step"]
    assert m.total_seconds() == 0 and m.all_pending() == []
    w.stop()
    root.close()


def test_a_change_of_an_automatic_setting_is_seen_by_the_what_if(world, model, page):
    """The API keys what-if workflows by object and has already asked the keys of the copy it returns (the
    suggestion of an automatic setting needs them): the model copies it again, so that scale, min_transparency, the
    iterations... are seen to change a result (they used to give the old key, and 0 s)."""
    import copy
    full = model.view(1)
    stale = api.with_change(model.p, 1, {"kind": "param", "stage": "stack", "name": "scale", "value": 2.0})
    mine = model._changed(full, {"kind": "param", "stage": "stack", "name": "scale", "value": 2.0})
    assert api.stage_key(model.p, mine, "stack") != model.stage_key(full, "stack")
    assert api.stage_key(model.p, copy.deepcopy(stale), "stack") == api.stage_key(model.p, mine, "stack")
    for name, value in (("scale", 2.0), ("half_block", 20), ("min_transparency", 0.7)):
        ch = {"kind": "param", "stage": "stack", "name": name, "value": value}
        assert model.throws_away(model.paths, ch), name
        assert model.cost_with([full], ch) > 0, name
    ch = {"kind": "param", "stage": "denoise", "name": "iterations", "value": 8000}
    assert model.throws_away([full], ch) and model.cost_with([full], ch) > 0
    # an invalid value is refused with the name of the setting, and nothing changes
    toasts = []
    page.toasted.connect(toasts.append)
    page.select_node(node(page, "empilement"))
    settle(world["w"], page)
    page.panel.propose("stack", "scale", 9.0, 1.5, "Drizzle scale: 1.5 → 9", "scale 9")
    assert toasts and "stack.scale" in toasts[0] and model.pending_change is None and not model.can_undo()
    assert model.view(1).params.get("stack", {}).get("scale") is None


def test_no_setting_of_the_engine_is_out_of_reach(world, model, page):
    """Every setting a stage describes (but the ones the project or the options own) has a row in the panel of its
    block, in some workflow; no row is listed twice in a panel."""
    from astro_gpu.client.settings import HIDDEN
    want = set()
    for v in model.paths:
        for stage in model.all_stages(v):
            for info in api.describe(model.p, v.pid, stage):
                if info.level != "internal" and (stage, info.name) not in HIDDEN:
                    want.add(f"{stage}.{info.name}")
    got = set()
    page.panel.expert = True
    for v in model.paths:
        for key in model.block_keys(v) + ["demi_piles"]:
            n = Graph(model).node_of(v, key) if key != "demi_piles" else None
            if n is None:
                continue
            page.select_node(n)
            settle(world["w"], page, 5)
            keys = [r.r["key"] for r in live_rows(page)]
            assert len(keys) == len(set(keys)), (v.name, key, keys)
            got |= set(keys)
    page.panel.expert = False
    assert not (want - got), sorted(want - got)


# ---------------------------------------------------------------------------------------------- last
def test_the_test_project_is_untouched():
    """Run last (pytest keeps the order of the file): nothing above wrote into the test project or its store."""
    assert _tree_snapshot(F_PROJECT) == PROJECT_BEFORE
