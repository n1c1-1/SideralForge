"""The whole client, screen by screen, EN then FR, offscreen, CPU only, on a COMPUTED bench project opened READ-ONLY:
the screenshots the orchestrator looks at before the user's review.

    python tools/shots_client.py PROJECT OUT [--parts home,graph,stack,inspect,post,export] [--drizzle NAME]
                                             [--classic NAME] [--lang en,fr]

PROJECT: F:/astro_essais/bench/<case>/v2x_express (never F:/astro_projets). ``--drizzle`` / ``--classic``: the
workflows of today's two methods (default PI-own and CL, the bench's workflows with today's defaults).

- Home: the sections and the bench's projects;
- Graph: the blocks, a block hovered (its tooltip painted where Qt shows it), a breakpoint (set, then taken off);
- Stacking settings: the two method cards, the Recommended frame with the tooltip of its "i", each method's settings,
  the Selection block's rule sentence;
- Inspect: every tab of every block of the drizzle workflow, the stacking tabs of the classic one, the frames (sessions
  and browser), the stack's histogram (the stretch), side-by-side compares (Gradient, Colour, Astrometry);
- Post: the Curves editor (a manual point, each channel, a classic tone preset, an astro preset, a user preset saved in
  the scratch settings folder), every new filter added as a layer with its sliders moved off their defaults, the filter
  menu (its 6 groups);
- Export: FITS, TIFF, PNG and JPEG written into a scratch folder.

Nothing is written into the project: the model is read-only (no definition, no state.json), the Post composition stays
in memory, the user's settings and ~/.astro_gpu are scratch ones, the exports go to a scratch folder; the project's
files (all but the store) are fingerprinted before and after, a difference is reported as a problem.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import client_walkthrough as cw  # noqa: E402


def arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    raise SystemExit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
project, out = Path(sys.argv[1]), Path(sys.argv[2])
assert "astro_projets" not in project.resolve().as_posix().lower(), "never the user's projects"
PARTS = arg("--parts", "home,graph,stack,inspect,post,export").split(",")
LANGS = arg("--lang", "en,fr").split(",")
DRIZZLE, CLASSIC = arg("--drizzle", "PI-own"), arg("--classic", "CL")
out.mkdir(parents=True, exist_ok=True)


def fingerprint(root: Path):
    return sorted((f.relative_to(root).as_posix(), f.stat().st_size, f.stat().st_mtime_ns)
                  for f in root.rglob("*") if f.is_file() and "store" not in f.relative_to(root).parts[:1])


BEFORE = fingerprint(project)
tmp = Path(tempfile.mkdtemp(prefix="shots_client_"))
cw.setup_environment(tmp)
EXPORTS = tmp / "exports"

if "--trace-wcs" in sys.argv:                            # who builds a WCS from a header-only file
    import traceback as _tb

    import astropy.wcs as _aw
    _init = _aw.WCS.__init__

    def _traced(self, header=None, *a, **k):
        if header is not None and not isinstance(header, (str, Path)) and hasattr(header, "get") and \
                header.get("NAXIS") == 0:
            print("WCS from a NAXIS=0 header:\n" + "".join(_tb.format_stack(limit=8)), flush=True)
        return _init(self, header, *a, **k)
    _aw.WCS.__init__ = _traced
    _aw.wcs.WCS.__init__ = _traced
walk = cw.Walk(out, tmp / "ws", True, tmp)
w = walk.w
w.readonly = True
from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtGui import QPainter  # noqa: E402

from astro_gpu.client.i18n import _  # noqa: E402

# Home: the recent projects and the default sections (the bench, the test copies)
T0 = time.time()
LOG = []


def note(text):
    LOG.append(text)
    print("  NOTE:", text, flush=True)


def shot_with_tip(name, widget, tip_text, at_widget=None, root=None):
    """``root`` (default the window) grabbed with a tooltip painted under ``widget``."""
    from PySide6.QtWidgets import QLabel
    root = root or w
    walk.settle()
    walk.n += 1
    f = out / f"{walk.n:02d}_{name}_{walk.lang}.png"
    img = root.grab()
    lab = QLabel(tip_text)
    lab.setStyleSheet("QLabel { background: #2b2f38; color: #e6e6e6; border: 1px solid #555; padding: 4px; }")
    lab.setWordWrap(True)
    lab.setMaximumWidth(420)
    lab.adjustSize()
    pm = lab.grab()
    pos = (at_widget or widget)
    p = QPainter(img)
    p.drawPixmap(pos, pm)
    p.end()
    img.save(str(f))
    walk.shots.append(f)
    if walk.lang == "fr":
        left = cw.untranslated([tip_text], walk.user_names(root))
        if left:
            walk.problem(f"{f.name}: English left in French (tooltip): {left[:6]}")
    print(f"  {f.name}  tip: {tip_text[:90]!r}", flush=True)
    return f


# ---------------------------------------------------------------------------------------------- the parts
def part_home():
    print("Home", flush=True)
    w.show_home()
    walk.settle()
    walk.pump(2.0)
    walk.settle(60)
    walk.shot("home")
    h = w.home
    from astro_gpu.client.home import ProjectCard
    names = [(c.info.name, getattr(c.info, "root", "")) for c in h.findChildren(ProjectCard)]
    bench = [n for n in names if "/bench/" in str(n[1]).replace("\\", "/")]
    print("  cards:", len(names), "bench:", len(bench), flush=True)
    if not any(project.as_posix().lower() == str(r).replace("\\", "/").lower() for _n, r in bench):
        walk.problem("the bench project is not in Home")
    # the bench section scrolled into view
    from PySide6.QtWidgets import QScrollArea
    sc = h.findChildren(QScrollArea)
    if sc:
        sc[0].verticalScrollBar().setValue(sc[0].verticalScrollBar().maximum() // 3)
        walk.settle()
        walk.shot("home_bench")
        sc[0].verticalScrollBar().setValue(0)


def open_it():
    if w.s is not None and w.s.root is not None and Path(w.s.root) == project:
        return
    w.open_project(project)
    walk.wait_ready()


def pid_of(name):
    for q in w.s.paths:
        if q.name == name:
            return q.pid
    raise KeyError(name)


def part_graph():
    print("Graph", flush=True)
    open_it()
    s = w.s
    p = s.view(pid_of(DRIZZLE))
    w.set_mode(0)
    walk.select("empilement", p.pid)
    walk.settle()
    walk.shot("graph")
    # a block hovered: its tooltip
    from PySide6.QtWidgets import QGraphicsView

    from astro_gpu.client.graph import BlockItem
    gv = w.graph.findChildren(QGraphicsView)
    items = [it for g in gv if g.scene() is not None for it in g.scene().items() if isinstance(it, BlockItem)]
    blk = next((it for it in items if it.node.key == "fond" and not it.dim), items[0] if items else None)
    if blk is None:
        walk.problem("no block in the graph")
    else:
        blk.hover = True
        blk.update()
        gv = blk.scene().views()[0]
        r = blk.sceneBoundingRect()
        pt = gv.mapFromScene(r.bottomLeft())
        at = gv.viewport().mapTo(w, pt) + QPoint(10, 8)
        shot_with_tip("graph_hover", gv, blk.toolTip(), at)
        blk.hover = False
        blk.update()
    # a breakpoint, set then taken off
    node = w.graph.graph.graph.node_of(p, "empilement")
    depth = len(s._undo)
    w.graph.toggle_pause(node)
    walk.settle()
    if s.review_state(p, "empilement") != "paused":
        walk.problem(f"breakpoint: {s.review_state(p, 'empilement')!r}")
    walk.shot("graph_breakpoint")
    node = w.graph.graph.graph.node_of(p, "empilement")
    if s.has_breakpoint(p, "empilement"):
        w.graph.toggle_pause(node)
    while len(s._undo) > depth:
        s.undo()
    walk.settle()
    walk.select("couleur", p.pid)
    walk.shot("graph_colour_panel")
    walk.select("fond", p.pid)
    walk.shot("graph_gradient_panel")
    walk.select("post", p.pid)
    walk.shot("graph_output_panel")


def part_stack():
    from PySide6.QtWidgets import QLabel
    print("Stacking settings", flush=True)
    open_it()
    drz, cla = pid_of(DRIZZLE), pid_of(CLASSIC)
    from astro_gpu.client.panel import StackMethodCard
    w.set_mode(0)
    for pid, tag in ((drz, "drizzle"), (cla, "classic")):
        walk.select("empilement", pid)
        panel = w.graph.panel
        panel.expert = False
        panel.refresh(panel.node)
        walk.settle()
        cards = [c for c in panel.findChildren(StackMethodCard) if c.isVisible()]
        rec = [c for c in cards if c.recommended]
        print("  cards:", [(c.key, c.on, c.recommended) for c in cards], flush=True)
        if len(cards) != 2:
            walk.problem(f"{tag}: {len(cards)} method cards")
        if rec and rec[0].info is not None:
            dot = rec[0].info
            at = dot.mapTo(w, QPoint(dot.width() - 300, dot.height() + 6))
            shot_with_tip(f"stack_{tag}_recommended", dot, dot.toolTip(), at)
        elif any(lb.objectName() == "advice_pending" and lb.isVisible() for lb in panel.findChildren(QLabel)):
            note(f"{tag}: no advice yet (the registration is not computed): the panel says so")
        else:
            walk.problem(f"{tag}: no Recommended card with its i")
            walk.shot(f"stack_{tag}")
        panel._expert(True)
        walk.settle()
        walk.shot(f"stack_{tag}_advanced", panel)
        panel._expert(False)
        walk.select("selection", pid)
        walk.settle()
        walk.shot(f"selection_{tag}", panel)


def part_inspect():
    print("Inspect", flush=True)
    open_it()
    s = w.s
    drz, cla = pid_of(DRIZZLE), pid_of(CLASSIC)
    p = s.view(drz)
    # the frames: sessions and browser
    walk.select("sources")
    w.set_mode(1)
    walk.until(lambda: w.frames.data is not None, 120, "the sessions table")
    walk.settle()
    walk.shot("frames_sessions")
    w.frames.show_page(1)
    walk.settle()
    walk.pump(1.0)
    walk.settle()
    walk.shot("frames_browse")
    w.frames.show_page(0)
    for k in s.block_keys(p):
        if k in ("sources",):
            continue
        keys = [v.key for v in w.tune.visuals(p, k) if not v.spec]
        print(f"  {k}: {keys}", flush=True)
        if not keys:
            walk.guarded(walk.show_tab, k, None, f"inspect_{k}", drz)
            continue
        for key in keys:
            walk.guarded(walk.show_tab, k, key, f"inspect_{k}_{key}", drz)
            if k == "empilement" and key == "stack":
                walk.guarded(stack_histogram)
            if k in ("fond", "couleur") and key == keys[0]:
                walk.guarded(side_by_side, k)
    for key in ("stack", "rejection", "coverage", "weights"):
        keys = [v.key for v in w.tune.visuals(s.view(cla), "empilement")]
        if key in keys:
            walk.guarded(walk.show_tab, "empilement", key, f"inspect_classic_{key}", cla)
    w.tune.tab_of.clear()


def stack_histogram():
    v = w.tune.view
    if v is None:
        walk.problem("no viewer on the stack")
        return
    v.hist_b.setChecked(True)
    walk.settle()
    walk.pump(0.5)
    walk.shot("inspect_stack_histogram")
    v.hist_b.setChecked(False)
    walk.settle()


def side_by_side(k):
    v = w.tune.view
    if v is None or not v._targets:
        note(f"{k}: nothing to compare with")
        return
    v.set_compare(0 if v._cmp_i is None else v._cmp_i)
    v.set_compare_mode("side")
    walk.wait_tab(f"{k} side by side")
    walk.pump(0.5)
    walk.shot(f"inspect_{k}_side_by_side")
    v.set_compare_mode("split")
    walk.settle()
    walk.shot(f"inspect_{k}_split")


# ------------------------------------------------------------ Post
NEW_FILTERS = {
    "ghs": {"stretch": 2.0, "symmetry": 0.15, "local": 2.0},
    "asinh": {"stretch": 30.0},
    "histogram": {"midtones": 0.3, "shadows": 0.02},
    "scnr": {"strength": 1.0},
    "clahe": {"amount": 0.8, "clip": 3.0},
    "ml_sharpen": {"l1": 1.0, "l2": 1.0},
    "ml_denoise": {"l1": 3.0, "l2": 2.0},
}


def post_render():
    walk.until(lambda: w.render is not None and not w.render.busy(), 300, "the Post render")
    walk.settle()
    walk.pump(0.4)
    walk.settle()


def props(view, where="top"):
    """The properties of the selection scrolled: "top" (the filter's graph or Curves editor in view) or "bottom"."""
    from PySide6.QtWidgets import QScrollArea
    sa = next((a for a in view.findChildren(QScrollArea) if a.widget() is not None and a.widget().layout() is view.pbox),
              None)
    if sa is None:
        return
    walk.settle()
    bar = sa.verticalScrollBar()
    target = view.curves_editor or view.function_graph
    if where == "bottom":
        bar.setValue(bar.maximum())
    elif target is not None:
        try:
            bar.setValue(max(0, target.mapTo(sa.widget(), QPoint(0, 0)).y() - 40))
        except RuntimeError:
            bar.setValue(0)
    walk.pump(0.2)


def part_post():
    print("Post", flush=True)
    open_it()
    w.set_mode(2)
    post_render()
    view = w.post.view
    walk.shot("post")
    wl = [lay for lay in view.layers if lay["kind"] == "workflow"]
    drz = pid_of(DRIZZLE)
    base = next((lay for lay in wl if lay["pid"] == drz), wl[-1] if wl else None)
    if base is None:
        walk.problem("no workflow layer in Post")
        return
    for lay in wl:
        view.set_visible(lay["id"], lay is base)
    view.select(base["id"], "layer")
    post_render()
    walk.shot("post_base")
    # the filter menu, its groups
    view.build_filter_menu()
    menu = view.filter_menu
    groups = [a.text() for a in menu.actions() if a.menu() is not None]
    print("  filter groups:", groups, flush=True)
    if len(groups) != 6:
        walk.problem(f"the filter menu has {len(groups)} groups")
    menu.popup(view.b_filter.mapToGlobal(QPoint(0, view.b_filter.height())))
    walk.pump(0.4)
    walk.n += 1
    f = out / f"{walk.n:02d}_post_filter_menu_{walk.lang}.png"
    img = w.grab()
    p = QPainter(img)
    p.drawPixmap(view.b_filter.mapTo(w, QPoint(-200, view.b_filter.height())), menu.grab())
    sub = menu.actions()[0].menu()
    sub.popup(menu.mapToGlobal(QPoint(menu.width(), 0)))
    walk.pump(0.3)
    p.drawPixmap(view.b_filter.mapTo(w, QPoint(-200 - sub.sizeHint().width() - 4, view.b_filter.height())),
                 sub.grab())
    p.end()
    img.save(str(f))
    sub.hide()
    menu.hide()
    walk.shots.append(f)
    if walk.lang == "fr":
        left = cw.untranslated(groups + [a.text() for g in menu.actions() if g.menu() for a in g.menu().actions()])
        if left:
            walk.problem(f"filter menu: English left in French: {left}")
    # Curves
    fl = view.add_filter("curves")
    view.select(fl["id"], "layer")
    post_render()
    ed = view.curves_editor
    if ed is None:
        walk.problem("no Curves editor")
    else:
        props(view)
        walk.shot("post_curves")
        props(view, "bottom")
        walk.shot("post_curves_bottom")
        g = ed.graph
        ed.starting.emit()
        g.add(0.25, 0.38)
        g.released.emit()
        post_render()
        props(view)
        walk.shot("post_curves_point")
        for ch in ("R", "L*", "S"):
            ed = view.curves_editor
            ed.set_channel(ch)
            walk.settle()
        props(view)
        walk.shot("post_curves_channel_S")
        ed = view.curves_editor
        ed.set_channel("RGB/K")
        from astro_gpu.client.postfilters.curves import PS
        ed.apply_preset(next(k for k in ed._presets if ed._presets[k]["source"] == PS))
        post_render()
        props(view)
        walk.shot("post_curves_classic_preset")
        ed = view.curves_editor
        astro = [k for k in ed._presets if ed._presets[k]["source"] != PS and k != "Linear"]
        ed.apply_preset(astro[0])
        post_render()
        props(view)
        walk.shot("post_curves_astro_preset")
        ed = view.curves_editor
        name = ed.save_preset("Mine " + walk.lang)
        walk.settle()
        ed = view.curves_editor
        if name is None or ed.preset.findData("user:" + name) < 0:
            walk.problem("the user's preset is not in the list")
        props(view)
        walk.shot("post_curves_user_preset")
        view.delete_layer = None
        view.select(fl["id"], "layer")
        view.delete()
        post_render()
    # every new filter, alone on the base layer, its sliders moved
    for key, params in NEW_FILTERS.items():
        view.select(base["id"], "layer")
        fl = view.add_filter(key)
        view.select(fl["id"], "layer")
        post_render()
        hist_before = _mean_of_view(view)
        # the sliders moved as the user does (each one let go: a render and an undo step)
        from astro_gpu.client.post3 import ContSlider
        from astro_gpu.client.postengine import FILTERS
        numeric = [q["key"] for q in FILTERS[key]["params"] if q["type"] not in ("curve", "bool", "choice")]
        sliders = [s for s in view.findChildren(ContSlider) if s.isVisibleTo(view)][-len(numeric):]
        by_key = dict(zip(numeric, sliders))
        for k, v in params.items():
            sl = by_key.get(k)
            if sl is None:
                walk.problem(f"{key}: no slider {k}")
                continue
            sl.set_value(v)
            sl._label()
            sl.done.emit(sl.value())
            walk.pump(0.05)
            if abs(fl["params"][k] - v) > 1e-6:
                walk.problem(f"{key}: the slider {k} set {fl['params'][k]}, not {v}")
        post_render()
        after = _mean_of_view(view)
        if hist_before is not None and after is not None and abs(after - hist_before) < 1e-4:
            walk.problem(f"{key}: no visible effect (mean {hist_before:.5f} -> {after:.5f})")
        props(view)
        walk.shot(f"post_filter_{key}")
        props(view, "bottom")
        walk.shot(f"post_filter_{key}_settings")
        view.delete()
        post_render()


def _mean_of_view(view):
    """The mean of the image shown (the screen's render), to tell a filter that does nothing."""
    try:
        import numpy as np
        a = view.src._half()
        if a is None:
            return None
        return float(np.asarray(a, dtype=np.float64).mean())
    except Exception:  # noqa: BLE001
        return None


def part_export():
    print("Export", flush=True)
    open_it()
    w.set_mode(2)
    post_render()
    EXPORTS.mkdir(parents=True, exist_ok=True)
    fmts = ("fits32f", "tiff16", "png16", "jpeg")
    from astro_gpu.client.exportdlg import SUFFIX

    def export(d):
        keys = [d.fmt.itemData(i) for i in range(d.fmt.count())]
        print("  formats:", keys, flush=True)
        walk.shot("export", d)
        for fk in fmts:
            if fk not in keys:
                walk.problem(f"export: no format {fk} in {keys}")
                continue
            d.fmt.setCurrentIndex(d.fmt.findData(fk))
            walk.pump(0.1)
            d.dest.setText((EXPORTS / walk.lang).as_posix() + "/image" + SUFFIX[fk])
            n0 = len(walk.messages)
            t0 = time.time()
            d.start()
            walk.until(lambda: not d.running, 600, f"the export {fk}")
            walk.settle()
            print(f"  {fk}: {d.files} {d.error!r} {time.time() - t0:.1f} s; messages: {walk.messages[n0:]}",
                  flush=True)
            if not d.files:
                walk.problem(f"export {fk} wrote nothing: {d.error or d.status.text()}")
            walk.shot(f"export_{fk}_done", d)
        d.accept()
    walk.modal(w.export, export, "Export")
    walk.settle()


def shot_with_popup(name, combo, root=None):
    """``root`` (default the window) grabbed with the list of ``combo`` opened under it (a popup is a window of its
    own: painted over the grab)."""
    root = root or w
    combo.showPopup()
    walk.pump(0.3)
    pop = combo.view().window()
    walk.n += 1
    f = out / f"{walk.n:02d}_{name}_{walk.lang}.png"
    img = root.grab()
    p = QPainter(img)
    p.drawPixmap(combo.mapTo(root, QPoint(0, combo.height())), pop.grab())
    p.end()
    img.save(str(f))
    walk.shots.append(f)
    items = [combo.itemText(i) for i in range(combo.count())]
    combo.hidePopup()
    print(f"  {f.name}  list: {items}", flush=True)
    return f


def part_naming():
    """The texts that named other software (2 November): the stacking settings of both methods (the drizzle's
    rejection list opened), the Selection rule, the Colour settings, the stacking's rejection in Inspect, the help of
    two Post filters."""
    print("Naming", flush=True)
    open_it()
    drz, cla = pid_of(DRIZZLE), pid_of(CLASSIC)
    from PySide6.QtWidgets import QComboBox

    from astro_gpu.client.panel import CompactRow
    w.set_mode(0)
    for pid, tag in ((drz, "drizzle"), (cla, "classic")):
        walk.select("empilement", pid)
        panel = w.graph.panel
        panel._expert(True)
        walk.settle()
        walk.shot(f"naming_stack_{tag}", panel)
        if tag == "drizzle":
            row = next((r for r in panel.findChildren(CompactRow)
                        if r.r.get("key") == "stack.drizzle_rejection" and r.isVisible()), None)
            combo = row.findChildren(QComboBox)[0] if row is not None and row.findChildren(QComboBox) else None
            if combo is None:
                walk.problem("no list for the drizzle's rejection")
            else:
                row.focused.emit(row.r)                       # its help in the Details
                walk.settle()
                shot_with_popup("naming_drizzle_rejection_list", combo)
        else:
            for key in ("stack.classic_clamping", "stack.classic_cfa_balance"):
                row = next((r for r in panel.findChildren(CompactRow) if r.r.get("key") == key and r.isVisible()),
                           None)
                if row is not None:
                    row.focused.emit(row.r)
                    walk.settle()
                    combo = row.findChildren(QComboBox)
                    if combo:
                        shot_with_popup(f"naming_{key.split('.')[1]}_list", combo[0])
                    else:
                        walk.shot(f"naming_{key.split('.')[1]}")
        panel._expert(False)
        walk.select("selection", pid)
        walk.settle()
        walk.shot(f"naming_selection_{tag}", panel)
    walk.select("couleur", drz)
    panel = w.graph.panel
    walk.settle()
    row = next((r for r in panel.findChildren(CompactRow) if r.r.get("key") == "color.correction" and r.isVisible()),
               None)
    if row is None:
        walk.problem("no Colour correction row")
        walk.shot("naming_colour", panel)
    else:
        row.focused.emit(row.r)
        walk.settle()
        combo = row.findChildren(QComboBox)
        shot_with_popup("naming_colour_correction", combo[0]) if combo else walk.shot("naming_colour", panel)
    for pid, tag in ((drz, "drizzle"), (cla, "classic")):
        walk.guarded(walk.show_tab, "empilement", "rejection", f"naming_inspect_rejection_{tag}", pid)
    walk.guarded(walk.show_tab, "couleur", None, "naming_inspect_colour", drz)
    w.tune.tab_of.clear()
    # Post: the help line of two filters
    w.set_mode(2)
    post_render()
    view = w.post.view
    wl = [lay for lay in view.layers if lay["kind"] == "workflow"]
    base = next((lay for lay in wl if lay["pid"] == drz), wl[-1] if wl else None)
    if base is None:
        walk.problem("no workflow layer in Post")
        return
    for lay in wl:
        view.set_visible(lay["id"], lay is base)
    for key in ("ghs", "ml_sharpen"):
        view.select(base["id"], "layer")
        fl = view.add_filter(key)
        view.select(fl["id"], "layer")
        post_render()
        props(view)
        walk.shot(f"naming_post_{key}")
        view.delete()
        post_render()


def part_evening(stale="S1"):
    """The user's points of 2 November evening: the x / y tracking in Selection and Alignment; Astrometry's background
    list on an up-to-date workflow and on an Outdated one (``stale``, its previous results), and the zoom kept when the
    background changes (the stack zoomed in, then the final image)."""
    print("Evening", flush=True)
    open_it()
    drz = pid_of(DRIZZLE)
    walk.guarded(walk.show_tab, "selection", None, "evening_selection", drz)
    walk.guarded(walk.show_tab, "alignement", "track", "evening_alignment_track", drz)
    tune = w.tune
    for pid, tag in ((drz, "current"), (pid_of(stale), "stale")):
        p = w.s.view(pid)
        bgs = tune.steps.backgrounds(p)
        print(f"  {tag}: backgrounds {[(b.key, b.previous) for b in bgs]}", flush=True)
        if not bgs:
            walk.problem(f"Astrometry of {tag}: no background listed")
        walk.guarded(walk.show_tab, "astrometrie", "solution", f"evening_astrometry_{tag}", pid)
    v = tune.view
    if v is None or v.canvas.display is None:
        walk.problem("no Astrometry viewer")
        return
    v.zoom_to(1.0, v.canvas.display.size()[0] * 0.3, v.canvas.display.size()[1] * 0.4)
    walk.settle()
    walk.shot("evening_astrometry_stack_zoom")
    tune._pick_background("astrometrie", "output")
    walk.until(lambda: tune.view is not None and tune.view.canvas.display is not None
               and tune.view.canvas.display.size()[0] < 3000, 300, "the final image")
    walk.pump(1.0)
    walk.settle()
    walk.shot("evening_astrometry_final_zoom")
    tune._pick_background("astrometrie", "stack")
    # Output: the final image beside Gradient removed, Same stretch greyed (one is already stretched for the eye)
    walk.guarded(walk.show_tab, "post", None, "evening_output", drz)
    v = tune.view
    if v is not None:
        walk.until(lambda: v._targets, 300, "the Output compare targets")
        names = [t.name for t in v._targets]
        i = next((j for j, n in enumerate(names) if n.startswith(_("Gradient removed"))), 0)
        v.set_compare_mode("side")
        v.set_compare(i)
        walk.pump(2.0)
        walk.settle()
        walk.shot("evening_output_compare")
        if v.same_c.isEnabled():
            walk.problem("Same stretch enabled between the final image and linear data")
    w.tune.tab_of.clear()
    # Graph: images sent to Post draw an arrow from their block (the composition set in memory: the model is read-only)
    import json as _json
    sp = w.s.p
    old = getattr(sp, "post", None)
    skey = w.s.stage_key(w.s.view(drz), "stack")
    sp.post = {**(old or {}), "layers": [
        {"id": "t1", "kind": "image", "key": "inspect:" + _json.dumps({"wid": drz, "kind": "gradient", "name": "",
                                                                      "channel": None, "base": [],
                                                                      "stages": ["gradient", "color"]}),
         "name": "Gradient removed", "visible": True},
        {"id": "t2", "kind": "image", "key": f"stack/{skey}/drz_full", "name": "Stack", "visible": True}]}
    w.set_mode(0)
    walk.select("fond", drz)
    w.graph.graph.rebuild()
    walk.settle()
    walk.shot("evening_graph_post_arrows")
    sp.post = old


def part_gradedit():
    from PySide6.QtCore import Qt
    """The Gradient block's samples and areas on the image (gradientedit.py), on the --drizzle workflow: the samples
    of the method shown, the generated grid, an ellipse and a brush stroke excluded, the previews, each edit a setting
    (in memory: the model is read-only), the editor then forgotten as when the project is opened again."""
    print("Gradient editing", flush=True)
    open_it()
    pid = pid_of(DRIZZLE)
    walk.show_tab("fond", "result", "gradedit_result", pid)
    ed = w.tune.grad_editor
    if ed._method() == "automatic":                   # the samples are the polynomial's (in memory only)
        w.s.apply_change([w.s.view(pid)], {"kind": "opt", "key": "fond", "value": "simple"})
        walk.show_tab("fond", "result", "gradedit_result", pid)

    def canvas():                        # Inspect is built again after each edit, as after any setting
        walk.pump(0.4)
        return w.tune.view.canvas

    def idle(what):
        walk.until(lambda: not ed.busy and not ed._timer.isActive(), 60, what)
        walk.pump(1.0)
        walk.settle()
    ed.set_mode("points")
    walk.until(lambda: not ed.busy and ed.small is not None, 120, "the reduced stack")
    walk.settle()
    walk.shot("gradedit_samples")                 # none placed: the ones the last run chose
    ed._set_avoid(False)                          # the plain grid first (Away from emission comes after)
    ed.generate()
    walk.until(lambda: not ed.busy and ed.work["sample_points"], 60, "the generated grid")
    idle("the grid's preview")
    walk.shot("gradedit_grid")
    if not ed.s.param_value(ed.p, "gradient", "sample_points"):
        walk.problem("the generated grid is not in the settings")
    # the H-alpha survey under the samples (H), then the grid again away from its emission
    n_all = len(ed.work["sample_points"])
    ed._set_survey_shown(True)
    walk.until(lambda: not ed.busy and ed.survey is not None, 180, "the H-alpha survey")
    walk.settle()
    walk.shot("gradedit_survey")
    if "level" in (ed.survey or {}):
        ed._set_avoid(True)
        ed.generate()
        walk.until(lambda: not ed.busy and len(ed.work["sample_points"]) != n_all, 60, "the grid away from emission")
        idle("the grid's preview")
        walk.shot("gradedit_grid_away")
    else:
        walk.shot("gradedit_survey_none")
    W, H = ed.small["W"], ed.small["H"]
    ed.set_mode("ellipse")

    def drag(a, b):                      # a drag of the tool, in stack pixels
        c = canvas()
        f = ed._f(c)
        ed.press(c, a[0] * f, a[1] * f, Qt.LeftButton)
        ed.move(c, b[0] * f, b[1] * f)
        ed.release(c)

    def handle(name):
        c = canvas()
        f = ed._f(c)
        x, y = c.w_to_img(*ed._handles(c, f)[name])
        return x / f, y / f
    drag((W * 0.32, H * 0.25), (W * 0.62, H * 0.8))            # drawn
    rot = handle("rot")
    e = ed.work["exclude_ellipses"][ed._sel]
    drag(rot, (e[0] + W * 0.15, e[1] - H * 0.3))                # turned
    rx = handle("rx")
    drag(rx, (rx[0] + W * 0.04, rx[1]))                         # widened
    stored = ed.s.param_value(ed.p, "gradient", "exclude_ellipses") or []
    if not (len(stored) == 1 and abs(stored[0][4]) > 5):
        walk.problem(f"the ellipse was not turned, or not in the settings: {stored}")
    idle("the ellipse's preview")
    walk.shot("gradedit_ellipse")
    ed.set_mode("brush")
    c = canvas()
    f = ed._f(c)
    ed.press(c, W * 0.15 * f, H * 0.2 * f, Qt.LeftButton)
    for t in range(1, 20):
        ed.move(c, W * (0.15 + 0.01 * t) * f, H * 0.2 * f)
    ed.release(c)
    ed._hover = (W * 0.36 * f, H * 0.2 * f)
    idle("the brush's preview")
    walk.shot("gradedit_brush")
    # the project opened again: an editor that remembers nothing draws all from the settings
    ed.set_mode(None)
    ed.p, ed.result, ed.samples, ed.small, ed.work = None, None, None, None, {k: [] for k in ed.work}
    walk.show_tab("fond", "result", "gradedit_reopened", pid)
    if len(ed.work["exclude_ellipses"]) != 1 or not ed.work["sample_points"] or not ed.work["exclude_brush"]:
        walk.problem(f"the edits did not come back from the settings: {[(k, len(x)) for k, x in ed.work.items()]}")
    for _i in range(20):                               # back as the project was, with the window's Undo
        if not any(ed.s.param_value(ed.p, "gradient", k) for k in ("sample_points", "exclude_ellipses",
                                                                    "exclude_brush")):
            break
        w.undo()
        walk.pump(0.3)
    else:
        walk.problem("Undo did not take the edits back")
    walk.settle()
    # Post's mask brush, the same gestures (for the comparison of the two editors)
    w.set_mode(2)
    post_render()
    view = w.post.view
    lay = next((x for x in view.layers if x["kind"] == "workflow"), None)
    if lay is not None:
        view.select(lay["id"], "layer")
        view.add_mask(lay)
        view.add_part(view.layer(lay["id"]), "brush")
        view.select(lay["id"], "mask")
        walk.settle()
        vw = view.viewer
        vw._mouse = vw.img_to_w(*[a / 2 for a in vw.source.size()])
        vw.update()
        walk.settle()
        walk.shot("gradedit_post_mask_brush")
    w.set_mode(1)


def part_denoise():
    """The Denoise block of the two denoise workflows of the bench (tools/bench/denoise_projects.py): the panel with
    its method cards and settings, then Inspect on the block."""
    print("Denoise", flush=True)
    open_it()
    for name, tag in (("Classic stack · denoise", "pretrained"), ("Classic stack · denoise adapted", "adapted")):
        try:
            pid = pid_of(name)
        except KeyError:
            walk.problem(f"no workflow {name!r}")
            continue
        w.set_mode(0)
        walk.select("debruitage", pid)
        walk.settle()
        walk.shot(f"denoise_{tag}_panel")
        walk.show_tab("debruitage", "denoised", f"denoise_{tag}_inspect", pid)     # the image loaded
    w.set_mode(0)


def part_stars(name="Full · denoise"):
    """The Stars block and the Composition section: the graph (the block's two ports, the Stars and background block),
    its panel, Inspect on the Stars block (the starless image against the image, the stars alone), Post on the block's
    layer, a star slider moved from the graph's panel, the starless image sent to Post as it is."""
    print("Stars", flush=True)
    open_it()
    s = w.s
    try:
        pid = pid_of(name)
    except KeyError:
        walk.problem(f"no workflow {name!r}")
        return
    w.set_mode(0)
    walk.select("etoiles", pid)
    walk.settle()
    walk.shot("stars_graph")
    g = w.graph.graph.graph
    comp = g.node_of(s.view(pid), "composition")
    if comp is None:
        walk.problem("no Stars and background block in the graph")
        return
    w.graph.select_node(comp)
    walk.settle()
    walk.shot("stars_composition_panel")
    lid = s.sel_layer
    walk.show_tab("etoiles", "starless", "stars_inspect_starless", pid)
    walk.show_tab("etoiles", "stars", "stars_inspect_stars", pid)
    w.open_layer(lid)
    walk.settle()
    walk.pump(1.0)
    walk.settle()
    walk.shot("stars_post")
    s.set_post_layer(lid, None, sliders={"star_strength": 1.0, "star_size": 1.0})
    walk.settle()
    walk.pump(1.0)
    walk.settle()
    walk.shot("stars_post_fainter")
    w.set_mode(0)
    stars = w.graph.graph.graph.node_of(s.view(pid), "etoiles")
    w.graph.part_to_post(stars, "starless")
    walk.settle()
    walk.shot("stars_graph_starless_sent")


PART_FN = {"gradedit": part_gradedit, "denoise": part_denoise, "stars": part_stars,
           "home": part_home, "evening": part_evening, "graph": part_graph, "stack": part_stack, "inspect": part_inspect,
           "post": part_post, "export": part_export, "naming": part_naming}

for lang in LANGS:
    if lang != walk.lang:
        walk.guarded(walk.step_settings, lang)
    for part in PARTS:
        t = time.time()
        walk.guarded(PART_FN[part])
        print(f"== {part} {lang}: {time.time() - t:.0f} s", flush=True)
from astro_gpu.client import i18n  # noqa: E402

miss = sorted(t for t in i18n.missing() if re.search(r"[A-Za-z]{3}", t))
if walk.lang != "en":
    walk.guarded(walk.step_settings, "en")
w.close()
walk.pump(0.5)
if fingerprint(project) != BEFORE:
    walk.problem("the project's files changed: " + str(sorted(set(fingerprint(project)) ^ set(BEFORE))[:10]))
else:
    print("project files untouched", flush=True)
noise = [m for m in walk.messages if not cw._harmless(m)]
rep = {"problems": walk.problems, "untranslated_asked": miss, "messages": noise, "notes": LOG,
       "exports": sorted(str(f.relative_to(tmp)) + f" {f.stat().st_size}" for f in EXPORTS.rglob("*") if f.is_file())
       if EXPORTS.exists() else [], "seconds": round(time.time() - T0)}
(out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(rep, indent=1, ensure_ascii=False), flush=True)
