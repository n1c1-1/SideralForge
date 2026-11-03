"""Screenshots of the stacking settings (both methods, the Recommended frame with its tooltip) and of the Selection
block (one per rule) in the real client, EN then FR, offscreen, CPU only, on a bench project opened READ-ONLY.

    python tools/shots_stacking.py PROJECT OUT [--fake-advice]

PROJECT: F:/astro_essais/bench/<case>/v2x_express (never F:/astro_projets). Its workflows "V2" (drizzle) and "V2-C"
(classic) are shown. Nothing is written: the model is read-only, the user's settings and ~/.astro_gpu are replaced by
scratch ones, the Selection shots pick the project's own workflows of each rule, and the project's files (all but
the store) are fingerprinted before and after: a difference is reported as a problem. ``--fake-advice``: when the
engine has no "method" automatic value, the stage's auto() is patched in this process to add one from its own coverage
rule (drizzle_suggestion).
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import client_walkthrough as cw  # noqa: E402

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    raise SystemExit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
project, out = Path(sys.argv[1]), Path(sys.argv[2])
fake = "--fake-advice" in sys.argv
assert "astro_projets" not in project.as_posix().lower()
out.mkdir(parents=True, exist_ok=True)


def fingerprint(root: Path):
    """Names, sizes and dates of every file of the project but the store's results (big, and only read)."""
    return sorted((f.relative_to(root).as_posix(), f.stat().st_size, f.stat().st_mtime_ns)
                  for f in root.rglob("*") if f.is_file() and "store" not in f.relative_to(root).parts[:1])


BEFORE = fingerprint(project)
tmp = Path(tempfile.mkdtemp(prefix="shots_stack_"))
cw.setup_environment(tmp)

from astro_gpu.core.params import Auto  # noqa: E402
from astro_gpu.stages import stack as stmod  # noqa: E402

PATCHED = False
if fake and 'advice=True' not in Path(stmod.__file__).read_text(encoding="utf-8"):
    orig = stmod.Stack.auto

    def auto(self, ctx):
        a = orig(self, ctx)
        if "method" not in a:
            import numpy as np
            try:
                reg = np.load(ctx.input("register", "reg.npz"))
                m = stmod.drizzle_suggestion(reg, ctx.load_json("calibrate", "cache.json")).method
                a["method"] = Auto(m.value, m.template, m.args, advice=True)
            except Exception as e:  # noqa: BLE001
                print("  no advice:", e, flush=True)
        return a
    stmod.Stack.auto = auto
    PATCHED = True
print("fake advice:", PATCHED, flush=True)

walk = cw.Walk(out, tmp / "ws", True, tmp)
w = walk.w
w.readonly = True                       # the model never writes the bench copy (definition, state.json)
w.open_project(project)
walk.wait_ready()
s = w.s
pids = {q.name: q.pid for q in s.paths}
drz, cla = pids["V2"], pids["V2-C"]


def panel():
    return w.graph.panel


def grab_with_tooltip(name):
    """The window with the tooltip of the "i" of the Recommended card painted where Qt shows it."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QPainter
    from PySide6.QtWidgets import QApplication, QToolTip

    from astro_gpu.client.panel import StackMethodCard
    walk.settle()
    cards = [c for c in panel().findChildren(StackMethodCard) if c.isVisible()]
    rec = [c for c in cards if c.recommended]
    print("  cards:", [(c.key, c.on, c.recommended) for c in cards], flush=True)
    if not rec or rec[0].info is None:
        walk.problem(f"{name}: no Recommended card with its i")
        return walk.shot(name)
    dot = rec[0].info
    g = dot.mapToGlobal(QPoint(dot.width() // 2, dot.height()))
    QToolTip.showText(g, dot.toolTip(), dot)
    walk.pump(0.6)
    tip = next((t for t in QApplication.topLevelWidgets() if t.metaObject().className() == "QTipLabel"
                and t.isVisible()), None)
    walk.n += 1
    f = out / f"{walk.n:02d}_{name}_{walk.lang}.png"
    img = w.grab()
    if tip is not None:
        pm = tip.grab()
        from PySide6.QtCore import QPoint as _P
        at = dot.mapTo(w, _P(dot.width() - pm.width(), dot.height() + 6))
        p = QPainter(img)
        p.drawPixmap(at, pm)
        p.end()
    else:
        walk.problem(f"{name}: no tooltip shown")
    img.save(str(f))
    QToolTip.hideText()
    print(f"  {f.name}  tooltip: {dot.toolTip()!r}", flush=True)
    return f


def graph_shots():
    w.set_mode(0)
    walk.select("empilement", drz)
    panel().expert = False
    panel().refresh(panel().node)
    walk.settle()
    grab_with_tooltip("stack_drizzle_recommended")
    panel()._expert(True)
    walk.shot("stack_drizzle_advanced", panel())
    panel()._expert(False)
    walk.select("empilement", cla)
    walk.settle()
    walk.shot("stack_classic", panel())
    panel()._expert(True)
    walk.shot("stack_classic_advanced", panel())
    panel()._expert(False)
    # the method changed on a computed block: Replace or Try as variant is asked; cancelled
    walk.select("empilement", drz)
    walk.settle()
    from astro_gpu.client.panel import StackMethodCard
    other = [c for c in panel().findChildren(StackMethodCard) if not c.on and c.ok]
    if other:
        depth = len(s._undo)
        other[0].clicked.emit()
        walk.settle()
        walk.shot("stack_method_changed", panel())
        if s.pending_change is not None:                 # a computed stack: Replace / Try as variant, cancelled
            panel()._cancel()
        else:                                            # an outdated stack: changed at once (in memory), undone
            print("  changed at once (the stack is not computed): undone", flush=True)
            while len(s._undo) > depth:
                s.undo()
        walk.settle()
        if [c.on for c in panel().findChildren(StackMethodCard) if c.isVisible()][:1] != [True]:
            walk.problem("the method was not put back")


def selection_shots():
    """The Selection block of one workflow under each rule (Automatic, k x sigma, Thresholds): the project's own
    workflows for the rules it has; for Thresholds, when none of its workflows has it, one workflow is changed in memory
    (the model is read-only) and the change undone."""
    by_rule = {}
    for q in s.paths:
        by_rule.setdefault(s.selection_rule(q), q)
    print("  selection rules in the project:", {k: v.name for k, v in by_rule.items()}, flush=True)
    for rule in ("min_weight", "sota"):
        if rule in by_rule:
            walk.select("selection", by_rule[rule].pid)
            walk.settle()
            walk.shot(f"selection_{rule}", panel())
            if rule != "sota":
                walk.shot(f"selection_{rule}_window")
        else:
            walk.problem(f"no workflow of the project has the {rule} rule")
    q = by_rule.get("thresholds")
    depth = len(s._undo)
    if q is None:
        q = by_rule["min_weight"]
        walk.select("selection", q.pid)
        s.scope = [q.pid]                                # only this workflow
        s.apply_change([q], dict(kind="param", stage="stack", name="selection", value="thresholds", suggested=None))
        walk.settle()
    walk.select("selection", q.pid)
    walk.settle()
    if s.selection_rule(q) != "thresholds":
        walk.problem("the Thresholds rule is not set")
    walk.shot("selection_thresholds", panel())
    walk.shot("selection_thresholds_window")
    while len(s._undo) > depth:
        s.undo()
    s.scope = None
    walk.settle()
    if s.selection_rule(q) == "thresholds" and "thresholds" not in by_rule:
        walk.problem("the rule was not put back")


def inspect_shots():
    for pid, tag in ((drz, "drizzle"), (cla, "classic")):
        for key in ("rejection", "coverage", "weights"):
            walk.show_tab("empilement", key, f"inspect_{tag}_{key}", pid)
    w.tune.tab_of.clear()


t0 = time.time()
walk.guarded(graph_shots)
walk.guarded(selection_shots)
walk.guarded(inspect_shots)
walk.guarded(walk.step_settings, "fr")
walk.guarded(graph_shots)
walk.guarded(selection_shots)
walk.guarded(inspect_shots)
from astro_gpu.client import i18n  # noqa: E402

miss = sorted(t for t in i18n.missing() if any(c.isalpha() for c in t))
walk.guarded(walk.step_settings, "en")
w.close()
walk.pump(0.5)
if fingerprint(project) != BEFORE:
    walk.problem("the project's files changed: " + str(sorted(set(fingerprint(project)) ^ set(BEFORE))[:10]))
else:
    print("project files untouched", flush=True)
print("untranslated asked:", miss[:40], flush=True)
print("problems:", walk.problems, flush=True)
print("messages:", walk.messages[:20], flush=True)
print(f"done in {time.time() - t0:.0f} s", flush=True)
