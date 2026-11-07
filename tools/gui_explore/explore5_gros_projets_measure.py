"""Round 4, axis "gros_projets": large projects. Builds a project of a given size (model-level setup: sessions whose
folders list many tiny fake frames, filter and colour workflows, branches, Fusion blocks, Post filter layers), then
measures with real Qt events: opening, painting, scroll and "Show all", selecting, placing from the palette, Del,
Ctrl+Z / Ctrl+Y, renaming a session, F5 (run window), the Post page; and checks frame counts, status line, overlaps,
cut labels, memory. Every gesture is measured twice.

    python explore5_gros_projets_measure.py S|M|L|XL [lang]

Prints one JSON line per measure (MEASURE {...}) and per check (CHECK {...}); screenshots in exploration4/gros_projets.
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
from harness import Explorer  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

SIZE = sys.argv[1] if len(sys.argv) > 1 else "S"
LANG = sys.argv[2] if len(sys.argv) > 2 else "fr"
WORK = Path("F:/astro_essais/_explore4_work/gros_projets")
# size: filters (mono sessions per filter), colour sessions, frames per session, the big session's frames, branches
# per workflow, fusions, Post filter layers, sends to Post
SIZES = {
    "S": dict(filters=(), per_filter=0, colour=1, frames=15, big=0, branches=0, fusions=0, filters_post=0, sends=0),
    "M": dict(filters=("Ha", "OIII"), per_filter=1, colour=2, frames=150, big=0, branches=1, fusions=1,
              filters_post=4, sends=1),
    "L": dict(filters=("Ha", "OIII", "SII"), per_filter=4, colour=4, frames=300, big=3000, branches=2, fusions=3,
              filters_post=15, sends=3),
    "XL": dict(filters=("Ha", "OIII", "SII"), per_filter=6, colour=6, frames=400, big=5000, branches=3, fusions=3,
               filters_post=25, sends=5),
}
CFG = SIZES[SIZE]
CHAIN = ("calibration", "registration", "stacking", "astrometry", "gradient", "denoise")

pc = time.perf_counter
ex = Explorer("gros_projets", lang=LANG)
OUT = ex.out
MEAS: list[dict] = []


def P(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ data: tiny fake frames
FITS_STUB = (b"SIMPLE  =                    T" + b" " * 50)          # the first card only (80 bytes)


def frames(folder: str, n: int, ext: str, filt: str = "") -> str:
    d = WORK / "frames" / folder
    d.mkdir(parents=True, exist_ok=True)
    have = sum(1 for _ in d.iterdir())
    if have == n:
        return str(d)
    for i in range(n):
        f = d / f"L_{filt or 'C'}_{i:05d}{ext}"
        if f.exists():
            continue
        if ext == ".fits" and i == 0:                        # the first one is read for its size: a real FITS file
            import numpy as np
            from astropy.io import fits
            hd = fits.Header()
            hd["FILTER"] = filt
            fits.writeto(f, np.zeros((8, 12), np.uint16), hd, overwrite=True)
        elif ext == ".fits":
            f.write_bytes(FITS_STUB)
        else:
            f.write_bytes(b"\0" * 16)
    return str(d)


def build(name: str) -> Path:
    """The project definition (model level: setup only)."""
    root = WORK / "projects" / name
    if root.exists():                                          # a fresh one each time (never deleted: renamed)
        root = WORK / "projects" / f"{name}_{int(time.time())}"
    p = project2.create(root, name, Target(name="M 31"))
    p.stack_mode = "per_session"
    sid = 0
    by_filter: dict[str, list[int]] = {}
    for f in CFG["filters"]:
        for k in range(CFG["per_filter"]):
            sid += 1
            p.sessions.append(project2.Session(sid, f"{f} nuit {k + 1}", filter=f,
                                               lights=[frames(f"{SIZE}/{f}_{k}", CFG["frames"], ".fits", f)]))
            by_filter.setdefault(f, []).append(sid)
    colour = []
    for k in range(CFG["colour"]):
        sid += 1
        n = CFG["big"] if k == 0 and CFG["big"] else CFG["frames"]
        s = project2.Session(sid, f"Couleur nuit {k + 1}", lights=[frames(f"{SIZE}/C_{k}", n, ".ARW")])
        if k < 2:
            s.darks = [frames(f"{SIZE}/darks", 40, ".ARW")]
            s.flats = [frames(f"{SIZE}/flats_{k}", 30, ".ARW")]
            s.bias = [frames(f"{SIZE}/bias", 50, ".ARW")]
        p.sessions.append(s)
        colour.append(sid)
    project2.save(p)
    for f, sids in by_filter.items():
        w = p.add_workflow(f, wfl.TEMPLATES["Classic stack"])
        w.sessions = sids
        p.set_blocks(w.id, [wfl.node(b) for b in CHAIN])
    for k, s in enumerate(colour):
        w = p.add_workflow(f"Couleur {k + 1}", wfl.TEMPLATES["Classic stack"])
        w.sessions = [s]
        p.set_blocks(w.id, [wfl.node(b) for b in CHAIN[:5] + ("color", "denoise")])
    project2.save(p)
    return root


def enrich():
    """Branches, Fusions, sends to Post (the model's calls the gestures make), then Post filter layers."""
    from astro_gpu.client.model import node_key
    s = ex.s
    base = [q for q in s.paths if q.source is None]
    made = 0
    for q in base:
        got = 0
        for n in s.graph(q):
            if got >= CFG["branches"]:
                break
            key = node_key(n)
            for blk in ("gradient", "denoise", "color"):
                if key in ("calibration", "registration") or not s.can_branch(q, key, blk):
                    continue
                try:
                    s.add_branch(q, key, blk)
                    got += 1
                    made += 1
                    break
                except ValueError as e:
                    P("branch refused", q.name, key, blk, e)
    fus = 0
    pid = {q.name: q.pid for q in s.paths}
    for pal in ("SHO", "HSO", "HOO"):
        roles = {"SHO": ("SII", "Ha", "OIII"), "HSO": ("Ha", "SII", "OIII"), "HOO": ("Ha", "OIII")}[pal]
        if fus < CFG["fusions"] and all(r in pid for r in roles):
            s.add_fusion(pal, pal, [{"pid": pid[r], "role": r, "channel": "L"} for r in roles])
            fus += 1
    sent = 0
    for q in [q for q in s.paths if q.source is None][:CFG["sends"]]:
        for n in s.graph(q):
            if s.can_send_to_post(q, node_key(n)):
                s.send_to_post(q, node_key(n), f"{q.name} · {n['block']}")
                sent += 1
                break
    ex.settle(10)
    ex.w.save()
    ex.settle(10)
    nfil = 0
    if CFG["filters_post"]:
        ex.click(ex.w.mode_btns[2])
        ex.settle(20)
        v = ex.w.post_holder.page.view
        keys = ["curves", "levels", "saturation", "local_contrast", "sharpen", "denoise", "scnr", "dehaze", "ghs",
                "clahe", "highlights", "balance", "mixer", "faint_smooth", "halpha_red"]
        for i in range(CFG["filters_post"]):
            v.add_filter(keys[i % len(keys)])
            nfil += 1
        ex.settle(10)
        ex.click(ex.w.mode_btns[0])
        ex.settle(10)
        ex.w.save()
        ex.settle(10)
    P("ENRICH branches", made, "fusions", fus, "sent", sent, "post filters", nfil, "workflows", len(s.paths))


# ------------------------------------------------------------------ measuring
_NoSleep = types.SimpleNamespace(sleep=lambda s: None, time=time.time, strftime=time.strftime)


def measure(gesture: str, fn, rep: int, extra: dict | None = None, quiet: float = 0.4) -> dict:
    """``fn`` timed: ``sync`` its own call (the event handler, what the user waits for before the window answers),
    ``freeze`` sync plus every later stall of the event loop longer than 15 ms until the window is quiet (worker idle
    and no stall for ``quiet`` s), ``done`` the time until then (background work included)."""
    old = harness.time
    harness.time = _NoSleep
    try:
        t0 = pc()
        r = fn()
        sync = pc() - t0
        blocked, last = 0.0, pc()
        while pc() - last < quiet and pc() - t0 < 180:
            a = pc()
            QApplication.processEvents()
            d = pc() - a
            if d > 0.015:
                blocked += d
                last = pc()
            if ex.w.worker.busy():
                last = pc()
            time.sleep(0.002)
        done = last - t0
    finally:
        harness.time = old
    rec = {"size": SIZE, "gesture": gesture, "rep": rep, "sync": round(sync, 3), "freeze": round(sync + blocked, 3),
           "done": round(done, 3), **(extra or {})}
    MEAS.append(rec)
    P("MEASURE", json.dumps(rec, ensure_ascii=False))
    return {"ret": r, **rec}


def rss_mb() -> float:
    import psutil
    return round(psutil.Process(os.getpid()).memory_info().rss / 2 ** 20, 1)


def check(name: str, **kw):
    P("CHECK", json.dumps({"size": SIZE, "check": name, **kw}, ensure_ascii=False, default=str))


def blocks_by(key):
    return [b for b in ex.blocks() if b.node.key == key]


def rect(b) -> QRectF:
    from astro_gpu.client.graph import BH
    return b.mapRectToScene(QRectF(0, 0, b.w, BH))


def overlaps():
    bs = ex.blocks()
    out = []
    for i, a in enumerate(bs):
        ra = rect(a).adjusted(2, 2, -2, -2)
        for b in bs[i + 1:]:
            if ra.intersects(rect(b).adjusted(2, 2, -2, -2)):
                out.append((a.title, b.title, [round(x) for x in rect(a).getRect()], [round(x) for x in rect(b).getRect()]))
    return out


def graph_texts():
    out = []
    for it in ex.view.scene().items():
        for attr in ("title", "summary"):
            v = getattr(it, attr, None)
            if isinstance(v, str) and v:
                out.append(v)
        if hasattr(it, "toPlainText"):
            t = it.toPlainText()
            if t:
                out.append(t)
    return out


def wheel(n=10, dy=-120):
    vp = ex.view.viewport()
    c = QPointF(vp.rect().center())
    for _ in range(n):
        e = QWheelEvent(c, QPointF(vp.mapToGlobal(vp.rect().center())), QPoint(0, 0), QPoint(0, dy), Qt.NoButton,
                        Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(vp, e)
        vp.repaint()


def key(k, mod=Qt.NoModifier, widget=None):
    w = widget or ex.view.viewport()
    ex.w.activateWindow()
    QApplication.processEvents()
    w.setFocus()
    QApplication.processEvents()
    QTest.keyClick(w, k, mod)


def click_block(b):
    vp = ex.view.viewport()
    p = ex.point_of(b)
    QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, p)


def counts():
    return {"blocks": len(ex.blocks()), "workflows": len(ex.s.paths), "sessions": len(ex.s.p.sessions),
            "layers": len((ex.s.p.post or {}).get("layers") or [])}


# ------------------------------------------------------------------ the run
t_build = pc()
root = build(f"gros_{SIZE}")
P("BUILD", root, round(pc() - t_build, 2), "s")
ex.open()
ex.settle(10)
ex.w.open_project(root)
ex.settle(20)
t_en = pc()
enrich()
P("ENRICH time", round(pc() - t_en, 2))
check("counts after setup", **counts())
ex.w.close_project()
ex.settle(10)
rss0 = rss_mb()

for rep in (1, 2):
    ex.w.close_project()
    ex.settle(10)
    m = measure("open project", lambda: ex.w.open_project(root), rep, quiet=1.0)
    check("rss after open", rep=rep, rss_mb=rss_mb(), before=rss0)
check("counts", **counts())
ex.shot(f"{SIZE}_01_open")

# frame counts on the Session blocks and in the status line
check("block keys", keys=sorted({b.node.key for b in ex.blocks()}))
ses = blocks_by("session") or blocks_by("sources")
check("session blocks", n=len(ses), texts=[(b.title, b.summary) for b in ses][:30])
check("status", text=ex.w.status.text(), run=(ex.w.bt_run.text(), ex.w.bt_run.isEnabled(), ex.w.bt_run.toolTip()))
ov = overlaps()
check("overlaps", n=len(ov), first=ov[:12])
cut = [t for t in graph_texts() if "…" in t]
check("cut texts", n=len(cut), first=cut[:20])
sr = ex.view.scene().sceneRect()
check("scene", w=round(sr.width()), h=round(sr.height()), scale=round(ex.view.transform().m11(), 3),
      bands=[getattr(it, "text", "") for it in ex.view.scene().items() if it.__class__.__name__ == "BandItem"][:40])

# paint, scroll, Show all
for rep in (1, 2):
    measure("paint graph", lambda: ex.view.viewport().grab(), rep)
for rep in (1, 2):
    measure("scroll 10 wheel steps", lambda: wheel(10), rep)
    measure("scroll back", lambda: wheel(10, 120), rep)
for rep in (1, 2):
    measure("Show all on", lambda: (ex.view.b_all.click(), ex.view.viewport().repaint()), rep)
    if rep == 1:
        ex.shot(f"{SIZE}_02_show_all")
    measure("Show all off", lambda: (ex.view.b_all.click(), ex.view.viewport().repaint()), rep)

# select, place from the palette, Del, Ctrl+Z, Ctrl+Y
for rep in (1, 2):
    tgt = next((b for b in blocks_by("debruitage") if b.node.paths and "Couleur" in b.node.paths[0].name), None) \
        or (blocks_by("fond") or ex.blocks())[-1]
    ex.view.centerOn(tgt)
    ex.settle(3)
    measure("select block", lambda: click_block(tgt), rep)
    before = len(ex.blocks())
    measure("palette double-click (Gradient)", lambda: QTest.mouseDClick(ex.card("gradient"), Qt.LeftButton,
                                                                         Qt.NoModifier, QPoint(20, 12)), rep,
            {"blocks_before": before, "blocks_after": None})
    MEAS[-1]["blocks_after"] = len(ex.blocks())
    added = len(ex.blocks()) - before
    measure("Del (selected block)", lambda: key(Qt.Key_Delete), rep)
    after_del = len(ex.blocks())
    measure("Ctrl+Z", lambda: key(Qt.Key_Z, Qt.ControlModifier, ex.w), rep)
    after_z = len(ex.blocks())
    measure("Ctrl+Y", lambda: key(Qt.Key_Y, Qt.ControlModifier, ex.w), rep)
    check("place/del/undo/redo", rep=rep, added=added, after_del=after_del, after_z=after_z, after_y=len(ex.blocks()),
          before=before, toast=ex.page.toast_text() if hasattr(ex.page, "toast_text") else None)
    measure("Ctrl+Z (back to start)", lambda: key(Qt.Key_Z, Qt.ControlModifier, ex.w), rep)
    key(Qt.Key_Z, Qt.ControlModifier, ex.w)
    ex.settle(6)
    check("back to start", rep=rep, blocks=len(ex.blocks()), before=before)

# a real drag from the palette onto a place (ghost) of a big graph
for rep in (1, 2):
    tgt = (blocks_by("empilement") or ex.blocks())[0]
    ex.view.centerOn(tgt)
    click_block(tgt)
    ex.settle(4)
    tgt = (blocks_by("empilement") or ex.blocks())[0]
    t0 = pc()
    ex.drag_palette_hover("denoise", ex.point_of(tgt))
    t_hover = pc() - t0
    gh = ex.ghosts()
    pts = [ex.ghost_center(g) for g in gh]
    ex.shot(f"{SIZE}_07_drag_ghosts_{rep}") if rep == 1 else None
    ex.drag_leave()
    if gh:
        pt = pts[0]
        before = len(ex.blocks())
        measure("drag palette onto a place (Denoise)", lambda: ex.drag_palette("denoise", pt), rep,
                {"hover_s": round(t_hover, 3), "ghosts": len(gh)})
        check("drag drop", rep=rep, before=before, after=len(ex.blocks()), ghosts=len(gh))
        measure("Ctrl+Z (drag)", lambda: key(Qt.Key_Z, Qt.ControlModifier, ex.w), rep)
    else:
        check("drag drop", rep=rep, ghosts=0, hover_s=round(t_hover, 3))


# rename a session from its block's menu (the name dialog answered)
def rename_session(new):
    b = (blocks_by("session") or blocks_by("sources"))[0]
    ex.view.centerOn(b)
    ex.settle(2)
    box = {}

    def h(d):
        e = d.findChildren(QLineEdit)
        if e:
            e[0].setText(new)
            box["t"] = pc()
            d.accept()
        else:
            d.reject()
    ex.dialog_handler = h
    ex.right_click_block(b, "Renommer session" if LANG == "fr" else "Rename session")
    return box


for rep in (1, 2):
    box = {}
    measure("rename session (menu + name dialog, incl. ~60 ms of the harness's popup timer)",
            lambda: box.update(rename_session(f"Nuit renommée {rep}")), rep)
    check("rename", rep=rep, names=[s.name for s in ex.s.p.sessions][:3], got_dialog="t" in box)
for rep in (1, 2):
    sid = ex.s.p.sessions[0].id
    measure("rename session (model call only: the refresh)", lambda: ex.s.rename_session(sid, f"Nuit R{rep}"), rep)


def big_session():
    bs = blocks_by("session") or blocks_by("sources")
    return next((b for b in bs if "Couleur nuit 1" in b.title), bs[0] if bs else None)


# the Session panel of the big session
if big_session() is not None:
    ex.view.centerOn(big_session())
    ex.settle(3)
    for rep in (1, 2):
        click_block(next(b for b in ex.blocks() if b.node.key == "calibration"))
        ex.settle(4)
        measure("select Session block (panel)", lambda: click_block(big_session()), rep)
    ex.shot(f"{SIZE}_03_session_panel")
    from PySide6.QtWidgets import QLabel
    panel_txt = [x.text() for x in ex.w.findChildren(QLabel) if x.isVisible() and x.text() and
                 any(c.isdigit() for c in x.text())][:40]
    check("session panel numbers", texts=panel_txt)

# F5: the run window
for rep in (1, 2):
    ex.close_dialogs()
    ex.w.run.dialog = None if ex.w.run.dialog is not None and not ex.w.run.dialog.isVisible() else ex.w.run.dialog
    measure("F5 (run window)", lambda: key(Qt.Key_F5), rep, quiet=0.8)
    d = ex.w.run.dialog if ex.w.run.dialog is not None and ex.w.run.dialog.isVisible() else None
    if d is not None:
        check("run window", rep=rep, rows=len(d.rows), title=d.windowTitle(), summary=d.summary.text(),
              dsize=(d.width(), d.height()), detail=d.detail.text()[:300])
        if rep == 1:
            ex.shot(f"{SIZE}_04_run_window", d)
        measure("close run window", lambda: d.reject(), rep)
    else:
        check("run window", rep=rep, dialog=None, status=ex.w.status.text(), popups=ex.popups[-2:])

# Post page
for rep in (1, 2):
    measure("open Post page", lambda: ex.click(ex.w.mode_btns[2]), rep, quiet=1.0)
    if rep == 1:
        ex.shot(f"{SIZE}_05_post")
        try:
            v = ex.w.post_holder.page.view
            check("post layers", n=len(v.layers), names=[x.get("name") for x in v.layers][:60])
        except Exception as e:  # noqa: BLE001
            check("post layers", error=repr(e))
    measure("back to graph", lambda: ex.click(ex.w.mode_btns[0]), rep, quiet=1.0)

# small screen
ex.w.resize(1366, 768)
ex.settle(10)
ex.shot(f"{SIZE}_06_1366")
check("rss end", rss_mb=rss_mb())
check("popups", popups=ex.popups[-6:], modals=ex.modals[-6:])
with (OUT / f"measures_{SIZE}.json").open("w", encoding="utf-8") as fh:
    json.dump(MEAS, fh, ensure_ascii=False, indent=1)
ex.close()
P("DONE", SIZE)
