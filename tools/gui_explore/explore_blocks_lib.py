"""block_actions: every action on every drawn block, in every project state (helpers; the scenarios are in
explore_blocks_run.py). Builds graphs through the GUI, keeps a master copy of each, and replays each action on a fresh
copy: what it changed, Ctrl+Z, Ctrl+Y, the words of menus and panel, overlaps, popups, exceptions."""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import TMP, Explorer  # noqa: E402,F401

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QPushButton  # noqa: E402

from astro_gpu.client import graph as graphmod  # noqa: E402

EXC: list[str] = []
TOASTS: list[str] = []
LOGS: list[str] = []


def _hook(t, v, tb):
    EXC.append("".join(traceback.format_exception(t, v, tb))[-3000:])
    sys.__stderr__.write("EXC " + str(v) + "\n")


sys.excepthook = _hook
_orig_toast = graphmod.GraphView.show_toast


def _toast(self, text):
    TOASTS.append(str(text))
    return _orig_toast(self, text)


graphmod.GraphView.show_toast = _toast


# ------------------------------------------------------------------ the model's state, for exact comparisons
def state(ex) -> dict:
    s = ex.s
    out = {"workflows": [], "sessions": [], "stack_mode": s.p.stack_mode, "post": []}
    for q in s.paths:
        w = s.p.workflow(q.pid)
        src = q.source
        if isinstance(src, dict):
            src = json.loads(json.dumps(src, default=str))
        out["workflows"].append({"pid": q.pid, "name": q.name, "source": src, "sessions": w.sessions, "end": w.end,
                                 "blocks": None if w.blocks is None else
                                 [(n["block"], n.get("name"), n.get("method"), bool(n.get("off"))) for n in w.blocks],
                                 "opts": json.loads(json.dumps(q.opts, default=str)),
                                 "bp": sorted(s.breakpoints(q)) if hasattr(s, "breakpoints") else []})
    for x in s.p.sessions:
        out["sessions"].append({"id": x.id, "name": x.name, "filter": x.filter, "lights": len(x.lights)})
    try:
        out["post"] = [{k: lay.get(k) for k in ("kind", "pid", "part", "name", "visible")} for lay in s.post_layers()]
    except Exception as e:  # noqa: BLE001
        out["post"] = f"error {e}"
    return out


def diff(a: dict, b: dict) -> list[str]:
    out = []
    for k in a:
        if a[k] != b[k]:
            if k == "workflows":
                pa = {w["pid"]: w for w in a[k]}
                pb = {w["pid"]: w for w in b[k]}
                for pid in sorted(set(pa) | set(pb)):
                    if pa.get(pid) != pb.get(pid):
                        if pid not in pa:
                            out.append(f"+workflow {pb[pid]['name']}: {pb[pid]['blocks']}")
                        elif pid not in pb:
                            out.append(f"-workflow {pa[pid]['name']}")
                        else:
                            for f in pa[pid]:
                                if pa[pid][f] != pb[pid][f]:
                                    out.append(f"workflow {pa[pid]['name']}.{f}: {pa[pid][f]} -> {pb[pid][f]}")
            else:
                out.append(f"{k}: {a[k]} -> {b[k]}")
    return out


# ------------------------------------------------------------------ what is drawn
def bid(it) -> tuple:
    n = it.node
    return (n.key, n.sid, tuple(sorted(q.name for q in n.paths)), n.lid is not None)


def find(ex, b: tuple, nth: int = 0):
    got = [it for it in ex.blocks() if bid(it) == b]
    got.sort(key=lambda it: (it.pos().x(), it.pos().y()))
    return got[nth] if len(got) > nth else None


def drawn(ex) -> list[dict]:
    out = []
    for it in sorted(ex.blocks(), key=lambda i: (i.pos().x(), i.pos().y())):
        out.append({"bid": bid(it), "title": it.title, "summary": it.summary, "removable": it.removable,
                    "ports": [w for w, _y in it.ports], "selected": it.selected, "x": round(it.pos().x()),
                    "y": round(it.pos().y())})
    return out


def overlaps(ex) -> list[str]:
    from astro_gpu.client.graph import BH, GhostItem
    rects = []
    for it in ex.blocks():
        rects.append((f"{it.title}", QRectF(it.scenePos().x(), it.scenePos().y(), it.w, BH)))
    for g in ex.view.scene().items():
        if isinstance(g, GhostItem) and g.isVisible():
            r = g.mapRectToScene(g.rect) if hasattr(g, "rect") else g.sceneBoundingRect()
            rects.append((f"ghost:{getattr(g, 'text', '')}", r))
    out = []
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i][1], rects[j][1]
            inter = a.intersected(b)
            if inter.width() > 2 and inter.height() > 2:
                out.append(f"{rects[i][0]} / {rects[j][0]}")
    return out


def panel_info(ex) -> dict:
    pn = ex.page.panel
    titles, buttons, texts = [], [], []
    for x in pn.findChildren(QLabel):
        if x.isVisible() and x.text():
            texts.append(x.text())
            if x.objectName() == "h1" or "h1" in (x.property("class") or "") or x.font().pointSizeF() >= 12:
                titles.append(x.text())
    from PySide6.QtWidgets import QLineEdit
    for e in pn.findChildren(QLineEdit):
        if e.isVisible():
            titles.append("[edit]" + e.text())
    for b in pn.findChildren(QAbstractButton):
        if b.isVisible() and b.text():
            buttons.append({"text": b.text(), "enabled": b.isEnabled(), "tip": b.toolTip(), "checked": b.isChecked()})
    return {"titles": titles, "buttons": buttons, "texts": texts[:40]}


def panel_buttons(ex) -> list[str]:
    pn = ex.page.panel
    return [b.text() for b in pn.findChildren(QPushButton) if b.isVisible() and b.isEnabled() and b.text()]


def click_panel_button(ex, text: str, nth: int = 0) -> bool:
    pn = ex.page.panel
    got = [b for b in pn.findChildren(QPushButton) if b.isVisible() and b.isEnabled() and b.text() == text]
    if len(got) > nth:
        ex.click(got[nth])
        ex.settle(6)
        return True
    return False


# ------------------------------------------------------------------ fresh copies of a built graph
class Lab:
    """One scenario: a master project built once; each action replayed on a fresh copy."""

    def __init__(self, ex: Explorer, name: str):
        self.ex, self.name = ex, name
        self.master: Path | None = None
        self.k = 0
        self.records: list[dict] = []
        self.counts = {"blocks": 0, "actions": 0}

    def save_master(self):
        ex = self.ex
        ex.w.save()
        ex.settle(4)
        self.master = Path(ex.s.p.root) if hasattr(ex.s.p, "root") else None
        if self.master is None:
            self.master = Path(ex.w.project.root)

    def fresh(self):
        ex = self.ex
        self.k += 1
        dst = TMP / "ws" / f"{self.name}_c{self.k}"
        shutil.copytree(self.master, dst)
        if ex.w.s is not None:
            ex.w.s._dirty = False                        # a throw-away copy: no "save?" question when it closes
        ex.w.open_project(dst)
        ex.settle(10)
        if ex.w.mode != 0:
            ex.w.set_mode(0)
            ex.settle(4)
        ex.close_dialogs()
        ex.w.activateWindow()                            # offscreen: shortcuts (Ctrl+Z, Del, D) need an active window
        ex.settle(2)
        return dst

    def mark(self):
        ex = self.ex
        return {"popups": len(ex.popups), "modals": len(ex.modals), "menus": len(ex.menus), "exc": len(EXC),
                "toasts": len(TOASTS), "files": len(ex.file_asks), "drops": len(ex.drops)}

    def since(self, m):
        ex = self.ex
        return {"popups": ex.popups[m["popups"]:], "modals": ex.modals[m["modals"]:], "exc": EXC[m["exc"]:],
                "toasts": TOASTS[m["toasts"]:], "files": ex.file_asks[m["files"]:]}
