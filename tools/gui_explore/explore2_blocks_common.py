"""block_actions round 2: shared gestures for the recheck and the new exploration (real Qt events through the harness)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from explore_blocks_lib import EXC, TOASTS, drawn, overlaps, panel_info, state  # noqa: E402,F401
from harness import Explorer  # noqa: E402,F401

from PySide6.QtCore import QPoint, Qt  # noqa: E402,F401
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton  # noqa: E402,F401

CHAIN = ["calibration", "registration", "selection", "stacking", "astrometry", "gradient", "denoise", "color", "stars"]
SHORT = ["calibration", "registration", "stacking", "astrometry", "gradient", "color"]


class G:
    """Gestures on one Explorer."""

    def __init__(self, ex: Explorer):
        self.ex = ex

    def act(self):
        self.ex.w.activateWindow()
        self.ex.settle(3)

    def place(self, *blocks):
        for b in blocks:
            self.ex.double_click_card(b)
            self.ex.settle(4)

    def new(self, kind, chain=(), clear=True):
        ex = self.ex
        if ex.w is not None and ex.w.s is not None:
            ex.w.s._dirty = False
        ex.open(ex.project(kind))
        self.act()
        self.place(*chain)
        self.act()
        if clear:
            ex.s._undo.clear()
            ex.s._redo.clear()

    def blk(self, key=None, pid=None, sid=None, title=None):
        if title is not None:
            return next((it for it in self.ex.blocks() if it.title == title), None)
        return self.ex.block(key, pid=pid, sid=sid)

    def menu(self, it, pick=None):
        m = self.ex.right_click_block(it, pick=pick)
        self.ex.settle(6)
        self.act()
        return m

    @staticmethod
    def texts(m):
        return [x.get("text") for x in m]

    def pid_of(self, name):
        return next(q.pid for q in self.ex.s.paths if q.name == name)

    def graph_of(self, name=None):
        s = self.ex.s
        q = next(q for q in s.paths if name is None or q.name == name)
        return [n["block"] + ("/off" if n.get("off") else "") for n in s.graph(q)]

    def titles(self):
        return [d["title"] for d in drawn(self.ex)]

    def toasts(self, n):
        return TOASTS[n:]

    def cross(self, it):
        ex = self.ex
        vp = ex.view.viewport()
        pt = ex.view.mapFromScene(it.mapToScene(it.w - 12, 12))
        ex.hover(vp, pt)
        ex.click(vp, pt)
        ex.settle(4)

    def ghost_idx(self, block, pid, idx):
        ex = self.ex
        ex.drag_palette_hover(block, QPoint(5, 5))
        gh = [g for g in ex.ghosts() if getattr(g, "idx", None) == idx and g.pid == pid]
        pt = ex.ghost_center(gh[0]) if gh else None
        ex.drag_leave()
        return pt

    def drop_idx(self, block, pid, idx) -> bool:
        pt = self.ghost_idx(block, pid, idx)
        if pt is None:
            return False
        self.ex.drag_palette(block, pt)
        self.ex.settle(6)
        self.act()
        return True

    def drag_branch(self, block, under_key, pid) -> bool:
        ex = self.ex
        ex.drag_palette_hover(block, QPoint(5, 5))
        gh = [g for g in ex.ghosts() if getattr(g, "branch_key", None) == under_key and g.pid == pid]
        pt = ex.ghost_center(gh[0]) if gh else None
        ex.drag_leave()
        if pt is None:
            return False
        ex.drag_palette(block, pt)
        ex.settle(6)
        self.act()
        return True

    def undo(self):
        self.act()
        self.ex.undo()
        self.ex.settle(4)

    def redo(self):
        self.act()
        self.ex.redo()
        self.ex.settle(4)

    def snap(self):
        """What a user sees changing: workflows (blocks, off, names, sessions), sessions, the Post layers."""
        st = state(self.ex)
        return json.dumps({"w": [(w["name"], w["blocks"], w["sessions"], w["source"]) for w in st["workflows"]],
                           "s": st["sessions"], "post": st["post"]}, sort_keys=True, default=str)

    def mono_fusion(self):
        self.new("mono", clear=False)
        for name in ("Ha", "OIII"):
            self.ex.click_block(self.blk("empilement", pid=self.pid_of(name)))
            self.place("astrometry")
        self.place("fusion")
        self.act()
        self.ex.s._undo.clear()
        self.ex.s._redo.clear()
