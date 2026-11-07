"""post_images (round 4): the Post module on a real computed project (a copy of a bench project), real gestures.

Helpers shared by the explore5_post_images_*.py scripts: open the copy on Post, wait for the renders, drag a slider
with the mouse and time the image that follows, save what the screen shows."""
from __future__ import annotations

import faulthandler
import json
import os
import sys
import time
import traceback
from pathlib import Path

from harness import Explorer, stray_qt_errors  # noqa: F401  (sets the environment first)
import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QSlider, QStyle, QStyleOptionSlider  # noqa: E402

WORK = Path("F:/astro_essais/_explore4_work/post_images")
COPY = WORK / "iris_copy"
SLOT_ERRORS: list[str] = []


def _hook(t, v, tb):
    SLOT_ERRORS.append("".join(traceback.format_exception(t, v, tb)))
    sys.__excepthook__(t, v, tb)


def start(timeout: int = 1500):
    faulthandler.dump_traceback_later(timeout, exit=True)
    stray_qt_errors()
    sys.excepthook = _hook


def R(tag, *a):
    print(f"R[{tag}]", *a, flush=True)


class PI:
    def __init__(self, ex: Explorer):
        self.ex = ex
        self.events: list[tuple] = []                 # (t, "req"|"res", kind, seq, ms)
        self._hooked = None

    @property
    def v(self):
        return self.ex.w.post_holder.page.view

    def open(self, root: Path = COPY, timeout=600):
        ex = self.ex
        t0 = time.monotonic()
        ex.open(root)
        R("open", f"project opened in {time.monotonic() - t0:.1f}s", "paths",
          [(q.pid, q.name, q.rendered) for q in ex.s.paths])
        t1 = time.monotonic()
        ex.click(ex.w.mode_btns[2])
        ex.settle(10)
        self.hook()
        ok = self.wait(lambda: self.v.first_image and self.v.src.img1 is not None, timeout)
        R("open", f"Post first image {'ok' if ok else 'NOT SHOWN'} after {time.monotonic() - t1:.1f}s",
          "device", self.v.worker.device, "canvas", self.v.src.canvas, "banner", self.v.viewer.banner)
        return ok

    def hook(self):
        v = self.v
        if self._hooked is v:
            return
        self._hooked = v
        w = v.worker
        orig = w.request

        def req(r, _o=orig):
            seq = _o(r)
            self.events.append((time.monotonic(), "req", r.kind, seq, 0.0))
            return seq
        w.request = req
        v.rendered.connect(lambda res: self.events.append((time.monotonic(), "res", res.kind, res.seq, res.ms)))

    def wait(self, pred, timeout=120.0, step=4) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self.ex.settle(step)
            try:
                if pred():
                    return True
            except Exception:  # noqa: BLE001
                pass
        return False

    def idle(self, timeout=300.0, quiet=1.5) -> bool:
        """No request pending: the last request answered (or nothing new for ``quiet`` seconds after it)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self.ex.settle(4)
            v = self.v
            busy = v._draft_t.isActive() or v._final_t.isActive() or v.worker.busy()
            if not busy:
                t1 = time.monotonic()
                while time.monotonic() - t1 < quiet:
                    self.ex.settle(2)
                if not (v._draft_t.isActive() or v._final_t.isActive() or v.worker.busy()):
                    return True
        return False

    # ------------------------------------------------------------ timing the image after a gesture
    def mark(self) -> int:
        return len(self.events)

    def after(self, i0: int, t_ref: float):
        """The requests and answers since ``i0``, their time from ``t_ref`` (s)."""
        return [(round(t - t_ref, 2), what, kind, seq, round(ms)) for t, what, kind, seq, ms in self.events[i0:]]

    def time_to_image(self, i0: int, t_rel: float, timeout=300.0) -> dict:
        """After a release at ``t_rel``: the first image that answers a request made after it (any kind but thumbs),
        and the final one (half, or refine when the CPU is zoomed out)."""
        def done():
            reqs = [e for e in self.events[i0:] if e[1] == "req" and e[0] >= t_rel - 0.001 and e[2] in
                    ("half", "refine")]
            if not reqs:
                return False
            last = max(e[3] for e in reqs)
            return any(e[1] == "res" and e[3] >= last and e[2] in ("half", "refine") for e in self.events[i0:]) and \
                not self.v._final_t.isActive()
        ok = self.wait(done, timeout, step=2)
        reqs = [e for e in self.events[i0:] if e[1] == "req" and e[0] >= t_rel - 0.001 and e[2] != "thumbs"]
        first_seq = min((e[3] for e in reqs), default=10 ** 9)
        res = [e for e in self.events[i0:] if e[1] == "res" and e[2] != "thumbs" and e[3] >= first_seq]
        first = res[0] if res else None
        finals = [e for e in res if e[2] in ("half", "refine")]
        last = finals[-1] if finals else None
        return {"ok": ok, "first": None if first is None else (first[2], round(first[0] - t_rel, 2)),
                "final": None if last is None else (last[2], round(last[0] - t_rel, 2), round(last[4])),
                "kinds": [(e[2], round(e[0] - t_rel, 2)) for e in res]}

    # ------------------------------------------------------------ gestures
    def handle_point(self, s: QSlider) -> QPoint:
        opt = QStyleOptionSlider()
        s.initStyleOption(opt)
        r = s.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, s)
        return r.center()

    def groove_x(self, s: QSlider, frac: float) -> int:
        opt = QStyleOptionSlider()
        s.initStyleOption(opt)
        g = s.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, s)
        h = s.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, s)
        lo, hi = g.left() + h.width() // 2, g.right() - h.width() // 2
        return int(lo + (hi - lo) * frac)

    def drag_slider(self, cs, to_frac: float, steps: int = 8, pause: float = 0.06) -> float:
        """The handle of a ContSlider pressed, moved to ``to_frac`` of the groove in ``steps`` moves (each followed by
        the event loop, as a hand would), let go. Returns the release time."""
        s = cs.s
        a = self.handle_point(s)
        bx = self.groove_x(s, to_frac)
        ex = self.ex
        ex._mouse(s, QEvent.MouseButtonPress, a)
        ex.settle(1)
        for k in range(1, steps + 1):
            p = QPoint(int(a.x() + (bx - a.x()) * k / steps), a.y())
            ex._mouse(s, QEvent.MouseMove, p, Qt.NoButton, Qt.LeftButton)
            QApplication.processEvents()
            t = time.monotonic()
            while time.monotonic() - t < pause:
                QApplication.processEvents()
                time.sleep(0.005)
        t_rel = time.monotonic()
        ex._mouse(s, QEvent.MouseButtonRelease, QPoint(bx, a.y()), Qt.LeftButton, Qt.NoButton)
        return t_rel

    def viewer_drag(self, a: QPoint, b: QPoint, steps=10, button=Qt.LeftButton, mods=Qt.NoModifier):
        vw = self.v.viewer
        ex = self.ex

        def send(kind, p, btn, btns):
            e = QMouseEvent(kind, QPointF(p), QPointF(vw.mapToGlobal(p)), btn, btns, mods)
            QApplication.sendEvent(vw, e)
        send(QEvent.MouseButtonPress, a, button, button)
        ex.settle(1)
        for k in range(1, steps + 1):
            p = QPoint(int(a.x() + (b.x() - a.x()) * k / steps), int(a.y() + (b.y() - a.y()) * k / steps))
            send(QEvent.MouseMove, p, Qt.NoButton, button)
            ex.settle(1)
        t = time.monotonic()
        send(QEvent.MouseButtonRelease, b, button, Qt.NoButton)
        return t

    # ------------------------------------------------------------ evidence
    def screen_array(self) -> np.ndarray | None:
        """What the viewer is fed now at half resolution (uint8 RGB)."""
        a = self.v.src._half()
        return None if a is None else np.array(a)

    def save_png(self, a: np.ndarray, name: str) -> str:
        from PIL import Image
        f = self.ex.out / f"{name}.png"
        Image.fromarray(np.ascontiguousarray(a)).save(f)
        return str(f)

    def shot_viewer(self, name: str) -> str:
        return self.ex.shot(name, self.v.viewer)

    def layers_desc(self):
        return [(lay["id"], lay["kind"], lay["name"], lay["visible"], lay.get("target"), lay["mode"],
                 round(lay["opacity"]), bool(lay.get("mask"))) for lay in self.v.layers]

    def rows_desc(self):
        return [(r.n.text(), r.sub.text() if r.sub.isVisible() else "") for r in self.v.rows]

    def slot_errors(self):
        out, SLOT_ERRORS[:] = list(SLOT_ERRORS), []
        return out


def stats(a: np.ndarray) -> dict:
    a = a.astype(np.float64)
    return {"mean": [round(float(a[..., c].mean()), 2) for c in range(3)],
            "p50": [round(float(np.percentile(a[..., c], 50)), 1) for c in range(3)],
            "p99": [round(float(np.percentile(a[..., c], 99)), 1) for c in range(3)]}


def dump_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
