"""Indicator charts drawn with QPainter: the frames (star size and transparency) and the sensor's spectral response.

``FramesChart`` draws the rows of the frame table of the engine (``api.frame_table``: star size, transparency, the time
of each frame); ``SpectralChart`` the curves the Color stage fitted (``api.spectral_response``).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QWidget

from .i18n import _, number

BG, GRID, TEXT, DIM = QColor("#181b21"), QColor("#2a2f38"), QColor("#d8dce4"), QColor("#8b93a1")
KEEP, OUT, HAND = QColor("#5ea1ff"), QColor("#e06c6c"), QColor("#e6a23c")


class FramesChart(QWidget):
    """Star size and transparency of every frame, Siril / PixInsight style: thresholds dragged, points clicked.

    ``rows``: the frame table's rows (name, time, fwhm, transparency, status, excluded). Thresholds: ``tmin`` in % of the
    best frames, ``fmax`` in px. ``hand``: the indices excluded by hand."""
    changed = Signal()
    picked = Signal(int)

    def __init__(self, rows, tmin: float, fmax: float, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(300)
        self.rows = list(rows)
        self.names = [r.name for r in self.rows]
        self.times = [r.time or "" for r in self.rows]
        self.fwhm = np.array([r.fwhm for r in self.rows], float)
        tr = np.array([r.transparency for r in self.rows], float)
        self.aligned = np.array([r.status != "not aligned" for r in self.rows], bool)
        ok = tr[self.aligned & np.isfinite(tr)]
        self.p90 = float(np.percentile(ok, 90)) if len(ok) else 1.0     # the engine's reference: the best frames
        self.tr = 100 * tr / self.p90                                    # % of the best frames
        self.n = len(self.rows)
        self.tmin, self.fmax = float(tmin), float(fmax)                  # thresholds (%, px)
        self.sugg = (self.tmin, self.fmax)
        self.hand = {i for i, r in enumerate(self.rows) if r.excluded}
        self.hover = None
        self.current = None
        self._drag = None
        self._sel = None

    # ------------------------------------------------------------ geometry
    def _panels(self):
        L, R, T, gap = 58, 16, 26, 34
        h = (self.height() - T - 38 - gap) / 2
        top = QRectF(L, T, self.width() - L - R, h)
        bot = QRectF(L, T + h + gap, self.width() - L - R, h)
        return top, bot

    def _ranges(self):
        fw, tr = self.fwhm[np.isfinite(self.fwhm)], self.tr[np.isfinite(self.tr)]
        f0, f1 = min(fw.min(), self.fmax) * 0.95, max(fw.max(), self.fmax) * 1.05
        t0, t1 = min(tr.min(), self.tmin) * 0.95, max(tr.max(), self.tmin, 100) * 1.03
        return (f0, f1), (t0, t1)

    def _xy(self, r: QRectF, i, v, lo, hi):
        return QPointF(r.left() + (i + 0.5) / self.n * r.width(), r.bottom() - (v - lo) / (hi - lo) * r.height())

    def excluded(self) -> np.ndarray:
        """Left out by the thresholds, by hand, or because the frame did not align."""
        ex = (self.fwhm > self.fmax) | (self.tr < self.tmin) | ~self.aligned
        for i in self.hand:
            ex[i] = True
        return ex

    def kept(self) -> int:
        return int(self.n - self.excluded().sum())

    # ------------------------------------------------------------ drawing
    def paintEvent(self, e):                                         # noqa: N802 - Qt's name
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p):
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
        if not self.n:
            return
        top, bot = self._panels()
        (f0, f1), (t0, t1) = self._ranges()
        ex = self.excluded()
        for r, vals, lo, hi, title, unit, thr, below in (
                (top, self.fwhm, f0, f1, _("FWHM (px)"), "px", self.fmax, False),
                (bot, self.tr, t0, t1, _("Transparency (% of the best frames)"), "%", self.tmin, True)):
            p.setPen(QPen(GRID, 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            for k in range(5):
                v = lo + (hi - lo) * k / 4
                y = r.bottom() - k / 4 * r.height()
                p.setPen(QPen(GRID, 1, Qt.DotLine))
                p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
                p.setPen(DIM)
                p.drawText(QRectF(0, y - 8, r.left() - 6, 16), Qt.AlignRight | Qt.AlignVCenter,
                           number(v, 1) if unit == "px" else f"{v:.0f}")
            p.setPen(TEXT)
            p.drawText(QRectF(r.left(), r.top() - 22, r.width(), 18), Qt.AlignLeft | Qt.AlignVCenter, title)
            p.setPen(Qt.NoPen)
            for i in range(self.n):
                if not np.isfinite(vals[i]):
                    continue
                pt = self._xy(r, i, vals[i], lo, hi)
                p.setBrush(HAND if i in self.hand else (OUT if ex[i] else KEEP))
                rad = 2.2 if i != self.hover else 4.5
                p.drawEllipse(pt, rad, rad)
            if self.current is not None and np.isfinite(vals[self.current]):
                p.setPen(QPen(QColor("#ffffff"), 1.6))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(self._xy(r, self.current, vals[self.current], lo, hi), 6, 6)
                p.setPen(Qt.NoPen)
            y = self._xy(r, 0, thr, lo, hi).y()
            p.setPen(QPen(OUT, 1.6, Qt.DashLine))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            lab = (_("threshold {v} %  (drag)").format(v=f"{thr:.0f}") if below
                   else _("threshold {v} px  (drag)").format(v=number(thr, 1)))
            p.setPen(OUT)
            p.drawText(QRectF(r.right() - 220, y - 18 if y - 18 > r.top() else y + 2, 216, 16),
                       Qt.AlignRight | Qt.AlignVCenter, lab)
        # time axis: a tick where each new hour starts
        p.setPen(DIM)
        last = self.times[0][11:13]
        last_x = -1e9
        for i in range(1, self.n):
            hh = self.times[i][11:13]
            if hh and hh != last:
                last = hh
                x = bot.left() + (i + 0.5) / self.n * bot.width()
                if x - last_x < 44:                          # a gap in the night: keep the labels apart
                    continue
                last_x = x
                p.drawText(QRectF(x - 30, bot.bottom() + 4, 60, 16), Qt.AlignCenter, f"{hh} h")
        if self._sel:
            a, b = self._sel
            p.setPen(QPen(HAND, 1, Qt.DashLine))
            p.setBrush(QColor(230, 162, 60, 40))
            p.drawRect(QRectF(a, b).normalized())
        if self.hover is not None:
            i = self.hover
            txt = (f"{self.names[i]} · {self.times[i][11:16]} · {number(self.fwhm[i], 2)} px · {self.tr[i]:.0f} %"
                   + (" · " + _("excluded by hand") if i in self.hand else
                      (" · " + _("excluded") if ex[i] else "")))
            p.setPen(TEXT)
            p.drawText(QRectF(top.left(), 4, top.width(), 18), Qt.AlignRight | Qt.AlignVCenter, txt)

    # ------------------------------------------------------------ interaction
    def _hit(self, pos):
        top, bot = self._panels()
        (f0, f1), (t0, t1) = self._ranges()
        for r, vals, lo, hi in ((top, self.fwhm, f0, f1), (bot, self.tr, t0, t1)):
            if r.adjusted(-4, -4, 4, 4).contains(pos):
                i = int((pos.x() - r.left()) / r.width() * self.n)
                best, bd = None, 9.0
                for j in range(max(0, i - 3), min(self.n, i + 4)):
                    if not np.isfinite(vals[j]):
                        continue
                    d = (self._xy(r, j, vals[j], lo, hi) - pos).manhattanLength()
                    if d < bd:
                        best, bd = j, d
                return best
        return None

    def _thr_hit(self, pos):
        top, bot = self._panels()
        (f0, f1), (t0, t1) = self._ranges()
        if top.contains(pos) and abs(self._xy(top, 0, self.fmax, f0, f1).y() - pos.y()) < 6:
            return "f"
        if bot.contains(pos) and abs(self._xy(bot, 0, self.tmin, t0, t1).y() - pos.y()) < 6:
            return "t"
        return None

    def mousePressEvent(self, e):                                    # noqa: N802
        pos = e.position()
        t = self._thr_hit(pos)
        if t:
            self._drag = ("thr", t)
            return
        i = self._hit(pos)
        if i is not None:
            self.current = i
            self.picked.emit(i)
            self.update()
            return
        self._drag = ("sel", pos)
        self._sel = (pos, pos)

    def mouseMoveEvent(self, e):                                     # noqa: N802
        pos = e.position()
        d = self._drag
        top, bot = self._panels()
        (f0, f1), (t0, t1) = self._ranges()
        if d and d[0] == "thr":
            if d[1] == "f":
                self.fmax = float(f1 - (pos.y() - top.top()) / top.height() * (f1 - f0))
            else:
                self.tmin = float(np.clip(t1 - (pos.y() - bot.top()) / bot.height() * (t1 - t0), 0, 100))
        elif d and d[0] == "sel":
            self._sel = (d[1], pos)
        else:
            self.hover = self._hit(pos)
            self.setCursor(Qt.SizeVerCursor if self._thr_hit(pos) else
                           (Qt.PointingHandCursor if self.hover is not None else Qt.ArrowCursor))
        self.update()

    def mouseReleaseEvent(self, e):                                  # noqa: N802
        d = self._drag
        self._drag = None
        if d and d[0] == "thr":
            self.changed.emit()                              # once, when the threshold is dropped
        if d and d[0] == "sel" and self._sel:
            a, b = self._sel
            self._sel = None
            r = QRectF(a, b).normalized()
            if r.width() > 6 and r.height() > 6:
                top, bot = self._panels()
                (f0, f1), (t0, t1) = self._ranges()
                inside = set()
                for pr, vals, lo, hi in ((top, self.fwhm, f0, f1), (bot, self.tr, t0, t1)):
                    for i in range(self.n):
                        if np.isfinite(vals[i]) and r.contains(self._xy(pr, i, vals[i], lo, hi)):
                            inside.add(i)
                if inside:
                    m = QMenu(self)
                    a1 = m.addAction(_("Exclude these {n} frames").format(n=len(inside)))
                    a2 = m.addAction(_("Keep these {n} frames").format(n=len(inside)))
                    act = m.exec(e.globalPosition().toPoint())
                    if act is a1:
                        self.hand |= inside
                    elif act is a2:
                        self.hand -= inside
                    self.changed.emit()
        self.update()

    def reset(self):
        self.tmin, self.fmax = self.sugg
        self.hand.clear()
        self.changed.emit()
        self.update()


class SpectralChart(QWidget):
    """Spectral response of the sensor, reconstructed from the Gaia spectra of the field's stars."""

    LINES = [(486.1, "Hβ"), (500.7, "OIII"), (656.3, "Hα")]

    def __init__(self, lam, S, parent=None):
        super().__init__(parent)
        self.lam, self.S = np.array(lam), np.array(S)
        self.setMinimumHeight(260)

    def paintEvent(self, e):                                         # noqa: N802
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p):
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
        r = QRectF(52, 30, self.width() - 70, self.height() - 66)
        p.setPen(QPen(GRID, 1))
        p.drawRect(r)
        l0, l1 = 380.0, 720.0
        top = float(self.S.max()) * 1.08

        def xy(lam, v):
            return QPointF(r.left() + (lam - l0) / (l1 - l0) * r.width(), r.bottom() - v / top * r.height())

        for lam in range(400, 720, 50):
            x = xy(lam, 0).x()
            p.setPen(QPen(GRID, 1, Qt.DotLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(DIM)
            p.drawText(QRectF(x - 30, r.bottom() + 4, 60, 16), Qt.AlignCenter, f"{lam}")
        p.drawText(QRectF(r.left(), r.bottom() + 18, r.width(), 16), Qt.AlignCenter, _("wavelength (nm)"))
        p.setPen(TEXT)
        p.drawText(QRectF(r.left(), 6, r.width(), 18), Qt.AlignLeft | Qt.AlignVCenter,
                   _("Sensor response, rebuilt from the Gaia spectra of the field stars"))
        for lam, name in self.LINES:
            x = xy(lam, 0).x()
            p.setPen(QPen(QColor("#6b7280"), 1, Qt.DashLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(DIM)
            p.drawText(QRectF(x - 30, r.top() + 2, 60, 16), Qt.AlignCenter, name)
        for c, col in enumerate((QColor("#ff6b6b"), QColor("#5fd38d"), QColor("#5ea1ff"))):
            path = QPainterPath()
            first = True
            for lam, v in zip(self.lam, self.S[c]):
                if l0 <= lam <= l1:
                    pt = xy(lam, max(float(v), 0.0))
                    if first:
                        path.moveTo(pt)
                        first = False
                    else:
                        path.lineTo(pt)
            p.setPen(QPen(col, 2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
