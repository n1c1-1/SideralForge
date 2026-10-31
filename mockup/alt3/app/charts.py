"""Indicator charts drawn with QPainter: the frames (star size and transparency) and the sensor's spectral response."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QWidget

from i18n import _

BG, GRID, TEXT, DIM = QColor("#181b21"), QColor("#2a2f38"), QColor("#d8dce4"), QColor("#8b93a1")
KEEP, OUT, HAND = QColor("#5ea1ff"), QColor("#e06c6c"), QColor("#e6a23c")


class FramesChart(QWidget):
    """Star size and transparency of every frame, Siril / PixInsight style: thresholds dragged, points clicked."""
    changed = Signal()
    picked = Signal(int)

    def __init__(self, frames: dict, min_transparency: float, max_fwhm_ratio: float, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(300)
        self.f = frames
        self.fwhm = np.array(frames["fwhm"])
        tr = np.array(frames["transp"])
        self.p90 = float(np.percentile(tr, 90))
        self.tr = 100 * tr / self.p90                       # % of the best frames
        self.n = len(self.fwhm)
        self.med = float(np.median(self.fwhm))
        self.tmin = 100 * min_transparency                  # thresholds (%, px)
        self.fmax = max_fwhm_ratio * self.med
        self.sugg = (self.tmin, self.fmax)
        self.hand = set()
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
        f0, f1 = min(self.fwhm.min(), self.fmax) * 0.95, max(self.fwhm.max(), self.fmax) * 1.05
        t0, t1 = min(self.tr.min(), self.tmin) * 0.95, max(self.tr.max(), self.tmin, 100) * 1.03
        return (f0, f1), (t0, t1)

    def _xy(self, r: QRectF, i, v, lo, hi):
        return QPointF(r.left() + (i + 0.5) / self.n * r.width(), r.bottom() - (v - lo) / (hi - lo) * r.height())

    def excluded(self) -> np.ndarray:
        ex = (self.fwhm > self.fmax) | (self.tr < self.tmin)
        for i in self.hand:
            ex[i] = True
        return ex

    def kept(self) -> int:
        return int(self.n - self.excluded().sum())

    # ------------------------------------------------------------ drawing
    def paintEvent(self, e):
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p):
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
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
                           f"{v:.1f}" if unit == "px" else f"{v:.0f}")
            p.setPen(TEXT)
            p.drawText(QRectF(r.left(), r.top() - 22, r.width(), 18), Qt.AlignLeft | Qt.AlignVCenter, title)
            # points
            p.setPen(Qt.NoPen)
            for i in range(self.n):
                pt = self._xy(r, i, vals[i], lo, hi)
                c = HAND if i in self.hand else (OUT if ex[i] else KEEP)
                p.setBrush(c)
                p.drawEllipse(pt, 2.2 if i != self.hover else 4.5, 2.2 if i != self.hover else 4.5)
            if self.current is not None:
                p.setPen(QPen(QColor('#ffffff'), 1.6))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(self._xy(r, self.current, vals[self.current], lo, hi), 6, 6)
                p.setPen(Qt.NoPen)
            # threshold
            y = self._xy(r, 0, thr, lo, hi).y()
            p.setPen(QPen(OUT, 1.6, Qt.DashLine))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            lab = (_("threshold {v} %  (drag)").format(v=f"{thr:.0f}") if below
                   else _("threshold {v} px  (drag)").format(v=f"{thr:.1f}"))
            p.setPen(OUT)
            p.drawText(QRectF(r.right() - 220, y - 18 if y - 18 > r.top() else y + 2, 216, 16),
                       Qt.AlignRight | Qt.AlignVCenter, lab)
        # time axis: a tick where each new hour starts
        p.setPen(DIM)
        times = self.f["time"]
        last, last_x = times[0][:2], -1e9
        for i in range(1, self.n):
            hhmm = times[i]
            if hhmm[:2] != last:
                last = hhmm[:2]
                x = bot.left() + (i + 0.5) / self.n * bot.width()
                if x - last_x < 44:                  # a gap in the night: keep the labels apart
                    continue
                last_x = x
                p.drawText(QRectF(x - 30, bot.bottom() + 4, 60, 16), Qt.AlignCenter, f"{hhmm[:2]} h")
        if self._sel:
            a, b = self._sel
            p.setPen(QPen(HAND, 1, Qt.DashLine))
            p.setBrush(QColor(230, 162, 60, 40))
            p.drawRect(QRectF(a, b).normalized())
        if self.hover is not None:
            i = self.hover
            txt = (f"{self.f['name'][i]} · {times[i]} · {self.fwhm[i]:.2f} px · {self.tr[i]:.0f} %"
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

    def mousePressEvent(self, e):
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

    def mouseMoveEvent(self, e):
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

    def mouseReleaseEvent(self, e):
        d = self._drag
        self._drag = None
        if d and d[0] == "thr":
            self.changed.emit()             # once, when the threshold is dropped
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
                        if r.contains(self._xy(pr, i, vals[i], lo, hi)):
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

    def paintEvent(self, e):
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
