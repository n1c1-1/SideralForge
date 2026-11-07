"""palette_matrix round 2: the palette at 1366x768 and 1920x1080 (PM_SIZE), fr/en (PM_LANG): every card readable (its
text not cut), the hint of each greyed card inside the window and not over the palette, the places while dragging."""
from __future__ import annotations

from explore2_palette_lib import LANG, SIZE, Matrix

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QLabel

m = Matrix(f"layout_{LANG}_{SIZE[0]}")
ex = m.ex
tag = f"{LANG}_{SIZE[0]}x{SIZE[1]}"
for kind in ("legacy", "image", "mono"):
    ex.open(ex.project(kind))
    ex.settle(10)
    if kind == "image":
        ex.click_block(ex.block("image"))
    pg = ex.page
    cut = []
    for b, c in pg.cards.items():
        for lab in c.findChildren(QLabel):
            fm = lab.fontMetrics()
            if not lab.wordWrap() and fm.horizontalAdvance(lab.text()) > lab.width():
                cut.append((b, lab.text()))
            if lab.wordWrap() and lab.heightForWidth(lab.width()) > lab.height() + 2:
                cut.append((b, lab.text(), "wrapped higher than shown"))
        if not c.visibleRegion().boundingRect().contains(c.rect().adjusted(1, 1, -1, -1)):
            cut.append((b, "card not fully visible (palette scrolls)"))
    bar = pg.palette_bar
    print(kind, tag, "palette scroll range", bar.verticalScrollBar().maximum(), "cut", cut, flush=True)
    ex.shot(f"layout_{tag}_{kind}")
    hints = []
    for b, c in pg.cards.items():
        if not c.reason:
            continue
        txt, geo = m.hint_of(b)
        if geo is None:
            hints.append((b, "no hint"))
            continue
        inside = pg.rect().contains(geo)
        over_pal = geo.intersects(bar.geometry())
        if not inside or over_pal:
            hints.append((b, txt, geo.getRect(), "outside" if not inside else "over palette"))
    print(kind, tag, "hint problems", hints, flush=True)
    c = next((b for b, c in pg.cards.items() if c.reason), None)
    if c:
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QEnterEvent
        from PySide6.QtWidgets import QApplication
        cc = pg.cards[c]
        p = QPointF(cc.width() / 2, cc.height() / 2)
        QApplication.sendEvent(cc, QEnterEvent(p, cc.mapToGlobal(p), cc.mapToGlobal(p)))
        ex.settle(2)
        ex.shot(f"layout_{tag}_{kind}_hint_{c}")
        QApplication.sendEvent(cc, QEvent(QEvent.Leave))
    vp = ex.view.viewport()
    ex.drag_palette_hover("gradient", QPoint(vp.width() // 2, 4))
    print(kind, tag, "scale while dragging", round(ex.view.transform().m11(), 3), "ghosts", len(ex.ghosts()), flush=True)
    ex.shot(f"layout_{tag}_{kind}_drag")
    ex.drag_leave()
ex.close()
