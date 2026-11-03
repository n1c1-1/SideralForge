"""Python's cyclic garbage collector, run on the GUI thread only (``GuiGarbageCollector``).

The client has worker threads (services/, the image viewer's loaders). Python starts a collection in whichever thread
happens to allocate when the thresholds are crossed; if that collection frees a Qt object kept in a reference cycle,
the object is destroyed outside the GUI thread, which can crash the process (an access violation on Windows). The
usual remedy for PyQt / PySide programs: turn the automatic collection off and run it from a timer of the GUI thread,
with the same thresholds (the recipe of calibre and pyqtgraph). Objects without cycles are freed at once as always.
"""
from __future__ import annotations

import gc

from PySide6.QtCore import QObject, QTimer


class GuiGarbageCollector(QObject):
    """Disables the automatic collection and collects every ``interval_ms`` on the GUI thread, the generations whose
    counts passed their thresholds, as the automatic collection would."""

    def __init__(self, parent: QObject | None = None, interval_ms: int = 500):
        super().__init__(parent)
        self.threshold = gc.get_threshold()
        gc.disable()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.check)
        self._timer.start(interval_ms)

    def check(self) -> None:
        c0, c1, c2 = gc.get_count()
        if c0 > self.threshold[0]:
            gc.collect(0)
            if c1 > self.threshold[1]:
                gc.collect(1)
                if c2 > self.threshold[2]:
                    gc.collect(2)
