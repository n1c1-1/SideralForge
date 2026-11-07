"""Words, round 3: the menu bar's Run menu and the Open dialog's filters, as the user reads them."""
from __future__ import annotations

import sys

from harness import Explorer

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
ex = Explorer("words_layout", lang=LANG, size=(1366, 768))
ex.open(ex.project("session"))
ex.settle(6)
for m in ex.w.menuBar().actions():
    if m.menu() is not None:
        print("MENU", m.text(), "->", [a.text() for a in m.menu().actions() if not a.isSeparator()], flush=True)
ex.close()
