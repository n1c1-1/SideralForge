"""Round 2 fuzz: the language switched at runtime (Settings), on an open project after a few edits: does every text of
the window (graph, palette, panel, menus) follow? Compared with the same project opened fresh in that language."""
import os
import re
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

fz = G.Fuzz2("fr")
fz.panels = False
ex = fz.ex


def texts():
    fz.ensure_graph([])
    out = set()
    for it in fz.items():                              # the graph, the palette, the menu bar, the mode buttons
        out.add(fz.title(it))
        out.add(str(it.summary or ""))
        out.add((it.toolTip() or "").split("\n")[-1])
    for c in ex.page.cards.values():
        out.add(c.title)
        out.add(c.reason or "")
    for a in ex.w.menuBar().actions():
        out.add(a.text())
        for a2 in (a.menu().actions() if a.menu() else []):
            out.add(a2.text())
    for b in ex.w.mode_btns:
        out.add(b.text())
    return {t.strip() for t in out if t and t.strip()}


for k in range(2):
    for tpl in ("sho", "classic_dn", "image_colours"):
        for a, b in (("fr", "en"), ("en", "fr")):
            # switched: open in a, a few edits, switch to b
            acts = [["dbl_card", "Gradient", 0.5, 0.5], ["click_block", 0.5, 0.5, 0.5]]
            res = fz.run({"seed": None, "start": {"template": tpl, "lang": a}, "actions": acts}, f"ls{k}{tpl}{a}",
                         watchdog=60)
            root = ex.w.project.root
            cur = ex.s.current
            from astro_gpu.client import prefs
            prefs.set("language", a)               # the harness applies a language without the setting: aligned
            ex.key(Qt.Key_Comma, ex.w, Qt.ControlModifier)          # Settings: the handler switches the language
            fz.wait_ready()
            switched = texts()
            # fresh: the same project reopened in b
            ex.w.close_project()
            ex.settle(4)
            ex.w.open_project(root)
            fz.wait_ready()
            fresh = texts()
            only_sw = sorted(switched - fresh)
            only_fr = sorted(fresh - switched)
            from astro_gpu.client.i18n import language
            print(k, tpl, a, "→", b, "now", language(), sorted(switched)[:4], "| only after the switch:", only_sw[:12], "| only when opened fresh:",
                  only_fr[:12], flush=True)
            if k == 1 and only_sw:
                ex.w.close_project()
                ex.settle(2)
                ex.w.open_project(root)
os._exit(0)
