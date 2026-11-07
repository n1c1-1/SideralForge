"""Round 4, clavier_fenetre: the Post page shown, language switched, Ctrl+Z / Ctrl+Y: which page stays shown. Prints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from explore_post_common import P, start  # noqa: E402
from explore2_blocks_common import G, Explorer  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from PySide6.QtWidgets import QComboBox, QApplication  # noqa: E402

start(600)
ex = Explorer("clavier_fenetre")
prefs.set("language", "fr")
g = G(ex)
pp = P(ex)


def lang(code):
    def h(d):
        cb = next(c for c in d.findChildren(QComboBox) if c.findData(code) >= 0)
        cb.setCurrentIndex(cb.findData(code))
        for _ in range(10):
            QApplication.processEvents()
        d.reject()
    ex.dialog_handler = h
    ex.w.open_settings()
    ex.settle(12)


for rep in range(2):
    g.new("session_empty_wf", ["calibration", "registration", "stacking", "astrometry", "gradient"], clear=False)
    g.menu(g.blk("empilement"), "Vers Post")
    ex.w.set_mode(2)
    ex.settle(8)
    print(rep, "post shown: mode", ex.w.mode)
    pp.undo()
    print(rep, "Ctrl+Z in Post: mode", ex.w.mode, "status", ex.w.status.text())
    pp.redo()
    print(rep, "Ctrl+Y in Post: mode", ex.w.mode, "status", ex.w.status.text(), "sel", ex.s.sel_step, ex.s.current)
    ex.shot(f"post_after_redo_{rep}")
    ex.w.set_mode(2)
    ex.settle(6)
    lang("en")
    print(rep, "after en: mode", ex.w.mode)
    lang("fr")
    print(rep, "after fr: mode", ex.w.mode)
ex.close()
