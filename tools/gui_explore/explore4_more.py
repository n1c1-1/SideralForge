"""Round 4 (agent I): the Fusion's Inspect panel says why it is refused (item 8); unreadable lights are not counted as
used (item 11); a session's Calibration in Inspect and the last block after a deletion say what the graph says
(items 4, 7, by their panels). Prints what each shows.

    python explore4_more.py fr
"""
from __future__ import annotations

import faulthandler
import sys

faulthandler.dump_traceback_later(400, exit=True)
from harness import TMP, Explorer, stray_qt_errors  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
stray_qt_errors()
ex = Explorer("inspect4", lang=LANG, size=(1366, 768))

# 8: a Fusion refused (the OIII session's filter changed to Ha): its Inspect panel says why
ex.open(ex.project("mono", f"F8 {LANG}"))
for q in list(ex.s.paths):                       # each channel needs its Astrometry before a Fusion (decision 2)
    ex.click_block(ex.block("empilement", pid=q.pid))
    ex.double_click_card("astrometry")
    ex.settle(4)
ex.click_block(ex.block("session", sid=1) or ex.blocks()[0])
ex.double_click_card("fusion")
ex.settle(6)
ex.s.set_session_filter(2, "Ha")
ex.settle(6)
fu = ex.block("fusion")
print("8. fusion broken:", fu.node.broken if fu else None, flush=True)
ex.double_click_block(fu)
ex.settle(8)
print("8. Inspect panel:", [w.text() for w in ex.w.tune.panel.findChildren(QLabel) if w.isVisible() and
                            ("refus" in w.text().lower())], flush=True)
ex.shot(f"i8_fusion_refused_{LANG}")
ex.w.set_mode(0)

# 11: five .fits that are not FITS files
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
root = ex.ws / f"U11 {LANG}"
bad = TMP / "data" / "bad11"
bad.mkdir(parents=True, exist_ok=True)
for i in range(5):
    (bad / f"light_{i:03d}.fits").write_bytes(b"not a fits file" * 20)
p = project2.create(root, root.name, Target(name="M 31"))
p.sessions.append(project2.Session(1, "Session 1", lights=[str(bad)]))
w = p.add_workflow("Workflow", {})
p.set_blocks(w.id, [{"block": "stacking"}])
project2.save(p)
ex.open(root)
ex.settle(8)
ex.click_block(ex.block("sources") or ex.block("session", sid=1) or ex.blocks()[0])
ex.settle(6)
print("11. session panel:", [t for t in ex.texts(ex.page.panel) if "/ 5" in t], flush=True)
ex.shot(f"i11_unreadable_{LANG}")
ex.close()
