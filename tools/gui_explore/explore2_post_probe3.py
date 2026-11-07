"""post_wires round 2: does the start hint stay after an Image is placed (waiting like a user); Ctrl+Z after a project is
reopened; the Post block double-click while Open Post is greyed."""
import time

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt

start(300)
ex = Explorer("post_wires")
pp = P(ex)


def wait(sec):
    t = time.time()
    while time.time() - t < sec:
        ex.settle(5)


ex.new_project("Indice")
ex.w.activateWindow()
print("hint at start:", repr(pp.toast()))
wait(1)
ex.double_click_card("image")
ex.settle(8)
print("just after image:", repr(pp.toast()))
wait(8)
print("8 s after image:", repr(pp.toast()))
ex.shot("probe3_hint_after_8s")
# reopen and Ctrl+Z
root = session_project(ex)
build_chain(ex, CHAINS["color"])
pp.click_wire(pp.wires()[0])
pp.del_key()
pp.key(Qt.Key_S, ex.w, Qt.ControlModifier)
ex.settle(6)
ex.w.open_project(root)
ex.settle(12)
pp.activate()
print("after reopen: can_undo", ex.s.can_undo(), "post", pp.final_summary(), "wires", len(pp.wires()))
pp.undo()
print("Ctrl+Z after reopen: post", pp.final_summary(), "wires", len(pp.wires()), "toast", repr(pp.toast()),
      [(q.name, [n['block'] for n in ex.s.p.workflow(q.pid).blocks]) for q in ex.s.paths])
pp.undo()
print("2nd Ctrl+Z:", pp.final_summary(), [(q.name, [n['block'] for n in ex.s.p.workflow(q.pid).blocks]) for q in ex.s.paths])
ex.close()
