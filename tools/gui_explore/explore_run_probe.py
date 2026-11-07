"""Probe (run_status): what the top bar, the status line and the plan window show for a few graphs. No notes."""
import faulthandler
import sys

from harness import Explorer

faulthandler.dump_traceback_later(300, exit=True)
ex = Explorer("run_status")


def state(tag):
    w = ex.w
    s = ex.s
    print(f"--- {tag}")
    print("  paths:", [(q.pid, q.name, q.source, s.graph(q) if hasattr(s, 'graph') else None) for q in s.paths])
    print("  blocks:", [(it.node.key, it.node.sid, getattr(it, 'title', '')) for it in ex.blocks()])
    print("  run btn:", repr(w.bt_run.text()), w.bt_run.isEnabled(), repr(w.bt_run.toolTip()))
    print("  status:", repr(w.status.text()))
    print("  combo:", [(w.device_cb.itemText(i), w.device_cb.itemData(i)) for i in range(w.device_cb.count())],
          "cur", w.device_cb.currentIndex(), "tip", repr(w.device_cb.toolTip()))
    print("  a_run", w.a_run.isEnabled(), "a_here", w.a_here.isEnabled(), "sel", s.sel_step)


def plan(tag, here=False):
    d = ex.w.run.run_to_here() if here else ex.w.run.launch()
    ex.settle(6)
    if d is None:
        print(f"  plan[{tag}] none")
        return None
    print(f"  plan[{tag}] title", repr(d.windowTitle()), "jobs", [(j.key, j.stage, j.block, j.wf) for j in d.plan.jobs])
    for r in d.rows:
        print("    row", repr(r.name.text()), "|", repr(r.who.text()), "|", repr(r.dur.text()), "| tip", repr(r.toolTip()))
    print("   summary", repr(d.summary.text()), "detail", repr(d.detail.text()), "pause", repr(d.pause_note.text()),
          "run vis/en", d.b_run.isVisible(), d.b_run.isEnabled())
    print("   texts", [t for t in ex.texts(d)][:40])
    return d


kinds = sys.argv[1:] or ["session"]
for kind in kinds:
    root = ex.project(kind)
    ex.open(root)
    ex.settle(20)
    state(kind + " opened")
    if kind == "session":
        ex.double_click_card("stacking")
        ex.settle(20)
        state("stacking placed")
        d = plan("all")
        if d:
            d.close()
        it = ex.block("empilement")
        if it:
            ex.click_block(it)
            d = plan("here", True)
            if d:
                d.close()
ex.close()
print("popups", ex.popups, "modals", ex.modals)
