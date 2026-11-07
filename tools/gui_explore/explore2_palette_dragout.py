"""palette_matrix round 2: placed blocks dragged back onto the palette (removed, with a message and Ctrl+Z; refused with
the reason when others need it): sources (Session, Image, Fusion), Recombine, a chain block others need, a copy."""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("dragout")
ex = m.ex


def item(key, nth=0, pid_name=None):
    its = [it for it in ex.blocks() if it.node.key.split("~")[0] == key and
           (pid_name is None or pid_name in [q.name for q in it.node.paths])]
    its.sort(key=lambda it: (it.sceneBoundingRect().y(), it.sceneBoundingRect().x()))
    return its[nth] if len(its) > nth else None


CASES = [("session+image", "image", 0, None), ("two_flows", "session", 1, None), ("mono", "fusion", 0, "fusion"),
         ("legacy", "composition", 0, "composition"), ("legacy", "empilement", 0, None), ("legacy", "astrometrie", 0, None),
         ("legacy", "calibration", 0, None), ("two_sessions", "session", 1, None)]
for kind, key, nth, pre in CASES:
    for attempt in (1, 2):
        ex.open(ex.project(kind))
        ex.settle(8)
        m.hook_page()
        if pre:
            ex.double_click_card(pre)
            ex.settle(4)
        it = item(key, nth)
        if it is None:
            print(kind, key, "no block")
            break
        title = it.title
        s0 = m.state()
        n0 = len(ex.blocks())
        t0 = len(m.toasts)
        ex.drag_block_to_palette(it)
        ex.settle(6)
        s1 = m.state()
        removed = s1 != s0
        print(kind, key, attempt, f"{title!r} dragged to palette: changed {removed}, blocks {n0}->{len(ex.blocks())}, "
              f"toasts {m.toasts[t0:]}, popups {ex.popups[-1:] if ex.popups else []}, drops {ex.drops[-1:]}", flush=True)
        if attempt == 2:
            ex.shot(f"dragout_{kind.replace('+', '_')}_{key}")
        if removed:
            ex.undo()
            print("   undo exact", m.state() == s0)
            ex.redo()
            print("   redo exact", m.state() == s1)
        # the same block's menu: Delete enabled? its tip
        it = item(key, nth) if not removed else None
        if it is not None:
            menu = ex.right_click_block(it)
            print("   menu", [(a.get("text"), a.get("enabled"), a.get("tip")) for a in menu if a.get("text") != "---"][:4])
        print("   exceptions", [x.splitlines()[0] for x in m.exc])
ex.close()
