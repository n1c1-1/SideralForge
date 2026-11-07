"""palette_matrix, layout pass: in each project state (and after each block placed by a double-click), does anything
drawn overlap (blocks, the bottom new-flow zone, "+ Session")?  python explore_palette_layout.py [state...]"""
from __future__ import annotations

import sys

from explore_palette_lib import Matrix

STATES = sys.argv[1:] or ["blank", "session", "session_empty_wf", "image", "session+image", "two_sessions", "two_flows",
                          "mono", "legacy", "jpg_mistake", "new_full", "new_nights", "new_sho", "new_stretched",
                          "new_image", "new_raw"]
NEW = {"new_blank": None, "new_full": "full", "new_nights": "nights", "new_sho": "sho",
       "new_stretched": "image_stretched", "new_image": "image", "new_raw": "raw"}
m = Matrix("layout")
ex = m.ex
k = [0]


def fresh(st):
    k[0] += 1
    if st in NEW:
        m.new_project(f"L{k[0]}", NEW[st])
    else:
        ex.open(ex.project(st))
    ex.settle(8)
    m.hook_page()


for st in STATES:
    m.watchdog(400)
    fresh(st)
    m.case = f"{st}|ouvert"
    o = m.overlaps()
    print(st, "open:", o, flush=True)
    if o:
        m.find(f"overlap-open:{st}", "\n".join(o), state=st)
    for card in list(ex.page.cards):
        c = ex.card(card)
        if c is None or c.reason:
            continue
        m.case = f"{st}|dbl {card}"
        before = set(m.overlaps())
        ex.guard(m.case, ex.double_click_card, card)
        ex.settle(6)
        o = [x for x in m.overlaps() if x not in before]
        if o:
            print(st, card, "after:", o, flush=True)
            m.find(f"overlap-after:{st}:{card}", "\n".join(o), state=st)
        ex.close_dialogs()
        fresh(st)
ex.close()
