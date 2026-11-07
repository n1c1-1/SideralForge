"""post_wires round 2: every middle block sent to Post (a busy graph): overlaps, names; Ctrl+Z x5 / Ctrl+Y x5 step by
step; then the Post block's panel and the run window's list (texts)."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start

lang = sys.argv[1] if len(sys.argv) > 1 else "fr"
size = tuple(int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1920x1080").split("x"))
start(600)
ex = Explorer("post_wires", lang=lang, size=size)
pp = P(ex)
session_project(ex)
build_chain(ex, CHAINS["stars"])
send = "Vers Post" if lang == "fr" else "To Post"
snap = []


def state():
    return ([q.name for q in ex.s.paths], pp.final_summary(), len(pp.wires()), len(pp.blocks_of("post")))


snap.append(state())
for k in ("empilement", "astrometrie", "fond", "debruitage", "couleur"):
    ex.right_click_block(pp.block(k, pid=ex.s.paths[0].pid), send)
    ex.settle(6)
    snap.append(state())
    print(k, "->", snap[-1], repr(pp.toast()))
outs = pp.blocks_of("post")
print("output blocks", [(b.title, b.summary) for b in outs])
rects = [(b.title + "/" + b.summary, b.sceneBoundingRect()) for b in ex.blocks()]
for i in range(len(rects)):
    for j in range(i + 1, len(rects)):
        a, b = rects[i][1], rects[j][1]
        if a.intersects(b) and a.intersected(b).width() > 3 and a.intersected(b).height() > 3:
            print("OVERLAP", rects[i][0], rects[j][0])
ex.click_block(pp.final())
print("post panel", pp.panel_texts()[2][:600])
ex.shot(f"busy_{lang}_{size[0]}")
# undo step by step
for i in range(5):
    pp.undo()
    s = state()
    exp = snap[-2 - i]
    print("undo", i + 1, s, "OK" if s == exp else f"EXPECTED {exp}")
for i in range(5):
    pp.redo()
    s = state()
    exp = snap[i + 1]
    print("redo", i + 1, s, "OK" if s == exp else f"EXPECTED {exp}")
ex.close()
