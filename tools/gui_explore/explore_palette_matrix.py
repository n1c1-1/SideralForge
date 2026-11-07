"""palette_matrix: every palette card × every way to place it × a project state × the chosen block, then Ctrl+Z and
Ctrl+Y. One project state per process:  python explore_palette_matrix.py <state> [cards...]
Raw findings go to the scratch folder (pm_raw/<state>.jsonl); the distinct ones are reproduced and noted later."""
from __future__ import annotations

import json
import sys
import time

from explore_palette_lib import Matrix, frenchness

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from astro_gpu.client import i18n
from astro_gpu.client.model import node_key

STATE = sys.argv[1]
ONLY = sys.argv[2:]
NEW = {"new_blank": None, "new_full": "full", "new_nights": "nights", "new_sho": "sho",
       "new_stretched": "image_stretched", "new_image": "image", "new_raw": "raw"}
SOURCES = ("session", "image", "fusion")
CHAIN_KEYS = ("calibration", "alignement", "selection", "empilement", "astrometrie", "fond", "debruitage", "couleur",
              "etoiles")

m = Matrix(STATE)
ex = m.ex
_n = [0]


def fresh():
    _n[0] += 1
    if STATE == "stretched":                       # an Image workflow of a stretched picture, empty (not a harness kind)
        from astro_gpu.core import project2
        from astro_gpu.core import workflow as wfl
        from astro_gpu.core.project import Target
        root = ex.ws / f"stretched_{_n[0]}"
        p = project2.create(root, root.name, Target(name="M 31"))
        p.stack_mode = "per_session"
        project2.save(p)
        w = p.add_workflow("Image", wfl.TEMPLATES["Classic stack"])
        w.source = {"path": str(ex.image_file("pic.png", stretched=True)), "stretched": True}
        p.set_blocks(w.id, [])
        project2.save(p)
        ex.open(root)
    elif STATE in NEW:
        root, got = m.new_project(f"N{_n[0]}", NEW[STATE])
        print("new project", got, [(q.name, ex.s.graph(q)) for q in ex.s.paths], flush=True)
    else:
        root = ex.project(STATE)
        ex.open(root)
    ex.settle(6)
    m.hook_page()
    s = ex.s
    return (s.current, s.sel_step)


INIT = fresh()
ex.popups.clear()
ex.modals.clear()


# ------------------------------------------------------------ selections
def desc(it):
    return (it.node.key, it.node.sid, tuple(sorted(q.pid for q in it.node.paths)))


def find_item(d):
    for it in ex.blocks():
        if desc(it) == d:
            return it
    return None


def selections():
    out = [("ouvert", None)]
    its = sorted(ex.blocks(), key=lambda it: (it.sceneBoundingRect().y(), it.sceneBoundingRect().x()))
    got = {}
    for it in its:
        k = it.node.key
        if k in ("session", "sources"):
            got.setdefault("session", desc(it))
            if "session" in got and desc(it) != got["session"]:
                got.setdefault("session2", desc(it))
        elif k == "image":
            got.setdefault("image", desc(it))
        elif k == "fusion":
            got.setdefault("fusion", desc(it))
        elif k.split("~")[0] in ("calibration", "alignement", "selection", "empilement"):
            got.setdefault("frame_sess" if it.node.sid is not None else "frame", desc(it))
        elif k.split("~")[0] in ("astrometrie", "fond", "debruitage", "couleur", "etoiles"):
            got.setdefault("imgblock", desc(it))
        elif k == "composition":
            got.setdefault("composition", desc(it))
        elif k == "final":
            got.setdefault("post", desc(it))
    out += list(got.items())
    out.append(("vide", "empty"))
    return out


def apply_sel(sel):
    name, d = sel
    pg = ex.page
    s = ex.s
    if s.paths:
        cur = INIT[0] if INIT[0] is not None and 0 <= INIT[0] < len(s.paths) else 0
        pg.select(index=cur, step=INIT[1])
    else:
        s.select(step=INIT[1])
    ex.settle(3)
    if d is None:
        return True
    if d == "empty":
        vp = ex.view.viewport()
        p = m.empty_point()
        if p is not None:
            ex.click(vp, p)
        return True
    it = find_item(d)
    if it is None:
        return False
    ex.click_block(it)
    ex.settle(3)
    return True


# ------------------------------------------------------------ the checks of one case
def snapshot_side():
    return {"toasts": len(m.toasts), "exc": len(m.exc), "pop": len(ex.popups), "mod": len(ex.modals),
            "file": len(ex.file_asks), "missing": set(i18n.missing()), "dlg": len(ex.dialogs())}


def check_side(side, where):
    if len(m.exc) > side["exc"]:
        for e in m.exc[side["exc"]:]:
            m.find("exception:" + e.splitlines()[0][:80], e, where=where)
    if len(ex.popups) > side["pop"]:
        m.find("popup", repr(ex.popups[side["pop"]:]), where=where)
    if len(ex.modals) > side["mod"]:
        m.find("modal:" + ",".join(x[0] for x in ex.modals[side["mod"]:]), repr(ex.modals[side["mod"]:]), where=where)
    if len(ex.file_asks) > side["file"]:
        m.find("filedialog", repr(ex.file_asks[side["file"]:]), where=where)
    dl = ex.dialogs()
    if dl:
        m.find("dialog-open:" + ",".join(d.__class__.__name__ for d in dl), repr([(d.__class__.__name__,
               d.windowTitle()) for d in dl]), where=where)
        ex.close_dialogs()
    new_missing = set(i18n.missing()) - side["missing"]
    for t in sorted(new_missing):
        m.find("untranslated:" + t[:60], t, where=where)
    for t in m.toasts[side["toasts"]:]:
        m.count("toast")
        TOASTS.setdefault(t, m.case)
        for prob in frenchness(t):
            m.find("toast-words:" + t[:70], f"{t}\n{prob}", where=where)


TOASTS: dict[str, str] = {}


def new_nodes(before, after):
    """(pid, index, node) of the nodes of ``after`` not in ``before`` (per workflow), and the new workflows."""
    b = {x[0]: json.loads(x[4]) for x in before["paths"]}
    out, new_paths = [], []
    for x in after["paths"]:
        pid, g = x[0], json.loads(x[4])
        if pid not in b:
            new_paths.append(pid)
            continue
        old = b[pid]
        if len(g) == len(old) + 1:
            for i in range(len(g)):
                if g[:i] + g[i + 1:] == old:
                    out.append((pid, i, g[i]))
                    break
        elif g != old:
            out.append((pid, None, None))
    return out, new_paths


def check_drawn(pid, nd, where):
    """The new block is drawn in its workflow, in order left to right, the chosen block, not overlapping."""
    s = ex.s
    q = s.view(pid)
    if q is None:
        m.find("placed-workflow-gone", f"pid {pid}", where=where)
        return
    key = node_key(nd)
    its = [it for it in ex.blocks() if it.node.key == key and pid in [p.pid for p in it.node.paths]]
    sep = key == "calibration" and q.source is None and s.separate_sessions()
    if not its:
        m.find("placed-not-drawn:" + key, f"{key} in {q.name}: no block drawn", where=where)
    # order: the display keys drawn left to right
    xs = []
    for k in s.display_keys(q):
        it = next((it for it in ex.blocks() if it.node.key == k and pid in [p.pid for p in it.node.paths]
                   and it.node.sid is None), None)
        if it is not None:
            xs.append((k, it.sceneBoundingRect().x()))
    if any(xs[i][1] >= xs[i + 1][1] for i in range(len(xs) - 1)):
        m.find("order-not-left-to-right", repr(xs), where=where)
    sel = s.sel_step
    if not sep and sel != key:
        m.find(f"placed-not-chosen:{key}", f"chosen step {sel!r} after placing {key}", where=where, shot=False)
    if s.path is None or s.path.pid != pid:
        m.find("placed-other-current-workflow", f"current {s.path.name if s.path else None} vs placed in {q.name}",
               where=where)
    h = m.heading()
    if q.name not in h:
        m.find("heading-after-place", f"heading {h!r}, block placed in {q.name!r}", where=where, shot=False)


def run_case(sel, card, method, action, expect=None):
    """``expect``: what should happen: ("insert", pid, idx) | ("branch", pid) | ("nothing",) | ("source",) |
    ("dbl", pid|None) | ("comp",) | None (unknown: recorded)."""
    m.case = f"{STATE}|{sel[0]}|{card}|{method}"
    m.watchdog()
    m.count(f"case:{method.split('#')[0]}")
    if not apply_sel(sel):
        m.count("sel-missing")
        return
    before = m.state()
    pp = ex.page.palette_path()
    side = snapshot_side()
    items_before = len(ex.blocks())
    over_before = set(m.overlaps())
    res = ex.guard(m.case, action)
    ex.settle(6)
    check_side(side, "action")
    after = m.state()
    changed = after != before
    where = "action"
    kind = expect[0] if expect else "?"
    if not changed:
        if kind in ("insert", "branch", "dbl", "source", "comp"):
            last = m.toasts[side["toasts"]:]
            m.find(f"nothing-placed:{card}:{method.split('#')[0]}:{kind}",
                   f"expected {expect}, nothing changed; toasts {last}; drop {res}", where=where)
        return
    if kind == "nothing":
        m.find(f"placed-unexpected:{card}:{method.split('#')[0]}", m.diff(before, after), where=where)
    nn, newp = new_nodes(before, after)
    over = [o for o in m.overlaps() if o not in over_before]
    if over:
        m.find("overlap-after-place:" + card, "\n".join(over), where=where)
    if kind == "insert":
        pid, idx = expect[1], expect[2]
        ok = any(x[0] == pid and x[1] == idx and x[2]["block"] == card for x in nn)
        if not ok:
            m.find(f"insert-elsewhere:{card}", f"expected pid {pid} idx {idx}; got {nn} new {newp}", where=where)
    elif kind == "branch":
        if not newp:
            m.find(f"branch-not-made:{card}", f"{nn}", where=where)
    elif kind == "dbl" and card not in SOURCES and card != "composition":
        want = expect[1]
        if want is not None and not any(x[0] == want for x in nn):
            m.find(f"dbl-other-workflow:{card}", f"palette heading pid {want}; changed {nn} new {newp}", where=where)
    for pid, idx, nd in nn:
        if nd is not None:
            ex.guard(m.case + " drawn", check_drawn, pid, nd, where)
    if card in SOURCES:
        SRC_BEHAV.setdefault((sel[0], card, method.split("#")[0]), describe(before, after))
    # undo / redo
    side = snapshot_side()
    ex.undo()
    ex.settle(4)
    u = m.state()
    check_side(side, "undo")
    if u != before:
        m.find(f"undo-not-exact:{card}:{method.split('#')[0]}", m.diff(before, u), where="undo")
    side = snapshot_side()
    ex.redo()
    ex.settle(4)
    r = m.state()
    check_side(side, "redo")
    if r != after:
        m.find(f"redo-not-exact:{card}:{method.split('#')[0]}", m.diff(after, r), where="redo")
    ex.undo()
    ex.settle(4)
    if m.state() != before:
        m.find("undo2-not-exact", m.diff(before, m.state()), where="undo2", shot=False)
        global INIT
        INIT = fresh()
    if len(ex.blocks()) != items_before:
        m.find(f"items-after-undo:{card}", f"{items_before} -> {len(ex.blocks())}", where="undo", shot=False)


SRC_BEHAV: dict = {}


def describe(before, after):
    d = []
    if len(after["sessions"]) != len(before["sessions"]):
        d.append(f"sessions {len(before['sessions'])}->{len(after['sessions'])}")
    bp = {x[0]: x for x in before["paths"]}
    for x in after["paths"]:
        if x[0] not in bp:
            d.append(f"new workflow {x[1]!r} src={json.loads(x[2]) and 'image/fusion'} sessions={x[3]}")
        elif x != bp[x[0]]:
            d.append(f"changed {x[1]!r} sessions {bp[x[0]][3]}->{x[3]}")
    if after["post"] != before["post"]:
        d.append("post layers changed")
    return "; ".join(d)


# ------------------------------------------------------------ the hover: what lights up
def hover(card):
    """Places shown while the card is held over the graph: ghosts, the zone, an empty point, blocks, Post."""
    vp = ex.view.viewport()
    ex.drag_palette_hover(card, QPoint(vp.width() // 2, 4))
    gh = []
    for i, g in enumerate(ex.ghosts()):
        gh.append({"i": i, "pt": ex.ghost_center(g), "pid": g.pid, "idx": getattr(g, "idx", None),
                   "branch": getattr(g, "branch_key", None), "text": g.text, "tip": g.toolTip(),
                   "comp": getattr(g, "node", None) is not None, "rect": g.sceneBoundingRect()})
    z = ex.new_flow_zone()
    zone = ex.view.mapFromScene(z.sceneBoundingRect().center()) if z is not None and z.scene() else None
    zone_hot = getattr(z, "hot", None)
    empty = m.empty_point()
    blocks = {}
    for it in sorted(ex.blocks(), key=lambda it: it.sceneBoundingRect().x()):
        k = it.node.key
        cat = "src" if k in ("session", "sources", "image", "fusion") else "post" if k == "final" else \
            "comp" if k == "composition" else "chain"
        if cat not in blocks:
            blocks[cat] = ex.point_of(it)
    over = m.overlaps([(f"ghost:{g['text']}[{g['pid']}]", g["rect"]) for g in gh])
    over = [o for o in over if "ghost" in o]
    ex.drag_leave()
    return gh, zone, zone_hot, empty, blocks, over


# ------------------------------------------------------------ the matrix
def main():
    t0 = time.time()
    sels = selections()
    print("selections:", [x[0] for x in sels], flush=True)
    cards = ONLY or list(ex.page.cards)
    for sel in sels:
        if not apply_sel(sel):
            continue
        h = m.heading()
        rs = m.reasons()
        node = ex.page.current_node()
        HEADS.append((sel[0], h, node.key if node else None, {k: v for k, v in rs.items() if v}))
        for b, r in rs.items():
            if r:
                for prob in frenchness(r):
                    m.find(f"reason-words:{b}:{r[:60]}", f"{r}\n{prob}", where=sel[0], shot=False)
                txt, geo = m.hint_of(b)
                if not txt:
                    m.find(f"no-hint:{b}", f"greyed {b} ({r}) shows no message when hovered", where=sel[0])
                elif txt.lower() != r.lower():
                    m.find(f"hint-differs:{b}", f"{txt!r} vs {r!r}", where=sel[0], shot=False)
                elif geo is not None and not ex.page.rect().contains(geo):
                    m.find(f"hint-outside:{b}", f"{geo} not inside {ex.page.rect()}", where=sel[0])
        for card in cards:
            if not apply_sel(sel):
                break
            c = ex.card(card)
            if c is None:
                continue
            reason = c.reason
            pp = ex.page.palette_path()
            if reason:
                run_case(sel, card, "dbl", lambda: ex.double_click_card(card), ("nothing",))
                if not apply_sel(sel):
                    break
                n = len(ex.drops)
                ex.drag_palette(card, QPoint(ex.view.viewport().width() // 2, ex.view.viewport().height() // 2))
                if len(ex.drops) > n:
                    m.find(f"greyed-drags:{card}", f"{reason}", where=sel[0])
                continue
            # available card
            exp_dbl = ("source",) if card in SOURCES else ("comp",) if card == "composition" else \
                ("dbl", pp.pid if pp is not None else None)
            run_case(sel, card, "dbl", lambda: ex.double_click_card(card), exp_dbl)
            if not apply_sel(sel):
                break
            gh, zone, zone_hot, empty, blocks, over = hover(card)
            if over:
                m.find(f"ghost-overlap:{card}", "\n".join(over), where=sel[0])
            for g in gh:
                for prob in frenchness(g["text"] + " " + (g["tip"] or "")):
                    m.find(f"ghost-words:{card}", f"{g['text']} | {g['tip']}\n{prob}", where=sel[0], shot=False)
            if card in SOURCES and zone_hot is False:
                m.count("zone-not-lit")
            if card not in SOURCES and not gh:
                m.find(f"available-but-no-place:{card}", f"card available (heading {m.heading()!r}) but dragging it "
                       f"shows no place", where=sel[0])
            if sel[0] == "ouvert" and gh:
                if not apply_sel(sel):
                    break
                ex.drag_palette_hover(card, gh[0]["pt"])
                ex.w.grab().save(str(m.shots / f"hover_{STATE}_{card}.png"))
                ex.drag_leave()
            for g in gh[:10]:
                if g["comp"]:
                    exp = ("comp",)
                elif g["branch"]:
                    exp = ("branch", g["pid"])
                else:
                    exp = ("insert", g["pid"], g["idx"])
                run_case(sel, card, f"ghost#{g['i']}", lambda pt=g["pt"]: ex.drag_palette(card, pt), exp)
            other = ("source",) if card in SOURCES else ("nothing",)
            if zone is not None:
                run_case(sel, card, "zone", lambda: ex.drag_palette(card, zone), other)
            if empty is not None:
                run_case(sel, card, "vide", lambda: ex.drag_palette(card, empty), None if card in SOURCES else
                         ("nothing",))
            for cat, pt in blocks.items():
                run_case(sel, card, f"sur-{cat}", lambda pt=pt: ex.drag_palette(card, pt), None if card in SOURCES
                         else ("nothing",))
            run_case(sel, card, "palette", lambda: m.drop_on_palette(card), ("nothing",))
        print(f"sel {sel[0]} done {time.time() - t0:.0f}s", flush=True)
    out = {"state": STATE, "counts": m.counts, "heads": HEADS, "toasts": TOASTS,
           "src": {"|".join(k): v for k, v in SRC_BEHAV.items()}, "time": time.time() - t0}
    (m.raw.parent / f"{STATE}_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                       encoding="utf-8")
    print("DONE", STATE, m.counts, f"{time.time() - t0:.0f}s", flush=True)


HEADS: list = []
try:
    main()
finally:
    import faulthandler
    faulthandler.cancel_dump_traceback_later()
    try:
        ex.close()
    except Exception:  # noqa: BLE001
        pass
    QApplication.processEvents()
