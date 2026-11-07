"""block_actions round 2: every action on every drawn block of one scenario, each on a fresh copy of the built project;
what changed, what was said, then ONE Ctrl+Z (back to exactly before?) and ONE Ctrl+Y (back to exactly after?).

    python explore2_blocks_matrix.py <scenario> [lang] [max_blocks]

Writes <out>/m2_<scenario>_<lang>.jsonl (one line per action). Suspects are printed with "SUSPECT"."""
from __future__ import annotations

import faulthandler
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(1500, exit=True)

from explore2_blocks_common import CHAIN, EXC, TOASTS, G, Explorer, panel_info, state  # noqa: E402
from explore_blocks_lib import Lab, bid, find  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLineEdit, QPushButton  # noqa: E402

SC = sys.argv[1]
LANG = sys.argv[2] if len(sys.argv) > 2 else "fr"
ex = Explorer("block_actions", lang=LANG)
g = G(ex)
OUT = ex.out / f"m2_{SC}_{LANG}.jsonl"
OUT.write_text("", encoding="utf-8")
SKIP = {"Inspection", "Inspect", "Enregistrer comme modèle…", "Save as template…", "Ouvrir dans Post", "Open in Post"}


def snap():
    st = state(ex)
    return {"w": [(w["name"], w["blocks"], w["sessions"], json.dumps(w["source"], default=str), w["bp"])
                  for w in st["workflows"]],
            "s": [(x["id"], x["name"], x["filter"], x["lights"]) for x in st["sessions"]],
            "post": json.dumps(getattr(ex.s.p, "post", None) or {}, sort_keys=True, default=str)}


def build():
    if SC == "chain":
        g.new("session", CHAIN, clear=False)
    elif SC == "short":
        g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient", "color"], clear=False)
    elif SC == "sessions2":
        g.new("two_sessions", ["calibration", "registration", "stacking", "astrometry", "gradient"], clear=False)
    elif SC == "mono":
        g.mono_fusion()
    elif SC == "image":
        g.new("image", clear=False)
        ex.click_block(g.blk("image"))
        g.place("astrometry", "gradient", "color")
    elif SC == "branch":
        g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"], clear=False)
        g.drag_branch("color", "astrometrie", ex.s.paths[0].pid)
    elif SC == "repeat":
        g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"], clear=False)
        g.drop_idx("gradient", ex.s.paths[0].pid, 3)
        g.place("denoise", "denoise")
    elif SC == "sent":
        g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient", "color"], clear=False)
        g.menu(g.blk("fond"), pick="Vers Post" if LANG == "fr" else "To Post")
    elif SC == "two_flows":
        g.new("two_flows", clear=False)
        for q in list(ex.s.paths):
            it = next((b for b in ex.blocks() if b.node.key in ("session", "sources") and q.pid in [p.pid for p in b.node.paths]), None)
            if it is not None:
                ex.click_block(it)
            g.place("calibration", "registration", "stacking")
    g.act()


def accept_name(d):
    for e in d.findChildren(QLineEdit):
        if e.isVisible():
            e.setText("Nom essai")
            break
    d.accept()


def gestures(b, nth) -> list[tuple]:
    """(label, how) of everything a user can do on this block: its menu entries, panel buttons, cross, keys."""
    out = []
    m = g.menu(find(ex, b, nth))

    def walk(items, prefix=""):
        for x in items:
            t = x.get("text") or ""
            if t == "---" or t.split("\t")[0] in SKIP:
                continue
            if x.get("sub"):
                walk(x["sub"], prefix + t + " > ")
            elif x.get("enabled"):
                out.append(("menu:" + prefix + t.split("\t")[0], t.split("\t")[0]))
    walk(m)
    out.append(("key:Del", "del"))
    out.append(("key:D", "d"))
    out.append(("cross", "cross"))
    out.append(("drag to palette", "palette"))
    ex.click_block(find(ex, b, nth))
    for b in ex.page.panel.findChildren(QPushButton):
        if b.isVisible() and b.isEnabled() and b.text() and b.text() not in ("×",) and "▾" not in b.text() \
                and not b.text().startswith("▸") and "…" not in b.text():
            out.append(("panel:" + b.text(), "panel:" + b.text()))
    seen, uniq = set(), []
    for x in out:
        if x[0] not in seen:
            seen.add(x[0])
            uniq.append(x)
    return uniq, m


def do(it, how):
    ex.dialog_handler = accept_name
    if how == "del":
        ex.click_block(it)
        g.act()
        ex.key(Qt.Key_Delete)
    elif how == "d":
        ex.click_block(it)
        g.act()
        ex.key(Qt.Key_D)
    elif how == "cross":
        g.cross(it)
    elif how == "palette":
        ex.drag_block_to_palette(it)
    elif how.startswith("panel:"):
        ex.click_block(it)
        t = how[6:]
        got = [b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible() and b.isEnabled() and b.text() == t]
        if got:
            ex.click(got[0])
    else:
        g.menu(it, pick=how)
    ex.settle(6)
    ex.dialog_handler = None
    g.act()


def main():
    build()
    lab = Lab(ex, SC + "_" + LANG)
    lab.save_master()
    lab.fresh()
    blocks = [(bid(it), k) for k, it in enumerate(sorted(ex.blocks(), key=lambda i: (i.pos().x(), i.pos().y())))]
    seen_bid: dict = {}
    targets = []
    for b, _k in blocks:
        n = seen_bid.get(b, 0)
        seen_bid[b] = n + 1
        targets.append((b, n))
    maxb = int(sys.argv[3]) if len(sys.argv) > 3 else 99
    for b, nth in targets[:maxb]:
        lab.fresh()
        it = find(ex, b, nth)
        if it is None:
            continue
        title = it.title
        gs, menu = gestures(b, nth)
        print(f"== {title} {b} : {len(gs)} gestures", flush=True)
        for label, how in gs:
            lab.fresh()
            ex.s._undo.clear()
            ex.s._redo.clear()
            it = find(ex, b, nth)
            if it is None:
                continue
            s0 = snap()
            mk = lab.mark()
            do(it, how)
            s1 = snap()
            got = lab.since(mk)
            changed = s1 != s0
            nund = len(ex.s._undo)
            g.undo()
            s2 = snap()
            g.redo()
            s3 = snap()
            rec = {"block": title, "bid": list(map(str, b)), "nth": nth, "gesture": label, "changed": changed,
                   "undo_entries": nund, "undo_ok": (s2 == s0) if changed else None,
                   "redo_ok": (s3 == s1) if changed else None, "toasts": got["toasts"], "popups": got["popups"],
                   "modals": got["modals"], "exc": [e.strip().splitlines()[-1] for e in got["exc"]],
                   "files": got["files"], "selected_after": ex.s.sel_step,
                   "titles_after": [x.title for x in sorted(ex.blocks(), key=lambda i: (i.pos().x(), i.pos().y()))]}
            if changed and s2 != s0:
                rec["diff_undo"] = {k: [s0[k], s2[k]] for k in s0 if s0[k] != s2[k]}
            if changed and s3 != s1:
                rec["diff_redo"] = {k: [s1[k], s3[k]] for k in s1 if s1[k] != s3[k]}
            sus = []
            if rec["exc"]:
                sus.append("EXC")
            if changed and (not rec["undo_ok"] or not rec["redo_ok"]):
                sus.append("UNDO")
            if changed and nund != 1:
                sus.append(f"UNDO_ENTRIES={nund}")
            if not changed and not got["toasts"] and not got["popups"] and how not in ("palette",):
                sus.append("SILENT")
            if got["popups"] or got["modals"]:
                sus.append("POPUP")
            rec["suspect"] = sus
            with OUT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            print(("SUSPECT " if sus else "  ") + f"{title} | {label} | changed={changed} undo={rec['undo_ok']} "
                  f"redo={rec['redo_ok']} n={nund} | {got['toasts'][-2:]} {got['popups'][-1:]} {rec['exc']} {sus}",
                  flush=True)
    ex.close()


main()
