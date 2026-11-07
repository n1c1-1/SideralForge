"""Words, probe: the texts drawn on image canvases (Inspect's viewer messages, Post's banner) that the widget walk
cannot read, after a few seconds of loading, for each block of a few projects; and the run window's tooltips.

    python explore_words_probe.py fr
"""
from __future__ import annotations

import faulthandler
import json
import sys
import time

faulthandler.dump_traceback_later(900, exit=True)
from harness import Explorer  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
ex = Explorer("words_layout", lang=LANG, size=(1920, 1080))
out = {}


def drawn(tag):
    got = []
    for x in ex.w.findChildren(QWidget):
        for attr in ("message", "banner"):
            v = getattr(x, attr, None)
            if isinstance(v, str) and v.strip() and x.isVisible():
                got.append([v, f"{type(x).__name__}.{attr}"])
    out[tag] = got
    print(tag, got, flush=True)


for kind in ("jpg_mistake", "image", "session", "mono"):
    root = ex.project(kind)
    ex.open(root)
    ex.settle(8)
    keys = [(it.node.key, it.node.sid) for it in ex.blocks()]
    for key, sid in keys:
        it = ex.block(key, sid=sid)
        if it is None or key == "final":
            continue
        ex.double_click_block(it)
        t0 = time.time()
        while time.time() - t0 < 4:
            ex.settle(5)
        drawn(f"{kind}/{key}@{sid}")
        f = ex.out / "shots" / f"probe_{LANG}_{kind}_{key}_{sid}.png"
        ex.w.grab().save(str(f))
        ex.w.set_mode(0)
        ex.settle(3)
    ex.w.set_mode(2)
    t0 = time.time()
    while time.time() - t0 < 6:
        ex.settle(5)
    drawn(f"{kind}/post")
    ex.w.grab().save(str(ex.out / "shots" / f"probe_{LANG}_{kind}_post.png"))
(ex.out / f"probe_{LANG}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
ex.close()
print("DONE")
