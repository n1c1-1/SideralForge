"""block_actions round 2: a run of block actions, then Ctrl+Z all the way back and Ctrl+Y all the way forward: each
step must give back exactly the state seen after the same step. python explore2_blocks_chain_undo.py [seeds]"""
from __future__ import annotations

import faulthandler
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(900, exit=True)

from explore2_blocks_common import CHAIN, EXC, TOASTS, G, Explorer, state  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

ex = Explorer("block_actions")
g = G(ex)
SEEDS = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4]


def snap():
    st = state(ex)
    return json.dumps({"w": [(w["name"], w["blocks"], w["sessions"], w["bp"]) for w in st["workflows"]],
                       "s": [(x["id"], x["name"], x["filter"]) for x in st["sessions"]],
                       "post": getattr(ex.s.p, "post", None)}, sort_keys=True, default=str)


def step(rng):
    its = [b for b in ex.blocks() if b.node.key not in ("final",)]
    it = rng.choice(its)
    kind = rng.choice(["D", "Del", "Vers Post", "Pause vérif", "cross", "Sans étoiles vers Post", "+ Recombiner",
                       "branch"])
    label = f"{kind} on {it.title}"
    if kind == "D":
        ex.click_block(it)
        g.act()
        ex.key(Qt.Key_D)
    elif kind == "Del":
        ex.click_block(it)
        g.act()
        ex.key(Qt.Key_Delete)
    elif kind == "cross":
        g.cross(it)
    elif kind == "branch":
        if not g.drag_branch(rng.choice(["color", "gradient", "denoise"]), it.node.key,
                             it.node.paths[0].pid if it.node.paths else None):
            return label + " (no place)"
    else:
        g.menu(it, pick=kind)
    ex.settle(4)
    return label


for seed in SEEDS:
    rng = random.Random(seed)
    if __import__("os").environ.get("BA_MONO"):
        g.mono_fusion()
    elif __import__("os").environ.get("BA_TWO"):
        g.new("two_sessions", CHAIN)
    else:
        g.new("session", CHAIN)
    e0 = len(EXC)
    states, labels = [snap()], []
    for _ in range(8):
        try:
            lab = step(rng)
        except RuntimeError as e:          # a block item gone with a rebuild: skip
            lab = f"(script: {e})"
        s = snap()
        if s != states[-1]:
            states.append(s)
            labels.append(lab)
    bad = []
    for i in range(len(states) - 1, 0, -1):
        g.undo()
        if snap() != states[i - 1]:
            bad.append(f"Ctrl+Z after « {labels[i - 1]} » does not give back the state before it")
            break
    if not bad:
        for i in range(1, len(states)):
            g.redo()
            if snap() != states[i]:
                bad.append(f"Ctrl+Y n°{i} (« {labels[i - 1]} ») does not give back the state after it")
                break
    print(f"seed {seed}: {len(labels)} changing steps {labels} -> {'OK' if not bad else 'SUSPECT ' + bad[0]} "
          f"exc={[x.strip().splitlines()[-1] for x in EXC[e0:]]}", flush=True)
ex.close()

