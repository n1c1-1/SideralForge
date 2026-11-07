"""gros_projets (v2: one workflow shown as computed, the other not): a filter added in Post (its "+ Filter" menu, real gesture) on a project never computed: do the
workflows keep their image in Post (the wire, the Post block's "n calques"), now and after Save and reopening?
Run twice (two fresh projects). Args: lang (fr)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Explorer  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

ex = Explorer("gros_projets", lang=sys.argv[1] if len(sys.argv) > 1 else "fr")


def P(*a):
    print(*a, flush=True)


def project(name, n_wf):
    root = ex.ws / name
    p = project2.create(root, name, Target(name="M 31"))
    p.stack_mode = "per_session"
    for k in range(n_wf):
        p.sessions.append(project2.Session(k + 1, f"Nuit {k + 1}", lights=[str(ex.raws(f"{name}/n{k}", 12))]))
    project2.save(p)
    for k in range(n_wf):
        w = p.add_workflow(f"Couleur {k + 1}", wfl.TEMPLATES["Classic stack"])
        w.sessions = [k + 1]
        p.set_blocks(w.id, [wfl.node(b) for b in ("calibration", "registration", "stacking", "astrometry", "gradient")])
    project2.save(p)
    return root


def state(tag):
    post = next((b for b in ex.blocks() if b.node.key == "final"), None)
    wires = [w for w in ex.view.scene().items() if w.__class__.__name__ == "WireItem" and getattr(w, "pid", None)]
    d = {"tag": tag, "post_block": post.summary if post else None, "wires_to_post": len(wires),
         "status": ex.w.status.text(), "saved_post": json.dumps(ex.s.p.post)[:300]}
    P("STATE", json.dumps(d, ensure_ascii=False))
    return d


for run in (1, 2):
    root = project(f"postfilter2_{run}", 2)
    if ex.w is not None and ex.w.project is not None:
        ex.w.close_project()
        ex.settle(10)
    ex.open(root)
    ex.settle(20)
    s = ex.s
    orig = s._sync

    def fake(*a, **k):                     # workflow 1 counts as computed (its image rendered), workflow 2 not
        orig(*a, **k)
        for v in s.paths:
            v.rendered = v.pid == s.paths[0].pid
    s._sync = fake
    s._sync()
    ex.settle(10)
    a = state(f"run{run} opened")
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    v = ex.w.post_holder.page.view
    P("post page layers before", [x.get("name") for x in v.layers])
    ex.menu_pick = "Courbes" if ex.lang == "fr" else "Curves"
    P("b_filter visible", v.b_filter.isVisible(), "layers rows", [x.get("name") for x in v.layers])
    ex.click(v.b_filter)
    ex.settle(10)
    P("menu", [m.get("text") for m in (ex.menus[-1] if ex.menus else [])][:20])
    P("post page layers after", [x.get("name") for x in v.layers])
    shot_post = ex.shot(f"postfilter2_run{run}_post")
    ex.click(ex.w.mode_btns[0])
    ex.settle(15)
    b = state(f"run{run} after + Filtre, back to graph")
    shot_graph = ex.shot(f"postfilter2_run{run}_graph")
    ex.w.save()
    ex.settle(10)
    ex.w.close_project()
    ex.settle(10)
    ex.open(root)
    ex.settle(20)
    c = state(f"run{run} saved and reopened")
    shot_re = ex.shot(f"postfilter2_run{run}_reopened")
ex.close()
P("DONE")
