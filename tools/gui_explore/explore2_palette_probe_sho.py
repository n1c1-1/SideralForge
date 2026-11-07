"""palette_matrix round 2: the SHO template project: which workflows each drawn block stands for, the Output blocks on
screen, the plan's stacking jobs."""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("probe_sho")
ex = m.ex
for attempt in (1, 2):
    m._k = getattr(m, "_k", 0) + 1
    m.new_project(f"SHO{m._k}", "sho")
    ex.settle(8)
    s = ex.s
    print(attempt, "workflows", [(q.name, list(s.sessions_of(q)), q.source and list(q.source)[:2],
                                  [n["block"] for n in s.graph(q)]) for q in s.paths])
    print(attempt, "sessions", [(x.id, x.name, x.filter) for x in s.p.sessions])
    for it in sorted(ex.blocks(), key=lambda i: (i.node.col, i.node.y)):
        print("   ", it.node.key, repr(it.title), [q.name for q in it.node.paths], "sid", it.node.sid,
              "|", it.summary[:40] if isinstance(it.summary, str) else "")
    try:
        pl, pause = s.plan_run()
        print(attempt, "plan jobs", [(j.stage, getattr(j, "pid", None)) for j in pl.jobs][:30])
    except Exception as e:  # noqa: BLE001
        print(attempt, "plan raises", type(e).__name__, e)
    if attempt == 2:
        ex.shot("sho_modele_ouverture")
    # the same project once each session has its frames
    root = s.p.root
    ex.w.close_project(go_home=True)
    ex.settle(4)
    from astro_gpu.core import project2
    p = project2.load(root)
    for x in p.sessions:
        x.lights = [str(ex.fits_frames(f"sho{m._k}/{x.filter}", 4, x.filter))]
    project2.save(p)
    ex.open(root)
    ex.settle(10)
    print(attempt, "WITH FRAMES")
    for it in sorted(ex.blocks(), key=lambda i: (i.node.col, i.node.y)):
        if it.node.key in ("alignement", "empilement", "etoiles", "post"):
            print("   ", it.node.key, repr(it.title), [q.name for q in it.node.paths])
    if attempt == 2:
        ex.shot("sho_modele_avec_brutes")
ex.close()
