"""sources_files, round 3, around the fixes: the Session panel's Delete button (two sessions, Ctrl+Z; a Fusion
channel), "+ Session" in a filter flow (the filter taken), a session renamed to an existing name."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore3_sources_common import TMP, Explorer, _, button, panel, ptexts, start, undo, visible, wait  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QLineEdit, QPushButton  # noqa: E402

start(400)
LOG = []


def log(*a):
    t = " ".join(str(x) for x in a)
    LOG.append(t)
    print(t, flush=True)


ex = Explorer("sources_files", lang="fr")
toasts = []


def hook():
    try:
        ex.page.toasted.connect(lambda t: toasts.append(t))
    except Exception:  # noqa: BLE001
        pass


def sess():
    return [(x.id, x.name, x.filter) for x in ex.s.p.sessions]


def flows():
    return [(q.name, ex.s.p.workflow(q.pid).sessions) for q in ex.s.paths]


for rep in (1, 2):
    # a. two sessions: the panel's Delete button, then Ctrl+Z
    root = ex.project("two_sessions", f"del{rep}")
    ex.open(root)
    wait(ex, 0.8)
    hook()
    ex.click_block(ex.block("session", sid=2))
    wait(ex, 0.3)
    n = len(toasts)
    b = button(panel(ex), _("Delete"), QPushButton)
    log("a", rep, "button", b is not None and b.isEnabled())
    ex.click(b)
    wait(ex, 0.6)
    s1 = sess()
    how = undo(ex)
    log("a", rep, "after delete", s1, toasts[n:], "undo", how, sess(), "sel", ex.s.sel_step)
    ex.w.close_project()
    wait(ex, 0.3)

    # b. mono with Fusion: the Ha Session's panel Delete button
    from astro_gpu.core import project2, workflow as wfl  # noqa: E401
    from astro_gpu.core.project import Target
    name = f"monoDel{rep}"
    root = ex.ws / name
    p = project2.create(root, name, Target(name="M 31"))
    p.stack_mode = "per_session"
    for i, f in enumerate(("Ha", "OIII")):
        p.sessions.append(project2.Session(i + 1, f, lights=[str(ex.fits_frames(f"{name}/{f}", 4, f))], filter=f))
    project2.save(p)
    for i, f in enumerate(("Ha", "OIII")):
        w = p.add_workflow(f, wfl.TEMPLATES["Classic stack"])
        w.sessions = [i + 1]
        p.set_blocks(w.id, [wfl.node(b) for b in ("calibration", "registration", "stacking", "astrometry")])
    project2.save(p)
    ex.open(root)
    wait(ex, 1.0)
    hook()
    ex.double_click_card("fusion")
    wait(ex, 1.0)
    ex.click_block(ex.block("session", sid=1))
    wait(ex, 0.3)
    b = button(panel(ex), _("Delete"), QPushButton)
    n = len(toasts)
    log("b", rep, "button", b is not None and b.isEnabled(), b.toolTip() if b else None)
    if b is not None and b.isEnabled():
        ex.click(b)
        wait(ex, 0.6)
    log("b", rep, "after", sess(), flows(), toasts[n:])
    ex.shot(f"x4_fusion_channel_session_button_{rep}")
    # c. "+ Session" under the Ha session: the filter taken?
    from astro_gpu.client.graph import GhostItem
    gh = [g for g in ex.view.scene().items() if isinstance(g, GhostItem) and "Session" in (g.text or "")
          and not getattr(g, "new_flow", False)]
    log("c", rep, "ghosts", [(g.text, round(g.sceneBoundingRect().center().y())) for g in gh])
    if gh:
        ex.file_queue = [str(ex.raws(f"plus{rep}/night", 4))]
        r = gh[0].sceneBoundingRect()
        n = len(toasts)
        ex.click(ex.view.viewport(), ex.view.mapFromScene(QPointF(r.center().x(), r.center().y())))
        wait(ex, 1.0)
        ex.close_dialogs()
        log("c", rep, "after + Session", sess(), flows(), toasts[n:])
        ex.shot(f"x4_plus_session_filter_{rep}")
    # d. rename the OIII session « Ha » (an existing name)
    ex.click_block(ex.block("session", sid=2))
    wait(ex, 0.3)
    ed = next((w for w in visible(panel(ex), QLineEdit)), None)
    n = len(toasts)
    if ed is not None:
        ed.setFocus()
        ed.selectAll()
        from PySide6.QtTest import QTest
        QTest.keyClicks(ed, "Ha")
        QTest.keyClick(ed, Qt.Key_Return)
        wait(ex, 0.5)
    log("d", rep, "rename to Ha", sess(), toasts[n:], ex.w.status.text(), ed.text() if ed else None)
    ex.shot(f"x4_rename_existing_{rep}")
    ex.w.close_project()
    wait(ex, 0.3)

(ex.out / "logs" / "extra4b_log.txt").write_text("\n".join(LOG), encoding="utf-8")
print("DONE", TMP)
ex.close()
