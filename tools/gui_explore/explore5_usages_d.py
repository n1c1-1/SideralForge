"""usages D: closing with unsaved changes (the real prompt), Save as, an outside change (a session folder renamed or
gone, the Image's file gone), two workflows sharing a trunk, a workflow deleted then Ctrl+Z then Save/reopen, Post
layers kept."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore5_usages_common import EXC, TMP, U, _, compare, log, start, wait  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

start(1200)
rep = sys.argv[1] if len(sys.argv) > 1 else "1"
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
u = U("fr")
ex = u.ex
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.core import project2  # noqa: E402

REAL_PROMPT = {"on": False}
_orig_platform = QApplication.platformName
try:
    QApplication.platformName = staticmethod(lambda: "windows" if REAL_PROMPT["on"] else "offscreen")
    log("platformName patched", QApplication.platformName())
except Exception as e:  # noqa: BLE001
    log("platformName not patchable", e)


def want(k):
    return only is None or k in only


def disk(root):
    d = json.loads((Path(root) / project2.FILE).read_text(encoding="utf-8"))
    return {"workflows": [(w["name"], [b["block"] for b in (w.get("blocks") or [])]) for w in d["workflows"]],
            "sessions": [(s["id"], s["name"], s.get("lights")) for s in d["sessions"]],
            "params": [w.get("params") for w in d["workflows"]], "post": d.get("post")}


def colour_project(name, chain=("calibration", "registration", "stacking", "astrometry", "gradient")):
    lights = ex.raws(f"{name}/lights", 6)
    root, cap = u.new_from_home(name, "blank")
    u.place("session")
    u.add_frames(1, lights, 0)
    u.place(*chain)
    return root, lights


# ================================================================== P. closing with unsaved changes: the prompt
if want("P"):
    # P1: a project just made from a template, closed at once: "Don't save"
    root, _c = u.new_from_home(f"FermeP1_{rep}", "full")
    s0 = u.snapshot()
    REAL_PROMPT["on"] = True
    u.box_answer = [_("Don't save")]
    asked, still = u.close_window()
    REAL_PROMPT["on"] = False
    log("P1 asked", asked, "still open", still, "dirty after create", s0["dirty"], s0["flows"])
    u.reopen_new_window(root)
    s1 = u.snapshot()
    log("P1 reopened", s1["flows"], s1["status"], s1["run"], [d.__class__.__name__ for d in ex.dialogs()], ex.modals[-2:])
    ex.shot(f"P1_{rep}_template_not_saved_reopened")
    (ex.out / f"P1_{rep}.json").write_text(json.dumps({"asked": asked, "created": s0, "reopened": s1},
                                                       ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ex.w.close_project()
    # P2: an edit, close: Cancel (stays), then Save (kept); another edit, close: Don't save (dropped)
    root, lights = colour_project(f"FermeP2_{rep}")
    u.save()
    u.place("color")
    e1 = u.snapshot()
    REAL_PROMPT["on"] = True
    u.box_answer = [_("Cancel")]
    asked_c, still_c = u.close_window()
    log("P2 cancel", asked_c, "still", still_c, "dirty", ex.w.s.dirty if still_c and ex.w.s else None,
        "project", ex.w.project is not None if still_c else None)
    if still_c:
        ex.shot(f"P2_{rep}_after_cancel")
    after_cancel = u.snapshot() if still_c and ex.w.s is not None else None
    u.box_answer = [_("Save")]
    asked_s, still_s = u.close_window()
    REAL_PROMPT["on"] = False
    u.reopen_new_window(root)
    s_saved = u.snapshot()
    log("P2 save", asked_s, still_s, compare(e1, s_saved).keys())
    u.place("denoise")
    e2 = u.snapshot()
    REAL_PROMPT["on"] = True
    u.box_answer = [_("Don't save")]
    asked_d, still_d = u.close_window()
    REAL_PROMPT["on"] = False
    u.reopen_new_window(root)
    s_drop = u.snapshot()
    log("P2 drop", asked_d, still_d, "denoise kept?", any(b[0] in ("denoise", "debruitage") for b in s_drop["blocks"]),
        compare(s_saved, s_drop).keys(), s_drop["selection"], e2["selection"])
    (ex.out / f"P2_{rep}.json").write_text(json.dumps({
        "asked_cancel": asked_c, "still_after_cancel": still_c, "cancel_diff": compare(e1, after_cancel) if after_cancel
        else None, "asked_save": asked_s, "saved_diff": compare(e1, s_saved), "asked_drop": asked_d,
        "drop_diff_vs_saved": compare(s_saved, s_drop), "edit2_sel": e2["selection"], "drop_sel": s_drop["selection"]},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ex.w.close_project()

# ================================================================== S. Save as
if want("S"):
    root, lights = colour_project(f"SousS_{rep}")
    u.save()
    u.branch("color", "astrometrie")
    st = u.blk("stacking") or u.blk("empilement")
    u.to_post(st)
    ex.click_block(u.blk("gradient") or u.blk("fond"))
    wait(ex, 0.3)
    edited = u.snapshot()
    disk_before = disk(root)
    # Save as, the folder dialog cancelled
    ex.file_queue = [""]
    ex.w.save_as()
    wait(ex, 0.8)
    disk_cancel = disk(root)
    log("S cancel: original written?", disk_cancel != disk_before, "dirty now", ex.w.s.dirty, ex.w.windowTitle())
    # Save as into a folder that holds a project already
    other = ex.ws / f"FermeP2_{rep}" if (ex.ws / f"FermeP2_{rep}").exists() else root
    n = len(ex.popups)
    ex.file_queue = [str(other)]
    ex.w.save_as()
    wait(ex, 0.6)
    log("S into a project folder", ex.popups[n:], ex.w.windowTitle())
    # Save as, a new folder (made by the dialog, empty)
    u.place("denoise")                               # an edit not saved yet
    edited2 = u.snapshot()
    disk_before2 = disk(root)
    newroot = ex.ws / f"SousS_{rep}_copie"
    newroot.mkdir(exist_ok=True)
    ex.file_queue = [str(newroot)]
    ex.w.save_as()
    wait(ex, 1.5)
    u.hook_toasts()
    now = u.snapshot()
    log("S saved as", ex.w.project.root if ex.w.project else None, now["title"], now["status"], now["dirty"])
    log("S original on disk changed by Save as?", disk(root) != disk_before2,
        [w[0] for w in disk(root)["workflows"]], [w[0] for w in disk_before2["workflows"]])
    d = compare(edited2, now)
    log("S diff edited -> saved-as window", json.dumps(d, ensure_ascii=False, default=str)[:2500])
    ex.shot(f"S_{rep}_saved_as")
    # the copy closed, reopened in a new window
    u.reopen_new_window(newroot)
    again = u.snapshot()
    d2 = compare(now, again)
    log("S copy reopened diff", json.dumps(d2, ensure_ascii=False, default=str)[:2500])
    newdef = json.loads((newroot / project2.FILE).read_text(encoding="utf-8"))
    log("S copy name/scratch", newdef.get("name"), newdef.get("scratch_dir"), [s.get("lights") for s in newdef["sessions"]])
    (ex.out / f"S_{rep}.json").write_text(json.dumps({
        "disk_before": disk_before, "disk_after_cancel": disk_cancel, "diff_saveas": d, "diff_copy_reopen": d2,
        "name": newdef.get("name"), "orig_after": disk(root), "orig_before2": disk_before2},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ex.w.close_project()

# ================================================================== O. outside change: a session folder renamed / gone, Image file gone
if want("O"):
    root, lights = colour_project(f"DehorsO_{rep}")
    u.save()
    base = u.snapshot()
    ex.w.close()
    ex.settle(4)
    ex.w = None
    moved = lights.with_name(lights.name + "_renomme")
    os.replace(lights, moved)
    u.reopen_new_window(root)
    s1 = u.snapshot()
    sess = u.blk("session") or u.blk("sources")
    ex.click_block(sess)
    wait(ex, 0.4)
    from explore5_usages_common import ptexts  # noqa: E402
    pt = ptexts(ex)
    sess = u.blk("session") or u.blk("sources")
    log("O renamed: status", s1["status"], "run", s1["run"], "session block", sess.summary if sess else None,
        "tip", sess.toolTip()[:200] if sess else None, "dirty", s1["dirty"], "title", s1["title"])
    log("O panel", pt[:14])
    ex.shot(f"O_{rep}_folder_renamed")
    n = len(ex.popups)
    ex.click(ex.w.bt_run) if ex.w.bt_run.isEnabled() else None
    wait(ex, 0.5)
    log("O run", ex.w.status.text(), ex.popups[n:], u.toasts[-2:])
    (ex.out / f"O_{rep}.json").write_text(json.dumps({"base": base, "renamed": s1, "panel": pt},
                                                      ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ex.w.close_project()
    os.replace(moved, lights)
    # the image file gone
    img = ex.image_file(f"partie_{rep}.fits")
    root2, _c = u.new_from_home(f"DehorsI_{rep}", "image", image=str(img))
    u.save()
    ex.w.close()
    ex.settle(4)
    ex.w = None
    os.replace(img, img.with_name("ailleurs_" + img.name))
    u.reopen_new_window(root2)
    s2 = u.snapshot()
    im = u.blk("image")
    ex.click_block(im)
    wait(ex, 0.4)
    im = u.blk("image")
    log("O image gone: status", s2["status"], "run", s2["run"], "image block", im.summary if im else None,
        "tip", im.toolTip()[:200] if im else None, "panel", ptexts(ex)[:10])
    ex.shot(f"O_{rep}_image_gone")
    ex.w.close_project()

# ================================================================== W. two workflows sharing a trunk; delete one, Ctrl+Z, Save, reopen; Post layers
if want("W"):
    root, lights = colour_project(f"TroncW_{rep}", ("calibration", "registration", "stacking", "astrometry",
                                                    "gradient", "color"))
    ok = u.branch("gradient", "empilement")
    log("W branch", ok, [q.name for q in ex.s.paths])
    u.to_post(u.blk("stacking") or u.blk("empilement"))
    # Post opened (its layers made), back to the graph
    ex.w.set_mode(2)
    wait(ex, 1.5)
    ex.shot(f"W_{rep}_post")
    ex.w.set_mode(0)
    wait(ex, 0.8)
    layers0 = json.loads(json.dumps(ex.s.p.post, default=str))
    log("W layers after Post opened", [(l.get("name"), l.get("pid"), l.get("visible", True)) for l in
                                       layers0.get("layers", [])], "dirty", ex.s.dirty)
    u.save()
    full = u.snapshot()
    # delete the branch's workflow from its block's menu
    br = next((q for q in ex.s.paths if "Gradient" in q.name and q.name != ex.s.paths[0].name), None)
    it = next((b for b in ex.blocks() if br is not None and [q.pid for q in b.node.paths] == [br.pid]), None)
    log("W branch block", it.title if it else None, br.name if br else None)
    if it is not None:
        menu = ex.right_click_block(it, _("Delete workflow"))
        wait(ex, 0.8)
    deleted = u.snapshot()
    log("W deleted", deleted["flows"], u.toasts[-1:], [l.get("name") for l in ex.s.p.post.get("layers", [])])
    from explore3_sources_common import undo as undo_k  # noqa: E402
    how = undo_k(ex)
    back = u.snapshot()
    log("W undo", how, back["flows"], "diff vs full", json.dumps(compare(full, back), ensure_ascii=False,
                                                                  default=str)[:2000])
    ex.shot(f"W_{rep}_undone")
    before, saved, after, diffs = u.roundtrip(root, "tronc partagé", ["branche", "Supprimer le workflow", "Ctrl+Z"])
    d_full = compare(full, after)
    log("W after reopen vs before delete", json.dumps(d_full, ensure_ascii=False, default=str)[:2500])
    ex.shot(f"W_{rep}_reopened")
    # delete, save, reopen: gone; Ctrl+Z after reopen
    br = next((q for q in ex.s.paths if "Gradient" in q.name and q.name != ex.s.paths[0].name), None)
    it = next((b for b in ex.blocks() if br is not None and [q.pid for q in b.node.paths] == [br.pid]), None)
    if it is not None:
        ex.right_click_block(it, _("Delete workflow"))
        wait(ex, 0.8)
    before2, saved2, after2, diffs2 = u.roundtrip(root, "workflow supprimé", ["Supprimer le workflow", "Ctrl+S"])
    log("W deleted+saved+reopened", after2["flows"], [l.get("name") for l in ex.s.p.post.get("layers", [])],
        "undo", after2["undo"])
    (ex.out / f"W_{rep}.json").write_text(json.dumps({"full": full, "deleted": deleted, "back": back, "after": after,
                                                       "d_full": d_full, "diffs": diffs, "after2": after2,
                                                       "diffs2": diffs2}, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
    ex.w.close_project()
log("EXC", len(EXC), "\n".join(e[-1200:] for e in EXC))
if ex.w is not None:
    ex.w.close()
