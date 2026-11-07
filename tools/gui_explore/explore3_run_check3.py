"""run_status round 3: the new strings of the fixes in English (python explore3_run_check3.py --en)."""
from __future__ import annotations

from explore2_run_explore import _fail_with, run_menu, win  # noqa: F401  (--en handled there)
from explore2_run_recheck import P, Qt, blk, dump, ex, f5, fresh, obs, place, run_checks, shift_f5, texts  # noqa: F401
from explore3_run_check2 import mono_fusion, set_filter, toast


def shot(name, widget=None):
    return ex.shot(f"k{name}", widget)


def g01_en_strings():
    fresh("jpg_mistake")
    win()
    place("stacking")
    d = f5()
    obs("g01", jpg=dump(d).get("texts"))
    ex.close_dialogs()
    d = _fail_with(RuntimeError("worker crashed: CUDA error 999"))
    obs("g01", fail=texts(d), console=ex.w.logp.console.toPlainText()[-300:], log=[x for _t, x in ex.w.logp.rows()][:3])
    ex.close_dialogs()
    fresh("session")
    obs("g01", menu=run_menu())


def g02_en_fusion():
    win()
    ha, oiii = mono_fusion()
    ex.click_block(blk("astrometrie", ha))
    ex.key(Qt.Key_Delete)
    ex.settle(6)
    obs("g02", toast=toast())
    set_filter("session@2", "SII")
    obs("g02", top=(ex.w.bt_run.text(), ex.w.bt_run.toolTip(), ex.w.status.text()), fusion=blk("fusion").toolTip(),
        shot=shot("g02_en_fusion"))
    fresh("mono")
    c = ex.card("fusion")
    obs("g02", card=getattr(c, "why", None) or getattr(c, "reason", None))


if __name__ == "__main__":
    run_checks(dict(globals()), prefix="g", dump_name="check3_dump.json")
