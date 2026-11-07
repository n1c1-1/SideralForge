"""usages C: every gallery template, one by one: made from Home, what is on disk and the unsaved mark right after
Create, Save, the window closed, the project opened in a new window, compared."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore5_usages_common import EXC, U, log, start  # noqa: E402

start(1500)
rep = sys.argv[1] if len(sys.argv) > 1 else "1"
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
u = U("fr")
ex = u.ex
from astro_gpu.client.gallery import GALLERY  # noqa: E402
from astro_gpu.core import project2  # noqa: E402

fits_img = ex.image_file("pile.fits")
png = ex.image_file("fini.png")
res = {}
for t in GALLERY:
    key = t["key"]
    if only and key not in only:
        continue
    img = ""
    if t.get("source") == "image":
        img = str(png if t.get("stretched") else fits_img)
    chips = ("Ha", "OIII") if key == "mono_each" else ()
    name = f"T_{key}_{rep}"
    n_exc = len(EXC)
    root, cap = u.new_from_home(name, key, image=img, chips=chips)
    if ex.w.s is None:
        log("NOT CREATED", key, cap)
        res[key] = {"created": False, "cap": cap}
        continue
    snap0 = u.snapshot()
    disk = json.loads((root / project2.FILE).read_text(encoding="utf-8")) if (root / project2.FILE).exists() else {}
    on_disk = [(w["name"], len(w.get("blocks") or [])) for w in disk.get("workflows", [])]
    in_model = [(w.name, len(w.blocks or [])) for w in ex.s.p.workflows]
    ex.shot(f"C_{key}_{rep}_created")
    before, saved, after, diffs = u.roundtrip(root, f"modèle {key}", [f"Accueil > Nouveau projet > {key}"])
    ex.shot(f"C_{key}_{rep}_reopened")
    res[key] = {"dirty_after_create": snap0["dirty"], "title_after_create": snap0["title"], "on_disk": on_disk,
                "in_model": in_model, "undo_after_create": snap0["undo"], "diffs": diffs,
                "flows": after["flows"], "blocks": [b[1] for b in after["blocks"]], "sel": after["selection"],
                "status": after["status"], "run": after["run"], "exc": EXC[n_exc:]}
    log("TEMPLATE", key, json.dumps({k: v for k, v in res[key].items() if k != "blocks"}, ensure_ascii=False,
                                    default=str)[:1500])
    ex.w.close_project()
(ex.out / f"C_templates_{rep}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                                encoding="utf-8")
ex.w.close()
