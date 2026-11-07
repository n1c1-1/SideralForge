"""block_actions: a compact reading of the raw_*.json files (what to judge), printed."""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path("F:/astro_essais/A_REGARDER/exploration2/block_actions")
ENGLISH = re.compile(r"\b(Delete|Switch|Send|Remove|Rename|Duplicate|Save|Inspect|Pause|Mark|Open|Take|New|block|"
                     r"only|Del|Split|stars|layer|Output|Image file|Add)\b")
BRIEF = "--brief" in sys.argv
args = [a for a in sys.argv[1:] if a != "--brief"]
files = sorted(OUT.glob("raw_*.json")) if not args else [OUT / f"raw_{a}.json" for a in args]
ovs = set()
menus_seen: dict = {}
for f in files:
    d = json.loads(f.read_text(encoding="utf-8"))
    print("=" * 30, f.stem, d.get("counts"), "build overlaps", d["built"]["overlaps"])
    for r in d["records"]:
        b = r["block"]
        if r["action"] == "observe":
            print(f"\n## {r['title']}  [{b[0]} sid={b[1]} {b[2]}] rem={r['removable']} ports={r['ports']}")
            if not BRIEF: print("   tip:", r["tooltip"].replace("\n", " | "))
            print("   menu:", [(x["text"] + ("" if x.get("enabled", True) else " [grey]")) for x in r["menu"]])
            print("   panel titles:", r["panel"]["titles"], "buttons:",
                  [(x["text"] + ("" if x["enabled"] else " [grey]")) for x in r["panel"]["buttons"]][:12])
            print("   click sel:", r["click_sel"], " dclick:", {k: v for k, v in r["dclick"].items() if k != "shot"
                                                               and k != "inspect_texts"},
                  (r["dclick"].get("inspect_texts") or [])[:3])
            for k in ("popups", "modals", "exc"):
                if r.get(k):
                    print(f"   !! {k}:", r[k])
            for x in r["menu"]:
                menus_seen.setdefault(x.get("text"), set()).add(b[0])
            continue
        fl = []
        if r.get("error"):
            print("   ", r["action"], "ERROR", r["error"])
            continue
        if r["exc"]:
            fl.append("EXC")
        if r["changed"] and not r["undo_ok"]:
            fl.append("UNDO")
        if r["changed"] and not r["redo_ok"]:
            fl.append("REDO")
        if r["popups"]:
            fl.append("POPUP")
        if r["modals"]:
            fl.append("MODAL")
        if r["overlaps"]:
            ovs.update(r["overlaps"])
            if not BRIEF:
                fl.append("OVERLAP")
        silent_kind = any(w in r["action"] for w in ("Del key", "D key", "cross", "drag to palette", "Supprimer", "Désactiver"))
        if silent_kind and not r["changed"] and not r["toasts"] and not r["modals"] and not r["popups"]:
            fl.append("SILENT")
        if BRIEF and not fl:
            continue
        ch = "; ".join(c[:160] for c in r["changed"])[:400]
        print(f"   - {r['action'][:45]:45s} {'/'.join(fl):12s} chg[{ch}] toast={r['toasts'][-2:]} "
              f"sel={r['sel_after'][0]}|{r['sel_after'][2]} mode={r['mode_after']}")
        if r["popups"]:
            print("        popups:", r["popups"])
        if r["modals"]:
            print("        modals:", r["modals"])
        if r["exc"]:
            print("        exc:", r["exc"][-1][-600:])
        if r["overlaps"] and not BRIEF:
            print("        overlaps:", r["overlaps"])
        if r["changed"] and not r["undo_ok"]:
            print("        undo diff:", r["undo_diff"][:3])
        if r["changed"] and not r["redo_ok"]:
            print("        redo diff:", r["redo_diff"][:3])
print("\nMENU TEXTS")
for t, ks in sorted(menus_seen.items(), key=lambda x: str(x[0])):
    print(f"  {t!r:60s} {sorted(ks)}{'  <EN?>' if t and ENGLISH.search(t) else ''}")
