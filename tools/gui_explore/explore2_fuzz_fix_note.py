"""Round 2 fuzz: the fuzz counts written into the notes they belong to (how many round-2 sequences hit each cause),
from runs2. Rewrites those lines of exploration2/fuzz/anomalies.jsonl."""
import json
import re
from pathlib import Path

OUT = Path("F:/astro_essais/A_REGARDER/exploration2/fuzz")
LOG = OUT / "anomalies.jsonl"
runs = []
for f in sorted((OUT / "runs2").glob("seed_*.json")):
    if not f.name.endswith(".progress.json"):
        runs.append(json.loads(f.read_text(encoding="utf-8")))


def seeds(pred):
    return sorted({r["seed"] for r in runs for x in r["failures"] if pred(x, r)})


fusion = seeds(lambda x, r: x["sig"].startswith("exc:KeyError") and "set_fusion_weight" in x["detail"])
delwf = seeds(lambda x, r: x["sig"].startswith("validate:") and "fusion reads" in x["sig"])
lines = [json.loads(x) for x in LOG.read_text(encoding="utf-8").splitlines() if x.strip()]
for x in lines:
    if x["title"].startswith("Un champ du panneau écrit"):
        x["detail"] = re.sub(r"vu aussi [^;]*?par le fuzz \(graines [^)]*\)",
                             f"vu aussi par le fuzz dans {len(fusion)} séquences sur {len(runs)} (graines "
                             f"{', '.join(map(str, fusion))})", x["detail"])
    if x["title"].startswith("« Supprimer workflow » sur un canal"):
        x["detail"] = re.sub(r"Vu par le fuzz \(graine 2010\)",
                             f"Vu par le fuzz dans {len(delwf)} séquences sur {len(runs)} (graines "
                             f"{', '.join(map(str, delwf))}) et par le balayage de tous les menus", x["detail"])
LOG.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
print(len(runs), "runs; fusion", fusion, "; delete workflow", delwf)
