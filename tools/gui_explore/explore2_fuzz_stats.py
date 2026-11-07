"""Round 2 fuzz: how many gestures of each kind were really played (not a no-op) in runs2."""
import collections
import json
from pathlib import Path

RUNS = Path("F:/astro_essais/A_REGARDER/exploration2/fuzz/runs2")
tot, noop = collections.Counter(), collections.Counter()
starts = collections.Counter()
for f in RUNS.glob("seed_*.json"):
    if f.name.endswith(".progress.json"):
        continue
    r = json.loads(f.read_text(encoding="utf-8"))
    starts[r["start"].get("template") or r["start"].get("kind")] += 1
    for a, s in zip(r["actions"], r["steps"]):
        tot[a[0]] += 1
        if s.endswith(")") and ("(aucun" in s or "(pas de" in s or "rien à" in s or "(Post: aucun" in s
                                or "menu vide" in s or "(panneau: aucun" in s):
            noop[a[0]] += 1
for k, n in tot.most_common():
    print(f"{k:12s} {n:4d}  sans effet {noop[k]:3d}")
print(sum(tot.values()), "gestes;", sum(noop.values()), "sans effet")
print(dict(starts))
