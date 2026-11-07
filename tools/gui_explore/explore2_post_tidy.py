"""post_wires round 2: the anomaly details in plain French (Python lists of names written as « a », « b »)."""
import json
import re
from pathlib import Path

F = Path("F:/astro_essais/A_REGARDER/exploration2/post_wires/anomalies.jsonl")
LIST = re.compile(r"\[('[^'\[\]]*'(?:, '[^'\[\]]*')*)\]")


def plain(m):
    names = [x for x in re.findall(r"'([^']*)'", m.group(1)) if x != "---"]
    return ", ".join(f"« {n.split(chr(9))[0].strip()} »" for n in names)


recs = [json.loads(x) for x in F.read_text(encoding="utf-8").splitlines() if x.strip()]
for r in recs:
    r["detail"] = LIST.sub(plain, r["detail"]).replace("«  »", "(aucun)")
F.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
for r in recs:
    print(r["severity"], "|", r["title"], "|", r["detail"][:300])
