"""Words, round 2: the compact-name rule (names of two words, no articles; sentences only in tooltips and explanations)
checked on what the walks saw on screen (walk_<lang>_<w>x<h>.json of explore2_words_walk.py): buttons, menu items,
tabs, palette cards, check boxes, column heads, window titles, and the short labels (setting names, section titles).
Writes <out>/compact_names.txt.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

OUT = Path("F:/astro_essais/A_REGARDER/exploration3/words_layout")
ART = {"fr": r"\b(le|la|les|l'|un|une|des|du|de la|de l'|au|aux|ce|cette|ces|son|sa|ses|leur|leurs)\b",
       "en": r"\b(the|a|an|this|these|its|their)\b"}
USER = re.compile(r"Session \d|Workflow|Flow|Full|Chaine complete|Mono fusion|Image etoiles|Galerie|Bascule|Projet \w+ "
                  r"au nom un peu long pour la carte|\w+_\d+|M 31|M 42|NGC 7000|NVIDIA RTX \d+|DSC\d+\.\w+|stack\.fits|"
                  r"gui_explore\w*|C:/\S+|Bench|Empilement brut \(copie\)|Raw stack \(copy\)|Copie de « \w+ »|"
                  r"Copy of \"\w+\"")


def kind_of(where: str) -> str | None:
    w = where
    if "tip" in w or "tooltip" in w or "placeholder" in w or "hint" in w:
        return None
    if w.startswith("menu of") or w.startswith("menu ") or " > " in w or w in ("menu bar",):
        return "menu"
    for k, name in (("QPushButton", "button"), ("QToolButton", "button"), ("QCheckBox", "check box"),
                    ("QRadioButton", "radio"), ("tab", "tab"), ("header", "column"), ("palette card", "palette"),
                    ("window title", "window title"), ("QGroupBox", "group")):
        if k in w:
            return name
    if w.startswith("QLabel") or w.startswith("ElidedLabel"):
        return "label"
    if "graph" in w and ".title" in w:
        return "graph title"
    return None


def words(t: str) -> list[str]:
    t = re.sub(r"\t.*$", "", t)                     # the shortcut
    t = re.sub(r"\s+—\s+.*$", "", t)                 # "Switch off — why": the reason is an explanation
    t = re.sub(r"[·:…(),«»\"“”'’]|\.\.\.|  +", " ", t)
    t = re.sub(r"[≈≤≥]\s*\d+\s*\w*|\d+([.,]\d+)?\s*(%|px|min|s|Go|GB|Mo|MB)?", " ", t)
    return [x for x in t.split() if re.search(r"[A-Za-zÀ-ÿ]", x) and x not in ("+", "→", "◂", "▸", "▾", "◇", "⑂")]


def main():
    found = defaultdict(lambda: defaultdict(set))      # lang -> (kind, text) -> screens
    for f in sorted(OUT.glob("walk_*.json")):
        lang = f.name.split("_")[1]
        data = json.loads(f.read_text(encoding="utf-8"))
        for screen, rec in data.items():
            if not isinstance(rec, dict) or screen.startswith("switch/"):
                continue
            for t, where in rec.get("texts", []):
                k = kind_of(where)
                if k is None or not t or "\n" in t:
                    continue
                if k == "label" and (len(t) > 45 or t.rstrip().endswith((".", ":")) and len(t) > 25):
                    continue                          # a sentence: an explanation, allowed
                c = USER.sub(" ", t)
                ws = words(c)
                art = re.search(ART[lang], c, re.I)
                if len(ws) > 2 or art:
                    found[lang][(k, t.split("\t")[0])].add(screen)
    out = []
    for lang in sorted(found):
        by_kind = defaultdict(list)
        for (k, t), ss in found[lang].items():
            by_kind[k].append((t, sorted(ss)))
        out.append(f"######## {lang}")
        for k in sorted(by_kind):
            items = sorted(by_kind[k])
            out.append(f"== {k} ({len(items)})")
            for t, ss in items:
                out.append(f"  «{t}»   on {len(ss)}: {', '.join(ss[:3])}{' …' if len(ss) > 3 else ''}")
    (OUT / "compact_names.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    for lang in sorted(found):
        print(lang, len(found[lang]))


if __name__ == "__main__":
    main()
