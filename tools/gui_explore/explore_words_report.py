"""Words and layout, the analysis: the walks (walk_<lang>_<w>x<h>.json) compared, the suspect strings listed in
suspect_strings.txt, the layout problems grouped in layout_groups.txt. The anomalies themselves are written by
explore_words_notes.py after a human look at these lists.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from explore_words_static import EXTERNAL, english_words, load_catalogue, punct_problems  # noqa: E402

OUT = Path("F:/astro_essais/A_REGARDER/exploration2/words_layout")
# names the harness gives (projects, workflows, sessions, folders): the user's words, never translated
USER_WORDS = re.compile(r"\b(Flow|Full|Bench|Workflow|Session \d|Chaine complete|Mono fusion|Image etoiles|Galerie|"
                        r"Bascule|two_flows|two_sessions|session_empty_wf|jpg_mistake|session\+image|legacy|mono|"
                        r"blank|image|session)_\d+\b|gui_explore_\w+|C:/Users/\S+|DSC\d+\.\w+|Light_\w+\.fits|"
                        r"stack\.fits|NVIDIA RTX \d+|\bFlow\b|\bFull\b|\bBench\b")
FRENCH_KEYS = re.compile(r"\b(debruitage|etoiles|empilement|astrometrie|alignement|couleur|fond|deconv)(~\d)?\b")
INTERNAL_ANY = re.compile(r"\w~\d|\b[a-z]+_[a-z_]+\b|\blegacy\b|\bv[12]\b|\bcuda:\d\b|Traceback|Error\b|\bNone\b|"
                          r"\{[a-z_]+\}|\.astroproj|project\.json|\bspec\b", re.I)


def load(tag):
    f = OUT / f"walk_{tag}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def clean(t: str) -> str:
    return USER_WORDS.sub(" ", t)


def main():
    cat, _where, _d = load_catalogue()
    values = set(cat.values())
    walks = {tag: load(tag) for tag in ("fr_1920x1080", "fr_1366x768", "en_1920x1080", "en_1366x768")}
    lines = defaultdict(lambda: defaultdict(set))       # kind -> text -> {screen}

    def add(kind, text, where):
        lines[kind][text].add(where)
    for tag, data in walks.items():
        lang = tag[:2]
        en = walks.get("en_" + tag[3:], {})
        for screen, rec in data.items():
            if not isinstance(rec, dict):
                continue
            where = f"{tag}:{screen}"
            en_texts = {t for t, _w in en.get(screen, {}).get("texts", [])}
            for m in rec.get("missing", []):
                if lang == "fr" and m not in values and re.search(r"[A-Za-z]{3,}", m) and not USER_WORDS.fullmatch(m):
                    add("FR: asked without a French entry (shows in English)", m, where)
            texts = [(t, w) for t, w in rec.get("texts", [])] + [(t, "toast") for t in rec.get("toasts", [])] + \
                [(" / ".join(p[1:]), "popup") for p in rec.get("popups", [])]
            for t, w in texts:
                c = clean(t)
                if lang == "fr":
                    ew = english_words(c)
                    if ew:
                        same = " (identical in English)" if t in en_texts else ""
                        add("FR: English words in a French text" + same, f"{t[:300]}   [{', '.join(ew)}] <{w}>", where)
                    for pb in punct_problems(c):
                        if "double space" in pb or "leading" in pb:
                            continue
                        add("FR: punctuation / plural", f"{pb}: {t[:200]} <{w}>", where)
                else:
                    if FRENCH_KEYS.search(c):
                        add("EN: an internal (French) key shown", f"{t[:250]} <{w}>", where)
                    if re.search(r"\b1 (frames|nights|workflows|blocks|layers|files|projects|sessions)\b", c) or \
                            re.search(r"\b([02-9]|\d{2,}) (frame|night|workflow|block|layer|file|project|session)\b", c):
                        add("EN: plural", f"{t[:200]} <{w}>", where)
                m = INTERNAL_ANY.search(c)
                if m:
                    add(f"{lang.upper()}: internal word", f"{t[:250]}   [{m.group(0)}] <{w}>", where)
                m = EXTERNAL.search(c)
                if m:
                    add(f"{lang.upper()}: external software name", f"{t[:250]}   [{m.group(0)}] <{w}>", where)
    out = []
    for kind in sorted(lines):
        items = lines[kind]
        out.append(f"== {kind} ({len(items)} texts)")
        for text, where in sorted(items.items()):
            ws = sorted(where)
            out.append(f"  {text}\n      on {len(ws)} screen(s): {', '.join(ws[:4])}{' …' if len(ws) > 4 else ''}")
    static = OUT / "static_catalogue.txt"
    if static.exists():
        out.append("\n\n######## the catalogue read as files (explore_words_static.py)\n")
        out.append(static.read_text(encoding="utf-8"))
    (OUT / "suspect_strings.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    # layout, grouped
    groups = defaultdict(set)
    for tag, data in walks.items():
        for screen, rec in data.items():
            if not isinstance(rec, dict):
                continue
            for x in rec.get("layout", []):
                key = re.sub(r"\d+(\.\d+)?", "#", x)
                groups[(tag, key)].add(f"{screen} :: {x}")
            if rec.get("error") or rec.get("errors"):
                groups[(tag, "ERROR")].add(f"{screen} :: {rec.get('error') or rec.get('errors')}")
    lo = []
    for (tag, key), ss in sorted(groups.items()):
        lo.append(f"== [{tag}] {key}  ({len(ss)})")
        for s in sorted(ss)[:8]:
            lo.append("   " + s[:400])
    (OUT / "layout_groups.txt").write_text("\n".join(lo) + "\n", encoding="utf-8")
    for kind in sorted(lines):
        print(f"{len(lines[kind]):5d}  {kind}")
    print(len(groups), "layout groups")


if __name__ == "__main__":
    main()
