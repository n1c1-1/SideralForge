"""Words, static part: the French catalogue read as files, and the English texts the client code asks for.

- entries whose French equals the English and reads as English, empty translations, placeholders that differ between
  the key and the value ({name} vs {nom}), French punctuation (a space before : ? !), "..." vs "…", a key given twice
  with two values (a later file wins);
- texts the code passes to _() / plural() / .log() that have no French entry (they would show in English);
- texts shown without _() (a literal string in a window title, a button, a label, a toast, a dialog).

Writes <out>/static_*.txt and prints a summary. Reads only; never writes the catalogue nor the code.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "astro_gpu" / "client"
CORE = ROOT / "astro_gpu" / "core"
FR = CLIENT / "locale" / "fr"
OUT = Path("F:/astro_essais/A_REGARDER/exploration/words_layout")
OUT.mkdir(parents=True, exist_ok=True)

# words that exist in English and practically never in French text (a French sentence holding one is suspect)
EN_ONLY = set("""the of and to is with from for this that it its be not no you your are was has have can cannot will
when then by at an as all any each only here there which what where how why into than these those been being would
should could does done made make makes takes gives needs first last after before yet still none off out up down open
close save run runs stop add remove delete new show hide block blocks output settings file files folder project
projects frame stars star background stack stacked stacking noise denoise color colour registration astrometry inspect
graph layer layers split combine merge starless alone left right top bottom kept keep set reset apply cancel browse
loading reading waiting failed skipped running pending outdated up-to-date computed compute minutes seconds hours
night nights selected select choose chosen please now again more less most least too very also just one two three
four five six seven eight nine ten zero many some other others same own both either neither such per via over under
between above below while until since because if else or nor but so""".split())
# the words a French interface keeps (the user's list: workflow, and the standard terms of stacking)
ALLOWED = set("""workflow workflows flats flat darks dark bias offsets offset lights light drizzle fwhm snr fits fit
tiff tif png jpeg jpg gpu cpu raw ha oiii sii sho hso hoo lrgb rgb adu iso gaia hfr ok l r g b ctrl shift del esc f5 f1
mo go ko gb mb tb ms px nm mm µm dec ra e""".split())
INTERNAL = re.compile(r"\b(legacy|v1|v2|render_simple|star_removal|debruitage|etoiles|astrometrie|deconv|"
                      r"traceback|none|null|true|false|nan|keyerror|valueerror|typeerror|attributeerror|runtimeerror|"
                      r"bench|banc|issue|rosette|heart|iris|m8|pilot|todo|fixme)\b|#\d+|"
                      r"\w~\d|[a-z]+_[a-z_]+|\.py\b|[A-Za-z]:[/\\]|/cache/|\.astroproj", re.I)
EXTERNAL = re.compile(r"\b(starnet|starnet\+\+|siril|pixinsight|photoshop|lightroom|graxpert|deepsnr|noisexterminator|"
                      r"blurxterminator|starxterminator|rc-astro|deepskystacker|dss|astro pixel processor|sequator|astap|"
                      r"cosmic clarity|topaz|gimp|affinity|nina|sharpcap|astrometry\.net|seti astro)\b", re.I)
PH = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?:[:!][^}]*)?\}")


def tokens(s: str) -> list[str]:
    s = re.sub(r"<[^>]+>", " ", s)          # rich text tags
    s = PH.sub(" ", s)
    return re.findall(r"[A-Za-zÀ-ÿœŒ][A-Za-zÀ-ÿœŒ'’\-]*", s)


def english_words(s: str) -> list[str]:
    out = []
    for t in tokens(s):
        w = t.lower().strip("'’-")
        if w in EN_ONLY and w not in ALLOWED:
            out.append(t)
    return out


def punct_problems(v: str) -> list[str]:
    """French typography: a space (or a no-break space) before : ; ? ! ; "..." instead of "…"."""
    out = []
    plain = PH.sub("X", re.sub(r"<[^>]+>", "", v))
    for m in re.finditer(r"(\S)([:;?!])(\s|$)", plain):
        before = m.group(1)
        if m.group(2) == ":" and before.isdigit():
            continue                           # 12:30, ratios
        if before in "?!:;":                   # ?! together
            continue
        out.append(f"no space before '{m.group(2)}' near «{plain[max(0, m.start() - 12):m.end() + 6]}»")
    if "..." in v:
        out.append("'...' (three dots) instead of '…'")
    if "  " in v.strip() and "  ·  " not in v and "  —  " not in v:
        out.append("double space")
    if v != v.strip() and v.strip():
        out.append("leading/trailing space")
    if re.search(r"\b[01] \w+s\b", plain) and not re.search(r"\b[01] (fois|pas|mois|jours?|images?)\b", plain):
        out.append("singular count with a plural noun?")
    return out


def load_catalogue():
    cat, where, dups = {}, {}, []
    files = sorted(FR.glob("*.json"))
    for f in files:
        raw = f.read_text(encoding="utf-8")
        pairs = json.loads(raw, object_pairs_hook=lambda kv: kv)
        seen = set()
        for k, v in pairs:
            if k in seen:
                dups.append((f.name, k, "twice in the same file"))
            seen.add(k)
            if k in cat and cat[k] != v:
                dups.append((f.name, k, f"also in {where[k]}: «{cat[k]}» vs «{v}» (the later file wins)"))
            cat[k] = v
            where[k] = f.name
    return cat, where, dups


def code_strings():
    """(text, file:line, kind) of the English texts the code asks to translate: _("..."), plural(n, "...", "..."),
    .log("...") and the fields of BlockRefused (core/workflow.py WHY_*, KIND_WORDS...)."""
    out = []
    for f in sorted(list(CLIENT.rglob("*.py")) + [CORE / "workflow.py", CORE / "project2.py"]):
        if "__pycache__" in f.parts:
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        rel = f.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
                lits = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                if name == "_" and node.args and isinstance(node.args[0], ast.Constant) and \
                        isinstance(node.args[0].value, str):
                    out.append((node.args[0].value, f"{rel}:{node.lineno}", "_"))
                elif name == "plural" and len(lits) >= 2:
                    for s in lits[:2]:
                        out.append((s, f"{rel}:{node.lineno}", "plural"))
                elif name == "log" and lits and isinstance(fn, ast.Attribute):
                    out.append((lits[0], f"{rel}:{node.lineno}", "log"))
    return out


# literal strings shown without _(): a window title, a button, a label, a message, a toast, a menu action
UNWRAPPED = re.compile(r"""(setWindowTitle|QPushButton|QLabel|label|addAction|addMenu|setText|setToolTip|
setPlaceholderText|toast|flash|ask_name|warning|information|critical|question|addTab|addItem|setTitle|
QCheckBox|QRadioButton|QGroupBox|show_toast)\(\s*(self,\s*|[a-z_]+,\s*)?(f?["'])([A-Z][^"']{2,})["']""", re.X)


def unwrapped_literals():
    out = []
    for f in sorted(CLIENT.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        rel = f.relative_to(ROOT).as_posix()
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            for m in UNWRAPPED.finditer(line):
                text = m.group(4)
                if re.fullmatch(r"[A-Z0-9 ·:%+\-/×°]+", text):          # units, codes
                    continue
                if any(w.lower() in EN_ONLY for w in tokens(text)) or len(tokens(text)) >= 1:
                    out.append((text, f"{rel}:{i}", line.strip()[:140]))
    return out


def main():
    cat, where, dups = load_catalogue()
    lines = defaultdict(list)
    for k, v in cat.items():
        if "#many" in k:
            base = k.split("#many")[0]
            if base not in cat:
                lines["plural form without its base entry"].append(f"[{where[k]}] {k!r}")
        if not isinstance(v, str):
            lines["value not text"].append(f"[{where[k]}] {k!r}: {v!r}")
            continue
        if not v.strip() and k.strip():
            lines["empty translation"].append(f"[{where[k]}] {k!r}")
            continue
        pk, pv = sorted(set(PH.findall(k))), sorted(set(PH.findall(v)))
        if pk != pv:
            lines["placeholders differ"].append(f"[{where[k]}] {k!r} -> {v!r}  (key {pk}, value {pv})")
        en_v = english_words(v)
        if v == k and en_v:
            lines["French = English, reads as English"].append(f"[{where[k]}] {k!r}  (words: {', '.join(en_v)})")
        elif en_v:
            lines["English words inside the French"].append(f"[{where[k]}] {v!r}  (words: {', '.join(en_v)})  <- {k!r}")
        if INTERNAL.search(PH.sub("", v)):
            lines["internal word in the French"].append(f"[{where[k]}] {v!r}  ({INTERNAL.search(PH.sub('', v)).group(0)})")
        if EXTERNAL.search(v) or EXTERNAL.search(k):
            lines["external software name"].append(f"[{where[k]}] {k!r} -> {v!r}")
        for pb in punct_problems(v):
            lines["French punctuation"].append(f"[{where[k]}] {pb}: {v!r}")
        if ("…" in k) != ("…" in v) and "..." not in v:
            lines["ellipsis in one only"].append(f"[{where[k]}] {k!r} -> {v!r}")
        if k.endswith(":") != v.rstrip().endswith(":") or (k.strip().endswith(".") != v.strip().endswith(".")):
            lines["end punctuation differs"].append(f"[{where[k]}] {k!r} -> {v!r}")
    for f, k, why in dups:
        lines["key given twice"].append(f"[{f}] {k!r}: {why}")
    # the texts of the code without a French entry
    used = code_strings()
    keys = set(cat)
    miss = defaultdict(list)
    for text, at, kind in used:
        if not text.strip() or text in keys:
            continue
        if not re.search(r"[A-Za-z]{2,}", PH.sub("", text)):
            continue
        miss[text].append(f"{at} ({kind})")
    for text, ats in sorted(miss.items()):
        lines["text of the code without a French entry"].append(f"{text!r}  at {', '.join(ats[:3])}")
    unused = keys - {t for t, _a, _k in used}
    lines["(info) catalogue entries the code does not name literally"].append(f"{len(unused)} entries")
    for text, at, line in unwrapped_literals():
        if text in keys:
            continue
        lines["literal text shown without _()?"].append(f"{text!r}  at {at}:  {line}")
    report = []
    for kind, items in lines.items():
        report.append(f"== {kind} ({len(items)})")
        report += ["  " + x for x in items]
    (OUT / "static_catalogue.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    for kind, items in lines.items():
        print(f"{len(items):5d}  {kind}")
    print("->", OUT / "static_catalogue.txt")


if __name__ == "__main__":
    sys.exit(main())
