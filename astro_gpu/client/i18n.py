"""Translations. English is the source language: every text shown is written in English in the code and goes
through _() where it is displayed (never at import time, so that a change of language takes effect when the views
are rebuilt). Other languages: locale/<code>/*.json, each a dictionary {English: translation}.

    from .i18n import _
    label(_("Stacking"))
    _("{n} frames kept of {total}").format(n=kept, total=n)

locale/fr/base.json holds the translations that came with the views; each package adds its own file (shell.json,
graph.json, inspect.json, run.json, post.json, engine.json). Files are read in name order, a later one wins.

Names typed by the user are never translated. Numbers shown with a decimal part go through ``number()``, which writes
a decimal comma in French.
"""
from __future__ import annotations

import json
from pathlib import Path

LOCALE = Path(__file__).parent / "locale"
LANGUAGES = {"en": "English", "fr": "Français"}
DECIMAL_COMMA = {"fr"}
_lang = "en"
_cat: dict[str, str] = {}
_missing: set[str] = set()


def set_language(code: str):
    global _lang, _cat
    _lang = code if code in LANGUAGES else "en"
    _cat = {}
    _missing.clear()
    if _lang != "en":
        for f in sorted((LOCALE / _lang).glob("*.json")):
            _cat.update(json.loads(f.read_text(encoding="utf-8")))


def language() -> str:
    return _lang


def _(s: str) -> str:
    if _lang == "en" or not s:
        return s
    t = _cat.get(s)
    if t is None:
        _missing.add(s)
        return s
    return t


def maybe(s: str) -> str:
    """A text the program made that may have no translation (a file name, an image layer's name): translated when the
    catalogue has it, else as it is (not counted as missing)."""
    return _cat.get(s, s) if _lang != "en" and s else s


def missing() -> set[str]:
    """The texts asked for since the last change of language that have no translation (for the tests)."""
    return set(_missing)


def plural(n: int, one: str, many: str, **args) -> str:
    """A count with its noun in the right form, translated: ``plural(n, "{n} night", "{n} nights")``. English takes the
    singular for 1 only; French for 0 and 1 ("0 nuit", "1 nuit", "2 nuits"). Other fields go in ``args``. When English
    has one form for both ("{n} outdated"), the catalogue gives the language's plural under "<text>#many"."""
    single = (abs(n) < 2) if _lang == "fr" else (n == 1)
    if not single and one == many and _lang != "en" and many + "#many" in _cat:
        return _cat[many + "#many"].format(n=n, **args)     # one English form, two in the language ("obsolète(s)")
    return _(one if single else many).format(n=n, **args)


def number(x: float, digits: int = 1) -> str:
    """A number with a fixed count of decimals, with a decimal comma in French ("2.5" → "2,5")."""
    t = f"{x:.{digits}f}"
    return t.replace(".", ",") if _lang in DECIMAL_COMMA else t


def all_workflows(n: int) -> str:
    """A change or a block for every workflow: "both workflows", "all 3 workflows" ("les 2 workflows"...)."""
    return _("both workflows") if n == 2 else _("all {n} workflows").format(n=n)


def hours_text(hours: float) -> str:
    """An exposure time given in hours: "10 min" under an hour, "3.2 h" above (a decimal comma in French)."""
    if hours < 1:
        return f"{max(1, round(hours * 60)) if hours > 0 else 0} min"
    return number(hours, 1) + " h"


def hm(seconds: float) -> str:
    """A duration as the views show it (SPEC §7): "≤ 1 min", "N min", "H h MM"."""
    m = int(round(seconds / 60))
    if m < 1:
        return "≤ 1 min"
    return f"{m // 60} h {m % 60:02d}" if m >= 60 else f"{m} min"
