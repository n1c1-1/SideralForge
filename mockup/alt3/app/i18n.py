"""Translations. English is the source language: every text shown is written in English in the code and goes
through _() where it is displayed (never at import time, so that a change of language takes effect when the views
are rebuilt). Other languages: locale/<code>/*.json, each a dictionary {English: translation}.

    from i18n import _
    label(_("Stacking"))
    _("{n} frames kept of {total}").format(n=kept, total=n)
"""
from __future__ import annotations

import json
from pathlib import Path

LOCALE = Path(__file__).parent / "locale"
LANGUAGES = {"en": "English", "fr": "Français"}
_lang = "en"
_cat: dict[str, str] = {}


def set_language(code: str):
    global _lang, _cat
    _lang = code if code in LANGUAGES else "en"
    _cat = {}
    if _lang != "en":
        for f in sorted((LOCALE / _lang).glob("*.json")):
            _cat.update(json.loads(f.read_text(encoding="utf-8")))


def language() -> str:
    return _lang


def _(s: str) -> str:
    if _lang == "en" or not s:
        return s
    return _cat.get(s, s)
