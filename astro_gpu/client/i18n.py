"""Translations. English is the source language: every text shown is written in English in the code and goes
through _() where it is displayed (never at import time, so that a change of language takes effect when the views
are rebuilt). Other languages: locale/<code>/*.json, each a dictionary {English: translation}.

    from .i18n import _
    label(_("Stacking"))
    _("{n} frames kept of {total}").format(n=kept, total=n)

The French catalogue is split by area: base.json (the shared views), shell.json, graph.json, inspect.json, run.json,
post.json, engine.json and a few more. Files are read in name order, a later one wins.

Names typed by the user are never translated. Numbers shown with a decimal part go through ``number()``, which writes
a decimal comma in French.
"""
from __future__ import annotations

import json
import re
import string
from pathlib import Path

LOCALE = Path(__file__).parent / "locale"
LANGUAGES = {"en": "English", "fr": "Français"}
DECIMAL_COMMA = {"fr"}
_lang = "en"
_cat: dict[str, str] = {}
_missing: set[str] = set()


#: the feminine nouns of the interface, by language: the names of the blocks and of the things a message names (a name
#: is known by its first word: "Gradient 2", "Sortie Gradient"); any other name is masculine
FEMININE = {"fr": {"Calibration", "Sélection", "Astrométrie", "Couleur", "Session", "Image", "Images",
                   "Sortie", "Branche", "Pause", "Variante", "Copie"}}
#: the forms a translation may ask of a name in its fields: "{block} retiré{block:e}", "{block:un} seul{block:e}",
#: "Retirer {name:de}" (French: agreement, articles, elision before a vowel; "de « Ha »" stays as it is)
_NOUN_FORMS = ("e", "un", "Un", "de", "le", "Le", "du", "au")
_NOUN_FIELD = re.compile(r"\{\w*:(" + "|".join(_NOUN_FORMS) + r")\}")
_ELIDED = set("AEIOUYÂÀÉÈÊËÎÏÔÛÜaeiouyâàéèêëîïôûü")


def noun_form(name: str, form: str, lang: str | None = None) -> str:
    """A name in the form a sentence needs (French): its gender's ending ("e" or ""), "un"/"une", "de"/"d'" before it,
    "le"/"la"/"l'" before it, "du"/"de la"/"de l'", "au"/"à la"/"à l'"."""
    lang = lang or _lang
    head = name.split(" ")[0] if name else ""
    fem = head in FEMININE.get(lang, ())
    vowel = bool(name) and name[0] in _ELIDED
    if form == "e":
        return "e" if fem else ""
    if form in ("un", "Un"):
        w = "une" if fem else "un"
        return w.capitalize() if form == "Un" else w
    if form == "de":
        return ("d'" if vowel else "de ") + name
    if form in ("le", "Le"):
        w = "l'" if vowel else "la " if fem else "le "
        return (w[0].upper() + w[1:] if form == "Le" else w) + name
    if form == "du":
        return ("de l'" if vowel else "de la " if fem else "du ") + name
    if form == "au":
        return ("à l'" if vowel else "à la " if fem else "au ") + name
    return name


class _NounFormatter(string.Formatter):
    def format_field(self, value, spec):
        if spec in _NOUN_FORMS and isinstance(value, str):      # a number keeps "{x:e}" (an exponent)
            return noun_form(value, spec)
        return super().format_field(value, spec)


_NOUNS = _NounFormatter()


class Text(str):
    """A translation whose fields ask a name in a form ("{block:e}"): its format() gives that form (the English template
    has the plain field, so the code formats both the same way)."""
    def format(self, *args, **kwargs):
        return _NOUNS.vformat(str(self), args, kwargs)


def set_language(code: str):
    global _lang, _cat
    _lang = code if code in LANGUAGES else "en"
    _cat = {}
    _missing.clear()
    if _lang != "en":
        for f in sorted((LOCALE / _lang).glob("*.json")):
            _cat.update(json.loads(f.read_text(encoding="utf-8")))
        for k, v in _cat.items():
            if _NOUN_FIELD.search(v):
                _cat[k] = Text(v)


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
    has one form for both ("{n} outdated"), the catalogue gives the language's plural under "<text>#many" (and, when
    the plain entry is the plural one, its singular under "<text>#one": "{n} bias" → "1 offset", "3 offsets")."""
    single = (abs(n) < 2) if _lang == "fr" else (n == 1)
    if not single and one == many and _lang != "en" and many + "#many" in _cat:
        return _cat[many + "#many"].format(n=n, **args)     # one English form, two in the language ("obsolète(s)")
    if single and one == many and _lang != "en" and one + "#one" in _cat:
        return _cat[one + "#one"].format(n=n, **args)
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
    """A duration as the views show it: "≤ 1 min", "N min", "H h MM"."""
    m = int(round(seconds / 60))
    if m < 1:
        return "≤ 1 min"
    return f"{m // 60} h {m % 60:02d}" if m >= 60 else f"{m} min"


#: a program error as Python writes it ("TypeError: ...", a traceback): never shown as it is (GUI exploration 9.9)
_RAW_ERROR = re.compile(r"^\s*(Traceback|[A-Za-z_.]*(Error|Exception|Interrupt)\b)")
#: the usual troubles behind a program error, in plain words (searched in the error's text)
_PLAIN_ERRORS = (
    (("PermissionError", "Permission denied", "Access is denied"), "no permission to write or read there"),
    (("FileNotFoundError", "No such file", "cannot find the path"), "a file or folder is not found"),
    (("No space left", "ENOSPC", "not enough space on the disk"), "the disk is full"),
    (("MemoryError", "out of memory", "Unable to allocate"), "not enough memory"),
)


def is_raw_error(text: str) -> bool:
    """Whether a text is a program error as Python writes it, not a sentence for the user."""
    return bool(text) and bool(_RAW_ERROR.match(text))


def plain_error(text: str) -> str:
    """The reason of a failure in plain words, translated: a sentence the program wrote is kept (translated when the
    catalogue has it); a program error becomes the usual trouble it shows (no permission, a file not found, a full
    disk, not enough memory), else "an error of the program", the technical text being for the Log only."""
    text = (text or "").strip()
    if not is_raw_error(text):
        return maybe(text.splitlines()[0]) if text else _("an error of the program (the details are in the Log)")
    for marks, words in _PLAIN_ERRORS:
        if any(m in text for m in marks):
            return _(words)
    return _("an error of the program (the details are in the Log)")


def log_detail(model, text: str) -> None:
    """The technical text of a failure, kept in the Log (the history of the project) when a model is at hand."""
    log = getattr(model, "log", None)
    if callable(log) and is_raw_error(text or ""):
        lines = (text or "").strip().splitlines()
        log("Technical detail: {detail}", detail=(lines[-1] if lines[0].startswith("Traceback") else lines[0])[:300])
