"""Parameter schema of the stages: ``P`` (a parameter with its texts), ``StageParams``, ``Auto``, ``resolve``.

Every stage declares its parameters once, as a pydantic model whose fields carry the client's metadata:

* ``level``   - ``"basic"`` (shown by default) or ``"advanced"`` (shown in advanced mode)
* ``label``   - short human name
* ``help``    - what the parameter does, in plain words
* ``tip``     - when and how to change it, typical values
* ``unit``    - unit shown next to the value
* ``step``    - spin-box / slider step
* ``choices`` - allowed values for string parameters

Stages may also compute *automatic defaults* from the loaded data (``Stage.auto``). An automatic value always
comes with a one-line reason so the user knows why it was chosen; a user override always wins. ``resolve`` builds
the effective parameters; ``check_choice`` validates a text setting against its choices.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

BASIC = "basic"
ADVANCED = "advanced"


def P(default: Any, *, label: str, help: str, tip: str = "", level: str = BASIC, unit: str = "",
      ge: float | None = None, le: float | None = None, gt: float | None = None, lt: float | None = None,
      step: float | None = None, choices: list[str] | None = None, key_neutral: Any = None) -> Any:
    """Declare a stage parameter with its UI metadata. ``key_neutral``: a value at which the parameter is left out of
    the stage's key (a parameter added later: the results computed before it keep their keys)."""
    extra = {"level": level, "label": label, "help": help, "tip": tip, "unit": unit}
    if key_neutral is not None:
        extra["key_neutral"] = key_neutral
    if step is not None:
        extra["step"] = step
    if choices is not None:
        extra["choices"] = choices
    return Field(default, ge=ge, le=le, gt=gt, lt=lt, json_schema_extra=extra)


class StageParams(BaseModel):
    """Base class of every stage parameter model."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def describe(cls) -> list[dict]:
        """Field descriptions for the client (name, type, default, bounds and metadata)."""
        out = []
        for name, f in cls.model_fields.items():
            meta = dict(f.json_schema_extra or {})
            bounds = {}
            for m in f.metadata:
                for key in ("ge", "le", "gt", "lt"):
                    if hasattr(m, key):
                        bounds[key] = getattr(m, key)
            out.append({"name": name, "type": _type_name(f.annotation), "default": f.default, **bounds, **meta})
        return out


def _type_name(annotation: Any) -> str:
    text = getattr(annotation, "__name__", None) or str(annotation)
    return text.replace("typing.", "")


@dataclass(init=False)
class Auto:
    """A data-driven default value and the reason it was chosen.

    The reason is a *template* (English, ``str.format`` fields, a format spec for numbers) and its *arguments*, so that
    a client can translate the template and write the numbers its own way (decimal comma in French): see
    ``astro_gpu/texts.py``. ``Auto(value, "text")`` without arguments is a fixed text, used as it is. Every template
    has its French in ``client/locale/fr/engine.json`` (``tests/test_texts.py`` checks it).

        Auto(2.0, "small stars (FWHM {fwhm:.1f} px) and {n} frames", {"fwhm": f, "n": n})

    ``advice=True``: an advice only (the stack's "method"): ``resolve`` never applies it, the setting keeps its default
    or the user's value; the client shows it beside the choices (api.ParamInfo.advice).
    """

    value: Any
    template: str
    args: dict | None
    advice: bool

    def __init__(self, value: Any, template: str = "", args: dict | None = None, *, advice: bool = False):
        self.value = value
        self.template = template
        self.args = args
        self.advice = bool(advice)

    @property
    def reason(self) -> str:
        """The reason in English (the text stored in the manifests)."""
        return self.template if self.args is None else self.template.format(**self.args)

    def text(self, lang: str = "en") -> str:
        """The reason in a language ("fr": translated template, decimal comma)."""
        from .. import texts
        return texts.format_reason(self.template, self.args, lang)


def resolve(model: type[StageParams], auto: dict[str, Auto], overrides: dict[str, Any]) -> tuple[StageParams, dict]:
    """Build the effective parameters: declared default < automatic value < user override.

    Returns the validated model and, per field, where its value came from ("default", "auto" or "user"). An advice
    (``Auto.advice``) is never applied.
    """
    values, origin = {}, {}
    for name in model.model_fields:
        if name in overrides:
            values[name], origin[name] = overrides[name], "user"
        elif name in auto and not getattr(auto[name], "advice", False):
            values[name], origin[name] = auto[name].value, "auto"
        else:
            origin[name] = "default"
    unknown = set(overrides) - set(model.model_fields)
    if unknown:
        raise ValueError(f"unknown parameter(s) for {model.__name__}: {sorted(unknown)}")
    return model(**values), origin


def check_choice(name: str, value: str, choices) -> str:
    """A text setting's value, refused (ValueError naming the setting) when it is not one of its choices."""
    if value not in choices:
        raise ValueError(f"{name}: {value!r} is not one of {', '.join(choices)}")
    return value


def params_hash(p: StageParams) -> str:
    d = p.model_dump(mode="json")
    for name, f in type(p).model_fields.items():             # a parameter at its key-neutral value is left out
        extra = f.json_schema_extra or {}
        if isinstance(extra, dict) and "key_neutral" in extra and d.get(name) == extra["key_neutral"]:
            d.pop(name, None)
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
