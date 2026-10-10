"""Parameter schema shared by the processing engine and the GUI.

Every stage declares its parameters once, as a pydantic model whose fields carry UI metadata:

* ``level``   - ``"basic"`` (shown by default) or ``"advanced"`` (shown in advanced mode)
* ``label``   - short human name
* ``help``    - what the parameter does, in plain words
* ``tip``     - when and how to change it, typical values
* ``unit``    - unit shown next to the value
* ``step``    - spin-box / slider step
* ``choices`` - allowed values for string parameters

Stages may also compute *automatic defaults* from the loaded data (``Stage.auto``). An automatic value always
comes with a one-line reason so the user knows why it was chosen; a user override always wins.
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
      step: float | None = None, choices: list[str] | None = None) -> Any:
    """Declare a stage parameter with its UI metadata."""
    extra = {"level": level, "label": label, "help": help, "tip": tip, "unit": unit}
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
        """Field descriptions for the GUI (name, type, default, bounds and metadata)."""
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


@dataclass
class Auto:
    """A data-driven default value and the reason it was chosen."""

    value: Any
    reason: str


def resolve(model: type[StageParams], auto: dict[str, Auto], overrides: dict[str, Any]) -> tuple[StageParams, dict]:
    """Build the effective parameters: declared default < automatic value < user override.

    Returns the validated model and, per field, where its value came from ("default", "auto" or "user").
    """
    values, origin = {}, {}
    for name in model.model_fields:
        if name in overrides:
            values[name], origin[name] = overrides[name], "user"
        elif name in auto:
            values[name], origin[name] = auto[name].value, "auto"
        else:
            origin[name] = "default"
    unknown = set(overrides) - set(model.model_fields)
    if unknown:
        raise ValueError(f"unknown parameter(s) for {model.__name__}: {sorted(unknown)}")
    return model(**values), origin


def params_hash(p: StageParams) -> str:
    blob = json.dumps(p.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
