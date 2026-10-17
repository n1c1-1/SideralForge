"""Views: named renderings of a project - the processing (main or a variant) and rendering settings. No single
rendering suits every part of an image; the views give at least one good rendering per zone, to compare and later
combine. A project's first views are its delivered renderings and, for each, the alternatives that cost no
computing (the glare drawn with the stars, the spikes drawn, the other star removal).
"""
from __future__ import annotations

import json

from .settings import RenderSettings


def _prep_has(project, stage: str, name: str) -> bool:
    return (project.stage_dir(stage) / name).exists()


def _learned_ready(project) -> bool:
    d = project.stage_dir("render_prep_learned")
    return (d / "N.npy").exists() and (d / "prep.json").exists()


def base_views(project) -> dict[str, dict]:
    """The delivered renderings (results/<name>/<name>_settings.json) of the processing they were made from."""
    out: dict[str, dict] = {}
    for f in sorted(project.results.glob("*/*_settings.json")):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        var = meta.get("variant", "main")
        var = "" if var == "main" else var
        if var and var not in project.config.variants:
            continue
        name = f.parent.name.replace("_", " ").capitalize()
        settings = RenderSettings.from_saved(meta.get("settings", {})).model_dump(exclude_defaults=True)
        if meta.get("preparation") == "render_prep":        # rendered by the star model, whatever the setting said
            settings["learned_stars"] = False
        out[name] = {"variant": var, "settings": settings, "created": meta.get("date", ""),
                     "note": f"the delivered rendering (results/{f.parent.name})"}
    return out


def default_views(project) -> dict[str, dict]:
    """The delivered renderings and, for each, the alternatives that need no computing."""
    out: dict[str, dict] = {}
    glare = _prep_has(project, "render_prep", "Shalo.npy")
    spikes = _prep_has(project, "render_prep", "Ssp.npy")
    learned = _learned_ready(project)
    for name, v in base_views(project).items():
        out[name] = v
        s = v["settings"]
        alts = []
        if glare and RenderSettings(**s).glare_nebula > 0:
            alts.append(("glare with the stars", {"glare_nebula": 0.0},
                         "the bright stars' glare drawn with the stars (compact) rather than with the nebula"))
        if spikes and RenderSettings(**s).spike_photo > 0:
            alts.append(("spikes drawn", {"spike_photo": 0.0}, "the diffraction spikes drawn instead of photographed"))
        if learned:
            if RenderSettings(**s).learned_stars:
                alts.append(("star model", {"learned_stars": False}, "the stars removed by the star model"))
            else:
                alts.append(("learned removal", {"learned_stars": True}, "the stars removed by the learned network"))
        for suffix, change, note in alts:
            out[f"{name} - {suffix}"] = {"variant": v["variant"], "settings": {**s, **change},
                                         "created": v.get("created", ""), "note": f"{name} with {note}"}
    return out


def ensure_default_views(project) -> bool:
    """Give a project its first views when it has none; True when some were created (the project is saved)."""
    views = project.config.render.setdefault("views", {})
    if views:
        return False
    views.update(default_views(project))
    if views:
        project.save()
    return bool(views)
