"""'Astro: Methods': the processing choices of the heavy client's Methods tab (astro_gpu/core/methods.py) as one node.
Each question is a list (the options in plain words; hover for what each does and what was seen on real data);
'as main' keeps what the main processing does. The choices that differ from main are computed as a variant, next to
the main processing, which is never changed here.
"""
from __future__ import annotations

from comfy_api.latest import io

from . import bridge

ASTRO = io.Custom("ASTRO")
AS_MAIN = "as main"
NOTE = "ComfyUI, Astro: Methods: "     # marks the variants this node made (and may change)


def _decisions():
    from astro_gpu.core.methods import DECISIONS
    return DECISIONS


def _tooltip(d) -> str:
    lines = [d.question]
    for o in d.options:
        lines.append(f"- {o.label}: {o.description}" + (f" Seen: {o.evidence}" if o.evidence else ""))
    if d.note:
        lines.append(d.note)
    return "\n".join(lines)


def _changes(choices: dict[str, str], files: dict[str, str]) -> tuple[dict, list[str]]:
    """The stage parameters of the chosen options (not 'as main'), and their names in plain words."""
    from astro_gpu.core.methods import merged, option_params
    params: dict = {}
    names = []
    for d in _decisions():
        label = choices.get(d.key, AS_MAIN)
        if label == AS_MAIN:
            continue
        o = next(o for o in d.options if o.label == label)
        if o.kind == "file" and not files.get(d.key, "").strip():
            raise ValueError(f"'{d.title}: {o.label}' needs a file ({d.key}_file)")
        params = merged(params, option_params(d, o, files.get(d.key, "").strip().strip('"')))
        names.append(f"{d.title}: {o.label}")
    return params, names


class AstroMethods(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs = [ASTRO.Input("astro", tooltip="The project (from Astro: Project)"),
                  io.String.Input("variant", default="", tooltip="Name of the variant that computes these choices "
                                                                 "(empty: named after the choices)")]
        for d in _decisions():
            inputs.append(io.Combo.Input(d.key, options=[AS_MAIN] + [o.label for o in d.options], default=AS_MAIN,
                                         display_name=d.title, tooltip=_tooltip(d)))
            if any(o.kind == "file" for o in d.options):
                inputs.append(io.String.Input(f"{d.key}_file", default="", optional=True, advanced=True,
                                              display_name=f"{d.title}: file",
                                              tooltip="The file the option asks for (a .npy layer...)"))
        return io.Schema(
            node_id="AstroMethods", display_name="Astro: Methods", category="astro/old - projects",
            description="The processing choices in plain words, as in the heavy client's Methods tab: one list per "
                        "question ('as main' = what the main processing does; hover a list for what each option "
                        "does and what was seen on real data). The choices are computed as a variant next to the "
                        "main processing (only the stages they change, and the ones after); connect a render node "
                        "after it to compare with the main processing.",
            inputs=inputs, outputs=[ASTRO.Output(display_name="astro")])

    @classmethod
    def execute(cls, astro, variant, **kw) -> io.NodeOutput:
        from astro_gpu.core.project import Project
        choices = {d.key: kw.get(d.key, AS_MAIN) for d in _decisions()}
        files = {k[:-5]: v for k, v in kw.items() if k.endswith("_file") and isinstance(v, str)}
        params, names = _changes(choices, files)
        if not params:                                    # nothing differs: the main processing
            return io.NodeOutput(bridge.astro_value(astro["root"], ""))
        name = variant.strip() or ("Methods - " + ", ".join(n.split(": ", 1)[1] for n in names))[:80]
        note = NOTE + "; ".join(names)
        p = Project(astro["root"])
        if name not in p.config.variants:
            p.create_variant(name, params, note=note)
        else:
            v = p.config.variants[name]
            if not str(v.get("note", "")).startswith(NOTE):
                raise ValueError(f"the variant {name!r} was not made by an Astro: Methods node: give another name")
            # as create_variant: a change back to automatic of a parameter main leaves automatic is no change
            want = {st: {k: x for k, x in kv.items() if not (x is None and k not in p.config.params.get(st, {}))}
                    for st, kv in params.items()}
            want = {st: kv for st, kv in want.items() if kv}
            if v.get("params") != want:
                v["params"] = want                        # the stages it changes are computed again
                v["note"] = note
                p.save()
        return io.NodeOutput(bridge.astro_value(astro["root"], name))
