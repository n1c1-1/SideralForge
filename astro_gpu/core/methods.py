"""The processing methods the user chooses between, in plain words: one decision per question, each option a set of
stage parameters. No method suits every instrument and object: they are compared side by side as variants (see
project.py) and the user keeps what looks best.

An option's ``params`` are {stage: {parameter: value}}; None puts a parameter back to its automatic value. An option
of kind "file" asks the user for a file, stored in its single parameter. ``evidence`` says what was seen on real data
(blind comparisons), so the choice is informed rather than guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    description: str
    params: dict[str, dict[str, Any]]
    evidence: str = ""
    kind: str = "fixed"          # fixed | file


@dataclass(frozen=True)
class Decision:
    key: str
    title: str
    question: str
    options: tuple[Option, ...]
    note: str = ""
    stages: tuple[str, ...] = field(default=())

    def touched(self) -> set[str]:
        return {st for o in self.options for st in o.params} | set(self.stages)


DECISIONS: tuple[Decision, ...] = (
    Decision(
        "star_layer", "Learned star removal",
        "Should a neural network learn, on this image, which light belongs to the stars?",
        (
            Option("none", "No network",
                   "Only the star model: every star drawn from the catalogue with its measured shape and subtracted.",
                   {"star_net": {"enabled": None}, "starless": {"learned_stars": None}}),
            Option("train", "Train on this image",
                   "A network learns this image's stars (about 30 min on the GPU), then removes them; the star model "
                   "stays available. The choice between the two is made at rendering ('Learned star removal').",
                   {"star_net": {"enabled": True}, "starless": {"learned_stars": None}},
                   evidence="Preferred in blind comparisons on the Horsehead, the Rosette (faithful), the Flaming Star "
                            "and the Iris: fewer shards and holes around the bright stars."),
            Option("file", "Use a trained network's layer",
                   "A star layer already computed for this image (a .npy file of the same size).",
                   {"starless": {"learned_stars": ""}}, kind="file",
                   evidence="The Horsehead uses a layer that keeps the core of NGC 2023 (the earlier one took it for "
                            "a star)."),
        ),
        note="Both removals are prepared when a network is used; 'Learned star removal' in Final adjustment switches "
             "between them instantly."),
    Decision(
        "glare", "Glare of the bright stars",
        "How is the faint glare that the optics spread around every star taken out of the nebula?",
        (
            Option("none", "Not modelled",
                   "The glare stays in the nebula (a pale fog around the brightest stars).",
                   {"stars": {"far_halo": False, "halo_bound": None}}),
            Option("halo", "Optical halo",
                   "One glare shape measured on the whole image, scaled by each star's brightness; the brightest stars "
                   "get their own amplitude and elongation.",
                   {"stars": {"far_halo": None, "halo_bound": None}},
                   evidence="Default. Kept on the Rosette and the Flaming Star after blind comparison."),
            Option("bounded", "Optical halo bounded by the data",
                   "The same, but a halo never exceeds the light the image shows around its star, per colour and "
                   "direction (a nebula rising toward the star had pulled it up: dark rings, dark lobes).",
                   {"stars": {"far_halo": None, "halo_bound": True}},
                   evidence="Won on the Horsehead (with its own network layer); lost on the Rosette (olive and green "
                            "discs in the cavity) and the Flaming Star (red smear at the bottom star)."),
        )),
    Decision(
        "star_shape", "Shape of the stars",
        "Is each star's shape (diffraction spikes, diamond) measured on the image?",
        (
            Option("measured", "Measured shape, spikes included",
                   "The shape every star shares is measured on the bright stars, spikes of a Newtonian included, and "
                   "removed with them.",
                   {"stars": {"angular_shape": None}}, evidence="Default."),
            Option("round", "Round model only",
                   "Only the round profile of the stars; their spikes stay in the image.",
                   {"stars": {"angular_shape": False}}),
        )),
    Decision(
        "faint_stars", "Faint stars",
        "Are the stars too faint to be measured one by one modelled from the catalogue?",
        (
            Option("model", "Modelled", "Faint catalogue stars are drawn and removed with the others.",
                   {"stars": {"faint_model": None}}, evidence="Default."),
            Option("leave", "Left in the image", "They stay in the nebula as a fine grain.",
                   {"stars": {"faint_model": False}}),
        )),
    Decision(
        "star_flat", "Flat field checked on the stars",
        "Are the stars' measured brightnesses used to correct the flat field?",
        (
            Option("on", "Corrected", "Vignetting and dust left by the flats are measured on the stars and corrected.",
                   {"background": {"star_flat": None}}, evidence="Default; it found errors of -35 to +16 % on the "
                                                                  "Horsehead flats."),
            Option("off", "As calibrated", "The flats are trusted as they are.",
                   {"background": {"star_flat": False}}),
        )),
    Decision(
        "ha_reference", "H-alpha reference of the sky",
        "Which all-sky survey anchors the faint H-alpha background?",
        (
            Option("auto", "Automatic", "The best survey covering the field.", {"surveys": {"survey": None}},
                   evidence="Default."),
            Option("nsns", "NSNS", "The northern H-alpha survey (finer).", {"surveys": {"survey": "nsns"}}),
            Option("finkbeiner", "Finkbeiner", "The all-sky H-alpha composite (coarser, whole sky).",
                   {"surveys": {"survey": "finkbeiner"}}),
            Option("none", "None", "The sky level is taken from the image alone.", {"surveys": {"survey": "none"}}),
        )),
    Decision(
        "star_guard", "Repairs around the stars",
        "Are the small dark or bright residues the star model leaves around stars repaired?",
        (
            Option("on", "Repaired", "Residues well outside the noise are replaced by their surroundings.",
                   {"render_prep": {"star_guard": None}, "render_prep_learned": {"star_guard": None}},
                   evidence="Default."),
            Option("off", "As removed", "Nothing is repaired.",
                   {"render_prep": {"star_guard": False}, "render_prep_learned": {"star_guard": False}}),
        )),
    Decision(
        "pattern", "Fixed pattern of the sensor",
        "Is a fixed pattern measured on the frames and removed before stacking?",
        (
            Option("off", "Not removed", "The frames are stacked as calibrated.", {"stack": {"pattern_removal": None}},
                   evidence="Default: the removal drew diagonal streaks on the Rosette."),
            Option("on", "Removed", "A pattern common to all frames is estimated and subtracted.",
                   {"stack": {"pattern_removal": True}}),
        )),
    Decision(
        "lateral", "Colour fringes of the optics",
        "Is the colour shift of the optics (red and blue images slightly offset) corrected?",
        (
            Option("on", "Corrected", "The offsets are measured on the stars across the field and corrected.",
                   {"optics": {"lateral": None}}, evidence="Default."),
            Option("off", "Not corrected", "The colour planes are used as stacked.", {"optics": {"lateral": False}}),
        )),
)

BY_KEY = {d.key: d for d in DECISIONS}


def option_params(decision: Decision, option: Option, file_value: str = "") -> dict[str, dict[str, Any]]:
    """The parameter changes of an option (a file option gets its file)."""
    out = {st: dict(kv) for st, kv in option.params.items()}
    if option.kind == "file":
        (st, kv), = out.items()
        (k,) = kv
        kv[k] = file_value
    return out


def current_option(decision: Decision, overrides) -> Option | None:
    """The option the processing matches (``overrides(stage)`` -> the user's values, as Project.overrides), or None
    when its parameters were set by hand to something else."""
    for o in decision.options:
        ok = True
        for st, kv in o.params.items():
            ov = overrides(st)
            for k, v in kv.items():
                if o.kind == "file":
                    ok &= bool(ov.get(k))
                elif v is None:
                    ok &= k not in ov
                else:
                    ok &= ov.get(k) == v
        if ok:
            return o
    return None


def merged(base: dict[str, dict[str, Any]], change: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Variant parameters ``base`` with ``change`` on top."""
    out = {st: dict(kv) for st, kv in base.items()}
    for st, kv in change.items():
        out.setdefault(st, {}).update(kv)
    return out


def downstream(stages: set[str], graph: dict[str, tuple[str, ...]]) -> list[str]:
    """The stages to compute again when ``stages`` change: them and every stage after them, in pipeline order."""
    out = set(stages)
    grew = True
    while grew:
        grew = False
        for st, deps in graph.items():
            if st not in out and any(d in out for d in deps):
                out.add(st)
                grew = True
    return [s for s in graph if s in out]
