"""The settings of each step, sorted by use rather than by engine module, and how one setting is shown.

Two kinds of settings:
* "engine": an existing setting of the engine (101 in all), with its value on Heart, its suggestion and reason;
* "new": a setting the engine does not have yet, shown marked "to build in the engine" so that the mockup doubles as
  the list of the engine work. Its value is kept like the others, so that changing it marks the step to recompute.

Levels: "normal" (always shown), "expert" (expert mode), "system" (speed, memory, reproducibility: kept apart).

Texts are in English and go through _() when shown: resolve() returns the label, help, tip, reason, unit and choices
already translated; group titles are translated by step_settings().
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from i18n import _, language
from model import TRUNK, PathDef, Session
from widgets import ACCENT, WARN, label

NEW = "#c9b37e"                     # colour of the settings to create in the engine


def E(stage, name, level="normal", label=None, help=None, unit=None):
    return dict(kind="engine", stage=stage, name=name, level=level, label=label, help=help, unit=unit)


def N(stage, key, label, typ, default, help, level="normal", choices=None, lo=None, hi=None, step=None, unit="",
      reason=""):
    return dict(kind="new", stage=stage, name="new_" + key, level=level, label=label, help=help, type=typ,
                default=default, choices=choices, min=lo, max=hi, step=step, unit=unit, reason=reason)


AS_TODAY = "as the engine does today"
NO_ZONE = "none"                    # value of a "zone" setting with nothing drawn
CATALOG = {
    "calibration": [
        ("Masters", [
            N("calibrate", "darks_combine", "Dark combination", "choice", "Average with rejection",
              "How the darks are merged into a master.",
              choices=["Average with rejection", "Median", "Average"], reason=AS_TODAY),
            N("calibrate", "dark_scaling", "Dark scaling", "bool", False,
              "Scales the thermal signal of the master to each frame: useful when the temperature changed during the "
              "night or the darks were taken at another temperature."),
        ]),
        ("Cosmetic correction", [
            E("calibrate", "hot_sigma", help="A pixel hotter than this threshold (in noise sigmas) is excluded; "
                                             "drizzle fills the hole with the other frames."),
            E("calibrate", "hot_from_lights", help="Also looks for hot pixels in the lights themselves, not only in "
                                                   "the darks."),
            N("calibrate", "cold_pixels", "Correct cold pixels too", "bool", False,
              "Excludes pixels that are too dark (dead) the same way as hot pixels."),
        ]),
        ("Flats", [
            E("calibrate", "flat_defect", help="A flat defect deeper than this threshold (dust, scratch) is "
                                               "corrected."),
            E("calibrate", "flat_scale_px", "expert", help="Keeps only the large scales of the flat (the "
                                                           "vignetting)."),
        ]),
        ("Sensor", [
            E("calibrate", "saturation", "expert", help="Fraction of the white level above which a pixel counts as "
                                                        "saturated."),
            E("inventory", "max_lights", "expert", help="Uses only the first N frames; 0 = all."),
        ]),
        ("System", [E("calibrate", "workers", "system")]),
    ],
    "alignement": [
        ("Reference", [
            N("register", "reference", "Reference frame", "choice", "Automatic: the sharpest",
              "The frame all the others are aligned on. Picked in the sources: the frame shown in the viewer.",
              choices=["Automatic: the sharpest", "Picked in the sources"], reason=AS_TODAY),
        ]),
        ("Detected stars", [
            E("register", "detect_sigma", help="A star must exceed this threshold (in background noise sigmas) to "
                                               "be used for registration."),
            E("register", "max_stars", "expert", help="The brightest ones, at most this number per frame."),
        ]),
        ("Distortion", [
            N("register", "transform", "Transformation model", "choice", "Automatic",
              "How a frame is warped to match the reference. The polynomials follow the distortion of a wide-field "
              "lens.",
              choices=["Automatic", "Rotation and translation", "Homography", "Polynomial, degree 2",
                       "Polynomial, degree 3"], reason=AS_TODAY),
        ]),
        ("System", [E("register", "workers", "system")]),
    ],
    "selection": [
        ("Other criteria", [
            N("stack", "max_background", "Maximum background", "float", 0.0,
              "Excludes frames with a too bright background (Moon, dawn, headlights), as a multiple of the median "
              "background; 0 = no threshold.", lo=0.0, hi=5.0, step=0.1, unit="× median"),
            N("stack", "min_roundness", "Minimum star roundness", "float", 0.0,
              "Excludes frames with elongated stars (wind, tracking); 1 = perfectly round, 0 = no threshold.",
              lo=0.0, hi=1.0, step=0.05),
            N("stack", "keep_best", "Keep the best", "int", 100,
              "After the thresholds, keeps only this percentage of the best frames.", lo=10, hi=100, step=5,
              unit="% of frames"),
        ]),
        ("Weighting", [
            N("stack", "weighting", "Frame weights", "choice", "Transparency and star size",
              "How much each frame counts in the stack.",
              choices=["Transparency and star size", "Signal-to-noise ratio", "Equal for all"],
              reason=AS_TODAY),
        ]),
    ],
    "empilement": [
        ("Resolution", [
            E("stack", "scale"),
            E("stack", "pixfrac", help="The size of the drop each raw pixel deposits on the fine grid."),
        ]),
        ("Pixel rejection", [
            N("stack", "rejection", "Rejection method", "choice", "Automatic: around the median",
              "How outlier pixels (satellites, planes, cosmic rays) are removed.",
              choices=["Automatic: around the median", "Sigma clipping", "Winsorized sigma clipping",
                       "Linear fit clipping", "Generalized ESD"], reason=AS_TODAY),
            E("stack", "kappa", help="A pixel farther from the median than this threshold (in noise sigmas) is "
                                     "rejected."),
            N("stack", "split_thresholds", "Separate low and high thresholds", "bool", False,
              "By default the same threshold applies to too bright and too dark pixels. Separate, bright trails can "
              "be rejected harder."),
            E("stack", "core_tolerance", "expert", help="Margin left at the core of bright stars, which varies from "
                                                        "frame to frame with the seeing."),
        ]),
        ("Normalization", [
            N("stack", "normalization", "Frame normalization", "choice", "Automatic: offset and scale",
              "Brings all the frames to the same background level and scale before combining them, despite a "
              "changing Moon or haze.",
              choices=["Automatic: offset and scale", "Offset only", "None"], reason=AS_TODAY),
        ]),
        ("Sensor", [
            E("stack", "gain", "expert", help="Electrons per digital unit: used by the noise model."),
            E("stack", "read_noise", "expert", help="Read noise of the sensor, in ADU."),
        ]),
        ("Fixed sensor pattern", [
            E("stack", "pattern_removal", "expert", help="Removes a fixed pattern estimated on the frames (bands, "
                                                         "streaks)."),
            E("stack", "pattern_frames", "expert", help="Number of frames used to estimate this pattern."),
        ]),
    ],
    "demi_piles": [
        ("Half-stacks", [
            E("stack", "half_block", "expert", help="The frames are split between the two half-stacks in blocks of "
                                                    "this number, so that each covers the whole night."),
        ]),
    ],
    "astrometrie": [
        ("Initial position", [
            E("astrometry", "target"), E("astrometry", "ra"), E("astrometry", "dec"), E("astrometry", "focal_mm"),
        ]),
        ("Plate solving", [
            E("astrometry", "sip_order", "expert", help="Degree of the polynomial that describes the optical "
                                                        "distortion."),
            E("astrometry", "blind", "expert", help="Finds the field without knowing the center or the object."),
            E("astrometry", "search_offsets", "expert", help="How far to search around the given center, as a "
                                                             "fraction of the field."),
            E("astrometry", "pixel_um", "expert", help="Pixel size of the sensor; empty = read from the files."),
            E("astrometry", "scale", "expert", help="Arcseconds per pixel; empty = computed."),
        ]),
        ("System", [E("catalogs", "cell_deg", "system", help="Size of the sky tiles requested from the Gaia "
                                                             "catalogue.")]),
    ],
    "couleur": [
        ("White reference", [
            N("color", "white_ref", "White reference", "choice", "The Sun",
              "What should appear white.",
              choices=["The Sun", "Average of the field stars", "Average spiral galaxy", "Custom temperature"],
              reason=AS_TODAY),
            E("color", "white_temp", label="White temperature",
              help="The temperature of this reference; 5778 K is that of the Sun."),
            N("color", "neutral_zone", "Background neutralization area", "choice", "Automatic: nebula-free",
              "Where the sky background is made neutral grey.",
              choices=["Automatic: nebula-free", "Drawn on the image"], reason=AS_TODAY),
        ]),
        ("Color measurement", [
            E("color", "aperture", "expert", help="Radius of the light measurement of each star."),
            E("color", "camera_profile", "expert", help="Uses the measured response of the sensor rather than a "
                                                        "typical response."),
            E("color", "synthetic_weight", "expert", help="Weight of the reconstructed Gaia spectra relative to the "
                                                          "measured spectra."),
            E("color", "profile_tolerance", "expert", help="Allowed deviation between the measured profile and the "
                                                           "typical profile."),
            E("color", "hii_weight", "expert", help="Weight given to the H-alpha line in the color fit."),
        ]),
        ("Color stars (Gaia)", [
            E("catalogs", "xp_gmin", "expert", help="Magnitude of the brightest star used for the color."),
            E("catalogs", "xp_gmax", "expert", help="Magnitude of the faintest."),
            E("catalogs", "xp_max", "expert", help="Maximum number of stars used."),
        ]),
    ],
    "debruitage": [
        ("Denoise", [
            N("denoise", "strength", "Strength", "int", 100,
              "Blend with the original stack: 100 % keeps all the denoising, 70 % leaves some natural grain.",
              lo=0, hi=100, step=5, unit="%", reason=AS_TODAY),
            E("denoise", "iterations"),
        ]),
        ("Training", [
            E("denoise", "patch", "expert", help="Size of the image tiles the network learns on."),
            E("denoise", "learning_rate", "expert", help="Learning step: too large, it diverges; too small, it does "
                                                         "not progress."),
            E("denoise", "reuse_model", "expert", help="Starts from an already trained network (file) instead of "
                                                       "from scratch."),
        ]),
        ("System", [E("denoise", "batch", "system"), E("denoise", "seed", "system")]),
    ],
    "etoiles": [
        ("Which stars", [
            E("catalogs", "g_deep", label="Limiting magnitude (G)",
              help="The larger the magnitude, the more stars are measured one by one, and the longer it takes.",
              unit="G"),
            E("stars", "faint_model"),
            E("catalogs", "g_faint", "expert", help="Stars fainter than the limit of the measured stars, down to "
                                                    "this one, are modelled from the catalogue.", unit="G"),
            E("catalogs", "max_faint", "expert", help="At most this number of modelled stars."),
        ]),
        ("Star shape", [
            E("stars", "local_psf"), E("stars", "angular_shape"), E("stars", "far_halo"), E("optics", "lateral"),
            E("stars", "halo_bound", "expert", help="The far halo cannot exceed what the data show."),
            E("optics", "lateral_grid", "expert", help="Resolution of the map of the color fringes."),
            E("optics", "psf_iterations", "expert", help="Iterations of the star shape fit."),
            E("optics", "psf_stars_per_cell", "expert", help="Stars used per area to measure the shape."),
        ]),
        ("Fit", [
            E("stars", "color_prior", "expert", help="How closely the color of each star must follow Gaia."),
            E("stars", "refine_passes", "expert", help="Refinement passes of the star positions."),
            E("stars", "solver_passes", "expert", help="Round trips between the background estimate and the star "
                                                       "flux estimate."),
            E("stars", "cg_iterations", "expert", help="Solver iterations at each pass."),
        ]),
    ],
    "appris": [
        ("Training", [
            E("star_net", "steps"),
            E("star_net", "glow", help="Keeps the nebula around a star instead of taking it for the star's light."),
        ]),
        ("System", [E("star_net", "device", "system", help="Which GPU trains the network."),
                    E("star_net", "batch", "system"), E("star_net", "seed", "system")]),
    ],
    "sans_etoiles": [
        ("Denoise", [
            E("starless", "wiener_k"),
            E("starless", "chroma_min_scale", "expert", help="The color is not denoised below this scale."),
        ]),
        ("Star residuals", [
            E("starless", "residual_k", help="Strength of the removal of what is left of the stars after "
                                             "subtraction."),
            E("starless", "residual_scales", "expert", help="Number of scales where the residuals are searched."),
            E("starless", "mask_k", "expert", help="Above this threshold, a star core is judged unreliable and "
                                                   "rebuilt."),
            E("starless", "halo_gmax", "expert", help="Stars brighter than this magnitude have their optical halo "
                                                      "removed."),
            E("starless", "lift_gmax", "expert", help="The dips left under stars brighter than this magnitude are "
                                                      "repaired."),
        ]),
        ("Nebula", [
            E("starless", "nebula_gate", help="Protects the nebula from the corrections made around the stars; 1 = "
                                              "maximum protection."),
        ]),
        ("Sharpening", [
            E("starless", "sharpen"),
            E("starless", "sharpen_steps", "expert", help="Training steps of the sharpening network."),
        ]),
    ],
    "fond:halpha": [
        ("Reference survey", [
            E("surveys", "survey"),
            E("surveys", "min_coverage", "expert", help="Share of the field the survey must cover to be used."),
        ]),
        ("Sky gradient", [
            E("background", "sky_order"), E("background", "star_flat"),
            E("background", "bin_px", "expert", help="Size of the cells where the background is measured."),
            E("background", "large_scale_px", "expert", help="Scale beyond which the background is corrected."),
            E("background", "object_clip", "expert", help="Areas brighter than this threshold are not used to "
                                                          "measure the background."),
        ]),
        ("Samples", [
            N("background", "exclusions", "Areas to exclude", "zone", NO_ZONE,
              "The drawn areas are not used to measure the background: faint nebula, galaxy, reflection."),
            N("background", "show_samples", "Show sample points", "bool", False,
              "Shows on the image the points where the background is measured, and lets them be moved or removed."),
        ]),
    ],
    "fond:simple": [
        ("Sky gradient", [
            N("fond_simple", "degree", "Surface degree", "int", 2,
              "1: a tilted plane; 2 or 3: a more flexible surface, for wide fields.", lo=1, hi=5, step=1),
            N("fond_simple", "samples", "Samples per side", "int", 16,
              "Number of background measurement points along each side of the image.", lo=4, hi=64, step=2),
        ]),
        ("Samples", [
            N("fond_simple", "exclusions", "Areas to exclude", "zone", NO_ZONE,
              "The drawn areas are not used to measure the background: faint nebula, galaxy, reflection."),
            N("fond_simple", "show_samples", "Show sample points", "bool", False,
              "Shows on the image the points where the background is measured, and lets them be moved or removed."),
        ]),
    ],
    "post": [
        ("Stars in the render", [
            E("@render", "star_guard", label="Protect against star artifacts",
              help="Finds the redrawn stars that do not match the image and corrects them."),
            E("@render", "guard_sigma", "expert", label="Guard tolerance"),
            E("@render", "real_g", "expert", label="Spikes measured down to magnitude", unit="G"),
            E("@render", "spike_snr", "expert", label="Spikes kept from a signal-to-noise of"),
        ]),
        ("Image edges", [
            E("@render", "cover_min", "expert", label="Minimum edge coverage",
              help="Edges covered by less than this share of the frames are cropped."),
            E("@render", "edge_frac", "expert", label="Edge cleanliness"),
            E("@render", "min_exposure", "expert", label="Minimum effective exposure"),
        ]),
        ("Detail and color", [
            E("@render", "fine_sigma", "expert", label="Fine detail scale", unit="px"),
            E("@render", "cont_smooth", "expert", label="Background smoothing under the star light", unit="px"),
            E("@render", "resid_smooth", "expert", label="Residual color smoothing", unit="px"),
            E("@render", "n2n_base", "expert", label="Learned denoising in the base"),
            E("@render", "n2n_scale", "expert", label="Learned denoising up to the scale", unit="px"),
            E("@render", "diffuse_scale", "expert", label="Diffuse glow scale"),
            E("@render", "oiii_scale", "expert", label="OIII map base smoothing", unit="px"),
        ]),
    ],
}
# the 32 usage tips of the engine
TIPS = {
    ("astrometry", "target"): "Not needed when the center is given or when the FITS files hold the pointing.",
    ("astrometry", "focal_mm"): "Read from the files when the camera records it; telescopes usually do not. Empty, "
                                "the solver searches for the scale (a few minutes).",
    ("astrometry", "scale"): "206.265 × pixel size (µm) / focal length (mm), divided by the drizzle scale.",
    ("astrometry", "blind"): "Works for fields wider than about 1°; the first time, two catalogues of bright stars "
                             "are downloaded.",
    ("astrometry", "sip_order"): "3 for refractors with a reducer and for wide fields; 2 for small fields; 0 = pure "
                                 "tangent plane.",
    ("background", "sky_order"): "2 or 3 for most fields; more for very wide fields with complex gradients. Too high, "
                                 "it can absorb a large faint nebula when no H-alpha survey serves as reference.",
    ("calibrate", "hot_sigma"): "Low values (0.5 to 1) exclude more pixels; with many frames this costs nothing. "
                                "Raise it if more than about 5 % of the pixels are excluded.",
    ("calibrate", "flat_defect"): "0.3 = 30 %. Lower it if dust remains visible after calibration.",
    ("calibrate", "hot_from_lights"): "Skipped on its own when the sky did not move on the sensor during the night.",
    ("calibrate", "flat_scale_px"): "About 40 px removes the dust and keeps the vignetting.",
    ("calibrate", "saturation"): "Keep the default, unless the star cores look clipped or rings appear around the "
                                 "bright stars.",
    ("calibrate", "workers"): "By default, the number of CPU cores minus two (12 at most).",
    ("catalogs", "g_faint"): "Equal to the limit of the measured stars, it turns off the modelling of the faint stars.",
    ("color", "aperture"): "About twice the star size; larger, it is safer in poor seeing, but noisier.",
    ("color", "white_temp"): "This is the physical calibration; the render can still warm or cool the image without "
                             "touching it.",
    ("denoise", "iterations"): "More steps give a slightly better denoising; the gain levels off near 20,000. "
                               "Training is saved every 1000 steps and resumes after an interruption.",
    ("inventory", "max_lights"): "Useful for a quick test of the whole chain; keep 0 for the real run.",
    ("register", "detect_sigma"): "Lower it (5) for very short or very faint frames; raise it in very dense Milky Way "
                                  "fields to keep only clean stars.",
    ("stack", "scale"): "1.5 for most setups with tens of frames; 2 only with many well-dithered frames and small "
                        "stars; 1 when the stars are large (over 5 px) or the frames few. Higher costs memory and "
                        "adds noise.",
    ("stack", "pixfrac"): "Smaller, the image is sharper but noisier, with a risk of holes; 0.6 to 0.8 is the usual "
                          "range.",
    ("stack", "min_transparency"): "0 keeps everything (hazy frames still get a low weight). Raise it if hazy frames "
                                   "bring gradients or swollen halos.",
    ("stack", "kappa"): "Lower (3 to 4), it removes more trails, but can eat into the star cores in poor seeing.",
    ("stack", "pattern_removal"): "Off by default: on the Rosette, the estimate also held the traces of the faint "
                                  "stars, and subtracting it drew the streaks it was meant to remove. Neutral on "
                                  "Heart.",
    ("star_net", "steps"): "11,000 steps take about 32 minutes on an RTX 4060 Ti. Far fewer leaves faint halos.",
    ("star_net", "glow"): "On, the core of NGC 2023 is kept (off, the network took it for the light of its star). "
                          "Also keeps a round glow of about 40 px around the central star of the Iris.",
    ("starless", "wiener_k"): "1 is the statistically neutral value. Higher = smoother, lower = more grain.",
    ("stars", "color_prior"): "Small, the colors follow Gaia closely and avoid failed fits (green dots, black holes); "
                              "large, the measurement is trusted more.",
    ("stars", "local_psf"): "Needed when the star shape changes quickly across the field (Newtonian coma, field "
                            "curvature): otherwise each star leaves a ring in the starless image.",
    ("stars", "angular_shape"): "The spikes then leave the nebula with the stars and are redrawn with them.",
    ("stars", "far_halo"): "The glow then leaves the nebula with the stars; the redrawn stars do not carry it.",
    ("surveys", "survey"): "\"auto\" takes NSNS (finer) when it covers the field, otherwise Finkbeiner's all-sky map. "
                           "\"none\" for fields without emission (galaxies, clusters): the background is then a "
                           "smooth polynomial.",
}
CHOICES = {("surveys", "survey"): [("auto", "auto"), ("nsns", "NSNS"), ("finkbeiner", "Finkbeiner"),
                                   ("none", "none")],
           ("star_net", "device"): [("auto", "auto"), ("main", "main"), ("aux", "secondary")]}
UNITS = {"x median": "× median", "frames": "frames", "deg": "°", "sigma": "σ", "x": "×", "G mag": "G",
         "arcsec/px": "″/px", "e-/ADU": "e⁻/ADU", "um": "µm", "x field": "× field", "x noise": "× noise",
         "x sky noise": "× sky noise", "x white level": "× white level"}
# where each stage is computed, as the chain names it
STAGE_STEP = {"inventory": "Calibration", "calibrate": "Calibration", "register": "Registration",
              "stack": "Stacking", "astrometry": "Astrometry", "catalogs": "Astrometry", "surveys": "Astrometry",
              "color": "Color", "denoise": "Denoise", "optics": "Stars", "stars": "Stars",
              "star_net": "Neural star removal", "starless": "Starless image", "background": "Background",
              "fond_simple": "Background"}
LEVEL_TEXT = {"normal": "normal", "expert": "expert", "system": "system"}


# ======================================================================== resolving a setting for a workflow
def catalog_key(p: PathDef, step: str) -> str:
    return f"fond:{p.opts['fond']}" if step == "fond" else step


def render_stage(s: Session, p: PathDef) -> str | None:
    return next((x for x in s.path_stages(p) if x in ("render_prep", "render_prep_learned")), None)


def resolve(s: Session, p: PathDef, step: str, it: dict) -> dict | None:
    """The setting as shown: label, help, tip, value, suggestion, reason, bounds; None when not in this workflow.
    The texts are translated (_()) here, at display time."""
    stage = it["stage"]
    if stage == "@render":
        stage = render_stage(s, p)
        if stage is None:
            return None
    if stage not in TRUNK and stage not in s.path_stages(p):
        return None
    target = s.trunk_params if stage in TRUNK else p.params
    stored = target.get(stage, {})
    r = dict(it, stage=stage, cat_stage=it["stage"], step_key=step, key=f"{stage}.{it['name']}")
    if it["kind"] == "engine":
        q = next((x for x in s.stages[stage]["params"] if x["name"] == it["name"]), None)
        if q is None:
            return None
        ch = CHOICES.get((stage, it["name"]))
        r.update(label=_(it["label"] or q["label"]), help=_(it["help"] or q["help"]), suggested=q["value"],
                 reason=_(q["reason"]) if q["origin"] == "auto" else (_("set in the project") if q["origin"] ==
                                                                       "user" else _("engine default")),
                 min=q["min"], max=q["max"], step=q["step"], tip=_(TIPS.get((stage, it["name"]), "")),
                 unit=_(it["unit"] if it["unit"] is not None else UNITS.get(q["unit"], q["unit"])),
                 choices=[(v, _(t)) for v, t in ch] if ch else None,
                 type=_type_of(q["value"], q, ch))
    else:
        r.update(label=_(it["label"]), help=_(it["help"]), unit=_(it["unit"]), suggested=it["default"], tip="",
                 reason=_(it.get("reason") or "proposed value"),
                 choices=[(c, _(c)) for c in it["choices"]] if it["choices"] else None)
    r["value"] = stored.get(it["name"], r["suggested"])
    r["changed"] = it["name"] in stored
    where = STAGE_STEP.get(stage)
    r["where"] = _(where) if where and where != "Background" and _home_step(stage) != step and step != "post" else ""
    return r


def _type_of(v, q, choices):
    if choices:
        return "choice"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if q["name"] in ("ra", "dec", "focal_mm", "pixel_um", "scale"):
        return "optfloat"
    return "str"


def _step_title(step):
    from steps import BY_KEY
    return BY_KEY[step]["title"] if step in BY_KEY else step


def _home_step(stage):
    """The key of the step that computes this stage (compared by key, so that it holds in every language)."""
    from steps import BY_KEY
    return next((k for k, b in BY_KEY.items() if stage in b.get("stages", [])), None)


def step_settings(s: Session, p: PathDef, step: str):
    """[(group title, [resolved settings])] for one step of the workflow; the group titles are translated."""
    out = []
    for title, items in CATALOG.get(catalog_key(p, step), []):
        rs = [r for r in (resolve(s, p, step, it) for it in items) if r is not None]
        if rs:
            out.append((_(title), rs))
    return out


def all_settings(s: Session, p: PathDef, steps_obj):
    """Every setting of the workflow, in chain order: [(step title, group, resolved)]."""
    rows = []
    for b in steps_obj.steps_of(p):
        for g, rs in step_settings(s, p, b["key"]):
            for r in rs:
                rows.append((steps_obj.title(p, b), g, r))
    return rows


def fmt(v, typ=None, choices=None) -> str:
    if choices:
        return dict(choices).get(v, str(v))
    if isinstance(v, bool):
        return _("yes") if v else _("no")
    if v is None or v == "":
        return "auto" if typ == "optfloat" else "—"
    if typ == "zone" and v in (NO_ZONE, "aucune"):
        return _("no area drawn")
    comma = language() == "fr"
    if isinstance(v, float):
        if abs(v) >= 1000 or v == int(v) and abs(v) >= 100:
            return f"{v:,.0f}".replace(",", " ")
        t = f"{v:.4g}"
        return t.replace(".", ",") if comma else t
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ") if abs(v) >= 10000 else str(v)
    return str(v)


def parse(text: str, r: dict):
    """A value typed as text (table or text view), or raises ValueError."""
    t = text.strip()
    typ = r["type"]
    if r.get("choices"):
        for val, shown in r["choices"]:
            if t.lower() in (str(val).lower(), shown.lower()):
                return val
        raise ValueError(_("possible choices: {choices}").format(choices=", ".join(c[1] for c in r["choices"])))
    if typ == "bool":
        if t.lower() in ("yes", "oui", "true", "1", "vrai", _("yes").lower()):
            return True
        if t.lower() in ("no", "non", "false", "0", "faux", _("no").lower()):
            return False
        raise ValueError(_("yes or no"))
    if typ == "zone":
        raise ValueError(_("drawn on the image"))
    if typ in ("int", "float", "optfloat"):
        if typ == "optfloat" and t.lower() in ("", "auto", "none"):
            return None
        x = float(t.replace(" ", "").replace(",", "."))
        if r.get("min") is not None and x < r["min"] or r.get("max") is not None and x > r["max"]:
            raise ValueError(_("out of range {lo} to {hi}").format(lo=fmt(r.get("min")), hi=fmt(r.get("max"))))
        return int(round(x)) if typ == "int" else x
    return "" if t == "—" else t


# ======================================================================== how one setting is shown
class RangeBar(QWidget):
    """The range of a numeric setting: the suggestion (tick) and the current value (dot)."""

    def __init__(self, lo, hi, sugg, cur, parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.sugg, self.cur = float(lo), float(hi), sugg, cur
        self.log = self.lo > 0 and self.hi / self.lo > 100
        self.setFixedHeight(22)
        self.setMinimumWidth(240)
        self.setMaximumWidth(380)

    def _x(self, v, x0, x1):
        if v is None:
            return None
        v = min(max(float(v), self.lo), self.hi)
        if self.log:
            t = (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        else:
            t = (v - self.lo) / ((self.hi - self.lo) or 1)
        return x0 + t * (x1 - x0)

    def paintEvent(self, e):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing)
            f = QFont(self.font())
            f.setPointSizeF(8)
            p.setFont(f)
            m = p.fontMetrics()
            a, b = fmt(self.lo), fmt(self.hi)
            x0 = m.horizontalAdvance(a) + 8
            x1 = self.width() - m.horizontalAdvance(b) - 8
            y = 11
            p.setPen(QColor("#6b7280"))
            p.drawText(QRectF(0, 0, x0 - 4, 22), Qt.AlignRight | Qt.AlignVCenter, a)
            p.drawText(QRectF(x1 + 4, 0, self.width() - x1, 22), Qt.AlignLeft | Qt.AlignVCenter, b)
            p.setPen(QPen(QColor("#353b47"), 3, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
            xs = self._x(self.sugg, x0, x1)
            if xs is not None:
                p.setPen(QPen(QColor(ACCENT), 2))
                p.drawLine(QPointF(xs, y - 7), QPointF(xs, y + 7))
            xc = self._x(self.cur, x0, x1)
            if xc is not None:
                changed = self.cur != self.sugg
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(WARN if changed else "#e8ebf0"))
                p.drawEllipse(QPointF(xc, y), 4.5, 4.5)
        finally:
            p.end()


class SettingRow(QFrame):
    """One setting, complete: editor, range with suggestion and value, reason, use, tip, where it is computed."""
    changed = Signal(dict, object)                  # resolved setting, new value

    def __init__(self, r: dict, parent=None):
        super().__init__(parent)
        self.r = r
        self.setObjectName("set")
        self.setStyleSheet("QFrame#set { background: transparent; } QLabel { background: transparent; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 6, 0, 8)
        v.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(8)
        name = QLabel(r["label"])
        name.setStyleSheet("font-weight: 600;")
        top.addWidget(name)
        if r["kind"] == "new":
            badge = QLabel(_("to build in the engine"))
            badge.setStyleSheet(f"color: {NEW}; border: 1px dashed {NEW}; border-radius: 8px; padding: 0 6px; "
                                "font-size: 8.5pt;")
            badge.setToolTip(_("This setting does not exist in the engine yet: the mockup shows it as a proposal."))
            top.addWidget(badge)
        if r["level"] != "normal":
            lv = QLabel(_(LEVEL_TEXT[r["level"]]))
            lv.setStyleSheet("color: #8b93a1; border: 1px solid #3a4150; border-radius: 8px; padding: 0 6px; "
                             "font-size: 8.5pt;")
            top.addWidget(lv)
        top.addStretch(1)
        top.addWidget(self._editor())
        if r.get("unit"):
            top.addWidget(label(r["unit"], "small"))
        if r["changed"]:
            rb = QPushButton("↺")
            rb.setFixedSize(24, 22)
            rb.setStyleSheet("padding: 0;")
            rb.setToolTip(_("Back to the suggestion: {value}").format(
                value=fmt(r["suggested"], r["type"], r.get("choices"))))
            rb.clicked.connect(lambda: self.changed.emit(self.r, self.r["suggested"]))
            top.addWidget(rb)
        v.addLayout(top)
        numeric = r["type"] in ("int", "float") and r.get("min") is not None and r.get("max") is not None
        if numeric:
            row = QHBoxLayout()
            row.addWidget(RangeBar(r["min"], r["max"], r["suggested"], r["value"]))
            row.addStretch(1)
            v.addLayout(row)
        sug = fmt(r["suggested"], r["type"], r.get("choices"))
        if r["changed"]:
            v.addWidget(label(_("changed · suggested: {value} ({reason})").format(value=sug, reason=r["reason"]),
                              "warn", wrap=True))
        else:
            v.addWidget(label(_("suggested: {value} · {reason}").format(value=sug, reason=r["reason"]), "reason",
                              wrap=True))
        if r.get("help"):
            v.addWidget(label(r["help"], "small", wrap=True))
        if r.get("tip"):
            t = label(_("Tip: {tip}").format(tip=r["tip"]), "small", wrap=True)
            t.setStyleSheet("color: #b9c2cf; font-size: 9pt;")
            v.addWidget(t)
        if r.get("where"):
            v.addWidget(label(_("Computed in the {step} step: changing it recomputes that step too.").format(
                step=r["where"]), "small", wrap=True))

    def _editor(self):
        r = self.r
        typ, cur = r["type"], r["value"]
        if r.get("choices"):
            e = QComboBox()
            for val, shown in r["choices"]:
                e.addItem(shown, val)
            i = e.findData(cur)
            e.setCurrentIndex(max(i, 0))
            e.setMinimumWidth(220)
            e.currentIndexChanged.connect(lambda i: self.changed.emit(self.r, e.itemData(i)))
            return e
        if typ == "bool":
            e = QCheckBox()
            e.setChecked(bool(cur))
            e.toggled.connect(lambda x: self.changed.emit(self.r, x))
            return e
        if typ == "zone":
            e = QPushButton(_("Draw on the image…"))
            e.clicked.connect(lambda: QMessageBox.information(
                self.window(), _("Mockup"), _("In the real client: the stack opens full size, the areas to exclude "
                                              "are drawn with a brush or a lasso, and the background is recomputed "
                                              "without them.")))
            return e
        if typ == "int":
            e = QSpinBox()
            e.setRange(int(r["min"]) if r.get("min") is not None else -10 ** 9,
                       int(r["max"]) if r.get("max") is not None else 10 ** 9)
            if r.get("step"):
                e.setSingleStep(int(r["step"]))
            e.setValue(int(cur))
            e.setFixedWidth(110)
            e.editingFinished.connect(lambda: e.value() != self.r["value"] and self.changed.emit(self.r, e.value()))
            return e
        if typ == "float":
            e = QDoubleSpinBox()
            v = float(r["suggested"] or 0)
            step = r.get("step") or (10 ** (math.floor(math.log10(abs(v))) - 1) if v else 0.1)
            dec = max(0, int(-math.floor(math.log10(step)))) if step < 1 else 0
            e.setDecimals(min(dec + 1, 5))
            e.setRange(r["min"] if r.get("min") is not None else -1e9, r["max"] if r.get("max") is not None else 1e9)
            e.setSingleStep(step)
            e.setValue(float(cur))
            e.setFixedWidth(110)
            e.editingFinished.connect(lambda: abs(e.value() - float(self.r["value"])) > 1e-12
                                      and self.changed.emit(self.r, round(e.value(), 6)))
            return e
        e = QLineEdit("" if cur is None else str(cur))
        e.setFixedWidth(160)
        e.setPlaceholderText("auto")

        def done():
            t = e.text().strip()
            try:
                val = parse(t, self.r) if typ == "optfloat" else (t or None)
            except ValueError:
                return
            if val != self.r["value"]:
                self.changed.emit(self.r, val)
        e.editingFinished.connect(done)
        return e
