"""The settings of each block, sorted by use rather than by engine module, and how one setting is shown.

Two kinds of settings:
* "engine": a setting the engine has (``api.describe``: its value, its suggestion and why, its bounds), placed in a group
  of a block by the catalogue below; the ones the catalogue does not place go to "Other settings" of their block, so
  that no setting of the engine is out of reach;
* "new" (◇): a setting the engine does not have yet. Shown in Spec mode only, disabled and marked ◇, so that the
  panel doubles as the list of the engine work. Its value is never stored and never changes a key.

Levels: "normal" (always shown), "expert" (Advanced settings), "system" (speed, memory, reproducibility: kept apart).

Texts are in English and go through ``_()`` when shown: ``resolve()`` returns the label, help, tip, reason, unit and
choices already translated (the engine's reasons through ``texts.format_reason``, with a decimal comma in French);
group titles are translated by ``step_settings()``. The settings of a stage come from ``ProjectModel.params`` (the
worker's ``describe``, cached): ``step_settings`` gives what it has, ``loading`` says whether some are still on their
way. ``SettingRow`` (with its ``RangeBar``) is the editor of one setting.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QLocale, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QLineEdit, QPushButton, QSpinBox, QWidget

from .. import texts
from . import prefs
from .i18n import _, language
from .widgets import ACCENT, WARN

AS_TODAY = "as the engine does today"
NO_ZONE = "none"                    # value of a "zone" setting with nothing drawn


def E(stage, name, level="normal", label=None, help=None, unit=None, method=None):   # noqa: A002 - the catalogue's words
    """An engine setting; ``method``: the stacking method it belongs to (shown only with it), None for every method."""
    return dict(kind="engine", stage=stage, name=name, level=level, label=label, help=help, unit=unit, method=method)


def N(stage, key, label, typ, default, help, level="normal", choices=None, lo=None, hi=None, step=None,   # noqa: A002
      unit="", reason="", method=None):
    return dict(kind="new", stage=stage, name="new_" + key, level=level, label=label, help=help, type=typ,
                default=default, choices=choices, min=lo, max=hi, step=step, unit=unit, reason=reason, method=method)


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
        ("Debayer", [
            N("calibrate", "debayer", "Debayer method", "choice", "RCD", "How the colour of each pixel is rebuilt "
              "from the Bayer matrix. Not used with Bayer drizzle.", choices=["RCD", "VNG", "AHD", "Bilinear"]),
            N("calibrate", "cfa_pattern", "CFA pattern", "choice", "From header", "Forces the Bayer pattern when the "
              "file header is wrong.", choices=["From header", "RGGB", "BGGR", "GRBG", "GBRG"], level="expert"),
        ]),
        ("Flat calibration", [
            N("calibrate", "flat_offset", "Flat offset", "choice", "Bias from library", "What is subtracted from the "
              "flats before they divide the lights.", choices=["Bias from library", "Dark flats",
                                                               "Camera black level"]),
            N("calibrate", "pedestal", "Output pedestal", "int", 0, "Added after calibration so that no pixel goes "
              "negative.", lo=0, hi=1000, step=10, unit="ADU", level="expert"),
        ]),
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
            N("register", "distortion", "Distortion correction", "choice", "Automatic", "Field distortion between "
              "frames (wide fields, mosaics).", choices=["Automatic", "None", "SIP polynomial", "Thin-plate splines"]),
            N("register", "framing", "Framing", "choice", "Reference frame", "The area of the stacked image.",
              choices=["Reference frame", "Union (maximum)", "Intersection (minimum)", "Centre of gravity"]),
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
        # the rule itself; the panel draws these rows above the thresholds (panel.Panel.selection)
        ("Rule", [
            E("stack", "selection"),
            E("stack", "selection_transparency", "expert"),
        ]),
    ],
    # the method itself is chosen on the cards above the settings (panel.Panel.stack_methods); each method shows only
    # its own settings (``method=``) and the ones both use
    "empilement": [
        ("Drizzle", [
            E("stack", "scale", method="drizzle"),
            E("stack", "pixfrac", help="The size of the drop each raw pixel deposits on the fine grid.",
              method="drizzle"),
            N("stack", "drizzle_kernel", "Drizzle kernel", "choice", "Square", "The shape each input pixel drops on "
              "the output grid.", choices=["Square", "Point", "Turbo", "Gaussian", "Lanczos-3"], level="expert",
              method="drizzle"),
        ]),
        ("Classic stack", [
            E("stack", "classic_debayer", label="Debayering",
              help="How the missing colours of each pixel are interpolated: RCD (sharp, round stars), or bilinear.",
              method="classic"),
            E("stack", "classic_interpolation", label="Interpolation",
              help="How each frame is resampled onto the reference frame: Lanczos-4 (sharper), or bilinear.",
              method="classic"),
            E("stack", "classic_clamping", label="Clamping",
              help="Holds back the dark rings Lanczos-4 draws around bright stars: a bilinear guide, a clamping of "
                   "its negative lobes, or none.", method="classic"),
            E("stack", "classic_rejection", label="Rejection",
              help="How outlier pixels (satellites, planes, cosmic rays) are rejected. Automatic: chosen by the "
                   "number of frames (percentile under 6 frames, winsorized sigma clipping from 6 to 15, linear fit "
                   "clipping from 16).", method="classic"),
            E("stack", "classic_cfa_balance", method="classic"),
        ]),
        ("Pixel rejection", [
            E("stack", "drizzle_rejection", method="drizzle"),
            N("stack", "rejection", "Rejection method", "choice", "Automatic: around the median",
              "How outlier pixels (satellites, planes, cosmic rays) are removed.",
              choices=["Automatic: around the median", "Sigma clipping", "Winsorized sigma clipping",
                       "Linear fit clipping", "Generalized ESD"], reason=AS_TODAY, method="drizzle"),
            E("stack", "kappa", help="A pixel farther from the median than this threshold (in noise sigmas) is "
                                     "rejected.", method="drizzle"),
            N("stack", "split_thresholds", "Separate low and high thresholds", "bool", False,
              "By default the same threshold applies to too bright and too dark pixels. Separate, bright trails can "
              "be rejected harder.", method="drizzle"),
            E("stack", "core_tolerance", "expert", help="Margin left at the core of bright stars, which varies from "
                                                        "frame to frame with the seeing.", method="drizzle"),
        ]),
        ("Normalization", [
            N("stack", "normalization", "Frame normalization", "choice", "Automatic: offset and scale",
              "Brings all the frames to the same background level and scale before combining them, despite a "
              "changing Moon or haze.",
              choices=["Automatic: offset and scale", "Offset only", "None"], reason=AS_TODAY),
        ]),
        ("Sensor", [
            E("stack", "gain", "expert", help="Electrons per digital unit: used by the noise model.",
              method="drizzle"),
            E("stack", "read_noise", "expert", help="Read noise of the sensor, in ADU.", method="drizzle"),
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
        ("Fine-tuning", [
            N("denoise_ft", "steps", "Fine-tuning iterations", "int", 600, "How many iterations the pretrained "
              "network is fine-tuned on the half-stacks.", lo=100, hi=3000, step=100),
            N("denoise_ft", "start", "Starting network", "choice", "Closest earlier project",
              "The network the fine-tuning starts from.",
              choices=["Closest earlier project", "Bundled generic network"]),
        ]),
        ("Pretrained network", [
            N("denoise_ai", "force", "Strength", "float", 0.8, "Blend between the denoised and the original image.",
              lo=0.0, hi=1.0, step=0.05),
            N("denoise_ai", "model", "Model", "choice", "Bundled", "The network applied.",
              choices=["Bundled", "Generic network"]),
        ]),
        ("NL-Bayes", [
            N("denoise_cl", "force", "Strength", "float", 1.0, "How much of the measured noise is removed.", lo=0.0,
              hi=3.0, step=0.05),
            N("denoise_cl", "vst", "Stabilise photon noise", "bool", True, "Anscombe transform before denoising: the "
              "noise becomes the same in faint and bright areas."),
            N("denoise_cl", "da3d", "DA3D refinement", "bool", False, "A second pass that keeps fine detail better; "
              "slower.", level="expert"),
        ]),
    ],
    "deconv": [
        ("Deconvolution", [
            N("deconv", "iterations", "Iterations", "int", 30, "More iterations sharpen more, and raise the noise.",
              lo=5, hi=200, step=5),
            N("deconv", "stars", "Correct the stars too", "bool", True, "Rounds and tightens the stars with the "
              "local PSF."),
            N("deconv", "regularization", "Regularization", "float", 0.002, "Holds back noise and ringing.", lo=0.0,
              hi=0.05, step=0.001, level="expert"),
        ]),
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
        ("Sky level", [
            E("background", "star_flat"),
            E("background", "bin_px", "expert", help="Size of the cells where the sky level is measured."),
            E("background", "object_clip", "expert", help="Areas brighter than this threshold are not used to "
                                                          "measure the sky level."),
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
        ("Earlier star layer", [
            E("starless", "learned_stars", "expert"),
        ]),
    ],
    "fond:simple": [
        ("Sky gradient", [
            E("gradient", "degree", help="1: a tilted plane; 2 or 3: a more flexible surface, for wide fields."),
            E("gradient", "samples", help="Number of background measurement points along each side of the image."),
            E("gradient", "object_clip", "expert", help="Areas brighter than this threshold are not used to "
                                                        "measure the background."),
        ]),
        ("Samples", [
            N("gradient", "exclusions", "Areas to exclude", "zone", NO_ZONE,
              "The drawn areas are not used to measure the background: faint nebula, galaxy, reflection."),
            N("gradient", "show_samples", "Show sample points", "bool", False,
              "Shows on the image the points where the background is measured, and lets them be moved or removed."),
        ]),
    ],
    "fond:auto": [
        ("Sky gradient", [
            E("gradient", "scale"), E("gradient", "smoothness"),
        ]),
        ("Structure protection", [
            E("gradient", "protect"), E("gradient", "protect_threshold"), E("gradient", "protect_amount"),
        ]),
        ("Strong gradients", [
            E("gradient", "simplified"), E("gradient", "simple_degree"),
        ]),
        ("System", [E("gradient", "downsample", "system")]),
    ],
    "post": [
        ("Stretch", [
            N("@render", "stretch", "Stretch", "choice", "Engine (statistical)", "How the linear output is "
              "made non-linear for Post. Post's sliders and the Stretch filter act after it.",
              choices=["Engine (statistical)", "GHS", "Arcsinh", "Histogram (STF)", "None: linear output"]),
            E("render_simple", "linked", label="Linked channels", help="The same stretch for the three channels: "
                                                                       "the colours stay as calibrated."),
        ]),
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
# the engine's settings that no panel shows: the project or the options set them, or a block control owns them
HIDDEN = {("stack", "exclude"), ("stack", "min_transparency"), ("stack", "max_fwhm_ratio"), ("star_net", "enabled"),
          ("render_simple", "stretch"), ("stack", "method"), ("gradient", "method")}
#: the stacking method: chosen on the cards of the Stacking panel, never as a row
STACK_METHOD = ("stack", "method")
#: the stacking methods' names and what each does, by the engine's key (an engine method missing here shows its
#: engine label): the list itself and the advice are the engine's (``stack_methods``, ``method_advice``)
METHOD_TEXTS = {
    "drizzle": ("Drizzle", "Each raw pixel is dropped at its exact place on the output grid: no debayering, no "
                           "interpolation. Best with many dithered frames."),
    "classic": ("Classic stack", "Each frame is resampled onto the reference (Lanczos-4), colour frames debayered "
                                 "first (RCD), then combined with a rejection chosen by the number of frames. Best "
                                 "with few or barely dithered frames."),
}
OUTPUT_STAGES = ("render_prep", "render_prep_learned", "render_simple")
# the usage tips of the settings, by (stage, setting) (the engine's own tip wins when it has one)
TIPS = {
    ("astrometry", "target"): "Not needed when the center is given or when the FITS files hold the pointing.",
    ("astrometry", "focal_mm"): "Read from the files when the camera records it; telescopes usually do not. Empty, "
                                "the solver searches for the scale (a few minutes).",
    ("astrometry", "scale"): "206.265 × pixel size (µm) / focal length (mm), divided by the drizzle scale.",
    ("astrometry", "blind"): "Works for fields wider than about 1°; the first time, two catalogues of bright stars "
                             "are downloaded.",
    ("astrometry", "sip_order"): "3 for refractors with a reducer and for wide fields; 2 for small fields; 0 = pure "
                                 "tangent plane.",
    ("calibrate", "hot_sigma"): "Low values (0.5 to 1) exclude more pixels; with many frames this costs nothing. "
                                "Raise it if more than about 5 % of the pixels are excluded.",
    ("calibrate", "flat_defect"): "0.3 = 30 %. Lower it if dust remains visible after calibration.",
    ("calibrate", "hot_from_lights"): "Skipped on its own when the sky did not move on the sensor during the night.",
    ("calibrate", "flat_scale_px"): "About 40 px removes the dust and keeps the vignetting.",
    ("calibrate", "saturation"): "Keep the default, unless the star cores look clipped or rings appear around the "
                                 "bright stars.",
    ("calibrate", "workers"): "By default, the number of CPU cores minus two (12 at most).",
    ("catalogs", "g_faint"): "Equal to the limit of the measured stars, it turns off the modelling of the faint "
                             "stars.",
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
}
UNITS = {"x median": "× median", "frames": "frames", "deg": "°", "sigma": "σ", "x": "×", "G mag": "G",
         "arcsec/px": "″/px", "e-/ADU": "e⁻/ADU", "um": "µm", "x field": "× field", "x noise": "× noise",
         "x sky noise": "× sky noise", "x white level": "× white level"}
# where each stage is computed, as the chain names it
STAGE_STEP = {"inventory": "Calibration", "calibrate": "Calibration", "register": "Registration",
              "stack": "Stacking", "astrometry": "Astrometry", "catalogs": "Astrometry", "surveys": "Astrometry",
              "reference": "Astrometry", "color": "Color", "denoise": "Denoise", "optics": "Stars",
              "stars": "Stars", "star_net": "Neural star removal", "starless": "Starless image",
              "background": "Stars", "gradient": "Gradient"}
ENGINE_LEVEL = {"basic": "normal", "advanced": "expert", "expert": "expert", "system": "system"}
TYPES = {"int": "int", "float": "float", "bool": "bool", "str": "str", "float | None": "optfloat"}


# ======================================================================== resolving a setting for a workflow
def catalog_key(p, step: str) -> str:
    return f"fond:{p.opts['fond']}" if step == "fond" else step


def render_stage(s, p) -> str | None:
    """The stage that makes the workflow's layer (render_prep, render_prep_learned or render_simple)."""
    return next((x for x in s.all_stages(p) if x in OUTPUT_STAGES), None)


def _stage_of(s, p, it: dict) -> str | None:
    stage = it["stage"]
    if stage == "@render":
        stage = render_stage(s, p)
    return stage if stage is not None and stage in s.all_stages(p) else None


def _covered() -> set[tuple[str, str]]:
    """The engine settings the catalogue places (the "@render" ones on every output stage)."""
    out = set()
    for groups in CATALOG.values():
        for _t, items in groups:
            for it in items:
                if it["kind"] == "engine":
                    for st in (OUTPUT_STAGES if it["stage"] == "@render" else (it["stage"],)):
                        out.add((st, it["name"]))
    return out


COVERED = _covered()


def _reason_text(s, info, stage: str) -> str:
    """Why the engine suggests this value, in the language shown (a decimal comma in French)."""
    lang = language()
    if not info.suggestion_known:
        return _("computed once the previous blocks are done")
    if info.reason:
        return texts.format_reason(info.reason, info.reason_args, lang)
    mem = s.reason_for(stage, info.name)
    if mem is not None:
        return texts.format_reason(mem[0], mem[1], lang)
    return _("engine default") if info.suggested == info.default else _("computed from the data")


def resolve(s, p, step: str, it: dict, stage: str | None = None) -> dict | None:
    """The setting as shown: label, help, tip, value, suggestion, reason, bounds; None when not in this workflow (or
    when the worker has not described its stage yet). The texts are translated here, at display time."""
    stage = stage or _stage_of(s, p, it)
    if stage is None:
        return None
    r = dict(it, stage=stage, cat_stage=it["stage"], step_key=step, key=f"{stage}.{it['name']}", locked=False,
             drift=False)
    if it["kind"] == "engine":
        info = s.param_info(p, stage, it["name"])
        if info is None or info.level == "internal" or info.type == "list":
            return None
        typ = TYPES.get(info.type, "str")
        choices = [(v, texts.choice_label(v, language())) for v in info.choices] if info.choices else None
        if choices:
            typ = "choice"
        lo = info.ge if info.ge is not None else info.gt
        hi = info.le if info.le is not None else info.lt
        tip = info.tip or TIPS.get((stage, it["name"]), "")
        r.update(label=_(it["label"] or info.label), help=_(it["help"] or info.help), suggested=info.suggested,
                 reason=_reason_text(s, info, stage), min=lo, max=hi, step=info.step, tip=_(tip),
                 unit=_(it["unit"] if it["unit"] is not None else UNITS.get(info.unit, info.unit)),
                 choices=choices, type=typ, value=info.value, changed=info.origin == "user", locked=info.locked,
                 drift=info.drift, result_value=info.result_value, optional=info.type == "float | None")
    else:
        r.update(label=_(it["label"]), help=_(it["help"]), unit=_(it["unit"]), suggested=it["default"], tip="",
                 reason=_(it.get("reason") or "proposed value"), value=it["default"], changed=False,
                 choices=[(c, _(c)) for c in it["choices"]] if it["choices"] else None)
    where = STAGE_STEP.get(stage)
    r["where"] = _(where) if where and _home_step(stage) != step and step != "post" else ""
    return r


def _home_step(stage: str):
    """The key of the block that computes this stage (compared by key, so that it holds in every language)."""
    from .blocks import BY_KEY
    return next((k for k, b in BY_KEY.items() if stage in b.get("stages", []) and k not in ("appris",)), None)


def _stages_of_step(s, p, step: str) -> list[str]:
    """The engine stages whose settings a block's panel lists (the half-stacks and the neural removal are the
    block's own)."""
    from .blocks import BY_KEY
    if step not in BY_KEY:
        return []
    if step == "post":
        return [x for x in s.all_stages(p) if x in OUTPUT_STAGES]
    return [x for x in BY_KEY[step]["stages"] if x in s.all_stages(p)]


def needed_stages(s, p, step: str) -> set[str]:
    """Every stage the rows of this block come from (so that the panel can ask the worker for them)."""
    out = set()
    for _t, items in CATALOG.get(catalog_key(p, step), []):
        for it in items:
            st = _stage_of(s, p, it) if it["kind"] == "engine" else None
            if st:
                out.add(st)
    out |= {x for x in _stages_of_step(s, p, step) if x not in ("inventory", "reference")}
    if step == "selection":
        out.add("stack")                                  # its thresholds are the stack's settings
    return out


def loading(s, p, step: str) -> bool:
    """Whether the settings of some stage of this block are still on their way from the worker."""
    return any(s.params(p, st) is None for st in needed_stages(s, p, step))


def stack_methods(s, p) -> list[str]:
    """The stacking methods the engine offers this workflow (the choices of its ``method`` setting), [] while the
    worker has not described the stage or when the engine has no choice of method."""
    info = s.param_info(p, *STACK_METHOD)
    return list(info.choices or []) if info is not None else []


def stack_method(s, p) -> str | None:
    """The stacking method of the workflow (the effective value), None when the engine has no choice of method."""
    info = s.param_info(p, *STACK_METHOD)
    return info.value if info is not None and info.choices else None


def method_advice(s, p) -> tuple[str, str] | None:
    """The engine's advice on the stacking method: (method, its reason in the language shown), or None when the engine
    gives none (``ParamInfo.advice`` of ``method``: an automatic value the engine never applies). The advice never
    changes the method."""
    info = s.param_info(p, *STACK_METHOD)
    adv = getattr(info, "advice", None)
    if adv is None or adv not in (info.choices or []):
        return None
    why = getattr(info, "advice_reason", "") or ""
    return adv, texts.format_reason(why, getattr(info, "advice_args", None), language()) if why else ""


def selection_rule_text(s, p) -> str:
    """The active frame selection rule in plain words (translated): what decides which frames are left out."""
    rule = s.selection_rule(p)
    if rule == "thresholds":
        return _("Frames under the minimum transparency or over the maximum star size below are left out.")
    if rule == "sota":
        off = s.param_value(p, "stack", "selection_transparency") == "off"
        return (_("Frames far worse than the night's median (over 2.5 σ) in star size or roundness are left out, "
                  "then those weighing less than 5 % of the best one.") if off else
                _("Frames far worse than the night's median (over 2.5 σ) in star size, roundness or transparency "
                  "are left out, then those weighing less than 5 % of the best one."))
    return _("Every aligned frame is kept, except those weighing less than 5 % of the best one.")


def method_text(key: str) -> tuple[str, str]:
    """(name, what it does) of a stacking method, translated."""
    name, what = METHOD_TEXTS.get(key, (texts.choice_label(key, "en"), ""))
    return _(name), _(what) if what else ""


def setting_method(r: dict) -> str | None:
    """The stacking method a setting belongs to: the catalogue's ``method``, else an engine setting of the stack named
    after a method ("classic_debayer"); None: every method."""
    if r.get("method"):
        return r["method"]
    if r.get("stage") == "stack" and "_" in r.get("name", ""):
        head = r["name"].split("_", 1)[0]
        return head if head in METHOD_TEXTS else None
    return None


def step_settings(s, p, step: str):
    """[(group title, [resolved settings])] for one block of the workflow; the group titles are translated. The
    ◇ settings come in Spec mode only. Only the settings of the workflow's stacking method are listed, with the ones
    every method uses."""
    spec = prefs.spec()
    cur = stack_method(s, p) if step == "empilement" else None

    def mine(r):
        m = setting_method(r)
        return m is None or cur is None or m == cur

    out = []
    for title, items in CATALOG.get(catalog_key(p, step), []):
        rs = [r for r in (resolve(s, p, step, it) for it in items if it["kind"] == "engine" or spec)
              if r is not None and mine(r)]
        if rs:
            out.append((_(title), rs))
    # the engine's settings the catalogue does not place: in "Other settings", by level
    rows = []
    for st in ([] if step in ("appris", "sans_etoiles", "demi_piles") else _stages_of_step(s, p, step)):
        for info in s.params(p, st) or []:
            if (st, info.name) in COVERED or (st, info.name) in HIDDEN or info.level == "internal":
                continue
            it = E(st, info.name, ENGINE_LEVEL.get(info.level, "normal"))
            r = resolve(s, p, step, it, st)
            if r is not None and mine(r):
                rows.append(r)
    if rows:
        out.append((_("Other settings"), rows))
    return out


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

    def paintEvent(self, e):                                    # noqa: N802 - Qt's name
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
    """The editor of one setting: the base of the panel's compact row. ``changed(r, value)``; value None goes back to
    the suggestion."""
    changed = Signal(dict, object)                  # resolved setting, new value

    def _editor(self):
        r = self.r
        typ, cur = r["type"], r["value"]
        off = r["kind"] == "new" or bool(r.get("locked"))            # ◇ settings and what the options set: shown, not editable
        if r.get("choices"):
            e = QComboBox()
            for val, shown in r["choices"]:
                e.addItem(shown, val)
            e.setCurrentIndex(max(e.findData(cur), 0))
            e.setMinimumWidth(220)
            e.currentIndexChanged.connect(lambda i: self.changed.emit(self.r, e.itemData(i)))
        elif typ == "bool":
            e = QCheckBox()
            e.setChecked(bool(cur))
            e.toggled.connect(lambda x: self.changed.emit(self.r, x))
        elif typ == "zone":
            e = QPushButton(_("Draw on the image…"))
        elif typ == "int":
            e = QSpinBox()
            e.setLocale(QLocale())                  # the language's numbers: a widget keeps its parent's locale
            e.setRange(int(r["min"]) if r.get("min") is not None else -10 ** 9,
                       int(r["max"]) if r.get("max") is not None else 10 ** 9)
            if r.get("step"):
                e.setSingleStep(int(r["step"]))
            e.setValue(int(cur))
            e.setFixedWidth(110)
            e.editingFinished.connect(lambda: e.value() != self.r["value"] and self.changed.emit(self.r, e.value()))
        elif typ == "float":
            e = QDoubleSpinBox()
            e.setLocale(QLocale())                  # a decimal comma in French
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
        else:
            e = QLineEdit("" if cur is None else fmt(cur, typ) if typ == "optfloat" else str(cur))
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
        if off:
            e.setEnabled(False)
            e.setToolTip("◇ " + _("Not in the engine yet.") if r["kind"] == "new" else
                         _("Set by the options of the workflow."))
        return e
