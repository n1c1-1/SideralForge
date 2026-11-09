"""The texts that come from the engine, in English and in French.

The client is in English with a French translation (``client/i18n.py``, ``client/locale/fr/*.json``). Five kinds of
engine text could not go through that dictionary as they were; this module gives each of them a form that does:

* **Reasons** of the automatic values. A stage writes ``Auto(value, template, args)`` (``core/params.py``): the
  template is English with ``str.format`` fields, the arguments are the numbers. ``format_reason`` translates the
  template and writes the numbers with a decimal comma in French ("1,5", not "1.5").
* **Metrics**: every key a stage writes in its manifest has a label, a unit, a format and a one-line meaning
  (``METRICS``); ``format_metric`` writes a value the way the language wants it.
* **Parameter texts**: stage titles and descriptions, parameter labels, help, tips, units and choices
  (``stage_title``, ``param_help``...). They stay in the stage files; only their French is added here.
* **Value texts**: the few fixed words some metrics hold (``VALUE_TEXTS``).
* ``translate`` is the plain lookup (English text in, translation out) behind all of it.

The French is in ``client/locale/fr/engine.json``, a dictionary {English: French} like the other catalogues; every
catalogue of the language is read, in name order, a later file winning (as ``client/i18n.py`` does), so that the
labels that already have a French in ``base.json`` are not written twice. ``coverage("fr")`` lists what is missing; the
tests (``tests/test_texts.py``) keep it empty.

This module does not import Qt, torch or the client package: it can be used by the API, the command line and the tests.
Languages are passed as arguments (``lang="en"`` or ``"fr"``); the client passes ``i18n.language()``.
"""
from __future__ import annotations

import ast
import json
import math
import numbers
import re
import string
from pathlib import Path
from typing import Any, NamedTuple

LANGUAGES = ("en", "fr")
DECIMAL_COMMA = frozenset({"fr"})
LOCALE = Path(__file__).parent / "client" / "locale"
STAGES_DIR = Path(__file__).parent / "stages"
CORE_DIR = Path(__file__).parent / "core"
#: the core modules of the stacking whose reasons reach the user (core/coverage.py writes ``Auto``s with plain
#: templates; core/frame_weights.py lists its own in ``REASON_TEMPLATES``, core/rejection_classic.py in ``REASONS``)
REASON_MODULES = ("coverage.py", "frame_weights.py", "rejection_classic.py")
DECLARED = ("REASON_TEMPLATES", "REASONS")

# ------------------------------------------------------------------------------------------------------ catalogues
_cat: dict[str, dict[str, str]] = {}


def catalogue(lang: str) -> dict[str, str]:
    """{English text: translation} of a language: all its files, in name order, a later one winning."""
    if lang not in _cat:
        out: dict[str, str] = {}
        if lang != "en":
            for f in sorted((LOCALE / lang).glob("*.json")):
                out.update(json.loads(f.read_text(encoding="utf-8")))
        _cat[lang] = out
    return _cat[lang]


def has_translation(text: str, lang: str) -> bool:
    return lang == "en" or not text or text in catalogue(lang)


def translate(text: str, lang: str = "en") -> str:
    """``text`` (English, as written in the code) in ``lang``; the English itself when there is no translation."""
    if lang == "en" or not text:
        return text
    return catalogue(lang).get(text, text)


# ------------------------------------------------------------------------------------------------------ numbers
class _Formatter(string.Formatter):
    """``str.format`` with a decimal comma for the numbers that have a decimal part (never for text arguments)."""

    def __init__(self, comma: bool):
        self.comma = comma

    def format_field(self, value, format_spec):
        out = super().format_field(value, format_spec)
        if self.comma and isinstance(value, numbers.Real) and not isinstance(value, bool):
            if not isinstance(value, numbers.Integral) or format_spec.endswith(("f", "e", "E", "g", "G", "%")):
                out = out.replace(".", ",")
        return out


_FORMATTERS = {False: _Formatter(False), True: _Formatter(True)}


def number(x: float, digits: int = 1, lang: str = "en") -> str:
    """A number with a fixed count of decimals, with a decimal comma in French ("2.5" in English, "2,5" in French)."""
    t = f"{x:.{digits}f}"
    return t.replace(".", ",") if lang in DECIMAL_COMMA else t


#: reason templates reworded after results had stored them (a stack's "reason_template"): the old English -> the new
#: one, so that an older result reads like a new one in every language (the texts do not name other software)
REWORDED: dict[str, str] = {
    "{n} frames: percentile clipping ({low:g} / {high:g}), as PixInsight's WBPP below 6 frames":
        "{n} frames: percentile clipping ({low:g} / {high:g}), the usual choice under 6 frames",
    "{n} frames: winsorized sigma clipping ({low:g} / {high:g} σ), as PixInsight's WBPP from 6 to 15 frames, with "
    "Siril's thresholds":
        "{n} frames: winsorized sigma clipping ({low:g} / {high:g} σ), the usual choice from 6 to 15 frames",
    "{n} frames: linear fit clipping ({low:g} / {high:g} σ), as PixInsight's WBPP from 16 frames":
        "{n} frames: linear fit clipping ({low:g} / {high:g} σ), the usual choice from 16 frames",
    "{n} frames: generalized ESD test (up to {low:.0%} outliers, significance {high:g}), as Siril advises above 50 "
    "frames":
        "{n} frames: generalized ESD test (up to {low:.0%} outliers, significance {high:g}), the usual advice above 50 "
        "frames",
}


def format_reason(template: str, args: dict | None = None, lang: str = "en") -> str:
    """The reason of an automatic value in ``lang``.

    ``template`` is the English template of ``Auto`` and ``args`` its arguments (None: a fixed text, used as it is).
    A translation that does not fit its arguments (a wrong field name) falls back on the English text.
    """
    template = REWORDED.get(template, template)
    if args is None:
        return translate(template, lang)
    t = translate(template, lang)
    try:
        return _FORMATTERS[lang in DECIMAL_COMMA].format(t, **args)
    except (KeyError, IndexError, ValueError):
        return _FORMATTERS[False].format(template, **args)


def reason(auto, lang: str = "en") -> str:
    """The reason of an ``Auto`` in ``lang`` (same as ``auto.text(lang)``)."""
    return format_reason(auto.template, auto.args, lang)


# ------------------------------------------------------------------------------------------------------ metrics
class Metric(NamedTuple):
    key: str
    label: str          # short name (English)
    unit: str           # written after the value, "" for none
    fmt: str            # format spec of a number ("d", ".2f", "g", ".1%"); "" for the default (see also DISPLAY)
    meaning: str        # one line (English)
    sep: str = " / "    # between the numbers of a list


_METRIC_ROWS: list[tuple] = [
    # inventory
    ("n_lights", "Lights", "", "d", "Number of light frames found."),
    ("n_darks", "Darks", "", "d", "Number of dark frames found."),
    ("n_flats", "Flats", "", "d", "Number of flat frames found."),
    ("n_bias", "Bias", "", "d", "Number of bias frames found."),
    ("n_flat_darks", "Dark flats", "", "d", "Number of dark flat frames found."),
    ("width", "Width", "px", "d", "Width of the raw frames."),
    ("height", "Height", "px", "d", "Height of the raw frames."),
    ("cfa", "Color pattern", "", "", "Order of the red, green and blue pixels on the sensor (Bayer pattern)."),
    ("camera", "Camera", "", "", "Camera model written in the files."),
    ("exposure_s", "Exposure", "s", "g", "Exposure time of one light frame."),
    ("integration_h", "Total exposure", "h", ".2f", "Exposure time of all the light frames together."),
    ("iso_gain", "ISO / gain", "", "g", "ISO or gain written in the files."),
    ("white_level", "White level", "ADU", "g", "Value at which the sensor saturates."),
    ("black_level", "Black level", "ADU", "g", "Value the sensor gives without light."),
    ("focal_mm", "Focal length", "mm", "g", "Focal length written in the files."),
    ("pixel_um", "Pixel size", "µm", ".2f", "Size of one sensor pixel."),
    ("pixel_scale", "Pixel scale", "arcsec/px", ".3f", "Sky angle of one sensor pixel, as the FITS headers give it."),
    ("first_time", "First frame", "", "", "Time of the first light frame."),
    ("last_time", "Last frame", "", "", "Time of the last light frame."),
    ("pointing_ra", "Pointing RA", "deg", ".3f", "Right ascension the telescope was pointing at."),
    ("pointing_dec", "Pointing Dec", "deg", ".3f", "Declination the telescope was pointing at."),
    ("object", "Object", "", "", "Name of the object written in the files."),
    ("mono", "Monochrome", "", "", "The frames have no colour filter array: a monochrome camera."),
    ("colour_frames", "Colour frames", "", "",
     "The frames are already in colour (JPEG, PNG, TIFF or RGB FITS): no debayering, no Bayer drizzle."),
    # calibrate
    ("dark_level", "Dark level", "ADU", ".1f", "Typical level of the master dark."),
    ("dark_frame_noise", "Dark noise", "ADU", ".2f", "Noise of a single dark frame (read noise)."),
    ("vignetting_min", "Flat minimum", "", ".3f", "Lowest value of the master flat (1 is the centre)."),
    ("hot_pixels_from_lights", "Hot pixels in the lights", "", "d",
     "Hot pixels found in the light frames that the darks missed."),
    ("bad_pixels_pct", "Excluded pixels", "%", ".2f", "Share of the sensor pixels left out (hot, cold, dusty)."),
    ("bad_pixels_filled", "Pixels filled", "", "",
     "Whether the excluded pixels were filled from their neighbours (the sky did not move on the sensor)."),
    ("sky_noise_adu", "Sky noise", "ADU", ".2f", "Noise of the sky background in one light frame."),
    ("cache_scale", "Cache scale", "", ".2f", "Size of the calibrated frames kept on disk (1 is full size)."),
    ("cache_gb", "Cache size", "GB", ".1f", "Disk space taken by the calibrated frames."),
    ("calibration_cut", "Calibration cut", "", "", "The frames were taken as they are, without calibration frames."),
    ("dark_scale", "Dark scaling", "", ".3f",
     "Ratio of the lights' exposure to the darks': their thermal signal was scaled by it."),
    # the sessions' join (issue #43)
    ("sessions", "Sessions", "", "d", "Number of sessions, each calibrated with its own frames, stacked together."),
    # an image source (issue #43)
    ("factor", "Value factor", "", "g", "Factor the image's values were multiplied by (65535 for values in 0-1)."),
    # the channels' fusion (issue #43)
    ("palette", "Palette", "", "", "How the filters became red, green and blue (SHO, HSO, HOO, RGB, LRGB)."),
    ("channels", "Channels", "", "", "The filters combined.", ", "),
    ("star_source", "Star source", "", "", "Where the stars put back on the starless fusion came from (palette, channel "
                                           "or workflow)."),
    # register
    ("reference_frame", "Reference frame", "", "", "The frame all the others are aligned on."),
    ("aligned", "Aligned frames", "", "d", "Frames that could be aligned."),
    ("not_aligned", "Not aligned", "", "d", "Frames left out because they could not be aligned."),
    ("rms_px", "Alignment error", "px", ".3f", "Typical gap of the stars after alignment."),
    ("fwhm_px", "Star FWHM", "px", ".2f", "Median width of the stars."),
    ("fwhm_p10_p90", "Star FWHM range", "px", ".2f", "Star width of the sharpest and of the blurriest frames (10 % and 90 %)."),
    ("transparency_p10_p90", "Transparency range", "", ".2f",
     "Transparency of the worst and of the best frames (10 % and 90 %)."),
    ("drift_px", "Drift", "px", ".1f", "How far the field moved on the sensor during the session."),
    ("registration_cut", "Registration cut", "", "", "The frames were taken as already aligned."),
    ("photometry_aperture_px", "Photometry aperture", "px", "g",
     "Diameter of the aperture the transparency of each frame is measured in."),
    ("fwhm_moffat_px", "Star FWHM (Moffat)", "px", ".2f",
     "Median width of the stars from a Moffat fit on the green pixels, at full resolution."),
    ("roundness_moffat", "Star roundness (Moffat)", "", ".2f",
     "Median ratio of the smallest to the largest width of the stars (1 is round)."),
    # stack
    ("frames_used", "Frames stacked", "", "d", "Frames kept in the stack."),
    ("integration_used_h", "Stacked exposure", "h", ".2f", "Exposure time of the frames kept in the stack."),
    ("grid", "Stack size", "px", "d", "Width and height of the stacked image.", " × "),
    ("fixed_pattern", "Fixed pattern", "", "", "What was measured of the sensor's fixed pattern."),
    ("rejected_pct", "Rejected pixels", "%", ".3f", "Share of pixels rejected as outliers (satellites, cosmic rays)."),
    ("mosaic_peaks", "Colour mosaic pattern", "", "d",
     "Peaks of the colour mosaic's periodic pattern taken off the output image."),
    ("grid_peaks", "Network grid", "", "d",
     "Peaks of the star-removal network's own grid taken off the starless image."),
    ("camera_white_balance", "Camera white balance", "", ".3f",
     "Multipliers of red, green and blue from the camera's daylight white balance, applied to the stack."),
    ("drizzle_excluded_pct", "Drops left out", "%", ".3f",
     "Share of the drizzle's drops left out where the frames' integration rejected the pixel."),
    ("channel_offsets", "Color offsets", "", "",
     "Where the red and blue images of the stars fell relative to the green one (lateral chromatic aberration, "
     "atmospheric dispersion), corrected in the stack."),
    ("near_saturation_pct", "Near saturation", "%", ".4f",
     "Share of the pixels of the stack with samples near saturation (star cores)."),
    ("coverage_variation_pct", "Coverage variation", "%", ".1f",
     "How much the weight of the pixels of the stack varies, red / green / blue (under 20 % is even enough)."),
    # astrometry
    ("ra", "Centre RA", "deg", ".4f", "Right ascension of the image centre."),
    ("dec", "Centre Dec", "deg", ".4f", "Declination of the image centre."),
    ("scale_arcsec", "Image scale", "arcsec/px", ".3f", "Size of a pixel of the stack on the sky."),
    ("sensor_scale_arcsec", "Sensor scale", "arcsec/px", ".3f", "Size of a sensor pixel on the sky."),
    ("field_deg", "Field of view", "deg", ".2f", "Width and height of the image on the sky.", " × "),
    ("up_position_angle", "Rotation", "deg", ".1f", "Angle between the top of the image and north, towards east."),
    ("parity_flipped", "Mirrored", "", "", "Whether the image is a mirror image of the sky."),
    ("residual_px", "Position residual", "px", ".3f", "Typical gap between the stars and their expected positions."),
    ("stars_used", "Stars used", "", "d", "Stars matched to the catalogue in the solution."),
    ("focal_estimate_mm", "Focal length found", "mm", ".1f", "Focal length deduced from the solution."),
    ("blind_solved", "Blind solved", "", "", "Whether the field was found on the whole sky, with no starting point."),
    # reference
    ("orientation", "Orientation", "", "", "Mirror and quarter turns that bring the reference image onto the stack."),
    ("stars_matched", "Stars matched", "", "d", "Stars found in both the stack and the reference image."),
    # catalogs
    ("stars_deep", "Field stars (Gaia)", "", "d", "Bright Gaia stars of the field (G < 16)."),
    ("stars_faint", "Stars to model", "", "d", "Fainter Gaia stars, drawn from their magnitude."),
    ("xp_spectra", "XP spectra", "", "d", "Gaia spectra downloaded for the colour calibration."),
    ("bright_stars_added", "Bright stars added", "", "d", "Hipparcos numbers of bright stars added to the Gaia list.", ", "),
    # surveys
    ("survey", "Survey", "", "", "The H-alpha survey used as a reference."),
    ("coverage", "Survey coverage", "", ".1%", "Share of the field covered by the survey."),
    ("ha_p50_p99", "H-alpha level", "R", ".2f", "Median and 99th percentile of the survey over the field (rayleigh)."),
    # render_simple
    ("saturated_places", "Saturated places", "", "d",
     "Star cores near saturation that take the color of their surroundings."),
    # color
    ("fit_agreement_pct", "Color fit error", "%", ".2f",
     "Gap between the measured star colors and those the fitted curves predict."),
    ("profile_agreement_pct", "Camera profile error", "%", ".2f", "Same gap with the saved profile of this camera."),
    ("generic_agreement_pct", "Generic sensor error", "%", ".2f", "Same gap with the curves of a typical color sensor."),
    ("response_source", "Sensitivity curves", "", "", "Where the camera's sensitivity curves come from."),
    ("correction", "Colour correction", "", "", "What turned the camera colours into true colours: a colour matrix or "
                                                "one factor per colour."),
    ("stars", "Color stars", "", "d", "Stars measured for the colour calibration."),
    ("fit_residual", "Fit residual", "", ".4f", "Relative error of the fit in red, green and blue."),
    ("chromaticity_residual", "Color residual", "", ".4f", "Typical gap between the measured and the predicted star colors."),
    ("white_balance", "White balance", "", ".3f",
     "Multipliers of red, green and blue that make the stars match their catalogue colors."),
    ("white_balance_stars", "White balance stars", "", "d", "Stars used for the white balance."),
    ("matrix", "Color matrix", "", ".3f", "Matrix that turns the camera colors into sRGB."),
    ("response_at_lines", "Response at the lines", "", "", "Camera response to H-beta, OIII and H-alpha (red, green, blue)."),
    # denoise
    ("noise_before", "Noise before", "", ".3f", "Noise measured before denoising."),
    ("noise_after", "Noise after", "", ".3f", "Noise measured after denoising."),
    ("noise_reduction", "Noise reduction", "×", ".2f", "How many times the noise went down."),
    ("noise_correlation", "Noise correlation between neighbours", "", ".2f",
     "Correlation of the noise between neighbouring pixels: near 0 for a session without dithering."),
    ("adaptation_steps", "Adaptation steps", "", "d", "Steps the network learned this image for (0: pretrained only)."),
    # gradient
    ("gradient_pct", "Sky gradient", "%", ".2f", "Size of the gradient removed, against the sky level."),
    ("samples_used", "Sky samples", "", "d", "Sky samples kept for the fit."),
    # render_simple
    ("crop", "Crop", "px", "d", "Left, right, top and bottom limits of the area kept.", ", "),
    ("sky_source", "Sky removed with", "", "", "How the sky was removed."),
    ("colour_calibrated", "Color calibrated", "", "", "Whether the colors come from the colour calibration."),
    ("denoised", "Denoised", "", "", "Whether the denoised stack was used."),
]

METRICS: dict[str, Metric] = {r[0]: Metric(*r) for r in _METRIC_ROWS}
#: metrics no stage writes any more, kept for the results computed before (the colour stage's reconstructed
#: sensitivity curves and matrix, taken out; the faint stars of the catalogue, for the star model that is gone)
RETIRED_METRICS = frozenset({"response_source", "correction", "fit_agreement_pct", "profile_agreement_pct",
                             "generic_agreement_pct", "fit_residual", "response_at_lines", "stars_faint"})
#: how some metrics read on screen, beyond their number format (display only: the stored values are unchanged):
#: "hours" a duration given in hours, in minutes under one hour (0.17 -> "10 min", 3.02 -> "3.0 h"); "sig2" two
#: significant digits (1.821 -> "1.8", 0.0123 -> "0.012")
DISPLAY: dict[str, str] = {"integration_used_h": "hours", "rejected_pct": "sig2"}
assert len(METRICS) == len(_METRIC_ROWS), "a metric key is listed twice"

#: the fixed words some metrics hold: stored value -> English text shown (translated like any text)
VALUE_TEXTS: dict[str, str] = {
    "fit": "fitted on this field",
    "generic colour sensor": "generic colour sensor",
    "one factor per colour (no reliable sensitivity curves)": "one factor per colour (no reliable sensitivity curves)",
    "colour matrix": "colour matrix",
    "one factor per colour": "one factor per colour",
    "aperture": "wide apertures",
    "fitted": "fitted",
    "none": "none",
    "nsns": "NSNS",
    "finkbeiner": "Finkbeiner",
    "channel medians": "median of each channel",
    "sky level of the Gradient block": "sky level of the Gradient block",
    "camera-colour polynomial, made grey": "smooth sky of each channel, made grey",
}
#: texts of the interface the formatting needs
UI_TEXTS = ("Yes", "No", "{n} values")
#: values with a number in them: (regular expression, English template, names of the numeric groups)
VALUE_PATTERNS: list[tuple[re.Pattern, str, tuple[str, ...]]] = [
    (re.compile(r"^camera profile \((?P<project>.*)\)$"), "camera profile ({project})", ()),
    (re.compile(r"^(?P<head>.*); the rejection on all channels left (?P<pct>\d+) % of the H-alpha amplitude of the whole "
                r"field: objects rejected on green and blue only$"),
     "{head}; the rejection on all channels left {pct} % of the H-alpha amplitude of the whole field: objects rejected "
     "on green and blue only", ()),
    (re.compile(r"^focused, FWHM (?P<fwhm>[\d.]+) px \(the measured stars are defocused rings\)$"),
     "focused, FWHM {fwhm} px (the measured stars are defocused rings)", ("fwhm",)),
]


def metric(key: str) -> Metric | None:
    return METRICS.get(key)


def metric_label(key: str, lang: str = "en") -> str:
    """The label of a metric; an unknown key is shown as itself, written out (``my_key`` -> ``My key``)."""
    m = METRICS.get(key)
    return translate(m.label, lang) if m else key.replace("_", " ").capitalize()


def metric_unit(key: str, lang: str = "en") -> str:
    m = METRICS.get(key)
    return translate(m.unit, lang) if m and m.unit else ""


def metric_meaning(key: str, lang: str = "en") -> str:
    m = METRICS.get(key)
    return translate(m.meaning, lang) if m else ""


def _sig2(x: float) -> str:
    """Two significant digits, without an exponent: 12.3 -> 12, 1.821 -> 1.8, 0.0123 -> 0.012."""
    a = abs(float(x))
    if a == 0 or a >= 10:
        return f"{x:.0f}"
    return f"{x:.{max(1, 1 - math.floor(math.log10(a)))}f}"


def _grouped(s: str, lang: str) -> str:
    """Thousands set apart in a whole number of five digits or more: "573,295" in English, "573 295" in French (a
    narrow no-break space); four digits stay as they are ("4742")."""
    body = s.lstrip("-")
    if not body.isdigit() or len(body) < 5:
        return s
    sep = "\u202f" if lang in DECIMAL_COMMA else ","
    out = f"{int(body):,}".replace(",", sep)
    return ("-" if s.startswith("-") else "") + out


def _plain_number(x: Any, spec: str, lang: str) -> str:
    if spec == "sig2":
        s = _sig2(x)
        return s.replace(".", ",") if lang in DECIMAL_COMMA else s
    if spec == "hours":
        if abs(x) < 1:
            return f"{max(1, round(x * 60)) if x > 0 else 0} min"
        s = f"{x:.1f} h"
        return s.replace(".", ",") if lang in DECIMAL_COMMA else s
    if spec == "d" and isinstance(x, numbers.Integral) and not isinstance(x, bool):
        return _grouped(format(x, "d"), lang)
    if spec.endswith("%"):
        s = format(x * 100, spec[:-1] + "f") + " %"
    else:
        if spec == "d" and not isinstance(x, numbers.Integral):
            spec = "g"
        s = format(x, spec)
    return s.replace(".", ",") if lang in DECIMAL_COMMA else s


def _text_value(v: str, lang: str) -> str:
    if v in VALUE_TEXTS:
        return translate(VALUE_TEXTS[v], lang)
    for rx, template, numeric in VALUE_PATTERNS:
        m = rx.match(v)
        if m:
            parts = m.groupdict()
            if "head" in parts:                  # the start of the sentence is a fixed word of its own
                parts["head"] = _text_value(parts["head"], lang)
            if lang in DECIMAL_COMMA:
                parts = {k: (x.replace(".", ",") if k in numeric else x) for k, x in parts.items()}
            return translate(template, lang).format(**parts)
    return v


def _show(v: Any, m: Metric | None, lang: str) -> str:
    spec = (DISPLAY.get(m.key) or m.fmt) if m else ""
    sep = m.sep if m else " / "
    if v is None:
        return "–"
    if isinstance(v, bool):
        return translate("Yes" if v else "No", lang)
    if isinstance(v, numbers.Real):
        return _plain_number(v, spec, lang)
    if isinstance(v, str):
        return _text_value(v, lang)
    if isinstance(v, (list, tuple)):
        if v and all(isinstance(x, (list, tuple)) for x in v):
            return " ; ".join(_show(x, m, lang) for x in v)
        return sep.join(_show(x, m, lang) for x in v)
    if isinstance(v, dict):
        return translate("{n} values", lang).format(n=len(v))
    return str(v)


def format_metric(key: str, value: Any, lang: str = "en") -> str:
    """A metric value as the screen shows it: "2,31 px" in French, "2.31 px" in English.

    Numbers follow the format of the catalogue; a list gives its numbers joined by the metric's separator, then the
    unit; a flag gives Yes / No; a fixed word is translated; a dictionary gives its size. An unknown key gives a
    plain rendering of the value.
    """
    m = METRICS.get(key)
    out = _show(value, m, lang)
    unit = metric_unit(key, lang)
    if m is not None and DISPLAY.get(m.key) == "hours":
        unit = ""                                     # the duration carries its own unit (min or h)
    if unit and value is not None and not isinstance(value, (bool, str, dict)):
        out += unit if unit.startswith("°") else " " + unit
    return out


# ------------------------------------------------------------------------------------------------------ parameters
#: how a choice of a parameter ("auto", "nsns"...) is shown: stored value -> English text (translated like any text)
CHOICE_TEXTS: dict[str, str] = {
    "auto": "Automatic", "none": "None", "nsns": "NSNS", "finkbeiner": "Finkbeiner",
    "main": "Main GPU", "aux": "Second GPU",
    # the stacking settings (stages/stack.py)
    "thresholds": "Thresholds", "sota": "k × σ filters", "on": "On", "off": "Off",
    "min_weight": "Automatic", "own": "On the raw pixels", "integration": "From the classic stack's rejection",
    "drizzle": "Bayer drizzle", "classic": "Classic",
    "rcd": "RCD", "bilinear": "Bilinear", "lanczos4": "Lanczos-4",
    # the Classic method's clamping (core/resample.CLAMPINGS): the stored values name where each comes from, the
    # interface says what it does
    "siril": "Bilinear guide", "pixinsight": "Lobe clamping",
    "percentile": "Percentile clipping", "winsorized": "Winsorized sigma clipping", "linear_fit": "Linear fit clipping",
    "gesdt": "Generalized ESD test",
    # the colour stage's correction (stages/color.py)
    "full": "Full correction", "per_colour": "One factor per colour",
    # the Gradient block's methods (stages/gradient.py)
    "polynomial": "Polynomial", "rbf": "RBF", "automatic": "Automatic",
    # the fusion's palettes (stages/fusion.py)
    "SHO": "SHO (Hubble palette)", "HSO": "HSO", "HOO": "HOO", "RGB": "RGB", "LRGB": "LRGB",
    # the Denoise block's methods and adaptation lengths (stages/denoise.py)
    "pretrained": "Pretrained network", "adapted": "Pretrained + adapted", "short": "Short", "long": "Long",
}

_stage_index: dict[str, dict] | None = None


def _stages() -> dict[str, dict]:
    """{stage name: {"title", "description", "params": {name: description dict}}} of every stage of v2 projects."""
    global _stage_index
    if _stage_index is None:
        from .stages.v2 import V2_STAGES
        _stage_index = {s.name: {"title": s.title, "description": s.description,
                                 "params": {d["name"]: d for d in s.Params.describe()}} for s in V2_STAGES()}
    return _stage_index


def stage_names() -> list[str]:
    return list(_stages())


def stage_title(stage: str, lang: str = "en") -> str:
    return translate(_stages()[stage]["title"], lang)


def stage_description(stage: str, lang: str = "en") -> str:
    return translate(_stages()[stage]["description"], lang)


def param_names(stage: str) -> list[str]:
    return list(_stages()[stage]["params"])


def _param_text(stage: str, name: str, field: str, lang: str) -> str:
    return translate(_stages()[stage]["params"][name].get(field) or "", lang)


def param_label(stage: str, name: str, lang: str = "en") -> str:
    return _param_text(stage, name, "label", lang)


def param_help(stage: str, name: str, lang: str = "en") -> str:
    """What the parameter does (the ``help`` of the stage file) in ``lang``."""
    return _param_text(stage, name, "help", lang)


def param_tip(stage: str, name: str, lang: str = "en") -> str:
    """When and how to change it (the ``tip`` of the stage file), "" when there is none."""
    return _param_text(stage, name, "tip", lang)


def param_unit(stage: str, name: str, lang: str = "en") -> str:
    return _param_text(stage, name, "unit", lang)


def param_texts(stage: str, name: str, lang: str = "en") -> dict[str, str]:
    """{"label", "help", "tip", "unit"} of a parameter in ``lang`` ("" for what it has not)."""
    return {f: _param_text(stage, name, f, lang) for f in ("label", "help", "tip", "unit")}


def choice_label(value: str, lang: str = "en") -> str:
    """A choice of a parameter in words ("nsns" -> "NSNS", "none" -> "None" / "Aucun")."""
    return translate(CHOICE_TEXTS.get(value, value), lang)


# ------------------------------------------------------------------------------------------------------ reasons
class ReasonSite(NamedTuple):
    where: str                  # "stack.py:93"
    templates: list[str]        # the template, or the templates a conditional chooses between
    arg_names: set[str] | None  # the keys of the arguments dictionary written there; None: no arguments


def _declared_templates(path: Path, tree: ast.Module) -> list[str] | None:
    """The templates a module lists in its top-level ``REASON_TEMPLATES`` or ``REASONS`` (a dict of them, or a
    sequence), None when it has none. Read as a literal when it is one (no import: core/rejection_classic.py needs
    torch), else from the module itself (core/frame_weights.py and core/coverage.py build theirs; they import numpy
    only)."""
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            name = next((t.id for t in targets if isinstance(t, ast.Name) and t.id in DECLARED), None)
            if name is not None:
                try:
                    value = ast.literal_eval(node.value)
                except ValueError:
                    import importlib
                    value = getattr(importlib.import_module(f"{__package__}.core.{path.stem}"), name)
                return list(value.values()) if isinstance(value, dict) else list(value)
    return None


def reason_sites() -> list[ReasonSite]:
    """Every ``Auto(...)`` written in ``stages/*.py`` and in the core modules of REASON_MODULES, found in the source;
    for a module that lists its templates in ``REASON_TEMPLATES`` (its ``Auto``s take them from variables), one site
    per template listed (for the tests and the coverage report).

    Raises ``ValueError`` for a reason that is still an f-string or a concatenation: a reason must be a plain string
    (or a choice between plain strings) with its arguments apart.
    """
    sites: list[ReasonSite] = []
    for path in sorted(STAGES_DIR.glob("*.py")) + [CORE_DIR / m for m in REASON_MODULES]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        declared = _declared_templates(path, tree) if path.parent == CORE_DIR else None
        if declared is not None:
            sites += [ReasonSite(f"{path.name}:REASON_TEMPLATES", [t], template_fields(t)) for t in declared]
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Auto"):
                continue
            where = f"{path.name}:{node.lineno}"
            kw = {k.arg: k.value for k in node.keywords}
            expr = node.args[1] if len(node.args) > 1 else kw.get("template", kw.get("reason"))
            args = node.args[2] if len(node.args) > 2 else kw.get("args")
            if expr is None:
                raise ValueError(f"{where}: Auto without a reason")
            if any(isinstance(sub, (ast.JoinedStr, ast.BinOp)) for sub in ast.walk(expr)):
                raise ValueError(f"{where}: the reason of an Auto must be a plain string, its numbers in the arguments")
            templates = [sub.value for sub in ast.walk(expr) if isinstance(sub, ast.Constant) and isinstance(sub.value, str)]
            names = None
            if isinstance(args, ast.Dict):
                names = {k.value for k in args.keys if isinstance(k, ast.Constant)}
            elif args is not None:
                names = set()            # arguments built elsewhere: not checkable here
            sites.append(ReasonSite(where, templates, names))
    return sites


def reason_templates() -> dict[str, list[str]]:
    """Every reason template written in ``stages/*.py``: {template: [where, ...]}."""
    out: dict[str, list[str]] = {}
    for site in reason_sites():
        for t in site.templates:
            out.setdefault(t, []).append(site.where)
    return out


def template_fields(template: str) -> set[str]:
    """The names of the fields of a template ("{n} frames, {fwhm:.1f} px" -> {"n", "fwhm"})."""
    return {name.split(".")[0].split("[")[0] for _, name, _, _ in string.Formatter().parse(template) if name}


# ------------------------------------------------------------------------------------------------------ coverage
def coverage(lang: str = "fr") -> dict[str, list[str]]:
    """The engine texts with no translation in ``lang``, by kind (empty lists: nothing missing).

    Kinds: reasons, metrics (labels, meanings, units), values, parameters (labels, help, tips, units, choices),
    stages (titles, descriptions), interface.
    """
    def missing(texts) -> list[str]:
        return sorted({t for t in texts if t and not has_translation(t, lang)})

    st = _stages()
    params = [d for s in st.values() for d in s["params"].values()]
    return {
        "reasons": missing(reason_templates()),
        "metrics": missing([t for m in METRICS.values() for t in (m.label, m.meaning, m.unit)]),
        "values": missing(list(VALUE_TEXTS.values()) + [p[1] for p in VALUE_PATTERNS]),
        "labels": missing(d.get("label") for d in params),
        "help": missing(d.get("help") for d in params),
        "tips": missing(d.get("tip") for d in params),
        "units": missing(d.get("unit") for d in params),
        "choices": missing([CHOICE_TEXTS.get(c, c) for d in params for c in d.get("choices") or []]),
        "stages": missing([s["title"] for s in st.values()] + [s["description"] for s in st.values()]),
        "interface": missing(UI_TEXTS),
    }


if __name__ == "__main__":                 # python -m astro_gpu.texts [fr]: what is still missing
    import sys
    report = coverage(sys.argv[1] if len(sys.argv) > 1 else "fr")
    for kind, items in report.items():
        print(f"{kind:10s} {len(items)} missing")
        for t in items:
            print("   ", t[:100])
    sys.exit(1 if any(report.values()) else 0)
