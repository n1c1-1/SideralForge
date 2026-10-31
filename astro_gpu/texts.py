"""The texts that come from the engine, in English and in French (DESIGN.md §5.4, P16).

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
import numbers
import re
import string
from pathlib import Path
from typing import Any, NamedTuple

LANGUAGES = ("en", "fr")
DECIMAL_COMMA = frozenset({"fr"})
LOCALE = Path(__file__).parent / "client" / "locale"
STAGES_DIR = Path(__file__).parent / "stages"

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


def reload() -> None:
    """Forget the catalogues read (after a file changed)."""
    _cat.clear()


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


def format_reason(template: str, args: dict | None = None, lang: str = "en") -> str:
    """The reason of an automatic value in ``lang``.

    ``template`` is the English template of ``Auto`` and ``args`` its arguments (None: a fixed text, used as it is).
    A translation that does not fit its arguments (a wrong field name) falls back on the English text.
    """
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
    fmt: str            # format spec of a number ("d", ".2f", "g", ".1%"); "" for the default
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
    ("first_time", "First frame", "", "", "Time of the first light frame."),
    ("last_time", "Last frame", "", "", "Time of the last light frame."),
    ("pointing_ra", "Pointing RA", "deg", ".3f", "Right ascension the telescope was pointing at."),
    ("pointing_dec", "Pointing Dec", "deg", ".3f", "Declination the telescope was pointing at."),
    ("object", "Object", "", "", "Name of the object written in the files."),
    # calibrate
    ("dark_level", "Dark level", "ADU", ".1f", "Typical level of the master dark."),
    ("dark_frame_noise", "Dark noise", "ADU", ".2f", "Noise of a single dark frame (read noise)."),
    ("vignetting_min", "Flat minimum", "", ".3f", "Lowest value of the master flat (1 is the centre)."),
    ("hot_pixels_from_lights", "Hot pixels in the lights", "", "d",
     "Hot pixels found in the light frames that the darks missed."),
    ("bad_pixels_pct", "Excluded pixels", "%", ".2f", "Share of the sensor pixels left out (hot, cold, dusty)."),
    ("sky_noise_adu", "Sky noise", "ADU", ".2f", "Noise of the sky background in one light frame."),
    ("cache_scale", "Cache scale", "", ".2f", "Size of the calibrated frames kept on disk (1 is full size)."),
    ("cache_gb", "Cache size", "GB", ".1f", "Disk space taken by the calibrated frames."),
    # register
    ("reference_frame", "Reference frame", "", "", "The frame all the others are aligned on."),
    ("aligned", "Aligned frames", "", "d", "Frames that could be aligned."),
    ("rms_px", "Alignment error", "px", ".3f", "Typical gap of the stars after alignment."),
    ("fwhm_px", "Star FWHM", "px", ".2f", "Median width of the stars."),
    ("fwhm_p10_p90", "Star FWHM range", "px", ".2f", "Star width of the sharpest and of the blurriest frames (10 % and 90 %)."),
    ("transparency_p10_p90", "Transparency range", "", ".2f",
     "Transparency of the worst and of the best frames (10 % and 90 %)."),
    ("drift_px", "Drift", "px", ".1f", "How far the field moved on the sensor during the session."),
    # stack
    ("frames_used", "Frames stacked", "", "d", "Frames kept in the stack."),
    ("integration_used_h", "Stacked exposure", "h", ".2f", "Exposure time of the frames kept in the stack."),
    ("grid", "Stack size", "px", "d", "Width and height of the stacked image.", " × "),
    ("fixed_pattern", "Fixed pattern", "", "", "What was measured of the sensor's fixed pattern."),
    ("rejected_pct", "Rejected pixels", "%", ".3f", "Share of pixels rejected as outliers (satellites, cosmic rays)."),
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
    ("stars_deep", "Stars to measure", "", "d", "Gaia stars of the field bright enough to be measured one by one."),
    ("stars_faint", "Stars to model", "", "d", "Fainter Gaia stars, drawn from their magnitude."),
    ("xp_spectra", "XP spectra", "", "d", "Gaia spectra downloaded for the colour calibration."),
    ("bright_stars_added", "Bright stars added", "", "d", "Hipparcos numbers of bright stars added to the Gaia list.", ", "),
    # surveys
    ("survey", "Survey", "", "", "The H-alpha survey used as a reference."),
    ("coverage", "Survey coverage", "", ".1%", "Share of the field covered by the survey."),
    ("ha_p50_p99", "H-alpha level", "R", ".2f", "Median and 99th percentile of the survey over the field (rayleigh)."),
    # color
    ("fit_agreement_pct", "Color fit error", "%", ".2f",
     "Gap between the measured star colors and those the fitted curves predict."),
    ("profile_agreement_pct", "Camera profile error", "%", ".2f", "Same gap with the saved profile of this camera."),
    ("generic_agreement_pct", "Generic sensor error", "%", ".2f", "Same gap with the curves of a typical color sensor."),
    ("response_source", "Sensitivity curves", "", "", "Where the camera's sensitivity curves come from."),
    ("stars", "Color stars", "", "d", "Stars measured for the colour calibration."),
    ("fit_residual", "Fit residual", "", ".4f", "Relative error of the fit in red, green and blue."),
    ("chromaticity_residual", "Color residual", "", ".4f", "Typical gap between the measured and the predicted star colors."),
    ("matrix", "Color matrix", "", ".3f", "Matrix that turns the camera colors into sRGB."),
    ("response_at_lines", "Response at the lines", "", "", "Camera response to H-beta, OIII and H-alpha (red, green, blue)."),
    # denoise
    ("noise_before", "Noise before", "", ".3f", "Noise measured before denoising."),
    ("noise_after", "Noise after", "", ".3f", "Noise measured after denoising."),
    ("noise_reduction", "Noise reduction", "×", ".2f", "How many times the noise went down."),
    # optics
    ("lateral_shift_max_px", "Largest color shift", "px", ".2f", "Largest shift of red or blue against green, at the edges."),
    ("g_sat", "Saturation magnitude", "mag", ".2f", "Gaia magnitude (G) where stars start to saturate."),
    ("mag_offset", "Magnitude offset", "mag", ".1f", "Shift of the magnitude thresholds for this instrument."),
    ("psf_residual_pct", "PSF residual", "%", ".1f",
     "Part of the star light the PSF model does not explain (red, green, blue)."),
    # stars
    ("prediction_scatter_pct", "Gaia prediction scatter", "%", ".1f",
     "Scatter of the star fluxes around the Gaia prediction (red, green, blue)."),
    ("capped_R", "Red fluxes capped", "", "d", "Stars whose red flux was cut to four times the Gaia prediction."),
    ("replaced_G", "Green fits replaced", "", "d", "Stars whose green flux failed the check and took the expected value."),
    ("replaced_B", "Blue fits replaced", "", "d", "Stars whose blue flux failed the check and took the expected value."),
    ("modelled_stars", "Stars measured", "", "d", "Stars measured by the joint fit."),
    ("collapsed_green", "Green collapsed", "", "d", "Stars whose green flux fell under 35 % of the expected value."),
    ("collapsed_blue", "Blue collapsed", "", "d", "Stars whose blue flux fell under 35 % of the expected value."),
    ("excess_green", "Green in excess", "", "d", "Stars whose green flux is over three times the expected value."),
    ("excess_blue", "Blue in excess", "", "d", "Stars whose blue flux is over three times the expected value."),
    ("faint_stars_bounded_by_data", "Faint stars held back", "", "d",
     "Faint stars whose flux was held below the light seen in the image."),
    ("faint_stars", "Faint stars", "", "d", "Faint Gaia stars drawn from their magnitude."),
    ("faint_light_adu", "Faint stars light", "ADU", ".3f", "Average light of the faint stars (red, green, blue)."),
    ("local_psf_grid", "Shape grid", "", "d", "Zones across and down where the star shape is corrected.", " × "),
    ("local_psf_stars_per_cell", "Stars per zone", "", "d", "Fewest and most isolated stars in a zone.", "–"),
    ("local_psf_refined_stars", "Stars refined", "", "d", "Stars whose position, width and flux were refined."),
    ("far_halo_axis_ratio", "Halo elongation", "", ".2f", "Long axis of the far halo over its short axis."),
    ("far_halo_angle_deg", "Halo angle", "deg", ".1f", "Direction of the long axis of the far halo."),
    ("far_halo", "Far halo", "", "", "Why the far halo of the stars was not measured."),
    ("far_halo_profile", "Halo profile", "", "", "Brightness of the far halo against the distance to the star."),
    ("far_halo_own", "Stars with their own halo", "", "", "Bright stars whose halo differs from the common one."),
    ("far_halo_own_amplitude", "Own halo strength", "", "", "Strength of those halos."),
    ("far_halo_kept_pct", "Halo measured on", "%", ".1f", "Share of the field used to measure the far halo."),
    ("far_halo_bounded", "Halos held back", "", "", "Stars whose halo was held below the light seen around them."),
    ("own_rays_stars", "Stars with own rays", "", ".1f",
     "Gaia magnitudes of the stars whose rays are kept with their glare.", ", "),
    ("angular_shape", "Star shape", "", "", "Result of the search for a non-round star shape shared by the stars."),
    ("angular_shape_cross_check_sigma", "Shape cross-check", "σ", ".1f",
     "How clearly each half of the bright stars shows the shape."),
    ("angular_flux_scale_rejected", "Flux scale refused", "", ".3g",
     "Flux scale the shape would need: too far from 1, so refused."),
    ("angular_shape_stars", "Stars for the shape", "", "d", "Stars used to measure the star shape."),
    ("angular_shape_saturated_measured", "Saturated stars used", "", "d", "Saturated stars among those used for the shape."),
    ("angular_shape_modelled_stars", "Stars with the shape", "", "d", "Stars drawn with the measured shape."),
    ("spike_width_px", "Spike width", "px", ".1f", "Width of the diffraction spikes."),
    ("angular_light_pct", "Light in the shape", "%", ".3f", "Share of a star's light carried by the shape beyond 12 px."),
    ("angular_flux_scale", "Shape flux scale", "", ".3f",
     "Factor between the shape and the star photometry (about 1 expected)."),
    ("angular_calibration_spread", "Shape scale spread", "", ".2f", "Scatter of that factor between the bright stars."),
    # starless
    ("filled_pct", "Filled cores", "%", ".2f", "Share of the image filled in where a star core could not be trusted."),
    ("spike_directions", "Spike directions", "deg", "d", "Directions of the diffraction spikes.", ", "),
    ("sharpen_target_fwhm", "Sharpening target", "px", ".1f", "Star width the sharpening network aims at."),
    ("sharpen_kernel_error_pct", "Sharpening error", "%", ".1f", "Worst error in rebuilding the measured star shape."),
    ("sharpen_loss", "Sharpening loss", "", ".4f", "Final training error of the sharpening network."),
    ("learned_layer_kept_pct", "Learned layer kept", "%", ".1f", "Share of the image where the learned star layer is used."),
    # star_net
    ("training_steps", "Training steps", "", "d", "Training steps of the star removal network."),
    ("training_minutes", "Training time", "min", ".1f", "Time the training took."),
    ("final_loss", "Final loss", "", ".4f", "Training error at the last step."),
    ("holdout_loss_with_stars", "Check error, stars", "", ".4f", "Error on the check regions, on the star part."),
    ("holdout_loss_without_stars", "Check error, no stars", "", ".4f", "Error on the check regions, away from the stars."),
    ("star_extent_px", "Star light reach", "px", "d", "How far the light of stars of magnitude 5, 7, 9 and 11 reaches."),
    ("holdout_regions", "Check regions", "px", "d", "Regions kept out of the training to check the network.", ", "),
    ("nebula_colour", "Nebula color", "", ".3f", "Color of the brightest extended nebula (red, green, blue)."),
    # background
    ("flat_error_pct", "Flat error", "%", ".1f", "Error of the flat field found on the stars: lowest and highest."),
    ("star_flat_photometry", "Star flat method", "", "", "How the star fluxes were measured for the star flat."),
    ("sky_fit_blocks_pct", "Sky blocks used", "%", ".1f",
     "Share of the sky blocks kept for the sky fit (the rest are objects)."),
    ("ha_fit", "H-alpha fit", "", "", "How the H-alpha part of the sky was fitted."),
    ("continuum_zero_offsets", "Continuum zero", "ADU", ".2f",
     "Offset added to red, green and blue to set the zero of the continuum."),
    ("unresolved_light_adu", "Unresolved starlight", "ADU", ".2f",
     "Light of the stars too faint to be resolved (red, green, blue)."),
    ("ha_signature", "H-alpha signature", "", ".3f", "Response to H-alpha in red, green and blue, relative to red."),
    ("large_scale_correction_adu", "Large-scale correction", "ADU", ".2f",
     "Correction toward the survey: 1 %, median and 99 % of the image."),
    ("emission_dips", "Emission dips", "", "", "Stars that leave a dark dip in the emission (4 and 8 sigma)."),
    ("gradient_pct", "Sky gradient", "%", ".2f", "Size of the gradient removed, against the sky level."),
    ("samples_used", "Sky samples", "", "d", "Sky samples kept for the fit."),
    # render_prep
    ("inner_low_coverage_kept_px", "Low coverage kept", "px", "d", "Pixels inside the image kept where few frames overlap."),
    ("vignetted_pct", "Vignetted area", "%", ".2f", "Share of the image dropped for strong vignetting."),
    ("crop", "Crop", "px", "d", "Left, right, top and bottom limits of the area kept.", ", "),
    ("emission_colour_measured", "Emission color measured", "", ".3f",
     "Color of the emission measured on the nebula (red, green, blue)."),
    ("emission_anchor_weight", "Emission anchor weight", "", ".2f",
     "Weight of the measured emission color against the fitted one."),
    ("emission_signature_source", "Emission color from", "", "", "Where the emission color comes from."),
    ("guard_footprint_pct", "Star guard area", "%", ".1f", "Share of the image covered by the star footprints."),
    ("fine_layers_off_pct", "Fine detail off", "%", ".1f", "Share of the image where the fine detail layers are switched off."),
    ("continuum_smoothed_pct", "Continuum smoothed", "%", ".1f", "Share of the image where the star color is smoothed."),
    ("emission_signature", "Emission signature", "", ".3f", "Response of the camera to H-alpha emission (red, green, blue)."),
    ("continuum_colour", "Continuum color", "", ".3f", "Color of the starlight (red, green, blue)."),
    ("oiii_noise", "OIII noise", "", ".3f", "Noise of the OIII map."),
    ("oiii_peak_snr", "OIII peak S/N", "", ".1f", "Signal to noise at the brightest OIII areas."),
    ("stars_with_measured_shape", "Stars with measured shape", "", "d", "Stars drawn with the shape measured on the image."),
    ("stars_real_light", "Stars with own spikes", "", "d", "Stars whose spikes are measured on the image."),
    ("star_depth_snr10", "Star depth", "mag", ".2f", "Gaia magnitude where the measured stars reach a signal to noise of 10."),
    ("star_depth_shift", "Star depth shift", "mag", ".2f", "Shift of the star settings for this depth of data."),
    ("star_profile", "Star profile", "", "", "The star profile used to draw the stars."),
    # render_simple
    ("sky_source", "Sky removed with", "", "", "How the sky was removed."),
    ("colour_calibrated", "Color calibrated", "", "", "Whether the colors come from the colour calibration."),
    ("denoised", "Denoised", "", "", "Whether the denoised stack was used."),
]

METRICS: dict[str, Metric] = {r[0]: Metric(*r) for r in _METRIC_ROWS}
assert len(METRICS) == len(_METRIC_ROWS), "a metric key is listed twice"

#: the fixed words some metrics hold: stored value -> English text shown (translated like any text)
VALUE_TEXTS: dict[str, str] = {
    "fit": "fitted on this field",
    "generic colour sensor": "generic colour sensor",
    "aperture": "wide apertures",
    "star models": "star models",
    "fitted": "fitted",
    "none (no survey)": "none (no survey)",
    "camera HII signature imposed (the survey correlates with other structures in this field)":
        "camera HII signature imposed (the survey correlates with other structures in this field)",
    "none": "none",
    "nsns": "NSNS",
    "finkbeiner": "Finkbeiner",
    "too few stars": "too few stars",
    "too few bright stars": "too few bright stars",
    "fewer than 3 bright stars": "fewer than 3 bright stars",
    "measured on the nebula": "measured on the nebula",
    "background_poly": "simple polynomial background",
    "channel medians": "median of each channel",
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


def metric_keys() -> list[str]:
    return list(METRICS)


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


def _plain_number(x: Any, spec: str, lang: str) -> str:
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
    spec = m.fmt if m else ""
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
    if unit and value is not None and not isinstance(value, (bool, str, dict)):
        out += unit if unit.startswith("°") else " " + unit
    return out


# ------------------------------------------------------------------------------------------------------ parameters
#: how a choice of a parameter ("auto", "nsns"...) is shown: stored value -> English text (translated like any text)
CHOICE_TEXTS: dict[str, str] = {
    "auto": "Automatic", "none": "None", "nsns": "NSNS", "finkbeiner": "Finkbeiner",
    "main": "Main GPU", "aux": "Second GPU",
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


def reason_sites() -> list[ReasonSite]:
    """Every ``Auto(...)`` written in ``stages/*.py``, found in the source (for the tests and the coverage report).

    Raises ``ValueError`` for a reason that is still an f-string or a concatenation: a reason must be a plain string
    (or a choice between plain strings) with its arguments apart.
    """
    sites: list[ReasonSite] = []
    for path in sorted(STAGES_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
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
