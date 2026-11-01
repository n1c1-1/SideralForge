"""What a block shows in Inspect: its pictures and its measurements.

The block table itself (title, stages, what it does) is ``blocks.STEPS``; the state of a block is ``blocks.block_state``.
Here: which pictures a block has (``visuals``: arrays of the done results and the engine's previews), and its
measurements (``facts``, read from the manifests by the worker and formatted with the metric catalogue of
``astro_gpu/texts.py``, so that a value reads the same everywhere and in both languages).

``Steps`` is a QObject: ``facts`` answers from what the worker has delivered and asks for the rest; ``changed`` fires when
more has arrived (the panel is built again).
"""
from __future__ import annotations

import html

from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QPainter

from .. import texts
from .blocks import BY_KEY, STEPS, block_state, fmt_int  # noqa: F401  (the views import them from here)
from .i18n import _, language, number
from .services.inspect import VisualSpec, service_of
from .viewer import Viewer
from .widgets import WARN

DIM = "#8b93a1"

#: the metrics each block quotes, in order: (stage, metric key). The catalogue (texts.py) gives the label, unit and format.
FACTS = {
    "calibration": [("calibrate", "vignetting_min"), ("calibrate", "bad_pixels_pct"),
                    ("calibrate", "hot_pixels_from_lights"), ("calibrate", "dark_level"),
                    ("calibrate", "dark_frame_noise"), ("calibrate", "sky_noise_adu"), ("calibrate", "cache_gb")],
    "alignement": [("register", "reference_frame"), ("register", "aligned"), ("register", "rms_px"),
                   ("register", "fwhm_p10_p90"), ("register", "drift_px")],
    "empilement": [("stack", "frames_used"), ("stack", "integration_used_h"), ("stack", "grid"),
                   ("stack", "rejected_pct")],
    "astrometrie": [("astrometry", "ra"), ("astrometry", "dec"), ("astrometry", "field_deg"),
                    ("astrometry", "scale_arcsec"), ("astrometry", "stars_used"), ("astrometry", "residual_px"),
                    ("astrometry", "focal_estimate_mm"), ("reference", "stars_matched"), ("surveys", "survey"),
                    ("surveys", "coverage")],
    "couleur": [("color", "stars"), ("color", "fit_agreement_pct"), ("color", "profile_agreement_pct"),
                ("color", "response_source"), ("color", "chromaticity_residual")],
    "debruitage": [("denoise", "noise_before"), ("denoise", "noise_after"), ("denoise", "noise_reduction")],
    "etoiles": [("stars", "modelled_stars"), ("stars", "faint_stars"), ("stars", "prediction_scatter_pct"),
                ("stars", "local_psf_grid"), ("optics", "psf_residual_pct"), ("optics", "lateral_shift_max_px"),
                ("starless", "filled_pct"), ("star_net", "training_steps"), ("star_net", "final_loss")],
    "fond": [("background", "sky_fit_blocks_pct"), ("background", "ha_fit"), ("background", "emission_dips"),
             ("background", "large_scale_correction_adu"), ("background_poly", "sky_fit_blocks_pct"),
             ("background_poly", "gradient_pct"), ("background_poly", "samples_used")],
    "post": [("render_prep", "crop"), ("render_prep", "vignetted_pct"), ("render_prep", "star_depth_snr10"),
             ("render_simple", "crop"), ("render_simple", "sky_source")],
}


def fnum(x, d=1) -> str:
    return number(x, d)


def kv(rows, pad=14):
    """Two columns of text, a name and its value (rich text)."""
    from PySide6.QtWidgets import QLabel
    table = "<table cellspacing='0' cellpadding='3'>" + "".join(
        f"<tr><td style='color:{DIM}; padding-right:{pad}px' valign='top'>{k}</td><td>{v}</td></tr>"
        for k, v in rows) + "</table>"
    lb = QLabel(table)
    lb.setTextFormat(Qt.RichText)
    lb.setWordWrap(True)
    return lb


def metric_row(stage: str, key: str, value) -> tuple[str, str]:
    """(label, value) of a metric, in the language of the interface, from the engine's catalogue."""
    lang = language()
    return html.escape(texts.metric_label(key, lang)), html.escape(texts.format_metric(key, value, lang))


class Steps(QObject):
    """The pictures and the measurements of the blocks of one model."""

    changed = Signal()

    def __init__(self, model, service=None, parent=None):
        super().__init__(parent)
        self.s = model
        self.service = service or service_of(model)
        self._asked: set[tuple] = set()
        self.service.outputsReady.connect(self._arrived)

    def _arrived(self, *_a):
        self.changed.emit()

    # ------------------------------------------------------------ the results a block holds
    def which_for(self, p, stage: str) -> str:
        """The result of a stage Inspect shows: the current one, or the one an Outdated stage had."""
        st = self.s.state_info(p, stage)
        return "previous" if st is not None and st.state != "done" and st.previous_key else "current"

    def key_of(self, p, stage: str, which: str = "current") -> str | None:
        st = self.s.state_info(p, stage)
        if st is None:
            return None
        return st.previous_key if which == "previous" else (st.key if st.state == "done" else None)

    def outputs(self, p, stage: str, which: str = "current"):
        """The manifest of a done result (api.StageOutputs), or None while the worker reads it (``changed`` follows)."""
        key = self.key_of(p, stage, which)
        if key is None:
            return None
        got = self.service.outputs_of(p.pid, stage, which, key)
        if got is None and (p.pid, stage, which, key) not in self._asked:
            self._asked.add((p.pid, stage, which, key))
            self.service.request_outputs(p.pid, stage, which, key)
        return got

    # ------------------------------------------------------------ the pictures
    def visuals(self, p, k: str) -> list[VisualSpec]:
        """The pictures of a block of a workflow, in the order of the tabs. The ◇ ones (not in the engine yet) are marked."""
        stages = set(self.s.all_stages(p))
        V = VisualSpec
        if k == "calibration":
            return [V("flat", "Master flat", "preview", "calibrate", "master_flat"),
                    V("calibrated", "A calibrated frame", "preview", "calibrate", "calibrated_frame"),
                    V("raw", "The same frame, raw", "preview", "inventory", "middle_frame")]
        if k == "alignement":
            return [V("quality", "Frame quality", "preview", "register", "frame_quality")]
        if k == "empilement":
            return [V("stack", "Stack (stretched for display)", "array", "stack", "drz_full", True, True)]
        if k == "astrometrie":
            out = [V("stack", "Stack (stretched for display)", "array", "stack", "drz_full", True, True)]
            if "surveys" in stages:
                out.append(V("survey", "Reference H-alpha survey", "preview", "surveys", "halpha_reference", True))
            out.append(V("objects", "Identified objects", "preview", "astrometry", "", spec=True))
            return out
        if k == "couleur":
            return [V("spectral", "Sensor response", "chart", "color"),
                    V("engine", "Engine preview", "preview", "color", "spectral_response")]
        if k == "debruitage":
            return [V("crop", "Before | after, at 100 %", "preview", "denoise", "before_after"),
                    V("denoised", "Denoised stack", "array", "denoise", "n2n_full", True, True)]
        if k == "etoiles":
            out = [V("starless", "Starless image", "array", "starless", "sl_final", True, True),
                   V("star_model", "Star model", "array", "stars", "stars_model", True, True),
                   V("psf", "Measured shape of a star", "preview", "optics", "psf")]
            if "star_net" in stages:
                out.append(V("removal", "Neural star removal", "preview", "star_net", "star_removal"))
            return out
        if k == "fond":
            if "background_poly" in stages:
                return [V("background", "Background", "preview", "background_poly", "background")]
            return [V("background", "Background", "preview", "background", "background"),
                    V("sky", "Sky model", "array", "background", "sky", True, True)]
        if k == "post":
            stage = next((x for x in ("render_prep_learned", "render_prep", "render_simple") if x in stages),
                         "render_prep")
            return [V("output", "Output preview", "preview", stage, "output" if stage == "render_simple"
                      else "nebula_layer")]
        return []

    def overlay(self, k: str, spec: VisualSpec):
        if spec.key == "crop":
            return self._denoise_overlay
        return None

    @staticmethod
    def _denoise_overlay(p: QPainter, v: Viewer):
        w, _h = v.source.size()
        for x, t in ((0, _("Before: the stack")), (w / 2 + 8, _("After: denoised"))):
            pt = v.img_to_w(x, 0)
            v._tag(p, QPointF(pt.x() + 8, pt.y() + 8), t)

    # ------------------------------------------------------------ the measurements
    def facts(self, p, k: str) -> list[tuple[str, str]]:
        """(name, value) rows of a block, from the done result's metrics (rich text, escaped). Empty until the worker has
        read the manifests; ``changed`` fires then."""
        s = self.s
        if k == "selection":
            return self._selection_facts(p)
        if k == "post":
            return self._output_facts(p)
        rows: list[tuple[str, str]] = []
        for stage, key in FACTS.get(k, []):
            if stage not in s.all_stages(p):
                continue
            out = self.outputs(p, stage, self.which_for(p, stage))
            if out is None or key not in out.metrics:
                continue
            value = out.metrics[key]
            if isinstance(value, dict) and not value:
                continue
            rows.append(metric_row(stage, key, value))
        if k == "debruitage":
            out = self.outputs(p, "denoise", self.which_for(p, "denoise"))
            if out is not None and out.params.get("iterations"):
                rows.append((_("Training"), _("{n} steps on the half-stacks").format(
                    n=fmt_int(out.params["iterations"]))))
        return rows

    def _selection_facts(self, p) -> list[tuple[str, str]]:
        s = self.s
        t, f = s.thresholds_now()
        n = s.selection_total()
        kept = s.kept_frames()
        info = s.info()
        unit = s.thresholds_unit()
        rows = [(_("Frames kept"), _("{kept} of {n}").format(kept=kept, n=n))]
        if info.get("exposure_s"):
            rows.append((_("Integration"), f"{fnum(kept * info['exposure_s'] / 3600, 2)} h"))
        rows += [(_("Minimum transparency"), _("{t} % of the best frames").format(t=f"{t:.0f}")),
                 (_("Maximum star size"), f"{fnum(f)} {unit if unit == 'px' else _('× median')}"),
                 (_("Excluded by hand"), str(len(s.frames_excluded)))]
        return rows

    def _output_facts(self, p) -> list[tuple[str, str]]:
        rows = []
        for sl in self.s.slider_infos(p):
            if sl.available:
                rows.append((html.escape(_(sl.name)), _("available")))
            else:
                rows.append((html.escape(_(sl.name)), f"<span style='color:{WARN}'>" +
                             _("greyed out: {why}").format(why=html.escape(_(sl.reason))) + "</span>"))
        return rows


__all__ = ["STEPS", "BY_KEY", "FACTS", "Steps", "fnum", "fmt_int", "kv", "metric_row", "DIM"]
