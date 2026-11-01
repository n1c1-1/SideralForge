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

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen, QPolygonF

from .. import texts
from .blocks import BY_KEY, STEPS, block_state, fmt_int  # noqa: F401  (the views import them from here)
from .i18n import _, hours_text, language, number
from .services.inspect import SkyData, VisualSpec, service_of
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


#: one line under the tabs of a picture whose role is not obvious (English, translated where shown)
SOLUTION_NOTE = ("Your stack with the sky grid (RA / Dec) of its astrometric solution and the brightest catalogue stars "
                 "where the solution puts them: they should sit on the stars of the image.")
SURVEY_NOTE = ("Not your data: a professional H-alpha map of the same field. The Gradient block compares it with your "
               "image to tell faint nebulosity from light pollution.")
OUTPUT_NOTE = "The workflow's result at the suggested render settings, as Post shows its layer. Adjust it in Post."


def fnum(x, d=1) -> str:
    return number(x, d)


class SkyOverlay:
    """The astrometric solution drawn over a picture of the stack, in the viewer's painter: the RA / Dec grid with its
    labels, the catalogue stars where the solution puts them, and the named objects (``SkyData.objects``, filled by the
    annotation package: nothing else to wire). ``show``: {"grid", "stars", "objects"} -> on or off."""

    GRID = QColor(94, 200, 224, 130)
    STAR = QColor(255, 210, 74, 215)
    OBJECT = QColor(255, 255, 255, 230)

    def __init__(self, sky: SkyData, show: dict):
        self.sky, self.show = sky, show

    def __call__(self, p: QPainter, v: Viewer):
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        try:
            if self.show.get("grid", True):
                self.grid(p, v)
            if self.show.get("stars", True):
                self.stars(p, v)
            if self.show.get("objects", True):
                self.objects(p, v)
        finally:
            p.restore()

    @staticmethod
    def _to_widget(pts: np.ndarray, v: Viewer) -> np.ndarray:
        return (pts.astype(np.float64) - (v.cx, v.cy)) * v.scale + (v.width() / 2, v.height() / 2)

    def grid(self, p: QPainter, v: Viewer):
        p.setPen(QPen(self.GRID, 1.0))
        p.setBrush(Qt.NoBrush)
        labels = []
        for line in self.sky.lines:
            for piece in line.pieces:
                w = self._to_widget(piece, v)
                p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in w]))
                at = self._label_at(w, v, line.kind)
                if at is not None:
                    labels.append((line.label, at, line.kind))
        f = p.font()
        f.setPointSizeF(8.5)
        p.setFont(f)
        fm = QFontMetricsF(f)
        taken: list[QRectF] = []
        for text, (x, y), kind in labels:
            tw, th = fm.horizontalAdvance(text) + 8, fm.height() + 2
            r = QRectF(x + 3, y - th / 2, tw, th) if kind == "dec" else QRectF(x - tw / 2, y + 3, tw, th)
            r.translate(max(0.0, 2 - r.left()) + min(0.0, v.width() - 2 - r.right()),
                        max(0.0, 2 - r.top()) + min(0.0, v.height() - 2 - r.bottom()))      # all of it inside the viewer
            if any(r.intersects(t) for t in taken):
                continue
            taken.append(r)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(15, 17, 21, 190))
            p.drawRoundedRect(r, 3, 3)
            p.setPen(QColor(150, 220, 236))
            p.drawText(r, Qt.AlignCenter, text)

    @staticmethod
    def _label_at(w: np.ndarray, v: Viewer, kind: str):
        """Where a grid line enters the viewer, on its left edge for a parallel and its top edge for a meridian (else
        the other edges): the point the label hangs from. None when the line does not cross the viewer."""
        W, H, m = v.width(), v.height(), 6.0
        edges = (("x", m), ("y", m), ("x", W - m), ("y", H - m)) if kind == "dec" else \
            (("y", m), ("x", m), ("y", H - m), ("x", W - m))
        for axis, c in edges:
            a = w[:, 0] if axis == "x" else w[:, 1]
            o = w[:, 1] if axis == "x" else w[:, 0]
            lim = H if axis == "x" else W
            s = np.flatnonzero((a[:-1] - c) * (a[1:] - c) <= 0)
            for i in s:
                d = a[i + 1] - a[i]
                t = 0.0 if d == 0 else (c - a[i]) / d
                other = o[i] + t * (o[i + 1] - o[i])
                if m <= other <= lim - m:
                    return (c, other) if axis == "x" else (other, c)
        return None

    def stars(self, p: QPainter, v: Viewer):
        s = self.sky.stars
        if s is None or not len(s):
            return
        w = self._to_widget(s[:, :2], v)
        W, H = v.width(), v.height()
        grow = 3.0 * min(v.scale, 1.0)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(self.STAR, 1.2))
        for (x, y), g in zip(w, s[:, 2]):
            if -12 <= x <= W + 12 and -12 <= y <= H + 12:
                r = min(max(2.0 + (10.0 - float(g)) * 0.9, 2.0), 9.0) + grow
                p.drawEllipse(QPointF(x, y), r, r)

    def objects(self, p: QPainter, v: Viewer):
        if not self.sky.objects:
            return
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(self.OBJECT, 1.4))
        for x, y, name, _kind in self.sky.objects:
            pt = v.img_to_w(x, y)
            p.drawEllipse(pt, 7, 7)
            v._tag(p, QPointF(pt.x() + 10, pt.y() - 8), str(name))


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
            # first the user's own image with its solution; the survey is another picture, with its role said
            out = [V("solution", "Stack and solution", "array", "stack", "drz_full", True, True, sky=True,
                     note=SOLUTION_NOTE)]
            if "surveys" in stages:
                out.append(V("survey", "Reference H-alpha survey", "preview", "surveys", "halpha_reference", True,
                             note=SURVEY_NOTE))
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
            # the final image, not an engine preview: rendered with the project's render worker, as Post shows it
            return [V("output", "Final image", "render", stage, note=OUTPUT_NOTE)]
        return []

    def overlay(self, k: str, spec: VisualSpec, im=None, show: dict | None = None):
        """What is drawn over a picture: the labels of the denoise crop, the sky grid and the catalogue stars of the
        solution (``show`` says which are on, the view owns it)."""
        if spec.key == "crop":
            return self._denoise_overlay
        if spec.sky and im is not None and im.sky is not None:
            return SkyOverlay(im.sky, show if show is not None else {})
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
            rows.append((_("Integration"), hours_text(kept * info["exposure_s"] / 3600)))
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
