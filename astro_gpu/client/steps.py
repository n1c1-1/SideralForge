"""What a block shows in Inspect: its pictures and its measurements.

The block table itself (title, stages, what it does) is ``blocks.STEPS``; the state of a block is ``blocks.block_state``.
Here: which pictures a block has (``visuals``: arrays of the done results and the engine's previews), and its
measurements (``facts``, read from the manifests by the worker and formatted with the metric catalogue of
``astro_gpu/texts.py``, so that a value reads the same everywhere and in both languages).

``Steps`` is a QObject: ``facts`` answers from what the worker has delivered and asks for the rest; ``changed`` fires when
more has arrived (the panel is built again).
"""
from __future__ import annotations

import dataclasses
import html

import numpy as np
from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen

from .. import texts
from . import APP_NAME
from .blocks import BY_KEY, STEPS, fmt_int
from .i18n import _, hours_text, language, number
from .imageview import ViewImage
from .services.inspect import SkyData, VisualSpec, service_of
from .widgets import WARN

Viewer = object                    # what an overlay draws on: the common viewer's canvas (cx, cy, scale, img_to_w)


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
                    ("astrometry", "focal_estimate_mm"), ("reference", "stars_matched")],
    "couleur": [("color", "stars"), ("color", "white_balance"), ("color", "white_balance_stars"),
                ("color", "chromaticity_residual")],
    "debruitage": [("denoise", "noise_before"), ("denoise", "noise_after"), ("denoise", "noise_reduction"),
                   ("denoise", "noise_correlation"), ("denoise", "adaptation_steps")],
    "fond": [("gradient", "gradient_pct"), ("gradient", "samples_used")],
    "post": [("render_simple", "crop"), ("render_simple", "sky_source")],
}


#: one line under the tabs of a picture whose role is not obvious (English, translated where shown)
SOLUTION_NOTE = ("Your image with the sky grid (RA / Dec) of its astrometric solution and the brightest catalogue stars "
                 "where the solution puts them: they should sit on the stars of the image. Choose the image behind "
                 "them, up to the final image.")
#: the pictures that can be behind the solution, in the order of the workflow (``Steps.picture`` makes each)
BACKGROUNDS = ("stack", "color", "denoised", "color_denoised", "result", "output")
OUTPUT_NOTE = "The workflow's result at the suggested render settings, as Post shows its layer. Adjust it in Post."
#: what the final image is, by the output of the workflow (``output_note``): the Gradient removed picture stretched by
#: Post (the same data, its sky stretched to another level, so its noise shows more or less)
OUTPUT_SIMPLE_NOTE = ("Final image = Gradient removed, then the steps of Post: its stretch (the suggested settings) and the "
                      "filters you add there. These steps are in Post, not in the graph. The stretch is already applied "
                      "(values 0 to 255), so Same stretch is not possible against Gradient removed: compare them by eye. "
                      "Cropped to the area every frame covers, at half size.")
OUTPUT_SIMPLE_DENOISED_NOTE = ("Final image = Gradient removed (of the denoised stack), then the steps of Post: its stretch "
                               "(the suggested settings) and the filters you add there. These steps are in Post, not in the "
                               "graph. The stretch is already applied (values 0 to 255), so Same stretch is not possible "
                               "against Gradient removed: compare them by eye. Cropped to the area every frame covers, at half "
                               "size.")
OUTPUT_COLOR_NOTE = ("Final image: the colour-calibrated stack with Post's suggested stretch. Cropped to the covered "
                     "area, at half size. Adjust it in Post.")
OUTPUT_RAW_NOTE = ("Final image: the stack in the camera's colours, the sky of each channel levelled, with Post's "
                   "suggested stretch. Cropped to the covered area, at half size. Adjust it in Post.")
STARLESS_NOTE = ("The image without its stars, compared with the image with them (the same stretch: drag the divider). "
                 "The Stars block's two images go to Post: combined by a Combine stars & background layers block, or as they "
                 "are.")
STARS_NOTE = ("The stars alone: the image minus its starless image, stretched on their own. They go back on the "
              "starless image with the blend mode of their Combine stars & background layers block, or as a layer of Post.")


def output_note(stage: str, stages: set) -> str:
    """One line on what the final image of a workflow is and which steps it holds (the Output block's note)."""
    if stage != "render_simple":
        return OUTPUT_NOTE
    if "color" not in stages:
        return OUTPUT_RAW_NOTE
    if "gradient" not in stages:
        return OUTPUT_COLOR_NOTE
    return OUTPUT_SIMPLE_DENOISED_NOTE if "denoise" in stages else OUTPUT_SIMPLE_NOTE


CALFRAMES_NOTE = "The raw frames this master is made from."
#: the Color block's diagnostics of the camera (core/spectral.py): they change no colour
SPECTRAL_NOTE = ("An approximation of the sensor's response in red, green and blue, rebuilt from the Gaia spectra of the "
                 "field's stars and what the camera measured of them. Star spectra are broad: the curves are only "
                 "approximate, at a narrow line such as H-alpha in particular. A diagnostic of the camera: the colours "
                 "of the image come from the white balance, not from these curves. Export: the chart's menu (CSV, PNG).")
LINES_NOTE = ("The colour the camera gives to each wavelength, from the approximate curves: each channel's share of the "
              "signal of a pure line. The emission lines of the nebulae are marked.")
CALIBRATED_NOTE = ("Each light after calibration: the master dark subtracted, divided by the master flat, the "
                   "defective pixels left empty.")
REFERENCE_NOTE = "The frame every other frame is aligned on, calibrated."
REJECTION_NOTE = "Where the stack rejected pixels: satellite trails, planes, cosmic rays, hot pixels."
COVERAGE_NOTE = "How many frames (weighted) cover each pixel of the stack: the edges and the dithering show here."
COVERAGE_DRIZZLE_NOTE = ("With the drizzle, red and blue are one pixel in four of the sensor: they need enough dithered "
                         "frames to cover the grid evenly, or holes and a grid pattern show.")
COLOR_NOTE = ("Before, left of the divider: the stack output as it is, in the camera's colours. After, right: the "
              "same stack colour-calibrated (the colour matrix measured on the Gaia stars, the sky set grey). Drag the "
              "divider.")
DENOISE_NOTE = ("Before (the stack, its gradient removed: what the block reads) and after (denoised), with the same "
                "stretch on both. Drag the divider.")
STACK_NOTE = "The stack output in the camera's colours, before colour calibration: its green cast is normal."
CAMERA_NOTE = "In the camera's colours, before colour calibration."
GRADIENT_NOTE = ("The stack with its sky gradient removed, in the camera's colours, as every later block reads it (the "
                 "colour calibration comes next). The sky keeps its level. Compare it with the stack.")
SKY_MODEL_NOTE = ("What the block removed: the sky model of its method, in the camera's colours. A nebula or a galaxy "
                  "that shows here was taken for sky.")
#: the line after the note of a model (display "model"): what its colours mean
MODEL_COLOURS_NOTE = ("Shown with the channels linked, from zero to its highest value: the colours are the model's own, "
                      "not stretched apart. Untick Linked channels to see the shape of each channel.")


class SkyOverlay:
    """The astrometric solution drawn over a picture of the stack, in the viewer's painter: the RA / Dec grid with its
    labels, the catalogue stars where the solution puts them, and the named objects outlined at their size
    (``SkyData.objects``, ``annotate.paint_objects``). ``show``: {"grid", "stars", "objects"} -> on or off."""

    STAR = QColor(255, 210, 74, 215)

    def __init__(self, sky: SkyData, show: dict):
        self.sky, self.show = sky, show

    def __call__(self, p: QPainter, v: Viewer):
        from .annotate import paint
        paint(p, self.sky, v, {"grid": self.show.get("grid", True), "objects": self.show.get("objects", True)})
        if self.show.get("stars", True):
            p.save()
            p.setRenderHint(QPainter.Antialiasing)
            try:
                self.stars(p, v)
            finally:
                p.restore()

    @staticmethod
    def _to_widget(pts: np.ndarray, v: Viewer) -> np.ndarray:
        return (pts.astype(np.float64) - (v.cx, v.cy)) * v.scale + (v.width() / 2, v.height() / 2)

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
    def _instance(self, p, stage: str) -> tuple[str, str]:
        """(the stage Inspect reads, its session) for the block shown: a session's block ("calibration@2") reads that
        session's instance of the stage (issue #43); any other block, the stage."""
        sid = str(getattr(self.s, "sel_step", "") or "").partition("@")[2]
        if sid and f"{stage}@{sid}" in self.s.all_stages(p):
            return f"{stage}@{sid}", sid
        return stage, ""

    def which_for(self, p, stage: str) -> str:
        """The result of a stage Inspect shows: the current one, or the one an Outdated stage had ("…@2": of that
        session)."""
        inst, sid = self._instance(p, stage)
        st = self.s.state_info(p, inst)
        which = "previous" if st is not None and st.state != "done" and st.previous_key else "current"
        return f"{which}@{sid}" if sid else which

    def key_of(self, p, stage: str, which: str = "current") -> str | None:
        which, _at, sid = which.partition("@")
        st = self.s.state_info(p, f"{stage}@{sid}" if sid else stage)
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
        """The tabs of a block of a workflow, in their order: pictures (shown in the common viewer), charts, tables,
        sequences of frames. The ◇ ones (not in the engine yet) are marked."""
        V = VisualSpec
        if k == "calibration":
            out = [V("masters", "Masters", "masters", "calibrate")]
            counts = {"darks": 0, "flats": 0, "bias": 0, "flat_darks": 0}
            for ses in self.s.sessions:
                for kind in counts:
                    counts[kind] += int(ses.get("dark_flats" if kind == "flat_darks" else kind) or 0)
            for kind, cap in (("darks", "Darks"), ("flats", "Flats"), ("bias", "Bias"), ("flat_darks", "Dark flats")):
                if counts[kind] or kind in ("darks", "flats"):       # no darks or flats: said there, with Add a folder
                    out.append(V(kind, cap, "calframes", "calibrate", kind, note=CALFRAMES_NOTE))
            out.append(V("calibrated", "Calibrated lights", "callights", "calibrate", note=CALIBRATED_NOTE))
            return out
        if k == "alignement":
            return [V("reference", "Reference frame", "regref", "register", note=REFERENCE_NOTE),
                    V("quality", "Quality", "regchart", "register", "quality"),
                    V("drift", "Drift", "regchart", "register", "drift"),
                    V("track", "Drift path (x / y)", "regchart", "register", "track"),
                    V("distribution", "Distribution", "regchart", "register", "distribution"),
                    V("table", "Frames", "regtable", "register")]
        if k == "empilement":
            return [V("stack", "Stack", "array", "stack", "drz_full", True, True, camera=True,
                      compare=("calibrated_ref", "half_a", "half_b"), note=STACK_NOTE),
                    V("rejection", "Rejection map", "rejection", "stack", note=REJECTION_NOTE),
                    V("coverage", "Coverage", "coverage", "stack", "drz_weight", True, True, camera=False, channel=1,
                      units="", note=COVERAGE_NOTE),
                    V("weights", "Weights", "weights", "stack")]
        if k == "astrometrie":
            # the user's own image with its solution, over the image of the workflow the user chooses (no compare: the
            # solution is drawn over one picture); the identified objects in Spec mode
            out = [V("solution", "Stack and solution", "array", "stack", "drz_full", True, True, sky=True,
                     note=SOLUTION_NOTE, camera=True, backgrounds=True)]
            out.append(V("objects", "Identified objects", "preview", "astrometry", "", spec=True))
            return out
        if k == "couleur":
            return [V("colors", "Before / after", "color", "color", "", True, True, camera=False,
                      base=("stack", "drz_full"), compare=("stack",), open_compare=True, note=COLOR_NOTE,
                      shown="After: colour-calibrated", compare_names=("Before: stack output, camera colours",)),
                    V("spectral", "Approximate spectral response", "chart", "color", "0", note=SPECTRAL_NOTE),
                    V("lines", "Colour of the lines", "chart", "color", "1", note=LINES_NOTE),
                    V("fit", "Star colours", "colorfit", "color")]
        if k == "debruitage":
            return [V("denoised", "Denoised stack", "array", "denoise", "n2n_full", True, True, camera=True,
                      neutral=True, compare=("result", "stack", "half_a"), open_compare=True, same_stretch=True,
                      note=DENOISE_NOTE)]
        if k == "etoiles":
            return [V("starless", "Starless image", "array", "star_removal", "starless", True, True, camera=False,
                      compare=("whole",), open_compare=True, same_stretch=True, note=STARLESS_NOTE,
                      compare_names=("With its stars",)),
                    V("stars", "Stars alone", "array", "star_removal", "stars", True, True, camera=False,
                      compare=("whole",), same_stretch=True, note=STARS_NOTE, compare_names=("With its stars",))]
        if k == "fond":
            # the block's own input and output, in the camera's colours: it works before the colour calibration
            return [V("result", "Gradient removed", "gradient", "gradient", "", True, True, camera=True, neutral=True,
                      compare=("stack",), open_compare=True, same_stretch=True, note=GRADIENT_NOTE),
                    V("removed", "Sky model", "removed", "gradient", "", True, True, camera=True, note=SKY_MODEL_NOTE,
                      display="model")]
        if k == "post":
            return [dataclasses.replace(self.picture(p, "output"), compare=("result", "color"))]
        return []

    def render_stage(self, p) -> str:
        """The stage whose result is the final image of the workflow (the Output block's)."""
        return "render_simple"

    def picture(self, p, key: str) -> VisualSpec | None:
        """A picture of the workflow another one is compared with (the previous steps), or None when the workflow
        does not make it."""
        stages = set(self.s.all_stages(p))
        V = VisualSpec
        if key == "stack":
            return V("stack", "Stack", "array", "stack", "drz_full", True, True, camera=True)
        if key in ("half_a", "half_b"):
            return V(key, "Half-stack A" if key == "half_a" else "Half-stack B", "array", "stack",
                     "drz_A" if key == "half_a" else "drz_B", True, True, camera=True)
        if key == "denoised" and "denoise" in stages:
            return V("denoised", "Denoised stack", "array", "denoise", "n2n_full", True, True, camera=True)
        if key == "whole" and "render_simple" in stages:
            return V("whole", "Image with its stars", "array", "render_simple", "lin", True, True, camera=False)
        if key == "color" and "color" in stages:
            return V("color", "Colour-calibrated stack", "color", "color", "", True, True, camera=False,
                     base=("stack", "drz_full"))
        if key == "color_denoised" and "color" in stages and "denoise" in stages:
            return V("color_denoised", "Colour-calibrated denoised stack", "color", "color", "", True, True,
                     camera=False, base=("denoise", "n2n_full"))
        if key == "result" and "gradient" in stages:
            return V("result", "Gradient removed", "gradient", "gradient", "", True, True, camera=True)
        if key == "output":
            # the final image, not an engine preview: rendered with the project's render worker, as Post shows it
            stage = self.render_stage(p)
            return V("output", "Final image", "render", stage, note=output_note(stage, stages), camera=False)
        return None

    def backgrounds(self, p) -> list[VisualSpec]:
        """The pictures the solution of Astrometry can be drawn over, in the order of the workflow: the ones that are
        computed, the stack first (its default). A picture made from an Outdated result (its previous one, as the rest
        of Inspect shows it) is listed too, marked ``previous`` (after an engine update every stage may be Outdated)."""
        out = []
        for key in BACKGROUNDS:
            sp = self.picture(p, key)
            if sp is None:
                continue
            stages = self.stages_of(p, sp)
            which = [self.which_for(p, x) for x in stages]
            if all(self.key_of(p, x, w) for x, w in zip(stages, which)):
                out.append(dataclasses.replace(sp, previous=True) if "previous" in which else sp)
        return out

    def stages_of(self, p, spec: VisualSpec) -> list[str]:
        """The results a picture is made of (its key is theirs)."""
        if spec.kind == "color":
            return ["color", spec.base[0]]
        if spec.kind == "calibrated":
            return ["color", spec.base[0]]
        if spec.kind in ("gradient", "removed"):
            return ["gradient", "stack"]
        return [spec.stage]

    def view_image(self, p, spec: VisualSpec, prefix: str = "") -> ViewImage | None:
        """The picture of a workflow as the common viewer takes it: its source is read in the viewer's thread, its
        identity is the keys of the results it is made of. None when one of them is not computed."""
        from .services import inspect as si
        P = self.s.p
        stages = self.stages_of(p, spec)
        which = {x: self.which_for(p, x) for x in stages}
        keys = [self.key_of(p, x, which[x]) for x in stages]
        if not all(keys):
            return None
        if spec.kind == "array":
            src = si.array_source(P, p.pid, spec.stage, spec.name, which, spec.channel)
        elif spec.kind == "preview":
            src = si.preview_source(P, p.pid, spec.stage, spec.name, which)
        elif spec.kind == "color":
            src = si.color_source(P, p.pid, spec.base, which)
        elif spec.kind == "gradient":
            src = si.gradient_source(P, p.pid, spec.stage, which)
        elif spec.kind == "removed":
            src = si.removed_source(P, p.pid, spec.stage, which)
        elif spec.kind == "calibrated":
            src = si.calibrated_source(P, p.pid, spec.base, which)
        else:
            return None
        wcs = None
        if spec.grid and "astrometry" in self.s.all_stages(p):
            aw = self.which_for(p, "astrometry")
            if self.key_of(p, "astrometry", aw):
                # a preview is a reduced copy of the stack grid: the solution scaled to it
                wcs = si.preview_wcs_source(P, p.pid, aw, spec.stage, spec.name, which) if spec.kind == "preview" \
                    else si.wcs_source(P, p.pid, aw)
        # where it sits for the compare: every picture on the stack grid (and a reduced copy of it, an engine preview)
        # is placed on the registration's reference frame at the drizzle scale (services/inspect.stack_place)
        place = None
        if spec.grid and "stack" in self.s.all_stages(p) and self.key_of(p, "stack", self.which_for(p, "stack")):
            place = si.stack_place(P, p.pid, self.which_for(p, "stack"))
        ident = ("inspect", spec.kind, spec.name, spec.channel, spec.base, tuple(zip(stages, keys)))
        name = (prefix + "  ·  " if prefix else "") + _(spec.shown or spec.caption)
        # an array in the camera's colours is shown as it is (its green cast and all): Neutral sky is a display
        # option the user ticks (on by default, the stack would look colour-calibrated); the blocks that work on the
        # sky before the colour calibration (Gradient, Denoise) open with it on, their effect hidden by the cast
        # otherwise
        neutral = spec.neutral if spec.neutral is not None else (False if spec.camera and spec.kind != "preview"
                                                                 else None)
        return ViewImage(src, name, key=ident, wcs=wcs, camera=spec.camera, units=spec.units, neutral=neutral,
                         header={"HISTORY": [f"{APP_NAME}: {spec.caption} of workflow {p.name}"]}, kind=spec.display,
                         place=place)

    def overlay(self, k: str, spec: VisualSpec, sky: SkyData | None = None, show: dict | None = None):
        """What is drawn over a picture: the sky grid and the catalogue stars of the solution (``show`` says which
        are on, the view owns it)."""
        if spec.sky and sky is not None:
            return SkyOverlay(sky, show if show is not None else {})
        return None

    # ------------------------------------------------------------ the measurements
    def facts(self, p, k: str) -> list[tuple[str, str]]:
        """(name, value) rows of a block, from the done result's metrics (rich text, escaped). Empty until the worker has
        read the manifests; ``changed`` fires then."""
        s = self.s
        if k == "selection":
            return self._selection_facts(p)
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
        if k == "empilement" and "stack" in s.all_stages(p):
            m = self._method_fact(p)
            if m is not None:
                rows.insert(0, m)
        if k == "debruitage":
            out = self.outputs(p, "denoise", self.which_for(p, "denoise"))
            if out is not None and out.params.get("iterations"):
                rows.append((_("Training"), _("{n} steps on the half-stacks").format(
                    n=fmt_int(out.params["iterations"]))))
        if k == "post":
            rows += self._output_facts(p)
        return rows

    def _method_fact(self, p) -> tuple[str, str] | None:
        """("Method", the stacking method of the done result and its main settings), from its manifest; a stack made
        before the methods existed is a drizzle."""
        from .settings import method_text
        out = self.outputs(p, "stack", self.which_for(p, "stack"))
        if out is None:
            return None
        prm = out.params or {}
        m = prm.get("method") or "drizzle"
        lang = language()

        def ch(k):
            return html.escape(texts.choice_label(str(prm[k]), lang)) if prm.get(k) is not None else None
        bits = [html.escape(method_text(m)[0])]
        if m == "drizzle":
            if prm.get("scale") is not None:
                bits.append(_("scale {v}×").format(v=number(float(prm["scale"]), 2).rstrip("0").rstrip(",.")))
            if prm.get("pixfrac") is not None:
                bits.append(_("drop {v}").format(v=number(float(prm["pixfrac"]), 2).rstrip("0").rstrip(",.")))
        else:
            bits += [b for b in (ch("classic_debayer"), ch("classic_interpolation")) if b]
        return _("Method"), " · ".join(bits)

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
        rows.append((_("Rule"), texts.choice_label(s.selection_rule(p), language())))
        if s.thresholds_active(p):                           # the thresholds act only with the "thresholds" rule
            rows += [(_("Minimum transparency"), _("{t} % of the best frames").format(t=f"{t:.0f}")),
                     (_("Maximum star size"), f"{number(f)} {unit if unit == 'px' else _('× median')}")]
        rows.append((_("Excluded by hand"), str(len(s.frames_excluded))))
        return rows

    def _output_facts(self, p) -> list[tuple[str, str]]:
        """Besides the measurements of the Output result: the sliders of Post this workflow cannot offer, and why (the
        ones it offers say nothing: they are in Post)."""
        rows = []
        infos = self.s.slider_infos(p)
        n_ok = sum(1 for sl in infos if sl.available)
        if infos:
            rows.append((_("Post sliders"), _("{n} of {total}").format(n=n_ok, total=len(infos))))
        for sl in infos:
            if not sl.available:
                rows.append((html.escape(_(sl.name)), f"<span style='color:{WARN}'>" +
                             _("greyed out: {why}").format(why=html.escape(_(sl.reason))) + "</span>"))
        return rows


__all__ = ["STEPS", "BY_KEY", "FACTS", "Steps", "fmt_int", "metric_row"]
