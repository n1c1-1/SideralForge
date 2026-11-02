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
                    ("astrometry", "focal_estimate_mm"), ("reference", "stars_matched"), ("surveys", "survey"),
                    ("surveys", "coverage")],
    "couleur": [("color", "stars"), ("color", "fit_agreement_pct"), ("color", "profile_agreement_pct"),
                ("color", "response_source"), ("color", "correction"), ("color", "chromaticity_residual"),
                ("color", "white_balance"), ("color", "white_balance_stars")],
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
SOLUTION_NOTE = ("Your image with the sky grid (RA / Dec) of its astrometric solution and the brightest catalogue stars "
                 "where the solution puts them: they should sit on the stars of the image. Choose the image behind "
                 "them, up to the final image.")
#: the pictures that can be behind the solution, in the order of the workflow (``Steps.picture`` makes each)
BACKGROUNDS = ("stack", "color", "denoised", "color_denoised", "result", "output")
SURVEY_NOTE = ("Not your data: a professional H-alpha map of the same field. The Gradient block compares it with your "
               "image to tell faint nebulosity from light pollution.")
OUTPUT_NOTE = "The workflow's result at the suggested render settings, as Post shows its layer. Adjust it in Post."
#: what the final image is, by the output of the workflow (``output_note``): the simple output is the Gradient removed
#: picture stretched by Post (the same data, its sky stretched to another level, so its noise shows more or less); the
#: engine's rendering is another preparation (the starless image and the stars drawn back)
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
OUTPUT_ENGINE_NOTE = ("Final image: Post's own rendering, not the Gradient removed picture: the starless image (denoised, "
                      "its gradient removed) with the stars drawn back from their model, at the suggested settings. "
                      "Cropped to the covered area, at half size. Adjust it in Post.")
OUTPUT_LEARNED_NOTE = ("Final image: Post's own rendering, not the Gradient removed picture: the starless image of the "
                       "neural star removal (denoised, its gradient removed) with the stars put back, at the suggested "
                       "settings. Cropped to the covered area, at half size. Adjust it in Post.")


def output_note(stage: str, stages: set) -> str:
    """One line on what the final image of a workflow is and which steps it holds (the Output block's note)."""
    if stage == "render_prep_learned":
        return OUTPUT_LEARNED_NOTE
    if stage == "render_prep":
        return OUTPUT_ENGINE_NOTE
    if stage != "render_simple":
        return OUTPUT_NOTE
    if "color" not in stages:
        return OUTPUT_RAW_NOTE
    if "background_poly" not in stages:
        return OUTPUT_COLOR_NOTE
    return OUTPUT_SIMPLE_DENOISED_NOTE if "denoise" in stages else OUTPUT_SIMPLE_NOTE


CALFRAMES_NOTE = "Every frame of this kind, raw, as the master is made from them."
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
DENOISE_NOTE = "Before (the stack) and after (denoised), with the same stretch on both. Drag the divider."
STACK_NOTE = "The stack output in the camera's colours, before colour calibration: its green cast is normal."
CAMERA_NOTE = "In the camera's colours, before colour calibration."
GRADIENT_NOTE = ("Your image with the gradient removed, stars included, in calibrated colours. Compare it with the "
                 "earlier steps.")
GRADIENT_FULL_NOTE = ("Your image with the gradient removed, stars included, in calibrated colours: the starless image "
                      "without its gradient (the next tab) with what the Stars block took away put back (the stars, and "
                      "the noise its filter removed): the stack itself, not denoised. Compare it with the earlier steps.")
STARLESS_BG_NOTE = ("The starless image without its gradient, in calibrated colours: what the block works on. The "
                    "stars are put back in the first tab.")
SKY_MODEL_NOTE = "What the block removed: its input minus its result."
SKY_MODEL_FULL_NOTE = ("What the block removed from the starless image (its input minus its result), in the camera's "
                       "colours.")
GRADIENT_SURFACE_NOTE = ("The smooth surface of light pollution the block fitted, channel by channel, in the camera's "
                         "colours.")
STAR_FLAT_NOTE = "The correction of the flat field measured on the stars (1: no change)."
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
        """The tabs of a block of a workflow, in their order: pictures (shown in the common viewer), charts, tables,
        sequences of frames. The ◇ ones (not in the engine yet) are marked."""
        stages = set(self.s.all_stages(p))
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
                    V("track", "Tracking x / y (Lissajous)", "regchart", "register", "track"),
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
            # first the user's own image with its solution, over the image of the workflow the user chooses (no compare:
            # the solution is drawn over one picture); the survey is another picture, with its role said, compared
            # with the stack on the same sky (it is reprojected on the stack grid)
            out = [V("solution", "Stack and solution", "array", "stack", "drz_full", True, True, sky=True,
                     note=SOLUTION_NOTE, camera=True, backgrounds=True)]
            if "surveys" in stages:
                out.append(V("survey", "Reference H-alpha survey", "preview", "surveys", "halpha_reference", True,
                             note=SURVEY_NOTE, camera=False, compare=("stack",)))
            out.append(V("objects", "Identified objects", "preview", "astrometry", "", spec=True))
            return out
        if k == "couleur":
            return [V("colors", "Before / after", "color", "color", "", True, True, camera=False,
                      base=("stack", "drz_full"), compare=("stack",), open_compare=True, note=COLOR_NOTE,
                      shown="After: colour-calibrated", compare_names=("Before: stack output, camera colours",)),
                    V("spectral", "Spectral response", "chart", "color", "0"),
                    V("lines", "Colour of the lines", "chart", "color", "1"),
                    V("fit", "Star colours", "colorfit", "color")]
        if k == "debruitage":
            return [V("denoised", "Denoised stack", "array", "denoise", "n2n_full", True, True, camera=True,
                      compare=("stack", "half_a"), open_compare=True, same_stretch=True, note=DENOISE_NOTE)]
        if k == "etoiles":
            out = [V("starless", "Starless image", "array", "starless", "sl_final", True, True, camera=True,
                     compare=("denoised", "stack")),
                   V("star_model", "Star model", "array", "stars", "stars_model", True, True, camera=True,
                     compare=("stack",)),
                   V("psf", "Measured shape of a star", "preview", "optics", "psf", camera=False)]
            if "star_net" in stages:
                out.append(V("removal", "Neural star removal", "preview", "star_net", "star_removal", camera=False))
            return out
        if k == "fond":
            # the main picture is always the whole image with the gradient removed, stars included, in calibrated
            # colours: the H-alpha background works on the starless image, its result gets the stars back
            # (services/inspect.full_gradient_image); the simple one's result is the whole image already
            st = "background" if "background" in stages else "background_poly"
            full = st == "background"
            cmp_ = ("color_denoised", "color", "stack")
            out = [V("result", "Gradient removed", "gradient", st, "", True, True, camera=False, compare=cmp_,
                     note=GRADIENT_FULL_NOTE if full else GRADIENT_NOTE)]
            if full:
                out += [V("starless_bg", "Starless, gradient removed", "calibrated", "background", "", True, True,
                          camera=False, base=("background", "astro"), compare=("starless_calibrated",),
                          note=STARLESS_BG_NOTE),
                        V("removed", "Sky model", "removed", st, "", True, True, camera=True,
                          note=SKY_MODEL_FULL_NOTE, display="model"),
                        V("sky", "Gradient", "array", "background", "sky", True, True, camera=True,
                          note=GRADIENT_SURFACE_NOTE, display="model"),
                        V("star_flat", "Star flat", "array", "background", "flat", True, True, camera=False,
                          note=STAR_FLAT_NOTE, display="model")]
            else:
                out.append(V("removed", "Sky model", "removed", st, "", True, True, camera=False, note=SKY_MODEL_NOTE,
                             display="model"))
            return out
        if k == "post":
            return [dataclasses.replace(self.picture(p, "output"), compare=("result", "color"))]
        return []

    def render_stage(self, p) -> str:
        """The stage whose result is the final image of the workflow."""
        stages = set(self.s.all_stages(p))
        return next((x for x in ("render_prep_learned", "render_prep", "render_simple") if x in stages), "render_prep")

    def picture(self, p, key: str) -> VisualSpec | None:
        """A picture of the workflow another one is compared with (the previous steps), or None when the workflow
        does not make it."""
        stages = set(self.s.all_stages(p))
        V = VisualSpec
        if key == "stack":
            return V("stack", "Stack", "array", "stack", "drz_full", True, True, camera=True)
        if key == "survey" and "surveys" in stages:
            return V("survey", "Reference H-alpha survey", "preview", "surveys", "halpha_reference", True,
                     camera=False)
        if key in ("half_a", "half_b"):
            return V(key, "Half-stack A" if key == "half_a" else "Half-stack B", "array", "stack",
                     "drz_A" if key == "half_a" else "drz_B", True, True, camera=True)
        if key == "denoised" and "denoise" in stages:
            return V("denoised", "Denoised stack", "array", "denoise", "n2n_full", True, True, camera=True)
        if key == "starless" and "starless" in stages:
            return V("starless", "Starless image", "array", "starless", "sl_final", True, True, camera=True)
        if key == "starless_calibrated" and "starless" in stages and "color" in stages:
            return V("starless_calibrated", "Starless image, before the gradient removal", "calibrated", "starless", "",
                     True, True, camera=False, base=("starless", "sl_final"))
        if key == "color" and "color" in stages:
            return V("color", "Colour-calibrated stack", "color", "color", "", True, True, camera=False,
                     base=("stack", "drz_full"))
        if key == "color_denoised" and "color" in stages and "denoise" in stages:
            return V("color_denoised", "Colour-calibrated denoised stack", "color", "color", "", True, True,
                     camera=False, base=("denoise", "n2n_full"))
        if key == "result":
            st = "background" if "background" in stages else ("background_poly" if "background_poly" in stages
                                                              else None)
            if st is not None:
                return V("result", "Gradient removed", "gradient", st, "", True, True, camera=False)
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
            if spec.stage == "background":
                return ["background", "starless"] + (["optics", "color"] if spec.kind == "gradient" else [])
            stages = set(self.s.all_stages(p))
            return ["background_poly", "color", "denoise" if "denoise" in stages else "stack"]
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
        # option the user ticks, never the default (on by default, the stack would look colour-calibrated)
        return ViewImage(src, name, key=ident, wcs=wcs, camera=spec.camera, units=spec.units,
                         neutral=False if spec.camera and spec.kind != "preview" else None,
                         header={"HISTORY": [f"astro_gpu: {spec.caption} of workflow {p.name}"]}, kind=spec.display,
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
        if k == "couleur" and "color" in s.all_stages(p):
            self._correction_fact(p, rows)
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

    def _correction_fact(self, p, rows: list) -> None:
        """A colour result made before the colour stage named its correction (``correction``): a colour matrix,
        unless no sensitivity curves were trusted (one factor per colour). Put after "Sensitivity curves"."""
        out = self.outputs(p, "color", self.which_for(p, "color"))
        if out is None or "correction" in out.metrics or "response_source" not in out.metrics:
            return
        # stages/color.py: PER_COLOUR and APPLIED (not imported: the engine's stage modules stay out of the client)
        per_colour = out.metrics["response_source"] == "one factor per colour (no reliable sensitivity curves)"
        v = "one factor per colour" if per_colour else "colour matrix"
        lab = html.escape(texts.metric_label("response_source", language()))
        at = next((i + 1 for i, r in enumerate(rows) if r[0] == lab), len(rows))
        rows.insert(at, metric_row("color", "correction", v))

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
