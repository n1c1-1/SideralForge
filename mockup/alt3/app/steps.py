"""The steps of a workflow as both the Workflow and the Results views show them: what each one is, its state, its
summary, its measurements and its images. Shared so that a step looks the same in the definition and in the run.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QLabel, QWidget

from i18n import _, language
from model import DATA, TRUNK, PathDef, Session, hm
from viewer import StaticSource, Viewer
from widgets import OK, SHARED, WARN

DIM = "#8b93a1"


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", " ")


def fnum(x, d=1) -> str:
    s = f"{x:.{d}f}"
    return s.replace(".", ",") if language() == "fr" else s


STEPS = [
    dict(key="calibration", phase=1, title="Calibration", kind="fixed", stages=["inventory", "calibrate"],
         what="Dark and flat masters, then each light calibrated: thermal signal, vignetting and defective pixels "
              "removed."),
    dict(key="alignement", phase=2, title="Registration", kind="fixed", stages=["register"],
         what="Measures the stars of each frame (size, transparency) and aligns all frames on the sharpest one."),
    dict(key="selection", phase=2, title="Selection", kind="fixed", stages=[], run=["register"],
         what="Which frames go into the stack: by star size and transparency thresholds, or one by one."),
    dict(key="empilement", phase=3, title="Stacking", kind="fixed", stages=["stack"],
         what="Stacks the frames straight from the sensor's color pixels (drizzle without debayering) and rejects "
              "trails and outlier pixels."),
    dict(key="demi_piles", phase=3, title="Half-stacks", kind="auto", stages=[], run=["stack"],
         what="Two independent stacks of about 360 frames each: their difference measures the noise, which lets "
              "Denoise work without inventing anything. Computed during stacking, at no extra time."),
    dict(key="astrometrie", phase=4, title="Astrometry", kind="fixed", stages=["astrometry", "catalogs", "surveys"],
         what="Finds where the image points on the sky by matching its stars to Gaia, then fetches the catalogue "
              "stars (and the H-alpha survey if the background uses it)."),
    dict(key="couleur", phase=4, title="Color", kind="fixed", stages=["color"],
         what="Measures the sensor's response on the Gaia spectra of the field stars, then the matrix to the true "
              "sky colors."),
    dict(key="debruitage", phase=4, title="Denoise", kind="tool", tool="debruitage", stages=["denoise"],
         what="A network learns on the two half-stacks to predict one from the other: it removes the noise without "
              "inventing anything."),
    dict(key="etoiles", phase=4, title="Stars", kind="tool", tool="etoiles", stages=["optics", "stars"],
         what="Each star measured on Gaia with the real PSF of the optics, subtracted, then redrawn at output: this "
              "gives the star sliders in Post."),
    dict(key="appris", phase=4, title="Neural star removal", kind="tool", tool="appris", stages=["star_net"],
         what="In addition to the star measurement: a star-removal network trained on this stack. Gives a second "
              "starless image."),
    dict(key="sans_etoiles", phase=4, title="Starless image", kind="auto", stages=["starless"],
         what="Subtracts the star model (the nebula under the stars is kept), then denoises the starless image. "
              "Comes with Stars."),
    dict(key="fond", phase=4, title="Background", kind="slot", stages=["background", "fond_simple"],
         what="Simple: a smooth surface fitted on the dark areas. H-alpha anchored: separates the true sky "
              "background from instrument defects using a professional survey."),
    dict(key="post", phase=5, title="Output", kind="fixed",
         stages=["render_prep", "render_prep_learned", "rendu_simple"],
         what="Prepares the render: what this workflow gives in Post, its image and its sliders."),
]
BY_KEY = {b["key"]: b for b in STEPS}
PHASES = {1: "1 · Calibration", 2: "2 · Registration", 3: "3 · Stacking", 4: "4 · Linear processing",
          5: "5 · Post"}


def phase_tag(phase) -> str:
    return _("common to all workflows") if phase < 4 else (_("specific to this workflow") if phase == 4 else "")


def load_rgb(rel: str) -> np.ndarray:
    return np.asarray(Image.open(DATA / rel).convert("RGB"))


def kv(rows, pad=14) -> QLabel:
    html = "<table cellspacing='0' cellpadding='3'>" + "".join(
        f"<tr><td style='color:{DIM}; padding-right:{pad}px' valign='top'>{k}</td><td>{v}</td></tr>"
        for k, v in rows) + "</table>"
    lb = QLabel(html)
    lb.setTextFormat(Qt.RichText)
    lb.setWordWrap(True)
    return lb


class Steps:
    """What a step is, for one session: inclusion, state, texts, measurements, images."""

    def __init__(self, session: Session):
        self.s = session
        self._img = {}

    # ------------------------------------------------------------ inclusion and state
    def title(self, p: PathDef | None, b) -> str:
        if b["key"] == "fond" and p is not None:
            return _("H-alpha anchored background") if p.opts["fond"] == "halpha" else _("Simple background")
        return _(b["title"])

    def included(self, p: PathDef, b) -> bool:
        o = p.opts
        if b["kind"] == "tool":
            return self.s.has_tool(o, b["tool"])
        if b["key"] == "demi_piles":
            return bool(o["debruitage"])
        if b["key"] == "sans_etoiles":
            return o["etoiles"] == "mesurer"
        return True

    def steps_of(self, p: PathDef):
        return [b for b in STEPS if self.included(p, b)]

    def stages_of(self, p: PathDef, b) -> list[str]:
        allst = TRUNK + self.s.path_stages(p)
        return [x for x in b["stages"] if x in allst]

    def step_titles(self, p: PathDef, stages) -> list[str]:
        """The steps (as the chain names them) that hold these stages, in chain order."""
        want = set(stages)
        return [self.title(p, b) for b in self.steps_of(p) if want & set(self.stages_of(p, b))]

    def run_stages(self, p: PathDef, b) -> list[str]:
        return b.get("run") or self.stages_of(p, b)

    def seconds(self, p: PathDef, b) -> float:
        return sum(self.s.seconds(x) for x in self.stages_of(p, b))

    def state(self, p: PathDef, b) -> str:
        """done | shared | pending | stale."""
        st = self.run_stages(p, b)
        if not st:
            return "done"
        states = [self.s.state(p, x) for x in st]
        if "stale" in states:
            return "stale"
        if "pending" in states:
            return "pending"
        if b["phase"] == 4 and st[0] not in TRUNK:
            own = self.s.owner(p, st[0])
            if own is not None and own.pid != p.pid:
                return "shared"
        return "done"

    def state_text(self, p: PathDef, b) -> tuple[str, str]:
        """(short text, colour) for a chain row."""
        k = b["key"]
        st = self.state(p, b)
        if k == "selection":
            if st == "pending":
                return _("after registration"), DIM
            n = len(self.s.data["frames"]["fwhm"])
            return _("{kept} / {n} frames").format(kept=self.s.kept_frames(), n=n), OK
        if k == "demi_piles":
            return (_("with stacking"), OK if st == "done" else (WARN if st == "stale" else DIM))
        sec = self.seconds(p, b)
        if st == "pending":
            return hm(sec), DIM
        if st == "stale":
            return _("outdated · {t}").format(t=hm(sec)), WARN
        if st == "shared":
            return _("✓ {t} · shared").format(t=hm(sec)), SHARED
        return f"✓ {hm(sec)}", OK

    def state_sentence(self, p: PathDef, b) -> str:
        st = self.state(p, b)
        sec = self.seconds(p, b)
        if b["key"] == "selection":
            return (_("The frames chart appears once registration is computed.") if st == "pending" else
                    _("The frames chart is on the Result side (Tab key)."))
        if b["key"] == "demi_piles":
            return _("Computed during stacking, at no extra time.")
        if st == "pending":
            return _("Pending · estimated {t} (measured on Heart).").format(t=hm(sec))
        if st == "stale":
            return _("Outdated: a setting changed since the run · {t}.").format(t=hm(sec))
        if st == "shared":
            own = self.s.owner(p, self.stages_of(p, b)[0])
            return _("Computed in {t}, once for this workflow and “{name}”.").format(t=hm(sec), name=own.name)
        return _("Computed in {t}.").format(t=hm(sec))

    def node_colors(self, state):
        return {"done": (OK, OK), "shared": (SHARED, SHARED), "stale": (WARN, WARN),
                "pending": ("#6b7280", None)}[state]

    # ------------------------------------------------------------ settings values
    def param(self, p: PathDef, stage, name):
        q = next((x for x in self.s.stages[stage]["params"] if x["name"] == name), None)
        if q is None:
            return None
        target = self.s.trunk_params if stage in TRUNK else p.params
        return target.get(stage, {}).get(name, q["value"])

    def summary(self, p: PathDef, b) -> str:
        inv = self.s.data["inventory"]
        k = b["key"]
        if k == "calibration":
            return _("{darks} darks, {flats} flats, no bias").format(darks=inv["n_darks"], flats=inv["n_flats"])
        if k == "alignement":
            return _("on the sharpest frame")
        if k == "selection":
            t, f = self.s.thresholds_now()
            return _("transparency ≥ {t} % · stars ≤ {f} px").format(t=f"{t:.0f}", f=fnum(f))
        if k == "empilement":
            sc = fnum(self.param(p, "stack", "scale"), 2).rstrip("0").rstrip(",.")
            return _("drizzle ×{scale} · rejection at {kappa} σ").format(
                scale=sc, kappa=fnum(self.param(p, "stack", "kappa"), 0))
        if k == "demi_piles":
            return _("comes with Denoise")
        if k == "sans_etoiles":
            return _("comes with Stars")
        if k == "debruitage":
            return _("{n} training steps").format(n=fmt_int(self.param(p, "denoise", "iterations")))
        if k == "appris":
            return _("{n} steps (fixed count)").format(n=fmt_int(self.param(p, "star_net", "steps")))
        if k == "post":
            av = self.s.available_sliders(p.opts)
            n = len(self.s.sliders)
            return _("{a} of {n} sliders").format(a=len(av), n=n)
        return ""

    # ------------------------------------------------------------ results
    def visuals(self, p: PathDef, k):
        if k == "calibration":
            return [("img:flat", _("Master flat")), ("img:calibrated", _("A calibrated frame")),
                    ("img:frame", _("The same frame, raw"))]
        if k == "alignement":
            return [("img:quality", _("Frame quality"))]
        if k in ("empilement", "demi_piles"):
            return [("img:stack", _("Stack (stretched for display)"))]
        if k == "astrometrie":
            return [("img:stack", _("Identified objects"))] + (
                [("img:survey", _("Reference H-alpha survey"))] if p.opts["fond"] == "halpha" else [])
        if k == "couleur":
            return [("chart:spectral", _("Sensor response")), ("img:spectral", _("Engine preview"))]
        if k == "debruitage":
            return [("img:denoise_crop", _("Before | after, at 100 %")), ("img:denoise", _("Engine preview"))]
        if k == "etoiles":
            return [("img:star_model", _("Star model")), ("img:psf", _("Measured shape of a star"))]
        if k == "appris":
            return [("img:star_removal", _("Neural star removal"))]
        if k == "sans_etoiles":
            return [("img:starless", _("Starless image"))]
        if k == "fond":
            if p.opts["fond"] == "simple":
                return [(f"layer:{self.s.layers['classique']['base']}", _("Stack with a simple background (preview)"))]
            return [("img:background", _("Background"))]
        if k == "post":
            return [(f"layer:{self.s.layers[p.layer]['base']}", _("The layer, sliders at suggested values"))]
        return []

    def source(self, kind):
        rel = self.s.data["steps"].get(kind[4:], kind[4:]) if kind.startswith("img:") else kind[6:]
        if rel not in self._img:
            self._img[rel] = StaticSource(load_rgb(rel))
        return self._img[rel]

    def overlay(self, k, kind):
        if k == "astrometrie" and kind == "img:stack":
            return self._labels_overlay
        if kind == "img:denoise_crop":
            return self._denoise_overlay
        return None

    def facts(self, p: PathDef, k):
        m = lambda st: self.s.stages.get(st, {}).get("metrics", {})       # noqa: E731
        if k == "calibration":
            c = m("calibrate")
            return [(_("Vignetting"), _("the corner gets {p} % of the center light").format(
                        p=round(100 * c["vignetting_min"]))),
                    (_("Defective pixels"), _("{p} %, of which {hot} hot found in the lights").format(
                        p=fnum(c["bad_pixels_pct"], 2), hot=c["hot_pixels_from_lights"])),
                    (_("Dark level"), _("{level} ADU, noise {noise} ADU").format(
                        level=fnum(c["dark_level"], 0), noise=fnum(c["dark_frame_noise"]))),
                    (_("Sky noise of one frame"), f"{fnum(c['sky_noise_adu'])} ADU"),
                    (_("Calibrated frames in cache"), _("{gb} GB").format(gb=fnum(c["cache_gb"])))]
        if k == "alignement":
            c = m("register")
            return [(_("Reference frame"), c["reference_frame"]), (_("Registered frames"), f"{c['aligned']}"),
                    (_("Registration residual"), f"{fnum(c['rms_px'], 2)} px"),
                    (_("Star size"), _("{a} to {b} px on 80 % of the frames").format(
                        a=fnum(c["fwhm_p10_p90"][0]), b=fnum(c["fwhm_p10_p90"][1]))),
                    (_("Drift over the night"), f"{round(c['drift_px'])} px")]
        if k == "selection":
            n = len(self.s.data["frames"]["fwhm"])
            kept = self.s.kept_frames()
            t, f = self.s.thresholds_now()
            return [(_("Frames kept"), _("{kept} of {n}").format(kept=kept, n=n)),
                    (_("Integration"), f"{fnum(kept * self.s.data['frames']['exposure_s'] / 3600)} h"),
                    (_("Minimum transparency"), _("{t} % of the best frames").format(t=f"{t:.0f}")),
                    (_("Maximum star size"), f"{fnum(f)} px"),
                    (_("Excluded by hand"), str(len(self.s.frames_excluded)))]
        if k in ("empilement", "demi_piles"):
            c = m("stack")
            return [(_("Stacked frames"), f"{c['frames_used']} ({fnum(c['integration_used_h'])} h)"),
                    (_("Stack size"), f"{c['grid'][0]} × {c['grid'][1]} px"),
                    (_("Rejected pixels"), _("{p} % (trails, cosmic rays)").format(p=fnum(c["rejected_pct"]))),
                    (_("Half-stacks"), _("two, of about 360 frames each") if p.opts["debruitage"] else _("none"))]
        if k == "astrometrie":
            c = m("astrometry")
            return [(_("Center"), f"α {fnum(c['ra'], 2)}°, δ {fnum(c['dec'], 2)}°"),
                    (_("Field"), f"{fnum(c['field_deg'][0])}° × {fnum(c['field_deg'][1])}°"),
                    (_("Scale"), _("{s}″ per pixel").format(s=fnum(c["scale_arcsec"], 2))),
                    (_("Gaia stars used"), _("{n}, residual {r} px").format(
                        n=fmt_int(c["stars_used"]), r=fnum(c["residual_px"], 2))),
                    (_("Measured focal length"), f"{round(c['focal_estimate_mm'])} mm")]
        if k == "couleur":
            c = m("color")
            return [(_("Color calibration stars"), fmt_int(c["stars"])),
                    (_("Deviation from Gaia colors"), f"{fnum(c['fit_agreement_pct'])} %"),
                    (_("H-alpha response (656 nm)"), _("red {r}: astro-modified camera").format(
                        r=fnum(c["response_at_lines"]["656nm"][0], 2)))]
        if k == "debruitage":
            c = m("denoise")
            return [(_("Sky noise"), _("{before} → {after}, divided by {k}").format(
                        before=fnum(c["noise_before"], 2), after=fnum(c["noise_after"], 2),
                        k=fnum(c["noise_reduction"], 2))),
                    (_("Training"), _("{n} steps on the half-stacks").format(
                        n=fmt_int(self.param(p, "denoise", "iterations"))))]
        if k == "etoiles":
            c = m("stars")
            return [(_("Stars measured one by one"), fmt_int(c["modelled_stars"])),
                    (_("Faint stars modelled"), fmt_int(c["faint_stars"])),
                    (_("Predicted / measured flux deviation"),
                     f"{fnum(float(np.mean(c['prediction_scatter_pct'])))} %"),
                    (_("Star shape"), _("measured on a grid of {a} × {b} zones").format(
                        a=c["local_psf_grid"][0], b=c["local_psf_grid"][1]))]
        if k == "appris":
            c = m("starless")
            return [(_("Learned layer kept"), f"{fnum(c['learned_layer_kept_pct'])} %"),
                    (_("Training"), _("{n} steps").format(n=fmt_int(self.param(p, "star_net", "steps"))))]
        if k == "sans_etoiles":
            c = m("starless")
            return [(_("Surface rebuilt under the stars"), f"{fnum(c['filled_pct'])} %")]
        if k == "fond":
            if p.opts["fond"] == "simple":
                return [(_("Method"), _("degree-2 surface on the dark areas")),
                        (_("Duration"), _("≈ 1 min (estimate: step to build in the engine)"))]
            c = m("background")
            return [(_("Image used as sky"), f"{fnum(c['sky_fit_blocks_pct'])} %"),
                    (_("H-alpha survey"), _("registered on this image")),
                    (_("Dips repaired under the stars"), fmt_int(c["emission_dips"]["below_8sigma"]))]
        if k == "post":
            rows = []
            for key, sd in self.s.sliders.items():
                ok, why, _need = self.s.slider_state(p, key)
                rows.append((sd["name"], _("available") if ok else
                             f"<span style='color:{WARN}'>" + _("greyed out: {why}").format(why=why) + "</span>"))
            return rows
        return []

    # ------------------------------------------------------------ overlays
    def _labels_overlay(self, p: QPainter, v: Viewer):
        sc = self.s.data["stack_view_scale"]
        f = QFont(v.font())
        f.setPointSizeF(9.5)
        f.setBold(True)
        p.setFont(f)
        for lb in self.s.data["labels"]:
            pt = v.img_to_w(lb["x"] * sc, lb["y"] * sc)
            p.setPen(QPen(QColor("#ffd28a"), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(pt, 14, 14)
            p.setPen(QColor("#ffd28a"))
            p.drawText(QPointF(pt.x() + 18, pt.y() + 5), lb["name"])

    def _denoise_overlay(self, p: QPainter, v: Viewer):
        w, _h = v.source.size()
        for x, t in ((0, _("Before: the stack")), (w / 2 + 8, _("After: denoised"))):
            pt = v.img_to_w(x, 0)
            v._tag(p, QPointF(pt.x() + 8, pt.y() + 8), t)


def fmt_value(x) -> str:
    if isinstance(x, bool):
        return _("yes") if x else _("no")
    if isinstance(x, float):
        return fnum(x, 2).rstrip("0").rstrip(",.")
    if x is None:
        return "auto"
    return str(x)


class Node(QWidget):
    """The step's node on the chain line: hollow grey = to compute, green = computed, violet = shared, orange = to
    recompute."""

    def __init__(self, state, first, last, dim=False, parent=None):
        super().__init__(parent)
        self.state, self.first, self.last, self.dim = state, first, last, dim
        self.setFixedWidth(24)
        self.setMinimumHeight(30)

    def paintEvent(self, e):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing)
            x, y = 12, 16
            p.setPen(QPen(QColor("#3a4150"), 2))
            p.drawLine(QPointF(x, 0 if not self.first else y), QPointF(x, self.height() if not self.last else y))
            line, fill = {"done": (OK, OK), "shared": (SHARED, SHARED), "stale": (WARN, WARN),
                          "pending": ("#7b8494", None)}[self.state]
            p.setPen(QPen(QColor(line), 2))
            p.setBrush(QColor(fill) if fill else QColor("#15171c"))
            r = 4.5 if self.dim else 6
            p.drawEllipse(QPointF(x, y), r, r)
        finally:
            p.end()
