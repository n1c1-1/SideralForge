"""The steps of a workflow as both the Workflow and the Results views show them: what each one is, its state, its
summary, its measurements and its images. Shared so that a step looks the same in the definition and in the run.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QLabel, QWidget

from model import DATA, TRUNK, PathDef, Session, hm
from viewer import StaticSource, Viewer
from widgets import OK, SHARED, WARN

DIM = "#8b93a1"


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", " ")


def fnum(x, d=1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


STEPS = [
    dict(key="calibration", phase=1, title="Calibration", kind="fixed", stages=["inventory", "calibrate"],
         what="Masters de darks et de flats, puis chaque pose calibrée : signal thermique, vignetage et pixels "
              "défectueux retirés."),
    dict(key="alignement", phase=2, title="Alignement", kind="fixed", stages=["register"],
         what="Mesure les étoiles de chaque pose (taille, transparence) et aligne toutes les poses sur la plus "
              "nette."),
    dict(key="selection", phase=2, title="Sélection des poses", kind="fixed", stages=[], run=["register"],
         what="Quelles poses entrent dans la pile : par seuils de taille des étoiles et de transparence, ou une à "
              "une."),
    dict(key="empilement", phase=3, title="Empilement", kind="fixed", stages=["stack"],
         what="Empile les poses directement depuis les pixels couleur du capteur (drizzle sans dématriçage) et "
              "rejette les traînées et les pixels aberrants."),
    dict(key="demi_piles", phase=3, title="Deux demi-piles", kind="auto", stages=[], run=["stack"],
         what="Deux piles indépendantes d'environ 360 poses : leur écart mesure le bruit, ce qui permet le débruitage "
              "sans rien inventer. Calculées pendant l'empilement, sans durée en plus."),
    dict(key="astrometrie", phase=4, title="Astrométrie", kind="fixed", stages=["astrometry", "catalogs", "surveys"],
         what="Trouve où l'image pointe sur le ciel en comparant ses étoiles à Gaia, puis récupère les étoiles du "
              "catalogue (et le relevé H-alpha si le fond l'utilise)."),
    dict(key="couleur", phase=4, title="Couleur", kind="fixed", stages=["color"],
         what="Mesure la réponse de ton capteur sur les spectres Gaia des étoiles du champ, puis la matrice vers les "
              "vraies couleurs du ciel."),
    dict(key="debruitage", phase=4, title="Débruitage", kind="tool", tool="debruitage", stages=["denoise"],
         what="Un réseau apprend sur tes deux demi-piles à prédire l'une à partir de l'autre : il retire le bruit "
              "sans rien inventer."),
    dict(key="etoiles", phase=4, title="Mesure des étoiles", kind="tool", tool="etoiles", stages=["optics", "stars"],
         what="Chaque étoile mesurée sur Gaia avec la forme réelle de ton optique, soustraite, puis redessinée au "
              "rendu : c'est ce qui donne les curseurs d'étoiles en post."),
    dict(key="appris", phase=4, title="Retrait appris des étoiles", kind="tool", tool="appris", stages=["star_net"],
         what="En plus de la mesure : un réseau de retrait des étoiles appris sur ta propre pile. Donne une seconde "
              "image sans étoiles."),
    dict(key="sans_etoiles", phase=4, title="Image sans étoiles", kind="auto", stages=["starless"],
         what="Soustrait le modèle des étoiles (la nébuleuse sous les étoiles est gardée), puis débruite l'image sans "
              "étoiles. Vient avec la mesure des étoiles."),
    dict(key="fond", phase=4, title="Fond de ciel", kind="slot", stages=["background", "fond_simple"],
         what="Simple : une surface lisse ajustée sur les zones sombres. Ancré sur le H-alpha : sépare le vrai fond "
              "du ciel des défauts de l'instrument grâce à un relevé professionnel."),
    dict(key="post", phase=5, title="Calque pour le post", kind="fixed",
         stages=["render_prep", "render_prep_learned", "rendu_simple"],
         what="Prépare le rendu : ce que ce workflow donnera en post, son image et ses curseurs."),
]
BY_KEY = {b["key"]: b for b in STEPS}
PHASES = {1: "1 · Calibration", 2: "2 · Alignement", 3: "3 · Empilement", 4: "4 · Traitement linéaire",
          5: "5 · Post"}


def phase_tag(phase) -> str:
    return "commun à tous les workflows" if phase < 4 else ("propre à ce workflow" if phase == 4 else "")


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
            return "Fond ancré sur le H-alpha" if p.opts["fond"] == "halpha" else "Fond simple"
        return b["title"]

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
                return "après l'alignement", DIM
            n = len(self.s.data["frames"]["fwhm"])
            return f"{self.s.kept_frames()} / {n} poses", OK
        if k == "demi_piles":
            return ("avec l'empilement", OK if st == "done" else (WARN if st == "stale" else DIM))
        sec = self.seconds(p, b)
        if st == "pending":
            return hm(sec), DIM
        if st == "stale":
            return f"à recalculer · {hm(sec)}", WARN
        if st == "shared":
            return f"✓ {hm(sec)} · partagée", SHARED
        return f"✓ {hm(sec)}", OK

    def state_sentence(self, p: PathDef, b) -> str:
        st = self.state(p, b)
        sec = self.seconds(p, b)
        if b["key"] == "selection":
            return ("Le graphe des poses apparaît une fois l'alignement calculé." if st == "pending" else
                    "Le graphe des poses est sur la face Résultat (touche Tab).")
        if b["key"] == "demi_piles":
            return "Calculées pendant l'empilement, sans durée en plus."
        if st == "pending":
            return f"Pas encore calculée · durée estimée {hm(sec)} (mesurée sur Heart)."
        if st == "stale":
            return f"À recalculer : un réglage a changé depuis le calcul · {hm(sec)}."
        if st == "shared":
            own = self.s.owner(p, self.stages_of(p, b)[0])
            return f"Calculée en {hm(sec)}, une seule fois pour ce workflow et « {own.name} »."
        return f"Calculée en {hm(sec)}."

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
            return f"{inv['n_darks']} darks, {inv['n_flats']} flats, pas d'offsets"
        if k == "alignement":
            return "sur la pose la plus nette"
        if k == "selection":
            t, f = self.s.thresholds_now()
            return f"transparence ≥ {t:.0f} % · étoiles ≤ {fnum(f)} px"
        if k == "empilement":
            sc = fnum(self.param(p, "stack", "scale"), 2).rstrip("0").rstrip(",")
            return f"drizzle ×{sc} · rejet à {fnum(self.param(p, 'stack', 'kappa'), 0)} σ"
        if k == "demi_piles":
            return "vient avec le débruitage"
        if k == "sans_etoiles":
            return "vient avec la mesure des étoiles"
        if k == "debruitage":
            return f"{fmt_int(self.param(p, 'denoise', 'iterations'))} pas d'apprentissage"
        if k == "appris":
            return f"{fmt_int(self.param(p, 'star_net', 'steps'))} pas (nombre fixe)"
        if k == "post":
            av = self.s.available_sliders(p.opts)
            n = len(self.s.sliders)
            return f"{len(av)} curseurs sur {n}"
        return ""

    # ------------------------------------------------------------ results
    def visuals(self, p: PathDef, k):
        if k == "calibration":
            return [("img:flat", "Flat maître"), ("img:calibrated", "Une pose calibrée"),
                    ("img:frame", "La même pose, brute")]
        if k == "alignement":
            return [("img:quality", "Qualité des poses")]
        if k in ("empilement", "demi_piles"):
            return [("img:stack", "Pile (étirée pour l'affichage)")]
        if k == "astrometrie":
            return [("img:stack", "Objets identifiés")] + (
                [("img:survey", "Relevé H-alpha de référence")] if p.opts["fond"] == "halpha" else [])
        if k == "couleur":
            return [("chart:spectral", "Réponse du capteur"), ("img:spectral", "Aperçu du moteur")]
        if k == "debruitage":
            return [("img:denoise_crop", "Avant | après, à 100 %"), ("img:denoise", "Aperçu du moteur")]
        if k == "etoiles":
            return [("img:star_model", "Modèle des étoiles"), ("img:psf", "Forme mesurée d'une étoile")]
        if k == "appris":
            return [("img:star_removal", "Retrait appris")]
        if k == "sans_etoiles":
            return [("img:starless", "Image sans étoiles")]
        if k == "fond":
            if p.opts["fond"] == "simple":
                return [(f"layer:{self.s.layers['classique']['base']}", "Pile avec un fond simple (aperçu)")]
            return [("img:background", "Fond de ciel")]
        if k == "post":
            return [(f"layer:{self.s.layers[p.layer]['base']}", "Le calque, curseurs à la suggestion")]
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
            return [("Vignetage", f"le coin reçoit {round(100 * c['vignetting_min'])} % de la lumière du centre"),
                    ("Pixels défectueux", f"{fnum(c['bad_pixels_pct'], 2)} %, dont {c['hot_pixels_from_lights']} "
                                          "chauds trouvés dans les poses"),
                    ("Niveau des darks", f"{fnum(c['dark_level'], 0)} ADU, bruit {fnum(c['dark_frame_noise'])} ADU"),
                    ("Bruit du ciel d'une pose", f"{fnum(c['sky_noise_adu'])} ADU"),
                    ("Poses calibrées en cache", f"{fnum(c['cache_gb'])} Go")]
        if k == "alignement":
            c = m("register")
            return [("Pose de référence", c["reference_frame"]), ("Poses alignées", f"{c['aligned']}"),
                    ("Écart d'alignement", f"{fnum(c['rms_px'], 2)} px"),
                    ("Taille des étoiles", f"{fnum(c['fwhm_p10_p90'][0])} à {fnum(c['fwhm_p10_p90'][1])} px sur 80 % "
                                           "des poses"),
                    ("Dérive pendant la nuit", f"{round(c['drift_px'])} px")]
        if k == "selection":
            n = len(self.s.data["frames"]["fwhm"])
            kept = self.s.kept_frames()
            t, f = self.s.thresholds_now()
            return [("Poses gardées", f"{kept} sur {n}"),
                    ("Intégration", f"{fnum(kept * self.s.data['frames']['exposure_s'] / 3600)} h"),
                    ("Transparence minimale", f"{t:.0f} % des meilleures poses"),
                    ("Taille maximale des étoiles", f"{fnum(f)} px"),
                    ("Exclues à la main", str(len(self.s.frames_excluded)))]
        if k in ("empilement", "demi_piles"):
            c = m("stack")
            return [("Poses empilées", f"{c['frames_used']} ({fnum(c['integration_used_h'])} h)"),
                    ("Taille de la pile", f"{c['grid'][0]} × {c['grid'][1]} px"),
                    ("Pixels rejetés", f"{fnum(c['rejected_pct'])} % (traînées, rayons cosmiques)"),
                    ("Demi-piles", "deux, d'environ 360 poses chacune" if p.opts["debruitage"] else "aucune")]
        if k == "astrometrie":
            c = m("astrometry")
            return [("Centre", f"α {fnum(c['ra'], 2)}°, δ {fnum(c['dec'], 2)}°"),
                    ("Champ", f"{fnum(c['field_deg'][0])}° × {fnum(c['field_deg'][1])}°"),
                    ("Échelle", f"{fnum(c['scale_arcsec'], 2)}″ par pixel"),
                    ("Étoiles Gaia utilisées", f"{fmt_int(c['stars_used'])}, écart {fnum(c['residual_px'], 2)} px"),
                    ("Focale mesurée", f"{round(c['focal_estimate_mm'])} mm")]
        if k == "couleur":
            c = m("color")
            return [("Étoiles de couleur", fmt_int(c["stars"])),
                    ("Écart avec les couleurs Gaia", f"{fnum(c['fit_agreement_pct'])} %"),
                    ("Réponse à H-alpha (656 nm)", f"rouge {fnum(c['response_at_lines']['656nm'][0], 2)} : "
                                                    "boîtier défiltré")]
        if k == "debruitage":
            c = m("denoise")
            return [("Bruit du ciel", f"{fnum(c['noise_before'], 2)} → {fnum(c['noise_after'], 2)}, divisé par "
                                      f"{fnum(c['noise_reduction'], 2)}"),
                    ("Apprentissage", f"{fmt_int(self.param(p, 'denoise', 'iterations'))} pas sur les demi-piles")]
        if k == "etoiles":
            c = m("stars")
            return [("Étoiles mesurées une à une", fmt_int(c["modelled_stars"])),
                    ("Étoiles faibles modélisées", fmt_int(c["faint_stars"])),
                    ("Écart flux prédit / mesuré", f"{fnum(float(np.mean(c['prediction_scatter_pct'])))} %"),
                    ("Forme des étoiles", f"mesurée sur une grille de {c['local_psf_grid'][0]} × "
                                          f"{c['local_psf_grid'][1]} zones")]
        if k == "appris":
            c = m("starless")
            return [("Couche apprise gardée", f"{fnum(c['learned_layer_kept_pct'])} %"),
                    ("Apprentissage", f"{fmt_int(self.param(p, 'star_net', 'steps'))} pas")]
        if k == "sans_etoiles":
            c = m("starless")
            return [("Surface reconstruite sous les étoiles", f"{fnum(c['filled_pct'])} %")]
        if k == "fond":
            if p.opts["fond"] == "simple":
                return [("Méthode", "surface de degré 2 sur les zones sombres"),
                        ("Durée", "≈ 1 min (estimation : étape à créer dans le moteur)")]
            c = m("background")
            return [("Image utilisée comme ciel", f"{fnum(c['sky_fit_blocks_pct'])} %"),
                    ("Relevé H-alpha", "recalé sur ton image"),
                    ("Creux réparés sous les étoiles", fmt_int(c["emission_dips"]["below_8sigma"]))]
        if k == "post":
            rows = []
            for key, sd in self.s.sliders.items():
                ok, why, _ = self.s.slider_state(p, key)
                rows.append((sd["name"], "disponible" if ok else f"<span style='color:{WARN}'>grisé : {why}</span>"))
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
        w, _ = v.source.size()
        for x, t in ((0, "Avant : la pile"), (w / 2 + 8, "Après : débruitée")):
            pt = v.img_to_w(x, 0)
            v._tag(p, QPointF(pt.x() + 8, pt.y() + 8), t)


def fmt_value(x) -> str:
    if isinstance(x, bool):
        return "oui" if x else "non"
    if isinstance(x, float):
        return fnum(x, 2).rstrip("0").rstrip(",")
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
