"""State of the mockup: the workflows of the project (the optional steps of each), what each workflow shares with the
others, the state of every step (to compute, computed, to recompute), costs, test mode, and the three demo projects
(blank, half run, run).

Nothing here computes images: the costs come from the stage durations measured on the real project, the images
from the files prepared by build/prepare.py.
"""
from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TRUNK = ["inventory", "calibrate", "register", "stack"]
# stages that do not exist in the code today, with an estimated duration
EXTRA = {"fond_simple": ("Fond simple", 60.0), "rendu_simple": ("Préparation du rendu simple", 30.0)}
# options of a workflow that change what a stage computes; a stage also depends on those of its upstream stages
RELEVANT = {"denoise": ("debruitage",), "starless": ("appris",), "fond_simple": ("etoiles",),
            "render_prep": ("fond",), "render_prep_learned": ("fond",)}
TEMPLATES = {
    "Vierge": dict(debruitage=False, etoiles="garder", appris=False, fond="simple"),
    "Classique + débruitage": dict(debruitage=True, etoiles="garder", appris=False, fond="simple"),
    "Complet": dict(debruitage=True, etoiles="mesurer", appris=False, fond="halpha"),
}
STAR_SLIDERS = ("nombre", "taille", "couleur", "halo")
# the optional tools: all go in phase 4 today
TOOLS = {
    "debruitage": dict(title="Débruitage", phase=4,
                       what="Retire le bruit sans rien inventer : un réseau apprend sur tes deux demi-piles à prédire "
                            "l'une à partir de l'autre."),
    "etoiles": dict(title="Mesure des étoiles", phase=4,
                    what="Mesure chaque étoile sur Gaia avec la forme réelle de ton optique, la retire, puis la "
                         "redessine au rendu : donne les curseurs d'étoiles en post."),
    "appris": dict(title="Retrait appris des étoiles", phase=4,
                   what="Un réseau de retrait des étoiles appris sur ta propre pile, en plus de la mesure : une "
                        "seconde image sans étoiles."),
    "halpha": dict(title="Fond ancré sur le H-alpha", phase=4,
                   what="Sépare le vrai fond du ciel des défauts de l'instrument grâce à un relevé professionnel. "
                        "Remplace le fond simple."),
    "simple": dict(title="Fond simple", phase=4,
                   what="Une surface lisse ajustée sur les zones sombres. Remplace le fond ancré sur le H-alpha."),
}
SCENARIOS = {"vierge": "Projet vierge", "moitie": "Projet à moitié calculé", "calcule": "Projet calculé"}


def hm(seconds: float) -> str:
    m = int(round(seconds / 60))
    if m < 1:
        return "≤ 1 min"
    return f"{m // 60} h {m % 60:02d}" if m >= 60 else f"{m} min"


@dataclass
class PathDef:
    """A workflow of the project."""
    pid: int
    name: str
    opts: dict
    layer: str                                    # key of the prepared images this workflow shows in Post
    computed: bool = True
    provisional: bool = False                     # its images are those of the closest prepared workflow
    params: dict = field(default_factory=dict)    # stage -> {param: value} changed by the user
    rendered: bool = False                        # has been computed to the end once: it has a layer in Post


class Session(QObject):
    pathsChanged = Signal()
    currentChanged = Signal()                      # current workflow or selected step
    testChanged = Signal()
    costChanged = Signal()                         # what is to compute changed, without a new workflow or option
    sourcesChanged = Signal()

    def __init__(self, scenario: str = "calcule"):
        super().__init__()
        self.scenario = scenario
        self.data = json.loads((DATA / "data.json").read_text(encoding="utf-8"))
        self.stages = self.data["stages"]
        self.layers = {lay["key"]: lay for lay in self.data["layers"]}
        self.sliders = {s["key"]: s for s in self.data["sliders"]}
        self.trunk_params: dict = {}               # stage -> {param: value} for phases 1-3
        self.frames_excluded = set()               # frame indices excluded by hand (sources or chart)
        self.thresholds = None                     # (min_transparency, max_fwhm_ratio) set on the chart
        self.paths: list[PathDef] = []
        self._ids = itertools.count(1)
        self.computed_keys: set = set()
        self.computed_trunk: set = set()
        self.ever: set = set()                     # (pid or None, stage) computed at least once, any settings
        self.current = -1
        self.sel_step = "calibration"
        self.sources_loaded = scenario != "vierge"
        if scenario == "calcule":
            for name, layer in [("Complet", "complet"), ("Complet + retrait appris", "complet_appris"),
                                ("Pile classique", "classique"), ("Classique + débruitage", "classique_dn")]:
                opts = dict(TEMPLATES["Complet" if name.startswith("Complet") else
                                      ("Vierge" if name == "Pile classique" else name)])
                if layer == "complet_appris":
                    opts["appris"] = True
                self.paths.append(PathDef(next(self._ids), name, opts, layer))
            self.mark_stages([(None, s) for s in TRUNK] + [(p, s) for p in self.paths for s in self.path_stages(p)],
                             emit=False)
            self.current = 0
            self.sel_step = "selection"
        elif scenario == "moitie":
            cl = PathDef(next(self._ids), "Pile classique", dict(TEMPLATES["Vierge"]), "classique")
            co = PathDef(next(self._ids), "Complet", dict(TEMPLATES["Complet"]), "complet")
            self.paths = [co, cl]
            done_co = [s for s in self.path_stages(co) if s in ("astrometry", "catalogs", "surveys", "color", "denoise")]
            self.mark_stages([(None, s) for s in TRUNK] + [(cl, s) for s in self.path_stages(cl)]
                             + [(co, s) for s in done_co], emit=False)
            self.current = 0
            self.sel_step = "etoiles"
        for p in self.paths:
            p.computed = not self.pending(p)
        # test mode
        self.test_on = False
        gx0, gx1, gy0, gy1 = self.data["crop"]
        w, h = gx1 - gx0, gy1 - gy0
        self.test_zone = (gx0 + 0.27 * w, gy0 + 0.21 * h, gx0 + 0.6 * w, gy0 + 0.54 * h)
        self.test_frames = 1.0
        self.test_short_learned = False

    @property
    def path(self) -> PathDef | None:
        return self.paths[self.current] if 0 <= self.current < len(self.paths) else None

    def select(self, index=None, step=None):
        if index is not None:
            self.current = index
        if step is not None:
            self.sel_step = step
        self.currentChanged.emit()

    def load_sources(self):
        self.sources_loaded = True
        self.sourcesChanged.emit()
        self.pathsChanged.emit()

    # ------------------------------------------------------------ stages of a workflow
    def title(self, stage: str) -> str:
        if stage in EXTRA:
            return EXTRA[stage][0]
        return self.stages[stage]["title"]

    def seconds(self, stage: str) -> float:
        if stage in EXTRA:
            return EXTRA[stage][1]
        return float(self.stages.get(stage, {}).get("seconds", 0.0))

    def path_stages(self, p: PathDef) -> list[str]:
        o = p.opts
        st = ["astrometry", "catalogs"]
        if o["fond"] == "halpha":
            st.append("surveys")
        st.append("color")
        if o["debruitage"]:
            st.append("denoise")
        if o["etoiles"] == "mesurer":
            st += ["optics", "stars"]
            if o["appris"]:
                st.append("star_net")
            st.append("starless")
        st.append("background" if o["fond"] == "halpha" else "fond_simple")
        if o["etoiles"] == "mesurer":
            st.append("render_prep_learned" if o["appris"] else "render_prep")
        else:
            st.append("rendu_simple")
        return st

    def deps(self, p: PathDef, stage: str) -> list[str]:
        here = set(TRUNK) | set(self.path_stages(p))
        if stage == "fond_simple":
            return [s for s in ("color", "denoise", "starless") if s in here]
        if stage == "rendu_simple":
            return ["fond_simple"]
        return [d for d in self.stages.get(stage, {}).get("deps", []) if d in here]

    def upstream(self, p: PathDef, stage: str) -> set[str]:
        out, todo = set(), list(self.deps(p, stage))
        while todo:
            s = todo.pop()
            if s not in out:
                out.add(s)
                todo += self.deps(p, s)
        return out

    def stage_key(self, p: PathDef, stage: str):
        """Two workflows share a stage when this key is equal: same options and same settings up to it."""
        chain = sorted({stage} | self.upstream(p, stage))
        rel = tuple(sorted({(k, p.opts[k]) for s in chain for k in RELEVANT.get(s, ())}))
        sets = tuple((s, tuple(sorted({**self.trunk_params.get(s, {}), **p.params.get(s, {})}.items())))
                     for s in chain if self.trunk_params.get(s) or p.params.get(s))
        return stage, rel, sets

    def owner(self, p: PathDef, stage: str):
        """The first workflow (in list order) that computes this stage the same way, or None."""
        k = self.stage_key(p, stage)
        for q in self.paths:
            if stage in self.path_stages(q) and self.stage_key(q, stage) == k:
                return q
        return None

    def trunk_key(self, stage: str):
        chain = TRUNK[: TRUNK.index(stage) + 1]
        return stage, tuple((c, tuple(sorted(self.trunk_params[c].items()))) for c in chain if self.trunk_params.get(c))

    def trunk_pending(self) -> list[str]:
        return [s for s in TRUNK if self.trunk_key(s) not in self.computed_trunk]

    def is_pending(self, p: PathDef | None, stage: str) -> bool:
        if stage in TRUNK:
            return self.trunk_key(stage) not in self.computed_trunk
        return self.stage_key(p, stage) not in self.computed_keys

    def state(self, p: PathDef | None, stage: str) -> str:
        """done | pending (never computed) | stale (computed once, a setting changed since)."""
        if not self.is_pending(p, stage):
            return "done"
        who = None if stage in TRUNK else p.pid
        return "stale" if (who, stage) in self.ever else "pending"

    def pending(self, p: PathDef) -> list[str]:
        return [s for s in self.path_stages(p) if self.stage_key(p, s) not in self.computed_keys]

    def path_seconds(self, p: PathDef) -> float:
        return sum(self.seconds(s) for s in self.path_stages(p))

    def trunk_seconds(self) -> float:
        return sum(self.seconds(s) for s in TRUNK)

    def opts_seconds(self, opts: dict) -> float:
        return self.trunk_seconds() + self.path_seconds(PathDef(0, "", dict(opts), ""))

    def cost_of(self, opts: dict, params: dict | None = None) -> tuple[float, list[str], list[str]]:
        """Cost of a workflow with these options: (seconds to compute, stages to compute, stages reused)."""
        tmp = PathDef(0, "", dict(opts), "", params=copy.deepcopy(params or {}))
        todo, reused = [], []
        for s in self.path_stages(tmp):
            (reused if self.stage_key(tmp, s) in self.computed_keys else todo).append(s)
        return sum(self.seconds(s) for s in todo), todo, reused

    def downstream(self, p: PathDef, stage: str) -> list[str]:
        """The stages of the workflow that follow (depend on) this one, in order, the stage included."""
        allst = TRUNK + self.path_stages(p)
        return [s for s in allst if s == stage or stage in self.upstream(p, s)]

    def pending_upto(self, p: PathDef, stages: list[str]) -> list[tuple[PathDef | None, str]]:
        """What 'Lancer jusqu'ici' computes: the given stages and everything they need, when not computed yet."""
        want = set()
        for s in stages:
            want |= {s} | self.upstream(p, s)
        return [(None if x in TRUNK else p, x) for x in TRUNK + self.path_stages(p)
                if x in want and self.is_pending(p, x)]

    # ------------------------------------------------------------ the toolbox
    @staticmethod
    def has_tool(o: dict, key: str) -> bool:
        return {"debruitage": o["debruitage"], "etoiles": o["etoiles"] == "mesurer", "appris": o["appris"],
                "halpha": o["fond"] == "halpha", "simple": o["fond"] == "simple"}[key]

    @staticmethod
    def with_tool(o: dict, key: str) -> tuple[dict, list[str]]:
        """The options with this tool added, and what came with it."""
        o, notes = dict(o), []

        def need_stars(why):
            if o["etoiles"] != "mesurer":
                o["etoiles"] = "mesurer"
                notes.append(f"Mesure des étoiles ajoutée : {why}")

        if key == "appris":
            o["appris"] = True
            need_stars("le retrait appris part du modèle des étoiles")
        elif key == "halpha":
            o["fond"] = "halpha"
            notes.append("Remplace le fond simple")
            need_stars("le fond ancré se calcule sur l'image sans étoiles")
        elif key == "simple":
            o["fond"] = "simple"
            notes.append("Remplace le fond ancré sur le H-alpha")
        elif key == "etoiles":
            o["etoiles"] = "mesurer"
        elif key == "debruitage":
            o["debruitage"] = True
        if o["etoiles"] == "mesurer" and not o["debruitage"]:
            o["debruitage"] = True
            notes.append("Débruitage ajouté : la mesure des étoiles en a besoin")
        return o, notes

    @staticmethod
    def without_tool(o: dict, key: str) -> tuple[dict | None, list[str]]:
        """The options with this tool removed and what went with it, or (None, [reason]) when it cannot go."""
        o, notes = dict(o), []
        if key == "debruitage":
            if o["etoiles"] == "mesurer":
                return None, ["la mesure des étoiles en a besoin : retire-la d'abord"]
            o["debruitage"] = False
        elif key == "etoiles":
            o["etoiles"] = "garder"
            if o["appris"]:
                o["appris"] = False
                notes.append("Retrait appris retiré aussi : il part du modèle des étoiles")
            if o["fond"] == "halpha":
                o["fond"] = "simple"
                notes.append("Fond simple à la place du fond ancré : celui-ci se calcule sur l'image sans étoiles")
        elif key == "appris":
            o["appris"] = False
        elif key == "halpha":
            o["fond"] = "simple"
            notes.append("Remplacé par le fond simple")
        elif key == "simple":
            return None, ["il faut une étape de fond : dépose le fond ancré sur le H-alpha pour le remplacer"]
        return o, notes

    def tool_status(self, o: dict, key: str) -> tuple[str, str]:
        """(state, text) of a tool for the workflow: in | add | replace."""
        if self.has_tool(o, key):
            return "in", "dans la chaîne"
        new, notes = self.with_tool(o, key)
        dt = self.opts_seconds(new) - self.opts_seconds(o)
        cost = f"+{hm(dt)}" if dt >= 30 else "même durée"
        extra = [n.split(" ajouté")[0] for n in notes if " ajouté" in n]
        if key in ("halpha", "simple"):
            return "replace", f"remplace le {'fond simple' if key == 'halpha' else 'fond ancré'} · {cost}" + (
                f" · ajoute aussi : {', '.join(extra)}" if extra else "")
        return "add", cost + (f" · ajoute aussi : {', '.join(extra)}" if extra else "")

    # ------------------------------------------------------------ what a workflow gives in Post
    def slider_state(self, p: PathDef, key: str):
        """(available, reason why not, options that would unlock it)."""
        o = p.opts
        if key in STAR_SLIDERS or key == "detail":
            if o["etoiles"] != "mesurer":
                why = ("les curseurs d'étoiles demandent la mesure des étoiles" if key != "detail"
                       else "l'accentuation demande l'image sans étoiles (mesure des étoiles)")
                need = {**o, "etoiles": "mesurer", "debruitage": True}
                return False, why, need
        if key == "bruit" and not o["debruitage"]:
            return False, "le lissage du bruit demande le débruitage", {**o, "debruitage": True}
        return True, "", None

    def unlock_text(self, p: PathDef, need: dict) -> str:
        sec, todo, reused = self.cost_of(need, p.params)
        parts = []
        if need["etoiles"] != p.opts["etoiles"]:
            parts.append("la mesure des étoiles")
        if need["debruitage"] != p.opts["debruitage"]:
            parts.append("le débruitage")
        t = f"Ajouter {' et '.join(parts)} au workflow : +{hm(sec)}"
        mine = set(self.path_stages(p))
        if [s for s in reused if s not in mine]:
            t += ", le reste est déjà calculé par un autre workflow"
        return t

    def available_sliders(self, opts: dict) -> list[str]:
        tmp = PathDef(0, "", opts, "")
        return [k for k in self.sliders if self.slider_state(tmp, k)[0]]

    def closest_layer(self, opts: dict) -> tuple[str, bool]:
        if opts["etoiles"] == "mesurer":
            key = "complet_appris" if opts["appris"] else "complet"
        else:
            key = "classique_dn" if opts["debruitage"] else "classique"
        ref = {"complet": TEMPLATES["Complet"], "complet_appris": {**TEMPLATES["Complet"], "appris": True},
               "classique": TEMPLATES["Vierge"], "classique_dn": TEMPLATES["Classique + débruitage"]}[key]
        return key, ref != opts

    # ------------------------------------------------------------ changes
    def unique_name(self, base: str) -> str:
        names = {p.name for p in self.paths}
        if base not in names:
            return base
        n = 2
        while f"{base} {n}" in names:
            n += 1
        return f"{base} {n}"

    def add_path(self, name: str, opts: dict, params: dict | None = None) -> PathDef:
        layer, prov = self.closest_layer(opts)
        p = PathDef(next(self._ids), name, dict(opts), layer, computed=False, provisional=prov,
                    params=copy.deepcopy(params or {}))
        self.paths.append(p)
        self._sync()
        p.rendered = p.computed and not self.trunk_pending()
        self.pathsChanged.emit()
        return p

    def remove_path(self, i: int):
        self.paths.pop(i)
        self.current = min(self.current, len(self.paths) - 1)
        self.pathsChanged.emit()
        self.currentChanged.emit()

    def set_opts(self, p: PathDef, opts: dict):
        p.opts = dict(opts)
        p.layer, p.provisional = self.closest_layer(opts)
        self._sync()
        self.pathsChanged.emit()

    def set_param(self, p: PathDef | None, stage: str, name: str, value, suggested):
        target = self.trunk_params if stage in TRUNK else p.params
        d = target.setdefault(stage, {})
        if value == suggested:
            d.pop(name, None)
            if not d:
                target.pop(stage, None)
        else:
            d[name] = value
        self._sync()
        self.pathsChanged.emit()

    def _sync(self):
        """After a change or a run: which steps have ever been computed, which workflows are up to date."""
        for s in TRUNK:
            if self.trunk_key(s) in self.computed_trunk:
                self.ever.add((None, s))
        for q in self.paths:
            for s in self.path_stages(q):
                if self.stage_key(q, s) in self.computed_keys:
                    self.ever.add((q.pid, s))
            q.computed = not self.pending(q) and not self.trunk_pending()

    def mark_stages(self, items, emit=True):
        """A (simulated) run of these stages is over."""
        for p, s in items:
            if s in TRUNK:
                self.computed_trunk.add(self.trunk_key(s))
            else:
                self.computed_keys.add(self.stage_key(p, s))
        self._sync()
        for q in self.paths:
            if q.computed:
                q.rendered = True
        if emit:
            self.pathsChanged.emit()

    def all_pending(self) -> list[tuple[PathDef | None, str]]:
        seen, out = set(), [(None, s) for s in self.trunk_pending()]
        for p in self.paths:
            for s in self.pending(p):
                k = self.stage_key(p, s)
                if k not in seen:
                    seen.add(k)
                    out.append((p, s))
        return out

    # ------------------------------------------------------------ frame selection (sources and results share it)
    def suggested_thresholds(self):
        sp = {q["name"]: q["value"] for q in self.stages["stack"]["params"]}
        return 100 * sp["min_transparency"], sp["max_fwhm_ratio"] * float(np.median(self.data["frames"]["fwhm"]))

    def thresholds_now(self):
        return self.thresholds or self.suggested_thresholds()

    def excluded_mask(self):
        f = self.data["frames"]
        fw, tr = np.array(f["fwhm"]), np.array(f["transp"])
        trp = 100 * tr / np.percentile(tr, 90)
        tmin, fmax = self.thresholds_now()
        ex = (fw > fmax) | (trp < tmin)
        ex[list(self.frames_excluded)] = True
        return ex

    def kept_frames(self) -> int:
        ex = self.excluded_mask()
        return int(len(ex) - ex.sum())

    def set_selection(self, tmin=None, fmax=None, hand=None):
        t0, f0 = self.thresholds_now()
        tmin = t0 if tmin is None else tmin
        fmax = f0 if fmax is None else fmax
        self.thresholds = (tmin, fmax)
        self.frames_excluded = set(self.frames_excluded if hand is None else hand)
        st, sf = self.suggested_thresholds()
        sugg = abs(tmin - st) < 1e-6 and abs(fmax - sf) < 1e-6 and not self.frames_excluded
        self.set_trunk_param("stack", "selection", None if sugg else
                             (round(tmin, 1), round(fmax, 2), tuple(sorted(self.frames_excluded))))

    def set_trunk_param(self, stage, name, value):
        d = self.trunk_params.setdefault(stage, {})
        if value is None:
            d.pop(name, None)
        else:
            d[name] = value
        if not d:
            self.trunk_params.pop(stage, None)
        self._sync()
        self.costChanged.emit()

    # ------------------------------------------------------------ test mode
    def zone_fraction(self) -> float:
        gx0, gx1, gy0, gy1 = self.data["crop"]
        zx0, zy0, zx1, zy1 = self.test_zone
        return max(1e-4, (zx1 - zx0) * (zy1 - zy0) / ((gx1 - gx0) * (gy1 - gy0)))

    def test_estimate(self, p: PathDef) -> list[tuple[str, float, float, str]]:
        """(stage, full seconds, test seconds, how) for the workflow computed from the raw frames in test mode."""
        a, f = self.zone_fraction(), self.test_frames
        rows = []
        for s in TRUNK + self.path_stages(p):
            full = self.seconds(s)
            if s in ("inventory",):
                t, how = full, "inchangé"
            elif s in ("calibrate", "register"):
                t, how = full * f, "poses entières" + (", une partie des poses" if f < 1 else "")
            elif s == "stack":
                t, how = full * f * (a + 1 / 16), "zone en pleine résolution + image entière réduite ×4"
            elif s in ("astrometry", "catalogs", "surveys", "color"):
                t, how = full, "image entière : il faut tout le champ"
            elif s in ("background", "fond_simple"):
                t, how = full / 16 + (full * 0.1), "image entière réduite ×4 : il faut tout le champ"
            elif s == "denoise":
                k = max(min((a ** 0.5), 1.0), 0.25)
                t, how = full * k, f"apprentissage raccourci avec la surface (×{k:.2f}, jamais sous le quart)"
            elif s == "star_net":
                k = 0.25 if self.test_short_learned else 1.0
                t, how = full * k, ("apprentissage raccourci ÷4 : aperçu moins fidèle" if k < 1
                                    else "nombre de pas fixe : la zone ne raccourcit pas l'apprentissage")
            else:
                t, how = full * a, "zone seulement"
            rows.append((s, full, t, how))
        return rows
