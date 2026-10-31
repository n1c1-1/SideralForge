"""The project of mockup alt3, on top of alt2's model.

What changes from alt2: there is no common trunk any more. Each workflow is complete, from the calibration to its
layer, with its own settings at every step. Two workflows share a block when they compute it the same way, with the
same settings up to it: the graph draws it once, and the run computes it once. A change made for one workflow only
splits the branch at that block.

The project-level decisions (frames excluded by hand in the sources) count for every workflow.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ALT2 = Path(__file__).resolve().parents[2] / "alt2" / "app"
if str(ALT2) not in sys.path:
    sys.path.append(str(ALT2))                    # after alt3's own modules: alt3's versions win

from model import EXTRA, RELEVANT, TEMPLATES, TOOLS, TRUNK, PathDef, Session  # noqa: E402
from steps import BY_KEY, STEPS  # noqa: E402

SCENARIOS3 = {"neuf": "New project", "existant": "Existing project (Heart)"}

# ---- the blocks of alt3, on top of alt2's steps
# the order of the standard tools: astrometry, background, colour, then denoise, deconvolution, stars
ORDER = ["calibration", "alignement", "selection", "empilement", "demi_piles", "astrometrie", "fond", "couleur",
         "deconv", "debruitage", "etoiles", "appris", "sans_etoiles", "post"]
# the spec: the background is extracted before the colour calibration and the star separation (as in Siril and
# PixInsight), from the stack and the catalogue; it no longer depends on the starless image
RELEVANT["fond_simple"] = ()
BG_STAGES = ("background", "fond_simple")
EXTRA["deconv"] = ("Deconvolution", 150.0)                          # not in the engine yet; estimated
STEPS.append(dict(key="deconv", phase=4, title="Deconvolution", kind="tool", tool="deconv", stages=["deconv"],
                  what="Sharpens the image with the point spread function measured on the stars."))
STEPS.sort(key=lambda b: ORDER.index(b["key"]))
BY_KEY.clear()
BY_KEY.update({b["key"]: b for b in STEPS})
BY_KEY["etoiles"]["stages"] = ["optics", "stars", "star_net"]     # one Stars block: the neural removal is a method
TOOLS["deconv"] = dict(title="Deconvolution", phase=4,
                       what="Sharpens the image with the point spread function measured on the stars.")
HIDDEN_BLOCKS = {"appris"}                                         # now a method of the Stars block

# ---- methods: a block can be computed by a learned method (GPU) or a classic one (CPU, as in Siril)
# not in the engine yet: seconds on the GPU machine, estimated
EXTRA["denoise_ft"] = ("Denoise: pretrained, fine-tuned on the half-stacks", 180.0)
EXTRA["denoise_ai"] = ("Denoise: pretrained network", 45.0)
EXTRA["denoise_cl"] = ("Denoise: NL-Bayes", 300.0)
DN_STAGE = {"appris": "denoise", "affine": "denoise_ft", "generique": "denoise_ai", "nlbayes": "denoise_cl"}
BY_KEY["debruitage"]["stages"] = list(DN_STAGE.values())
# runs: "gpu" (needs a GPU), "cpu" (processor only), "both"; opt: the workflow option the method sets
METHODS = {
    "debruitage": [
        dict(key="appris", name="Noise2Noise on this image", short="Noise2Noise", runs="gpu",
             what="A network learns, on this image alone, to predict one half-stack from the other: the noise goes, "
                  "nothing is invented."),
        dict(key="affine", name="Pretrained + fine-tuned", short="fine-tuned", runs="both", new=True,
             what="Starts from a network trained on earlier projects and fine-tunes it on the half-stacks: nearly the "
                  "same fidelity, much shorter."),
        dict(key="generique", name="Pretrained (GraXpert-like)", short="pretrained", runs="both", new=True,
             what="A pretrained network applied as is: the fastest. May smooth or invent detail: compare."),
        dict(key="nlbayes", name="NL-Bayes (as in Siril)", short="NL-Bayes", runs="cpu", new=True,
             what="Averages similar patches after stabilising the photon noise. Classic, predictable."),
    ],
    "fond": [
        dict(key="halpha", name="H-alpha survey", short="H-alpha survey", runs="cpu",
             what="Tells the sky background from the instrument's defects with a professional H-alpha survey."),
        dict(key="simple", name="Polynomial (as in Siril)", short="polynomial", runs="cpu", new=True,
             what="A smooth surface fitted on dark samples."),
    ],
    "etoiles": [
        dict(key="psf", name="PSF model (Gaia)", short="PSF model", runs="both",
             what="Each star measured with the optics' real PSF, positioned by Gaia, then subtracted: star sliders in "
                  "Post."),
        dict(key="psf_neural", name="PSF model + neural removal", short="PSF + neural", runs="gpu",
             what="Adds a star-removal network trained on this stack (StarNet-like): a second starless image, cleaner "
                  "around bright stars."),
    ],
    "deconv": [
        dict(key="psf", name="Measured PSF (Richardson-Lucy)", short="Richardson-Lucy", runs="both", new=True,
             what="Classic deconvolution with the PSF measured across the field."),
        dict(key="neural", name="Neural (BlurXTerminator-like)", short="neural", runs="both", new=True,
             what="A pretrained network that corrects the PSF locally, stars included."),
    ],
}
METHOD_OPT = {"debruitage": "dn", "fond": "fond", "etoiles": "appris", "deconv": "deconv_m"}
DN_SUMMARY = {"affine": "fine-tuned", "generique": "pretrained", "nlbayes": "NL-Bayes"}
# without a GPU, each stage takes this many times longer (estimates of the mockup, to measure on the engine)
CPU_FACTOR = {"inventory": 1, "calibrate": 1.3, "register": 1.5, "stack": 5, "astrometry": 1, "catalogs": 1,
              "surveys": 1, "color": 1.5, "denoise": 45, "denoise_ft": 8, "denoise_ai": 8, "denoise_cl": 1,
              "deconv": 6, "optics": 3, "stars": 8, "star_net": 60,
              "starless": 4, "background": 2, "fond_simple": 1, "render_prep": 10, "render_prep_learned": 10,
              "rendu_simple": 2}
MACHINES = {True: "NVIDIA RTX 4060 Ti", False: "CPU only (no GPU)"}
DECONV_AFTER = ("optics", "stars", "star_net", "starless", "background", "fond_simple", "render_prep",
                "render_prep_learned", "rendu_simple")


def method_of(p: PathDef, block: str) -> str:
    if block == "fond":
        return p.opts["fond"]
    if block == "etoiles":
        return "psf_neural" if p.opts.get("appris") else "psf"
    if block == "deconv":
        return p.opts.get("deconv_m", "psf")
    return p.opts.get("dn", "appris")


def method_value(block: str, key: str):
    """The option value that selects a method."""
    if block == "etoiles":
        return key == "psf_neural"
    return key


# ---- sessions (nights) of the demo project; the second night is a demo addition
SESSIONS_DEMO = [
    dict(id=1, name="2026-08-16", night="16 → 17 Aug 2026", lights=721, exposure=15, iso=5000, temp=21.5,
         flats=46, flat_note="", darks="library", bias="none", dark_flats="none", rotation=0.0, filter="OSC",
         fwhm=3.9, kept=721),
    dict(id=2, name="2026-08-19", night="19 → 20 Aug 2026", lights=240, exposure=15, iso=5000, temp=18.0,
         flats=30, flat_note="", darks="library", bias="none", dark_flats="none", rotation=2.1, filter="OSC",
         fwhm=4.4, kept=226, demo=True),
]
MASTERS_DEMO = [
    dict(kind="Dark", exposure="15 s", iso=5000, temp="20 °C", frames=28, date="2026-08-17", used=True),
    dict(kind="Dark", exposure="30 s", iso=3200, temp="12 °C", frames=40, date="2026-03-02", used=False),
    dict(kind="Dark", exposure="120 s", iso=1600, temp="8 °C", frames=30, date="2025-12-11", used=False),
    dict(kind="Bias", exposure="1/4000 s", iso=5000, temp="—", frames=100, date="2026-01-05", used=False),
]
PROJECTS_DEMO = [
    dict(name="HeartSoul", target="IC 1805 · Heart nebula", nights=1, integration="3.0 h", date="2026-08-17",
         state="All workflows up to date", scenario="existant", thumb=True),
    dict(name="NGC 7822", target="Cepheus · emission nebula", nights=3, integration="6.2 h", date="2026-09-02",
         state="2 workflows · 1 outdated"),
    dict(name="Iris", target="NGC 7023 · reflection nebula", nights=1, integration="2.5 h", date="2026-07-28",
         state="1 workflow up to date"),
    dict(name="M31", target="Andromeda galaxy", nights=2, integration="3.1 h", date="2026-10-12",
         state="Never run"),
    dict(name="Rosette", target="NGC 2237", nights=1, integration="1.8 h", date="2026-02-14",
         state="1 workflow up to date"),
    dict(name="Horsehead", target="IC 434 / B33", nights=2, integration="3.4 h", date="2025-12-28",
         state="1 workflow up to date"),
]


class Project(Session):
    def __init__(self, scenario: str = "neuf"):
        self._proj: dict = {}                       # project-level settings: exclusions made in the sources
        self.scope: list[int] | None = None         # the workflows a change applies to (None: the current one)
        self.pending_change: dict | None = None     # a change that would throw away results, awaiting a choice
        self.gpu = True                             # the machine (demo menu)
        super().__init__("vierge")
        try:
            from datatext import englishize
            englishize(self.data)
        except ImportError:
            pass
        self.scenario = scenario
        self.sel_step = "sources"
        self.sessions: list[dict] = []
        self.stack_mode = "together"                # together | per_session
        self.history: list[tuple] = []              # (time, text) of the runs and changes, for the Log
        if scenario == "existant":
            self.sources_loaded = True
            self.sessions = [dict(SESSIONS_DEMO[0])]         # Demo > Add a second night adds the other one
            for name, opts in (("Classic stack", TEMPLATES["Vierge"]), ("Full", TEMPLATES["Complet"]),
                               ("Full + neural stars", {**TEMPLATES["Complet"], "appris": True})):
                layer, _ = self.closest_layer(opts)
                self.paths.append(PathDef(next(self._ids), name, dict(opts), layer))
            self.mark_stages([(p, s) for p in self.paths for s in self.all_stages(p)], emit=False)
            self.current = 1
            self.sel_step = "empilement"
            self.log("Project opened: 3 workflows, all up to date")

    def log(self, text):
        import time
        self.history.append((time.strftime("%H:%M:%S"), text))

    def load_sources(self):
        self.sessions = [dict(SESSIONS_DEMO[0])]
        self.log("Folder imported: 1 session, 721 lights, 46 flats; darks from the master library")
        super().load_sources()

    def add_session(self):
        """Demo: a second night added to the project."""
        if len(self.sessions) < 2:
            self.sessions.append(dict(SESSIONS_DEMO[1]))
            self._proj["sessions"] = len(self.sessions)
            self.log("Session 2026-08-19 added: 240 lights, 30 flats")
            self._sync()
            self.pathsChanged.emit()
            return True
        return False

    def set_stack_mode(self, mode):
        """Several nights: one stack of all, or one per night, made as variants of the current workflow to compare
        the nights side by side."""
        if self.stack_mode == mode:
            return
        self.stack_mode = mode
        if mode == "per_session" and self.path is not None:
            base = self.path
            for ses in self.sessions:
                ch = dict(kind="param", stage="stack", name="sessions", value=(ses["id"],), suggested=None)
                self.add_variant(base, ch, f"{base.name} · {ses['name']}")
            self.log(f"Stacking per session: {len(self.sessions)} variants of {base.name}")
        self.pathsChanged.emit()

    def n_lights(self) -> int:
        return sum(s["kept"] for s in self.sessions) or self.data["inventory"]["n_lights"]

    # ------------------------------------------------------------ the Deconvolution tool (new block)
    @staticmethod
    def has_tool(o, key):
        if key == "deconv":
            return bool(o.get("deconv"))
        if key not in ("debruitage", "etoiles", "appris", "halpha", "simple"):
            return False
        return Session.has_tool(o, key)

    @staticmethod
    def with_tool(o, key):
        if key == "deconv":
            return {**o, "deconv": True}, []
        return Session.with_tool(o, key)

    @staticmethod
    def without_tool(o, key):
        if key == "deconv":
            return {**o, "deconv": False}, []
        new, notes = Session.without_tool(o, key)
        return new, notes

    # ------------------------------------------------------------ alt2 compatibility
    @property
    def trunk_params(self):
        """alt2's settings code reads the trunk settings here: in alt3 they belong to the current workflow."""
        p = self.path
        return p.params if p is not None else {}

    @trunk_params.setter
    def trunk_params(self, value):
        pass

    def trunk_pending(self):
        return []

    def trunk_seconds(self) -> float:
        return 0.0

    # ------------------------------------------------------------ a workflow pulled to the post from a block
    @staticmethod
    def end_of(p: PathDef):
        """The block a workflow's layer comes out of, when it does not go to the end (None otherwise)."""
        return getattr(p, "end", None)

    def _kept(self, p: PathDef) -> list[str]:
        """The stages of the full chain that come before the workflow's exit, in order."""
        from steps import STEPS
        end = self.end_of(p)
        keys = [b["key"] for b in STEPS]
        upto = set()
        for b in STEPS[: keys.index(end) + 1]:
            upto |= set(b["stages"])
        if "starless" in Session.path_stages(self, p) and keys.index(end) >= keys.index("etoiles"):
            upto.add("starless")
        return [s for s in Session.path_stages(self, p) if s in upto]

    def path_stages(self, p: PathDef) -> list[str]:
        st = super().path_stages(p) if self.end_of(p) is None else self._kept(p) + ["rendu_simple"]
        dn = DN_STAGE[p.opts.get("dn", "appris")]
        if dn != "denoise":
            st = [dn if s == "denoise" else s for s in st]
        if p.opts.get("deconv") and (self.end_of(p) is None or ORDER.index(self.end_of(p)) >= ORDER.index("deconv")):
            st.insert(st.index("color") + 1, "deconv")        # before the denoising, as BlurX before NoiseX
        return st

    def deps(self, p: PathDef, stage: str) -> list[str]:
        dn = DN_STAGE[p.opts.get("dn", "appris")]
        if stage == "rendu_simple" and self.end_of(p) is not None:
            kept = self._kept(p)
            k = kept[-1] if kept else "stack"
            return [dn if k == "denoise" else k]
        here = set(TRUNK) | set(self.path_stages(p))
        if stage in BG_STAGES:
            return [x for x in ("stack", "astrometry", "catalogs", "surveys") if x in here]
        if stage == "deconv":
            return ["color"]
        if stage in DN_STAGE.values():
            return ["stack"] + (["deconv"] if "deconv" in here else [])
        d = super().deps(p, stage)
        if stage == "color":
            d = d + [x for x in BG_STAGES if x in here]
        if stage in DECONV_AFTER and "deconv" in here:
            d = d + ["deconv"]
        if dn != "denoise" and "denoise" in self.stages.get(stage, {}).get("deps", []):
            d = d + [dn]
        return d

    # ------------------------------------------------------------ the machine
    def seconds(self, stage: str) -> float:
        base = super().seconds(stage)
        return base if self.gpu or stage == "denoise_cl" else base * CPU_FACTOR.get(stage, 2)

    def set_machine(self, gpu: bool):
        self.gpu = gpu
        self.pathsChanged.emit()

    def add_path(self, name, opts, params=None):
        opts = dict(opts)
        if opts.get("debruitage") and "dn" not in opts and not self.gpu:
            opts["dn"] = "affine"                      # without a GPU: the network already learned, fine-tuned
        return self._add_path(name, opts, params)

    def add_exit(self, src: PathDef, end: str, name: str) -> PathDef:
        """A new workflow: the same blocks as src up to `end`, whose output goes straight to the post."""
        opts = dict(src.opts)
        layer = "classique_dn" if opts["debruitage"] and end in ("debruitage", "etoiles", "appris") else "classique"
        p = PathDef(next(self._ids), self.unique_name(name), opts, layer, computed=False, provisional=True,
                    params=copy.deepcopy(src.params))
        p.end = end
        self.paths.append(p)
        self._sync()
        p.rendered = p.computed
        self.pathsChanged.emit()
        return p

    def slider_state(self, p: PathDef, key: str):
        end = self.end_of(p)
        if end is not None:
            from blocks import COLUMNS
            reach = COLUMNS.index(end)
            if key in ("nombre", "taille", "couleur", "halo", "detail") and reach < COLUMNS.index("etoiles"):
                return False, "this layer leaves the graph before the Stars block", None
            if key == "bruit" and (not p.opts["debruitage"] or reach < COLUMNS.index("debruitage")):
                return False, "this layer leaves the graph before the Denoise block", None
        return super().slider_state(p, key)

    # ------------------------------------------------------------ a change that would throw away computed results
    @staticmethod
    def _params_with(p: PathDef, stage, name, value, suggested) -> dict:
        params = copy.deepcopy(p.params)
        d = params.setdefault(stage, {})
        if value == suggested or value is None:
            d.pop(name, None)
        else:
            d[name] = value
        if not d:
            params.pop(stage, None)
        return params

    def _with(self, p: PathDef, ch) -> tuple[dict, dict]:
        """(options, settings) of p once the change is made. A change is a setting (kind "param": stage, name,
        value, suggested) or an option such as a method (kind "opt": key, value)."""
        if ch["kind"] == "opt":
            return {**p.opts, ch["key"]: ch["value"]}, copy.deepcopy(p.params)
        return dict(p.opts), self._params_with(p, ch["stage"], ch["name"], ch["value"], ch["suggested"])

    def _tmp(self, p: PathDef, ch) -> PathDef:
        opts, params = self._with(p, ch)
        t = PathDef(-p.pid, p.name, opts, p.layer, params=params)
        if self.end_of(p) is not None:
            t.end = p.end
        return t

    def throws_away(self, paths, stage) -> bool:
        """Changing this stage would invalidate something already computed in one of these workflows."""
        return any(not self.is_pending(q, x) for q in paths for x in self.downstream(q, stage))

    def cost_with(self, paths, ch) -> float:
        """What would have to be computed, in seconds, once the change is made in these workflows."""
        keys = {}
        for q in paths:
            t = self._tmp(q, ch)
            for x in self.all_stages(t):
                k = self.stage_key(t, x)
                if k not in self.computed_keys:
                    keys[k] = self.seconds(x)
        return sum(keys.values())

    def apply_change(self, paths, ch):
        if ch["kind"] == "opt":
            for q in paths:
                self.set_opts(q, {**q.opts, ch["key"]: ch["value"]})
        else:
            self.scope = [q.pid for q in paths]
            self.set_param(None, ch["stage"], ch["name"], ch["value"], ch["suggested"])

    def add_variant(self, p: PathDef, ch, title) -> PathDef:
        """A copy of p with the change; p stays as it was. Both share everything before the changed block."""
        opts, params = self._with(p, ch)
        q = PathDef(next(self._ids), self.unique_name(title), opts, p.layer, computed=False,
                    provisional=p.provisional, params=params)
        if self.end_of(p) is not None:
            q.end = p.end
        q.variant_of = p.pid
        self.paths.append(q)
        self._sync()
        q.rendered = q.computed
        self.pathsChanged.emit()
        return q

    def keep_variant(self, q: PathDef):
        """The variant replaces its original, and takes its name."""
        p = next((x for x in self.paths if x.pid == getattr(q, "variant_of", None)), None)
        if p is not None:
            name = p.name
            self.paths.remove(p)
            q.name = name
        q.variant_of = None
        self.current = self.paths.index(q)
        self._sync()
        self.pathsChanged.emit()

    def drop_variant(self, q: PathDef):
        p = next((x for x in self.paths if x.pid == getattr(q, "variant_of", None)), None)
        self.paths.remove(q)
        self.current = self.paths.index(p) if p in self.paths else min(self.current, len(self.paths) - 1)
        self.pathsChanged.emit()

    # ------------------------------------------------------------ stages and keys
    def all_stages(self, p: PathDef) -> list[str]:
        return TRUNK + self.path_stages(p)

    def path_seconds(self, p: PathDef) -> float:
        return sum(self.seconds(s) for s in self.all_stages(p))

    def stage_key(self, p: PathDef, stage: str):
        """Two workflows compute a stage the same way when this key is equal: same options and settings up to it."""
        chain = sorted({stage} | self.upstream(p, stage))
        rel = tuple(sorted({(k, p.opts[k]) for s in chain for k in RELEVANT.get(s, ())}))
        sets = tuple((s, tuple(sorted(p.params[s].items()))) for s in chain if p.params.get(s))
        return stage, rel, sets, tuple(sorted(self._proj.items()))

    def is_pending(self, p, stage):
        p = p or self.path
        return p is None or self.stage_key(p, stage) not in self.computed_keys

    def state(self, p, stage):
        """done | pending (never computed) | stale (computed once, a setting changed since)."""
        p = p or self.path
        if p is None:
            return "pending"
        if not self.is_pending(p, stage):
            return "done"
        return "stale" if (p.pid, stage) in self.ever else "pending"

    def pending(self, p: PathDef) -> list[str]:
        return [s for s in self.all_stages(p) if self.is_pending(p, s)]

    def cost_of(self, opts, params=None):
        tmp = PathDef(0, "", dict(opts), "", params=copy.deepcopy(params or {}))
        todo, reused = [], []
        for s in self.all_stages(tmp):
            (reused if self.stage_key(tmp, s) in self.computed_keys else todo).append(s)
        return sum(self.seconds(s) for s in todo), todo, reused

    def pending_upto(self, p, stages):
        want = set()
        for s in stages:
            want |= {s} | self.upstream(p, s)
        return [(p, x) for x in self.all_stages(p) if x in want and self.is_pending(p, x)]

    def sharing(self, p: PathDef, stage: str) -> list[PathDef]:
        """The workflows that compute this stage exactly as p does (p included)."""
        k = self.stage_key(p, stage)
        return [q for q in self.paths if stage in self.all_stages(q) and self.stage_key(q, stage) == k]

    def all_pending(self):
        seen, out = set(), []
        for p in self.paths:
            for s in self.pending(p):
                k = self.stage_key(p, s)
                if k not in seen:
                    seen.add(k)
                    out.append((p, s))
        return out

    # ------------------------------------------------------------ changes
    def scope_paths(self) -> list[PathDef]:
        if self.scope:
            out = [q for q in self.paths if q.pid in self.scope]
            if out:
                return out
        return [self.path] if self.path is not None else []

    def _add_path(self, name, opts, params=None):
        params = copy.deepcopy(params or {})
        if not params and self.path is not None:              # a new workflow joins the current pre-processing
            params = {s: dict(v) for s, v in self.path.params.items() if s in TRUNK}
        return Session.add_path(self, name, opts, params)

    def set_param(self, p, stage, name, value, suggested):
        for q in self.scope_paths() if p is None else [p]:
            d = q.params.setdefault(stage, {})
            if value == suggested:
                d.pop(name, None)
                if not d:
                    q.params.pop(stage, None)
            else:
                d[name] = value
        self._sync()
        self.pathsChanged.emit()

    def _sync(self):
        for q in self.paths:
            for s in self.all_stages(q):
                if self.stage_key(q, s) in self.computed_keys:
                    self.ever.add((q.pid, s))
            q.computed = not self.pending(q)

    def mark_stages(self, items, emit=True):
        for p, s in items:
            for q in self.sharing(p, s) or [p]:
                self.computed_keys.add(self.stage_key(q, s))
        self._sync()
        for q in self.paths:
            if q.computed:
                q.rendered = True
        if emit:
            self.pathsChanged.emit()

    # ------------------------------------------------------------ frame selection: thresholds per workflow
    def thresholds_now(self):
        p = self.path
        sel = p.params.get("stack", {}).get("selection") if p is not None else None
        return (sel[0], sel[1]) if sel else self.suggested_thresholds()

    def set_selection(self, tmin=None, fmax=None, hand=None):
        t0, f0 = self.thresholds_now()
        tmin = t0 if tmin is None else tmin
        fmax = f0 if fmax is None else fmax
        if hand is not None:
            self.frames_excluded = set(hand)
            if self.frames_excluded:
                self._proj["frames_excluded"] = tuple(sorted(self.frames_excluded))
            else:
                self._proj.pop("frames_excluded", None)
        st, sf = self.suggested_thresholds()
        sugg = abs(tmin - st) < 1e-6 and abs(fmax - sf) < 1e-6
        value = None if sugg else (round(tmin, 1), round(fmax, 2))
        for q in self.scope_paths():
            d = q.params.setdefault("stack", {})
            if value is None:
                d.pop("selection", None)
            else:
                d["selection"] = value
            if not d:
                q.params.pop("stack", None)
        self._sync()
        self.costChanged.emit()

    def set_trunk_param(self, stage, name, value):
        """Decisions taken in the sources (calibration frames excluded): they count for every workflow."""
        k = f"{stage}.{name}"
        if value is None:
            self._proj.pop(k, None)
        else:
            self._proj[k] = value
        self._sync()
        self.costChanged.emit()
