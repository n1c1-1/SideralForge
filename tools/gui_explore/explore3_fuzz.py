"""GUI exploration, round 2, area "fuzz": the round-1 fuzz (explore_fuzz.py) with more gesture kinds and invariants.

New gestures: save + reopen (Ctrl+S, the project closed and opened again), Post layers (their menu, their eye),
files dropped on a Session's lists, a placed block moved in the graph, a block double-clicked (Inspect), Run (F5,
Shift+F5: blocked, nothing computes), the language switched at runtime.
New invariants: the project saved and reopened equals what was shown (definition and drawn graph); every block drawn
has a name and its panel shows something; no Post layer of a workflow that is gone or of a part it no longer gives.

    python explore2_fuzz.py run 1017
    python explore2_fuzz.py drive 1000 1040 2
    python explore2_fuzz.py summary
    python explore2_fuzz.py show <runs2/seed_N.json>
    python explore2_fuzz.py minimize <sig>
    python explore2_fuzz.py confirm <case.json>          # replayed twice; noted when the case has a title
"""
from __future__ import annotations

import faulthandler
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
_argv = sys.argv
sys.argv = [sys.argv[0], "drive" if MODE in ("drive", "summary", "show", "minimize_all") else "run"]
import explore_fuzz as F  # noqa: E402
sys.argv = _argv

OUT = Path("F:/astro_essais/A_REGARDER/exploration3/fuzz")
F.OUT = OUT
F.RUNS = OUT / "runs_r1"
RUNS = OUT / "runs3"
RUNS.mkdir(parents=True, exist_ok=True)
CASES = OUT / "cases3"
PY = sys.executable

ACTIONS = [("dbl_card", 9), ("drag_card", 12), ("click_block", 6), ("menu", 14), ("cross", 4), ("del", 4),
           ("d", 3), ("to_palette", 4), ("pull_port", 4), ("wire", 4), ("undo", 8), ("redo", 5), ("add_session", 3),
           ("panel", 9), ("files", 4), ("image_file", 3), ("mode", 3), ("lang", 2), ("enter", 1), ("esc", 1),
           ("save_reopen", 3), ("post_layer", 4), ("drop_urls", 3), ("move_block", 3), ("dbl_block", 2),
           ("run", 2), ("panel_type", 5)]
if os.environ.get("FUZZ_PROFILE") == "struct":     # structure edits first: few no-op gestures (seeds 4000+)
    ACTIONS = [("dbl_card", 12), ("drag_card", 14), ("menu", 18), ("cross", 5), ("del", 5), ("d", 3),
               ("to_palette", 6), ("pull_port", 4), ("undo", 10), ("redo", 6), ("add_session", 3), ("panel", 6),
               ("save_reopen", 3), ("lang", 2), ("move_block", 2), ("click_block", 4)]
NEW = {"save_reopen", "post_layer", "drop_urls", "move_block", "dbl_block", "run", "panel_type"}
F.NEEDS_GRAPH |= {"drop_urls", "move_block", "dbl_block"}
F.ORDINARY = (F.NEEDS_GRAPH - {"panel", "files", "image_file", "add_session", "enter", "drop_urls", "dbl_block"}) | \
    {"post_layer"}
F.SEV.update({"reopen": "gênant", "noname": "gênant", "nopanel": "gênant", "orphan_layer": "gênant",
              "run_nodata": "gênant"})


def _pick(lst, r):
    """round 1's pick, where a text names a plain string too (a menu item), and a palette card by its key or title"""
    if not lst:
        return None
    if isinstance(r, str):
        def name(x):
            if isinstance(x, str):
                return x
            if isinstance(x, tuple):
                return " ".join(y if isinstance(y, str) else str(getattr(y, "title", y)) for y in x[:2])
            return str(getattr(x, "title", x))
        hit = [x for x in lst if r == name(x)] or [x for x in lst if r in name(x).split(" ")] or \
            [x for x in lst if r in name(x)]
        return hit[0] if hit else None
    return lst[min(int(r * len(lst)), len(lst) - 1)]


F.pick = _pick


def make_sequence(seed: int) -> dict:
    rng = random.Random(seed * 7919 + 13)
    start = {"kind": rng.choice(F.KINDS)} if rng.random() < 0.5 else {"template": rng.choice(F.TEMPLATES)}
    start["lang"] = "fr" if rng.random() < 0.6 else "en"
    names, weights = zip(*ACTIONS)
    n = rng.randint(80, 120) if os.environ.get("FUZZ_LONG") else rng.randint(18, 40)   # long ones: seeds 6000+
    acts = [[rng.choices(names, weights)[0], rng.random(), rng.random(), rng.random()] for _ in range(n)]
    return {"seed": seed, "start": start, "actions": acts}


class Fuzz2(F.Fuzz):
    def __init__(self, lang):
        super().__init__(lang)
        self._rebase = None

    def start(self, start, tag):
        super().start(start, tag)
        from astro_gpu.client import prefs               # the harness applies a language without its setting: the
        from astro_gpu.client.i18n import language       # Settings' switch would first pick the language shown
        prefs.set("language", language())

    # ------------------------------------------------------------ new gestures
    def act(self, a):
        from PySide6.QtCore import QPoint, Qt
        name, r1, r2, r3 = a
        if name not in NEW:
            return super().act(a)
        ex = self.ex
        self.fileans = r3
        desc: list[str] = []
        if ex.w.project is None:
            return "(pas de projet ouvert)"
        self.activate()
        if name in F.NEEDS_GRAPH:
            self.ensure_graph(desc)
        if name == "save_reopen":
            return self.save_reopen(desc)
        if name == "run":
            k = Qt.Key_F5
            mod = Qt.ShiftModifier if F.lt(r1, 0.5) else Qt.NoModifier
            if mod == Qt.ShiftModifier:
                its = self.items()
                it = F.pick(its, r2)
                if it is not None:
                    desc.append(f"clic sur le bloc « {self.title(it)} »")
                    ex.click_block(it)
            desc.append(("Maj+F5" if mod == Qt.ShiftModifier else "F5") + " (exécution bloquée par le banc)")
            ex.key(k, ex.w, mod)
            ex.settle(6)
            p = ex.w.s.p
            has = any(s.lights for s in p.sessions) or any((w.source or {}).get("path") for w in p.workflows)
            opened = [d for d in ex.dialogs() if d.__class__.__name__ == "RunDialog"]
            if opened and not has:
                self.fail("run_nodata", "RunDialog", "la fenêtre d'exécution s'ouvre alors que le projet n'a ni brute "
                                                     "ni fichier image")
            for d in opened:                             # the run window (a run that cannot start): closed, as the
                d.reject()                               # user would
            ex.settle(4)
            return "; ".join(desc)
        if name == "dbl_block":
            it = F.pick(self.items(), r1)
            if it is None:
                return "(aucun bloc)"
            desc.append(f"double-clic sur le bloc « {self.title(it)} »")
            ex.double_click_block(it)
            return "; ".join(desc)
        if name == "move_block":
            it = F.pick(self.items(), r1)
            if it is None:
                return "(aucun bloc)"
            vp = ex.view.viewport()
            a_ = ex.point_of(it)
            b_ = QPoint(int(vp.width() * (0.05 + 0.9 * r2)), int(vp.height() * (0.05 + 0.9 * r3)))
            desc.append(f"glisser le bloc « {self.title(it)} » ailleurs dans le graphe")
            ex.drag_mouse(vp, a_, b_, vp, b_)
            return "; ".join(desc)
        if name == "drop_urls":
            return self.drop_urls(desc, r1, r2, r3)
        if name == "panel_type":                         # a value typed in a field, not validated: the field keeps
            from PySide6.QtTest import QTest              # the cursor into the next gesture
            from PySide6.QtWidgets import QAbstractSpinBox, QLineEdit
            it = F.pick(self.items(), r1)
            if it is not None:
                desc.append(f"clic sur le bloc « {self.title(it)} »")
                ex.click_block(it)
            pn = ex.page.panel
            eds = [w for w in pn.findChildren(QAbstractSpinBox) + pn.findChildren(QLineEdit)
                   if w.isVisible() and w.isEnabled() and not isinstance(w.parent(), QAbstractSpinBox)]
            e = F.pick(eds, r2)
            if e is None:
                return "; ".join(desc + ["(panneau: aucun champ)"])
            tip = (e.toolTip() or "").split("\n")[0][:40]
            ex.click(e)
            e.setFocus()
            if isinstance(e, QAbstractSpinBox):
                QTest.keyClick(e, Qt.Key_Up if F.lt(r3, 0.5) else Qt.Key_Down)
                desc.append(f"panneau: flèche dans le champ « {tip or type(e).__name__} », sans valider")
            else:
                e.selectAll()
                QTest.keyClicks(e, "Nuit 3" if F.lt(r3, 0.5) else "")
                desc.append(f"panneau: taper dans le champ « {e.text()[:30]} », sans Entrée")
            ex.settle()
            return "; ".join(desc)
        if name == "post_layer":
            return self.post_layer(desc, r1, r2, r3)
        return "?"

    def save_reopen(self, desc):
        from PySide6.QtCore import Qt
        ex = self.ex
        w = ex.w
        root = w.project.root
        self.ensure_graph(desc)
        before = self.state()
        g0 = self.drawn()
        desc.append("Ctrl+S, puis fermer le projet et le rouvrir")
        ex.key(Qt.Key_S, w, Qt.ControlModifier)
        self.wait_ready()
        w.close_project()
        ex.settle(4)
        w.open_project(root)
        self.wait_ready()
        if w.mode != 0:
            ex.click(w.mode_btns[0])
            self.wait_ready()
        after = self.state()
        d = self.diff(before, after)
        if d:
            self.fail("reopen", ",".join(d), "le projet rouvert n'est pas celui qui était affiché: " +
                      " ; ".join(self.explain(before, after, c) for c in d))
        g1 = self.drawn()
        if g0 != g1:
            only0 = sorted(set(g0) - set(g1))
            only1 = sorted(set(g1) - set(g0))
            self.fail("reopen", "graph", f"le graphe rouvert n'est pas celui affiché: avant seulement {only0[:8]}, "
                                         f"après seulement {only1[:8]}")
        s = w.s
        self._rebase = (after, s._undo[-1] if s._undo else None, len(s._undo))
        return "; ".join(desc)

    def drawn(self) -> list:
        return sorted((self.title(it), it.node.key, str(it.summary or ""), str(it.bp), bool(it.node.cut), bool(it.dim))
                      for it in self.ex.blocks())

    def drop_urls(self, desc, r1, r2, r3):
        from PySide6.QtCore import QMimeData, QPoint, QUrl
        from PySide6.QtWidgets import QWidget
        ex = self.ex
        its = [it for it in self.items() if it.node.key in ("session", "sources")]
        it = F.pick(its, r1)
        if it is None:
            return "(pas de bloc Session)"
        desc.append(f"clic sur le bloc « {self.title(it)} »")
        ex.click_block(it)
        pn = ex.page.panel
        zones = [x for x in pn.findChildren(QWidget) if x.isVisible() and x.acceptDrops()
                 and type(x).__module__.startswith("astro_gpu")]       # the app's drop zones, not a text field's
        z = F.pick(zones, r2)
        target = None
        if z is None:                                    # none in the panel: on the Session block in the graph
            it = next((x for x in self.items() if x.node.key in ("session", "sources")), None)
            if it is None:
                return "; ".join(desc + ["(aucune zone où déposer)"])
            z = ex.view.viewport()
            target = ex.point_of(it)
        self._data += 1
        k = self._data
        kind = F.pick(["raws", "folder", "jpg", "fits", "mixed"], r3)
        if kind == "raws":
            d = ex.raws(f"url{k}", 4, start=9000 + 10 * k)
            paths = [str(x) for x in sorted(d.iterdir())]
        elif kind == "folder":
            paths = [str(ex.raws(f"urld{k}", 5, start=9500 + 10 * k))]
        elif kind == "fits":
            paths = [str(ex.fits_frames(f"urlf{k}_Ha", 3, "Ha"))]
        else:
            d = ex.raws(f"urlj{k}", 2, start=9800 + 10 * k)
            from PIL import Image
            import numpy as np
            j = d / "PHONE_0002.JPG"
            Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(j)
            paths = [str(j)] if kind == "jpg" else [str(x) for x in sorted(d.iterdir())]
        what = "le bloc Session du graphe" if target is not None else \
            f"la zone « {getattr(z, 'key', None) or z.__class__.__name__} » du panneau"
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        n0 = sum(len(s.lights) for s in ex.s.p.sessions)
        ok = ex._drop(z, target if target is not None else z.rect().center(), mime)
        self.wait_ready()
        n1 = sum(len(s.lights) for s in ex.s.p.sessions)
        desc.append(f"déposer {len(paths)} élément(s) ({kind}) depuis l'Explorateur sur {what}"
                    f" ({'accepté' if ok else 'refusé'}, entrées de brutes {n0} → {n1})")
        return "; ".join(desc)

    def post_layer(self, desc, r1, r2, r3):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtWidgets import QApplication, QCheckBox
        ex = self.ex
        w = ex.w
        if w.mode != 2:
            ex.click(w.mode_btns[2])
            desc.append("clic sur « " + w.mode_btns[2].text() + " »")
            self.wait_ready()
        from astro_gpu.client.post3 import LayerRow
        rows = [x for x in w.findChildren(LayerRow) if x.isVisible()]
        row = F.pick(rows, r1)
        if row is None:
            return "; ".join(desc + ["(Post: aucun calque)"])
        name = row.n.text() if hasattr(row, "n") else "?"
        if F.lt(r2, 0.3):
            eye = row.findChild(QCheckBox)
            desc.append(f"Post: clic sur l'œil du calque « {name} »")
            if eye is not None:
                ex.click(eye)
        else:
            n = len(ex.menus)
            ex.menu_pick = None
            p = row.rect().center()
            QApplication.sendEvent(row, QMouseEvent(QEvent.MouseButtonPress, QPointF(p), QPointF(row.mapToGlobal(p)),
                                                    Qt.RightButton, Qt.RightButton, Qt.NoModifier))
            ex.settle()
            items = ex.menus[-1] if len(ex.menus) > n else []
            leaves = F._leaves(items)
            txt = F.pick(leaves, r3)
            if txt is None:
                desc.append(f"Post: clic droit sur le calque « {name} » (menu vide)")
            else:
                desc.append(f"Post: clic droit sur le calque « {name} » → « {txt} »")
                rows = [x for x in w.findChildren(LayerRow) if x.isVisible() and x.lay.get("id") == row.lay.get("id")]
                if rows:
                    ex.menu_pick = txt
                    r = rows[0]
                    QApplication.sendEvent(r, QMouseEvent(QEvent.MouseButtonPress, QPointF(p),
                                                          QPointF(r.mapToGlobal(p)), Qt.RightButton, Qt.RightButton,
                                                          Qt.NoModifier))
                    ex.settle(6)
        if F.lt(r3, 0.7):                                # back to the graph most of the time: its checks run
            ex.click(w.mode_btns[0])
            desc.append("clic sur « " + w.mode_btns[0].text() + " »")
            self.wait_ready()
        return "; ".join(desc)

    # ------------------------------------------------------------ new invariants
    def check(self, name, before, last_undo, popups0, modals0):
        super().check(name, before, last_undo, popups0, modals0)
        ex = self.ex
        if ex.w.project is None or ex.w.s is None or not ex.w.s.ready:
            return
        try:
            self.check_post_layers()
        except Exception:  # noqa: BLE001
            self.fail("harness", "post_layers", traceback.format_exc()[-1500:])
        if ex.w.mode != 0:
            return
        for it in ex.blocks():
            if not (it.title or "").strip():
                self.fail("noname", it.node.key, f"un bloc dessiné sans nom (clé {it.node.key})")

    def check_post_layers(self):
        s = self.ex.w.s
        wids = {w.id for w in s.p.workflows}
        for src, layers in (("Post", s.post_layers()),
                            ("enregistré", list(((getattr(s.p, "post", None) or {}).get("layers")) or []))):
            for lay in layers:
                if lay.get("kind") != "workflow":
                    continue
                if lay.get("pid") not in wids:
                    self.fail("orphan_layer", "gone", f"calque « {lay.get('name')} » ({src}) d'un workflow qui "
                                                      f"n'existe plus (id {lay.get('pid')})")
                    continue
                part = lay.get("part")
                if part:
                    q = s.view(lay["pid"])
                    keys = s.display_keys(q) if q is not None else []
                    fusion_stars = bool(((q.source or {}).get("fusion") or {}).get("stars")) if q is not None else False
                    if "etoiles" not in keys and not fusion_stars:
                        self.fail("orphan_layer", f"part:{part}", f"calque « {lay.get('name')} » ({src}) : partie "
                                  f"« {part} » d'un workflow sans bloc Étoiles/fond ({', '.join(keys)})")

    def check_panels(self):
        """Every block drawn: a click shows its panel (something in it), no Python error."""
        ex = self.ex
        if ex.w.project is None or ex.w.s is None:
            return
        self.ensure_graph([])
        n = len(self.items())
        for i in range(min(n, 30)):
            its = self.items()
            if i >= len(its):
                break
            it = its[i]
            t = self.title(it)
            key = it.node.key
            ex.click_block(it)
            self.wait_ready(5)
            pn = ex.page.panel
            cnt = pn.hv.count() + pn.bv.count()
            if pn.node is None or cnt == 0:
                self.fail("nopanel", key, f"clic sur le bloc « {t} » : le panneau reste vide")
            for e in self.errs:
                self.fail("exc", f"{F._exc_type(e)}@{F._where(e)}", f"(clic sur « {t} » pour voir son panneau)\n" + e)
            self.errs.clear()

    # ------------------------------------------------------------ one sequence (the round-1 run, with a reopen rebase)
    def run(self, seq, tag, stop_sig=None, progress=None, watchdog=90.0, hangfile=None):
        ex = self.ex
        t0 = time.time()
        res = {"seed": seq.get("seed"), "start": seq["start"], "actions": seq["actions"], "steps": [], "failures": [],
               "done": 0}
        faulthandler.dump_traceback_later(watchdog * 2, exit=True, file=hangfile or sys.stderr)
        self.step_fails = []
        self._rebase = None
        try:
            self.start(seq["start"], tag)
        except Exception:  # noqa: BLE001
            res["failures"].append({"step": -1, "kind": "harness", "sig": "harness:start",
                                    "detail": traceback.format_exc()[-2000:]})
            return res
        for e in self.errs:
            res["failures"].append({"step": -1, "kind": "exc", "sig": f"exc:{F._exc_type(e)}@{F._where(e)}",
                                    "detail": "(à l'ouverture)\n" + e})
        self.errs.clear()
        self.qt.clear()
        s0 = self.state() if ex.w.s is not None else None
        u0 = ex.w.s._undo[-1] if ex.w.s is not None and ex.w.s._undo else None
        n_u0 = len(ex.w.s._undo) if ex.w.s is not None else 0
        seen: dict[str, dict] = {}

        def take(i):
            for f in self.step_fails:
                f["step"] = i
                if f["sig"] in seen:
                    seen[f["sig"]]["count"] = seen[f["sig"]].get("count", 1) + 1
                    continue
                seen[f["sig"]] = f
                res["failures"].append(f)
        for i, a in enumerate(seq["actions"]):
            if progress is not None:
                progress.write_text(json.dumps({"seed": seq.get("seed"), "step": i, "action": a,
                                                "steps": res["steps"]}, ensure_ascii=False), encoding="utf-8")
            faulthandler.dump_traceback_later(watchdog, exit=True, file=hangfile or sys.stderr)
            self.step_fails = []
            s = ex.w.s
            before = self.state() if s is not None else None
            last = s._undo[-1] if s is not None and s._undo else None
            p0, m0 = len(ex.popups), len(ex.modals)
            self._menus0, self._popups0 = len(ex.menus), len(ex.popups)
            try:
                d = self.act(a)
            except F.RunsBlocked:
                d = f"{a[0]} (lancement bloqué)"
            except Exception:  # noqa: BLE001
                d = f"{a[0]} (erreur du banc)"
                self.fail("harness", a[0], traceback.format_exc()[-2000:])
            res["steps"].append(d)
            if not self.wait_ready():
                self.fail("never_ready", a[0], "le projet reste « en lecture » plus de 25 s")
            try:
                self.check(a[0] if a[0] != "save_reopen" else "undo", before, last, p0, m0)
            except Exception:  # noqa: BLE001
                self.fail("harness", "check:" + a[0], traceback.format_exc()[-2000:])
            if self._rebase is not None:                 # reopened: Undo starts again from there
                s0, u0, n_u0 = self._rebase
                self._rebase = None
            take(i)
            res["done"] = i + 1
            if stop_sig and any(f["sig"] == stop_sig for f in self.step_fails):
                res["stopped"] = i
                faulthandler.cancel_dump_traceback_later()
                return res
            if any(f["kind"] == "validate" for f in self.step_fails):
                res["broken"] = i
                faulthandler.cancel_dump_traceback_later()
                res["time"] = round(time.time() - t0, 1)
                return res
        faulthandler.dump_traceback_later(watchdog * 4, exit=True, file=hangfile or sys.stderr)
        self.step_fails = []
        if ex.w.s is not None and s0 is not None and (not stop_sig or stop_sig.startswith(("chain", "exc", "nopanel"))):
            try:
                self._u0 = u0
                ex.view.setFocus()                       # a value still being typed is validated first (the field
                ex.settle(4)                             # loses the cursor, as when the user clicks the graph)
                self.wait_ready()
                self._chains(s0, n_u0, len(seq["actions"]))
            except Exception:  # noqa: BLE001
                self.fail("harness", "chains", traceback.format_exc()[-2000:])
            try:
                if getattr(self, "panels", True):        # a sweep of single gestures turns it off (slow)
                    self.check_panels()
            except Exception:  # noqa: BLE001
                self.fail("harness", "panels", traceback.format_exc()[-2000:])
            take(len(seq["actions"]))
        faulthandler.cancel_dump_traceback_later()
        res["time"] = round(time.time() - t0, 1)
        return res


# ---------------------------------------------------------------------------------------------- modes
def cmd_run(seed: int):
    seq = make_sequence(seed)
    fz = Fuzz2(seq["start"]["lang"])
    prog = RUNS / f"seed_{seed}.progress.json"
    hang = open(RUNS / f"seed_{seed}.hang.txt", "w", encoding="utf-8")
    res = fz.run(seq, f"t{seed}", progress=prog, hangfile=hang)
    (RUNS / f"seed_{seed}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    hang.close()
    print("SEED", seed, "steps", res["done"], "fails", len(res["failures"]), "sigs",
          sorted({f["sig"] for f in res["failures"]}), "t", res.get("time"), flush=True)
    os._exit(0)


def cmd_drive(a: int, b: int, par: int, timeout: float = 900.0):
    todo = [s for s in range(a, b) if not (RUNS / f"seed_{s}.json").exists()]
    running = []
    log = open(OUT / "drive3.log", "a", encoding="utf-8")
    while todo or running:
        while todo and len(running) < par:
            s = todo.pop(0)
            p = subprocess.Popen([PY, str(Path(__file__)), "run", str(s)], cwd=str(HERE.parents[1]), stdout=log,
                                 stderr=subprocess.STDOUT)
            running.append((s, p, time.time()))
        time.sleep(2)
        for s, p, t in list(running):
            rc = p.poll()
            if rc is None and time.time() - t < timeout:
                continue
            if rc is None:
                p.kill()
                rc = "timeout"
            running.remove((s, p, t))
            if not (RUNS / f"seed_{s}.json").exists():
                prog = RUNS / f"seed_{s}.progress.json"
                info = json.loads(prog.read_text(encoding="utf-8")) if prog.exists() else {}
                hang = RUNS / f"seed_{s}.hang.txt"
                ht = hang.read_text(encoding="utf-8", errors="replace") if hang.exists() else ""
                seq = make_sequence(s)
                res = {"seed": s, "start": seq["start"], "actions": seq["actions"], "steps": info.get("steps", []),
                       "done": info.get("step", 0), "failures": [{
                           "step": info.get("step", 0), "kind": "hang",
                           "sig": f"hang:{(info.get('action') or ['?'])[0]}@{F._where(ht) if ht else rc}",
                           "detail": f"le processus ne répond plus (code {rc})\n" + ht[-2500:]}]}
                (RUNS / f"seed_{s}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print("done", s, rc, round(time.time() - t), flush=True)
    log.close()


def load_runs():
    out = []
    for f in sorted(RUNS.glob("seed_*.json")):
        if f.name.endswith(".progress.json"):
            continue
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except ValueError:
            pass
    return out


def cmd_summary():
    runs = load_runs()
    by: dict[str, list] = {}
    n_act = 0
    for r in runs:
        n_act += r.get("done", 0)
        for f in r["failures"]:
            by.setdefault(f["sig"], []).append((r["seed"], f["step"]))
    print(f"{len(runs)} sequences, {n_act} actions")
    for sig, hits in sorted(by.items(), key=lambda kv: -len({h[0] for h in kv[1]})):
        print(f"{len({h[0] for h in hits}):4d} seq  {sig}   e.g. seed {hits[0][0]} step {hits[0][1]}")


def replay(fz, start, actions, sig, tag):
    return fz.run({"seed": None, "start": start, "actions": actions}, tag, stop_sig=sig, watchdog=60)


def cmd_minimize(sig: str, max_trials: int = 40):
    runs = load_runs()
    cands = []
    for r in runs:
        st = [f["step"] for f in r["failures"] if f["sig"] == sig]
        if st:
            cands.append((min(st) + 1, r))
    if not cands:
        print("no run with", sig)
        os._exit(0)
    cands.sort(key=lambda x: x[0])
    n, r = cands[0]
    acts = r["actions"][:n]
    fz = Fuzz2(r["start"]["lang"])
    k = [0]

    def fails(lst) -> bool:
        k[0] += 1
        res = replay(fz, r["start"], lst, sig, f"m{k[0]}_{abs(hash(sig)) % 10000}")
        return any(f["sig"] == sig for f in res["failures"])
    if not fails(acts):
        print("NOT REPRODUCED", sig, "seed", r["seed"], flush=True)
        best = None
    else:
        best = acts
        k_ = 1
        while k_ < len(best):
            if fails(best[-k_:]):
                best = best[-k_:]
                break
            k_ *= 2
        gran = 2
        while len(best) >= 2 and k[0] < max_trials:
            chunk = max(1, len(best) // gran)
            reduced = False
            for i in range(0, len(best), chunk):
                cand = best[:i] + best[i + chunk:]
                if cand and fails(cand):
                    best = cand
                    gran = max(gran - 1, 2)
                    reduced = True
                    break
                if k[0] >= max_trials:
                    break
            if not reduced:
                if chunk == 1:
                    break
                gran = min(len(best), gran * 2)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", sig)[:80]
    CASES.mkdir(exist_ok=True)
    case = {"sig": sig, "seed": r["seed"], "start": r["start"], "actions": best, "trials": k[0],
            "n_seq": len({c[1]["seed"] for c in cands}), "orig_len": n}
    (CASES / f"{slug}.json").write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8")
    print("MIN", sig, "from", n, "to", None if best is None else len(best), "trials", k[0], flush=True)
    os._exit(0)


def cmd_confirm(case_file: str):
    case = json.loads(Path(case_file).read_text(encoding="utf-8"))
    fz = Fuzz2(case["start"]["lang"])
    outs = []
    for k in range(2):
        res = replay(fz, case["start"], case["actions"] or [], case["sig"], f"c{k}_{abs(hash(case['sig'])) % 10000}")
        hit = [f for f in res["failures"] if f["sig"] == case["sig"]]
        outs.append({"ok": bool(hit), "steps": res["steps"], "detail": hit[0]["detail"] if hit else ""})
    case["confirm"] = outs
    Path(case_file).write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8")
    print("CONFIRM", case["sig"], [o["ok"] for o in outs], flush=True)
    for o in outs:
        print("  steps:", o["steps"], "\n  detail:", o["detail"][:1500], flush=True)
    if case.get("title") and all(o["ok"] for o in outs):
        start = case["start"]
        st = ("projet de départ: " + (f"type « {start['kind']} »" if "kind" in start else
                                       f"nouveau projet, modèle « {start['template']} »") + f", langue {start['lang']}")
        runs = load_runs()
        seeds = sorted({r["seed"] for r in runs for f in r["failures"] if f["sig"] in case.get("sigs", [case["sig"]])})
        detail = case["detail"] + f"\n\nReproduit 2/2 fois sur la séquence minimale. Vu dans {len(seeds)} " \
            f"séquence(s) sur {len(runs)}." + ("\n\nCe que le contrôle a vu: " + outs[-1]["detail"][:700]
                                                if case.get("show_detail", True) else "")
        fz.ex.note(case["severity"], case["title"], detail, shot=True, steps=[st] + outs[-1]["steps"])
    os._exit(0)


if __name__ == "__main__":
    if MODE == "run":
        cmd_run(int(sys.argv[2]))
    elif MODE == "drive":
        cmd_drive(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    elif MODE == "summary":
        cmd_summary()
    elif MODE == "show":
        F.cmd_show(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 400)
    elif MODE == "minimize":
        cmd_minimize(sys.argv[2])
    elif MODE == "confirm":
        cmd_confirm(sys.argv[2])
