"""Helpers of the palette_matrix exploration, round 3 (on top of harness.py, never editing it)."""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

from harness import Explorer, TMP, _slug  # noqa: F401  (TMP: the throw-away folder)

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QApplication, QLabel

from astro_gpu.client import i18n

SCRATCH = Path(os.environ.get("PM_RAW", "F:/astro_essais/A_REGARDER/exploration3/palette_matrix/raw"))
LANG = os.environ.get("PM_LANG", "fr")
SCRATCH.mkdir(parents=True, exist_ok=True)

# English words that should not remain in a French text ("workflow" is allowed)
ENGLISH = re.compile(r"\b(the|needs?|first|only|takes?|from|with|and|this|its|it|has|there|before|after|cannot|"
                     r"can't|here|drop|place|placed|added|deleted|shared|computed|once|undoes|branch|flow|new|"
                     r"blocks?|frames|night|layer|stars|background|stacking|registration|color|"
                     r"denoise|astrometry|selection|first|should|must|already)\b", re.I)
# words a French user would not understand: engine keys, internal names, raw errors
INTERNAL = re.compile(r"(legacy|\bv[12]\b|render_|star_removal|~\d|\betoiles\b|\bdebruitage\b|\bastrometrie\b|"
                      r"\balignement~|\bempilement\b(?=~)|Traceback|Error\b|KeyError|ValueError|\bNone\b|\{[a-z_]*\}|"
                      r"@\d|\bpid\b|\bsid\b|sel_step)")


SIZE = tuple(int(x) for x in os.environ.get("PM_SIZE", "1600x950").split("x"))
FRENCH = re.compile(r"[éèêàçùôî]|\b(le|la|les|des|du|pour|avec|dans|requiert|poser|depuis|pas de|ici|avant)\b", re.I)
ELISION = re.compile(r"\b(de|le|la|que) [AEIOUÉaeiouéÉ]")


def englishness(text: str) -> list[str]:
    """What is wrong with an English text: French words left, internal words."""
    out = []
    fr = sorted({m.group(0).lower() for m in FRENCH.finditer(re.sub(r"«[^»]*»|“[^”]*”|\"[^\"]*\"", "", text))})
    if fr:
        out.append("mots français: " + ", ".join(fr))
    bad = sorted({m.group(0) for m in INTERNAL.finditer(text)})
    if bad:
        out.append("mots internes: " + ", ".join(bad))
    return out


def wording(text: str) -> list[str]:
    return frenchness(text) if LANG == "fr" else englishness(text)


def frenchness(text: str) -> list[str]:
    """What is wrong with a French text shown to the user (English words left, internal words)."""
    out = []
    el = sorted({m.group(0) for m in ELISION.finditer(text)})
    if el:
        out.append("élision manquante: " + ", ".join(el))
    t = re.sub(r"«[^»]*»|“[^”]*”|\"[^\"]*\"", "", text)          # names typed by the user, quoted
    en = sorted({m.group(0).lower() for m in ENGLISH.finditer(t)})
    # French homographs: "selection" never (French has an accent), "image", "Session", "Fusion", "Post" fine
    en = [w for w in en if w not in ()]
    if en:
        out.append("mots anglais: " + ", ".join(en))
    bad = sorted({m.group(0) for m in INTERNAL.finditer(text)})
    if bad:
        out.append("mots internes: " + ", ".join(bad))
    return out


class Matrix:
    """One process: a project state, all its selections × cards × ways to place; raw findings to a jsonl."""

    def __init__(self, tag: str):
        self.tag = tag
        self.ex = Explorer("palette_matrix", lang=LANG, size=SIZE)
        self.raw = SCRATCH / f"{tag}.jsonl"
        self.raw.write_text("", encoding="utf-8")
        self.shots = SCRATCH / "shots"
        self.shots.mkdir(exist_ok=True)
        self.seen: set[str] = set()
        self.counts: dict[str, int] = {}
        self.toasts: list[str] = []
        self.exc: list[str] = []
        self._pages: set[int] = set()
        self.case = ""
        sys.excepthook = self._hook
        # offscreen, the window is not active by itself: its shortcuts (Ctrl+Z, Ctrl+Y) would never fire (a harness
        # artefact): activate it before each key, as a user's window is
        ex = self.ex
        _key = ex.key

        def key(k, widget=None, mod=Qt.NoModifier):
            if ex.w is not None:
                ex.w.activateWindow()
                ex.settle(1)
            _key(k, widget, mod)
        ex.key = key

    # ------------------------------------------------------------ recording
    def _hook(self, t, v, tb):
        self.exc.append(f"{t.__name__}: {v}\n" + "".join(traceback.format_tb(tb))[-2500:])
        print("EXC", self.case, t.__name__, v, flush=True)

    def hook_page(self):
        pg = self.ex.page
        if pg is not None and id(pg) not in self._pages:
            self._pages.add(id(pg))
            pg.toasted.connect(lambda t: self.toasts.append(t))

    def find(self, sig: str, detail: str, shot: bool = True, **ctx):
        """A raw finding (first occurrence per signature in this process gets a screenshot)."""
        rec = {"tag": self.tag, "sig": sig, "detail": detail, "case": self.case, **ctx, "t": time.strftime("%H:%M:%S")}
        if sig not in self.seen and shot:
            f = self.shots / f"{self.tag}_{len(self.seen):03d}_{_slug(sig)[:40]}.png"
            try:
                self.ex.w.grab().save(str(f))
                rec["shot"] = str(f)
            except Exception:  # noqa: BLE001
                pass
        self.seen.add(sig)
        with self.raw.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("FIND", sig, "|", self.case, flush=True)

    def count(self, k: str):
        self.counts[k] = self.counts.get(k, 0) + 1

    # ------------------------------------------------------------ state
    def state(self):
        s = self.ex.s
        paths = tuple((q.pid, q.name, json.dumps(q.source, sort_keys=True, default=str), tuple(s.sessions_of(q)),
                       json.dumps(s.graph(q), sort_keys=True)) for q in s.paths)
        sess = tuple((x.id, x.name, x.filter) for x in s.p.sessions)
        wfs = tuple((w.id, w.name, json.dumps(w.blocks, sort_keys=True, default=str),
                     json.dumps(w.source, sort_keys=True, default=str), json.dumps(w.sessions, default=str))
                    for w in s.p.workflows)
        layers = json.dumps([(x.get("kind"), x.get("pid"), x.get("part")) for x in s.post_layers()], default=str)
        return {"paths": paths, "sessions": sess, "workflows": wfs, "post": layers}

    @staticmethod
    def diff(a, b) -> str:
        out = []
        for k in a:
            if a[k] != b[k]:
                out.append(f"{k}: {a[k]!r}\n  -> {b[k]!r}")
        return "\n".join(out)[:3000]

    # ------------------------------------------------------------ what is shown
    def heading(self) -> str:
        labs = [x.text() for x in self.ex.page.palette_bar.widget().findChildren(QLabel)]
        return labs[1] if len(labs) > 1 else ""

    def reasons(self) -> dict:
        return {b: c.reason for b, c in self.ex.page.cards.items()}

    def items(self):
        return self.ex.blocks()

    def overlaps(self, extra=()):
        its = [(it.node.key + (f"@{it.node.sid}" if it.node.sid else "") + f"[{','.join(str(q.pid) for q in it.node.paths)}]",
                it.sceneBoundingRect()) for it in self.items()]
        its += list(extra)
        # the static dashed places too: the bottom new-flow zone, "+ Session" (the dragged ones come in ``extra``)
        from astro_gpu.client.graph import GhostItem
        dyn = {id(g) for g in self.ex.ghosts()}
        for g in self.ex.view.scene().items():
            if isinstance(g, GhostItem) and id(g) not in dyn:
                its.append((f"place:{g.text.strip()[:30]}", g.sceneBoundingRect().adjusted(2, 2, -2, -2)))
        out = []
        for i in range(len(its)):
            for j in range(i + 1, len(its)):
                r = its[i][1].intersected(its[j][1])
                if r.width() > 4 and r.height() > 4:
                    out.append(f"{its[i][0]} / {its[j][0]} ({r.width():.0f}x{r.height():.0f})")
        return out

    def hint_of(self, block: str):
        """Hover a palette card (enter event): the small message shown, or None."""
        c = self.ex.card(block)
        pg = self.ex.page
        p = QPointF(c.width() / 2, c.height() / 2)
        QApplication.sendEvent(c, QEnterEvent(p, c.mapToGlobal(p), c.mapToGlobal(p)))
        self.ex.settle(2)
        txt = pg.hint.text() if pg.hint.isVisible() else None
        geo = pg.hint.geometry() if pg.hint.isVisible() else None
        QApplication.sendEvent(c, QEvent(QEvent.Leave))
        self.ex.settle(1)
        return txt, geo

    def empty_point(self) -> QPoint | None:
        """A point of the graph's viewport with nothing drawn under it (computed while the drag is held)."""
        v = self.ex.view
        vp = v.viewport()
        best = None
        for fy in (0.12, 0.88, 0.5, 0.3, 0.7):
            for fx in (0.9, 0.1, 0.5, 0.7, 0.3):
                p = QPoint(int(vp.width() * fx), int(vp.height() * fy))
                if not v.items(p):
                    sp = v.mapToScene(p)
                    if not any(g.rect.adjusted(-30, -30, 30, 30).contains(sp) for g in self.ex.ghosts()):
                        return p
                    best = best or p
        return best

    def drop_on_palette(self, block: str):
        c = self.ex.card(block)
        if c is None:
            return None
        bar = self.ex.page.palette_bar.viewport()
        n = len(self.ex.drops)
        self.ex.drag_mouse(c, QPoint(20, 12), QPoint(60, 14), bar, QPoint(bar.width() // 2, bar.height() - 20))
        return self.ex.drops[-1] if len(self.ex.drops) > n else None

    def new_project(self, name: str, template: str | None):
        """The real "new project" path. harness.new_project looks for the template window after it was already
        closed by the popup timer (which rejects any modal dialog): here the timer's handler picks the template."""
        ex = self.ex
        ex.open()
        got = {}

        def h(gal):
            got["cls"] = gal.__class__.__name__
            if gal.__class__.__name__ != "TemplateGallery":
                gal.reject()
                return
            if template:
                from astro_gpu.client.gallery import GALLERY
                tab = next((x["tab"] for x in GALLERY if x["key"] == template), None)
                keys = getattr(gal, "tab_keys", [])
                if tab in keys:
                    gal.tabs.setCurrentIndex(keys.index(tab))
                    ex.settle(2)
            t = next((c.t for c in gal.cards if (c.t.get("key") == template if template else c.t.get("blank"))), None)
            got["found"] = t is not None
            if t is not None:
                gal.pick(t)
                if t.get("source") == "image" and hasattr(gal, "file"):
                    f = ex.image_file("pic.png" if t.get("stretched") else "stack.fits", bool(t.get("stretched")))
                    gal.file.setText(str(f))
                    ex.settle(1)
            gal.ok.click()
        ex.dialog_handler = h
        root = ex.ws / name
        ex.w.create_project(str(root), name, "M 31")
        ex.settle(15)
        ex.dialog_handler = None
        return root, got

    def watchdog(self, sec=240):
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(sec, exit=True)
