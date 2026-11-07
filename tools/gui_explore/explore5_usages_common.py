"""Round 4, axis "usages": complete uses start to finish, then persistence (Save, close the window, open the project in a
new window, compare). Shared helpers: the project made from Home (New project, then the template gallery), frames added
in a Session's panel, a snapshot of everything the user sees, Save / close window / reopen."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore3_sources_common import (  # noqa: E402,F401
    EXC, TMP, Dispatch, Explorer, _, button, buttons, menu_button, panel, ptexts, start, undo, visible, wait,
)
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QLineEdit, QMessageBox, QPushButton  # noqa: E402

WORK = Path("F:/astro_essais/_explore4_work/usages")
WORK.mkdir(parents=True, exist_ok=True)
LOG: list[str] = []


def log(*a):
    t = " ".join(str(x) for x in a)
    LOG.append(t)
    print(t, flush=True)


class U:
    def __init__(self, lang="fr"):
        self.ex = Explorer("usages", lang=lang)
        self.seen = {n["title"] for n in self.ex.notes()}
        self.toasts: list[str] = []
        self.box_answer: list[str] = []          # the button to click in the next QMessageBox.exec (its text)
        self.boxes: list[dict] = []
        ex = self.ex
        u = self

        def box_exec(mb):
            rec = {"title": mb.windowTitle(), "text": mb.text(), "info": mb.informativeText(),
                   "buttons": [b.text() for b in mb.buttons()]}
            u.boxes.append(rec)
            ex.popups.append(("box", mb.windowTitle(), mb.text()))
            want = u.box_answer.pop(0) if u.box_answer else None
            if want is not None:
                b = next((b for b in mb.buttons() if b.text().replace("&", "") == want), None)
                if b is not None:
                    b.click()
            return QMessageBox.Ok
        QMessageBox.exec = box_exec

    # ------------------------------------------------------------ notes (once per title)
    def note(self, sev, title, detail="", steps=None, shot=True):
        if title in self.seen:
            log("(already noted)", title)
            return
        self.seen.add(title)
        self.ex.note(sev, title, detail, shot=shot, steps=steps)

    # ------------------------------------------------------------ the window
    @property
    def s(self):
        return self.ex.w.s

    def hook_toasts(self):
        if self.ex.w is not None and self.ex.w.graph is not None:
            try:
                self.ex.page.toasted.connect(lambda t: self.toasts.append(t))
            except Exception:  # noqa: BLE001
                pass

    def new_from_home(self, name, template, image="", chips=()):
        """Home > New project: the name typed, Create; the gallery: the template's tab, its card, its file/filters,
        Create."""
        ex = self.ex
        ex.open()
        cap = {}

        def npd(d):
            d.name.setText(name)
            d.target.setText("M 31")
            ex.settle(2)
            cap["folder"] = d.folder().as_posix()
            d.ok.click()

        def gal(d):
            if template is None:
                d.reject()
                return
            t = next(x for x in d.templates if x.get("key") == template)
            if t.get("tab") in d.tab_keys:
                d.tabs.setCurrentIndex(d.tab_keys.index(t["tab"]))
                ex.settle(2)
            card = next((c for c in d.cards if c.t.get("key") == template), None)
            if card is None:
                cap["no_card"] = True
                d.reject()
                return
            ex.click(card)
            for c in chips:
                b = next((x for x in d.findChildren(QPushButton) if x.isVisible() and x.text() == c), None)
                if b is not None:
                    ex.click(b)
            if image:
                d.file.setText(image)
            ex.settle(3)
            cap["ok"] = d.ok.isEnabled()
            cap["ok_tip"] = d.ok.toolTip()
            if d.ok.isEnabled():
                d.ok.click()
            else:
                d.reject()
        disp = Dispatch(ex, NewProjectDialog=npd, TemplateGallery=gal)
        ex.w.new_project()
        wait(ex, 1.5)
        disp.stop()
        log("new_from_home", name, template, cap, [x[:2] for x in disp.seen])
        self.hook_toasts()
        return Path(cap.get("folder", "")), cap

    def blk(self, key, pid=None, sid=None):
        return self.ex.block(key, pid=pid, sid=sid)

    def title_blk(self, title):
        return next((it for it in self.ex.blocks() if it.title == title), None)

    def add_frames(self, sid, folder, kind=0):
        """Frames added in the Session's panel: Add ▾ (lights 0, darks 1, flats 2) > Add folder…"""
        ex = self.ex
        it = self.blk("session", sid=sid) or (self.blk("sources") if sid in (None, 1) else None)
        if it is None:
            log("add_frames: no session block", sid)
            return False
        ex.click_block(it)
        wait(ex, 0.3)
        bs = buttons(panel(ex), _("Add") + " ▾", QPushButton)
        if len(bs) <= kind:
            log("add_frames: no Add button", sid, kind, len(bs))
            return False
        ex.file_queue = [str(folder)]
        menu_button(ex, bs[kind], _("Add folder…"))
        wait(ex, 0.8)
        return True

    def place(self, *keys):
        for k in keys:
            self.ex.double_click_card(k)
            wait(self.ex, 0.5)

    def branch(self, block, under_key, pid=None):
        """A palette block dropped under a block: a branch."""
        ex = self.ex
        ex.drag_palette_hover(block, QPoint(5, 5))
        gh = [g for g in ex.ghosts() if getattr(g, "branch_key", None) == under_key
              and (pid is None or g.pid == pid)]
        pt = ex.ghost_center(gh[0]) if gh else None
        ex.drag_leave()
        if pt is None:
            log("branch: no place under", under_key, [(getattr(g, "branch_key", None), g.pid) for g in ex.ghosts()])
            return False
        ex.drag_palette(block, pt)
        wait(ex, 0.8)
        return True

    def to_post(self, it):
        items = self.ex.right_click_block(it, _("To Post"))
        wait(self.ex, 0.6)
        return items

    def change_setting(self, it, nth=0):
        """A block clicked, the nth combo of its panel's settings set to another choice: what changed."""
        ex = self.ex
        ex.click_block(it)
        wait(ex, 0.4)
        from astro_gpu.client.panel import CompactRow
        from PySide6.QtWidgets import QAbstractSpinBox, QCheckBox
        rows = [r for r in panel(ex).findChildren(CompactRow) if r.isVisible() and r.editor is not None
                and r.editor.isEnabled() and (not isinstance(r.editor, QComboBox) or r.editor.count() > 1)
                and isinstance(r.editor, (QComboBox, QAbstractSpinBox, QCheckBox))]
        if len(rows) <= nth:
            rows2 = [r for r in panel(ex).findChildren(CompactRow) if r.isVisible()]
            log("change_setting: no editor", it.title, [r.r["label"] for r in rows2])
            return None
        r = rows[nth]
        e = r.editor
        lab = r.r["label"]
        if isinstance(e, QComboBox):
            old = e.currentText()
            e.setCurrentIndex((e.currentIndex() + 1) % e.count())
        elif isinstance(e, QCheckBox):
            old = e.isChecked()
            ex.click(e)
        else:
            old = e.text()
            e.setFocus()
            e.stepBy(1)
            e.editingFinished.emit()
        wait(ex, 0.6)
        # a change that throws away results asks Replace / Try as variant: Replace
        rep = button(panel(ex), _("Replace"), QPushButton)
        if rep is not None:
            ex.click(rep)
            wait(ex, 0.6)
        log("setting", it.title, lab, old)
        return (lab, old)

    # ------------------------------------------------------------ what the user sees
    def snapshot(self) -> dict:
        ex = self.ex
        s = self.s
        w = ex.w
        from astro_gpu.client.graph import WireItem
        d = s.p.to_dict()
        d.pop("created", None)
        blocks = sorted((it.node.key, it.title, it.summary, it.node.sid,
                         tuple(sorted(q.pid for q in it.node.paths))) for it in ex.blocks())
        pos = {f"{it.title}@{sorted(q.pid for q in it.node.paths)}@{it.node.sid}":
               (round(it.sceneBoundingRect().x()), round(it.sceneBoundingRect().y())) for it in ex.blocks()}
        wires = sorted((x.pid if x.pid is not None else -1, x.toolTip().split("\n")[0], str(x.part))
                       for x in ex.view.scene().items() if isinstance(x, WireItem))
        title = next((x.text() for x in visible(panel(ex), QLabel) if x.text()), "")
        ed = next((x.text() for x in visible(panel(ex), QLineEdit)), "")
        return {
            "definition": json.loads(json.dumps(d, default=str)),
            "flows": [(q.pid, q.name) for q in s.paths],
            "blocks": [list(b) for b in blocks],
            "pos": pos,
            "wires": [list(x) for x in wires],
            "selection": {"flow": s.path.name if s.path else None, "block": s.sel_step, "panel": ed or title},
            "show_all": ex.view.b_all.isChecked() if hasattr(ex.view, "b_all") else None,
            "undo": (len(s._undo), len(s._redo)),
            "dirty": s.dirty,
            "title": w.windowTitle(),
            "status": w.status.text(),
            "run": (w.bt_run.text(), w.bt_run.isEnabled(), w.bt_run.toolTip()),
            "mode": w.mode,
        }

    def save(self):
        ex = self.ex
        ex.key(Qt.Key_S, ex.w, Qt.ControlModifier)
        wait(ex, 0.6)

    def close_window(self):
        """The window closed (its X): what it asked."""
        ex = self.ex
        n = len(self.boxes)
        ex.w.close()
        ex.settle(6)
        asked = self.boxes[n:]
        still = ex.w.isVisible()
        if not still:
            ex.w = None
        return asked, still

    def reopen_new_window(self, root):
        """A new window (the app started again), the project opened from it."""
        ex = self.ex
        if ex.w is not None:
            ex.w.close()
            ex.settle(4)
            ex.w = None
        ex.open()
        ex.w.open_project(root)
        wait(ex, 1.5)
        self.hook_toasts()

    def roundtrip(self, root, label, steps, keep_selection=True, extra=None):
        """Save, close the window, open the project in a new window, compare."""
        before = self.snapshot()
        self.save()
        saved = self.snapshot()
        asked, still = self.close_window()
        if asked:
            self.note("gênant", f"{label} : fermer la fenêtre après Ctrl+S demande encore d'enregistrer",
                      f"Boîtes : {asked}", steps=steps + ["Ctrl+S", "fermer la fenêtre"])
        self.reopen_new_window(root)
        after = self.snapshot()
        diffs = compare(saved, after)
        log("ROUNDTRIP", label, "before-save dirty", before["dirty"], "saved dirty", saved["dirty"])
        log("  diffs:", json.dumps(diffs, ensure_ascii=False, default=str)[:3000])
        return before, saved, after, diffs


def compare(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        if k in ("undo",):
            continue
        if k == "definition":
            dd = diff_dict(a[k], b[k])
            if dd:
                out[k] = dd
        elif a[k] != b[k]:
            out[k] = {"before": a[k], "after": b[k]}
    return out


def diff_dict(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff_dict(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_dict(x, y, f"{path}[{i}]")
    elif a != b:
        out.append((path, a, b))
    return out
