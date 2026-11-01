"""The common image viewer (astro_gpu/client/imageview.py) on a test project, outside the client.

Three tabs: the stack (6384 x 4272 RGB float, with its astrometric solution, compare targets: the denoised image and
the two half-stacks), the frames of the project (camera raws, with the numbers the engine measured), and a compare
pair (stack | denoised, split). Nothing is written but the screenshots (``--shots``) and what Export… is asked to.

    python tools/imageview_demo.py [--project F:/astro_essais/v2/heart_a7s_p10] [--lang fr]
    python tools/imageview_demo.py --shots %TEMP%/a3_shots          # EN and FR screenshots, then quit

Never point it at F:/astro_projets: the test copies live in F:/astro_essais.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PySide6.QtCore import QLocale, QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.imageview import ImageView, ViewImage, default_loader, quick_thumbnail  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402

DEFAULT = Path("F:/astro_essais/v2/heart_a7s_p10")
EXTRA_QSS = "QSlider:horizontal { margin-left: 6px; margin-right: 6px; }"      # as the client's window (app.py)


def set_language(code: str):
    i18n.set_language(code)
    QLocale.setDefault(QLocale(QLocale.French, QLocale.France) if code == "fr"
                       else QLocale(QLocale.English, QLocale.UnitedStates))


def one(folder: Path, pattern: str) -> Path | None:
    hits = sorted(folder.glob(pattern))
    return hits[0] if hits else None


def project_images(root: Path) -> dict:
    """The arrays, the solution and the frames of a v2 test project (read where they are, never written)."""
    if "astro_projets" in root.as_posix().lower():
        raise SystemExit("refusing F:/astro_projets: use a test copy under F:/astro_essais")
    store = root / "store"
    out = {"stack": one(store, "stack/*/drz_full.npy"), "denoise": one(store, "denoise/*/n2n_full.npy"),
           "half_a": one(store, "stack/*/drz_A.npy"), "half_b": one(store, "stack/*/drz_B.npy"),
           "wcs": one(store, "astrometry/*/wcs.fits"), "frames": []}
    fj = one(store, "stack/*/frames.json")
    if fj is not None:
        out["frames"] = json.loads(fj.read_text(encoding="utf-8"))
    return out


def frame_images(frames: list[dict]) -> list[ViewImage]:
    """The frames as the viewer wants them: the path, and the numbers the engine measured (the caller's info)."""
    from astro_gpu.io.frames import read_info
    paths = [f["path"] for f in frames]
    with ThreadPoolExecutor(8) as ex:
        infos = list(ex.map(lambda p: read_info(p) if Path(p).is_file() else None, paths))
    tops = sorted(float(f.get("transparency") or 0) for f in frames)
    p90 = tops[int(0.9 * (len(tops) - 1))] if tops else 1.0
    out = []
    for f, inf in zip(frames, infos):
        if inf is None:
            continue
        info = {"time": f.get("time") or inf.time, "exposure": inf.exposure, "iso": inf.iso,
                "temperature": inf.temperature, "fwhm": f.get("fwhm"),
                "transparency": (f.get("transparency") or 0) / (p90 or 1.0), "weight": f.get("weight"),
                "excluded": bool(f.get("excluded")), "reason": f.get("excluded") or ""}
        vi = ViewImage(f["path"], Path(f["path"]).name, info=info, camera=True, white=inf.white, cfa=inf.cfa)
        vi.thumb = partial(quick_thumbnail, ViewImage(f["path"], cfa=inf.cfa, camera=True))
        out.append(vi)
    return out


class Demo(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.setWindowTitle("astro_gpu · " + i18n._("Image viewer"))
        imgs = project_images(root)
        self.imgs = imgs
        tabs = QTabWidget()
        tabs.setObjectName("faces")
        self.tabs = tabs
        self.setCentralWidget(tabs)
        stack = ViewImage(str(imgs["stack"]), i18n._("Stack"), key=("demo", "stack"), wcs=imgs["wcs"], camera=True,
                          header={"HISTORY": ["astro_gpu demo: the stack of heart_a7s"]})
        self.stack_img = stack
        targets = []
        for k, name in (("denoise", "Denoised"), ("half_a", "Half-stack A"), ("half_b", "Half-stack B")):
            if imgs[k] is not None:
                targets.append(ViewImage(str(imgs[k]), i18n._(name), key=("demo", k), wcs=imgs["wcs"], camera=True))
        self.v_stack = ImageView(post=True)
        self.v_stack.set_image(stack)
        self.v_stack.set_compare_targets(targets)
        self.v_stack.sendToPost.connect(lambda key: self.statusBar().showMessage(f"Send to Post: {key}", 4000))
        self.v_stack.toasted.connect(lambda t: self.statusBar().showMessage(t, 4000))
        tabs.addTab(self.v_stack, i18n._("Stack"))
        self.v_seq = ImageView(histogram=False)
        self.frames = frame_images(imgs["frames"])
        self.v_seq.set_sequence(self.frames, 0)
        self.v_seq.excludeToggled.connect(lambda i, ex: self.statusBar().showMessage(
            f"{self.frames[i].name}: {'excluded' if ex else 'included'}", 4000))
        tabs.addTab(self.v_seq, i18n._("Frames"))
        self.v_cmp = ImageView()
        self.v_cmp.set_image(ViewImage(str(imgs["denoise"]), i18n._("Denoised"), key=("demo", "denoise"),
                                       wcs=imgs["wcs"], camera=True) if imgs["denoise"] else stack)
        self.v_cmp.set_compare_targets([stack], current=0)
        self.v_cmp.set_same_stretch(True)
        tabs.addTab(self.v_cmp, i18n._("Compare"))


def settle(win: Demo, seconds: float = 60.0):
    end = time.monotonic() + seconds
    for v in (win.v_stack, win.v_seq, win.v_cmp):
        v.wait(max(1.0, end - time.monotonic()))
    default_loader().wait(max(1.0, end - time.monotonic()))
    QApplication.processEvents()


def shots(root: Path, folder: Path):
    """The screenshots of the viewer in English and French: stack with the histogram, the compare split, the
    sequence with its info band, the export dialog."""
    folder.mkdir(parents=True, exist_ok=True)
    made = []
    for lang in ("en", "fr"):
        set_language(lang)
        win = Demo(root)
        win.resize(1680, 1000)
        win.show()
        settle(win)
        v = win.v_stack
        win.tabs.setCurrentIndex(0)
        settle(win)
        v.canvas.set_zoom_percent(50, QPointF(v.canvas.width() * 0.47, v.canvas.height() * 0.42))
        v.canvas._mouse = None
        QApplication.processEvents()
        _hover(v, 0.46, 0.40)
        made.append(_save(win, folder / f"stack_histogram_{lang}.png"))
        win.tabs.setCurrentIndex(2)
        c = win.v_cmp
        settle(win)
        c.canvas.set_zoom_percent(100, QPointF(c.canvas.width() * 0.5, c.canvas.height() * 0.45))
        c.canvas.split = 0.5
        QApplication.processEvents()
        _hover(c, 0.62, 0.45)
        made.append(_save(win, folder / f"compare_split_{lang}.png"))
        win.tabs.setCurrentIndex(1)
        s = win.v_seq
        s.set_index(3)
        settle(win)
        s.loader.wait(30)
        settle(win)
        _hover(s, 0.5, 0.5)
        made.append(_save(win, folder / f"sequence_info_{lang}.png"))
        win.tabs.setCurrentIndex(0)
        QApplication.processEvents()
        dlg = v.export_dialog()
        dlg.dest.setText((folder / "export_demo" / "Stack.fits").as_posix())
        QApplication.processEvents()
        dlg.adjustSize()
        made.append(_save(dlg, folder / f"export_dialog_{lang}.png"))
        dlg.reject()
        win.close()
        win.deleteLater()
        QApplication.processEvents()
    set_language("en")
    return made


def _hover(v: ImageView, fx: float, fy: float):
    c = v.canvas
    x, y = c.w_to_img(c.width() * fx, c.height() * fy)
    v._hover(c, x, y, 0)


def _save(w, path: Path) -> Path:
    QApplication.processEvents()
    w.grab().save(str(path))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=str(DEFAULT))
    ap.add_argument("--lang", default="en", choices=["en", "fr"])
    ap.add_argument("--shots", default="", help="write the screenshots there and quit")
    a = ap.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyleSheet(QSS + EXTRA_QSS)
    root = Path(a.project)
    if a.shots:
        for p in shots(root, Path(a.shots)):
            print(p)
        return 0
    set_language(a.lang)
    win = Demo(root)
    win.resize(1680, 1000)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
