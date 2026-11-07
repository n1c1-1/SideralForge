"""post_images 4: Export… as the user does it (the dialog's defaults: TIFF 16), the file read back and compared with
what the screen shows (size, bit depth, colours, seams at regular intervals); then JPEG and PNG 16 the same way."""
import sys
import time
from pathlib import Path

import numpy as np
from explore5_post_images_common import COPY, PI, WORK, R, Explorer, dump_json, start, stats

start(3000)
ex = Explorer("post_images", lang="fr")
pi = PI(ex)
pi.open(COPY)
v = pi.v
pi.idle(300)
fmts = sys.argv[1:] or ["tiff16", "jpeg", "png16"]
OUT = WORK / "exports"
OUT.mkdir(exist_ok=True)
report = {}
v.viewer.fit()
pi.idle(120)
screen = pi.screen_array()                       # the half image the screen is fed (8-bit)
R("screen", None if screen is None else screen.shape, None if screen is None else stats(screen), "canvas",
  v.src.canvas, "zoom", v.zoom_lab.text())


def run_dialog(fmt_key: str, name: str):
    info = {}

    def handle(d):
        info["defaults"] = {"format": d.fmt.currentText(), "format_key": d.format_key(), "profile": d.prof.currentText(),
                            "size": d.size_cb.currentText(), "dest": d.dest.text(), "wcs": (d.c_wcs.isChecked(),
                            d.c_wcs.isEnabled()), "history": d.c_hist.isChecked(), "go": d.go.text(),
                            "warn": d.warn.text(), "formats": [d.fmt.itemText(i) for i in range(d.fmt.count())]}
        ex.shot(f"export_dialog_{fmt_key}", d)
        i = d.fmt.findData(fmt_key)
        d.fmt.setCurrentIndex(i)
        folder = (OUT / name).as_posix()
        from astro_gpu.client.exportdlg import SUFFIX
        d.dest.setText(f"{folder}/{name}{SUFFIX[fmt_key]}")
        t0 = time.monotonic()
        d.go.click()
        while d.running and time.monotonic() - t0 < 1500:
            ex.settle(4)
        info["seconds"] = round(time.monotonic() - t0, 1)
        info["status"] = d.status.text()
        info["files"] = [str(f) for f in d.files]
        info["error"] = d.error
        info["go_after"] = d.go.text()
        ex.shot(f"export_dialog_{fmt_key}_done", d)
        d.go.click()                                 # Close
    ex.dialog_handler = handle
    ex.click(v.b_export)
    t0 = time.monotonic()
    while "seconds" not in info and time.monotonic() - t0 < 1600:
        ex.settle(4)
    ex.settle(6)
    return info


def compare(arr16: np.ndarray, tag: str):
    import cv2
    H, W = arr16.shape[:2]
    a = arr16.astype(np.float32) / (65535.0 if arr16.dtype == np.uint16 else 255.0)
    half = cv2.resize(a, (screen.shape[1], screen.shape[0]), interpolation=cv2.INTER_AREA) * 255
    d = np.abs(half - screen.astype(np.float32))
    out = {"shape": arr16.shape, "dtype": str(arr16.dtype), "export_stats_8bit": stats((a * 255)),
           "screen_stats": stats(screen), "diff_mean": round(float(d.mean()), 2),
           "diff_p99": round(float(np.percentile(d, 99)), 1), "diff_p999": round(float(np.percentile(d, 99.9)), 1),
           "diff_per_channel_mean": [round(float((half[..., c] - screen[..., c]).mean()), 2) for c in range(3)]}
    # seams: the mean |horizontal / vertical step| per column / row, against its neighbours
    g = a.mean(-1)
    cx = np.abs(np.diff(g, axis=1)).mean(0)
    cy = np.abs(np.diff(g, axis=0)).mean(1)

    def peaks(c):
        med = np.median(c)
        loc = np.convolve(c, np.ones(9) / 9, mode="same")
        z = (c - loc) / (med + 1e-9)
        idx = np.argsort(z)[-6:][::-1]
        return [(int(i), round(float(z[i]), 2)) for i in idx]
    out["seam_cols"] = peaks(cx)
    out["seam_rows"] = peaks(cy)
    hm = (d.max(-1) > 12).astype(np.uint8) * 255
    pi.save_png(np.repeat(hm[..., None], 3, -1), f"export_vs_screen_{tag}_diffmap")
    R("compare", tag, out)
    return out, half


for fk in fmts:
    name = f"iris_{fk}"
    info = run_dialog(fk, name)
    R("export", fk, info)
    report[fk] = {"dialog": info}
    files = [Path(f) for f in info.get("files", [])]
    img = next((f for f in files if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg")), None)
    if img is None:
        continue
    report[fk]["size_bytes"] = img.stat().st_size
    if img.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(img) as tf:
            page = tf.pages[0]
            report[fk]["tiff"] = {"bits": page.bitspersample, "photometric": str(page.photometric),
                                  "icc": "InterColorProfile" in [t.name for t in page.tags],
                                  "compression": str(page.compression), "shape": page.shape}
            arr = page.asarray()
    elif img.suffix.lower() == ".png":
        import cv2
        arr = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    else:
        from PIL import Image
        with Image.open(img) as im:
            report[fk]["jpeg"] = {"mode": im.mode, "size": im.size, "icc": bool(im.info.get("icc_profile"))}
            arr = np.array(im)
    cmp, half = compare(arr, fk)
    report[fk]["compare"] = cmp
    if fk == fmts[0]:
        pi.save_png(screen, "export_screen_half")
        pi.save_png(np.clip(half + 0.5, 0, 255).astype(np.uint8), f"export_{fk}_reduced_to_screen")
dump_json(ex.out / "exports.json", report)
R("errors", pi.slot_errors(), "popups", ex.popups, "modals", ex.modals)
ex.close()
