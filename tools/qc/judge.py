"""Comparison sheets for quality control: the user's reference next to renders, in the reference's own framing.

judge.py PROJECT OUT_PREFIX label=render.png [label=render.png ...]
Each render is a saved render PNG of the project (cropped, scaled frame); it is remapped into the reference frame
with the project's reference alignment (work/reference/align_ref.json) and the render preparation geometry.
Writes OUT_PREFIX_overview.jpg and OUT_PREFIX_zooms.jpg (zones at 100 % of the reference scale).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
ZONES = {
    "Horsehead": [("Alnitak", .35, .22), ("Flame", .22, .42), ("Horsehead", .62, .46), ("NGC 2023", .47, .54),
                  ("IC 434 ribbon", .78, .58), ("IC 435", .42, .85), ("faint dust", .25, .75),
                  ("sigma Ori", .855, .05)],
    "Flaming Star": [("AE Aur", .45, .46), ("filaments", .25, .38), ("north faint", .45, .08),
                    ("dust background", .15, .75), ("star field", .80, .70), ("bright star bottom", .67, .94)],
    "Iris": [("Iris core", .42, .62), ("dark lanes", .38, .40), ("faint dust NE", .92, .10), ("dust left", .06, .82),
             ("bright star right", .71, .63), ("star field", .75, .35), ("reflection bottom", .72, .90)],
    "HeartSoul": [("Fish Head", .16, .77), ("Melotte 15 core", .27, .60), ("Heart west rim", .10, .50),
                  ("Heart north rim", .30, .41), ("Soul core", .62, .22), ("Soul east", .80, .16),
                  ("faint filament", .62, .58), ("dark dust", .45, .33), ("star field", .85, .90)],
    "Rosette": [("OIII cavity", .45, .52), ("globules", .35, .40), ("inner rim", .48, .62), ("west rim", .19, .55),
                ("faint extension NE", .72, .22), ("faint extension S", .45, .92), ("star field", .10, .85)],
}
try:
    F = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
except OSError:
    F = ImageFont.load_default()


def to_reference(render_png, proj):
    info = json.loads((proj / "work/reference/align_ref.json").read_text())
    prep = json.loads((proj / "work/render_prep/prep.json").read_text())
    SC = prep["drizzle_scale"]
    x0, x1, y0, y1 = prep["crop"]
    img = np.asarray(Image.open(render_png).convert("RGB")).astype(np.float32)
    Hr, Wr = info["ref_shape"]
    vv, uu = np.mgrid[0:Hr, 0:Wr].astype(np.float64)
    a, b = uu / Wr * 2 - 1, vv / Hr * 2 - 1
    terms = [a ** i * b ** j for i in range(4) for j in range(4 - i)]
    mx = sum(c * t for c, t in zip(info["cx"], terms))
    my = sum(c * t for c, t in zip(info["cy"], terms))
    # stack coordinates -> render PNG coordinates (crop, then the drizzle scale)
    sx = img.shape[1] / ((x1 - x0) / SC)
    sy = img.shape[0] / ((y1 - y0) / SC)
    px = ((mx - x0) / SC * sx).astype(np.float32)
    py = ((my - y0) / SC * sy).astype(np.float32)
    out = np.stack([cv2.remap(np.ascontiguousarray(img[..., c]), px, py, cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT) for c in range(3)], -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def label(im, text, w):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 26], fill=(20, 22, 30))
    d.text((5, 1), text, font=F, fill=(240, 240, 235))
    return im


def main():
    proj = Path(sys.argv[1])
    out = sys.argv[2]
    info = json.loads((proj / "work/reference/align_ref.json").read_text())
    ref = np.asarray(Image.open(info["reference"]).convert("RGB"))
    ims = [("reference", ref)] + [(a.split("=", 1)[0], to_reference(a.split("=", 1)[1], proj)) for a in sys.argv[3:]]
    # overview
    W = 900
    tiles = []
    for lab, im in ims:
        t = Image.fromarray(im).resize((W, int(W * im.shape[0] / im.shape[1])), Image.LANCZOS)
        tiles.append(label(t, lab, W))
    sheet = Image.new("RGB", (len(tiles) * (W + 6), tiles[0].height), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (W + 6), 0))
    sheet.save(f"{out}_overview.jpg", quality=90)
    # zooms at 100 %
    H, Wr = ref.shape[:2]
    s, T = 180, 400
    rows = []
    for zn, fx, fy in ZONES[json.loads((proj / "project.json").read_text(encoding="utf-8"))["name"]]:
        cx, cy = int(fx * Wr), int(fy * H)
        cx, cy = min(max(cx, s), Wr - s), min(max(cy, s), H - s)
        rows.append([label(Image.fromarray(np.ascontiguousarray(im[cy - s:cy + s, cx - s:cx + s])).resize(
            (T, T), Image.NEAREST), f"{lab}: {zn}", T) for lab, im in ims])
    sheet = Image.new("RGB", (len(ims) * (T + 4), len(rows) * (T + 4)), (255, 255, 255))
    for j, r in enumerate(rows):
        for i, c in enumerate(r):
            sheet.paste(c, (i * (T + 4), j * (T + 4)))
    sheet.save(f"{out}_zooms.jpg", quality=90)
    print(f"{out}_overview.jpg", f"{out}_zooms.jpg")


if __name__ == "__main__":
    main()
