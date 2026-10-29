"""Side-by-side sheets of render variants of one project: bright stars at 100 % (x2.5), then zones (x1.5).

variants_sheet.py PROJECT OUT_PREFIX name=render.png [name=render.png ...] [--stars N] [--grange G0:G1] [--zones x:y ...]

The renders are saved render PNGs of the project (its crop, at the sensor scale). Zones in stack pixels, or fractions
of the render when both are <= 1. --grange: stars spread over that Gaia magnitude range instead of the brightest.
Writes OUT_PREFIX_stars.jpg and OUT_PREFIX_zones.jpg.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def main():
    P = Path(sys.argv[1])
    out = sys.argv[2]
    args = sys.argv[3:]
    vars_, nstars, zones, grange = [], 8, [], None
    while args:
        a = args.pop(0)
        if a == "--stars":
            nstars = int(args.pop(0))
        elif a == "--grange":
            grange = tuple(float(v) for v in args.pop(0).split(":"))
        elif a == "--zones":
            while args and not args[0].startswith("--"):
                zones.append(tuple(float(v) for v in args.pop(0).split(":")))
        else:
            vars_.append(a.split("=", 1))
    # the render preparation may be running (its folder emptied): keep its geometry next to the sheets
    cache = Path(out).parent / "_prep" / (P.name + ".json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    pf = P / "work/render_prep/prep.json"
    if pf.exists():
        cache.write_text(pf.read_text())
    prep = json.loads(cache.read_text())
    SC = prep["drizzle_scale"]
    x0, x1, y0, y1 = prep["crop"]
    with np.load(P / "work/stars/star_fluxes.npz") as fs:          # read once: an npz reads an array at each access
        fs = {k: fs[k] for k in ("x", "y", "g")}
    g = fs["g"]
    ims = [(n, Image.open(f).convert("RGB")) for n, f in vars_]
    inside = (fs["x"] > x0 + 150) & (fs["x"] < x1 - 150) & (fs["y"] > y0 + 150) & (fs["y"] < y1 - 150)
    sel = [int(i) for i in np.argsort(g) if inside[i]]
    if grange:
        sel = [i for i in sel if grange[0] <= g[i] < grange[1]]
        sel = [sel[int(k)] for k in np.linspace(0, len(sel) - 1, nstars)] if len(sel) > nstars else sel
    else:
        sel = sel[:nstars]
    W0, H0 = ims[0][1].size
    zones = [((x * W0) * SC + x0, (y * H0) * SC + y0) if x <= 1 and y <= 1 else (x, y) for x, y in zones]

    def sheet(points, half, scale, path, labels):
        T = int(2 * half * scale)
        S = Image.new("RGB", (len(ims) * (T + 4), len(points) * (T + 4)), (255, 255, 255))
        for r, ((xs, ys), lab) in enumerate(zip(points, labels)):
            cx, cy = (xs - x0) / SC, (ys - y0) / SC
            for c, (n, im) in enumerate(ims):
                t = im.crop((int(cx) - half, int(cy) - half, int(cx) + half, int(cy) + half)).resize((T, T),
                                                                                                    Image.NEAREST)
                ImageDraw.Draw(t).text((4, 2), f"{n}: {lab}", fill=(255, 255, 0))
                S.paste(t, (c * (T + 4), r * (T + 4)))
        S.save(path, quality=90)

    sheet([(fs["x"][i], fs["y"][i]) for i in sel], 80, 2.5, f"{out}_stars.jpg",
          [f'G{g[i]:.1f} ({fs["x"][i]:.0f},{fs["y"][i]:.0f})' for i in sel])
    if zones:
        sheet(zones, 150, 1.5, f"{out}_zones.jpg", [f"zone ({x:.0f},{y:.0f})" for x, y in zones])
    print(out)


if __name__ == "__main__":
    main()
