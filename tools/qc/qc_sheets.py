"""Blind comparison sheets of v1 and v2 renders, in one frame, each render mapped with its own project's files.

What the judges get (``<out>/sheets/``): the reference image first when the projects aligned one, then the renders
under random letters, as an overview, zones at 100 % of the view, the brightest stars and stars of G 8-12 at x2.5,
extra zones at x1.5, and a "100 % checks" sheet (a faint star on plain sky, a star in bright nebula, the densest
field, the brightest star), each cut in parts of three rows; and their brief (the QC prompt with the letters filled in).
Nothing in a sheet or in a file name says which render is old or new, which workflow or project it is.

What they never get (``<out>/_keys/<short>_key.json``): which letter is which render, the seed, the frame used.

The letters are drawn with a random seed (or the one given, for a rerun): the old image is not always A.
"""
from __future__ import annotations

import json
import re
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import blind as old_blind
import cv2
import judge as old_judge
import numpy as np
from PIL import Image, ImageDraw
from qc_addr import QCError, Source
from qc_frame import Render, View, load_render, to_u8
from qc_out import KEYS, SHEETS, sub

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).parent
LETTERS = old_blind.LETTERS
#: words a sheet name must not carry: they would tell the judges which render is which
GIVEAWAY = re.compile(r"(?<![a-z])(old|new|v1|v2|before|after|classic|full|neural|learned|legacy|current|previous|"
                      r"baseline|candidate)(?![a-z])", re.I)


def check_short(short: str) -> str:
    if not short or not re.fullmatch(r"[A-Za-z0-9_.-]+", short):
        raise QCError(f"the sheet name must be letters, digits, _ . -: {short!r}")
    m = GIVEAWAY.search(short)
    if m:
        raise QCError(f"the sheet name {short!r} would tell the judges which render is which ({m.group(0)!r}): "
                      "use a neutral name such as heart_01")
    return short


SHEET_FILES = ("overview.jpg", "zooms.jpg", "bright_p*.jpg", "mid_p*.jpg", "check100.jpg", "zones_p*.jpg", "prompt.md")


def set_files(out: Path, short: str) -> list[Path]:
    """The files of one sheet set (its sheets, brief and key) that exist in the output folder."""
    found = [f for pat in SHEET_FILES for f in (Path(out) / SHEETS).glob(f"{short}_{pat}")]
    key = Path(out) / KEYS / f"{short}_key.json"
    return found + ([key] if key.exists() else [])


def new_seed() -> int:
    return secrets.randbits(32)


def letter_order(n: int, seed: int) -> list[int]:
    """order[i] = the candidate shown under letter i, drawn from the seed."""
    if n > len(LETTERS):
        raise QCError(f"at most {len(LETTERS)} renders on a sheet")
    order = list(range(n))
    import random
    random.Random(seed).shuffle(order)
    return order


# ---------------------------------------------------------------------------------------------- zones and stars

def changed_zones(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None, n: int = 6,
                  k: int = 8) -> list[tuple[float, float, float]]:
    """The n zones where two renders (float images of one frame) differ most, as (x, y fractions, strength in 8-bit
    levels): the largest channel difference smoothed over about 60 px at 1/k, zones at least 600 / k apart."""
    H, W = a.shape[:2]
    d = np.abs(b - a).max(-1)
    if valid is not None:
        d = np.where(valid, d, 0)
    d = cv2.resize(d, (max(W // k, 1), max(H // k, 1)), interpolation=cv2.INTER_AREA)
    d = cv2.GaussianBlur(d, (0, 0), max(60 / k, 1.0))
    out = []
    for _ in range(n):
        y, x = np.unravel_index(np.argmax(d), d.shape)
        if d[y, x] <= 0:
            break
        out.append(((x + 0.5) * k / W, (y + 0.5) * k / H, round(float(d[y, x]) * 255, 2)))
        cv2.circle(d, (int(x), int(y)), max(600 // k, 3), 0.0, -1)
    return out


def default_zones(project_names: list[str], imgs: list[np.ndarray], valids: list[np.ndarray],
                  kind: str) -> list[tuple[str, float, float]]:
    """The named zones of the project when judge.py knows it (they are fractions of the reference image: only in the
    reference frame), else where the first two renders differ most, else a centre and four quarters."""
    for n in project_names if kind == "reference" else []:
        if n in old_judge.ZONES:
            return list(old_judge.ZONES[n])
    if len(imgs) >= 2:
        v = valids[0] & valids[1]
        zs = changed_zones(imgs[0], imgs[1], v, 6)
        if zs:
            return [(f"changed {i + 1}", x, y) for i, (x, y, _) in enumerate(zs)]
    return [("centre", .5, .5), ("NW", .25, .25), ("NE", .75, .25), ("SW", .25, .75), ("SE", .75, .75)]


def parse_zone(text: str) -> tuple[str, float, float]:
    try:
        x, y = (float(v) for v in text.split(":"))
    except ValueError:
        raise QCError(f"a zone is x:y, fractions of the image: {text!r}") from None
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise QCError(f"a zone is x:y with both between 0 and 1: {text!r}")
    return f"zone {x:.2f},{y:.2f}", x, y


def load_catalog(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as f:
        return np.asarray(f["x"], float), np.asarray(f["y"], float), np.asarray(f["g"], float)


def catalog_in_view(view: View, renders: list[Render], valids: list[np.ndarray], margin: int):
    """(u, v, g) of the stars of the first render that has a star list, in the view, away from its edges and inside every
    render. None when no render has one."""
    k = next((i for i, r in enumerate(renders) if r.src.stars), None)
    if k is None:
        return None
    x, y, g = load_catalog(renders[k].src.stars)
    u, v = view.view_of_grid(k, x, y)
    ok = (u >= margin) & (u < view.W - margin) & (v >= margin) & (v < view.H - margin) & np.isfinite(g)
    ui, vi = np.clip(np.round(u).astype(int), 0, view.W - 1), np.clip(np.round(v).astype(int), 0, view.H - 1)
    for va in valids:
        ok &= va[vi, ui]
    return u[ok], v[ok], g[ok]


def check_points(base: np.ndarray, cat, half: int) -> list[tuple[str, float, float]]:
    """The four places of the 100 % look (ENGINE_KNOWLEDGE rule 5): a faint star on plain sky, a star in bright nebula,
    the densest field, the brightest star (spikes). From the star list when there is one, else from the image."""
    H, W = base.shape[:2]
    Y = base.mean(2)
    small = cv2.GaussianBlur(cv2.resize(Y, (max(W // 4, 1), max(H // 4, 1)), interpolation=cv2.INTER_AREA), (0, 0), 8)

    def bg(u, v):
        return small[np.clip((np.asarray(v) / 4).astype(int), 0, small.shape[0] - 1),
                     np.clip((np.asarray(u) / 4).astype(int), 0, small.shape[1] - 1)]

    out: list[tuple[str, float, float]] = []
    if cat is not None and len(cat[0]):
        u, v, g = cat
        ok = (u >= half) & (u < W - half) & (v >= half) & (v < H - half)
        u, v, g = u[ok], v[ok], g[ok]
    else:
        u = v = g = np.array([])
    if len(u):
        faint = np.nonzero((g >= 13) & (g <= 17))[0]
        if not len(faint):
            faint = np.argsort(g)[-max(len(g) // 4, 1):]
        i = faint[np.argmin(bg(u[faint], v[faint]))]
        out.append(("faint star on plain sky", u[i], v[i]))
        mid = np.nonzero((g >= 8) & (g <= 14))[0]
        if len(mid):
            i = mid[np.argmax(bg(u[mid], v[mid]))]
            out.append(("star in bright nebula", u[i], v[i]))
        nx, ny = max(W // 200, 1), max(H // 200, 1)
        hist, xe, ye = np.histogram2d(u, v, bins=(nx, ny), range=((0, W), (0, H)))
        a, b = np.unravel_index(np.argmax(hist), hist.shape)
        out.append(("densest field", (xe[a] + xe[a + 1]) / 2, (ye[b] + ye[b + 1]) / 2))
        i = int(np.argmin(g))
        out.append(("brightest star", u[i], v[i]))
    else:
        c = small[half // 4:-half // 4 or None, half // 4:-half // 4 or None]
        if c.size:
            for name, fn in (("darkest quiet zone", np.argmin), ("brightest zone", np.argmax)):
                y, x = np.unravel_index(fn(c), c.shape)
                out.append((name, (x + half // 4) * 4.0, (y + half // 4) * 4.0))
    return out


# ---------------------------------------------------------------------------------------------- drawing

def _tile(img: np.ndarray, text: str, width: int) -> Image.Image:
    im = Image.fromarray(to_u8(img)).resize((width, max(1, int(width * img.shape[0] / img.shape[1]))), Image.LANCZOS)
    return old_judge.label(im, text, width)


def _crop(u8: np.ndarray, cx: float, cy: float, half: int, scale: float, text: str) -> Image.Image:
    H, W = u8.shape[:2]
    x, y = int(min(max(round(cx), half), W - half)), int(min(max(round(cy), half), H - half))
    t = int(2 * half * scale)
    im = Image.fromarray(np.ascontiguousarray(u8[y - half:y + half, x - half:x + half])).resize((t, t), Image.NEAREST)
    ImageDraw.Draw(im).text((4, 2), text, fill=(255, 255, 0))
    return im


def _grid(rows: list[list[Image.Image]], gap: int = 4) -> Image.Image:
    w, h = rows[0][0].size
    sheet = Image.new("RGB", (len(rows[0]) * (w + gap), len(rows) * (h + gap)), (255, 255, 255))
    for j, r in enumerate(rows):
        for i, t in enumerate(r):
            sheet.paste(t, (i * (w + gap), j * (h + gap)))
    return sheet


def _save_parts(sheet: Image.Image, path: Path, row_h: int, files: list[str]) -> None:
    """The sheet in parts of three rows (blind.cut), written as <path>_pN.jpg."""
    tmp = path.with_suffix(".jpg")
    sheet.save(tmp, quality=92)
    old_blind.cut(tmp, row_h, files)


# ---------------------------------------------------------------------------------------------- the sheets

def blind_sheets(out: str | Path, short: str, sources: list[Source], *, seed: int | None = None,
                 labels: list[str] | None = None, zones: list[str] | None = None, frame: str = "auto",
                 anchor: int = 0, n_bright: int = 6, n_mid: int = 9, extra: dict[str, Any] | None = None,
                 overwrite: bool = False) -> dict[str, Any]:
    """Write the blind sheets of ``sources`` into ``out``. ``labels``: what to call each render in the key file only.
    ``zones``: "x:y" fractions for the extra zone sheet. A sheet set that exists is refused unless ``overwrite``
    (new letters with an old key would mislead): then its sheets and key are replaced. Returns {letters, files, key,
    prompt, seed}."""
    out = Path(out)
    check_short(short)
    if len(sources) < 1:
        raise QCError("no render to put on a sheet")
    mine = set_files(out, short)
    if mine:
        if not overwrite:
            raise QCError(f"sheets {short!r} exist in {out}: pick another name, or overwrite them together with "
                          "their key")
        for f in mine:
            f.unlink()
    seed = new_seed() if seed is None else int(seed)
    renders = [load_render(s) for s in sources]
    view = View(renders, frame, anchor)
    mapped = [view.remap(k) for k in range(len(renders))]
    imgs, valids = [m[0] for m in mapped], [m[1] for m in mapped]
    order = letter_order(len(renders), seed)
    letters = [LETTERS[i] for i in range(len(order))]
    shown = [imgs[i] for i in order]
    sheets, keys = sub(out, SHEETS), sub(out, KEYS)
    files: list[str] = []

    # overview: the reference first (its own picture, not a render), then the letters
    ref = view.reference_image()
    tiles = ([("reference", ref)] if ref is not None else []) + list(zip(letters, shown))
    W = 900
    ts = [_tile(im, t, W) for t, im in tiles]
    ov = Image.new("RGB", (len(ts) * (W + 6), ts[0].height), (255, 255, 255))
    for i, t in enumerate(ts):
        ov.paste(t, (i * (W + 6), 0))
    p = sheets / f"{short}_overview.jpg"
    ov.save(p, quality=90)
    files.append(str(p))

    # zones at 100 % of the view
    named = default_zones(sorted({s.project_name for s in sources}), imgs, valids, view.kind)
    u8 = [to_u8(im) for im in shown]
    ref8 = to_u8(ref) if ref is not None else None
    s, T = 180, 400
    cols = ([("reference", ref8)] if ref8 is not None else []) + list(zip(letters, u8))
    rows = [[_crop(im, fx * view.W, fy * view.H, s, T / (2 * s), f"{lab}: {zn}") for lab, im in cols]
            for zn, fx, fy in named]
    p = sheets / f"{short}_zooms.jpg"
    _grid(rows).save(p, quality=90)
    files.append(str(p))

    # stars at x2.5 and the 100 % checks, from the first star list there is
    cat = catalog_in_view(view, renders, valids, 90)
    if cat is not None and len(cat[0]):
        u, v, g = cat
        bright = np.argsort(g)[:n_bright]
        in_mid = np.nonzero((g >= 8) & (g < 12))[0]
        in_mid = in_mid[np.argsort(g[in_mid])]
        mid = in_mid[np.linspace(0, len(in_mid) - 1, n_mid).astype(int)] if len(in_mid) > n_mid else in_mid
        for name, idx in (("bright", bright), ("mid", mid)):
            if not len(idx):
                continue
            rows = [[_crop(im, u[i], v[i], 80, 2.5, f"{lab}: G{g[i]:.1f}") for lab, im in zip(letters, u8)]
                    for i in idx]
            _save_parts(_grid(rows), sheets / f"{short}_{name}.jpg", 404, files)
    pts = check_points(shown[0], cat, 100)
    if pts:
        rows = [[_crop(im, x, y, 100, 2.0, f"{lab}: {nm}") for lab, im in zip(letters, u8)] for nm, x, y in pts]
        p = sheets / f"{short}_check100.jpg"
        _grid(rows).save(p, quality=92)
        files.append(str(p))

    # extra zones x1.5
    if zones:
        zs = [parse_zone(z) for z in zones]
        rows = [[_crop(im, fx * view.W, fy * view.H, 150, 1.5, f"{lab}: {zn}") for lab, im in zip(letters, u8)]
                for zn, fx, fy in zs]
        _save_parts(_grid(rows), sheets / f"{short}_zones.jpg", 454, files)

    prompt = write_prompt(out, short, files, ref is not None, letters)
    key = {"short": short, "seed": seed, "created": datetime.now().isoformat(timespec="seconds"),
           "letters": {L: {**sources[order[i]].describe(), "label": (labels[order[i]] if labels else None)}
                       for i, L in enumerate(letters)},
           "frame": view.summary(), "files": files, **(extra or {})}
    kp = keys / f"{short}_key.json"
    kp.write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"short": short, "letters": letters, "files": files, "key": str(kp), "prompt": str(prompt), "seed": seed,
            "frame": view.summary()}


def write_prompt(out: Path, short: str, files: list[str], has_reference: bool, letters: list[str]) -> Path:
    """The judges' brief: tools/qc/qc_prompt.md with this sheet set's files and letters. It names letters only."""
    tpl = (HERE / "qc_prompt.md").read_text(encoding="utf-8")
    sheets = Path(out) / SHEETS
    by = lambda tag: sorted(f for f in files if f"_{tag}" in Path(f).name)     # noqa: E731
    rest = [f for f in files if not Path(f).name.endswith(("_overview.jpg", "_zooms.jpg"))]
    tmp = sub(Path(out), "judge_tmp") / short
    text = tpl.format(
        overview=next(iter(by("overview")), ""), zooms=next(iter(by("zooms")), ""), sheets=", ".join(rest),
        columns=", ".join((["reference"] if has_reference else []) + letters), candidates=", ".join(letters),
        python=sys.executable, tmp=tmp)
    p = sheets / f"{short}_prompt.md"
    p.write_text(text, encoding="utf-8")
    return p
