"""Blind comparison sheets: candidate renders of one project shown under letters, the key saved apart.

    python tools/qc/blind.py PROJECT OUT_DIR SHORT label=render.png [label=render.png ...] [--zones x:y ...]

The candidates are shuffled under the letters A, B, C... (the order is fixed by SHORT); the key (letter -> label) is
written to OUT_DIR/_keys/SHORT_key.json, which the judges are never shown. Sheets written in OUT_DIR:
- SHORT_overview.jpg, SHORT_zooms.jpg: the user's reference first, then the letters, in the reference's framing
  (only when the project has an aligned reference and its zones are listed in judge.py);
- SHORT_bright_stars_p*.jpg: the brightest stars at 100 % (x2.5); SHORT_mid_stars_p*.jpg: stars of G 8-12;
- SHORT_zones_p*.jpg: the zones given as fractions of the render (x1.5).
Star and zone sheets are cut in parts of three rows. Prints the letters and the files as JSON; the instruction for the
judges is tools/qc/qc_prompt.md.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).parent
LETTERS = "ABCDEFGH"


def cut(path: Path, row_h: int, parts: list) -> None:
    S = Image.open(path)
    rows = S.size[1] // row_h
    for k in range(0, rows, 3):
        f = path.with_name(f"{path.stem}_p{k // 3 + 1}.jpg")
        S.crop((0, k * row_h, S.size[0], min(S.size[1], (k + 3) * row_h))).save(f, quality=92)
        parts.append(str(f))
    path.unlink()


def main():
    P, out, short = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    args, cands, zones = sys.argv[4:], [], []
    while args:
        a = args.pop(0)
        if a == "--zones":
            while args and not args[0].startswith("--"):
                zones.append(args.pop(0))
        else:
            cands.append(a.split("=", 1))
    order = cands[:]
    random.Random(short).shuffle(order)
    key = {LETTERS[i]: n for i, (n, _) in enumerate(order)}
    (out / "_keys").mkdir(parents=True, exist_ok=True)
    (out / "_keys" / f"{short}_key.json").write_text(json.dumps(key, indent=1))
    lettered = [f"{LETTERS[i]}={f}" for i, (_, f) in enumerate(order)]
    py = sys.executable
    files = []
    if (P / "work/reference/align_ref.json").exists():
        r = subprocess.run([py, str(HERE / "judge.py"), str(P), str(out / short)] + lettered,
                           capture_output=True, text=True)
        if r.returncode == 0:
            files += [str(out / f"{short}_overview.jpg"), str(out / f"{short}_zooms.jpg")]
        else:
            print("reference sheets skipped:", r.stderr.strip().splitlines()[-1:], file=sys.stderr)
    vs = str(HERE / "variants_sheet.py")
    subprocess.run([py, vs, str(P), str(out / f"{short}_bright")] + lettered + ["--stars", "6"], check=True)
    cut(out / f"{short}_bright_stars.jpg", 404, files)
    extra = ["--zones"] + zones if zones else []
    subprocess.run([py, vs, str(P), str(out / f"{short}_mid")] + lettered + ["--stars", "9", "--grange", "8:12"]
                   + extra, check=True)
    cut(out / f"{short}_mid_stars.jpg", 404, files)
    if zones:
        cut(out / f"{short}_mid_zones.jpg", 454, files)
    print(json.dumps({"letters": list(key), "files": files}))


if __name__ == "__main__":
    main()
