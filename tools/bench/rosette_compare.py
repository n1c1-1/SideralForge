"""The Rosette verdict's comparison: one of our images against the old render2, on the same pixels.

    python tools/bench/rosette_compare.py IMAGE OUT_DIR [--name "label"]

IMAGE is an export of ours (TIFF/PNG, 8 or 16 bits, any size and framing). It is registered onto the reference
(render2.png, 4025 x 2752, 16 bits) by its stars and nebula (SIFT features, a similarity transform fitted with RANSAC),
then compared:
- a board per zone (the reference on the left, ours on the right, the same crop at full resolution): the core, the
  dark globules at the bottom left of the ring, the faint filaments at the right edge, the sky in a corner;
- measures on 0-1 display values (the same for both): the sky's noise (robust sigma of the fine scale in the corner),
  the black and white clipping over the whole frame, the local contrast in the core and in the filaments (the robust
  sigma of a band-pass, 2-16 px), each image's own sky level and colour.
Writes OUT_DIR/board.jpg, OUT_DIR/measures.json and OUT_DIR/registered.png (ours on the reference's pixels).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REF = Path(r"E:\astro_gpu\projects\Rosette\results\render2\render2.png")      # read only
#: the zones (x, y, w, h) on the reference's 4025 x 2752 pixels
ZONES = {
    "Cœur": (1640, 960, 900, 620),
    "Globules sombres": (1560, 1660, 900, 620),
    "Filaments faibles du bord": (2650, 1000, 900, 620),
    "Fond de ciel": (150, 2050, 900, 620),
}


def read(path: Path) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise SystemExit(f"cannot read {path}")
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    a = a[..., :3]
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    return (a.astype(np.float32) / scale)[..., ::-1].copy()             # RGB, 0-1


def gray8(a: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)                         # the same features in both renderings


#: the registration of the full project's exports, found once on a well-exposed trial and reused for every trial
#: (they all have the same framing: the same project, the same crop): a dark or a strongly filtered rendering has too
#: few features for SIFT
TRANSFORM = Path("F:/astro_essais/rosette_verdict/controles/recalage.json")


def register(ours: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, dict]:
    """Ours warped onto the reference's pixels: the transform saved for this image size, else SIFT on both (ours
    brought near the reference's size first), the matches filtered by Lowe's ratio, a similarity transform (rotation,
    scale, shift) by RANSAC, saved when it is sound."""
    k = ref.shape[1] / ours.shape[1]
    small = cv2.resize(ours, (ref.shape[1], int(round(ours.shape[0] * k))), interpolation=cv2.INTER_AREA)
    saved = json.loads(TRANSFORM.read_text(encoding="utf-8")) if TRANSFORM.exists() else {}
    key = f"{ours.shape[1]}x{ours.shape[0]}"
    if key in saved:
        T = np.array(saved[key]["T"], np.float64)
        out = cv2.warpAffine(small, T, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LANCZOS4, borderValue=0)
        return out, {**saved[key]["info"], "from": "saved transform"}
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(gray8(small), None)
    kb, db = sift.detectAndCompute(gray8(ref), None)
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [a for a, b in m if a.distance < 0.75 * b.distance]
    src = np.float32([ka[g.queryIdx].pt for g in good])
    dst = np.float32([kb[g.trainIdx].pt for g in good])
    T, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=20000)
    if T is None:
        raise SystemExit("the registration failed")
    out = cv2.warpAffine(small, T, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LANCZOS4, borderValue=0)
    res = np.hypot(*(cv2.transform(src[None], T)[0] - dst).T)[inl.ravel() == 1]
    info = {"matches": len(good), "inliers": int(inl.sum()), "residual_px": round(float(np.median(res)), 3),
            "scale": round(float(np.hypot(T[0, 0], T[0, 1]) * k), 4),
            "rotation_deg": round(float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))), 3)}
    sound = info["inliers"] >= 8 and abs(info["rotation_deg"]) < 2 and 0.5 < info["scale"] < 2
    if not sound:
        raise SystemExit(f"the registration is not sound: {info}")
    saved[key] = {"T": T.tolist(), "info": info}
    TRANSFORM.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    return out, info


def robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def band(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = a.mean(2)
    return cv2.GaussianBlur(g, (0, 0), lo) - cv2.GaussianBlur(g, (0, 0), hi)


def measures(a: np.ndarray, valid: np.ndarray) -> dict:
    out = {}
    lum = a.mean(2)[valid]
    out["noir_ecrete_pct"] = round(100 * float((lum <= 1 / 255).mean()), 3)
    out["blanc_ecrete_pct"] = round(100 * float((a[valid].max(1) >= 254 / 255).mean()), 3)
    x, y, w, h = ZONES["Fond de ciel"]
    sky = a[y:y + h, x:x + w]
    out["ciel_niveau"] = [round(float(np.median(sky[..., c])), 4) for c in range(3)]
    out["ciel_bruit_fin"] = round(robust_sigma(band(sky, 0.7, 3.0)), 5)
    for z in ("Cœur", "Filaments faibles du bord", "Globules sombres"):
        x, y, w, h = ZONES[z]
        out[f"contraste_local_{z}"] = round(robust_sigma(band(a[y:y + h, x:x + w], 2.0, 16.0)), 5)
    return out


def board(ref: np.ndarray, ours: np.ndarray, name: str) -> np.ndarray:
    rows = []
    for z, (x, y, w, h) in ZONES.items():
        pair = []
        for img, who in ((ref, "render2"), (ours, name)):
            t = (np.clip(img[y:y + h, x:x + w], 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
            cv2.putText(t, f"{z} - {who}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            pair.append(t)
        rows.append(np.hstack([pair[0], np.full((h, 8, 3), 40, np.uint8), pair[1]]))
    return np.vstack([np.vstack([r, np.full((8, r.shape[1], 3), 40, np.uint8)]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--name", default="ours")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = read(REF)
    ours, info = register(read(Path(a.image)), ref)
    valid = ours.max(2) > 0
    cv2.imwrite(str(out / "registered.png"), (np.clip(ours, 0, 1) * 65535).astype(np.uint16)[..., ::-1])
    cv2.imwrite(str(out / "board.jpg"), board(ref, ours, a.name), [cv2.IMWRITE_JPEG_QUALITY, 92])
    whole = [cv2.resize((np.clip(x, 0, 1) * 255).astype(np.uint8)[..., ::-1], (1400, 957), interpolation=cv2.INTER_AREA)
             for x in (ref, ours)]
    for im, who in zip(whole, ("render2", a.name)):
        cv2.putText(im, who, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(out / "whole.jpg"), np.vstack([whole[0], np.full((8, 1400, 3), 40, np.uint8), whole[1]]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    both = {"recalage": info, "render2": measures(ref, valid), a.name: measures(ours, valid)}
    (out / "measures.json").write_text(json.dumps(both, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(both, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
