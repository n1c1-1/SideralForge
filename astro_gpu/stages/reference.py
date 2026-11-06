"""Optional: align a previous processing of the same data (e.g. your own version) for comparison and framing."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.params import P, StageParams
from ..core.stage import Context, Stage, StageError


class ReferenceParams(StageParams):
    max_stars: int = P(2500, label="Stars used", ge=200, le=20000,
                       help="Stars detected in the reference image for the alignment.")


class Reference(Stage):
    name = "reference"
    title = "Reference image (optional)"
    description = ("If the project has a reference image (for example your own earlier processing of these data), "
                   "finds its orientation and a precise polynomial mapping to the stack, so that it can be shown side "
                   "by side with synchronised zoom and the renders can be exported with exactly the same framing.")
    deps = ("stack",)
    Params = ReferenceParams
    version = "1"

    def run(self, ctx: Context, p: ReferenceParams) -> None:
        ref_path = ctx.project.config.reference_image
        if not ref_path:
            ctx.save_json("align_ref.json", {"reference": None})
            ctx.log("no reference image in this project")
            return
        if not Path(ref_path).exists():
            raise StageError(f"reference image not found: {ref_path}")
        S = float(ctx.load_json("stack", "drz_meta.json")["scale"])
        try:
            out = align_reference(ref_path, ctx.load("stack", "drz_full.npy", mmap=True), S, p.max_stars,
                                  lambda f, m: ctx.progress(f, m))
        except StageError as e:
            # optional stage: never block the processing
            ctx.save_json("align_ref.json", {"reference": None, "error": str(e)})
            ctx.log(f"reference image not used: {e}", "warning")
            return
        out["reference"] = ref_path
        ctx.save_json("align_ref.json", out)
        ctx.metric("orientation", {"mirror": out["flip"], "rot90": out["k"]})
        ctx.metric("stars_matched", out["n"])
        ctx.metric("residual_px", out["residual"])
        ctx.log(f"reference aligned: mirror {out['flip']}, rotation {90 * out['k']} deg, {out['n']} stars, "
                f"residual {out['residual']:.2f} px")


def _stars(gray, n, k=10.0):
    import cv2
    import sep
    x = gray.astype(np.float32)
    hp = np.ascontiguousarray(x - cv2.GaussianBlur(x, (0, 0), 6))
    s = 1.4826 * np.median(np.abs(hp - np.median(hp)))
    sep.set_extract_pixstack(3000000)
    from ..core.imgutil import sep_extract
    o = sep_extract(hp, k * s, minarea=4, deblend_nthresh=8)
    o = np.sort(o[o["flag"] == 0], order="flux")[::-1][:n]
    return np.c_[o["x"], o["y"]]


def orient_xy(x, y, H, W, k, flip):
    """Coordinates after a vertical mirror (optional) then k rotations by 90 deg (numpy rot90 convention)."""
    if flip:
        y = H - 1 - y
    h, w = H, W
    for _ in range(k % 4):
        x, y = y, w - 1 - x
        h, w = w, h
    return x, y


def _poly(u, v, sx, sy):
    a, b = u / sx * 2 - 1, v / sy * 2 - 1
    return np.stack([a ** i * b ** j for i in range(4) for j in range(4 - i)], -1)


def _feature_orientation(ref, gd, S, size=1400, min_inliers=12):
    """Orientation and similarity transform (oriented stack / S -> reference) from SIFT features of both images,
    in the format of the star matching result; None when nothing matches."""
    import cv2
    from skimage.transform import SimilarityTransform

    def prep(a):
        a = a - np.percentile(a, 1)
        a = np.clip(a / max(float(np.percentile(a, 99.5)), 1e-6), 0, 1)
        return (a * 255).astype(np.uint8)

    sr = size / max(ref.shape)
    a = cv2.resize(prep(ref), None, fx=sr, fy=sr, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(5000)
    ka, da = sift.detectAndCompute(a, None)
    if da is None:
        return None
    best = None
    for flip in (False, True):
        for k in range(4):
            g = np.rot90(gd[::-1] if flip else gd, k)          # the orientation convention of orient_xy
            sb = size / max(g.shape)
            b = cv2.resize(prep(np.ascontiguousarray(g)), None, fx=sb, fy=sb, interpolation=cv2.INTER_AREA)
            kb, db = sift.detectAndCompute(b, None)
            if db is None:
                continue
            good = [m for m, n in cv2.BFMatcher().knnMatch(db, da, k=2) if m.distance < 0.75 * n.distance]
            if len(good) < min_inliers:
                continue
            pb = np.float32([kb[m.queryIdx].pt for m in good]) / sb / S       # oriented stack, sensor scale
            pa = np.float32([ka[m.trainIdx].pt for m in good]) / sr           # reference
            M, inl = cv2.estimateAffinePartial2D(pb, pa, ransacReprojThreshold=3.0 / sr)
            n_in = int(inl.sum()) if inl is not None else 0
            if M is None or n_in < min_inliers:
                continue
            T = SimilarityTransform(matrix=np.vstack([M, [0, 0, 1]]))
            score = (abs(np.degrees(T.rotation)) < 5, n_in)
            if best is None or score > best[4]:
                best = (flip, k, T, n_in, score)
    return best


def align_reference(path, drz, S, nmax, progress=lambda f, m: None):
    import astroalign as aa
    from PIL import Image
    from scipy.spatial import cKDTree
    Image.MAX_IMAGE_PIXELS = None
    ref = np.asarray(Image.open(path).convert("RGB"), np.float32).mean(2) / 255
    Hr, Wr = ref.shape
    gd = np.asarray(drz[1], np.float32)
    H, W = gd.shape
    gd = np.arcsinh(np.clip(gd - np.median(gd), 0, None) / 20)
    A = _stars(ref, nmax)
    Bd = _stars(gd, int(nmax * 1.6))
    # the reference may be cropped, resized or rotated: the similarity transform absorbs scale and rotation
    best = None
    for flip in (False, True):
        for k in range(4):
            progress((2 * k + flip) / 10, "testing orientations")
            bx, by = orient_xy(Bd[:, 0], Bd[:, 1], H, W, k, flip)
            try:
                T, (s_, _) = aa.find_transform(np.c_[bx, by][:1500] / S, A[:1000], max_control_points=100)
            except Exception:  # noqa: BLE001
                continue
            # prefer the orientation needing the least residual rotation (exact rot90 exports), then more stars
            score = (abs(np.degrees(T.rotation)) < 5, len(s_))
            if best is None or score > best[4]:
                best = (flip, k, T, len(s_), score)
    coarse = False
    if best is None:
        # a heavily processed reference (stretched, saturated, sharpened) no longer ranks its stars like the stack:
        # the star triangles find nothing. Its image features (nebula and star pattern together) still match
        progress(0.85, "matching image features")
        best = _feature_orientation(ref, gd, S)
        coarse = True
    if best is None:
        from ..core.stage import StageError
        raise StageError("the reference image could not be matched with the stack")
    flip, k, T, _, _ = best
    bx, by = orient_xy(Bd[:, 0], Bd[:, 1], H, W, k, flip)
    if coarse:                     # the feature transform is good to a few pixels: refine it on the stars
        from skimage.transform import SimilarityTransform
        for rad in (8.0, 4.0):
            dd, j = cKDTree(A).query(T(np.c_[bx, by] / S), distance_upper_bound=rad)
            m = np.isfinite(dd)
            if m.sum() >= 20:
                T = SimilarityTransform()
                T.estimate(np.c_[bx, by][m] / S, A[j[m]])
    Pp = T(np.c_[bx, by] / S)
    dd, j = cKDTree(A).query(Pp, distance_upper_bound=2.0)
    m = np.isfinite(dd)
    U, V = A[j[m], 0], A[j[m], 1]
    Xd, Yd = Bd[m, 0], Bd[m, 1]
    Tm = _poly(U, V, Wr, Hr)
    cx = np.linalg.lstsq(Tm, Xd, rcond=None)[0]
    cy = np.linalg.lstsq(Tm, Yd, rcond=None)[0]
    ok = np.ones(len(U), bool)
    for _ in range(3):
        r = np.hypot(Tm @ cx - Xd, Tm @ cy - Yd)
        ok = r < 3 * np.median(r) + 0.3
        cx = np.linalg.lstsq(Tm[ok], Xd[ok], rcond=None)[0]
        cy = np.linalg.lstsq(Tm[ok], Yd[ok], rcond=None)[0]
    r = np.hypot(Tm @ cx - Xd, Tm @ cy - Yd)
    return {"flip": bool(flip), "k": int(k), "ref_shape": [Hr, Wr], "cx": cx.tolist(), "cy": cy.tolist(),
            "n": int(ok.sum()), "residual": float(np.median(r[ok])), "rotation_deg": float(np.degrees(T.rotation)),
            "scale": float(T.scale)}
