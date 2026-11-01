"""P17: the QC tools of v2 (tools/qc/qc2.py and the qc_*.py modules), on toy projects in temporary folders.

The toy sky is an analytic scene (nebula and stars) sampled by projects whose grids differ by a few pixels: a render is
the scene seen through its own project's crop, so a mapping is right when every render comes back to the scene.
v1 toys are project.json + work/ + results/; v2 toys are project.astroproj + a store at the real keys.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from astro_gpu.core import import_v1 as iv
from astro_gpu.core import project2 as p2
from astro_gpu.core import workflow as wfl
from astro_gpu.core.store import Keys

QC = Path(__file__).resolve().parents[1] / "tools" / "qc"
if str(QC) not in sys.path:
    sys.path.insert(0, str(QC))

import qc2  # noqa: E402
import qc_addr  # noqa: E402
import qc_frame  # noqa: E402
import qc_out  # noqa: E402
import qc_pairs  # noqa: E402
import qc_render  # noqa: E402
import qc_report  # noqa: E402
import qc_sheets  # noqa: E402

# ---------------------------------------------------------------------------------------------- the toy sky
SC = 2.0                                  # drizzle scale: grid pixels per sensor pixel
GRID = (1320, 920)                        # the grid of each project (W, H), in grid pixels
CROP = (20, 1300, 16, 904)                # the useful frame, in grid pixels -> a 640 x 444 sensor-scale image
REF = (600, 400)                          # the user's reference image (W, H), at sensor scale
REF0 = (30.0, 25.0)                       # where its first pixel is on the sky
BLOBS = [(160, 140, 40, (.50, .20, .10)), (330, 230, 55, (.10, .20, .50)), (450, 150, 25, (.30, .30, .00))]
#: stars (x, y, amplitude, G): on the sky, away from the edges of the reference
STARS = [(150 + 37 * i, 125 + 31 * (i % 6), .95 - .05 * i, 6.5 + 0.9 * i) for i in range(11)]


def scene(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """The sky at sensor-scale edge coordinates (px, py): float32 (..., 3)."""
    out = np.zeros(px.shape + (3,), np.float32) + .08
    for cx, cy, s, col in BLOBS:
        out += np.exp(-((px - cx) ** 2 + (py - cy) ** 2) / (2 * s * s))[..., None].astype(np.float32) * np.array(col, np.float32)
    for cx, cy, a, _ in STARS:
        out += (a * np.exp(-((px - cx) ** 2 + (py - cy) ** 2) / (2 * 2.2 ** 2)))[..., None].astype(np.float32)
    return np.clip(out, 0, 1)


@dataclasses.dataclass(frozen=True)
class Geo:
    """One project's grid: where its first pixel is on the sky (sensor pixels)."""
    shift: tuple[float, float]

    def image(self, scale: float = 1.0, crop=CROP) -> np.ndarray:
        """The render of the useful frame at ``scale`` x the sensor scale (float32 HWC)."""
        x0, x1, y0, y1 = crop
        w, h = round((x1 - x0) / SC * scale), round((y1 - y0) / SC * scale)
        f = (x1 - x0) / w, (y1 - y0) / h
        gx = x0 + (np.arange(w) + .5) * f[0]
        gy = y0 + (np.arange(h) + .5) * f[1]
        px, py = np.meshgrid(gx / SC + self.shift[0], gy / SC + self.shift[1])
        return scene(px, py)

    def align(self, ref_file: Path) -> dict:
        """align_ref.json: the cubic map from a reference pixel to a grid pixel (here a plain scaling and shift)."""
        Wr, Hr = REF
        cx = [0.0] * 10
        cy = [0.0] * 10
        # grid = SC * (sky - shift) - .5, sky = ref pixel + .5 + REF0; a = u / Wr * 2 - 1
        cx[0] = SC * Wr / 2 + SC * (.5 + REF0[0] - self.shift[0]) - .5
        cx[4] = SC * Wr / 2                                  # the term a
        cy[0] = SC * Hr / 2 + SC * (.5 + REF0[1] - self.shift[1]) - .5
        cy[1] = SC * Hr / 2                                  # the term b
        return {"reference": str(ref_file), "ref_shape": [Hr, Wr], "cx": cx, "cy": cy, "k": 0, "flip": False,
                "n": 50, "residual": .1}

    def wcs(self, path: Path) -> None:
        from astropy.io import fits
        from astropy.wcs import WCS
        w = WCS(naxis=2)
        s = 3.0 / SC / 3600                                  # 3 arcsec per sensor pixel
        pc = (330.0, 230.0)                                  # the tangent point, on the sky
        gc = [SC * (pc[i] - self.shift[i]) - .5 for i in range(2)]
        w.wcs.crpix = [gc[0] + 1, gc[1] + 1]
        w.wcs.cd = [[-s, 0], [0, s]]
        w.wcs.crval = [150.0, 20.0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        fits.PrimaryHDU(header=w.to_header()).writeto(path, overwrite=True)

    def stars(self, path: Path) -> None:
        x = [SC * (s[0] - self.shift[0]) - .5 for s in STARS]
        y = [SC * (s[1] - self.shift[1]) - .5 for s in STARS]
        np.savez(path, x=x, y=y, g=[s[3] for s in STARS])


A, B = Geo((0.0, 0.0)), Geo((3.2, -2.4))


def png16(path: Path, img: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", (np.clip(img, 0, 1)[:, :, ::-1] * 65535 + .5).astype(np.uint16))
    assert ok
    path.write_bytes(buf.tobytes())
    return path


def reference_png(path: Path) -> Path:
    px, py = np.meshgrid(np.arange(REF[0]) + .5 + REF0[0], np.arange(REF[1]) + .5 + REF0[1])
    return png16(path, scene(px, py))


def prep_json(simple: bool = False) -> dict:
    d = {"crop": list(CROP), "drizzle_scale": SC, "width": GRID[0], "height": GRID[1]}
    return {**d, "exit": None, "denoised": True} if simple else d


def settings_json(prep: str | None, learned: bool | None, date: str, orientation=(0, False), qa=None) -> dict:
    s = {"date": date, "settings": {"chroma": .9, **({"learned_stars": learned} if learned is not None else {})},
         "export": {"name": "x"}, "orientation": {"rot90": orientation[0], "flip": orientation[1]},
         "qa": qa or {"pure_black_pct": 0.0, "warnings": []}, "variant": "main"}
    if prep:
        s["preparation"] = prep
    return s


# ---------------------------------------------------------------------------------------------- toy v1 projects

def make_v1(root: Path, geo: Geo, name: str = "Toy", params: dict | None = None, images: dict | None = None,
            with_ref: bool = True) -> Path:
    """A v1 project. ``images``: {deliverable: (preparation or None, learned_stars or None)}."""
    images = images if images is not None else {"faithful": ("render_prep_learned", True)}
    ref = reference_png(root / "reference" / "ref.png") if with_ref else None
    (root / "work").mkdir(parents=True, exist_ok=True)
    cfg = {"name": name, "inputs": {"lights": [str(root / "raw")], "darks": [], "flats": [], "bias": [], "flat_darks": []},
           "target": {"name": "T"}, "reference_image": str(ref) if ref else "", "params": params or {}, "variants": {},
           "active_variant": "", "format": 1}
    (root / "project.json").write_text(json.dumps(cfg), encoding="utf-8")
    for prep in ("render_prep", "render_prep_learned"):
        (root / "work" / prep).mkdir(parents=True, exist_ok=True)
        (root / "work" / prep / "prep.json").write_text(json.dumps(prep_json()), encoding="utf-8")
    for st, metrics in {"inventory": {"n_lights": 40, "exposure_s": 10.0}, "stack": {"frames_used": 38},
                        "background": {"flat_error_pct": 0.4}}.items():
        (root / "work" / st).mkdir(parents=True, exist_ok=True)
        (root / "work" / st / "manifest.json").write_text(json.dumps({"status": "done", "metrics": metrics}))
    (root / "work" / "astrometry").mkdir(parents=True, exist_ok=True)
    geo.wcs(root / "work" / "astrometry" / "wcs.fits")
    (root / "work" / "stars").mkdir(parents=True, exist_ok=True)
    geo.stars(root / "work" / "stars" / "star_fluxes.npz")
    if ref:
        (root / "work" / "reference").mkdir(parents=True, exist_ok=True)
        (root / "work" / "reference" / "align_ref.json").write_text(json.dumps(geo.align(ref)), encoding="utf-8")
    for i, (n, (prep, learned)) in enumerate(images.items()):
        d = root / "results" / n
        png16(d / f"{n}.png", geo.image())
        (d / f"{n}_settings.json").write_text(json.dumps(settings_json(prep, learned, f"2026-10-17T19:2{i}:00")))
    return root


# ---------------------------------------------------------------------------------------------- toy v2 projects
LAYER = "earlier_layer.npy"


def make_v2(root: Path, geo: Geo, workflows: list[str] | None = None, with_ref: bool = True,
            exports: dict[str, np.ndarray] | None = None, name: str = "Toy") -> p2.Project2:
    """A v2 project with the given template workflows computed in its store (at the keys the tools will compute)."""
    names = workflows or ["Classic stack", "Full", "Full + neural stars", "Full + earlier neural layer"]
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for f in ("a.fits", "b.fits"):
        (raw / f).write_bytes(b"")
    p = p2.create(root, name)
    p.sessions.append(p2.Session(id=1, name="n1", lights=[str(raw)]))
    ref = reference_png(root / "inputs" / "ref.png") if with_ref else None
    p.reference_image = "inputs/ref.png" if ref else ""
    layer = root / "inputs" / LAYER
    layer.parent.mkdir(parents=True, exist_ok=True)
    layer.write_bytes(b"x")
    for n in names:
        tpl = n if n in wfl.TEMPLATES else "Full + neural stars"
        params = {"starless": {"learned_stars": str(layer)}} if n == "Full + earlier neural layer" else {}
        p.add_workflow(n, dict(wfl.TEMPLATES[tpl]), params)
    p2.save(p)
    keys = Keys(p)
    for w in p.workflows:
        for s in keys.chain(w):
            d = root / "store" / s / keys.key(w, s)
            d.mkdir(parents=True, exist_ok=True)
            metrics = {"stack": {"frames_used": 39}, "inventory": {"n_lights": 40}}.get(s, {})
            (d / "manifest.json").write_text(json.dumps({"status": "done", "key": keys.key(w, s), "metrics": metrics}))
            if s in ("render_prep", "render_prep_learned"):
                (d / "prep.json").write_text(json.dumps(prep_json()))
            elif s == "render_simple":
                (d / "prep_simple.json").write_text(json.dumps(prep_json(True)))
            elif s == "astrometry":
                geo.wcs(d / "wcs.fits")
            elif s == "stars":
                geo.stars(d / "star_fluxes.npz")
            elif s == "reference" and ref:
                (d / "align_ref.json").write_text(json.dumps(geo.align(ref)))
    for n, img in (exports or {}).items():
        slug = qc_out.slug(n)
        out = root / "export" / slug
        out.mkdir(parents=True, exist_ok=True)
        import tifffile
        tifffile.imwrite(out / f"{slug}_16bit.tif", (np.clip(img, 0, 1) * 65535 + .5).astype(np.uint16), photometric="rgb")
    return p


def rms(a: np.ndarray, b: np.ndarray, valid: np.ndarray, margin: int = 12) -> float:
    ok = cv2.erode(valid.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)).astype(bool)
    return float(np.sqrt(((a - b) ** 2)[ok].mean()))


def truth() -> np.ndarray:
    px, py = np.meshgrid(np.arange(REF[0]) + .5 + REF0[0], np.arange(REF[1]) + .5 + REF0[1])
    return scene(px, py)


@pytest.fixture(scope="module")
def pair_roots(tmp_path_factory):
    """An old and a new project of the same sky, their grids shifted by a few pixels; the new one has every workflow."""
    t = tmp_path_factory.mktemp("pair")
    imgs = {"Classic stack": B.image(), "Full": B.image(), "Full + neural stars": B.image(),
            "Full + earlier neural layer": B.image()}
    old = make_v1(t / "old" / "Heart", A, "Heart", images={"faithful": ("render_prep_learned", True),
                                                          "artistic": ("render_prep_learned", True)})
    new = make_v2(t / "new" / "Heart", B, exports=imgs, name="Heart")
    return t, old, new.root


# ---------------------------------------------------------------------------------------------- addresses

def test_addresses_v1_v2_and_windows_paths():
    a = qc_addr.parse_address("F:/astro_projets/Horsehead::Full + earlier neural layer")
    assert a.project == Path("F:/astro_projets/Horsehead") and a.name == "Full + earlier neural layer" and a.image is None
    a = qc_addr.parse_address(r"E:\astro_gpu\projects\Horsehead::faithful=F:\x\y.tif")
    assert a.name == "faithful" and a.image == Path(r"F:\x\y.tif")
    assert qc_addr.parse_address("P::#3").name == "#3"
    for bad in ("F:/astro_projets/Horsehead", "::Full", "F:/p::"):
        with pytest.raises(qc_addr.QCError):
            qc_addr.parse_address(bad)


def test_kind_of_tells_v1_from_v2(tmp_path):
    make_v1(tmp_path / "o", A)
    make_v2(tmp_path / "n", A, ["Full"])
    assert qc_addr.kind_of(tmp_path / "o") == 1 and qc_addr.kind_of(tmp_path / "n") == 2
    with pytest.raises(qc_addr.QCError):
        qc_addr.kind_of(tmp_path)


def test_a_v1_deliverable_resolves_to_the_preparation_it_was_rendered_from(tmp_path):
    root = make_v1(tmp_path / "r", A, "Rosette", images={"faithful": ("render_prep_learned", True),
                                                          "artistic": ("render_prep", False), "render": (None, None)})
    f, a, r = (qc_addr.resolve(f"{root}::{n}") for n in ("faithful", "artistic", "render"))
    assert f.stage == "render_prep_learned" and f.out_dir == root / "work" / "render_prep_learned"
    assert a.stage == "render_prep" and a.out_dir == root / "work" / "render_prep"
    assert r.stage == "render_prep" and r.notes and "not recorded" in r.notes[0]       # older file: taken as render_prep
    assert f.image == root / "results" / "faithful" / "faithful.png"
    assert f.align == root / "work" / "reference" / "align_ref.json" and f.wcs and f.stars and f.reference
    with pytest.raises(qc_addr.QCError, match="no deliverable"):
        qc_addr.resolve(f"{root}::nothing")


def test_a_v2_workflow_resolves_by_name_id_or_case(pair_roots):
    _, _, new = pair_roots
    full = qc_addr.resolve(f"{new}::Full")
    assert full.stage == "render_prep" and full.workflow.name == "Full" and full.version == 2
    assert full.image == new / "export" / "Full" / "Full_16bit.tif" and full.align and full.wcs and full.stars
    assert qc_addr.resolve(f"{new}::#{full.workflow.id}").workflow.id == full.workflow.id
    assert qc_addr.resolve(f"{new}::full").workflow.name == "Full"
    assert qc_addr.resolve(f"{new}::Classic stack").stage == "render_simple"
    assert qc_addr.resolve(f"{new}::Full + neural stars").stage == "render_prep_learned"
    assert qc_addr.resolve(f"{new}::Full + earlier neural layer").stage == "render_prep_learned"
    with pytest.raises(qc_addr.QCError, match="no workflow"):
        qc_addr.resolve(f"{new}::Nope")


def test_a_workflow_that_is_not_computed_is_refused_with_the_reason(tmp_path):
    make_v2(tmp_path / "n", A, ["Full"])
    p = p2.load(tmp_path / "n")
    p.add_workflow("Classic stack", dict(wfl.TEMPLATES["Classic stack"]))
    p2.save(p)
    with pytest.raises(qc_addr.QCError, match="not computed"):
        qc_addr.resolve(f"{tmp_path / 'n'}::Classic stack")


def test_the_render_of_a_workflow_without_a_reference_has_no_alignment(tmp_path):
    make_v2(tmp_path / "n", A, ["Full"], with_ref=False)
    s = qc_addr.resolve(f"{tmp_path / 'n'}::Full")
    assert s.align is None and s.wcs is not None and s.reference is None


# ---------------------------------------------------------------------------------------------- pairing

def v1_with(tmp_path, params, images, name="P"):
    return make_v1(tmp_path / name, A, name, params=params, images=images)


def pairing(root, new=None):
    return {p.old.name: p for p in qc_pairs.pair_project(root, new)}


LAYER_PARAMS = {"star_net": {"enabled": False}, "starless": {"learned_stars": "D:/diag/layer.npy"}}


def test_a_learned_preparation_image_goes_with_a_learned_workflow_never_with_full(pair_roots, tmp_path):
    _, _, new = pair_roots
    heart = pairing(v1_with(tmp_path, {}, {"artistic": ("render_prep_learned", True)}, "Heart"), new)
    assert heart["artistic"].workflow.name == "Full + neural stars"            # it trained its own layer
    horse = pairing(v1_with(tmp_path, LAYER_PARAMS, {"faithful": ("render_prep_learned", True)}, "Horsehead"), new)
    assert horse["faithful"].workflow.name == "Full + earlier neural layer"    # it used an earlier layer
    for pr in (*heart.values(), *horse.values()):
        assert pr.status == "paired" and pr.workflow.name != "Full" and pr.stage == "render_prep_learned"
        assert qc_pairs.output_stage(pr.workflow.options) == "render_prep_learned"


def test_rosette_has_both_preparations_in_one_project(pair_roots, tmp_path):
    _, _, new = pair_roots
    ros = pairing(v1_with(tmp_path, LAYER_PARAMS, {"artistic": ("render_prep", False),
                                                    "faithful": ("render_prep_learned", True)}, "Rosette"), new)
    assert ros["artistic"].workflow.name == "Full" and ros["artistic"].old.stage == "render_prep"
    assert ros["faithful"].workflow.name == "Full + earlier neural layer"


def test_an_image_without_a_recorded_preparation_goes_with_full_and_says_so(pair_roots, tmp_path):
    _, _, new = pair_roots
    pr = pairing(v1_with(tmp_path, {}, {"render2": (None, None)}, "Old"), new)["render2"]
    assert pr.workflow.name == "Full" and "inferred" in pr.row()["preparation"] and "older" in pr.reason


def test_the_table_without_a_v2_project_gives_the_expected_workflow(tmp_path):
    root = v1_with(tmp_path, LAYER_PARAMS, {"artistic": ("render_prep", False), "faithful": ("render_prep_learned", True)})
    rows = {r["image"]: r for r in (p.row() for p in qc_pairs.pair_project(root))}
    assert rows["artistic"]["v2_expected"] == "Full" and rows["faithful"]["v2_expected"] == "Full + earlier neural layer"
    assert all(r["status"] == "expected" and r["v2_workflow"] is None for r in rows.values())
    text = qc_pairs.table(list(rows.values()))
    assert "Full + earlier neural layer" in text and "render_prep_learned" in text


def test_pairing_falls_back_or_reports_what_is_missing(tmp_path):
    root = v1_with(tmp_path, LAYER_PARAMS, {"faithful": ("render_prep_learned", True), "artistic": ("render_prep", False)})
    only_own = make_v2(tmp_path / "n1", A, ["Classic stack", "Full + neural stars"])
    pr = pairing(root, only_own)
    assert pr["faithful"].workflow.name == "Full + neural stars" and "earlier neural layer" in pr["faithful"].reason
    assert pr["artistic"].status == "missing" and "no workflow ending in render_prep" in pr["artistic"].reason
    # two workflows that fit equally: the tool does not choose; the plain name breaks the tie
    two = make_v2(tmp_path / "n2", A, ["Classic stack"])
    for n in ("Full A", "Full B"):
        two.add_workflow(n, dict(wfl.TEMPLATES["Full"]))
    p2.save(two)
    amb = pairing(root, p2.load(two.root))["artistic"]
    assert amb.status == "ambiguous" and "'Full A'" in amb.reason and "'Full B'" in amb.reason
    two.add_workflow("Full", dict(wfl.TEMPLATES["Full"]))
    p2.save(two)
    assert pairing(root, p2.load(two.root))["artistic"].workflow.name == "Full"


def test_classic_against_full_pairs_inside_a_project(pair_roots):
    _, _, new = pair_roots
    pairs = qc_pairs.within_project(p2.load(new))
    assert [(c.name, f.name) for c, f in pairs] == [("Classic stack", "Full")]


# ---------------------------------------------------------------------------------------------- framing

def render_of(src: qc_addr.Source) -> qc_frame.Render:
    return qc_frame.load_render(src)


def test_each_render_returns_to_the_sky_through_its_own_projects_alignment(tmp_path):
    a = qc_addr.resolve(f"{make_v1(tmp_path / 'a', A)}::faithful")
    b = qc_addr.resolve(f"{make_v2(tmp_path / 'b', B, ['Full + neural stars'], exports={'Full + neural stars': B.image()}).root}"
                        "::Full + neural stars")
    view = qc_frame.View([render_of(a), render_of(b)], "auto")
    assert view.kind == "reference" and (view.W, view.H) == REF
    T = truth()
    (ia, va), (ib, vb) = view.remap(0), view.remap(1)
    own = (rms(ia, T, va), rms(ib, T, vb))
    assert max(own) < .01, own
    # the old way: the new render mapped through the old project's alignment and geometry is shifted by a few pixels
    wrong = dataclasses.replace(b, align=a.align, prep=a.prep)
    wv = qc_frame.View([render_of(a), render_of(wrong)], "reference")
    iw, vw = wv.remap(1)
    assert rms(iw, T, vw) > 4 * max(own) and rms(iw, T, vw) > .02


def test_without_a_reference_the_sky_brings_renders_together_by_their_own_solutions(tmp_path):
    a = qc_addr.resolve(f"{make_v1(tmp_path / 'a', A, with_ref=False)}::faithful")
    new = make_v2(tmp_path / "b", B, ["Full + neural stars"], with_ref=False, exports={"Full + neural stars": B.image()})
    b = qc_addr.resolve(f"{new.root}::Full + neural stars")
    assert a.align is None and b.align is None
    view = qc_frame.View([render_of(a), render_of(b)], "auto")
    assert view.kind == "sky"
    (ia, va), (ib, vb) = view.remap(0), view.remap(1)
    own = rms(ia, ib, va & vb)
    assert own < .01, own
    # with the other project's solution (the same file for both) the new render is not moved
    wrong = dataclasses.replace(b, wcs=a.wcs, prep=a.prep)
    wv = qc_frame.View([render_of(a), render_of(wrong)], "sky")
    iw, vw = wv.remap(1)
    assert rms(ia, iw, va & vw) > 50 * own and rms(ia, iw, va & vw) > .01


def test_two_renders_of_one_stack_meet_without_any_resampling(tmp_path):
    p = make_v2(tmp_path / "n", A, ["Full", "Full + neural stars"], with_ref=False,
                exports={"Full": A.image(), "Full + neural stars": A.image()})
    r = [render_of(qc_addr.resolve(f"{p.root}::{n}")) for n in ("Full", "Full + neural stars")]
    view = qc_frame.View(r, "auto")
    for k in (0, 1):
        img, _ = view.remap(k)
        assert np.array_equal(img, r[k].img)                      # bit for bit: the pixels as they were


def test_stars_of_a_project_land_where_the_sky_puts_them_in_the_view(tmp_path):
    a = qc_addr.resolve(f"{make_v1(tmp_path / 'a', A)}::faithful")
    b = qc_addr.resolve(f"{make_v2(tmp_path / 'b', B, ['Full'], exports={'Full': B.image()}).root}::Full")
    sa, sb = np.load(a.stars), np.load(b.stars)
    for mode in ("reference", "sky"):
        view = qc_frame.View([render_of(a), render_of(b)], mode)
        ua = view.view_of_grid(0, sa["x"], sa["y"])
        ub = view.view_of_grid(1, sb["x"], sb["y"])
        assert np.abs(np.array(ua) - np.array(ub)).max() < .05, mode          # the same stars, the same places


def test_a_render_of_the_wrong_proportions_or_turned_is_refused_and_a_turn_is_undone(tmp_path):
    root = make_v1(tmp_path / "a", A)
    img = A.image()
    f = png16(tmp_path / "turned.png", np.rot90(img, 1, axes=(0, 1)))
    with pytest.raises(qc_addr.QCError, match="not the render of this workflow|turned"):
        qc_frame.load_render(qc_addr.resolve(f"{root}::faithful", image=f))
    s = qc_addr.resolve(f"{root}::faithful", image=f, rot90=1, flip=False)        # exported turned once -> undone
    assert np.abs(qc_frame.load_render(s).img - img).max() < 2e-5
    h = png16(tmp_path / "turned3.png", np.rot90(img[::-1], 3, axes=(0, 1)))
    s = qc_addr.resolve(f"{root}::faithful", image=h, rot90=3, flip=True)          # mirrored, then turned three times
    assert np.abs(qc_frame.load_render(s).img - img).max() < 2e-5
    g = png16(tmp_path / "flipped.png", img[::-1])
    s = qc_addr.resolve(f"{root}::faithful", image=g, flip=True)
    assert np.abs(qc_frame.load_render(s).img - img).max() < 2e-5
    cv2.imwrite(str(tmp_path / "square.png"), np.zeros((300, 300, 3), np.uint8))
    with pytest.raises(qc_addr.QCError, match="useful frame"):
        qc_frame.load_render(qc_addr.resolve(f"{root}::faithful", image=tmp_path / "square.png"))


def test_a_large_render_is_reduced_first_so_that_all_are_sampled_alike(tmp_path):
    a = qc_addr.resolve(f"{make_v1(tmp_path / 'a', A)}::faithful")
    big = png16(tmp_path / "big.png", B.image(scale=2.0))                      # twice the sensor scale
    b = qc_addr.resolve(f"{make_v2(tmp_path / 'b', B, ['Full'], exports={'Full': B.image()}).root}::Full", image=big)
    view = qc_frame.View([render_of(a), render_of(b)], "reference")
    ib, vb = view.remap(1)
    assert rms(ib, truth(), vb) < .01


def test_the_polynomial_inverse_finds_the_reference_pixel_of_a_grid_pixel():
    info = A.align(Path("x.png"))
    # make it truly cubic so the Newton iteration is exercised
    info["cx"][7] = 3.0
    info["cy"][5] = -2.0
    u, v = np.array([10.0, 300.0, 590.0]), np.array([20.0, 200.0, 390.0])
    x, y = qc_frame.poly_points(info, u, v)
    ru, rv = qc_frame.poly_inverse(info, x, y)
    assert np.abs(ru - u).max() < 1e-6 and np.abs(rv - v).max() < 1e-6


# ---------------------------------------------------------------------------------------------- output folder

def test_the_output_folder_is_refused_inside_a_project_or_an_image_folder(tmp_path):
    v1 = make_v1(tmp_path / "old", A)
    v2 = make_v2(tmp_path / "new", A, ["Full"], exports={"Full": A.image()}).root
    img = tmp_path / "cands" / "x.tif"
    img.parent.mkdir()
    img.write_bytes(b"")
    for bad in (v1, v1 / "results", v1 / "results" / "qc", v2 / "export" / "qc", v2, img.parent, img.parent / "qc"):
        with pytest.raises(qc_out.OutputRefused):
            qc_out.check_out(bad, [v1, v2, img])
    assert qc_out.check_out(tmp_path / "_qc", [v1, v2, img]) == tmp_path / "_qc"


def test_nothing_can_be_written_under_a_project_while_a_run_is_guarded(tmp_path):
    v1 = make_v1(tmp_path / "old", A)
    with qc_out.protect(tmp_path / "_qc", [v1]) as out:
        (out / "ok.txt").write_text("fine")
    assert (tmp_path / "_qc" / "ok.txt").exists()
    for attempt in (lambda: (v1 / "results" / "faithful" / "extra.txt").write_text("no"),
                    lambda: (v1 / "results" / "new_dir").mkdir(),
                    lambda: (v1 / "results" / "faithful" / "faithful.png").unlink()):
        with pytest.raises(iv.WriteRefused), qc_out.protect(tmp_path / "_qc", [v1]):
            attempt()
    assert not (v1 / "results" / "faithful" / "extra.txt").exists() and not (v1 / "results" / "new_dir").exists()
    assert (v1 / "results" / "faithful" / "faithful.png").exists()
    # the guard also fails a write whose error the code under it swallows
    with pytest.raises(iv.WriteRefused), qc_out.protect(tmp_path / "_qc", [v1]):
        try:
            (v1 / "results" / "faithful" / "swallowed.txt").write_text("no")
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------- blind sheets

def two(pair_roots):
    t, old, new = pair_roots
    return [f"{old}::faithful", f"{new}::Full + neural stars"]


def test_letters_are_random_and_the_same_for_a_seed():
    orders = {tuple(qc_sheets.letter_order(2, s)) for s in range(64)}
    assert orders == {(0, 1), (1, 0)}
    assert qc_sheets.letter_order(5, 7) == qc_sheets.letter_order(5, 7)
    assert sorted(qc_sheets.letter_order(5, 7)) == list(range(5))


def test_a_sheet_name_that_tells_which_is_which_is_refused():
    for bad in ("heart_old_new", "full_01", "v2-heart", "classic", "Before"):
        with pytest.raises(qc_addr.QCError):
            qc_sheets.check_short(bad)
    for good in ("heart_01", "Horsehead_faithful", "gold_1"):
        assert qc_sheets.check_short(good) == good


def test_blind_sheets_hide_which_render_is_which_and_keep_the_key_apart(pair_roots, tmp_path):
    out = tmp_path / "_qc"
    r = qc2.run_sheets(out, "heart_01", two(pair_roots), seed=5, zones=["0.5:0.5"])
    sheets, keys = out / "sheets", out / "_keys"
    assert r["letters"] == ["A", "B"] and (keys / "heart_01_key.json").is_file()
    assert not list(sheets.glob("*key*")) and not list(keys.glob("*.jpg"))
    names = sorted(p.name for p in sheets.iterdir())
    for want in ("heart_01_overview.jpg", "heart_01_zooms.jpg", "heart_01_check100.jpg", "heart_01_prompt.md"):
        assert want in names, names
    assert any(n.startswith("heart_01_bright_p") for n in names) and any(n.startswith("heart_01_zones_p") for n in names)
    _, old, new = pair_roots
    for p in sheets.iterdir():
        blob = p.read_bytes()
        assert not any(s.encode() in blob for s in (str(old), str(new), "faithful", "neural stars", "astroproj")), p.name
    prompt = (sheets / "heart_01_prompt.md").read_text(encoding="utf-8")
    assert "reference, A, B" in prompt and "faithful" not in prompt and "neural" not in prompt and str(old) not in prompt
    key = json.loads((keys / "heart_01_key.json").read_text(encoding="utf-8"))
    assert key["seed"] == 5 and key["frame"]["frame"] == "reference"
    got = {L: v["address"] for L, v in key["letters"].items()}
    assert sorted(got.values()) == sorted(f"{Path(a.split('::')[0]).as_posix()}::{a.split('::')[1]}" for a in two(pair_roots))


def test_the_old_render_is_not_always_a(pair_roots, tmp_path):
    seeds = [next(s for s in range(64) if qc_sheets.letter_order(2, s) == o) for o in ([0, 1], [1, 0])]
    first = []
    for seed in seeds:
        r = qc2.run_sheets(tmp_path / f"o{seed}", "h_01", two(pair_roots), seed=seed)
        key = json.loads(Path(r["key"]).read_text(encoding="utf-8"))
        first.append(key["letters"]["A"]["version"])
    assert sorted(first) == [1, 2]                                               # A is the old one once, the new one once


def test_sheets_refuse_to_replace_a_set_without_its_key(pair_roots, tmp_path):
    out = tmp_path / "_qc"
    qc2.run_sheets(out, "h_01", two(pair_roots), seed=1)
    with pytest.raises(qc_addr.QCError, match="exist"):
        qc2.run_sheets(out, "h_01", two(pair_roots), seed=2)
    r = qc2.run_sheets(out, "h_01", two(pair_roots), seed=2, overwrite=True)
    assert json.loads(Path(r["key"]).read_text(encoding="utf-8"))["seed"] == 2


def test_sheets_in_the_sky_frame_when_there_is_no_reference(tmp_path):
    a = make_v1(tmp_path / "a", A, with_ref=False)
    b = make_v2(tmp_path / "b", B, ["Full"], with_ref=False, exports={"Full": B.image()}).root
    r = qc2.run_sheets(tmp_path / "_qc", "set_01", [f"{a}::faithful", f"{b}::Full"], seed=3)
    assert r["frame"]["frame"] == "sky"
    assert (tmp_path / "_qc" / "sheets" / "set_01_overview.jpg").is_file()


def test_sheets_compare_workflows_of_one_project(pair_roots, tmp_path):
    _, _, new = pair_roots
    r = qc2.run_sheets(tmp_path / "_qc", "cf_01", [f"{new}::Classic stack", f"{new}::Full"], seed=4)
    assert r["letters"] == ["A", "B"]
    key = json.loads(Path(r["key"]).read_text(encoding="utf-8"))
    assert {v["stage"] for v in key["letters"].values()} == {"render_simple", "render_prep"}


def test_sheets_of_a_v1_project_against_its_v2_import_come_out_with_the_right_pairs(pair_roots, tmp_path):
    _, old, new = pair_roots
    out = tmp_path / "_qc"
    res = qc2.run_pair_sheets(out, old, new, seed=11)
    assert set(res) == {"faithful", "artistic"}
    for name, r in res.items():
        key = json.loads(Path(r["key"]).read_text(encoding="utf-8"))
        who = {v["version"]: v for v in key["letters"].values()}
        assert who[1]["name"] == name and who[1]["stage"] == "render_prep_learned"
        assert who[2]["name"] == "Full + neural stars" and who[2]["stage"] == "render_prep_learned"
        assert key["pairing"]["v2_workflow"] == "Full + neural stars" and key["pairing"]["status"] == "paired"
        assert r["short"] == {"faithful": "Heart_01", "artistic": "Heart_02"}[name]
        assert (out / "sheets" / f"{r['short']}_overview.jpg").is_file()


def test_pair_sheets_find_candidates_by_name_and_say_what_cannot_be_compared(tmp_path):
    old = make_v1(tmp_path / "old", A, "Rosette", params=LAYER_PARAMS,
                  images={"faithful": ("render_prep_learned", True), "artistic": ("render_prep", False)})
    new = make_v2(tmp_path / "new", B, ["Full + earlier neural layer", "Classic stack"]).root
    cand = tmp_path / "cands"
    png16(cand / "Full_earlier_neural_layer__faithful.png", B.image())
    res = qc2.run_pair_sheets(tmp_path / "_qc", old, new, candidates=cand, seed=1)
    assert res["faithful"]["pairing"]["v2_workflow"] == "Full + earlier neural layer"
    assert res["artistic"]["status"] == "missing" and "render_prep" in res["artistic"]["reason"]
    # a paired workflow with no candidate and no device: said, not guessed
    bare = qc2.run_pair_sheets(tmp_path / "_qc2", old, new, seed=1)
    assert bare["faithful"]["status"] == "no image" and "--new-image faithful=PATH" in bare["faithful"]["reason"]


# ---------------------------------------------------------------------------------------------- report cards

def test_bulletin_reads_v1_and_v2_stage_metrics_and_writes_sheets_only_to_the_output(pair_roots, tmp_path):
    _, old, new = pair_roots
    out = tmp_path / "_qc"
    r = qc2.run_bulletin(out, [f"{old}::faithful", f"{new}::Full"], sheet=True)
    v1, v2 = r["Heart__faithful"], r["Heart__Full"]
    assert v1["lights"] == 40 and v1["stacked"] == 38 and v1["flat_error_pct"] == 0.4 and v1["pure_black_pct"] == 0.0
    assert v2["lights"] == 40 and v2["stacked"] == 39 and v2["stages_not_done"] == []
    for c in (v1, v2):
        assert "sky_floor_255" in c and c["image_px_per_sensor_px"] == pytest.approx(1.0, abs=.01)
        assert Path(c["sheet"]).parent == out / "bulletin" and Path(c["sheet"]).is_file()
    assert (out / "bulletin" / "bulletin.json").is_file()
    assert not list((old / "results").glob("*/*planche*"))                           # bulletin.py --sheet's habit


def test_bulletin_lists_the_stages_of_a_workflow_that_are_not_done(tmp_path):
    p = make_v2(tmp_path / "n", A, ["Full"], exports={"Full": A.image()})
    k = Keys(p)
    d = tmp_path / "n" / "store" / "color" / k.key(p.workflows[0], "color")
    (d / "manifest.json").write_text(json.dumps({"status": "failed", "key": k.key(p.workflows[0], "color")}))
    c = qc_report.card(qc_addr.resolve(f"{p.root}::Full"))
    assert c["stages_not_done"] == ["color"]


# ---------------------------------------------------------------------------------------------- non-regression

def test_two_identical_renders_are_identical_and_a_change_is_found_where_it_is(tmp_path):
    p = make_v2(tmp_path / "n", A, ["Full", "Full + neural stars"], with_ref=False)
    img = A.image()
    changed = img.copy()
    changed[150:220, 400:470] += .12                                 # a patch: ~30 levels
    for n, i in (("Full", img), ("Full + neural stars", changed)):
        import tifffile
        d = tmp_path / "n" / "export" / qc_out.slug(n)
        d.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(d / f"{qc_out.slug(n)}_16bit.tif", (np.clip(i, 0, 1) * 65535 + .5).astype(np.uint16), photometric="rgb")
    a, b = (f"{p.root}::{n}" for n in ("Full", "Full + neural stars"))
    r = qc2.run_nonreg(tmp_path / "_qc", {"same": (a, a), "patch": (a, b)})
    assert r["same"]["identical"] and r["same"]["max_levels"] == 0 and not r["same"]["differs"]
    q = r["patch"]
    assert q["differs"] and q["max_levels"] > 25 and q["share_over_2_levels"] > 0
    x, y = (float(v) for v in q["zones"][0].split(":"))
    assert abs(x - 435 / 640) < .08 and abs(y - 185 / 444) < .1                    # the patch, as fractions of the image
    assert (tmp_path / "_qc" / "nonreg" / "nonreg.json").is_file() and Path(q["map"]).is_file()


def test_nonreg_across_projects_uses_each_projects_files(pair_roots, tmp_path):
    _, old, new = pair_roots
    r = qc2.run_nonreg(tmp_path / "_qc", {"heart": (f"{old}::faithful", f"{new}::Full + neural stars")})
    assert r["heart"]["frame"]["frame"] == "reference" and r["heart"]["p999_levels"] < 3
    a = qc_addr.resolve(f"{old}::faithful")
    wrong = dataclasses.replace(qc_addr.resolve(f"{new}::Full + neural stars"), align=a.align, prep=a.prep)
    bad = qc_report.compare_renders(a, wrong, tmp_path / "_qc2", "bad")
    assert bad["p999_levels"] > 5 * r["heart"]["p999_levels"]


def test_nonreg_makes_blind_sheets_for_the_cases_that_differ(tmp_path):
    p = make_v2(tmp_path / "n", A, ["Full", "Full + neural stars"], with_ref=False)
    import tifffile
    for n, i in (("Full", A.image()), ("Full + neural stars", np.clip(A.image() * 1.2, 0, 1))):
        d = tmp_path / "n" / "export" / qc_out.slug(n)
        d.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(d / f"{qc_out.slug(n)}_16bit.tif", (i * 65535 + .5).astype(np.uint16), photometric="rgb")
    r = qc2.run_nonreg(tmp_path / "_qc", {"heart": (f"{p.root}::Full", f"{p.root}::Full + neural stars")}, sheets=True, seed=2)
    assert r["heart"]["differs"] and r["heart"]["sheets"]["letters"] == ["A", "B"]
    assert (tmp_path / "_qc" / "sheets" / "nr_heart_overview.jpg").is_file()


# ---------------------------------------------------------------------------------------------- nothing next to the inputs

def tree(*roots):
    return iv.snapshot(roots)


def test_no_tool_writes_next_to_its_inputs(pair_roots, tmp_path):
    t, old, new = pair_roots
    before = tree(old, new)
    out = tmp_path / "_qc"
    qc2.pairs(old, new, out)
    qc2.run_sheets(out, "h_01", two(pair_roots), seed=1, zones=["0.4:0.4"])
    qc2.run_pair_sheets(out, old, new, seed=1)
    qc2.run_bulletin(out, two(pair_roots), sheet=True)
    qc2.run_nonreg(out, {"c": tuple(two(pair_roots))}, sheets=True)
    assert iv.snapshot_changes(before, tree(old, new)) == []
    assert not list(old.rglob("*planche*")) and not (old / "qc").exists()
    produced = {p.relative_to(out).parts[0] for p in out.rglob("*") if p.is_file()}
    assert {"sheets", "_keys", "bulletin", "nonreg", "pairs"} <= produced


def test_a_run_aimed_at_a_project_is_refused_before_anything_is_written(pair_roots):
    _, old, new = pair_roots
    before = tree(old, new)
    for bad in (old / "results" / "qc", new / "export" / "qc", old):
        with pytest.raises(qc_out.OutputRefused):
            qc2.run_sheets(bad, "h_01", two(pair_roots), seed=1)
        with pytest.raises(qc_out.OutputRefused):
            qc2.run_bulletin(bad, two(pair_roots), sheet=True)
        with pytest.raises(qc_out.OutputRefused):
            qc2.run_nonreg(bad, {"c": tuple(two(pair_roots))})
    assert iv.snapshot_changes(before, tree(old, new)) == []


# ---------------------------------------------------------------------------------------------- rendering a v2 workflow, CLI

def toy_simple_layer(d: Path, H: int = 300, W: int = 400) -> None:
    from astro_gpu.render.simple import image_stats
    rng = np.random.default_rng(4)
    neb = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 20) * 400
    lin = np.stack([neb, .5 * neb, .3 * neb]) - 150 + rng.normal(0, 3, (3, H, W)).astype(np.float32)
    valid = np.ones((H, W), bool)
    prep = {"M": np.eye(3).tolist(), "crop": [4, W - 4, 4, H - 4], "stats": image_stats(lin, valid), "linked": True,
            "stretch": "auto", "denoised": False, "color": True, "exit": None, "drizzle_scale": 2.0, "width": W, "height": H}
    np.save(d / "lin.npy", lin.astype(np.float32))
    np.save(d / "valid.npy", valid)
    (d / "prep_simple.json").write_text(json.dumps(prep))


def test_a_workflow_without_an_image_is_rendered_into_the_output_folder(tmp_path):
    p = make_v2(tmp_path / "n", A, ["Classic stack"], with_ref=False)
    s = qc_addr.resolve(f"{p.root}::Classic stack")
    toy_simple_layer(s.out_dir)
    s = qc_addr.resolve(f"{p.root}::Classic stack")
    with pytest.raises(qc_addr.QCError, match="no image"):
        qc2.run_bulletin(tmp_path / "_qc", [f"{p.root}::Classic stack"])
    r = qc2.run_bulletin(tmp_path / "_qc", [f"{p.root}::Classic stack"], device="cpu")
    png = next((tmp_path / "_qc" / "renders").glob("*.png"))
    im = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    assert im.dtype == np.uint16 and im.shape[:2] == (146, 196)                      # the crop (392 x 292) / 2
    assert (png.with_name(png.stem + "_history.json")).is_file() and list(r.values())[0]["sky_floor_255"] >= 0
    assert qc_addr.orientation_of(png) == (0, False) and not list((p.root / "export").glob("*"))
    assert s.image is None


def test_the_command_line(pair_roots, tmp_path, capsys):
    _, old, new = pair_roots
    assert qc2.main(["pairs", str(old), "--new", str(new)]) == 0
    text = capsys.readouterr().out
    assert "faithful" in text and "Full + neural stars" in text and "also: Classic stack  against  Full" in text
    out = tmp_path / "_qc"
    assert qc2.main(["sheets", "--out", str(out), "--short", "cli_01", "--seed", "3", *two(pair_roots)]) == 0
    assert (out / "sheets" / "cli_01_overview.jpg").is_file()
    assert qc2.main(["sheets", "--out", str(old / "results" / "qc"), "--short", "cli_02", *two(pair_roots)]) == 2
    assert "do not write under what they read" in capsys.readouterr().err
    assert qc2.main(["sheets", "--out", str(out), "--short", "old_new", *two(pair_roots)]) == 2
    assert qc2.main(["sheets", "--out", str(out), "--short", "x_01", f"{new}::Nope"]) == 2


def test_the_render_module_refuses_a_v1_deliverable(tmp_path):
    s = qc_addr.resolve(f"{make_v1(tmp_path / 'a', A)}::faithful")
    with pytest.raises(qc_addr.QCError, match="only a v2 workflow"):
        qc_render.render_v2(s, tmp_path / "_qc")


# ---------------------------------------------------------------------------------------------- pairing table of many projects

def test_the_table_of_several_old_projects_is_written_into_the_output_folder_only(tmp_path):
    heart = make_v1(tmp_path / "in" / "Heart", A, "Heart", images={"faithful": ("render_prep_learned", True)})
    horse = make_v1(tmp_path / "in" / "Horsehead", A, "Horsehead", params=LAYER_PARAMS,
                    images={"faithful": ("render_prep_learned", True), "artistic": ("render_prep", False)})
    bare = tmp_path / "in" / "M31"
    bare.mkdir()
    (bare / "project.json").write_text(json.dumps({"name": "M31", "params": {}}), encoding="utf-8")
    before = tree(heart, horse, bare)
    out = tmp_path / "_qc"
    r = qc2.pairs_all([heart, horse, bare], out)
    got = {(x["project"], x["image"]): x["v2_expected"] for x in r["rows"]}
    assert got == {("Heart", "faithful"): "Full + neural stars", ("Horsehead", "faithful"): "Full + earlier neural layer",
                   ("Horsehead", "artistic"): "Full"}
    assert r["no_image"] == [str(bare)]
    assert (out / "pairs" / "pairing_table.txt").read_text(encoding="utf-8").startswith("project")
    assert json.loads((out / "pairs" / "pairing_table.json").read_text(encoding="utf-8"))["rows"] == r["rows"]
    assert iv.snapshot_changes(before, tree(heart, horse, bare)) == []
    with pytest.raises(qc_out.OutputRefused):
        qc2.pairs_all([heart, horse], horse / "results" / "qc")
    assert not (horse / "results" / "qc").exists()


def test_the_command_line_pairs_several_projects_and_refuses_new_with_many(tmp_path, capsys):
    a = make_v1(tmp_path / "a", A, "Heart")
    b = make_v1(tmp_path / "b", A, "Iris", params=LAYER_PARAMS)
    assert qc2.main(["pairs", str(a), str(b), "--out", str(tmp_path / "_qc")]) == 0
    text = capsys.readouterr().out
    assert "Heart" in text and "Iris" in text and "(Full + earlier neural layer)" in text
    assert qc2.main(["pairs", str(a), str(b), "--new", str(a)]) == 2


# ---------------------------------------------------------------------------------------------- the real projects (read only)

REAL_OLD = {
    "Heart": "F:/photos/heart_nebula/astro_gpu", "Iris": "C:/Users/Nicolas/astro_gpu_scratch/projects/Iris",
    "Horsehead": "E:/astro_gpu/projects/Horsehead", "Rosette": "E:/astro_gpu/projects/Rosette",
}


@pytest.mark.skipif(not all((Path(p) / "project.json").is_file() for p in REAL_OLD.values()),
                    reason="the user's v1 projects are not on this machine")
def test_the_real_projects_old_images_are_paired_by_their_own_settings_files(tmp_path):
    """Heart's, Horsehead's and Iris's images and Rosette's faithful one come from render_prep_learned (their settings
    files say so): none goes with "Full". Read only: the project folders are snapshotted before and after."""
    roots = [Path(p) for p in REAL_OLD.values()]
    before = tree(*roots)
    r = qc2.pairs_all(roots, tmp_path / "_qc")
    want = {("HeartSoul", "faithful"): "Full + neural stars", ("HeartSoul", "artistic"): "Full + neural stars",
            ("Iris", "faithful"): "Full + earlier neural layer", ("Iris", "artistic"): "Full + earlier neural layer",
            ("Horsehead", "faithful"): "Full + earlier neural layer", ("Horsehead", "artistic"): "Full + earlier neural layer",
            ("Rosette", "faithful"): "Full + earlier neural layer", ("Rosette", "artistic"): "Full"}
    got = {(x["project"], x["image"]): x["v2_expected"] for x in r["rows"]}
    assert {k: got[k] for k in want} == want
    assert all(x["preparation"].startswith("render_prep_learned") for x in r["rows"]
               if x["v2_expected"] != "Full")
    assert iv.snapshot_changes(before, tree(*roots)) == []


def test_two_old_images_paired_with_one_workflow_get_two_renders(tmp_path, monkeypatch):
    """Heart's faithful and artistic go with the same workflow: each is rendered with its own settings, into its own file
    (the second must not replace the first, whose key points at it)."""
    old = make_v1(tmp_path / "old" / "Heart", A, "Heart", images={"faithful": ("render_prep_learned", True),
                                                                  "artistic": ("render_prep_learned", True)})
    new = make_v2(tmp_path / "new" / "Heart", B, ["Full + neural stars"], name="Heart").root     # no export yet
    calls = []

    def fake_render(src, out, base=None, sliders=None, device="cpu", service=None, name=None):
        calls.append((name, base))
        png = png16(Path(out) / "renders" / f"{name}.png", B.image())
        src.image, src.rot90, src.flip = png, 0, False
        return png

    import astro_gpu.render.layers as layers
    monkeypatch.setattr(qc_render, "render_v2", fake_render)
    monkeypatch.setattr(layers, "LayerService", lambda device: object())
    out = tmp_path / "_qc"
    res = qc2.run_pair_sheets(out, old, new, device="cpu", seed=2)
    assert [n for n, _ in calls] == ["Heart__Full_neural_stars__faithful", "Heart__Full_neural_stars__artistic"]
    assert all(b == {"chroma": .9, "learned_stars": True} for _, b in calls)           # the old image's own settings
    assert len(list((out / "renders").glob("*.png"))) == 2 and set(res) == {"faithful", "artistic"}
    for name, r in res.items():
        key = json.loads(Path(r["key"]).read_text(encoding="utf-8"))
        imgs = {v["version"]: v["image"] for v in key["letters"].values()}
        assert imgs[2].endswith(f"Heart__Full_neural_stars__{name}.png")
