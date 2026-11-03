"""Tables of the bench report (docs/pilot/BENCH.md) from the measures, and the sheets copied for it.

    python tools/bench/report.py LEVEL [CASE ...]
    python tools/bench/report.py fill            fill the tables of docs/pilot/BENCH.md (between the markers
                                                 <!-- table:LEVEL:CASE --> and <!-- /table -->) and copy the sheets
    python tools/bench/report.py variants CASE   a case with variants (rosette) only: fill its blocks
                                                 <!-- vtable:LEVEL:CASE -->, <!-- vdurations:CASE -->,
                                                 <!-- darkcheck:CASE --> and copy its sheets

Writes docs/pilot/bench/tables_<level>.md (one table per case: ours, Siril, which is better) and copies
<case>/sheet_<level>.jpg to docs/pilot/bench/ (a folder kept out of git). Also prints the durations.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

C.utf8_console()
DOCS = C.REPO / "docs" / "pilot" / "bench"
#: the final pass of the bench (the engine with the base patches): its runs are the ones counted in the durations
FINAL_SINCE = "2026-11-01T16:22"


def _f(v, fmt="{:.2f}"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return fmt.format(v)


def better(o, s, higher: bool, tol: float, rel: bool = False, abs_tol: float = 0.0) -> str:
    if o is None or s is None or not (math.isfinite(o) and math.isfinite(s)):
        return "–"
    d = (o - s) / abs(s) if rel and s else (o - s)
    if abs(d) <= tol or abs(o - s) <= abs_tol:
        return "égal"
    return "**nous**" if (d > 0) == higher else "Siril"


def durations(case: str, level: str, variant: str = "", siril_variant: str | None = None) -> dict:
    out = {}
    lv = level
    man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
    if man["levels"][level].get("same_as"):
        lv = man["levels"][level]["same_as"]
    s = C.siril_dir(case, lv, variant if siril_variant is None else siril_variant) / "siril.json"
    if s.exists():
        j = json.loads(s.read_text(encoding="utf-8"))
        out["siril_pre_min"] = j.get("pre_s", 0) / 60
        out["siril_post_min"] = j.get("post_s", 0) / 60
        out["siril_min"] = out["siril_pre_min"] + out["siril_post_min"]
        out["siril_colour"] = j.get("colour")
    o = C.ours_dir(case, lv, variant) / "bench_run.json"
    if o.exists():
        j = json.loads(o.read_text(encoding="utf-8"))
        # the final pass (the patched engine): every stage computed since FINAL_SINCE, counted once (a run stopped and
        # resumed counts the stages it finished), reused ones not counted
        stages: dict = {}
        for r in j["runs"]:
            if str(r.get("when") or "") < FINAL_SINCE:
                continue
            for k, v in (r.get("stages") or {}).items():
                if v.get("status") == "done" and not v.get("reused") and v.get("duration"):
                    stages[k] = v["duration"] / 60
        if "render_simple" in stages:
            out["ours_min"] = sum(stages.values())
            out["ours_stages_min"] = stages
        last = j.get("last") or {}
        st = (last.get("metrics") or {}).get("stack", {}).get("metrics") or {}
        out["ours_frames"] = st.get("frames_used")
    return out


def siril_frames(case: str, level: str, variant: str = "") -> int | None:
    import re
    f = C.siril_dir(case, level, variant) / "pre.log"
    if not f.exists():
        return None
    m = re.findall(r"(\d+) images? (?:have been|were|are) stacked|Stacking (\d+) images|Integration of (\d+) images",
                   f.read_text(encoding="utf-8", errors="replace"))
    vals = [int(x) for t in m for x in t if x]
    return vals[-1] if vals else None


def table(case: str, level: str) -> str:
    lv = level
    man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
    if man["levels"][level].get("same_as"):
        lv = man["levels"][level]["same_as"]
    m = json.loads((C.BENCH / case / f"measures_{lv}.json").read_text(encoding="utf-8"))
    du = durations(case, level)
    o, s = "ours", "siril"
    n, st, ff, tg, rs, fl, co = (m["noise"], m["stars"], m["faint_stars"], m["target"], m["residuals"],
                                 m["flatness"], m["colours"])
    rows = []

    def row(label, vo, vs, fmt, higher, tol, rel=False, abs_tol=0.0):
        rows.append(f"| {label} | {_f(vo, fmt)} | {_f(vs, fmt)} | {better(vo, vs, higher, tol, rel, abs_tol)} |")

    rows.append(f"| Poses empilées | {du.get('ours_frames', '–')} | {siril_frames(case, lv) or '–'} | |")
    row(f"Bruit du fond (mag/arcsec² sur {m['box_arcsec']:.1f}″, plus haut = moins de bruit)",
        n[o].get("sb_mag_arcsec2"), n[s].get("sb_mag_arcsec2"), "{:.2f}", True, 0.05)
    row(f"Bruit du fond à plus grande échelle ({4 * m['box_arcsec']:.0f}″)", n[o].get("sb_large_mag_arcsec2"),
        n[s].get("sb_large_mag_arcsec2"), "{:.2f}", True, 0.05)
    row("Profondeur : étoile détectable à 5σ (magnitude G)", n[o].get("depth5_mag"), n[s].get("depth5_mag"),
        "{:.2f}", True, 0.05)
    row(f"Étoiles faibles trouvées à 5σ (sur {ff[o]['of']} étoiles Gaia isolées)", ff[o]["found"], ff[s]["found"],
        "{:.0f}", True, 0.03, True)
    row("Magnitude où la moitié des étoiles est trouvée", ff[o].get("g50"), ff[s].get("g50"), "{:.2f}", True, 0.05)
    row("Rapport signal/bruit sur la cible", tg.get("snr_ours"), tg.get("snr_siril"), "{:.1f}", True, 0.05, True)
    row("Finesse des étoiles, FWHM au centre (″)", st[o].get("fwhm_centre_arcsec"), st[s].get("fwhm_centre_arcsec"),
        "{:.2f}", False, 0.03, True)
    row("FWHM au bord du champ (″)", st[o].get("fwhm_outer_arcsec"), st[s].get("fwhm_outer_arcsec"), "{:.2f}", False,
        0.03, True)
    row("Rondeur au centre (1 = rond)", st[o].get("roundness_centre"), st[s].get("roundness_centre"), "{:.2f}", True,
        0.02)
    row("Rondeur au bord", st[o].get("roundness_outer"), st[s].get("roundness_outer"), "{:.2f}", True, 0.02)
    row("Traînées (satellites, avions) restantes", rs[o]["trails"], rs[s]["trails"], "{:.0f}", False, 0)
    for kind, lab in (("hot", "claires"), ("cold", "sombres")):
        a_o = rs[o][f"{kind}_only_here"] / max(rs[o]["area_deg2"], 1e-6)
        a_s = rs[s][f"{kind}_only_here"] / max(rs[s]["area_deg2"], 1e-6)
        row(f"Taches {lab} propres à l'image (par degré²)", a_o, a_s, "{:.0f}", False, 0.15, True, 3.0)
    cs = m.get("coloured_spots") or {}
    if cs:
        row("Points d'une seule couleur : pixels chauds restants (par degré²)", cs[o].get("all_per_deg2"),
            cs[s].get("all_per_deg2"), "{:.0f}", False, 0.15, True, 3.0)
    sk = m.get("streaks") or {}
    if sk:
        row("Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune)",
            sk[o].get("ratio_along"), sk[s].get("ratio_along"), "{:.2f}", False, 0.03)
    row("Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat)",
        fl[o].get("spread_sb"), fl[s].get("spread_sb"), "{:.2f}", True, 0.1)
    for k, lab in (("b_r", "B−R"), ("g_r", "V−R")):
        vo, vs = (co[o].get(k) or {}), (co[s].get(k) or {})
        row(f"Couleurs des étoiles : erreur {lab} face à Gaia, en BP−RP (plus bas = plus fidèle)",
            vo.get("scatter_bp_rp"), vs.get("scatter_bp_rp"), "{:.3f}", False, 0.005)
    vo, vs = (co[o].get("b_r") or {}), (co[s].get("b_r") or {})
    rows.append(f"| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | "
                f"{_f(vo.get('slope'))} | {_f(vs.get('slope'))} | |")
    rows.append(f"| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | {_f(vo.get('white_at_bp_rp'))} | "
                f"{_f(vs.get('white_at_bp_rp'))} | |")
    if tg.get("signal_sb_ours") is not None:
        rows.append(f"| Cible : signal / bruit en mag/arcsec² (info) | {_f(tg.get('signal_sb_ours'))} / "
                    f"{_f(tg.get('noise_sb_ours'))} | {_f(tg.get('signal_sb_siril'))} / "
                    f"{_f(tg.get('noise_sb_siril'))} | |")
    rows.append(f"| Durée (min) | {_f(du.get('ours_min'), '{:.1f}')} | {_f(du.get('siril_min'), '{:.1f}')} | |")
    head = "| Mesure | astro_gpu | Siril | Meilleur |\n|---|---|---|---|\n"
    return head + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------------------------- the overall table

def _fr(v: float, fmt: str, sign: bool = True) -> str:
    s = (("{:+" + fmt[2:]) if sign else fmt).format(v)
    return s.replace("-", "−").replace(".", ",")


def overall() -> str:
    """Our gap to Siril on every case: Base, (Express) in brackets; a case whose Base is its Express shows one."""
    def load(case, level):
        f = C.BENCH / case / f"measures_{level}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def cell(case, fn):
        out = []
        for level in ("base", "express"):
            m = load(case, level)
            if m is None:
                continue
            try:
                out.append(fn(m))
            except (KeyError, TypeError, ZeroDivisionError, ValueError):
                out.append("–")
        if not out:
            return "–"
        return out[0] if len(out) == 1 else f"{out[0]} ({out[1]})"
    o, s = "ours", "siril"
    rows = [
        ("Bruit fin (mag/arcsec²)", lambda m: _fr(m["noise"][o]["sb_mag_arcsec2"] - m["noise"][s]["sb_mag_arcsec2"],
                                                  "{:.2f}")),
        ("Bruit à grande échelle", lambda m: _fr(m["noise"][o]["sb_large_mag_arcsec2"] -
                                                 m["noise"][s]["sb_large_mag_arcsec2"], "{:.2f}")),
        ("Profondeur 5σ (mag)", lambda m: _fr(m["noise"][o]["depth5_mag"] - m["noise"][s]["depth5_mag"], "{:.2f}")),
        ("Étoiles faibles trouvées", lambda m: _fr(100 * (m["faint_stars"][o]["found"] /
                                                          m["faint_stars"][s]["found"] - 1), "{:.0f}") + " %"),
        ("Signal/bruit sur la cible", lambda m: _fr(100 * (m["target"]["snr_ours"] / m["target"]["snr_siril"] - 1),
                                                    "{:.0f}") + " %"),
        ("Finesse des étoiles (FWHM au centre)", lambda m: _fr(100 * (m["stars"][o]["fwhm_centre_arcsec"] /
                                                                      m["stars"][s]["fwhm_centre_arcsec"] - 1),
                                                               "{:.0f}") + " %"),
        ("Rondeur au centre", lambda m: _fr(m["stars"][o]["roundness_centre"] - m["stars"][s]["roundness_centre"],
                                            "{:.2f}")),
        ("Planéité du fond (mag/arcsec²)", lambda m: _fr(m["flatness"][o]["spread_sb"] - m["flatness"][s]["spread_sb"],
                                                         "{:.2f}")),
        ("Erreur de couleur B−R (nous / Siril)", lambda m: _fr(m["colours"][o]["b_r"]["scatter_bp_rp"], "{:.3f}",
                                                               False) + " / " +
         _fr(m["colours"][s]["b_r"]["scatter_bp_rp"], "{:.3f}", False)),
        ("Erreur de couleur V−R (nous / Siril)", lambda m: _fr(m["colours"][o]["g_r"]["scatter_bp_rp"], "{:.3f}",
                                                               False) + " / " +
         _fr(m["colours"][s]["g_r"]["scatter_bp_rp"], "{:.3f}", False)),
        ("Points d'une seule couleur par degré² (nous / Siril)",
         lambda m: f"{m['coloured_spots'][o]['all_per_deg2']:.0f} / {m['coloured_spots'][s]['all_per_deg2']:.0f}"),
        ("Traînées le long de la dérive (nous / Siril ; 1 = aucune)",
         lambda m: _fr(m["streaks"][o]["ratio_along"], "{:.2f}", False) + " / " +
         _fr(m["streaks"][s]["ratio_along"], "{:.2f}", False)),
    ]
    cases = [c for c in C.CASES if load(c, "express") is not None]
    head = "| | " + " | ".join(cases) + " |\n|---|" + "---|" * len(cases) + "\n"
    return head + "\n".join(f"| {lab} | " + " | ".join(cell(c, fn) for c in cases) + " |" for lab, fn in rows) + "\n"


def calcheck_table() -> str:
    f = C.BENCH / "calcheck.json"
    if not f.exists():
        return "(contrôle pas encore fait)\n"
    d = json.loads(f.read_text(encoding="utf-8"))["cases"]
    rows = ["| Cas | Darks | Flats | Offsets | Verdict |", "|---|---|---|---|---|"]
    for case in C.CASES:
        projs = d.get(case) or {}
        main = projs.get("ours_base") or projs.get("ours_express") or {}
        if not main and not projs:
            continue

        def k(kind):
            e = main.get(kind)
            if not e:
                return "–"
            t = f"{e['count']} : {e['ok']} bonnes"
            if e.get("warn"):
                t += f", {e['warn']} à vérifier"
            if e.get("bad"):
                t += f", **{e['bad']} mauvaises**"
            return t
        bad = sum((e or {}).get("bad", 0) for e in main.values())
        verdict = "pas d'images de calibration" if not main else ("à refaire" if bad else "aucune image mauvaise")
        rows.append(f"| {case} | {k('darks')} | {k('flats')} | {k('bias')} | {verdict} |")
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------------------------- a case with variants

def _vcols(case: str, level: str) -> list[tuple[str, str, dict]]:
    """The columns of a variant table, from the measures of every image of the level on the same sky
    (measures_<level>_all.json): (label, image name, its measures) — ours for every variant, then Siril's."""
    f = C.BENCH / case / f"measures_{level}_all.json"
    if not f.exists():
        return []
    m = json.loads(f.read_text(encoding="utf-8"))
    cols = []
    for side, lab in (("ours", "nous"), ("siril", "Siril")):
        for n in m["images"]:
            if n.startswith(side + "_"):
                v = n.split("_", 1)[1]
                e = dict(m["by_image"][n])
                e["_du"] = durations(case, level, v)
                e["_side"], e["_v"] = side, v
                cols.append((f"{lab} {v}", n, e))
    return cols


def _get(e: dict, key: str = ""):
    n, st, ff, tg = e["noise"], e["stars"], e["faint_stars"], e["target"]
    fl, co, sk, cs, rs = e["flatness"], e["colours"], e["streaks"], e["coloured_spots"], e["residuals"]
    side = e["_side"]
    return {
        "fine": n.get("sb_mag_arcsec2"), "large": n.get("sb_large_mag_arcsec2"), "depth": n.get("depth5_mag"),
        "faint": ff.get("found"), "g50": ff.get("g50"), "snr": tg.get("snr"),
        "fwhm_c": st.get("fwhm_centre_arcsec"), "fwhm_o": st.get("fwhm_outer_arcsec"),
        "rnd_c": st.get("roundness_centre"), "rnd_o": st.get("roundness_outer"), "trails": rs.get("trails"),
        "spots": cs.get("all_per_deg2"), "streak_ratio": sk.get("ratio_along"), "streak_sb": sk.get("streak_sb"),
        "streak_max": sk.get("max_ratio"), "streak_max_angle": sk.get("max_angle_deg"),
        "flat": fl.get("spread_sb"), "b_r": (co.get("b_r") or {}).get("scatter_bp_rp"),
        "g_r": (co.get("g_r") or {}).get("scatter_bp_rp"),
        "frames": e["_du"].get("ours_frames") if side == "ours" else None,
        "minutes": e["_du"].get("ours_min") if side == "ours" else e["_du"].get("siril_min"),
    }


#: the rows of a variant table: key, label, format, higher is better, tolerance (relative when rel)
VROWS = [
    ("frames", "Poses empilées", "{:.0f}", None, 0, False),
    ("fine", "Bruit du fond, petite échelle (mag/arcsec², plus haut = moins de bruit)", "{:.2f}", True, 0.05, False),
    ("large", "Bruit du fond, 4 fois plus grand", "{:.2f}", True, 0.05, False),
    ("depth", "Profondeur : étoile détectable à 5σ (magnitude G)", "{:.2f}", True, 0.05, False),
    ("faint", "Étoiles faibles trouvées à 5σ (les mêmes étoiles Gaia pour toutes)", "{:.0f}", True, 0.03, True),
    ("g50", "Magnitude où la moitié des étoiles est trouvée", "{:.2f}", True, 0.05, False),
    ("snr", "Signal/bruit sur la nébuleuse", "{:.1f}", True, 0.05, True),
    ("fwhm_c", "Finesse des étoiles, FWHM au centre (″)", "{:.2f}", False, 0.03, True),
    ("fwhm_o", "FWHM au bord du champ (″)", "{:.2f}", False, 0.03, True),
    ("rnd_c", "Rondeur au centre (1 = rond)", "{:.2f}", True, 0.02, False),
    ("rnd_o", "Rondeur au bord", "{:.2f}", True, 0.02, False),
    ("spots", "Pixels chauds restants : points d'une seule couleur (par degré²)", "{:.0f}", False, 0.15, True),
    ("streak_ratio", "Traînées le long de la dérive : bruit le long / bruit moyen (1 = aucune)", "{:.2f}", False, 0.03,
     False),
    ("streak_sb", "Traînées le long de la dérive : leur part du bruit (mag/arcsec², plus haut = plus faibles)",
     "{:.2f}", True, 0.1, False),
    ("trails", "Traînées longues détectées (satellites, avions, défauts)", "{:.0f}", False, 0, False),
    ("flat", "Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat)", "{:.2f}", True,
     0.1, False),
    ("b_r", "Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle)", "{:.3f}", False,
     0.005, False),
    ("g_r", "Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP", "{:.3f}", False, 0.005, False),
    ("minutes", "Durée (min)", "{:.1f}", None, 0, False),
]


def _cmp(a, b, higher, tol, rel) -> str:
    """b against a: "mieux", "égal", "moins bien"."""
    if higher is None or a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return ""
    d = (b - a) / abs(a) if rel and a else (b - a)
    if abs(d) <= tol:
        return "égal"
    return "**mieux**" if (d > 0) == higher else "moins bien"


def vtable(case: str, level: str) -> str:
    cols = _vcols(case, level)
    if not cols:
        return "(mesures pas encore faites)\n"
    vals = [(lab, _get(e)) for lab, _n, e in cols]
    by = dict(vals)
    pairs = [("B face à A (nous)", "nous A", "nous B"), ("B face à A (Siril)", "Siril A", "Siril B")]
    pairs = [p for p in pairs if p[1] in by and p[2] in by]
    head = "| Mesure | " + " | ".join(lab for lab, _ in vals) + "".join(f" | {p[0]}" for p in pairs) + " |\n"
    head += "|---|" + "---|" * (len(vals) + len(pairs)) + "\n"
    rows = []
    for key, label, fmt, higher, tol, rel in VROWS:
        cells = []
        for lab, v in vals:
            x = v.get(key)
            if key == "frames" and lab.startswith("Siril"):
                x = siril_frames(case, level, lab.split()[-1])
            cells.append(_f(x, fmt) if x is not None else "–")
        cmp = [_cmp(by[a].get(key), by[b].get(key), higher, tol, rel) for _, a, b in pairs]
        rows.append(f"| {label} | " + " | ".join(cells) + "".join(f" | {c}" for c in cmp) + " |")
    return head + "\n".join(rows) + "\n"


def darkcheck_table(case: str) -> str:
    f = C.BENCH / case / "darkcheck.json"
    if not f.exists():
        return "(contrôle pas encore fait)\n"
    d = json.loads(f.read_text(encoding="utf-8"))
    vs = d["variants"]
    names = [v for v in ("A", "B", "C") if v in vs]
    lab = {"A": "A : sans dark", "B": "B : darks d'Iris", "C": "C : fausses darks"}
    head = "| | " + " | ".join(lab[v] for v in names) + " |\n|---|" + "---|" * len(names) + "\n"

    def r(label, fn):
        cells = []
        for v in names:
            try:
                x = fn(vs[v])
            except (KeyError, TypeError):
                x = None
            cells.append("–" if x is None else x)
        return f"| {label} | " + " | ".join(cells) + " |"
    rows = [
        r("Niveau de la master dark (ADU ; noir du capteur 512, fond des lights "
          f"{d['lights_floor_adu']:.0f})", lambda e: f"{e['master_level_adu']:.1f}" if "master_level_adu" in e else
          "512 (noir)"),
        r(f"Pixels chauds de la master (excès > {d['thr_adu']:.0f} ADU ; les lights en montrent "
          f"{d['lights_hot']})", lambda e: f"{e['master_hot']}" if "master_hot" in e else "0"),
        r("Pixels chauds des lights que la master a aussi", lambda e: f"{e['lights_hot_also_in_master_pct']:.0f} %"),
        r("Pixels chauds de la master absents des lights (creuseraient un trou)",
          lambda e: f"{e['master_hot_not_in_lights_pct']:.0f} %"),
        r("Intensité dans les lights / dans la master (pixels chauds forts ; 1 = même température)",
          lambda e: f"{e['amplitude_ratio_lights_over_master']:.2f} ({e['amplitude_ratio_p25_p75'][0]:.2f}–"
                    f"{e['amplitude_ratio_p25_p75'][1]:.2f})"),
        r("Pixels chauds restants après soustraction (par million de pixels)", lambda e: f"{e['left_hot_ppm']:.0f}"),
        r("Trous creusés par la soustraction (par million)", lambda e: f"{e.get('holes_ppm', 0):.0f}"),
        r(f"Motif de lignes / colonnes de la master (ADU ; lights : {d['lights_pattern_adu']['rows']:.2f} / "
          f"{d['lights_pattern_adu']['cols']:.2f})",
          lambda e: f"{e['pattern_adu']['rows']:.2f} / {e['pattern_adu']['cols']:.2f}"),
        r("Ressemblance de ce motif avec celui des lights (corrélation lignes / colonnes)",
          lambda e: f"{e['pattern_corr_with_lights']['rows']:.2f} / {e['pattern_corr_with_lights']['cols']:.2f}"),
        r("Structure à grande échelle de la master (ADU, p99 − p1 de blocs de 64 px)",
          lambda e: f"{e['large_scale_p99_p1_adu']:.1f}"),
    ]
    return head + "\n".join(rows) + "\n"


def vdurations(case: str) -> str:
    rows = ["| Niveau | Variante | Siril (préparation + fin) | Nous (tout) | dont astrométrie + catalogues | "
            "Nous sans le réseau |", "|---|---|---|---|---|---|"]
    for level in C.LEVELS:
        for v in C.variants_of(case):
            if level not in C.variant_levels(case, v):
                continue
            d = durations(case, level, v)
            if "siril_min" not in d and "ours_min" not in d:
                continue
            st = d.get("ours_stages_min") or {}
            net = st.get("astrometry", 0) + st.get("catalogs", 0)
            o = d.get("ours_min")
            sir = (f"{_f(d.get('siril_min'), '{:.1f}')} min ({_f(d.get('siril_pre_min'), '{:.1f}')} + "
                   f"{_f(d.get('siril_post_min'), '{:.1f}')})") if "siril_min" in d else "–"
            rows.append(f"| {level} | {v} | {sir} | {_f(o, '{:.1f}')} min | {net:.1f} min | "
                        f"{_f(o - net if o else None, '{:.1f}')} min |")
    return "\n".join(rows) + "\n"


def fill_variants(case: str) -> int:
    import re
    md = C.REPO / "docs" / "pilot" / "BENCH.md"
    text = md.read_text(encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)

    def one(m):
        level = m.group(1)
        sh = C.BENCH / case / f"sheet_{level}.jpg"
        if sh.exists():
            shutil.copy2(sh, DOCS / f"{case}_{level}.jpg")
        return f"<!-- vtable:{level}:{case} -->\n{vtable(case, level)}<!-- /vtable -->"
    text = re.sub(rf"<!-- vtable:(\w+):{case} -->.*?<!-- /vtable -->", one, text, flags=re.S)
    text = re.sub(rf"<!-- vdurations:{case} -->.*?<!-- /vdurations -->",
                  lambda m: f"<!-- vdurations:{case} -->\n{vdurations(case)}<!-- /vdurations -->", text, flags=re.S)
    text = re.sub(rf"<!-- darkcheck:{case} -->.*?<!-- /darkcheck -->",
                  lambda m: f"<!-- darkcheck:{case} -->\n{darkcheck_table(case)}<!-- /darkcheck -->", text,
                  flags=re.S)
    md.write_text(text, encoding="utf-8")
    print(f"filled the {case} blocks of {md}")
    return 0


def durations_table() -> str:
    rows = ["| Cas | Niveau | Siril (préparation + fin) | Nous (tout) | dont astrométrie + catalogues (réseau) | "
            "Nous sans le réseau |", "|---|---|---|---|---|---|"]
    for case in C.CASES:
        for level in C.LEVELS:
            man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
            if man["levels"][level].get("same_as"):
                continue
            d = durations(case, level)
            if "siril_min" not in d and "ours_min" not in d:
                continue
            st = d.get("ours_stages_min") or {}
            net = st.get("astrometry", 0) + st.get("catalogs", 0)
            o = d.get("ours_min")
            rows.append(f"| {case} | {level} | {_f(d.get('siril_min'), '{:.1f}')} min "
                        f"({_f(d.get('siril_pre_min'), '{:.1f}')} + {_f(d.get('siril_post_min'), '{:.1f}')}) | "
                        f"{_f(o, '{:.1f}')} min | {net:.1f} min | {_f(o - net if o else None, '{:.1f}')} min |")
    return "\n".join(rows) + "\n"


def fill() -> int:
    """Replace every table block of BENCH.md with the current measures (and copy the sheets)."""
    import re
    md = C.REPO / "docs" / "pilot" / "BENCH.md"
    text = md.read_text(encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)

    def one(m):
        level, case = m.group(1), m.group(2)
        try:
            t = table(case, level)
        except FileNotFoundError:
            t = "(mesures pas encore faites)\n"
        lv = level
        man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
        if man["levels"][level].get("same_as"):
            lv = man["levels"][level]["same_as"]
        sh = C.BENCH / case / f"sheet_{lv}.jpg"
        if sh.exists():
            shutil.copy2(sh, DOCS / f"{case}_{lv}.jpg")
        return f"<!-- table:{level}:{case} -->\n{t}<!-- /table -->"
    text = re.sub(r"<!-- table:(\w+):(\w+) -->.*?<!-- /table -->", one, text, flags=re.S)
    text = re.sub(r"<!-- durations -->.*?<!-- /durations -->",
                  lambda m: f"<!-- durations -->\n{durations_table()}<!-- /durations -->", text, flags=re.S)
    text = re.sub(r"<!-- overall -->.*?<!-- /overall -->",
                  lambda m: f"<!-- overall -->\n{overall()}<!-- /overall -->", text, flags=re.S)
    text = re.sub(r"<!-- calcheck -->.*?<!-- /calcheck -->",
                  lambda m: f"<!-- calcheck -->\n{calcheck_table()}<!-- /calcheck -->", text, flags=re.S)
    text = re.sub(r"<!-- darkcheck:(\w+) -->.*?<!-- /darkcheck -->",
                  lambda m: f"<!-- darkcheck:{m.group(1)} -->\n{darkcheck_table(m.group(1))}<!-- /darkcheck -->",
                  text, flags=re.S)
    md.write_text(text, encoding="utf-8")
    print(f"filled {md}")
    return 0


def main() -> int:
    if sys.argv[1] == "fill":
        return fill()
    if sys.argv[1] == "variants":
        return fill_variants(sys.argv[2])
    if sys.argv[1] == "show":                    # print a case's variant tables (no file written)
        for lv in C.LEVELS:
            print(f"### {lv}\n{vtable(sys.argv[2], lv)}")
        print(darkcheck_table(sys.argv[2]))
        print(vdurations(sys.argv[2]))
        return 0
    level = sys.argv[1]
    names = sys.argv[2:] or list(C.CASES)
    DOCS.mkdir(parents=True, exist_ok=True)
    parts = []
    for c in names:
        try:
            t = table(c, level)
        except FileNotFoundError as e:
            print(f"{c}: no measures yet ({e})")
            continue
        parts.append(f"### {c}\n\n{t}")
        sh = C.BENCH / c / f"sheet_{level}.jpg"
        if sh.exists():
            shutil.copy2(sh, DOCS / f"{c}_{level}.jpg")
        print(c, json.dumps(durations(c, level), default=str))
    (DOCS / f"tables_{level}.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"written {DOCS / f'tables_{level}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
