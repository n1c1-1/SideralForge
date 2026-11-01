"""Tables of the bench report (docs/pilot/BENCH.md) from the measures, and the sheets copied for it.

    python tools/bench/report.py LEVEL [CASE ...]
    python tools/bench/report.py fill            fill the tables of docs/pilot/BENCH.md (between the markers
                                                 <!-- table:LEVEL:CASE --> and <!-- /table -->) and copy the sheets

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


def durations(case: str, level: str) -> dict:
    out = {}
    lv = level
    man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
    if man["levels"][level].get("same_as"):
        lv = man["levels"][level]["same_as"]
    s = C.BENCH / case / f"siril_{lv}" / "siril.json"
    if s.exists():
        j = json.loads(s.read_text(encoding="utf-8"))
        out["siril_pre_min"] = j.get("pre_s", 0) / 60
        out["siril_post_min"] = j.get("post_s", 0) / 60
        out["siril_min"] = out["siril_pre_min"] + out["siril_post_min"]
        out["siril_colour"] = j.get("colour")
    o = C.BENCH / case / f"ours_{lv}" / "bench_run.json"
    if o.exists():
        j = json.loads(o.read_text(encoding="utf-8"))
        # the first complete computation: the runs up to the one that ends with render_simple (Base runs in two
        # parts around the catalogue pre-fetch)
        tot, stages = 0.0, {}
        for r in j["runs"]:
            if not r.get("wall_s") or r.get("status") != "done":      # a failed attempt is not counted
                continue
            tot += r["wall_s"]
            stages.update({k: (v.get("duration") or 0) / 60 for k, v in r.get("stages", {}).items()})
            if "render_simple" in r.get("stages", {}):
                out["ours_min"] = tot / 60
                out["ours_stages_min"] = stages
                break
        last = j.get("last") or {}
        st = (last.get("metrics") or {}).get("stack", {}).get("metrics") or {}
        out["ours_frames"] = st.get("frames_used")
    return out


def siril_frames(case: str, level: str) -> int | None:
    import re
    f = C.BENCH / case / f"siril_{level}" / "pre.log"
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
    md.write_text(text, encoding="utf-8")
    print(f"filled {md}")
    return 0


def main() -> int:
    if sys.argv[1] == "fill":
        return fill()
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
