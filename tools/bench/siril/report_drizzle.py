"""The tables of "Le drizzle de Siril" (docs/pilot/BENCH.md) from the six <case>/drizzle_siril_express.json.

    python tools/bench/siril/report_drizzle.py [OUT.md]        (default: print)
One table per measure: the cases in rows, the four stacks in columns, then the two gaps that answer the question
(Siril: drizzle against classic; us: drizzle against classic).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH = Path("F:/astro_essais/bench")
CASES = [("heart", "Heart"), ("iris", "Iris"), ("m51", "M51"), ("m8", "M8"), ("seestar", "Seestar"),
         ("rosette", "Rosette")]
NAMES = ["Siril", "Siril drizzle", "LEG", "CL-off"]      # (LEG: our drizzle; CL-off: our classic stack)
HEAD = ["Siril classique", "Siril drizzle", "Notre drizzle", "Notre classique"]
# key, title, path, format, gap kind (mag: difference; rel: percent; abs: difference), plain-words hint
MEASURES = [
    ("Profondeur : magnitude de l'étoile détectée à 5σ (plus haut = plus profond)", "noise.depth5_mag", "{:.2f}", "abs"),
    ("Bruit du fond, petite échelle (mag/arcsec² ; plus haut = moins de bruit)", "noise.sb_mag_arcsec2", "{:.2f}",
     "abs"),
    ("Bruit du fond, grande échelle (mag/arcsec² ; plus haut = moins de bruit)", "noise.sb_large_mag_arcsec2",
     "{:.2f}", "abs"),
    ("Signal/bruit sur la cible", "target.snr", "{:.2f}", "rel"),
    ("Étoiles faibles trouvées à 5σ (les mêmes étoiles Gaia pour les quatre)", "faint_stars.found", "{:.0f}", "rel"),
    ("Finesse des étoiles : largeur à mi-hauteur au centre (″ ; plus bas = plus fines)", "stars.fwhm_centre_arcsec",
     "{:.2f}", "rel"),
    ("Points d'une seule couleur (pixels chauds restants, par degré²)", "coloured_spots.all_per_deg2", "{:.0f}", "abs"),
    ("Couleurs des étoiles : erreur bleu−rouge face à Gaia (plus bas = plus fidèle)", "colours.b_r.scatter_bp_rp",
     "{:.3f}", "abs"),
    ("Couleurs des étoiles : erreur vert−rouge face à Gaia (plus bas = plus fidèle)", "colours.g_r.scatter_bp_rp",
     "{:.3f}", "abs"),
    ("Planéité du fond après retrait du dégradé (mag/arcsec² ; plus haut = plus plat)", "flatness.spread_sb", "{:.2f}",
     "abs"),
    ("Taux de rejet de l'empilement (% des échantillons écartés)", "engine.rejected_pct", "{:.3f}", "abs"),
]


def get(e: dict, path: str):
    for k in path.split("."):
        if not isinstance(e, dict):
            return None
        e = e.get(k)
    return e


def fr(v, fmt: str, sign: bool = False) -> str:
    if v is None or v != v:
        return "–"
    s = ("{:+" + fmt[2:]) if sign else fmt
    return s.format(v).replace("-", "−").replace(".", ",")


def gap(a, b, kind: str, fmt: str) -> str:
    if a is None or b is None or a != a or b != b:
        return "–"
    if kind == "rel":
        return fr(100 * (a / b - 1), "{:.0f}", True) + " %" if b else "–"
    return fr(a - b, fmt, True)


def main() -> None:
    data = {}
    for c, _ in CASES:
        f = BENCH / c / "drizzle_siril_express.json"
        if f.exists():
            data[c] = json.loads(f.read_text(encoding="utf-8"))["by_image"]
    out = []
    for title, path, fmt, kind in MEASURES:
        out.append(f"**{title}**\n")
        out.append("| Cas | " + " | ".join(HEAD) + " | Siril : drizzle − classique | Nous : drizzle − classique |")
        out.append("|---|" + "---:|" * 6)
        for c, label in CASES:
            if c not in data:
                continue
            v = [get(data[c].get(n, {}), path) for n in NAMES]
            out.append(f"| {label} | " + " | ".join(fr(x, fmt) for x in v) + f" | {gap(v[1], v[0], kind, fmt)} | "
                       f"{gap(v[2], v[3], kind, fmt)} |")
        out.append("")
    text = "\n".join(out)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
