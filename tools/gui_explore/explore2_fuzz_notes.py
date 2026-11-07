"""Round 2 fuzz: anomalies seen while rechecking / fuzzing that no invariant of the fuzz checks, each replayed twice
(fresh projects) before it is noted.

    python explore2_fuzz_notes.py check     # replay and print only
    python explore2_fuzz_notes.py note      # replay and note those that reproduce 2/2
"""
import os
import sys
from pathlib import Path

WHAT = sys.argv[1] if len(sys.argv) > 1 else "check"
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_fuzz as F  # noqa: E402

fz = F.Fuzz("fr")
ex = fz.ex


def stars_card_reason(k, lang):
    fz.start({"kind": "two_flows", "lang": lang}, f"n1_{lang}_{k}")
    c = ex.page.cards.get("stars")
    return (c.title, c.reason) if c is not None else None


def session_kept(k):
    fz.start({"template": "sho", "lang": "fr"}, f"n2_{k}")
    s0 = [s.name for s in ex.s.p.sessions]
    it = next(x for x in fz.items() if x.node.key == "session" and fz.title(x) == "Ha")
    ex.drag_block_to_palette(it)
    fz.wait_ready()
    toast = ex.view.toast.text() if ex.view.toast.isVisible() else ""
    s1 = [s.name for s in ex.s.p.sessions]
    drawn = [fz.title(x) for x in fz.items() if x.node.key == "session"]
    return s0, s1, drawn, toast


res = {}
for k in range(2):
    res.setdefault("stars_fr", []).append(stars_card_reason(k, "fr"))
    res.setdefault("stars_en", []).append(stars_card_reason(k, "en"))
    res.setdefault("kept", []).append(session_kept(k))
for key, v in res.items():
    print(key, v, flush=True)

if WHAT == "note":
    sf = res["stars_fr"]
    if all(x and "Sortie" in (x[1] or "") and "Sortie" not in x[0] for x in sf):
        fz.start({"kind": "two_flows", "lang": "fr"}, "n1_note")
        ex.hover(ex.page.cards["stars"], ex.page.cards["stars"].rect().center())
        se = res["stars_en"][-1]
        ex.note("mineur", "La carte grisée « Étoiles/fond » s'appelle « Sortie » dans sa raison",
                f"Dans la palette, la carte « {sf[-1][0]} » est grisée tant qu'il n'y a pas d'Empilement ; la raison "
                f"affichée dit « {sf[-1][1]} » (en anglais : « {se[0]} » / « {se[1]} »). Le même bloc porte deux noms ; "
                "« Sortie » est aussi le nom des blocs de fin de workflow dans le graphe. Reproduit 2/2.",
                shot=True, steps=["projet de deux flux sans bloc (ou toute page sans Empilement)",
                                  "regarder la carte « Étoiles/fond » de la palette, section Post (grisée) et sa raison"])
    kp = res["kept"]
    if all(x[3] and "Ha" in x[2] for x in kp):
        fz.start({"template": "sho", "lang": "fr"}, "n2_note")
        it = next(x for x in fz.items() if x.node.key == "session" and fz.title(x) == "Ha")
        ex.drag_block_to_palette(it)
        fz.wait_ready()
        ex.note("question", "Session d'une Fusion retirée : le message dit « retirée » mais le bloc reste",
                "Projet SHO (sessions SII, Ha, OIII et leur Fusion) : glisser la Session « Ha » sur la palette (ou Suppr, "
                f"ou son menu « Supprimer ») affiche « {kp[-1][3]} », mais un bloc « Ha » reste dessiné : la session est "
                "remplacée par une session « Ha » vide (même nom, même filtre), redescendue en bas de la colonne "
                f"(avant : {', '.join(kp[-1][0])} ; après : {', '.join(kp[-1][1])}), et la Fusion est cerclée de rouge. "
                "Sur un projet dont les sessions sont déjà vides, rien ne semble avoir été retiré. Le projet reste valide "
                "et Ctrl+Z le rend. À juger : le message, ou dire « vidée » / garder la place du bloc. Reproduit 2/2.",
                shot=True, steps=["nouveau projet, modèle « SHO »", "glisser le bloc « Ha » sur la palette"])
os._exit(0)
