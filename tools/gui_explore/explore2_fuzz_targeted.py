"""Round 2 fuzz: hand-written sequences on the new gestures (save + reopen, language, files dropped from the
Explorer) with the fuzz's checks.

    python explore2_fuzz_targeted.py [drop]
"""
import os
import sys
from pathlib import Path

WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402

fz = G.Fuzz2("fr")
CASES = [
    ("image: fil retiré, enregistré, rouvert", {"kind": "image", "lang": "fr"},
     [["wire", 0.0, 0.1, 0.5], ["save_reopen", 0, 0, 0], ["undo", 0, 0, 0], ["save_reopen", 0, 0, 0]]),
    ("legacy: Vers Post, enregistré, rouvert", {"kind": "legacy", "lang": "fr"},
     [["menu", "Astrométrie", "Vers Post", 0.5], ["save_reopen", 0, 0, 0], ["lang", 0, 0, 0], ["save_reopen", 0, 0, 0]]),
    ("sho: Ha retirée, rouvert", {"template": "sho", "lang": "fr"},
     [["to_palette", "Ha", 0.5, 0.5], ["save_reopen", 0, 0, 0], ["undo", 0, 0, 0]]),
    ("classic_dn: Couleur 2, pause, rouvert", {"template": "classic_dn", "lang": "en"},
     [["dbl_card", "Gradient", 0.9, 0.5], ["menu", "Gradient 2", "Review pause", 0.5], ["save_reopen", 0, 0, 0],
      ["lang", 0, 0, 0], ["undo", 0, 0, 0], ["redo", 0, 0, 0]]),
    ("deux flux: session renommée, rouvert", {"kind": "two_flows", "lang": "fr"},
     [["menu", "Session 2", "Renommer session", 0.6], ["save_reopen", 0, 0, 0], ["dbl_card", "Session", 0.5, 0.5],
      ["save_reopen", 0, 0, 0]]),
    ("mono: Fusion, Étoiles, Recombiner, rouvert", {"kind": "mono", "lang": "fr"},
     [["dbl_card", "Fusion", 0.5, 0.5], ["dbl_card", "Étoiles/fond", 0.5, 0.5], ["dbl_card", "Recombiner", 0.5, 0.5],
      ["save_reopen", 0, 0, 0], ["undo", 0, 0, 0], ["undo", 0, 0, 0]]),
]
DROPS = [
    ("session: brutes déposées sur la Session", {"kind": "session", "lang": "fr"},
     [["drop_urls", 0.0, 0.0, 0.0], ["drop_urls", 0.0, 0.0, 0.3]]),
    ("raw: dossier déposé sur la Session", {"template": "raw", "lang": "fr"}, [["drop_urls", 0.0, 0.0, 0.3]]),
]
for k in range(2):
    for title, start, acts in (DROPS if WHICH == "drop" else CASES + DROPS):
        res = fz.run({"seed": None, "start": start, "actions": acts}, f"tg{k}_{abs(hash(title)) % 1000}", watchdog=60)
        print(k, title, "|", res["steps"], "|", [(f["step"], f["sig"], f["detail"][:600]) for f in res["failures"]],
              flush=True)
        if k == 1 and title.startswith("raw"):
            fz.ex.shot("drop_files_on_session")
os._exit(0)
