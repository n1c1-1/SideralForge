"""sources_files, round 3: an empty folder / a folder without frames added to a session: what the user is told."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore3_sources_common import TMP, Explorer, _, buttons, menu_button, panel, ptexts, start, visible, wait  # noqa: E402
from PySide6.QtWidgets import QPushButton  # noqa: E402

start(300)
LOG = []


def log(*a):
    t = " ".join(str(x) for x in a)
    LOG.append(t)
    print(t, flush=True)


ex = Explorer("sources_files", lang="fr")
root = ex.project("session", "emptyfolder")
ex.open(root)
wait(ex, 0.8)
toasts = []
ex.page.toasted.connect(lambda t: toasts.append(t))
empty = TMP / "data" / "vide"
empty.mkdir(parents=True, exist_ok=True)
nofr = TMP / "data" / "textes"
nofr.mkdir(parents=True, exist_ok=True)
(nofr / "a.txt").write_text("x", encoding="utf-8")
ex.click_block(ex.block("session") or ex.block("sources"))
wait(ex, 0.4)
order = ["Lights", "Darks", "Flats", "Bias", "Flat darks"]
for kind, folder in (("Flats", empty), ("Darks", nofr)):
    bs = buttons(panel(ex), _("Add") + " ▾", QPushButton)
    ex.file_queue = [str(folder)]
    n = len(toasts)
    items = menu_button(ex, bs[order.index(kind)], _("Add folder…"))
    wait(ex, 0.8)
    rows = [b.text() for b in visible(panel(ex), QPushButton) if "📁" in b.text()]
    log(kind, folder.name, "toasts", toasts[n:], "status", ex.w.statusBar().currentMessage(), "popups", ex.popups,
        "rows", rows)
    ex.shot(f"x_empty_folder_{kind}")
(ex.out / "logs" / "extra2_log.txt").write_text("\n".join(LOG), encoding="utf-8")
print("DONE")
ex.close()
