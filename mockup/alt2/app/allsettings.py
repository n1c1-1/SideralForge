"""All the settings of a workflow on one page (expert mode): search, "changed only", level filter, comparison with
another workflow, and the whole text to copy or load back.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QHBoxLayout, QLineEdit,
                               QPlainTextEdit, QPushButton, QStackedWidget, QTabBar, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from model import Session
from settings import LEVEL_TEXT, NEW, all_settings, fmt, parse, resolve
from widgets import WARN, label

COLS = ["Étape", "Groupe", "Réglage", "Valeur", "Suggérée", "Plage", "Niveau", "État"]


def selection_rows(s: Session):
    """The two selection thresholds, set on the frames chart, as rows of the page."""
    t, f = s.thresholds_now()
    st, sf = s.suggested_thresholds()
    base = dict(kind="sel", step_key="selection", stage="stack", level="normal", choices=None, type="float", tip="",
                help="", where="", cat_stage="stack")
    return [dict(base, which="t", name="min_transparency", key="selection.transparence_min",
                 label="Transparence minimale", value=round(t, 1), suggested=round(st, 1), min=0, max=100,
                 unit="% des meilleures", changed=abs(t - st) > 1e-6),
            dict(base, which="f", name="max_fwhm", key="selection.taille_max_etoiles",
                 label="Taille maximale des étoiles", value=round(f, 2), suggested=round(sf, 2), min=1, max=20,
                 unit="px", changed=abs(f - sf) > 1e-6)]


class AllSettingsDialog(QDialog):
    def __init__(self, session: Session, view, parent=None):
        super().__init__(parent)
        self.s = session
        self.view = view
        p = session.path
        self.setWindowTitle(f"Tous les réglages · « {p.name} »")
        v = QVBoxLayout(self)
        v.addWidget(label(f"Tous les réglages du workflow « {p.name} »", "h1"))
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Chercher un réglage, une étape, une clé…")
        self.search.textChanged.connect(self.fill)
        bar.addWidget(self.search, 2)
        self.only_changed = QCheckBox("Modifiés seulement")
        self.only_changed.toggled.connect(self.fill)
        bar.addWidget(self.only_changed)
        self.level = QComboBox()
        self.level.addItems(["Tous les niveaux", "Normaux", "Expert", "Système", "À créer dans le moteur"])
        self.level.currentIndexChanged.connect(self.fill)
        bar.addWidget(self.level)
        bar.addSpacing(12)
        bar.addWidget(label("Comparer avec", "dim"))
        self.cmp = QComboBox()
        self.cmp.addItem("—", None)
        for q in session.paths:
            if q.pid != p.pid:
                self.cmp.addItem(q.name, q.pid)
        self.cmp.currentIndexChanged.connect(self.fill)
        bar.addWidget(self.cmp)
        self.only_diff = QCheckBox("Différences seulement")
        self.only_diff.toggled.connect(self.fill)
        bar.addWidget(self.only_diff)
        v.addLayout(bar)
        self.tabs = QTabBar()
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        self.tabs.addTab("Tableau")
        self.tabs.addTab("Texte")
        self.tabs.currentChanged.connect(self._tab)
        v.addWidget(self.tabs)
        self.pages = QStackedWidget()
        self.table = QTableWidget()
        self.table.setStyleSheet("QTableWidget { background: #1c1f26; gridline-color: #2a2f38; } "
                                 "QHeaderView::section { background: #23272f; border: none; padding: 4px; }")
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.table.itemChanged.connect(self._edited)
        self.table.cellDoubleClicked.connect(self._jump)
        self.pages.addWidget(self.table)
        self.text = QPlainTextEdit()
        f = QFont("Consolas")
        f.setPointSizeF(9.5)
        self.text.setFont(f)
        self.text.setStyleSheet("QPlainTextEdit { background: #12141a; border: 1px solid #353b47; "
                                "font-family: Consolas; font-size: 9.5pt; }")
        self.pages.addWidget(self.text)
        v.addWidget(self.pages, 1)
        foot = QHBoxLayout()
        self.status = label("", "small", wrap=True)
        foot.addWidget(self.status, 1)
        self.copy_b = QPushButton("Copier le texte")
        self.copy_b.clicked.connect(self._copy)
        self.apply_b = QPushButton("Appliquer le texte")
        self.apply_b.setObjectName("primary")
        self.apply_b.clicked.connect(self.apply_text)
        close = QPushButton("Fermer")
        close.clicked.connect(self.accept)
        for w in (self.copy_b, self.apply_b, close):
            foot.addWidget(w)
        v.addLayout(foot)
        self.resize(1500, 860)
        self._tab(0)

    # ------------------------------------------------------------ data
    def rows(self):
        p = self.s.path
        out = []
        for step_title, group, r in all_settings(self.s, p, self.view.steps):
            out.append((step_title, group, r))
        sel = [("Sélection des poses", "Seuils", r) for r in selection_rows(self.s)]
        i = next((k for k, x in enumerate(out) if x[2]["step_key"] == "selection"), None)
        if i is None:
            i = next((k for k, x in enumerate(out) if x[2]["step_key"] == "empilement"), len(out))
        return out[:i] + sel + out[i:]

    def other_value(self, r):
        pid = self.cmp.currentData()
        q = next((x for x in self.s.paths if x.pid == pid), None)
        if q is None:
            return None, False
        if r["kind"] == "sel":
            return r["value"], True                   # the selection is common to all workflows
        b = next((x for x in self.view.steps.steps_of(q) if x["key"] == r["step_key"]), None)
        if b is None:
            return None, False
        rq = resolve(self.s, q, r["step_key"], dict(r, stage=r["cat_stage"]))
        return (rq["value"] if rq else None), rq is not None

    def state(self, r):
        if r["kind"] == "new":
            return "à créer" + (" · modifié" if r["changed"] else "")
        return "modifié" if r["changed"] else "suggéré"

    # ------------------------------------------------------------ table
    def fill(self):
        if self.tabs.currentIndex() == 1:
            self.text.setPlainText(self.as_text())
            return
        q = self.search.text().strip().lower()
        lv = self.level.currentIndex()
        comparing = self.cmp.currentData() is not None
        self.only_diff.setEnabled(comparing)
        cols = COLS + ([f"« {self.cmp.currentText()} »"] if comparing else [])
        shown = []
        for step_title, group, r in self.rows():
            if q and q not in " ".join((step_title, group, r["label"], r["key"])).lower():
                continue
            if self.only_changed.isChecked() and not r["changed"]:
                continue
            if lv == 1 and r["level"] != "normal" or lv == 2 and r["level"] != "expert" or \
                    lv == 3 and r["level"] != "system" or lv == 4 and r["kind"] != "new":
                continue
            other, present = self.other_value(r) if comparing else (None, False)
            diff = comparing and (not present or other != r["value"])
            if comparing and self.only_diff.isChecked() and not diff:
                continue
            shown.append((step_title, group, r, other, present, diff))
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(shown))
        self._rows = []
        for i, (step_title, group, r, other, present, diff) in enumerate(shown):
            rng = (f"{fmt(r['min'])} à {fmt(r['max'])}" if r.get("min") is not None and r.get("max") is not None
                   else (" / ".join(c for _, c in r["choices"]) if r.get("choices") else ""))
            val = fmt(r["value"], r["type"], r.get("choices")) + (f" {r['unit']}" if r.get("unit") and
                                                                  r["type"] in ("int", "float") else "")
            cells = [step_title, group, r["label"], val, fmt(r["suggested"], r["type"], r.get("choices")), rng,
                     LEVEL_TEXT[r["level"]], self.state(r)]
            if comparing:
                cells.append(fmt(other, r["type"], r.get("choices")) if present else "pas dans ce workflow")
            for j, c in enumerate(cells):
                it = QTableWidgetItem(c)
                if j != 3 or r["type"] == "zone":
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if j == 3:
                    it.setToolTip("Double-clic pour modifier")
                    it.setText(fmt(r["value"], r["type"], r.get("choices")))
                if j == 2:
                    it.setToolTip(f"{r['key']} · double-clic sur l'étape pour l'ouvrir")
                if r["changed"] and j in (3, 7):
                    it.setForeground(QColor(WARN))
                if r["kind"] == "new" and j in (2, 7):
                    it.setForeground(QColor(NEW))
                if comparing and j == len(cells) - 1 and diff:
                    it.setForeground(QColor(WARN))
                self.table.setItem(i, j, it)
            self._rows.append(r)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        for j, cap in enumerate((150, 170, 330, 230, 230, 240, 80, 110)):
            if j < self.table.columnCount():
                self.table.setColumnWidth(j, min(self.table.columnWidth(j) + 12, cap))
        self.table.horizontalHeader().setStretchLastSection(True)
        n_all = len(self.rows())
        n_chg = sum(r["changed"] for _, _, r in self.rows())
        n_new = sum(r["kind"] == "new" for _, _, r in self.rows())
        self.status.setText(f"{len(shown)} affichés sur {n_all} · {n_chg} modifiés · {n_new} à créer dans le "
                            "moteur. Double-clic sur une valeur pour la modifier, sur l'étape pour l'ouvrir.")

    def _edited(self, item):
        if item.column() != 3:
            return
        r = self._rows[item.row()]
        try:
            val = parse(item.text(), r)
        except ValueError as e:
            self.status.setText(f"<span style='color:{WARN}'>{r['label']} : {e}.</span>")
            self.fill()
            return
        self.set_value(r, val)
        self.fill()

    def set_value(self, r, val):
        if val == r["value"]:
            return
        if r["kind"] == "sel":
            self.s.set_selection(tmin=val if r["which"] == "t" else None, fmax=val if r["which"] == "f" else None)
        else:
            self.view.set_setting(r, val)

    def _jump(self, row, col):
        if col == 3:
            return
        r = self._rows[row]
        self.s.select(step=r["step_key"])
        self.view.set_face(0)
        self.accept()

    # ------------------------------------------------------------ text
    def _tab(self, i):
        self.pages.setCurrentIndex(i)
        self.apply_b.setVisible(i == 1)
        self.copy_b.setVisible(i == 1)
        for w in (self.only_changed, self.level, self.only_diff):
            w.setEnabled(i == 0 and (w is not self.only_diff or self.cmp.currentData() is not None))
        self.fill()
        if i == 1:
            self.status.setText("Une ligne par réglage : clé = valeur. Modifie les valeurs, puis « Appliquer le "
                                "texte ». Les virgules et les points décimaux sont acceptés.")

    def as_text(self) -> str:
        p = self.s.path
        lines = [f"# Workflow « {p.name} » : tous les réglages, une ligne par réglage.",
                 "# clé = valeur   # nom · * = modifié · (à créer) = pas encore dans le moteur", ""]
        cur = None
        for step_title, group, r in self.rows():
            if step_title != cur:
                cur = step_title
                lines += ["", f"[{step_title}]"]
            v = r["value"]
            val = ("" if v == "" else f"{v:.10g}".replace(".", ",") if isinstance(v, float) else
                   str(v) if isinstance(v, int) and not isinstance(v, bool) else fmt(v, r["type"], r.get("choices")))
            note = r["label"] + (" *" if r["changed"] else "") + (" (à créer)" if r["kind"] == "new" else "")
            lines.append(f"{r['key']} = {val}".ljust(58) + f"# {note}")
        return "\n".join(lines).replace("\n\n\n", "\n\n") + "\n"

    def _copy(self):
        QGuiApplication.clipboard().setText(self.text.toPlainText())
        self.status.setText("Texte copié.")

    def apply_text(self):
        by_key = {r["key"]: r for _, _, r in self.rows()}
        changed, errors = [], []
        for n, line in enumerate(self.text.toPlainText().splitlines(), 1):
            body = line.split("#", 1)[0].strip()
            if not body or body.startswith("["):
                continue
            if "=" not in body:
                errors.append(f"ligne {n} : pas de « = »")
                continue
            key, val = (x.strip() for x in body.split("=", 1))
            r = by_key.get(key)
            if r is None:
                errors.append(f"ligne {n} : réglage inconnu « {key} »")
                continue
            if r["type"] == "zone":
                continue
            try:
                x = parse(val, r)
            except ValueError as e:
                errors.append(f"ligne {n} ({r['label']}) : {e}")
                continue
            same = x == r["value"] or (isinstance(x, float) and isinstance(r["value"], (int, float))
                                       and not isinstance(r["value"], bool)
                                       and abs(x - r["value"]) <= 1e-9 * max(1.0, abs(r["value"])))
            if not same:
                changed.append((r, x))
        for r, x in changed:
            self.set_value(r, x)
        self.text.setPlainText(self.as_text())
        msg = f"{len(changed)} réglage{'s' if len(changed) != 1 else ''} appliqué{'s' if len(changed) != 1 else ''}."
        if errors:
            msg += f" <span style='color:{WARN}'>{len(errors)} ligne(s) ignorée(s) : " + " ; ".join(errors[:3]) + \
                   ("…" if len(errors) > 3 else "") + "</span>"
        self.status.setText(msg)

