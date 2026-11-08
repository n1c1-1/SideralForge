"""The icons module (icons.py): every role has a file that renders, the colours come from the theme, nothing else."""
from __future__ import annotations

import re

import pytest
from PySide6.QtCore import QByteArray
from PySide6.QtGui import QColor
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QPushButton, QToolButton

from astro_gpu.client import icons, widgets


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_every_role_has_a_file_and_every_file_a_role():
    files = {f.stem for f in icons.FOLDER.glob("*.svg")}
    assert files == set(icons.ROLES)
    assert len(icons.ROLES) == len(icons.CARBON) + len(set(r for r in icons.BLOCK_ROLES.values() if r.startswith("block_")))
    assert all(r in icons.ROLES for r in icons.BLOCK_ROLES.values())
    assert (icons.FOLDER / "LICENSE-carbon.txt").exists()


@pytest.mark.parametrize("role", sorted(set(icons.CARBON) | {r for r in icons.BLOCK_ROLES.values() if r.startswith("block_")}))
def test_every_svg_loads_and_paints(role, qapp):
    svg = icons.svg_text(role, "#123456")
    assert QSvgRenderer(QByteArray(svg.encode("utf-8"))).isValid()
    img = icons.pixmap(role, 20, "normal", 1.0).toImage()
    assert any(img.pixelColor(x, y).alpha() for x in range(20) for y in range(20))


def test_files_hold_no_colour_and_are_recoloured_in_the_text():
    for role in icons.ROLES:
        src = (icons.FOLDER / f"{role}.svg").read_text(encoding="utf-8")
        assert "currentColor" in src
        assert not re.search(r"#[0-9a-fA-F]{3,6}\b", src), role           # no colour of its own, ever
        assert "<style" not in src and 'class="' not in src, role
    text = icons.svg_text("add", widgets.ACCENT)
    assert "currentColor" not in text and widgets.ACCENT in text


def test_states_use_the_theme_constants_and_two_sizes_only(qapp):
    assert icons.COLOURS["selected"] == widgets.ACCENT and icons.COLOURS["error"] == widgets.BAD
    assert icons.COLOURS["warning"] == widgets.WARN and icons.COLOURS["disabled"] == widgets.DISABLED
    assert icons.SIZES == (16, 20)
    with pytest.raises(ValueError):
        icons.pixmap("add", 24)
    with pytest.raises(KeyError):
        icons.pixmap("no_such_role")
    with pytest.raises(KeyError):
        icons.pixmap("add", 16, "no_such_state")


def test_a_rendering_is_cached_and_sharp_at_its_ratio(qapp):
    a = icons.pixmap("undo", 16, "error", 2.0)
    assert a.width() == 32 and a.devicePixelRatio() == 2.0
    assert icons.pixmap("undo", 16, "error", 2.0).cacheKey() == a.cacheKey()
    b = icons.pixmap("undo", 16, "dim", 2.0)
    assert b.cacheKey() != a.cacheKey() and b.toImage().pixelColor(*_ink(b)) != a.toImage().pixelColor(*_ink(a))
    assert icons.pixmap("undo", 16, "error", 1.0).width() == 16


def _ink(pm) -> tuple[int, int]:
    """A pixel the icon covers fully."""
    img = pm.toImage()
    best, at = 0, (0, 0)
    for x in range(img.width()):
        for y in range(img.height()):
            al = img.pixelColor(x, y).alpha()
            if al > best:
                best, at = al, (x, y)
    return at


def test_the_colour_of_a_state_is_what_is_drawn(qapp):
    pm = icons.pixmap("add", 20, "error", 2.0)
    c = pm.toImage().pixelColor(*_ink(pm))
    want = QColor(widgets.BAD)
    assert abs(c.red() - want.red()) <= 2 and abs(c.green() - want.green()) <= 2 and abs(c.blue() - want.blue()) <= 2


def test_a_control_says_its_name_and_its_role(qapp):
    b = icons.icon_only(QPushButton("x"), "undo", "Undo (Ctrl+Z)", box=24)
    assert b.text() == "" and b.toolTip() == "Undo (Ctrl+Z)" and b.property("iconRole") == "undo"
    assert b.iconSize().width() == 16 and b.width() == 24 and not b.icon().isNull()
    t = icons.set_icon(QToolButton(), "add", 20)
    assert t.iconSize().width() == 20 and t.property("iconRole") == "add"


def test_the_about_box_credits_the_icons(qapp):
    from astro_gpu.client import credits
    text = credits.icons_html()
    assert "Apache" in text and "IBM Carbon" in text and "side-panel--open" in text
    assert "icons" in credits.data_sources_html().lower()
