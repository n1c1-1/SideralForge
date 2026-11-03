"""Layout and export details of the client: FITS export WCS and header, histogram ticks, widths that never grow.

Offscreen, on scratch data only (never F:/astro_projets, never the user's settings)."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")


def _wcs_file(path: Path) -> Path:
    """A header-only wcs.fits as the astrometry stage writes it (NAXIS = 0, a TAN solution)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.68, 41.27]
    w.wcs.crpix = [600.0, 400.0]
    w.wcs.cd = np.array([[-2.6e-4, 1e-6], [1e-6, 2.6e-4]])
    hdu = fits.PrimaryHDU(header=w.to_header())
    hdu.writeto(path)
    assert fits.getheader(path)["NAXIS"] == 0
    return path


def test_the_fits_export_builds_its_wcs_without_the_astropy_warning(tmp_path):
    """The FITS export read the header-only wcs.fits as it is: astropy warned on every export that the WCS has more
    axes than the image (NAXIS = 0). The export now gets the header without the empty image axes: same WCS written, no
    warning."""
    from astropy.io import fits

    from astro_gpu.client.services.render import export_header
    from astro_gpu.render.layers import oriented_wcs
    f = _wcs_file(tmp_path / "wcs.fits")
    with warnings.catch_warnings(record=True) as before:
        warnings.simplefilter("always")
        old = oriented_wcs(fits.getheader(f), 800, 1200, (10, 20), 1, True, 0.5)
    assert any("more axes" in str(x.message) for x in before)               # what the export used to note
    with warnings.catch_warnings(record=True) as after:
        warnings.simplefilter("always")
        new = oriented_wcs(export_header(f), 800, 1200, (10, 20), 1, True, 0.5)
    assert not [x for x in after if "more axes" in str(x.message)]
    assert dict(new.items()) == dict(old.items())
    h = export_header(str(f))
    assert "NAXIS" not in h and h["CTYPE1"] == "RA---TAN"
    assert export_header(None) is None and export_header(h) is h               # a header goes through


def test_the_render_worker_writes_the_header_run_export_read(tmp_path, monkeypatch):
    """The worker's writer handed the engine only the workflow ("wf"): the engine read the header-only wcs.fits again
    (the warning), and the layers exported one by one (no "wf") had no WCS at all. It now hands the header."""
    from astro_gpu import api
    from astro_gpu.client.services.render import RenderWorker, export_header
    seen = {}
    monkeypatch.setattr(api, "export_image", lambda p, rgb, m, opts: seen.update(m) or [])
    h = export_header(_wcs_file(tmp_path / "wcs.fits"))

    class Fake:
        project = None
    RenderWorker._write(Fake(), None, tmp_path / "out", {}, {"wcs": h, "origin": (0, 0), "wf": 3})
    assert seen["wcs"] is h and "wf" not in seen
    seen.clear()
    RenderWorker._write(Fake(), None, tmp_path / "out", {}, {"wcs": None, "origin": (0, 0), "wf": 3})
    assert seen["wf"] == 3 and "wcs" not in seen                             # no header read: the engine finds it


def test_histogram_tick_labels_never_touch():
    """The stretch histogram's labels: each at least its half width plus the next one's and 8 px from the previous
    (a fixed 34 px let wide labels such as "-1000" and "-500" run into each other)."""
    from astro_gpu.client.imageview import spaced_ticks

    def width(t):
        return 7.0 * len(t)                                                   # 7 px a character
    ticks = [(10, "-1000"), (46, "-500"), (80, "-200"), (120, "0"), (150, "100"), (160, "200"), (400, "5000")]
    kept = spaced_ticks(ticks, width, 0, 300)
    assert kept == [(10, "-1000"), (80, "-200"), (120, "0"), (160, "200")]    # 5000 is outside
    for (x0, a), (x1, b) in zip(kept, kept[1:]):
        assert x1 - x0 >= width(a) / 2 + width(b) / 2 + 8 - 1e-9 and x1 - x0 >= 34


def test_a_tiny_rejection_share_keeps_its_figures():
    """Inspect › Stacking › Rejection map said "Rejected pixels: 0.00 %" where the Measurements said 0.0010 %."""
    from types import SimpleNamespace

    from astro_gpu.client import i18n
    from astro_gpu.client.tune import rejection_rate_text
    i18n.set_language("en")
    t = rejection_rate_text(SimpleNamespace(rejection={"rejected_pct": 0.001}))
    assert "Rejected pixels: 0.0010 %" in t and "below it" in t
    assert "Rejected pixels: 0.30 %" in rejection_rate_text(SimpleNamespace(rejection={"rejected_pct": 0.3}))
    assert "Rejected pixels: 12.5 %" in rejection_rate_text(SimpleNamespace(rejection={"rejected_pct": 12.5}))
    assert "Rejected pixels: 0.00 %" in rejection_rate_text(SimpleNamespace(rejection={"rejected_pct": 0.0}))


def test_the_frame_facts_never_widen_the_window():
    """The frame's facts under the viewer (name, time, exposure… Included, Exclude) made the Inspect window at least
    1960 px wide; they now scroll sideways in a narrow window."""
    from PySide6.QtWidgets import QApplication

    from astro_gpu.client.imageview import ImageView, SideScroll
    QApplication.instance() or QApplication([])
    v = ImageView()
    band = v.band
    assert isinstance(band.parentWidget().parentWidget(), SideScroll)            # band -> viewport -> scroll area
    assert v.seq.minimumSizeHint().width() < 900 < band.minimumSizeHint().width() + 400
    v.resize(800, 600)
    v.seq.show()
    v.show()
    QApplication.processEvents()
    sa = band.parentWidget().parentWidget()
    assert sa.horizontalScrollBar().maximum() > 0                              # it scrolls rather than widens
    assert not sa.isAncestorOf(band.toggle) and band.toggle.isVisibleTo(v)    # Exclude stays in view
    from PySide6.QtWidgets import QFrame
    bar = next(x for x in v.findChildren(QFrame) if x.objectName() == "ivbar")   # the tools too (1771 px in French)
    assert isinstance(bar.parentWidget().parentWidget(), SideScroll)
    assert v.minimumSizeHint().width() < 900                                   # was 1340 and more
    v.close()


def test_the_stretch_panel_scrolls_instead_of_squeezing():
    """In a short viewer the stretch panel's rows ran into each other (the histogram's numbers over Black / Mid /
    White); it now sits in a column that scrolls, and shows and hides with the Histogram button."""
    from PySide6.QtWidgets import QApplication

    from astro_gpu.client.imageview import ImageView
    QApplication.instance() or QApplication([])
    v = ImageView(histogram=False)
    v.resize(900, 300)
    v.show()
    v.show_histogram(True)
    QApplication.processEvents()
    assert v.panel_box.isVisibleTo(v) and v.panel.isVisibleTo(v)
    assert v.panel.height() >= v.panel.minimumSizeHint().height()               # never squeezed
    assert v.panel_box.verticalScrollBar().maximum() > 0                        # it scrolls
    v.show_histogram(False)
    assert not v.panel_box.isVisibleTo(v) and not v.panel.isVisibleTo(v)
    v.close()


def test_the_curves_channel_button_is_translated():
    """Post › Curves showed "RGB" in French (RVB)."""
    from PySide6.QtWidgets import QApplication

    from astro_gpu.client import i18n
    from astro_gpu.client.curveseditor import CurvesEditor
    QApplication.instance() or QApplication([])
    i18n.set_language("fr")
    try:
        ed = CurvesEditor({}, None)
        assert ed.ch_buttons["RGB/K"].text() == "RVB"
        ed.graph.add(0.3, 0.4)
        ed._marks()
        assert ed.ch_buttons["RGB/K"].text() == "RVB •"
    finally:
        i18n.set_language("en")
    assert CurvesEditor({}, None).ch_buttons["RGB/K"].text() == "RGB"


def test_a_long_compare_name_never_widens_the_toolbar():
    """Inspect › Colour in French: "Avant : sortie de l'empilement, couleurs de l'appareil" made the viewer's toolbar,
    so the window, 1771 px wide at least. The list keeps a short width and opens wide enough for the whole names."""
    from PySide6.QtWidgets import QApplication

    from astro_gpu.client.imageview import ImageView, ViewImage
    QApplication.instance() or QApplication([])
    v = ImageView()
    short = v.cmp_cb.minimumSizeHint().width()
    long_name = "Avant : sortie de l'empilement, couleurs de l'appareil, encore plus long"
    v.set_compare_targets([ViewImage(np.zeros((4, 4), np.float32), name=long_name)], 0)
    assert v.cmp_cb.minimumSizeHint().width() == short <= 300
    assert v.cmp_cb.view().minimumWidth() >= v.cmp_cb.view().sizeHintForColumn(0)
    assert v.cmp_cb.toolTip() == long_name
    v.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
