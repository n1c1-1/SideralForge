"""The Report tab and its night card (issue #81), and the mask brush's size (issue #80)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client import report as rep  # noqa: E402
from astro_gpu.client.charts import RegistrationData  # noqa: E402
from astro_gpu.client.chartview import ChartView  # noqa: E402
from astro_gpu.client.post3 import BRUSH_MAX, BRUSH_MIN, brush_pos, brush_px  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class Service(QObject):
    jobReady = Signal(str, int, object)
    jobFailed = Signal(str, int, str)

    def request_job(self, name, fn, *args):
        return 1


class Steps(QObject):
    changed = Signal()
    service = Service()

    def facts(self, p, key):
        return [("Value of " + key, "1")]


def data(n=12, timed=True, figures=True):
    t = 1000.0 + 15.0 * np.arange(n) if timed else np.full(n, np.nan)
    reg = RegistrationData(names=[f"f{i}" for i in range(n)], iso=[""] * n, secs=t, fwhm=np.full(n, 9.0),
                           transp=np.linspace(70, 100, n), nmatch=np.full(n, 50.0), nstars=np.full(n, 80.0),
                           rms=np.full(n, 0.2), aligned=np.ones(n, bool), dx=np.linspace(0, 5, n), dy=np.zeros(n), ref=2,
                           scale=2.0, roundness=np.full(n, 0.5))
    kept = np.ones(n, bool)
    kept[0] = False
    d = rep.ReportData(reg=reg, sessions=[("Night 1", list(range(6))), ("Night 2", list(range(6, n)))], kept=kept,
                       fwhm=np.linspace(2.0, 3.0, n), roundness=np.full(n, 0.9), sky=np.full(n, 800.0),
                       exposure=np.full(n, 15.0), settings=[("Stacking", "Kappa", 5.0, True), ("Stacking", "Method", "drizzle", False)],
                       astro=dict(focal_estimate_mm=272.4, field_deg=[7.33, 5.04]))
    if figures:
        rep.draw_figures(d, "M 45")
    return d


def page(monkeypatch, state="done", d=None):
    q = SimpleNamespace(pid=1, name="Full")
    s = SimpleNamespace(paths=[q], all_stages=lambda p: ("stack",), info=lambda: dict(camera="A7S", n_lights=12, iso_gain=5000.0, n_darks=48, n_flats=62),
                        p=SimpleNamespace(acquisition={}), target="M 45", name="Pleiades", block_keys=lambda p: ["alignement", "empilement"])
    monkeypatch.setattr(rep, "block_state", lambda *_a: state)
    p = rep.ReportPage(s, Steps())
    p.refresh()
    p.data = d if d is not None else data()
    p.build()
    return p


def text(p) -> str:
    return p.document().toPlainText()


def test_the_sizes_are_in_arcseconds_from_the_moffat_fit_and_the_kept_frames(qapp):
    rows = dict(rep.card_rows(data(figures=False), list(range(6))))
    assert rows["Frames kept"] == "5 / 6"
    assert rows["Total exposure"] and rows["Best FWHM"] == '4.2"'           # 2.0 x the 2.09 px of frame 1, the first kept


def test_the_card_counts_time_from_the_first_frame_never_the_clock(qapp):
    d = data(figures=False)
    x, minutes = rep.elapsed(d.reg, list(range(6)))
    assert minutes and x[0] == 0.0 and abs(x[-1] - 1.25) < 1e-9


def test_a_card_without_times_counts_the_frames(qapp):
    d = data(timed=False, figures=False)
    x, minutes = rep.elapsed(d.reg, list(range(6)))
    assert not minutes and list(x) == [1, 2, 3, 4, 5, 6]


def test_the_report_is_numbered_gives_each_block_and_the_recipe_and_a_stale_result_says_so(qapp, monkeypatch):
    t = text(page(monkeypatch, "stale"))
    assert "Processing report" in t and "Table 1. Acquisition" in t and "1. Registration" in t and "2. Stacking" in t
    assert "3. Recipe" in t and "4. Appendix" in t and "Value of alignement" in t and "Registration → Stacking" in t
    assert t.count("Outdated") == 2


def test_each_figure_has_its_number_and_caption(qapp, monkeypatch):
    t = text(page(monkeypatch))
    assert "Figure 1." in t and "Figure 2." in t and "Figure 3." in t and "Open circles" in t


def test_a_duration_is_exact_and_a_star_size_has_its_pixels():
    assert rep.exact_time(450.0) == "7 min 30 s" and rep.exact_time(3900.0) == "1 h 05" and rep.exact_time(480.0) == "8 min"
    assert rep.size_text(1.68, 6.37, '"', lambda x: f"{x:.1f}") == '10.7" (1.7 px)'


def test_the_english_text_has_no_space_before_its_colons(qapp, monkeypatch):
    p = page(monkeypatch)
    p.lang.setCurrentIndex(p.lang.findData("en"))
    assert "Camera: A7S" in p.details()


def test_every_part_can_be_copied_as_a_picture(qapp, monkeypatch):
    from PySide6.QtWidgets import QApplication
    p = page(monkeypatch)
    assert p.part.count() >= 5 and p.part.itemText(0).startswith("Night card")
    for i in range(p.part.count()):
        p.part.setCurrentIndex(i)
        p.copy_image()
        assert not QApplication.clipboard().image().isNull()


def test_of_the_two_stacking_methods_only_the_one_that_ran_is_listed(qapp, monkeypatch):
    d = data()
    d.settings = [("Stacking", "Stacking method", "drizzle", False), ("Stacking", "Rejection (Classic)", "auto", False),
                  ("Stacking", "Rejection (drizzle)", "own", False)]
    t = text(page(monkeypatch, d=d))
    assert "Rejection (drizzle)" in t and "Rejection (Classic)" not in t


def test_the_report_uses_none_of_the_interactive_charts(qapp, monkeypatch):
    assert not page(monkeypatch).findChildren(ChartView)


def test_a_block_the_workflow_does_not_have_is_not_in_its_report(qapp, monkeypatch):
    assert "Value of fond" not in text(page(monkeypatch))


def test_nothing_computed_says_so(qapp, monkeypatch):
    p = page(monkeypatch, d=rep.ReportData())
    assert "Nothing computed yet" in text(p) and not p.card_ready()


@pytest.mark.parametrize("value,ok", [(5.0, True), ("drizzle", True), (r"F:\photos\raw", False), ("C:/a/b", False),
                                      ("/home/me/x.fits", False), (["a"], False), (None, False)])
def test_a_setting_that_names_a_place_stays_out(value, ok):
    assert rep.shareable(value) is ok


def test_the_exports_hold_the_report_and_no_path(qapp, monkeypatch, tmp_path):
    p = page(monkeypatch)
    assert not rep.PATHS.search(text(p))
    p.save_png(tmp_path / "r.png")
    p.save_pdf(tmp_path / "r.pdf")
    assert (tmp_path / "r.png").stat().st_size > 2000 and (tmp_path / "r.pdf").read_bytes()[:4] == b"%PDF"
    # the night test pass: 7 s without a sign, the file proposed in the program's own folder
    said, asked = [], []
    p.said.connect(said.append)
    monkeypatch.setattr(p.s, "root", tmp_path, raising=False)
    monkeypatch.setattr(rep.QFileDialog, "getSaveFileName",
                        lambda *a: (asked.append(a[2]), ((tmp_path / "x.pdf").as_posix(), ""))[1])
    p._ask_pdf()
    assert asked[0].startswith(tmp_path.as_posix()) and asked[0].endswith("_report.pdf")
    assert said[0].startswith("Writing x.pdf") and "x.pdf" in said[-1] and (tmp_path / "x.pdf").is_file()


def test_one_night_card_per_session_beside_the_image(qapp, monkeypatch, tmp_path):
    p = page(monkeypatch)
    assert p.card_ready()
    names = sorted(f.name for f in p.save_cards(tmp_path, "m45"))
    assert names == ["m45_night_card_1.png", "m45_night_card_2.png"]
    assert all((tmp_path / n).stat().st_size > 2000 for n in names)


def test_the_brush_goes_to_1200_px_and_small_sizes_stay_on_the_slider():
    assert BRUSH_MAX == 1200.0 and brush_px(0.0) == BRUSH_MIN and brush_px(1.0) == 1200
    assert brush_px(brush_pos(60.0)) == 60
    assert brush_pos(60.0) > 0.4                         # not squeezed at the start of the rail


# ------------------------------------------------------------ the shooting details
def test_the_text_to_paste_says_what_is_known_in_the_language_of_the_forum(qapp, monkeypatch):
    p = page(monkeypatch)
    p.lang.setCurrentIndex(p.lang.findData("fr"))
    t = p.details()
    assert t.startswith("M 45") and "Appareil : A7S · ISO 5000" in t and "Poses : 11 × 15 s" in t and "Darks : 48" in t
    assert "Champ : 7,3° × 5,0°" in t and "Échantillonnage : 2,0\"/px" in t and "Traitement : FLOAT" in t
    assert "Monture" not in t and "Date" not in t                                      # nobody said
    p.lang.setCurrentIndex(p.lang.findData("en"))
    assert "Field: 7.3° × 5.0°" in p.details() and "Processing: FLOAT" in p.details()


def test_the_camera_clock_never_gives_the_date(qapp, monkeypatch):
    p = page(monkeypatch)
    assert not [r for r in rep.shot_rows(p.data, p.s.info(), {}, "en") if r[0] == "date" and r[2]]


def test_a_field_is_kept_in_the_project_and_goes_in_the_report(qapp, monkeypatch):
    p = page(monkeypatch)
    p.fields["instrument"].setText("Newton 130/900")
    p.set_field("instrument")
    assert p.s.p.acquisition == {"instrument": "Newton 130/900"} and "Newton 130/900" in text(p)
    assert "Instrument: Newton 130/900" in p.details()


def test_a_field_left_empty_is_forgotten(qapp, monkeypatch):
    p = page(monkeypatch)
    p.s.p.acquisition["mount"] = "EQ6"
    p.fields["mount"].setText("  ")
    p.set_field("mount")
    assert p.s.p.acquisition == {}


def test_an_empty_field_is_not_in_the_report(qapp, monkeypatch):
    assert "Mount" not in text(page(monkeypatch))


def test_every_section_says_what_it_is_for_and_every_figure_what_it_means(qapp, monkeypatch):
    p = page(monkeypatch)
    monkeypatch.setattr(p.steps, "facts", lambda q, key: [("Aligned frames", "30")], raising=False)
    t = text(p)
    assert "by matching their stars" in t and "Frames that could be aligned." in t


def test_the_reader_zooms_and_comes_back_to_the_width(qapp, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from PySide6.QtPdfWidgets import QPdfView
    p = page(monkeypatch)
    p.resize(1200, 800)
    QApplication.processEvents()
    assert p.view.zoomMode() == QPdfView.ZoomMode.FitToWidth
    p.zoom_by(1.5)
    assert p.view.zoomMode() == QPdfView.ZoomMode.Custom
    was = p.view.zoomFactor()
    p.zoom_by(1.5)
    assert abs(p.view.zoomFactor() / was - 1.5) < 1e-6
    p.fit_width()
    assert p.view.zoomMode() == QPdfView.ZoomMode.FitToWidth


def test_the_text_and_the_curves_go_to_the_clipboard(qapp, monkeypatch):
    from PySide6.QtWidgets import QApplication
    p = page(monkeypatch)
    p.copy_details()
    assert QApplication.clipboard().text().startswith("M 45")


def test_the_recipe_gives_only_what_differs_from_the_defaults_and_the_appendix_all(qapp, monkeypatch):
    t = text(page(monkeypatch))
    recipe, appendix = t.split("4. Appendix")
    assert "Kappa" in recipe and "Method" not in recipe and "Method" in appendix


def test_the_details_stay_in_the_project_file(tmp_path):
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    p = project2.create(tmp_path / "A", "A", Target())
    p.acquisition["mount"] = "EQ6"
    project2.save(p)
    assert project2.load(tmp_path / "A").acquisition == {"mount": "EQ6"}


def test_the_figures_have_a_decimal_comma_in_french():
    """The night test pass: the night card's axis read 0.5, 1.0, 1.5 where the rest of the program writes 0,5."""
    from matplotlib.figure import Figure
    i18n.set_language("fr")
    try:
        fig = Figure()
        ax = fig.add_subplot()
        ax.plot([0, 0.5, 1.5], [1, 2, 3])
        rep._png(fig)
        assert ax.xaxis.get_major_formatter()(0.5, 0) == "0,5"
    finally:
        i18n.set_language("en")


def test_the_session_card_quotes_only_what_was_measured():
    """The head of the report, made to be posted: its figures come from this session's results, and say how."""
    assert rep.sky_position(56.76679, 24.05996) == "03h47m04s +24°03′36″"
    assert rep.sky_position(83.82, -5.39) == "05h35m17s −05°23′24″"
    d = rep.ReportData(astro={"focal_estimate_mm": 893.7, "stars_used": 242, "residual_px": 0.189, "scale_arcsec": 1.9386},
                       metrics={"register": {"drift_px": 24.8, "transparency_p10_p90": [0.74, 1.02]},
                                "inventory": {"first_time": "2023-08-21T00:56:30", "last_time": "2023-08-21T00:59:20"},
                                "denoise": {"noise_reduction": 14.81}, "catalogs": {"stars_deep": 3203, "xp_spectra": 609},
                                "stack": {"channel_offsets": {"applied": True, "stars": 253,
                                                              "centre_px": {"R": [-0.471, 0.289], "B": [0.513, -0.304]}}}})
    got = {t: (v, h) for v, t, h in rep.highlights(d, "en")}
    assert got["Focal length, measured"][0] == "894 mm" and "242 stars, residual 0.37″" in got["Focal length, measured"][1]
    assert got["Field drift"][0] == "48″" and "17.0″ a minute" in got["Field drift"][1]
    assert got["Colours offset, corrected"][0] == "0.60 px" and got["Sky noise"][0] == "÷ 14.8"
    assert got["Gaia DR3 stars in the field"][0] == "3 203" and got["Transparency changes"][0] == "27 %"
    assert not rep.highlights(rep.ReportData(), "en")                      # nothing measured: nothing quoted
