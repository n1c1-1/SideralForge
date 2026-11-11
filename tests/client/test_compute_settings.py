"""Settings > Compute: the card test's result per block, the button that tests again, the beta cards, the translations
of the new words; the test itself replaced by a function (core/selftest.py is tested in tests/test_selftest.py).
Offscreen, scratch settings file."""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QPushButton  # noqa: E402

from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.services import selftest as jobmod  # noqa: E402
from astro_gpu.client.services.run import engine_text  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402
from astro_gpu.client.settingsdlg import SettingsDialog, device_items  # noqa: E402
from astro_gpu.core import selftest as st  # noqa: E402

MACHINE = Machine([Gpu("cuda:0", "NVIDIA RTX 4060 Ti", 16.0)], 16, 64.0)
BETA = Machine([Gpu("xpu:0", "Intel Arc A770", 8.0, beta=True)], 16, 64.0)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    jobmod._JOB.clear()
    yield tmp_path
    i18n.set_language("en")
    jobmod._JOB.clear()


def _result(name, verdict, card=0.1, cpu=1.0):
    return {"op": name, "stage": st.OPERATIONS[name], "verdict": verdict, "card_s": card, "cpu_s": cpu, "diff": 0.0,
            "tolerance": 1e-3, "error": ""}


def _profile(**verdicts):
    res = [_result(n, verdicts.get(n, st.OK)) for n in st.OPERATIONS]
    return st.make_profile("cuda:0", "RTX 4060 Ti", "610.88", res, auto=True)


def _dialog(qapp, machine=MACHINE):
    d = SettingsDialog(None, machine)
    d.show()
    d.show_page("compute")
    qapp.processEvents()
    return d, d.pages.currentWidget()


def _texts(page):
    return [w.text() for w in page.findChildren(QLabel) if w.isVisibleTo(page) and w.text()]


def test_the_page_says_each_block_and_the_card(env, qapp):
    prefs.set("compute_profiles", {"auto": _profile(sampling=st.WRONG, post=st.SLOWER)})
    d, page = _dialog(qapp)
    texts = _texts(page)
    assert any("RTX 4060 Ti" in t and "610.88" in t for t in texts)
    assert "CPU · wrong result" in texts                                  # the Stack block
    assert "CPU · slower" in texts                                        # Post
    assert any(t.startswith("GPU · ") and "faster" in t for t in texts)   # Denoise, Stars
    d.close()


def test_the_page_before_any_test(env, qapp):
    d, page = _dialog(qapp)
    assert "Not tested yet" in _texts(page)
    d.close()


def test_the_cpu_setting_has_nothing_to_test(env, qapp):
    prefs.set("device", "cpu")
    d, page = _dialog(qapp)
    b = next(x for x in page.findChildren(QPushButton) if x.text() == "Test again")
    assert not b.isEnabled() and "The processor computes: nothing to test" in _texts(page)
    # the night test pass: four empty lines, and the greyed button did not say why
    assert "nothing to test" in b.toolTip()
    assert not any(lab.isVisibleTo(page) for lab in d.ct["rows"].values())
    d.close()


def test_the_button_tests_in_the_background_and_saves_the_profile(env, qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(jobmod.st, "card_identity", lambda dev: ("RTX 4060 Ti", "610.88"))

    def fake_runner(device, on_proc=None):
        calls.append(device)
        time.sleep(0.05)
        return [_result(n, st.OK) for n in st.OPERATIONS]

    jobmod.job().runner = fake_runner
    d, page = _dialog(qapp)
    b = next(x for x in page.findChildren(QPushButton) if x.text() == "Test again")
    b.click()                                                            # never blocks: the job is a thread
    assert jobmod.job().running()
    qapp.processEvents()
    assert not b.isEnabled()
    t0 = time.time()
    while jobmod.job().running() and time.time() - t0 < 5:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert calls == ["cuda:0"]
    p = prefs.get("compute_profiles")
    assert p["auto"]["stages"]["stack"] == "card" and p["cuda:0"]["card"] == "RTX 4060 Ti"
    assert b.isEnabled() and any("610.88" in t for t in _texts(page))
    d.close()


def test_a_valid_profile_is_not_tested_again(env, qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(jobmod.st, "card_identity", lambda dev: ("RTX 4060 Ti", "610.88"))
    monkeypatch.setattr(jobmod.st, "card_busy", lambda dev: False)
    jobmod.job().runner = lambda device, on_proc=None: calls.append(device) or []
    prefs.set("compute_profiles", {"cuda:0": _profile()})
    jobmod.job().start("cuda:0", "cuda:0")
    t0 = time.time()
    while jobmod.job().running() and time.time() - t0 < 5:
        qapp.processEvents()
        time.sleep(0.01)
    assert calls == []
    monkeypatch.setattr(jobmod.st, "card_identity", lambda dev: ("RTX 4060 Ti", "700.00"))    # a new driver
    jobmod.job().start("cuda:0", "cuda:0")
    t0 = time.time()
    while jobmod.job().running() and time.time() - t0 < 5:
        qapp.processEvents()
        time.sleep(0.01)
    assert calls == ["cuda:0"]


def test_a_busy_card_is_not_tested_by_itself(env, qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(jobmod.st, "card_identity", lambda dev: ("RTX 4060 Ti", "610.88"))
    monkeypatch.setattr(jobmod.st, "card_busy", lambda dev: True)
    jobmod.job().runner = lambda device, on_proc=None: calls.append(device) or []
    jobmod.job().start("auto", "cuda:0")
    t0 = time.time()
    while jobmod.job().running() and time.time() - t0 < 5:
        qapp.processEvents()
        time.sleep(0.01)
    assert calls == [] and jobmod.job().note == "busy"


def test_a_beta_card_is_offered_and_marked(env, qapp):
    items = device_items(BETA)
    text, dev, tip = next(x for x in items if x[1] == "xpu:0")
    assert "beta" in text and "Beta" in tip
    d, page = _dialog(qapp, BETA)
    cb = page.findChildren(QComboBox)[0]
    assert any("beta" in cb.itemText(i) for i in range(cb.count()))
    d.close()


def test_the_new_words_are_french(env):
    i18n.set_language("fr")
    i18n._missing.clear()
    assert i18n._("Card test") == "Test de la carte" and i18n._("Test again") == "Retester"
    assert i18n._("GPU · {n} x faster").startswith("Carte graphique")         # one name for it (the night test pass)
    msg = st.STAGE_ON_CPU.format(why=st.REASONS[st.WRONG])
    assert "carte" in engine_text(msg) and "faux" in engine_text(msg)
    msg = "the graphics card has 3.1 GB free and this stack needs about 8.0 GB: it is computed on the CPU"
    assert "Go" in engine_text(msg)
