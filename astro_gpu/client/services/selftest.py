"""The card test (core/selftest.py), off the GUI thread: at first launch, when the card or its driver changed, and on
demand from Settings > Compute.

One job for the whole program (``job()``): a plain thread runs the test's child process and tells the GUI by a signal
(Settings may be closed meanwhile), like the download of the extracts: the ApiWorker is not used, it is one thread
shared by every short request of the client. The window never waits for it. The profile is saved in the settings in
the GUI thread, under the device tested and, for a test made under Automatic, under "auto" too.

    j = job()
    j.changed.connect(slot, Qt.QueuedConnection)      # a new profile, or the test started or ended
    j.start("auto", "cuda:0")                        # the setting, the device it names; force=True: test again
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QCoreApplication, QObject, Qt, Signal

from ...core import selftest as st
from .. import prefs


class CardTest(QObject):
    changed = Signal()
    _result = Signal(str, str, object)                 # pref, device, the new profile (None: nothing new)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = "idle"                            # idle | running
        self.note = ""                                 # why the last test did not give a profile
        self._proc = None
        self._thread: threading.Thread | None = None
        self.runner = st.run_test                      # (tests) the test itself
        self._result.connect(self._store, Qt.QueuedConnection)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)

    def running(self) -> bool:
        return self.state == "running"

    def start(self, pref: str, device: str | None, force: bool = False) -> bool:
        """Test the card ``device`` (a setting ``pref``: 'auto' or the device), in the background, unless a saved
        profile still holds for it (``force``: test it anyway). Returns whether a test started (GUI thread)."""
        if self.running() or device in (None, "", "cpu", "auto"):
            return False
        self.state, self.note = "running", ""
        self._thread = threading.Thread(target=self._run, args=(pref, str(device), force), daemon=True, name="card-test")
        self._thread.start()
        self.changed.emit()
        return True

    def stop(self):
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass

    def _run(self, pref: str, device: str, force: bool):
        profile, note = None, ""
        try:
            card, driver = st.card_identity(device)
            old = st.profile_of(prefs.get("compute_profiles"), device)
            if card and (force or not st.still_valid(old, card, driver)):
                if force or not st.card_busy(device):
                    results = self.runner(device, on_proc=lambda p: setattr(self, "_proc", p))
                    profile = st.make_profile(device, card, driver, results, pref == "auto") if results else None
                else:
                    note = "busy"
        except Exception as e:  # noqa: BLE001 - the test could not run: the settings stay as they are
            note = f"{type(e).__name__}: {e}"
        self._proc = None
        self._result.emit(pref, device, {"profile": profile, "note": note})

    def _store(self, pref: str, device: str, res: dict):
        self.state = "idle"
        self.note = res.get("note") or ""
        profile = res.get("profile")
        if profile is not None:
            profiles = dict(prefs.get("compute_profiles") or {})
            profiles[device] = profile
            if pref == "auto":
                profiles["auto"] = profile
            prefs.set("compute_profiles", profiles)
        self.changed.emit()


_JOB: list[CardTest] = []


def job() -> CardTest:
    """The program's card test (created in the GUI thread)."""
    if not _JOB:
        _JOB.append(CardTest())
    return _JOB[0]
