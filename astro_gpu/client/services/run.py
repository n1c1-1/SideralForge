"""The run service: launch a run, follow it, stop it, reattach to it, and tell the views what happens.

A run is a detached process (core/runner2.py). The client never waits for it: ``RunFollower`` tails
``logs/events.jsonl`` every 500 ms through ``api.read_events``, and every read happens in the service's own worker
thread (an ``ApiWorker``), never in the GUI thread. What the events say is kept in a ``RunTrack`` (plain Python, no Qt):
one ``JobState`` per job of the plan (waiting, running, done, failed, interrupted, skipped, not started), their
progress, and the console lines. ``RunController`` is the one object a window needs: it plans (Run all, Run to here),
opens the plan dialog (runs.py), starts the run, follows it, refreshes the model (the stripes of the graph turn running,
green, red…) when a stage starts or ends, writes the history line, and reattaches to a live run when a project is opened.

The window (app.py ``make_model``) makes one controller per project, after the model, and connects
``runningChanged``, ``updated``, ``finished`` and ``logRequested``; it shows ``status_widget()`` in the status bar,
calls ``watch_graph`` with each new graph, ``attach`` when a project opens (a live run is followed again),
``retranslate`` when the project screen is rebuilt and ``close`` when the project closes (the run goes on).

Breakpoints ("Review pause", set on the blocks of the graph): ``launch`` and ``run_to_here`` plan the run up to
the first review not done yet (``ProjectModel.plan_run``); the dialog says "Stops after Calibration for your review", and
at the end of the run its recap offers Continue (``continue_run``: the review is marked, the run goes on to the next pause
or to the end) and Open in Inspect (``open_in_inspect``: selects the block, then the window's graph page emits the
``openBlock`` it already routes; ``openInspect(workflow id, block)`` is emitted too, for a window that prefers it).

The console of the last run is the Log panel's right pane (``set_console`` / ``append_console``); the controller keeps
its lines, so that a rebuilt panel is filled again (``set_console``) and a change of language translates them.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from ... import api
from ...core.project2 import Project2
from ..i18n import _, hm, is_raw_error, number, plain_error, plural
from .worker import ApiWorker

# job states (what the events say; the graph's states come from the engine through the model)
WAITING, RUNNING, DONE, FAILED, INTERRUPTED, SKIPPED, NOT_STARTED = (
    "waiting", "running", "done", "failed", "interrupted", "skipped", "not_started")
SETTLED = (DONE, FAILED, SKIPPED, NOT_STARTED, INTERRUPTED)
POLL_MS = 500
#: the model reads the states of the graph again at most this often while a run goes (it replans and rebuilds the graph:
#: tens of ms each time, more on a busy machine); a stage that ends is shown within this time
REFRESH_MS = 1500
#: and only once the run has said nothing new of its stages for this long: when a stage starts or ends, the runner
#: rewrites its manifest and renames its folder, and on Windows a manifest being read cannot be replaced (the job would
#: fail); the states are read in the quiet that follows
QUIET_MS = 400


def duration_text(seconds: float, measured: bool = True) -> str:
    """A duration as the views show it, with ≈ when it is an estimate ("≤ 1 min" is one already)."""
    t = hm(seconds)
    return t if measured or t.startswith("≤") else "≈ " + t


def size_text(gb: float) -> str:
    """A size in GB, with the language's decimal mark ("7,6" in French)."""
    return number(gb, 2 if gb < 0.1 else 1)


#: the stages the model's titles do not name (exploration 6.1: "sessions", "inventory@2" were shown as they are)
STAGE_TITLES = {"sessions": "Sessions joined", "image": "Image", "fusion": "Combine channels"}


def stage_title(model, stage: str) -> str:
    """A stage of the engine in plain words, translated: a session's instance with its session's name
    ("Inventory · Session 2"), a block placed twice with its number ("Sky gradient 2")."""
    st, at, sid = stage.partition("@")
    b, sep, n = st.partition("~")
    text = (_(STAGE_TITLES[b]) + (f" {n}" if sep else "")) if b in STAGE_TITLES else model.title(st)
    if at:
        try:
            name = model.p.session(int(sid)).name
        except (AttributeError, KeyError, ValueError):
            name = ""
        text += " · " + (name or _("Session {n}").format(n=sid))
    return text


def start_failure(text: str) -> str:
    """Why a run could not start, in plain English words translated where shown (exploration 6.4: the worker's error
    was shown raw): the common causes by what the error says, else a general sentence."""
    t = text or ""
    low = t.lower()
    if "errno 28" in low or "no space left" in low or "disk is full" in low or "not enough space" in low:
        return "the project's drive is full"
    if t.startswith("PermissionError") or "errno 13" in low or "access is denied" in low:
        return "the project's folder cannot be written: access is refused"
    if t.startswith("FileNotFoundError") or "errno 2]" in low or "winerror 3]" in low:
        # which one (GUI exploration 2: "a folder of the project" did not say): the path the error names, said in the
        # language of the moment (a sentence with a path has no entry in the catalogue)
        m = re.search(r"""(?:errno 2\]|winerror 3\])[^'"]*['"]([^'"]+)['"]""", t, re.IGNORECASE)
        if m:
            return _("{path} cannot be found (a drive unplugged?)").format(path=Path(m.group(1)).as_posix())
        return "a folder of the project cannot be found (a drive unplugged?)"
    if "has no folder yet" in low:
        return "the project has no folder yet: save it first"
    return "an unexpected error stopped it"


def log_not_started(model, why: str, raw: str) -> None:
    """The history line of a run that could not start: the plain reason, and the real cause after it (GUI exploration
    2: "Show log" said the same sentence again, and the exported log had no cause)."""
    detail = start_detail(raw)
    if detail and detail != why:
        model.log("Run not started: {why} · cause: {detail}", why=why, detail=detail)
    else:
        model.log("Run not started: {why}", why=why)


def start_detail(text: str) -> str:
    """The real cause of a run that could not start, as the worker said it (its last meaningful line, a traceback's
    error line): written under the plain sentence in the console and the history, never in the run window (GUI
    exploration 2: the cause was written nowhere)."""
    lines = [x.strip() for x in (text or "").strip().splitlines() if x.strip()]
    errors = [x for x in lines if is_raw_error(x) and not x.startswith("Traceback")]
    return (errors[-1] if errors else lines[-1])[:300] if lines else ""


def pause_text(model, pause, stops=None, named: bool | None = None) -> str:
    """The blocks a pause stops after, as the graph names them ("Gradient 1", "Calibration · Session 2"); ``named``
    (by default: different blocks in several workflows, Run all with a pause in each), each with its workflow when
    the project has more than one ("Registration (Flow), Stacking (Workflow)"). ``stops``: (workflow id, block)
    pairs, the pause's own by default."""
    return ", ".join(pause_titles(model, pause, stops, named))


def pause_titles(model, pause, stops=None, named: bool | None = None) -> list[str]:
    """``pause_text``'s names, each once."""
    if pause is None and stops is None:
        return []
    stops = pause.all_stops() if stops is None else stops
    from ..blocks import review_title
    titles = [(model.pause_name(pid, k) if hasattr(model, "pause_name") else review_title(k), pid) for pid, k in stops]
    if named is None:
        named = len({p for _t, p in titles}) > 1 and len({t for t, _p in titles}) > 1
    many = named and len(getattr(model, "paths", None) or []) > 1
    out: list[str] = []
    for t, pid in titles:
        v = model.view(pid) if many and hasattr(model, "view") else None
        text = f"{t} ({v.name})" if v is not None else t
        if text not in out:
            out.append(text)
    return out


class PlanRefused(Exception):
    """A plan that cannot be made, and why, in plain words (translated)."""


def run_key(key: str) -> str:
    """The block Run to here plans for a block of the views: its own, a block placed twice included ("fond~2"); for
    Combine layers, the Split block it reads (exploration #21: it answered "Nothing to run"); for Crop, which computes
    nothing, the Registration its frame comes from (the night test pass: "Crop is no longer in the graph")."""
    return {"composition": "etoiles", "align_images": "image", "crop": "alignement"}.get(key, key)


def plan_to(model, p, key: str):
    """The plan of Run to here for a block of the views (``run_key``)."""
    return model.plan_align(p) if key == "align_images" else model.plan_to_block(p, run_key(key))


def run_here_refusal(model, key: str | None) -> str:
    """Why Run to here cannot run on this block (English, translated where shown), '' when it can (exploration #29: a
    Session block opened "Nothing to run")."""
    if model is None or model.path is None or not key:
        return "Select a block first"
    if key in ("sources", "final") or key.startswith("session"):
        return "A Session block is not computed: select a block after it" if key != "final" else \
            "Post is not computed by a run: select a block of a workflow"
    return ""


# ---------------------------------------------------------------------------------------------- the engine's texts
# The progress messages of the stages with numbers in them, and the common lines they log: (pattern, English template
# with positional fields). The fixed messages are translated as they are (run.json); the other free diagnostics the
# stages log stay in English.
PATTERNS: list[tuple[re.Pattern, str, tuple]] = [
    (re.compile(p), t, tr) for p, t, tr in [
        (r"^(\d+)/(\d+) frames calibrated$", "{0}/{1} frames calibrated", ()),
        (r"^reading (.+?) (\d+)/(\d+)$", "reading {0} {1}/{2}", (0,)),
        (r"^reading metadata (\d+)/(\d+)$", "reading metadata {0}/{1}", ()),
        (r"^aligning frame (\d+)/(\d+)$", "aligning frame {0}/{1}", ()),
        (r"^stars detected in (\d+)/(\d+) frames$", "stars detected in {0}/{1} frames", ()),
        (r"^pass (\d): (\d+)/(\d+) frames$", "pass {0}: {1}/{2} frames", ()),
        (r"^pass (\d): (\d+)/(\d+) frames, rejected ([\d.]+) %$", "pass {0}: {1}/{2} frames, rejected {3} %", ()),
        (r"^training step (\d+)/(\d+), loss ([\d.]+)$", "training step {0}/{1}, loss {2}", ()),
        (r"^training step (\d+), loss ([\d.]+)$", "training step {0}, loss {1}", ()),
        (r"^round (\d+)/(\d+), step (\d+)/(\d+) ?(.*)$", "round {0}/{1}, step {2}/{3} {4}", ()),
        (r"^Gaia (.+): cell (\d+)/(\d+)$", "Gaia {0}: cell {1}/{2}", ()),
        # issue #129: a field's first run downloads its sky tiles and spectra, counted
        (r"^XP spectra: (\d+) of (\d+)$", "XP spectra: {0} of {1}", ()),
        (r"^(Gaia stars|Gaia bright stars): tile (\d+) of (\d+)$", "{0}: tile {1} of {2}", (0,)),
        (r"^(Gaia stars|Gaia bright stars): (\d+) sky tiles downloaded in ([\d.]+) min$",
         "{0}: {1} sky tiles downloaded in {2} min", (0,)),
        (r"^refining star positions \(pass (\d+)\)$", "refining star positions (pass {0})", ()),
        (r"^searching the field at ([\d.]+)\"/px( \(shifted centres\))?$", "searching the field at {0}\"/px{1}", ()),
        (r"^realigning colours of (.+)$", "realigning colours of {0}", ()),
        (r"^PSF channel ([RGB]): (.*)$", "PSF channel {0}: {1}", ()),
        (r"^training target: (.*)$", "training target: {0}", ()),
        (r"^downloading (.+)$", "downloading {0}", ()),
        (r"^the card test sends this block to the processor: (.+)$",
         "the card test sends this block to the processor: {0}", (0,)),
        (r"^the graphics card has ([\d.]+) GB free and this stack needs about ([\d.]+) GB: it is computed on the CPU$",
         "the graphics card has {0} GB free and this stack needs about {1} GB: it is computed on the CPU", ()),
        # issue #125: the Classic stack within the machine's memory (core/stack_classic.MemoryPlan)
        (r"^memory: ([\d.]+) GB available, the (\d+) registered frames \(([\d.]+) GB\) are kept on disk and the "
         r"rejection is computed in (\d+) parts$",
         "memory: {0} GB available, the {1} registered frames ({2} GB) are kept on disk and the rejection is "
         "computed in {3} parts", ()),
        (r"^memory: ([\d.]+) GB available, the (\d+) registered frames \(([\d.]+) GB\) are kept on disk$",
         "memory: {0} GB available, the {1} registered frames ({2} GB) are kept on disk", ()),
        (r"^memory: ([\d.]+) GB available, the rejection is computed in (\d+) parts$",
         "memory: {0} GB available, the rejection is computed in {1} parts", ()),
        # the runner's hand-over of a run to a fresh process after a graphics card fault (core/runner2.py)
        (r"^the graphics card stopped responding during (.+); the rest of the run continues in a fresh process$",
         "the graphics card stopped responding during {0}; the rest of the run continues in a fresh process", (0,)),
        (r"^the graphics card stopped responding again during (.+): it is not tried a third time; the rest of the "
         r"run continues in a fresh process$",
         "the graphics card stopped responding again during {0}: it is not tried a third time; the rest of the run "
         "continues in a fresh process", (0,)),
        (r"^the graphics card stopped responding during (.+) and the run cannot go on: (.+)$",
         "the graphics card stopped responding during {0} and the run cannot go on: {1}", (0, 1)),
        (r"^the graphics card stopped responding again during (.+)$",
         "the graphics card stopped responding again during {0}", (0,)),
        (r"^it already stopped responding (\d+) times in this run$",
         "it already stopped responding {0} times in this run", ()),
        (r"^no fresh process could take over \((.*)\)$", "no fresh process could take over ({0})", ()),
        (r"^the run cannot start: (.*)$", "the run cannot start: {0}", ()),
        # GUI exploration 7.9: a session's folder not there
        (r"^the folder of the (lights|darks|flats|bias|flat darks) is not found: (.+) \(is its disk plugged in\?\)$",
         "the folder of the {0} is not found: {1} (is its disk plugged in?)", (0,)),
        # GUI exploration 2: the failures data can cause, with what to do (stages/stack.py, inventory.py, astrometry.py,
        # io/frames.py)
        (r"^useful area from the frames' header: (\d+)x(\d+) of (\d+)x(\d+) pixels, the overscan left out$",
         "useful area from the frames' header: {0}x{1} of {2}x{3} pixels, the overscan left out", ()),
        (r"^sparse field: (\d+) frame\(s\) aligned from a few stars \(rotation, shift and scale only\)$",
         "sparse field: {0} frame(s) aligned from a few stars (rotation, shift and scale only)", ()),
        # issue #63: frames of a camera turned between two nights (half a turn, or any angle)
        (r"^(\d+) frame\(s\) aligned after a turn of the camera of ([\d.]+) degrees$",
         "{0} frame(s) aligned after a turn of the camera of {1} degrees", ()),
        (r"^(\d+) frame\(s\) aligned after turns of the camera of ([\d.]+) to ([\d.]+) degrees$",
         "{0} frame(s) aligned after turns of the camera of {1} to {2} degrees", ()),
        (r"^only (\d+) light frame\(s\): a stack needs at least 2; add light frames to the session$",
         "only {0} light frame(s): a stack needs at least 2; add light frames to the session", ()),
        (r"^only (\d+) of (\d+) light frames could be aligned: a stack needs at least 2; check the frames$",
         "only {0} of {1} light frames could be aligned: a stack needs at least 2; check the frames", ()),
        (r"^only (\d+) of (\d+) light frames are kept, (\d+) taken out by hand: a stack needs at least 2; put frames "
         r"back$", "only {0} of {1} light frames are kept, {2} taken out by hand: a stack needs at least 2; put frames "
                   "back", ()),
        (r"^only (\d+) of (\d+) light frames are kept by the frame selection: a stack needs at least 2; relax the "
         r"selection thresholds$", "only {0} of {1} light frames are kept by the frame selection: a stack needs at "
                                   "least 2; relax the selection thresholds", ()),
        (r"^(.+) cannot be read: it is not a valid FITS file \(damaged, or wrongly named\); take it out of the "
         r"session$", "{0} cannot be read: it is not a valid FITS file (damaged, or wrongly named); take it out of the "
                      "session", ()),
        (r"^(\d+) (darks|flats|bias|flat darks) have another size than the lights \((\d+x\d+)\), e\.g\. (.+): take "
         r"them out of the session, or use those of this camera$",
         "{0} {1} have another size than the lights ({2}), e.g. {3}: take them out of the session, or use those of "
         "this camera", (1,)),
        # issue #57: frames already in colour
        (r"^(.+) cannot be used as a colour frame: (.+); take it out of the session$",
         "{0} cannot be used as a colour frame: {1}; take it out of the session", (1,)),
        (r"^the (darks|flats|bias|flat darks) \((.+)\) are already in colour, the lights a raw mosaic or monochrome: "
         r"they cannot calibrate them; take them out of the session, or use (?:darks|flats|bias|flat darks) of the "
         r"same kind as the lights$",
         "the {0} ({1}) are already in colour, the lights a raw mosaic or monochrome: they cannot calibrate them; "
         "take them out of the session, or use {0} of the same kind as the lights", (0,)),
        (r"^the (darks|flats|bias|flat darks) \((.+)\) are a raw mosaic or monochrome, the lights already in colour: "
         r"they cannot calibrate them; take them out of the session, or use (?:darks|flats|bias|flat darks) of the "
         r"same kind as the lights$",
         "the {0} ({1}) are a raw mosaic or monochrome, the lights already in colour: they cannot calibrate them; "
         "take them out of the session, or use {0} of the same kind as the lights", (0,)),
        (r"^the (darks|flats|bias|flat darks) \((.+)\) reach (\d+) and the lights (\d+) \(another bit depth\): they "
         r"cannot calibrate them; take them out of the session, or use (?:darks|flats|bias|flat darks) of the same "
         r"kind as the lights$",
         "the {0} ({1}) reach {2} and the lights {3} (another bit depth): they cannot calibrate them; take them out "
         "of the session, or use {0} of the same kind as the lights", (0,)),
        (r"^Bayer drizzle does not apply to frames already in colour \(no mosaic of raw colour samples to drop\): "
         r"choose the Classic stack$",
         "Bayer drizzle does not apply to frames already in colour (no mosaic of raw colour samples to drop): choose "
         "the Classic stack", ()),
        (r"^the (\d+) stars of this sparse field do not identify it on the sky beyond doubt, so no solution is given "
         r"rather than a wrong one: take Astrometry out of this workflow$",
         "the {0} stars of this sparse field do not identify it on the sky beyond doubt, so no solution is given "
         "rather than a wrong one: take Astrometry out of this workflow", ()),
        (r"^sparse field: the solution holds (\d+) powers of ten over chance, 9 needed \((\d+) of (\d+) stars\)$",
         "sparse field: the solution holds {0} powers of ten over chance, 9 needed ({1} of {2} stars)", ()),
        (r"^reference frame: (.+) aligned too few of the frames \(another field\?\), (.+) is used$",
         "reference frame: {0} aligned too few of the frames (another field?), {1} is used", ()),
        (r"^only (\d+) stars found in the stack, at least (\d+) are needed to identify the field: add light frames, or "
         r"take Astrometry out of this workflow$", "only {0} stars found in the stack, at least {1} are needed to "
                                                   "identify the field: add light frames, or take Astrometry out of "
                                                   "this workflow", ()),

        # the lines the stages log (the common ones: the rest of their free diagnostics stays in English)
        (r"^(\d+) lights of ([\d.]+) s \(([\d.]+) h\), (\d+)x(\d+) (\w+), (\d+) darks, (\d+) flats, (\d+) bias, camera (.*)$",
         "{0} lights of {1} s ({2} h), {3}x{4} {5}, {6} darks, {7} flats, {8} bias, camera {9}", ()),
        # the night test pass: the lines of a picture's first run, and the failures it met, were in English
        (r"^no colour calibration: camera colours, each channel's sky made grey and scaled to the same spread \(gains "
         r"(.+)\)$", "no colour calibration: camera colours, each channel's sky made grey and scaled to the same spread "
                     "(gains {0})", ()),
        (r"^colour mosaic pattern: (\d+) peaks notched \(strongest (\d+) x the spectrum around\)$",
         "colour mosaic pattern: {0} peaks notched (strongest {1} x the spectrum around)", ()),
        (r"^(.+): its stars match none of the reference's: not the same field$",
         "{0}: its stars match none of the reference's: not the same field", ()),
        (r"^the instrument's star needs the scale of the image: place the Astrometry block$",
         "the instrument's star needs the scale of the image: place the Astrometry block", ()),
        # the night test pass: an empty or damaged frame, named
        (r"^(\d+) (lights|darks|flats|bias|flat darks) cannot be read and are left out \(damaged or empty files\?\): "
         r"(.+)$", "{0} {1} cannot be read and are left out (damaged or empty files?): {2}", (1,)),
        (r"^no light frame can be read \(damaged or empty files\?\): (.+); take them out of the session$",
         "no light frame can be read (damaged or empty files?): {0}; take them out of the session", ()),
        (r"^(\d+) light frame\(s\) with a different size or pattern are ignored$",
         "{0} light frame(s) with a different size or pattern are ignored", ()),
        (r"^master dark level ([\d.]+) ADU, flat min ([\d.]+), ([\d.]+) % pixels excluded, sky noise ([\d.]+) ADU$",
         "master dark level {0} ADU, flat min {1}, {2} % pixels excluded, sky noise {3} ADU", ()),
        (r"^resuming calibration: (\d+)/(\d+) frames already done$", "resuming calibration: {0}/{1} frames already done", ()),
        (r"^(\d+) frames calibrated in ([\d.]+) min \(cache ([\d.]+) GB\)$",
         "{0} frames calibrated in {1} min (cache {2} GB)", ()),
        (r"^(\d+)/(\d+) frames aligned, residual ([\d.]+) px, FWHM ([\d.]+) px, transparency p10/p90 ([\d.]+)/([\d.]+)$",
         "{0}/{1} frames aligned, residual {2} px, FWHM {3} px, transparency p10/p90 {4}/{5}", ()),
        (r"^drizzle (\d+)x(\d+) from (\d+) frames in ([\d.]+) min, ([\d.]+) % pixels rejected$",
         "drizzle {0}x{1} from {2} frames in {3} min, {4} % pixels rejected", ()),
        # issue #125: the time of each pass of the stack (stages/stack.PassClock, core/stack_classic.py)
        (r"^pass (\d) done in ([\d.]+) min, of which (\d+) s reading the frames$",
         "pass {0} done in {1} min, of which {2} s reading the frames", ()),
        (r"^normalisation done in ([\d.]+) min, of which (\d+) s reading the frames$",
         "normalisation done in {0} min, of which {1} s reading the frames", ()),
        (r"^classic: debayering and resampling done in ([\d.]+) min, of which (\d+) s reading the frames$",
         "classic: debayering and resampling done in {0} min, of which {1} s reading the frames", ()),
        (r"^classic: rejection and means done in ([\d.]+) min$", "classic: rejection and means done in {0} min", ()),
        (r"^(\d+) stars detected in the stack$", "{0} stars detected in the stack", ()),
        (r"^solved: centre RA ([\d.]+) Dec ([+-]?[\d.]+), ([\d.]+)\"/px, field ([\d.]+) x ([\d.]+) deg, "
         r"residual ([\d.]+) px on (\d+) stars$",
         "solved: centre RA {0} Dec {1}, {2}\"/px, field {3} x {4} deg, residual {5} px on {6} stars", ()),
        (r"^(\d+) Gaia stars G < ([\d.]+) in the field$", "{0} Gaia stars G < {1} in the field", ()),
        (r"^background noise divided by ([\d.]+)$", "background noise divided by {0}", ()),
        (r"^resuming the training at step (\d+)/(\d+)$", "resuming the training at step {0}/{1}", ()),
        (r"^training done in ([\d.]+) min, final loss ([\d.]+)$", "training done in {0} min, final loss {1}", ()),
        (r"^sky fitted on (\d+) of (\d+) sky samples \((\d+) px blocks\); gradient ([\d.]+) on a sky of ([\d.]+) \((.*)\)$",
         "sky fitted on {0} of {1} sky samples ({2} px blocks); gradient {3} on a sky of {4} ({5})", ()),
        (r"^simple output (\d+) x (\d+) from (.+), useful frame x (\d+)-(\d+), y (\d+)-(\d+); sky: (.*)$",
         "simple output {0} x {1} from {2}, useful frame x {3}-{4}, y {5}-{6}; sky: {7}", ()),
    ]]


def engine_text(msg: str) -> str:
    """A progress message of a stage, in the language of the client (decimals with the language's mark)."""
    msg = (msg or "").strip()
    if is_raw_error(msg):                         # a program error ("OSError: …"): never shown as it is (exploration 2)
        return plain_error(msg)
    for rx, template, translated in PATTERNS:
        m = rx.match(msg)
        if m:
            args = []
            for i, a in enumerate(m.groups("")):
                if i in translated:
                    a = engine_text(a)                    # a word of the interface, or a message of its own
                if re.fullmatch(r"[+-]?\d+\.\d+", a):
                    a = ("+" if a.startswith("+") else "") + number(float(a), len(a.split(".")[1]))
                elif a.strip() == "(shifted centres)":
                    a = " " + _("(shifted centres)")
                args.append(a)
            return _(template).format(*args)
    return _(msg)


# ---------------------------------------------------------------------------------------------- the track of a run
Line = tuple  # (epoch seconds, kind, data dict): one line of the console, translated when it is shown


@dataclass
class JobState:
    """One job of the plan, and what the events said of it."""

    key: str
    stage: str
    block: str                                   # the engine's block
    wf: list[int]
    run_wf: int = 0
    est_s: float = 0.0
    measured: bool = False
    disk_gb: float = 0.0
    rerun: bool = False
    deps: list[str] = field(default_factory=list)            # keys of the jobs it reads
    state: str = WAITING
    frac: float = 0.0
    msg: str = ""
    eta: float | None = None
    started: float | None = None
    ended: float | None = None                   # when its stage_end came (a failure says no duration)
    duration: float | None = None
    error: str = ""
    reused: bool = False                         # found done and usable: nothing was computed
    upstream: list[str] = field(default_factory=list)        # stages whose failure skipped it
    last_norm: str | None = None                 # what the last console line of its progress said
    last_decile: int = -1

    @property
    def settled(self) -> bool:
        return self.state in SETTLED

    @classmethod
    def of(cls, j: Any) -> JobState:
        """From a runner2.Job, or the job dict of the ``plan`` event."""
        g = j.get if isinstance(j, dict) else (lambda k, d=None: getattr(j, k, d))
        deps = g("deps") or {}
        return cls(g("key"), g("stage"), g("block") or g("stage"), [int(w) for w in g("wf") or []],
                   int(g("run_wf") or 0), float(g("est_s") or 0.0), bool(g("measured")), float(g("disk_gb") or 0.0),
                   bool(g("rerun")), list(deps.values()) if isinstance(deps, dict) else list(deps))


@dataclass
class Batch:
    """What one read of the events changed."""

    lines: list = field(default_factory=list)
    stage_changed: bool = False                  # a stage started or ended: the states on disk changed
    progressed: bool = False
    ended: bool = False                          # run_end was seen


class RunTrack:
    """What a run is doing, built from its events (``apply``). No Qt, no disk."""

    def __init__(self, jobs=(), run_id: str = "", skipped: dict | None = None):
        self.run_id = run_id
        self.jobs: list[JobState] = [j if isinstance(j, JobState) else JobState.of(j) for j in jobs]
        self.by_key = {j.key: j for j in self.jobs}
        self.skipped: dict[int, str] = {int(k): v for k, v in (skipped or {}).items()}     # workflows left out
        self.status = "planned"                  # planned | running | done | failed | interrupted
        self.t0: float | None = None
        self.t_end: float | None = None
        self.device = ""
        self.pid: int | None = None
        self.error = ""                          # why the run ended badly, as the worker said it
        self.lost = False                        # the worker ended without saying run_end (killed, crashed)
        self.stop_asked = False
        self.lines: list[Line] = []
        self.counts: dict[str, int] = {}
        self.plan = None                         # the runner2.Plan this is the run of (set by the controller)
        #: the job keys of each block, as the run window's rows (set by the controller; #3: the console counted the
        #: engine's stages); None: each job counts as a block
        self.groups: list[list[str]] | None = None

    # ------------------------------------------------------------ construction
    @classmethod
    def of_plan(cls, plan, run_id: str = "") -> RunTrack:
        return cls(plan.jobs, run_id, getattr(plan, "skipped", None))

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed", "interrupted")

    @property
    def running(self) -> bool:
        return self.status == "running"

    # ------------------------------------------------------------ the events
    def apply(self, events: list[dict]) -> Batch:
        out = Batch()
        for e in events:
            if self.run_id and e.get("run") not in (None, self.run_id):
                continue
            if not self.run_id and e.get("run"):
                self.run_id = e["run"]
            self._one(e, out)
        return out

    def _job(self, e: dict) -> JobState | None:
        j = self.by_key.get(e.get("key") or "")
        if j is None and e.get("stage"):
            same = [x for x in self.jobs if x.stage == e["stage"]]
            j = next((x for x in same if x.state == RUNNING), None) or next((x for x in same if not x.settled), None) \
                or (same[0] if same else None)
        return j

    def _say(self, out: Batch, t: float, kind: str, **data):
        line = (t, kind, data)
        self.lines.append(line)
        out.lines.append(line)

    def _one(self, e: dict, out: Batch):
        t, kind = float(e.get("t") or time.time()), e.get("type")
        j = self._job(e)
        if kind == "run_start":
            self.status, self.t0, self.device, self.pid = "running", t, e.get("device") or "", e.get("pid")
            n = len(self.groups) if self.groups else len(self.jobs) or int(e.get("jobs") or 0)
            self._say(out, t, "run_start", n=n, sec=self.seconds(),
                      measured=self.measured())
        elif kind == "plan":
            if not self.jobs:
                self.jobs = [JobState.of(x) for x in e.get("jobs") or []]
                self.by_key = {x.key: x for x in self.jobs}
            self.skipped = {int(k): v for k, v in (e.get("skipped") or {}).items()} or self.skipped
        elif kind == "stage_start" and j is not None:
            j.state, j.started, j.frac, j.msg, j.eta, j.error = RUNNING, t, 0.0, "", None, ""
            self._say(out, t, "stage_start", stage=j.stage)
            out.stage_changed = out.progressed = True
        elif kind == "progress" and j is not None:
            if j.state in (WAITING, NOT_STARTED):
                j.state, j.started = RUNNING, j.started or t
            j.frac = min(max(float(e.get("frac") or 0.0), 0.0), 1.0)
            j.msg, j.eta = e.get("msg") or "", e.get("eta")
            norm, decile = re.sub(r"\d+(?:\.\d+)?", "#", j.msg), int(j.frac * 10)
            if norm != j.last_norm or decile != j.last_decile:
                j.last_norm, j.last_decile = norm, decile
                self._say(out, t, "progress", stage=j.stage, pct=int(round(100 * j.frac)), msg=j.msg)
            out.progressed = True
        elif kind == "log":
            msg, level = str(e.get("msg") or ""), e.get("level") or "info"
            if j is not None and level == "error":
                j.error = msg
            self._say(out, t, "log", stage=j.stage if j is not None else (e.get("stage") or ""), level=level, msg=msg)
        elif kind == "stage_end" and j is not None:
            status = e.get("status")
            j.ended = t
            if e.get("duration") is not None:
                j.duration = float(e["duration"])
            if status == "done":
                j.state, j.frac, j.reused = DONE, 1.0, bool(e.get("reused"))
                self._say(out, t, "stage_reused" if j.reused else "stage_done", stage=j.stage, sec=j.duration or 0.0)
            elif status == "failed":
                j.state, j.error = FAILED, str(e.get("error") or j.error or "")
                self._say(out, t, "stage_failed", stage=j.stage, error=j.error)
            elif status == "interrupted":
                j.state = INTERRUPTED
                self._say(out, t, "stage_stopped", stage=j.stage)
            elif status == "skipped":
                j.state, j.upstream = SKIPPED, list(e.get("upstream") or [])
                self._say(out, t, "stage_skipped", stage=j.stage, upstream=j.upstream)
            out.stage_changed = out.progressed = True
        elif kind == "run_end":
            self._end(str(e.get("status") or "done"), t, e, out)

    def _end(self, status: str, t: float, e: dict, out: Batch):
        self.status, self.t_end = status, t
        self.error = str(e.get("error") or self.error or "")
        for k in e.get("not_started") or []:
            if k in self.by_key and not self.by_key[k].settled:
                self.by_key[k].state = NOT_STARTED
        for k in e.get("interrupted") or []:
            if k in self.by_key and self.by_key[k].state in (WAITING, RUNNING):
                self.by_key[k].state = INTERRUPTED
        for k in e.get("failed") or []:
            if k in self.by_key and not self.by_key[k].settled:
                self.by_key[k].state = FAILED
        for j in self.jobs:                      # nothing stays "running" or "waiting" in a run that is over
            if j.state == RUNNING:
                j.state = INTERRUPTED
            elif j.state == WAITING:
                j.state = NOT_STARTED
        self.counts = self.count()
        self._say(out, t, "run_end", status=status, sec=self.elapsed(t), error=self.error, **self.counts,
                  blocks=self.block_count())
        out.ended = out.stage_changed = out.progressed = True

    def abort(self, why: str = "") -> Batch:
        """The run never started (its worker ended before saying anything): nothing ran. The console says why in plain
        words (6.4); ``error`` keeps what the worker said."""
        out = Batch()
        self._end("failed", time.time(), {"error": why}, out)
        self.lines[-1] = (self.lines[-1][0], "not_started", {"why": start_failure(why)})
        out.lines[-1] = self.lines[-1]
        detail = start_detail(why)
        if detail:                               # the real cause, for the Log and its export (exploration 2)
            line = (self.lines[-1][0], "start_detail", {"detail": detail})
            self.lines.append(line)
            out.lines.append(line)
        return out

    @property
    def never_started(self) -> bool:
        """A run over that never started (6.4: the recap says so, not "Failed")."""
        return self.finished and self.t0 is None and self.status == "failed"

    def end_lost(self, why: str = "") -> Batch:
        """The worker is gone and never said run_end (killed, crashed, never started): the run is over, interrupted."""
        out = Batch()
        self.lost = True
        self._end("interrupted", time.time(), {"error": why}, out)
        self.lines[-1] = (self.lines[-1][0], "run_lost", {"error": why, **self.counts})
        out.lines[-1] = self.lines[-1]
        return out

    # ------------------------------------------------------------ what it adds up to
    def count(self) -> dict[str, int]:
        c = {s: 0 for s in (DONE, FAILED, SKIPPED, INTERRUPTED, NOT_STARTED, WAITING, RUNNING)}
        for j in self.jobs:
            c[j.state] += 1
        c["reused"] = sum(1 for j in self.jobs if j.reused)
        c["computed"] = c[DONE] - c["reused"]
        return c

    def block_count(self) -> dict[str, int]:
        """What ``count`` says, by block (``groups``; #3): a block is done when all its jobs are (computed when one of
        them was), failed when one of them failed, not run when none of them ran, stopped when one was stopped."""
        c = {DONE: 0, FAILED: 0, INTERRUPTED: 0, "notrun": 0, "computed": 0}
        for keys in self.groups or [[j.key] for j in self.jobs]:
            js = [self.by_key[k] for k in keys if k in self.by_key]
            st = {j.state for j in js}
            if not js:
                continue
            if FAILED in st:
                c[FAILED] += 1
            elif INTERRUPTED in st:
                c[INTERRUPTED] += 1
            elif st == {DONE}:
                c[DONE] += 1
                c["computed"] += any(not j.reused for j in js)
            elif st <= {SKIPPED, NOT_STARTED}:
                c["notrun"] += 1
        return c

    def seconds(self) -> float:
        return sum(j.est_s for j in self.jobs)

    def measured(self) -> bool:
        return all(j.measured for j in self.jobs)

    def disk_gb(self) -> float:
        return sum(j.disk_gb for j in self.jobs)

    def elapsed(self, now: float | None = None) -> float:
        if self.t0 is None:
            return 0.0
        return max((self.t_end if self.t_end is not None else (now or time.time())) - self.t0, 0.0)

    def spent(self, j: JobState, now: float | None = None) -> float:
        """Seconds one job took in this run: the time since it started while it runs, its duration once over (the
        engine's, else until its end), nothing for a job that never started (reused: no time)."""
        if j.started is None:
            return 0.0
        if j.state == RUNNING:
            return max((now or time.time()) - j.started, 0.0)
        if j.duration is not None:
            return j.duration
        end = j.ended if j.ended is not None else self.t_end if self.t_end is not None else (now or time.time())
        return max(end - j.started, 0.0)

    @staticmethod
    def to_go(j: JobState) -> float:
        """Seconds one job has, or had when it ended, still to go: its estimate before it starts, nothing once done. The
        running job's ETA is a straight line through its bar (core/stage.py), which the stages fill by phases and search
        steps, not by time (#50; Astrometry's line assumes every scale of its list is tried, it stops at the first that
        solves): early it is no time at all, so the estimate's remaining share leads and the ETA takes over as the bar
        fills. A job stopped or failed keeps what it had to go when it ended."""
        if j.state == DONE:
            return 0.0
        prior = max(j.est_s * (1.0 - j.frac), 0.0)
        return prior if j.eta is None else (1.0 - j.frac) * prior + j.frac * max(float(j.eta), 0.0)

    @staticmethod
    def left(j: JobState) -> float:
        """Seconds to go in one job in this run: ``to_go`` while it waits or runs, nothing once it is over (reused or
        not)."""
        return RunTrack.to_go(j) if j.state in (WAITING, RUNNING) else 0.0

    def progress(self, jobs, now: float | None = None) -> float:
        """How far a block is, 0..1: the time its jobs took (``spent``) over that and the time they still have to go
        (``to_go``). The row of the run window, the line under the summary and the block's bar in the graph all show
        this one figure, and it comes from the time left the row says (#136: the status bar said 32 %, the window's
        line the running job's own bar 22 %, the row the jobs' bars weighted by their estimates 8 %; a bar filled by
        phases and search steps is no measure of time). A job stopped or failed counts what it had still to go, a job
        not run its estimate: the bar of a block stopped shows how far it got."""
        jobs = list(jobs)
        gone, rest = sum(self.spent(j, now) for j in jobs), sum(self.to_go(j) for j in jobs)
        if gone + rest > 0:
            return gone / (gone + rest)
        return 1.0 if jobs and all(j.state == DONE for j in jobs) else 0.0

    def fraction(self, now: float | None = None) -> float:
        """The share of the whole run done, 0..1, in time as ``progress``: what its jobs took over that and the time
        left (``remaining``). A job over has nothing left to go in this run, whatever it reached: the run ends at 1."""
        gone, rest = sum(self.spent(j, now) for j in self.jobs), self.remaining()
        if gone + rest > 0:
            return gone / (gone + rest)
        return 1.0 if self.jobs and all(j.settled for j in self.jobs) else 0.0

    def remaining(self) -> float:
        """Seconds to go: the sum of the jobs' ``left``, as the run window's rows show them (#116: the status bar counted
        the running job's ETA, its row its whole estimate: "28 min left" over rows adding up to 17)."""
        return sum(self.left(j) for j in self.jobs)

    def current(self) -> JobState | None:
        return next((j for j in self.jobs if j.state == RUNNING), None)

    def culprits(self, job: JobState) -> list[JobState]:
        """The failed jobs a skipped job was waiting on (through the jobs skipped in between)."""
        seen, out, todo = set(), [], list(job.deps)
        while todo:
            k = todo.pop()
            j = self.by_key.get(k)
            if j is None or k in seen:
                continue
            seen.add(k)
            if j.state == FAILED:
                out.append(j)
            elif j.state in (SKIPPED, NOT_STARTED):
                todo += j.deps
        if not out:                              # a track rebuilt from the events alone has no dependencies
            out = [x for s in job.upstream for x in self.jobs if x.stage == s and x.state == FAILED][:1]
        return out


# ---------------------------------------------------------------------------------------------- the console
def render_line(line: Line, title=lambda stage: stage) -> str:
    """One console line, translated: ``HH:MM:SS  text``."""
    t, kind, d = line
    when = time.strftime("%H:%M:%S", time.localtime(t))
    stage = title(d["stage"]) if d.get("stage") else ""
    if kind == "starting":                       # #3: blocks, as the run window counts them
        text = plural(d["n"], "Starting: {n} block", "Starting: {n} blocks")
    elif kind == "run_start":
        text = plural(d["n"], "Run started: {n} block · {t}", "Run started: {n} blocks · {t}",
                      t=duration_text(d["sec"], d["measured"]))
    elif kind == "stage_start":
        text = _("{stage} · started").format(stage=stage)
    elif kind == "progress":
        text = (_("{stage} · {pct} % · {msg}").format(stage=stage, pct=d["pct"], msg=engine_text(d["msg"])) if d["msg"]
                else _("{stage} · {pct} %").format(stage=stage, pct=d["pct"]))
    elif kind == "log":
        what = {"error": "{stage} · error: {msg}", "warning": "{stage} · warning: {msg}"}.get(d["level"], "{stage} · {msg}")
        text = _(what).format(stage=stage, msg=engine_text(d["msg"])) if stage else d["msg"]
    elif kind == "stage_done":
        text = _("{stage} · done in {t}").format(stage=stage, t=hm(d["sec"]))
    elif kind == "stage_reused":
        text = _("{stage} · already computed").format(stage=stage)
    elif kind == "stage_failed":                 # in plain words; a program error keeps its own words after (2)
        err = (d["error"] or "").strip()
        plain = engine_text(err.splitlines()[0]) if err else ""
        if err and is_raw_error(err):
            plain += " · " + start_detail(err)
        text = (_("{stage} · failed: {error}").format(stage=stage, error=plain) if err
                else _("{stage} · failed").format(stage=stage))
    elif kind == "stage_stopped":
        text = _("{stage} · stopped").format(stage=stage)
    elif kind == "stage_skipped":
        up = ", ".join(title(s) for s in d["upstream"])
        text = (_("{stage} · not run, {up} failed").format(stage=stage, up=up) if up
                else _("{stage} · not run, earlier failure").format(stage=stage))
    elif kind == "stop_asked":
        text = _("Stop asked")
    elif kind == "run_end":
        st = d["status"]
        b = d.get("blocks") or {DONE: d[DONE], FAILED: d[FAILED], "notrun": d[SKIPPED] + d[NOT_STARTED]}
        if st == "done":
            text = plural(b[DONE], "Run finished: {n} block in {t}", "Run finished: {n} blocks in {t}",
                          t=hm(d["sec"]))
        elif st == "interrupted":
            text = plural(b[DONE], "Run stopped: {n} block done", "Run stopped: {n} blocks done")
        else:
            text = _("Run failed: {failed} failed, {skipped} not run").format(failed=b[FAILED], skipped=b["notrun"])
    elif kind == "paused":
        from ..blocks import review_title
        text = _("Review pause: {block} · Inspect, then Continue").format(
            block=d.get("name") or review_title(d["block"]))
    elif kind == "run_lost":
        text = _("Run ended unexpectedly")
    elif kind == "not_started":
        text = _("Run not started: {why}").format(why=_(d["why"]))
    elif kind == "start_detail":                 # the worker's own words, under the plain sentence (exploration 2)
        text = _("Cause: {detail}").format(detail=d["detail"])
    else:
        text = kind
    return f"{when}  {text}"


# ---------------------------------------------------------------------------------------------- worker jobs
def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001 - a process we cannot see is gone
        return False


def _tail(path: str, lines: int = 6) -> str:
    """The last lines of the worker's own log (why it could not start)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join([x for x in text.splitlines() if x.strip()][-lines:])


def _start_job(snapshot: Project2, plan, device: str | None):
    """Launch the run (worker thread): the plan and the definition it was made from are frozen in its plan file. The
    stages the card test sends to the CPU go there."""
    from ...core import selftest
    from .. import prefs
    profile = selftest.profile_of(prefs.get("compute_profiles"), device)
    return api.start_run(snapshot, plan, device, selftest.cpu_jobs(profile, plan))


def _stop_job(p: Project2):
    api.cancel(p)


def _poll_job(p: Project2, offset: int, run_id: str, pid: int | None, log_file: str):
    """Read the new events of the run, and whether its worker is still there."""
    events, offset = api.read_events(p, offset, run_id)
    live = api.running(p)
    alive = live is not None and live.run_id == run_id
    proc = _pid_alive(pid) if pid else alive
    return events, offset, alive, proc, ("" if alive or proc else _tail(log_file))


def _attach_job(p: Project2):
    """The live run of the project, if any: its info, its frozen plan, the names of its workflows, and every event
    since it was launched."""
    from ...core import runner2
    info = api.running(p)
    if info is None:
        return None
    try:
        plan, body = runner2.load_plan(info.plan_file)
    except (OSError, ValueError, KeyError):
        return None
    names = {int(w["id"]): w["name"] for w in (body.get("definition") or {}).get("workflows") or []}
    events, offset = api.read_events(p, info.offset, info.run_id)
    return info, plan, names, events, offset


def _free_job(path: str) -> float:
    import shutil
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


# ---------------------------------------------------------------------------------------------- the follower
class RunFollower(QObject):
    """Tails the events of one run from a worker thread. ``batch(Batch)`` after each read that brought something;
    ``ended()`` once the run is over (run_end seen, or its worker gone)."""

    batch = Signal(object)
    ended = Signal()

    def __init__(self, project: Project2, worker: ApiWorker, track: RunTrack, info, offset: int | None = None,
                 interval: int = POLL_MS, parent=None):
        super().__init__(parent)
        self.p, self.worker, self.track, self.info = project, worker, track, info
        self.offset = info.offset if offset is None else offset
        self.channel = f"run.poll:{id(self)}"
        self._busy = False
        self._dead = 0
        self._timer = QTimer(self)
        self._timer.setInterval(interval)
        self._timer.timeout.connect(self.tick)
        worker.done.connect(self._done)
        worker.failed.connect(self._failed)

    def start(self):
        self._timer.start()
        self.tick()

    def stop(self):
        self._timer.stop()
        self.worker.cancel(self.channel)
        for sig, slot in ((self.worker.done, self._done), (self.worker.failed, self._failed)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def tick(self):
        if self._busy or self.track.finished:
            return
        self._busy = True
        self.worker.submit(self.channel, _poll_job, self.p, self.offset, self.track.run_id or self.info.run_id,
                           self.info.pid, self.info.log_file)

    def _failed(self, channel: str, _n: int, _text: str):
        if channel == self.channel:
            self._busy = False                   # a read that failed is asked again at the next tick

    def _done(self, channel: str, _n: int, result):
        if channel != self.channel:
            return
        self._busy = False
        events, self.offset, alive, proc, tail = result
        b = self.track.apply(events)
        if not self.track.finished:
            self._dead = 0 if (alive or proc or events) else self._dead + 1
            if self._dead >= 2:                  # gone, and a second read found nothing more: it is over
                lost = self.track.abort(tail) if self.track.t0 is None else self.track.end_lost()
                b.lines += lost.lines
                b.stage_changed = b.progressed = b.ended = True
        if b.lines or b.stage_changed or b.progressed:
            self.batch.emit(b)
        if self.track.finished:
            self._timer.stop()
            self.ended.emit()


# ---------------------------------------------------------------------------------------------- the controller
class RunController(QObject):
    """Everything the window needs to run a project: ``launch``, ``run_to_here``, ``stop``, ``attach``."""

    runningChanged = Signal(bool)            # a run started or ended
    statusChanged = Signal(str)              # the status line of a run in progress ('' when none)
    updated = Signal()                       # the track changed: the dialog, the status widget and the graph redraw
    finished = Signal(str, str)              # status (done | failed | interrupted), one line to flash
    logRequested = Signal()                  # "Show log" of the recap
    openInspect = Signal(int, str)           # "Open in Inspect" of a paused run: workflow id, block (of the views)

    def __init__(self, model, window=None, console=None, worker: ApiWorker | None = None):
        super().__init__(window)
        self.model, self.window = model, window
        self._console = console
        self._own_worker = worker is None
        self.worker = worker or ApiWorker(self)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.phase = "idle"                  # idle | starting | running | stopping
        self.track: RunTrack | None = None   # the run in progress, or the last one
        self.plan = None
        self.title = None                    # the heading of the run's dialog: a text, or a function giving it
        self.pause = None                    # the review the plan stops at (model.Pause), or None: it runs to the end
        self._planner = None                 # how the plan was made (Continue makes the next one the same way)
        self.names: dict[int, str] = {}      # workflow names of the run's frozen definition (when the model has moved on)
        self.follower: RunFollower | None = None
        self.dialog = None
        self._status_widget = None
        self._snapshot: Project2 | None = None
        self._free: float | None = None
        self._free_asked = False
        self._attaching = False
        self.ch_start, self.ch_attach, self.ch_free, self.ch_stop = (
            f"run.{n}:{id(self)}" for n in ("start", "attach", "free", "stop"))
        self._refresh_at = 0.0               # when the model last read the states
        self._refresh_due = False            # a stage changed since, and the model has not been told yet
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_model)
        self.plan_error = ""                 # why the plan of the window could not be made ('' when it was)
        self._stop_asked = False             # Stop pressed while the run was starting
        self.scope: set | None = None        # the workflows of Run to here (what may refuse it); None: Run all
        self._follow_timer = QTimer(self)    # the plan window follows the graph (coalesced)
        self._follow_timer.setSingleShot(True)
        self._follow_timer.setInterval(120)
        self._follow_timer.timeout.connect(self._replan)
        for name in ("pathsChanged", "costChanged", "sourcesChanged"):
            sig = getattr(model, name, None)
            if sig is not None and hasattr(sig, "connect"):
                sig.connect(self._graph_changed)

    # ------------------------------------------------------------ state
    @property
    def running(self) -> bool:
        return self.phase != "idle"

    def progress(self) -> float:
        """The fraction of the run done, 0..1."""
        return self.track.fraction() if self.track is not None else 0.0

    def status_text(self) -> str:
        t = self.track
        if self.phase == "idle" or t is None:
            return ""
        if self.phase == "stopping":
            return _("Stopping…")
        if self.phase == "starting" or not t.running:
            return _("Starting…")
        j = t.current()
        left = t.remaining()
        what = self.stage_title(j.stage) if j is not None else ""
        pct = int(round(100 * t.fraction()))
        # the share of the whole run, said so (#136: "Running: Astrometry · 32 %" read as Astrometry's, its row said 8 %)
        text = (_("Running: {stage} · {pct} % of the run").format(stage=what, pct=pct) if what
                else _("Running · {pct} % of the run").format(pct=pct))
        return text + " · " + _("≈ {t} left").format(t=hm(left)) if left > 30 else text

    def stage_title(self, stage: str) -> str:
        """A stage of the run in plain words (``stage_title``)."""
        return stage_title(self.model, stage)

    def groups(self, plan) -> list[list[str]]:
        """The job keys of each block of the run window (runs.build_rows): what the console counts (#3)."""
        from ..runs import build_rows
        try:
            return [list(r.keys) for r in build_rows(self.model, plan.jobs, self.names) if not r.echo]
        except Exception:  # noqa: BLE001 - a graph that cannot be built now: one block per job
            return [[j.key] for j in plan.jobs]

    # ------------------------------------------------------------ planning
    #: before a run: False when it must not start (the window sets it: a project without a name gets one first)
    before_run = staticmethod(lambda: True)

    def launch(self, start: bool = False):
        """Run all (F5): the dialog of the plan, or the progress of the run in progress. With breakpoints the plan
        stops after the first review not done yet. ``start`` (the Run button, issue #51): the run starts at once, the
        dialog following it; not without frames (issue #48)."""
        if self.running:
            return self.show_dialog()
        if not self.before_run():
            return None
        self.scope = None
        d = self._plan_and_open(self.model.plan_run, lambda: _("Run project"), show_done=True)
        # never at once when the plan has something to say first (a file that may be a mistake, a workflow left out,
        # the disk): the window shows it, Run is the user's (GUI exploration round 2)
        if start and d is not None and self.plan and self.model.has_data() and not d.warnings(sync=True):
            d.start()
        return d

    def run_to_here(self):
        """Run to the selected block (Shift+F5): the blocks it needs, in the current workflow (or the first review of
        this workflow before it)."""
        m = self.model
        p = m.path
        if self.running:
            return self.show_dialog()
        key = m.sel_step
        if p is None or run_here_refusal(m, key) or not self.before_run():   # a Session block is not computed (#29)
            return None
        pid, name = p.pid, self.block_name(p, key)
        self.scope = {q.pid for q in m.align_group(p)} if key == "align_images" else {pid}

        def planner():
            q = m.view(pid)                                  # the window follows the graph: Ctrl+Z may take it away
            if q is not None and key == "align_images":      # every image of the block
                return m.plan_align(q), None
            if q is not None and m.ready and m.is_cut(q, run_key(key)):   # drawn, but switched off: not computed
                raise PlanRefused(_("{block} is switched off: D switches it on again").format(block=name))
            if q is None or (m.ready and run_key(key) not in m.pause_keys(q) + ["post", "etoiles"]):
                raise PlanRefused(_("{block} is no longer in the graph").format(block=name))
            return m.plan_run_to(q, run_key(key))
        return self._plan_and_open(planner, lambda: _('Run "{name}" to {block}').format(name=p.name, block=name))

    def block_name(self, p, key: str) -> str:
        """A block as the graph and the run window name it ("Gradient 2", "Calibration · Session 2", 4.1 and 6.1)."""
        from ..blocks import Graph, block_title
        from ..runs import row_title
        if key == "post":
            return _("Output")
        try:
            node = Graph(self.model).node_of(p, key)
        except Exception:  # noqa: BLE001 - a graph that cannot be built now: its plain name
            node = None
        return row_title(self.model, node, p) if node is not None else block_title(p, key)

    def _plan_and_open(self, planner, title, show_done: bool = False):
        """``planner()`` gives (plan, pause); the dialog of that plan. A plan that cannot be made is said in the
        dialog, never an empty "Nothing to run" (GUI exploration round 2: a pause on "Gradient 2" hid an error)."""
        plan, pause, self.plan_error = self._make_plan(planner)
        self._planner = planner
        return self.open_plan(plan, title, show_done, pause)

    @staticmethod
    def _make_plan(planner):
        """(plan, pause, why it could not be planned: '' when it was)."""
        try:
            plan, pause = planner()
            return plan, pause, ""
        except PlanRefused as e:                              # said by the planner, in plain words already
            return api.Plan(), None, str(e)
        except (ValueError, KeyError, OSError) as e:
            from ..i18n import plain_error
            return api.Plan(), None, plain_error(str(e))

    def _graph_changed(self, *_a):
        """The graph, the settings or the device changed: a plan window not run yet follows (GUI exploration round 2:
        Ctrl+Z left the undone block in it, and its Run ran the old graph)."""
        if self.dialog is not None and self.phase == "idle" and self._planner is not None:
            self._follow_timer.start()

    def _replan(self):
        d = self.dialog
        if d is None or self.running or self._planner is None or d.phase() != "plan":
            return
        plan, pause, self.plan_error = self._make_plan(self._planner)
        self._snapshot = Project2.from_dict(self.model.p.to_dict(), self.model.p.root)
        self.plan, self.pause = plan, pause
        d.set_plan(plan)

    def open_plan(self, plan, title, show_done: bool = False, pause=None):
        """The plan dialog (a snapshot of the definition is taken now: the plan is frozen). ``title``: the text, or a
        function giving it (so that a change of language rewrites it). ``pause``: the review the plan stops at."""
        from ..runs import RunDialog
        self._snapshot = Project2.from_dict(self.model.p.to_dict(), self.model.p.root)
        self._drop_dialog()
        self.plan, self.title, self.pause = plan, title, pause
        self.dialog = RunDialog(self.model, plan, title, show_done, self, self.window)
        self._free = None
        self._free_asked = False
        self.dialog.show()
        self.ask_free()
        return self.dialog

    def ask_free(self):
        """Ask the worker how much room the project's drive has (the dialog warns when the plan needs more)."""
        if not self._free_asked:
            self._free_asked = True
            self.worker.submit(self.ch_free, _free_job, str(self.model.p.root))

    def show_dialog(self):
        """The dialog of the run in progress, or of the last run (its recap)."""
        if self.dialog is None and self.track is not None and self.plan is not None:
            self._make_dialog()
        if self.dialog is not None:
            self.dialog.show()
            self.dialog.raise_()
        return self.dialog

    # ------------------------------------------------------------ pause: continue, or look first
    def continue_run(self):
        """Continue, in the recap of a pause: the results the user checked are marked as reviewed, and the run goes on to
        the next pause, or to the end. If they changed meanwhile (the block is Outdated) it computes them again and
        pauses again."""
        if self.running or self.pause is None:
            return None
        self.model.mark_reviewed(self.pause)
        d = self._plan_and_open(self._planner or self.model.plan_run, self.title or (lambda: _("Run project")))
        if self.plan and d is not None:
            self.begin(self.plan)
            d.refresh()
        return d

    def open_in_inspect(self):
        """Open in Inspect, in the recap of a pause: the block is selected, and the window shows it in Inspect."""
        pa, m = self.pause, self.model
        q = m.view(pa.pids[0]) if pa is not None and pa.pids else None
        if q is None:
            return
        page = getattr(self.window, "graph", None)            # the window's GraphPage: it routes openBlock to Inspect
        if page is not None and hasattr(page, "select"):
            page.select(index=m.index_of(q.pid), step=pa.block)
        else:
            m.select(index=m.index_of(q.pid), step=pa.block)
            m.scope = list(pa.pids)
        if self.dialog is not None:
            self.dialog.hide()
        self.openInspect.emit(q.pid, pa.block)
        if page is not None and hasattr(page, "openBlock"):
            page.openBlock.emit()

    def pause_text(self) -> str:
        """The block the run stopped at, for a line of text."""
        return pause_text(self.model, self.pause)          # as the graph names it ("Gradient 1", "Calibration · Session 2")

    def _make_dialog(self):
        from ..runs import RunDialog
        self._drop_dialog()
        self.dialog = RunDialog(self.model, self.plan, self.title or (lambda: _("Run")), False, self, self.window,
                                track=self.track, names=self.names)
        return self.dialog

    def _drop_dialog(self):
        if self.dialog is not None:
            self.dialog.hide()
            self.dialog.deleteLater()
            self.dialog = None

    # ------------------------------------------------------------ starting
    def begin(self, plan) -> bool:
        """The Run button of the plan dialog: start the run (in the worker: nothing blocks the window)."""
        if self.running or not plan or self._snapshot is None:
            return False
        if self.refusal(plan):                               # a fusion or data that cannot run (1.1): never started
            return False
        self.plan = plan
        self.track = RunTrack.of_plan(plan)
        self.track.plan = plan
        self.names = {}
        self.phase = "starting"
        self.track.groups = self.groups(plan)                 # #3: the console counts blocks, as the window shows
        self.track.lines.append((time.time(), "starting", {"n": len(self.track.groups)}))
        self.set_console()
        self.worker.submit(self.ch_start, _start_job, self._snapshot, plan, self.model.device)
        self._emit_state(True)
        return True

    def refusal(self, plan) -> list[tuple[str, str]]:
        """(workflow, why) of what refuses this plan: the fusions and the data that cannot run, among the workflows the
        plan computes (Run to here on a channel's Stacking is not refused for the fusion after it)."""
        problems = getattr(self.model, "run_problems", None)
        if problems is None:
            return []
        if self.scope is None:                               # Run all: the whole project
            return problems()
        try:
            return problems(set(self.scope))                 # the workflow Run to here runs (sharing jobs or not)
        except TypeError:                                    # a model without the workflows' filter
            return problems()

    def _done(self, channel: str, _n: int, result):
        if channel == self.ch_start:
            self._started(result)
        elif channel == self.ch_attach:
            self._attached(result)
        elif channel == self.ch_free:
            self._free = float(result)
            self.updated.emit()

    def _failed(self, channel: str, _n: int, text: str):
        why = text.splitlines()[0] if text else "?"
        if channel == self.ch_start:
            self.phase = "idle"
            self._stop_asked = False
            if "already in progress" in why:     # another window, or a run started before this client: follow it
                self.track = None
                self._drop_dialog()
                self.attach()
            else:                                # 6.4: plain words in the console, the history and the window
                self.append_console(self.track.abort(why).lines)
                plain = start_failure(why)
                log_not_started(self.model, plain, text)
                self.finished.emit("failed", _("Run not started: {why}").format(why=_(plain)))
            self._emit_state(False)
        elif channel == self.ch_attach:
            self._attaching = False

    def _started(self, info):
        t = self.track
        t.run_id, t.pid = info.run_id, info.pid
        self.phase = "running"
        if self._stop_asked:                                 # Stop was pressed while it started
            self._stop_asked = False
            self.phase = "stopping"
            self.worker.submit(self.ch_stop, _stop_job, self.model.p)
        self._follow(info, info.offset)
        self._emit_state(True)

    # ------------------------------------------------------------ following
    def _follow(self, info, offset: int):
        if self.follower is not None:
            self.follower.stop()
            self.follower.deleteLater()
        self.follower = RunFollower(self.model.p, self.worker, self.track, info, offset, parent=self)
        self.follower.batch.connect(self._batch)
        self.follower.ended.connect(self._over)
        self.follower.start()

    def _batch(self, b: Batch):
        self.append_console(b.lines)
        if b.stage_changed:
            self._model_changed()          # the stripes of the graph: running, done, failed…
        self.updated.emit()
        self.statusChanged.emit(self.status_text())

    def _model_changed(self):
        """A stage started or ended: the model must read the states again, once the run is quiet (QUIET_MS after its
        last stage event) and at most every REFRESH_MS."""
        self._refresh_due = True
        wait = max(QUIET_MS / 1000, REFRESH_MS / 1000 - (time.monotonic() - self._refresh_at))
        self._refresh_timer.start(int(wait * 1000) + 1)       # a newer stage event starts the wait again

    def _refresh_model(self):
        """Tell the model that results changed on disk."""
        if not self._refresh_due:
            return
        self._refresh_due = False
        self._refresh_at = time.monotonic()
        self.model.results_changed()

    def _over(self):
        """The run is over: the states are read again, the history gets its line, the recap is shown."""
        t = self.track
        self._refresh_timer.stop()
        self._refresh_due, self._refresh_at = False, time.monotonic()
        self.model.results_changed()
        n, c = t.elapsed(), t.block_count()                  # #3: blocks, as the run window shows them
        if t.status == "done" and self.pause is not None:
            text = plural(c[DONE], "Review pause at {block}: {n} block in {t}",
                          "Review pause at {block}: {n} blocks in {t}", block=self.pause_text(), t=hm(n))
            # the history line is written in the language shown when it is read (#31): a template and its words
            self.model.log("Review pause at {block}", block=self.pause_text())
            # the console says what comes next
            line = (time.time(), "paused", {"block": self.pause.block, "name": self.pause_text()})
            t.lines.append(line)
            self.append_console([line])
        elif t.status == "done":
            text = plural(c[DONE], "Run finished: {n} block in {t}", "Run finished: {n} blocks in {t}", t=hm(n))
            self.model.log("Run finished in {t}", t=hm(n))
        elif t.status == "interrupted":
            text = plural(c[DONE], "Run stopped: {n} block done", "Run stopped: {n} blocks done")
            self.model.log("Run stopped")
        elif t.never_started:                                # 6.4: said in plain words, in the history too
            why = start_failure(t.error)
            text = _("Run not started: {why}").format(why=_(why))
            log_not_started(self.model, why, t.error)
        else:
            text = _("Run failed: {failed} failed, {skipped} not run").format(failed=c[FAILED], skipped=c["notrun"])
            self.model.log("Run failed: {failed} failed, {skipped} not run", failed=c[FAILED], skipped=c["notrun"])
            bad = next((j for j in t.jobs if j.state == FAILED and (j.error or "").strip()), None)
            if bad is not None:                              # why, without opening the log (the night test pass)
                err = bad.error.strip().splitlines()[0]
                text += " · " + _("{stage}: {error}").format(
                    stage=self.stage_title(bad.stage), error=engine_text(err) if not is_raw_error(bad.error) else
                    _("a program error (the detail is in the Log)"))
        self.phase = "idle"
        c_ = self._console_obj()
        if c_ is not None and hasattr(c_, "refresh"):
            c_.refresh()
        self.updated.emit()
        self._emit_state(False)
        self.finished.emit(t.status, text)
        self.show_dialog()

    def _emit_state(self, running: bool):
        self.model.run_active = running                  # the views grey their Run to here meanwhile (issue #33)
        self.runningChanged.emit(running)
        self.statusChanged.emit(self.status_text())
        self.updated.emit()

    # ------------------------------------------------------------ stopping
    def stop(self):
        """Stop the run at the next check of the stage in progress: what is computed is kept, and the next run
        goes on from the stage's last checkpoint."""
        if self.phase == "starting":                       # round 2: asked while starting, done once it has started
            self._stop_asked = True
        elif self.phase != "running":
            return
        else:
            self.worker.submit(self.ch_stop, _stop_job, self.model.p)
        self.phase = "stopping"
        self.track.stop_asked = True
        self.track.lines.append((time.time(), "stop_asked", {}))
        self.append_console()
        self.updated.emit()
        self.statusChanged.emit(self.status_text())

    # ------------------------------------------------------------ reattaching
    def attach(self):
        """A project opened with a live run (or a run started before this client): follow it again, from its plan
        file and the events since it was launched."""
        if self.running or self._attaching:
            return
        self._attaching = True
        self.worker.submit(self.ch_attach, _attach_job, self.model.p)

    def _attached(self, result):
        self._attaching = False
        if result is None or self.running:
            return
        info, plan, names, events, offset = result
        self.plan, self.names, self.title = plan, names, lambda: _("Run in progress")
        self.pause = None
        self.track = RunTrack.of_plan(plan, info.run_id)
        self.track.plan, self.track.pid = plan, info.pid
        self.track.groups = self.groups(plan)
        b = self.track.apply(events)
        self.set_console()
        self.phase = "running"
        if self.track.finished:                  # it ended while we were looking
            self._over()
            return
        self._follow(info, offset)
        self._emit_state(True)
        if b.stage_changed:
            self._model_changed()

    # ------------------------------------------------------------ the console of the last run
    def _console_obj(self):
        c = self._console
        if c is not None and not hasattr(c, "append_console") and callable(c):
            c = c()
        return c

    def set_console(self):
        """Fill the Log panel's console with the lines of the run."""
        c = self._console_obj()
        if c is not None and self.track is not None:
            c.set_console("\n".join(render_line(x, self.stage_title) for x in self.track.lines))

    def append_console(self, lines=None):
        c = self._console_obj()
        if c is None or self.track is None:
            return
        for x in (self.track.lines[-1:] if lines is None else lines):
            c.append_console(render_line(x, self.stage_title))

    # ------------------------------------------------------------ views
    def status_widget(self):
        """The progress and the Stop button of a run in progress, for the status bar (hidden while idle)."""
        if self._status_widget is None:
            from ..runs import RunStatus
            self._status_widget = RunStatus(self)
        return self._status_widget

    def watch_graph(self, view):
        """Draw the progress of each running block on the graph (the GraphView)."""
        from ..runs import RunOverlay
        return RunOverlay(view, self)                          # a child of the view: it goes with it

    def retranslate(self):
        """The language changed: the dialog, the status widget and the console are written again."""
        if self.dialog is not None:
            self.dialog.rebuild()
        if self._status_widget is not None:
            self._status_widget.rebuild()
        self.set_console()
        self.updated.emit()

    def free_gb(self) -> float | None:
        return self._free

    def close(self):
        """The project is closed: stop following (a run in progress goes on without us)."""
        if self.follower is not None:
            self.follower.stop()
            self.follower = None
        self._refresh_timer.stop()
        self._follow_timer.stop()
        for name in ("pathsChanged", "costChanged", "sourcesChanged"):
            try:
                getattr(self.model, name).disconnect(self._graph_changed)
            except (AttributeError, RuntimeError, TypeError):
                pass
        self._drop_dialog()
        self.phase = "idle"
        for sig, slot in ((self.worker.done, self._done), (self.worker.failed, self._failed)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        if self._own_worker:
            self.worker.stop()
