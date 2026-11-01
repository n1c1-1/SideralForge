"""The run service (P11): launch a run, follow it, stop it, reattach to it, and tell the views what happens.

A run is a detached process (core/runner2.py). The client never waits for it: ``RunFollower`` tails
``logs/events.jsonl`` every 500 ms through ``api.read_events``, and every read happens in the service's own worker
thread (an ``ApiWorker``), never in the GUI thread. What the events say is kept in a ``RunTrack`` (plain Python, no Qt):
one ``JobState`` per job of the plan (waiting, running, done, failed, interrupted, skipped, not started), their
progress, and the console lines. ``RunController`` is the one object a window needs: it plans (Run all, Run to here),
opens the plan dialog (runs.py), starts the run, follows it, refreshes the model (the stripes of the graph turn running,
green, red…) when a stage starts or ends, writes the history line, and reattaches to a live run when a project is opened.

Wiring, for the window (P13)::

    self.run = RunController(self.s, window=self, console=lambda: self.logp)    # once per project, after the model
    self.run.runningChanged.connect(self.refresh)             # Run button, the Stop item of the Run menu
    self.run.finished.connect(lambda status, text: self.flash(text))      # one line at the end of a run
    self.run.logRequested.connect(lambda: self.b_log.setChecked(True))    # "Show the log" of the recap
    self.statusBar().addPermanentWidget(self.run.status_widget())         # the progress and Stop, while a run lasts
    self.run.watch_graph(page.graph)                          # the progress of each block, drawn on the graph
    self.run.attach()                                         # a project opened with a live run: follow it again
    launch -> self.run.launch(); run_to_here -> self.run.run_to_here(); stop_run -> self.run.stop()
    language changed, project screen rebuilt (new Log panel, new graph): self.run.retranslate()
    project closed: self.run.close()                          # the run goes on; opening the project follows it again

The console of the last run is the Log panel's right pane (``set_console`` / ``append_console``); the controller keeps
its lines, so that a rebuilt panel is filled again (``sync_console``) and a change of language translates them.
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
from ..i18n import _, hm, number
from .worker import ApiWorker

# job states (what the events say; the graph's states come from the engine through the model)
WAITING, RUNNING, DONE, FAILED, INTERRUPTED, SKIPPED, NOT_STARTED = (
    "waiting", "running", "done", "failed", "interrupted", "skipped", "not_started")
SETTLED = (DONE, FAILED, SKIPPED, NOT_STARTED, INTERRUPTED)
POLL_MS = 500
#: the model reads the states of the graph again at most this often while a run goes (it replans and rebuilds the graph:
#: tens of ms each time, more on a busy machine); a stage that ends is shown within this time
REFRESH_MS = 1500


def duration_text(seconds: float, measured: bool = True) -> str:
    """A duration as the views show it, with ≈ when it is an estimate (SPEC §7)."""
    t = hm(seconds)
    return t if measured or t.startswith("≤") else "≈ " + t


def size_text(gb: float) -> str:
    """A size in GB, with the language's decimal mark ("7,6" in French)."""
    return number(gb, 2 if gb < 0.1 else 1)


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
        (r"^XP spectra (\d+)/(\d+)$", "XP spectra {0}/{1}", ()),
        (r"^refining star positions \(pass (\d+)\)$", "refining star positions (pass {0})", ()),
        (r"^fitting the spectral response on (\d+) stars$", "fitting the spectral response on {0} stars", ()),
        (r"^searching the field at ([\d.]+)\"/px( \(shifted centres\))?$", "searching the field at {0}\"/px{1}", ()),
        (r"^realigning colours of (.+)$", "realigning colours of {0}", ()),
        (r"^PSF channel ([RGB]): (.*)$", "PSF channel {0}: {1}", ()),
        (r"^training target: (.*)$", "training target: {0}", ()),
        (r"^downloading (.+)$", "downloading {0}", ()),

        # the lines the stages log (the common ones: the rest of their free diagnostics stays in English)
        (r"^(\d+) lights of ([\d.]+) s \(([\d.]+) h\), (\d+)x(\d+) (\w+), (\d+) darks, (\d+) flats, (\d+) bias, camera (.*)$",
         "{0} lights of {1} s ({2} h), {3}x{4} {5}, {6} darks, {7} flats, {8} bias, camera {9}", ()),
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
        (r"^(\d+) stars detected in the stack$", "{0} stars detected in the stack", ()),
        (r"^solved: centre RA ([\d.]+) Dec ([+-]?[\d.]+), ([\d.]+)\"/px, field ([\d.]+) x ([\d.]+) deg, "
         r"residual ([\d.]+) px on (\d+) stars$",
         "solved: centre RA {0} Dec {1}, {2}\"/px, field {3} x {4} deg, residual {5} px on {6} stars", ()),
        (r"^spectral response from (\d+) Gaia XP stars, chromaticity residual ([\d.]+)$",
         "spectral response from {0} Gaia XP stars, chromaticity residual {1}", ()),
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
    for rx, template, translated in PATTERNS:
        m = rx.match(msg)
        if m:
            args = []
            for i, a in enumerate(m.groups("")):
                if i in translated:
                    a = _(a)
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
            self._say(out, t, "run_start", n=len(self.jobs) or int(e.get("jobs") or 0), sec=self.seconds(),
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
        self._say(out, t, "run_end", status=status, sec=self.elapsed(t), error=self.error, **self.counts)
        out.ended = out.stage_changed = out.progressed = True

    def abort(self, why: str = "") -> Batch:
        """The run never started (its worker ended before saying anything): nothing ran."""
        out = Batch()
        self._end("failed", time.time(), {"error": why}, out)
        self.lines[-1] = (self.lines[-1][0], "not_started", {"why": why or _("its process ended")})
        out.lines[-1] = self.lines[-1]
        return out

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

    @staticmethod
    def weight(j: JobState) -> float:
        return max(j.est_s, 1.0)

    def share(self, j: JobState, shown: bool = False) -> float:
        """How much of the job is done, 0..1. For the whole run a job that is over counts whole (the time is gone);
        ``shown``: what a bar draws, the progress a stopped or failed job had reached, nothing for a job not run."""
        if j.state == DONE:
            return 1.0
        if j.state in (RUNNING, FAILED, INTERRUPTED):
            return j.frac if (j.state == RUNNING or shown) else 1.0
        return 0.0 if shown else (1.0 if j.settled else 0.0)

    def fraction(self) -> float:
        tot = sum(self.weight(j) for j in self.jobs)
        return sum(self.weight(j) * self.share(j) for j in self.jobs) / tot if tot else 0.0

    def remaining(self) -> float:
        """Seconds to go: the running job's own estimate (its ETA), and the estimates of the jobs that wait."""
        left = 0.0
        for j in self.jobs:
            if j.state == WAITING:
                left += j.est_s
            elif j.state == RUNNING:
                left += j.eta if j.eta is not None else max(j.est_s * (1.0 - j.frac), 0.0)
        return left

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
    if kind == "starting":
        text = _("Starting the run: {n} stages").format(n=d["n"])
    elif kind == "run_start":
        text = _("Run started: {n} stages · {t}").format(n=d["n"], t=duration_text(d["sec"], d["measured"]))
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
    elif kind == "stage_failed":
        text = (_("{stage} · failed: {error}").format(stage=stage, error=d["error"]) if d["error"]
                else _("{stage} · failed").format(stage=stage))
    elif kind == "stage_stopped":
        text = _("{stage} · stopped").format(stage=stage)
    elif kind == "stage_skipped":
        up = ", ".join(title(s) for s in d["upstream"])
        text = (_("{stage} · not run, {up} failed").format(stage=stage, up=up) if up
                else _("{stage} · not run, an earlier stage failed").format(stage=stage))
    elif kind == "stop_asked":
        text = _("Stop asked: the stage in progress stops at its next check")
    elif kind == "run_end":
        st = d["status"]
        if st == "done":
            text = _("Run finished: {n} stages in {t}").format(n=d[DONE], t=hm(d["sec"]))
        elif st == "interrupted":
            text = _("Run stopped: {n} stages done").format(n=d[DONE])
        else:
            text = _("Run failed: {failed} failed, {skipped} not run").format(
                failed=d[FAILED], skipped=d[SKIPPED] + d[NOT_STARTED])
    elif kind == "run_lost":
        text = _("The run stopped without warning: its process ended")
    elif kind == "not_started":
        text = _("The run could not start: {why}").format(why=d["why"])
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
    """Launch the run (worker thread): the plan and the definition it was made from are frozen in its plan file."""
    return api.start_run(snapshot, plan, device)


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
    logRequested = Signal()                  # "Show the log" of the recap

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
        self.names: dict[int, str] = {}      # workflow names of the run's frozen definition (when the model has moved on)
        self.follower: RunFollower | None = None
        self.dialog = None
        self._status_widget = None
        self._snapshot: Project2 | None = None
        self._last_text = ""
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

    # ------------------------------------------------------------ state
    @property
    def running(self) -> bool:
        return self.phase != "idle"

    def progress(self) -> float:
        """The fraction of the run done, 0..1."""
        return self.track.fraction() if self.track is not None else 0.0

    def titler(self):
        m = self.model
        return lambda stage: m.title(stage)

    def status_text(self) -> str:
        t = self.track
        if self.phase == "idle" or t is None:
            return ""
        if self.phase == "starting" or not t.running:
            return _("Starting the run…")
        if self.phase == "stopping":
            return _("Stopping…")
        j = t.current()
        left = t.remaining()
        what = self.model.title(j.stage) if j is not None else ""
        pct = int(round(100 * t.fraction()))
        text = (_("Running: {stage} · {pct} %").format(stage=what, pct=pct) if what
                else _("Running · {pct} %").format(pct=pct))
        return text + " · " + _("≈ {t} left").format(t=hm(left)) if left > 30 else text

    # ------------------------------------------------------------ planning
    def launch(self):
        """Run all (F5): the dialog of the plan, or the progress of the run in progress."""
        if self.running:
            return self.show_dialog()
        try:
            plan = self.model.plan_all()
        except (ValueError, KeyError, OSError):
            plan = api.Plan()
        return self.open_plan(plan, lambda: _("Run project"), show_done=True)

    def run_to_here(self):
        """Run to the selected block (Shift+F5): the blocks it needs, in the current workflow."""
        m = self.model
        p = m.path
        if self.running:
            return self.show_dialog()
        if p is None or m.sel_step in ("sources", "final"):
            return None
        from ..blocks import block_title
        key = m.sel_step
        plan = m.plan_to_block(p, key)
        return self.open_plan(plan, lambda: _('Run "{name}" to {block}').format(
            name=p.name, block=_("Output") if key == "post" else block_title(p, key)))

    def open_plan(self, plan, title, show_done: bool = False):
        """The plan dialog (a snapshot of the definition is taken now: the plan is frozen). ``title``: the text, or a
        function giving it (so that a change of language rewrites it)."""
        from ..runs import RunDialog
        self._snapshot = Project2.from_dict(self.model.p.to_dict(), self.model.p.root)
        self._drop_dialog()
        self.plan, self.title = plan, title
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
        self.plan = plan
        self.track = RunTrack.of_plan(plan)
        self.track.plan = plan
        self.names = {}
        self.phase = "starting"
        self.track.lines.append((time.time(), "starting", {"n": len(plan.jobs)}))
        self.set_console()
        self.worker.submit(self.ch_start, _start_job, self._snapshot, plan, self.model.device)
        self._emit_state(True)
        return True

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
            if "already in progress" in why:     # another window, or a run started before this client: follow it
                self.track = None
                self._drop_dialog()
                self.attach()
            else:
                self.append_console(self.track.abort(why.split(": ", 1)[-1]).lines)
            self._emit_state(False)
        elif channel == self.ch_attach:
            self._attaching = False

    def _started(self, info):
        t = self.track
        t.run_id, t.pid = info.run_id, info.pid
        self.phase = "running"
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
        """A stage started or ended: the model must read the states again, soon."""
        self._refresh_due = True
        self._refresh_model()

    def _refresh_model(self):
        """Tell the model that results changed on disk: at once the first time, then at most every REFRESH_MS."""
        if not self._refresh_due:
            return
        wait = REFRESH_MS / 1000 - (time.monotonic() - self._refresh_at)
        if wait > 0:
            if not self._refresh_timer.isActive():
                self._refresh_timer.start(int(wait * 1000) + 1)
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
        n, c = t.elapsed(), t.counts or t.count()
        if t.status == "done":
            text = _("Run finished: {n} stages in {t}").format(n=c[DONE], t=hm(n))
            self.model.log("Run: {n} stages · {t}", n=c[DONE], t=hm(n))
        elif t.status == "interrupted":
            text = _("Run stopped: {n} stages done").format(n=c[DONE])
            self.model.log("Run stopped: {n} stages done", n=c[DONE])
        else:
            text = _("Run failed: {failed} failed, {skipped} not run").format(
                failed=c[FAILED], skipped=c[SKIPPED] + c[NOT_STARTED])
            self.model.log("Run failed: {failed} failed, {skipped} not run", failed=c[FAILED],
                           skipped=c[SKIPPED] + c[NOT_STARTED])
        self._last_text = text
        self.phase = "idle"
        c_ = self._console_obj()
        if c_ is not None and hasattr(c_, "refresh"):
            c_.refresh()
        self.updated.emit()
        self._emit_state(False)
        self.finished.emit(t.status, text)
        self.show_dialog()

    def _emit_state(self, running: bool):
        self.runningChanged.emit(running)
        self.statusChanged.emit(self.status_text())
        self.updated.emit()

    # ------------------------------------------------------------ stopping
    def stop(self):
        """Stop the run at the next check of the stage in progress: what is computed is kept, and the next run
        goes on from the stage's last checkpoint."""
        if self.phase != "running":
            return
        self.phase = "stopping"
        self.track.stop_asked = True
        self.track.lines.append((time.time(), "stop_asked", {}))
        self.append_console()
        self.worker.submit(self.ch_stop, _stop_job, self.model.p)
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
        self.track = RunTrack.of_plan(plan, info.run_id)
        self.track.plan, self.track.pid = plan, info.pid
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
            c.set_console("\n".join(render_line(x, self.model.title) for x in self.track.lines))

    sync_console = set_console

    def append_console(self, lines=None):
        c = self._console_obj()
        if c is None or self.track is None:
            return
        for x in (self.track.lines[-1:] if lines is None else lines):
            c.append_console(render_line(x, self.model.title))

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
        self._drop_dialog()
        self.phase = "idle"
        for sig, slot in ((self.worker.done, self._done), (self.worker.failed, self._failed)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        if self._own_worker:
            self.worker.stop()
