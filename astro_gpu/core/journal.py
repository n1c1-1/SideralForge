"""Processing journal: an append-only JSON-lines event stream plus a readable Markdown log.

The worker process writes events; the GUI tails ``logs/events.jsonl``. Event types:

    run_start, run_end            a pipeline run (``stages``, ``status``)
    stage_start, stage_end        ``stage``, ``status`` (done | failed | interrupted | skipped), ``duration``
    progress                      ``stage``, ``frac`` (0..1), ``msg``, ``eta`` (seconds, optional)
    log                           ``stage``, ``level``, ``msg``
    metric                        ``stage``, ``key``, ``value``
    preview                       ``stage``, ``name``, ``path`` (an image the GUI can show)
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any


class Journal:
    def __init__(self, log_dir: Path, echo: bool = True):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events = self.dir / "events.jsonl"
        self.md = self.dir / "journal.md"
        self.echo = echo
        self._lock = threading.Lock()

    def emit(self, type_: str, **kw: Any) -> None:
        ev = {"t": time.time(), "type": type_, **kw}
        line = json.dumps(ev, default=_jsonable)
        with self._lock:
            with open(self.events, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if type_ in ("stage_start", "stage_end", "log", "metric", "run_start", "run_end"):
                with open(self.md, "a", encoding="utf-8") as f:
                    f.write(_md_line(ev) + "\n")
        if self.echo and type_ != "progress":
            _echo(_md_line(ev))
        elif self.echo and type_ == "progress" and kw.get("msg"):
            _echo(f"    [{kw.get('stage')}] {100 * kw.get('frac', 0):5.1f} %  {kw['msg']}")


def _echo(text: str) -> None:
    """Print to the console. A worker's stdout redirected to a file on Windows has the locale's code page, which has no
    σ: the line is then written with a replacement character instead of failing the stage."""
    import sys
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc), flush=True)


def _jsonable(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def _md_line(ev: dict) -> str:
    ts = datetime.fromtimestamp(ev["t"]).strftime("%Y-%m-%d %H:%M:%S")
    t = ev["type"]
    st = ev.get("stage", "")
    if t == "stage_start":
        return f"- {ts} **{st}** started"
    if t == "stage_end":
        dur = ev.get("duration")
        return f"- {ts} **{st}** {ev.get('status')}" + (f" in {dur / 60:.1f} min" if dur else "")
    if t == "log":
        return f"- {ts} [{st}] {ev.get('msg')}"
    if t == "metric":
        return f"- {ts} [{st}] {ev.get('key')} = {ev.get('value')}"
    if t == "run_start":
        return f"\n## Run {ts} - stages: {', '.join(ev.get('stages', []))}"
    if t == "run_end":
        return f"- {ts} run {ev.get('status')}"
    return f"- {ts} {t} {st}"


def read_events(path: Path, start: int = 0) -> tuple[list[dict], int]:
    """Read events appended since byte offset ``start``; returns (events, new offset)."""
    path = Path(path)
    if not path.exists():
        return [], start
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read()
    end = data.rfind(b"\n")
    if end < 0:
        return [], start
    out = []
    for line in data[:end].splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out, start + end + 1


def tail(path: Path) -> Iterator[dict]:
    pos = 0
    while True:
        evs, pos = read_events(path, pos)
        yield from evs
        time.sleep(0.5)
