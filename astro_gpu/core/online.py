"""Online services given little time: a short connect timeout, a total budget per query, and a service that does not
answer is set aside for a while so that the next source is asked at once, with one line in the log saying so.

Why: VizieR, SIMBAD, the CDS cut-out service (all at the CDS, Strasbourg), GAVO and the ESA Gaia archive are often
down or silent, sometimes for days (no connection at all, or a connection that never answers). Without a budget, a
query waits its whole 600-900 s timeout before the next source is tried, for a computation of a few seconds.

How:
- ``available(name)``: a tiny request to the service (connect 8 s, answer 12 s), once per 10 minutes; a service that
  fails it is set aside for 10 minutes. The services of a chain are probed together, so a silent one costs ~20 s once.
- ``call(name, fn, budget)``: ``fn`` runs in a background thread for at most ``budget`` seconds; a network failure or a
  query over budget sets the service aside and raises ``ServiceDown``.
- ``first_answer(chain, what, say)``: the sources of a chain in order, one plain line per fallback.
- ``ASTRO_GPU_OFFLINE=1`` sets every service aside (no network at all).

THE RULE: NOTHING IS DOWNLOADED DURING A RUN. A run (the runner's worker process, and any stage run in another process:
``runner.Pipeline._run_stage``) switches ``forbid`` on; from then on every way to the network raises
``NetworkForbidden`` at once, with what was asked and of whom: the request layer here (``call``, ``first_answer``,
``available``, ``probe_all``, ``requests_session``, ``get``), and below it every socket of the process (an audit hook
on ``socket.connect`` and the name lookups: a library that does not go through this module is stopped too; the
machine's own address is left alone). The stages read only what is on the disk (the offline Gaia extracts, the caches);
what a run needs is checked before it starts (``core/sky_data.py``) and fetched by an explicit action of the user
outside the run (Settings › Catalogs & astrometry, the Gradient block's Get survey). tests/test_no_download_in_run.py
holds the rule.
"""
from __future__ import annotations

import os
import socket
import threading
import time
import urllib.error
import weakref
from collections.abc import Callable
from typing import Any

CONNECT_S = 15.0            # seconds to open a connection
READ_S = 30.0               # seconds a server may stay silent in the middle of a query
PROBE_S = (8.0, 12.0)       # (connect, answer) of the reachability probe
DOWN_FOR_S = 600.0          # a service that failed is set aside for this long

LABELS = {
    "vizier": "VizieR (CDS, Strasbourg)",
    "vizier_cfa": "the VizieR mirror at Harvard (CfA)",
    "gavo": "GAVO (Heidelberg)",
    "esa": "the ESA Gaia archive",
    "simbad": "SIMBAD (CDS)",
    "ned": "NED (NASA/IPAC)",
    "hips2fits": "the CDS cut-out service",
    "hips": "the survey's own tile server",
}
VIZIER_HOSTS = {"vizier": "vizier.cds.unistra.fr", "vizier_cfa": "vizier.cfa.harvard.edu"}
PROBES = {
    "vizier": "https://vizier.cds.unistra.fr/viz-bin/votable?-source=I/239/hip_main&-out=HIP&-out.max=1",
    "vizier_cfa": "https://vizier.cfa.harvard.edu/viz-bin/votable?-source=I/239/hip_main&-out=HIP&-out.max=1",
    "gavo": "https://dc.zah.uni-heidelberg.de/tap/availability",
    "esa": "https://gea.esac.esa.int/tap-server/tap/availability",
    "simbad": "https://simbad.cds.unistra.fr/simbad/sim-tap/availability",
    "ned": "https://ned.ipac.caltech.edu/tap/availability",
    "hips2fits": "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
}

#: queries a service takes at once (GAVO dropped the connection with the catalogue stage's 6 parallel queries)
SLOTS: dict[str, threading.Semaphore] = {"gavo": threading.Semaphore(2)}

_LOCK = threading.Lock()
_DOWN: dict[str, tuple[float, str]] = {}        # name -> (time until which it is set aside, why)
_PROBED: dict[str, float] = {}                  # name -> time of the last successful probe
_PROBE_LOCKS: dict[str, threading.Lock] = {}
_SAID: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class ServiceDown(Exception):
    """A service did not answer (or failed); ``why`` says it in plain words."""

    def __init__(self, name: str, why: str):
        super().__init__(f"{label(name)} {why}")
        self.name, self.why = name, why


class NetworkForbidden(BaseException):
    """The network was asked while a run forbids it (``forbid``). A BaseException, as KeyboardInterrupt: the broad
    ``except Exception`` of a fallback cannot take it for a service that does not answer and go on; the runner makes it
    the stage's failure, with this message."""


def label(name: str | None) -> str:
    return LABELS.get(name or "", name or "")


# ------------------------------------------------------------------ the guard: no network during a run

#: why the network is forbidden now ("the Astrometry block", "this run"), innermost last; empty: allowed. Process-wide,
#: not per thread: the threads a stage starts (the sky tiles, the spectra) are stopped as the stage itself
_FORBID: list[str] = []
#: the socket events stopped while forbidden, and the addresses left alone (the machine itself: a pipe of a library)
_SOCKET_EVENTS = frozenset({"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
                            "socket.gethostbyaddr", "socket.sendto", "socket.sendmsg"})
_LOCAL = frozenset({None, "", "localhost", "127.0.0.1", "::1", b"localhost", b"127.0.0.1", b"::1"})
_HOOKED: list[bool] = []


def forbidden() -> str | None:
    """Why the network is forbidden now (the run, the block running), None when it is allowed."""
    return _FORBID[-1] if _FORBID else None


def _refuse(what: str) -> NetworkForbidden:
    return NetworkForbidden(f"nothing is downloaded during a run: asked for {what} while {_FORBID[-1]} ran. What a run "
                            "needs comes from the disk: install it first (Settings › Catalogs & astrometry)")


def guard(what: str) -> None:
    """Raise ``NetworkForbidden`` at once when the network is forbidden; ``what``: what was asked, of whom."""
    if _FORBID:
        raise _refuse(what)


def _audit(event: str, args: tuple) -> None:
    """The socket net under the request layer (an audit hook: it costs one test of a list per audited event)."""
    if not _FORBID or event not in _SOCKET_EVENTS:
        return
    if event.startswith("socket.get"):                       # a name lookup: (host, ...)
        host = args[0] if args else None
    else:                                                    # (socket, address): (host, port[, ...]), a pipe's path
        addr = args[1] if len(args) > 1 else None
        if not isinstance(addr, tuple):
            return                                           # AF_UNIX and the like: the machine itself
        host = addr[0] if addr else None
    if host in _LOCAL or (isinstance(host, str) and host.startswith("127.")):
        return
    raise _refuse(f"a connection to {host if isinstance(host, str) else repr(host)}")


def _hook() -> None:
    if not _HOOKED:                              # an audit hook cannot be taken out: added once, read through _FORBID
        import sys
        sys.addaudithook(_audit)
        _HOOKED.append(True)


class forbid:  # noqa: N801 - used as a function: ``with online.forbid("the Astrometry block"):``
    """No network while the block runs (nested: the innermost reason is said). ``forbid(why).on()`` without ``with``:
    for the rest of the process (the runner's worker)."""

    def __init__(self, why: str):
        self.why = why

    def on(self) -> forbid:
        _hook()
        _FORBID.append(self.why)
        return self

    def __enter__(self) -> forbid:
        return self.on()

    def __exit__(self, *exc) -> None:
        if self.why in _FORBID:
            del _FORBID[len(_FORBID) - 1 - _FORBID[::-1].index(self.why)]


def offline() -> bool:
    return os.environ.get("ASTRO_GPU_OFFLINE", "").strip() not in ("", "0", "false", "no")


def is_down(name: str) -> bool:
    if offline():
        return True
    with _LOCK:
        d = _DOWN.get(name)
    return d is not None and time.monotonic() < d[0]


def why_down(name: str) -> str:
    if offline():
        return "is not asked (offline mode)"
    with _LOCK:
        d = _DOWN.get(name)
    return d[1] if d else "does not answer"


def mark_down(name: str, why: str = "does not answer") -> None:
    with _LOCK:
        _DOWN[name] = (time.monotonic() + DOWN_FOR_S, why)
        _PROBED.pop(name, None)


def reset() -> None:
    """Forget every verdict (tests; a new session)."""
    with _LOCK:
        _DOWN.clear()
        _PROBED.clear()


def _probe(name: str) -> bool:
    url = PROBES.get(name)
    if not url:
        return True
    import requests
    try:
        r = requests.get(url, timeout=PROBE_S)
        # a probe asks a fixed page that exists: "not found" or "gone" there is the service's own outage page (the CDS
        # answered its VizieR queries with a 404 page during its November 2026 incident), not a wrong query; it was
        # taken for an answer, and every first query of a chain lost a request on VizieR before the next source
        ok = r.status_code < 500 and r.status_code not in (404, 410)
        why = f"answers with an error (HTTP {r.status_code})"
    except Exception as e:  # noqa: BLE001 - any failure means: not reachable now
        ok, why = False, _plain(e)
    if not ok:
        mark_down(name, why)
    return ok


def available(name: str) -> bool:
    """False when the service is set aside or does not answer a tiny request (asked once per 10 minutes)."""
    guard(f"an answer of {label(name)}")
    if is_down(name):
        return False
    if name not in PROBES:
        return True
    with _LOCK:
        lk = _PROBE_LOCKS.setdefault(name, threading.Lock())
    with lk:                                     # the threads of a stage wait for one probe
        if is_down(name):
            return False
        with _LOCK:
            t = _PROBED.get(name)
        if t is not None and time.monotonic() - t < DOWN_FOR_S:
            return True
        ok = _probe(name)
        if ok:
            with _LOCK:
                _PROBED[name] = time.monotonic()
        return ok


def probe_all(names) -> None:
    """Start probing the services of a chain at once, in the background (a fallback then waits only for what is
    left of its own probe, not for one probe after the other; nobody waits for the probe of a service not asked)."""
    guard(f"an answer of {', '.join(label(n) for n in names)}")
    with _LOCK:
        fresh = {n for n, t in _PROBED.items() if time.monotonic() - t < DOWN_FOR_S}
    for n in names:
        if n in PROBES and n not in fresh and not is_down(n):
            threading.Thread(target=available, args=(n,), daemon=True, name=f"probe-{n}").start()


def network_error(e: BaseException) -> bool:
    """Whether an exception says the service is unreachable / failing (not: our query was wrong)."""
    if isinstance(e, NetworkForbidden):
        return False
    try:
        import requests
        if isinstance(e, (requests.ConnectionError, requests.Timeout)):
            return True
        if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code >= 500:
            return True
    except ImportError:
        pass
    if isinstance(e, urllib.error.HTTPError):
        return e.code >= 500
    if isinstance(e, (TimeoutError, ConnectionError, socket.timeout, urllib.error.URLError, OSError)):
        return True
    txt = f"{type(e).__name__} {e}".lower()
    return any(k in txt for k in ("timed out", "timeout", "connection", "unreachable", "503", "502", "504",
                                  "service unavailable", "remote end closed"))


def unreachable(e: BaseException) -> bool:
    """The service cannot be reached at all (no connection, no name, refused): not merely slow on a query."""
    w = _plain(e)
    return w.startswith(("does not answer (no connection)", "cannot be reached", "refuses"))


def _plain(e: BaseException) -> str:
    name, msg = type(e).__name__.lower(), str(e).lower()
    txt = f"{name} {msg}"
    if "connecttimeout" in name or "connect timeout" in msg or "connecttimeouterror" in msg:
        return "does not answer (no connection)"
    if "timed out" in txt or "timeout" in txt:
        return "does not answer (silent)"
    if "name resolution" in txt or "getaddrinfo" in txt or "nodename" in txt:
        return "cannot be reached (no network?)"
    if "refused" in txt:
        return "refuses the connection"
    return f"failed ({str(e)[:120]})"


def call(name: str, fn: Callable[[], Any], budget: float, what: str = "data") -> Any:
    """``fn()`` in a background thread for at most ``budget`` seconds. Raises ``ServiceDown`` when the service is set
    aside or unreachable, fails with a network error, or takes longer than the budget (the thread is abandoned; its own
    socket timeouts end it); ``NetworkForbidden`` during a run (``what``: what is asked, said then)."""
    guard(f"{what} from {label(name)}")
    if not available(name):
        raise ServiceDown(name, why_down(name))
    box: dict = {}

    def run():
        try:
            box["r"] = fn()
        except BaseException as e:  # noqa: BLE001 - handed to the caller
            box["e"] = e
    slots = SLOTS.get(name)
    if slots is not None:                       # waiting for a slot is not counted in the budget
        slots.acquire()
    try:
        t = threading.Thread(target=run, daemon=True, name=f"online-{name}")
        t.start()
        t.join(budget)
    finally:
        if slots is not None:
            slots.release()
    if t.is_alive():
        why = f"took more than {budget:.0f} s"
        mark_down(name, why)
        raise ServiceDown(name, why)
    if "e" in box:
        e = box["e"]
        if network_error(e):
            why = _plain(e)
            mark_down(name, why)
            raise ServiceDown(name, why) from e
        raise e
    return box["r"]


def sayer(log: Callable | None) -> Callable[[str], None]:
    """A function that logs a message once per log target (a stage context's log, or print): one line per fallback,
    not one per sky tile."""
    if log is None:
        return lambda msg, level="warning": None
    owner = getattr(log, "__self__", log)
    try:
        said = _SAID.setdefault(owner, set())
    except TypeError:                            # not weak-referenceable: no deduplication
        said = set()

    def say(msg: str, level: str = "warning") -> None:
        if msg in said:
            return
        said.add(msg)
        try:
            log(msg, level)
        except TypeError:
            log(msg)
    return say


def first_answer(chain: list[tuple[str, Callable[[], Any]]], what: str, say: Callable | None = None,
                 budget: float = 300.0) -> Any:
    """The answer of the first source of ``chain`` ([(service, fn)]) that gives one; one line per fallback, e.g.
    "VizieR (CDS, Strasbourg) does not answer (no connection): Gaia stars from GAVO (Heidelberg)". Raises
    ``ServiceDown`` (of the last source) when none answers."""
    guard(f"{what} from {label(chain[0][0]) if chain else 'a service'}")
    say = say or (lambda msg, level="warning": None)
    probe_all([n for n, _ in chain])
    last: ServiceDown | None = None
    for i, (name, fn) in enumerate(chain):
        try:
            return call(name, fn, budget, what)
        except Exception as e:  # noqa: BLE001 - this source failed: the next one
            last = e if isinstance(e, ServiceDown) else ServiceDown(name, f"failed ({str(e)[:120]})")
            nxt = chain[i + 1][0] if i + 1 < len(chain) else None
            say(f"{label(name)} {last.why}: {what} from {label(nxt)}" if nxt else
                f"{label(name)} {last.why}: no other source for {what}")
    raise last if last is not None else ServiceDown("", f"no source for {what}")


def requests_session(timeout: tuple[float, float] | None = None):
    """A requests session whose requests have connect / read timeouts by default (pyvo sets none)."""
    guard("a connection to an online service")
    import requests

    class _Session(requests.Session):
        def request(self, *a, **k):
            k.setdefault("timeout", timeout or (CONNECT_S, READ_S))
            return super().request(*a, **k)
    s = _Session()
    try:
        from pyvo.utils.http import create_session
        s.headers.update(create_session().headers)
    except Exception:  # noqa: BLE001 - only the user agent
        pass
    return s


def get(url: str, timeout: float = 60.0) -> bytes:
    """The bytes at ``url`` (one GET; for ``call``): the only plain download of the stages' modules, through the guard."""
    guard(f"the file {url}")
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()
