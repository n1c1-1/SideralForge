"""NOTHING IS DOWNLOADED DURING A RUN. Said by the user many times, and on 11 November 2026 once more: a real run waited
6 minutes in the Catalogues block on three remote services in a row, and an Astrometry downloaded hundreds of sky
tiles, while the offline star catalogue was installed on the machine. A run reads what it needs from the disk (the
offline Gaia extracts, the caches); what it needs is checked before it starts (core/sky_data.py) and fetched only by
an explicit action of the user, outside the run (Settings › Catalogs & astrometry, the Gradient block's Get survey).

The guard (core/online.py): the runner switches ``online.forbid`` on while a stage runs, and for the whole life of a
run's worker process; then the request layer and every socket of the process raise ``NetworkForbidden`` at once.

If this file fails, a change lets a run reach the network: fix the change, NEVER this test. Do not weaken it, skip it,
mark it slow, or allow one more entry point. A new stage that needs data of the sky reads it from the disk, says what
it needs in core/sky_data.py, and leaves the download to an explicit action outside the run."""
from __future__ import annotations

import ast
import json
import socket
from pathlib import Path

import pytest

from astro_gpu.core import online, project2, runner2, sirilcat
from astro_gpu.core.project import Target
from astro_gpu.core.workflow import TEMPLATES, node
from tests import skyfield

ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "astro_gpu" / "stages"
#: the libraries that reach the network: a stage module never uses them but in a function it hands to the request
#: layer (online.call / online.first_answer), which the guard stops during a run
NETWORK = ("socket", "requests", "urllib.request", "urllib3", "http.client", "httpx", "aiohttp", "ftplib",
           "astroquery", "pyvo")


@pytest.fixture
def offline_sky(tmp_path, monkeypatch):
    """A private home and cache (no download of the user's to read), no other program's catalogue folders."""
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "userhome"))
    for v in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(v, str(tmp_path / "nowhere"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    monkeypatch.delenv("ASTRO_GPU_OFFLINE", raising=False)    # not set aside by the offline mode: stopped by the guard
    from astro_gpu.stages import astrometry
    monkeypatch.setattr(astrometry, "_GAIA_CACHE", {})          # the stars an earlier test's run kept in this process
    online.reset()
    sirilcat.reset()
    yield tmp_path
    sirilcat.reset()
    online.reset()


@pytest.fixture
def no_socket(monkeypatch):
    """Every way a Python program reaches the network, made to record who touched it and fail."""
    touched: list[str] = []

    def stop(name):
        def f(*a, **k):
            touched.append(f"{name}{a[1:2] if name.startswith('socket.socket') else a[:1]}")
            raise AssertionError(f"the network was touched during a run: {name}")
        return f
    for owner, attr, name in ((socket.socket, "connect", "socket.socket.connect"),
                              (socket.socket, "connect_ex", "socket.socket.connect_ex"),
                              (socket, "create_connection", "socket.create_connection"),
                              (socket, "getaddrinfo", "socket.getaddrinfo"),
                              (socket, "gethostbyname", "socket.gethostbyname")):
        monkeypatch.setattr(owner, attr, stop(name))
    import http.client
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", stop("urllib.request.urlopen"))
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", stop("urllib.request.OpenerDirector.open"))
    monkeypatch.setattr(http.client.HTTPConnection, "connect", stop("http.client.HTTPConnection.connect"))
    try:
        import requests
        monkeypatch.setattr(requests.Session, "request", stop("requests.Session.request"))
    except ImportError:
        pass
    return touched


def _events(root: Path) -> list[dict]:
    return [json.loads(x) for x in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def test_a_chain_run_computes_from_the_disk_and_reaches_no_socket(offline_sky, no_socket, monkeypatch):
    """A small chain from an image (Astrometry, its Catalogues and Survey, Color, the Output), the field's offline
    extracts installed, no cache at all, every network entry point made to fail: the run is done, every stage from the
    disk; the survey of the field, not on the disk, gives 'none' with its reason (its documented way); no socket was
    touched and nothing asked the request layer."""
    sky = skyfield.field(offline_sky / "sky")
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(sky["extracts"]))
    sirilcat.reset()
    p = project2.create(offline_sky / "P", "P", Target(**sky["target"]))
    w = p.add_workflow("Image", TEMPLATES["Classic stack"])
    w.source = {"path": str(sky["image"]), "stretched": False}
    p.set_blocks(w.id, [node("astrometry"), node("color")])
    project2.save(p)
    plan = runner2.plan_all(p)
    assert {"astrometry", "catalogs", "surveys", "color"} <= {j.stage for j in plan.jobs}
    status = runner2.run_plan(p.root, runner2.prepare_run(p, plan, "cpu").plan_file, echo=False)
    ev = _events(p.root)
    ends = {e["stage"]: e["status"] for e in ev if e.get("type") == "stage_end"}
    errors = [e.get("msg") for e in ev if e.get("level") == "error"]
    assert no_socket == [], f"the network was touched: {no_socket}"
    assert not any("nothing is downloaded" in (m or "") for m in errors), errors
    assert status == "done" and all(s == "done" for s in ends.values()), (ends, errors)
    said = [e.get("msg") for e in ev if e.get("stage") == "surveys" and e.get("type") == "log"]
    assert any("not on this computer" in m for m in said), said
    assert any("from the offline Gaia extract" in (e.get("msg") or "") for e in ev if e.get("stage") == "astrometry")
    assert online.forbidden() is None                           # the run's stages switched it off after themselves


def test_the_guard_stops_a_request_and_a_socket_at_once_with_what_was_asked(monkeypatch):
    monkeypatch.setattr(online, "_FORBID", [])
    asked = []
    with online.forbid("the Catalogues step"):
        assert online.forbidden() == "the Catalogues step"
        with pytest.raises(online.NetworkForbidden, match=r"nothing is downloaded during a run: asked for XP spectra "
                                                          r"from VizieR \(CDS, Strasbourg\) while the Catalogues step ran"):
            online.call("vizier", lambda: asked.append(1), 5.0, "XP spectra")
        with pytest.raises(online.NetworkForbidden, match="Gaia stars from GAVO"):
            online.first_answer([("gavo", lambda: asked.append(2))], "Gaia stars")
        for f in (lambda: online.available("vizier"), lambda: online.probe_all(["gavo"]), online.requests_session,
                  lambda: online.get("https://example.org/tile.fits")):
            with pytest.raises(online.NetworkForbidden):
                f()
        # under the request layer: a socket of any library, stopped before its name is even looked up
        with pytest.raises(online.NetworkForbidden, match="example.org"):
            socket.getaddrinfo("example.org", 443)
        with pytest.raises(online.NetworkForbidden, match="203.0.113.9"):
            socket.create_connection(("203.0.113.9", 80), timeout=0.01)
        # the machine itself stays reachable (the pipes some libraries make with a local socket)
        a, b = socket.socketpair()
        a.close()
        b.close()
        # a fallback's broad except cannot take it for a service that does not answer
        try:
            online.call("esa", lambda: 1, 5.0)
        except Exception:  # noqa: BLE001 - what a stage's fallback does
            pytest.fail("NetworkForbidden was caught as an ordinary failure")
        except online.NetworkForbidden:
            pass
    assert asked == [] and online.forbidden() is None


def test_a_stage_that_asks_the_network_fails_with_the_plain_reason(tmp_path, monkeypatch):
    """The runner's own path (Pipeline._run_stage): the stage asks, the step fails at once with the reason; the next
    stage runs without the network either, and the network is allowed again after the run."""
    from astro_gpu.core.journal import Journal
    from astro_gpu.core.project import Inputs, Project
    from astro_gpu.core.runner import Pipeline, read_manifest
    from astro_gpu.core.stage import Stage
    monkeypatch.setattr(online, "_FORBID", [])

    class Asks(Stage):
        name, title = "asks", "Asking"

        def run(self, ctx, p):
            try:
                online.call("simbad", lambda: (1.0, 2.0), 20.0, "the position of 'M 1'")
            except Exception:  # noqa: BLE001 - a fallback of earlier days: must not take it
                ctx.log("went on")
    root = tmp_path / "v1"
    proj = Project.create(root, "v1", Inputs(), scratch_dir=tmp_path / "scratch")
    pipe = Pipeline(proj, [Asks()], Journal(root / "logs", echo=False))
    assert pipe._run_stage(Asks()) == "failed"
    m = read_manifest(proj, "asks")
    assert m["error"].startswith("nothing is downloaded during a run: asked for the position of 'M 1' from SIMBAD "
                                 "(CDS) while the Asking step ran")
    assert online.forbidden() is None


def test_the_worker_of_a_run_forbids_the_network_for_its_whole_life(tmp_path, monkeypatch):
    monkeypatch.setattr(online, "_FORBID", [])
    seen = []
    monkeypatch.setattr(runner2, "run_plan", lambda root, plan: seen.append(online.forbidden()) or "done")
    monkeypatch.chdir(tmp_path)
    assert runner2.main([str(tmp_path), str(tmp_path / "plan.json")]) == 0
    assert seen == ["this run"]


def _imports(fn: ast.AST) -> list[tuple[str, int]]:
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Import):
            out += [(a.name, n.lineno) for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            out.append((n.module, n.lineno))
    return [(m, ln) for m, ln in out if any(m == x or m.startswith(x + ".") for x in NETWORK)]


def _handed_to_the_request_layer(top: ast.AST) -> set[str]:
    """The names a top-level function uses when it calls online.call / online.first_answer: the functions it hands to
    the request layer (as an argument, or in the list of sources it goes through)."""
    calls = [n for n in ast.walk(top) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in ("call", "first_answer") and isinstance(n.func.value, ast.Name)
             and n.func.value.id == "online"]
    if not calls:
        return set()
    return {n.id for n in ast.walk(top) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def test_no_stage_module_reaches_the_network_outside_the_request_layer():
    """Every module of the stages, read: no network library imported at the module's level, and inside a function only
    when that function is handed to the request layer, which the guard stops during a run. A stage that needs data of
    the sky reads the disk (core/skycache.py, core/sirilcat.py)."""
    bad = []
    mods = sorted(STAGES.glob("*.py"))
    assert len(mods) > 15
    for f in mods:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                bad += [f"{f.name}:{ln} imports {m} at its top" for m, ln in _imports(n)]
        funcs = (ast.FunctionDef, ast.AsyncFunctionDef)
        tops = [n for n in tree.body if isinstance(n, funcs)] + \
            [m for c in tree.body if isinstance(c, ast.ClassDef) for m in c.body if isinstance(m, funcs)]
        for top in tops:
            handed = _handed_to_the_request_layer(top)
            inner = [x for x in ast.walk(top) if isinstance(x, funcs) and x is not top]
            for m, ln in _imports(top):
                # the innermost function holding the import: allowed only when the request layer is handed it
                around = [x for x in inner if x.lineno <= ln <= (x.end_lineno or x.lineno)]
                fn = max(around, key=lambda x: x.lineno) if around else top
                if fn is top or fn.name not in handed:
                    bad.append(f"{f.name}:{ln} {fn.name}() imports {m} outside a function handed to online.call")
    assert bad == [], "a stage module reaches the network outside the guarded request layer:\n" + "\n".join(bad)
