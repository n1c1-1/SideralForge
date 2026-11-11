"""The Windows installer's parts that can be tested without Windows being clean: which card, which PyTorch build, what
the lock file holds, the disk check, the download (cut, resumed, damaged). The real installation is tested by hand with
the .exe (see installer/build.py)."""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import json
import ssl
import sys
import threading
import tomllib
import urllib.error
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer" / "setup_env"))

import detect  # noqa: E402
import fetch  # noqa: E402
import setup_env  # noqa: E402

LOCK = json.loads((ROOT / "installer" / "lock" / "lock.json").read_text(encoding="utf-8"))
NVIDIA = detect.Gpu("nvidia", "GeForce RTX 4060 Ti", (610, 88))


# ------------------------------------------------------------------ driver numbers
@pytest.mark.parametrize("text,value", [("610.88", (610, 88)), (" 572.13 ", (572, 13)), ("", None), ("abc", None)])
def test_parse_driver(text, value):
    assert detect.parse_driver(text) == value


@pytest.mark.parametrize("text,value", [("32.0.15.8088", (580, 88)), ("32.0.16.1088", (610, 88)), ("31.0.15.2849", (528, 49)),
                                        ("nonsense", None)])
def test_parse_registry_driver(text, value):
    assert detect.parse_registry_driver(text) == value


# ------------------------------------------------------------------ the choice of the build
@pytest.mark.parametrize("gpus,build,why", [
    ([NVIDIA], "cu128", "ok"),
    ([detect.Gpu("nvidia", "Old", detect.CUDA_MIN_DRIVER)], "cu128", "ok"),
    ([detect.Gpu("nvidia", "Old", (474, 89))], "cpu", "old-driver"),
    ([detect.Gpu("nvidia", "Mystery", None)], "cpu", "no-driver"),
    ([detect.Gpu("amd", "Radeon RX 7600")], "cpu", "other-vendor"),
    ([detect.Gpu("intel", "Arc A770")], "cpu", "other-vendor"),
    ([], "cpu", "no-card"),
    ([detect.Gpu("intel", "UHD 770"), NVIDIA], "cu128", "ok"),
])
def test_choose(gpus, build, why):
    choice = detect.choose(gpus)
    assert (choice.build, choice.why) == (build, why)


def test_detect_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("no driver")
    monkeypatch.setattr(detect, "_smi_gpus", boom)
    assert detect.detect({}).build == "cpu"


@pytest.mark.parametrize("value,build,why", [("none", "cpu", "no-card"), ("nvidia:610.88", "cu128", "ok"),
                                             ("nvidia:400.00", "cpu", "old-driver"), ("amd:Radeon", "cpu", "other-vendor")])
def test_detect_test_variable(value, build, why):
    choice = detect.detect({detect.TEST_VARIABLE: value})
    assert (choice.build, choice.why) == (build, why)


def test_nvidia_smi_output(monkeypatch):
    class R:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 4060 Ti, 610.88\nNVIDIA GeForce RTX 3070, 610.88\n"

    monkeypatch.setattr(detect.subprocess, "run", lambda *a, **k: R())
    gpus = detect._smi_gpus()
    assert [g.driver for g in gpus] == [(610, 88), (610, 88)] and "4060 Ti" in gpus[0].name


def test_path_limit(tmp_path):
    assert detect.root_limit() == detect.MAX_PATH - detect.MAX_RELATIVE_PATH
    assert detect.root_limit() >= len("C:/Users/" + "x" * 20 + "/AppData/Local/Programs/FLOAT")  # the default folder

    class Ui:
        pass
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"python": "3.12.13", "builds": {}}), encoding="utf-8")
    inst = setup_env.Installer(tmp_path, Ui(), lock, "en", {})
    inst.root = Path("C:/" + "x" * 200)
    with pytest.raises(setup_env.Failure) as e:
        inst.check_folder()
    assert not e.value.retry and "too long" in e.value.message


# ------------------------------------------------------------------ the lock file
@pytest.fixture(params=sorted(LOCK["builds"]))
def build(request):
    return request.param


def test_lock_holds_every_dependency(build):
    with open(ROOT / "pyproject.toml", "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    names = {w["name"].replace("_", "-").lower() for w in LOCK["builds"][build]}
    for d in deps:
        name = d.split(">")[0].split("=")[0].split("<")[0].strip().replace("_", "-").lower()
        assert name in names, f"{name} missing from the {build} build; run: python installer/build.py lock"


def test_lock_is_exact(build):
    for w in LOCK["builds"][build]:
        assert len(w["sha256"]) == 64 and w["size"] > 0 and w["version"] and w["filename"].endswith(".whl")
        assert w.get("bundled") or w["url"].startswith("https://")


def test_lock_torch_builds():
    torch = {b: next(w for w in LOCK["builds"][b] if w["name"] == "torch") for b in LOCK["builds"]}
    assert set(torch) == {detect.CUDA_BUILD, detect.CPU_BUILD}
    assert torch["cu128"]["version"].endswith("+cu128") and torch["cpu"]["version"].endswith("+cpu")
    assert torch["cu128"]["version"].split("+")[0] == torch["cpu"]["version"].split("+")[0]


def test_lock_python_and_numba():
    assert LOCK["python"].startswith("3.12.")
    assert any(w["name"] == "numba" for w in LOCK["builds"]["cu128"])


def test_lock_asterion():
    """The plate solver's released program and its index, each pinned by size and checksum; the index has no file of
    its own to download (the release packs it), so its entries say where the bytes come from instead of a URL."""
    a = LOCK["asterion"]
    files = {f["filename"]: f for f in a["files"]}
    assert set(files) == {f"asterion-{a['version']}-windows-x86_64.zip", "tycho2.astx", "tycho2.astx.wide"}
    assert a["version"] == "0.4.2"
    for f in files.values():
        assert len(f["sha256"]) == 64 and int(f["sha256"], 16) >= 0 and f["size"] > 0
        assert f["url"].startswith("https://") or f["origin"]
    assert files["asterion-0.4.2-windows-x86_64.zip"]["url"].endswith("/v0.4.2/asterion-0.4.2-windows-x86_64.zip")
    assert files["tycho2.astx"]["dest"] == files["tycho2.astx.wide"]["dest"] == "catalogs"


def load_build():
    spec = importlib.util.spec_from_file_location("float_build", ROOT / "installer" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_asterion(cache: Path) -> dict:
    """A cache of small fake files laid out as the real ones (the archive with its top folder, the web interface and
    README it carries, the index one folder down) and the lock part that pins them."""
    cache.mkdir()
    top = "asterion-0.4.2-windows-x86_64"
    with zipfile.ZipFile(cache / f"{top}.zip", "w") as z:
        for name, data in [("asterion.exe", b"MZ program"), ("asterion-web.exe", b"MZ web"), ("README.md", b"r"),
                           ("LICENSE.txt", b"licence"), ("third-party/openngc.md", b"notice")]:
            z.writestr(f"{top}/{name}", data)
    (cache / "catalogs").mkdir()
    (cache / "catalogs" / "tycho2.astx").write_bytes(b"index" * 100)
    (cache / "catalogs" / "tycho2.astx.wide").write_bytes(b"wide" * 10)
    files = []
    for path, dest in [(cache / f"{top}.zip", ""), (cache / "catalogs" / "tycho2.astx", "catalogs"),
                       (cache / "catalogs" / "tycho2.astx.wide", "catalogs")]:
        files.append({"name": path.stem, "version": "0.4.2", "filename": path.name, "url": "", "origin": "test",
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size, "dest": dest})
    return {"asterion": {"version": "0.4.2", "files": files}}


def test_staged_asterion_is_what_float_finds(tmp_path, monkeypatch):
    """The staged folder, put beside ``app`` as the installer does, is the one FLOAT's lookup finds: the program, then
    the index in the ``catalogs`` folder beside it."""
    from astro_gpu.core import asterion
    build_mod = load_build()
    lock = fake_asterion(tmp_path / "cache")
    root = tmp_path / "FLOAT"
    folder = build_mod.stage_asterion(root, lock, tmp_path / "cache", ask_version=False)
    assert folder == root / "asterion"
    assert sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()) == [
        "LICENSE.txt", "asterion.exe", "catalogs/tycho2.astx", "catalogs/tycho2.astx.wide", "third-party/openngc.md"]
    # the module as the installation holds it: app/astro_gpu/core/asterion.py
    here = root / "app" / "astro_gpu" / "core" / "asterion.py"
    assert asterion.install_dir(here) == folder
    monkeypatch.delenv(asterion.ENV_PROGRAM, raising=False)
    monkeypatch.delenv(asterion.ENV_CATALOGS, raising=False)
    monkeypatch.setattr(asterion, "own_dir", lambda: tmp_path / "nobody")
    monkeypatch.setattr(asterion, "own_catalogs", lambda: tmp_path / "nobody" / "catalogs")
    monkeypatch.setattr(asterion, "install_dir", lambda here=None: folder)
    monkeypatch.setattr(asterion, "program_name", lambda: "asterion.exe")   # the Windows name, on any system
    prog = asterion.find_program()
    assert prog == folder / "asterion.exe"
    assert asterion.find_catalogs(None, prog) == folder / "catalogs"
    assert asterion.indexes(folder / "catalogs") == ["tycho2"]


def test_staging_refuses_a_missing_or_different_file(tmp_path):
    build_mod = load_build()
    lock = fake_asterion(tmp_path / "cache")
    (tmp_path / "cache" / "catalogs" / "tycho2.astx").write_bytes(b"index" * 99 + b"other")    # same size, other bytes
    with pytest.raises(SystemExit, match="not the file the lock pins"):
        build_mod.stage_asterion(tmp_path / "a", lock, tmp_path / "cache", ask_version=False)
    (tmp_path / "cache" / "catalogs" / "tycho2.astx").unlink()
    with pytest.raises(SystemExit, match="tycho2.astx is missing"):
        build_mod.stage_asterion(tmp_path / "b", lock, tmp_path / "cache", ask_version=False)


def test_inno_setup_installs_and_removes_the_asterion_folder():
    """The folder comes with the [Files] line that copies the whole stage; an update removes the former one first and
    the uninstaller removes it, by name, as the other folders of the program."""
    text = (ROOT / "installer" / "float.iss").read_text(encoding="utf-8")
    sections = {}
    for block in text.split("\n["):
        head, _, body = block.partition("]")
        sections[head.strip("[")] = body
    line = 'Type: filesandordirs; Name: "{app}\\asterion"'
    assert line in sections["InstallDelete"] and line in sections["UninstallDelete"]
    assert 'Source: "{#StageDir}\\*"' in sections["Files"] and "recursesubdirs" in sections["Files"]


def test_bundled_wheels_are_shipped(build):
    for w in LOCK["builds"][build]:
        if w.get("bundled"):
            f = ROOT / "installer" / "lock" / "wheels" / w["filename"]
            assert hashlib.sha256(f.read_bytes()).hexdigest() == w["sha256"]


# ------------------------------------------------------------------ the disk and the download
def test_needed_bytes():
    w = fetch.Wheel("x", "1", "x.whl", "u", "0" * 64, 1 << 30)
    assert fetch.needed_bytes([]) == 0
    assert fetch.needed_bytes([w]) > 3 << 30


def test_disk_check_blocks_before_download(tmp_path, monkeypatch):
    class Ui:
        def stage(self, text): pass
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"python": "3.12.13", "builds": {}}), encoding="utf-8")
    inst = setup_env.Installer(tmp_path, Ui(), lock, "en", {})
    monkeypatch.setattr(fetch, "free_bytes", lambda p: 1 << 20)
    with pytest.raises(setup_env.Failure) as e:
        inst.check_disk([fetch.Wheel("x", "1", "x.whl", "u", "0" * 64, 1 << 30)])
    assert "disk space" in e.value.message


@pytest.fixture
def server(tmp_path):
    """A local server of one 300 kB file that honours Range; ``cut`` makes it stop short."""
    data = bytes(range(256)) * 1200
    state = {"cut": None, "served": []}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            start = 0
            if "Range" in self.headers:
                start = int(self.headers["Range"].split("=")[1].split("-")[0])
            state["served"].append(start)
            body = data[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state["cut"] is not None:
                n, state["cut"] = state["cut"], None
                self.wfile.write(body[:n])
                self.wfile.flush()
                self.connection.close()
                return
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield data, state, f"http://127.0.0.1:{httpd.server_address[1]}/f.whl"
    httpd.shutdown()


def wheel_of(data: bytes, url: str, filename: str = "f.whl") -> fetch.Wheel:
    return fetch.Wheel("f", "1", filename, url, hashlib.sha256(data).hexdigest(), len(data))


def test_download_complete(tmp_path, server):
    data, state, url = server
    fetch.download_one(wheel_of(data, url), tmp_path, lambda n: None, threading.Event(), ssl.create_default_context())
    assert (tmp_path / "f.whl").read_bytes() == data
    assert state["served"] == [0]


def test_download_resumes_after_a_cut(tmp_path, server):
    data, state, url = server
    state["cut"] = 1000
    w = wheel_of(data, url)
    done: list[int] = []
    with pytest.raises(fetch.DownloadError) as e:
        fetch.download_one(w, tmp_path, done.append, threading.Event(), ssl.create_default_context())
    assert e.value.kind == "network"
    assert (tmp_path / "f.whl.part").is_file()
    fetch.download_one(w, tmp_path, done.append, threading.Event(), ssl.create_default_context())
    assert (tmp_path / "f.whl").read_bytes() == data
    assert state["served"][-1] > 0                                  # continued where it stopped


def test_download_refuses_damaged_bytes(tmp_path, server):
    data, _state, url = server
    w = fetch.Wheel("f", "1", "f.whl", url, "0" * 64, len(data))
    with pytest.raises(fetch.DownloadError) as e:
        fetch.download_one(w, tmp_path, lambda n: None, threading.Event(), ssl.create_default_context())
    assert e.value.kind == "hash"
    assert not (tmp_path / "f.whl").exists() and not (tmp_path / "f.whl.part").exists()


def test_download_all_reports_progress(tmp_path, server):
    data, _state, url = server
    seen: list[fetch.Progress] = []
    fetch.download_all([wheel_of(data, url, "a.whl"), wheel_of(data, url, "b.whl")], tmp_path, seen.append, threading.Event(),
                       ssl.create_default_context(), attempts=1)
    assert (tmp_path / "a.whl").is_file() and (tmp_path / "b.whl").is_file()
    assert seen[-1].overall_done == 2 * len(data) and seen[-1].count == 2


def test_download_network_failure_kind(tmp_path, monkeypatch):
    def no_network(*a, **k):
        raise urllib.error.URLError("no network")
    monkeypatch.setattr(fetch, "_open", no_network)
    w = fetch.Wheel("x", "1", "x.whl", "http://example.invalid/x.whl", "0" * 64, 10)
    with pytest.raises(fetch.DownloadError) as e:
        fetch.download_one(w, tmp_path, lambda n: None, threading.Event(), ssl.create_default_context())
    assert e.value.kind == "network"


def test_cancel_stops_the_download(tmp_path, server):
    data, _state, url = server
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(fetch.Cancelled):
        fetch.download_one(wheel_of(data, url), tmp_path, lambda n: None, cancel, ssl.create_default_context())


# ------------------------------------------------------------------ the environment
def test_installed_versions(tmp_path):
    (tmp_path / "torch-2.11.0+cu128.dist-info").mkdir()
    (tmp_path / "pyside6_addons-6.12.0.dist-info").mkdir()
    have = setup_env.installed_versions(tmp_path)
    assert have == {"torch": "2.11.0+cu128", "pyside6_addons": "6.12.0"}


def test_missing_wheels_compares_versions(tmp_path):
    class Ui:
        pass
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"python": "3.12.13", "builds": {}}), encoding="utf-8")
    inst = setup_env.Installer(tmp_path, Ui(), lock, "en", {})
    inst.site.mkdir(parents=True)
    (inst.site / "torch-2.11.0+cpu.dist-info").mkdir()
    wheels = [fetch.Wheel("torch", "2.11.0+cu128", "t.whl", "u", "0" * 64, 1),
              fetch.Wheel("numba", "0.68.1", "n.whl", "u", "0" * 64, 1)]
    assert [w.name for w in inst.missing_wheels(wheels)] == ["torch", "numba"]  # another build of torch is a change


def test_sync_with_nothing_to_fetch_has_its_folder(tmp_path):
    """A re-install: nothing was downloaded, the folder uv reads must still exist."""
    class Ui:
        def stage(self, text): pass
        def indeterminate(self): pass
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"python": "3.12.13", "builds": {}}), encoding="utf-8")
    (tmp_path / "setup").mkdir()
    inst = setup_env.Installer(tmp_path, Ui(), lock, "en", {})
    seen = []
    inst.run_tool = lambda cmd, key: seen.append(inst.downloads.is_dir()) or ""
    inst.sync([fetch.Wheel("numba", "0.68.1", "n.whl", "u", "0" * 64, 1)])
    assert seen == [True]
    assert (inst.site / "float_app.pth").read_text(encoding="ascii").strip() == r"..\..\..\app"


def test_ready_marker(tmp_path):
    root = tmp_path
    (root / "setup").mkdir()
    (root / "env").mkdir()
    (root / "setup" / "lock.json").write_text("{}", encoding="utf-8")
    assert not setup_env.is_ready(root)
    mark = {"lock": fetch.sha256_of(root / "setup" / "lock.json")}
    (root / "env" / setup_env.MARKER).write_text(json.dumps(mark), encoding="utf-8")
    assert setup_env.is_ready(root)
    (root / "setup" / "lock.json").write_text('{"a":1}', encoding="utf-8")
    assert not setup_env.is_ready(root)


def asterion_install(tmp_path, monkeypatch, answer="Asterion Solver 0.4.2\n"):
    """An installation with the plate solver's folder (fake files at the lock's sizes, sparse) and a program that
    answers ``answer`` to --version."""
    class Ui:
        pass
    part = {"version": "0.4.2", "files": [
        {"filename": "asterion-0.4.2-windows-x86_64.zip", "dest": "", "size": 10},
        {"filename": "tycho2.astx", "dest": "catalogs", "size": 5000},
        {"filename": "tycho2.astx.wide", "dest": "catalogs", "size": 300}]}
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"python": "3.12.13", "builds": {}, "asterion": part}), encoding="utf-8")
    cat = tmp_path / "asterion" / "catalogs"
    cat.mkdir(parents=True)
    (tmp_path / "asterion" / "asterion.exe").write_bytes(b"MZ")
    (cat / "tycho2.astx").write_bytes(b"x" * 5000)
    (cat / "tycho2.astx.wide").write_bytes(b"x" * 300)
    asked = []

    def run(cmd, **kw):
        asked.append(cmd)
        return type("R", (), {"returncode": 0, "stdout": answer})()
    monkeypatch.setattr(setup_env.subprocess, "run", run)
    inst = setup_env.Installer(tmp_path, Ui(), lock, "en", {})
    return inst, asked


def test_asterion_check_passes(tmp_path, monkeypatch):
    inst, asked = asterion_install(tmp_path, monkeypatch)
    assert inst.check_asterion() == ""
    assert asked == [[str(tmp_path / "asterion" / "asterion.exe"), "--version"]]


@pytest.mark.parametrize("answer", ["Asterion Solver 0.4.1\n", "", "not a solver"])
def test_asterion_check_wrong_program(tmp_path, monkeypatch, answer):
    inst, _ = asterion_install(tmp_path, monkeypatch, answer)
    assert inst.check_asterion() == setup_env.TEXTS["en"]["asterion_program"]


def test_asterion_check_missing_program(tmp_path, monkeypatch):
    inst, _ = asterion_install(tmp_path, monkeypatch)

    def missing(cmd, **kw):
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(setup_env.subprocess, "run", missing)
    assert inst.check_asterion() == setup_env.TEXTS["en"]["asterion_program"]


@pytest.mark.parametrize("damage", ["missing", "short"])
def test_asterion_check_index(tmp_path, monkeypatch, damage):
    inst, _ = asterion_install(tmp_path, monkeypatch)
    index = tmp_path / "asterion" / "catalogs" / "tycho2.astx"
    index.unlink() if damage == "missing" else index.write_bytes(b"x" * 10)
    assert inst.check_asterion() == setup_env.TEXTS["en"]["asterion_index"]


def test_asterion_problem_is_said_and_installation_finishes(tmp_path, monkeypatch):
    """A broken plate solver is said in the closing summary; the installation is still ready (FLOAT starts)."""
    inst, _ = asterion_install(tmp_path, monkeypatch, "")
    (tmp_path / "setup").mkdir()
    inst.env_dir.mkdir()
    inst.choice = detect.Choice("cpu", "no-card", "", "")
    inst.run_tool = lambda cmd, key: ""
    inst.finish("cpu", {}, False)
    assert inst.summary.endswith(setup_env.TEXTS["en"]["asterion_program"])
    assert (inst.env_dir / setup_env.MARKER).is_file()


def test_asterion_check_without_a_plate_solver_in_the_lock(tmp_path):
    class Ui:
        pass
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"python": "3.12.13", "builds": {}}), encoding="utf-8")
    assert setup_env.Installer(tmp_path, Ui(), lock, "en", {}).check_asterion() == ""


# ------------------------------------------------------------------ the texts
def test_two_languages_have_the_same_texts():
    assert set(setup_env.TEXTS["en"]) == set(setup_env.TEXTS["fr"])


@pytest.mark.parametrize("lang", ["fr", "en"])
@pytest.mark.parametrize("why", ["ok", "no-card", "old-driver", "no-driver", "other-vendor", "error"])
def test_card_messages(lang, why):
    text = setup_env.card_message(detect.Choice("cpu", why, "RTX 3060", "400.00"), lang)
    assert text and "{" not in text
    if why == "old-driver":
        assert "nvidia.com/drivers" in text


def test_human_sizes():
    assert setup_env.human(2.5e9, "fr") == "2,5 Go"
    assert setup_env.human(2.5e9, "en") == "2.5 GB"
    assert setup_env.human(35e6, "en") == "35 MB"
