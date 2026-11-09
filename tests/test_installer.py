"""The Windows installer's parts that can be tested without Windows being clean: which card, which PyTorch build, what
the lock file holds, the disk check, the download (cut, resumed, damaged). The real installation is tested by hand with
the .exe (see installer/build.py)."""
from __future__ import annotations

import hashlib
import http.server
import json
import ssl
import sys
import threading
import tomllib
import urllib.error
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
