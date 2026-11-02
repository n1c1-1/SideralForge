"""The offline Gaia extracts (core/gaia_extracts.py): the download from a local fake Zenodo (file list API, ranges,
md5), the places the engine looks in (core/sirilcat.py), and the engine without any extract.

No network: a small HTTP server in a thread answers like Zenodo does (the record's JSON, and the files with Range).
"""
from __future__ import annotations

import bz2
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import requests

from astro_gpu.core import gaia_extracts as ge
from astro_gpu.core import online, sirilcat

ASTRO_NAME = "siril_cat_healpix8_astro.dat"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """A private ASTRO_GPU_HOME and user home, no other program's data folder, no extract anywhere, small chunks (a
    small file spans several), the network allowed (it is only the local server)."""
    home = tmp_path / "userhome"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "aghome"))
    (tmp_path / "nowhere").mkdir()
    for v in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(v, str(tmp_path / "nowhere"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    monkeypatch.delenv("ASTRO_GPU_OFFLINE", raising=False)
    monkeypatch.setattr(ge, "CHUNK", 4096)
    monkeypatch.setattr(ge, "REPORT_EVERY_S", 0.0)
    monkeypatch.setattr(ge, "MAX_RETRIES", 1)
    online.reset()
    sirilcat.reset()
    yield
    online.reset()
    sirilcat.reset()


# ------------------------------------------------------------------ fake extract files

def _header(level: int, kind: int, chunk: int | None) -> bytearray:
    h = bytearray(128)
    h[:48] = b"Siril test extract".ljust(48, b"\0")
    h[48:53] = bytes([3, level, kind, 1 if chunk is not None else 0, level if chunk is not None else 0])
    if chunk is not None:
        for off in (53, 57, 61):
            h[off:off + 4] = int(chunk).to_bytes(4, "little")
    return h


def _astro_file(n: int = 3000, seed: int = 1) -> bytes:
    """A valid astrometric extract of HEALPix level 1 (48 pixels) with n random stars, all in pixel 5 (random bytes: it
    does not compress, so that the compressed file spans several chunks)."""
    rng = np.random.default_rng(seed)
    rec = np.zeros(n, sirilcat.ASTRO_DT)
    for name, dt in sirilcat.ASTRO_DT.fields.items():
        rec[name] = rng.integers(-1000, 1000, n).astype(dt[0])
    idx = np.where(np.arange(48) >= 5, n, 0).astype("<u4")
    return bytes(_header(1, sirilcat.TYPE_ASTRO, None)) + idx.tobytes() + rec.tobytes()


def _xp_file(chunk: int, n: int = 2, seed: int = 2) -> bytes:
    """A valid spectra extract holding one level-1 pixel (the chunk) and n stars."""
    rng = np.random.default_rng(seed + chunk)
    rec = np.zeros(n, sirilcat.XP_DT)
    rec["ra"] = rng.integers(0, 1000, n)
    rec["dec"] = rng.integers(0, 1000, n)
    rec["g"] = 12000 + np.arange(n)
    rec["flux"] = rng.uniform(0.5, 2.0, (n, 343)).astype(np.float16)
    return bytes(_header(1, sirilcat.TYPE_XPSAMP, chunk)) + np.array([n], "<u4").tobytes() + rec.tobytes()


def _bz2_two_streams(data: bytes) -> bytes:
    half = len(data) // 2
    return bz2.compress(data[:half]) + bz2.compress(data[half:])


# ------------------------------------------------------------------ a fake Zenodo

class FakeZenodo:
    """The records' JSON and the files, with Range. Options: ``ignore_range`` (always answers 200 with the whole file),
    ``corrupt`` (names whose body is damaged while the listed md5 is the good one), ``cut`` (the first answer of a
    file stops after this many bytes), ``fail_api``."""

    def __init__(self, records: dict[int, dict[str, bytes]]):
        self.records, self.requests = records, []
        self.ignore_range, self.corrupt, self.cut, self.fail_api = False, set(), None, False
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Range")))
                if self.path.startswith("/api/records/"):
                    return self.api(int(self.path.rsplit("/", 1)[1]))
                name = self.path.split("/")[2]                 # /files/<name>/content
                body = next((r[name] for r in outer.records.values() if name in r), None)
                if body is None:
                    return self.reply(404, b"")
                if name in outer.corrupt:
                    body = bytes([body[0] ^ 0xFF]) + body[1:]
                rng = self.headers.get("Range")
                start = int(rng.split("=")[1].split("-")[0]) if rng and not outer.ignore_range else 0
                if start >= len(body):
                    return self.reply(416, b"")
                part = body[start:]
                cut = outer.cut
                if cut is not None:
                    outer.cut = None
                    self.send_response(200 if start == 0 else 206)
                    self.send_header("Content-Length", str(len(part)))
                    if start:
                        self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                    self.end_headers()
                    self.wfile.write(part[:cut])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                if start:
                    return self.reply(206, part, {"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"})
                self.reply(200, part)

            def api(self, record):
                if outer.fail_api:
                    return self.reply(503, b"")
                files = [{"key": n, "size": len(b), "checksum": "md5:" + hashlib.md5(b).hexdigest(),
                          "links": {"self": f"{outer.base}/files/{n}/content"}} for n, b in outer.records[record].items()]
                files.append({"key": "something.sha256sum", "size": 5, "checksum": "md5:00", "links": {"self": "x"}})
                self.reply(200, json.dumps({"id": record, "files": files}).encode(), {"Content-Type": "application/json"})

            def reply(self, code, body, headers=None):
                try:
                    self.send_response(code)
                    for k, v in (headers or {}).items():
                        self.send_header(k, v)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.api_url = self.base + "/api/records/{record}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def file_requests(self):
        return [r for r in self.requests if "/files/" in r[0]]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class _NoNetwork:
    def get(self, *a, **k):
        raise requests.ConnectionError("Failed to resolve 'zenodo.org' (getaddrinfo failed)")


def _session():
    s = requests.Session()
    s.trust_env = False                              # no system proxy for 127.0.0.1
    return s


@pytest.fixture()
def zen():
    made = []

    def make(records):
        z = FakeZenodo(records)
        made.append(z)
        return z
    yield make
    for z in made:
        z.close()


def _astro_server(zen, **kw):
    raw = _astro_file(**kw)
    z = zen({ge.RECORDS[ge.ASTRO]: {ASTRO_NAME + ".bz2": _bz2_two_streams(raw)}})
    return z, raw


def _run(kind, z, cancel=None, progress=None):
    ge.download(kind, progress, cancel, session=_session(), api=z.api_url)


# ------------------------------------------------------------------ the download

def test_download_astrometric_extract_end_to_end(zen):
    z, raw = _astro_server(zen)
    assert sirilcat.locate()["astro"] is None and sirilcat.astro_cone(0.0, 0.0, 5.0, 30.0) is None
    assert ge.status(ge.ASTRO).state == "missing"
    seen = []
    _run(ge.ASTRO, z, progress=lambda d, t, text: seen.append((d, t, text)))
    d = ge.folder()
    assert d == sirilcat.own_dir() and d.parent.name == "cache"
    assert (d / ASTRO_NAME).read_bytes() == raw                       # two bz2 streams unpacked, whole
    assert sorted(p.name for p in d.iterdir()) == [ASTRO_NAME]          # no .bz2, no .part left
    assert not [r for r in z.requests if r[1]]                          # first run: no Range
    done = [s[0] for s in seen]
    assert done == sorted(done) and seen[-1][0] == seen[-1][1] and seen[-1][2] == "Done"
    assert {"downloading", "checking", "unpacking"} <= {w for s in seen for w in ("downloading", "checking", "unpacking")
                                                        if w in s[2]}
    st = ge.status(ge.ASTRO)                                           # the download reset the engine's lookup
    assert (st.state, st.files, st.own, st.outside) == ("installed", 1, True, None)
    assert sirilcat.locate()["astro"].path == d / ASTRO_NAME
    assert sirilcat.astro_cone(0.0, 0.0, 5.0, 30.0) is not None        # the extract answers
    # asked again: nothing is downloaded
    n = len(z.file_requests())
    _run(ge.ASTRO, z)
    assert len(z.file_requests()) == n


def test_download_spectra_extract_goes_to_its_sub_folder_and_counts_as_partial(zen):
    files = {f"siril_cat1_healpix8_xpsamp_{n}.dat.bz2": bz2.compress(_xp_file(n)) for n in (2, 0, 10)}
    z = zen({ge.RECORDS[ge.XP]: files})
    _run(ge.XP, z)
    sub = ge.folder() / "siril_cat1_healpix8_xpsamp"
    assert sorted(p.name for p in sub.iterdir()) == [f"siril_cat1_healpix8_xpsamp_{n}.dat" for n in (0, 10, 2)]
    assert sorted(sirilcat.locate()["xp"]) == [0, 2, 10]
    # the files were fetched in numeric order
    assert [p.rsplit("/", 2)[1] for p, _r in z.file_requests()] == [f"siril_cat1_healpix8_xpsamp_{n}.dat.bz2"
                                                                     for n in (0, 2, 10)]
    st = ge.status(ge.XP)
    assert (st.state, st.files, st.total) == ("partial", 3, 48)
    rows = sirilcat.xp_spectra(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert rows == [None]


def test_download_resumes_a_partial_file_with_a_range_request(zen):
    z, raw = _astro_server(zen)
    bz = z.records[ge.RECORDS[ge.ASTRO]][ASTRO_NAME + ".bz2"]
    d = ge.folder()
    d.mkdir(parents=True)
    (d / (ASTRO_NAME + ".bz2.part")).write_bytes(bz[:5000])
    _run(ge.ASTRO, z)
    assert z.file_requests()[0][1] == "bytes=5000-"
    assert (d / ASTRO_NAME).read_bytes() == raw
    assert sorted(p.name for p in d.iterdir()) == [ASTRO_NAME]


def test_a_200_answer_to_a_range_request_starts_again_from_zero(zen):
    z, raw = _astro_server(zen)
    z.ignore_range = True
    bz = z.records[ge.RECORDS[ge.ASTRO]][ASTRO_NAME + ".bz2"]
    d = ge.folder()
    d.mkdir(parents=True)
    (d / (ASTRO_NAME + ".bz2.part")).write_bytes(bz[:5000])
    _run(ge.ASTRO, z)
    assert z.file_requests()[0][1] == "bytes=5000-"                      # asked, and answered with the whole file
    assert (d / ASTRO_NAME).read_bytes() == raw                          # not 5000 bytes twice


def test_a_part_longer_than_the_file_is_dropped(zen):
    z, raw = _astro_server(zen)
    d = ge.folder()
    d.mkdir(parents=True)
    (d / (ASTRO_NAME + ".bz2.part")).write_bytes(b"x" * 10_000_000)
    _run(ge.ASTRO, z)
    assert z.file_requests()[0][1] is None and (d / ASTRO_NAME).read_bytes() == raw


def test_a_connection_cut_in_the_middle_goes_on_from_there(zen, monkeypatch):
    monkeypatch.setattr(ge.time, "sleep", lambda s: None)
    z, raw = _astro_server(zen)
    z.cut = 4000
    _run(ge.ASTRO, z)
    reqs = z.file_requests()
    assert len(reqs) == 2 and reqs[0][1] is None                         # asked again, from what had arrived
    assert reqs[1][1] is None or int(reqs[1][1].split("=")[1].rstrip("-")) <= 4000
    assert (ge.folder() / ASTRO_NAME).read_bytes() == raw


def test_md5_mismatch_deletes_the_file_and_says_so(zen):
    z, _raw = _astro_server(zen)
    z.corrupt = {ASTRO_NAME + ".bz2"}
    with pytest.raises(ge.DownloadError, match="damaged during the download.*checksum"):
        _run(ge.ASTRO, z)
    assert [p.name for p in ge.folder().iterdir()] == []
    assert sirilcat.locate()["astro"] is None


def test_cancel_keeps_the_part_and_the_next_run_goes_on(zen):
    z, raw = _astro_server(zen)
    cancel = threading.Event()

    def stop_after_a_while(done, total, text):
        if "downloading" in text:
            cancel.set()
    with pytest.raises(ge.Cancelled):
        _run(ge.ASTRO, z, cancel=cancel, progress=stop_after_a_while)
    d = ge.folder()
    part = d / (ASTRO_NAME + ".bz2.part")
    have = part.stat().st_size
    assert 0 < have < len(z.records[ge.RECORDS[ge.ASTRO]][ASTRO_NAME + ".bz2"]) and not (d / ASTRO_NAME).exists()
    assert ge.status(ge.ASTRO).state == "partial" and ge.status(ge.ASTRO).part_bytes == have
    _run(ge.ASTRO, z)                                                  # like after a restart of the program
    assert z.file_requests()[-1][1] == f"bytes={have}-"
    assert (d / ASTRO_NAME).read_bytes() == raw and ge.status(ge.ASTRO).state == "installed"


def test_a_verified_bz2_left_by_a_stopped_run_is_only_unpacked(zen):
    z, raw = _astro_server(zen)
    d = ge.folder()
    d.mkdir(parents=True)
    (d / (ASTRO_NAME + ".bz2")).write_bytes(z.records[ge.RECORDS[ge.ASTRO]][ASTRO_NAME + ".bz2"])
    (d / (ASTRO_NAME + ".part")).write_bytes(b"half of an unpacking")
    _run(ge.ASTRO, z)
    assert z.file_requests() == [] and (d / ASTRO_NAME).read_bytes() == raw
    assert sorted(p.name for p in d.iterdir()) == [ASTRO_NAME]


def test_a_file_that_is_not_an_extract_is_refused(zen):
    junk = b"not an extract" * 100
    z = zen({ge.RECORDS[ge.ASTRO]: {ASTRO_NAME + ".bz2": bz2.compress(junk)}})
    with pytest.raises(ge.DownloadError, match="not an extract"):
        _run(ge.ASTRO, z)
    assert [p.name for p in ge.folder().iterdir()] == []
    z2 = zen({ge.RECORDS[ge.ASTRO]: {ASTRO_NAME + ".bz2": b"BZh9 this is no bz2 stream at all"}})
    with pytest.raises(ge.DownloadError, match="damaged and cannot be unpacked"):
        _run(ge.ASTRO, z2)
    assert [p.name for p in ge.folder().iterdir()] == []


def test_plain_errors_for_the_list_offline_and_a_second_download(zen, monkeypatch):
    z, _raw = _astro_server(zen)
    z.fail_api = True
    with pytest.raises(ge.DownloadError, match="Cannot read the file list.*HTTP 503"):
        _run(ge.ASTRO, z)
    z.fail_api = False
    with pytest.raises(ge.DownloadError, match="Cannot read the file list"):      # nobody listening
        ge.download(ge.ASTRO, session=_NoNetwork(), api=z.api_url)
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")
    with pytest.raises(ge.DownloadError, match="offline mode"):
        _run(ge.ASTRO, z)
    monkeypatch.delenv("ASTRO_GPU_OFFLINE")
    assert ge._RUN.acquire(blocking=False)
    try:
        with pytest.raises(ge.DownloadError, match="already running"):
            _run(ge.ASTRO, z)
    finally:
        ge._RUN.release()
    assert not ge.folder().exists() or not list(ge.folder().iterdir())


def test_not_enough_disk_space(zen, monkeypatch):
    z, _raw = _astro_server(zen)
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda p: type("U", (), {"free": 1000})())
    with pytest.raises(ge.DownloadError, match="Not enough free disk space"):
        _run(ge.ASTRO, z)


def test_texts_go_through_the_language_hooks(zen, monkeypatch):
    z, _raw = _astro_server(zen)
    z.fail_api = True
    monkeypatch.setattr(ge, "translate", lambda s: "FR " + s)
    with pytest.raises(ge.DownloadError, match="^FR Cannot read"):
        _run(ge.ASTRO, z)
    monkeypatch.setattr(ge, "translate", lambda s: s)
    monkeypatch.setattr(ge, "number", lambda x, d=1: f"{x:.{d}f}".replace(".", ","))
    assert ge.human(1.5e9) == "1,5 GB" and ge.human(2.5e8) == "250 MB"


# ------------------------------------------------------------------ where the engine looks

def test_our_folder_is_searched_and_the_environment_comes_first(tmp_path, monkeypatch):
    own = sirilcat.own_dir()
    assert own == tmp_path / "aghome" / "cache" / "gaia_extracts"
    (own / "siril_cat1_healpix8_xpsamp").mkdir(parents=True)
    (own / ASTRO_NAME).write_bytes(_astro_file(n=10))
    (own / "siril_cat1_healpix8_xpsamp" / "siril_cat1_healpix8_xpsamp_7.dat").write_bytes(_xp_file(7))
    (own / "siril_cat1_healpix8_xpsamp" / "siril_cat1_healpix8_xpsamp_8.dat.part").write_bytes(_xp_file(8)[:300])
    found = sirilcat.locate()
    assert found["astro"].path == own / ASTRO_NAME and list(found["xp"]) == [7]      # a .part is not an extract
    # a folder of the environment wins over ours (the new name, and the older one still accepted)
    other = tmp_path / "other"
    other.mkdir()
    (other / ASTRO_NAME).write_bytes(_astro_file(n=20, seed=5))
    for var in ("ASTRO_GPU_GAIA_EXTRACTS", "ASTRO_GPU_SIRIL_CATALOGUES"):
        monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
        monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
        monkeypatch.setenv(var, str(other))
        sirilcat.reset()
        assert sirilcat.locate()["astro"].path == other / ASTRO_NAME


def test_another_programs_copy_is_an_extra_place_only(tmp_path, monkeypatch):
    share = tmp_path / "nowhere" / "siril"                  # where that program keeps its data (LOCALAPPDATA/siril)
    share.mkdir(parents=True)
    (share / ASTRO_NAME).write_bytes(_astro_file(n=10))
    found = sirilcat.locate()
    assert found["astro"].path == share / ASTRO_NAME
    st = ge.status(ge.ASTRO)
    assert st.state == "installed" and not st.own and st.outside == share
    # ours, once downloaded or copied there, is preferred
    (sirilcat.own_dir()).mkdir(parents=True)
    (sirilcat.own_dir() / ASTRO_NAME).write_bytes(_astro_file(n=10))
    sirilcat.reset()
    assert sirilcat.locate()["astro"].path == sirilcat.own_dir() / ASTRO_NAME and ge.status(ge.ASTRO).own


def test_no_extract_anywhere_the_engine_answers_none(tmp_path):
    assert sirilcat.locate() == {"astro": None, "xp": {}}
    assert sirilcat.astro_cone(315.3, 68.17, 1.0, 14.0) is None
    assert sirilcat.xp_spectra(np.array([315.3]), np.array([68.17]), np.array([10.0])) == [None]
    assert sirilcat.describe() == "none"
    assert not sirilcat.own_dir().exists()                       # looking never creates the folder
    s = ge.status(ge.XP)
    assert (s.state, s.files, s.total, s.part_bytes, s.outside) == ("missing", 0, 48, 0, None)


def test_constants_match_the_published_records():
    assert ge.RECORDS == {"astro": 14692304, "xp": 14738271} and ge.LICENCE == "CC BY 4.0"
    assert ge.VERSIONS == {"astro": "1.0.0", "xp": "1.0.1"} and ge.FILES == {"astro": 1, "xp": 48}
    assert ge.DOWNLOAD_BYTES["astro"] == 1_142_276_749 and 10.5e9 < ge.DOWNLOAD_BYTES["xp"] < 10.6e9
    assert ge.ZENODO_API.format(record=1) == "https://zenodo.org/api/records/1"
