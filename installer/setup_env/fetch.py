"""Downloading the wheels of the lock file: exact bytes (SHA-256), resumed after a cut, with a progress report.

Standard library only. ``needed_bytes`` is the disk space check made before any download. The files land in one
folder, a ``.part`` file holds what a cut interrupted, the next attempt continues it (HTTP Range).
"""
from __future__ import annotations

import errno
import hashlib
import http.client
import os
import shutil
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CHUNK = 1 << 20
#: disk room per byte of wheel at the peak: the wheel (1), what it unpacks to in uv's cache (1.8 measured on the CUDA
#: build) and, where hard links do not exist (exFAT), a second full copy for the installed files
DISK_FACTOR_LINKED = 2.9
DISK_FACTOR_COPIED = 4.7
DISK_MARGIN = 800 << 20


@dataclass(frozen=True)
class Wheel:
    name: str
    version: str
    filename: str
    url: str
    sha256: str
    size: int
    bundled: bool = False       # in the installer itself (setup/wheels), never downloaded


@dataclass
class Progress:
    wheel: Wheel
    index: int                  # 1-based, among the files to download
    count: int
    file_done: int
    overall_done: int
    overall_total: int
    speed: float                # bytes per second, smoothed


class DownloadError(Exception):
    """kind: network | disk | hash. Retrying is the remedy for network and hash."""

    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


class Cancelled(Exception):
    pass


def free_bytes(path: str | os.PathLike) -> int:
    """Free space on the volume of ``path``, even when the folder does not exist yet."""
    p = Path(path).absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def hard_links_work(folder: str | os.PathLike) -> bool:
    """Whether this volume has hard links: uv then installs without a second copy of everything."""
    probe = Path(folder) / ".link_probe"
    link = Path(folder) / ".link_probe2"
    try:
        probe.write_bytes(b"x")
        os.link(probe, link)
        return True
    except OSError:
        return False
    finally:
        for f in (link, probe):
            try:
                f.unlink()
            except OSError:
                pass


def needed_bytes(missing: list[Wheel], linked: bool = True) -> int:
    """The room these wheels need at the peak: downloaded, unpacked and installed. Nothing to install: none."""
    total = sum(w.size for w in missing)
    return int(total * (DISK_FACTOR_LINKED if linked else DISK_FACTOR_COPIED)) + DISK_MARGIN if total else 0


def ssl_context(cafile: str | os.PathLike | None) -> ssl.SSLContext:
    """TLS with the bundled root certificates (a fresh Windows fetches its own roots lazily, not for Python) and the
    system's (a company's proxy certificate)."""
    ctx = ssl.create_default_context(cafile=str(cafile) if cafile and Path(cafile).is_file() else None)
    try:
        ctx.load_default_certs()
    except (ssl.SSLError, OSError):
        pass
    return ctx


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _is_network(e: OSError) -> bool:
    return isinstance(e, (urllib.error.URLError, socket.timeout, ConnectionError, ssl.SSLError))


def _open(url: str, start: int, context: ssl.SSLContext, timeout: float):
    headers = {"User-Agent": "FLOAT-installer", "Accept-Encoding": "identity"}
    if start:
        headers["Range"] = f"bytes={start}-"
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    return opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)


def download_one(w: Wheel, folder: Path, on_bytes, cancel: threading.Event, context: ssl.SSLContext,
                 timeout: float = 30.0) -> Path:
    """Fetch one wheel into ``folder``, continuing a ``.part``; its bytes are checked against the lock. ``on_bytes(n)``
    is called with each number of new bytes, negative when a bad file is started over."""
    final = folder / w.filename
    part = folder / (w.filename + ".part")
    if final.is_file():
        if final.stat().st_size == w.size and sha256_of(final) == w.sha256:
            on_bytes(w.size)
            return final
        final.unlink()
    have = part.stat().st_size if part.is_file() else 0
    if have > w.size:
        part.unlink()
        have = 0
    on_bytes(have)
    try:
        with _open(w.url, have, context, timeout) as r:
            if have and getattr(r, "status", 200) != 206:       # the server ignored Range: from the start
                on_bytes(-have)
                have = 0
            with open(part, "ab" if have else "wb") as f:
                while True:
                    if cancel.is_set():
                        raise Cancelled
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    on_bytes(len(chunk))
    except OSError as e:                                         # a full disk is not a network problem
        if e.errno == errno.ENOSPC:
            raise DownloadError("disk", str(e)) from e
        if _is_network(e):
            raise DownloadError("network", str(e)) from e
        raise
    except http.client.HTTPException as e:
        raise DownloadError("network", str(e)) from e
    size = part.stat().st_size
    if size != w.size:
        raise DownloadError("network", f"{w.filename}: incomplete, {size} of {w.size} bytes")
    if sha256_of(part) != w.sha256:
        part.unlink()
        on_bytes(-size)
        raise DownloadError("hash", f"{w.filename}: wrong checksum")
    part.replace(final)
    return final


def download_all(items: list[Wheel], folder: Path, on_progress, cancel: threading.Event, context: ssl.SSLContext,
                 attempts: int = 3, pause: float = 2.0) -> None:
    """Fetch every wheel, largest counted in the total; a passing cut is retried on the spot (``attempts``), a lasting
    one raises DownloadError for the user's Retry; what was fetched stays."""
    folder.mkdir(parents=True, exist_ok=True)
    total = sum(w.size for w in items)
    done_before = 0
    speed = 0.0
    for i, w in enumerate(items, 1):
        file_done = 0
        last_t, last_b, accum = time.monotonic(), 0, 0
        def on_bytes(n: int):
            nonlocal file_done, last_t, last_b, accum, speed
            file_done += n
            accum += max(n, 0)
            now = time.monotonic()
            if now - last_t >= 0.5:
                inst = (accum - last_b) / (now - last_t)
                speed = inst if speed == 0 else 0.7 * speed + 0.3 * inst
                last_t, last_b = now, accum
            on_progress(Progress(w, i, len(items), file_done, done_before + file_done, total, speed))

        on_progress(Progress(w, i, len(items), 0, done_before, total, speed))
        for attempt in range(attempts):
            try:
                file_done = 0
                last_b = accum = 0
                last_t = time.monotonic()
                download_one(w, folder, on_bytes, cancel, context)
                break
            except DownloadError as e:
                if e.kind == "disk" or attempt == attempts - 1:
                    raise
                if cancel.wait(pause * (attempt + 1)):
                    raise Cancelled from e
        done_before += w.size
