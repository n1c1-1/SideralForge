"""Builds the FLOAT Windows installer: one command.

    python installer/build.py            # staging + compilation, the .exe in F:/astro_essais/A_REGARDER/installeur
    python installer/build.py lock       # regenerates installer/lock/lock.json (then build)
    python installer/build.py --out D:/x
    python installer/build.py asterion --stage D:/x   # stages the plate solver's folder alone (a check, no .exe)

The lock file: the versions of the environment the tests pass on (``uv pip freeze`` of LOCK_SOURCE), restricted to
pyproject's dependencies and what they need, for two builds of PyTorch (CUDA and CPU), with each wheel's URL, size and
SHA-256; and the Asterion plate solver (its released archive and its index), pinned the same way, which ``lock`` keeps
as it is. Staging gathers what the .exe carries: a private Python (standalone, with Tk), uv, the Visual C++ runtime next
to python.exe (PyTorch needs it and a clean Windows lacks it), the app's tracked files plus the Stars network (not in
git), the plate solver and its index in ``asterion`` (the layout astro_gpu/core/asterion.py looks for beside ``app``),
the setup scripts, an icon. Inno Setup compiles float.iss around it.

The plate solver's files come from a local folder (``--asterion-cache``, or FLOAT_ASTERION_CACHE), each found by its
file name there or one folder down (the archive's folder and the catalogs' folder of a solver workspace both do), and
checked against the lock: nothing is downloaded, a file missing or different stops the build.

Run with a Python that has Pillow and certifi (the ComfyUI environment here). Inno Setup and uv must be installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "setup_env"))

import detect  # noqa: E402

LOCK_FILE = HERE / "lock" / "lock.json"
LOCK_SOURCE = Path(r"C:\Users\Nicolas\sandbox\ComfyUI\.venv\Scripts\python.exe")
PLATFORM = "x86_64-pc-windows-msvc"
BUILDS = {"cu128": "cu128", "cpu": "cpu"}
OUT_DEFAULT = Path("F:/astro_essais/A_REGARDER/installeur")
STAR_MODEL = Path("astro_gpu/resources/stars/starless_nafnet.pt")
VC_RUNTIME = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_codecvt_ids.dll", "vcruntime140.dll",
              "vcruntime140_1.dll", "vcomp140.dll", "concrt140.dll")
ASTERION_CACHE = Path(os.environ.get("FLOAT_ASTERION_CACHE") or REPO / "build" / "asterion-cache")
#: what the installation keeps of the released archive: the program, its licence and the third-party notices (not the
#: web interface, nor the README of a command line FLOAT runs itself)
ASTERION_KEEP = ("asterion.exe", "LICENSE.txt", "third-party")


def run(*cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def uv() -> str:
    path = shutil.which("uv")
    if not path:
        raise SystemExit("uv is not installed")
    return path


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def project_dependencies() -> list[str]:
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["dependencies"]


def requirement_name(req: str) -> str:
    return norm(re.split(r"[<>=!~;\[\s]", req.strip(), maxsplit=1)[0])


def freeze() -> dict[str, str]:
    out = subprocess.run([uv(), "pip", "freeze", "--python", str(LOCK_SOURCE)], capture_output=True, text=True,
                         check=True).stdout
    pins = {}
    for line in out.splitlines():
        if "==" in line and not line.startswith("-e"):
            name, _, version = line.partition("==")
            pins[norm(name)] = version.strip()
    return pins


def python_version() -> str:
    return subprocess.run([str(LOCK_SOURCE), "-c", "import sys;print(sys.version.split()[0])"], capture_output=True,
                          text=True, check=True).stdout.strip()


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "FLOAT-build"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def bundled_wheel(package: dict, tmp: Path) -> dict:
    """A package published as source only (pure Python): a wheel built once, kept in installer/lock/wheels and shipped
    inside the .exe, so that the user's computer never builds anything."""
    name, version = package["name"], package["version"]
    folder = LOCK_FILE.parent / "wheels"
    found = sorted(folder.glob(f"{name.replace('-', '_')}-{version}-*.whl")) if folder.is_dir() else []
    if not found:
        folder.mkdir(parents=True, exist_ok=True)
        sdist = tmp / package["sdist"]["url"].rsplit("/", 1)[1]
        urllib.request.urlretrieve(package["sdist"]["url"], sdist)
        if hashlib.sha256(sdist.read_bytes()).hexdigest() != package["sdist"]["hashes"]["sha256"]:
            raise SystemExit(f"{name}: wrong checksum for the source archive")
        run(uv(), "build", sdist, "--wheel", "-o", folder)
        (folder / ".gitignore").unlink(missing_ok=True)      # uv's: would hide the wheel from git
        found = sorted(folder.glob(f"{name.replace('-', '_')}-{version}-*.whl"))
    wheel = found[-1]
    if not wheel.name.endswith("-py3-none-any.whl"):
        raise SystemExit(f"{wheel.name}: only a pure-Python wheel can be built here")
    return {"name": name, "version": version, "filename": wheel.name, "url": "",
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "size": wheel.stat().st_size, "bundled": True}


def make_lock() -> dict:
    """lock.json from the tested environment: each build's closure of pyproject's dependencies, at the frozen versions."""
    pins = freeze()
    deps = project_dependencies()
    lock: dict = {"python": python_version(), "builds": {}}
    with tempfile.TemporaryDirectory(dir=REPO / "build" if (REPO / "build").is_dir() else None) as td:
        tmp = Path(td)
        (tmp / "req.in").write_text("\n".join(deps) + "\n", encoding="utf-8")
        # PyTorch's local version tag is dropped: the backend flag below picks the build
        (tmp / "c.txt").write_text("".join(f"{n}=={v.split('+')[0]}\n" for n, v in pins.items()), encoding="utf-8")
        for build, backend in BUILDS.items():
            out = tmp / f"pylock.{build}.toml"
            run(uv(), "pip", "compile", tmp / "req.in", "-c", tmp / "c.txt", "--python-version",
                lock["python"].rsplit(".", 1)[0], "--python-platform", PLATFORM, "--no-sources", "--torch-backend", backend,
                "--format", "pylock.toml", "-q", "-o", out)
            with open(out, "rb") as f:
                pkgs = tomllib.load(f)["packages"]
            wheels = []
            for p in pkgs:
                base = p["version"].split("+")[0]
                if norm(p["name"]) in pins and pins[norm(p["name"])].split("+")[0] != base:
                    raise SystemExit(f"{p['name']}: {p['version']} is not the tested {pins[norm(p['name'])]}")
                if not p.get("wheels"):
                    wheels.append(bundled_wheel(p, tmp))
                    continue
                w = p["wheels"][0]
                url = w["url"]
                wheels.append({"name": p["name"], "version": p["version"],
                               "filename": urllib.parse.unquote(url.rsplit("/", 1)[1]), "url": url,
                               "sha256": w["hashes"]["sha256"], "size": w.get("size") or head_size(url)})
                if norm(p["name"]) not in pins:
                    print("not in the tested environment:", p["name"], p["version"])
            lock["builds"][build] = wheels
    # the plate solver is pinned by hand (a release of its own, not a Python package): kept through a new lock
    if LOCK_FILE.is_file():
        former = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        if "asterion" in former:
            lock["asterion"] = former["asterion"]
    LOCK_FILE.parent.mkdir(exist_ok=True)
    LOCK_FILE.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    for build, wheels in lock["builds"].items():
        print(f"{build}: {len(wheels)} packages, {sum(w['size'] for w in wheels) / 1e9:.2f} GB")
    return lock


def check_lock() -> None:
    """Every dependency of pyproject is locked in both builds, with its PyTorch build (a stale lock stops the build)."""
    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    for build, wheels in lock["builds"].items():
        names = {norm(w["name"]) for w in wheels}
        missing = [d for d in project_dependencies() if requirement_name(d) not in names]
        if missing:
            raise SystemExit(f"lock file, build {build}: missing {missing}; run: python installer/build.py lock")


def find_star_model() -> Path:
    """The Stars network: not in git, it sits in the main checkout (the first worktree listed)."""
    local = REPO / STAR_MODEL
    if local.is_file():
        return local
    out = subprocess.run(["git", "worktree", "list", "--porcelain"], capture_output=True, text=True, cwd=REPO,
                         check=True).stdout
    for line in out.splitlines():
        if line.startswith("worktree "):
            cand = Path(line[len("worktree "):]) / STAR_MODEL
            if cand.is_file():
                return cand
    raise SystemExit(f"the Stars network is missing: {STAR_MODEL}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def pinned_file(entry: dict, cache: Path) -> Path:
    """A file of the lock's ``asterion`` part from the local cache, by its name there or one folder down, with the
    lock's size and checksum (the size first: a wrong file is said without reading 378 MB)."""
    name = entry["filename"]
    found = [cache / name] + (sorted(cache.glob(f"*/{name}")) if cache.is_dir() else [])
    path = next((p for p in found if p.is_file()), None)
    where = entry.get("url") or entry.get("origin") or "?"
    if path is None:
        raise SystemExit(f"{name} is missing from {cache} (--asterion-cache): put the pinned file there ({where})")
    if path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
        raise SystemExit(f"{path} is not the file the lock pins (size or checksum differs): get {name} again ({where})")
    return path


def link_or_copy(src: Path, dest: Path) -> None:
    """A hard link when both are on one drive (the index is 378 MB: never twice on the build disk), else a copy; the
    dates are kept either way: the index and its wide-field file stay as Asterion wrote them, one beside the other."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def stage_asterion(stage_dir: Path, lock: dict, cache: Path = ASTERION_CACHE, ask_version: bool = True) -> Path:
    """The plate solver's folder beside ``app``: the program, its licence and third-party notices unzipped from the
    released archive (its top folder dropped), the index in ``catalogs``; the program asked its version (not by the
    tests, whose program is a few fake bytes)."""
    part = lock["asterion"]
    folder = stage_dir / "asterion"
    folder.mkdir(parents=True)
    for entry in part["files"]:
        src = pinned_file(entry, cache)
        if not entry["filename"].endswith(".zip"):
            link_or_copy(src, folder / entry["dest"] / entry["filename"])
            continue
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                rel = PurePosixPath(info.filename).parts[1:]           # under the archive's own top folder
                if not rel or rel[0] not in ASTERION_KEEP or info.is_dir():
                    continue
                if ".." in rel or ":" in info.filename:
                    raise SystemExit(f"{src.name}: an unexpected path in the archive: {info.filename}")
                dest = folder.joinpath(*rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as fin, open(dest, "wb") as fout:
                    shutil.copyfileobj(fin, fout, 1 << 22)
    missing = [k for k in ASTERION_KEEP if not (folder / k).exists()]
    if missing:
        raise SystemExit(f"the Asterion archive has no {', '.join(missing)}")
    if ask_version and sys.platform == "win32":
        out = subprocess.run([str(folder / "asterion.exe"), "--version"], capture_output=True, text=True, timeout=60,
                             stdin=subprocess.DEVNULL).stdout
        if part["version"] not in out:
            raise SystemExit(f"the staged asterion.exe answers {out.strip()!r}, not {part['version']}")
    print("asterion:", part["version"], ", ".join(e["filename"] for e in part["files"]))
    return folder


def make_icon(path: Path) -> None:
    """The program's icon (client/float.svg) as a Windows .ico, drawn for each of its sizes."""
    from PIL import Image
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    rend = QSvgRenderer(str(REPO / "astro_gpu" / "client" / "float.svg"))
    images = []
    for n in (16, 24, 32, 48, 64, 128, 256):
        img = QImage(n, n, QImage.Format_RGBA8888)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        rend.render(p, QRectF(0, 0, n, n))
        p.end()
        images.append(Image.frombuffer("RGBA", (n, n), bytes(img.constBits()), "raw", "RGBA", img.bytesPerLine(), 1).copy())
    images[-1].save(path, format="ICO", sizes=[i.size for i in images], append_images=images[:-1])


def next_stage(base: Path) -> Path:
    """A new numbered folder under ``base``: a former stage stays where it is (this script never deletes a folder)."""
    base.mkdir(parents=True, exist_ok=True)
    n = 1 + max((int(d.name) for d in base.iterdir() if d.name.isdigit()), default=0)
    return base / f"{n:03d}"


def stage(stage_dir: Path, lock: dict, cache: Path) -> None:
    python = lock["python"]
    stage_dir.mkdir(parents=True)
    # the plate solver first: a missing or wrong pinned file stops the build before the long copies
    stage_asterion(stage_dir, lock, cache)
    # the private Python, a standalone build (with Tk for the progress window)
    tmp_py = stage_dir.with_name(stage_dir.name + "_python")
    run(uv(), "python", "install", python, "--install-dir", tmp_py, "--no-bin")
    inner = next(d for d in tmp_py.iterdir() if d.is_dir() and d.name.startswith(f"cpython-{python}-"))
    py = stage_dir / "python"
    # no pip (uv does its work) and no compiled caches: smaller, and the deepest paths the installer must copy are gone
    shutil.copytree(inner, py, ignore=lambda folder, names: [n for n in names if n in ("__pycache__", "pip")
                                                             or n.startswith("pip-")])
    if not (py / "pythonw.exe").is_file() or not (py / "Lib" / "tkinter").is_dir():
        raise SystemExit("the private Python has no pythonw/tkinter")
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for dll in VC_RUNTIME:
        shutil.copy2(system32 / dll, py / dll)
    shutil.copy2(uv(), stage_dir / "uv.exe")
    # the app: what git tracks, and the network git does not carry
    app = stage_dir / "app"
    listed = subprocess.run(["git", "ls-files", "-z", "astro_gpu"], capture_output=True, text=True, cwd=REPO, check=True).stdout
    for rel in filter(None, listed.split("\0")):
        src = REPO / rel
        if src.is_file():
            dest = app / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    star = find_star_model()
    dest = app / STAR_MODEL
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(star, dest)
    print("star model:", star, hashlib.sha256(star.read_bytes()).hexdigest()[:10])
    # the setup scripts, the lock, the root certificates
    setup = stage_dir / "setup"
    setup.mkdir()
    for f in (HERE / "setup_env").glob("*.py"):
        shutil.copy2(f, setup / f.name)
    shutil.copy2(LOCK_FILE, setup / "lock.json")
    if (LOCK_FILE.parent / "wheels").is_dir():
        shutil.copytree(LOCK_FILE.parent / "wheels", setup / "wheels")
    import certifi
    shutil.copy2(certifi.where(), setup / "cacert.pem")
    make_icon(stage_dir / "float.ico")


def iscc() -> str:
    for cand in (os.environ.get("ISCC"), shutil.which("iscc"),
                 Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
                 Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe")):
        if cand and Path(cand).is_file():
            return str(cand)
    raise SystemExit("Inno Setup is not installed (ISCC.exe); see https://jrsoftware.org/isdl.php")


def compile_installer(stage_dir: Path, out: Path, version: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    limit = detect.root_limit()
    iss = stage_dir / "float.iss"
    # ISCC reads non-ASCII text only from a file with a byte-order mark
    iss.write_text((HERE / "float.iss").read_text(encoding="utf-8"), encoding="utf-8-sig")
    run(iscc(), f"/DStageDir={stage_dir}", f"/DOutputDir={out}", f"/DAppVersion={version}", f"/DMaxRootLen={limit}", iss)
    return out / "FLOAT-Setup.exe"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", choices=("lock", "asterion"))
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--stage", type=Path, default=REPO / "build" / "installer-stage")
    ap.add_argument("--asterion-cache", type=Path, default=ASTERION_CACHE)
    args = ap.parse_args()
    if args.what == "lock":
        make_lock()
        return 0
    check_lock()
    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    if args.what == "asterion":
        folder = stage_asterion(next_stage(args.stage), lock, args.asterion_cache)
        print(folder)
        return 0
    with open(REPO / "pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    stage_dir = next_stage(args.stage)
    stage(stage_dir, lock, args.asterion_cache)
    exe = compile_installer(stage_dir, args.out, version)
    print(f"{exe}  {exe.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
