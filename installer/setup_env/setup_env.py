"""The second half of the installation: the libraries (2 to 3 GB for a NVIDIA card), with a progress window.

Run by the installer right after it copied the files, and again by the launcher if a former run did not finish. Every
step can be run again: what is already right stays, a re-install or an upgrade only fetches what changed, and a change
of card (a new driver) is picked up the same way. The Asterion plate solver and its index came with the files: they are
only checked at the end, and a problem with them is said without failing the installation. Standard library only, from
the private Python.

    python setup_env.py --root <install folder> [--no-gui] [--lock lock.json] [--lang fr|en]

Exit code 0: ready. 2: stopped, or failed after the retries (the launcher will offer it again).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detect  # noqa: E402
import fetch  # noqa: E402

MARKER = ".float_ok.json"
SMOKE_CODE = """\
import json, torch, numba, cv2, PySide6.QtWidgets
import astro_gpu, astro_gpu.api, astro_gpu.client.app
print("SMOKE" + json.dumps({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "count": torch.cuda.device_count(),
                            "app": astro_gpu.__version__}))
"""

TEXTS = {
    "en": {
        "title": "FLOAT setup",
        "heading": "Preparing FLOAT",
        "checking": "Looking at this computer...",
        "disk": "Checking the free disk space...",
        "python": "Preparing the private Python...",
        "download": "Downloading {what}",
        "download_stat": "File {i} of {n}  ·  {done} of {total} in all  ·  {speed}/s  ·  about {left} left",
        "install": "Installing the libraries: a few minutes, up to half an hour on a hard disk...",
        "smoke": "Checking the installation...",
        "done": "Installation finished.",
        "failed": "The installation did not finish.",
        "retry": "Retry",
        "cancel": "Cancel",
        "close": "Close",
        "cancel_ask": "Stop the installation?\n\nWhat was already downloaded is kept for the next attempt.",
        "cancelled": "Stopped. Open FLOAT from the Start menu to finish the installation.",
        "net_error": "The download failed: no connection, or the server did not answer.\n"
                     "Check your internet connection, then press Retry; what was already downloaded is kept.\n\n({detail})",
        "disk_error": "Not enough disk space on this drive: {need} needed, {free} free.\n"
                      "Free some space, then press Retry, or re-run the installer and choose another folder.",
        "disk_full": "The disk became full during the download.\nFree some space, then press Retry.",
        "hash_error": "A downloaded file is damaged (checksum).\nPress Retry to download it again.\n\n({detail})",
        "install_error": "Installing the libraries failed.\n"
                         "If FLOAT is open, close it, then press Retry.\nDetails: {log}",
        "smoke_error": "FLOAT does not start in the new installation.\nPress Retry, or send the details: {log}",
        "too_long": "The installation folder's path is too long for Windows ({n} characters, {max} at most).\n"
                    "Run the installer again and choose a shorter folder.",
        "torch_cuda": "PyTorch for NVIDIA cards",
        "torch_cpu": "PyTorch (processor version)",
        "qt": "the interface (Qt)",
        "library": "the library {name}",
        "card_ok": "Computations will run on your NVIDIA card: {name}, driver {driver}.",
        "card_none": "No NVIDIA card found: FLOAT will compute on the processor, complete but slower.",
        "card_old": "Your {name} has an old driver ({driver}; at least {min} is needed): FLOAT will compute on the "
                    "processor.\nTo use the card, update its driver from https://www.nvidia.com/drivers, then run this "
                    "installer again.",
        "card_unknown": "The driver of your {name} could not be read: FLOAT will compute on the processor.\n"
                        "To use the card, update its driver from https://www.nvidia.com/drivers, then run this installer "
                        "again.",
        "card_other": "Your {name} is not supported by default yet: FLOAT will compute on the processor. "
                      "Settings > Compute has the beta options.",
        "cuda_no": "PyTorch does not see your NVIDIA card yet. FLOAT works on the processor; updating the driver "
                   "from https://www.nvidia.com/drivers should fix it.",
        "asterion_program": "The Asterion plate solver does not start in this installation. FLOAT works all the same: "
                            "Astrometry uses its own search, slower. Running this installer again puts it back.",
        "asterion_index": "The star index of the Asterion plate solver is missing or incomplete in this installation. "
                          "FLOAT works all the same: Astrometry uses its own search, slower. Running this installer "
                          "again puts it back.",
    },
    "fr": {
        "title": "Installation de FLOAT",
        "heading": "Préparation de FLOAT",
        "checking": "Analyse de cet ordinateur...",
        "disk": "Vérification de l'espace disque...",
        "python": "Préparation du Python privé...",
        "download": "Téléchargement : {what}",
        "download_stat": "Fichier {i} sur {n}  ·  {done} sur {total} au total  ·  {speed}/s  ·  encore environ {left}",
        "install": "Installation des bibliothèques : quelques minutes, jusqu'à une demi-heure sur un disque dur...",
        "smoke": "Vérification de l'installation...",
        "done": "Installation terminée.",
        "failed": "L'installation n'est pas allée au bout.",
        "retry": "Réessayer",
        "cancel": "Annuler",
        "close": "Fermer",
        "cancel_ask": "Arrêter l'installation ?\n\nCe qui est déjà téléchargé est conservé pour la prochaine fois.",
        "cancelled": "Arrêté. Ouvrez FLOAT depuis le menu Démarrer pour terminer l'installation.",
        "net_error": "Le téléchargement a échoué : pas de connexion, ou le serveur n'a pas répondu.\n"
                     "Vérifiez votre connexion internet, puis appuyez sur Réessayer ; ce qui est déjà téléchargé est "
                     "conservé.\n\n({detail})",
        "disk_error": "Espace disque insuffisant sur ce disque : {need} nécessaires, {free} libres.\n"
                      "Libérez de la place puis appuyez sur Réessayer, ou relancez l'installation sur un autre dossier.",
        "disk_full": "Le disque s'est rempli pendant le téléchargement.\nLibérez de la place puis appuyez sur Réessayer.",
        "hash_error": "Un fichier téléchargé est abîmé (somme de contrôle).\nAppuyez sur Réessayer pour le télécharger à "
                      "nouveau.\n\n({detail})",
        "install_error": "L'installation des bibliothèques a échoué.\n"
                         "Si FLOAT est ouvert, fermez-le, puis appuyez sur Réessayer.\nDétails : {log}",
        "smoke_error": "FLOAT ne démarre pas dans la nouvelle installation.\n"
                       "Appuyez sur Réessayer, ou envoyez les détails : {log}",
        "too_long": "Le chemin du dossier d'installation est trop long pour Windows ({n} caractères, {max} au maximum).\n"
                    "Relancez l'installation et choisissez un dossier plus court.",
        "torch_cuda": "PyTorch pour cartes NVIDIA",
        "torch_cpu": "PyTorch (version processeur)",
        "qt": "l'interface (Qt)",
        "library": "la bibliothèque {name}",
        "card_ok": "Les calculs se feront sur votre carte NVIDIA : {name}, pilote {driver}.",
        "card_none": "Aucune carte NVIDIA détectée : FLOAT calculera sur le processeur, complet mais plus lent.",
        "card_old": "Votre {name} a un pilote ancien ({driver} ; il faut au moins {min}) : FLOAT calculera sur le "
                    "processeur.\nPour utiliser la carte, mettez à jour son pilote depuis https://www.nvidia.com/drivers, "
                    "puis relancez cette installation.",
        "card_unknown": "Le pilote de votre {name} n'a pas pu être lu : FLOAT calculera sur le processeur.\n"
                        "Pour utiliser la carte, mettez à jour son pilote depuis https://www.nvidia.com/drivers, puis "
                        "relancez cette installation.",
        "card_other": "Votre {name} n'est pas encore prise en charge par défaut : FLOAT calculera sur le "
                      "processeur. Réglages > Calcul propose les options bêta.",
        "cuda_no": "PyTorch ne voit pas encore votre carte NVIDIA. FLOAT fonctionne sur le processeur ; mettre à "
                   "jour le pilote depuis https://www.nvidia.com/drivers devrait résoudre cela.",
        "asterion_program": "Le solveur astrométrique Asterion ne démarre pas dans cette installation. FLOAT fonctionne "
                            "quand même : l'Astrométrie utilise sa propre recherche, plus lente. Relancer cette "
                            "installation le remet en place.",
        "asterion_index": "L'index d'étoiles du solveur astrométrique Asterion manque ou est incomplet dans cette "
                          "installation. FLOAT fonctionne quand même : l'Astrométrie utilise sa propre recherche, plus "
                          "lente. Relancer cette installation le remet en place.",
    },
}


def windows_language() -> str:
    """French when Windows' display language is, else English."""
    try:
        return "fr" if ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0xFF == 0x0C else "en"
    except (AttributeError, OSError):
        return "en"


def human(n: float, lang: str) -> str:
    unit = "Go" if lang == "fr" else "GB"
    small = "Mo" if lang == "fr" else "MB"
    s = f"{n / 1e9:.1f} {unit}" if n >= 1e9 else f"{n / 1e6:.0f} {small}"
    return s.replace(".", ",") if lang == "fr" else s


def duration(seconds: float, lang: str) -> str:
    seconds = max(1, int(seconds))
    if seconds < 90:
        return f"{seconds} s"
    return f"{round(seconds / 60)} min"


def card_message(choice: detect.Choice, lang: str) -> str:
    t = TEXTS[lang]
    key = {"ok": "card_ok", "no-card": "card_none", "old-driver": "card_old", "no-driver": "card_unknown",
           "other-vendor": "card_other", "error": "card_none"}[choice.why]
    return t[key].format(name=choice.name, driver=choice.driver, min=detect.driver_text(detect.CUDA_MIN_DRIVER))


def what_is(w: fetch.Wheel, lang: str) -> str:
    t = TEXTS[lang]
    n = w.name.lower()
    if n == "torch":
        return t["torch_cuda"] if "cu" in w.version.split("+")[-1] and "cpu" not in w.version else t["torch_cpu"]
    if n.startswith(("pyside", "shiboken")):
        return t["qt"]
    return t["library"].format(name=w.name)


def load_lock(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wheels_of(lock: dict, build: str) -> list[fetch.Wheel]:
    return [fetch.Wheel(**w) for w in lock["builds"][build]]


def norm(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").lower()


def installed_versions(site: Path) -> dict[str, str]:
    """What the environment holds, from its *.dist-info folders."""
    out = {}
    if site.is_dir():
        for d in site.glob("*.dist-info"):
            name, _, version = d.name[:-len(".dist-info")].partition("-")
            out[norm(name)] = version
    return out


class Failure(Exception):
    def __init__(self, message: str, retry: bool = True):
        super().__init__(message)
        self.message = message
        self.retry = retry


class Installer:
    def __init__(self, root: Path, ui, lock_path: Path | None = None, lang: str | None = None, env=None):
        self.root = Path(root)
        self.ui = ui
        self.env = os.environ if env is None else env
        self.lang = lang or windows_language()
        self.t = TEXTS[self.lang]
        self.lock_path = lock_path or self.root / "setup" / "lock.json"
        self.lock = load_lock(self.lock_path)
        self.lock_id = fetch.sha256_of(self.lock_path)
        self.env_dir = self.root / "env"
        self.site = self.env_dir / "Lib" / "site-packages"
        self.env_py = self.env_dir / "Scripts" / "python.exe"
        self.downloads = self.root / "downloads"
        self.log = self.root / "setup.log"
        self.cancel = threading.Event()
        self.choice: detect.Choice | None = None
        self.summary = ""
        self.proc = None

    def note(self, text: str):
        try:
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}\n")
        except OSError:
            pass

    def tool_env(self) -> dict:
        env = {k: v for k, v in self.env.items() if not k.startswith(("UV_", "PIP_")) and
               k not in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX")}
        env.update({"UV_CACHE_DIR": str(self.root / "cache"), "UV_PYTHON_DOWNLOADS": "never", "UV_NO_CONFIG": "1",
                    "PYTHONUTF8": "1", "UV_NO_PROGRESS": "1"})
        return env

    def run_tool(self, cmd: list, fail_key: str) -> str:
        self.note("run: " + " ".join(str(c) for c in cmd))
        self.proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", env=self.tool_env(), cwd=str(self.root),
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = []
        for line in self.proc.stdout:
            lines.append(line)
            self.note(line.rstrip())
            if self.cancel.is_set():
                self.proc.kill()
        code = self.proc.wait()
        if self.cancel.is_set():
            raise fetch.Cancelled
        if code != 0:
            raise Failure(self.t[fail_key].format(log=self.log))
        return "".join(lines)

    # ------------------------------------------------------------------ steps
    def check_folder(self):
        n, limit = detect.root_length(self.root), detect.root_limit()
        if n > limit:
            raise Failure(self.t["too_long"].format(n=n, max=limit), retry=False)

    def choose_build(self):
        self.ui.stage(self.t["checking"])
        self.choice = detect.detect(self.env)
        self.note(f"card: {self.choice}")
        return self.choice.build

    def env_is_right(self) -> bool:
        if not self.env_py.is_file():
            return False
        try:
            r = subprocess.run([str(self.env_py), "-I", "-c", "import sys;print(sys.version.split()[0])"], text=True,
                               capture_output=True, timeout=60, env=self.tool_env(),
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            return False
        return r.returncode == 0 and r.stdout.strip() == self.lock["python"]

    def ensure_env(self):
        self.ui.stage(self.t["python"])
        if self.env_is_right():
            return
        uv = self.root / "uv.exe"
        extra = ["--clear"] if self.env_dir.exists() else []
        self.run_tool([uv, "venv", "--python", self.root / "python" / "python.exe", "--no-project", *extra, self.env_dir],
                      "install_error")

    def missing_wheels(self, wheels: list[fetch.Wheel]) -> list[fetch.Wheel]:
        have = installed_versions(self.site)
        return [w for w in wheels if have.get(norm(w.name)) != w.version]

    def check_disk(self, missing: list[fetch.Wheel]):
        self.ui.stage(self.t["disk"])
        need = fetch.needed_bytes(missing, fetch.hard_links_work(self.root))
        free = fetch.free_bytes(self.root)
        self.note(f"disk: need {need}, free {free}")
        if free < need:
            raise Failure(self.t["disk_error"].format(need=human(need, self.lang), free=human(free, self.lang)))

    def download(self, missing: list[fetch.Wheel]):
        context = fetch.ssl_context(self.root / "setup" / "cacert.pem")
        self.downloads.mkdir(parents=True, exist_ok=True)
        for w in missing:
            if w.bundled:
                shutil.copy2(self.root / "setup" / "wheels" / w.filename, self.downloads / w.filename)
        to_fetch = [w for w in missing if not w.bundled]
        self.note(f"downloading {len(to_fetch)} files, {sum(w.size for w in to_fetch)} bytes")
        try:
            fetch.download_all(to_fetch, self.downloads, self.ui.download, self.cancel, context)
        except fetch.DownloadError as e:
            self.note(f"download error: {e.kind}: {e.detail}")
            key = {"network": "net_error", "disk": "disk_full", "hash": "hash_error"}[e.kind]
            raise Failure(self.t[key].format(detail=e.detail)) from e

    def sync(self, wheels: list[fetch.Wheel]):
        self.ui.stage(self.t["install"])
        self.ui.indeterminate()
        self.downloads.mkdir(parents=True, exist_ok=True)      # uv refuses a missing folder, even with nothing to fetch
        self.site.mkdir(parents=True, exist_ok=True)
        req = self.root / "setup" / "requirements.txt"
        req.write_text("".join(f"{w.name}=={w.version}\n" for w in wheels), encoding="utf-8")
        self.run_tool([self.root / "uv.exe", "pip", "sync", req, "--python", self.env_py, "--no-index", "--find-links",
                       self.downloads, "--offline"], "install_error")
        # the app's own code, found by every Python of the environment (relative: the path may have accents)
        pth = self.site / "float_app.pth"
        pth.write_text("..\\..\\..\\app\n", encoding="ascii")

    def smoke(self, wheels: list[fetch.Wheel]):
        self.ui.stage(self.t["smoke"])
        self.ui.indeterminate()
        out = self.run_tool([self.env_py, "-c", SMOKE_CODE], "smoke_error")
        info = {}
        for line in out.splitlines():
            if line.startswith("SMOKE"):
                info = json.loads(line[5:])
        self.note(f"smoke: {info}")
        cuda_build = any(w.name.lower() == "torch" and "cu1" in w.version for w in wheels)
        return info, cuda_build

    def check_asterion(self) -> str:
        """The plate solver the package carries: ``asterion --version`` answers the lock's version and the index files
        are there at their pinned sizes (not their checksums: 378 MB read at each setup for what Setup has just copied).
        The text to tell the user when not, "" when all is right or the lock carries no plate solver. Never a failure:
        without it FLOAT starts all the same, and its Astrometry uses its own search."""
        part = self.lock.get("asterion")
        if not part:
            return ""
        folder = self.root / "asterion"
        try:
            r = subprocess.run([str(folder / "asterion.exe"), "--version"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=60, stdin=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            answer = r.stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            answer = f"{type(e).__name__}: {e}"
        m = re.search(r"Asterion\D*(\d+\.\d+\.\d+)", answer)
        if not m or m.group(1) != part["version"]:
            self.note(f"asterion: the program answers {answer[:200]!r}, not {part['version']}")
            return self.t["asterion_program"]
        for f in part["files"]:
            if f.get("dest") == "catalogs":
                p = folder / "catalogs" / f["filename"]
                size = p.stat().st_size if p.is_file() else -1
                if size != f["size"]:
                    self.note(f"asterion: {p} has {size} bytes, not {f['size']}")
                    return self.t["asterion_index"]
        self.note(f"asterion: {m.group(1)}, index there")
        return ""

    def finish(self, build: str, info: dict, cuda_build: bool):
        self.summary = card_message(self.choice, self.lang)
        if cuda_build and not info.get("cuda"):
            self.summary += "\n" + self.t["cuda_no"]
        try:
            problem = self.check_asterion()
        except Exception as e:  # noqa: BLE001 - a check that breaks must not stop the installation
            self.note(f"asterion: check failed: {type(e).__name__}: {e}")
            problem = self.t["asterion_program"]
        if problem:
            self.summary += "\n" + problem
        self.note("summary: " + self.summary)
        (self.env_dir / MARKER).write_text(json.dumps({"lock": self.lock_id, "build": build, "python": self.lock["python"],
                                                       "date": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": info.get("torch")},
                                                      indent=1), encoding="utf-8")
        # the wheels are installed now: free their place (the app itself does not use them)
        for f in self.downloads.glob("*") if self.downloads.is_dir() else []:
            try:
                f.unlink()
            except OSError:
                pass
        try:
            self.downloads.rmdir()
        except OSError:
            pass
        try:
            self.run_tool([self.root / "uv.exe", "cache", "clean"], "install_error")
        except (Failure, OSError):
            self.note("cache not cleaned")

    def attempt(self) -> None:
        self.check_folder()
        build = self.choose_build()
        wheels = wheels_of(self.lock, build)
        self.ensure_env()
        missing = self.missing_wheels(wheels)
        self.note(f"build {build}: {len(wheels)} packages, {len(missing)} to fetch")
        self.check_disk(missing)
        if missing:                                          # the environment changes: not "ready" until the end
            (self.env_dir / MARKER).unlink(missing_ok=True)
            self.download(missing)
        self.sync(wheels)
        info, cuda_build = self.smoke(wheels)
        self.finish(build, info, cuda_build)

    def run(self) -> int:
        """The whole work, in its own thread; the UI decides whether a failure is retried."""
        while True:
            try:
                self.attempt()
                self.ui.done(True, self.summary)
                return 0
            except fetch.Cancelled:
                self.ui.done(False, self.t["cancelled"])
                return 2
            except Failure as e:
                self.note("failure: " + e.message)
                if not (e.retry and self.ui.ask_retry(e.message)):
                    self.ui.done(False, e.message)
                    return 2
            except Exception as e:  # noqa: BLE001 - never a crash without a message
                self.note(f"unexpected: {type(e).__name__}: {e}")
                msg = self.t["install_error"].format(log=self.log)
                if not self.ui.ask_retry(msg):
                    self.ui.done(False, msg)
                    return 2


def is_ready(root: Path) -> bool:
    """Whether the installation is complete for this lock file (the launcher's question)."""
    m = root / "env" / MARKER
    try:
        return json.loads(m.read_text(encoding="utf-8"))["lock"] == fetch.sha256_of(root / "setup" / "lock.json")
    except (OSError, ValueError, KeyError):
        return False


# ---------------------------------------------------------------------- the two faces: a window, or plain text
class TextUi:
    """--no-gui: lines on the console (or the log), and a failure is retried on its own a few times."""

    def __init__(self, lang: str):
        self.lang = lang
        self.retries = 4
        self.last = 0.0

    def write(self, text: str):
        try:
            print(text, flush=True)
        except (OSError, ValueError, AttributeError):
            pass

    def stage(self, text: str):
        self.write(text)

    def indeterminate(self):
        pass

    def download(self, p: fetch.Progress):
        if time.monotonic() - self.last > 5:
            self.last = time.monotonic()
            self.write(f"{p.wheel.filename}: {human(p.overall_done, self.lang)} / {human(p.overall_total, self.lang)}")

    def ask_retry(self, message: str) -> bool:
        self.write(message)
        if self.retries <= 0:
            return False
        self.retries -= 1
        time.sleep(15)
        return True

    def done(self, ok: bool, message: str):
        self.write(message)


class TkUi:
    """A window whose worker thread reports through a queue (Tk stays in the main thread)."""

    def __init__(self, lang: str):
        self.lang = lang
        self.t = TEXTS[lang]
        self.q: queue.Queue = queue.Queue()
        self.answer: bool = False
        self.answered = threading.Event()
        self.result = 2
        self.cancel_event: threading.Event | None = None

    # worker-thread side
    def stage(self, text): self.q.put(("stage", text))
    def indeterminate(self): self.q.put(("indeterminate",))
    def download(self, p: fetch.Progress): self.q.put(("download", p))
    def done(self, ok, message): self.q.put(("done", ok, message))

    def ask_retry(self, message) -> bool:
        self.answered.clear()
        self.q.put(("fail", message))
        self.answered.wait()
        return self.answer

    # main-thread side
    def build(self):
        import tkinter as tk
        from tkinter import ttk
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
        self.tk = tk
        root = self.window = tk.Tk()
        root.title(self.t["title"])
        root.geometry("620x340")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        frame = ttk.Frame(root, padding=18)
        frame.pack(fill="both", expand=True)
        buttons = ttk.Frame(frame)                      # packed first: a long message never pushes the buttons out
        buttons.pack(side="bottom", fill="x")
        self.heading = ttk.Label(frame, text=self.t["heading"], font=("Segoe UI", 12, "bold"))
        self.heading.pack(anchor="w")
        self.line = ttk.Label(frame, text="", wraplength=580, justify="left")
        self.line.pack(anchor="w", pady=(10, 4))
        self.bar = ttk.Progressbar(frame, mode="indeterminate", maximum=1000, length=580)
        self.bar.pack(fill="x")
        self.bar.start(40)
        self.stat = ttk.Label(frame, text="", foreground="#555555")
        self.stat.pack(anchor="w", pady=(4, 0))
        self.message = ttk.Label(frame, text="", wraplength=580, justify="left")
        self.message.pack(anchor="w", pady=(10, 0), fill="x")
        self.b_cancel = ttk.Button(buttons, text=self.t["cancel"], command=self.on_cancel)
        self.b_cancel.pack(side="right")
        self.b_retry = ttk.Button(buttons, text=self.t["retry"], command=self.on_retry)
        self.b_close = ttk.Button(buttons, text=self.t["close"], command=root.destroy)
        root.after(80, self.poll)

    def poll(self):
        try:
            while True:
                self.apply(self.q.get_nowait())
        except queue.Empty:
            pass
        if self.window.winfo_exists():
            self.window.after(80, self.poll)

    def apply(self, item):
        kind = item[0]
        if kind == "stage":
            self.line.config(text=item[1])
            self.stat.config(text="")
            self.message.config(text="")
        elif kind == "indeterminate":
            self.bar.config(mode="indeterminate")
            self.bar.start(40)
            self.stat.config(text="")
        elif kind == "download":
            p = item[1]
            self.bar.stop()
            self.bar.config(mode="determinate")
            self.bar["value"] = 1000 * p.overall_done / max(p.overall_total, 1)
            self.line.config(text=self.t["download"].format(what=what_is(p.wheel, self.lang)))
            left = (p.overall_total - p.overall_done) / p.speed if p.speed > 0 else 0
            self.stat.config(text=self.t["download_stat"].format(i=p.index, n=p.count, done=human(p.overall_done, self.lang),
                                                                 total=human(p.overall_total, self.lang),
                                                                 speed=human(p.speed, self.lang), left=duration(left, self.lang)))
        elif kind == "fail":
            self.bar.stop()
            self.heading.config(text=self.t["failed"])
            self.message.config(text=item[1])
            self.b_retry.pack(side="right", padx=(0, 8))
        elif kind == "done":
            self.result = 0 if item[1] else 2
            self.bar.stop()
            self.bar.config(mode="determinate")
            self.bar["value"] = 1000 if item[1] else 0
            self.heading.config(text=self.t["done"] if item[1] else self.t["failed"])
            self.line.config(text="")
            self.stat.config(text="")
            self.message.config(text=item[2])
            self.b_cancel.pack_forget()
            self.b_retry.pack_forget()
            self.b_close.pack(side="right")

    def on_retry(self):
        self.b_retry.pack_forget()
        self.heading.config(text=self.t["heading"])
        self.message.config(text="")
        self.bar.config(mode="indeterminate")
        self.bar.start(40)
        self.answer = True
        self.answered.set()

    def on_cancel(self):
        from tkinter import messagebox
        if not messagebox.askyesno(self.t["title"], self.t["cancel_ask"], parent=self.window):
            return
        self.b_cancel.config(state="disabled")
        if self.cancel_event is not None:
            self.cancel_event.set()
        self.answer = False
        self.answered.set()

    def on_close(self):
        if self.b_close.winfo_ismapped():
            self.window.destroy()
        else:
            self.on_cancel()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lock")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--lang", choices=("fr", "en"))
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    lang = args.lang or windows_language()
    gui = not args.no_gui
    if gui:
        try:
            import tkinter  # noqa: F401
        except ImportError:
            gui = False
    ui = TkUi(lang) if gui else TextUi(lang)
    inst = Installer(root, ui, Path(args.lock) if args.lock else None, lang)
    if not gui:
        return inst.run()
    ui.cancel_event = inst.cancel
    try:
        ui.build()
    except Exception as e:  # noqa: BLE001 - no window possible here: the same work, in text
        inst.note(f"no window: {e}")
        inst.ui = TextUi(lang)
        return inst.run()
    worker = threading.Thread(target=inst.run, daemon=True)
    worker.start()
    ui.window.mainloop()
    return ui.result


if __name__ == "__main__":
    sys.exit(main())
