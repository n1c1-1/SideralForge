"""Stage base class and the execution context handed to each stage."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from .params import Auto, StageParams

if TYPE_CHECKING:
    from .journal import Journal
    from .project import Project


class Interrupted(Exception):
    """Raised inside a stage when the user asked to stop; the stage can be resumed later."""


class StageError(Exception):
    """A stage cannot run (missing input, impossible data...). The message is shown to the user."""


class Stage:
    """One processing step. Subclasses define ``name``, ``title``, ``Params`` and ``run``.

    ``version`` must be bumped whenever the algorithm changes in a way that alters outputs: stored results of an
    older version are then considered stale and recomputed.
    """

    name: ClassVar[str]
    title: ClassVar[str]
    description: ClassVar[str] = ""
    deps: ClassVar[tuple[str, ...]] = ()
    version: ClassVar[str] = "1"
    Params: ClassVar[type[StageParams]] = StageParams
    #: rough cost model used for the ETA before the stage reports progress: seconds per megapixel-frame
    cost_hint: ClassVar[float] = 0.0
    #: whether the stage needs internet access (catalogues, sky surveys)
    network: ClassVar[bool] = False

    def auto(self, ctx: Context) -> dict[str, Auto]:
        """Data-driven defaults. May read outputs of dependencies (they are complete when this is called)."""
        return {}

    def run(self, ctx: Context, p: Any) -> None:
        raise NotImplementedError


class Context:
    """Everything a running stage may touch: its folder, its inputs, the journal, the GPU, checkpoints."""

    def __init__(self, project: Project, stage: Stage, journal: Journal, device: str = "cuda:0"):
        self.project = project
        self.stage = stage
        self.journal = journal
        self.dir = project.stage_dir(stage.name)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device_name = device
        self._t0 = time.time()
        self._last_progress = 0.0
        self.metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------ files
    def out(self, name: str) -> Path:
        return self.dir / name

    def input(self, stage: str, name: str) -> Path:
        p = self.project.stage_dir(stage) / name
        if not p.exists():
            raise StageError(f"missing input {stage}/{name}: run stage '{stage}' first")
        return p

    def save(self, name: str, arr: np.ndarray) -> Path:
        p = self.out(name)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(p)
        return p

    def load(self, stage: str, name: str, mmap: bool = False) -> np.ndarray:
        return np.load(self.input(stage, name), mmap_mode="r" if mmap else None)

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.out(name)
        p.write_text(json.dumps(obj, indent=1, default=_np_default), encoding="utf-8")
        return p

    def load_json(self, stage: str, name: str) -> Any:
        return json.loads(self.input(stage, name).read_text(encoding="utf-8"))

    def manifest(self, stage: str) -> dict:
        p = self.project.stage_dir(stage) / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def metric_of(self, stage: str, key: str, default: Any = None) -> Any:
        return self.manifest(stage).get("metrics", {}).get(key, default)

    def has(self, stage: str) -> bool:
        """Whether there is a result of ``stage`` to read. In a v2 workflow, a stage its chain does not have (an
        optional input) resolves to a folder that does not exist (core/store.py)."""
        return self.project.stage_dir(stage).is_dir()

    @property
    def scratch(self) -> Path:
        p = self.project.scratch / self.stage.name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------ checkpoints (resume inside a stage)
    @property
    def ckpt(self) -> Path:
        p = self.dir / "_ckpt"
        p.mkdir(exist_ok=True)
        return p

    def clear_ckpt(self) -> None:
        shutil.rmtree(self.dir / "_ckpt", ignore_errors=True)

    # ------------------------------------------------------------------ journal
    def log(self, msg: str, level: str = "info") -> None:
        self.journal.emit("log", stage=self.stage.name, level=level, msg=msg)

    def metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value
        self.journal.emit("metric", stage=self.stage.name, key=key, value=value)

    def progress(self, frac: float, msg: str = "", force: bool = False) -> None:
        """Report progress (0..1). Throttled to one event per second; also checks for a stop request."""
        self.check_stop()
        now = time.time()
        if not force and now - self._last_progress < 1.0 and frac < 1.0:
            return
        self._last_progress = now
        el = now - self._t0
        eta = el / frac * (1 - frac) if frac > 0.02 else None
        self.journal.emit("progress", stage=self.stage.name, frac=float(frac), msg=msg, eta=eta, elapsed=el)

    def preview(self, name: str, rgb: np.ndarray, max_side: int = 1600) -> Path:
        """Save a preview: (H, W, 3), (3, H, W) or greyscale (H, W), float [0,1] or uint8, and announce it."""
        import cv2
        a = np.asarray(rgb)
        if a.ndim == 2:                                   # greyscale map
            a = np.repeat(a[:, :, None], 3, axis=2)
        if a.ndim == 3 and a.shape[0] == 3 and a.shape[2] != 3:
            a = a.transpose(1, 2, 0)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
        f = max_side / max(a.shape[:2])
        if f < 1:
            a = cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        p = self.dir / "previews" / f"{name}.jpg"
        p.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(p), np.ascontiguousarray(a[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.journal.emit("preview", stage=self.stage.name, name=name, path=str(p))
        return p

    def check_stop(self) -> None:
        if (self.project.logs / "STOP").exists():
            raise Interrupted("stop requested")

    # ------------------------------------------------------------------ GPU
    @property
    def device(self):
        import torch
        if not torch.cuda.is_available():
            return torch.device("cpu")
        d = torch.device(self.device_name)
        if d.type == "cuda" and (d.index or 0) >= torch.cuda.device_count():
            # the project names a GPU this machine (or this program) does not see: the first one is used
            if not getattr(self, "_device_warned", False):
                self._device_warned = True
                self.log(f"GPU {self.device_name} not available here ({torch.cuda.device_count()} visible): "
                         f"cuda:0 used", "warning")
            return torch.device("cuda:0")
        return d


def _np_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
