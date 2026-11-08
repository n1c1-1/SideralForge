"""The computing devices of the engine: what PyTorch offers on this machine, named as the engine names them ('cuda:N',
'xpu:N', 'mps', 'privateml:N', 'cpu'), and the few calls that differ from one to another (free memory, emptying the
cache, waiting for the card), which work or do nothing on each device.

An AMD card comes through AMD's own PyTorch build: as 'cuda' (the Radeon names and ``torch.version.hip`` tell it
apart), or as a device of its own ('privateml', through the DirectML package); an Intel card as 'xpu', an Apple one as
'mps'. Only NVIDIA is tested on the machines of this project: the others are offered as beta, and the self-test
(core/selftest.py) tells, stage by stage, whether the user's card gives the right result.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: what a run computes on, besides the processor
_DEVICE = re.compile(r"cuda(:\d+)?|xpu(:\d+)?|mps(:0)?|privateml(:\d+)?")
#: the memory the Stack stage needs on the card: about 2 GB for the accumulators of the last pass of a 12 Mpx output
#: (measured), proportional to the output pixels
STACK_GB_PER_MPX = 2.0 / 12.0
#: the smallest tile the networks accept when the memory is short (Denoise: 512 normally, Stars: 256)
SMALLEST_TILE = 128


@dataclass
class DeviceInfo:
    device: str                      # the engine's name: 'cuda:0', 'xpu:0', 'mps', 'privateml:0'
    name: str                        # the name the driver gives
    memory_gb: float
    kind: str                        # nvidia | amd | intel | apple | other
    beta: bool                       # offered, but never tested on the machines of this project


def is_device(name: str) -> bool:
    """Whether ``name`` is a device of the engine other than the CPU (present on this machine or not)."""
    return bool(_DEVICE.fullmatch((name or "").strip().lower()))


def _kind_of(name: str) -> str:
    low = name.lower()
    if "amd" in low or "radeon" in low:
        return "amd"
    if "intel" in low or "arc" in low.split():
        return "intel"
    if "apple" in low:
        return "apple"
    return "nvidia" if any(w in low for w in ("nvidia", "geforce", "quadro", "tesla", "rtx", "gtx")) else "other"


def list_devices() -> list[DeviceInfo]:
    """The cards PyTorch sees here, in the engine's order: the 'cuda' ones, then the others. Raises ImportError when
    PyTorch is missing; a backend that fails is left out."""
    import torch
    out: list[DeviceInfo] = []
    try:
        if torch.cuda.is_available():
            hip = getattr(torch.version, "hip", None)
            for i in range(torch.cuda.device_count()):
                pr = torch.cuda.get_device_properties(i)
                kind = "amd" if hip else _kind_of(pr.name)
                out.append(DeviceInfo(f"cuda:{i}", pr.name, pr.total_memory / 2 ** 30, kind, kind != "nvidia"))
    except Exception:  # noqa: BLE001 - a backend that fails is simply not offered
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                pr = torch.xpu.get_device_properties(i)
                out.append(DeviceInfo(f"xpu:{i}", str(getattr(pr, "name", "Intel GPU")),
                                      float(getattr(pr, "total_memory", 0)) / 2 ** 30, "intel", True))
    except Exception:  # noqa: BLE001
        pass
    try:
        if torch.backends.mps.is_available():
            mem = float(torch.mps.recommended_max_memory()) if hasattr(torch.mps, "recommended_max_memory") else 0.0
            out.append(DeviceInfo("mps", "Apple GPU", mem / 2 ** 30, "apple", True))
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch_directml as dml  # AMD (or any other card) through DirectML
        for i in range(dml.device_count()):
            name = str(dml.device_name(i))
            out.append(DeviceInfo(f"privateml:{i}", name, 0.0, _kind_of(name), True))
    except Exception:  # noqa: BLE001 - no such package here
        pass
    return out


def _index(device: Any) -> int:
    d = str(device)
    return int(d.split(":")[1]) if ":" in d else 0


def _type(device: Any) -> str:
    return str(device).split(":")[0]


def available(device: Any) -> bool:
    """Whether this process can compute on ``device``, a name or a torch.device."""
    import torch
    t = _type(device)
    if t == "cpu":
        return True
    if t == "cuda":
        return torch.cuda.is_available()
    if t == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if t == "mps":
        return torch.backends.mps.is_available()
    return any(d.device == str(device) for d in list_devices())


def usable(device: Any) -> str:
    """``device`` if this process has it, else 'cpu'."""
    return str(device) if available(device) else "cpu"


def free_memory(device: Any) -> int | None:
    """Bytes free on the card now; None when this device does not tell."""
    import torch
    t, i = _type(device), _index(device)
    try:
        if t == "cuda":
            return int(torch.cuda.mem_get_info(i)[0])
        if t == "xpu":
            total = int(getattr(torch.xpu.get_device_properties(i), "total_memory", 0))
            return max(total - int(torch.xpu.memory_reserved(i)), 0) if total else None
        if t == "mps":
            return max(int(torch.mps.recommended_max_memory()) - int(torch.mps.driver_allocated_memory()), 0)
    except Exception:  # noqa: BLE001 - not known
        return None
    return None


def total_memory(device: Any) -> int | None:
    """Bytes on the card; None when this device does not tell."""
    import torch
    t, i = _type(device), _index(device)
    try:
        if t == "cuda":
            return int(torch.cuda.get_device_properties(i).total_memory)
        if t == "xpu":
            return int(getattr(torch.xpu.get_device_properties(i), "total_memory", 0)) or None
        if t == "mps":
            return int(torch.mps.recommended_max_memory())
    except Exception:  # noqa: BLE001
        return None
    return None


def empty_cache(device: Any = None) -> None:
    """Give the card's unused memory back (every card when ``device`` is None); nothing on the others."""
    import torch
    t = None if device is None else _type(device)
    try:
        if t in (None, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if t in (None, "xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        if t in (None, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - a cache that cannot be emptied changes nothing
        pass


def synchronize(device: Any) -> None:
    """Wait for the card to finish what it was given; nothing on the others."""
    import torch
    t = _type(device)
    try:
        if t == "cuda":
            torch.cuda.synchronize(device if isinstance(device, torch.device) else str(device))
        elif t == "xpu":
            torch.xpu.synchronize()
        elif t == "mps":
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001
        pass


def is_out_of_memory(e: BaseException) -> bool:
    """Whether an exception says the card is out of memory (each backend words it its own way)."""
    import torch
    if isinstance(e, torch.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and any(w in str(e).lower() for w in ("out of memory", "out_of_resources"))


def stack_memory(grid_mpx: float) -> float:
    """GB the Stack stage needs on the card for an output of ``grid_mpx`` megapixels."""
    return float(grid_mpx) * STACK_GB_PER_MPX


def shrink_on_memory(fn, tile: int, device: Any, smallest: int = SMALLEST_TILE):
    """``fn(tile)``; when the card runs out of memory, again with a tile half the size, down to ``smallest``. The
    tile is what the card holds at once, so a card with little memory is only slower."""
    while True:
        try:
            return fn(tile)
        except Exception as e:  # noqa: BLE001 - a memory error shrinks the tile, any other is raised
            if not is_out_of_memory(e) or tile / 2 < smallest:
                raise
            empty_cache(device)
            tile //= 2


def tile_that_fits(tile: int, device: Any, gb_at_tile: float, smallest: int = SMALLEST_TILE) -> int:
    """The largest tile, from ``tile`` down by halves, that leaves a margin in the card's free memory; ``gb_at_tile``:
    what the network takes at ``tile`` (the memory goes as the tile's area). A device that does not tell its memory:
    ``tile``, and ``shrink_on_memory`` handles a memory error."""
    free = free_memory(device)
    if free is None:
        return tile
    while tile / 2 >= smallest and gb_at_tile * 1.5 * 2 ** 30 > free:         # 1.5: the driver's own use, fragmentation
        tile //= 2
        gb_at_tile /= 4
    return tile
