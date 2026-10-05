"""CUDA compute capabilities: what a GPU generation can do and whether a llama.cpp build has kernels
for it. Pure functions shared by the agent (blocks GPUs it cannot run on) and the coordinator
(tuning advice, UI)."""
from __future__ import annotations

from typing import Literal

KernelSupport = Literal["native", "compatible", "jit", "missing", "unknown"]

# (major, minor) lower bound -> generation, newest first.
_ARCH_NAMES = [((12, 0), "Blackwell"), ((10, 0), "Blackwell"), ((9, 0), "Hopper"), ((8, 9), "Ada"),
               ((8, 0), "Ampere"), ((7, 5), "Turing"), ((7, 0), "Volta"), ((6, 0), "Pascal"),
               ((5, 0), "Maxwell"), ((3, 0), "Kepler")]


def parse_cc(cc: str | None) -> tuple[int, int] | None:
    """"8.6" -> (8, 6); also accepts the CMake spelling "86" / "120" / "120a"."""
    if not cc:
        return None
    s = cc.strip().rstrip("af")
    try:
        if "." in s:
            a, b = s.split(".", 1)
            return int(a), int(b)
        n = int(s)
        return n // 10, n % 10
    except ValueError:
        return None


def arch_name(cc: str | None) -> str | None:
    v = parse_cc(cc)
    if v is None:
        return None
    return next((name for low, name in _ARCH_NAMES if v >= low), None)


def has_tensor_cores(cc: str | None) -> bool | None:
    """Volta (7.0) and newer. None when unknown."""
    v = parse_cc(cc)
    return None if v is None else v >= (7, 0)


def kernel_support(cc: str | None, archs: list[str] | None) -> KernelSupport:
    """Whether a build with SASS for `archs` (CMake numbers, e.g. ["61", "86"]) runs on a GPU of `cc`.

    native: built for exactly this GPU. compatible: built for an older minor of the same major (SASS
    is binary compatible within a major). jit: only older majors; the driver compiles their PTX at
    first load (slow start, and newer-generation instructions go unused). missing: every build
    target is newer than the GPU, so CUDA fails with "no kernel image is available"."""
    v = parse_cc(cc)
    built = [b for b in (parse_cc(a) for a in (archs or [])) if b is not None]
    if v is None or not built:
        return "unknown"
    if v in built:
        return "native"
    if any(b[0] == v[0] and b[1] <= v[1] for b in built):
        return "compatible"
    if any(b < v for b in built):
        return "jit"
    return "missing"
