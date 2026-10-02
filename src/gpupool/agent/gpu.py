"""Device probing: NVML for CUDA, psutil for CPU. Never raises."""
from __future__ import annotations

import json
import logging
import os
import threading

import psutil

from gpupool.common.config import AgentConfig
from gpupool.common.models import Device, GpuProcess

log = logging.getLogger(__name__)

_MB = 1024 * 1024


def _usable(cfg: AgentConfig, device_id: str, kind: str, total_mb: int, free_mb: int) -> int:
    if kind == "cpu":
        usable = max(0, free_mb - cfg.margin_min_mb)
    else:
        margin = max(cfg.margin_min_mb, int(total_mb * cfg.margin_pct))
        usable = max(0, free_mb - margin)
    if device_id in cfg.budget_mb:
        usable = min(usable, max(0, cfg.budget_mb[device_id]))
    return usable


def _fake_devices(cfg: AgentConfig, raw: str) -> list[Device]:
    out: list[Device] = []
    for d in json.loads(raw):
        kind = d.get("kind", "cuda")
        total, free = int(d["total_mb"]), int(d["free_mb"])
        out.append(Device(
            device_id=d["device_id"], kind=kind, name=d.get("name", d["device_id"]),
            total_mb=total, free_mb=free,
            usable_mb=_usable(cfg, d["device_id"], kind, total, free),
            budget_mb=cfg.budget_mb.get(d["device_id"]),
            util_pct=d.get("util_pct"),
            temp_c=d.get("temp_c"), power_w=d.get("power_w"),
            processes=[GpuProcess(**p) for p in d.get("processes", [])],
            driver=d.get("driver"), cuda=d.get("cuda"),
            uuid=d.get("uuid"), pci_bus_id=d.get("pci_bus_id"),
            bandwidth_gbps=d.get("bandwidth_gbps"),
        ))
    return out


def _best_effort(fn, default=None):
    """Call an NVML getter; any failure (NOT_SUPPORTED on laptops/WSL, ...) yields default."""
    try:
        return fn()
    except Exception:
        return default


def _text(v) -> str | None:
    if v is None:
        return None
    return v.decode() if isinstance(v, bytes) else str(v)


def _bandwidth_gbps(pynvml, h) -> float | None:
    """Peak memory bandwidth in GB/s: bus width (bits) x max memory clock (MHz) x 2 (DDR) / 8."""
    try:
        width = int(pynvml.nvmlDeviceGetMemoryBusWidth(h))
        clock = int(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM))
        if width <= 0 or clock <= 0:
            return None
        return round(width * clock * 2 / 8 / 1000, 1)
    except Exception:
        return None


def _cuda_version(raw) -> str | None:
    # NVML encodes 12020 as 12.2 (major*1000 + minor*10).
    if not isinstance(raw, int) or raw <= 0:
        return None
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def _proc_name(pid: int) -> str:
    try:
        return psutil.Process(pid).name()
    except Exception:  # other users' processes are often hidden; also NoSuchProcess
        return ""


def _gpu_processes(pynvml, h) -> list[GpuProcess]:
    na = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
    by_pid: dict[int, GpuProcess] = {}
    for getter in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
        fn = getattr(pynvml, getter, None)
        if fn is None:
            continue
        for p in _best_effort(lambda: fn(h), []) or []:
            try:
                pid = int(p.pid)
                if pid in by_pid:
                    continue
                used = p.usedGpuMemory
                used_mb = None if used is None or used == na else int(used // _MB)
                by_pid[pid] = GpuProcess(pid=pid, name=_proc_name(pid), used_mb=used_mb)
            except Exception:
                continue
    return list(by_pid.values())


_nvml_lock = threading.Lock()
_nvml_mod = None  # the pynvml module we initialised; None = not initialised


def _nvml_init(pynvml) -> None:
    """Init NVML once and keep it: init/shutdown on every 2 s probe is pure overhead."""
    global _nvml_mod
    with _nvml_lock:
        if _nvml_mod is not pynvml:
            pynvml.nvmlInit()  # raises when unusable; the flag stays unset so we retry
            _nvml_mod = pynvml


def _nvml_reset(pynvml) -> None:
    global _nvml_mod
    with _nvml_lock:
        if _nvml_mod is pynvml:
            _nvml_mod = None
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# NVML_ERROR_UNINITIALIZED, DRIVER_NOT_LOADED, LIB_RM_VERSION_MISMATCH: the library (or the
# driver under it) changed or went away, so a fresh nvmlInit is needed. Per-device errors
# (GPU_IS_LOST, NOT_SUPPORTED) say nothing about the library and must not trigger it.
_NVML_LIBRARY_ERRORS = {1, 9, 18}


def _note_nvml_error(pynvml, e: Exception) -> None:
    if (type(e).__name__ == "NVMLError_Uninitialized"
            or getattr(e, "value", None) in _NVML_LIBRARY_ERRORS):
        log.warning("NVML needs re-initialisation (%s)", e)
        _nvml_reset(pynvml)


def _cuda_devices(cfg: AgentConfig) -> list[Device]:
    import pynvml

    _nvml_init(pynvml)
    try:
        driver = _text(_best_effort(pynvml.nvmlSystemGetDriverVersion))
        cuda = _cuda_version(_best_effort(pynvml.nvmlSystemGetCudaDriverVersion_v2))
        found = []
        lost = False
        for i in range(pynvml.nvmlDeviceGetCount()):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                pci = pynvml.nvmlDeviceGetPciInfo(h).busId
            except Exception as e:
                # A GPU that fell off the bus still counts but every call fails (GPU_IS_LOST).
                # Skip only that GPU: raising here would report the node with no GPUs at all
                # and fail replicas on its healthy GPUs too.
                log.warning("NVML device %d unavailable, not reported: %s", i, e)
                _note_nvml_error(pynvml, e)
                lost = True
                continue
            if isinstance(pci, bytes):
                pci = pci.decode()
            found.append((str(pci), h))
        # CUDA_DEVICE_ORDER=PCI_BUS_ID is set for every engine we spawn, so
        # "CUDA<i>" in llama.cpp == i-th device in PCI order here.
        found.sort(key=lambda t: t[0])
        out = []
        for idx, (pci, h) in enumerate(found):
            try:
                name = pynvml.nvmlDeviceGetName(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            except Exception as e:  # lost between enumeration and query: skip it alone
                log.warning("NVML device %s unavailable, not reported: %s", pci, e)
                _note_nvml_error(pynvml, e)
                lost = True
                continue
            if isinstance(name, bytes):
                name = name.decode()
            total, free = int(mem.total // _MB), int(mem.free // _MB)
            try:
                util = int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                util = None
            temp = _best_effort(
                lambda: int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)))
            power = _best_effort(lambda: int(round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000)))
            # Stable identity across reboots/bus loss; "CUDA<i>" shifts when a card drops.
            uuid = _text(_best_effort(lambda: pynvml.nvmlDeviceGetUUID(h)))
            bandwidth = _bandwidth_gbps(pynvml, h)
            did = f"CUDA{idx}"
            out.append(Device(device_id=did, kind="cuda", name=name, total_mb=total, free_mb=free,
                              usable_mb=_usable(cfg, did, "cuda", total, free),
                              budget_mb=cfg.budget_mb.get(did), util_pct=util,
                              temp_c=temp, power_w=power, processes=_gpu_processes(pynvml, h),
                              driver=driver, cuda=cuda, uuid=uuid, pci_bus_id=pci,
                              bandwidth_gbps=bandwidth))
        if lost:
            # The CUDA runtime behind llama.cpp may or may not still count the lost GPU, so
            # "CUDA<i>" -> physical card is no longer trustworthy: a per-device flag or a new
            # placement could land on the wrong card. Block new placements on the whole node
            # (existing replicas are untouched: the coordinator uses usable_mb only for new ones).
            log.error("a GPU on this node is lost; CUDA device numbering is unreliable, so new "
                      "placements are blocked (usable_mb=0) until the GPU recovers or the node "
                      "is rebooted")
            for d in out:
                d.usable_mb = 0
        return out
    except Exception as e:
        _note_nvml_error(pynvml, e)
        raise


def _cpu_device(cfg: AgentConfig) -> Device:
    vm = psutil.virtual_memory()
    total, free = int(vm.total // _MB), int(vm.available // _MB)
    return Device(device_id="CPU", kind="cpu", name="CPU", total_mb=total, free_mb=free,
                  usable_mb=_usable(cfg, "CPU", "cpu", total, free),
                  budget_mb=cfg.budget_mb.get("CPU"))


def probe_devices(cfg: AgentConfig) -> list[Device]:
    fake = os.environ.get("GPUPOOL_FAKE_DEVICES")
    if fake:
        try:
            return _fake_devices(cfg, fake)
        except Exception as e:  # malformed test hook must not kill the agent
            log.warning("bad GPUPOOL_FAKE_DEVICES: %s", e)
            return []
    devices: list[Device] = []
    try:
        devices.extend(_cuda_devices(cfg))
    except Exception as e:
        log.warning("NVML unavailable (%s); no CUDA devices reported", e)
    if cfg.include_cpu:
        try:
            devices.append(_cpu_device(cfg))
        except Exception as e:
            log.warning("cpu probe failed: %s", e)
    return devices
