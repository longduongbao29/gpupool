"""Device probing: NVML for CUDA, psutil for CPU. Never raises."""
from __future__ import annotations

import json
import logging
import os

import psutil

from gpupool.common.config import AgentConfig
from gpupool.common.models import Device

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
            util_pct=d.get("util_pct"),
        ))
    return out


def _cuda_devices(cfg: AgentConfig) -> list[Device]:
    import pynvml

    pynvml.nvmlInit()
    try:
        found = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            pci = pynvml.nvmlDeviceGetPciInfo(h).busId
            if isinstance(pci, bytes):
                pci = pci.decode()
            found.append((str(pci), h))
        # CUDA_DEVICE_ORDER=PCI_BUS_ID is set for every engine we spawn, so
        # "CUDA<i>" in llama.cpp == i-th device in PCI order here.
        found.sort(key=lambda t: t[0])
        out = []
        for idx, (_, h) in enumerate(found):
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            total, free = int(mem.total // _MB), int(mem.free // _MB)
            try:
                util = int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                util = None
            did = f"CUDA{idx}"
            out.append(Device(device_id=did, kind="cuda", name=name, total_mb=total, free_mb=free,
                              usable_mb=_usable(cfg, did, "cuda", total, free), util_pct=util))
        return out
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _cpu_device(cfg: AgentConfig) -> Device:
    vm = psutil.virtual_memory()
    total, free = int(vm.total // _MB), int(vm.available // _MB)
    return Device(device_id="CPU", kind="cpu", name="CPU", total_mb=total, free_mb=free,
                  usable_mb=_usable(cfg, "CPU", "cpu", total, free))


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
