"""Real per-device buffer sizes from a llama-server log (needs `-lv 4`).

The scheduler's estimate is only calibrated on small models; these numbers are what llama.cpp
actually allocated, so the coordinator can compare them with the estimate and correct itself.
"""
from __future__ import annotations

import re

# e.g. "0.01.024.240 I llama_kv_cache:      CUDA0 KV buffer size =    24.00 MiB"
#      "0.01.162.453 I load_tensors: RPC0[127.0.0.1:9555] model buffer size =   244.33 MiB"
_LINE = re.compile(
    r"(?P<dev>\S+)\s+(?P<kind>model|KV|compute) buffer size\s*=\s*(?P<mib>\d+(?:\.\d+)?)\s*MiB")
_KIND = {"model": "model_mb", "KV": "kv_mb", "compute": "compute_mb"}


def _is_host(name: str) -> bool:
    # CPU, CPU_Mapped, CUDA_Host, ...: system RAM, not device memory
    return name == "CPU" or name.endswith(("_Host", "_Mapped"))


def parse_buffers(text: str) -> dict[str, dict[str, float]]:
    """Per device (CUDA0, RPC0, ...): model/kv/compute/total MiB. The last occurrence of each
    (device, kind) wins: a context is sometimes reserved again, and the final size is live."""
    out: dict[str, dict[str, float]] = {}
    for m in _LINE.finditer(text):
        name = m["dev"].split("[", 1)[0]  # RPC0[host:port] -> RPC0
        if not name or _is_host(name):
            continue
        out.setdefault(name, {})[_KIND[m["kind"]]] = float(m["mib"])
    res = {}
    for name, d in out.items():
        full = {k: d.get(k, 0.0) for k in ("model_mb", "kv_mb", "compute_mb")}
        full["total_mb"] = round(sum(full.values()), 2)
        res[name] = full
    return res
