import functools
import http.server
import threading
from pathlib import Path

import gguf
import pytest

from gpupool.scheduler.estimate import total_need_mb
from gpupool.scheduler.gguf_meta import read_meta

MODEL = Path(__file__).resolve().parents[1] / ".cache" / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="real model not downloaded")


def _reference():
    r = gguf.GGUFReader(str(MODEL))
    layers: dict[int, int] = {}
    other = 0
    for t in r.tensors:
        if t.name.startswith("blk."):
            i = int(t.name.split(".")[1])
            layers[i] = layers.get(i, 0) + int(t.n_bytes)
        else:
            other += int(t.n_bytes)
    return layers, other


def test_real_file_matches_gguf_reader():
    m = read_meta(str(MODEL))
    layers, other = _reference()
    assert (m.arch, m.n_layers, m.n_embd, m.n_head, m.n_head_kv) == ("qwen2", 24, 896, 14, 2)
    assert m.head_dim == 896 // 14
    assert m.layer_bytes == [layers[i] for i in range(24)]
    assert m.other_bytes == other
    assert sum(m.layer_bytes) / 2**20 == pytest.approx(235.76, abs=0.01)
    assert m.other_bytes / 2**20 == pytest.approx(227.20, abs=0.01)
    assert m.file_bytes == MODEL.stat().st_size
    # output = output.weight (+ output_norm); token_embd stays in host RAM
    r = gguf.GGUFReader(str(MODEL))
    sizes = {t.name: int(t.n_bytes) for t in r.tensors}
    assert m.output_bytes == sizes["output.weight"] + sizes["output_norm.weight"]
    assert total_need_mb(m, 4096) > 0


class _Counting(http.server.SimpleHTTPRequestHandler):
    sent = 0

    def log_message(self, *a):
        pass

    def copyfile(self, source, outputfile):
        try:
            while chunk := source.read(64 * 1024):
                outputfile.write(chunk)
                type(self).sent += len(chunk)
        except (BrokenPipeError, ConnectionError, OSError):
            pass


def test_url_reads_only_header():
    _Counting.sent = 0
    handler = functools.partial(_Counting, directory=str(MODEL.parent))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/{MODEL.name}"
        m = read_meta(url)
    finally:
        srv.shutdown()
        srv.server_close()
    assert m.file_bytes == MODEL.stat().st_size
    assert m.layer_bytes == read_meta(str(MODEL)).layer_bytes
    assert _Counting.sent < MODEL.stat().st_size // 10


def test_truncated_and_garbage(tmp_path):
    data = MODEL.read_bytes()[:200_000]
    p = tmp_path / "trunc.gguf"
    p.write_bytes(data)
    with pytest.raises(ValueError, match="truncated"):
        read_meta(str(p))
    g = tmp_path / "garbage.gguf"
    g.write_bytes(b"this is not a gguf file at all" * 10)
    with pytest.raises(ValueError, match="magic"):
        read_meta(str(g))
    e = tmp_path / "empty.gguf"
    e.write_bytes(b"")
    with pytest.raises(ValueError):
        read_meta(str(e))


def test_estimate_matches_measured_llama_cpp_buffers():
    # llama.cpp b11342 on a GTX 1650, Qwen2.5-0.5B Q4_K_M, ctx 4096, verbose load log:
    #   CUDA0 model buffer 373.73 MiB, KV 48.00 MiB, compute 37.76 MiB.
    # The VRAM drop seen by NVML after load was 525 MB (incl. CUDA context).
    from gpupool.scheduler.estimate import compute_mb, device_need_mb, kv_bytes_per_layer
    meta = read_meta(str(MODEL))
    weights_mib = (sum(meta.layer_bytes) + meta.output_bytes) / 2**20
    assert abs(weights_mib - 373.73) < 0.5
    assert kv_bytes_per_layer(meta, 4096) * meta.n_layers / 2**20 == 48.0
    assert abs(compute_mb(meta) - 37.76) < 2
    need = device_need_mb(meta, range(meta.n_layers), 4096, "cuda", True)
    assert 525 <= need <= 525 * 1.2  # never under, at most 20% over
