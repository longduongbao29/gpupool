import functools
import http.server
import threading
from pathlib import Path

import gguf
import numpy as np
import pytest

from gpupool.scheduler.estimate import total_need_mb
from gpupool.scheduler.gguf_meta import read_meta, read_meta_parts

MODEL = Path(__file__).resolve().parents[1] / ".cache" / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="real model not downloaded")


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


@needs_model
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


@needs_model
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


@needs_model
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


@needs_model
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


# ---- split GGUF (synthetic files: no real model needed) -------------------------------

N_LAYERS = 4


def _tensors():
    """(name, rows) of a tiny 4-layer model, f32 so sizes are rows * 8 * 4 bytes."""
    t = [("token_embd.weight", 5), ("output_norm.weight", 1), ("output.weight", 6)]
    for i in range(N_LAYERS):
        t += [(f"blk.{i}.attn_q.weight", 2 + i), (f"blk.{i}.ffn_up.weight", 3)]
    return t


def _write(path, tensors, *, split: tuple[int, int, int] | None = None, model_kv: bool = True):
    w = gguf.GGUFWriter(str(path), "llama")
    if model_kv:
        w.add_block_count(N_LAYERS)
        w.add_embedding_length(8)
        w.add_head_count(2)
        w.add_head_count_kv(1)
    if split:
        no, count, total = split
        w.add_uint16("split.no", no)
        w.add_uint16("split.count", count)
        w.add_int32("split.tensors.count", total)
    for name, rows in tensors:
        w.add_tensor(name, np.zeros((rows, 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def _split_files(tmp_path):
    t = _tensors()
    whole = _write(tmp_path / "whole.gguf", t)
    # gguf-split layout: part 1 carries the model KV, later parts only the split KV
    cuts = [t[:5], t[5:8], t[8:]]
    parts = [
        _write(tmp_path / f"m-0000{i + 1}-of-00003.gguf", c, split=(i, 3, len(t)), model_kv=i == 0)
        for i, c in enumerate(cuts)
    ]
    return whole, parts


def test_read_meta_parts_equals_unsplit(tmp_path):
    whole, parts = _split_files(tmp_path)
    ref, m = read_meta(whole), read_meta_parts(parts)
    assert (m.arch, m.n_layers, m.n_embd, m.n_head, m.n_head_kv) == (
        ref.arch, ref.n_layers, ref.n_embd, ref.n_head, ref.n_head_kv)
    assert m.layer_bytes == ref.layer_bytes and sum(m.layer_bytes) > 0
    assert m.other_bytes == ref.other_bytes
    assert m.output_bytes == ref.output_bytes
    assert m.file_bytes == sum(Path(p).stat().st_size for p in parts)


def test_read_meta_rejects_first_part_of_split(tmp_path):
    _, parts = _split_files(tmp_path)
    with pytest.raises(ValueError, match="read_meta_parts"):
        read_meta(parts[0])


def test_read_meta_parts_part_count_mismatch(tmp_path):
    _, parts = _split_files(tmp_path)
    with pytest.raises(ValueError, match="3 parts"):
        read_meta_parts(parts[:2])


def test_read_meta_parts_single_unsplit_file_ok(tmp_path):
    whole, _ = _split_files(tmp_path)
    assert read_meta_parts([whole]).layer_bytes == read_meta(whole).layer_bytes
