import json
import struct
from pathlib import Path

import httpx
import pytest
import respx

from gpupool.converter.models import ClusterVram, ConvertError, SourceFile, SourceSpec
from gpupool.converter.source import (
    HfClient, inspect_source, local_files, select_files,
)

BASE = "https://hf.test"
CLUSTER = ClusterVram(largest_gpu_mb=24000, pool_mb=48000)


def sf(*names, size=10):
    return [SourceFile(name=n, bytes=size) for n in names]


def write_st(path: Path, tensors: dict[str, list[int]]):
    header = {"__metadata__": {"format": "pt"}}
    off = 0
    for k, shape in tensors.items():
        n = 1
        for d in shape:
            n *= d
        header[k] = {"dtype": "F16", "shape": shape, "data_offsets": [off, off + 2 * n]}
        off += 2 * n
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * 16)  # data is never read


@pytest.fixture
async def hf():
    async with httpx.AsyncClient() as c:
        yield HfClient(c, token="tok", base=BASE)


# ---- select_files --------------------------------------------------------------------------

def test_select_files_whitelist():
    files = sf("config.json", "tokenizer.json", "model-00001-of-00002.safetensors",
               "model.safetensors.index.json", "pytorch_model.bin", "model.gguf", "modeling.py",
               "onnx/model.onnx", "original/consolidated.00.pth", "onnx/config.json",
               ".git/config.json", "sub/tokenizer.model", "tek.tiktoken", "README.md",
               "consolidated.00.pth", "tf_model.h5", "flax_model.msgpack")
    keep, skipped = select_files(files)
    names = [f.name for f in keep]
    assert names == ["config.json", "tokenizer.json", "model-00001-of-00002.safetensors",
                     "model.safetensors.index.json", "sub/tokenizer.model", "tek.tiktoken"]
    assert "pytorch_model.bin" in skipped and "modeling.py" in skipped and "onnx/config.json" in skipped
    assert ".git/config.json" in skipped and "model.gguf" in skipped


def test_select_files_bin_only_without_safetensors_and_remote_code():
    files = sf("config.json", "pytorch_model-00001-of-00002.bin", "pytorch_model.bin.index.json", "tok.py")
    keep, skipped = select_files(files)
    assert [f.name for f in keep] == ["config.json", "pytorch_model-00001-of-00002.bin", "pytorch_model.bin.index.json"]
    assert skipped == ["tok.py"]
    keep, _ = select_files(files, allow_remote_code=True)
    assert "tok.py" in [f.name for f in keep]


# ---- HfClient ------------------------------------------------------------------------------

@respx.mock
async def test_list_files_files_only_with_pagination_and_auth(hf):
    route = respx.get(f"{BASE}/api/models/o/n/tree/main", params={"recursive": "true"}).mock(
        return_value=httpx.Response(200, json=[{"type": "directory", "path": "d", "size": 0},
                                               {"type": "file", "path": "a.json", "size": 5}],
                                    headers={"Link": f'<{BASE}/api/models/o/n/tree/main?cursor=x>; rel="next"'}))
    respx.get(f"{BASE}/api/models/o/n/tree/main", params={"cursor": "x"}).mock(
        return_value=httpx.Response(200, json=[{"type": "file", "path": "d/b.safetensors", "size": 7}]))
    files = await hf.list_files("o/n")
    assert [(f.name, f.bytes) for f in files] == [("a.json", 5), ("d/b.safetensors", 7)]
    assert route.calls[0].request.headers["authorization"] == "Bearer tok"


@respx.mock
@pytest.mark.parametrize("code,status", [(401, 403), (403, 403), (404, 404), (500, 502)])
async def test_status_mapping(hf, code, status):
    respx.get(f"{BASE}/api/models/o/n/revision/main").mock(return_value=httpx.Response(code))
    with pytest.raises(ConvertError) as e:
        await hf.model_info("o/n")
    assert e.value.status == status
    if status == 403:
        assert "HF_TOKEN" in e.value.message


@respx.mock
async def test_network_error_is_502(hf):
    respx.get(f"{BASE}/api/models/o/n/revision/main").mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(ConvertError) as e:
        await hf.model_info("o/n")
    assert e.value.status == 502


async def test_bad_repo_id(hf):
    with pytest.raises(ConvertError) as e:
        await hf.list_files("../etc")
    assert e.value.status == 400


@respx.mock
async def test_gguf_alternatives_never_raises(hf):
    route = respx.get(f"{BASE}/api/models").mock(
        return_value=httpx.Response(200, json=[{"id": "a/x-GGUF"}, {"id": "b/x-GGUF"}]))
    assert await hf.gguf_alternatives("o/n", limit=5) == ["a/x-GGUF", "b/x-GGUF"]
    q = route.calls[0].request.url.params.get_list("filter")
    assert q == ["base_model:quantized:o/n", "gguf"]
    respx.get(f"{BASE}/api/models").mock(side_effect=httpx.ConnectError("x"))
    assert await hf.gguf_alternatives("o/n") == []
    respx.get(f"{BASE}/api/models").mock(return_value=httpx.Response(500))
    assert await hf.gguf_alternatives("o/n") == []


@respx.mock
async def test_download_ok_progress_and_cache_hit(hf, tmp_path):
    body = b"x" * 3000
    route = respx.get(f"{BASE}/o/n/resolve/main/sub%20dir/w.safetensors").mock(
        return_value=httpx.Response(200, content=body))
    seen: list[int] = []
    dest = tmp_path / "w.safetensors"
    assert await hf.download("o/n", "main", "sub dir/w.safetensors", dest, 3000, seen.append) == 3000
    assert dest.read_bytes() == body and seen[-1] == 3000 and not Path(str(dest) + ".part").exists()
    # cache hit: no second request
    assert await hf.download("o/n", "main", "sub dir/w.safetensors", dest, 3000, seen.append) == 3000
    assert route.call_count == 1


@respx.mock
async def test_download_truncated_leaves_no_files(hf, tmp_path):
    respx.get(f"{BASE}/o/n/resolve/main/w.bin").mock(return_value=httpx.Response(200, content=b"x" * 10))
    dest = tmp_path / "w.bin"
    with pytest.raises(ConvertError) as e:
        await hf.download("o/n", "main", "w.bin", dest, 99, lambda n: None)
    assert e.value.status == 502 and "truncated" in e.value.message
    assert list(tmp_path.iterdir()) == []


@respx.mock
async def test_download_stale_cache_is_replaced_and_403(hf, tmp_path):
    dest = tmp_path / "f.json"
    dest.write_bytes(b"old")
    respx.get(f"{BASE}/o/n/resolve/main/f.json").mock(return_value=httpx.Response(200, content=b"newer"))
    assert await hf.download("o/n", "main", "f.json", dest, 5, lambda n: None) == 5
    assert dest.read_bytes() == b"newer"
    respx.get(f"{BASE}/o/n/resolve/main/g.json").mock(return_value=httpx.Response(403))
    with pytest.raises(ConvertError) as e:
        await hf.download("o/n", "main", "g.json", tmp_path / "g.json", None, lambda n: None)
    assert e.value.status == 403 and not (tmp_path / "g.json.part").exists()


# ---- local folders -------------------------------------------------------------------------

def test_local_files_skips_symlinked_dirs(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.json").write_text("{}")
    outside = tmp_path.parent / (tmp_path.name + "_out")
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    try:
        (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted")
    assert [f.name for f in local_files(tmp_path)] == ["a/x.json"]


def _model_dir(tmp_path: Path, config: dict, shards=({"w": [10, 20], "b": [20]},)):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    for i, t in enumerate(shards):
        write_st(d / f"model-{i}.safetensors", t)
    return d


CFG = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "num_hidden_layers": 4,
       "num_attention_heads": 8, "hidden_size": 64, "max_position_embeddings": 2048}


async def test_inspect_path_reads_safetensors_headers(tmp_path, hf):
    d = _model_dir(tmp_path, CFG, shards=({"w": [10, 20], "b": [20]}, {"c": [5, 5]}))
    (d / "tok.py").write_text("print(1)")
    r = await inspect_source(SourceSpec(path="/x"), hf=hf, locate_dir=lambda p: d, cluster=CLUSTER,
                             supported_architectures={"LlamaForCausalLM"})
    assert r.params == 200 + 20 + 25
    assert (r.architecture, r.model_type, r.n_layers, r.context_length) == ("LlamaForCausalLM", "llama", 4, 2048)
    assert r.supported is True and r.weight_format == "safetensors" and r.remote_code is True
    assert "tok.py" in r.skipped and r.gated is False
    assert len(r.options) == 24 and sum(o.recommended for o in r.options) == 1
    assert r.recommended == [o.type for o in r.options if o.recommended][0]
    assert r.options[0].est_bytes is not None and r.options[0].fits_single_gpu is True


async def test_inspect_path_errors(tmp_path, hf):
    class LibErr(Exception):
        def __init__(self):
            self.message, self.status = "outside the allowed roots", 403

    def boom(p):
        raise LibErr()

    with pytest.raises(ConvertError) as e:
        await inspect_source(SourceSpec(path="/x"), hf=hf, locate_dir=boom, cluster=CLUSTER,
                             supported_architectures=None)
    assert (e.value.status, e.value.message) == (403, "outside the allowed roots")
    empty = tmp_path / "e"
    empty.mkdir()
    with pytest.raises(ConvertError) as e:
        await inspect_source(SourceSpec(path="/x"), hf=hf, locate_dir=lambda p: empty, cluster=CLUSTER,
                             supported_architectures=None)
    assert e.value.status == 422 and "config.json" in e.value.message


async def test_inspect_unsupported_prequant_and_no_toolchain(tmp_path, hf):
    cfg = {**CFG, "quantization_config": {"quant_method": "awq"}}
    d = _model_dir(tmp_path, cfg)
    r = await inspect_source(SourceSpec(path="/x"), hf=hf, locate_dir=lambda p: d, cluster=CLUSTER,
                             supported_architectures=None)
    assert r.prequantized == "awq" and r.prequant_supported is False and r.supported is None
    assert any("toolchain" in w for w in r.warnings) and any("quantized" in w for w in r.warnings)
    cfg["quantization_config"]["quant_method"] = "fp8"
    (d / "config.json").write_text(json.dumps(cfg))
    r = await inspect_source(SourceSpec(path="/x"), hf=hf, locate_dir=lambda p: d, cluster=CLUSTER,
                             supported_architectures={"Other"})
    assert r.prequant_supported is True and r.supported is False


async def test_inspect_no_cluster_info_and_bin_estimate(tmp_path, hf):
    d = tmp_path / "b"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(CFG))
    (d / "pytorch_model.bin").write_bytes(b"\0" * 2000)
    r = await inspect_source(SourceSpec(path="/x"), hf=hf, locate_dir=lambda p: d,
                             cluster=ClusterVram(), supported_architectures=set())
    assert r.params == 1000 and r.weight_format == "pytorch_bin"
    assert any("estimated" in w for w in r.warnings)
    assert all(o.fits_single_gpu is None and o.fits_pool is None for o in r.options)
    assert r.recommended == "Q8_0"  # < 3B, size rule


@respx.mock
async def test_inspect_hf(hf):
    respx.get(f"{BASE}/api/models/o/n/revision/main").mock(return_value=httpx.Response(200, json={
        "gated": "manual", "safetensors": {"total": 7_000_000_000, "parameters": {"BF16": 7_000_000_000}},
        "cardData": {"base_model": ["org/base"]}}))
    respx.get(f"{BASE}/api/models/o/n/tree/main").mock(return_value=httpx.Response(200, json=[
        {"type": "file", "path": "config.json", "size": 100},
        {"type": "file", "path": "model.safetensors", "size": 14_000_000_000},
        {"type": "file", "path": "README.md", "size": 3}]))
    respx.get(f"{BASE}/o/n/resolve/main/config.json").mock(return_value=httpx.Response(200, json=CFG))
    respx.get(f"{BASE}/api/models").mock(return_value=httpx.Response(200, json=[{"id": "q/n-GGUF"}]))
    r = await inspect_source(SourceSpec(hf_repo="o/n"), hf=hf, locate_dir=lambda p: Path(p),
                             cluster=CLUSTER, supported_architectures={"LlamaForCausalLM"})
    assert r.params == 7_000_000_000 and r.gated is True and r.base_model == "org/base"
    assert r.source_bytes == 14_000_000_100 and r.skipped == ["README.md"]
    assert r.gguf_alternatives == ["q/n-GGUF"] and r.weight_format == "safetensors"


@respx.mock
async def test_inspect_hf_without_config_is_422(hf):
    respx.get(f"{BASE}/api/models/o/n/revision/main").mock(return_value=httpx.Response(200, json={}))
    respx.get(f"{BASE}/api/models/o/n/tree/main").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{BASE}/o/n/resolve/main/config.json").mock(return_value=httpx.Response(404))
    with pytest.raises(ConvertError) as e:
        await inspect_source(SourceSpec(hf_repo="o/n"), hf=hf, locate_dir=lambda p: Path(p),
                             cluster=CLUSTER, supported_architectures=set())
    assert e.value.status == 422 and "not a transformers model repo" in e.value.message


@pytest.mark.real
async def test_real_smollm2():
    async with httpx.AsyncClient(timeout=30) as c:
        r = await inspect_source(
            SourceSpec(hf_repo="HuggingFaceTB/SmolLM2-135M-Instruct"), hf=HfClient(c),
            locate_dir=lambda p: Path(p), cluster=CLUSTER, supported_architectures={"LlamaForCausalLM"})
    assert r.architecture == "LlamaForCausalLM" and r.weight_format == "safetensors"
    assert 134_000_000 < r.params < 135_000_000
    names = {f.name for f in r.files}
    assert {"model.safetensors", "tokenizer.json", "config.json"} <= names


def test_select_files_skips_mistral_consolidated_copy_and_unsafe_names():
    from gpupool.converter.source import select_files
    files = [SourceFile(name=n, bytes=1) for n in (
        "config.json", "consolidated.safetensors", "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors", "model.safetensors.index.json",
        "../evil.safetensors", "a/../../b.safetensors", "/abs.safetensors", "dir\\x.safetensors",
        "c:x.safetensors", "a//b.safetensors")]
    keep, skipped = select_files(files)
    assert [f.name for f in keep] == ["config.json", "model-00001-of-00002.safetensors",
                                      "model-00002-of-00002.safetensors", "model.safetensors.index.json"]
    assert "consolidated.safetensors" in skipped and "../evil.safetensors" in skipped
