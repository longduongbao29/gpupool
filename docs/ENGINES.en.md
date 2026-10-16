# Engines: llama.cpp, vLLM, SGLang

> English. Tiếng Việt: [ENGINES.vi.md](ENGINES.vi.md)

gpupool started as a control plane for llama.cpp. A model now names the **engine** that serves it:

| | llama.cpp (default) | vLLM | SGLang |
| --- | --- | --- | --- |
| Model format | GGUF from the library | Hugging Face repo (safetensors, AWQ, GPTQ, FP8, ...) | Hugging Face repo |
| Spans | GPUs and servers (layer split over RPC), CPU devices | GPUs of **one** server (tensor parallel 1, 2, 4, 8) | GPUs of **one** server (tensor parallel) |
| Vision-language models | needs an mmproj file (not managed by gpupool yet) | yes, as vLLM supports them | yes, as SGLang supports them |
| Model download | coordinator library, then the head node | the engine itself, into the agent's Hugging Face cache | the engine itself |
| Memory estimate | exact layer layout from the GGUF header, self-calibrated | `config.json` + file sizes (coarser, see below) | same as vLLM |
| llama.cpp settings (KV cache type, speculative decoding, flash attention, batching, reasoning) | yes | no: pass the engine's own flags | no: pass the engine's own flags |

Everything else is shared: placement scoring, priorities and preemption, autoscaling and scale-to-zero, rebalancing,
per-model GPU limits, the router (one OpenAI-compatible URL), events and the UI.

Pick vLLM or SGLang for models llama.cpp cannot serve well: vision-language models (Qwen2.5-VL, Gemma 3, Llama 3.2
Vision, InternVL...), new architectures, AWQ/GPTQ/FP8 checkpoints, embedding and reranking models, or high-throughput
serving of a model that fits on one server. Keep llama.cpp to pool memory across servers or to use quantized GGUFs.

## Setting up a GPU server

The agent does not install vLLM or SGLang: it runs a command when it resolves on that server, and reports the engine
to the coordinator (`engine:vllm`, `engine:sglang` in the server's features). Models for that engine are only placed on
servers that report it.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPUPOOL_VLLM_CMD` | `vllm serve` | Base command of vLLM. Any wrapper works: `/opt/vllm/bin/vllm serve`, `uvx vllm serve`, a script that runs a container. |
| `GPUPOOL_SGLANG_CMD` | `python3 -m sglang.launch_server` | Base command of SGLang. For `python -m ...`, the interpreter must have the package. |
| `GPUPOOL_HF_HOME` | `<cache_dir>/hf` | Where the engines download models (`HF_HOME`). A `HF_HOME` already set in the agent's environment wins. |
| `HF_TOKEN` | | Set on the agent for gated or private repos (the engine downloads them). Set it on the coordinator too: it reads `config.json` to plan. |
| `GPUPOOL_ALLOW_ENGINE_ARGS` | `0` | `1` accepts a model's extra engine flags. Off by default: the cluster token alone must not pass arbitrary flags to an engine. |

Example, vLLM in a virtualenv next to the agent:

```bash
python3 -m venv /opt/vllm && /opt/vllm/bin/pip install vllm
GPUPOOL_VLLM_CMD="/opt/vllm/bin/vllm serve" gpupool agent --llama-dir /opt/llama --join "http://coordinator:8080#<token>"
```

The agent image ships llama.cpp only (vLLM and SGLang bring their own CUDA, PyTorch and several GB of wheels): install
the engine in the agent's environment or point the command at a wrapper.

## Adding a model

UI: **New model**, then **Engine**: vLLM or SGLang, and the **Hugging Face repo** (`owner/name`, optionally
`@revision`). **Tensor parallel** stays on Auto (the fewest GPUs that fit) unless the model must be spread wider.

API (see [API.en.md](API.en.md)):

```bash
curl -X PUT http://coordinator:8080/api/models/qwen-vl -H "Authorization: Bearer $ADMIN" \
  -H 'content-type: application/json' -d '{
    "engine": "vllm", "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
    "ctx_size": 32768, "parallel": 8, "tensor_parallel": null,
    "trust_remote_code": false, "engine_args": []}'
curl -X POST http://coordinator:8080/api/models/qwen-vl/start -H "Authorization: Bearer $ADMIN"
```

`ctx_size` is the maximum length of one request (`--max-model-len` / `--context-length`) and the size of the KV cache
the estimate reserves; the engines share that cache between requests (paged), so there is no division by `parallel`.
`parallel` is the number of requests served at once (`--max-num-seqs` / `--max-running-requests`).

The coordinator reads `config.json` and the file list of the repo when the model is saved: a typo, a gated repo
without a token, or a `tensor_parallel` that does not divide the attention heads is refused right away (422).

Images go in the chat messages, as with OpenAI:

```json
{"model": "qwen-vl", "messages": [{"role": "user", "content": [
  {"type": "text", "text": "What is in this picture?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}
```

The router forwards `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/rerank` and
`/v1/score` to any engine; what a model answers depends on the engine and the model. It adds llama.cpp's
`cache_prompt` only to llama.cpp replicas.

## How placement works

A replica of a vLLM or SGLang model runs as one process on one server, with `k` of its GPUs (`k` in 1, 2, 4, 8 and
dividing the model's attention heads). Each GPU needs:

- weights / k (the repo's `model*.safetensors`, else all top-level safetensors, else `pytorch_model*.bin`);
- KV cache / k at `ctx_size` tokens with a bf16 cache (MLA models: the whole latent cache on every GPU; sliding-window
  and linear-attention layers from `layer_types`);
- activations for a 2048-token chunk plus 1 GB of runtime (CUDA context, NCCL, CUDA graphs).

The engines size their memory as a fraction of each GPU's **total** memory (`--gpu-memory-utilization`,
`--mem-fraction-static`). gpupool computes that fraction from the estimate, so the engine never takes memory other
users hold, and checks the result against what each GPU has free (a bigger GPU in the set takes more for the same
fraction). Among the servers that fit, the usual score decides (speed, sharing, spread, waste); without a set
`tensor_parallel`, each server offers its smallest `k` that fits. The estimate is coarser than llama.cpp's, and
memory self-calibration does not apply: if a model fails to start for lack of memory, lower `ctx_size` or
`parallel`, or set `tensor_parallel`.

## Limits of this version

- One server per replica: vLLM / SGLang multi-node (Ray, NCCL across hosts) is not orchestrated.
- No llama.cpp memory calibration or speed learning for these engines; the scheduler's speed estimate is
  bandwidth-based and only compares candidates of the same model.
- llama.cpp vision models (`--mmproj`) are not managed yet.
- Engine flags are free text: gpupool does not check them, and a flag that changes memory use
  (`--kv-cache-dtype fp8`, `--max-num-batched-tokens`...) is not reflected in the estimate.
