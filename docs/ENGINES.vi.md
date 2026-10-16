# Engine: llama.cpp, vLLM, SGLang

> Tiếng Việt. English: [ENGINES.en.md](ENGINES.en.md)

Ban đầu gpupool là lớp điều khiển cho llama.cpp. Giờ mỗi model chọn **engine** phục vụ nó:

| | llama.cpp (mặc định) | vLLM | SGLang |
| --- | --- | --- | --- |
| Định dạng model | GGUF trong thư viện | Repo Hugging Face (safetensors, AWQ, GPTQ, FP8...) | Repo Hugging Face |
| Phạm vi | Nhiều GPU và nhiều server (chia layer qua RPC), cả thiết bị CPU | Các GPU của **một** server (tensor parallel 1, 2, 4, 8) | Các GPU của **một** server (tensor parallel) |
| Model thị giác–ngôn ngữ (VLM) | cần file mmproj (gpupool chưa quản lý) | có, theo mức vLLM hỗ trợ | có, theo mức SGLang hỗ trợ |
| Tải model | thư viện của coordinator, rồi tới node head | engine tự tải về cache Hugging Face của agent | engine tự tải |
| Ước lượng bộ nhớ | chính xác theo layout layer trong header GGUF, tự hiệu chỉnh | `config.json` + kích thước file (thô hơn, xem bên dưới) | như vLLM |
| Thiết lập của llama.cpp (kiểu KV cache, speculative decoding, flash attention, batch, reasoning) | có | không: truyền flag riêng của engine | không: truyền flag riêng của engine |

Mọi thứ khác dùng chung: chấm điểm placement, priority và preemption, autoscale và scale-to-zero, rebalance, giới hạn
GPU theo model, router (một URL tương thích OpenAI), sự kiện và UI.

Chọn vLLM hoặc SGLang cho những model llama.cpp phục vụ chưa tốt: VLM (Qwen2.5-VL, Gemma 3, Llama 3.2 Vision,
InternVL...), kiến trúc mới, checkpoint AWQ/GPTQ/FP8, model embedding và rerank, hoặc khi cần thông lượng cao cho một
model vừa một server. Giữ llama.cpp khi cần gộp bộ nhớ nhiều server hoặc dùng GGUF đã lượng tử hóa.

## Chuẩn bị GPU server

### Một container chứa mọi engine

Image `gpupool-agent` có hai bản:

| Tag | Engine | Dung lượng |
| --- | --- | --- |
| `gpupool-agent:<version>` | llama.cpp | ~2 GB |
| `gpupool-agent:<version>-all` | llama.cpp, vLLM, SGLang | ~18 GB (tải về ~6 GB) |

Bản `-all` cài vLLM và SGLang vào hai virtualenv (`/opt/engines/vllm`, `/opt/engines/sglang`: hai engine pin phiên bản
khác nhau cho các package dùng chung) và đặt sẵn `GPUPOOL_VLLM_CMD` / `GPUPOOL_SGLANG_CMD`, nên agent đề xuất cả ba engine
mà không cần cấu hình gì. Chạy giống image thường, thêm `--ipc host` (các worker tensor-parallel trao đổi dữ liệu qua
`/dev/shm`):

```bash
docker run -d --name gpupool-agent --gpus all --network host --pid host --ipc host -v gpupool-agent:/data \
  -e GPUPOOL_JOIN="http://10.0.0.1:8080#<cluster_token>" -e HF_TOKEN \
  ghcr.io/longduongbao29/gpupool-agent:latest-all
```

Wheel của các engine mang theo thư viện CUDA riêng: các phiên bản đã pin (vLLM 0.30.0, SGLang 0.5.20, PyTorch 2.13 cho
CUDA 13) cần **driver NVIDIA >= 580**; phần llama.cpp vẫn chạy trên driver cũ hơn. Tự build, hoặc dùng phiên bản khác
hay CUDA cũ hơn:

```bash
docker build -f docker/agent.Dockerfile --build-arg WITH_ENGINES=1 \
  --build-arg VLLM_VERSION=0.30.0 --build-arg SGLANG_VERSION=0.5.20 \
  --build-arg VLLM_PIP_ARGS="--extra-index-url <index wheel cu128>" -t gpupool-agent:all .
```

### Tự cài engine

Agent không cài vLLM hay SGLang. Agent chạy một lệnh nếu lệnh đó có trên server, và báo engine cho coordinator
(`engine:vllm`, `engine:sglang` trong features của server). Model của engine nào chỉ được đặt lên server báo engine đó.

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `GPUPOOL_VLLM_CMD` | `vllm serve` | Lệnh gốc của vLLM. Dùng wrapper nào cũng được: `/opt/vllm/bin/vllm serve`, `uvx vllm serve`, một script chạy container. |
| `GPUPOOL_SGLANG_CMD` | `python3 -m sglang.launch_server` | Lệnh gốc của SGLang. Với `python -m ...`, interpreter đó phải có package. |
| `GPUPOOL_HF_HOME` | `<cache_dir>/hf` | Nơi engine tải model về (`HF_HOME`). Nếu môi trường của agent đã có `HF_HOME` thì dùng giá trị đó. |
| `HF_TOKEN` | | Đặt trên agent cho repo gated hoặc private (engine tự tải). Đặt cả trên coordinator: nó đọc `config.json` để lập kế hoạch. |
| `GPUPOOL_ALLOW_ENGINE_ARGS` | `0` | `1` thì chấp nhận flag engine bổ sung của model. Mặc định tắt: chỉ với cluster token thì không được truyền flag tùy ý cho engine. |

Ví dụ, vLLM trong một virtualenv cạnh agent:

```bash
python3 -m venv /opt/vllm && /opt/vllm/bin/pip install vllm
GPUPOOL_VLLM_CMD="/opt/vllm/bin/vllm serve" gpupool agent --llama-dir /opt/llama --join "http://coordinator:8080#<token>"
```

Với image thường, cài engine vào môi trường của agent, hoặc trỏ lệnh tới một wrapper.

## Chọn engine

Form model gợi ý engine ngay khi bạn điền (`POST /api/engines/recommend`, xem [API.vi.md](API.vi.md)), còn quyết định
vẫn là của bạn. Với mỗi engine, form cho biết engine có hỗ trợ model không, server nào chạy engine đó, và model có vừa
phần VRAM đang trống không, xếp theo mức phù hợp:

- GGUF trong thư viện: llama.cpp (vLLM và SGLang phục vụ repo Hugging Face);
- repo Hugging Face: vLLM, rồi SGLang, khi có server chạy chúng và model vừa một server; VLM, checkpoint AWQ/GPTQ hay
  model embedding/rerank không được đề xuất cho llama.cpp (gpupool không chuyển đổi hay phục vụ chúng bằng llama.cpp);
- model văn bản không vừa server nào: llama.cpp sau khi chuyển sang GGUF (nút **Convert to GGUF** mở trình chuyển đổi
  với repo đã điền sẵn), vì chỉ llama.cpp chia được model qua nhiều server.

Phần gợi ý đọc `config.json` và danh sách file của repo (không tải trọng số), và lập kế hoạch bằng chính scheduler dùng
khi launch.

## Thêm model

UI: **New model**, chọn **Engine** là vLLM hoặc SGLang, rồi điền **Hugging Face repo** (`owner/name`, có thể thêm
`@revision`). **Tensor parallel** để Auto (ít GPU nhất mà vừa), trừ khi muốn trải model ra nhiều GPU hơn.

API (xem [API.vi.md](API.vi.md)):

```bash
curl -X PUT http://coordinator:8080/api/models/qwen-vl -H "Authorization: Bearer $ADMIN" \
  -H 'content-type: application/json' -d '{
    "engine": "vllm", "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
    "ctx_size": 32768, "parallel": 8, "tensor_parallel": null,
    "trust_remote_code": false, "engine_args": []}'
curl -X POST http://coordinator:8080/api/models/qwen-vl/start -H "Authorization: Bearer $ADMIN"
```

`ctx_size` là độ dài tối đa của một request (`--max-model-len` / `--context-length`) và cũng là cỡ KV cache mà ước lượng
dành ra. Engine chia sẻ cache này giữa các request (paged), nên không chia cho `parallel`. `parallel` là số request
phục vụ cùng lúc (`--max-num-seqs` / `--max-running-requests`).

Coordinator đọc `config.json` và danh sách file của repo ngay khi lưu model: sai tên, repo gated mà thiếu token, hoặc
`tensor_parallel` không chia hết số attention head đều bị từ chối ngay (422).

Ảnh gửi trong tin nhắn chat, giống OpenAI:

```json
{"model": "qwen-vl", "messages": [{"role": "user", "content": [
  {"type": "text", "text": "Trong ảnh có gì?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}
```

Router chuyển tiếp `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/rerank` và
`/v1/score` tới mọi engine; model trả lời được endpoint nào là tùy engine và model. Trường `cache_prompt` của llama.cpp
chỉ được thêm cho replica llama.cpp.

## Cách đặt model

Một replica vLLM hoặc SGLang chạy thành một tiến trình trên một server, dùng `k` GPU của server đó (`k` thuộc 1, 2, 4, 8
và chia hết số attention head). Mỗi GPU cần:

- trọng số / k (file `model*.safetensors` của repo, nếu không có thì mọi safetensors ở thư mục gốc, nếu không nữa thì
  `pytorch_model*.bin`);
- KV cache / k cho `ctx_size` token với cache bf16 (model MLA: toàn bộ latent cache trên mọi GPU; layer sliding-window
  và linear-attention lấy theo `layer_types`);
- activation cho một chunk 2048 token, cộng 1 GB runtime (CUDA context, NCCL, CUDA graph).

Engine định cỡ bộ nhớ theo tỉ lệ trên **tổng** bộ nhớ mỗi GPU (`--gpu-memory-utilization`, `--mem-fraction-static`).
gpupool tính tỉ lệ đó từ ước lượng, nên engine không chiếm phần bộ nhớ người khác đang dùng, và kiểm tra kết quả với
phần trống của từng GPU (cùng tỉ lệ thì GPU lớn hơn chiếm nhiều hơn). Giữa các server vừa, điểm số thông thường quyết
định (tốc độ, dùng chung, spread, lãng phí); khi không đặt `tensor_parallel`, mỗi server đề xuất `k` nhỏ nhất vừa.
Ước lượng thô hơn llama.cpp và không có tự hiệu chỉnh bộ nhớ: nếu model không khởi động được vì thiếu bộ nhớ, giảm
`ctx_size` hoặc `parallel`, hoặc đặt `tensor_parallel`.

## Giới hạn của phiên bản này

- Mỗi replica một server: chưa điều phối vLLM / SGLang nhiều node (Ray, NCCL giữa các máy).
- Không có hiệu chỉnh bộ nhớ và học tốc độ kiểu llama.cpp cho các engine này; ước lượng tốc độ dựa trên băng thông và
  chỉ dùng để so các phương án của cùng một model.
- Chưa quản lý model thị giác của llama.cpp (`--mmproj`).
- Flag engine là văn bản tự do: gpupool không kiểm tra, và flag làm đổi lượng bộ nhớ (`--kv-cache-dtype fp8`,
  `--max-num-batched-tokens`...) không được tính vào ước lượng.
