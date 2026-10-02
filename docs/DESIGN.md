# gpupool — Design (bước 1)

Nguồn: kế hoạch "Framework serve LLM trên GPU dị thể nhiều server". Tài liệu này chốt
phạm vi bản đầu, kiến trúc, và **contract chính xác giữa các module** trước khi code.

## 1. Quyết định thay đổi so với kế hoạch

| Mục | Kế hoạch | Chốt | Lý do |
| --- | --- | --- | --- |
| Ngôn ngữ control plane | Go hoặc Rust | **Python 3.12, cài bằng `uv`** | `uv` là 1 binary user-space, tự tải Python riêng → không phụ thuộc Python/sudo của server. Đây đúng là lý do ban đầu muốn Go/Rust. |
| Tên binary RPC | `rpc-server` | `ggml-rpc-server` (b11342) | Đã đổi tên ở upstream; agent dò cả hai tên. |
| Bản llama.cpp | build từ source | pin **b11342**, ưu tiên bản prebuilt, build khi không có | Máy dev: driver hỗ trợ CUDA 13.3 → bản `cuda-13.4` không chạy, phải dùng `cuda-12.4`. Agent phải chọn bản theo driver. |
| Thứ tự device | ngầm định | luôn truyền `--device CUDA0,RPC0,...` + `--tensor-split` cùng thứ tự | Đã kiểm: local liệt kê trước, RPC sau. Truyền tường minh để không phụ thuộc thứ tự đó. |
| Remote device | 1 rpc-server / node | **1 rpc-server / device** | Mỗi device có port riêng, tên `RPCi` xác định được; dễ dừng từng phần. |

## 2. Phạm vi bản đầu

Làm: giai đoạn 1 (agent), 2 (scheduler), 3 (router + LB + prefix-aware, metrics), phần lõi
của 4 (mất node → re-place, retry trước token đầu, drain, API key). Giai đoạn 0 làm trên máy
dev như một phần của bước test (số đo baseline).

Hoãn: autoscale, dashboard web, rate limit, giai đoạn 5, backend vLLM.

## 3. Kiến trúc

```
client ──HTTP (OpenAI API, API key)──► coordinator ──────────────┐
                                       ├ router   (proxy, LB, prefix)
                                       ├ scheduler (estimate + placement, thuần hàm)
                                       ├ reconciler (vòng 2s: health, re-place, drain)
                                       └ store    (SQLite, WAL)
                                            ▲ heartbeat 2s        │ start/stop engine
                                            │                     ▼
   server A: agent ─ llama-server (head) ──TCP RPC──► server B: agent ─ ggml-rpc-server (CUDA0)
                                         └─TCP RPC──► server C: agent ─ ggml-rpc-server (CUDA0)
```

- Một package `gpupool`, một CLI: `gpupool agent`, `gpupool coordinator`, `gpupool plan|deploy|undeploy|status`.
- Chỉ head node cần file GGUF; rpc-server nhận weights qua mạng (bật `-c` để cache phía rpc).
- Agent ↔ coordinator xác thực bằng `Authorization: Bearer <cluster_token>`. rpc-server chỉ bind
  IP nội bộ của node (RPC không mã hoá).

## 4. Contract dữ liệu (`gpupool/common/models.py`, pydantic v2)

```python
class Device(BaseModel):
    device_id: str            # tên llama.cpp: "CUDA0", "CPU"
    kind: Literal["cuda", "cpu"]
    name: str
    total_mb: int
    free_mb: int              # đo thực (NVML / psutil)
    usable_mb: int            # min(free_mb - margin, budget_mb) ; >= 0
    util_pct: int | None

class EngineSpec(BaseModel):  # coordinator -> agent
    engine_id: str            # "<replica_id>-head" | "<replica_id>-rpc-<device_id>"
    kind: Literal["rpc", "server"]
    port: int
    devices: list[str]        # rpc: đúng 1 local device; server: thứ tự đầy đủ, vd ["CUDA0","RPC0","RPC1"]
    model: str | None = None          # server: tên model (alias) 
    model_path: str | None = None     # server: đường dẫn GGUF trên head node
    rpc_endpoints: list[str] = []     # server: "host:port", thứ tự khớp RPC0, RPC1...
    tensor_split: list[float] = []    # server: cùng độ dài, cùng thứ tự với devices
    ctx_size: int = 4096              # tổng ctx (chia đều cho các slot)
    parallel: int = 1
    extra_args: list[str] = []

class EngineStatus(BaseModel):
    engine_id: str
    kind: Literal["rpc", "server"]
    state: Literal["starting", "running", "exited", "failed"]
    pid: int | None
    port: int
    exit_code: int | None
    log_tail: list[str]       # <= 50 dòng cuối

class NodeReport(BaseModel):  # agent -> coordinator (heartbeat)
    node_id: str
    agent_url: str            # "http://10.0.0.5:7070"
    host: str                 # IP mà node khác dùng để kết nối RPC
    devices: list[Device]
    engines: list[EngineStatus]
    llama_version: str        # "b11342"
    models: list[str]         # file GGUF đã có trong cache
    ts: float

class ModelSpec(BaseModel):
    name: str                 # alias trả về ở /v1/models
    source: str               # URL tải (https://...gguf) hoặc đường dẫn local tuyệt đối
    ctx_size: int = 4096
    parallel: int = 1
    replicas: int = 1

class ModelMeta(BaseModel):   # đọc từ header GGUF
    n_layers: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    layer_bytes: list[int]    # tổng bytes tensor của từng block "blk.{i}."
    other_bytes: int          # token_embd, output, output_norm...
    file_bytes: int

class DeviceAssignment(BaseModel):
    node_id: str
    device_id: str            # tên trên node của nó
    llama_device: str         # tên ở phía head: "CUDA0" hoặc "RPC0"...
    rpc_endpoint: str | None  # None nếu local trên head
    layers: int
    est_mb: int

class Placement(BaseModel):
    model: str
    replica_id: str
    tier: Literal["single_gpu", "single_node", "multi_node"]
    head_node: str
    head_port: int
    assignments: list[DeviceAssignment]   # thứ tự = thứ tự --device
    tensor_split: list[float]
    est_total_mb: int
```

Replica state trong store: `pending → launching → ready → draining → stopped`, hoặc `failed`.

## 5. API

**Agent** (`:7070`, cần bearer token):

| Method | Path | Vào → Ra |
| --- | --- | --- |
| GET | `/health` | → `{"ok": true}` (không cần token) |
| GET | `/report` | → `NodeReport` |
| POST | `/engines` | `EngineSpec` → `EngineStatus` (409 nếu engine_id đã chạy, 422 nếu port bận) |
| GET | `/engines/{id}` | → `EngineStatus` |
| DELETE | `/engines/{id}` | → `EngineStatus` (terminate, 10s rồi kill) |
| POST | `/models/ensure` | `{"name","source"}` → `{"path","bytes"}` (tải về `.part`, rename nguyên tử) |

**Coordinator** (`:8080`):

| Method | Path | Ghi chú |
| --- | --- | --- |
| POST | `/internal/heartbeat` | `NodeReport` → `{"ok": true}` (cluster token) |
| POST | `/admin/models` | `ModelSpec`, lưu registry (admin key) |
| POST | `/admin/deploy/{model}?dry_run=1` | → `Placement` (dry-run không launch) |
| DELETE | `/admin/replicas/{replica_id}` | drain rồi stop |
| GET | `/admin/status` | node, device, replica, outstanding |
| GET | `/v1/models` | OpenAI format |
| POST | `/v1/chat/completions`, `/v1/completions` | proxy, hỗ trợ `stream: true` (API key) |
| GET | `/metrics` | Prometheus text |

## 6. Scheduler

**Ước lượng bộ nhớ** (thuần hàm, `estimate.py`):

- `kv_bytes_per_layer = 2 * ctx_size * n_head_kv * head_dim * 2` (K và V, f16).
- Device giữ `L` layer cần `sum(layer_bytes của L layer) + L * kv_bytes_per_layer + overhead`.
- `overhead` = 300 MB / CUDA device (context + compute buffer), 150 MB / CPU device.
- `output_bytes` (output.weight hoặc token_embd khi tied + output_norm) tính vào **device cuối** theo thứ tự --device; token_embd nằm ở RAM host, không tính vào GPU.

**Luật chọn** (`placement.py`, thuần hàm `plan(meta, spec, nodes, exclude) -> Placement`, raise `NoFit`):

1. **single_gpu**: device nào chứa trọn → chọn *best-fit* (usable nhỏ nhất mà vẫn vừa) để chừa device lớn cho model lớn.
2. **single_node**: trên cùng 1 node, cộng device theo usable giảm dần tới khi vừa; chọn node cần ít device nhất.
3. **multi_node**: thêm node theo tổng usable giảm dần, ít node nhất. Head = node có phần lớn nhất.
4. Chia layer tỉ lệ `usable_mb` (largest-remainder để tổng = n_layers), rồi kiểm từng device `est_mb <= usable_mb`; nếu vượt, dời 1 layer sang device còn dư nhiều nhất, lặp; không được → thử tier sau.
5. `tensor_split` = số layer của từng device (llama.cpp chuẩn hoá tỉ lệ).

Không có số đo mạng ở bản đầu → không sắp theo latency (để giai đoạn 5).

## 7. Router

- Ứng viên = replica `ready` của model, node head còn sống.
- **Prefix key** = sha256 của messages trừ message cuối (JSON chuẩn hoá), cắt 4 KB; nếu chỉ có 1 message thì 512 ký tự đầu.
- Chọn bằng rendezvous hash(prefix, replica_id); nếu replica đó có `outstanding > min_outstanding + 2` thì lấy replica ít outstanding nhất.
- Gửi kèm `cache_prompt: true`; head chạy với `--cache-reuse 256 --metrics`.
- Retry: lỗi kết nối hoặc 5xx **trước byte đầu tiên** → thử replica khác, tối đa 2 lần. Đã stream được byte thì không retry, đóng stream với lỗi.
- Đếm outstanding trong `try/finally` để không rò khi client ngắt.

## 8. Reconciler (vòng 2s)

- Node không heartbeat > 10s → `dead`; replica dùng node đó → `failed`, router ngừng route.
- Engine báo `exited/failed` → replica `failed`.
- Số replica `ready+launching` < `replicas` → `plan()` (loại node dead) → launch.
- Launch: start các rpc engine → chờ port mở → `ensure` model trên head → start server → poll `GET /health` của llama-server tới 200 (timeout 600s) → `ready`. Lỗi bất kỳ bước nào → dừng hết engine đã tạo, `failed`.
- Device có `free_mb < 256` mà đang chạy engine → tạo replica thay thế trước, rồi drain replica cũ.
- Drain: `draining` → chờ outstanding = 0 (tối đa 60s) → stop engine → `stopped`.

## 9. Cấu trúc code và ownership (cho bước implement song song)

```
src/gpupool/
  common/models.py   common/auth.py   common/config.py      ← lead viết trước (contract)
  agent/gpu.py       agent/procs.py   agent/models_cache.py  agent/app.py   ← nhóm A
  scheduler/gguf_meta.py  scheduler/estimate.py  scheduler/placement.py    ← nhóm B
  coordinator/store.py  coordinator/reconciler.py  coordinator/app.py      ← nhóm C
  router/balancer.py  router/proxy.py                                      ← nhóm D
  cli.py                                                                   ← lead (tích hợp)
tests/  test_agent_*.py | test_scheduler_*.py | test_coordinator_*.py | test_router_*.py
```

Dependencies: fastapi, uvicorn, httpx, pydantic, nvidia-ml-py, gguf, psutil; test: pytest, pytest-asyncio, respx.

## 10. Kế hoạch test (bước 3)

Máy dev chỉ có 1 GTX 1650 4 GB → giả lập 3 server bằng 3 agent trên localhost:

- Agent A: `CUDA0` thật, `budget_mb` giới hạn để ép split.
- Agent B, C: device `CPU` với `budget_mb` cố định, chạy `ggml-rpc-server -d CPU` thật → đường RPC thật qua TCP.

| Test | Cách | Đạt khi |
| --- | --- | --- |
| Unit scheduler | bảng case: vừa 1 GPU / 1 node / nhiều node / NoFit | đúng tier, tổng layer = n_layers, không device nào vượt usable |
| Estimator vs thực tế | so `est_mb` với VRAM đo được khi load | sai số ≤ 20% |
| Router | replica giả (httpx mock) | prefix giống → cùng replica; retry trước byte đầu; outstanding về 0 |
| Baseline (GĐ 0) | Qwen2.5-0.5B Q4_K_M trên CUDA0 | có tokens/s prefill, decode, TTFT |
| Split thật | Qwen2.5-3B Q4_K_M, A giới hạn ~1.2 GB → split qua B, C | trả lời đúng qua `/v1/chat/completions`, có số tokens/s |
| Failover | 2 replica, kill agent + engine của 1 replica | request sau vẫn thành công, replica được re-place |

## 11. Trả lời câu hỏi mở (2026-10-02)

- Các server gọi được nhau; internet không đảm bảo → `ModelSpec.source` nhận thêm
  `coordinator://<file>`: agent tải GGUF từ `GET /files/<file>` của coordinator (đọc từ `models_dir`).
  Binary llama.cpp do agent nhận qua `--llama-dir` (prebuilt hoặc tự build), không bắt buộc tải GitHub.
- Không giới hạn kích thước model; mục tiêu là dùng hết tổng VRAM khả dụng của pool → split nhiều node
  là đường chính, margin cấu hình được theo node (`margin_pct`, `margin_min_mb`, `budget_mb` theo device).
- Coordinator đọc metadata GGUF bằng parser tự viết đọc tuần tự (file local hoặc HTTP Range), không cần tải cả file.
