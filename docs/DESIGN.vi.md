# gpupool — Thiết kế

> Bản tiếng Việt. Bản tiếng Anh: [DESIGN.en.md](DESIGN.en.md). Hai bản phải được cập nhật cùng nhau.

gpupool gom VRAM trống rải rác trên nhiều server thành một pool và serve LLM qua một API
OpenAI-compatible duy nhất. Engine là llama.cpp (`llama-server` + `ggml-rpc-server`); phần
tự viết là control plane: agent, scheduler, coordinator, router.

## 1. Quyết định thay đổi so với kế hoạch gốc

| Mục | Kế hoạch | Chốt | Lý do |
| --- | --- | --- | --- |
| Ngôn ngữ control plane | Go hoặc Rust | Python 3.12, cài bằng `uv` | `uv` là 1 binary user-space, tự tải Python riêng → không phụ thuộc Python/sudo của server. |
| Tên binary RPC | `rpc-server` | `ggml-rpc-server` (b11342) | Upstream đã đổi tên; agent dò cả hai. |
| Bản llama.cpp | build từ source | pin b11342; prebuilt hoặc tự build, truyền qua `llama_dir` | Driver máy dev hỗ trợ CUDA 13.3 → bản `cuda-13.4` không chạy, dùng `cuda-12.4`. |
| Thứ tự device | ngầm định | luôn truyền `--device` + `--tensor-split` cùng thứ tự | Đã kiểm: llama.cpp liệt kê device local trước, RPC sau. |
| Remote device | 1 rpc-server / node | 1 rpc-server / device | Port riêng, tên `RPCi` xác định được, dừng được từng phần. |
| Nguồn model | URL | URL, đường dẫn local, hoặc `coordinator://<file>` | Server gọi được nhau nhưng không chắc có internet. |

## 2. Phạm vi bản đầu

Làm: agent (GĐ 1), scheduler (GĐ 2), router + cân bằng tải + prefix-aware + metrics (GĐ 3),
phần lõi của GĐ 4 (mất node → đặt lại, retry trước token đầu, drain, API key). GĐ 0 (số đo
baseline) chạy trên máy dev trong bước test.

Hoãn: autoscale, dashboard web, rate limit, GĐ 5, backend vLLM.

## 3. Kiến trúc

```
client ──HTTP (OpenAI API, API key)──► coordinator
                                       ├ router     (proxy, cân bằng tải, prefix)
                                       ├ scheduler  (ước lượng + placement, hàm thuần)
                                       ├ reconciler (vòng 2s: health, đặt lại, drain)
                                       └ store      (SQLite, WAL)
                                            ▲ heartbeat 2s        │ start/stop engine
                                            │                     ▼
   server A: agent ─ llama-server (head) ──TCP RPC──► server B: agent ─ ggml-rpc-server (CUDA0)
                                         └─TCP RPC──► server C: agent ─ ggml-rpc-server (CUDA0)
```

- Một package `gpupool`, một CLI: `gpupool agent | coordinator | register | plan | scale | undeploy | status`.
- Chỉ head node cần file GGUF; rpc-server nhận weights qua mạng (`-c` bật cache phía rpc).
- Agent ↔ coordinator xác thực bằng `Authorization: Bearer <cluster_token>`. RPC không mã hoá:
  rpc-server chỉ bind IP nội bộ của node.

## 4. Contract dữ liệu

Nguồn chuẩn: `src/gpupool/common/models.py` (pydantic v2). Tóm tắt:

| Model | Hướng | Nội dung chính |
| --- | --- | --- |
| `Device` | agent → coordinator | `device_id` ("CUDA0", "CPU"), `kind`, `total_mb`, `free_mb`, `usable_mb` = max(0, min(free − margin, budget)), `uuid` / `pci_bus_id` (định danh cố định của GPU vật lý; `device_id` chỉ là vị trí, bị dịch khi một GPU rớt khỏi bus), `budget_mb` (mức giới hạn đã cấu hình, nếu có: coordinator còn trừ thêm ước lượng của chính các replica đang chạy của nó khỏi mức này, vì bộ nhớ trống không phản ánh phần các replica đó đang giữ) |
| `NodeReport` | heartbeat | `node_id`, `agent_url`, `host` (IP node khác dùng cho RPC), devices, engines, `llama_version`, models |
| `EngineSpec` | coordinator → agent | `engine_id`, `kind` rpc/server, `port`, `devices` (thứ tự = `--device`), `rpc_endpoints`, `tensor_split`, `ctx_size`, `parallel` |
| `EngineStatus` | agent → coordinator | `state` starting/running/exited/failed, `exit_code`, `log_tail` (≤ 50 dòng) |
| `ModelSpec` | admin | `name`, `source`, `ctx_size`, `parallel`, `replicas` |
| `ModelMeta` | đọc từ GGUF | `n_layers`, `n_head_kv`, `head_dim`, `layer_bytes[i]`, `output_bytes` |
| `Placement` | scheduler | `tier`, `head_node`, `head_port`, `assignments` (node, device, `llama_device`, `rpc_endpoint`, layers, est_mb), `tensor_split` |
| `ReplicaRecord` | store | placement + state `pending → launching → ready → draining → stopped`, hoặc `failed` |

## 5. API

**Agent** (`:7070`, bearer token trừ `/health`):

| Method | Path | Vào → Ra |
| --- | --- | --- |
| GET | `/health` | `{"ok": true}` |
| GET | `/report` | `NodeReport` |
| POST | `/engines` | `EngineSpec` → `EngineStatus` (409 đã chạy, 422 port bận / thiếu model) |
| GET / DELETE | `/engines/{id}` | `EngineStatus` (DELETE: terminate, 10s rồi kill) |
| POST | `/models/ensure` | `{"name","source"}` → `{"path","bytes"}` (tải vào `.part`, đổi tên nguyên tử) |

**Coordinator** (`:8080`):

| Method | Path | Ghi chú |
| --- | --- | --- |
| POST | `/internal/heartbeat` | cluster token |
| GET | `/files/{name}` | cluster token; file trong `models_dir`, chặn path traversal |
| POST | `/admin/models` | đăng ký `ModelSpec` (admin key) |
| DELETE | `/admin/models/{name}` | drain mọi replica rồi xoá |
| POST | `/admin/models/{name}/scale?replicas=N` | đổi số replica mong muốn |
| POST | `/admin/deploy/{model}?dry_run=1` | trả `Placement`, không launch |
| DELETE | `/admin/replicas/{id}` | drain một replica |
| GET | `/admin/status` | node, device, replica, outstanding |
| GET | `/v1/models` | OpenAI format |
| POST | `/v1/chat/completions`, `/v1/completions` | proxy, hỗ trợ `stream: true` (API key) |
| GET | `/metrics` | Prometheus text |

## 6. Scheduler

**Ước lượng bộ nhớ** (`scheduler/estimate.py`):

- `kv_bytes_per_layer = 2 × ctx_size × n_head_kv × head_dim × 2` (K và V, f16).
- Device giữ dải layer `L` cần: tổng `layer_bytes` của `L` + `|L| × kv` + overhead
  (300 MB với CUDA, 150 MB với CPU).
- `output_bytes` (output.weight, hoặc token_embd khi tied, + output_norm) tính vào **device cuối**
  theo thứ tự `--device`. token_embd nằm ở RAM host, không tính vào GPU.
- Metadata đọc bằng parser GGUF tự viết, chỉ đọc phần header (file local hoặc HTTP stream),
  không tải cả file.

**Luật chọn** (`scheduler/placement.py`, `plan(...) -> Placement`, raise `NoFit`):

1. **Chỉ GPU trước.** Chạy cả ba tier dưới đây với các device CUDA. Chỉ khi GPU của cả pool
   không chứa nổi mới thêm device CPU; khi đó một device CPU đơn lẻ không được thắng tier
   single_gpu (chia GPU + CPU nhanh hơn chạy toàn CPU).
2. **single_gpu**: best-fit — device nhỏ nhất vẫn vừa, để chừa device lớn cho model lớn.
3. **single_node**: trên một node, cộng device theo usable giảm dần tới khi vừa; chọn node cần ít device nhất.
4. **multi_node**: thêm node theo tổng usable giảm dần (ít node nhất), rồi bỏ các device không cần. Khi pool có cả CUDA lẫn CPU, sinh thêm một ứng viên xuất phát từ mọi device CUDA, chỉ thêm node CPU khi cần và chỉ bỏ device CPU; bộ chấm điểm chọn phương án nhanh hơn, nên GPU nhỏ không bị loại để đặt toàn bộ lên CPU trừ khi CPU thật sự nhanh hơn.
5. Chia layer theo tỉ lệ dung lượng, rồi sửa bằng số byte thật của từng layer cho tới khi mọi
   device có `est_mb ≤ usable_mb`. Head = node giữ nhiều layer nhất. `tensor_split` = số layer.
6. Thứ tự device: device CUDA của head (`CUDA0`…), sau đó mọi device còn lại qua RPC
   (`RPC0`, `RPC1`…), kể cả CPU của chính head.

Chưa có số đo mạng nên chưa sắp theo latency (để GĐ 5).

## 7. Router

- Ứng viên = replica `ready` của model có head node còn sống.
- **Prefix key** = sha256 của messages trừ message cuối (JSON chuẩn hoá, cắt 4 KB); nếu chỉ có
  1 message thì 512 ký tự đầu.
- Rendezvous hash(prefix, replica) chọn replica ưu tiên; nếu nó bận hơn replica rảnh nhất quá
  2 request thì chuyển sang replica rảnh nhất.
- Gửi kèm `cache_prompt: true`; head chạy với `--cache-reuse 256 --metrics`.
- Retry khi lỗi kết nối hoặc 5xx **trước byte đầu tiên**, tối đa 2 lần. Đã gửi byte thì không retry.
- Bộ đếm outstanding được giảm đúng một lần, kể cả khi client ngắt giữa stream.

## 8. Reconciler (vòng 2s)

- Node bị coi là chết khi báo cáo cũ hơn 10s VÀ ít nhất 2 lần poll liên tiếp thực sự thất bại (coordinator tự bị đứng vòng lặp thì không làm node bị khai tử) → replica dùng node đó `failed`, dừng engine còn lại. Một watchdog ghi cảnh báo khi event loop của coordinator bị chặn quá 1s.
- Engine báo `exited/failed` hoặc biến mất khỏi report → replica `failed`.
- Thiếu replica so với `replicas` → `plan()` rồi launch, mỗi tick tối đa 1 replica mỗi model;
  launch lỗi → backoff 5s, 10s, 20s… tối đa 300s.
- **Giữ chỗ VRAM**: khi plan, `usable_mb` bị trừ phần của replica đang `launching`, và của
  replica vừa `ready` cho tới khi node gửi report mới hơn 5s sau thời điểm ready (report cũ có
  thể đo trước khi model load xong → cấp trùng VRAM).
- Launch: start rpc engine → chờ running → `ensure` model trên head → start head → chờ
  `/health` 200 (`launch_timeout_s`) → `ready`. Lỗi ở bất kỳ bước nào → dừng mọi engine đã tạo.
- Device đang chạy engine có `free_mb < 256` → launch replica thay thế trước, sẵn sàng rồi mới drain replica cũ.
- Drain: chờ outstanding = 0 (tối đa 60s) → dừng engine → `stopped`.

## 9. Cấu trúc code

```
src/gpupool/
  common/     models.py auth.py config.py           contract dùng chung
  agent/      gpu.py procs.py models_cache.py app.py
  scheduler/  gguf_meta.py estimate.py placement.py
  coordinator/ store.py agent_client.py reconciler.py app.py
  router/     balancer.py proxy.py
  cli.py
tests/        test_agent_* test_scheduler_* test_coordinator_* test_router_*
```

Chạy test: `uv run pytest` (unit), `uv run pytest -m real` (cần binary llama.cpp, GGUF, GPU).

## 10. Kế hoạch test

Máy dev có 1 GTX 1650 4 GB → giả lập 3 server bằng 3 agent trên localhost:

- Agent A: `CUDA0` thật, giới hạn `budget_mb` để buộc phải split.
- Agent B, C: device `CPU` với `budget_mb` cố định, chạy `ggml-rpc-server -d CPU` thật → RPC thật qua TCP.

| Test | Cách | Đạt khi |
| --- | --- | --- |
| Baseline (GĐ 0) | Qwen2.5-0.5B Q4_K_M trên CUDA0 | có tokens/s prefill, decode, TTFT |
| Ước lượng vs thực tế | so `est_mb` với VRAM đo được | sai số ≤ 20% |
| Split thật | Qwen2.5-3B Q4_K_M, A bị giới hạn → split qua B, C | trả lời đúng qua `/v1/chat/completions`, có tokens/s |
| Failover | kill agent của một replica | request sau vẫn thành công, replica được đặt lại |

## 11. Câu hỏi mở đã trả lời (2026-10-02)

- Server gọi được nhau, internet không đảm bảo → nguồn `coordinator://<file>`; binary
  llama.cpp truyền qua `llama_dir`.
- Không giới hạn kích thước model; mục tiêu là dùng hết tổng VRAM khả dụng → split nhiều node là
  đường chính; margin cấu hình theo node (`margin_pct`, `margin_min_mb`, `budget_mb`).
