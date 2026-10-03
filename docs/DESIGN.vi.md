# gpupool — Thiết kế

> Bản tiếng Việt. Bản tiếng Anh: [DESIGN.en.md](DESIGN.en.md). Hai bản phải được cập nhật cùng nhau.

gpupool gom VRAM rảnh rải rác trên nhiều server thành một pool và phục vụ nhiều LLM qua một API tương
thích OpenAI. Engine là llama.cpp (`llama-server` + `ggml-rpc-server`, build b11342); phần chúng ta viết là
control plane: một **agent** trên mỗi server GPU, và một **coordinator** chứa scheduler, reconciler,
autoscaler, router và web UI.

Tài liệu này mô tả hệ thống đúng như trong code hiện tại. Tài liệu liên quan:

- Toàn bộ HTTP API (từng endpoint, body, lỗi): [API.vi.md](API.vi.md).
- Vì sao nền tảng đa model hoạt động như vậy (mô hình tài nguyên, chính sách, thuật toán): [PLATFORM_DESIGN.vi.md](PLATFORM_DESIGN.vi.md).
- Thiết kế web UI: [UI_DESIGN.vi.md](UI_DESIGN.vi.md). Cài đặt: [QUICKSTART.vi.md](QUICKSTART.vi.md). Kết quả kiểm thử: [TEST_REPORT.vi.md](TEST_REPORT.vi.md).

## 1. Các quyết định đã thay đổi so với kế hoạch gốc

| Hạng mục | Kế hoạch | Quyết định | Lý do |
| --- | --- | --- | --- |
| Ngôn ngữ control plane | Go hoặc Rust | Python 3.12, cài bằng `uv` | `uv` là một binary user-space tự tải Python riêng, nên không phụ thuộc Python của server hay sudo. |
| Tên binary RPC | `rpc-server` | `ggml-rpc-server` (b11342) | Upstream đã đổi tên; agent tìm cả hai. |
| Bản llama.cpp | build từ source | cố định b11342; build sẵn hoặc tự build, truyền qua `llama_dir`; image Docker đã kèm sẵn | Driver trên máy dev hỗ trợ CUDA 13.3 nên build `cuda-13.4` không chạy; dùng `cuda-12.4`. |
| Thứ tự device | ngầm định | luôn truyền `--device` + `--tensor-split` cùng thứ tự, và `--rpc` trước `--device` | llama.cpp liệt kê device local trước rồi mới tới RPC; nó phân giải tên device ngay khi parse đối số, nên `RPC0` chỉ tồn tại sau khi `--rpc` đã đăng ký các server. |
| Device từ xa | 1 rpc-server mỗi node | 1 rpc-server mỗi device | Có cổng riêng, tên `RPCi` xác định, dừng được riêng lẻ. |
| Nguồn model | URL | URL, đường dẫn tuyệt đối trên máy, hoặc `coordinator://<file>` | Các server thấy nhau nhưng có thể không có internet. |
| Kiểm tra sống của agent | agent đẩy heartbeat mỗi 2 s | coordinator **kéo** `GET /report` từ mọi agent đã đăng ký (`poll_s` = 2 s) | Với push, server đã xóa trên UI sẽ tự xuất hiện lại ở nhịp tiếp theo. Push vẫn còn (`push_heartbeat`) cho cấu hình cũ, nhưng chỉ nhận server đã đăng ký. |
| Đăng ký server | coordinator biết agent qua config | agent tự đăng ký (`POST /internal/join`, chạy bởi `--join "<url>#<token>"`); UI cũng thêm được bằng URL | Một lệnh cho mỗi server. Coordinator thăm dò `/report` trước, nên chỉ agent truy cập được và đúng token mới đăng ký được; server đã xóa trên UI nhận 403 để việc xóa có hiệu lực lâu dài. |
| Autoscaling, dashboard | để sau | đã làm (mục 7 đến 9, web UI) | |
| Định danh GPU | `device_id` ("CUDA0") | `uuid` / `pci_bus_id` là khóa ổn định; `device_id` chỉ là vị trí | `device_id` bị dịch khi một GPU rơi khỏi bus. Placement lưu `device_uuid`; phát hiện lỗi và tính chiếm dụng đều phân giải theo uuid. |
| Ước lượng bộ nhớ | công thức tĩnh | công thức + tự hiệu chỉnh theo từng model (mục 12) | Công thức chỉ chính xác trên model nhỏ; chạy thật sẽ chỉnh lại. |

## 2. Phạm vi

Đã có: agent, scheduler, router với cân bằng tải và định tuyến theo prefix, metrics, phục hồi khi mất node,
drain, API key, **chính sách đa model** (priority, spread, giới hạn replica, autoscaling, scale-to-zero,
preemption), **rebalancing**, lượng tử hóa KV cache và speculative decoding, tường lửa RPC, tự hiệu chỉnh
VRAM, lưu trạng thái điều khiển, web UI, thư viện model (tải Hugging Face, đường dẫn local), sự kiện và
webhook, cluster 3 server giả lập cho demo và CI.

Chưa làm: rate limiting, sắp xếp theo độ trễ đo được giữa các node, backend vLLM, mã hóa lưu lượng RPC
(RPC của llama.cpp là TCP thuần; xem mục 11).

## 3. Kiến trúc

```
client ──HTTP (OpenAI API, API key)──► coordinator
                                       ├ router      (proxy, cân bằng tải, prefix, cold start)
                                       ├ scheduler   (ước lượng + placement + chấm điểm, hàm thuần)
                                       ├ reconciler  (vòng 2 s: sức khỏe, launch, drain, preempt, rebalance)
                                       ├ autoscaler  (vòng 2 s: scrape /metrics, số replica mong muốn)
                                       ├ poller      (2 s: GET /report từ mọi agent)
                                       ├ library     (file GGUF: tải HF / đường dẫn local)
                                       ├ events      (+ webhook tùy chọn)
                                       ├ web UI + /api
                                       └ store       (SQLite, WAL)
                                            ▲ kéo /report 2 s     │ start/stop engine
                                            │                     ▼
   server A: agent ─ llama-server (head) ──TCP RPC──► server B: agent ─ ggml-rpc-server (CUDA0)
                                         └─TCP RPC──► server C: agent ─ ggml-rpc-server (CUDA0)
```

- Một package `gpupool`, một CLI: `gpupool agent | coordinator | register | plan | scale | undeploy | status`.
- Agent và coordinator xác thực bằng `Authorization: Bearer <cluster_token>`; `/admin/*` và `/api/*` dùng
  admin key; `/v1/*` dùng API key (để mở khi chưa cấu hình key nào, và coordinator in cảnh báo). `admin_key` /
  `cluster_token` còn trống sẽ được sinh một lần và lưu trong `secrets.json` cạnh database (ghi nguyên tử;
  file không đọc được thì không bao giờ bị ghi đè).
- Lưu lượng giữa coordinator, agent và llama.cpp không bao giờ đi qua HTTP proxy (`common/net.py`); chỉ lưu
  lượng ra ngoài cluster (Hugging Face, URL model, webhook) mới được dùng proxy.

### 3.1 Chia layer qua RPC hoạt động thế nào

llama.cpp chia model **theo layer** lên các device (`--split-mode layer`). GPU từ xa chỉ là một device khác:
`llama-server` của head được chạy với `--rpc host:port,...` và thấy chúng là `RPC0`, `RPC1`, v.v.

- **Chỉ node head cần file GGUF.** Coordinator không bao giờ copy model sang các node RPC; head nạp model và
  gửi cho từng device từ xa các tensor của nó qua TCP khi model được nạp.
- Mỗi `ggml-rpc-server` chạy với `-c`, bật **cache tensor cục bộ**: lần nạp sau của cùng model gần như không
  truyền gì.
- Activation đi qua mạng một lần cho mỗi token mỗi chặng (hop), nên chia qua RPC là đổi tốc độ lấy dung
  lượng. Scheduler mô hình hóa việc này bằng chi phí cố định mỗi hop (mục 6.3) và chỉ dùng placement
  multi-node khi không có GPU đơn hay node đơn nào chứa vừa.
- Mỗi device một `ggml-rpc-server`, bind vào `host` của agent trên một cổng trong `port_range` (9000-9999) do
  coordinator cấp, nên cổng không bao giờ đụng nhau giữa các replica.
- Device CPU của head, nếu được mở (`include_cpu`), cũng được truy cập qua một rpc-server (`-d CPU`), nên nó
  cũng là device `RPCi`.

### 3.2 Head chạy gì

Dòng lệnh của head (`agent/procs.py`, `build_command`) luôn có `-ngl 999`, `--split-mode layer`,
`--cache-reuse 256`, `--metrics`, `--fit off` (scheduler đã chọn cách chia; auto-fit của llama.cpp không
được đổi nó) và `-lv 4` (mức log 4 in kích thước buffer model, KV và compute của từng device để hiệu chỉnh
đọc; mức 5 thêm một lượt dry-run nên phải đúng là 4). `--tensor-split` là số layer, được truyền khi có hơn
một device.

## 4. Hợp đồng dữ liệu

Nguồn sự thật: `src/gpupool/common/models.py` (pydantic v2). Đổi một field là đổi giao thức; agent và
coordinator khác phiên bản phải vẫn nói chuyện được, nên mọi field thêm sau 0.1 đều tùy chọn.

| Model | Hướng | Field chính |
| --- | --- | --- |
| `Device` | agent tới coordinator | `device_id` ("CUDA0", "CPU"), `kind`, `total_mb`, `free_mb`, `usable_mb` = max(0, min(free - margin, budget)), `budget_mb` (mức trần đã cấu hình; coordinator còn trừ ước lượng của các replica đang chạy của chính nó vì free memory không cho thấy phần đó), `uuid` / `pci_bus_id` (định danh ổn định), `bandwidth_gbps` (độ rộng bus NVML x xung nhớ; dùng để xếp hạng GPU), telemetry (`util_pct`, `temp_c`, `power_w`, `processes`, `driver`, `cuda`) |
| `NodeReport` | agent tới coordinator (`GET /report`) | `node_id`, `agent_url`, `host` (IP các node khác dùng cho RPC), devices, engines, `llama_version`, `models` (file GGUF trong cache local), telemetry CPU/RAM |
| `EngineSpec` | coordinator tới agent | `engine_id`, `kind` rpc/server, `port`, `devices` (thứ tự = `--device`), `rpc_endpoints`, `tensor_split`, `ctx_size`, `parallel`, `cache_type`, `spec_type`, `draft_model_path`, `draft_device`, `draft_n_max`, `allowed_peers` (rpc: host được phép kết nối) |
| `EngineStatus` | agent tới coordinator | `state` starting/running/exited/failed, `exit_code`, `log_tail` (tối đa 50 dòng) |
| `ModelSpec` | admin | `name`, `source`, `ctx_size`, `parallel`, `replicas` (0 = dừng), `pin_devices`, `priority`, `spread`, `min_replicas`, `max_replicas`, `autoscale`, `idle_unload_s`, `preemptible`, `kv_cache_type`, `speculative`, `draft`, `draft_n_max` |
| `AutoscalePolicy` | nằm trong `ModelSpec` | `target_busy` 0.7, `up_after_s` 30, `down_after_s` 300 |
| `ModelMeta` | đọc từ header GGUF | `n_layers`, `n_embd`, `n_head_kv`, `head_dim`, `layer_bytes[i]`, `output_bytes`, `vocab_size`, `tokenizer_model` |
| `Placement` | scheduler | `tier`, `head_node`, `head_port`, `assignments` (node, device, `device_uuid`, `llama_device`, `rpc_endpoint`, layers, `est_mb`), `tensor_split`, `est_total_mb`, `score`, `est_decode_tps`, `reasons`, `draft_est_mb`, `mem_factor` (hệ số calibration đã nhân vào các ước lượng) |
| `Occupant` | reconciler tới scheduler | một engine đang chiếm GPU: node, device, model, `est_mb`, `busy` (0..1) |
| `ReplicaRecord` | store | placement + trạng thái `pending`, `launching`, `ready`, `draining`, `stopped`, hoặc `failed` |
| `LibraryItem` | library | một file GGUF mà coordinator có thể phục vụ dưới dạng `coordinator://<name>` |

Nhóm trạng thái replica: *active* = pending, launching, ready (được tính vào số lượng mong muốn); *live* =
active + draining (vẫn giữ cổng và engine); *terminal* = stopped, failed.

## 5. Tổng quan API

Chi tiết, body và mã lỗi nằm trong [API.vi.md](API.vi.md). Các nhóm:

| Nhóm | Xác thực | Tiền tố và mục đích |
| --- | --- | --- |
| Agent | cluster token (`/health` để mở) | `/health`, `/report`, `POST /engines`, `GET/DELETE /engines/{id}`, `GET /engines/{id}/memory` (kích thước buffer parse từ log engine), `POST /models/ensure` (tải vào `.part`, đổi tên nguyên tử; cũng tải mọi phần của GGUF bị chia nhỏ, phần 1 sau cùng) |
| Coordinator nội bộ | cluster token | `/internal/join` (tự đăng ký), `/internal/heartbeat` (push, chỉ server đã đăng ký), `GET /files/{name}` (file thư viện cho head; tên được phân giải qua library, không bao giờ ghép vào đường dẫn), `/healthz` để mở |
| Admin | admin key | `/admin/models`, `/admin/models/{name}/scale`, `/admin/deploy/{model}?dry_run=1`, `/admin/replicas/{id}`, `/admin/status` (CLI dùng nhóm này) |
| API cho Web UI | admin key | `/api/state`, `/api/servers`, `/api/servers/{id}/gpus/{dev}` (bật/tắt một GPU trong pool), `/api/models/{name}` (PUT, start, stop, scaling, plan, delete), `/api/capacity`, `/api/simulate`, `/api/rebalance`, `/api/recommend`, `/api/events`, `/api/library`, `/api/hf/files` |
| OpenAI | API key | `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions` (hỗ trợ `stream: true`) |
| Metrics | không | `GET /metrics` (text Prometheus) |

## 6. Scheduler

`src/gpupool/scheduler/`: `gguf_meta.py`, `estimate.py`, `scoring.py`, `placement.py`. Tất cả là hàm thuần
của (metadata model, spec, báo cáo node, occupant); cổng chỉ được cấp cho placement thắng cuộc.

### 6.1 Ước lượng bộ nhớ (`estimate.py`)

Với một device giữ dải layer `L` (đã hiệu chỉnh theo log nạp verbose của llama.cpp b11342):

```
need = ceil( sum(layer_bytes[i] for i in L) + |L| x kv_bytes_per_layer [+ output_bytes nếu là device cuối] )
       + compute_buffer + runtime_context
kv_bytes_per_layer = 2 x ctx_size x n_head_kv x head_dim x bytes_per_element(kv_cache_type)
compute_buffer     = ceil(21 x 512 x n_embd x 4 bytes)      (ubatch mặc định 512)
runtime_context    = 128 MB (CUDA) | 32 MB (CPU)
```

- Byte mỗi phần tử KV: f16 2, q8_0 34/32, q4_0 18/32 (bố cục block của ggml).
- `output_bytes` (output.weight, hoặc token_embd khi tied, cộng output_norm) tính vào **device cuối** theo
  thứ tự `--device`. token_embd nằm trong RAM host và không tính vào GPU nào.
- Kết quả được nhân với hệ số bộ nhớ đã hiệu chỉnh của model (mục 12), bằng 1.0 cho tới khi model được đo.
- Draft model (mục 8.2) được ước lượng như một model nguyên vẹn trên một device CUDA, cùng ctx và kiểu
  cache, với compute buffer và context riêng.
- Metadata đến từ parser header GGUF tự viết (file local hoặc HTTP range stream), không bao giờ đọc cả file;
  GGUF bị chia nhỏ được cộng qua mọi phần (chỉ đọc phần 1 sẽ đếm thiếu VRAM).

### 6.2 Sinh ứng viên (`placement.py`)

`plan(...)` trả về `Placement` tốt nhất hoặc ném `NoFit`; `rank(...)` trả về vài phương án đầu để hiển thị,
mô phỏng và rebalancing.

1. **GPU trước.** Mọi tier chạy chỉ với device CUDA. Device CPU chỉ tham gia khi các GPU của cả pool không
   cho ứng viên nào; lúc đó các ứng viên một device bị bỏ qua (GPU + CPU nhanh hơn CPU một mình). Lý do:
   trong một tier, CPU được tính là dung lượng như GPU, và trộn vào ngay từ đầu sẽ đặt layer vào RAM host
   (chậm hơn 10 lần trở lên) trong khi nơi khác còn VRAM trống.
2. **single_gpu**: mọi device mà cả model chứa vừa đều là ứng viên.
3. **single_node**: với mỗi node, lấy device theo VRAM usable giảm dần, thêm từng cái một (bắt đầu từ hai)
   tới khi có cách chia khả thi.
4. **multi_node**: chỉ khi tier 2 và 3 không cho gì. Ứng viên là hợp của
   - tham lam: thêm node theo tổng usable giảm dần (ít node nhất), rồi bỏ các device không cần; khi trộn
     CUDA và CPU, biến thể thứ hai bắt đầu từ mọi device CUDA và chỉ bỏ device CPU, để GPU nhỏ không bị bỏ
     vì một placement chỉ-CPU trừ khi cái đó thật sự nhanh hơn;
   - vét cạn: với tối đa 8 node, mọi tập con node khả thi có kích thước khả thi nhỏ nhất và lớn hơn một,
     ước lượng nhanh nhất trước; với nhiều node hơn, điền theo thứ tự băng thông (chưa cắt, cắt theo băng
     thông, cắt theo kích thước, cộng thêm một node);
   - loại trùng và giới hạn ở `MAX_MULTI_NODE_CANDIDATES` = 12, để chặn khối lượng chấm điểm.
5. **Chia layer** cho một tập device: tỉ lệ theo dung lượng (usable trừ overhead), mỗi device ít nhất một
   layer, rồi một bước sửa chuyển từng layer từ device quá tải nhất sang device dư nhất, kiểm bằng byte
   từng layer chính xác, tới khi mọi device có `est_mb <= usable_mb`. `tensor_split` = số layer.
6. **Head** = node giữ nhiều layer nhất (tìm bằng điểm bất động: sắp xếp lại, chia lại, lặp). **Thứ tự
   device**: các device CUDA của head, CPU của head, rồi mọi node khác theo tổng usable giảm dần (device
   trong một node theo usable giảm dần). Device không local được đặt tên `RPC0`, `RPC1`...

Pin (`pin_devices`), GPU bị tắt trong pool và VRAM đang được các replica đang launch giữ chỗ đều do
reconciler áp dụng trước khi plan, bằng cách đặt `usable_mb` về 0 hoặc giảm nó, nên bản thân planner không
biết về chúng.

### 6.3 Chấm điểm (`scoring.py`, `placement.py`)

Tốc độ decode ước lượng bị giới hạn bởi băng thông bộ nhớ: mỗi token đọc toàn bộ weight một lần, các layer
trên device khác nhau chạy lần lượt, nên

```
thời gian mỗi token = tổng theo device( byte trên device / (băng thông x 0.5) ) + số_hop_rpc x 2 ms
est_decode_tps = 1 / thời gian mỗi token
```

(0.5 = tỉ lệ băng thông đỉnh mà llama.cpp đạt được; hiệu chỉnh trên GTX 1650, 160 GB/s, nơi Qwen2.5-0.5B
q4_k_m đo được 182 tok/s.) Băng thông chưa biết: CPU 25 GB/s; GPU CUDA không biết được xếp như GPU chậm nhất
đã biết (100 GB/s khi không biết cái nào).

Điểm của một ứng viên (cao hơn thắng):

| Thành phần | Trọng số | Ý nghĩa |
| --- | --- | --- |
| tốc độ | +100 x tps / tps tốt nhất trong các ứng viên | tốc độ tương đối |
| dùng chung | -10 cho mỗi engine đã có trên GPU được chọn, -30 x tỉ lệ bận của nó | ưu tiên GPU rảnh |
| cùng model | -40 cho mỗi replica của cùng model trên GPU được chọn (spread `gpu` hoặc `node`) | dàn trải replica |
| cùng node | -20 cho mỗi replica của cùng model trên node được chọn (chỉ spread `node`) | dàn trải qua các server |
| lãng phí | -15 x trung bình(usable / usable lớn nhất) | best fit: giữ GPU lớn cho model lớn |
| số device | -5 cho mỗi device thêm | ít device hơn |
| hop | -10 cho mỗi chặng mạng | ít liên kết RPC hơn |

Hòa điểm thì xét tier nhỏ hơn, rồi tên node và device, nên kết quả xác định. Ba đến bốn **lý do** đầu (tốc
độ so với phương án nhanh nhất, GPU dùng chung, hàng xóm cùng model, số GPU, số hop mạng) được lưu trong
`Placement.reasons` và hiển thị trên UI. Spread là mềm: GPU dùng chung vẫn được dùng khi không còn chỗ nào
khác. Occupant lấy từ mọi replica live, kể cả đang draining, vì chúng vẫn giữ bộ nhớ.

### 6.4 Giữ chỗ cho draft model

Với `speculative = "draft"`, draft chạy trong `llama-server` của head trên device CUDA local đầu tiên của
head. Với mỗi device head khả dĩ `D`, cách chia được giải trên một pool mà chỉ `D` bị trừ bộ nhớ của draft,
và `D` được ghim làm device đầu tiên của head; tính draft vào mọi device CUDA sẽ loại nhầm những pool thật ra
chứa vừa nó một lần. Trên 16 device CUDA, chỉ 8 cái rộng nhất được thử làm head. MB của draft được cộng
vào `est_mb` của assignment đó và vào `est_total_mb` (`draft_est_mb`). Nếu draft không vừa trên device CUDA
local nào của head thì không có placement.

## 7. Chính sách đa model

Lý do và thuật toán cấp phát nằm trong [PLATFORM_DESIGN.vi.md](PLATFORM_DESIGN.vi.md); đây là hành vi đã cài đặt.

| Field | Mặc định | Tác dụng |
| --- | --- | --- |
| `priority` 0..100 | 50 | Cao hơn được đặt trước trong mỗi tick, nên được VRAM khan hiếm trước; trong cùng priority, mọi model có replica đầu tiên trước khi model nào có replica thứ hai, rồi xếp theo tên. |
| `spread` | `gpu` | `gpu`: tránh GPU đã có replica của model này; `node`: tránh cả server; `none`: không ưu tiên. Chỉ là phạt mềm. |
| `replicas` | 1 | Công tắc bật/tắt: 0 là dừng model bất kể giới hạn. |
| `min_replicas`, `max_replicas` | chưa đặt = `replicas` | Khoảng của autoscaler. Chưa đặt nghĩa là số lượng cố định. |
| `autoscale` | chưa đặt = mặc định | `target_busy` 0.7, `up_after_s` 30, `down_after_s` 300. |
| `idle_unload_s` | chưa đặt | Chỉ với `min_replicas = 0`: gỡ model sau chừng này giây không có request. |
| `preemptible` | true | False: model priority cao hơn không bao giờ được dừng replica của model này. |
| `pin_devices` | rỗng | Các mục `node/device` mà replica được phép dùng; mọi GPU khác coi như không dùng được cho model này, kể cả khi preemption và rebalancing. |

### 7.1 Autoscaler (`coordinator/autoscaler.py`)

Reconciler hỏi `desired(spec)` thay vì đọc `replicas`. Mỗi `poll_s`, autoscaler scrape `/metrics` của
llama-server trên từng head ready (`requests_processing`, `requests_deferred`) và quyết định theo từng model:

- busy = số slot đang xử lý / `parallel`, lấy trung bình trên các replica ready; scrape cũ hơn 3 chu kỳ poll
  thì không tin và dùng số outstanding của router.
- **Scale up** một replica khi có request xếp hàng hoặc busy > `target_busy`, kéo dài `up_after_s`, không có
  replica nào còn đang launch, và desired < max.
- **Scale down** một replica khi không có gì xếp hàng và busy < `target_busy` x 0.5 (độ trễ chống dao
  động), kéo dài `down_after_s`, và desired > sàn.
- Sàn là `max(min_replicas, 1)`; model chưa có trạng thái lưu thì bắt đầu ở đó.
- **Scale về 0**: với `min_replicas = 0` và `idle_unload_s` được đặt, model không có request trong chừng đó
  thời gian và không có gì đang chạy sẽ được gỡ (desired 0). Request kế tiếp gọi `note_request`, đặt desired
  về 1 và đánh thức reconciler (**cold start**). Router giữ request đó tối đa `cold_start_timeout_s` (120 s)
  và poll chờ replica ready, rồi trả 503 kèm `Retry-After: 10`.
- Số lượng mong muốn, thời điểm request cuối và quyết định cuối được ghi vào `control_state` (khóa
  `autoscaler:<model>`) mỗi khi số lượng đổi, nên khởi động lại vẫn giữ model đã gỡ ở trạng thái gỡ và model
  đã scale up ở trạng thái scale up. Bộ đếm up/down cố ý không được lưu (khởi động lại chỉ làm chậm một bước).
  `last_request` được làm mới trong store nhiều nhất 60 s một lần, không phải mỗi request.

### 7.2 Preemption (`preemption.py`, `reconciler.py`)

Khi một model không có placement (`NoFit`) và **dưới mức tối thiểu đang chạy** (`active < min(wanted,
max(min_replicas, 1))`), nó có thể dừng replica khác. Replica autoscale thêm không bao giờ được đuổi ai.

- **Nạn nhân** phải `preemptible` và có priority **thấp hơn hẳn** (priority bằng nhau không bao giờ
  preempt). Thứ tự ứng viên: replica vượt mức tối thiểu của model của nó trước, rồi ít bận nhất, rồi mới
  nhất. Tham lam: thêm nạn nhân tới khi model đặt được trên bộ nhớ đã giải phóng, rồi bỏ nạn nhân nào mà
  phần còn lại làm cho dư thừa. Kết quả là tập nhỏ nhất đủ dùng.
- Nạn nhân được **drain** (không bị kill); preemptor không được đặt ngay trong cùng tick, vì replica đang
  drain vẫn giữ bộ nhớ; tick sau sẽ đặt khi chúng đã dừng.
- **Claim**: trong lúc nạn nhân drain, replica draining không còn là active, nên model của nó thấy thiếu và
  launch lại ngay vào chính bộ nhớ đang được giải phóng (thấy trên phần cứng thật: model bị đuổi quay lại
  ngay và preemptor kẹt). Vì vậy model priority thấp hơn không launch cho tới khi preemptor có replica, hoặc
  claim hết hạn sau `drain_timeout_s` + 120 s.
- **Cooldown**: model đã đuổi người khác không được làm lại trong 600 s, và cũng không khi nạn nhân trước
  đó còn live, để hai model có nhu cầu chồng nhau không thể cứ dừng replica của nhau mãi. Cả cooldown lẫn
  tập nạn nhân đều được lưu.
- Preemption không bao giờ đặt model ở nơi mà launch bình thường không thể đến: pin và GPU bị tắt được áp
  dụng lại sau khi giải phóng bộ nhớ.

### 7.3 Mô phỏng và gợi ý

`POST /api/simulate` chạy cùng thứ tự, placement và luật preemption trên một bản sao của cluster và báo
`start`, `stop`, `preempt` và `unplaced` (kèm đúng câu `NoFit` của planner). Nó bỏ qua cooldown và coi bộ
nhớ do việc dừng giải phóng là dùng được ngay. `POST /api/recommend` xếp hạng các placement cho một file
trong thư viện ở một ctx và tùy chọn cho trước, báo ctx lớn nhất vừa một GPU (tìm nhị phân trên bội số của
256), một phương án đòi hỏi preemption, hoặc câu trả lời `not_possible` kèm các con số. Cả hai không thay đổi gì.

## 8. Tùy chọn hiệu năng

Theo từng model, đặt trong `ModelSpec`, scheduler tính vào kế hoạch và reconciler truyền cho head.

### 8.1 Lượng tử hóa KV cache (`kv_cache_type`: f16 | q8_0 | q4_0)

Thêm `-ctk T -ctv T` cho head (và `-ctkd/-ctvd` cho draft). Ước lượng dùng số byte mỗi phần tử tương ứng, nên
cache lượng tử hóa thật sự cho phép context dài hơn vừa bộ nhớ. Đo trên Qwen2.5-3B ở ctx 8192: q8_0 tiết kiệm
132 MB, q4_0 204 MB so với f16 (lý thuyết 142 / 217 MB).

### 8.2 Speculative decoding (`speculative`: none | ngram | draft)

Ít lượt chạy model đích hơn nghĩa là ít vòng RPC hơn, điều này quan trọng nhất với placement multi-node.

- `ngram`: đoán phần tiếp theo từ văn bản đã có; không tốn thêm bộ nhớ. Cờ `--spec-type ngram-mod`.
- `draft`: một model nhỏ **cùng tokenizer** chạy trên device CUDA local đầu tiên của head. Cờ:
  `--spec-type draft-simple -md <file> -devd <device CUDA> -ngld 999 --spec-draft-n-max N`.
  **Trong b11342, chỉ `-md` thì nạp draft model nhưng không bao giờ dùng nó**, nên `--spec-type draft-simple`
  được đặt tường minh. API từ chối draft có tokenizer khác, hoặc kích thước từ vựng lệch quá 128 token
  (llama.cpp cũng từ chối). Reconciler cũng từ chối launch nếu assignment đầu tiên không phải device CUDA
  local của head.
- `draft_n_max` (1..16, mặc định 4): đo trên GTX 1650 với Qwen2.5-3B cộng draft 0.5B, 4 token nháp cho +5 %,
  8 chậm hơn không dùng.

## 9. Router (`router/`)

- Ứng viên = các replica `ready` của model mà node head còn sống. Router đọc một snapshot của store, chỉ được
  dựng lại khi version của store đổi; tình trạng sống được xét ở mỗi lần gọi, vì node im lặng không gây ra
  lần ghi nào.
- **Prefix key** = sha256 của mọi message trừ cái cuối (JSON chuẩn hóa, cắt ở 4 KB); nếu chỉ có một
  message, lấy 512 ký tự đầu của nó; với `/v1/completions`, lấy 512 ký tự đầu của prompt.
- Rendezvous hash(prefix, replica) chọn replica ưu tiên; nếu nó có nhiều hơn replica rảnh nhất quá 2 request
  đang chờ thì dùng replica rảnh nhất.
- Gửi `cache_prompt: true`; head chạy với `--cache-reuse 256 --metrics`.
- Thử lại khi lỗi kết nối hoặc 5xx **trước byte đầu tiên**, tối đa hai lần, không bao giờ sau khi đã gửi byte.
  Mọi lỗi đều đưa vào `note_error`, khiến reconciler kiểm tra `/health` của replica đó ở tick tiếp theo.
- Lỗi giữa chừng khi stream kết thúc luồng SSE bằng một sự kiện lỗi; bộ đếm outstanding được trả đúng một
  lần, kể cả khi client ngắt kết nối.
- Body lớn hơn `max_request_mb` (32) nhận 413, kể cả khi đang đọc upload chunked.
- Model lạ 404; không có replica ready 503; mọi lần thử đều lỗi 502; model đang nạp từ 0 thì được giữ (mục 7.1).
- `/metrics`: request theo model và mã, số lần retry, tổng và số lượng TTFT, outstanding theo replica, MB
  free và usable theo device, node còn sống không, số replica theo model và trạng thái.

## 10. Reconciler (`coordinator/reconciler.py`)

Một vòng lặp mỗi `reconcile_s` (2 s), hoặc sớm hơn khi `wake()` được gọi sau khi trạng thái mong muốn đổi. API
handler gọi `wake()`, không gọi `tick()`, vì một tick chờ khóa tick và các lời gọi HTTP tới agent. Một tick,
theo thứ tự:

1. **Theo dõi node**: phát `node_offline` / `node_online` một lần cho mỗi lần chuyển trạng thái.
2. **Fail các launch mồ côi** (mục 13).
3. **Phát hiện lỗi**.
4. **Kiểm tra nghi vấn**: các replica mà router báo lỗi; `GET /health` trên head, fail nếu không phải 200.
5. **Xử lý drain**.
6. **Xóa backoff đã ổn định**.
7. **Tiến hành move đang chạy** (mục 10.3).
8. **Thực thi số lượng**: thứ tự priority, preemption, launch, drain phần dư, thay thế replica thiếu bộ nhớ.
9. **Rebalance nếu đến hạn**.
10. **Dọn** các dòng replica terminal (giữ 10 dòng mỗi model).

### 10.1 Phát hiện lỗi

- **Node chết** = báo cáo cũ hơn `heartbeat_timeout_s` (10 s) VÀ có ít nhất 2 lần poll liên tiếp thật sự thất
  bại. Báo cáo cũ một mình có thể là do event loop của chính coordinator bị đứng, nên nó không bao giờ giết
  một node. Một watchdog đo độ trễ vòng lặp ghi cảnh báo khi event loop của coordinator bị chặn quá 1 s.
  Replica dùng node chết chuyển sang `failed`; các engine còn sống của replica đó bị dừng.
- **GPU biến mất**: replica có GPU được gán (khớp theo uuid) mà server còn sống không còn báo nữa sẽ bị fail
  kèm sự kiện `gpu_missing`.
- **Engine sập**: engine báo `exited` / `failed`, hoặc vắng trong một báo cáo mới hơn thời điểm ready của
  replica, làm replica fail (sự kiện `engine_crashed` kèm các dòng log cuối).
- **Realloc**: replica fail ghi nhận một lần cấp phát lại đang chờ; các sự kiện `realloc_started` và
  `realloc_done` mang thời gian model không được phục vụ. Đặt chỗ thất bại phát `realloc_failed`.
- **Backoff**: launch thất bại thì lùi 5 s, 10 s, 20 s... tối đa 300 s cho mỗi model. Replica sập trong vòng
  300 s sau khi ready (`model_fault`) cũng đưa vào backoff này và, từ lần thứ hai, phát `crash_loop`. Replica
  giữ ready được 300 s thì xóa backoff. Node chết hoặc mất GPU không phải lỗi của model nên không được tính.

### 10.2 Số lượng mong muốn, launch và drain

- Theo từng model theo thứ tự priority: ít replica active hơn mong muốn thì launch một cái (plan, rồi tạo
  task, nhiều nhất một replica mới mỗi model mỗi tick); nhiều hơn thì drain cái mới nhất.
- **Giữ chỗ VRAM**: khi plan, `usable_mb` bị giảm bởi ước lượng của các replica đang `launching`, và của các
  replica vừa chuyển `ready` cho tới khi node của chúng gửi báo cáo muộn hơn thời điểm ready hơn 5 s (báo cáo
  cũ hơn có thể có trước lúc nạp model, và cùng VRAM sẽ bị cấp hai lần). Device có `budget_mb` còn bị chặn bởi
  budget trừ ước lượng của mọi replica live trên đó.
- Replica ready trên device có `free_mb < low_free_mb` (256) được thay thế trước: launch một replica mới, và
  drain cái cũ khi cái mới đã ready.
- **Launch**: khởi động các engine rpc (mỗi cái với `allowed_peers` = host của head) và chờ từng cái running;
  `ensure` model (và draft) trên head; khởi động head; chờ `/health` 200 (`launch_timeout_s`, 600 s); đánh dấu
  `ready`; hiệu chỉnh (mục 12). Bất kỳ lỗi nào đều dừng mọi engine đã tạo, nên replica launch dở không bao
  giờ ghim VRAM trên GPU dùng chung.
- Replica bị drain hoặc fail trong lúc launch được rollback lặng lẽ (không tính là launch thất bại).
- **Drain**: chờ outstanding = 0 (tối đa `drain_timeout_s`, 60 s), dừng engine, đánh dấu `stopped`.
- **Gỡ server** (`DELETE /api/servers/{id}`): các replica chạm tới server được đánh dấu `stopped` (không phải
  `failed`) để tick sau chỉ việc đặt lại chúng; các task đang launch bị hủy trước.

### 10.3 Rebalancing

Replica được đặt từng cái một, nên cluster có thể lệch dần: model đặt lúc pool đầy có thể nằm vắt qua một
hop mạng sau khi một GPU lớn được giải phóng. Rebalancing tìm và sửa việc đó.

- **Ứng viên**: mỗi replica `ready` được chấm trong cùng một lượt với các phương án thay thế (`rank(...,
  extra=[placement của nó])`), với bộ nhớ của chính nó bị bỏ khỏi occupant nhưng vẫn bị trừ khỏi báo cáo,
  vì chỗ mới phải vừa **trong khi replica cũ vẫn đang chạy**. Một move cần tăng điểm ít nhất 25 (move nạp lại
  cả model, nên phải tốt hơn rõ rệt chứ không chỉ tốt hơn). Move giữ trong pin của model. Lợi nhất trước.
- **Make-before-break (làm trước, ngắt sau)**: đích được plan với model ghim vào các device đích, một replica
  mới launch, và số lượng mong muốn của model đó tạm tăng thêm một. Khi replica mới ready, cái cũ được drain
  (sự kiện `rebalanced`). Replica cũ phục vụ liên tục suốt thời gian đó.
- **Bỏ dở** (sự kiện `rebalance_failed`, replica cũ không bị đụng tới) khi replica mới fail, replica cũ không
  còn ready, hoặc replica mới chưa ready trong `launch_timeout_s` + 60 s.
- **Mỗi lần một move, trên toàn cluster**, và chỉ khi cluster yên: không có move đang chạy, không replica nào
  pending hay launching, không preemption nào đang chờ bộ nhớ của nó.
- **Khi nào**: mỗi `rebalance_s` (600 s; 0 tắt lần chạy định kỳ), hoặc theo yêu cầu bằng `POST /api/rebalance`
  (`dry_run` true liệt kê các move, false bắt đầu move tốt nhất). Bộ đếm bắt đầu từ lúc khởi động, không từ 0,
  và không được lưu: ngay sau khi restart các báo cáo đã cũ và một move sẽ chỉ là đoán mò.

## 11. Bảo mật

- **API**: bearer token ba cấp (cluster token, admin key, API key). Agent không bao giờ nhận `extra_args` (cluster
  token không được trở thành cờ llama-server tùy ý), và `model_path` / `draft_model_path` phải nằm trong cache
  model của agent hoặc đã được `/models/ensure` trả về. File thư viện được phục vụ theo tên item, không bao
  giờ ghép đầu vào người dùng vào đường dẫn.
- **`ggml-rpc-server` không có xác thực.** Ai chạm tới cổng của nó đều có thể cấp phát bộ nhớ GPU, chạy graph
  và đọc hoặc ghi tensor; và lưu lượng không mã hóa. Hai lớp bảo vệ:
  1. Engine chỉ bind vào `host` của agent (địa chỉ nội bộ của node), không bao giờ bind wildcard. Agent ghi
     cảnh báo nếu địa chỉ đó là public hoặc wildcard mà tường lửa đang tắt.
  2. **Tường lửa RPC** (`rpc_firewall`, `agent/firewall.py`, cần root hoặc `NET_ADMIN`; Docker
     `--cap-add NET_ADMIN`): luật iptables (và ip6tables) cho từng cổng RPC, trong một chain riêng
     `GPUPOOL-RPC` được nhảy tới từ `INPUT` một lần. Với mỗi cổng, theo thứ tự: cho phép loopback, cho phép
     từng `allowed_peers` (head của replica), cho phép **địa chỉ bind của chính agent**, rồi drop mọi thứ còn
     lại. Luật được thêm trước khi tiến trình chạy, nên cổng không bao giờ truy cập được khi chưa được bảo vệ,
     và bị xóa khi engine dừng.
- **Vì sao cho phép địa chỉ của chính agent**: bước thăm dò sẵn sàng kết nối tới địa chỉ bind, và một kết nối
  tới IP của chính mình có nguồn là IP đó (không phải 127.0.0.1). Thiếu luật này, engine không bao giờ trông
  như `running` (thấy trên cluster Docker thật).
- Khi khởi động, agent xả toàn bộ chain: engine của agent trước đã bị dọn, nên mọi luật đều cũ. Nếu iptables
  không có, agent ghi lỗi và chạy không có bảo vệ; một luật bị lỗi thì rollback các luật của cổng đó và để cổng
  không bị hạn chế kèm log lỗi. Trong trường hợp đó hãy tự bảo vệ các cổng 9000-9999 bằng tường lửa riêng.

## 12. Tự hiệu chỉnh VRAM

Công thức ở 6.1 chỉ chính xác trên model nhỏ; model, driver và GPU khác sẽ lệch. Sau khi một replica ready,
reconciler hỏi agent của head về buffer thật (`GET /engines/{id}/memory`, parse từ log `-lv 4`: MiB buffer
model, KV và compute theo device, lần xuất hiện cuối thắng; buffer host và mapped bị bỏ qua) và so sánh:

```
sample = buffer đo được (log của head, cộng qua các device của placement)
         / (tổng est_mb / placement.mem_factor - runtime_context mỗi device
            [- thêm một context nếu draft nằm trên head])
```

`est_mb` đã được nhân với hệ số dùng lúc lập placement (`Placement.mem_factor`), nên phải chia ngược lại:
nếu lấy mẫu trên ước lượng đã nhân thì đo được `tỉ lệ thật / hệ số`, và EMA sẽ hội tụ về căn bậc hai của tỉ lệ
thật (1.2 thay vì 1.44), khiến VRAM bị đặt trước thiếu.
Runtime context bị trừ vì llama.cpp không báo nó như một buffer. Nếu thiếu dữ liệu của bất kỳ device nào trong
placement thì mẫu bị bỏ (dữ liệu thiếu sẽ làm lệch tỉ lệ xuống thấp).

- **EMA**: `factor = 0.5 x sample + 0.5 x trước đó` (mẫu đầu tiên: chính mẫu), lưu dạng thô theo từng model
  trong `model_calibration` cùng số mẫu.
- **Chặn (clamp)**: planning dùng hệ số bị chặn vào khoảng **0.9..2.0**. Một lần đo may mắn không bao giờ
  được thu nhỏ biên an toàn dưới 0.9, và một lần đo bất thường không được làm model không thể đặt chỗ.
- Planner nhân mọi nhu cầu của model đó (và draft của nó) với hệ số; sự kiện `calibrated` được phát khi hệ số đã
  chặn dịch chuyển hơn 5 %. Hệ số và số mẫu hiển thị trong `/api/state`.
- Hiệu chỉnh chỉ là sổ sách về một replica đã phục vụ: nó không bao giờ làm launch thất bại, và agent cũ, log
  không có `-lv 4` hay thiếu device chỉ đơn giản để nguyên hệ số.

## 13. Lưu trạng thái và phục hồi sau sự cố

SQLite (WAL, busy timeout 5 s). Các bảng: `nodes`, `models`, `replicas`, `servers`, `removed_servers`,
`gpu_flags`, `events` (1000 dòng cuối), `control_state`, `model_calibration`, cộng các bảng riêng của library.

| Được lưu | Ở đâu | Sống sót sau restart |
| --- | --- | --- |
| Model, replica, server, server đã gỡ, cờ bật/tắt GPU | các bảng trên | có |
| Sự kiện | `events` | có; gửi webhook là best effort |
| Cooldown preemption và tập nạn nhân, backoff crash-loop, move đang chạy | `control_state` khóa `preempted`, `backoff`, `move` | có |
| Số lượng mong muốn, request cuối, quyết định cuối của autoscaler | `control_state` khóa `autoscaler:<model>` | có (bộ đếm thì không) |
| Hệ số hiệu chỉnh | `model_calibration` | có |
| Bộ đếm `_last_rebalance` | bộ nhớ | không, cố ý (báo cáo cũ sau khi khởi động) |

Control state được ghi ngay mỗi khi đổi (đây là các sự kiện hiếm) và thời gian là giờ đồng hồ thật nên có cùng ý
nghĩa sau restart. Lỗi store khi lưu được ghi log và không bao giờ làm hỏng reconciliation; dòng không đọc được
được coi như không có. Một move đã lưu mà replica của nó không còn tồn tại thì bị xóa lặng lẽ.

**Phục hồi sau sự cố**
- *Coordinator bị kill giữa lúc launch* (kill -9, OOM, mất điện; tắt sạch thì hủy task launch và tự đánh dấu
  failed): bản ghi replica và task launch của nó được tạo cùng nhau, nên bản ghi `launching` mà không có task
  trong tiến trình này nghĩa là coordinator đã chết. `_fail_orphaned_launches` fail nó ở tick đầu tiên và dừng
  các engine có thể đang chạy dở, rồi tick sau plan lại. Thấy trên cluster thật: sau `docker kill`, replica
  kẹt `launching` mãi mãi và, vì được tính là active, ngăn model của nó không bao giờ được launch lại.
- *Agent bị kill*: mỗi engine có một file pid (kèm thời điểm tạo tiến trình, chống tái sử dụng PID); agent mới
  trên cùng `log_dir` dừng các tiến trình llama.cpp còn sót (chúng sẽ cứ giữ VRAM mà không coordinator nào
  biết) và xả chain tường lửa.
- *Agent không liên lạc được*: node chỉ bị coi là chết khi có bằng chứng (mục 10.1); poller đếm các lần poll
  chạy xong với lỗi, và agent báo `node_id` khác với đã đăng ký thì bị bỏ qua.
- *Mất node*: replica fail, engine trên các node còn sống bị dừng, model được đặt lại ở tick sau (chuỗi sự kiện `realloc_*`).

## 14. Cấu trúc code

```
src/gpupool/
  cli.py                      gpupool agent | coordinator | register | plan | scale | undeploy | status
  common/
    models.py                 hợp đồng wire (pydantic)
    config.py                 AgentConfig, CoordinatorConfig, TOML + env (GPUPOOL_*), chuỗi join, secrets
    auth.py                   dependency bearer-token
    net.py                    HTTP client nội bộ và bên ngoài, xử lý proxy
  agent/
    app.py                    FastAPI: /report /engines /models/ensure; tự join; push heartbeat tùy chọn
    procs.py                  dòng lệnh engine, giám sát tiến trình, file pid, dọn tiến trình mồ côi
    firewall.py               chain iptables GPUPOOL-RPC cho các cổng RPC
    memlog.py                 kích thước buffer theo device từ log llama-server
    gpu.py                    dò device NVML / psutil, budget, margin
    models_cache.py           tải vào .part, đổi tên nguyên tử, GGUF chia nhỏ
  scheduler/
    gguf_meta.py              parser header GGUF (file hoặc URL)
    estimate.py               ước lượng bộ nhớ
    scoring.py                ước lượng tốc độ decode theo băng thông
    placement.py              ứng viên, chia layer, chấm điểm, giữ chỗ draft, plan / rank
  coordinator/
    app.py                    ghép FastAPI, /internal/*, /admin/*, /metrics, mount UI, watchdog
    api.py                    /api/* cho web UI
    library.py, library_api.py  thư viện GGUF (tải HF, đường dẫn), /files/{name}
    store.py                  schema SQLite và các hàm truy cập
    poller.py                 kéo /report từ mọi agent đã đăng ký
    reconciler.py             vòng tick, launch, drain, lỗi, preemption, rebalance, hiệu chỉnh
    preemption.py             chọn nạn nhân
    autoscaler.py             metric busy, số lượng mong muốn, cold start, gỡ khi rảnh
    events.py                 sự kiện + notifier webhook
    agent_client.py           HTTP client cho API của agent
  router/
    balancer.py               prefix key, rendezvous hash, bộ đếm outstanding
    proxy.py                  /v1/*, retry, streaming, chờ cold start, metric Prometheus
  ui/                         index.html, app.js, styles.css, vendor/alpine.min.js, favicon.svg
tests/                        test_agent_*  test_scheduler_*  test_coordinator_*  test_router_*
                              test_library_unit  test_net  test_config_env  test_onecmd
                              test_ui_static  test_ci_e2e_static
scripts/                      e2e_local.py  ci_e2e.py  ui_mock_server.py
docker/                       agent.Dockerfile  coordinator.Dockerfile
docker-compose.*.yml          coordinator, agent, sim (cluster 3 server giả lập), ci
docs/                         DESIGN, PLATFORM_DESIGN, API, QUICKSTART, TEST_REPORT, UI_DESIGN (en + vi)
```

## 15. Kế hoạch kiểm thử

Unit test: `uv run pytest` (mặc định loại marker `real`); `uv run pytest -m real` cần binary llama.cpp, một
file GGUF và một GPU. CI chạy bộ unit, rồi build cả hai image Docker.

Quy tắc: test viết trên mock chỉ chứng minh mock chạy được. Mọi lỗi nghiêm trọng đã gặp đều sống sót qua bộ
unit xanh và xuất hiện ở lần chạy thật đầu tiên, nên mỗi tính năng còn được chạy với thực tế ít nhất một lần.

Các lần chạy với engine thật (kết quả trong [TEST_REPORT.vi.md](TEST_REPORT.vi.md)):

- `scripts/e2e_local.py`: máy dev có một GTX 1650 4 GB, nên 3 server được giả lập bằng 3 agent trên
  127.0.0.1/.2/.3: agent A có `CUDA0` thật với `budget_mb` bị chặn để buộc phải chia; agent B và C mở một
  device `CPU` chạy `ggml-rpc-server -d CPU` thật, nên RPC là thật qua TCP.
- `scripts/ci_e2e.py` với `docker-compose.ci.yml`: một coordinator thật và 3 agent thật (image Docker, chỉ CPU)
  phục vụ một model nhỏ chia trên nhiều server; bắt lỗi Dockerfile và Python sai trong image.
- `docker-compose.sim.yml`: cluster 3 server giả lập cho demo; `GPUPOOL_FAKE_DEVICES` làm agent báo các GPU
  khác nhau trong khi dùng chung một card thật.
- `scripts/ui_mock_server.py`: API coordinator trong bộ nhớ để làm việc với UI mà không cần cluster.

| Test | Cách làm | Đạt khi |
| --- | --- | --- |
| Baseline | Qwen2.5-0.5B Q4_K_M trên CUDA0 | ghi lại tokens/s prefill và decode, TTFT |
| Ước lượng so với thực tế | so `est_mb` với buffer đo được | sai số trong 20 % |
| Chia thật | Qwen2.5-3B Q4_K_M, A bị chặn, chia qua B và C | trả lời đúng qua `/v1/chat/completions`, ghi lại tokens/s |
| Failover | kill agent của một replica | các request sau vẫn thành công, replica được đặt lại |
| Preemption, rebalance, autoscaling, cold start | unit test với agent giả (`test_coordinator_preemption`, `_reconciler`, `_autoscaler`) | nạn nhân được chọn và drain, move là make-before-break, scaling theo đúng ngưỡng |
| Tường lửa RPC | unit test với runner iptables giả (`test_agent_firewall`), cộng cluster Docker thật | chỉ head (và chính agent) chạm được cổng RPC; engine đạt `running` |

## 16. Các câu hỏi mở đã được trả lời

- Các server thấy nhau; internet không được đảm bảo, nên có nguồn `coordinator://<file>` và binary llama.cpp
  được truyền qua `llama_dir` hoặc đóng sẵn vào image.
- Không giới hạn kích thước model; mục tiêu là dùng tổng VRAM usable của pool, nên chia multi-node là đường
  chính cho model lớn. Margin cấu hình được theo từng node (`margin_pct`, `margin_min_mb`, `budget_mb`).
