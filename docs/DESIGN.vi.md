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
| Device từ xa | 1 rpc-server mỗi node | 1 rpc-server mỗi node và replica, phục vụ các device liền nhau của replica trên node đó (`-d CUDA0,CUDA1`); 1 cái mỗi device với agent cũ hơn 0.6 | Activation giữa hai GPU của cùng server được copy ngay trong server (`RPC_CMD_COPY_TENSOR`) thay vì đi server → head → server; mỗi replica một cổng riêng, tên `RPCi` xác định (device của một server được đánh số liên tiếp theo thứ tự `-d`). |
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
  truyền gì. Cache nằm ở `$LLAMA_CACHE/rpc` (mỗi tensor trên 10 MiB một file, đặt tên theo hash); agent đặt
  `LLAMA_CACHE=<cache_dir>/llama.cpp` (volume `/data` trong image, trừ khi người dùng đã đặt `LLAMA_CACHE`) và cứ
  10 phút cắt nó về `rpc_cache_gb` (100 GB), xoá file ít dùng nhất trước. llama.cpp không bao giờ xoá file
  cache, và thiếu một file chỉ khiến head gửi lại tensor đó.
- RPC dùng TCP, hoặc RDMA (RoCE / InfiniBand) khi cả hai đầu có thiết bị RDMA: image agent build ggml-rpc với
  libibverbs và llama.cpp tự thương lượng cho từng kết nối (`GGML_RPC_NO_RDMA=1` ép dùng TCP).
- Activation đi qua mạng một lần cho mỗi token mỗi chặng (hop), nên chia qua RPC là đổi tốc độ lấy dung
  lượng. Scheduler mô hình hóa việc này bằng chi phí cố định mỗi hop (mục 6.3) và chỉ dùng placement
  multi-node khi không có GPU đơn hay node đơn nào chứa vừa.
- Mỗi node và replica một `ggml-rpc-server`, phục vụ mọi device liền nhau của replica trên node đó (agent báo
  tính năng `rpc_multi_device`; agent cũ hơn nhận một server mỗi device), bind vào `host` của agent trên một cổng
  trong `port_range` (9000-9999) do coordinator cấp, nên cổng không bao giờ đụng nhau giữa các replica. Bộ lập
  lịch tính một hop mạng cho mỗi server, không phải mỗi device.
- Device CPU của head, nếu được mở (`include_cpu`), cũng được truy cập qua một rpc-server (`-d CPU`), nên nó
  cũng là device `RPCi`.

### 3.2 Head chạy gì

Dòng lệnh của head (`agent/procs.py`, `build_command`) luôn có `-ngl 999`, `--split-mode layer`,
`--cache-reuse 256`, `--metrics`, `--fit off` (scheduler đã chọn cách chia; auto-fit của llama.cpp không
được đổi nó) và `-lv 4` (mức log 4 in kích thước buffer model, KV và compute của từng device để hiệu chỉnh
đọc; mức 5 thêm một lượt dry-run nên phải đúng là 4). `--tensor-split` là số layer, được truyền khi có hơn
một device.

Tuỳ chọn theo model chỉ thêm cờ khi khác mặc định của llama.cpp: `-ctk/-ctv` (kiểu KV cache), `-fa` (flash
attention), `-b` / `-ub` (batch, micro-batch), `-kvu` (`kv_unified`: các slot dùng chung một vùng KV thay vì mỗi
slot `ctx_size / parallel`), và với speculative decoding là `--spec-type ngram-mod`, `--spec-type draft-simple
-md ...` hoặc `--spec-type draft-mtp` (mục 8.2). `--rpc` liệt kê mỗi RPC server một lần, theo thứ tự các device
`RPCi` của nó.

## 4. Hợp đồng dữ liệu

Nguồn sự thật: `src/gpupool/common/models.py` (pydantic v2). Đổi một field là đổi giao thức; agent và
coordinator khác phiên bản phải vẫn nói chuyện được, nên mọi field thêm sau 0.1 đều tùy chọn.

| Model | Hướng | Field chính |
| --- | --- | --- |
| `Device` | agent tới coordinator | `device_id` ("CUDA0", "CPU"), `kind`, `total_mb`, `free_mb`, `usable_mb` = max(0, min(free - margin, budget)), `budget_mb` (mức trần đã cấu hình; coordinator còn trừ ước lượng của các replica đang chạy của chính nó vì free memory không cho thấy phần đó), `uuid` / `pci_bus_id` (định danh ổn định), `bandwidth_gbps` (độ rộng bus NVML x xung nhớ; dùng để xếp hạng GPU), telemetry (`util_pct`, `temp_c`, `power_w`, `processes`, `driver`, `cuda`) |
| `NodeReport` | agent tới coordinator (`GET /report`) | `node_id`, `agent_url`, `host` (IP các node khác dùng cho RPC), devices, engines, `llama_version`, `cuda_archs`, `models` (file GGUF trong cache local), `features` (khả năng ngoài bản 0.5, ví dụ `rpc_multi_device`; coordinator chỉ dùng khả năng mà agent báo), telemetry CPU/RAM |
| `EngineSpec` | coordinator tới agent | `engine_id`, `kind` rpc/server, `port`, `devices` (server: thứ tự = `--device`; rpc: các device một `ggml-rpc-server` phục vụ, theo thứ tự `-d`), `rpc_endpoints` (mỗi RPC server một cái), `tensor_split`, `ctx_size`, `parallel`, `cache_type`, `spec_type` (`none`/`ngram`/`draft`/`mtp`), `draft_model_path`, `draft_device`, `draft_n_max`, `flash_attn`, `batch`, `ubatch`, `kv_unified`, `allowed_peers` (rpc: host được phép kết nối) |
| `EngineStatus` | agent tới coordinator | `state` starting/running/exited/failed, `exit_code`, `log_tail` (tối đa 50 dòng) |
| `ModelSpec` | admin | `name`, `source`, `ctx_size`, `parallel`, `replicas` (0 = dừng), `pin_devices`, `priority`, `spread`, `min_replicas`, `max_replicas`, `autoscale`, `idle_unload_s`, `preemptible`, `kv_cache_type`, `speculative`, `draft`, `draft_n_max`, `flash_attn`, `batch`, `ubatch`, `kv_unified` |
| `AutoscalePolicy` | nằm trong `ModelSpec` | `target_busy` 0.7, `up_after_s` 30, `down_after_s` 300 |
| `ModelMeta` | đọc từ header GGUF | `n_layers`, `n_embd`, `n_head_kv`, `head_dim`, `layer_bytes[i]`, `output_bytes`, `vocab_size`, `tokenizer_model`; bố cục cache theo layer `kv_k[i]` / `kv_v[i]` (kích thước hàng được cache), `swa[i]` + `n_swa`, `state_bytes[i]` (state hồi quy mỗi sequence); `n_nextn` / `nextn_bytes` (block MTP chỉ nạp khi `draft-mtp`); `active_bytes[i]` (MoE: số byte mỗi token đọc) |
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
| API cho Web UI | admin key | `/api/state`, `/api/servers`, `/api/servers/{id}/gpus/{dev}` (bật/tắt một GPU trong pool), `/api/models/{name}` (PUT, start, stop, scaling, plan, delete), `/api/capacity`, `/api/simulate`, `/api/rebalance`, `/api/recommend`, `/api/events`, `/api/library`, `/api/hf/files`, `/api/convert*` (chuyển Hugging Face sang GGUF, mục 17) |
| OpenAI | API key | `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions` (hỗ trợ `stream: true`) |
| Metrics | không | `GET /metrics` (text Prometheus) |

## 6. Scheduler

`src/gpupool/scheduler/`: `gguf_meta.py`, `estimate.py`, `scoring.py`, `placement.py`. Tất cả là hàm thuần
của (metadata model, spec, báo cáo node, occupant); cổng chỉ được cấp cho placement thắng cuộc.

### 6.1 Ước lượng bộ nhớ (`estimate.py`)

Với một device giữ dải layer `L` (đã hiệu chỉnh theo log nạp verbose của llama.cpp b11342):

```
need = ceil( sum(layer_bytes[i] + cache_bytes[i] for i in L) [+ output_bytes nếu là device cuối] )
       + compute_buffer + runtime_context
cache_bytes[i]     = cells[i] x (k_row[i] + v_row[i]) x bytes_per_element(kv_cache_type)
                     + state_bytes[i] x parallel
cells[i]           = ctx_size, hoặc với layer sliding-window
                     parallel x pad256(min(pad256(ctx_size / parallel), n_swa + ubatch))
                     (kv_unified: pad256(min(ctx_size, n_swa x parallel + ubatch)))
compute_buffer     = ceil(21 x 512 x n_embd x 4 bytes)      (ubatch mặc định 512)
runtime_context    = 128 MB (CUDA) | 32 MB (CPU)
```

- Byte mỗi phần tử KV: f16 2, q8_0 34/32, q4_0 18/32 (bố cục block của ggml).
- Bố cục theo từng layer làm theo loader của llama.cpp b11342: `k_row = n_head_kv[i] x key_length`
  (`key_length_swa` ở layer SWA), `v_row` tương tự, bằng 0 với MLA (`key_length_mla`: chỉ cache K latent).
  Layer hồi quy của model lai (`full_attention_interval` cho Qwen3-Next / Qwen3.5, `recurrent_layers`, hoặc
  0 KV head) không có KV nhưng có state f32 cho mỗi sequence, tính từ các khoá `ssm.*`. Layer SWA lấy từ mảng
  `sliding_window_pattern`, hoặc với gemma2/3/3n, gpt-oss, cohere2 và olmo2 từ chu kỳ của chính loader khi có
  `sliding_window`; kiến trúc khác tính như layer đầy đủ (ước lượng dư, không bao giờ OOM). Không có dữ liệu
  bố cục (metadata cũ) thì mọi layer dùng `2 x ctx_size x n_head_kv x head_dim`.
- Các block MTP (`nextn`) mà llama.cpp chỉ nạp khi `--spec-type draft-mtp` được tách riêng (`nextn_bytes`) và
  chỉ tính (cùng cache của chúng) khi bật MTP.
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
   device**: mọi node khác theo tổng usable giảm dần (device trong một node theo usable giảm dần), rồi CPU của
   head, rồi các device CUDA của head **ở cuối**. Device cuối giữ layer đầu ra, và llama-server đọc `n_vocab x 4`
   byte logits từ nó ở mỗi token (0,5 MB với từ vựng 128k); khi GPU của chính head đứng cuối, chỉ hidden state
   (`n_embd x 4` byte) đi qua mạng. Embedding đầu vào luôn được tính trên CPU của head. Device không local được
   đặt tên `RPC0`, `RPC1`... theo thứ tự này.

Pin (`pin_devices`), GPU bị tắt trong pool và VRAM đang được các replica đang launch giữ chỗ đều do
reconciler áp dụng trước khi plan, bằng cách đặt `usable_mb` về 0 hoặc giảm nó, nên bản thân planner không
biết về chúng. Danh sách pin là một **tập được phép**, không phải chỗ đặt: mọi thiết bị nằm ngoài nó bị đặt
`usable_mb = 0` (`Reconciler._apply_pins`), rồi planner chọn cách đặt tốt nhất trong phần còn lại. Mỗi mục là
`"<node>/<device>"` hoặc `"<node>/*"`; ký tự đại diện cho phép mọi thiết bị của server đó, kể cả GPU đăng ký sau, nên
lựa chọn "cả server này" không lỗi thời khi server có thêm GPU. Pin và công tắc bật/tắt GPU của pool độc lập với
nhau: GPU bị bỏ ngoài tập của một model vẫn được bật cho các model khác.

### 6.3 Chấm điểm (`scoring.py`, `placement.py`)

Tốc độ decode ước lượng bị giới hạn bởi băng thông bộ nhớ: mỗi token đọc toàn bộ weight một lần, các layer
trên device khác nhau chạy lần lượt, nên

```
thời gian mỗi token = tổng theo device( byte trên device / (băng thông x 0.5) ) + số_hop_rpc x 2 ms
est_decode_tps = 1 / thời gian mỗi token
```

(0.5 = tỉ lệ băng thông đỉnh mà llama.cpp đạt được; hiệu chỉnh trên GTX 1650, 160 GB/s, nơi Qwen2.5-0.5B
q4_k_m đo được 182 tok/s.) Băng thông chưa biết: CPU 25 GB/s; GPU CUDA không biết được xếp như GPU chậm nhất
đã biết (100 GB/s khi không biết cái nào). Với layer MoE, byte mỗi token là phần weight dùng chung cộng
`expert_used_count / expert_count` của các expert được định tuyến (`ffn_*_exps`).

`số_hop_rpc` là số RPC server mà đồ thị đi qua, không phải số device ở xa: các device liền nhau của một node
dùng chung một `ggml-rpc-server` khi agent của nó báo `rpc_multi_device` (mục 3.1), và llama.cpp copy activation
giữa chúng ngay trong server đó.

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

Hòa điểm thì xét tier nhỏ hơn, rồi tên node head và device đầu tiên của nó, nên kết quả xác định. Ba đến bốn **lý do** đầu (tốc
độ so với phương án nhanh nhất, GPU dùng chung, hàng xóm cùng model, số GPU, số hop mạng) được lưu trong
`Placement.reasons` và hiển thị trên UI. Spread là mềm: GPU dùng chung vẫn được dùng khi không còn chỗ nào
khác. Occupant lấy từ mọi replica live, kể cả đang draining, vì chúng vẫn giữ bộ nhớ.

### 6.4 Giữ chỗ cho draft model

Với `speculative = "draft"`, draft chạy trong `llama-server` của head trên device CUDA local đầu tiên của
head. Với mỗi device head khả dĩ `D`, cách chia được giải trên một pool mà chỉ `D` bị trừ bộ nhớ của draft,
và `D` được ghim làm GPU local đầu tiên của head; tính draft vào mọi device CUDA sẽ loại nhầm những pool thật ra
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
| `pin_devices` | rỗng | Tập được phép: các mục `node/device` hoặc `node/*` (mọi thiết bị của server đó, kể cả GPU thêm sau) mà replica được phép dùng; mọi GPU khác coi như không dùng được cho model này, kể cả khi preemption và rebalancing. Rỗng = tất cả. Nó giới hạn, scheduler vẫn tự chọn. |

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

### 8.2 Speculative decoding (`speculative`: none | ngram | draft | mtp)

Ít lượt chạy model đích hơn nghĩa là ít vòng RPC hơn, điều này quan trọng nhất với placement multi-node.

- `ngram`: đoán phần tiếp theo từ văn bản đã có; không tốn thêm bộ nhớ. Cờ `--spec-type ngram-mod`.
- `draft`: một model nhỏ **cùng tokenizer** chạy trên device CUDA local đầu tiên của head. Cờ:
  `--spec-type draft-simple -md <file> -devd <device CUDA> -ngld 999 --spec-draft-n-max N`.
  **Trong b11342, chỉ `-md` thì nạp draft model nhưng không bao giờ dùng nó**, nên `--spec-type draft-simple`
  được đặt tường minh. API từ chối draft có tokenizer khác, hoặc kích thước từ vựng lệch quá 128 token
  (llama.cpp cũng từ chối). Reconciler cũng từ chối launch nếu assignment đầu tiên không phải device CUDA
  local của head.
- `mtp`: chính các block dự đoán nhiều token (`nextn`) của model nháp token, trong một context llama.cpp thứ
  hai trên cùng các device. Cờ: `--spec-type draft-mtp --spec-draft-n-max N`. llama.cpp chỉ nạp các block này ở
  chế độ này, nên ước lượng chỉ cộng chúng (ở đúng vị trí layer trong split), KV của chúng và một compute buffer
  thứ hai trên device cuối khi có `mtp`. API từ chối `mtp` với GGUF không có `nextn_predict_layers`.
- `draft_n_max` (1..16, mặc định 4): đo trên GTX 1650 với Qwen2.5-3B cộng draft 0.5B, 4 token nháp cho +5 %,
  8 chậm hơn không dùng.

### 8.3 Vùng KV dùng chung (`kv_unified`)

Với `parallel` > 1, llama.cpp cho mỗi slot một luồng riêng `ctx_size / parallel` ô, nên một request dài thất bại
trong khi các slot khác giữ request ngắn. `kv_unified` thêm `-kvu`: một vùng `ctx_size` ô cho mọi slot, nên bất kỳ
request nào cũng có thể dùng hết. Bộ nhớ không đổi; chỉ layer sliding-window đổi, từ `parallel` cửa sổ
`n_swa + ubatch` thành một cửa sổ `n_swa x parallel + ubatch` (mục 6.1). Recommend gợi ý nó (tip `kv_unified`) cho
mọi model có nhiều slot.

## 9. Router (`router/`)

- Ứng viên = các replica `ready` của model mà node head còn sống. Router đọc một snapshot của store, chỉ được
  dựng lại khi version của store đổi; tình trạng sống được xét ở mỗi lần gọi, vì node im lặng không gây ra
  lần ghi nào.
- **Prefix key** = sha256 của JSON chuẩn hóa cắt ở 4 KB: với một lượt (các message system + một message
  user), mọi message trừ cái cuối, để các request chung system prompt gặp nhau trên một replica; với hội thoại
  nhiều lượt, các message system cộng message user đầu tiên, phần mọi lượt sau đều lặp lại, để hội thoại ở
  cùng KV cache của nó. Nếu chỉ có một message, lấy 512 ký tự đầu của nó; với `/v1/completions`, lấy 512 ký
  tự đầu của prompt.
- Rendezvous hash(prefix, replica) chọn replica ưu tiên; nếu nó có nhiều hơn replica rảnh nhất quá 2 request
  đang chờ thì dùng replica rảnh nhất.
- Gửi `cache_prompt: true`; head chạy với `--cache-reuse 256 --metrics`. Body được chuyển tiếp đúng như client
  gửi: khi thiếu `cache_prompt`, `"cache_prompt":true,` được chèn ngay sau dấu ngoặc mở thay vì serialize lại hàng
  MB JSON trên event loop (body có BOM hoặc UTF-16 thì được mã hoá lại).
- HTTP client của router không giới hạn số kết nối tới upstream (httpx sẽ giữ request thứ 101 trở đi trong
  coordinator mà không có timeout); giới hạn là các slot của llama-server và bộ cân bằng tải.
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
- **Launch**: `ensure` model (và draft) trên head trong lúc các engine rpc khởi động (mỗi cái với
  `allowed_peers` = host của head) và chuyển sang running; khi cả hai xong thì khởi động head; chờ `/health` 200 (`launch_timeout_s`, 600 s); đánh dấu
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
- **Phiên bản bộ ước lượng**: một hệ số được đo theo một phiên bản của ước lượng. Khi `estimate.py` đổi cách ước
  lượng, `ESTIMATOR_VERSION` (trong `store.py`) được tăng và store xoá mọi hệ số một lần lúc khởi động (khoá
  `control_state` là `estimator`); nếu không, hệ số cũ sẽ nhân ước lượng mới với sai số của cái cũ (bị chặn ở
  0.9, tức thiếu tới 10 %). Phiên bản 2: bố cục cache theo layer.

## 13. Lưu trạng thái và phục hồi sau sự cố

SQLite (WAL, `synchronous=NORMAL`, busy timeout 5 s). NORMAL không bao giờ làm hỏng database WAL; mất điện chỉ có
thể làm mất vài commit cuối, còn riêng báo cáo của agent đã commit mỗi 2 s cho mỗi server trên chính event loop
chuyển tiếp suy luận, mà FULL thì fsync từng lần. Các bảng: `nodes`, `models`, `replicas`, `servers`, `removed_servers`,
`gpu_flags`, `events` (1000 dòng cuối), `control_state`, `model_calibration`, `convert_jobs` (mục 17.6), cộng các bảng riêng của library.

| Được lưu | Ở đâu | Sống sót sau restart |
| --- | --- | --- |
| Model, replica, server, server đã gỡ, cờ bật/tắt GPU | các bảng trên | có |
| Sự kiện | `events` | có; gửi webhook là best effort |
| Cooldown preemption và tập nạn nhân, backoff crash-loop, move đang chạy | `control_state` khóa `preempted`, `backoff`, `move` | có |
| Số lượng mong muốn, request cuối, quyết định cuối của autoscaler | `control_state` khóa `autoscaler:<model>` | có (bộ đếm thì không) |
| Hệ số hiệu chỉnh | `model_calibration` | có, cho tới khi phiên bản bộ ước lượng đổi (mục 12) |
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
    rpc_cache.py              giới hạn dung lượng cache trọng số của ggml-rpc-server (LRU)
  scheduler/
    gguf_meta.py              parser header GGUF (file hoặc URL), bố cục cache theo layer, kích thước MoE / MTP
    estimate.py               ước lượng bộ nhớ
    scoring.py                ước lượng tốc độ decode theo băng thông
    placement.py              ứng viên, chia layer, chấm điểm, giữ chỗ draft, plan / rank
  coordinator/
    app.py                    ghép FastAPI, /internal/*, /admin/*, /metrics, mount UI, watchdog
    api.py                    /api/* cho web UI
    library.py, library_api.py  thư viện GGUF (tải HF, đường dẫn, file đã chuyển đổi), /files/{name}
    convert_api.py            /api/convert/* (9 route của tính năng chuyển đổi)
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
  converter/                  chuyển Hugging Face -> GGUF (mục 17)
    models.py                 hợp đồng API: SourceSpec, InspectResult, ConvertRequest, ConvertJob, Validation
    quant.py                  bảng lượng tử hoá, ước lượng dung lượng/VRAM, đề xuất, tên file đầu ra
    source.py                 HfClient, chọn file, inspect
    jobs.py                   ConvertManager: job trong SQLite, worker duy nhất, pipeline, dọn dẹp
    toolchain.py              tìm công cụ, dựng câu lệnh, run_tool, kiến trúc được hỗ trợ
    validate.py               kiểm tra header, tokenizer và sinh văn bản
    hf_tokenize.py            script độc lập chạy dưới Python của bộ chuyển đổi (id token HF)
  ui/                         index.html, app.js, styles.css, vendor/alpine.min.js, favicon.svg
tests/                        test_agent_*  test_scheduler_*  test_coordinator_*  test_router_*
                              test_library_unit  test_net  test_config_env  test_onecmd
                              test_ui_static  test_ci_e2e_static  test_converter_*  test_coordinator_convert_api
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
  Các giai đoạn cuối chạy ba lần chuyển đổi ngay trong coordinator (tổng cộng 33 phép kiểm tra): một model Hugging
  Face (SmolLM2-135M-Instruct) sang `Q4_K_M`, phục vụ chia qua RPC kèm một chat; một nguồn là thư mục sang `Q8_0`
  (thư mục nguồn phải không bị đổi); và `IQ2_XS`, loại cần importance matrix, kiểm tra các giai đoạn và cờ ma trận
  (`--skip-convert` bỏ các giai đoạn này).
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

## 17. Chuyển Hugging Face sang GGUF (`converter/`)

Chỉ file GGUF mới serve được, nhưng nhiều model chỉ được phát hành dưới dạng trọng số safetensors / PyTorch. Vì vậy
coordinator chạy các **job chuyển đổi**: file nguồn (một repo Hugging Face hoặc một thư mục) →
`convert_hf_to_gguf.py` của llama.cpp → `llama-quantize` → kiểm tra → thư viện model. Cách dùng:
[QUICKSTART.vi.md](QUICKSTART.vi.md#serve-model-chưa-có-gguf-chuyển-đổi); các route:
[API.vi.md](API.vi.md#12-chuyển-đổi-apiconvert).

Bộ công cụ là tuỳ chọn. `Toolchain.problem()` trả về những gì còn thiếu (`GPUPOOL_CONVERT_DIR`,
`GPUPOOL_CONVERT_PYTHON`, `GPUPOOL_LLAMA_TOOLS_DIR`); khi thiếu thì mọi thứ khác vẫn chạy và `POST /api/convert`
trả 503 kèm lời giải thích đó. `convert_api.py` import converter một cách lười (lazy) nên router vẫn nạp được trong
cả hai trường hợp. `llama-imatrix` là phần tuỳ chọn duy nhất trong bộ công cụ (`Toolchain.has_imatrix()`, được báo
là `imatrix_available` bởi `GET /api/convert/options`), xem 17.8.

| Module | Việc |
| --- | --- |
| `converter/models.py` | các hợp đồng pydantic: `SourceSpec`, `InspectResult`, `QuantOption`, `ConvertRequest`, `ConvertJob`, `Validation`, các tập trạng thái. Mọi trường đều là một phần của API HTTP và UI |
| `converter/quant.py` | bảng lượng tử hoá (bpw, tier, ghi chú chất lượng), ước lượng dung lượng và VRAM, `recommend`, `plan_steps`, đặt tên file đầu ra |
| `converter/source.py` | `HfClient` (danh sách file, thông tin model, `config.json`, tải về), `select_files`, `local_files`, `inspect_source` |
| `converter/jobs.py` | `ConvertManager`: bảng job trong SQLite, worker duy nhất, pipeline, dọn dẹp |
| `converter/toolchain.py` | vị trí các công cụ, hàm dựng câu lệnh, `run_tool` (ưu tiên thấp, kill cả cây tiến trình), danh sách kiến trúc được hỗ trợ |
| `converter/data/calibration.txt` | văn bản hiệu chỉnh có sẵn cho importance matrix (dữ liệu của package, kèm README giải thích nguồn gốc) |
| `converter/validate.py` | các bước kiểm tra header, tokenizer và sinh văn bản |
| `converter/hf_tokenize.py` | chạy *dưới Python của bộ chuyển đổi*: id token của các đoạn văn thử theo Hugging Face (gpupool không được cài ở đó) |
| `coordinator/convert_api.py` | 9 route, ánh xạ `ConvertError.status` sang HTTP |

### 17.1 Pipeline

```
queued → downloading → converting → [calibrating →] quantizing → validating → done
                                                                  └→ needs_review → (accept) → done
(mọi trạng thái đang hoạt động) → failed | cancelled
```

`calibrating` chỉ có ở job tính importance matrix (17.8). Khi job lỗi hoặc bị huỷ, giai đoạn nó đang ở được lưu thành
`failed_stage`, để UI đánh dấu được bước đó và người đọc API phân biệt được lỗi tải với lỗi đĩa hay lỗi hiệu chỉnh mà
không phải phân tích câu thông báo.
1. **File nguồn.** Với Hugging Face, danh sách cây thư mục cho tên và dung lượng; `select_files` giữ `config.json`, các
   file tokenizer và trọng số, bỏ mọi thứ khác. Safetensors được ưu tiên hơn `pytorch_model*.bin` khi có cả hai;
   `consolidated.*` (bản trọng số gốc của Mistral, trùng với bản kia và từng làm gấp đôi lượng tải) bị bỏ qua; `*.py`
   chỉ được giữ khi có `allow_remote_code`; tên có thể thoát ra ngoài thư mục tải (tuyệt đối, `..`, dấu gạch ngược,
   ký tự ổ đĩa, thư mục ẩn hoặc `onnx`/`openvino`/...) bị loại, và `_check_rel` kiểm tra lại phần đã chọn trước khi dùng.
2. **Tải về** (chỉ Hugging Face). Mỗi lần một file vào `.part`, kiểm tra dung lượng so với danh sách và
   `Content-Length`, rồi đổi tên nguyên tử; việc ghi chạy trong một thread để event loop (vốn còn phục vụ inference)
   không bao giờ bị chặn. File đã có trong cache với đúng dung lượng thì không tải lại.
3. **Staging.** Các file đã chọn được liên kết (symlink, không được thì hardlink, không nữa thì copy) vào
   `.convert/<job>/src`.
4. **Chuyển đổi.** `convert_hf_to_gguf.py src --outfile <outtype>.gguf --outtype <t>` theo `plan_steps`: `F16`, `BF16` và
   `Q8_0` do bộ chuyển đổi ghi trực tiếp; mọi loại khác trước hết ghi một bản trung gian 16 bit (`auto` = bf16 nếu
   trọng số là bf16, còn lại f16).
5. **Hiệu chỉnh** (chỉ khi có importance matrix, 17.8). `llama-imatrix` chạy trên bản trung gian 16 bit và ghi
   `imatrix.gguf` vào thư mục tạm của job.
6. **Lượng tử hoá.** `llama-quantize [--imatrix imatrix.gguf] [cờ] bản-trung-gian out.gguf LOẠI [luồng]`; bản trung gian bị xoá ngay. Các cờ
   nâng cao được đối chiếu với danh sách tên kiểu ggml cho phép, nên giá trị người dùng nhập không thể trở thành một
   đối số dòng lệnh thừa.
7. **Kiểm tra** (17.4). **Xuất bản**: chuyển file tới `models_dir/<name>` và đăng ký (`LibraryItem.source`
   `"convert"`), rồi dọn dẹp, và chỉ sau đó mới báo `done`.

Tiến độ lấy từ chính output của các công cụ: phần trăm tqdm cho bước tải và chuyển đổi, các dòng `[ i/ n]` của
llama-quantize; thanh tiến độ vẽ lại trên một dòng chỉ giữ trạng thái mới nhất, và dòng trong SQLite được ghi tối đa
mỗi giây một lần. 200 dòng cuối được giữ trong bộ nhớ và 50 dòng trong cơ sở dữ liệu.

### 17.2 Một worker, FIFO, ưu tiên thấp

Một worker asyncio lấy mỗi lần một job, theo thứ tự gửi. Lý do: chuyển đổi và lượng tử hoá dùng mọi CPU cùng nhiều RAM
và đĩa, chạy hai cái cùng lúc chỉ làm chậm nhau và có thể làm đầy đĩa; và coordinator còn phải phục vụ router. Mọi công
cụ được khởi chạy trong nhóm tiến trình riêng với ưu tiên thấp (`BELOW_NORMAL_PRIORITY_CLASS` trên Windows, `nice 10`
ở nơi khác). Huỷ job sẽ kill cả cây tiến trình (bộ chuyển đổi tạo tiến trình con), chờ các việc thread đang chạy, và chỉ
sau đó mới xoá thư mục tạm, nên job bị huỷ không để lại tiến trình hay file đang mở. Bộ chuyển đổi chạy với
`HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1`.

Tên được kiểm tra khi gửi (thư viện, một file trong `models_dir`, một job chưa xong) và không có `await` nào giữa
bước kiểm tra và bước chèn, nên hai lần gửi không thể cùng qua; `retry` và `accept` kiểm tra lại.

### 17.3 Cache và đĩa

Bản tải về nằm ở `models_dir/.hf/<owner>__<name>@<revision>/`, dùng chung cho retry và cho các job khác của cùng model
và revision (nhiều loại lượng tử hoá được xếp hàng cùng lúc, hoặc các lần sau nếu `keep_source` đã giữ file): file đã
có ở đó thì không tải lại. Cache bị bỏ khi job kết thúc, trừ khi đặt `keep_source` hoặc một job đang hoạt động khác
cần nó; job lỗi hoặc đã huỷ giữ nó cho lần retry, và xoá job đó sẽ giải phóng nó. Thư mục tạm của một job là
`models_dir/.convert/<job>/`.

Dung lượng đĩa trống được kiểm tra hai lần. Trước hết **lúc gửi** (`_early_disk_check`): từ phần chưa có trong cache
của lượt tải (file đã có trong cache không tính), bản trung gian và đầu ra ước lượng cộng biên; khi đĩa rõ ràng không
chứa nổi job thì `POST /api/convert` trả 507 và không xếp gì vào hàng đợi. Nếu không có bước này, cùng job đó sẽ
được nhận, chờ trong hàng đợi, tải vài phút rồi mới lỗi trong worker. Lần hai là **trước mỗi giai đoạn nặng** trong
worker, với biên 512 MB, vì dung lượng trống có thể giảm khi các job chờ: trước khi tải
(`phần còn phải tải + bản trung gian + đầu ra`), trước khi chuyển đổi (`bản trung gian + đầu ra`) và trước khi lượng
tử hoá (`đầu ra`). Bản trung gian được ước lượng là `params × 2` byte (×4 với f32), hoặc bằng dung lượng nguồn khi chưa
biết số tham số. Lỗi sớm kèm con số còn hơn một file nhiều GB ghi dở; bước kiểm tra phát `ConvertError` với mã 507,
worker biến nó thành job *failed* có `error` giải thích.

### 17.4 Kiểm tra, và vì sao chính sách là như vậy

`validate.py` không bao giờ ném lỗi vì một model xấu; vấn đề được ghi vào kết quả `Validation` và `jobs.py` áp dụng
chính sách:

| Kết quả | Hệ quả | Vì sao |
| --- | --- | --- |
| `header_ok` false (GGUF không đọc được, không có kiến trúc, không có tokenizer) | job **failed** | file như vậy hoàn toàn không serve được; không xuất bản gì |
| `tokenizer_ok` false hoặc `generation_ok` false | **needs_review**, giữ file, chưa vào thư viện | có thể là lỗi của bộ chuyển đổi hoặc một sai khác chấp nhận được: chỉ con người mới phân biệt được, nên quyết định (chấp nhận hay xoá) thuộc về họ. Tokenizer sai âm thầm làm câu trả lời kém đi, vì thế nó mới được kiểm tra |
| một bước kiểm tra không chạy được (`null`) | chỉ cảnh báo | không kiểm tra được (không đủ RAM để chạy CPU, tokenizer Hugging Face không nạp được, hết thời gian chờ) không nói lên điều gì chống lại model, và chặn vì điều đó sẽ làm coordinator nhỏ không dùng được |

Bước kiểm tra tokenizer tách token 8 đoạn văn thử cố định (tiếng Anh, tiếng Việt, code, số, emoji có chuỗi ZWJ, khoảng
trắng lạ, dòng trống, CJK) bằng Hugging Face (`hf_tokenize.py`, `add_special_tokens=False`) và bằng
`llama-tokenize --no-bos --no-parse-special --no-escape -f <file>` trên file GGUF, rồi so sánh các id. Đó là những chỗ
tokenizer sau khi chuyển đổi thường sai. Bước sinh văn bản chạy `llama-simple -ngl 0` (luôn trên CPU, để dành GPU cho
inference) 16 token cho câu "The capital of France is"; nó bị bỏ qua khi RAM trống dưới 1.2 × dung lượng file, và thời
gian chờ tối đa là `120 s + 60 s mỗi GB`. File GGUF được đọc bằng `gguf` và mọi đối tượng dẫn xuất được bỏ đi trước khi
trả về, vì trên Windows một memory map còn sót sẽ chặn việc di chuyển file sau đó.

### 17.5 Ước lượng dung lượng và VRAM

`inspect` liệt kê mọi loại kèm dung lượng file ước lượng, VRAM (file + KV cache f16 ở ngữ cảnh 4096 + 300 MB) và việc
nó vừa một GPU hay vừa pool. Phiên bản đầu dùng một giá trị bit-trên-trọng-số trung bình cho mỗi loại, lấy từ dung
lượng llama-quantize công bố cho Llama-3-8B; trên Qwen2.5-0.5B nó ra **thấp hơn 24 %**. Thiếu hai yếu tố:

- **Embedding.** llama-quantize giữ ma trận token embedding và ma trận đầu ra gần 8 bit ở các loại bit thấp. Chúng
  chiếm khoảng 13 % Llama-3-8B nhưng 28 % Qwen2.5-0.5B. `estimate_bytes` tính chúng riêng
  (`vocab × hidden`, gấp đôi khi embedding vào và ra không dùng chung, ở 8.5 bpw) và suy ra bpw của phần trọng số còn
  lại từ giá trị trung bình tham chiếu của cả model.
- **Phương án dự phòng của K-quant.** K-quant và `IQ4_XS` cần các hàng có số giá trị chia hết cho 256. Khi hidden size
  không như vậy (Qwen2.5-0.5B: 896, SmolLM2-135M: 576), llama-quantize dự phòng theo từng tensor sang một loại cũ hơn
  (`Q4_K` → `Q5_0`, `Q5_K` → `Q5_1`, `Q6_K` → `Q8_0`, các loại còn lại → `IQ4_NL`), nên bảng có bpw của các loại dự
  phòng đó (4.5 đến 8.5).

Với cả hai yếu tố, ước lượng sai lệch khoảng 2 % so với file thật (Qwen2.5-0.5B `Q4_K_M`: ước lượng 390.7 MB, thật
397.8 MB; SmolLM2-135M: 103.1 so với 105.5 MB). Cùng ước lượng đó được lưu trên job (`est_output_bytes`) và điều khiển
các bước kiểm tra đĩa.

Phần đề xuất (`quant.recommend`) đi theo bậc thang `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M` và lấy loại đầu tiên vừa một GPU,
rồi loại đầu tiên vừa pool (chia qua RPC chậm hơn, và lý do có nói rõ). Model dưới 3 tỷ tham số chỉ dùng `Q8_0`, `Q6_K`,
`Q5_K_M`, vì model nhỏ mất chất lượng nhanh nhất. Khi không có GPU thì không có thông tin vừa/không vừa và áp dụng giá
trị mặc định theo kích thước (`Q8_0` dưới 3 tỷ, `Q5_K_M` dưới 15 tỷ, còn lại `Q4_K_M`). Các loại `IQ` (`IQ1_*`, `IQ2_*`,
`IQ3_*`) có trong danh sách lựa chọn nhưng không bao giờ là mặc định; phần đề xuất chỉ nhắc tới chúng khi ngay cả
`Q4_K_M` cũng không vừa, kèm ghi chú rằng chúng cần importance matrix.

Các dòng IQ đã được đối chiếu với file thật, vì bpw của chúng không phải số danh nghĩa của định dạng (llama-quantize
giữ ma trận đầu ra và các tensor nhạy nhất ở loại cao hơn, nên cả file lớn hơn). Bảng dùng trung bình cả file và ước
lượng trong hộp thoại lệch +6 % (`IQ2_XS`), -4 % (`IQ3_M`) và -1 % (`Q8_0`) so với file thật. Các loại IQ khối 256 cũng
lùi về `IQ4_NL` (4.5 bpw) cho các hàng không chia hết cho 256, giống K-quant.

### 17.6 Lưu trạng thái, khởi động lại và ranh giới dọn dẹp

Các dòng job nằm trong cơ sở dữ liệu SQLite của coordinator (bảng `convert_jobs`, kết nối riêng giống library, WAL), ghi
mỗi lần đổi trạng thái và tối đa mỗi giây một lần cho tiến độ. Khi khởi động, các job mà tiến trình trước để ở trạng
thái hoạt động khác `queued` được đưa về `queued` và đặt lại tiến độ (đầu ra dang dở bị bỏ, bản tải đã xong vẫn ở
cache) rồi worker chạy; một lượt quét xoá các thư mục tạm không job nào sở hữu. Tắt máy sạch sẽ huỷ worker và để dòng
ở trạng thái hoạt động, nên lần khởi động sau sẽ xếp lại hàng đợi.

Phạm vi xoá được giữ rất hẹp có chủ đích:

- Chỉ các đường dẫn **bên trong** `models_dir/.convert` và `models_dir/.hf` mới bị xoá (`_rmtree` phân giải đường dẫn
  và từ chối mọi thứ khác, không bao giờ đi theo liên kết). Không đụng tới thứ gì khác trong `models_dir`, trừ file
  thư viện của một mục `convert` bị xoá, và chỉ khi đó là một file thường nằm trực tiếp trong `models_dir`.
- Thư mục được đưa vào làm nguồn không bao giờ bị sửa: bộ chuyển đổi đọc một thư mục staging chứa liên kết, và
  `local_files` không đi vào các thư mục là symlink.
- Job `needs_review` chỉ giữ file đầu ra trong `.convert/<job>/`; cache của nó được giải phóng.
- Nếu đăng ký file vào thư viện lỗi sau khi đã chuyển, file bị xoá lại, nên không có file chưa đăng ký nào nằm dưới một
  tên đã bị chiếm. Sự cố dọn dẹp sau một lần xuất bản thành công chỉ được ghi log và không làm job lỗi.

### 17.7 Bảo mật

`convert_hf_to_gguf.py` nạp tokenizer với `trust_remote_code=True`, tức là thực thi các file `*.py` nằm cạnh trọng
số. Mã Python của một repo vì thế sẽ chạy bên trong coordinator, với quyền của nó. Cho nên các file `*.py` của repo
không bao giờ được tải hay đưa vào staging trừ khi yêu cầu đặt `allow_remote_code`, bộ chuyển đổi chạy ngoại tuyến và
chỉ thấy thư mục staging, và UI đánh dấu tuỳ chọn này là nguy hiểm. Cùng cờ đó được truyền cho tokenizer Hugging Face
của bước kiểm tra. Image chạy bộ công cụ từ `/opt` (bộ chuyển đổi của llama.cpp và một venv PyTorch chỉ dùng CPU, cùng
các bản build CPU tĩnh của `llama-quantize`, `llama-tokenize`, `llama-simple`, `llama-imatrix`); `WITH_CONVERT=0` bỏ toàn bộ chúng.

### 17.8 Importance matrix và giai đoạn hiệu chỉnh

**Là gì và vì sao.** Lượng tử hoá ít bit chỉ có vài mức cho mỗi trọng số, nên việc trọng số nào được mức chính xác là
quan trọng. Importance matrix là ước lượng theo từng trọng số về mức ảnh hưởng của nó tới đầu ra trên văn bản điển hình,
đo bằng cách cho model chạy (`llama-imatrix`) qua một văn bản hiệu chỉnh; `llama-quantize --imatrix` sau đó dồn độ
chính xác vào chỗ quan trọng. llama-quantize b11342 thậm chí từ chối các loại IQ1, IQ2, `IQ3_XXS` và `IQ3_XS` nếu thiếu
nó (file `IQ2_M` và `IQ3_XS` chứa tensor `IQ2_XS` / `IQ3_XXS`), đó là `QuantOption.needs_imatrix`. Ma trận được tính từ
**bản trung gian 16 bit**, không phải từ file đã lượng tử hoá, nên nó thấy các kích hoạt thật của model; CPU chạy nó
(`-ngl 0`, để GPU cho inference, giống phần còn lại của bộ công cụ) ở ưu tiên thấp.

**Chính sách** (`quant.imatrix_wanted`, `ConvertManager._decide_imatrix`, quyết định lúc gửi và lưu thành
`imatrix_used`):

| `imatrix` | Loại do bộ chuyển đổi ghi (`F16`, `BF16`, `Q8_0`) | Loại bắt buộc có ma trận | Các loại khác |
| --- | --- | --- | --- |
| `auto` | không | có | có khi `bpw` < 4.0, không thì không |
| `on` | không | có | có |
| `off` | không | **422**, bộ lượng tử hoá sẽ lỗi | không |

Vì sao `auto` là "bắt buộc hoặc dưới 4 bit trên mỗi trọng số": lợi ích tăng khi số bit giảm (càng ít mức thì việc chọn
trọng số nào giữ chính xác càng quan trọng), còn từ 4 bit trở lên mức cải thiện chất lượng nhỏ mà cái giá lớn:
hiệu chỉnh là giai đoạn chậm nhất trên CPU với khoảng cách xa (đo được, Qwen2.5-1.5B sang `IQ3_M`: 1175 giây trên
1687 giây). Vì vậy từ `Q4_K_M` trở lên không bị làm chậm theo mặc định, và `on` vẫn dành cho ai muốn. `Q8_0`, `F16` và
`BF16` không có bước llama-quantize để đưa ma trận vào.

**Công cụ là tuỳ chọn, và lỗi được báo ở chỗ giải thích được.** `llama-imatrix` là phần duy nhất của bộ công cụ mà một
coordinator có thể thiếu trong khi phần còn lại vẫn chạy (bản cài tự build; `imatrix_available` là false và UI vô hiệu
hoá các loại cần nó). Job không thể chạy nếu thiếu nó, tức loại bắt buộc có ma trận hoặc `on`, bị từ chối ngay lúc gửi
với 503 và thông báo rõ ràng, không phải vài phút sau bên trong `llama-quantize`. Job chỉ hưởng lợi từ nó (`auto` với
loại như `Q3_K_S`) thì chạy không có nó, lặng lẽ, vì làm hỏng một lần chuyển đổi chỉ vì thiếu một cải tiến không bắt
buộc còn tệ hơn việc thiếu cải tiến đó. Cùng lập luận đó khiến văn bản hiệu chỉnh sai bị từ chối lúc gửi (422: không phải
`.txt`, rỗng, quá 20 MB) và văn bản có sẵn bị thiếu bị từ chối với 503.

**Văn bản hiệu chỉnh và vì sao nó là văn bản gốc.** Các bộ hiệu chỉnh công khai thông dụng là bản trích bách khoa tiếng
Anh (`wikitext`) hoặc các tập cộng đồng (như `calibration_datav3`) có giấy phép không rõ hoặc share-alike, điều mà một
dự án phát hành và phân phối lại tệp không nên gánh. Văn bản còn là một quyết định thiết kế: ma trận tính trên văn bản
hẹp (một ngôn ngữ, không có code) khiến bộ lượng tử hoá bảo vệ các trọng số mà văn bản đó dùng và cẩu thả với phần còn
lại, nên model hiệu chỉnh trên văn xuôi tiếng Anh có thể kém đi khi dùng tiếng Việt hay code. Vì vậy gpupool đi kèm
`converter/data/calibration.txt` (khoảng 115 KB, viết riêng cho gpupool, không có dữ liệu cá nhân thật): văn xuôi tiếng
Anh nhiều thể loại, tiếng Việt có dấu đầy đủ, các ngôn ngữ khác, code nhiều ngôn ngữ, toán và dữ liệu có cấu trúc, hội
thoại dạng chat, và nội dung biên (emoji, khoảng trắng lạ, nhiều hệ chữ). README đi kèm nêu thành phần và giấy phép (của
dự án). Có thể đưa một tệp `.txt` khác tối đa 20 MB qua `advanced.calibration_path` (dịch như đường dẫn thư viện); dù
cách nào tệp cũng được **sao chép vào thư mục tạm của job** khi giai đoạn bắt đầu, nên một văn bản ổn định được đọc dù
bản gốc đổi hay biến mất trong lúc đó. (Văn bản nằm dưới `data/`, tên mà `.gitignore` cũng dùng cho thư mục dữ liệu
lúc chạy; lần commit đầu đã bỏ sót nó vì đúng lý do này, và một bản checkout mới sẽ từ chối mọi job ma trận với 503. Quy
tắc ignore nay có ngoại lệ cho thư mục của package; xem TEST_REPORT.)

**Lần chạy.** `llama-imatrix -m bản-trung-gian -f calibration.txt -o imatrix.gguf --chunks N -c 512 --no-ppl -ngl 0
[-t luồng]`. `N` là `advanced.imatrix_chunks`, mặc định 100; ngữ cảnh 512 token vì chuỗi ngắn giúp lần chạy trên CPU
khả thi; `--no-ppl` bỏ lượt tính perplexity, vốn chỉ tốn thời gian. Với `--no-ppl` công cụ không in dòng nào theo từng
chunk (ở b11342 các dòng đó nằm trong nhánh tính perplexity), nên tiến độ được ước lượng từ thời gian của lượt đầu và
ETA mà nó in ra, cập nhật mỗi giây (không bao giờ chạm 100 % chỉ nhờ đồng hồ: tệp được ghi sau chunk cuối); nếu phiên
bản sau có in dòng theo từng chunk thì các dòng đó được dùng thay. Kết quả phải tồn tại và không rỗng, nếu không job lỗi.

**Bộ nhớ và thời gian (đo được).** Đỉnh bộ nhớ resident của một job là 1045 MiB (Qwen2.5-0.5B, `Q4_K_M`) và 1697 MiB
(Qwen2.5-1.5B, `IQ3_M`, có hiệu chỉnh); đỉnh cgroup 3,0 và 3,4 GB gồm cả page cache. Xem TEST_REPORT.
