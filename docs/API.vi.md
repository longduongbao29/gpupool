# Tài liệu tham chiếu HTTP API của gpupool

> Bản tiếng Việt. Bản tiếng Anh: [API.en.md](API.en.md). Giữ hai bản đồng bộ.

Tài liệu này được viết từ code (`src/gpupool/coordinator/api.py`, `app.py`, `library_api.py`,
`src/gpupool/router/proxy.py`, `src/gpupool/agent/app.py`, `src/gpupool/common/models.py`). Nó liệt kê mọi
route HTTP của hai dịch vụ: **42 route** (coordinator 35, agent 7), cùng giao diện web tĩnh mà coordinator
mount tại `/`. Danh sách đầy đủ nằm ở mục 11.

| File đã quét | Số route |
| --- | --- |
| `coordinator/api.py` (`/api`) | 16 |
| `coordinator/library_api.py` | 6 (5 route thư viện, 1 route phục vụ file) |
| `coordinator/app.py` | 10 (`/healthz`, `/metrics`, 2 internal, 6 `/admin`) |
| `router/proxy.py` (`/v1`) | 3 |
| `agent/app.py` | 7 |

## 0. Quy ước

### Xác thực

Có ba bí mật độc lập. Route cần một bí mật đang để trống thì **mở** (chế độ dev: `require_bearer` không
kiểm tra khi chưa cấu hình khóa nào khác rỗng).

| Tên | Cấu hình | Header | Bảo vệ |
| --- | --- | --- | --- |
| API key | `api_keys` / `GPUPOOL_API_KEYS` (danh sách) | `Authorization: Bearer <key>` | `/v1/*` (một khóa bất kỳ trong danh sách là được) |
| Admin key | `admin_key` / `GPUPOOL_ADMIN_KEY` | `Authorization: Bearer <admin key>` | `/api/*`, `/admin/*` |
| Cluster token | `cluster_token` / `GPUPOOL_CLUSTER_TOKEN` | `Authorization: Bearer <token>` | `/internal/*`, `/files/{name}` và mọi route của agent trừ `GET /health` |

Khi coordinator chạy bằng `gpupool coordinator`, admin key và cluster token để trống sẽ được sinh và lưu
trong `secrets.json` cạnh database. Token sai hoặc thiếu trả
`401 {"detail": "invalid or missing bearer token"}` (với `/v1/*` là dạng lỗi OpenAI bên dưới).

`GET /healthz` (coordinator), `GET /health` (agent), `GET /metrics` và UI tĩnh không cần token.

### Lỗi

- Route của coordinator và agent dùng lỗi FastAPI: `{"detail": "<thông báo>"}` kèm status code; kiểm tra
  request thất bại là `422` với danh sách `detail` của FastAPI.
- Route `/v1/*` dùng dạng lỗi OpenAI: `{"error": {"message": "...", "type": "...", "code": "..."}}`.
- Lỗi khi lập kế hoạch đặt (`/api/models/{name}/plan`, `/api/simulate`, `/api/recommend`) ánh xạ exception
  sang status: `NoFit` là `409`, thiếu file model là `404`, còn lại `400`. Phần detail là
  `"<KiểuException>: <thông báo>"`.

### Ví dụ

```bash
export COORD=http://localhost:8080
export ADMIN=...   # admin key
curl -s -H "Authorization: Bearer $ADMIN" $COORD/api/state | jq .summary
```

## 1. API tương thích OpenAI (`/v1`)

Do router của coordinator phục vụ. Xác thực: API key (mở khi `api_keys` trống).

| Method | Đường dẫn | Mục đích |
| --- | --- | --- |
| GET | `/v1/models` | liệt kê tên model (mọi model đã đăng ký, đang chạy hay không) |
| POST | `/v1/chat/completions` | chat completion, chuyển tiếp tới một replica sẵn sàng |
| POST | `/v1/completions` | text completion, chuyển tiếp tới một replica sẵn sàng |

**`GET /v1/models`** trả `{"object": "list", "data": [{"id": "<name>", "object": "model", "owned_by":
"gpupool"}]}`.

**`POST /v1/chat/completions` và `POST /v1/completions`**: body là request của llama-server (OpenAI), được
chuyển tiếp nguyên vẹn, chỉ có `cache_prompt` mặc định là `true`. Các quy tắc router thêm vào:

- Body phải là một JSON object có `model` là chuỗi, trỏ tới một model đã đăng ký.
- Kích thước body bị giới hạn bởi `max_request_mb` (mặc định 32 MB), kể cả khi upload dạng chunked.
- `"stream": true` trả `text/event-stream`, chuyển tiếp nguyên như nhận được. Nếu upstream đứt sau khi
  stream đã bắt đầu, router gửi một frame SSE `data:` chứa object `error` rồi đóng stream.
- Chọn replica: ưu tiên theo prefix của prompt (rendezvous hash trên phần đầu hội thoại, nên cùng prefix đi
  vào cùng replica, để tận dụng prompt cache của llama.cpp), và chuyển sang replica ít tải nhất khi replica
  được ưu tiên đang có nhiều hơn replica ít tải nhất quá 2 request.
- Lần thử lỗi (lỗi kết nối hoặc upstream trả status >= 500) được thử lại trên replica khác, tối đa 2 lần
  thử lại (3 lần thử), và chỉ khi chưa có byte nào tới client.
- **Khởi động lạnh**: nếu model có `min_replicas = 0`, đã start (`replicas > 0`) và chưa có replica sẵn
  sàng, request được giữ tối đa `cold_start_timeout_s` (mặc định 120 s, cấu hình) trong lúc model load.
  Client ngắt kết nối thì chờ kết thúc.

| Status | `error.code` | Khi nào |
| --- | --- | --- |
| 400 | `invalid_json` | body không phải JSON hợp lệ |
| 400 | `missing_model` | không phải object, hoặc thiếu `model`, hoặc `model` không phải chuỗi |
| 401 | `invalid_api_key` | API key sai hoặc thiếu |
| 404 | `model_not_found` | `model` chưa đăng ký |
| 413 | `request_too_large` | body vượt `max_request_mb` |
| 502 | `upstream_error` | mọi lần thử đều lỗi (`upstream failed: ...` / `all replicas failed: ...`) |
| 503 | `no_replica` | model chưa có replica sẵn sàng và không được phép khởi động lạnh |
| 503 | `model_loading` | khởi động lạnh chưa xong trong `cold_start_timeout_s`; kèm `Retry-After: 10` |

```bash
curl -s $COORD/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen-3b", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

## 2. ModelSpec

`ModelSpec` (trong `common/models.py`) là thứ coordinator lưu cho mỗi model. API nhận nó ở hai dạng:
`ModelSpec` đầy đủ ở `POST /admin/models`, và `ModelBody` thân thiện hơn ở `PUT /api/models/{name}`
(mục 3). `GET /api/state` trả nó dưới `models[].spec`.

| Trường | Kiểu, mặc định | Giá trị hợp lệ và ý nghĩa |
| --- | --- | --- |
| `name` | string, bắt buộc | alias phục vụ ở `/v1/models`. Qua `/api/models/{name}`: `[A-Za-z0-9._-]{1,64}` |
| `source` | string, bắt buộc | `https://...gguf`, đường dẫn tuyệt đối trên coordinator, hoặc `coordinator://<file trong thư viện>`. `/api` luôn ghi `coordinator://<file>` |
| `ctx_size` | int, 4096 | tổng context (token), chia đều cho `parallel` slot (llama.cpp `-c`); qua `/api` phải `>= 1` |
| `parallel` | int, 1 | số slot chạy đồng thời (llama.cpp `-np`); qua `/api` phải `>= 1`. Mức bận của autoscale là `processing / parallel` |
| `replicas` | int, 1 | số lượng mong muốn; **`0` = dừng**. Đây là công tắc bật/tắt: khi autoscale thì giữ `> 0` và `min/max` giới hạn số lượng |
| `pin_devices` | danh sách `"node_id/device_id"`, rỗng | giới hạn replica vào các GPU này; rỗng = scheduler tự chọn. `/api` từ chối mục không đúng dạng `node/device` hoặc trỏ tới server chưa đăng ký (422) |
| `priority` | int 0..100, 50 | càng cao càng được xếp trước mỗi tick và có thể giành chỗ của mức thấp hơn |
| `preemptible` | bool, true | `false`: model ưu tiên cao hơn không bao giờ dừng replica của model này |
| `spread` | `"gpu"` / `"node"` / `"none"`, `"gpu"` | mềm: các replica của model tránh dùng chung GPU / server (vẫn dùng chung GPU khi không còn chỗ nào khác) |
| `min_replicas` | int `>= 0` hoặc null, null | sàn của autoscale; null = `replicas` (số cố định). `0` cho phép về 0 |
| `max_replicas` | int `>= 1` hoặc null, null | trần của autoscale; null = `replicas`. Autoscale hoạt động khi `max > min` |
| `autoscale` | object hoặc null, null | `{target_busy: 0.7 (0,1], up_after_s: 30 >= 0, down_after_s: 300 >= 0}`; null = các giá trị mặc định đó. Thêm replica khi (slot bận / tổng slot) trung bình vượt `target_busy` liên tục `up_after_s` (hoặc có request xếp hàng); bớt một replica khi dưới `target_busy / 2` liên tục `down_after_s` |
| `idle_unload_s` | float `> 0` hoặc null, null | chỉ khi `min_replicas == 0`: dỡ model sau chừng này giây không có request; request kế tiếp sẽ khởi động lạnh nó |
| `kv_cache_type` | `"f16"` / `"q8_0"` / `"q4_0"`, `"f16"` | kiểu phần tử của KV cache (`-ctk/-ctv`). Byte mỗi phần tử 2 / 34/32 / 18/32, nên q8_0 / q4_0 giảm bộ nhớ KV còn khoảng một nửa / một phần tư |
| `speculative` | `"none"` / `"ngram"` / `"draft"`, `"none"` | giải mã suy đoán: `ngram` đoán từ văn bản đã có (không tốn thêm bộ nhớ); `draft` chạy một model nhỏ cùng tokenizer trên GPU đầu tiên của head |
| `draft` | string hoặc null, null | nguồn của model draft, `coordinator://<file>`; chỉ dùng với `speculative: "draft"` (bị bỏ nếu không) |
| `draft_n_max` | int 1..16, 4 | số token draft mỗi bước. Trên GTX 1650 (3B + draft 0.5B) mức 4 nhanh hơn 5 %, mức 8 chậm hơn không dùng |

Kiểm tra do `PUT /api/models/{name}` và `/api/simulate` áp dụng (giống nhau):

- `min_replicas <= max_replicas` (422).
- `idle_unload_s` cần `min_replicas == 0` (422).
- `speculative: "draft"` cần `draft_file` là một mục thư viện **ready**, khác chính model, có tokenizer
  model trùng với model và kích thước từ vựng chênh nhau tối đa 128 token (422 nếu không).

Không thuộc `ModelSpec` (có trong thảo luận thiết kế nhưng chưa triển khai): `share_gpu`, `gpu_selector`,
nhãn server/GPU, `reserve_mb`.

## 3. Model (`/api/models`)

Tất cả nằm dưới `/api`, dùng admin key.

| Method | Đường dẫn | Mục đích |
| --- | --- | --- |
| PUT | `/api/models/{name}` | tạo hoặc cập nhật spec của model (**không** start nó) |
| POST | `/api/models/{name}/start` | đặt `replicas` (start) |
| POST | `/api/models/{name}/stop` | đặt `replicas = 0` (các replica được drain) |
| DELETE | `/api/models/{name}` | drain replica và xóa spec |
| GET | `/api/models/{name}/scaling` | trạng thái autoscaler |
| POST | `/api/models/{name}/plan` | chạy thử việc đặt một replica |

### `PUT /api/models/{name}`

Body `ModelBody`. Trường ghi "giữ" sẽ giữ giá trị đã lưu khi bỏ trống (hoặc `null`) ở model đã có; model
mới lấy giá trị mặc định.

| Trường | Kiểu | Mặc định / giữ |
| --- | --- | --- |
| `file` | string, bắt buộc | tên mục thư viện; phải `ready` (nếu không là 422). Lưu thành `source = coordinator://<file>` |
| `ctx_size` | int `>= 1` | 4096 (luôn ghi đè) |
| `parallel` | int `>= 1` | 1 (luôn ghi đè) |
| `pin_devices` | danh sách string | rỗng (luôn ghi đè, bỏ trùng) |
| `priority` | int 0..100 hoặc null | giữ, nếu không có thì 50 |
| `spread` | `gpu`/`node`/`none` hoặc null | giữ, nếu không có thì `gpu` |
| `min_replicas`, `max_replicas` | int hoặc null | giữ, nếu không có thì để trống |
| `autoscale` | object hoặc null | giữ, nếu không có thì để trống |
| `idle_unload_s` | float `> 0` hoặc null | giữ, nếu không có thì để trống |
| `preemptible` | bool hoặc null | giữ, nếu không có thì true |
| `kv_cache_type` | enum hoặc null | giữ, nếu không có thì `f16` |
| `speculative` | enum hoặc null | giữ, nếu không có thì `none` |
| `draft_file` | string hoặc null | giữ (draft đã lưu), nếu không có thì không dùng |
| `draft_n_max` | int 1..16 hoặc null | giữ, nếu không có thì 4 |

`replicas` không nằm trong body: model mới bắt đầu với `replicas = 0`, model đã có giữ giá trị cũ. Trả về
`ModelSpec` đã lưu (JSON). Lỗi: 422 (tên sai, file chưa ready, pin sai, min > max, idle mà min khác 0, các
kiểm tra draft), 401. Reconciler được đánh thức.

```bash
curl -s -X PUT $COORD/api/models/qwen-7b -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" -d '{
  "file": "qwen2.5-7b-instruct-q4_k_m.gguf", "ctx_size": 8192, "parallel": 4, "priority": 80,
  "min_replicas": 1, "max_replicas": 3, "spread": "node", "kv_cache_type": "q8_0"}'
```

### `POST /api/models/{name}/start`

Body tùy chọn `{"replicas": 1}` (`int >= 1`, mặc định 1). Đặt `replicas` và phát event `model_started`. Trả
về `ModelSpec`. 404 nếu model không tồn tại. Khi có autoscale, số lượng sau đó do `min/max` điều khiển.

```bash
curl -s -X POST $COORD/api/models/qwen-7b/start -H "Authorization: Bearer $ADMIN"
```

### `POST /api/models/{name}/stop`

Không có body. Đặt `replicas = 0` (phát `model_stopped`); reconciler drain các replica. Trả về `ModelSpec`.
404 nếu không tồn tại.

### `DELETE /api/models/{name}`

Drain mọi replica đang hoạt động rồi xóa spec. Trả `{"ok": true}`. 404 nếu không tồn tại. File trong thư
viện được giữ lại (sau đó có thể xóa khỏi thư viện).

### `GET /api/models/{name}/scaling`

Trạng thái autoscaler của một model. 404 nếu model không tồn tại (hoặc `autoscaling not available`).

```json
{"model": "q05", "min": 0, "max": 2, "desired": 1, "ready": 1, "launching": 0,
 "avg_busy": 0.25, "queued": false, "idle_s": 12.4,
 "state": "steady", "last_decision": {"...": "..."},
 "replicas": [{"replica_id": "q05-1", "busy": 0.25, "requests_processing": 1, "requests_deferred": 0,
               "measured_decode_tps": 182.0, "est_decode_tps": 204.0, "metrics_ok": true}]}
```

`state` là một trong `stopped` (`replicas == 0`), `fixed` (min == max, không dỡ khi rảnh), `unloaded`
(desired 0), `scaling_up`, `scaling_down`, `steady`. `busy` và các bộ đếm lấy từ `/metrics` của
llama-server trên head (`requests_processing`, `requests_deferred`, `predicted_tokens_seconds`); khi lần
scrape cũ hơn 3 chu kỳ poll thì dùng số request đang chạy do router đếm và `metrics_ok` là false.

### `POST /api/models/{name}/plan`

Không có body. Một replica nữa sẽ được đặt ở đâu ngay bây giờ? Không có tác dụng phụ (cổng chỉ được chọn
trong phạm vi lời gọi). Trả về một `Placement`:

```json
{"model": "qwen-7b", "replica_id": "...", "tier": "single_gpu", "head_node": "b", "head_port": 9001,
 "assignments": [{"node_id": "b", "device_id": "CUDA1", "llama_device": "CUDA0", "rpc_endpoint": null,
                  "layers": 28, "est_mb": 6120, "device_uuid": "GPU-..."}],
 "tensor_split": [1.0], "est_total_mb": 6120, "score": 91.0, "est_decode_tps": 41.2,
 "reasons": ["fastest GPU with room"], "draft_est_mb": null, "mem_factor": 1.0}
```

Lỗi: 404 model không tồn tại, 409 `NoFit`, 404 thiếu file model, 400 các lỗi khác.

## 4. Server, node và GPU

Tất cả nằm dưới `/api`, dùng admin key. "Server" là một agent đã đăng ký; `node_id` là id của nó.

| Method | Đường dẫn | Mục đích |
| --- | --- | --- |
| POST | `/api/servers` | đăng ký một agent theo URL |
| DELETE | `/api/servers/{node_id}` | dừng replica của nó và hủy đăng ký |
| PUT | `/api/servers/{node_id}/gpus/{device_id}` | bật hoặc tắt một GPU |

### `POST /api/servers`

Body `{"agent_url": "http://host:7070"}` (phải dạng `http(s)://host[:port]`). Coordinator thăm dò `/report`
của agent bằng cluster token. Trả về mục server (cùng dạng với `servers[]` trong `/api/state`). Lỗi: `400`
(URL sai, không tới được agent, sai token, report không hợp lệ), `409` đã đăng ký. Xóa dấu "đã gỡ" từ lần
trước (xem `DELETE`).

```bash
curl -s -X POST $COORD/api/servers -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" \
  -d '{"agent_url": "http://10.0.0.5:7070"}'
```

### `DELETE /api/servers/{node_id}`

Đánh dấu mọi replica chạm vào node là stopped (tick kế tiếp đặt lại số lượng mong muốn ở nơi khác), quên
server và ghi nhớ việc gỡ, nên `--join` của agent **không** đưa nó trở lại (`/internal/join` trả 403) cho
tới khi thêm lại bằng `POST /api/servers`. Trả `{"ok": true}`. 404 nếu không tồn tại.

### `PUT /api/servers/{node_id}/gpus/{device_id}`

Body `{"enabled": true|false}`. GPU bị tắt không nhận replica mới (coi như usable 0). Cờ được lưu theo UUID
của card khi agent báo cáo (device id dịch chuyển khi một GPU biến mất), nếu không thì theo `device_id`.
Trả `{"node_id", "device_id", "enabled"}`. 404 nếu server không tồn tại.

## 5. Thư viện model và file

Route thư viện: admin key. `/files/{name}`: cluster token (agent tải model qua đường này).

| Method | Đường dẫn | Xác thực | Mục đích |
| --- | --- | --- | --- |
| GET | `/api/hf/files?repo=owner/name` | admin | liệt kê file `.gguf` của một repo Hugging Face |
| GET | `/api/library` | admin | liệt kê các mục thư viện |
| GET | `/api/library/browse` | admin | liệt kê file `.gguf` tìm thấy dưới các thư mục gốc của model |
| POST | `/api/library` | admin | thêm một mục (tải từ HF hoặc đường dẫn cục bộ) |
| DELETE | `/api/library/{name}` | admin | xóa một mục và file của nó |
| GET | `/files/{name}` | cluster token | tải một file thư viện (nội bộ) |

**`GET /api/hf/files?repo=owner/name`** trả `[{"file": "path/in/repo.gguf", "bytes": 123}]`, đã sắp xếp. Lỗi:
`400` repo không đúng dạng `owner/name`, `403` repo gated hoặc private (đặt `HF_TOKEN`), `404` không có
repo, `502` không tới được HF hoặc câu trả lời không như mong đợi.

**`GET /api/library`** trả danh sách `LibraryItem`:

| Trường | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `name` | string | tên file duy nhất, ví dụ `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| `path` | string | đường dẫn tuyệt đối trên coordinator |
| `source` | `"hf"` / `"path"` | tải từ HF hay file cục bộ đã đăng ký |
| `hf_repo`, `hf_file` | string hoặc null | nguồn HF |
| `bytes` | int hoặc null | tổng kích thước khi biết |
| `downloaded` | int | số byte đã ghi |
| `status` | `"downloading"` / `"ready"` / `"failed"` | chỉ mục `ready` mới dùng được cho model |
| `error` | string hoặc null | lý do lỗi |
| `created_at` | float | thời gian unix |

**`GET /api/library/browse`** trả `{"roots": [{"path", "exists", "host_path"}], "files": [{"path", "name",
"bytes", "in_library", "split_part", "broken_link", "host_path"}], "truncated": bool}`. Nó duyệt các thư
mục gốc của model (`model_roots`, mặc định `models_dir`) sâu tối đa 6 cấp và 1000 file.

**`POST /api/library`** body: hoặc `{"hf_repo": "owner/name", "hf_file": "file.gguf"}` (bắt đầu tải nền vào
`models_dir`) hoặc `{"path": "/abs/path/model.gguf"}` (đăng ký file có sẵn, dịch qua `path_map` khi
coordinator chạy trong Docker). Đưa cả hai dạng, hoặc một dạng thiếu, là 422. Trả `LibraryItem`. Lỗi: `400`
tên/đường dẫn không hợp lệ, không phải `.gguf`, hoặc file split đăng ký bằng đường dẫn, `409` trùng tên
trong thư viện, `404`/`403`/`502` từ HF. GGUF split (`-00001-of-NNNNN`) được thêm từ HF qua phần đầu tiên và
tải đủ; các phần được đo và phục vụ cùng nhau.

**`DELETE /api/library/{name}`** hủy tải đang chạy và xóa mục cùng file. Trả `{"ok": true}`. `404` không
tồn tại, `409` nếu có model đang dùng file.

**`GET /files/{name}`** trả file dạng `application/octet-stream` (kể cả một phần riêng của mục split). Nếu
không có: `404 no such model file`.

```bash
curl -s -X POST $COORD/api/library -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" \
  -d '{"hf_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "hf_file": "qwen2.5-0.5b-instruct-q4_k_m.gguf"}'
```

## 6. Dung lượng, gợi ý và giả lập

Tất cả nằm dưới `/api`, dùng admin key. Không route nào thay đổi cụm.

| Method | Đường dẫn | Mục đích |
| --- | --- | --- |
| GET | `/api/capacity` | bộ nhớ trống, băng thông, mức bận và các replica của từng GPU |
| POST | `/api/recommend` | xếp hạng GPU/server cho một model trước khi deploy |
| POST | `/api/simulate` | reconciler sẽ làm gì với các thay đổi spec giả định |

### `GET /api/capacity`

```json
{"gpus": [{"node_id": "a", "device_id": "CUDA0", "uuid": "GPU-...", "name": "NVIDIA ...", "kind": "cuda",
           "enabled": true, "alive": true, "total_mb": 16384, "usable_mb": 9800, "free_for_new_mb": 5500,
           "reserved_mb": 4300, "bandwidth_gbps": 320.0, "busy": 0.42,
           "replicas": [{"replica_id": "chat-1", "model": "chat", "est_mb": 4300, "busy": 0.42}]}],
 "summary": {"gpus": 6, "free_for_new_mb": 31200, "largest_single_gpu_mb": 9800, "largest_single_node_mb": 18100}}
```

`free_for_new_mb` bằng 0 với GPU bị tắt hoặc node chết; `reserved_mb = usable_mb - free_for_new_mb`.
`summary` chỉ tính thiết bị CUDA.

### `POST /api/recommend`

Body `RecommendBody`:

| Trường | Kiểu, mặc định | Ý nghĩa |
| --- | --- | --- |
| `file` | string, bắt buộc | mục thư viện (phải `ready`, nếu không là 422) |
| `ctx_size` | int `>= 1`, 4096 | |
| `parallel` | int `>= 1`, 1 | |
| `priority` | int 0..100, 50 | dùng để xác định replica nào có thể bị giành chỗ |
| `spread` | enum, `gpu` | |
| `pin_devices` | danh sách, rỗng | |
| `limit` | int 1..10, 3 | số phương án |
| `kv_cache_type` | enum, `f16` | |
| `speculative` | enum, `none` | |
| `draft_file` | string hoặc null | cho `speculative: "draft"` (cùng các kiểm tra 422 như `PUT /api/models`) |
| `draft_n_max` | int 1..16, 4 | |

Phản hồi:

```json
{"need_mb": 6120,
 "options": [{"rank": 1, "score": 91.0, "tier": "single_gpu", "fits_now": true,
              "assignments": [{"node_id": "b", "device_id": "CUDA1", "layers": 28, "est_mb": 6120}],
              "est_decode_tps": 41.0, "est_total_mb": 6120, "reasons": ["..."]}],
 "max_ctx_single_gpu": 16384,
 "not_possible": null}
```

- `options`: tối đa `limit` phương án xếp hạng theo điểm của chính scheduler. Khi chưa có phương án nào đặt
  được ngay nhưng dừng replica ưu tiên thấp hơn thì được, một phương án bổ sung được thêm vào cuối với
  `fits_now: false` và `requires_preemption: [{"replica_id", "model", "priority"}]`.
- `max_ctx_single_gpu`: `ctx_size` lớn nhất (bội của 256, tối đa 131072) vừa trên một GPU đơn ngay bây giờ,
  hoặc null.
- `not_possible`: khi không có gì vừa kể cả khi giành chỗ:
  `{"need_mb", "largest_single_gpu_mb", "largest_single_node_mb", "max_ctx_that_fits"}`, nếu không thì null.

```bash
curl -s -X POST $COORD/api/recommend -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" \
  -d '{"file": "qwen2.5-7b-instruct-q4_k_m.gguf", "ctx_size": 8192, "parallel": 4, "priority": 80}'
```

### `POST /api/simulate`

Chạy đúng các quy tắc sắp thứ tự, đặt và giành chỗ của reconciler trên toàn cụm với các thay đổi giả định.
Body:

```json
{"changes": [{"model": "qwen-7b", "min_replicas": 2, "priority": 90}],
 "add": [{"name": "new", "file": "x.gguf", "replicas": 1, "ctx_size": 4096}]}
```

- `changes[]`: `model` (đã tồn tại, nếu không là 404) cùng bất kỳ trường nào trong `replicas`,
  `min_replicas`, `max_replicas`, `priority`, `preemptible`, `ctx_size`, `parallel`, `spread`,
  `pin_devices`, `kv_cache_type`, `speculative`, `draft_file`, `draft_n_max`. Vắng = không đổi.
- `add[]`: `name` (mới, `[A-Za-z0-9._-]{1,64}`), `file` (mục thư viện ready), cùng các trường tùy chọn
  như trên. Model mới bắt đầu ở mức sàn (`max(min_replicas, 1)`) khi `replicas > 0`.
- Kiểm tra giống `PUT /api/models` (422); kết quả của từng model được kiểm tra đầy đủ.

Phản hồi:

```json
{"start":   [{"model": "qwen-7b", "tier": "single_gpu", "est_decode_tps": 41.0,
              "assignments": [{"node_id": "b", "device_id": "CUDA1", "layers": 28, "est_mb": 6120}]}],
 "stop":    [{"replica_id": "chat-2", "model": "chat", "reason": "2 running, 1 wanted"}],
 "preempt": [{"replica_id": "q05-1", "model": "q05", "priority": 20, "for_model": "q3b"}],
 "unplaced":[{"model": "qwen-7b", "missing": 1, "why": "NoFit: ..."}]}
```

Khác với một tick thật: bỏ qua cooldown giành chỗ, bộ nhớ giải phóng do dừng và gỡ có sẵn ngay, và
`ctx_size`/`parallel` thay đổi chỉ ảnh hưởng replica mới (replica đang chạy không bị khởi động lại). Nó
không báo các lần di chuyển; dùng `/api/rebalance` cho việc đó.

## 7. Co giãn và cân bằng lại

| Method | Đường dẫn | Mục đích |
| --- | --- | --- |
| GET | `/api/models/{name}/scaling` | xem mục 3 |
| POST | `/api/rebalance` | tìm (và tùy chọn bắt đầu) một cách đặt replica tốt hơn |

### `POST /api/rebalance`

Body tùy chọn `{"dry_run": true}` (mặc định **true**, kể cả khi không có body). Ứng viên là các replica
ready sẽ có điểm cao hơn ít nhất 25 ở nơi khác, tốt nhất đứng đầu. Với `dry_run: false`, ứng viên tốt nhất
được bắt đầu theo kiểu **make-before-break**: launch replica mới, chỉ drain replica cũ khi replica mới đã
ready. Bị từ chối (không bắt đầu gì) trừ khi cụm đang yên: không có lần di chuyển nào đang chạy, không có
replica đang launch, không có lần giành chỗ nào đang chờ bộ nhớ. Mỗi lần chỉ một lần di chuyển trong cả cụm.

```json
{"moves": [{"replica_id": "y-1", "model": "y", "from": [{"node_id": "b", "device_id": "CPU"}],
            "to": [{"node_id": "a", "device_id": "CUDA0"}], "current_score": -20.0, "new_score": 80.0,
            "gain": 100.0, "reasons": ["..."]}],
 "started": null,
 "in_progress": null}
```

`started` là `{"replica_id", "model"}` khi một lần di chuyển bắt đầu. `in_progress` là lần di chuyển đang
chạy (`{"model", "old", "new", "since"}`) hoặc null. Cứ mỗi `rebalance_s` (mặc định 600 s, 0 = tắt) cũng có
một lần chạy định kỳ khi cụm đang yên.

```bash
curl -s -X POST $COORD/api/rebalance -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" -d '{"dry_run": false}'
```

## 8. Trạng thái và event

### `GET /api/state`

Toàn bộ những gì web UI hiển thị, trong một lời gọi (admin key).

| Khóa | Nội dung |
| --- | --- |
| `summary` | `servers_total`, `servers_online`, `gpus_total`, `gpus_enabled`, `pool_total_mb`, `pool_usable_mb` (các GPU CUDA đang bật của server còn sống), `models_running` |
| `servers[]` | `node_id`, `agent_url`, `added_at`, `alive`, `last_seen`, `report` (`NodeReport` của agent hoặc null), `gpu_enabled` (`{device_id: bool}`) |
| `models[]` | xem bên dưới |
| `library[]` | danh sách `LibraryItem` |
| `settings` | `public_url`, `cluster_token`, `api_keys_set` (bool) |
| `events[]` | 50 event mới nhất |
| `unread_events` | số event chưa đọc |
| `rebalance` | `{"in_progress": {...} hoặc null, "next_run_ts": thời gian unix hoặc null}` |

Mục trong `models[]`:

| Trường | Ý nghĩa |
| --- | --- |
| `spec` | `ModelSpec` |
| `file` | tên file thư viện khi source là `coordinator://`, nếu không là null |
| `state` | `running`, `starting`, `idle` (đã start nhưng đang dỡ, request kế tiếp sẽ load), `stopping`, `stopped`, `failed` |
| `error` | lý do của `failed`, hoặc lỗi của replica mới nhất khi đang `starting` |
| `scaling` | `{"min", "max", "desired", "avg_busy", "unloaded"}` |
| `calibration` | khối **tự hiệu chỉnh VRAM**, `{"factor": 1.04, "samples": 3}`, hoặc `null` cho tới khi một replica của model được đo. Xem bên dưới |
| `replicas[]` | các trường của `ReplicaRecord` (`replica_id`, `model`, `placement`, `state`, `created_at`, `updated_at`, `error`) cộng `outstanding` (số request đang chạy). Liệt kê replica còn sống và replica lỗi mới nhất |

**`calibration`.** Khi một replica chuyển sang ready, coordinator hỏi agent của head xem llama.cpp thực sự
đã cấp phát bao nhiêu (`GET /engines/{engine_id}/memory`, mục 10), so với ước lượng (trừ phần context
runtime mỗi thiết bị mà llama.cpp không báo) và gộp tỷ lệ đó vào một hệ số riêng của model (trung bình trượt
mũ, trọng số 0.5 mỗi mẫu). Khi đặt, nhu cầu thiết bị của model được nhân với `factor`, giới hạn trong
0.9..2.0, nên các kế hoạch sau dùng số thật. `factor` ở đây là giá trị đã giới hạn mà việc đặt dùng,
`samples` là số lần đo. Event mức info `calibrated` được phát khi hệ số đổi hơn 5 %. Một mẫu bị bỏ qua khi
agent cũ, log không có dòng buffer, hoặc thiếu thiết bị; khi đó hệ số giữ nguyên.

### Event

| Method | Đường dẫn | Mục đích |
| --- | --- | --- |
| GET | `/api/events?limit=200&after_id=` | liệt kê event, mới nhất trước |
| POST | `/api/events/read` | đánh dấu event đã đọc |

**`GET /api/events`**: `limit` bị kẹp trong 1..1000 (mặc định 200); `after_id` chỉ trả event có id lớn hơn.
Trả `{"events": [...], "unread": n}`. Một event là
`{"id", "ts", "level": "info"|"warning"|"error", "kind", "message", "node_id", "model", "read"}`.

**`POST /api/events/read`** body `{"up_to_id": 123}` đánh dấu các event tới id đó là đã đọc. Trả
`{"unread": n}`.

Các loại event có trong code: `server_added`, `server_removed`, `node_online`, `node_offline`,
`model_started`, `model_stopped`, `launch_failed`, `engine_crashed`, `crash_loop`, `gpu_missing`,
`realloc_started`, `realloc_failed`, `realloc_done`, `preempted`, `scaled_up`, `scaled_down`,
`unloaded_idle`, `cold_start`, `rebalance_started`, `rebalanced`, `rebalance_failed`, `calibrated`. Cảnh
báo và lỗi còn được POST tới `webhook_url` khi có cấu hình.

## 9. Coordinator: health, metrics và admin API cũ

| Method | Đường dẫn | Xác thực | Mục đích |
| --- | --- | --- | --- |
| GET | `/healthz` | không | `{"ok": true}` |
| GET | `/metrics` | không | văn bản Prometheus |
| POST | `/admin/models` | admin | tạo hoặc thay thế model từ một `ModelSpec` đầy đủ |
| DELETE | `/admin/models/{name}` | admin | drain và xóa model |
| POST | `/admin/models/{name}/scale?replicas=N` | admin | đặt `replicas` (`N >= 0`, nếu không là 422) |
| POST | `/admin/deploy/{model}?dry_run=0` | admin | `dry_run=1`: trả `Placement` (như `/plan`); nếu không thì chạy một tick reconcile và trả các replica của model |
| DELETE | `/admin/replicas/{replica_id}` | admin | drain một replica (404 nếu không tồn tại) |
| GET | `/admin/status` | admin | node thô (thiết bị, engine), model và replica |

`/admin/*` là giao diện cũ, mức thấp hơn, dùng cho CLI và script. Nó nhận `ModelSpec` thô (không kiểm tra
thư viện, không kiểm tra draft hay pin, không phát event), nên nên ưu tiên `/api/models`.
`POST /admin/models` trả spec đã lưu. `GET /admin/status` trả `{"nodes": [{"node_id", "alive", "last_seen",
"agent_url", "host", "devices", "engines"}], "models": [...], "replicas": [... cộng "outstanding"]}`.

`/metrics` công bố các bộ đếm và histogram của router (request theo model và status, số lần thử lại, thời
gian tới token đầu tiên, request đang chạy) cùng: `gpupool_device_free_mb{node,device}`,
`gpupool_device_usable_mb{node,device}`, `gpupool_node_alive{node}`, `gpupool_replicas{model,state}`.

## 10. API nội bộ (không gọi bằng tay)

### Coordinator, cluster token

| Method | Đường dẫn | Mục đích |
| --- | --- | --- |
| POST | `/internal/heartbeat` | agent đẩy `NodeReport` (chỉ khi bật `push_heartbeat`; mặc định là coordinator tự poll `GET /report`) |
| POST | `/internal/join` | agent tự đăng ký (`gpupool agent --join ...`) |

`POST /internal/heartbeat` body: một `NodeReport`. `403` khi node chưa đăng ký (server đã gỡ không tự quay
lại được). Trả `{"ok": true}`.

`POST /internal/join` body `{"agent_url": "http://10.0.0.5:7070"}`. Coordinator thăm dò agent như
`POST /api/servers`; lỗi là `502` (không phải 400). `403` nếu server đã bị gỡ trong UI. Trả `{"node_id",
"agent_url", "added_at", "new": true|false}`; nếu node đã biết dưới URL khác thì URL được cập nhật
(`new: false`).

### API của agent

Agent (cổng mặc định 7070) do coordinator điều khiển. Xác thực: cluster token, trừ `/health`.

| Method | Đường dẫn | Mục đích |
| --- | --- | --- |
| GET | `/health` | `{"ok": true}`, không xác thực |
| GET | `/report` | `NodeReport`: thiết bị, engine, phiên bản llama.cpp, file model đã cache, CPU/RAM |
| POST | `/engines` | khởi động một tiến trình llama.cpp từ `EngineSpec` |
| GET | `/engines/{engine_id}` | `EngineStatus` |
| GET | `/engines/{engine_id}/memory` | các buffer theo thiết bị mà llama.cpp báo lúc load |
| DELETE | `/engines/{engine_id}` | dừng một engine, trả `EngineStatus` của nó |
| POST | `/models/ensure` | đảm bảo file model có trong cache cục bộ |

**`POST /engines`** body `EngineSpec`: `engine_id` (`"<replica_id>-head"` hoặc
`"<replica_id>-rpc-<device_id>"`), `kind` (`"rpc"`/`"server"`), `port`, `devices` (rpc: đúng một thiết bị
cục bộ; server: danh sách có thứ tự như `["CUDA0","RPC0"]`), `model` (alias), `model_path` (GGUF trên head),
`rpc_endpoints` (`"host:port"`, theo thứ tự `RPC0..`), `tensor_split`, `ctx_size` (4096), `parallel` (1),
`extra_args`, `cache_type` (`f16`), `spec_type` (`none`), `draft_model_path`, `draft_device`, `draft_n_max`
(4), `allowed_peers` (các host được phép nối tới engine rpc; chỉ có hiệu lực khi agent chạy với
`rpc_firewall`). Trả `EngineStatus` (`engine_id`, `kind`, `state` `starting|running|exited|failed`, `pid`,
`port`, `exit_code`, `log_tail` tối đa 50 dòng). Lỗi: `422` với `extra_args` (agent này không chấp nhận, để
token không biến thành cờ llama-server tùy ý), engine server thiếu `model_path`, `model_path` hoặc
`draft_model_path` nằm ngoài cache model và không do `/models/ensure` trả về, file không tồn tại, engine rpc
không có đúng một thiết bị, hoặc cổng đang bị dùng; `409` engine đã chạy; `500` không tìm thấy binary.

**`GET /engines/{engine_id}/memory`**: llama.cpp in kích thước các buffer khi load model; agent phân tích
phần đầu log của engine (2 MiB đầu) và trả, theo từng thiết bị (`CUDA0`, `RPC0`, ...), đơn vị MiB, giá trị
cuối cùng của từng loại:

```json
{"engine_id": "chat-1-head",
 "devices": {"CUDA0": {"model_mb": 2090.5, "kv_mb": 144.0, "compute_mb": 80.5, "total_mb": 2315.0},
             "RPC0":  {"model_mb": 1003.2, "kv_mb": 72.0, "compute_mb": 80.5, "total_mb": 1155.7}}}
```

Buffer host (`CPU`, `*_Host`, `*_Mapped`) bị bỏ qua. `devices` là `{}` khi log không có dòng buffer.
`404 unknown engine`. Đây là dữ liệu đứng sau khối `calibration` của `/api/state`.

**`DELETE /engines/{engine_id}`**: `404 unknown engine`.

**`POST /models/ensure`** body `{"name": "file.gguf", "source": "coordinator://file.gguf" | "https://...gguf"}`.
Tải (từ `/files/{name}` của coordinator bằng cluster token, hoặc từ URL) vào cache của agent nếu chưa có;
các phần của GGUF split được tải cùng nhau. Trả `{"path": "...", "bytes": n}`. Lỗi: `422` không tìm thấy
file, `502` tải lỗi.

## 11. Chỉ mục route

Cả 42 route, để kiểm tra đầy đủ:

| # | Method | Đường dẫn | Mục |
| --- | --- | --- | --- |
| 1 | GET | `/v1/models` | 1 |
| 2 | POST | `/v1/chat/completions` | 1 |
| 3 | POST | `/v1/completions` | 1 |
| 4 | GET | `/api/state` | 8 |
| 5 | POST | `/api/servers` | 4 |
| 6 | DELETE | `/api/servers/{node_id}` | 4 |
| 7 | PUT | `/api/servers/{node_id}/gpus/{device_id}` | 4 |
| 8 | PUT | `/api/models/{name}` | 3 |
| 9 | POST | `/api/models/{name}/start` | 3 |
| 10 | GET | `/api/models/{name}/scaling` | 3 |
| 11 | POST | `/api/models/{name}/stop` | 3 |
| 12 | DELETE | `/api/models/{name}` | 3 |
| 13 | POST | `/api/models/{name}/plan` | 3 |
| 14 | GET | `/api/capacity` | 6 |
| 15 | POST | `/api/simulate` | 6 |
| 16 | POST | `/api/rebalance` | 7 |
| 17 | POST | `/api/recommend` | 6 |
| 18 | GET | `/api/events` | 8 |
| 19 | POST | `/api/events/read` | 8 |
| 20 | GET | `/api/hf/files` | 5 |
| 21 | GET | `/api/library` | 5 |
| 22 | GET | `/api/library/browse` | 5 |
| 23 | POST | `/api/library` | 5 |
| 24 | DELETE | `/api/library/{name}` | 5 |
| 25 | GET | `/files/{name}` | 5 |
| 26 | GET | `/healthz` | 9 |
| 27 | POST | `/internal/heartbeat` | 10 |
| 28 | POST | `/internal/join` | 10 |
| 29 | POST | `/admin/models` | 9 |
| 30 | DELETE | `/admin/models/{name}` | 9 |
| 31 | POST | `/admin/models/{name}/scale` | 9 |
| 32 | POST | `/admin/deploy/{model}` | 9 |
| 33 | DELETE | `/admin/replicas/{replica_id}` | 9 |
| 34 | GET | `/admin/status` | 9 |
| 35 | GET | `/metrics` | 9 |
| 36 | GET | `/health` (agent) | 10 |
| 37 | GET | `/report` (agent) | 10 |
| 38 | POST | `/engines` (agent) | 10 |
| 39 | GET | `/engines/{engine_id}` (agent) | 10 |
| 40 | GET | `/engines/{engine_id}/memory` (agent) | 10 |
| 41 | DELETE | `/engines/{engine_id}` (agent) | 10 |
| 42 | POST | `/models/ensure` (agent) | 10 |

Số lượng khớp với các decorator route tìm thấy trong code: 16 + 6 + 10 + 3 + 7 = 42.
