# gpupool — Giao diện quản lý + Docker: thiết kế

> Bản tiếng Việt. Bản tiếng Anh: [UI_DESIGN.en.md](UI_DESIGN.en.md). Hai bản phải được cập nhật cùng nhau.

Mục tiêu: một trang web quản lý, do coordinator phục vụ trên một máy, để thêm/xoá server, bật/tắt
từng GPU, đưa model vào (Hugging Face hoặc đường dẫn file), và bật/tắt serve một model bằng một nút.
Toàn bộ đóng gói thành Docker image.

## 1. Quyết định

| Chủ đề | Quyết định | Lý do |
| --- | --- | --- |
| Thêm server | Đăng ký agent đang chạy: nhập URL; coordinator gọi tới, đọc `node_id`, thêm vào pool | Theo lựa chọn của bạn. Không lưu thông tin SSH ở đâu cả. |
| Phát hiện node | **Coordinator chủ động kéo** `GET /report` từ mọi server đã đăng ký, mỗi 2 giây | Xoá server phải có hiệu lực thật. Với heartbeat đẩy lên, node đã xoá sẽ tự xuất hiện lại ở heartbeat kế tiếp. Endpoint heartbeat cũ vẫn giữ nhưng chỉ nhận node đã đăng ký. |
| Chọn GPU | Cả hai: (1) công tắc bật/tắt từng GPU trong pool; (2) tuỳ chọn tick GPU khi Start model, không chọn thì tự động | Theo lựa chọn của bạn. |
| Frontend | File tĩnh (HTML + CSS + [Alpine.js](https://alpinejs.dev), đóng kèm, ~45 KB), coordinator phục vụ ở `/` | Không cần Node.js hay bước build, chạy được khi không có internet, cùng origin nên không vướng CORS. |
| Đăng nhập UI | Admin key, lưu trong localStorage của trình duyệt, gửi dạng bearer token | Dùng lại `admin_key` sẵn có. |
| Dữ liệu trực tiếp | UI gọi `GET /api/state` mỗi 2 giây | Đơn giản; cụm nhỏ. |
| Tải model | Coordinator tải từ Hugging Face vào `models_dir`; head lấy từ coordinator (`coordinator://`) | Chỉ coordinator cần internet; các server giữ nguyên. |
| Start / Stop | Start = số replica mong muốn ← N (mặc định 1); Stop = ← 0 (drain rồi dừng engine) | Dựa trên reconciler sẵn có. |
| Docker | `gpupool-coordinator` (Python slim, không cần GPU) và `gpupool-agent` (CUDA runtime + llama.cpp build có CUDA và RPC) | Agent chạy với `--gpus all --network host`; host network giữ port RPC và IP đơn giản. |

## 2. Màn hình (theo mockup)

- **Overview**: thẻ số server (online/tổng), GPU (đang bật/tổng), VRAM trống trong pool, số model đang chạy.
- **Servers**: mỗi server một thẻ: trạng thái, IP, thanh CPU/RAM, bảng GPU (utilization, bộ nhớ dùng/tổng,
  nhiệt độ, công suất, số process, trạng thái), checkbox mỗi GPU = bật/tắt trong pool. **Add Server** mở
  hộp thoại nhập URL kèm lệnh `docker run` cho agent để copy. Xoá có xác nhận, dừng engine của server
  đó rồi gỡ khỏi pool.
- **Panel chi tiết GPU** (bấm vào một GPU): vòng utilization, bộ nhớ, nhiệt độ, công suất, danh sách process
  đang chạy (PID, tên, bộ nhớ GPU), thông tin tĩnh (model, CUDA, driver).
- **Models**: thư viện model (file) + triển khai.
  - *Thêm model*: tab **Hugging Face** (repo id → danh sách file `.gguf` lấy từ HF API → chọn một → tải,
    có thanh tiến độ) hoặc tab **Path** (đường dẫn tuyệt đối tới file `.gguf` trên máy coordinator; khi
    chạy Docker là thư mục được mount).
  - Mỗi dòng model: tên, dung lượng, trạng thái (stopped / starting / running / failed + lỗi), endpoint,
    số replica, ctx, **chọn GPU** (Auto hoặc tick GPU trên các server), nút **Start** / **Stop**.
- **Settings**: hiện API base URL, đoạn code client OpenAI, lệnh cài agent.

## 3. Thay đổi contract (`common/models.py`, mọi field mới đều tuỳ chọn để agent cũ vẫn chạy)

```python
class GpuProcess(BaseModel):
    pid: int
    name: str            # "" khi hệ điều hành ẩn (process của user khác)
    used_mb: int | None

class Device(BaseModel):
    ...                  # các field cũ
    temp_c: int | None = None
    power_w: int | None = None
    processes: list[GpuProcess] = []
    driver: str | None = None
    cuda: str | None = None      # bản CUDA tối đa driver hỗ trợ

class NodeReport(BaseModel):
    ...                  # các field cũ
    cpu_pct: float | None = None
    ram_used_mb: int | None = None
    ram_total_mb: int | None = None

class ModelSpec(BaseModel):
    ...                  # các field cũ
    pin_devices: list[str] = []  # "node_id/device_id"; rỗng = scheduler tự chọn
```

Bảng store mới: `servers(node_id, agent_url, added_at)`, `gpu_flags(node_id, device_id, enabled)`,
`library(name, path, source, bytes, status, progress, error)`.

Scheduler không đổi hàm `plan()`: trước khi plan, reconciler đặt `usable_mb = 0` cho GPU bị tắt và, khi
có `pin_devices`, cho mọi device không nằm trong danh sách. Tắt GPU chỉ ảnh hưởng placement mới; replica
đang chạy trên GPU đó vẫn chạy cho tới khi bị Stop.

## 4. HTTP API cho UI (admin key)

| Method | Path | Vào → Ra |
| --- | --- | --- |
| GET | `/api/state` | → `{servers:[{node_id, agent_url, alive, last_seen, report, gpu_enabled:{device_id:bool}}], models:[{spec, file, replicas:[ReplicaRecord + outstanding]}], library:[LibraryItem], summary:{...}}` |
| POST | `/api/servers` | `{agent_url}` → server (400 nếu không gọi được / sai token, 409 nếu node_id đã có) |
| DELETE | `/api/servers/{node_id}` | dừng replica trên server đó, gỡ khỏi pool |
| PUT | `/api/servers/{node_id}/gpus/{device_id}` | `{enabled: bool}` |
| GET | `/api/hf/files?repo=owner/name` | → `[{file, bytes}]` (chỉ `.gguf`; dùng HF token trong env nếu có) |
| POST | `/api/library` | `{hf_repo, hf_file}` hoặc `{path}` → LibraryItem (HF: tải nền) |
| DELETE | `/api/library/{name}` | xoá mục (chỉ xoá file nếu do mình tải; 409 nếu đang có model dùng) |
| PUT | `/api/models/{name}` | `{file, ctx_size, parallel, replicas, pin_devices}` → spec (tạo hoặc sửa) |
| POST | `/api/models/{name}/start` | `{replicas?}` → spec |
| POST | `/api/models/{name}/stop` | → spec (replicas 0) |
| DELETE | `/api/models/{name}` | stop + xoá |
| POST | `/api/models/{name}/plan` | → Placement (chạy thử; cho biết sẽ dùng GPU nào) |

`/files/{name}` (cluster token) chỉ phục vụ các mục trong thư viện theo tên: file đã tải và đường dẫn đã
đăng ký, không gì khác (tra theo tên chứ không ghép đường dẫn, nên không thể path traversal).

## 5. Docker

- `docker/coordinator.Dockerfile`: `python:3.12-slim` + uv, cài app, `EXPOSE 8080`, volume `/data`
  (database + `models`). Cấu hình qua biến môi trường: `GPUPOOL_ADMIN_KEY`, `GPUPOOL_CLUSTER_TOKEN`,
  `GPUPOOL_API_KEYS`, `HF_TOKEN`.
- `docker/agent.Dockerfile`: stage build `nvidia/cuda:12.4.1-devel-ubuntu22.04` biên dịch llama.cpp
  (pin tag, `GGML_CUDA=ON GGML_RPC=ON`, build arg `LLAMA_CPP_REF`, `CUDA_ARCHS`); stage chạy
  `nvidia/cuda:12.4.1-runtime-ubuntu22.04` + uv + app. Cấu hình qua biến môi trường:
  `GPUPOOL_NODE_ID`, `GPUPOOL_HOST` (IP gọi tới được), `GPUPOOL_CLUSTER_TOKEN`, `GPUPOOL_PORT`.
  CUDA 12.4 chạy được trên driver ≥ 525 (tương thích minor version của CUDA).
- `docker-compose.coordinator.yml` và `docker-compose.agent.yml`.
- Lưu ý: Docker cần quyền nhóm `docker` hoặc root trên mỗi server. Cách cài bằng uv vẫn được giữ cho
  server không có quyền Docker.

## 6. Chia việc implement (song song, file không trùng nhau)

| Người làm | File |
| --- | --- |
| Lead | `common/models.py` (contract ở trên), `cli.py`, ghép nối, file Docker, doc |
| Agent A: số liệu agent | `agent/gpu.py`, `agent/app.py` (field report), test |
| Agent B: coordinator server + model | `coordinator/store.py`, `coordinator/poller.py` (mới), `coordinator/reconciler.py`, `coordinator/api.py` (mới, `/api/servers*`, `/api/models*`, `/api/state`), test |
| Agent C: thư viện + Hugging Face | `coordinator/library.py` (mới: bảng store, liệt kê HF, tải nền có tiến độ, đăng ký path, tra `/files`), `coordinator/library_api.py` (router mới), test |
| Agent D: frontend | `src/gpupool/ui/` (index.html, app.js, styles.css, vendor/alpine.min.js) |

Ghép `coordinator/app.py` là việc của lead sau khi các agent xong.

## 7. Kế hoạch test

- Unit test từng module (mock HF API, agent giả).
- Chạy thật trên laptop này: coordinator + 3 agent; trên trình duyệt: thêm server bằng URL, tắt một GPU,
  tải một GGUF nhỏ từ Hugging Face, Start → chat trả lời → Stop → engine biến mất, xoá một server. Chụp
  màn hình UI cho báo cáo.
- Docker: build image trên GitHub Actions (laptop này không có Docker). Chạy image agent với GPU **không**
  test được ở đây và sẽ được ghi rõ là chưa test.
