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
    số replica, ctx, **server/GPU được phép** (tất cả, hoặc chỉ những cái đã tick; xem mục 10), nút **Start** / **Stop**.
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
    pin_devices: list[str] = []  # tập được phép: "node_id/device_id" hoặc "node_id/*"; rỗng = tất cả
```

Bảng store mới: `servers(node_id, agent_url, added_at)`, `gpu_flags(node_id, device_id, enabled)` (cột thứ hai lưu `uuid` của GPU khi agent có báo, để cờ bật/tắt đi theo card vật lý; các dòng cũ theo `CUDA<i>` được chuyển sang uuid ở report đầu tiên có uuid),
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
- `docker/agent.Dockerfile`: stage build `nvidia/cuda:12.8.1-devel-ubuntu22.04` biên dịch llama.cpp
  (pin tag, `GGML_CUDA=ON GGML_RPC=ON`, build arg `LLAMA_CPP_REF`, `CUDA_ARCHS`); stage chạy
  `nvidia/cuda:12.8.1-runtime-ubuntu22.04` + uv + app. Cấu hình qua biến môi trường:
  `GPUPOOL_NODE_ID`, `GPUPOOL_HOST` (IP gọi tới được), `GPUPOOL_CLUSTER_TOKEN`, `GPUPOOL_PORT`.
  CUDA 12.8 chạy được trên driver ≥ 525 (tương thích minor version của CUDA); card RTX 50 cần ≥ 570.
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

## 8. Phát hiện sự cố và thông báo (bổ sung 2026-10-02)

Một server có thể mất điện, hoặc một GPU rớt khỏi bus, trong lúc model đang chạy trên đó.

| Tình huống | Phát hiện bằng | Xử lý |
| --- | --- | --- |
| Server chết (mất điện, mất mạng, agent crash) | không lấy được `/report` trong `heartbeat_timeout_s` (10 giây; hỏi mỗi 2 giây) | replica dùng server đó → `failed`; engine còn sống ở server khác bị dừng; xếp lại trên các GPU còn lại |
| Một GPU mất, server vẫn sống | GPU biến mất khỏi report của server | replica dùng GPU đó → `failed`, xếp lại |
| Tiến trình llama.cpp crash | engine `exited`/`failed` trong report | replica → `failed`, xếp lại |
| Không còn đủ chỗ | `plan()` báo `NoFit` | model hiện failed kèm "cần X GB, pool còn Y GB"; tự thử lại khi có chỗ |

Mỗi lần chuyển trạng thái được ghi thành một **event** (`info` / `warning` / `error`): `node_offline`,
`node_online`, `gpu_missing`, `engine_crashed`, `realloc_started`, `realloc_done`, `realloc_failed`,
`launch_failed`, cùng các thao tác với server/model. Mỗi lần chuyển trạng thái chỉ sinh một event, không
lặp lại mỗi vòng reconcile.

Người dùng được báo qua: chuông có số chưa đọc và toast trên UI, trang Events, thông báo desktop tuỳ
chọn (Notification API của trình duyệt), và webhook tuỳ chọn (`webhook_url`, tương thích Slack hoặc
Discord) cho mức warning và error, để vẫn nhận báo động khi không mở UI.

Request đang chạy trên replica bị chết sẽ mất, trừ khi chưa nhận byte nào (router thử lại trên replica
khác). Request mới đi ngay sang các replica còn lại; nếu model chỉ có một replica thì sẽ gián đoạn cho tới
khi cấp phát lại xong (vài giây với model nhỏ, lâu hơn khi weights phải truyền qua RPC).

## 9. Giao diện chuyển đổi mô hình (Hugging Face / thư mục sang GGUF)

Coordinator biến trọng số Hugging Face thành GGUF (xem PLATFORM_DESIGN về pipeline). UI gồm hai phần:
**hộp thoại Convert** và **bảng Conversions** ở trang Models. Cả hai dùng các design token sẵn có
(`--surface-2`, `--accent`, `--warning-soft`, `--danger-soft`, `--info-soft`, `.pill`, `.chip`,
`.notice`, `.seg`, `.tabs`); không dùng mã màu cứng.

### 9.1 Lối vào

- Nút **Convert a model** ở đầu thư viện và ở đầu bảng Conversions.
- Trong *Add model*: khi một repo Hugging Face không có file `.gguf`, một thông báo đề nghị
  **Convert to GGUF** (repo được điền sẵn và kiểm tra ngay); khi chưa liệt kê được gì thì có liên kết
  "Not a GGUF repository? Convert it".
- Dòng thư viện của file đã chuyển hiện nguồn **Converted**.

### 9.2 Hộp thoại Convert

1. **Tab nguồn**: *Hugging Face repo* (repo id, revision tuỳ chọn, Inspect) và *Folder on the server*
   (đường dẫn tuyệt đối; khi coordinator chạy trong Docker, đường dẫn của máy host được dịch như
   đường dẫn thư viện). Nhấn Enter trong ô nhập sẽ bắt đầu kiểm tra. Lỗi (404, gated, thư mục sai,
   **507 không đủ dung lượng đĩa**: thông báo nêu thư mục, số GB cần và số GB còn trống) hiện trong khối
   đỏ dưới các tab.
2. **Thông tin kiểm tra**: kiến trúc, số tham số, số layer, độ dài context, định dạng trọng số, dung
   lượng tải (hoặc dung lượng trên đĩa) trong lưới sáu ô, một pill trạng thái (Supported / Not
   supported / Support unknown / Cannot convert), danh sách file được dùng và bị bỏ qua (thu gọn được).
3. **Thông báo** (chỉ hiện cái nào áp dụng): không thể chuyển (kèm *Inspect the base model instead*
   khi repo đã lượng tử hoá sẵn), repo gated, đã lượng tử hoá, repo có mã Python riêng, bản GGUF có
   sẵn (*download instead* chuyển sang Add model), cảnh báo của converter.
4. **Bộ chọn lượng tử hoá**: danh sách radio, chất lượng tốt nhất trước. Mỗi dòng: tên loại, pill tier
   (Lossless, Near-lossless, Balanced, Small, Tiny), *Recommended*, huy hiệu **Needs calibration**
   (loại cần importance matrix), huy hiệu vừa vặn (*Fits one GPU* / *Needs several GPUs* / *Does not
   fit the cluster*), ghi chú chất lượng, dung lượng file và VRAM ước tính. Khuyến nghị và lý do hiện
   phía trên danh sách. Các loại dưới 3 bit mỗi trọng số (IQ2, IQ1) được gập vào **Show smaller,
   lower-quality types (n)**; loại đang chọn luôn hiển thị. Khi `imatrix_available` là false trong
   `GET /api/convert/options`, các loại cần importance matrix bị làm mờ, vô hiệu hoá và nêu lý do.
5. **Tên file đầu ra**: ghép từ `name_stem` của bước kiểm tra và tên loại (`<name_stem>-<QUANT>.gguf`);
   tên tự đổi theo loại cho đến khi người dùng sửa. Tên không hợp lệ chặn nút Start.
6. **Keep downloaded source** (chỉ Hugging Face) và mục **Advanced** thu gọn được: độ chính xác trung
   gian, kiểu output tensor, kiểu token embedding, leave output tensor, pure (bốn mục này bị vô hiệu
   hoá với F16/BF16/Q8_0 vì converter ghi thẳng), **Importance matrix**, validate generation, allow
   remote code (cảnh báo đỏ), threads.
   - *Importance matrix* là điều khiển ba lựa chọn **Auto / On / Off**. Phần trợ giúp giải thích: nó là
     gì (một lượt chạy mô hình trên văn bản mẫu để ghi lại trọng số nào quan trọng, giúp bước lượng tử
     hoá giữ chúng chính xác hơn), Auto nghĩa là bật cho loại dưới khoảng 4 bit và loại bắt buộc có
     nó, On tốn thêm thời gian (xấp xỉ một lượt chạy mô hình trên văn bản, bằng CPU), và Off không
     dùng được cho loại bắt buộc có nó (nút bị vô hiệu hoá; chọn loại như vậy khi đang để Off thì tự
     chuyển sang Auto). Một dòng dưới điều khiển cho biết có bước Calibrate hay không. Với
     F16/BF16/Q8_0 cả khối bị vô hiệu hoá kèm giải thích. Khi thiếu llama-imatrix, điều khiển cố định
     ở Off.
   - **Calibration text** (tuỳ chọn): đường dẫn tuyệt đối của file `.txt` trên máy chủ, tối đa 20 MB;
     để trống = văn bản đa ngôn ngữ có sẵn của gpupool. Được kiểm tra ngay trên trình duyệt (tuyệt đối,
     đuôi `.txt`).
   - **Calibration chunks**: số đoạn 512 token; 0 = mặc định (100).
7. Chân hộp thoại: lý do nút Start bị khoá (thiếu toolchain, tên sai, đường dẫn calibration sai) và
   **Start conversion**. Khi thành công hộp thoại đóng, trang Models mở ra và có toast xác nhận.

### 9.3 Bảng Conversions

Mỗi job một thẻ, mới nhất trước:

- Đầu thẻ: tên file đầu ra, pill trạng thái (có spinner khi đang chạy), chip loại lượng tử hoá, chip
  *importance matrix* khi job có tính nó; dòng nguồn và thời gian (*queued 2m ago*, *started 2m ago*,
  hoặc *ran 3m 20s, finished 5m ago* tính từ `started_at` / `finished_at`).
- **Stepper**: Download, Convert, **Calibrate** (chỉ khi `imatrix_used`), Quantize, Validate. Bước xong
  màu xanh, bước đang chạy nhấp nháy, job lỗi đánh dấu đỏ ở `failed_stage`, job bị huỷ đánh dấu bằng
  màu cảnh báo; Download là nét đứt (bỏ qua) với thư mục, Quantize với các loại ghi thẳng. Job không có
  `failed_stage` dùng bước cuối cùng thấy được khi polling. Trên điện thoại chỉ bước đang chạy hoặc
  bước lỗi còn giữ nhãn.
- Thanh tiến độ (có phần trăm cho tải và cho các bước có `stage_progress`, còn lại là thanh chạy qua
  lại) kèm chữ (số byte, phần trăm, "Computing the importance matrix on the CPU: 45%").
- Meta: dung lượng ước tính, dung lượng cuối, file nguồn được giữ lại.
- Khung lỗi (`error`), thông báo *Needs your review* cho `needs_review`.
- **Bảng validation** (thu gọn được, tự mở với needs_review): chip kiểm tra (header GGUF, tokenizer,
  văn bản sinh ra, chat template), kiến trúc / số layer / từ vựng, lỗi và cảnh báo, bảng tokenizer (id
  HF so với id GGUF, dòng lệch được tô nổi) và mẫu văn bản sinh ra.
- **Log** (thu gọn được): `log_tail`.
- Hành động theo trạng thái: *Deploy this model* (done, mở form New model), *Accept anyway*
  (needs_review, có hỏi xác nhận), *Cancel* (đang chạy), *Retry* (failed / cancelled), biểu tượng xoá
  (không đang chạy); mọi thao tác có tác dụng huỷ dữ liệu đều hỏi lại.

### 9.4 Polling, toast, các trạng thái

- `GET /api/convert` mỗi 2 giây khi có job đang chạy, mỗi 10 giây khi không, lúc trang Models đang mở.
  Các chuyển trạng thái thấy được giữa hai lần poll sinh toast: đã chuyển xong (và thư viện được làm
  mới), cần xem xét, thất bại.
- Trạng thái trống: "No conversions yet". Trạng thái không khả dụng: thông báo cảnh báo kèm lý do từ
  `options.problem` (vẫn kiểm tra được, không bắt đầu được); 404 nghĩa là coordinator không hỗ trợ
  chuyển đổi. Lỗi polling hiện trong khối đỏ, không bao giờ hiện như một danh sách trống.

### 9.5 Điện thoại và khả năng truy cập

- Hộp thoại là modal vừa khít màn hình; các dòng lượng tử hoá xuống dòng (số liệu chuyển xuống dưới
  chữ), lưới thông tin co lại, không cuộn ngang (đã kiểm tra ở 375 px).
- Bộ chọn là một `radiogroup` với nhãn cho từng radio; nút mở nhóm bit thấp có `aria-expanded`; điều
  khiển importance matrix là một `radiogroup`; thanh tiến độ dùng `role="progressbar"`; mỗi bước của
  stepper có trạng thái cho trình đọc màn hình; lỗi dùng `role="alert"`; hộp thoại có `aria-modal` và
  tiêu đề. Màu không bao giờ là tín hiệu duy nhất (mọi pill đều có chữ).

## 10. Thẻ model: nút sao chép, và giới hạn model vào một số server hoặc GPU

### 10.1 Sao chép

Dòng endpoint của thẻ model có hai dòng: **endpoint** với nút biểu tượng *Copy endpoint*, và **model**
(tên mà client điền vào `"model"`) với nút biểu tượng *Copy model name* cùng một thao tác nhỏ *Copy
curl* sao chép sẵn lệnh `curl` cho đúng model này (có chỗ giữ chỗ cho header `Authorization` khi đã đặt
API key). Mỗi nút hiện toast "Copied". Các dòng tự xuống hàng thay vì tràn trên điện thoại.

### 10.2 Server và GPU được phép (`pin_devices`)

`pin_devices` là một **tập được phép**, không phải đặt chỗ thủ công: scheduler vẫn chọn cách đặt tốt
nhất, nhưng chỉ trên các thiết bị trong tập. Phần tử là `"<node>/<device>"` (một GPU) hoặc
`"<node>/*"` (mọi thiết bị của server đó, kể cả GPU thêm vào sau). Rỗng = tất cả. Việc này không đổi gì
ở tab Servers: GPU vẫn bật trong pool cho các model khác.

Trong form New / Edit model:

- **Use**: điều khiển hai lựa chọn *All servers and GPUs* (mặc định) / *Only selected ones*, kèm phần
  trợ giúp nói trên.
- **Cây lựa chọn** (hiện khi chọn *Only selected ones*): mỗi server một checkbox (tích = `"<node>/*"`,
  trạng thái lưng chừng khi chỉ chọn một số GPU) và các GPU của nó bên dưới, mỗi GPU có bộ nhớ trống
  và dùng được. Tích một server lưu `"<node>/*"` và bỏ các pin GPU lẻ của server đó; bỏ tích một GPU của
  lựa chọn cả server sẽ đổi thành các pin `"<node>/<device>"` còn lại một cách tường minh. Server offline
  và GPU bị tắt được làm mờ kèm lý do nhưng vẫn chọn được.
- Một dòng tóm tắt ("3 of 7 GPUs on 2 servers"); khi chưa chọn gì thì có cảnh báo và nút Save từ chối
  (danh sách rỗng sẽ có nghĩa "tất cả").
- Pin đã lưu, kể cả ký tự thay thế, được nạp lại vào cây khi mở lại form. Recommend, Check placement và
  Preview impact gửi cùng `pin_devices`.
- Thẻ model hiện chip như *Limited to server-a, server-b/CUDA1* (`/*` hiển thị bằng tên server).

## 11. Các trường hiệu năng trong form model

Phần *Performance* của form New / Edit model ánh xạ một-một vào các trường `ModelSpec`; Recommend, Check
placement và Preview impact cũng gửi chúng, nên mọi ước lượng đều thấy cùng một cấu hình.

- **Parallel slots** và, khi có hơn một slot, công tắc **Share the context between slots** (`kv_unified`). Bên
  dưới là dòng cho biết context mỗi request nhận được: `context ÷ slots` (màu hổ phách khi dưới 2048), hoặc cả
  context "at most, shared with the other slots" khi bật công tắc.
- **KV cache** (`kv_cache_type`), **Flash attention**, **Micro-batch** và **Batch**, kèm cảnh báo khi cache lượng
  tử hóa đi với flash attention *Off*.
- **Speculative decoding**: *Off*, *N-gram*, *Draft model* (một file trong thư viện và số token) hoặc *MTP* (chính
  các layer dự đoán của model, kèm số token). Dòng trợ giúp dưới *MTP* nói GGUF nào có các layer này; server trả
  422 với model không có, hiển thị như mọi lỗi khác.
- Một gợi ý Recommend có `apply` đặt bất kỳ trường nào ở trên (`kv_unified`, `speculative`, `draft_n_max`, ...)
  sẽ điền chúng vào form; chip trên thẻ model hiện *Spec: MTP* và *N slots · ctx shared*.
