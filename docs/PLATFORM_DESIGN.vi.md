# gpupool — Thiết kế platform đa model, đa server (đã triển khai)

> Bản tiếng Việt. Bản tiếng Anh: [PLATFORM_DESIGN.en.md](PLATFORM_DESIGN.en.md). Giữ hai bản đồng bộ.

**Trạng thái: đã triển khai.** Cả bốn giai đoạn của lộ trình (mục 9) đã có trong code, cùng ba bổ sung
muộn hơn: tự hiệu chỉnh VRAM theo từng model và lưu bền trạng thái điều khiển (các câu hỏi mở của thiết kế
đã đòi hỏi hai việc này), và chuyển đổi model Hugging Face sang GGUF (mục 6). Đây là tài liệu giải thích lý do thiết kế: vì sao hệ thống hành xử như vậy. Giao diện HTTP chính
xác nằm ở [API.vi.md](API.vi.md).

Mục tiêu: gpupool quản lý **nhiều model trên nhiều server, nhiều GPU** cùng lúc, chia tài nguyên có
chủ đích (ưu tiên, dàn trải, tự co giãn theo tải) và **gợi ý GPU/server** cho từng model kèm lý do và số
ước lượng. Tài liệu này mô tả hành vi đã thúc đẩy thiết kế, mô hình tài nguyên, thuật toán phân bổ, cách model vào thư viện
(chuyển đổi, đồng thời chọn kiểu lượng tử hóa vừa với cụm), tóm tắt API, thay đổi dữ liệu và lộ trình triển khai.

## 1. Mốc ban đầu: hành vi khi chạy nhiều model trước thiết kế này

Đo bằng reconciler và scheduler **thật** như trước giai đoạn 1 (agent giả, metadata GGUF thật của Qwen2.5
0.5B = 587 MB và 3B = 2191 MB ở ctx 4096). Mỗi vấn đề dưới đây về sau đã được xử lý ở mục tiếp theo.

| Tình huống | Kết quả | Vấn đề |
| --- | --- | --- |
| 2 replica của `chat`, server a: 2×8 GB, b: 1×8 GB | cả 2 replica nằm trên **a/CUDA0** | không chịu lỗi (1 GPU chết = mất cả model), 2 replica tranh nhau một GPU trong khi 2 GPU rảnh |
| `alpha` và `zeta` (3B), một GPU 4 GB chỉ đủ 1 model | `alpha` chạy, `zeta` NoFit | thắng thua theo **thứ tự tên**; không có cách khai báo model nào quan trọng hơn |
| `big` 3B + `small` 0.5B, a: 8 GB + 8 GB | cả hai trên **a/CUDA0**, CUDA1 bỏ trống | scheduler chỉ nhìn bộ nhớ; hai model cùng chạy sẽ chia đôi băng thông của một GPU |
| `small` rồi `big`, a: 3 GB + 2.4 GB | `big` → CUDA0, `small` → CUDA1 | đúng: best-fit theo bộ nhớ hoạt động tốt |

Nguyên nhân gốc, đọc từ code lúc đó:

- `Reconciler._enforce_counts` duyệt model theo `ORDER BY name`, mỗi tick một replica mỗi model. Không có
  khái niệm ưu tiên, không có giành chỗ (preemption).
- `scheduler.placement.plan` là hàm thuần chỉ thấy `usable_mb`. Nó không biết replica nào khác đang chạy
  ở đâu, không biết GPU nào nhanh hơn, không biết GPU nào đang bận. Tier `single_gpu` chọn GPU **nhỏ
  nhất còn vừa** (best-fit), nên tự nhiên dồn mọi thứ vào cùng một GPU.
- `replicas` là số cố định. Model không ai dùng vẫn giữ VRAM mãi, model quá tải không tự thêm replica.
- NoFit là tất cả hoặc không có gì: không gợi ý giảm `ctx_size`, không nói thiếu bao nhiêu, ở đâu.

## 2. Mô hình tài nguyên

Một GPU có hai loại tài nguyên, và chúng cư xử khác nhau:

| Tài nguyên | Tính chất | Nguồn số liệu | Cách dùng |
| --- | --- | --- | --- |
| VRAM | **cứng**: thiếu là không load được | `usable_mb`, ước lượng `est_mb` (`scheduler/estimate.py`), được hiệu chỉnh theo từng model bằng hệ số calibration (mục 4.7) | ràng buộc bắt buộc |
| Băng thông bộ nhớ | **mềm**: chia sẻ thì chậm đi | NVML: bus width × mem clock (đo trên GTX 1650 Ti: 128 bit × 5001 MHz ≈ 160 GB/s) | ước lượng tok/s, điểm hiệu năng |
| Mức bận | **mềm**, thay đổi theo thời gian | llama-server `/metrics`: `requests_processing`, `requests_deferred`, `predicted_tokens_seconds` (đã kiểm trên b11342); router `outstanding` | phạt khi ghép chung GPU, tín hiệu autoscale |

**Ước lượng tốc độ decode.** Sinh mỗi token phải đọc toàn bộ trọng số một lần, nên decode bị giới hạn bởi
băng thông:

```latex
t_{token} = \sum_{d \in \text{devices}} \frac{\text{bytes}_d}{BW_d \cdot \eta} + n_{rpc} \cdot t_{hop}
\qquad \text{tok/s} \approx 1 / t_{token}
```

`bytes_d` là tổng `layer_bytes` của các layer đặt trên thiết bị d (đã có trong `ModelMeta`; tensor đầu ra
tính vào thiết bị cuối). Với layer MoE chỉ tính số byte một token đọc: phần weight dùng chung
cộng `expert_used_count / expert_count` của các expert được định tuyến (`ModelMeta.active_bytes`). `n_rpc` đếm số
RPC server, không phải số thiết bị ở xa: các GPU của một server mà replica dùng nằm sau một `ggml-rpc-server`
(agent có tính năng `rpc_multi_device`), server này tự copy activation giữa chúng. η là hiệu suất thực tế: số đo trong `TEST_REPORT` cho η ≈ 0.45 với 0.5B (182
tok/s trên lý thuyết ~400) và ≈ 0.6 với 3B (51 tok/s trên lý thuyết ~84). Code dùng hằng số **η = 0.5** và
`t_hop` = 2 ms mỗi bước nhảy RPC (`scheduler/scoring.py`). GPU chưa biết băng thông được gán giá trị mặc
định. Prefill phụ thuộc compute hơn băng thông; ước lượng chỉ xếp hạng theo decode. Tốc độ decode đo được
được đọc từ `/metrics` và hiển thị cạnh số ước lượng ở `GET /api/models/{name}/scaling`, nhưng **không**
được đưa ngược lại vào η (xem câu hỏi mở).

**Trước khi file tồn tại.** Với model chưa được chuyển đổi thì chưa có header GGUF để đọc, nên nhu cầu VRAM được
ước lượng từ `config.json` và kiểu lượng tử hóa (mục 6.2). Ước lượng đó chỉ dẫn hướng việc chọn kiểu; khi GGUF đã
có, scheduler dùng header thật và hệ số hiệu chỉnh.

**Ghép chung GPU.** Hai model cùng bận trên một GPU chia nhau băng thông, mỗi model còn khoảng một nửa
tốc độ. Hai model mà một bên hầu như rảnh thì ghép được. Vì vậy hình phạt ghép chung dựa trên **mức bận
đo được**, không cấm cứng.

## 3. Chính sách cho từng model

Mọi trường đều tùy chọn và có giá trị mặc định giữ hành vi của một model số replica cố định. Bảng trường đầy
đủ, kèm kiểu và giới hạn, nằm ở [API.vi.md](API.vi.md#2-modelspec).

| Trường | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `priority` | 50 | 0–100. Model ưu tiên cao được xếp trước, và khi thiếu chỗ có thể giành chỗ của model ưu tiên thấp hơn |
| `min_replicas` | = `replicas` | số replica luôn giữ (0 = cho phép dỡ khi rảnh) |
| `max_replicas` | = `replicas` | > `min_replicas` thì bật autoscale |
| `autoscale` | `{target_busy: 0.7, up_after_s: 30, down_after_s: 300}` | ngưỡng thêm/bớt replica theo tỷ lệ slot bận |
| `idle_unload_s` | null | chỉ khi `min_replicas = 0`: không có request trong N giây thì dỡ model; request đầu tiên sẽ load lại |
| `spread` | `"gpu"` | `gpu`: các replica ưu tiên khác GPU; `node`: khác server; `none`: không quan tâm |
| `pin_devices` | rỗng | các mục `"node_id/device_id"` mà replica được dùng; rỗng = chọn tự do |
| `preemptible` | true | false = không bao giờ bị giành chỗ |
| `kv_cache_type`, `speculative`, `draft`, `draft_n_max`, `flash_attn`, `batch`, `ubatch`, `kv_unified` | `f16`, `none`, null, 4, `auto`, 2048, 512, false | tùy chọn bộ nhớ/tốc độ mà phần ước lượng và planner tính đến (thêm sau thiết kế này); `speculative` là `none`, `ngram`, `draft` hoặc `mtp` |

`replicas` là công tắc bật/tắt: 0 là dừng model bất kể `min/max`. Khi `min_replicas` và `max_replicas` chưa
đặt thì cả hai bằng `replicas`, nên model số lượng cố định hành xử như trước.

**Quyết định không xây.** Bản nháp đầu còn đề xuất `share_gpu`, `gpu_selector` (nhãn, `min_vram_mb`,
`min_bandwidth_gbps`), **nhãn** cho server/GPU và `reserve_mb` theo từng GPU. Chúng không có trong code.
`pin_devices` đáp ứng việc đặt chỉ định rõ, và `budget_mb` theo thiết bị trên agent (đã trừ cả VRAM do chính
các replica của gpupool giữ) đáp ứng việc giữ lại VRAM cho việc khác. Nhãn chỉ nên thêm nếu xuất hiện một
cụm mà pin từng thiết bị là quá thô.

## 4. Thuật toán phân bổ

```mermaid
flowchart LR
  A[Autoscaler: số replica mong muốn mỗi model] --> B[Sắp model: priority, model chưa có replica nào trước]
  B --> C[Sinh các phương án đặt khả thi]
  C --> D[Chấm điểm và chọn phương án tốt nhất]
  D -->|không có phương án| E[Thử giành chỗ model ưu tiên thấp hơn]
  D --> F[Launch]
  E --> F
  G[Rebalancer, chạy thưa] -->|make-before-break| F
```

**4.1 Thứ tự.** Mỗi tick, sắp model theo `priority` giảm dần. Trong cùng mức, model **chưa có replica
nào** đi trước model đã có (để mọi model có ít nhất 1 replica rồi mới tới replica thứ hai), sau đó theo
**tên** (một tiêu chí phân định ổn định; bản nháp đầu ghi là thời gian tạo). Thứ tự tên giờ chỉ dùng để phá
hòa.

**4.2 Sinh phương án.** `scheduler/placement.py` sinh nhiều phương án khả thi thay vì một phương án mỗi
tier: mỗi GPU đơn đủ chỗ, các tổ hợp nhiều GPU trong một node, và các phương án multi-node (chỉ khi cần;
các biến thể tham lam trước, rồi các tập con theo băng thông). `plan()` trả phương án tốt nhất đã cấp cổng;
`rank()` trả top `k` không cấp cổng, được `recommend`, `simulate` và rebalance dùng. Số GPU trong một cụm
thực tế nhỏ (vài chục), nên liệt kê được. Thuật toán chia layer (`_split`) giữ nguyên. Model draft của
speculative được đặt nguyên khối trên thiết bị CUDA đầu tiên của head và tính vào `est_mb` của thiết bị đó.

**4.3 Chấm điểm.** Mỗi phương án có điểm; điểm cao thắng. Các trọng số là hằng số trong
`scheduler/placement.py`, không phải cấu hình:

```latex
\text{score} = 100 \cdot \frac{tps}{tps_{best}} - \sum_{o \in \text{engines on chosen GPUs}} (10 + 30 \cdot busy_o) - 40 \cdot same_{gpu} - 20 \cdot same_{node} - 15 \cdot waste - 5 \cdot (n_{dev}-1) - 10 \cdot n_{rpc}
```

- `tps / tps_best`: tốc độ decode ước lượng so với phương án nhanh nhất.
- Mỗi engine đã có trên GPU được chọn bị trừ 10, cộng 30 nhân mức bận (0–1) của nó.
- `same_gpu`, `same_node`: số replica cùng model đã có trên GPU / server đó (theo `spread`: `gpu` và `node`
  phạt GPU dùng chung; chỉ `node` phạt server dùng chung).
- `waste`: trung bình của `usable / usable lớn nhất` trên các GPU được chọn. Đây là cách giữ lại ưu điểm
  của best-fit: không xé nhỏ một GPU lớn khi GPU vừa khít còn trống.
- `n_dev`, `n_rpc`: số thiết bị thêm và số lần nhảy qua mạng (mỗi RPC server một lần).

Hòa điểm thì chọn tier nhỏ hơn, rồi tên node head và thiết bị đầu tiên của nó, nên kết quả xác định. Phương án thắng được lưu
cùng replica (`score`, `est_decode_tps`, `reasons`), để UI giải thích được vì sao model nằm ở đó.

**4.4 Giành chỗ (preemption).** Model đang dưới mức tối thiểu đang chạy (`max(min_replicas, 1)`, bị chặn bởi
số lượng mong muốn, nên replica thêm do autoscale không bao giờ đi giành) và không có phương án khả thi có
thể dừng replica của model ưu tiên thấp hơn:

1. Nạn nhân hợp lệ: replica của model `preemptible` có priority **thấp hơn hẳn** P (bằng nhau thì không
   bao giờ giành). Thứ tự: replica vượt quá mức tối thiểu đang chạy của model đó trước, rồi cái ít bận
   nhất, rồi cái mới nhất.
2. Thêm dần nạn nhân vào tập giả lập "đã gỡ" cho tới khi có phương án, rồi bỏ lại những nạn nhân hóa ra
   không cần. Kết quả là một tập nhỏ, không nhất thiết nhỏ nhất có thể.
3. Drain chúng (cơ chế `drain` hiện có: chờ request đang chạy, tối đa `drain_timeout_s`). Model đi giành
   chỗ không được đặt ngay trong cùng tick; replica đang drain vẫn giữ bộ nhớ, nên một tick sau mới đặt.
   Mỗi nạn nhân phát một event cảnh báo `preempted`.
4. **Cooldown** 10 phút theo từng model đi giành, và không gỡ thêm khi các nạn nhân trước còn đang drain, để
   hai model không giành qua giành lại.
5. **Giữ chỗ (claim).** Trong lúc model đi giành chưa có replica (tối đa `drain_timeout_s` + 120 s), model
   ưu tiên thấp hơn không được launch. Thiếu cơ chế này, model bị gỡ launch lại ngay vào phần bộ nhớ đang
   giải phóng (lỗi tìm ra trên phần cứng thật).

**4.5 Autoscaler.** Mỗi replica có `busy = requests_processing / parallel`, đọc từ `/metrics` của
llama-server mỗi `poll_s` (lần scrape cũ hơn 3 chu kỳ thì thay bằng số request đang chạy do router đếm).
`requests_deferred > 0` nghĩa là request đang phải xếp hàng.

- Tăng 1 replica khi trung bình `busy` > `target_busy`, hoặc có request xếp hàng, liên tục `up_after_s` và
  không còn replica nào đang launch. Không vượt `max_replicas`.
- Giảm 1 replica khi `busy` < `target_busy / 2` và không có request xếp hàng liên tục `down_after_s`.
  Không xuống dưới mức sàn đang chạy `max(min_replicas, 1)`: về 0 chỉ do `idle_unload_s` thực hiện.
- `idle_unload_s`: không có request trong khoảng đó và không có request đang chạy, với `min_replicas = 0`
  thì desired về 0.
- **Khởi động lạnh**: router nhận request cho model đã start có `min_replicas = 0` và chưa có replica thì
  autoscaler đặt desired = 1, phát `cold_start` và đánh thức reconciler; router giữ request tối đa
  `cold_start_timeout_s` (mặc định 120 s). Quá hạn thì trả 503 kèm `Retry-After`.
- **Lưu bền.** Số lượng desired, thời điểm request gần nhất và quyết định gần nhất được ghi vào store
  (`control_state`, khóa `autoscaler:<model>`) khi chúng đổi, nên khởi động lại vẫn giữ model đã dỡ ở trạng
  thái dỡ và model đã tăng ở trạng thái tăng. Bộ đếm thời gian tăng/giảm không được lưu: khởi động lại chỉ
  làm chậm một bước đúng bằng `up_after_s`/`down_after_s`.

**4.6 Cân bằng lại (rebalance).** Chạy thưa (mỗi `rebalance_s`, mặc định 600 s, 0 = tắt; lần chạy đầu cách
lúc khởi động một chu kỳ vì ngay sau khi khởi động các report còn cũ) hoặc theo yêu cầu
(`POST /api/rebalance`). Mục tiêu: replica nay có điểm cao hơn ít nhất 25 ở nơi khác (ví dụ model đang ở
CPU hoặc chia 2 GPU nay vừa 1 GPU); lần di chuyển nằm trong `pin_devices` của model. Cách làm là
**make-before-break**: launch replica mới (được lập kế hoạch trên các thiết bị đích trong lúc replica cũ vẫn
giữ bộ nhớ), chờ ready, rồi drain replica cũ. Mỗi lần chỉ di chuyển một replica trong cả cụm, và chỉ khi cụm
đang yên: không có replica đang launch, không có lần giành chỗ nào đang chờ bộ nhớ. Lần di chuyển mà replica
mới lỗi, hoặc không ready trong `launch_timeout_s` + 60 s, bị hủy với cảnh báo `rebalance_failed` và
replica cũ tiếp tục phục vụ.

**4.7 Tự hiệu chỉnh VRAM.** Ước lượng bộ nhớ chính xác tới từng MiB với các trường hợp đã đo trên GTX 1650,
nhưng chưa bao giờ được đo với model lớn chia nhiều GPU. Thay vì tin nó, mỗi model tự học một hệ số hiệu
chỉnh. Sau khi replica ready, coordinator đọc những gì llama.cpp đã báo lúc load
(`GET /engines/{id}/memory` trên agent của head), cộng các buffer đo được trên các thiết bị của phương án
đặt, và so với ước lượng (chia lại cho hệ số đã dùng lúc lập phương án, trừ phần context runtime mà
llama.cpp không báo). Tỷ lệ được gộp vào một hệ số
riêng của model bằng trung bình trượt mũ (trọng số 0.5) và lưu trong `model_calibration`. Khi đặt, nhu cầu
thiết bị của model được nhân với hệ số này, giới hạn trong 0.9–2.0. Event `calibrated` được phát khi hệ số
đổi hơn 5 %; mẫu có dữ liệu thiếu bị bỏ qua thay vì làm lệch hệ số. Hệ số hiển thị là `calibration` trong
`GET /api/state`.

**4.8 Lưu bền trạng thái điều khiển.** Ngoài trạng thái của autoscaler, reconciler ghi vào `control_state`
các cooldown và claim giành chỗ (`preempted`), backoff khi crash lặp (`backoff`) và lần di chuyển replica
đang diễn ra (`move`), rồi nạp lại khi khởi động. Nhờ vậy coordinator khởi động lại không quên cooldown (để
hai model lại giành nhau), không quên backoff (dồn dập model đang crash) hay quên nửa lần di chuyển (để lại
hai replica của cùng một model). Lần di chuyển mà các replica của nó không còn tồn tại được xóa lặng lẽ.
Thời gian dùng đồng hồ thật. Thời điểm rebalance lần cuối cố ý không được lưu.

## 5. Gợi ý GPU/server

Trả lời câu hỏi "model này nên chạy ở đâu?" **trước khi** deploy, với cùng bộ chấm điểm như scheduler,
nên điều được gợi ý cũng là điều scheduler sẽ làm.

- Xếp hạng tối đa `limit` phương án, mỗi phương án có VRAM dự kiến, tok/s ước lượng, điểm và lý do.
- Có đặt được ngay không, hay cần giành chỗ của ai (`requires_preemption`, liệt kê các replica cần dừng).
- Báo `ctx_size` lớn nhất vừa một GPU ngay bây giờ (`max_ctx_single_gpu`).
- Không đặt được thì nói rõ vì sao: bộ nhớ cần, GPU đơn và node trống lớn nhất, và `ctx_size` lớn nhất sẽ
  vừa (`not_possible`).

Cùng câu hỏi ấy được hỏi sớm hơn một bước với model phải chuyển đổi trước: kiểu lượng tử hóa nào sẽ vừa cụm này.
Gợi ý đó (mục 6.2) dùng GPU lớn nhất và tổng VRAM của cụm thay vì chấm điểm của scheduler, vì lúc ấy chưa có
file nào để đặt.

## 6. Đưa model vào hệ thống: chuyển đổi

Scheduler đặt các file GGUF, nhưng rất nhiều model chỉ được phát hành dưới dạng safetensors hoặc trọng số
PyTorch. Nếu không có cách chuyển chúng sang GGUF, nền tảng bị giới hạn ở những gì người khác đã chuyển sẵn,
với kiểu lượng tử hóa mà họ tình cờ chọn. Mục này giải thích lý do thiết kế; bản thân pipeline (các giai đoạn,
cache, kiểm tra đĩa, lưu bền) nằm ở [DESIGN.vi.md](DESIGN.vi.md), cách dùng ở
[QUICKSTART.vi.md](QUICKSTART.vi.md) và các route ở [API.vi.md](API.vi.md).

### 6.1 Chuyển đổi nằm ở đâu

```mermaid
flowchart LR
  A[GGUF từ Hugging Face] --> L[Thư viện model]
  B[GGUF ở một đường dẫn trên server] --> L
  C[Trọng số nguồn: repo HF hoặc thư mục] --> V[Chuyển đổi, lượng tử hóa, kiểm tra]
  V -->|qua cổng kiểm tra| L
  L --> S[Scheduler: ước lượng, đặt, hiệu chỉnh]
```

Có ba cách đưa model vào thư viện. GGUF có sẵn được tải hoặc đăng ký như trước. Trọng số nguồn đi qua một job
chuyển đổi, sản phẩm duy nhất của nó là một file GGUF mà thư viện đăng ký như mọi file khác
(`LibraryItem.source` là `"convert"`). Từ đó scheduler không quan tâm file đến bằng đường nào: nó đọc header
GGUF thật và tự hiệu chỉnh VRAM (4.7) y như với model tải về. Đó chính là lý do để chuyển đổi tạo ra một mục
thư viện bình thường thay vì một loại model đặc biệt.

**Vì sao chuyển đổi nằm ở coordinator.** Coordinator sở hữu thư viện (`models_dir`), nên kết quả nằm đúng chỗ
sẽ được dùng và không phải chuyển file giữa các máy. Chuyển đổi và lượng tử hóa là công việc CPU, RAM và đĩa,
không cần GPU, nên bộ công cụ CPU (converter của llama.cpp, `llama-quantize`, `llama-imatrix`) là đủ, và GPU
vẫn rảnh cho suy luận. Cái giá là coordinator còn phục vụ router, nên công việc được định hình để không cản
đường nó: **mỗi lần một job**, theo thứ tự gửi, mọi công cụ chạy ở độ ưu tiên thấp. Hai job cùng lúc chỉ làm
nhau chậm đi và có thể làm đầy đĩa; một job ở độ ưu tiên thấp làm coordinator chậm đi chút ít nhưng không
bỏ đói việc định tuyến. Bộ công cụ là tùy chọn, nên coordinator không có nó vẫn chạy bình thường và chỉ từ
chối yêu cầu chuyển đổi kèm lời giải thích.

### 6.2 Chọn kiểu lượng tử hóa theo dung lượng của cụm

Kiểu lượng tử hóa quyết định kích thước file, nên quyết định model vừa một GPU, cần nhiều GPU (chia qua RPC,
chậm hơn) hay không vừa. Bắt người dùng chọn kiểu mà không có thông tin nghĩa là chuyển đổi xong mới biết không
vừa rồi chuyển lại, tốn từ vài phút đến vài giờ CPU. Vì vậy `inspect` xem nguồn **trước khi** làm gì, và báo
cho người dùng, theo từng kiểu, kích thước ước lượng và việc nó có vừa cụm hiện tại không.

- **Ước lượng kích thước.** Mỗi kiểu có một số bit trên trọng số (bpw) tính trên toàn model, lấy từ kích thước
  `llama-quantize` công bố cho Llama-3-8B. Áp thẳng thì với Qwen2.5-0.5B bị thấp 24 %, vì có hai yếu tố phụ
  thuộc vào model. Ma trận embedding và output vẫn giữ gần 8 bit ở các kiểu bit thấp; chúng chiếm khoảng 13 %
  Llama-3-8B nhưng 28 % một model nhỏ có từ vựng lớn, nên được tính riêng (khoảng 8.5 bpw) và bpw của phần
  trọng số còn lại được suy ra từ số tham chiếu. Và K-quant cần số cột của hàng chia hết cho 256; khi kích thước
  ẩn không chia hết (Qwen2.5-0.5B: 896), `llama-quantize` lùi từng tensor về một kiểu cũ, lớn hơn, nên ước
  lượng dùng bpw của kiểu lùi đó. Số đo: Qwen2.5-0.5B `Q4_K_M` ước lượng 390.7 MB so với thực tế 397.8 MB,
  SmolLM2-135M 103.1 so với 105.5 MB. Cùng ước lượng này dùng để kiểm tra đĩa (DESIGN mục 17.3), kể cả bước
  kiểm tra lúc gửi từ chối ngay một job mà đĩa rõ ràng không chứa nổi.
- **Các kiểu IQ.** Số bit trên trọng số danh nghĩa của chúng thấp hơn file thật, vì `llama-quantize` giữ ma
  trận output và các tensor nhạy nhất ở kiểu cao hơn, nên các dòng của chúng dùng kích thước nguyên file đã
  công bố cho Llama-3-8B thay vì số danh nghĩa (số danh nghĩa được nêu trong ghi chú). Với hàng không chia hết
  cho 256, chúng lùi về một kiểu 4.5 bpw, như K-quant.
- **VRAM.** Kích thước file + KV cache f16 ở ngữ cảnh 4096 + 300 MB chi phí runtime. Đây là giả định cố định,
  cố ý đơn giản cho việc ra quyết định, không phải lời hứa về `ctx_size` cuối cùng.
- **Vừa hay không.** Mỗi kiểu được đánh dấu là vừa **GPU lớn nhất** và vừa **cả pool** (tổng mọi GPU). Khác
  biệt này quan trọng: model chỉ vừa pool phải chia qua RPC, chậm hơn, nên vừa một GPU được ưu tiên và lý do
  nói rõ khi phải dùng pool.
- **Bậc thang gợi ý.** Kiểu đầu tiên vừa một GPU trong `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M` (chất lượng tốt nhất
  trước); không có thì kiểu đầu tiên vừa pool. Model dưới 3 B tham số chỉ dùng ba kiểu đầu, vì model nhỏ mất
  chất lượng nhanh nhất; nếu ngay cả ba kiểu đó cũng không vừa thì thử cả bậc thang và lý do nói chất lượng sẽ
  giảm. Các kiểu thấp hơn (Q3, Q2, IQ) được liệt kê như lựa chọn nhưng không nằm trên bậc thang: chọn chúng là
  chủ động đánh đổi chất lượng lấy bộ nhớ. Không có server GPU thì không có thông tin về độ vừa, và một quy tắc
  theo kích thước được dùng (`Q8_0` dưới 3 B, `Q5_K_M` dưới 15 B, còn lại `Q4_K_M`).

**Quan hệ với scheduler.** Ước lượng lúc chuyển đổi chỉ dẫn hướng việc chọn kiểu. Khi job xong, file là một mục
thư viện bình thường, và việc đặt dùng header thật (`ModelMeta`) cùng hệ số hiệu chỉnh theo từng model (4.7).
Hai ước lượng không cần khớp tuyệt đối, và số thật luôn thắng.

### 6.3 Ma trận tầm quan trọng (imatrix)

Lượng tử hóa làm tròn trọng số, và làm tròn mọi trọng số như nhau thì phí bit vào những trọng số hầu như không
quan trọng. **Ma trận tầm quan trọng** (imatrix) ghi lại, cho từng trọng số, nó được dùng mạnh đến đâu khi model
xử lý văn bản thật, để `llama-quantize` dồn ngân sách sai số vào chỗ ít gây hại nhất. Lợi ích tăng khi kiểu
càng nhỏ; từ khoảng 4 bit trở lên mức mất mát vốn đã nhỏ và công việc thêm không đáng làm theo mặc định.

`imatrix` có ba chế độ `auto`, `on`, `off`:

- `auto` bật khi kiểu cần ma trận, hoặc số bit trên trọng số dưới 4 (`Q3_K_S`, `IQ3_*`, `Q2_K`, `IQ2_*`,
  `IQ1_*`).
- Một số kiểu **bắt buộc** có: `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ2_M`, `IQ3_XXS`, `IQ3_XS`
  (`llama-quantize` từ chối các tensor loại này nếu không có ma trận; file `IQ2_M` và `IQ3_XS` có chứa các
  tensor đó). Đặt `off` cho chúng bị từ chối ngay lúc gửi với mã 422, thay vì thất bại sau nhiều giờ chuyển đổi.
- `on` tính ma trận cho mọi kiểu có lượng tử hóa (chậm hơn, tốt hơn ở mọi cỡ). `F16`, `BF16` và `Q8_0` do
  converter ghi thẳng, không có bước `llama-quantize` nào để đưa ma trận vào, nên không bao giờ tính ma trận.
- Quyết định được đưa ra lúc gửi và lưu thành `imatrix_used`, nên thấy được trước khi job chạy.

**Chi phí.** Thêm một lượt chạy: `llama-imatrix` chạy model 16-bit trên CPU qua N đoạn 512 token (mặc định 100,
`imatrix_chunks`) rồi ghi ma trận; đây là một trạng thái job riêng, `calibrating`, nằm giữa `converting` và
`quantizing`. Đó là một lượt forward của model chưa lượng tử hóa, nên là phần chậm nhất với model lớn và là lý
do `auto` không đơn giản là "luôn luôn". `llama-imatrix` là tùy chọn như phần còn lại của bộ công cụ
(`imatrix_available` trong `GET /api/convert/options`); không có nó, việc cần ma trận bị từ chối bằng một lỗi
503 rõ ràng, còn các chuyển đổi khác vẫn chạy.

**Văn bản hiệu chỉnh.** Ma trận phản ánh văn bản dùng để tính nó, nên văn bản là một quyết định thiết kế.
gpupool đi kèm văn bản đa ngôn ngữ **do chính mình viết** (văn xuôi, tiếng Việt, tiếng Trung/Nhật/Hàn, code,
toán, JSON, hội thoại theo định dạng chat). Tự viết, vì các bộ hiệu chỉnh công khai thông dụng có giấy phép mà
gpupool không thể tùy ý phân phối lại; đa ngôn ngữ và có code, vì ma trận tính trên văn xuôi tiếng Anh bảo vệ
tiếng Anh với cái giá là các ngôn ngữ khác và code (độ lệch của ma trận chính là độ lệch của văn bản), mà nền
tảng này không chỉ dành cho tiếng Anh. Người dùng biết lĩnh vực của model có thể đưa `calibration_path` (một
file `.txt` có đường dẫn tuyệt đối trên server, đường dẫn máy chủ được dịch như đường dẫn thư viện, tối đa
20 MB).

### 6.4 Kiểm tra như một cổng vào thư viện

Model đã chuyển đổi không được tin mù quáng: converter có code riêng cho từng kiến trúc, tokenizer được chuyển
riêng với trọng số, và tokenizer sai làm câu trả lời kém đi một cách âm thầm, không có gì sập để báo cho ai.
Vì vậy kết quả chỉ được đưa vào thư viện sau bước `validating`:

- header GGUF phải đọc được, có kiến trúc và tokenizer (nếu không thì job **thất bại**; file đó không thể
  phục vụ);
- tokenizer được so với bản Hugging Face trên các văn bản thử cố định, gồm tiếng Việt, CJK và một chuỗi emoji,
  những chỗ tokenizer đã chuyển đổi thường lệch;
- một lượt sinh văn bản ngắn chạy trên CPU làm smoke test.

So sánh hoặc smoke test thất bại thì job sang `needs_review`: file được giữ nhưng **không** vào thư viện, và một
người quyết định chấp nhận hay xóa, vì đó có thể là lỗi converter hoặc một điểm lạ chấp nhận được và chỉ người
mới phân biệt được. Kiểm tra không chạy được (thiếu RAM, hết thời gian) chỉ là cảnh báo, vì chặn vì lý do đó sẽ
khiến coordinator nhỏ không dùng được. Chi tiết ở DESIGN mục 17.4.

### 6.5 Ranh giới an toàn

- **Mặc định không chạy code của repo.** Converter nạp tokenizer với `trust_remote_code=True`, tức sẽ chạy các
  file `*.py` của repo ngay trong coordinator. Các file đó không được tải hay đưa vào staging trừ khi yêu cầu
  đặt `allow_remote_code`, mà UI đánh dấu là nguy hiểm.
- **Converter chạy offline.** Nó chạy với Hugging Face hub ở chế độ offline và chỉ thấy một **thư mục staging
  gồm các liên kết** tới các file đã chọn, nên thư mục nguồn không bao giờ bị sửa và không bị đọc ngoài phần
  đã chọn.
- **Xóa có giới hạn hẹp.** Chỉ các đường dẫn bên trong `models_dir/.convert` và `models_dir/.hf` mới bị xóa.
- **Đầu vào được kiểm tra.** Tên file đầu ra và các cờ lượng tử hóa do người dùng đưa vào đều được kiểm
  (tên, danh sách cho phép các tên kiểu ggml), nên một giá trị không bao giờ trở thành đối số dòng lệnh thừa.

## 7. Tóm tắt API

Tài liệu tham chiếu đầy đủ, kèm dạng request và response, là [API.vi.md](API.vi.md). Mọi endpoint của thiết
kế nằm dưới `/api` và dùng admin key; các trường request mới đều tùy chọn, nên client cũ vẫn chạy.

| Khả năng | Endpoint |
| --- | --- |
| Chính sách model (mục 3, 4.5) | `PUT /api/models/{name}`, `POST .../start`, `POST .../stop`, `DELETE /api/models/{name}`, `GET .../scaling`, `POST .../plan` |
| Dung lượng và gợi ý (mục 5) | `GET /api/capacity`, `POST /api/recommend` |
| Giả lập (mục 4.4) | `POST /api/simulate`: các lần start, stop và giành chỗ mà reconciler sẽ làm với thay đổi giả định |
| Cân bằng lại (mục 4.6) | `POST /api/rebalance` (`dry_run` mặc định là true) |
| Trạng thái và event | `GET /api/state` (gồm `calibration`, `rebalance`), `GET /api/events`, `POST /api/events/read` |
| Router | khởi động lạnh của `/v1/*` trả 503 + `Retry-After` quá `cold_start_timeout_s` |
| Chuyển đổi (mục 6) | `GET /api/convert/options`, `POST /api/convert/inspect`, `POST /api/convert`, `GET /api/convert[/{job_id}]`, `POST .../cancel`, `.../retry`, `.../accept`, `DELETE /api/convert/{job_id}` |

Event do thiết kế thêm: `preempted` (warning), `scaled_up`, `scaled_down`, `unloaded_idle`, `cold_start`,
`rebalance_started`, `rebalanced`, `calibrated` (info), `rebalance_failed` (warning).

Khác biệt so với bản nháp đầu: không có endpoint nhãn (`PUT /api/servers/{id}/labels`) và
`PUT .../gpus/{device_id}` chỉ nhận `enabled`; `POST /api/simulate` báo `start`, `stop`, `preempt` và
`unplaced` nhưng không có `move` (các lần di chuyển của rebalance được xem trước bằng `POST /api/rebalance`);
`not_possible` gồm `need_mb`, `largest_single_gpu_mb`, `largest_single_node_mb` và `max_ctx_that_fits`.

## 8. Thay đổi dữ liệu

- `ModelSpec`: các trường chính sách (JSON trong bảng `models`, không cần migration), về sau thêm
  `kv_cache_type`, `speculative`, `draft`, `draft_n_max`, `flash_attn`, `batch`, `ubatch`, `kv_unified`.
- `NodeReport.features`: khả năng của agent ngoài bản 0.5, theo bản build llama.cpp của nó (`rpc_multi_device`, `spec_mtp`,
  `kv_unified`); coordinator chỉ dùng khả năng agent
  báo, nên có thể nâng cấp coordinator trước các agent.
- `ModelMeta`: bố cục cache theo layer (`kv_k`, `kv_v`, `swa`, `n_swa`, `state_bytes`), block MTP (`n_nextn`,
  `nextn_bytes`) và `active_bytes` của MoE, đều tùy chọn (metadata không có chúng giữ ước lượng cũ).
- `Device`: `bandwidth_gbps` (agent đọc từ NVML), `uuid`, `budget_mb`. Agent cũ không gửi thì coi như bằng
  nhau.
- `Placement`: `score`, `est_decode_tps`, `reasons`, `draft_est_mb`.
- `gpu_flags` được khóa theo uuid của card khi có báo cáo. Không có bảng `server_labels` và không có cột
  `labels`/`reserve_mb` (xem mục 3).
- Bảng `control_state(key, value, updated_at)`: trạng thái autoscaler theo từng model, `preempted`,
  `backoff`, `move`.
- Bảng `model_calibration(model, factor, samples, updated_at)`; bị xoá một lần khi phiên bản bộ ước lượng đổi
  (khoá `control_state` là `estimator`).
- Bảng `convert_jobs`: các job chuyển đổi và trạng thái của chúng; file của job đã xong là một mục thư viện bình
  thường có `source` là `"convert"` (mục 6).
- Thời điểm request gần nhất của từng model nằm trong autoscaler (lưu với nhịp chậm); router chỉ báo request
  cho nó.

## 9. Lộ trình

| Giai đoạn | Nội dung | Rủi ro | Trạng thái |
| --- | --- | --- | --- |
| 1. Nền tảng | `priority` và thứ tự công bằng; `spread`; phạt ghép chung GPU bận; `bandwidth_gbps` + ước lượng tok/s; sinh phương án + chấm điểm; `GET /api/capacity`; `POST /api/recommend`; panel gợi ý trong form "New model" | thấp: không gỡ replica nào đang chạy | xong |
| 2. Co giãn | đọc `/metrics` của llama-server; `min/max_replicas` + autoscaler; `idle_unload_s` + khởi động lạnh ở router; `GET /scaling` | trung bình: thêm/bớt replica tự động | xong |
| 3. Giành chỗ | preemption + cooldown + claim; `POST /api/simulate` | trung bình: gỡ replica đang phục vụ, cần drain đúng | xong |
| 4. Cân bằng lại | rebalance make-before-break; `POST /api/rebalance` | cao nhất: load lại model lớn tốn thời gian | xong |
| Sau 4 | tự hiệu chỉnh VRAM (4.7); lưu bền trạng thái điều khiển (4.8) | thấp | xong |
| Tối ưu engine | một `ggml-rpc-server` mỗi server và replica; speculative decoding `mtp`; `kv_unified`; ước lượng KV theo layer (SWA, MLA, model lai, MTP) và tốc độ decode MoE; transport RDMA trong image agent | trung bình: đổi những gì chạy trên mọi replica bị chia | xong, chưa chạy trên phần cứng thật nhiều GPU |
| Chuyển đổi | Hugging Face / thư mục sang GGUF ngay tại coordinator; chọn kiểu lượng tử hóa theo dung lượng cụm; imatrix; cổng kiểm tra (mục 6) | trung bình: nặng CPU, RAM, đĩa, chạy cạnh router | xong |

**Trạng thái giai đoạn 1:** xong. Chạy thật trên GTX 1650: model `priority` 80 thắng model có tên đứng trước; replica và model mới dàn sang GPU khác (mô phỏng với scheduler thật); tok/s ước lượng 41.6 so với đo thật 51.8 (model 3B) và 204 so với 182 (0.5B). `budget_mb` giờ trừ cả VRAM do chính các replica của gpupool đang giữ. Chưa có cụm thật nhiều GPU để kiểm phần dàn replica trên phần cứng.

**Trạng thái giai đoạn 2:** xong. Chạy thật trên GTX 1650: model 3B ở chế độ theo yêu cầu tự dỡ sau 20 s không có request, request tiếp theo khởi động lạnh và nhận trả lời sau 2.9 s; model 0.5B (min 1, max 2) có request xếp hàng thì lên 2 replica sau khoảng 9 s, 82 request chia 49/33 cho hai replica, hết tải 16 s thì về 1. Lúc đo, trạng thái autoscale chỉ nằm trong bộ nhớ và coordinator khởi động lại thì mọi model quay về mức tối thiểu đang chạy; nay đã được lưu bền (4.5, 4.8).

**Trạng thái giai đoạn 3:** xong. Chạy thật trên GTX 1650 (budget 2500 MB): `/api/simulate` báo trước đúng việc sẽ gỡ `q05` (priority 20) để chạy `q3b` (priority 80), `/api/recommend` trả phương án `fits_now: false` kèm replica cần gỡ, và khi `q05` không cho giành chỗ thì báo `q3b` không đặt được. Giành chỗ thật: `q3b` chạy 23.5 s sau lệnh start, `q05` không giành lại được. Lần chạy đầu lộ ra một lỗi: replica bị gỡ chuyển sang draining nên model của nó launch lại ngay vào phần bộ nhớ đang giải phóng. Đã sửa bằng một khoảng giữ chỗ: trong lúc model giành chỗ chưa có replica (tối đa `drain_timeout_s` + 120 s), model ưu tiên thấp hơn không được launch.

**Trạng thái giai đoạn 4:** xong. Chạy thật với hai agent trên một máy (a: GPU giới hạn 1000 MB, b: chỉ CPU): replica `y` đang chạy trên b/CPU (~30 tok/s ước lượng) được chuyển sang a/CUDA0 (~204 tok/s) khi GPU rảnh, điểm +100, mất khoảng 9.5 s. Trong lúc chuyển, một client gửi liên tục 23 request và không request nào lỗi. Sau khi chuyển, kiểm tra lại không còn đề xuất nào. Chạy định kỳ theo `rebalance_s` (mặc định 600 s, 0 = tắt).

**Trạng thái chuyển đổi:** xong. Ước lượng kích thước đã được so với file thật trên hai model (6.2: lệch khoảng 2 %). So sánh tokenizer và smoke test sinh văn bản chạy với bộ công cụ thật. Imatrix và các kiểu IQ được thêm sau lần chạy đầu đó; ước lượng kích thước của chúng và mức cải thiện chất lượng chúng mang lại chưa được đo ở đây. Pipeline chưa chạy trên model đủ lớn để thử RAM và đĩa của coordinator.

Mỗi giai đoạn được kiểm trên cụm thật, ít nhất 2 server và 2 model, trước khi sang giai đoạn sau.

## 10. Câu hỏi mở và các quyết định

Đã quyết định:

- **Độ chính xác của ước lượng VRAM với model lớn chia nhiều GPU.** Không đo thủ công; thay vào đó mỗi
  model tự hiệu chỉnh theo những gì llama.cpp thực sự cấp phát (4.7). Ước lượng vẫn là điểm xuất phát, hệ số
  đo được sửa nó theo từng model.
- **Nhãn, `share_gpu`, `gpu_selector`, `reserve_mb`.** Không xây (mục 3).
- **Hành vi khi khởi động lại.** Trạng thái điều khiển được lưu bền (4.8); chỉ bộ đếm rebalance cố ý khởi
  động lại.

Vẫn còn mở:

- **η theo kiến trúc GPU.** η và `t_hop` giờ tự học từ tốc độ decode đo được (mỗi thứ một giá trị cho cả cụm,
  DESIGN mục 6.3), bắt đầu từ số của GTX 1650. Mỗi kiến trúc GPU một giá trị thì cần mẫu theo từng kiến trúc và chưa
  triển khai; GPU datacenter có thể khác GPU phổ thông.
- **Hiệu chỉnh trên model lớn.** Cơ chế đã có, nhưng chưa chạy một model ≥ 7B thật chia nhiều GPU qua nó;
  mức kẹp 0.9–2.0 và trọng số 0.5 chưa được chỉnh.
- **Prefill** bị giới hạn bởi compute (SM × clock), không phải băng thông. Model chủ yếu nhận prompt dài
  có thể cần một điểm hiệu năng riêng.
- **Trọng số chấm điểm** ở mục 4.3 là điểm khởi đầu, là hằng số trong code, cần chỉnh theo số đo thật.
- **Imatrix trên server GPU.** Ma trận được tính trên CPU của coordinator, chậm với model lớn trong khi GPU ở
  nơi khác trong cụm đang rảnh. Chạy `llama-imatrix` trên một agent có GPU sẽ nhanh hơn nhiều, đổi lại phải
  chuyển model 16-bit sang đó và theo dõi thêm một loại job.
- **Phân tán việc chuyển đổi.** Mỗi lần một job trên một máy thì đơn giản và an toàn, nhưng hàng đợi các model
  lớn phải chờ. Rải job ra nhiều máy cần lưu trữ dùng chung hoặc chuyển file, và cách giữ cho các kiểm tra đĩa
  vẫn đúng.
- **LoRA adapter.** Gộp adapter vào model gốc trước khi lượng tử hóa, hoặc phục vụ riêng, chưa được hỗ trợ; chỉ
  trọng số đầy đủ mới chuyển đổi được.
- **Bộ chiếu hình ảnh (`mmproj`).** Model đa phương thức cần một GGUF thứ hai cho phần hình ảnh. Chỉ model văn
  bản được chuyển đổi, nên model như vậy chỉ phục vụ văn bản.
- **Hiệu chỉnh theo họ model.** Một văn bản đa ngôn ngữ dựng sẵn dùng cho mọi model. Chọn hoặc sinh văn bản
  theo từng họ model, hoặc theo mục đích sử dụng (code, chat, một ngôn ngữ), có thể cho ma trận tốt hơn, và
  chưa có phép đo nào cho biết có đáng với độ phức tạp thêm vào hay không.
- **Dàn replica và rebalance trên phần cứng thật nhiều GPU, nhiều server** mới chỉ được kiểm với agent giả
  lập và một máy một GPU.
- **Đợt tối ưu engine trên phần cứng thật.** Một RPC server mỗi server, `draft-mtp`, `kv_unified`, bố cục KV theo
  layer và RDMA làm theo mã nguồn llama.cpp b11342 và có unit test, nhưng chưa được đo trên một model bị chia thật.
  Kỳ vọng: ít vòng RPC hơn mỗi token khi hai GPU cùng server; ước lượng nhỏ hơn và sát hơn cho model SWA / MLA / lai
  (hiệu chỉnh sẽ sửa phần còn lại).
- **Chi phí mỗi hop.** `t_hop` = 2 ms mỗi RPC server là con số đoán từ một máy; với RDMA hoặc LAN nhanh nó nhỏ hơn
  nhiều và nên được đo (hoặc hiệu chỉnh từ tốc độ decode đo được) trước khi dùng để xếp hạng placement.
