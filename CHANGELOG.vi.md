# Nhật ký thay đổi

> Tiếng Việt. English: [CHANGELOG.md](CHANGELOG.md)

Mọi thay đổi đáng chú ý của gpupool được ghi ở đây. Định dạng theo
[Keep a Changelog](https://keepachangelog.com/vi/1.1.0/) và dự án dùng
[Semantic Versioning](https://semver.org/lang/vi/) (bản 0.x: phiên bản phụ có thể thay đổi hành vi).

## [Chưa phát hành]

### Thay đổi

- Chia layer giữa các GPU theo băng thông chứ không chỉ theo bộ nhớ trống. Sau khi có cách chia vừa, layer
  được dời từ thiết bị chậm sang thiết bị nhanh khi còn vừa và tốc độ decode ước tính tăng (thời gian decode là
  tổng byte / băng thông của từng thiết bị). Ước tính ví dụ, 70B Q4 trên RTX 5090 + RTX 4090: 46/34 layer thành
  57/23, khoảng +9 % token/s. GPU cùng băng thông hoặc chưa rõ băng thông chia như cũ; mỗi thiết bị vẫn giữ ít
  nhất một layer nên số bước RPC không đổi.

## [0.5.0] - 2026-10-05

### Thêm mới

- Thiết lập tốc độ khi chạy, trong form model và `PUT /api/models`: flash attention (`flash_attn`
  auto/on/off, `-fa` của llama.cpp), micro-batch (`ubatch`, `-ub`) và batch (`batch`, `-b`). Đây là các trường
  có kiểu, được kiểm tra, gửi xuống agent, không phải cờ llama-server tùy ý (agent vẫn từ chối `extra_args`).
  Ước lượng bộ nhớ tính compute buffer theo micro-batch đã chọn (và ma trận điểm attention khi tắt flash
  attention), nên micro-batch lớn không làm GPU bị cấp quá. KV cache lượng tử hóa khi tắt flash attention bị từ
  chối (422): llama.cpp không chạy V cache lượng tử hóa nếu thiếu nó.
- Gợi ý thiết lập: "Recommend placement & settings" trong form model (`tips` trong `POST /api/recommend`) liệt
  kê các thay đổi giúp model nhanh hơn hoặc phục vụ nhiều người hơn, mỗi cái đều được kiểm trên pool thật và không
  bao giờ cần nhiều GPU hơn hiện tại, kèm nút Apply: KV cache lượng tử hóa, bản lượng tử hóa nhỏ hơn của cùng
  model hoặc context nhỏ hơn để vừa một GPU thay vì nhiều (các layer trên nhiều GPU chạy nối tiếp nên chia ra
  không bao giờ decode nhanh hơn); số slot song song giữ nguyên context mỗi request; model draft tương thích trong
  thư viện hoặc speculative n-gram; micro-batch lớn hơn cho prompt dài.
- Nhận biết thế hệ GPU: agent báo compute capability của từng GPU (`compute_cap`, hiển thị dạng
  "Ampere · cc 8.6" ở trang GPUs) và các kiến trúc CUDA mà llama.cpp của nó được build (`cuda_archs`, từ
  `cuda-archs.txt` do image agent ghi, hoặc `GPUPOOL_CUDA_ARCHS`). Gợi ý dựa theo đó: micro-batch lớn chỉ được
  gợi ý khi mọi GPU có tensor core (Volta, cc 7.0+), ép bật flash attention hoặc micro-batch lớn trên Pascal sẽ bị
  cảnh báo, và draft đề xuất ít token hơn ở đó.
- GPU mà llama.cpp của agent không có kernel (mọi kiến trúc được build đều mới hơn card) được báo với
  `kernels_ok: false` và bộ nhớ dùng được bằng 0, nên không có gì được đặt lên nó, thay vì mọi lần chạy đều lỗi
  "no kernel image is available". Trang GPUs đánh dấu "No kernels in this build".
- Form model hiển thị context mỗi request nhận được (context ÷ số slot song song).

## [0.4.2] - 2026-10-05

### Sửa lỗi

- Công tắc "In pool" ở trang Servers và GPUs không có tác dụng: phần chọn GPU riêng cho từng model trong form model
  khai báo thêm một hàm `toggleGpu` trong component UI, đè lên hàm gọi API, nên mỗi lần bấm chỉ thay đổi form model
  (đang ẩn). Các handler của form model giờ là `pinToggleServer` / `pinToggleGpu`, và có test báo lỗi nếu component
  có hai method trùng tên.

## [0.4.1] - 2026-10-04

### Thêm mới

- GPU Blackwell (RTX 5090/5080, RTX PRO 6000): image agent build kernel cho compute capability 120 (llama.cpp đổi thành
  120a để dùng FP4 tensor core). Build image giờ báo lỗi nếu thiếu bất kỳ kiến trúc nào đã yêu cầu trong backend CUDA,
  thay vì để một server lỗi "no kernel image is available" lúc chạy.

### Thay đổi

- Image agent build trên CUDA 12.8.1 (trước là 12.4.1), bản toolkit đầu tiên hỗ trợ Blackwell. Vẫn chạy trên driver
  >= 525; card RTX 50 cần driver >= 570.

### Sửa lỗi

- Dừng một model chạy chia qua RPC làm máy chính bị crash: mọi engine bị dừng cùng lúc, `ggml-rpc-server` thoát ngay,
  rồi `llama-server` không giải phóng được buffer ở máy RPC và tự abort (SIGABRT, mỗi lần một file core dump 0,1-0,5 GB,
  WSL lưu trong `%TEMP%\wsl-crashes`). Giờ máy chính được dừng trước và chờ thoát hẳn, lệnh dừng chờ lâu hơn thời gian
  ân hạn của agent, và engine chạy không sinh core dump.

## [0.4.0] - 2026-10-03

### Thêm mới

- **Chuyển đổi Hugging Face sang GGUF trong coordinator**: tải về (hoặc dùng thư mục trên máy chủ),
  `convert_hf_to_gguf.py`, `llama-quantize`, kiểm định, rồi đưa vào thư viện mô hình; dùng từ giao diện hoặc các
  route `/api/convert*`. Có bước inspect đọc cấu hình mà không tải trọng số và liệt kê mọi kiểu lượng tử hóa kèm
  dung lượng, VRAM ước tính, và việc có vừa một GPU hay cả pool; job có các giai đoạn, log, hủy/thử lại/chấp
  nhận/xóa, cache bản tải, và tự xếp hàng lại job bị ngắt sau khi khởi động lại.
- **Cổng kiểm định** trước khi tệp đã chuyển vào thư viện: header GGUF, so id tokenizer với Hugging Face trên 8 văn
  bản, và một lần sinh ngắn trên CPU. Nếu lệch, job chuyển sang `needs_review`.
- **Ma trận tầm quan trọng và 10 kiểu lượng tử hóa IQ1/IQ2/IQ3**: giai đoạn "calibrating" chạy `llama-imatrix` (chế
  độ auto/on/off). Có sẵn văn bản hiệu chỉnh đa ngôn ngữ do dự án tự viết; có thể thay bằng tệp `.txt` riêng (tối
  đa 20 MB).
- **Giới hạn server/GPU theo từng mô hình**: `pin_devices` nhận `<node>/*`; giao diện hiện cây "Mọi máy chủ và GPU |
  Chỉ những cái đã chọn".
- **Nút sao chép** trên thẻ mô hình: endpoint, tên mô hình và lệnh `curl` dựng sẵn.
- **Tường lửa RPC** (`GPUPOOL_RPC_FIREWALL=1`, cần root và `NET_ADMIN`): mỗi cổng RPC chỉ nhận node đầu, loopback và
  địa chỉ của chính agent.
- **Tự hiệu chỉnh VRAM**: dung lượng buffer đo thực của engine hiệu chỉnh ước tính bộ nhớ bằng một hệ số lưu lại theo
  từng mô hình.
- **Lưu trạng thái điều khiển**: trạng thái autoscaler, quyền chiếm và thời gian chờ của preemption, backoff và bước
  rebalance đang dở đều sống sót qua lần khởi động lại coordinator.
- **Kiểm thử end-to-end trên CI** (`scripts/ci_e2e.py`): image thật gồm coordinator và 3 agent CPU phục vụ một mô hình
  nhỏ chia qua RPC, kèm giai đoạn chuyển đổi.
- Tài liệu: chuyển đổi, ma trận tầm quan trọng, giới hạn theo mô hình, báo cáo kiểm thử (EN + VI). Tệp giấy phép và
  cộng đồng: `LICENSE` (MIT), thông báo bên thứ ba, hướng dẫn đóng góp, quy tắc ứng xử, chính sách bảo mật, mẫu issue
  và pull request.

### Thay đổi

- Repo đổi tên từ `multi-gpu-inference` thành `gpupool` (link cũ tự chuyển hướng); tên image không đổi.
- Image coordinator có thể kèm bộ công cụ chuyển đổi (`WITH_CONVERT=1`, thêm khoảng 1,2 GB); `WITH_CONVERT=0` giữ
  image gọn.
- Gửi job chuyển đổi bị từ chối với mã 507 khi ổ đĩa rõ ràng không đủ chỗ, thay vì thất bại sau vài phút.
- Ước tính dung lượng cho các kiểu lượng tử hóa tính riêng phần embedding và các trường hợp dự phòng của
  `llama-quantize` (trước đây trung bình đơn giản thấp hơn 24 % ở một mô hình nhỏ; nay lệch vài phần trăm so với tệp
  thật).

### Sửa lỗi

- Image Docker không build được sau khi `pyproject.toml` khai báo file license: Dockerfile không chép `LICENSE` và
  `THIRD_PARTY_NOTICES.md` vào bước build.
- CI bị đỏ từ khi có tính năng chuyển đổi: trình quản lý job chuyển đổi không tạo thư mục dữ liệu của coordinator
  trước khi mở cơ sở dữ liệu, nên bản checkout mới hoặc `db_path` ở thư mục mới đều lỗi. Nay thư mục được tạo, có
  test hồi quy.
- Văn bản hiệu chỉnh có sẵn bị bỏ sót khỏi một commit do quy tắc ignore `data/`; bản checkout mới sẽ từ chối mọi job
  ma trận tầm quan trọng. Thư mục dữ liệu của gói nay được loại khỏi quy tắc đó.
- Hiệu chỉnh chia hệ số hoạch định ra khỏi ước tính trước khi lấy mẫu.
- Một test agent chập chờn do tranh ghi log với engine giả nay đã biết chờ; test thuần khiết của simulate bỏ qua tải
  trực tiếp của mock.

### Bảo mật

- Chuyển đổi không bao giờ tải hay chạy tệp Python của repo trừ khi bật `allow_remote_code`, chạy bộ chuyển đổi ở
  chế độ offline trên một thư mục tạm chứa liên kết tới các tệp trong danh sách cho phép, và chỉ xóa tệp dưới
  `models_dir/.convert` và `.hf`.

## [0.3.0] - 2026-10-03

### Thêm mới

- **Nền tảng nhiều mô hình**, các giai đoạn 1 đến 4 của thiết kế nền tảng: độ ưu tiên, spread và xếp chỗ có chấm điểm
  cùng `/api/recommend` và `/api/capacity`; autoscaling với scale-to-zero và cold start; preemption theo độ ưu tiên
  cùng `/api/simulate`; rebalance kiểu make-before-break cùng `/api/rebalance`.
- Lượng tử hóa KV cache (`f16`, `q8_0`, `q4_0`) và giải mã suy đoán (`ngram` hoặc mô hình `draft`) theo từng mô hình.
- Cụm 3 máy chủ giả lập (`docker-compose.sim.yml`) để thử hệ thống trên một máy.
- Build Docker chạy được ở nơi git hoặc GitHub bị chặn (tarball llama.cpp có sẵn, mirror, tài liệu proxy và đường
  dẫn); tìm tệp mô hình theo đường dẫn trên máy chủ khi coordinator chạy trong Docker.
- Định danh GPU theo UUID; giới hạn kích thước body request của router.
- Giao diện thiết kế lại: design token, giao diện sáng, hiệu ứng chuyển động, favicon, bộ chọn tệp trên máy chủ.

### Thay đổi

- Xếp chỗ đa node xét mọi tổ hợp máy chủ khả thi chứ không chỉ cách tham lam; bộ chấm điểm chọn.
- Cả hai image ghim Python 3.12 (`.python-version` nay được sao chép; trước đó uv chọn 3.14).
- Dọn lại phần lõi coordinator: hàm dùng chung, trạng thái có kiểu, không so khớp lỗi bằng chuỗi.

### Sửa lỗi

- Giữ nguyên GPU trong các xếp chỗ đa node; không còn nhầm coordinator bị đứng với node đã chết.
- Mất GPU, tệp GGUF chia nhỏ, rò rỉ cổng và replica, và vòng lặp khởi chạy lại khi crash.
- Một chuỗi `\n` nguyên văn trong khối `ENV` của Dockerfile coordinator làm `docker build` thất bại.
- Gia cố cache của agent và API engine; router và `/report` nhẹ hơn; khởi động mô hình không còn chặn một tick.

## [0.2.0] - 2026-10-02

### Thêm mới

- Giao diện quản lý, sổ đăng ký máy chủ có thăm dò định kỳ, công tắc GPU và nhật ký sự kiện.
- Thông báo khi có sự cố.
- Cài đặt Docker một lệnh (khóa sinh sẵn, agent tự tham gia, hướng dẫn bắt đầu nhanh), cấu hình bằng biến môi
  trường và build image trên CI.
- Thư viện mô hình: tải từ Hugging Face và đường dẫn cục bộ đã đăng ký.
- Số liệu GPU cho giao diện; mất một GPU không còn che các GPU khỏe.
- Hỗ trợ proxy: lưu lượng ra ngoài đi qua `http_proxy`/`https_proxy`, lưu lượng nội bộ thì không bao giờ.

## Trước 0.2.0 (chưa gắn tag)

- Mặt phẳng điều khiển xây theo từng lớp: hợp đồng dùng chung (mô hình dữ liệu truyền, xác thực, cấu hình), agent trên
  node (dò thiết bị, giám sát engine, cache mô hình, HTTP API), bộ lập lịch (đọc header GGUF dạng luồng, ước tính bộ
  nhớ, xếp chỗ), router (cân bằng theo tiền tố, proxy tương thích OpenAI) và coordinator (kho lưu trữ, client gọi
  agent, reconciler, API quản trị).
- Ước tính bộ nhớ hiệu chỉnh theo buffer thật của llama.cpp; script end-to-end; khởi chạy đa node truyền `--rpc`
  trước `--device`; agent dọn các engine mồ côi.
- Tài liệu song ngữ Anh và Việt: README, thiết kế, báo cáo kiểm thử.

[Chưa phát hành]: https://github.com/longduongbao29/gpupool/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/longduongbao29/gpupool/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/longduongbao29/gpupool/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/longduongbao29/gpupool/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/longduongbao29/gpupool/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/longduongbao29/gpupool/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/longduongbao29/gpupool/releases/tag/v0.2.0
