# gpupool — Báo cáo test

> Bản tiếng Việt. Bản tiếng Anh: [TEST_REPORT.en.md](TEST_REPORT.en.md). Hai bản phải được cập nhật cùng nhau.

## 2026-10-05 (tối ưu engine: cấu trúc RPC, MTP, bố cục KV)

Thay đổi: một `ggml-rpc-server` mỗi server và replica, speculative decoding `draft-mtp`, `kv_unified`, ước lượng KV
theo layer (SWA, MLA, model lai, block MTP nạp khi cần), tốc độ decode MoE, xoá hiệu chỉnh khi bộ ước lượng đổi, khoá
định tuyến ổn định cho hội thoại nhiều lượt, bỏ giới hạn pool của router, chuyển tiếp nguyên body, tải model song
song khi khởi động nguội, cache RPC trên `/data` có giới hạn dung lượng, RDMA trong image agent (pull request #5;
thiết kế ở [DESIGN.vi.md](DESIGN.vi.md)).

### Kết quả

| Kiểm tra | Cách làm | Kết quả |
| --- | --- | --- |
| Unit test và API test | `uv run pytest -q` | 885 đạt, 5 bỏ qua |
| Cờ so với llama.cpp | mọi cờ, quy tắc bố cục KV và hành vi RPC đều đọc từ mã nguồn b11342 (`common/arg.cpp`, `llama-hparams.cpp`, `llama-kv-cache*.cpp`, `src/models/*.cpp`, `ggml-rpc.cpp`, `transport.cpp`) | có `-kvu`, `--spec-type draft-mtp`, `-d CUDA0,CUDA1`; RDMA quay về TCP; thiếu file cache chỉ khiến gửi lại tensor |
| Bố cục KV | GGUF tổng hợp cho từng họ (gemma3 SWA, deepseek2 MLA, qwen35 lai + MTP, nemotron_h, MoE) | kích thước khớp công thức của llama.cpp, kiến trúc chưa biết giữ quy tắc cũ |
| UI | Chromium với `scripts/ui_mock_server.py` | công tắc dùng chung context và lựa chọn MTP hiển thị đúng, không có lỗi trang |
| Image | các job CI `agent` / `coordinator` / `e2e` | chạy trên pull request |

### Còn phải làm

- Chạy một model bị chia trên GPU thật: tốc độ decode khi một RPC server cho hai GPU cùng server so với mỗi GPU một
  server; tỉ lệ chấp nhận và tốc độ của `draft-mtp` trên GGUF Qwen3.5 hoặc GLM; buffer đo được của một model SWA và
  một model lai so với ước lượng mới.
- RDMA trên phần cứng InfiniBand / RoCE thật.

## 2026-10-03 (chuyển đổi, đợt 2: importance matrix, loại IQ, kiểm tra đĩa sớm)

Tính năng: các loại lượng tử hoá IQ1/IQ2/IQ3 với importance matrix (giai đoạn *calibrating*), kiểm tra đĩa lúc gửi,
`failed_stage` / `imatrix_used` trên job, server và GPU được phép theo từng model (`"<node>/*"`), nút sao chép trên
thẻ model (commit 002d42e, kèm f3bf76b; thiết kế ở [DESIGN.vi.md](DESIGN.vi.md#178-importance-matrix-và-giai-đoạn-hiệu-chỉnh)).
Máy: cùng laptop Windows 11 (chuyển đổi chỉ dùng CPU), Docker trong WSL2, llama.cpp b11342, image coordinator build
kèm bộ công cụ gồm cả `llama-imatrix`.

### Kết quả

| Test | Kết quả |
| --- | --- |
| Unit test (`uv run pytest`) | 809 pass |
| End-to-end trong CI (`scripts/ci_e2e.py`) | qua 33/33 phép kiểm tra trong 249 giây |
| E2E: `HuggingFaceTB/SmolLM2-135M-Instruct` sang `Q4_K_M` | 71 giây, 105,453,984 byte; được phục vụ chia qua RPC, trả lời một chat |
| E2E: nguồn là **thư mục** sang `Q8_0` | 25 giây, 144,810,912 byte; không có importance matrix (`imatrix_used` false), file nằm trong thư viện, thư mục nguồn không bị đổi |
| E2E: Hugging Face sang `IQ2_XS` **có importance matrix** (4 chunk) | 76 giây, 84,573,088 byte; các giai đoạn *converting*, *calibrating*, *quantizing*, *validating*; `imatrix_used` true, header GGUF hợp lệ, file nằm trong thư viện |

### Chuyển đổi thật (đo được, mỗi lần một job, CPU laptop, Docker trong WSL)

| Model, loại | Tải | Convert | Calibrate | Quantize | Validate | Tổng | Đầu ra | RSS đỉnh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct, `Q4_K_M` | 107 s | 12 s | không | 4 s | 20 s | 143 s | 397,807,488 B | 1045 MiB |
| Qwen2.5-1.5B-Instruct, `IQ3_M` (100 chunk) | 314 s | 55 s | 1175 s | 123 s | 21 s | 1687 s (28 phút) | 776,663,904 B | 1697 MiB |

Đỉnh cgroup, gồm cả page cache, là 3,0 GB và 3,4 GB. Calibrate chiếm khoảng 70 % job thứ hai: trên CPU, importance
matrix chính là cái giá của các loại IQ, nên UI nói rõ điều đó và có tuỳ chọn *Calibration chunks* (ít hơn = nhanh hơn).

### Ước lượng dung lượng cho các loại mới

Ước lượng hiển thị trong hộp thoại so với file thật: `IQ2_XS` +6 %, `IQ3_M` -4 %, `Q8_0` -1 %. Các dòng IQ dùng
số bit trên trọng số tính cả file (lớp đầu ra và các tensor nhạy nhất giữ độ chính xác cao hơn), không phải số danh
nghĩa của định dạng.

### Lỗi phát hiện khi review và chạy thật (đã sửa hết)

| Lỗi | Hậu quả | Cách sửa |
| --- | --- | --- |
| `.gitignore` có quy tắc `data/` (dành cho thư mục dữ liệu lúc chạy), quy tắc này cũng khớp `src/gpupool/converter/data/` | văn bản hiệu chỉnh có sẵn và README của nó lặng lẽ bị bỏ khỏi commit; một bản checkout mới hay image CI sẽ không có văn bản hiệu chỉnh và mọi job importance matrix không có `calibration_path` sẽ bị từ chối với 503. Test cục bộ vẫn qua vì file có trên đĩa | thư mục dữ liệu của package được loại khỏi quy tắc (commit f3bf76b) |
| Form model ghi "Auto placement / choose GPUs yourself" | nghe như đặt thủ công, trong khi pin luôn là một *tập được phép* mà scheduler chọn bên trong | nay là "All servers and GPUs / Only selected ones" với cây server và GPU; `"<node>/*"` cho phép cả server kể cả GPU thêm sau; tab Servers không đổi |
| Trên thẻ model chỉ sao chép được endpoint | người dùng phải tự gõ tên model và lệnh curl | thẻ sao chép riêng endpoint, tên model và một lệnh `curl` dựng sẵn |

Chưa được các lần chạy này bao phủ: model vài GB trở lên (lớn nhất là 1,5 tỷ tham số), importance matrix trên GPU
(hiệu chỉnh luôn chạy trên CPU), `calibration_path` tuỳ chỉnh từ đầu đến cuối (chỉ có unit test), đường `needs_review`
từ một model thật, và LAN nhiều server thật.

## 2026-10-03 (chuyển Hugging Face sang GGUF)

Tính năng: coordinator chuyển model Hugging Face sang GGUF với loại lượng tử hoá tuỳ chọn
(commit 6766ba1, thiết kế ở [DESIGN.vi.md](DESIGN.vi.md#17-chuyển-hugging-face-sang-gguf-converter)). Máy: cùng
laptop Windows 11, Docker trong WSL2, llama.cpp b11342, image coordinator được build kèm bộ công cụ chuyển đổi.

### Kết quả

| Test | Kết quả |
| --- | --- |
| Unit test (`uv run pytest`) | 768 pass |
| End-to-end trong CI (`scripts/ci_e2e.py`) có giai đoạn chuyển đổi | 21/21 kiểm tra pass. `HuggingFaceTB/SmolLM2-135M-Instruct` được chuyển sang `Q4_K_M` (105,453,984 byte) trong 64 s gồm cả tải về, serve chia qua RPC, và trả lời một chat |
| `Qwen/Qwen2.5-0.5B-Instruct` sang `Q4_K_M` | 397,807,488 byte. Header GGUF ổn, có chat template, id token trùng Hugging Face ở 8/8 trường hợp (tiếng Anh, tiếng Việt, code, số, chuỗi emoji ZWJ, khoảng trắng, xuống dòng, CJK), sinh văn bản "The capital of France is Paris..." |
| Thư mục sau các job | `.convert` và `.hf` trong thư mục model trống |
| Dung lượng image coordinator | 1.77 GB có bộ công cụ (`WITH_CONVERT=1`), 560 MB không có (`WITH_CONVERT=0`) |

### Ước lượng dung lượng

Ước lượng hiển thị trước khi chuyển đổi từng thấp hơn file thật 24 % trên Qwen2.5-0.5B, vì nó dùng một giá trị
bit-trên-trọng-số trung bình cho mỗi loại. Tính riêng các ma trận embedding và phương án dự phòng của llama-quantize
cho hàng không chia hết cho 256 (xem DESIGN mục 17.5) đã đưa sai lệch về khoảng 2 %:

| Model, loại | Ước lượng | Thật |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct, `Q4_K_M` | 390.7 MB | 397.8 MB |
| SmolLM2-135M-Instruct, `Q4_K_M` | 103.1 MB | 105.5 MB |

### Lỗi phát hiện khi review và chạy thật (đã sửa hết)

| Lỗi | Hậu quả | Cách sửa |
| --- | --- | --- |
| Repo Mistral kèm các bản `consolidated.*` của cùng trọng số | lượng tải bị gấp đôi | bỏ qua các file `consolidated.*` |
| Tên file trong danh sách repo không được lọc | một tên độc hại như `../x` có thể trỏ ra ngoài thư mục tải | tên không an toàn (tuyệt đối, `..`, dấu gạch ngược, ký tự ổ đĩa) bị loại khi chọn file và được kiểm tra lại trước khi dùng |
| Ước lượng dung lượng dùng một bpw trung bình cho mỗi loại | thấp hơn 24 % trên Qwen2.5-0.5B, nên hộp thoại và bước kiểm tra đĩa báo thiếu | embedding và phương án dự phòng K-quant được tính riêng; sai lệch khoảng 2 % |

Chưa được các lần chạy này bao phủ: model vài GB trở lên (chỉ chuyển model 135M và 0.5B tham số), repo gated có
token, nguồn AWQ / đã lượng tử hoá sẵn, `allow_remote_code`, và một job `needs_review` từ model thật (các đường này
chỉ có unit test).

## 2026-10-03

Máy: cùng laptop Windows 11 (GTX 1650 Ti Max-Q, 4 GB), Docker trong WSL2. llama.cpp b11342 thật xuyên suốt.

### Kết quả

| Test | Kết quả |
| --- | --- |
| Unit test (`uv run pytest`) | 609 pass, 5 bị loại (deselected) |
| End-to-end trong CI (`scripts/ci_e2e.py`, `docker-compose.ci.yml`) | 14/14 kiểm tra pass khi chạy local, khoảng 21 s |
| GitHub Actions run 37082433728 trên master | mọi job đều pass, gồm cả `e2e` (3m24s) |
| Cụm 3 server mô phỏng (`docker-compose.sim.yml`) | chia RPC khi bật firewall, hiệu chỉnh VRAM được lưu lại, phục hồi sau crash: xem bên dưới |

Test end-to-end trong CI chạy một coordinator và 3 agent chỉ dùng CPU (budget 120 MB mỗi agent, nên
model không vừa một server), serve SmolLM2-135M-Instruct Q8_0 chia qua RPC với RPC firewall bật, và chạy
trong job `e2e` của GitHub Actions.

### Cụm 3 server mô phỏng

Cấu hình: coordinator cùng server-a (GTX 1650 thật như NVML báo, budget 1300 MB), server-b (Tesla T4
16 GB mô phỏng, budget 1100 MB) và server-c (A100 40 GB mô phỏng, budget 1100 MB). Cả ba dùng chung một
GPU thật; mọi agent đều bật RPC firewall.

- **RPC với firewall:** Qwen2.5-3B Q4 chia với head ở server-a và RPC ở server-c; chat hoạt động qua RPC.
  Trên server-c, iptables cho phép 127.0.0.1, head (172.30.0.11) và chính nó (172.30.0.13), sau đó DROP;
  server-b bị chặn khỏi cổng RPC. Chỉ head giữ file GGUF 2.1 GB; server-c chỉ giữ cache tensor RPC khoảng 724 MB.
- **Tự hiệu chỉnh VRAM:** bộ đệm đo được cao hơn ước lượng 1.5 %, nên hệ số thành 1.015 và vẫn còn hiệu lực
  sau khi khởi động lại coordinator.
- **Phục hồi sau crash:** `docker kill` coordinator giữa lúc launch. Replica "launching" mồ côi bị đánh
  fail ("coordinator restarted during launch") trong khoảng 5 s và replica mới sẵn sàng trong khoảng 30 s,
  không còn engine sót lại.
- **Tốc độ (Qwen2.5-3B, GTX 1650, cũng có trong hướng dẫn nhanh):** KV cache ở ctx 8192 tiết kiệm 132 MB
  (`q8_0`) và 204 MB (`q4_0`), decode 51.9 / 51.2 / 50.8 tok/s (f16 / q8_0 / q4_0). Chia qua 2 server:
  không speculative 48.9, ngram 53.6, draft 0.5B 53.9 tok/s.

### Lỗi phát hiện khi chạy thật (đã sửa hết, mỗi lỗi có test chống tái phát)

Đợt này:

| Lỗi | Hậu quả | Cách sửa |
| --- | --- | --- |
| RPC firewall chặn chính probe kiểm tra sẵn sàng của agent (probe kết nối từ địa chỉ bind của agent) | engine kẹt ở "launching" 600 s | cho phép địa chỉ của chính agent |
| Launch mồ côi sau khi coordinator bị kill cứng | replica kẹt "launching" mãi và chặn launch lại | các launch bị gián đoạn bị đánh fail khi khởi động |
| Test simulate-purity chập chờn do so sánh busy ratio sống của mock | test fail ngẫu nhiên | test bỏ qua tải sống của mock |

Các đợt trước:

| Lỗi | Cách sửa |
| --- | --- |
| Giữ chỗ cho draft model trừ vào mọi GPU thay vì chỉ head | chỉ giữ chỗ trên head |
| Chỉ dùng `-md` thì draft được load nhưng không bao giờ được dùng ở b11342 | đặt `--spec-type` tường minh |
| Placement nhiều node chỉ đưa ra một ứng viên | liệt kê các tập con server, giới hạn 12 |
| Image agent dùng Python 3.14 vì `.python-version` không được copy | copy file, ghim Python 3.12 |
| `budget_mb` không tính replica của chính model | tính cả replica của chính model |

### Ghi chú môi trường

WSL tự tắt khi rảnh làm các container khởi động lại (gpupool tự phục hồi). Cách sửa: `vmIdleTimeout=-1` trong `.wslconfig`.

### Còn lại

- Chạy trên LAN nhiều server thật.
- Đo speculative decoding qua mạng thật.
- Model từ 7B trở lên cần GPU lớn hơn laptop test.
- API key và quota theo từng client, `/v1/embeddings`, TLS.
- Thay model không downtime.
- Nâng cấp llama.cpp.

## 2026-10-02 (lịch sử)

Máy: 1 laptop, GTX 1650 Ti Max-Q 4 GB, RAM 16 GB, Windows 11, llama.cpp b11342 (bản CUDA 12.4).
Giả lập 3 "server" bằng 3 agent trên 127.0.0.1 / .2 / .3 (`scripts/e2e_local.py`):
a = CUDA0 thật, giới hạn 1200 MB; b, c = device CPU (1000 MB mỗi node) sau một `ggml-rpc-server` thật.

### Kết quả

| Test | Kết quả |
| --- | --- |
| Unit test (`uv run pytest`) | 90 pass |
| Test với binary thật (`uv run pytest -m real`) | 3 pass |
| Kịch bản 1: 0.5B trên 1 GPU | pass |
| Kịch bản 2: 3B chia qua 3 node bằng RPC | pass trong 1 lần chạy trọn vẹn; lần chạy lại sạch bị máy dừng vì thiếu RAM |
| Kịch bản 3: failover | **chưa chạy** (bị dừng vì thiếu RAM trước khi tới) |

### Số đo

| Cấu hình | Prefill | Decode | TTFT qua router | Thời gian load |
| --- | --- | --- | --- | --- |
| 0.5B, llama-bench, 1 GPU (baseline) | 1292 t/s (pp512) | 182 t/s | — | — |
| 0.5B qua gpupool, 1 GPU | 177 t/s (prompt ngắn) | 185 t/s (170 đo ở client) | 76 ms | 5.2 s |
| 3B, llama-bench, 1 GPU (baseline) | 250 t/s (pp512) | 51.4 t/s | — | — |
| 3B qua gpupool, chia GPU + 2 node CPU qua RPC | 13 t/s | 11.6 t/s | 7.7 s | 16.2 s |

- Router không làm chậm decode đáng kể (185 so với baseline 182 t/s).
- Bản chia 3B chậm hơn 1 GPU 4.4 lần vì 23/36 layer chạy trên device CPU trong môi trường giả lập.
  Trên server thật các device từ xa là GPU; lần chạy này kiểm chứng tính đúng của đường RPC, không
  phải tốc độ. Câu trả lời đúng ("Paris").
- Placement của 3B: a/CUDA0 13 layer (842 MB), b/CPU 12 layer (683 MB), c/CPU 11 layer + output (899 MB).

### Ước lượng bộ nhớ so với thực tế

| Model | Ước lượng (cũ) | Ước lượng (đã hiệu chỉnh) | VRAM đo được |
| --- | --- | --- | --- |
| 0.5B, ctx 4096 | 722 MB (+38%) | 587 MB (+12%) | 525 MB |

Weights và KV cache khớp log load của llama.cpp tới từng MiB. Overhead cố định 300 MB cũ được thay
bằng công thức compute buffer đo được cộng 128 MB cho context.

### Lỗi phát hiện khi chạy thật (đã sửa hết, mỗi lỗi có test chống tái phát)

| Lỗi | Hậu quả | Cách sửa |
| --- | --- | --- |
| Truyền `--device` trước `--rpc` | mọi lần launch nhiều node đều thoát với lỗi usage | `--rpc` trước; test chạy câu lệnh sinh ra qua parser thật |
| Overhead cố định 300 MB / device | ước cao 38%, phí dung lượng pool | ước lượng đã hiệu chỉnh, khoá bằng test |
| Agent bị kill cứng để lại `ggml-rpc-server` | engine mồ côi giữ bộ nhớ, coordinator không biết | file pid + dọn khi agent khởi động |
| Review trước: RAM CPU tính như VRAM | layer nằm ở RAM dù node khác còn VRAM | thử chỉ GPU trước |
| Review trước: nhả VRAM của replica vừa ready quá sớm | cùng một vùng VRAM có thể bị cấp hai lần | giữ chỗ tới khi có report mới |

### Còn lại

- Chạy kịch bản 3 (failover): `uv run python scripts/e2e_local.py --only 3`. Cần khoảng 4 GB RAM trống.
- Test trên server Linux thật có GPU ở mọi node (số tốc độ ở trên không đại diện cho trường hợp đó).
