# gpupool — Báo cáo test

> Bản tiếng Việt. Bản tiếng Anh: [TEST_REPORT.en.md](TEST_REPORT.en.md). Hai bản phải được cập nhật cùng nhau.

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
