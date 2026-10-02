# gpupool — Báo cáo test (2026-10-02)

> Bản tiếng Việt. Bản tiếng Anh: [TEST_REPORT.en.md](TEST_REPORT.en.md). Hai bản phải được cập nhật cùng nhau.

Máy: 1 laptop, GTX 1650 Ti Max-Q 4 GB, RAM 16 GB, Windows 11, llama.cpp b11342 (bản CUDA 12.4).
Giả lập 3 "server" bằng 3 agent trên 127.0.0.1 / .2 / .3 (`scripts/e2e_local.py`):
a = CUDA0 thật, giới hạn 1200 MB; b, c = device CPU (1000 MB mỗi node) sau một `ggml-rpc-server` thật.

## Kết quả

| Test | Kết quả |
| --- | --- |
| Unit test (`uv run pytest`) | 90 pass |
| Test với binary thật (`uv run pytest -m real`) | 3 pass |
| Kịch bản 1: 0.5B trên 1 GPU | pass |
| Kịch bản 2: 3B chia qua 3 node bằng RPC | pass trong 1 lần chạy trọn vẹn; lần chạy lại sạch bị máy dừng vì thiếu RAM |
| Kịch bản 3: failover | **chưa chạy** (bị dừng vì thiếu RAM trước khi tới) |

## Số đo

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

## Ước lượng bộ nhớ so với thực tế

| Model | Ước lượng (cũ) | Ước lượng (đã hiệu chỉnh) | VRAM đo được |
| --- | --- | --- | --- |
| 0.5B, ctx 4096 | 722 MB (+38%) | 587 MB (+12%) | 525 MB |

Weights và KV cache khớp log load của llama.cpp tới từng MiB. Overhead cố định 300 MB cũ được thay
bằng công thức compute buffer đo được cộng 128 MB cho context.

## Lỗi phát hiện khi chạy thật (đã sửa hết, mỗi lỗi có test chống tái phát)

| Lỗi | Hậu quả | Cách sửa |
| --- | --- | --- |
| Truyền `--device` trước `--rpc` | mọi lần launch nhiều node đều thoát với lỗi usage | `--rpc` trước; test chạy câu lệnh sinh ra qua parser thật |
| Overhead cố định 300 MB / device | ước cao 38%, phí dung lượng pool | ước lượng đã hiệu chỉnh, khoá bằng test |
| Agent bị kill cứng để lại `ggml-rpc-server` | engine mồ côi giữ bộ nhớ, coordinator không biết | file pid + dọn khi agent khởi động |
| Review trước: RAM CPU tính như VRAM | layer nằm ở RAM dù node khác còn VRAM | thử chỉ GPU trước |
| Review trước: nhả VRAM của replica vừa ready quá sớm | cùng một vùng VRAM có thể bị cấp hai lần | giữ chỗ tới khi có report mới |

## Còn lại

- Chạy kịch bản 3 (failover): `uv run python scripts/e2e_local.py --only 3`. Cần khoảng 4 GB RAM trống.
- Test trên server Linux thật có GPU ở mọi node (số tốc độ ở trên không đại diện cho trường hợp đó).
