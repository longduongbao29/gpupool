# gpupool

> Tiếng Việt. English: [README.md](README.md)

Gom VRAM trống rải rác trên nhiều server (ví dụ 3 GB + 5 GB + 10 GB ở ba máy) thành một pool và
serve LLM qua một endpoint OpenAI-compatible duy nhất. Engine là
[llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server` + `ggml-rpc-server`, model GGUF);
gpupool là control plane bên trên: đo VRAM trống từng node, quyết định model đặt ở đâu và chia layer
thế nào, khởi chạy và giám sát engine, định tuyến request, và đặt lại replica khi một node chết.

Dành cho server dùng chung: chạy hoàn toàn ở user space (không sudo), CUDA khác nhau giữa các máy,
VRAM có người khác cùng dùng.

## Tính năng

- **Nhiều model, nhiều server, nhiều GPU.** Nhiều model chạy cùng lúc; mỗi model được đặt trên một GPU,
  một server, hoặc chia qua nhiều server bằng llama.cpp RPC.
- **Các bậc placement và chấm điểm.** Các phương án được xếp hạng theo tốc độ decode ước lượng (băng thông
  bộ nhớ, số bước nhảy mạng, việc dùng chung GPU); placement thắng được lưu kèm lý do và hiện trên UI.
- **Gợi ý, dung lượng và what-if.** `/api/recommend` xếp hạng các phương án GPU/server cho một model,
  `/api/capacity` cho biết còn vừa gì, `/api/simulate` trả lời "nếu... thì sao" mà không thay đổi gì.
- **Ưu tiên và preemption.** Model ưu tiên cao hơn có thể lấy chỗ của model ưu tiên thấp hơn.
- **Autoscaling, gồm cả scale-to-zero.** Số replica theo tải; model rảnh có thể về 0 và cold start
  khi có request kế tiếp.
- **Rebalance make-before-break.** Khi có placement tốt hơn, replica mới sẵn sàng trước rồi replica cũ mới dừng.
- **Lượng tử hoá KV cache** (`f16`, `q8_0`, `q4_0`) để model vừa ít GPU hơn.
- **Speculative decoding** (`ngram`, hoặc model `draft`) để giảm số vòng RPC khi model bị chia.
- **RPC firewall.** Mỗi agent giới hạn cổng `ggml-rpc-server` (iptables) chỉ cho head và chính nó.
- **Tự hiệu chỉnh VRAM.** Bộ đệm engine đo được sẽ hiệu chỉnh ước lượng bộ nhớ; hệ số được lưu lại.
- **Trạng thái điều khiển được giữ qua lần khởi động lại coordinator** (SQLite), gồm cả việc xử lý các lần
  launch bị crash làm dở.
- **Triển khai.** Image Docker trên GHCR (`ghcr.io/longduongbao29/gpupool-coordinator`, `ghcr.io/longduongbao29/gpupool-agent`),
  build image không cần truy cập GitHub, hỗ trợ HTTP proxy, lấy file model từ đường dẫn trên host.
- **Chuyển model Hugging Face sang GGUF.** Model chỉ được phát hành dưới dạng trọng số safetensors / PyTorch
  có thể được chuyển ngay trong coordinator (UI hoặc `/api/convert`) với loại lượng tử hoá tuỳ chọn
  (Q8_0 ... Q2_K), ước lượng dung lượng file và VRAM cho từng loại, và bước kiểm tra (header GGUF, id token
  so với Hugging Face, một lượt sinh văn bản ngắn trên CPU) trước khi vào thư viện model. Image Docker đã kèm
  sẵn bộ công cụ (`WITH_CONVERT=0` build image gọn, không có bộ công cụ).
- **Web UI** cho server, GPU, model, deployment, gợi ý và sự kiện.

## Trạng thái

Đã test trên một laptop Windows (GTX 1650 Ti, 4 GB): chạy trực tiếp với các node giả lập và llama.cpp thật,
và dưới dạng cụm 3 server mô phỏng bằng Docker trong WSL2 (`docker-compose.sim.yml`). Một bài test
end-to-end 3 server chỉ dùng CPU cũng chạy trong CI, gồm cả một lần chuyển Hugging Face sang GGUF. Chưa chạy trên LAN nhiều server thật. Xem
[docs/TEST_REPORT.vi.md](docs/TEST_REPORT.vi.md).

## Cách hoạt động

```
client ──OpenAI API──► coordinator (router + scheduler + reconciler, SQLite)
                            ▲ heartbeat             │ start/stop engine
   server A: agent ─ llama-server (head) ──RPC──► server B: agent ─ ggml-rpc-server
                                         └─RPC──► server C: agent ─ ggml-rpc-server
```

- **agent** (mỗi server một cái): báo GPU qua NVML, chạy/dừng tiến trình llama.cpp, cache file GGUF, áp dụng RPC firewall.
- **scheduler**: đọc header GGUF (không tải cả file), ước lượng bộ nhớ theo layer, sinh các placement ứng viên
  (1 GPU, 1 server, tập con các server; GPU trước RAM CPU) và chấm điểm theo tốc độ decode ước lượng.
- **router**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming, cân bằng tải theo
  prefix (system prompt giống nhau vào replica đã cache), retry trước byte đầu tiên.
- **reconciler**: giữ đủ số replica mong muốn, rollback khi launch lỗi, failover khi node chết, drain,
  autoscale, preempt và rebalance.

Thiết kế: [docs/DESIGN.vi.md](docs/DESIGN.vi.md), [docs/PLATFORM_DESIGN.vi.md](docs/PLATFORM_DESIGN.vi.md).

## Bắt đầu nhanh

Một lệnh cho coordinator, mỗi server GPU một lệnh, rồi trỏ client OpenAI bất kỳ vào.
Hướng dẫn đầy đủ: [docs/QUICKSTART.vi.md](docs/QUICKSTART.vi.md).

```bash
# 1. coordinator (máy bất kỳ): log in ra admin key; mở http://<IP máy đó>:8080
docker run -d --name gpupool -p 8080:8080 -v gpupool:/data ghcr.io/longduongbao29/gpupool-coordinator
docker logs gpupool

# 2. mỗi server GPU: chạy lệnh join hiện trên UI (Servers -> Add Server)
docker run -d --name gpupool-agent --gpus all --network host --pid host -v gpupool-agent:/data \
  -e GPUPOOL_JOIN="http://10.0.0.1:8080#<cluster-token>" ghcr.io/longduongbao29/gpupool-agent

# 3. trên UI: Models -> Add model (Hugging Face hoặc đường dẫn) -> New model -> Start, rồi:
curl http://10.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen7b", "messages": [{"role": "user", "content": "Xin chào"}]}'
```

## Test

```bash
uv run pytest                 # unit test (khoảng 768)
uv run pytest -m real         # cần llama.cpp ở .cache/llama/b11342-cuda12.4 và GGUF ở .cache/models
uv run python scripts/e2e_local.py   # 3 node giả lập trên một máy, llama.cpp thật

# cụm 3 server mô phỏng trên một máy (cần Docker + NVIDIA Container Toolkit)
docker compose -f docker-compose.sim.yml up -d

# end-to-end trong CI: coordinator + 3 agent chỉ dùng CPU (docker-compose.ci.yml), model nhỏ chia qua RPC,
# sau đó một model Hugging Face được chuyển sang GGUF trong coordinator và phục vụ
uv run python scripts/ci_e2e.py [--project gpupool-ci] [--port 8080] [--keep] [--skip-convert]
```

`scripts/ci_e2e.py` cần image `gpupool-agent` và `gpupool-coordinator` (đổi bằng
`GPUPOOL_AGENT_IMAGE` / `GPUPOOL_COORDINATOR_IMAGE`); trong GitHub Actions đó là job `e2e`. Bước chuyển đổi
cần image coordinator được build kèm bộ công cụ (mặc định) và truy cập được huggingface.co;
`--skip-convert` bỏ bước đó.

## Tài liệu

- [Bắt đầu nhanh và triển khai](docs/QUICKSTART.vi.md)
- [HTTP API](docs/API.vi.md)
- [Thiết kế](docs/DESIGN.vi.md)
- [Thiết kế nền tảng (scheduling nhiều model)](docs/PLATFORM_DESIGN.vi.md)
- [Thiết kế UI](docs/UI_DESIGN.vi.md)
- [Báo cáo test](docs/TEST_REPORT.vi.md)
