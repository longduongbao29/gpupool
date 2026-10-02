# gpupool

> Tiếng Việt. English: [README.md](README.md)

Gom VRAM trống rải rác trên nhiều server (ví dụ 3 GB + 5 GB + 10 GB ở ba máy) thành một pool và
serve LLM qua một endpoint OpenAI-compatible duy nhất. Engine là
[llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server` + `ggml-rpc-server`, model GGUF);
gpupool là control plane bên trên: đo VRAM trống từng node, quyết định model đặt ở đâu và chia layer
thế nào, khởi chạy và giám sát engine, định tuyến request, và đặt lại replica khi một node chết.

Dành cho server dùng chung: chạy hoàn toàn ở user space (không sudo), CUDA khác nhau giữa các máy,
VRAM có người khác cùng dùng.

## Trạng thái

Giai đoạn đầu, đã test trên một laptop Windows giả lập ba node với llama.cpp thật. Xem
[docs/TEST_REPORT.vi.md](docs/TEST_REPORT.vi.md). Chưa chạy trên cụm nhiều server thật.

## Cách hoạt động

```
client ──OpenAI API──► coordinator (router + scheduler + reconciler, SQLite)
                            ▲ heartbeat             │ start/stop engine
   server A: agent ─ llama-server (head) ──RPC──► server B: agent ─ ggml-rpc-server
                                         └─RPC──► server C: agent ─ ggml-rpc-server
```

- **agent** (mỗi server một cái): báo GPU qua NVML, chạy/dừng tiến trình llama.cpp, cache file GGUF.
- **scheduler**: đọc header GGUF (không tải cả file), ước lượng bộ nhớ theo layer, ưu tiên 1 GPU, rồi
  1 server, rồi ít server nhất; GPU trước RAM CPU.
- **router**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming, cân bằng tải theo
  prefix (system prompt giống nhau vào replica đã cache), retry trước byte đầu tiên.
- **reconciler**: giữ đủ số replica mong muốn, rollback khi launch lỗi, failover khi node chết, drain.

Thiết kế: [docs/DESIGN.vi.md](docs/DESIGN.vi.md).

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
uv run pytest                 # unit test
uv run pytest -m real         # cần llama.cpp ở .cache/llama/b11342-cuda12.4 và GGUF ở .cache/models
uv run python scripts/e2e_local.py   # 3 node giả lập trên một máy, llama.cpp thật
```
