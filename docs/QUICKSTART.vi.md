# gpupool — Bắt đầu nhanh

> Bản tiếng Việt. Bản tiếng Anh: [QUICKSTART.en.md](QUICKSTART.en.md). Hai bản phải được cập nhật cùng nhau.

Ba bước, mỗi bước một lệnh. Không phải cấu hình tay: key được tự sinh, server tự lấy tên và IP, rồi
tự vào pool.

```
1. máy coordinator   docker run ... gpupool-coordinator         (một lần)
2. mỗi server GPU    docker run ... gpupool-agent (lệnh join)    (mỗi server một lần)
3. ứng dụng của bạn  base_url = http://<coordinator>:8080/v1     (client OpenAI bất kỳ)
```

## 1. Chạy coordinator (một máy, không cần GPU)

```bash
docker run -d --name gpupool --restart unless-stopped -p 8080:8080 -v gpupool:/data \
  ghcr.io/longduongbao29/gpupool-coordinator
docker logs gpupool
```

Log in ra **admin key**. Mở `http://<IP của máy này>:8080` trên trình duyệt và đăng nhập bằng key đó.
Key được sinh ở lần chạy đầu và lưu trong volume `gpupool`, nên khởi động lại vẫn giữ nguyên.

Log cũng in ra lệnh join, nhưng trong Docker nó hiện IP nội bộ của container (`172.17.x.x`). Hãy lấy
lệnh join từ UI: UI dùng đúng địa chỉ bạn đang mở.

## 2. Thêm từng server GPU (mỗi server một lệnh)

Trên UI mở **Servers → Add Server**, copy lệnh join rồi chạy trên server GPU. Lệnh có dạng:

```bash
docker run -d --name gpupool-agent --restart unless-stopped --gpus all --network host --pid host \
  -v gpupool-agent:/data \
  -e GPUPOOL_JOIN="http://10.0.0.1:8080#<cluster-token>" \
  ghcr.io/longduongbao29/gpupool-agent
```

Vài giây sau server hiện trong UI cùng toàn bộ GPU của nó. Tên server là hostname, IP được tự dò.

Yêu cầu trên server GPU: driver NVIDIA ≥ 525, Docker và
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Kiểm tra bằng:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 3. Serve một model (trên UI)

1. **Models → Add model**: nhập repo Hugging Face (ví dụ `Qwen/Qwen2.5-7B-Instruct-GGUF`), chọn file
   `.gguf`, bấm **Download**. Hoặc tab **Path** cho file đã có sẵn trên máy coordinator.
2. **New model**: đặt tên (đây là giá trị `model` mà client sẽ dùng), chọn file. GPU: để **Auto**, hoặc
   tick các GPU muốn dùng.
3. **Start**. Trạng thái chuyển *starting → running*. **Stop** để trả GPU.

Nếu model lớn hơn mọi GPU đơn lẻ, nó được tự chia qua nhiều GPU và nhiều server.

## 4. Kết nối ứng dụng

Client nào tương thích OpenAI cũng dùng được. Chỉ cần đổi hai giá trị:

| Thiết lập | Giá trị |
| --- | --- |
| Base URL | `http://<coordinator>:8080/v1` |
| Model | tên bạn đặt cho model trên UI |
| API key | mặc định không cần; nếu đặt `GPUPOOL_API_KEYS` thì dùng một trong các key đó |

curl:

```bash
curl http://10.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen7b", "messages": [{"role": "user", "content": "Xin chào"}]}'
```

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://10.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="qwen7b", messages=[{"role": "user", "content": "Xin chào"}])
print(r.choices[0].message.content)
```

Với Open WebUI, LangChain, LlamaIndex hay Continue: chọn "OpenAI-compatible" và nhập cùng base URL.
Hỗ trợ streaming (`stream: true`). Trang Settings trên UI hiện sẵn các đoạn code này với địa chỉ thật
của bạn.

Muốn bắt client gửi key: chạy coordinator với `-e GPUPOOL_API_KEYS=key1,key2`.

## Không dùng Docker

Coordinator:

```bash
git clone https://github.com/longduongbao29/multi-gpu-inference && cd multi-gpu-inference
uv sync && uv run gpupool coordinator          # in ra cùng admin key và lệnh join
```

Server GPU (cần [uv](https://docs.astral.sh/uv/) và bản build llama.cpp có CUDA và RPC, b11342):

```bash
uv run gpupool agent --join "http://10.0.0.1:8080#<cluster-token>" --llama-dir /path/to/llama.cpp/build/bin
```

## Mạng

| Port | Mở giữa | Dùng cho |
| --- | --- | --- |
| 8080 | client và server GPU → coordinator | UI, API, tải model |
| 7070 | coordinator → server GPU | bật và tắt engine |
| 9000–9999 | coordinator và server GPU → server GPU | llama.cpp (HTTP và RPC giữa các server) |

RPC của llama.cpp không mã hoá: giữ các server GPU trong mạng nội bộ tin cậy.

## Sau HTTP proxy

Nếu các server ra internet qua proxy, hãy truyền biến proxy của máy chủ vào container
(ưu tiên đọc `http_proxy`, `https_proxy`, `no_proxy` viết thường; viết hoa cũng được):

```bash
docker run -d ... -e http_proxy -e https_proxy -e no_proxy ghcr.io/longduongbao29/gpupool-agent
```

Với docker compose, các biến này tự được lấy từ môi trường máy chủ hoặc `.env`.

- Đi qua proxy: Hugging Face (liệt kê và tải file), tải model từ URL `https://` trên các server, và webhook cảnh báo.
- Không bao giờ đi qua proxy: lưu lượng giữa coordinator, các server và llama.cpp (heartbeat, điều khiển engine,
  file model từ coordinator, request tới model của bạn).
- `no_proxy` được tôn trọng: danh sách phân tách bằng dấu phẩy gồm host, hậu tố miền (`.corp.local`), IP, CIDR (`10.0.0.0/8`) hoặc `*`.

Để build image cục bộ khi đứng sau proxy:

```bash
docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy   --build-arg no_proxy=$no_proxy -f docker/agent.Dockerfile -t gpupool-agent .
```

(`docker compose ... --build` tự truyền các biến này.) Giá trị proxy không được lưu trong image.

## Thiết lập tuỳ chọn (biến môi trường của coordinator)

| Biến | Mặc định | Tác dụng |
| --- | --- | --- |
| `GPUPOOL_API_KEYS` | rỗng (mở) | các key client phải gửi, phân cách bằng dấu phẩy |
| `GPUPOOL_MAX_REQUEST_MB` | 32 | kích thước tối đa của một request gửi tới /v1, tính bằng MB (lớn hơn sẽ nhận HTTP 413) |
| `GPUPOOL_COLD_START_TIMEOUT_S` | 120 | thời gian một request chờ model đã dỡ (chế độ theo yêu cầu) load lại, quá thì trả HTTP 503 |
| `GPUPOOL_REBALANCE_S` | 600 | chu kỳ di chuyển replica sang vị trí tốt hơn rõ rệt (mỗi lần một replica, chạy bản mới trước); 0 = chỉ khi gọi tay |
| `GPUPOOL_PUBLIC_URL` | tự dò | địa chỉ server dùng để gọi coordinator, nếu tự dò sai |
| `GPUPOOL_WEBHOOK_URL` | rỗng | webhook Slack/Discord để nhận cảnh báo (server chết, mất GPU, thiếu VRAM) |
| `HF_TOKEN` | rỗng | cho repo Hugging Face bị giới hạn hoặc private |
| `GPUPOOL_ADMIN_KEY`, `GPUPOOL_CLUSTER_TOKEN` | tự sinh | tự đặt thay cho key tự sinh |

Trên server GPU: `GPUPOOL_MARGIN_PCT` (mặc định `0.10`) là phần bộ nhớ mỗi GPU luôn để trống cho
người khác.

## Xử lý sự cố

| Hiện tượng | Nguyên nhân / cách sửa |
| --- | --- |
| Server không bao giờ hiện trên UI | `docker logs gpupool-agent`: "connection refused" → port 8080 bị chặn hoặc sai địa chỉ trong lệnh join (dùng IP LAN của coordinator); "wrong cluster token" → copy lại lệnh join |
| Log agent báo server đã bị xoá | server đã bị xoá trên UI; thêm lại ở đó (Servers → Add Server → agent URL `http://<ip-server>:7070`) |
| Model kẹt ở *failed*: "not enough VRAM" | giải phóng GPU, bật thêm GPU, thêm server, hoặc dùng bản quantize nhỏ hơn |
| `could not select device driver "" with capabilities: [[gpu]]` | server đó chưa cài NVIDIA Container Toolkit |
