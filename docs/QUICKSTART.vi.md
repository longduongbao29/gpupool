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

## Thử trên một máy (cụm 3 server giả lập)

`docker-compose.sim.yml` chạy một coordinator và ba "server", tự join đúng như cài thật (`GPUPOOL_JOIN`), để
thử UI, việc đặt model trên nhiều server và chia qua RPC chỉ với một GPU. Build hai image trước
(`docker compose -f docker-compose.coordinator.yml build` và file compose của agent), rồi:

```bash
GPUPOOL_HOST_MODELS_DIR=/srv/gguf docker compose -f docker-compose.sim.yml up -d
```

UI: `http://<máy chạy docker>:8080`, admin key `sim-admin`. server-a báo GPU thật; server-b và server-c báo một
Tesla T4 và một A100 giả lập (`GPUPOOL_FAKE_DEVICES`) nhưng chạy trên cùng GPU thật. Tổng các giới hạn
(`SIM_BUDGET_A/B/C`, mặc định 1300/1100/1100 MB) phải nhỏ hơn bộ nhớ GPU thật. Mỗi server có IP riêng trong một
mạng Docker riêng thay cho `--network host`.

## Tùy chọn hiệu năng (theo từng model, trong form deploy)

| Tùy chọn | Giá trị | Tác dụng | Đo thật (GTX 1650, Qwen2.5-3B) |
| --- | --- | --- | --- |
| KV cache | f16, q8_0, q4_0 | KV cache nhỏ hơn nên model có thể vừa ít GPU hơn | ctx 8192: −132 / −204 MB, tốc độ gần như không đổi (51.9 / 51.2 / 50.8 tok/s) |
| Speculative | none, ngram, draft | model lớn chạy ít lượt hơn cho mỗi token, tức ít vòng RPC hơn khi bị chia | chia qua 2 server: none 48.9, ngram 53.6, draft 0.5B 53.9 tok/s |

N-gram không tốn thêm bộ nhớ nhưng chỉ có lợi khi câu trả lời lặp lại phần văn bản trước đó. Model draft phải
dùng chung tokenizer với model chính (được kiểm khi lưu) và chạy trên GPU của head (bộ nhớ đã được tính vào kế
hoạch). Mặc định đoán 4 token; 8 token chậm hơn trong các lần đo.

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

Những việc bạn vẫn phải tự làm trên mỗi máy chạy Docker:

1. **Proxy của Docker daemon**, để kéo base image (`nvidia/cuda`, `python`, image uv). Biến proxy của shell và
   build arg không được dùng cho việc này vì chính daemon kéo image. Tạo
   `/etc/systemd/system/docker.service.d/http-proxy.conf`:

   ```ini
   [Service]
   Environment="HTTP_PROXY=http://proxy.corp.local:3128"
   Environment="HTTPS_PROXY=http://proxy.corp.local:3128"
   Environment="NO_PROXY=localhost,127.0.0.1,.corp.local,10.0.0.0/8"
   ```

   rồi `sudo systemctl daemon-reload && sudo systemctl restart docker`. Kiểm tra bằng
   `systemctl show --property=Environment docker`.
2. **Tuỳ chọn: `~/.docker/config.json`.** Khi đó Docker client tự chèn proxy vào mọi lần build (dưới dạng build
   arg) và mọi container (dưới dạng biến môi trường), không cần `-e` hay `--build-arg`:

   ```json
   { "proxies": { "default": {
       "httpProxy": "http://proxy.corp.local:3128",
       "httpsProxy": "http://proxy.corp.local:3128",
       "noProxy": "localhost,127.0.0.1,.corp.local,10.0.0.0/8" } } }
   ```
3. **`no_proxy` phải liệt kê coordinator và mọi server GPU** (IP, hostname, hậu tố `.domain` hoặc CIDR như
   `10.0.0.0/8`). Bản thân gpupool không bao giờ đưa lưu lượng cluster qua proxy, nhưng proxy đặt cho các
   công cụ khác trong cùng container hoặc trên máy (curl, apt...) thì có. Luôn thêm `localhost,127.0.0.1`.
4. `git clone` (khi build agent) tuân theo `http_proxy` / `https_proxy`. Nếu proxy chặn hẳn GitHub, xem
   [Build không cần GitHub](#build-không-cần-github).

| Hiện tượng | Nguyên nhân | Cách sửa |
| --- | --- | --- |
| `docker pull` / bước `FROM`: `i/o timeout`, `TLS handshake timeout`, `connection refused` | daemon chưa có proxy | bước 1 (proxy của daemon), restart docker |
| Bước build `git clone` treo hoặc `Failed to connect to github.com` | GitHub bị chặn kể cả qua proxy | đặt `vendor/llama.cpp-b11342.tar.gz` vào repo, hoặc đặt `LLAMA_CPP_URL` |
| `uv sync`: `Failed to download ... cpython-3.12` | python-build-standalone đặt trên GitHub | đặt `UV_PYTHON_INSTALL_MIRROR` (xem bên dưới) |
| `failed to resolve source metadata for ghcr.io/astral-sh/uv` | không vào được ghcr.io | đặt `UV_IMAGE` là bản mirror; coordinator: `UV_FROM_PYPI=1` |
| `apt-get` hoặc `pip` lỗi trong lúc build | thiếu build arg proxy | truyền `--build-arg http_proxy=... https_proxy=...` hoặc dùng compose / `config.json` |
| `407 Proxy Authentication Required` | proxy đòi tài khoản | dùng `http://user:pass@host:port`; mã hoá URL các ký tự đặc biệt trong mật khẩu |
| Tìm hoặc tải Hugging Face trên UI lỗi, các thứ khác vẫn chạy | container coordinator không có biến proxy | đặt `http_proxy` / `https_proxy` trong `.env` (compose) hoặc `-e` (docker run) rồi tạo lại container |
| Server offline, request model lỗi, chỉ khi bật proxy | lưu lượng cluster bị đẩy qua proxy | thêm IP hoặc CIDR của coordinator và các server vào `no_proxy` |
| Proxy dùng được với `curl` nhưng container thì không | khác biệt biến viết thường / viết hoa | đặt cả hai cách viết (file compose đã truyền cả hai) |

## Build không cần GitHub

Trên server không ra được GitHub (hoặc git), có thể build image từ các file bạn chép vào. Nếu proxy cho
phép đi qua GitHub thì không cần phần này.

1. **Mã nguồn llama.cpp** (image agent). Trên máy ra được GitHub, tải bản nén của tag vào `vendor/` của repo
   rồi chép repo sang server:

   ```bash
   curl -L -o vendor/llama.cpp-b11342.tar.gz https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11342.tar.gz
   ```

   Build dùng file này trước, rồi `git clone`, rồi `curl` tới `LLAMA_CPP_URL` (mirror nội bộ của cùng file nén).
   Khi không có `.git`, số build được truyền thẳng cho CMake nên `llama_version` vẫn đúng.
2. **Python 3.12** (image agent). Base image CUDA Ubuntu không có Python 3.12 nên uv tải một bản từ
   python-build-standalone. Đặt file nén vào `vendor/python/<release>/` (cấu trúc và tên file chính xác:
   [vendor/README.md](../vendor/README.md)) rồi build với `UV_PYTHON_INSTALL_MIRROR=file:///vendor/python`, hoặc
   trỏ tới mirror HTTP nội bộ.
3. **Image uv** (cả hai image). `UV_IMAGE=registry.corp.local/astral-sh/uv:0.10` cho registry mirror. Image
   coordinator có thể dùng `UV_FROM_PYPI=1`: uv được cài bằng pip, và PyPI thường vào được qua proxy.
4. Tuỳ chọn: `LLAMA_USE_PREBUILT_UI=OFF` bỏ qua việc llama.cpp tải web UI của nó (gpupool không dùng).

Với docker compose, đặt các biến vào `.env` (xem `.env.example`) rồi:

```bash
docker compose -f docker-compose.agent.yml up -d --build            # agent
docker compose -f docker-compose.coordinator.yml up -d --build      # coordinator
```

Hoặc với docker thuần:

```bash
docker build -f docker/agent.Dockerfile -t gpupool-agent \
  --build-arg UV_PYTHON_INSTALL_MIRROR=file:///vendor/python \
  --build-arg UV_IMAGE=registry.corp.local/astral-sh/uv:0.10 .
docker build -f docker/coordinator.Dockerfile -t gpupool-coordinator --build-arg UV_FROM_PYPI=1 .
```

Các file đặt trong `vendor/` cần BuildKit (mặc định từ Docker 23; bản cũ hơn: `DOCKER_BUILDKIT=1`). Đừng commit
chúng. Các gói Python vẫn lấy từ PyPI (qua proxy, hoặc `UV_INDEX_URL` / `PIP_INDEX_URL` cho index nội bộ).

## Dùng file model có sẵn trên server

Đừng chép model vào container: hãy mount thư mục chứa nó. Chỉ máy coordinator cần các file này: coordinator
phục vụ model cho các server GPU nên không phải chép gì sang chúng.

1. Trong `.env` trên máy coordinator đặt thư mục đó, dạng **đường dẫn tuyệt đối** với dấu gạch chéo xuôi:

   ```
   GPUPOOL_HOST_MODELS_DIR=/srv/gguf
   ```

   `docker-compose.coordinator.yml` mount nó chỉ đọc tại `/models` và báo cho coordinator biết ánh xạ
   (`GPUPOOL_PATH_MAP`). Với docker thuần: `-v /srv/gguf:/models:ro -e GPUPOOL_PATH_MAP='{"/srv/gguf": "/models"}'`.
2. Trên UI (**Models → Add model → Path**) duyệt `/models` (coordinator liệt kê các file `.gguf` ở đó), hoặc gõ
   đường dẫn. Cả `/srv/gguf/qwen.gguf` (đường dẫn trên máy chủ) lẫn `/models/qwen.gguf` đều được: đường dẫn máy
   chủ không tồn tại trong container sẽ được đổi thành `/models/...`.
3. **Symlink phải trỏ vào bên trong thư mục đã mount.** Link có đích nằm ngoài thư mục mount sẽ bị hỏng trong
   container. Cache của Hugging Face có đúng cấu trúc này (`snapshots/<rev>/file.gguf` là link tới
   `../../blobs/<hash>`): hãy mount cả thư mục `hub/models--<org>--<repo>` (hoặc `hub`), không chỉ `snapshots`,
   hoặc chép file bằng `cp -L` / `cp --dereference`.
4. Không cần có file trên các server GPU.

Nhiều thư mục: thêm các mount `-v`, mở rộng `GPUPOOL_MODEL_ROOTS` (phân tách bằng dấu phẩy) và
`GPUPOOL_PATH_MAP` (mỗi thư mục một mục JSON).

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
