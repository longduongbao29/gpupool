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
mạng Docker riêng thay cho `--network host`. Firewall RPC được bật ở cả ba (`cap_add: [NET_ADMIN]`), nên các
lần chia model qua nhiều server cũng thử luôn nó. Dừng và xoá sạch bằng
`docker compose -f docker-compose.sim.yml down -v`. Biến ghi đè: `SIM_ADMIN_KEY`, `SIM_CLUSTER_TOKEN`,
`GPUPOOL_AGENT_IMAGE`, `GPUPOOL_COORDINATOR_IMAGE`.

## Test end-to-end cho CI (không cần GPU)

`scripts/ci_e2e.py` chạy image Docker thật với `docker-compose.ci.yml`: một coordinator và ba server chỉ dùng CPU
(`GPUPOOL_INCLUDE_CPU`, bật firewall RPC). Mỗi server cung cấp một thiết bị CPU với giới hạn `CI_BUDGET_MB`
(mặc định 120 MB), nhỏ hơn model test, nên SmolLM2-135M (Q8_0, khoảng 140 MB) buộc phải chia qua ít nhất hai
server bằng RPC của llama.cpp. Cần Docker, Python 3 (chỉ thư viện chuẩn) và internet cho lần tải model đầu tiên
(kiểm bằng SHA-256, rồi được cache). Build image trước (hoặc đặt `GPUPOOL_AGENT_IMAGE` /
`GPUPOOL_COORDINATOR_IMAGE`), rồi:

```bash
python scripts/ci_e2e.py                       # tuỳ chọn: --project gpupool-ci --port 8080 --keep
```

Nó kiểm tra: ba agent tự join, thư viện model, các API, start đến *running*, một chat completion, `/metrics`, và
stop (không còn engine nào trên mọi agent). Mã thoát 0 nghĩa là qua hết; khi lỗi nó in `docker compose logs` và
thoát với mã 1. Cụm được xoá sau khi chạy, trừ khi có `--keep`. Model được cache trong `GPUPOOL_CI_MODELS_DIR`
(mặc định `<repo>/.cache/ci-models`).

## Tùy chọn hiệu năng (theo từng model, trong form deploy)

| Tùy chọn | Giá trị | Tác dụng | Đo thật (GTX 1650, Qwen2.5-3B) |
| --- | --- | --- | --- |
| KV cache | f16, q8_0, q4_0 | KV cache nhỏ hơn nên model có thể vừa ít GPU hơn | ctx 8192: −132 / −204 MB, tốc độ gần như không đổi (51.9 / 51.2 / 50.8 tok/s) |
| Speculative | none, ngram, draft | model lớn chạy ít lượt hơn cho mỗi token, tức ít vòng RPC hơn khi bị chia | chia qua 2 server: none 48.9, ngram 53.6, draft 0.5B 53.9 tok/s |

Trong API các tuỳ chọn này là `kv_cache_type` (`f16`, `q8_0`, `q4_0`), `speculative` (`none`, `ngram`, `draft`),
`draft_file` (một model trong thư viện, cho `draft`) và `draft_n_max` (1 đến 16, mặc định 4); xem
[API.vi.md](API.vi.md). Chúng có hiệu lực ở lần chạy model kế tiếp.

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

## Bảo mật

- **Cổng RPC không có xác thực.** `ggml-rpc-server` (cổng 9000–9999 trên mỗi server GPU) nhận mọi kết nối: ai kết
  nối được đều có thể cấp phát VRAM và đọc/ghi tensor. Có hai cách giới hạn:
  - đặt `GPUPOOL_RPC_FIREWALL=1` trên server GPU. Với mỗi engine RPC, agent thêm luật iptables (và ip6tables) vào
    một chain riêng: cho phép loopback, server đầu (head) của replica và địa chỉ của chính agent trên cổng đó, và
    **chặn mọi thứ còn lại**. Luật được gỡ khi engine dừng, và luật còn sót do agent bị sập sẽ được xoá khi khởi
    động. Cần quyền root và iptables (đã có trong image); trong Docker thêm `--cap-add NET_ADMIN` (compose:
    `cap_add: [NET_ADMIN]`, đã có sẵn trong các file compose của agent, sim và CI). Nếu thiếu, agent ghi một dòng
    lỗi và chạy không có bảo vệ (xem Xử lý sự cố);
  - hoặc dùng tường lửa của bạn: chỉ cho phép 9000–9999 giữa các máy trong cụm.
- Khi khởi động, agent cảnh báo nếu firewall đang tắt.
- Các key: **admin key** (`GPUPOOL_ADMIN_KEY`) bảo vệ UI và API quản trị; **cluster token**
  (`GPUPOOL_CLUSTER_TOKEN`) xác thực server với coordinator và nằm trong lệnh join; **API key**
  (`GPUPOOL_API_KEYS`) là thứ client của `/v1` gửi. Admin key và cluster token được sinh ở lần chạy đầu và lưu
  trong volume dữ liệu (`secrets.json`); tự đặt để ghi đè.
- Không có `GPUPOOL_API_KEYS` thì `/v1` mở cho bất kỳ ai tới được cổng 8080: hãy đặt.
- Cluster token và API key đi qua HTTP thường (không TLS): hãy để cụm trong mạng riêng hoặc VPN, và đặt proxy
  chấm dứt TLS trước coordinator nếu client ở ngoài.

## Tự hiệu chỉnh VRAM

Ước lượng bộ nhớ của bộ lập kế hoạch được hiệu chỉnh theo thực tế. Khi một replica đã chạy, coordinator so
sánh bộ nhớ engine thật sự dùng với ước lượng và giữ một hệ số làm mượt cho mỗi model (đo được / ước lượng).
Lần đặt model đó kế tiếp dùng hệ số này, nên model cần nhiều hay ít hơn ước lượng sẽ không còn bị xếp quá tải hay
lãng phí GPU. Bạn thấy nó trong `/api/state`: mỗi model có `calibration`, là `{"factor": ..., "samples": ...}`
hoặc `null` cho tới lần đo đầu tiên; sự kiện `calibrated` được ghi khi hệ số thay đổi đáng kể. Không có gì cần
cấu hình. Xoá model thì hệ số của nó cũng bị quên.

## Những gì còn lại sau khi coordinator khởi động lại

Mọi thứ trong volume dữ liệu (`/data`): file, secret, cơ sở dữ liệu. Engine chạy trên các server nên vẫn chạy
trong lúc coordinator khởi động lại. Trạng thái điều khiển cũng được lưu và nạp lại khi khởi động:

- trạng thái autoscaler của từng model: số replica mong muốn, thời điểm request gần nhất và quyết định gần nhất
  (model theo yêu cầu đã dỡ vẫn ở trạng thái dỡ, model đã scale lên vẫn ở mức đã scale); chỉ các bộ đếm giờ
  "đã lên/xuống từ lúc nào" được đặt lại, việc đó chỉ làm chậm một bước scale;
- quyền chiếm chỗ (preemption) và thời gian chờ, bộ đếm backoff khi crash lặp, và lần di chuyển (rebalance) đang
  dở;
- không được giữ: bộ đếm giờ rebalance định kỳ, nó chạy lại từ lúc khởi động vì các báo cáo đầu tiên đã cũ.

Một lần launch bị ngắt bởi kill cứng (`kill -9`, hết bộ nhớ, mất điện) không thể tiếp tục: ở lần khởi động sau,
replica đó được đánh dấu *failed* ("coordinator restarted during launch"), các engine mới chạy dở được dừng trên
các server, và nhịp reconcile kế tiếp lập kế hoạch lại cho model. Tắt êm cũng tự làm đúng như vậy. Các lượt tải
Hugging Face bị ngắt bởi khởi động lại sẽ được tiếp tục.

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

## Thiết lập (biến môi trường)

Mọi thiết lập đều đặt được bằng `GPUPOOL_<TÊN>` (tên trường viết hoa), trong file TOML (`--config`) hoặc, với
một số mục, bằng cờ CLI. Thứ tự ưu tiên: cờ CLI > biến môi trường > file TOML > mặc định. Danh sách phân cách
bằng dấu phẩy, dict (`GPUPOOL_PATH_MAP`, `GPUPOOL_BUDGET_MB`) là JSON, boolean nhận `1/0/true/false`, dải cổng
viết `9000-9999`. Mọi mục dưới đây đều tuỳ chọn, trừ `GPUPOOL_JOIN` trên agent.

### Coordinator (22 thiết lập)

| Biến | Mặc định | Tác dụng |
| --- | --- | --- |
| `GPUPOOL_HOST` | `0.0.0.0` | địa chỉ coordinator lắng nghe |
| `GPUPOOL_PORT` | `8080` | cổng lắng nghe |
| `GPUPOOL_DB_PATH` | `.gpupool/coordinator.db` (image: `/data/coordinator.db`) | cơ sở dữ liệu SQLite; `secrets.json` nằm cạnh nó |
| `GPUPOOL_ADMIN_KEY` | tự sinh | key cho UI và API quản trị |
| `GPUPOOL_CLUSTER_TOKEN` | tự sinh | token để server tham gia cụm |
| `GPUPOOL_API_KEYS` | rỗng (mở) | các key client phải gửi tới `/v1`, phân cách bằng dấu phẩy |
| `GPUPOOL_MODELS_DIR` | `.gpupool/models` (image: `/data/models`) | thư viện model, phục vụ cho các server tại `/files/<tên>` |
| `GPUPOOL_PATH_MAP` | `{}` | JSON `{"<thư mục trên máy chủ>": "<thư mục trong container>"}` để người dùng gõ đường dẫn máy chủ |
| `GPUPOOL_MODEL_ROOTS` | rỗng (image: `/models`) | các thư mục UI được duyệt tìm `.gguf`, phân cách bằng dấu phẩy (rỗng = chỉ thư mục models) |
| `GPUPOOL_HEARTBEAT_TIMEOUT_S` | `10` | số giây không có báo cáo thì server bị coi là chết |
| `GPUPOOL_RECONCILE_S` | `2` | chu kỳ đối chiếu trạng thái mong muốn và thực tế |
| `GPUPOOL_MAX_REQUEST_MB` | `32` | kích thước tối đa của một request `/v1`, tính bằng MB (lớn hơn nhận HTTP 413) |
| `GPUPOOL_COLD_START_TIMEOUT_S` | `120` | thời gian một request chờ model đã dỡ (theo yêu cầu) load lại, quá thì trả HTTP 503 |
| `GPUPOOL_REBALANCE_S` | `600` | chu kỳ di chuyển replica sang vị trí tốt hơn rõ rệt (mỗi lần một replica, chạy bản mới trước); `0` = chỉ khi gọi tay (`POST /api/rebalance`) |
| `GPUPOOL_LAUNCH_TIMEOUT_S` | `600` | replica chưa sẵn sàng sau thời gian này bị đánh dấu failed |
| `GPUPOOL_LOW_FREE_MB` | `256` | thiết bị đang chạy engine mà bộ nhớ trống thấp hơn mức này thì replica bị di chuyển |
| `GPUPOOL_DRAIN_TIMEOUT_S` | `60` | thời gian replica đang dừng được cho để xử lý nốt request |
| `GPUPOOL_PORT_RANGE` | `9000-9999` | các cổng cấp cho engine trên server (cần mở giữa các server) |
| `GPUPOOL_POLL_S` | `2` | chu kỳ coordinator lấy `/report` của từng server |
| `HF_TOKEN` (hoặc `GPUPOOL_HF_TOKEN`) | rỗng | cho repo Hugging Face bị giới hạn hoặc private |
| `GPUPOOL_PUBLIC_URL` | tự dò | địa chỉ server dùng để gọi coordinator, nếu tự dò sai (dùng trong lệnh join) |
| `GPUPOOL_WEBHOOK_URL` | rỗng | webhook Slack/Discord/JSON chung để nhận cảnh báo (server chết, mất GPU, thiếu VRAM) |

Image Docker đặt sẵn `GPUPOOL_HOST`, `GPUPOOL_PORT`, `GPUPOOL_DB_PATH`, `GPUPOOL_MODELS_DIR` và
`GPUPOOL_MODEL_ROOTS` như trên. `docker-compose.coordinator.yml` còn đọc `GPUPOOL_HOST_MODELS_DIR` (thư mục trên
máy chủ được mount vào `/models`) và các biến proxy.

### Agent, mỗi server GPU một agent (17 thiết lập)

| Biến | Mặc định | Tác dụng |
| --- | --- | --- |
| `GPUPOOL_JOIN` | không có | `http://<coordinator>:8080#<cluster-token>`: thay cho hai biến kế tiếp (`--join`) |
| `GPUPOOL_COORDINATOR_URL` | `http://127.0.0.1:8080` | địa chỉ coordinator (thắng `GPUPOOL_JOIN`) |
| `GPUPOOL_CLUSTER_TOKEN` | rỗng | cluster token (thắng `GPUPOOL_JOIN`) |
| `GPUPOOL_AUTO_JOIN` | `true` | tự đăng ký với coordinator (`--no-auto-join` tắt đi; khi đó thêm server trên UI) |
| `GPUPOOL_NODE_ID` | hostname | tên server trong pool |
| `GPUPOOL_HOST` | tự dò | địa chỉ các server khác dùng để gọi server này, cũng là địa chỉ engine bind; mặc định là IP cục bộ dùng để tới coordinator |
| `GPUPOOL_PORT` | `7070` | cổng API của agent |
| `GPUPOOL_LLAMA_DIR` | bắt buộc (image: `/opt/llama`) | thư mục chứa `llama-server` và `rpc-server` (`--llama-dir`) |
| `GPUPOOL_CACHE_DIR` | `.gpupool/cache` (image: `/data/cache`) | các file GGUF đã tải |
| `GPUPOOL_LOG_DIR` | `.gpupool/logs` (image: `/data/logs`) | mỗi engine một file log |
| `GPUPOOL_MARGIN_PCT` | `0.10` | phần bộ nhớ mỗi GPU luôn để trống cho người khác |
| `GPUPOOL_MARGIN_MIN_MB` | `512` | ...nhưng không bao giờ ít hơn mức này |
| `GPUPOOL_BUDGET_MB` | `{}` | JSON giới hạn theo thiết bị, ví dụ `{"CUDA0": 1200, "CPU": 2000}`; cũng là cách cho thiết bị CPU một dung lượng |
| `GPUPOOL_INCLUDE_CPU` | `false` | cung cấp thêm thiết bị `CPU` (chạy qua `rpc-server -d CPU`) |
| `GPUPOOL_HEARTBEAT_S` | `2` | chu kỳ heartbeat (chỉ dùng khi bật push heartbeat) |
| `GPUPOOL_PUSH_HEARTBEAT` | `false` | chủ động đẩy heartbeat, cho coordinator cũ chưa có chế độ pull; thường để tắt |
| `GPUPOOL_RPC_FIREWALL` | `false` | `1` = giới hạn từng cổng RPC bằng iptables, xem [Bảo mật](#bảo-mật); cần root và `NET_ADMIN` |

Các biến chương trình có đọc nhưng không nằm trong danh sách thiết lập trên: `GPUPOOL_LOG` (mức log, mặc định
`INFO`, cho cả hai chương trình), `GPUPOOL_FAKE_DEVICES` (agent: danh sách JSON các GPU giả lập, dùng bởi
`docker-compose.sim.yml`), và `GPUPOOL_URL` (mặc định `http://127.0.0.1:8080`) cùng `GPUPOOL_ADMIN_KEY` cho các
lệnh CLI `gpupool register | plan | scale | undeploy | status`. `docker-compose.agent.yml` đọc `GPUPOOL_JOIN` (bắt
buộc), `GPUPOOL_NODE_ID`, `GPUPOOL_HOST`, `GPUPOOL_MARGIN_PCT`, `GPUPOOL_RPC_FIREWALL`, `CUDA_ARCHS` và các build
arg ở trên. API HTTP: [API.vi.md](API.vi.md). Thiết kế bên trong: [DESIGN.vi.md](DESIGN.vi.md).

## Xử lý sự cố

| Hiện tượng | Nguyên nhân / cách sửa |
| --- | --- |
| Server không bao giờ hiện trên UI | `docker logs gpupool-agent`: "connection refused" → port 8080 bị chặn hoặc sai địa chỉ trong lệnh join (dùng IP LAN của coordinator); "wrong cluster token" → copy lại lệnh join |
| Log agent báo server đã bị xoá | server đã bị xoá trên UI; thêm lại ở đó (Servers → Add Server → agent URL `http://<ip-server>:7070`) |
| Model kẹt ở *failed*: "not enough VRAM" | giải phóng GPU, bật thêm GPU, thêm server, hoặc dùng bản quantize nhỏ hơn |
| `could not select device driver "" with capabilities: [[gpu]]` | server đó chưa cài NVIDIA Container Toolkit |
| Log agent: `RPC firewall unavailable (...); RPC ports are NOT restricted` | đã đặt `GPUPOOL_RPC_FIREWALL=1` nhưng thiếu iptables hoặc container không có root/`NET_ADMIN`: thêm `--cap-add NET_ADMIN` (compose `cap_add: [NET_ADMIN]`) rồi tạo lại container, hoặc bỏ biến đó và tự đặt tường lửa cho 9000–9999 |
| Log agent: `RPC firewall: ... rule for port N failed ... port left unrestricted` | một lệnh iptables lỗi và luật đã được hoàn tác; đọc nội dung lỗi (thường là thiếu cùng quyền đó). `ip6tables unavailable` chỉ có nghĩa là IPv6 chưa được giới hạn |
| Model nhiều server kẹt ở *starting* khi bật firewall, engine trên server khác không bao giờ trả lời | peer bị từ chối. `RPC firewall: cannot resolve peer ...` nghĩa là một tên không phân giải được trên server đó: dùng IP (`GPUPOOL_HOST`) hoặc sửa DNS. Để xác nhận, bỏ `GPUPOOL_RPC_FIREWALL` trên server đó rồi tạo lại container |
| Model đang *launching* thì coordinator bị kill | được đánh dấu *failed* ("coordinator restarted during launch") và tự lập kế hoạch lại trong một nhịp reconcile |
| Windows / WSL2: mọi container khởi động lại khoảng một phút sau khi đóng terminal cuối cùng | WSL tắt VM khoảng 1 phút sau phiên `wsl.exe` cuối cùng, kể cả khi Docker đang chạy. Trong `%UserProfile%\.wslconfig` thêm `vmIdleTimeout=-1` dưới `[wsl2]`, rồi chạy `wsl --shutdown` và bật lại Docker |
| Windows / WSL2: model lớn load lỗi hoặc engine bị kill (hết bộ nhớ) | WSL2 mặc định giới hạn bộ nhớ (ở đây là 4 GB). Tăng `memory=` dưới `[wsl2]` trong `%UserProfile%\.wslconfig` (ví dụ `memory=16GB`), rồi `wsl --shutdown` |
