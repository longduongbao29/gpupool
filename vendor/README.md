# vendor/ - offline build inputs

> English version. Vietnamese version: below. Keep both in sync.

For servers where git / GitHub is blocked. Everything here is optional: an empty folder (just this
file and `.gitkeep`) means the build downloads as usual. **Do not commit the archives** (they are tens
of MB): this folder is only a drop-off for the build context. `.dockerignore` does not exclude it, and
the Dockerfiles bind-mount it read-only (BuildKit, the default since Docker 23), so nothing from here
ends up in an image layer.

## 1. llama.cpp source (agent image and coordinator image)

Download the GitHub tag archive for the pinned tag (`LLAMA_CPP_REF`, default `b11413`) on any machine
that can reach GitHub, and save it under exactly this name:

```bash
curl -L -o vendor/llama.cpp-b11413.tar.gz \
  https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11413.tar.gz     # ~37 MB
```

It must be the `archive/refs/tags` tarball: one top-level directory (`llama.cpp-b11413/`), which the build
strips. The build picks the source in this order: this tarball, then `git clone`, then `curl` of
`LLAMA_CPP_URL` (build arg; point it at an internal mirror of the same archive).

llama.cpp's CMake derives its build number from git, which a tarball does not have, so the Dockerfile
passes `-DLLAMA_BUILD_NUMBER=11342` (the tag without `b`) and `-DLLAMA_BUILD_COMMIT=<short sha>` (read
from the commit id GitHub stores in the tarball header). `llama-server --version` then reports
`build 11342`, which the agent shows as `llama_version`.

The coordinator image uses the same tarball when it is built with `WITH_CONVERT=1` (the default): it
compiles `llama-quantize`, `llama-tokenize` and `llama-simple` (CPU only) and ships `convert_hf_to_gguf.py`
for the Hugging Face -> GGUF conversion. Same lookup order (tarball, `git clone`, `LLAMA_CPP_URL`). The
conversion also installs CPU-only PyTorch from `https://download.pytorch.org/whl/cpu`; behind a mirror pass
`--build-arg TORCH_INDEX_URL=https://mirror.corp/pytorch/whl/cpu` (PyPI packages follow the usual
`PIP_INDEX_URL` / `UV_INDEX_URL` / proxy settings). `--build-arg WITH_CONVERT=0` builds the lean coordinator
without any of this.

During the build llama.cpp also tries to download its web UI from Hugging Face (gpupool does not use
it). If that fails it only prints a warning; `--build-arg LLAMA_USE_PREBUILT_UI=OFF` skips the attempt.

## 2. Python 3.12 for the agent image (managed CPython)

The CUDA Ubuntu 22.04 base image has no Python 3.12, so `uv sync` downloads one
(python-build-standalone). To avoid GitHub, give uv a mirror directory laid out as
`<mirror>/<release>/<archive>`, with the release tag and file name that uv requests:

```
vendor/python/20260303/cpython-3.12.13+20260303-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
```

(`+` literally in the file name.) The exact version is whatever the uv in the image wants. Find out on any
machine with the same uv version (`0.10.x`):

```bash
uv python list --only-downloads --all-platforms --output-format json
```

and take the `url` of the entry with `key` `cpython-3.12.<N>-linux-x86_64-gnu`: the last two path
segments are `<release>/<archive>` (the `%2B` in the URL is the `+` in the file name). Download that URL
into `vendor/python/<release>/`. Then build with:

```bash
docker build --build-arg UV_PYTHON_INSTALL_MIRROR=file:///vendor/python -f docker/agent.Dockerfile -t gpupool-agent .
```

An internal HTTP mirror with the same layout works too (`UV_PYTHON_INSTALL_MIRROR=https://mirror.corp/pbs`).
If the file is missing, uv fails with the URL it tried: use that to correct the layout.

## 3. uv binary

Not stored here. Both Dockerfiles take `--build-arg UV_IMAGE=<registry mirror>/astral-sh/uv:0.10`. The
coordinator image can alternatively use `--build-arg UV_FROM_PYPI=1` (installs uv with pip).

---

# vendor/ - đầu vào để build offline

Dành cho server chặn git / GitHub. Mọi thứ ở đây đều tuỳ chọn: thư mục trống (chỉ có file này và
`.gitkeep`) nghĩa là build tải như bình thường. **Không commit các file nén** (vài chục MB): thư mục này chỉ
để đặt file vào build context. `.dockerignore` không loại trừ nó và Dockerfile bind-mount nó ở chế độ chỉ
đọc (BuildKit, mặc định từ Docker 23), nên không có gì ở đây lọt vào layer của image.

## 1. Mã nguồn llama.cpp (image agent và image coordinator)

Trên một máy ra được GitHub, tải bản nén của tag đã ghim (`LLAMA_CPP_REF`, mặc định `b11413`) và lưu đúng tên:

```bash
curl -L -o vendor/llama.cpp-b11413.tar.gz \
  https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11413.tar.gz     # ~37 MB
```

Phải là tarball `archive/refs/tags`: một thư mục gốc (`llama.cpp-b11413/`), build sẽ bỏ nó đi. Thứ tự chọn
nguồn: tarball này, rồi `git clone`, rồi `curl` tới `LLAMA_CPP_URL` (build arg; trỏ tới bản mirror nội bộ
của cùng file nén).

CMake của llama.cpp lấy số build từ git, mà tarball không có git, nên Dockerfile truyền
`-DLLAMA_BUILD_NUMBER=11342` (tên tag bỏ chữ `b`) và `-DLLAMA_BUILD_COMMIT=<sha ngắn>` (đọc từ commit id
GitHub ghi trong header của tarball). `llama-server --version` sẽ báo `build 11342`, agent hiển thị đó là
`llama_version`.

Image coordinator dùng chính tarball này khi build với `WITH_CONVERT=1` (mặc định): nó biên dịch
`llama-quantize`, `llama-tokenize`, `llama-simple` (chỉ CPU) và kèm `convert_hf_to_gguf.py` để chuyển đổi
Hugging Face -> GGUF. Thứ tự tìm nguồn giống nhau (tarball, `git clone`, `LLAMA_CPP_URL`). Việc chuyển đổi còn
cài PyTorch bản CPU từ `https://download.pytorch.org/whl/cpu`; sau mirror thì truyền
`--build-arg TORCH_INDEX_URL=https://mirror.corp/pytorch/whl/cpu` (gói PyPI theo cấu hình `PIP_INDEX_URL` /
`UV_INDEX_URL` / proxy như thường lệ). `--build-arg WITH_CONVERT=0` build coordinator gọn nhẹ, không có phần này.

Khi build, llama.cpp còn thử tải web UI của nó từ Hugging Face (gpupool không dùng). Tải lỗi chỉ in cảnh báo;
`--build-arg LLAMA_USE_PREBUILT_UI=OFF` bỏ qua bước thử này.

## 2. Python 3.12 cho image agent (CPython do uv quản lý)

Base image CUDA Ubuntu 22.04 không có Python 3.12 nên `uv sync` tải một bản (python-build-standalone). Để
tránh GitHub, đưa cho uv một thư mục mirror theo cấu trúc `<mirror>/<release>/<file nén>`, đúng tên release và
tên file mà uv yêu cầu:

```
vendor/python/20260303/cpython-3.12.13+20260303-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
```

(dấu `+` ghi nguyên trong tên file). Phiên bản chính xác là bản mà uv trong image cần. Xem trên máy bất kỳ có
cùng uv (`0.10.x`):

```bash
uv python list --only-downloads --all-platforms --output-format json
```

lấy `url` của mục có `key` là `cpython-3.12.<N>-linux-x86_64-gnu`: hai đoạn cuối của đường dẫn là
`<release>/<file>` (`%2B` trong URL chính là dấu `+` trong tên file). Tải URL đó vào `vendor/python/<release>/`.
Rồi build:

```bash
docker build --build-arg UV_PYTHON_INSTALL_MIRROR=file:///vendor/python -f docker/agent.Dockerfile -t gpupool-agent .
```

Mirror HTTP nội bộ cùng cấu trúc cũng dùng được (`UV_PYTHON_INSTALL_MIRROR=https://mirror.corp/pbs`). Nếu
thiếu file, uv báo lỗi kèm URL nó đã thử: dựa vào đó để sửa lại cấu trúc.

## 3. Binary uv

Không lưu ở đây. Cả hai Dockerfile nhận `--build-arg UV_IMAGE=<registry mirror>/astral-sh/uv:0.10`. Image
coordinator còn có thể dùng `--build-arg UV_FROM_PYPI=1` (cài uv bằng pip).
