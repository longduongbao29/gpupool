# Thông báo về phần mềm bên thứ ba

> Tiếng Việt. English: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

Bản thân gpupool được phát hành theo [Giấy phép MIT](LICENSE). gpupool đi kèm hoặc kéo về các thành phần dưới đây,
mỗi thành phần theo giấy phép riêng. Tệp này chỉ mang tính thông tin, không phải tư vấn pháp lý: văn bản giấy phép
đi kèm từng thành phần mới là căn cứ.

"Đã xác minh" nghĩa là giấy phép được đọc từ metadata của gói cài trong môi trường của kho mã này
(`importlib.metadata`, đối chiếu `uv.lock`) hoặc từ chính tệp. "Theo upstream" nghĩa là giấy phép do dự án gốc công
bố, chưa đối chiếu với bản cài cục bộ.

## Mô hình không thuộc phạm vi giấy phép này

**gpupool không bao gồm, không cấp phép và không trao quyền đối với bất kỳ mô hình nào.** Trọng số mô hình (Llama,
Qwen, Gemma, Mistral và mọi mô hình khác) là các tác phẩm riêng, có giấy phép và chính sách sử dụng riêng do bên
phát hành đặt ra.

- Bạn tự chịu trách nhiệm đọc và tuân thủ giấy phép của mọi mô hình bạn tải, chuyển đổi hoặc phục vụ.
- Mô hình bị khóa (gated) trên Hugging Face yêu cầu bạn chấp nhận điều khoản của bên phát hành trên huggingface.co
  bằng tài khoản của bạn thì gpupool (dùng token của bạn) mới tải được.
- Chuyển một mô hình sang GGUF hoặc lượng tử hóa nó **không** làm thay đổi giấy phép của mô hình. Kết quả là bản phái
  sinh của trọng số gốc và vẫn theo điều khoản gốc.
- Dữ liệu liên quan đến mô hình duy nhất gpupool đi kèm là văn bản hiệu chỉnh có sẵn
  (`src/gpupool/converter/data/calibration.txt`). Đó là văn bản gốc viết riêng cho gpupool, dùng giấy phép MIT như
  phần còn lại của dự án; xem `src/gpupool/converter/data/README.md`.

## Đóng gói sẵn trong mã nguồn

| Thành phần | Phiên bản | Giấy phép | Vị trí | Liên kết |
| --- | --- | --- | --- | --- |
| Alpine.js | 3.14.9 | MIT (đã xác minh: dòng đầu của tệp) | `src/gpupool/ui/vendor/alpine.min.js` | <https://github.com/alpinejs/alpine> |
| marked | 15.0.12 | MIT (đã xác minh: dòng đầu của tệp) | `src/gpupool/ui/vendor/marked.min.js` (Markdown trong Playground) | <https://github.com/markedjs/marked> |
| DOMPurify | 3.2.6 | Apache-2.0 hoặc MPL-2.0, tùy chọn (đã xác minh: dòng đầu của tệp); dùng theo Apache-2.0 | `src/gpupool/ui/vendor/purify.min.js` (làm sạch Markdown đã render) | <https://github.com/cure53/DOMPurify> |

## llama.cpp (engine và bộ chuyển đổi)

| Thành phần | Phiên bản | Giấy phép | Liên kết |
| --- | --- | --- | --- |
| llama.cpp (`llama-server`, `ggml-rpc-server`, `llama-quantize`, `llama-imatrix`, `llama-tokenize`, ggml) | tag `b11413` | MIT (theo upstream) | <https://github.com/ggml-org/llama.cpp> |

- **Image agent** (`docker/agent.Dockerfile`) biên dịch llama.cpp b11413 với CUDA và chứa các tệp thực thi.
- **Image coordinator** build với `WITH_CONVERT=1` (`docker/coordinator.Dockerfile`) biên dịch các công cụ CPU
  `llama-quantize`, `llama-tokenize`, `llama-simple`, `llama-imatrix`, và sao chép `convert_hf_to_gguf.py`,
  `conversion/`, `gguf-py/` từ cùng tag vào `/opt/llama.cpp`, kèm tệp `LICENSE` của upstream
  (`/opt/llama.cpp/LICENSE`).
- gpupool không sửa llama.cpp. Dockerfile tải hoặc build nguyên bản từ tag đã ghim.

## Thư viện Python chạy của gpupool

Phụ thuộc trực tiếp (`pyproject.toml`), phiên bản theo `uv.lock`. Tất cả đã xác minh từ metadata đã cài.

| Gói | Phiên bản | Giấy phép |
| --- | --- | --- |
| fastapi | 0.142.2 | MIT |
| gguf | 0.19.0 | MIT |
| httpx | 0.28.1 | BSD-3-Clause |
| nvidia-ml-py | 13.615.71 | BSD (theo khai báo của gói) |
| psutil | 7.2.2 | BSD-3-Clause |
| pydantic | 2.13.5 | MIT |
| uvicorn (extra `standard`) | 0.54.0 | BSD-3-Clause |

Các gói phụ thuộc gián tiếp đáng chú ý (cũng đã xác minh từ metadata):

| Gói | Phiên bản | Giấy phép |
| --- | --- | --- |
| starlette | 1.7.0 | BSD-3-Clause |
| pydantic-core | 2.46.5 | MIT |
| anyio | 4.15.1 | MIT |
| h11 | 0.16.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httptools | 0.8.0 | MIT |
| websockets | 17.1 | BSD-3-Clause |
| watchfiles | 1.3.0 | MIT |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| PyYAML | 6.0.3 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| requests | 2.34.2 | Apache-2.0 |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| certifi | 2026.7.22 | MPL-2.0 |
| idna | 3.20 | BSD-3-Clause |
| urllib3 | 2.8.0 | MIT |
| charset-normalizer | 3.5.2 | MIT |
| typing-extensions | 4.16.0 | PSF-2.0 |
| annotated-types, typing-inspection, annotated-doc | 0.8.0, 0.4.4, 0.0.5 | MIT |
| opentelemetry-api | 1.45.0 | Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| colorama (chỉ Windows) | 0.4.6 | BSD-3-Clause (metadata ghi "BSD License") |
| uvloop (chỉ Linux/macOS, qua `uvicorn[standard]`) | theo lúc cài | MIT OR Apache-2.0 (theo upstream; không được cài trên máy Windows dùng để viết tệp này) |

Danh sách đầy đủ và chính xác cho nền tảng của bạn nằm trong `uv.lock`. Các gói chỉ dùng khi phát triển (pytest,
pytest-asyncio, respx, pluggy, iniconfig, Pygments) không nằm trong image hay wheel; theo thứ tự giấy phép là MIT,
Apache-2.0, BSD-3-Clause, MIT, MIT và BSD-2-Clause (đã xác minh).

## Môi trường chuyển đổi (image coordinator với `WITH_CONVERT=1`)

Image chứa một virtualenv riêng, `/opt/convert-venv`, cài từ `requirements/requirements-convert_hf_to_gguf.txt` của
llama.cpp tại tag b11413 (bản PyTorch CPU). Các gói này không được cài trong môi trường của kho mã này nên giấy phép
được ghi theo upstream; nếu cần chắc chắn, xem danh sách chính xác trong image bằng
`docker run --rm --entrypoint /opt/convert-venv/bin/python <image> -m pip list`.

| Gói | Giấy phép (theo upstream) |
| --- | --- |
| torch | BSD-3-Clause |
| transformers | Apache-2.0 |
| sentencepiece | Apache-2.0 |
| numpy | BSD-3-Clause (kèm các thành phần đóng gói, xem trên) |
| protobuf | BSD-3-Clause |
| gguf | MIT |

Tệp requirements có thể kéo thêm các gói gián tiếp (ví dụ huggingface-hub, tokenizers, safetensors, PyYAML,
requests, tqdm); theo upstream chúng đều dùng giấy phép cho phép (Apache-2.0, MIT, BSD). Bản wheel PyTorch đóng gói
các thành phần bên thứ ba kèm thông báo riêng; xem <https://github.com/pytorch/pytorch/blob/main/LICENSE> và `NOTICE`.

## Công cụ và image nền

| Thành phần | Giấy phép | Ghi chú |
| --- | --- | --- |
| uv (tệp thực thi sao chép từ `ghcr.io/astral-sh/uv`, hoặc `pip install uv`) | MIT OR Apache-2.0 (theo upstream) | <https://github.com/astral-sh/uv> |
| `python:3.12-slim` (image nền của coordinator) | Python theo Giấy phép PSF; image còn chứa các gói Debian, mỗi gói có giấy phép riêng | <https://docs.python.org/3/license.html>, <https://hub.docker.com/_/python>; văn bản từng gói nằm ở `/usr/share/doc/*/copyright` trong image |
| `nvidia/cuda:*-devel-ubuntu*` và `nvidia/cuda:*-runtime-ubuntu*` (image nền của agent, mặc định CUDA 12.8.1 trên Ubuntu 22.04) | NVIDIA CUDA EULA và giấy phép NVIDIA Deep Learning Container, cùng giấy phép các gói Ubuntu | xem lưu ý bên dưới |
| Python 3.12 (image agent; bản CPython do uv tải) | Giấy phép PSF | python-build-standalone, <https://github.com/astral-sh/python-build-standalone> |
| iptables, libgomp1, libibverbs1 / ibverbs-providers (rdma-core) và các gói Debian/Ubuntu khác cài trong image | GPL/LGPL và khác, theo từng gói | văn bản ở `/usr/share/doc/*/copyright` trong image |

### Điều khoản của NVIDIA đối với image agent

Image agent được xây trên các image CUDA của NVIDIA và chứa thư viện runtime CUDA của NVIDIA. **Nếu bạn build, xuất
bản hoặc phân phối lại image agent, bạn chấp nhận điều khoản của NVIDIA cho các thành phần đó.** Chúng chịu sự điều
chỉnh của [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) và
[NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license).
Hãy đọc kỹ trước khi phân phối image ra ngoài tổ chức của bạn. Giấy phép MIT của gpupool không áp dụng cho các thành
phần của NVIDIA, và driver GPU NVIDIA trên máy chủ không thuộc gpupool.

## Cập nhật tệp này

Khi thêm hoặc nâng cấp một phụ thuộc, image nền hoặc tag llama.cpp đã ghim, hãy cập nhật cả hai phiên bản ngôn ngữ
của tệp này trong cùng một thay đổi.
