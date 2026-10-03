# Đóng góp cho gpupool

> Tiếng Việt. English: [CONTRIBUTING.md](CONTRIBUTING.md)

Cảm ơn bạn đã giúp đỡ. Hướng dẫn này cố ý ngắn: làm theo những gì đã có, chứng minh thay đổi của bạn chạy được, và
giữ tài liệu hai ngôn ngữ đồng bộ.

Khi đóng góp, bạn đồng ý công sức của mình được phát hành theo [Giấy phép MIT](LICENSE). Hãy tuân theo
[Quy tắc ứng xử](CODE_OF_CONDUCT.vi.md). Với vấn đề bảo mật, đừng mở issue công khai: xem
[SECURITY.vi.md](SECURITY.vi.md).

## Thiết lập môi trường phát triển

Bạn cần Python 3.12 và [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/longduongbao29/multi-gpu-inference.git
cd multi-gpu-inference
uv sync                 # tạo .venv với thư viện chạy và thư viện dev theo uv.lock
uv run pytest -q        # khoảng 810 test, không cần GPU, llama.cpp hay mạng
```

Nếu `python` không nằm trong PATH, luôn chạy qua `uv run python ...`.

### Test cần phần cứng thật

Các test gắn nhãn `real` cần tệp thực thi llama.cpp, một tệp GGUF và một GPU. Mặc định chúng bị bỏ qua
(`addopts = "-m 'not real'"`). Chạy bằng:

```bash
uv run pytest -m real
```

### Test end-to-end với image thật

`scripts/ci_e2e.py` khởi động coordinator thật và 3 agent CPU từ image Docker, phục vụ một mô hình nhỏ chia qua RPC
và (trừ khi dùng `--skip-convert`) chuyển đổi một mô hình Hugging Face. Cần Docker và, cho giai đoạn chuyển đổi,
image coordinator build với `WITH_CONVERT=1` cùng quyền truy cập huggingface.co. Đây là thứ CI chạy; hãy chạy cục bộ
với mọi thay đổi đụng tới image, llama.cpp hoặc đường chạy nhiều máy chủ.

### Làm việc với giao diện

Giao diện là HTML/CSS thuần và [Alpine.js](https://alpinejs.dev/) (đặt sẵn trong `src/gpupool/ui/vendor/`), **không
có bước build**. Đừng thêm bundler, framework hay phụ thuộc npm. Khi sửa giao diện, chạy máy chủ giả:

```bash
uv run python scripts/ui_mock_server.py   # http://127.0.0.1:8090, khóa quản trị: dev
```

## Phong cách mã

- Làm giống mã xung quanh: cách đặt tên, cấu trúc, kiểu dữ liệu, xử lý lỗi. Không có công cụ định dạng để chạy; hãy
  đọc các tệp lân cận trước.
- **Chú thích trong mã chỉ bằng tiếng Anh.** Giải thích *tại sao* với mọi thứ không hiển nhiên (một ngưỡng, một cách
  né lỗi, thứ tự cờ, một bất biến), không mô tả dòng lệnh làm gì. Nếu chú thích ghi lại một lỗi hay một số đo thật,
  hãy nói rõ.
- Giữ phụ thuộc nhẹ. Đừng thêm phụ thuộc nặng (framework ML, bộ web lớn, công cụ build) khi chưa bàn trong một issue.
  Mọi thứ mới được đóng gói phải ghi thêm vào [THIRD_PARTY_NOTICES.vi.md](THIRD_PARTY_NOTICES.vi.md) (cả hai ngôn
  ngữ).
- Đừng nuốt lỗi, đừng ghi nguồn dữ liệu gốc theo cách không nguyên tử, và đừng để lại tiến trình, tệp hay cổng khi
  thất bại.

## Kiểm thử

- **Mỗi bản sửa lỗi đi kèm một test hồi quy** thất bại nếu thiếu bản sửa.
- Hành vi mới đi kèm test. Test không được phụ thuộc hay ghi vào thư mục làm việc (dùng `tmp_path`); bộ test phải
  chạy qua từ một thư mục trống.
- Mọi thứ đụng tới llama.cpp, Dockerfile, cách agent khởi chạy engine hay đường RPC còn phải được chạy thật (tệp thực
  thi thật, image thật, `scripts/ci_e2e.py`). Test dùng mock chỉ chứng minh mock chạy được: hãy ghi trong pull request
  bạn đã chạy gì trên thực tế và điều gì bạn chưa làm được.
- `uv run pytest -q` phải xanh trước khi mở pull request.

## Tài liệu

Mọi tài liệu tồn tại bằng tiếng Anh và tiếng Việt: `X.md` và `X.vi.md` (hoặc `X.en.md` và `X.vi.md` trong `docs/`),
mỗi tệp dẫn liên kết tới tệp kia ở dòng 3. Sửa một bản thì sửa bản kia trong cùng pull request. Tiếng Việt phải tự
nhiên và có đủ dấu. Ngoại lệ duy nhất là `LICENSE`, chỉ có tiếng Anh. Chú thích trong mã vẫn là tiếng Anh.

Cập nhật [CHANGELOG.vi.md](CHANGELOG.vi.md) (và `CHANGELOG.md`) ở mục *Chưa phát hành* với mọi thay đổi người dùng
nhìn thấy.

## Thông điệp commit

Nói rõ cái gì thay đổi và **tại sao**. Với lỗi, ghi lại: cái gì hỏng, trong điều kiện nào, và điều gì giờ ngăn nó xảy
ra.

```
Create the data folder before opening the database

The conversion job manager opened <data>/coordinator.db without creating
<data>/, so create_app failed on a fresh checkout. The manager now creates
the folder like the store does; a regression test fails without the fix.
```

Commit nhỏ, tập trung dễ duyệt hơn một commit lớn.

## Pull request

1. Mở issue trước với việc lớn hoặc việc đổi hành vi, để thống nhất hướng đi.
2. Tạo nhánh từ `master`, thực hiện thay đổi, chạy `uv run pytest -q` và (khi liên quan) các lần chạy thật ở trên.
3. Điền mẫu pull request: cái gì và tại sao, đã kiểm thử thế nào, tài liệu đã cập nhật cả hai ngôn ngữ.
4. Mỗi pull request chỉ một mối quan tâm. CI (test đơn vị, build image, job end-to-end) phải qua.
5. Hãy chờ nhận xét khi review; chúng nói về mã, không nói về bạn.

## Hỏi ở đâu

- Lỗi và ý tưởng tính năng: [GitHub Issues](https://github.com/longduongbao29/multi-gpu-inference/issues) dùng các
  mẫu có sẵn.
- Câu hỏi: mở issue và gắn nhãn `question`.
- Bảo mật: [SECURITY.vi.md](SECURITY.vi.md).
