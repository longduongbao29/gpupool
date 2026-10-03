## What and why / Cái gì và tại sao

<!-- What changes, and why. For a bug: what broke, under what conditions, what now prevents it.
     Thay đổi gì và tại sao. Với lỗi: cái gì hỏng, điều kiện nào, điều gì giờ ngăn nó. -->

Closes #

## How it was tested / Đã kiểm thử thế nào

- [ ] `uv run pytest -q` passes / đã xanh
- [ ] A regression test for every bug fixed / có test hồi quy cho mỗi lỗi đã sửa
- [ ] Run for real if it touches llama.cpp, Docker or the multi-server path (`scripts/ci_e2e.py`, real binaries); say what you ran and what you could not / chạy thật nếu đụng tới llama.cpp, Docker hoặc đường nhiều máy chủ; ghi rõ đã chạy gì, chưa làm được gì

## Checklist

- [ ] Docs updated in English and Vietnamese (`X.md` + `X.vi.md`) / tài liệu cập nhật cả EN và VI
- [ ] `CHANGELOG.md` and `CHANGELOG.vi.md` updated under Unreleased / đã cập nhật mục Unreleased
- [ ] New dependency or base image added to `THIRD_PARTY_NOTICES.md` (both languages) / đã ghi vào thông báo bên thứ ba
- [ ] Code comments are in English and explain why / chú thích mã bằng tiếng Anh, giải thích tại sao
- [ ] No secrets, tokens or private paths in the diff / không có khóa, token hay đường dẫn riêng tư
