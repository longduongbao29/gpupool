# Built-in calibration text

[English](#english) | [Tiếng Việt](#tiếng-việt)

## English

`calibration.txt` is the text gpupool feeds to llama.cpp `llama-imatrix` when it
quantizes a model to a low-bit GGUF type. An importance matrix records which
weights matter most for typical inputs, and the quantizer keeps those more
precise. A calibration text that covers many kinds of input gives better
low-bit models than one that is narrow (for example only English encyclopedia
prose).

**Origin and license.** The text is original, written for gpupool. It is not
copied or paraphrased from wikitext, calibration_datav3 or any other dataset,
book, article or lyrics. It contains no real personal data (phone numbers, IDs
and e-mail addresses are fictional). It is distributed under the same license
as the project.

**Composition** (approximate share of bytes, about 115 KB in total):

| Share | Content |
|-------|---------|
| ~30 % | English prose: news, explanations, how-tos, stories, e-mails, reviews, contract clauses, abstracts, forum posts |
| ~17 % | Vietnamese (full diacritics): news, conversation, instructions, story, verse, GPU/computer text, math |
| ~10 % | Chinese, Japanese, Korean, French, German, Spanish, Russian, Arabic, Hindi, Thai, Indonesian |
| ~28 % | Code: Python, JS/TS, Go, Rust, C/C++/CUDA, Java, SQL, Bash, YAML, Dockerfile, HTML/CSS, stack traces, logs |
| ~7 % | Math and structured data: worked problems, LaTeX-style formulas, tables, CSV, JSON, Markdown, identifiers |
| ~14 % | Chat-formatted dialogue: Q&A, troubleshooting, support, polite refusals, multilingual turns |
| ~3 % | Edge content: emoji, punctuation runs, URLs, very short/long lines, odd whitespace, mixed scripts |

**Use your own text.** Set `ConvertAdvanced.calibration_path` to an absolute
path of a `.txt` file on the coordinator machine (maximum 20 MB). It replaces
this built-in file for that conversion. Aim for varied, representative text in
the languages and domains your users actually send.

## Tiếng Việt

`calibration.txt` là văn bản gpupool đưa cho `llama-imatrix` của llama.cpp khi
lượng tử hóa mô hình xuống kiểu GGUF ít bit. Ma trận tầm quan trọng ghi lại
những trọng số quan trọng với đầu vào điển hình, và bộ lượng tử hóa giữ chúng
chính xác hơn. Văn bản hiệu chỉnh càng đa dạng thì mô hình ít bit càng tốt; văn
bản hẹp (ví dụ chỉ văn xuôi bách khoa tiếng Anh) làm lệch ma trận.

**Nguồn gốc và giấy phép.** Văn bản hoàn toàn gốc, viết riêng cho gpupool,
không sao chép hay diễn đạt lại từ wikitext, calibration_datav3 hay bất kỳ tập
dữ liệu, sách, bài báo, lời bài hát nào. Không chứa dữ liệu cá nhân thật (số
điện thoại, mã định danh, e-mail đều là giả). Dùng cùng giấy phép với dự án.

**Thành phần** (tỷ lệ byte xấp xỉ, tổng khoảng 115 KB):

| Tỷ lệ | Nội dung |
|-------|----------|
| ~30 % | Văn xuôi tiếng Anh: tin tức, giải thích, hướng dẫn, truyện, e-mail, đánh giá, điều khoản hợp đồng, tóm tắt khoa học, bài diễn đàn |
| ~17 % | Tiếng Việt có dấu đầy đủ: tin tức, hội thoại, hướng dẫn, truyện, thơ, văn bản về GPU/máy tính, toán |
| ~10 % | Trung, Nhật, Hàn, Pháp, Đức, Tây Ban Nha, Nga, Ả Rập, Hindi, Thái, Indonesia |
| ~28 % | Mã nguồn: Python, JS/TS, Go, Rust, C/C++/CUDA, Java, SQL, Bash, YAML, Dockerfile, HTML/CSS, stack trace, log |
| ~7 % | Toán và dữ liệu có cấu trúc: bài giải, công thức kiểu LaTeX, bảng, CSV, JSON, Markdown, mã định danh |
| ~14 % | Hội thoại dạng chat: hỏi đáp, xử lý sự cố, hỗ trợ khách hàng, từ chối lịch sự, nhiều ngôn ngữ |
| ~3 % | Nội dung biên: emoji, dấu câu lặp, URL, dòng rất ngắn/dài, khoảng trắng lạ, nhiều hệ chữ trong một dòng |

**Dùng văn bản riêng.** Đặt `ConvertAdvanced.calibration_path` là đường dẫn
tuyệt đối tới tệp `.txt` trên máy coordinator (tối đa 20 MB). Tệp này thay thế
tệp có sẵn cho lần chuyển đổi đó. Nên chọn văn bản đa dạng, đại diện cho các
ngôn ngữ và lĩnh vực người dùng thực sự gửi.
