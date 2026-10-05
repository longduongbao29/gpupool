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

Yêu cầu trên server GPU: driver NVIDIA ≥ 525 (≥ 570 với RTX 50 / Blackwell), Docker và
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Kiểm tra bằng:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

## 3. Serve một model (trên UI)

1. **Models → Add model**: nhập repo Hugging Face (ví dụ `Qwen/Qwen2.5-7B-Instruct-GGUF`), chọn file
   `.gguf`, bấm **Download**. Hoặc tab **Path** cho file đã có sẵn trên máy coordinator.
2. **New model**: đặt tên (đây là giá trị `model` mà client sẽ dùng), chọn file. Server và GPU: để
   **All servers and GPUs**, hoặc chọn **Only selected ones** (xem bên dưới).
3. **Start**. Trạng thái chuyển *starting → running*. **Stop** để trả GPU.

Nếu model lớn hơn mọi GPU đơn lẻ, nó được tự chia qua nhiều GPU và nhiều server.

### Giới hạn model trong một số server hoặc GPU

Mặc định model được dùng mọi server và GPU trong pool. Chọn **Only selected ones** trong form model để giới hạn: một
cây gồm các server và GPU của chúng hiện ra (kèm bộ nhớ trống), tick những gì model được phép dùng. Tick cả một
server nghĩa là cho phép mọi GPU của server đó, **kể cả GPU được thêm vào sau này** (lưu dưới dạng `"<node>/*"`);
tick từng GPU thì chỉ cho phép các GPU đó (`"<node>/<device>"`). Đây là **giới hạn chứ không phải đặt thủ công**:
gpupool vẫn tự chọn cách đặt tốt nhất, nhưng chỉ trong các thiết bị được phép. Tab Servers không bị ảnh hưởng: GPU
bạn không chọn ở đây vẫn được bật trong pool cho các model khác. Lưu với *Only selected ones* mà chưa tick gì sẽ bị
từ chối (danh sách rỗng nghĩa là "tất cả"). Thẻ model hiện nhãn *Limited to ...*. Qua HTTP đây là `pin_devices`, xem
[API.vi.md](API.vi.md#3-model-apimodels).

### Sao chép những gì client cần

Mỗi thẻ model có nút sao chép cho **endpoint** (URL gốc), **tên model** (giá trị của `"model"` trong request) và một
lệnh **curl** dựng sẵn cho model này (kèm chỗ giữ cho header `Authorization` khi có đặt API key).

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

## Serve model chưa có GGUF (chuyển đổi)

gpupool phục vụ file GGUF. Nhiều model chỉ được phát hành dưới dạng trọng số Hugging Face (`safetensors` hoặc
`.bin` của PyTorch). Coordinator có thể tự chuyển model đó sang GGUF và đưa vào thư viện model.

**Nếu đã có bản GGUF thì nên dùng bản đó.** Tải file có sẵn nhanh hơn và không cần chuyển đổi. Bước inspect liệt
kê các bản GGUF đã được phát hành của model (`gguf_alternatives`, kèm liên kết *download instead*). Chỉ dùng chuyển
đổi khi chưa có bản nào, khi bạn muốn một kiểu lượng tử hoá chưa ai phát hành, hoặc khi trọng số đã nằm sẵn trên
ổ đĩa của server.

### Cách làm trên UI

1. **Models → Convert a model**. (Hộp thoại **Add model** cũng có nút **Convert to GGUF** khi repo bạn nhập không
   chứa file `.gguf` nào.) Chọn **Hugging Face repo** (`owner/name`, có thể kèm revision) hoặc **Folder on the
   server** (đường dẫn tuyệt đối của thư mục có `config.json`, các file tokenizer và trọng số; đường dẫn trên máy
   chủ được dịch giống đường dẫn thư viện, xem [Dùng file model có sẵn trên server](#dùng-file-model-có-sẵn-trên-server)).
2. **Inspect**. Chưa tải gì cả: gpupool chỉ đọc config và danh sách file. Bạn thấy kiến trúc và việc bộ chuyển đổi
   được ghim (llama.cpp b11413) có hỗ trợ hay không, số tham số, số lớp, độ dài ngữ cảnh, dung lượng tải về, cùng các
   cảnh báo (repo gated, đã lượng tử hoá sẵn, kèm mã tuỳ biến, đã có bản GGUF).
3. Chọn **loại lượng tử hoá** (mục kế tiếp). Mỗi dòng cho biết dung lượng file và VRAM ước lượng, vừa một GPU hay
   chỉ vừa pool, và loại nào được đề xuất kèm lý do.
4. Kiểm tra **tên file đầu ra** (mặc định `<model>-<LOẠI>.gguf`; phải kết thúc bằng `.gguf`), tuỳ chọn **Keep downloaded
   source**, tuỳ chọn **Advanced**. Rồi bấm **Start conversion**.
5. Theo dõi bảng **Conversions** ngay trên trang đó: các giai đoạn *Download → Convert → Calibrate (chỉ khi có tính
   importance matrix) → Quantize → Validate*, thanh
   tiến độ, log và chi tiết kiểm tra. Các job chạy lần lượt từng cái, ở mức ưu tiên CPU thấp để việc inference trên
   máy coordinator không bị chậm. Có sẵn **Cancel**, **Retry** (job lỗi hoặc đã huỷ) và xoá.
6. Khi job ở trạng thái *done*, file đã nằm trong thư viện (nguồn `convert`). **Deploy this model** mở form model
   mới với file đó.

Các bước tương tự qua HTTP: [API.vi.md](API.vi.md#12-chuyển-đổi-apiconvert).

### Chọn loại lượng tử hoá

Lượng tử hoá làm file nhỏ đi và cần ít bộ nhớ hơn, đổi lại một phần chất lượng. Các con số chất lượng là mức thay đổi
perplexity so với F16 trên Llama-3-8B do llama-quantize công bố (càng gần 0 càng tốt); *bpw* là số bit trên mỗi
trọng số, tính trung bình cả model.

| Loại | bpw | Chất lượng | Dùng khi |
| --- | --- | --- | --- |
| `BF16`, `F16` | 16 | không mất mát | bản tham chiếu, file lớn nhất; do bộ chuyển đổi ghi trực tiếp |
| `Q8_0` | 8.52 | +0.0026, thực tế không phân biệt được với F16 | model nhỏ, hoặc dư VRAM; ghi trực tiếp |
| `Q6_K` | 6.57 | +0.0217, rất gần bản gốc | ưu tiên chất lượng, khi vừa bộ nhớ |
| `Q5_K_M` | 5.70 | +0.0569, rất tốt | mặc định tốt cho model nhỏ |
| `Q5_K_S` | 5.57 | +0.1049 | nhỏ hơn và kém hơn `Q5_K_M` một chút |
| `Q4_K_M` | 4.90 | +0.1754 | điểm cân bằng dung lượng/chất lượng thông dụng cho model lớn |
| `Q4_K_S` | 4.67 | +0.2689 | nhỏ hơn `Q4_K_M` một chút |
| `IQ4_XS` | 4.25 | gần `Q4_K_S` | nhỏ hơn `Q4_K_S` với chất lượng tương đương |
| `Q4_0` | 4.64 | +0.4685 | định dạng cũ; thường `Q4_K_S` tốt hơn |
| `Q3_K_L` | 4.31 | +0.5562, mất chất lượng thấy rõ | bộ nhớ eo hẹp |
| `Q3_K_M` | 4.00 | +0.6569, mất chất lượng rõ | chỉ khi bộ nhớ eo hẹp |
| `IQ3_M` | 3.76 | chưa có số công bố; thường tốt hơn `Q3_K_M` ở dung lượng nhỏ hơn | bộ nhớ eo hẹp, tốt hơn các loại `Q3_K` |
| `IQ3_S` | 3.67 | chưa có số công bố; tốt hơn `Q3_K_S` ở dung lượng tương đương | bộ nhớ eo hẹp |
| `Q3_K_S` | 3.65 | +1.6321, mất nhiều | phương án cuối cùng |
| `IQ3_XS`, `IQ3_XXS` | 3.50, 3.26 | mất rõ đến mất nhiều | bộ nhớ rất eo hẹp (**bắt buộc** có importance matrix) |
| `Q2_K` | 3.17 | +3.5199, mất rất nhiều | phương án cuối cùng |
| `IQ2_M`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS` | 2.94, 2.75, 2.60, 2.39 | mất nhiều | khi không có loại nào lớn hơn vừa (**bắt buộc** có importance matrix) |
| `IQ1_M`, `IQ1_S` | 2.15, 2.01 | mất cực nhiều | chỉ cho model rất lớn buộc phải vừa bộ nhớ (**bắt buộc** có importance matrix) |

Các loại `IQ` nén model xuống dưới 4 bit trên mỗi trọng số. Với các loại IQ1, IQ2, `IQ3_XXS` và `IQ3_XS`,
llama-quantize từ chối chạy nếu không có **importance matrix** (ma trận tầm quan trọng); các loại còn lại sẽ tốt hơn
khi có nó. gpupool tự tính giúp bạn, xem mục kế tiếp. bpw của các dòng IQ là trung bình cả file (lớp đầu ra và các
tensor nhạy nhất giữ độ chính xác cao hơn), nên cao hơn số bit danh nghĩa của định dạng. Sai số của ước lượng dung
lượng so với file thật: `IQ2_XS` +6 %, `IQ3_M` -4 %, `Q8_0` -1 %.

Loại được đề xuất theo các quy tắc sau:

- **Model lớn** (từ 3 tỷ tham số trở lên): loại tốt nhất trong `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M` mà vừa **một GPU**;
  nếu không loại nào vừa thì lấy loại tốt nhất vừa **cả pool** (khi đó model bị chia qua RPC, chậm hơn). Nếu ngay cả
  `Q4_K_M` cũng không vừa, vẫn đề xuất `Q4_K_M` kèm lý do nói rõ điều đó; `Q3_K_M` và `Q2_K` vẫn nằm trong danh
  sách như các lựa chọn.
- **Model nhỏ** (dưới 3 tỷ): không bao giờ mặc định thấp hơn `Q5_K_M`, vì model nhỏ mất chất lượng nhanh nhất. Chỉ
  khi không loại nào trong `Q8_0`, `Q6_K`, `Q5_K_M` vừa thì mới xét `Q4_K_M`, và lý do có nêu.
- Khi cụm không có GPU nào (không có thông tin vừa/không vừa): `Q8_0` cho model nhỏ, `Q5_K_M` cho model dưới 15 tỷ,
  còn lại `Q4_K_M`. Khi chưa biết số tham số: `Q4_K_M`.

K-quant cần các hàng có số giá trị chia hết cho 256. Model có hidden size khác (SmolLM2-135M: 576, Qwen2.5-0.5B: 896)
sẽ dùng một loại cũ hơn cho các tensor đó, nên file lớn hơn so với bảng. Dung lượng hiển thị trong hộp thoại đã tính
đến điều này và cả các ma trận embedding.

**Nguồn đã lượng tử hoá sẵn sẽ mất chất lượng hai lần** (lần của chính nó, rồi lần của bạn). gpupool đọc ngược được
các nguồn `fp8`, `gptq`, `bitnet`, `compressed-tensors`, `modelopt` và `mxfp4` (kèm cảnh báo). **Model AWQ và
bitsandbytes không được hỗ trợ**: yêu cầu bị từ chối với HTTP 422. Hãy chuyển model gốc (base model) của nó; hộp
thoại gợi ý model đó khi model card có ghi.

### Importance matrix (loại IQ và file rất nhỏ)

**Importance matrix** ghi lại những trọng số nào quan trọng nhất khi model đọc văn bản điển hình. Bộ lượng tử hoá
nhờ đó giữ các trọng số ấy chính xác hơn và nén phần còn lại mạnh tay hơn; càng ít bit thì càng có ích. gpupool tính
nó trong một giai đoạn riêng, **Calibrate**: chạy `llama-imatrix` trên file trung gian 16 bit, với một văn bản hiệu
chỉnh, trên CPU, trước bước Quantize.

Tuỳ chọn **Importance matrix** có ba giá trị:

| Giá trị | Chuyện gì xảy ra |
| --- | --- |
| **Auto** (mặc định) | tính ma trận khi loại đó bắt buộc phải có (các loại IQ1, IQ2, `IQ3_XXS`, `IQ3_XS`) hoặc dưới 4 bit trên mỗi trọng số (các loại `IQ3` còn lại, `Q3_K_S`, `Q2_K`), nơi nó có ích nhất. Các loại từ 4 bit trở lên (`Q4_K_M`, `Q5_K_M`...) thì không tính |
| **On** | luôn tính, cho mọi loại do llama-quantize ghi. Chậm hơn, và cải thiện chất lượng ở mọi kích cỡ, kể cả `Q4_K_M` |
| **Off** | không bao giờ tính. Bị từ chối (HTTP 422) với các loại bắt buộc phải có, nên chỉ dùng cho các loại khác để tiết kiệm thời gian |

`F16`, `BF16` và `Q8_0` không bao giờ dùng ma trận (không có bước llama-quantize). Khi image coordinator không có
`llama-imatrix` (cài ngoài Docker mà thiếu), loại bắt buộc có ma trận, hoặc **On**, bị từ chối với HTTP 503; với
**Auto** trên loại chỉ hưởng lợi từ ma trận thì job cứ chạy không có nó. UI làm mờ các loại bắt buộc có ma trận khi
`imatrix_available` là false (`GET /api/convert/options`).

**Chi phí thời gian.** Calibrate là phần chậm nhất trên CPU, vì nó cho cả model chạy qua văn bản. Đo trên CPU laptop
trong WSL, với mặc định 100 chunk (mỗi chunk 512 token): Qwen2.5-1.5B sang `IQ3_M` mất tổng cộng **28 phút**, trong
đó calibrate khoảng **19 đến 20 phút**. Để so sánh, Qwen2.5-0.5B sang `Q4_K_M` (không có ma trận) mất khoảng 2,5 phút
gồm cả 107 giây tải. Thời gian tăng theo kích thước model và số chunk, nên **ít chunk hơn = nhanh hơn** nhưng ma trận
thô hơn: đặt *Calibration chunks* thấp (ví dụ 20 đến 30) để thử nhanh, giữ mặc định cho model bạn sẽ dùng lâu. CPU
mạnh hơn hoặc model nhỏ hơn thì rút ngắn tương ứng.

**Văn bản hiệu chỉnh.** Mặc định gpupool dùng văn bản đi kèm: văn bản gốc đa ngôn ngữ (văn xuôi tiếng Anh và tiếng
Việt, nhiều ngôn ngữ khác, code, toán, JSON, lượt chat), viết riêng cho gpupool nên không vướng giấy phép và không bị
lệch do văn bản hẹp (xem `src/gpupool/converter/data/README.md`). Muốn hiệu chỉnh theo lĩnh vực của bạn, hãy đưa
**Calibration text**: đường dẫn tuyệt đối của một tệp `.txt` trên server (tối đa 20 MB; đường dẫn trên máy chủ được
dịch giống đường dẫn thư viện). Tệp được sao chép khi job bắt đầu, nên sửa tệp về sau không ảnh hưởng job đang chạy.
Ma trận tính trên văn bản hẹp (ví dụ chỉ tiếng Anh) làm model tốt hơn trên loại văn bản đó và kém đi ở phần còn lại.

### Các tuỳ chọn nâng cao nói đơn giản

| Tuỳ chọn | Tác dụng |
| --- | --- |
| Intermediate precision | File 16/32 bit được ghi trước khi lượng tử hoá. *Auto* giữ độ chính xác của chính model (trọng số bf16 giữ bf16, còn lại f16). f32 hiếm khi cần và làm gấp đôi dung lượng đĩa tạm |
| Output tensor type, Token embedding type | Độ chính xác của lớp đầu ra và của bảng embedding từ. Giữ cao hơn (ví dụ `q8_0`) tốn thêm ít dung lượng và có thể giúp chất lượng ở các loại nhỏ. Mặc định để llama-quantize tự quyết |
| Leave output tensor | Không lượng tử hoá lớp đầu ra: file lớn hơn một chút, chất lượng nhỉnh hơn một chút |
| Pure | Dùng loại đã chọn cho mọi tensor thay vì hỗn hợp thông thường. Thường làm giảm chất lượng ở cùng dung lượng; để thử nghiệm |
| Importance matrix | *Auto* / *On* / *Off*, xem mục trước |
| Calibration text | Đường dẫn tuyệt đối của tệp `.txt` (tối đa 20 MB) để hiệu chỉnh, thay cho văn bản gpupool đi kèm |
| Calibration chunks | Số đoạn 512 token của văn bản được xử lý; `0` = mặc định (100). Ít hơn thì nhanh hơn |
| Validate generation | Nạp file trên CPU và sinh vài token (bật mặc định). Kiểm tra header và tokenizer luôn chạy |
| Allow remote code | Tải các file `*.py` của chính repo và chạy chúng trong coordinator. Xem mục Bảo mật bên dưới |
| Threads | Số luồng CPU khi lượng tử hoá (`0` = `GPUPOOL_CONVERT_THREADS`; nếu biến đó cũng `0` = mọi CPU) |

Với `F16`, `BF16` và `Q8_0`, bộ chuyển đổi ghi thẳng file cuối, nên các tuỳ chọn lượng tử hoá không áp dụng (UI làm
mờ chúng).

### Kiểm tra làm gì, và `needs_review`

Trước khi file vào thư viện, gpupool kiểm tra nó:

1. **Header GGUF**: đọc được, có kiến trúc và tokenizer. Nếu bước này lỗi, job ở trạng thái *failed* và không có gì
   vào thư viện.
2. **Tokenizer**: 8 đoạn văn cố định (tiếng Anh, tiếng Việt, code, số, emoji gồm cả cụm gia đình nối bằng ZWJ, khoảng
   trắng lạ, dòng trống, tiếng Nhật/Trung) được Hugging Face và `llama-tokenize` (trên file GGUF) tách token. Các id
   token phải trùng khớp hoàn toàn.
3. **Sinh văn bản** (trừ khi tắt): model viết tiếp "The capital of France is" 16 token trên CPU. Bước này bị bỏ qua,
   kèm cảnh báo, khi RAM trống thấp hơn 1.2 lần dung lượng file; hết thời gian chờ cũng chỉ là cảnh báo.

Tokenizer lệch hoặc sinh văn bản lỗi sẽ đưa job vào **needs_review**: file có tồn tại nhưng **chưa** nằm trong thư
viện. Mở bảng kiểm tra để xem những đoạn nào khác nhau. **Accept anyway** (`POST /api/convert/{id}/accept`) thêm file
vào thư viện; xoá job thì bỏ file. Một bước kiểm tra không chạy được (ví dụ tokenizer của Hugging Face không nạp
được) chỉ là cảnh báo và không chặn job.

### Đĩa, RAM và thời gian

- **Đĩa.** Đỉnh khoảng *nguồn + bản trung gian 16 bit + đầu ra*. Ước chừng, model 7 tỷ cần khoảng 14 GB tải về,
  14 GB bản trung gian và khoảng 4 GB đầu ra ở `Q4_K_M`. gpupool kiểm tra dung lượng trống trước khi tải và trước mỗi
  giai đoạn nặng (cộng biên 512 MB); nếu không đủ, job lỗi kèm thông báo cần bao nhiêu và đang trống bao nhiêu. Lúc
  gửi yêu cầu cũng có một **lần kiểm tra đầu**: khi đĩa rõ ràng không chứa nổi *bản tải chưa có trong cache + bản
  trung gian 16 bit + đầu ra*, yêu cầu bị từ chối ngay với HTTP 507 (UI hiện thông báo) thay vì lỗi sau vài phút. Bản
  trung gian bị xoá ngay khi có file đã lượng tử hoá. `F16`, `BF16` và `Q8_0` không có bản trung gian riêng.
- **RAM.** Bộ chuyển đổi nạp trọng số qua PyTorch, nên hãy chuẩn bị RAM trống cỡ dung lượng model 16 bit. Đo được,
  đỉnh bộ nhớ resident của cả một job: khoảng **1,0 GB** (1045 MiB) với Qwen2.5-0.5B và khoảng **1,7 GB** (1697 MiB)
  với Qwen2.5-1.5B; con số tăng theo kích thước model, nên model 7 tỷ cần gấp nhiều lần. Đỉnh cgroup của container
  cao hơn vì tính cả page cache (3,0 và 3,4 GB trong các lần chạy đó); phần đó thu hồi được. Bước thử sinh văn bản cần RAM trống bằng 1.2 lần file đầu ra, nếu không sẽ bị bỏ qua.
- **Thời gian.** Đo trên CPU, đã gồm thời gian tải: SmolLM2-135M-Instruct sang `Q4_K_M` mất khoảng 70 giây từ đầu
  đến cuối trong CI; Qwen2.5-0.5B-Instruct sang `Q4_K_M` mất 143 giây (tải 107 giây, convert 12 giây, quantize
  4 giây, validate 20 giây). Khi có importance matrix thì calibrate chiếm phần lớn: Qwen2.5-1.5B sang `IQ3_M` mất
  1687 giây (tải 314 giây, convert 55 giây, calibrate 1175 giây, quantize 123 giây, validate 21 giây), xem mục
  *Importance matrix*. Model lớn hơn tăng gần tỉ lệ với kích thước và tốc độ tải.
- **Khi job lỗi.** Job ghi lại `failed_stage`, giai đoạn đang chạy lúc nó lỗi hoặc bị huỷ (ví dụ `calibrating`), nên
  bạn phân biệt ngay được lỗi tải với lỗi đĩa hay lượng tử hoá; UI đánh dấu bước đó trên thanh các bước.

### File nằm ở đâu

| Đường dẫn (trong `GPUPOOL_MODELS_DIR`) | Nội dung |
| --- | --- |
| `<name>.gguf` | model hoàn chỉnh, đã đăng ký trong thư viện (nguồn `convert`) |
| `.hf/<owner>__<name>@<revision>/` | cache tải về. Được dùng lại khi retry và bởi các job khác của cùng model (chuyển sang loại lượng tử hoá khác về sau sẽ bỏ qua bước tải nếu đã bật **Keep downloaded source**); nếu không thì bị xoá khi job kết thúc (job lỗi giữ lại cho lần retry cho đến khi job bị xoá) |
| `.convert/<job id>/` | chỗ làm việc tạm của một job (liên kết tới file nguồn, bản trung gian, đầu ra). Bị xoá khi job kết thúc; job *needs_review* chỉ giữ lại file đầu ra ở đây |

gpupool chỉ xoá bên trong `.convert` và `.hf`. Thư mục bạn đưa vào làm nguồn chỉ được đọc, không bao giờ bị sửa.
Job sống sót qua lần khởi động lại coordinator: job bị ngắt quay lại hàng đợi và chạy lại từ đầu (bản tải đã xong không
tải lại).

### Bảo mật

Bộ chuyển đổi nạp tokenizer với `trust_remote_code`, nên file Python của chính repo sẽ chạy bên trong coordinator.
Vì vậy chúng **không bao giờ được tải hay đưa vào thư mục làm việc** trừ khi bạn bật **Allow remote code**. Bộ chuyển
đổi cũng chạy ngoại tuyến (`HF_HUB_OFFLINE=1`) và chỉ thấy một thư mục chứa liên kết tới các file trọng số, config và
tokenizer đã chọn. Chỉ bật tuỳ chọn này cho repo bạn tin cậy, và chỉ khi chuyển đổi lỗi nếu không có nó (tokenizer
tuỳ biến). Tên file trong danh sách repo có thể thoát ra ngoài thư mục tải sẽ bị bỏ qua. Repo Mistral còn kèm các
bản `consolidated.*` của cùng trọng số; chúng bị bỏ qua.

Repo gated hoặc riêng tư cần `HF_TOKEN` trên coordinator (và đã chấp nhận giấy phép trên huggingface.co), nếu không
inspect và tải về bị từ chối với HTTP 403.

### Build image không kèm bộ công cụ, hoặc sau mirror

Image coordinator mặc định có sẵn bộ công cụ chuyển đổi (`WITH_CONVERT=1`): bộ chuyển đổi của llama.cpp, một venv
PyTorch chỉ dùng CPU, và bản build CPU tĩnh của `llama-quantize`, `llama-tokenize`, `llama-simple` và
`llama-imatrix`. Đo được (trước khi thêm `llama-imatrix`):
**1.77 GB** có bộ công cụ so với **560 MB** không có. Coordinator không bao giờ chuyển đổi có thể dùng image gọn:

```bash
docker build --build-arg WITH_CONVERT=0 -f docker/coordinator.Dockerfile -t gpupool-coordinator .
# với compose: WITH_CONVERT=0 docker compose -f docker-compose.coordinator.yml build
```

Không có bộ công cụ thì mọi thứ khác vẫn chạy; inspect vẫn mô tả được model nhưng không biết kiến trúc có được hỗ
trợ không, và bắt đầu job trả về HTTP 503 kèm lời giải thích (UI hiện thành một thông báo).
Riêng `llama-imatrix` là tuỳ chọn: coordinator thiếu nó vẫn chuyển đổi bình thường mọi loại không cần ma trận. Nơi `download.pytorch.org`
bị chặn, hãy build với mirror của chỉ mục wheel PyTorch CPU (`--build-arg TORCH_INDEX_URL=https://mirror.corp/pytorch/whl/cpu`;
`docker-compose.coordinator.yml` đọc `TORCH_INDEX_URL` từ môi trường). Mã nguồn llama.cpp lấy từ
`vendor/llama.cpp-b11413.tar.gz` nếu có, giống image agent, xem [Build không cần GitHub](#build-không-cần-github).
Ngoài Docker, đặt `GPUPOOL_CONVERT_DIR`, `GPUPOOL_CONVERT_PYTHON` và `GPUPOOL_LLAMA_TOOLS_DIR` (xem Thiết lập).

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
python scripts/ci_e2e.py                       # tuỳ chọn: --project gpupool-ci --port 8080 --keep --skip-convert
```

Nó thực hiện 33 phép kiểm tra: ba agent tự join, thư viện model, các API, start đến *running*, một chat completion,
`/metrics`, và stop (không còn engine nào trên mọi agent). Các giai đoạn cuối chạy ba lần chuyển đổi ngay trong
coordinator: `HuggingFaceTB/SmolLM2-135M-Instruct` sang `Q4_K_M` (khoảng 100 MB; file đã chuyển sau đó được phục vụ,
chia qua RPC, và trả lời một chat), một nguồn là **thư mục** (các file của model được tải vào một thư mục, chuyển
sang `Q8_0`, thư mục nguồn không bị đổi) và `IQ2_XS`, loại cần **importance matrix** (4 chunk hiệu chỉnh để chạy
ngắn; job đi qua *converting*, *calibrating*, *quantizing*, *validating*). Chúng cần image coordinator có bộ công cụ (`WITH_CONVERT=1`, mặc định) và truy cập được
huggingface.co (đặt `HF_TOKEN` để tránh giới hạn tốc độ cho người dùng ẩn danh). `--skip-convert` bỏ giai đoạn này.
Mã thoát 0 nghĩa là qua hết; khi lỗi nó in `docker compose logs` và
thoát với mã 1. Cụm được xoá sau khi chạy, trừ khi có `--keep`. Model được cache trong `GPUPOOL_CI_MODELS_DIR`
(mặc định `<repo>/.cache/ci-models`).

## Tùy chọn hiệu năng (theo từng model, trong form deploy)

| Tùy chọn | Giá trị | Tác dụng | Đo thật (GTX 1650, Qwen2.5-3B) |
| --- | --- | --- | --- |
| KV cache | f16, q8_0, q4_0 | KV cache nhỏ hơn nên model có thể vừa ít GPU hơn | ctx 8192: −132 / −204 MB, tốc độ gần như không đổi (51.9 / 51.2 / 50.8 tok/s) |
| Speculative | none, ngram, draft, mtp | model lớn chạy ít lượt hơn cho mỗi token, tức ít vòng RPC hơn khi bị chia | chia qua 2 server: none 48.9, ngram 53.6, draft 0.5B 53.9 tok/s |
| Dùng chung context giữa các slot | tắt, bật | khi có nhiều slot song song, một request có thể dùng cả context thay vì context ÷ số slot; cùng lượng bộ nhớ | chưa đo |

Trong API các tuỳ chọn này là `kv_cache_type` (`f16`, `q8_0`, `q4_0`), `speculative` (`none`, `ngram`, `draft`, `mtp`: chỉ cho GGUF có layer dự đoán nhiều token),
`draft_file` (một model trong thư viện, cho `draft`), `draft_n_max` (1 đến 16, mặc định 4) và `kv_unified` (dùng chung context); xem
[API.vi.md](API.vi.md). Chúng có hiệu lực ở lần chạy model kế tiếp.

N-gram không tốn thêm bộ nhớ nhưng chỉ có lợi khi câu trả lời lặp lại phần văn bản trước đó. Model draft phải
dùng chung tokenizer với model chính (được kiểm khi lưu) và chạy trên GPU của head (bộ nhớ đã được tính vào kế
hoạch). Mặc định đoán 4 token; 8 token chậm hơn trong các lần đo. MTP dùng chính các layer dự đoán nhiều token
của model (Qwen3.5, GLM-4.5 trở lên, DeepSeek V3...): không cần file thứ hai, không phải khớp tokenizer, và bộ nhớ của
các layer đó chỉ được tính khi bật MTP. Lưu `mtp` cho model không có các layer này sẽ bị từ chối. Với MTP, nên bắt
đầu với 2 hoặc 3 token nháp.

## Không dùng Docker

Coordinator:

```bash
git clone https://github.com/longduongbao29/gpupool && cd gpupool
uv sync && uv run gpupool coordinator          # in ra cùng admin key và lệnh join
```

Server GPU (cần [uv](https://docs.astral.sh/uv/) và bản build llama.cpp có CUDA và RPC, b11413):

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

Model bị chia gửi activation giữa các server ở mỗi token, nên độ trễ mạng quyết định tốc độ của chúng. Hai thứ
giúp được:

- **Nhiều GPU trong một server.** Replica dùng từ hai GPU trở lên của cùng một server ở xa chỉ cần một
  `ggml-rpc-server` cho chúng, server này copy activation giữa các GPU ngay tại chỗ (agent bản này; agent cũ chạy
  mỗi GPU một cái).
- **RDMA (InfiniBand / RoCE).** llama.cpp trong image agent nói được RDMA và dùng nó khi cả hai đầu có thiết bị
  RDMA, nếu không thì quay về TCP. Cấp cho container agent thiết bị và bộ nhớ khoá:

  ```bash
  docker run -d --name gpupool-agent --gpus all --network host --pid host \
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 \
    -v gpupool-agent:/data -e GPUPOOL_JOIN=... ghcr.io/longduongbao29/gpupool-agent
  ```

  `GGML_RPC_NO_RDMA=1` (`-e GGML_RPC_NO_RDMA=1`) ép dùng TCP. RDMA được thương lượng theo từng kết nối: server không
  có RDMA vẫn làm việc với các server khác qua TCP.

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
| Bước build `git clone` treo hoặc `Failed to connect to github.com` | GitHub bị chặn kể cả qua proxy | đặt `vendor/llama.cpp-b11413.tar.gz` vào repo, hoặc đặt `LLAMA_CPP_URL` |
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
   curl -L -o vendor/llama.cpp-b11413.tar.gz https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11413.tar.gz
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

### Coordinator (26 thiết lập)

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
| `GPUPOOL_CONVERT_DIR` | không có (image: `/opt/llama.cpp`) | thư mục chứa `convert_hf_to_gguf.py`, `conversion/` và `gguf-py/` của llama.cpp; không đặt = không chuyển đổi được (job nhận HTTP 503) |
| `GPUPOOL_CONVERT_PYTHON` | `python3` (image: `/opt/convert-venv/bin/python`) | trình thông dịch có các thư viện của bộ chuyển đổi (PyTorch, transformers) |
| `GPUPOOL_LLAMA_TOOLS_DIR` | không có (image: `/opt/llama/bin`) | thư mục chứa bản build CPU của `llama-quantize`, `llama-tokenize` và `llama-simple` |
| `GPUPOOL_CONVERT_THREADS` | `0` | số luồng cho `llama-quantize`; `0` = mọi CPU (tuỳ chọn *Threads* của từng job thắng) |

Image Docker đặt sẵn `GPUPOOL_HOST`, `GPUPOOL_PORT`, `GPUPOOL_DB_PATH`, `GPUPOOL_MODELS_DIR` và
`GPUPOOL_MODEL_ROOTS` như trên, cộng thêm `GPUPOOL_CONVERT_DIR`, `GPUPOOL_CONVERT_PYTHON` và
`GPUPOOL_LLAMA_TOOLS_DIR` khi được build kèm bộ công cụ chuyển đổi (mặc định). `docker-compose.coordinator.yml` còn
đọc `GPUPOOL_HOST_MODELS_DIR` (thư mục trên máy chủ được mount vào `/models`), các biến proxy, và các thiết lập build
`WITH_CONVERT` và `TORCH_INDEX_URL` (xem [Serve model chưa có GGUF](#serve-model-chưa-có-gguf-chuyển-đổi)).

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
| `GPUPOOL_RPC_CACHE_GB` | `100` | giới hạn cache trọng số RPC (`<GPUPOOL_CACHE_DIR>/llama.cpp/rpc`, giúp nạp lại model mà không phải gửi trọng số qua mạng); file ít dùng nhất bị xoá trước, `0` = không giới hạn |

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
| Chuyển đổi: inspect hoặc job báo kiến trúc không được hỗ trợ ("not supported by the pinned llama.cpp converter", HTTP 422) | họ model mới hơn bộ chuyển đổi được ghim, hoặc không phải model văn bản. Tìm bản GGUF có sẵn của nó (inspect có liệt kê), hoặc chờ bản gpupool dùng llama.cpp mới hơn |
| Chuyển đổi: HTTP 503 "conversion is not set up ..." | image được build với `WITH_CONVERT=0`, hoặc ngoài Docker thì `GPUPOOL_CONVERT_DIR` / `GPUPOOL_CONVERT_PYTHON` / `GPUPOOL_LLAMA_TOOLS_DIR` thiếu hoặc sai. Dùng image đầy đủ hoặc sửa đường dẫn; thông báo nêu rõ thiếu gì |
| Chuyển đổi: HTTP 507 "not enough free disk space ..." ngay khi gửi job | đĩa rõ ràng không chứa nổi *bản tải chưa có trong cache + bản trung gian 16 bit + đầu ra* (xem *Đĩa, RAM và thời gian*). Chưa có gì được xếp hàng. Giải phóng dung lượng, hoặc trỏ `GPUPOOL_MODELS_DIR` sang đĩa lớn hơn, rồi gửi lại |
| Chuyển đổi: job *failed* với "not enough free disk space ..." | lần kiểm tra đầu qua được nhưng một giai đoạn sau thấy ít chỗ hơn (job hoặc tiến trình khác đã dùng đĩa). `failed_stage` của job cho biết ở đâu. Giải phóng dung lượng, rồi **Retry** (bản tải đã xong được dùng lại) |
| Chuyển đổi: HTTP 422 "`<loại>` needs an importance matrix ..." | loại đó thuộc nhóm IQ1, IQ2, `IQ3_XXS`, `IQ3_XS` mà tuỳ chọn đang là *Off*. Đặt *Auto* hoặc *On*, hoặc chọn loại lớn hơn |
| Chuyển đổi: HTTP 503 nhắc tới `llama-imatrix` hoặc văn bản hiệu chỉnh có sẵn | bản cài không có `llama-imatrix` (dùng image coordinator đầy đủ), hoặc thiếu văn bản đi kèm (cài lại, hoặc đưa vào một văn bản hiệu chỉnh). Chọn loại không cần ma trận, hoặc *Off* / *Auto* với loại chỉ hưởng lợi từ ma trận |
| Chuyển đổi: job có importance matrix như đứng yên ở *calibrating* | nó đang chạy: calibrate lâu nhất trên CPU (khoảng 20 phút cho model 1,5 tỷ, xem *Importance matrix*). Thanh tiến độ chỉ là ước lượng. Nếu quá chậm, huỷ rồi thử lại với ít *Calibration chunks* hơn |
| Chuyển đổi: job ở *needs_review* ("the tokenizer differs from Hugging Face on N of 8 test texts") | GGUF tách token một số đoạn văn khác với bản gốc, nên model có thể chạy sai ở các đoạn đó. Mở bảng kiểm tra để xem đoạn nào. Chỉ chấp nhận nếu chịu được rủi ro; nếu không, xoá job và dùng bản GGUF có sẵn |
| Chuyển đổi: HTTP 403 "repo is gated or private" | đặt `HF_TOKEN` trên coordinator và chấp nhận giấy phép trên trang model, rồi thử lại |
| Model kẹt ở *failed*: "not enough VRAM" | giải phóng GPU, bật thêm GPU, thêm server, hoặc dùng bản quantize nhỏ hơn |
| `could not select device driver "" with capabilities: [[gpu]]` | server đó chưa cài NVIDIA Container Toolkit |
| Log agent: `RPC firewall unavailable (...); RPC ports are NOT restricted` | đã đặt `GPUPOOL_RPC_FIREWALL=1` nhưng thiếu iptables hoặc container không có root/`NET_ADMIN`: thêm `--cap-add NET_ADMIN` (compose `cap_add: [NET_ADMIN]`) rồi tạo lại container, hoặc bỏ biến đó và tự đặt tường lửa cho 9000–9999 |
| Log agent: `RPC firewall: ... rule for port N failed ... port left unrestricted` | một lệnh iptables lỗi và luật đã được hoàn tác; đọc nội dung lỗi (thường là thiếu cùng quyền đó). `ip6tables unavailable` chỉ có nghĩa là IPv6 chưa được giới hạn |
| Model nhiều server kẹt ở *starting* khi bật firewall, engine trên server khác không bao giờ trả lời | peer bị từ chối. `RPC firewall: cannot resolve peer ...` nghĩa là một tên không phân giải được trên server đó: dùng IP (`GPUPOOL_HOST`) hoặc sửa DNS. Để xác nhận, bỏ `GPUPOOL_RPC_FIREWALL` trên server đó rồi tạo lại container |
| Model đang *launching* thì coordinator bị kill | được đánh dấu *failed* ("coordinator restarted during launch") và tự lập kế hoạch lại trong một nhịp reconcile |
| Windows / WSL2: mọi container khởi động lại khoảng một phút sau khi đóng terminal cuối cùng | WSL tắt VM khoảng 1 phút sau phiên `wsl.exe` cuối cùng, kể cả khi Docker đang chạy. Trong `%UserProfile%\.wslconfig` thêm `vmIdleTimeout=-1` dưới `[wsl2]`, rồi chạy `wsl --shutdown` và bật lại Docker |
| Windows / WSL2: model lớn load lỗi hoặc engine bị kill (hết bộ nhớ) | WSL2 mặc định giới hạn bộ nhớ (ở đây là 4 GB). Tăng `memory=` dưới `[wsl2]` trong `%UserProfile%\.wslconfig` (ví dụ `memory=16GB`), rồi `wsl --shutdown` |
