# Nhật ký thay đổi

> Tiếng Việt. English: [CHANGELOG.md](CHANGELOG.md)

Mọi thay đổi đáng chú ý của gpupool được ghi ở đây. Định dạng theo
[Keep a Changelog](https://keepachangelog.com/vi/1.1.0/) và dự án dùng
[Semantic Versioning](https://semver.org/lang/vi/) (bản 0.x: phiên bản phụ có thể thay đổi hành vi).

## [Chưa phát hành]

## [0.8.0] - 2026-10-07

### Thêm

- Cài đặt suy nghĩ cho model có thinking (Qwen3, DeepSeek-R1, gpt-oss...): **Thinking** auto / on / off, **Effort**
  (minimal ... max, chuyển cho chat template) và **giới hạn token suy nghĩ**, trong form model và API model
  (`reasoning`, `reasoning_effort`, `reasoning_budget`; llama-server `-rea`, `--reasoning-effort`,
  `--reasoning-budget`). Đây là giá trị mặc định; mỗi request vẫn tự đặt được. Chỉ áp dụng trên head có llama.cpp hỗ
  trợ các cờ này (b11413 trở lên, tính năng agent `reasoning`); head khác chạy với mặc định của template kèm cảnh
  báo `reasoning_unavailable`, thay vì không khởi động được.
- Playground: **Thinking** (mặc định của model / on / off) và **Effort** cho từng cuộc chat, gửi dưới dạng
  `reasoning_effort` và `chat_template_kwargs.enable_thinking`.
- Playground: câu trả lời và phần suy nghĩ hiển thị dạng Markdown (tiêu đề, danh sách, khối code, bảng, trích dẫn,
  link mở ở tab mới), được làm sạch bằng DOMPurify nên câu trả lời không thể chạy script. Đóng gói sẵn marked 15.0.12
  và DOMPurify 3.2.6 (xem THIRD_PARTY_NOTICES.vi.md).

### Sửa lỗi

- Model không khởi động được không còn tạo thông báo ở mỗi lần thử lại (mỗi một vài phút khi thời gian chờ tăng
  dần tới 5 phút). `launch_failed`, `engine_crashed` và `crash_loop` cho cùng một lỗi chỉ thông báo tối đa mỗi 10
  phút, kèm số lần thử lại đã lỗi y như vậy; lỗi khác, hoặc lỗi đầu tiên sau khi model đã chạy lại được, thông báo
  ngay. `launch_failed` còn cho biết khi nào thử lại.

## [0.7.3] - 2026-10-07

### Sửa lỗi

- Một port trong `GPUPOOL_PORT_RANGE` bị tiến trình ngoài gpupool chiếm không còn chặn model mãi mãi: coordinator
  chọn lại đúng port đó ở mỗi lần thử. Port bị agent từ chối giờ được bỏ qua trên máy đó trong một giờ, launch được
  thử lại ngay với port khác, và cảnh báo `port_busy` nêu rõ port đó.

### Thay đổi

- Launch đang nạp weights không còn bị dừng ở `launch_timeout_s`: khi llama-server còn báo tiến độ nạp, thời gian chờ
  tính từ lần tiến độ gần nhất. Trước đây lần nạp đầu gửi nhiều GB sang GPU máy khác qua mạng chậm bị lỗi ở phút
  thứ 10 và phải bắt đầu lại.
- Bước launch hiện phần trăm đã nạp (các dấu chấm tiến độ của llama.cpp) và bỏ tiền tố thời gian trong log llama.cpp.

## [0.7.2] - 2026-10-07

### Thêm

- Replica đang launch hiển thị nó đang làm gì và đã bao lâu, ở trang Models và trong `replicas[].stage`: tải file
  model về head (phần, số GB đã tải trên tổng, phần trăm), khởi động RPC server, rồi nạp model kèm dòng log mới nhất
  của llama-server. Lần launch đầu của model lớn (tải 50 GB, rồi gửi weights sang GPU ở máy khác) không còn trông như
  bị treo. API agent: `GET /models/progress`.

## [0.7.1] - 2026-10-07

### Sửa lỗi

- Placement: một GPU đầy tới mức không chứa nổi một layer (vd. còn 250 MB) không còn làm mọi cách chia trên node của
  nó bất khả thi. Placement nhiều node lấy nguyên node và mỗi thiết bị trong cách chia phải nhận ít nhất một layer, nên
  một GPU như vậy cạnh một GPU còn rộng gây NoFit dù pool còn dư chỗ; giờ nó được bỏ ra khỏi cách chia.

## [0.7.0] - 2026-10-06

### Thêm

- UI: diện mạo mới. Icon vẽ riêng cho gpupool (duotone; gradient và quầng sáng ở thanh điều hướng, ô thống kê
  và Playground) và logo isometric; chuyển trang mờ dần có nhòe trong khi các khối của trang mới lần lượt nổi lên (View
  Transitions API), vệt sáng điều hướng trượt giữa các mục, hộp thoại và menu bật ra có độ nảy, thẻ nhấc lên khi rê
  chuột, nút chính dạng gradient và trang đăng nhập có aurora. Tất cả tắt khi bật "giảm chuyển động".
- UI: xóa được sự kiện (*Clear all* ở trang Events, *Clear* trong menu thông báo); API
  `DELETE /api/events?up_to_id=`.
- UI: hộp xác nhận của UI thay cho `confirm()` của trình duyệt ở mọi nơi (nút mang tên hành động, màu đỏ khi xóa,
  Esc / Enter).

### Thay đổi

- Thông báo `NoFit` liệt kê MB usable của từng device.
- Form model: *Save* và *Save & Start* nằm bên phải.
- Form model: các bước kiểm tra chuyển sang một cột bên phải gồm các mục thu gọn được, mặc định đóng. Mở *Recommend
  placement & settings* hoặc *Preview impact* là chạy luôn, và mục đang mở tự chạy lại khi form thay đổi (có debounce).
  *Check placement* vẫn chỉ chạy khi bấm, vì nó lưu model trước.
- Thẻ model: *Copy curl* nằm ở góc trên bên phải khung endpoint, để hai dòng kết thúc ở cùng một cột icon copy.

### Sửa lỗi

- Placement: model lớn có thể báo "no feasible layer split" dù pool còn chỗ (ví dụ model 27B BF16, 52 GB, trên 66 GB
  của 5 GPU). Khi head có một GPU nhỏ, GPU đó đứng cuối và phải giữ thêm tensor đầu ra (2,5 GB với từ vựng 248k). Khi
  thứ tự mặc định không có cách chia khả thi, planner giờ thử GPU lớn nhất của head ở cuối, rồi bất kỳ node nào làm
  đuôi; placement vốn đã vừa không đổi.
- Model không vừa bị lập kế hoạch lại ở mỗi tick 2 s, và sự kiện "cannot place" của nó lặp lại mỗi lần (thông báo chứa
  bộ nhớ trống, vốn luôn đổi), làm ngập danh sách sự kiện. Giờ nó được lập kế hoạch lại khi dung lượng thay đổi hoặc
  sau 60 s, và sự kiện lặp lại tối đa 10 phút một lần. Lỗi lập kế hoạch khác (header GGUF không đọc được) dùng backoff
  thay vì tải header ở mỗi tick. Start hoặc Save thử lại ngay.
- Agent không bao giờ xóa log engine: mỗi lần launch ghi log mới, nên model liên tục lỗi làm đầy thư mục log. Giờ agent
  giữ 200 log mới nhất của engine đã dừng và xóa log quá 7 ngày.
- UI: GPU đã tắt trong pool vẫn chọn được cho model; tên server trong bộ chọn bị các dòng mờ vẽ đè khi cuộn.
- Playground: tên model đã xóa vẫn nằm trong lựa chọn.
- UI: menu thông báo bị trang vẽ đè (header bảng dính, nút) từ khi header thành phần tử View Transitions; giờ header
  được xếp lớp trên trang.

## [0.6.0] - 2026-10-05

### Thêm

- UI: các khung lớn (Servers, GPUs, Model library, Conversions, Placement health, Deployments, Events, các mục
  Settings) thu gọn được còn phần đầu bằng mũi tên hoặc bấm vào tiêu đề. Mặc định đều mở; khung đã thu gọn được nhớ theo
  trình duyệt.

- **Playground** trong UI: chat với model đã deploy qua `/v1/chat/completions` (router, balancer và replica, đúng như
  client). Câu trả lời stream từng ký tự, kèm dải số đo trực tiếp: thời gian tới token đầu, token/giây khi sinh, số
  token và tổng độ trễ; mỗi câu trả lời còn hiện số token và tốc độ xử lý prompt (prefill), tốc độ placement ước tính đặt cạnh tốc độ đo được,
  và replica đã trả lời.
  Tốc độ lấy từ `timings` của chính llama-server khi xong (đếm trong trình duyệt khi đang stream), `reasoning_content`
  của model có suy nghĩ vào một khối thu gọn được, *Stop* để huỷ, model on-demand đang ngủ được nạp (cold start) ở tin
  nhắn đầu tiên. Khung chat bên trái, cài đặt bên phải, *Clear chat* ở đầu khung chat. Nút *Chat* trên thẻ model đang chạy mở Playground. Phần cài đặt (không phải nội dung chat) được nhớ
  theo trình duyệt.

- `/v1` nhận thêm admin key khi đã đặt API key (Playground đăng nhập bằng khóa này); `/v1` đang mở (không có API key)
  vẫn mở. Mọi phản hồi được chuyển tiếp có header `x-gpupool-replica`, tức replica đã trả lời.

- UI: thẻ server hiện bản llama.cpp của nó và đánh dấu server có bản khác với bản phổ biến nhất; *Placement health*
  hiện mô hình tốc độ đã học (phần băng thông đỉnh đạt được, thời gian mỗi hop mạng); *Why here* tách thời gian của
  một token thành đọc weight, các hop mạng và logits.

- Mô hình tốc độ decode tự học từ tốc độ đo được: eta (tỉ lệ băng thông đỉnh) từ replica trên một server và thời gian
  mỗi hop RPC từ replica bị chia, đều lấy từ tốc độ sinh token đo được của llama-server ở các replica chạy một luồng
  thuần. Được lưu lại, hiển thị là `speed_model` trong `GET /api/state`; placement khi đó xếp hạng theo mạng và GPU thật
  của cụm thay vì hằng số đo trên một GTX 1650.

- Sự kiện cảnh báo `llama_version_mismatch` (kèm webhook) khi các server đã đăng ký báo bản llama.cpp khác nhau, một
  lần cho mỗi thay đổi của tập các bản build (server chập chờn không làm nó lặp lại): model bị chia qua nhiều server cần cùng giao thức RPC trên head và mọi RPC server.

- `kv_unified` (llama.cpp `-kvu`): các slot song song dùng chung một vùng KV, nên một request dài có thể dùng cả
  context trong khi các slot khác giữ request ngắn, với cùng lượng bộ nhớ. Có trường API, công tắc trong form
  triển khai và gợi ý Recommend cho model có nhiều slot; ước lượng tính layer sliding-window theo vùng dùng chung. Agent
  cũ hơn bỏ qua cờ này (khi đó mỗi slot vẫn giữ phần context của riêng nó).

- Speculative decoding `mtp`: GGUF có sẵn layer dự đoán nhiều token (nextn) (Qwen3.5, GLM-4.5 trở lên,
  DeepSeek V3...) nháp bằng chính các layer đó qua `--spec-type draft-mtp` của llama.cpp, không cần file model
  phụ. Model đích chạy ít lượt hơn cho mỗi token nên ít vòng RPC hơn khi model bị chia. API từ chối `mtp` với
  model không có các layer này, ước lượng tính cả các layer (chỉ được nạp ở chế độ này) cùng cache của chúng,
  và bảng Recommend gợi ý nó trước n-gram và model draft. Head có agent không báo tính năng `spec_mtp` (agent hoặc bản llama.cpp cũ hơn)
  phục vụ model mà không speculative và phát cảnh báo `mtp_unavailable` thay vì làm launch thất bại.

### Thay đổi

- Xếp chỗ nhanh hơn khoảng 10 lần trên cụm lớn (8 máy, mỗi máy 5 thiết bị: 3,0 s → 0,27 s; 4 máy: 256 → 29 ms),
  cho ra đúng các phương án như trước (đã đối chiếu trên 400 cụm ngẫu nhiên). Nhu cầu bộ nhớ của mỗi thiết bị là
  một phép trừ trên tổng cộng dồn theo layer thay vì vòng lặp qua từng layer; dời một layer giữa hai thiết bị chỉ
  kiểm tra lại các thiết bị có dải layer bị dịch; bước ưu tiên GPU nhanh tiếp tục dời trên cùng một cặp thiết bị
  khi còn có lợi thay vì quét lại mọi cặp sau mỗi layer. Các lượt xếp hạng của Recommend, mô phỏng và chấm điểm
  rebalance chạy trong thread riêng nên không còn làm khựng các phản hồi đang stream.

- Speculative decoding kiểu draft và MTP lấy mẫu bản nháp và kiểm bằng rejection (`--spec-draft-sampling
  probabilistic`, llama.cpp b11413+): cùng phân phối đầu ra, nhiều bản nháp được chấp nhận hơn khi temperature > 0
  (+4-8 % throughput theo số đo của llama.cpp). Agent dùng bản llama.cpp cũ hơn vẫn nháp kiểu greedy.

- llama.cpp b11342 → b11413 trong cả hai image (cùng giao thức RPC 7.0.0; vẫn nâng mọi agent cùng lúc như trước).
  Mang lại: draft n-gram không còn bị từ chối khi temperature > 0, lấy mẫu draft theo xác suất cho draft và MTP, sửa
  lỗi bộ nhớ CUDA với MoE nhiều expert, gộp shared expert và matmul f16/bf16 batch nhỏ nhanh hơn trên CUDA, sửa flash
  attention trên Volta, và `llama-imatrix --nextn`, mà gpupool giờ truyền cho model có layer MTP để các loại lượng tử
  cần importance matrix lượng tử hoá được chúng.

- Router chia request giữa các replica của một model theo tốc độ của chúng: weighted rendezvous hashing với trọng
  số là tok/s decode ước lượng của placement (được mô hình tốc độ tự học giữ cho sát), và phép kiểm tra quá tải tính số
  request theo năng lực. Replica bị chia
  qua mạng chạy 10 tok/s không còn nhận phần bằng replica một GPU chạy 50 tok/s. Replica cùng tốc độ định tuyến y
  như trước.

- Placement bị chia đặt các GPU của chính head ở cuối thứ tự device. Device cuối giữ layer đầu ra, và llama-server
  đọc logits (`n_vocab x 4` byte, khoảng 0,5 MB với từ vựng 128k) từ nó ở mỗi token: khi device cuối ở xa, logits đi
  qua mạng mỗi lần, giờ chỉ còn hidden state (`n_embd x 4` byte). Model draft vẫn nằm trên GPU local đầu tiên của head.

- Cache trọng số RPC có giới hạn dung lượng (`GPUPOOL_RPC_CACHE_GB`, mặc định 100): cứ 10 phút agent xoá các file
  tensor ít dùng nhất khi vượt giới hạn. llama.cpp không bao giờ xoá chúng, nên mọi model từng chia sang một máy
  chủ đều nằm lại trên đĩa máy đó.
- Router chuyển tiếp nguyên body của request (thêm `"cache_prompt": true` khi thiếu) thay vì parse rồi serialize
  lại, việc tốn vài mili giây event loop cho mỗi request context dài.

- Image agent build kèm transport RDMA của llama.cpp (RoCE / InfiniBand, qua libibverbs). Mỗi kết nối RPC tự
  thương lượng và quay về TCP ở nơi một trong hai bên không có thiết bị RDMA; để dùng, chạy agent với
  `--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1`.
- Công cụ chuyển đổi trong image coordinator (llama-quantize, llama-imatrix) được build với AVX2/FMA/F16C ghi rõ,
  thay vì dựa vào mặc định CMake mà môi trường build có `SOURCE_DATE_EPOCH` sẽ tắt đi.

- Mỗi máy chủ và replica chỉ chạy một `ggml-rpc-server` phục vụ mọi GPU của replica trên máy đó (`-d CUDA0,CUDA1`)
  thay vì mỗi GPU một process. llama.cpp khi đó copy activation giữa các GPU này ngay trong server; với mỗi GPU một
  process, mỗi ranh giới phải đi server → head → server, hai lần truyền mạng cho mỗi token. Placement tính một hop
  mạng cho mỗi RPC server nên các cách chia như vậy cũng được chấm điểm cao hơn. Agent báo khả năng này
  (`features: ["rpc_multi_device"]`); agent cũ vẫn chạy mỗi GPU một server.

- Ước lượng bộ nhớ theo đúng bố cục cache từng layer của llama.cpp. Layer sliding-window (Gemma 2/3/4,
  gpt-oss, Cohere2, OLMo2) chỉ cache cửa sổ của nó, model MLA (DeepSeek, Kimi, GLM-DSA) chỉ cache K latent,
  model lai (Qwen3-Next, Qwen3.5, Nemotron-H, Jamba...) chỉ có KV ở layer attention cộng một state hồi quy nhỏ
  cho mỗi sequence, và các block MTP mà llama.cpp không nạp khi không có `draft-mtp` không còn bị tính. Các model
  này trước đây bị ước lượng dư (tới vài lần KV thật ở context dài), khiến chúng bị chia ra nhiều GPU hoặc máy
  chủ hơn mức cần. Kiến trúc chưa biết giữ quy tắc cũ.
- Tốc độ decode của model MoE chỉ tính các expert mà mỗi token thực sự đọc, nên placement của model MoE được
  chấm điểm theo tốc độ sát thực tế.
- Hệ số hiệu chỉnh VRAM đã lưu được xoá một lần khi bộ ước lượng thay đổi (lần này), vì hệ số học theo ước
  lượng cũ sẽ nhân ước lượng mới với sai số của cái cũ.

- Khởi động nguội nhanh hơn: head tải model (và draft, song song với model) trong lúc các engine RPC khởi
  động, thay vì đợi chúng chạy xong mới tải.
- SQLite của coordinator chạy với `synchronous=NORMAL` (an toàn khi dùng WAL): báo cáo của agent, mỗi máy chủ
  một commit mỗi 2 s trên chính event loop chuyển tiếp suy luận, không còn fsync mỗi lần.

### Sửa lỗi

- UI: ô chọn có option sinh động (model của Playground, file thư viện, draft model) có thể hiện option đầu tiên thay
  vì giá trị nó đang giữ; GPU của server offline và sự kiện đã đọc không được làm mờ (hiệu ứng xuất hiện của dòng đè
  opacity inline); các thẻ model trên một hàng cao khác nhau.

- UI trên điện thoại: các nút ở đầu khung (Placement health) xuống dòng thay vì tràn ra ngoài màn hình.

- Hội thoại nhiều lượt ở yên trên một bản sao. Router lấy khoá theo mọi tin nhắn trừ tin cuối, nên khoá đổi ở
  mỗi lượt (cho tới khi tiền tố quá 4096 ký tự) và khi có nhiều bản sao, hội thoại nhảy sang bản sao phải xử lý
  lại toàn bộ lịch sử. Giờ yêu cầu nhiều lượt lấy khoá theo các tin system cộng tin user đầu tiên, phần mọi lượt
  sau đều lặp lại. Yêu cầu một lượt giữ cách lấy khoá cũ (theo system prompt).
- Router không còn giới hạn 100 kết nối tới upstream. Pool mặc định của httpx giữ yêu cầu thứ 101 trở đi trong
  coordinator mà không có timeout, bộ cân bằng tải và hàng đợi của llama-server đều không thấy.
- Cache trọng số RPC còn lại sau khi khởi động lại hay nâng cấp agent. `ggml-rpc-server -c` lưu tensor nhận được
  ở `$LLAMA_CACHE/rpc`, mặc định là `~/.cache` trong lớp ghi của container; giờ engine nhận
  `LLAMA_CACHE=<GPUPOOL_CACHE_DIR>/llama.cpp` (nằm trên volume `/data` trong image). `LLAMA_CACHE` do người
  dùng đặt được giữ nguyên.

## [0.5.1] - 2026-10-05

### Thay đổi

- Chia layer giữa các GPU theo băng thông chứ không chỉ theo bộ nhớ trống. Sau khi có cách chia vừa, layer
  được dời từ thiết bị chậm sang thiết bị nhanh khi còn vừa và tốc độ decode ước tính tăng (thời gian decode là
  tổng byte / băng thông của từng thiết bị). Ước tính ví dụ, 70B Q4 trên RTX 5090 + RTX 4090: 46/34 layer thành
  57/23, khoảng +9 % token/s. GPU cùng băng thông hoặc chưa rõ băng thông chia như cũ; mỗi thiết bị vẫn giữ ít
  nhất một layer nên số bước RPC không đổi.

## [0.5.0] - 2026-10-05

### Thêm mới

- Thiết lập tốc độ khi chạy, trong form model và `PUT /api/models`: flash attention (`flash_attn`
  auto/on/off, `-fa` của llama.cpp), micro-batch (`ubatch`, `-ub`) và batch (`batch`, `-b`). Đây là các trường
  có kiểu, được kiểm tra, gửi xuống agent, không phải cờ llama-server tùy ý (agent vẫn từ chối `extra_args`).
  Ước lượng bộ nhớ tính compute buffer theo micro-batch đã chọn (và ma trận điểm attention khi tắt flash
  attention), nên micro-batch lớn không làm GPU bị cấp quá. KV cache lượng tử hóa khi tắt flash attention bị từ
  chối (422): llama.cpp không chạy V cache lượng tử hóa nếu thiếu nó.
- Gợi ý thiết lập: "Recommend placement & settings" trong form model (`tips` trong `POST /api/recommend`) liệt
  kê các thay đổi giúp model nhanh hơn hoặc phục vụ nhiều người hơn, mỗi cái đều được kiểm trên pool thật và không
  bao giờ cần nhiều GPU hơn hiện tại, kèm nút Apply: KV cache lượng tử hóa, bản lượng tử hóa nhỏ hơn của cùng
  model hoặc context nhỏ hơn để vừa một GPU thay vì nhiều (các layer trên nhiều GPU chạy nối tiếp nên chia ra
  không bao giờ decode nhanh hơn); số slot song song giữ nguyên context mỗi request; model draft tương thích trong
  thư viện hoặc speculative n-gram; micro-batch lớn hơn cho prompt dài.
- Nhận biết thế hệ GPU: agent báo compute capability của từng GPU (`compute_cap`, hiển thị dạng
  "Ampere · cc 8.6" ở trang GPUs) và các kiến trúc CUDA mà llama.cpp của nó được build (`cuda_archs`, từ
  `cuda-archs.txt` do image agent ghi, hoặc `GPUPOOL_CUDA_ARCHS`). Gợi ý dựa theo đó: micro-batch lớn chỉ được
  gợi ý khi mọi GPU có tensor core (Volta, cc 7.0+), ép bật flash attention hoặc micro-batch lớn trên Pascal sẽ bị
  cảnh báo, và draft đề xuất ít token hơn ở đó.
- GPU mà llama.cpp của agent không có kernel (mọi kiến trúc được build đều mới hơn card) được báo với
  `kernels_ok: false` và bộ nhớ dùng được bằng 0, nên không có gì được đặt lên nó, thay vì mọi lần chạy đều lỗi
  "no kernel image is available". Trang GPUs đánh dấu "No kernels in this build".
- Form model hiển thị context mỗi request nhận được (context ÷ số slot song song).

## [0.4.2] - 2026-10-05

### Sửa lỗi

- Công tắc "In pool" ở trang Servers và GPUs không có tác dụng: phần chọn GPU riêng cho từng model trong form model
  khai báo thêm một hàm `toggleGpu` trong component UI, đè lên hàm gọi API, nên mỗi lần bấm chỉ thay đổi form model
  (đang ẩn). Các handler của form model giờ là `pinToggleServer` / `pinToggleGpu`, và có test báo lỗi nếu component
  có hai method trùng tên.

## [0.4.1] - 2026-10-04

### Thêm mới

- GPU Blackwell (RTX 5090/5080, RTX PRO 6000): image agent build kernel cho compute capability 120 (llama.cpp đổi thành
  120a để dùng FP4 tensor core). Build image giờ báo lỗi nếu thiếu bất kỳ kiến trúc nào đã yêu cầu trong backend CUDA,
  thay vì để một server lỗi "no kernel image is available" lúc chạy.

### Thay đổi

- Image agent build trên CUDA 12.8.1 (trước là 12.4.1), bản toolkit đầu tiên hỗ trợ Blackwell. Vẫn chạy trên driver
  >= 525; card RTX 50 cần driver >= 570.

### Sửa lỗi

- Dừng một model chạy chia qua RPC làm máy chính bị crash: mọi engine bị dừng cùng lúc, `ggml-rpc-server` thoát ngay,
  rồi `llama-server` không giải phóng được buffer ở máy RPC và tự abort (SIGABRT, mỗi lần một file core dump 0,1-0,5 GB,
  WSL lưu trong `%TEMP%\wsl-crashes`). Giờ máy chính được dừng trước và chờ thoát hẳn, lệnh dừng chờ lâu hơn thời gian
  ân hạn của agent, và engine chạy không sinh core dump.

## [0.4.0] - 2026-10-03

### Thêm mới

- **Chuyển đổi Hugging Face sang GGUF trong coordinator**: tải về (hoặc dùng thư mục trên máy chủ),
  `convert_hf_to_gguf.py`, `llama-quantize`, kiểm định, rồi đưa vào thư viện mô hình; dùng từ giao diện hoặc các
  route `/api/convert*`. Có bước inspect đọc cấu hình mà không tải trọng số và liệt kê mọi kiểu lượng tử hóa kèm
  dung lượng, VRAM ước tính, và việc có vừa một GPU hay cả pool; job có các giai đoạn, log, hủy/thử lại/chấp
  nhận/xóa, cache bản tải, và tự xếp hàng lại job bị ngắt sau khi khởi động lại.
- **Cổng kiểm định** trước khi tệp đã chuyển vào thư viện: header GGUF, so id tokenizer với Hugging Face trên 8 văn
  bản, và một lần sinh ngắn trên CPU. Nếu lệch, job chuyển sang `needs_review`.
- **Ma trận tầm quan trọng và 10 kiểu lượng tử hóa IQ1/IQ2/IQ3**: giai đoạn "calibrating" chạy `llama-imatrix` (chế
  độ auto/on/off). Có sẵn văn bản hiệu chỉnh đa ngôn ngữ do dự án tự viết; có thể thay bằng tệp `.txt` riêng (tối
  đa 20 MB).
- **Giới hạn server/GPU theo từng mô hình**: `pin_devices` nhận `<node>/*`; giao diện hiện cây "Mọi máy chủ và GPU |
  Chỉ những cái đã chọn".
- **Nút sao chép** trên thẻ mô hình: endpoint, tên mô hình và lệnh `curl` dựng sẵn.
- **Tường lửa RPC** (`GPUPOOL_RPC_FIREWALL=1`, cần root và `NET_ADMIN`): mỗi cổng RPC chỉ nhận node đầu, loopback và
  địa chỉ của chính agent.
- **Tự hiệu chỉnh VRAM**: dung lượng buffer đo thực của engine hiệu chỉnh ước tính bộ nhớ bằng một hệ số lưu lại theo
  từng mô hình.
- **Lưu trạng thái điều khiển**: trạng thái autoscaler, quyền chiếm và thời gian chờ của preemption, backoff và bước
  rebalance đang dở đều sống sót qua lần khởi động lại coordinator.
- **Kiểm thử end-to-end trên CI** (`scripts/ci_e2e.py`): image thật gồm coordinator và 3 agent CPU phục vụ một mô hình
  nhỏ chia qua RPC, kèm giai đoạn chuyển đổi.
- Tài liệu: chuyển đổi, ma trận tầm quan trọng, giới hạn theo mô hình, báo cáo kiểm thử (EN + VI). Tệp giấy phép và
  cộng đồng: `LICENSE` (MIT), thông báo bên thứ ba, hướng dẫn đóng góp, quy tắc ứng xử, chính sách bảo mật, mẫu issue
  và pull request.

### Thay đổi

- Repo đổi tên từ `multi-gpu-inference` thành `gpupool` (link cũ tự chuyển hướng); tên image không đổi.
- Image coordinator có thể kèm bộ công cụ chuyển đổi (`WITH_CONVERT=1`, thêm khoảng 1,2 GB); `WITH_CONVERT=0` giữ
  image gọn.
- Gửi job chuyển đổi bị từ chối với mã 507 khi ổ đĩa rõ ràng không đủ chỗ, thay vì thất bại sau vài phút.
- Ước tính dung lượng cho các kiểu lượng tử hóa tính riêng phần embedding và các trường hợp dự phòng của
  `llama-quantize` (trước đây trung bình đơn giản thấp hơn 24 % ở một mô hình nhỏ; nay lệch vài phần trăm so với tệp
  thật).

### Sửa lỗi

- Image Docker không build được sau khi `pyproject.toml` khai báo file license: Dockerfile không chép `LICENSE` và
  `THIRD_PARTY_NOTICES.md` vào bước build.
- CI bị đỏ từ khi có tính năng chuyển đổi: trình quản lý job chuyển đổi không tạo thư mục dữ liệu của coordinator
  trước khi mở cơ sở dữ liệu, nên bản checkout mới hoặc `db_path` ở thư mục mới đều lỗi. Nay thư mục được tạo, có
  test hồi quy.
- Văn bản hiệu chỉnh có sẵn bị bỏ sót khỏi một commit do quy tắc ignore `data/`; bản checkout mới sẽ từ chối mọi job
  ma trận tầm quan trọng. Thư mục dữ liệu của gói nay được loại khỏi quy tắc đó.
- Hiệu chỉnh chia hệ số hoạch định ra khỏi ước tính trước khi lấy mẫu.
- Một test agent chập chờn do tranh ghi log với engine giả nay đã biết chờ; test thuần khiết của simulate bỏ qua tải
  trực tiếp của mock.

### Bảo mật

- Chuyển đổi không bao giờ tải hay chạy tệp Python của repo trừ khi bật `allow_remote_code`, chạy bộ chuyển đổi ở
  chế độ offline trên một thư mục tạm chứa liên kết tới các tệp trong danh sách cho phép, và chỉ xóa tệp dưới
  `models_dir/.convert` và `.hf`.

## [0.3.0] - 2026-10-03

### Thêm mới

- **Nền tảng nhiều mô hình**, các giai đoạn 1 đến 4 của thiết kế nền tảng: độ ưu tiên, spread và xếp chỗ có chấm điểm
  cùng `/api/recommend` và `/api/capacity`; autoscaling với scale-to-zero và cold start; preemption theo độ ưu tiên
  cùng `/api/simulate`; rebalance kiểu make-before-break cùng `/api/rebalance`.
- Lượng tử hóa KV cache (`f16`, `q8_0`, `q4_0`) và giải mã suy đoán (`ngram` hoặc mô hình `draft`) theo từng mô hình.
- Cụm 3 máy chủ giả lập (`docker-compose.sim.yml`) để thử hệ thống trên một máy.
- Build Docker chạy được ở nơi git hoặc GitHub bị chặn (tarball llama.cpp có sẵn, mirror, tài liệu proxy và đường
  dẫn); tìm tệp mô hình theo đường dẫn trên máy chủ khi coordinator chạy trong Docker.
- Định danh GPU theo UUID; giới hạn kích thước body request của router.
- Giao diện thiết kế lại: design token, giao diện sáng, hiệu ứng chuyển động, favicon, bộ chọn tệp trên máy chủ.

### Thay đổi

- Xếp chỗ đa node xét mọi tổ hợp máy chủ khả thi chứ không chỉ cách tham lam; bộ chấm điểm chọn.
- Cả hai image ghim Python 3.12 (`.python-version` nay được sao chép; trước đó uv chọn 3.14).
- Dọn lại phần lõi coordinator: hàm dùng chung, trạng thái có kiểu, không so khớp lỗi bằng chuỗi.

### Sửa lỗi

- Giữ nguyên GPU trong các xếp chỗ đa node; không còn nhầm coordinator bị đứng với node đã chết.
- Mất GPU, tệp GGUF chia nhỏ, rò rỉ cổng và replica, và vòng lặp khởi chạy lại khi crash.
- Một chuỗi `\n` nguyên văn trong khối `ENV` của Dockerfile coordinator làm `docker build` thất bại.
- Gia cố cache của agent và API engine; router và `/report` nhẹ hơn; khởi động mô hình không còn chặn một tick.

## [0.2.0] - 2026-10-02

### Thêm mới

- Giao diện quản lý, sổ đăng ký máy chủ có thăm dò định kỳ, công tắc GPU và nhật ký sự kiện.
- Thông báo khi có sự cố.
- Cài đặt Docker một lệnh (khóa sinh sẵn, agent tự tham gia, hướng dẫn bắt đầu nhanh), cấu hình bằng biến môi
  trường và build image trên CI.
- Thư viện mô hình: tải từ Hugging Face và đường dẫn cục bộ đã đăng ký.
- Số liệu GPU cho giao diện; mất một GPU không còn che các GPU khỏe.
- Hỗ trợ proxy: lưu lượng ra ngoài đi qua `http_proxy`/`https_proxy`, lưu lượng nội bộ thì không bao giờ.

## Trước 0.2.0 (chưa gắn tag)

- Mặt phẳng điều khiển xây theo từng lớp: hợp đồng dùng chung (mô hình dữ liệu truyền, xác thực, cấu hình), agent trên
  node (dò thiết bị, giám sát engine, cache mô hình, HTTP API), bộ lập lịch (đọc header GGUF dạng luồng, ước tính bộ
  nhớ, xếp chỗ), router (cân bằng theo tiền tố, proxy tương thích OpenAI) và coordinator (kho lưu trữ, client gọi
  agent, reconciler, API quản trị).
- Ước tính bộ nhớ hiệu chỉnh theo buffer thật của llama.cpp; script end-to-end; khởi chạy đa node truyền `--rpc`
  trước `--device`; agent dọn các engine mồ côi.
- Tài liệu song ngữ Anh và Việt: README, thiết kế, báo cáo kiểm thử.

[Chưa phát hành]: https://github.com/longduongbao29/gpupool/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/longduongbao29/gpupool/compare/v0.7.3...v0.8.0
[0.7.3]: https://github.com/longduongbao29/gpupool/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/longduongbao29/gpupool/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/longduongbao29/gpupool/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/longduongbao29/gpupool/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/longduongbao29/gpupool/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/longduongbao29/gpupool/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/longduongbao29/gpupool/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/longduongbao29/gpupool/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/longduongbao29/gpupool/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/longduongbao29/gpupool/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/longduongbao29/gpupool/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/longduongbao29/gpupool/releases/tag/v0.2.0
