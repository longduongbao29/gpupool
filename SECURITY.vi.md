# Chính sách bảo mật

> Tiếng Việt. English: [SECURITY.md](SECURITY.md)

## Phiên bản được hỗ trợ

| Phiên bản | Được hỗ trợ |
| --- | --- |
| 0.3.x | có |
| cũ hơn | không |

Bản sửa được đưa vào bản 0.3.x mới nhất; hãy nâng cấp để nhận chúng.

## Báo cáo lỗ hổng

**Đừng mở issue công khai.** Hãy báo riêng qua GitHub:

1. Mở <https://github.com/longduongbao29/multi-gpu-inference/security/advisories/new> (kho mã, tab *Security*,
   *Report a vulnerability*).
2. Mô tả vấn đề như dưới đây. Chỉ người bảo trì xem được báo cáo.

Vui lòng nêu:

- phiên bản gpupool hoặc tag image, và cách triển khai (Docker hay chạy trực tiếp, coordinator và agent);
- bạn tìm thấy gì và tác động của nó (kẻ tấn công đọc, sửa hay chạy được gì, từ đâu trong mạng);
- các bước tái hiện, tốt nhất là tối giản, kèm cấu hình liên quan (tuyệt đối không đưa khóa, token hay bí mật thật);
- log hoặc bản chứng minh khái niệm nếu có, và cách sửa bạn đề xuất.

Đây là dự án nhỏ do một người duy trì. Chúng tôi cố gắng xác nhận báo cáo và sửa lỗi đã xác nhận trong khả năng; **không
cam kết thời gian phản hồi hay sửa lỗi**. Xin cho chúng tôi thời gian hợp lý để sửa trước khi bạn công bố; nếu bạn
muốn, chúng tôi sẽ ghi nhận tên bạn trong thông báo bảo mật.

## Mô hình bảo mật

gpupool dành cho mạng riêng hoặc VPN giữa các máy bạn kiểm soát. Những gì được bảo vệ và không được bảo vệ:

- **Khóa quản trị và cluster token.** Khóa quản trị (`GPUPOOL_ADMIN_KEY`) bảo vệ giao diện web và API quản trị;
  cluster token (`GPUPOOL_CLUSTER_TOKEN`) xác thực máy chủ GPU với coordinator và nằm trong lệnh tham gia. Cả hai được
  sinh ở lần khởi động đầu và lưu trong `secrets.json` cạnh cơ sở dữ liệu (volume dữ liệu trong Docker); đặt giá trị
  riêng nếu muốn ghi đè. Hãy bảo vệ tệp đó và volume dữ liệu.
- **Khóa API cho `/v1`.** Máy khách của API tương thích OpenAI gửi các khóa trong `GPUPOOL_API_KEYS`. Nếu không đặt
  biến này, `/v1` mở cho bất kỳ ai tới được cổng 8080: hãy đặt nó.
- **Cổng RPC không có xác thực.** `ggml-rpc-server` của llama.cpp (cổng 9000 đến 9999 trên mỗi máy chủ GPU) nhận mọi
  kết nối: ai tới được nó có thể cấp phát bộ nhớ GPU và đọc hoặc ghi tensor, và lưu lượng không được mã hóa. Hoặc đặt
  `GPUPOOL_RPC_FIREWALL=1` (agent sẽ thêm luật iptables/ip6tables cho từng engine RPC, chỉ cho phép loopback, node đầu
  và địa chỉ của chính agent; cần root và `NET_ADMIN`, ví dụ `--cap-add NET_ADMIN`), hoặc giới hạn 9000 đến 9999 chỉ
  cho các máy trong cụm bằng tường lửa của bạn, hoặc giữ các máy chủ trong mạng riêng. Agent cảnh báo khi khởi động
  nếu tường lửa đang tắt.
- **Không có TLS tích hợp.** Cluster token, khóa quản trị và khóa API đi qua HTTP thường. Hãy đặt reverse proxy kết
  thúc TLS phía trước coordinator nếu có ai kết nối từ ngoài mạng tin cậy, và giữ các máy chủ trong mạng riêng hoặc
  VPN.
- **Chuyển đổi không chạy mã của repo.** Chuyển đổi một mô hình Hugging Face không bao giờ tải hay thực thi các tệp
  Python của repo, trừ khi bạn đặt `allow_remote_code` rõ ràng cho job đó. Bộ chuyển đổi chạy offline trên một thư mục
  tạm chứa các tệp trong danh sách cho phép, và gpupool chỉ xóa tệp trong các thư mục chuyển đổi và cache của riêng
  nó. Hãy coi `allow_remote_code` như việc chạy mã không tin cậy trên coordinator.
- **Agent được cấp quyền cao có chủ đích.** Container agent chạy với `--pid host` và `--network host` (và `NET_ADMIN`
  khi bật tường lửa RPC) để thấy GPU, tiến trình và gắn cổng cho engine. Chỉ chạy nó trên máy bạn kiểm soát, và đừng
  mở cổng của nó (7070) ra ngoài coordinator.
- **Mô hình là dữ liệu của bên thứ ba.** Tệp mô hình được llama.cpp phân tích; chỉ nạp mô hình từ nguồn bạn tin cậy và
  giữ llama.cpp được cập nhật.

Chi tiết vận hành xem mục Bảo mật trong [docs/QUICKSTART.vi.md](docs/QUICKSTART.vi.md).
