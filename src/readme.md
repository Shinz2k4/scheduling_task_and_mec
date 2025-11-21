# 1. Cấu trúc và Mục đích của Dự án

## Cấu trúc

Thư mục dự án gồm các thành phần sau:

- **build.sh**: Script xây dựng môi trường và khởi tạo dự án.
- **clearqueue.sh**: Script xóa hàng đợi các tiến trình mô phỏng.
- **display_computing_result.py**: Hiển thị kết quả tính toán mô phỏng.
- **docker/**: Thư mục chứa các tập tin phục vụ triển khai bằng Docker.
- **env.py**: Định nghĩa môi trường mô phỏng.
- **handle_host_request.py**, **host_send_request.py**: Quản lý và gửi yêu cầu giữa các tiến trình hoặc máy chủ.
- **infer_service_proc.py**, **inference.py**: Xử lý và thực hiện suy luận mô phỏng.
- **kill.sh**: Script dừng các tiến trình mô phỏng cụ thể.
- **kill_all.py**: Script dừng tất cả các service mô phỏng đang chạy trên một server.
- **launch.py**: Khởi chạy các thành phần của dự án.
- **main.py**: Điểm bắt đầu của chương trình mô phỏng.
- **requirements.txt**: Danh sách các thư viện/phụ thuộc cần thiết.
- **run_many_inspection.sh**: Script kiểm thử nhiều mô phỏng cùng lúc.
- **scheduler_proc.py**: Quản lý và lên lịch các tiến trình mô phỏng.
- **service/**: Thư mục chứa các dịch vụ liên quan đến mô phỏng.
- **trainer.py**, **trainer_processing_time.py**: Huấn luyện mô hình và đo thời gian xử lý.
- **utils.py**: Các hàm tiện ích hỗ trợ mô phỏng.
- **val2017/**, **val2017.zip**: Bộ dữ liệu mẫu phục vụ kiểm thử và đánh giá.
- **config.yaml**: Tập tin cấu hình dự án.

# 2. Hướng dẫn chạy dự án

## Bước 1: Cài đặt Docker

Để chạy dự án, bạn cần cài đặt Docker. Tham khảo hướng dẫn cài đặt tại [Docker Documentation](https://docs.docker.com/get-docker/).

## Bước 2: Xây dựng môi trường

Sau khi cài đặt Docker, mở terminal và chuyển đến thư mục dự án. Thực hiện lệnh sau để khởi tạo môi trường:

```bash
source build.sh
```

Script `build.sh` sẽ thiết lập các phụ thuộc cần thiết và chuẩn bị môi trường để chạy mô phỏng.

## Bước 3: Khởi chạy mô phỏng

Sau khi hoàn tất các bước trên, bạn có thể tiếp tục với các hướng dẫn kiểm thử hoặc triển khai mô phỏng theo nhu cầu.

## Bước 4: Dừng các tiến trình mô phỏng

Để dừng các tiến trình mô phỏng đang chạy, bạn có thể sử dụng các script sau:

- **kill.sh**: Dừng các tiến trình mô phỏng cụ thể. Thực hiện lệnh:
  
  ```bash
  bash kill.sh
  ```

- **kill_all.py**: Dừng toàn bộ các service mô phỏng trên server. Thực hiện lệnh:
  
  ```bash
  python3 kill_all.py
  ```

Việc sử dụng các script này giúp đảm bảo các tiến trình mô phỏng được dừng đúng cách, tránh xung đột khi khởi động lại dự án.Bước 4: Dừng các tiến trình mô phỏng

Để dừng các tiến trình mô phỏng đang chạy, bạn có thể sử dụng các script sau:

- **kill.sh**: Script này dùng để dừng các tiến trình mô phỏng cụ thể. Chạy lệnh sau trong terminal:
  
  ```bash
  bash kill.sh
  ```

- **kill_all.py**: Script này dùng để dừng toàn bộ các tiến trinh cua service in the same server. Chạy lệnh sau:
  
  ```bash
  python3 kill_all.py
  ```

Sử dụng các script này giúp đảm bảo các tiến trình mô phỏng được dừng an toàn và tránh xung đột khi khởi chạy lại dự án.

