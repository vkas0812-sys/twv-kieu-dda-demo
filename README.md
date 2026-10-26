# 10_app_demo — Gói Ứng dụng Độc lập (Self-Contained Product Package)

Thư mục **`10_app_demo/`** đã được đóng gói **hoàn toàn độc lập (100% self-contained)**, chứa đầy đủ mã nguồn giao diện Streamlit, API FastAPI, thư viện lõi `kieu_pipeline`, toàn bộ dữ liệu gốc Parquet (`preclean`, `active`, `COGS`), toàn bộ 13 file CSV kết quả thực nghiệm (`03a–03e`, `published_base_stock_targets.csv`) và 27 biểu đồ luận văn (`data/figures/`).

Bạn có thể **nén ZIP thư mục `10_app_demo` này mang sang bất kỳ máy tính nào khác**, đẩy lên GitHub để triển khai Cloud, hoặc chạy trực tiếp khi bảo vệ luận văn mà không phụ thuộc vào các thư mục bên ngoài.

---

## 1. Cấu trúc Thư mục `10_app_demo/`

```text
10_app_demo/
├── .streamlit/config.toml        # Cấu hình giao diện & cổng 8501
├── apps/
│   ├── api/                      # Backend FastAPI & các dịch vụ tính toán
│   └── web/                      # Giao diện Streamlit (Trang chủ + 8 phân hệ)
├── packages/
│   └── kieu_pipeline/            # Thư viện lõi: ABC-XYZ, SAA, Solver LP/MILP (PuLP CBC)
├── data/
│   ├── parquet/                  # File dữ liệu thật (Clean Baseline, Active, COGS)
│   ├── results/                  # File kết quả thật 03a -> 03e & thư mục app_runs/
│   └── figures/                  # 27 biểu đồ chuẩn xuất bản Chương 4 & 5
├── tests/
│   └── test_acceptance.py        # Bộ 8 bài kiểm thử tự động (Acceptance Tests)
├── run_app.command               # File click đúp chạy ngay trên macOS
├── Makefile                      # Lệnh tắt: make web | make api | make test | make public
├── Dockerfile & docker-compose.yml
└── requirements.txt
```

---

## 2. Cách Chạy Nhanh trên Máy (Local)

- **Cách 1 (Click đúp trên macOS):** Click đúp vào file `run_app.command`.
- **Cách 2 (Chạy từ Terminal):**
  ```bash
  cd 10_app_demo
  make web
  ```
  Sau đó mở trình duyệt tại: **`http://localhost:8501`** (hoặc `http://localhost:8501/Tong_quan`).

---

## 3. Ba Cách Public Ứng dụng ra Internet (Lấy Link gửi Hội đồng / Giảng viên)

### Cách A: Public tức thì từ máy tính của bạn bằng Cloudflare Tunnel (Khuyên dùng — Nhanh & Ổn định nhất, không cần tạo tài khoản)
Khi ứng dụng đang chạy ở cổng `8501`, mở thêm 1 cửa sổ Terminal và chạy:
```bash
cloudflared tunnel --url http://localhost:8501
```
*(Nếu máy chưa có `cloudflared`, cài bằng `brew install cloudflared`).*  
Hệ thống sẽ cấp ngay một đường link HTTPS công khai dạng `https://xxxx-xxxx.trycloudflare.com` để mọi người truy cập từ điện thoại/máy tính bất kỳ.

### Cách B: Public tức thì qua LocalTunnel (`make public`)
```bash
cd 10_app_demo
make public
```

### Cách C: Đưa lên **Streamlit Community Cloud** (Miễn phí vĩnh viễn 24/7)
Vì `10_app_demo/` đã chứa các file thật (không còn symlink), bạn chỉ cần:
1. Tạo 1 repository GitHub mới và đẩy toàn bộ nội dung bên trong `10_app_demo/` lên đó.
2. Truy cập [share.streamlit.io](https://share.streamlit.io), chọn repo vừa tạo.
3. Đặt **Main file path** là: `apps/web/streamlit_app.py`.
4. Bấm **Deploy!** — Sau 2 phút bạn sẽ có tên miền cố định dạng `https://twv-kieu-dda.streamlit.app`.
