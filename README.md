# Signature Validation API

API xác thực chữ ký viết tay dành cho phiếu giao hàng, sử dụng Python + FastAPI.

## Tính năng

- Quản lý nhân viên và chữ ký mẫu (tối đa 2 chữ ký/nhân viên từ hợp đồng)
- Xác thực chữ ký trên phiếu giao hàng bằng 3 metric kết hợp
- Ngưỡng chấp nhận và trọng số cấu hình được qua API
- Lưu lịch sử toàn bộ các lần xác thực

## Thuật toán xác thực

Mỗi ảnh chữ ký được tiền xử lý (grayscale → binarize Otsu → khử nhiễu → crop → resize chuẩn hoá), sau đó so sánh qua 3 metric:

| Metric | Trọng số mặc định | Mô tả |
|--------|:-----------------:|-------|
| **SSIM** (Structural Similarity) | 40% | Độ giống cấu trúc tổng thể |
| **ORB** (Feature Matching) | 35% | So khớp keypoint, chịu được xoay nhỏ |
| **Contour** (Hu Moments) | 25% | So sánh hình dạng nét chữ |

```
overall_score = 0.4×SSIM + 0.35×ORB + 0.25×Contour
is_valid      = overall_score >= threshold  (mặc định 0.75)
```

Khi nhân viên có 2 chữ ký mẫu, lấy **score cao nhất** trong hai lần so sánh.

## Cài đặt

```bash
# Clone / tạo thư mục project
cd signature-validation

# Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Cài dependencies
pip install -r requirements.txt
```

## Chạy server

```bash
uvicorn main:app --reload --port 8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

## Cấu hình (tuỳ chọn)

Tạo file `.env` để override các giá trị mặc định:

```dotenv
DATABASE_URL=sqlite:///./signature_validation.db
STORAGE_DIR=storage/signatures
SIMILARITY_THRESHOLD=0.75
SSIM_WEIGHT=0.4
ORB_WEIGHT=0.35
CONTOUR_WEIGHT=0.25
```

## API Endpoints

### Nhân viên

| Method | Path | Mô tả |
|--------|------|-------|
| `POST` | `/employees` | Tạo nhân viên mới |
| `GET` | `/employees` | Danh sách nhân viên |
| `GET` | `/employees/{code}` | Chi tiết nhân viên |
| `DELETE` | `/employees/{code}` | Vô hiệu hoá nhân viên |

### Chữ ký mẫu

| Method | Path | Mô tả |
|--------|------|-------|
| `POST` | `/employees/{code}/signatures` | Upload chữ ký mẫu từ hợp đồng |
| `GET` | `/employees/{code}/signatures` | Xem danh sách chữ ký mẫu |
| `DELETE` | `/employees/{code}/signatures/{id}` | Xóa chữ ký mẫu |

> Mỗi nhân viên tối đa **2 chữ ký mẫu**. Xóa trước khi upload mới.

### Xác thực

| Method | Path | Mô tả |
|--------|------|-------|
| `POST` | `/validate` | Xác thực chữ ký trên phiếu giao hàng |
| `GET` | `/validate/history/{code}` | Lịch sử xác thực của nhân viên |

### Cấu hình

| Method | Path | Mô tả |
|--------|------|-------|
| `GET` | `/config` | Xem cấu hình hiện tại |
| `PUT` | `/config` | Cập nhật threshold và trọng số |
| `POST` | `/config/reset` | Reset về mặc định |

## Ví dụ sử dụng

### 1. Tạo nhân viên

```bash
curl -X POST http://localhost:8000/employees \
  -H "Content-Type: application/json" \
  -d '{"employee_code": "NV001", "full_name": "Nguyen Van A"}'
```

### 2. Upload chữ ký mẫu từ hợp đồng

```bash
# Chữ ký mẫu 1
curl -X POST http://localhost:8000/employees/NV001/signatures \
  -F "file=@/path/to/signature1.png" \
  -F "contract_ref=HD-2024-001"

# Chữ ký mẫu 2
curl -X POST http://localhost:8000/employees/NV001/signatures \
  -F "file=@/path/to/signature2.png" \
  -F "contract_ref=HD-2024-001"
```

### 3. Xác thực chữ ký trên phiếu giao hàng

```bash
curl -X POST http://localhost:8000/validate \
  -F "employee_code=NV001" \
  -F "file=@/path/to/delivery_signature.png" \
  -F "delivery_ref=PGH-2024-0042"
```

Kết quả:

```json
{
  "is_valid": true,
  "overall_score": 0.8312,
  "ssim_score": 0.8754,
  "orb_score": 0.7891,
  "contour_score": 0.8201,
  "threshold_used": 0.75,
  "employee_code": "NV001",
  "delivery_ref": "PGH-2024-0042",
  "detail": {
    "per_reference": [...],
    "threshold": 0.75
  }
}
```

### 4. Điều chỉnh ngưỡng chấp nhận

```bash
curl -X PUT http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{
    "similarity_threshold": 0.70,
    "ssim_weight": 0.4,
    "orb_weight": 0.35,
    "contour_weight": 0.25
  }'
```

> Tổng `ssim_weight + orb_weight + contour_weight` phải bằng **1.0**.

### 5. Override threshold cho một lần xác thực cụ thể

```bash
curl -X POST http://localhost:8000/validate \
  -F "employee_code=NV001" \
  -F "file=@/path/to/signature.png" \
  -F "threshold=0.65"
```

## Cấu trúc project

```
signature-validation/
├── main.py                      # FastAPI app, lifespan, middleware
├── config.py                    # Settings (pydantic-settings + .env)
├── database.py                  # SQLAlchemy engine & session
├── models.py                    # ORM models
├── schemas.py                   # Pydantic request/response schemas
├── requirements.txt
├── validators/
│   ├── __init__.py
│   └── signature_validator.py   # Preprocessing + SSIM/ORB/Contour logic
├── routers/
│   ├── employees.py
│   ├── signatures.py
│   ├── validation.py
│   └── config.py
└── storage/
    └── signatures/              # Ảnh chữ ký mẫu và input (audit trail)
```

## Định dạng ảnh hỗ trợ

JPEG, PNG, WebP, TIFF

## Lưu ý khi triển khai thực tế

- **Chất lượng ảnh**: Chụp trên nền trắng, ánh sáng đều, độ phân giải tối thiểu 150 DPI.
- **Góc chụp**: Càng thẳng càng tốt — ORB chịu được xoay nhỏ (< 15°) nhưng không xử lý perspective skew.
- **Threshold**: Bắt đầu với `0.70`–`0.75` và tinh chỉnh dựa trên dữ liệu thực tế. Chữ ký tự nhiên có biến động nên tránh đặt quá cao (> 0.90).
- **Lưu trữ**: Thư mục `storage/` chứa toàn bộ ảnh input để audit. Cần backup định kỳ.
- **Database**: SQLite phù hợp cho môi trường nhỏ. Chuyển sang PostgreSQL khi scale lên bằng cách đổi `DATABASE_URL` trong `.env`.
