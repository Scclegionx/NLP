# NLP Django Project

Dự án Django xử lý ngôn ngữ tự nhiên với API endpoint để xử lý văn bản.

## Cài đặt và chạy local

1. Clone repository:
```bash
git clone <your-repo-url>
cd NLP
```

2. Tạo và kích hoạt virtual environment:
```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# Linux/Mac
source myenv/bin/activate
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

4. Chạy migrations:
```bash
python manage.py migrate
```

5. Chạy server:
```bash
python manage.py runserver
```

## API Endpoints

### POST /api/process-text/
Xử lý văn bản và trả về kết quả phân tích.

**Request:**
```json
{
    "text": "Mở ứng dụng Chrome"
}
```

**Response:**
```json
{
    "command": "open_app",
    "entities": ["Chrome"],
    "values": ["chrome"],
    "confidence": 0.95
}
```

## Workflow CI/CD

Dự án sử dụng GitHub Actions để tự động hóa quá trình test và deploy.

### Cách hoạt động:

1. **Khi push code lên các branch**: `develop`, `feature/*`, `hotfix/*`
2. **Workflow sẽ tự động chạy**:
   - Cài đặt dependencies từ `requirements.txt`
   - Kiểm tra và chạy migrations nếu có
   - Chạy tests (Django tests và pytest)
   - Kiểm tra server có khởi động được không
   - Nếu tất cả pass → tự động merge vào `main` branch

### Cách sử dụng:

1. Tạo branch mới từ main:
```bash
git checkout -b feature/your-feature-name
```

2. Code và commit:
```bash
git add .
git commit -m "Add new feature"
```

3. Push lên remote:
```bash
git push origin feature/your-feature-name
```

4. Workflow sẽ tự động chạy và merge vào main nếu thành công.

### Cấu trúc branch:

- `main`: Branch chính, code production
- `develop`: Branch development
- `feature/*`: Các tính năng mới
- `hotfix/*`: Sửa lỗi khẩn cấp

## Testing

Chạy tests:
```bash
# Django tests
python manage.py test

# Pytest
pytest

# Test cụ thể
python manage.py test base.tests.ProcessTextAPITest
```

## Lưu ý

- Workflow không cần chạy `myenv` vì GitHub Actions sẽ tự tạo virtual environment
- Đảm bảo tất cả dependencies được liệt kê trong `requirements.txt`
- Tests phải pass trước khi merge vào main 