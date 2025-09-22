# Auto NLP - Hướng dẫn Chạy Dự án

## 4 Bước Chạy Dự án

### 1. Tạo và kích hoạt virtual environment
```bash
python -m venv env
env\Scripts\activate
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Huấn luyện model
```bash
python train.py
```

### 4. Chạy API server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Test API
- **API Endpoint:** http://localhost:8000/infer

