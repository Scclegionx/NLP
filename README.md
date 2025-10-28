# PhoBERT NLP Server

Dự án NLP server sử dụng PhoBERT để phân loại intent và trích xuất entities từ tiếng Việt.

## Cài đặt

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Tải model PhoBERT (sẽ tự động tải khi chạy lần đầu):
```bash
python train.py
```

## Sử dụng

### 1. Training Model

Chạy script training để fine-tune PhoBERT trên dataset:

```bash
python train.py
```

Model sẽ được lưu trong thư mục `./phobert_model/`

### 2. Chạy Server

Khởi động FastAPI server:

```bash
python app.py
```

Server sẽ chạy tại `http://localhost:8000`

### 3. API Endpoints

#### Health Check
```
GET /health
```

#### Predict Intent
```
POST /predict
Content-Type: application/json

{
    "text": "nhắn tin cho mẹ là con sắp về"
}
```

Response:
```json
{
    "text": "nhắn tin cho mẹ là con sắp về",
    "intent": "send-msg",
    "confidence": 0.95,
    "command": "send-msg",
    "entities": {
        "RECEIVER": "mẹ",
        "MESSAGE": "con sắp về",
        "PLATFORM": "sms"
    },
    "value": "con sắp về",
    "timestamp": "2025-01-05T11:07:19.436163"
}
```

#### Batch Predict
```
POST /batch_predict
Content-Type: application/json

[
    {"text": "nhắn tin cho mẹ là con sắp về"},
    {"text": "gọi điện cho bố"}
]
```

### 4. Test Model

Chạy script test để kiểm tra model:

```bash
python predict.py
```

## Các Intent được hỗ trợ

1. **send-msg**: Gửi tin nhắn SMS
2. **send-msg-whatsapp**: Gửi tin nhắn qua WhatsApp
3. **call**: Gọi điện thoại
4. **call-whatsapp**: Gọi video qua WhatsApp
5. **add-contacts**: Thêm liên hệ
6. **search**: Tìm kiếm thông tin
7. **control-device**: Điều khiển thiết bị (đèn pin, wifi, âm lượng)
8. **open-cam**: Mở camera (chụp ảnh/quay video)
9. **set-alarm**: Đặt báo thức
10. **unknown**: Không khớp với intent nào

## Entities được trích xuất

- **RECEIVER**: Người nhận (mẹ, bố, chị, anh, em, bạn...)
- **MESSAGE**: Nội dung tin nhắn
- **PLATFORM**: Nền tảng (sms, whatsapp, phone, chrome, youtube)
- **DEVICE**: Thiết bị (flash, wifi, volume)
- **DATE**: Ngày tháng (tomorrow, day_after_tomorrow, thứ trong tuần)
- **value**: Giá trị cụ thể (ON/OFF, +/-, image/video, thời gian)

## Cấu trúc dự án

```
NLP_test/
├── app.py                 # FastAPI server
├── train.py               # Script training
├── predict.py             # Script prediction
├── dataset.json           # Dataset training
├── requirements.txt       # Dependencies
├── phobert_model/         # Model đã train (sẽ tạo sau khi train)
└── README.md              # Hướng dẫn này
```

## Lưu ý

- Model PhoBERT sẽ được tải từ Hugging Face khi chạy lần đầu
- Dataset hiện tại có 45 samples training và 15 samples test
- Có thể mở rộng dataset bằng cách thêm samples vào `dataset.json`
- Server hỗ trợ batch prediction để xử lý nhiều text cùng lúc
