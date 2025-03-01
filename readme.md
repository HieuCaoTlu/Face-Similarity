Đi tới thư mục chính, chạy dòng lệnh sau:

```bash
docker-compose up --build
```

Để sử dụng, ta hãy gọi:

```bash
localhost:5000/detect
POST - form-data với key và value
image: ảnh 1
image2: ảnh 2
```

Kết quả trả về là điểm tương quan và đánh giá mức độ
Người dùng có thể tự xem xét đâu nên là ngưỡng đánh giá

```bash
{
    "cccd_url": "processed/cccd_20250301075124.jpg",
    "face_url": "processed/face_20250301075124.jpg",
    "result": "medium", #mức độ tương quan low - medium - high
    "similarity": 0.41671857237815857 #điểm tương quan
}
```
