from ultralytics import YOLO
import torch

try:
    # 1. Build model từ file yaml custom
    model = YOLO("custom_11.yaml")
    
    # 2. Tạo một ảnh giả (dummy input) để test luồng dữ liệu
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # 3. Chạy thử inference (Forward pass)
    results = model(dummy_input)
    
    print("✅ CẤU HÌNH THÀNH CÔNG! Model đã nhận diện RepC3 và chạy không lỗi.")
    print(model.info())

except Exception as e:
    print("❌ CÓ LỖI XẢY RA:")
    print(e)