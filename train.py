from ultralytics import YOLO

model = YOLO(r'D:\ultralytics_v2\Model_yaml\yolov11n_custom.yaml')


# Huấn luyện mô hình với cấu hình dữ liệu
model.train(data=r'D:\fish-disease-object-detect.v1i.yolov8\data.yaml',
            epochs=1,  # Số lượng epoch
            imgsz=640, # Kích thước ảnh đầu vào
            save_period=5,
            device='cpu')  # train trên 2 cpu 0 và 1 (train trên 1 cpu device=0)