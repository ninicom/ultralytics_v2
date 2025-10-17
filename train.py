from ultralytics import YOLO

model = YOLO(r'D:\python\ultralytics_v2\Model_yaml\Yolov11n_SCFB.yaml')


# Huấn luyện mô hình với cấu hình dữ liệu
model.train(data=r'E:\data\gan nhan.v15-more-fish-v2.yolov8\data.yaml',
            epochs=1,  # Số lượng epoch
            imgsz=640, # Kích thước ảnh đầu vào
            save_period=5,
            device='cpu',
            loss_type = 'siou')  