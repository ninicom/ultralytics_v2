from ultralytics import YOLO

model =YOLO(r'D:\Fish_v3+disease_1_train YoloV11\train\weights\best.pt')

results = model(source='D:/test_grad/5.jpg', show=True, save=True)