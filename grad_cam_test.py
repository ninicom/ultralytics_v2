import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Lớp GradCAM đã sửa
class YOLOv8GradCAM(nn.Module):
    def __init__(self, base_model, feature_module, target_layer):
        super(YOLOv8GradCAM, self).__init__()
        self.model = base_model
        self.feature_module = feature_module
        self.target_layer = target_layer

    def forward(self, x):
        # Forward pass up to the target layer
        for layer_name, layer in self.model._modules[self.feature_module]._modules.items():
            x = layer(x)
            if layer_name == self.target_layer:
                break

        # Grad-CAM calculation
        feature_map = x
        _, _, H, W = feature_map.size()
        cam = F.relu(feature_map)  # Apply ReLU
        cam = F.adaptive_avg_pool2d(cam, 1)  # Global average pooling
        cam = torch.mul(feature_map, cam)  # Element-wise multiplication
        cam = cam.sum(dim=1, keepdim=True)  # Channel-wise sum

        return cam

# Hàm overlay CAM lên ảnh gốc
def overlay_cam_on_image(img, cam):
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    return overlayed

# Tải ảnh
image_path = r'G:\Other computers\Máy tính xách tay của tôi\ảnh\meme\286454603_573403627693843_7524854635302619202_n.jpg'  # Thay bằng ảnh thật
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)

# Tải model YOLOv8
yolo = YOLO("yolov8n.pt")
model = yolo.model.eval()
results = yolo(source=image_path, show=True)  # Dự đoán để đảm bảo model đã sẵn sàng
# GradCAM module
cam_extractor = YOLOv8GradCAM(model, feature_module='model', target_layer='2')
with torch.no_grad():
    cam = cam_extractor(img_tensor)

# Overlay và hiển thị
result = overlay_cam_on_image(img_bgr, cam)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Grad-CAM")
plt.show()
