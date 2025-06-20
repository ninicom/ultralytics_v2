import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob

def save_gradcam(cam, save_dir, width=640, height=640):
    """
    Lưu tensor cam (heatmap) dưới dạng ảnh vào thư mục chỉ định với tên file theo thứ tự tăng dần,
    resize thành kích thước 320x240.
    
    Args:
        cam: Tensor heatmap từ Grad-CAM, kích thước (N, 1, H, W)
        save_dir: Đường dẫn thư mục để lưu file ảnh (e.g., './output/')
        width: Chiều rộng của ảnh đầu ra (mặc định 320)
        height: Chiều cao của ảnh đầu ra (mặc định 240)
    """
    save_dir = save_dir + '\img_20'
    # Đảm bảo thư mục đích tồn tại
    os.makedirs(save_dir, exist_ok=True)
    
    # Tìm tất cả các file heatmap_*.png trong thư mục
    existing_files = glob.glob(os.path.join(save_dir, "heatmap_*.png"))
    
    # Xác định số thứ tự tiếp theo
    if existing_files:
        # Lấy số thứ tự từ tên file (phần số sau "heatmap_")
        indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_files]
        next_index = max(indices) + 1
    else:
        next_index = 1
    
    # Tạo tên file với định dạng heatmap_XXXX.png
    save_path = os.path.join(save_dir, f"heatmap_{next_index:04d}.png")
    
    # Chuyển tensor sang numpy và loại bỏ batch dimension
    cam = cam.squeeze(0)  # Loại bỏ batch dimension: (1, 1, H, W) -> (1, H, W)
    cam = cam.squeeze(0)  # Loại bỏ channel dimension: (1, H, W) -> (H, W)
    cam = cam.detach().cpu().numpy()  # Chuyển sang numpy array
    
    # Chuẩn hóa heatmap về khoảng [0, 1]
    cam = np.maximum(cam, 0)  # Đảm bảo không có giá trị âm
    cam = cam / (cam.max() + 1e-10)  # Chuẩn hóa về [0, 1]
    
    # Chuyển đổi thành ảnh màu (RGB) sử dụng colormap
    cam = np.uint8(255 * cam)  # Chuyển về [0, 255]
    cam = cv2.resize(cam, (width, height))  # Resize thành 320x240
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # Áp dụng colormap (JET)
    
    # Lưu ảnh vào đường dẫn
    cv2.imwrite(save_path, cam)
    print(f"Heatmap đã được lưu tại: {save_path} với kích thước {width}x{height}")

def grad_cam(x):
    feature_map = x
    _, _, H, W = feature_map.size()
    cam = F.relu(feature_map)  # Apply ReLU
    cam = F.adaptive_avg_pool2d(cam, 1)  # Global average pooling
    cam = torch.mul(feature_map, cam)  # Element-wise multiplication
    cam = cam.sum(dim=1, keepdim=True)  # Channel-wise sum
    #save_gradcam(cam, r'D:/gradcam')

def grad_cam2(feature_map, outputs, class_idx):
    # Đảm bảo gradient được bật
    torch.enable_grad()
    
    # Lấy kích thước feature map
    _, _, H, W = feature_map.size()
    
    # Tính gradient của score class theo feature map
    score = outputs[0, class_idx]  # Lấy score của class cụ thể
    gradients = torch.autograd.grad(score, feature_map, retain_graph=True)[0]
    
    # Tính trọng số bằng cách lấy trung bình gradient theo chiều không gian
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    
    # Tạo heatmap bằng cách nhân feature map với trọng số và tổng hợp theo kênh
    cam = torch.mul(feature_map, weights).sum(dim=1, keepdim=True)
    cam = F.relu(cam)  # Chỉ giữ giá trị dương
    
    # Chuẩn hóa heatmap
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Upsample để khớp với kích thước ảnh gốc (nếu cần)
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    
    return cam