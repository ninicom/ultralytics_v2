import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Model': [
        'SDD', 
        'Faster R-CNN', 
        'YOLOv8', 
        'YOLOv10', 
        'YOLOv11', 
        'DETR', 
        'FishDet-YOLO', 
        'YOLO-SCFB'
    ],
    'FPS': [
        22, 
        2, 
        87, 
        73, 
        103, 
        20, 
        68, 
        89
    ],
    'mAP50': [
        75.03, 
        81.28, 
        88.12, 
        91.44, 
        94.39, 
        86.34, 
        92.5, 
        97.58
    ]
}

df = pd.DataFrame(data)

print("--- Dữ liệu dưới dạng Bảng (Pandas DataFrame) ---")
print(df)
print("\n" + "="*50 + "\n")

plt.figure(figsize=(12, 7))

for i, row in df.iterrows():
    model_name = row['Model']
    fps_val = row['FPS']
    map50_val = row['mAP50']
    
    # Khởi tạo các thuộc tính mặc định
    marker_style = 'o'
    marker_size = 100
    marker_color = 'blue'
    edge_color = 'w'
    font_weight = 'normal' # Mặc định là normal (không in đậm)

    if model_name == 'YOLO-SCFB':
        marker_style = '*' # Ngôi sao
        marker_size = 200  # Kích thước lớn hơn
        marker_color = 'red'
        edge_color = 'black'
        font_weight = 'bold' # Đặt in đậm cho YOLO-SCFB
        
    plt.scatter(
        fps_val, 
        map50_val, 
        s=marker_size, 
        alpha=0.9, 
        c=marker_color, 
        edgecolors=edge_color, 
        marker=marker_style, 
        label=model_name
    )
    
    plt.annotate(
        model_name,
        (fps_val, map50_val),
        textcoords="offset points",
        xytext=(5, 5),
        ha='left',
        fontsize=10, # Có thể điều chỉnh cỡ chữ nếu muốn
        fontweight=font_weight # Áp dụng thuộc tính in đậm
    )

plt.title('So sánh Hiệu suất Mô hình: Tốc độ (FPS) vs. Độ chính xác (mAP@0.5)', fontsize=16)
plt.xlabel('Tốc độ xử lý (FPS) - Càng cao càng nhanh $\longrightarrow$', fontsize=12)
plt.ylabel('Độ chính xác (mAP@0.5) (%) - Càng cao càng tốt $\longrightarrow$', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

print("Đang hiển thị biểu đồ so sánh FPS và mAP50 với YOLO-SCFB in đậm và là hình ngôi sao...")
plt.show()