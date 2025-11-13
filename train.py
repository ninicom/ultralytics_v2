import os
import shutil
import sys
import cv2  # Cần thư viện này

# --- 1. Thiết lập đường dẫn ---
input_base_dir = r"D:\img_grad2" 
output_base_dir = r"D:\temp\resized_640x6402"
target_size = (640, 640) # (width, height)

# Xóa thư mục output cũ nếu tồn tại
if os.path.exists(output_base_dir):
    print(f"Đang xóa thư mục kết quả cũ: {output_base_dir}")
    shutil.rmtree(output_base_dir)
os.makedirs(output_base_dir, exist_ok=True)

# Các định dạng ảnh cần tìm
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# --- 2. Duyệt qua các thư mục lớp trong input_base_dir ---
if not os.path.exists(input_base_dir):
    print(f"LỖI: Thư mục input '{input_base_dir}' không tồn tại!", file=sys.stderr)
    sys.exit(1)

try:
    # Lấy danh sách các thư mục con (ví dụ: 'Red_Spot', 'Healthy', ...)
    class_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
except Exception as e:
    print(f"LỖI: Không thể đọc thư mục {input_base_dir}. Lỗi: {e}", file=sys.stderr)
    sys.exit(1)

if not class_dirs:
    print(f"Không tìm thấy thư mục lớp nào trong: {input_base_dir}")
    
print(f"Tìm thấy {len(class_dirs)} thư mục. Bắt đầu xử lý...")

# --- 3. Duyệt qua từng thư mục lớp ---
for class_name_str in class_dirs:
    print(f"\n--- 📁 Đang xử lý thư mục: '{class_name_str}' ---")
    
    # Đường dẫn thư mục input của lớp hiện tại
    current_class_input_dir = os.path.join(input_base_dir, class_name_str)
    
    # Tạo thư mục output tương ứng cho lớp này
    current_class_output_dir = os.path.join(output_base_dir, class_name_str)
    os.makedirs(current_class_output_dir, exist_ok=True)
    
    # Lấy danh sách file ảnh trong thư mục
    image_files = [f for f in os.listdir(current_class_input_dir) if f.lower().endswith(image_extensions)]

    # --- 4. Resize và Lưu ảnh ---
    print(f"   Tìm thấy {len(image_files)} ảnh. Đang resize về {target_size}...")
    processed_count = 0
    
    for image_name in image_files:
        try:
            input_path = os.path.join(current_class_input_dir, image_name)
            output_path = os.path.join(current_class_output_dir, image_name)
            
            # Đọc ảnh
            image = cv2.imread(input_path)
            if image is None:
                print(f"   Lỗi: Không thể đọc ảnh {input_path}. Bỏ qua.", file=sys.stderr)
                continue
            
            # Resize ảnh
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
            # Lưu ảnh
            cv2.imwrite(output_path, resized_image)
            processed_count += 1
            
        except Exception as e:
            print(f"   Lỗi khi xử lý ảnh {image_name}: {e}", file=sys.stderr)
            
    print(f"   Hoàn tất. Đã resize và lưu {processed_count}/{len(image_files)} ảnh.")

print("\n=== XỬ LÝ TẤT CẢ ĐÃ HOÀN TẤT ===")