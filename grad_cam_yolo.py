import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys, copy
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
import glob

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (top, bottom, left, right)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        if self.model.end2end:
            logits_ = result[:, :, 4:]
            boxes_ = result[:, :, :4]
            sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
            return logits_[0][indices[0]], boxes_[0][indices[0]]
        elif self.model.task == 'detect':
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'segment':
            logits_ = result[0][:, 4:4 + self.model.nc]
            boxes_ = result[0][:, :4]
            mask_p, mask_nm = result[1][2].squeeze(), result[1][1].squeeze().transpose(1, 0)
            c, h, w = mask_p.size()
            mask = (mask_nm @ mask_p.view(c, -1))
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], mask[indices[0]]
        elif self.model.task == 'pose':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            poses_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(poses_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'obb':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            angles_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(angles_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'classify':
            return result[0]
 
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        if self.model.task == 'detect':
            post_result, pre_post_boxes = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes]]
        elif self.model.task == 'segment':
            post_result, pre_post_boxes, pre_post_mask = self.post_process(model_output)
            return [[post_result, pre_post_boxes, pre_post_mask]]
        elif self.model.task == 'pose':
            post_result, pre_post_boxes, pre_post_pose = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_pose]]
        elif self.model.task == 'obb':
            post_result, pre_post_boxes, pre_post_angle = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_angle]]
        elif self.model.task == 'classify':
            data = self.post_process(model_output)
            return [data]

    def release(self):
        for handle in self.handles:
            handle.remove()

class yolo_detect_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end, target_class=None) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end
        self.target_class = target_class  # 🎯 Lớp cần tính Grad-CAM (None nếu muốn tính cho tất cả)
        print(f'Grad-CAM target class: {self.target_class}')

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []

        for i in trange(int(post_result.size(0) * self.ratio)):
            # Bỏ qua box nếu dưới ngưỡng confidence
            if (self.end2end and float(post_result[i, 0]) < self.conf) or \
               (not self.end2end and float(post_result[i].max()) < self.conf):
                break

            # 🎯 Lọc theo target_class nếu được chỉ định
            if self.target_class is not None:
                if not self.end2end:
                    class_index = post_result[i].argmax().item()
                else:
                    class_index = 0  # end2end chỉ có 1 class score tại post_result[i, 0]

                if class_index != self.target_class:
                    continue  # Bỏ qua nếu không phải class mong muốn
                
            # 🧠 Tính Grad-CAM theo loại đầu ra mong muốn
            if self.ouput_type in ['class', 'all']:
                if self.end2end:
                    result.append(post_result[i, 0])
                else:
                    result.append(post_result[i].max())

            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])

        return sum(result)
    
class yolo_segment_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_mask = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'segment' or self.ouput_type == 'all':
                result.append(pre_post_mask[i].mean())
        return sum(result)

class yolo_pose_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_pose = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'pose' or self.ouput_type == 'all':
                result.append(pre_post_pose[i].mean())
        return sum(result)

class yolo_obb_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_angle = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'obb' or self.ouput_type == 'all':
                result.append(pre_post_angle[i])
        return sum(result)

class yolo_classify_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        return data.max()

class yolo_heatmap:
    # ------------------- THAY ĐỔI 1: Thêm 'show_overlay_image=True' vào __init__ -------------------
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize, task, img_size, target_class=None, show_overlay_image=True):
        device = torch.device(device)
        model_yolo = YOLO(weight)
        model_names = model_yolo.names
        print(f'model class info:{model_names}')
        model = copy.deepcopy(model_yolo.model)
        model.to(device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        
        model.task = task
        if not hasattr(model, 'end2end'):
            model.end2end = False
        
        if task == 'detect':
            target = yolo_detect_target(backward_type, conf_threshold, ratio, model.end2end, target_class)
        elif task == 'segment':
            target = yolo_segment_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'pose':
            target = yolo_pose_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'obb':
            target = yolo_obb_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'classify':
            target = yolo_classify_target(backward_type, conf_threshold, ratio, model.end2end)
        else:
            raise Exception(f"not support task({task}).")
        
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
        
        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        
        # Lưu tham số mới
        self.show_overlay_image = show_overlay_image
        self.__dict__.update(locals())
    
    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2) # 绘制检测框
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)  # 绘制类别、置信度
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
    
    # ------------------- THAY ĐỔI 2: Cập nhật hàm process -------------------
    def process(self, img_path, save_path):
        # img process
        try:
            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        except:
            print(f"Warning... {img_path} read failure.")
            return
        img, _, (top, bottom, left, right) = letterbox(img, new_shape=(self.img_size, self.img_size), auto=True) # 如果需要完全固定成宽高一样就把auto设置为False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        print(f'tensor size:{tensor.size()}')
        
        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            print(f"Warning... self.method(tensor, [self.target]) failure.")
            return
        
        grayscale_cam = grayscale_cam[0, :]
        
        # --- LOGIC MỚI BẮT ĐẦU TỪ ĐÂY ---
        if self.show_overlay_image:
            # --- A: Hiển thị overlay (ảnh gốc + heatmap) (Hành vi cũ) ---
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            
            # Lấy dự đoán (chỉ cần khi show_result=True hoặc renormalize=True)
            pred = self.model_yolo.predict(tensor, conf=self.conf_threshold, iou=0.7)[0]
            
            if self.renormalize and self.task in ['detect', 'segment', 'pose']:
                cam_image = self.renormalize_cam_in_bounding_boxes(pred.boxes.xyxy.cpu().detach().numpy().astype(np.int32), img, grayscale_cam)
            
            if self.show_result:
                cam_image = pred.plot(img=cam_image,
                                      conf=True, # 显示置信度
                                      font_size=None, # 字体大小，None为根据当前image尺寸计算
                                      line_width=None, # 线条宽度，None为根据当前image尺寸计算
                                      labels=False, # 显示标签
                                      )
        else:
            # --- B: Chỉ hiển thị heatmap (Theo yêu cầu mới) ---
            # Chuyển grayscale_cam (float 0-1) thành ảnh màu (uint8 0-255)
            cam_uint8 = np.uint8(255 * grayscale_cam)
            cam_image = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB) # Chuyển BGR (cv2) -> RGB (PIL)
        # --- LOGIC MỚI KẾT THÚC ---
            
        
        # Điểm chung: Cả cam_image (A) và (B) đều là mảng numpy uint8 (0-255)
        
        # 去掉padding边界
        cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)
    

    def __call__(self, img_path, save_path_or_dir):
        # Hàm phụ: tạo tên heatmap_XXXX.png tiếp theo
        def get_next_filename(save_dir):
            existing_files = glob.glob(os.path.join(save_dir, "heatmap_*.png"))
            if existing_files:
                indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_files]
                next_index = max(indices) + 1
            else:
                next_index = 1
            return os.path.join(save_dir, f"heatmap_{next_index:04d}.png")

        # Trường hợp 1: nếu là file cụ thể (.png)
        if os.path.splitext(save_path_or_dir)[1].lower() == '.png':
            save_dir = os.path.dirname(save_path_or_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = save_path_or_dir  # dùng đúng tên file
        else:
            # Trường hợp 2: là thư mục
            save_dir = os.path.abspath(save_path_or_dir)
            os.makedirs(save_dir, exist_ok=True)

        # Xử lý thư mục ảnh hoặc ảnh đơn
        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                img_full_path = os.path.join(img_path, img_path_)
                save_path = get_next_filename(save_dir) if os.path.isdir(save_path_or_dir) else save_path_or_dir
                self.process(img_full_path, save_path)
        else:
            if os.path.isdir(save_path_or_dir):
                save_path = get_next_filename(save_dir)
            else:
                save_path = save_path_or_dir
            self.process(img_path, save_path)

def get_params():
    params = {
        'weight': r'D:\CIoU_Fish_v4.5\train\weights\best.pt', # 现在只需要指定权重即可,不需要指定cfg
        'device': 'cpu',
        'method': 'GradCAM', # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM
        'layer': [16, 19, 22],
        'backward_type': 'all', # detect:<class, box, all> segment:<class, box, segment, all> pose:<box, keypoint, all> obb:<box, angle, all> classify:<all>
        'conf_threshold': 0.4, # 0.2
        'ratio': 0.02, # 0.02-0.1
        'show_result': False, # (Giữ nguyên) Điều khiển việc VẼ BOX, chỉ hoạt động nếu show_overlay_image=True
        'renormalize': False, # 需要把热力图限制在框内请设置为True(仅对detect,segment,pose有效)
        'task':'detect', # 任务(detect,segment,pose,obb,classify)
        'img_size': 640, # 图像尺寸
    }
    return params
    
def get_param_list():
    base = {
        'weight': r'D:\CIoU_Fish_v4.5\train\weights\best.pt',
        'device': 'cpu',
        'method': 'GradCAM',
        'backward_type': 'all',
        'conf_threshold': 0.4,
        'ratio': 0.02,
        'show_result': False, # (Giữ nguyên)
        'renormalize': False,
        'task': 'detect',
        'img_size': 640,
    }
    layers = [16, 19, 22]
    return [{**base, 'layer': [l]} for l in layers]

# ------------------- THAY ĐỔI 3: Cập nhật khối main -------------------
if __name__ == '__main__':

    # --- 1. Thiết lập đường dẫn ---
    input_base_dir = r"D:\temp\gra" 
    output_base_dir = r"D:\temp\grad_cam_YOLOv11_ONLY_HEATMAP" # Đổi tên thư mục output
    
    # === TÙY CHỌN MỚI ===
    # True: Chồng heatmap lên ảnh gốc (hành vi cũ)
    # False: Chỉ hiển thị heatmap đã tô màu (theo yêu cầu của bạn)
    SHOW_OVERLAY_ON_IMAGE = False 
    # =====================

    if os.path.exists(output_base_dir):
        print(f"Đang xóa thư mục kết quả cũ: {output_base_dir}")
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)


    # --- 2. Lấy mapping Tên Lớp -> Index ---
    base_params = get_params()
    model_weight = base_params['weight']
    
    print("Đang tải model để lấy danh sách tên lớp...")
    try:
        temp_model = YOLO(model_weight)
        class_names_map = temp_model.names 
        class_name_to_index_map = {name: idx for idx, name in class_names_map.items()}
        print(f"Đã tìm thấy {len(class_name_to_index_map)} lớp. Mapping: {class_name_to_index_map}")
        del temp_model 
    except Exception as e:
        print(f"LỖI: Không thể tải model từ '{model_weight}' để lấy tên lớp. Lỗi: {e}", file=sys.stderr)
        sys.exit(1) 
        
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    # --- 3. Duyệt qua các thư mục lớp trong input_base_dir ---
    if not os.path.exists(input_base_dir):
        print(f"LỖI: Thư mục input '{input_base_dir}' không tồn tại!", file=sys.stderr)
        sys.exit(1)

    try:
        class_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
    except Exception as e:
        print(f"LỖI: Không thể đọc thư mục {input_base_dir}. Lỗi: {e}", file=sys.stderr)
        sys.exit(1)

    if not class_dirs:
        print(f"Không tìm thấy thư mục lớp nào trong: {input_base_dir}")
    
    for class_name_str in class_dirs:
        # --- 4. Lấy Target Class Index ---
        if class_name_str not in class_name_to_index_map:
            print(f"Cảnh báo: Tên thư mục '{class_name_str}' không khớp với bất kỳ lớp nào trong model. Bỏ qua...")
            continue
            
        target_class_index = class_name_to_index_map[class_name_str]
        print(f"\n--- 🎯 Đang xử lý lớp: '{class_name_str}' (Index: {target_class_index}) ---")
        
        current_class_input_dir = os.path.join(input_base_dir, class_name_str)
        current_class_output_dir = os.path.join(output_base_dir, class_name_str)
        os.makedirs(current_class_output_dir, exist_ok=True)
        
        # --- 5. Duyệt qua từng ảnh trong thư mục lớp ---
        image_files = [f for f in os.listdir(current_class_input_dir) if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f" 	Không tìm thấy ảnh nào trong: {current_class_input_dir}")
            continue

        print(f" 	Tìm thấy {len(image_files)} ảnh. Bắt đầu chạy Grad-CAM...")
        
        for img_filename in tqdm(image_files, desc=f" 	Lớp {class_name_str}", unit="ảnh"):
            img_path = os.path.join(current_class_input_dir, img_filename)
            img_basename = os.path.splitext(img_filename)[0]
            img_save_subdir = os.path.join(current_class_output_dir, img_basename)
            os.makedirs(img_save_subdir, exist_ok=True)
            
            # Copy ảnh gốc vào thư mục kết quả để tiện so sánh
            # (Bạn có thể tắt dòng này nếu muốn)
            shutil.copy(img_path, os.path.join(img_save_subdir, "original_" + img_filename))
            
            # --- 6. Chạy Grad-CAM (Sử dụng tùy chọn mới) ---
            try:
                # Chạy cho từng layer
                for params in get_param_list():
                    # Truyền target_class VÀ show_overlay_image
                    model = yolo_heatmap(**params, 
                                         target_class=target_class_index, 
                                         show_overlay_image=SHOW_OVERLAY_ON_IMAGE)
                    
                    layer_idx = params['layer'][0]
                    save_file_path = os.path.join(img_save_subdir, f'cls_{target_class_index}_layer{layer_idx:02d}.png')
                    model(img_path, save_file_path)

                # Chạy cho 'mean' (all layers)
                model_mean = yolo_heatmap(**get_params(), 
                                          target_class=target_class_index, 
                                          show_overlay_image=SHOW_OVERLAY_ON_IMAGE)
                
                mean_save_path = os.path.join(img_save_subdir, f'mean_cls_{target_class_index}.png')
                model_mean(img_path, mean_save_path)
                
            except Exception as e:
                print(f"LỖI khi chạy Grad-CAM cho ảnh {img_path}: {e}", file=sys.stderr)
                
    print("\n--- Hoàn tất! 🚀 ---")
    print(f"Tất cả kết quả Grad-CAM đã được lưu tại: {output_base_dir}")