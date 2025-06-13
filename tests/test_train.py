import torch
import torch.nn as nn
from ultralytics import YOLO
from tabulate import tabulate

# === Hàm tính receptive field ===
def compute_rf_stride(model, input_size=(3, 640, 640)):
    rf = 1  # Receptive field
    j = 1   # Jump (cumulative stride)
    start = 0.5  # Center position
    rfs = {}  # Dictionary to store receptive fields

    def register_hook(name):
        def hook(module, input, output):
            nonlocal rf, j, start
            if isinstance(module, nn.Conv2d):
                k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                p = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                d = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
                rf_new = rf + (k - 1) * j * d
                start = start + ((k - 1) / 2 - p) * j
                j_new = j * s
                rf, j = rf_new, j_new
            elif isinstance(module, nn.MaxPool2d):
                k = module.kernel_size if isinstance(module.kernel_size, tuple) else module.kernel_size
                s = module.stride if isinstance(module.stride, tuple) else module.stride
                p = module.padding if isinstance(module.padding, tuple) else module.padding
                k = k[0] if isinstance(k, tuple) else k
                s = s[0] if isinstance(s, tuple) else s
                p = p[0] if isinstance(p, tuple) else p
                rf_new = rf + (k - 1) * j
                start = start + ((k - 1) / 2 - p) * j
                j_new = j * s
                rf, j = rf_new, j_new
            rfs[name] = rf
        return hook

    # Register hooks for all modules
    for name, module in model.named_modules():
        module.register_forward_hook(register_hook(name))

    # Forward pass with dummy input
    with torch.no_grad():
        model(torch.zeros(1, *input_size).to(next(model.parameters()).device))

    return rfs

# === In bảng giống summary ===
def print_model_summary_with_rf(model, input_size=(3, 640, 640)):
    rfs = compute_rf_stride(model, input_size)
    layers = model.model if hasattr(model, 'model') else model  # Handle YOLO model or direct nn.Module
    table = []

    for i, module in enumerate(layers):
        mtype = f"{type(module).__module__}.{type(module).__name__}"
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        # Extract arguments safely
        if hasattr(module, 'args') and module.args:
            args = module.args
        else:
            # Fallback to string representation, extracting content within parentheses
            args_str = repr(module).split('(', 1)[-1].rsplit(')', 1)[0]
            args = args_str if args_str else '-'

        # Get receptive field for the module
        name = f'model.{i}' if hasattr(model, 'model') else f'{i}'
        rf_val = rfs.get(name, '-')

        # Get 'from' and 'n' attributes if available
        from_idx = getattr(module, 'f', '-')
        n_repeats = getattr(module, 'n', 1)

        table.append([
            i,
            from_idx,
            n_repeats,
            n_params,
            mtype,
            rf_val
        ])

    # Print table
    headers = ['idx', 'from', 'n', 'params', 'module', 'receptive_field']
    print(f"\nYOLOv8 summary with Receptive Field (RF): {sum(p.numel() for p in model.parameters())} parameters")
    print(tabulate(table, headers=headers, tablefmt='grid'))

# === Sử dụng ===
if __name__ == '__main__':
    model = YOLO("yolov8n.pt").model  # Replace with your model path, e.g., 'yolov8n.pt'
    print_model_summary_with_rf(model)