from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO( r'D:\SCFB_SIoU_P2\best.pt')

# Define path to the image file
source = r"D:\OIP.jpg"

# Run inference on the source
results = model(source, save=True)  # list of Results objects