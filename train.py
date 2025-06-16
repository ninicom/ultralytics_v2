import sys
import os

# Trỏ trực tiếp đến thư mục chứa YOLO
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ultralytics/ultralytics')))

from engine.model import YOLO  # import trực tiếp
