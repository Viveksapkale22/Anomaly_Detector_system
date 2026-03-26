import os
import torch
from ultralytics import YOLO

# Get absolute path to the Models directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# ==========================================
# 🚀 AUTOMATIC GPU/CPU DETECTION
# ==========================================
if torch.cuda.is_available():
    device = 'cuda'
    print(f"[SYSTEM] 🟢 NVIDIA GPU Detected! AI inference will run on: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("[SYSTEM] 🟡 No GPU Detected. AI inference falling back to: CPU")

try:
    # 1. Load the standalone Fire model (V2)
    fire_path = os.path.join(MODELS_DIR, 'Fire_yolov8_V2.pt')
    fire_model = YOLO(fire_path)
    fire_model.to(device) # Move model to GPU or CPU
    
    # 2. Load the standalone Weapon model (V3)
    weapon_path = os.path.join(MODELS_DIR, 'Weapon_yolov8_V3.pt')
    weapon_model = YOLO(weapon_path)
    weapon_model.to(device)
    
    # 3. Load the Person model (v2d)
    person_path = os.path.join(MODELS_DIR, 'Person_yolov8_v2d.pt')
    person_model = YOLO(person_path)
    person_model.to(device)
    
    print("[SYSTEM] All standalone AI Models loaded successfully.")
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to load models: {e}")
    # Set all to None so the app doesn't crash, it just won't detect
    fire_model = None
    weapon_model = None
    person_model = None