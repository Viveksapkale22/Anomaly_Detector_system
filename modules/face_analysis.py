# face_analysis.py

import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(weights_path):
    """Build MobileNetV2 gender classification model and load weights safely."""
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 2)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Model weights not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)

    # ✅ Handle both cases: full checkpoint or raw state_dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            print("ℹ️ Loading weights from checkpoint['model_state_dict']")
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("ℹ️ Loading weights directly from state_dict-like dict")
            model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"❌ Unexpected checkpoint type: {type(checkpoint)}")

    model.to(device)
    model.eval()
    print("✅ Gender model loaded successfully")
    return model

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_gender(face_crop, model):
    """Analyze gender of a cropped face using the given model."""
    try:
        print(f"[DEBUG] model type: {type(model)}")
        if not isinstance(model, torch.nn.Module):
            print("❌ Model is not a torch.nn.Module! Please check loading.")
            return "Unknown"

        if face_crop is None or face_crop.size == 0:
            return "Unknown"

        img = Image.fromarray(face_crop[..., ::-1])  # BGR → RGB
        x_in = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x_in)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        return "Female" if probs[1] > 0.5 else "Male"

    except Exception as e:
        print(f"[GenderModel] Error: {e}")
        return "Unknown"
