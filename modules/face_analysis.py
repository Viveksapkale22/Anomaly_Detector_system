# File: modules/face_analysis.py

import os
import cv2
import torch
from torchvision import models, transforms
from PIL import Image

# Default path to the best gender model weights (PyTorch)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "test", "best_gender_model.pth")

# Use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = (224, 224)

_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(weights_path: str = None):
    """Load the gender classification model.

    Uses EfficientNet-B0 architecture and loads weights from a .pth checkpoint.
    """
    path = weights_path or DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gender model weights not found: {path}")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(bgr_image):
    if bgr_image is None or bgr_image.size == 0:
        return None

    try:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = _transform(pil_img).unsqueeze(0).to(DEVICE)
        return tensor
    except Exception as e:
        print(f"[GenderModel] Preprocess error: {e}")
        return None


def analyze_gender(face_crop, model):
    """Return 'male' or 'female' using the supplied model."""
    if model is None:
        return "unknown"

    tensor = preprocess_image(face_crop)
    if tensor is None:
        return "unknown"

    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return "male" if pred.item() == 0 else "female"
