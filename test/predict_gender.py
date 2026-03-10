import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import torch.nn.functional as F
from ultralytics import YOLO

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load gender model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load("best_gender_model.pth", map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# YOLO Face model
yolo = YOLO("model.pt")

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    results = yolo(frame)

    for r in results:

        boxes = r.boxes.xyxy

        for box in boxes:

            x1,y1,x2,y2 = map(int, box)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            img = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img)
                probs = F.softmax(outputs, dim=1)
                conf, pred = torch.max(probs,1)

            label = "Male" if pred.item()==0 else "Female"
            confidence = conf.item()*100

            text = f"{label} {confidence:.1f}%"

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(frame,text,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,255,0),2)

    cv2.imshow("YOLOv8 Face + Gender Detection", frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()