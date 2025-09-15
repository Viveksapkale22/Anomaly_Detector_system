# File: modules/utils.py

import cv2
import numpy as np
import time
import winsound
import os
from deepface import DeepFace
import smtplib
from config import Config


SENDER_EMAIL = Config.SENDER_EMAIL
SENDER_PASSWORD = Config.SENDER_PASSWORD

def send_alert_email(email, frame, person_id, person_count , upload_folder="uploads" ):
    import cv2
    import time
    from email.message import EmailMessage

    timestamp = int(time.time())
    filename = f"alert_{person_id}_{timestamp}.jpg"
    filepath = f"{upload_folder}/{filename}"
    cv2.imwrite(filepath, frame)

    msg = EmailMessage()
    msg["Subject"] = "Security Alert: Intrusion Detected"
    msg["From"] = SENDER_EMAIL
    msg["To"] = email
    msg.set_content(f"A person has been detected in a restricted area. the number of persons count {person_count} See attached image.")

    with open(filepath, "rb") as f:
        img_data = f.read()
        msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=filename)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print(f"Alert email sent to {email}")
    except Exception as e:
        print(f"Failed to send alert email: {e}")

# Allowed video formats
ALLOWED_EXTENSIONS = {'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def play_alert():
    """Play alert sound (Windows only)."""
    winsound.Beep(1000, 500)

def boxes_intersect(box1, box2):
    """Check if two bounding boxes intersect."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

def detect_motion(frame, bg_subtractor=None):
    """Detect motion using background subtraction or simple frame change."""
    if bg_subtractor is not None:
        fg_mask = bg_subtractor.apply(frame)
        thresh = cv2.threshold(fg_mask, 30, 255, cv2.THRESH_BINARY)[1]
        return np.count_nonzero(thresh) > 500
    else:
        # Simple motion detection fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        if not hasattr(detect_motion, "previous_frame"):
            detect_motion.previous_frame = blurred
            return False

        # ✅ Ensure dimensions match before absdiff
        if blurred.shape != detect_motion.previous_frame.shape:
            detect_motion.previous_frame = cv2.resize(
                detect_motion.previous_frame,
                (blurred.shape[1], blurred.shape[0])
            )

        delta = cv2.absdiff(detect_motion.previous_frame, blurred)
        detect_motion.previous_frame = blurred
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]

        return np.count_nonzero(thresh) > 500



def count_persons(model, frame):
    """Count number of detected persons (class ID 0 assumed to be person)."""
    results = model(frame)
    count = 0
    for result in results:
        for box in result.boxes:
            if int(box.cls[0].item()) == 0:
                count += 1
    return count



def analyze_gender(person_id, face_crop, global_state):
    """Run DeepFace for gender detection."""
    global_state['processing'][person_id] = True
    try:
        result = DeepFace.analyze(face_crop, actions=['gender'], enforce_detection=False)
        gender_dict = result[0]['gender']
        gender = max(gender_dict, key=gender_dict.get)
        global_state['gender_labels'][person_id] = gender
    except Exception as e:
        print(f"[DeepFace] Gender detection error for ID {person_id}:", e)
        global_state['gender_labels'][person_id] = "Unknown"
    global_state['processing'][person_id] = False

def save_frame_and_get_path(frame, person_id, upload_folder="uploads"):
    """Save alert image frame and return the file path."""
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    timestamp = int(time.time())
    filename = f"alert_{person_id}_{timestamp}.jpg"
    filepath = os.path.join(upload_folder, filename)
    cv2.imwrite(filepath, frame)
    return filepath
