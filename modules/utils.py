import time
import cv2
import os
import smtplib
import threading
from email.message import EmailMessage

# Import your Config file (Assuming config.py is in the root directory)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Create a folder to store the alert images if it doesn't exist
ALERTS_DIR = os.path.join('static', 'alerts')
os.makedirs(ALERTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# PULLING SETTINGS DIRECTLY FROM config.py
# ---------------------------------------------------------
SENDER_EMAIL = Config.SENDER_EMAIL
SENDER_PASSWORD = Config.SENDER_PASSWORD
ALERT_COOLDOWN_SECONDS = Config.ALERT_INTERVAL 
# ---------------------------------------------------------

LAST_ALERT_TIME = {
    "FIRE": 0,
    "WEAPON": 0,
    "PERSON": 0
}

def send_alert_email(to_email, subject, body, image_path):
    """Helper function to send an email with an image attachment."""
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg.set_content(body)

        # Attach the image
        with open(image_path, 'rb') as img:
            img_data = img.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

        # Send the email securely
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print(f"[EMAIL SUCCESS] Alert sent securely to {to_email}")
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email: {e}")

def handle_threat_alert(alert_type, frame, username, user_email, alerts_collection, count=0):
    """
    Handles saving the image, logging to MongoDB, and sending the email ASYNCHRONOUSLY.
    """
    global LAST_ALERT_TIME
    current_time = time.time()
    
    if current_time - LAST_ALERT_TIME[alert_type] > ALERT_COOLDOWN_SECONDS:
        LAST_ALERT_TIME[alert_type] = current_time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        safe_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. Save the Image with Bounding Boxes
        image_filename = f"{alert_type}_{username}_{safe_timestamp}.jpg"
        image_path = os.path.join(ALERTS_DIR, image_filename)
        cv2.imwrite(image_path, frame)
        
        # 2. Prepare the Anomaly Text
        if alert_type == "PERSON":
            anomaly_text = f"High Crowd Detected! {count} persons identified."
        else:
            anomaly_text = f"{alert_type} Threat Detected in camera feed."

        # 3. Save to MongoDB Backend Structure
        alert_document = {
            "username": username,
            "email": user_email,
            "image_path": f"/static/alerts/{image_filename}", 
            "timestamp": timestamp,
            "anomaly_type": alert_type,
            "details": anomaly_text
        }
        
        if alerts_collection is not None:
            alerts_collection.insert_one(alert_document)
            
            # Print logic tied to your debug config!
            if getattr(Config, 'DEBUG_ALERTS', True):
                print(f"💾 [DATABASE] Alert saved to MongoDB for {username}")

        # 4. Send the Email ASYNCHRONOUSLY (Background Thread)
        email_subject = f"🚨 URGENT: {alert_type} Detected!"
        email_body = f"Hello {username},\n\nOur system has detected an anomaly at {timestamp}.\n\nDetails: {anomaly_text}\n\nPlease check the attached image for visual confirmation."
        
        # This starts the email process in the background so the video doesn't freeze!
        email_thread = threading.Thread(target=send_alert_email, args=(user_email, email_subject, email_body, image_path))
        email_thread.start()

        print(f"🔥🔫 [ALERT LOGGED] {timestamp} - {alert_type} DETECTED! 🚨")


def draw_smooth_boxes(frame, boxes, color, label):
    """Helper function to cleanly draw boxes on the frame."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label)*12, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame