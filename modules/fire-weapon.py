import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
from ultralytics import YOLO
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Assuming you have a config file, otherwise replace with actual strings
# from config import Config 

SENDER_EMAIL = Config.SENDER_EMAIL
SENDER_PASSWORD = Config.SENDER_PASSWORD

app = Flask(__name__)
app.secret_key = "super_secret_key"

# --- Upload Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------------
# UNIQUE STATE FOR FIRE/WEAPON
# ---------------------------------------------------------
FIRE_STATE = {
    "video_terminated": False,
    "feed_source": 0 # Default to webcam
}

# --- Email Configuration ---
EMAIL_SENDER = SENDER_EMAIL # Replace with Config.SENDER_EMAIL
EMAIL_PASSWORD = SENDER_PASSWORD    # Replace with Config.SENDER_PASSWORD
EMAIL_RECEIVER = "viveksapkale022@gmail.com"
ALERT_COOLDOWN = 60
last_alert_time = 0

# --- Load YOLO Model ---
model = YOLO(r'fire_weapon_model.pt') 

FIRE_CLASS_ID = 0
WEAPON_CLASS_ID = 1

# ---------------------------------------------------------
# EMAIL ALERT FUNCTION
# ---------------------------------------------------------
def send_alert_email(threat_type):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"URGENT ALERT: {threat_type} Detected!"

        body = f"A {threat_type} has been detected by the AI Enhanced Home system. Please check the camera feeds immediately."
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"[ALERT] Email sent successfully for: {threat_type}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

# ---------------------------------------------------------
# VIDEO PROCESSING PIPELINE
# ---------------------------------------------------------
def generate_frames(source):
    global last_alert_time
    cap = cv2.VideoCapture(source)
    
    while True:
        if FIRE_STATE["video_terminated"]:
            break
            
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        threat_detected = None

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf > 0.5:
                    if cls == FIRE_CLASS_ID:
                        threat_detected = "FIRE"
                        color = (0, 165, 255)
                    elif cls == WEAPON_CLASS_ID:
                        threat_detected = "WEAPON"
                        color = (0, 0, 255)
                    else:
                        continue
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"{threat_detected} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        current_time = time.time()
        if threat_detected and (current_time - last_alert_time > ALERT_COOLDOWN):
            last_alert_time = current_time
            threading.Thread(target=send_alert_email, args=(threat_detected,)).start()

        if threat_detected:
            cv2.putText(frame, "THREAT DETECTED - ALERT SENT", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ---------------------------------------------------------
# UNIQUE ROUTES FOR FIRE/WEAPON
# ---------------------------------------------------------
@app.route('/fire_dashboard')
def fire_dashboard():
    # Render a unique HTML file so it doesn't use the old detector's index
    return render_template('fire_index.html', username="Admin")

@app.route('/fire_video_feed')
def fire_video_feed():
    FIRE_STATE["video_terminated"] = False
    return Response(generate_frames(FIRE_STATE["feed_source"]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fire_camera_feed')
def fire_camera_feed():
    FIRE_STATE["feed_source"] = 0
    return redirect(url_for('fire_video_feed'))

@app.route('/fire_cctv_feed')
def fire_cctv_feed():
    FIRE_STATE["feed_source"] = "rtsp://your_camera_ip/stream"
    return redirect(url_for('fire_video_feed'))

@app.route('/fire_terminate', methods=['POST'])
def fire_terminate():
    FIRE_STATE["video_terminated"] = True
    return "OK", 200

@app.route('/fire_upload', methods=['GET', 'POST'])
def fire_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            FIRE_STATE["feed_source"] = filepath
            FIRE_STATE["video_terminated"] = False
            
            flash('Video successfully uploaded! Detection has started.')
            return redirect(url_for('fire_dashboard'))
        else:
            flash('Invalid file type. Please upload an MP4, AVI, or MOV.')
            return redirect(request.url)

    # Render a unique HTML file
    return render_template('fire_upload.html')

if __name__ == '__main__':
    # Run on a different port if running at the same time as the old detector
    app.run(debug=True, threaded=True, port=5001)