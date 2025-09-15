# File: modules/routes.py

from flask import render_template, request, redirect, url_for, session, flash, jsonify, Response
import threading
import time
import cv2
from modules.utils import allowed_file, play_alert, boxes_intersect, detect_motion, save_frame_and_get_path
from modules.auth import login_user, register_user, forget_password, logout_user
from modules.face_analysis import analyze_gender
from modules.detection import generate_frames
import os
from werkzeug.utils import secure_filename

bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

def register_routes(app, model, tracker, users_collection, bcrypt, user_data_store):
    """Registers all app routes."""

    global_state = {
        "restricted_area": None,
        "counting_enabled": False,
        "stop_video_flag": threading.Event(),
        "selected_gender": "both",
        "gender_labels": {},
        "processing": {},
        "last_alert_time": {},
        "person_tracks": {},
        "person_counter": 0,
        "detected_persons": [],
        "email": "viveksapkale022@gmail.com"
    }


    @app.route('/login', methods=['POST'])
    def login():
        return login_user(users_collection, bcrypt)

    @app.route('/register', methods=['POST'])
    def register():
        return register_user( users_collection, bcrypt)

    @app.route('/logout', methods=['POST'])
    def logout():
        return logout_user()

    @app.route('/forget-password', methods=['POST'])
    def forget_password_route():
        return forget_password(users_collection)

    @app.route('/main')
    def main():
        if 'username' not in session:
            flash('You need to log in first!')
            return redirect(url_for('index'))
        return render_template('front.html', username=session['username'])

    # @app.route('/upload', methods=['GET', 'POST'])
    # def upload():
    #     if request.method == 'POST':
    #         if 'file' not in request.files:
    #             flash('No file part')
    #             return redirect(request.url)
    #         file = request.files['file']
    #         if file.filename == '':
    #             flash('No selected file')
    #             return redirect(request.url)
    #         if file and allowed_file(file.filename):
    #             filename = 'uploaded_video.mp4'
    #             file_path = f"uploads/{filename}"
    #             file.save(file_path)
    #             app.config['CURRENT_VIDEO'] = file_path
    #             global_state['restricted_area'] = None
    #             flash('Video uploaded successfully!')
    #             return redirect(url_for('auth_area_detection'))
    #     return render_template('upload.html')


    ALLOWED_EXTENSIONS = {'mp4'}
 
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if 'username' not in session:
            flash('You must log in to upload videos.', 'warning')
            return redirect(url_for('index'))

        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part.', 'danger')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('No selected file.', 'danger')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename('uploaded_video.mp4')  # overwrite each time
                upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
                os.makedirs(upload_folder, exist_ok=True)

                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)

                app.config['CURRENT_VIDEO'] = file_path
                global_state['restricted_area'] = None
                global_state['stop_video_flag'].set()  # optional: stop running stream
                flash('Video uploaded successfully!', 'success')

                return redirect(url_for('auth_area_detection'))

        return render_template('upload.html')



    @app.route('/auth_area_detection')
    def auth_area_detection():
        if 'username' not in session:
            flash('You need to log in first!', 'danger')
            return redirect(url_for('index'))
        return render_template('auth_area_detection.html', username=session['username'])
    
    @app.route('/normal_detection')
    def normal_detection():
        if 'username' not in session:
            flash('You need to log in first!', 'danger')
            return redirect(url_for('index'))
        return render_template('normal_detection.html', username=session['username'])
 

    @app.route('/set_restricted_area', methods=['POST'])
    def set_restricted_area():
        x = int(request.form.get("x", 0))
        y = int(request.form.get("y", 0))
        w = int(request.form.get("w", 0))
        h = int(request.form.get("h", 0))
        global_state['restricted_area'] = (x, y, x + w, y + h)
        return '', 204

    @app.route('/update_gender', methods=['GET'])
    def update_gender():
        gender = request.args.get("gender", "both")
        global_state['selected_gender'] = gender
        return jsonify({"status": "updated", "selected_gender": gender})

    @app.route('/clear_detection_settings', methods=['POST'])
    def clear_detection_settings():
        global_state['restricted_area'] = None
        return "OK", 200

    @app.route('/toggle_counting', methods=['POST'])
    def toggle_counting():
        global_state['counting_enabled'] = not global_state['counting_enabled']
        return {"counting_enabled": global_state['counting_enabled']}

    @app.route('/video_feed')
    def video_feed():
        global_state['stop_video_flag'].clear()
        video_path = app.config.get('CURRENT_VIDEO', 'demo_browser/demo1.mp4')
        return Response(generate_frames(video_path, model, tracker, global_state),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/camera_feed')
    def camera_feed():
        global_state['stop_video_flag'].clear()
        flash('camera_feed', 'success')
        return Response(generate_frames(0, model, tracker, global_state),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/cctv_feed')
    def cctv_feed():
        global_state['stop_video_flag'].clear()
        return Response(generate_frames(1, model, tracker, global_state),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


    @app.route('/terminate_video_feed', methods=['POST'])
    def terminate_video_feed():
        global_state['stop_video_flag'].set()

        return redirect(url_for('auth_area_detection'))
    
    @app.route('/alert_status')
    def alert_status():
        # Placeholder logic: return fake alert status
        return jsonify({"alert": False, "message": "No alerts triggered"})

