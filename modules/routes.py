# File: modules/routes.py

from flask import render_template, request, redirect, url_for, session, flash, jsonify, Response
import os
from werkzeug.utils import secure_filename
import datetime

# Import your custom auth functions
from modules.auth import login_user, register_user, forget_password, logout_user
# Import your new detection logic
from modules.detection import generate_security_frames

# This stores what the user currently wants to detect
USER_DETECTION_STATE = {
    "FIRE": True,
    "WEAPON": True,
    "PERSON": True,
    "video_terminated": False,
    "feed_source": 0 # Default webcam
}

def register_routes(app, db, users_collection, bcrypt):
    alerts_collection = db["alerts"] 

    # ==========================================
    # --- AUTHENTICATION ROUTES ---
    # ==========================================

    @app.route('/login', methods=['POST'])
    def login():
        return login_user(users_collection, bcrypt)

    @app.route('/register', methods=['POST'])
    def register():
        return register_user(users_collection, bcrypt)

    @app.route('/logout', methods=['POST'])
    def logout():
        return logout_user()

    @app.route('/forget-password', methods=['POST'])
    def forget_password_route():
        return forget_password(users_collection)

    # ==========================================
    # --- DASHBOARD & PAGE ROUTES ---
    # ==========================================

    @app.route('/dashboard')
    def dashboard():
        if 'username' not in session:
            return redirect(url_for('index'))
        # Going to front.html per your request
        return render_template('front.html', username=session['username'])
        
    @app.route('/auth_area_detection')
    def auth_area_detection():
        if 'username' not in session:
            flash('You need to log in first!', 'danger')
            return redirect(url_for('index'))
        return render_template('normal_detection.html', username=session['username'])
    
    @app.route('/normal_detection')
    def normal_detection():
        if 'username' not in session:
            flash('You need to log in first!', 'danger')
            return redirect(url_for('index'))
        return render_template('normal_detection.html', username=session['username'])

    # ==========================================
    # --- VIDEO FEED & CONTROLS ---
    # ==========================================

    @app.route('/video_feed')
    def video_feed():
        USER_DETECTION_STATE["video_terminated"] = False
        USER_DETECTION_STATE["username"] = session.get('username', 'Guest')
        USER_DETECTION_STATE["email"] = session.get('email', 'viveksapkale022@gmail.com') # Assuming you store email in session!
        
        video_path = app.config.get('CURRENT_VIDEO', 'demo_browser/demo1.mp4')
        return Response(generate_security_frames(video_path, USER_DETECTION_STATE, alerts_collection),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/camera_feed')
    def camera_feed():
        USER_DETECTION_STATE["video_terminated"] = False
        USER_DETECTION_STATE["username"] = session.get('username', 'Guest')
        USER_DETECTION_STATE["email"] = session.get('email', 'viveksapkale022@gmail.com')
        flash('camera_feed started', 'success')
        return Response(generate_security_frames(0, USER_DETECTION_STATE),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/cctv_feed')
    def cctv_feed():
        USER_DETECTION_STATE["video_terminated"] = False
        USER_DETECTION_STATE["username"] = session.get('username', 'Guest')
        USER_DETECTION_STATE["email"] = session.get('email', 'viveksapkale022@gmail.com')
        return Response(generate_security_frames(1, USER_DETECTION_STATE),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/terminate', methods=['POST'])
    def terminate():
        USER_DETECTION_STATE["video_terminated"] = True
        return jsonify({"status": "terminated"})

    # ==========================================
    # --- MODEL TOGGLES & ALERTS ---
    # ==========================================

    @app.route('/toggle_model', methods=['POST'])
    def toggle_model():
        data = request.json
        model_name = data.get("model") # "FIRE", "WEAPON", or "PERSON"
        state = data.get("state")      # True or False
        
        if model_name in USER_DETECTION_STATE:
            USER_DETECTION_STATE[model_name] = state
            
        return jsonify({"status": "updated", "state": USER_DETECTION_STATE})

    @app.route('/get_alerts', methods=['GET'])
    def get_alerts():
        if 'username' not in session:
            return jsonify({"error": "Unauthorized"}), 401
            
        # Fetch only alerts belonging to the logged-in user
        user_alerts = list(alerts_collection.find(
            {"username": session['username']}, 
            {"_id": 0} 
        ).sort("timestamp", -1).limit(20))
        
        return jsonify(user_alerts)




    # ==========================================
    # --- VIDEO UPLOAD ROUTE ---
    # ==========================================
    @app.route('/upload_video', methods=['GET', 'POST'])
    def upload_video():
        if 'username' not in session:
            flash('You need to log in first!', 'danger')
            return redirect(url_for('index'))

        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file selected!', 'warning')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No selected file!', 'warning')
                return redirect(request.url)
                
            if file:
                filename = secure_filename(file.filename)
                # Save it to the 'uploads' folder we created in app.py
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Tell the app to use this new video!
                app.config['CURRENT_VIDEO'] = filepath
                USER_DETECTION_STATE["video_terminated"] = False
                
                flash('Video uploaded successfully! Analyzing now...', 'success')
                # Send them back to the dashboard to see the results
                return redirect(url_for('normal_detection')) 
                
        # If it's a GET request, just show the upload page
        return render_template('fire_upload.html')