# File: app.py

import os
from flask import Flask, render_template, render_template_string, request, redirect, url_for, session, flash, Response, jsonify
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from modules import auth, utils, model_loader
from modules.routes import register_routes

# Flask App
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

# Bcrypt for password hashing
bcrypt = Bcrypt(app)

# MongoDB Configuration
MONGO_URI = "mongodb+srv://cluster0.mcjuw.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCertificateKeyFile=r"templates\X509-cert-7398551624606348947.pem"
)
db = client["UserDB"]
users_collection = db["users"]

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
model = model_loader.load_yolo_model()
tracker = model_loader.init_tracker()

# Shared session data
user_data_store = {}

# Register all modular routes
register_routes(app, model, tracker, users_collection, bcrypt, user_data_store)

# Main route
@app.route('/')
def index():
    return render_template('front.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
