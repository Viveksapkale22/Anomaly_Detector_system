# File: app.py

import os
from flask import Flask, render_template
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from modules.routes import register_routes

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Security & Database ---
bcrypt = Bcrypt(app)
MONGO_URI = "mongodb+srv://cluster0.mcjuw.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCertificateKeyFile=r"templates\X509-cert-2052612786362920307.pem",
)
db = client["UserDB"]
users_collection = db["users"]

# --- Register Routes (Exactly 4 Arguments!) ---
register_routes(app, db, users_collection, bcrypt)

# --- Main Route ---
@app.route('/')
def index():
    return render_template('front.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)