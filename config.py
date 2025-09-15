# config.py

import os

class Config:
    # Secret key for session and JWT
    JWT_SECRET = os.getenv("JWT_SECRET", "S!mpleJWTS3cretK3y!2025@Secure")
    
    # Email settings (use environment variables for production)
    SENDER_EMAIL = os.getenv("SENDER_EMAIL", "viveksapkale0022@gmail.com")
    SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "vppp mprd fvbz mqbn")

    # MongoDB connection
    MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://cluster0.mcjuw.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=Cluster0")
    
    # Upload settings
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {"mp4"}

    # Alert system
    ALERT_INTERVAL = 20  # seconds
