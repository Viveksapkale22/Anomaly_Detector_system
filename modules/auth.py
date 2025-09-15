# File: modules/auth.py

import jwt
import datetime
from flask import request, session, redirect, url_for, flash
from werkzeug.security import check_password_hash
import smtplib
from email.message import EmailMessage
from config import Config

JWT_SECRET = Config.JWT_SECRET
SENDER_EMAIL = Config.SENDER_EMAIL
SENDER_PASSWORD = Config.SENDER_PASSWORD


def login_user(users_collection, bcrypt):
    username = request.form.get('username')
    password = request.form.get('password')
    user = users_collection.find_one({'username': username})

    if user and bcrypt.check_password_hash(user['password'], password):
        session['username'] = user['username']
        session['email'] = user['email']
        flash("Login successful!", "success")
        return redirect(url_for('index'))
    else:
        flash("Invalid username or password. Please try again!", "error")
        return redirect(url_for('index'))


def register_user(users_collection, bcrypt):
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    if users_collection.find_one({'username': username}):
        flash("Username already exists. Please choose another.", "error")
        return redirect(url_for('index'))

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    users_collection.insert_one({'username': username, 'email': email, 'password': hashed_password})

    flash("Registration successful! Please login.", "success")
    return redirect(url_for('index'))


def forget_password(users_collection):
    email = request.form.get("email")
    user = users_collection.find_one({"email": email})

    if not user:
        flash("Email not found. Please check or sign up.", "danger")
        return redirect(url_for("index"))

    token = jwt.encode(
        {"email": email, "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=15)},
        JWT_SECRET,
        algorithm="HS256"
    )

    if send_reset_email(email, token):
        flash("Password reset link sent to your email!", "success")
    else:
        flash("Error sending email. Try again later!", "danger")

    return redirect(url_for("index"))


def send_reset_email(email, token):
    reset_link = f"https://change-password-0cym.onrender.com/reset-password?token={token}"
    subject = "Password Reset Request"
    message = f"Subject: {subject}\n\nClick the link to reset your password: {reset_link} (Valid for 15 mins)"

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, email, message)
        return True
    except Exception as e:
        print("Error sending email:", e)
        return False


def logout_user():
    session.pop('username', None)
    session.pop('email', None)
    flash('You have been logged out.')
    return redirect(url_for('index'))
