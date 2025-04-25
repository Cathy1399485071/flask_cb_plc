from flask import Flask, render_template, request, redirect, url_for, session
from module.quiz_upload import handle_quiz_upload
from module.chatbot import get_chatbot_response
import os
import csv

app = Flask(__name__)
app.secret_key = "test123"

# Enforce login before every request (except a few routes)
@app.before_request
def require_login():
    allowed_routes = ["login", "register", "static"]
    if request.endpoint is None or (request.endpoint not in allowed_routes and "username" not in session):
        return redirect(url_for("login"))

@app.route("/")
def index():
    return "Welcome! Visit /upload for professors and /chat for students."

@app.route("/register", methods=["GET", "POST"])
def register():
    save_path = os.path.join("..", "plc_storage", "registered_users.txt")
    history_dir = os.path.join("..", "plc_storage", "history")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    if request.method == "POST":
        username = request.form["username"]

        # Check if username is alphanumeric
        if not username.isalnum():
            return "Username must only contain letters and numbers."

        # Check if username already exists
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                registered_users = [line.strip() for line in f.readlines()]
            if username in registered_users:
                return "Username already exists."

        # Append new username to file
        with open(save_path, "a") as f:
            f.write(username + "\n")

        # Create user's history CSV file
        history_path = os.path.join(history_dir, f"{username}_history.csv")
        with open(history_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Query", "Response"])  # header row

        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]

        # Load registered usernames from file
        save_path = os.path.join("..", "plc_storage", "registered_users.txt")
        if not os.path.exists(save_path):
            return "No users registered yet."

        with open(save_path, "r") as f:
            registered_users = [line.strip() for line in f.readlines()]

        if username in registered_users:
            session["username"] = username
            return redirect(url_for("select_course"))
        else:
            return "Username not found."

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("selected_course", None)
    session.pop("messages", None)
    return redirect(url_for("login"))

@app.route("/select-course", methods=["GET", "POST"])
def select_course():
    if request.method == "POST":
        selected_course = request.form.get("course")
        session["selected_course"] = selected_course
        return redirect(url_for("chat"))
    return render_template("select_course.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_quiz():
    if request.method == "POST":
        uploaded_file = request.files["quiz_file"]
        result = handle_quiz_upload(uploaded_file)
        return f"Upload successful: {result}"
    return render_template("upload.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "selected_course" not in session:
        return redirect(url_for("select_course"))

    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form["user_input"]
        bot_reply = get_chatbot_response(user_input, session["selected_course"], session["username"])
        session["messages"].append({"sender": "user", "text": user_input})
        session["messages"].append({"sender": "bot", "text": bot_reply})
        session.modified = True

    return render_template("chatbot.html", messages=session["messages"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
