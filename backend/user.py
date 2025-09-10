from flask import Blueprint, jsonify, request, session
from database import get_db_connection
import hashlib
import uuid
import os
import json
from functools import wraps

# Create a blueprint
user_bp = Blueprint("user", __name__)

# Setup user settings and history directories
USER_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_data')
os.makedirs(USER_DATA_DIR, exist_ok=True)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        print(f"DEBUG: Session contents: {session}")
        print(f"DEBUG: Session user_id: {session.get('user_id')}")
        print(f"DEBUG: Session keys: {list(session.keys())}")
        
        if 'user_id' not in session:
            print(f"DEBUG: Authentication failed - no user_id in session")
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

def get_user_data_dir(user_id):
    """Get user-specific data directory"""
    user_dir = os.path.join(USER_DATA_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def hash_password(password):
    """Hash a password for storing"""
    salt = uuid.uuid4().hex
    return hashlib.sha256(salt.encode() + password.encode()).hexdigest() + ':' + salt

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    hash_part, salt = stored_password.split(':')
    return hash_part == hashlib.sha256(salt.encode() + provided_password.encode()).hexdigest()

@user_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or not "username" in data or not "password" in data:
        return jsonify({"error": "Username and password are required"}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if user already exists
    cursor.execute("SELECT id FROM users WHERE username = ?", (data["username"],))
    if cursor.fetchone() is not None:
        conn.close()
        return jsonify({"error": "Username already exists"}), 409
    
    # Hash the password
    hashed_password = hash_password(data["password"])
    
    # Insert new user
    cursor.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (data["username"], hashed_password)
    )
    conn.commit()
    
    # Get the new user's ID
    user_id = cursor.lastrowid
    conn.close()
    
    # Create user's data directory and default settings
    user_dir = get_user_data_dir(user_id)
    with open(os.path.join(user_dir, 'settings.json'), 'w') as f:
        json.dump({"defaultSensitivity": 0.5}, f)
    
    with open(os.path.join(user_dir, 'history.json'), 'w') as f:
        json.dump([], f)
    
    return jsonify({"message": "User registered successfully", "user_id": user_id}), 201

@user_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or not "username" in data or not "password" in data:
        return jsonify({"error": "Username and password are required"}), 400
    
    print(f"DEBUG: Login attempt for username: {data['username']}")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, password FROM users WHERE username = ?", (data["username"],))
    user = cursor.fetchone()
    conn.close()
    
    if user and verify_password(user["password"], data["password"]):
        print(f"DEBUG: Login successful for user ID: {user['id']}")
        session["user_id"] = user["id"]
        session["username"] = data["username"]
        session.permanent = True  # Make the session permanent according to PERMANENT_SESSION_LIFETIME
        
        # Debug session after setting
        print(f"DEBUG: Session after login: {session}")
        print(f"DEBUG: Session user_id after login: {session.get('user_id')}")
        
        return jsonify({"message": "Login successful", "user_id": user["id"]}), 200
    
    print(f"DEBUG: Login failed for username: {data['username']}")
    return jsonify({"error": "Invalid username or password"}), 401

@user_bp.route("/logout", methods=["POST"])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return jsonify({"message": "Logout successful"}), 200

@user_bp.route("/settings", methods=["GET"])
@login_required
def get_settings():
    user_dir = get_user_data_dir(session["user_id"])
    settings_file = os.path.join(user_dir, 'settings.json')
    
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
    else:
        settings = {"defaultSensitivity": 0.5}
        with open(settings_file, 'w') as f:
            json.dump(settings, f)
    
    return jsonify(settings), 200

@user_bp.route("/settings", methods=["POST"])
@login_required
def update_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No settings provided"}), 400
    
    user_dir = get_user_data_dir(session["user_id"])
    settings_file = os.path.join(user_dir, 'settings.json')
    
    # Load existing settings
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
    else:
        settings = {}
    
    # Update settings with new values
    settings.update(data)
    
    # Save updated settings
    with open(settings_file, 'w') as f:
        json.dump(settings, f)
    
    return jsonify({"message": "Settings updated successfully", "settings": settings}), 200

@user_bp.route("/check-auth", methods=["GET"])
def check_auth():
    """Endpoint to check if the user is authenticated"""
    print(f"DEBUG: Check auth session: {session}")
    if 'user_id' in session:
        return jsonify({
            "authenticated": True,
            "user_id": session["user_id"],
            "username": session.get("username")
        }), 200
    return jsonify({"authenticated": False}), 200

@user_bp.route("/history", methods=["GET"])
@login_required
def get_history():
    user_dir = get_user_data_dir(session["user_id"])
    history_file = os.path.join(user_dir, 'history.json')
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
        with open(history_file, 'w') as f:
            json.dump(history, f)
    
    return jsonify({"history": history}), 200
