from flask import Blueprint, jsonify, request

# Create a blueprint
user_bp = Blueprint("user", __name__)

# Example in-memory "database"
users = []

@user_bp.route("/", methods=["GET"])
def get_users():
    return jsonify(users)

@user_bp.route("/", methods=["POST"])
def add_user():
    data = request.get_json()
    if "name" not in data:
        return jsonify({"error": "Name is required"}), 400
    
    users.append(data)
    return jsonify({"message": "User added", "users": users}), 201
