from flask import Blueprint, jsonify, request

document_check_bp = Blueprint("document", __name__)

@document_check_bp.route("/verify", methods=["POST"])
def verify_document():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Dummy document analysis results for now
    authenticity = "Authentic"  
    status = "Valid"
    return jsonify({"authenticity": authenticity, "status": status})
