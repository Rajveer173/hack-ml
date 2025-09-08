from flask import Flask, request, jsonify
from user import user_bp
from plagiarism_api import plagiarism_api  # Import new ML-based plagiarism blueprint
from document_check import document_check_bp  # Import document check blueprint
from enhanced_api import enhanced_api  # Import enhanced features API

def create_app():
    app = Flask(__name__)

    # Enable CORS
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    # Register blueprints
    app.register_blueprint(user_bp, url_prefix="/user")
    app.register_blueprint(plagiarism_api, url_prefix="/plagiarism")
    app.register_blueprint(document_check_bp, url_prefix="/document")
    app.register_blueprint(enhanced_api, url_prefix="/enhanced")

    @app.route("/")
    def home():
        return {"message": "Flask Backend Running!"}

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
