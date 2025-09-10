from flask import Flask, request, jsonify
from user import user_bp
from plagiarism_api import plagiarism_api  # Import new ML-based plagiarism blueprint
from document_check import document_check_bp  # Import document check blueprint
from enhanced_api import enhanced_api  # Import enhanced features API
import os
from datetime import timedelta
from database import init_db

def create_app():
    app = Flask(__name__)
    
    # Configure the app
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_replace_in_production')
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session lasts 7 days
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Allows cookies on same-site requests
    
    # Initialize the database
    init_db()

    # Enable CORS with credentials support
    @app.after_request
    def after_request(response):
        # Get the origin from the request
        origin = request.headers.get('Origin')
        
        # Allow specific origins (you could also use a configuration variable)
        allowed_origins = ['http://localhost:3000', 'http://localhost:5173']
        
        # If the request origin is in our allowed origins, set it in the response header
        if origin in allowed_origins:
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
        
        # Handle preflight requests
        if request.method == 'OPTIONS':
            response.status_code = 200
        
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
