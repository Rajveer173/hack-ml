"""
Modified routes for the plagiarism blueprint to use the new ML modules.
"""

from flask import Blueprint, jsonify, request, current_app
import os
from werkzeug.utils import secure_filename
import ml_integration

# Set up the blueprint
plagiarism_api = Blueprint("plagiarism_api", __name__)

# Constants
ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'doc', 'docx', 'py', 'java', 'js', 'jsx', 
    'ts', 'tsx', 'c', 'cpp', 'cs', 'html', 'css'
}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_content(file):
    """Extract text content from uploaded files, handling different file types"""
    filename = secure_filename(file.filename)
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    try:
        # For text-based files, read directly
        if extension in ['txt', 'py', 'java', 'js', 'jsx', 'ts', 'tsx', 'c', 'cpp', 'cs', 'html', 'css']:
            content = file.read().decode('utf-8')
            return content
            
        # For binary files like PDF, DOC, etc. we would need more complex parsing
        # For this example, we'll simulate content extraction
        elif extension in ['pdf', 'doc', 'docx']:
            # In a real app, use libraries like PyPDF2, python-docx, etc.
            # For this demo, just return file name to indicate successful upload
            return f"Content from {filename} (simulated extraction)"
        else:
            return ""
    except Exception as e:
        current_app.logger.error(f"Error extracting content from {filename}: {str(e)}")
        return ""

@plagiarism_api.route("/check", methods=["POST"])
def check_plagiarism():
    """Check a single file for AI-generated content"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        content = extract_text_content(file)
        
        if not content:
            return jsonify({"error": "Could not extract content from file"}), 400
            
        # Get AI detection results using the new ML integration
        ai_detection = ml_integration.detect_ai_content(content)
        
        # Get comprehensive ML analysis
        ml_analysis = ml_integration.get_ml_analysis_summary(content)
        
        # Format response
        result = {
            "filename": file.filename,
            "ai_score": ai_detection["ai_score"],
            "classification": ai_detection["classification"],
            "ai_sections": ai_detection["ai_sections"],
            "feature_analysis": ml_analysis["features"],
            "text_stats": ml_analysis["text_stats"],
            "status": "Original" if ai_detection["ai_score"] < 30 else 
                      "Moderate AI Content" if ai_detection["ai_score"] < 70 else 
                      "High AI Content"
        }
        
        return jsonify(result)
    
    return jsonify({"error": "File type not allowed"}), 400

@plagiarism_api.route("/compare", methods=["POST"])
def compare_files():
    """Compare multiple files for plagiarism"""
    if 'files[]' not in request.files:
        return jsonify({"error": "No files part"}), 400
        
    files = request.files.getlist('files[]')
    
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No selected files"}), 400
        
    if len(files) < 2:
        return jsonify({"error": "Need at least 2 files to compare"}), 400
        
    # Process files
    files_content = {}
    for file in files:
        if file and allowed_file(file.filename):
            content = extract_text_content(file)
            if content:
                files_content[file.filename] = {"content": content}
        
    if len(files_content) < 2:
        return jsonify({"error": "Could not extract content from enough files"}), 400
    
    # Use the new ML integration module to analyze similarity
    results = ml_integration.analyze_document_similarity(files_content)
    
    # Get AI detection for each file
    ai_results = {}
    for filename, data in files_content.items():
        ai_results[filename] = ml_integration.detect_ai_content(data["content"])
    
    # Add AI detection results to the response
    for comp in results["comparisons"]:
        comp["file1_ai_score"] = ai_results[comp["file1"]]["ai_score"]
        comp["file2_ai_score"] = ai_results[comp["file2"]]["ai_score"]
        comp["file1_ai_classification"] = ai_results[comp["file1"]]["classification"]
        comp["file2_ai_classification"] = ai_results[comp["file2"]]["classification"]
    
    return jsonify(results)
