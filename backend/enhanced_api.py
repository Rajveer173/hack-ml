"""
API endpoints for enhanced ML features and user controls.
"""

from flask import Blueprint, jsonify, request, current_app
import os
import json
from werkzeug.utils import secure_filename
import feature_extraction as fe
import enhanced_ml as eml
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt

# Initialize enhanced models
ai_detector = eml.EnhancedAIDetector()
plagiarism_detector = eml.EnhancedPlagiarismDetector()

# Set up the blueprint
enhanced_api = Blueprint("enhanced", __name__)

# Constants
ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'doc', 'docx', 'py', 'java', 'js', 'jsx', 
    'ts', 'tsx', 'c', 'cpp', 'cs', 'html', 'css'
}

# History storage
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history')
os.makedirs(HISTORY_DIR, exist_ok=True)

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

def save_to_history(analysis_type, filename, content, result):
    """Save analysis result to history"""
    import datetime
    import uuid
    
    # Generate unique ID
    analysis_id = str(uuid.uuid4())
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare history entry
    history_entry = {
        'id': analysis_id,
        'timestamp': timestamp,
        'type': analysis_type,
        'filename': filename,
        'result': result
    }
    
    # Save to history file
    history_file = os.path.join(HISTORY_DIR, 'analysis_history.json')
    
    try:
        # Load existing history
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
            
        # Add new entry
        history.append(history_entry)
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
            
        return analysis_id
    except Exception as e:
        current_app.logger.error(f"Error saving to history: {str(e)}")
        return None

def create_visualization(data, viz_type):
    """Create visualization image as base64 string"""
    plt.figure(figsize=(10, 6))
    
    if viz_type == 'feature_importance':
        # Sort by importance
        sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
        
        # Create bar chart
        plt.barh(list(sorted_data.keys())[-10:], list(sorted_data.values())[-10:])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
    
    elif viz_type == 'similarity_heatmap':
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        pivot_df = df.pivot(index='file1', columns='file2', values='similarity')
        
        # Create heatmap
        plt.imshow(pivot_df.values, cmap='YlOrRd')
        plt.colorbar(label='Similarity %')
        plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=90)
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)
        plt.title('Document Similarity Heatmap')
        plt.tight_layout()
        
    # Convert to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

@enhanced_api.route("/ai-detection", methods=["POST"])
def enhanced_ai_detection():
    """Enhanced AI detection with adjustable sensitivity"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    # Get sensitivity setting if provided
    sensitivity = request.form.get('sensitivity', None)
    if sensitivity:
        try:
            sensitivity = float(sensitivity)
        except ValueError:
            sensitivity = None
            
    if file and allowed_file(file.filename):
        content = extract_text_content(file)
        
        if not content:
            return jsonify({"error": "Could not extract content from file"}), 400
            
        # Get AI detection results using the enhanced model
        ai_detection = ai_detector.predict(content, sensitivity=sensitivity)
        
        # Create feature importance visualization
        if 'feature_importance' in ai_detection:
            ai_detection['feature_importance_viz'] = create_visualization(
                ai_detection['feature_importance'], 'feature_importance')
        
        # Save to history
        analysis_id = save_to_history('ai_detection', file.filename, content, ai_detection)
        
        # Format response
        result = {
            "analysis_id": analysis_id,
            "filename": file.filename,
            "ai_score": ai_detection["ai_score"],
            "classification": ai_detection["classification"],
            "ai_sections": ai_detection["ai_sections"],
            "feature_analysis": ai_detection["feature_analysis"],
            "feature_importance": ai_detection.get("feature_importance", {}),
            "feature_importance_viz": ai_detection.get("feature_importance_viz", ""),
            "model_version": ai_detection["model_version"],
            "sensitivity_used": ai_detection["sensitivity_used"],
            "status": "Original" if ai_detection["ai_score"] < 30 else 
                     "Moderate AI Content" if ai_detection["ai_score"] < 70 else 
                     "High AI Content"
        }
        
        return jsonify(result)
    
    return jsonify({"error": "File type not allowed"}), 400

@enhanced_api.route("/plagiarism", methods=["POST"])
def enhanced_plagiarism_check():
    """Enhanced plagiarism detection with adjustable settings"""
    if 'files[]' not in request.files:
        return jsonify({"error": "No files part"}), 400
        
    files = request.files.getlist('files[]')
    
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No selected files"}), 400
        
    if len(files) < 2:
        return jsonify({"error": "Need at least 2 files to compare"}), 400
        
    # Get sensitivity setting if provided
    sensitivity = request.form.get('sensitivity', None)
    if sensitivity:
        try:
            sensitivity = float(sensitivity)
        except ValueError:
            sensitivity = None
            
    # Get threshold settings if provided
    threshold_high = request.form.get('threshold_high', None)
    threshold_moderate = request.form.get('threshold_moderate', None)
    
    threshold_settings = {}
    if threshold_high:
        try:
            threshold_settings['high_similarity'] = float(threshold_high)
        except ValueError:
            pass
    
    if threshold_moderate:
        try:
            threshold_settings['moderate_similarity'] = float(threshold_moderate)
        except ValueError:
            pass
        
    # Process files
    files_content = {}
    for file in files:
        if file and allowed_file(file.filename):
            content = extract_text_content(file)
            if content:
                files_content[file.filename] = content
        
    if len(files_content) < 2:
        return jsonify({"error": "Could not extract content from enough files"}), 400
    
    # Use the enhanced plagiarism detector
    results = plagiarism_detector.compare_documents(
        files_content, 
        sensitivity=sensitivity,
        threshold_settings=threshold_settings if threshold_settings else None
    )
    
    # Create similarity heatmap visualization
    if 'heatmap_data' in results:
        results['heatmap_viz'] = create_visualization(results['heatmap_data'], 'similarity_heatmap')
    
    # Save to history
    analysis_id = save_to_history('plagiarism', 'multiple_files', files_content, results)
    results['analysis_id'] = analysis_id
    
    return jsonify(results)

@enhanced_api.route("/settings/sensitivity", methods=["POST"])
def update_sensitivity():
    """Update sensitivity settings for detection"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    detector_type = data.get('detector_type')
    sensitivity = data.get('sensitivity')
    
    if not detector_type or not sensitivity:
        return jsonify({"error": "Missing required fields"}), 400
        
    try:
        sensitivity = float(sensitivity)
        if detector_type == 'ai':
            result = ai_detector.set_sensitivity(sensitivity)
        elif detector_type == 'plagiarism':
            result = plagiarism_detector.set_sensitivity(sensitivity)
        else:
            return jsonify({"error": "Invalid detector type"}), 400
            
        return jsonify(result)
    except ValueError:
        return jsonify({"error": "Invalid sensitivity value"}), 400

@enhanced_api.route("/settings/thresholds", methods=["POST"])
def update_thresholds():
    """Update threshold settings for plagiarism detection"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    high = data.get('high_similarity')
    moderate = data.get('moderate_similarity')
    
    if high is None and moderate is None:
        return jsonify({"error": "No threshold values provided"}), 400
        
    try:
        if high is not None:
            high = float(high)
        if moderate is not None:
            moderate = float(moderate)
            
        result = plagiarism_detector.set_thresholds(high, moderate)
        return jsonify(result)
    except ValueError:
        return jsonify({"error": "Invalid threshold values"}), 400

@enhanced_api.route("/history", methods=["GET"])
def get_analysis_history():
    """Get analysis history"""
    history_file = os.path.join(HISTORY_DIR, 'analysis_history.json')
    
    if not os.path.exists(history_file):
        return jsonify({"history": []})
        
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
            
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Limit to most recent entries
        limit = request.args.get('limit', default=50, type=int)
        history = history[:limit]
        
        return jsonify({"history": history})
    except Exception as e:
        current_app.logger.error(f"Error reading history: {str(e)}")
        return jsonify({"error": "Could not read history"}), 500

@enhanced_api.route("/history/<analysis_id>", methods=["GET"])
def get_analysis_details(analysis_id):
    """Get details of a specific analysis"""
    history_file = os.path.join(HISTORY_DIR, 'analysis_history.json')
    
    if not os.path.exists(history_file):
        return jsonify({"error": "History not found"}), 404
        
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
            
        # Find the requested analysis
        analysis = next((item for item in history if item['id'] == analysis_id), None)
        
        if not analysis:
            return jsonify({"error": "Analysis not found"}), 404
            
        return jsonify(analysis)
    except Exception as e:
        current_app.logger.error(f"Error reading analysis details: {str(e)}")
        return jsonify({"error": "Could not read analysis details"}), 500

@enhanced_api.route("/export/<analysis_id>", methods=["GET"])
def export_analysis(analysis_id):
    """Export analysis results in various formats"""
    format_type = request.args.get('format', default='json')
    
    # Get analysis details
    history_file = os.path.join(HISTORY_DIR, 'analysis_history.json')
    
    if not os.path.exists(history_file):
        return jsonify({"error": "History not found"}), 404
        
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
            
        # Find the requested analysis
        analysis = next((item for item in history if item['id'] == analysis_id), None)
        
        if not analysis:
            return jsonify({"error": "Analysis not found"}), 404
            
        # Export based on requested format
        if format_type == 'json':
            return jsonify(analysis)
        
        elif format_type == 'csv':
            # Convert to CSV
            if analysis['type'] == 'ai_detection':
                # Create DataFrame for AI detection
                data = {
                    'Filename': [analysis['filename']],
                    'AI Score': [analysis['result']['ai_score']],
                    'Classification': [analysis['result']['classification']],
                    'Timestamp': [analysis['timestamp']],
                    'Model Version': [analysis['result'].get('model_version', 'N/A')]
                }
                
                # Add features
                if 'feature_analysis' in analysis['result']:
                    for feature, value in analysis['result']['feature_analysis'].items():
                        data[feature] = [value]
                
                df = pd.DataFrame(data)
                
            elif analysis['type'] == 'plagiarism':
                # Create DataFrame for plagiarism
                rows = []
                for comp in analysis['result'].get('comparisons', []):
                    rows.append({
                        'File 1': comp['file1'],
                        'File 2': comp['file2'],
                        'Similarity Score': comp['similarity_score'],
                        'Similarity Level': comp['similarity_level'],
                        'ML Confidence': comp.get('ml_confidence_score', 'N/A'),
                        'Timestamp': analysis['timestamp']
                    })
                
                df = pd.DataFrame(rows)
                
            else:
                return jsonify({"error": "Unknown analysis type"}), 400
            
            # Convert to CSV
            csv_data = df.to_csv(index=False)
            
            # Prepare response
            from flask import Response
            return Response(
                csv_data,
                mimetype="text/csv",
                headers={"Content-disposition": f"attachment; filename={analysis_id}.csv"}
            )
            
        else:
            return jsonify({"error": "Unsupported export format"}), 400
            
    except Exception as e:
        current_app.logger.error(f"Error exporting analysis: {str(e)}")
        return jsonify({"error": "Could not export analysis"}), 500

@enhanced_api.route("/feedback", methods=["POST"])
def submit_feedback():
    """Submit user feedback for model improvement"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    analysis_type = data.get('analysis_type')
    original_result = data.get('original_result')
    corrected_result = data.get('corrected_result')
    features = data.get('features')
    
    if not analysis_type or not original_result or not corrected_result:
        return jsonify({"error": "Missing required fields"}), 400
        
    try:
        if analysis_type == 'ai_detection':
            result = ai_detector.save_feedback(original_result, corrected_result, features)
        elif analysis_type == 'plagiarism':
            result = plagiarism_detector.save_feedback(original_result, corrected_result, features)
        else:
            return jsonify({"error": "Invalid analysis type"}), 400
            
        return jsonify(result)
    except Exception as e:
        current_app.logger.error(f"Error saving feedback: {str(e)}")
        return jsonify({"error": "Could not save feedback"}), 500

@enhanced_api.route("/model/info", methods=["GET"])
def get_model_info():
    """Get information about the current models"""
    model_type = request.args.get('type', default='all')
    
    if model_type == 'ai_detection' or model_type == 'all':
        ai_info = ai_detector.get_model_info()
    else:
        ai_info = None
        
    if model_type == 'plagiarism' or model_type == 'all':
        plagiarism_info = plagiarism_detector.get_model_info()
    else:
        plagiarism_info = None
        
    result = {
        'ai_detection': ai_info,
        'plagiarism': plagiarism_info
    }
    
    return jsonify(result)

@enhanced_api.route("/model/update", methods=["POST"])
def update_model():
    """Update model using collected feedback"""
    data = request.get_json()
    model_type = data.get('model_type', 'ai_detector')
    
    try:
        if model_type == 'ai_detector':
            result = ai_detector.update_from_feedback()
        elif model_type == 'plagiarism':
            result = plagiarism_detector.update_from_feedback()
        else:
            return jsonify({"error": "Invalid model type"}), 400
            
        return jsonify(result)
    except Exception as e:
        current_app.logger.error(f"Error updating model: {str(e)}")
        return jsonify({"error": "Could not update model"}), 500
