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
    """Check a single file for AI-generated content with advanced options"""
    try:
        # Get sensitivity parameter (default 0.5)
        sensitivity = float(request.form.get('sensitivity', 0.5))
        sensitivity = min(max(sensitivity, 0.0), 1.0)  # Clamp to 0.0-1.0 range
        
        # Get analysis depth
        analysis_depth = request.form.get('analysisDepth', 'standard')
        
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if file and allowed_file(file.filename):
            content = extract_text_content(file)
            
            if not content:
                return jsonify({"error": "Could not extract content from file"}), 400
            
            # Import with proper error handling
            try:
                from ml_model import AIDetector
                # Initialize detector with sensitivity
                detector = AIDetector()
                detector.set_sensitivity(sensitivity)
            except ImportError:
                # Use the fallback detector from ml_integration
                detector = ml_integration.get_ai_detector()()
                detector.set_sensitivity(sensitivity)
            
            try:
                # Get comprehensive analysis
                ai_detection = detector.predict(content, sensitivity)
            except Exception as e:
                current_app.logger.error(f"Error in AI detection prediction: {str(e)}")
                # Provide a basic fallback response
                ai_detection = {
                    "ai_score": 50,
                    "classification": "Analysis Failed",
                    "ai_sections": [],
                    "feature_analysis": {},
                    "feature_importance": {},
                    "probability": 0.5  # Add probability for frontend compatibility
                }            # For deep analysis, include visualization data
            visualization_data = {}
            if analysis_depth == 'deep':
                # Get visualization data for feature distributions
                visualization_data = ml_integration.get_visualization_data(content)
            
            # Format response with enhanced details
            result = {
                "filename": file.filename,
                "ai_score": ai_detection["ai_score"],
                "classification": ai_detection["classification"],
                "ai_sections": ai_detection["ai_sections"],
                "confidence": ai_detection.get("confidence", 85.0),
                "feature_analysis": ai_detection.get("feature_analysis", {}),
                "feature_importance": ai_detection.get("feature_importance", {}),
                "model_version": ai_detection.get("model_version", "2.1.0"),
                "execution_time": ai_detection.get("execution_time", 0),
                "sensitivity_used": sensitivity,
                "similarity_score": ai_detection.get("ai_score", 50),  # Use ai_score as similarity_score
                "probability": ai_detection.get("ai_score", 50) / 100,  # Convert ai_score to 0-1 range for probability
                "visualization_data": visualization_data if analysis_depth == 'deep' else None,
                "text_length": len(content),
                "text_stats": {
                    "word_count": len(content.split()),
                    "sentence_count": len([s for s in content.split('.') if s.strip()]),
                    "paragraph_count": len([p for p in content.split('\n\n') if p.strip()])
                }
            }
            
            return jsonify(result)
        
        return jsonify({"error": "File type not allowed"}), 400
    
    except Exception as e:
        current_app.logger.error(f"Error in AI detection: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@plagiarism_api.route("/check/text", methods=["POST"])
def check_text_plagiarism():
    """Check text input for AI-generated content"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        text = data.get('text')
        sensitivity = data.get('sensitivity', 0.5)
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Apply sensitivity bounds
        sensitivity = min(max(float(sensitivity), 0.0), 1.0)
        
        # Import directly from ml_model for the latest version
        from ml_model import AIDetector
        
        # Initialize detector with sensitivity
        detector = AIDetector()
        detector.set_sensitivity(sensitivity)
        
        # Get comprehensive analysis
        ai_detection = detector.predict(text, sensitivity)
        
        # Format response with enhanced details
        result = {
            "ai_score": ai_detection["ai_score"],
            "classification": ai_detection["classification"],
            "ai_sections": ai_detection["ai_sections"],
            "confidence": ai_detection.get("confidence", 85.0),
            "feature_analysis": ai_detection.get("feature_analysis", {}),
            "feature_importance": ai_detection.get("feature_importance", {}),
            "model_version": ai_detection.get("model_version", "2.1.0"),
            "execution_time": ai_detection.get("execution_time", 0),
            "sensitivity_used": sensitivity,
            "text_length": len(text),
            "text_stats": {
                "word_count": len(text.split()),
                "sentence_count": len([s for s in text.split('.') if s.strip()]),
                "paragraph_count": len([p for p in text.split('\n\n') if p.strip()])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        current_app.logger.error(f"Error in text plagiarism check: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@plagiarism_api.route("/compare", methods=["POST"])
def compare_files():
    """Compare multiple files for plagiarism with advanced options"""
    try:
        # Get similarity method parameter
        similarity_method = request.form.get('similarityMethod', 'hybrid')
        if similarity_method not in ['hybrid', 'tfidf', 'ngram', 'difflib']:
            similarity_method = 'hybrid'  # Default to hybrid if invalid
        
        # Check if we should ensure exact matches get 100%
        ensure_exact_matches = request.form.get('ensureExactMatches', 'true').lower() == 'true'
        
        # Get sensitivity for AI detection
        sensitivity = float(request.form.get('sensitivity', 0.5))
        sensitivity = min(max(sensitivity, 0.0), 1.0)  # Clamp to 0.0-1.0 range
        
        # Log the request parameters
        current_app.logger.info(f"Comparing files with method: {similarity_method}, sensitivity: {sensitivity}, ensure_exact_matches: {ensure_exact_matches}")
        
        # Check if files were uploaded
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
                    # Extract file extension for language-specific analysis
                    extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
                    files_content[file.filename] = {
                        "content": content,
                        "extension": extension,
                        "length": len(content),
                        "word_count": len(content.split())
                    }
            
        if len(files_content) < 2:
            return jsonify({"error": "Could not extract content from enough files"}), 400
        
        # Import with proper error handling
        try:
            # Import directly from plagiarism.py for the latest version
            from plagiarism import calculate_file_similarity, detect_ai_content
            from ml_model import AIDetector
            
            # Initialize AI detector
            detector = AIDetector()
            detector.set_sensitivity(sensitivity)
        except ImportError:
            # Use ml_integration as fallback
            from ml_integration import FallbackPlagiarismDetector, ai_detector
            
            detector = ai_detector
            detector.set_sensitivity(sensitivity)
            
            # Create a fallback similarity calculator
            def calculate_file_similarity(text1, text2, method='hybrid'):
                # Use our fallback detector
                fallback_detector = FallbackPlagiarismDetector()
                # Create a dummy dictionary for the two texts
                docs_dict = {'file1': text1, 'file2': text2}
                # Compare them using the fallback detector
                results = fallback_detector.compare_documents(docs_dict, method)
                
                # Special case - if the texts are identical, return 100% similarity
                if text1 == text2:
                    return {
                        'similarity': 100.0,
                        'method': method,
                        'details': {'exact_match': True}
                    }
                
                # Otherwise, extract the similarity between the two documents
                return {
                    'similarity': results['similarity_matrix'].get('file1', {}).get('file2', 0),
                    'method': method,
                    'details': {'matched_blocks': len(results.get('suspected_sections', []))}
                }
        
        # Calculate pairwise comparisons with the specified method
        comparisons = []
        for i, (file1, data1) in enumerate(files_content.items()):
            for file2, data2 in list(files_content.items())[i+1:]:
                # Check for identical content first
                if data1["content"] == data2["content"]:
                    # Special case for identical files
                    similarity_results = {
                        'similarity': 100.0,
                        'method': similarity_method,
                        'details': {'exact_match': True}
                    }
                else:
                    # Calculate similarity for non-identical files
                    similarity_results = calculate_file_similarity(
                        data1["content"], 
                        data2["content"],
                        method=similarity_method
                    )
                
                # Extract matching sections (only for non-binary content)
                matching_sections = []
                try:
                    if all(ext in ['txt', 'py', 'java', 'js', 'jsx', 'ts', 'tsx', 'c', 'cpp', 'cs', 'html', 'css'] 
                           for ext in [data1["extension"], data2["extension"]]):
                        # Special case for identical content
                        if data1["content"] == data2["content"]:
                            # For identical files, add a single matching section for the whole file
                            matching_sections.append({
                                "file1_start": 0,
                                "file2_start": 0,
                                "length": len(data1["content"]),
                                "content": data1["content"][:100] + "..." if len(data1["content"]) > 100 else data1["content"]
                            })
                        else:
                            # Use difflib to find matching blocks of text
                            import difflib
                            matcher = difflib.SequenceMatcher(None, data1["content"], data2["content"])
                            # Get longest matching blocks
                            matches = [match for match in matcher.get_matching_blocks() if match.size > 20]
                        # Limit to top 3 matches
                        for match in matches[:3]:
                            matching_sections.append({
                                "file1_start": match.a,
                                "file2_start": match.b,
                                "length": match.size,
                                "content": data1["content"][match.a:match.a + match.size][:100] + "..." 
                                if match.size > 100 else data1["content"][match.a:match.a + match.size]
                            })
                except Exception as e:
                    current_app.logger.error(f"Error finding matching sections: {str(e)}")
                
                # Create the comparison result
                comparison_result = {
                    "file1": file1,
                    "file2": file2,
                    "similarity_score": similarity_results["similarity"],
                    "similarity_method": similarity_results["method"],
                    "similarity_details": similarity_results["details"],
                    "matching_sections": matching_sections
                }
                
                # Check if files are identical and ensure they get 100% if needed
                if ensure_exact_matches and data1["content"] == data2["content"]:
                    comparison_result["similarity_score"] = 100.0
                    current_app.logger.info(f"Files {file1} and {file2} are identical - setting similarity to 100%")
                
                # Log the similarity score for debugging
                current_app.logger.info(f"Similarity between {file1} and {file2}: {comparison_result['similarity_score']}%")
                
                comparisons.append(comparison_result)
        
        # Get AI detection for each file
        ai_results = {}
        for filename, data in files_content.items():
            try:
                ai_results[filename] = detector.predict(data["content"], sensitivity)
            except Exception as e:
                current_app.logger.error(f"Error detecting AI in {filename}: {str(e)}")
                ai_results[filename] = {
                    "ai_score": 0, 
                    "classification": "Analysis failed",
                    "error": str(e)
                }
        
        # Add AI detection results to the comparisons
        for comp in comparisons:
            comp["file1_ai_score"] = ai_results[comp["file1"]].get("ai_score", 0)
            comp["file2_ai_score"] = ai_results[comp["file2"]].get("ai_score", 0)
            comp["file1_ai_classification"] = ai_results[comp["file1"]].get("classification", "Unknown")
            comp["file2_ai_classification"] = ai_results[comp["file2"]].get("classification", "Unknown")
        
        # Analyze common patterns across all files
        common_patterns = {}
        try:
            # Look for code patterns if we have code files
            code_extensions = ['py', 'java', 'js', 'jsx', 'ts', 'tsx', 'c', 'cpp', 'cs']
            code_files = [data for filename, data in files_content.items() 
                         if data["extension"] in code_extensions]
            
            if code_files:
                # Detect common code patterns
                from plagiarism import detect_code_patterns
                
                all_patterns = []
                for data in code_files:
                    patterns = detect_code_patterns(data["content"])
                    all_patterns.append(patterns)
                
                # Analyze pattern similarity
                import numpy as np
                
                if all_patterns:
                    # Convert to array for analysis
                    pattern_arrays = []
                    keys = list(all_patterns[0].keys())
                    
                    for pattern in all_patterns:
                        pattern_arrays.append([pattern.get(key, 0) for key in keys])
                    
                    pattern_array = np.array(pattern_arrays)
                    
                    # Calculate variance across files for each pattern type
                    pattern_variance = np.var(pattern_array, axis=0)
                    
                    # Low variance indicates similar structure across files
                    common_patterns = {
                        "pattern_types": keys,
                        "similarity_score": 100 - min(100, 100 * np.mean(pattern_variance) / 
                                                     (np.mean(pattern_array) + 0.001)),
                        "common_structure_detected": np.mean(pattern_variance) < 
                                                     (0.3 * np.mean(pattern_array))
                    }
        except Exception as e:
            current_app.logger.error(f"Error analyzing common patterns: {str(e)}")
            common_patterns = {"error": str(e)}
        
        # One final check to ensure identical files always show 100% similarity
        if ensure_exact_matches:
            for comp in comparisons:
                file1 = comp["file1"]
                file2 = comp["file2"]
                if files_content[file1]["content"] == files_content[file2]["content"]:
                    comp["similarity_score"] = 100.0
                    current_app.logger.info(f"Final check: {file1} and {file2} have identical content, ensuring 100% score")
        
        # Format the final response
        results = {
            "file_count": len(files_content),
            "comparisons": comparisons,
            "ai_results": {filename: {
                "ai_score": result.get("ai_score", 0),
                "classification": result.get("classification", "Unknown"),
                "confidence": result.get("confidence", 0)
            } for filename, result in ai_results.items()},
            "common_patterns": common_patterns,
            "analysis_method": similarity_method,
            "sensitivity_used": sensitivity
        }
        
        # Log the final comparison results
        for comp in comparisons:
            current_app.logger.info(f"FINAL COMPARISON: {comp['file1']} vs {comp['file2']}: {comp['similarity_score']}%")
        
        return jsonify(results)
        
    except Exception as e:
        current_app.logger.error(f"Error in plagiarism comparison: {str(e)}")
        return jsonify({"error": f"Comparison failed: {str(e)}"}), 500
        
@plagiarism_api.route("/train", methods=["POST"])
def train_model():
    """Train or fine-tune the AI detection model with custom data"""
    try:
        # Check if user has admin privileges (you would need to implement auth)
        # if not has_admin_privileges():
        #    return jsonify({"error": "Unauthorized access"}), 403
        
        if 'human_files[]' not in request.files or 'ai_files[]' not in request.files:
            return jsonify({"error": "Missing files for training"}), 400
            
        human_files = request.files.getlist('human_files[]')
        ai_files = request.files.getlist('ai_files[]')
        
        if not human_files or not ai_files:
            return jsonify({"error": "Need both human and AI samples for training"}), 400
            
        # Process files
        human_texts = []
        ai_texts = []
        
        # Process human-written texts
        for file in human_files:
            if file and allowed_file(file.filename):
                content = extract_text_content(file)
                if content and len(content) > 100:  # Minimum content length
                    human_texts.append(content)
        
        # Process AI-generated texts
        for file in ai_files:
            if file and allowed_file(file.filename):
                content = extract_text_content(file)
                if content and len(content) > 100:  # Minimum content length
                    ai_texts.append(content)
        
        if len(human_texts) < 5 or len(ai_texts) < 5:
            return jsonify({"error": "Need at least 5 samples of each type for training"}), 400
            
        # Import the AIDetector
        from ml_model import AIDetector
        
        # Initialize detector
        detector = AIDetector()
        
        # Prepare training data
        texts = human_texts + ai_texts
        labels = [0] * len(human_texts) + [1] * len(ai_texts)  # 0 for human, 1 for AI
        
        # Train model
        training_results = detector.train(texts, labels, test_size=0.3, cv_folds=5)
        
        # Return training results
        return jsonify({
            "success": True,
            "message": "Model training completed successfully",
            "metrics": training_results,
            "human_samples": len(human_texts),
            "ai_samples": len(ai_texts),
            "model_version": detector.version
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in model training: {str(e)}")
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

@plagiarism_api.route("/check/text/enhanced", methods=["POST"])
def check_text_plagiarism_enhanced():
    """Check text input for AI-generated content using enhanced method"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        text = data.get('text')
        sensitivity = data.get('sensitivity', 0.5)
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Apply sensitivity bounds
        sensitivity = min(max(float(sensitivity), 0.0), 1.0)
        
        # Import with proper error handling
        try:
            from ml_model import AIDetector
            # Initialize detector with sensitivity
            detector = AIDetector()
            detector.set_sensitivity(sensitivity)
        except ImportError:
            # Use the fallback detector from ml_integration
            detector = ml_integration.ai_detector
            detector.set_sensitivity(sensitivity)
            
        try:
            # Get comprehensive analysis
            ai_detection = detector.predict(text, sensitivity)
        except Exception as e:
            current_app.logger.error(f"Error in AI detection prediction: {str(e)}")
            # Provide a basic fallback response
            ai_detection = {
                "ai_score": 50,
                "classification": "Analysis Failed",
                "ai_sections": [],
                "feature_analysis": {},
                "feature_importance": {}
            }
        
        # Format response with enhanced details
        result = {
            "ai_score": ai_detection["ai_score"],
            "classification": ai_detection["classification"],
            "ai_sections": ai_detection["ai_sections"],
            "confidence": ai_detection.get("confidence", 85.0),
            "feature_analysis": ai_detection.get("feature_analysis", {}),
            "feature_importance": ai_detection.get("feature_importance", {}),
            "model_version": ai_detection.get("model_version", "2.1.0"),
            "execution_time": ai_detection.get("execution_time", 0),
            "sensitivity_used": sensitivity,
            "similarity_score": ai_detection.get("ai_score", 50),  # Use ai_score as similarity_score
            "probability": ai_detection.get("ai_score", 50) / 100,  # Convert ai_score to 0-1 range for probability
            "text_length": len(text),
            "text_stats": {
                "word_count": len(text.split()),
                "sentence_count": len([s for s in text.split('.') if s.strip()]),
                "paragraph_count": len([p for p in text.split('\n\n') if p.strip()])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        current_app.logger.error(f"Error in text plagiarism check: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@plagiarism_api.route("/api/external", methods=["POST"])
def external_api_check():
    """Check text using external AI detection APIs"""
    try:
        # Get text from request
        if not request.json or 'text' not in request.json:
            return jsonify({"error": "No text provided"}), 400
            
        text = request.json['text']
        api_type = request.json.get('api', 'openai')  # Default to openai
        
        if not text or len(text) < 50:
            return jsonify({"error": "Text too short for analysis"}), 400
            
        # In a real implementation, we would connect to external APIs
        # For this demo, we'll simulate the API calls
        
        # Simulated responses (in a real app, these would be API calls)
        if api_type == "openai":
            # Simulate OpenAI detector response
            import random
            score = random.uniform(0.6, 0.95) if "artificial intelligence" in text.lower() else random.uniform(0.1, 0.4)
            
            response = {
                "api": "openai",
                "ai_probability": score,
                "classification": "AI-generated" if score > 0.7 else "Likely human",
                "confidence": 0.85,
                "response_time": 0.8
            }
        elif api_type == "gptzero":
            # Simulate GPTZero response
            import random
            score = random.uniform(0.7, 0.98) if "learning" in text.lower() else random.uniform(0.05, 0.3)
            
            response = {
                "api": "gptzero",
                "ai_probability": score,
                "perplexity": 75 - (score * 50),
                "burstiness": 0.2 + (score * 0.3),
                "classification": "AI" if score > 0.65 else "Human",
                "response_time": 1.2
            }
        else:
            # Use our local model as fallback
            from ml_model import AIDetector
            
            # Initialize detector
            detector = AIDetector()
            
            # Get prediction
            result = detector.predict(text)
            
            response = {
                "api": "local",
                "ai_probability": result["ai_score"] / 100,
                "classification": result["classification"],
                "confidence": result.get("confidence", 85) / 100,
                "response_time": result.get("execution_time", 0.5)
            }
        
        return jsonify(response)
        
    except Exception as e:
        current_app.logger.error(f"Error in external API check: {str(e)}")
        return jsonify({"error": f"Check failed: {str(e)}"}), 500
