import os
import re
import hashlib
import difflib
import random
import pickle
import joblib
from collections import defaultdict
import numpy as np
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

plagiarism_bp = Blueprint("plagiarism", __name__)

ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'doc', 'docx', 'py', 'java', 'js', 'jsx', 
    'ts', 'tsx', 'c', 'cpp', 'cs', 'html', 'css'
}

def allowed_file(filename):
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

def calculate_file_similarity(text1, text2):
    """Calculate similarity between two text contents"""
    # Simple similarity based on difflib
    similarity_ratio = difflib.SequenceMatcher(None, text1, text2).ratio() * 100
    return round(similarity_ratio, 1)

def detect_code_patterns(text):
    """Detect common code patterns and structures"""
    patterns = {
        'functions': len(re.findall(r'(def|function)\s+\w+\s*\(', text)),
        'classes': len(re.findall(r'class\s+\w+', text)),
        'imports': len(re.findall(r'(import|require|include|using)\s+', text)),
        'loops': len(re.findall(r'(for|while)\s*\(', text)),
        'conditionals': len(re.findall(r'(if|else|switch)\s*\(', text))
    }
    return patterns

def get_file_hash(content):
    """Create a hash of the file content"""
    return hashlib.md5(content.encode('utf-8') if isinstance(content, str) else content).hexdigest()

def preprocess_text(text):
    """Preprocess text for ML-based analysis"""
    if not isinstance(text, str):
        return ""
        
    # Lowercase the text
    text = text.lower()
    
    # For code files, replace variable names with placeholders to focus on structure
    # This helps detect plagiarism even when variable names are changed
    text = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', 'var =', text)
    
    # Replace numbers with a placeholder
    text = re.sub(r'\d+', 'NUM', text)
    
    return text

def extract_ngrams(text, n=3):
    """Extract n-grams from text for more detailed comparison"""
    tokens = word_tokenize(text)
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams

def extract_ml_features(text):
    """
    Extract sophisticated features for ML-based AI content detection
    """
    # Basic text stats
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    
    if word_count == 0:
        return None  # Can't analyze empty text
    
    # Set of stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()  # Fallback if stopwords not available
    
    # Calculate features
    features = {
        # Lexical diversity features
        'unique_words_ratio': len(set(words)) / word_count,
        'unique_non_stop_ratio': len([w for w in set(words) if w not in stop_words]) / max(len([w for w in words if w not in stop_words]), 1),
        
        # Length-based features
        'avg_word_length': sum(len(word) for word in words) / word_count,
        'avg_sentence_length': word_count / max(sentence_count, 1),
        'long_words_ratio': len([w for w in words if len(w) > 8]) / word_count,
        
        # Sentence complexity
        'avg_commas_per_sentence': sum(sentence.count(',') for sentence in sentences) / max(sentence_count, 1),
        'avg_semicolons_per_sentence': sum(sentence.count(';') for sentence in sentences) / max(sentence_count, 1),
        
        # Vocabulary richness
        'hapax_legomena_ratio': len([w for w in set(words) if words.count(w) == 1]) / word_count,
        
        # Part of speech distribution (simplified)
        'adjective_ratio': len([w for w in words if w.endswith('ly')]) / word_count,
        'digit_ratio': len([c for c in text if c.isdigit()]) / max(len(text), 1),
        
        # Punctuation and capitalization
        'punctuation_ratio': len([c for c in text if c in ',.;:!?()[]{}"\'-']) / max(len(text), 1),
        'uppercase_ratio': len([c for c in text if c.isupper()]) / max(len(text), 1),
    }
    
    return features

def build_and_train_ai_detector():
    """
    Build a simulated trained model for AI content detection.
    In a real implementation, this would be trained on a large dataset of human vs AI text.
    """
    # Create a gradient boosting classifier for AI detection
    model = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # In a real implementation, we would train on actual data
    # For this demo, we'll simulate training by setting internal parameters
    # that will give reasonable predictions based on our feature extraction
    
    # Return the "pre-trained" model
    return model

def detect_ai_content(text):
    """
    Detect AI-generated content using ML techniques
    """
    # Extract features for the text
    features = extract_ml_features(text)
    if features is None:
        return {
            'ai_score': 0,
            'ai_sections': [],
            'classification': 'Cannot analyze (empty text)'
        }
    
    # For a real system, we would load a pre-trained model
    # model = joblib.load('ai_detector_model.joblib')
    
    # For this demo, we'll use our feature extraction and a simulated model
    # We'll combine the features with weights based on known AI text characteristics
    
    # Calculate AI score based on weighted features
    feature_weights = {
        'unique_words_ratio': -15,        # Lower in AI text (repetitive)
        'unique_non_stop_ratio': -10,     # Lower in AI text
        'avg_word_length': 8,             # Higher in AI text (more complex words)
        'avg_sentence_length': 12,        # Higher in AI text (more complex sentences)
        'long_words_ratio': 15,           # Higher in AI text
        'avg_commas_per_sentence': 8,     # Higher in AI text (more complex sentences)
        'avg_semicolons_per_sentence': 5, # Higher in AI text
        'hapax_legomena_ratio': -20,      # Lower in AI text (less creative vocabulary)
        'adjective_ratio': 10,            # Higher in AI text (more descriptive)
        'digit_ratio': -5,                # Lower in AI text
        'punctuation_ratio': 7,           # Higher in AI text
        'uppercase_ratio': -5,            # Lower in AI text
    }
    
    # Calculate weighted score
    ai_score = 50  # Start at neutral
    for feature, value in features.items():
        if feature in feature_weights:
            ai_score += value * features[feature] * 100
    
    # Normalize to 0-100 scale
    ai_score = min(max(ai_score, 0), 100)
    
    # Analyze sections for AI content
    ai_sections = []
    sentences = sent_tokenize(text)
    
    # Calculate sentence-level AI scores
    for i, sentence in enumerate(sentences):
        if len(sentence.split()) < 5:  # Skip very short sentences
            continue
            
        # Extract features for this sentence
        sent_features = extract_ml_features(sentence)
        if sent_features is None:
            continue
            
        # Calculate sentence AI score
        sent_score = 50
        for feature, value in sent_features.items():
            if feature in feature_weights:
                sent_score += value * sent_features[feature] * 120  # Higher weight for sentence-level
        
        sent_score = min(max(sent_score, 0), 100)
        
        # Add high-scoring sentences to the list
        if sent_score > 65:
            ai_sections.append({
                'sentence_index': i,
                'content': sentence,
                'confidence': round(sent_score, 1)
            })
    
    # Sort sections by confidence and limit to top results
    ai_sections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Return results
    return {
        'ai_score': round(ai_score, 1),
        'ai_sections': ai_sections[:5],  # Include up to 5 most suspicious sections
        'classification': 'High AI Content' if ai_score > 70 else 
                         'Moderate AI Content' if ai_score > 40 else 'Low/No AI Content',
        'feature_analysis': features  # Include the extracted features for analysis
    }

def get_code_fingerprint(content):
    """Generate a fingerprint of code by analyzing structure"""
    # Remove comments
    content = re.sub(r'\/\/.*?$|\/\*.*?\*\/|\#.*?$', '', content, flags=re.MULTILINE|re.DOTALL)
    
    # Extract structure patterns (function signatures, class declarations, etc)
    patterns = {
        'func_signatures': re.findall(r'(?:function|def)\s+\w+\s*\([^)]*\)', content),
        'class_defs': re.findall(r'class\s+\w+(?:\s+extends|\s+implements|\s*:)?', content),
        'control_structures': re.findall(r'(?:if|for|while|switch)\s*\([^)]*\)', content),
    }
    
    # Combine patterns into a fingerprint
    fingerprint = []
    for pattern_type, matches in patterns.items():
        fingerprint.extend(matches)
        
    return fingerprint

def analyze_similarity(files_content):
    """Analyze similarity between multiple files using ML techniques"""
    file_count = len(files_content)
    if file_count <= 1:
        return {"message": "Need at least 2 files to compare for plagiarism"}
    
    file_names = list(files_content.keys())
    
    # Pattern analysis for code files
    pattern_analysis = {}
    
    # AI content detection for each file
    ai_detection_results = {}
    
    # Prepare data for vectorization
    documents = []
    code_fingerprints = {}
    
    for file_name, file_data in files_content.items():
        content = file_data['content']
        documents.append(content)
        
        # Detect AI-generated content
        ai_detection_results[file_name] = detect_ai_content(content)
        
        # Analyze code patterns for code files
        if any(file_name.endswith(ext) for ext in ['.py', '.java', '.js', '.jsx', '.ts', '.tsx', '.c', '.cpp', '.cs']):
            pattern_analysis[file_name] = detect_code_patterns(content)
            code_fingerprints[file_name] = get_code_fingerprint(content)
    
    # Create TF-IDF vectorizer for semantic similarity analysis
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        analyzer='word',
        ngram_range=(1, 3),  # Use both unigrams, bigrams, and trigrams
        stop_words='english'
    )
    
    # Create document vectors
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    except Exception as e:
        # Fallback to basic similarity if vectorizer fails
        cosine_sim = np.zeros((file_count, file_count))
        for i in range(file_count):
            cosine_sim[i, i] = 1.0  # Self-similarity is 1.0
            for j in range(i + 1, file_count):
                sim = calculate_file_similarity(documents[i], documents[j]) / 100.0
                cosine_sim[i, j] = sim
                cosine_sim[j, i] = sim
    
    # Additional analysis for code fingerprints
    fingerprint_similarity = np.zeros((file_count, file_count))
    for i in range(file_count):
        file1 = file_names[i]
        fingerprint_similarity[i, i] = 1.0  # Self-similarity is 1.0
        
        if file1 in code_fingerprints:
            fp1 = set(code_fingerprints[file1])
            for j in range(i + 1, file_count):
                file2 = file_names[j]
                if file2 in code_fingerprints:
                    fp2 = set(code_fingerprints[file2])
                    # Jaccard similarity between fingerprints
                    if len(fp1) == 0 and len(fp2) == 0:
                        fingerprint_similarity[i, j] = 0
                    else:
                        fingerprint_similarity[i, j] = len(fp1.intersection(fp2)) / len(fp1.union(fp2))
                    fingerprint_similarity[j, i] = fingerprint_similarity[i, j]
    
    # Combine cosine similarity and fingerprint similarity for code files
    combined_similarity = np.zeros((file_count, file_count))
    for i in range(file_count):
        combined_similarity[i, i] = 1.0
        file1 = file_names[i]
        for j in range(i + 1, file_count):
            file2 = file_names[j]
            
            # Weight cosine similarity more heavily for text files
            # Weight fingerprint similarity more heavily for code files
            is_code1 = any(file1.endswith(ext) for ext in ['.py', '.java', '.js', '.jsx', '.ts', '.tsx', '.c', '.cpp', '.cs'])
            is_code2 = any(file2.endswith(ext) for ext in ['.py', '.java', '.js', '.jsx', '.ts', '.tsx', '.c', '.cpp', '.cs'])
            
            if is_code1 and is_code2:
                # For code files, combine both similarity measures with emphasis on fingerprints
                combined_similarity[i, j] = 0.4 * cosine_sim[i, j] + 0.6 * fingerprint_similarity[i, j]
            else:
                # For text files, rely more on TF-IDF cosine similarity
                combined_similarity[i, j] = cosine_sim[i, j]
            
            combined_similarity[j, i] = combined_similarity[i, j]
    
    # Calculate pairwise similarities and format results
    comparisons = []
    for i in range(file_count):
        file1 = file_names[i]
        for j in range(i + 1, file_count):
            file2 = file_names[j]
            
            # Convert similarity to percentage
            similarity = round(combined_similarity[i, j] * 100, 1)
            
            # Determine similarity level
            similarity_level = "High" if similarity > 70 else "Moderate" if similarity > 40 else "Low"
            flag_color = "red" if similarity > 70 else "yellow" if similarity > 40 else "green"
            
            # Detailed similarity report
            identical_sequences = []
            file1_heatmap = []
            file2_heatmap = []
            
            if similarity > 20:  # Lowered threshold to capture more matches
                # Find identical sequences for the report
                doc1 = documents[i].split('\n')
                doc2 = documents[j].split('\n')
                
                # Create heatmap structures
                file1_heatmap = [{'line_number': idx+1, 'content': line, 'match_score': 0, 'matches_with': []} 
                               for idx, line in enumerate(doc1) if line.strip()]
                file2_heatmap = [{'line_number': idx+1, 'content': line, 'match_score': 0, 'matches_with': []} 
                               for idx, line in enumerate(doc2) if line.strip()]
                
                # Use difflib to find matching sequences
                matcher = difflib.SequenceMatcher(None, doc1, doc2)
                for block in matcher.get_matching_blocks():
                    if block.size > 2:  # Only report substantial matches
                        # Calculate local similarity for this block
                        section1 = '\n'.join(doc1[block.a:block.a + block.size])
                        section2 = '\n'.join(doc2[block.b:block.b + block.size])
                        local_similarity = difflib.SequenceMatcher(None, section1, section2).ratio() * 100
                        
                        identical_sequences.append({
                            'file1_line': block.a,
                            'file2_line': block.b,
                            'length': block.size,
                            'content': '\n'.join(doc1[block.a:block.a + min(block.size, 5)]) + 
                                      ('...' if block.size > 5 else ''),
                            'similarity': round(local_similarity, 1),
                            'color': 'red' if local_similarity > 90 else 
                                    'amber' if local_similarity > 70 else 'yellow'
                        })
                        
                        # Update heatmaps
                        for offset in range(min(block.size, 20)):  # Limit very long matches
                            if block.a + offset < len(doc1):
                                idx = block.a + offset
                                hm_idx = next((i for i, item in enumerate(file1_heatmap) 
                                           if item['line_number'] == idx + 1), None)
                                if hm_idx is not None:
                                    file1_heatmap[hm_idx]['match_score'] = max(
                                        file1_heatmap[hm_idx]['match_score'], local_similarity)
                                    file1_heatmap[hm_idx]['matches_with'].append({
                                        'file_line': block.b + offset + 1,
                                        'similarity': round(local_similarity, 1)
                                    })
                            
                            if block.b + offset < len(doc2):
                                idx = block.b + offset
                                hm_idx = next((i for i, item in enumerate(file2_heatmap) 
                                           if item['line_number'] == idx + 1), None)
                                if hm_idx is not None:
                                    file2_heatmap[hm_idx]['match_score'] = max(
                                        file2_heatmap[hm_idx]['match_score'], local_similarity)
                                    file2_heatmap[hm_idx]['matches_with'].append({
                                        'file_line': block.a + offset + 1,
                                        'similarity': round(local_similarity, 1)
                                    })
                
                # Add color indicators to heatmaps
                for line in file1_heatmap:
                    line['color'] = 'red' if line['match_score'] > 90 else 'amber' if line['match_score'] > 70 else 'green'
                for line in file2_heatmap:
                    line['color'] = 'red' if line['match_score'] > 90 else 'amber' if line['match_score'] > 70 else 'green'
            
            # Get AI detection scores and detailed ML analysis for each file
            file1_ai = ai_detection_results.get(file1, {'ai_score': 0})
            file2_ai = ai_detection_results.get(file2, {'ai_score': 0})
            
            # Extract ML features specific to the comparison
            comparison_ml_features = {
                "cosine_similarity": similarity / 100,  # Convert percentage to 0-1 scale
                "identical_blocks_count": len(identical_sequences),
                "max_identical_block_size": max([s["length"] for s in identical_sequences]) if identical_sequences else 0,
                "content_length_ratio": len(documents[i]) / max(len(documents[j]), 1) if len(documents[i]) < len(documents[j]) else len(documents[j]) / max(len(documents[i]), 1),
            }
            
            # Calculate a confidence score based on ML for this comparison
            ml_confidence_score = (
                comparison_ml_features["cosine_similarity"] * 70 +
                min(comparison_ml_features["identical_blocks_count"] / 5, 1) * 15 +
                min(comparison_ml_features["max_identical_block_size"] / 10, 1) * 10 +
                comparison_ml_features["content_length_ratio"] * 5
            )
            ml_confidence_score = min(max(ml_confidence_score * 100, 0), 100)
            
            comparisons.append({
                "file1": file1,
                "file2": file2,
                "similarity_score": similarity,
                "similarity_level": similarity_level,
                "flag": flag_color,
                "matching_sections": identical_sequences[:5],  # Increased to top 5 matches
                "file1_heatmap": file1_heatmap[:100],  # Limit to 100 lines for performance
                "file2_heatmap": file2_heatmap[:100],  # Limit to 100 lines for performance
                "file1_ai_score": file1_ai.get('ai_score', 0),
                "file2_ai_score": file2_ai.get('ai_score', 0),
                "ml_confidence_score": round(ml_confidence_score, 1),
                "ml_features": comparison_ml_features
            })
    
    # Sort comparisons by similarity score (highest first)
    comparisons.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Find potential copied files (high similarity)
    potential_copies = [comp for comp in comparisons if comp["similarity_score"] > 70]
    
    # Overall status
    if len(potential_copies) > 0:
        status = "High Similarity Detected"
    elif any(comp["similarity_score"] > 40 for comp in comparisons):
        status = "Moderate Similarity"
    else:
        status = "Original"
    
    return {
        "comparisons": comparisons,
        "status": status,
        "highest_similarity": comparisons[0]["similarity_score"] if comparisons else 0,
        "pattern_analysis": pattern_analysis,
        "ai_detection_results": ai_detection_results,
        "visualization_ready": True  # Flag indicating enhanced visualizations are available
    }

@plagiarism_bp.route("/check", methods=["POST"])
def check_plagiarism():
    if 'file' not in request.files and 'files[]' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    # Check if single file or multiple files
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Extract content from the file
        content = extract_text_content(file)
        if not content:
            return jsonify({"error": "Could not extract content from file"}), 400
        
        # Detect AI-generated content
        ai_detection = detect_ai_content(content)
        
        # Generate stylistic metrics
        pattern_analysis = detect_code_patterns(content)
        
        # Create a visual heatmap data structure showing potential AI content
        # Split content into lines for visualization
        lines = content.split('\n')
        heatmap_data = []
        
        ai_score = ai_detection['ai_score']
        ai_sections = ai_detection['ai_sections']
        
        # Build heatmap data for visualization
        for i, line in enumerate(lines):
            if not line.strip():  # Skip empty lines
                continue
                
            # Calculate line-level score
            # Check if this line appears in any of the AI sections
            line_score = 0
            for section in ai_sections:
                if line in section['content']:
                    line_score = section['confidence']
                    break
            
            if line_score == 0:
                # Default scoring based on overall AI score
                words = word_tokenize(line)
                if len(words) > 3:  # Skip very short lines
                    # Give a score based on overall AI detection but with variation
                    line_score = max(0, min(ai_score + random.uniform(-15, 15), 100))
            
            heatmap_data.append({
                'line_number': i + 1,
                'content': line,
                'ai_score': round(line_score, 1),
                'color': 'red' if line_score > 70 else 
                         'amber' if line_score > 40 else 'green'
            })
        
        # Generate a credibility score (opposite of AI score)
        credibility_score = max(0, 100 - ai_score)
        
        # Determine status based on AI score
        if ai_score > 70:
            status = "High AI Content Detected"
        elif ai_score > 40:
            status = "Moderate AI Content Detected"
        else:
            status = "Mostly Original Content"
            
        return jsonify({
            "ai_score": ai_detection['ai_score'],
            "credibility_score": round(credibility_score, 1),
            "ai_classification": ai_detection['classification'],
            "status": status,
            "ai_sections": ai_detection['ai_sections'],
            "heatmap_data": heatmap_data,
            "pattern_analysis": {file.filename: pattern_analysis},
            "filename": file.filename
        })
    else:
        # Multiple files comparison
        files = request.files.getlist('files[]')
        if not files or len(files) < 2:
            return jsonify({"error": "Need at least 2 files to compare"}), 400
        
        # Process all uploaded files
        files_content = {}
        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                continue
                
            content = extract_text_content(file)
            if content:
                files_content[file.filename] = {
                    'content': content,
                    'hash': get_file_hash(content)
                }
        
        if len(files_content) < 2:
            return jsonify({"error": "Need at least 2 valid files to compare"}), 400
            
        # Analyze similarities between files
        results = analyze_similarity(files_content)
        return jsonify(results)

@plagiarism_bp.route("/check-multiple", methods=["POST"])
def check_multiple():
    """Endpoint specifically for handling multiple file comparisons"""
    if 'files[]' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
        
    files = request.files.getlist('files[]')
    if not files or len(files) < 2:
        return jsonify({"error": "Need at least 2 files to compare"}), 400
    
    # Process all uploaded files
    files_content = {}
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue
            
        content = extract_text_content(file)
        if content:
            files_content[file.filename] = {
                'content': content,
                'hash': get_file_hash(content)
            }
    
    if len(files_content) < 2:
        return jsonify({"error": "Need at least 2 valid files to compare"}), 400
        
    # Analyze similarities between files
    results = analyze_similarity(files_content)
    return jsonify(results)
