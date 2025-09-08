"""
Integration module for connecting ML models with the plagiarism detection system.
This module bridges the feature extraction and ML models with the Flask routes.
"""

import os
from ml_model import AIDetector, PlagiarismDetector
import feature_extraction as fe

# Initialize the models
ai_detector = AIDetector()
plagiarism_detector = PlagiarismDetector()

def detect_ai_content(text):
    """
    Detect AI-generated content using the ML model
    
    Args:
        text: The text content to analyze
        
    Returns:
        Dictionary with AI detection results
    """
    return ai_detector.predict(text)

def analyze_document_similarity(files_content):
    """
    Analyze similarity between multiple documents
    
    Args:
        files_content: Dictionary of {filename: content}
        
    Returns:
        Dictionary with comparison results
    """
    # Convert files_content from {filename: {content: str}} to {filename: str}
    documents = {filename: data['content'] for filename, data in files_content.items()}
    return plagiarism_detector.compare_documents(documents)

def extract_document_features(text):
    """
    Extract all relevant features from a document
    
    Args:
        text: The text content to analyze
        
    Returns:
        Dictionary with extracted features
    """
    # Determine if this is likely code
    is_code = any([
        'def ' in text,
        'function ' in text,
        'class ' in text,
        'import ' in text,
        'from ' in text and ' import ' in text,
        '{' in text and '}' in text and ';' in text,  # Likely JS or similar
    ])
    
    if is_code:
        # Extract code-specific features
        language = 'python' if 'def ' in text else 'javascript' if 'function ' in text else 'unknown'
        features = fe.extract_code_features(text, language)
    else:
        # Extract text features
        features = fe.extract_advanced_features(text)
        
    return features

def get_ml_analysis_summary(text):
    """
    Get a comprehensive ML analysis summary for a text
    
    Args:
        text: The text content to analyze
        
    Returns:
        Dictionary with comprehensive analysis
    """
    # Extract features
    features = extract_document_features(text)
    
    # Get AI detection results
    ai_detection = detect_ai_content(text)
    
    # Prepare summary
    summary = {
        'ai_detection': ai_detection,
        'features': features,
        'text_stats': {
            'word_count': len(text.split()),
            'character_count': len(text),
            'sentence_count': len(text.split('.')),
        }
    }
    
    return summary
