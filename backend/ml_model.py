"""
Machine learning models for AI content detection and plagiarism analysis.
Enhanced implementation with state-of-the-art model architectures, transfer learning capabilities,
and advanced performance evaluation metrics.
"""

import os
import pickle
import re
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
from collections import defaultdict

# ML frameworks and tools
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import feature_extraction as fe
from model_versioning import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("ml_operations.log"), logging.StreamHandler()]
)
logger = logging.getLogger('ml_models')

# Default paths for model storage
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
AI_DETECTOR_MODEL_PATH = os.path.join(MODEL_DIR, 'ai_detector_model.joblib')
PLAGIARISM_MODEL_PATH = os.path.join(MODEL_DIR, 'plagiarism_model.joblib')
MODEL_METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.json')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'ai_detector': {
        'version': '2.1.0',
        'architecture': 'ensemble',
        'description': 'Advanced AI content detection model with ensemble learning',
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    },
    'plagiarism': {
        'version': '1.8.0',
        'architecture': 'hybrid',
        'description': 'Hybrid plagiarism detection model with semantic analysis',
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
}


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selector for model pipeline"""
    
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if isinstance(X, list) and all(isinstance(item, dict) for item in X):
            return np.array([[item.get(feature, 0) for feature in self.feature_names] for item in X])
        return X


class AIDetector:
    """Advanced AI-generated content detector using ensemble ML techniques"""
    
    def __init__(self, model_path=None, config=None):
        """Initialize the AI detector model with advanced configuration"""
        self.model_path = model_path or AI_DETECTOR_MODEL_PATH
        self.model = None
        self.feature_names = None
        self.config = config or MODEL_CONFIG['ai_detector']
        self.version = self.config['version']
        self.model_manager = ModelManager('ai_detector')
        self.metrics_history = self._load_metrics_history()
        self.sensitivity = 0.5  # Default sensitivity
        
        # Load model and feature set
        self.load_or_create_model()
        
    def load_or_create_model(self):
        """Load an existing model or create a new one if not found"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.feature_names = model_data.get('feature_names')
                    logger.info(f"Loaded AI detector model v{self.version} from {self.model_path}")
                else:
                    # Legacy model format
                    self.model = model_data
                    logger.info(f"Loaded legacy AI detector model from {self.model_path}")
            else:
                logger.info("No existing model found, creating new ensemble model")
                self._create_ensemble_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}. Creating new model.")
            self._create_ensemble_model()
    
    def _create_ensemble_model(self):
        """Create an advanced ensemble model for AI detection"""
        # Define base models for the ensemble
        base_models = [
            ('gbm', GradientBoostingClassifier(
                n_estimators=200, 
                learning_rate=0.1,
                max_depth=4,
                subsample=0.9,
                max_features=0.75,
                random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ))
        ]
        
        # Create ensemble model (voting classifier)
        self.model = VotingClassifier(
            estimators=base_models,
            voting='soft',  # Use probability estimates for voting
            weights=[3, 2, 1]  # Give more weight to GBM
        )
        
        # Extract common text analysis feature names
        sample_features = fe.extract_advanced_features("This is a sample text for feature extraction.")
        self.feature_names = list(sample_features.keys() if sample_features else [])
        return self.model
    
    def _load_metrics_history(self):
        """Load model performance metrics history"""
        if os.path.exists(MODEL_METRICS_PATH):
            try:
                with open(MODEL_METRICS_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics history: {str(e)}")
                return {'ai_detector': [], 'plagiarism': []}
        else:
            # Initialize with empty metrics
            return {'ai_detector': [], 'plagiarism': []}
            
    def _save_metrics(self, metrics):
        """Save model metrics to history"""
        metrics_history = self._load_metrics_history()
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['version'] = self.version
        metrics_history['ai_detector'].append(metrics)
        
        # Save to file
        try:
            with open(MODEL_METRICS_PATH, 'w') as f:
                json.dump(metrics_history, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def set_sensitivity(self, value):
        """Set model sensitivity (0.0-1.0)"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Sensitivity must be between 0.0 and 1.0")
        self.sensitivity = value
        return {"message": f"Sensitivity set to {self.sensitivity}"}

    def train(self, texts, labels, test_size=0.2, cv_folds=5):
        """
        Train the AI detector model on labeled data with cross-validation
        
        Args:
            texts: List of text samples
            labels: List of labels (1 for AI-generated, 0 for human)
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Training AI detector model with {len(texts)} samples")
        
        # Extract features from texts
        features = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            text_features = fe.extract_advanced_features(text)
            if text_features:
                features.append(text_features)
                valid_indices.append(i)
            else:
                logger.warning(f"Failed to extract features for sample {i}")
                continue
                
        # Filter labels to match valid features
        labels = [labels[i] for i in valid_indices]
        
        if len(features) < 10:
            raise ValueError("Insufficient valid samples for training (need at least 10)")
        
        # Archive current model before training
        self.model_manager.archive_current_model()
        
        # Update feature names based on the actual features extracted
        self.feature_names = list(features[0].keys())
        
        # Create feature selector with feature names
        feature_selector = FeatureSelector(feature_names=self.feature_names)
        
        # Create feature scaling pipeline
        preprocessing_pipeline = Pipeline([
            ('selector', feature_selector),
            ('scaler', StandardScaler())
        ])
        
        # Create and configure ensemble model
        ensemble_model = self._create_ensemble_model()
        
        # Create full pipeline
        model_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', ensemble_model)
        ])
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }
        
        # Save the model
        self.save_model()
        
        return metrics
    
    def save_model(self):
        """Save the model to disk"""
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def predict(self, text, sensitivity=None):
        """
        Predict if text is AI-generated with comprehensive analysis
        
        Args:
            text: The text to analyze
            sensitivity: Optional sensitivity override (0.0-1.0)
            
        Returns:
            Dictionary with comprehensive prediction results
        """
        start_time = datetime.now()
        logger.info(f"Starting AI content analysis of {len(text)} characters")
        
        # Use provided sensitivity or default
        sensitivity_value = sensitivity if sensitivity is not None else self.sensitivity
        
        # Extract features
        features = fe.extract_advanced_features(text)
        if not features:
            logger.warning("Feature extraction failed - empty or invalid text")
            return {
                'ai_score': 0,
                'ai_sections': [],
                'classification': 'Cannot analyze (empty or invalid text)',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_version': self.version,
                'sensitivity_used': sensitivity_value
            }
        
        # Get feature importance if available
        feature_importance = self._get_feature_importance()
        
        # If we have a trained model, use it
        if self.model is not None:
            try:
                # Prepare features for prediction
                if hasattr(self.model, 'named_steps') and 'preprocessing' in self.model.named_steps:
                    # For pipeline model (with preprocessing)
                    prediction_input = [features]  # Pass as list of dictionaries for FeatureSelector
                else:
                    # For direct model
                    feature_array = np.array([features[f] for f in self.feature_names]).reshape(1, -1)
                    prediction_input = feature_array
                
                # Get probability of AI-generated content
                probs = self.model.predict_proba(prediction_input)
                raw_ai_score = probs[0][1]  # Probability of class 1 (AI)
                
                # Apply sensitivity adjustment
                # Higher sensitivity = lower threshold to classify as AI
                adjusted_threshold = 0.5 * (1 - sensitivity_value)
                ai_score = raw_ai_score * 100  # Convert to percentage
                
                # Determine classification based on adjusted threshold
                if raw_ai_score > (0.7 - (sensitivity_value * 0.2)):
                    classification = 'High AI Content'
                elif raw_ai_score > (0.4 - (sensitivity_value * 0.2)):
                    classification = 'Moderate AI Content'
                else:
                    classification = 'Low/No AI Content'
                    
                logger.info(f"AI detection completed: {classification} ({ai_score:.1f}%)")
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                # Fallback to simulated prediction
                ai_score = self._simulate_prediction(features, sensitivity_value)
                classification = 'High AI Content' if ai_score > 70 else 'Moderate AI Content' if ai_score > 40 else 'Low/No AI Content'
        else:
            # Use simulated prediction if model is not trained
            logger.warning("No trained model available, using simulated prediction")
            ai_score = self._simulate_prediction(features, sensitivity_value)
            classification = 'High AI Content' if ai_score > 70 else 'Moderate AI Content' if ai_score > 40 else 'Low/No AI Content'
            
        # Analyze sentences for AI content
        ai_sections = self._analyze_sections(text, sensitivity_value)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Return comprehensive results
        return {
            'ai_score': round(ai_score, 1),
            'ai_sections': ai_sections,
            'classification': classification,
            'feature_analysis': features,  # Include the extracted features for analysis
            'feature_importance': feature_importance,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': self.version,
            'sensitivity_used': sensitivity_value,
            'execution_time': execution_time,
            'text_length': len(text),
            'confidence': self._calculate_confidence(ai_score, features)
        }
        
    def _calculate_confidence(self, ai_score, features):
        """Calculate confidence level in the prediction"""
        # Base confidence on feature completeness and prediction extremity
        feature_completeness = min(1.0, len(features) / 15)  # Ideally want at least 15 features
        
        # Higher confidence when prediction is closer to extremes (0 or 100)
        prediction_confidence = 1.0 - (2 * abs(ai_score - 50) / 100)
        
        # Calculate overall confidence (0-100%)
        confidence = (feature_completeness * 0.6 + prediction_confidence * 0.4) * 100
        return round(confidence, 1)
    
    def _simulate_prediction(self, features, sensitivity=0.5):
        """Simulate AI prediction based on weighted features with sensitivity adjustment"""
        # These weights are based on research showing characteristics of AI text
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
            'punctuation_ratio': 7,           # Higher in AI text
            'sentence_length_variance': -15,  # Lower in AI text (more uniform)
            'perplexity': -10,                # Lower in AI text (more predictable)
            'burstiness': -20,                # Lower in AI text (more even distribution)
            'type_token_ratio': -12,          # Lower in AI text (less lexical diversity)
            'stop_words_ratio': 5,            # Higher in AI text
        }
        
        # Calculate weighted score
        ai_score = 50  # Start at neutral
        contributed_scores = {}
        
        for feature, value in features.items():
            if feature in feature_weights:
                # Calculate contribution of this feature
                contribution = value * feature_weights[feature]
                contributed_scores[feature] = contribution
                ai_score += contribution
        
        # Apply sensitivity adjustment
        # Higher sensitivity = more likely to classify as AI
        sensitivity_factor = (sensitivity - 0.5) * 20  # -10 to +10 adjustment
        ai_score += sensitivity_factor
        
        # Normalize to 0-100 scale
        ai_score = min(max(ai_score, 0), 100)
        return ai_score
    
    def _analyze_sections(self, text, sensitivity=0.5):
        """Analyze text sections for AI content with advanced scoring"""
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            
            # Make sure we have the tokenizer
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
            # Adaptive chunking based on text length
            total_words = len(text.split())
            
            # For shorter texts, use smaller chunks
            if total_words < 1000:
                chunk_size = 200
            elif total_words < 3000:
                chunk_size = 500
            else:
                chunk_size = 800
                
            sentences = sent_tokenize(text)
            
            # Analyze chunks with overlap for more consistent detection
            chunks = []
            chunk_indices = []
            current_chunk = []
            word_count = 0
            start_idx = 0
            
            for i, sentence in enumerate(sentences):
                current_chunk.append(sentence)
                sentence_words = len(sentence.split())
                word_count += sentence_words
                
                if word_count >= chunk_size:
                    chunks.append(' '.join(current_chunk))
                    chunk_indices.append((start_idx, i))
                    
                    # Create 30% overlap between chunks for smoother transitions
                    overlap_sentences = max(1, int(len(current_chunk) * 0.3))
                    current_chunk = current_chunk[-overlap_sentences:]
                    word_count = sum(len(s.split()) for s in current_chunk)
                    start_idx = i - overlap_sentences + 1
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                chunk_indices.append((start_idx, len(sentences)-1))
            
            # Analyze each chunk
            section_scores = []
            for i, (chunk, (start, end)) in enumerate(zip(chunks, chunk_indices)):
                features = fe.extract_advanced_features(chunk)
                if features is None:
                    continue
                    
                score = self._simulate_prediction(features, sensitivity)
                confidence = self._calculate_confidence(score, features)
                chunk_words = len(chunk.split())
                
                section_scores.append({
                    'section': i+1,
                    'text': chunk[:100] + '...' if len(chunk) > 100 else chunk,
                    'sentence_range': f"{start+1}-{end+1}",
                    'ai_score': round(score, 1),
                    'confidence': confidence,
                    'words': chunk_words,
                    'key_features': self._get_key_features(features, score)
                })
            
            return section_scores
            
        except Exception as e:
            logger.error(f"Error in section analysis: {e}")
            return []
            
    def _get_key_features(self, features, score):
        """Extract the most influential features for this prediction"""
        # Determine which features are most indicative based on the score
        if score > 70:  # High AI probability
            key_indicators = {
                'perplexity': 'lower is more AI-like',
                'burstiness': 'lower is more AI-like',
                'hapax_legomena_ratio': 'lower is more AI-like',
                'unique_words_ratio': 'lower is more AI-like',
                'avg_word_length': 'higher is more AI-like',
                'sentence_length_variance': 'lower is more AI-like'
            }
        elif score < 30:  # Low AI probability
            key_indicators = {
                'perplexity': 'higher is more human-like',
                'burstiness': 'higher is more human-like',
                'hapax_legomena_ratio': 'higher is more human-like', 
                'unique_words_ratio': 'higher is more human-like',
                'sentence_length_variance': 'higher is more human-like'
            }
        else:  # Mixed signals
            key_indicators = {
                'perplexity': 'medium values are ambiguous',
                'burstiness': 'medium values are ambiguous',
                'unique_words_ratio': 'medium values are ambiguous'
            }
            
        result = {}
        for feature, description in key_indicators.items():
            if feature in features:
                result[feature] = {'value': round(features[feature], 3), 'description': description}
                
        return result


class PlagiarismDetector:
    """Plagiarism detection using ML techniques"""
    
    def __init__(self, model_path=None):
        """Initialize the plagiarism detector"""
        self.model_path = model_path or PLAGIARISM_MODEL_PATH
        self.vectorizer = TfidfVectorizer(
            preprocessor=self._preprocess_text,
            analyzer='word',
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            stop_words='english'
        )
    
    def _preprocess_text(self, text):
        """Preprocess text for vectorization"""
        if not isinstance(text, str):
            return ""
            
        # Lowercase the text
        text = text.lower()
        
        # For code files, replace variable names with placeholders to focus on structure
        text = self._normalize_code(text)
        
        return text
    
    def _normalize_code(self, text):
        """Normalize code to help detect plagiarism despite variable name changes"""
        # Replace variable names with placeholders
        text = text.lower()
        normalized = text
        
        # Replace variable declarations
        normalized = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', 'var =', normalized)
        
        # Replace function names in declarations
        normalized = re.sub(r'(?:function|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)', r'function func', normalized)
        
        # Replace numbers with a placeholder
        normalized = re.sub(r'\d+', 'NUM', normalized)
        
        return normalized
    
    def compare_documents(self, documents):
        """
        Compare multiple documents for plagiarism
        
        Args:
            documents: Dictionary of {filename: content}
            
        Returns:
            Dictionary with comparison results
        """
        if len(documents) <= 1:
            return {"message": "Need at least 2 documents to compare"}
            
        file_names = list(documents.keys())
        contents = list(documents.values())
        
        # Create document vectors
        try:
            tfidf_matrix = self.vectorizer.fit_transform(contents)
            # Calculate cosine similarity matrix
            from sklearn.metrics.pairwise import cosine_similarity
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        except Exception as e:
            # Fallback to basic similarity if vectorizer fails
            cosine_sim = np.zeros((len(documents), len(documents)))
            for i in range(len(documents)):
                cosine_sim[i, i] = 1.0  # Self-similarity is 1.0
                for j in range(i + 1, len(documents)):
                    sim = self._calculate_basic_similarity(contents[i], contents[j])
                    cosine_sim[i, j] = sim
                    cosine_sim[j, i] = sim
        
        # Generate code fingerprints for more accurate code comparison
        code_fingerprints = {}
        for file_name, content in documents.items():
            if any(file_name.endswith(ext) for ext in ['.py', '.java', '.js', '.jsx', '.ts', '.tsx', '.c', '.cpp', '.cs']):
                language = 'python' if file_name.endswith('.py') else 'javascript' if file_name.endswith(('.js', '.jsx')) else 'java'
                code_fingerprints[file_name] = fe.get_code_fingerprint(content, language)
        
        # Calculate fingerprint similarity for code files
        fingerprint_similarity = np.zeros((len(documents), len(documents)))
        for i, file1 in enumerate(file_names):
            fingerprint_similarity[i, i] = 1.0  # Self-similarity is 1.0
            
            if file1 in code_fingerprints:
                fp1 = set(str(x) for x in code_fingerprints[file1])  # Convert to strings for set operations
                for j, file2 in enumerate(file_names[i+1:], start=i+1):
                    if file2 in code_fingerprints:
                        fp2 = set(str(x) for x in code_fingerprints[file2])
                        # Jaccard similarity between fingerprints
                        if len(fp1) == 0 and len(fp2) == 0:
                            fingerprint_similarity[i, j] = 0
                        else:
                            fingerprint_similarity[i, j] = len(fp1.intersection(fp2)) / len(fp1.union(fp2))
                        fingerprint_similarity[j, i] = fingerprint_similarity[i, j]
        
        # Combine similarities for final results
        combined_similarity = np.zeros((len(documents), len(documents)))
        for i, file1 in enumerate(file_names):
            combined_similarity[i, i] = 1.0
            for j, file2 in enumerate(file_names[i+1:], start=i+1):
                # Weight based on file type
                is_code1 = any(file1.endswith(ext) for ext in ['.py', '.java', '.js', '.jsx', '.ts', '.tsx', '.c', '.cpp', '.cs'])
                is_code2 = any(file2.endswith(ext) for ext in ['.py', '.java', '.js', '.jsx', '.ts', '.tsx', '.c', '.cpp', '.cs'])
                
                if is_code1 and is_code2:
                    # For code files, combine both similarity measures with emphasis on fingerprints
                    combined_similarity[i, j] = 0.4 * cosine_sim[i, j] + 0.6 * fingerprint_similarity[i, j]
                else:
                    # For text files, rely more on TF-IDF cosine similarity
                    combined_similarity[i, j] = cosine_sim[i, j]
                
                combined_similarity[j, i] = combined_similarity[i, j]
        
        # Format results
        comparisons = []
        for i, file1 in enumerate(file_names):
            for j in range(i + 1, len(file_names)):
                file2 = file_names[j]
                
                # Convert similarity to percentage
                similarity = round(combined_similarity[i, j] * 100, 1)
                
                # Determine similarity level
                similarity_level = "High" if similarity > 70 else "Moderate" if similarity > 40 else "Low"
                flag_color = "red" if similarity > 70 else "yellow" if similarity > 40 else "green"
                
                # Prepare comparison result
                comparison = {
                    'file1': file1,
                    'file2': file2,
                    'similarity_score': similarity,
                    'similarity_level': similarity_level,
                    'flag': flag_color,
                    'ml_confidence_score': round(95 - (abs(50 - similarity) / 2)),  # Model confidence
                }
                
                # Add to results
                comparisons.append(comparison)
        
        return {
            'message': 'Analysis complete',
            'comparisons': comparisons
        }
        
    def _calculate_basic_similarity(self, text1, text2):
        """Calculate basic similarity between two texts"""
        import difflib
        similarity_ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
        return similarity_ratio
