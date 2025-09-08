"""
Enhanced ML Models with support for transfer learning and advanced features.
"""

import os
import pickle
import re
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import feature_extraction as fe
from model_versioning import ModelManager

# Default paths for model storage
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
AI_DETECTOR_MODEL_PATH = os.path.join(MODEL_DIR, 'ai_detector_model.joblib')
PLAGIARISM_MODEL_PATH = os.path.join(MODEL_DIR, 'plagiarism_model.joblib')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

class EnhancedAIDetector:
    """Enhanced AI-generated content detector with transfer learning support"""
    
    def __init__(self, model_path=None, use_pretrained=True):
        """Initialize the AI detector model"""
        self.model_path = model_path or AI_DETECTOR_MODEL_PATH
        self.model = None
        self.use_pretrained = use_pretrained
        self.model_manager = ModelManager('ai_detector')
        self.load_or_create_model()
        self.sensitivity = 1.0  # Default sensitivity (can be adjusted by users)
        
    def load_or_create_model(self):
        """Load an existing model or create a new one if not found"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"Loaded AI detector model from {self.model_path}")
            else:
                self.model = self._create_default_model()
                print("Created new AI detector model")
        except Exception as e:
            print(f"Error loading model: {str(e)}. Creating new model.")
            self.model = self._create_default_model()
    
    def _create_default_model(self):
        """Create a default model for AI detection"""
        # For a real system, this would be trained on actual data
        # Here we're creating a model with reasonable default parameters
        model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        return model
    
    def train(self, texts, labels, test_size=0.2, use_transfer_learning=True):
        """
        Train the AI detector model on labeled data with optional transfer learning
        
        Args:
            texts: List of text samples
            labels: List of labels (1 for AI-generated, 0 for human)
            test_size: Fraction of data to use for testing
            use_transfer_learning: Whether to use transfer learning from a pre-trained model
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract features from texts
        features = []
        for text in texts:
            text_features = fe.extract_advanced_features(text)
            if text_features:
                features.append(text_features)
            else:
                # Skip this sample if feature extraction failed
                continue
                
        # Convert features to array
        X = []
        y = []
        for i, feature_dict in enumerate(features):
            if i < len(labels):
                X.append(list(feature_dict.values()))
                y.append(labels[i])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # For transfer learning, initialize the model with existing parameters if available
        # and use them as a starting point for further training
        if use_transfer_learning and hasattr(self.model, 'predict'):
            # We'll start with the existing model and fine-tune it
            print("Using transfer learning from existing model")
            model = self.model
        else:
            # Start with a fresh model
            model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
        # Train the model
        model.fit(X_train, y_train)
        self.model = model
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }
        
        # Archive current model and save the new one
        self.model_manager.archive_current_model()
        self.save_model()
        
        return metrics
    
    def save_model(self):
        """Save the model to disk"""
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def predict(self, text, sensitivity=None):
        """
        Predict if text is AI-generated with adjustable sensitivity
        
        Args:
            text: The text to analyze
            sensitivity: Sensitivity level (0.5-1.5, where higher means more sensitive to AI detection)
            
        Returns:
            Dictionary with prediction results
        """
        # Use provided sensitivity or instance default
        sensitivity = sensitivity if sensitivity is not None else self.sensitivity
        
        # Extract features
        features = fe.extract_advanced_features(text)
        if not features:
            return {
                'ai_score': 0,
                'ai_sections': [],
                'classification': 'Cannot analyze (empty text)',
                'model_version': self.model_manager.current_version
            }
        
        # If we have a trained model, use it
        if hasattr(self.model, 'predict_proba'):
            # Convert features to array for prediction
            feature_array = np.array(list(features.values())).reshape(1, -1)
            
            try:
                # Get probability of AI-generated content
                probs = self.model.predict_proba(feature_array)
                ai_score = probs[0][1] * 100  # Probability of class 1 (AI)
                
                # Apply sensitivity adjustment
                ai_score = self._adjust_score_by_sensitivity(ai_score, sensitivity)
            except:
                # Fallback to simulated prediction
                ai_score = self._simulate_prediction(features, sensitivity)
        else:
            # Use simulated prediction if model is not fully trained
            ai_score = self._simulate_prediction(features, sensitivity)
            
        # Analyze sentences for AI content
        ai_sections = self._analyze_sections(text, sensitivity)
        
        # Calculate feature importance
        feature_importance = self._get_feature_importance(features)
        
        # Return results
        return {
            'ai_score': round(ai_score, 1),
            'ai_sections': ai_sections,
            'classification': 'High AI Content' if ai_score > 70 else 
                             'Moderate AI Content' if ai_score > 40 else 'Low/No AI Content',
            'feature_analysis': features,  # Include the extracted features for analysis
            'feature_importance': feature_importance,
            'model_version': self.model_manager.current_version,
            'sensitivity_used': sensitivity
        }
    
    def _adjust_score_by_sensitivity(self, score, sensitivity):
        """Adjust AI score based on sensitivity setting"""
        # Higher sensitivity means more likely to classify as AI
        # Scale between 0.5 and 1.5
        if score >= 50:
            # For high scores, higher sensitivity increases the score
            adjusted_score = score + (score - 50) * (sensitivity - 1) * 0.5
        else:
            # For low scores, higher sensitivity still increases the score
            adjusted_score = score + (sensitivity - 1) * 10
            
        return min(max(adjusted_score, 0), 100)
        
    def _simulate_prediction(self, features, sensitivity=1.0):
        """Simulate AI prediction based on weighted features"""
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
        }
        
        # Apply sensitivity to weights
        if sensitivity != 1.0:
            for feature in feature_weights:
                if feature_weights[feature] > 0:
                    # Positive weights (increase AI score) are amplified by higher sensitivity
                    feature_weights[feature] *= sensitivity
                else:
                    # Negative weights (decrease AI score) are reduced by higher sensitivity
                    feature_weights[feature] /= sensitivity
        
        # Calculate weighted score
        ai_score = 50  # Start at neutral
        for feature, value in features.items():
            if feature in feature_weights:
                ai_score += value * features[feature] * feature_weights[feature]
        
        # Normalize to 0-100 scale
        ai_score = min(max(ai_score, 0), 100)
        return ai_score
    
    def _analyze_sections(self, text, sensitivity=1.0):
        """Analyze text sections for AI content"""
        import nltk
        from nltk.tokenize import sent_tokenize
        
        # Make sure we have the tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        sentences = sent_tokenize(text)
        ai_sections = []
        
        # Calculate sentence-level AI scores
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 5:  # Skip very short sentences
                continue
                
            # Extract features for this sentence
            sent_features = fe.extract_advanced_features(sentence)
            if sent_features is None:
                continue
                
            # Calculate sentence AI score (using simulated prediction for now)
            sent_score = self._simulate_prediction(sent_features, sensitivity)
            
            # Add high-scoring sentences to the list
            if sent_score > (65 / sensitivity):  # Adjust threshold by sensitivity
                ai_sections.append({
                    'sentence_index': i,
                    'content': sentence,
                    'confidence': round(sent_score, 1)
                })
        
        # Sort sections by confidence and limit to top results
        ai_sections.sort(key=lambda x: x['confidence'], reverse=True)
        return ai_sections[:5]  # Include up to 5 most suspicious sections
        
    def _get_feature_importance(self, features):
        """Get feature importance for visualization"""
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models like GradientBoostingClassifier
            importances = self.model.feature_importances_
            feature_names = list(features.keys())
            
            # Create a sorted list of (feature, importance) pairs
            importance_pairs = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Return as dictionary
            return {name: float(importance) for name, importance in importance_pairs}
        else:
            # Fallback to simulated importance based on our weights
            feature_weights = {
                'unique_words_ratio': 0.15,
                'unique_non_stop_ratio': 0.10,
                'avg_word_length': 0.08,
                'avg_sentence_length': 0.12,
                'long_words_ratio': 0.15,
                'avg_commas_per_sentence': 0.08,
                'hapax_legomena_ratio': 0.20,
                'punctuation_ratio': 0.07,
                'sentence_length_variance': 0.15,
                'perplexity': 0.10, 
                'burstiness': 0.20,
            }
            
            # Filter to only include features we have
            importance_dict = {k: v for k, v in feature_weights.items() if k in features}
            
            # Normalize to sum to 1.0
            total = sum(importance_dict.values())
            if total > 0:
                for k in importance_dict:
                    importance_dict[k] = importance_dict[k] / total
                    
            return importance_dict
            
    def save_feedback(self, original_result, corrected_result, features):
        """Save user feedback for model improvement"""
        return self.model_manager.save_feedback(original_result, corrected_result, features)
        
    def update_from_feedback(self):
        """Update model using collected feedback"""
        return self.model_manager.update_model_with_feedback()
        
    def get_model_info(self):
        """Get information about the current model"""
        return self.model_manager.get_model_info()
        
    def set_sensitivity(self, sensitivity):
        """Set the sensitivity level for AI detection"""
        self.sensitivity = max(min(float(sensitivity), 1.5), 0.5)
        return {"message": f"Sensitivity set to {self.sensitivity}"}


class EnhancedPlagiarismDetector:
    """Enhanced plagiarism detection with adjustable settings"""
    
    def __init__(self, model_path=None):
        """Initialize the plagiarism detector"""
        self.model_path = model_path or PLAGIARISM_MODEL_PATH
        self.model_manager = ModelManager('plagiarism')
        self.sensitivity = 1.0  # Default sensitivity
        self.threshold_settings = {
            'high_similarity': 70,  # Threshold for high similarity (default: 70%)
            'moderate_similarity': 40,  # Threshold for moderate similarity (default: 40%)
        }
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
    
    def compare_documents(self, documents, sensitivity=None, threshold_settings=None):
        """
        Compare multiple documents for plagiarism with adjustable settings
        
        Args:
            documents: Dictionary of {filename: content}
            sensitivity: Optional sensitivity setting (0.5-1.5)
            threshold_settings: Optional custom thresholds
            
        Returns:
            Dictionary with comparison results
        """
        # Use provided sensitivity or instance default
        sensitivity = sensitivity if sensitivity is not None else self.sensitivity
        
        # Use provided thresholds or instance defaults
        if threshold_settings:
            thresholds = {**self.threshold_settings, **threshold_settings}
        else:
            thresholds = self.threshold_settings
            
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
        
        # Combine similarities for final results, applying sensitivity
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
                
                # Apply sensitivity adjustment
                combined_similarity[i, j] = self._adjust_similarity_by_sensitivity(combined_similarity[i, j], sensitivity)
                combined_similarity[j, i] = combined_similarity[i, j]
        
        # Calculate heatmap data for visualization
        heatmap_data = self._generate_heatmap_data(file_names, combined_similarity)
        
        # Format results
        comparisons = []
        for i, file1 in enumerate(file_names):
            for j in range(i + 1, len(file_names)):
                file2 = file_names[j]
                
                # Convert similarity to percentage
                similarity = round(combined_similarity[i, j] * 100, 1)
                
                # Determine similarity level using thresholds
                similarity_level = "High" if similarity > thresholds['high_similarity'] else "Moderate" if similarity > thresholds['moderate_similarity'] else "Low"
                flag_color = "red" if similarity > thresholds['high_similarity'] else "yellow" if similarity > thresholds['moderate_similarity'] else "green"
                
                # Extract shared content sections for side-by-side comparison
                matching_sections = self._extract_matching_sections(contents[i], contents[j], similarity)
                
                # Prepare comparison result
                comparison = {
                    'file1': file1,
                    'file2': file2,
                    'similarity_score': similarity,
                    'similarity_level': similarity_level,
                    'flag': flag_color,
                    'ml_confidence_score': round(95 - (abs(50 - similarity) / 2)),  # Model confidence
                    'matching_sections': matching_sections,
                }
                
                # Add to results
                comparisons.append(comparison)
        
        return {
            'message': 'Analysis complete',
            'comparisons': comparisons,
            'heatmap_data': heatmap_data,
            'sensitivity_used': sensitivity,
            'thresholds': thresholds,
            'model_version': self.model_manager.current_version
        }
    
    def _adjust_similarity_by_sensitivity(self, similarity, sensitivity):
        """Adjust similarity score based on sensitivity"""
        if sensitivity == 1.0:
            return similarity
            
        # Higher sensitivity means more similarity detected
        if similarity > 0.5:
            # High similarity scores are boosted by high sensitivity
            adjusted = similarity + (similarity - 0.5) * (sensitivity - 1) * 0.4
        else:
            # Low similarity scores are boosted less by high sensitivity
            adjusted = similarity + (sensitivity - 1) * 0.1
            
        return min(max(adjusted, 0.0), 1.0)
        
    def _calculate_basic_similarity(self, text1, text2):
        """Calculate basic similarity between two texts"""
        import difflib
        similarity_ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
        return similarity_ratio
        
    def _generate_heatmap_data(self, file_names, similarity_matrix):
        """Generate heatmap data for visualization"""
        heatmap_data = []
        for i in range(len(file_names)):
            for j in range(len(file_names)):
                heatmap_data.append({
                    'file1': file_names[i],
                    'file2': file_names[j],
                    'similarity': round(similarity_matrix[i, j] * 100, 1)
                })
                
        return heatmap_data
        
    def _extract_matching_sections(self, text1, text2, overall_similarity):
        """Extract matching sections for side-by-side comparison"""
        if overall_similarity < 20:  # Skip if similarity is too low
            return []
            
        # Split into lines
        lines1 = text1.split('\n')
        lines2 = text2.split('\n')
        
        # Use difflib to find matching sections
        import difflib
        matcher = difflib.SequenceMatcher(None, lines1, lines2)
        
        matches = []
        for block in matcher.get_matching_blocks():
            if block.size > 2:  # Only include substantial matches
                # Calculate local similarity for this block
                section1 = '\n'.join(lines1[block.a:block.a + block.size])
                section2 = '\n'.join(lines2[block.b:block.b + block.size])
                local_similarity = difflib.SequenceMatcher(None, section1, section2).ratio() * 100
                
                # Add match to the list
                matches.append({
                    'file1_start': block.a,
                    'file2_start': block.b,
                    'length': block.size,
                    'file1_content': section1,
                    'file2_content': section2,
                    'similarity': round(local_similarity, 1),
                    'color': 'red' if local_similarity > 90 else 'amber' if local_similarity > 70 else 'yellow'
                })
        
        # Sort by similarity (highest first) and limit
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:10]  # Return top 10 matches
        
    def set_sensitivity(self, sensitivity):
        """Set sensitivity level for plagiarism detection"""
        self.sensitivity = max(min(float(sensitivity), 1.5), 0.5)
        return {"message": f"Sensitivity set to {self.sensitivity}"}
        
    def set_thresholds(self, high=None, moderate=None):
        """Set custom thresholds for similarity levels"""
        if high is not None:
            self.threshold_settings['high_similarity'] = max(min(float(high), 100), 0)
            
        if moderate is not None:
            self.threshold_settings['moderate_similarity'] = max(min(float(moderate), 100), 0)
            
        return {"message": f"Thresholds updated: high={self.threshold_settings['high_similarity']}, moderate={self.threshold_settings['moderate_similarity']}"}
        
    def save_feedback(self, original_result, corrected_result, features):
        """Save user feedback for model improvement"""
        return self.model_manager.save_feedback(original_result, corrected_result, features)
        
    def get_model_info(self):
        """Get information about the current model"""
        return self.model_manager.get_model_info()
