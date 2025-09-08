"""
Model versioning and feedback loop system for ML models.
"""

import os
import json
import datetime
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
HISTORY_DIR = os.path.join(MODEL_DIR, 'history')
FEEDBACK_FILE = os.path.join(MODEL_DIR, 'feedback.json')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

class ModelManager:
    """
    Manages ML model versioning, training, and feedback.
    """
    
    def __init__(self, model_type='ai_detector'):
        """Initialize model manager"""
        self.model_type = model_type
        self.model_path = os.path.join(MODEL_DIR, f'{model_type}_model.joblib')
        self.current_version = self._get_current_version()
        self.feedback_data = self._load_feedback_data()
        
    def _get_current_version(self):
        """Get current model version"""
        version_file = os.path.join(MODEL_DIR, f'{self.model_type}_version.json')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version_data = json.load(f)
                return version_data.get('version', 1.0)
        else:
            # Initialize version file
            self._save_version(1.0)
            return 1.0
            
    def _save_version(self, version):
        """Save model version information"""
        version_file = os.path.join(MODEL_DIR, f'{self.model_type}_version.json')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        version_data = {
            'version': version,
            'timestamp': timestamp,
            'model_type': self.model_type
        }
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=4)
            
    def _load_feedback_data(self):
        """Load user feedback data for model improvement"""
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r') as f:
                return json.load(f)
        else:
            # Initialize empty feedback file
            feedback_data = {'ai_detector': [], 'plagiarism': []}
            with open(FEEDBACK_FILE, 'w') as f:
                json.dump(feedback_data, f, indent=4)
            return feedback_data
            
    def save_feedback(self, original_result, corrected_result, features):
        """
        Save user feedback for model improvement
        
        Args:
            original_result: Original model prediction
            corrected_result: User-corrected prediction
            features: Features used for the prediction
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        feedback_item = {
            'timestamp': timestamp,
            'original_result': original_result,
            'corrected_result': corrected_result,
            'features': features,
            'used_for_training': False
        }
        
        self.feedback_data[self.model_type].append(feedback_item)
        
        # Save feedback data
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(self.feedback_data, f, indent=4)
            
        return {'message': 'Feedback saved successfully'}
        
    def archive_current_model(self):
        """Archive current model before updating"""
        if os.path.exists(self.model_path):
            # Create archive filename with version
            archive_path = os.path.join(
                HISTORY_DIR, 
                f'{self.model_type}_v{self.current_version}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.joblib'
            )
            # Copy current model to archive
            if os.path.exists(self.model_path):
                model = joblib.load(self.model_path)
                joblib.dump(model, archive_path)
                return archive_path
        return None
        
    def update_model_with_feedback(self):
        """
        Update model using collected feedback
        
        Returns:
            Dictionary with update status and metrics
        """
        # Get unused feedback data
        unused_feedback = [item for item in self.feedback_data[self.model_type] if not item['used_for_training']]
        
        if len(unused_feedback) < 5:
            return {
                'status': 'not_updated',
                'message': 'Not enough feedback data collected for model update',
                'feedback_count': len(unused_feedback)
            }
            
        # Archive current model
        archived_path = self.archive_current_model()
        
        # Load current model if exists
        if os.path.exists(self.model_path):
            model = joblib.load(self.model_path)
        else:
            # Create a new model
            model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
        # Prepare training data from feedback
        X = []
        y = []
        
        for item in unused_feedback:
            # Get features and corrected output
            features = item['features']
            if not features:
                continue
                
            # For AI detector, we use corrected ai_score converted to binary (>50 = AI)
            if self.model_type == 'ai_detector':
                corrected_score = item['corrected_result'].get('ai_score', 50)
                label = 1 if corrected_score > 50 else 0
                X.append(list(features.values()))
                y.append(label)
                
            # For plagiarism, we use corrected similarity_score
            elif self.model_type == 'plagiarism':
                # Skip feedback that doesn't have proper structure
                if 'corrected_result' not in item or 'similarity_score' not in item['corrected_result']:
                    continue
                    
                X.append(list(features.values()))
                y.append(item['corrected_result']['similarity_score'] / 100.0)  # Normalize to 0-1
        
        if not X or not y:
            return {
                'status': 'not_updated',
                'message': 'No valid feedback data found for training',
                'feedback_count': len(unused_feedback)
            }
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model with new data
        model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        
        # Save updated model
        joblib.dump(model, self.model_path)
        
        # Update version
        new_version = self.current_version + 0.1
        self._save_version(new_version)
        self.current_version = new_version
        
        # Mark feedback as used
        for item in unused_feedback:
            item['used_for_training'] = True
            
        # Save updated feedback data
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(self.feedback_data, f, indent=4)
            
        return {
            'status': 'updated',
            'message': f'Model updated to version {self.current_version}',
            'version': self.current_version,
            'accuracy': accuracy,
            'training_samples': len(X),
            'archived_path': archived_path
        }
        
    def get_model_info(self):
        """Get information about the current model"""
        # Count feedback items
        feedback_count = len(self.feedback_data.get(self.model_type, []))
        unused_count = len([item for item in self.feedback_data.get(self.model_type, []) if not item['used_for_training']])
        
        # Get model file size
        model_size = os.path.getsize(self.model_path) / (1024 * 1024) if os.path.exists(self.model_path) else 0
        
        # List archived versions
        archives = [f for f in os.listdir(HISTORY_DIR) if f.startswith(f'{self.model_type}_v')]
        
        return {
            'model_type': self.model_type,
            'current_version': self.current_version,
            'feedback_count': feedback_count,
            'unused_feedback': unused_count,
            'model_size_mb': round(model_size, 2),
            'archived_versions': len(archives),
            'archives': archives
        }
