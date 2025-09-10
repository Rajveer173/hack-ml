import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import random
from flask import Blueprint, jsonify, request, current_app

# This file simulates ML-based visualization functions for the demo
# In a real application, these would be more sophisticated

def generate_feature_visualization(analysis_data, visualization_type):
    """Generate visualizations based on analysis data"""
    plt.figure(figsize=(10, 6))
    
    if visualization_type == 'features':
        # Extract feature data from analysis
        if 'detailed_results' in analysis_data:
            features = analysis_data['detailed_results']
        else:
            # Generate random feature data for demo
            features = {
                'sentence_complexity': random.uniform(0.2, 0.9),
                'vocabulary_richness': random.uniform(0.3, 0.8),
                'coherence_score': random.uniform(0.4, 0.9),
                'repetition_patterns': random.uniform(0.1, 0.7),
                'semantic_consistency': random.uniform(0.5, 0.9),
                'language_structure': random.uniform(0.3, 0.9),
                'predictable_patterns': random.uniform(0.2, 0.8),
                'content_originality': random.uniform(0.4, 0.9)
            }
        
        # Create bar chart
        feature_names = list(features.keys())
        feature_values = [features[key] for key in feature_names]
        
        # Sort by value for better visualization
        sorted_indices = np.argsort(feature_values)
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_values = [feature_values[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(sorted_names))
        plt.barh(y_pos, sorted_values, align='center')
        plt.yticks(y_pos, [name.replace('_', ' ').title() for name in sorted_names])
        plt.xlabel('Score')
        plt.title('Feature Analysis')
        
    elif visualization_type == 'distribution':
        # Generate a distribution visualization
        # In a real app, this would show the distribution of scores across the corpus
        mu, sigma = 0.5, 0.15
        x = np.linspace(0, 1, 100)
        y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        # Mark the current analysis on the distribution
        current_score = analysis_data.get('probability', 0.5)
        
        plt.plot(x, y, 'b-', linewidth=2)
        plt.axvline(x=current_score, color='r', linestyle='--', linewidth=2)
        
        # Add labels
        plt.xlabel('AI Probability Score')
        plt.ylabel('Density')
        plt.title('Score Distribution')
        plt.text(current_score + 0.02, max(y)/2, f'Your score: {current_score:.2f}', 
                 verticalalignment='center')
        
    elif visualization_type == 'comparison':
        # Create a comparison visualization
        categories = ['Vocabulary', 'Grammar', 'Structure', 'Coherence', 'Style']
        
        # Generate random data for human vs AI writing
        human_values = [random.uniform(0.5, 0.9) for _ in range(len(categories))]
        ai_values = [random.uniform(0.2, 0.7) for _ in range(len(categories))]
        
        # Number of variables
        N = len(categories)
        
        # Create angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add values for human and AI
        human_values += human_values[:1]
        ai_values += ai_values[:1]
        
        # Draw the plot
        ax = plt.subplot(111, polar=True)
        
        # Draw human values
        ax.plot(angles, human_values, 'b-', linewidth=2, label='Human')
        ax.fill(angles, human_values, 'b', alpha=0.1)
        
        # Draw AI values
        ax.plot(angles, ai_values, 'r-', linewidth=2, label='AI')
        ax.fill(angles, ai_values, 'r', alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Human vs AI Writing Characteristics')
    
    # Save to base64 string
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

def get_analysis_by_id(analysis_id):
    """Get analysis data by ID from history files"""
    from enhanced_api import HISTORY_DIR, USER_DATA_DIR, get_user_id_from_session
    
    # First try user-specific history if user is logged in
    user_id = get_user_id_from_session()
    
    if user_id:
        user_history_file = os.path.join(USER_DATA_DIR, str(user_id), 'history.json')
        if os.path.exists(user_history_file):
            try:
                with open(user_history_file, 'r') as f:
                    history = json.load(f)
                
                for entry in history:
                    if entry.get('id') == analysis_id:
                        return entry
            except Exception as e:
                current_app.logger.error(f"Error reading user history: {str(e)}")
    
    # Then try general history file
    general_history_file = os.path.join(HISTORY_DIR, 'analysis_history.json')
    if os.path.exists(general_history_file):
        try:
            with open(general_history_file, 'r') as f:
                history = json.load(f)
            
            for entry in history:
                if entry.get('id') == analysis_id:
                    return entry
        except Exception as e:
            current_app.logger.error(f"Error reading general history: {str(e)}")
    
    return None
