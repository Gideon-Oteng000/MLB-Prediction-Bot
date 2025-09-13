#!/usr/bin/env python3
"""
OPTIMIZED PROFESSIONAL MLB PREDICTION SYSTEM
Optimized confidence thresholds based on performance analysis
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime

def get_optimized_confidence(home_win_prob):
    """
    Optimized confidence levels based on performance analysis
    HIGH threshold: 0.120 distance from 50%
    MEDIUM threshold: 0.060 distance from 50%
    """
    distance_from_50 = abs(home_win_prob - 0.5)
    
    if distance_from_50 >= 0.120:
        return 'HIGH'
    elif distance_from_50 >= 0.060:
        return 'MEDIUM'
    else:
        return 'LOW'

def load_professional_models():
    """Load the professional model components"""
    try:
        model = joblib.load('professional_mlb_model.pkl')
        scaler = joblib.load('professional_mlb_scaler.pkl')
        features = joblib.load('professional_features.pkl')
        return model, scaler, features
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

def make_optimized_prediction(home_team, away_team, game_date=None):
    """
    Make prediction using optimized professional system
    
    Note: This is a simplified version for demonstration.
    For full functionality, integrate with the complete feature calculation
    from the professional system.
    """
    
    # Load models
    model, scaler, features = load_professional_models()
    if model is None:
        return None
    
    if game_date is None:
        game_date = datetime.now()
    
    # For demonstration, create simplified features
    # In production, use the full feature calculation pipeline
    feature_dict = {f: 0.0 for f in features}
    
    # Add some realistic baseline values
    feature_dict.update({
        'home_field': 1.0,
        'win_rate_advantage': 0.02,  # Slight home advantage
        'run_diff_advantage': 0.1,
        'home_advantage': 0.04,
        'season_progress': 0.75,
        'power_rating_diff': 0.01,
        'h2h_home_win_rate': 0.52,
        'momentum_advantage': 0.0,
        'quality_advantage': 0.01
    })
    
    # Convert to DataFrame and prepare
    feature_df = pd.DataFrame([feature_dict])
    feature_df = feature_df[features].fillna(0.0)
    
    # Scale and predict
    try:
        feature_scaled = scaler.transform(feature_df)
        home_win_prob = model.predict_proba(feature_scaled)[0][1]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None
    
    # Apply optimized confidence
    confidence = get_optimized_confidence(home_win_prob)
    predicted_winner = home_team if home_win_prob > 0.5 else away_team
    
    # Generate recommendation
    recommendation = get_betting_recommendation(home_win_prob, confidence, home_team, away_team)
    
    return {
        'home_win_probability': home_win_prob,
        'confidence': confidence,
        'predicted_winner': predicted_winner,
        'recommendation': recommendation,
        'model_version': 'professional_v1.1_optimized',
        'home_team': home_team,
        'away_team': away_team
    }

def get_betting_recommendation(home_win_prob, confidence, home_team, away_team):
    """Generate betting recommendation"""
    
    if confidence == 'HIGH':
        if home_win_prob > 0.5:
            return f"STRONG BET: {home_team} ({home_win_prob:.1%} confidence)"
        else:
            return f"STRONG BET: {away_team} ({1-home_win_prob:.1%} confidence)"
    
    elif confidence == 'MEDIUM':
        if home_win_prob > 0.5:
            return f"MODERATE BET: {home_team} ({home_win_prob:.1%} confidence)"
        else:
            return f"MODERATE BET: {away_team} ({1-home_win_prob:.1%} confidence)"
    
    else:
        return f"SKIP: Too close to call ({home_win_prob:.1%} vs {1-home_win_prob:.1%})"

def test_optimized_system():
    """Test the optimized system"""
    print("Testing optimized professional system...")
    
    # Test prediction
    result = make_optimized_prediction("New York Yankees", "Boston Red Sox")
    if result:
        print(f"Sample prediction: {result}")
        return True
    else:
        print("Test failed")
        return False

if __name__ == "__main__":
    test_optimized_system()
