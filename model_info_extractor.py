#!/usr/bin/env python3
"""
Extract and display model information for analysis
Run this script and share the output
"""

import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime

def extract_model_info():
    """Extract all model information for sharing"""
    
    print("="*60)
    print("MLB MODEL INFORMATION EXTRACTION")
    print("="*60)
    
    # Check which files exist
    model_files = [
        'final_mlb_model.pkl',
        'mlb_model.pkl', 
        'model_features.pkl',
        'team_stats.pkl',
        'prediction_results.csv',
        'historical_mlb_games.csv'
    ]
    
    print("\n1. FILE AVAILABILITY:")
    available_files = []
    for file in model_files:
        exists = os.path.exists(file)
        print(f"   {file}: {'✅ EXISTS' if exists else '❌ MISSING'}")
        if exists:
            available_files.append(file)
    
    # Extract model information
    print("\n2. MODEL DETAILS:")
    try:
        # Try different model file names
        model = None
        for model_file in ['final_mlb_model.pkl', 'mlb_model.pkl']:
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                print(f"   Loaded model from: {model_file}")
                break
        
        if model:
            print(f"   Model type: {type(model).__name__}")
            
            # Get model parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                print(f"   Key parameters:")
                for key, value in list(params.items())[:5]:
                    print(f"     {key}: {value}")
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                print(f"   Feature importances: {model.feature_importances_}")
            
            # Get number of features
            if hasattr(model, 'n_features_in_'):
                print(f"   Number of features: {model.n_features_in_}")
                
    except Exception as e:
        print(f"   Error loading model: {e}")
    
    # Extract feature information
    print("\n3. FEATURES:")
    try:
        if os.path.exists('model_features.pkl'):
            features = joblib.load('model_features.pkl')
            print(f"   Feature list: {features}")
        else:
            print("   No model_features.pkl found")
    except Exception as e:
        print(f"   Error loading features: {e}")
    
    # Extract team stats
    print("\n4. TEAM STATISTICS:")
    try:
        if os.path.exists('team_stats.pkl'):
            team_stats = joblib.load('team_stats.pkl')
            print(f"   Number of teams: {len(team_stats)}")
            
            # Show sample team
            sample_team = list(team_stats.keys())[0]
            sample_stats = team_stats[sample_team]
            print(f"   Sample team ({sample_team}): {sample_stats}")
            
            # Show all stat types
            if team_stats:
                all_keys = set()
                for team_data in team_stats.values():
                    all_keys.update(team_data.keys())
                print(f"   Available stats: {list(all_keys)}")
                
    except Exception as e:
        print(f"   Error loading team stats: {e}")
    
    # Extract results data
    print("\n5. PREDICTION RESULTS:")
    try:
        if os.path.exists('prediction_results.csv'):
            results = pd.read_csv('prediction_results.csv')
            print(f"   Total predictions: {len(results)}")
            print(f"   Date range: {results['date'].min()} to {results['date'].max()}")
            print(f"   Columns: {list(results.columns)}")
            
            # Accuracy by confidence
            if 'confidence' in results.columns and 'correct' in results.columns:
                accuracy_by_conf = results.groupby('confidence')['correct'].agg(['count', 'sum', 'mean'])
                print(f"   Accuracy by confidence:")
                for conf in accuracy_by_conf.index:
                    count = accuracy_by_conf.loc[conf, 'count']
                    correct = accuracy_by_conf.loc[conf, 'sum'] 
                    accuracy = accuracy_by_conf.loc[conf, 'mean']
                    print(f"     {conf}: {correct}/{count} ({accuracy:.1%})")
            
            # Show sample row
            print(f"   Sample prediction:")
            sample_row = results.iloc[0].to_dict()
            for key, value in sample_row.items():
                print(f"     {key}: {value}")
                
    except Exception as e:
        print(f"   Error loading results: {e}")
    
    # Extract historical data info
    print("\n6. HISTORICAL DATA:")
    try:
        if os.path.exists('historical_mlb_games.csv'):
            historical = pd.read_csv('historical_mlb_games.csv')
            print(f"   Total games: {len(historical)}")
            print(f"   Columns: {list(historical.columns)}")
            
            if 'date' in historical.columns:
                historical['date'] = pd.to_datetime(historical['date'])
                print(f"   Date range: {historical['date'].min().date()} to {historical['date'].max().date()}")
            
            # Show sample row
            print(f"   Sample game:")
            sample_game = historical.iloc[0].to_dict()
            for key, value in list(sample_game.items())[:8]:  # First 8 columns
                print(f"     {key}: {value}")
                
    except Exception as e:
        print(f"   Error loading historical data: {e}")
    
    print(f"\n7. CURRENT DIRECTORY FILES:")
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    
    print(f"   CSV files: {csv_files}")
    print(f"   PKL files: {pkl_files}")
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("Share this entire output for model analysis")
    print("="*60)

if __name__ == "__main__":
    extract_model_info()