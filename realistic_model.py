import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime, timedelta

def simulate_real_betting_conditions():
    """Simulate realistic betting where we only know past data"""
    
    df = pd.read_csv('historical_mlb_games.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print("Simulating real-time betting conditions...")
    print("- Each prediction uses ONLY data available before that game")
    print("- Rolling window retraining")
    print("- Proper time-series validation\n")
    
    # Start predictions after we have enough history
    start_idx = 1000
    predictions = []
    actual_results = []
    
    # Rolling window parameters
    window_size = 800  # Use last 800 games for training
    retrain_frequency = 100  # Retrain every 100 games
    
    for idx in range(start_idx, len(df)):
        if idx % 500 == 0:
            print(f"Processing game {idx}/{len(df)} ({df.iloc[idx]['date'].date()})")
        
        current_game = df.iloc[idx]
        historical_data = df.iloc[max(0, idx-window_size):idx]  # Only past data
        
        # Calculate features using only historical data
        home_team = current_game['home_team']
        away_team = current_game['away_team']
        
        # Home team recent performance (last 20 games)
        home_recent = historical_data[
            (historical_data['home_team'] == home_team) | 
            (historical_data['away_team'] == home_team)
        ].tail(20)
        
        # Away team recent performance (last 20 games)  
        away_recent = historical_data[
            (historical_data['home_team'] == away_team) | 
            (historical_data['away_team'] == away_team)
        ].tail(20)
        
        # Skip if insufficient data
        if len(home_recent) < 10 or len(away_recent) < 10:
            continue
            
        # Calculate win rates
        home_wins = len(home_recent[
            ((home_recent['home_team'] == home_team) & (home_recent['winner'] == 'home')) |
            ((home_recent['away_team'] == home_team) & (home_recent['winner'] == 'away'))
        ])
        home_win_rate = home_wins / len(home_recent)
        
        away_wins = len(away_recent[
            ((away_recent['home_team'] == away_team) & (away_recent['winner'] == 'home')) |
            ((away_recent['away_team'] == away_team) & (away_recent['winner'] == 'away'))
        ])
        away_win_rate = away_wins / len(away_recent)
        
        # Calculate run differentials
        home_runs_scored, home_runs_allowed = 0, 0
        for _, game in home_recent.iterrows():
            if game['home_team'] == home_team:
                home_runs_scored += game['home_score']
                home_runs_allowed += game['away_score']
            else:
                home_runs_scored += game['away_score'] 
                home_runs_allowed += game['home_score']
        
        away_runs_scored, away_runs_allowed = 0, 0
        for _, game in away_recent.iterrows():
            if game['home_team'] == away_team:
                away_runs_scored += game['home_score']
                away_runs_allowed += game['away_score']
            else:
                away_runs_scored += game['away_score']
                away_runs_allowed += game['home_score']
        
        home_run_diff = (home_runs_scored - home_runs_allowed) / len(home_recent)
        away_run_diff = (away_runs_scored - away_runs_allowed) / len(away_recent)
        
        # Create feature vector
        features = [
            1,  # home_field
            home_win_rate - away_win_rate,  # win_rate_advantage
            home_win_rate,
            away_win_rate,
            home_run_diff - away_run_diff,  # run_diff_advantage
            home_run_diff,
            away_run_diff
        ]
        
        # Train model if it's time to retrain
        if idx == start_idx or (idx - start_idx) % retrain_frequency == 0:
            print(f"  Retraining model at game {idx}")
            
            # Prepare training data from historical games
            train_features = []
            train_labels = []
            
            for train_idx in range(max(500, idx-window_size), idx-50):  # Leave gap to avoid overlap
                train_game = df.iloc[train_idx]
                train_historical = df.iloc[max(0, train_idx-200):train_idx]
                
                # Calculate same features for training game
                t_home = train_game['home_team']
                t_away = train_game['away_team']
                
                t_home_recent = train_historical[
                    (train_historical['home_team'] == t_home) | 
                    (train_historical['away_team'] == t_home)
                ].tail(20)
                
                t_away_recent = train_historical[
                    (train_historical['home_team'] == t_away) | 
                    (train_historical['away_team'] == t_away)
                ].tail(20)
                
                if len(t_home_recent) < 10 or len(t_away_recent) < 10:
                    continue
                
                # Same feature calculation as above
                t_home_wins = len(t_home_recent[
                    ((t_home_recent['home_team'] == t_home) & (t_home_recent['winner'] == 'home')) |
                    ((t_home_recent['away_team'] == t_home) & (t_home_recent['winner'] == 'away'))
                ])
                t_home_wr = t_home_wins / len(t_home_recent)
                
                t_away_wins = len(t_away_recent[
                    ((t_away_recent['home_team'] == t_away) & (t_away_recent['winner'] == 'home')) |
                    ((t_away_recent['away_team'] == t_away) & (t_away_recent['winner'] == 'away'))
                ])
                t_away_wr = t_away_wins / len(t_away_recent)
                
                # Run differentials for training
                t_h_scored, t_h_allowed = 0, 0
                for _, g in t_home_recent.iterrows():
                    if g['home_team'] == t_home:
                        t_h_scored += g['home_score']
                        t_h_allowed += g['away_score']
                    else:
                        t_h_scored += g['away_score']
                        t_h_allowed += g['home_score']
                
                t_a_scored, t_a_allowed = 0, 0
                for _, g in t_away_recent.iterrows():
                    if g['home_team'] == t_away:
                        t_a_scored += g['home_score']
                        t_a_allowed += g['away_score']
                    else:
                        t_a_scored += g['away_score']
                        t_a_allowed += g['home_score']
                
                t_h_diff = (t_h_scored - t_h_allowed) / len(t_home_recent)
                t_a_diff = (t_a_scored - t_a_allowed) / len(t_away_recent)
                
                train_features.append([
                    1, t_home_wr - t_away_wr, t_home_wr, t_away_wr,
                    t_h_diff - t_a_diff, t_h_diff, t_a_diff
                ])
                train_labels.append(1 if train_game['winner'] == 'home' else 0)
            
            # Train model
            if len(train_features) > 100:
                model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
                model.fit(train_features, train_labels)
        
        # Make prediction
        if 'model' in locals():
            prediction_proba = model.predict_proba([features])[0][1]
            prediction = 1 if prediction_proba > 0.5 else 0
        else:
            prediction = 1  # Default to home if no model yet
            prediction_proba = 0.54  # Home field advantage
        
        predictions.append(prediction)
        actual_results.append(1 if current_game['winner'] == 'home' else 0)
    
    # Calculate realistic accuracy
    accuracy = accuracy_score(actual_results, predictions)
    
    print(f"\n{'='*60}")
    print(f"REALISTIC TIME-SERIES MODEL RESULTS")
    print(f"{'='*60}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Analyze by confidence level
    results_df = pd.DataFrame({
        'prediction': predictions,
        'actual': actual_results,
        'correct': [p == a for p, a in zip(predictions, actual_results)],
        'game_idx': range(start_idx, start_idx + len(predictions))
    })
    
    # Add dates
    results_df['date'] = [df.iloc[idx]['date'] for idx in results_df['game_idx']]
    
    overall_accuracy = results_df['correct'].mean()
    
    print(f"\nRESULTS BY YEAR:")
    for year in sorted(results_df['date'].dt.year.unique()):
        year_results = results_df[results_df['date'].dt.year == year]
        year_accuracy = year_results['correct'].mean()
        print(f"  {year}: {len(year_results)} games, {year_accuracy:.1%} accuracy")
    
    if overall_accuracy > 0.52:
        print(f"\n🎯 GOOD NEWS: {overall_accuracy:.1%} beats random chance and typical home field!")
        print("This is a realistic edge you could potentially exploit.")
    else:
        print(f"\n📊 REALISTIC: {overall_accuracy:.1%} reflects the true difficulty of sports prediction.")
        print("Even sophisticated models struggle against efficient markets.")
    
    return results_df

if __name__ == "__main__":
    print("Building realistic time-series betting model...")
    print("This simulates actual betting conditions with proper data constraints.\n")
    
    proceed = input("This will take 5-10 minutes. Continue? (y/n): ").lower()
    if proceed == 'y':
        results = simulate_real_betting_conditions()
        
        print(f"\n🎯 FINAL TAKEAWAY:")
        print("This accuracy represents what you could realistically expect")
        print("if you were actually betting with this model in real-time.")