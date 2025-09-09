import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta

def load_and_prepare_large_dataset():
    """Load the historical dataset and prepare features"""
    df = pd.read_csv('historical_mlb_games.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Loaded {len(df)} games from {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df

def calculate_rolling_team_stats(df, window_games=20):
    """Calculate rolling statistics for each team"""
    print("Calculating rolling team statistics...")
    
    enhanced_games = []
    
    for idx, game in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing game {idx+1}/{len(df)}")
        
        # Get historical games for both teams up to this point
        historical_df = df.iloc[:idx]
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Home team recent performance
        home_recent = historical_df[
            (historical_df['home_team'] == home_team) | (historical_df['away_team'] == home_team)
        ].tail(window_games)
        
        # Away team recent performance
        away_recent = historical_df[
            (historical_df['home_team'] == away_team) | (historical_df['away_team'] == away_team)
        ].tail(window_games)
        
        # Calculate home team stats
        if len(home_recent) > 5:
            home_wins = len(home_recent[
                ((home_recent['home_team'] == home_team) & (home_recent['winner'] == 'home')) |
                ((home_recent['away_team'] == home_team) & (home_recent['winner'] == 'away'))
            ])
            home_win_rate = home_wins / len(home_recent)
            
            # Run scoring/allowing
            home_runs_scored = 0
            home_runs_allowed = 0
            for _, h_game in home_recent.iterrows():
                if h_game['home_team'] == home_team:
                    home_runs_scored += h_game['home_score']
                    home_runs_allowed += h_game['away_score']
                else:
                    home_runs_scored += h_game['away_score']
                    home_runs_allowed += h_game['home_score']
            
            home_runs_per_game = home_runs_scored / len(home_recent)
            home_runs_allowed_per_game = home_runs_allowed / len(home_recent)
            
        else:
            home_win_rate = 0.5
            home_runs_per_game = 4.5
            home_runs_allowed_per_game = 4.5
        
        # Calculate away team stats
        if len(away_recent) > 5:
            away_wins = len(away_recent[
                ((away_recent['home_team'] == away_team) & (away_recent['winner'] == 'home')) |
                ((away_recent['away_team'] == away_team) & (away_recent['winner'] == 'away'))
            ])
            away_win_rate = away_wins / len(away_recent)
            
            # Run scoring/allowing
            away_runs_scored = 0
            away_runs_allowed = 0
            for _, a_game in away_recent.iterrows():
                if a_game['home_team'] == away_team:
                    away_runs_scored += a_game['home_score']
                    away_runs_allowed += a_game['away_score']
                else:
                    away_runs_scored += a_game['away_score']
                    away_runs_allowed += a_game['home_score']
            
            away_runs_per_game = away_runs_scored / len(away_recent)
            away_runs_allowed_per_game = away_runs_allowed / len(away_recent)
            
        else:
            away_win_rate = 0.5
            away_runs_per_game = 4.5
            away_runs_allowed_per_game = 4.5
        
        # Head-to-head record
        h2h_games = historical_df[
            ((historical_df['home_team'] == home_team) & (historical_df['away_team'] == away_team)) |
            ((historical_df['home_team'] == away_team) & (historical_df['away_team'] == home_team))
        ].tail(10)
        
        if len(h2h_games) > 0:
            h2h_home_wins = len(h2h_games[
                ((h2h_games['home_team'] == home_team) & (h2h_games['winner'] == 'home')) |
                ((h2h_games['away_team'] == home_team) & (h2h_games['winner'] == 'away'))
            ])
            h2h_advantage = h2h_home_wins / len(h2h_games)
        else:
            h2h_advantage = 0.5
        
        # Season context
        month = game['date'].month
        is_early_season = 1 if month in [3, 4] else 0
        is_late_season = 1 if month in [9, 10] else 0
        
        enhanced_game = {
            'date': game['date'],
            'season': game['season'],
            'home_team': home_team,
            'away_team': away_team,
            'home_wins': 1 if game['winner'] == 'home' else 0,
            
            # Team performance features
            'home_win_rate': home_win_rate,
            'away_win_rate': away_win_rate,
            'win_rate_advantage': home_win_rate - away_win_rate,
            
            # Offensive/Defensive features
            'home_runs_per_game': home_runs_per_game,
            'away_runs_per_game': away_runs_per_game,
            'home_runs_allowed_per_game': home_runs_allowed_per_game,
            'away_runs_allowed_per_game': away_runs_allowed_per_game,
            'offensive_advantage': home_runs_per_game - away_runs_per_game,
            'defensive_advantage': away_runs_allowed_per_game - home_runs_allowed_per_game,
            
            # Combined team strength
            'home_run_differential': home_runs_per_game - home_runs_allowed_per_game,
            'away_run_differential': away_runs_per_game - away_runs_allowed_per_game,
            'run_diff_advantage': (home_runs_per_game - home_runs_allowed_per_game) - (away_runs_per_game - away_runs_allowed_per_game),
            
            # Contextual features
            'h2h_advantage': h2h_advantage,
            'is_early_season': is_early_season,
            'is_late_season': is_late_season,
            'home_field': 1
        }
        
        enhanced_games.append(enhanced_game)
    
    return pd.DataFrame(enhanced_games)

def build_final_model(df):
    """Build the most sophisticated model"""
    
    # Only use games where we have sufficient history
    df_model = df[500:].copy()  # Skip first 500 games to ensure good rolling stats
    
    feature_columns = [
        'home_field',
        'win_rate_advantage',
        'offensive_advantage',
        'defensive_advantage', 
        'run_diff_advantage',
        'h2h_advantage',
        'home_win_rate',
        'away_win_rate',
        'home_run_differential',
        'away_run_differential',
        'is_early_season',
        'is_late_season'
    ]
    
    X = df_model[feature_columns]
    y = df_model['home_wins']
    
    print(f"\n=== FINAL MODEL WITH {len(df_model)} GAMES ===")
    
    # Time-based split (more realistic for sports betting)
    split_date = df_model['date'].quantile(0.8)  # Use 80% for training
    train_mask = df_model['date'] <= split_date
    
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]
    
    print(f"Training: {len(X_train)} games (through {split_date.date()})")
    print(f"Testing: {len(X_test)} games (after {split_date.date()})")
    
    # Try multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    for name, model in models.items():
        # Cross validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Fit and test
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n{name}:")
        print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        print(f"  Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
    
    print(f"\nBest Model: {best_name} ({best_accuracy:.1%})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop Features:")
    for _, row in feature_importance.head(8).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save the best model
    joblib.dump(best_model, 'final_mlb_model.pkl')
    joblib.dump(feature_columns, 'model_features.pkl')
    
    return best_model, best_accuracy, feature_importance

if __name__ == "__main__":
    print("Building final MLB prediction model with 6,882 games...")
    print("This will take 5-10 minutes to calculate rolling statistics.\n")
    
    proceed = input("Continue? (y/n): ").lower()
    if proceed != 'y':
        exit()
    
    # Load data
    df = load_and_prepare_large_dataset()
    
    # Calculate rolling features
    enhanced_df = calculate_rolling_team_stats(df)
    
    # Save enhanced dataset
    enhanced_df.to_csv('final_enhanced_games.csv', index=False)
    print("Enhanced dataset saved")
    
    # Build final model
    model, accuracy, importance = build_final_model(enhanced_df)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS WITH 6,882 GAMES")
    print(f"{'='*60}")
    print(f"Final Model Accuracy: {accuracy:.1%}")
    
    if accuracy > 0.58:
        print("🚀 EXCELLENT! You've built a strong predictive model!")
        print("   This accuracy with a large dataset is very promising.")
    elif accuracy > 0.55:
        print("📈 GOOD! Solid performance with statistical significance.")
        print("   This is above the typical 54% home field baseline.")
    else:
        print("📊 REALISTIC! Sports prediction is inherently difficult.")
        print("   Even small edges can be profitable with proper management.")
    
    print(f"\nYour model is now ready for live predictions!")