import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

def calculate_recent_form(df, team, date, days_back=10):
    """Calculate team's record in last N days before given date"""
    team_games = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) & 
        (pd.to_datetime(df['date']) < pd.to_datetime(date))
    ].copy()
    
    # Get last N days
    cutoff_date = pd.to_datetime(date) - timedelta(days=days_back)
    recent_games = team_games[pd.to_datetime(team_games['date']) >= cutoff_date]
    
    if len(recent_games) == 0:
        return 0.5, 0  # Neutral record if no recent games
    
    # Calculate wins
    wins = 0
    for _, game in recent_games.iterrows():
        if (game['home_team'] == team and game['winner'] == 'home') or \
           (game['away_team'] == team and game['winner'] == 'away'):
            wins += 1
    
    win_rate = wins / len(recent_games)
    games_played = len(recent_games)
    
    return win_rate, games_played

def calculate_head_to_head(df, home_team, away_team, date, games_back=20):
    """Calculate head-to-head record between teams"""
    h2h_games = df[
        (((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
         ((df['home_team'] == away_team) & (df['away_team'] == home_team))) &
        (pd.to_datetime(df['date']) < pd.to_datetime(date))
    ].tail(games_back)
    
    if len(h2h_games) == 0:
        return 0.5  # Neutral if no history
    
    home_wins = 0
    for _, game in h2h_games.iterrows():
        if game['home_team'] == home_team and game['winner'] == 'home':
            home_wins += 1
        elif game['away_team'] == home_team and game['winner'] == 'away':
            home_wins += 1
    
    return home_wins / len(h2h_games)

def calculate_run_differential(df, team, date, days_back=15):
    """Calculate team's run differential in recent games"""
    team_games = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) & 
        (pd.to_datetime(df['date']) < pd.to_datetime(date))
    ].copy()
    
    cutoff_date = pd.to_datetime(date) - timedelta(days=days_back)
    recent_games = team_games[pd.to_datetime(team_games['date']) >= cutoff_date]
    
    if len(recent_games) == 0:
        return 0
    
    total_runs_scored = 0
    total_runs_allowed = 0
    
    for _, game in recent_games.iterrows():
        if game['home_team'] == team:
            total_runs_scored += game['home_score']
            total_runs_allowed += game['away_score']
        else:
            total_runs_scored += game['away_score']
            total_runs_allowed += game['home_score']
    
    return (total_runs_scored - total_runs_allowed) / len(recent_games)

def enhance_with_recent_form():
    """Add recent performance features to games data"""
    df = pd.read_csv('mlb_games.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print("Calculating recent form for each game...")
    enhanced_games = []
    
    for idx, game in df.iterrows():
        print(f"Processing game {idx+1}/{len(df)}: {game['away_team']} @ {game['home_team']}")
        
        # Recent form (last 10 days)
        home_form, home_games = calculate_recent_form(df, game['home_team'], game['date'], 10)
        away_form, away_games = calculate_recent_form(df, game['away_team'], game['date'], 10)
        
        # Head-to-head
        h2h_home_advantage = calculate_head_to_head(df, game['home_team'], game['away_team'], game['date'])
        
        # Run differential
        home_run_diff = calculate_run_differential(df, game['home_team'], game['date'])
        away_run_diff = calculate_run_differential(df, game['away_team'], game['date'])
        
        enhanced_game = {
            'date': game['date'],
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'home_wins': 1 if game['winner'] == 'home' else 0,
            
            # Recent form features
            'home_recent_form': home_form,
            'away_recent_form': away_form,
            'form_advantage': home_form - away_form,
            'home_recent_games': home_games,
            'away_recent_games': away_games,
            
            # Head-to-head
            'h2h_home_advantage': h2h_home_advantage,
            
            # Run differential
            'home_run_diff': home_run_diff,
            'away_run_diff': away_run_diff,
            'run_diff_advantage': home_run_diff - away_run_diff,
            
            # Basic home field
            'home_field': 1
        }
        
        enhanced_games.append(enhanced_game)
    
    enhanced_df = pd.DataFrame(enhanced_games)
    enhanced_df.to_csv('recent_form_games.csv', index=False)
    print(f"\nSaved enhanced data with recent form features")
    
    return enhanced_df

def build_recent_form_model(df):
    """Build model with recent performance features"""
    
    # Only use games where we have some recent history
    df_filtered = df[(df['home_recent_games'] >= 3) & (df['away_recent_games'] >= 3)].copy()
    
    feature_columns = [
        'home_field',
        'form_advantage',
        'home_recent_form',
        'away_recent_form', 
        'h2h_home_advantage',
        'run_diff_advantage',
        'home_run_diff',
        'away_run_diff'
    ]
    
    X = df_filtered[feature_columns]
    y = df_filtered['home_wins']
    
    if len(X) < 20:
        print("Not enough games with recent history. Need more data.")
        return None, 0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n=== RECENT FORM MODEL ===")
    print(f"Using {len(feature_columns)} features including recent performance")
    print(f"Training on {len(X_train)} games, testing on {len(X_test)} games")
    
    # Build model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Recent Form Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nMost Important Features:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return model, accuracy

if __name__ == "__main__":
    print("Building MLB model with recent team performance...\n")
    
    # Enhance data with recent form
    df = enhance_with_recent_form()
    
    # Build model
    model, accuracy = build_recent_form_model(df)
    
    if model:
        print(f"\n=== ACCURACY PROGRESSION ===")
        print(f"Simple home field model: 58.3%")
        print(f"Advanced baseball stats: 59.0%")
        print(f"Recent form model: {accuracy*100:.1f}%")
        
        if accuracy > 0.60:
            print("🎯 Breaking 60%! This is getting good!")
        elif accuracy > 0.59:
            print("📈 Solid improvement! Recent form matters.")
        else:
            print("🤔 May need more data or different features.")