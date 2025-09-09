import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from datetime import datetime

def get_pitcher_stats(pitcher_id, season=2024):
    """Get detailed pitcher statistics"""
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=season&season={season}&group=pitching"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'stats' in data and len(data['stats']) > 0:
            stats = data['stats'][0]['stats']
            return {
                'era': float(stats.get('era', 5.00)),
                'whip': float(stats.get('whip', 1.50)),
                'strikeouts_per_9': float(stats.get('strikeoutsPer9Inn', 6.0)),
                'walks_per_9': float(stats.get('baseOnBallsPer9Inn', 3.5)),
                'hits_per_9': float(stats.get('hitsPer9Inn', 9.0)),
                'wins': int(stats.get('wins', 5)),
                'losses': int(stats.get('losses', 5)),
                'games_started': int(stats.get('gamesStarted', 10)),
                'innings_pitched': float(stats.get('inningsPitched', 50.0)),
                'opponent_avg': float(stats.get('avg', 0.270))
            }
        else:
            return get_default_pitcher_stats()
    except:
        return get_default_pitcher_stats()

def get_default_pitcher_stats():
    """Return average pitcher stats when API fails"""
    return {
        'era': 4.50,
        'whip': 1.35,
        'strikeouts_per_9': 8.0,
        'walks_per_9': 3.2,
        'hits_per_9': 8.8,
        'wins': 8,
        'losses': 8,
        'games_started': 20,
        'innings_pitched': 100.0,
        'opponent_avg': 0.260
    }

def get_game_pitchers(game_id):
    """Get starting pitchers for a specific game"""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Try to get probable pitchers
        if 'gameData' in data and 'probablePitchers' in data['gameData']:
            prob_pitchers = data['gameData']['probablePitchers']
            
            away_pitcher_id = None
            home_pitcher_id = None
            
            if 'away' in prob_pitchers and prob_pitchers['away']:
                away_pitcher_id = prob_pitchers['away']['id']
                
            if 'home' in prob_pitchers and prob_pitchers['home']:
                home_pitcher_id = prob_pitchers['home']['id']
                
            return away_pitcher_id, home_pitcher_id
        
        return None, None
    except:
        return None, None

def enhance_games_with_pitchers():
    """Add pitcher data to games"""
    df = pd.read_csv('mlb_games.csv')
    
    print("Collecting pitcher data for each game...")
    enhanced_games = []
    pitcher_cache = {}
    
    for idx, game in df.iterrows():
        print(f"Processing game {idx+1}/{len(df)}: {game['away_team']} @ {game['home_team']}")
        
        # Get pitcher IDs for this game
        away_pitcher_id, home_pitcher_id = get_game_pitchers(game['game_id'])
        
        if away_pitcher_id and home_pitcher_id:
            # Get away pitcher stats (with caching)
            if away_pitcher_id not in pitcher_cache:
                pitcher_cache[away_pitcher_id] = get_pitcher_stats(away_pitcher_id)
                time.sleep(0.3)
            
            # Get home pitcher stats (with caching)  
            if home_pitcher_id not in pitcher_cache:
                pitcher_cache[home_pitcher_id] = get_pitcher_stats(home_pitcher_id)
                time.sleep(0.3)
            
            away_pitcher = pitcher_cache[away_pitcher_id]
            home_pitcher = pitcher_cache[home_pitcher_id]
            
        else:
            # Use default stats if we can't get pitcher info
            away_pitcher = get_default_pitcher_stats()
            home_pitcher = get_default_pitcher_stats()
        
        # Calculate pitcher advantages
        era_advantage = away_pitcher['era'] - home_pitcher['era']  # Positive = home pitcher better
        whip_advantage = away_pitcher['whip'] - home_pitcher['whip']
        k9_advantage = home_pitcher['strikeouts_per_9'] - away_pitcher['strikeouts_per_9']
        
        # Win percentage
        home_pitcher_win_pct = home_pitcher['wins'] / max(home_pitcher['wins'] + home_pitcher['losses'], 1)
        away_pitcher_win_pct = away_pitcher['wins'] / max(away_pitcher['wins'] + away_pitcher['losses'], 1)
        
        enhanced_game = {
            'date': game['date'],
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'home_wins': 1 if game['winner'] == 'home' else 0,
            
            # Raw pitcher stats
            'away_pitcher_era': away_pitcher['era'],
            'home_pitcher_era': home_pitcher['era'],
            'away_pitcher_whip': away_pitcher['whip'],
            'home_pitcher_whip': home_pitcher['whip'],
            'away_pitcher_k9': away_pitcher['strikeouts_per_9'],
            'home_pitcher_k9': home_pitcher['strikeouts_per_9'],
            
            # Pitcher advantages (positive = home advantage)
            'era_advantage': era_advantage,
            'whip_advantage': whip_advantage,
            'strikeout_advantage': k9_advantage,
            'pitcher_win_pct_diff': home_pitcher_win_pct - away_pitcher_win_pct,
            
            # Combined pitcher quality score
            'home_pitcher_quality': (5.00 - home_pitcher['era']) + (1.50 - home_pitcher['whip']) + (home_pitcher['strikeouts_per_9'] / 10),
            'away_pitcher_quality': (5.00 - away_pitcher['era']) + (1.50 - away_pitcher['whip']) + (away_pitcher['strikeouts_per_9'] / 10),
            
            # Home field
            'home_field': 1
        }
        
        enhanced_game['pitcher_quality_advantage'] = enhanced_game['home_pitcher_quality'] - enhanced_game['away_pitcher_quality']
        enhanced_games.append(enhanced_game)
    
    enhanced_df = pd.DataFrame(enhanced_games)
    enhanced_df.to_csv('pitcher_enhanced_games.csv', index=False)
    print(f"\nSaved {len(enhanced_games)} games with pitcher data")
    
    return enhanced_df

def build_pitcher_model(df):
    """Build model focused on pitcher matchups"""
    
    feature_columns = [
        'home_field',
        'era_advantage',
        'whip_advantage', 
        'strikeout_advantage',
        'pitcher_quality_advantage',
        'pitcher_win_pct_diff',
        'home_pitcher_era',
        'away_pitcher_era'
    ]
    
    X = df[feature_columns]
    y = df['home_wins']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n=== PITCHER-FOCUSED MODEL ===")
    print(f"Using {len(feature_columns)} pitcher-focused features")
    print(f"Training on {len(X_train)} games, testing on {len(X_test)} games")
    
    # Build model
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Pitcher Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nMost Important Pitcher Features:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Show some insights
    print(f"\n=== PITCHER INSIGHTS ===")
    era_importance = feature_importance[feature_importance['feature'] == 'era_advantage']['importance'].iloc[0]
    print(f"ERA advantage importance: {era_importance:.3f}")
    
    if era_importance > 0.15:
        print("💡 ERA difference between pitchers is a major factor!")
    
    return model, accuracy

if __name__ == "__main__":
    print("Building MLB pitcher-focused prediction model...\n")
    print("⚠️  This will take several minutes to collect pitcher data from MLB API")
    
    proceed = input("Continue? (y/n): ").lower()
    if proceed != 'y':
        print("Cancelled.")
        exit()
        
    
    # Enhance data with pitcher stats
    df = enhance_games_with_pitchers()
    
    # Build pitcher-focused model
    model, accuracy = build_pitcher_model(df)
    
    print(f"\n=== MODEL COMPARISON ===")
    print(f"Home field only: 58.3%")
    print(f"Team stats: 59.0%")
    print(f"Recent form: ~59.0%")
    print(f"Pitcher-focused: {accuracy*100:.1f}%")
    
    if accuracy > 0.62:
        print("🚀 Excellent! Pitchers make a big difference!")
    elif accuracy > 0.60:
        print("📈 Good improvement! Pitcher matchups matter.")
    else:
        print("🤔 Similar to previous models. May need more pitcher context.")