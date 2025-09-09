import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time
from datetime import datetime

def get_team_stats(team_id, season=2024):
    """Get team statistics from MLB API"""
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&season={season}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Extract hitting and pitching stats
        stats = {}
        for stat_group in data['stats']:
            if stat_group['group']['displayName'] == 'hitting':
                hitting = stat_group['stats']
                stats['batting_avg'] = float(hitting.get('avg', 0))
                stats['on_base_pct'] = float(hitting.get('obp', 0))
                stats['slugging'] = float(hitting.get('slg', 0))
                stats['runs_per_game'] = float(hitting.get('runs', 0)) / float(hitting.get('gamesPlayed', 1))
                
            elif stat_group['group']['displayName'] == 'pitching':
                pitching = stat_group['stats']
                stats['era'] = float(pitching.get('era', 5.00))
                stats['whip'] = float(pitching.get('whip', 1.50))
                stats['runs_allowed_per_game'] = float(pitching.get('earnedRuns', 0)) / float(pitching.get('gamesPlayed', 1))
        
        return stats
    except:
        # Return default stats if API fails
        return {
            'batting_avg': 0.250,
            'on_base_pct': 0.320,
            'slugging': 0.400,
            'runs_per_game': 4.5,
            'era': 4.50,
            'whip': 1.35,
            'runs_allowed_per_game': 4.5
        }

def get_team_id_mapping():
    """Create mapping of team names to MLB API team IDs"""
    # This is a simplified mapping - in a real system you'd get this from the API
    team_mapping = {
        'Arizona Diamondbacks': 109, 'Atlanta Braves': 144, 'Baltimore Orioles': 110,
        'Boston Red Sox': 111, 'Chicago Cubs': 112, 'Chicago White Sox': 145,
        'Cincinnati Reds': 113, 'Cleveland Guardians': 114, 'Colorado Rockies': 115,
        'Detroit Tigers': 116, 'Houston Astros': 117, 'Kansas City Royals': 118,
        'Los Angeles Angels': 108, 'Los Angeles Dodgers': 119, 'Miami Marlins': 146,
        'Milwaukee Brewers': 158, 'Minnesota Twins': 142, 'New York Mets': 121,
        'New York Yankees': 147, 'Oakland Athletics': 133, 'Philadelphia Phillies': 143,
        'Pittsburgh Pirates': 134, 'San Diego Padres': 135, 'San Francisco Giants': 137,
        'Seattle Mariners': 136, 'St. Louis Cardinals': 138, 'Tampa Bay Rays': 139,
        'Texas Rangers': 140, 'Toronto Blue Jays': 141, 'Washington Nationals': 120
    }
    return team_mapping

def enhance_games_data():
    """Add team statistics to each game"""
    df = pd.read_csv('mlb_games.csv')
    team_mapping = get_team_id_mapping()
    
    print("Collecting team statistics...")
    team_stats_cache = {}
    
    enhanced_games = []
    
    for idx, game in df.iterrows():
        print(f"Processing game {idx+1}/{len(df)}: {game['away_team']} @ {game['home_team']}")
        
        # Get team IDs
        away_team_id = team_mapping.get(game['away_team'])
        home_team_id = team_mapping.get(game['home_team'])
        
        if away_team_id and home_team_id:
            # Get away team stats (with caching)
            if away_team_id not in team_stats_cache:
                team_stats_cache[away_team_id] = get_team_stats(away_team_id)
                time.sleep(0.3)  # Be nice to the API
            
            # Get home team stats (with caching)
            if home_team_id not in team_stats_cache:
                team_stats_cache[home_team_id] = get_team_stats(home_team_id)
                time.sleep(0.3)
            
            away_stats = team_stats_cache[away_team_id]
            home_stats = team_stats_cache[home_team_id]
            
            # Create enhanced game record
            enhanced_game = {
                'date': game['date'],
                'away_team': game['away_team'],
                'home_team': game['home_team'],
                'away_score': game['away_score'],
                'home_score': game['home_score'],
                'home_wins': 1 if game['winner'] == 'home' else 0,
                
                # Away team features
                'away_batting_avg': away_stats['batting_avg'],
                'away_on_base_pct': away_stats['on_base_pct'],
                'away_slugging': away_stats['slugging'],
                'away_runs_per_game': away_stats['runs_per_game'],
                'away_era': away_stats['era'],
                'away_whip': away_stats['whip'],
                
                # Home team features  
                'home_batting_avg': home_stats['batting_avg'],
                'home_on_base_pct': home_stats['on_base_pct'],
                'home_slugging': home_stats['slugging'],
                'home_runs_per_game': home_stats['runs_per_game'],
                'home_era': home_stats['era'],
                'home_whip': home_stats['whip'],
                
                # Comparative features
                'batting_avg_diff': home_stats['batting_avg'] - away_stats['batting_avg'],
                'era_diff': away_stats['era'] - home_stats['era'],  # Lower ERA is better
                'runs_per_game_diff': home_stats['runs_per_game'] - away_stats['runs_per_game'],
            }
            
            enhanced_games.append(enhanced_game)
    
    enhanced_df = pd.DataFrame(enhanced_games)
    enhanced_df.to_csv('enhanced_games.csv', index=False)
    print(f"\nSaved {len(enhanced_games)} enhanced games to enhanced_games.csv")
    
    return enhanced_df

def build_advanced_model(df):
    """Build a model with real baseball features"""
    
    # Select features
    feature_columns = [
        'batting_avg_diff', 'era_diff', 'runs_per_game_diff',
        'home_batting_avg', 'home_era', 'home_runs_per_game',
        'away_batting_avg', 'away_era', 'away_runs_per_game'
    ]
    
    X = df[feature_columns]
    y = df['home_wins']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n=== ADVANCED MODEL ===")
    print(f"Using {len(feature_columns)} baseball features")
    print(f"Training on {len(X_train)} games, testing on {len(X_test)} games")
    
    # Try Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"Random Forest Accuracy: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nMost Important Features:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return rf_model, rf_accuracy

if __name__ == "__main__":
    print("Building advanced MLB prediction model with real baseball stats...\n")
    
    # Enhance the data with team statistics
    df = enhance_games_data()
    
    # Build advanced model
    model, accuracy = build_advanced_model(df)
    
    print(f"\n=== RESULTS COMPARISON ===")
    print(f"Simple model (home field only): ~58.3%")
    print(f"Advanced model (with baseball stats): {accuracy*100:.1f}%")
    
    if accuracy > 0.583:
        print("🚀 Improvement! Baseball stats are helping!")
    else:
        print("📊 Similar performance - might need more data or features")