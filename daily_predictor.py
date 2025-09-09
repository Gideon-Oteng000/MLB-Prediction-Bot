import requests
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_today_games():
    """Get today's MLB games"""
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        games_today = []
        if data['dates'] and len(data['dates']) > 0:
            games = data['dates'][0]['games']
            
            for game in games:
                if game['status']['statusCode'] in ['S', 'P', 'I']:  # Scheduled, Pre-game, In progress
                    game_info = {
                        'game_id': game['gamePk'],
                        'away_team': game['teams']['away']['team']['name'],
                        'home_team': game['teams']['home']['team']['name'],
                        'game_time': game['gameDate'],
                        'status': game['status']['detailedState']
                    }
                    games_today.append(game_info)
        
        return games_today
    except Exception as e:
        print(f"Error getting today's games: {e}")
        return []

def train_simple_effective_model():
    """Train our best performing simple model"""
    try:
        df = pd.read_csv('mlb_games.csv')
    except:
        print("Error: mlb_games.csv not found. Run collect_games.py first.")
        return None
    
    # Use the simple model that worked best (58.3%)
    df['home_field'] = 1
    df['home_wins'] = (df['winner'] == 'home').astype(int)
    
    # Calculate basic team performance
    team_stats = {}
    
    for team in pd.concat([df['home_team'], df['away_team']]).unique():
        home_games = df[df['home_team'] == team]
        away_games = df[df['away_team'] == team]
        
        home_wins = len(home_games[home_games['winner'] == 'home'])
        away_wins = len(away_games[away_games['winner'] == 'away'])
        total_games = len(home_games) + len(away_games)
        
        win_rate = (home_wins + away_wins) / total_games if total_games > 0 else 0.5
        
        team_stats[team] = {
            'win_rate': win_rate,
            'total_games': total_games,
            'home_win_rate': home_wins / len(home_games) if len(home_games) > 0 else 0.5,
            'away_win_rate': away_wins / len(away_games) if len(away_games) > 0 else 0.5
        }
    
    # Enhance games with team stats
    enhanced_games = []
    for _, game in df.iterrows():
        home_stats = team_stats[game['home_team']]
        away_stats = team_stats[game['away_team']]
        
        enhanced_game = {
            'home_field': 1,
            'home_win_rate': home_stats['win_rate'],
            'away_win_rate': away_stats['win_rate'],
            'win_rate_diff': home_stats['win_rate'] - away_stats['win_rate'],
            'home_home_rate': home_stats['home_win_rate'],
            'away_away_rate': away_stats['away_win_rate'],
            'home_wins': game['home_wins']
        }
        enhanced_games.append(enhanced_game)
    
    enhanced_df = pd.DataFrame(enhanced_games)
    
    # Train model
    feature_columns = ['home_field', 'win_rate_diff', 'home_win_rate', 'away_win_rate']
    X = enhanced_df[feature_columns]
    y = enhanced_df['home_wins']
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Save model and team stats
    joblib.dump(model, 'mlb_model.pkl')
    joblib.dump(team_stats, 'team_stats.pkl')
    
    # Quick accuracy check
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    print(f"Model trained with {accuracy:.1%} accuracy on training data")
    
    return model, team_stats

def predict_today_games():
    """Make predictions for today's games"""
    # Load model and stats
    try:
        model = joblib.load('mlb_model.pkl')
        team_stats = joblib.load('team_stats.pkl')
    except:
        print("Training new model...")
        model, team_stats = train_simple_effective_model()
        if model is None:
            return
    
    # Get today's games
    today_games = get_today_games()
    
    if not today_games:
        print("No games scheduled for today.")
        return
    
    print(f"\n=== MLB PREDICTIONS FOR {datetime.now().strftime('%Y-%m-%d')} ===")
    print(f"Found {len(today_games)} games\n")
    
    predictions = []
    
    for game in today_games:
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Get team stats (use defaults if team not in our data)
        home_stats = team_stats.get(home_team, {'win_rate': 0.5, 'home_win_rate': 0.54})
        away_stats = team_stats.get(away_team, {'win_rate': 0.5, 'away_win_rate': 0.46})
        
        # Create features
        features = [[
            1,  # home_field
            home_stats['win_rate'] - away_stats['win_rate'],  # win_rate_diff
            home_stats['win_rate'],  # home_win_rate
            away_stats['win_rate']   # away_win_rate
        ]]
        
        # Make prediction
        home_win_prob = model.predict_proba(features)[0][1]
        predicted_winner = "HOME" if home_win_prob > 0.5 else "AWAY"
        confidence = max(home_win_prob, 1-home_win_prob)
        
        prediction = {
            'away_team': away_team,
            'home_team': home_team,
            'home_win_prob': home_win_prob,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'game_time': game['game_time']
        }
        predictions.append(prediction)
        
        # Display prediction
        print(f"{away_team} @ {home_team}")
        print(f"  Predicted Winner: {predicted_winner}")
        print(f"  Home Win Probability: {home_win_prob:.1%}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Game Time: {game['game_time']}")
        print()
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    filename = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    pred_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")
    
    return predictions

def update_data_and_retrain():
    """Collect new games and retrain model"""
    print("Updating data with recent games...")
    
    # This would run your collect_games.py logic
    # For now, just retrain with existing data
    model, team_stats = train_simple_effective_model()
    print("Model updated!")

if __name__ == "__main__":
    print("MLB Daily Predictor")
    print("==================")
    
    choice = input("What would you like to do?\n1. Predict today's games\n2. Update data and retrain\n3. Both\nChoice (1-3): ")
    
    if choice in ['2', '3']:
        update_data_and_retrain()
        
    if choice in ['1', '3']:
        predict_today_games()