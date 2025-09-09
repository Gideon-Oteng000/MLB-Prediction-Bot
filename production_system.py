import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta
import time
import os

class MLBBettingSystem:
    def __init__(self):
        self.model = None
        self.historical_data = None
        self.min_games_for_prediction = 10
        self.window_size = 800
        
    def get_recent_completed_games(self, days_back=3):
        """Get completed games from the last N days"""
        print(f"Collecting completed games from last {days_back} days...")
        
        all_recent_games = []
        
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d')
            print(f"  Checking {date}...")
            
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
            
            try:
                response = requests.get(url)
                data = response.json()
                
                if data['dates'] and len(data['dates']) > 0:
                    games = data['dates'][0]['games']
                    
                    for game in games:
                        if game['status']['statusCode'] == 'F':  # Final games only
                            game_info = {
                                'season': datetime.now().year,
                                'date': date,
                                'game_id': game['gamePk'],
                                'away_team': game['teams']['away']['team']['name'],
                                'home_team': game['teams']['home']['team']['name'],
                                'away_score': game['teams']['away']['score'],
                                'home_score': game['teams']['home']['score'],
                                'winner': 'home' if game['teams']['home']['score'] > game['teams']['away']['score'] else 'away'
                            }
                            all_recent_games.append(game_info)
                
                time.sleep(0.3)  # Be nice to the API
                
            except Exception as e:
                print(f"    Error getting games for {date}: {e}")
        
        print(f"Found {len(all_recent_games)} completed games to check")
        return all_recent_games
    
    def update_historical_data(self):
        """Update historical data with recent completed games"""
        print("\n=== UPDATING HISTORICAL DATABASE ===")
        
        # Load existing historical data
        try:
            historical_df = pd.read_csv('historical_mlb_games.csv')
            historical_df['date'] = pd.to_datetime(historical_df['date'])
            print(f"Loaded {len(historical_df)} existing historical games")
            
            last_date = historical_df['date'].max()
            print(f"Last game in database: {last_date.date()}")
            
        except FileNotFoundError:
            print("No existing historical file found")
            return False
        
        # Get recent games (adaptive based on staleness)
        days_back = getattr(self, 'days_to_check', 3)
        recent_games = self.get_recent_completed_games(days_back=days_back)
        
        if not recent_games:
            print("No recent games to add")
            self.historical_data = historical_df
            return True
        
        recent_df = pd.DataFrame(recent_games)
        recent_df['date'] = pd.to_datetime(recent_df['date'])
        
        # Find truly new games (not already in database)
        new_games = []
        for _, game in recent_df.iterrows():
            existing_game = historical_df[
                (historical_df['game_id'] == game['game_id']) |
                ((historical_df['date'] == game['date']) & 
                 (historical_df['home_team'] == game['home_team']) & 
                 (historical_df['away_team'] == game['away_team']))
            ]
            
            if len(existing_game) == 0:
                new_games.append(game.to_dict())
        
        if new_games:
            print(f"📈 Adding {len(new_games)} NEW games to database")
            
            new_games_df = pd.DataFrame(new_games)
            updated_df = pd.concat([historical_df, new_games_df], ignore_index=True)
            updated_df = updated_df.sort_values('date').drop_duplicates(subset=['game_id'], keep='last')
            
            # Save updated data
            updated_df.to_csv('historical_mlb_games.csv', index=False)
            self.historical_data = updated_df
            print(f"Updated database now contains {len(updated_df)} games")
            
            # Show what was added
            print("Recently completed games added:")
            for game in new_games[-5:]:  # Show last 5 added
                winner_indicator = "🏠" if game['winner'] == 'home' else "✈️"
                print(f"  {winner_indicator} {game['date']}: {game['away_team']} @ {game['home_team']} ({game['away_score']}-{game['home_score']})")
            
            # Force model retrain since we have new data
            print("🔄 New games detected - will retrain model with fresh data")
            self.force_model_retrain()
            
            return True
        else:
            print("✅ Database is already up to date")
            self.historical_data = historical_df
            return True
    
    def force_model_retrain(self):
        """Force the production system to retrain with new data"""
        model_files = ['production_mlb_model.pkl']
        
        for model_file in model_files:
            if os.path.exists(model_file):
                os.remove(model_file)
                print(f"Removed {model_file} - will retrain with new data")
        
        self.model = None  # Clear loaded model
    
    def load_historical_data(self):
        """Load historical games data"""
        try:
            self.historical_data = pd.read_csv('historical_mlb_games.csv')
            self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
            self.historical_data = self.historical_data.sort_values('date').reset_index(drop=True)
            print(f"Loaded {len(self.historical_data)} historical games")
            return True
        except:
            print("Error: Could not load historical_mlb_games.csv")
            return False
    
    def check_data_freshness(self):
        """Check if historical data needs updating"""
        if self.historical_data is None:
            return False
        
        last_date = self.historical_data['date'].max()
        days_since_update = (datetime.now() - last_date).days
        hours_since_update = (datetime.now() - last_date).total_seconds() / 3600
        
        print(f"Last game in database: {last_date.date()} ({days_since_update} days, {hours_since_update:.1f} hours ago)")
        
        # Determine how many days back to check based on staleness
        if days_since_update >= 7:
            self.days_to_check = 10  # Haven't run in a week - check 10 days
            print(f"⚠️  Data is {days_since_update} days old - will check last 10 days")
            return False
        elif days_since_update >= 3:
            self.days_to_check = 7   # Haven't run in 3+ days - check week
            print(f"⚠️  Data is {days_since_update} days old - will check last 7 days")
            return False
        elif hours_since_update > 18:
            self.days_to_check = 3   # Daily updates - check last 3 days
            print(f"⚠️  Data is {hours_since_update:.1f} hours old - will check last 3 days")
            return False
        else:
            print("✅ Historical data is current")
            return True
    
    def get_todays_games(self):
        """Get today's scheduled MLB games"""
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
            
            print(f"Found {len(games_today)} games scheduled for {today}")
            return games_today
            
        except Exception as e:
            print(f"Error getting today's games: {e}")
            return []
    
    def calculate_team_features(self, team, as_of_date):
        """Calculate team features using only data before as_of_date"""
        # Get team's recent games (last 20)
        team_games = self.historical_data[
            ((self.historical_data['home_team'] == team) | 
             (self.historical_data['away_team'] == team)) &
            (self.historical_data['date'] < as_of_date)
        ].tail(20)
        
        if len(team_games) < self.min_games_for_prediction:
            # Return league average stats
            return {
                'win_rate': 0.5,
                'runs_per_game': 4.5,
                'runs_allowed_per_game': 4.5,
                'run_differential': 0.0,
                'games_played': 0
            }
        
        # Calculate win rate
        wins = len(team_games[
            ((team_games['home_team'] == team) & (team_games['winner'] == 'home')) |
            ((team_games['away_team'] == team) & (team_games['winner'] == 'away'))
        ])
        win_rate = wins / len(team_games)
        
        # Calculate run statistics
        runs_scored = 0
        runs_allowed = 0
        
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                runs_scored += game['home_score']
                runs_allowed += game['away_score']
            else:
                runs_scored += game['away_score']
                runs_allowed += game['home_score']
        
        runs_per_game = runs_scored / len(team_games)
        runs_allowed_per_game = runs_allowed / len(team_games)
        run_differential = runs_per_game - runs_allowed_per_game
        
        return {
            'win_rate': win_rate,
            'runs_per_game': runs_per_game,
            'runs_allowed_per_game': runs_allowed_per_game,
            'run_differential': run_differential,
            'games_played': len(team_games)
        }
    
    def train_model(self):
        """Train model on historical data"""
        print("\n=== TRAINING MODEL ===")
        print("Training model on updated historical data...")
        
        # Use last 1500 games for training (more data now)
        recent_data = self.historical_data.tail(1500)
        
        features = []
        labels = []
        
        for idx, game in recent_data.iterrows():
            # Skip early games where we don't have enough history
            if idx < 500:
                continue
                
            game_date = game['date']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Calculate features using only data before this game
            home_stats = self.calculate_team_features(home_team, game_date)
            away_stats = self.calculate_team_features(away_team, game_date)
            
            # Skip if insufficient data
            if home_stats['games_played'] < 5 or away_stats['games_played'] < 5:
                continue
            
            # Create feature vector
            feature_vector = [
                1,  # home_field
                home_stats['win_rate'] - away_stats['win_rate'],  # win_rate_advantage
                home_stats['win_rate'],
                away_stats['win_rate'],
                home_stats['run_differential'] - away_stats['run_differential'],  # run_diff_advantage
                home_stats['run_differential'],
                away_stats['run_differential']
            ]
            
            features.append(feature_vector)
            labels.append(1 if game['winner'] == 'home' else 0)
        
        print(f"Training on {len(features)} games with updated data")
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        self.model.fit(features, labels)
        
        # Quick accuracy check on training data
        predictions = self.model.predict(features)
        accuracy = (predictions == labels).mean()
        print(f"Training accuracy: {accuracy:.1%}")
        
        # Save model
        joblib.dump(self.model, 'production_mlb_model.pkl')
        print("✅ Model saved to production_mlb_model.pkl")
        
        return True
    
    def predict_game(self, home_team, away_team):
        """Predict outcome of a single game"""
        if self.model is None:
            try:
                self.model = joblib.load('production_mlb_model.pkl')
                print("📂 Loaded existing model")
            except:
                print("🔄 No saved model found - training new model...")
                self.train_model()
        
        # Use current date for feature calculation
        current_date = datetime.now()
        
        # Get team statistics
        home_stats = self.calculate_team_features(home_team, current_date)
        away_stats = self.calculate_team_features(away_team, current_date)
        
        # Check if we have sufficient data
        if home_stats['games_played'] < 5 or away_stats['games_played'] < 5:
            return {
                'home_win_probability': 0.54,  # Default home field advantage
                'prediction': 'HOME',
                'confidence': 'LOW',
                'reason': 'Insufficient recent data for one or both teams'
            }
        
        # Create feature vector
        features = [[
            1,  # home_field
            home_stats['win_rate'] - away_stats['win_rate'],
            home_stats['win_rate'],
            away_stats['win_rate'],
            home_stats['run_differential'] - away_stats['run_differential'],
            home_stats['run_differential'],
            away_stats['run_differential']
        ]]
        
        # Make prediction
        home_win_prob = self.model.predict_proba(features)[0][1]
        prediction = 'HOME' if home_win_prob > 0.5 else 'AWAY'
        
        # Determine confidence level
        if home_win_prob > 0.6 or home_win_prob < 0.4:
            confidence = 'HIGH'
        elif home_win_prob > 0.55 or home_win_prob < 0.45:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'home_win_probability': home_win_prob,
            'prediction': prediction,
            'confidence': confidence,
            'home_stats': home_stats,
            'away_stats': away_stats
        }
    
    def generate_daily_predictions(self):
        """Generate predictions for today's games"""
        print(f"\n{'='*60}")
        print(f"MLB BETTING PREDICTIONS - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Get today's games
        todays_games = self.get_todays_games()
        
        if not todays_games:
            print("No games scheduled for today.")
            return []
        
        predictions = []
        
        for game in todays_games:
            home_team = game['home_team']
            away_team = game['away_team']
            
            prediction = self.predict_game(home_team, away_team)
            
            # Display prediction
            print(f"\n{away_team} @ {home_team}")
            print(f"  Prediction: {prediction['prediction']}")
            print(f"  Home Win Probability: {prediction['home_win_probability']:.1%}")
            print(f"  Confidence: {prediction['confidence']}")
            
            # Show team stats
            if 'home_stats' in prediction:
                print(f"  Home team recent: {prediction['home_stats']['win_rate']:.1%} win rate, {prediction['home_stats']['run_differential']:.1f} run diff")
                print(f"  Away team recent: {prediction['away_stats']['win_rate']:.1%} win rate, {prediction['away_stats']['run_differential']:.1f} run diff")
            
            if prediction['confidence'] in ['HIGH', 'MEDIUM']:
                print(f"  📊 BETTING RECOMMENDATION: Consider this game")
            else:
                print(f"  ⚠️  Skip - Low confidence prediction")
            
            # Add to predictions list
            prediction_record = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'away_team': away_team,
                'home_team': home_team,
                'predicted_winner': prediction['prediction'],
                'home_win_probability': prediction['home_win_probability'],
                'confidence': prediction['confidence'],
                'game_time': game['game_time']
            }
            predictions.append(prediction_record)
        
        # Save predictions
        predictions_df = pd.DataFrame(predictions)
        filename = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        predictions_df.to_csv(filename, index=False)
        
        print(f"\n🎯 SUMMARY:")
        high_conf = len([p for p in predictions if p['confidence'] == 'HIGH'])
        medium_conf = len([p for p in predictions if p['confidence'] == 'MEDIUM'])
        
        print(f"  Total games: {len(predictions)}")
        print(f"  High confidence: {high_conf}")
        print(f"  Medium confidence: {medium_conf}")
        print(f"  Recommended bets: {high_conf + medium_conf}")
        print(f"  Predictions saved to: {filename}")
        
        return predictions

def main():
    print("MLB Production Betting System with Auto-Updates")
    print("Based on 52.6% accuracy model with automatic data updates")
    print("="*70)
    
    system = MLBBettingSystem()
    
    # Step 1: Load historical data
    print("\n=== STEP 1: LOADING HISTORICAL DATA ===")
    if not system.load_historical_data():
        print("❌ Failed to load historical data")
        return
    
    # Step 2: Check data freshness and update if needed
    print("\n=== STEP 2: CHECKING DATA FRESHNESS ===")
    if not system.check_data_freshness():
        print("🔄 Updating data with recent games...")
        if not system.update_historical_data():
            print("❌ Failed to update data")
            return
    else:
        # Even if data is fresh, still load it into the system
        system.historical_data = pd.read_csv('historical_mlb_games.csv')
        system.historical_data['date'] = pd.to_datetime(system.historical_data['date'])
        system.historical_data = system.historical_data.sort_values('date').reset_index(drop=True)
    
    # Step 3: Generate predictions
    print("\n=== STEP 3: GENERATING PREDICTIONS ===")
    predictions = system.generate_daily_predictions()
    
    if predictions:
        print(f"\n✅ SUCCESS! Generated {len(predictions)} predictions")
        print("Your model is using the most current data available.")
    else:
        print("\n⚠️  No predictions generated (no games scheduled)")

if __name__ == "__main__":
    main()