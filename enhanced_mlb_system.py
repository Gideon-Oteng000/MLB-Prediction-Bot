import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta
import time
import os

class EnhancedMLBBettingSystem:
    def __init__(self):
        self.model = None
        self.historical_data = None
        self.pitcher_data = None
        self.weather_api_key = None  # You'll need to set this
        self.min_games_for_prediction = 10
        
    def set_weather_api_key(self, api_key):
        """Set your OpenWeatherMap API key"""
        self.weather_api_key = api_key
    
    def get_ballpark_coordinates(self):
        """MLB ballpark coordinates for weather data"""
        return {
            'Arizona Diamondbacks': (33.4455, -112.0667),
            'Atlanta Braves': (33.8906, -84.4677),
            'Baltimore Orioles': (39.2838, -76.6218),
            'Boston Red Sox': (42.3467, -71.0972),
            'Chicago Cubs': (41.9484, -87.6553),
            'Chicago White Sox': (41.8299, -87.6338),
            'Cincinnati Reds': (39.0974, -84.5062),
            'Cleveland Guardians': (41.4958, -81.6852),
            'Colorado Rockies': (39.7562, -104.9942),
            'Detroit Tigers': (42.3390, -83.0485),
            'Houston Astros': (29.7573, -95.3555),
            'Kansas City Royals': (39.0517, -94.4803),
            'Los Angeles Angels': (33.8003, -117.8827),
            'Los Angeles Dodgers': (34.0739, -118.2400),
            'Miami Marlins': (25.7781, -80.2197),
            'Milwaukee Brewers': (43.0280, -87.9712),
            'Minnesota Twins': (44.9817, -93.2776),
            'New York Mets': (40.7571, -73.8458),
            'New York Yankees': (40.8291, -73.9262),
            'Oakland Athletics': (37.7516, -122.2008),
            'Philadelphia Phillies': (39.9061, -75.1665),
            'Pittsburgh Pirates': (40.4469, -80.0057),
            'San Diego Padres': (32.7073, -117.1566),
            'San Francisco Giants': (37.7786, -122.3893),
            'Seattle Mariners': (47.5914, -122.3325),
            'St. Louis Cardinals': (38.6226, -90.1928),
            'Tampa Bay Rays': (27.7682, -82.6534),
            'Texas Rangers': (32.7511, -97.0825),
            'Toronto Blue Jays': (43.6414, -79.3894),
            'Washington Nationals': (38.8730, -77.0074)
        }
    
    def get_weather_data(self, team, game_time=None):
        """Get weather data for a team's ballpark"""
        if not self.weather_api_key:
            return self.get_default_weather()
        
        coordinates = self.get_ballpark_coordinates()
        if team not in coordinates:
            return self.get_default_weather()
        
        lat, lon = coordinates[team]
        
        try:
            # Get current weather (or forecast if game_time provided)
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=imperial"
            
            response = requests.get(url)
            data = response.json()
            
            # Extract relevant weather data
            weather = {
                'temperature': data.get('main', {}).get('temp', 70),
                'humidity': data.get('main', {}).get('humidity', 50),
                'wind_speed': data.get('wind', {}).get('speed', 5),
                'wind_direction': data.get('wind', {}).get('deg', 180),
                'pressure': data.get('main', {}).get('pressure', 1013),
                'weather_condition': data.get('weather', [{}])[0].get('main', 'Clear')
            }
            
            # Calculate wind factor (positive = helping home runs, negative = hurting)
            # This is simplified - real implementation would need ballpark-specific directions
            wind_factor = self.calculate_wind_factor(weather['wind_speed'], weather['wind_direction'])
            weather['wind_factor'] = wind_factor
            
            return weather
            
        except Exception as e:
            print(f"Weather API error: {e}")
            return self.get_default_weather()
    
    def get_default_weather(self):
        """Default weather when API unavailable"""
        return {
            'temperature': 75,
            'humidity': 50,
            'wind_speed': 5,
            'wind_direction': 180,
            'pressure': 1013,
            'weather_condition': 'Clear',
            'wind_factor': 0
        }
    
    def calculate_wind_factor(self, wind_speed, wind_direction):
        """Calculate wind impact on offense (simplified)"""
        # This is a basic implementation
        # Wind blowing out (90-270 degrees) helps offense
        # Wind blowing in (270-90 degrees) hurts offense
        
        if 90 <= wind_direction <= 270:  # Blowing out
            return wind_speed * 0.1
        else:  # Blowing in
            return wind_speed * -0.1
    
    def get_pitcher_handedness_data(self):
        """Get pitcher handedness and basic stats - this would need to be expanded"""
        # This is a placeholder - you'd need to collect this data from MLB API or other sources
        # For now, return a basic structure
        return {}
    
    def calculate_handedness_advantage(self, team, opposing_pitcher_hand, as_of_date):
        """Calculate team's performance vs LHP/RHP"""
        # Get team's recent games
        team_games = self.historical_data[
            ((self.historical_data['home_team'] == team) | 
             (self.historical_data['away_team'] == team)) &
            (self.historical_data['date'] < as_of_date)
        ].tail(50)  # Look at more games for handedness splits
        
        if len(team_games) < 20:
            return 0.5, 4.5  # Default values
        
        # This is simplified - in reality, you'd need to know the opposing pitcher's handedness
        # For now, assume roughly 70% RHP, 30% LHP in MLB
        
        # Calculate basic splits (this is a placeholder implementation)
        vs_rhp_games = team_games.sample(frac=0.7)  # Simulate RHP games
        vs_lhp_games = team_games.drop(vs_rhp_games.index)  # Remaining are LHP
        
        if opposing_pitcher_hand == 'R':
            relevant_games = vs_rhp_games
        else:
            relevant_games = vs_lhp_games
        
        if len(relevant_games) == 0:
            return 0.5, 4.5
        
        # Calculate performance vs this handedness
        wins = 0
        total_runs = 0
        
        for _, game in relevant_games.iterrows():
            is_home = (game['home_team'] == team)
            team_score = game['home_score'] if is_home else game['away_score']
            team_won = (game['winner'] == 'home' and is_home) or (game['winner'] == 'away' and not is_home)
            
            if team_won:
                wins += 1
            total_runs += team_score
        
        win_rate_vs_hand = wins / len(relevant_games)
        runs_per_game_vs_hand = total_runs / len(relevant_games)
        
        return win_rate_vs_hand, runs_per_game_vs_hand
    
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
    
    def update_historical_data(self):
        """Update with recent completed games"""
        print("\n=== UPDATING HISTORICAL DATABASE ===")
        
        try:
            historical_df = pd.read_csv('historical_mlb_games.csv')
            historical_df['date'] = pd.to_datetime(historical_df['date'])
            last_date = historical_df['date'].max()
            print(f"Last game in database: {last_date.date()}")
        except:
            print("No existing historical file found")
            return False
        
        # Get recent games (simplified version)
        recent_games = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d')
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
            
            try:
                response = requests.get(url)
                data = response.json()
                
                if data['dates'] and len(data['dates']) > 0:
                    games = data['dates'][0]['games']
                    
                    for game in games:
                        if game['status']['statusCode'] == 'F':
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
                            recent_games.append(game_info)
                
                time.sleep(0.3)
            except Exception as e:
                print(f"Error getting games for {date}: {e}")
        
        if recent_games:
            recent_df = pd.DataFrame(recent_games)
            recent_df['date'] = pd.to_datetime(recent_df['date'])
            
            # Find new games
            new_games = []
            for _, game in recent_df.iterrows():
                existing = historical_df[historical_df['game_id'] == game['game_id']]
                if len(existing) == 0:
                    new_games.append(game.to_dict())
            
            if new_games:
                print(f"Adding {len(new_games)} new games")
                new_df = pd.DataFrame(new_games)
                updated_df = pd.concat([historical_df, new_df], ignore_index=True)
                updated_df = updated_df.sort_values('date').drop_duplicates(subset=['game_id'])
                updated_df.to_csv('historical_mlb_games.csv', index=False)
                self.historical_data = updated_df
                return True
            else:
                print("No new games to add")
                self.historical_data = historical_df
                return True
        
        return False
    
    def calculate_enhanced_features(self, team, as_of_date, is_home=True, opposing_team=None):
        """Calculate enhanced features including weather and handedness"""
        
        # Get basic team features (from your existing system)
        team_games = self.historical_data[
            ((self.historical_data['home_team'] == team) | 
             (self.historical_data['away_team'] == team)) &
            (self.historical_data['date'] < as_of_date)
        ].tail(20)
        
        if len(team_games) < self.min_games_for_prediction:
            return self.get_default_enhanced_features()
        
        # Basic stats
        wins = len(team_games[
            ((team_games['home_team'] == team) & (team_games['winner'] == 'home')) |
            ((team_games['away_team'] == team) & (team_games['winner'] == 'away'))
        ])
        win_rate = wins / len(team_games)
        
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
        
        # Enhanced features
        features = {
            'win_rate': win_rate,
            'runs_per_game': runs_per_game,
            'runs_allowed_per_game': runs_allowed_per_game,
            'run_differential': run_differential,
            'games_played': len(team_games),
            
            # Weather features (if available)
            'temperature': 75,
            'humidity': 50,
            'wind_factor': 0,
            'pressure': 1013,
            
            # Handedness features (placeholder)
            'vs_rhp_win_rate': win_rate,
            'vs_lhp_win_rate': win_rate,
            'vs_rhp_runs_per_game': runs_per_game,
            'vs_lhp_runs_per_game': runs_per_game
        }
        
        # Get weather data if this team is playing at home
        if is_home:
            weather = self.get_weather_data(team)
            features.update({
                'temperature': weather['temperature'],
                'humidity': weather['humidity'],
                'wind_factor': weather['wind_factor'],
                'pressure': weather['pressure']
            })
        
        return features
    
    def get_default_enhanced_features(self):
        """Default features when insufficient data"""
        return {
            'win_rate': 0.5,
            'runs_per_game': 4.5,
            'runs_allowed_per_game': 4.5,
            'run_differential': 0.0,
            'games_played': 0,
            'temperature': 75,
            'humidity': 50,
            'wind_factor': 0,
            'pressure': 1013,
            'vs_rhp_win_rate': 0.5,
            'vs_lhp_win_rate': 0.5,
            'vs_rhp_runs_per_game': 4.5,
            'vs_lhp_runs_per_game': 4.5
        }
    
    def train_enhanced_model(self, rolling_window_days=60):
        """Train model with rolling window approach"""
        print(f"\n=== TRAINING ENHANCED MODEL (Rolling {rolling_window_days} days) ===")
        
        # Use only recent data for training (rolling window)
        cutoff_date = datetime.now() - timedelta(days=rolling_window_days)
        recent_data = self.historical_data[self.historical_data['date'] >= cutoff_date]
        
        print(f"Using {len(recent_data)} games from {cutoff_date.date()} onwards")
        
        if len(recent_data) < 100:
            print("Not enough recent data, expanding window...")
            recent_data = self.historical_data.tail(500)
        
        features = []
        labels = []
        
        for idx, game in recent_data.iterrows():
            if idx < 50:  # Need some history
                continue
            
            game_date = game['date']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Calculate enhanced features
            home_features = self.calculate_enhanced_features(home_team, game_date, True, away_team)
            away_features = self.calculate_enhanced_features(away_team, game_date, False, home_team)
            
            if home_features['games_played'] < 5 or away_features['games_played'] < 5:
                continue
            
            # Create feature vector with enhanced data
            feature_vector = [
                # Basic team features
                1,  # home_field
                home_features['win_rate'] - away_features['win_rate'],
                home_features['win_rate'],
                away_features['win_rate'],
                home_features['run_differential'] - away_features['run_differential'],
                home_features['run_differential'],
                away_features['run_differential'],
                
                # Weather features
                home_features['temperature'],
                home_features['humidity'] / 100,  # Normalize
                home_features['wind_factor'],
                home_features['pressure'] / 1000,  # Normalize
                
                # Temperature effects (warmer = more offense)
                (home_features['temperature'] - 70) * 0.01,
                
                # Handedness features (simplified for now)
                home_features['vs_rhp_win_rate'] - away_features['vs_rhp_win_rate'],
                home_features['vs_lhp_win_rate'] - away_features['vs_lhp_win_rate']
            ]
            
            features.append(feature_vector)
            labels.append(1 if game['winner'] == 'home' else 0)
        
        print(f"Training on {len(features)} games with enhanced features")
        
        if len(features) < 50:
            print("Insufficient training data")
            return False
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            min_samples_leaf=5
        )
        self.model.fit(features, labels)
        
        # Accuracy check
        predictions = self.model.predict(features)
        accuracy = (predictions == labels).mean()
        print(f"Enhanced model training accuracy: {accuracy:.1%}")
        
        # Feature importance
        feature_names = [
            'home_field', 'win_rate_diff', 'home_win_rate', 'away_win_rate',
            'run_diff_advantage', 'home_run_diff', 'away_run_diff',
            'temperature', 'humidity', 'wind_factor', 'pressure', 'temp_boost',
            'rhp_advantage', 'lhp_advantage'
        ]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop Enhanced Features:")
        for _, row in importance_df.head(8).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Save model
        joblib.dump(self.model, 'enhanced_mlb_model.pkl')
        print("Enhanced model saved")
        
        return True
    
    def predict_game_enhanced(self, home_team, away_team):
        """Make prediction with enhanced features"""
        if self.model is None:
            try:
                self.model = joblib.load('enhanced_mlb_model.pkl')
                print("Loaded existing enhanced model")
            except:
                print("Training new enhanced model...")
                if not self.train_enhanced_model():
                    return None
        
        current_date = datetime.now()
        
        # Get enhanced features
        home_features = self.calculate_enhanced_features(home_team, current_date, True, away_team)
        away_features = self.calculate_enhanced_features(away_team, current_date, False, home_team)
        
        if home_features['games_played'] < 5 or away_features['games_played'] < 5:
            return {
                'home_win_probability': 0.54,
                'prediction': 'HOME',
                'confidence': 'LOW',
                'reason': 'Insufficient recent data'
            }
        
        # Create feature vector
        features = [[
            1,  # home_field
            home_features['win_rate'] - away_features['win_rate'],
            home_features['win_rate'],
            away_features['win_rate'],
            home_features['run_differential'] - away_features['run_differential'],
            home_features['run_differential'],
            away_features['run_differential'],
            
            home_features['temperature'],
            home_features['humidity'] / 100,
            home_features['wind_factor'],
            home_features['pressure'] / 1000,
            (home_features['temperature'] - 70) * 0.01,
            
            home_features['vs_rhp_win_rate'] - away_features['vs_rhp_win_rate'],
            home_features['vs_lhp_win_rate'] - away_features['vs_lhp_win_rate']
        ]]
        
        # Make prediction
        home_win_prob = self.model.predict_proba(features)[0][1]
        prediction = 'HOME' if home_win_prob > 0.5 else 'AWAY'
        
        # Determine confidence
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
            'home_features': home_features,
            'away_features': away_features,
            'weather_impact': home_features['wind_factor'],
            'temperature_boost': (home_features['temperature'] - 70) * 0.01
        }
    
    def get_todays_games(self):
        """Get today's scheduled games"""
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            games_today = []
            if data['dates'] and len(data['dates']) > 0:
                games = data['dates'][0]['games']
                
                for game in games:
                    if game['status']['statusCode'] in ['S', 'P']:
                        game_info = {
                            'away_team': game['teams']['away']['team']['name'],
                            'home_team': game['teams']['home']['team']['name'],
                            'game_time': game['gameDate']
                        }
                        games_today.append(game_info)
            
            return games_today
            
        except Exception as e:
            print(f"Error getting today's games: {e}")
            return []
    
    def generate_enhanced_predictions(self):
        """Generate enhanced daily predictions"""
        print(f"\n{'='*70}")
        print(f"ENHANCED MLB PREDICTIONS - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"With Weather, Handedness, and Rolling Training")
        print(f"{'='*70}")
        
        todays_games = self.get_todays_games()
        
        if not todays_games:
            print("No games scheduled for today.")
            return []
        
        predictions = []
        
        for game in todays_games:
            home_team = game['home_team']
            away_team = game['away_team']
            
            prediction = self.predict_game_enhanced(home_team, away_team)
            
            if prediction is None:
                continue
            
            print(f"\n{away_team} @ {home_team}")
            print(f"  Prediction: {prediction['prediction']}")
            print(f"  Home Win Probability: {prediction['home_win_probability']:.1%}")
            print(f"  Confidence: {prediction['confidence']}")
            
            # Enhanced information
            if 'home_features' in prediction:
                home_f = prediction['home_features']
                away_f = prediction['away_features']
                
                print(f"  Weather Impact: {prediction.get('weather_impact', 0):+.2f}")
                print(f"  Temperature Boost: {prediction.get('temperature_boost', 0):+.2f}")
                print(f"  Temperature: {home_f['temperature']:.0f}°F")
                print(f"  Wind Factor: {home_f['wind_factor']:+.1f}")
                
                print(f"  Recent Form:")
                print(f"    Home: {home_f['win_rate']:.1%} ({home_f['run_differential']:+.1f} run diff)")
                print(f"    Away: {away_f['win_rate']:.1%} ({away_f['run_differential']:+.1f} run diff)")
            
            if prediction['confidence'] in ['HIGH', 'MEDIUM']:
                print(f"  BETTING RECOMMENDATION: Consider this game")
            else:
                print(f"  Skip - Low confidence")
            
            prediction_record = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'away_team': away_team,
                'home_team': home_team,
                'predicted_winner': prediction['prediction'],
                'home_win_probability': prediction['home_win_probability'],
                'confidence': prediction['confidence'],
                'weather_impact': prediction.get('weather_impact', 0),
                'temperature': prediction.get('home_features', {}).get('temperature', 75)
            }
            predictions.append(prediction_record)
        
        # Save predictions
        if predictions:
            filename = f"enhanced_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
            pd.DataFrame(predictions).to_csv(filename, index=False)
            
            print(f"\nSUMMARY:")
            high_conf = len([p for p in predictions if p['confidence'] == 'HIGH'])
            medium_conf = len([p for p in predictions if p['confidence'] == 'MEDIUM'])
            
            print(f"  Total games: {len(predictions)}")
            print(f"  High confidence: {high_conf}")
            print(f"  Medium confidence: {medium_conf}")
            print(f"  Recommended bets: {high_conf + medium_conf}")
            print(f"  Predictions saved to: {filename}")
        
        return predictions

def main():
    print("Enhanced MLB Betting System v2.0")
    print("Features: Weather Data + Pitcher Handedness + Rolling Training")
    print("="*65)
    
    system = EnhancedMLBBettingSystem()
    
    # Optional: Set weather API key
    api_key = input("Enter OpenWeatherMap API key (or press Enter to skip weather data): ").strip()
    if api_key:
        system.set_weather_api_key(api_key)
        print("Weather data enabled")
    else:
        print("Using default weather values")
    
    if not system.load_historical_data():
        return
    
    choice = input("\nWhat would you like to do?\n1. Generate enhanced predictions\n2. Train enhanced model\n3. Update data and train\n4. All of the above\nChoice (1-4): ")
    
    if choice in ['3', '4']:
        system.update_historical_data()
    
    if choice in ['2', '3', '4']:
        system.train_enhanced_model()
    
    if choice in ['1', '4']:
        predictions = system.generate_enhanced_predictions()

if __name__ == "__main__":
    main()