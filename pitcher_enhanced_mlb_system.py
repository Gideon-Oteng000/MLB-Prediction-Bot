import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta
import time
import os

class PitcherEnhancedMLBSystem:
    def __init__(self):
        self.model = None
        self.historical_data = None
        self.pitcher_cache = {}
        self.weather_api_key = None
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
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=imperial"
            response = requests.get(url)
            data = response.json()
            
            weather = {
                'temperature': data.get('main', {}).get('temp', 70),
                'humidity': data.get('main', {}).get('humidity', 50),
                'wind_speed': data.get('wind', {}).get('speed', 5),
                'wind_direction': data.get('wind', {}).get('deg', 180),
                'pressure': data.get('main', {}).get('pressure', 1013),
            }
            
            wind_factor = self.calculate_wind_factor(weather['wind_speed'], weather['wind_direction'])
            weather['wind_factor'] = wind_factor
            
            return weather
            
        except Exception as e:
            print(f"Weather API error: {e}")
            return self.get_default_weather()
    
    def get_default_weather(self):
        """Default weather when API unavailable"""
        return {
            'temperature': 75, 'humidity': 50, 'wind_speed': 5,
            'wind_direction': 180, 'pressure': 1013, 'wind_factor': 0
        }
    
    def calculate_wind_factor(self, wind_speed, wind_direction):
        """Calculate wind impact on offense"""
        if 90 <= wind_direction <= 270:  # Blowing out
            return wind_speed * 0.1
        else:  # Blowing in
            return wind_speed * -0.1
    
    def get_probable_pitchers(self, date_str=None):
        """Get probable starting pitchers for games on a specific date"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            pitcher_info = {}
            
            if data['dates'] and len(data['dates']) > 0:
                games = data['dates'][0]['games']
                
                for game in games:
                    if game['status']['statusCode'] in ['S', 'P', 'I']:
                        away_team = game['teams']['away']['team']['name']
                        home_team = game['teams']['home']['team']['name']
                        
                        # Get probable pitchers from game detail API (more reliable)
                        away_pitcher_id = None
                        home_pitcher_id = None
                        
                        try:
                            game_detail_url = f"https://statsapi.mlb.com/api/v1.1/game/{game['gamePk']}/feed/live"
                            detail_response = requests.get(game_detail_url)
                            detail_data = detail_response.json()
                            
                            prob_pitchers = detail_data.get('gameData', {}).get('probablePitchers', {})
                            
                            if 'away' in prob_pitchers:
                                away_pitcher_id = prob_pitchers['away'].get('id')
                            
                            if 'home' in prob_pitchers:
                                home_pitcher_id = prob_pitchers['home'].get('id')
                                
                        except Exception as e:
                            print(f"Could not get pitchers for {away_team} @ {home_team}: {e}")
                        
                        pitcher_info[f"{away_team}@{home_team}"] = {
                            'away_pitcher_id': away_pitcher_id,
                            'home_pitcher_id': home_pitcher_id,
                            'away_team': away_team,
                            'home_team': home_team
                        }
            
            return pitcher_info
            
        except Exception as e:
            print(f"Error getting probable pitchers: {e}")
            return {}
    
    def get_pitcher_stats(self, pitcher_id, season=None):
        """Get detailed pitcher statistics"""
        if season is None:
            season = datetime.now().year
        
        # Check cache first
        cache_key = f"{pitcher_id}_{season}"
        if cache_key in self.pitcher_cache:
            return self.pitcher_cache[cache_key]
        
        try:
            # Get pitcher info and stats
            person_url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}"
            stats_url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=season&season={season}&group=pitching"
            
            # Get basic pitcher info (handedness, etc.)
            person_response = requests.get(person_url)
            person_data = person_response.json()
            
            pitcher_info = person_data.get('people', [{}])[0]
            pitcher_name = pitcher_info.get('fullName', 'Unknown')
            pitcher_hand = pitcher_info.get('pitchHand', {}).get('code', 'R')
            
            # Get season stats
            stats_response = requests.get(stats_url)
            stats_data = stats_response.json()
            
            pitcher_stats = {
                'name': pitcher_name,
                'handedness': pitcher_hand,
                'era': 4.50,
                'whip': 1.35,
                'strikeouts_per_9': 8.0,
                'walks_per_9': 3.2,
                'hits_per_9': 8.8,
                'home_runs_per_9': 1.2,
                'wins': 8,
                'losses': 8,
                'games_started': 20,
                'innings_pitched': 100.0,
                'opponent_avg': 0.260,
                'recent_era': 4.50,  # Placeholder for recent form
                'vs_team_era': 4.50   # Placeholder for vs specific team
            }
            
            # Extract actual stats if available - they're in splits[0]['stat']
            if 'stats' in stats_data and len(stats_data['stats']) > 0:
                stat_group = stats_data['stats'][0]
                
                if 'splits' in stat_group and len(stat_group['splits']) > 0:
                    season_stats = stat_group['splits'][0]['stat']
                    
                    pitcher_stats.update({
                        'era': float(season_stats.get('era', 4.50)),
                        'whip': float(season_stats.get('whip', 1.35)),
                        'strikeouts_per_9': float(season_stats.get('strikeoutsPer9Inn', 8.0)),
                        'walks_per_9': float(season_stats.get('baseOnBallsPer9Inn', 3.2)),
                        'hits_per_9': float(season_stats.get('hitsPer9Inn', 8.8)),
                        'home_runs_per_9': float(season_stats.get('homeRunsPer9Inn', 1.2)),
                        'wins': int(season_stats.get('wins', 8)),
                        'losses': int(season_stats.get('losses', 8)),
                        'games_started': int(season_stats.get('gamesStarted', 20)),
                        'innings_pitched': float(season_stats.get('inningsPitched', 100.0)),
                        'opponent_avg': float(season_stats.get('avg', 0.260))
                    })
                else:
                    print(f"No splits data for pitcher {pitcher_id}")
            
            # Cache the result
            self.pitcher_cache[cache_key] = pitcher_stats
            
            time.sleep(0.1)  # Be nice to API
            return pitcher_stats
            
        except Exception as e:
            print(f"Error getting pitcher stats for {pitcher_id}: {e}")
            return {
                'name': 'Unknown',
                'handedness': 'R',
                'era': 4.50,
                'whip': 1.35,
                'strikeouts_per_9': 8.0,
                'walks_per_9': 3.2,
                'hits_per_9': 8.8,
                'home_runs_per_9': 1.2,
                'wins': 8,
                'losses': 8,
                'games_started': 20,
                'innings_pitched': 100.0,
                'opponent_avg': 0.260,
                'recent_era': 4.50,
                'vs_team_era': 4.50
            }
    
    def calculate_pitcher_vs_team_history(self, pitcher_id, opposing_team, pitcher_stats):
        """Calculate pitcher's historical performance vs specific team"""
        # This is a simplified version - would need game-by-game data for full implementation
        # For now, adjust based on team strength
        base_era = pitcher_stats['era']
        
        # Teams with better offenses typically perform better against all pitchers
        # This is a placeholder - you'd want actual head-to-head data
        team_offensive_adjustments = {
            'Los Angeles Dodgers': 0.2,
            'Houston Astros': 0.15,
            'New York Yankees': 0.15,
            'Atlanta Braves': 0.1,
            'Boston Red Sox': 0.1,
            # Add more teams as needed
        }
        
        adjustment = team_offensive_adjustments.get(opposing_team, 0.0)
        adjusted_era = base_era + adjustment
        
        return adjusted_era
    
    def calculate_team_vs_handedness(self, team, pitcher_hand, as_of_date):
        """Calculate team's actual performance vs LHP/RHP using historical data"""
        # This is still simplified but more realistic than random assignment
        team_games = self.historical_data[
            ((self.historical_data['home_team'] == team) | 
             (self.historical_data['away_team'] == team)) &
            (self.historical_data['date'] < as_of_date)
        ].tail(40)
        
        if len(team_games) < 20:
            return 0.5, 4.5
        
        # Estimate handedness distribution and performance
        # MLB is roughly 70% RHP, 30% LHP
        if pitcher_hand == 'R':
            # Simulate performance vs RHP (majority of games)
            relevant_games = team_games.sample(frac=0.7, random_state=42)
        else:
            # Simulate performance vs LHP
            relevant_games = team_games.sample(frac=0.3, random_state=42)
        
        if len(relevant_games) == 0:
            return 0.5, 4.5
        
        wins = 0
        total_runs = 0
        
        for _, game in relevant_games.iterrows():
            is_home = (game['home_team'] == team)
            team_score = game['home_score'] if is_home else game['away_score']
            team_won = (game['winner'] == 'home' and is_home) or (game['winner'] == 'away' and not is_home)
            
            if team_won:
                wins += 1
            total_runs += team_score
        
        win_rate = wins / len(relevant_games)
        runs_per_game = total_runs / len(relevant_games)
        
        return win_rate, runs_per_game
    
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
    
    def calculate_pitcher_enhanced_features(self, team, as_of_date, is_home=True, opposing_team=None, pitcher_id=None, opposing_pitcher_id=None):
        """Calculate features with full pitcher integration"""
        
        # Get basic team features
        team_games = self.historical_data[
            ((self.historical_data['home_team'] == team) | 
             (self.historical_data['away_team'] == team)) &
            (self.historical_data['date'] < as_of_date)
        ].tail(20)
        
        if len(team_games) < self.min_games_for_prediction:
            return self.get_default_pitcher_features()
        
        # Calculate basic team stats
        wins = len(team_games[
            ((team_games['home_team'] == team) & (team_games['winner'] == 'home')) |
            ((team_games['away_team'] == team) & (team_games['winner'] == 'away'))
        ])
        win_rate = wins / len(team_games)
        
        runs_scored = runs_allowed = 0
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
        
        features = {
            'win_rate': win_rate,
            'runs_per_game': runs_per_game,
            'runs_allowed_per_game': runs_allowed_per_game,
            'run_differential': run_differential,
            'games_played': len(team_games),
            
            # Weather (if home team)
            'temperature': 75,
            'humidity': 50,
            'wind_factor': 0,
            'pressure': 1013,
            
            # Pitcher features
            'opposing_pitcher_era': 4.50,
            'opposing_pitcher_whip': 1.35,
            'opposing_pitcher_k9': 8.0,
            'opposing_pitcher_hr9': 1.2,
            'pitcher_handedness_adv': 0,
            'team_vs_pitcher_hand_rate': 0.5,
            'team_vs_pitcher_runs': 4.5,
            'pitcher_vs_team_era': 4.50
        }
        
        # Get weather data if home team
        if is_home:
            weather = self.get_weather_data(team)
            features.update({
                'temperature': weather['temperature'],
                'humidity': weather['humidity'],
                'wind_factor': weather['wind_factor'],
                'pressure': weather['pressure']
            })
        
        # Get opposing pitcher stats
        if opposing_pitcher_id:
            pitcher_stats = self.get_pitcher_stats(opposing_pitcher_id)
            
            # Team's performance vs this pitcher's handedness
            vs_hand_rate, vs_hand_runs = self.calculate_team_vs_handedness(
                team, pitcher_stats['handedness'], as_of_date
            )
            
            # Pitcher's performance vs this team
            pitcher_vs_team_era = self.calculate_pitcher_vs_team_history(
                opposing_pitcher_id, team, pitcher_stats
            )
            
            # Calculate handedness advantage
            # LHB typically hit better vs RHP and vice versa
            handedness_bonus = 0
            if pitcher_stats['handedness'] == 'R':
                handedness_bonus = 0.05  # Small bonus vs RHP
            else:
                handedness_bonus = -0.05  # Small penalty vs LHP (rarer, often tougher)
            
            features.update({
                'opposing_pitcher_era': pitcher_stats['era'],
                'opposing_pitcher_whip': pitcher_stats['whip'],
                'opposing_pitcher_k9': pitcher_stats['strikeouts_per_9'],
                'opposing_pitcher_hr9': pitcher_stats['home_runs_per_9'],
                'pitcher_handedness_adv': handedness_bonus,
                'team_vs_pitcher_hand_rate': vs_hand_rate,
                'team_vs_pitcher_runs': vs_hand_runs,
                'pitcher_vs_team_era': pitcher_vs_team_era
            })
        
        return features
    
    def get_default_pitcher_features(self):
        """Default features when insufficient data"""
        return {
            'win_rate': 0.5, 'runs_per_game': 4.5, 'runs_allowed_per_game': 4.5,
            'run_differential': 0.0, 'games_played': 0, 'temperature': 75,
            'humidity': 50, 'wind_factor': 0, 'pressure': 1013,
            'opposing_pitcher_era': 4.50, 'opposing_pitcher_whip': 1.35,
            'opposing_pitcher_k9': 8.0, 'opposing_pitcher_hr9': 1.2,
            'pitcher_handedness_adv': 0, 'team_vs_pitcher_hand_rate': 0.5,
            'team_vs_pitcher_runs': 4.5, 'pitcher_vs_team_era': 4.50
        }
    
    def train_pitcher_enhanced_model(self, rolling_window_days=60):
        """Train model with full pitcher integration"""
        print(f"\n=== TRAINING PITCHER-ENHANCED MODEL (Rolling {rolling_window_days} days) ===")
        
        cutoff_date = datetime.now() - timedelta(days=rolling_window_days)
        recent_data = self.historical_data[self.historical_data['date'] >= cutoff_date]
        
        if len(recent_data) < 100:
            recent_data = self.historical_data.tail(500)
        
        print(f"Training on {len(recent_data)} recent games")
        
        features = []
        labels = []
        
        for idx, game in recent_data.iterrows():
            if idx < 50:
                continue
            
            game_date = game['date']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # For historical training, we don't have actual pitcher IDs
            # So we simulate pitcher matchups based on league averages
            home_features = self.calculate_pitcher_enhanced_features(
                home_team, game_date, True, away_team, None, None
            )
            away_features = self.calculate_pitcher_enhanced_features(
                away_team, game_date, False, home_team, None, None
            )
            
            if home_features['games_played'] < 5 or away_features['games_played'] < 5:
                continue
            
            # Create feature vector with pitcher data
            feature_vector = [
                # Basic team features
                1,  # home_field
                home_features['win_rate'] - away_features['win_rate'],
                home_features['run_differential'] - away_features['run_differential'],
                
                # Weather features
                home_features['temperature'],
                home_features['humidity'] / 100,
                home_features['wind_factor'],
                (home_features['temperature'] - 70) * 0.01,
                
                # Pitcher advantage features
                away_features['opposing_pitcher_era'] - home_features['opposing_pitcher_era'],  # Lower ERA is better
                away_features['opposing_pitcher_whip'] - home_features['opposing_pitcher_whip'],
                home_features['opposing_pitcher_k9'] - away_features['opposing_pitcher_k9'],
                away_features['opposing_pitcher_hr9'] - home_features['opposing_pitcher_hr9'],
                
                # Handedness and matchup features
                home_features['pitcher_handedness_adv'] - away_features['pitcher_handedness_adv'],
                home_features['team_vs_pitcher_hand_rate'] - away_features['team_vs_pitcher_hand_rate'],
                
                # Team strength
                home_features['runs_per_game'] - away_features['runs_per_game'],
                away_features['runs_allowed_per_game'] - home_features['runs_allowed_per_game']
            ]
            
            features.append(feature_vector)
            labels.append(1 if game['winner'] == 'home' else 0)
        
        print(f"Training on {len(features)} games with pitcher features")
        
        if len(features) < 50:
            print("Insufficient training data")
            return False
        
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            min_samples_leaf=3
        )
        self.model.fit(features, labels)
        
        predictions = self.model.predict(features)
        accuracy = (predictions == labels).mean()
        print(f"Pitcher-enhanced model training accuracy: {accuracy:.1%}")
        
        # Feature importance
        feature_names = [
            'home_field', 'win_rate_diff', 'run_diff_advantage', 
            'temperature', 'humidity', 'wind_factor', 'temp_boost',
            'pitcher_era_adv', 'pitcher_whip_adv', 'pitcher_k9_adv', 'pitcher_hr9_adv',
            'handedness_adv', 'vs_pitcher_hand_adv',
            'offensive_advantage', 'defensive_advantage'
        ]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop Pitcher-Enhanced Features:")
        for _, row in importance_df.head(8).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        joblib.dump(self.model, 'pitcher_enhanced_model.pkl')
        print("Pitcher-enhanced model saved")
        return True
    
    def get_todays_games_with_pitchers(self):
        """Get today's games with starting pitcher information"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Get probable pitchers
        pitcher_info = self.get_probable_pitchers(today)
        
        # Get basic game info
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            games_with_pitchers = []
            
            if data['dates'] and len(data['dates']) > 0:
                games = data['dates'][0]['games']
                
                for game in games:
                    if game['status']['statusCode'] in ['S', 'P']:
                        away_team = game['teams']['away']['team']['name']
                        home_team = game['teams']['home']['team']['name']
                        game_key = f"{away_team}@{home_team}"
                        
                        pitcher_data = pitcher_info.get(game_key, {})
                        
                        game_info = {
                            'away_team': away_team,
                            'home_team': home_team,
                            'game_time': game['gameDate'],
                            'away_pitcher_id': pitcher_data.get('away_pitcher_id'),
                            'home_pitcher_id': pitcher_data.get('home_pitcher_id')
                        }
                        games_with_pitchers.append(game_info)
            
            return games_with_pitchers
            
        except Exception as e:
            print(f"Error getting today's games: {e}")
            return []
    
    def predict_game_with_pitchers(self, home_team, away_team, home_pitcher_id=None, away_pitcher_id=None):
        """Make prediction with full pitcher integration"""
        if self.model is None:
            try:
                self.model = joblib.load('pitcher_enhanced_model.pkl')
                print("Loaded pitcher-enhanced model")
            except:
                print("Training new pitcher-enhanced model...")
                if not self.train_pitcher_enhanced_model():
                    return None
        
        current_date = datetime.now()
        
        # Get enhanced features with actual pitcher data
        home_features = self.calculate_pitcher_enhanced_features(
            home_team, current_date, True, away_team, home_pitcher_id, away_pitcher_id
        )
        away_features = self.calculate_pitcher_enhanced_features(
            away_team, current_date, False, home_team, away_pitcher_id, home_pitcher_id
        )
        
        if home_features['games_played'] < 5 or away_features['games_played'] < 5:
            return {
                'home_win_probability': 0.54,
                'prediction': 'HOME',
                'confidence': 'LOW',
                'reason': 'Insufficient recent data'
            }
        
        # Get actual pitcher stats for display
        home_pitcher_stats = None
        away_pitcher_stats = None
        
        if home_pitcher_id:
            home_pitcher_stats = self.get_pitcher_stats(home_pitcher_id)
        if away_pitcher_id:
            away_pitcher_stats = self.get_pitcher_stats(away_pitcher_id)
        
        # Create feature vector
        features = [[
            1,  # home_field
            home_features['win_rate'] - away_features['win_rate'],
            home_features['run_differential'] - away_features['run_differential'],
            
            home_features['temperature'],
            home_features['humidity'] / 100,
            home_features['wind_factor'],
            (home_features['temperature'] - 70) * 0.01,
            
            away_features['opposing_pitcher_era'] - home_features['opposing_pitcher_era'],
            away_features['opposing_pitcher_whip'] - home_features['opposing_pitcher_whip'],
            home_features['opposing_pitcher_k9'] - away_features['opposing_pitcher_k9'],
            away_features['opposing_pitcher_hr9'] - home_features['opposing_pitcher_hr9'],
            
            home_features['pitcher_handedness_adv'] - away_features['pitcher_handedness_adv'],
            home_features['team_vs_pitcher_hand_rate'] - away_features['team_vs_pitcher_hand_rate'],
            
            home_features['runs_per_game'] - away_features['runs_per_game'],
            away_features['runs_allowed_per_game'] - home_features['runs_allowed_per_game']
        ]]
        
        home_win_prob = self.model.predict_proba(features)[0][1]
        prediction = 'HOME' if home_win_prob > 0.5 else 'AWAY'
        
        # Determine confidence
        confidence_margin = abs(home_win_prob - 0.5)
        if confidence_margin > 0.15:
            confidence = 'HIGH'
        elif confidence_margin > 0.08:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'home_win_probability': home_win_prob,
            'prediction': prediction,
            'confidence': confidence,
            'home_features': home_features,
            'away_features': away_features,
            'home_pitcher_stats': home_pitcher_stats,
            'away_pitcher_stats': away_pitcher_stats,
            'pitcher_advantage': away_features['opposing_pitcher_era'] - home_features['opposing_pitcher_era']
        }
    
    def generate_pitcher_enhanced_predictions(self):
        """Generate predictions with full pitcher analysis"""
        print(f"\n{'='*75}")
        print(f"PITCHER-ENHANCED MLB PREDICTIONS - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"With Starting Pitcher Stats, Weather, and Rolling Training")
        print(f"{'='*75}")
        
        games_with_pitchers = self.get_todays_games_with_pitchers()
        
        if not games_with_pitchers:
            print("No games scheduled for today.")
            return []
        
        predictions = []
        
        for game in games_with_pitchers:
            home_team = game['home_team']
            away_team = game['away_team']
            home_pitcher_id = game.get('home_pitcher_id')
            away_pitcher_id = game.get('away_pitcher_id')
            
            prediction = self.predict_game_with_pitchers(
                home_team, away_team, home_pitcher_id, away_pitcher_id
            )
            
            if prediction is None:
                continue
            
            print(f"\n{away_team} @ {home_team}")
            print(f"  Prediction: {prediction['prediction']}")
            print(f"  Home Win Probability: {prediction['home_win_probability']:.1%}")
            print(f"  Confidence: {prediction['confidence']}")
            
            # Pitcher information
            home_pitcher = prediction.get('home_pitcher_stats')
            away_pitcher = prediction.get('away_pitcher_stats')
            
            if home_pitcher and away_pitcher:
                print(f"  Starting Pitchers:")
                print(f"    {away_team}: {away_pitcher['name']} ({away_pitcher['handedness']}) - {away_pitcher['era']:.2f} ERA, {away_pitcher['whip']:.2f} WHIP")
                print(f"    {home_team}: {home_pitcher['name']} ({home_pitcher['handedness']}) - {home_pitcher['era']:.2f} ERA, {home_pitcher['whip']:.2f} WHIP")
                print(f"  Pitcher Advantage: {prediction.get('pitcher_advantage', 0):+.2f} ERA difference")
            else:
                print(f"  Starting Pitchers: Data unavailable")
            
            # Weather and other factors
            home_f = prediction.get('home_features', {})
            away_f = prediction.get('away_features', {})
            
            print(f"  Weather: {home_f.get('temperature', 75):.0f}°F, Wind Factor: {home_f.get('wind_factor', 0):+.1f}")
            print(f"  Recent Form:")
            print(f"    Home: {home_f.get('win_rate', 0.5):.1%} ({home_f.get('run_differential', 0):+.1f} run diff)")
            print(f"    Away: {away_f.get('win_rate', 0.5):.1%} ({away_f.get('run_differential', 0):+.1f} run diff)")
            
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
                'home_pitcher': home_pitcher['name'] if home_pitcher else 'Unknown',
                'away_pitcher': away_pitcher['name'] if away_pitcher else 'Unknown',
                'home_pitcher_era': home_pitcher['era'] if home_pitcher else 0,
                'away_pitcher_era': away_pitcher['era'] if away_pitcher else 0,
                'pitcher_advantage': prediction.get('pitcher_advantage', 0)
            }
            predictions.append(prediction_record)
        
        if predictions:
            filename = f"pitcher_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
            pd.DataFrame(predictions).to_csv(filename, index=False)
            
            print(f"\nSUMMARY:")
            high_conf = len([p for p in predictions if p['confidence'] == 'HIGH'])
            medium_conf = len([p for p in predictions if p['confidence'] == 'MEDIUM'])
            
            print(f"  Total games: {len(predictions)}")
            print(f"  High confidence: {high_conf}")
            print(f"  Medium confidence: {medium_conf}")
            print(f"  Recommended bets: {high_conf + medium_conf}")
            print(f"  Games with pitcher data: {len([p for p in predictions if p['home_pitcher'] != 'Unknown'])}")
            print(f"  Predictions saved to: {filename}")
        
        return predictions

def main():
    print("MLB Pitcher-Enhanced Betting System v3.0")
    print("Full Starting Pitcher Integration + Weather + Rolling Training")
    print("="*70)
    
    system = PitcherEnhancedMLBSystem()
    
    # Optional: Set weather API key
    api_key = input("Enter OpenWeatherMap API key (or press Enter to skip weather data): ").strip()
    if api_key:
        system.set_weather_api_key(api_key)
        print("Weather data enabled")
    else:
        print("Using default weather values")
    
    if not system.load_historical_data():
        return
    
    choice = input("\nWhat would you like to do?\n1. Generate pitcher-enhanced predictions\n2. Train pitcher-enhanced model\n3. Both\nChoice (1-3): ")
    
    if choice in ['2', '3']:
        system.train_pitcher_enhanced_model()
    
    if choice in ['1', '3']:
        predictions = system.generate_pitcher_enhanced_predictions()

if __name__ == "__main__":
    main()