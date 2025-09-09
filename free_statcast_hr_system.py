import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class StatcastHRPredictor:
    def __init__(self):
        self.player_data = {}
        self.rate_limit_delay = 2  # Seconds between requests to be nice to Baseball Savant
        
    def get_player_statcast_data(self, player_name, start_date, end_date, max_results=100):
        """
        Get limited Statcast data from Baseball Savant's public search
        WARNING: This is heavily rate limited and not suitable for production
        """
        
        print(f"Getting Statcast data for {player_name} from {start_date} to {end_date}")
        
        # Baseball Savant search endpoint (public, but very limited)
        base_url = "https://baseballsavant.mlb.com/statcast_search/csv"
        
        params = {
            'all': 'true',
            'hfPT': '',
            'hfAB': '',
            'hfBBT': '',
            'hfPR': '',
            'hfZ': '',
            'stadium': '',
            'hfBBL': '',
            'hfNewZones': '',
            'hfGT': 'R%7C',
            'hfC': '',
            'hfSea': '2024%7C',  # 2024 season
            'hfSit': '',
            'player_type': 'batter',
            'hfOuts': '',
            'opponent': '',
            'pitcher_throws': '',
            'batter_stands': '',
            'hfSA': '',
            'game_date_gt': start_date,
            'game_date_lt': end_date,
            'hfInfield': '',
            'team': '',
            'position': '',
            'hfOutfield': '',
            'hfRO': '',
            'home_road': '',
            'batters_lookup%5B%5D': '',  # Would need player ID here
            'hfFlag': '',
            'hfPull': '',
            'metric_1': '',
            'hfInn': '',
            'min_pitches': '0',
            'min_results': '0',
            'group_by': 'name',
            'sort_col': 'pitches',
            'player_event_sort': 'h_launch_speed',
            'sort_order': 'desc',
            'min_abs': '0',
            'type': 'details'
        }
        
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            response = requests.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                # Try to parse CSV response
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                
                if len(df) > 0:
                    print(f"Retrieved {len(df)} Statcast records")
                    return df.head(max_results)  # Limit results
                else:
                    print("No data returned from Baseball Savant")
                    return pd.DataFrame()
            else:
                print(f"Failed to get data: HTTP {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error accessing Baseball Savant: {e}")
            return pd.DataFrame()
    
    def get_recent_player_data_simplified(self, player_name, days_back=30):
        """
        Simplified approach using pybaseball (requires pip install pybaseball)
        This is more reliable than scraping Baseball Savant directly
        """
        try:
            import pybaseball as pyb
            
            # Get player lookup
            players = pyb.playerid_lookup(player_name.split()[1], player_name.split()[0])
            
            if len(players) == 0:
                print(f"Player {player_name} not found")
                return pd.DataFrame()
            
            player_id = players.iloc[0]['key_mlbam']
            
            # Get recent Statcast data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # This uses Baseball Savant's data but through a Python library
            data = pyb.statcast_batter(
                start_dt=start_date.strftime('%Y-%m-%d'),
                end_dt=end_date.strftime('%Y-%m-%d'),
                player_id=player_id
            )
            
            return data
            
        except ImportError:
            print("pybaseball not installed. Run: pip install pybaseball")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error getting player data: {e}")
            return pd.DataFrame()
    
    def calculate_hr_features(self, statcast_data):
        """
        Calculate home run predictive features from Statcast data
        """
        if len(statcast_data) == 0:
            return self.get_default_hr_features()
        
        # Filter out nulls
        data = statcast_data.dropna(subset=['launch_speed', 'launch_angle'])
        
        if len(data) == 0:
            return self.get_default_hr_features()
        
        # Calculate key metrics
        features = {}
        
        # Exit velocity metrics
        features['avg_exit_velocity'] = data['launch_speed'].mean()
        features['max_exit_velocity'] = data['launch_speed'].max()
        features['hard_hit_rate'] = (data['launch_speed'] >= 95).mean()
        features['very_hard_hit_rate'] = (data['launch_speed'] >= 105).mean()
        
        # Launch angle metrics
        features['avg_launch_angle'] = data['launch_angle'].mean()
        features['optimal_launch_angle_rate'] = ((data['launch_angle'] >= 25) & (data['launch_angle'] <= 35)).mean()
        
        # Barrel rate (combination of exit velo and launch angle)
        barrels = ((data['launch_speed'] >= 98) & 
                  (data['launch_angle'] >= 26) & 
                  (data['launch_angle'] <= 30))
        features['barrel_rate'] = barrels.mean()
        
        # Power metrics
        features['hr_rate'] = (data['events'] == 'home_run').mean() if 'events' in data.columns else 0
        features['recent_hrs'] = (data['events'] == 'home_run').sum() if 'events' in data.columns else 0
        
        # Contact quality
        features['sweet_spot_rate'] = ((data['launch_angle'] >= 8) & (data['launch_angle'] <= 32)).mean()
        
        # Recent form (last 10 games)
        if len(data) >= 10:
            recent_data = data.tail(10)
            features['recent_exit_velo'] = recent_data['launch_speed'].mean()
            features['recent_hard_hit_rate'] = (recent_data['launch_speed'] >= 95).mean()
            features['recent_barrel_rate'] = ((recent_data['launch_speed'] >= 98) & 
                                            (recent_data['launch_angle'] >= 26) & 
                                            (recent_data['launch_angle'] <= 30)).mean()
        else:
            features['recent_exit_velo'] = features['avg_exit_velocity']
            features['recent_hard_hit_rate'] = features['hard_hit_rate']
            features['recent_barrel_rate'] = features['barrel_rate']
        
        # Sample size
        features['plate_appearances'] = len(data)
        
        return features
    
    def get_default_hr_features(self):
        """Default features when no data available"""
        return {
            'avg_exit_velocity': 88.0,
            'max_exit_velocity': 105.0,
            'hard_hit_rate': 0.35,
            'very_hard_hit_rate': 0.15,
            'avg_launch_angle': 12.0,
            'optimal_launch_angle_rate': 0.25,
            'barrel_rate': 0.08,
            'hr_rate': 0.05,
            'recent_hrs': 1,
            'sweet_spot_rate': 0.35,
            'recent_exit_velo': 88.0,
            'recent_hard_hit_rate': 0.35,
            'recent_barrel_rate': 0.08,
            'plate_appearances': 20
        }
    
    def get_pitcher_hr_allowed_data(self, pitcher_name):
        """
        Get pitcher's home run allowance data (simplified)
        In reality, you'd want detailed Statcast data on pitcher performance
        """
        # This would require similar Statcast calls for pitchers
        # For now, return basic estimates
        return {
            'hr_per_9': 1.2,  # League average
            'hard_contact_allowed': 0.38,
            'fb_velocity': 92.5,
            'recent_hrs_allowed': 2
        }
    
    def get_ballpark_hr_factor(self, ballpark):
        """
        Ballpark home run factors (simplified)
        Real implementation would use detailed ballpark data
        """
        hr_factors = {
            'Yankee Stadium': 1.15,
            'Coors Field': 1.25,
            'Fenway Park': 1.10,
            'Minute Maid Park': 1.05,
            'Petco Park': 0.85,
            'Marlins Park': 0.90,
            'Tropicana Field': 0.95,
            'Kauffman Stadium': 0.90
        }
        
        return hr_factors.get(ballpark, 1.0)  # Default neutral
    
    def predict_hr_probability(self, player_name, opposing_pitcher, ballpark, weather_temp=75):
        """
        Predict home run probability for a specific player
        """
        
        print(f"Predicting HR probability for {player_name}")
        
        # Get player Statcast data (last 30 days)
        player_data = self.get_recent_player_data_simplified(player_name, days_back=30)
        
        # Calculate features
        player_features = self.calculate_hr_features(player_data)
        pitcher_features = self.get_pitcher_hr_allowed_data(opposing_pitcher)
        
        # Environmental factors
        ballpark_factor = self.get_ballpark_hr_factor(ballpark)
        temp_factor = 1 + ((weather_temp - 70) * 0.004)  # 0.4% per degree
        
        # Simple probability model based on key factors
        base_hr_rate = player_features['hr_rate']
        
        # Adjustments
        exit_velo_boost = (player_features['avg_exit_velocity'] - 88) * 0.002
        barrel_boost = (player_features['barrel_rate'] - 0.08) * 2.0
        recent_form_boost = (player_features['recent_barrel_rate'] - player_features['barrel_rate']) * 1.5
        
        # Environmental boosts
        ballpark_boost = (ballpark_factor - 1.0)
        weather_boost = (temp_factor - 1.0)
        
        # Calculate final probability
        adjusted_probability = base_hr_rate * (
            1 + exit_velo_boost + barrel_boost + recent_form_boost + 
            ballpark_boost + weather_boost
        )
        
        # Cap at reasonable limits
        final_probability = max(0.01, min(0.25, adjusted_probability))
        
        return {
            'player_name': player_name,
            'hr_probability': final_probability,
            'base_rate': base_hr_rate,
            'player_features': player_features,
            'adjustments': {
                'exit_velocity': exit_velo_boost,
                'barrel_rate': barrel_boost,
                'recent_form': recent_form_boost,
                'ballpark': ballpark_boost,
                'weather': weather_boost
            },
            'recommendation': 'BET' if final_probability > 0.15 else 'PASS'
        }
    
    def analyze_todays_hr_props(self):
        """
        Analyze today's home run props (simplified example)
        """
        print("=== TODAY'S HOME RUN PROP ANALYSIS ===\n")
        
        # Example players (you'd get this from today's lineups)
        sample_players = [
            {'name': 'Aaron Judge', 'pitcher': 'Shane Bieber', 'ballpark': 'Yankee Stadium'},
            {'name': 'Mookie Betts', 'pitcher': 'Jacob deGrom', 'ballpark': 'Dodger Stadium'},
            {'name': 'Vladimir Guerrero Jr.', 'pitcher': 'Gerrit Cole', 'ballpark': 'Rogers Centre'}
        ]
        
        predictions = []
        
        for player_info in sample_players:
            try:
                prediction = self.predict_hr_probability(
                    player_info['name'], 
                    player_info['pitcher'],
                    player_info['ballpark']
                )
                predictions.append(prediction)
                
                # Display results
                print(f"{player_info['name']} vs {player_info['pitcher']}:")
                print(f"  HR Probability: {prediction['hr_probability']:.1%}")
                print(f"  Base Rate: {prediction['base_rate']:.1%}")
                print(f"  Exit Velocity: {prediction['player_features']['avg_exit_velocity']:.1f} MPH")
                print(f"  Barrel Rate: {prediction['player_features']['barrel_rate']:.1%}")
                print(f"  Recent HRs: {prediction['player_features']['recent_hrs']}")
                print(f"  Recommendation: {prediction['recommendation']}")
                print(f"  Key Adjustments:")
                for adj_name, adj_value in prediction['adjustments'].items():
                    if abs(adj_value) > 0.001:
                        print(f"    {adj_name.title()}: {adj_value:+.1%}")
                print()
                
            except Exception as e:
                print(f"Error analyzing {player_info['name']}: {e}")
                continue
        
        return predictions

def main():
    print("Free Statcast Home Run Predictor")
    print("="*50)
    print("WARNING: This uses limited free data and is for testing only!")
    print("Production systems require paid Statcast subscriptions.")
    print()
    
    predictor = StatcastHRPredictor()
    
    choice = input("What would you like to do?\n1. Analyze specific player\n2. Today's HR props analysis\n3. Test with sample data\nChoice (1-3): ")
    
    if choice == '1':
        player_name = input("Enter player name (First Last): ")
        pitcher = input("Enter opposing pitcher: ")
        ballpark = input("Enter ballpark: ")
        
        try:
            result = predictor.predict_hr_probability(player_name, pitcher, ballpark)
            print("\n=== PREDICTION RESULTS ===")
            print(f"Player: {result['player_name']}")
            print(f"HR Probability: {result['hr_probability']:.1%}")
            print(f"Recommendation: {result['recommendation']}")
        except Exception as e:
            print(f"Error: {e}")
    
    elif choice == '2':
        predictor.analyze_todays_hr_props()
    
    elif choice == '3':
        print("\n=== TESTING WITH SAMPLE DATA ===")
        print("This demonstrates the concept with simulated data...")
        
        # Create sample Statcast-like data
        sample_data = pd.DataFrame({
            'launch_speed': np.random.normal(90, 8, 100),
            'launch_angle': np.random.normal(15, 12, 100),
            'events': ['single', 'out', 'home_run', 'double', 'out'] * 20
        })
        
        features = predictor.calculate_hr_features(sample_data)
        print("Sample player features:")
        for key, value in features.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    print(f"\n=== IMPORTANT LIMITATIONS ===")
    print("• Free data is heavily rate-limited (25 requests/hour)")
    print("• Historical data only, no real-time lineup info")
    print("• Missing advanced pitcher metrics")
    print("• No injury/rest day information")
    print("• Simplified ballpark factors")
    print("\nFor production use, you need paid Statcast subscriptions!")

if __name__ == "__main__":
    main()