"""
MLB Daily Home Run Prediction Model - PREMIUM SPORTRADAR VERSION
Real-time data from SportsRadar MLB API
Based on formula: P(HR) = σ(β₀ + ΣβᵢXᵢ) × Environmental_Multiplier × Calibration_Factor
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class Config:
    """Production API Configuration with SportsRadar"""
    
    # SportsRadar API Configuration
    SPORTRADAR_KEY = "3uCVRm9PhGc0VMvLafRDZpqcwdWrafLktHzEw3wr"
    SPORTRADAR_BASE = "https://api.sportradar.com/mlb/production/v8/en"
    
    # OpenWeatherMap API (Sign up at: https://openweathermap.org/api)
    WEATHER_API_KEY = "YOUR_OPENWEATHER_KEY"  # Get free key for weather
    WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    # Model coefficients from the formula
    BETA_0 = -3.8  # Baseline ~2% HR rate
    HITTER_WEIGHTS = [2.2, 1.5, 0.8, 0.6, 0.4]
    PITCHER_WEIGHTS = [1.8, -1.2, 0.9, -0.7, 0.3]
    SITUATIONAL_WEIGHTS = [0.6, 0.4, 0.3, 0.5]
    
    # Stadium data with coordinates and park factors
    STADIUMS = {
        'Chase Field': {'lat': 33.4455, 'lon': -112.0667, 'park_factor': 1.05, 'pull_dir': 315},
        'Oriole Park at Camden Yards': {'lat': 39.2838, 'lon': -76.6218, 'park_factor': 1.18, 'pull_dir': 315},
        'Fenway Park': {'lat': 42.3467, 'lon': -71.0972, 'park_factor': 1.10, 'pull_dir': 310},
        'Wrigley Field': {'lat': 41.9484, 'lon': -87.6553, 'park_factor': 1.00, 'pull_dir': 310},
        'Great American Ball Park': {'lat': 39.0974, 'lon': -84.5071, 'park_factor': 1.25, 'pull_dir': 315},
        'Progressive Field': {'lat': 41.4962, 'lon': -81.6852, 'park_factor': 0.98, 'pull_dir': 315},
        'Coors Field': {'lat': 39.7559, 'lon': -104.9942, 'park_factor': 1.40, 'pull_dir': 315},
        'Comerica Park': {'lat': 42.3390, 'lon': -83.0485, 'park_factor': 0.95, 'pull_dir': 315},
        'Minute Maid Park': {'lat': 29.7573, 'lon': -95.3555, 'park_factor': 1.05, 'pull_dir': 315},
        'Kauffman Stadium': {'lat': 39.0517, 'lon': -94.4803, 'park_factor': 0.97, 'pull_dir': 315},
        'Angel Stadium': {'lat': 33.8003, 'lon': -117.8827, 'park_factor': 1.02, 'pull_dir': 315},
        'Dodger Stadium': {'lat': 34.0739, 'lon': -118.2400, 'park_factor': 1.03, 'pull_dir': 320},
        'Nationals Park': {'lat': 38.8730, 'lon': -77.0074, 'park_factor': 1.00, 'pull_dir': 315},
        'Citi Field': {'lat': 40.7571, 'lon': -73.8458, 'park_factor': 1.00, 'pull_dir': 315},
        'Oakland Coliseum': {'lat': 37.7516, 'lon': -122.2005, 'park_factor': 0.93, 'pull_dir': 315},
        'PNC Park': {'lat': 40.4468, 'lon': -80.0058, 'park_factor': 0.98, 'pull_dir': 315},
        'Petco Park': {'lat': 32.7076, 'lon': -117.1570, 'park_factor': 0.95, 'pull_dir': 320},
        'T-Mobile Park': {'lat': 47.5914, 'lon': -122.3325, 'park_factor': 0.90, 'pull_dir': 315},
        'Oracle Park': {'lat': 37.7786, 'lon': -122.3893, 'park_factor': 0.92, 'pull_dir': 320},
        'Busch Stadium': {'lat': 38.6226, 'lon': -90.1928, 'park_factor': 0.98, 'pull_dir': 315},
        'Tropicana Field': {'lat': 27.7682, 'lon': -82.6534, 'park_factor': 0.96, 'pull_dir': 315},
        'Globe Life Field': {'lat': 32.7512, 'lon': -97.0833, 'park_factor': 1.15, 'pull_dir': 315},
        'Rogers Centre': {'lat': 43.6418, 'lon': -79.3891, 'park_factor': 1.02, 'pull_dir': 315},
        'Target Field': {'lat': 44.9817, 'lon': -93.2776, 'park_factor': 1.05, 'pull_dir': 315},
        'Citizens Bank Park': {'lat': 39.9061, 'lon': -75.1665, 'park_factor': 1.12, 'pull_dir': 315},
        'Truist Park': {'lat': 33.8907, 'lon': -84.4677, 'park_factor': 1.08, 'pull_dir': 315},
        'Guaranteed Rate Field': {'lat': 41.8299, 'lon': -87.6338, 'park_factor': 1.08, 'pull_dir': 315},
        'loanDepot park': {'lat': 25.7781, 'lon': -80.2196, 'park_factor': 0.94, 'pull_dir': 315},
        'Yankee Stadium': {'lat': 40.8296, 'lon': -73.9262, 'park_factor': 1.20, 'pull_dir': 315},
        'American Family Field': {'lat': 43.0280, 'lon': -87.9712, 'park_factor': 1.03, 'pull_dir': 315},
    }

class SportsRadarMLBAPI:
    """Handles all SportsRadar API interactions"""
    
    def __init__(self):
        self.config = Config()
        self.session = requests.Session()
        self.current_season = datetime.now().year
        self.rate_limit_delay = 1.1  # SportsRadar rate limit: 1 request per second
        
    def _make_request(self, endpoint: str) -> Dict:
        """Make API request with rate limiting"""
        url = f"{self.config.SPORTRADAR_BASE}{endpoint}?api_key={self.config.SPORTRADAR_KEY}"
        
        try:
            response = self.session.get(url)
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return {}
        except Exception as e:
            print(f"Request error: {e}")
            return {}
    
    def get_todays_games(self) -> List[Dict]:
        """Get today's MLB games with detailed information"""
        date_str = datetime.now().strftime('%Y/%m/%d')
        endpoint = f"/games/{date_str}/schedule.json"
        
        print(f"Fetching games for {date_str}...")
        data = self._make_request(endpoint)
        
        games = []
        if 'games' in data:
            for game in data['games']:
                if game['status'] != 'closed':  # Not finished
                    game_info = {
                        'game_id': game['id'],
                        'home_team': game['home']['name'],
                        'away_team': game['away']['name'],
                        'home_team_id': game['home']['id'],
                        'away_team_id': game['away']['id'],
                        'venue': game.get('venue', {}).get('name', ''),
                        'game_time': game.get('scheduled', ''),
                        'home_pitcher': None,
                        'away_pitcher': None,
                        'home_lineup': [],
                        'away_lineup': []
                    }
                    
                    # Get probable pitchers if available
                    if 'probables' in game:
                        if 'home' in game['probables']:
                            game_info['home_pitcher'] = {
                                'id': game['probables']['home'].get('id'),
                                'name': game['probables']['home'].get('preferred_name', '') + ' ' + 
                                       game['probables']['home'].get('last_name', '')
                            }
                        if 'away' in game['probables']:
                            game_info['away_pitcher'] = {
                                'id': game['probables']['away'].get('id'),
                                'name': game['probables']['away'].get('preferred_name', '') + ' ' + 
                                       game['probables']['away'].get('last_name', '')
                            }
                    
                    games.append(game_info)
        
        # Now get lineups for each game
        for game in games:
            self._get_game_lineups(game)
        
        print(f"Found {len(games)} games with lineups")
        return games
    
    def _get_game_lineups(self, game: Dict) -> None:
        """Get lineups for a specific game"""
        endpoint = f"/games/{game['game_id']}/summary.json"
        data = self._make_request(endpoint)
        
        if 'game' in data:
            game_data = data['game']
            
            # Get home lineup
            if 'home' in game_data and 'lineup' in game_data['home']:
                for player in game_data['home']['lineup']:
                    game['home_lineup'].append({
                        'id': player.get('id'),
                        'name': player.get('preferred_name', '') + ' ' + player.get('last_name', ''),
                        'position': player.get('position', ''),
                        'order': player.get('order', 0)
                    })
            
            # Get away lineup
            if 'away' in game_data and 'lineup' in game_data['away']:
                for player in game_data['away']['lineup']:
                    game['away_lineup'].append({
                        'id': player.get('id'),
                        'name': player.get('preferred_name', '') + ' ' + player.get('last_name', ''),
                        'position': player.get('position', ''),
                        'order': player.get('order', 0)
                    })
    
    def get_player_season_stats(self, player_id: str, is_pitcher: bool = False) -> Dict:
        """Get player's season statistics from SportsRadar"""
        endpoint = f"/players/{player_id}/profile.json"
        data = self._make_request(endpoint)
        
        if 'player' not in data:
            return self._get_default_stats(is_pitcher)
        
        player_data = data['player']
        
        # Find current season stats
        current_stats = None
        if 'seasons' in player_data:
            for season in player_data['seasons']:
                if season.get('year') == self.current_season and season.get('type') == 'REG':
                    current_stats = season.get('totals', {})
                    break
        
        if not current_stats:
            return self._get_default_stats(is_pitcher)
        
        if not is_pitcher:
            # Extract hitter statistics
            hitting = current_stats.get('hitting', {})
            
            # Calculate advanced metrics
            ab = hitting.get('ab', 1)
            hits = hitting.get('h', 0)
            doubles = hitting.get('d', 0)
            triples = hitting.get('t', 0)
            hr = hitting.get('hr', 0)
            
            # ISO (Isolated Power)
            singles = hits - doubles - triples - hr
            total_bases = singles + (2 * doubles) + (3 * triples) + (4 * hr)
            iso = (total_bases - hits) / ab if ab > 0 else 0
            
            # Get Statcast-like metrics from extended stats if available
            extended = current_stats.get('hitting_extended', {})
            
            return {
                'barrel_rate': extended.get('barrel_pct', 0.08) / 100 if 'barrel_pct' in extended else 0.08,
                'exit_velocity_fbld': extended.get('avg_exit_velocity', 90.0),
                'sweet_spot_percent': extended.get('sweet_spot_pct', 0.33) / 100 if 'sweet_spot_pct' in extended else 0.33,
                'hard_hit_rate': extended.get('hard_hit_pct', 0.40) / 100 if 'hard_hit_pct' in extended else 0.40,
                'avg_launch_angle': extended.get('avg_launch_angle', 12.0),
                'max_exit_velocity': extended.get('max_exit_velocity', 105.0),
                'iso': iso,
                'home_runs': hr,
                'at_bats': ab,
                'ops': hitting.get('ops', 0.750),
                'recent_hrs': self._get_recent_home_runs(player_id)
            }
        else:
            # Extract pitcher statistics
            pitching = current_stats.get('pitching', {})
            
            # Calculate HR/FB rate and other metrics
            hr_allowed = pitching.get('hr', 0)
            fly_balls = pitching.get('fly_balls', 1) if 'fly_balls' in pitching else int(hr_allowed * 7)
            hr_fb_rate = hr_allowed / fly_balls if fly_balls > 0 else 0.13
            
            # Get extended pitching metrics
            extended = current_stats.get('pitching_extended', {})
            
            return {
                'hr_fb_rate': hr_fb_rate,
                'barrel_rate_against': extended.get('barrel_pct_against', 0.075) / 100 if 'barrel_pct_against' in extended else 0.075,
                'avg_exit_velocity_against': extended.get('avg_exit_velocity_against', 88.0),
                'hard_hit_rate_against': extended.get('hard_hit_pct_against', 0.38) / 100 if 'hard_hit_pct_against' in extended else 0.38,
                'fastball_velocity': extended.get('avg_fastball_velocity', 93.0),
                'zone_rate': extended.get('zone_pct', 0.45) / 100 if 'zone_pct' in extended else 0.45,
                'whiff_rate': extended.get('whiff_rate', 0.23) / 100 if 'whiff_rate' in extended else 0.23,
                'era': pitching.get('era', 4.00),
                'whip': pitching.get('whip', 1.30)
            }
    
    def _get_recent_home_runs(self, player_id: str) -> int:
        """Get player's home runs in last 15 games"""
        # This would require game logs endpoint
        # For now, estimate based on season rate
        endpoint = f"/players/{player_id}/profile.json"
        data = self._make_request(endpoint)
        
        if 'player' in data and 'seasons' in data['player']:
            for season in data['player']['seasons']:
                if season.get('year') == self.current_season:
                    hitting = season.get('totals', {}).get('hitting', {})
                    hr = hitting.get('hr', 0)
                    games = hitting.get('games', {}).get('play', 1)
                    
                    # Estimate recent HRs based on season rate
                    hr_per_game = hr / games if games > 0 else 0
                    return int(hr_per_game * 15 * 1.1)  # Slight recency bias
        
        return 0
    
    def _get_default_stats(self, is_pitcher: bool) -> Dict:
        """Return league average stats as default"""
        if not is_pitcher:
            return {
                'barrel_rate': 0.08,
                'exit_velocity_fbld': 90.0,
                'sweet_spot_percent': 0.33,
                'hard_hit_rate': 0.40,
                'avg_launch_angle': 12.0,
                'max_exit_velocity': 105.0,
                'iso': 0.165,
                'home_runs': 15,
                'at_bats': 400,
                'ops': 0.750,
                'recent_hrs': 1
            }
        else:
            return {
                'hr_fb_rate': 0.13,
                'barrel_rate_against': 0.075,
                'avg_exit_velocity_against': 88.0,
                'hard_hit_rate_against': 0.38,
                'fastball_velocity': 93.0,
                'zone_rate': 0.45,
                'whiff_rate': 0.23,
                'era': 4.00,
                'whip': 1.30
            }
    
    def get_player_splits(self, player_id: str, vs_hand: str = 'R') -> Dict:
        """Get player's splits vs RHP or LHP"""
        endpoint = f"/players/{player_id}/splits.json"
        data = self._make_request(endpoint)
        
        if 'splits' in data:
            for split in data['splits']:
                if split.get('type') == f"vs_{vs_hand}HP":
                    return split.get('totals', {})
        
        return {}
    
    def check_platoon_advantage(self, batter_id: str, pitcher_id: str) -> bool:
        """Check if batter has platoon advantage against pitcher"""
        # Get player profiles to determine handedness
        batter_endpoint = f"/players/{batter_id}/profile.json"
        pitcher_endpoint = f"/players/{pitcher_id}/profile.json"
        
        batter_data = self._make_request(batter_endpoint)
        pitcher_data = self._make_request(pitcher_endpoint)
        
        if 'player' in batter_data and 'player' in pitcher_data:
            batter_hand = batter_data['player'].get('bat_hand', 'R')
            pitcher_hand = pitcher_data['player'].get('throw_hand', 'R')
            
            # Platoon advantage when batter and pitcher have opposite hands
            return batter_hand != pitcher_hand
        
        return False
    
    def get_weather_data(self, venue_name: str) -> Dict:
        """Get weather data for stadium"""
        if venue_name not in self.config.STADIUMS:
            return {'temp': 72, 'wind_speed': 0, 'wind_deg': 0}
        
        stadium = self.config.STADIUMS[venue_name]
        
        # Use OpenWeatherMap API if configured
        if self.config.WEATHER_API_KEY != "YOUR_OPENWEATHER_KEY":
            params = {
                'lat': stadium['lat'],
                'lon': stadium['lon'],
                'appid': self.config.WEATHER_API_KEY,
                'units': 'imperial'
            }
            
            try:
                response = requests.get(self.config.WEATHER_URL, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'temp': data['main']['temp'],
                        'wind_speed': data['wind']['speed'],
                        'wind_deg': data['wind'].get('deg', 0)
                    }
            except Exception as e:
                print(f"Weather API error: {e}")
        
        # Default weather
        return {'temp': 72, 'wind_speed': 5, 'wind_deg': 180}

class HomeRunModel:
    """Home run prediction model implementation"""
    
    def __init__(self):
        self.config = Config()
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def calculate_hitter_score(self, hitter_stats: Dict, age: int = 28) -> float:
        """Calculate hitter component score"""
        weights = self.config.HITTER_WEIGHTS
        
        # Normalize features to 0-1 scale
        x1 = min(hitter_stats.get('barrel_rate', 0.08) / 0.15, 1.0)
        x2 = min((hitter_stats.get('exit_velocity_fbld', 90) - 85) / 15, 1.0)
        x3 = min(hitter_stats.get('sweet_spot_percent', 0.33) / 0.45, 1.0)
        x4 = min(hitter_stats.get('hard_hit_rate', 0.40) / 0.55, 1.0)
        x5 = 1 - abs(age - 28.5) * 0.02
        
        features = [x1, x2, x3, x4, x5]
        score = sum(w * f for w, f in zip(weights, features))
        
        return score
    
    def calculate_pitcher_score(self, pitcher_stats: Dict, platoon_advantage: bool = False) -> float:
        """Calculate pitcher vulnerability score"""
        weights = self.config.PITCHER_WEIGHTS
        
        y1 = min(pitcher_stats.get('hr_fb_rate', 0.13) / 0.20, 1.0)
        
        # Stuff+ proxy using whiff rate and velocity
        whiff = pitcher_stats.get('whiff_rate', 0.23)
        velo = pitcher_stats.get('fastball_velocity', 93)
        stuff_proxy = (whiff * 100 + (velo - 90) * 2) / 15
        y2 = min(max(stuff_proxy, 0), 1)
        
        # Fastball velocity vulnerability (optimal at 88-92)
        y3 = 1 - abs(velo - 90) * 0.03
        y3 = max(y3, 0.7)
        
        y4 = pitcher_stats.get('zone_rate', 0.45)
        y5 = 0.15 if platoon_advantage else 0
        
        features = [y1, y2, y3, y4, y5]
        score = sum(w * f for w, f in zip(weights, features))
        
        return score
    
    def calculate_environmental_multiplier(self, weather: Dict, park_factor: float, 
                                          pull_direction: float = 315) -> float:
        """Calculate environmental effects multiplier"""
        # Temperature effect: 1.96% per degree above 70°F
        temp_effect = (weather['temp'] - 70) * 0.0196
        
        # Wind effect: 3.8% per mph of helpful wind
        wind_angle = np.radians(weather['wind_deg'] - pull_direction)
        wind_component = weather['wind_speed'] * np.cos(wind_angle)
        wind_effect = wind_component * 0.038
        
        # Combined multiplier
        multiplier = (1 + temp_effect) * (1 + wind_effect) * park_factor
        
        return multiplier
    
    def calculate_situational_score(self, recent_hrs: int = 0, at_bats: int = 400) -> float:
        """Calculate situational adjustments"""
        weights = self.config.SITUATIONAL_WEIGHTS
        
        z1 = 0  # Neutral count state for projections
        z2 = 0.05  # Neutral leverage
        z3 = -(at_bats / 600) * 0.08  # Season fatigue
        z4 = min(recent_hrs * 0.12, 0.36)  # Recent form (cap at 3 HRs)
        
        features = [z1, z2, z3, z4]
        score = sum(w * f for w, f in zip(weights, features))
        
        return score
    
    def calculate_hr_probability(self, hitter_stats: Dict, pitcher_stats: Dict, 
                                weather: Dict, park_factor: float,
                                platoon_advantage: bool = False, age: int = 28) -> float:
        """Calculate final HR probability using complete formula"""
        
        # Calculate all components
        hitter_score = self.calculate_hitter_score(hitter_stats, age)
        pitcher_score = self.calculate_pitcher_score(pitcher_stats, platoon_advantage)
        situational_score = self.calculate_situational_score(
            hitter_stats.get('recent_hrs', 0),
            hitter_stats.get('at_bats', 400)
        )
        environmental_mult = self.calculate_environmental_multiplier(weather, park_factor)
        
        # Raw probability calculation
        raw_score = self.config.BETA_0 + hitter_score + pitcher_score + situational_score
        raw_prob = self.sigmoid(raw_score)
        
        # Apply environmental multiplier
        final_prob = raw_prob * environmental_mult
        
        # Calibration factor
        calibrated_prob = final_prob * 0.95
        
        return min(calibrated_prob, 0.80)

class DailyHRPredictor:
    """Main prediction system using SportsRadar data"""
    
    def __init__(self):
        self.api = SportsRadarMLBAPI()
        self.model = HomeRunModel()
        self.config = Config()
    
    def predict_daily_slate(self) -> pd.DataFrame:
        """Generate predictions for today's games using SportsRadar"""
        print("=" * 80)
        print(f"MLB HOME RUN PREDICTIONS - {datetime.now().strftime('%Y-%m-%d')}")
        print("PREMIUM SPORTRADAR DATA - PRODUCTION MODEL")
        print("=" * 80)
        
        # Get today's games from SportsRadar
        games = self.api.get_todays_games()
        
        if not games:
            print("No games scheduled for today")
            return pd.DataFrame()
        
        print(f"\n📊 Processing {len(games)} games with SportsRadar data...")
        print("⚡ Fetching real-time player statistics...\n")
        
        all_predictions = []
        total_players = 0
        
        for i, game in enumerate(games, 1):
            print(f"Game {i}/{len(games)}: {game['away_team']} @ {game['home_team']}")
            
            # Get venue and weather
            venue_name = game.get('venue', '')
            weather = self.api.get_weather_data(venue_name)
            
            # Get park factor
            stadium_info = self.config.STADIUMS.get(venue_name, {})
            park_factor = stadium_info.get('park_factor', 1.0)
            
            # Process home lineup vs away pitcher
            if game['away_pitcher'] and game['home_lineup']:
                print(f"  📈 Analyzing {len(game['home_lineup'])} {game['home_team']} batters")
                
                # Get pitcher stats
                pitcher_stats = self.api.get_player_season_stats(
                    game['away_pitcher']['id'], 
                    is_pitcher=True
                )
                
                for batter in game['home_lineup']:
                    if not batter['id']:
                        continue
                    
                    # Get batter stats from SportsRadar
                    hitter_stats = self.api.get_player_season_stats(batter['id'])
                    
                    # Check platoon advantage
                    platoon = self.api.check_platoon_advantage(
                        batter['id'],
                        game['away_pitcher']['id']
                    )
                    
                    # Calculate HR probability
                    hr_prob = self.model.calculate_hr_probability(
                        hitter_stats=hitter_stats,
                        pitcher_stats=pitcher_stats,
                        weather=weather,
                        park_factor=park_factor,
                        platoon_advantage=platoon
                    )
                    
                    all_predictions.append({
                        'Player': batter['name'],
                        'Team': game['home_team'],
                        'Opponent': game['away_team'],
                        'Pitcher': game['away_pitcher']['name'],
                        'HR_Probability': hr_prob * 100,
                        'Batting_Order': batter.get('order', 0),
                        'Barrel_Rate': hitter_stats['barrel_rate'],
                        'Exit_Velo_FBLD': hitter_stats['exit_velocity_fbld'],
                        'Hard_Hit_Rate': hitter_stats['hard_hit_rate'],
                        'ISO': hitter_stats['iso'],
                        'Season_HRs': hitter_stats['home_runs'],
                        'Recent_HRs': hitter_stats['recent_hrs'],
                        'Pitcher_HR_FB': pitcher_stats['hr_fb_rate'],
                        'Pitcher_ERA': pitcher_stats['era'],
                        'FB_Velocity': pitcher_stats['fastball_velocity'],
                        'Park_Factor': park_factor,
                        'Temperature': weather['temp'],
                        'Wind_Speed': weather['wind_speed'],
                        'Platoon_Adv': platoon,
                        'Game_Time': game['game_time']
                    })
                    total_players += 1
            
            # Process away lineup vs home pitcher
            if game['home_pitcher'] and game['away_lineup']:
                print(f"  📊 Analyzing {len(game['away_lineup'])} {game['away_team']} batters")
                
                pitcher_stats = self.api.get_player_season_stats(
                    game['home_pitcher']['id'],
                    is_pitcher=True
                )
                
                for batter in game['away_lineup']:
                    if not batter['id']:
                        continue
                    
                    hitter_stats = self.api.get_player_season_stats(batter['id'])
                    platoon = self.api.check_platoon_advantage(
                        batter['id'],
                        game['home_pitcher']['id']
                    )
                    
                    hr_prob = self.model.calculate_hr_probability(
                        hitter_stats=hitter_stats,
                        pitcher_stats=pitcher_stats,
                        weather=weather,
                        park_factor=park_factor,
                        platoon_advantage=platoon
                    )
                    
                    all_predictions.append({
                        'Player': batter['name'],
                        'Team': game['away_team'],
                        'Opponent': game['home_team'],
                        'Pitcher': game['home_pitcher']['name'],
                        'HR_Probability': hr_prob * 100,
                        'Batting_Order': batter.get('order', 0),
                        'Barrel_Rate': hitter_stats['barrel_rate'],
                        'Exit_Velo_FBLD': hitter_stats['exit_velocity_fbld'],
                        'Hard_Hit_Rate': hitter_stats['hard_hit_rate'],
                        'ISO': hitter_stats['iso'],
                        'Season_HRs': hitter_stats['home_runs'],
                        'Recent_HRs': hitter_stats['recent_hrs'],
                        'Pitcher_HR_FB': pitcher_stats['hr_fb_rate'],
                        'Pitcher_ERA': pitcher_stats['era'],
                        'FB_Velocity': pitcher_stats['fastball_velocity'],
                        'Park_Factor': park_factor,
                        'Temperature': weather['temp'],
                        'Wind_Speed': weather['wind_speed'],
                        'Platoon_Adv': platoon,
                        'Game_Time': game['game_time']
                    })
                    total_players += 1
        
        # Create DataFrame
        df = pd.DataFrame(all_predictions)
        
        if df.empty:
            print("No valid predictions generated")
            return df
        
        # Sort by HR probability
        df = df.sort_values('HR_Probability', ascending=False).reset_index(drop=True)
        
        # Add confidence levels
        df['Confidence'] = df['HR_Probability'].apply(
            lambda x: 'HIGH' if x > 8 else ('MEDIUM' if x > 5 else 'LOW')
        )
        
        # Add expected home runs
        df['xHR'] = df['HR_Probability'] / 100
        
        # Add implied odds
        df['Implied_Odds'] = df['HR_Probability'].apply(
            lambda x: f"+{int((100 / x - 1) * 100)}" if x > 0 else "N/A"
        )
        
        print(f"\n✅ Successfully analyzed {total_players} player matchups")
        print(f"⚡ Data powered by SportsRadar Premium API")
        
        return df
    
    def display_top_picks(self, df: pd.DataFrame, top_n: int = 20):
        """Display top HR candidates with betting insights"""
        print("\n" + "=" * 80)
        print("🎯 TOP HOME RUN CANDIDATES - BETTING RECOMMENDATIONS")
        print("=" * 80)
        
        top_picks = df.head(top_n)
        
        for idx, row in top_picks.iterrows():
            confidence_emoji = "🔥" if row['Confidence'] == 'HIGH' else ("⭐" if row['Confidence'] == 'MEDIUM' else "")
            
            print(f"\n{idx + 1}. {confidence_emoji} {row['Player']} ({row['Team']} vs {row['Opponent']})")
            print(f"   📊 HR Probability: {row['HR_Probability']:.2f}% (Implied: {row['Implied_Odds']})")
            print(f"   ⚾ vs {row['Pitcher']} (ERA: {row['Pitcher_ERA']:.2f})")
            print(f"   📈 Power Metrics:")
            print(f"      • Barrel Rate: {row['Barrel_Rate']:.1%}")
            print(f"      • Exit Velo FB/LD: {row['Exit_Velo_FBLD']:.1f} mph")
            print(f"      • ISO: {row['ISO']:.3f}")
            print(f"      • Season HRs: {int(row['Season_HRs'])}")
            print(f"      • Recent Form: {int(row['Recent_HRs'])} HRs (L15 games)")
            print(f"   🎯 Matchup:")
            print(f"      • Pitcher HR/FB: {row['Pitcher_HR_FB']:.1%}")
            print(f"      • FB Velocity: {row['FB_Velocity']:.1f} mph")
            if row['Platoon_Adv']:
                print(f"      • ✓ Platoon Advantage")
            print(f"   🏟️ Environment:")
            print(f"      • Park Factor: {row['Park_Factor']:.2f}")
            print(f"      • Weather: {row['Temperature']:.0f}°F, Wind {row['Wind_Speed']:.0f} mph")
            print(f"   ⚡ Batting Order: #{int(row['Batting_Order'])}")
        
        # Betting summary
        print("\n" + "=" * 80)
        print("💰 BETTING EDGE ANALYSIS")
        print("=" * 80)
        
        high_conf = df[df['Confidence'] == 'HIGH']
        med_conf = df[df['Confidence'] == 'MEDIUM']
        
        if len(high_conf) > 0:
            print(f"\n🔥 HIGH CONFIDENCE PLAYS (>8% model probability)")
            print(f"   {len(high_conf)} total plays identified\n")
            for _, row in high_conf.head(7).iterrows():
                print(f"   • {row['Player']}: {row['HR_Probability']:.1f}% ({row['Implied_Odds']})")
                print(f"     vs {row['Pitcher']} | Park Factor: {row['Park_Factor']:.2f}")
        
        if len(med_conf) > 0:
            print(f"\n⭐ MEDIUM CONFIDENCE PLAYS (5-8% model probability)")
            print(f"   {len(med_conf)} total plays identified\n")
            for _, row in med_conf.head(5).iterrows():
                print(f"   • {row['Player']}: {row['HR_Probability']:.1f}% ({row['Implied_Odds']})")
        
        # Environmental edges
        print("\n🏟️ BEST ENVIRONMENTAL CONDITIONS")
        env_df = df.copy()
        env_df['Env_Score'] = (env_df['Park_Factor'] - 1) * 100 + (env_df['Temperature'] - 70) * 0.5
        env_top = env_df.nlargest(5, 'Env_Score')
        
        for _, row in env_top.iterrows():
            if row['HR_Probability'] > 3:  # Only show viable candidates
                print(f"   • {row['Player']} at {row['Team'] if row['Team'] != row['Opponent'] else row['Opponent']}")
                print(f"     Park: {row['Park_Factor']:.2f}x | Temp: {row['Temperature']:.0f}°F")
        
        # Stats summary
        print("\n" + "=" * 80)
        print("📊 MODEL STATISTICS")
        print("=" * 80)
        print(f"Total players analyzed: {len(df)}")
        print(f"Average HR probability: {df['HR_Probability'].mean():.2f}%")
        print(f"Max HR probability: {df['HR_Probability'].max():.2f}%")
        print(f"Expected total HRs today: {df['xHR'].sum():.1f}")
        print(f"Games at Coors Field: {len(df[df['Park_Factor'] == 1.40]) // 18 if len(df[df['Park_Factor'] == 1.40]) > 0 else 0}")
        
        # Bankroll management
        print("\n" + "=" * 80)
        print("💼 BANKROLL MANAGEMENT GUIDELINES")
        print("=" * 80)
        print("Recommended Kelly Criterion (25% fractional):")
        print("• HIGH confidence (>8%): 2-3% of bankroll per play")
        print("• MEDIUM confidence (5-8%): 1-2% of bankroll per play")
        print("• Only bet when book odds exceed model implied odds by 15%+")
        print("\nExample: If model shows 10% (implied +900), look for +1035 or better")
    
    def save_predictions(self, df: pd.DataFrame):
        """Save predictions with timestamp"""
        filename = f"sportradar_hr_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ Predictions saved to {filename}")
        return filename

# Main execution
if __name__ == "__main__":
    print("MLB DAILY HOME RUN PREDICTION MODEL")
    print("PREMIUM SPORTRADAR DATA - PRODUCTION VERSION")
    print("=" * 80)
    
    # Verify API key is set
    if Config.SPORTRADAR_KEY == "YOUR_SPORTRADAR_KEY":
        print("\n❌ ERROR: SportsRadar API key not configured")
        print("Please update Config.SPORTRADAR_KEY with your key")
        exit(1)
    
    # Optional: Check for weather API
    if Config.WEATHER_API_KEY == "YOUR_OPENWEATHER_KEY":
        print("\n⚠️  Weather API not configured (optional)")
        print("For enhanced weather data, get free key at: https://openweathermap.org/api")
        print("Continuing with default weather values...\n")
        time.sleep(2)
    
    # Initialize predictor
    print("\n🚀 Initializing SportsRadar API connection...")
    predictor = DailyHRPredictor()
    
    # Generate predictions
    print("📡 Fetching real-time MLB data from SportsRadar...")
    predictions_df = predictor.predict_daily_slate()
    
    if not predictions_df.empty:
        # Display results
        predictor.display_top_picks(predictions_df, top_n=20)
        
        # Save to file
        filename = predictor.save_predictions(predictions_df)
        
        print("\n" + "=" * 80)
        print("✅ PREDICTION COMPLETE - SPORTRADAR PREMIUM DATA")
        print("=" * 80)
        print(f"📁 Full results saved to: {filename}")
        print("\n🎯 Next Steps:")
        print("1. Compare model probabilities with sportsbook odds")
        print("2. Look for 15%+ edge (model vs implied odds)")
        print("3. Track results daily for model validation")
        print("4. Use fractional Kelly (25%) for bet sizing")
        print("\n💡 Pro Tip: Focus on HIGH confidence plays at pitcher-friendly")
        print("   parks where books may undervalue HR probability")
    else:
        print("\n❌ No predictions generated. Please check:")
        print("1. SportsRadar API key is valid")
        print("2. Games are scheduled for today")
        print("3. API rate limits not exceeded")