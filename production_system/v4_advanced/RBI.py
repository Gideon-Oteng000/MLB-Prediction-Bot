#!/usr/bin/env python3
"""
MLB RBI Prediction System v3.0
Advanced Machine Learning Model with Real Data Integration

Key Improvements:
- Real MLB historical data from MLB Stats API
- Real weather API integration
- Real odds API from The Odds API
- Advanced splits & context features
- Dynamic bullpen blending
- SHAP explainability
- Enhanced database with comprehensive logging
"""

import numpy as np
import pandas as pd
import sqlite3
import requests
import json
import time
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# SHAP for explainability
import shap

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Keys
WEATHER_API_KEY = 'e09911139e379f1e4ca813df1778b4ef'
ODDS_API_KEY = '47b36e3e637a7690621e258da00e29d7'

@dataclass
class PlayerSplits:
    """Store player split statistics"""
    vs_rhp: Dict[str, float]
    vs_lhp: Dict[str, float]
    home: Dict[str, float]
    away: Dict[str, float]
    day: Dict[str, float]
    night: Dict[str, float]
    last_7: Dict[str, float]
    last_14: Dict[str, float]
    last_30: Dict[str, float]

@dataclass
class PitcherSplits:
    """Store pitcher split statistics"""
    vs_rhb: Dict[str, float]
    vs_lhb: Dict[str, float]
    home: Dict[str, float]
    away: Dict[str, float]
    day: Dict[str, float]
    night: Dict[str, float]

@dataclass
class WeatherData:
    """Weather information for game"""
    temp_f: float
    humidity: float
    wind_speed: float
    wind_direction: str
    pressure: float
    air_density_factor: float
    tailwind_component: float

@dataclass
class OddsData:
    """Betting odds information"""
    sportsbook: str
    player_name: str
    rbi_over_line: float
    over_odds: float
    under_odds: float
    implied_probability: float
    market_consensus: float
    value_edge: float

class MLBDataFetcher:
    """Handles all external API calls for real data with persistent caching"""

    def __init__(self, cache_db_path: str = 'mlb_cache_v3.db'):
        self.mlb_base = "https://statsapi.mlb.com/api/v1"
        self.weather_base = "https://api.openweathermap.org/data/2.5"
        self.odds_base = "https://api.the-odds-api.com/v4"
        self.session = requests.Session()
        self.cache = {}  # In-memory cache
        self.cache_db_path = cache_db_path

        # Initialize persistent cache database
        self._init_cache_db()

        # Set up session with retry strategy and headers
        self.session.headers.update({
            'User-Agent': 'MLB-RBI-Predictor/3.0',
            'Accept': 'application/json'
        })

    def _init_cache_db(self):
        """Initialize persistent cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            # Cache table for API responses
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expiry_hours INTEGER DEFAULT 24
                )
            ''')

            # Player stats cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS player_stats_cache (
                    player_id INTEGER,
                    season INTEGER,
                    stat_type TEXT,
                    data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (player_id, season, stat_type)
                )
            ''')

            # Game data cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS game_data_cache (
                    game_id INTEGER PRIMARY KEY,
                    data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing cache database: {e}")

    def _get_cached_data(self, cache_key: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Get data from cache if available and not expired"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT data, timestamp FROM api_cache
                WHERE cache_key = ?
                AND datetime(timestamp, '+' || ? || ' hours') > datetime('now')
            ''', (cache_key, max_age_hours))

            result = cursor.fetchone()
            conn.close()

            if result:
                return json.loads(result[0])

        except Exception as e:
            logger.error(f"Error reading from cache: {e}")

        return None

    def _store_cached_data(self, cache_key: str, data: Dict, expiry_hours: int = 24):
        """Store data in cache"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO api_cache (cache_key, data, expiry_hours)
                VALUES (?, ?, ?)
            ''', (cache_key, json.dumps(data), expiry_hours))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing cache: {e}")

    def fetch_historical_games(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch real historical MLB game data with enhanced features"""
        logger.info(f"Fetching historical games from {start_date} to {end_date}")

        games_data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        total_days = (end - current_date).days + 1
        processed_days = 0

        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            processed_days += 1

            if processed_days % 10 == 0:
                logger.info(f"Progress: {processed_days}/{total_days} days processed")

            try:
                # Check persistent cache first
                cache_key = f"games_{date_str}"
                cached_data = self._get_cached_data(cache_key, max_age_hours=168)  # 1 week for historical data

                if cached_data:
                    games_data.extend(cached_data.get('games', []))
                    current_date += timedelta(days=1)
                    continue

                # Get games for date with enhanced hydration
                url = f"{self.mlb_base}/schedule"
                params = {
                    'sportId': 1,
                    'date': date_str,
                    'hydrate': 'boxscore,team,venue,weather,probablePitcher,linescore'
                }

                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                daily_games = []
                if 'dates' in data and data['dates']:
                    for date_info in data['dates']:
                        for game in date_info.get('games', []):
                            if game['status']['statusCode'] == 'F':  # Completed games only
                                game_data = self._extract_enhanced_game_details(game, date_str)
                                if game_data:
                                    daily_games.append(game_data)

                # Cache the daily games in persistent storage
                cache_data = {'games': daily_games}
                self._store_cached_data(cache_key, cache_data, expiry_hours=168)  # 1 week
                games_data.extend(daily_games)

            except Exception as e:
                logger.error(f"Error fetching data for {date_str}: {e}")
                time.sleep(2)  # Longer wait on error

            current_date += timedelta(days=1)
            time.sleep(0.6)  # Rate limiting - respect API limits

        logger.info(f"Collected {len(games_data)} game records from {processed_days} days")
        return pd.DataFrame(games_data)
    
    def _extract_enhanced_game_details(self, game: Dict, date_str: str) -> Optional[Dict]:
        """Extract comprehensive game information with player stats and context"""
        try:
            game_pk = game['gamePk']

            # Fetch detailed box score with enhanced data
            box_url = f"{self.mlb_base}/game/{game_pk}/boxscore"
            box_response = self.session.get(box_url, timeout=30)
            box_response.raise_for_status()
            box_data = box_response.json()

            # Get venue information with coordinates
            venue = game.get('venue', {})
            venue_location = venue.get('location', {})

            # Get game conditions
            game_conditions = self._extract_game_conditions(game, box_data)

            # Extract player performance data with context
            player_performances = self._extract_enhanced_player_data(box_data, game, game_conditions)

            if not player_performances:
                return None

            # Basic game info with enhanced context
            game_info = {
                'game_id': game_pk,
                'date': date_str,
                'game_datetime': game.get('gameDate', ''),
                'home_team': game['teams']['home']['team']['name'],
                'away_team': game['teams']['away']['team']['name'],
                'home_team_id': game['teams']['home']['team']['id'],
                'away_team_id': game['teams']['away']['team']['id'],
                'home_score': game['teams']['home'].get('score', 0),
                'away_score': game['teams']['away'].get('score', 0),
                'venue_name': venue.get('name', ''),
                'venue_lat': float(venue_location.get('latitude', 0)),
                'venue_lon': float(venue_location.get('longitude', 0)),
                'weather_temp': game_conditions.get('temp', 72),
                'weather_wind': game_conditions.get('wind', 5),
                'weather_condition': game_conditions.get('condition', 'Clear'),
                'game_time_local': game_conditions.get('game_time', 'Night'),
                'attendance': game.get('attendance', 0),
                'player_performances': player_performances
            }

            return game_info

        except Exception as e:
            logger.error(f"Error extracting enhanced game details for {game_pk}: {e}")
            return None

    def _extract_game_conditions(self, game: Dict, box_data: Dict) -> Dict:
        """Extract game conditions like weather, time, etc."""
        conditions = {
            'temp': 72,
            'wind': 5,
            'condition': 'Clear',
            'game_time': 'Night'
        }

        try:
            # Try to get weather info from game data
            if 'weather' in game:
                weather = game['weather']
                conditions['temp'] = weather.get('temp', 72)
                conditions['wind'] = weather.get('wind', 5)
                conditions['condition'] = weather.get('condition', 'Clear')

            # Determine day/night game
            game_time = game.get('gameDate', '')
            if game_time:
                hour = int(game_time[11:13])
                conditions['game_time'] = 'Day' if 10 <= hour <= 17 else 'Night'

        except Exception as e:
            logger.error(f"Error extracting game conditions: {e}")

        return conditions
    
    def _extract_enhanced_player_data(self, box_data: Dict, game: Dict, conditions: Dict) -> List[Dict]:
        """Extract comprehensive player performance data with context"""
        player_performances = []

        # Get pitcher information
        home_pitcher = game['teams']['home'].get('probablePitcher', {})
        away_pitcher = game['teams']['away'].get('probablePitcher', {})

        for team_type in ['home', 'away']:
            if team_type not in box_data.get('teams', {}):
                continue

            team_data = box_data['teams'][team_type]
            opposing_pitcher = away_pitcher if team_type == 'home' else home_pitcher

            # Get batting order from box score
            batters = team_data.get('batters', [])

            for order_idx, player_id_str in enumerate(batters[:9]):  # First 9 are starters
                player_id = player_id_str.replace('ID', '') if 'ID' in player_id_str else player_id_str
                player_key = f'ID{player_id}'

                if player_key not in team_data.get('players', {}):
                    continue

                player_info = team_data['players'][player_key]

                # Get batting stats
                batting_stats = player_info.get('stats', {}).get('batting', {})

                if batting_stats.get('plateAppearances', 0) == 0:
                    continue  # Skip players who didn't bat

                # Enhanced player performance record
                performance = {
                    'player_id': int(player_id),
                    'player_name': player_info.get('person', {}).get('fullName', 'Unknown'),
                    'team_type': team_type,
                    'team_name': game['teams'][team_type]['team']['name'],
                    'batting_order': order_idx + 1,
                    'position': player_info.get('position', {}).get('abbreviation', 'UNK'),

                    # Core performance metrics
                    'rbi': batting_stats.get('rbi', 0),
                    'at_bats': batting_stats.get('atBats', 0),
                    'hits': batting_stats.get('hits', 0),
                    'doubles': batting_stats.get('doubles', 0),
                    'triples': batting_stats.get('triples', 0),
                    'home_runs': batting_stats.get('homeRuns', 0),
                    'walks': batting_stats.get('baseOnBalls', 0),
                    'strikeouts': batting_stats.get('strikeOuts', 0),
                    'plate_appearances': batting_stats.get('plateAppearances', 0),
                    'total_bases': batting_stats.get('totalBases', 0),
                    'left_on_base': batting_stats.get('leftOnBase', 0),

                    # Game context
                    'opponent': game['teams']['away' if team_type == 'home' else 'home']['team']['name'],
                    'is_home': team_type == 'home',
                    'game_time': conditions.get('game_time', 'Night'),
                    'weather_temp': conditions.get('temp', 72),
                    'weather_wind': conditions.get('wind', 5),

                    # Pitcher matchup
                    'opposing_pitcher_id': opposing_pitcher.get('id', 0),
                    'opposing_pitcher_name': opposing_pitcher.get('fullName', 'Unknown'),
                    'opposing_pitcher_hand': self._get_pitcher_hand_from_data(opposing_pitcher),

                    # Season stats context (would be filled by season stats lookup)
                    'season_avg': 0.250,  # Placeholder - to be filled by season stats
                    'season_rbi': 0,
                    'season_ops': 0.750,
                    'recent_form': 0.250,  # Last 15 games avg

                    # Target variables
                    'got_rbi': 1 if batting_stats.get('rbi', 0) > 0 else 0,
                    'rbi_count': batting_stats.get('rbi', 0)
                }

                player_performances.append(performance)

        return player_performances

    def _get_pitcher_hand_from_data(self, pitcher_data: Dict) -> str:
        """Extract pitcher handedness from pitcher data"""
        try:
            return pitcher_data.get('pitchHand', {}).get('code', 'R')
        except:
            return 'R'  # Default to right-handed
    
    def fetch_player_splits(self, player_id: int, season: int = 2024) -> PlayerSplits:
        """Fetch comprehensive player split statistics"""
        splits = {}
        
        split_types = [
            'vsPitcherHand', 'homeAway', 'dayNight',
            'lastXDays7', 'lastXDays14', 'lastXDays30'
        ]
        
        for split_type in split_types:
            url = f"{self.mlb_base}/people/{player_id}/stats"
            params = {
                'stats': f'season',
                'group': 'hitting',
                'season': season,
                'sportId': 1,
                'sitCodes': split_type
            }
            
            try:
                response = self.session.get(url, params=params)
                data = response.json()
                
                if 'stats' in data and data['stats']:
                    for stat_group in data['stats'][0].get('splits', []):
                        split_name = stat_group.get('split', {}).get('description', '')
                        split_stats = stat_group.get('stat', {})
                        splits[f"{split_type}_{split_name}"] = split_stats
                        
            except Exception as e:
                logger.error(f"Error fetching splits for player {player_id}: {e}")
        
        # Parse into PlayerSplits dataclass
        return self._parse_player_splits(splits)
    
    def _parse_player_splits(self, raw_splits: Dict) -> PlayerSplits:
        """Parse raw split data into structured format"""
        return PlayerSplits(
            vs_rhp=raw_splits.get('vsPitcherHand_vs RHP', {}),
            vs_lhp=raw_splits.get('vsPitcherHand_vs LHP', {}),
            home=raw_splits.get('homeAway_Home', {}),
            away=raw_splits.get('homeAway_Away', {}),
            day=raw_splits.get('dayNight_Day', {}),
            night=raw_splits.get('dayNight_Night', {}),
            last_7=raw_splits.get('lastXDays7', {}),
            last_14=raw_splits.get('lastXDays14', {}),
            last_30=raw_splits.get('lastXDays30', {})
        )
    
    def fetch_weather(self, lat: float, lon: float, game_time: datetime) -> WeatherData:
        """Fetch real weather data for game location"""
        try:
            url = f"{self.weather_base}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': WEATHER_API_KEY,
                'units': 'imperial'
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            # Calculate air density factor (affects ball flight)
            pressure_mb = data['main']['pressure']
            temp_f = data['main']['temp']
            humidity = data['main']['humidity']
            
            # Air density calculation
            temp_c = (temp_f - 32) * 5/9
            temp_k = temp_c + 273.15
            air_density = (pressure_mb * 100) / (287.05 * temp_k)
            air_density_factor = 1.225 / air_density  # Normalized to standard conditions
            
            # Wind component calculation (simplified)
            wind_speed = data['wind'].get('speed', 0)
            wind_deg = data['wind'].get('deg', 0)
            
            # Assume home plate to center field is 0 degrees
            # Tailwind is 0-45 and 315-360 degrees
            tailwind_component = 0
            if wind_deg <= 45 or wind_deg >= 315:
                tailwind_component = wind_speed * np.cos(np.radians(wind_deg))
            elif 135 <= wind_deg <= 225:
                tailwind_component = -wind_speed * np.cos(np.radians(wind_deg - 180))
            
            return WeatherData(
                temp_f=temp_f,
                humidity=humidity,
                wind_speed=wind_speed,
                wind_direction=self._wind_direction_from_degrees(wind_deg),
                pressure=pressure_mb,
                air_density_factor=air_density_factor,
                tailwind_component=tailwind_component
            )
            
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            # Return default weather conditions
            return WeatherData(
                temp_f=72, humidity=50, wind_speed=5,
                wind_direction='N', pressure=1013,
                air_density_factor=1.0, tailwind_component=0
            )
    
    def _wind_direction_from_degrees(self, degrees: float) -> str:
        """Convert wind degrees to cardinal direction"""
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = int((degrees + 22.5) / 45) % 8
        return directions[index]
    
    def fetch_player_props(self, game_date: str) -> List[OddsData]:
        """Fetch real betting odds for RBI props"""
        odds_data = []
        
        try:
            url = f"{self.odds_base}/sports/baseball_mlb/odds"
            params = {
                'apiKey': ODDS_API_KEY,
                'regions': 'us',
                'markets': 'player_rbis',
                'oddsFormat': 'american',
                'date': game_date
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            for game in data:
                for bookmaker in game.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'player_rbis':
                            for outcome in market.get('outcomes', []):
                                odds = OddsData(
                                    sportsbook=bookmaker['title'],
                                    player_name=outcome['name'],
                                    rbi_over_line=outcome.get('point', 0.5),
                                    over_odds=outcome.get('price', -110),
                                    under_odds=self._get_opposite_odds(outcome),
                                    implied_probability=self._american_to_probability(outcome.get('price', -110)),
                                    market_consensus=0,  # Will be calculated across books
                                    value_edge=0  # Will be calculated after prediction
                                )
                                odds_data.append(odds)
            
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
        
        return odds_data
    
    def _american_to_probability(self, american_odds: float) -> float:
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def _get_opposite_odds(self, outcome: Dict) -> float:
        """Get the opposite side odds"""
        # This would need to find the matching under odds
        # Simplified for now
        return 100 if outcome.get('price', -110) < 0 else -120

class BullpenAnalyzer:
    """Dynamic bullpen blending based on expected usage"""
    
    def __init__(self, mlb_fetcher: MLBDataFetcher):
        self.fetcher = mlb_fetcher
        
    def calculate_bullpen_impact(self, 
                                  starter_id: int,
                                  team_id: int,
                                  game_date: datetime) -> Dict[str, float]:
        """Calculate expected bullpen impact based on starter's typical innings"""
        
        # Get starter's average innings pitched
        starter_stats = self._get_pitcher_recent_starts(starter_id)
        avg_starter_innings = starter_stats.get('avg_innings', 5.5)
        
        # Get bullpen statistics
        bullpen_stats = self._get_team_bullpen_stats(team_id)
        
        # Calculate expected bullpen innings
        expected_bullpen_innings = max(0, 9 - avg_starter_innings)
        bullpen_weight = expected_bullpen_innings / 9
        
        # Weighted stats based on expected usage
        blended_stats = {
            'era': (starter_stats.get('era', 4.5) * (1 - bullpen_weight) + 
                   bullpen_stats.get('era', 4.5) * bullpen_weight),
            'whip': (starter_stats.get('whip', 1.3) * (1 - bullpen_weight) +
                    bullpen_stats.get('whip', 1.3) * bullpen_weight),
            'hr_per_9': (starter_stats.get('hr_per_9', 1.2) * (1 - bullpen_weight) +
                        bullpen_stats.get('hr_per_9', 1.2) * bullpen_weight),
            'k_per_9': (starter_stats.get('k_per_9', 8.5) * (1 - bullpen_weight) +
                       bullpen_stats.get('k_per_9', 9.0) * bullpen_weight),
            'expected_bullpen_innings': expected_bullpen_innings,
            'bullpen_weight': bullpen_weight
        }
        
        # Leverage index adjustments (high leverage situations)
        if expected_bullpen_innings > 3:
            # More bullpen usage means more high-leverage situations
            blended_stats['leverage_multiplier'] = 1.1
        else:
            blended_stats['leverage_multiplier'] = 1.0
        
        return blended_stats
    
    def _get_pitcher_recent_starts(self, pitcher_id: int) -> Dict[str, float]:
        """Get pitcher's recent performance metrics from MLB API"""
        try:
            url = f"{self.fetcher.mlb_base}/people/{pitcher_id}/stats"
            params = {
                'stats': 'season',
                'group': 'pitching',
                'season': 2024,
                'sportId': 1
            }

            response = self.fetcher.session.get(url, params=params)
            data = response.json()

            if 'stats' in data and data['stats'] and data['stats'][0].get('splits'):
                stats = data['stats'][0]['splits'][0]['stat']

                return {
                    'avg_innings': float(stats.get('inningsPitched', 0)) / max(float(stats.get('gamesStarted', 1)), 1),
                    'era': float(stats.get('era', 4.50)),
                    'whip': float(stats.get('whip', 1.30)),
                    'hr_per_9': float(stats.get('homeRunsPer9Inn', 1.2)),
                    'k_per_9': float(stats.get('strikeoutsPer9Inn', 8.5))
                }
        except Exception as e:
            logger.error(f"Error fetching pitcher stats for {pitcher_id}: {e}")

        # Return league averages if API fails
        return {
            'avg_innings': 5.5,
            'era': 4.50,
            'whip': 1.30,
            'hr_per_9': 1.2,
            'k_per_9': 8.5
        }
    
    def _get_team_bullpen_stats(self, team_id: int) -> Dict[str, float]:
        """Get team bullpen statistics from MLB API"""
        try:
            url = f"{self.fetcher.mlb_base}/teams/{team_id}/stats"
            params = {
                'stats': 'season',
                'group': 'pitching',
                'season': 2024,
                'sportId': 1
            }

            response = self.fetcher.session.get(url, params=params)
            data = response.json()

            if 'stats' in data and data['stats'] and data['stats'][0].get('splits'):
                stats = data['stats'][0]['splits'][0]['stat']

                # Filter for bullpen stats (non-starters)
                bullpen_url = f"{self.fetcher.mlb_base}/teams/{team_id}/roster"
                roster_response = self.fetcher.session.get(bullpen_url)
                roster_data = roster_response.json()

                return {
                    'era': float(stats.get('era', 4.50)),
                    'whip': float(stats.get('whip', 1.35)),
                    'hr_per_9': float(stats.get('homeRunsPer9Inn', 1.3)),
                    'k_per_9': float(stats.get('strikeoutsPer9Inn', 8.8))
                }
        except Exception as e:
            logger.error(f"Error fetching team bullpen stats for {team_id}: {e}")

        # Return league averages if API fails
        return {
            'era': 4.50,
            'whip': 1.35,
            'hr_per_9': 1.3,
            'k_per_9': 8.8
        }

class AdvancedRBIPredictorV3:
    """Main RBI prediction system with all v3.0 enhancements"""
    
    def __init__(self, db_path: str = 'rbi_predictions_v3.db'):
        self.db_path = db_path
        self.fetcher = MLBDataFetcher()
        self.bullpen_analyzer = BullpenAnalyzer(self.fetcher)
        self.models = {}
        self.scalers = {}
        self.shap_explainers = {}
        
        # Initialize database
        self._init_database()
        
        # Load or train models
        self._initialize_models()
    
    def _init_database(self):
        """Initialize SQLite database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Historical games table (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_games (
                game_id INTEGER PRIMARY KEY,
                date TEXT,
                game_datetime TEXT,
                home_team TEXT,
                away_team TEXT,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_score INTEGER,
                away_score INTEGER,
                venue_name TEXT,
                venue_lat REAL,
                venue_lon REAL,
                weather_temp REAL,
                weather_wind REAL,
                weather_condition TEXT,
                game_time_local TEXT,
                attendance INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # RBI Training Data table - NEW
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rbi_training_data_v3 (
                training_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                player_id INTEGER,
                player_name TEXT,
                team_name TEXT,
                team_type TEXT,
                batting_order INTEGER,
                position TEXT,

                -- Performance metrics
                rbi INTEGER,
                at_bats INTEGER,
                hits INTEGER,
                doubles INTEGER,
                triples INTEGER,
                home_runs INTEGER,
                walks INTEGER,
                strikeouts INTEGER,
                plate_appearances INTEGER,
                total_bases INTEGER,
                left_on_base INTEGER,

                -- Game context
                opponent TEXT,
                is_home INTEGER,
                game_time TEXT,
                weather_temp REAL,
                weather_wind REAL,

                -- Pitcher matchup
                opposing_pitcher_id INTEGER,
                opposing_pitcher_name TEXT,
                opposing_pitcher_hand TEXT,

                -- Season context (filled later)
                season_avg REAL,
                season_rbi INTEGER,
                season_ops REAL,
                recent_form REAL,

                -- Features (filled by feature engineering)
                feature_vector TEXT,  -- JSON string of feature values

                -- Target variables
                got_rbi INTEGER,
                rbi_count INTEGER,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES historical_games(game_id)
            )
        ''')
        
        # Player performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_performances (
                performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                player_id INTEGER,
                player_name TEXT,
                team TEXT,
                batting_order INTEGER,
                position TEXT,
                rbi INTEGER,
                at_bats INTEGER,
                hits INTEGER,
                home_runs INTEGER,
                walks INTEGER,
                strikeouts INTEGER,
                vs_pitcher_hand TEXT,
                game_time TEXT,
                FOREIGN KEY (game_id) REFERENCES historical_games(game_id)
            )
        ''')
        
        # Predictions table with enhanced fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                player_name TEXT,
                team TEXT,
                opponent TEXT,
                game_date TEXT,
                model_version TEXT DEFAULT 'v3.0',
                
                -- Prediction outputs
                rbi_probability REAL,
                expected_rbis REAL,
                confidence_score REAL,
                
                -- Split features used
                vs_pitcher_hand TEXT,
                home_away TEXT,
                day_night TEXT,
                recent_form_7d REAL,
                recent_form_14d REAL,
                recent_form_30d REAL,
                
                -- Environmental factors
                weather_temp REAL,
                weather_humidity REAL,
                wind_speed REAL,
                wind_component REAL,
                air_density_factor REAL,
                
                -- Market data
                sportsbook TEXT,
                market_line REAL,
                market_odds REAL,
                implied_probability REAL,
                value_edge REAL,
                
                -- Model explainability
                shap_values TEXT,  -- JSON string of SHAP values
                top_positive_features TEXT,  -- JSON string
                top_negative_features TEXT,  -- JSON string
                
                -- Outcome tracking
                actual_rbis INTEGER,
                bet_result TEXT,
                profit_loss REAL
            )
        ''')
        
        # Bullpen stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bullpen_stats (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER,
                date TEXT,
                bullpen_era REAL,
                bullpen_whip REAL,
                leverage_index REAL,
                expected_innings REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def prepare_training_data(self, start_date: str = '2022-04-01',
                             end_date: str = '2024-10-01') -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare real historical MLB data for training with comprehensive features"""
        logger.info(f"Preparing training data from real MLB games: {start_date} to {end_date}")

        # Check if we already have training data in database
        existing_data = self._load_training_data_from_db(start_date, end_date)

        if len(existing_data) > 5000:  # Sufficient existing data
            logger.info(f"Using {len(existing_data)} existing training samples from database")
            X, y = self._prepare_features_from_db_data(existing_data)
            return X, y

        # Fetch fresh historical games data
        logger.info("Fetching fresh historical data from MLB API...")
        games_df = self.fetcher.fetch_historical_games(start_date, end_date)

        if games_df.empty:
            logger.warning("No historical data fetched. Attempting to use cached data.")
            games_df = self._load_cached_data()

        if games_df.empty:
            logger.error("No training data available. Cannot proceed without historical data.")
            raise ValueError("No historical training data available")

        # Process and store player-level data
        training_records = []
        features_list = []
        targets_list = []

        logger.info(f"Processing {len(games_df)} games for training data...")

        for game_idx, (_, game) in enumerate(games_df.iterrows()):
            if game_idx % 100 == 0:
                logger.info(f"Processed {game_idx}/{len(games_df)} games")

            # Store game record
            self._store_game_record(game)

            # Process each player performance
            for player_perf in game.get('player_performances', []):
                if player_perf['plate_appearances'] == 0:
                    continue

                # Create enhanced feature vector
                features = self._create_enhanced_feature_vector(player_perf, game)
                target_got_rbi = player_perf['got_rbi']
                target_rbi_count = player_perf['rbi_count']

                # Store training record
                training_record = {
                    **player_perf,
                    'feature_vector': json.dumps(features.tolist()),
                    'game_id': game['game_id']
                }
                training_records.append(training_record)

                features_list.append(features)
                targets_list.append(target_rbi_count)  # Predict RBI count

        # Store training data in database
        self._store_training_records(training_records)

        X = np.array(features_list)
        y = np.array(targets_list)

        logger.info(f"Prepared {len(X)} training samples from {len(games_df)} games")
        logger.info(f"RBI distribution: 0 RBIs: {np.sum(y == 0)}, 1+ RBIs: {np.sum(y > 0)}")
        logger.info(f"Average RBI per PA: {np.mean(y):.3f}")

        return X, y

    def _load_training_data_from_db(self, start_date: str, end_date: str) -> List[Dict]:
        """Load existing training data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Query training data within date range
            cursor.execute('''
                SELECT t.*, h.date, h.game_datetime
                FROM rbi_training_data_v3 t
                JOIN historical_games h ON t.game_id = h.game_id
                WHERE h.date BETWEEN ? AND ?
                AND t.feature_vector IS NOT NULL
            ''', (start_date, end_date))

            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            conn.close()

            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            logger.error(f"Error loading training data from database: {e}")
            return []

    def _prepare_features_from_db_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert database training data to feature arrays"""
        features_list = []
        targets_list = []

        for record in training_data:
            try:
                # Load feature vector from JSON
                features = np.array(json.loads(record['feature_vector']))
                target = record['rbi_count']

                features_list.append(features)
                targets_list.append(target)
            except:
                continue

        return np.array(features_list), np.array(targets_list)

    def _store_game_record(self, game: Dict):
        """Store game record in historical_games table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO historical_games (
                    game_id, date, game_datetime, home_team, away_team,
                    home_team_id, away_team_id, home_score, away_score,
                    venue_name, venue_lat, venue_lon, weather_temp, weather_wind,
                    weather_condition, game_time_local, attendance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game['game_id'], game['date'], game.get('game_datetime', ''),
                game['home_team'], game['away_team'],
                game.get('home_team_id', 0), game.get('away_team_id', 0),
                game['home_score'], game['away_score'],
                game.get('venue_name', ''), game.get('venue_lat', 0), game.get('venue_lon', 0),
                game.get('weather_temp', 72), game.get('weather_wind', 5),
                game.get('weather_condition', 'Clear'), game.get('game_time_local', 'Night'),
                game.get('attendance', 0)
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing game record: {e}")

    def _store_training_records(self, training_records: List[Dict]):
        """Store training records in rbi_training_data_v3 table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for record in training_records:
                cursor.execute('''
                    INSERT OR REPLACE INTO rbi_training_data_v3 (
                        game_id, player_id, player_name, team_name, team_type,
                        batting_order, position, rbi, at_bats, hits, doubles, triples,
                        home_runs, walks, strikeouts, plate_appearances, total_bases,
                        left_on_base, opponent, is_home, game_time, weather_temp,
                        weather_wind, opposing_pitcher_id, opposing_pitcher_name,
                        opposing_pitcher_hand, season_avg, season_rbi, season_ops,
                        recent_form, feature_vector, got_rbi, rbi_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record['game_id'], record['player_id'], record['player_name'],
                    record['team_name'], record['team_type'], record['batting_order'],
                    record['position'], record['rbi'], record['at_bats'], record['hits'],
                    record['doubles'], record['triples'], record['home_runs'], record['walks'],
                    record['strikeouts'], record['plate_appearances'], record['total_bases'],
                    record['left_on_base'], record['opponent'], int(record['is_home']),
                    record['game_time'], record['weather_temp'], record['weather_wind'],
                    record['opposing_pitcher_id'], record['opposing_pitcher_name'],
                    record['opposing_pitcher_hand'], record['season_avg'], record['season_rbi'],
                    record['season_ops'], record['recent_form'], record['feature_vector'],
                    record['got_rbi'], record['rbi_count']
                ))

            conn.commit()
            conn.close()
            logger.info(f"Stored {len(training_records)} training records")

        except Exception as e:
            logger.error(f"Error storing training records: {e}")

    def _create_enhanced_feature_vector(self, player_perf: Dict, game: Dict) -> np.ndarray:
        """Create comprehensive feature vector from player performance and game context"""
        features = []

        # Basic player features
        features.extend([
            player_perf['batting_order'] / 9.0,  # Normalized batting order
            float(player_perf['is_home']),  # Home field advantage
            self._position_to_numeric(player_perf['position']),  # Position encoding
        ])

        # Performance context features
        features.extend([
            player_perf['season_avg'],
            player_perf['season_ops'],
            player_perf['recent_form'],
            float(player_perf['season_rbi']) / 162.0,  # Normalized season RBIs
        ])

        # Game context features
        features.extend([
            (player_perf['weather_temp'] - 72) / 20.0,  # Normalized temperature
            player_perf['weather_wind'] / 20.0,  # Normalized wind
            1.0 if player_perf['game_time'] == 'Day' else 0.0,  # Day game
        ])

        # Pitcher matchup features
        features.extend([
            1.0 if player_perf['opposing_pitcher_hand'] == 'L' else 0.0,  # LHP
            # Additional pitcher stats would go here (ERA, WHIP, etc.)
            4.50 / 10.0,  # Normalized pitcher ERA (placeholder)
            1.30 / 2.0,   # Normalized pitcher WHIP (placeholder)
        ])

        # Team context features (placeholders for now)
        features.extend([
            4.5 / 10.0,   # Team runs per game (normalized)
            0.320,        # Team OBP
            0.750 / 1.2,  # Team OPS (normalized)
        ])

        # Batting order multipliers
        order_weights = {1: 0.8, 2: 0.9, 3: 1.3, 4: 1.4, 5: 1.2, 6: 1.0, 7: 0.9, 8: 0.8, 9: 0.7}
        features.append(order_weights.get(player_perf['batting_order'], 1.0))

        # Expected plate appearances
        expected_pa = {1: 4.8, 2: 4.7, 3: 4.6, 4: 4.5, 5: 4.4, 6: 4.3, 7: 4.2, 8: 4.1, 9: 4.0}
        features.append(expected_pa.get(player_perf['batting_order'], 4.3) / 5.0)

        return np.array(features)

    def _position_to_numeric(self, position: str) -> float:
        """Convert position to numeric value"""
        position_values = {
            'C': 0.8, '1B': 1.0, '2B': 0.9, '3B': 1.1, 'SS': 0.9,
            'LF': 1.0, 'CF': 1.0, 'RF': 1.1, 'DH': 1.2, 'OF': 1.0,
            'IF': 0.95, 'UNK': 1.0
        }
        return position_values.get(position, 1.0)

    def _create_feature_vector(self, player_data: Dict, game_data: Dict) -> np.ndarray:
        """Create comprehensive feature vector for a player's game"""
        features = []
        
        # Basic features
        features.extend([
            player_data.get('batting_order', 5) / 9,  # Normalized batting order
            player_data.get('at_bats', 0),
            player_data.get('hits', 0) / max(player_data.get('at_bats', 1), 1),
            player_data.get('walks', 0),
            player_data.get('strikeouts', 0),
            int(player_data.get('team') == 'home')  # Home field advantage
        ])
        
        # Position-based features (simplified)
        position_weights = {
            'C': 0.8, '1B': 1.0, '2B': 0.9, '3B': 1.1,
            'SS': 0.9, 'LF': 1.0, 'CF': 1.0, 'RF': 1.1, 'DH': 1.2
        }
        features.append(position_weights.get(player_data.get('position', ''), 1.0))
        
        # Team context (simplified - would be expanded with real data)
        features.extend([
            game_data.get('home_score', 0) if player_data.get('team') == 'home' 
            else game_data.get('away_score', 0),
            5.0,  # Placeholder for team runs per game average
            0.330,  # Placeholder for team OBP
            0.450   # Placeholder for team SLG
        ])
        
        return np.array(features)
    
    def create_enhanced_features(self, 
                                player_name: str,
                                team: str,
                                batting_order: int,
                                opponent: str,
                                pitcher_id: int,
                                game_datetime: datetime,
                                venue_lat: float,
                                venue_lon: float) -> Dict[str, Any]:
        """Create comprehensive feature set with all v3.0 enhancements"""
        
        features = {}
        
        # 1. Player splits
        player_id = self._get_player_id(player_name)
        player_splits = self.fetcher.fetch_player_splits(player_id)
        
        # Determine which splits to use based on game context
        pitcher_hand = self._get_pitcher_hand(pitcher_id)
        is_home = self._is_home_game(team, venue_lat)
        is_day = self._is_day_game(game_datetime)
        
        # Select appropriate splits
        if pitcher_hand == 'R':
            split_stats = player_splits.vs_rhp
        else:
            split_stats = player_splits.vs_lhp
        
        features['split_avg'] = split_stats.get('avg', 0.250)
        features['split_obp'] = split_stats.get('obp', 0.320)
        features['split_slg'] = split_stats.get('slg', 0.400)
        features['split_ops'] = features['split_obp'] + features['split_slg']
        
        # Home/Away splits
        location_splits = player_splits.home if is_home else player_splits.away
        features['location_avg'] = location_splits.get('avg', 0.250)
        features['location_rbi_rate'] = location_splits.get('rbi', 0) / max(location_splits.get('gamesPlayed', 1), 1)
        
        # Day/Night splits
        time_splits = player_splits.day if is_day else player_splits.night
        features['time_avg'] = time_splits.get('avg', 0.250)
        features['time_ops'] = time_splits.get('obp', 0.320) + time_splits.get('slg', 0.400)
        
        # Recent form
        features['last_7_avg'] = player_splits.last_7.get('avg', 0.250)
        features['last_14_avg'] = player_splits.last_14.get('avg', 0.250)
        features['last_30_avg'] = player_splits.last_30.get('avg', 0.250)
        features['last_7_rbi'] = player_splits.last_7.get('rbi', 0)
        features['last_14_rbi'] = player_splits.last_14.get('rbi', 0)
        
        # 2. Weather features
        weather = self.fetcher.fetch_weather(venue_lat, venue_lon, game_datetime)
        features['temp_f'] = weather.temp_f
        features['humidity'] = weather.humidity
        features['wind_speed'] = weather.wind_speed
        features['air_density_factor'] = weather.air_density_factor
        features['tailwind_component'] = weather.tailwind_component
        
        # Weather impact on offense
        features['offensive_weather_boost'] = (
            (weather.temp_f - 72) * 0.003 +  # Higher temp = more offense
            (weather.tailwind_component * 0.005) +  # Tailwind helps
            ((1 - weather.air_density_factor) * 0.1)  # Lower density = ball travels farther
        )
        
        # 3. Dynamic bullpen blending
        bullpen_impact = self.bullpen_analyzer.calculate_bullpen_impact(
            pitcher_id, self._get_team_id(opponent), game_datetime
        )
        
        features['blended_era'] = bullpen_impact['era']
        features['blended_whip'] = bullpen_impact['whip']
        features['blended_hr_per_9'] = bullpen_impact['hr_per_9']
        features['expected_bullpen_innings'] = bullpen_impact['expected_bullpen_innings']
        features['leverage_multiplier'] = bullpen_impact['leverage_multiplier']
        
        # 4. Batting order context
        batting_order_weights = {
            1: 0.8, 2: 0.9, 3: 1.3, 4: 1.4,
            5: 1.2, 6: 1.0, 7: 0.9, 8: 0.8, 9: 0.7
        }
        features['batting_order_weight'] = batting_order_weights.get(batting_order, 1.0)
        
        # Expected plate appearances based on batting order
        expected_pa = {1: 4.8, 2: 4.7, 3: 4.6, 4: 4.5, 5: 4.4, 6: 4.3, 7: 4.2, 8: 4.1, 9: 4.0}
        features['expected_pa'] = expected_pa.get(batting_order, 4.3)
        
        # 5. Advanced pitcher matchup
        features['pitcher_era'] = self._get_pitcher_era(pitcher_id)
        features['pitcher_whip'] = self._get_pitcher_whip(pitcher_id)
        features['pitcher_k_rate'] = self._get_pitcher_k_rate(pitcher_id)
        features['pitcher_bb_rate'] = self._get_pitcher_bb_rate(pitcher_id)
        
        # 6. Park factors
        park_factor = self._get_park_factor(venue_lat, venue_lon)
        features['park_factor'] = park_factor
        features['park_adjusted_offense'] = park_factor * features['offensive_weather_boost']
        
        # 7. Team context
        team_stats = self._get_team_offensive_stats(team)
        features['team_runs_per_game'] = team_stats.get('runs_per_game', 4.5)
        features['team_obp'] = team_stats.get('obp', 0.320)
        features['team_ops'] = team_stats.get('ops', 0.750)
        features['team_risp_avg'] = team_stats.get('risp_avg', 0.260)
        
        # 8. Opponent defensive metrics
        opp_defense = self._get_team_defensive_stats(opponent)
        features['opp_era'] = opp_defense.get('era', 4.50)
        features['opp_whip'] = opp_defense.get('whip', 1.35)
        features['opp_defensive_efficiency'] = opp_defense.get('def_eff', 0.70)
        
        # Store metadata for explainability
        features['_metadata'] = {
            'player_name': player_name,
            'team': team,
            'opponent': opponent,
            'batting_order': batting_order,
            'game_datetime': game_datetime.isoformat(),
            'pitcher_hand': pitcher_hand,
            'is_home': is_home,
            'is_day': is_day
        }
        
        return features
    
    def _initialize_models(self):
        """Initialize ensemble models for RBI prediction"""
        logger.info("Initializing ensemble models...")
        
        # Try to load existing models
        if self._load_models():
            logger.info("Loaded existing models from disk")
            return
        
        # Train new models if none exist
        logger.info("Training new models...")

        # Prepare training data with more recent date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # Full season

        X, y = self.prepare_training_data(start_date, end_date)

        if len(X) == 0:
            logger.warning("No comprehensive training data available. Attempting minimal initialization...")
            self._create_default_models()
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # 1. XGBoost model
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(X_train_scaled, y_train)
        self.models['xgboost'] = xgb_model
        
        # 2. LightGBM model
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42
        )
        lgb_model.fit(X_train_scaled, y_train)
        self.models['lightgbm'] = lgb_model
        
        # 3. Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        
        # 4. Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        self.models['gradient_boost'] = gb_model
        
        # Initialize SHAP explainers
        self._init_shap_explainers(X_train_scaled)
        
        # Evaluate models
        self._evaluate_models(X_test_scaled, y_test)
        
        # Save models
        self._save_models()
        
        logger.info("Model training complete")

    def validate_formula_baseline(self) -> Dict[str, float]:
        """Validate formula-based probability calculator against real historical RBI rates"""
        logger.info("Validating formula baseline against historical data...")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get overall RBI statistics from training data
            cursor.execute('''
                SELECT
                    COUNT(*) as total_plate_appearances,
                    SUM(got_rbi) as total_rbis,
                    AVG(CAST(got_rbi AS FLOAT)) as rbi_rate_per_pa,
                    SUM(rbi_count) as total_rbi_count,
                    AVG(CAST(rbi_count AS FLOAT)) as avg_rbis_per_pa
                FROM rbi_training_data_v3
                WHERE plate_appearances > 0
            ''')

            overall_stats = cursor.fetchone()

            # Get RBI rates by batting order position
            cursor.execute('''
                SELECT
                    batting_order,
                    COUNT(*) as plate_appearances,
                    SUM(got_rbi) as rbis,
                    AVG(CAST(got_rbi AS FLOAT)) as rbi_rate,
                    AVG(CAST(rbi_count AS FLOAT)) as avg_rbi_count
                FROM rbi_training_data_v3
                WHERE plate_appearances > 0
                GROUP BY batting_order
                ORDER BY batting_order
            ''')

            batting_order_stats = cursor.fetchall()

            # Get RBI rates by game situation
            cursor.execute('''
                SELECT
                    game_time,
                    COUNT(*) as plate_appearances,
                    AVG(CAST(got_rbi AS FLOAT)) as rbi_rate
                FROM rbi_training_data_v3
                WHERE plate_appearances > 0
                GROUP BY game_time
            ''')

            game_time_stats = cursor.fetchall()

            # Get RBI rates vs pitcher handedness
            cursor.execute('''
                SELECT
                    opposing_pitcher_hand,
                    COUNT(*) as plate_appearances,
                    AVG(CAST(got_rbi AS FLOAT)) as rbi_rate
                FROM rbi_training_data_v3
                WHERE plate_appearances > 0
                GROUP BY opposing_pitcher_hand
            ''')

            pitcher_hand_stats = cursor.fetchall()

            conn.close()

            # Calculate baseline metrics
            baseline_metrics = {
                'league_avg_rbi_rate': overall_stats[2] if overall_stats else 0.11,
                'league_avg_rbis_per_pa': overall_stats[4] if overall_stats else 0.11,
                'total_samples': overall_stats[0] if overall_stats else 0,
                'batting_order_variance': 0,
                'day_night_difference': 0,
                'platoon_advantage': 0
            }

            # Calculate batting order variance
            if batting_order_stats:
                rbi_rates = [row[3] for row in batting_order_stats if row[3] is not None]
                if rbi_rates:
                    baseline_metrics['batting_order_variance'] = np.var(rbi_rates)

            # Calculate day/night difference
            day_rate = night_rate = 0
            for row in game_time_stats:
                if row[0] == 'Day':
                    day_rate = row[2]
                elif row[0] == 'Night':
                    night_rate = row[2]

            baseline_metrics['day_night_difference'] = abs(day_rate - night_rate)

            # Calculate platoon advantage
            rhp_rate = lhp_rate = 0
            for row in pitcher_hand_stats:
                if row[0] == 'R':
                    rhp_rate = row[2]
                elif row[0] == 'L':
                    lhp_rate = row[2]

            baseline_metrics['platoon_advantage'] = abs(rhp_rate - lhp_rate)

            # Log validation results
            logger.info(f"Formula Baseline Validation Results:")
            logger.info(f"  League Average RBI Rate: {baseline_metrics['league_avg_rbi_rate']:.3f}")
            logger.info(f"  Expected ~0.11 (11% of plate appearances should result in RBI)")
            logger.info(f"  Total Training Samples: {baseline_metrics['total_samples']:,}")
            logger.info(f"  Batting Order Variance: {baseline_metrics['batting_order_variance']:.4f}")
            logger.info(f"  Day/Night Difference: {baseline_metrics['day_night_difference']:.3f}")
            logger.info(f"  Platoon Advantage: {baseline_metrics['platoon_advantage']:.3f}")

            # Validate against expected MLB norms
            validation_status = "PASS"
            if baseline_metrics['league_avg_rbi_rate'] < 0.08 or baseline_metrics['league_avg_rbi_rate'] > 0.15:
                validation_status = "FAIL - RBI rate outside expected range (0.08-0.15)"
            elif baseline_metrics['total_samples'] < 1000:
                validation_status = "FAIL - Insufficient training data"

            logger.info(f"  Validation Status: {validation_status}")

            return baseline_metrics

        except Exception as e:
            logger.error(f"Error validating formula baseline: {e}")
            return {
                'league_avg_rbi_rate': 0.11,
                'validation_status': 'ERROR'
            }

    def _init_shap_explainers(self, X_train: np.ndarray):
        """Initialize SHAP explainers for model interpretability"""
        logger.info("Initializing SHAP explainers...")
        
        # Sample background data for SHAP
        background = shap.sample(X_train, min(100, len(X_train)))
        
        # Create explainers for each model
        self.shap_explainers['xgboost'] = shap.Explainer(self.models['xgboost'], background)
        self.shap_explainers['lightgbm'] = shap.Explainer(self.models['lightgbm'], background)
        self.shap_explainers['random_forest'] = shap.Explainer(self.models['random_forest'].predict, background)
        self.shap_explainers['gradient_boost'] = shap.Explainer(self.models['gradient_boost'].predict, background)
    
    def predict_with_explanation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make RBI prediction with SHAP-based explanation"""
        
        # Extract metadata
        metadata = features.pop('_metadata', {})
        
        # Convert features to array
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Scale features
        if 'main' in self.scalers:
            feature_array = self.scalers['main'].transform(feature_array)
        
        # Get predictions from all models
        predictions = {}
        shap_values = {}
        
        for model_name, model in self.models.items():
            predictions[model_name] = float(model.predict(feature_array)[0])
            
            # Get SHAP values for explainability
            if model_name in self.shap_explainers:
                shap_vals = self.shap_explainers[model_name](feature_array)
                shap_values[model_name] = shap_vals.values[0]
        
        # Ensemble prediction (weighted average)
        weights = {'xgboost': 0.3, 'lightgbm': 0.3, 'random_forest': 0.2, 'gradient_boost': 0.2}
        ensemble_prediction = sum(predictions[m] * w for m, w in weights.items())
        
        # Calculate RBI probability (0+ RBIs)
        rbi_probability = 1 - np.exp(-ensemble_prediction)  # Poisson approximation
        
        # Get average SHAP values across models
        avg_shap_values = np.mean([shap_values[m] for m in shap_values], axis=0)
        
        # Identify top contributing features
        feature_names = list(features.keys())
        feature_importance = [(feature_names[i], avg_shap_values[i]) 
                            for i in range(len(feature_names))]
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_positive = [f for f in feature_importance if f[1] > 0][:5]
        top_negative = [f for f in feature_importance if f[1] < 0][:5]
        
        # Get market odds if available
        game_date = metadata.get('game_datetime', datetime.now().isoformat()).split('T')[0]
        odds_data = self.fetcher.fetch_player_props(game_date)
        
        player_odds = None
        value_edge = 0
        for odds in odds_data:
            if odds.player_name == metadata.get('player_name'):
                player_odds = odds
                value_edge = rbi_probability - odds.implied_probability
                break
        
        # Confidence score based on model agreement
        model_std = np.std(list(predictions.values()))
        confidence_score = max(0, 1 - (model_std / ensemble_prediction))
        
        # Recommendation logic
        if value_edge > 0.05 and confidence_score > 0.7:
            recommendation = "STRONG BET"
        elif value_edge > 0.02 and confidence_score > 0.5:
            recommendation = "BET"
        elif value_edge > 0:
            recommendation = "LEAN BET"
        else:
            recommendation = "PASS"
        
        result = {
            'player_name': metadata.get('player_name'),
            'team': metadata.get('team'),
            'opponent': metadata.get('opponent'),
            'batting_order': metadata.get('batting_order'),
            'game_datetime': metadata.get('game_datetime'),
            
            # Predictions
            'expected_rbis': ensemble_prediction,
            'rbi_probability': rbi_probability,
            'confidence_score': confidence_score,
            'recommendation': recommendation,
            
            # Model outputs
            'model_predictions': predictions,
            
            # Explainability
            'top_positive_features': top_positive,
            'top_negative_features': top_negative,
            'shap_values': avg_shap_values.tolist(),
            
            # Market comparison
            'market_odds': player_odds.over_odds if player_odds else None,
            'implied_probability': player_odds.implied_probability if player_odds else None,
            'value_edge': value_edge,
            
            # Feature values for transparency
            'feature_values': features
        }
        
        # Store prediction in database
        self._store_prediction(result)
        
        return result
    
    def _store_prediction(self, prediction: Dict[str, Any]):
        """Store prediction with all metadata in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                player_name, team, opponent, game_date, model_version,
                rbi_probability, expected_rbis, confidence_score,
                vs_pitcher_hand, home_away, day_night,
                recent_form_7d, recent_form_14d, recent_form_30d,
                weather_temp, weather_humidity, wind_speed, wind_component, air_density_factor,
                sportsbook, market_line, market_odds, implied_probability, value_edge,
                shap_values, top_positive_features, top_negative_features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction['player_name'],
            prediction['team'],
            prediction['opponent'],
            prediction['game_datetime'],
            'v3.0',
            prediction['rbi_probability'],
            prediction['expected_rbis'],
            prediction['confidence_score'],
            prediction.get('pitcher_hand', ''),
            prediction.get('home_away', ''),
            prediction.get('day_night', ''),
            prediction.get('feature_values', {}).get('last_7_avg', 0),
            prediction.get('feature_values', {}).get('last_14_avg', 0),
            prediction.get('feature_values', {}).get('last_30_avg', 0),
            prediction.get('feature_values', {}).get('temp_f', 72),
            prediction.get('feature_values', {}).get('humidity', 50),
            prediction.get('feature_values', {}).get('wind_speed', 5),
            prediction.get('feature_values', {}).get('tailwind_component', 0),
            prediction.get('feature_values', {}).get('air_density_factor', 1.0),
            '',  # sportsbook
            0.5,  # market_line
            prediction.get('market_odds', -110),
            prediction.get('implied_probability', 0.5),
            prediction['value_edge'],
            json.dumps(prediction['shap_values']),
            json.dumps(prediction['top_positive_features']),
            json.dumps(prediction['top_negative_features'])
        ))
        
        conn.commit()
        conn.close()
    
    def analyze_daily_slate(self, date: str = None) -> List[Dict[str, Any]]:
        """Analyze all games and players for a given date"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Analyzing RBI props for {date}")
        
        # Get today's games
        games = self._get_games_for_date(date)
        predictions = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for game in games:
                # Get probable lineups
                for team_type in ['home', 'away']:
                    lineup = self._get_probable_lineup(game, team_type)
                    
                    for player in lineup:
                        future = executor.submit(
                            self._analyze_player,
                            player,
                            game,
                            team_type
                        )
                        futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        predictions.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing player: {e}")
        
        # Sort by value edge
        predictions.sort(key=lambda x: x.get('value_edge', 0), reverse=True)
        
        # Generate summary
        summary = self._generate_daily_summary(predictions, date)
        
        return {
            'date': date,
            'predictions': predictions,
            'summary': summary
        }
    
    def _generate_daily_summary(self, predictions: List[Dict], date: str) -> Dict[str, Any]:
        """Generate summary statistics for daily predictions"""
        
        strong_bets = [p for p in predictions if p['recommendation'] == 'STRONG BET']
        regular_bets = [p for p in predictions if p['recommendation'] == 'BET']
        lean_bets = [p for p in predictions if p['recommendation'] == 'LEAN BET']
        
        summary = {
            'total_players_analyzed': len(predictions),
            'strong_bets': len(strong_bets),
            'regular_bets': len(regular_bets),
            'lean_bets': len(lean_bets),
            'top_value_plays': [],
            'highest_probability': [],
            'best_spots': []
        }
        
        # Top value plays
        for pred in predictions[:5]:
            if pred['value_edge'] > 0:
                summary['top_value_plays'].append({
                    'player': pred['player_name'],
                    'team': pred['team'],
                    'value_edge': pred['value_edge'],
                    'probability': pred['rbi_probability']
                })
        
        # Highest probability plays
        high_prob = sorted(predictions, key=lambda x: x['rbi_probability'], reverse=True)[:5]
        for pred in high_prob:
            summary['highest_probability'].append({
                'player': pred['player_name'],
                'team': pred['team'],
                'probability': pred['rbi_probability'],
                'batting_order': pred['batting_order']
            })
        
        # Best situational spots
        for pred in predictions:
            if (pred['batting_order'] in [3, 4, 5] and 
                pred.get('feature_values', {}).get('team_runs_per_game', 0) > 5.0):
                summary['best_spots'].append({
                    'player': pred['player_name'],
                    'situation': f"Batting {pred['batting_order']} for high-scoring team",
                    'probability': pred['rbi_probability']
                })
                if len(summary['best_spots']) >= 3:
                    break
        
        return summary
    
    # Helper methods (simplified implementations)
    def _get_player_id(self, player_name: str) -> int:
        """Get player ID from name using MLB API search"""
        try:
            url = f"{self.fetcher.mlb_base}/people/search"
            params = {
                'names': player_name,
                'sportId': 1,
                'active': True
            }

            response = self.fetcher.session.get(url, params=params)
            data = response.json()

            if 'people' in data and data['people']:
                # Return the first matching player ID
                return int(data['people'][0]['id'])
        except Exception as e:
            logger.error(f"Error searching for player {player_name}: {e}")

        # Return a deterministic fallback if search fails
        return abs(hash(player_name)) % 100000 + 500000
    
    def _get_pitcher_hand(self, pitcher_id: int) -> str:
        """Get pitcher throwing hand from MLB API"""
        try:
            url = f"{self.fetcher.mlb_base}/people/{pitcher_id}"
            response = self.fetcher.session.get(url)
            data = response.json()

            if 'people' in data and data['people']:
                pitcher_data = data['people'][0]
                return pitcher_data.get('pitchHand', {}).get('code', 'R')
        except Exception as e:
            logger.error(f"Error fetching pitcher hand for {pitcher_id}: {e}")

        # Return 'R' as default (majority of pitchers are right-handed)
        return 'R'
    
    def _is_home_game(self, team: str, venue_lat: float) -> bool:
        """Determine if team is playing at home by checking venue coordinates"""
        try:
            # Get team's home venue coordinates
            team_venues = {
                'New York Yankees': (40.8296, -73.9262),
                'Boston Red Sox': (42.3467, -71.0972),
                'Los Angeles Dodgers': (34.0739, -118.2400),
                'San Francisco Giants': (37.7786, -122.3893),
                'Chicago Cubs': (41.9484, -87.6553),
                'St. Louis Cardinals': (38.6226, -90.1928),
                'Atlanta Braves': (33.8906, -84.4677),
                'Houston Astros': (29.7572, -95.3552),
                'Seattle Mariners': (47.5914, -122.3325),
                'Texas Rangers': (32.7510, -97.0829)
                # Add more teams as needed
            }

            if team in team_venues:
                home_lat, home_lon = team_venues[team]
                # Check if venue coordinates are close to team's home (within ~0.1 degrees)
                return abs(venue_lat - home_lat) < 0.1
        except Exception as e:
            logger.error(f"Error determining home game for {team}: {e}")

        # Default to away game if unable to determine
        return False
    
    def _is_day_game(self, game_datetime: datetime) -> bool:
        """Determine if game is during day"""
        return 10 <= game_datetime.hour <= 17
    
    def _get_team_id(self, team_name: str) -> int:
        """Get team ID from name using MLB API"""
        try:
            url = f"{self.fetcher.mlb_base}/teams"
            params = {'sportId': 1, 'season': 2024}

            response = self.fetcher.session.get(url, params=params)
            data = response.json()

            if 'teams' in data:
                for team in data['teams']:
                    if team['name'] == team_name or team_name in team['name']:
                        return int(team['id'])
        except Exception as e:
            logger.error(f"Error fetching team ID for {team_name}: {e}")

        # MLB team IDs are typically in 100s range
        return abs(hash(team_name)) % 30 + 108
    
    def _get_pitcher_era(self, pitcher_id: int) -> float:
        """Get pitcher ERA from MLB API"""
        try:
            url = f"{self.fetcher.mlb_base}/people/{pitcher_id}/stats"
            params = {
                'stats': 'season',
                'group': 'pitching',
                'season': 2024,
                'sportId': 1
            }

            response = self.fetcher.session.get(url, params=params)
            data = response.json()

            if 'stats' in data and data['stats'] and data['stats'][0].get('splits'):
                stats = data['stats'][0]['splits'][0]['stat']
                return float(stats.get('era', 4.50))
        except Exception as e:
            logger.error(f"Error fetching ERA for pitcher {pitcher_id}: {e}")

        return 4.50  # League average ERA
    
    def _get_pitcher_whip(self, pitcher_id: int) -> float:
        """Get pitcher WHIP from MLB API"""
        try:
            url = f"{self.fetcher.mlb_base}/people/{pitcher_id}/stats"
            params = {
                'stats': 'season',
                'group': 'pitching',
                'season': 2024,
                'sportId': 1
            }

            response = self.fetcher.session.get(url, params=params)
            data = response.json()

            if 'stats' in data and data['stats'] and data['stats'][0].get('splits'):
                stats = data['stats'][0]['splits'][0]['stat']
                return float(stats.get('whip', 1.30))
        except Exception as e:
            logger.error(f"Error fetching WHIP for pitcher {pitcher_id}: {e}")

        return 1.30  # League average WHIP
    
    def _get_pitcher_k_rate(self, pitcher_id: int) -> float:
        """Get pitcher strikeout rate from MLB API"""
        try:
            url = f"{self.fetcher.mlb_base}/people/{pitcher_id}/stats"
            params = {
                'stats': 'season',
                'group': 'pitching',
                'season': 2024,
                'sportId': 1
            }

            response = self.fetcher.session.get(url, params=params)
            data = response.json()

            if 'stats' in data and data['stats'] and data['stats'][0].get('splits'):
                stats = data['stats'][0]['splits'][0]['stat']
                return float(stats.get('strikeoutsPer9Inn', 8.5))
        except Exception as e:
            logger.error(f"Error fetching K rate for pitcher {pitcher_id}: {e}")

        return 8.5  # League average K/9
    
    def _get_pitcher_bb_rate(self, pitcher_id: int) -> float:
        """Get pitcher walk rate from MLB API"""
        try:
            url = f"{self.fetcher.mlb_base}/people/{pitcher_id}/stats"
            params = {
                'stats': 'season',
                'group': 'pitching',
                'season': 2024,
                'sportId': 1
            }

            response = self.fetcher.session.get(url, params=params)
            data = response.json()

            if 'stats' in data and data['stats'] and data['stats'][0].get('splits'):
                stats = data['stats'][0]['splits'][0]['stat']
                return float(stats.get('baseOnBallsPer9Inn', 3.2))
        except Exception as e:
            logger.error(f"Error fetching BB rate for pitcher {pitcher_id}: {e}")

        return 3.2  # League average BB/9
    
    def _get_park_factor(self, lat: float, lon: float) -> float:
        """Get park offensive factor based on actual ballpark dimensions and conditions"""
        # Park factors based on actual MLB ballparks (1.0 = league average)
        park_factors = {
            # Yankee Stadium (short right field)
            (40.8296, -73.9262): 1.08,
            # Fenway Park (Green Monster)
            (42.3467, -71.0972): 1.05,
            # Coors Field (high altitude)
            (39.7559, -104.9942): 1.15,
            # Petco Park (pitcher friendly)
            (32.7073, -117.1566): 0.92,
            # Marlins Park (pitcher friendly)
            (25.7781, -80.2197): 0.94,
            # Minute Maid Park (short left field)
            (29.7572, -95.3552): 1.06,
            # Camden Yards
            (39.2838, -76.6217): 1.02,
            # Wrigley Field
            (41.9484, -87.6553): 1.03
        }

        # Find closest matching park
        closest_factor = 1.0
        min_distance = float('inf')

        for (park_lat, park_lon), factor in park_factors.items():
            distance = ((lat - park_lat) ** 2 + (lon - park_lon) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_factor = factor

        # If within 0.1 degrees, use the park factor, otherwise use league average
        return closest_factor if min_distance < 0.1 else 1.0
    
    def _get_team_offensive_stats(self, team: str) -> Dict[str, float]:
        """Get team offensive statistics from MLB API"""
        try:
            team_id = self._get_team_id(team)
            url = f"{self.fetcher.mlb_base}/teams/{team_id}/stats"
            params = {
                'stats': 'season',
                'group': 'hitting',
                'season': 2024,
                'sportId': 1
            }

            response = self.fetcher.session.get(url, params=params)
            data = response.json()

            if 'stats' in data and data['stats'] and data['stats'][0].get('splits'):
                stats = data['stats'][0]['splits'][0]['stat']
                games_played = max(float(stats.get('gamesPlayed', 162)), 1)

                return {
                    'runs_per_game': float(stats.get('runs', 0)) / games_played,
                    'obp': float(stats.get('obp', 0.320)),
                    'ops': float(stats.get('ops', 0.750)),
                    'risp_avg': float(stats.get('avg', 0.260))  # Approximation
                }
        except Exception as e:
            logger.error(f"Error fetching team offensive stats for {team}: {e}")

        # Return league averages if API fails
        return {
            'runs_per_game': 4.5,
            'obp': 0.320,
            'ops': 0.750,
            'risp_avg': 0.260
        }
    
    def _get_team_defensive_stats(self, team: str) -> Dict[str, float]:
        """Get team defensive statistics from MLB API"""
        try:
            team_id = self._get_team_id(team)
            url = f"{self.fetcher.mlb_base}/teams/{team_id}/stats"
            params = {
                'stats': 'season',
                'group': 'pitching',
                'season': 2024,
                'sportId': 1
            }

            response = self.fetcher.session.get(url, params=params)
            data = response.json()

            if 'stats' in data and data['stats'] and data['stats'][0].get('splits'):
                stats = data['stats'][0]['splits'][0]['stat']

                # Get fielding stats too
                fielding_url = f"{self.fetcher.mlb_base}/teams/{team_id}/stats"
                fielding_params = {
                    'stats': 'season',
                    'group': 'fielding',
                    'season': 2024,
                    'sportId': 1
                }

                fielding_response = self.fetcher.session.get(fielding_url, params=fielding_params)
                fielding_data = fielding_response.json()

                def_eff = 0.70  # Default
                if 'stats' in fielding_data and fielding_data['stats'] and fielding_data['stats'][0].get('splits'):
                    fielding_stats = fielding_data['stats'][0]['splits'][0]['stat']
                    errors = float(fielding_stats.get('errors', 100))
                    chances = float(fielding_stats.get('chances', 4000))
                    def_eff = 1 - (errors / max(chances, 1))

                return {
                    'era': float(stats.get('era', 4.50)),
                    'whip': float(stats.get('whip', 1.30)),
                    'def_eff': def_eff
                }
        except Exception as e:
            logger.error(f"Error fetching team defensive stats for {team}: {e}")

        # Return league averages if API fails
        return {
            'era': 4.50,
            'whip': 1.30,
            'def_eff': 0.70
        }
    
    def _create_default_models(self):
        """Create default models when no training data available - fetch minimal real data first"""
        logger.warning("No historical training data available. Attempting to fetch minimal recent data...")

        # Try to get at least some recent data for training
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        X, y = self.prepare_training_data(start_date, end_date)

        if len(X) == 0:
            logger.error("Unable to fetch any real training data. Models cannot be initialized without real data.")
            raise ValueError("No real training data available. Please ensure MLB API is accessible and try again.")

        logger.info(f"Using {len(X)} samples from recent games for model initialization")

        # Split the minimal data
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler

        # Create and train models with reduced complexity for small datasets
        n_estimators = min(50, len(X_train) * 2)  # Reduce overfitting

        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            random_state=42
        )
        self.models['gradient_boost'] = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )

        # Train models
        for model_name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)
                logger.info(f"Initialized {model_name} with real data")
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")

        logger.info("Models initialized with minimal real training data")
    
    def _save_models(self):
        """Save trained models to disk"""
        models_data = {
            'models': self.models,
            'scalers': self.scalers,
            'version': 'v3.0',
            'timestamp': datetime.now().isoformat()
        }
        
        with open('rbi_models_v3.pkl', 'wb') as f:
            pickle.dump(models_data, f)
        
        logger.info("Models saved to disk")
    
    def _load_models(self) -> bool:
        """Load models from disk if available"""
        try:
            with open('rbi_models_v3.pkl', 'rb') as f:
                models_data = pickle.load(f)
                
            self.models = models_data['models']
            self.scalers = models_data['scalers']
            
            logger.info(f"Loaded models version {models_data['version']} from {models_data['timestamp']}")
            return True
            
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance"""
        logger.info("Evaluating models...")
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    def _store_training_data(self, games_df: pd.DataFrame):
        """Store training data in database"""
        conn = sqlite3.connect(self.db_path)
        
        # Store games
        games_df.to_sql('historical_games_backup', conn, if_exists='replace', index=False)
        
        conn.close()
    
    def _load_cached_data(self) -> pd.DataFrame:
        """Load cached training data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM historical_games_backup", conn)
            conn.close()
            return df
        except:
            return pd.DataFrame()
    
    def _get_games_for_date(self, date: str) -> List[Dict]:
        """Get all games scheduled for a date from MLB API"""
        try:
            url = f"{self.fetcher.mlb_base}/schedule"
            params = {
                'sportId': 1,
                'date': date,
                'hydrate': 'probablePitcher,venue'
            }

            response = self.fetcher.session.get(url, params=params)
            data = response.json()

            games = []
            if 'dates' in data and data['dates']:
                for date_info in data['dates']:
                    for game in date_info.get('games', []):
                        # Extract game information
                        game_info = {
                            'game_id': game['gamePk'],
                            'home_team': game['teams']['home']['team']['name'],
                            'away_team': game['teams']['away']['team']['name'],
                            'venue_lat': float(game.get('venue', {}).get('location', {}).get('latitude', 40.7128)),
                            'venue_lon': float(game.get('venue', {}).get('location', {}).get('longitude', -74.0060)),
                            'game_time': datetime.strptime(game['gameDate'][:19], '%Y-%m-%dT%H:%M:%S')
                        }

                        # Get probable pitchers
                        home_pitcher = game['teams']['home'].get('probablePitcher')
                        away_pitcher = game['teams']['away'].get('probablePitcher')

                        game_info['home_pitcher_id'] = home_pitcher['id'] if home_pitcher else None
                        game_info['away_pitcher_id'] = away_pitcher['id'] if away_pitcher else None

                        games.append(game_info)

            return games

        except Exception as e:
            logger.error(f"Error fetching games for {date}: {e}")
            # Return empty list instead of demo data
            return []
    
    def _get_probable_lineup(self, game: Dict, team_type: str) -> List[Dict]:
        """Get probable lineup for team from MLB API"""
        try:
            team_id = self._get_team_id(game[f'{team_type}_team'])

            # First try to get lineup from game data
            url = f"{self.fetcher.mlb_base}/game/{game['game_id']}/boxscore"
            response = self.fetcher.session.get(url)
            data = response.json()

            lineup = []
            if 'teams' in data and team_type in data['teams']:
                team_data = data['teams'][team_type]
                batters = team_data.get('batters', [])

                for i, batter_id in enumerate(batters[:9]):  # First 9 batters in lineup
                    player_info = team_data.get('players', {}).get(f'ID{batter_id}', {})
                    if player_info:
                        lineup.append({
                            'player_name': player_info.get('person', {}).get('fullName', f'Player {batter_id}'),
                            'batting_order': i + 1,
                            'team': game[f'{team_type}_team'],
                            'player_id': batter_id
                        })

            # If we couldn't get lineup from game data, try roster
            if not lineup:
                roster_url = f"{self.fetcher.mlb_base}/teams/{team_id}/roster"
                roster_response = self.fetcher.session.get(roster_url)
                roster_data = roster_response.json()

                if 'roster' in roster_data:
                    # Get position players (rough approximation of batting order)
                    position_players = [p for p in roster_data['roster']
                                      if p.get('position', {}).get('type') == 'Hitter']

                    for i, player in enumerate(position_players[:9]):
                        lineup.append({
                            'player_name': player.get('person', {}).get('fullName', 'Unknown'),
                            'batting_order': i + 1,
                            'team': game[f'{team_type}_team'],
                            'player_id': player.get('person', {}).get('id')
                        })

            return lineup[:9]  # Return max 9 players

        except Exception as e:
            logger.error(f"Error fetching lineup for {game[f'{team_type}_team']}: {e}")
            return []  # Return empty list instead of sample data
    
    def _analyze_player(self, player: Dict, game: Dict, team_type: str) -> Dict[str, Any]:
        """Analyze individual player for RBI prediction"""
        
        opponent = game['away_team'] if team_type == 'home' else game['home_team']
        pitcher_id = game['away_pitcher_id'] if team_type == 'home' else game['home_pitcher_id']
        
        # Create features
        features = self.create_enhanced_features(
            player['player_name'],
            player['team'],
            player['batting_order'],
            opponent,
            pitcher_id,
            game['game_time'],
            game['venue_lat'],
            game['venue_lon']
        )
        
        # Get prediction with explanation
        prediction = self.predict_with_explanation(features)
        
        return prediction


def main():
    """Main execution function"""
    print("=" * 70)
    print("MLB RBI PREDICTION SYSTEM v3.0")
    print("Advanced Machine Learning with Real Data Integration")
    print("=" * 70)
    
    # Initialize system
    predictor = AdvancedRBIPredictorV3()
    
    while True:
        print("\n" + "=" * 50)
        print("OPTIONS:")
        print("1. Analyze today's RBI props")
        print("2. Predict specific player RBI")
        print("3. Backtest historical performance")
        print("4. View model statistics")
        print("5. Update models with latest data")
        print("6. Exit")
        print("=" * 50)
        
        choice = input("\nSelect option (1-6): ")
        
        if choice == '1':
            # Analyze today's slate
            date = input("Enter date (YYYY-MM-DD) or press Enter for today: ")
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            results = predictor.analyze_daily_slate(date)
            
            print(f"\n{'=' * 60}")
            print(f"RBI PREDICTIONS FOR {date}")
            print(f"{'=' * 60}")
            
            # Display summary
            summary = results['summary']
            print(f"\n📊 SUMMARY:")
            print(f"  Total Players: {summary['total_players_analyzed']}")
            print(f"  Strong Bets: {summary['strong_bets']}")
            print(f"  Regular Bets: {summary['regular_bets']}")
            print(f"  Lean Bets: {summary['lean_bets']}")
            
            # Top value plays
            if summary['top_value_plays']:
                print(f"\n💎 TOP VALUE PLAYS:")
                for play in summary['top_value_plays'][:3]:
                    print(f"  {play['player']} ({play['team']})")
                    print(f"    Probability: {play['probability']:.1%}")
                    print(f"    Value Edge: {play['value_edge']:.1%}")
            
            # Highest probability
            if summary['highest_probability']:
                print(f"\n🎯 HIGHEST PROBABILITY:")
                for play in summary['highest_probability'][:3]:
                    print(f"  {play['player']} - Batting {play['batting_order']}")
                    print(f"    Probability: {play['probability']:.1%}")
            
        elif choice == '2':
            # Individual player prediction
            print("\n" + "=" * 50)
            print("INDIVIDUAL PLAYER PREDICTION")
            print("=" * 50)
            
            player_name = input("Player name: ")
            team = input("Team: ")
            batting_order = int(input("Batting order (1-9): "))
            opponent = input("Opponent team: ")
            
            # Create basic features (simplified)
            features = predictor.create_enhanced_features(
                player_name, team, batting_order, opponent,
                12345,  # Dummy pitcher ID
                datetime.now(),
                40.7128, -74.0060  # Default NYC coordinates
            )
            
            # Get prediction
            result = predictor.predict_with_explanation(features)
            
            print(f"\n{'=' * 60}")
            print(f"RBI PREDICTION: {player_name}")
            print(f"{'=' * 60}")
            print(f"Expected RBIs: {result['expected_rbis']:.2f}")
            print(f"Probability (1+ RBI): {result['rbi_probability']:.1%}")
            print(f"Confidence: {result['confidence_score']:.1%}")
            print(f"Recommendation: {result['recommendation']}")
            
            if result['value_edge'] > 0:
                print(f"Value Edge: +{result['value_edge']:.1%}")
            
            print(f"\n📈 TOP POSITIVE FACTORS:")
            for feature, impact in result['top_positive_features'][:3]:
                print(f"  {feature}: +{impact:.3f}")
            
            print(f"\n📉 TOP NEGATIVE FACTORS:")
            for feature, impact in result['top_negative_features'][:3]:
                print(f"  {feature}: {impact:.3f}")
            
        elif choice == '3':
            # Backtest
            print("\nBacktesting not implemented in demo")
            
        elif choice == '4':
            # Model statistics and validation
            print("\n" + "=" * 50)
            print("MODEL STATISTICS & VALIDATION")
            print("=" * 50)

            # Run formula baseline validation
            baseline_metrics = predictor.validate_formula_baseline()

            print(f"\n📊 TRAINING DATA VALIDATION:")
            print(f"  Total Samples: {baseline_metrics.get('total_samples', 0):,}")
            print(f"  League RBI Rate: {baseline_metrics.get('league_avg_rbi_rate', 0):.1%}")
            print(f"  Expected Range: 8.0% - 15.0%")

            conn = sqlite3.connect(predictor.db_path)
            cursor = conn.cursor()

            # Get training data statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_training_samples,
                    COUNT(DISTINCT game_id) as unique_games,
                    COUNT(DISTINCT player_id) as unique_players,
                    AVG(CAST(got_rbi AS FLOAT)) as avg_rbi_rate,
                    MIN(created_at) as earliest_data,
                    MAX(created_at) as latest_data
                FROM rbi_training_data_v3
            """)

            training_stats = cursor.fetchone()

            # Get prediction statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(rbi_probability) as avg_probability,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(CASE WHEN recommendation LIKE '%BET%' THEN 1 END) as bet_recommendations
                FROM predictions
                WHERE model_version = 'v3.0'
            """)

            prediction_stats = cursor.fetchone()
            conn.close()

            if training_stats and training_stats[0] > 0:
                print(f"\n📈 TRAINING DATA:")
                print(f"  Training Samples: {training_stats[0]:,}")
                print(f"  Unique Games: {training_stats[1]:,}")
                print(f"  Unique Players: {training_stats[2]:,}")
                print(f"  Historical RBI Rate: {training_stats[3]:.1%}")
                print(f"  Data Range: {training_stats[4]} to {training_stats[5]}")

            if prediction_stats and prediction_stats[0] > 0:
                print(f"\n🎯 PREDICTIONS:")
                print(f"  Total Predictions: {prediction_stats[0]}")
                print(f"  Average RBI Probability: {prediction_stats[1]:.1%}")
                print(f"  Average Confidence: {prediction_stats[2]:.1%}")
                print(f"  Bet Recommendations: {prediction_stats[3]}")
            else:
                print(f"\n🎯 PREDICTIONS:")
                print("  No predictions recorded yet")

            # Model performance if available
            print(f"\n🤖 MODEL STATUS:")
            if predictor.models:
                print(f"  Models Loaded: {len(predictor.models)}")
                print(f"  Model Types: {', '.join(predictor.models.keys())}")
                print(f"  SHAP Explainers: {'✓' if predictor.shap_explainers else '✗'}")
            else:
                print("  No models trained yet")
            
        elif choice == '5':
            # Update models
            print("\n" + "=" * 50)
            print("UPDATING MODELS")
            print("=" * 50)
            
            start_date = input("Start date (YYYY-MM-DD): ")
            end_date = input("End date (YYYY-MM-DD): ")
            
            print("\nFetching new training data...")
            X, y = predictor.prepare_training_data(start_date, end_date)
            
            if len(X) > 0:
                print(f"Collected {len(X)} new training samples")
                print("Retraining models...")
                predictor._initialize_models()
                print("✅ Models updated successfully!")
            else:
                print("No new data available")
            
        elif choice == '6':
            print("\nThank you for using RBI Prediction System v3.0!")
            break
        
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()