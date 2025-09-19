"""
MLB Home Run Prediction Model v6.0 - Advanced Feature Engineering & Sportsbook Integration

DESCRIPTION:
A comprehensive MLB home run prediction system that combines multiple data sources to generate
probability estimates and identify value betting opportunities.

INPUTS:
- SportsRadar v8 API: Game schedules, lineups, pitcher assignments
- MLB Stats API: Basic player statistics (HR, AB, OPS, ERA, etc.)
- Statcast data: Advanced metrics (barrel rate, hard hit rate, exit velocity)
- Weather API: Temperature, wind, humidity for ballpark conditions
- The Odds API: Sportsbook odds for home run propositions
- Team statistics: Offensive context and bullpen performance

OUTPUTS:
- Console: Top 10 predictions with odds and value edges
- CSV: Exportable prediction results
- SQLite: Full feature logging for ML training (predictions_log_v6 table)

TABLE SCHEMA (predictions_log_v6):
- run_id (TEXT): Unique identifier for this prediction run
- run_timestamp (TEXT): ISO timestamp of prediction run
- model_version (TEXT): Version identifier (v6.0)
- date (TEXT): Game date
- game_id (TEXT): Unique game identifier
- venue (TEXT): Stadium name
- park_factor (REAL): HR-friendly park multiplier
- lineup_type (TEXT): 'actual' or 'projected'
- lineup_spot (INTEGER): Batting order position
- player_id (TEXT): Unique player identifier
- player_name (TEXT): Player display name
- team (TEXT): Player's team abbreviation
- opponent (TEXT): Opposing team abbreviation
- opp_pitcher_id (TEXT): Opposing pitcher ID
- opp_pitcher_name (TEXT): Opposing pitcher name
- pitcher_hand (TEXT): L/R handedness
- season_hr (INTEGER): Season home runs
- season_ab (INTEGER): Season at-bats
- season_iso (REAL): Season isolated power
- season_ops (REAL): Season OPS
- statcast_barrel_7d (REAL): 7-day barrel rate
- statcast_barrel_30d (REAL): 30-day barrel rate
- statcast_hardhit_7d (REAL): 7-day hard hit rate
- statcast_hardhit_30d (REAL): 30-day hard hit rate
- statcast_ev_30d (REAL): 30-day average exit velocity
- recent_hr_7d (INTEGER): Home runs in last 7 days
- recent_hr_30d (INTEGER): Home runs in last 30 days
- starter_era (REAL): Starting pitcher ERA
- starter_hr9 (REAL): Starting pitcher HR/9
- starter_whip (REAL): Starting pitcher WHIP
- bullpen_hr9 (REAL): Team bullpen HR/9
- bullpen_era (REAL): Team bullpen ERA
- team_ops_season (REAL): Team season OPS
- team_ops_30d (REAL): Team last 30 days OPS
- temp_f (REAL): Temperature in Fahrenheit
- wind_mph (REAL): Wind speed in MPH
- wind_dir_deg (REAL): Wind direction in degrees
- wind_component (REAL): Wind helping (+) or hindering (-)
- humidity (REAL): Relative humidity percentage
- adi_proxy (REAL): Air density index proxy
- model_prob (REAL): Final model probability
- model_implied_odds (REAL): Model implied American odds
- market_odds (REAL): Sportsbook American odds (nullable)
- market_implied_prob (REAL): Market implied probability (nullable)
- value_edge (REAL): Edge over market (nullable)
- data_sources_used (TEXT): JSON of data sources
- errors (TEXT): Any errors encountered (nullable)

USAGE:
python mlb_hr_v6.py [--dry-run] [--no-odds] [--log-csv path]

API KEYS (environment variables):
- SPORTRADAR_API_KEY: SportsRadar v8 API key
- OPENWEATHER_API_KEY: OpenWeather API key (optional)
- ODDS_API_KEY: The Odds API key (optional)
"""

import pandas as pd
import numpy as np
import requests
import json
import sqlite3
import statsapi
import pybaseball as pyb
import uuid
import argparse
import os
import math
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import time

warnings.filterwarnings('ignore')
pyb.cache.enable()

class Config:
    """Centralized configuration"""

    # API Keys (from environment variables)
    SPORTRADAR_KEY = os.getenv('SPORTRADAR_API_KEY', '3uCVRm9PhGc0VMvLafRDZpqcwdWrafLktHzEw3wr')
    OPENWEATHER_KEY = os.getenv('OPENWEATHER_API_KEY', '')
    ODDS_API_KEY = os.getenv('ODDS_API_KEY', '')

    # API Endpoints
    SPORTRADAR_BASE = "https://api.sportradar.com/mlb/trial/v8/en"
    OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5/weather"
    ODDS_API_BASE = "https://api.the-odds-api.com/v4"

    # Database
    CACHE_DB = "mlb_v6_cache.db"

    # Model Parameters
    MODEL_VERSION = "v6.0"
    MAX_HR_PROB = 0.15  # 15% cap

    # Bullpen Weighting (starter vs bullpen influence)
    STARTER_WEIGHT = 0.6
    BULLPEN_WEIGHT = 0.4

    # Value Betting Threshold
    VALUE_THRESHOLD = 0.02  # 2 percentage points

    # Park Factors (HR-friendly multipliers)
    PARK_FACTORS = {
        'Coors Field': 1.40,
        'Great American Ball Park': 1.25,
        'Yankee Stadium': 1.20,
        'Oriole Park at Camden Yards': 1.18,
        'Globe Life Field': 1.15,
        'Citizens Bank Park': 1.12,
        'Fenway Park': 1.10,
        'Guaranteed Rate Field': 1.08,
        'Truist Park': 1.08,
        'Minute Maid Park': 1.05,
        'Chase Field': 1.05,
        'Dodger Stadium': 1.03,
        'Oracle Park': 0.92,
        'T-Mobile Park': 0.90,
        'Petco Park': 0.95,
        'default': 1.00
    }

    # Stadium Coordinates for Weather
    STADIUMS = {
        'Yankee Stadium': {'lat': 40.8296, 'lon': -73.9262, 'elevation': 54, 'domed': False},
        'Fenway Park': {'lat': 42.3467, 'lon': -71.0972, 'elevation': 20, 'domed': False},
        'Coors Field': {'lat': 39.7559, 'lon': -104.9942, 'elevation': 5280, 'domed': False},
        'Dodger Stadium': {'lat': 34.0739, 'lon': -118.2400, 'elevation': 512, 'domed': False},
        'Oracle Park': {'lat': 37.7786, 'lon': -122.3893, 'elevation': 0, 'domed': False},
        'Chase Field': {'lat': 33.4453, 'lon': -112.0667, 'elevation': 1100, 'domed': True},
        'Tropicana Field': {'lat': 27.7682, 'lon': -82.6534, 'elevation': 15, 'domed': True},
        'default': {'lat': 40.0, 'lon': -95.0, 'elevation': 500, 'domed': False}
    }

    # League Averages (fallback values)
    LEAGUE_AVG = {
        'hr_rate': 0.033,
        'barrel_rate': 0.08,
        'hard_hit_rate': 0.35,
        'exit_velocity': 88.0,
        'era': 4.50,
        'hr_per_9': 1.3,
        'whip': 1.35,
        'ops': 0.740,
        'iso': 0.165
    }

class DatabaseManager:
    """Handles SQLite database operations"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        """Create necessary tables"""
        cursor = self.conn.cursor()

        # Cache table for API responses
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_cache (
                key TEXT PRIMARY KEY,
                data TEXT,
                cached_at TIMESTAMP,
                expires_at TIMESTAMP
            )
        ''')

        # Predictions log table for ML training
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions_log_v6 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                run_timestamp TEXT,
                model_version TEXT,
                date TEXT,
                game_id TEXT,
                venue TEXT,
                park_factor REAL,
                lineup_type TEXT,
                lineup_spot INTEGER,
                player_id TEXT,
                player_name TEXT,
                team TEXT,
                opponent TEXT,
                opp_pitcher_id TEXT,
                opp_pitcher_name TEXT,
                pitcher_hand TEXT,
                season_hr INTEGER,
                season_ab INTEGER,
                season_iso REAL,
                season_ops REAL,
                statcast_barrel_7d REAL,
                statcast_barrel_30d REAL,
                statcast_hardhit_7d REAL,
                statcast_hardhit_30d REAL,
                statcast_ev_30d REAL,
                recent_hr_7d INTEGER,
                recent_hr_30d INTEGER,
                starter_era REAL,
                starter_hr9 REAL,
                starter_whip REAL,
                bullpen_hr9 REAL,
                bullpen_era REAL,
                team_ops_season REAL,
                team_ops_30d REAL,
                temp_f REAL,
                wind_mph REAL,
                wind_dir_deg REAL,
                wind_component REAL,
                humidity REAL,
                adi_proxy REAL,
                model_prob REAL,
                model_implied_odds REAL,
                market_odds REAL,
                market_implied_prob REAL,
                value_edge REAL,
                data_sources_used TEXT,
                errors TEXT
            )
        ''')

        self.conn.commit()

    def get_cached(self, key: str) -> Optional[Dict]:
        """Get cached data if not expired"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT data FROM api_cache WHERE key = ? AND expires_at > ?',
            (key, datetime.now().isoformat())
        )
        result = cursor.fetchone()
        return json.loads(result[0]) if result else None

    def set_cache(self, key: str, data: Dict, ttl_hours: int = 24):
        """Cache data with TTL"""
        cursor = self.conn.cursor()
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        cursor.execute(
            'INSERT OR REPLACE INTO api_cache (key, data, cached_at, expires_at) VALUES (?, ?, ?, ?)',
            (key, json.dumps(data), datetime.now().isoformat(), expires_at.isoformat())
        )
        self.conn.commit()

    def log_prediction(self, prediction_data: Dict):
        """Log a prediction row for ML training"""
        cursor = self.conn.cursor()

        columns = list(prediction_data.keys())
        placeholders = ', '.join(['?' for _ in columns])
        values = [prediction_data[col] for col in columns]

        cursor.execute(
            f'INSERT INTO predictions_log_v6 ({", ".join(columns)}) VALUES ({placeholders})',
            values
        )
        self.conn.commit()

    def export_predictions_to_csv(self, output_path: str, run_id: Optional[str] = None):
        """Export predictions log to CSV"""
        query = 'SELECT * FROM predictions_log_v6'
        params = []

        if run_id:
            query += ' WHERE run_id = ?'
            params.append(run_id)

        df = pd.read_sql_query(query, self.conn, params=params)
        df.to_csv(output_path, index=False)
        print(f"[INFO] Exported {len(df)} prediction logs to {output_path}")

class SportsRadarClient:
    """Handles SportsRadar API calls"""

    def __init__(self, config: Config, db: DatabaseManager):
        self.config = config
        self.db = db
        self.session = requests.Session()

    def get_todays_games_with_lineups(self) -> List[Dict]:
        """Get today's games with lineups"""
        print("\n[INFO] Fetching games from SportsRadar v8...")

        cache_key = f"sportradar_games_{datetime.now().strftime('%Y-%m-%d')}"
        cached = self.db.get_cached(cache_key)
        if cached:
            print("[INFO] Using cached game data")
            return cached

        try:
            # Get schedule
            date_str = datetime.now().strftime('%Y/%m/%d')
            schedule_url = f"{self.config.SPORTRADAR_BASE}/games/{date_str}/schedule.json"

            response = self.session.get(
                schedule_url,
                params={'api_key': self.config.SPORTRADAR_KEY},
                timeout=10
            )

            if response.status_code != 200:
                print(f"[ERROR] Schedule API error: {response.status_code}")
                return []

            schedule_data = response.json()
            all_games = schedule_data.get('games', [])
            scheduled_games = [g for g in all_games if g.get('status') in ['scheduled', 'created']]

            print(f"[INFO] Found {len(scheduled_games)} scheduled games")

            games_with_lineups = []

            for game in scheduled_games:
                time.sleep(1.2)  # Rate limiting

                game_info = self._get_game_details(game)
                if game_info:
                    games_with_lineups.append(game_info)

            # Cache for 1 hour (lineups change frequently)
            self.db.set_cache(cache_key, games_with_lineups, ttl_hours=1)

            print(f"[INFO] Retrieved {len(games_with_lineups)} games with lineups")
            return games_with_lineups

        except Exception as e:
            print(f"[ERROR] SportsRadar error: {e}")
            return []

    def _get_game_details(self, game: Dict) -> Optional[Dict]:
        """Get detailed game information with lineups"""
        try:
            game_id = game['id']
            home_team = game['home']['name']
            away_team = game['away']['name']

            summary_url = f"{self.config.SPORTRADAR_BASE}/games/{game_id}/summary.json"

            response = self.session.get(
                summary_url,
                params={'api_key': self.config.SPORTRADAR_KEY},
                timeout=10
            )

            if response.status_code != 200:
                return None

            summary_data = response.json()
            game_data = summary_data.get('game', {})

            game_info = {
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'home_team_abbr': self._get_team_abbr(home_team),
                'away_team_abbr': self._get_team_abbr(away_team),
                'venue': game_data.get('venue', {}).get('name', ''),
                'home_lineup': [],
                'away_lineup': [],
                'home_pitcher': None,
                'away_pitcher': None
            }

            # Process home team
            if 'home' in game_data:
                home = game_data['home']
                if 'probable_pitcher' in home:
                    pitcher = home['probable_pitcher']
                    game_info['home_pitcher'] = {
                        'name': pitcher.get('full_name', pitcher.get('last_name', 'TBD')),
                        'id': pitcher.get('id'),
                        'throws': pitcher.get('throws', 'R')
                    }

                if 'lineup' in home and 'players' in home:
                    game_info['home_lineup'] = self._extract_lineup(home['lineup'], home['players'])

            # Process away team
            if 'away' in game_data:
                away = game_data['away']
                if 'probable_pitcher' in away:
                    pitcher = away['probable_pitcher']
                    game_info['away_pitcher'] = {
                        'name': pitcher.get('full_name', pitcher.get('last_name', 'TBD')),
                        'id': pitcher.get('id'),
                        'throws': pitcher.get('throws', 'R')
                    }

                if 'lineup' in away and 'players' in away:
                    game_info['away_lineup'] = self._extract_lineup(away['lineup'], away['players'])

            # Return if we have meaningful data
            if (game_info['home_lineup'] or game_info['away_lineup']) or \
               (game_info['home_pitcher'] or game_info['away_pitcher']):
                print(f"[SUCCESS] {away_team} @ {home_team}: Data available")
                return game_info

            return None

        except Exception as e:
            print(f"[WARNING] Error processing {away_team} @ {home_team}: {e}")
            return None

    def _extract_lineup(self, lineup_data: List, players_data: List) -> List[Dict]:
        """Extract player names from lineup IDs"""
        lineup = []

        # Create player ID to info mapping
        player_map = {}
        for player_info in players_data:
            if isinstance(player_info, dict):
                player_id = player_info.get('id')
                if player_id:
                    player_map[player_id] = {
                        'name': player_info.get('full_name', 'Unknown'),
                        'position': player_info.get('position', ''),
                        'id': player_id
                    }

        # Extract lineup with names
        for entry in lineup_data:
            if entry.get('position') == 1:  # Skip pitcher
                continue

            player_id = entry.get('id')
            if player_id and player_id in player_map:
                player_info = player_map[player_id]
                lineup.append({
                    'name': player_info['name'],
                    'id': player_info['id'],
                    'order': entry.get('order', 0),
                    'position': entry.get('position', 0)
                })

        # Sort by batting order
        lineup.sort(key=lambda x: x['order'])
        return lineup[:9]  # Return top 9 batters

    def _get_team_abbr(self, team_name: str) -> str:
        """Get team abbreviation"""
        abbr_map = {
            'Yankees': 'NYY', 'Red Sox': 'BOS', 'Rays': 'TB', 'Orioles': 'BAL', 'Blue Jays': 'TOR',
            'Guardians': 'CLE', 'Twins': 'MIN', 'White Sox': 'CHW', 'Royals': 'KC', 'Tigers': 'DET',
            'Astros': 'HOU', 'Athletics': 'OAK', 'Rangers': 'TEX', 'Angels': 'LAA', 'Mariners': 'SEA',
            'Mets': 'NYM', 'Braves': 'ATL', 'Phillies': 'PHI', 'Marlins': 'MIA', 'Nationals': 'WSN',
            'Brewers': 'MIL', 'Cardinals': 'STL', 'Cubs': 'CHC', 'Reds': 'CIN', 'Pirates': 'PIT',
            'Dodgers': 'LAD', 'Giants': 'SF', 'Padres': 'SD', 'Rockies': 'COL', 'Diamondbacks': 'ARI'
        }

        for key, value in abbr_map.items():
            if key in team_name:
                return value
        return team_name[:3].upper()

class StatcastClient:
    """Handles Statcast data collection"""

    def __init__(self, config: Config, db: DatabaseManager):
        self.config = config
        self.db = db

    def get_player_statcast_data(self, player_name: str) -> Dict:
        """Get Statcast rolling metrics for a player"""
        cache_key = f"statcast_{player_name}_{datetime.now().strftime('%Y-%m-%d')}"
        cached = self.db.get_cached(cache_key)
        if cached:
            return cached

        try:
            # Get recent date range
            end_date = datetime.now()
            start_date_7d = end_date - timedelta(days=7)
            start_date_30d = end_date - timedelta(days=30)

            # Try to get Statcast data
            player_data_30d = pyb.statcast_batter(
                start_dt=start_date_30d.strftime('%Y-%m-%d'),
                end_dt=end_date.strftime('%Y-%m-%d'),
                player_id=None  # Will need to resolve player ID
            )

            if player_data_30d.empty:
                return self._get_default_statcast_data()

            # Calculate metrics
            data_7d = player_data_30d[player_data_30d['game_date'] >= start_date_7d.strftime('%Y-%m-%d')]

            result = {
                'barrel_rate_7d': self._calculate_barrel_rate(data_7d),
                'barrel_rate_30d': self._calculate_barrel_rate(player_data_30d),
                'hard_hit_rate_7d': self._calculate_hard_hit_rate(data_7d),
                'hard_hit_rate_30d': self._calculate_hard_hit_rate(player_data_30d),
                'avg_exit_velocity_30d': player_data_30d['launch_speed'].mean() if not player_data_30d.empty else self.config.LEAGUE_AVG['exit_velocity'],
                'recent_hr_7d': len(data_7d[data_7d['events'] == 'home_run']),
                'recent_hr_30d': len(player_data_30d[player_data_30d['events'] == 'home_run'])
            }

            self.db.set_cache(cache_key, result, ttl_hours=6)
            return result

        except Exception as e:
            print(f"[WARNING] Statcast error for {player_name}: {e}")
            return self._get_default_statcast_data()

    def _get_default_statcast_data(self) -> Dict:
        """Return league average Statcast data"""
        return {
            'barrel_rate_7d': self.config.LEAGUE_AVG['barrel_rate'],
            'barrel_rate_30d': self.config.LEAGUE_AVG['barrel_rate'],
            'hard_hit_rate_7d': self.config.LEAGUE_AVG['hard_hit_rate'],
            'hard_hit_rate_30d': self.config.LEAGUE_AVG['hard_hit_rate'],
            'avg_exit_velocity_30d': self.config.LEAGUE_AVG['exit_velocity'],
            'recent_hr_7d': 0,
            'recent_hr_30d': 1
        }

    def _calculate_barrel_rate(self, df: pd.DataFrame) -> float:
        """Calculate barrel rate from Statcast data"""
        if df.empty:
            return self.config.LEAGUE_AVG['barrel_rate']

        # Simplified barrel calculation: launch_speed >= 98 and optimal launch angle
        barrels = df[(df['launch_speed'] >= 98) & (df['launch_angle'].between(26, 30))]
        return len(barrels) / len(df) if len(df) > 0 else self.config.LEAGUE_AVG['barrel_rate']

    def _calculate_hard_hit_rate(self, df: pd.DataFrame) -> float:
        """Calculate hard hit rate (95+ mph)"""
        if df.empty:
            return self.config.LEAGUE_AVG['hard_hit_rate']

        hard_hits = df[df['launch_speed'] >= 95]
        return len(hard_hits) / len(df) if len(df) > 0 else self.config.LEAGUE_AVG['hard_hit_rate']

class WeatherClient:
    """Handles weather data collection"""

    def __init__(self, config: Config, db: DatabaseManager):
        self.config = config
        self.db = db
        self.session = requests.Session()

    def get_stadium_weather(self, venue: str) -> Dict:
        """Get weather conditions for a stadium"""
        if not self.config.OPENWEATHER_KEY:
            return self._get_neutral_weather()

        stadium_info = self.config.STADIUMS.get(venue, self.config.STADIUMS['default'])

        # Domed stadiums have neutral weather
        if stadium_info.get('domed', False):
            return self._get_domed_weather()

        cache_key = f"weather_{venue}_{datetime.now().strftime('%Y-%m-%d-%H')}"
        cached = self.db.get_cached(cache_key)
        if cached:
            return cached

        try:
            response = self.session.get(
                self.config.OPENWEATHER_BASE,
                params={
                    'lat': stadium_info['lat'],
                    'lon': stadium_info['lon'],
                    'appid': self.config.OPENWEATHER_KEY,
                    'units': 'imperial'
                },
                timeout=5
            )

            if response.status_code != 200:
                return self._get_neutral_weather()

            data = response.json()

            # Calculate wind component (positive = helping, negative = hindering)
            wind_speed = data.get('wind', {}).get('speed', 0)
            wind_dir = data.get('wind', {}).get('deg', 0)
            wind_component = self._calculate_wind_component(wind_speed, wind_dir)

            # Calculate air density proxy
            temp_f = data['main']['temp']
            humidity = data['main']['humidity']
            elevation = stadium_info['elevation']
            adi_proxy = self._calculate_air_density_index(temp_f, humidity, elevation)

            result = {
                'temp_f': temp_f,
                'wind_mph': wind_speed,
                'wind_dir_deg': wind_dir,
                'wind_component': wind_component,
                'humidity': humidity,
                'adi_proxy': adi_proxy
            }

            self.db.set_cache(cache_key, result, ttl_hours=1)
            return result

        except Exception as e:
            print(f"[WARNING] Weather error for {venue}: {e}")
            return self._get_neutral_weather()

    def _get_neutral_weather(self) -> Dict:
        """Return neutral weather conditions"""
        return {
            'temp_f': 75.0,
            'wind_mph': 5.0,
            'wind_dir_deg': 0.0,
            'wind_component': 0.0,
            'humidity': 50.0,
            'adi_proxy': 1.0
        }

    def _get_domed_weather(self) -> Dict:
        """Return controlled conditions for domed stadiums"""
        return {
            'temp_f': 72.0,
            'wind_mph': 0.0,
            'wind_dir_deg': 0.0,
            'wind_component': 0.0,
            'humidity': 45.0,
            'adi_proxy': 1.0
        }

    def _calculate_wind_component(self, speed: float, direction: float) -> float:
        """Calculate wind component relative to home run direction"""
        # Assume home runs typically go to center field (direction 0)
        # Wind blowing out (direction ~0) helps, wind blowing in (direction ~180) hurts
        radians = math.radians(direction)
        component = speed * math.cos(radians)
        return component

    def _calculate_air_density_index(self, temp_f: float, humidity: float, elevation: float) -> float:
        """Calculate air density proxy (higher = denser air = shorter flights)"""
        # Simplified air density calculation
        # Higher temperature = lower density (helps HRs)
        # Higher humidity = lower density (helps HRs)
        # Higher elevation = lower density (helps HRs)

        temp_factor = 1.0 - ((temp_f - 70) * 0.001)  # Cooler air is denser
        humidity_factor = 1.0 - (humidity * 0.001)   # Dry air is denser
        elevation_factor = 1.0 - (elevation * 0.00005)  # Sea level is denser

        return max(0.5, temp_factor * humidity_factor * elevation_factor)

class OddsClient:
    """Handles sportsbook odds collection"""

    def __init__(self, config: Config, db: DatabaseManager):
        self.config = config
        self.db = db
        self.session = requests.Session()

    def get_hr_odds(self) -> Dict[str, Dict]:
        """Get home run prop odds for today's games"""
        if not self.config.ODDS_API_KEY:
            print("[INFO] No odds API key provided, skipping odds")
            return {}

        cache_key = f"odds_hr_{datetime.now().strftime('%Y-%m-%d')}"
        cached = self.db.get_cached(cache_key)
        if cached:
            print("[INFO] Using cached odds data")
            return cached

        try:
            # Get MLB odds for home run props
            response = self.session.get(
                f"{self.config.ODDS_API_BASE}/sports/baseball_mlb/odds",
                params={
                    'api_key': self.config.ODDS_API_KEY,
                    'regions': 'us',
                    'markets': 'batter_home_runs',
                    'oddsFormat': 'american'
                },
                timeout=10
            )

            if response.status_code != 200:
                print(f"[WARNING] Odds API error: {response.status_code}")
                return {}

            data = response.json()

            # Process odds data
            odds_dict = {}
            for game in data:
                for bookmaker in game.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'batter_home_runs':
                            for outcome in market.get('outcomes', []):
                                player_name = outcome.get('description', '')
                                odds_value = outcome.get('price')

                                if player_name and odds_value:
                                    normalized_name = self._normalize_player_name(player_name)
                                    odds_dict[normalized_name] = {
                                        'american_odds': odds_value,
                                        'implied_prob': self._american_to_implied_prob(odds_value)
                                    }

            self.db.set_cache(cache_key, odds_dict, ttl_hours=6)
            print(f"[INFO] Retrieved odds for {len(odds_dict)} players")
            return odds_dict

        except Exception as e:
            print(f"[WARNING] Odds collection error: {e}")
            return {}

    def _normalize_player_name(self, name: str) -> str:
        """Normalize player name for matching"""
        # Simple normalization - remove extra spaces, convert to title case
        return ' '.join(name.strip().split()).title()

    def _american_to_implied_prob(self, american_odds: int) -> float:
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return (-american_odds) / ((-american_odds) + 100)

class StatsCollector:
    """Collects basic player and team statistics"""

    def __init__(self, config: Config, db: DatabaseManager):
        self.config = config
        self.db = db

    def get_batter_stats(self, player_name: str, player_id: str = None) -> Dict:
        """Get batter statistics with caching"""
        cache_key = f"batter_stats_{player_name}_{datetime.now().strftime('%Y-%m-%d')}"
        cached = self.db.get_cached(cache_key)
        if cached:
            return cached

        try:
            if not player_id:
                search = statsapi.lookup_player(player_name)
                if not search:
                    return self._get_default_batter_stats()
                player_id = search[0]['id']

            stats = statsapi.player_stat_data(player_id, group='hitting', type='season')

            if stats.get('stats'):
                current = stats['stats'][0]['stats']

                result = {
                    'season_hr': int(current.get('homeRuns', 0)),
                    'season_ab': int(current.get('atBats', 1)),
                    'season_ops': float(current.get('ops', self.config.LEAGUE_AVG['ops'])),
                    'season_iso': float(current.get('slg', 0.400)) - float(current.get('avg', 0.250))
                }

                self.db.set_cache(cache_key, result, ttl_hours=24)
                return result

        except Exception as e:
            print(f"[WARNING] Stats error for {player_name}: {e}")

        return self._get_default_batter_stats()

    def get_pitcher_stats(self, pitcher_name: str, pitcher_id: str = None) -> Dict:
        """Get pitcher statistics"""
        cache_key = f"pitcher_stats_{pitcher_name}_{datetime.now().strftime('%Y-%m-%d')}"
        cached = self.db.get_cached(cache_key)
        if cached:
            return cached

        try:
            if not pitcher_id:
                search = statsapi.lookup_player(pitcher_name)
                if not search:
                    return self._get_default_pitcher_stats()
                pitcher_id = search[0]['id']

            stats = statsapi.player_stat_data(pitcher_id, group='pitching', type='season')

            if stats.get('stats'):
                current = stats['stats'][0]['stats']

                result = {
                    'starter_era': float(current.get('era', self.config.LEAGUE_AVG['era'])),
                    'starter_hr9': float(current.get('homeRunsPer9', self.config.LEAGUE_AVG['hr_per_9'])),
                    'starter_whip': float(current.get('whip', self.config.LEAGUE_AVG['whip']))
                }

                self.db.set_cache(cache_key, result, ttl_hours=24)
                return result

        except Exception as e:
            print(f"[WARNING] Pitcher stats error for {pitcher_name}: {e}")

        return self._get_default_pitcher_stats()

    def get_team_stats(self, team_abbr: str) -> Dict:
        """Get team offensive and bullpen statistics"""
        cache_key = f"team_stats_{team_abbr}_{datetime.now().strftime('%Y-%m-%d')}"
        cached = self.db.get_cached(cache_key)
        if cached:
            return cached

        try:
            # This would need more complex implementation to get team stats
            # For now, return defaults
            result = {
                'team_ops_season': self.config.LEAGUE_AVG['ops'],
                'team_ops_30d': self.config.LEAGUE_AVG['ops'],
                'bullpen_hr9': self.config.LEAGUE_AVG['hr_per_9'],
                'bullpen_era': self.config.LEAGUE_AVG['era']
            }

            self.db.set_cache(cache_key, result, ttl_hours=24)
            return result

        except Exception as e:
            print(f"[WARNING] Team stats error for {team_abbr}: {e}")
            return {
                'team_ops_season': self.config.LEAGUE_AVG['ops'],
                'team_ops_30d': self.config.LEAGUE_AVG['ops'],
                'bullpen_hr9': self.config.LEAGUE_AVG['hr_per_9'],
                'bullpen_era': self.config.LEAGUE_AVG['era']
            }

    def _get_default_batter_stats(self) -> Dict:
        """Default batter statistics"""
        return {
            'season_hr': 15,
            'season_ab': 400,
            'season_ops': self.config.LEAGUE_AVG['ops'],
            'season_iso': self.config.LEAGUE_AVG['iso']
        }

    def _get_default_pitcher_stats(self) -> Dict:
        """Default pitcher statistics"""
        return {
            'starter_era': self.config.LEAGUE_AVG['era'],
            'starter_hr9': self.config.LEAGUE_AVG['hr_per_9'],
            'starter_whip': self.config.LEAGUE_AVG['whip']
        }

class HRProbabilityModel:
    """Advanced home run probability model"""

    def __init__(self, config: Config):
        self.config = config

    def calculate_probability(self, features: Dict) -> Tuple[float, float]:
        """Calculate HR probability and implied odds"""

        # Base rate
        base_rate = self.config.LEAGUE_AVG['hr_rate']

        # Batter power components
        batter_mult = self._calculate_batter_multiplier(features)

        # Pitcher vulnerability
        pitcher_mult = self._calculate_pitcher_multiplier(features)

        # Park factor
        park_mult = features.get('park_factor', 1.0)

        # Weather impact
        weather_mult = self._calculate_weather_multiplier(features)

        # Team context
        team_mult = self._calculate_team_multiplier(features)

        # Statcast rolling form
        statcast_mult = self._calculate_statcast_multiplier(features)

        # Combine all factors
        total_mult = (batter_mult * pitcher_mult * park_mult *
                     weather_mult * team_mult * statcast_mult)

        # Calculate probability
        hr_prob = base_rate * total_mult

        # Apply cap
        hr_prob = min(hr_prob, self.config.MAX_HR_PROB)

        # Convert to American odds
        implied_odds = self._prob_to_american_odds(hr_prob)

        return hr_prob, implied_odds

    def _calculate_batter_multiplier(self, features: Dict) -> float:
        """Calculate batter power multiplier"""
        season_hr = features.get('season_hr', 15)
        season_ab = features.get('season_ab', 400)
        season_iso = features.get('season_iso', self.config.LEAGUE_AVG['iso'])

        # HR rate factor
        hr_rate = season_hr / max(season_ab, 100)
        hr_mult = hr_rate / self.config.LEAGUE_AVG['hr_rate']
        hr_mult = np.clip(hr_mult, 0.3, 3.0)

        # ISO factor
        iso_mult = 1.0
        if season_iso > 0.250:
            iso_mult = 1.35
        elif season_iso > 0.200:
            iso_mult = 1.15
        elif season_iso < 0.120:
            iso_mult = 0.7

        return hr_mult * iso_mult

    def _calculate_pitcher_multiplier(self, features: Dict) -> float:
        """Calculate pitcher vulnerability with bullpen context"""
        starter_hr9 = features.get('starter_hr9', self.config.LEAGUE_AVG['hr_per_9'])
        starter_era = features.get('starter_era', self.config.LEAGUE_AVG['era'])
        bullpen_hr9 = features.get('bullpen_hr9', self.config.LEAGUE_AVG['hr_per_9'])

        # Blend starter and bullpen (weighted)
        blended_hr9 = (starter_hr9 * self.config.STARTER_WEIGHT +
                      bullpen_hr9 * self.config.BULLPEN_WEIGHT)

        # HR/9 factor
        if blended_hr9 > 1.5:
            hr9_mult = 1.3
        elif blended_hr9 > 1.2:
            hr9_mult = 1.1
        elif blended_hr9 < 0.9:
            hr9_mult = 0.7
        else:
            hr9_mult = 1.0

        # ERA factor
        era_mult = 1.0
        if starter_era > 5.0:
            era_mult = 1.15
        elif starter_era < 3.0:
            era_mult = 0.85

        return hr9_mult * era_mult

    def _calculate_weather_multiplier(self, features: Dict) -> float:
        """Calculate weather impact on HR probability"""
        temp_f = features.get('temp_f', 75.0)
        wind_component = features.get('wind_component', 0.0)
        adi_proxy = features.get('adi_proxy', 1.0)

        # Temperature factor (warmer helps)
        temp_mult = 1.0 + ((temp_f - 70) * 0.005)
        temp_mult = np.clip(temp_mult, 0.9, 1.15)

        # Wind factor
        wind_mult = 1.0 + (wind_component * 0.01)
        wind_mult = np.clip(wind_mult, 0.85, 1.20)

        # Air density factor (thinner air helps)
        density_mult = 2.0 - adi_proxy  # Lower ADI = thinner air = helps HRs
        density_mult = np.clip(density_mult, 0.9, 1.15)

        return temp_mult * wind_mult * density_mult

    def _calculate_team_multiplier(self, features: Dict) -> float:
        """Calculate team offensive context"""
        team_ops = features.get('team_ops_30d', self.config.LEAGUE_AVG['ops'])

        # Modest team factor
        team_mult = 0.95 + ((team_ops - self.config.LEAGUE_AVG['ops']) * 0.5)
        return np.clip(team_mult, 0.9, 1.1)

    def _calculate_statcast_multiplier(self, features: Dict) -> float:
        """Calculate Statcast rolling form multiplier"""
        barrel_30d = features.get('statcast_barrel_30d', self.config.LEAGUE_AVG['barrel_rate'])
        hard_hit_30d = features.get('statcast_hardhit_30d', self.config.LEAGUE_AVG['hard_hit_rate'])
        recent_hr_7d = features.get('recent_hr_7d', 0)

        # Barrel rate factor
        barrel_mult = barrel_30d / self.config.LEAGUE_AVG['barrel_rate']
        barrel_mult = np.clip(barrel_mult, 0.7, 1.4)

        # Hard hit rate factor
        hard_hit_mult = hard_hit_30d / self.config.LEAGUE_AVG['hard_hit_rate']
        hard_hit_mult = np.clip(hard_hit_mult, 0.8, 1.3)

        # Recent form (hot streak bonus)
        hot_mult = 1.0
        if recent_hr_7d >= 2:
            hot_mult = 1.15
        elif recent_hr_7d >= 1:
            hot_mult = 1.05

        return barrel_mult * hard_hit_mult * hot_mult

    def _prob_to_american_odds(self, prob: float) -> float:
        """Convert probability to American odds"""
        if prob <= 0:
            return 10000  # Very high positive odds
        elif prob >= 1:
            return -10000  # Very negative odds
        elif prob < 0.5:
            return round((1 / prob - 1) * 100)
        else:
            return round(-prob / (1 - prob) * 100)

class MLBHomeRunPredictorV6:
    """Main prediction system v6.0"""

    def __init__(self, enable_odds: bool = True):
        self.config = Config()
        self.db = DatabaseManager(self.config.CACHE_DB)
        self.sportradar = SportsRadarClient(self.config, self.db)
        self.statcast = StatcastClient(self.config, self.db)
        self.weather = WeatherClient(self.config, self.db)
        self.odds = OddsClient(self.config, self.db) if enable_odds else None
        self.stats = StatsCollector(self.config, self.db)
        self.model = HRProbabilityModel(self.config)
        self.run_id = str(uuid.uuid4())
        self.run_timestamp = datetime.now().isoformat()

    def run_predictions(self) -> pd.DataFrame:
        """Generate comprehensive HR predictions"""
        print("=" * 80)
        print(f"MLB HOME RUN PREDICTIONS {self.config.MODEL_VERSION}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Run ID: {self.run_id}")
        print("=" * 80)

        # Get games and lineups
        games = self.sportradar.get_todays_games_with_lineups()
        if not games:
            print("\n[ERROR] No games available")
            return pd.DataFrame()

        # Get odds if enabled
        odds_data = {}
        if self.odds:
            odds_data = self.odds.get_hr_odds()

        print(f"\n[INFO] Processing {len(games)} games")
        print(f"[INFO] Odds available for {len(odds_data)} players")

        all_predictions = []

        for game in games:
            # Determine lineup type
            lineup_type = "actual" if (game['home_lineup'] and game['away_lineup']) else "projected"

            # Get weather for venue
            weather_data = self.weather.get_stadium_weather(game['venue'])

            # Get team stats
            home_team_stats = self.stats.get_team_stats(game['home_team_abbr'])
            away_team_stats = self.stats.get_team_stats(game['away_team_abbr'])

            # Process home batters vs away pitcher
            if game['home_lineup'] and game['away_pitcher']:
                away_pitcher_stats = self.stats.get_pitcher_stats(
                    game['away_pitcher']['name'],
                    game['away_pitcher'].get('id')
                )

                for batter in game['home_lineup']:
                    prediction = self._create_prediction(
                        batter, game['away_pitcher'], game, lineup_type,
                        weather_data, home_team_stats, away_pitcher_stats,
                        odds_data, is_home=True
                    )
                    if prediction:
                        all_predictions.append(prediction)

            # Process away batters vs home pitcher
            if game['away_lineup'] and game['home_pitcher']:
                home_pitcher_stats = self.stats.get_pitcher_stats(
                    game['home_pitcher']['name'],
                    game['home_pitcher'].get('id')
                )

                for batter in game['away_lineup']:
                    prediction = self._create_prediction(
                        batter, game['home_pitcher'], game, lineup_type,
                        weather_data, away_team_stats, home_pitcher_stats,
                        odds_data, is_home=False
                    )
                    if prediction:
                        all_predictions.append(prediction)

        if not all_predictions:
            print("\n[ERROR] No valid predictions generated")
            return pd.DataFrame()

        # Create DataFrame and sort
        df = pd.DataFrame(all_predictions)
        df = df.sort_values('model_prob', ascending=False).reset_index(drop=True)

        # Display results
        self._display_results(df, odds_data)

        # Save to CSV
        self._save_results(df)

        return df.head(10)

    def _create_prediction(self, batter: Dict, pitcher: Dict, game: Dict,
                          lineup_type: str, weather_data: Dict, team_stats: Dict,
                          pitcher_stats: Dict, odds_data: Dict, is_home: bool) -> Optional[Dict]:
        """Create a single prediction with full feature logging"""

        try:
            # Get batter stats
            batter_stats = self.stats.get_batter_stats(batter['name'], batter.get('id'))

            # Get Statcast data
            statcast_data = self.statcast.get_player_statcast_data(batter['name'])

            # Prepare features
            features = {
                # Basic info
                'park_factor': self.config.PARK_FACTORS.get(game['venue'], 1.0),

                # Batter features
                'season_hr': batter_stats['season_hr'],
                'season_ab': batter_stats['season_ab'],
                'season_iso': batter_stats['season_iso'],
                'season_ops': batter_stats['season_ops'],

                # Statcast features
                'statcast_barrel_7d': statcast_data['barrel_rate_7d'],
                'statcast_barrel_30d': statcast_data['barrel_rate_30d'],
                'statcast_hardhit_7d': statcast_data['hard_hit_rate_7d'],
                'statcast_hardhit_30d': statcast_data['hard_hit_rate_30d'],
                'statcast_ev_30d': statcast_data['avg_exit_velocity_30d'],
                'recent_hr_7d': statcast_data['recent_hr_7d'],
                'recent_hr_30d': statcast_data['recent_hr_30d'],

                # Pitcher features
                'starter_era': pitcher_stats['starter_era'],
                'starter_hr9': pitcher_stats['starter_hr9'],
                'starter_whip': pitcher_stats['starter_whip'],

                # Team features
                'bullpen_hr9': team_stats['bullpen_hr9'],
                'bullpen_era': team_stats['bullpen_era'],
                'team_ops_season': team_stats['team_ops_season'],
                'team_ops_30d': team_stats['team_ops_30d'],

                # Weather features
                'temp_f': weather_data['temp_f'],
                'wind_mph': weather_data['wind_mph'],
                'wind_dir_deg': weather_data['wind_dir_deg'],
                'wind_component': weather_data['wind_component'],
                'humidity': weather_data['humidity'],
                'adi_proxy': weather_data['adi_proxy']
            }

            # Calculate probability
            model_prob, model_implied_odds = self.model.calculate_probability(features)

            # Get market odds
            normalized_name = batter['name'].strip().title()
            market_data = odds_data.get(normalized_name, {})
            market_odds = market_data.get('american_odds')
            market_implied_prob = market_data.get('implied_prob')
            value_edge = None

            if market_implied_prob:
                value_edge = model_prob - market_implied_prob

            # Prepare prediction row
            prediction_row = {
                # Run metadata
                'run_id': self.run_id,
                'run_timestamp': self.run_timestamp,
                'model_version': self.config.MODEL_VERSION,

                # Game info
                'date': datetime.now().strftime('%Y-%m-%d'),
                'game_id': game['game_id'],
                'venue': game['venue'],
                'park_factor': features['park_factor'],
                'lineup_type': lineup_type,
                'lineup_spot': batter.get('order', 0),

                # Player info
                'player_id': batter.get('id', ''),
                'player_name': batter['name'],
                'team': game['home_team_abbr'] if is_home else game['away_team_abbr'],
                'opponent': game['away_team_abbr'] if is_home else game['home_team_abbr'],
                'opp_pitcher_id': pitcher.get('id', ''),
                'opp_pitcher_name': pitcher['name'],
                'pitcher_hand': pitcher.get('throws', 'R'),

                # All features
                **{k: v for k, v in features.items()},

                # Model outputs
                'model_prob': model_prob,
                'model_implied_odds': model_implied_odds,
                'market_odds': market_odds,
                'market_implied_prob': market_implied_prob,
                'value_edge': value_edge,

                # Metadata
                'data_sources_used': json.dumps({
                    'lineup': 'sportradar',
                    'stats': 'mlb_api',
                    'statcast': 'pybaseball',
                    'weather': 'openweather' if self.config.OPENWEATHER_KEY else 'default',
                    'odds': 'odds_api' if market_odds else 'none'
                }),
                'errors': None
            }

            # Log to database
            self.db.log_prediction(prediction_row)

            return prediction_row

        except Exception as e:
            print(f"[WARNING] Error creating prediction for {batter['name']}: {e}")
            return None

    def _display_results(self, df: pd.DataFrame, odds_data: Dict):
        """Display prediction results"""
        top_10 = df.head(10)

        print("\n" + "=" * 95)
        print("TOP 10 HOME RUN PREDICTIONS")
        print("=" * 95)
        print(f"{'Rank':<4} {'Hitter':<18} {'Pitcher':<18} {'Teams':<10} {'Prob%':<6} {'Odds':<8} {'Edge':<6} {'Value':<5}")
        print("-" * 95)

        for idx, row in top_10.iterrows():
            # Format values
            prob_pct = f"{row['model_prob']*100:.1f}%"

            odds_str = "N/A"
            if row['market_odds']:
                odds_str = f"{int(row['market_odds']):+d}"

            edge_str = "N/A"
            value_flag = ""
            if row['value_edge'] is not None:
                edge_pct = row['value_edge'] * 100
                edge_str = f"{edge_pct:+.1f}%"
                if row['value_edge'] >= self.config.VALUE_THRESHOLD:
                    value_flag = "VALUE"

            print(f"{idx+1:<4} {row['player_name'][:17]:<18} {row['opp_pitcher_name'][:17]:<18} "
                  f"{row['team']} v {row['opponent']:<6} {prob_pct:<6} {odds_str:<8} {edge_str:<6} {value_flag:<5}")

        # Summary
        total_predictions = len(df)
        value_plays = len(df[df['value_edge'] >= self.config.VALUE_THRESHOLD]) if 'value_edge' in df.columns else 0
        lineup_types = df['lineup_type'].value_counts()

        print("\n" + "=" * 95)
        print(f"[INFO] Total predictions: {total_predictions}")
        print(f"[INFO] Value plays detected: {value_plays}")
        print(f"[INFO] Lineup sources: {dict(lineup_types)}")
        print(f"[INFO] Run ID: {self.run_id}")

    def _save_results(self, df: pd.DataFrame):
        """Save results to CSV"""
        # Save top predictions
        filename = f"hr_predictions_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

        # Select key columns for CSV export
        export_cols = [
            'player_name', 'opp_pitcher_name', 'team', 'opponent', 'venue',
            'model_prob', 'market_odds', 'value_edge', 'lineup_type'
        ]

        export_df = df[export_cols].head(10)
        export_df.to_csv(filename, index=False)

        print(f"[SUCCESS] Predictions saved to: {filename}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='MLB Home Run Predictions v6.0')
    parser.add_argument('--dry-run', action='store_true', help='Skip odds collection')
    parser.add_argument('--no-odds', action='store_true', help='Disable odds entirely')
    parser.add_argument('--log-csv', type=str, help='Export prediction logs to CSV')

    args = parser.parse_args()

    # Initialize predictor
    enable_odds = not (args.dry_run or args.no_odds)
    predictor = MLBHomeRunPredictorV6(enable_odds=enable_odds)

    try:
        # Run predictions
        predictions = predictor.run_predictions()

        # Export logs if requested
        if args.log_csv:
            predictor.db.export_predictions_to_csv(args.log_csv, predictor.run_id)

        if predictions.empty:
            print("\n[INFO] No predictions generated. Try again closer to game time.")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()