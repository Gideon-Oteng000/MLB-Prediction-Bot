#!/usr/bin/env python3
"""
Historical Baseball Data Fetcher for ML Training (2018-2024)
Comprehensive data collection for home run prediction model
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import time
import os
import sqlite3
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('historical_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HistoricalDataFetcher:
    """
    Fetches historical baseball data for ML training (2018-2024)
    Handles lineups, game results, player metrics, and environmental data
    """

    def __init__(self, db_path: str = "historical_baseball_data.db"):
        self.db_path = db_path
        self.mlb_stats_base = "https://statsapi.mlb.com/api/v1"
        self.espn_base = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"

        # Weather API (using your existing key)
        self.weather_api_key = "e09911139e379f1e4ca813df1778b4ef"
        self.weather_base = "https://api.openweathermap.org/data/2.5"

        # Initialize database
        self._init_database()

        # MLB team mappings for consistency
        self.team_mappings = self._get_team_mappings()

        # Stadium coordinates for weather data
        self.stadium_coords = self._get_stadium_coordinates()

        logger.info("Historical Data Fetcher initialized")
        logger.info(f"Database: {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Games table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                stadium TEXT,
                game_time_hour INTEGER,
                day_night INTEGER,
                temperature REAL,
                wind_speed REAL,
                wind_direction REAL,
                humidity REAL,
                barometric_pressure REAL,
                park_hr_factor REAL,
                elevation REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Player game results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_game_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                opponent TEXT NOT NULL,
                date TEXT NOT NULL,
                batting_order INTEGER,
                position TEXT,
                home_away INTEGER,
                plate_appearances INTEGER DEFAULT 0,
                at_bats INTEGER DEFAULT 0,
                hits INTEGER DEFAULT 0,
                home_runs INTEGER DEFAULT 0,
                doubles INTEGER DEFAULT 0,
                triples INTEGER DEFAULT 0,
                walks INTEGER DEFAULT 0,
                strikeouts INTEGER DEFAULT 0,
                rbi INTEGER DEFAULT 0,
                stolen_bases INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games (game_id)
            )
        ''')

        # Historical player metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_player_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                date TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                time_period TEXT,  -- 'season', 'last_30d', 'last_7d', 'career'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Collection status tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_status (
                date TEXT PRIMARY KEY,
                lineups_collected INTEGER DEFAULT 0,
                results_collected INTEGER DEFAULT 0,
                metrics_collected INTEGER DEFAULT 0,
                weather_collected INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

    def collect_historical_data(self, start_date: str, end_date: str, skip_existing: bool = True):
        """
        Main method to collect all historical data for date range

        Args:
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            skip_existing: Skip dates that are already collected
        """
        logger.info(f"Starting historical data collection: {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

        current_date = start_dt
        total_days = (end_dt - start_dt).days + 1
        processed_days = 0
        successful_days = 0

        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m-%d')

            try:
                if skip_existing and self._is_date_collected(date_str):
                    logger.info(f"Skipping {date_str} - already collected")
                    current_date += timedelta(days=1)
                    processed_days += 1
                    continue

                logger.info(f"Processing {date_str} ({processed_days + 1}/{total_days})")

                # Step 1: Get games and lineups for this date
                games = self._fetch_historical_lineups(date_str)

                if not games:
                    logger.warning(f"No games found for {date_str}")
                    self._mark_date_processed(date_str, lineups=True, results=True,
                                            metrics=True, weather=True)
                    current_date += timedelta(days=1)
                    processed_days += 1
                    continue

                # Step 2: Get game results
                self._fetch_historical_game_results(date_str, games)

                # Step 3: Get weather data for each game
                self._fetch_historical_weather(date_str, games)

                # Step 4: Mark as processed
                self._mark_date_processed(date_str, lineups=True, results=True,
                                        metrics=False, weather=True)

                successful_days += 1
                logger.info(f"✅ Completed {date_str} - found {len(games)} games")

                # Reduced rate limiting for faster collection
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ Error processing {date_str}: {e}")

            current_date += timedelta(days=1)
            processed_days += 1

            # Progress update every 10 days
            if processed_days % 10 == 0:
                logger.info(f"Progress: {processed_days}/{total_days} days ({successful_days} successful)")

        logger.info(f"🏁 Historical data collection completed: {successful_days}/{processed_days} days successful")

    def _fetch_historical_lineups(self, date_str: str) -> List[Dict]:
        """Fetch lineups and basic game info for a specific date"""
        try:
            # Use MLB Stats API for historical lineups
            url = f"{self.mlb_stats_base}/schedule"
            params = {
                'sportId': 1,
                'date': date_str,
                'hydrate': 'team,linescore,boxscore'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            games = []
            if 'dates' in data and data['dates']:
                for game in data['dates'][0].get('games', []):
                    try:
                        game_info = self._process_historical_game(game, date_str)
                        if game_info:
                            games.append(game_info)
                            self._save_game_to_db(game_info)
                    except Exception as e:
                        logger.error(f"Error processing game: {e}")
                        continue

            return games

        except Exception as e:
            logger.error(f"Error fetching lineups for {date_str}: {e}")
            return []

    def _process_historical_game(self, game_data: Dict, date_str: str) -> Optional[Dict]:
        """Process a single game from MLB Stats API"""
        try:
            game_pk = game_data.get('gamePk')
            if not game_pk:
                return None

            # Basic game info
            teams = game_data.get('teams', {})
            away_team = teams.get('away', {}).get('team', {}).get('abbreviation', '')
            home_team = teams.get('home', {}).get('team', {}).get('abbreviation', '')

            if not away_team or not home_team:
                return None

            # Get venue info
            venue = game_data.get('venue', {})
            stadium_name = venue.get('name', '')

            # Game timing
            game_datetime = game_data.get('gameDate', '')
            game_time_hour = None
            day_night = None

            if game_datetime:
                try:
                    dt = datetime.fromisoformat(game_datetime.replace('Z', '+00:00'))
                    game_time_hour = dt.hour
                    day_night = 1 if dt.hour >= 17 else 0  # Night if after 5 PM
                except:
                    pass

            # Try to get lineups from boxscore
            lineups = self._extract_lineups_from_game(game_pk)

            game_info = {
                'game_id': str(game_pk),
                'date': date_str,
                'home_team': home_team,
                'away_team': away_team,
                'stadium': stadium_name,
                'game_time_hour': game_time_hour,
                'day_night': day_night,
                'lineups': lineups
            }

            return game_info

        except Exception as e:
            logger.error(f"Error processing game data: {e}")
            return None

    def _extract_lineups_from_game(self, game_pk: str) -> Dict:
        """Extract lineups from game boxscore"""
        try:
            url = f"{self.mlb_stats_base}/game/{game_pk}/boxscore"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            lineups = {'away': [], 'home': []}
            teams = data.get('teams', {})

            for team_type in ['away', 'home']:
                team_data = teams.get(team_type, {})
                batting_order = team_data.get('battingOrder', [])
                players = team_data.get('players', {})

                for order_num, player_id in enumerate(batting_order[:9], 1):
                    player_key = f"ID{player_id}"
                    if player_key in players:
                        player = players[player_key]
                        player_info = {
                            'name': player.get('person', {}).get('fullName', ''),
                            'position': player.get('position', {}).get('abbreviation', ''),
                            'batting_order': order_num
                        }
                        lineups[team_type].append(player_info)

            return lineups

        except Exception as e:
            logger.error(f"Error extracting lineups for game {game_pk}: {e}")
            return {'away': [], 'home': []}

    def _fetch_historical_game_results(self, date_str: str, games: List[Dict]):
        """Fetch actual game results and player stats"""
        for game in games:
            try:
                game_id = game['game_id']

                # Get detailed game results from MLB Stats API
                self._get_player_game_stats(game_id, game, date_str)

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Error fetching results for game {game_id}: {e}")

    def _get_player_game_stats(self, game_id: str, game_info: Dict, date_str: str):
        """Get individual player statistics for the game"""
        try:
            url = f"{self.mlb_stats_base}/game/{game_id}/boxscore"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            teams = data.get('teams', {})

            for team_type in ['away', 'home']:
                team_data = teams.get(team_type, {})
                team_name = game_info['away_team'] if team_type == 'away' else game_info['home_team']
                opponent = game_info['home_team'] if team_type == 'away' else game_info['away_team']
                home_away = 0 if team_type == 'away' else 1

                # Get batting stats
                batting_stats = team_data.get('batters', [])
                lineup = game_info['lineups'].get(team_type, [])

                for batter_id in batting_stats:
                    try:
                        player_key = f"ID{batter_id}"
                        if player_key in team_data.get('players', {}):
                            player = team_data['players'][player_key]
                            stats = player.get('stats', {}).get('batting', {})

                            # Find batting order from lineup
                            batting_order = None
                            position = None
                            player_name = player.get('person', {}).get('fullName', '')

                            for lineup_player in lineup:
                                if lineup_player['name'] == player_name:
                                    batting_order = lineup_player['batting_order']
                                    position = lineup_player['position']
                                    break

                            # Save player game result
                            player_result = {
                                'game_id': game_id,
                                'player_name': player_name,
                                'team': team_name,
                                'opponent': opponent,
                                'date': date_str,
                                'batting_order': batting_order,
                                'position': position,
                                'home_away': home_away,
                                'plate_appearances': stats.get('plateAppearances', 0),
                                'at_bats': stats.get('atBats', 0),
                                'hits': stats.get('hits', 0),
                                'home_runs': stats.get('homeRuns', 0),
                                'doubles': stats.get('doubles', 0),
                                'triples': stats.get('triples', 0),
                                'walks': stats.get('baseOnBalls', 0),
                                'strikeouts': stats.get('strikeOuts', 0),
                                'rbi': stats.get('rbi', 0),
                                'stolen_bases': stats.get('stolenBases', 0)
                            }

                            self._save_player_result_to_db(player_result)

                    except Exception as e:
                        logger.error(f"Error processing player stats: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error fetching game stats for {game_id}: {e}")

    def _fetch_historical_weather(self, date_str: str, games: List[Dict]):
        """Fetch historical weather data for games (simplified for now)"""
        # Note: Historical weather is complex and often requires paid APIs
        # For now, we'll use static park factors and add weather later
        for game in games:
            try:
                stadium = game.get('stadium', '')
                home_team = game.get('home_team', '')

                # Get park factor and elevation
                park_info = self._get_park_info(home_team)

                # Update game in database with park info
                self._update_game_park_info(game['game_id'], park_info)

            except Exception as e:
                logger.error(f"Error processing park info for game {game['game_id']}: {e}")

    def _get_park_info(self, team: str) -> Dict:
        """Get static park information"""
        # Using the stadium data from your weather fetcher
        park_factors = {
            'NYY': {'hr_factor': 1.12, 'elevation': 55},
            'BOS': {'hr_factor': 1.06, 'elevation': 21},
            'TB': {'hr_factor': 0.95, 'elevation': 15},
            'TOR': {'hr_factor': 1.02, 'elevation': 91},
            'BAL': {'hr_factor': 1.08, 'elevation': 20},
            'CLE': {'hr_factor': 0.98, 'elevation': 660},
            'MIN': {'hr_factor': 1.01, 'elevation': 815},
            'KC': {'hr_factor': 0.97, 'elevation': 750},
            'CWS': {'hr_factor': 1.03, 'elevation': 595},
            'DET': {'hr_factor': 0.94, 'elevation': 585},
            'HOU': {'hr_factor': 1.09, 'elevation': 22},
            'SEA': {'hr_factor': 0.92, 'elevation': 134},
            'LAA': {'hr_factor': 0.96, 'elevation': 153},
            'TEX': {'hr_factor': 1.05, 'elevation': 551},
            'OAK': {'hr_factor': 0.89, 'elevation': 13},
            'ATL': {'hr_factor': 1.04, 'elevation': 1050},
            'NYM': {'hr_factor': 0.93, 'elevation': 37},
            'PHI': {'hr_factor': 1.07, 'elevation': 20},
            'WSH': {'hr_factor': 1.01, 'elevation': 12},
            'MIA': {'hr_factor': 0.85, 'elevation': 8},
            'MIL': {'hr_factor': 1.02, 'elevation': 635},
            'CHC': {'hr_factor': 1.15, 'elevation': 595},
            'STL': {'hr_factor': 0.99, 'elevation': 465},
            'PIT': {'hr_factor': 0.91, 'elevation': 730},
            'CIN': {'hr_factor': 1.03, 'elevation': 550},
            'LAD': {'hr_factor': 0.88, 'elevation': 340},
            'SD': {'hr_factor': 0.88, 'elevation': 62},
            'SF': {'hr_factor': 0.81, 'elevation': 12},
            'COL': {'hr_factor': 1.25, 'elevation': 5200},
            'ARI': {'hr_factor': 1.06, 'elevation': 1059}
        }

        return park_factors.get(team, {'hr_factor': 1.0, 'elevation': 0})

    def _save_game_to_db(self, game_info: Dict):
        """Save game information to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO games
            (game_id, date, home_team, away_team, stadium, game_time_hour, day_night)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_info['game_id'],
            game_info['date'],
            game_info['home_team'],
            game_info['away_team'],
            game_info.get('stadium', ''),
            game_info.get('game_time_hour'),
            game_info.get('day_night')
        ))

        conn.commit()
        conn.close()

    def _save_player_result_to_db(self, player_result: Dict):
        """Save player game result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO player_game_results
            (game_id, player_name, team, opponent, date, batting_order, position,
             home_away, plate_appearances, at_bats, hits, home_runs, doubles,
             triples, walks, strikeouts, rbi, stolen_bases)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            player_result['game_id'],
            player_result['player_name'],
            player_result['team'],
            player_result['opponent'],
            player_result['date'],
            player_result['batting_order'],
            player_result['position'],
            player_result['home_away'],
            player_result['plate_appearances'],
            player_result['at_bats'],
            player_result['hits'],
            player_result['home_runs'],
            player_result['doubles'],
            player_result['triples'],
            player_result['walks'],
            player_result['strikeouts'],
            player_result['rbi'],
            player_result['stolen_bases']
        ))

        conn.commit()
        conn.close()

    def _update_game_park_info(self, game_id: str, park_info: Dict):
        """Update game with park factor and elevation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE games
            SET park_hr_factor = ?, elevation = ?
            WHERE game_id = ?
        ''', (park_info['hr_factor'], park_info['elevation'], game_id))

        conn.commit()
        conn.close()

    def _is_date_collected(self, date_str: str) -> bool:
        """Check if date has already been processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT lineups_collected, results_collected FROM collection_status WHERE date = ?', (date_str,))
        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0] == 1 and result[1] == 1
        return False

    def _mark_date_processed(self, date_str: str, lineups: bool = False,
                           results: bool = False, metrics: bool = False, weather: bool = False):
        """Mark date as processed in tracking table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO collection_status
            (date, lineups_collected, results_collected, metrics_collected, weather_collected)
            VALUES (?, ?, ?, ?, ?)
        ''', (date_str, int(lineups), int(results), int(metrics), int(weather)))

        conn.commit()
        conn.close()

    def _get_team_mappings(self) -> Dict:
        """Get team abbreviation mappings"""
        return {
            'Arizona Diamondbacks': 'ARI',
            'Atlanta Braves': 'ATL',
            'Baltimore Orioles': 'BAL',
            'Boston Red Sox': 'BOS',
            'Chicago Cubs': 'CHC',
            'Chicago White Sox': 'CWS',
            'Cincinnati Reds': 'CIN',
            'Cleveland Guardians': 'CLE',
            'Colorado Rockies': 'COL',
            'Detroit Tigers': 'DET',
            'Houston Astros': 'HOU',
            'Kansas City Royals': 'KC',
            'Los Angeles Angels': 'LAA',
            'Los Angeles Dodgers': 'LAD',
            'Miami Marlins': 'MIA',
            'Milwaukee Brewers': 'MIL',
            'Minnesota Twins': 'MIN',
            'New York Mets': 'NYM',
            'New York Yankees': 'NYY',
            'Oakland Athletics': 'OAK',
            'Philadelphia Phillies': 'PHI',
            'Pittsburgh Pirates': 'PIT',
            'San Diego Padres': 'SD',
            'San Francisco Giants': 'SF',
            'Seattle Mariners': 'SEA',
            'St. Louis Cardinals': 'STL',
            'Tampa Bay Rays': 'TB',
            'Texas Rangers': 'TEX',
            'Toronto Blue Jays': 'TOR',
            'Washington Nationals': 'WSH'
        }

    def _get_stadium_coordinates(self) -> Dict:
        """Get stadium coordinates for weather data"""
        # Return the stadium coordinates from your weather fetcher
        return {}  # Will implement if we add historical weather

    def get_collection_status(self) -> pd.DataFrame:
        """Get status of data collection"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM collection_status ORDER BY date', conn)
        conn.close()
        return df

    def get_sample_data(self, limit: int = 100) -> pd.DataFrame:
        """Get sample of collected data"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT pgr.*, g.stadium, g.park_hr_factor, g.elevation, g.day_night
            FROM player_game_results pgr
            JOIN games g ON pgr.game_id = g.game_id
            ORDER BY pgr.date DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df


def main():
    """Main function for historical data collection"""
    import argparse

    parser = argparse.ArgumentParser(description='Historical Baseball Data Collection')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip dates already collected')
    parser.add_argument('--db-path', default='historical_baseball_data.db',
                       help='Database file path')

    args = parser.parse_args()

    # Create fetcher
    fetcher = HistoricalDataFetcher(db_path=args.db_path)

    # Start collection
    fetcher.collect_historical_data(
        start_date=args.start_date,
        end_date=args.end_date,
        skip_existing=args.skip_existing
    )

    # Show status
    print("\nCollection Status:")
    status_df = fetcher.get_collection_status()
    print(status_df.tail(10))

    print("\nSample Data:")
    sample_df = fetcher.get_sample_data(10)
    print(sample_df[['player_name', 'date', 'home_runs', 'at_bats', 'team']].head())


if __name__ == "__main__":
    main()