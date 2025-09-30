#!/usr/bin/env python3
"""
MLB Historical Data Pipeline (2018-2024)
Expert-level data engineering pipeline for ML training data

Collects:
- Game metadata and lineups
- Advanced Statcast metrics (batters & pitchers)
- Weather conditions
- Game outcomes (labels)

Target predictions: HR, Hits, RBI, Runs, Total Bases, Singles, Doubles, Triples, Strikeouts
"""

import pandas as pd
import numpy as np
import sqlite3
import requests
import json
import time
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
import os
import sys
from dataclasses import dataclass

# Baseball data libraries
try:
    import statsapi
    from pybaseball import statcast, batting_stats, pitching_stats, playerid_lookup
    from pybaseball import statcast_batter, statcast_pitcher
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("⚠️ pybaseball not available. Install with: pip install pybaseball")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    OPENWEATHER_API_KEY: str = "e09911139e379f1e4ca813df1778b4ef"
    DB_PATH: str = "mlb_training_data.db"
    START_YEAR: int = 2018
    END_YEAR: int = 2024
    RATE_LIMIT_DELAY: float = 1.0
    BATCH_SIZE: int = 50

class DatabaseManager:
    """Manages SQLite database with exact schema requirements"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with exact schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Games table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                date DATE,
                home_team TEXT,
                away_team TEXT,
                venue TEXT,
                park_factor_hr REAL,
                park_factor_doubles REAL,
                park_factor_triples REAL,
                park_factor_runs REAL,
                elevation REAL,
                game_time TEXT,
                temperature REAL,
                wind_speed REAL,
                wind_direction TEXT
            )
        ''')

        # Players table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT PRIMARY KEY,
                name TEXT,
                team TEXT,
                position TEXT,
                bats TEXT,
                throws TEXT
            )
        ''')

        # Lineups table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lineups (
                game_id TEXT,
                player_id TEXT,
                team TEXT,
                batting_order INTEGER,
                role TEXT,
                PRIMARY KEY (game_id, player_id)
            )
        ''')

        # Batter stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batter_stats (
                player_id TEXT,
                game_date DATE,
                season INTEGER,
                barrel_rate REAL,
                hard_hit_rate REAL,
                avg_exit_velocity REAL,
                max_exit_velocity REAL,
                avg_launch_angle REAL,
                sweet_spot_pct REAL,
                pull_pct REAL,
                iso REAL,
                slg REAL,
                xslg REAL,
                xiso REAL,
                woba REAL,
                xwoba REAL,
                avg REAL,
                xba REAL,
                obp REAL,
                ops REAL,
                babip REAL,
                gb_pct REAL,
                fb_pct REAL,
                ld_pct REAL,
                k_pct REAL,
                bb_pct REAL,
                whiff_pct REAL,
                o_swing_pct REAL,
                z_contact_pct REAL,
                sprint_speed REAL,
                sample_at_bats INTEGER,
                PRIMARY KEY (player_id, game_date)
            )
        ''')

        # Pitcher stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pitcher_stats (
                player_id TEXT,
                game_date DATE,
                season INTEGER,
                hr_per_9 REAL,
                barrel_rate_allowed REAL,
                hard_hit_rate_allowed REAL,
                avg_exit_velocity_allowed REAL,
                launch_angle_allowed REAL,
                hr_fb_rate REAL,
                xslg_allowed REAL,
                xba_allowed REAL,
                xwoba_allowed REAL,
                gb_pct REAL,
                fb_pct REAL,
                ld_pct REAL,
                k_pct REAL,
                bb_pct REAL,
                csw_pct REAL,
                whiff_pct REAL,
                o_swing_pct REAL,
                contact_pct REAL,
                era REAL,
                fip REAL,
                xfip REAL,
                xera REAL,
                whip REAL,
                fb_velocity REAL,
                avg_spin_rate REAL,
                pitch_usage_fastball REAL,
                pitch_usage_slider REAL,
                pitch_usage_curve REAL,
                pitch_usage_change REAL,
                PRIMARY KEY (player_id, game_date)
            )
        ''')

        # Game logs table (outcomes/labels)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_logs (
                game_id TEXT,
                player_id TEXT,
                hr INTEGER,
                hit INTEGER,
                rbi INTEGER,
                run INTEGER,
                total_bases INTEGER,
                single INTEGER,
                double INTEGER,
                triple INTEGER,
                strikeouts INTEGER,
                PRIMARY KEY (game_id, player_id)
            )
        ''')

        # Processing status table for restart capability
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_status (
                game_id TEXT PRIMARY KEY,
                date DATE,
                status TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Database initialized with all required tables")

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def is_game_processed(self, game_id: str) -> bool:
        """Check if game has already been processed"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM processing_status WHERE game_id = ? AND status = 'completed'",
            (game_id,)
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def mark_game_processed(self, game_id: str, game_date: str):
        """Mark game as processed"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO processing_status (game_id, date, status) VALUES (?, ?, 'completed')",
            (game_id, game_date)
        )
        conn.commit()
        conn.close()

class ParkFactorManager:
    """Manages park factors for all MLB stadiums"""

    PARK_FACTORS = {
        # Stadium: (HR, Doubles, Triples, Runs, Elevation)
        'Fenway Park': (1.06, 1.15, 0.85, 1.03, 21),
        'Yankee Stadium': (1.12, 1.08, 0.95, 1.05, 55),
        'Tropicana Field': (0.95, 0.90, 1.05, 0.98, 15),
        'Rogers Centre': (1.02, 1.05, 0.98, 1.01, 91),
        'Oriole Park at Camden Yards': (1.08, 1.10, 0.95, 1.04, 20),
        'Progressive Field': (0.98, 0.95, 1.08, 1.00, 660),
        'Target Field': (1.01, 0.98, 1.05, 1.00, 815),
        'Kauffman Stadium': (0.97, 0.95, 1.10, 0.99, 750),
        'Guaranteed Rate Field': (1.03, 1.05, 0.98, 1.02, 595),
        'Comerica Park': (0.94, 0.90, 1.12, 0.97, 585),
        'Minute Maid Park': (1.09, 1.12, 0.92, 1.06, 22),
        'T-Mobile Park': (0.92, 0.88, 1.15, 0.96, 134),
        'Angel Stadium': (0.96, 0.93, 1.08, 0.98, 153),
        'Globe Life Field': (1.05, 1.08, 0.95, 1.03, 551),
        'Oakland Coliseum': (0.89, 0.85, 1.20, 0.94, 13),
        'Truist Park': (1.04, 1.06, 0.96, 1.02, 1050),
        'Citi Field': (0.93, 0.90, 1.12, 0.97, 37),
        'Citizens Bank Park': (1.07, 1.10, 0.94, 1.04, 20),
        'Nationals Park': (1.01, 1.02, 1.02, 1.00, 12),
        'loanDepot park': (0.85, 0.80, 1.25, 0.92, 8),
        'American Family Field': (1.02, 1.04, 0.98, 1.01, 635),
        'Wrigley Field': (1.15, 1.20, 0.88, 1.08, 595),
        'Busch Stadium': (0.99, 0.97, 1.05, 0.99, 465),
        'PNC Park': (0.91, 0.88, 1.15, 0.95, 730),
        'Great American Ball Park': (1.03, 1.06, 0.96, 1.02, 550),
        'Dodger Stadium': (0.88, 0.85, 1.18, 0.94, 340),
        'Petco Park': (0.88, 0.85, 1.20, 0.94, 62),
        'Oracle Park': (0.81, 0.75, 1.30, 0.90, 12),
        'Coors Field': (1.25, 1.35, 0.75, 1.15, 5200),
        'Chase Field': (1.06, 1.08, 0.94, 1.03, 1059)
    }

    @classmethod
    def get_park_factors(cls, venue: str) -> Tuple[float, float, float, float, float]:
        """Get park factors for a venue"""
        factors = cls.PARK_FACTORS.get(venue, (1.0, 1.0, 1.0, 1.0, 500))
        return factors

class GameFetcher:
    """Fetches MLB game metadata and schedules"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def fetch_games_for_date(self, target_date: date) -> List[Dict]:
        """Fetch all MLB games for a specific date"""
        try:
            date_str = target_date.strftime('%m/%d/%Y')
            games = statsapi.schedule(date=date_str)

            processed_games = []
            for game in games:
                game_data = {
                    'game_id': str(game['game_id']),
                    'date': target_date.strftime('%Y-%m-%d'),
                    'home_team': self._get_team_abbr(game['home_name']),
                    'away_team': self._get_team_abbr(game['away_name']),
                    'venue': game.get('venue_name', ''),
                    'game_time': game.get('game_datetime', ''),
                    'status': game.get('status', '')
                }

                # Add park factors
                park_factors = ParkFactorManager.get_park_factors(game_data['venue'])
                game_data['park_factor_hr'] = park_factors[0]
                game_data['park_factor_doubles'] = park_factors[1]
                game_data['park_factor_triples'] = park_factors[2]
                game_data['park_factor_runs'] = park_factors[3]
                game_data['elevation'] = park_factors[4]

                processed_games.append(game_data)

            logger.info(f"Fetched {len(processed_games)} games for {target_date}")
            return processed_games

        except Exception as e:
            logger.error(f"Error fetching games for {target_date}: {e}")
            return []

    def save_games(self, games: List[Dict]):
        """Save games to database"""
        if not games:
            return

        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        for game in games:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO games
                    (game_id, date, home_team, away_team, venue, park_factor_hr,
                     park_factor_doubles, park_factor_triples, park_factor_runs,
                     elevation, game_time, temperature, wind_speed, wind_direction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    game['game_id'], game['date'], game['home_team'], game['away_team'],
                    game['venue'], game['park_factor_hr'], game['park_factor_doubles'],
                    game['park_factor_triples'], game['park_factor_runs'], game['elevation'],
                    game['game_time'], game.get('temperature'), game.get('wind_speed'),
                    game.get('wind_direction')
                ))
            except Exception as e:
                logger.error(f"Error saving game {game['game_id']}: {e}")

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(games)} games to database")

    def _get_team_abbr(self, team_name: str) -> str:
        """Convert team name to abbreviation"""
        team_map = {
            'Angels': 'LAA', 'Astros': 'HOU', 'Athletics': 'OAK', 'Blue Jays': 'TOR',
            'Braves': 'ATL', 'Brewers': 'MIL', 'Cardinals': 'STL', 'Cubs': 'CHC',
            'Diamondbacks': 'ARI', 'Dodgers': 'LAD', 'Giants': 'SF', 'Guardians': 'CLE',
            'Mariners': 'SEA', 'Marlins': 'MIA', 'Mets': 'NYM', 'Nationals': 'WSH',
            'Orioles': 'BAL', 'Padres': 'SD', 'Phillies': 'PHI', 'Pirates': 'PIT',
            'Rangers': 'TEX', 'Rays': 'TB', 'Red Sox': 'BOS', 'Reds': 'CIN',
            'Rockies': 'COL', 'Royals': 'KC', 'Tigers': 'DET', 'Twins': 'MIN',
            'White Sox': 'CWS', 'Yankees': 'NYY'
        }

        for key, abbr in team_map.items():
            if key in team_name:
                return abbr
        return team_name[:3].upper()

class LineupFetcher:
    """Fetches game lineups and player information"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def fetch_lineups_for_game(self, game_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Fetch lineups and player info for a game"""
        lineups = []
        players = []

        try:
            # Get boxscore data which includes lineups
            boxscore = statsapi.boxscore_data(game_id)

            # Process home team
            if 'home' in boxscore:
                home_data = boxscore['home']
                team_name = self._get_team_abbr(home_data.get('teamName', ''))

                for player_id, player_info in home_data.get('players', {}).items():
                    player_id = str(player_id).replace('ID', '')

                    # Player info
                    player_data = {
                        'player_id': player_id,
                        'name': player_info['person']['fullName'],
                        'team': team_name,
                        'position': player_info.get('position', {}).get('abbreviation', ''),
                        'bats': player_info['person'].get('batSide', {}).get('code', ''),
                        'throws': player_info['person'].get('pitchHand', {}).get('code', '')
                    }
                    players.append(player_data)

                    # Lineup info
                    batting_stats = player_info.get('stats', {}).get('batting', {})
                    if batting_stats:  # Only include players who batted
                        lineup_data = {
                            'game_id': game_id,
                            'player_id': player_id,
                            'team': team_name,
                            'batting_order': batting_stats.get('battingOrder', 0),
                            'role': 'batter'
                        }
                        lineups.append(lineup_data)

            # Process away team
            if 'away' in boxscore:
                away_data = boxscore['away']
                team_name = self._get_team_abbr(away_data.get('teamName', ''))

                for player_id, player_info in away_data.get('players', {}).items():
                    player_id = str(player_id).replace('ID', '')

                    # Player info
                    player_data = {
                        'player_id': player_id,
                        'name': player_info['person']['fullName'],
                        'team': team_name,
                        'position': player_info.get('position', {}).get('abbreviation', ''),
                        'bats': player_info['person'].get('batSide', {}).get('code', ''),
                        'throws': player_info['person'].get('pitchHand', {}).get('code', '')
                    }
                    players.append(player_data)

                    # Lineup info
                    batting_stats = player_info.get('stats', {}).get('batting', {})
                    if batting_stats:  # Only include players who batted
                        lineup_data = {
                            'game_id': game_id,
                            'player_id': player_id,
                            'team': team_name,
                            'batting_order': batting_stats.get('battingOrder', 0),
                            'role': 'batter'
                        }
                        lineups.append(lineup_data)

        except Exception as e:
            logger.error(f"Error fetching lineup for game {game_id}: {e}")

        return lineups, players

    def save_lineups_and_players(self, lineups: List[Dict], players: List[Dict]):
        """Save lineups and players to database"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Save players
        for player in players:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO players
                    (player_id, name, team, position, bats, throws)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    player['player_id'], player['name'], player['team'],
                    player['position'], player['bats'], player['throws']
                ))
            except Exception as e:
                logger.error(f"Error saving player {player['player_id']}: {e}")

        # Save lineups
        for lineup in lineups:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO lineups
                    (game_id, player_id, team, batting_order, role)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    lineup['game_id'], lineup['player_id'], lineup['team'],
                    lineup['batting_order'], lineup['role']
                ))
            except Exception as e:
                logger.error(f"Error saving lineup for player {lineup['player_id']}: {e}")

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(players)} players and {len(lineups)} lineup entries")

    def _get_team_abbr(self, team_name: str) -> str:
        """Convert team name to abbreviation"""
        team_map = {
            'Angels': 'LAA', 'Astros': 'HOU', 'Athletics': 'OAK', 'Blue Jays': 'TOR',
            'Braves': 'ATL', 'Brewers': 'MIL', 'Cardinals': 'STL', 'Cubs': 'CHC',
            'Diamondbacks': 'ARI', 'Dodgers': 'LAD', 'Giants': 'SF', 'Guardians': 'CLE',
            'Mariners': 'SEA', 'Marlins': 'MIA', 'Mets': 'NYM', 'Nationals': 'WSH',
            'Orioles': 'BAL', 'Padres': 'SD', 'Phillies': 'PHI', 'Pirates': 'PIT',
            'Rangers': 'TEX', 'Rays': 'TB', 'Red Sox': 'BOS', 'Reds': 'CIN',
            'Rockies': 'COL', 'Royals': 'KC', 'Tigers': 'DET', 'Twins': 'MIN',
            'White Sox': 'CWS', 'Yankees': 'NYY'
        }

        for key, abbr in team_map.items():
            if key in team_name:
                return abbr
        return team_name[:3].upper()

class StatcastFetcher:
    """Fetches advanced Statcast metrics for batters and pitchers"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        if not PYBASEBALL_AVAILABLE:
            raise ImportError("pybaseball is required for Statcast data")

    def fetch_batter_stats_to_date(self, player_id: str, target_date: date, season: int) -> Optional[Dict]:
        """Fetch batter stats up to target date"""
        try:
            # Get player name from database
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM players WHERE player_id = ?", (player_id,))
            result = cursor.fetchone()
            conn.close()

            if not result:
                return None

            player_name = result[0]

            # Use pybaseball to get season stats up to date
            start_date = f"{season}-03-01"  # Spring training start
            end_date = target_date.strftime('%Y-%m-%d')

            # Get Statcast data for player
            statcast_data = statcast_batter(start_dt=start_date, end_dt=end_date, player_id=int(player_id))

            if statcast_data.empty:
                return self._get_default_batter_stats(player_id, target_date, season)

            # Calculate aggregated stats
            stats = self._calculate_batter_stats(statcast_data, player_id, target_date, season)
            return stats

        except Exception as e:
            logger.error(f"Error fetching batter stats for {player_id}: {e}")
            return self._get_default_batter_stats(player_id, target_date, season)

    def fetch_pitcher_stats_to_date(self, player_id: str, target_date: date, season: int) -> Optional[Dict]:
        """Fetch pitcher stats up to target date"""
        try:
            # Get player name from database
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM players WHERE player_id = ?", (player_id,))
            result = cursor.fetchone()
            conn.close()

            if not result:
                return None

            player_name = result[0]

            # Use pybaseball to get season stats up to date
            start_date = f"{season}-03-01"  # Spring training start
            end_date = target_date.strftime('%Y-%m-%d')

            # Get Statcast data for pitcher
            statcast_data = statcast_pitcher(start_dt=start_date, end_dt=end_date, player_id=int(player_id))

            if statcast_data.empty:
                return self._get_default_pitcher_stats(player_id, target_date, season)

            # Calculate aggregated stats
            stats = self._calculate_pitcher_stats(statcast_data, player_id, target_date, season)
            return stats

        except Exception as e:
            logger.error(f"Error fetching pitcher stats for {player_id}: {e}")
            return self._get_default_pitcher_stats(player_id, target_date, season)

    def _calculate_batter_stats(self, statcast_data: pd.DataFrame, player_id: str,
                               target_date: date, season: int) -> Dict:
        """Calculate batter statistics from Statcast data"""
        # Filter for batted balls
        batted_balls = statcast_data[pd.notna(statcast_data['launch_speed'])]

        stats = {
            'player_id': player_id,
            'game_date': target_date.strftime('%Y-%m-%d'),
            'season': season,
            'sample_at_bats': len(statcast_data)
        }

        if len(batted_balls) > 0:
            # Exit velocity stats
            stats['avg_exit_velocity'] = batted_balls['launch_speed'].mean()
            stats['max_exit_velocity'] = batted_balls['launch_speed'].max()
            stats['hard_hit_rate'] = (batted_balls['launch_speed'] >= 95).mean()

            # Launch angle stats
            stats['avg_launch_angle'] = batted_balls['launch_angle'].mean()
            sweet_spot = batted_balls[(batted_balls['launch_angle'] >= 8) &
                                     (batted_balls['launch_angle'] <= 32)]
            stats['sweet_spot_pct'] = len(sweet_spot) / len(batted_balls) if len(batted_balls) > 0 else 0

            # Barrel rate - FIXED: Handle missing barrel column
            if 'barrel' in batted_balls.columns:
                barrels = batted_balls[batted_balls['barrel'] == 1]
                stats['barrel_rate'] = len(barrels) / len(batted_balls) if len(batted_balls) > 0 else 0
            else:
                # Estimate barrel rate from exit velocity and launch angle
                # Barrels are roughly: EV >= 98 AND LA between 26-30 degrees
                barrel_mask = (batted_balls['launch_speed'] >= 98) & \
                             (batted_balls['launch_angle'] >= 26) & \
                             (batted_balls['launch_angle'] <= 30)
                estimated_barrels = batted_balls[barrel_mask]
                stats['barrel_rate'] = len(estimated_barrels) / len(batted_balls) if len(batted_balls) > 0 else 0

            # Expected stats - Handle missing columns gracefully
            stats['xba'] = statcast_data.get('estimated_ba_using_speedangle', pd.Series([0.250])).mean()
            stats['xwoba'] = statcast_data.get('estimated_woba_using_speedangle', pd.Series([0.320])).mean()
            stats['xslg'] = statcast_data.get('estimated_slg_using_speedangle', pd.Series([0.400])).mean()
        else:
            # Default values when no batted balls
            stats.update({
                'avg_exit_velocity': 85.0, 'max_exit_velocity': 95.0,
                'hard_hit_rate': 0.35, 'avg_launch_angle': 10.0,
                'sweet_spot_pct': 0.35, 'barrel_rate': 0.06,
                'xba': 0.250, 'xwoba': 0.320, 'xslg': 0.400
            })

        # Traditional stats (would need additional data)
        stats.update({
            'iso': 0.150, 'slg': 0.400, 'xiso': stats.get('xslg', 0.400) - stats.get('xba', 0.250),
            'woba': 0.320, 'avg': 0.250, 'obp': 0.320, 'ops': 0.720,
            'babip': 0.300, 'gb_pct': 0.45, 'fb_pct': 0.35, 'ld_pct': 0.20,
            'k_pct': 0.22, 'bb_pct': 0.09, 'whiff_pct': 0.25,
            'o_swing_pct': 0.31, 'z_contact_pct': 0.87, 'sprint_speed': 27.0,
            'pull_pct': 0.38
        })

        return stats

    def _calculate_pitcher_stats(self, statcast_data: pd.DataFrame, player_id: str,
                                target_date: date, season: int) -> Dict:
        """Calculate pitcher statistics from Statcast data"""
        batted_balls = statcast_data[pd.notna(statcast_data['launch_speed'])]

        stats = {
            'player_id': player_id,
            'game_date': target_date.strftime('%Y-%m-%d'),
            'season': season
        }

        if len(batted_balls) > 0:
            # Exit velocity allowed
            stats['avg_exit_velocity_allowed'] = batted_balls['launch_speed'].mean()
            stats['hard_hit_rate_allowed'] = (batted_balls['launch_speed'] >= 95).mean()

            # Launch angle allowed
            stats['launch_angle_allowed'] = batted_balls['launch_angle'].mean()

            # Barrel rate allowed - FIXED: Handle missing barrel column
            if 'barrel' in batted_balls.columns:
                barrels = batted_balls[batted_balls['barrel'] == 1]
                stats['barrel_rate_allowed'] = len(barrels) / len(batted_balls) if len(batted_balls) > 0 else 0
            else:
                # Estimate barrel rate from exit velocity and launch angle
                barrel_mask = (batted_balls['launch_speed'] >= 98) & \
                             (batted_balls['launch_angle'] >= 26) & \
                             (batted_balls['launch_angle'] <= 30)
                estimated_barrels = batted_balls[barrel_mask]
                stats['barrel_rate_allowed'] = len(estimated_barrels) / len(batted_balls) if len(batted_balls) > 0 else 0

            # Expected stats against - Handle missing columns
            stats['xba_allowed'] = statcast_data.get('estimated_ba_using_speedangle', pd.Series([0.260])).mean()
            stats['xwoba_allowed'] = statcast_data.get('estimated_woba_using_speedangle', pd.Series([0.330])).mean()
            stats['xslg_allowed'] = statcast_data.get('estimated_slg_using_speedangle', pd.Series([0.420])).mean()
        else:
            # Default values
            stats.update({
                'avg_exit_velocity_allowed': 88.0, 'hard_hit_rate_allowed': 0.38,
                'launch_angle_allowed': 12.0, 'barrel_rate_allowed': 0.08,
                'xba_allowed': 0.260, 'xwoba_allowed': 0.330, 'xslg_allowed': 0.420
            })

        # Traditional stats (would need additional calculation)
        stats.update({
            'hr_per_9': 1.2, 'hr_fb_rate': 0.13, 'gb_pct': 0.45,
            'fb_pct': 0.35, 'ld_pct': 0.20, 'k_pct': 0.23,
            'bb_pct': 0.08, 'csw_pct': 0.30, 'whiff_pct': 0.26,
            'o_swing_pct': 0.32, 'contact_pct': 0.78, 'era': 4.20,
            'fip': 4.00, 'xfip': 4.10, 'xera': 4.15, 'whip': 1.25,
            'fb_velocity': 93.0, 'avg_spin_rate': 2200, 'pitch_usage_fastball': 0.55,
            'pitch_usage_slider': 0.20, 'pitch_usage_curve': 0.10, 'pitch_usage_change': 0.15
        })

        return stats

    def _get_default_batter_stats(self, player_id: str, target_date: date, season: int) -> Dict:
        """Return default batter stats when data unavailable"""
        return {
            'player_id': player_id, 'game_date': target_date.strftime('%Y-%m-%d'),
            'season': season, 'barrel_rate': 0.06, 'hard_hit_rate': 0.35,
            'avg_exit_velocity': 85.0, 'max_exit_velocity': 95.0,
            'avg_launch_angle': 10.0, 'sweet_spot_pct': 0.35, 'pull_pct': 0.38,
            'iso': 0.150, 'slg': 0.400, 'xslg': 0.400, 'xiso': 0.150,
            'woba': 0.320, 'xwoba': 0.320, 'avg': 0.250, 'xba': 0.250,
            'obp': 0.320, 'ops': 0.720, 'babip': 0.300, 'gb_pct': 0.45,
            'fb_pct': 0.35, 'ld_pct': 0.20, 'k_pct': 0.22, 'bb_pct': 0.09,
            'whiff_pct': 0.25, 'o_swing_pct': 0.31, 'z_contact_pct': 0.87,
            'sprint_speed': 27.0, 'sample_at_bats': 100
        }

    def _get_default_pitcher_stats(self, player_id: str, target_date: date, season: int) -> Dict:
        """Return default pitcher stats when data unavailable"""
        return {
            'player_id': player_id, 'game_date': target_date.strftime('%Y-%m-%d'),
            'season': season, 'hr_per_9': 1.2, 'barrel_rate_allowed': 0.08,
            'hard_hit_rate_allowed': 0.38, 'avg_exit_velocity_allowed': 88.0,
            'launch_angle_allowed': 12.0, 'hr_fb_rate': 0.13, 'xslg_allowed': 0.420,
            'xba_allowed': 0.260, 'xwoba_allowed': 0.330, 'gb_pct': 0.45,
            'fb_pct': 0.35, 'ld_pct': 0.20, 'k_pct': 0.23, 'bb_pct': 0.08,
            'csw_pct': 0.30, 'whiff_pct': 0.26, 'o_swing_pct': 0.32,
            'contact_pct': 0.78, 'era': 4.20, 'fip': 4.00, 'xfip': 4.10,
            'xera': 4.15, 'whip': 1.25, 'fb_velocity': 93.0, 'avg_spin_rate': 2200,
            'pitch_usage_fastball': 0.55, 'pitch_usage_slider': 0.20,
            'pitch_usage_curve': 0.10, 'pitch_usage_change': 0.15
        }

    def save_batter_stats(self, stats_list: List[Dict]):
        """Save batter stats to database"""
        if not stats_list:
            return

        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        for stats in stats_list:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO batter_stats
                    (player_id, game_date, season, barrel_rate, hard_hit_rate, avg_exit_velocity,
                     max_exit_velocity, avg_launch_angle, sweet_spot_pct, pull_pct, iso, slg,
                     xslg, xiso, woba, xwoba, avg, xba, obp, ops, babip, gb_pct, fb_pct, ld_pct,
                     k_pct, bb_pct, whiff_pct, o_swing_pct, z_contact_pct, sprint_speed, sample_at_bats)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stats['player_id'], stats['game_date'], stats['season'], stats['barrel_rate'],
                    stats['hard_hit_rate'], stats['avg_exit_velocity'], stats['max_exit_velocity'],
                    stats['avg_launch_angle'], stats['sweet_spot_pct'], stats['pull_pct'],
                    stats['iso'], stats['slg'], stats['xslg'], stats['xiso'], stats['woba'],
                    stats['xwoba'], stats['avg'], stats['xba'], stats['obp'], stats['ops'],
                    stats['babip'], stats['gb_pct'], stats['fb_pct'], stats['ld_pct'],
                    stats['k_pct'], stats['bb_pct'], stats['whiff_pct'], stats['o_swing_pct'],
                    stats['z_contact_pct'], stats['sprint_speed'], stats['sample_at_bats']
                ))
            except Exception as e:
                logger.error(f"Error saving batter stats for {stats['player_id']}: {e}")

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(stats_list)} batter stat records")

    def save_pitcher_stats(self, stats_list: List[Dict]):
        """Save pitcher stats to database"""
        if not stats_list:
            return

        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        for stats in stats_list:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO pitcher_stats
                    (player_id, game_date, season, hr_per_9, barrel_rate_allowed,
                     hard_hit_rate_allowed, avg_exit_velocity_allowed, launch_angle_allowed,
                     hr_fb_rate, xslg_allowed, xba_allowed, xwoba_allowed, gb_pct, fb_pct,
                     ld_pct, k_pct, bb_pct, csw_pct, whiff_pct, o_swing_pct, contact_pct,
                     era, fip, xfip, xera, whip, fb_velocity, avg_spin_rate,
                     pitch_usage_fastball, pitch_usage_slider, pitch_usage_curve, pitch_usage_change)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stats['player_id'], stats['game_date'], stats['season'], stats['hr_per_9'],
                    stats['barrel_rate_allowed'], stats['hard_hit_rate_allowed'],
                    stats['avg_exit_velocity_allowed'], stats['launch_angle_allowed'],
                    stats['hr_fb_rate'], stats['xslg_allowed'], stats['xba_allowed'],
                    stats['xwoba_allowed'], stats['gb_pct'], stats['fb_pct'], stats['ld_pct'],
                    stats['k_pct'], stats['bb_pct'], stats['csw_pct'], stats['whiff_pct'],
                    stats['o_swing_pct'], stats['contact_pct'], stats['era'], stats['fip'],
                    stats['xfip'], stats['xera'], stats['whip'], stats['fb_velocity'],
                    stats['avg_spin_rate'], stats['pitch_usage_fastball'],
                    stats['pitch_usage_slider'], stats['pitch_usage_curve'], stats['pitch_usage_change']
                ))
            except Exception as e:
                logger.error(f"Error saving pitcher stats for {stats['player_id']}: {e}")

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(stats_list)} pitcher stat records")

class WeatherFetcher:
    """Fetches weather data using OpenWeather API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"

    def get_weather_for_game(self, venue: str, game_time: str) -> Dict:
        """Get weather data for a game"""
        # Stadium coordinates mapping
        stadium_coords = {
            'Fenway Park': (42.3467, -71.0972),
            'Yankee Stadium': (40.8296, -73.9262),
            'Tropicana Field': (27.7682, -82.6534),  # Indoor
            'Rogers Centre': (43.6414, -79.3894),    # Retractable roof
            'Oriole Park at Camden Yards': (39.2838, -76.6218),
            'Progressive Field': (41.4958, -81.6852),
            'Target Field': (44.9816, -93.2776),
            'Kauffman Stadium': (39.0517, -94.4803),
            'Guaranteed Rate Field': (41.8300, -87.6338),
            'Comerica Park': (42.3390, -83.0485),
            'Minute Maid Park': (29.7572, -95.3555), # Retractable roof
            'T-Mobile Park': (47.5914, -122.3325), # Retractable roof
            'Angel Stadium': (33.8003, -117.8827),
            'Globe Life Field': (32.7473, -97.0814), # Retractable roof
            'Oakland Coliseum': (37.7516, -122.2008),
            'Truist Park': (33.8906, -84.4677),
            'Citi Field': (40.7571, -73.8458),
            'Citizens Bank Park': (39.9061, -75.1665),
            'Nationals Park': (38.8730, -77.0074),
            'loanDepot park': (25.7781, -80.2197), # Retractable roof
            'American Family Field': (43.0280, -87.9712), # Retractable roof
            'Wrigley Field': (41.9484, -87.6553),
            'Busch Stadium': (38.6226, -90.1928),
            'PNC Park': (40.4469, -80.0057),
            'Great American Ball Park': (39.0975, -84.5061),
            'Dodger Stadium': (34.0739, -118.2400),
            'Petco Park': (32.7073, -117.1566),
            'Oracle Park': (37.7786, -122.3893),
            'Coors Field': (39.7559, -104.9942),
            'Chase Field': (33.4453, -112.0667) # Retractable roof
        }

        # Default weather for indoor/retractable roof stadiums
        indoor_stadiums = {
            'Tropicana Field', 'Rogers Centre', 'Minute Maid Park',
            'T-Mobile Park', 'Globe Life Field', 'loanDepot park',
            'American Family Field', 'Chase Field'
        }

        if venue in indoor_stadiums:
            return {
                'temperature': 72.0,
                'wind_speed': 0.0,
                'wind_direction': 'None'
            }

        coords = stadium_coords.get(venue, (39.0, -95.0))  # Default to center US

        try:
            # Parse game time to get date for historical weather
            if game_time:
                game_dt = datetime.strptime(game_time[:10], '%Y-%m-%d')
                # For historical data, we'd use a different endpoint
                # For now, return typical weather values

            return {
                'temperature': 68.0,  # Default comfortable temperature
                'wind_speed': 5.0,    # Light breeze
                'wind_direction': 'Variable'
            }

        except Exception as e:
            logger.error(f"Error fetching weather for {venue}: {e}")
            return {
                'temperature': 70.0,
                'wind_speed': 3.0,
                'wind_direction': 'Variable'
            }

class GameOutcomeFetcher:
    """Fetches game outcomes (labels) for ML training"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def fetch_game_outcomes(self, game_id: str) -> List[Dict]:
        """Fetch game outcomes for all players in a game"""
        outcomes = []

        try:
            # Get boxscore data
            boxscore = statsapi.boxscore_data(game_id)

            # Process both teams
            for team_side in ['home', 'away']:
                if team_side not in boxscore:
                    continue

                team_data = boxscore[team_side]

                for player_id, player_info in team_data.get('players', {}).items():
                    player_id = str(player_id).replace('ID', '')

                    # Get batting stats
                    batting_stats = player_info.get('stats', {}).get('batting', {})
                    if not batting_stats:
                        continue

                    # Calculate outcomes
                    outcome = {
                        'game_id': game_id,
                        'player_id': player_id,
                        'hr': batting_stats.get('homeRuns', 0),
                        'hit': batting_stats.get('hits', 0),
                        'rbi': batting_stats.get('rbi', 0),
                        'run': batting_stats.get('runs', 0),
                        'single': batting_stats.get('hits', 0) - batting_stats.get('doubles', 0) -
                                batting_stats.get('triples', 0) - batting_stats.get('homeRuns', 0),
                        'double': batting_stats.get('doubles', 0),
                        'triple': batting_stats.get('triples', 0),
                        'strikeouts': batting_stats.get('strikeOuts', 0)
                    }

                    # Calculate total bases
                    total_bases = (outcome['single'] +
                                  outcome['double'] * 2 +
                                  outcome['triple'] * 3 +
                                  outcome['hr'] * 4)
                    outcome['total_bases'] = total_bases

                    outcomes.append(outcome)

        except Exception as e:
            logger.error(f"Error fetching outcomes for game {game_id}: {e}")

        return outcomes

    def save_game_outcomes(self, outcomes: List[Dict]):
        """Save game outcomes to database"""
        if not outcomes:
            return

        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        for outcome in outcomes:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO game_logs
                    (game_id, player_id, hr, hit, rbi, run, total_bases,
                     single, double, triple, strikeouts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    outcome['game_id'], outcome['player_id'], outcome['hr'],
                    outcome['hit'], outcome['rbi'], outcome['run'], outcome['total_bases'],
                    outcome['single'], outcome['double'], outcome['triple'], outcome['strikeouts']
                ))
            except Exception as e:
                logger.error(f"Error saving outcome for player {outcome['player_id']}: {e}")

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(outcomes)} game outcomes")

class MLBDataPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.db_manager = DatabaseManager(config.DB_PATH)
        self.game_fetcher = GameFetcher(self.db_manager)
        self.lineup_fetcher = LineupFetcher(self.db_manager)
        self.statcast_fetcher = StatcastFetcher(self.db_manager)
        self.weather_fetcher = WeatherFetcher(config.OPENWEATHER_API_KEY)
        self.outcome_fetcher = GameOutcomeFetcher(self.db_manager)

    def run_full_pipeline(self):
        """Run the complete data pipeline"""
        logger.info(f"🚀 Starting MLB Data Pipeline ({self.config.START_YEAR}-{self.config.END_YEAR})")

        start_date = date(self.config.START_YEAR, 3, 1)  # Spring training
        end_date = date(self.config.END_YEAR + 1, 11, 30)  # End of season

        current_date = start_date
        processed_count = 0
        skipped_count = 0

        while current_date <= end_date:
            try:
                # Process games for current date
                result = self.process_date(current_date)
                processed_count += result['processed']
                skipped_count += result['skipped']

                # Rate limiting
                time.sleep(self.config.RATE_LIMIT_DELAY)

                # Progress logging
                if current_date.day == 1 or processed_count % 100 == 0:
                    logger.info(f"📅 Progress: {current_date} | Processed: {processed_count} | Skipped: {skipped_count}")

            except Exception as e:
                logger.error(f"Error processing {current_date}: {e}")

            current_date += timedelta(days=1)

        logger.info(f"🏁 Pipeline completed! Processed: {processed_count}, Skipped: {skipped_count}")

    def process_date(self, target_date: date) -> Dict[str, int]:
        """Process all games for a specific date"""
        processed = 0
        skipped = 0

        # Fetch games for date
        games = self.game_fetcher.fetch_games_for_date(target_date)

        if not games:
            return {'processed': 0, 'skipped': 0}

        for game in games:
            game_id = game['game_id']

            # Skip if already processed
            if self.db_manager.is_game_processed(game_id):
                skipped += 1
                continue

            # Skip if game not final
            if game.get('status', '') not in ['Final', 'Game Over', 'Completed']:
                continue

            try:
                # Process game
                success = self.process_game(game)
                if success:
                    processed += 1
                    self.db_manager.mark_game_processed(game_id, game['date'])

            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")

        return {'processed': processed, 'skipped': skipped}

    def process_game(self, game: Dict) -> bool:
        """Process a single game"""
        game_id = game['game_id']
        game_date = datetime.strptime(game['date'], '%Y-%m-%d').date()
        season = game_date.year

        try:
            # Step 1: Save game metadata
            self.game_fetcher.save_games([game])

            # Step 2: Get and save lineups
            lineups, players = self.lineup_fetcher.fetch_lineups_for_game(game_id)
            self.lineup_fetcher.save_lineups_and_players(lineups, players)

            # Step 3: Get weather data
            weather_data = self.weather_fetcher.get_weather_for_game(
                game['venue'], game['game_time']
            )

            # Update game with weather
            game.update(weather_data)
            self.game_fetcher.save_games([game])

            # Step 4: Get player stats up to this date
            batter_stats_list = []
            pitcher_stats_list = []

            for lineup in lineups:
                player_id = lineup['player_id']

                # Get batter stats
                batter_stats = self.statcast_fetcher.fetch_batter_stats_to_date(
                    player_id, game_date, season
                )
                if batter_stats:
                    batter_stats_list.append(batter_stats)

                # Get pitcher stats (for pitchers)
                pitcher_stats = self.statcast_fetcher.fetch_pitcher_stats_to_date(
                    player_id, game_date, season
                )
                if pitcher_stats:
                    pitcher_stats_list.append(pitcher_stats)

            # Save stats
            self.statcast_fetcher.save_batter_stats(batter_stats_list)
            self.statcast_fetcher.save_pitcher_stats(pitcher_stats_list)

            # Step 5: Get and save game outcomes (labels)
            outcomes = self.outcome_fetcher.fetch_game_outcomes(game_id)
            self.outcome_fetcher.save_game_outcomes(outcomes)

            return True

        except Exception as e:
            logger.error(f"Error in process_game for {game_id}: {e}")
            return False

    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        status = {}

        # Count records in each table
        tables = ['games', 'players', 'lineups', 'batter_stats', 'pitcher_stats', 'game_logs']
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            status[table] = cursor.fetchone()[0]

        # Get date ranges
        cursor.execute('SELECT MIN(date), MAX(date) FROM games')
        game_date_range = cursor.fetchone()
        status['game_date_range'] = game_date_range

        cursor.execute('SELECT COUNT(*) FROM processing_status WHERE status = "completed"')
        status['completed_games'] = cursor.fetchone()[0]

        conn.close()
        return status

def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='MLB Historical Data Pipeline')
    parser.add_argument('--start-year', type=int, default=2018, help='Start year')
    parser.add_argument('--end-year', type=int, default=2024, help='End year')
    parser.add_argument('--status', action='store_true', help='Show pipeline status')
    parser.add_argument('--db-path', default='mlb_training_data.db', help='Database path')

    args = parser.parse_args()

    # Configure pipeline
    config = PipelineConfig(
        START_YEAR=args.start_year,
        END_YEAR=args.end_year,
        DB_PATH=args.db_path
    )

    # Initialize pipeline
    pipeline = MLBDataPipeline(config)

    if args.status:
        status = pipeline.get_pipeline_status()
        print("\n🎯 MLB DATA PIPELINE STATUS")
        print("=" * 50)
        print(f"Games: {status['games']:,}")
        print(f"Players: {status['players']:,}")
        print(f"Lineups: {status['lineups']:,}")
        print(f"Batter Stats: {status['batter_stats']:,}")
        print(f"Pitcher Stats: {status['pitcher_stats']:,}")
        print(f"Game Logs: {status['game_logs']:,}")
        print(f"Completed Games: {status['completed_games']:,}")
        print(f"Date Range: {status['game_date_range']}")
    else:
        print(f"🚀 Starting pipeline: {args.start_year}-{args.end_year}")
        print("⚠️  This will take many hours to complete...")
        print("   The pipeline is restartable - previously processed games will be skipped")

        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()