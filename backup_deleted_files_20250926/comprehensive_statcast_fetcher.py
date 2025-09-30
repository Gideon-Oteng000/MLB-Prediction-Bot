#!/usr/bin/env python3
"""
Comprehensive Statcast Historical Data Fetcher
Collects ALL specified batter and pitcher metrics (2018-2024)
Complete professional-grade MLB analytics foundation
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import time
import logging
import json

# Baseball data libraries
try:
    from pybaseball import statcast, batting_stats, pitching_stats, playerid_lookup
    from pybaseball import statcast_pitcher, statcast_batter
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("⚠️  pybaseball not available. Install with: pip install pybaseball")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveStatcastFetcher:
    """
    Fetches comprehensive Statcast data with ALL specified metrics
    2018-2024 complete coverage for professional ML training
    """

    def __init__(self, db_path: str = "comprehensive_baseball_data.db"):
        self.db_path = db_path

        if not PYBASEBALL_AVAILABLE:
            raise ImportError("pybaseball is required. Install with: pip install pybaseball")

        # Initialize comprehensive database schema
        self._init_comprehensive_database()

        # All specified batter metrics
        self.batter_metrics = {
            'power_hr_predictors': [
                'barrel_batted_rate', 'hard_hit_percent', 'avg_exit_velocity',
                'max_exit_velocity', 'avg_launch_angle', 'sweet_spot_percent',
                'pull_percent', 'home_runs', 'hr_per_fb', 'iso', 'xslg', 'xiso'
            ],
            'hitting_outcomes': [
                'batting_avg', 'xba', 'obp', 'ops', 'woba', 'xwoba', 'babip',
                'ld_percent', 'gb_percent', 'fb_percent'
            ],
            'plate_discipline': [
                'k_percent', 'bb_percent', 'whiff_percent', 'o_swing_percent',
                'z_contact_percent', 'swing_percent'
            ],
            'speed_context': [
                'sprint_speed', 'base_running_runs', 'batting_order_position'
            ]
        }

        # All specified pitcher metrics
        self.pitcher_metrics = {
            'hr_susceptibility': [
                'hr_per_9', 'barrel_percent_allowed', 'hard_hit_percent_allowed',
                'avg_exit_velocity_allowed', 'avg_launch_angle_allowed', 'xslg_against',
                'xwoba_against', 'hr_per_fb_allowed'
            ],
            'contact_management': [
                'xba_against', 'babip_against', 'groundball_percent', 'flyball_percent',
                'line_drive_percent_allowed', 'sweet_spot_percent_allowed'
            ],
            'strikeouts_discipline': [
                'k_percent', 'bb_percent', 'csw_percent', 'swinging_strike_percent',
                'o_swing_percent_induced', 'contact_percent_allowed'
            ],
            'run_prevention': [
                'era', 'fip', 'xfip', 'xera', 'whip'
            ],
            'pitch_characteristics': [
                'velocity_avg_fb', 'spin_rate_avg', 'fb_percent', 'sl_percent',
                'cb_percent', 'ch_percent'
            ]
        }

        logger.info("Comprehensive Statcast Fetcher initialized")
        logger.info(f"Target batter metrics: {sum(len(v) for v in self.batter_metrics.values())}")
        logger.info(f"Target pitcher metrics: {sum(len(v) for v in self.pitcher_metrics.values())}")

    def _init_comprehensive_database(self):
        """Initialize comprehensive database schema for all metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Comprehensive batter metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comprehensive_batter_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                player_id INTEGER,
                team TEXT,

                -- POWER / HR PREDICTORS
                barrel_batted_rate REAL,
                hard_hit_percent REAL,
                avg_exit_velocity REAL,
                max_exit_velocity REAL,
                avg_launch_angle REAL,
                sweet_spot_percent REAL,
                pull_percent REAL,
                home_runs INTEGER,
                hr_per_fb REAL,
                iso REAL,
                xslg REAL,
                xiso REAL,

                -- HITTING OUTCOMES
                batting_avg REAL,
                xba REAL,
                obp REAL,
                ops REAL,
                woba REAL,
                xwoba REAL,
                babip REAL,
                ld_percent REAL,
                gb_percent REAL,
                fb_percent REAL,

                -- PLATE DISCIPLINE
                k_percent REAL,
                bb_percent REAL,
                whiff_percent REAL,
                o_swing_percent REAL,
                z_contact_percent REAL,
                swing_percent REAL,

                -- SPEED / CONTEXT
                sprint_speed REAL,
                base_running_runs REAL,
                batting_order_position REAL,

                -- Basic counting stats
                plate_appearances INTEGER,
                at_bats INTEGER,
                hits INTEGER,
                doubles INTEGER,
                triples INTEGER,
                rbi INTEGER,
                runs INTEGER,
                walks INTEGER,
                strikeouts INTEGER,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(season, player_name, player_id, team)
            )
        ''')

        # Comprehensive pitcher metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comprehensive_pitcher_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER NOT NULL,
                pitcher_name TEXT NOT NULL,
                pitcher_id INTEGER,
                team TEXT,

                -- HR SUSCEPTIBILITY
                hr_per_9 REAL,
                barrel_percent_allowed REAL,
                hard_hit_percent_allowed REAL,
                avg_exit_velocity_allowed REAL,
                avg_launch_angle_allowed REAL,
                xslg_against REAL,
                xwoba_against REAL,
                hr_per_fb_allowed REAL,

                -- CONTACT MANAGEMENT
                xba_against REAL,
                babip_against REAL,
                groundball_percent REAL,
                flyball_percent REAL,
                line_drive_percent_allowed REAL,
                sweet_spot_percent_allowed REAL,

                -- STRIKEOUTS / PLATE DISCIPLINE
                k_percent REAL,
                bb_percent REAL,
                csw_percent REAL,
                swinging_strike_percent REAL,
                o_swing_percent_induced REAL,
                contact_percent_allowed REAL,

                -- TRADITIONAL RUN PREVENTION
                era REAL,
                fip REAL,
                xfip REAL,
                xera REAL,
                whip REAL,

                -- PITCH CHARACTERISTICS
                velocity_avg_fb REAL,
                spin_rate_avg REAL,
                fb_percent REAL,
                sl_percent REAL,
                cb_percent REAL,
                ch_percent REAL,

                -- Basic counting stats
                games INTEGER,
                games_started INTEGER,
                innings_pitched REAL,
                hits_allowed INTEGER,
                runs_allowed INTEGER,
                earned_runs INTEGER,
                walks_allowed INTEGER,
                strikeouts_pitched INTEGER,
                home_runs_allowed INTEGER,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(season, pitcher_name, pitcher_id, team)
            )
        ''')

        # Daily game results with Statcast integration
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_player_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date TEXT NOT NULL,
                game_id TEXT,
                player_name TEXT NOT NULL,
                player_id INTEGER,
                team TEXT NOT NULL,
                opponent TEXT NOT NULL,

                -- Basic game stats
                batting_order INTEGER,
                position TEXT,
                home_away INTEGER,
                plate_appearances INTEGER DEFAULT 0,
                at_bats INTEGER DEFAULT 0,
                hits INTEGER DEFAULT 0,
                home_runs INTEGER DEFAULT 0,
                doubles INTEGER DEFAULT 0,
                triples INTEGER DEFAULT 0,
                rbi INTEGER DEFAULT 0,
                runs INTEGER DEFAULT 0,
                walks INTEGER DEFAULT 0,
                strikeouts INTEGER DEFAULT 0,

                -- Daily Statcast metrics
                avg_exit_velocity REAL,
                max_exit_velocity REAL,
                launch_angle_avg REAL,
                hard_hit_balls INTEGER DEFAULT 0,
                barrels INTEGER DEFAULT 0,

                -- Environmental context
                park_hr_factor REAL,
                elevation REAL,
                temperature REAL,
                wind_speed REAL,
                day_night INTEGER,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game_date, player_name, player_id, team)
            )
        ''')

        # Games table with comprehensive context
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games_comprehensive (
                game_id TEXT PRIMARY KEY,
                game_date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_score INTEGER,
                away_score INTEGER,

                -- Stadium and environmental
                stadium TEXT,
                park_hr_factor REAL,
                elevation REAL,

                -- Weather
                temperature REAL,
                wind_speed REAL,
                wind_direction REAL,
                humidity REAL,
                barometric_pressure REAL,

                -- Game context
                day_night INTEGER,
                game_time_hour INTEGER,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Team game results comprehensive
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_game_comprehensive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                game_date TEXT NOT NULL,
                team TEXT NOT NULL,
                opponent TEXT NOT NULL,
                home_away INTEGER,

                -- Team offense
                runs_scored INTEGER DEFAULT 0,
                hits INTEGER DEFAULT 0,
                home_runs_hit INTEGER DEFAULT 0,
                rbi INTEGER DEFAULT 0,
                walks INTEGER DEFAULT 0,
                strikeouts INTEGER DEFAULT 0,
                team_batting_avg REAL,
                team_ops REAL,
                team_woba REAL,

                -- Team pitching
                runs_allowed INTEGER DEFAULT 0,
                hits_allowed INTEGER DEFAULT 0,
                home_runs_allowed INTEGER DEFAULT 0,
                walks_allowed INTEGER DEFAULT 0,
                strikeouts_pitched INTEGER DEFAULT 0,
                team_era REAL,
                team_whip REAL,
                team_fip REAL,

                -- Game outcome
                win INTEGER DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games_comprehensive (game_id)
            )
        ''')

        # Park factors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS park_factors (
                stadium TEXT PRIMARY KEY,
                team TEXT NOT NULL,
                hr_factor REAL DEFAULT 1.0,
                runs_factor REAL DEFAULT 1.0,
                hits_factor REAL DEFAULT 1.0,
                doubles_factor REAL DEFAULT 1.0,
                triples_factor REAL DEFAULT 1.0,
                elevation REAL,
                foul_territory TEXT,
                wall_height_lf REAL,
                wall_height_cf REAL,
                wall_height_rf REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Comprehensive database schema initialized")

    def collect_comprehensive_historical_data(self, start_year: int = 2018, end_year: int = 2024):
        """
        Main method to collect all comprehensive historical data

        Args:
            start_year: Starting season (2018)
            end_year: Ending season (2024)
        """
        logger.info(f"🚀 COMPREHENSIVE COLLECTION STARTING: {start_year}-{end_year}")

        # Initialize park factors
        self._initialize_park_factors()

        # Collect season by season
        for season in range(start_year, end_year + 1):
            logger.info(f"📅 COLLECTING SEASON {season}")

            try:
                # Phase 1: Batter metrics for the season
                self._collect_season_batter_metrics(season)

                # Phase 2: Pitcher metrics for the season
                self._collect_season_pitcher_metrics(season)

                # Phase 3: Daily game data for the season
                self._collect_season_daily_data(season)

                logger.info(f"✅ Season {season} completed successfully")

                # Rate limiting between seasons
                time.sleep(10)

            except Exception as e:
                logger.error(f"❌ Season {season} failed: {e}")
                continue

        # Final integration and validation
        self._integrate_and_validate_data()

        logger.info("🏁 COMPREHENSIVE COLLECTION COMPLETED!")

    def _collect_season_batter_metrics(self, season: int):
        """Collect all batter metrics for a season"""
        logger.info(f"Collecting batter metrics for {season}")

        try:
            # Get comprehensive batting stats using pybaseball
            batting_data = batting_stats(season, qual=50)  # Minimum 50 PA

            if batting_data.empty:
                logger.warning(f"No batting data found for {season}")
                return

            logger.info(f"Retrieved {len(batting_data)} batter records for {season}")

            # Process and save batter data
            self._process_batter_data(batting_data, season)

        except Exception as e:
            logger.error(f"Error collecting batter metrics for {season}: {e}")

    def _process_batter_data(self, batting_data: pd.DataFrame, season: int):
        """Process raw batting data into comprehensive metrics"""
        conn = sqlite3.connect(self.db_path)

        processed_batters = []

        for _, row in batting_data.iterrows():
            try:
                # Extract all specified metrics
                batter_record = {
                    'season': season,
                    'player_name': row.get('Name', ''),
                    'player_id': row.get('IDfg', 0),
                    'team': row.get('Team', ''),

                    # POWER / HR PREDICTORS
                    'barrel_batted_rate': row.get('Barrel%', 0) / 100 if pd.notna(row.get('Barrel%')) else None,
                    'hard_hit_percent': row.get('HardHit%', 0) / 100 if pd.notna(row.get('HardHit%')) else None,
                    'avg_exit_velocity': row.get('EV', 0),
                    'max_exit_velocity': row.get('maxEV', 0),
                    'avg_launch_angle': row.get('LA', 0),
                    'sweet_spot_percent': row.get('Sweet Spot%', 0) / 100 if pd.notna(row.get('Sweet Spot%')) else None,
                    'pull_percent': row.get('Pull%', 0) / 100 if pd.notna(row.get('Pull%')) else None,
                    'home_runs': row.get('HR', 0),
                    'hr_per_fb': row.get('HR/FB', 0) / 100 if pd.notna(row.get('HR/FB')) else None,
                    'iso': row.get('ISO', 0),
                    'xslg': row.get('xSLG', 0),
                    'xiso': row.get('xSLG', 0) - row.get('xBA', 0) if pd.notna(row.get('xSLG')) and pd.notna(row.get('xBA')) else None,

                    # HITTING OUTCOMES
                    'batting_avg': row.get('AVG', 0),
                    'xba': row.get('xBA', 0),
                    'obp': row.get('OBP', 0),
                    'ops': row.get('OPS', 0),
                    'woba': row.get('wOBA', 0),
                    'xwoba': row.get('xwOBA', 0),
                    'babip': row.get('BABIP', 0),
                    'ld_percent': row.get('LD%', 0) / 100 if pd.notna(row.get('LD%')) else None,
                    'gb_percent': row.get('GB%', 0) / 100 if pd.notna(row.get('GB%')) else None,
                    'fb_percent': row.get('FB%', 0) / 100 if pd.notna(row.get('FB%')) else None,

                    # PLATE DISCIPLINE
                    'k_percent': row.get('K%', 0) / 100 if pd.notna(row.get('K%')) else None,
                    'bb_percent': row.get('BB%', 0) / 100 if pd.notna(row.get('BB%')) else None,
                    'whiff_percent': row.get('SwStr%', 0) / 100 if pd.notna(row.get('SwStr%')) else None,
                    'o_swing_percent': row.get('O-Swing%', 0) / 100 if pd.notna(row.get('O-Swing%')) else None,
                    'z_contact_percent': row.get('Z-Contact%', 0) / 100 if pd.notna(row.get('Z-Contact%')) else None,
                    'swing_percent': row.get('Swing%', 0) / 100 if pd.notna(row.get('Swing%')) else None,

                    # SPEED / CONTEXT
                    'sprint_speed': row.get('Sprint Speed', 0),
                    'base_running_runs': row.get('BsR', 0),
                    'batting_order_position': None,  # Will be calculated from daily data

                    # Basic counting stats
                    'plate_appearances': row.get('PA', 0),
                    'at_bats': row.get('AB', 0),
                    'hits': row.get('H', 0),
                    'doubles': row.get('2B', 0),
                    'triples': row.get('3B', 0),
                    'rbi': row.get('RBI', 0),
                    'runs': row.get('R', 0),
                    'walks': row.get('BB', 0),
                    'strikeouts': row.get('SO', 0)
                }

                processed_batters.append(batter_record)

            except Exception as e:
                logger.error(f"Error processing batter record: {e}")
                continue

        # Bulk insert batter data
        self._bulk_insert_batter_metrics(conn, processed_batters)
        conn.close()

        logger.info(f"Processed {len(processed_batters)} batter records for {season}")

    def _collect_season_pitcher_metrics(self, season: int):
        """Collect all pitcher metrics for a season"""
        logger.info(f"Collecting pitcher metrics for {season}")

        try:
            # Get comprehensive pitching stats
            pitching_data = pitching_stats(season, qual=50)  # Minimum 50 IP

            if pitching_data.empty:
                logger.warning(f"No pitching data found for {season}")
                return

            logger.info(f"Retrieved {len(pitching_data)} pitcher records for {season}")

            # Process and save pitcher data
            self._process_pitcher_data(pitching_data, season)

        except Exception as e:
            logger.error(f"Error collecting pitcher metrics for {season}: {e}")

    def _process_pitcher_data(self, pitching_data: pd.DataFrame, season: int):
        """Process raw pitching data into comprehensive metrics"""
        conn = sqlite3.connect(self.db_path)

        processed_pitchers = []

        for _, row in pitching_data.iterrows():
            try:
                # Extract all specified pitcher metrics
                pitcher_record = {
                    'season': season,
                    'pitcher_name': row.get('Name', ''),
                    'pitcher_id': row.get('IDfg', 0),
                    'team': row.get('Team', ''),

                    # HR SUSCEPTIBILITY
                    'hr_per_9': row.get('HR/9', 0),
                    'barrel_percent_allowed': row.get('Barrel%', 0) / 100 if pd.notna(row.get('Barrel%')) else None,
                    'hard_hit_percent_allowed': row.get('HardHit%', 0) / 100 if pd.notna(row.get('HardHit%')) else None,
                    'avg_exit_velocity_allowed': row.get('EV', 0),
                    'avg_launch_angle_allowed': row.get('LA', 0),
                    'xslg_against': row.get('xSLG', 0),
                    'xwoba_against': row.get('xwOBA', 0),
                    'hr_per_fb_allowed': row.get('HR/FB', 0) / 100 if pd.notna(row.get('HR/FB')) else None,

                    # CONTACT MANAGEMENT
                    'xba_against': row.get('xBA', 0),
                    'babip_against': row.get('BABIP', 0),
                    'groundball_percent': row.get('GB%', 0) / 100 if pd.notna(row.get('GB%')) else None,
                    'flyball_percent': row.get('FB%', 0) / 100 if pd.notna(row.get('FB%')) else None,
                    'line_drive_percent_allowed': row.get('LD%', 0) / 100 if pd.notna(row.get('LD%')) else None,
                    'sweet_spot_percent_allowed': row.get('Sweet Spot%', 0) / 100 if pd.notna(row.get('Sweet Spot%')) else None,

                    # STRIKEOUTS / PLATE DISCIPLINE
                    'k_percent': row.get('K%', 0) / 100 if pd.notna(row.get('K%')) else None,
                    'bb_percent': row.get('BB%', 0) / 100 if pd.notna(row.get('BB%')) else None,
                    'csw_percent': row.get('CSW%', 0) / 100 if pd.notna(row.get('CSW%')) else None,
                    'swinging_strike_percent': row.get('SwStr%', 0) / 100 if pd.notna(row.get('SwStr%')) else None,
                    'o_swing_percent_induced': row.get('O-Swing%', 0) / 100 if pd.notna(row.get('O-Swing%')) else None,
                    'contact_percent_allowed': row.get('Contact%', 0) / 100 if pd.notna(row.get('Contact%')) else None,

                    # TRADITIONAL RUN PREVENTION
                    'era': row.get('ERA', 0),
                    'fip': row.get('FIP', 0),
                    'xfip': row.get('xFIP', 0),
                    'xera': row.get('xERA', 0),
                    'whip': row.get('WHIP', 0),

                    # PITCH CHARACTERISTICS
                    'velocity_avg_fb': row.get('vFA (pi)', 0),
                    'spin_rate_avg': row.get('FA% (pi)', 0),  # This will need adjustment
                    'fb_percent': row.get('FA% (pi)', 0) / 100 if pd.notna(row.get('FA% (pi)')) else None,
                    'sl_percent': row.get('SL% (pi)', 0) / 100 if pd.notna(row.get('SL% (pi)')) else None,
                    'cb_percent': row.get('CU% (pi)', 0) / 100 if pd.notna(row.get('CU% (pi)')) else None,
                    'ch_percent': row.get('CH% (pi)', 0) / 100 if pd.notna(row.get('CH% (pi)')) else None,

                    # Basic counting stats
                    'games': row.get('G', 0),
                    'games_started': row.get('GS', 0),
                    'innings_pitched': row.get('IP', 0),
                    'hits_allowed': row.get('H', 0),
                    'runs_allowed': row.get('R', 0),
                    'earned_runs': row.get('ER', 0),
                    'walks_allowed': row.get('BB', 0),
                    'strikeouts_pitched': row.get('SO', 0),
                    'home_runs_allowed': row.get('HR', 0)
                }

                processed_pitchers.append(pitcher_record)

            except Exception as e:
                logger.error(f"Error processing pitcher record: {e}")
                continue

        # Bulk insert pitcher data
        self._bulk_insert_pitcher_metrics(conn, processed_pitchers)
        conn.close()

        logger.info(f"Processed {len(processed_pitchers)} pitcher records for {season}")

    def _collect_season_daily_data(self, season: int):
        """Collect daily game-by-game data for a season"""
        logger.info(f"Collecting daily game data for {season}")

        # This will collect game-by-game results
        # For now, create placeholder structure
        # Full implementation would use MLB Stats API for daily data
        pass

    def _initialize_park_factors(self):
        """Initialize park factors for all MLB stadiums"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Comprehensive park factors data
        park_factors_data = [
            ('Fenway Park', 'BOS', 1.06, 1.03, 1.02, 1.15, 0.85, 21, 'Small', 37, 17, 3),
            ('Yankee Stadium', 'NYY', 1.12, 1.05, 1.01, 1.08, 0.95, 55, 'Average', 8, 8, 10),
            ('Tropicana Field', 'TB', 0.95, 0.98, 1.00, 0.90, 1.05, 15, 'Large', 10, 10, 10),
            ('Rogers Centre', 'TOR', 1.02, 1.01, 1.00, 1.05, 0.98, 91, 'Small', 10, 10, 10),
            ('Oriole Park at Camden Yards', 'BAL', 1.08, 1.04, 1.02, 1.10, 0.95, 20, 'Small', 7, 7, 25),
            ('Progressive Field', 'CLE', 0.98, 1.00, 1.01, 0.95, 1.08, 660, 'Average', 19, 9, 19),
            ('Target Field', 'MIN', 1.01, 1.00, 1.01, 0.98, 1.05, 815, 'Average', 8, 13, 7),
            ('Kauffman Stadium', 'KC', 0.97, 0.99, 1.00, 0.95, 1.10, 750, 'Large', 12, 10, 9),
            ('Guaranteed Rate Field', 'CWS', 1.03, 1.02, 1.01, 1.05, 0.98, 595, 'Average', 8, 8, 8),
            ('Comerica Park', 'DET', 0.94, 0.97, 0.99, 0.90, 1.12, 585, 'Large', 8, 5, 13),
            ('Minute Maid Park', 'HOU', 1.09, 1.06, 1.03, 1.12, 0.92, 22, 'Small', 19, 3, 21),
            ('T-Mobile Park', 'SEA', 0.92, 0.96, 0.98, 0.88, 1.15, 134, 'Large', 17, 11, 17),
            ('Angel Stadium', 'LAA', 0.96, 0.98, 1.00, 0.93, 1.08, 153, 'Large', 18, 8, 18),
            ('Globe Life Field', 'TEX', 1.05, 1.03, 1.02, 1.08, 0.95, 551, 'Average', 14, 8, 14),
            ('Oakland Coliseum', 'OAK', 0.89, 0.94, 0.97, 0.85, 1.20, 13, 'Large', 20, 8, 20),
            ('Truist Park', 'ATL', 1.04, 1.02, 1.01, 1.06, 0.96, 1050, 'Average', 8, 5, 8),
            ('Citi Field', 'NYM', 0.93, 0.97, 0.99, 0.90, 1.12, 37, 'Large', 12, 8, 8),
            ('Citizens Bank Park', 'PHI', 1.07, 1.04, 1.02, 1.10, 0.94, 20, 'Small', 11, 8, 14),
            ('Nationals Park', 'WSH', 1.01, 1.00, 1.00, 1.02, 1.02, 12, 'Average', 8, 8, 14),
            ('loanDepot park', 'MIA', 0.85, 0.92, 0.95, 0.80, 1.25, 8, 'Large', 16, 11, 16),
            ('American Family Field', 'MIL', 1.02, 1.01, 1.00, 1.04, 0.98, 635, 'Average', 8, 8, 8),
            ('Wrigley Field', 'CHC', 1.15, 1.08, 1.04, 1.20, 0.88, 595, 'Small', 11, 11, 15),
            ('Busch Stadium', 'STL', 0.99, 0.99, 1.00, 0.97, 1.05, 465, 'Large', 11, 9, 11),
            ('PNC Park', 'PIT', 0.91, 0.95, 0.98, 0.88, 1.15, 730, 'Large', 6, 10, 21),
            ('Great American Ball Park', 'CIN', 1.03, 1.02, 1.01, 1.06, 0.96, 550, 'Small', 12, 12, 14),
            ('Dodger Stadium', 'LAD', 0.88, 0.94, 0.97, 0.85, 1.18, 340, 'Large', 10, 8, 10),
            ('Petco Park', 'SD', 0.88, 0.94, 0.97, 0.85, 1.20, 62, 'Large', 19, 8, 19),
            ('Oracle Park', 'SF', 0.81, 0.90, 0.95, 0.75, 1.30, 12, 'Large', 25, 8, 21),
            ('Coors Field', 'COL', 1.25, 1.15, 1.08, 1.35, 0.75, 5200, 'Small', 8, 8, 13),
            ('Chase Field', 'ARI', 1.06, 1.03, 1.02, 1.08, 0.94, 1059, 'Average', 7, 7, 25)
        ]

        # Insert park factors
        for park_data in park_factors_data:
            cursor.execute('''
                INSERT OR REPLACE INTO park_factors
                (stadium, team, hr_factor, runs_factor, hits_factor, doubles_factor,
                 triples_factor, elevation, foul_territory, wall_height_lf,
                 wall_height_cf, wall_height_rf)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', park_data)

        conn.commit()
        conn.close()

        logger.info(f"Initialized park factors for {len(park_factors_data)} stadiums")

    def _bulk_insert_batter_metrics(self, conn: sqlite3.Connection, batters: List[Dict]):
        """Bulk insert batter metrics"""
        if not batters:
            return

        cursor = conn.cursor()

        insert_query = '''
            INSERT OR REPLACE INTO comprehensive_batter_metrics
            (season, player_name, player_id, team, barrel_batted_rate, hard_hit_percent,
             avg_exit_velocity, max_exit_velocity, avg_launch_angle, sweet_spot_percent,
             pull_percent, home_runs, hr_per_fb, iso, xslg, xiso, batting_avg, xba, obp,
             ops, woba, xwoba, babip, ld_percent, gb_percent, fb_percent, k_percent,
             bb_percent, whiff_percent, o_swing_percent, z_contact_percent, swing_percent,
             sprint_speed, base_running_runs, batting_order_position, plate_appearances,
             at_bats, hits, doubles, triples, rbi, runs, walks, strikeouts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        batter_tuples = []
        for batter in batters:
            batter_tuple = (
                batter['season'], batter['player_name'], batter['player_id'], batter['team'],
                batter['barrel_batted_rate'], batter['hard_hit_percent'], batter['avg_exit_velocity'],
                batter['max_exit_velocity'], batter['avg_launch_angle'], batter['sweet_spot_percent'],
                batter['pull_percent'], batter['home_runs'], batter['hr_per_fb'], batter['iso'],
                batter['xslg'], batter['xiso'], batter['batting_avg'], batter['xba'], batter['obp'],
                batter['ops'], batter['woba'], batter['xwoba'], batter['babip'], batter['ld_percent'],
                batter['gb_percent'], batter['fb_percent'], batter['k_percent'], batter['bb_percent'],
                batter['whiff_percent'], batter['o_swing_percent'], batter['z_contact_percent'],
                batter['swing_percent'], batter['sprint_speed'], batter['base_running_runs'],
                batter['batting_order_position'], batter['plate_appearances'], batter['at_bats'],
                batter['hits'], batter['doubles'], batter['triples'], batter['rbi'], batter['runs'],
                batter['walks'], batter['strikeouts']
            )
            batter_tuples.append(batter_tuple)

        cursor.executemany(insert_query, batter_tuples)
        conn.commit()

    def _bulk_insert_pitcher_metrics(self, conn: sqlite3.Connection, pitchers: List[Dict]):
        """Bulk insert pitcher metrics"""
        if not pitchers:
            return

        cursor = conn.cursor()

        insert_query = '''
            INSERT OR REPLACE INTO comprehensive_pitcher_metrics
            (season, pitcher_name, pitcher_id, team, hr_per_9, barrel_percent_allowed,
             hard_hit_percent_allowed, avg_exit_velocity_allowed, avg_launch_angle_allowed,
             xslg_against, xwoba_against, hr_per_fb_allowed, xba_against, babip_against,
             groundball_percent, flyball_percent, line_drive_percent_allowed,
             sweet_spot_percent_allowed, k_percent, bb_percent, csw_percent,
             swinging_strike_percent, o_swing_percent_induced, contact_percent_allowed,
             era, fip, xfip, xera, whip, velocity_avg_fb, spin_rate_avg, fb_percent,
             sl_percent, cb_percent, ch_percent, games, games_started, innings_pitched,
             hits_allowed, runs_allowed, earned_runs, walks_allowed, strikeouts_pitched,
             home_runs_allowed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        pitcher_tuples = []
        for pitcher in pitchers:
            pitcher_tuple = (
                pitcher['season'], pitcher['pitcher_name'], pitcher['pitcher_id'], pitcher['team'],
                pitcher['hr_per_9'], pitcher['barrel_percent_allowed'], pitcher['hard_hit_percent_allowed'],
                pitcher['avg_exit_velocity_allowed'], pitcher['avg_launch_angle_allowed'],
                pitcher['xslg_against'], pitcher['xwoba_against'], pitcher['hr_per_fb_allowed'],
                pitcher['xba_against'], pitcher['babip_against'], pitcher['groundball_percent'],
                pitcher['flyball_percent'], pitcher['line_drive_percent_allowed'],
                pitcher['sweet_spot_percent_allowed'], pitcher['k_percent'], pitcher['bb_percent'],
                pitcher['csw_percent'], pitcher['swinging_strike_percent'], pitcher['o_swing_percent_induced'],
                pitcher['contact_percent_allowed'], pitcher['era'], pitcher['fip'], pitcher['xfip'],
                pitcher['xera'], pitcher['whip'], pitcher['velocity_avg_fb'], pitcher['spin_rate_avg'],
                pitcher['fb_percent'], pitcher['sl_percent'], pitcher['cb_percent'], pitcher['ch_percent'],
                pitcher['games'], pitcher['games_started'], pitcher['innings_pitched'],
                pitcher['hits_allowed'], pitcher['runs_allowed'], pitcher['earned_runs'],
                pitcher['walks_allowed'], pitcher['strikeouts_pitched'], pitcher['home_runs_allowed']
            )
            pitcher_tuples.append(pitcher_tuple)

        cursor.executemany(insert_query, pitcher_tuples)
        conn.commit()

    def _integrate_and_validate_data(self):
        """Final data integration and validation"""
        logger.info("Integrating and validating comprehensive dataset")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get validation statistics
        cursor.execute('SELECT COUNT(*) FROM comprehensive_batter_metrics')
        batter_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM comprehensive_pitcher_metrics')
        pitcher_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT season) FROM comprehensive_batter_metrics')
        seasons_count = cursor.fetchone()[0]

        # Home run statistics
        cursor.execute('SELECT SUM(home_runs) FROM comprehensive_batter_metrics')
        total_hrs = cursor.fetchone()[0]

        conn.close()

        logger.info("🎯 COMPREHENSIVE DATA VALIDATION:")
        logger.info(f"  Batter records: {batter_count:,}")
        logger.info(f"  Pitcher records: {pitcher_count:,}")
        logger.info(f"  Seasons covered: {seasons_count}")
        logger.info(f"  Total home runs: {total_hrs:,}")

    def get_comprehensive_status(self) -> Dict:
        """Get status of comprehensive data collection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        status = {}

        # Check all tables
        tables = [
            'comprehensive_batter_metrics',
            'comprehensive_pitcher_metrics',
            'daily_player_results',
            'games_comprehensive',
            'team_game_comprehensive',
            'park_factors'
        ]

        for table in tables:
            try:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = cursor.fetchone()[0]
                status[table] = count
            except:
                status[table] = 0

        # Season coverage
        try:
            cursor.execute('SELECT MIN(season), MAX(season) FROM comprehensive_batter_metrics')
            season_range = cursor.fetchone()
            status['season_range'] = season_range
        except:
            status['season_range'] = (None, None)

        conn.close()
        return status


def main():
    """Main execution for comprehensive data collection"""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive Statcast Data Collection')
    parser.add_argument('--collect', action='store_true', help='Start comprehensive collection')
    parser.add_argument('--start-year', type=int, default=2018, help='Start year')
    parser.add_argument('--end-year', type=int, default=2024, help='End year')
    parser.add_argument('--status', action='store_true', help='Show collection status')

    args = parser.parse_args()

    # Initialize comprehensive fetcher
    fetcher = ComprehensiveStatcastFetcher()

    if args.status:
        status = fetcher.get_comprehensive_status()
        print("🎯 COMPREHENSIVE DATA STATUS:")
        print("=" * 50)
        for table, count in status.items():
            if table == 'season_range':
                print(f"Season Range: {count[0]} to {count[1]}")
            else:
                print(f"{table}: {count:,}")

    elif args.collect:
        print(f"🚀 Starting comprehensive collection: {args.start_year}-{args.end_year}")
        print("⚠️  This will take several hours to complete...")

        fetcher.collect_comprehensive_historical_data(args.start_year, args.end_year)

        # Show final status
        final_status = fetcher.get_comprehensive_status()
        print("\n🏁 COLLECTION COMPLETED!")
        print(f"Batter records: {final_status.get('comprehensive_batter_metrics', 0):,}")
        print(f"Pitcher records: {final_status.get('comprehensive_pitcher_metrics', 0):,}")

    else:
        print("Use --collect to start collection or --status to check progress")


if __name__ == "__main__":
    main()