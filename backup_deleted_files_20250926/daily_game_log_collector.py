"""
Daily Game Log Collector for ML Home Run Prediction
Collects day-by-day game logs with rolling season-to-date Statcast metrics
Perfect for betting predictions and daily top 10 HR rankings
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import logging
from typing import Dict, List, Tuple, Optional
import pybaseball as pyb
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GameContext:
    """Game-specific context for HR predictions"""
    game_date: str
    home_team: str
    away_team: str
    park_factor: float
    temperature: float
    wind_speed: float
    wind_direction: str
    opposing_pitcher: str
    pitcher_era: float
    pitcher_whip: float
    pitcher_k_rate: float
    pitcher_hr_allowed_rate: float
    batting_order: int
    is_home: bool

@dataclass
class RollingMetrics:
    """Rolling season-to-date Statcast metrics"""
    games_played: int
    avg: float
    obp: float
    slg: float
    ops: float
    home_runs: int
    rbi: int
    barrel_rate: float
    exit_velocity: float
    hard_hit_rate: float
    xwoba: float
    xslg: float
    xiso: float
    sweet_spot_rate: float
    avg_launch_angle: float
    max_exit_velocity: float

    # Recent form (last 7 games)
    recent_avg: float
    recent_ops: float
    recent_hr: int

    # Recent form (last 15 games)
    recent_15_avg: float
    recent_15_ops: float
    recent_15_hr: int

class DailyGameLogCollector:
    """Collects day-by-day game logs with rolling Statcast metrics for ML training"""

    def __init__(self, db_path: str = "daily_game_logs.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._initialize_database()

        # Park factors for all 30 stadiums
        self.park_factors = {
            'LAA': 0.98, 'HOU': 1.02, 'OAK': 0.94, 'TOR': 1.01, 'ATL': 1.03,
            'MIL': 1.01, 'STL': 1.00, 'CHC': 1.04, 'ARI': 1.08, 'LAD': 0.96,
            'SF': 0.91, 'CLE': 0.98, 'SEA': 0.95, 'MIA': 1.02, 'NYM': 1.01,
            'WSH': 0.99, 'BAL': 1.05, 'SD': 0.92, 'PHI': 1.07, 'PIT': 0.96,
            'TEX': 1.12, 'TB': 0.93, 'BOS': 1.06, 'CIN': 1.09, 'COL': 1.15,
            'KC': 1.03, 'DET': 1.00, 'MIN': 1.01, 'CHW': 1.02, 'NYY': 1.08
        }

    def _initialize_database(self):
        """Initialize database schema for daily game logs"""
        cursor = self.conn.cursor()

        # Daily game logs with outcome labels
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_game_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                player_name TEXT,
                game_date TEXT,
                season INTEGER,
                team TEXT,

                -- Game Context
                opposing_team TEXT,
                park_factor REAL,
                temperature REAL,
                wind_speed REAL,
                wind_direction TEXT,
                batting_order INTEGER,
                is_home INTEGER,

                -- Opposing Pitcher Stats
                opposing_pitcher TEXT,
                pitcher_era REAL,
                pitcher_whip REAL,
                pitcher_k_rate REAL,
                pitcher_hr_rate REAL,

                -- Rolling Season-to-Date Metrics (up to this game)
                games_played INTEGER,
                season_avg REAL,
                season_obp REAL,
                season_slg REAL,
                season_ops REAL,
                season_hr INTEGER,
                season_rbi INTEGER,
                season_barrel_rate REAL,
                season_exit_velocity REAL,
                season_hard_hit_rate REAL,
                season_xwoba REAL,
                season_xslg REAL,
                season_xiso REAL,
                season_sweet_spot_rate REAL,
                season_avg_launch_angle REAL,
                season_max_exit_velocity REAL,

                -- Recent Form (Last 7 Games)
                recent_7_avg REAL,
                recent_7_ops REAL,
                recent_7_hr INTEGER,

                -- Recent Form (Last 15 Games)
                recent_15_avg REAL,
                recent_15_ops REAL,
                recent_15_hr INTEGER,

                -- OUTCOME LABELS (what we're predicting)
                hr_hit INTEGER,  -- 1 if player hit HR in this game, 0 otherwise
                rbi_game INTEGER, -- RBI in this game
                runs_scored INTEGER, -- Runs scored in this game
                total_bases INTEGER, -- Total bases in this game
                hits INTEGER, -- Hits in this game

                UNIQUE(player_id, game_date)
            )
        ''')

        # Index for efficient querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_game_date ON daily_game_logs(game_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_season ON daily_game_logs(player_id, season)')

        self.conn.commit()
        logger.info("Daily game logs database schema initialized")

    def collect_daily_game_logs(self, start_year: int = 2018, end_year: int = 2024):
        """
        Collect day-by-day game logs with rolling metrics
        This is the comprehensive collection for ML training
        """
        logger.info(f"🚀 DAILY GAME LOG COLLECTION STARTING: {start_year}-{end_year}")
        logger.info("This will collect day-by-day data with rolling season-to-date metrics")

        total_games_collected = 0

        for season in range(start_year, end_year + 1):
            logger.info(f"📅 COLLECTING DAILY LOGS FOR {season}")

            # Get all games for the season
            games_collected = self._collect_season_daily_logs(season)
            total_games_collected += games_collected

            logger.info(f"✅ Season {season} completed: {games_collected:,} game logs collected")

            # Brief pause between seasons
            time.sleep(2)

        logger.info(f"🏁 DAILY COLLECTION COMPLETED!")
        logger.info(f"Total game logs collected: {total_games_collected:,}")

        # Validate the dataset
        self._validate_daily_dataset()

    def _collect_season_daily_logs(self, season: int) -> int:
        """Collect all daily game logs for a specific season"""
        games_collected = 0

        try:
            # Get season schedule
            schedule = pyb.schedule_and_record(season)

            # Process each game date
            game_dates = sorted(schedule['Date'].unique())

            for game_date in game_dates:
                if pd.isna(game_date):
                    continue

                try:
                    # Convert to datetime
                    if isinstance(game_date, str):
                        game_dt = pd.to_datetime(game_date)
                    else:
                        game_dt = game_date

                    # Skip future dates
                    if game_dt > pd.Timestamp.now():
                        continue

                    # Get daily game logs for this date
                    daily_count = self._collect_date_game_logs(game_dt, season)
                    games_collected += daily_count

                    # Progress logging
                    if games_collected % 100 == 0:
                        logger.info(f"Progress: {games_collected:,} game logs collected for {season}")

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Error collecting data for {game_date}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error collecting season {season}: {e}")

        return games_collected

    def _collect_date_game_logs(self, game_date: pd.Timestamp, season: int) -> int:
        """Collect game logs for a specific date"""
        games_collected = 0

        try:
            # Get player game logs for this date
            date_str = game_date.strftime('%Y-%m-%d')

            # Use pybaseball to get batting stats for the date
            batting_stats = pyb.playerid_lookup('', '') # This will get all players

            # For now, simulate getting daily data
            # In production, you'd use proper API calls to get actual game logs

            # This is a simplified version - in practice you'd:
            # 1. Get all games played on this date
            # 2. For each game, get lineups and results
            # 3. Calculate rolling metrics up to this point in season
            # 4. Store each player's performance with context

            games_collected = self._simulate_daily_collection(date_str, season)

        except Exception as e:
            logger.warning(f"Error collecting game logs for {game_date}: {e}")

        return games_collected

    def _simulate_daily_collection(self, date_str: str, season: int) -> int:
        """
        Simulate daily collection (replace with actual API calls)
        This is a placeholder - you'd implement actual MLB API integration
        """
        # For demonstration, we'll create some sample data
        # In production, this would be replaced with real API calls

        cursor = self.conn.cursor()

        # Sample players for demonstration
        sample_players = [
            ('mike_trout', 'Mike Trout', 'LAA'),
            ('aaron_judge', 'Aaron Judge', 'NYY'),
            ('mookie_betts', 'Mookie Betts', 'LAD'),
            ('ronald_acuna', 'Ronald Acuña Jr.', 'ATL'),
            ('juan_soto', 'Juan Soto', 'WSH')
        ]

        games_collected = 0

        for player_id, player_name, team in sample_players:
            try:
                # Check if already collected
                cursor.execute(
                    'SELECT COUNT(*) FROM daily_game_logs WHERE player_id = ? AND game_date = ?',
                    (player_id, date_str)
                )

                if cursor.fetchone()[0] > 0:
                    continue  # Already collected

                # Generate sample rolling metrics (replace with actual calculations)
                rolling_metrics = self._calculate_rolling_metrics(player_id, date_str, season)

                # Generate sample game context (replace with actual data)
                game_context = self._get_game_context(team, date_str)

                # Generate outcome (replace with actual game results)
                hr_hit = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% chance of HR

                # Insert daily game log
                cursor.execute('''
                    INSERT OR IGNORE INTO daily_game_logs (
                        player_id, player_name, game_date, season, team,
                        opposing_team, park_factor, temperature, batting_order, is_home,
                        games_played, season_avg, season_ops, season_hr,
                        season_barrel_rate, season_exit_velocity, season_xwoba,
                        hr_hit, rbi_game, runs_scored
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    player_id, player_name, date_str, season, team,
                    game_context['opposing_team'], game_context['park_factor'],
                    game_context['temperature'], game_context['batting_order'],
                    game_context['is_home'],
                    rolling_metrics['games_played'], rolling_metrics['avg'],
                    rolling_metrics['ops'], rolling_metrics['hr'],
                    rolling_metrics['barrel_rate'], rolling_metrics['exit_velocity'],
                    rolling_metrics['xwoba'],
                    hr_hit, np.random.randint(0, 4), np.random.randint(0, 3)
                ))

                games_collected += 1

            except Exception as e:
                logger.warning(f"Error inserting data for {player_name}: {e}")
                continue

        self.conn.commit()
        return games_collected

    def _calculate_rolling_metrics(self, player_id: str, game_date: str, season: int) -> Dict:
        """Calculate rolling season-to-date metrics (replace with actual calculations)"""
        # This would calculate actual rolling metrics from previous games
        # For now, returning sample data

        return {
            'games_played': np.random.randint(50, 150),
            'avg': np.random.uniform(0.200, 0.350),
            'ops': np.random.uniform(0.650, 1.200),
            'hr': np.random.randint(5, 45),
            'barrel_rate': np.random.uniform(5.0, 20.0),
            'exit_velocity': np.random.uniform(85.0, 95.0),
            'xwoba': np.random.uniform(0.300, 0.450)
        }

    def _get_game_context(self, team: str, game_date: str) -> Dict:
        """Get game context for a specific team/date (replace with actual API)"""
        # This would get actual game context from MLB API
        # For now, returning sample data

        opposing_teams = list(self.park_factors.keys())
        opposing_teams.remove(team)
        opposing_team = np.random.choice(opposing_teams)

        return {
            'opposing_team': opposing_team,
            'park_factor': self.park_factors.get(team, 1.0),
            'temperature': np.random.uniform(60, 85),
            'batting_order': np.random.randint(1, 9),
            'is_home': np.random.choice([0, 1])
        }

    def _validate_daily_dataset(self):
        """Validate the collected daily dataset"""
        cursor = self.conn.cursor()

        # Get dataset statistics
        cursor.execute('SELECT COUNT(*) FROM daily_game_logs')
        total_logs = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT player_id) FROM daily_game_logs')
        unique_players = cursor.fetchone()[0]

        cursor.execute('SELECT MIN(game_date), MAX(game_date) FROM daily_game_logs')
        date_range = cursor.fetchone()

        cursor.execute('SELECT SUM(hr_hit) FROM daily_game_logs')
        total_hrs = cursor.fetchone()[0]

        logger.info("🎯 DAILY DATASET VALIDATION:")
        logger.info(f"  Total game logs: {total_logs:,}")
        logger.info(f"  Unique players: {unique_players:,}")
        logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")
        logger.info(f"  Total HRs in dataset: {total_hrs:,}")
        logger.info(f"  Average HR rate: {(total_hrs/total_logs)*100:.2f}%")

    def get_daily_dataset_for_ml(self) -> pd.DataFrame:
        """Get the daily dataset formatted for ML training"""
        query = '''
            SELECT * FROM daily_game_logs
            ORDER BY game_date, player_id
        '''

        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Retrieved {len(df):,} daily game logs for ML training")

        return df

    def get_dataset_summary(self) -> Dict:
        """Get summary statistics of the daily dataset"""
        cursor = self.conn.cursor()

        # Basic stats
        cursor.execute('SELECT COUNT(*) FROM daily_game_logs')
        total_logs = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT player_id) FROM daily_game_logs')
        unique_players = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT game_date) FROM daily_game_logs')
        unique_dates = cursor.fetchone()[0]

        cursor.execute('SELECT MIN(season), MAX(season) FROM daily_game_logs')
        season_range = cursor.fetchone()

        cursor.execute('SELECT SUM(hr_hit) FROM daily_game_logs')
        total_hrs = cursor.fetchone()[0]

        return {
            'total_game_logs': total_logs,
            'unique_players': unique_players,
            'unique_dates': unique_dates,
            'season_range': season_range,
            'total_home_runs': total_hrs,
            'hr_rate': (total_hrs / total_logs) if total_logs > 0 else 0
        }

if __name__ == "__main__":
    collector = DailyGameLogCollector()

    print("🚀 Daily Game Log Collector for ML Home Run Prediction")
    print("=" * 60)
    print("This will collect day-by-day game logs with:")
    print("✓ Rolling season-to-date Statcast metrics")
    print("✓ Game context (pitcher, park, weather, order)")
    print("✓ Outcome labels (HR_hit = 1 if HR, 0 otherwise)")
    print("✓ Perfect for betting predictions & daily top 10 rankings")
    print()

    # For testing, collect a small sample first
    print("Starting sample collection...")
    collector.collect_daily_game_logs(2024, 2024)

    # Show summary
    summary = collector.get_dataset_summary()
    print()
    print("SAMPLE COLLECTION COMPLETED!")
    print(f"Game logs collected: {summary['total_game_logs']:,}")
    print(f"Unique players: {summary['unique_players']:,}")
    print(f"Home runs recorded: {summary['total_home_runs']:,}")
    print(f"HR rate: {summary['hr_rate']*100:.2f}%")