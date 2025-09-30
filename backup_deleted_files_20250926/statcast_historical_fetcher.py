#!/usr/bin/env python3
"""
Historical Statcast Data Fetcher
Integrates with existing historical pipeline to collect advanced metrics from Baseball Savant
For comprehensive ML training data (2018-2024)
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

# Baseball Savant / pybaseball integration
try:
    from pybaseball import statcast, playerid_lookup, batting_stats, pitching_stats
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("Warning: pybaseball not installed. Install with: pip install pybaseball")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StatcastHistoricalFetcher:
    """
    Fetches historical Statcast data and integrates with existing historical database
    Collects advanced metrics for all batting events (2018-2024)
    """

    def __init__(self, db_path: str = "historical_baseball_data.db"):
        self.db_path = db_path

        if not PYBASEBALL_AVAILABLE:
            raise ImportError("pybaseball is required. Install with: pip install pybaseball")

        # Initialize additional database tables for Statcast data
        self._init_statcast_tables()

        # Key Statcast metrics for different predictions
        self.hr_metrics = [
            'launch_speed', 'launch_angle', 'estimated_ba_using_speedangle',
            'estimated_woba_using_speedangle', 'woba_value', 'barrel',
            'hit_distance_sc'
        ]

        self.hitting_metrics = [
            'launch_speed', 'launch_angle', 'estimated_ba_using_speedangle',
            'estimated_woba_using_speedangle', 'woba_value', 'iso_value',
            'babip_value', 'hit_location', 'bb_type'
        ]

        self.pitching_metrics = [
            'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z',
            'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0',
            'ax', 'ay', 'az', 'sz_top', 'sz_bot'
        ]

        logger.info("Statcast Historical Fetcher initialized")

    def _init_statcast_tables(self):
        """Initialize database tables for Statcast data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Historical Statcast batting events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statcast_batting_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date TEXT NOT NULL,
                player_name TEXT NOT NULL,
                batter_id INTEGER,
                pitcher_id INTEGER,
                team TEXT,
                opponent TEXT,

                -- Event outcome
                events TEXT,
                description TEXT,
                home_run INTEGER DEFAULT 0,
                hit INTEGER DEFAULT 0,
                rbi INTEGER DEFAULT 0,

                -- Statcast metrics
                launch_speed REAL,
                launch_angle REAL,
                hit_distance_sc REAL,
                barrel INTEGER DEFAULT 0,

                -- Expected stats
                estimated_ba_using_speedangle REAL,
                estimated_woba_using_speedangle REAL,
                woba_value REAL,
                babip_value REAL,
                iso_value REAL,

                -- Ball trajectory
                hc_x REAL,
                hc_y REAL,
                hit_location INTEGER,
                bb_type TEXT,

                -- Count and situation
                balls INTEGER,
                strikes INTEGER,
                outs_when_up INTEGER,
                inning INTEGER,
                inning_topbot TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Historical Statcast pitching events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statcast_pitching_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date TEXT NOT NULL,
                pitcher_name TEXT NOT NULL,
                batter_name TEXT NOT NULL,
                pitcher_id INTEGER,
                batter_id INTEGER,
                team TEXT,
                opponent TEXT,

                -- Pitch details
                pitch_type TEXT,
                pitch_number INTEGER,
                release_speed REAL,
                release_pos_x REAL,
                release_pos_z REAL,
                release_spin_rate REAL,

                -- Pitch movement
                pfx_x REAL,
                pfx_z REAL,
                plate_x REAL,
                plate_z REAL,

                -- Pitch outcome
                type TEXT,
                description TEXT,
                events TEXT,
                home_run_allowed INTEGER DEFAULT 0,
                strikeout INTEGER DEFAULT 0,

                -- Zone and swing
                zone INTEGER,
                swing_flag INTEGER DEFAULT 0,
                miss_flag INTEGER DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Player aggregated Statcast stats by date
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statcast_player_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date TEXT NOT NULL,
                player_name TEXT NOT NULL,
                player_id INTEGER,
                team TEXT,

                -- Batting aggregates
                plate_appearances INTEGER DEFAULT 0,
                at_bats INTEGER DEFAULT 0,
                hits INTEGER DEFAULT 0,
                home_runs INTEGER DEFAULT 0,

                -- Statcast batting averages
                avg_exit_velocity REAL,
                max_exit_velocity REAL,
                avg_launch_angle REAL,
                hard_hit_rate REAL,  -- 95+ mph
                barrel_rate REAL,
                sweet_spot_rate REAL,  -- 8-32 degree launch angle

                -- Expected stats
                expected_ba REAL,
                expected_woba REAL,
                expected_slg REAL,
                expected_ops REAL,

                -- Advanced metrics
                whiff_rate REAL,
                chase_rate REAL,
                zone_contact_rate REAL,
                swing_rate REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game_date, player_name, player_id)
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Statcast database tables initialized")

    def collect_historical_statcast_data(self, start_date: str, end_date: str):
        """
        Main method to collect historical Statcast data

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        logger.info(f"Starting historical Statcast collection: {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

        # Collect data in monthly chunks to avoid overwhelming the API
        current_date = start_dt.replace(day=1)  # Start of month

        while current_date <= end_dt:
            # Calculate month end
            if current_date.month == 12:
                month_end = current_date.replace(year=current_date.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = current_date.replace(month=current_date.month + 1, day=1) - timedelta(days=1)

            month_end = min(month_end, end_dt)

            try:
                logger.info(f"Collecting Statcast data for {current_date.strftime('%Y-%m')} ({current_date} to {month_end})")

                # Collect batting events for this month
                self._collect_monthly_batting_data(current_date, month_end)

                # Collect pitching events for this month
                self._collect_monthly_pitching_data(current_date, month_end)

                # Calculate daily aggregates
                self._calculate_daily_aggregates(current_date, month_end)

                # Rate limiting - be respectful to Baseball Savant
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error collecting data for {current_date.strftime('%Y-%m')}: {e}")

            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)

        logger.info("Historical Statcast collection completed")

    def _collect_monthly_batting_data(self, start_date: date, end_date: date):
        """Collect batting Statcast data for a month"""
        try:
            # Use pybaseball to get Statcast data
            statcast_data = statcast(
                start_dt=start_date.strftime('%Y-%m-%d'),
                end_dt=end_date.strftime('%Y-%m-%d')
            )

            if statcast_data.empty:
                logger.warning(f"No Statcast data found for {start_date} to {end_date}")
                return

            logger.info(f"Retrieved {len(statcast_data)} Statcast events for {start_date.strftime('%Y-%m')}")

            # Process and save batting events
            self._process_batting_events(statcast_data)

        except Exception as e:
            logger.error(f"Error collecting batting data: {e}")

    def _process_batting_events(self, statcast_data: pd.DataFrame):
        """Process raw Statcast data into batting events"""
        conn = sqlite3.connect(self.db_path)

        processed_events = []

        for _, row in statcast_data.iterrows():
            try:
                # Extract key information
                event_data = {
                    'game_date': row.get('game_date', ''),
                    'player_name': row.get('player_name', ''),
                    'batter_id': row.get('batter', 0),
                    'pitcher_id': row.get('pitcher', 0),
                    'team': self._get_team_abbrev(row.get('home_team', '')),
                    'opponent': self._get_team_abbrev(row.get('away_team', '')),

                    # Event outcome
                    'events': row.get('events', ''),
                    'description': row.get('description', ''),
                    'home_run': 1 if row.get('events') == 'home_run' else 0,
                    'hit': 1 if pd.notna(row.get('hit_location')) else 0,
                    'rbi': row.get('rbi', 0) or 0,

                    # Statcast metrics
                    'launch_speed': row.get('launch_speed'),
                    'launch_angle': row.get('launch_angle'),
                    'hit_distance_sc': row.get('hit_distance_sc'),
                    'barrel': 1 if row.get('barrel') == 1 else 0,

                    # Expected stats
                    'estimated_ba_using_speedangle': row.get('estimated_ba_using_speedangle'),
                    'estimated_woba_using_speedangle': row.get('estimated_woba_using_speedangle'),
                    'woba_value': row.get('woba_value'),
                    'babip_value': row.get('babip_value'),
                    'iso_value': row.get('iso_value'),

                    # Ball trajectory
                    'hc_x': row.get('hc_x'),
                    'hc_y': row.get('hc_y'),
                    'hit_location': row.get('hit_location'),
                    'bb_type': row.get('bb_type', ''),

                    # Count and situation
                    'balls': row.get('balls', 0),
                    'strikes': row.get('strikes', 0),
                    'outs_when_up': row.get('outs_when_up', 0),
                    'inning': row.get('inning', 0),
                    'inning_topbot': row.get('inning_topbot', '')
                }

                processed_events.append(event_data)

            except Exception as e:
                logger.error(f"Error processing batting event: {e}")
                continue

        # Bulk insert
        self._bulk_insert_batting_events(conn, processed_events)
        conn.close()

        logger.info(f"Processed {len(processed_events)} batting events")

    def _collect_monthly_pitching_data(self, start_date: date, end_date: date):
        """Collect pitching Statcast data for a month"""
        # This would collect pitch-by-pitch data for pitcher analysis
        # Similar structure to batting data but focused on pitch characteristics
        pass

    def _calculate_daily_aggregates(self, start_date: date, end_date: date):
        """Calculate daily aggregated Statcast stats for each player"""
        conn = sqlite3.connect(self.db_path)

        # Query to calculate daily aggregates from event data
        aggregate_query = '''
            INSERT OR REPLACE INTO statcast_player_daily
            (game_date, player_name, player_id, team, plate_appearances, at_bats,
             hits, home_runs, avg_exit_velocity, max_exit_velocity, avg_launch_angle,
             hard_hit_rate, barrel_rate, sweet_spot_rate, expected_ba, expected_woba)
            SELECT
                game_date,
                player_name,
                batter_id as player_id,
                team,
                COUNT(*) as plate_appearances,
                SUM(CASE WHEN bb_type != 'null' OR events IN ('strikeout', 'field_out', 'force_out', 'grounded_into_double_play') THEN 1 ELSE 0 END) as at_bats,
                SUM(hit) as hits,
                SUM(home_run) as home_runs,
                AVG(CASE WHEN launch_speed IS NOT NULL THEN launch_speed END) as avg_exit_velocity,
                MAX(launch_speed) as max_exit_velocity,
                AVG(CASE WHEN launch_angle IS NOT NULL THEN launch_angle END) as avg_launch_angle,
                AVG(CASE WHEN launch_speed >= 95 THEN 1.0 ELSE 0.0 END) as hard_hit_rate,
                AVG(CAST(barrel AS FLOAT)) as barrel_rate,
                AVG(CASE WHEN launch_angle BETWEEN 8 AND 32 THEN 1.0 ELSE 0.0 END) as sweet_spot_rate,
                AVG(estimated_ba_using_speedangle) as expected_ba,
                AVG(estimated_woba_using_speedangle) as expected_woba
            FROM statcast_batting_events
            WHERE game_date BETWEEN ? AND ?
            GROUP BY game_date, player_name, batter_id, team
        '''

        cursor = conn.cursor()
        cursor.execute(aggregate_query, [start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])

        conn.commit()
        conn.close()

        logger.info(f"Calculated daily aggregates for {start_date} to {end_date}")

    def _bulk_insert_batting_events(self, conn: sqlite3.Connection, events: List[Dict]):
        """Bulk insert batting events into database"""
        if not events:
            return

        cursor = conn.cursor()

        insert_query = '''
            INSERT OR REPLACE INTO statcast_batting_events
            (game_date, player_name, batter_id, pitcher_id, team, opponent,
             events, description, home_run, hit, rbi, launch_speed, launch_angle,
             hit_distance_sc, barrel, estimated_ba_using_speedangle,
             estimated_woba_using_speedangle, woba_value, babip_value, iso_value,
             hc_x, hc_y, hit_location, bb_type, balls, strikes, outs_when_up,
             inning, inning_topbot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        event_tuples = []
        for event in events:
            event_tuple = (
                event['game_date'], event['player_name'], event['batter_id'],
                event['pitcher_id'], event['team'], event['opponent'],
                event['events'], event['description'], event['home_run'],
                event['hit'], event['rbi'], event['launch_speed'],
                event['launch_angle'], event['hit_distance_sc'], event['barrel'],
                event['estimated_ba_using_speedangle'], event['estimated_woba_using_speedangle'],
                event['woba_value'], event['babip_value'], event['iso_value'],
                event['hc_x'], event['hc_y'], event['hit_location'],
                event['bb_type'], event['balls'], event['strikes'],
                event['outs_when_up'], event['inning'], event['inning_topbot']
            )
            event_tuples.append(event_tuple)

        cursor.executemany(insert_query, event_tuples)
        conn.commit()

    def _get_team_abbrev(self, team_name: str) -> str:
        """Convert team name to abbreviation"""
        team_mapping = {
            'Los Angeles Angels': 'LAA',
            'Houston Astros': 'HOU',
            'Oakland Athletics': 'OAK',
            'Toronto Blue Jays': 'TOR',
            'Atlanta Braves': 'ATL',
            'Milwaukee Brewers': 'MIL',
            'St. Louis Cardinals': 'STL',
            'Chicago Cubs': 'CHC',
            'Arizona Diamondbacks': 'ARI',
            'Los Angeles Dodgers': 'LAD',
            'San Francisco Giants': 'SF',
            'Cleveland Guardians': 'CLE',
            'Seattle Mariners': 'SEA',
            'Miami Marlins': 'MIA',
            'New York Mets': 'NYM',
            'Washington Nationals': 'WSH',
            'Baltimore Orioles': 'BAL',
            'San Diego Padres': 'SD',
            'Philadelphia Phillies': 'PHI',
            'Pittsburgh Pirates': 'PIT',
            'Texas Rangers': 'TEX',
            'Tampa Bay Rays': 'TB',
            'Boston Red Sox': 'BOS',
            'Cincinnati Reds': 'CIN',
            'Colorado Rockies': 'COL',
            'Kansas City Royals': 'KC',
            'Detroit Tigers': 'DET',
            'Minnesota Twins': 'MIN',
            'Chicago White Sox': 'CWS',
            'New York Yankees': 'NYY'
        }
        return team_mapping.get(team_name, team_name)

    def get_statcast_collection_status(self) -> Dict:
        """Get status of Statcast data collection"""
        conn = sqlite3.connect(self.db_path)

        status = {}
        tables = ['statcast_batting_events', 'statcast_pitching_events', 'statcast_player_daily']

        for table in tables:
            try:
                cursor = conn.cursor()
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = cursor.fetchone()[0]

                cursor.execute(f'SELECT MIN(game_date), MAX(game_date) FROM {table}')
                date_range = cursor.fetchone()

                status[table] = {
                    'records': count,
                    'date_range': date_range
                }
            except Exception as e:
                status[table] = {'error': str(e)}

        conn.close()
        return status

    def enhance_existing_historical_data(self):
        """
        Enhance existing historical database with Statcast data
        Links Statcast metrics to existing player_game_results
        """
        logger.info("Enhancing existing historical data with Statcast metrics")

        conn = sqlite3.connect(self.db_path)

        # Update existing player_game_results with Statcast data
        update_query = '''
            UPDATE player_game_results
            SET
                avg_exit_velocity = (
                    SELECT spd.avg_exit_velocity
                    FROM statcast_player_daily spd
                    WHERE spd.player_name = player_game_results.player_name
                    AND spd.game_date = player_game_results.date
                ),
                max_exit_velocity = (
                    SELECT spd.max_exit_velocity
                    FROM statcast_player_daily spd
                    WHERE spd.player_name = player_game_results.player_name
                    AND spd.game_date = player_game_results.date
                ),
                hard_hit_rate = (
                    SELECT spd.hard_hit_rate
                    FROM statcast_player_daily spd
                    WHERE spd.player_name = player_game_results.player_name
                    AND spd.game_date = player_game_results.date
                ),
                barrel_rate = (
                    SELECT spd.barrel_rate
                    FROM statcast_player_daily spd
                    WHERE spd.player_name = player_game_results.player_name
                    AND spd.game_date = player_game_results.date
                )
            WHERE EXISTS (
                SELECT 1 FROM statcast_player_daily spd
                WHERE spd.player_name = player_game_results.player_name
                AND spd.game_date = player_game_results.date
            )
        '''

        cursor = conn.cursor()
        cursor.execute(update_query)
        updated_records = cursor.rowcount

        conn.commit()
        conn.close()

        logger.info(f"Enhanced {updated_records} historical records with Statcast data")
        return updated_records


def main():
    """Test historical Statcast data collection"""
    import argparse

    parser = argparse.ArgumentParser(description='Historical Statcast Data Collection')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--enhance-existing', action='store_true',
                       help='Enhance existing historical data with Statcast')
    parser.add_argument('--status', action='store_true',
                       help='Show Statcast collection status')

    args = parser.parse_args()

    # Initialize Statcast fetcher
    fetcher = StatcastHistoricalFetcher()

    if args.status:
        status = fetcher.get_statcast_collection_status()
        print("\nStatcast Collection Status:")
        print("=" * 40)
        for table, info in status.items():
            print(f"{table}:")
            if 'error' in info:
                print(f"  Error: {info['error']}")
            else:
                print(f"  Records: {info['records']:,}")
                print(f"  Date Range: {info['date_range']}")
            print()

    elif args.enhance_existing:
        updated = fetcher.enhance_existing_historical_data()
        print(f"Enhanced {updated:,} historical records with Statcast data")

    else:
        # Collect historical data
        fetcher.collect_historical_statcast_data(args.start_date, args.end_date)

        # Show final status
        status = fetcher.get_statcast_collection_status()
        print(f"\nCollection completed. Final status:")
        for table, info in status.items():
            if 'records' in info:
                print(f"{table}: {info['records']:,} records")


if __name__ == "__main__":
    main()