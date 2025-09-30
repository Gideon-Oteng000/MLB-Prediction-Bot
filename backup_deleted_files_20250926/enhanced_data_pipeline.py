#!/usr/bin/env python3
"""
Enhanced Data Pipeline with Statcast Integration
Combines existing historical pipeline with comprehensive Statcast data collection
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import time
import logging

# Import existing components
from data_pipeline_manager import DataPipelineManager
from statcast_historical_fetcher import StatcastHistoricalFetcher
from advanced_metrics_fetcher import AdvancedMetricsFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedDataPipeline:
    """
    Enhanced pipeline that integrates:
    1. Existing historical data collection (basic stats)
    2. Historical Statcast data collection (advanced metrics)
    3. Current daily advanced metrics
    4. Unified training dataset generation
    """

    def __init__(self, db_path: str = "historical_baseball_data.db"):
        self.db_path = db_path

        # Initialize all pipeline components
        self.basic_pipeline = DataPipelineManager(db_path=db_path)
        self.statcast_historical = StatcastHistoricalFetcher(db_path=db_path)
        self.current_metrics = AdvancedMetricsFetcher()

        # Pipeline configuration
        self.config = {
            'statcast_start_year': 2018,  # Statcast data availability
            'batch_size_months': 1,       # Process 1 month at a time for Statcast
            'rate_limit_delay': 10,       # Seconds between API calls
            'enhance_existing_data': True # Link Statcast to existing records
        }

        logger.info("Enhanced Data Pipeline initialized")

    def run_complete_historical_collection(self, start_date: str = "2018-03-29",
                                         end_date: str = "2024-10-31"):
        """
        Run complete historical data collection with both basic and Statcast data

        Args:
            start_date: Start of collection (2018-03-29 for full dataset)
            end_date: End of collection (current season end)
        """
        logger.info(f"🚀 COMPLETE HISTORICAL COLLECTION: {start_date} to {end_date}")

        # Phase 1: Basic historical data (if not already complete)
        logger.info("Phase 1: Basic Historical Data Collection")
        self._ensure_basic_data_complete(start_date, end_date)

        # Phase 2: Statcast historical data collection
        logger.info("Phase 2: Statcast Historical Data Collection")
        self._collect_comprehensive_statcast_data(start_date, end_date)

        # Phase 3: Data integration and enhancement
        logger.info("Phase 3: Data Integration and Enhancement")
        self._integrate_and_enhance_data()

        # Phase 4: Validation and quality checks
        logger.info("Phase 4: Data Validation")
        validation_results = self._validate_enhanced_dataset()

        # Phase 5: Generate unified ML training datasets
        logger.info("Phase 5: ML Training Dataset Generation")
        self._generate_ml_training_datasets()

        logger.info("🏁 Complete historical collection finished!")
        return validation_results

    def _ensure_basic_data_complete(self, start_date: str, end_date: str):
        """Ensure basic historical data collection is complete"""

        # Check current status
        status = self.basic_pipeline.get_pipeline_status()
        current_range = status.get('date_range', (None, None))

        logger.info(f"Current basic data range: {current_range}")

        # Run basic collection if gaps exist
        if not current_range[0] or not current_range[1]:
            logger.info("Running basic historical data collection")
            self.basic_pipeline.run_full_historical_collection(start_date, end_date)
        else:
            # Check for gaps
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
            current_start = datetime.strptime(current_range[0], '%Y-%m-%d').date()

            if current_start > start_dt:
                logger.info(f"Filling gap: {start_date} to {current_range[0]}")
                gap_end = (current_start - timedelta(days=1)).strftime('%Y-%m-%d')
                self.basic_pipeline.run_full_historical_collection(start_date, gap_end)

    def _collect_comprehensive_statcast_data(self, start_date: str, end_date: str):
        """Collect comprehensive Statcast data for the full date range"""

        # Check what Statcast data we already have
        statcast_status = self.statcast_historical.get_statcast_collection_status()

        logger.info("Current Statcast status:")
        for table, info in statcast_status.items():
            if 'records' in info:
                logger.info(f"  {table}: {info['records']:,} records")

        # Determine date range for Statcast collection
        statcast_start = max(start_date, "2018-03-29")  # Statcast availability

        logger.info(f"Collecting Statcast data: {statcast_start} to {end_date}")

        # Collect Statcast data
        try:
            self.statcast_historical.collect_historical_statcast_data(
                start_date=statcast_start,
                end_date=end_date
            )
        except Exception as e:
            logger.error(f"Statcast collection error: {e}")
            logger.info("Continuing with available data...")

    def _integrate_and_enhance_data(self):
        """Integrate Statcast data with existing historical records"""
        logger.info("Integrating Statcast data with historical records")

        try:
            # Enhance existing player_game_results with Statcast metrics
            enhanced_records = self.statcast_historical.enhance_existing_historical_data()
            logger.info(f"Enhanced {enhanced_records:,} historical records")

            # Create comprehensive feature tables
            self._create_comprehensive_feature_tables()

        except Exception as e:
            logger.error(f"Data integration error: {e}")

    def _create_comprehensive_feature_tables(self):
        """Create comprehensive feature tables for ML training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create comprehensive player features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_player_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date TEXT NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                opponent TEXT NOT NULL,

                -- Basic stats
                at_bats INTEGER DEFAULT 0,
                hits INTEGER DEFAULT 0,
                home_runs INTEGER DEFAULT 0,
                doubles INTEGER DEFAULT 0,
                triples INTEGER DEFAULT 0,
                rbi INTEGER DEFAULT 0,
                walks INTEGER DEFAULT 0,
                strikeouts INTEGER DEFAULT 0,

                -- Statcast metrics
                avg_exit_velocity REAL,
                max_exit_velocity REAL,
                hard_hit_rate REAL,
                barrel_rate REAL,
                sweet_spot_rate REAL,
                avg_launch_angle REAL,

                -- Expected stats
                expected_ba REAL,
                expected_woba REAL,
                expected_slg REAL,

                -- Environmental
                park_hr_factor REAL,
                elevation REAL,
                day_night INTEGER,
                temperature REAL,

                -- Historical context (calculated features)
                career_hr_rate REAL,
                season_hr_rate REAL,
                last_30d_hr_rate REAL,
                last_7d_hr_rate REAL,

                -- Target variables
                target_hr INTEGER DEFAULT 0,
                target_rbi INTEGER DEFAULT 0,
                target_runs INTEGER DEFAULT 0,
                target_hits INTEGER DEFAULT 0,
                target_total_bases INTEGER DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game_date, player_name, team)
            )
        ''')

        # Populate the comprehensive features table
        populate_query = '''
            INSERT OR REPLACE INTO ml_player_features
            (game_date, player_name, team, opponent, at_bats, hits, home_runs,
             doubles, triples, rbi, walks, strikeouts, avg_exit_velocity,
             max_exit_velocity, hard_hit_rate, barrel_rate, expected_ba,
             expected_woba, park_hr_factor, elevation, day_night,
             target_hr, target_rbi, target_runs, target_hits, target_total_bases)
            SELECT
                pgr.date as game_date,
                pgr.player_name,
                pgr.team,
                pgr.opponent,
                pgr.at_bats,
                pgr.hits,
                pgr.home_runs,
                pgr.doubles,
                pgr.triples,
                pgr.rbi,
                pgr.walks,
                pgr.strikeouts,
                spd.avg_exit_velocity,
                spd.max_exit_velocity,
                spd.hard_hit_rate,
                spd.barrel_rate,
                spd.expected_ba,
                spd.expected_woba,
                g.park_hr_factor,
                g.elevation,
                g.day_night,
                pgr.home_runs as target_hr,
                pgr.rbi as target_rbi,
                -- Calculate runs and total bases from available data
                CASE WHEN pgr.home_runs > 0 THEN 1 ELSE 0 END as target_runs,
                pgr.hits as target_hits,
                (pgr.hits + pgr.doubles + 2*pgr.triples + 3*pgr.home_runs) as target_total_bases
            FROM player_game_results pgr
            JOIN games g ON pgr.game_id = g.game_id
            LEFT JOIN statcast_player_daily spd ON pgr.player_name = spd.player_name
                AND pgr.date = spd.game_date
            WHERE pgr.at_bats > 0
        '''

        cursor.execute(populate_query)
        rows_inserted = cursor.rowcount

        conn.commit()
        conn.close()

        logger.info(f"Created comprehensive ML features table with {rows_inserted:,} records")

    def _validate_enhanced_dataset(self) -> Dict:
        """Validate the quality of the enhanced dataset"""
        conn = sqlite3.connect(self.db_path)

        validation = {}

        # Check data completeness
        cursor = conn.cursor()

        # Basic stats validation
        cursor.execute('SELECT COUNT(*) FROM ml_player_features')
        total_records = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM ml_player_features WHERE avg_exit_velocity IS NOT NULL')
        statcast_enhanced = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT player_name) FROM ml_player_features WHERE target_hr > 0')
        unique_hr_hitters = cursor.fetchone()[0]

        cursor.execute('SELECT MIN(game_date), MAX(game_date) FROM ml_player_features')
        date_range = cursor.fetchone()

        # Target variable distributions
        cursor.execute('''
            SELECT
                SUM(target_hr) as total_hrs,
                SUM(target_rbi) as total_rbis,
                SUM(target_hits) as total_hits,
                AVG(CAST(target_hr AS FLOAT)) as hr_rate
            FROM ml_player_features
        ''')
        target_stats = cursor.fetchone()

        validation = {
            'total_ml_records': total_records,
            'statcast_enhanced_records': statcast_enhanced,
            'statcast_coverage_pct': (statcast_enhanced / total_records * 100) if total_records > 0 else 0,
            'unique_hr_hitters': unique_hr_hitters,
            'date_range': date_range,
            'target_variables': {
                'total_home_runs': target_stats[0],
                'total_rbis': target_stats[1],
                'total_hits': target_stats[2],
                'home_run_rate': target_stats[3]
            }
        }

        conn.close()

        logger.info("Dataset validation completed:")
        logger.info(f"  Total ML records: {validation['total_ml_records']:,}")
        logger.info(f"  Statcast coverage: {validation['statcast_coverage_pct']:.1f}%")
        logger.info(f"  Home run rate: {validation['target_variables']['home_run_rate']:.1%}")

        return validation

    def _generate_ml_training_datasets(self):
        """Generate comprehensive ML training datasets for different prediction targets"""

        datasets = {
            'home_runs': self._create_hr_training_data(),
            'rbi': self._create_rbi_training_data(),
            'runs': self._create_runs_training_data(),
            'hits': self._create_hits_training_data(),
            'total_bases': self._create_total_bases_training_data()
        }

        # Save datasets to files
        for target, (features_df, targets_df) in datasets.items():
            if not features_df.empty:
                features_file = f'ml_training_features_{target}.csv'
                targets_file = f'ml_training_targets_{target}.csv'

                features_df.to_csv(features_file, index=False)
                targets_df.to_csv(targets_file, index=False)

                logger.info(f"{target.upper()} training data: {len(features_df)} samples, {len(features_df.columns)-3} features")

    def _create_hr_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create training dataset specifically for home run predictions"""
        conn = sqlite3.connect(self.db_path)

        # Get comprehensive features for HR prediction
        query = '''
            SELECT
                game_date, player_name, team,
                -- Basic features
                CAST(at_bats AS REAL) as at_bats,
                CAST(hits AS REAL) as hits,
                CAST(doubles AS REAL) as doubles,
                CAST(walks AS REAL) as walks,
                CAST(strikeouts AS REAL) as strikeouts,

                -- Statcast features
                COALESCE(avg_exit_velocity, 85.0) as avg_exit_velocity,
                COALESCE(max_exit_velocity, 95.0) as max_exit_velocity,
                COALESCE(hard_hit_rate, 0.35) as hard_hit_rate,
                COALESCE(barrel_rate, 0.08) as barrel_rate,
                COALESCE(sweet_spot_rate, 0.30) as sweet_spot_rate,
                COALESCE(expected_ba, 0.25) as expected_ba,
                COALESCE(expected_woba, 0.32) as expected_woba,

                -- Environmental
                COALESCE(park_hr_factor, 1.0) as park_hr_factor,
                COALESCE(elevation, 0.0) as elevation,
                COALESCE(day_night, 0) as day_night,

                -- Target
                target_hr
            FROM ml_player_features
            WHERE at_bats > 0
            ORDER BY game_date, player_name
        '''

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Split features and targets
        feature_cols = [col for col in df.columns if col not in ['game_date', 'player_name', 'team', 'target_hr']]

        features_df = df[['game_date', 'player_name', 'team'] + feature_cols].copy()
        targets_df = df[['game_date', 'player_name', 'team', 'target_hr']].copy()

        return features_df, targets_df

    def _create_rbi_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create training dataset for RBI predictions"""
        # Similar to HR but with RBI-specific features
        return self._create_generic_training_data('target_rbi')

    def _create_runs_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create training dataset for runs predictions"""
        return self._create_generic_training_data('target_runs')

    def _create_hits_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create training dataset for hits predictions"""
        return self._create_generic_training_data('target_hits')

    def _create_total_bases_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create training dataset for total bases predictions"""
        return self._create_generic_training_data('target_total_bases')

    def _create_generic_training_data(self, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generic training data creation for any target variable"""
        conn = sqlite3.connect(self.db_path)

        query = f'''
            SELECT * FROM ml_player_features
            WHERE at_bats > 0
            ORDER BY game_date, player_name
        '''

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Split features and targets
        feature_cols = [col for col in df.columns
                       if col not in ['game_date', 'player_name', 'team']
                       and not col.startswith('target_')
                       and col != 'id' and col != 'created_at']

        features_df = df[['game_date', 'player_name', 'team'] + feature_cols].copy()
        targets_df = df[['game_date', 'player_name', 'team', target_column]].copy()

        return features_df, targets_df

    def get_comprehensive_status(self) -> Dict:
        """Get comprehensive status of the enhanced pipeline"""

        # Basic pipeline status
        basic_status = self.basic_pipeline.get_pipeline_status()

        # Statcast status
        statcast_status = self.statcast_historical.get_statcast_collection_status()

        # ML features status
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT COUNT(*) FROM ml_player_features')
            ml_features_count = cursor.fetchone()[0]
        except:
            ml_features_count = 0

        conn.close()

        return {
            'basic_pipeline': basic_status,
            'statcast_data': statcast_status,
            'ml_features': ml_features_count,
            'pipeline_ready': ml_features_count > 0
        }


def main():
    """Test the enhanced data pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Data Pipeline with Statcast')
    parser.add_argument('--run-complete', action='store_true',
                       help='Run complete historical collection')
    parser.add_argument('--start-date', default='2018-03-29',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-10-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--status', action='store_true',
                       help='Show comprehensive pipeline status')
    parser.add_argument('--generate-datasets', action='store_true',
                       help='Generate ML training datasets')

    args = parser.parse_args()

    # Initialize enhanced pipeline
    pipeline = EnhancedDataPipeline()

    if args.status:
        status = pipeline.get_comprehensive_status()
        print("=" * 60)
        print("ENHANCED PIPELINE STATUS")
        print("=" * 60)

        print("\nBasic Historical Data:")
        basic = status['basic_pipeline']
        print(f"  Games: {basic['games']:,}")
        print(f"  Player Records: {basic['player_records']:,}")
        print(f"  Date Range: {basic['date_range']}")

        print("\nStatcast Data:")
        for table, info in status['statcast_data'].items():
            if 'records' in info:
                print(f"  {table}: {info['records']:,}")

        print(f"\nML Features: {status['ml_features']:,}")
        print(f"Pipeline Ready: {'✅' if status['pipeline_ready'] else '❌'}")

    elif args.generate_datasets:
        pipeline._generate_ml_training_datasets()
        print("ML training datasets generated")

    elif args.run_complete:
        print("🚀 STARTING COMPLETE HISTORICAL COLLECTION")
        print(f"Date range: {args.start_date} to {args.end_date}")
        print("This will take several hours/days...")

        validation = pipeline.run_complete_historical_collection(
            args.start_date, args.end_date
        )

        print("\n🏁 COLLECTION COMPLETED!")
        print(f"Total ML records: {validation['total_ml_records']:,}")
        print(f"Statcast coverage: {validation['statcast_coverage_pct']:.1f}%")


if __name__ == "__main__":
    main()