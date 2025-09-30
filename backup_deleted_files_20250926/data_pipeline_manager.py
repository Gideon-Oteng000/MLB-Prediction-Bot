#!/usr/bin/env python3
"""
Data Pipeline Manager for Historical Baseball Data Collection
Coordinates historical_data_fetcher.py and game_results_scraper.py for massive dataset collection
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import time
import logging
from pathlib import Path

# Import our custom modules
from historical_data_fetcher import HistoricalDataFetcher
from game_results_scraper import GameResultsScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPipelineManager:
    """
    Manages the complete data collection pipeline for 2018-2024 historical data
    Coordinates fetching, processing, and quality control
    """

    def __init__(self, db_path: str = "historical_baseball_data.db"):
        self.db_path = db_path
        self.fetcher = HistoricalDataFetcher(db_path=db_path)
        self.scraper = GameResultsScraper(db_path=db_path)

        # Pipeline configuration
        self.config = {
            'batch_size_days': 30,  # Process 30 days at a time
            'max_retries': 3,
            'retry_delay': 60,  # seconds
            'quality_check_frequency': 100,  # games
            'backup_frequency': 1000,  # games
        }

        logger.info("Data Pipeline Manager initialized")

    def run_full_historical_collection(self, start_date: str = "2018-03-29", end_date: str = "2024-10-31"):
        """
        Run complete historical data collection for 2018-2024

        Args:
            start_date: Start of 2018 season
            end_date: End of 2024 season
        """
        logger.info(f"🚀 Starting FULL historical data collection: {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        total_days = (end_dt - start_dt).days + 1

        logger.info(f"📊 Total timespan: {total_days} days ({total_days/365.25:.1f} years)")

        # Phase 1: Basic data collection in batches
        self._phase_1_basic_collection(start_date, end_date)

        # Phase 2: Comprehensive game enhancement
        self._phase_2_comprehensive_enhancement()

        # Phase 3: Data quality validation
        self._phase_3_quality_validation()

        # Phase 4: Final reporting
        self._phase_4_final_report()

        logger.info("🏁 Full historical data collection completed!")

    def _phase_1_basic_collection(self, start_date: str, end_date: str):
        """Phase 1: Collect basic game data and lineups in batches"""
        logger.info("📥 Phase 1: Basic Data Collection Starting")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

        current_date = start_dt
        batch_num = 1

        while current_date <= end_dt:
            # Calculate batch end date
            batch_end = min(current_date + timedelta(days=self.config['batch_size_days'] - 1), end_dt)

            batch_start_str = current_date.strftime('%Y-%m-%d')
            batch_end_str = batch_end.strftime('%Y-%m-%d')

            logger.info(f"🔄 Processing Batch {batch_num}: {batch_start_str} to {batch_end_str}")

            try:
                # Run basic data collection for this batch
                self.fetcher.collect_historical_data(
                    start_date=batch_start_str,
                    end_date=batch_end_str,
                    skip_existing=True
                )

                # Quality check after each batch
                self._run_batch_quality_check(batch_start_str, batch_end_str)

                # Backup after large batches
                if batch_num % 10 == 0:
                    self._create_backup(f"batch_{batch_num}")

                logger.info(f"✅ Batch {batch_num} completed successfully")

            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                # Continue with next batch rather than stopping entire process

            current_date = batch_end + timedelta(days=1)
            batch_num += 1

        logger.info("📥 Phase 1: Basic Data Collection Completed")

    def _phase_2_comprehensive_enhancement(self):
        """Phase 2: Enhance all games with comprehensive statistics"""
        logger.info("⚡ Phase 2: Comprehensive Enhancement Starting")

        # Get count of games needing enhancement
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(DISTINCT g.game_id)
            FROM games g
            LEFT JOIN team_game_results tgr ON g.game_id = tgr.game_id
            WHERE tgr.game_id IS NULL
        ''')
        games_to_enhance = cursor.fetchone()[0]
        conn.close()

        logger.info(f"🎯 Found {games_to_enhance} games needing comprehensive enhancement")

        if games_to_enhance > 0:
            # Process in chunks to avoid memory issues
            chunk_size = 100
            processed = 0

            while processed < games_to_enhance:
                try:
                    logger.info(f"🔄 Enhancing games {processed + 1}-{min(processed + chunk_size, games_to_enhance)}")

                    self.scraper.enhance_existing_games(limit=chunk_size)

                    processed += chunk_size

                    # Progress backup every 500 games
                    if processed % 500 == 0:
                        self._create_backup(f"enhanced_{processed}")

                    # Brief pause to avoid overwhelming APIs
                    time.sleep(10)

                except Exception as e:
                    logger.error(f"❌ Enhancement chunk failed: {e}")
                    # Continue processing

        logger.info("⚡ Phase 2: Comprehensive Enhancement Completed")

    def _phase_3_quality_validation(self):
        """Phase 3: Data quality validation and integrity checks"""
        logger.info("🔍 Phase 3: Quality Validation Starting")

        validation_results = {}

        # Check 1: Game counts by year
        validation_results['games_by_year'] = self._validate_games_by_year()

        # Check 2: Home run counts and consistency
        validation_results['home_run_validation'] = self._validate_home_runs()

        # Check 3: Player statistics consistency
        validation_results['player_stats_validation'] = self._validate_player_stats()

        # Check 4: Team statistics consistency
        validation_results['team_stats_validation'] = self._validate_team_stats()

        # Check 5: Data completeness
        validation_results['completeness_check'] = self._validate_data_completeness()

        # Save validation report
        self._save_validation_report(validation_results)

        logger.info("🔍 Phase 3: Quality Validation Completed")

    def _phase_4_final_report(self):
        """Phase 4: Generate comprehensive final report"""
        logger.info("📋 Phase 4: Final Reporting Starting")

        # Get comprehensive statistics
        final_stats = self._generate_final_statistics()

        # Create summary report
        self._create_final_report(final_stats)

        logger.info("📋 Phase 4: Final Reporting Completed")

    def _run_batch_quality_check(self, start_date: str, end_date: str):
        """Quick quality check for a batch"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check games collected in this batch
        cursor.execute('''
            SELECT COUNT(*) FROM games
            WHERE date BETWEEN ? AND ?
        ''', (start_date, end_date))
        games_count = cursor.fetchone()[0]

        # Check player results
        cursor.execute('''
            SELECT COUNT(*) FROM player_game_results
            WHERE date BETWEEN ? AND ?
        ''', (start_date, end_date))
        players_count = cursor.fetchone()[0]

        conn.close()

        logger.info(f"📊 Batch Quality: {games_count} games, {players_count} player records")

        if games_count == 0:
            logger.warning(f"⚠️  No games found for batch {start_date} to {end_date}")

    def _validate_games_by_year(self) -> Dict:
        """Validate expected number of games per year"""
        conn = sqlite3.connect(self.db_path)

        query = '''
            SELECT SUBSTR(date, 1, 4) as year, COUNT(*) as game_count
            FROM games
            GROUP BY SUBSTR(date, 1, 4)
            ORDER BY year
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Expected games per year (approximately)
        expected_games = {
            '2018': 2400,  # ~162 games × 15 teams (season started late)
            '2019': 2400,
            '2020': 900,   # COVID shortened season
            '2021': 2400,
            '2022': 2400,
            '2023': 2400,
            '2024': 2400
        }

        validation = {}
        for _, row in df.iterrows():
            year = row['year']
            actual = row['game_count']
            expected = expected_games.get(year, 2400)

            validation[year] = {
                'actual': actual,
                'expected': expected,
                'percentage': (actual / expected) * 100 if expected > 0 else 0
            }

        return validation

    def _validate_home_runs(self) -> Dict:
        """Validate home run data consistency"""
        conn = sqlite3.connect(self.db_path)

        # Total home runs by year
        query = '''
            SELECT SUBSTR(date, 1, 4) as year,
                   SUM(home_runs) as total_hrs,
                   COUNT(DISTINCT player_name) as unique_hitters
            FROM player_game_results
            WHERE home_runs > 0
            GROUP BY SUBSTR(date, 1, 4)
            ORDER BY year
        '''
        df = pd.read_sql_query(query, conn)

        # Check for anomalies
        query_anomalies = '''
            SELECT player_name, date, home_runs, at_bats
            FROM player_game_results
            WHERE home_runs > at_bats OR home_runs > 4
        '''
        anomalies_df = pd.read_sql_query(query_anomalies, conn)

        conn.close()

        return {
            'yearly_totals': df.to_dict('records'),
            'anomalies': len(anomalies_df),
            'anomaly_details': anomalies_df.to_dict('records')
        }

    def _validate_player_stats(self) -> Dict:
        """Validate player statistics for consistency"""
        conn = sqlite3.connect(self.db_path)

        # Check for impossible statistics
        query = '''
            SELECT COUNT(*) as inconsistent_records
            FROM player_game_results
            WHERE hits > at_bats
               OR home_runs > hits
               OR rbi < 0
               OR strikeouts > at_bats
        '''
        cursor = conn.cursor()
        cursor.execute(query)
        inconsistent = cursor.fetchone()[0]

        # Top home run hitters
        query_top_hrs = '''
            SELECT player_name, SUM(home_runs) as total_hrs, COUNT(*) as games
            FROM player_game_results
            WHERE home_runs > 0
            GROUP BY player_name
            ORDER BY total_hrs DESC
            LIMIT 10
        '''
        top_hr_df = pd.read_sql_query(query_top_hrs, conn)

        conn.close()

        return {
            'inconsistent_records': inconsistent,
            'top_home_run_hitters': top_hr_df.to_dict('records')
        }

    def _validate_team_stats(self) -> Dict:
        """Validate team statistics consistency"""
        conn = sqlite3.connect(self.db_path)

        # Check win/loss consistency
        query = '''
            SELECT COUNT(*) as total_team_games,
                   SUM(win) as total_wins
            FROM team_game_results
        '''
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        total_games = result[0]
        total_wins = result[1]

        # Each game should have exactly one winner (total_wins = total_games / 2)
        expected_wins = total_games / 2

        conn.close()

        return {
            'total_team_games': total_games,
            'total_wins': total_wins,
            'expected_wins': expected_wins,
            'win_loss_consistency': abs(total_wins - expected_wins) < 10
        }

    def _validate_data_completeness(self) -> Dict:
        """Check data completeness across all tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        completeness = {}

        # Tables to check
        tables = [
            'games', 'player_game_results', 'team_game_results',
            'pitcher_game_results', 'advanced_player_stats'
        ]

        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            completeness[table] = count

        # Check for games with missing enhancements
        cursor.execute('''
            SELECT COUNT(*) FROM games g
            LEFT JOIN team_game_results tgr ON g.game_id = tgr.game_id
            WHERE tgr.game_id IS NULL
        ''')
        missing_enhancements = cursor.fetchone()[0]
        completeness['missing_enhancements'] = missing_enhancements

        conn.close()
        return completeness

    def _create_backup(self, backup_name: str):
        """Create database backup"""
        backup_path = f"backups/historical_data_{backup_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        os.makedirs("backups", exist_ok=True)

        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"💾 Backup created: {backup_path}")
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")

    def _save_validation_report(self, validation_results: Dict):
        """Save validation report to file"""
        report_path = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(report_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            logger.info(f"📊 Validation report saved: {report_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save validation report: {e}")

    def _generate_final_statistics(self) -> Dict:
        """Generate comprehensive final statistics"""
        stats = self.scraper.get_comprehensive_stats_summary()

        # Add pipeline-specific stats
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Collection status
        cursor.execute('SELECT COUNT(*) FROM collection_status WHERE lineups_collected = 1')
        collected_dates = cursor.fetchone()[0]

        # Date range
        cursor.execute('SELECT MIN(date), MAX(date) FROM games')
        date_range = cursor.fetchone()

        conn.close()

        stats.update({
            'pipeline_info': {
                'collected_dates': collected_dates,
                'date_range': date_range,
                'collection_completed': datetime.now().isoformat()
            }
        })

        return stats

    def _create_final_report(self, final_stats: Dict):
        """Create comprehensive final report"""
        report_content = f"""
# Historical Baseball Data Collection Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Collection Summary
- **Date Range**: {final_stats['date_range'][0]} to {final_stats['date_range'][1]}
- **Total Games**: {final_stats['games']:,}
- **Total Player Records**: {final_stats['player_game_results']:,}
- **Total Home Runs**: {final_stats['home_run_stats']['total_home_runs']:,}
- **Unique HR Hitters**: {final_stats['home_run_stats']['unique_hr_hitters']:,}

## Database Tables
- Games: {final_stats['games']:,} records
- Player Game Results: {final_stats['player_game_results']:,} records
- Team Game Results: {final_stats['team_game_results']:,} records
- Pitcher Game Results: {final_stats['pitcher_game_results']:,} records
- Advanced Player Stats: {final_stats['advanced_player_stats']:,} records
- Inning Scoring: {final_stats['inning_scoring']:,} records

## Pipeline Information
- Collection Dates Processed: {final_stats['pipeline_info']['collected_dates']:,}
- Collection Completed: {final_stats['pipeline_info']['collection_completed']}

## Home Run Analysis
- Maximum HRs in Single Game: {final_stats['home_run_stats']['max_hrs_in_game']}
- Average HRs per Game: {final_stats['home_run_stats']['total_home_runs'] / final_stats['games']:.2f}

---
*Generated by Data Pipeline Manager*
"""

        report_path = f"FINAL_COLLECTION_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        try:
            with open(report_path, 'w') as f:
                f.write(report_content)
            logger.info(f"📋 Final report created: {report_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create final report: {e}")

    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        conn = sqlite3.connect(self.db_path)

        # Basic counts
        counts_query = '''
            SELECT
                (SELECT COUNT(*) FROM games) as games,
                (SELECT COUNT(*) FROM player_game_results) as players,
                (SELECT COUNT(*) FROM team_game_results) as teams,
                (SELECT COUNT(*) FROM pitcher_game_results) as pitchers
        '''
        cursor = conn.cursor()
        cursor.execute(counts_query)
        counts = cursor.fetchone()

        # Date range
        cursor.execute('SELECT MIN(date), MAX(date) FROM games')
        date_range = cursor.fetchone()

        # Latest collection
        cursor.execute('SELECT MAX(created_at) FROM games')
        latest = cursor.fetchone()[0]

        conn.close()

        return {
            'games': counts[0],
            'player_records': counts[1],
            'team_records': counts[2],
            'pitcher_records': counts[3],
            'date_range': date_range,
            'latest_collection': latest
        }


def main():
    """Main function for pipeline management"""
    import argparse

    parser = argparse.ArgumentParser(description='Historical Baseball Data Pipeline Manager')
    parser.add_argument('--run-full', action='store_true',
                       help='Run full historical collection (2018-2024)')
    parser.add_argument('--start-date', default='2018-03-29',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-10-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--status', action='store_true',
                       help='Show current pipeline status')
    parser.add_argument('--db-path', default='historical_baseball_data.db',
                       help='Database file path')

    args = parser.parse_args()

    # Create pipeline manager
    pipeline = DataPipelineManager(db_path=args.db_path)

    if args.status:
        status = pipeline.get_pipeline_status()
        print("Pipeline Status:")
        print("=" * 40)
        for key, value in status.items():
            print(f"{key}: {value}")

    elif args.run_full:
        print("WARNING: Full historical collection will take many hours!")
        print(f"Date range: {args.start_date} to {args.end_date}")
        print("Starting in 10 seconds... (Ctrl+C to cancel)")
        time.sleep(10)

        pipeline.run_full_historical_collection(
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        print("Use --status to check pipeline status or --run-full to start collection")


if __name__ == "__main__":
    main()