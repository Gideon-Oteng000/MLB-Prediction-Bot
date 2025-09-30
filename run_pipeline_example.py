#!/usr/bin/env python3
"""
MLB Data Pipeline - Example Usage Script
Demonstrates how to run the pipeline with different configurations
"""

import sys
import os
from datetime import datetime, date

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mlb_historical_data_pipeline import MLBDataPipeline, PipelineConfig, DatabaseManager

def test_single_day():
    """Test pipeline with a single day"""
    print("🧪 Testing pipeline with single day (2024-09-01)")

    config = PipelineConfig(
        START_YEAR=2024,
        END_YEAR=2024,
        DB_PATH="test_mlb_data.db",
        RATE_LIMIT_DELAY=0.5
    )

    pipeline = MLBDataPipeline(config)

    # Process a single date
    test_date = date(2024, 9, 1)
    result = pipeline.process_date(test_date)

    print(f"✅ Test completed: {result}")

    # Show status
    status = pipeline.get_pipeline_status()
    print("\n📊 Test Results:")
    for table, count in status.items():
        if isinstance(count, int):
            print(f"  {table}: {count}")

def test_week():
    """Test pipeline with a week of data"""
    print("🧪 Testing pipeline with one week (2024-09-01 to 2024-09-07)")

    config = PipelineConfig(
        START_YEAR=2024,
        END_YEAR=2024,
        DB_PATH="test_week_mlb_data.db",
        RATE_LIMIT_DELAY=0.5
    )

    pipeline = MLBDataPipeline(config)

    # Process week
    for day in range(1, 8):
        test_date = date(2024, 9, day)
        result = pipeline.process_date(test_date)
        print(f"  {test_date}: {result}")

    # Show final status
    status = pipeline.get_pipeline_status()
    print("\n📊 Week Test Results:")
    for table, count in status.items():
        if isinstance(count, int):
            print(f"  {table}: {count}")

def run_full_season():
    """Run pipeline for a full season"""
    print("🚀 Running pipeline for full 2024 season")

    config = PipelineConfig(
        START_YEAR=2024,
        END_YEAR=2024,
        DB_PATH="mlb_2024_data.db",
        RATE_LIMIT_DELAY=1.0
    )

    pipeline = MLBDataPipeline(config)
    pipeline.run_full_pipeline()

def run_historical_data():
    """Run pipeline for full historical data (2018-2024)"""
    print("🚀 Running FULL HISTORICAL DATA PIPELINE (2018-2024)")
    print("⚠️  WARNING: This will take MANY HOURS to complete!")
    print("   - Approximately 6-12 hours depending on your internet speed")
    print("   - The pipeline is restartable if interrupted")
    print("   - Monitor the log file: data_pipeline.log")

    confirm = input("\nProceed with full historical collection? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Cancelled")
        return

    config = PipelineConfig(
        START_YEAR=2018,
        END_YEAR=2024,
        DB_PATH="mlb_training_data.db",
        RATE_LIMIT_DELAY=1.5  # Be respectful to APIs
    )

    pipeline = MLBDataPipeline(config)
    pipeline.run_full_pipeline()

def show_status():
    """Show current pipeline status"""
    db_files = [
        "mlb_training_data.db",
        "mlb_2024_data.db",
        "test_mlb_data.db",
        "test_week_mlb_data.db"
    ]

    for db_file in db_files:
        if os.path.exists(db_file):
            print(f"\n📊 STATUS: {db_file}")
            print("=" * 50)

            db_manager = DatabaseManager(db_file)
            pipeline = MLBDataPipeline(PipelineConfig(DB_PATH=db_file))
            status = pipeline.get_pipeline_status()

            for key, value in status.items():
                if key == 'game_date_range':
                    print(f"Date Range: {value[0]} to {value[1]}")
                else:
                    print(f"{key}: {value:,}")

def create_sample_query():
    """Create a sample query to demonstrate the database structure"""
    print("📝 Creating sample SQL query for ML training")

    sample_query = """
-- Sample query to create ML training dataset
-- This joins all tables to create features and labels for each player-game

SELECT
    -- Game context
    g.game_id,
    g.date,
    g.home_team,
    g.away_team,
    g.venue,
    g.park_factor_hr,
    g.park_factor_doubles,
    g.temperature,
    g.wind_speed,

    -- Player info
    p.player_id,
    p.name,
    p.team,
    p.position,
    p.bats,

    -- Lineup position
    l.batting_order,

    -- Batter advanced metrics (features)
    bs.barrel_rate,
    bs.hard_hit_rate,
    bs.avg_exit_velocity,
    bs.max_exit_velocity,
    bs.avg_launch_angle,
    bs.sweet_spot_pct,
    bs.pull_pct,
    bs.iso,
    bs.xslg,
    bs.xwoba,
    bs.k_pct,
    bs.bb_pct,
    bs.whiff_pct,
    bs.sprint_speed,

    -- Pitcher metrics (features)
    ps.hr_per_9 as pitcher_hr_per_9,
    ps.barrel_rate_allowed,
    ps.hard_hit_rate_allowed,
    ps.avg_exit_velocity_allowed,
    ps.k_pct as pitcher_k_pct,
    ps.bb_pct as pitcher_bb_pct,
    ps.era,
    ps.whip,
    ps.fb_velocity,

    -- OUTCOMES (LABELS for ML)
    gl.hr,
    gl.hit,
    gl.rbi,
    gl.run,
    gl.total_bases,
    gl.single,
    gl.double,
    gl.triple,
    gl.strikeouts

FROM games g
JOIN lineups l ON g.game_id = l.game_id
JOIN players p ON l.player_id = p.player_id
LEFT JOIN batter_stats bs ON p.player_id = bs.player_id AND g.date = bs.game_date
LEFT JOIN pitcher_stats ps ON p.player_id = ps.player_id AND g.date = ps.game_date
LEFT JOIN game_logs gl ON g.game_id = gl.game_id AND p.player_id = gl.player_id

WHERE l.role = 'batter'
  AND g.date >= '2018-03-01'
  AND g.date <= '2024-11-30'
  AND gl.hr IS NOT NULL  -- Only completed games

ORDER BY g.date DESC, g.game_id, l.batting_order;
"""

    # Save query to file
    with open('sample_ml_query.sql', 'w') as f:
        f.write(sample_query)

    print("✅ Sample query saved to: sample_ml_query.sql")
    print("\nThis query creates a training dataset where each row is a player-game with:")
    print("  - Features: All advanced metrics up to that game")
    print("  - Labels: Actual game outcomes (HR, hits, RBI, etc.)")
    print("  - Context: Park factors, weather, batting order")

def main():
    """Main menu for pipeline operations"""
    print("\n🏆 MLB DATA PIPELINE - MENU")
    print("=" * 50)
    print("1. Test single day (quick)")
    print("2. Test one week (medium)")
    print("3. Run full 2024 season")
    print("4. Run FULL HISTORICAL (2018-2024) - LONG!")
    print("5. Show database status")
    print("6. Create sample ML query")
    print("0. Exit")

    while True:
        try:
            choice = input("\nSelect option (0-6): ").strip()

            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                test_single_day()
            elif choice == '2':
                test_week()
            elif choice == '3':
                run_full_season()
            elif choice == '4':
                run_historical_data()
            elif choice == '5':
                show_status()
            elif choice == '6':
                create_sample_query()
            else:
                print("❌ Invalid choice. Please select 0-6.")

        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()