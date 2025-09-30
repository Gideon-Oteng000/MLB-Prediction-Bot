#!/usr/bin/env python3
"""
Start Full MLB Historical Data Collection (2018-2024)
This script will collect all 6+ years of data for ML training
"""

from mlb_historical_data_pipeline import MLBDataPipeline, PipelineConfig
import sys
import os
from datetime import datetime

def main():
    print("🏆 MLB HISTORICAL DATA COLLECTION (2018-2024)")
    print("=" * 60)
    print("This will collect 6+ years of comprehensive MLB data for ML training")
    print("Expected runtime: 6-12 hours")
    print("Database size: ~2-5 GB")
    print("Target predictions: HR, Hits, RBI, Runs, Total Bases, Singles, Doubles, Triples, K's")
    print()

    # Configuration for full historical collection
    config = PipelineConfig(
        START_YEAR=2018,
        END_YEAR=2024,
        DB_PATH="mlb_training_data.db",
        RATE_LIMIT_DELAY=1.5,  # Be respectful to APIs
        BATCH_SIZE=50
    )

    print(f"Configuration:")
    print(f"  Years: {config.START_YEAR}-{config.END_YEAR}")
    print(f"  Database: {config.DB_PATH}")
    print(f"  Rate limit: {config.RATE_LIMIT_DELAY}s between calls")
    print(f"  OpenWeather API: Configured")
    print()

    # Check if database already exists
    if os.path.exists(config.DB_PATH):
        pipeline = MLBDataPipeline(config)
        status = pipeline.get_pipeline_status()

        print("📊 Current database status:")
        for table, count in status.items():
            if isinstance(count, int):
                print(f"  {table}: {count:,}")

        print(f"  Completed games: {status['completed_games']:,}")
        if status['game_date_range'][0]:
            print(f"  Date range: {status['game_date_range'][0]} to {status['game_date_range'][1]}")
        print()

    confirm = input("🚀 Start/Continue full historical collection? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Collection cancelled")
        return

    print("\n🏁 STARTING FULL HISTORICAL DATA COLLECTION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Log file: data_pipeline.log")
    print("This process is restartable - it will skip completed games")
    print("Press Ctrl+C to stop (data will be saved)")
    print()

    try:
        # Initialize and run pipeline
        pipeline = MLBDataPipeline(config)
        pipeline.run_full_pipeline()

        # Show final status
        print("\n🎉 COLLECTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        final_status = pipeline.get_pipeline_status()

        print("Final database contents:")
        for table, count in final_status.items():
            if isinstance(count, int):
                print(f"  {table}: {count:,}")

        print(f"\nDatabase file: {config.DB_PATH}")
        print("Ready for ML training!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Collection interrupted by user")
        print("Progress has been saved - restart anytime to continue")

        # Show current status
        status = pipeline.get_pipeline_status()
        print(f"Current progress: {status['completed_games']:,} games completed")

    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        print("Check data_pipeline.log for details")
        print("The pipeline can be restarted to continue from where it left off")

if __name__ == "__main__":
    main()