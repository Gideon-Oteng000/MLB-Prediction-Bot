#!/usr/bin/env python3
"""
Restart the MLB Pipeline with Fixes Applied
This script will restart your collection with the barrel error fixed
"""

from mlb_historical_data_pipeline import MLBDataPipeline, PipelineConfig
import os
from datetime import datetime

def main():
    print("🔧 MLB PIPELINE RESTART WITH FIXES")
    print("=" * 50)
    print("✅ Fixed: 'barrel' column error for early 2018 data")
    print("✅ Fixed: Missing expected stats columns")
    print("✅ Added: Barrel rate estimation from exit velocity + launch angle")
    print()

    # Check current status
    db_path = "mlb_training_data.db"
    if os.path.exists(db_path):
        config = PipelineConfig(DB_PATH=db_path)
        pipeline = MLBDataPipeline(config)
        status = pipeline.get_pipeline_status()

        print("📊 Current Progress:")
        print(f"  Completed Games: {status['completed_games']:,}")
        print(f"  Total Records:")
        for table, count in status.items():
            if isinstance(count, int) and table != 'completed_games':
                print(f"    {table}: {count:,}")

        if status['game_date_range'][0]:
            print(f"  Date Range: {status['game_date_range'][0]} to {status['game_date_range'][1]}")
        print()

        print("🚀 Pipeline will continue from where it left off")
        print("   (Previously processed games will be skipped)")
    else:
        print("🆕 Starting fresh collection")

    print()
    confirm = input("Continue with fixed pipeline? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Restart cancelled")
        return

    print(f"\n🏁 RESTARTING PIPELINE WITH FIXES")
    print("=" * 50)
    print(f"Restarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("The 'barrel' errors should now be resolved!")
    print()

    try:
        # Configuration
        config = PipelineConfig(
            START_YEAR=2018,
            END_YEAR=2024,
            DB_PATH="mlb_training_data.db",
            RATE_LIMIT_DELAY=1.5
        )

        # Run pipeline
        pipeline = MLBDataPipeline(config)
        pipeline.run_full_pipeline()

        print("\n🎉 COLLECTION COMPLETED!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted")
        print("Progress saved - can restart anytime")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Check data_pipeline.log for details")

if __name__ == "__main__":
    main()