"""
Auto-Tracking Integration Script
Automatically runs the HR prediction model and sets up tracking
"""

import os
import sys
import subprocess
import glob
from datetime import datetime, timedelta
import pandas as pd

# Import the tracking system
try:
    from hr_tracking_system import HRPredictionTracker
except ImportError:
    print("❌ Error: hr_tracking_system.py not found. Please ensure both files are in the same directory.")
    sys.exit(1)

class AutoTracker:
    """Automated tracking system for HR predictions"""
    
    def __init__(self):
        self.tracker = HRPredictionTracker()
        self.prediction_script = "mlb_hr_clean_v4.py"
        
    def run_daily_pipeline(self):
        """Complete daily pipeline: predict -> track -> report"""
        print("🚀 Starting automated HR prediction tracking pipeline")
        print("=" * 60)
        
        # Step 1: Run prediction model
        print("📊 Step 1: Running HR prediction model...")
        prediction_file = self._run_prediction_model()
        
        if not prediction_file:
            print("❌ Failed to generate predictions")
            return False
        
        # Step 2: Import predictions
        print(f"📝 Step 2: Importing predictions from {prediction_file}...")
        if not self.tracker.import_predictions_from_csv(prediction_file):
            print("❌ Failed to import predictions")
            return False
        
        # Step 3: Check if we should verify results
        date_str = self.tracker._extract_date_from_filename(prediction_file)
        prediction_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # Only check results if the prediction date has passed
        if prediction_date <= datetime.now().date():
            print(f"✅ Step 3: Checking actual results for {prediction_date}...")
            self.tracker.check_actual_results(date_str)
            
            print(f"📋 Step 4: Displaying tracking results...")
            self.tracker.display_tracking_results(date_str)
        else:
            print(f"⏳ Step 3: Predictions are for future date {prediction_date}, skipping result check")
            print(f"📋 Step 4: Displaying predictions...")
            self.tracker.display_tracking_results(date_str)
        
        print("\n✅ Pipeline completed successfully!")
        return True
    
    def _run_prediction_model(self):
        """Run the original prediction script and return CSV filename"""
        try:
            # Check if prediction script exists
            if not os.path.exists(self.prediction_script):
                print(f"❌ Prediction script not found: {self.prediction_script}")
                print("   Please ensure mlb_hr_clean_v4.py is in the same directory")
                return None
            
            # Get existing CSV files before running
            existing_files = set(glob.glob("hr_predictions_*.csv"))
            
            # Run the prediction script
            print(f"   Running {self.prediction_script}...")
            result = subprocess.run([sys.executable, self.prediction_script], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"❌ Prediction script failed:")
                print(f"   Error: {result.stderr}")
                return None
            
            # Find the new CSV file
            new_files = set(glob.glob("hr_predictions_*.csv")) - existing_files
            
            if not new_files:
                print("❌ No new prediction file was created")
                return None
            
            # Return the newest file
            newest_file = max(new_files, key=os.path.getctime)
            print(f"✅ Generated predictions: {newest_file}")
            return newest_file
            
        except subprocess.TimeoutExpired:
            print("❌ Prediction script timed out (>5 minutes)")
            return None
        except Exception as e:
            print(f"❌ Error running prediction script: {e}")
            return None
    
    def backfill_tracking(self, days_back=7):
        """Backfill tracking for existing prediction files"""
        print(f"🔄 Backfilling tracking data for last {days_back} days...")
        
        # Find all prediction CSV files
        csv_files = glob.glob("hr_predictions_*.csv")
        
        if not csv_files:
            print("❌ No prediction CSV files found")
            return
        
        processed = 0
        
        for csv_file in sorted(csv_files):
            try:
                # Extract date from filename
                date_str = self.tracker._extract_date_from_filename(csv_file)
                file_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                # Skip if too old
                if file_date < datetime.now().date() - timedelta(days=days_back):
                    continue
                
                print(f"\n📅 Processing {csv_file} ({file_date})")
                
                # Import predictions
                if self.tracker.import_predictions_from_csv(csv_file):
                    processed += 1
                    
                    # Check results if date has passed
                    if file_date <= datetime.now().date():
                        self.tracker.check_actual_results(date_str)
                
            except Exception as e:
                print(f"⚠️ Error processing {csv_file}: {e}")
                continue
        
        print(f"\n✅ Backfilled {processed} prediction files")
        
        # Show performance report
        if processed > 0:
            print(f"\n📊 Performance summary:")
            self.tracker.generate_performance_report(days_back)
    
    def setup_daily_schedule(self):
        """Provide instructions for setting up daily automation"""
        print("⏰ Setting up Daily Automation")
        print("=" * 40)
        print("To automatically run predictions and tracking daily:")
        print()
        
        script_path = os.path.abspath(__file__)
        
        if os.name == 'nt':  # Windows
            print("Windows Task Scheduler:")
            print(f"1. Open Task Scheduler")
            print(f"2. Create Basic Task")
            print(f"3. Set trigger: Daily at 4:00 PM")
            print(f"4. Action: Start a program")
            print(f"5. Program: {sys.executable}")
            print(f"6. Arguments: {script_path} --daily")
            print(f"7. Start in: {os.path.dirname(script_path)}")
        else:  # Linux/Mac
            print("Cron job setup:")
            print(f"1. Edit crontab: crontab -e")
            print(f"2. Add line: 0 16 * * * cd {os.path.dirname(script_path)} && {sys.executable} {script_path} --daily")
            print(f"   (This runs daily at 4:00 PM)")
        
        print()
        print("Manual daily run:")
        print(f"python {os.path.basename(__file__)} --daily")
    
    def export_results(self, format='csv'):
        """Export tracking results for analysis"""
        print(f"📊 Exporting tracking results to {format.upper()}...")
        
        # Connect to database
        cursor = self.tracker.conn.cursor()
        
        # Get comprehensive results
        cursor.execute('''
            SELECT p.date, p.rank_position, p.player_name, p.pitcher_name, 
                   p.teams, p.hr_probability, p.venue,
                   COALESCE(r.hit_hr, 0) as hit_hr, 
                   COALESCE(r.hr_count, 0) as hr_count,
                   COALESCE(r.at_bats, 0) as at_bats,
                   r.game_result
            FROM predictions p
            LEFT JOIN results r ON p.id = r.prediction_id
            ORDER BY p.date DESC, p.rank_position
        ''')
        
        data = cursor.fetchall()
        
        if not data:
            print("❌ No data to export")
            return
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=[
            'Date', 'Rank', 'Player', 'Pitcher', 'Teams', 'HR_Probability', 'Venue',
            'Hit_HR', 'HR_Count', 'At_Bats', 'Game_Result'
        ])
        
        # Export
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"hr_tracking_export_{timestamp}.{format}"
        
        if format == 'csv':
            df.to_csv(filename, index=False)
        elif format == 'excel':
            df.to_excel(filename, index=False)
        
        print(f"✅ Exported {len(df)} records to {filename}")
        
        # Show summary stats
        total_checked = len(df[df['Hit_HR'].notna()])
        total_hits = df['Hit_HR'].sum()
        hit_rate = (total_hits / total_checked * 100) if total_checked > 0 else 0
        
        print(f"📈 Summary: {total_hits}/{total_checked} HRs hit ({hit_rate:.1f}% success rate)")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MLB HR Prediction Auto-Tracking System')
    parser.add_argument('--daily', action='store_true', help='Run daily pipeline')
    parser.add_argument('--backfill', type=int, default=7, help='Backfill tracking for N days')
    parser.add_argument('--setup', action='store_true', help='Show automation setup instructions')
    parser.add_argument('--export', choices=['csv', 'excel'], help='Export tracking results')
    parser.add_argument('--check-date', type=str, help='Check results for specific date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    auto_tracker = AutoTracker()
    
    if args.daily:
        auto_tracker.run_daily_pipeline()
    elif args.setup:
        auto_tracker.setup_daily_schedule()
    elif args.export:
        auto_tracker.export_results(args.export)
    elif args.check_date:
        auto_tracker.tracker.check_actual_results(args.check_date)
        auto_tracker.tracker.display_tracking_results(args.check_date)
    elif args.backfill:
        auto_tracker.backfill_tracking(args.backfill)
    else:
        # Interactive mode
        print("🎯 MLB HR Prediction Auto-Tracking System")
        print("=" * 50)
        print("1. Run daily pipeline (predict + track)")
        print("2. Backfill tracking from existing CSVs")
        print("3. Check specific date results")
        print("4. Export tracking data")
        print("5. Performance report")
        print("6. Setup automation")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            auto_tracker.run_daily_pipeline()
        elif choice == "2":
            days = input("How many days back? (default 7): ").strip()
            days = int(days) if days.isdigit() else 7
            auto_tracker.backfill_tracking(days)
        elif choice == "3":
            date = input("Enter date (YYYY-MM-DD): ").strip()
            auto_tracker.tracker.check_actual_results(date)
            auto_tracker.tracker.display_tracking_results(date)
        elif choice == "4":
            fmt = input("Export format (csv/excel): ").strip().lower()
            if fmt in ['csv', 'excel']:
                auto_tracker.export_results(fmt)
            else:
                print("❌ Invalid format")
        elif choice == "5":
            days = input("How many days? (default 7): ").strip()
            days = int(days) if days.isdigit() else 7
            auto_tracker.tracker.generate_performance_report(days)
        elif choice == "6":
            auto_tracker.setup_daily_schedule()
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()