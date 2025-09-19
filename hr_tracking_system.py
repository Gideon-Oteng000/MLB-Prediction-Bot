"""
MLB Home Run Prediction Tracking System
Tracks predictions from mlb_hr_clean_v4.py and validates against actual results
"""

import pandas as pd
import numpy as np
import sqlite3
import requests
import statsapi
import json
from datetime import datetime, timedelta
import os
import time
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class HRPredictionTracker:
    """Main tracking system for home run predictions"""
    
    def __init__(self, db_name="hr_tracking.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.setup_database()
        
    def setup_database(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                player_name TEXT NOT NULL,
                pitcher_name TEXT NOT NULL,
                teams TEXT NOT NULL,
                hr_probability REAL NOT NULL,
                venue TEXT,
                rank_position INTEGER,
                prediction_source TEXT DEFAULT 'mlb_hr_clean_v4',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                player_name TEXT NOT NULL,
                date DATE NOT NULL,
                hit_hr BOOLEAN NOT NULL,
                hr_count INTEGER DEFAULT 0,
                at_bats INTEGER DEFAULT 0,
                game_id TEXT,
                game_result TEXT,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            )
        ''')
        
        # Daily summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date DATE PRIMARY KEY,
                total_predictions INTEGER,
                total_hrs INTEGER,
                hit_rate REAL,
                top_3_hits INTEGER,
                top_5_hits INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        print(f"✅ Database initialized: {self.db_name}")
    
    def import_predictions_from_csv(self, csv_file: str) -> bool:
        """Import predictions from the CSV output of mlb_hr_clean_v4.py"""
        try:
            if not os.path.exists(csv_file):
                print(f"❌ File not found: {csv_file}")
                return False
            
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            required_cols = ['Hitter', 'Pitcher', 'Teams', 'HR_Probability']
            if not all(col in df.columns for col in required_cols):
                print(f"❌ Missing required columns. Found: {df.columns.tolist()}")
                return False
            
            # Extract date from filename or use today
            date_str = self._extract_date_from_filename(csv_file)
            prediction_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            cursor = self.conn.cursor()
            
            # Clear existing predictions for this date
            cursor.execute('DELETE FROM predictions WHERE date = ?', (prediction_date,))
            
            # Insert new predictions
            for idx, row in df.iterrows():
                cursor.execute('''
                    INSERT INTO predictions 
                    (date, player_name, pitcher_name, teams, hr_probability, rank_position)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_date,
                    row['Hitter'],
                    row['Pitcher'], 
                    row['Teams'],
                    row['HR_Probability'],
                    idx + 1
                ))
            
            self.conn.commit()
            print(f"✅ Imported {len(df)} predictions for {prediction_date}")
            return True
            
        except Exception as e:
            print(f"❌ Error importing predictions: {e}")
            return False
    
    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract date from filename like 'hr_predictions_20241215_1430.csv'"""
        try:
            # Look for pattern like 20241215
            import re
            date_match = re.search(r'(\d{8})', filename)
            if date_match:
                date_str = date_match.group(1)
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except:
            pass
        
        # Default to today
        return datetime.now().strftime('%Y-%m-%d')
    
    def check_actual_results(self, date: str = None) -> Dict:
        """Check actual HR results for a specific date"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d').date()
        except:
            print(f"❌ Invalid date format: {date}")
            return {}
        
        print(f"🔍 Checking actual results for {target_date}...")
        
        # Get predictions for this date
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, player_name, pitcher_name, teams, hr_probability, rank_position
            FROM predictions 
            WHERE date = ?
            ORDER BY rank_position
        ''', (target_date,))
        
        predictions = cursor.fetchall()
        
        if not predictions:
            print(f"❌ No predictions found for {target_date}")
            return {}
        
        print(f"📋 Found {len(predictions)} predictions to check")
        
        results = {
            'date': target_date,
            'total_predictions': len(predictions),
            'total_hrs': 0,
            'hit_rate': 0.0,
            'top_3_hits': 0,
            'top_5_hits': 0,
            'details': []
        }
        
        # Get MLB games for this date
        mlb_date = target_date.strftime('%m/%d/%Y')
        
        try:
            schedule = statsapi.schedule(date=mlb_date)
            game_ids = [game['game_id'] for game in schedule if game['status'] == 'Final']
            
            print(f"🎮 Found {len(game_ids)} completed games")
            
            if not game_ids:
                print(f"⚠️ No completed games found for {target_date}")
                return results
            
        except Exception as e:
            print(f"❌ Error getting game schedule: {e}")
            return results
        
        # Check each prediction
        for pred_id, player_name, pitcher_name, teams, hr_prob, rank_pos in predictions:
            print(f"  Checking #{rank_pos}: {player_name} ({hr_prob:.1f}%)")
            
            hit_hr = False
            hr_count = 0
            at_bats = 0
            game_result = ""
            
            # Check all games for this player
            for game_id in game_ids:
                try:
                    time.sleep(0.5)  # Rate limiting
                    
                    boxscore = statsapi.boxscore_data(game_id)
                    
                    # Check both teams
                    for team_side in ['home', 'away']:
                        if team_side in boxscore:
                            players = boxscore[team_side].get('players', {})
                            
                            for player_id, player_data in players.items():
                                if isinstance(player_data, dict):
                                    full_name = player_data.get('person', {}).get('fullName', '')
                                    
                                    if self._names_match(full_name, player_name):
                                        batting_stats = player_data.get('stats', {}).get('batting', {})
                                        
                                        if batting_stats:
                                            hr_count = int(batting_stats.get('homeRuns', 0))
                                            at_bats = int(batting_stats.get('atBats', 0))
                                            
                                            if hr_count > 0:
                                                hit_hr = True
                                                game_result = f"{boxscore['away']['team']['name']} {boxscore['away']['runs']} - {boxscore['home']['runs']} {boxscore['home']['team']['name']}"
                                                print(f"    ✅ HIT {hr_count} HR!")
                                            
                                            break
                            
                            if hit_hr:
                                break
                    
                    if hit_hr:
                        break
                        
                except Exception as e:
                    print(f"    ⚠️ Error checking game {game_id}: {e}")
                    continue
            
            if not hit_hr:
                print(f"    ❌ No HR")
            
            # Save result
            cursor.execute('''
                INSERT OR REPLACE INTO results 
                (prediction_id, player_name, date, hit_hr, hr_count, at_bats, game_result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (pred_id, player_name, target_date, hit_hr, hr_count, at_bats, game_result))
            
            # Update counters
            if hit_hr:
                results['total_hrs'] += 1
                if rank_pos <= 3:
                    results['top_3_hits'] += 1
                if rank_pos <= 5:
                    results['top_5_hits'] += 1
            
            results['details'].append({
                'rank': rank_pos,
                'player': player_name,
                'pitcher': pitcher_name,
                'teams': teams,
                'probability': hr_prob,
                'hit_hr': hit_hr,
                'hr_count': hr_count,
                'at_bats': at_bats
            })
        
        # Calculate hit rate
        results['hit_rate'] = (results['total_hrs'] / results['total_predictions']) * 100
        
        # Save daily summary
        cursor.execute('''
            INSERT OR REPLACE INTO daily_summary 
            (date, total_predictions, total_hrs, hit_rate, top_3_hits, top_5_hits)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            target_date,
            results['total_predictions'],
            results['total_hrs'],
            results['hit_rate'],
            results['top_3_hits'],
            results['top_5_hits']
        ))
        
        self.conn.commit()
        
        print(f"\n📊 Results Summary:")
        print(f"   Total Predictions: {results['total_predictions']}")
        print(f"   Actual HRs: {results['total_hrs']}")
        print(f"   Hit Rate: {results['hit_rate']:.1f}%")
        print(f"   Top 3 Hits: {results['top_3_hits']}/3")
        print(f"   Top 5 Hits: {results['top_5_hits']}/5")
        
        return results
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two player names match (handles variations)"""
        if not name1 or not name2:
            return False
        
        # Simple exact match first
        if name1.lower() == name2.lower():
            return True
        
        # Check last name match (most reliable)
        name1_parts = name1.split()
        name2_parts = name2.split()
        
        if len(name1_parts) >= 2 and len(name2_parts) >= 2:
            # Compare last names and first initial
            last1, first1 = name1_parts[-1].lower(), name1_parts[0][0].lower()
            last2, first2 = name2_parts[-1].lower(), name2_parts[0][0].lower()
            
            return last1 == last2 and first1 == first2
        
        return False
    
    def display_tracking_results(self, date: str = None):
        """Display tracking results with visual indicators"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d').date()
        except:
            print(f"❌ Invalid date format: {date}")
            return
        
        cursor = self.conn.cursor()
        
        # Get results for this date
        cursor.execute('''
            SELECT p.rank_position, p.player_name, p.pitcher_name, p.teams, 
                   p.hr_probability, r.hit_hr, r.hr_count, r.at_bats
            FROM predictions p
            LEFT JOIN results r ON p.id = r.prediction_id
            WHERE p.date = ?
            ORDER BY p.rank_position
        ''', (target_date,))
        
        results = cursor.fetchall()
        
        if not results:
            print(f"❌ No tracking data found for {target_date}")
            return
        
        print("\n" + "=" * 80)
        print(f"🎯 HOME RUN PREDICTION TRACKING - {target_date}")
        print("=" * 80)
        print(f"{'Rank':<5} {'✓':<3} {'Player':<20} {'Pitcher':<18} {'Teams':<12} {'Prob%':<6} {'Result'}")
        print("-" * 80)
        
        total_checked = 0
        total_hrs = 0
        
        for rank, player, pitcher, teams, prob, hit_hr, hr_count, at_bats in results:
            if hit_hr is not None:  # Result available
                total_checked += 1
                if hit_hr:
                    total_hrs += 1
                    checkmark = "✅"
                    result = f"{hr_count} HR" if hr_count else "1 HR"
                else:
                    checkmark = "❌"
                    result = f"0/{at_bats}" if at_bats else "No HR"
            else:
                checkmark = "⏳"
                result = "Pending"
            
            print(f"{rank:<5} {checkmark:<3} {player[:19]:<20} {pitcher[:17]:<18} {teams:<12} {prob:<6.1f} {result}")
        
        # Summary
        if total_checked > 0:
            hit_rate = (total_hrs / total_checked) * 100
            print("-" * 80)
            print(f"📊 SUMMARY: {total_hrs}/{total_checked} hit HRs ({hit_rate:.1f}% success rate)")
        
        # Get overall stats
        cursor.execute('''
            SELECT COUNT(*) as total_days, 
                   AVG(hit_rate) as avg_hit_rate,
                   SUM(total_hrs) as total_hrs_hit,
                   SUM(total_predictions) as total_predictions_made
            FROM daily_summary
        ''')
        
        overall = cursor.fetchone()
        if overall and overall[0] > 0:
            print(f"📈 OVERALL: {overall[2]}/{overall[3]} HRs hit across {overall[0]} days (avg {overall[1]:.1f}%)")
    
    def generate_performance_report(self, days: int = 7):
        """Generate performance report for last N days"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT date, total_predictions, total_hrs, hit_rate, top_3_hits, top_5_hits
            FROM daily_summary
            WHERE date BETWEEN ? AND ?
            ORDER BY date DESC
        ''', (start_date, end_date))
        
        data = cursor.fetchall()
        
        if not data:
            print(f"❌ No performance data found for last {days} days")
            return
        
        print("\n" + "=" * 60)
        print(f"📊 PERFORMANCE REPORT - Last {days} Days")
        print("=" * 60)
        print(f"{'Date':<12} {'Pred':<5} {'HRs':<4} {'Rate%':<6} {'Top3':<5} {'Top5'}")
        print("-" * 60)
        
        total_preds = 0
        total_hrs = 0
        
        for date, preds, hrs, rate, top3, top5 in data:
            total_preds += preds
            total_hrs += hrs
            print(f"{date:<12} {preds:<5} {hrs:<4} {rate:<6.1f} {top3:<5} {top5}")
        
        if total_preds > 0:
            overall_rate = (total_hrs / total_preds) * 100
            print("-" * 60)
            print(f"📈 TOTALS: {total_hrs}/{total_preds} HRs hit ({overall_rate:.1f}% overall rate)")

def main():
    """Main function to run tracking system"""
    tracker = HRPredictionTracker()
    
    print("🎯 MLB Home Run Prediction Tracking System")
    print("=" * 50)
    print("1. Import predictions from CSV")
    print("2. Check actual results")
    print("3. Display tracking results") 
    print("4. Performance report")
    print("5. Auto-track (import + check)")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        csv_file = input("Enter CSV filename: ").strip()
        tracker.import_predictions_from_csv(csv_file)
    
    elif choice == "2":
        date = input("Enter date (YYYY-MM-DD) or press Enter for today: ").strip()
        if not date:
            date = None
        tracker.check_actual_results(date)
    
    elif choice == "3":
        date = input("Enter date (YYYY-MM-DD) or press Enter for today: ").strip()
        if not date:
            date = None
        tracker.display_tracking_results(date)
    
    elif choice == "4":
        days = input("Enter number of days (default 7): ").strip()
        days = int(days) if days.isdigit() else 7
        tracker.generate_performance_report(days)
    
    elif choice == "5":
        csv_file = input("Enter CSV filename: ").strip()
        if tracker.import_predictions_from_csv(csv_file):
            date = tracker._extract_date_from_filename(csv_file)
            tracker.check_actual_results(date)
            tracker.display_tracking_results(date)
    
    else:
        print("❌ Invalid option")

if __name__ == "__main__":
    main()