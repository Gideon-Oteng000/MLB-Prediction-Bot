#!/usr/bin/env python3
"""
MLB PREDICTION TRACKING SYSTEM
Track betting performance, ROI, and validate the professional system
Expected Performance: 12.8% ROI | 59.1% accuracy on HIGH/MEDIUM confidence
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import glob
import requests

class MLBPredictionTracker:
    def __init__(self):
        self.betting_history_file = 'betting_history.csv'
        self.performance_file = 'performance_summary.csv'
        
    def initialize_tracking_files(self):
        """Initialize tracking files if they don't exist"""
        
        # Betting history columns
        betting_columns = [
            'date', 'away_team', 'home_team', 'predicted_winner', 'actual_winner',
            'home_win_probability', 'confidence', 'bet_amount', 'bet_team',
            'bet_odds', 'game_result', 'bet_won', 'profit_loss', 'running_total'
        ]
        
        if not os.path.exists(self.betting_history_file):
            pd.DataFrame(columns=betting_columns).to_csv(self.betting_history_file, index=False)
            print(f"✅ Created {self.betting_history_file}")
        
        # Performance summary columns
        performance_columns = [
            'date', 'total_bets', 'wins', 'losses', 'win_rate', 'total_wagered',
            'total_profit', 'roi', 'high_conf_bets', 'high_conf_wins', 'high_conf_rate',
            'medium_conf_bets', 'medium_conf_wins', 'medium_conf_rate'
        ]
        
        if not os.path.exists(self.performance_file):
            pd.DataFrame(columns=performance_columns).to_csv(self.performance_file, index=False)
            print(f"✅ Created {self.performance_file}")
    
    def load_daily_predictions(self, date=None):
        """Load predictions from daily prediction files"""
        
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        elif isinstance(date, str):
            # Convert YYYY-MM-DD to YYYYMMDD
            if '-' in date:
                date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
        
        filename = f"daily_predictions_{date}.csv"
        
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print(f"✅ Loaded predictions from {filename}")
            return df
        else:
            print(f"❌ Prediction file {filename} not found")
            print("Available prediction files:")
            prediction_files = glob.glob("daily_predictions_*.csv")
            for file in sorted(prediction_files)[-5:]:  # Show last 5 files
                print(f"  • {file}")
            return None
    
    def get_available_prediction_dates(self):
        """Get list of available prediction dates"""
        prediction_files = glob.glob("daily_predictions_*.csv")
        dates = []
        for file in prediction_files:
            date_str = file.replace('daily_predictions_', '').replace('.csv', '')
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                dates.append(date_obj.strftime('%Y-%m-%d'))
            except:
                continue
        return sorted(dates)
    
    def input_betting_session(self):
        """Interactive session to input betting amounts and results"""
        
        print("🎯 MLB BETTING SESSION TRACKER")
        print("="*50)
        
        # Get available dates
        available_dates = self.get_available_prediction_dates()
        if not available_dates:
            print("❌ No prediction files found. Run daily_prediction_system.py first.")
            return
        
        print("Available prediction dates:")
        for i, date in enumerate(available_dates[-7:], 1):  # Show last 7 days
            print(f"  {i}. {date}")
        
        # Select date
        try:
            choice = input(f"\nSelect date (1-{len(available_dates[-7:])}) or enter YYYY-MM-DD: ").strip()
            
            if choice.isdigit():
                selected_date = available_dates[-7:][int(choice)-1]
            else:
                # Validate date format
                datetime.strptime(choice, '%Y-%m-%d')
                selected_date = choice
                
        except (ValueError, IndexError):
            print("❌ Invalid date selection")
            return
        
        # Load predictions for selected date
        predictions = self.load_daily_predictions(selected_date)
        if predictions is None:
            return
        
        # Filter for betting opportunities (HIGH and MEDIUM confidence)
        betting_games = predictions[predictions['bet_worthy'] == True].copy()
        
        if len(betting_games) == 0:
            print(f"📊 No betting opportunities found for {selected_date}")
            return
        
        print(f"\n🎯 BETTING OPPORTUNITIES FOR {selected_date}")
        print("="*60)
        
        betting_records = []
        
        for idx, game in betting_games.iterrows():
            print(f"\n{game['away_team']} @ {game['home_team']}")
            print(f"Prediction: {game['predicted_winner']} ({game['home_win_probability']:.1%})")
            print(f"Confidence: {game['confidence']}")
            print(f"Recommendation: {game['recommendation']}")
            
            # Ask if they want to bet on this game
            bet_choice = input(f"\nDid you bet on this game? (y/n/skip): ").lower()
            
            if bet_choice == 'skip':
                break
            elif bet_choice != 'y':
                continue
            
            # Get betting details
            try:
                bet_amount = float(input("Enter bet amount ($): "))
                
                # Determine which team they bet on based on prediction
                if game['home_win_probability'] > 0.5:
                    bet_team = game['home_team']
                else:
                    bet_team = game['away_team']
                
                print(f"Betting ${bet_amount:.2f} on {bet_team}")
                
                # Get actual game result
                actual_winner = self.get_game_result(game['away_team'], game['home_team'], selected_date)
                
                if actual_winner is None:
                    manual_result = input("Enter actual winner (home/away/postponed): ").lower()
                    if manual_result == 'home':
                        actual_winner = game['home_team']
                    elif manual_result == 'away':
                        actual_winner = game['away_team']
                    else:
                        print("Game postponed/cancelled - skipping")
                        continue
                
                # Calculate result
                bet_won = (bet_team == actual_winner)
                
                # Calculate profit/loss (assuming -110 odds)
                if bet_won:
                    profit_loss = bet_amount * 0.909  # Win $90.90 for every $100 bet
                else:
                    profit_loss = -bet_amount
                
                # Create betting record
                betting_record = {
                    'date': selected_date,
                    'away_team': game['away_team'],
                    'home_team': game['home_team'],
                    'predicted_winner': game['predicted_winner'],
                    'actual_winner': actual_winner,
                    'home_win_probability': game['home_win_probability'],
                    'confidence': game['confidence'],
                    'bet_amount': bet_amount,
                    'bet_team': bet_team,
                    'bet_odds': -110,
                    'game_result': f"{actual_winner} won",
                    'bet_won': bet_won,
                    'profit_loss': profit_loss,
                    'running_total': 0  # Will be calculated later
                }
                
                betting_records.append(betting_record)
                
                # Show result
                status = "✅ WON" if bet_won else "❌ LOST"
                print(f"{status} | Profit/Loss: ${profit_loss:+.2f}")
                
            except ValueError:
                print("❌ Invalid input - skipping this game")
                continue
        
        # Save betting records
        if betting_records:
            self.save_betting_records(betting_records)
            self.update_performance_summary()
            
            print(f"\n✅ Saved {len(betting_records)} betting records")
            print("Updated performance summary")
        else:
            print("\n📊 No betting records to save")
    
    def get_game_result(self, away_team, home_team, date):
        """Try to get actual game result from MLB API"""
        try:
            # Convert date format for API
            api_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
            
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={api_date}"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if 'dates' in data and len(data['dates']) > 0:
                for game in data['dates'][0]['games']:
                    game_away = game['teams']['away']['team']['name']
                    game_home = game['teams']['home']['team']['name']
                    
                    if game_away == away_team and game_home == home_team:
                        if game['status']['detailedState'] == 'Final':
                            away_score = game['teams']['away']['score']
                            home_score = game['teams']['home']['score']
                            
                            if home_score > away_score:
                                return home_team
                            else:
                                return away_team
            
            return None
            
        except Exception as e:
            print(f"Could not fetch game result automatically: {e}")
            return None
    
    def save_betting_records(self, betting_records):
        """Save betting records to history file"""
        
        # Load existing history
        if os.path.exists(self.betting_history_file):
            existing_history = pd.read_csv(self.betting_history_file)
        else:
            existing_history = pd.DataFrame()
        
        # Add new records
        new_records = pd.DataFrame(betting_records)
        combined_history = pd.concat([existing_history, new_records], ignore_index=True)
        
        # Calculate running total
        combined_history = combined_history.sort_values('date')
        combined_history['running_total'] = combined_history['profit_loss'].cumsum()
        
        # Save updated history
        combined_history.to_csv(self.betting_history_file, index=False)
    
    def update_performance_summary(self):
        """Update performance summary statistics"""
        
        if not os.path.exists(self.betting_history_file):
            return
        
        history = pd.read_csv(self.betting_history_file)
        
        if len(history) == 0:
            return
        
        # Overall statistics
        total_bets = len(history)
        wins = history['bet_won'].sum()
        losses = total_bets - wins
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        total_wagered = history['bet_amount'].sum()
        total_profit = history['profit_loss'].sum()
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        
        # By confidence level
        high_conf = history[history['confidence'] == 'HIGH']
        high_conf_bets = len(high_conf)
        high_conf_wins = high_conf['bet_won'].sum() if high_conf_bets > 0 else 0
        high_conf_rate = high_conf_wins / high_conf_bets if high_conf_bets > 0 else 0
        
        medium_conf = history[history['confidence'] == 'MEDIUM']
        medium_conf_bets = len(medium_conf)
        medium_conf_wins = medium_conf['bet_won'].sum() if medium_conf_bets > 0 else 0
        medium_conf_rate = medium_conf_wins / medium_conf_bets if medium_conf_bets > 0 else 0
        
        # Create summary record
        summary_record = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': roi,
            'high_conf_bets': high_conf_bets,
            'high_conf_wins': high_conf_wins,
            'high_conf_rate': high_conf_rate,
            'medium_conf_bets': medium_conf_bets,
            'medium_conf_wins': medium_conf_wins,
            'medium_conf_rate': medium_conf_rate
        }
        
        # Save summary (overwrite with latest)
        summary_df = pd.DataFrame([summary_record])
        summary_df.to_csv(self.performance_file, index=False)
    
    def show_performance_dashboard(self):
        """Display comprehensive performance dashboard"""
        
        print("📊 MLB BETTING PERFORMANCE DASHBOARD")
        print("="*60)
        
        if not os.path.exists(self.betting_history_file):
            print("❌ No betting history found. Place some bets first!")
            return
        
        history = pd.read_csv(self.betting_history_file)
        
        if len(history) == 0:
            print("❌ No betting records found")
            return
        
        # Overall Performance
        total_bets = len(history)
        wins = history['bet_won'].sum()
        losses = total_bets - wins
        win_rate = wins / total_bets
        
        total_wagered = history['bet_amount'].sum()
        total_profit = history['profit_loss'].sum()
        roi = (total_profit / total_wagered * 100)
        
        print(f"📈 OVERALL PERFORMANCE")
        print(f"Total Bets: {total_bets}")
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Total Wagered: ${total_wagered:,.2f}")
        print(f"Total Profit: ${total_profit:+,.2f}")
        print(f"ROI: {roi:+.1f}%")
        
        # Performance vs Expectations
        expected_roi = 12.8
        expected_win_rate = 0.591
        
        print(f"\n🎯 VS EXPECTED PERFORMANCE")
        print(f"Expected ROI: +{expected_roi:.1f}% | Actual: {roi:+.1f}% | Difference: {roi-expected_roi:+.1f}%")
        print(f"Expected Win Rate: {expected_win_rate:.1%} | Actual: {win_rate:.1%} | Difference: {win_rate-expected_win_rate:+.1%}")
        
        # Performance by Confidence Level
        print(f"\n🎯 PERFORMANCE BY CONFIDENCE")
        
        for conf_level in ['HIGH', 'MEDIUM']:
            conf_bets = history[history['confidence'] == conf_level]
            
            if len(conf_bets) > 0:
                conf_wins = conf_bets['bet_won'].sum()
                conf_rate = conf_wins / len(conf_bets)
                conf_profit = conf_bets['profit_loss'].sum()
                conf_wagered = conf_bets['bet_amount'].sum()
                conf_roi = (conf_profit / conf_wagered * 100) if conf_wagered > 0 else 0
                
                # Expected rates
                expected_rates = {'HIGH': 0.529, 'MEDIUM': 0.597}
                expected_rate = expected_rates[conf_level]
                
                print(f"  {conf_level}: {conf_wins}/{len(conf_bets)} ({conf_rate:.1%}) | ROI: {conf_roi:+.1f}% | Expected: {expected_rate:.1%}")
            else:
                print(f"  {conf_level}: No bets placed")
        
        # Recent Performance (last 10 bets)
        if total_bets >= 10:
            recent_bets = history.tail(10)
            recent_wins = recent_bets['bet_won'].sum()
            recent_rate = recent_wins / len(recent_bets)
            recent_profit = recent_bets['profit_loss'].sum()
            
            print(f"\n📅 RECENT PERFORMANCE (Last 10 bets)")
            print(f"Wins: {recent_wins}/10 ({recent_rate:.1%})")
            print(f"Profit: ${recent_profit:+.2f}")
        
        # Monthly Breakdown
        history['date'] = pd.to_datetime(history['date'])
        history['month'] = history['date'].dt.to_period('M')
        
        monthly_stats = history.groupby('month').agg({
            'bet_won': ['count', 'sum'],
            'profit_loss': 'sum',
            'bet_amount': 'sum'
        }).round(2)
        
        if len(monthly_stats) > 0:
            print(f"\n📅 MONTHLY BREAKDOWN")
            for month in monthly_stats.index:
                bets = monthly_stats.loc[month, ('bet_won', 'count')]
                wins = monthly_stats.loc[month, ('bet_won', 'sum')]
                profit = monthly_stats.loc[month, ('profit_loss', 'sum')]
                wagered = monthly_stats.loc[month, ('bet_amount', 'sum')]
                monthly_roi = (profit / wagered * 100) if wagered > 0 else 0
                
                print(f"  {month}: {wins}/{bets} ({wins/bets:.1%}) | ${profit:+.2f} | ROI: {monthly_roi:+.1f}%")
        
        # Betting Unit Analysis
        avg_bet = history['bet_amount'].mean()
        min_bet = history['bet_amount'].min()
        max_bet = history['bet_amount'].max()
        unit_consistency = (min_bet == max_bet)
        
        print(f"\n💰 BETTING UNIT ANALYSIS")
        if unit_consistency:
            print(f"Betting Unit: ${avg_bet:.2f} (consistent)")
            print("✅ Good bankroll management - using fixed unit size!")
        else:
            print(f"Average Bet: ${avg_bet:.2f}")
            print(f"Range: ${min_bet:.2f} - ${max_bet:.2f}")
            print("💡 Consider using consistent unit sizes for better bankroll management")
        
        # Profitability Status
        print(f"\n🏆 PROFITABILITY STATUS")
        if roi > 5:
            print("🎉 EXCELLENT! System is highly profitable!")
        elif roi > 2:
            print("📈 GOOD! System is profitable - keep it up!")
        elif roi > 0:
            print("💚 POSITIVE! System is making money!")
        elif roi > -5:
            print("⚠️ BREAK-EVEN: Close to profitability")
        else:
            print("📊 NEEDS IMPROVEMENT: Consider refining strategy")
        
        # Save current snapshot
        print(f"\n💾 Performance data saved to:")
        print(f"  • {self.betting_history_file}")
        print(f"  • {self.performance_file}")
    
    def show_betting_history(self, limit=20):
        """Show recent betting history"""
        
        if not os.path.exists(self.betting_history_file):
            print("❌ No betting history found")
            return
        
        history = pd.read_csv(self.betting_history_file)
        
        if len(history) == 0:
            print("❌ No betting records found")
            return
        
        print(f"📋 RECENT BETTING HISTORY (Last {limit} bets)")
        print("="*80)
        
        recent_history = history.tail(limit)
        
        for _, bet in recent_history.iterrows():
            status = "✅" if bet['bet_won'] else "❌"
            
            print(f"{status} {bet['date']} | {bet['away_team']} @ {bet['home_team']}")
            print(f"    Bet: ${bet['bet_amount']:.2f} on {bet['bet_team']} ({bet['confidence']})")
            print(f"    Result: {bet['actual_winner']} won | P&L: ${bet['profit_loss']:+.2f}")
            print(f"    Running Total: ${bet['running_total']:+.2f}")
            print()
    
    def export_performance_report(self):
        """Export detailed performance report"""
        
        if not os.path.exists(self.betting_history_file):
            print("❌ No betting history found")
            return
        
        history = pd.read_csv(self.betting_history_file)
        
        if len(history) == 0:
            print("❌ No betting records found")
            return
        
        # Create detailed report
        report_filename = f"performance_report_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Add additional analysis columns
        history['cumulative_roi'] = (history['profit_loss'].cumsum() / history['bet_amount'].cumsum() * 100)
        history['prediction_correct'] = history['predicted_winner'] == history['actual_winner']
        
        # Save detailed report
        history.to_csv(report_filename, index=False)
        
        print(f"✅ Detailed performance report exported to: {report_filename}")
        
        # Summary statistics
        summary_stats = {
            'total_bets': len(history),
            'total_wins': history['bet_won'].sum(),
            'win_rate': history['bet_won'].mean(),
            'total_wagered': history['bet_amount'].sum(),
            'total_profit': history['profit_loss'].sum(),
            'roi': (history['profit_loss'].sum() / history['bet_amount'].sum() * 100),
            'avg_bet_amount': history['bet_amount'].mean(),
            'prediction_accuracy': history['prediction_correct'].mean(),
            'high_conf_accuracy': history[history['confidence'] == 'HIGH']['prediction_correct'].mean() if len(history[history['confidence'] == 'HIGH']) > 0 else 0,
            'medium_conf_accuracy': history[history['confidence'] == 'MEDIUM']['prediction_correct'].mean() if len(history[history['confidence'] == 'MEDIUM']) > 0 else 0
        }
        
        summary_filename = f"summary_stats_{datetime.now().strftime('%Y%m%d')}.csv"
        pd.DataFrame([summary_stats]).to_csv(summary_filename, index=False)
        
        print(f"✅ Summary statistics exported to: {summary_filename}")

def main():
    """Main menu for prediction tracking"""
    
    tracker = MLBPredictionTracker()
    tracker.initialize_tracking_files()
    
    while True:
        print(f"\n🏆 MLB PREDICTION TRACKING SYSTEM")
        print("="*50)
        print("1. Input betting session (add new bets)")
        print("2. Show performance dashboard")
        print("3. Show betting history")
        print("4. Export performance report")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            tracker.input_betting_session()
        elif choice == '2':
            tracker.show_performance_dashboard()
        elif choice == '3':
            limit = input("Number of recent bets to show (default 20): ").strip()
            limit = int(limit) if limit.isdigit() else 20
            tracker.show_betting_history(limit)
        elif choice == '4':
            tracker.export_performance_report()
        elif choice == '5':
            print("👋 Happy betting! Track those profits!")
            break
        else:
            print("❌ Invalid option. Please try again.")

if __name__ == "__main__":
    main()