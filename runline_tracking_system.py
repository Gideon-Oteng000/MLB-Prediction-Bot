#!/usr/bin/env python3
"""
PROFESSIONAL MLB RUNLINE TRACKING SYSTEM
Track runline betting performance with fixed unit betting amounts
Tracks spread predictions (favorite -1.5 vs underdog +1.5)
Expected Performance: 55-60% accuracy on HIGH/MEDIUM confidence
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import glob
import requests

class RunLineTracker:
    def __init__(self):
        self.betting_history_file = 'runline_betting_history.csv'
        self.performance_file = 'runline_performance_summary.csv'
        
    def initialize_tracking_files(self):
        """Initialize runline tracking files if they don't exist"""
        
        # Runline betting history columns
        betting_columns = [
            'date', 'favorite_team', 'underdog_team', 'predicted_winner', 'actual_result',
            'favorite_cover_probability', 'confidence', 'bet_amount', 'bet_team',
            'bet_type', 'favorite_score', 'underdog_score', 'margin', 'favorite_covered',
            'bet_won', 'profit_loss', 'running_total'
        ]
        
        if not os.path.exists(self.betting_history_file):
            pd.DataFrame(columns=betting_columns).to_csv(self.betting_history_file, index=False)
            print(f"✅ Created {self.betting_history_file}")
        
        # Runline performance summary columns
        performance_columns = [
            'date', 'total_bets', 'wins', 'losses', 'cover_rate', 'total_wagered',
            'total_profit', 'roi', 'high_conf_bets', 'high_conf_wins', 'high_conf_rate',
            'medium_conf_bets', 'medium_conf_wins', 'medium_conf_rate', 'favorite_bets',
            'favorite_wins', 'favorite_rate', 'underdog_bets', 'underdog_wins', 'underdog_rate'
        ]
        
        if not os.path.exists(self.performance_file):
            pd.DataFrame(columns=performance_columns).to_csv(self.performance_file, index=False)
            print(f"✅ Created {self.performance_file}")
    
    def load_runline_predictions(self, date=None):
        """Load predictions from runline prediction files"""
        
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        elif isinstance(date, str):
            # Convert YYYY-MM-DD to YYYYMMDD
            if '-' in date:
                date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
        
        # Try different runline prediction file patterns
        possible_files = [
            f"professional_runline_{date}.csv",
            f"runline_predictions_{date}.csv",
            f"fixed_runline_predictions_{date}.csv"
        ]
        
        for filename in possible_files:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                print(f"✅ Loaded runline predictions from {filename}")
                return df
        
        print(f"❌ No runline prediction files found for {date}")
        print("Available runline files:")
        runline_files = glob.glob("*runline*.csv")
        for file in sorted(runline_files)[-5:]:  # Show last 5 files
            print(f"  • {file}")
        return None
    
    def get_available_runline_dates(self):
        """Get list of available runline prediction dates"""
        runline_files = glob.glob("*runline*.csv")
        dates = []
        
        for file in runline_files:
            # Extract date from various file patterns
            if 'professional_runline_' in file:
                date_str = file.replace('professional_runline_', '').replace('.csv', '')
            elif 'runline_predictions_' in file:
                date_str = file.replace('runline_predictions_', '').replace('.csv', '')
            elif 'fixed_runline_predictions_' in file:
                date_str = file.replace('fixed_runline_predictions_', '').replace('.csv', '')
            else:
                continue
                
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                dates.append(date_obj.strftime('%Y-%m-%d'))
            except:
                continue
        
        return sorted(list(set(dates)))  # Remove duplicates
    
    def input_runline_betting_session(self):
        """Interactive session to input runline betting results with fixed unit size"""
        
        print("🎯 RUNLINE BETTING SESSION TRACKER")
        print("="*50)
        
        # Get betting unit amount
        try:
            betting_unit = float(input("Enter your runline betting unit amount ($): "))
            print(f"✅ Using ${betting_unit:.2f} per runline bet")
        except ValueError:
            print("❌ Invalid amount entered")
            return
        
        # Get available dates
        available_dates = self.get_available_runline_dates()
        if not available_dates:
            print("❌ No runline prediction files found. Run professional_runline_system.py first.")
            return
        
        print("\nAvailable runline prediction dates:")
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
        predictions = self.load_runline_predictions(selected_date)
        if predictions is None:
            return
        
        # Filter for betting opportunities (HIGH and MEDIUM confidence)
        betting_games = predictions[predictions['bet_worthy'] == True].copy() if 'bet_worthy' in predictions.columns else predictions[predictions['confidence'].isin(['HIGH', 'MEDIUM'])]
        
        if len(betting_games) == 0:
            print(f"📊 No runline betting opportunities found for {selected_date}")
            return
        
        print(f"\n🎯 RUNLINE OPPORTUNITIES FOR {selected_date}")
        print("="*60)
        print(f"Betting unit: ${betting_unit:.2f} per game")
        print()
        
        betting_records = []
        total_bets_placed = 0
        
        for idx, game in betting_games.iterrows():
            favorite = game.get('favorite_team', 'Unknown')
            underdog = game.get('underdog_team', 'Unknown')
            prediction = game.get('prediction', f"{favorite} -1.5")
            confidence = game.get('confidence', 'MEDIUM')
            cover_prob = game.get('favorite_cover_probability', 0.5)
            
            print(f"{underdog} @ {favorite}")
            print(f"Runline: {favorite} -1.5 vs {underdog} +1.5")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence}")
            print(f"Favorite cover probability: {cover_prob:.1%}")
            
            # Ask if they want to bet on this game
            bet_choice = input(f"Did you bet ${betting_unit:.2f} on this runline? (y/n/skip): ").lower()
            
            if bet_choice == 'skip':
                break
            elif bet_choice != 'y':
                print("Skipping this runline\n")
                continue
            
            total_bets_placed += 1
            
            # Determine which side they bet based on prediction
            if "-1.5" in prediction and favorite in prediction:
                bet_team = favorite
                bet_type = "Favorite -1.5"
            else:
                bet_team = underdog  
                bet_type = "Underdog +1.5"
            
            print(f"✅ Betting ${betting_unit:.2f} on {bet_team} ({bet_type})")
            
            # Get actual game result
            actual_result = self.get_runline_result(favorite, underdog, selected_date)
            
            if actual_result is None:
                print("Enter game result manually:")
                try:
                    fav_score = int(input(f"  {favorite} score: "))
                    und_score = int(input(f"  {underdog} score: "))
                    
                    margin = fav_score - und_score
                    favorite_covered = margin >= 2
                    
                    actual_result = {
                        'favorite_score': fav_score,
                        'underdog_score': und_score,
                        'margin': margin,
                        'favorite_covered': favorite_covered
                    }
                    
                except ValueError:
                    print("Invalid scores entered - skipping\n")
                    continue
            
            # Calculate bet result
            if bet_type == "Favorite -1.5":
                bet_won = actual_result['favorite_covered']
            else:  # Underdog +1.5
                bet_won = not actual_result['favorite_covered']
            
            # Calculate profit/loss (assuming -110 odds)
            if bet_won:
                profit_loss = betting_unit * 0.909  # Win $90.90 for every $100 bet
            else:
                profit_loss = -betting_unit
            
            # Create betting record
            betting_record = {
                'date': selected_date,
                'favorite_team': favorite,
                'underdog_team': underdog,
                'predicted_winner': bet_team,
                'actual_result': f"Favorite by {actual_result['margin']}" if actual_result['margin'] > 0 else f"Underdog by {abs(actual_result['margin'])}",
                'favorite_cover_probability': cover_prob,
                'confidence': confidence,
                'bet_amount': betting_unit,
                'bet_team': bet_team,
                'bet_type': bet_type,
                'favorite_score': actual_result['favorite_score'],
                'underdog_score': actual_result['underdog_score'],
                'margin': actual_result['margin'],
                'favorite_covered': actual_result['favorite_covered'],
                'bet_won': bet_won,
                'profit_loss': profit_loss,
                'running_total': 0  # Will be calculated later
            }
            
            betting_records.append(betting_record)
            
            # Show result
            status = "✅ WON" if bet_won else "❌ LOST"
            margin_text = f"Favorite won by {actual_result['margin']}" if actual_result['margin'] > 0 else f"Underdog won by {abs(actual_result['margin'])}"
            cover_text = "COVERED" if actual_result['favorite_covered'] else "DID NOT COVER"
            
            print(f"{status} | {margin_text} | Favorite {cover_text} | P&L: ${profit_loss:+.2f}\n")
        
        # Session summary
        if total_bets_placed > 0:
            total_wagered = total_bets_placed * betting_unit
            total_profit = sum([record['profit_loss'] for record in betting_records])
            session_roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
            wins = sum([record['bet_won'] for record in betting_records])
            
            print(f"📊 RUNLINE SESSION SUMMARY:")
            print(f"Bets placed: {total_bets_placed}")
            print(f"Wins: {wins} | Losses: {total_bets_placed - wins}")
            print(f"Cover rate: {wins/total_bets_placed:.1%}")
            print(f"Total wagered: ${total_wagered:.2f}")
            print(f"Total profit/loss: ${total_profit:+.2f}")
            print(f"Session ROI: {session_roi:+.1f}%")
        
        # Save betting records
        if betting_records:
            self.save_runline_records(betting_records)
            self.update_runline_performance()
            
            print(f"\n✅ Saved {len(betting_records)} runline betting records")
            print("Updated runline performance summary")
        else:
            print("\n📊 No runline betting records to save")
    
    def get_runline_result(self, favorite_team, underdog_team, date):
        """Try to get actual runline result from MLB API"""
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
                    
                    # Match the favorite/underdog to home/away
                    if (game_home == favorite_team and game_away == underdog_team) or \
                       (game_home == underdog_team and game_away == favorite_team):
                        
                        if game['status']['detailedState'] == 'Final':
                            away_score = game['teams']['away']['score']
                            home_score = game['teams']['home']['score']
                            
                            # Determine which team is favorite and calculate margin
                            if game_home == favorite_team:
                                fav_score = home_score
                                und_score = away_score
                            else:
                                fav_score = away_score
                                und_score = home_score
                            
                            margin = fav_score - und_score
                            favorite_covered = margin >= 2
                            
                            return {
                                'favorite_score': fav_score,
                                'underdog_score': und_score,
                                'margin': margin,
                                'favorite_covered': favorite_covered
                            }
            
            return None
            
        except Exception as e:
            print(f"Could not fetch runline result automatically: {e}")
            return None
    
    def save_runline_records(self, betting_records):
        """Save runline betting records to history file"""
        
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
    
    def update_runline_performance(self):
        """Update runline performance summary statistics"""
        
        if not os.path.exists(self.betting_history_file):
            return
        
        history = pd.read_csv(self.betting_history_file)
        
        if len(history) == 0:
            return
        
        # Overall statistics
        total_bets = len(history)
        wins = history['bet_won'].sum()
        losses = total_bets - wins
        cover_rate = wins / total_bets if total_bets > 0 else 0
        
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
        
        # By bet type (favorite vs underdog)
        favorite_bets = history[history['bet_type'].str.contains('Favorite', na=False)]
        favorite_bet_count = len(favorite_bets)
        favorite_wins = favorite_bets['bet_won'].sum() if favorite_bet_count > 0 else 0
        favorite_rate = favorite_wins / favorite_bet_count if favorite_bet_count > 0 else 0
        
        underdog_bets = history[history['bet_type'].str.contains('Underdog', na=False)]
        underdog_bet_count = len(underdog_bets)
        underdog_wins = underdog_bets['bet_won'].sum() if underdog_bet_count > 0 else 0
        underdog_rate = underdog_wins / underdog_bet_count if underdog_bet_count > 0 else 0
        
        # Create summary record
        summary_record = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'cover_rate': cover_rate,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': roi,
            'high_conf_bets': high_conf_bets,
            'high_conf_wins': high_conf_wins,
            'high_conf_rate': high_conf_rate,
            'medium_conf_bets': medium_conf_bets,
            'medium_conf_wins': medium_conf_wins,
            'medium_conf_rate': medium_conf_rate,
            'favorite_bets': favorite_bet_count,
            'favorite_wins': favorite_wins,
            'favorite_rate': favorite_rate,
            'underdog_bets': underdog_bet_count,
            'underdog_wins': underdog_wins,
            'underdog_rate': underdog_rate
        }
        
        # Save summary (overwrite with latest)
        summary_df = pd.DataFrame([summary_record])
        summary_df.to_csv(self.performance_file, index=False)
    
    def show_runline_dashboard(self):
        """Display comprehensive runline performance dashboard"""
        
        print("📊 RUNLINE BETTING PERFORMANCE DASHBOARD")
        print("="*60)
        
        if not os.path.exists(self.betting_history_file):
            print("❌ No runline betting history found. Place some runline bets first!")
            return
        
        history = pd.read_csv(self.betting_history_file)
        
        if len(history) == 0:
            print("❌ No runline betting records found")
            return
        
        # Overall Performance
        total_bets = len(history)
        wins = history['bet_won'].sum()
        losses = total_bets - wins
        cover_rate = wins / total_bets
        
        total_wagered = history['bet_amount'].sum()
        total_profit = history['profit_loss'].sum()
        roi = (total_profit / total_wagered * 100)
        
        print(f"📈 OVERALL RUNLINE PERFORMANCE")
        print(f"Total Bets: {total_bets}")
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Cover Rate: {cover_rate:.1%}")
        print(f"Total Wagered: ${total_wagered:,.2f}")
        print(f"Total Profit: ${total_profit:+,.2f}")
        print(f"ROI: {roi:+.1f}%")
        
        # Performance vs Expectations
        expected_roi = 8.0  # Expected for runline betting
        expected_cover_rate = 0.56
        
        print(f"\n🎯 VS EXPECTED PERFORMANCE")
        print(f"Expected ROI: +{expected_roi:.1f}% | Actual: {roi:+.1f}% | Difference: {roi-expected_roi:+.1f}%")
        print(f"Expected Cover Rate: {expected_cover_rate:.1%} | Actual: {cover_rate:.1%} | Difference: {cover_rate-expected_cover_rate:+.1%}")
        
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
                
                # Expected rates for runline
                expected_rates = {'HIGH': 0.60, 'MEDIUM': 0.55}
                expected_rate = expected_rates[conf_level]
                
                print(f"  {conf_level}: {conf_wins}/{len(conf_bets)} ({conf_rate:.1%}) | ROI: {conf_roi:+.1f}% | Expected: {expected_rate:.1%}")
            else:
                print(f"  {conf_level}: No bets placed")
        
        # Performance by Bet Type (Favorite vs Underdog)
        print(f"\n🎯 FAVORITE VS UNDERDOG PERFORMANCE")
        
        favorite_bets = history[history['bet_type'].str.contains('Favorite', na=False)]
        underdog_bets = history[history['bet_type'].str.contains('Underdog', na=False)]
        
        if len(favorite_bets) > 0:
            fav_wins = favorite_bets['bet_won'].sum()
            fav_rate = fav_wins / len(favorite_bets)
            fav_profit = favorite_bets['profit_loss'].sum()
            print(f"  Favorite -1.5: {fav_wins}/{len(favorite_bets)} ({fav_rate:.1%}) | Profit: ${fav_profit:+.2f}")
        
        if len(underdog_bets) > 0:
            und_wins = underdog_bets['bet_won'].sum()
            und_rate = und_wins / len(underdog_bets)
            und_profit = underdog_bets['profit_loss'].sum()
            print(f"  Underdog +1.5: {und_wins}/{len(underdog_bets)} ({und_rate:.1%}) | Profit: ${und_profit:+.2f}")
        
        # Margin Analysis
        print(f"\n📊 MARGIN ANALYSIS")
        margins = history['margin'].values
        
        blowout_wins = len([m for m in margins if m >= 3])  # 3+ run margins
        close_games = len([m for m in margins if abs(m) <= 1])  # 1-run games
        
        print(f"  Blowout games (3+ runs): {blowout_wins} ({blowout_wins/total_bets:.1%})")
        print(f"  Close games (1 run): {close_games} ({close_games/total_bets:.1%})")
        print(f"  Average margin: {np.mean(margins):+.1f} runs")
        
        # Recent Performance (last 10 bets)
        if total_bets >= 10:
            recent_bets = history.tail(10)
            recent_wins = recent_bets['bet_won'].sum()
            recent_rate = recent_wins / len(recent_bets)
            recent_profit = recent_bets['profit_loss'].sum()
            
            print(f"\n📅 RECENT PERFORMANCE (Last 10 bets)")
            print(f"Covers: {recent_wins}/10 ({recent_rate:.1%})")
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
            print(f"\n📅 MONTHLY RUNLINE BREAKDOWN")
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
            print(f"Runline Betting Unit: ${avg_bet:.2f} (consistent)")
            print("✅ Good bankroll management - using fixed runline unit!")
        else:
            print(f"Average Runline Bet: ${avg_bet:.2f}")
            print(f"Range: ${min_bet:.2f} - ${max_bet:.2f}")
            print("💡 Consider using consistent runline unit sizes")
        
        # Profitability Status
        print(f"\n🏆 RUNLINE PROFITABILITY STATUS")
        if roi > 5:
            print("🎉 EXCELLENT! Runline system is highly profitable!")
        elif roi > 2:
            print("📈 GOOD! Runline system is profitable - keep it up!")
        elif roi > 0:
            print("💚 POSITIVE! Runline system is making money!")
        elif roi > -5:
            print("⚠️ BREAK-EVEN: Close to runline profitability")
        else:
            print("📊 NEEDS IMPROVEMENT: Consider refining runline strategy")
        
        # Save current snapshot
        print(f"\n💾 Runline performance data saved to:")
        print(f"  • {self.betting_history_file}")
        print(f"  • {self.performance_file}")
    
    def show_runline_history(self, limit=20):
        """Show recent runline betting history"""
        
        if not os.path.exists(self.betting_history_file):
            print("❌ No runline betting history found")
            return
        
        history = pd.read_csv(self.betting_history_file)
        
        if len(history) == 0:
            print("❌ No runline betting records found")
            return
        
        print(f"📋 RECENT RUNLINE BETTING HISTORY (Last {limit} bets)")
        print("="*80)
        
        recent_history = history.tail(limit)
        
        for _, bet in recent_history.iterrows():
            status = "✅" if bet['bet_won'] else "❌"
            
            print(f"{status} {bet['date']} | {bet['underdog_team']} @ {bet['favorite_team']}")
            print(f"    Bet: ${bet['bet_amount']:.2f} on {bet['bet_team']} ({bet['bet_type']}) - {bet['confidence']}")
            print(f"    Result: {bet['actual_result']} | Favorite {'COVERED' if bet['favorite_covered'] else 'FAILED'}")
            print(f"    P&L: ${bet['profit_loss']:+.2f} | Running Total: ${bet['running_total']:+.2f}")
            print()
    
    def export_runline_report(self):
        """Export detailed runline performance report"""
        
        if not os.path.exists(self.betting_history_file):
            print("❌ No runline betting history found")
            return
        
        history = pd.read_csv(self.betting_history_file)
        
        if len(history) == 0:
            print("❌ No runline betting records found")
            return
        
        # Create detailed report
        report_filename = f"runline_performance_report_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Add additional analysis columns
        history['cumulative_roi'] = (history['profit_loss'].cumsum() / history['bet_amount'].cumsum() * 100)
        history['prediction_correct'] = history['bet_won']  # For runline, prediction correct = bet won
        history['margin_category'] = history['margin'].apply(lambda x: 'Blowout' if abs(x) >= 3 else 'Close' if abs(x) <= 1 else 'Moderate')
        
        # Save detailed report
        history.to_csv(report_filename, index=False)
        
        print(f"✅ Detailed runline performance report exported to: {report_filename}")
        
        # Summary statistics
        summary_stats = {
            'total_runline_bets': len(history),
            'total_covers': history['bet_won'].sum(),
            'cover_rate': history['bet_won'].mean(),
            'total_wagered': history['bet_amount'].sum(),
            'total_profit': history['profit_loss'].sum(),
            'roi': (history['profit_loss'].sum() / history['bet_amount'].sum() * 100),
            'avg_bet_amount': history['bet_amount'].mean(),
            'favorite_bet_rate': len(history[history['bet_type'].str.contains('Favorite', na=False)]) / len(history),
            'underdog_bet_rate': len(history[history['bet_type'].str.contains('Underdog', na=False)]) / len(history),
            'high_conf_cover_rate': history[history['confidence'] == 'HIGH']['bet_won'].mean() if len(history[history['confidence'] == 'HIGH']) > 0 else 0,
            'medium_conf_cover_rate': history[history['confidence'] == 'MEDIUM']['bet_won'].mean() if len(history[history['confidence'] == 'MEDIUM']) > 0 else 0,
            'avg_margin': history['margin'].mean(),
            'blowout_rate': len(history[history['margin_category'] == 'Blowout']) / len(history),
            'close_game_rate': len(history[history['margin_category'] == 'Close']) / len(history)
        }
        
        summary_filename = f"runline_summary_stats_{datetime.now().strftime('%Y%m%d')}.csv"
        pd.DataFrame([summary_stats]).to_csv(summary_filename, index=False)
        
        print(f"✅ Runline summary statistics exported to: {summary_filename}")

def main():
    """Main menu for runline prediction tracking"""
    
    tracker = RunLineTracker()
    tracker.initialize_tracking_files()
    
    while True:
        print(f"\n🏆 MLB RUNLINE TRACKING SYSTEM")
        print("="*50)
        print("💡 Track your runline spread betting performance!")
        print()
        print("1. Input runline betting session (set unit amount, track bets)")
        print("2. Show runline performance dashboard")
        print("3. Show runline betting history")
        print("4. Export runline performance report")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            print("\n📝 How runline tracking works:")
            print("• Set your runline betting unit (e.g., $50)")
            print("• System shows HIGH/MEDIUM confidence runline games")
            print("• Track favorite -1.5 vs underdog +1.5 results")
            print("• Just say yes/no for each bet - amount stays the same!")
            print()
            tracker.input_runline_betting_session()
        elif choice == '2':
            tracker.show_runline_dashboard()
        elif choice == '3':
            limit = input("Number of recent runline bets to show (default 20): ").strip()
            limit = int(limit) if limit.isdigit() else 20
            tracker.show_runline_history(limit)
        elif choice == '4':
            tracker.export_runline_report()
        elif choice == '5':
            print("👋 Happy runline betting! Cover those spreads!")
            break
        else:
            print("❌ Invalid option. Please try again.")

if __name__ == "__main__":
    main()