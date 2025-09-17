#!/usr/bin/env python3
"""
AUTOMATED MLB BETTING TRACKER
Automatically tracks betting performance from fixed_daily_prediction_system.py
Uses The Odds API for real betting odds and calculates actual P&L
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
import glob
import time
import warnings
warnings.filterwarnings('ignore')

class AutomatedMLBBettingTracker:
    def __init__(self, odds_api_key):
        self.odds_api_key = odds_api_key
        self.odds_base_url = "https://api.the-odds-api.com/v4"
        
        # File names
        self.betting_log_file = 'automated_betting_log.csv'
        self.daily_summary_file = 'daily_betting_summary.csv'
        self.weekly_summary_file = 'weekly_betting_summary.csv'
        self.monthly_summary_file = 'monthly_betting_summary.csv'
        self.processed_files_log = 'processed_prediction_files.txt'
        
        # Default assumptions
        self.default_odds = -110  # Standard American odds when unavailable
        self.max_daily_loss_warning = 500  # Warning threshold
        self.max_monthly_loss_warning = 2000  # Warning threshold
        
        # Team name mapping for API consistency
        self.team_name_mapping = {
            'Arizona Diamondbacks': 'Arizona Diamondbacks',
            'Atlanta Braves': 'Atlanta Braves',
            'Baltimore Orioles': 'Baltimore Orioles',
            'Boston Red Sox': 'Boston Red Sox',
            'Chicago Cubs': 'Chicago Cubs',
            'Chicago White Sox': 'Chicago White Sox',
            'Cincinnati Reds': 'Cincinnati Reds',
            'Cleveland Guardians': 'Cleveland Guardians',
            'Colorado Rockies': 'Colorado Rockies',
            'Detroit Tigers': 'Detroit Tigers',
            'Houston Astros': 'Houston Astros',
            'Kansas City Royals': 'Kansas City Royals',
            'Los Angeles Angels': 'Los Angeles Angels',
            'Los Angeles Dodgers': 'Los Angeles Dodgers',
            'Miami Marlins': 'Miami Marlins',
            'Milwaukee Brewers': 'Milwaukee Brewers',
            'Minnesota Twins': 'Minnesota Twins',
            'New York Mets': 'New York Mets',
            'New York Yankees': 'New York Yankees',
            'Oakland Athletics': 'Oakland Athletics',
            'Philadelphia Phillies': 'Philadelphia Phillies',
            'Pittsburgh Pirates': 'Pittsburgh Pirates',
            'San Diego Padres': 'San Diego Padres',
            'San Francisco Giants': 'San Francisco Giants',
            'Seattle Mariners': 'Seattle Mariners',
            'St. Louis Cardinals': 'St. Louis Cardinals',
            'Tampa Bay Rays': 'Tampa Bay Rays',
            'Texas Rangers': 'Texas Rangers',
            'Toronto Blue Jays': 'Toronto Blue Jays',
            'Washington Nationals': 'Washington Nationals'
        }
    
    def initialize_tracking_files(self):
        """Initialize all tracking files if they don't exist"""
        
        # Betting log columns
        betting_log_columns = [
            'date', 'game_id', 'away_team', 'home_team', 'predicted_winner',
            'actual_winner', 'confidence', 'bet_amount', 'bet_team',
            'bet_odds', 'odds_source', 'game_status', 'bet_result',
            'profit_loss', 'running_total', 'processed_timestamp'
        ]
        
        # Daily summary columns  
        daily_summary_columns = [
            'date', 'total_bets', 'bet_amount_per_game', 'total_wagered',
            'games_won', 'games_lost', 'games_postponed', 'win_rate',
            'daily_profit_loss', 'running_total', 'high_conf_bets',
            'high_conf_wins', 'medium_conf_bets', 'medium_conf_wins'
        ]
        
        # Weekly summary columns
        weekly_summary_columns = [
            'week_start', 'week_end', 'days_active', 'total_bets', 
            'total_wagered', 'total_wins', 'total_losses', 'win_rate',
            'weekly_profit_loss', 'avg_daily_pnl', 'best_day', 'worst_day'
        ]
        
        # Monthly summary columns
        monthly_summary_columns = [
            'month', 'days_active', 'total_bets', 'total_wagered',
            'total_wins', 'total_losses', 'win_rate', 'monthly_profit_loss',
            'avg_daily_pnl', 'best_day', 'worst_day', 'roi_percent'
        ]
        
        # Initialize files
        for filename, columns in [
            (self.betting_log_file, betting_log_columns),
            (self.daily_summary_file, daily_summary_columns),
            (self.weekly_summary_file, weekly_summary_columns),
            (self.monthly_summary_file, monthly_summary_columns)
        ]:
            if not os.path.exists(filename):
                pd.DataFrame(columns=columns).to_csv(filename, index=False)
                print(f"Created {filename}")
        
        # Initialize processed files log
        if not os.path.exists(self.processed_files_log):
            with open(self.processed_files_log, 'w') as f:
                f.write("# Processed prediction files log\n")
            print(f"Created {self.processed_files_log}")
    
    def get_processed_files(self):
        """Get list of already processed prediction files"""
        if not os.path.exists(self.processed_files_log):
            return set()
        
        processed = set()
        with open(self.processed_files_log, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    processed.add(line.strip())
        return processed
    
    def mark_file_as_processed(self, filename):
        """Mark a prediction file as processed"""
        with open(self.processed_files_log, 'a') as f:
            f.write(f"{filename}\n")
    
    def get_unprocessed_prediction_files(self):
        """Find new prediction files to process"""
        
        # Find all fixed prediction files
        prediction_files = glob.glob("fixed_daily_predictions_*.csv")
        processed_files = self.get_processed_files()
        
        # Filter to unprocessed files only
        unprocessed = []
        for file in sorted(prediction_files):
            if file not in processed_files:
                # Check if file is from today or future (only process forward)
                try:
                    date_str = file.replace('fixed_daily_predictions_', '').replace('.csv', '')
                    file_date = datetime.strptime(date_str, '%Y%m%d').date()
                    today = datetime.now().date()
                    
                    if file_date >= today - timedelta(days=1):  # Allow 1 day back for completion
                        unprocessed.append(file)
                except ValueError:
                    continue
        
        return unprocessed
    
    def fetch_mlb_odds(self, date_str):
        """Fetch MLB odds from The Odds API for specific date"""
        
        try:
            # Convert date format for API
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # The Odds API endpoint for MLB
            url = f"{self.odds_base_url}/sports/baseball_mlb/odds"
            
            params = {
                'api_key': self.odds_api_key,
                'regions': 'us',
                'markets': 'h2h',  # Head to head (moneyline)
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            print(f"Fetching odds for {date_str}...")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                odds_data = response.json()
                print(f"Retrieved odds for {len(odds_data)} games")
                return odds_data
            else:
                print(f"Odds API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Error fetching odds: {e}")
            return []
    
    def match_game_to_odds(self, away_team, home_team, odds_data):
        """Match prediction game to odds data"""
        
        # Normalize team names
        away_normalized = self.team_name_mapping.get(away_team, away_team)
        home_normalized = self.team_name_mapping.get(home_team, home_team)
        
        for game in odds_data:
            try:
                # The Odds API format
                if 'home_team' in game and 'away_team' in game:
                    odds_home = game['home_team']
                    odds_away = game['away_team']
                elif 'teams' in game and len(game['teams']) == 2:
                    # Alternative format
                    odds_away = game['teams'][0]
                    odds_home = game['teams'][1]
                else:
                    continue
                
                # Check for match (fuzzy matching for team names)
                if (self.teams_match(away_normalized, odds_away) and 
                    self.teams_match(home_normalized, odds_home)):
                    
                    # Extract bookmaker odds (use first available)
                    if game.get('bookmakers') and len(game['bookmakers']) > 0:
                        bookmaker = game['bookmakers'][0]
                        markets = bookmaker.get('markets', [])
                        
                        for market in markets:
                            if market.get('key') == 'h2h':
                                outcomes = market.get('outcomes', [])
                                
                                odds_dict = {}
                                for outcome in outcomes:
                                    team_name = outcome.get('name', '')
                                    odds_value = outcome.get('price', self.default_odds)
                                    odds_dict[team_name] = odds_value
                                
                                return {
                                    'bookmaker': bookmaker.get('title', 'Unknown'),
                                    'odds': odds_dict,
                                    'game_data': game
                                }
                
            except Exception as e:
                print(f"Error processing odds for game: {e}")
                continue
        
        return None
    
    def teams_match(self, team1, team2):
        """Check if two team names match (fuzzy matching)"""
        # Simple fuzzy matching - can be improved
        team1_clean = team1.lower().replace('.', '').replace('st ', 'st. ')
        team2_clean = team2.lower().replace('.', '').replace('st ', 'st. ')
        
        return (team1_clean == team2_clean or 
                team1_clean in team2_clean or 
                team2_clean in team1_clean)
    
    def get_game_result(self, away_team, home_team, date_str):
        """Get actual game result from MLB API"""
        
        try:
            # MLB Stats API for game results
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'dates' in data and len(data['dates']) > 0:
                for game in data['dates'][0]['games']:
                    game_away = game['teams']['away']['team']['name']
                    game_home = game['teams']['home']['team']['name']
                    
                    if (self.teams_match(away_team, game_away) and 
                        self.teams_match(home_team, game_home)):
                        
                        status = game['status']['detailedState']
                        
                        if status == 'Final':
                            away_score = game['teams']['away']['score']
                            home_score = game['teams']['home']['score']
                            
                            if home_score > away_score:
                                return home_team, 'completed'
                            else:
                                return away_team, 'completed'
                        
                        elif status in ['Postponed', 'Cancelled', 'Suspended']:
                            return None, 'postponed'
                        else:
                            return None, 'pending'
            
            return None, 'not_found'
            
        except Exception as e:
            print(f"Error getting game result: {e}")
            return None, 'error'
    
    def calculate_payout(self, bet_amount, odds, won):
        """Calculate profit/loss based on American odds"""
        
        if not won:
            return -bet_amount
        
        if odds > 0:
            # Positive odds: profit = bet_amount * (odds / 100)
            profit = bet_amount * (odds / 100.0)
        else:
            # Negative odds: profit = bet_amount / (abs(odds) / 100)
            profit = bet_amount / (abs(odds) / 100.0)
        
        return profit
    
    def process_daily_predictions(self, bet_amount_per_game):
        """Process new prediction files and calculate betting results"""
        
        print(f"Processing daily predictions with ${bet_amount_per_game} per HIGH/MEDIUM confidence game...")
        
        unprocessed_files = self.get_unprocessed_prediction_files()
        
        if not unprocessed_files:
            print("No new prediction files to process")
            return
        
        processed_count = 0
        
        for prediction_file in unprocessed_files:
            try:
                print(f"\nProcessing {prediction_file}...")
                
                # Load predictions
                predictions = pd.read_csv(prediction_file)
                
                # Extract date from filename
                date_str = prediction_file.replace('fixed_daily_predictions_', '').replace('.csv', '')
                date_formatted = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
                
                # Filter for HIGH and MEDIUM confidence games only
                betting_games = predictions[
                    predictions['confidence'].isin(['HIGH', 'MEDIUM'])
                ].copy()
                
                if len(betting_games) == 0:
                    print(f"No HIGH/MEDIUM confidence games found in {prediction_file}")
                    self.mark_file_as_processed(prediction_file)
                    continue
                
                print(f"Found {len(betting_games)} betting opportunities")
                
                # Fetch odds for this date
                odds_data = self.fetch_mlb_odds(date_formatted)
                time.sleep(1)  # Rate limiting
                
                daily_results = []
                
                for _, game in betting_games.iterrows():
                    # Match game to odds
                    odds_match = self.match_game_to_odds(
                        game['away_team'], 
                        game['home_team'], 
                        odds_data
                    )
                    
                    # Determine bet details
                    predicted_winner = game['predicted_winner']
                    bet_odds = self.default_odds
                    odds_source = 'default'
                    
                    if odds_match:
                        # Find odds for predicted winner
                        for team_name, team_odds in odds_match['odds'].items():
                            if self.teams_match(predicted_winner, team_name):
                                bet_odds = team_odds
                                odds_source = odds_match['bookmaker']
                                break
                    
                    # Get actual game result
                    actual_winner, game_status = self.get_game_result(
                        game['away_team'], 
                        game['home_team'], 
                        date_formatted
                    )
                    
                    # Calculate bet result
                    if game_status == 'completed':
                        bet_won = (predicted_winner == actual_winner)
                        profit_loss = self.calculate_payout(bet_amount_per_game, bet_odds, bet_won)
                        bet_result = 'won' if bet_won else 'lost'
                    elif game_status == 'postponed':
                        bet_won = None
                        profit_loss = 0  # Refund
                        bet_result = 'refunded'
                    else:
                        bet_won = None
                        profit_loss = 0
                        bet_result = 'pending'
                    
                    # Create betting record
                    betting_record = {
                        'date': date_formatted,
                        'game_id': f"{game['away_team']}@{game['home_team']}_{date_str}",
                        'away_team': game['away_team'],
                        'home_team': game['home_team'],
                        'predicted_winner': predicted_winner,
                        'actual_winner': actual_winner,
                        'confidence': game['confidence'],
                        'bet_amount': bet_amount_per_game,
                        'bet_team': predicted_winner,
                        'bet_odds': bet_odds,
                        'odds_source': odds_source,
                        'game_status': game_status,
                        'bet_result': bet_result,
                        'profit_loss': profit_loss,
                        'running_total': 0,  # Will calculate later
                        'processed_timestamp': datetime.now().isoformat()
                    }
                    
                    daily_results.append(betting_record)
                    
                    print(f"  {game['away_team']} @ {game['home_team']}: "
                          f"Bet ${bet_amount_per_game} on {predicted_winner} "
                          f"({bet_odds}) -> {bet_result.upper()} "
                          f"${profit_loss:+.2f}")
                
                # Save daily results
                if daily_results:
                    self.save_betting_results(daily_results)
                    self.update_summaries()
                    processed_count += 1
                
                # Mark file as processed
                self.mark_file_as_processed(prediction_file)
                
                print(f"Processed {prediction_file} - {len(daily_results)} bets recorded")
                
            except Exception as e:
                print(f"Error processing {prediction_file}: {e}")
                continue
        
        if processed_count > 0:
            print(f"\n✅ Successfully processed {processed_count} prediction files")
            self.check_responsible_gambling_limits()
        else:
            print("No files were successfully processed")
    
    def save_betting_results(self, betting_results):
        """Save betting results to log file"""
        
        # Load existing log
        if os.path.exists(self.betting_log_file):
            existing_log = pd.read_csv(self.betting_log_file)
        else:
            existing_log = pd.DataFrame()
        
        # Add new results
        new_results = pd.DataFrame(betting_results)
        combined_log = pd.concat([existing_log, new_results], ignore_index=True)
        
        # Calculate running total
        combined_log = combined_log.sort_values(['date', 'processed_timestamp'])
        combined_log['running_total'] = combined_log['profit_loss'].cumsum()
        
        # Save updated log
        combined_log.to_csv(self.betting_log_file, index=False)
    
    def update_summaries(self):
        """Update daily, weekly, and monthly summary files"""
        
        if not os.path.exists(self.betting_log_file):
            return
        
        betting_log = pd.read_csv(self.betting_log_file)
        betting_log['date'] = pd.to_datetime(betting_log['date'])
        
        # Update daily summary
        self.update_daily_summary(betting_log)
        
        # Update weekly summary
        self.update_weekly_summary(betting_log)
        
        # Update monthly summary
        self.update_monthly_summary(betting_log)
    
    def update_daily_summary(self, betting_log):
        """Update daily summary statistics"""
        
        daily_stats = betting_log.groupby(betting_log['date'].dt.date).agg({
            'bet_amount': ['count', 'first', 'sum'],
            'profit_loss': 'sum',
            'bet_result': lambda x: (x == 'won').sum(),
            'confidence': lambda x: ((x == 'HIGH') & (betting_log.loc[x.index, 'bet_result'] == 'won')).sum()
        }).round(2)
        
        daily_summaries = []
        
        for date, stats in daily_stats.iterrows():
            total_bets = int(stats[('bet_amount', 'count')])
            bet_amount_per_game = stats[('bet_amount', 'first')]
            total_wagered = stats[('bet_amount', 'sum')]
            daily_pnl = stats[('profit_loss', 'sum')]
            games_won = int(stats[('bet_result', '<lambda>')])
            
            # Additional calculations
            day_data = betting_log[betting_log['date'].dt.date == date]
            games_lost = len(day_data[day_data['bet_result'] == 'lost'])
            games_postponed = len(day_data[day_data['bet_result'] == 'refunded'])
            
            high_conf_bets = len(day_data[day_data['confidence'] == 'HIGH'])
            high_conf_wins = len(day_data[(day_data['confidence'] == 'HIGH') & (day_data['bet_result'] == 'won')])
            medium_conf_bets = len(day_data[day_data['confidence'] == 'MEDIUM'])
            medium_conf_wins = len(day_data[(day_data['confidence'] == 'MEDIUM') & (day_data['bet_result'] == 'won')])
            
            win_rate = games_won / (games_won + games_lost) if (games_won + games_lost) > 0 else 0
            
            # Running total (last entry for this date)
            running_total = day_data['running_total'].iloc[-1] if len(day_data) > 0 else 0
            
            daily_summary = {
                'date': date.strftime('%Y-%m-%d'),
                'total_bets': total_bets,
                'bet_amount_per_game': bet_amount_per_game,
                'total_wagered': total_wagered,
                'games_won': games_won,
                'games_lost': games_lost,
                'games_postponed': games_postponed,
                'win_rate': round(win_rate, 3),
                'daily_profit_loss': daily_pnl,
                'running_total': running_total,
                'high_conf_bets': high_conf_bets,
                'high_conf_wins': high_conf_wins,
                'medium_conf_bets': medium_conf_bets,
                'medium_conf_wins': medium_conf_wins
            }
            
            daily_summaries.append(daily_summary)
        
        # Save daily summary
        daily_df = pd.DataFrame(daily_summaries)
        daily_df.to_csv(self.daily_summary_file, index=False)
    
    def update_weekly_summary(self, betting_log):
        """Update weekly summary statistics"""
        
        betting_log['week'] = betting_log['date'].dt.to_period('W')
        
        weekly_stats = []
        
        for week, week_data in betting_log.groupby('week'):
            week_start = week.start_time.date()
            week_end = week.end_time.date()
            
            completed_games = week_data[week_data['bet_result'].isin(['won', 'lost'])]
            
            if len(completed_games) == 0:
                continue
            
            days_active = week_data['date'].dt.date.nunique()
            total_bets = len(week_data)
            total_wagered = week_data['bet_amount'].sum()
            total_wins = len(week_data[week_data['bet_result'] == 'won'])
            total_losses = len(week_data[week_data['bet_result'] == 'lost'])
            win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
            weekly_pnl = week_data['profit_loss'].sum()
            avg_daily_pnl = weekly_pnl / days_active if days_active > 0 else 0
            
            # Best and worst days
            daily_pnl = week_data.groupby(week_data['date'].dt.date)['profit_loss'].sum()
            best_day = daily_pnl.max() if len(daily_pnl) > 0 else 0
            worst_day = daily_pnl.min() if len(daily_pnl) > 0 else 0
            
            weekly_summary = {
                'week_start': week_start.strftime('%Y-%m-%d'),
                'week_end': week_end.strftime('%Y-%m-%d'),
                'days_active': days_active,
                'total_bets': total_bets,
                'total_wagered': total_wagered,
                'total_wins': total_wins,
                'total_losses': total_losses,
                'win_rate': round(win_rate, 3),
                'weekly_profit_loss': weekly_pnl,
                'avg_daily_pnl': round(avg_daily_pnl, 2),
                'best_day': best_day,
                'worst_day': worst_day
            }
            
            weekly_stats.append(weekly_summary)
        
        if weekly_stats:
            weekly_df = pd.DataFrame(weekly_stats)
            weekly_df.to_csv(self.weekly_summary_file, index=False)
    
    def update_monthly_summary(self, betting_log):
        """Update monthly summary statistics"""
        
        betting_log['month'] = betting_log['date'].dt.to_period('M')
        
        monthly_stats = []
        
        for month, month_data in betting_log.groupby('month'):
            completed_games = month_data[month_data['bet_result'].isin(['won', 'lost'])]
            
            if len(completed_games) == 0:
                continue
            
            days_active = month_data['date'].dt.date.nunique()
            total_bets = len(month_data)
            total_wagered = month_data['bet_amount'].sum()
            total_wins = len(month_data[month_data['bet_result'] == 'won'])
            total_losses = len(month_data[month_data['bet_result'] == 'lost'])
            win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
            monthly_pnl = month_data['profit_loss'].sum()
            avg_daily_pnl = monthly_pnl / days_active if days_active > 0 else 0
            roi_percent = (monthly_pnl / total_wagered * 100) if total_wagered > 0 else 0
            
            # Best and worst days
            daily_pnl = month_data.groupby(month_data['date'].dt.date)['profit_loss'].sum()
            best_day = daily_pnl.max() if len(daily_pnl) > 0 else 0
            worst_day = daily_pnl.min() if len(daily_pnl) > 0 else 0
            
            monthly_summary = {
                'month': str(month),
                'days_active': days_active,
                'total_bets': total_bets,
                'total_wagered': total_wagered,
                'total_wins': total_wins,
                'total_losses': total_losses,
                'win_rate': round(win_rate, 3),
                'monthly_profit_loss': monthly_pnl,
                'avg_daily_pnl': round(avg_daily_pnl, 2),
                'best_day': best_day,
                'worst_day': worst_day,
                'roi_percent': round(roi_percent, 2)
            }
            
            monthly_stats.append(monthly_summary)
        
        if monthly_stats:
            monthly_df = pd.DataFrame(monthly_stats)
            monthly_df.to_csv(self.monthly_summary_file, index=False)
    
    def check_responsible_gambling_limits(self):
        """Check for responsible gambling warning signs"""
        
        if not os.path.exists(self.daily_summary_file):
            return
        
        daily_summary = pd.read_csv(self.daily_summary_file)
        
        if len(daily_summary) == 0:
            return
        
        # Check recent performance
        recent_days = daily_summary.tail(7)  # Last 7 days
        total_recent_loss = recent_days[recent_days['daily_profit_loss'] < 0]['daily_profit_loss'].sum()
        
        if abs(total_recent_loss) > self.max_daily_loss_warning:
            print(f"\n⚠️  RESPONSIBLE GAMBLING WARNING")
            print(f"Recent 7-day losses: ${total_recent_loss:.2f}")
            print(f"Consider taking a break or reducing bet sizes.")
        
        # Check monthly losses
        if len(daily_summary) >= 20:  # At least 20 days of data
            monthly_loss = daily_summary.tail(30)['daily_profit_loss'].sum()
            if monthly_loss < -self.max_monthly_loss_warning:
                print(f"\n🚨 MONTHLY LOSS WARNING")
                print(f"Monthly losses: ${monthly_loss:.2f}")
                print(f"Please consider reviewing your betting strategy.")
    
    def show_daily_summary(self, days=7):
        """Show recent daily summary"""
        
        if not os.path.exists(self.daily_summary_file):
            print("No daily summary available")
            return
        
        daily_summary = pd.read_csv(self.daily_summary_file)
        recent_days = daily_summary.tail(days)
        
        print(f"\n📊 DAILY BETTING SUMMARY (Last {days} days)")
        print("="*80)
        
        for _, day in recent_days.iterrows():
            status = "📈" if day['daily_profit_loss'] > 0 else "📉" if day['daily_profit_loss'] < 0 else "📊"
            
            print(f"{status} {day['date']}")
            print(f"    Bets: {day['total_bets']} | Win Rate: {day['win_rate']:.1%} | "
                  f"P&L: ${day['daily_profit_loss']:+.2f} | Running: ${day['running_total']:+.2f}")
            
            if day['high_conf_bets'] > 0:
                high_rate = day['high_conf_wins'] / day['high_conf_bets']
                print(f"    HIGH: {day['high_conf_wins']}/{day['high_conf_bets']} ({high_rate:.1%})", end="")
            
            if day['medium_conf_bets'] > 0:
                med_rate = day['medium_conf_wins'] / day['medium_conf_bets']
                print(f" | MEDIUM: {day['medium_conf_wins']}/{day['medium_conf_bets']} ({med_rate:.1%})")
            else:
                print()
    
    def show_monthly_summary(self):
        """Show monthly performance summary"""
        
        if not os.path.exists(self.monthly_summary_file):
            print("No monthly summary available")
            return
        
        monthly_summary = pd.read_csv(self.monthly_summary_file)
        
        print(f"\n📅 MONTHLY BETTING PERFORMANCE")
        print("="*80)
        
        for _, month in monthly_summary.iterrows():
            status = "🚀" if month['monthly_profit_loss'] > 0 else "⚠️" if month['monthly_profit_loss'] < -500 else "📊"
            
            print(f"{status} {month['month']}")
            print(f"    Bets: {month['total_bets']} | Win Rate: {month['win_rate']:.1%} | "
                  f"ROI: {month['roi_percent']:+.1f}%")
            print(f"    Total P&L: ${month['monthly_profit_loss']:+.2f} | "
                  f"Avg Daily: ${month['avg_daily_pnl']:+.2f}")
            print(f"    Best Day: ${month['best_day']:+.2f} | "
                  f"Worst Day: ${month['worst_day']:+.2f}")
            print()

def main():
    """Main function for automated betting tracking"""
    
    # Your Odds API key
    ODDS_API_KEY = "47b36e3e637a7690621e258da00e29d7"
    
    tracker = AutomatedMLBBettingTracker(ODDS_API_KEY)
    tracker.initialize_tracking_files()
    
    print("🤖 AUTOMATED MLB BETTING TRACKER")
    print("="*60)
    print("⚠️  RESPONSIBLE GAMBLING REMINDER:")
    print("   • Only bet what you can afford to lose")
    print("   • Set daily/monthly loss limits")
    print("   • Take breaks if you experience losses")
    print("   • Sports betting involves risk")
    print("="*60)
    
    while True:
        print(f"\n📊 AUTOMATED BETTING TRACKER MENU")
        print("1. Process new predictions (set bet amount)")
        print("2. Show daily summary")
        print("3. Show monthly summary") 
        print("4. Check for new prediction files")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            try:
                bet_amount = float(input("Enter bet amount per HIGH/MEDIUM confidence game: $"))
                if bet_amount <= 0:
                    print("Bet amount must be positive")
                    continue
                
                tracker.process_daily_predictions(bet_amount)
                
            except ValueError:
                print("Invalid bet amount")
                
        elif choice == '2':
            days = input("Number of recent days to show (default 7): ").strip()
            days = int(days) if days.isdigit() else 7
            tracker.show_daily_summary(days)
            
        elif choice == '3':
            tracker.show_monthly_summary()
            
        elif choice == '4':
            unprocessed = tracker.get_unprocessed_prediction_files()
            if unprocessed:
                print(f"Found {len(unprocessed)} unprocessed prediction files:")
                for file in unprocessed:
                    print(f"  • {file}")
            else:
                print("No new prediction files found")
                
        elif choice == '5':
            print("Happy betting! Remember to gamble responsibly.")
            break
            
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()