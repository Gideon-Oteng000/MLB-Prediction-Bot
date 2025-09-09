import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
import os

class WeeklyPerformanceTracker:
    def __init__(self, odds_api_key):
        self.odds_api_key = odds_api_key
        self.bet_amount = 50  # $50 per bet
        self.base_url = "https://api.the-odds-api.com/v4"
        self.tracking_file = 'weekly_betting_tracker.csv'
        
    def get_week_dates(self, date=None):
        """Get Sunday to Saturday week dates for a given date"""
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        # Find the Sunday of this week
        days_since_sunday = date.weekday() + 1  # Monday = 0, Sunday = 6
        if days_since_sunday == 7:  # If it's Sunday
            days_since_sunday = 0
            
        week_start = date - timedelta(days=days_since_sunday)
        week_end = week_start + timedelta(days=6)
        
        return week_start.date(), week_end.date()
    
    def get_current_odds(self, away_team, home_team, market_type='h2h'):
        """Get current odds for a specific game"""
        url = f"{self.base_url}/sports/baseball_mlb/odds"
        params = {
            'api_key': self.odds_api_key,
            'regions': 'us',
            'markets': market_type,
            'oddsFormat': 'american'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                return None
                
            data = response.json()
            
            # Find matching game
            for game in data:
                if (game['home_team'] == home_team and game['away_team'] == away_team):
                    # Extract odds from first major sportsbook
                    for bookmaker in game['bookmakers']:
                        if bookmaker['key'] in ['fanduel', 'draftkings', 'betmgm']:
                            for market in bookmaker['markets']:
                                if market['key'] == market_type:
                                    odds_data = {}
                                    for outcome in market['outcomes']:
                                        if market_type == 'h2h':  # Moneyline
                                            if outcome['name'] == home_team:
                                                odds_data['home_ml'] = outcome['price']
                                            else:
                                                odds_data['away_ml'] = outcome['price']
                                        elif market_type == 'spreads':  # Run line
                                            if outcome['name'] == home_team:
                                                odds_data['home_rl'] = outcome['price']
                                                odds_data['home_spread'] = outcome['point']
                                            else:
                                                odds_data['away_rl'] = outcome['price']
                                                odds_data['away_spread'] = outcome['point']
                                        elif market_type == 'totals':  # Over/Under
                                            if outcome['name'] == 'Over':
                                                odds_data['over_odds'] = outcome['price']
                                                odds_data['total_line'] = outcome['point']
                                            else:
                                                odds_data['under_odds'] = outcome['price']
                                    return odds_data
            return None
            
        except Exception as e:
            print(f"Error fetching odds: {e}")
            return None
    
    def american_to_decimal(self, american_odds):
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def calculate_payout(self, bet_amount, american_odds, won=True):
        """Calculate payout for a bet"""
        if not won:
            return -bet_amount
        
        if american_odds > 0:
            profit = (american_odds / 100) * bet_amount
        else:
            profit = (100 / abs(american_odds)) * bet_amount
        
        return profit  # Just profit, not total return
    
    def load_predictions(self, date_str, bet_types=['moneyline', 'runline', 'total_bases']):
        """Load predictions for a specific date"""
        predictions = []
        
        # Load different prediction files
        files_to_check = [
            f"odds_api_predictions_{date_str}.csv",  # From odds API system
            f"fixed_runline_predictions_{date_str}.csv",  # Run line predictions
            f"fixed_total_bases_{date_str}.csv",  # Total bases props
            f"home_run_picks_{date_str}.csv"  # Home run props
        ]
        
        for filename in files_to_check:
            try:
                df = pd.read_csv(filename)
                
                # Determine bet type from filename
                if 'runline' in filename:
                    bet_type = 'runline'
                elif 'total_bases' in filename:
                    bet_type = 'total_bases'
                elif 'home_run' in filename:
                    bet_type = 'home_run'
                else:
                    bet_type = 'moneyline'
                
                # Add bet type and filter for HIGH/MEDIUM confidence
                df['bet_type'] = bet_type
                df['date'] = date_str.replace('_', '-') if '_' in date_str else date_str
                
                # Filter for bets we would actually place
                if 'confidence' in df.columns:
                    betting_predictions = df[df['confidence'].isin(['HIGH', 'MEDIUM'])]
                else:
                    betting_predictions = df  # Assume all are betting worthy
                
                predictions.append(betting_predictions)
                print(f"Loaded {len(betting_predictions)} betting predictions from {filename}")
                
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        if predictions:
            return pd.concat(predictions, ignore_index=True)
        else:
            print(f"No prediction files found for {date_str}")
            return pd.DataFrame()
    
    def get_game_results(self, date_str):
        """Get completed game results for a date"""
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            results = []
            if 'dates' in data and data['dates']:
                for game in data['dates'][0].get('games', []):
                    if game['status']['statusCode'] == 'F':  # Final
                        home_score = game['teams']['home']['score']
                        away_score = game['teams']['away']['score']
                        margin = abs(home_score - away_score)
                        total_runs = home_score + away_score
                        
                        results.append({
                            'away_team': game['teams']['away']['team']['name'],
                            'home_team': game['teams']['home']['team']['name'],
                            'away_score': away_score,
                            'home_score': home_score,
                            'winner': 'home' if home_score > away_score else 'away',
                            'margin': margin,
                            'total_runs': total_runs,
                            'date': date_str
                        })
            
            return results
            
        except Exception as e:
            print(f"Error getting game results for {date_str}: {e}")
            return []
    
    def determine_bet_outcome(self, prediction, game_result, odds_data):
        """Determine if a bet won and calculate profit/loss"""
        bet_type = prediction.get('bet_type', 'moneyline')
        
        if bet_type == 'moneyline':
            # Moneyline bet outcome
            predicted_winner = prediction.get('prediction', '')
            actual_winner = game_result['winner']
            
            if 'home' in predicted_winner.lower() or game_result['home_team'] in predicted_winner:
                bet_on = 'home'
                odds = odds_data.get('home_ml', -110)
            else:
                bet_on = 'away'
                odds = odds_data.get('away_ml', -110)
            
            won = (bet_on == actual_winner)
            
        elif bet_type == 'runline':
            # Run line bet outcome
            prediction_text = prediction.get('prediction', '')
            margin = game_result['margin']
            actual_winner = game_result['winner']
            
            if '-1.5' in prediction_text:
                # Betting on favorite to cover
                if game_result['home_team'] in prediction_text:
                    won = (actual_winner == 'home' and margin >= 2)
                    odds = odds_data.get('home_rl', -110)
                else:
                    won = (actual_winner == 'away' and margin >= 2)
                    odds = odds_data.get('away_rl', -110)
            else:
                # Betting on underdog +1.5
                if game_result['home_team'] in prediction_text:
                    won = (actual_winner == 'home' or margin <= 1)
                    odds = odds_data.get('home_rl', -110)
                else:
                    won = (actual_winner == 'away' or margin <= 1)
                    odds = odds_data.get('away_rl', -110)
        
        elif bet_type == 'total_bases':
            # For props, assume standard -110 odds
            odds = -110
            won = True  # Placeholder - would need actual player stats
            
        elif bet_type == 'home_run':
            # Home run props
            odds = 400 if 'YES' in prediction.get('recommendation', '') else -500
            won = True  # Placeholder - would need actual player stats
            
        else:
            odds = -110
            won = False
        
        payout = self.calculate_payout(self.bet_amount, odds, won)
        
        return {
            'won': won,
            'odds': odds,
            'payout': payout,
            'bet_amount': self.bet_amount
        }
    
    def process_daily_bets(self, target_date):
        """Process all bets for a specific date"""
        date_str = target_date.replace('-', '') if '-' in target_date else target_date
        date_formatted = target_date if '-' in target_date else f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"
        
        print(f"\nProcessing bets for {date_formatted}...")
        
        # Load predictions
        predictions = self.load_predictions(date_str)
        if predictions.empty:
            print("No predictions found for this date")
            return []
        
        # Get game results
        game_results = self.get_game_results(date_formatted)
        if not game_results:
            print("No completed games found for this date")
            return []
        
        processed_bets = []
        
        for _, prediction in predictions.iterrows():
            # Find matching game result
            game_result = None
            for result in game_results:
                if (prediction.get('home_team') == result['home_team'] and 
                    prediction.get('away_team') == result['away_team']):
                    game_result = result
                    break
            
            if not game_result:
                continue
            
            # Get current odds
            odds_data = self.get_current_odds(
                game_result['away_team'], 
                game_result['home_team'], 
                'spreads' if prediction.get('bet_type') == 'runline' else 'h2h'
            )
            
            if not odds_data:
                odds_data = {'home_ml': -110, 'away_ml': -110}  # Default odds
            
            # Determine outcome
            outcome = self.determine_bet_outcome(prediction, game_result, odds_data)
            
            bet_record = {
                'date': date_formatted,
                'game': f"{game_result['away_team']} @ {game_result['home_team']}",
                'bet_type': prediction.get('bet_type', 'moneyline'),
                'prediction': prediction.get('prediction', ''),
                'confidence': prediction.get('confidence', 'MEDIUM'),
                'odds': outcome['odds'],
                'bet_amount': outcome['bet_amount'],
                'won': outcome['won'],
                'payout': outcome['payout'],
                'game_result': f"{game_result['away_score']}-{game_result['home_score']}",
                'week_start': self.get_week_dates(datetime.strptime(date_formatted, '%Y-%m-%d'))[0]
            }
            
            processed_bets.append(bet_record)
        
        return processed_bets
    
    def save_bets_to_tracker(self, bets):
        """Save processed bets to tracking file"""
        if not bets:
            return
        
        new_bets_df = pd.DataFrame(bets)
        
        # Load existing data or create new file
        if os.path.exists(self.tracking_file):
            existing_df = pd.read_csv(self.tracking_file)
            existing_df['date'] = pd.to_datetime(existing_df['date']).dt.date
            existing_df['week_start'] = pd.to_datetime(existing_df['week_start']).dt.date
            
            # Remove existing bets for this date to avoid duplicates
            dates_to_update = new_bets_df['date'].unique()
            existing_df = existing_df[~existing_df['date'].astype(str).isin(dates_to_update)]
            
            # Combine with new bets
            combined_df = pd.concat([existing_df, new_bets_df], ignore_index=True)
        else:
            combined_df = new_bets_df
        
        combined_df.to_csv(self.tracking_file, index=False)
        print(f"Saved {len(bets)} bets to {self.tracking_file}")
    
    def generate_weekly_report(self, week_start_date=None):
        """Generate weekly performance report"""
        if not os.path.exists(self.tracking_file):
            print("No betting data found. Please process some daily bets first.")
            return
        
        df = pd.read_csv(self.tracking_file)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['week_start'] = pd.to_datetime(df['week_start']).dt.date
        
        if week_start_date:
            if isinstance(week_start_date, str):
                week_start_date = datetime.strptime(week_start_date, '%Y-%m-%d').date()
            week_data = df[df['week_start'] == week_start_date]
        else:
            # Current week
            current_week_start = self.get_week_dates()[0]
            week_data = df[df['week_start'] == current_week_start]
            week_start_date = current_week_start
        
        if week_data.empty:
            print(f"No betting data found for week starting {week_start_date}")
            return
        
        week_end_date = week_start_date + timedelta(days=6)
        
        # Calculate weekly stats
        total_bets = len(week_data)
        wins = len(week_data[week_data['won'] == True])
        losses = total_bets - wins
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        total_wagered = week_data['bet_amount'].sum()
        total_profit = week_data['payout'].sum()
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"WEEKLY BETTING REPORT")
        print(f"Week: {week_start_date} to {week_end_date}")
        print(f"{'='*60}")
        
        print(f"OVERALL PERFORMANCE:")
        print(f"  Total Bets: {total_bets}")
        print(f"  Wins: {wins}")
        print(f"  Losses: {losses}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Total Wagered: ${total_wagered:.2f}")
        
        if total_profit > 0:
            print(f"  Total Profit: +${total_profit:.2f}")
        else:
            print(f"  Total Loss: ${total_profit:.2f}")
        print(f"  ROI: {roi:.1f}%")
        
        # Performance by bet type
        print(f"\nPERFORMANCE BY BET TYPE:")
        for bet_type in week_data['bet_type'].unique():
            type_data = week_data[week_data['bet_type'] == bet_type]
            type_wins = len(type_data[type_data['won'] == True])
            type_total = len(type_data)
            type_profit = type_data['payout'].sum()
            type_win_rate = type_wins / type_total if type_total > 0 else 0
            
            print(f"  {bet_type.upper()}: {type_wins}/{type_total} ({type_win_rate:.1%}) - ${type_profit:+.2f}")
        
        # Performance by confidence
        print(f"\nPERFORMANCE BY CONFIDENCE:")
        for confidence in ['HIGH', 'MEDIUM']:
            conf_data = week_data[week_data['confidence'] == confidence]
            if not conf_data.empty:
                conf_wins = len(conf_data[conf_data['won'] == True])
                conf_total = len(conf_data)
                conf_profit = conf_data['payout'].sum()
                conf_win_rate = conf_wins / conf_total if conf_total > 0 else 0
                
                print(f"  {confidence}: {conf_wins}/{conf_total} ({conf_win_rate:.1%}) - ${conf_profit:+.2f}")
        
        # Daily breakdown
        print(f"\nDAILY BREAKDOWN:")
        daily_summary = week_data.groupby('date').agg({
            'won': ['count', 'sum'],
            'payout': 'sum'
        }).round(2)
        
        for date in daily_summary.index:
            day_bets = daily_summary.loc[date, ('won', 'count')]
            day_wins = daily_summary.loc[date, ('won', 'sum')]
            day_profit = daily_summary.loc[date, ('payout', 'sum')]
            day_win_rate = day_wins / day_bets if day_bets > 0 else 0
            
            print(f"  {date}: {int(day_wins)}/{int(day_bets)} ({day_win_rate:.1%}) - ${day_profit:+.2f}")
        
        return {
            'week_start': week_start_date,
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': roi
        }
    
    def update_tracker(self, date_str=None):
        """Update tracker with bets from a specific date"""
        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        bets = self.process_daily_bets(date_str)
        if bets:
            self.save_bets_to_tracker(bets)
            return len(bets)
        return 0

def main():
    print("Weekly Betting Performance Tracker")
    print("$50 per bet | Sunday-Saturday weeks")
    print("="*50)
    
    # Get API key
    api_key = input("Enter your Odds API key: ").strip()
    if not api_key:
        print("No API key provided")
        return
    
    tracker = WeeklyPerformanceTracker(api_key)
    
    print("\nWhat would you like to do?")
    print("1. Update tracker with yesterday's bets")
    print("2. Update tracker with specific date")
    print("3. Generate current week report")
    print("4. Generate specific week report")
    
    choice = input("Choice (1-4): ").strip()
    
    if choice == '1':
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        bet_count = tracker.update_tracker(yesterday)
        print(f"Processed {bet_count} bets for {yesterday}")
        tracker.generate_weekly_report()
        
    elif choice == '2':
        date_str = input("Enter date (YYYY-MM-DD): ").strip()
        bet_count = tracker.update_tracker(date_str)
        print(f"Processed {bet_count} bets for {date_str}")
        
    elif choice == '3':
        tracker.generate_weekly_report()
        
    elif choice == '4':
        week_start = input("Enter week start date (YYYY-MM-DD, Sunday): ").strip()
        tracker.generate_weekly_report(week_start)

if __name__ == "__main__":
    main()