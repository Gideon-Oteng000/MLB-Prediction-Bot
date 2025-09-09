import pandas as pd
import requests
from datetime import datetime, timedelta
import time

class TotalBasesTracker:
    def __init__(self):
        self.predictions_file = None
        
    def load_predictions(self, date_str=None):
        """Load predictions from CSV file"""
        if not date_str:
            date_str = datetime.now().strftime('%Y%m%d')
        
        filename = f"fixed_total_bases_{date_str}.csv"
        try:
            predictions = pd.read_csv(filename)
            print(f"Loaded {len(predictions)} predictions from {filename}")
            return predictions
        except FileNotFoundError:
            print(f"Could not find {filename}")
            return None
    
    def get_player_stats_from_game(self, game_id):
        """Get player statistics from a completed game"""
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'liveData' not in data or 'boxscore' not in data['liveData']:
                return []
            
            boxscore = data['liveData']['boxscore']
            player_stats = []
            
            # Process both teams
            for team_type in ['home', 'away']:
                if team_type in boxscore['teams']:
                    team_data = boxscore['teams'][team_type]
                    team_name = data['gameData']['teams'][team_type]['name']
                    
                    if 'batters' in team_data:
                        for player_id in team_data['batters']:
                            if str(player_id) in team_data['players']:
                                player = team_data['players'][str(player_id)]
                                
                                if 'stats' in player and 'batting' in player['stats']:
                                    batting = player['stats']['batting']
                                    
                                    # Calculate total bases
                                    hits = int(batting.get('hits', 0))
                                    doubles = int(batting.get('doubles', 0)) 
                                    triples = int(batting.get('triples', 0))
                                    home_runs = int(batting.get('homeRuns', 0))
                                    
                                    singles = hits - doubles - triples - home_runs
                                    total_bases = singles + (doubles * 2) + (triples * 3) + (home_runs * 4)
                                    
                                    player_stats.append({
                                        'player_name': player['person']['fullName'],
                                        'team': team_name,
                                        'at_bats': int(batting.get('atBats', 0)),
                                        'hits': hits,
                                        'singles': singles,
                                        'doubles': doubles,
                                        'triples': triples,
                                        'home_runs': home_runs,
                                        'total_bases': total_bases
                                    })
            
            return player_stats
            
        except Exception as e:
            print(f"Error getting game data for {game_id}: {e}")
            return []
    
    def get_completed_games(self, date_str):
        """Get completed games for a specific date"""
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            completed_games = []
            if 'dates' in data and data['dates']:
                for game in data['dates'][0].get('games', []):
                    if game['status']['statusCode'] == 'F':  # Final
                        completed_games.append({
                            'game_id': game['gamePk'],
                            'away_team': game['teams']['away']['team']['name'],
                            'home_team': game['teams']['home']['team']['name']
                        })
            
            return completed_games
            
        except Exception as e:
            print(f"Error getting games for {date_str}: {e}")
            return []
    
    def check_predictions(self, target_date=None):
        """Check predictions against actual results"""
        if not target_date:
            # Default to yesterday's games
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        date_str = target_date.replace('-', '')
        
        print(f"Checking total bases predictions for {target_date}...")
        
        # Load predictions
        predictions = self.load_predictions(date_str)
        if predictions is None:
            return
        
        # Get completed games
        completed_games = self.get_completed_games(target_date)
        if not completed_games:
            print(f"No completed games found for {target_date}")
            return
        
        print(f"Found {len(completed_games)} completed games")
        
        # Get actual player stats
        all_actual_stats = []
        for game in completed_games:
            print(f"Processing {game['away_team']} @ {game['home_team']}...")
            game_stats = self.get_player_stats_from_game(game['game_id'])
            all_actual_stats.extend(game_stats)
            time.sleep(0.5)  # Be nice to API
        
        # Match predictions with actual results
        results = []
        unmatched_predictions = []
        available_players = [stat['player_name'] for stat in all_actual_stats]
        
        print(f"\nFound {len(all_actual_stats)} players with stats:")
        for stat in all_actual_stats[:10]:  # Show first 10
            print(f"  {stat['player_name']} ({stat['team']}) - {stat['total_bases']} TB")
        if len(all_actual_stats) > 10:
            print(f"  ... and {len(all_actual_stats) - 10} more players")
        
        print(f"\nMatching {len(predictions)} predictions to actual results...")
        
        for _, pred in predictions.iterrows():
            player_name = pred['player']
            predicted_tb = pred['predicted_tb']
            recommendation = pred['recommendation']
            confidence = pred['confidence']
            
            # Find matching actual stats
            actual_stat = None
            for stat in all_actual_stats:
                if self.names_match(player_name, stat['player_name']):
                    actual_stat = stat
                    break
            
            if actual_stat:
                actual_tb = actual_stat['total_bases']
                
                # Determine if prediction hit
                prediction_hit = self.evaluate_prediction(recommendation, actual_tb)
                
                results.append({
                    'player': player_name,
                    'team': pred['team'],
                    'predicted_tb': predicted_tb,
                    'actual_tb': actual_tb,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'hit': prediction_hit,
                    'at_bats': actual_stat['at_bats'],
                    'hits': actual_stat['hits'],
                    'singles': actual_stat['singles'],
                    'doubles': actual_stat['doubles'],
                    'triples': actual_stat['triples'],
                    'home_runs': actual_stat['home_runs']
                })
                print(f"  Matched: {player_name} -> {actual_stat['player_name']} ({actual_tb} TB)")
            else:
                # Player didn't play or couldn't find
                unmatched_predictions.append(player_name)
                results.append({
                    'player': player_name,
                    'team': pred['team'],
                    'predicted_tb': predicted_tb,
                    'actual_tb': None,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'hit': 'DNP',  # Did not play
                    'at_bats': 0,
                    'hits': 0,
                    'singles': 0,
                    'doubles': 0,
                    'triples': 0,
                    'home_runs': 0
                })
        
        # Show unmatched predictions for debugging
        if unmatched_predictions:
            print(f"\nCould not match {len(unmatched_predictions)} predictions:")
            for pred_name in unmatched_predictions[:10]:
                print(f"  {pred_name}")
                # Try to find similar names
                similar = []
                for actual_name in available_players:
                    if (pred_name.split()[-1].lower() in actual_name.lower() or
                        actual_name.split()[-1].lower() in pred_name.lower()):
                        similar.append(actual_name)
                if similar:
                    print(f"    Similar: {similar[:3]}")
            if len(unmatched_predictions) > 10:
                print(f"  ... and {len(unmatched_predictions) - 10} more")
        
        # Display results
        self.display_results(results, target_date)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_filename = f"tb_results_{date_str}.csv"
        results_df.to_csv(results_filename, index=False)
        print(f"\nResults saved to: {results_filename}")
        
        return results
    
    def names_match(self, pred_name, actual_name):
        """Check if player names match (handle variations)"""
        pred_clean = pred_name.lower().replace('.', '').replace(',', '').replace("'", "").replace(" jr", "").replace(" sr", "").strip()
        actual_clean = actual_name.lower().replace('.', '').replace(',', '').replace("'", "").replace(" jr", "").replace(" sr", "").strip()
        
        # Exact match
        if pred_clean == actual_clean:
            return True
        
        # Split into parts
        pred_parts = pred_clean.split()
        actual_parts = actual_clean.split()
        
        if len(pred_parts) >= 2 and len(actual_parts) >= 2:
            # Last name match + first name/initial match
            last_name_match = pred_parts[-1] == actual_parts[-1]
            first_match = (pred_parts[0] == actual_parts[0] or 
                          pred_parts[0][0] == actual_parts[0][0] or
                          actual_parts[0][0] == pred_parts[0][0])
            
            if last_name_match and first_match:
                return True
            
            # Handle middle names - check if first and last match ignoring middle
            if (pred_parts[0] == actual_parts[0] and 
                pred_parts[-1] == actual_parts[-1]):
                return True
        
        # Check if one name is contained in the other
        if pred_clean in actual_clean or actual_clean in pred_clean:
            return True
        
        return False
    
    def evaluate_prediction(self, recommendation, actual_tb):
        """Determine if prediction hit based on recommendation"""
        if "OVER 1.5" in recommendation:
            return actual_tb > 1.5
        elif "UNDER 1.5" in recommendation:
            return actual_tb < 1.5
        elif "OVER 2.5" in recommendation:
            return actual_tb > 2.5
        elif "UNDER 2.5" in recommendation:
            return actual_tb < 2.5
        elif "OVER 0.5" in recommendation:
            return actual_tb > 0.5
        elif "UNDER 0.5" in recommendation:
            return actual_tb < 0.5
        else:
            return False
    
    def display_results(self, results, target_date):
        """Display prediction results"""
        print(f"\n" + "="*70)
        print(f"TOTAL BASES PREDICTION RESULTS - {target_date}")
        print(f"="*70)
        
        # Filter out DNP
        played_results = [r for r in results if r['hit'] != 'DNP']
        dnp_count = len([r for r in results if r['hit'] == 'DNP'])
        
        if not played_results:
            print("No results to analyze (all players DNP)")
            return
        
        # Overall statistics
        total_predictions = len(played_results)
        hits = len([r for r in played_results if r['hit']])
        accuracy = hits / total_predictions if total_predictions > 0 else 0
        
        print(f"Total predictions: {total_predictions}")
        print(f"Players who didn't play: {dnp_count}")
        print(f"Correct predictions: {hits}")
        print(f"Overall accuracy: {accuracy:.1%}")
        
        # By confidence level
        print(f"\nAccuracy by Confidence:")
        for conf_level in ['HIGH', 'MEDIUM']:
            conf_results = [r for r in played_results if r['confidence'] == conf_level]
            if conf_results:
                conf_hits = len([r for r in conf_results if r['hit']])
                conf_accuracy = conf_hits / len(conf_results)
                print(f"  {conf_level}: {conf_hits}/{len(conf_results)} ({conf_accuracy:.1%})")
        
        # Show individual results
        print(f"\nINDIVIDUAL RESULTS:")
        print("-" * 70)
        
        for result in played_results:
            hit_symbol = "✅" if result['hit'] else "❌"
            print(f"{hit_symbol} {result['player']} ({result['team']})")
            print(f"    Predicted: {result['predicted_tb']:.1f} TB | Actual: {result['actual_tb']} TB")
            print(f"    Bet: {result['recommendation']} ({result['confidence']})")
            
            if result['actual_tb'] > 0:
                breakdown = []
                if result['singles'] > 0:
                    breakdown.append(f"{result['singles']}×1B")
                if result['doubles'] > 0:
                    breakdown.append(f"{result['doubles']}×2B")
                if result['triples'] > 0:
                    breakdown.append(f"{result['triples']}×3B")
                if result['home_runs'] > 0:
                    breakdown.append(f"{result['home_runs']}×HR")
                
                if breakdown:
                    print(f"    Breakdown: {', '.join(breakdown)} = {result['actual_tb']} TB")
            print()
        
        # Best and worst predictions
        if played_results:
            biggest_miss = max(played_results, 
                             key=lambda x: abs(x['predicted_tb'] - x['actual_tb']) if x['actual_tb'] is not None else 0)
            closest_pred = min(played_results,
                             key=lambda x: abs(x['predicted_tb'] - x['actual_tb']) if x['actual_tb'] is not None else float('inf'))
            
            print(f"Biggest miss: {biggest_miss['player']} ")
            print(f"  Predicted {biggest_miss['predicted_tb']:.1f}, actual {biggest_miss['actual_tb']}")
            print(f"Closest prediction: {closest_pred['player']}")
            print(f"  Predicted {closest_pred['predicted_tb']:.1f}, actual {closest_pred['actual_tb']}")

def main():
    print("Total Bases Prediction Tracker")
    print("="*40)
    
    tracker = TotalBasesTracker()
    
    # Get date to check
    date_input = input("Enter date to check (YYYY-MM-DD) or press Enter for yesterday: ").strip()
    
    if date_input:
        target_date = date_input
    else:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Check predictions
    results = tracker.check_predictions(target_date)

if __name__ == "__main__":
    main()