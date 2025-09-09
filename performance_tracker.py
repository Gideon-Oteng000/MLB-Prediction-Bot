import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

class PerformanceTracker:
    def __init__(self):
        self.predictions_file = 'all_predictions.csv'
        self.results_file = 'prediction_results.csv'
    
    def collect_yesterdays_results(self):
        """Get results for yesterday's games and update predictions"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Try to find yesterday's prediction file
        prediction_file = f"predictions_{yesterday.replace('-', '')}.csv"
        
        try:
            predictions = pd.read_csv(prediction_file)
            print(f"Found predictions for {yesterday}: {len(predictions)} games")
        except:
            print(f"No prediction file found for {yesterday}")
            return None
        
        # Get actual results from MLB API
        print("Fetching actual results...")
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={yesterday}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            actual_results = {}
            if data['dates'] and len(data['dates']) > 0:
                games = data['dates'][0]['games']
                
                for game in games:
                    if game['status']['statusCode'] == 'F':  # Final games only
                        home_team = game['teams']['home']['team']['name']
                        away_team = game['teams']['away']['team']['name']
                        actual_winner = 'HOME' if game['teams']['home']['score'] > game['teams']['away']['score'] else 'AWAY'
                        
                        actual_results[f"{away_team}@{home_team}"] = {
                            'actual_winner': actual_winner,
                            'home_score': game['teams']['home']['score'],
                            'away_score': game['teams']['away']['score']
                        }
            
            print(f"Found results for {len(actual_results)} completed games")
            
        except Exception as e:
            print(f"Error getting results: {e}")
            return None
        
        # Match predictions with results
        results = []
        for _, pred in predictions.iterrows():
            game_key = f"{pred['away_team']}@{pred['home_team']}"
            
            if game_key in actual_results:
                result = actual_results[game_key]
                
                was_correct = pred['predicted_winner'] == result['actual_winner']
                
                results.append({
                    'date': yesterday,
                    'away_team': pred['away_team'],
                    'home_team': pred['home_team'],
                    'predicted_winner': pred['predicted_winner'],
                    'actual_winner': result['actual_winner'],
                    'home_win_probability': pred['home_win_probability'],
                    'confidence': pred['confidence'],
                    'correct': was_correct,
                    'home_score': result['home_score'],
                    'away_score': result['away_score']
                })
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Save/append to results file
            try:
                existing_results = pd.read_csv(self.results_file)
                combined_results = pd.concat([existing_results, results_df], ignore_index=True)
            except:
                combined_results = results_df
            
            combined_results.to_csv(self.results_file, index=False)
            
            # Print summary
            correct_predictions = sum([r['correct'] for r in results])
            accuracy = correct_predictions / len(results)
            
            print(f"\n=== YESTERDAY'S RESULTS ({yesterday}) ===")
            print(f"Total predictions: {len(results)}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Accuracy: {accuracy:.1%}")
            
            # Breakdown by confidence
            for conf_level in ['HIGH', 'MEDIUM', 'LOW']:
                conf_results = [r for r in results if r['confidence'] == conf_level]
                if conf_results:
                    conf_correct = sum([r['correct'] for r in conf_results])
                    conf_accuracy = conf_correct / len(conf_results)
                    print(f"  {conf_level} confidence: {conf_correct}/{len(conf_results)} ({conf_accuracy:.1%})")
            
            return results_df
        
        else:
            print("No matching results found")
            return None
    
    def show_overall_performance(self):
        """Show cumulative performance statistics"""
        try:
            results = pd.read_csv(self.results_file)
            results['date'] = pd.to_datetime(results['date'])
        except:
            print("No historical results found")
            return
        
        print(f"\n{'='*60}")
        print(f"OVERALL PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        total_predictions = len(results)
        total_correct = results['correct'].sum()
        overall_accuracy = total_correct / total_predictions
        
        print(f"Total predictions: {total_predictions}")
        print(f"Correct predictions: {total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.1%}")
        
        # Performance by confidence level
        print(f"\nPerformance by Confidence Level:")
        for conf_level in ['HIGH', 'MEDIUM', 'LOW']:
            conf_results = results[results['confidence'] == conf_level]
            if len(conf_results) > 0:
                conf_accuracy = conf_results['correct'].mean()
                print(f"  {conf_level}: {conf_results['correct'].sum()}/{len(conf_results)} ({conf_accuracy:.1%})")
        
        # Recent performance (last 10 days)
        recent_date = datetime.now() - timedelta(days=10)
        recent_results = results[results['date'] >= recent_date]
        
        if len(recent_results) > 0:
            recent_accuracy = recent_results['correct'].mean()
            print(f"\nLast 10 days: {recent_results['correct'].sum()}/{len(recent_results)} ({recent_accuracy:.1%})")
        
        # Profitable betting simulation
        print(f"\nPROFIT SIMULATION (betting $100 per HIGH/MEDIUM confidence game):")
        
        betting_games = results[results['confidence'].isin(['HIGH', 'MEDIUM'])]
        if len(betting_games) > 0:
            wins = betting_games['correct'].sum()
            losses = len(betting_games) - wins
            
            # Assuming -110 odds (win $90.90 for every $100 bet)
            profit = (wins * 90.90) - (losses * 100)
            roi = (profit / (len(betting_games) * 100)) * 100
            
            print(f"  Games bet: {len(betting_games)}")
            print(f"  Wins: {wins} (${wins * 90.90:.0f})")
            print(f"  Losses: {losses} (-${losses * 100:.0f})")
            print(f"  Net profit: ${profit:.0f}")
            print(f"  ROI: {roi:.1f}%")
            
            if roi > 5:
                print("  🚀 PROFITABLE! Your model is making money!")
            elif roi > 0:
                print("  📈 Slightly profitable - good start!")
            else:
                print("  📊 Need improvement - track more games")
        
        # Show recent predictions
        print(f"\nLAST 5 PREDICTIONS:")
        recent_5 = results.tail(5)
        for _, pred in recent_5.iterrows():
            status = "✅" if pred['correct'] else "❌"
            print(f"  {status} {pred['date']}: {pred['away_team']} @ {pred['home_team']} - Predicted {pred['predicted_winner']}, Actual {pred['actual_winner']}")
    
    def generate_insights(self):
        """Generate insights for model improvement"""
        try:
            results = pd.read_csv(self.results_file)
        except:
            print("Need more data for insights")
            return
        
        if len(results) < 10:
            print("Need at least 10 predictions for meaningful insights")
            return
        
        print(f"\n{'='*60}")
        print(f"MODEL INSIGHTS & RECOMMENDATIONS")
        print(f"{'='*60}")
        
        # Which confidence levels are most accurate?
        high_conf = results[results['confidence'] == 'HIGH']
        medium_conf = results[results['confidence'] == 'MEDIUM']
        
        if len(high_conf) > 0:
            high_accuracy = high_conf['correct'].mean()
            print(f"High confidence accuracy: {high_accuracy:.1%}")
            if high_accuracy > 0.6:
                print("  💡 HIGH confidence predictions are strong - bet more aggressively")
            else:
                print("  ⚠️  HIGH confidence may need tuning")
        
        # Home vs Away prediction accuracy
        home_preds = results[results['predicted_winner'] == 'HOME']
        away_preds = results[results['predicted_winner'] == 'AWAY']
        
        if len(home_preds) > 0 and len(away_preds) > 0:
            home_accuracy = home_preds['correct'].mean()
            away_accuracy = away_preds['correct'].mean()
            
            print(f"\nPrediction type accuracy:")
            print(f"  HOME predictions: {home_accuracy:.1%}")
            print(f"  AWAY predictions: {away_accuracy:.1%}")
            
            if abs(home_accuracy - away_accuracy) > 0.1:
                print("  💡 Consider adjusting model - significant bias detected")
        
        # Probability calibration
        high_prob = results[results['home_win_probability'] > 0.6]
        low_prob = results[results['home_win_probability'] < 0.4]
        
        if len(high_prob) > 0:
            high_prob_accuracy = high_prob['correct'].mean()
            print(f"\nHigh probability games (>60%): {high_prob_accuracy:.1%}")
        
        if len(low_prob) > 0:
            low_prob_accuracy = low_prob['correct'].mean()
            print(f"Low probability games (<40%): {low_prob_accuracy:.1%}")

def main():
    tracker = PerformanceTracker()
    
    print("MLB Performance Tracker")
    print("="*40)
    
    choice = input("What would you like to do?\n1. Update with yesterday's results\n2. Show overall performance\n3. Generate insights\n4. All of the above\nChoice: ")
    
    if choice in ['1', '4']:
        tracker.collect_yesterdays_results()
        print()
    
    if choice in ['2', '4']:
        tracker.show_overall_performance()
        print()
    
    if choice in ['3', '4']:
        tracker.generate_insights()

if __name__ == "__main__":
    main()