import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta
import time
import os

class FixedMLBRunLineSystem:
    def __init__(self):
        self.model = None
        self.historical_data = None
        
    def load_historical_data(self):
        """Load historical games data"""
        try:
            self.historical_data = pd.read_csv('historical_mlb_games.csv')
            self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
            self.historical_data = self.historical_data.sort_values('date').reset_index(drop=True)
            print(f"Loaded {len(self.historical_data)} historical games for run line analysis")
            return True
        except:
            print("Error: Could not load historical_mlb_games.csv")
            return False
    
    def calculate_team_runline_stats(self, team, as_of_date, games_back=20):
        """Calculate realistic run line stats for a team"""
        
        # Get team's recent games
        team_games = self.historical_data[
            ((self.historical_data['home_team'] == team) | 
             (self.historical_data['away_team'] == team)) &
            (self.historical_data['date'] < as_of_date)
        ].tail(games_back)
        
        if len(team_games) < 10:
            return {
                'total_games': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.5,
                'avg_runs_scored': 4.5,
                'avg_runs_allowed': 4.5,
                'run_differential': 0.0,
                'wins_by_2_plus': 0,
                'losses_by_1': 0,
                'blowout_rate': 0.3,  # When they win, % of time it's by 2+
                'close_loss_rate': 0.5  # When they lose, % of time it's by 1
            }
        
        # Analyze each game
        wins = 0
        losses = 0
        total_runs_scored = 0
        total_runs_allowed = 0
        wins_by_2_plus = 0
        losses_by_1 = 0
        
        for _, game in team_games.iterrows():
            is_home = (game['home_team'] == team)
            team_score = game['home_score'] if is_home else game['away_score']
            opp_score = game['away_score'] if is_home else game['home_score']
            
            total_runs_scored += team_score
            total_runs_allowed += opp_score
            
            margin = abs(team_score - opp_score)
            team_won = (team_score > opp_score)
            
            if team_won:
                wins += 1
                if margin >= 2:
                    wins_by_2_plus += 1
            else:
                losses += 1
                if margin == 1:
                    losses_by_1 += 1
        
        # Calculate rates
        total_games = len(team_games)
        win_rate = wins / total_games
        avg_runs_scored = total_runs_scored / total_games
        avg_runs_allowed = total_runs_allowed / total_games
        run_differential = avg_runs_scored - avg_runs_allowed
        
        # Key run line metrics
        blowout_rate = wins_by_2_plus / max(wins, 1)  # Of wins, what % are by 2+?
        close_loss_rate = losses_by_1 / max(losses, 1)  # Of losses, what % are by 1?
        
        return {
            'total_games': total_games,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_runs_scored': avg_runs_scored,
            'avg_runs_allowed': avg_runs_allowed,
            'run_differential': run_differential,
            'wins_by_2_plus': wins_by_2_plus,
            'losses_by_1': losses_by_1,
            'blowout_rate': blowout_rate,
            'close_loss_rate': close_loss_rate
        }
    
    def determine_favorite(self, home_stats, away_stats):
        """Determine which team should be the favorite"""
        
        # Calculate team strength scores
        home_strength = (
            home_stats['win_rate'] * 0.4 +
            (home_stats['run_differential'] + 3) / 6 * 0.4 +  # Normalize run diff
            home_stats['blowout_rate'] * 0.2
        ) + 0.03  # Small home field advantage
        
        away_strength = (
            away_stats['win_rate'] * 0.4 +
            (away_stats['run_differential'] + 3) / 6 * 0.4 +
            away_stats['blowout_rate'] * 0.2
        )
        
        if home_strength > away_strength:
            return 'HOME', home_strength - away_strength
        else:
            return 'AWAY', away_strength - home_strength
    
    def predict_runline_simple(self, home_team, away_team):
        """Simple, logical run line prediction"""
        
        current_date = datetime.now()
        
        # Get team stats
        home_stats = self.calculate_team_runline_stats(home_team, current_date)
        away_stats = self.calculate_team_runline_stats(away_team, current_date)
        
        # Check data sufficiency
        if home_stats['total_games'] < 10 or away_stats['total_games'] < 10:
            return {
                'prediction': 'Insufficient data',
                'confidence': 'LOW',
                'home_stats': home_stats,
                'away_stats': away_stats
            }
        
        # Determine favorite
        predicted_favorite, strength_diff = self.determine_favorite(home_stats, away_stats)
        
        if predicted_favorite == 'HOME':
            favorite_team = home_team
            underdog_team = away_team
            fav_stats = home_stats
            und_stats = away_stats
        else:
            favorite_team = away_team
            underdog_team = home_team
            fav_stats = away_stats
            und_stats = home_stats
        
        # Calculate favorite's chance to cover -1.5
        base_cover_chance = 0.45  # Base probability for covering -1.5
        
        # Adjustments
        adjustments = []
        
        # 1. Team strength difference
        strength_adj = min(strength_diff * 0.3, 0.15)  # Cap at 15%
        adjustments.append(('Strength difference', strength_adj))
        
        # 2. Favorite's blowout tendency
        blowout_adj = (fav_stats['blowout_rate'] - 0.4) * 0.2  # Above 40% = positive
        adjustments.append(('Favorite blowout rate', blowout_adj))
        
        # 3. Underdog's close game tendency
        close_adj = (und_stats['close_loss_rate'] - 0.5) * 0.15  # Above 50% = helps underdog
        adjustments.append(('Underdog close games', -close_adj))  # Negative helps favorite
        
        # 4. Run differential advantage
        run_diff_advantage = fav_stats['run_differential'] - und_stats['run_differential']
        run_diff_adj = min(run_diff_advantage * 0.08, 0.12)  # Cap at 12%
        adjustments.append(('Run differential', run_diff_adj))
        
        # Calculate final probability
        total_adjustment = sum(adj[1] for adj in adjustments)
        favorite_cover_prob = base_cover_chance + total_adjustment
        favorite_cover_prob = max(0.15, min(0.85, favorite_cover_prob))  # Keep between 15-85%
        
        # Determine recommendation and confidence
        if favorite_cover_prob > 0.58:
            prediction = f"{favorite_team} -1.5"
            confidence = "HIGH" if favorite_cover_prob > 0.65 else "MEDIUM"
        elif favorite_cover_prob < 0.42:
            prediction = f"{underdog_team} +1.5"
            confidence = "HIGH" if favorite_cover_prob < 0.35 else "MEDIUM"
        else:
            prediction = "No strong lean"
            confidence = "LOW"
        
        return {
            'favorite_team': favorite_team,
            'underdog_team': underdog_team,
            'favorite_cover_probability': favorite_cover_prob,
            'underdog_cover_probability': 1 - favorite_cover_prob,
            'prediction': prediction,
            'confidence': confidence,
            'strength_difference': strength_diff,
            'adjustments': adjustments,
            'home_stats': home_stats,
            'away_stats': away_stats,
            'analysis': self.generate_analysis(fav_stats, und_stats, favorite_team, underdog_team, adjustments)
        }
    
    def generate_analysis(self, fav_stats, und_stats, fav_team, und_team, adjustments):
        """Generate clear analysis"""
        analysis = []
        
        # Key stats
        if fav_stats['blowout_rate'] > 0.5:
            analysis.append(f"{fav_team} wins by 2+ runs {fav_stats['blowout_rate']:.1%} of the time")
        
        if und_stats['close_loss_rate'] > 0.6:
            analysis.append(f"{und_team} loses by 1 run {und_stats['close_loss_rate']:.1%} of the time")
        
        run_diff = fav_stats['run_differential'] - und_stats['run_differential']
        if run_diff > 1.0:
            analysis.append(f"{fav_team} has +{run_diff:.1f} run differential advantage")
        
        # Strength comparison
        if fav_stats['win_rate'] - und_stats['win_rate'] > 0.1:
            analysis.append(f"{fav_team} much better team ({fav_stats['win_rate']:.1%} vs {und_stats['win_rate']:.1%})")
        
        return analysis
    
    def get_todays_games(self):
        """Get today's games"""
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            games_today = []
            if data['dates'] and len(data['dates']) > 0:
                games = data['dates'][0]['games']
                
                for game in games:
                    if game['status']['statusCode'] in ['S', 'P']:
                        game_info = {
                            'away_team': game['teams']['away']['team']['name'],
                            'home_team': game['teams']['home']['team']['name'],
                            'game_time': game['gameDate']
                        }
                        games_today.append(game_info)
            
            return games_today
            
        except Exception as e:
            print(f"Error getting today's games: {e}")
            return []
    
    def generate_daily_runline_predictions(self):
        """Generate run line predictions for today's games"""
        print(f"\n{'='*60}")
        print(f"FIXED MLB RUN LINE PREDICTIONS - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        todays_games = self.get_todays_games()
        
        if not todays_games:
            print("No games scheduled for today.")
            return []
        
        predictions = []
        good_bets = []
        
        for game in todays_games:
            home_team = game['home_team']
            away_team = game['away_team']
            
            prediction = self.predict_runline_simple(home_team, away_team)
            
            print(f"\n{away_team} @ {home_team}")
            
            if prediction['confidence'] == 'LOW':
                print(f"  No strong prediction available")
                print(f"  Reason: Close matchup or insufficient data")
                continue
            
            print(f"  Predicted Favorite: {prediction['favorite_team']}")
            print(f"  Run Line Prediction: {prediction['prediction']}")
            print(f"  Favorite Covers (-1.5): {prediction['favorite_cover_probability']:.1%}")
            print(f"  Underdog Covers (+1.5): {prediction['underdog_cover_probability']:.1%}")
            print(f"  Confidence: {prediction['confidence']}")
            
            # Show key stats
            fav_team = prediction['favorite_team']
            und_team = prediction['underdog_team']
            
            if fav_team == home_team:
                fav_stats = prediction['home_stats']
                und_stats = prediction['away_stats']
            else:
                fav_stats = prediction['away_stats']
                und_stats = prediction['home_stats']
            
            print(f"  Key Stats:")
            print(f"    {fav_team} (fav): {fav_stats['win_rate']:.1%} wins, {fav_stats['blowout_rate']:.1%} blowouts")
            print(f"    {und_team} (dog): {und_stats['win_rate']:.1%} wins, {und_stats['close_loss_rate']:.1%} close losses")
            print(f"    Run diff advantage: {fav_stats['run_differential'] - und_stats['run_differential']:+.1f}")
            
            # Analysis
            if prediction['analysis']:
                print(f"  Analysis:")
                for point in prediction['analysis']:
                    print(f"    • {point}")
            
            if prediction['confidence'] in ['HIGH', 'MEDIUM']:
                print(f"  📊 RECOMMENDATION: {prediction['prediction']}")
                good_bets.append(prediction)
            
            # Save prediction
            prediction_record = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'away_team': away_team,
                'home_team': home_team,
                'favorite_team': prediction['favorite_team'],
                'runline_prediction': prediction['prediction'],
                'favorite_cover_probability': prediction['favorite_cover_probability'],
                'confidence': prediction['confidence']
            }
            predictions.append(prediction_record)
        
        # Summary
        if good_bets:
            print(f"\n🎯 FIXED RUN LINE SUMMARY:")
            high_conf = len([p for p in good_bets if p['confidence'] == 'HIGH'])
            medium_conf = len([p for p in good_bets if p['confidence'] == 'MEDIUM'])
            
            print(f"  Total games analyzed: {len(predictions)}")
            print(f"  Strong recommendations: {len(good_bets)}")
            print(f"  High confidence: {high_conf}")
            print(f"  Medium confidence: {medium_conf}")
            
            print(f"\n📊 TOP PICKS:")
            sorted_bets = sorted(good_bets, key=lambda x: abs(x['favorite_cover_probability'] - 0.5), reverse=True)
            for bet in sorted_bets[:5]:
                prob = bet['favorite_cover_probability']
                if prob > 0.5:
                    print(f"    {bet['favorite_team']} -1.5 ({prob:.1%} confidence)")
                else:
                    print(f"    {bet['underdog_team']} +1.5 ({1-prob:.1%} confidence)")
            
            # Save predictions
            filename = f"fixed_runline_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
            pd.DataFrame(predictions).to_csv(filename, index=False)
            print(f"  Saved to: {filename}")
        else:
            print(f"\n⚠️  No strong run line recommendations today")
        
        return predictions

def main():
    print("Fixed MLB Run Line Betting System")
    print("Logical approach to -1.5/+1.5 predictions")
    print("="*50)
    
    system = FixedMLBRunLineSystem()
    
    # Load data
    if not system.load_historical_data():
        return
    
    # Generate predictions
    predictions = system.generate_daily_runline_predictions()

if __name__ == "__main__":
    main()