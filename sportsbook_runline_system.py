import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SportsbookRunLineSystem:
    def __init__(self):
        self.historical_data = None
        # Today's actual sportsbook lines
        self.todays_lines = {
            ('St. Louis Cardinals', 'Cincinnati Reds'): ('Cincinnati Reds', -1.5),
            ('Tampa Bay Rays', 'Washington Nationals'): ('Tampa Bay Rays', -1.5),
            ('Atlanta Braves', 'Philadelphia Phillies'): ('Philadelphia Phillies', -1.5),
            ('Milwaukee Brewers', 'Toronto Blue Jays'): ('Toronto Blue Jays', -1.5),
            ('Pittsburgh Pirates', 'Boston Red Sox'): ('Pittsburgh Pirates', -1.5),
            ('Seattle Mariners', 'Cleveland Guardians'): ('Seattle Mariners', -1.5),
            ('Miami Marlins', 'New York Mets'): ('New York Mets', -1.5),
            ('New York Yankees', 'Chicago White Sox'): ('New York Yankees', -1.5),
            ('Los Angeles Angels', 'Houston Astros'): ('Houston Astros', -1.5),
            ('Detroit Tigers', 'Kansas City Royals'): ('Kansas City Royals', -1.5),
            ('San Diego Padres', 'Minnesota Twins'): ('San Diego Padres', -1.5),
            ('Chicago Cubs', 'Colorado Rockies'): ('Chicago Cubs', -1.5),
            ('Texas Rangers', 'Oakland Athletics'): ('Oakland Athletics', -1.5),
            ('Arizona Diamondbacks', 'Los Angeles Dodgers'): ('Los Angeles Dodgers', -1.5),
            ('Baltimore Orioles', 'San Francisco Giants'): ('San Francisco Giants', -1.5)
        }
        
    def load_historical_data(self):
        """Load historical games data"""
        try:
            self.historical_data = pd.read_csv('historical_mlb_games.csv')
            self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
            self.historical_data = self.historical_data.sort_values('date').reset_index(drop=True)
            print(f"Loaded {len(self.historical_data)} historical games")
            return True
        except:
            print("Error: Could not load historical_mlb_games.csv")
            return False
    
    def calculate_team_cover_stats(self, team, as_of_date, games_back=15):
        """Calculate how often a team covers run lines"""
        
        team_games = self.historical_data[
            ((self.historical_data['home_team'] == team) | 
             (self.historical_data['away_team'] == team)) &
            (self.historical_data['date'] < as_of_date)
        ].tail(games_back)
        
        if len(team_games) < 10:
            return {
                'total_games': 0,
                'wins': 0,
                'win_rate': 0.5,
                'avg_runs_scored': 4.5,
                'avg_runs_allowed': 4.5,
                'run_differential': 0.0,
                'blowout_wins': 0,
                'blowout_rate': 0.4,
                'close_losses': 0,
                'close_loss_rate': 0.5
            }
        
        wins = 0
        total_runs_scored = 0
        total_runs_allowed = 0
        blowout_wins = 0  # Wins by 2+ runs
        close_losses = 0  # Losses by 1 run
        losses = 0
        
        for _, game in team_games.iterrows():
            is_home = (game['home_team'] == team)
            team_score = game['home_score'] if is_home else game['away_score']
            opp_score = game['away_score'] if is_home else game['home_score']
            
            total_runs_scored += team_score
            total_runs_allowed += opp_score
            
            margin = team_score - opp_score  # Positive if team won
            
            if margin > 0:  # Team won
                wins += 1
                if margin >= 2:
                    blowout_wins += 1
            else:  # Team lost
                losses += 1
                if abs(margin) == 1:  # Lost by exactly 1
                    close_losses += 1
        
        total_games = len(team_games)
        win_rate = wins / total_games
        avg_runs_scored = total_runs_scored / total_games
        avg_runs_allowed = total_runs_allowed / total_games
        run_differential = avg_runs_scored - avg_runs_allowed
        
        blowout_rate = blowout_wins / max(wins, 1)  # Of wins, what % are blowouts?
        close_loss_rate = close_losses / max(losses, 1)  # Of losses, what % are close?
        
        return {
            'total_games': total_games,
            'wins': wins,
            'win_rate': win_rate,
            'avg_runs_scored': avg_runs_scored,
            'avg_runs_allowed': avg_runs_allowed,
            'run_differential': run_differential,
            'blowout_wins': blowout_wins,
            'blowout_rate': blowout_rate,
            'close_losses': close_losses,
            'close_loss_rate': close_loss_rate
        }
    
    def predict_favorite_covers(self, favorite_team, underdog_team):
        """Predict if the sportsbook favorite covers -1.5"""
        
        current_date = datetime.now()
        
        # Get stats for both teams
        fav_stats = self.calculate_team_cover_stats(favorite_team, current_date)
        und_stats = self.calculate_team_cover_stats(underdog_team, current_date)
        
        if fav_stats['total_games'] < 10 or und_stats['total_games'] < 10:
            return {
                'prediction': 'Insufficient data',
                'confidence': 'LOW',
                'cover_probability': 0.5
            }
        
        # Base probability that favorite covers -1.5
        base_prob = 0.45  # Historically, run line favorites cover ~45% of time
        
        # Adjustments
        adjustments = []
        
        # 1. Favorite's blowout tendency
        blowout_adj = (fav_stats['blowout_rate'] - 0.45) * 0.25
        adjustments.append(('Favorite blowout rate', blowout_adj))
        
        # 2. Underdog's close game tendency
        close_adj = (und_stats['close_loss_rate'] - 0.45) * 0.15
        adjustments.append(('Underdog close games', -close_adj))  # More close losses = worse for favorite
        
        # 3. Run differential gap
        run_diff_gap = fav_stats['run_differential'] - und_stats['run_differential']
        run_diff_adj = min(run_diff_gap * 0.06, 0.15)  # Cap at 15%
        adjustments.append(('Run differential gap', run_diff_adj))
        
        # 4. Win rate difference
        win_rate_diff = fav_stats['win_rate'] - und_stats['win_rate']
        win_rate_adj = min(win_rate_diff * 0.20, 0.10)  # Cap at 10%
        adjustments.append(('Win rate difference', win_rate_adj))
        
        # 5. Team quality check (avoid bad favorites)
        if fav_stats['win_rate'] < 0.45:  # Bad team favored (probably due to pitcher)
            quality_adj = -0.10
            adjustments.append(('Poor favorite quality', quality_adj))
        elif fav_stats['win_rate'] > 0.60:  # Great team favored
            quality_adj = 0.05
            adjustments.append(('Strong favorite quality', quality_adj))
        else:
            quality_adj = 0
        
        # Calculate final probability
        total_adjustment = sum(adj[1] for adj in adjustments)
        cover_prob = base_prob + total_adjustment
        cover_prob = max(0.20, min(0.80, cover_prob))  # Keep between 20-80%
        
        # Determine recommendation
        if cover_prob >= 0.58:
            prediction = f"{favorite_team} -1.5"
            confidence = "HIGH" if cover_prob >= 0.65 else "MEDIUM"
        elif cover_prob <= 0.42:
            prediction = f"{underdog_team} +1.5"
            confidence = "HIGH" if cover_prob <= 0.35 else "MEDIUM"
        else:
            prediction = "No strong lean"
            confidence = "LOW"
        
        return {
            'favorite_team': favorite_team,
            'underdog_team': underdog_team,
            'cover_probability': cover_prob,
            'prediction': prediction,
            'confidence': confidence,
            'adjustments': adjustments,
            'fav_stats': fav_stats,
            'und_stats': und_stats
        }
    
    def generate_analysis(self, fav_stats, und_stats, favorite_team, underdog_team):
        """Generate readable analysis"""
        analysis = []
        
        if fav_stats['blowout_rate'] > 0.55:
            analysis.append(f"{favorite_team} wins by 2+ runs {fav_stats['blowout_rate']:.1%} of the time")
        elif fav_stats['blowout_rate'] < 0.35:
            analysis.append(f"{favorite_team} rarely blows out opponents ({fav_stats['blowout_rate']:.1%})")
        
        if und_stats['close_loss_rate'] > 0.55:
            analysis.append(f"{underdog_team} loses close games {und_stats['close_loss_rate']:.1%} of the time")
        
        run_diff_gap = fav_stats['run_differential'] - und_stats['run_differential']
        if run_diff_gap > 1.5:
            analysis.append(f"{favorite_team} has strong run differential advantage (+{run_diff_gap:.1f})")
        elif run_diff_gap < 0.5:
            analysis.append(f"Small talent gap between teams (run diff: +{run_diff_gap:.1f})")
        
        win_rate_diff = fav_stats['win_rate'] - und_stats['win_rate']
        if win_rate_diff < 0:
            analysis.append(f"⚠️ {underdog_team} actually has better record ({und_stats['win_rate']:.1%} vs {fav_stats['win_rate']:.1%})")
        
        return analysis
    
    def generate_recommendations(self):
        """Generate recommendations for all games with sportsbook lines"""
        print(f"\n{'='*70}")
        print(f"MLB RUN LINE PREDICTIONS - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Using ACTUAL Sportsbook Lines")
        print(f"{'='*70}")
        
        recommendations = []
        strong_bets = []
        
        for (away_team, home_team), (favorite, spread) in self.todays_lines.items():
            underdog = home_team if favorite == away_team else away_team
            
            prediction = self.predict_favorite_covers(favorite, underdog)
            
            print(f"\n{away_team} @ {home_team}")
            print(f"  Sportsbook Favorite: {favorite} -1.5")
            print(f"  Sportsbook Underdog: {underdog} +1.5")
            
            if prediction['confidence'] == 'LOW':
                print(f"  ⚠️ No strong prediction - Skip this game")
                continue
            
            print(f"  Prediction: {prediction['prediction']}")
            print(f"  Favorite covers probability: {prediction['cover_probability']:.1%}")
            print(f"  Confidence: {prediction['confidence']}")
            
            # Show key stats
            fav_stats = prediction['fav_stats']
            und_stats = prediction['und_stats']
            
            print(f"  Key Stats:")
            print(f"    {favorite}: {fav_stats['win_rate']:.1%} win rate, {fav_stats['blowout_rate']:.1%} blowouts, {fav_stats['run_differential']:+.1f} run diff")
            print(f"    {underdog}: {und_stats['win_rate']:.1%} win rate, {und_stats['close_loss_rate']:.1%} close losses, {und_stats['run_differential']:+.1f} run diff")
            
            # Analysis
            analysis = self.generate_analysis(fav_stats, und_stats, favorite, underdog)
            if analysis:
                print(f"  Analysis:")
                for point in analysis:
                    print(f"    • {point}")
            
            if prediction['confidence'] in ['HIGH', 'MEDIUM']:
                print(f"  ✅ RECOMMENDATION: {prediction['prediction']}")
                strong_bets.append({
                    'game': f"{away_team} @ {home_team}",
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'probability': prediction['cover_probability']
                })
            
            recommendations.append(prediction)
        
        # Summary
        print(f"\n{'='*70}")
        print(f"BETTING SUMMARY")
        print(f"{'='*70}")
        
        if strong_bets:
            high_conf = [bet for bet in strong_bets if bet['confidence'] == 'HIGH']
            medium_conf = [bet for bet in strong_bets if bet['confidence'] == 'MEDIUM']
            
            print(f"Total games analyzed: {len(recommendations)}")
            print(f"Strong recommendations: {len(strong_bets)}")
            print(f"High confidence: {len(high_conf)}")
            print(f"Medium confidence: {len(medium_conf)}")
            
            print(f"\n🎯 TOP PICKS:")
            sorted_bets = sorted(strong_bets, 
                               key=lambda x: abs(x['probability'] - 0.5), 
                               reverse=True)
            
            for i, bet in enumerate(sorted_bets[:8], 1):
                conf_emoji = "🔥" if bet['confidence'] == 'HIGH' else "📊"
                print(f"  {i}. {conf_emoji} {bet['prediction']} ({bet['confidence']}) - {bet['game']}")
        else:
            print(f"❌ No strong recommendations found today")
        
        return recommendations

def main():
    print("Sportsbook-Based MLB Run Line System")
    print("Uses ACTUAL betting lines from your sportsbook")
    print("="*55)
    
    system = SportsbookRunLineSystem()
    
    # Load data
    if not system.load_historical_data():
        return
    
    # Generate predictions
    recommendations = system.generate_recommendations()

if __name__ == "__main__":
    main() 