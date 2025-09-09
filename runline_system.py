import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta
import time
import os

class MLBRunLineSystem:
    def __init__(self):
        self.model = None
        self.historical_data = None
        self.min_games_for_prediction = 10
        
    def load_historical_data(self):
        """Load historical games data (same as moneyline system)"""
        try:
            self.historical_data = pd.read_csv('historical_mlb_games.csv')
            self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
            self.historical_data = self.historical_data.sort_values('date').reset_index(drop=True)
            print(f"Loaded {len(self.historical_data)} historical games for run line analysis")
            return True
        except:
            print("Error: Could not load historical_mlb_games.csv")
            print("Run your moneyline system first to create the data file")
            return False
    
    def calculate_game_margin(self, game):
        """Calculate margin of victory for a game"""
        return abs(game['home_score'] - game['away_score'])
    
    def calculate_runline_features(self, team, as_of_date, is_home=True):
        """Calculate run line specific features for a team"""
        
        # Get team's recent games (last 15 for volatility analysis)
        team_games = self.historical_data[
            ((self.historical_data['home_team'] == team) | 
             (self.historical_data['away_team'] == team)) &
            (self.historical_data['date'] < as_of_date)
        ].tail(15)
        
        if len(team_games) < self.min_games_for_prediction:
            return self.get_default_runline_stats()
        
        # Analyze margins and patterns
        wins = []
        losses = []
        all_margins = []
        total_runs_scored = 0
        total_runs_allowed = 0
        big_offensive_games = 0  # 7+ runs
        shutout_games = 0  # 0-2 runs scored
        
        for _, game in team_games.iterrows():
            is_team_home = (game['home_team'] == team)
            team_score = game['home_score'] if is_team_home else game['away_score']
            opp_score = game['away_score'] if is_team_home else game['home_score']
            
            margin = abs(team_score - opp_score)
            all_margins.append(margin)
            total_runs_scored += team_score
            total_runs_allowed += opp_score
            
            # Big offensive game
            if team_score >= 7:
                big_offensive_games += 1
            
            # Low scoring game
            if team_score <= 2:
                shutout_games += 1
            
            # Win/loss margin analysis
            team_won = (game['winner'] == 'home' and is_team_home) or (game['winner'] == 'away' and not is_team_home)
            
            if team_won:
                wins.append(margin)
            else:
                losses.append(margin)
        
        # Calculate key run line metrics
        total_games = len(team_games)
        
        # Blowout rates (2+ run margins)
        blowout_wins = sum(1 for m in wins if m >= 2)
        blowout_win_rate = blowout_wins / max(len(wins), 1)
        
        close_losses = sum(1 for m in losses if m == 1)
        close_loss_rate = close_losses / max(len(losses), 1)
        
        # Overall margin tendencies
        avg_margin_when_winning = np.mean(wins) if wins else 1.0
        avg_margin_when_losing = np.mean(losses) if losses else 1.0
        
        # Offensive patterns
        big_game_rate = big_offensive_games / total_games
        low_scoring_rate = shutout_games / total_games
        avg_runs_scored = total_runs_scored / total_games
        avg_runs_allowed = total_runs_allowed / total_games
        
        # Volatility (standard deviation of margins)
        margin_volatility = np.std(all_margins) if len(all_margins) > 1 else 1.0
        
        # Run differential
        run_differential = avg_runs_scored - avg_runs_allowed
        
        return {
            'blowout_win_rate': blowout_win_rate,
            'close_loss_rate': close_loss_rate,
            'avg_margin_winning': avg_margin_when_winning,
            'avg_margin_losing': avg_margin_when_losing,
            'big_game_rate': big_game_rate,
            'low_scoring_rate': low_scoring_rate,
            'avg_runs_scored': avg_runs_scored,
            'avg_runs_allowed': avg_runs_allowed,
            'margin_volatility': margin_volatility,
            'run_differential': run_differential,
            'games_played': total_games,
            'win_rate': len(wins) / total_games
        }
    
    def get_default_runline_stats(self):
        """Return default stats when insufficient data"""
        return {
            'blowout_win_rate': 0.4,
            'close_loss_rate': 0.5,
            'avg_margin_winning': 2.5,
            'avg_margin_losing': 2.0,
            'big_game_rate': 0.25,
            'low_scoring_rate': 0.15,
            'avg_runs_scored': 4.5,
            'avg_runs_allowed': 4.5,
            'margin_volatility': 2.0,
            'run_differential': 0.0,
            'games_played': 0,
            'win_rate': 0.5
        }
    
    def train_runline_model(self):
        """Train model specifically for run line predictions"""
        print("\n=== TRAINING RUN LINE MODEL ===")
        print("Training model to predict if favorite covers -1.5 run line...")
        
        # Use recent data for training
        recent_data = self.historical_data.tail(2000)
        
        features = []
        labels = []
        
        for idx, game in recent_data.iterrows():
            if idx < 200:  # Need some history
                continue
            
            game_date = game['date']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Calculate run line features
            home_stats = self.calculate_runline_features(home_team, game_date, True)
            away_stats = self.calculate_runline_features(away_team, game_date, False)
            
            # Skip if insufficient data
            if home_stats['games_played'] < 5 or away_stats['games_played'] < 5:
                continue
            
            # Determine actual margin (for labeling)
            actual_margin = abs(game['home_score'] - game['away_score'])
            home_won = (game['winner'] == 'home')
            
            # Create feature vector focused on margin prediction
            feature_vector = [
                # Home team margin tendencies
                home_stats['blowout_win_rate'],
                home_stats['avg_margin_winning'],
                home_stats['big_game_rate'],
                home_stats['margin_volatility'],
                home_stats['run_differential'],
                
                # Away team margin tendencies  
                away_stats['close_loss_rate'],
                away_stats['avg_margin_losing'],
                away_stats['low_scoring_rate'],
                away_stats['margin_volatility'],
                away_stats['run_differential'],
                
                # Relative advantages
                home_stats['run_differential'] - away_stats['run_differential'],
                home_stats['blowout_win_rate'] - away_stats['close_loss_rate'],
                home_stats['big_game_rate'] - away_stats['low_scoring_rate'],
                
                # Basic team strength
                home_stats['win_rate'] - away_stats['win_rate'],
                1  # home_field_advantage
            ]
            
            # Label: 1 if home team covers -1.5 (wins by 2+), 0 otherwise
            home_covers_runline = (home_won and actual_margin >= 2)
            
            features.append(feature_vector)
            labels.append(1 if home_covers_runline else 0)
        
        print(f"Training on {len(features)} games with run line data")
        
        if len(features) < 100:
            print("Not enough data for run line model")
            return False
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=8, 
            random_state=42,
            min_samples_leaf=5  # Prevent overfitting on margin data
        )
        self.model.fit(features, labels)
        
        # Check accuracy
        predictions = self.model.predict(features)
        accuracy = (predictions == labels).mean()
        print(f"Training accuracy: {accuracy:.1%}")
        
        # Feature importance
        feature_names = [
            'home_blowout_rate', 'home_avg_margin_win', 'home_big_games', 'home_volatility', 'home_run_diff',
            'away_close_loss_rate', 'away_avg_margin_loss', 'away_low_scoring', 'away_volatility', 'away_run_diff',
            'run_diff_advantage', 'blowout_vs_close', 'offensive_advantage', 'win_rate_advantage', 'home_field'
        ]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nMost Important Run Line Features:")
        for _, row in importance_df.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Save model
        joblib.dump(self.model, 'runline_model.pkl')
        print("✅ Run line model saved")
        
        return True
    
    def determine_favorite(self, home_stats, away_stats):
        """Determine which team should be the favorite based on team strength"""
        
        # Calculate overall team strength score
        home_strength = (
            home_stats['win_rate'] * 0.4 +
            (home_stats['run_differential'] + 2) / 6 * 0.4 +  # Normalize run diff
            home_stats['blowout_win_rate'] * 0.2
        )
        
        away_strength = (
            away_stats['win_rate'] * 0.4 +
            (away_stats['run_differential'] + 2) / 6 * 0.4 +  # Normalize run diff  
            away_stats['blowout_win_rate'] * 0.2
        )
        
        # Add small home field advantage (about 0.03)
        home_strength += 0.03
        
        return 'HOME' if home_strength > away_strength else 'AWAY'
    
    def predict_runline(self, home_team, away_team):
        """Predict run line outcome"""
        if self.model is None:
            try:
                self.model = joblib.load('runline_model.pkl')
                print("📂 Loaded existing run line model")
            except:
                print("🔄 Training new run line model...")
                if not self.train_runline_model():
                    return None
        
        current_date = datetime.now()
        
        # Get team statistics
        home_stats = self.calculate_runline_features(home_team, current_date, True)
        away_stats = self.calculate_runline_features(away_team, current_date, False)
        
        # Check data sufficiency
        if home_stats['games_played'] < 5 or away_stats['games_played'] < 5:
            return {
                'favorite_covers_probability': 0.45,
                'underdog_covers_probability': 0.55,
                'prediction': 'Insufficient data',
                'confidence': 'LOW',
                'reason': 'Insufficient recent data',
                'home_stats': home_stats,
                'away_stats': away_stats,
                'analysis': ['Insufficient recent data for reliable prediction'],
                'favorite_team': 'Unknown',
                'underdog_team': 'Unknown'
            }
        
        # Determine who should be the favorite
        predicted_favorite = self.determine_favorite(home_stats, away_stats)
        
        if predicted_favorite == 'HOME':
            favorite_team = home_team
            underdog_team = away_team
            favorite_stats = home_stats
            underdog_stats = away_stats
        else:
            favorite_team = away_team
            underdog_team = home_team
            favorite_stats = away_stats
            underdog_stats = home_stats
        
        # Create feature vector with favorite as the "strong" team
        if predicted_favorite == 'HOME':
            # Home team is favorite - use original order
            features = [[
                home_stats['blowout_win_rate'],
                home_stats['avg_margin_winning'],
                home_stats['big_game_rate'],
                home_stats['margin_volatility'],
                home_stats['run_differential'],
                
                away_stats['close_loss_rate'],
                away_stats['avg_margin_losing'],
                away_stats['low_scoring_rate'],
                away_stats['margin_volatility'],
                away_stats['run_differential'],
                
                home_stats['run_differential'] - away_stats['run_differential'],
                home_stats['blowout_win_rate'] - away_stats['close_loss_rate'],
                home_stats['big_game_rate'] - away_stats['low_scoring_rate'],
                home_stats['win_rate'] - away_stats['win_rate'],
                1  # home field
            ]]
            
            favorite_covers_prob = self.model.predict_proba(features)[0][1]
            
        else:
            # Away team is favorite - flip the features
            features = [[
                away_stats['blowout_win_rate'],
                away_stats['avg_margin_winning'],  
                away_stats['big_game_rate'],
                away_stats['margin_volatility'],
                away_stats['run_differential'],
                
                home_stats['close_loss_rate'],
                home_stats['avg_margin_losing'],
                home_stats['low_scoring_rate'],
                home_stats['margin_volatility'],
                home_stats['run_differential'],
                
                away_stats['run_differential'] - home_stats['run_differential'],
                away_stats['blowout_win_rate'] - home_stats['close_loss_rate'],
                away_stats['big_game_rate'] - home_stats['low_scoring_rate'],
                away_stats['win_rate'] - home_stats['win_rate'],
                0  # away team (no home field)
            ]]
            
            favorite_covers_prob = self.model.predict_proba(features)[0][1]
        
        # Determine recommendation
        if favorite_covers_prob > 0.5:
            prediction = f"{favorite_team} -1.5"
            recommendation = f"{favorite_team} -1.5"
        else:
            prediction = f"{underdog_team} +1.5"
            recommendation = f"{underdog_team} +1.5"
        
        # Confidence levels
        if favorite_covers_prob > 0.6 or favorite_covers_prob < 0.4:
            confidence = 'HIGH'
        elif favorite_covers_prob > 0.55 or favorite_covers_prob < 0.45:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'favorite_covers_probability': favorite_covers_prob,
            'underdog_covers_probability': 1 - favorite_covers_prob,
            'prediction': recommendation,
            'confidence': confidence,
            'home_stats': home_stats,
            'away_stats': away_stats,
            'analysis': self.generate_runline_analysis_v2(favorite_stats, underdog_stats, favorite_team, underdog_team),
            'favorite_team': favorite_team,
            'underdog_team': underdog_team
        }
    
    def generate_runline_analysis_v2(self, favorite_stats, underdog_stats, favorite_team, underdog_team):
        """Generate analysis explaining the run line prediction"""
        analysis = []
        
        # Favorite blowout tendency
        if favorite_stats['blowout_win_rate'] > 0.5:
            analysis.append(f"{favorite_team} wins by 2+ runs {favorite_stats['blowout_win_rate']:.1%} of the time")
        
        # Underdog close game tendency
        if underdog_stats['close_loss_rate'] > 0.6:
            analysis.append(f"{underdog_team} loses close games {underdog_stats['close_loss_rate']:.1%} of the time")
        
        # Offensive power
        if favorite_stats['big_game_rate'] > 0.3:
            analysis.append(f"{favorite_team} scores 7+ runs {favorite_stats['big_game_rate']:.1%} of games")
        
        # Run differential
        run_diff = favorite_stats['run_differential'] - underdog_stats['run_differential']
        if abs(run_diff) > 1.0:
            analysis.append(f"Run differential advantage: {run_diff:+.1f} runs/game")
        
        return analysis
    
    def get_todays_games(self):
        """Get today's games (same as moneyline system)"""
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
        print(f"MLB RUN LINE PREDICTIONS - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        todays_games = self.get_todays_games()
        
        if not todays_games:
            print("No games scheduled for today.")
            return []
        
        predictions = []
        
        for game in todays_games:
            home_team = game['home_team']
            away_team = game['away_team']
            
            prediction = self.predict_runline(home_team, away_team)
            
            if prediction is None:
                continue
            
            print(f"\n{away_team} @ {home_team}")
            print(f"  Predicted Favorite: {prediction['favorite_team']}")
            print(f"  Run Line Prediction: {prediction['prediction']}")
            print(f"  Favorite Covers (-1.5) Probability: {prediction['favorite_covers_probability']:.1%}")
            print(f"  Underdog Covers (+1.5) Probability: {prediction['underdog_covers_probability']:.1%}")
            print(f"  Confidence: {prediction['confidence']}")
            
            # Show key stats
            if 'home_stats' in prediction and 'away_stats' in prediction:
                favorite_team = prediction['favorite_team']
                underdog_team = prediction['underdog_team']
                
                if favorite_team == home_team:
                    fav_stats = prediction['home_stats']
                    und_stats = prediction['away_stats']
                else:
                    fav_stats = prediction['away_stats']
                    und_stats = prediction['home_stats']
                
                print(f"  Key Factors:")
                print(f"    {favorite_team} (fav) blowout rate: {fav_stats['blowout_win_rate']:.1%}")
                print(f"    {underdog_team} (dog) close loss rate: {und_stats['close_loss_rate']:.1%}")
                print(f"    Run diff advantage: {fav_stats['run_differential'] - und_stats['run_differential']:+.1f}")
            
            # Analysis
            if 'analysis' in prediction and prediction['analysis']:
                print(f"  Analysis:")
                for point in prediction['analysis']:
                    print(f"    • {point}")
            
            if prediction['confidence'] in ['HIGH', 'MEDIUM']:
                print(f"  📊 BETTING RECOMMENDATION: Consider this run line bet")
            else:
                print(f"  ⚠️  Skip - Low confidence")
            
            # Save prediction
            prediction_record = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'away_team': away_team,
                'home_team': home_team,
                'favorite_team': prediction.get('favorite_team', 'Unknown'),
                'runline_prediction': prediction['prediction'],
                'favorite_covers_probability': prediction.get('favorite_covers_probability', 0.5),
                'confidence': prediction['confidence']
            }
            predictions.append(prediction_record)
        
        # Save predictions
        if predictions:
            filename = f"runline_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
            pd.DataFrame(predictions).to_csv(filename, index=False)
            
            print(f"\n🎯 RUN LINE SUMMARY:")
            high_conf = len([p for p in predictions if p['confidence'] == 'HIGH'])
            medium_conf = len([p for p in predictions if p['confidence'] == 'MEDIUM'])
            
            print(f"  Total games: {len(predictions)}")
            print(f"  High confidence: {high_conf}")
            print(f"  Medium confidence: {medium_conf}")
            print(f"  Recommended bets: {high_conf + medium_conf}")
            print(f"  Predictions saved to: {filename}")
        
        return predictions

def main():
    print("MLB Run Line Betting System")
    print("Predicts if favorites cover -1.5 run line")
    print("="*50)
    
    system = MLBRunLineSystem()
    
    # Load data
    if not system.load_historical_data():
        return
    
    choice = input("\nWhat would you like to do?\n1. Generate today's run line predictions\n2. Train run line model\n3. Both\nChoice (1-3): ")
    
    if choice in ['2', '3']:
        system.train_runline_model()
    
    if choice in ['1', '3']:
        predictions = system.generate_daily_runline_predictions()

if __name__ == "__main__":
    main()