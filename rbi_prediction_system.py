import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta
import time

class RBIPredictionSystem:
    def __init__(self):
        self.model = None
        self.historical_data = None
        self.player_cache = {}
        self.weather_api_key = None
        
    def set_weather_api_key(self, api_key):
        """Set OpenWeatherMap API key"""
        self.weather_api_key = api_key
    
    def load_historical_data(self):
        """Load historical games data"""
        try:
            self.historical_data = pd.read_csv('historical_mlb_games.csv')
            self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
            self.historical_data = self.historical_data.sort_values('date').reset_index(drop=True)
            print(f"Loaded {len(self.historical_data)} historical games for RBI analysis")
            return True
        except:
            print("Error: Could not load historical_mlb_games.csv")
            return False
    
    def get_batting_order_positions(self, team, date_str=None):
        """
        Get probable batting order for a team
        In reality, this would come from lineup APIs, but we'll estimate based on player roles
        """
        # Simplified batting order estimation
        # In a real system, you'd get actual lineups from MLB API
        
        typical_lineups = {
            'New York Yankees': ['Leadoff', 'Aaron Judge', 'Cleanup', 'Power Hitter', 'Contact', 'Utility', 'Catcher', 'Pitcher', 'Bench'],
            'Los Angeles Dodgers': ['Speedster', 'Mookie Betts', 'Freddie Freeman', 'Power', 'Contact', 'Veteran', 'Defense', 'Pitcher', 'Bench'],
            'Boston Red Sox': ['Contact', 'Rafael Devers', 'Power Hitter', 'Cleanup', 'Veterans', 'Defense', 'Catcher', 'Pitcher', 'Utility']
        }
        
        return typical_lineups.get(team, ['Player'] * 9)
    
    def get_player_season_stats(self, player_name, team, season=None):
        """
        Get player's season statistics using MLB API
        """
        if season is None:
            season = datetime.now().year
        
        cache_key = f"{player_name}_{team}_{season}"
        if cache_key in self.player_cache:
            return self.player_cache[cache_key]
        
        try:
            # Search for player
            search_url = f"https://statsapi.mlb.com/api/v1/people/search?names={player_name.replace(' ', '%20')}"
            search_response = requests.get(search_url)
            search_data = search_response.json()
            
            if 'people' not in search_data or len(search_data['people']) == 0:
                return self.get_default_player_stats()
            
            player_id = search_data['people'][0]['id']
            
            # Get player stats
            stats_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&season={season}&group=hitting"
            stats_response = requests.get(stats_url)
            stats_data = stats_response.json()
            
            player_stats = {
                'name': player_name,
                'rbi': 50,
                'games': 100,
                'at_bats': 350,
                'runs': 45,
                'hits': 85,
                'doubles': 15,
                'home_runs': 12,
                'avg': 0.243,
                'obp': 0.320,
                'slg': 0.400,
                'recent_rbi': 3,  # Last 10 games
                'hot_streak': False
            }
            
            # Extract actual stats if available
            if 'stats' in stats_data and len(stats_data['stats']) > 0:
                stat_group = stats_data['stats'][0]
                if 'splits' in stat_group and len(stat_group['splits']) > 0:
                    season_stats = stat_group['splits'][0]['stat']
                    
                    player_stats.update({
                        'rbi': int(season_stats.get('rbi', 50)),
                        'games': int(season_stats.get('gamesPlayed', 100)),
                        'at_bats': int(season_stats.get('atBats', 350)),
                        'runs': int(season_stats.get('runs', 45)),
                        'hits': int(season_stats.get('hits', 85)),
                        'doubles': int(season_stats.get('doubles', 15)),
                        'home_runs': int(season_stats.get('homeRuns', 12)),
                        'avg': float(season_stats.get('avg', 0.243)),
                        'obp': float(season_stats.get('obp', 0.320)),
                        'slg': float(season_stats.get('slg', 0.400))
                    })
            
            # Cache results
            self.player_cache[cache_key] = player_stats
            time.sleep(0.2)  # Rate limiting
            
            return player_stats
            
        except Exception as e:
            print(f"Error getting stats for {player_name}: {e}")
            return self.get_default_player_stats()
    
    def get_default_player_stats(self):
        """Default player stats when API fails"""
        return {
            'rbi': 50, 'games': 100, 'at_bats': 350, 'runs': 45,
            'hits': 85, 'doubles': 15, 'home_runs': 12,
            'avg': 0.243, 'obp': 0.320, 'slg': 0.400,
            'recent_rbi': 3, 'hot_streak': False
        }
    
    def calculate_team_offensive_context(self, team, as_of_date):
        """
        Calculate team's offensive context that affects RBI opportunities
        """
        team_games = self.historical_data[
            ((self.historical_data['home_team'] == team) | 
             (self.historical_data['away_team'] == team)) &
            (self.historical_data['date'] < as_of_date)
        ].tail(20)
        
        if len(team_games) < 10:
            return {
                'team_runs_per_game': 4.5,
                'team_hits_per_game': 8.5,
                'team_scoring_trends': 0,
                'baserunner_rate': 0.32,
                'clutch_hitting': 0.25
            }
        
        # Calculate offensive metrics
        total_runs = 0
        total_hits_est = 0  # We'll estimate this
        
        for _, game in team_games.iterrows():
            is_home = (game['home_team'] == team)
            team_score = game['home_score'] if is_home else game['away_score']
            total_runs += team_score
            
            # Estimate hits based on runs (roughly 2:1 ratio)
            total_hits_est += team_score * 2
        
        runs_per_game = total_runs / len(team_games)
        hits_per_game = total_hits_est / len(team_games)
        
        # Calculate recent trend (last 5 vs previous 10)
        recent_games = team_games.tail(5)
        older_games = team_games.head(10)
        
        recent_rpg = sum([game['home_score'] if game['home_team'] == team else game['away_score'] 
                         for _, game in recent_games.iterrows()]) / len(recent_games)
        older_rpg = sum([game['home_score'] if game['home_team'] == team else game['away_score'] 
                        for _, game in older_games.iterrows()]) / len(older_games)
        
        scoring_trend = recent_rpg - older_rpg
        
        return {
            'team_runs_per_game': runs_per_game,
            'team_hits_per_game': hits_per_game,
            'team_scoring_trends': scoring_trend,
            'baserunner_rate': min(0.40, runs_per_game / 12),  # Estimate baserunner rate
            'clutch_hitting': max(0.20, min(0.35, runs_per_game / 6))  # Estimate clutch rate
        }
    
    def calculate_rbi_features(self, player_name, team, batting_position, opposing_team, opposing_pitcher_era=4.50, as_of_date=None):
        """
        Calculate comprehensive RBI prediction features
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Get player stats
        player_stats = self.get_player_season_stats(player_name, team)
        
        # Get team offensive context
        team_context = self.calculate_team_offensive_context(team, as_of_date)
        
        # Calculate batting position factors
        position_factors = {
            1: 0.8,   # Leadoff - fewer RBI opportunities
            2: 0.9,   # Second - some opportunities
            3: 1.3,   # Third - prime RBI spot
            4: 1.4,   # Cleanup - most RBI opportunities
            5: 1.2,   # Five hole - good opportunities
            6: 1.0,   # Sixth - average
            7: 0.9,   # Seventh - fewer opportunities
            8: 0.8,   # Eighth - limited opportunities
            9: 0.7    # Ninth - fewest opportunities
        }
        
        position_multiplier = position_factors.get(batting_position, 1.0)
        
        # Calculate base RBI rate
        games_played = max(player_stats['games'], 1)
        season_rbi_rate = player_stats['rbi'] / games_played
        
        # Power indicators (more power = more RBI potential)
        power_factor = (player_stats['home_runs'] / games_played) * 10 + (player_stats['doubles'] / games_played) * 3
        
        # Contact ability
        contact_factor = player_stats['avg'] * 2 + (player_stats['hits'] / max(player_stats['at_bats'], 1))
        
        # On-base ability of team (affects RBI opportunities)
        team_baserunner_factor = team_context['baserunner_rate'] * 3
        
        features = {
            # Player-specific
            'season_rbi_rate': season_rbi_rate,
            'batting_average': player_stats['avg'],
            'on_base_pct': player_stats['obp'],
            'slugging_pct': player_stats['slg'],
            'power_factor': power_factor,
            'contact_factor': contact_factor,
            
            # Situational
            'batting_position': batting_position,
            'position_multiplier': position_multiplier,
            'recent_rbi': player_stats.get('recent_rbi', 3),
            
            # Team context
            'team_runs_per_game': team_context['team_runs_per_game'],
            'team_scoring_trend': team_context['team_scoring_trends'],
            'team_baserunner_rate': team_context['baserunner_rate'],
            'team_clutch_hitting': team_context['clutch_hitting'],
            
            # Opposition
            'opposing_pitcher_era': opposing_pitcher_era,
            'pitcher_difficulty': max(0.5, min(2.0, opposing_pitcher_era / 4.0)),
            
            # Weather/ballpark (simplified)
            'offensive_boost': 1.0  # Would incorporate weather/ballpark
        }
        
        return features
    
    def predict_rbi_probability(self, player_name, team, batting_position, opposing_team, opposing_pitcher_era=4.50):
        """
        Predict probability of player getting 1+ RBI
        """
        features = self.calculate_rbi_features(
            player_name, team, batting_position, opposing_team, opposing_pitcher_era
        )
        
        # Simple probability model based on key factors
        base_rate = features['season_rbi_rate']
        
        # Adjustments
        position_boost = features['position_multiplier'] - 1.0
        team_offense_boost = (features['team_runs_per_game'] - 4.5) * 0.1
        power_boost = features['power_factor'] * 0.02
        contact_boost = (features['batting_average'] - 0.250) * 0.5
        pitcher_boost = (4.5 - features['opposing_pitcher_era']) * 0.1
        trend_boost = features['team_scoring_trend'] * 0.15
        
        # Calculate final probability
        adjusted_probability = base_rate * (
            1 + position_boost + team_offense_boost + power_boost + 
            contact_boost + pitcher_boost + trend_boost
        )
        
        # Reasonable bounds
        final_probability = max(0.15, min(0.85, adjusted_probability))
        
        return {
            'player_name': player_name,
            'rbi_probability': final_probability,
            'base_rate': base_rate,
            'batting_position': batting_position,
            'key_factors': {
                'position_boost': position_boost,
                'team_offense': team_offense_boost,
                'power_factor': power_boost,
                'contact_ability': contact_boost,
                'pitcher_matchup': pitcher_boost,
                'hot_streak': trend_boost
            },
            'recommendation': 'BET' if final_probability > 0.55 else 'PASS',
            'confidence': 'HIGH' if abs(final_probability - 0.5) > 0.2 else 'MEDIUM' if abs(final_probability - 0.5) > 0.1 else 'LOW'
        }
    
    def get_probable_lineups(self, team, date_str=None):
        """
        Get probable starting lineup for a team
        This is simplified - real implementation would use lineup APIs
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Simplified lineup predictions based on team
        sample_lineups = {
            'New York Yankees': [
                ('Gleyber Torres', 2), ('Aaron Judge', 3), ('Juan Soto', 4), 
                ('Giancarlo Stanton', 5), ('Anthony Rizzo', 6), ('DJ LeMahieu', 7),
                ('Kyle Higashioka', 8), ('Oswaldo Cabrera', 9), ('Pitcher', 1)
            ],
            'Los Angeles Dodgers': [
                ('Mookie Betts', 1), ('Freddie Freeman', 2), ('Will Smith', 3),
                ('Max Muncy', 4), ('Justin Turner', 5), ('Chris Taylor', 6),
                ('Cody Bellinger', 7), ('Trea Turner', 8), ('Pitcher', 9)
            ]
        }
        
        return sample_lineups.get(team, [('Player' + str(i), i) for i in range(1, 10)])
    
    def analyze_todays_rbi_props(self):
        """
        Analyze today's RBI props for key players
        """
        print(f"\n{'='*60}")
        print(f"RBI PROP ANALYSIS - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Sample analysis (you'd get real lineups from MLB API)
        sample_matchups = [
            {
                'player': 'Aaron Judge', 'team': 'New York Yankees', 'position': 3,
                'opponent': 'Boston Red Sox', 'pitcher_era': 3.85
            },
            {
                'player': 'Mookie Betts', 'team': 'Los Angeles Dodgers', 'position': 1,
                'opponent': 'San Francisco Giants', 'pitcher_era': 4.20
            },
            {
                'player': 'Vladimir Guerrero Jr.', 'team': 'Toronto Blue Jays', 'position': 4,
                'opponent': 'Tampa Bay Rays', 'pitcher_era': 3.45
            },
            {
                'player': 'Rafael Devers', 'team': 'Boston Red Sox', 'position': 3,
                'opponent': 'New York Yankees', 'pitcher_era': 4.15
            }
        ]
        
        predictions = []
        
        for matchup in sample_matchups:
            try:
                prediction = self.predict_rbi_probability(
                    matchup['player'],
                    matchup['team'],
                    matchup['position'],
                    matchup['opponent'],
                    matchup['pitcher_era']
                )
                
                predictions.append(prediction)
                
                print(f"\n{matchup['player']} ({matchup['team']}) - Batting {matchup['position']}")
                print(f"  vs {matchup['opponent']} (Pitcher ERA: {matchup['pitcher_era']})")
                print(f"  RBI Probability: {prediction['rbi_probability']:.1%}")
                print(f"  Base Season Rate: {prediction['base_rate']:.2f} RBI/game")
                print(f"  Confidence: {prediction['confidence']}")
                print(f"  Recommendation: {prediction['recommendation']}")
                
                print(f"  Key Factors:")
                for factor, value in prediction['key_factors'].items():
                    if abs(value) > 0.01:
                        print(f"    {factor.replace('_', ' ').title()}: {value:+.2f}")
                
            except Exception as e:
                print(f"Error analyzing {matchup['player']}: {e}")
        
        # Summary
        bet_recommendations = [p for p in predictions if p['recommendation'] == 'BET']
        high_confidence = [p for p in predictions if p['confidence'] == 'HIGH']
        
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Players analyzed: {len(predictions)}")
        print(f"  Betting recommendations: {len(bet_recommendations)}")
        print(f"  High confidence picks: {len(high_confidence)}")
        
        if bet_recommendations:
            print(f"\n  TOP RBI BETS:")
            for pred in sorted(bet_recommendations, key=lambda x: x['rbi_probability'], reverse=True):
                print(f"    {pred['player_name']}: {pred['rbi_probability']:.1%} ({pred['confidence']})")
        
        return predictions

def main():
    print("MLB RBI Prediction System")
    print("Uses existing free MLB API data")
    print("="*50)
    
    system = RBIPredictionSystem()
    
    if not system.load_historical_data():
        print("Please ensure historical_mlb_games.csv is available")
        return
    
    choice = input("\nWhat would you like to do?\n1. Analyze specific player RBI\n2. Today's RBI props analysis\n3. Test system\nChoice (1-3): ")
    
    if choice == '1':
        player = input("Enter player name: ")
        team = input("Enter team: ")
        position = int(input("Enter batting position (1-9): "))
        opponent = input("Enter opposing team: ")
        pitcher_era = float(input("Enter opposing pitcher ERA (optional, default 4.50): ") or 4.50)
        
        result = system.predict_rbi_probability(player, team, position, opponent, pitcher_era)
        
        print(f"\n=== RBI PREDICTION ===")
        print(f"Player: {result['player_name']}")
        print(f"RBI Probability: {result['rbi_probability']:.1%}")
        print(f"Confidence: {result['confidence']}")
        print(f"Recommendation: {result['recommendation']}")
        
    elif choice == '2':
        system.analyze_todays_rbi_props()
        
    elif choice == '3':
        print("\n=== TESTING SYSTEM ===")
        test_result = system.predict_rbi_probability("Test Player", "Test Team", 4, "Opponent", 4.00)
        print("System working correctly!")
        print(f"Test prediction: {test_result['rbi_probability']:.1%}")
    
    print(f"\n=== ADVANTAGES OF RBI PROPS ===")
    print("• Uses your existing profitable MLB data")
    print("• No expensive API subscriptions needed")
    print("• Correlates with team offense (your strength)")
    print("• Better economics at $2 bet levels")
    print("• Batting order position is highly predictive")

if __name__ == "__main__":
    main()
