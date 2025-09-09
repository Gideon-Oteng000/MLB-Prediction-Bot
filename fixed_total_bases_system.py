import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
import time

class FixedTotalBasesSystem:
    def __init__(self, odds_api_key):
        self.odds_api_key = odds_api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        
        # Enhanced player database with 2024 projections
        self.player_database = {
            # Elite Tier (2.0+ TB/game)
            'Aaron Judge': {'tb_avg': 2.3, 'team': 'New York Yankees', 'power': 'elite', 'consistency': 0.85},
            'Mookie Betts': {'tb_avg': 2.1, 'team': 'Los Angeles Dodgers', 'power': 'elite', 'consistency': 0.88},
            'Bryce Harper': {'tb_avg': 2.1, 'team': 'Philadelphia Phillies', 'power': 'elite', 'consistency': 0.82},
            'Ronald Acuña Jr.': {'tb_avg': 2.2, 'team': 'Atlanta Braves', 'power': 'elite', 'consistency': 0.80},
            'Juan Soto': {'tb_avg': 2.0, 'team': 'New York Yankees', 'power': 'elite', 'consistency': 0.90},
            'Yordan Alvarez': {'tb_avg': 2.0, 'team': 'Houston Astros', 'power': 'elite', 'consistency': 0.83},
            
            # High Tier (1.7-1.9 TB/game)
            'Freddie Freeman': {'tb_avg': 1.9, 'team': 'Los Angeles Dodgers', 'power': 'high', 'consistency': 0.85},
            'Matt Olson': {'tb_avg': 1.9, 'team': 'Atlanta Braves', 'power': 'high', 'consistency': 0.78},
            'Pete Alonso': {'tb_avg': 1.9, 'team': 'New York Mets', 'power': 'high', 'consistency': 0.75},
            'Corey Seager': {'tb_avg': 1.9, 'team': 'Texas Rangers', 'power': 'high', 'consistency': 0.80},
            'Fernando Tatis Jr.': {'tb_avg': 1.9, 'team': 'San Diego Padres', 'power': 'high', 'consistency': 0.77},
            'Giancarlo Stanton': {'tb_avg': 1.8, 'team': 'New York Yankees', 'power': 'high', 'consistency': 0.70},
            'Austin Riley': {'tb_avg': 1.8, 'team': 'Atlanta Braves', 'power': 'high', 'consistency': 0.82},
            'Trea Turner': {'tb_avg': 1.8, 'team': 'Philadelphia Phillies', 'power': 'high', 'consistency': 0.85},
            'Francisco Lindor': {'tb_avg': 1.8, 'team': 'New York Mets', 'power': 'high', 'consistency': 0.83},
            'Manny Machado': {'tb_avg': 1.8, 'team': 'San Diego Padres', 'power': 'high', 'consistency': 0.81},
            'José Ramírez': {'tb_avg': 1.8, 'team': 'Cleveland Guardians', 'power': 'high', 'consistency': 0.86},
            'Rafael Devers': {'tb_avg': 1.8, 'team': 'Boston Red Sox', 'power': 'high', 'consistency': 0.79},
            'Vladimir Guerrero Jr.': {'tb_avg': 1.8, 'team': 'Toronto Blue Jays', 'power': 'high', 'consistency': 0.81},
            'Alex Bregman': {'tb_avg': 1.8, 'team': 'Houston Astros', 'power': 'high', 'consistency': 0.84},
            'Kyle Tucker': {'tb_avg': 1.9, 'team': 'Houston Astros', 'power': 'high', 'consistency': 0.82},
            'Gunnar Henderson': {'tb_avg': 1.8, 'team': 'Baltimore Orioles', 'power': 'high', 'consistency': 0.79},
            'Ketel Marte': {'tb_avg': 1.8, 'team': 'Arizona Diamondbacks', 'power': 'high', 'consistency': 0.83},
            'Julio Rodríguez': {'tb_avg': 1.7, 'team': 'Seattle Mariners', 'power': 'high', 'consistency': 0.78},
            'Kyle Schwarber': {'tb_avg': 1.7, 'team': 'Philadelphia Phillies', 'power': 'high', 'consistency': 0.74},
            
            # Medium Tier (1.4-1.6 TB/game)
            'Gleyber Torres': {'tb_avg': 1.5, 'team': 'New York Yankees', 'power': 'medium', 'consistency': 0.78},
            'Ozzie Albies': {'tb_avg': 1.6, 'team': 'Atlanta Braves', 'power': 'medium', 'consistency': 0.82},
            'Nick Castellanos': {'tb_avg': 1.6, 'team': 'Philadelphia Phillies', 'power': 'medium', 'consistency': 0.80},
            'Will Smith': {'tb_avg': 1.6, 'team': 'Los Angeles Dodgers', 'power': 'medium', 'consistency': 0.83},
            'Brandon Nimmo': {'tb_avg': 1.6, 'team': 'New York Mets', 'power': 'medium', 'consistency': 0.81},
            'Xander Bogaerts': {'tb_avg': 1.6, 'team': 'San Diego Padres', 'power': 'medium', 'consistency': 0.82},
            'Anthony Volpe': {'tb_avg': 1.4, 'team': 'New York Yankees', 'power': 'medium', 'consistency': 0.75},
            'Bo Bichette': {'tb_avg': 1.6, 'team': 'Toronto Blue Jays', 'power': 'medium', 'consistency': 0.79},
            'Jose Altuve': {'tb_avg': 1.7, 'team': 'Houston Astros', 'power': 'medium', 'consistency': 0.84},
            'Adley Rutschman': {'tb_avg': 1.7, 'team': 'Baltimore Orioles', 'power': 'medium', 'consistency': 0.81},
            'Corbin Carroll': {'tb_avg': 1.6, 'team': 'Arizona Diamondbacks', 'power': 'medium', 'consistency': 0.77},
            'Cal Raleigh': {'tb_avg': 1.6, 'team': 'Seattle Mariners', 'power': 'medium', 'consistency': 0.74},
            'Bobby Witt Jr.': {'tb_avg': 1.7, 'team': 'Kansas City Royals', 'power': 'medium', 'consistency': 0.80},
            'Byron Buxton': {'tb_avg': 1.7, 'team': 'Minnesota Twins', 'power': 'medium', 'consistency': 0.72},
            'Christian Yelich': {'tb_avg': 1.6, 'team': 'Milwaukee Brewers', 'power': 'medium', 'consistency': 0.79},
            'Cody Bellinger': {'tb_avg': 1.7, 'team': 'Chicago Cubs', 'power': 'medium', 'consistency': 0.76},
            'Ian Happ': {'tb_avg': 1.6, 'team': 'Chicago Cubs', 'power': 'medium', 'consistency': 0.78},
            'Salvador Perez': {'tb_avg': 1.5, 'team': 'Kansas City Royals', 'power': 'medium', 'consistency': 0.77},
            'Carlos Correa': {'tb_avg': 1.6, 'team': 'Minnesota Twins', 'power': 'medium', 'consistency': 0.81},
            'William Contreras': {'tb_avg': 1.7, 'team': 'Milwaukee Brewers', 'power': 'medium', 'consistency': 0.80},
            'Luis Robert Jr.': {'tb_avg': 1.6, 'team': 'Chicago White Sox', 'power': 'medium', 'consistency': 0.73},
            'Eloy Jiménez': {'tb_avg': 1.5, 'team': 'Chicago White Sox', 'power': 'medium', 'consistency': 0.71},
            'Brent Rooker': {'tb_avg': 1.5, 'team': 'Oakland Athletics', 'power': 'medium', 'consistency': 0.75},
            
            # Lower Tier (1.1-1.3 TB/game)
            'Andrew Vaughn': {'tb_avg': 1.4, 'team': 'Chicago White Sox', 'power': 'low', 'consistency': 0.74},
            'Seth Brown': {'tb_avg': 1.3, 'team': 'Oakland Athletics', 'power': 'low', 'consistency': 0.70},
            'Abraham Toro': {'tb_avg': 1.2, 'team': 'Oakland Athletics', 'power': 'low', 'consistency': 0.68},
            'Ryan McMahon': {'tb_avg': 1.5, 'team': 'Colorado Rockies', 'power': 'low', 'consistency': 0.72},
            'Charlie Blackmon': {'tb_avg': 1.6, 'team': 'Colorado Rockies', 'power': 'low', 'consistency': 0.76}
        }
        
        # Ballpark factors for total bases
        self.ballpark_factors = {
            'Colorado Rockies': 1.25,      # Coors Field - massive advantage
            'Boston Red Sox': 1.15,        # Fenway Park - Green Monster
            'Houston Astros': 1.12,        # Minute Maid Park
            'Texas Rangers': 1.10,         # Globe Life Field
            'Philadelphia Phillies': 1.08, # Citizens Bank Park
            'New York Yankees': 1.06,      # Yankee Stadium - short right field
            'Cincinnati Reds': 1.04,       # Great American Ball Park
            'Atlanta Braves': 1.02,        # Truist Park
            'Baltimore Orioles': 1.02,     # Camden Yards
            'Toronto Blue Jays': 1.00,     # Rogers Centre
            'Los Angeles Dodgers': 0.96,   # Dodger Stadium
            'Milwaukee Brewers': 0.94,     # American Family Field
            'New York Mets': 0.92,         # Citi Field - pitcher friendly
            'Chicago White Sox': 0.90,     # Guaranteed Rate Field
            'Cleveland Guardians': 0.90,   # Progressive Field
            'Pittsburgh Pirates': 0.88,    # PNC Park
            'Kansas City Royals': 0.88,    # Kauffman Stadium
            'Washington Nationals': 0.88,  # Nationals Park
            'Tampa Bay Rays': 0.86,        # Tropicana Field - dome
            'Seattle Mariners': 0.84,      # T-Mobile Park
            'Detroit Tigers': 0.84,        # Comerica Park
            'Los Angeles Angels': 0.84,    # Angel Stadium
            'San Francisco Giants': 0.80,  # Oracle Park - very pitcher friendly
            'Miami Marlins': 0.80,         # loanDepot park
            'Chicago Cubs': 0.78,          # Wrigley Field - depends on wind
            'Arizona Diamondbacks': 0.78,  # Chase Field
            'San Diego Padres': 0.76,      # Petco Park - very pitcher friendly
            'Minnesota Twins': 0.76,       # Target Field
            'Oakland Athletics': 0.74,     # Oakland Coliseum - foul territory
        }
    
    def get_player_props_odds(self):
        """Get player prop odds from The Odds API (if available)"""
        # Note: Player props require a paid API plan
        # This is a placeholder for when you upgrade
        print("ℹ️  Player props require paid API plan - using database projections")
        return None
    
    def get_todays_games(self):
        """Get today's MLB games"""
        url = f"{self.base_url}/sports/baseball_mlb/odds"
        params = {
            'api_key': self.odds_api_key,
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                return []
            
            data = response.json()
            games = []
            
            for game in data:
                games.append({
                    'away_team': game['away_team'],
                    'home_team': game['home_team'],
                    'commence_time': game['commence_time']
                })
            
            return games
            
        except Exception as e:
            print(f"Error getting games: {e}")
            return []
    
    def get_star_players_for_team(self, team_name):
        """Get star players we have data for"""
        players = []
        for player_name, data in self.player_database.items():
            if data['team'] == team_name:
                players.append({
                    'name': player_name,
                    'tb_avg': data['tb_avg'],
                    'power': data['power'],
                    'consistency': data['consistency']
                })
        return players
    
    def calculate_daily_variance(self, base_tb, consistency):
        """Add daily variance based on player consistency"""
        # More consistent players have less daily variance
        variance_factor = (1 - consistency) * 0.3  # Max 30% variance for inconsistent players
        daily_adjustment = np.random.normal(0, variance_factor * base_tb)
        return daily_adjustment
    
    def predict_total_bases(self, player_data, team_name, opponent_strength='average'):
        """Enhanced total bases prediction"""
        base_tb = player_data['tb_avg']
        
        # 1. Ballpark adjustment
        ballpark_factor = self.ballpark_factors.get(team_name, 1.0)
        ballpark_adj = base_tb * (ballpark_factor - 1.0)
        
        # 2. Opponent pitching strength adjustment
        pitching_adj = 0
        if opponent_strength == 'strong':
            if player_data['power'] == 'elite':
                pitching_adj = -0.15  # Elite hitters less affected
            elif player_data['power'] == 'high':
                pitching_adj = -0.25
            else:
                pitching_adj = -0.35
        elif opponent_strength == 'weak':
            pitching_adj = 0.2  # All players benefit vs weak pitching
        
        # 3. Player power tier adjustment
        power_adj = 0
        if player_data['power'] == 'elite':
            power_adj = 0.15
        elif player_data['power'] == 'high':
            power_adj = 0.08
        elif player_data['power'] == 'low':
            power_adj = -0.15
        
        # 4. Consistency factor (reliable players get small boost)
        consistency_adj = (player_data['consistency'] - 0.8) * 0.1
        
        # 5. Daily variance
        daily_variance = self.calculate_daily_variance(base_tb, player_data['consistency'])
        
        # Calculate final prediction
        predicted_tb = base_tb + ballpark_adj + pitching_adj + power_adj + consistency_adj + daily_variance
        predicted_tb = max(0.8, predicted_tb)  # Floor at 0.8
        
        return {
            'predicted_tb': predicted_tb,
            'base_tb': base_tb,
            'ballpark_adj': ballpark_adj,
            'pitching_adj': pitching_adj,
            'power_adj': power_adj,
            'consistency_adj': consistency_adj,
            'daily_variance': daily_variance
        }
    
    def generate_recommendations(self, prediction, player_name):
        """Generate betting recommendations focusing on 1.5 TB line"""
        predicted_tb = prediction['predicted_tb']
        recommendations = []
        
        # Focus on the 1.5 TB line (most common and valuable)
        edge = abs(predicted_tb - 1.5)
        
        if predicted_tb > 1.75:  # Strong OVER 1.5
            conf = 'HIGH' if predicted_tb > 2.0 else 'MEDIUM'
            recommendations.append({
                'line': f"OVER 1.5 TB",
                'confidence': conf,
                'predicted_value': predicted_tb,
                'edge': predicted_tb - 1.5
            })
        elif predicted_tb < 1.25:  # Strong UNDER 1.5
            conf = 'HIGH' if predicted_tb < 1.0 else 'MEDIUM'
            recommendations.append({
                'line': f"UNDER 1.5 TB",
                'confidence': conf,
                'predicted_value': predicted_tb,
                'edge': 1.5 - predicted_tb
            })
        
        # Add 2.5 TB line for elite players only
        if predicted_tb > 2.7:  # OVER 2.5 for superstars
            recommendations.append({
                'line': f"OVER 2.5 TB",
                'confidence': 'MEDIUM',
                'predicted_value': predicted_tb,
                'edge': predicted_tb - 2.5
            })
        elif predicted_tb < 2.2 and predicted_tb > 1.8:  # UNDER 2.5 for good but not elite
            recommendations.append({
                'line': f"UNDER 2.5 TB",
                'confidence': 'MEDIUM',
                'predicted_value': predicted_tb,
                'edge': 2.5 - predicted_tb
            })
        
        return recommendations
    
    def generate_daily_props(self):
        """Generate total bases props for today's games"""
        print(f"\n{'='*70}")
        print(f"FIXED TOTAL BASES PROPS - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Enhanced Player Database + Ballpark Factors")
        print(f"{'='*70}")
        
        games = self.get_todays_games()
        if not games:
            print("❌ No games found")
            return []
        
        all_recommendations = []
        
        for game in games:
            home_team = game['home_team']
            away_team = game['away_team']
            
            print(f"\n{away_team} @ {home_team}")
            
            # Away team players
            away_players = self.get_star_players_for_team(away_team)
            if away_players:
                print(f"\n  {away_team} Stars:")
                for player in away_players:
                    prediction = self.predict_total_bases(player, away_team)
                    recommendations = self.generate_recommendations(prediction, player['name'])
                    
                    predicted_tb = prediction['predicted_tb']
                    ballpark_effect = prediction['ballpark_adj']
                    
                    print(f"    {player['name']}: {predicted_tb:.1f} TB (ballpark: {ballpark_effect:+.1f})")
                    
                    if recommendations:
                        for rec in recommendations:
                            conf_emoji = "🔥" if rec['confidence'] == 'HIGH' else "📊"
                            edge_text = f" (edge: {rec.get('edge', 0):.1f})" if 'edge' in rec else ""
                            print(f"      {conf_emoji} {rec['line']}{edge_text}")
                            
                            all_recommendations.append({
                                'player': player['name'],
                                'team': away_team,
                                'opponent': home_team,
                                'predicted_tb': predicted_tb,
                                'recommendation': rec['line'],
                                'confidence': rec['confidence'],
                                'power_tier': player['power']
                            })
            
            # Home team players
            home_players = self.get_star_players_for_team(home_team)
            if home_players:
                print(f"\n  {home_team} Stars:")
                for player in home_players:
                    prediction = self.predict_total_bases(player, home_team)
                    recommendations = self.generate_recommendations(prediction, player['name'])
                    
                    predicted_tb = prediction['predicted_tb']
                    ballpark_effect = prediction['ballpark_adj']
                    
                    print(f"    {player['name']}: {predicted_tb:.1f} TB (ballpark: {ballpark_effect:+.1f})")
                    
                    if recommendations:
                        for rec in recommendations:
                            conf_emoji = "🔥" if rec['confidence'] == 'HIGH' else "📊"
                            edge_text = f" (edge: {rec.get('edge', 0):.1f})" if 'edge' in rec else ""
                            print(f"      {conf_emoji} {rec['line']}{edge_text}")
                            
                            all_recommendations.append({
                                'player': player['name'],
                                'team': home_team,
                                'opponent': away_team,
                                'predicted_tb': predicted_tb,
                                'recommendation': rec['line'],
                                'confidence': rec['confidence'],
                                'power_tier': player['power']
                            })
        
        # Summary and top picks
        if all_recommendations:
            print(f"\n{'='*70}")
            print(f"TOTAL BASES BETTING SUMMARY")
            print(f"{'='*70}")
            
            high_conf = [r for r in all_recommendations if r['confidence'] == 'HIGH']
            medium_conf = [r for r in all_recommendations if r['confidence'] == 'MEDIUM']
            
            print(f"Total recommendations: {len(all_recommendations)}")
            print(f"High confidence: {len(high_conf)}")
            print(f"Medium confidence: {len(medium_conf)}")
            
            print(f"\n🔥 TOP TOTAL BASES PICKS:")
            sorted_recs = sorted(all_recommendations, 
                               key=lambda x: (x['confidence'] == 'HIGH', x['predicted_tb']), 
                               reverse=True)
            
            for i, rec in enumerate(sorted_recs[:15], 1):
                conf_emoji = "🔥" if rec['confidence'] == 'HIGH' else "📊"
                tier_emoji = "⭐" if rec['power_tier'] == 'elite' else "🌟" if rec['power_tier'] == 'high' else "📈"
                
                print(f"  {i:2}. {conf_emoji} {tier_emoji} {rec['player']} ({rec['team']}): {rec['recommendation']}")
                print(f"       Predicted: {rec['predicted_tb']:.1f} TB vs {rec['opponent']}")
            
            # Save recommendations
            filename = f"fixed_total_bases_{datetime.now().strftime('%Y%m%d')}.csv"
            pd.DataFrame(all_recommendations).to_csv(filename, index=False)
            print(f"\n💾 Saved {len(all_recommendations)} recommendations to: {filename}")
            
        else:
            print("❌ No strong recommendations found")
        
        return all_recommendations

def main():
    print("Fixed Total Bases System with Odds API Integration")
    print("="*55)
    
    # Get API key
    api_key = input("Enter your Odds API key: ").strip()
    if not api_key:
        print("❌ No API key provided")
        return
    
    # Initialize system
    system = FixedTotalBasesSystem(api_key)
    
    # Generate predictions
    recommendations = system.generate_daily_props()

if __name__ == "__main__":
    main()