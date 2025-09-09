import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time

class SimpleTotalBasesSystem:
    def __init__(self):
        self.elite_player_stats = self.load_elite_player_database()
        
    def load_elite_player_database(self):
        """Database of elite players with their 2024-style stats"""
        return {
            # Yankees
            592450: {'name': 'Aaron Judge', 'tb_per_game': 2.1, 'ops': 0.950, 'power': 'elite'},
            665742: {'name': 'Juan Soto', 'tb_per_game': 1.9, 'ops': 0.900, 'power': 'high'},
            519317: {'name': 'Giancarlo Stanton', 'tb_per_game': 1.8, 'ops': 0.820, 'power': 'high'},
            650402: {'name': 'Gleyber Torres', 'tb_per_game': 1.6, 'ops': 0.770, 'power': 'medium'},
            683011: {'name': 'Anthony Volpe', 'tb_per_game': 1.4, 'ops': 0.720, 'power': 'medium'},
            
            # Phillies  
            547180: {'name': 'Bryce Harper', 'tb_per_game': 2.0, 'ops': 0.920, 'power': 'elite'},
            607208: {'name': 'Trea Turner', 'tb_per_game': 1.8, 'ops': 0.840, 'power': 'high'},
            656941: {'name': 'Kyle Schwarber', 'tb_per_game': 1.7, 'ops': 0.800, 'power': 'high'},
            592206: {'name': 'Nick Castellanos', 'tb_per_game': 1.6, 'ops': 0.780, 'power': 'medium'},
            664761: {'name': 'Alec Bohm', 'tb_per_game': 1.5, 'ops': 0.750, 'power': 'medium'},
            
            # Braves
            660670: {'name': 'Ronald Acuña Jr.', 'tb_per_game': 2.2, 'ops': 0.960, 'power': 'elite'},
            621566: {'name': 'Matt Olson', 'tb_per_game': 1.9, 'ops': 0.880, 'power': 'high'},
            542303: {'name': 'Marcell Ozuna', 'tb_per_game': 1.7, 'ops': 0.820, 'power': 'high'},
            663586: {'name': 'Austin Riley', 'tb_per_game': 1.8, 'ops': 0.850, 'power': 'high'},
            645277: {'name': 'Ozzie Albies', 'tb_per_game': 1.6, 'ops': 0.780, 'power': 'medium'},
            
            # Mets
            624413: {'name': 'Pete Alonso', 'tb_per_game': 1.9, 'ops': 0.870, 'power': 'elite'},
            596019: {'name': 'Francisco Lindor', 'tb_per_game': 1.8, 'ops': 0.840, 'power': 'high'},
            516782: {'name': 'Starling Marte', 'tb_per_game': 1.5, 'ops': 0.760, 'power': 'medium'},
            607043: {'name': 'Brandon Nimmo', 'tb_per_game': 1.6, 'ops': 0.780, 'power': 'medium'},
            643446: {'name': 'Jeff McNeil', 'tb_per_game': 1.4, 'ops': 0.740, 'power': 'medium'},
            
            # White Sox
            673357: {'name': 'Luis Robert Jr.', 'tb_per_game': 1.8, 'ops': 0.820, 'power': 'high'},
            650391: {'name': 'Eloy Jiménez', 'tb_per_game': 1.7, 'ops': 0.800, 'power': 'high'},
            683734: {'name': 'Andrew Vaughn', 'tb_per_game': 1.5, 'ops': 0.760, 'power': 'medium'},
            643217: {'name': 'Andrew Benintendi', 'tb_per_game': 1.4, 'ops': 0.730, 'power': 'medium'},
            660162: {'name': 'Yoán Moncada', 'tb_per_game': 1.4, 'ops': 0.720, 'power': 'medium'},
            
            # Marlins
            665862: {'name': 'Jazz Chisholm Jr.', 'tb_per_game': 1.7, 'ops': 0.800, 'power': 'high'},
            624585: {'name': 'Jorge Soler', 'tb_per_game': 1.6, 'ops': 0.780, 'power': 'high'},
            542583: {'name': 'Jesús Aguilar', 'tb_per_game': 1.4, 'ops': 0.740, 'power': 'medium'},
            593160: {'name': 'Jake Burger', 'tb_per_game': 1.5, 'ops': 0.750, 'power': 'medium'},
        }
    
    def get_ballpark_factor(self, team_name):
        """Ballpark factors for total bases"""
        factors = {
            'Colorado Rockies': 1.15,
            'Boston Red Sox': 1.08,
            'Houston Astros': 1.05,
            'Philadelphia Phillies': 1.02,
            'New York Yankees': 1.02,
            'Atlanta Braves': 1.01,
            'Chicago Cubs': 1.00,
            'New York Mets': 0.97,
            'Chicago White Sox': 0.96,
            'Miami Marlins': 0.92,
        }
        return factors.get(team_name, 1.0)
    
    def get_todays_games(self):
        """Get today's scheduled games"""
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            games = []
            if data['dates'] and len(data['dates']) > 0:
                for game in data['dates'][0]['games']:
                    if game['status']['statusCode'] in ['S', 'P']:
                        games.append({
                            'away_team': game['teams']['away']['team']['name'],
                            'home_team': game['teams']['home']['team']['name']
                        })
            return games
        except:
            # Fallback with sample games
            return [
                {'away_team': 'Atlanta Braves', 'home_team': 'Philadelphia Phillies'},
                {'away_team': 'Miami Marlins', 'home_team': 'New York Mets'},
                {'away_team': 'New York Yankees', 'home_team': 'Chicago White Sox'}
            ]
    
    def get_star_players_for_team(self, team_name):
        """Get star players we have data for"""
        team_stars = {
            'New York Yankees': [592450, 665742, 519317, 650402, 683011],
            'Philadelphia Phillies': [547180, 607208, 656941, 592206, 664761], 
            'Atlanta Braves': [660670, 621566, 542303, 663586, 645277],
            'New York Mets': [624413, 596019, 516782, 607043, 643446],
            'Chicago White Sox': [673357, 650391, 683734, 643217, 660162],
            'Miami Marlins': [665862, 624585, 542583, 593160]
        }
        
        player_ids = team_stars.get(team_name, [])
        players = []
        
        for player_id in player_ids:
            if player_id in self.elite_player_stats:
                player_data = self.elite_player_stats[player_id].copy()
                player_data['id'] = player_id
                players.append(player_data)
        
        return players
    
    def predict_total_bases(self, player_data, team_name, vs_good_pitcher=False):
        """Predict total bases for a player"""
        base_tb = player_data['tb_per_game']
        
        # Ballpark adjustment
        ballpark_factor = self.get_ballpark_factor(team_name)
        ballpark_adj = base_tb * (ballpark_factor - 1.0)
        
        # Pitcher adjustment
        pitcher_adj = 0
        if vs_good_pitcher:
            if player_data['power'] == 'elite':
                pitcher_adj = -0.2  # Elite hitters less affected
            elif player_data['power'] == 'high':
                pitcher_adj = -0.3
            else:
                pitcher_adj = -0.4  # Average hitters more affected
        
        # Player quality boost
        quality_adj = 0
        if player_data['ops'] > 0.900:
            quality_adj = 0.3
        elif player_data['ops'] > 0.800:
            quality_adj = 0.1
        elif player_data['ops'] < 0.750:
            quality_adj = -0.1
        
        predicted_tb = base_tb + ballpark_adj + pitcher_adj + quality_adj
        predicted_tb = max(0.8, predicted_tb)  # Floor at 0.8
        
        return predicted_tb
    
    def generate_props(self):
        """Generate total bases prop predictions"""
        print(f"\n{'='*60}")
        print(f"MLB TOTAL BASES PROPS - {datetime.now().strftime('%Y-%m-%d')}")  
        print(f"(Using Elite Player Database)")
        print(f"{'='*60}")
        
        games = self.get_todays_games()
        all_predictions = []
        
        for game in games:
            print(f"\n{game['away_team']} @ {game['home_team']}")
            
            # Away team
            away_stars = self.get_star_players_for_team(game['away_team'])
            if away_stars:
                print(f"\n  {game['away_team']} Star Hitters:")
                for player in away_stars:
                    predicted_tb = self.predict_total_bases(
                        player, game['away_team'], vs_good_pitcher=False
                    )
                    
                    # Determine recommendation
                    if predicted_tb >= 2.1:
                        rec = "OVER 1.5 TB"
                        conf = "HIGH"
                    elif predicted_tb >= 1.8:
                        rec = "OVER 1.5 TB" 
                        conf = "MEDIUM"
                    elif predicted_tb <= 1.2:
                        rec = "UNDER 1.5 TB"
                        conf = "MEDIUM"
                    else:
                        rec = "No strong lean"
                        conf = "LOW"
                    
                    print(f"    {player['name']}: {predicted_tb:.1f} TB - {rec}")
                    
                    if conf in ['HIGH', 'MEDIUM']:
                        all_predictions.append({
                            'player': player['name'],
                            'team': game['away_team'],
                            'predicted_tb': predicted_tb,
                            'recommendation': rec,
                            'confidence': conf
                        })
            
            # Home team  
            home_stars = self.get_star_players_for_team(game['home_team'])
            if home_stars:
                print(f"\n  {game['home_team']} Star Hitters:")
                for player in home_stars:
                    predicted_tb = self.predict_total_bases(
                        player, game['home_team'], vs_good_pitcher=False
                    )
                    
                    if predicted_tb >= 2.1:
                        rec = "OVER 1.5 TB"
                        conf = "HIGH"
                    elif predicted_tb >= 1.8:
                        rec = "OVER 1.5 TB"
                        conf = "MEDIUM" 
                    elif predicted_tb <= 1.2:
                        rec = "UNDER 1.5 TB"
                        conf = "MEDIUM"
                    else:
                        rec = "No strong lean"
                        conf = "LOW"
                    
                    print(f"    {player['name']}: {predicted_tb:.1f} TB - {rec}")
                    
                    if conf in ['HIGH', 'MEDIUM']:
                        all_predictions.append({
                            'player': player['name'],
                            'team': game['home_team'],
                            'predicted_tb': predicted_tb,
                            'recommendation': rec,
                            'confidence': conf
                        })
        
        # Summary
        if all_predictions:
            print(f"\n🎯 TOTAL BASES RECOMMENDATIONS:")
            high_conf = len([p for p in all_predictions if p['confidence'] == 'HIGH'])
            medium_conf = len([p for p in all_predictions if p['confidence'] == 'MEDIUM'])
            
            print(f"  Total recommendations: {len(all_predictions)}")
            print(f"  High confidence: {high_conf}")
            print(f"  Medium confidence: {medium_conf}")
            
            print(f"\n📊 TOP RECOMMENDATIONS:")
            # Show top recommendations
            sorted_preds = sorted(all_predictions, key=lambda x: x['predicted_tb'], reverse=True)
            for pred in sorted_preds[:8]:
                print(f"    {pred['player']}: {pred['predicted_tb']:.1f} TB - {pred['recommendation']} ({pred['confidence']})")
            
            # Save to CSV
            filename = f"simple_tb_props_{datetime.now().strftime('%Y%m%d')}.csv"
            pd.DataFrame(all_predictions).to_csv(filename, index=False)
            print(f"\n💾 Saved to: {filename}")
            
        else:
            print(f"\n⚠️  No strong recommendations found")
        
        return all_predictions

def main():
    print("Simplified MLB Total Bases System")
    print("Using curated elite player database")
    print("="*50)
    
    system = SimpleTotalBasesSystem()
    predictions = system.generate_props()

if __name__ == "__main__":
    main()