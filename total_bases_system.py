import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import datetime, timedelta
import time
import os

class MLBTotalBasesSystem:
    def __init__(self):
        self.model = None
        self.historical_data = None
        self.player_stats_cache = {}
        
    def load_historical_data(self):
        """Load historical games data"""
        try:
            self.historical_data = pd.read_csv('historical_mlb_games.csv')
            self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
            self.historical_data = self.historical_data.sort_values('date').reset_index(drop=True)
            print(f"Loaded {len(self.historical_data)} historical games for total bases analysis")
            return True
        except:
            print("Error: Could not load historical_mlb_games.csv")
            return False
    
    def get_player_season_stats(self, player_id, season=2024):
        """Get player's season statistics from MLB API"""
        if f"{player_id}_{season}" in self.player_stats_cache:
            return self.player_stats_cache[f"{player_id}_{season}"]
            
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&season={season}&group=hitting"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            # Debug: print response structure for troubleshooting
            # print(f"API Response for player {player_id}: {data}")
            
            player_stats = None
            
            # Try different ways the API might return stats
            if 'stats' in data and len(data['stats']) > 0:
                # Method 1: Standard structure
                if 'stats' in data['stats'][0] and len(data['stats'][0]['stats']) > 0:
                    stats = data['stats'][0]['stats']
                    player_stats = self.parse_player_stats(stats)
                # Method 2: Stats directly in the first element
                elif len(data['stats'][0]) > 0:
                    # Sometimes stats are directly in the array element
                    for key, value in data['stats'][0].items():
                        if isinstance(value, dict) and 'hits' in value:
                            stats = value
                            player_stats = self.parse_player_stats(stats)
                            break
            
            # Method 3: Look for any nested object with hitting stats
            if not player_stats:
                player_stats = self.find_hitting_stats_recursive(data)
            
            # If all methods fail, use defaults
            if not player_stats:
                print(f"No stats found for player {player_id}, using defaults")
                player_stats = self.get_default_player_stats()
            
            self.player_stats_cache[f"{player_id}_{season}"] = player_stats
            return player_stats
            
        except Exception as e:
            print(f"Error getting stats for player {player_id}: {e}")
            return self.get_default_player_stats()
    
    def parse_player_stats(self, stats):
        """Parse player statistics from API response"""
        try:
            # Calculate total bases and rates
            singles = int(stats.get('hits', 0)) - int(stats.get('doubles', 0)) - int(stats.get('triples', 0)) - int(stats.get('homeRuns', 0))
            total_bases = singles + (int(stats.get('doubles', 0)) * 2) + (int(stats.get('triples', 0)) * 3) + (int(stats.get('homeRuns', 0)) * 4)
            
            games_played = max(int(stats.get('gamesPlayed', 1)), 1)
            at_bats = max(int(stats.get('atBats', 1)), 1)
            
            return {
                'games': games_played,
                'at_bats': at_bats,
                'hits': int(stats.get('hits', 0)),
                'doubles': int(stats.get('doubles', 0)),
                'triples': int(stats.get('triples', 0)),
                'home_runs': int(stats.get('homeRuns', 0)),
                'total_bases': total_bases,
                'avg': float(stats.get('avg', 0.250)),
                'obp': float(stats.get('obp', 0.320)),
                'slg': float(stats.get('slg', 0.400)),
                'ops': float(stats.get('ops', 0.720)),
                'total_bases_per_game': total_bases / games_played,
                'extra_base_hit_rate': (int(stats.get('doubles', 0)) + int(stats.get('triples', 0)) + int(stats.get('homeRuns', 0))) / at_bats
            }
        except Exception as e:
            print(f"Error parsing stats: {e}")
            return self.get_default_player_stats()
    
    def find_hitting_stats_recursive(self, data, depth=0):
        """Recursively search for hitting stats in API response"""
        if depth > 3:  # Prevent infinite recursion
            return None
            
        if isinstance(data, dict):
            # Look for hitting stats indicators
            if 'hits' in data and 'atBats' in data:
                return self.parse_player_stats(data)
            
            # Recurse through dictionary values
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    result = self.find_hitting_stats_recursive(value, depth + 1)
                    if result:
                        return result
        
        elif isinstance(data, list):
            # Recurse through list elements
            for item in data:
                if isinstance(item, (dict, list)):
                    result = self.find_hitting_stats_recursive(item, depth + 1)
                    if result:
                        return result
        
        return None
    
    def get_default_player_stats(self):
        """Default stats for when API fails - more realistic values"""
        return {
            'games': 100,
            'at_bats': 350,
            'hits': 85,
            'doubles': 18,
            'triples': 2,
            'home_runs': 15,
            'total_bases': 152,  # 85 singles + 36 doubles + 6 triples + 60 HRs
            'avg': 0.243,  # League average
            'obp': 0.315,
            'slg': 0.434,
            'ops': 0.749,
            'total_bases_per_game': 1.52,  # More realistic
            'extra_base_hit_rate': 0.10
        }
    
    def get_todays_starting_lineups(self):
        """Get starting lineups for today's games"""
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=lineups"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            games_with_lineups = []
            
            if data['dates'] and len(data['dates']) > 0:
                games = data['dates'][0]['games']
                
                for game in games:
                    if game['status']['statusCode'] in ['S', 'P']:
                        game_info = {
                            'game_id': game['gamePk'],
                            'away_team': game['teams']['away']['team']['name'],
                            'home_team': game['teams']['home']['team']['name'],
                            'away_lineup': [],
                            'home_lineup': [],
                            'away_pitcher': None,
                            'home_pitcher': None
                        }
                        
                        # Try to get probable pitchers
                        if 'probablePitchers' in game:
                            if 'away' in game['probablePitchers']:
                                game_info['away_pitcher'] = {
                                    'id': game['probablePitchers']['away']['id'],
                                    'name': game['probablePitchers']['away']['fullName']
                                }
                            if 'home' in game['probablePitchers']:
                                game_info['home_pitcher'] = {
                                    'id': game['probablePitchers']['home']['id'], 
                                    'name': game['probablePitchers']['home']['fullName']
                                }
                        
                        # Note: Full lineup data often not available until closer to game time
                        # For now, we'll work with a simplified approach using top players
                        games_with_lineups.append(game_info)
            
            return games_with_lineups
            
        except Exception as e:
            print(f"Error getting lineups: {e}")
            return []
    
    def get_team_top_players(self, team_name):
        """Get top hitters for a team (simplified approach)"""
        # This is a simplified version - in production you'd want full roster data
        team_id_map = {
            'Los Angeles Dodgers': 119, 'New York Yankees': 147, 'Houston Astros': 117,
            'Atlanta Braves': 144, 'Philadelphia Phillies': 143, 'San Diego Padres': 135,
            'Toronto Blue Jays': 141, 'Boston Red Sox': 111, 'Chicago White Sox': 145,
            'Minnesota Twins': 142, 'Tampa Bay Rays': 139, 'Oakland Athletics': 133,
            'Seattle Mariners': 136, 'Texas Rangers': 140, 'Los Angeles Angels': 108,
            'Detroit Tigers': 116, 'Kansas City Royals': 118, 'Cleveland Guardians': 114,
            'Chicago Cubs': 112, 'Milwaukee Brewers': 158, 'St. Louis Cardinals': 138,
            'Pittsburgh Pirates': 134, 'Cincinnati Reds': 113, 'Arizona Diamondbacks': 109,
            'Colorado Rockies': 115, 'San Francisco Giants': 137, 'New York Mets': 121,
            'Washington Nationals': 120, 'Miami Marlins': 146, 'Baltimore Orioles': 110
        }
        
        team_id = team_id_map.get(team_name)
        if not team_id:
            return []
        
        # Get team roster
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            hitters = []
            if 'roster' in data:
                for player in data['roster']:
                    # Check for position in different possible locations
                    position = None
                    
                    # Try different ways the API might structure position data
                    if 'position' in player:
                        position = player['position'].get('abbreviation', 'UNK')
                    elif 'person' in player and 'primaryPosition' in player['person']:
                        position = player['person']['primaryPosition'].get('abbreviation', 'UNK')
                    elif 'person' in player and 'position' in player['person']:
                        position = player['person']['position'].get('abbreviation', 'UNK')
                    else:
                        # If we can't find position, assume it's a position player
                        position = 'OF'  # Default to outfielder
                    
                    # Only get position players (not pitchers)
                    if position and position not in ['P']:
                        player_info = {
                            'id': player['person']['id'],
                            'name': player['person']['fullName'],
                            'position': position
                        }
                        hitters.append(player_info)
            
            # Limit to top 9 hitters (typical lineup)
            return hitters[:9]
            
        except Exception as e:
            print(f"Error getting roster for {team_name}: {e}")
            # Return empty list - we'll use a fallback method
            return []
    
    def get_pitcher_stats(self, pitcher_id, season=2024):
        """Get pitcher statistics that affect total bases allowed"""
        if f"pitcher_{pitcher_id}_{season}" in self.player_stats_cache:
            return self.player_stats_cache[f"pitcher_{pitcher_id}_{season}"]
            
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=season&season={season}&group=pitching"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'stats' in data and len(data['stats']) > 0 and len(data['stats'][0]['stats']) > 0:
                stats = data['stats'][0]['stats']
                
                pitcher_stats = {
                    'era': float(stats.get('era', 4.50)),
                    'whip': float(stats.get('whip', 1.35)),
                    'hits_per_9': float(stats.get('hitsPer9Inn', 9.0)),
                    'hr_per_9': float(stats.get('homeRunsPer9Inn', 1.2)),
                    'avg_against': float(stats.get('avg', 0.260)),
                    'ops_against': float(stats.get('ops', 0.750))
                }
            else:
                pitcher_stats = {
                    'era': 4.50, 'whip': 1.35, 'hits_per_9': 9.0,
                    'hr_per_9': 1.2, 'avg_against': 0.260, 'ops_against': 0.750
                }
            
            self.player_stats_cache[f"pitcher_{pitcher_id}_{season}"] = pitcher_stats
            return pitcher_stats
            
        except Exception as e:
            return {
                'era': 4.50, 'whip': 1.35, 'hits_per_9': 9.0,
                'hr_per_9': 1.2, 'avg_against': 0.260, 'ops_against': 0.750
            }
    
    def calculate_ballpark_factor(self, team_name):
        """Calculate ballpark factor for total bases (simplified)"""
        # Ballpark factors for total bases (1.0 = neutral)
        ballpark_factors = {
            'Colorado Rockies': 1.15,  # Coors Field - high altitude
            'Boston Red Sox': 1.08,     # Fenway - Green Monster doubles
            'Houston Astros': 1.05,     # Minute Maid - short left field
            'Texas Rangers': 1.03,      # Globe Life - hitter friendly
            'Toronto Blue Jays': 1.02,  # Rogers Centre
            'Arizona Diamondbacks': 1.02,
            'Cincinnati Reds': 1.01,
            'Minnesota Twins': 1.01,
            'Baltimore Orioles': 1.01,
            'Chicago Cubs': 1.00,       # Wrigley - depends on wind
            'Atlanta Braves': 1.00,
            'Philadelphia Phillies': 1.00,
            'New York Yankees': 1.00,
            'Los Angeles Dodgers': 0.98,
            'Milwaukee Brewers': 0.98,
            'Pittsburgh Pirates': 0.98,
            'Kansas City Royals': 0.97,
            'Washington Nationals': 0.97,
            'New York Mets': 0.97,      # Citi Field - pitcher friendly
            'Chicago White Sox': 0.96,
            'Cleveland Guardians': 0.96,
            'Tampa Bay Rays': 0.95,     # Tropicana - indoor, spacious
            'Seattle Mariners': 0.95,   # T-Mobile Park - large foul territory
            'Los Angeles Angels': 0.95,
            'San Francisco Giants': 0.93,  # Oracle Park - large, cold
            'Detroit Tigers': 0.93,
            'Miami Marlins': 0.92,      # Marlins Park - spacious
            'Oakland Athletics': 0.90,  # Oakland Coliseum - large foul territory
            'San Diego Padres': 0.90,   # Petco Park - pitcher friendly
            'St. Louis Cardinals': 0.98
        }
        
        return ballpark_factors.get(team_name, 1.0)
    
    def predict_player_total_bases(self, player_id, player_name, team_name, opposing_pitcher_id=None):
        """Predict total bases for a specific player"""
        
        # Get player stats
        player_stats = self.get_player_season_stats(player_id)
        
        # Base prediction from season average
        base_total_bases = player_stats['total_bases_per_game']
        
        # Adjustments
        adjustments = []
        
        # 1. Ballpark factor
        ballpark_factor = self.calculate_ballpark_factor(team_name)
        ballpark_adjustment = base_total_bases * (ballpark_factor - 1.0)
        adjustments.append(('Ballpark', ballpark_adjustment))
        
        # 2. Opposing pitcher quality
        pitcher_adjustment = 0
        if opposing_pitcher_id:
            pitcher_stats = self.get_pitcher_stats(opposing_pitcher_id)
            # Better pitcher = fewer total bases
            if pitcher_stats['avg_against'] < 0.240:  # Excellent pitcher
                pitcher_adjustment = -0.4
            elif pitcher_stats['avg_against'] < 0.260:  # Good pitcher
                pitcher_adjustment = -0.2
            elif pitcher_stats['avg_against'] > 0.280:  # Poor pitcher
                pitcher_adjustment = +0.3
            
            adjustments.append(('Pitcher Quality', pitcher_adjustment))
        
        # 3. Player quality vs league average
        quality_adjustment = 0
        if player_stats['ops'] > 0.900:  # Elite hitter
            quality_adjustment = 0.4
        elif player_stats['ops'] > 0.800:  # Above average hitter
            quality_adjustment = 0.2
        elif player_stats['ops'] < 0.650:  # Below average hitter
            quality_adjustment = -0.2
        elif player_stats['ops'] < 0.600:  # Poor hitter
            quality_adjustment = -0.4
        
        adjustments.append(('Player Quality', quality_adjustment))
        
        # 4. Extra base hit tendency
        if player_stats['extra_base_hit_rate'] > 0.12:  # Power hitter
            power_adjustment = 0.15
        elif player_stats['extra_base_hit_rate'] < 0.08:  # Contact hitter
            power_adjustment = -0.1
        else:
            power_adjustment = 0
        
        adjustments.append(('Power Factor', power_adjustment))
        
        # Calculate final prediction
        total_adjustment = sum(adj[1] for adj in adjustments)
        predicted_total_bases = max(0.5, base_total_bases + total_adjustment)
        
        return {
            'player_name': player_name,
            'predicted_total_bases': predicted_total_bases,
            'base_average': base_total_bases,
            'adjustments': adjustments,
            'season_stats': player_stats
        }
    
    def get_star_players_fallback(self, team_name):
        """Fallback method: manually curated list of star players by team"""
        star_players = {
            'Atlanta Braves': [
                {'name': 'Ronald Acuña Jr.', 'id': 660670},
                {'name': 'Ozzie Albies', 'id': 645277},
                {'name': 'Matt Olson', 'id': 621566},
                {'name': 'Austin Riley', 'id': 663586},
                {'name': 'Marcell Ozuna', 'id': 542303},
            ],
            'Philadelphia Phillies': [
                {'name': 'Bryce Harper', 'id': 547180},
                {'name': 'Trea Turner', 'id': 607208},
                {'name': 'Kyle Schwarber', 'id': 656941},
                {'name': 'Nick Castellanos', 'id': 592206},
                {'name': 'J.T. Realmuto', 'id': 592663},
            ],
            'New York Yankees': [
                {'name': 'Aaron Judge', 'id': 592450},
                {'name': 'Juan Soto', 'id': 665742},
                {'name': 'Giancarlo Stanton', 'id': 519317},
                {'name': 'Gleyber Torres', 'id': 650402},
                {'name': 'Anthony Rizzo', 'id': 519203},
            ],
            'Chicago White Sox': [
                {'name': 'Luis Robert Jr.', 'id': 673357},
                {'name': 'Eloy Jiménez', 'id': 650391},
                {'name': 'Andrew Vaughn', 'id': 683734},
                {'name': 'Tim Anderson', 'id': 641313},
                {'name': 'Yoán Moncada', 'id': 660162},
            ],
            'Miami Marlins': [
                {'name': 'Jazz Chisholm Jr.', 'id': 665862},
                {'name': 'Jorge Soler', 'id': 624585},
                {'name': 'Jesús Aguilar', 'id': 542583},
                {'name': 'Avisaíl García', 'id': 541645},
                {'name': 'Jacob Stallings', 'id': 607732},
            ],
            'New York Mets': [
                {'name': 'Pete Alonso', 'id': 624413},
                {'name': 'Francisco Lindor', 'id': 596019},
                {'name': 'Starling Marte', 'id': 516782},
                {'name': 'Brandon Nimmo', 'id': 607043},
                {'name': 'Eduardo Escobar', 'id': 500871},
            ],
        }
        
        players = star_players.get(team_name, [])
        # Add position as 'UNK' since we don't have it
        for player in players:
            player['position'] = 'UNK'
        
        return players
    
    def generate_total_bases_props(self):
        """Generate total bases predictions for today's games"""
        print(f"\n{'='*60}")
        print(f"MLB TOTAL BASES PROPS - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        games = self.get_todays_starting_lineups()
        
        if not games:
            print("No games with available data found for today.")
            return []
        
        all_predictions = []
        
        for game in games:
            print(f"\n{game['away_team']} @ {game['home_team']}")
            
            # Get top players for both teams, with fallback
            away_players = self.get_team_top_players(game['away_team'])
            if not away_players:  # Use fallback if API fails
                print(f"  Using fallback player list for {game['away_team']}")
                away_players = self.get_star_players_fallback(game['away_team'])
                
            home_players = self.get_team_top_players(game['home_team'])
            if not home_players:  # Use fallback if API fails
                print(f"  Using fallback player list for {game['home_team']}")
                home_players = self.get_star_players_fallback(game['home_team'])
            
            # Process away team players
            if away_players:
                print(f"\n  {game['away_team']} Top Hitters:")
                away_pitcher_id = game['home_pitcher']['id'] if game['home_pitcher'] else None
                
                for i, player in enumerate(away_players[:6]):  # Top 6 hitters
                    prediction = self.predict_player_total_bases(
                        player['id'], 
                        player['name'], 
                        game['away_team'],
                        away_pitcher_id
                    )
                    
                    # Generate betting recommendations
                    pred_bases = prediction['predicted_total_bases']
                    
                    # Common prop lines
                    if pred_bases >= 2.0:
                        recommendation = f"OVER 1.5 Total Bases"
                        confidence = "HIGH" if pred_bases >= 2.3 else "MEDIUM"
                    elif pred_bases >= 1.7:
                        recommendation = f"OVER 1.5 Total Bases"
                        confidence = "MEDIUM"
                    elif pred_bases <= 1.2:
                        recommendation = f"UNDER 1.5 Total Bases"
                        confidence = "MEDIUM"
                    else:
                        recommendation = f"No strong lean"
                        confidence = "LOW"
                    
                    print(f"    {player['name']}: {pred_bases:.1f} TB - {recommendation}")
                    
                    if confidence in ['HIGH', 'MEDIUM']:
                        all_predictions.append({
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'player': player['name'],
                            'team': game['away_team'],
                            'opponent': game['home_team'],
                            'predicted_total_bases': pred_bases,
                            'recommendation': recommendation,
                            'confidence': confidence
                        })
            else:
                print(f"  No player data available for {game['away_team']}")
            
            # Process home team players  
            if home_players:
                print(f"\n  {game['home_team']} Top Hitters:")
                home_pitcher_id = game['away_pitcher']['id'] if game['away_pitcher'] else None
                
                for i, player in enumerate(home_players[:6]):
                    prediction = self.predict_player_total_bases(
                        player['id'],
                        player['name'],
                        game['home_team'], 
                        home_pitcher_id
                    )
                    
                    pred_bases = prediction['predicted_total_bases']
                    
                    if pred_bases >= 2.0:
                        recommendation = f"OVER 1.5 Total Bases"
                        confidence = "HIGH" if pred_bases >= 2.3 else "MEDIUM"
                    elif pred_bases >= 1.7:
                        recommendation = f"OVER 1.5 Total Bases"
                        confidence = "MEDIUM"
                    elif pred_bases <= 1.2:
                        recommendation = f"UNDER 1.5 Total Bases"
                        confidence = "MEDIUM"
                    else:
                        recommendation = f"No strong lean"
                        confidence = "LOW"
                    
                    print(f"    {player['name']}: {pred_bases:.1f} TB - {recommendation}")
                    
                    if confidence in ['HIGH', 'MEDIUM']:
                        all_predictions.append({
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'player': player['name'],
                            'team': game['home_team'],
                            'opponent': game['away_team'],
                            'predicted_total_bases': pred_bases,
                            'recommendation': recommendation,
                            'confidence': confidence
                        })
            else:
                print(f"  No player data available for {game['home_team']}")
            
            time.sleep(1)  # Be nice to the API
        
        # Save predictions
        if all_predictions:
            filename = f"total_bases_props_{datetime.now().strftime('%Y%m%d')}.csv"
            pd.DataFrame(all_predictions).to_csv(filename, index=False)
            
            print(f"\n🎯 TOTAL BASES SUMMARY:")
            high_conf = len([p for p in all_predictions if p['confidence'] == 'HIGH'])
            medium_conf = len([p for p in all_predictions if p['confidence'] == 'MEDIUM'])
            
            print(f"  Total prop bets found: {len(all_predictions)}")
            print(f"  High confidence: {high_conf}")
            print(f"  Medium confidence: {medium_conf}")
            print(f"  Saved to: {filename}")
        else:
            print(f"\n⚠️  No strong prop recommendations found for today")
        
        return all_predictions

def main():
    print("MLB Total Bases Prop Betting System")
    print("Predicts player total bases (1B=1, 2B=2, 3B=3, HR=4)")
    print("="*60)
    
    system = MLBTotalBasesSystem()
    
    # Load historical data (for future model training)
    if not system.load_historical_data():
        return
    
    print("\nGenerating total bases prop predictions...")
    print("Note: This may take a few minutes to collect player data...")
    
    predictions = system.generate_total_bases_props()
    
    if predictions:
        print(f"\n✅ Generated {len(predictions)} total bases recommendations!")
        print("\nNext steps:")
        print("1. Compare these predictions to sportsbook prop lines")
        print("2. Look for value where your prediction exceeds the implied probability")
        print("3. Focus on HIGH confidence bets first")
    else:
        print("\n⚠️  No strong recommendations found for today")

if __name__ == "__main__":
    main()