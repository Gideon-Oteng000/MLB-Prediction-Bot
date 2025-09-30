#!/usr/bin/env python3
"""
MLB Games and Starting Lineups Fetcher
Pulls today's MLB games, starting lineups, and starting pitchers from ESPN and MLB Stats APIs
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import time

class MLBLineupFetcher:
    def __init__(self):
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.espn_base = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
        self.mlb_stats_base = "https://statsapi.mlb.com/api/v1"
        
        # Team name mappings for consistency across APIs
        self.team_mappings = {
            'WSH': 'Nationals', 'LAA': 'Angels', 'LAD': 'Dodgers',
            'NYM': 'Mets', 'NYY': 'Yankees', 'SD': 'Padres',
            'SF': 'Giants', 'TB': 'Rays', 'KC': 'Royals'
        }
    
    def get_espn_games_and_lineups(self) -> Dict:
        """Fetch games, lineups, and starting pitchers from ESPN API"""
        games_data = {}
        
        try:
            # Get today's games from ESPN
            scoreboard_url = f"{self.espn_base}/scoreboard?dates={self.today.replace('-', '')}"
            response = requests.get(scoreboard_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            games = data.get('events', [])
            
            for game in games:
                game_id = game.get('id')
                competition = game.get('competitions', [{}])[0]
                competitors = competition.get('competitors', [])
                
                if len(competitors) >= 2:
                    away_team_data = competitors[1]
                    home_team_data = competitors[0]
                    
                    away_team = away_team_data.get('team', {}).get('abbreviation')
                    home_team = home_team_data.get('team', {}).get('abbreviation')
                    
                    # Get starting pitchers from competitors data
                    away_pitcher = self._get_espn_starting_pitcher(away_team_data)
                    home_pitcher = self._get_espn_starting_pitcher(home_team_data)
                    
                    # Try to get lineups for each game
                    lineup_data = self._get_espn_lineup(game_id)
                    
                    games_data[f"{away_team}_vs_{home_team}"] = {
                        'away_team': away_team,
                        'home_team': home_team,
                        'away_lineup': lineup_data.get('away', []),
                        'home_lineup': lineup_data.get('home', []),
                        'away_pitcher': away_pitcher,
                        'home_pitcher': home_pitcher,
                        'game_time': game.get('date', '')
                    }
                    
        except requests.RequestException as e:
            print(f"ESPN API Error: {e}")
        except Exception as e:
            print(f"ESPN Processing Error: {e}")
            
        return games_data
    
    def _get_espn_starting_pitcher(self, team_data: Dict) -> Dict:
        """Extract starting pitcher from ESPN team data"""
        pitcher_info = {'name': 'TBD', 'wins': '', 'losses': '', 'era': ''}
        
        try:
            # ESPN provides probable pitchers in the probables array
            probables = team_data.get('probables', [])
            if probables and len(probables) > 0:
                pitcher = probables[0]
                athlete = pitcher.get('athlete', {})
                pitcher_info['name'] = athlete.get('displayName', 'TBD')
                
                # Get pitcher statistics if available
                stats = pitcher.get('statistics', [])
                for stat in stats:
                    if stat.get('name') == 'wins':
                        pitcher_info['wins'] = stat.get('displayValue', '')
                    elif stat.get('name') == 'losses':
                        pitcher_info['losses'] = stat.get('displayValue', '')
                    elif stat.get('name') == 'ERA':
                        pitcher_info['era'] = stat.get('displayValue', '')
            
            # Alternative: check for starting pitcher in team's players
            if pitcher_info['name'] == 'TBD':
                starter = team_data.get('starter')
                if starter:
                    pitcher_info['name'] = starter.get('athlete', {}).get('displayName', 'TBD')
                    
        except Exception as e:
            print(f"Error extracting ESPN pitcher: {e}")
            
        return pitcher_info
    
    def _get_espn_lineup(self, game_id: str) -> Dict:
        """Get lineup for specific ESPN game"""
        lineups = {'away': [], 'home': []}
        
        try:
            # ESPN game details endpoint
            game_url = f"{self.espn_base}/summary?event={game_id}"
            response = requests.get(game_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for lineups in boxscore
            boxscore = data.get('boxscore', {})
            players = boxscore.get('players', [])
            
            for team_idx, team_players in enumerate(players):
                # ESPN API: index 0 = away team, index 1 = home team (based on competitors order)
                team_type = 'away' if team_idx == 0 else 'home'
                batting_stats = team_players.get('statistics', [{}])[0].get('athletes', [])

                batting_order = 1
                for player in batting_stats[:9]:  # Get first 9 batters
                    player_info = {
                        'name': player.get('athlete', {}).get('displayName', 'Unknown'),
                        'position': player.get('position', {}).get('abbreviation', ''),
                        'batting_order': batting_order
                    }
                    lineups[team_type].append(player_info)
                    batting_order += 1
                    
        except Exception as e:
            print(f"ESPN Lineup Error for game {game_id}: {e}")
            
        return lineups
    
    def get_mlb_stats_games_and_lineups(self) -> Dict:
        """Fetch games, lineups, and starting pitchers from MLB Stats API"""
        games_data = {}
        
        try:
            # Get today's schedule with hydration to include probable pitchers
            schedule_url = f"{self.mlb_stats_base}/schedule?sportId=1&date={self.today}&hydrate=probablePitcher"
            response = requests.get(schedule_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            games = data.get('dates', [{}])[0].get('games', []) if data.get('dates') else []
            
            for game in games:
                game_pk = game.get('gamePk')
                away_team = game.get('teams', {}).get('away', {}).get('team', {}).get('name')
                home_team = game.get('teams', {}).get('home', {}).get('team', {}).get('name')
                
                # Get probable pitchers from the schedule data
                away_pitcher = self._get_mlb_stats_pitcher(game.get('teams', {}).get('away', {}))
                home_pitcher = self._get_mlb_stats_pitcher(game.get('teams', {}).get('home', {}))
                
                # Get lineups for this game
                lineup_data = self._get_mlb_stats_lineup(game_pk)
                
                # If pitchers not found in schedule, try to get from game data
                if away_pitcher['name'] == 'TBD' or home_pitcher['name'] == 'TBD':
                    pitcher_data = self._get_mlb_stats_game_pitchers(game_pk)
                    if away_pitcher['name'] == 'TBD':
                        away_pitcher = pitcher_data.get('away', away_pitcher)
                    if home_pitcher['name'] == 'TBD':
                        home_pitcher = pitcher_data.get('home', home_pitcher)
                
                games_data[f"{game_pk}"] = {
                    'away_team': away_team,
                    'home_team': home_team,
                    'away_lineup': lineup_data.get('away', []),
                    'home_lineup': lineup_data.get('home', []),
                    'away_pitcher': away_pitcher,
                    'home_pitcher': home_pitcher,
                    'game_time': game.get('gameDate', '')
                }
                
        except requests.RequestException as e:
            print(f"MLB Stats API Error: {e}")
        except Exception as e:
            print(f"MLB Stats Processing Error: {e}")
            
        return games_data
    
    def _get_mlb_stats_pitcher(self, team_data: Dict) -> Dict:
        """Extract starting pitcher from MLB Stats team data"""
        pitcher_info = {'name': 'TBD', 'wins': '', 'losses': '', 'era': ''}
        
        try:
            probable_pitcher = team_data.get('probablePitcher', {})
            if probable_pitcher:
                pitcher_info['name'] = probable_pitcher.get('fullName', 'TBD')
                
                # Get pitcher stats for the season
                pitcher_id = probable_pitcher.get('id')
                if pitcher_id:
                    stats_url = f"{self.mlb_stats_base}/people/{pitcher_id}/stats?stats=season&group=pitching&season={datetime.now().year}"
                    try:
                        response = requests.get(stats_url, timeout=5)
                        if response.status_code == 200:
                            stats_data = response.json()
                            splits = stats_data.get('stats', [{}])[0].get('splits', [])
                            if splits:
                                stat = splits[0].get('stat', {})
                                pitcher_info['wins'] = str(stat.get('wins', ''))
                                pitcher_info['losses'] = str(stat.get('losses', ''))
                                pitcher_info['era'] = str(stat.get('era', ''))
                    except:
                        pass  # If stats fail, continue with basic info
                        
        except Exception as e:
            print(f"Error extracting MLB Stats pitcher: {e}")
            
        return pitcher_info
    
    def _get_mlb_stats_game_pitchers(self, game_pk: int) -> Dict:
        """Get starting pitchers from game live data"""
        pitchers = {'away': {'name': 'TBD', 'wins': '', 'losses': '', 'era': ''}, 
                   'home': {'name': 'TBD', 'wins': '', 'losses': '', 'era': ''}}
        
        try:
            # Try the live game data endpoint
            live_url = f"{self.mlb_stats_base}/game/{game_pk}/linescore"
            response = requests.get(live_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                teams = data.get('teams', {})
                
                for team_type in ['away', 'home']:
                    team_data = teams.get(team_type, {})
                    pitcher = team_data.get('pitchers', {}).get('starter')
                    if pitcher:
                        # Get pitcher details
                        pitcher_id = pitcher
                        pitcher_url = f"{self.mlb_stats_base}/people/{pitcher_id}"
                        pitcher_response = requests.get(pitcher_url, timeout=5)
                        if pitcher_response.status_code == 200:
                            pitcher_data = pitcher_response.json()
                            people = pitcher_data.get('people', [])
                            if people:
                                pitchers[team_type]['name'] = people[0].get('fullName', 'TBD')
                                
        except Exception as e:
            pass  # Fail silently and keep TBD
            
        return pitchers
    
    def _get_mlb_stats_lineup(self, game_pk: int) -> Dict:
        """Get lineup for specific MLB Stats game"""
        lineups = {'away': [], 'home': []}
        
        try:
            # MLB Stats API game endpoint
            game_url = f"{self.mlb_stats_base}/game/{game_pk}/boxscore"
            response = requests.get(game_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            teams = data.get('teams', {})
            
            for team_type in ['away', 'home']:
                team_data = teams.get(team_type, {})
                batting_order = team_data.get('battingOrder', [])
                players = team_data.get('players', {})
                
                order_num = 1
                for player_id in batting_order[:9]:  # Get first 9 batters
                    player_key = f"ID{player_id}"
                    if player_key in players:
                        player = players[player_key]
                        player_info = {
                            'name': player.get('person', {}).get('fullName', 'Unknown'),
                            'position': player.get('position', {}).get('abbreviation', ''),
                            'batting_order': order_num
                        }
                        lineups[team_type].append(player_info)
                        order_num += 1
                        
        except Exception as e:
            print(f"MLB Stats Lineup Error for game {game_pk}: {e}")
            
        return lineups
    
    def format_lineup(self, lineup: List[Dict]) -> str:
        """Format lineup for display"""
        if not lineup:
            return "  No lineup available"
        
        formatted = []
        for player in lineup:
            order = player.get('batting_order', '')
            name = player.get('name', 'Unknown')
            position = player.get('position', '')
            formatted.append(f"  {order}. {name} ({position})")
        
        return '\n'.join(formatted) if formatted else "  No lineup available"
    
    def format_pitcher(self, pitcher: Dict) -> str:
        """Format pitcher information for display"""
        name = pitcher.get('name', 'TBD')
        wins = pitcher.get('wins', '')
        losses = pitcher.get('losses', '')
        era = pitcher.get('era', '')
        
        if wins and losses:
            return f"{name} ({wins}-{losses}, {era} ERA)"
        else:
            return name
    
    def get_best_available_data(self):
        """Get lineup data using fallback strategy: MLB Stats primary, ESPN fallback"""
        print("=" * 80)
        print(f"MLB GAMES, STARTING PITCHERS, AND LINEUPS FOR {self.today}")
        print("=" * 80)

        # Try MLB Stats API first (primary)
        print("Fetching from MLB Stats API (primary source)...")
        mlb_games = self.get_mlb_stats_games_and_lineups()

        if mlb_games:
            print(f"✅ MLB Stats API: Found {len(mlb_games)} games")
            return {'source': 'mlb_stats', 'games': mlb_games}

        # Fallback to ESPN if MLB Stats fails
        print("❌ MLB Stats API failed, trying ESPN API (fallback)...")
        time.sleep(1)  # Rate limiting

        espn_games = self.get_espn_games_and_lineups()

        if espn_games:
            print(f"✅ ESPN API: Found {len(espn_games)} games")
            return {'source': 'espn', 'games': espn_games}

        # Both failed
        print("❌ Both APIs failed - no data available")
        return {'source': 'none', 'games': {}}

    def display_all_lineups(self):
        """Main method to fetch and display all lineups using fallback strategy"""

        # Get best available data
        data_result = self.get_best_available_data()
        source = data_result['source']
        games = data_result['games']

        if not games:
            print("No games found from any data source")
            return

        # Display data with source information
        print(f"\nData Source: {source.upper()}")
        print("=" * 50)

        for game_key, game_data in games.items():
            print(f"\n{game_data['away_team']} @ {game_data['home_team']}")
            print(f"Game Time: {game_data['game_time']}")
            print(f"\nStarting Pitchers:")
            print(f"  {game_data['away_team']}: {self.format_pitcher(game_data['away_pitcher'])}")
            print(f"  {game_data['home_team']}: {self.format_pitcher(game_data['home_pitcher'])}")
            print(f"\n{game_data['away_team']} Lineup:")
            print(self.format_lineup(game_data['away_lineup']))
            print(f"\n{game_data['home_team']} Lineup:")
            print(self.format_lineup(game_data['home_lineup']))
            print("-" * 40)
    
    def save_to_json(self, filename: str = "mlb_lineups.json"):
        """Save lineup data to JSON file using fallback strategy"""
        data_result = self.get_best_available_data()

        all_data = {
            'date': self.today,
            'source': data_result['source'],
            'games': data_result['games']
        }

        # For backwards compatibility, also include the data under the source name
        if data_result['source'] == 'mlb_stats':
            all_data['mlb_stats'] = data_result['games']
            all_data['espn'] = {}  # Empty ESPN data
        elif data_result['source'] == 'espn':
            all_data['espn'] = data_result['games']
            all_data['mlb_stats'] = {}  # Empty MLB Stats data
        else:
            all_data['espn'] = {}
            all_data['mlb_stats'] = {}

        with open(filename, 'w') as f:
            json.dump(all_data, f, indent=2)

        print(f"\nData saved to {filename} (source: {data_result['source']})")


def main():
    """Main function to run the lineup fetcher"""
    fetcher = MLBLineupFetcher()
    
    # Display all lineups
    fetcher.display_all_lineups()
    
    # Optional: Save to JSON file
    save_option = input("\nWould you like to save the data to a JSON file? (y/n): ")
    if save_option.lower() == 'y':
        fetcher.save_to_json()
    
    print("\nDone!")


if __name__ == "__main__":
    main()