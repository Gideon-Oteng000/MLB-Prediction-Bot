"""
MLB Home Run Prediction Model v5.0 - Using SportsRadar v8
Clean version with proper lineup extraction
Only shows: Hitter, Pitcher, Teams, HR_Probability
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import statsapi
import pybaseball as pyb
import sqlite3
import math
import warnings
warnings.filterwarnings('ignore')

# Enable PyBaseball cache
pyb.cache.enable()

class Config:
    """Configuration"""
    
    # SportsRadar v8 API
    SPORTRADAR_KEY = "3uCVRm9PhGc0VMvLafRDZpqcwdWrafLktHzEw3wr"
    SPORTRADAR_BASE = "https://api.sportradar.com/mlb/trial/v8/en"
    
    # Weather API (optional)
    WEATHER_API_KEY = "YOUR_OPENWEATHER_KEY"
    
    # Cache
    CACHE_DB = "mlb_v5_cache.db"
    
    # Park factors
    PARK_FACTORS = {
        'Coors Field': 1.40,
        'Great American Ball Park': 1.25,
        'Yankee Stadium': 1.20,
        'Oriole Park at Camden Yards': 1.18,
        'Globe Life Field': 1.15,
        'Citizens Bank Park': 1.12,
        'Fenway Park': 1.10,
        'Guaranteed Rate Field': 1.08,
        'Truist Park': 1.08,
        'Minute Maid Park': 1.05,
        'Chase Field': 1.05,
        'Dodger Stadium': 1.03,
        'Oracle Park': 0.92,
        'T-Mobile Park': 0.90,
        'Petco Park': 0.95,
        'default': 1.00
    }
    
    # Stadium data for weather
    STADIUMS = {
        'Yankee Stadium': {'lat': 40.8296, 'lon': -73.9262, 'elevation': 54},
        'Fenway Park': {'lat': 42.3467, 'lon': -71.0972, 'elevation': 20},
        'Coors Field': {'lat': 39.7559, 'lon': -104.9942, 'elevation': 5280},
        'Dodger Stadium': {'lat': 34.0739, 'lon': -118.2400, 'elevation': 512},
        'Oracle Park': {'lat': 37.7786, 'lon': -122.3893, 'elevation': 0},
        'default': {'lat': 40.0, 'lon': -95.0, 'elevation': 500}
    }

class SportsRadarV8:
    """SportsRadar v8 API handler"""
    
    def __init__(self):
        self.config = Config()
        self.session = requests.Session()
        
    def get_todays_games_with_lineups(self):
        """Get today's games with lineups from SportsRadar v8"""
        print("\nFetching games from SportsRadar v8...")
        
        # Get today's schedule
        date_str = datetime.now().strftime('%Y/%m/%d')
        schedule_url = f"{self.config.SPORTRADAR_BASE}/games/{date_str}/schedule.json"
        
        try:
            response = self.session.get(
                schedule_url,
                params={'api_key': self.config.SPORTRADAR_KEY},
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"   [ERROR] Error getting schedule: {response.status_code}")
                return []
            
            schedule_data = response.json()
            
            if 'games' not in schedule_data:
                print("   [ERROR] No games found")
                return []
            
            all_games = schedule_data['games']
            print(f"   [SCHEDULE] Found {len(all_games)} total games")
            
            # Filter to scheduled games only
            scheduled_games = [g for g in all_games if g.get('status') in ['scheduled', 'created']]
            print(f"   [TIME] {len(scheduled_games)} games not yet started")
            
            games_with_lineups = []
            
            # Get details for each scheduled game
            for game in scheduled_games:
                time.sleep(1.2)  # Rate limiting
                
                game_id = game['id']
                home_team = game['home']['name']
                away_team = game['away']['name']
                
                # Get game summary with lineups
                summary_url = f"{self.config.SPORTRADAR_BASE}/games/{game_id}/summary.json"
                
                try:
                    summary_response = self.session.get(
                        summary_url,
                        params={'api_key': self.config.SPORTRADAR_KEY},
                        timeout=10
                    )
                    
                    if summary_response.status_code != 200:
                        print(f"   [WARNING] Could not get details for {away_team} @ {home_team}")
                        continue
                    
                    summary_data = summary_response.json()
                    
                    if 'game' not in summary_data:
                        continue
                    
                    game_data = summary_data['game']
                    
                    game_info = {
                        'game_id': game_id,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_team_abbr': self._get_team_abbr(home_team),
                        'away_team_abbr': self._get_team_abbr(away_team),
                        'venue': game_data.get('venue', {}).get('name', ''),
                        'home_lineup': [],
                        'away_lineup': [],
                        'home_pitcher': None,
                        'away_pitcher': None
                    }
                    
                    # Process home team
                    if 'home' in game_data:
                        home = game_data['home']
                        
                        # Get pitcher
                        if 'probable_pitcher' in home:
                            pitcher = home['probable_pitcher']
                            game_info['home_pitcher'] = {
                                'name': pitcher.get('full_name', pitcher.get('last_name', 'TBD')),
                                'id': pitcher.get('id')
                            }
                        
                        # Get lineup
                        if 'lineup' in home and 'players' in home:
                            game_info['home_lineup'] = self._extract_lineup(
                                home['lineup'],
                                home['players']
                            )
                    
                    # Process away team
                    if 'away' in game_data:
                        away = game_data['away']
                        
                        # Get pitcher
                        if 'probable_pitcher' in away:
                            pitcher = away['probable_pitcher']
                            game_info['away_pitcher'] = {
                                'name': pitcher.get('full_name', pitcher.get('last_name', 'TBD')),
                                'id': pitcher.get('id')
                            }
                        
                        # Get lineup
                        if 'lineup' in away and 'players' in away:
                            game_info['away_lineup'] = self._extract_lineup(
                                away['lineup'],
                                away['players']
                            )
                    
                    # Add game if we have at least partial data
                    if (game_info['home_lineup'] or game_info['away_lineup']) or \
                       (game_info['home_pitcher'] or game_info['away_pitcher']):
                        games_with_lineups.append(game_info)
                        print(f"   [SUCCESS] {away_team} @ {home_team}: Lineups/Pitchers available")
                    
                except Exception as e:
                    print(f"   [WARNING] Error getting {away_team} @ {home_team}: {e}")
                    continue
            
            print(f"\n   [DATA] Successfully retrieved {len(games_with_lineups)} games with data")
            return games_with_lineups
            
        except Exception as e:
            print(f"   [ERROR] Error fetching schedule: {e}")
            return []
    
    def _extract_lineup(self, lineup_data, players_data):
        """Extract player names from lineup IDs"""
        lineup = []


        # Create player ID to name mapping
        player_map = {}

        # Handle both list and dict formats
        if isinstance(players_data, list):
            # If it's a list, each item should be a player object
            for player_info in players_data:
                if isinstance(player_info, dict):
                    player_id = player_info.get('id')
                    player_name = player_info.get('full_name') or player_info.get('name', 'Unknown')
                    if player_id:
                        player_map[player_id] = player_name
        elif isinstance(players_data, dict):
            # Original logic for dict format
            for player_id, player_info in players_data.items():
                player_map[player_id] = player_info.get('full_name', 'Unknown')

        # Extract lineup with names
        for entry in lineup_data:
            if entry.get('position') == 1:  # Skip pitcher
                continue

            player_id = entry.get('id')
            if player_id and player_id in player_map:
                lineup.append({
                    'name': player_map[player_id],
                    'order': entry.get('order', 0),
                    'position': entry.get('position', 0)
                })

        # Sort by batting order
        lineup.sort(key=lambda x: x['order'])

        return lineup[:9]  # Return top 9 batters
    
    def _get_team_abbr(self, team_name):
        """Get team abbreviation"""
        abbr_map = {
            'Yankees': 'NYY', 'Red Sox': 'BOS', 'Rays': 'TB', 'Orioles': 'BAL', 'Blue Jays': 'TOR',
            'Guardians': 'CLE', 'Twins': 'MIN', 'White Sox': 'CHW', 'Royals': 'KC', 'Tigers': 'DET',
            'Astros': 'HOU', 'Athletics': 'OAK', 'Rangers': 'TEX', 'Angels': 'LAA', 'Mariners': 'SEA',
            'Mets': 'NYM', 'Braves': 'ATL', 'Phillies': 'PHI', 'Marlins': 'MIA', 'Nationals': 'WSN',
            'Brewers': 'MIL', 'Cardinals': 'STL', 'Cubs': 'CHC', 'Reds': 'CIN', 'Pirates': 'PIT',
            'Dodgers': 'LAD', 'Giants': 'SF', 'Padres': 'SD', 'Rockies': 'COL', 'Diamondbacks': 'ARI'
        }
        
        for key, value in abbr_map.items():
            if key in team_name:
                return value
        
        return team_name[:3].upper()

class LineupManager:
    """Unified lineup manager using multiple API sources"""

    def __init__(self):
        self.config = Config()
        self.session = requests.Session()
        self.processed_games = set()  # Track processed game IDs to avoid duplicates

    def get_all_todays_games(self):
        """Get all today's games with lineups from multiple sources"""
        print("\n" + "=" * 60)
        print("FETCHING ALL TODAY'S GAMES WITH LINEUPS")
        print("=" * 60)

        all_games = []
        self.processed_games.clear()

        # 1. Primary source: SportsRadar v8
        print("\n[1/4] Fetching from SportsRadar v8...")
        sportradar_games = self._get_sportradar_games()
        all_games.extend(sportradar_games)
        print(f"   [SUCCESS] SportsRadar: {len(sportradar_games)} games")

        # 2. Fill gaps with MLB StatsAPI
        print("\n[2/4] Filling gaps with MLB StatsAPI...")
        statsapi_games = self._get_statsapi_games()
        all_games.extend(statsapi_games)
        print(f"   [SUCCESS] StatsAPI: {len(statsapi_games)} additional games")

        # 3. Fill remaining gaps with ESPN
        print("\n[3/4] Filling gaps with ESPN...")
        espn_games = self._get_espn_games()
        all_games.extend(espn_games)
        print(f"   [SUCCESS] ESPN: {len(espn_games)} additional games")

        # 4. Final fallback: Rotogrinders
        print("\n[4/4] Final check with Rotogrinders...")
        roto_games = self._get_rotogrinders_games()
        all_games.extend(roto_games)
        print(f"   [SUCCESS] Rotogrinders: {len(roto_games)} additional games")

        print(f"\n" + "=" * 60)
        print(f"TOTAL GAMES COLLECTED: {len(all_games)}")
        print("=" * 60)

        return all_games

    def _get_sportradar_games(self):
        """Get games from SportsRadar v8 (existing logic)"""
        try:
            # Get today's schedule
            date_str = datetime.now().strftime('%Y/%m/%d')
            schedule_url = f"{self.config.SPORTRADAR_BASE}/games/{date_str}/schedule.json"

            response = self.session.get(
                schedule_url,
                params={'api_key': self.config.SPORTRADAR_KEY},
                timeout=10
            )

            if response.status_code != 200:
                print(f"   [ERROR] SportsRadar error: {response.status_code}")
                return []

            schedule_data = response.json()
            if 'games' not in schedule_data:
                return []

            all_games = schedule_data['games']
            scheduled_games = [g for g in all_games if g.get('status') in ['scheduled', 'created']]

            games_with_lineups = []

            for game in scheduled_games:
                time.sleep(1.2)  # Rate limiting

                game_id = game['id']
                if game_id in self.processed_games:
                    continue

                # Get game summary
                summary_url = f"{self.config.SPORTRADAR_BASE}/games/{game_id}/summary.json"
                summary_response = self.session.get(
                    summary_url,
                    params={'api_key': self.config.SPORTRADAR_KEY},
                    timeout=10
                )

                if summary_response.status_code != 200:
                    continue

                summary_data = summary_response.json()
                if 'game' not in summary_data:
                    continue

                game_data = summary_data['game']

                game_info = {
                    'game_id': game_id,
                    'home_team': game['home']['name'],
                    'away_team': game['away']['name'],
                    'home_team_abbr': self._get_team_abbr(game['home']['name']),
                    'away_team_abbr': self._get_team_abbr(game['away']['name']),
                    'venue': game_data.get('venue', {}).get('name', ''),
                    'home_lineup': [],
                    'away_lineup': [],
                    'home_pitcher': None,
                    'away_pitcher': None,
                    'source': 'SportsRadar'
                }

                # Extract lineups and pitchers (existing logic)
                if 'home' in game_data:
                    home = game_data['home']
                    if 'probable_pitcher' in home:
                        pitcher = home['probable_pitcher']
                        game_info['home_pitcher'] = {
                            'name': pitcher.get('full_name', pitcher.get('last_name', 'TBD')),
                            'id': pitcher.get('id')
                        }
                    if 'lineup' in home and 'players' in home:
                        game_info['home_lineup'] = self._extract_sportradar_lineup(
                            home['lineup'], home['players']
                        )

                if 'away' in game_data:
                    away = game_data['away']
                    if 'probable_pitcher' in away:
                        pitcher = away['probable_pitcher']
                        game_info['away_pitcher'] = {
                            'name': pitcher.get('full_name', pitcher.get('last_name', 'TBD')),
                            'id': pitcher.get('id')
                        }
                    if 'lineup' in away and 'players' in away:
                        game_info['away_lineup'] = self._extract_sportradar_lineup(
                            away['lineup'], away['players']
                        )

                self.processed_games.add(game_id)
                games_with_lineups.append(game_info)

            return games_with_lineups

        except Exception as e:
            print(f"   [ERROR] SportsRadar error: {e}")
            return []

    def _get_statsapi_games(self):
        """Get missing games from MLB StatsAPI"""
        try:
            today = datetime.now().strftime('%m/%d/%Y')
            schedule = statsapi.schedule(date=today)

            games = []
            for game_data in schedule:
                if game_data['status'] in ['Final', 'Game Over', 'Completed']:
                    continue

                game_id = str(game_data['game_id'])
                if game_id in self.processed_games:
                    continue

                game_info = {
                    'game_id': game_id,
                    'home_team': game_data['home_name'],
                    'away_team': game_data['away_name'],
                    'home_team_abbr': self._get_team_abbr(game_data['home_name']),
                    'away_team_abbr': self._get_team_abbr(game_data['away_name']),
                    'venue': game_data.get('venue_name', ''),
                    'home_lineup': self._get_frequent_starters(game_data['home_id'], game_data['home_name']),
                    'away_lineup': self._get_frequent_starters(game_data['away_id'], game_data['away_name']),
                    'home_pitcher': {'name': 'TBD'},
                    'away_pitcher': {'name': 'TBD'},
                    'source': 'StatsAPI'
                }

                self.processed_games.add(game_id)
                games.append(game_info)

            return games

        except Exception as e:
            print(f"   [ERROR] StatsAPI error: {e}")
            return []

    def _get_espn_games(self):
        """Get missing games from ESPN API"""
        try:
            date_str = datetime.now().strftime('%Y%m%d')
            espn_url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date_str}"

            response = self.session.get(espn_url, timeout=10)
            if response.status_code != 200:
                return []

            data = response.json()
            games = []

            for event in data.get('events', []):
                game_id = f"espn_{event['id']}"
                if game_id in self.processed_games:
                    continue

                if event['status']['type']['state'] in ['post']:
                    continue

                competitions = event.get('competitions', [{}])[0]
                competitors = competitions.get('competitors', [])

                if len(competitors) < 2:
                    continue

                home_team = next((c for c in competitors if c['homeAway'] == 'home'), {})
                away_team = next((c for c in competitors if c['homeAway'] == 'away'), {})

                game_info = {
                    'game_id': game_id,
                    'home_team': home_team.get('team', {}).get('displayName', ''),
                    'away_team': away_team.get('team', {}).get('displayName', ''),
                    'home_team_abbr': home_team.get('team', {}).get('abbreviation', ''),
                    'away_team_abbr': away_team.get('team', {}).get('abbreviation', ''),
                    'venue': competitions.get('venue', {}).get('fullName', ''),
                    'home_lineup': self._generate_generic_lineup(home_team.get('team', {}).get('displayName', '')),
                    'away_lineup': self._generate_generic_lineup(away_team.get('team', {}).get('displayName', '')),
                    'home_pitcher': {'name': 'TBD'},
                    'away_pitcher': {'name': 'TBD'},
                    'source': 'ESPN'
                }

                self.processed_games.add(game_id)
                games.append(game_info)

            return games

        except Exception as e:
            print(f"   [ERROR] ESPN error: {e}")
            return []

    def _get_rotogrinders_games(self):
        """Get missing games from Rotogrinders API"""
        try:
            roto_url = "https://rotogrinders.com/api/lineups?sport=MLB"
            response = self.session.get(roto_url, timeout=10)

            if response.status_code != 200:
                return []

            data = response.json()
            games = []

            for game in data.get('games', []):
                game_id = f"roto_{game.get('id', '')}"
                if game_id in self.processed_games:
                    continue

                game_info = {
                    'game_id': game_id,
                    'home_team': game.get('home_team', {}).get('name', ''),
                    'away_team': game.get('away_team', {}).get('name', ''),
                    'home_team_abbr': game.get('home_team', {}).get('code', ''),
                    'away_team_abbr': game.get('away_team', {}).get('code', ''),
                    'venue': game.get('venue', ''),
                    'home_lineup': self._extract_roto_lineup(game.get('home_lineup', [])),
                    'away_lineup': self._extract_roto_lineup(game.get('away_lineup', [])),
                    'home_pitcher': {'name': game.get('home_pitcher', {}).get('name', 'TBD')},
                    'away_pitcher': {'name': game.get('away_pitcher', {}).get('name', 'TBD')},
                    'source': 'Rotogrinders'
                }

                self.processed_games.add(game_id)
                games.append(game_info)

            return games

        except Exception as e:
            print(f"   [ERROR] Rotogrinders error: {e}")
            return []

    def _extract_sportradar_lineup(self, lineup_data, players_data):
        """Extract lineup from SportsRadar format (existing logic)"""
        lineup = []

        player_map = {}
        if isinstance(players_data, list):
            for player_info in players_data:
                if isinstance(player_info, dict):
                    player_id = player_info.get('id')
                    player_name = player_info.get('full_name') or player_info.get('name', 'Unknown')
                    if player_id:
                        player_map[player_id] = player_name
        elif isinstance(players_data, dict):
            for player_id, player_info in players_data.items():
                player_map[player_id] = player_info.get('full_name', 'Unknown')

        for entry in lineup_data:
            if entry.get('position') == 1:  # Skip pitcher
                continue

            player_id = entry.get('id')
            if player_id and player_id in player_map:
                lineup.append({
                    'name': player_map[player_id],
                    'order': entry.get('order', 0),
                    'position': entry.get('position', 0)
                })

        lineup.sort(key=lambda x: x['order'])
        return lineup[:9]

    def _get_frequent_starters(self, team_id, team_name):
        """Get frequent starters from StatsAPI (existing logic)"""
        try:
            roster_data = statsapi.get('team_roster', {'teamId': team_id})
            roster = {p['person']['fullName']: p['person']['id'] for p in roster_data.get('roster', [])}

            end_date = datetime.now()
            start_date = end_date - timedelta(days=15)

            schedule = statsapi.schedule(
                team=team_id,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            player_counts = {}

            for game in schedule[-10:]:
                if game['status'] != 'Final':
                    continue

                try:
                    boxscore = statsapi.boxscore_data(game['game_id'])
                    is_home = game['home_name'] == team_name
                    team_key = 'home' if is_home else 'away'

                    for player_id, player in boxscore[team_key]['players'].items():
                        if player.get('stats', {}).get('batting', {}):
                            name = player['person']['fullName']
                            if name in roster:
                                player_counts[name] = player_counts.get(name, 0) + 1
                except:
                    continue

            sorted_players = sorted(player_counts.items(), key=lambda x: x[1], reverse=True)

            lineup = []
            for i, (name, count) in enumerate(sorted_players[:9], 1):
                lineup.append({'name': name, 'order': i})

            while len(lineup) < 9:
                lineup.append({'name': f'{team_name} Player {len(lineup)+1}', 'order': len(lineup)+1})

            return lineup

        except:
            return self._generate_generic_lineup(team_name)

    def _generate_generic_lineup(self, team_name):
        """Generate a generic lineup when no data is available"""
        return [{'name': f'{team_name} Player {i}', 'order': i} for i in range(1, 10)]

    def _extract_roto_lineup(self, lineup_data):
        """Extract lineup from Rotogrinders format"""
        lineup = []
        for i, player in enumerate(lineup_data[:9], 1):
            lineup.append({
                'name': player.get('name', f'Player {i}'),
                'order': i,
                'position': player.get('position', i)
            })
        return lineup

    def _get_team_abbr(self, team_name):
        """Get team abbreviation"""
        abbr_map = {
            'Yankees': 'NYY', 'Red Sox': 'BOS', 'Rays': 'TB', 'Orioles': 'BAL', 'Blue Jays': 'TOR',
            'Guardians': 'CLE', 'Twins': 'MIN', 'White Sox': 'CHW', 'Royals': 'KC', 'Tigers': 'DET',
            'Astros': 'HOU', 'Athletics': 'OAK', 'Rangers': 'TEX', 'Angels': 'LAA', 'Mariners': 'SEA',
            'Mets': 'NYM', 'Braves': 'ATL', 'Phillies': 'PHI', 'Marlins': 'MIA', 'Nationals': 'WSN',
            'Brewers': 'MIL', 'Cardinals': 'STL', 'Cubs': 'CHC', 'Reds': 'CIN', 'Pirates': 'PIT',
            'Dodgers': 'LAD', 'Giants': 'SF', 'Padres': 'SD', 'Rockies': 'COL', 'Diamondbacks': 'ARI'
        }

        for key, value in abbr_map.items():
            if key in team_name:
                return value

        return team_name[:3].upper()

class SimpleCache:
    """Simple caching for player stats"""
    
    def __init__(self):
        self.config = Config()
        self.conn = sqlite3.connect(self.config.CACHE_DB)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_stats (
                player_name TEXT PRIMARY KEY,
                stats_json TEXT,
                cached_at TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def get_cached_stats(self, player_name):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT stats_json, cached_at 
            FROM player_stats 
            WHERE player_name = ?
        ''', (player_name,))
        
        result = cursor.fetchone()
        if result:
            stats_json, cached_at = result
            cached_time = datetime.fromisoformat(cached_at)
            
            if datetime.now() - cached_time < timedelta(hours=24):
                return json.loads(stats_json)
        
        return None
    
    def save_stats(self, player_name, stats):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO player_stats (player_name, stats_json, cached_at)
            VALUES (?, ?, ?)
        ''', (player_name, json.dumps(stats), datetime.now().isoformat()))
        self.conn.commit()

class StatsCollector:
    """Get player statistics"""
    
    def __init__(self):
        self.cache = SimpleCache()
    
    def get_batter_stats(self, player_name):
        """Get batter statistics with caching"""
        
        # Skip generic players
        if 'Player' in player_name or 'TBD' in player_name:
            return self._default_batter_stats()
        
        # Check cache
        cached = self.cache.get_cached_stats(player_name)
        if cached:
            return cached
        
        try:
            # Get from MLB Stats API
            search = statsapi.lookup_player(player_name)
            
            if not search:
                return self._default_batter_stats()
            
            player_id = search[0]['id']
            stats = statsapi.player_stat_data(player_id, group='hitting', type='season')
            
            if stats.get('stats'):
                current = stats['stats'][0]['stats']
                
                result = {
                    'home_runs': int(current.get('homeRuns', 0)),
                    'at_bats': int(current.get('atBats', 1)),
                    'ops': float(current.get('ops', 0.700)),
                    'iso': float(current.get('slg', 0.400)) - float(current.get('avg', 0.250)),
                    'barrel_rate': 0.08  # Would need Statcast
                }
                
                # Cache it
                self.cache.save_stats(player_name, result)
                return result
                
        except:
            pass
        
        return self._default_batter_stats()
    
    def get_pitcher_stats(self, pitcher_name):
        """Get pitcher statistics"""
        
        if not pitcher_name or pitcher_name == 'TBD':
            return self._default_pitcher_stats()
        
        try:
            search = statsapi.lookup_player(pitcher_name)
            
            if not search:
                return self._default_pitcher_stats()
            
            player_id = search[0]['id']
            stats = statsapi.player_stat_data(player_id, group='pitching', type='season')
            
            if stats.get('stats'):
                current = stats['stats'][0]['stats']
                
                return {
                    'era': float(current.get('era', 4.50)),
                    'hr_per_9': float(current.get('homeRunsPer9', 1.3)),
                    'whip': float(current.get('whip', 1.35))
                }
                
        except:
            pass
        
        return self._default_pitcher_stats()
    
    def _default_batter_stats(self):
        return {
            'home_runs': 15,
            'at_bats': 400,
            'ops': 0.740,
            'iso': 0.165,
            'barrel_rate': 0.075
        }
    
    def _default_pitcher_stats(self):
        return {
            'era': 4.50,
            'hr_per_9': 1.3,
            'whip': 1.35
        }

class HRProbabilityModel:
    """Calculate HR probability"""
    
    def __init__(self):
        self.config = Config()
        self.stats = StatsCollector()
    
    def calculate_probability(self, batter_name, pitcher_name, venue):
        """Calculate HR probability for a batter"""
        
        # Get stats
        batter_stats = self.stats.get_batter_stats(batter_name)
        pitcher_stats = self.stats.get_pitcher_stats(pitcher_name)
        
        # Base rate (MLB average ~3.3%)
        base_rate = 0.033
        
        # Batter power factor
        hr_rate = batter_stats['home_runs'] / max(batter_stats['at_bats'], 100)
        batter_mult = hr_rate / 0.033
        batter_mult = min(max(batter_mult, 0.3), 3.0)
        
        # ISO bonus
        iso = batter_stats['iso']
        if iso > 0.250:
            batter_mult *= 1.35
        elif iso > 0.200:
            batter_mult *= 1.15
        elif iso < 0.120:
            batter_mult *= 0.7
        
        # Pitcher vulnerability
        pitcher_mult = 1.0
        if pitcher_stats:
            hr_per_9 = pitcher_stats['hr_per_9']
            if hr_per_9 > 1.5:
                pitcher_mult = 1.3
            elif hr_per_9 > 1.2:
                pitcher_mult = 1.1
            elif hr_per_9 < 0.9:
                pitcher_mult = 0.7
            
            # ERA factor
            era = pitcher_stats['era']
            if era > 5.0:
                pitcher_mult *= 1.15
            elif era < 3.0:
                pitcher_mult *= 0.85
        
        # Park factor
        park_mult = self.config.PARK_FACTORS.get(venue, 1.0)
        
        # Weather multiplier (simplified since weather API is optional)
        weather_mult = 1.0
        
        # Elevation bonus for Coors Field
        if 'Coors' in venue:
            weather_mult = 1.15  # Additional altitude boost
        
        # Calculate final probability
        total_mult = batter_mult * pitcher_mult * park_mult * weather_mult
        hr_prob = base_rate * total_mult
        
        # Cap at 15%
        return min(hr_prob, 0.15)

class CleanHRPredictorV5:
    """Main prediction system using unified LineupManager"""

    def __init__(self):
        self.lineup_manager = LineupManager()
        self.model = HRProbabilityModel()

    def run_predictions(self):
        """Generate clean HR predictions"""
        print("=" * 80)
        print("MLB HOME RUN PREDICTIONS v5.0 - ENHANCED")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 80)

        # Get all games using unified LineupManager
        games = self.lineup_manager.get_all_todays_games()

        if not games:
            print("\n[ERROR] No games available from any source")
            print("Please check your internet connection and API availability")
            return pd.DataFrame()

        # Count games by source
        source_counts = {}
        for game in games:
            source = game.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        print(f"\n[GAME] Processing {len(games)} games from multiple sources:")
        for source, count in source_counts.items():
            print(f"   [DATA] {source}: {count} games")

        all_predictions = []
        games_processed = 0
        
        for game_num, game in enumerate(games, 1):
            
            # Process home batters vs away pitcher
            if game['home_lineup'] and game['away_pitcher']:
                games_processed += 1
                pitcher_name = game['away_pitcher']['name']
                
                for batter in game['home_lineup'][:9]:
                    if 'Player' in batter['name'] and 'Projected' not in source:
                        continue  # Skip generic players unless using projections
                    
                    prob = self.model.calculate_probability(
                        batter['name'],
                        pitcher_name,
                        game['venue']
                    )
                    
                    all_predictions.append({
                        'Hitter': batter['name'],
                        'Pitcher': pitcher_name,
                        'Teams': f"{game['home_team_abbr']} vs {game['away_team_abbr']}",
                        'HR_Probability': round(prob * 100, 2)
                    })
            
            # Process away batters vs home pitcher
            if game['away_lineup'] and game['home_pitcher']:
                pitcher_name = game['home_pitcher']['name']
                
                for batter in game['away_lineup'][:9]:
                    if 'Player' in batter['name'] and 'Projected' not in source:
                        continue
                    
                    prob = self.model.calculate_probability(
                        batter['name'],
                        pitcher_name,
                        game['venue']
                    )
                    
                    all_predictions.append({
                        'Hitter': batter['name'],
                        'Pitcher': pitcher_name,
                        'Teams': f"{game['away_team_abbr']} @ {game['home_team_abbr']}",
                        'HR_Probability': round(prob * 100, 2)
                    })
        
        if not all_predictions:
            print("\n[ERROR] No valid predictions generated")
            print("Lineups may not be posted yet. Try again closer to game time.")
            return pd.DataFrame()
        
        # Remove duplicate players (keep highest probability prediction)
        deduplicated_predictions = self._deduplicate_predictions(all_predictions)

        # Create DataFrame and sort
        df = pd.DataFrame(deduplicated_predictions)
        df = df.sort_values('HR_Probability', ascending=False).reset_index(drop=True)
        
        # Keep only top 10
        df = df.head(10)
        
        print("\n" + "=" * 80)
        print("TOP 10 HOME RUN PREDICTIONS")
        print("=" * 80)
        print(f"{'Rank':<5} {'Hitter':<20} {'Pitcher':<20} {'Teams':<12} {'Prob %':<8}")
        print("-" * 80)
        
        for idx, row in df.iterrows():
            print(f"{idx+1:<5} {row['Hitter'][:19]:<20} {row['Pitcher'][:19]:<20} {row['Teams']:<12} {row['HR_Probability']:<8.1f}%")
        
        # Save to CSV (clean format)
        filename = f"hr_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)
        
        print("\n" + "=" * 80)
        print(f"[SUCCESS] Predictions saved to: {filename}")
        print(f"[DATA] Games processed: {games_processed}")
        print(f"[INFO] Total predictions: {len(all_predictions)}")
        print("[TARGET] Data sources:")
        for source, count in source_counts.items():
            print(f"   {source}: {count} games")

        print("\n[WARNING] Note: Lineups subject to change. Bets void if player doesn't start.")
        print("[INFO] Enhanced system now processes ALL daily games with multiple API sources")

        return df

    def _deduplicate_predictions(self, all_predictions):
        """Remove duplicate players, keeping the highest probability prediction for each"""
        from collections import defaultdict

        # Group predictions by player name
        player_predictions = defaultdict(list)

        for prediction in all_predictions:
            player_name = prediction['Hitter']
            player_predictions[player_name].append(prediction)

        # For each player, keep only the highest probability prediction
        deduplicated = []

        for player_name, predictions in player_predictions.items():
            if len(predictions) == 1:
                # Only one prediction, keep it
                deduplicated.append(predictions[0])
            else:
                # Multiple predictions, keep the highest probability
                best_prediction = max(predictions, key=lambda x: x['HR_Probability'])

                # Add note about multiple matchups
                num_matchups = len(predictions)
                best_prediction['Teams'] += f" ({num_matchups} games)"

                deduplicated.append(best_prediction)

        return deduplicated

# Main execution
if __name__ == "__main__":
    print("MLB HOME RUN PREDICTION MODEL v5.0")
    print("Using SportsRadar v8 API with proper lineup extraction")
    print("-" * 80)
    
    try:
        predictor = CleanHRPredictorV5()
        predictions = predictor.run_predictions()
        
        if predictions.empty:
            print("\n[TIP] Tips for best results:")
            print("• Run 2-3 hours before first pitch")
            print("• Lineups typically post around 3-4 PM for night games")
            print("• Check back later if lineups aren't available")
            
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nPlease check:")
        print("1. Internet connection")
        print("2. API rate limits")
        print("3. Required packages installed (statsapi, pybaseball, pandas)")