"""
Test lineup extraction with SportsRadar v8 API
"""
import requests
import json
from datetime import datetime
import time

class TestSportsRadar:
    def __init__(self):
        self.SPORTRADAR_KEY = "3uCVRm9PhGc0VMvLafRDZpqcwdWrafLktHzEw3wr"
        self.SPORTRADAR_BASE = "https://api.sportradar.com/mlb/trial/v8/en"
        self.session = requests.Session()

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

    def test_one_game(self):
        """Test lineup extraction for one game"""
        print("Testing lineup extraction...")

        # Get today's schedule
        date_str = datetime.now().strftime('%Y/%m/%d')
        schedule_url = f"{self.SPORTRADAR_BASE}/games/{date_str}/schedule.json"

        try:
            response = self.session.get(
                schedule_url,
                params={'api_key': self.SPORTRADAR_KEY},
                timeout=10
            )

            if response.status_code != 200:
                print(f"❌ Error getting schedule: {response.status_code}")
                return

            schedule_data = response.json()

            if 'games' not in schedule_data:
                print("❌ No games found")
                return

            all_games = schedule_data['games']
            print(f"📅 Found {len(all_games)} total games")

            # Filter to scheduled games only
            scheduled_games = [g for g in all_games if g.get('status') in ['scheduled', 'created']]
            print(f"⏰ {len(scheduled_games)} games not yet started")

            if not scheduled_games:
                print("❌ No scheduled games")
                return

            # Test first scheduled game
            game = scheduled_games[0]
            game_id = game['id']
            home_team = game['home']['name']
            away_team = game['away']['name']

            print(f"\n🎮 Testing: {away_team} @ {home_team}")

            # Get game summary with lineups
            summary_url = f"{self.SPORTRADAR_BASE}/games/{game_id}/summary.json"

            time.sleep(1.2)  # Rate limiting

            summary_response = self.session.get(
                summary_url,
                params={'api_key': self.SPORTRADAR_KEY},
                timeout=10
            )

            if summary_response.status_code != 200:
                print(f"❌ Could not get details: {summary_response.status_code}")
                return

            summary_data = summary_response.json()

            if 'game' not in summary_data:
                print("❌ No game data")
                return

            game_data = summary_data['game']

            success = False

            # Test home team lineup
            if 'home' in game_data:
                home = game_data['home']
                if 'lineup' in home and 'players' in home:
                    print(f"\n🏠 Processing home lineup for {home_team}")
                    home_lineup = self._extract_lineup(home['lineup'], home['players'])
                    print(f"✅ Home lineup extracted: {len(home_lineup)} players")
                    for i, player in enumerate(home_lineup[:5], 1):
                        print(f"   {i}. {player['name']} (pos: {player['position']})")
                    success = True
                else:
                    print(f"❌ Home lineup/players data missing")

            # Test away team lineup
            if 'away' in game_data:
                away = game_data['away']
                if 'lineup' in away and 'players' in away:
                    print(f"\n✈️ Processing away lineup for {away_team}")
                    away_lineup = self._extract_lineup(away['lineup'], away['players'])
                    print(f"✅ Away lineup extracted: {len(away_lineup)} players")
                    for i, player in enumerate(away_lineup[:5], 1):
                        print(f"   {i}. {player['name']} (pos: {player['position']})")
                    success = True
                else:
                    print(f"❌ Away lineup/players data missing")

            if success:
                print(f"\n🎉 SUCCESS: Lineup extraction working!")
            else:
                print(f"\n❌ FAILED: No lineups extracted")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    tester = TestSportsRadar()
    tester.test_one_game()