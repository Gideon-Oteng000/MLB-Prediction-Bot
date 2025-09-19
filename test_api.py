"""
Quick test script to debug SportsRadar API response
"""
import requests
import json
from datetime import datetime

SPORTRADAR_KEY = "3uCVRm9PhGc0VMvLafRDZpqcwdWrafLktHzEw3wr"
SPORTRADAR_BASE = "https://api.sportradar.com/mlb/trial/v8/en"

def test_api():
    print("Testing SportsRadar v8 API...")

    # Get today's schedule
    date_str = datetime.now().strftime('%Y/%m/%d')
    schedule_url = f"{SPORTRADAR_BASE}/games/{date_str}/schedule.json"

    print(f"Fetching schedule from: {schedule_url}")

    try:
        response = requests.get(
            schedule_url,
            params={'api_key': SPORTRADAR_KEY},
            timeout=10
        )

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Games found: {len(data.get('games', []))}")

            # Test one game's summary
            if data.get('games'):
                first_game = data['games'][0]
                game_id = first_game['id']
                print(f"\nTesting game ID: {game_id}")

                summary_url = f"{SPORTRADAR_BASE}/games/{game_id}/summary.json"
                summary_response = requests.get(
                    summary_url,
                    params={'api_key': SPORTRADAR_KEY},
                    timeout=10
                )

                print(f"Summary response status: {summary_response.status_code}")

                if summary_response.status_code == 200:
                    summary_data = summary_response.json()

                    if 'game' in summary_data:
                        game_data = summary_data['game']

                        # Check home team structure
                        if 'home' in game_data:
                            home = game_data['home']
                            print(f"\nHome team keys: {list(home.keys())}")

                            if 'players' in home:
                                players = home['players']
                                print(f"Players type: {type(players)}")
                                print(f"Players length: {len(players) if hasattr(players, '__len__') else 'N/A'}")

                                if isinstance(players, list) and players:
                                    print(f"First player: {players[0]}")
                                elif isinstance(players, dict):
                                    first_key = list(players.keys())[0]
                                    print(f"First player (key {first_key}): {players[first_key]}")

                            if 'lineup' in home:
                                lineup = home['lineup']
                                print(f"Lineup type: {type(lineup)}")
                                print(f"Lineup length: {len(lineup) if hasattr(lineup, '__len__') else 'N/A'}")
                                if lineup:
                                    print(f"First lineup entry: {lineup[0]}")

                else:
                    print(f"Summary error: {summary_response.text}")
        else:
            print(f"Schedule error: {response.text}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api()