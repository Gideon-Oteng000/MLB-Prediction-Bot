"""
Simplified test to check if lineup extraction is working
"""
import sys
sys.path.append('.')

from mlb_hr_clean_v4 import SportsRadarV8

def test_sportradar():
    print("Testing SportsRadar...")
    sr = SportsRadarV8()

    # Just test the first part
    print("Getting today's games...")
    try:
        games = sr.get_todays_games_with_lineups()
        print(f"Got {len(games)} games")

        if games:
            first_game = games[0]
            print(f"First game: {first_game['away_team']} @ {first_game['home_team']}")
            print(f"Home lineup: {len(first_game.get('home_lineup', []))} players")
            print(f"Away lineup: {len(first_game.get('away_lineup', []))} players")

            if first_game.get('home_lineup'):
                print("Home lineup players:")
                for i, player in enumerate(first_game['home_lineup'][:5], 1):
                    print(f"  {i}. {player['name']}")

            if first_game.get('away_lineup'):
                print("Away lineup players:")
                for i, player in enumerate(first_game['away_lineup'][:5], 1):
                    print(f"  {i}. {player['name']}")
        else:
            print("No games found")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sportradar()