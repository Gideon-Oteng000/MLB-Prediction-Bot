"""
Test SportsRadar v8 API to see lineup structure
"""

import requests
import json
from datetime import datetime
import time

# Your SportsRadar key
SPORTRADAR_KEY = "3uCVRm9PhGc0VMvLafRDZpqcwdWrafLktHzEw3wr"

def test_v8_api():
    """Test v8 API to see what data we get"""
    
    print("=" * 80)
    print("TESTING SPORTRADAR v8 API")
    print("=" * 80)
    
    # First, get today's schedule using v8
    date_str = datetime.now().strftime('%Y/%m/%d')
    schedule_url = f"https://api.sportradar.com/mlb/trial/v8/en/games/{date_str}/schedule.json?api_key={SPORTRADAR_KEY}"
    
    print(f"\n1. Getting today's schedule from v8...")
    print(f"   URL: {schedule_url}")
    
    try:
        response = requests.get(schedule_url)
        print(f"   Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Error: {response.text}")
            return
        
        schedule_data = response.json()
        
        if 'games' not in schedule_data:
            print("   No games found")
            return
        
        games = schedule_data['games']
        print(f"   ✓ Found {len(games)} games")
        
        # Take first scheduled game
        test_game = None
        for game in games:
            if game.get('status') in ['scheduled', 'created']:
                test_game = game
                break
        
        if not test_game:
            print("   No scheduled games to test")
            return
        
        game_id = test_game['id']
        home_team = test_game['home']['name']
        away_team = test_game['away']['name']
        
        print(f"\n2. Testing game summary for: {away_team} @ {home_team}")
        print(f"   Game ID: {game_id}")
        
        time.sleep(1.5)  # Rate limiting
        
        # Now get the game summary
        summary_url = f"https://api.sportradar.com/mlb/trial/v8/en/games/{game_id}/summary.json?api_key={SPORTRADAR_KEY}"
        
        print(f"\n3. Getting game summary from v8...")
        response = requests.get(summary_url)
        print(f"   Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Error: {response.text}")
            return
        
        summary_data = response.json()
        
        # Analyze the structure
        print("\n4. Data Structure Analysis:")
        print(f"   Top-level keys: {list(summary_data.keys())}")
        
        if 'game' in summary_data:
            game_data = summary_data['game']
            print(f"   Game keys: {list(game_data.keys())}")
            
            # Check for home team data
            if 'home' in game_data:
                home = game_data['home']
                print(f"\n   Home team keys: {list(home.keys())}")
                
                # Check for lineup
                if 'lineup' in home:
                    print(f"   ✓ HOME LINEUP FOUND: {len(home['lineup'])} players")
                    
                    # Show first player structure
                    if home['lineup']:
                        first_player = home['lineup'][0]
                        print(f"   Player structure keys: {list(first_player.keys())}")
                        
                        # Get player details
                        if 'player' in first_player:
                            player = first_player['player']
                            print(f"   Player info: {player.get('full_name', 'Unknown')}")
                            print(f"   Position: {first_player.get('position', 'Unknown')}")
                            print(f"   Order: {first_player.get('order', 'Unknown')}")
                else:
                    print("   ✗ No home lineup")
                
                # Check for starting pitcher
                if 'starting_pitcher' in home:
                    pitcher = home['starting_pitcher']
                    print(f"   ✓ Home starting pitcher: {pitcher.get('full_name', 'Unknown')}")
                elif 'probable_pitcher' in home:
                    pitcher = home['probable_pitcher']
                    print(f"   ✓ Home probable pitcher: {pitcher.get('full_name', 'Unknown')}")
                else:
                    print("   ✗ No home pitcher info")
            
            # Check for away team data
            if 'away' in game_data:
                away = game_data['away']
                print(f"\n   Away team keys: {list(away.keys())}")
                
                if 'lineup' in away:
                    print(f"   ✓ AWAY LINEUP FOUND: {len(away['lineup'])} players")
                else:
                    print("   ✗ No away lineup")
                
                if 'starting_pitcher' in away:
                    pitcher = away['starting_pitcher']
                    print(f"   ✓ Away starting pitcher: {pitcher.get('full_name', 'Unknown')}")
                elif 'probable_pitcher' in away:
                    pitcher = away['probable_pitcher']
                    print(f"   ✓ Away probable pitcher: {pitcher.get('full_name', 'Unknown')}")
                else:
                    print("   ✗ No away pitcher info")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        # Determine if we can use this
        has_lineups = False
        has_pitchers = False
        
        if 'game' in summary_data:
            game = summary_data['game']
            if 'home' in game and 'away' in game:
                if 'lineup' in game['home'] and 'lineup' in game['away']:
                    has_lineups = True
                if ('starting_pitcher' in game['home'] or 'probable_pitcher' in game['home']):
                    has_pitchers = True
        
        if has_lineups:
            print("✅ v8 API has lineups!")
        else:
            print("❌ No lineups available (probably not posted yet)")
        
        if has_pitchers:
            print("✅ v8 API has pitcher info!")
        else:
            print("❌ No pitcher info available")
        
        print(f"\nCurrent time: {datetime.now().strftime('%H:%M')}")
        print("Note: Lineups typically post 2-3 hours before game time")
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    test_v8_api()