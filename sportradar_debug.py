"""
SportsRadar API Debug Script
Let's see what data structure we're actually getting
"""

import requests
import json
from datetime import datetime
import time

# Your SportsRadar API key
SPORTRADAR_KEY = "3uCVRm9PhGc0VMvLafRDZpqcwdWrafLktHzEw3wr"

def test_api_connection():
    """Test basic API connection and see response structure"""
    
    print("=" * 80)
    print("SPORTRADAR API DEBUG TEST")
    print("=" * 80)
    
    # Test 1: Check API access with a simple endpoint
    print("\n1. Testing API Key validity...")
    test_url = f"https://api.sportradar.com/mlb/trial/v7/en/league/seasons.json?api_key={SPORTRADAR_KEY}"
    
    try:
        response = requests.get(test_url)
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✓ API Key is valid!")
            data = response.json()
            print(f"   Available seasons: {len(data.get('seasons', []))}")
        else:
            print(f"   ✗ Error: {response.text}")
            return
    except Exception as e:
        print(f"   ✗ Connection Error: {e}")
        return
    
    time.sleep(1.1)  # Rate limiting
    
    # Test 2: Get today's schedule
    print("\n2. Fetching today's schedule...")
    date_str = datetime.now().strftime('%Y/%m/%d')
    schedule_url = f"https://api.sportradar.com/mlb/trial/v7/en/games/{date_str}/schedule.json?api_key={SPORTRADAR_KEY}"
    
    try:
        response = requests.get(schedule_url)
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Found {len(data.get('games', []))} games")
            
            # Show game structure
            if data.get('games'):
                print("\n   First game structure:")
                game = data['games'][0]
                print(f"   - Game ID: {game.get('id')}")
                print(f"   - Status: {game.get('status')}")
                print(f"   - Home: {game.get('home', {}).get('name')}")
                print(f"   - Away: {game.get('away', {}).get('name')}")
                
                # Check for probables
                if 'probables' in game:
                    print("   - Has probable pitchers: YES")
                else:
                    print("   - Has probable pitchers: NO")
                
                return data['games']
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    return []

def test_game_details(games):
    """Test getting detailed game info including lineups"""
    
    if not games:
        print("\n3. No games to test")
        return
    
    print("\n3. Testing game details and lineups...")
    
    # Test first game
    game_id = games[0]['id']
    print(f"   Testing game: {game_id}")
    
    time.sleep(1.1)  # Rate limiting
    
    # Try game summary endpoint
    summary_url = f"https://api.sportradar.com/mlb/trial/v7/en/games/{game_id}/summary.json?api_key={SPORTRADAR_KEY}"
    
    try:
        response = requests.get(summary_url)
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n   Game Summary Structure:")
            print(f"   - Has 'game' key: {'game' in data}")
            
            if 'game' in data:
                game_data = data['game']
                
                # Check home team structure
                if 'home' in game_data:
                    print(f"   - Home team: {game_data['home'].get('name')}")
                    print(f"   - Has home lineup: {'lineup' in game_data['home']}")
                    
                    if 'lineup' in game_data['home']:
                        lineup = game_data['home']['lineup']
                        print(f"   - Home lineup size: {len(lineup)}")
                        
                        if lineup:
                            print("\n   First player structure:")
                            player = lineup[0]
                            print(f"     - Keys: {list(player.keys())}")
                            print(f"     - ID: {player.get('id')}")
                            print(f"     - Name: {player.get('preferred_name')} {player.get('last_name')}")
                            print(f"     - Position: {player.get('position')}")
                
                # Check for starting pitchers
                if 'starting_pitcher' in game_data.get('home', {}):
                    print(f"\n   Home Starting Pitcher: {game_data['home']['starting_pitcher'].get('last_name')}")
                
                if 'starting_pitcher' in game_data.get('away', {}):
                    print(f"   Away Starting Pitcher: {game_data['away']['starting_pitcher'].get('last_name')}")
                
                # Check for probables
                if 'probables' in game_data:
                    print("\n   Probable Pitchers found in game data")
                
                return data
        else:
            print(f"   ✗ Error: {response.text}")
            
            # Try different endpoint version
            print("\n   Trying alternate endpoint (v6)...")
            alt_url = f"https://api.sportradar.com/mlb/production/v6/en/games/{game_id}/summary.json?api_key={SPORTRADAR_KEY}"
            response = requests.get(alt_url)
            print(f"   Response Status: {response.status_code}")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    return None

def test_player_profile():
    """Test player profile endpoint"""
    
    print("\n4. Testing player profile endpoint...")
    
    # Test with a known player ID (Mike Trout as example)
    test_player_id = "dc1b40ba-f834-425b-b772-19f18e8dc1f0"  # Mike Trout
    
    time.sleep(1.1)
    
    profile_url = f"https://api.sportradar.com/mlb/trial/v7/en/players/{test_player_id}/profile.json?api_key={SPORTRADAR_KEY}"
    
    try:
        response = requests.get(profile_url)
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if 'player' in data:
                player = data['player']
                print(f"   ✓ Player found: {player.get('full_name')}")
                
                # Check for seasons data
                if 'seasons' in player:
                    print(f"   - Has seasons data: YES ({len(player['seasons'])} seasons)")
                    
                    # Check current season structure
                    for season in player['seasons']:
                        if season.get('year') == 2024:  # Check 2024 season
                            print(f"\n   2024 Season Structure:")
                            print(f"   - Type: {season.get('type')}")
                            
                            if 'totals' in season:
                                totals = season['totals']
                                print(f"   - Has totals: YES")
                                
                                if 'hitting' in totals:
                                    hitting = totals['hitting']
                                    print(f"   - Batting Average: {hitting.get('avg')}")
                                    print(f"   - Home Runs: {hitting.get('hr')}")
                                    print(f"   - At Bats: {hitting.get('ab')}")
                            break
        else:
            print(f"   ✗ Error: {response.text}")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")

def check_api_tier():
    """Check which API tier/version we have access to"""
    
    print("\n5. Checking API tier and access level...")
    
    endpoints_to_test = [
        ("Trial v7", "https://api.sportradar.com/mlb/trial/v7/en/league/seasons.json"),
        ("Production v7", "https://api.sportradar.com/mlb/production/v7/en/league/seasons.json"),
        ("Trial v6", "https://api.sportradar.com/mlb/trial/v6/en/league/seasons.json"),
    ]
    
    for name, base_url in endpoints_to_test:
        time.sleep(1.1)
        url = f"{base_url}?api_key={SPORTRADAR_KEY}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"   ✓ {name}: ACCESS GRANTED")
            else:
                print(f"   ✗ {name}: No access ({response.status_code})")
        except:
            print(f"   ✗ {name}: Connection failed")

# Main execution
if __name__ == "__main__":
    print("Testing SportsRadar API Configuration...")
    print(f"API Key: {SPORTRADAR_KEY[:10]}...{SPORTRADAR_KEY[-4:]}")
    print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Run tests
    games = test_api_connection()
    
    if games:
        game_data = test_game_details(games)
    
    test_player_profile()
    check_api_tier()
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
    print("\nBased on the results above, we can identify:")
    print("1. Which API tier you have access to")
    print("2. The correct endpoint structure")
    print("3. Whether lineups are available")
    print("4. The correct data field names")
    print("\nRun this script to diagnose the issue!")