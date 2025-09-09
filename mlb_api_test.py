import requests
import json
from datetime import datetime

def test_mlb_api():
    """Test what's actually available from the MLB API"""
    
    print("=== TESTING MLB API RESPONSES ===\n")
    
    # Test 1: Today's schedule
    print("1. Testing today's schedule...")
    today = datetime.now().strftime('%Y-%m-%d')
    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
    
    try:
        response = requests.get(schedule_url)
        data = response.json()
        
        if data.get('dates') and len(data['dates']) > 0:
            games = data['dates'][0]['games']
            print(f"Found {len(games)} games")
            
            if len(games) > 0:
                game = games[0]
                print(f"Sample game: {game['teams']['away']['team']['name']} @ {game['teams']['home']['team']['name']}")
                
                # Check for probable pitcher data
                away_pitcher = game.get('teams', {}).get('away', {}).get('probablePitcher')
                home_pitcher = game.get('teams', {}).get('home', {}).get('probablePitcher')
                
                print(f"Away probable pitcher in schedule: {away_pitcher}")
                print(f"Home probable pitcher in schedule: {home_pitcher}")
                
                # Try the game detail API
                print(f"\n2. Testing game detail API...")
                game_id = game['gamePk']
                detail_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
                
                try:
                    detail_response = requests.get(detail_url)
                    detail_data = detail_response.json()
                    
                    probable_pitchers = detail_data.get('gameData', {}).get('probablePitchers', {})
                    print(f"Probable pitchers from detail: {probable_pitchers}")
                    
                    if 'away' in probable_pitchers:
                        away_pitcher_id = probable_pitchers['away'].get('id')
                        print(f"Away pitcher ID: {away_pitcher_id}")
                        
                        if away_pitcher_id:
                            # Test pitcher stats API
                            print(f"\n3. Testing pitcher stats API...")
                            test_pitcher_stats(away_pitcher_id)
                    
                except Exception as e:
                    print(f"Game detail API error: {e}")
                    
        else:
            print("No games found for today")
            
    except Exception as e:
        print(f"Schedule API error: {e}")

def test_pitcher_stats(pitcher_id):
    """Test pitcher stats API with a known pitcher ID"""
    
    current_year = datetime.now().year
    
    # Test person API
    print(f"Testing person API for pitcher {pitcher_id}...")
    person_url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}"
    
    try:
        person_response = requests.get(person_url)
        person_data = person_response.json()
        print(f"Person API response structure: {list(person_data.keys())}")
        
        if 'people' in person_data and len(person_data['people']) > 0:
            pitcher_info = person_data['people'][0]
            print(f"Pitcher name: {pitcher_info.get('fullName', 'Unknown')}")
            print(f"Pitcher handedness: {pitcher_info.get('pitchHand', {}).get('code', 'Unknown')}")
            
    except Exception as e:
        print(f"Person API error: {e}")
    
    # Test stats API
    print(f"\nTesting stats API for pitcher {pitcher_id}...")
    stats_url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=season&season={current_year}&group=pitching"
    
    try:
        stats_response = requests.get(stats_url)
        stats_data = stats_response.json()
        
        print(f"Stats API response structure: {list(stats_data.keys())}")
        print(f"Full stats response: {json.dumps(stats_data, indent=2)[:500]}...")
        
        if 'stats' in stats_data:
            print(f"Stats array length: {len(stats_data['stats'])}")
            if len(stats_data['stats']) > 0:
                stat_group = stats_data['stats'][0]
                print(f"First stat group keys: {list(stat_group.keys())}")
                if 'stats' in stat_group:
                    actual_stats = stat_group['stats']
                    print(f"Actual stats keys: {list(actual_stats.keys())}")
                    print(f"ERA: {actual_stats.get('era', 'Not found')}")
                else:
                    print("No 'stats' key in stat group")
            else:
                print("Empty stats array")
        else:
            print("No 'stats' key in response")
            
    except Exception as e:
        print(f"Stats API error: {e}")

def test_known_pitcher():
    """Test with a known active pitcher ID"""
    
    print(f"\n4. Testing with known pitcher IDs...")
    
    # Try some known pitcher IDs (these might need updating for current season)
    known_pitchers = [
        608331,  # Gerrit Cole
        621244,  # Shane Bieber  
        592789,  # Jacob deGrom
        608566   # Walker Buehler
    ]
    
    for pitcher_id in known_pitchers:
        print(f"\nTesting pitcher ID: {pitcher_id}")
        test_pitcher_stats(pitcher_id)
        break  # Just test the first one for now

if __name__ == "__main__":
    test_mlb_api()
    test_known_pitcher()
    
    print(f"\n=== SUMMARY ===")
    print("Run this script to see what's actually available from the MLB API")
    print("This will help us fix the pitcher data integration")