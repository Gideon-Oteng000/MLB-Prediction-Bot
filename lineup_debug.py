"""
Debug script to check why predictions aren't being generated
"""

import requests
import time
from datetime import datetime

SPORTRADAR_KEY = "3uCVRm9PhGc0VMvLafRDZpqcwdWrafLktHzEw3wr"
SPORTRADAR_BASE = "https://api.sportradar.com/mlb/production/v7/en"

def check_games_and_lineups():
    """Check each game's status and lineup availability"""
    
    print("=" * 80)
    print("LINEUP AVAILABILITY CHECK")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    # Get today's schedule
    date_str = datetime.now().strftime('%Y/%m/%d')
    url = f"{SPORTRADAR_BASE}/games/{date_str}/schedule.json?api_key={SPORTRADAR_KEY}"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error getting schedule: {response.status_code}")
        return
    
    data = response.json()
    
    if 'games' not in data:
        print("No games found")
        return
    
    games = data['games']
    print(f"\nFound {len(games)} total games today\n")
    
    scheduled_count = 0
    has_lineups_count = 0
    
    # Check each game
    for i, game in enumerate(games, 1):
        status = game.get('status', '')
        home = game['home']['name']
        away = game['away']['name']
        scheduled_time = game.get('scheduled', '')
        
        print(f"{i}. {away} @ {home}")
        print(f"   Status: {status}")
        print(f"   Scheduled: {scheduled_time}")
        
        # Only check scheduled games
        if status in ['scheduled', 'created', 'pre-game']:
            scheduled_count += 1
            
            # Get game details to check lineups
            time.sleep(1.5)  # Rate limiting
            game_id = game['id']
            detail_url = f"{SPORTRADAR_BASE}/games/{game_id}/summary.json?api_key={SPORTRADAR_KEY}"
            
            try:
                detail_response = requests.get(detail_url, timeout=10)
                if detail_response.status_code == 200:
                    detail_data = detail_response.json()
                    
                    if 'game' in detail_data:
                        game_data = detail_data['game']
                        
                        # Check for lineups
                        home_lineup = []
                        away_lineup = []
                        home_pitcher = None
                        away_pitcher = None
                        
                        if 'home' in game_data:
                            if 'lineup' in game_data['home']:
                                home_lineup = game_data['home']['lineup']
                            if 'starting_pitcher' in game_data['home']:
                                home_pitcher = game_data['home']['starting_pitcher']
                        
                        if 'away' in game_data:
                            if 'lineup' in game_data['away']:
                                away_lineup = game_data['away']['lineup']
                            if 'starting_pitcher' in game_data['away']:
                                away_pitcher = game_data['away']['starting_pitcher']
                        
                        print(f"   Home lineup: {len(home_lineup)} players")
                        print(f"   Away lineup: {len(away_lineup)} players")
                        print(f"   Home pitcher: {'YES' if home_pitcher else 'NO'}")
                        print(f"   Away pitcher: {'YES' if away_pitcher else 'NO'}")
                        
                        # Check if we have what we need for predictions
                        can_predict_home = len(home_lineup) > 0 and away_pitcher is not None
                        can_predict_away = len(away_lineup) > 0 and home_pitcher is not None
                        
                        if can_predict_home or can_predict_away:
                            print(f"   ✅ CAN MAKE PREDICTIONS")
                            has_lineups_count += 1
                        else:
                            print(f"   ❌ MISSING DATA FOR PREDICTIONS")
                            if len(home_lineup) == 0:
                                print(f"      - No home lineup")
                            if len(away_lineup) == 0:
                                print(f"      - No away lineup")
                            if not home_pitcher:
                                print(f"      - No home starting pitcher")
                            if not away_pitcher:
                                print(f"      - No away starting pitcher")
                else:
                    print(f"   ⚠️ Could not fetch game details: {detail_response.status_code}")
            except Exception as e:
                print(f"   ⚠️ Error fetching details: {e}")
        else:
            print(f"   ⏩ Skipping ({status})")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total games: {len(games)}")
    print(f"Scheduled games: {scheduled_count}")
    print(f"Games with lineups: {has_lineups_count}")
    
    if has_lineups_count == 0:
        print("\n⚠️ NO GAMES HAVE LINEUPS POSTED YET")
        print("\nPossible reasons:")
        print("1. Too early - lineups post 2-3 hours before game time")
        print("2. Day off - no games scheduled")
        print("3. All games already started (in-progress)")
        
        # Get current time and suggest when to run
        current_hour = datetime.now().hour
        if current_hour < 15:  # Before 3 PM
            print("\n💡 Try running again after 3 PM for evening games")
        elif current_hour < 18:  # 3-6 PM
            print("\n💡 Try running again after 5 PM - lineups should be posted by then")
        else:
            print("\n💡 Games may have already started. Try tomorrow morning for next day's games")
    else:
        print(f"\n✅ {has_lineups_count} games have lineups available")
        print("The prediction model should work with these games")

if __name__ == "__main__":
    check_games_and_lineups()