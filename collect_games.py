import requests
import pandas as pd
from datetime import datetime, timedelta
import json

def get_games_for_date(date_str):
    """Get all MLB games for a specific date"""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        games_list = []
        
        if data['dates'] and len(data['dates']) > 0:
            games = data['dates'][0]['games']
            
            for game in games:
                # Only get completed games
                if game['status']['statusCode'] == 'F':
                    game_info = {
                        'date': date_str,
                        'game_id': game['gamePk'],
                        'away_team': game['teams']['away']['team']['name'],
                        'home_team': game['teams']['home']['team']['name'],
                        'away_score': game['teams']['away']['score'],
                        'home_score': game['teams']['home']['score'],
                        'winner': 'home' if game['teams']['home']['score'] > game['teams']['away']['score'] else 'away'
                    }
                    games_list.append(game_info)
        
        return games_list
    
    except Exception as e:
        print(f"Error getting games for {date_str}: {e}")
        return []

def collect_recent_games():
    """Collect games from the last 10 days"""
    all_games = []
    
    # Get last 10 days of games
    for i in range(10):
        date = datetime.now() - timedelta(days=i+1)
        date_str = date.strftime('%Y-%m-%d')
        
        print(f"Getting games for {date_str}...")
        games = get_games_for_date(date_str)
        all_games.extend(games)
        
        # Be nice to the API
        import time
        time.sleep(0.5)
    
    return all_games

if __name__ == "__main__":
    print("Collecting MLB games...")
    games = collect_recent_games()
    
    if games:
        # Convert to DataFrame and save
        df = pd.DataFrame(games)
        df.to_csv('mlb_games.csv', index=False)
        
        print(f"Success! Collected {len(games)} games")
        print("Saved to mlb_games.csv")
        print("\nFirst few games:")
        print(df.head())
        
        # Show some basic stats
        print(f"\nHome teams won: {len(df[df['winner'] == 'home'])} games")
        print(f"Away teams won: {len(df[df['winner'] == 'away'])} games")
        home_win_rate = len(df[df['winner'] == 'home']) / len(df) * 100
        print(f"Home win rate: {home_win_rate:.1f}%")
    else:
        print("No games collected. Try adjusting the date range.")