import pandas as pd
import requests
from datetime import datetime, timedelta
import time

def get_recent_completed_games(days_back=7):
    """Get completed games from the last N days"""
    print(f"Collecting completed games from last {days_back} days...")
    
    all_recent_games = []
    
    for i in range(days_back):
        date = (datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d')
        print(f"  Checking {date}...")
        
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if data['dates'] and len(data['dates']) > 0:
                games = data['dates'][0]['games']
                
                for game in games:
                    if game['status']['statusCode'] == 'F':  # Final games only
                        game_info = {
                            'season': datetime.now().year,
                            'date': date,
                            'game_id': game['gamePk'],
                            'away_team': game['teams']['away']['team']['name'],
                            'home_team': game['teams']['home']['team']['name'],
                            'away_score': game['teams']['away']['score'],
                            'home_score': game['teams']['home']['score'],
                            'winner': 'home' if game['teams']['home']['score'] > game['teams']['away']['score'] else 'away'
                        }
                        all_recent_games.append(game_info)
            
            time.sleep(0.5)  # Be nice to the API
            
        except Exception as e:
            print(f"    Error getting games for {date}: {e}")
    
    print(f"Found {len(all_recent_games)} completed games")
    return all_recent_games

def update_historical_data():
    """Update historical data with recent completed games"""
    print("Updating historical database...")
    
    # Load existing historical data
    try:
        historical_df = pd.read_csv('historical_mlb_games.csv')
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        print(f"Loaded {len(historical_df)} existing historical games")
        
        last_date = historical_df['date'].max()
        print(f"Last game in database: {last_date.date()}")
        
    except FileNotFoundError:
        print("No existing historical file found")
        return False
    
    # Get recent games
    recent_games = get_recent_completed_games(days_back=10)
    
    if not recent_games:
        print("No recent games to add")
        return True
    
    recent_df = pd.DataFrame(recent_games)
    recent_df['date'] = pd.to_datetime(recent_df['date'])
    
    # Find truly new games (not already in database)
    new_games = []
    for _, game in recent_df.iterrows():
        existing_game = historical_df[
            (historical_df['game_id'] == game['game_id']) |
            ((historical_df['date'] == game['date']) & 
             (historical_df['home_team'] == game['home_team']) & 
             (historical_df['away_team'] == game['away_team']))
        ]
        
        if len(existing_game) == 0:
            new_games.append(game.to_dict())
    
    if new_games:
        print(f"Adding {len(new_games)} new games to database")
        
        new_games_df = pd.DataFrame(new_games)
        updated_df = pd.concat([historical_df, new_games_df], ignore_index=True)
        updated_df = updated_df.sort_values('date').drop_duplicates(subset=['game_id'], keep='last')
        
        # Save updated data
        updated_df.to_csv('historical_mlb_games.csv', index=False)
        print(f"Updated database now contains {len(updated_df)} games")
        
        # Show what was added
        print("New games added:")
        for game in new_games[-5:]:  # Show last 5 added
            print(f"  {game['date']}: {game['away_team']} @ {game['home_team']} ({game['away_score']}-{game['home_score']})")
        
        return True
    else:
        print("Database is already up to date")
        return True

def force_model_retrain():
    """Force the production system to retrain with new data"""
    import os
    
    # Remove existing model files to force retraining
    model_files = ['production_mlb_model.pkl', 'mlb_model.pkl', 'final_mlb_model.pkl']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            os.remove(model_file)
            print(f"Removed {model_file} to force retraining")
    
    print("Model will retrain with updated data on next prediction run")

if __name__ == "__main__":
    print("MLB Data Updater")
    print("="*40)
    print("This will update your historical data with recent completed games")
    print("and force the model to retrain with new data.\n")
    
    # Update historical data
    success = update_historical_data()
    
    if success:
        print("\n" + "="*40)
        print("SUCCESS: Historical data updated")
        
        # Ask if user wants to force retrain
        retrain = input("\nForce model retrain with new data? (y/n): ").lower()
        if retrain == 'y':
            force_model_retrain()
            print("\nNext time you run production_system.py, it will:")
            print("1. Use the updated historical data")
            print("2. Retrain the model with recent games")
            print("3. Give you fresh predictions based on latest team performance")
        
        print(f"\nTo get updated predictions, run:")
        print(f"python production_system.py")
    else:
        print("Failed to update data")