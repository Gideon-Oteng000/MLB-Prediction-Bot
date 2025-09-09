import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_season_schedule(year):
    """Get all games for a specific season"""
    print(f"Collecting {year} season...")
    
    # Get season dates
    if year == 2024:
        start_date = "2024-03-28"
        end_date = "2024-09-29"  # Regular season end
    elif year == 2023:
        start_date = "2023-03-30"
        end_date = "2023-10-01"
    elif year == 2022:
        start_date = "2022-04-07"
        end_date = "2022-10-05"
    else:
        print(f"Year {year} not configured")
        return []
    
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        all_games = []
        
        if 'dates' in data:
            for date_info in data['dates']:
                for game in date_info['games']:
                    if game['status']['statusCode'] == 'F':  # Final games only
                        game_info = {
                            'season': year,
                            'date': game['gameDate'][:10],  # YYYY-MM-DD format
                            'game_id': game['gamePk'],
                            'away_team': game['teams']['away']['team']['name'],
                            'home_team': game['teams']['home']['team']['name'],
                            'away_score': game['teams']['away']['score'],
                            'home_score': game['teams']['home']['score'],
                            'winner': 'home' if game['teams']['home']['score'] > game['teams']['away']['score'] else 'away'
                        }
                        all_games.append(game_info)
        
        print(f"  Collected {len(all_games)} games from {year}")
        return all_games
        
    except Exception as e:
        print(f"Error collecting {year} season: {e}")
        return []

def collect_multiple_seasons():
    """Collect data from multiple seasons"""
    print("Collecting historical MLB data...")
    print("This will take several minutes...\n")
    
    all_historical_games = []
    
    # Collect 2022, 2023, and 2024 seasons
    for year in [2022, 2023, 2024]:
        season_games = get_season_schedule(year)
        all_historical_games.extend(season_games)
        
        # Be nice to the API
        time.sleep(2)
    
    if all_historical_games:
        # Save historical data
        df = pd.DataFrame(all_historical_games)
        df.to_csv('historical_mlb_games.csv', index=False)
        
        print(f"\n=== HISTORICAL DATA SUMMARY ===")
        print(f"Total games collected: {len(all_historical_games)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Seasons: {sorted(df['season'].unique())}")
        
        # Season breakdown
        season_counts = df['season'].value_counts().sort_index()
        for season, count in season_counts.items():
            print(f"  {season}: {count} games")
        
        # Home field advantage analysis
        home_wins = len(df[df['winner'] == 'home'])
        total_games = len(df)
        home_win_rate = home_wins / total_games
        
        print(f"\nOverall home win rate: {home_win_rate:.3f} ({home_win_rate*100:.1f}%)")
        
        # Save summary
        with open('data_summary.txt', 'w') as f:
            f.write(f"MLB Historical Data Summary\n")
            f.write(f"==========================\n")
            f.write(f"Total games: {len(all_historical_games)}\n")
            f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n")
            f.write(f"Home win rate: {home_win_rate:.1%}\n")
            f.write(f"Data collected: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"\nSaved to 'historical_mlb_games.csv'")
        print(f"Summary saved to 'data_summary.txt'")
        
        return df
    else:
        print("No historical data collected")
        return None

def quick_model_test(df):
    """Test model performance with large dataset"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print(f"\n=== LARGE DATASET MODEL TEST ===")
    
    # Simple features
    df['home_field'] = 1
    df['home_wins'] = (df['winner'] == 'home').astype(int)
    
    # Calculate team win rates (using expanding window)
    team_win_rates = {}
    enhanced_games = []
    
    for idx, game in df.iterrows():
        # Get historical win rates for both teams up to this point
        historical_df = df.iloc[:idx]  # Only games before this one
        
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Calculate win rates
        home_games = historical_df[(historical_df['home_team'] == home_team) | (historical_df['away_team'] == home_team)]
        away_games = historical_df[(historical_df['home_team'] == away_team) | (historical_df['away_team'] == away_team)]
        
        if len(home_games) > 0:
            home_wins = len(home_games[
                ((home_games['home_team'] == home_team) & (home_games['winner'] == 'home')) |
                ((home_games['away_team'] == home_team) & (home_games['winner'] == 'away'))
            ])
            home_win_rate = home_wins / len(home_games)
        else:
            home_win_rate = 0.5
            
        if len(away_games) > 0:
            away_wins = len(away_games[
                ((away_games['home_team'] == away_team) & (away_games['winner'] == 'home')) |
                ((away_games['away_team'] == away_team) & (away_games['winner'] == 'away'))
            ])
            away_win_rate = away_wins / len(away_games)
        else:
            away_win_rate = 0.5
        
        enhanced_games.append({
            'home_field': 1,
            'home_win_rate': home_win_rate,
            'away_win_rate': away_win_rate,
            'win_rate_diff': home_win_rate - away_win_rate,
            'home_wins': game['home_wins']
        })
    
    # Use only games where we have some history for both teams
    enhanced_df = pd.DataFrame(enhanced_games)
    valid_games = enhanced_df[400:].copy()  # Skip first 400 games to ensure we have history
    
    print(f"Training on {len(valid_games)} games with historical context")
    
    # Build model
    feature_columns = ['home_field', 'win_rate_diff', 'home_win_rate', 'away_win_rate']
    X = valid_games[feature_columns]
    y = valid_games['home_wins']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Large Dataset Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    if accuracy > 0.55:
        print("🎉 With more data, the model is improving!")
    else:
        print("📊 Even with more data, prediction remains challenging")
    
    return accuracy

if __name__ == "__main__":
    print("MLB Historical Data Collector")
    print("============================")
    print("This will collect ~4,500 games from 2022-2024 seasons")
    print("Estimated time: 3-5 minutes\n")
    
    proceed = input("Continue? (y/n): ").lower()
    if proceed != 'y':
        print("Cancelled.")
        exit()
    
    # Collect historical data
    df = collect_multiple_seasons()
    
    if df is not None and len(df) > 1000:
        print("\n" + "="*50)
        print("SUCCESS! You now have a substantial dataset")
        print("="*50)
        
        # Quick test with large dataset
        accuracy = quick_model_test(df)
        
        print(f"\n=== NEXT STEPS ===")
        print("1. You now have 3+ seasons of data (~4,500 games)")
        print("2. Rerun your previous models with 'historical_mlb_games.csv'")
        print("3. Look for accuracy improvements with more data")
        print("4. Focus on specific situations or bet types")
        
        if accuracy > 0.56:
            print("5. 🚀 Your model is now beating the typical 54-55% threshold!")
        else:
            print("5. 📈 Consider specialized models for specific situations")
    
    else:
        print("Data collection incomplete. Try again later.")