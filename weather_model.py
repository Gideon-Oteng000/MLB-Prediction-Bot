import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import time

def get_weather_for_game(game_date, city_mapping):
    """Get historical weather data for game location"""
    # This is a simplified version - in reality you'd use a weather API
    # For now, we'll simulate some weather patterns based on city and date
    
    weather_patterns = {
        'hot_cities': ['Houston', 'Phoenix', 'Arlington', 'Miami', 'Tampa Bay'],
        'cold_cities': ['Boston', 'Minneapolis', 'Chicago', 'Detroit', 'Cleveland'],
        'windy_cities': ['Chicago', 'San Francisco', 'Boston'],
        'dome_cities': ['Houston', 'Tampa Bay', 'Minneapolis', 'Toronto']  # Domed stadiums
    }
    
    # Extract month from date
    month = int(game_date.split('-')[1])
    
    # Simulate weather based on city and month
    temp = 75  # Default temperature
    wind_speed = 5  # Default wind
    humidity = 50  # Default humidity
    is_dome = 0
    
    city = city_mapping.get('default', 'Unknown')
    
    # Temperature adjustments
    if city in weather_patterns['hot_cities']:
        temp += 10 if month in [6, 7, 8, 9] else 5
    elif city in weather_patterns['cold_cities']:
        temp -= 10 if month in [4, 5, 9, 10] else 5
    
    # Wind adjustments
    if city in weather_patterns['windy_cities']:
        wind_speed += np.random.randint(5, 15)
    
    # Dome games have controlled conditions
    if city in weather_patterns['dome_cities']:
        temp = 72
        wind_speed = 0
        humidity = 45
        is_dome = 1
    
    # Seasonal adjustments
    if month in [4, 10]:  # Early/late season
        temp -= 5
        wind_speed += 3
    elif month in [7, 8]:  # Peak summer
        temp += 5
        humidity += 10
    
    return {
        'temperature': temp,
        'wind_speed': wind_speed,
        'humidity': humidity,
        'is_dome': is_dome,
        'month': month
    }

def get_team_situational_factors(df, team, date):
    """Calculate situational factors for a team"""
    team_games = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) & 
        (pd.to_datetime(df['date']) < pd.to_datetime(date))
    ].copy()
    
    if len(team_games) == 0:
        return {
            'games_last_7_days': 0,
            'rest_advantage': 0,
            'recent_runs_scored': 4.5,
            'recent_runs_allowed': 4.5
        }
    
    # Get last game date to calculate rest
    last_game_date = team_games['date'].max()
    days_rest = (pd.to_datetime(date) - pd.to_datetime(last_game_date)).days
    
    # Games in last 7 days (fatigue factor)
    recent_games = team_games[
        pd.to_datetime(team_games['date']) >= pd.to_datetime(date) - pd.Timedelta(days=7)
    ]
    
    # Calculate recent offensive/defensive performance
    recent_runs_scored = 0
    recent_runs_allowed = 0
    
    for _, game in recent_games.tail(5).iterrows():  # Last 5 games
        if game['home_team'] == team:
            recent_runs_scored += game['home_score']
            recent_runs_allowed += game['away_score']
        else:
            recent_runs_scored += game['away_score']
            recent_runs_allowed += game['home_score']
    
    games_count = len(recent_games.tail(5))
    avg_runs_scored = recent_runs_scored / max(games_count, 1)
    avg_runs_allowed = recent_runs_allowed / max(games_count, 1)
    
    return {
        'games_last_7_days': len(recent_games),
        'rest_advantage': min(days_rest, 3),  # Cap at 3 days
        'recent_runs_scored': avg_runs_scored,
        'recent_runs_allowed': avg_runs_allowed
    }

def create_city_mapping():
    """Map team names to cities for weather"""
    return {
        'Arizona Diamondbacks': 'Phoenix',
        'Atlanta Braves': 'Atlanta',
        'Baltimore Orioles': 'Baltimore',
        'Boston Red Sox': 'Boston',
        'Chicago Cubs': 'Chicago',
        'Chicago White Sox': 'Chicago',
        'Cincinnati Reds': 'Cincinnati',
        'Cleveland Guardians': 'Cleveland',
        'Colorado Rockies': 'Denver',
        'Detroit Tigers': 'Detroit',
        'Houston Astros': 'Houston',
        'Kansas City Royals': 'Kansas City',
        'Los Angeles Angels': 'Los Angeles',
        'Los Angeles Dodgers': 'Los Angeles',
        'Miami Marlins': 'Miami',
        'Milwaukee Brewers': 'Milwaukee',
        'Minnesota Twins': 'Minneapolis',
        'New York Mets': 'New York',
        'New York Yankees': 'New York',
        'Oakland Athletics': 'Oakland',
        'Philadelphia Phillies': 'Philadelphia',
        'Pittsburgh Pirates': 'Pittsburgh',
        'San Diego Padres': 'San Diego',
        'San Francisco Giants': 'San Francisco',
        'Seattle Mariners': 'Seattle',
        'St. Louis Cardinals': 'St. Louis',
        'Tampa Bay Rays': 'Tampa Bay',
        'Texas Rangers': 'Arlington',
        'Toronto Blue Jays': 'Toronto',
        'Washington Nationals': 'Washington'
    }

def enhance_with_situational_factors():
    """Add weather and situational factors"""
    df = pd.read_csv('mlb_games.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    city_mapping = create_city_mapping()
    
    print("Adding weather and situational factors...")
    enhanced_games = []
    
    for idx, game in df.iterrows():
        print(f"Processing game {idx+1}/{len(df)}: {game['away_team']} @ {game['home_team']}")
        
        # Get weather for home team's city
        home_city = city_mapping.get(game['home_team'], 'Unknown')
        weather = get_weather_for_game(game['date'].strftime('%Y-%m-%d'), {'default': home_city})
        
        # Get situational factors
        home_situation = get_team_situational_factors(df, game['home_team'], game['date'])
        away_situation = get_team_situational_factors(df, game['away_team'], game['date'])
        
        # Travel factor (simplified - assume division rivals travel less)
        travel_factor = 1 if game['away_team'].split()[0] != game['home_team'].split()[0] else 0
        
        enhanced_game = {
            'date': game['date'],
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'home_wins': 1 if game['winner'] == 'home' else 0,
            
            # Weather factors
            'temperature': weather['temperature'],
            'wind_speed': weather['wind_speed'],
            'humidity': weather['humidity'],
            'is_dome': weather['is_dome'],
            'month': weather['month'],
            
            # Situational factors
            'home_rest_days': home_situation['rest_advantage'],
            'away_rest_days': away_situation['rest_advantage'],
            'rest_advantage': home_situation['rest_advantage'] - away_situation['rest_advantage'],
            'home_games_last_week': home_situation['games_last_7_days'],
            'away_games_last_week': away_situation['games_last_7_days'],
            'fatigue_advantage': away_situation['games_last_7_days'] - home_situation['games_last_7_days'],
            
            # Recent performance
            'home_recent_offense': home_situation['recent_runs_scored'],
            'away_recent_offense': away_situation['recent_runs_scored'],
            'home_recent_defense': home_situation['recent_runs_allowed'],
            'away_recent_defense': away_situation['recent_runs_allowed'],
            'offensive_advantage': home_situation['recent_runs_scored'] - away_situation['recent_runs_scored'],
            'defensive_advantage': away_situation['recent_runs_allowed'] - home_situation['recent_runs_allowed'],
            
            # Travel
            'travel_factor': travel_factor,
            
            # Home field
            'home_field': 1
        }
        
        enhanced_games.append(enhanced_game)
    
    enhanced_df = pd.DataFrame(enhanced_games)
    enhanced_df.to_csv('situational_games.csv', index=False)
    print(f"\nSaved enhanced data with situational factors")
    
    return enhanced_df

def build_situational_model(df):
    """Build model with weather and situational factors"""
    
    feature_columns = [
        'home_field',
        'temperature',
        'wind_speed',
        'is_dome',
        'rest_advantage',
        'fatigue_advantage',
        'offensive_advantage',
        'defensive_advantage',
        'travel_factor',
        'month'
    ]
    
    X = df[feature_columns]
    y = df['home_wins']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n=== SITUATIONAL MODEL ===")
    print(f"Using {len(feature_columns)} weather/situational features")
    print(f"Training on {len(X_train)} games, testing on {len(X_test)} games")
    
    # Build model
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Situational Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nMost Important Situational Features:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Insights
    print(f"\n=== SITUATIONAL INSIGHTS ===")
    top_feature = feature_importance.iloc[0]
    if top_feature['feature'] == 'home_field':
        print("🏠 Home field advantage still dominates")
    elif 'rest' in top_feature['feature']:
        print("😴 Rest/fatigue factors are important!")
    elif 'weather' in top_feature['feature'] or 'temperature' in top_feature['feature']:
        print("🌤️ Weather conditions matter!")
    
    return model, accuracy

if __name__ == "__main__":
    print("Building situational factors model (weather, rest, travel)...\n")
    
    # Enhance data
    df = enhance_with_situational_factors()
    
    # Build model
    model, accuracy = build_situational_model(df)
    
    print(f"\n=== COMPLETE MODEL COMPARISON ===")
    print(f"Home field only: 58.3%")
    print(f"Team stats: 59.0%") 
    print(f"Recent form: ~59.0%")
    print(f"Pitcher-focused: ~59.0%")
    print(f"Situational factors: {accuracy*100:.1f}%")
    
    if accuracy > 0.62:
        print("🎯 Breakthrough! Situational factors are key!")
    elif accuracy > 0.60:
        print("📊 Solid improvement with game context")
    else:
        print("🤔 Consistent performance across all models")
        
    print(f"\n💡 Key insight: With {len(df)} games, you're consistently around 58-60% accuracy")
    print("This suggests you need either:")
    print("1. Much more historical data (1000+ games)")  
    print("2. Focus on specific bet types or situations")
    print("3. Real-time factors (lineup changes, injuries, etc.)")