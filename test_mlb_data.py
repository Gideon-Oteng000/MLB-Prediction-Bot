import requests
import json

# Test connection to MLB API
url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=2024-08-25"

try:
    response = requests.get(url)
    data = response.json()
    print("Success! Connected to MLB API")
    print(f"Found {len(data['dates'][0]['games']) if data['dates'] else 0} games")
    
    # Show what data looks like
    if data['dates'] and data['dates'][0]['games']:
        game = data['dates'][0]['games'][0]
        print(f"Sample game: {game['teams']['away']['team']['name']} @ {game['teams']['home']['team']['name']}")
        
except Exception as e:
    print(f"Error: {e}")