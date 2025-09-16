"""
SportsRadar API - Find Working Stats Endpoints
Let's discover which endpoints actually return real player statistics
"""

import requests
import json
import time
from datetime import datetime

SPORTRADAR_KEY = "3uCVRm9PhGc0VMvLafRDZpqcwdWrafLktHzEw3wr"
BASE_URL = "https://api.sportradar.com/mlb/production/v7/en"

def test_endpoint(name, endpoint):
    """Test an endpoint and show what data it returns"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Endpoint: {endpoint}")
    print('-'*60)
    
    url = f"{BASE_URL}{endpoint}?api_key={SPORTRADAR_KEY}"
    
    try:
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS - Data structure:")
            
            # Show top-level keys
            print(f"Top-level keys: {list(data.keys())}")
            
            # Try to find player stats
            if 'players' in data:
                print(f"\nFound {len(data['players'])} players")
                if data['players']:
                    # Show first player's structure
                    player = data['players'][0]
                    print(f"First player keys: {list(player.keys())}")
                    
                    # Look for stats
                    if 'statistics' in player:
                        print(f"Statistics found! Keys: {list(player['statistics'].keys())}")
                        if 'hitting' in player['statistics']:
                            hitting = player['statistics']['hitting']
                            print(f"Hitting stats sample: HR={hitting.get('hr')}, AVG={hitting.get('avg')}")
                    
                    # Show player name if available
                    name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
                    print(f"Player example: {name}")
            
            # Check for league leaders structure
            if 'leaders' in data:
                print(f"\nFound leaders data")
                if 'categories' in data['leaders']:
                    cats = data['leaders']['categories']
                    print(f"Categories: {[c.get('type') for c in cats[:3]]}")
            
            # Check for team statistics
            if 'teams' in data:
                print(f"\nFound {len(data['teams'])} teams")
                if data['teams'] and 'players' in data['teams'][0]:
                    print(f"Team has player stats: YES")
            
            return True
        else:
            print(f"❌ FAILED: {response.status_code}")
            if response.status_code == 403:
                print("   (Access forbidden - endpoint not available)")
            elif response.status_code == 404:
                print("   (Not found - endpoint doesn't exist)")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    finally:
        time.sleep(1.1)  # Rate limiting

def main():
    print("="*60)
    print("SPORTRADAR MLB API - STATS ENDPOINT EXPLORER")
    print(f"Testing on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    current_year = datetime.now().year
    
    # Test various endpoints that might have player stats
    endpoints_to_test = [
        # Season statistics
        ("Season 2024 Statistics", f"/seasons/2024/REG/statistics.json"),
        ("Season 2025 Statistics", f"/seasons/2025/REG/statistics.json"),
        
        # League leaders
        ("League Leaders 2024", f"/seasons/2024/REG/leaders.json"),
        ("League Leaders - Hitting", f"/seasons/2024/REG/leaders/hitting.json"),
        
        # Team-based statistics (using Yankees as example)
        ("Yankees Team Statistics", f"/teams/a09ec676-f887-43dc-bbb9-3c9c5bd7c4dd/statistics.json"),
        ("Yankees Season 2024 Stats", f"/seasons/2024/REG/teams/a09ec676-f887-43dc-bbb9-3c9c5bd7c4dd/statistics.json"),
        
        # Daily changes (might have updated stats)
        ("Daily Changes", f"/league/2024/09/14/changes.json"),
        
        # Standings with extended stats
        ("Standings 2024", f"/seasons/2024/REG/standings.json"),
        
        # Player season statistics (different format)
        ("Player Season Stats", f"/seasons/2024/REG/players/statistics.json"),
    ]
    
    working_endpoints = []
    
    for name, endpoint in endpoints_to_test:
        if test_endpoint(name, endpoint):
            working_endpoints.append((name, endpoint))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if working_endpoints:
        print(f"\n✅ Working endpoints with potential stats ({len(working_endpoints)}):")
        for name, endpoint in working_endpoints:
            print(f"   • {name}")
    else:
        print("\n❌ No working stats endpoints found")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if working_endpoints:
        print("We found working endpoints! I can now:")
        print("1. Parse the season statistics to get real player HRs, AVG, etc.")
        print("2. Match players from lineups to their season stats")
        print("3. Use REAL data for predictions")
    else:
        print("SportsRadar player stats are not accessible.")
        print("We should switch to:")
        print("1. MLB Stats API (free, reliable)")
        print("2. PyBaseball (gets Statcast data)")
        print("3. A different paid API with better stats access")
    
    print("\nWhat would you like to do based on these results?")

if __name__ == "__main__":
    main()