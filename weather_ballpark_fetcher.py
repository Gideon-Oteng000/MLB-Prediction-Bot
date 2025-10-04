#!/usr/bin/env python3
"""
Weather and Ballpark Factors Fetcher
Fetches weather data and ballpark factors for MLB games using OpenWeatherMap API
Integrates with lineup data to assign weather conditions to players
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import os

class WeatherBallparkFetcher:
    """
    Fetches weather data and ballpark factors for MLB games
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"

        # MLB Stadium coordinates and information
        self.stadiums = {
            # AL East
            'NYY': {
                'name': 'Yankee Stadium',
                'city': 'Bronx',
                'lat': 40.8296,
                'lon': -73.9262,
                'hr_factor': 1.12,
                'elevation': 55,
                'orientation': 75
            },
            'BOS': {
                'name': 'Fenway Park',
                'city': 'Boston',
                'lat': 42.3467,
                'lon': -71.0972,
                'hr_factor': 1.06,
                'elevation': 21,
                'orientation': 45
            },
            'TB': {
                'name': 'Tropicana Field',
                'city': 'St. Petersburg',
                'lat': 27.7682,
                'lon': -82.6534,
                'hr_factor': 0.95,
                'elevation': 15,
                'orientation': 45
            },
            'TOR': {
                'name': 'Rogers Centre',
                'city': 'Toronto',
                'lat': 43.6414,
                'lon': -79.3894,
                'hr_factor': 1.02,
                'elevation': 91,
                'orientation': 0
            },
            'BAL': {
                'name': 'Oriole Park at Camden Yards',
                'city': 'Baltimore',
                'lat': 39.2838,
                'lon': -76.6218,
                'hr_factor': 1.08,
                'elevation': 20,
                'orientation': 30
            },

            # AL Central
            'CLE': {
                'name': 'Progressive Field',
                'city': 'Cleveland',
                'lat': 41.4958,
                'lon': -81.6852,
                'hr_factor': 0.98,
                'elevation': 660,
                'orientation': 0
            },
            'MIN': {
                'name': 'Target Field',
                'city': 'Minneapolis',
                'lat': 44.9816,
                'lon': -93.2776,
                'hr_factor': 1.01,
                'elevation': 815,
                'orientation': 90
            },
            'KC': {
                'name': 'Kauffman Stadium',
                'city': 'Kansas City',
                'lat': 39.0517,
                'lon': -94.4803,
                'hr_factor': 0.97,
                'elevation': 750,
                'orientation': 45
            },
            'CWS': {
                'name': 'Guaranteed Rate Field',
                'city': 'Chicago',
                'lat': 41.8300,
                'lon': -87.6338,
                'hr_factor': 1.03,
                'elevation': 595,
                'orientation': 135
            },
            'DET': {
                'name': 'Comerica Park',
                'city': 'Detroit',
                'lat': 42.3390,
                'lon': -83.0485,
                'hr_factor': 0.94,
                'elevation': 585,
                'orientation': 150
            },

            # AL West
            'HOU': {
                'name': 'Minute Maid Park',
                'city': 'Houston',
                'lat': 42.7762,
                'lon': -95.3900,
                'hr_factor': 1.09,
                'elevation': 22,
                'orientation': 345
            },
            'SEA': {
                'name': 'T-Mobile Park',
                'city': 'Seattle',
                'lat': 47.5914,
                'lon': -122.3326,
                'hr_factor': 0.92,
                'elevation': 134,
                'orientation': 45
            },
            'LAA': {
                'name': 'Angel Stadium',
                'city': 'Anaheim',
                'lat': 33.8003,
                'lon': -117.8827,
                'hr_factor': 0.96,
                'elevation': 153,
                'orientation': 45
            },
            'TEX': {
                'name': 'Globe Life Field',
                'city': 'Arlington',
                'lat': 32.7473,
                'lon': -97.0817,
                'hr_factor': 1.05,
                'elevation': 551,
                'orientation': 67
            },
            'OAK': {
                'name': 'Oakland Coliseum',
                'city': 'Oakland',
                'lat': 37.7516,
                'lon': -122.2008,
                'hr_factor': 0.89,
                'elevation': 13,
                'orientation': 60
            },

            # NL East
            'ATL': {
                'name': 'Truist Park',
                'city': 'Atlanta',
                'lat': 33.8906,
                'lon': -84.4677,
                'hr_factor': 1.04,
                'elevation': 1050,
                'orientation': 135
            },
            'NYM': {
                'name': 'Citi Field',
                'city': 'Flushing',
                'lat': 40.7571,
                'lon': -73.8458,
                'hr_factor': 0.93,
                'elevation': 37,
                'orientation': 30
            },
            'PHI': {
                'name': 'Citizens Bank Park',
                'city': 'Philadelphia',
                'lat': 39.9061,
                'lon': -75.1665,
                'hr_factor': 1.07,
                'elevation': 20,
                'orientation': 15
            },
            'WSH': {
                'name': 'Nationals Park',
                'city': 'Washington',
                'lat': 38.8730,
                'lon': -77.0074,
                'hr_factor': 1.01,
                'elevation': 12,
                'orientation': 30
            },
            'MIA': {
                'name': 'loanDepot Park',
                'city': 'Miami',
                'lat': 25.7781,
                'lon': -80.2197,
                'hr_factor': 0.85,
                'elevation': 8,
                'orientation': 120
            },

            # NL Central
            'MIL': {
                'name': 'American Family Field',
                'city': 'Milwaukee',
                'lat': 43.0280,
                'lon': -87.9712,
                'hr_factor': 1.02,
                'elevation': 635,
                'orientation': 135
            },
            'CHC': {
                'name': 'Wrigley Field',
                'city': 'Chicago',
                'lat': 41.9484,
                'lon': -87.6553,
                'hr_factor': 1.15,
                'elevation': 595,
                'orientation': 30
            },
            'STL': {
                'name': 'Busch Stadium',
                'city': 'St. Louis',
                'lat': 38.6226,
                'lon': -90.1928,
                'hr_factor': 0.99,
                'elevation': 465,
                'orientation': 60
            },
            'PIT': {
                'name': 'PNC Park',
                'city': 'Pittsburgh',
                'lat': 40.4469,
                'lon': -80.0057,
                'hr_factor': 0.91,
                'elevation': 730,
                'orientation': 120
            },
            'CIN': {
                'name': 'Great American Ball Park',
                'city': 'Cincinnati',
                'lat': 39.0975,
                'lon': -84.5068,
                'hr_factor': 1.03,
                'elevation': 550,
                'orientation': 120
            },

            # NL West
            'LAD': {
                'name': 'Dodger Stadium',
                'city': 'Los Angeles',
                'lat': 34.0739,
                'lon': -118.2400,
                'hr_factor': 0.88,
                'elevation': 340,
                'orientation': 30
            },
            'SD': {
                'name': 'Petco Park',
                'city': 'San Diego',
                'lat': 32.7073,
                'lon': -117.1566,
                'hr_factor': 0.88,
                'elevation': 62,
                'orientation': 0
            },
            'SF': {
                'name': 'Oracle Park',
                'city': 'San Francisco',
                'lat': 37.7786,
                'lon': -122.3893,
                'hr_factor': 0.81,
                'elevation': 12,
                'orientation': 90
            },
            'COL': {
                'name': 'Coors Field',
                'city': 'Denver',
                'lat': 39.7559,
                'lon': -104.9942,
                'hr_factor': 1.25,
                'elevation': 5200,
                'orientation': 0
            },
            'ARI': {
                'name': 'Chase Field',
                'city': 'Phoenix',
                'lat': 33.4452,
                'lon': -112.0667,
                'hr_factor': 1.06,
                'elevation': 1059,
                'orientation': 0
            }
        }

        print("Weather and Ballpark Fetcher initialized")
        print(f"Tracking {len(self.stadiums)} MLB stadiums")

    def enrich_lineup_data(self, lineup_file: str = "mlb_lineups.json") -> Dict:
        """
        Main method: Takes lineup data and adds weather/ballpark information
        """
        if not os.path.exists(lineup_file):
            print(f"Lineup file {lineup_file} not found")
            return {}

        try:
            with open(lineup_file, 'r') as f:
                lineup_data = json.load(f)

            print(f"Loaded lineup data from {lineup_file}")

            # Enrich with weather and ballpark data
            enriched_data = self._add_weather_ballpark_data(lineup_data)

            return enriched_data

        except Exception as e:
            print(f"Error enriching lineup data: {e}")
            return {}

    def _add_weather_ballpark_data(self, lineup_data: Dict) -> Dict:
        """
        Add weather and ballpark information directly to each game
        """
        enriched_data = lineup_data.copy()

        # Process games and add weather data directly to each game
        games = enriched_data.get('games', {})

        for game_key, game_data in games.items():
            # Get weather and ballpark data for this game
            weather_info = self._get_game_weather_info(game_data)

            if weather_info:
                # Add weather data directly to the game
                enriched_data['games'][game_key]['weather_ballpark'] = weather_info
                print(f"Added weather data to game {game_key}")

        return enriched_data

    def _get_game_weather_info(self, game_data: Dict) -> Optional[Dict]:
        """
        Get weather and ballpark info for a single game
        """
        try:
            home_team = game_data.get('home_team', '')
            away_team = game_data.get('away_team', '')

            # Convert full team names to abbreviations
            converted_data = self._convert_team_names(game_data)
            home_team_abbr = converted_data.get('home_team', home_team)

            # Get stadium info (always use home team's stadium)
            stadium_info = self.stadiums.get(home_team_abbr)
            if not stadium_info:
                print(f"Stadium not found for home team: {home_team} ({home_team_abbr})")
                return None

            # Get weather data
            weather_data = self._fetch_weather(stadium_info['lat'], stadium_info['lon'])

            if not weather_data:
                print(f"Could not fetch weather for {stadium_info['name']}")
                return None

            # Calculate weather effects on HR probability
            weather_effects = self._calculate_weather_effects(weather_data, stadium_info)

            weather_info = {
                'stadium': {
                    'name': stadium_info['name'],
                    'city': stadium_info['city'],
                    'hr_factor': stadium_info['hr_factor'],
                    'elevation': stadium_info['elevation']
                },
                'weather': weather_data,
                'hr_effects': weather_effects,
                'total_hr_multiplier': stadium_info['hr_factor'] * weather_effects['total_multiplier']
            }

            return weather_info

        except Exception as e:
            print(f"Error getting weather info: {e}")
            return None

    def _process_game_weather(self, game_data: Dict, games_processed: set) -> Optional[Dict]:
        """
        Process weather data for a single game
        """
        try:
            home_team = game_data.get('home_team', '')
            away_team = game_data.get('away_team', '')

            # Create unique game identifier
            game_id = f"{away_team}_at_{home_team}"

            if game_id in games_processed:
                return None

            games_processed.add(game_id)

            # Get stadium info (always use home team's stadium)
            stadium_info = self.stadiums.get(home_team)
            if not stadium_info:
                print(f"Stadium not found for home team: {home_team}")
                return None

            # Get weather data
            weather_data = self._fetch_weather(stadium_info['lat'], stadium_info['lon'])

            if not weather_data:
                print(f"Could not fetch weather for {stadium_info['name']}")
                return None

            # Calculate weather effects on HR probability
            weather_effects = self._calculate_weather_effects(weather_data, stadium_info)

            game_info = {
                'home_team': home_team,
                'away_team': away_team,
                'stadium': {
                    'name': stadium_info['name'],
                    'city': stadium_info['city'],
                    'hr_factor': stadium_info['hr_factor'],
                    'elevation': stadium_info['elevation']
                },
                'weather': weather_data,
                'hr_effects': weather_effects,
                'total_hr_multiplier': stadium_info['hr_factor'] * weather_effects['total_multiplier']
            }

            print(f"Processed weather for {game_id} at {stadium_info['name']}")
            return game_info

        except Exception as e:
            print(f"Error processing game weather: {e}")
            return None

    def _fetch_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Fetch current weather data from OpenWeatherMap API
        """
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'imperial'  # Fahrenheit, mph
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Extract relevant weather information
            weather_info = {
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'wind_direction': data.get('wind', {}).get('deg', 0),
                'conditions': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                'timestamp': datetime.now().isoformat()
            }

            # Add small delay to respect API rate limits
            time.sleep(0.1)

            return weather_info

        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None

    def _calculate_weather_effects(self, weather: Dict, stadium: Dict) -> Dict:
        """
        Calculate how weather affects home run probability
        Based on physics and historical data
        """
        effects = {
            'temperature_effect': 1.0,
            'wind_effect': 1.0,
            'humidity_effect': 1.0,
            'pressure_effect': 1.0,
            'total_multiplier': 1.0
        }

        try:
            temp = weather['temperature']
            wind_speed = weather['wind_speed']
            wind_direction = weather['wind_direction']
            humidity = weather['humidity']
            pressure = weather['pressure']

            # Temperature effect: Every 10°F above 70°F adds ~2% to HR distance
            if temp > 70:
                effects['temperature_effect'] = 1 + ((temp - 70) / 10) * 0.02
            elif temp < 50:
                effects['temperature_effect'] = 1 - ((50 - temp) / 10) * 0.015

            # Wind effect: Calculate relative wind direction based on stadium orientation
            # Stadium orientation is the bearing from home plate to center field
            stadium_orientation = stadium.get('orientation', 67)  # Default to MLB Rule 1.04 (ENE)

            # Calculate angle difference between wind direction and stadium orientation
            # This tells us if wind is blowing from behind home plate (favorable) or from outfield (unfavorable)
            angle_diff = (wind_direction - stadium_orientation + 180) % 360 - 180

            # Wind blowing out (from home plate toward outfield) increases HR distance
            # Wind blowing in (from outfield toward home plate) decreases HR distance
            wind_factor = 0

            if -45 <= angle_diff <= 45:
                # Wind blowing straight out to center field (most favorable)
                wind_factor = wind_speed * 0.005  # 5% per 10mph
            elif 45 < angle_diff <= 90 or -90 <= angle_diff < -45:
                # Wind blowing out to power alleys (favorable)
                wind_factor = wind_speed * 0.004  # 4% per 10mph
            elif 90 < angle_diff <= 135 or -135 <= angle_diff < -90:
                # Cross-wind (minimal effect)
                wind_factor = wind_speed * 0.001  # 1% per 10mph
            else:
                # Wind blowing in from outfield (unfavorable)
                wind_factor = -wind_speed * 0.004  # -4% per 10mph

            effects['wind_effect'] = max(0.7, min(1.3, 1 + wind_factor))

            # Humidity effect: High humidity = denser air = less distance
            if humidity > 60:
                effects['humidity_effect'] = 1 + ((humidity - 60) / 100) * 0.05  # slight boost
            elif humidity < 40:
                effects['humidity_effect'] = 1 - ((40 - humidity) / 100) * 0.05

            # Pressure effect: Low pressure = thinner air = more distance
            # Standard pressure ~30.00 inHg (~1013 mb)
            if pressure < 1010:
                effects['pressure_effect'] = 1 + ((1010 - pressure) / 20) * 0.02

            # Calculate total multiplier
            effects['total_multiplier'] = (
                effects['temperature_effect'] *
                effects['wind_effect'] *
                effects['humidity_effect'] *
                effects['pressure_effect']
            )

            # Cap the effects to reasonable bounds
            effects['total_multiplier'] = max(0.7, min(1.4, effects['total_multiplier']))

        except Exception as e:
            print(f"Error calculating weather effects: {e}")

        return effects

    def _convert_team_names(self, game_data: Dict) -> Dict:
        """
        Convert full team names to abbreviations for stadium lookup
        """
        team_name_mapping = {
            'New York Yankees': 'NYY',
            'Boston Red Sox': 'BOS',
            'Tampa Bay Rays': 'TB',
            'Toronto Blue Jays': 'TOR',
            'Baltimore Orioles': 'BAL',
            'Cleveland Guardians': 'CLE',
            'Minnesota Twins': 'MIN',
            'Kansas City Royals': 'KC',
            'Chicago White Sox': 'CWS',
            'Detroit Tigers': 'DET',
            'Houston Astros': 'HOU',
            'Seattle Mariners': 'SEA',
            'Los Angeles Angels': 'LAA',
            'Texas Rangers': 'TEX',
            'Oakland Athletics': 'OAK',
            'Atlanta Braves': 'ATL',
            'New York Mets': 'NYM',
            'Philadelphia Phillies': 'PHI',
            'Washington Nationals': 'WSH',
            'Miami Marlins': 'MIA',
            'Milwaukee Brewers': 'MIL',
            'Chicago Cubs': 'CHC',
            'St. Louis Cardinals': 'STL',
            'Pittsburgh Pirates': 'PIT',
            'Cincinnati Reds': 'CIN',
            'Los Angeles Dodgers': 'LAD',
            'San Diego Padres': 'SD',
            'San Francisco Giants': 'SF',
            'Colorado Rockies': 'COL',
            'Arizona Diamondbacks': 'ARI'
        }

        converted_data = game_data.copy()

        home_team = game_data.get('home_team', '')
        away_team = game_data.get('away_team', '')

        converted_data['home_team'] = team_name_mapping.get(home_team, home_team)
        converted_data['away_team'] = team_name_mapping.get(away_team, away_team)

        return converted_data

    def save_enriched_data(self, enriched_data: Dict, filename: str = "enriched_lineup_data.json"):
        """
        Save enriched data to JSON file
        """
        try:
            with open(filename, 'w') as f:
                json.dump(enriched_data, f, indent=2, default=str)
            print(f"Enriched data saved to {filename}")
        except Exception as e:
            print(f"Error saving enriched data: {e}")

    def enrich_lineups_with_weather(self, input_file: str = "lineups_with_metrics.json",
                                  output_file: str = "final_integrated_hr_data.json") -> Dict:
        """
        Final cascading enrichment: Add weather/ballpark data to lineups with metrics
        """
        print("="*60)
        print("FINAL CASCADING ENRICHMENT: Adding Weather & Ballpark Data")
        print("="*60)

        # Load lineup data with metrics
        if not os.path.exists(input_file):
            print(f"Input file {input_file} not found")
            print("Run advanced_metrics_fetcher.py with option 2 first")
            return {}

        try:
            with open(input_file, 'r') as f:
                lineup_data = json.load(f)
        except Exception as e:
            print(f"Error reading input file: {e}")
            return {}

        print(f"Loaded data from {input_file}")

        # Add weather and ballpark data
        enriched_data = self._add_weather_ballpark_data(lineup_data)

        # Add final integration metadata
        enriched_data['final_integration_info'] = {
            'date_final_enrichment': datetime.now().isoformat(),
            'data_sources': ['lineups', 'advanced_metrics', 'weather_ballpark'],
            'ready_for_hr_prediction': True,
            'total_games': len(enriched_data.get('games', {}))
        }

        # Save final integrated data
        try:
            with open(output_file, 'w') as f:
                json.dump(enriched_data, f, indent=2, default=str)
            print(f"Final integrated data saved to {output_file}")
        except Exception as e:
            print(f"Error saving final integrated data: {e}")

        # Display summary
        self._display_integration_summary(enriched_data)

        return enriched_data

    def _display_integration_summary(self, data: Dict):
        """
        Display summary of final integrated data
        """
        print("\n" + "="*60)
        print("FINAL INTEGRATION SUMMARY")
        print("="*60)

        games = data.get('games', {})
        total_games = len(games)
        total_batters = 0
        total_pitchers = 0
        batters_with_metrics = 0
        pitchers_with_metrics = 0
        games_with_weather = 0

        for game_key, game_data in games.items():
            # Count batters
            for lineup_type in ['away_lineup', 'home_lineup']:
                lineup = game_data.get(lineup_type, [])
                total_batters += len(lineup)
                batters_with_metrics += sum(1 for player in lineup
                                          if player.get('blended_metrics'))

            # Count pitchers
            for pitcher_type in ['away_pitcher', 'home_pitcher']:
                pitcher = game_data.get(pitcher_type, {})
                if pitcher.get('name'):
                    total_pitchers += 1
                    if pitcher.get('blended_metrics'):
                        pitchers_with_metrics += 1

            # Check weather data
            if game_data.get('weather_ballpark'):
                games_with_weather += 1

        print(f"Total Games: {total_games}")
        print(f"Total Batters: {total_batters}")
        print(f"Batters with Metrics: {batters_with_metrics} ({batters_with_metrics/total_batters*100:.1f}%)")
        print(f"Total Pitchers: {total_pitchers}")
        print(f"Pitchers with Metrics: {pitchers_with_metrics} ({pitchers_with_metrics/total_pitchers*100:.1f}%)")
        print(f"Games with Weather: {games_with_weather} ({games_with_weather/total_games*100:.1f}%)")
        print(f"\nData Ready for HR Prediction: {data.get('final_integration_info', {}).get('ready_for_hr_prediction', False)}")
        print("="*60)

    def display_weather_summary(self, enriched_data: Dict):
        """
        Display a summary of weather and ballpark data
        """
        print("\n" + "="*80)
        print("WEATHER AND BALLPARK SUMMARY")
        print("="*80)

        weather_data = enriched_data.get('weather_ballpark', {})

        if not weather_data:
            print("No weather data available")
            return

        for game_key, game_info in weather_data.items():
            stadium = game_info.get('stadium', {})
            weather = game_info.get('weather', {})
            effects = game_info.get('hr_effects', {})

            print(f"\n{game_info.get('away_team', 'Unknown')} @ {game_info.get('home_team', 'Unknown')}")
            print(f"Stadium: {stadium.get('name', 'Unknown')} (HR Factor: {stadium.get('hr_factor', 1.0):.2f})")
            print(f"Weather: {weather.get('temperature', 'N/A')}°F, {weather.get('description', 'N/A')}")
            print(f"Wind: {weather.get('wind_speed', 0):.1f} mph")
            print(f"Humidity: {weather.get('humidity', 'N/A')}%")
            print(f"Total HR Multiplier: {game_info.get('total_hr_multiplier', 1.0):.3f}")
            print("-" * 50)


def main():
    """
    Main function for cascading enrichment with weather/ballpark data
    """
    # Use your OpenWeatherMap API key
    API_KEY = "e09911139e379f1e4ca813df1778b4ef"

    fetcher = WeatherBallparkFetcher(API_KEY)

    print("WEATHER & BALLPARK FETCHER")
    print("="*50)
    print("1. Standard weather enrichment (from mlb_lineups.json)")
    print("2. Final cascading enrichment (from lineups_with_metrics.json)")

    choice = input("\nChoose option (1 or 2): ").strip()

    if choice == "2":
        # Final cascading enrichment mode
        input_file = "lineups_with_metrics.json"
        if os.path.exists(input_file):
            print(f"\nFound {input_file}, starting final cascading enrichment...")
            final_data = fetcher.enrich_lineups_with_weather(input_file, "final_integrated_hr_data.json")
            if final_data:
                print(f"\nCOMPLETE INTEGRATION PIPELINE FINISHED!")
                print(f"Final output: final_integrated_hr_data.json")
                print(f"Ready for HR prediction model!")
        else:
            print(f"\n{input_file} not found.")
            print("Run advanced_metrics_fetcher.py with option 2 first")
    else:
        # Standard mode
        lineup_file = "mlb_lineups.json"
        if not os.path.exists(lineup_file):
            print(f"{lineup_file} not found. Please run mlb_lineups_fetcher.py first.")
            return

        # Enrich lineup data with weather and ballpark information
        enriched_data = fetcher.enrich_lineup_data(lineup_file)

        if enriched_data:
            # Display summary
            fetcher.display_weather_summary(enriched_data)

            # Ask to save enriched data
            save_choice = input("\nSave enriched data to JSON file? (y/n): ").strip().lower()
            if save_choice == 'y':
                fetcher.save_enriched_data(enriched_data)

    print("\nDone!")


if __name__ == "__main__":
    main()