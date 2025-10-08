#!/usr/bin/env python3
"""
Weather and Ballpark Factors Fetcher for RBI Prediction
Fetches weather data and applies RBI-specific ballpark factors
Integrates with lineups_with_rbi_metrics.json to create final_integrated_rbi_data.json
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import os


class WeatherBallparkRBIFetcher:
    """
    Fetches weather data and applies RBI-specific ballpark factors
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"

        # MLB Stadium coordinates and RBI-specific park factors
        self.stadiums = {
            # AL East
            'NYY': {
                'name': 'Yankee Stadium',
                'city': 'Bronx',
                'lat': 40.8296,
                'lon': -73.9262,
                'run_factor': 1.05,
                'batting_avg_factor': 1.02,
                'extra_base_hit_factor': 1.08,
                'hr_factor': 1.12,
                'line_drive_factor': 1.03,
                'elevation': 55,
                'surface': 'grass',
                'orientation': 75
            },
            'BOS': {
                'name': 'Fenway Park',
                'city': 'Boston',
                'lat': 42.3467,
                'lon': -71.0972,
                'run_factor': 1.08,
                'batting_avg_factor': 1.06,
                'extra_base_hit_factor': 1.12,
                'hr_factor': 1.06,
                'line_drive_factor': 1.07,
                'elevation': 21,
                'surface': 'grass',
                'orientation': 45
            },
            'TB': {
                'name': 'Tropicana Field',
                'city': 'St. Petersburg',
                'lat': 27.7682,
                'lon': -82.6534,
                'run_factor': 0.97,
                'batting_avg_factor': 1.01,
                'extra_base_hit_factor': 0.94,
                'hr_factor': 0.95,
                'line_drive_factor': 0.98,
                'elevation': 15,
                'surface': 'turf',
                'orientation': 45
            },
            'TOR': {
                'name': 'Rogers Centre',
                'city': 'Toronto',
                'lat': 43.6414,
                'lon': -79.3894,
                'run_factor': 1.03,
                'batting_avg_factor': 1.02,
                'extra_base_hit_factor': 1.00,
                'hr_factor': 1.02,
                'line_drive_factor': 1.01,
                'elevation': 91,
                'surface': 'turf',
                'orientation': 0
            },
            'BAL': {
                'name': 'Oriole Park at Camden Yards',
                'city': 'Baltimore',
                'lat': 39.2838,
                'lon': -76.6218,
                'run_factor': 1.06,
                'batting_avg_factor': 1.04,
                'extra_base_hit_factor': 1.07,
                'hr_factor': 1.08,
                'line_drive_factor': 1.05,
                'elevation': 20,
                'surface': 'grass',
                'orientation': 30
            },

            # AL Central
            'CLE': {
                'name': 'Progressive Field',
                'city': 'Cleveland',
                'lat': 41.4958,
                'lon': -81.6852,
                'run_factor': 1.00,
                'batting_avg_factor': 1.00,
                'extra_base_hit_factor': 0.98,
                'hr_factor': 0.98,
                'line_drive_factor': 1.00,
                'elevation': 660,
                'surface': 'grass',
                'orientation': 0
            },
            'MIN': {
                'name': 'Target Field',
                'city': 'Minneapolis',
                'lat': 44.9816,
                'lon': -93.2776,
                'run_factor': 1.02,
                'batting_avg_factor': 1.01,
                'extra_base_hit_factor': 1.01,
                'hr_factor': 1.01,
                'line_drive_factor': 1.01,
                'elevation': 815,
                'surface': 'grass',
                'orientation': 90
            },
            'KC': {
                'name': 'Kauffman Stadium',
                'city': 'Kansas City',
                'lat': 39.0517,
                'lon': -94.4803,
                'run_factor': 0.99,
                'batting_avg_factor': 1.00,
                'extra_base_hit_factor': 0.97,
                'hr_factor': 0.97,
                'line_drive_factor': 0.99,
                'elevation': 750,
                'surface': 'grass',
                'orientation': 45
            },
            'CWS': {
                'name': 'Guaranteed Rate Field',
                'city': 'Chicago',
                'lat': 41.8300,
                'lon': -87.6338,
                'run_factor': 1.04,
                'batting_avg_factor': 1.03,
                'extra_base_hit_factor': 1.03,
                'hr_factor': 1.03,
                'line_drive_factor': 1.02,
                'elevation': 595,
                'surface': 'grass',
                'orientation': 135
            },
            'DET': {
                'name': 'Comerica Park',
                'city': 'Detroit',
                'lat': 42.3390,
                'lon': -83.0485,
                'run_factor': 0.96,
                'batting_avg_factor': 0.98,
                'extra_base_hit_factor': 0.92,
                'hr_factor': 0.94,
                'line_drive_factor': 0.97,
                'elevation': 585,
                'surface': 'grass',
                'orientation': 150
            },

            # AL West
            'HOU': {
                'name': 'Minute Maid Park',
                'city': 'Houston',
                'lat': 29.7570,
                'lon': -95.3553,
                'run_factor': 1.07,
                'batting_avg_factor': 1.05,
                'extra_base_hit_factor': 1.09,
                'hr_factor': 1.09,
                'line_drive_factor': 1.06,
                'elevation': 22,
                'surface': 'grass',
                'orientation': 345
            },
            'SEA': {
                'name': 'T-Mobile Park',
                'city': 'Seattle',
                'lat': 47.5914,
                'lon': -122.3326,
                'run_factor': 0.94,
                'batting_avg_factor': 0.97,
                'extra_base_hit_factor': 0.90,
                'hr_factor': 0.92,
                'line_drive_factor': 0.95,
                'elevation': 134,
                'surface': 'grass',
                'orientation': 45
            },
            'LAA': {
                'name': 'Angel Stadium',
                'city': 'Anaheim',
                'lat': 33.8003,
                'lon': -117.8827,
                'run_factor': 0.98,
                'batting_avg_factor': 0.99,
                'extra_base_hit_factor': 0.96,
                'hr_factor': 0.96,
                'line_drive_factor': 0.98,
                'elevation': 153,
                'surface': 'grass',
                'orientation': 45
            },
            'TEX': {
                'name': 'Globe Life Field',
                'city': 'Arlington',
                'lat': 32.7473,
                'lon': -97.0817,
                'run_factor': 1.04,
                'batting_avg_factor': 1.03,
                'extra_base_hit_factor': 1.05,
                'hr_factor': 1.05,
                'line_drive_factor': 1.03,
                'elevation': 551,
                'surface': 'grass',
                'orientation': 67
            },
            'OAK': {
                'name': 'Oakland Coliseum',
                'city': 'Oakland',
                'lat': 37.7516,
                'lon': -122.2008,
                'run_factor': 0.91,
                'batting_avg_factor': 0.95,
                'extra_base_hit_factor': 0.87,
                'hr_factor': 0.89,
                'line_drive_factor': 0.93,
                'elevation': 13,
                'surface': 'grass',
                'orientation': 60
            },

            # NL East
            'ATL': {
                'name': 'Truist Park',
                'city': 'Atlanta',
                'lat': 33.8906,
                'lon': -84.4677,
                'run_factor': 1.05,
                'batting_avg_factor': 1.03,
                'extra_base_hit_factor': 1.04,
                'hr_factor': 1.04,
                'line_drive_factor': 1.04,
                'elevation': 1050,
                'surface': 'grass',
                'orientation': 135
            },
            'NYM': {
                'name': 'Citi Field',
                'city': 'Flushing',
                'lat': 40.7571,
                'lon': -73.8458,
                'run_factor': 0.95,
                'batting_avg_factor': 0.97,
                'extra_base_hit_factor': 0.91,
                'hr_factor': 0.93,
                'line_drive_factor': 0.96,
                'elevation': 37,
                'surface': 'grass',
                'orientation': 30
            },
            'PHI': {
                'name': 'Citizens Bank Park',
                'city': 'Philadelphia',
                'lat': 39.9061,
                'lon': -75.1665,
                'run_factor': 1.06,
                'batting_avg_factor': 1.04,
                'extra_base_hit_factor': 1.07,
                'hr_factor': 1.07,
                'line_drive_factor': 1.05,
                'elevation': 20,
                'surface': 'grass',
                'orientation': 15
            },
            'WSH': {
                'name': 'Nationals Park',
                'city': 'Washington',
                'lat': 38.8730,
                'lon': -77.0074,
                'run_factor': 1.01,
                'batting_avg_factor': 1.01,
                'extra_base_hit_factor': 1.00,
                'hr_factor': 1.01,
                'line_drive_factor': 1.00,
                'elevation': 12,
                'surface': 'grass',
                'orientation': 30
            },
            'MIA': {
                'name': 'loanDepot Park',
                'city': 'Miami',
                'lat': 25.7781,
                'lon': -80.2197,
                'run_factor': 0.89,
                'batting_avg_factor': 0.93,
                'extra_base_hit_factor': 0.83,
                'hr_factor': 0.85,
                'line_drive_factor': 0.91,
                'elevation': 8,
                'surface': 'grass',
                'orientation': 120
            },

            # NL Central
            'MIL': {
                'name': 'American Family Field',
                'city': 'Milwaukee',
                'lat': 43.0280,
                'lon': -87.9712,
                'run_factor': 1.02,
                'batting_avg_factor': 1.01,
                'extra_base_hit_factor': 1.02,
                'hr_factor': 1.02,
                'line_drive_factor': 1.01,
                'elevation': 635,
                'surface': 'grass',
                'orientation': 135
            },
            'CHC': {
                'name': 'Wrigley Field',
                'city': 'Chicago',
                'lat': 41.9484,
                'lon': -87.6553,
                'run_factor': 1.09,
                'batting_avg_factor': 1.07,
                'extra_base_hit_factor': 1.13,
                'hr_factor': 1.15,
                'line_drive_factor': 1.08,
                'elevation': 595,
                'surface': 'grass',
                'orientation': 30
            },
            'STL': {
                'name': 'Busch Stadium',
                'city': 'St. Louis',
                'lat': 38.6226,
                'lon': -90.1928,
                'run_factor': 1.00,
                'batting_avg_factor': 1.00,
                'extra_base_hit_factor': 0.99,
                'hr_factor': 0.99,
                'line_drive_factor': 1.00,
                'elevation': 465,
                'surface': 'grass',
                'orientation': 60
            },
            'PIT': {
                'name': 'PNC Park',
                'city': 'Pittsburgh',
                'lat': 40.4469,
                'lon': -80.0057,
                'run_factor': 0.93,
                'batting_avg_factor': 0.96,
                'extra_base_hit_factor': 0.89,
                'hr_factor': 0.91,
                'line_drive_factor': 0.94,
                'elevation': 730,
                'surface': 'grass',
                'orientation': 120
            },
            'CIN': {
                'name': 'Great American Ball Park',
                'city': 'Cincinnati',
                'lat': 39.0975,
                'lon': -84.5068,
                'run_factor': 1.05,
                'batting_avg_factor': 1.03,
                'extra_base_hit_factor': 1.04,
                'hr_factor': 1.03,
                'line_drive_factor': 1.04,
                'elevation': 550,
                'surface': 'grass',
                'orientation': 120
            },

            # NL West
            'LAD': {
                'name': 'Dodger Stadium',
                'city': 'Los Angeles',
                'lat': 34.0739,
                'lon': -118.2400,
                'run_factor': 0.92,
                'batting_avg_factor': 0.95,
                'extra_base_hit_factor': 0.87,
                'hr_factor': 0.88,
                'line_drive_factor': 0.93,
                'elevation': 340,
                'surface': 'grass',
                'orientation': 30
            },
            'SD': {
                'name': 'Petco Park',
                'city': 'San Diego',
                'lat': 32.7073,
                'lon': -117.1566,
                'run_factor': 0.90,
                'batting_avg_factor': 0.94,
                'extra_base_hit_factor': 0.86,
                'hr_factor': 0.88,
                'line_drive_factor': 0.92,
                'elevation': 62,
                'surface': 'grass',
                'orientation': 0
            },
            'SF': {
                'name': 'Oracle Park',
                'city': 'San Francisco',
                'lat': 37.7786,
                'lon': -122.3893,
                'run_factor': 0.87,
                'batting_avg_factor': 0.91,
                'extra_base_hit_factor': 0.80,
                'hr_factor': 0.81,
                'line_drive_factor': 0.89,
                'elevation': 12,
                'surface': 'grass',
                'orientation': 90
            },
            'COL': {
                'name': 'Coors Field',
                'city': 'Denver',
                'lat': 39.7559,
                'lon': -104.9942,
                'run_factor': 1.22,
                'batting_avg_factor': 1.15,
                'extra_base_hit_factor': 1.28,
                'hr_factor': 1.25,
                'line_drive_factor': 1.18,
                'elevation': 5200,
                'surface': 'grass',
                'orientation': 0
            },
            'ARI': {
                'name': 'Chase Field',
                'city': 'Phoenix',
                'lat': 33.4452,
                'lon': -112.0667,
                'run_factor': 1.05,
                'batting_avg_factor': 1.04,
                'extra_base_hit_factor': 1.06,
                'hr_factor': 1.06,
                'line_drive_factor': 1.04,
                'elevation': 1059,
                'surface': 'grass',
                'orientation': 0
            }
        }

        print("[INFO] Weather and Ballpark RBI Fetcher initialized")
        print(f"[INFO] Tracking {len(self.stadiums)} MLB stadiums")

    def enrich_lineups_with_weather(self, input_file: str = "lineups_with_rbi_metrics.json",
                                    output_file: str = "final_integrated_rbi_data.json") -> Dict:
        """
        Final cascading enrichment: Add weather/ballpark data to lineups with RBI metrics
        """
        print("="*80)
        print("[INFO] FINAL RBI ENRICHMENT: Adding Weather & Ballpark Data")
        print("="*80)

        # Load lineup data with RBI metrics
        if not os.path.exists(input_file):
            print(f"[ERROR] {input_file} not found")
            print("[INFO] Run rbi_metrics_fetcher.py first")
            return {}

        try:
            with open(input_file, 'r') as f:
                lineup_data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read input file: {e}")
            return {}

        print(f"[INFO] Loaded data from {input_file}")

        # Add weather and ballpark data
        enriched_data = self._add_weather_ballpark_data(lineup_data)

        # Add final integration metadata
        enriched_data['final_integration_info'] = {
            'date_final_enrichment': datetime.now().isoformat(),
            'data_sources': ['lineups', 'rbi_metrics', 'weather_ballpark'],
            'ready_for_rbi_prediction': True,
            'total_games': len(enriched_data.get('games', {}))
        }

        # Save final integrated data
        try:
            with open(output_file, 'w') as f:
                json.dump(enriched_data, f, indent=2, default=str)
            print(f"[SUCCESS] Final integrated data saved to {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save final integrated data: {e}")

        # Display summary
        self._display_integration_summary(enriched_data)

        return enriched_data

    def _add_weather_ballpark_data(self, lineup_data: Dict) -> Dict:
        """
        Add RBI-specific weather and ballpark information to each game
        """
        enriched_data = lineup_data.copy()

        games = enriched_data.get('games', {})

        for game_key, game_data in games.items():
            # Get weather and ballpark data for this game
            weather_info = self._get_game_weather_info(game_data)

            if weather_info:
                # Add weather data directly to the game
                enriched_data['games'][game_key]['weather_ballpark'] = weather_info
                print(f"[INFO] Added weather/park data to {game_key}")

        return enriched_data

    def _get_game_weather_info(self, game_data: Dict) -> Optional[Dict]:
        """
        Get RBI-specific weather and ballpark info for a single game
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
                print(f"[WARN] Stadium not found for home team: {home_team} ({home_team_abbr})")
                return None

            # Get weather data
            weather_data = self._fetch_weather(stadium_info['lat'], stadium_info['lon'])

            if not weather_data:
                print(f"[WARN] Could not fetch weather for {stadium_info['name']}, using neutral defaults")
                # Use neutral weather defaults when API fails
                weather_data = self._get_default_weather()

            # Calculate RBI-specific weather effects
            weather_effects = self._calculate_rbi_weather_effects(weather_data, stadium_info)

            # Determine if day game (simplified - would need actual game time)
            game_time = game_data.get('game_time', '')
            is_day_game = self._is_day_game(game_time)

            # Day game bonus
            day_game_factor = 1.025 if is_day_game else 1.0

            # Calculate total RBI multiplier
            total_rbi_multiplier = (
                stadium_info['run_factor'] *
                stadium_info['batting_avg_factor'] *
                stadium_info['extra_base_hit_factor'] *
                weather_effects['total_multiplier'] *
                day_game_factor
            )

            # Bound multiplier between 0.8 and 1.3
            total_rbi_multiplier = max(0.8, min(1.3, total_rbi_multiplier))

            weather_info = {
                'stadium': {
                    'name': stadium_info['name'],
                    'city': stadium_info['city'],
                    'run_factor': stadium_info['run_factor'],
                    'batting_avg_factor': stadium_info['batting_avg_factor'],
                    'extra_base_hit_factor': stadium_info['extra_base_hit_factor'],
                    'hr_factor': stadium_info['hr_factor'],
                    'line_drive_factor': stadium_info['line_drive_factor'],
                    'elevation': stadium_info['elevation'],
                    'surface': stadium_info['surface']
                },
                'weather': weather_data,
                'rbi_effects': weather_effects,
                'game_time': 'day' if is_day_game else 'night',
                'day_game_bonus': day_game_factor,
                'total_rbi_multiplier': round(total_rbi_multiplier, 3)
            }

            return weather_info

        except Exception as e:
            print(f"[ERROR] Error getting weather info: {e}")
            return None

    def _get_default_weather(self) -> Dict:
        """
        Get neutral weather defaults when API is unavailable
        Assumes average MLB game conditions
        """
        return {
            'temperature': 72.0,  # Average game temperature
            'feels_like': 72.0,
            'humidity': 55,  # Moderate humidity
            'pressure': 1013,  # Standard atmospheric pressure
            'wind_speed': 5.0,  # Light breeze
            'wind_direction': 0,  # Neutral direction
            'conditions': 'Clear',
            'description': 'default weather (API unavailable)',
            'visibility': 10.0,
            'timestamp': datetime.now().isoformat(),
            'source': 'default_fallback'
        }

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
                'timestamp': datetime.now().isoformat(),
                'source': 'openweathermap_api'
            }

            # Add small delay to respect API rate limits
            time.sleep(0.1)

            return weather_info

        except Exception as e:
            print(f"[ERROR] Failed to fetch weather data: {e}")
            return None

    def _calculate_rbi_weather_effects(self, weather: Dict, stadium: Dict) -> Dict:
        """
        Calculate how weather affects RBI probability (RBI-specific logic)
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

            # 1. Temperature effect (same as HR model)
            # Warmer air = better ball carry = more hits and XBH
            if temp > 70:
                effects['temperature_effect'] = 1 + ((temp - 70) / 10) * 0.02
            elif temp < 60:
                effects['temperature_effect'] = 1 - ((60 - temp) / 10) * 0.015

            # 2. Wind effect (DAMPENED by 50% compared to HR model)
            # RBI cares about hits/doubles, not just HRs
            stadium_orientation = stadium.get('orientation', 67)
            angle_diff = (wind_direction - stadium_orientation + 180) % 360 - 180

            wind_factor = 0

            if -45 <= angle_diff <= 45:
                # Wind blowing out to center - helps all batted balls
                wind_factor = wind_speed * 0.0025  # 50% of HR effect (0.005 → 0.0025)
            elif 45 < angle_diff <= 90 or -90 <= angle_diff < -45:
                # Wind to power alleys - helps gap hitting
                wind_factor = wind_speed * 0.002
            elif 90 < angle_diff <= 135 or -135 <= angle_diff < -90:
                # Cross-wind - minimal effect
                wind_factor = wind_speed * 0.0005
            else:
                # Wind blowing in - hurts offense
                wind_factor = -wind_speed * 0.002

            effects['wind_effect'] = max(0.8, min(1.2, 1 + wind_factor))

            # 3. Humidity effect (small negative for RBI)
            # High humidity = denser air = less ball travel
            if humidity > 60:
                # Reduce by 1-2% per 10% humidity above 60
                humidity_penalty = ((humidity - 60) / 10) * 0.015
                effects['humidity_effect'] = 1 - humidity_penalty
            elif humidity < 40:
                # Slight boost for dry air
                humidity_boost = ((40 - humidity) / 10) * 0.01
                effects['humidity_effect'] = 1 + humidity_boost

            # 4. Pressure effect (scaled down for RBI - max ±3%)
            # Low pressure = thinner air = better ball carry
            if pressure < 1010:
                effects['pressure_effect'] = 1 + ((1010 - pressure) / 20) * 0.015
            elif pressure > 1020:
                effects['pressure_effect'] = 1 - ((pressure - 1020) / 20) * 0.015

            # Calculate total weather multiplier
            effects['total_multiplier'] = (
                effects['temperature_effect'] *
                effects['wind_effect'] *
                effects['humidity_effect'] *
                effects['pressure_effect']
            )

            # Cap the weather effects (tighter bounds for RBI)
            effects['total_multiplier'] = max(0.85, min(1.25, effects['total_multiplier']))

        except Exception as e:
            print(f"[ERROR] Error calculating weather effects: {e}")

        return effects

    def _is_day_game(self, game_time: str) -> bool:
        """
        Determine if game is day or night game
        """
        try:
            if not game_time:
                return False

            # Parse game time (ISO format)
            game_dt = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
            hour = game_dt.hour

            # Day games typically start before 5 PM (17:00)
            return 10 <= hour < 17

        except:
            # Default to night game if can't parse
            return False

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

    def _display_integration_summary(self, data: Dict):
        """
        Display summary of final integrated RBI data
        """
        print("\n" + "="*80)
        print("[INFO] FINAL RBI INTEGRATION SUMMARY")
        print("="*80)

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
        print(f"Batters with Metrics: {batters_with_metrics} ({batters_with_metrics/total_batters*100:.1f}%)" if total_batters > 0 else "Batters with Metrics: 0")
        print(f"Total Pitchers: {total_pitchers}")
        print(f"Pitchers with Metrics: {pitchers_with_metrics} ({pitchers_with_metrics/total_pitchers*100:.1f}%)" if total_pitchers > 0 else "Pitchers with Metrics: 0")
        print(f"Games with Weather: {games_with_weather} ({games_with_weather/total_games*100:.1f}%)" if total_games > 0 else "Games with Weather: 0")
        print(f"\nData Ready for RBI Prediction: {data.get('final_integration_info', {}).get('ready_for_rbi_prediction', False)}")
        print("="*80)


def main():
    """
    Main function for RBI weather/ballpark enrichment
    """
    # Use your OpenWeatherMap API key
    API_KEY = "e09911139e379f1e4ca813df1778b4ef"

    fetcher = WeatherBallparkRBIFetcher(API_KEY)

    print("\n[INFO] RBI WEATHER & BALLPARK ENRICHMENT")
    print("="*80)
    print("[INFO] This will add weather and ballpark factors to RBI metrics")
    print("="*80)

    # Run final cascading enrichment
    input_file = "lineups_with_rbi_metrics.json"
    if os.path.exists(input_file):
        print(f"\n[INFO] Found {input_file}, starting final enrichment...")
        final_data = fetcher.enrich_lineups_with_weather(input_file, "final_integrated_rbi_data.json")
        if final_data:
            print(f"\n[SUCCESS] RBI INTEGRATION PIPELINE COMPLETE!")
            print(f"[INFO] Final output: final_integrated_rbi_data.json")
            print(f"[INFO] Ready for RBI prediction model!")
    else:
        print(f"\n[ERROR] {input_file} not found.")
        print("[INFO] Run rbi_metrics_fetcher.py first")

    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
