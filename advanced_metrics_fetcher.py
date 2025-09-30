#!/usr/bin/env python3
"""
Advanced Baseball Metrics Fetcher
Fetches advanced metrics from Baseball Savant for players from lineup data
Uses sophisticated blending and shrinkage techniques for predictive modeling
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import os
import sys
from fuzzywuzzy import fuzz
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import pybaseball as pyb
    pyb.cache.enable()  # Enable caching for faster subsequent runs
except ImportError:
    print("pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)

class AdvancedMetricsFetcher:
    """
    Fetches and blends advanced baseball metrics using Baseball Savant data
    with sophisticated weighting and shrinkage techniques
    """

    def __init__(self):
        self.current_year = datetime.now().year
        self.today = datetime.now().date()

        # Metric weights: [Season, Last30, Last7, Career/Platoon]
        self.batter_weights = {
            'barrel_pct': [0.60, 0.30, 0.10, 0.00],
            'hard_hit_pct': [0.60, 0.30, 0.10, 0.00],
            'avg_exit_velocity': [0.65, 0.25, 0.10, 0.00],
            'max_exit_velocity': [0.70, 0.20, 0.10, 0.00],
            'launch_angle': [0.60, 0.30, 0.10, 0.00],
            'sweet_spot_pct': [0.60, 0.30, 0.10, 0.00],
            'iso': [0.70, 0.25, 0.05, 0.00],
            'hr_per_fb': [0.75, 0.20, 0.05, 0.00],
            'obp': [0.70, 0.25, 0.05, 0.00],
            'k_pct': [0.70, 0.25, 0.05, 0.00],
            'bb_pct': [0.70, 0.25, 0.05, 0.00],
            'platoon_iso': [0.40, 0.30, 0.10, 0.20]
        }

        self.pitcher_weights = {
            'barrel_pct_allowed': [0.60, 0.30, 0.10, 0.00],
            'hard_hit_pct_allowed': [0.60, 0.30, 0.10, 0.00],
            'avg_exit_velocity_allowed': [0.65, 0.25, 0.10, 0.00],
            'launch_angle_allowed': [0.60, 0.30, 0.10, 0.00],
            'hr_per_9': [0.70, 0.25, 0.05, 0.00],
            'k_pct': [0.70, 0.25, 0.05, 0.00],
            'bb_pct': [0.70, 0.25, 0.05, 0.00],
            'fb_pct': [0.65, 0.25, 0.10, 0.00],
            'platoon_hr_allowed': [0.40, 0.30, 0.10, 0.20],
            'recent_workload': [0.20, 0.60, 0.20, 0.00]
        }

        # Shrinkage parameters (alpha values for empirical Bayes)
        self.shrinkage_params = {
            'hr_rate': 10,      # Conservative for HR/AB
            'barrel_pct': 35,   # More prior for barrel%
            'hard_hit_pct': 30,
            'k_pct': 25,
            'bb_pct': 20,
            'iso': 15
        }

        # Minimum sample sizes
        self.min_samples = {
            'season': 40,  # Minimum PAs for season stats
            'monthly': 20, # Minimum PAs for 30-day stats
            'weekly': 10   # Minimum PAs for 7-day stats
        }

        # League averages (will be updated from actual data)
        self.league_averages = {
            'hr_rate': 0.032,
            'barrel_pct': 0.084,
            'hard_hit_pct': 0.395,
            'k_pct': 0.223,
            'bb_pct': 0.085,
            'iso': 0.182,
            'obp': 0.315
        }

        print("Advanced Metrics Fetcher initialized")
        print(f"Using {self.current_year} season data")

    def from_lineup_json(self, json_file: str = "mlb_lineups.json") -> Dict:
        """
        Load player data from lineup JSON file and fetch advanced metrics
        """
        if not os.path.exists(json_file):
            print(f"Lineup file {json_file} not found. Run mlb_lineups_fetcher.py first or provide player names manually.")
            return {}

        try:
            with open(json_file, 'r') as f:
                lineup_data = json.load(f)

            print(f"Loaded lineup data from {json_file}")

            # Extract all players from both ESPN and MLB Stats data
            all_players = self._extract_players_from_lineups(lineup_data)

            if not all_players:
                print("No players found in lineup data")
                return {}

            print(f"Found {len(all_players)} unique players across all lineups")

            # Fetch advanced metrics for all players
            return self.fetch_and_blend_metrics(all_players)

        except Exception as e:
            print(f"Error reading lineup file: {e}")
            return {}

    def from_player_list(self, player_names: List[str], teams: Optional[List[str]] = None) -> Dict:
        """
        Fetch advanced metrics for manually provided player list
        """
        if not player_names:
            print("No player names provided")
            return {}

        players = []
        for i, name in enumerate(player_names):
            player_info = {
                'name': name,
                'team': teams[i] if teams and i < len(teams) else 'Unknown',
                'position': 'Unknown'
            }
            players.append(player_info)

        print(f"Processing {len(players)} manually provided players")
        return self.fetch_and_blend_metrics(players)

    def _extract_players_from_lineups(self, lineup_data: Dict) -> List[Dict]:
        """
        Extract all unique players from lineup data (handles new fallback structure)
        """
        players = []
        seen_players = set()

        # Handle new structure: check if 'games' key exists (new format)
        if 'games' in lineup_data:
            games_data = lineup_data.get('games', {})
            source_name = lineup_data.get('source', 'unknown')
        else:
            # Fallback to old structure - process both ESPN and MLB Stats
            games_data = {}

            # Add ESPN games
            espn_data = lineup_data.get('espn', {})
            for key, game in espn_data.items():
                games_data[f"espn_{key}"] = game

            # Add MLB Stats games
            mlb_data = lineup_data.get('mlb_stats', {})
            for key, game in mlb_data.items():
                games_data[f"mlb_{key}"] = game

            source_name = 'mixed'

        # Process all games
        for game_key, game_data in games_data.items():
            # Away team players
            for player in game_data.get('away_lineup', []):
                player_key = (player.get('name', '').lower(), game_data.get('away_team', ''))
                if player_key not in seen_players and player.get('name'):
                    seen_players.add(player_key)
                    players.append({
                        'name': player.get('name'),
                        'team': game_data.get('away_team'),
                        'position': player.get('position', ''),
                        'batting_order': player.get('batting_order', 0),
                        'source': source_name
                    })

            # Home team players
            for player in game_data.get('home_lineup', []):
                player_key = (player.get('name', '').lower(), game_data.get('home_team', ''))
                if player_key not in seen_players and player.get('name'):
                    seen_players.add(player_key)
                    players.append({
                        'name': player.get('name'),
                        'team': game_data.get('home_team'),
                        'position': player.get('position', ''),
                        'batting_order': player.get('batting_order', 0),
                        'source': source_name
                    })

            # Add pitchers
            away_pitcher = game_data.get('away_pitcher', {})
            home_pitcher = game_data.get('home_pitcher', {})

            for pitcher, team in [(away_pitcher, game_data.get('away_team')),
                                (home_pitcher, game_data.get('home_team'))]:
                if pitcher.get('name') and pitcher.get('name') != 'TBD':
                    pitcher_key = (pitcher.get('name').lower(), team)
                    if pitcher_key not in seen_players:
                        seen_players.add(pitcher_key)
                        players.append({
                            'name': pitcher.get('name'),
                            'team': team,
                            'position': 'P',
                            'is_pitcher': True,
                            'source': source_name
                        })

        return players

    def fetch_and_blend_metrics(self, players: List[Dict]) -> Dict:
        """
        Main method to fetch and blend advanced metrics for all players
        """
        results = {
            'date': str(self.today),
            'players': {},
            'metadata': {
                'total_players': len(players),
                'processing_time': None,
                'data_source': 'Baseball Savant via pybaseball'
            }
        }

        start_time = time.time()

        print(f"Fetching advanced metrics for {len(players)} players...")

        for i, player in enumerate(players):
            print(f"Processing {i+1}/{len(players)}: {player['name']}")

            try:
                if player.get('player_type') == 'pitcher':
                    player_metrics = self._get_pitcher_metrics(player)
                else:
                    player_metrics = self._get_batter_metrics(player)

                results['players'][player['name']] = player_metrics

            except Exception as e:
                print(f"Error processing {player['name']}: {e}")
                results['players'][player['name']] = {
                    'error': str(e),
                    'team': player.get('team'),
                    'position': player.get('position')
                }

            # Add small delay to be respectful to data sources
            time.sleep(0.1)

        processing_time = time.time() - start_time
        results['metadata']['processing_time'] = f"{processing_time:.2f} seconds"

        print(f"Completed processing in {processing_time:.2f} seconds")
        return results

    def _get_batter_metrics(self, player: Dict) -> Dict:
        """
        Fetch and blend batter metrics using the sophisticated weighting system
        """
        metrics = {
            'player_info': {
                'name': player['name'],
                'team': player.get('team'),
                'position': player.get('position'),
                'batting_order': player.get('batting_order')
            },
            'raw_metrics': {
                'season': {},
                'last_30': {},
                'last_7': {},
                'career': {}
            },
            'blended_metrics': {},
            'sample_sizes': {},
            'is_pitcher': False
        }

        try:
            # 1. Fetch season Statcast data
            season_data = self._fetch_statcast_batter_data(
                player['name'],
                start_date=f"{self.current_year}-03-01",
                end_date=str(self.today)
            )

            # 2. Fetch last 30 days data
            thirty_days_ago = self.today - timedelta(days=30)
            month_data = self._fetch_statcast_batter_data(
                player['name'],
                start_date=str(thirty_days_ago),
                end_date=str(self.today)
            )

            # 3. Fetch last 7 days data
            seven_days_ago = self.today - timedelta(days=7)
            week_data = self._fetch_statcast_batter_data(
                player['name'],
                start_date=str(seven_days_ago),
                end_date=str(self.today)
            )

            # 4. Process each time period
            metrics['raw_metrics']['season'] = self._process_batter_statcast(season_data)
            metrics['raw_metrics']['last_30'] = self._process_batter_statcast(month_data)
            metrics['raw_metrics']['last_7'] = self._process_batter_statcast(week_data)

            # 5. Get sample sizes
            metrics['sample_sizes'] = {
                'season_pa': len(season_data) if season_data is not None else 0,
                'month_pa': len(month_data) if month_data is not None else 0,
                'week_pa': len(week_data) if week_data is not None else 0
            }

            # 6. Apply shrinkage and blending
            metrics['blended_metrics'] = self._blend_batter_metrics(metrics)

        except Exception as e:
            print(f"Error fetching batter metrics for {player['name']}: {e}")
            metrics['error'] = str(e)

        return metrics

    def _get_pitcher_metrics(self, player: Dict) -> Dict:
        """
        Fetch and blend pitcher metrics using the sophisticated weighting system
        """
        metrics = {
            'player_info': {
                'name': player['name'],
                'team': player.get('team'),
                'position': 'P'
            },
            'raw_metrics': {
                'season': {},
                'last_30': {},
                'last_7': {},
                'career': {}
            },
            'blended_metrics': {},
            'sample_sizes': {},
            'is_pitcher': True
        }

        try:
            # 1. Fetch season Statcast data
            season_data = self._fetch_statcast_pitcher_data(
                player['name'],
                start_date=f"{self.current_year}-03-01",
                end_date=str(self.today)
            )

            # 2. Fetch last 30 days data
            thirty_days_ago = self.today - timedelta(days=30)
            month_data = self._fetch_statcast_pitcher_data(
                player['name'],
                start_date=str(thirty_days_ago),
                end_date=str(self.today)
            )

            # 3. Fetch last 7 days data
            seven_days_ago = self.today - timedelta(days=7)
            week_data = self._fetch_statcast_pitcher_data(
                player['name'],
                start_date=str(seven_days_ago),
                end_date=str(self.today)
            )

            # 4. Process each time period
            metrics['raw_metrics']['season'] = self._process_pitcher_statcast(season_data)
            metrics['raw_metrics']['last_30'] = self._process_pitcher_statcast(month_data)
            metrics['raw_metrics']['last_7'] = self._process_pitcher_statcast(week_data)

            # 5. Get sample sizes (batters faced)
            metrics['sample_sizes'] = {
                'season_bf': len(season_data) if season_data is not None else 0,
                'month_bf': len(month_data) if month_data is not None else 0,
                'week_bf': len(week_data) if week_data is not None else 0
            }

            # 6. Apply shrinkage and blending
            metrics['blended_metrics'] = self._blend_pitcher_metrics(metrics)

        except Exception as e:
            print(f"Error fetching pitcher metrics for {player['name']}: {e}")
            metrics['error'] = str(e)

        return metrics

    def _fetch_statcast_batter_data(self, player_name: str, start_date: str, end_date: str):
        """
        Fetch Statcast data for a batter using pybaseball
        """
        try:
            # Use fuzzy matching to find player in pybaseball
            player_id = self._find_player_id(player_name)
            if not player_id:
                print(f"Could not find player ID for {player_name}")
                return None

            # Fetch Statcast data
            data = pyb.statcast_batter(start_dt=start_date, end_dt=end_date, player_id=player_id)

            if data is None or data.empty:
                return None

            return data

        except Exception as e:
            print(f"Error fetching Statcast data for {player_name}: {e}")
            return None

    def _fetch_statcast_pitcher_data(self, player_name: str, start_date: str, end_date: str):
        """
        Fetch Statcast data for a pitcher using pybaseball
        """
        try:
            player_id = self._find_player_id(player_name)
            if not player_id:
                return None

            data = pyb.statcast_pitcher(start_dt=start_date, end_dt=end_date, player_id=player_id)

            if data is None or data.empty:
                return None

            return data

        except Exception as e:
            print(f"Error fetching pitcher Statcast data for {player_name}: {e}")
            return None

    def _find_player_id(self, player_name: str):
        """
        Find player ID using improved fuzzy matching and multiple fallback methods
        """
        # Clean the name first
        clean_name = self._clean_player_name(player_name)

        # Try multiple search strategies
        search_strategies = [
            self._lookup_by_full_name,
            self._lookup_by_last_first,
            self._lookup_by_last_name_only,
            self._lookup_fuzzy_match
        ]

        for strategy in search_strategies:
            try:
                player_id = strategy(clean_name)
                if player_id:
                    return player_id
            except Exception as e:
                continue

        print(f"Could not find player ID for {player_name} after trying all strategies")
        return None

    def _clean_player_name(self, name: str) -> str:
        """
        Clean player name by removing special characters and suffixes
        """
        # Remove Jr., Sr., III, etc.
        name = name.replace('Jr.', '').replace('Sr.', '').replace('III', '').replace('II', '')

        # Replace special characters with ASCII equivalents
        replacements = {
            'ñ': 'n', 'Ñ': 'N',
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ü': 'u', 'Ü': 'U'
        }

        for char, replacement in replacements.items():
            name = name.replace(char, replacement)

        return name.strip()

    def _lookup_by_full_name(self, name: str):
        """Strategy 1: Lookup by last name, first name"""
        parts = name.split()
        if len(parts) >= 2:
            lookup_data = pyb.playerid_lookup(parts[-1], parts[0])
            if lookup_data is not None and not lookup_data.empty:
                # Get most recent active player
                recent_player = lookup_data.iloc[0]
                return recent_player['key_mlbam']
        return None

    def _lookup_by_last_first(self, name: str):
        """Strategy 2: Try different name combinations"""
        parts = name.split()
        if len(parts) >= 2:
            # Try various combinations
            combinations = [
                (parts[-1], parts[0]),  # Last, First
                (parts[0], parts[-1]),  # First, Last
            ]

            # If middle name exists, try including it
            if len(parts) > 2:
                combinations.extend([
                    (parts[-1], f"{parts[0]} {parts[1]}"),  # Last, "First Middle"
                    (parts[-1], parts[1]),  # Last, Middle
                ])

            for last, first in combinations:
                try:
                    lookup_data = pyb.playerid_lookup(last, first)
                    if lookup_data is not None and not lookup_data.empty:
                        recent_player = lookup_data.iloc[0]
                        return recent_player['key_mlbam']
                except:
                    continue
        return None

    def _lookup_by_last_name_only(self, name: str):
        """Strategy 3: Lookup by last name only"""
        parts = name.split()
        if parts:
            lookup_data = pyb.playerid_lookup(parts[-1])
            if lookup_data is not None and not lookup_data.empty:
                # If multiple matches, try to find best match
                if len(lookup_data) > 1:
                    # Prefer more recent players
                    lookup_data = lookup_data.sort_values('mlb_played_last', ascending=False)

                    # Try to match first name too
                    if len(parts) >= 2:
                        first_name_matches = lookup_data[
                            lookup_data['name_first'].str.contains(parts[0], case=False, na=False)
                        ]
                        if not first_name_matches.empty:
                            return first_name_matches.iloc[0]['key_mlbam']

                # Return most recent player
                recent_player = lookup_data.iloc[0]
                return recent_player['key_mlbam']
        return None

    def _lookup_fuzzy_match(self, name: str):
        """Strategy 4: Fuzzy matching using fuzzywuzzy (if available)"""
        try:
            from fuzzywuzzy import fuzz, process

            # This is a simplified version - in production you'd want a player database
            # For now, try some common name variations
            name_variations = [
                name,
                name.replace(' Jr', ''),
                name.replace('Jr.', ''),
                ' '.join(name.split()[:2]),  # First two words only
            ]

            for variation in name_variations:
                parts = variation.split()
                if len(parts) >= 2:
                    try:
                        lookup_data = pyb.playerid_lookup(parts[-1], parts[0])
                        if lookup_data is not None and not lookup_data.empty:
                            return lookup_data.iloc[0]['key_mlbam']
                    except:
                        continue

        except ImportError:
            pass  # fuzzywuzzy not available

        return None

    def _process_batter_statcast(self, data):
        """
        Process raw Statcast data to calculate key metrics
        """
        if data is None or data.empty:
            return {}

        try:
            metrics = {}

            # Filter for valid batted balls
            batted_balls = data[data['type'] == 'X']  # Only batted ball events

            if not batted_balls.empty:
                # Barrel percentage
                barrels = batted_balls['launch_speed'] >= 98
                if 'launch_angle' in batted_balls.columns:
                    barrels = barrels & (batted_balls['launch_angle'].between(26, 30))
                metrics['barrel_pct'] = barrels.mean() * 100 if len(batted_balls) > 0 else 0

                # Hard hit percentage (95+ mph)
                hard_hits = batted_balls['launch_speed'] >= 95
                metrics['hard_hit_pct'] = hard_hits.mean() * 100 if len(batted_balls) > 0 else 0

                # Exit velocity metrics
                metrics['avg_exit_velocity'] = batted_balls['launch_speed'].mean()
                metrics['max_exit_velocity'] = batted_balls['launch_speed'].max()

                # Launch angle
                if 'launch_angle' in batted_balls.columns:
                    metrics['launch_angle'] = batted_balls['launch_angle'].mean()
                    metrics['launch_angle_std'] = batted_balls['launch_angle'].std()

                    # Sweet spot percentage (8-32 degree launch angle)
                    sweet_spot = batted_balls['launch_angle'].between(8, 32)
                    metrics['sweet_spot_pct'] = sweet_spot.mean() * 100 if len(batted_balls) > 0 else 0

            # Plate discipline metrics
            total_pitches = len(data)
            if total_pitches > 0:
                strikes = data['description'].isin(['called_strike', 'swinging_strike', 'foul', 'foul_tip'])
                balls = data['description'].isin(['ball'])
                metrics['strike_pct'] = strikes.mean() * 100
                metrics['ball_pct'] = balls.mean() * 100

            # Outcome-based metrics
            plate_appearances = data.groupby(['game_date', 'at_bat_number']).size()
            pa_count = len(plate_appearances)

            if pa_count > 0:
                # Walks and strikeouts
                walks = data['events'].eq('walk').sum()
                strikeouts = data['events'].isin(['strikeout', 'strikeout_double_play']).sum()

                metrics['bb_pct'] = (walks / pa_count) * 100
                metrics['k_pct'] = (strikeouts / pa_count) * 100

                # Home runs and other power metrics
                home_runs = data['events'].eq('home_run').sum()
                hits = data['events'].isin(['single', 'double', 'triple', 'home_run']).sum()

                metrics['hr_rate'] = home_runs / pa_count

                # ISO (requires at-bats)
                abs = data['events'].isin(['single', 'double', 'triple', 'home_run',
                                          'field_out', 'grounded_into_double_play',
                                          'fielders_choice_out']).sum()
                if abs > 0:
                    total_bases = (data['events'].eq('single').sum() +
                                 data['events'].eq('double').sum() * 2 +
                                 data['events'].eq('triple').sum() * 3 +
                                 data['events'].eq('home_run').sum() * 4)

                    avg = hits / abs
                    slg = total_bases / abs
                    metrics['iso'] = slg - avg

            return metrics

        except Exception as e:
            print(f"Error processing Statcast data: {e}")
            return {}

    def _process_pitcher_statcast(self, data):
        """
        Process raw Statcast data for pitcher metrics
        """
        if data is None or data.empty:
            return {}

        try:
            metrics = {}

            # Filter for valid batted balls against pitcher
            batted_balls = data[data['type'] == 'X']

            if not batted_balls.empty:
                # Barrel percentage allowed
                barrels = batted_balls['launch_speed'] >= 98
                if 'launch_angle' in batted_balls.columns:
                    barrels = barrels & (batted_balls['launch_angle'].between(26, 30))
                metrics['barrel_pct_allowed'] = barrels.mean() * 100

                # Hard hit percentage allowed
                hard_hits = batted_balls['launch_speed'] >= 95
                metrics['hard_hit_pct_allowed'] = hard_hits.mean() * 100

                # Exit velocity allowed
                metrics['avg_exit_velocity_allowed'] = batted_balls['launch_speed'].mean()
                metrics['launch_angle_allowed'] = batted_balls['launch_angle'].mean() if 'launch_angle' in batted_balls.columns else None

            # Pitcher outcome metrics
            plate_appearances = data.groupby(['game_date', 'at_bat_number']).size()
            pa_count = len(plate_appearances)

            if pa_count > 0:
                walks = data['events'].eq('walk').sum()
                strikeouts = data['events'].isin(['strikeout', 'strikeout_double_play']).sum()
                home_runs = data['events'].eq('home_run').sum()

                metrics['bb_pct'] = (walks / pa_count) * 100
                metrics['k_pct'] = (strikeouts / pa_count) * 100
                metrics['hr_rate'] = home_runs / pa_count

            return metrics

        except Exception as e:
            print(f"Error processing pitcher Statcast data: {e}")
            return {}

    def _blend_batter_metrics(self, metrics_data: Dict) -> Dict:
        """
        Apply shrinkage and blending to batter metrics
        """
        blended = {}

        raw_metrics = metrics_data['raw_metrics']
        sample_sizes = metrics_data['sample_sizes']

        # Define metrics to blend
        metrics_to_blend = [
            'barrel_pct', 'hard_hit_pct', 'avg_exit_velocity', 'max_exit_velocity',
            'launch_angle', 'sweet_spot_pct', 'iso', 'bb_pct', 'k_pct', 'hr_rate'
        ]

        for metric in metrics_to_blend:
            # Get raw values for each time period
            season_val = raw_metrics['season'].get(metric)
            month_val = raw_metrics['last_30'].get(metric)
            week_val = raw_metrics['last_7'].get(metric)
            career_val = None  # Would need separate career data fetch

            # Apply shrinkage if sample sizes are small
            if season_val is not None and sample_sizes['season_pa'] > 0:
                season_val = self._apply_shrinkage(season_val, sample_sizes['season_pa'], metric)

            if month_val is not None and sample_sizes['month_pa'] > 0:
                month_val = self._apply_shrinkage(month_val, sample_sizes['month_pa'], metric)

            if week_val is not None and sample_sizes['week_pa'] > 0:
                week_val = self._apply_shrinkage(week_val, sample_sizes['week_pa'], metric)

            # Blend using metric-specific weights
            blended_val = self._blend_metrics(season_val, month_val, week_val, career_val, metric)

            if blended_val is not None:
                blended[metric] = blended_val

        return blended

    def _blend_pitcher_metrics(self, metrics_data: Dict) -> Dict:
        """
        Apply shrinkage and blending to pitcher metrics
        """
        blended = {}

        raw_metrics = metrics_data['raw_metrics']
        sample_sizes = metrics_data['sample_sizes']

        # Define pitcher metrics to blend
        metrics_to_blend = [
            'barrel_pct_allowed', 'hard_hit_pct_allowed', 'avg_exit_velocity_allowed',
            'launch_angle_allowed', 'bb_pct', 'k_pct', 'hr_rate'
        ]

        for metric in metrics_to_blend:
            # Get raw values for each time period
            season_val = raw_metrics['season'].get(metric)
            month_val = raw_metrics['last_30'].get(metric)
            week_val = raw_metrics['last_7'].get(metric)
            career_val = None  # Would need separate career data fetch

            # Apply shrinkage if sample sizes are small
            if season_val is not None and sample_sizes['season_bf'] > 0:
                season_val = self._apply_shrinkage(season_val, sample_sizes['season_bf'], metric)

            if month_val is not None and sample_sizes['month_bf'] > 0:
                month_val = self._apply_shrinkage(month_val, sample_sizes['month_bf'], metric)

            if week_val is not None and sample_sizes['week_bf'] > 0:
                week_val = self._apply_shrinkage(week_val, sample_sizes['week_bf'], metric)

            # Use pitcher weights for blending
            weights = self.pitcher_weights.get(metric, [0.65, 0.25, 0.10, 0.00])
            blended_val = self._blend_metrics_with_weights(season_val, month_val, week_val, career_val, weights)

            if blended_val is not None:
                blended[metric] = blended_val

        return blended

    def _blend_metrics_with_weights(self, season_val: float, month_val: float, week_val: float,
                                   career_val: float, weights: List[float]) -> float:
        """
        Blend metrics using custom weights (alternative to metric-type lookup)
        """
        # Handle missing values
        values = [season_val, month_val, week_val, career_val]
        weights_adj = []
        values_adj = []

        for val, weight in zip(values, weights):
            if val is not None and not np.isnan(val):
                values_adj.append(val)
                weights_adj.append(weight)

        if not values_adj:
            return None

        # Normalize weights
        total_weight = sum(weights_adj)
        if total_weight == 0:
            return None

        weights_norm = [w / total_weight for w in weights_adj]

        # Calculate weighted average
        blended = sum(val * weight for val, weight in zip(values_adj, weights_norm))
        return blended

    def _apply_shrinkage(self, observed: float, n_samples: int, metric_type: str) -> float:
        """
        Apply empirical Bayes shrinkage for small sample sizes
        """
        if metric_type not in self.shrinkage_params:
            return observed

        alpha = self.shrinkage_params[metric_type]
        league_avg = self.league_averages.get(metric_type, observed)

        # Shrunk estimate: (x + α * λ) / (n + α)
        shrunk = (observed * n_samples + alpha * league_avg) / (n_samples + alpha)
        return shrunk

    def _blend_metrics(self, season_val: float, month_val: float, week_val: float,
                      career_val: float, metric_type: str) -> float:
        """
        Blend metrics using the specified weights for each metric type
        """
        if metric_type not in self.batter_weights:
            # Default weights if metric not specified
            weights = [0.65, 0.25, 0.10, 0.00]
        else:
            weights = self.batter_weights[metric_type]

        # Handle missing values
        values = [season_val, month_val, week_val, career_val]
        weights_adj = []
        values_adj = []

        for val, weight in zip(values, weights):
            if val is not None and not np.isnan(val):
                values_adj.append(val)
                weights_adj.append(weight)

        if not values_adj:
            return None

        # Normalize weights
        total_weight = sum(weights_adj)
        if total_weight == 0:
            return None

        weights_norm = [w / total_weight for w in weights_adj]

        # Calculate weighted average
        blended = sum(val * weight for val, weight in zip(values_adj, weights_norm))
        return blended

    def save_to_json(self, data: Dict, filename: str = "advanced_metrics.json"):
        """
        Save processed metrics to JSON file
        """
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"Advanced metrics saved to {filename}")
        except Exception as e:
            print(f"Error saving to {filename}: {e}")

    def enrich_lineup_with_metrics(self, lineup_file: str = "mlb_lineups.json",
                                 output_file: str = "lineups_with_metrics.json") -> Dict:
        """
        Cascading enrichment: Read lineup data, add blended metrics for batters and pitchers
        """
        print("="*60)
        print("CASCADING ENRICHMENT: Adding Advanced Metrics to Lineups")
        print("="*60)

        # Load lineup data
        if not os.path.exists(lineup_file):
            print(f"Lineup file {lineup_file} not found")
            return {}

        try:
            with open(lineup_file, 'r') as f:
                lineup_data = json.load(f)
        except Exception as e:
            print(f"Error reading lineup file: {e}")
            return {}

        # Extract all players (batters + pitchers) from lineups
        all_players = self._extract_all_players_and_pitchers(lineup_data)

        if not all_players:
            print("No players found in lineup data")
            return lineup_data

        print(f"Found {len(all_players)} unique players (batters + pitchers)")

        # Fetch metrics for all players
        metrics_data = self.fetch_and_blend_metrics(all_players)

        # Enrich the original lineup structure with blended metrics
        enriched_data = self._integrate_metrics_into_lineups(lineup_data, metrics_data)

        # Save enriched data
        try:
            with open(output_file, 'w') as f:
                json.dump(enriched_data, f, indent=2, default=str)
            print(f"Enriched lineup data saved to {output_file}")
        except Exception as e:
            print(f"Error saving enriched data: {e}")

        return enriched_data

    def _extract_all_players_and_pitchers(self, lineup_data: Dict) -> List[Dict]:
        """
        Extract all unique players (batters AND pitchers) from lineup data
        """
        players = []
        seen_players = set()

        # Get games data
        games = lineup_data.get('games', {})
        if not games:
            # Try old format
            games = {}
            espn_data = lineup_data.get('espn', {})
            mlb_data = lineup_data.get('mlb_stats', {})
            games.update(espn_data)
            games.update(mlb_data)

        for game_key, game_data in games.items():
            # Extract batters from lineups
            for lineup_type in ['away_lineup', 'home_lineup']:
                lineup = game_data.get(lineup_type, [])
                team = game_data.get('away_team' if 'away' in lineup_type else 'home_team', 'Unknown')

                for player in lineup:
                    if player.get('name'):
                        player_key = f"{player['name']}_{team}".lower()
                        if player_key not in seen_players:
                            seen_players.add(player_key)
                            players.append({
                                'name': player['name'],
                                'team': team,
                                'position': player.get('position', 'Unknown'),
                                'batting_order': player.get('batting_order', 0),
                                'player_type': 'batter'
                            })

            # Extract pitchers
            for pitcher_type in ['away_pitcher', 'home_pitcher']:
                pitcher = game_data.get(pitcher_type, {})
                if pitcher.get('name') and pitcher['name'] != 'TBD':
                    team = game_data.get('away_team' if 'away' in pitcher_type else 'home_team', 'Unknown')
                    player_key = f"{pitcher['name']}_{team}".lower()
                    if player_key not in seen_players:
                        seen_players.add(player_key)
                        players.append({
                            'name': pitcher['name'],
                            'team': team,
                            'position': 'P',
                            'batting_order': 0,
                            'player_type': 'pitcher',
                            'pitcher_stats': {
                                'wins': pitcher.get('wins', ''),
                                'losses': pitcher.get('losses', ''),
                                'era': pitcher.get('era', '')
                            }
                        })

        return players

    def _integrate_metrics_into_lineups(self, lineup_data: Dict, metrics_data: Dict) -> Dict:
        """
        Integrate blended metrics into the lineup structure
        """
        enriched_data = lineup_data.copy()

        # Add metadata
        enriched_data['enrichment_info'] = {
            'date_enriched': datetime.now().isoformat(),
            'metrics_added': 'blended_statcast_metrics',
            'players_processed': len(metrics_data.get('players', {})),
            'processing_time': metrics_data.get('metadata', {}).get('processing_time', 'Unknown')
        }

        games = enriched_data.get('games', {})
        player_metrics = metrics_data.get('players', {})

        for game_key, game_data in games.items():
            # Enrich batter lineups
            for lineup_type in ['away_lineup', 'home_lineup']:
                lineup = game_data.get(lineup_type, [])
                for i, player in enumerate(lineup):
                    player_name = player.get('name', '')
                    if player_name in player_metrics:
                        # Add only the blended metrics (what we need for HR prediction)
                        blended = player_metrics[player_name].get('blended_metrics', {})
                        enriched_data['games'][game_key][lineup_type][i]['blended_metrics'] = blended
                        enriched_data['games'][game_key][lineup_type][i]['sample_sizes'] = player_metrics[player_name].get('sample_sizes', {})
                        enriched_data['games'][game_key][lineup_type][i]['metrics_source'] = 'advanced_statcast'
                    else:
                        # Mark as missing metrics
                        enriched_data['games'][game_key][lineup_type][i]['blended_metrics'] = {}
                        enriched_data['games'][game_key][lineup_type][i]['metrics_source'] = 'missing'

            # Enrich pitcher data
            for pitcher_type in ['away_pitcher', 'home_pitcher']:
                pitcher = game_data.get(pitcher_type, {})
                pitcher_name = pitcher.get('name', '')
                if pitcher_name and pitcher_name != 'TBD' and pitcher_name in player_metrics:
                    # Add blended metrics for pitchers
                    blended = player_metrics[pitcher_name].get('blended_metrics', {})
                    enriched_data['games'][game_key][pitcher_type]['blended_metrics'] = blended
                    enriched_data['games'][game_key][pitcher_type]['sample_sizes'] = player_metrics[pitcher_name].get('sample_sizes', {})
                    enriched_data['games'][game_key][pitcher_type]['metrics_source'] = 'advanced_statcast'
                else:
                    # Mark as missing metrics
                    enriched_data['games'][game_key][pitcher_type]['blended_metrics'] = {}
                    enriched_data['games'][game_key][pitcher_type]['metrics_source'] = 'missing'

        return enriched_data

    def display_summary(self, data: Dict):
        """
        Display a summary of the fetched metrics
        """
        if not data.get('players'):
            print("No player data to display")
            return

        print("\n" + "="*80)
        print("ADVANCED METRICS SUMMARY")
        print("="*80)

        total_players = len(data['players'])
        successful = sum(1 for p in data['players'].values() if 'error' not in p)

        print(f"Total players processed: {total_players}")
        print(f"Successful: {successful}")
        print(f"Errors: {total_players - successful}")
        print(f"Processing time: {data.get('metadata', {}).get('processing_time', 'Unknown')}")

        # Display sample of players
        print(f"\nSample of processed players:")
        count = 0
        for name, metrics in data['players'].items():
            if count >= 5:  # Show first 5 players
                break
            if 'error' not in metrics:
                team = metrics.get('player_info', {}).get('team', 'Unknown')
                pos = metrics.get('player_info', {}).get('position', 'Unknown')
                print(f"  {name} ({team}) - {pos}")
                count += 1


def main():
    """
    Main function with command line interface for cascading enrichment
    """
    fetcher = AdvancedMetricsFetcher()

    print("ADVANCED METRICS FETCHER")
    print("="*50)
    print("1. Standard metrics fetch (save to advanced_metrics.json)")
    print("2. Cascading enrichment (enrich lineup data with metrics)")

    choice = input("\nChoose option (1 or 2): ").strip()

    if choice == "2":
        # Cascading enrichment mode
        lineup_file = "mlb_lineups.json"
        if os.path.exists(lineup_file):
            print(f"\nFound {lineup_file}, starting cascading enrichment...")
            enriched_data = fetcher.enrich_lineup_with_metrics(lineup_file, "lineups_with_metrics.json")
            if enriched_data:
                print(f"\n✅ Cascading enrichment completed!")
                print(f"📁 Output: lineups_with_metrics.json")
                print(f"📊 Games: {len(enriched_data.get('games', {}))}")
                print(f"📈 Players processed: {enriched_data.get('enrichment_info', {}).get('players_processed', 0)}")
        else:
            print(f"\n❌ {lineup_file} not found.")
            print("Run mlb_lineups_fetcher.py first to generate lineup data")
    else:
        # Standard mode
        results = None
        lineup_file = "mlb_lineups.json"
        if os.path.exists(lineup_file):
            print(f"Found {lineup_file}, processing lineup data...")
            results = fetcher.from_lineup_json(lineup_file)
        else:
            print(f"{lineup_file} not found.")
            print("Options:")
            print("1. Run mlb_lineups_fetcher.py first to generate lineup data")
            print("2. Enter player names manually")

            choice = input("Enter choice (1 or 2): ").strip()

            if choice == "1":
                print("Please run mlb_lineups_fetcher.py first and try again.")
                return
            elif choice == "2":
                print("Enter player names (comma-separated):")
                player_input = input().strip()
                if player_input:
                    player_names = [name.strip() for name in player_input.split(',')]
                    results = fetcher.from_player_list(player_names)
                else:
                    print("No players entered.")
                    return
            else:
                print("Invalid choice.")
                return

    if results:
        # Display summary
        fetcher.display_summary(results)

        # Ask to save
        save_choice = input("\nSave results to JSON file? (y/n): ").strip().lower()
        if save_choice == 'y':
            fetcher.save_to_json(results)

    print("\nDone!")


if __name__ == "__main__":
    main()