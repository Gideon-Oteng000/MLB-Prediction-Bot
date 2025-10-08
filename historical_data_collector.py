#!/usr/bin/env python3
"""
Historical HR Data Collector
Collects historical game data from 2018-2024 with player metrics and actual HR outcomes
for training machine learning models
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import os
import sys
import time

warnings.filterwarnings('ignore')

try:
    import pybaseball as pyb
    pyb.cache.enable()
except ImportError:
    print("pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)


class HistoricalDataCollector:
    """
    Collects historical MLB game data with player metrics and HR outcomes
    """

    def __init__(self, start_year: int = 2018, end_year: int = 2024):
        self.start_year = start_year
        self.end_year = end_year
        self.current_year = datetime.now().year

        # Storage for collected data
        self.games_data = []
        self.training_rows = []

        # Stadium information (from your existing weather_ballpark_fetcher.py)
        self.stadiums = self._load_stadium_info()

        # Progress tracking
        self.progress_file = "collection_progress.json"
        self.checkpoint_file = "checkpoint_data.json"

        print(f"Historical Data Collector initialized")
        print(f"Collection period: {start_year} - {end_year}")
        print(f"Estimated games: ~{(end_year - start_year + 1) * 2430} games")

    def _load_stadium_info(self) -> Dict:
        """Load stadium information for ballpark factors"""
        # Simplified version - you can expand with full stadium data
        return {
            'NYY': {'name': 'Yankee Stadium', 'hr_factor': 1.12, 'elevation': 55},
            'BOS': {'name': 'Fenway Park', 'hr_factor': 1.06, 'elevation': 21},
            'TB': {'name': 'Tropicana Field', 'hr_factor': 0.95, 'elevation': 15},
            'TOR': {'name': 'Rogers Centre', 'hr_factor': 1.02, 'elevation': 91},
            'BAL': {'name': 'Oriole Park', 'hr_factor': 1.08, 'elevation': 20},
            'CLE': {'name': 'Progressive Field', 'hr_factor': 0.98, 'elevation': 660},
            'MIN': {'name': 'Target Field', 'hr_factor': 1.01, 'elevation': 815},
            'KC': {'name': 'Kauffman Stadium', 'hr_factor': 0.97, 'elevation': 750},
            'CWS': {'name': 'Guaranteed Rate Field', 'hr_factor': 1.03, 'elevation': 595},
            'DET': {'name': 'Comerica Park', 'hr_factor': 0.94, 'elevation': 585},
            'HOU': {'name': 'Minute Maid Park', 'hr_factor': 1.09, 'elevation': 22},
            'SEA': {'name': 'T-Mobile Park', 'hr_factor': 0.92, 'elevation': 134},
            'LAA': {'name': 'Angel Stadium', 'hr_factor': 0.96, 'elevation': 153},
            'TEX': {'name': 'Globe Life Field', 'hr_factor': 1.05, 'elevation': 551},
            'OAK': {'name': 'Oakland Coliseum', 'hr_factor': 0.89, 'elevation': 13},
            'ATL': {'name': 'Truist Park', 'hr_factor': 1.04, 'elevation': 1050},
            'NYM': {'name': 'Citi Field', 'hr_factor': 0.93, 'elevation': 37},
            'PHI': {'name': 'Citizens Bank Park', 'hr_factor': 1.07, 'elevation': 20},
            'WSH': {'name': 'Nationals Park', 'hr_factor': 1.01, 'elevation': 12},
            'MIA': {'name': 'loanDepot Park', 'hr_factor': 0.85, 'elevation': 8},
            'MIL': {'name': 'American Family Field', 'hr_factor': 1.02, 'elevation': 635},
            'CHC': {'name': 'Wrigley Field', 'hr_factor': 1.15, 'elevation': 595},
            'STL': {'name': 'Busch Stadium', 'hr_factor': 0.99, 'elevation': 465},
            'PIT': {'name': 'PNC Park', 'hr_factor': 0.91, 'elevation': 730},
            'CIN': {'name': 'Great American Ball Park', 'hr_factor': 1.03, 'elevation': 550},
            'LAD': {'name': 'Dodger Stadium', 'hr_factor': 0.88, 'elevation': 340},
            'SD': {'name': 'Petco Park', 'hr_factor': 0.88, 'elevation': 62},
            'SF': {'name': 'Oracle Park', 'hr_factor': 0.81, 'elevation': 12},
            'COL': {'name': 'Coors Field', 'hr_factor': 1.25, 'elevation': 5200},
            'ARI': {'name': 'Chase Field', 'hr_factor': 1.06, 'elevation': 1059}
        }

    def collect_all_data(self):
        """
        Main method to collect all historical data
        """
        print("\n" + "="*80)
        print("STARTING HISTORICAL DATA COLLECTION")
        print("="*80)

        # Load progress if exists
        progress = self._load_progress()

        for year in range(self.start_year, self.end_year + 1):
            if year in progress.get('completed_years', []):
                print(f"\nYear {year} already completed, skipping...")
                continue

            print(f"\n{'='*80}")
            print(f"COLLECTING DATA FOR {year} SEASON")
            print(f"{'='*80}")

            self._collect_year_data(year)

            # Mark year as completed
            progress['completed_years'] = progress.get('completed_years', []) + [year]
            self._save_progress(progress)

            # Save checkpoint after each year
            self._save_checkpoint()

        # Final export
        print("\n" + "="*80)
        print("EXPORTING FINAL DATA")
        print("="*80)
        self._export_data()

        print("\n" + "="*80)
        print("DATA COLLECTION COMPLETE!")
        print("="*80)

    def _collect_year_data(self, year: int):
        """
        Collect data for a single year using statcast data directly
        """
        print(f"Fetching {year} season games...")

        try:
            # Instead of using schedule, fetch statcast data by date ranges
            # Get all game dates for the year from statcast
            start_date = f"{year}-03-15"
            end_date = f"{year}-11-15"

            print(f"Fetching statcast data from {start_date} to {end_date}...")

            # Fetch sample statcast data to get game dates
            statcast_sample = pyb.statcast(start_dt=start_date, end_dt=end_date)

            if statcast_sample is not None and not statcast_sample.empty:
                # Get unique game dates
                game_dates = sorted(statcast_sample['game_date'].unique())
                total_games = len(game_dates)

                print(f"Found {total_games} game dates in {year}")

                # Process each game date (limit to 3 for testing)
                for i, game_date in enumerate(game_dates[:3], 1):
                    print(f"\nProcessing {year} game {i}/{min(10, total_games)}: {game_date}")
                    self._collect_game_date_data(year, game_date)

                    # Rate limiting
                    time.sleep(0.5)

                    # Save checkpoint every 10 games
                    if i % 10 == 0:
                        self._save_checkpoint()
            else:
                print(f"No statcast data found for {year}")

        except Exception as e:
            print(f"Error collecting data for {year}: {e}")

    def _collect_game_date_data(self, year: int, game_date):
        """
        Collect data for all games on a specific date
        """
        try:
            # Convert game_date to string if needed
            if isinstance(game_date, pd.Timestamp):
                game_date_str = game_date.strftime('%Y-%m-%d')
            else:
                game_date_str = str(game_date)

            # Fetch all games on this date
            games = self._fetch_games_on_date(game_date_str)

            if not games:
                return

            for game in games:
                game_id = game['game_id']
                print(f"    Game ID: {game_id}")

                # Fetch lineups (returns all batters in game)
                all_lineup, _ = self._fetch_game_lineups(game_id, game_date_str)

                if not all_lineup:
                    print(f"      No lineup data found, skipping...")
                    continue

                # Fetch HR outcomes for this game
                hr_outcomes = self._fetch_hr_outcomes(game_id, game_date_str)

                # Process each batter
                all_batters = []

                for batter in all_lineup:
                    batter_data = self._process_batter(
                        batter, game, game_date_str, 'unknown', hr_outcomes
                    )
                    if batter_data:
                        all_batters.append(batter_data)
                        self.training_rows.append(batter_data)

                # Store game-level data
                game_data = {
                    'game_id': game_id,
                    'date': game_date_str,
                    'year': year,
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'stadium': game.get('stadium', ''),
                    'batters': all_batters,
                    'total_hrs': sum(hr_outcomes.values()) if hr_outcomes else 0
                }

                self.games_data.append(game_data)

                print(f"      Processed {len(all_batters)} batters, {game_data['total_hrs']} HRs")

        except Exception as e:
            print(f"  Error collecting game date {game_date}: {e}")

    def _fetch_games_on_date(self, game_date: str) -> List[Dict]:
        """
        Fetch all games on a specific date from statcast data
        """
        try:
            # Fetch statcast data for this date
            statcast_data = pyb.statcast(start_dt=game_date, end_dt=game_date)

            if statcast_data is None or statcast_data.empty:
                return []

            # Get unique game_pk (game identifiers)
            unique_games = statcast_data['game_pk'].unique()

            games = []

            for game_pk in unique_games:
                game_data = statcast_data[statcast_data['game_pk'] == game_pk]

                if game_data.empty:
                    continue

                # Extract home and away teams from the data
                # In statcast, we can infer teams from player data
                first_row = game_data.iloc[0]

                # Create a simple game record
                # Note: We'll use generic identifiers since team names aren't directly in statcast
                game_id = f"{game_date}_{game_pk}"

                games.append({
                    'game_id': game_id,
                    'game_pk': game_pk,
                    'date': game_date,
                    'home_team': 'Unknown',  # Will be inferred from player data
                    'away_team': 'Unknown',
                    'stadium': 'Unknown'
                })

            return games

        except Exception as e:
            print(f"    Error fetching games on {game_date}: {e}")
            return []

    def _fetch_game_lineups(self, game_id: str, game_date: str) -> Tuple[List, List]:
        """
        Fetch lineups for both teams using statcast data
        """
        try:
            # Extract game_pk from game_id
            parts = game_id.split('_')
            if len(parts) < 2:
                return [], []

            game_pk = int(parts[-1])

            # Fetch statcast data for this date
            statcast_data = pyb.statcast(start_dt=game_date, end_dt=game_date)

            if statcast_data is None or statcast_data.empty:
                return [], []

            # Filter for this specific game
            game_data = statcast_data[statcast_data['game_pk'] == game_pk]

            if game_data.empty:
                return [], []

            # Get unique batters and their info
            batters = game_data.groupby(['batter', 'player_name', 'stand']).size().reset_index()
            batters = batters.rename(columns={0: 'count'})

            all_batters = []

            for _, row in batters.iterrows():
                batter_info = {
                    'player_id': int(row['batter']),
                    'player_name': row['player_name'],
                    'bat_side': row['stand']
                }
                all_batters.append(batter_info)

            # For simplicity, return all batters in one list
            # In reality, we'd need additional data to separate home/away
            return all_batters, []

        except Exception as e:
            print(f"      Error fetching lineups: {e}")
            return [], []

    def _fetch_player_metrics_asof(self, player_id: int, player_name: str,
                                   game_date: str, lookback_days: int = 30) -> Dict:
        """
        Fetch player metrics as-of a specific game date
        Only uses data available BEFORE that date
        """
        try:
            date_obj = pd.to_datetime(game_date)

            # Calculate date ranges
            season_start = f"{date_obj.year}-03-01"
            day_before_game = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')

            last_30_start = (date_obj - timedelta(days=30)).strftime('%Y-%m-%d')
            last_7_start = (date_obj - timedelta(days=7)).strftime('%Y-%m-%d')

            metrics = {}

            # Fetch season stats (up to day before game)
            try:
                season_data = pyb.statcast_batter(start_dt=season_start, end_dt=day_before_game,
                                                  player_id=player_id)
                if season_data is not None and not season_data.empty:
                    metrics['season'] = self._calculate_batter_metrics(season_data)
                    metrics['season_pa'] = len(season_data)
            except:
                metrics['season'] = {}
                metrics['season_pa'] = 0

            # Fetch last 30 days
            try:
                month_data = pyb.statcast_batter(start_dt=last_30_start, end_dt=day_before_game,
                                                 player_id=player_id)
                if month_data is not None and not month_data.empty:
                    metrics['last_30'] = self._calculate_batter_metrics(month_data)
                    metrics['month_pa'] = len(month_data)
            except:
                metrics['last_30'] = {}
                metrics['month_pa'] = 0

            # Fetch last 7 days
            try:
                week_data = pyb.statcast_batter(start_dt=last_7_start, end_dt=day_before_game,
                                               player_id=player_id)
                if week_data is not None and not week_data.empty:
                    metrics['last_7'] = self._calculate_batter_metrics(week_data)
                    metrics['week_pa'] = len(week_data)
            except:
                metrics['last_7'] = {}
                metrics['week_pa'] = 0

            return metrics

        except Exception as e:
            print(f"        Error fetching metrics for {player_name}: {e}")
            return {}

    def _calculate_batter_metrics(self, statcast_data) -> Dict:
        """Calculate batter metrics from statcast data"""
        if statcast_data is None or statcast_data.empty:
            return {}

        metrics = {}

        try:
            # Batted ball metrics
            batted_balls = statcast_data[statcast_data['type'] == 'X']

            if not batted_balls.empty:
                # Barrel percentage
                barrels = (batted_balls['launch_speed'] >= 98) & \
                         (batted_balls['launch_angle'].between(26, 30))
                metrics['barrel_pct'] = barrels.mean() * 100 if len(batted_balls) > 0 else 0

                # Hard hit percentage
                metrics['hard_hit_pct'] = (batted_balls['launch_speed'] >= 95).mean() * 100

                # Exit velocity
                metrics['avg_exit_velo'] = batted_balls['launch_speed'].mean()
                metrics['max_exit_velo'] = batted_balls['launch_speed'].max()

                # Launch angle
                metrics['avg_launch_angle'] = batted_balls['launch_angle'].mean()

                # Sweet spot
                sweet_spot = batted_balls['launch_angle'].between(8, 32)
                metrics['sweet_spot_pct'] = sweet_spot.mean() * 100

            # Outcome metrics
            plate_appearances = statcast_data.groupby(['game_date', 'at_bat_number']).size()
            pa_count = len(plate_appearances)

            if pa_count > 0:
                # HR rate
                hrs = statcast_data['events'].eq('home_run').sum()
                metrics['hr_rate'] = hrs / pa_count

                # K and BB rates
                strikeouts = statcast_data['events'].isin(['strikeout', 'strikeout_double_play']).sum()
                walks = statcast_data['events'].eq('walk').sum()

                metrics['k_pct'] = (strikeouts / pa_count) * 100
                metrics['bb_pct'] = (walks / pa_count) * 100

                # ISO
                abs = statcast_data['events'].isin(['single', 'double', 'triple', 'home_run',
                                                    'field_out', 'grounded_into_double_play']).sum()
                if abs > 0:
                    hits = statcast_data['events'].isin(['single', 'double', 'triple', 'home_run']).sum()
                    total_bases = (statcast_data['events'].eq('single').sum() +
                                 statcast_data['events'].eq('double').sum() * 2 +
                                 statcast_data['events'].eq('triple').sum() * 3 +
                                 statcast_data['events'].eq('home_run').sum() * 4)
                    avg = hits / abs
                    slg = total_bases / abs
                    metrics['iso'] = slg - avg

        except Exception as e:
            print(f"          Error calculating metrics: {e}")

        return metrics

    def _fetch_hr_outcomes(self, game_id: str, game_date: str) -> Dict:
        """
        Fetch actual HR outcomes from statcast data
        Returns dict mapping player_name -> number of HRs hit
        """
        try:
            # Fetch statcast data for this date
            statcast_data = pyb.statcast(start_dt=game_date, end_dt=game_date)

            if statcast_data is None or statcast_data.empty:
                return {}

            # Get home runs
            hrs = statcast_data[statcast_data['events'] == 'home_run']

            if hrs.empty:
                return {}

            # Count HRs per player
            hr_counts = hrs.groupby('player_name').size().to_dict()

            return hr_counts

        except Exception as e:
            print(f"      Error fetching HR outcomes: {e}")
            return {}

    def _process_batter(self, batter: Dict, game: Dict, game_date: str,
                       home_away: str, hr_outcomes: Dict) -> Optional[Dict]:
        """
        Process a single batter and create training row
        """
        try:
            player_name = batter.get('player_name', '')
            player_id = batter.get('player_id', 0)

            if not player_name or not player_id:
                return None

            # Fetch metrics as-of game date
            metrics = self._fetch_player_metrics_asof(player_id, player_name, game_date)

            # Get HR outcome
            hr_hit = hr_outcomes.get(player_name, 0)

            # Get stadium info
            home_team = game['home_team']
            stadium_info = self.stadiums.get(home_team, {})

            # Create training row
            # Handle team assignment
            if home_away in ['home', 'away']:
                team = game[f'{home_away}_team']
                opponent = game['away_team' if home_away == 'home' else 'home_team']
            else:
                team = 'Unknown'
                opponent = 'Unknown'

            row = {
                # Identifiers
                'game_id': game['game_id'],
                'date': game_date,
                'player_name': player_name,
                'player_id': player_id,
                'team': team,
                'opponent': opponent,
                'home_away': home_away,

                # Stadium
                'stadium': game.get('stadium', ''),
                'stadium_hr_factor': stadium_info.get('hr_factor', 1.0),
                'elevation': stadium_info.get('elevation', 0),

                # Player info
                'bat_side': batter.get('bat_side', 'R'),

                # Season metrics
                'season_pa': metrics.get('season_pa', 0),
                'season_barrel_pct': metrics.get('season', {}).get('barrel_pct', 0),
                'season_hard_hit_pct': metrics.get('season', {}).get('hard_hit_pct', 0),
                'season_avg_exit_velo': metrics.get('season', {}).get('avg_exit_velo', 0),
                'season_hr_rate': metrics.get('season', {}).get('hr_rate', 0),
                'season_k_pct': metrics.get('season', {}).get('k_pct', 0),
                'season_bb_pct': metrics.get('season', {}).get('bb_pct', 0),
                'season_iso': metrics.get('season', {}).get('iso', 0),

                # Last 30 days metrics
                'month_pa': metrics.get('month_pa', 0),
                'month_barrel_pct': metrics.get('last_30', {}).get('barrel_pct', 0),
                'month_hard_hit_pct': metrics.get('last_30', {}).get('hard_hit_pct', 0),
                'month_hr_rate': metrics.get('last_30', {}).get('hr_rate', 0),

                # Last 7 days metrics
                'week_pa': metrics.get('week_pa', 0),
                'week_barrel_pct': metrics.get('last_7', {}).get('barrel_pct', 0),
                'week_hr_rate': metrics.get('last_7', {}).get('hr_rate', 0),

                # TARGET VARIABLE
                'hr_hit': 1 if hr_hit > 0 else 0,
                'hr_count': hr_hit
            }

            # Add historical weather if available (optional - historical weather APIs are limited)
            # For now, we'll use stadium factors which are constant
            # In production, you could integrate with historical weather APIs

            return row

        except Exception as e:
            print(f"        Error processing batter: {e}")
            return None

    def _fetch_historical_weather(self, stadium: str, game_date: str) -> Dict:
        """
        Fetch historical weather data for a game
        Note: Historical weather requires paid APIs (e.g., Visual Crossing, WeatherAPI)
        This is a placeholder for future implementation
        """
        # Placeholder - would integrate with historical weather API
        # For now, return default values
        return {
            'temperature': 72,  # Default
            'wind_speed': 5,
            'humidity': 50,
            'conditions': 'Clear'
        }

    def _load_progress(self) -> Dict:
        """Load collection progress from file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'completed_years': []}

    def _save_progress(self, progress: Dict):
        """Save collection progress"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            print(f"Error saving progress: {e}")

    def _save_checkpoint(self):
        """Save checkpoint data"""
        try:
            checkpoint = {
                'games_collected': len(self.games_data),
                'training_rows': len(self.training_rows),
                'timestamp': datetime.now().isoformat()
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"Checkpoint saved: {checkpoint['games_collected']} games, {checkpoint['training_rows']} training rows")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def _export_data(self):
        """
        Export collected data to CSV and JSON formats
        """
        # 1. Export to JSON (full nested structure)
        json_file = f"historical_hr_data_{self.start_year}_{self.end_year}.json"
        try:
            with open(json_file, 'w') as f:
                json.dump({
                    'metadata': {
                        'start_year': self.start_year,
                        'end_year': self.end_year,
                        'total_games': len(self.games_data),
                        'total_training_rows': len(self.training_rows),
                        'collection_date': datetime.now().isoformat()
                    },
                    'games': self.games_data
                }, f, indent=2, default=str)
            print(f"✅ JSON exported: {json_file}")
        except Exception as e:
            print(f"❌ Error exporting JSON: {e}")

        # 2. Export to CSV (flat structure for ML)
        csv_file = f"historical_hr_training_data_{self.start_year}_{self.end_year}.csv"
        try:
            if self.training_rows:
                df = pd.DataFrame(self.training_rows)
                df.to_csv(csv_file, index=False)
                print(f"✅ CSV exported: {csv_file}")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
            else:
                print("⚠️  No training rows to export to CSV")
        except Exception as e:
            print(f"❌ Error exporting CSV: {e}")

    def display_summary(self):
        """Display collection summary"""
        print("\n" + "="*80)
        print("COLLECTION SUMMARY")
        print("="*80)
        print(f"Total games collected: {len(self.games_data)}")
        print(f"Total training rows: {len(self.training_rows)}")
        print(f"Period: {self.start_year}-{self.end_year}")
        print("="*80)


def main():
    """
    Main function
    """
    print("HISTORICAL HR DATA COLLECTOR")
    print("="*80)
    print("This will collect historical game data from 2018-2024 seasons")
    print("Data includes: player metrics, weather, ballpark factors, and actual HR outcomes")
    print("\nEstimated time: 6-12 hours (depending on API rate limits)")
    print("Data will be saved as both CSV and JSON formats")
    print("="*80)

    proceed = input("\nProceed with data collection? (y/n): ").strip().lower()

    if proceed != 'y':
        print("Collection cancelled")
        return

    # Initialize collector (testing with 2024 only first)
    collector = HistoricalDataCollector(start_year=2024, end_year=2024)

    # Collect all data
    collector.collect_all_data()

    # Display summary
    collector.display_summary()

    print("\n✅ Done! Your training data is ready.")


if __name__ == "__main__":
    main()
