"""
hit_metrics_fetcher.py

Fetches and blends all Statcast and leaderboard metrics that influence
a player's probability of getting at least one hit in a game.

Workflow:
1. Load lineups from final_lineups.json
2. Fetch hitter metrics (season, 30-day, 7-day)
3. Fetch pitcher metrics (season, 30-day, 7-day)
4. Blend metrics using weighted formula
5. Output final_integrated_hit_data.json
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from pybaseball import (
    statcast_batter,
    statcast_pitcher,
    batting_stats,
    pitching_stats,
    playerid_lookup
)

# Suppress pandas warnings
import warnings
warnings.filterwarnings('ignore')


class HitMetricsFetcher:
    """Fetches and processes hit-related metrics for batters and pitchers."""

    def __init__(self, cache_db: str = "hit_metrics_cache.db"):
        self.cache_db = cache_db
        self.current_year = datetime.now().year
        # Check if we're past baseball season - use 2025 for current season
        current_month = datetime.now().month
        if current_month >= 3:  # Season starts in March/April
            self.current_year = 2025
        self._init_cache()

    def _init_cache(self):
        """Initialize SQLite cache for metrics data."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        # Cache for batter metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batter_metrics (
                player_name TEXT,
                metric_type TEXT,
                data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_name, metric_type)
            )
        """)

        # Cache for pitcher metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pitcher_metrics (
                player_name TEXT,
                metric_type TEXT,
                data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_name, metric_type)
            )
        """)

        conn.commit()
        conn.close()
        print(f"[INFO] Cache initialized at {self.cache_db}")

    def _get_cached_data(self, player_name: str, metric_type: str, table: str) -> Optional[Dict]:
        """Retrieve cached data if available and recent (< 6 hours old)."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT data, timestamp FROM {table}
            WHERE player_name = ? AND metric_type = ?
        """, (player_name, metric_type))

        result = cursor.fetchone()
        conn.close()

        if result:
            data_json, timestamp = result
            cached_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            if datetime.now() - cached_time < timedelta(hours=6):
                return json.loads(data_json)

        return None

    def _cache_data(self, player_name: str, metric_type: str, data: Dict, table: str):
        """Cache metrics data."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute(f"""
            INSERT OR REPLACE INTO {table} (player_name, metric_type, data, timestamp)
            VALUES (?, ?, ?, ?)
        """, (player_name, metric_type, json.dumps(data), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
        conn.close()

    def _blend_metrics(self, season: float, last_30: float, last_7: float) -> float:
        """
        Blend metrics using weighted formula:
        blended = (0.60 * season) + (0.25 * last_30) + (0.15 * last_7)
        """
        # Handle None values
        season = season if season is not None else 0.0
        last_30 = last_30 if last_30 is not None else season
        last_7 = last_7 if last_7 is not None else last_30

        return round((0.60 * season) + (0.25 * last_30) + (0.15 * last_7), 4)

    def _safe_get(self, df: pd.DataFrame, column: str, default: float = 0.0) -> float:
        """Safely extract a value from a DataFrame column."""
        if df is None or df.empty or column not in df.columns:
            return default

        val = df[column].iloc[0] if len(df) > 0 else default
        return float(val) if pd.notna(val) else default

    def _fetch_batter_season_stats(self, player_name: str) -> Dict[str, float]:
        """Fetch season-level batting statistics."""
        print(f"[INFO] Fetching season stats for batter: {player_name}")

        # Check cache first
        cached = self._get_cached_data(player_name, "season", "batter_metrics")
        if cached:
            print(f"[INFO] Using cached season data for {player_name}")
            return cached

        try:
            # Fetch batting stats for current year
            stats_df = batting_stats(self.current_year, self.current_year)
            if stats_df is None or stats_df.empty:
                return {}

            # Try to find player (fuzzy match on last name)
            last_name = player_name.split()[-1]
            player_df = stats_df[stats_df['Name'].str.contains(last_name, case=False, na=False)]

            if player_df.empty:
                print(f"[WARN] No season stats found for {player_name}")
                return {}

            # Take first match
            player_df = player_df.iloc[0:1]

            metrics = {
                'avg': self._safe_get(player_df, 'AVG', 0.250),
                'obp': self._safe_get(player_df, 'OBP', 0.300),
                'babip': self._safe_get(player_df, 'BABIP', 0.290),
                'k_pct': self._safe_get(player_df, 'K%', 20.0) / 100.0,
                'bb_pct': self._safe_get(player_df, 'BB%', 8.0) / 100.0,
            }

            # Cache results
            self._cache_data(player_name, "season", metrics, "batter_metrics")
            return metrics

        except Exception as e:
            print(f"[ERROR] Failed to fetch season stats for {player_name}: {e}")
            return {}

    def _fetch_batter_statcast(self, player_name: str, days: int) -> Dict[str, float]:
        """Fetch Statcast data for a batter over the last N days."""
        print(f"[INFO] Fetching {days}-day Statcast data for {player_name}")

        # Check cache
        metric_type = f"statcast_{days}d"
        cached = self._get_cached_data(player_name, metric_type, "batter_metrics")
        if cached:
            print(f"[INFO] Using cached {days}-day data for {player_name}")
            return cached

        try:
            # Use dates from current baseball season (2025)
            # If we're in October, use September games as recent data
            end_date = "2025-09-30"  # End of regular season
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")

            # Get player ID first
            name_parts = player_name.split()
            if len(name_parts) < 2:
                print(f"[WARN] Invalid player name format: {player_name}")
                return {}

            first_name = name_parts[0]
            last_name = name_parts[-1]

            # Lookup player ID
            player_lookup = playerid_lookup(last_name, first_name)
            if player_lookup is None or player_lookup.empty:
                print(f"[WARN] Could not find player ID for {player_name}")
                return {}

            player_id = player_lookup.iloc[0]['key_mlbam']

            # Fetch Statcast data using player ID
            sc_df = statcast_batter(start_date, end_date, player_id)

            if sc_df is None or sc_df.empty:
                print(f"[WARN] No Statcast data found for {player_name} ({days} days)")
                return {}

            # Calculate metrics
            balls_in_play = sc_df[sc_df['type'] == 'X']
            hits = sc_df[sc_df['events'].isin(['single', 'double', 'triple', 'home_run'])]

            metrics = {
                'avg': len(hits) / max(len(balls_in_play), 1),
                'contact_pct': len(sc_df[sc_df['description'].str.contains('hit_into_play|foul', case=False, na=False)]) / max(len(sc_df), 1),
                'hard_hit_pct': len(sc_df[sc_df['launch_speed'] >= 95]) / max(len(balls_in_play), 1),
                'launch_speed': sc_df['launch_speed'].mean() if 'launch_speed' in sc_df.columns else 0.0,
                'launch_angle': sc_df['launch_angle'].mean() if 'launch_angle' in sc_df.columns else 0.0,
                'barrel_pct': len(sc_df[sc_df['barrel'] == 1]) / max(len(balls_in_play), 1) if 'barrel' in sc_df.columns else 0.0,
                'ld_pct': len(sc_df[sc_df['bb_type'] == 'line_drive']) / max(len(balls_in_play), 1),
                'gb_pct': len(sc_df[sc_df['bb_type'] == 'ground_ball']) / max(len(balls_in_play), 1),
                'fb_pct': len(sc_df[sc_df['bb_type'] == 'fly_ball']) / max(len(balls_in_play), 1),
                'swstr_pct': len(sc_df[sc_df['description'].str.contains('swinging_strike', case=False, na=False)]) / max(len(sc_df), 1),
            }

            # Round all values
            metrics = {k: round(v, 4) for k, v in metrics.items()}

            # Cache results
            self._cache_data(player_name, metric_type, metrics, "batter_metrics")
            return metrics

        except Exception as e:
            print(f"[ERROR] Failed to fetch Statcast data for {player_name} ({days} days): {e}")
            return {}

    def _fetch_batter_metrics(self, player_name: str, bat_side: str) -> Dict[str, Any]:
        """
        Fetch and blend all batter metrics:
        - Season stats
        - 30-day stats
        - 7-day stats
        - Platoon splits
        """
        print(f"[INFO] Processing batter: {player_name} ({bat_side})")

        # Fetch all time periods
        season_stats = self._fetch_batter_season_stats(player_name)
        stats_30d = self._fetch_batter_statcast(player_name, 30)
        stats_7d = self._fetch_batter_statcast(player_name, 7)

        # Blend key metrics
        blended = {}

        # Contact & Discipline
        blended['contact_pct'] = self._blend_metrics(
            season_stats.get('contact_pct', stats_30d.get('contact_pct', 0.75)),
            stats_30d.get('contact_pct', 0.75),
            stats_7d.get('contact_pct', 0.75)
        )

        blended['k_pct'] = self._blend_metrics(
            season_stats.get('k_pct', 0.22),
            stats_30d.get('k_pct', 0.22),
            stats_7d.get('k_pct', 0.22)
        )

        blended['swstr_pct'] = self._blend_metrics(
            stats_30d.get('swstr_pct', 0.11),
            stats_30d.get('swstr_pct', 0.11),
            stats_7d.get('swstr_pct', 0.11)
        )

        # Quality of Contact
        blended['launch_speed'] = self._blend_metrics(
            stats_30d.get('launch_speed', 87.0),
            stats_30d.get('launch_speed', 87.0),
            stats_7d.get('launch_speed', 87.0)
        )

        blended['launch_angle'] = self._blend_metrics(
            stats_30d.get('launch_angle', 12.0),
            stats_30d.get('launch_angle', 12.0),
            stats_7d.get('launch_angle', 12.0)
        )

        blended['hard_hit_pct'] = self._blend_metrics(
            stats_30d.get('hard_hit_pct', 0.35),
            stats_30d.get('hard_hit_pct', 0.35),
            stats_7d.get('hard_hit_pct', 0.35)
        )

        blended['barrel_pct'] = self._blend_metrics(
            stats_30d.get('barrel_pct', 0.08),
            stats_30d.get('barrel_pct', 0.08),
            stats_7d.get('barrel_pct', 0.08)
        )

        # Results
        blended['avg'] = self._blend_metrics(
            season_stats.get('avg', 0.250),
            stats_30d.get('avg', 0.250),
            stats_7d.get('avg', 0.250)
        )

        blended['obp'] = self._blend_metrics(
            season_stats.get('obp', 0.310),
            stats_30d.get('obp', 0.310),
            stats_7d.get('obp', 0.310)
        )

        blended['babip'] = self._blend_metrics(
            season_stats.get('babip', 0.290),
            stats_30d.get('babip', 0.290),
            stats_7d.get('babip', 0.290)
        )

        # Batted ball profile
        blended['ld_pct'] = self._blend_metrics(
            stats_30d.get('ld_pct', 0.21),
            stats_30d.get('ld_pct', 0.21),
            stats_7d.get('ld_pct', 0.21)
        )

        blended['gb_pct'] = self._blend_metrics(
            stats_30d.get('gb_pct', 0.45),
            stats_30d.get('gb_pct', 0.45),
            stats_7d.get('gb_pct', 0.45)
        )

        blended['fb_pct'] = self._blend_metrics(
            stats_30d.get('fb_pct', 0.34),
            stats_30d.get('fb_pct', 0.34),
            stats_7d.get('fb_pct', 0.34)
        )

        return blended

    def _fetch_pitcher_season_stats(self, player_name: str) -> Dict[str, float]:
        """Fetch season-level pitching statistics."""
        print(f"[INFO] Fetching season stats for pitcher: {player_name}")

        # Check cache first
        cached = self._get_cached_data(player_name, "season", "pitcher_metrics")
        if cached:
            print(f"[INFO] Using cached season data for {player_name}")
            return cached

        try:
            # Fetch pitching stats for current year
            stats_df = pitching_stats(self.current_year, self.current_year)
            if stats_df is None or stats_df.empty:
                return {}

            # Try to find player (fuzzy match on last name)
            last_name = player_name.split()[-1]
            player_df = stats_df[stats_df['Name'].str.contains(last_name, case=False, na=False)]

            if player_df.empty:
                print(f"[WARN] No season stats found for {player_name}")
                return {}

            # Take first match
            player_df = player_df.iloc[0:1]

            metrics = {
                'k_pct': self._safe_get(player_df, 'K%', 20.0) / 100.0,
                'bb_pct': self._safe_get(player_df, 'BB%', 8.0) / 100.0,
                'era': self._safe_get(player_df, 'ERA', 4.00),
                'whip': self._safe_get(player_df, 'WHIP', 1.30),
                'xfip': self._safe_get(player_df, 'xFIP', 4.00),
                'babip_allowed': self._safe_get(player_df, 'BABIP', 0.290),
            }

            # Cache results
            self._cache_data(player_name, "season", metrics, "pitcher_metrics")
            return metrics

        except Exception as e:
            print(f"[ERROR] Failed to fetch season stats for {player_name}: {e}")
            return {}

    def _fetch_pitcher_statcast(self, player_name: str, days: int) -> Dict[str, float]:
        """Fetch Statcast data for a pitcher over the last N days."""
        print(f"[INFO] Fetching {days}-day Statcast data for pitcher {player_name}")

        # Check cache
        metric_type = f"statcast_{days}d"
        cached = self._get_cached_data(player_name, metric_type, "pitcher_metrics")
        if cached:
            print(f"[INFO] Using cached {days}-day data for {player_name}")
            return cached

        try:
            # Use dates from current baseball season (2025)
            end_date = "2025-09-30"  # End of regular season
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")

            # Get player ID first
            name_parts = player_name.split()
            if len(name_parts) < 2:
                print(f"[WARN] Invalid player name format: {player_name}")
                return {}

            first_name = name_parts[0]
            last_name = name_parts[-1]

            # Lookup player ID
            player_lookup = playerid_lookup(last_name, first_name)
            if player_lookup is None or player_lookup.empty:
                print(f"[WARN] Could not find player ID for {player_name}")
                return {}

            player_id = player_lookup.iloc[0]['key_mlbam']

            # Fetch Statcast data using player ID
            sc_df = statcast_pitcher(start_date, end_date, player_id)

            if sc_df is None or sc_df.empty:
                print(f"[WARN] No Statcast data found for {player_name} ({days} days)")
                return {}

            # Calculate metrics
            balls_in_play = sc_df[sc_df['type'] == 'X']
            hits_allowed = sc_df[sc_df['events'].isin(['single', 'double', 'triple', 'home_run'])]

            metrics = {
                'avg_allowed': len(hits_allowed) / max(len(balls_in_play), 1),
                'hard_hit_pct_allowed': len(sc_df[sc_df['launch_speed'] >= 95]) / max(len(balls_in_play), 1),
                'barrel_pct_allowed': len(sc_df[sc_df['barrel'] == 1]) / max(len(balls_in_play), 1) if 'barrel' in sc_df.columns else 0.0,
                'contact_allowed_pct': len(sc_df[sc_df['description'].str.contains('hit_into_play|foul', case=False, na=False)]) / max(len(sc_df), 1),
                'swstr_pct': len(sc_df[sc_df['description'].str.contains('swinging_strike', case=False, na=False)]) / max(len(sc_df), 1),
                'gb_pct': len(sc_df[sc_df['bb_type'] == 'ground_ball']) / max(len(balls_in_play), 1),
                'ld_pct': len(sc_df[sc_df['bb_type'] == 'line_drive']) / max(len(balls_in_play), 1),
                'fb_pct': len(sc_df[sc_df['bb_type'] == 'fly_ball']) / max(len(balls_in_play), 1),
            }

            # Round all values
            metrics = {k: round(v, 4) for k, v in metrics.items()}

            # Cache results
            self._cache_data(player_name, metric_type, metrics, "pitcher_metrics")
            return metrics

        except Exception as e:
            print(f"[ERROR] Failed to fetch Statcast data for {player_name} ({days} days): {e}")
            return {}

    def _fetch_pitcher_metrics(self, player_name: str, throws: str) -> Dict[str, Any]:
        """
        Fetch and blend all pitcher metrics:
        - Season stats
        - 30-day stats
        - 7-day stats
        """
        print(f"[INFO] Processing pitcher: {player_name} ({throws})")

        # Fetch all time periods
        season_stats = self._fetch_pitcher_season_stats(player_name)
        stats_30d = self._fetch_pitcher_statcast(player_name, 30)
        stats_7d = self._fetch_pitcher_statcast(player_name, 7)

        # Blend key metrics
        blended = {}

        # Contact suppression
        blended['k_pct'] = self._blend_metrics(
            season_stats.get('k_pct', 0.22),
            stats_30d.get('k_pct', 0.22),
            stats_7d.get('k_pct', 0.22)
        )

        blended['swstr_pct'] = self._blend_metrics(
            stats_30d.get('swstr_pct', 0.11),
            stats_30d.get('swstr_pct', 0.11),
            stats_7d.get('swstr_pct', 0.11)
        )

        blended['contact_allowed_pct'] = self._blend_metrics(
            stats_30d.get('contact_allowed_pct', 0.75),
            stats_30d.get('contact_allowed_pct', 0.75),
            stats_7d.get('contact_allowed_pct', 0.75)
        )

        # Contact quality allowed
        blended['hard_hit_pct_allowed'] = self._blend_metrics(
            stats_30d.get('hard_hit_pct_allowed', 0.35),
            stats_30d.get('hard_hit_pct_allowed', 0.35),
            stats_7d.get('hard_hit_pct_allowed', 0.35)
        )

        blended['barrel_pct_allowed'] = self._blend_metrics(
            stats_30d.get('barrel_pct_allowed', 0.08),
            stats_30d.get('barrel_pct_allowed', 0.08),
            stats_7d.get('barrel_pct_allowed', 0.08)
        )

        blended['babip_allowed'] = self._blend_metrics(
            season_stats.get('babip_allowed', 0.290),
            stats_30d.get('babip_allowed', 0.290),
            stats_7d.get('babip_allowed', 0.290)
        )

        # Batted-ball profile
        blended['gb_pct'] = self._blend_metrics(
            stats_30d.get('gb_pct', 0.45),
            stats_30d.get('gb_pct', 0.45),
            stats_7d.get('gb_pct', 0.45)
        )

        blended['ld_pct'] = self._blend_metrics(
            stats_30d.get('ld_pct', 0.21),
            stats_30d.get('ld_pct', 0.21),
            stats_7d.get('ld_pct', 0.21)
        )

        blended['fb_pct'] = self._blend_metrics(
            stats_30d.get('fb_pct', 0.34),
            stats_30d.get('fb_pct', 0.34),
            stats_7d.get('fb_pct', 0.34)
        )

        # Run environment
        blended['era'] = self._blend_metrics(
            season_stats.get('era', 4.00),
            stats_30d.get('era', 4.00),
            stats_7d.get('era', 4.00)
        )

        blended['xfip'] = self._blend_metrics(
            season_stats.get('xfip', 4.00),
            stats_30d.get('xfip', 4.00),
            stats_7d.get('xfip', 4.00)
        )

        blended['whip'] = self._blend_metrics(
            season_stats.get('whip', 1.30),
            stats_30d.get('whip', 1.30),
            stats_7d.get('whip', 1.30)
        )

        return blended

    def process_lineups(self, input_file: str = "mlb_lineups.json",
                       output_file: str = "final_integrated_hit_data.json"):
        """
        Main processing function:
        1. Load lineups
        2. Fetch batter metrics
        3. Fetch pitcher metrics
        4. Integrate and save
        """
        print(f"[INFO] Loading lineups from {input_file}")

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lineups_data = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] File not found: {input_file}")
            return
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in {input_file}: {e}")
            return

        integrated_data = {"games": {}}

        # Process each game
        for game_key, game_data in lineups_data.get("games", {}).items():
            print(f"\n[INFO] Processing game: {game_key}")

            integrated_game = {
                "home_team": game_data.get("home_team"),
                "away_team": game_data.get("away_team"),
                "home_lineup": [],
                "away_lineup": [],
                "home_pitcher": {},
                "away_pitcher": {}
            }

            # Process home lineup
            print(f"[INFO] Processing home lineup for {game_data.get('home_team')}")
            for batter in game_data.get("home_lineup", []):
                player_name = batter.get("name", "")
                bat_side = batter.get("bat_side", "R")  # Default to R if not provided
                position = batter.get("position", "")

                batter_data = {
                    "name": player_name,
                    "position": position,
                    "bat_side": bat_side,
                    "batting_order": batter.get("batting_order", 0),
                    "blended_metrics": self._fetch_batter_metrics(player_name, bat_side)
                }
                integrated_game["home_lineup"].append(batter_data)

            # Process away lineup
            print(f"[INFO] Processing away lineup for {game_data.get('away_team')}")
            for batter in game_data.get("away_lineup", []):
                player_name = batter.get("name", "")
                bat_side = batter.get("bat_side", "R")  # Default to R if not provided
                position = batter.get("position", "")

                batter_data = {
                    "name": player_name,
                    "position": position,
                    "bat_side": bat_side,
                    "batting_order": batter.get("batting_order", 0),
                    "blended_metrics": self._fetch_batter_metrics(player_name, bat_side)
                }
                integrated_game["away_lineup"].append(batter_data)

            # Process home pitcher
            if "home_pitcher" in game_data:
                pitcher = game_data["home_pitcher"]
                pitcher_name = pitcher.get("name", "")
                throws = pitcher.get("throws", "R")

                integrated_game["home_pitcher"] = {
                    "name": pitcher_name,
                    "throws": throws,
                    "blended_metrics": self._fetch_pitcher_metrics(pitcher_name, throws)
                }

            # Process away pitcher
            if "away_pitcher" in game_data:
                pitcher = game_data["away_pitcher"]
                pitcher_name = pitcher.get("name", "")
                throws = pitcher.get("throws", "R")

                integrated_game["away_pitcher"] = {
                    "name": pitcher_name,
                    "throws": throws,
                    "blended_metrics": self._fetch_pitcher_metrics(pitcher_name, throws)
                }

            integrated_data["games"][game_key] = integrated_game

        # Save integrated data
        print(f"\n[INFO] Saving integrated data to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(integrated_data, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] Hit metrics fetching complete! Output saved to {output_file}")


def main():
    """Main execution function."""
    fetcher = HitMetricsFetcher()
    fetcher.process_lineups()


if __name__ == "__main__":
    main()
