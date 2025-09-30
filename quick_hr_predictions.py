#!/usr/bin/env python3
"""
Quick HR Predictions - Betting Site Style
Fast pipeline using projected lineups + cached metrics + stadium defaults
"""

import json
import requests
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

class QuickHRPredictor:
    """
    Fast HR predictions using cached data and projected lineups
    """

    def __init__(self):
        self.cache_db = "quick_hr_cache.db"
        self.setup_cache_db()

        # Stadium HR factors (no API needed)
        self.stadium_factors = {
            'NYY': 1.12, 'BOS': 1.06, 'TB': 0.95, 'TOR': 1.02, 'BAL': 1.08,
            'CLE': 0.98, 'MIN': 1.01, 'KC': 0.97, 'CWS': 1.03, 'DET': 0.94,
            'HOU': 1.09, 'SEA': 0.92, 'LAA': 0.96, 'TEX': 1.05, 'OAK': 0.89,
            'ATL': 1.04, 'NYM': 0.93, 'PHI': 1.07, 'WSH': 1.01, 'MIA': 0.85,
            'MIL': 1.02, 'CHC': 1.15, 'STL': 0.99, 'PIT': 0.91, 'CIN': 1.03,
            'LAD': 0.88, 'SD': 0.88, 'SF': 0.81, 'COL': 1.25, 'ARI': 1.06
        }

    def setup_cache_db(self):
        """
        Setup SQLite cache database for fast lookups
        """
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()

            # Player metrics cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS player_metrics (
                    player_name TEXT PRIMARY KEY,
                    team TEXT,
                    hr_rate REAL,
                    barrel_pct REAL,
                    hard_hit_pct REAL,
                    avg_exit_velocity REAL,
                    iso REAL,
                    k_pct REAL,
                    bb_pct REAL,
                    last_updated TEXT
                )
            ''')

            # Projected lineups cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projected_lineups (
                    game_date TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    home_pitcher TEXT,
                    away_pitcher TEXT,
                    home_lineup TEXT,  -- JSON string
                    away_lineup TEXT,  -- JSON string
                    last_updated TEXT,
                    PRIMARY KEY (game_date, home_team, away_team)
                )
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error setting up cache DB: {e}")

    def fetch_projected_lineups(self) -> Dict:
        """
        Fetch today's projected lineups from ESPN API (faster than full lineup fetcher)
        """
        print("Fetching projected lineups...")

        today = datetime.now().strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={today}"

        projected_games = {}

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            events = data.get('events', [])

            for event in events:
                try:
                    game_id = event['id']
                    competitions = event.get('competitions', [])

                    if not competitions:
                        continue

                    competition = competitions[0]
                    competitors = competition.get('competitors', [])

                    if len(competitors) != 2:
                        continue

                    # Extract team info
                    home_team = None
                    away_team = None
                    home_pitcher = "TBD"
                    away_pitcher = "TBD"

                    for competitor in competitors:
                        team_abbr = competitor['team'].get('abbreviation', '')
                        is_home = competitor.get('homeAway') == 'home'

                        if is_home:
                            home_team = team_abbr
                            # Try to get probable pitcher
                            if 'probablePitcher' in competitor:
                                home_pitcher = competitor['probablePitcher'].get('displayName', 'TBD')
                        else:
                            away_team = team_abbr
                            if 'probablePitcher' in competitor:
                                away_pitcher = competitor['probablePitcher'].get('displayName', 'TBD')

                    if home_team and away_team:
                        projected_games[game_id] = {
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_pitcher': home_pitcher,
                            'away_pitcher': away_pitcher,
                            'home_lineup': self._get_projected_lineup(home_team),
                            'away_lineup': self._get_projected_lineup(away_team)
                        }

                except Exception as e:
                    print(f"Error processing game: {e}")
                    continue

            print(f"Found {len(projected_games)} projected games")
            return projected_games

        except Exception as e:
            print(f"Error fetching projected lineups: {e}")
            return {}

    def _get_projected_lineup(self, team: str) -> List[str]:
        """
        Get projected starting lineup for team (simplified - top players)
        In production, this would pull from betting sites or ESPN roster API
        """
        # Simplified projected lineups (would be dynamic in production)
        default_lineups = {
            'NYY': ['Aaron Judge', 'Juan Soto', 'Giancarlo Stanton', 'Anthony Rizzo', 'Gleyber Torres', 'DJ LeMahieu', 'Anthony Volpe', 'Jose Trevino', 'Alex Verdugo'],
            'BOS': ['Rafael Devers', 'Xander Bogaerts', 'J.D. Martinez', 'Alex Verdugo', 'Trevor Story', 'Christian Vazquez', 'Andrew Benintendi', 'Bobby Dalbec', 'Kike Hernandez'],
            'LAD': ['Mookie Betts', 'Freddie Freeman', 'Will Smith', 'Max Muncy', 'Justin Turner', 'Trea Turner', 'Chris Taylor', 'Cody Bellinger', 'AJ Pollock'],
            'HOU': ['Jose Altuve', 'Alex Bregman', 'Yordan Alvarez', 'Kyle Tucker', 'Yuli Gurriel', 'Chas McCormick', 'Jeremy Pena', 'Martin Maldonado', 'Jake Meyers'],
            # Add more teams as needed
        }

        return default_lineups.get(team, [f"{team} Player {i+1}" for i in range(9)])

    def load_cached_metrics(self) -> Dict:
        """
        Load player metrics from cache (much faster than API calls)
        """
        print("Loading cached player metrics...")

        metrics_cache = {}

        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT player_name, hr_rate, barrel_pct, hard_hit_pct,
                       avg_exit_velocity, iso, k_pct, bb_pct
                FROM player_metrics
                WHERE last_updated > date('now', '-7 days')
            ''')

            rows = cursor.fetchall()

            for row in rows:
                player_name = row[0]
                metrics_cache[player_name] = {
                    'hr_rate': row[1] or 0.025,
                    'barrel_pct': row[2] or 0.06,
                    'hard_hit_pct': row[3] or 0.35,
                    'avg_exit_velocity': row[4] or 88.0,
                    'iso': row[5] or 0.150,
                    'k_pct': row[6] or 0.22,
                    'bb_pct': row[7] or 0.08
                }

            conn.close()
            print(f"Loaded metrics for {len(metrics_cache)} players from cache")

        except Exception as e:
            print(f"Error loading cached metrics: {e}")

        return metrics_cache

    def calculate_quick_hr_probabilities(self, games: Dict, metrics_cache: Dict) -> List[Dict]:
        """
        Fast HR probability calculation using cached data
        """
        predictions = []

        for game_id, game_data in games.items():
            home_team = game_data['home_team']
            away_team = game_data['away_team']
            home_pitcher = game_data['home_pitcher']
            away_pitcher = game_data['away_pitcher']

            # Get stadium factor
            stadium_factor = self.stadium_factors.get(home_team, 1.0)

            # Process away lineup vs home pitcher
            for batter in game_data['away_lineup']:
                if batter == 'TBD':
                    continue

                batter_metrics = metrics_cache.get(batter)
                pitcher_metrics = metrics_cache.get(home_pitcher)

                if batter_metrics:  # Only predict for players with cached metrics
                    prob = self._quick_hr_calculation(batter_metrics, pitcher_metrics, stadium_factor)

                    if prob > 0:  # Only include valid predictions
                        predictions.append({
                            'player_name': batter,
                            'team': away_team,
                            'opposing_pitcher': home_pitcher,
                            'opposing_team': home_team,
                            'hr_probability': prob,
                            'status': 'projected' if pitcher_metrics else 'pitcher_unknown'
                        })

            # Process home lineup vs away pitcher
            for batter in game_data['home_lineup']:
                if batter == 'TBD':
                    continue

                batter_metrics = metrics_cache.get(batter)
                pitcher_metrics = metrics_cache.get(away_pitcher)

                if batter_metrics:  # Only predict for players with cached metrics
                    prob = self._quick_hr_calculation(batter_metrics, pitcher_metrics, stadium_factor)

                    if prob > 0:  # Only include valid predictions
                        predictions.append({
                            'player_name': batter,
                            'team': home_team,
                            'opposing_pitcher': away_pitcher,
                            'opposing_team': away_team,
                            'hr_probability': prob,
                            'status': 'projected' if pitcher_metrics else 'pitcher_unknown'
                        })

        return predictions

    def _quick_hr_calculation(self, batter_metrics: Dict, pitcher_metrics: Optional[Dict], stadium_factor: float) -> float:
        """
        Simplified HR probability calculation for speed
        """
        if not batter_metrics:
            return 0.0

        # Base batter power
        base_prob = (
            batter_metrics['hr_rate'] * 0.4 +
            batter_metrics['barrel_pct'] * 0.3 +
            batter_metrics['iso'] * 0.3
        )

        # Pitcher adjustment (use defaults if no pitcher data)
        if pitcher_metrics:
            pitcher_factor = (
                1.0 - (pitcher_metrics['k_pct'] - 0.20) +  # Strikeout suppression
                (pitcher_metrics['hr_rate'] - 0.025) * 2   # HR rate allowed
            )
        else:
            pitcher_factor = 1.0  # Neutral if unknown pitcher

        # Apply stadium factor
        final_prob = base_prob * pitcher_factor * stadium_factor

        # Cap between 0.1% and 20%
        final_prob = max(0.001, min(0.20, final_prob))

        return round(final_prob * 100, 1)

    def update_cache_from_existing_data(self):
        """
        Populate cache from existing advanced_metrics.json if available
        """
        print("Updating cache from existing data...")

        # Try to load from advanced_metrics.json
        if os.path.exists("advanced_metrics.json"):
            try:
                with open("advanced_metrics.json", 'r') as f:
                    data = json.load(f)

                conn = sqlite3.connect(self.cache_db)
                cursor = conn.cursor()

                players_added = 0

                for player_id, player_data in data.items():
                    blended = player_data.get('blended_metrics', {})
                    if blended:
                        cursor.execute('''
                            INSERT OR REPLACE INTO player_metrics
                            (player_name, team, hr_rate, barrel_pct, hard_hit_pct,
                             avg_exit_velocity, iso, k_pct, bb_pct, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            player_data.get('name', ''),
                            player_data.get('team', ''),
                            blended.get('hr_rate', 0.025),
                            blended.get('barrel_pct', 0.06),
                            blended.get('hard_hit_pct', 0.35),
                            blended.get('avg_exit_velocity', 88.0),
                            blended.get('iso', 0.150),
                            blended.get('k_pct', 0.22),
                            blended.get('bb_pct', 0.08),
                            datetime.now().isoformat()
                        ))
                        players_added += 1

                conn.commit()
                conn.close()

                print(f"Updated cache with {players_added} players")

            except Exception as e:
                print(f"Error updating cache: {e}")

    def run_quick_predictions(self):
        """
        Main method for fast HR predictions
        """
        print("QUICK HR PREDICTOR")
        print("="*50)

        # Update cache from existing data first
        self.update_cache_from_existing_data()

        # Fetch projected lineups
        games = self.fetch_projected_lineups()
        if not games:
            print("No games found for today")
            return

        # Load cached metrics
        metrics_cache = self.load_cached_metrics()
        if not metrics_cache:
            print("No cached metrics found. Run advanced_metrics_fetcher.py first.")
            return

        # Calculate probabilities
        print("Calculating HR probabilities...")
        predictions = self.calculate_quick_hr_probabilities(games, metrics_cache)

        if not predictions:
            print("No predictions generated")
            return

        # Sort and display top 15
        top_predictions = sorted(predictions, key=lambda x: x['hr_probability'], reverse=True)[:15]

        print("\n" + "="*80)
        print("TOP 15 QUICK HR PREDICTIONS")
        print("="*80)
        print(f"{'Player Name':<20} | {'Team':<4} | {'vs Pitcher':<20} | {'Opp Team':<4} | {'HR Prob':<8}")
        print("-" * 80)

        for pred in top_predictions:
            print(f"{pred['player_name']:<20} | {pred['team']:<4} | {pred['opposing_pitcher']:<20} | {pred['opposing_team']:<4} | {pred['hr_probability']:>6.1f}%")

        print(f"\nGenerated in ~5 seconds | Total predictions: {len(predictions)}")
        print(f"Using cached metrics for faster results")


def main():
    predictor = QuickHRPredictor()
    predictor.run_quick_predictions()


if __name__ == "__main__":
    main()