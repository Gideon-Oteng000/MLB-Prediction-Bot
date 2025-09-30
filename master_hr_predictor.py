#!/usr/bin/env python3
"""
Master HR Predictor
Uses integrated data from cascading pipeline to predict home run probabilities
"""

import json
import os
from typing import Dict, List, Tuple
from datetime import datetime
import math

class MasterHRPredictor:
    """
    Master HR predictor using integrated lineup, metrics, and weather data
    """

    def __init__(self):
        self.predictions = []

    def load_integrated_data(self, data_file: str = "final_integrated_hr_data.json") -> Dict:
        """
        Load the final integrated data from cascading pipeline
        """
        if not os.path.exists(data_file):
            print(f"{data_file} not found")
            print("Run the cascading pipeline first:")
            print("1. mlb_lineups_fetcher.py")
            print("2. advanced_metrics_fetcher.py (option 2)")
            print("3. weather_ballpark_fetcher.py (option 2)")
            return {}

        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded integrated data from {data_file}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return {}

    def calculate_hr_probabilities(self, data: Dict) -> List[Dict]:
        """
        Calculate HR probability for each batter
        """
        predictions = []
        games = data.get('games', {})

        for game_key, game_data in games.items():
            # Get weather/ballpark multiplier
            weather_multiplier = self._get_weather_multiplier(data, game_key)

            # Process away team lineup
            away_team = game_data.get('away_team', 'Unknown')
            home_pitcher = game_data.get('home_pitcher', {})
            away_lineup = game_data.get('away_lineup', [])

            for batter in away_lineup:
                if batter.get('blended_metrics'):  # Only require batter metrics
                    prob = self._calculate_batter_hr_probability(
                        batter, home_pitcher, weather_multiplier
                    )
                    predictions.append({
                        'player_name': batter.get('name', 'Unknown'),
                        'team': away_team,
                        'opposing_pitcher': home_pitcher.get('name', 'Unknown'),
                        'opposing_team': game_data.get('home_team', 'Unknown'),
                        'hr_probability': prob
                    })

            # Process home team lineup
            home_team = game_data.get('home_team', 'Unknown')
            away_pitcher = game_data.get('away_pitcher', {})
            home_lineup = game_data.get('home_lineup', [])

            for batter in home_lineup:
                if batter.get('blended_metrics'):  # Only require batter metrics
                    prob = self._calculate_batter_hr_probability(
                        batter, away_pitcher, weather_multiplier
                    )
                    predictions.append({
                        'player_name': batter.get('name', 'Unknown'),
                        'team': home_team,
                        'opposing_pitcher': away_pitcher.get('name', 'Unknown'),
                        'opposing_team': away_team,
                        'hr_probability': prob
                    })

        return predictions

    def _get_weather_multiplier(self, data: Dict, game_key: str) -> float:
        """
        Get weather/ballpark multiplier for the game
        """
        weather_data = data.get('weather_ballpark', {})

        # Try different game key formats
        possible_keys = [game_key, f"mlb_{game_key}"]

        for key in possible_keys:
            if key in weather_data:
                return weather_data[key].get('total_hr_multiplier', 1.0)

        # Default multiplier if no weather data
        return 1.0

    def _calculate_batter_hr_probability(self, batter: Dict, pitcher: Dict, weather_multiplier: float) -> float:
        """
        Calculate HR probability for batter vs pitcher matchup
        """
        batter_metrics = batter.get('blended_metrics', {})
        pitcher_metrics = pitcher.get('blended_metrics', {})

        # Base batter power metrics (convert percentages to decimals)
        batter_hr_rate = batter_metrics.get('hr_rate', 0.025)
        batter_barrel_pct = batter_metrics.get('barrel_pct', 6.0) / 100  # Convert % to decimal
        batter_hard_hit_pct = batter_metrics.get('hard_hit_pct', 35.0) / 100  # Convert % to decimal
        batter_exit_velocity = batter_metrics.get('avg_exit_velocity', 88.0)
        batter_iso = batter_metrics.get('iso', 0.150)

        # Pitcher suppression metrics (use defaults if no metrics available)
        if pitcher_metrics and pitcher_metrics:
            pitcher_hr_rate = pitcher_metrics.get('hr_rate', 0.025)
            pitcher_k_pct = pitcher_metrics.get('k_pct', 22.0) / 100  # Convert % to decimal
            pitcher_hard_hit_pct = pitcher_metrics.get('hard_hit_pct', 35.0) / 100  # Convert % to decimal
            pitcher_barrel_pct = pitcher_metrics.get('barrel_pct', 6.0) / 100  # Convert % to decimal
        else:
            # Use league average defaults when no pitcher metrics
            pitcher_hr_rate = 0.025
            pitcher_k_pct = 0.22
            pitcher_hard_hit_pct = 0.35
            pitcher_barrel_pct = 0.06

        # Simpler, more stable calculation

        # Base probability from batter's HR rate (scale it down for daily game)
        base_prob = batter_hr_rate * 0.75  # Daily game is less ABs than season rate

        # Power adjustment: Smaller bonuses for high performers
        power_bonus = (batter_barrel_pct - 0.06) * 0.15 + (batter_iso - 0.150) * 0.08

        # Contact adjustment: Smaller bonus for hard hit%
        contact_bonus = (batter_hard_hit_pct - 0.35) * 0.05

        # Pitcher adjustment: Reduce for good strikeout pitchers
        pitcher_penalty = (pitcher_k_pct - 0.22) * 0.05

        # Combined probability
        matchup_prob = base_prob + power_bonus + contact_bonus - pitcher_penalty

        # Apply weather/ballpark multiplier
        final_prob = matchup_prob * weather_multiplier

        # Cap probabilities at reasonable bounds (0.1% to 12%)
        final_prob = max(0.001, min(0.12, final_prob))

        return round(final_prob * 100, 1)  # Convert to percentage

    def get_top_predictions(self, predictions: List[Dict], top_n: int = 15) -> List[Dict]:
        """
        Get top N HR predictions sorted by probability
        """
        sorted_predictions = sorted(predictions, key=lambda x: x['hr_probability'], reverse=True)
        return sorted_predictions[:top_n]

    def display_predictions(self, predictions: List[Dict]):
        """
        Display top predictions in clean format
        """
        print("\n" + "="*80)
        print("TOP 15 HR PREDICTIONS")
        print("="*80)
        print(f"{'Player Name':<20} | {'Team':<4} | {'vs Pitcher':<20} | {'Opp Team':<4} | {'HR Prob':<8}")
        print("-" * 80)

        for i, pred in enumerate(predictions, 1):
            print(f"{pred['player_name']:<20} | {pred['team']:<4} | {pred['opposing_pitcher']:<20} | {pred['opposing_team']:<4} | {pred['hr_probability']:>6.1f}%")

    def run_predictions(self):
        """
        Main method to run HR predictions
        """
        print("MASTER HR PREDICTOR")
        print("="*50)

        # Load integrated data
        data = self.load_integrated_data()
        if not data:
            return

        # Calculate probabilities
        print("Calculating HR probabilities...")
        all_predictions = self.calculate_hr_probabilities(data)

        if not all_predictions:
            print("No predictions generated. Check data quality.")
            return

        # Get top 15
        top_predictions = self.get_top_predictions(all_predictions, 15)

        # Display results
        self.display_predictions(top_predictions)

        print(f"\nTotal players analyzed: {len(all_predictions)}")
        print(f"Predictions generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """
    Main function
    """
    predictor = MasterHRPredictor()
    predictor.run_predictions()


if __name__ == "__main__":
    main()