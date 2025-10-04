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
        # Get weather data from the game itself
        game_weather = data.get('games', {}).get(game_key, {}).get('weather_ballpark', {})

        if game_weather:
            return game_weather.get('total_hr_multiplier', 1.0)

        # Default multiplier if no weather data
        return 1.0

    def _calculate_batter_hr_probability(self, batter: Dict, pitcher: Dict, weather_multiplier: float) -> float:
        """
        Statistically grounded HR probability:
        League-normalized + batter/pitcher factors + lineup slot + park/weather
        Enhanced with GB%, barrel suppression, hard-hit suppression, and platoon splits (additive adjustments)
        """
        # League baseline
        league_hr_rate = 0.033

        # Batter stats
        bm = batter.get('blended_metrics', {})
        batter_hr_rate = bm.get('hr_rate', 0.025)
        batter_barrel_pct = bm.get('barrel_pct', 7.0) / 100
        batter_iso = bm.get('iso', 0.150)
        batter_order = batter.get('order', 5)
        batter_handedness = batter.get('bat_side', 'R')  # L or R

        # Pitcher stats with regression to the mean for small samples
        pm = pitcher.get('blended_metrics', {})
        pitcher_sample_size = pitcher.get('sample_sizes', {}).get('season_bf', 0)

        # Regression weight: 500 batters faced
        regression_weight = 500

        # Regress pitcher HR rate toward league mean
        raw_pitcher_hr_rate = pm.get('hr_rate', 0.025)
        pitcher_hr_rate = (raw_pitcher_hr_rate * pitcher_sample_size + league_hr_rate * regression_weight) / (pitcher_sample_size + regression_weight)

        pitcher_k_pct = pm.get('k_pct', 22.0) / 100
        pitcher_gb_pct = pm.get('gb_pct', 45.0) / 100
        pitcher_barrel_pct = pm.get('barrel_pct_allowed', 6.0) / 100
        pitcher_hard_hit_pct = pm.get('hard_hit_pct_allowed', 35.0) / 100

        # Platoon-specific pitcher stats (also regressed)
        raw_pitcher_hr_rate_vs_lhb = pm.get('hr_rate_vs_lhb', raw_pitcher_hr_rate)
        raw_pitcher_hr_rate_vs_rhb = pm.get('hr_rate_vs_rhb', raw_pitcher_hr_rate)

        pitcher_hr_rate_vs_lhb = (raw_pitcher_hr_rate_vs_lhb * pitcher_sample_size + league_hr_rate * regression_weight) / (pitcher_sample_size + regression_weight)
        pitcher_hr_rate_vs_rhb = (raw_pitcher_hr_rate_vs_rhb * pitcher_sample_size + league_hr_rate * regression_weight) / (pitcher_sample_size + regression_weight)

        # Batter factors (multiplicative)
        batter_hr_factor = batter_hr_rate / league_hr_rate
        barrel_factor = 1 + (batter_barrel_pct - 0.07) * 1.5
        iso_factor = 1 + (batter_iso - 0.150) * 0.8
        batter_factor = batter_hr_factor * barrel_factor * iso_factor

        # Base pitcher factor (multiplicative)
        pitcher_hr_factor = pitcher_hr_rate / league_hr_rate
        k_factor = 1 - ((pitcher_k_pct - 0.22) * 0.5)
        pitcher_base_factor = pitcher_hr_factor * k_factor

        # Pitcher adjustments (ADDITIVE)
        pitcher_adjustments = 0.0

        # 1. Ground ball bonus/penalty
        # HR penalty if GB% > 45%, bonus if GB% < 45%
        gb_bonus = (0.45 - pitcher_gb_pct) * 0.10
        pitcher_adjustments += gb_bonus

        # 2. Barrel suppression
        # Penalty if pitcher allows more barrels than 6% league avg
        barrel_penalty = (pitcher_barrel_pct - 0.06) * 0.12
        pitcher_adjustments += barrel_penalty

        # 3. Hard-hit suppression
        # Penalty if pitcher allows more hard contact than 35% league avg
        hardhit_penalty = (pitcher_hard_hit_pct - 0.35) * 0.08
        pitcher_adjustments += hardhit_penalty

        # 4. Platoon splits
        # Compare handedness-specific HR rate to overall HR rate
        platoon_adjust = 0.0
        if batter_handedness == "L" and pitcher_hr_rate_vs_lhb:
            platoon_adjust = (pitcher_hr_rate_vs_lhb - pitcher_hr_rate) * 0.5
        elif batter_handedness == "R" and pitcher_hr_rate_vs_rhb:
            platoon_adjust = (pitcher_hr_rate_vs_rhb - pitcher_hr_rate) * 0.5
        pitcher_adjustments += platoon_adjust

        # Lineup slot factor
        lineup_pa_weights = {1:1.10, 2:1.08, 3:1.06, 4:1.04, 5:1.02, 6:1.00, 7:0.98, 8:0.96, 9:0.94}
        lineup_factor = lineup_pa_weights.get(batter_order, 1.0)

        # Final probability calculation
        # Base probability from multiplicative factors
        base_hr_prob = league_hr_rate * batter_factor * pitcher_base_factor * lineup_factor * weather_multiplier

        # Apply additive pitcher adjustments
        hr_prob = base_hr_prob * (1 + pitcher_adjustments)

        # Bound results
        hr_prob = max(0.001, min(0.20, hr_prob))

        return round(hr_prob * 100, 1)  # percentage

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