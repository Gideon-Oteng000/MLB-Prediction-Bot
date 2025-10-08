#!/usr/bin/env python3
"""
Master RBI Predictor
Uses integrated data from cascading pipeline to predict RBI probabilities
"""

import json
import os
from typing import Dict, List, Tuple
from datetime import datetime
import math


class MasterRBIPredictor:
    """
    Master RBI predictor using integrated lineup, metrics, and weather data
    """

    def __init__(self):
        self.predictions = []

    def load_integrated_data(self, data_file: str = "final_integrated_rbi_data.json") -> Dict:
        """
        Load the final integrated data from cascading pipeline
        """
        if not os.path.exists(data_file):
            print(f"[ERROR] {data_file} not found")
            print("[INFO] Run the cascading pipeline first:")
            print("  1. mlb_lineups_fetcher.py")
            print("  2. rbi_metrics_fetcher.py")
            print("  3. weather_ballpark_rbi_fetcher.py")
            return {}

        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            print(f"[INFO] Loaded integrated data from {data_file}")
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return {}

    def calculate_rbi_probabilities(self, data: Dict) -> List[Dict]:
        """
        Calculate RBI probability for each batter
        """
        predictions = []
        games = data.get('games', {})

        for game_key, game_data in games.items():
            # Get weather/ballpark multiplier
            rbi_multiplier = self._get_rbi_multiplier(data, game_key)

            # Process away team lineup
            away_team = game_data.get('away_team', 'Unknown')
            home_pitcher = game_data.get('home_pitcher', {})
            away_lineup = game_data.get('away_lineup', [])

            for batter in away_lineup:
                if batter.get('blended_metrics'):
                    prob = self._calculate_batter_rbi_probability(
                        batter, home_pitcher, rbi_multiplier
                    )
                    predictions.append({
                        'player_name': batter.get('name', 'Unknown'),
                        'team': away_team,
                        'batting_order': batter.get('batting_order', 0),
                        'opposing_pitcher': home_pitcher.get('name', 'Unknown'),
                        'opposing_team': game_data.get('home_team', 'Unknown'),
                        'rbi_probability': prob,
                        'game_key': game_key
                    })

            # Process home team lineup
            home_team = game_data.get('home_team', 'Unknown')
            away_pitcher = game_data.get('away_pitcher', {})
            home_lineup = game_data.get('home_lineup', [])

            for batter in home_lineup:
                if batter.get('blended_metrics'):
                    prob = self._calculate_batter_rbi_probability(
                        batter, away_pitcher, rbi_multiplier
                    )
                    predictions.append({
                        'player_name': batter.get('name', 'Unknown'),
                        'team': home_team,
                        'batting_order': batter.get('batting_order', 0),
                        'opposing_pitcher': away_pitcher.get('name', 'Unknown'),
                        'opposing_team': away_team,
                        'rbi_probability': prob,
                        'game_key': game_key
                    })

        return predictions

    def _get_rbi_multiplier(self, data: Dict, game_key: str) -> float:
        """
        Get RBI multiplier for the game (weather + ballpark)
        """
        game_weather = data.get('games', {}).get(game_key, {}).get('weather_ballpark', {})

        if game_weather:
            return game_weather.get('total_rbi_multiplier', 1.0)

        # Default multiplier if no weather data
        return 1.0

    def _calculate_batter_rbi_probability(self, batter: Dict, pitcher: Dict, rbi_multiplier: float) -> float:
        """
        Statistically grounded RBI probability calculation

        RBI Formula:
        RBI Probability = League Baseline × Batter Factors × Pitcher Factors × Situational Factors × Park/Weather

        Key differences from HR model:
        - Contact% and batting average matter more than raw power
        - Lineup position is critical (middle order = more RBI opportunities)
        - Pitcher's ability to strand runners (LOB%, K%) matters
        - Park's run environment and batting average factor are key

        Enhanced pitcher context:
        - LOB% (Left On Base %) directly influences how often runners score
        - Platoon suppression (L vs R advantage) for K% matchups
        - Barrel% allowed as proxy for damage contact prevention
        - xFIP (ERA estimator) as proxy for overall run suppression
        - Platoon adjustment multiplier (pitcher_throws vs batter_side) applied to final calculation

        Enhanced batter context:
        - RISP performance (runners in scoring position AVG) - clutch hitting factor
        - Measures actual performance differential with RISP vs overall batting

        Future enhancements:
        - OBP of preceding hitters in lineup (affects RBI opportunity volume)
          → Would weight lineup position factor based on actual OBP of batters hitting ahead
        """

        # League baseline RBI rate per PA
        league_rbi_rate = 0.12  # ~12% of PAs result in RBI

        # Batter stats
        bm = batter.get('blended_metrics', {})
        batter_avg = bm.get('avg', 0.248)
        batter_obp = bm.get('obp', 0.315)
        batter_slg = bm.get('slg', 0.411)
        batter_iso = bm.get('iso', 0.163)
        batter_contact_pct = bm.get('contact_pct', 76.0) / 100
        batter_k_pct = bm.get('k_pct', 22.0) / 100
        batter_hard_hit_pct = bm.get('hard_hit_pct', 37.5) / 100
        batter_ld_pct = bm.get('ld_pct', 21.0) / 100
        batter_order = batter.get('batting_order', 5)
        batter_handedness = batter.get('bat_side', 'R')

        # RISP (Runners In Scoring Position) performance - critical for RBI prediction
        batter_risp_avg = bm.get('risp_avg', batter_avg)
        batter_risp_slg = bm.get('risp_slg', batter_slg)

        # Pitcher stats with regression to mean
        pm = pitcher.get('blended_metrics', {})
        pitcher_sample_size = pitcher.get('sample_sizes', {}).get('season_bf', 0)
        pitcher_throws = pitcher.get('throws', 'R')  # L or R

        # Regression weight for pitchers
        regression_weight = 500

        # Pitcher's ability to prevent RBIs
        pitcher_k_pct = pm.get('k_pct', 22.0) / 100
        pitcher_contact_allowed = pm.get('contact_allowed_pct', 76.0) / 100
        pitcher_hard_hit_allowed = pm.get('hard_hit_pct_allowed', 37.5) / 100
        pitcher_barrel_allowed = pm.get('barrel_pct_allowed', 7.5) / 100
        pitcher_gb_pct = pm.get('gb_pct', 45.0) / 100
        pitcher_whip_proxy = pm.get('whip_proxy', 1.30)
        pitcher_lob_pct = pm.get('lob_pct', 72.0) / 100
        pitcher_xfip = pm.get('xfip', 4.00)

        # Platoon split metrics
        pitcher_k_vs_lhh = pm.get('k_pct_vs_lhh', 22.0) / 100
        pitcher_k_vs_rhh = pm.get('k_pct_vs_rhh', 22.0) / 100

        # === BATTER FACTORS (Multiplicative) ===

        # 1. Contact ability (critical for RBI - can't drive in runs if you strike out)
        contact_factor = 1 + (batter_contact_pct - 0.76) * 1.2

        # 2. Batting average factor (hits drive in runs)
        avg_factor = 1 + (batter_avg - 0.248) * 2.0

        # 3. Power factor (ISO for extra bases and driving in multiple runs)
        power_factor = 1 + (batter_iso - 0.163) * 1.5

        # 4. Hard contact factor (hard-hit balls = more RBIs)
        hard_contact_factor = 1 + (batter_hard_hit_pct - 0.375) * 0.8

        # 5. Line drive factor (line drives find gaps = RBI hits)
        ld_factor = 1 + (batter_ld_pct - 0.21) * 0.6

        # 6. RISP factor (clutch hitting - batters differ significantly with RISP)
        # Measures how much better/worse a batter performs with runners in scoring position
        risp_avg_diff = batter_risp_avg - batter_avg
        risp_factor = 1 + (risp_avg_diff * 1.2)

        # Bound RISP factor to reasonable range (0.85 to 1.15)
        risp_factor = max(0.85, min(1.15, risp_factor))

        # Combined batter factor
        batter_factor = (contact_factor * avg_factor * power_factor *
                        hard_contact_factor * ld_factor * risp_factor)

        # === PITCHER FACTORS (Multiplicative) ===

        # 1. Strikeout factor (high K% = fewer RBI opportunities)
        # Use platoon-adjusted K% if available
        if batter_handedness == 'L' and pitcher_k_vs_lhh > 0:
            effective_k_pct = pitcher_k_vs_lhh
        elif batter_handedness == 'R' and pitcher_k_vs_rhh > 0:
            effective_k_pct = pitcher_k_vs_rhh
        else:
            effective_k_pct = pitcher_k_pct

        k_factor = 1 - ((effective_k_pct - 0.22) * 0.8)

        # 2. Contact suppression (low contact allowed = fewer RBIs)
        contact_suppression_factor = 1 + ((pitcher_contact_allowed - 0.76) * 0.4)

        # 3. Hard contact prevention
        hard_hit_prevention = 1 - ((pitcher_hard_hit_allowed - 0.375) * 0.3)

        # 4. Barrel prevention (critical for damage limitation)
        barrel_prevention = 1 - ((pitcher_barrel_allowed - 0.075) * 0.4)

        # 5. Ground ball factor (high GB% = double plays, fewer runs)
        gb_factor = 1 - ((pitcher_gb_pct - 0.45) * 0.5)

        # 6. WHIP proxy (runners on base = RBI opportunities)
        # Higher WHIP = more baserunners = more RBI chances for next batter
        whip_factor = 1 + ((pitcher_whip_proxy - 1.30) * 0.15)

        # 7. LOB% (strand rate - critical for RBI prevention)
        # Higher LOB% = pitcher strands runners = fewer RBIs
        lob_factor = 1 - ((pitcher_lob_pct - 0.72) * 0.6)

        # 8. xFIP factor (ERA estimator - overall run suppression proxy)
        # Lower xFIP = better run prevention
        xfip_factor = 1 - ((pitcher_xfip - 4.00) * 0.05)

        # Combined pitcher factor
        pitcher_factor = (k_factor * contact_suppression_factor * hard_hit_prevention *
                         barrel_prevention * gb_factor * whip_factor * lob_factor * xfip_factor)

        # === LINEUP POSITION FACTOR (Critical for RBI) ===
        # Middle of order (3-5) get most RBI opportunities
        # Top of order (1-2) have fewer runners on base
        # Bottom of order (7-9) have weaker hitters ahead

        lineup_rbi_weights = {
            1: 0.75,   # Leadoff - few runners on base
            2: 0.85,   # #2 hitter - some RBI chances
            3: 1.15,   # #3 hitter - prime RBI position
            4: 1.20,   # Cleanup - most RBI opportunities
            5: 1.15,   # #5 hitter - strong RBI position
            6: 1.00,   # #6 hitter - average
            7: 0.90,   # #7 hitter - fewer opportunities
            8: 0.80,   # #8 hitter - limited chances
            9: 0.70    # #9 hitter - fewest opportunities
        }
        lineup_factor = lineup_rbi_weights.get(batter_order, 1.0)

        # === PLATOON ADJUSTMENT MULTIPLIER ===
        # Batters typically perform better vs opposite-hand pitchers
        # Same-handed matchups (L vs L, R vs R) favor the pitcher
        # Opposite-handed matchups (L vs R, R vs L) favor the batter

        platoon_multiplier = 1.0

        # Same-handed matchup (harder for batter)
        if batter_handedness == pitcher_throws:
            # Same side = pitcher advantage
            # LHB vs LHP or RHB vs RHP
            platoon_multiplier = 0.92  # 8% penalty for batter
        else:
            # Opposite side = batter advantage
            # LHB vs RHP or RHB vs LHP
            platoon_multiplier = 1.08  # 8% boost for batter

        # Switch hitters get neutral treatment
        if batter_handedness == 'S':
            platoon_multiplier = 1.02  # Slight advantage (can choose better side)

        # === FINAL CALCULATION ===

        # Base RBI probability
        base_rbi_prob = league_rbi_rate * batter_factor * pitcher_factor * lineup_factor * rbi_multiplier

        # Apply platoon adjustment multiplier
        rbi_prob = base_rbi_prob * platoon_multiplier

        # Bound results to realistic range
        rbi_prob = max(0.01, min(0.45, rbi_prob))

        return round(rbi_prob * 100, 1)  # Return as percentage

    def get_top_predictions(self, predictions: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        Get top N RBI predictions sorted by probability
        """
        sorted_predictions = sorted(predictions, key=lambda x: x['rbi_probability'], reverse=True)
        return sorted_predictions[:top_n]

    def display_predictions(self, predictions: List[Dict]):
        """
        Display top predictions in clean format
        """
        print("\n" + "="*90)
        print("TOP 10 RBI PREDICTIONS")
        print("="*90)
        print(f"{'Player Name':<20} | {'Team':<4} | {'Order':<5} | {'vs Pitcher':<20} | {'Opp':<4} | {'RBI Prob':<8}")
        print("-" * 90)

        for i, pred in enumerate(predictions, 1):
            print(f"{pred['player_name']:<20} | {pred['team']:<4} | #{pred['batting_order']:<4} | "
                  f"{pred['opposing_pitcher']:<20} | {pred['opposing_team']:<4} | {pred['rbi_probability']:>6.1f}%")

    def display_predictions_by_lineup_spot(self, predictions: List[Dict]):
        """
        Display predictions grouped by lineup position
        """
        print("\n" + "="*90)
        print("RBI PREDICTIONS BY LINEUP POSITION")
        print("="*90)

        # Group by lineup position
        by_position = {}
        for pred in predictions:
            order = pred['batting_order']
            if order not in by_position:
                by_position[order] = []
            by_position[order].append(pred)

        # Display each position
        for position in sorted(by_position.keys()):
            if position == 0:
                continue  # Skip if no batting order

            print(f"\n{position}-Hole Hitters (Lineup Position #{position}):")
            print("-" * 90)

            # Sort by probability within position
            position_preds = sorted(by_position[position], key=lambda x: x['rbi_probability'], reverse=True)

            # Show top 5 for each position
            for pred in position_preds[:5]:
                print(f"  {pred['player_name']:<20} ({pred['team']}) vs {pred['opposing_pitcher']:<20} - {pred['rbi_probability']:.1f}%")

    def save_predictions(self, predictions: List[Dict], filename: str = "rbi_predictions.json"):
        """
        Save predictions to JSON file
        """
        try:
            output = {
                'date': datetime.now().isoformat(),
                'total_predictions': len(predictions),
                'predictions': predictions
            }

            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)

            print(f"\n[SUCCESS] Predictions saved to {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save predictions: {e}")

    def run_predictions(self):
        """
        Main method to run RBI predictions
        """
        print("\n" + "="*90)
        print("MASTER RBI PREDICTOR")
        print("="*90)

        # Load integrated data
        data = self.load_integrated_data()
        if not data:
            return

        # Calculate probabilities
        print("[INFO] Calculating RBI probabilities...")
        all_predictions = self.calculate_rbi_probabilities(data)

        if not all_predictions:
            print("[ERROR] No predictions generated. Check data quality.")
            return

        # Get top 10
        top_predictions = self.get_top_predictions(all_predictions, 10)

        # Display results
        self.display_predictions(top_predictions)

        # Display by lineup position
        self.display_predictions_by_lineup_spot(all_predictions)

        # Summary statistics
        print("\n" + "="*90)
        print("SUMMARY STATISTICS")
        print("="*90)
        print(f"Total players analyzed: {len(all_predictions)}")

        # Stats by lineup position
        print(f"\nAverage RBI Probability by Lineup Position:")
        by_position = {}
        for pred in all_predictions:
            order = pred['batting_order']
            if order not in by_position:
                by_position[order] = []
            by_position[order].append(pred['rbi_probability'])

        for position in sorted(by_position.keys()):
            if position == 0:
                continue
            avg_prob = sum(by_position[position]) / len(by_position[position])
            print(f"  #{position}: {avg_prob:.1f}%")

        print(f"\nPredictions generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*90)

        # Save predictions
        self.save_predictions(all_predictions)


def main():
    """
    Main function
    """
    predictor = MasterRBIPredictor()
    predictor.run_predictions()


if __name__ == "__main__":
    main()
