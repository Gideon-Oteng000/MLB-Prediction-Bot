#!/usr/bin/env python3
"""
Master Run Predictor
Predicts Runs Scored probabilities using integrated data from cascading pipeline

Pipeline Stage 4 of 4:
1. mlb_lineups_fetcher.py
2. run_metrics_fetcher.py
3. weather_ballpark_run_fetcher.py
4. master_run_predictor.py  ← THIS SCRIPT
"""

import json
import os
from typing import Dict, List, Tuple
from datetime import datetime
import math


class MasterRunPredictor:
    """
    Master Run predictor using integrated lineup, metrics, and weather data
    """

    def __init__(self):
        self.predictions = []

    def load_integrated_data(self, data_file: str = "final_integrated_run_data.json") -> Dict:
        """
        Load the final integrated data from cascading pipeline
        """
        if not os.path.exists(data_file):
            print(f"[ERROR] {data_file} not found")
            print("[INFO] Run the cascading pipeline first:")
            print("  1. mlb_lineups_fetcher.py")
            print("  2. run_metrics_fetcher.py")
            print("  3. weather_ballpark_run_fetcher.py")
            return {}

        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            print(f"[INFO] Loaded integrated data from {data_file}")
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return {}

    def calculate_run_probabilities(self, data: Dict) -> List[Dict]:
        """
        Calculate Run Scored probability for each batter
        """
        predictions = []
        games = data.get('games', {})

        for game_key, game_data in games.items():
            # Get weather/ballpark multiplier
            run_multiplier = self._get_run_multiplier(data, game_key)

            # Process away team lineup
            away_team = game_data.get('away_team', 'Unknown')
            home_pitcher = game_data.get('home_pitcher', {})
            away_lineup = game_data.get('away_lineup', [])

            for batter in away_lineup:
                if batter.get('blended_metrics'):
                    prob = self._calculate_batter_run_probability(
                        batter, home_pitcher, run_multiplier
                    )
                    predictions.append({
                        'player_name': batter.get('name', 'Unknown'),
                        'team': away_team,
                        'batting_order': batter.get('batting_order', 0),
                        'opposing_pitcher': home_pitcher.get('name', 'Unknown'),
                        'opposing_team': game_data.get('home_team', 'Unknown'),
                        'run_probability': prob,
                        'game_key': game_key
                    })

            # Process home team lineup
            home_team = game_data.get('home_team', 'Unknown')
            away_pitcher = game_data.get('away_pitcher', {})
            home_lineup = game_data.get('home_lineup', [])

            for batter in home_lineup:
                if batter.get('blended_metrics'):
                    prob = self._calculate_batter_run_probability(
                        batter, away_pitcher, run_multiplier
                    )
                    predictions.append({
                        'player_name': batter.get('name', 'Unknown'),
                        'team': home_team,
                        'batting_order': batter.get('batting_order', 0),
                        'opposing_pitcher': away_pitcher.get('name', 'Unknown'),
                        'opposing_team': away_team,
                        'run_probability': prob,
                        'game_key': game_key
                    })

        return predictions

    def _get_run_multiplier(self, data: Dict, game_key: str) -> float:
        """
        Get Run multiplier for the game (weather + ballpark)
        """
        game_weather = data.get('games', {}).get(game_key, {}).get('weather_ballpark', {})

        if game_weather:
            return game_weather.get('total_run_multiplier', 1.0)

        # Default multiplier if no weather data
        return 1.0

    def _calculate_batter_run_probability(self, batter: Dict, pitcher: Dict, run_multiplier: float) -> float:
        """
        Statistically grounded Run Scored probability calculation

        Run Formula:
        Run Probability = League Baseline × Batter Factors × Pitcher Factors ×
                         Lineup Context × Park/Weather × Platoon Multiplier

        Key focus for Run model:
        - On-base ability (OBP) is paramount - can't score if you don't get on base
        - Speed matters significantly (BSR, SB) - converts singles into runs
        - Lineup context - having good hitters behind you drives you in
        - Pitcher's ability to limit baserunners (WHIP, K%, LOB%)
        - Park/weather effects on run-scoring environment
        """

        # League baseline Run rate per PA
        league_run_rate = 0.15  # ~15% of PAs result in a run scored

        # Batter stats
        bm = batter.get('blended_metrics', {})
        batter_obp = bm.get('obp', 0.315)
        batter_slg = bm.get('slg', 0.411)
        batter_iso = bm.get('iso', 0.163)
        batter_contact_pct = bm.get('contact_pct', 76.0) / 100
        batter_k_pct = bm.get('k_pct', 22.0) / 100
        batter_bb_pct = bm.get('bb_pct', 8.2) / 100
        batter_ld_pct = bm.get('ld_pct', 21.0) / 100
        batter_bsr = bm.get('bsr', 0.0)
        batter_sb = bm.get('sb', 3)
        batter_cs = bm.get('cs', 1)
        batter_order = batter.get('batting_order', 5)
        batter_handedness = batter.get('bat_side', 'R')

        # Speed/advancement metrics
        sb_attempts = batter_sb + batter_cs
        sb_rate = batter_sb / sb_attempts if sb_attempts > 0 else 0.7

        # Lineup support factor (computed in run_metrics_fetcher)
        lineup_support_factor = bm.get('lineup_support_factor', 1.0)

        # Pitcher stats
        pm = pitcher.get('blended_metrics', {})
        pitcher_throws = pitcher.get('throws', 'R')

        # Pitcher's ability to prevent runs
        pitcher_whip = pm.get('whip', 1.30)
        pitcher_k_pct = pm.get('k_pct', 22.0) / 100
        pitcher_bb_pct = pm.get('bb_pct', 8.5) / 100
        pitcher_lob_pct = pm.get('lob_pct', 72.0) / 100
        pitcher_xfip = pm.get('xfip', 4.00)

        # === BATTER FACTORS (Multiplicative) ===

        # 1. On-Base Factor (CRITICAL for scoring runs - can't score if you don't get on)
        # OBP is the single most important factor for run scoring
        obp_factor = 1 + (batter_obp - 0.315) * 1.8

        # 2. Speed & Baserunning Factor
        # Good baserunners convert singles into runs by taking extra bases
        bsr_factor = 1 + (batter_bsr * 0.05)

        # 3. Stolen Base Success Rate
        # Speed on the bases helps score from 1st on doubles, etc.
        sb_factor = 1 + (sb_rate - 0.7) * 0.4

        # 4. Contact Ability Factor
        # Can't score if you strike out
        contact_factor = 1 + (batter_contact_pct - 0.76) * 1.0

        # 5. Power Factor (SLG)
        # Extra base hits = easier to score
        power_factor = 1 + (batter_slg - 0.411) * 0.8

        # 6. Line Drive Factor
        # Line drives find gaps = better scoring opportunities
        ld_factor = 1 + (batter_ld_pct - 0.21) * 0.5

        # 7. Walk Rate Factor
        # Walks get you on base (already captured in OBP, but add small bonus)
        walk_factor = 1 + (batter_bb_pct - 0.082) * 0.3

        # Combined batter factor
        batter_factor = (
            obp_factor *
            bsr_factor *
            sb_factor *
            contact_factor *
            power_factor *
            ld_factor *
            walk_factor
        )

        # === PITCHER FACTORS (Multiplicative) ===

        # 1. WHIP Factor
        # More baserunners = more chances for runs to score
        # Higher WHIP = more runs allowed
        whip_factor = 1 + (pitcher_whip - 1.30) * 0.25

        # 2. Strikeout Factor
        # High K% = fewer balls in play = fewer runs
        k_factor = 1 - (pitcher_k_pct - 0.22) * 0.6

        # 3. LOB% (Strand Rate) Factor
        # Higher LOB% = pitcher strands more runners = fewer runs score
        lob_factor = 1 - (pitcher_lob_pct - 0.72) * 0.7

        # 4. Walk Rate Factor
        # More walks = more baserunners = more runs
        bb_factor = 1 + (pitcher_bb_pct - 0.085) * 0.4

        # 5. xFIP Factor (ERA estimator - overall run prevention)
        # Lower xFIP = better at preventing runs
        xfip_factor = 1 - (pitcher_xfip - 4.00) * 0.05

        # Combined pitcher factor
        pitcher_factor = (
            whip_factor *
            k_factor *
            lob_factor *
            bb_factor *
            xfip_factor
        )

        # === LINEUP CONTEXT FACTOR ===
        # Having good hitters behind you means they'll drive you in once you're on base
        # lineup_support_factor already computed from next two batters' OBP
        lineup_factor = 1 + (lineup_support_factor - 1.0) * 0.6

        # === LINEUP POSITION FACTOR ===
        # Leadoff hitters get more PAs = more run opportunities
        # But middle-order hitters have better protection
        lineup_position_weights = {
            1: 1.10,   # Leadoff - most PAs, gets on base often
            2: 1.05,   # #2 - second most PAs
            3: 1.00,   # #3 - balanced
            4: 0.95,   # Cleanup - fewer PAs, more RBI focus
            5: 0.95,   # #5 - similar to cleanup
            6: 1.00,   # #6 - average
            7: 1.00,   # #7 - average
            8: 1.05,   # #8 - gets on for top of order
            9: 1.05    # #9 - pitcher's spot NL, turns lineup over
        }
        position_factor = lineup_position_weights.get(batter_order, 1.0)

        # === PLATOON ADJUSTMENT MULTIPLIER ===
        # Run scoring slightly increases for favorable matchups
        platoon_multiplier = 1.0

        # Same-handed matchup (harder for batter)
        if batter_handedness == pitcher_throws:
            platoon_multiplier = 0.95  # 5% penalty
        elif batter_handedness == 'S':
            platoon_multiplier = 1.02  # Switch hitter advantage
        else:
            # Opposite-handed matchup (favorable for batter)
            platoon_multiplier = 1.06  # 6% boost

        # === FINAL CALCULATION ===

        # Base run probability
        base_run_prob = (
            league_run_rate *
            batter_factor *
            pitcher_factor *
            lineup_factor *
            position_factor *
            run_multiplier
        )

        # Apply platoon adjustment
        run_prob = base_run_prob * platoon_multiplier

        # Bound results to realistic range (1% to 50%)
        run_prob = max(0.01, min(0.50, run_prob))

        return round(run_prob * 100, 1)  # Return as percentage

    def get_top_predictions(self, predictions: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        Get top N Run predictions sorted by probability
        """
        sorted_predictions = sorted(predictions, key=lambda x: x['run_probability'], reverse=True)
        return sorted_predictions[:top_n]

    def display_predictions(self, predictions: List[Dict]):
        """
        Display top predictions in clean format
        """
        print("\n" + "="*90)
        print("TOP 10 RUN SCORED PREDICTIONS")
        print("="*90)
        print(f"{'Player Name':<20} | {'Team':<4} | {'Order':<5} | {'vs Pitcher':<20} | {'Opp':<4} | {'Run Prob':<8}")
        print("-" * 90)

        for i, pred in enumerate(predictions, 1):
            print(f"{pred['player_name']:<20} | {pred['team']:<4} | #{pred['batting_order']:<4} | "
                  f"{pred['opposing_pitcher']:<20} | {pred['opposing_team']:<4} | {pred['run_probability']:>6.1f}%")

    def display_predictions_by_lineup_spot(self, predictions: List[Dict]):
        """
        Display predictions grouped by lineup position
        """
        print("\n" + "="*90)
        print("RUN SCORED PREDICTIONS BY LINEUP POSITION")
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
            position_preds = sorted(by_position[position], key=lambda x: x['run_probability'], reverse=True)

            # Show top 5 for each position
            for pred in position_preds[:5]:
                print(f"  {pred['player_name']:<20} ({pred['team']}) vs {pred['opposing_pitcher']:<20} - {pred['run_probability']:.1f}%")

    def save_predictions(self, predictions: List[Dict], filename: str = "run_predictions.json"):
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
        Main method to run Run Scored predictions
        """
        print("\n" + "="*90)
        print("MASTER RUN SCORED PREDICTOR")
        print("="*90)

        # Load integrated data
        data = self.load_integrated_data()
        if not data:
            return

        # Calculate probabilities
        print("[INFO] Calculating Run Scored probabilities...")
        all_predictions = self.calculate_run_probabilities(data)

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
        print(f"\nAverage Run Probability by Lineup Position:")
        by_position = {}
        for pred in all_predictions:
            order = pred['batting_order']
            if order not in by_position:
                by_position[order] = []
            by_position[order].append(pred['run_probability'])

        for position in sorted(by_position.keys()):
            if position == 0:
                continue
            avg_prob = sum(by_position[position]) / len(by_position[position])
            print(f"  #{position}: {avg_prob:.1f}%")

        # Highest probability
        if all_predictions:
            max_pred = max(all_predictions, key=lambda x: x['run_probability'])
            print(f"\nHighest Run Probability:")
            print(f"  {max_pred['player_name']} ({max_pred['team']}) - {max_pred['run_probability']:.1f}%")

        print(f"\nPredictions generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*90)

        # Save predictions
        self.save_predictions(all_predictions)


def main():
    """
    Main function
    """
    predictor = MasterRunPredictor()
    predictor.run_predictions()


if __name__ == "__main__":
    main()
