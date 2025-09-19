"""
Fast version of MLB HR predictor that uses default stats to avoid slow API calls
"""
import pandas as pd
from mlb_hr_clean_v4 import SportsRadarV8, HRProbabilityModel, Config
from datetime import datetime

class FastStatsCollector:
    """Fast stats collector that uses reasonable defaults"""

    def get_batter_stats(self, player_name):
        """Get default batter stats"""
        return {
            'home_runs': 15,
            'at_bats': 400,
            'ops': 0.740,
            'iso': 0.165,
            'barrel_rate': 0.075
        }

    def get_pitcher_stats(self, pitcher_name):
        """Get default pitcher stats"""
        return {
            'era': 4.50,
            'hr_per_9': 1.3,
            'whip': 1.35
        }

class FastHRProbabilityModel:
    """Fast HR probability model using defaults"""

    def __init__(self):
        self.config = Config()
        self.stats = FastStatsCollector()

    def calculate_probability(self, batter_name, pitcher_name, venue):
        """Calculate HR probability for a batter"""

        # Get stats
        batter_stats = self.stats.get_batter_stats(batter_name)
        pitcher_stats = self.stats.get_pitcher_stats(pitcher_name)

        # Base rate (MLB average ~3.3%)
        base_rate = 0.033

        # Batter power factor
        hr_rate = batter_stats['home_runs'] / max(batter_stats['at_bats'], 100)
        batter_mult = hr_rate / 0.033
        batter_mult = min(max(batter_mult, 0.3), 3.0)

        # ISO bonus
        iso = batter_stats['iso']
        if iso > 0.250:
            batter_mult *= 1.35
        elif iso > 0.200:
            batter_mult *= 1.15
        elif iso < 0.120:
            batter_mult *= 0.7

        # Pitcher vulnerability
        pitcher_mult = 1.0
        if pitcher_stats:
            hr_per_9 = pitcher_stats['hr_per_9']
            if hr_per_9 > 1.5:
                pitcher_mult = 1.3
            elif hr_per_9 > 1.2:
                pitcher_mult = 1.1
            elif hr_per_9 < 0.9:
                pitcher_mult = 0.7

            # ERA factor
            era = pitcher_stats['era']
            if era > 5.0:
                pitcher_mult *= 1.15
            elif era < 3.0:
                pitcher_mult *= 0.85

        # Park factor
        park_mult = self.config.PARK_FACTORS.get(venue, 1.0)

        # Weather multiplier (simplified since weather API is optional)
        weather_mult = 1.0

        # Elevation bonus for Coors Field
        if 'Coors' in venue:
            weather_mult = 1.15  # Additional altitude boost

        # Calculate final probability
        total_mult = batter_mult * pitcher_mult * park_mult * weather_mult
        hr_prob = base_rate * total_mult

        # Cap at 15%
        return min(hr_prob, 0.15)

class FastHRPredictor:
    """Fast HR prediction system"""

    def __init__(self):
        self.sportradar = SportsRadarV8()
        self.model = FastHRProbabilityModel()

    def run_predictions(self):
        """Generate fast HR predictions"""
        print("=" * 80)
        print("MLB HOME RUN PREDICTIONS v5.0 (FAST MODE)")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 80)

        # Get games from SportsRadar
        games = self.sportradar.get_todays_games_with_lineups()

        if not games:
            print("\n[ERROR] No games available")
            return pd.DataFrame()

        print(f"\n[GAME] Processing {len(games)} games")
        print("[INFO] Using default player stats for fast processing")

        all_predictions = []

        for game in games:
            # Process home batters vs away pitcher
            if game['home_lineup'] and game['away_pitcher']:
                pitcher_name = game['away_pitcher']['name']

                for batter in game['home_lineup'][:9]:
                    prob = self.model.calculate_probability(
                        batter['name'],
                        pitcher_name,
                        game['venue']
                    )

                    all_predictions.append({
                        'Hitter': batter['name'],
                        'Pitcher': pitcher_name,
                        'Teams': f"{game['home_team_abbr']} vs {game['away_team_abbr']}",
                        'HR_Probability': round(prob * 100, 2)
                    })

            # Process away batters vs home pitcher
            if game['away_lineup'] and game['home_pitcher']:
                pitcher_name = game['home_pitcher']['name']

                for batter in game['away_lineup'][:9]:
                    prob = self.model.calculate_probability(
                        batter['name'],
                        pitcher_name,
                        game['venue']
                    )

                    all_predictions.append({
                        'Hitter': batter['name'],
                        'Pitcher': pitcher_name,
                        'Teams': f"{game['away_team_abbr']} @ {game['home_team_abbr']}",
                        'HR_Probability': round(prob * 100, 2)
                    })

        if not all_predictions:
            print("\n[ERROR] No valid predictions generated")
            return pd.DataFrame()

        # Create DataFrame and sort
        df = pd.DataFrame(all_predictions)
        df = df.sort_values('HR_Probability', ascending=False).reset_index(drop=True)

        # Keep only top 15
        df = df.head(15)

        print("\n" + "=" * 80)
        print("TOP 15 HOME RUN PREDICTIONS (DEFAULT STATS)")
        print("=" * 80)
        print(f"{'Rank':<5} {'Hitter':<20} {'Pitcher':<20} {'Teams':<12} {'Prob %':<8}")
        print("-" * 80)

        for idx, row in df.iterrows():
            print(f"{idx+1:<5} {row['Hitter'][:19]:<20} {row['Pitcher'][:19]:<20} {row['Teams']:<12} {row['HR_Probability']:<8.1f}%")

        # Save to CSV
        filename = f"hr_predictions_fast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)

        print("\n" + "=" * 80)
        print(f"[SUCCESS] Predictions saved to: {filename}")
        print(f"[INFO] Total predictions: {len(all_predictions)}")
        print(f"[WARNING] Note: Using default player stats for demonstration")

        return df

if __name__ == "__main__":
    try:
        predictor = FastHRPredictor()
        predictions = predictor.run_predictions()

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()