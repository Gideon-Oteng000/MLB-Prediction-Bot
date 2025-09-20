#!/usr/bin/env python3
"""
Test script for the enhanced RBI prediction system with real MLB data
Tests the complete pipeline from data collection to prediction
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RBI import AdvancedRBIPredictorV3, MLBDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mlb_data_fetcher():
    """Test the MLB data fetcher with cache"""
    print("=" * 60)
    print("TESTING MLB DATA FETCHER")
    print("=" * 60)

    fetcher = MLBDataFetcher()

    # Test date range (last 7 days for quick test)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    print(f"Testing data fetch for {start_date} to {end_date}")

    try:
        games_df = fetcher.fetch_historical_games(start_date, end_date)

        if not games_df.empty:
            print(f"✓ Successfully fetched {len(games_df)} games")
            print(f"✓ Sample game data:")

            for idx, (_, game) in enumerate(games_df.head(2).iterrows()):
                print(f"  Game {idx + 1}: {game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')}")
                print(f"    Date: {game.get('date', 'Unknown')}")
                print(f"    Score: {game.get('away_score', 0)}-{game.get('home_score', 0)}")
                print(f"    Player performances: {len(game.get('player_performances', []))}")

                # Show sample player performance
                if game.get('player_performances'):
                    sample_perf = game['player_performances'][0]
                    print(f"    Sample player: {sample_perf.get('player_name', 'Unknown')} - {sample_perf.get('rbi', 0)} RBIs")
        else:
            print("⚠ No games fetched (may be off-season or API issues)")

    except Exception as e:
        print(f"✗ Error testing data fetcher: {e}")
        return False

    return True

def test_training_data_preparation():
    """Test training data preparation"""
    print("\n" + "=" * 60)
    print("TESTING TRAINING DATA PREPARATION")
    print("=" * 60)

    try:
        predictor = AdvancedRBIPredictorV3()

        # Test with recent date range (smaller for testing)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        print(f"Preparing training data for {start_date} to {end_date}")

        X, y = predictor.prepare_training_data(start_date, end_date)

        if len(X) > 0:
            print(f"✓ Successfully prepared {len(X)} training samples")
            print(f"✓ Feature vector size: {X.shape[1] if len(X.shape) > 1 else 'N/A'}")
            print(f"✓ RBI distribution: 0 RBIs: {sum(y == 0)}, 1+ RBIs: {sum(y > 0)}")
            print(f"✓ Average RBI rate: {sum(y > 0) / len(y):.1%}")

            return True
        else:
            print("⚠ No training data prepared")
            return False

    except Exception as e:
        print(f"✗ Error in training data preparation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_baseline_validation():
    """Test formula baseline validation"""
    print("\n" + "=" * 60)
    print("TESTING BASELINE VALIDATION")
    print("=" * 60)

    try:
        predictor = AdvancedRBIPredictorV3()

        # Run baseline validation
        baseline_metrics = predictor.validate_formula_baseline()

        print(f"✓ Baseline validation completed")
        print(f"  League RBI Rate: {baseline_metrics.get('league_avg_rbi_rate', 0):.1%}")
        print(f"  Total Samples: {baseline_metrics.get('total_samples', 0):,}")
        print(f"  Batting Order Variance: {baseline_metrics.get('batting_order_variance', 0):.4f}")

        # Check if results are reasonable
        rbi_rate = baseline_metrics.get('league_avg_rbi_rate', 0)
        if 0.08 <= rbi_rate <= 0.15:
            print(f"✓ RBI rate within expected range (8%-15%)")
        else:
            print(f"⚠ RBI rate outside expected range: {rbi_rate:.1%}")

        return True

    except Exception as e:
        print(f"✗ Error in baseline validation: {e}")
        return False

def test_model_training():
    """Test model training with real data"""
    print("\n" + "=" * 60)
    print("TESTING MODEL TRAINING")
    print("=" * 60)

    try:
        predictor = AdvancedRBIPredictorV3()

        print("Models loaded/trained:")
        for model_name in predictor.models:
            print(f"  ✓ {model_name}")

        print(f"✓ SHAP explainers: {'Available' if predictor.shap_explainers else 'Not available'}")

        return True

    except Exception as e:
        print(f"✗ Error in model training: {e}")
        return False

def test_prediction_pipeline():
    """Test the complete prediction pipeline"""
    print("\n" + "=" * 60)
    print("TESTING PREDICTION PIPELINE")
    print("=" * 60)

    try:
        predictor = AdvancedRBIPredictorV3()

        # Create test features for a hypothetical player
        print("Testing prediction for sample player...")

        # Get today's date
        today = datetime.now()

        # Create enhanced features for a test player
        features = predictor.create_enhanced_features(
            player_name="Mike Trout",
            team="Los Angeles Angels",
            batting_order=3,
            opponent="Houston Astros",
            pitcher_id=543135,  # Sample pitcher ID
            game_datetime=today,
            venue_lat=33.8003,  # Angel Stadium
            venue_lon=-117.8827
        )

        # Get prediction
        result = predictor.predict_with_explanation(features)

        print(f"✓ Prediction generated for {result.get('player_name', 'Test Player')}")
        print(f"  Expected RBIs: {result.get('expected_rbis', 0):.2f}")
        print(f"  RBI Probability: {result.get('rbi_probability', 0):.1%}")
        print(f"  Confidence: {result.get('confidence_score', 0):.1%}")
        print(f"  Recommendation: {result.get('recommendation', 'N/A')}")

        if result.get('top_positive_features'):
            print(f"  Top positive factors:")
            for feature, impact in result['top_positive_features'][:3]:
                print(f"    {feature}: +{impact:.3f}")

        return True

    except Exception as e:
        print(f"✗ Error in prediction pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_operations():
    """Test database operations"""
    print("\n" + "=" * 60)
    print("TESTING DATABASE OPERATIONS")
    print("=" * 60)

    try:
        predictor = AdvancedRBIPredictorV3()

        import sqlite3
        conn = sqlite3.connect(predictor.db_path)
        cursor = conn.cursor()

        # Check table existence
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            'historical_games',
            'rbi_training_data_v3',
            'player_performances',
            'predictions',
            'bullpen_stats'
        ]

        for table in expected_tables:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  ✓ {table}: {count:,} records")
            else:
                print(f"  ⚠ {table}: Not found")

        conn.close()
        return True

    except Exception as e:
        print(f"✗ Error in database operations: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TESTING ENHANCED RBI PREDICTION SYSTEM")
    print("Real MLB Historical Data Integration")
    print("=" * 60)

    tests = [
        ("MLB Data Fetcher", test_mlb_data_fetcher),
        ("Training Data Preparation", test_training_data_preparation),
        ("Baseline Validation", test_baseline_validation),
        ("Model Training", test_model_training),
        ("Prediction Pipeline", test_prediction_pipeline),
        ("Database Operations", test_database_operations),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System is ready for real MLB data.")
    elif passed >= total * 0.8:
        print("⚠ Most tests passed. System is mostly functional.")
    else:
        print("❌ Multiple tests failed. System needs attention.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)