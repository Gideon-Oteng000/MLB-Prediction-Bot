import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib

def analyze_model_performance():
    """Find where your model performs best"""
    
    # Load your enhanced dataset and model
    df = pd.read_csv('final_enhanced_games.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    model = joblib.load('final_mlb_model.pkl')
    feature_columns = joblib.load('model_features.pkl')
    
    # Only use games with sufficient history (same as training)
    df_model = df[500:].copy()
    
    # Get predictions
    X = df_model[feature_columns]
    y_true = df_model['home_wins']
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # Home win probability
    
    df_model['predicted_home_win'] = y_pred
    df_model['home_win_probability'] = y_proba
    df_model['correct_prediction'] = (y_pred == y_true)
    
    print("=== FINDING YOUR BETTING EDGE ===\n")
    
    # 1. High Confidence Predictions
    print("1. HIGH CONFIDENCE PREDICTIONS:")
    high_conf = df_model[(df_model['home_win_probability'] > 0.65) | (df_model['home_win_probability'] < 0.35)]
    if len(high_conf) > 0:
        high_conf_accuracy = high_conf['correct_prediction'].mean()
        print(f"   Games with >65% or <35% confidence: {len(high_conf)}")
        print(f"   Accuracy: {high_conf_accuracy:.1%}")
        if high_conf_accuracy > 0.55:
            print("   🎯 EDGE FOUND! High confidence predictions are profitable")
    else:
        print("   No high confidence predictions found")
    
    # 2. Strong Team Advantages
    print(f"\n2. STRONG TEAM ADVANTAGES:")
    strong_advantage = df_model[abs(df_model['win_rate_advantage']) > 0.15]
    if len(strong_advantage) > 0:
        strong_adv_accuracy = strong_advantage['correct_prediction'].mean()
        print(f"   Games with >15% win rate difference: {len(strong_advantage)}")
        print(f"   Accuracy: {strong_adv_accuracy:.1%}")
        if strong_adv_accuracy > 0.55:
            print("   🎯 EDGE FOUND! Big talent gaps are predictable")
    
    # 3. Run Differential Extremes
    print(f"\n3. RUN DIFFERENTIAL EXTREMES:")
    run_diff_extreme = df_model[abs(df_model['run_diff_advantage']) > 1.0]
    if len(run_diff_extreme) > 0:
        run_diff_accuracy = run_diff_extreme['correct_prediction'].mean()
        print(f"   Games with >1.0 run differential advantage: {len(run_diff_extreme)}")
        print(f"   Accuracy: {run_diff_accuracy:.1%}")
        if run_diff_accuracy > 0.55:
            print("   🎯 EDGE FOUND! Offensive/defensive mismatches matter")
    
    # 4. Head-to-Head Dominance
    print(f"\n4. HEAD-TO-HEAD DOMINANCE:")
    h2h_extreme = df_model[(df_model['h2h_advantage'] > 0.7) | (df_model['h2h_advantage'] < 0.3)]
    if len(h2h_extreme) > 0:
        h2h_accuracy = h2h_extreme['correct_prediction'].mean()
        print(f"   Games with strong H2H history: {len(h2h_extreme)}")
        print(f"   Accuracy: {h2h_accuracy:.1%}")
        if h2h_accuracy > 0.55:
            print("   🎯 EDGE FOUND! Head-to-head matchups are predictive")
    
    # 5. Season Context
    print(f"\n5. SEASON TIMING:")
    early_season = df_model[df_model['is_early_season'] == 1]
    late_season = df_model[df_model['is_late_season'] == 1]
    
    if len(early_season) > 0:
        early_accuracy = early_season['correct_prediction'].mean()
        print(f"   Early season games: {len(early_season)} (Accuracy: {early_accuracy:.1%})")
    
    if len(late_season) > 0:
        late_accuracy = late_season['correct_prediction'].mean()
        print(f"   Late season games: {len(late_season)} (Accuracy: {late_accuracy:.1%})")
    
    # 6. Combined Edge Strategy
    print(f"\n6. COMBINED EDGE STRATEGY:")
    
    # Define "good bet" criteria
    good_bets = df_model[
        ((df_model['home_win_probability'] > 0.60) | (df_model['home_win_probability'] < 0.40)) &
        (abs(df_model['win_rate_advantage']) > 0.10) &
        (abs(df_model['run_diff_advantage']) > 0.5)
    ]
    
    if len(good_bets) > 0:
        good_bet_accuracy = good_bets['correct_prediction'].mean()
        pct_of_games = len(good_bets) / len(df_model) * 100
        
        print(f"   'Good Bet' criteria games: {len(good_bets)} ({pct_of_games:.1f}% of all games)")
        print(f"   Accuracy: {good_bet_accuracy:.1%}")
        
        if good_bet_accuracy > 0.54:
            print("   🚀 STRATEGY FOUND! Selective betting could be profitable")
            
            # Show ROI simulation
            print(f"\n   PROFIT SIMULATION (assuming -110 odds):")
            print(f"   - Bet $100 on {len(good_bets)} games")
            print(f"   - Wins: {sum(good_bets['correct_prediction'])} (${sum(good_bets['correct_prediction']) * 90.90:.0f})")
            print(f"   - Losses: {len(good_bets) - sum(good_bets['correct_prediction'])} (-${(len(good_bets) - sum(good_bets['correct_prediction'])) * 100:.0f})")
            profit = (sum(good_bets['correct_prediction']) * 90.90) - ((len(good_bets) - sum(good_bets['correct_prediction'])) * 100)
            print(f"   - Net Profit: ${profit:.0f}")
            print(f"   - ROI: {profit / (len(good_bets) * 100) * 100:.1f}%")
        else:
            print("   📊 Even selective betting shows the market efficiency")
    
    # 7. Monthly Performance
    print(f"\n7. MONTHLY BREAKDOWN:")
    monthly_performance = []
    for month in sorted(df_model['date'].dt.month.unique()):
        month_games = df_model[df_model['date'].dt.month == month]
        if len(month_games) > 50:  # Only months with sufficient games
            month_accuracy = month_games['correct_prediction'].mean()
            month_name = pd.to_datetime(f'2024-{month:02d}-01').strftime('%B')
            monthly_performance.append({
                'month': month_name,
                'games': len(month_games),
                'accuracy': month_accuracy
            })
            print(f"   {month_name}: {len(month_games)} games, {month_accuracy:.1%} accuracy")
    
    # Find best month
    if monthly_performance:
        best_month = max(monthly_performance, key=lambda x: x['accuracy'])
        if best_month['accuracy'] > 0.55:
            print(f"   🎯 BEST MONTH: {best_month['month']} ({best_month['accuracy']:.1%})")

    return df_model

def generate_betting_strategy(df_model):
    """Create a practical betting strategy"""
    
    print(f"\n{'='*60}")
    print(f"YOUR PERSONALIZED BETTING STRATEGY")
    print(f"{'='*60}")
    
    # Load model
    model = joblib.load('final_mlb_model.pkl')
    feature_columns = joblib.load('model_features.pkl')
    
    # Find the most profitable conditions
    X = df_model[feature_columns]
    y_proba = model.predict_proba(X)[:, 1]
    df_model['home_win_probability'] = y_proba
    
    # Strategy 1: High Confidence + Team Advantage
    strategy_games = df_model[
        ((df_model['home_win_probability'] > 0.58) | (df_model['home_win_probability'] < 0.42)) &
        (abs(df_model['win_rate_advantage']) > 0.08)
    ]
    
    if len(strategy_games) > 0:
        strategy_accuracy = (
            (strategy_games['home_win_probability'] > 0.5) == strategy_games['home_wins']
        ).mean()
        
        print(f"RECOMMENDED STRATEGY:")
        print(f"- Only bet games with 58%+ or 42%- confidence")
        print(f"- AND team win rate difference >8%") 
        print(f"- This covers {len(strategy_games)} games ({len(strategy_games)/len(df_model)*100:.1f}% of total)")
        print(f"- Historical accuracy: {strategy_accuracy:.1%}")
        
        if strategy_accuracy > 0.53:
            print(f"- 🎯 This strategy beats typical sportsbook juice!")
        else:
            print(f"- 📊 This shows market efficiency - even selective betting is tough")
    
    print(f"\nKEY TAKEAWAYS:")
    print(f"1. You built a sophisticated model with 6,882 games")
    print(f"2. 51.8% accuracy is realistic for sports prediction")
    print(f"3. Look for edges in specific situations, not all games")
    print(f"4. Professional betting requires strict discipline and bankroll management")

if __name__ == "__main__":
    print("Analyzing where your model finds edges...")
    df_model = analyze_model_performance()
    generate_betting_strategy(df_model)