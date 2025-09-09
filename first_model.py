import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def load_and_prepare_data():
    """Load the games data and prepare it for modeling"""
    df = pd.read_csv('mlb_games.csv')
    
    print(f"Loaded {len(df)} games")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Create features (starting very simple)
    # Feature 1: Home field (1 for home team, 0 for away)
    df['home_field'] = 1
    
    # Create target variable (1 if home team wins, 0 if away team wins)
    df['home_wins'] = (df['winner'] == 'home').astype(int)
    
    return df

def create_simple_model(df):
    """Create a simple logistic regression model"""
    
    # Features (just home field advantage for now)
    X = df[['home_field']]
    y = df['home_wins']
    
    # Split data: use 70% for training, 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train)} games")
    print(f"Testing on {len(X_test)} games")
    
    # Create and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\n=== MODEL RESULTS ===")
    print(f"Training Accuracy: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")
    print(f"Testing Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    
    # Show the model's "logic"
    home_advantage = model.coef_[0][0]
    print(f"\nHome field advantage coefficient: {home_advantage:.3f}")
    
    # Predict probability of home team winning
    home_win_prob = model.predict_proba([[1]])[0][1]
    print(f"Model predicts home teams win {home_win_prob*100:.1f}% of the time")
    
    return model, test_accuracy

def baseline_comparison(df):
    """Compare against simple baselines"""
    actual_home_wins = df['home_wins'].sum()
    total_games = len(df)
    actual_home_rate = actual_home_wins / total_games
    
    print(f"\n=== BASELINE COMPARISON ===")
    print(f"Actual home win rate: {actual_home_rate:.3f} ({actual_home_rate*100:.1f}%)")
    print(f"Random guessing would be: 0.500 (50.0%)")
    
    # If we always predicted the majority class
    majority_baseline = max(actual_home_rate, 1-actual_home_rate)
    print(f"Always predicting majority class: {majority_baseline:.3f} ({majority_baseline*100:.1f}%)")

if __name__ == "__main__":
    print("Building your first MLB prediction model...\n")
    
    # Load data
    df = load_and_prepare_data()
    
    # Show baseline stats
    baseline_comparison(df)
    
    # Build model
    model, accuracy = create_simple_model(df)
    
    print(f"\n=== SUMMARY ===")
    if accuracy > 0.52:
        print("🎉 Great! Your model beats random guessing!")
    else:
        print("📈 Model needs improvement, but this is normal for a first try.")
        
    print("\nNext steps:")
    print("1. Add more features (team stats, recent form)")
    print("2. Collect more historical data") 
    print("3. Try different models")