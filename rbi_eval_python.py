"""
RBI Prediction Model Evaluation Framework
==========================================
Comprehensive evaluation suite for RBI prediction models
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RBIPredictionEvaluator:
    """Complete evaluation framework for RBI predictions"""
    
    def __init__(self, db_path):
        """Initialize evaluator with database connection"""
        self.conn = sqlite3.connect(db_path)
        self.setup_views()
        
    def setup_views(self):
        """Create all evaluation views (run the SQL from previous artifact)"""
        # Note: Execute the CREATE VIEW statements from the SQL artifact
        pass
    
    def get_latest_run_id(self):
        """Get the most recent run_id"""
        query = "SELECT MAX(run_id) FROM rbi_predictions_log_v3"
        return pd.read_sql(query, self.conn).iloc[0, 0]
    
    def generate_performance_report(self, run_id=None):
        """Generate comprehensive performance report"""
        if run_id is None:
            run_id = self.get_latest_run_id()
        
        print(f"\n{'='*60}")
        print(f"RBI PREDICTION MODEL EVALUATION REPORT")
        print(f"Run ID: {run_id}")
        print(f"{'='*60}\n")
        
        # 1. Overall Performance
        self._print_overall_performance(run_id)
        
        # 2. Calibration Analysis
        self._print_calibration_analysis(run_id)
        
        # 3. Betting Performance
        self._print_betting_performance(run_id)
        
        # 4. Subgroup Analysis
        self._print_subgroup_analysis(run_id)
        
        # 5. Feature Importance
        self._print_feature_importance(run_id)
        
        return self._generate_visualizations(run_id)
    
    def _print_overall_performance(self, run_id):
        """Print overall model performance metrics"""
        query = f"""
        SELECT * FROM model_performance_summary 
        WHERE run_id = '{run_id}'
        """
        df = pd.read_sql(query, self.conn)
        
        if df.empty:
            print("No data found for this run_id")
            return
        
        row = df.iloc[0]
        print("1. OVERALL PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Total Predictions: {row['total_predictions']:,}")
        print(f"Date Range: {row['start_date']} to {row['end_date']}")
        print(f"Actual Hit Rate: {row['actual_hit_rate']:.1%}")
        print(f"Avg Predicted Prob: {row['avg_predicted_prob']:.1%}")
        print(f"Calibration Error: {row['calibration_diff']:+.1%}")
        print(f"Brier Score: {row['brier_score']:.4f}")
        print(f"Log Loss: {row['log_loss']:.4f}")
        print(f"Total RBIs: {row['total_actual_rbi']:.0f}")
        print(f"Expected RBIs: {row['total_expected_rbi']:.0f}")
        print(f"RBI Ratio (Actual/Expected): {row['rbi_ratio']:.2f}")
        print()
    
    def _print_calibration_analysis(self, run_id):
        """Print calibration analysis"""
        query = f"""
        SELECT * FROM calibration_buckets 
        WHERE run_id = '{run_id}'
        ORDER BY prob_bucket
        """
        df = pd.read_sql(query, self.conn)
        
        print("2. CALIBRATION ANALYSIS")
        print("-" * 40)
        print(f"{'Prob Range':<12} {'N':<6} {'Pred%':<8} {'Act%':<8} {'Error':<8}")
        print("-" * 40)
        
        for _, row in df.iterrows():
            print(f"{row['prob_bucket']:<12} {row['count']:<6} "
                  f"{row['avg_predicted_prob']:<8.1%} "
                  f"{row['actual_hit_rate']:<8.1%} "
                  f"{row['calibration_error']:+7.1%}")
        
        # Calculate ECE (Expected Calibration Error)
        ece = np.average(np.abs(df['calibration_error']), 
                        weights=df['count'])
        print(f"\nExpected Calibration Error (ECE): {ece:.3f}")
        print()
    
    def _print_betting_performance(self, run_id):
        """Print betting performance metrics"""
        query = f"""
        SELECT * FROM betting_performance 
        WHERE run_id = '{run_id}'
        """
        df = pd.read_sql(query, self.conn)
        
        if not df.empty and df.iloc[0]['total_bets'] > 0:
            row = df.iloc[0]
            print("3. BETTING PERFORMANCE")
            print("-" * 40)
            print(f"Total Bets Placed: {row['total_bets']}")
            print(f"Winning Bets: {row['winning_bets']}")
            print(f"Win Rate: {row['winning_bets']/row['total_bets']:.1%}")
            print(f"Average Edge: {row['avg_edge']:.1%}")
            print(f"Total Profit: {row['total_profit_units']:+.2f} units")
            print(f"ROI per Bet: {row['roi_percentage']:.1%}")
            
            # Kelly Criterion suggestion
            win_rate = row['winning_bets'] / row['total_bets']
            avg_odds = 1 / 0.25  # Approximate
            kelly = (win_rate * avg_odds - 1) / (avg_odds - 1)
            print(f"Kelly Fraction: {max(0, kelly):.1%}")
            print()
    
    def _print_subgroup_analysis(self, run_id):
        """Print performance by subgroups"""
        query = f"""
        SELECT * FROM subgroup_performance 
        WHERE run_id = '{run_id}' AND category != 'Overall'
        ORDER BY category, actual_hit_rate DESC
        """
        df = pd.read_sql(query, self.conn)
        
        print("4. SUBGROUP PERFORMANCE")
        print("-" * 40)
        
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            print(f"\n{category}:")
            print(f"{'Subgroup':<15} {'N':<6} {'Act%':<8} {'Brier':<8}")
            print("-" * 37)
            
            for _, row in cat_df.iterrows():
                print(f"{row['subcategory']:<15} {row['count']:<6} "
                      f"{row['actual_hit_rate']:<8.1%} "
                      f"{row['brier_score']:<8.4f}")
        print()
    
    def _print_feature_importance(self, run_id):
        """Print top important features"""
        query = f"""
        SELECT * FROM feature_importance_summary 
        WHERE run_id = '{run_id}'
        ORDER BY frequency DESC
        LIMIT 10
        """
        df = pd.read_sql(query, self.conn)
        
        print("\n5. TOP FEATURE IMPORTANCE")
        print("-" * 40)
        print(f"{'Feature':<25} {'Freq':<8} {'Avg SHAP':<10} {'Hit%':<8}")
        print("-" * 40)
        
        for _, row in df.iterrows():
            print(f"{row['top_positive_feature']:<25} "
                  f"{row['frequency']:<8} "
                  f"{row['avg_shap_value']:<10.3f} "
                  f"{row['hit_rate_when_important']:<8.1%}")
        print()
    
    def _generate_visualizations(self, run_id):
        """Generate evaluation visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'RBI Model Evaluation - Run: {run_id}', fontsize=16)
        
        # 1. Calibration Plot
        self._plot_calibration(axes[0, 0], run_id)
        
        # 2. ROC Curve
        self._plot_roc_curve(axes[0, 1], run_id)
        
        # 3. Precision-Recall Curve
        self._plot_precision_recall(axes[0, 2], run_id)
        
        # 4. Performance Over Time
        self._plot_performance_timeline(axes[1, 0], run_id)
        
        # 5. Profit Over Time
        self._plot_profit_timeline(axes[1, 1], run_id)
        
        # 6. Feature Importance
        self._plot_feature_importance(axes[1, 2], run_id)
        
        plt.tight_layout()
        return fig
    
    def _plot_calibration(self, ax, run_id):
        """Plot calibration curve"""
        query = f"""
        SELECT avg_predicted_prob, actual_hit_rate, count
        FROM calibration_buckets
        WHERE run_id = '{run_id}'
        ORDER BY avg_predicted_prob
        """
        df = pd.read_sql(query, self.conn)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.scatter(df['avg_predicted_prob'], df['actual_hit_rate'], 
                  s=df['count']/10, alpha=0.6)
        ax.plot(df['avg_predicted_prob'], df['actual_hit_rate'], 'b-', 
               label='Model Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Actual Hit Rate')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_roc_curve(self, ax, run_id):
        """Plot ROC curve"""
        query = f"""
        SELECT model_prob, got_rbi
        FROM rbi_predictions_log_v3
        WHERE run_id = '{run_id}'
        """
        df = pd.read_sql(query, self.conn)
        
        fpr, tpr, _ = roc_curve(df['got_rbi'], df['model_prob'])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_precision_recall(self, ax, run_id):
        """Plot Precision-Recall curve"""
        query = f"""
        SELECT model_prob, got_rbi
        FROM rbi_predictions_log_v3
        WHERE run_id = '{run_id}'
        """
        df = pd.read_sql(query, self.conn)
        
        precision, recall, _ = precision_recall_curve(df['got_rbi'], 
                                                       df['model_prob'])
        
        ax.plot(recall, precision, 'b-')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.axhline(y=df['got_rbi'].mean(), color='k', linestyle='--', 
                  label='Baseline')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_timeline(self, ax, run_id):
        """Plot performance over time"""
        query = f"""
        SELECT prediction_date, actual_hit_rate, avg_predicted_prob,
               daily_predictions
        FROM performance_over_time
        WHERE run_id = '{run_id}'
        ORDER BY prediction_date
        """
        df = pd.read_sql(query, self.conn)
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        
        ax.plot(df['prediction_date'], df['actual_hit_rate'], 
               'b-', label='Actual', alpha=0.7)
        ax.plot(df['prediction_date'], df['avg_predicted_prob'], 
               'r--', label='Predicted', alpha=0.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('Hit Rate')
        ax.set_title('Performance Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_profit_timeline(self, ax, run_id):
        """Plot cumulative profit over time"""
        query = f"""
        SELECT prediction_date, daily_profit_units, bets_placed
        FROM performance_over_time
        WHERE run_id = '{run_id}'
        ORDER BY prediction_date
        """
        df = pd.read_sql(query, self.conn)
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df['cumulative_profit'] = df['daily_profit_units'].cumsum()
        
        ax.plot(df['prediction_date'], df['cumulative_profit'], 
               'g-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.fill_between(df['prediction_date'], 0, df['cumulative_profit'],
                        where=(df['cumulative_profit'] >= 0), 
                        color='green', alpha=0.3)
        ax.fill_between(df['prediction_date'], 0, df['cumulative_profit'],
                        where=(df['cumulative_profit'] < 0), 
                        color='red', alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Profit (Units)')
        ax.set_title('Betting P&L Over Time')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_feature_importance(self, ax, run_id):
        """Plot feature importance"""
        query = f"""
        SELECT top_positive_feature, frequency, avg_shap_value
        FROM feature_importance_summary
        WHERE run_id = '{run_id}'
        ORDER BY frequency DESC
        LIMIT 10
        """
        df = pd.read_sql(query, self.conn)
        
        ax.barh(range(len(df)), df['frequency'], color='steelblue')
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['top_positive_feature'])
        ax.set_xlabel('Frequency as Top Feature')
        ax.set_title('Feature Importance (by frequency)')
        ax.grid(True, alpha=0.3, axis='x')
    
    def evaluate_edge_thresholds(self, run_id=None):
        """Find optimal betting edge threshold"""
        if run_id is None:
            run_id = self.get_latest_run_id()
            
        query = f"""
        WITH edge_analysis AS (
            SELECT 
                ROUND(value_edge * 100) / 100.0 as edge_threshold,
                COUNT(*) as num_bets,
                AVG(got_rbi) as win_rate,
                SUM(CASE 
                    WHEN got_rbi = 1 THEN (1.0 / market_implied_prob) - 1
                    ELSE -1
                END) as total_profit,
                SUM(CASE 
                    WHEN got_rbi = 1 THEN (1.0 / market_implied_prob) - 1
                    ELSE -1
                END) / COUNT(*) as roi
            FROM rbi_predictions_log_v3
            WHERE run_id = '{run_id}' 
                AND value_edge IS NOT NULL
                AND value_edge > -0.1
            GROUP BY ROUND(value_edge * 100) / 100.0
            HAVING COUNT(*) >= 5
        )
        SELECT * FROM edge_analysis
        ORDER BY edge_threshold
        """
        
        df = pd.read_sql(query, self.conn)
        
        print("\nEDGE THRESHOLD ANALYSIS")
        print("-" * 60)
        print(f"{'Edge':<8} {'Bets':<8} {'Win%':<8} {'Profit':<10} {'ROI':<8}")
        print("-" * 60)
        
        for _, row in df.iterrows():
            print(f"{row['edge_threshold']:>6.1%}   "
                  f"{row['num_bets']:<8} "
                  f"{row['win_rate']:<8.1%} "
                  f"{row['total_profit']:>9.2f} "
                  f"{row['roi']:>7.1%}")
        
        # Find optimal threshold
        positive_roi = df[df['roi'] > 0]
        if not positive_roi.empty:
            optimal_idx = positive_roi['roi'].idxmax()
            optimal = positive_roi.loc[optimal_idx]
            print(f"\nOptimal Edge Threshold: {optimal['edge_threshold']:.1%}")
            print(f"Expected ROI: {optimal['roi']:.1%}")
            print(f"Sample Size: {optimal['num_bets']} bets")
        
        return df
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Usage Example
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = RBIPredictionEvaluator("rbi_predictions.db")
    
    # Generate comprehensive report
    fig = evaluator.generate_performance_report()
    plt.show()
    
    # Analyze betting edge thresholds
    edge_df = evaluator.evaluate_edge_thresholds()
    
    # Close connection
    evaluator.close()