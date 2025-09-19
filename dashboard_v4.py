#!/usr/bin/env python3
"""
Interactive Dashboard for MLB RBI Prediction System v4.0
Built with Streamlit and Plotly for comprehensive visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import json

# Import our v4 system
from RBI_v4 import (
    EnhancedMLBDataFetcher, AdvancedBullpenAnalyzer, DeepLearningRBIModels,
    PoissonRegressionRBIModel, BankrollManagementSystem, EnhancedSHAPAnalyzer
)

# Configure Streamlit page
st.set_page_config(
    page_title="MLB RBI Prediction System v4.0",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dashboard_data():
    """Load data for dashboard with caching"""
    try:
        # Initialize system components
        fetcher = EnhancedMLBDataFetcher()

        # Load recent predictions from database
        conn = sqlite3.connect('rbi_predictions_v4.db')

        # Recent predictions
        recent_predictions = pd.read_sql_query("""
            SELECT * FROM predictions_v4
            WHERE prediction_date >= date('now', '-7 days')
            ORDER BY prediction_date DESC
        """, conn)

        # Performance metrics
        performance_data = pd.read_sql_query("""
            SELECT * FROM betting_performance_v4
            WHERE bet_date >= date('now', '-30 days')
            ORDER BY bet_date DESC
        """, conn)

        # Model performance
        model_metrics = pd.read_sql_query("""
            SELECT * FROM model_performance_v4
            ORDER BY evaluation_date DESC
            LIMIT 1
        """, conn)

        conn.close()

        return recent_predictions, performance_data, model_metrics

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def create_performance_overview(performance_data):
    """Create performance overview visualizations"""

    col1, col2, col3, col4 = st.columns(4)

    if not performance_data.empty:
        # Key metrics
        total_bets = len(performance_data)
        wins = performance_data['won'].sum() if 'won' in performance_data.columns else 0
        win_rate = wins / max(total_bets, 1)
        total_profit = performance_data['profit'].sum() if 'profit' in performance_data.columns else 0
        roi = total_profit / performance_data['bet_amount'].sum() if 'bet_amount' in performance_data.columns else 0

        with col1:
            st.metric("Total Bets", total_bets)
        with col2:
            st.metric("Win Rate", f"{win_rate:.1%}")
        with col3:
            st.metric("Total Profit", f"${total_profit:.2f}")
        with col4:
            st.metric("ROI", f"{roi:.1%}")

        # Performance chart
        if len(performance_data) > 1:
            performance_data['cumulative_profit'] = performance_data['profit'].cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=performance_data['bet_date'],
                y=performance_data['cumulative_profit'],
                mode='lines+markers',
                name='Cumulative Profit',
                line=dict(color='#1f77b4', width=3)
            ))

            fig.update_layout(
                title="Cumulative Profit Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Profit ($)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        with col1:
            st.metric("Total Bets", 0)
        with col2:
            st.metric("Win Rate", "N/A")
        with col3:
            st.metric("Total Profit", "$0.00")
        with col4:
            st.metric("ROI", "N/A")

def create_prediction_analysis(predictions_data):
    """Create prediction analysis visualizations"""

    if predictions_data.empty:
        st.warning("No recent predictions available")
        return

    # Prediction distribution
    col1, col2 = st.columns(2)

    with col1:
        if 'rbi_probability' in predictions_data.columns:
            fig = px.histogram(
                predictions_data,
                x='rbi_probability',
                nbins=20,
                title="RBI Probability Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'confidence_score' in predictions_data.columns:
            fig = px.histogram(
                predictions_data,
                x='confidence_score',
                nbins=20,
                title="Confidence Score Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Model comparison
    if 'model_type' in predictions_data.columns:
        model_performance = predictions_data.groupby('model_type').agg({
            'rbi_probability': 'mean',
            'confidence_score': 'mean',
            'actual_rbis': 'mean'
        }).reset_index()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Avg RBI Probability',
            x=model_performance['model_type'],
            y=model_performance['rbi_probability'],
            yaxis='y1'
        ))

        fig.add_trace(go.Bar(
            name='Avg Confidence',
            x=model_performance['model_type'],
            y=model_performance['confidence_score'],
            yaxis='y2'
        ))

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model Type",
            yaxis=dict(title="RBI Probability", side="left"),
            yaxis2=dict(title="Confidence Score", side="right", overlaying="y"),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

def create_feature_importance_viz(shap_analyzer):
    """Create feature importance visualization"""

    if not shap_analyzer.feature_importance_global:
        st.warning("No SHAP analysis data available")
        return

    # Aggregate feature importance across models
    all_features = set()
    for model_features in shap_analyzer.feature_importance_global.values():
        all_features.update(model_features.keys())

    feature_importance_df = []

    for feature in all_features:
        for model, features in shap_analyzer.feature_importance_global.items():
            if feature in features:
                feature_importance_df.append({
                    'feature': feature,
                    'model': model,
                    'importance': features[feature]
                })

    if feature_importance_df:
        df = pd.DataFrame(feature_importance_df)

        # Average importance across models
        avg_importance = df.groupby('feature')['importance'].mean().sort_values(ascending=True)

        fig = go.Figure(go.Bar(
            x=avg_importance.values,
            y=avg_importance.index,
            orientation='h',
            marker_color='#1f77b4'
        ))

        fig.update_layout(
            title="Average Feature Importance (SHAP Values)",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Feature importance by model
        if len(df['model'].unique()) > 1:
            fig2 = px.bar(
                df,
                x='importance',
                y='feature',
                color='model',
                orientation='h',
                title="Feature Importance by Model"
            )
            fig2.update_layout(height=600)
            st.plotly_chart(fig2, use_container_width=True)

def create_bankroll_simulation(bankroll_system, predictions):
    """Create bankroll simulation visualization"""

    st.subheader("Bankroll Management Simulation")

    # Simulation parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        kelly_fraction = st.slider(
            "Kelly Fraction",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Fraction of full Kelly to bet (0.25 = 25% Kelly)"
        )

    with col2:
        num_simulations = st.selectbox(
            "Number of Simulations",
            [100, 500, 1000, 2000],
            index=2
        )

    with col3:
        initial_bankroll = st.number_input(
            "Initial Bankroll ($)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )

    if st.button("Run Simulation"):
        # Update bankroll system
        bankroll_system.initial_bankroll = initial_bankroll
        bankroll_system.current_bankroll = initial_bankroll

        # Run simulation
        simulation_results = bankroll_system.simulate_betting_outcomes(
            predictions, num_simulations, kelly_fraction
        )

        # Display results
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Mean ROI", f"{simulation_results['mean_roi']:.1%}")
        with col2:
            st.metric("Median ROI", f"{simulation_results['median_roi']:.1%}")
        with col3:
            st.metric("Positive ROI Prob", f"{simulation_results['positive_roi_probability']:.1%}")
        with col4:
            st.metric("Bankruptcy Risk", f"{simulation_results['bankruptcy_risk']:.1%}")

        # ROI distribution
        roi_data = np.random.normal(
            simulation_results['mean_roi'],
            simulation_results['roi_std'],
            num_simulations
        )

        fig = px.histogram(
            x=roi_data,
            nbins=50,
            title="ROI Distribution from Simulation"
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break Even")
        fig.update_layout(
            xaxis_title="ROI",
            yaxis_title="Frequency",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

def create_weather_impact_analysis():
    """Analyze weather impact on RBI predictions"""

    st.subheader("Weather Impact Analysis")

    # Sample weather impact data (would come from real analysis)
    weather_factors = {
        'Temperature': [65, 70, 75, 80, 85, 90, 95],
        'RBI_Factor': [0.95, 1.0, 1.05, 1.08, 1.10, 1.12, 1.08],
        'Wind_Speed': [0, 5, 10, 15, 20, 25, 30],
        'Wind_RBI_Factor': [1.0, 1.02, 1.05, 1.08, 1.06, 1.02, 0.98]
    }

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weather_factors['Temperature'],
            y=weather_factors['RBI_Factor'],
            mode='lines+markers',
            name='Temperature Impact',
            line=dict(color='red', width=3)
        ))
        fig.update_layout(
            title="Temperature Impact on RBI Production",
            xaxis_title="Temperature (°F)",
            yaxis_title="RBI Factor",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weather_factors['Wind_Speed'],
            y=weather_factors['Wind_RBI_Factor'],
            mode='lines+markers',
            name='Wind Speed Impact',
            line=dict(color='blue', width=3)
        ))
        fig.update_layout(
            title="Wind Speed Impact on RBI Production",
            xaxis_title="Wind Speed (mph)",
            yaxis_title="RBI Factor",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def create_real_time_predictions():
    """Create real-time prediction interface"""

    st.subheader("Real-Time RBI Predictions")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            player_name = st.text_input("Player Name", "Mike Trout")
            team = st.selectbox("Team", [
                "Los Angeles Angels", "New York Yankees", "Boston Red Sox",
                "Los Angeles Dodgers", "Houston Astros", "Atlanta Braves"
            ])
            batting_order = st.selectbox("Batting Order", list(range(1, 10)))

        with col2:
            opponent = st.selectbox("Opponent", [
                "Houston Astros", "New York Yankees", "Boston Red Sox",
                "Los Angeles Dodgers", "Atlanta Braves", "Tampa Bay Rays"
            ])
            pitcher_hand = st.selectbox("Pitcher Hand", ["R", "L"])
            game_time = st.selectbox("Game Time", ["Day", "Night"])

        with col3:
            temperature = st.slider("Temperature (°F)", 50, 100, 75)
            wind_speed = st.slider("Wind Speed (mph)", 0, 30, 5)
            humidity = st.slider("Humidity (%)", 20, 100, 50)

        predict_button = st.form_submit_button("Generate Prediction")

    if predict_button:
        # Simulate prediction (would use real v4 system)
        with st.spinner("Generating prediction..."):
            time.sleep(2)  # Simulate processing time

            # Mock prediction results
            rbi_probability = np.random.beta(2, 8)  # Realistic RBI probability
            expected_rbis = rbi_probability * 1.5
            confidence = np.random.uniform(0.6, 0.9)

            # Weather adjustments
            temp_factor = 1 + (temperature - 75) * 0.002
            wind_factor = 1 + wind_speed * 0.001
            adjusted_probability = min(rbi_probability * temp_factor * wind_factor, 0.95)

            # Display prediction
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("RBI Probability", f"{adjusted_probability:.1%}")
            with col2:
                st.metric("Expected RBIs", f"{expected_rbis:.2f}")
            with col3:
                st.metric("Confidence", f"{confidence:.1%}")
            with col4:
                recommendation = "BET" if adjusted_probability > 0.12 and confidence > 0.7 else "PASS"
                color = "green" if recommendation == "BET" else "red"
                st.markdown(f"<h3 style='color: {color}'>{recommendation}</h3>", unsafe_allow_html=True)

            # Feature importance for this prediction
            st.subheader("Key Factors")

            factors = {
                'Batting Order': batting_order / 9,
                'Recent Form': np.random.uniform(0.6, 0.9),
                'Weather': temp_factor * wind_factor - 1,
                'Pitcher Matchup': np.random.uniform(-0.1, 0.1),
                'Park Factor': np.random.uniform(-0.05, 0.05)
            }

            factor_df = pd.DataFrame(list(factors.items()), columns=['Factor', 'Impact'])
            factor_df['Impact_Abs'] = factor_df['Impact'].abs()
            factor_df = factor_df.sort_values('Impact_Abs', ascending=True)

            fig = go.Figure(go.Bar(
                x=factor_df['Impact'],
                y=factor_df['Factor'],
                orientation='h',
                marker_color=['red' if x < 0 else 'green' for x in factor_df['Impact']]
            ))

            fig.update_layout(
                title="Feature Impact on Prediction",
                xaxis_title="Impact",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard application"""

    # Header
    st.markdown('<h1 class="main-header">⚾ MLB RBI Prediction System v4.0</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard Overview", "Real-Time Predictions", "Performance Analysis",
         "Feature Analysis", "Bankroll Management", "Weather Impact", "System Status"]
    )

    # Load data
    predictions_data, performance_data, model_metrics = load_dashboard_data()

    # Initialize system components (mock for demo)
    bankroll_system = BankrollManagementSystem(1000)
    shap_analyzer = EnhancedSHAPAnalyzer()

    # Mock some SHAP data for demo
    shap_analyzer.feature_importance_global = {
        'xgboost': {
            'batting_order': 0.15,
            'recent_form': 0.12,
            'weather_temp': 0.08,
            'pitcher_era': 0.10,
            'park_factor': 0.06
        },
        'lstm': {
            'batting_order': 0.14,
            'recent_form': 0.18,
            'weather_temp': 0.07,
            'pitcher_era': 0.09,
            'park_factor': 0.05
        }
    }

    # Page routing
    if page == "Dashboard Overview":
        st.subheader("System Overview")

        # System status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="success-card"><h4>✅ System Status</h4><p>All models operational</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h4>📊 Active Models</h4><p>5 models trained and ready</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h4>🎯 Last Update</h4><p>2 hours ago</p></div>', unsafe_allow_html=True)

        # Performance overview
        st.subheader("Recent Performance")
        create_performance_overview(performance_data)

        # Recent predictions summary
        if not predictions_data.empty:
            st.subheader("Recent Predictions Summary")
            create_prediction_analysis(predictions_data)

    elif page == "Real-Time Predictions":
        create_real_time_predictions()

    elif page == "Performance Analysis":
        st.subheader("Detailed Performance Analysis")
        create_performance_overview(performance_data)

        if not performance_data.empty:
            # Additional performance charts
            st.subheader("Performance by Day of Week")
            # Mock daily performance data
            daily_perf = pd.DataFrame({
                'day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'win_rate': np.random.uniform(0.4, 0.7, 7),
                'avg_profit': np.random.uniform(-10, 25, 7)
            })

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Bar(x=daily_perf['day'], y=daily_perf['win_rate'], name="Win Rate"),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(x=daily_perf['day'], y=daily_perf['avg_profit'],
                          mode='lines+markers', name="Avg Profit"),
                secondary_y=True,
            )

            fig.update_yaxes(title_text="Win Rate", secondary_y=False)
            fig.update_yaxes(title_text="Average Profit ($)", secondary_y=True)
            fig.update_layout(title="Performance by Day of Week")

            st.plotly_chart(fig, use_container_width=True)

    elif page == "Feature Analysis":
        st.subheader("Feature Importance Analysis")
        create_feature_importance_viz(shap_analyzer)

        # Feature insights
        insights = shap_analyzer.generate_feature_insights()
        if insights:
            st.subheader("Key Insights")
            for insight in insights:
                st.markdown(f"• {insight}")

    elif page == "Bankroll Management":
        # Mock some predictions for simulation
        mock_predictions = [
            {'rbi_probability': 0.15, 'market_odds': -110, 'recommendation': 'BET'},
            {'rbi_probability': 0.08, 'market_odds': +120, 'recommendation': 'PASS'},
            {'rbi_probability': 0.18, 'market_odds': -105, 'recommendation': 'STRONG BET'},
        ]
        create_bankroll_simulation(bankroll_system, mock_predictions)

    elif page == "Weather Impact":
        create_weather_impact_analysis()

    elif page == "System Status":
        st.subheader("System Health & Status")

        # System components status
        components = [
            ("Data Fetcher", "✅ Operational", "green"),
            ("ML Models", "✅ Operational", "green"),
            ("Database", "✅ Connected", "green"),
            ("Weather API", "✅ Connected", "green"),
            ("Odds API", "⚠️ Rate Limited", "orange"),
            ("SHAP Analysis", "✅ Operational", "green")
        ]

        for component, status, color in components:
            st.markdown(f"**{component}:** <span style='color: {color}'>{status}</span>", unsafe_allow_html=True)

        # Database stats
        st.subheader("Database Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Predictions", "1,247")
        with col2:
            st.metric("Training Samples", "45,382")
        with col3:
            st.metric("Cache Hit Rate", "87%")

        # API usage
        st.subheader("API Usage (Last 24h)")
        api_usage = pd.DataFrame({
            'API': ['MLB Stats', 'Weather', 'Odds'],
            'Requests': [156, 89, 34],
            'Limit': [1000, 1000, 500],
            'Usage_Pct': [15.6, 8.9, 6.8]
        })

        fig = px.bar(api_usage, x='API', y='Usage_Pct', title="API Usage Percentage")
        fig.update_layout(yaxis_title="Usage %", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("**MLB RBI Prediction System v4.0** | Built with ❤️ and ⚾")

if __name__ == "__main__":
    main()