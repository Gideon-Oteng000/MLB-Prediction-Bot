"""
RBI Prediction Real-Time Monitoring Dashboard
==============================================
Streamlit dashboard for monitoring model performance
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="RBI Prediction Monitor",
    page_icon="⚾",
    layout="wide"
)

@st.cache_resource
def get_connection():
    """Create database connection"""
    return sqlite3.connect("rbi_predictions.db", check_same_thread=False)

def load_summary_stats(conn, run_id):
    """Load summary statistics for a run"""
    query = f"""
    SELECT 
        COUNT(*) as total_predictions,
        SUM(got_rbi) as total_hits,
        AVG(model_prob) as avg_predicted_prob,
        AVG(CAST(got_rbi AS REAL)) as actual_hit_rate,
        AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as brier_score,
        SUM(CASE WHEN value_edge > 0.05 THEN 1 ELSE 0 END) as bets_placed,
        SUM(CASE 
            WHEN value_edge > 0.05 AND got_rbi = 1 THEN (1.0 / market_implied_prob) - 1
            WHEN value_edge > 0.05 AND got_rbi = 0 THEN -1
            ELSE 0
        END) as total_profit
    FROM rbi_predictions_log_v3
    WHERE run_id = '{run_id}'
    """
    return pd.read_sql(query, conn).iloc[0]

def load_recent_predictions(conn, run_id, limit=20):
    """Load most recent predictions"""
    query = f"""
    SELECT 
        date,
        player_name,
        team,
        opponent,
        model_prob,
        market_implied_prob,
        value_edge,
        got_rbi,
        rbi_count
    FROM rbi_predictions_log_v3
    WHERE run_id = '{run_id}'
    ORDER BY date DESC, model_prob DESC
    LIMIT {limit}
    """
    return pd.read_sql(query, conn)

def create_calibration_plot(conn, run_id):
    """Create calibration plot"""
    query = f"""
    SELECT 
        ROUND(model_prob * 10) / 10.0 as prob_bucket,
        AVG(model_prob) as avg_predicted,
        AVG(CAST(got_rbi AS REAL)) as actual_rate,
        COUNT(*) as count
    FROM rbi_predictions_log_v3
    WHERE run_id = '{run_id}'
    GROUP BY ROUND(model_prob * 10) / 10.0
    """
    df = pd.read_sql(query, conn)
    
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='gray')
    ))
    
    # Model calibration
    fig.add_trace(go.Scatter(
        x=df['avg_predicted'],
        y=df['actual_rate'],
        mode='lines+markers',
        name='Model',
        marker=dict(size=df['count']/df['count'].max()*20),
        text=df['count'],
        hovertemplate='Predicted: %{x:.1%}<br>Actual: %{y:.1%}<br>Count: %{text}'
    ))
    
    fig.update_layout(
        title="Calibration Plot",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Hit Rate",
        xaxis=dict(tickformat='.0%'),
        yaxis=dict(tickformat='.0%'),
        height=400
    )
    
    return fig

def create_profit_timeline(conn, run_id):
    """Create profit timeline"""
    query = f"""
    SELECT 
        DATE(date) as prediction_date,
        SUM(CASE 
            WHEN value_edge > 0.05 AND got_rbi = 1 THEN (1.0 / market_implied_prob) - 1
            WHEN value_edge > 0.05 AND got_rbi = 0 THEN -1
            ELSE 0
        END) as daily_profit,
        COUNT(CASE WHEN value_edge > 0.05 THEN 1 END) as daily_bets
    FROM rbi_predictions_log_v3
    WHERE run_id = '{run_id}'
    GROUP BY DATE(date)
    ORDER BY prediction_date
    """
    df = pd.read_sql(query, conn)
    df['cumulative_profit'] = df['daily_profit'].cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['prediction_date'],
        y=df['cumulative_profit'],
        mode='lines',
        name='Cumulative P&L',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgb(0,100,80)', width=2)
    ))
    
    fig.add_trace(go.Bar(
        x=df['prediction_date'],
        y=df['daily_profit'],
        name='Daily P&L',
        marker_color=np.where(df['daily_profit'] >= 0, 'lightgreen', 'lightcoral'),
        yaxis='y2',
        opacity=0.5
    ))
    
    fig.update_layout(
        title="Betting Performance Timeline",
        xaxis_title="Date",
        yaxis_title="Cumulative Profit (Units)",
        yaxis2=dict(
            title="Daily Profit",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_feature_importance_chart(conn, run_id):
    """Create feature importance chart"""
    query = f"""
    SELECT 
        top_positive_feature as feature,
        COUNT(*) as frequency,
        AVG(top_positive_value) as avg_importance
    FROM rbi_predictions_log_v3
    WHERE run_id = '{run_id}' AND top_positive_feature IS NOT NULL
    GROUP BY top_positive_feature
    ORDER BY frequency DESC
    LIMIT 15
    """
    df = pd.read_sql(query, conn)
    
    fig = px.bar(
        df, 
        y='feature', 
        x='frequency',
        orientation='h',
        title='Top Features by Frequency',
        labels={'frequency': 'Times as Top Feature', 'feature': ''},
        color='avg_importance',
        color_continuous_scale='Blues',
        height=400
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig

def main():
    st.title("⚾ RBI Prediction Model Monitor")
    st.markdown("Real-time monitoring dashboard for RBI prediction models")
    
    conn = get_connection()
    
    # Sidebar for run selection
    st.sidebar.header("Configuration")
    
    # Get available runs
    runs_query = """
    SELECT DISTINCT run_id, MIN(date) as start_date, MAX(date) as end_date, COUNT(*) as predictions
    FROM rbi_predictions_log_v3
    GROUP BY run_id
    ORDER BY run_id DESC
    """
    runs_df = pd.read_sql(runs_query, conn)
    
    if runs_df.empty:
        st.error("No prediction runs found in database")
        return
    
    selected_run = st.sidebar.selectbox(
        "Select Run ID",
        runs_df['run_id'].tolist(),
        format_func=lambda x: f"{x} ({runs_df[runs_df['run_id']==x]['predictions'].iloc[0]:,} predictions)"
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Load data
    stats = load_summary_stats(conn, selected_run)
    
    # Header metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Predictions",
            f"{int(stats['total_predictions']):,}",
            delta=None
        )
    
    with col2:
        hit_rate = stats['actual_hit_rate']
        predicted_rate = stats['avg_predicted_prob']
        st.metric(
            "Hit Rate",
            f"{hit_rate:.1%}",
            delta=f"{(hit_rate - predicted_rate):.1%} vs predicted"
        )
    
    with col3:
        st.metric(
            "Brier Score",
            f"{stats['brier_score']:.4f}",
            delta="Lower is better",
            delta_color="off"
        )
    
    with col4:
        st.metric(
            "Bets Placed",
            f"{int(stats['bets_placed']):,}",
            delta=None
        )
    
    with col5:
        profit = stats['total_profit'] or 0
        roi = profit / stats['bets_placed'] if stats['bets_placed'] > 0 else 0
        st.metric(
            "Total P&L",
            f"{profit:+.1f} units",
            delta=f"ROI: {roi:.1%}"
        )
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "💰 Betting", "📈 Analytics", "📋 Recent Predictions"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_calibration_plot(conn, selected_run), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_feature_importance_chart(conn, selected_run), use_container_width=True)
        
        # Performance by subgroup
        st.subheader("Performance by Subgroup")
        
        subgroup_query = f"""
        SELECT 
            lineup_spot,
            COUNT(*) as predictions,
            AVG(model_prob) as avg_predicted,
            AVG(CAST(got_rbi AS REAL)) as actual_rate,
            AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as brier_score
        FROM rbi_predictions_log_v3
        WHERE run_id = '{selected_run}'
        GROUP BY lineup_spot
        ORDER BY lineup_spot
        """
        subgroup_df = pd.read_sql(subgroup_query, conn)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Predicted', x=subgroup_df['lineup_spot'], y=subgroup_df['avg_predicted']))
        fig.add_trace(go.Bar(name='Actual', x=subgroup_df['lineup_spot'], y=subgroup_df['actual_rate']))
        fig.update_layout(
            title="Hit Rate by Lineup Position",
            xaxis_title="Lineup Spot",
            yaxis_title="Hit Rate",
            yaxis=dict(tickformat='.0%'),
            barmode='group',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_profit_timeline(conn, selected_run), use_container_width=True)
        
        # Edge threshold analysis
        st.subheader("Edge Threshold Analysis")
        
        edge_query = f"""
        SELECT 
            ROUND(value_edge * 20) / 20.0 as edge_bucket,
            COUNT(*) as bets,
            AVG(got_rbi) as win_rate,
            SUM(CASE 
                WHEN got_rbi = 1 THEN (1.0 / market_implied_prob) - 1
                ELSE -1
            END) / COUNT(*) as roi
        FROM rbi_predictions_log_v3
        WHERE run_id = '{selected_run}' AND value_edge IS NOT NULL
        GROUP BY ROUND(value_edge * 20) / 20.0
        HAVING COUNT(*) >= 5
        ORDER BY edge_bucket
        """
        edge_df = pd.read_sql(edge_query, conn)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(edge_df, x='edge_bucket', y='roi', 
                         title='ROI by Value Edge',
                         labels={'edge_bucket': 'Value Edge', 'roi': 'ROI'})
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(yaxis=dict(tickformat='.0%'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(edge_df, x='edge_bucket', y='bets',
                        title='Bet Distribution by Edge',
                        labels={'edge_bucket': 'Value Edge', 'bets': 'Number of Bets'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Time series analysis
        st.subheader("Model Performance Over Time")
        
        time_query = f"""
        SELECT 
            DATE(date) as prediction_date,
            COUNT(*) as daily_predictions,
            AVG(model_prob) as avg_predicted,
            AVG(CAST(got_rbi AS REAL)) as actual_rate,
            AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as brier_score
        FROM rbi_predictions_log_v3
        WHERE run_id = '{selected_run}'
        GROUP BY DATE(date)
        ORDER BY prediction_date
        """
        time_df = pd.read_sql(time_query, conn)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_df['prediction_date'], y=time_df['actual_rate'],
                                mode='lines', name='Actual Hit Rate'))
        fig.add_trace(go.Scatter(x=time_df['prediction_date'], y=time_df['avg_predicted'],
                                mode='lines', name='Predicted Rate', line=dict(dash='dash')))
        fig.update_layout(
            title="Hit Rate Over Time",
            xaxis_title="Date",
            yaxis_title="Rate",
            yaxis=dict(tickformat='.0%'),
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Player performance
        st.subheader("Top Player Performance")
        
        player_query = f"""
        SELECT 
            player_name,
            COUNT(*) as predictions,
            AVG(model_prob) as avg_predicted,
            AVG(CAST(got_rbi AS REAL)) as actual_rate,
            SUM(got_rbi) as total_rbis,
            AVG(model_prob) - AVG(CAST(got_rbi AS REAL)) as calibration_error
        FROM rbi_predictions_log_v3
        WHERE run_id = '{selected_run}'
        GROUP BY player_name
        HAVING COUNT(*) >= 10
        ORDER BY COUNT(*) DESC
        LIMIT 20
        """
        player_df = pd.read_sql(player_query, conn)
        
        st.dataframe(
            player_df.style.format({
                'avg_predicted': '{:.1%}',
                'actual_rate': '{:.1%}',
                'calibration_error': '{:+.1%}'
            }),
            use_container_width=True
        )
    
    with tab4:
        st.subheader("Recent Predictions")
        
        recent_df = load_recent_predictions(conn, selected_run, limit=50)
        
        # Format for display
        display_df = recent_df.copy()
        display_df['Result'] = display_df.apply(
            lambda x: f"✅ {int(x['rbi_count'])} RBI" if x['got_rbi'] else "❌ No RBI", 
            axis=1
        )
        display_df['Edge'] = display_df['value_edge'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        
        st.dataframe(
            display_df[['date', 'player_name', 'team', 'opponent', 'model_prob', 
                       'market_implied_prob', 'Edge', 'Result']].style.format({
                'model_prob': '{:.1%}',
                'market_implied_prob': '{:.1%}'
            }),
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()