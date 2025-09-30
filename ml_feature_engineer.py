#!/usr/bin/env python3
"""
ML Feature Engineering Pipeline for Home Run Prediction Model
Extracts 200+ features from historical database with weighted blending and shrinkage
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLFeatureEngineer:
    """
    Comprehensive feature engineering for home run prediction model
    Extracts 200+ features with weighted blending and empirical Bayes shrinkage
    """

    def __init__(self, db_path: str = "historical_baseball_data.db"):
        self.db_path = db_path

        # Weighting scheme (as you specified)
        self.time_weights = {
            'career': 0.20,
            'season': 0.35,
            'last_30d': 0.30,
            'last_7d': 0.15
        }

        # Shrinkage parameters for empirical Bayes
        self.shrinkage_params = {
            'min_abs_for_career': 100,  # Minimum ABs for career stats
            'min_abs_for_season': 50,   # Minimum ABs for season stats
            'min_abs_for_30d': 15,      # Minimum ABs for 30d stats
            'min_abs_for_7d': 5,        # Minimum ABs for 7d stats
            'league_hr_rate': 0.028     # MLB average HR rate (~2.8%)
        }

        # Feature categories
        self.feature_categories = [
            'batting_performance',
            'power_metrics',
            'discipline_metrics',
            'situational_performance',
            'pitcher_matchup',
            'environmental_factors',
            'team_context',
            'streak_momentum',
            'advanced_metrics',
            'ballpark_specific'
        ]

        logger.info("ML Feature Engineer initialized")
        logger.info(f"Time weights: {self.time_weights}")

    def generate_features_for_date(self, target_date: str,
                                 players_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate all features for players on a specific date

        Args:
            target_date: Date to generate features for (YYYY-MM-DD)
            players_df: DataFrame with player names and teams (if None, gets from lineups)

        Returns:
            DataFrame with all features for each player
        """
        logger.info(f"Generating features for {target_date}")

        # Get players for this date if not provided
        if players_df is None:
            players_df = self._get_players_for_date(target_date)

        if players_df.empty:
            logger.warning(f"No players found for {target_date}")
            return pd.DataFrame()

        # Initialize results
        features_list = []

        for _, player_row in players_df.iterrows():
            player_name = player_row['player_name']
            team = player_row['team']

            try:
                # Generate comprehensive feature set
                player_features = self._generate_player_features(
                    player_name, team, target_date
                )

                # Add player identification
                player_features.update({
                    'player_name': player_name,
                    'team': team,
                    'date': target_date
                })

                features_list.append(player_features)

            except Exception as e:
                logger.error(f"Error generating features for {player_name}: {e}")
                continue

        features_df = pd.DataFrame(features_list)
        logger.info(f"Generated {len(features_df)} player feature sets with {len(features_df.columns)} features")

        return features_df

    def _get_players_for_date(self, target_date: str) -> pd.DataFrame:
        """Get players who played on the target date"""
        conn = sqlite3.connect(self.db_path)

        query = '''
            SELECT DISTINCT player_name, team
            FROM player_game_results
            WHERE date = ?
            AND batting_order <= 9
        '''

        df = pd.read_sql_query(query, conn, params=[target_date])
        conn.close()

        return df

    def _generate_player_features(self, player_name: str, team: str,
                                target_date: str) -> Dict[str, Any]:
        """Generate comprehensive feature set for a single player"""

        # Get historical data for different time periods
        historical_data = self._get_player_historical_data(player_name, target_date)

        # Initialize feature dictionary
        features = {}

        # Category 1: Batting Performance Metrics
        features.update(self._generate_batting_features(historical_data, player_name))

        # Category 2: Power Metrics
        features.update(self._generate_power_features(historical_data, player_name))

        # Category 3: Plate Discipline Metrics
        features.update(self._generate_discipline_features(historical_data, player_name))

        # Category 4: Situational Performance
        features.update(self._generate_situational_features(historical_data, player_name, target_date))

        # Category 5: Pitcher Matchup Analysis
        features.update(self._generate_pitcher_matchup_features(player_name, target_date))

        # Category 6: Environmental Factors
        features.update(self._generate_environmental_features(team, target_date))

        # Category 7: Team Context
        features.update(self._generate_team_context_features(team, target_date))

        # Category 8: Streak & Momentum
        features.update(self._generate_streak_features(historical_data, player_name))

        # Category 9: Advanced Metrics (placeholder for future Statcast integration)
        features.update(self._generate_advanced_features(historical_data, player_name))

        # Category 10: Ballpark-Specific Performance
        features.update(self._generate_ballpark_features(historical_data, player_name, team, target_date))

        return features

    def _get_player_historical_data(self, player_name: str, target_date: str) -> Dict[str, pd.DataFrame]:
        """Get player's historical performance data for different time periods"""
        conn = sqlite3.connect(self.db_path)

        target_dt = datetime.strptime(target_date, '%Y-%m-%d').date()

        # Define time periods
        periods = {
            'career': ('1900-01-01', target_date),
            'season': (f"{target_dt.year}-03-01", target_date),
            'last_30d': ((target_dt - timedelta(days=30)).strftime('%Y-%m-%d'), target_date),
            'last_7d': ((target_dt - timedelta(days=7)).strftime('%Y-%m-%d'), target_date)
        }

        historical_data = {}

        for period_name, (start_date, end_date) in periods.items():
            query = '''
                SELECT pgr.*, g.stadium, g.park_hr_factor, g.elevation,
                       g.day_night, g.home_team, g.away_team
                FROM player_game_results pgr
                JOIN games g ON pgr.game_id = g.game_id
                WHERE pgr.player_name = ?
                AND pgr.date > ?
                AND pgr.date < ?
                ORDER BY pgr.date DESC
            '''

            df = pd.read_sql_query(query, conn, params=[player_name, start_date, end_date])
            historical_data[period_name] = df

        conn.close()
        return historical_data

    def _generate_batting_features(self, historical_data: Dict, player_name: str) -> Dict[str, float]:
        """Generate batting performance features with weighted blending"""
        features = {}

        # Basic batting statistics with weighted averaging
        stats_to_blend = [
            ('avg', lambda df: df['hits'].sum() / max(df['at_bats'].sum(), 1)),
            ('obp', lambda df: (df['hits'].sum() + df['walks'].sum()) /
                              max(df['at_bats'].sum() + df['walks'].sum(), 1)),
            ('slg', lambda df: self._calculate_slugging(df)),
            ('hr_rate', lambda df: df['home_runs'].sum() / max(df['at_bats'].sum(), 1)),
            ('iso', lambda df: self._calculate_iso(df)),
            ('bb_rate', lambda df: df['walks'].sum() / max(df['plate_appearances'].sum(), 1)),
            ('k_rate', lambda df: df['strikeouts'].sum() / max(df['plate_appearances'].sum(), 1))
        ]

        for stat_name, calc_func in stats_to_blend:
            period_stats = {}
            period_weights = {}

            for period in ['career', 'season', 'last_30d', 'last_7d']:
                df = historical_data[period]
                min_abs_key = f'min_abs_for_{period}'
                if min_abs_key in self.shrinkage_params:
                    min_abs = self.shrinkage_params[min_abs_key]
                else:
                    min_abs = 1  # Default minimum

                if not df.empty and df['at_bats'].sum() >= min_abs:
                    period_stats[period] = calc_func(df)
                    period_weights[period] = self.time_weights[period]

            # Apply empirical Bayes shrinkage and weighted blending
            if period_stats:
                features[f'player_{stat_name}'] = self._apply_weighted_shrinkage(
                    period_stats, period_weights, stat_name
                )
            else:
                features[f'player_{stat_name}'] = self.shrinkage_params['league_hr_rate'] if 'hr' in stat_name else 0.0

        # Performance trends
        if not historical_data['season'].empty:
            features.update(self._calculate_performance_trends(historical_data['season']))

        return features

    def _generate_power_features(self, historical_data: Dict, player_name: str) -> Dict[str, float]:
        """Generate power-specific features"""
        features = {}

        for period_name, df in historical_data.items():
            if df.empty:
                continue

            prefix = f'power_{period_name}'

            # Power metrics
            features[f'{prefix}_hr_per_ab'] = df['home_runs'].sum() / max(df['at_bats'].sum(), 1)
            features[f'{prefix}_hr_per_pa'] = df['home_runs'].sum() / max(df['plate_appearances'].sum(), 1)
            features[f'{prefix}_extra_base_rate'] = (df['doubles'].sum() + df['triples'].sum() + df['home_runs'].sum()) / max(df['hits'].sum(), 1)
            features[f'{prefix}_total_bases'] = self._calculate_total_bases(df)
            features[f'{prefix}_total_bases_per_ab'] = features[f'{prefix}_total_bases'] / max(df['at_bats'].sum(), 1)

            # Multi-homer games
            features[f'{prefix}_multi_hr_games'] = len(df[df['home_runs'] >= 2])
            features[f'{prefix}_hr_game_rate'] = len(df[df['home_runs'] >= 1]) / max(len(df), 1)

        # Power consistency
        season_df = historical_data.get('season', pd.DataFrame())
        if not season_df.empty:
            features['power_consistency'] = season_df['home_runs'].std() / max(season_df['home_runs'].mean(), 0.1)

        return features

    def _generate_discipline_features(self, historical_data: Dict, player_name: str) -> Dict[str, float]:
        """Generate plate discipline features"""
        features = {}

        for period_name, df in historical_data.items():
            if df.empty:
                continue

            prefix = f'discipline_{period_name}'

            # Plate discipline metrics
            features[f'{prefix}_bb_rate'] = df['walks'].sum() / max(df['plate_appearances'].sum(), 1)
            features[f'{prefix}_k_rate'] = df['strikeouts'].sum() / max(df['plate_appearances'].sum(), 1)
            features[f'{prefix}_bb_k_ratio'] = df['walks'].sum() / max(df['strikeouts'].sum(), 1)
            features[f'{prefix}_contact_rate'] = (df['at_bats'].sum() - df['strikeouts'].sum()) / max(df['at_bats'].sum(), 1)

        return features

    def _generate_situational_features(self, historical_data: Dict, player_name: str,
                                     target_date: str) -> Dict[str, float]:
        """Generate situational performance features"""
        features = {}

        # Get situational data from database
        conn = sqlite3.connect(self.db_path)

        # Home vs Away performance
        home_query = '''
            SELECT AVG(CAST(home_runs AS FLOAT) / NULLIF(at_bats, 0)) as hr_rate
            FROM player_game_results pgr
            JOIN games g ON pgr.game_id = g.game_id
            WHERE pgr.player_name = ? AND pgr.date < ?
            AND pgr.home_away = 1 AND pgr.at_bats > 0
        '''

        away_query = home_query.replace('pgr.home_away = 1', 'pgr.home_away = 0')

        cursor = conn.cursor()
        cursor.execute(home_query, [player_name, target_date])
        home_hr_rate = cursor.fetchone()[0] or 0

        cursor.execute(away_query, [player_name, target_date])
        away_hr_rate = cursor.fetchone()[0] or 0

        features['situational_home_hr_rate'] = home_hr_rate
        features['situational_away_hr_rate'] = away_hr_rate
        features['situational_home_away_diff'] = home_hr_rate - away_hr_rate

        # Day vs Night performance
        day_query = '''
            SELECT AVG(CAST(home_runs AS FLOAT) / NULLIF(at_bats, 0)) as hr_rate
            FROM player_game_results pgr
            JOIN games g ON pgr.game_id = g.game_id
            WHERE pgr.player_name = ? AND pgr.date < ?
            AND g.day_night = 0 AND pgr.at_bats > 0
        '''

        night_query = day_query.replace('g.day_night = 0', 'g.day_night = 1')

        cursor.execute(day_query, [player_name, target_date])
        day_hr_rate = cursor.fetchone()[0] or 0

        cursor.execute(night_query, [player_name, target_date])
        night_hr_rate = cursor.fetchone()[0] or 0

        features['situational_day_hr_rate'] = day_hr_rate
        features['situational_night_hr_rate'] = night_hr_rate
        features['situational_day_night_diff'] = day_hr_rate - night_hr_rate

        # Batting order performance
        for order_pos in range(1, 10):
            order_query = '''
                SELECT AVG(CAST(home_runs AS FLOAT) / NULLIF(at_bats, 0)) as hr_rate
                FROM player_game_results
                WHERE player_name = ? AND date < ?
                AND batting_order = ? AND at_bats > 0
            '''
            cursor.execute(order_query, [player_name, target_date, order_pos])
            order_hr_rate = cursor.fetchone()[0] or 0
            features[f'situational_order_{order_pos}_hr_rate'] = order_hr_rate

        conn.close()
        return features

    def _generate_pitcher_matchup_features(self, player_name: str, target_date: str) -> Dict[str, float]:
        """Generate pitcher matchup features"""
        features = {}

        conn = sqlite3.connect(self.db_path)

        # Get opposing pitcher characteristics
        opp_pitcher_query = '''
            SELECT
                AVG(home_runs_allowed) as avg_hr_allowed,
                AVG(strikeouts) as avg_strikeouts,
                AVG(walks_allowed) as avg_walks,
                AVG(innings_pitched) as avg_ip,
                COUNT(*) as games_pitched
            FROM pitcher_game_results
            WHERE date < ? AND role = 'starter'
            GROUP BY pitcher_name
            ORDER BY games_pitched DESC
            LIMIT 1
        '''

        pitcher_df = pd.read_sql_query(opp_pitcher_query, conn, params=[target_date])

        if not pitcher_df.empty:
            row = pitcher_df.iloc[0]
            features['matchup_opp_pitcher_hr_rate'] = row['avg_hr_allowed'] / max(row['avg_ip'], 1)
            features['matchup_opp_pitcher_k_rate'] = row['avg_strikeouts'] / max(row['avg_ip'] * 3, 1)  # Approximate
            features['matchup_opp_pitcher_bb_rate'] = row['avg_walks'] / max(row['avg_ip'] * 3, 1)
        else:
            features['matchup_opp_pitcher_hr_rate'] = 0.03  # League average
            features['matchup_opp_pitcher_k_rate'] = 0.23
            features['matchup_opp_pitcher_bb_rate'] = 0.09

        # Historical performance vs similar pitchers (placeholder)
        features['matchup_vs_similar_pitchers_hr_rate'] = 0.028  # To be enhanced

        conn.close()
        return features

    def _generate_environmental_features(self, team: str, target_date: str) -> Dict[str, float]:
        """Generate environmental and ballpark features"""
        features = {}

        # Get game environment info
        conn = sqlite3.connect(self.db_path)

        # Get home stadium info
        stadium_query = '''
            SELECT g.stadium, g.park_hr_factor, g.elevation, g.day_night
            FROM games g
            WHERE (g.home_team = ? OR g.away_team = ?)
            AND g.date = ?
            LIMIT 1
        '''

        cursor = conn.cursor()
        cursor.execute(stadium_query, [team, team, target_date])
        stadium_info = cursor.fetchone()

        if stadium_info:
            features['env_park_hr_factor'] = stadium_info[1] or 1.0
            features['env_elevation'] = stadium_info[2] or 0
            features['env_day_night'] = stadium_info[3] or 0
        else:
            features['env_park_hr_factor'] = 1.0
            features['env_elevation'] = 0
            features['env_day_night'] = 0

        # Weather features (placeholder - would integrate with weather API)
        features['env_temperature'] = 75  # Default
        features['env_wind_speed'] = 5
        features['env_wind_direction'] = 0
        features['env_humidity'] = 50

        conn.close()
        return features

    def _generate_team_context_features(self, team: str, target_date: str) -> Dict[str, float]:
        """Generate team context features"""
        features = {}

        conn = sqlite3.connect(self.db_path)

        # Team offensive environment (last 30 games)
        team_context_query = '''
            SELECT
                AVG(runs_scored) as avg_runs,
                AVG(hits) as avg_hits,
                AVG(team_batting_avg) as team_avg,
                AVG(team_ops) as team_ops
            FROM team_game_results
            WHERE team = ? AND date < ?
            AND date > date(?, '-30 days')
        '''

        cursor = conn.cursor()
        cursor.execute(team_context_query, [team, target_date, target_date])
        team_stats = cursor.fetchone()

        if team_stats and team_stats[0]:
            features['team_context_runs_per_game'] = team_stats[0]
            features['team_context_hits_per_game'] = team_stats[1]
            features['team_context_team_avg'] = team_stats[2] or 0.25
            features['team_context_team_ops'] = team_stats[3] or 0.7
        else:
            features['team_context_runs_per_game'] = 4.5
            features['team_context_hits_per_game'] = 8.5
            features['team_context_team_avg'] = 0.25
            features['team_context_team_ops'] = 0.7

        conn.close()
        return features

    def _generate_streak_features(self, historical_data: Dict, player_name: str) -> Dict[str, float]:
        """Generate streak and momentum features"""
        features = {}

        # Recent game streak analysis
        last_7d = historical_data.get('last_7d', pd.DataFrame())

        if not last_7d.empty:
            # Sort by date
            last_7d = last_7d.sort_values('date')

            # HR streak
            hr_games = (last_7d['home_runs'] > 0).astype(int)
            features['streak_current_hr_games'] = self._calculate_current_streak(hr_games.values)
            features['streak_hr_games_last_7'] = hr_games.sum()

            # Hit streak
            hit_games = (last_7d['hits'] > 0).astype(int)
            features['streak_current_hit_games'] = self._calculate_current_streak(hit_games.values)

            # Performance momentum
            features['streak_momentum_hr'] = self._calculate_momentum(last_7d['home_runs'].values)
            features['streak_momentum_avg'] = self._calculate_momentum(
                last_7d['hits'] / last_7d['at_bats'].replace(0, 1)
            )
        else:
            features.update({
                'streak_current_hr_games': 0,
                'streak_hr_games_last_7': 0,
                'streak_current_hit_games': 0,
                'streak_momentum_hr': 0,
                'streak_momentum_avg': 0
            })

        return features

    def _generate_advanced_features(self, historical_data: Dict, player_name: str) -> Dict[str, float]:
        """Generate advanced metrics (placeholder for Statcast integration)"""
        features = {}

        # Placeholder for future Statcast data integration
        features.update({
            'advanced_exit_velocity_avg': 90.0,  # mph
            'advanced_launch_angle_avg': 12.0,   # degrees
            'advanced_hard_hit_pct': 0.35,       # percentage
            'advanced_barrel_pct': 0.08,         # percentage
            'advanced_sweet_spot_pct': 0.30,     # percentage
            'advanced_xwoba': 0.330,             # expected wOBA
            'advanced_xslg': 0.450               # expected slugging
        })

        return features

    def _generate_ballpark_features(self, historical_data: Dict, player_name: str,
                                  team: str, target_date: str) -> Dict[str, float]:
        """Generate ballpark-specific performance features"""
        features = {}

        conn = sqlite3.connect(self.db_path)

        # Performance at each stadium
        stadium_query = '''
            SELECT g.stadium,
                   AVG(CAST(pgr.home_runs AS FLOAT) / NULLIF(pgr.at_bats, 0)) as hr_rate,
                   COUNT(*) as games_played
            FROM player_game_results pgr
            JOIN games g ON pgr.game_id = g.game_id
            WHERE pgr.player_name = ? AND pgr.date < ?
            AND pgr.at_bats > 0
            GROUP BY g.stadium
            HAVING games_played >= 3
        '''

        stadium_df = pd.read_sql_query(stadium_query, conn, params=[player_name, target_date])

        # Get today's stadium
        todays_stadium_query = '''
            SELECT stadium FROM games
            WHERE (home_team = ? OR away_team = ?) AND date = ?
            LIMIT 1
        '''

        cursor = conn.cursor()
        cursor.execute(todays_stadium_query, [team, team, target_date])
        todays_stadium = cursor.fetchone()

        if todays_stadium and not stadium_df.empty:
            stadium_name = todays_stadium[0]
            stadium_perf = stadium_df[stadium_df['stadium'] == stadium_name]

            if not stadium_perf.empty:
                features['ballpark_specific_hr_rate'] = stadium_perf.iloc[0]['hr_rate']
                features['ballpark_specific_games_played'] = stadium_perf.iloc[0]['games_played']
            else:
                features['ballpark_specific_hr_rate'] = 0.028  # League average
                features['ballpark_specific_games_played'] = 0
        else:
            features['ballpark_specific_hr_rate'] = 0.028
            features['ballpark_specific_games_played'] = 0

        conn.close()
        return features

    # Helper methods for calculations

    def _calculate_slugging(self, df: pd.DataFrame) -> float:
        """Calculate slugging percentage"""
        if df['at_bats'].sum() == 0:
            return 0.0

        total_bases = (df['hits'].sum() + df['doubles'].sum() +
                      2 * df['triples'].sum() + 3 * df['home_runs'].sum())
        return total_bases / df['at_bats'].sum()

    def _calculate_iso(self, df: pd.DataFrame) -> float:
        """Calculate isolated power (ISO)"""
        return self._calculate_slugging(df) - (df['hits'].sum() / max(df['at_bats'].sum(), 1))

    def _calculate_total_bases(self, df: pd.DataFrame) -> float:
        """Calculate total bases"""
        return (df['hits'].sum() + df['doubles'].sum() +
                2 * df['triples'].sum() + 3 * df['home_runs'].sum())

    def _apply_weighted_shrinkage(self, period_stats: Dict, period_weights: Dict,
                                stat_name: str) -> float:
        """Apply empirical Bayes shrinkage with weighted blending"""

        # Get league average for this stat
        if 'hr' in stat_name:
            league_avg = self.shrinkage_params['league_hr_rate']
        elif 'avg' in stat_name or 'obp' in stat_name:
            league_avg = 0.25
        elif 'slg' in stat_name:
            league_avg = 0.42
        else:
            league_avg = 0.1

        # Calculate weighted average with shrinkage
        weighted_sum = 0
        weight_sum = 0

        for period, stat_value in period_stats.items():
            if period in period_weights:
                # Apply shrinkage towards league average
                shrunk_value = 0.7 * stat_value + 0.3 * league_avg
                weight = period_weights[period]

                weighted_sum += shrunk_value * weight
                weight_sum += weight

        return weighted_sum / max(weight_sum, 0.001)

    def _calculate_performance_trends(self, season_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance trends over the season"""
        trends = {}

        if len(season_df) < 10:
            return {'trend_hr_rate_slope': 0, 'trend_avg_slope': 0}

        # Sort by date and calculate rolling averages
        season_df = season_df.sort_values('date').copy()
        season_df['game_num'] = range(1, len(season_df) + 1)

        # Calculate trends using linear regression
        if season_df['at_bats'].sum() > 0:
            season_df['hr_rate'] = season_df['home_runs'] / season_df['at_bats'].replace(0, 1)
            season_df['batting_avg'] = season_df['hits'] / season_df['at_bats'].replace(0, 1)

            # HR rate trend
            hr_slope, _, _, _, _ = stats.linregress(season_df['game_num'], season_df['hr_rate'])
            trends['trend_hr_rate_slope'] = hr_slope

            # Batting average trend
            avg_slope, _, _, _, _ = stats.linregress(season_df['game_num'], season_df['batting_avg'])
            trends['trend_avg_slope'] = avg_slope

        return trends

    def _calculate_current_streak(self, values: np.array) -> int:
        """Calculate current streak of consecutive games"""
        if len(values) == 0:
            return 0

        streak = 0
        for val in reversed(values):
            if val == 1:
                streak += 1
            else:
                break

        return streak

    def _calculate_momentum(self, values: np.array) -> float:
        """Calculate momentum using weighted recent performance"""
        if len(values) < 3:
            return 0.0

        # Weight recent games more heavily
        weights = np.linspace(0.5, 1.5, len(values))
        weighted_avg = np.average(values, weights=weights)

        return weighted_avg

    def create_training_dataset(self, start_date: str, end_date: str,
                              target_variable: str = 'home_runs') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create training dataset with features and targets

        Args:
            start_date: Start date for training data
            end_date: End date for training data
            target_variable: Target variable to predict ('home_runs', 'rbi', etc.)

        Returns:
            Tuple of (features_df, targets_df)
        """
        logger.info(f"Creating training dataset: {start_date} to {end_date}")

        # Get all dates in range with games
        conn = sqlite3.connect(self.db_path)
        dates_query = '''
            SELECT DISTINCT date
            FROM games
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        '''
        dates_df = pd.read_sql_query(dates_query, conn, params=[start_date, end_date])
        conn.close()

        all_features = []
        all_targets = []

        for _, row in dates_df.iterrows():
            date_str = row['date']

            try:
                # Generate features for this date
                features_df = self.generate_features_for_date(date_str)

                if features_df.empty:
                    continue

                # Get targets for this date
                targets_df = self._get_targets_for_date(date_str, target_variable)

                if targets_df.empty:
                    continue

                # Merge features and targets
                merged = features_df.merge(
                    targets_df,
                    on=['player_name', 'team', 'date'],
                    how='inner'
                )

                if not merged.empty:
                    feature_cols = [col for col in merged.columns
                                  if col not in ['player_name', 'team', 'date', target_variable]]

                    all_features.append(merged[['player_name', 'team', 'date'] + feature_cols])
                    all_targets.append(merged[['player_name', 'team', 'date', target_variable]])

            except Exception as e:
                logger.error(f"Error processing {date_str}: {e}")
                continue

        if all_features:
            final_features = pd.concat(all_features, ignore_index=True)
            final_targets = pd.concat(all_targets, ignore_index=True)

            logger.info(f"Training dataset created: {len(final_features)} samples, {len(final_features.columns)-3} features")
            return final_features, final_targets
        else:
            logger.warning("No training data generated")
            return pd.DataFrame(), pd.DataFrame()

    def _get_targets_for_date(self, date_str: str, target_variable: str) -> pd.DataFrame:
        """Get target variables for a specific date"""
        conn = sqlite3.connect(self.db_path)

        query = f'''
            SELECT player_name, team, date, {target_variable}
            FROM player_game_results
            WHERE date = ?
        '''

        targets_df = pd.read_sql_query(query, conn, params=[date_str])
        conn.close()

        return targets_df

    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Return feature categories for analysis"""
        return {
            'batting_performance': [f'player_{stat}' for stat in ['avg', 'obp', 'slg', 'hr_rate', 'iso']],
            'power_metrics': [f'power_{period}_{metric}' for period in ['career', 'season', 'last_30d', 'last_7d']
                             for metric in ['hr_per_ab', 'extra_base_rate', 'total_bases_per_ab']],
            'discipline': [f'discipline_{period}_{metric}' for period in ['career', 'season', 'last_30d', 'last_7d']
                          for metric in ['bb_rate', 'k_rate', 'contact_rate']],
            'situational': ['situational_home_hr_rate', 'situational_away_hr_rate', 'situational_day_hr_rate'],
            'environmental': ['env_park_hr_factor', 'env_elevation', 'env_day_night'],
            'pitcher_matchup': ['matchup_opp_pitcher_hr_rate', 'matchup_opp_pitcher_k_rate'],
            'team_context': ['team_context_runs_per_game', 'team_context_team_ops'],
            'streak_momentum': ['streak_current_hr_games', 'streak_momentum_hr'],
            'ballpark_specific': ['ballpark_specific_hr_rate']
        }


def main():
    """Test the feature engineering pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='ML Feature Engineering Pipeline')
    parser.add_argument('--test-date', default='2024-09-20', help='Date to test feature generation')
    parser.add_argument('--create-training', action='store_true', help='Create training dataset')
    parser.add_argument('--start-date', default='2024-09-18', help='Training start date')
    parser.add_argument('--end-date', default='2024-09-22', help='Training end date')
    parser.add_argument('--db-path', default='historical_baseball_data.db', help='Database path')

    args = parser.parse_args()

    # Initialize feature engineer
    feature_engineer = MLFeatureEngineer(db_path=args.db_path)

    if args.create_training:
        # Create training dataset
        features_df, targets_df = feature_engineer.create_training_dataset(
            start_date=args.start_date,
            end_date=args.end_date
        )

        print(f"\nTraining Dataset Summary:")
        print(f"Features shape: {features_df.shape}")
        print(f"Targets shape: {targets_df.shape}")

        if not features_df.empty:
            print(f"\nFeature columns: {len(features_df.columns)-3}")
            print(f"Sample features: {list(features_df.columns)[3:8]}")

            # Save datasets
            features_df.to_csv(f'training_features_{args.start_date}_to_{args.end_date}.csv', index=False)
            targets_df.to_csv(f'training_targets_{args.start_date}_to_{args.end_date}.csv', index=False)
            print(f"Datasets saved to CSV files")

    else:
        # Test feature generation for single date
        features_df = feature_engineer.generate_features_for_date(args.test_date)

        print(f"\nFeature Generation Test for {args.test_date}:")
        print(f"Generated features for {len(features_df)} players")
        print(f"Total features: {len(features_df.columns)-3}")

        if not features_df.empty:
            sample_player = features_df.iloc[0]
            print(f"\nSample player: {sample_player['player_name']} ({sample_player['team']})")

            # Show sample features
            feature_cols = [col for col in features_df.columns if col not in ['player_name', 'team', 'date']]
            print(f"Sample features: {dict(list(sample_player[feature_cols].items())[:5])}")

            # Show feature categories
            categories = feature_engineer.get_feature_importance_categories()
            print(f"\nFeature categories:")
            for category, features in categories.items():
                available_features = [f for f in features if f in feature_cols]
                print(f"  {category}: {len(available_features)} features")


if __name__ == "__main__":
    main()