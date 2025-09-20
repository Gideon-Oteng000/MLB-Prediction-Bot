#!/usr/bin/env python3
"""
Enhanced Database Schema for MLB RBI Prediction System v4.0
Extensible, normalized design with comprehensive relationships
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DatabaseManagerV4:
    """Enhanced database manager with normalized schema"""

    def __init__(self, db_path: str = 'rbi_predictions_v4.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the complete v4 database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        # Core entities
        self._create_teams_table(cursor)
        self._create_players_table(cursor)
        self._create_venues_table(cursor)
        self._create_games_table(cursor)
        self._create_weather_table(cursor)

        # Player performance tracking
        self._create_player_stats_table(cursor)
        self._create_player_splits_table(cursor)
        self._create_performance_trends_table(cursor)

        # Pitching data
        self._create_pitchers_table(cursor)
        self._create_pitcher_stats_table(cursor)
        self._create_bullpen_metrics_table(cursor)

        # Market data
        self._create_sportsbooks_table(cursor)
        self._create_odds_table(cursor)
        self._create_line_movements_table(cursor)

        # Predictions and models
        self._create_models_table(cursor)
        self._create_predictions_table(cursor)
        self._create_feature_importance_table(cursor)
        self._create_shap_values_table(cursor)

        # Betting and performance
        self._create_bets_table(cursor)
        self._create_betting_performance_table(cursor)
        self._create_bankroll_history_table(cursor)

        # System monitoring
        self._create_api_usage_table(cursor)
        self._create_model_performance_table(cursor)
        self._create_data_quality_table(cursor)

        # Create indexes for performance
        self._create_indexes(cursor)

        # Create views for common queries
        self._create_views(cursor)

        conn.commit()
        conn.close()
        logger.info("Database v4.0 schema initialized successfully")

    def _create_teams_table(self, cursor):
        """Teams reference table"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                team_name TEXT NOT NULL UNIQUE,
                team_abbreviation TEXT NOT NULL UNIQUE,
                league TEXT NOT NULL,
                division TEXT NOT NULL,
                city TEXT NOT NULL,
                stadium_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    def _create_players_table(self, cursor):
        """Players reference table"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY,
                mlb_id INTEGER UNIQUE,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                full_name TEXT NOT NULL,
                team_id INTEGER,
                position TEXT,
                bats TEXT,
                throws TEXT,
                height_inches INTEGER,
                weight_lbs INTEGER,
                birth_date DATE,
                debut_date DATE,
                active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_id) REFERENCES teams(team_id)
            )
        ''')

    def _create_venues_table(self, cursor):
        """Venues reference table"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS venues (
                venue_id INTEGER PRIMARY KEY,
                venue_name TEXT NOT NULL UNIQUE,
                city TEXT NOT NULL,
                state TEXT,
                country TEXT DEFAULT 'USA',
                latitude REAL,
                longitude REAL,
                elevation_feet INTEGER,
                capacity INTEGER,
                surface_type TEXT,
                roof_type TEXT,
                park_factor REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    def _create_games_table(self, cursor):
        """Games table with comprehensive game information"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id INTEGER PRIMARY KEY,
                mlb_game_id INTEGER UNIQUE,
                game_date DATE NOT NULL,
                game_datetime TIMESTAMP,
                home_team_id INTEGER NOT NULL,
                away_team_id INTEGER NOT NULL,
                venue_id INTEGER NOT NULL,
                season INTEGER NOT NULL,
                game_type TEXT DEFAULT 'R',
                status TEXT DEFAULT 'Scheduled',
                inning INTEGER,
                top_bottom TEXT,
                home_score INTEGER DEFAULT 0,
                away_score INTEGER DEFAULT 0,
                attendance INTEGER,
                game_duration_minutes INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
                FOREIGN KEY (away_team_id) REFERENCES teams(team_id),
                FOREIGN KEY (venue_id) REFERENCES venues(venue_id)
            )
        ''')

    def _create_weather_table(self, cursor):
        """Weather conditions for games"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_conditions (
                weather_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                venue_id INTEGER NOT NULL,
                observation_time TIMESTAMP NOT NULL,
                temperature_f REAL,
                feels_like_f REAL,
                humidity_percent REAL,
                dewpoint_f REAL,
                pressure_mb REAL,
                visibility_miles REAL,
                wind_speed_mph REAL,
                wind_direction_degrees REAL,
                wind_direction_cardinal TEXT,
                uv_index REAL,
                weather_condition TEXT,
                precipitation_inches REAL,
                air_density_factor REAL,
                tailwind_component REAL,
                crosswind_component REAL,
                weather_severity_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games(game_id),
                FOREIGN KEY (venue_id) REFERENCES venues(venue_id)
            )
        ''')

    def _create_player_stats_table(self, cursor):
        """Player statistics by season"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_season_stats (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                season INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                games_played INTEGER DEFAULT 0,
                plate_appearances INTEGER DEFAULT 0,
                at_bats INTEGER DEFAULT 0,
                runs INTEGER DEFAULT 0,
                hits INTEGER DEFAULT 0,
                doubles INTEGER DEFAULT 0,
                triples INTEGER DEFAULT 0,
                home_runs INTEGER DEFAULT 0,
                rbi INTEGER DEFAULT 0,
                stolen_bases INTEGER DEFAULT 0,
                caught_stealing INTEGER DEFAULT 0,
                walks INTEGER DEFAULT 0,
                strikeouts INTEGER DEFAULT 0,
                hit_by_pitch INTEGER DEFAULT 0,
                sacrifice_flies INTEGER DEFAULT 0,
                sacrifice_hits INTEGER DEFAULT 0,
                gidp INTEGER DEFAULT 0,
                avg REAL GENERATED ALWAYS AS (CASE WHEN at_bats > 0 THEN CAST(hits AS REAL) / at_bats ELSE 0 END),
                obp REAL GENERATED ALWAYS AS (CASE WHEN (at_bats + walks + hit_by_pitch + sacrifice_flies) > 0 THEN CAST(hits + walks + hit_by_pitch AS REAL) / (at_bats + walks + hit_by_pitch + sacrifice_flies) ELSE 0 END),
                slg REAL GENERATED ALWAYS AS (CASE WHEN at_bats > 0 THEN CAST(hits + doubles + 2*triples + 3*home_runs AS REAL) / at_bats ELSE 0 END),
                ops REAL GENERATED ALWAYS AS (obp + slg),
                wrc_plus INTEGER,
                war REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(player_id),
                FOREIGN KEY (team_id) REFERENCES teams(team_id),
                UNIQUE(player_id, season, team_id)
            )
        ''')

    def _create_player_splits_table(self, cursor):
        """Player split statistics"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_splits (
                split_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                season INTEGER NOT NULL,
                split_type TEXT NOT NULL,
                split_value TEXT NOT NULL,
                plate_appearances INTEGER DEFAULT 0,
                at_bats INTEGER DEFAULT 0,
                hits INTEGER DEFAULT 0,
                doubles INTEGER DEFAULT 0,
                triples INTEGER DEFAULT 0,
                home_runs INTEGER DEFAULT 0,
                rbi INTEGER DEFAULT 0,
                walks INTEGER DEFAULT 0,
                strikeouts INTEGER DEFAULT 0,
                avg REAL,
                obp REAL,
                slg REAL,
                ops REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(player_id),
                UNIQUE(player_id, season, split_type, split_value)
            )
        ''')

    def _create_performance_trends_table(self, cursor):
        """Player performance trends over time"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_trends (
                trend_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                calculation_date DATE NOT NULL,
                period_days INTEGER NOT NULL,
                games_included INTEGER,
                trend_slope REAL,
                consistency_score REAL,
                recent_avg REAL,
                recent_obp REAL,
                recent_slg REAL,
                recent_rbi INTEGER,
                form_rating TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(player_id),
                UNIQUE(player_id, calculation_date, period_days)
            )
        ''')

    def _create_pitchers_table(self, cursor):
        """Pitcher information"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pitchers (
                pitcher_id INTEGER PRIMARY KEY,
                player_id INTEGER NOT NULL UNIQUE,
                role TEXT DEFAULT 'SP',
                handedness TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        ''')

    def _create_pitcher_stats_table(self, cursor):
        """Pitcher statistics"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pitcher_season_stats (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pitcher_id INTEGER NOT NULL,
                season INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                games INTEGER DEFAULT 0,
                games_started INTEGER DEFAULT 0,
                complete_games INTEGER DEFAULT 0,
                shutouts INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                saves INTEGER DEFAULT 0,
                innings_pitched REAL DEFAULT 0,
                hits_allowed INTEGER DEFAULT 0,
                runs_allowed INTEGER DEFAULT 0,
                earned_runs INTEGER DEFAULT 0,
                home_runs_allowed INTEGER DEFAULT 0,
                walks_allowed INTEGER DEFAULT 0,
                strikeouts INTEGER DEFAULT 0,
                hit_batters INTEGER DEFAULT 0,
                balks INTEGER DEFAULT 0,
                wild_pitches INTEGER DEFAULT 0,
                era REAL GENERATED ALWAYS AS (CASE WHEN innings_pitched > 0 THEN (earned_runs * 9.0) / innings_pitched ELSE 0 END),
                whip REAL GENERATED ALWAYS AS (CASE WHEN innings_pitched > 0 THEN CAST(hits_allowed + walks_allowed AS REAL) / innings_pitched ELSE 0 END),
                k_per_9 REAL GENERATED ALWAYS AS (CASE WHEN innings_pitched > 0 THEN (strikeouts * 9.0) / innings_pitched ELSE 0 END),
                bb_per_9 REAL GENERATED ALWAYS AS (CASE WHEN innings_pitched > 0 THEN (walks_allowed * 9.0) / innings_pitched ELSE 0 END),
                hr_per_9 REAL GENERATED ALWAYS AS (CASE WHEN innings_pitched > 0 THEN (home_runs_allowed * 9.0) / innings_pitched ELSE 0 END),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pitcher_id) REFERENCES pitchers(pitcher_id),
                FOREIGN KEY (team_id) REFERENCES teams(team_id),
                UNIQUE(pitcher_id, season, team_id)
            )
        ''')

    def _create_bullpen_metrics_table(self, cursor):
        """Bullpen usage and performance metrics"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bullpen_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL,
                game_id INTEGER,
                date DATE NOT NULL,
                starter_id INTEGER,
                starter_expected_ip REAL,
                bullpen_era REAL,
                bullpen_whip REAL,
                high_leverage_era REAL,
                medium_leverage_era REAL,
                low_leverage_era REAL,
                closer_usage_prob REAL,
                setup_usage_prob REAL,
                long_relief_usage_prob REAL,
                risp_era REAL,
                bases_loaded_era REAL,
                leverage_index_avg REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_id) REFERENCES teams(team_id),
                FOREIGN KEY (game_id) REFERENCES games(game_id),
                FOREIGN KEY (starter_id) REFERENCES pitchers(pitcher_id)
            )
        ''')

    def _create_sportsbooks_table(self, cursor):
        """Sportsbooks reference table"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sportsbooks (
                sportsbook_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                region TEXT DEFAULT 'US',
                active BOOLEAN DEFAULT 1,
                vig_average REAL,
                reliability_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    def _create_odds_table(self, cursor):
        """Betting odds data"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS odds (
                odds_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sportsbook_id INTEGER NOT NULL,
                player_id INTEGER NOT NULL,
                game_id INTEGER NOT NULL,
                market_type TEXT NOT NULL,
                line_value REAL,
                over_odds INTEGER,
                under_odds INTEGER,
                over_prob_raw REAL,
                under_prob_raw REAL,
                over_prob_true REAL,
                under_prob_true REAL,
                vig_percentage REAL,
                market_efficiency REAL,
                volume_indicator TEXT,
                sharp_money_direction TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sportsbook_id) REFERENCES sportsbooks(sportsbook_id),
                FOREIGN KEY (player_id) REFERENCES players(player_id),
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        ''')

    def _create_line_movements_table(self, cursor):
        """Line movement tracking"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS line_movements (
                movement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                odds_id INTEGER NOT NULL,
                previous_line REAL,
                new_line REAL,
                previous_odds INTEGER,
                new_odds INTEGER,
                movement_size REAL,
                movement_direction TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (odds_id) REFERENCES odds(odds_id)
            )
        ''')

    def _create_models_table(self, cursor):
        """ML models tracking"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                version TEXT NOT NULL,
                algorithm TEXT,
                hyperparameters TEXT,
                training_data_size INTEGER,
                training_date TIMESTAMP,
                validation_score REAL,
                test_score REAL,
                feature_count INTEGER,
                model_file_path TEXT,
                active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_name, version)
            )
        ''')

    def _create_predictions_table(self, cursor):
        """Predictions with comprehensive tracking"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions_v4 (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                game_id INTEGER NOT NULL,
                model_id INTEGER NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- Core predictions
                rbi_probability REAL NOT NULL,
                expected_rbis REAL NOT NULL,
                rbi_distribution TEXT,
                confidence_score REAL,

                -- Model ensemble results
                xgboost_prediction REAL,
                lightgbm_prediction REAL,
                random_forest_prediction REAL,
                lstm_prediction REAL,
                attention_prediction REAL,
                poisson_prediction REAL,

                -- Context features
                batting_order INTEGER,
                opposing_pitcher_id INTEGER,
                weather_id INTEGER,

                -- Market comparison
                market_line REAL,
                market_odds INTEGER,
                implied_probability REAL,
                value_edge REAL,

                -- Recommendation
                recommendation TEXT,
                bet_size_kelly REAL,

                -- Outcome tracking
                actual_rbis INTEGER,
                prediction_correct BOOLEAN,
                absolute_error REAL,
                squared_error REAL,

                FOREIGN KEY (player_id) REFERENCES players(player_id),
                FOREIGN KEY (game_id) REFERENCES games(game_id),
                FOREIGN KEY (model_id) REFERENCES models(model_id),
                FOREIGN KEY (opposing_pitcher_id) REFERENCES pitchers(pitcher_id),
                FOREIGN KEY (weather_id) REFERENCES weather_conditions(weather_id)
            )
        ''')

    def _create_feature_importance_table(self, cursor):
        """Feature importance tracking"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                importance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                feature_name TEXT NOT NULL,
                importance_value REAL NOT NULL,
                importance_rank INTEGER,
                calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(model_id),
                UNIQUE(model_id, feature_name, calculation_date)
            )
        ''')

    def _create_shap_values_table(self, cursor):
        """SHAP values for explainability"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shap_values (
                shap_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                feature_name TEXT NOT NULL,
                shap_value REAL NOT NULL,
                feature_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions_v4(prediction_id),
                UNIQUE(prediction_id, feature_name)
            )
        ''')

    def _create_bets_table(self, cursor):
        """Betting activity tracking"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bets (
                bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                sportsbook_id INTEGER NOT NULL,
                bet_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                bet_amount REAL NOT NULL,
                bet_fraction REAL,
                odds INTEGER NOT NULL,
                bet_type TEXT DEFAULT 'over',
                line_value REAL,
                bankroll_before REAL,
                bankroll_after REAL,
                won BOOLEAN,
                profit REAL,
                settled_date TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions_v4(prediction_id),
                FOREIGN KEY (sportsbook_id) REFERENCES sportsbooks(sportsbook_id)
            )
        ''')

    def _create_betting_performance_table(self, cursor):
        """Betting performance metrics"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS betting_performance_v4 (
                performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                calculation_date DATE NOT NULL UNIQUE,
                period_days INTEGER NOT NULL,
                total_bets INTEGER DEFAULT 0,
                winning_bets INTEGER DEFAULT 0,
                losing_bets INTEGER DEFAULT 0,
                win_rate REAL,
                total_wagered REAL DEFAULT 0,
                total_profit REAL DEFAULT 0,
                roi REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                avg_bet_size REAL,
                avg_odds INTEGER,
                kelly_efficiency REAL,
                confidence_calibration REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    def _create_bankroll_history_table(self, cursor):
        """Bankroll tracking over time"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bankroll_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                starting_bankroll REAL NOT NULL,
                ending_bankroll REAL NOT NULL,
                daily_profit REAL,
                daily_roi REAL,
                bets_placed INTEGER DEFAULT 0,
                largest_bet REAL,
                largest_win REAL,
                largest_loss REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
        ''')

    def _create_api_usage_table(self, cursor):
        """API usage monitoring"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage (
                usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_name TEXT NOT NULL,
                endpoint TEXT,
                request_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_code INTEGER,
                response_time_ms INTEGER,
                data_quality_score REAL,
                cache_hit BOOLEAN DEFAULT 0,
                error_message TEXT
            )
        ''')

    def _create_model_performance_table(self, cursor):
        """Model performance tracking"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance_v4 (
                performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                evaluation_date DATE NOT NULL,
                evaluation_period_days INTEGER,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                mae REAL,
                mse REAL,
                rmse REAL,
                r2_score REAL,
                log_loss REAL,
                calibration_score REAL,
                feature_drift_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(model_id),
                UNIQUE(model_id, evaluation_date)
            )
        ''')

    def _create_data_quality_table(self, cursor):
        """Data quality monitoring"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_quality (
                quality_id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                check_date DATE NOT NULL,
                total_records INTEGER,
                null_records INTEGER,
                duplicate_records INTEGER,
                outlier_records INTEGER,
                completeness_score REAL,
                consistency_score REAL,
                validity_score REAL,
                overall_quality_score REAL,
                issues_detected TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(table_name, check_date)
            )
        ''')

    def _create_indexes(self, cursor):
        """Create performance indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)",
            "CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team_id, away_team_id)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_player_game ON predictions_v4(player_id, game_id)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions_v4(prediction_date)",
            "CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(bet_date)",
            "CREATE INDEX IF NOT EXISTS idx_player_stats_season ON player_season_stats(player_id, season)",
            "CREATE INDEX IF NOT EXISTS idx_odds_game_player ON odds(game_id, player_id)",
            "CREATE INDEX IF NOT EXISTS idx_weather_game ON weather_conditions(game_id)",
            "CREATE INDEX IF NOT EXISTS idx_shap_prediction ON shap_values(prediction_id)",
            "CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(request_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_performance_trends_player ON performance_trends(player_id, calculation_date)"
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

    def _create_views(self, cursor):
        """Create useful database views"""

        # Player performance summary view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS v_player_performance_summary AS
            SELECT
                p.player_id,
                p.full_name,
                p.position,
                t.team_name,
                ps.season,
                ps.games_played,
                ps.avg,
                ps.obp,
                ps.slg,
                ps.ops,
                ps.rbi,
                ps.home_runs,
                pt.trend_slope,
                pt.consistency_score,
                pt.form_rating
            FROM players p
            JOIN player_season_stats ps ON p.player_id = ps.player_id
            JOIN teams t ON ps.team_id = t.team_id
            LEFT JOIN performance_trends pt ON p.player_id = pt.player_id
                AND pt.period_days = 30
                AND pt.calculation_date = (
                    SELECT MAX(calculation_date)
                    FROM performance_trends pt2
                    WHERE pt2.player_id = p.player_id
                )
        ''')

        # Recent predictions with outcomes view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS v_recent_predictions_with_outcomes AS
            SELECT
                pr.prediction_id,
                pl.full_name as player_name,
                t1.team_name as player_team,
                t2.team_name as opponent_team,
                g.game_date,
                pr.rbi_probability,
                pr.expected_rbis,
                pr.confidence_score,
                pr.recommendation,
                pr.actual_rbis,
                pr.prediction_correct,
                pr.absolute_error,
                b.bet_amount,
                b.profit,
                b.won as bet_won
            FROM predictions_v4 pr
            JOIN players pl ON pr.player_id = pl.player_id
            JOIN games g ON pr.game_id = g.game_id
            JOIN teams t1 ON pl.team_id = t1.team_id
            JOIN teams t2 ON (CASE WHEN g.home_team_id = t1.team_id THEN g.away_team_id ELSE g.home_team_id END) = t2.team_id
            LEFT JOIN bets b ON pr.prediction_id = b.prediction_id
            WHERE pr.prediction_date >= date('now', '-30 days')
            ORDER BY pr.prediction_date DESC
        ''')

        # Model performance comparison view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS v_model_performance_comparison AS
            SELECT
                m.model_name,
                m.model_type,
                mp.evaluation_date,
                mp.accuracy,
                mp.mae,
                mp.rmse,
                mp.calibration_score,
                COUNT(pr.prediction_id) as recent_predictions
            FROM models m
            JOIN model_performance_v4 mp ON m.model_id = mp.model_id
            LEFT JOIN predictions_v4 pr ON m.model_id = pr.model_id
                AND pr.prediction_date >= date('now', '-7 days')
            WHERE mp.evaluation_date = (
                SELECT MAX(evaluation_date)
                FROM model_performance_v4 mp2
                WHERE mp2.model_id = m.model_id
            )
            GROUP BY m.model_id, m.model_name, m.model_type, mp.evaluation_date,
                     mp.accuracy, mp.mae, mp.rmse, mp.calibration_score
        ''')

        # Betting ROI by period view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS v_betting_roi_analysis AS
            SELECT
                DATE(b.bet_date) as bet_date,
                COUNT(*) as daily_bets,
                SUM(b.bet_amount) as daily_wagered,
                SUM(b.profit) as daily_profit,
                AVG(CASE WHEN b.won IS NOT NULL THEN CAST(b.won AS REAL) ELSE NULL END) as win_rate,
                SUM(b.profit) / SUM(b.bet_amount) as daily_roi
            FROM bets b
            WHERE b.settled_date IS NOT NULL
            GROUP BY DATE(b.bet_date)
            ORDER BY bet_date DESC
        ''')

    def get_schema_info(self) -> Dict:
        """Get database schema information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get table information
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Get view information
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = [row[0] for row in cursor.fetchall()]

        # Get index information
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]

        # Get record counts
        record_counts = {}
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                record_counts[table] = cursor.fetchone()[0]
            except:
                record_counts[table] = 0

        conn.close()

        return {
            'tables': tables,
            'views': views,
            'indexes': indexes,
            'record_counts': record_counts,
            'total_tables': len(tables),
            'total_views': len(views),
            'total_indexes': len(indexes)
        }

if __name__ == "__main__":
    # Initialize database
    db_manager = DatabaseManagerV4()

    # Display schema information
    schema_info = db_manager.get_schema_info()

    print("🗄️ MLB RBI Prediction System v4.0 Database Schema")
    print("=" * 60)
    print(f"Tables: {schema_info['total_tables']}")
    print(f"Views: {schema_info['total_views']}")
    print(f"Indexes: {schema_info['total_indexes']}")
    print()

    print("📊 Table Record Counts:")
    for table, count in schema_info['record_counts'].items():
        print(f"  {table}: {count:,}")

    print("\n✅ Database v4.0 initialized successfully!")