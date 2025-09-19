-- =====================================================
-- RBI PREDICTION EVALUATION FRAMEWORK
-- =====================================================

-- 1. OVERALL MODEL PERFORMANCE METRICS
-- -----------------------------------------------------
CREATE VIEW model_performance_summary AS
WITH predictions AS (
    SELECT 
        run_id,
        model_version,
        COUNT(*) as total_predictions,
        SUM(got_rbi) as total_hits,
        AVG(model_prob) as avg_predicted_prob,
        AVG(CAST(got_rbi AS REAL)) as actual_hit_rate,
        
        -- Brier Score (lower is better)
        AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as brier_score,
        
        -- Log Loss (lower is better)
        -AVG(
            CASE 
                WHEN got_rbi = 1 THEN LOG(MAX(model_prob, 0.0001))
                ELSE LOG(1 - MIN(model_prob, 0.9999))
            END
        ) as log_loss,
        
        -- Expected vs Actual RBIs
        SUM(expected_rbi) as total_expected_rbi,
        SUM(rbi_count) as total_actual_rbi,
        
        DATE(MIN(date)) as start_date,
        DATE(MAX(date)) as end_date
    FROM rbi_predictions_log_v3
    GROUP BY run_id, model_version
)
SELECT 
    *,
    actual_hit_rate - avg_predicted_prob as calibration_diff,
    total_actual_rbi / NULLIF(total_expected_rbi, 0) as rbi_ratio
FROM predictions
ORDER BY run_id DESC;

-- 2. CALIBRATION ANALYSIS
-- -----------------------------------------------------
CREATE VIEW calibration_buckets AS
WITH bucketed AS (
    SELECT 
        run_id,
        CASE 
            WHEN model_prob < 0.1 THEN '0.00-0.10'
            WHEN model_prob < 0.2 THEN '0.10-0.20'
            WHEN model_prob < 0.3 THEN '0.20-0.30'
            WHEN model_prob < 0.4 THEN '0.30-0.40'
            WHEN model_prob < 0.5 THEN '0.40-0.50'
            WHEN model_prob < 0.6 THEN '0.50-0.60'
            WHEN model_prob < 0.7 THEN '0.60-0.70'
            WHEN model_prob < 0.8 THEN '0.70-0.80'
            WHEN model_prob < 0.9 THEN '0.80-0.90'
            ELSE '0.90-1.00'
        END as prob_bucket,
        model_prob,
        got_rbi,
        rbi_count
    FROM rbi_predictions_log_v3
)
SELECT 
    run_id,
    prob_bucket,
    COUNT(*) as count,
    AVG(model_prob) as avg_predicted_prob,
    AVG(CAST(got_rbi AS REAL)) as actual_hit_rate,
    AVG(model_prob) - AVG(CAST(got_rbi AS REAL)) as calibration_error,
    SUM(rbi_count) as total_rbis,
    AVG(rbi_count) as avg_rbis_when_occurs
FROM bucketed
GROUP BY run_id, prob_bucket
ORDER BY run_id DESC, prob_bucket;

-- 3. ROC CURVE DATA POINTS
-- -----------------------------------------------------
CREATE VIEW roc_curve_data AS
WITH thresholds AS (
    SELECT DISTINCT model_prob as threshold
    FROM rbi_predictions_log_v3
    WHERE model_prob IS NOT NULL
),
roc_points AS (
    SELECT 
        r.run_id,
        t.threshold,
        SUM(CASE WHEN r.model_prob >= t.threshold AND r.got_rbi = 1 THEN 1 ELSE 0 END) as true_positives,
        SUM(CASE WHEN r.model_prob >= t.threshold AND r.got_rbi = 0 THEN 1 ELSE 0 END) as false_positives,
        SUM(CASE WHEN r.model_prob < t.threshold AND r.got_rbi = 1 THEN 1 ELSE 0 END) as false_negatives,
        SUM(CASE WHEN r.model_prob < t.threshold AND r.got_rbi = 0 THEN 1 ELSE 0 END) as true_negatives,
        SUM(r.got_rbi) as total_positives,
        SUM(1 - r.got_rbi) as total_negatives
    FROM thresholds t
    CROSS JOIN rbi_predictions_log_v3 r
    GROUP BY r.run_id, t.threshold
)
SELECT 
    run_id,
    threshold,
    true_positives,
    false_positives,
    true_positives / NULLIF(CAST(total_positives AS REAL), 0) as tpr_recall,
    false_positives / NULLIF(CAST(total_negatives AS REAL), 0) as fpr,
    true_positives / NULLIF(CAST(true_positives + false_positives AS REAL), 0) as precision
FROM roc_points
ORDER BY run_id, threshold DESC;

-- 4. BETTING PERFORMANCE ANALYSIS
-- -----------------------------------------------------
CREATE VIEW betting_performance AS
WITH bet_results AS (
    SELECT 
        run_id,
        date,
        player_name,
        model_prob,
        market_implied_prob,
        value_edge,
        got_rbi,
        CASE 
            WHEN value_edge > 0.05 THEN 'BET'
            ELSE 'NO_BET'
        END as decision,
        CASE 
            WHEN value_edge > 0.05 AND got_rbi = 1 THEN (1.0 / market_implied_prob) - 1
            WHEN value_edge > 0.05 AND got_rbi = 0 THEN -1
            ELSE 0
        END as profit_units
    FROM rbi_predictions_log_v3
    WHERE market_implied_prob IS NOT NULL
)
SELECT 
    run_id,
    COUNT(CASE WHEN decision = 'BET' THEN 1 END) as total_bets,
    SUM(CASE WHEN decision = 'BET' AND got_rbi = 1 THEN 1 ELSE 0 END) as winning_bets,
    AVG(CASE WHEN decision = 'BET' THEN value_edge END) as avg_edge,
    SUM(profit_units) as total_profit_units,
    AVG(CASE WHEN decision = 'BET' THEN profit_units END) as roi_per_bet,
    SUM(profit_units) / NULLIF(COUNT(CASE WHEN decision = 'BET' THEN 1 END), 0) as roi_percentage
FROM bet_results
GROUP BY run_id;

-- 5. FEATURE IMPORTANCE TRACKING
-- -----------------------------------------------------
CREATE VIEW feature_importance_summary AS
SELECT 
    run_id,
    top_positive_feature,
    COUNT(*) as frequency,
    AVG(top_positive_value) as avg_shap_value,
    AVG(model_prob) as avg_prob_when_important,
    AVG(CAST(got_rbi AS REAL)) as hit_rate_when_important
FROM rbi_predictions_log_v3
WHERE top_positive_feature IS NOT NULL
GROUP BY run_id, top_positive_feature
ORDER BY run_id DESC, frequency DESC;

-- 6. PERFORMANCE BY SUBGROUPS
-- -----------------------------------------------------
CREATE VIEW subgroup_performance AS
SELECT 
    run_id,
    'Overall' as category,
    'All' as subcategory,
    COUNT(*) as count,
    AVG(model_prob) as avg_predicted_prob,
    AVG(CAST(got_rbi AS REAL)) as actual_hit_rate,
    AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as brier_score
FROM rbi_predictions_log_v3
GROUP BY run_id

UNION ALL

-- By lineup spot
SELECT 
    run_id,
    'Lineup Spot' as category,
    CAST(lineup_spot AS TEXT) as subcategory,
    COUNT(*) as count,
    AVG(model_prob) as avg_predicted_prob,
    AVG(CAST(got_rbi AS REAL)) as actual_hit_rate,
    AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as brier_score
FROM rbi_predictions_log_v3
GROUP BY run_id, lineup_spot

UNION ALL

-- By handedness
SELECT 
    run_id,
    'Batter Hand' as category,
    handedness as subcategory,
    COUNT(*) as count,
    AVG(model_prob) as avg_predicted_prob,
    AVG(CAST(got_rbi AS REAL)) as actual_hit_rate,
    AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as brier_score
FROM rbi_predictions_log_v3
GROUP BY run_id, handedness

UNION ALL

-- By venue type
SELECT 
    run_id,
    'Venue' as category,
    CASE WHEN venue = team THEN 'Home' ELSE 'Away' END as subcategory,
    COUNT(*) as count,
    AVG(model_prob) as avg_predicted_prob,
    AVG(CAST(got_rbi AS REAL)) as actual_hit_rate,
    AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as brier_score
FROM rbi_predictions_log_v3
GROUP BY run_id, CASE WHEN venue = team THEN 'Home' ELSE 'Away' END

UNION ALL

-- By park factor
SELECT 
    run_id,
    'Park Type' as category,
    CASE 
        WHEN park_run_factor < 0.95 THEN 'Pitcher Park'
        WHEN park_run_factor > 1.05 THEN 'Hitter Park'
        ELSE 'Neutral'
    END as subcategory,
    COUNT(*) as count,
    AVG(model_prob) as avg_predicted_prob,
    AVG(CAST(got_rbi AS REAL)) as actual_hit_rate,
    AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as brier_score
FROM rbi_predictions_log_v3
GROUP BY run_id, 
    CASE 
        WHEN park_run_factor < 0.95 THEN 'Pitcher Park'
        WHEN park_run_factor > 1.05 THEN 'Hitter Park'
        ELSE 'Neutral'
    END;

-- 7. TIME-BASED PERFORMANCE TRACKING
-- -----------------------------------------------------
CREATE VIEW performance_over_time AS
SELECT 
    run_id,
    DATE(date) as prediction_date,
    COUNT(*) as daily_predictions,
    AVG(model_prob) as avg_predicted_prob,
    AVG(CAST(got_rbi AS REAL)) as actual_hit_rate,
    AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as brier_score,
    SUM(CASE WHEN value_edge > 0.05 THEN 1 ELSE 0 END) as bets_placed,
    SUM(CASE 
        WHEN value_edge > 0.05 AND got_rbi = 1 THEN (1.0 / market_implied_prob) - 1
        WHEN value_edge > 0.05 AND got_rbi = 0 THEN -1
        ELSE 0
    END) as daily_profit_units
FROM rbi_predictions_log_v3
GROUP BY run_id, DATE(date)
ORDER BY run_id DESC, prediction_date DESC;

-- 8. PLAYER PERFORMANCE ANALYSIS
-- -----------------------------------------------------
CREATE VIEW player_model_performance AS
SELECT 
    run_id,
    player_id,
    player_name,
    COUNT(*) as predictions,
    AVG(model_prob) as avg_predicted_prob,
    AVG(CAST(got_rbi AS REAL)) as actual_hit_rate,
    SUM(got_rbi) as total_rbis_predicted,
    AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as brier_score,
    AVG(model_prob) - AVG(CAST(got_rbi AS REAL)) as calibration_error
FROM rbi_predictions_log_v3
GROUP BY run_id, player_id, player_name
HAVING COUNT(*) >= 10  -- Only players with sufficient sample
ORDER BY COUNT(*) DESC;

-- 9. EXTREME PREDICTIONS ANALYSIS
-- -----------------------------------------------------
CREATE VIEW extreme_predictions_audit AS
SELECT 
    run_id,
    date,
    player_name,
    model_prob,
    got_rbi,
    rbi_count,
    top_positive_feature,
    top_positive_value,
    season_rbi_per_pa,
    ops_30d,
    starter_era
FROM rbi_predictions_log_v3
WHERE model_prob > 0.7 OR model_prob < 0.1
ORDER BY ABS(model_prob - 0.5) DESC;

-- 10. MODEL COMPARISON (if multiple models)
-- -----------------------------------------------------
CREATE VIEW model_comparison AS
SELECT 
    model_version,
    COUNT(DISTINCT run_id) as num_runs,
    COUNT(*) as total_predictions,
    AVG(model_prob) as avg_predicted_prob,
    AVG(CAST(got_rbi AS REAL)) as actual_hit_rate,
    AVG((model_prob - got_rbi) * (model_prob - got_rbi)) as avg_brier_score,
    MIN((model_prob - got_rbi) * (model_prob - got_rbi)) as best_brier_score,
    AVG(ABS(model_prob - CAST(got_rbi AS REAL))) as avg_calibration_error
FROM rbi_predictions_log_v3
GROUP BY model_version
ORDER BY avg_brier_score;

-- =====================================================
-- UTILITY QUERIES FOR ANALYSIS
-- =====================================================

-- Get latest run performance summary
SELECT * FROM model_performance_summary 
WHERE run_id = (SELECT MAX(run_id) FROM rbi_predictions_log_v3);

-- Check calibration for latest run
SELECT * FROM calibration_buckets 
WHERE run_id = (SELECT MAX(run_id) FROM rbi_predictions_log_v3)
ORDER BY prob_bucket;

-- Get betting performance for profitable edge thresholds
WITH edge_analysis AS (
    SELECT 
        run_id,
        ROUND(value_edge * 20) / 20.0 as edge_threshold,
        COUNT(*) as bets,
        SUM(got_rbi) as wins,
        AVG(CAST(got_rbi AS REAL)) as win_rate,
        SUM(CASE 
            WHEN got_rbi = 1 THEN (1.0 / market_implied_prob) - 1
            ELSE -1
        END) as profit_units
    FROM rbi_predictions_log_v3
    WHERE value_edge >= 0
    GROUP BY run_id, ROUND(value_edge * 20) / 20.0
)
SELECT * FROM edge_analysis
ORDER BY run_id DESC, edge_threshold;