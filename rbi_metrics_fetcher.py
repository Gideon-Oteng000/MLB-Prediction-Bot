#!/usr/bin/env python3
"""
RBI Metrics Fetcher
Fetches and blends advanced metrics for RBI prediction modeling
Integrates with mlb_lineups_fetcher.py and weather_ballpark_fetcher.py
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import os
import sys
import time
import sqlite3

warnings.filterwarnings('ignore')

try:
    import pybaseball as pyb
    pyb.cache.enable()
except ImportError:
    print("[ERROR] pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)


class RBIMetricsFetcher:
    """
    Fetches and blends RBI-relevant metrics for batters and pitchers
    """

    def __init__(self):
        self.today = datetime.now().date()
        self.current_year = datetime.now().year

        # Time windows for metric aggregation
        self.season_start = f"{self.current_year}-03-01"
        self.last_30_start = (self.today - timedelta(days=30)).strftime('%Y-%m-%d')
        self.last_7_start = (self.today - timedelta(days=7)).strftime('%Y-%m-%d')
        self.today_str = self.today.strftime('%Y-%m-%d')

        # Blending weights: [Season, Last30, Last7]
        self.weights = [0.6, 0.3, 0.1]

        # League average defaults (for missing values)
        self.league_defaults = {
            'avg': 0.248,
            'obp': 0.315,
            'slg': 0.411,
            'iso': 0.163,
            'babip': 0.295,
            'woba': 0.315,
            'hard_hit_pct': 37.5,
            'contact_pct': 76.0,
            'barrel_pct': 7.5,
            'avg_launch_speed': 88.5,
            'avg_launch_angle': 12.0,
            'gb_pct': 44.0,
            'fb_pct': 35.0,
            'ld_pct': 21.0,
            'rbi_rate': 0.12
        }

        # Cache database
        self.cache_db = "rbi_metrics_cache.db"
        self._init_cache()

        print("[INFO] RBI Metrics Fetcher initialized")
        print(f"[INFO] Season: {self.season_start} to {self.today_str}")

    def _init_cache(self):
        """Initialize SQLite cache database"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS player_cache (
                    player_id INTEGER,
                    player_name TEXT,
                    date TEXT,
                    timeframe TEXT,
                    metrics TEXT,
                    PRIMARY KEY (player_id, date, timeframe)
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[WARN] Cache initialization failed: {e}")

    def _get_cached_metrics(self, player_id: int, date: str, timeframe: str) -> Optional[Dict]:
        """Retrieve cached metrics"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT metrics FROM player_cache WHERE player_id=? AND date=? AND timeframe=?',
                (player_id, date, timeframe)
            )
            result = cursor.fetchone()
            conn.close()

            if result:
                return json.loads(result[0])
        except:
            pass
        return None

    def _cache_metrics(self, player_id: int, player_name: str, date: str, timeframe: str, metrics: Dict):
        """Store metrics in cache"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR REPLACE INTO player_cache VALUES (?, ?, ?, ?, ?)',
                (player_id, player_name, date, timeframe, json.dumps(metrics))
            )
            conn.commit()
            conn.close()
        except:
            pass

    def fetch_batter_metrics(self, player_id: int, player_name: str) -> Dict:
        """
        Fetch RBI-relevant batter metrics across all time windows
        """
        print(f"[INFO] Fetching batter metrics: {player_name}")

        metrics = {
            'player_id': player_id,
            'player_name': player_name,
            'season': {},
            'last_30': {},
            'last_7': {},
            'blended_metrics': {},
            'sample_sizes': {}
        }

        # Fetch data for each timeframe
        timeframes = {
            'season': (self.season_start, self.today_str),
            'last_30': (self.last_30_start, self.today_str),
            'last_7': (self.last_7_start, self.today_str)
        }

        for timeframe, (start_dt, end_dt) in timeframes.items():
            # Check cache first
            cached = self._get_cached_metrics(player_id, self.today_str, timeframe)
            if cached:
                metrics[timeframe] = cached
                print(f"  [CACHE] Loaded {timeframe} from cache")
                continue

            try:
                # Fetch Statcast data
                statcast_data = pyb.statcast_batter(
                    start_dt=start_dt,
                    end_dt=end_dt,
                    player_id=player_id
                )

                if statcast_data is not None and not statcast_data.empty:
                    calculated = self._calculate_batter_metrics(statcast_data)
                    metrics[timeframe] = calculated
                    metrics['sample_sizes'][f'{timeframe}_pa'] = len(statcast_data)

                    # Cache results
                    self._cache_metrics(player_id, player_name, self.today_str, timeframe, calculated)
                    print(f"  [FETCH] {timeframe}: {len(statcast_data)} PA")
                else:
                    metrics[timeframe] = {}
                    metrics['sample_sizes'][f'{timeframe}_pa'] = 0
                    print(f"  [WARN] No data for {timeframe}")

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                print(f"  [ERROR] Failed to fetch {timeframe}: {e}")
                metrics[timeframe] = {}
                metrics['sample_sizes'][f'{timeframe}_pa'] = 0

        # Blend metrics
        metrics['blended_metrics'] = self.blend_batter_metrics(
            metrics['season'],
            metrics['last_30'],
            metrics['last_7']
        )

        return metrics

    def _calculate_batter_metrics(self, statcast_data: pd.DataFrame) -> Dict:
        """
        Calculate RBI-relevant metrics from Statcast data
        """
        metrics = {}

        try:
            # Batted ball metrics
            batted_balls = statcast_data[statcast_data['type'] == 'X']

            if not batted_balls.empty:
                # Launch metrics
                metrics['avg_launch_speed'] = float(batted_balls['launch_speed'].mean())
                metrics['max_launch_speed'] = float(batted_balls['launch_speed'].max())
                metrics['avg_launch_angle'] = float(batted_balls['launch_angle'].mean())

                # Barrel and hard-hit
                barrels = (batted_balls['launch_speed'] >= 98) & \
                         (batted_balls['launch_angle'].between(26, 30))
                metrics['barrel_pct'] = float(barrels.mean() * 100)

                hard_hits = batted_balls['launch_speed'] >= 95
                metrics['hard_hit_pct'] = float(hard_hits.mean() * 100)

                # Sweet spot (8-32 degrees)
                sweet_spot = batted_balls['launch_angle'].between(8, 32)
                metrics['sweet_spot_pct'] = float(sweet_spot.mean() * 100)

                # Batted ball type percentages
                if 'bb_type' in batted_balls.columns:
                    bb_counts = batted_balls['bb_type'].value_counts()
                    total_bb = len(batted_balls)

                    metrics['gb_pct'] = float((bb_counts.get('ground_ball', 0) / total_bb) * 100)
                    metrics['fb_pct'] = float((bb_counts.get('fly_ball', 0) / total_bb) * 100)
                    metrics['ld_pct'] = float((bb_counts.get('line_drive', 0) / total_bb) * 100)

                # Hit distance (RBI-relevant for XBH)
                if 'hit_distance_sc' in batted_balls.columns:
                    metrics['avg_hit_distance'] = float(batted_balls['hit_distance_sc'].mean())

                # Expected metrics
                if 'estimated_woba_using_speedangle' in batted_balls.columns:
                    metrics['xwoba'] = float(batted_balls['estimated_woba_using_speedangle'].mean())

            # Plate discipline & outcomes
            plate_appearances = statcast_data.groupby(['game_date', 'at_bat_number']).size()
            pa_count = len(plate_appearances)

            if pa_count > 0:
                # RBI rate (critical metric)
                rbis = statcast_data['events'].isin(['field_out', 'sac_fly', 'single', 'double',
                                                      'triple', 'home_run'])  # Simplified - actual RBI tracking would need game logs
                # Note: Statcast doesn't directly have RBIs, would need to join with game logs
                # For now, use contact/power as proxy

                # K and BB rates
                strikeouts = statcast_data['events'].isin(['strikeout', 'strikeout_double_play']).sum()
                walks = statcast_data['events'].eq('walk').sum()

                metrics['k_pct'] = float((strikeouts / pa_count) * 100)
                metrics['bb_pct'] = float((walks / pa_count) * 100)

                # Contact rate (crucial for RBI)
                swings = statcast_data['description'].isin(['foul', 'hit_into_play',
                                                            'swinging_strike', 'foul_tip']).sum()
                contact = statcast_data['description'].isin(['foul', 'hit_into_play',
                                                             'foul_tip']).sum()
                if swings > 0:
                    metrics['contact_pct'] = float((contact / swings) * 100)

                # Batting outcomes
                hits = statcast_data['events'].isin(['single', 'double', 'triple', 'home_run']).sum()
                abs = statcast_data['events'].isin(['single', 'double', 'triple', 'home_run',
                                                    'field_out', 'grounded_into_double_play',
                                                    'fielders_choice_out']).sum()

                if abs > 0:
                    metrics['avg'] = float(hits / abs)

                    # Total bases for SLG
                    total_bases = (
                        statcast_data['events'].eq('single').sum() +
                        statcast_data['events'].eq('double').sum() * 2 +
                        statcast_data['events'].eq('triple').sum() * 3 +
                        statcast_data['events'].eq('home_run').sum() * 4
                    )
                    metrics['slg'] = float(total_bases / abs)
                    metrics['iso'] = float(metrics['slg'] - metrics['avg'])

                # OBP
                if pa_count > 0:
                    on_base = hits + walks
                    metrics['obp'] = float(on_base / pa_count)

                # wOBA (simplified calculation)
                if 'woba_value' in statcast_data.columns:
                    woba_events = statcast_data[statcast_data['woba_value'].notna()]
                    if not woba_events.empty:
                        metrics['woba'] = float(woba_events['woba_value'].mean())

                # BABIP
                balls_in_play = statcast_data['events'].isin(['single', 'double', 'triple',
                                                              'field_out', 'grounded_into_double_play']).sum()
                babip_hits = statcast_data['events'].isin(['single', 'double', 'triple']).sum()
                if balls_in_play > 0:
                    metrics['babip'] = float(babip_hits / balls_in_play)

            # === RISP (Runners In Scoring Position) PERFORMANCE ===
            # Critical RBI metric - measures clutch hitting ability
            # RISP = runners on 2nd and/or 3rd base
            if 'on_2b' in statcast_data.columns and 'on_3b' in statcast_data.columns:
                # Filter for RISP situations (runner on 2nd or 3rd)
                risp_situations = statcast_data[
                    (statcast_data['on_2b'].notna()) | (statcast_data['on_3b'].notna())
                ]

                if not risp_situations.empty:
                    # RISP PAs
                    risp_pa = risp_situations.groupby(['game_date', 'at_bat_number']).size()
                    risp_pa_count = len(risp_pa)

                    if risp_pa_count > 0:
                        # RISP batting outcomes
                        risp_hits = risp_situations['events'].isin(['single', 'double', 'triple', 'home_run']).sum()
                        risp_abs = risp_situations['events'].isin(['single', 'double', 'triple', 'home_run',
                                                                    'field_out', 'grounded_into_double_play',
                                                                    'fielders_choice_out']).sum()

                        if risp_abs > 0:
                            metrics['risp_avg'] = float(risp_hits / risp_abs)

                            # RISP total bases for SLG
                            risp_total_bases = (
                                risp_situations['events'].eq('single').sum() +
                                risp_situations['events'].eq('double').sum() * 2 +
                                risp_situations['events'].eq('triple').sum() * 3 +
                                risp_situations['events'].eq('home_run').sum() * 4
                            )
                            metrics['risp_slg'] = float(risp_total_bases / risp_abs)
                        else:
                            # Not enough RISP ABs, use regular stats
                            metrics['risp_avg'] = metrics.get('avg', 0.248)
                            metrics['risp_slg'] = metrics.get('slg', 0.411)

                        # RISP OBP
                        risp_walks = risp_situations['events'].eq('walk').sum()
                        risp_on_base = risp_hits + risp_walks
                        metrics['risp_obp'] = float(risp_on_base / risp_pa_count)

                        # RISP contact rate
                        risp_swings = risp_situations['description'].isin(['foul', 'hit_into_play',
                                                                           'swinging_strike', 'foul_tip']).sum()
                        risp_contact = risp_situations['description'].isin(['foul', 'hit_into_play',
                                                                            'foul_tip']).sum()
                        if risp_swings > 0:
                            metrics['risp_contact_pct'] = float((risp_contact / risp_swings) * 100)

                        # Sample size indicator
                        metrics['risp_pa_count'] = risp_pa_count
                    else:
                        # No RISP situations - use overall stats as fallback
                        metrics['risp_avg'] = metrics.get('avg', 0.248)
                        metrics['risp_slg'] = metrics.get('slg', 0.411)
                        metrics['risp_obp'] = metrics.get('obp', 0.315)
                        metrics['risp_contact_pct'] = metrics.get('contact_pct', 76.0)
                        metrics['risp_pa_count'] = 0
                else:
                    # No RISP data available - use overall stats
                    metrics['risp_avg'] = metrics.get('avg', 0.248)
                    metrics['risp_slg'] = metrics.get('slg', 0.411)
                    metrics['risp_obp'] = metrics.get('obp', 0.315)
                    metrics['risp_contact_pct'] = metrics.get('contact_pct', 76.0)
                    metrics['risp_pa_count'] = 0
            else:
                # Statcast data doesn't have baserunner info - use overall stats
                metrics['risp_avg'] = metrics.get('avg', 0.248)
                metrics['risp_slg'] = metrics.get('slg', 0.411)
                metrics['risp_obp'] = metrics.get('obp', 0.315)
                metrics['risp_contact_pct'] = metrics.get('contact_pct', 76.0)
                metrics['risp_pa_count'] = 0

        except Exception as e:
            print(f"    [ERROR] Metric calculation error: {e}")

        return metrics

    def fetch_pitcher_metrics(self, player_id: int, player_name: str) -> Dict:
        """
        Fetch RBI-prevention pitcher metrics across all time windows
        """
        print(f"[INFO] Fetching pitcher metrics: {player_name}")

        metrics = {
            'player_id': player_id,
            'player_name': player_name,
            'season': {},
            'last_30': {},
            'last_7': {},
            'blended_metrics': {},
            'sample_sizes': {}
        }

        # Fetch data for each timeframe
        timeframes = {
            'season': (self.season_start, self.today_str),
            'last_30': (self.last_30_start, self.today_str),
            'last_7': (self.last_7_start, self.today_str)
        }

        for timeframe, (start_dt, end_dt) in timeframes.items():
            # Check cache
            cached = self._get_cached_metrics(player_id, self.today_str, f"pitcher_{timeframe}")
            if cached:
                metrics[timeframe] = cached
                print(f"  [CACHE] Loaded {timeframe} from cache")
                continue

            try:
                # Fetch Statcast pitcher data
                statcast_data = pyb.statcast_pitcher(
                    start_dt=start_dt,
                    end_dt=end_dt,
                    player_id=player_id
                )

                if statcast_data is not None and not statcast_data.empty:
                    calculated = self._calculate_pitcher_metrics(statcast_data)
                    metrics[timeframe] = calculated
                    metrics['sample_sizes'][f'{timeframe}_bf'] = len(statcast_data)

                    # Cache
                    self._cache_metrics(player_id, player_name, self.today_str,
                                      f"pitcher_{timeframe}", calculated)
                    print(f"  [FETCH] {timeframe}: {len(statcast_data)} BF")
                else:
                    metrics[timeframe] = {}
                    metrics['sample_sizes'][f'{timeframe}_bf'] = 0
                    print(f"  [WARN] No pitcher data for {timeframe}")

                time.sleep(0.1)

            except Exception as e:
                print(f"  [ERROR] Failed to fetch {timeframe}: {e}")
                metrics[timeframe] = {}
                metrics['sample_sizes'][f'{timeframe}_bf'] = 0

        # Blend metrics
        metrics['blended_metrics'] = self.blend_pitcher_metrics(
            metrics['season'],
            metrics['last_30'],
            metrics['last_7']
        )

        return metrics

    def _calculate_pitcher_metrics(self, statcast_data: pd.DataFrame) -> Dict:
        """
        Calculate RBI-prevention metrics from pitcher Statcast data
        Enhanced with LOB%, platoon splits, and Barrel% allowed
        """
        metrics = {}

        try:
            # Batted balls allowed
            batted_balls = statcast_data[statcast_data['type'] == 'X']

            if not batted_balls.empty:
                # Launch metrics allowed
                metrics['avg_launch_speed_allowed'] = float(batted_balls['launch_speed'].mean())
                metrics['avg_launch_angle_allowed'] = float(batted_balls['launch_angle'].mean())

                # Barrel allowed (critical for hard contact/damage)
                barrels = (batted_balls['launch_speed'] >= 98) & \
                         (batted_balls['launch_angle'].between(26, 30))
                metrics['barrel_pct_allowed'] = float(barrels.mean() * 100)

                hard_hits = batted_balls['launch_speed'] >= 95
                metrics['hard_hit_pct_allowed'] = float(hard_hits.mean() * 100)

                # Batted ball types
                if 'bb_type' in batted_balls.columns:
                    bb_counts = batted_balls['bb_type'].value_counts()
                    total_bb = len(batted_balls)

                    metrics['gb_pct'] = float((bb_counts.get('ground_ball', 0) / total_bb) * 100)
                    metrics['fb_pct'] = float((bb_counts.get('fly_ball', 0) / total_bb) * 100)
                    metrics['ld_pct'] = float((bb_counts.get('line_drive', 0) / total_bb) * 100)

                # GB/FB ratio (high GB% prevents RBIs)
                gb_count = bb_counts.get('ground_ball', 0)
                fb_count = bb_counts.get('fly_ball', 0)
                if fb_count > 0:
                    metrics['gb_fb_ratio'] = float(gb_count / fb_count)

            # Batters faced
            batters_faced = statcast_data.groupby(['game_date', 'at_bat_number']).size()
            bf_count = len(batters_faced)

            if bf_count > 0:
                # K and BB rates (K% prevents RBIs, BB% increases)
                strikeouts = statcast_data['events'].isin(['strikeout', 'strikeout_double_play']).sum()
                walks = statcast_data['events'].eq('walk').sum()

                metrics['k_pct'] = float((strikeouts / bf_count) * 100)
                metrics['bb_pct'] = float((walks / bf_count) * 100)

                # Hits and runs allowed
                hits_allowed = statcast_data['events'].isin(['single', 'double', 'triple', 'home_run']).sum()
                metrics['hits_per_bf'] = float(hits_allowed / bf_count)

                # HR rate
                hrs_allowed = statcast_data['events'].eq('home_run').sum()
                metrics['hr_rate_allowed'] = float(hrs_allowed / bf_count)

                # WHIP proxy (hits + walks per BF, scaled)
                metrics['whip_proxy'] = float((hits_allowed + walks) / bf_count * 3)

                # LOB% (Left On Base %) - critical for RBI prevention
                # Estimate using runners on base vs runs scored relationship
                # Formula: LOB% ≈ (H + BB - R) / (H + BB - HR)
                runs_allowed = statcast_data['events'].isin(['single', 'double', 'triple', 'home_run']).sum()
                baserunners = hits_allowed + walks
                if baserunners - hrs_allowed > 0:
                    # Simplified LOB% estimate (actual would need game-level scoring data)
                    # Using K% and contact suppression as proxy for strand rate
                    strand_rate_proxy = 1 - (runs_allowed / (baserunners + 1))
                    metrics['lob_pct'] = float(max(0.5, min(0.9, strand_rate_proxy)) * 100)
                else:
                    metrics['lob_pct'] = 72.0  # League average

                # Contact suppression
                contact_events = statcast_data['description'].isin(['hit_into_play', 'foul']).sum()
                total_pitches = len(statcast_data)
                if total_pitches > 0:
                    metrics['contact_allowed_pct'] = float((contact_events / total_pitches) * 100)

            # Platoon splits (Lefty vs Righty advantage)
            if 'stand' in statcast_data.columns:
                # Performance vs LHH (left-handed hitters)
                vs_lhh = statcast_data[statcast_data['stand'] == 'L']
                vs_rhh = statcast_data[statcast_data['stand'] == 'R']

                if not vs_lhh.empty:
                    lhh_batted = vs_lhh[vs_lhh['type'] == 'X']
                    if not lhh_batted.empty:
                        lhh_hard_hits = (lhh_batted['launch_speed'] >= 95).sum()
                        metrics['hard_hit_pct_vs_lhh'] = float((lhh_hard_hits / len(lhh_batted)) * 100)

                    # K% vs LHH
                    lhh_pa = vs_lhh.groupby(['game_date', 'at_bat_number']).size()
                    lhh_k = vs_lhh['events'].isin(['strikeout', 'strikeout_double_play']).sum()
                    if len(lhh_pa) > 0:
                        metrics['k_pct_vs_lhh'] = float((lhh_k / len(lhh_pa)) * 100)

                if not vs_rhh.empty:
                    rhh_batted = vs_rhh[vs_rhh['type'] == 'X']
                    if not rhh_batted.empty:
                        rhh_hard_hits = (rhh_batted['launch_speed'] >= 95).sum()
                        metrics['hard_hit_pct_vs_rhh'] = float((rhh_hard_hits / len(rhh_batted)) * 100)

                    # K% vs RHH
                    rhh_pa = vs_rhh.groupby(['game_date', 'at_bat_number']).size()
                    rhh_k = vs_rhh['events'].isin(['strikeout', 'strikeout_double_play']).sum()
                    if len(rhh_pa) > 0:
                        metrics['k_pct_vs_rhh'] = float((rhh_k / len(rhh_pa)) * 100)

                # Calculate platoon advantage (positive = better vs RHH, negative = better vs LHH)
                if 'k_pct_vs_lhh' in metrics and 'k_pct_vs_rhh' in metrics:
                    metrics['platoon_k_advantage'] = metrics['k_pct_vs_lhh'] - metrics['k_pct_vs_rhh']
                if 'hard_hit_pct_vs_lhh' in metrics and 'hard_hit_pct_vs_rhh' in metrics:
                    metrics['platoon_contact_advantage'] = metrics['hard_hit_pct_vs_rhh'] - metrics['hard_hit_pct_vs_lhh']

            # === xFIP CALCULATION (ERA estimator for overall run suppression) ===
            # xFIP = Fielding Independent Pitching with normalized HR rate
            # Formula: xFIP = ((13*lgHR/FB*FB + 3*BB - 2*K) / IP) + constant
            # Simplified version using available Statcast data
            if bf_count > 0:
                # Estimate innings pitched (roughly 3 BF per inning)
                ip_estimate = bf_count / 3.0

                # Get FB count from earlier calculation
                if 'fb_pct' in metrics and batted_balls is not None and not batted_balls.empty:
                    fb_count = batted_balls['bb_type'].value_counts().get('fly_ball', 0)

                    # League average HR/FB rate (typically ~13-14%)
                    lg_hr_fb = 0.135

                    # Expected HRs based on league average HR/FB rate
                    expected_hrs = fb_count * lg_hr_fb

                    # xFIP calculation
                    # (13 * Expected HRs) + (3 * BB) - (2 * K) all divided by IP
                    k_count = strikeouts
                    bb_count = walks

                    if ip_estimate > 0:
                        xfip_numerator = (13 * expected_hrs) + (3 * bb_count) - (2 * k_count)
                        xfip_raw = (xfip_numerator / ip_estimate) + 3.20  # League constant

                        # Bound xFIP to reasonable range (2.0 to 6.0)
                        metrics['xfip'] = float(max(2.0, min(6.0, xfip_raw)))
                    else:
                        metrics['xfip'] = 4.00  # League average
                else:
                    metrics['xfip'] = 4.00  # League average

        except Exception as e:
            print(f"    [ERROR] Pitcher metric calculation error: {e}")

        return metrics

    def blend_batter_metrics(self, season: Dict, last_30: Dict, last_7: Dict) -> Dict:
        """
        Blend batter metrics using weighted average
        """
        blended = {}

        # All RBI-relevant metrics to blend
        metrics_to_blend = [
            'avg', 'obp', 'slg', 'iso', 'woba', 'babip',
            'avg_launch_speed', 'avg_launch_angle', 'barrel_pct', 'hard_hit_pct',
            'sweet_spot_pct', 'contact_pct', 'k_pct', 'bb_pct',
            'gb_pct', 'fb_pct', 'ld_pct', 'avg_hit_distance', 'xwoba',
            'risp_avg', 'risp_slg', 'risp_obp', 'risp_contact_pct'
        ]

        for metric in metrics_to_blend:
            values = []
            weights_used = []

            # Collect available values
            if metric in season and season[metric] is not None:
                values.append(season[metric])
                weights_used.append(self.weights[0])

            if metric in last_30 and last_30[metric] is not None:
                values.append(last_30[metric])
                weights_used.append(self.weights[1])

            if metric in last_7 and last_7[metric] is not None:
                values.append(last_7[metric])
                weights_used.append(self.weights[2])

            # Calculate weighted average
            if values:
                total_weight = sum(weights_used)
                blended[metric] = round(
                    sum(v * w for v, w in zip(values, weights_used)) / total_weight,
                    3
                )
            else:
                # Use league default
                blended[metric] = self.league_defaults.get(metric, 0.0)

        return blended

    def blend_pitcher_metrics(self, season: Dict, last_30: Dict, last_7: Dict) -> Dict:
        """
        Blend pitcher metrics using weighted average
        Enhanced with LOB%, platoon splits, and Barrel% allowed
        """
        blended = {}

        metrics_to_blend = [
            'avg_launch_speed_allowed', 'avg_launch_angle_allowed',
            'barrel_pct_allowed', 'hard_hit_pct_allowed',
            'gb_pct', 'fb_pct', 'ld_pct', 'gb_fb_ratio',
            'k_pct', 'bb_pct', 'hits_per_bf', 'hr_rate_allowed',
            'whip_proxy', 'contact_allowed_pct', 'lob_pct',
            'k_pct_vs_lhh', 'k_pct_vs_rhh', 'hard_hit_pct_vs_lhh', 'hard_hit_pct_vs_rhh',
            'platoon_k_advantage', 'platoon_contact_advantage', 'xfip'
        ]

        for metric in metrics_to_blend:
            values = []
            weights_used = []

            if metric in season and season[metric] is not None:
                values.append(season[metric])
                weights_used.append(self.weights[0])

            if metric in last_30 and last_30[metric] is not None:
                values.append(last_30[metric])
                weights_used.append(self.weights[1])

            if metric in last_7 and last_7[metric] is not None:
                values.append(last_7[metric])
                weights_used.append(self.weights[2])

            if values:
                total_weight = sum(weights_used)
                blended[metric] = round(
                    sum(v * w for v, w in zip(values, weights_used)) / total_weight,
                    3
                )
            else:
                blended[metric] = 0.0

        return blended

    def enrich_lineup_with_metrics(self, lineup_file: str = "mlb_lineups.json",
                                   output_file: str = "lineups_with_rbi_metrics.json") -> Dict:
        """
        Load lineups and enrich with RBI metrics
        """
        print("[INFO] Enriching lineups with RBI metrics")
        print("="*60)

        # Load lineup data
        if not os.path.exists(lineup_file):
            print(f"[ERROR] {lineup_file} not found")
            return {}

        try:
            with open(lineup_file, 'r') as f:
                lineup_data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load lineup file: {e}")
            return {}

        # Extract all players
        all_players = self._extract_players_from_lineups(lineup_data)
        print(f"[INFO] Found {len(all_players)} unique players")

        # Fetch metrics for all players
        player_metrics = {}

        for i, player in enumerate(all_players, 1):
            print(f"\n[{i}/{len(all_players)}] Processing: {player['name']}")

            try:
                # Get player ID (would need lookup - using placeholder)
                player_id = self._lookup_player_id(player['name'])

                if not player_id:
                    print(f"  [WARN] Could not find player ID for {player['name']}")
                    continue

                # Fetch metrics based on player type
                if player.get('player_type') == 'pitcher':
                    metrics = self.fetch_pitcher_metrics(player_id, player['name'])
                else:
                    metrics = self.fetch_batter_metrics(player_id, player['name'])

                player_metrics[player['name']] = metrics

            except Exception as e:
                print(f"  [ERROR] Failed to process {player['name']}: {e}")

        # Integrate metrics into lineup structure
        enriched_data = self._integrate_metrics_into_lineups(lineup_data, player_metrics)

        # Save output
        try:
            with open(output_file, 'w') as f:
                json.dump(enriched_data, f, indent=2)
            print(f"\n[SUCCESS] Saved enriched data to {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save output: {e}")

        return enriched_data

    def _extract_players_from_lineups(self, lineup_data: Dict) -> List[Dict]:
        """Extract all unique players from lineup data"""
        players = []
        seen = set()

        games = lineup_data.get('games', {})

        for game_key, game_data in games.items():
            # Batters
            for lineup_type in ['away_lineup', 'home_lineup']:
                lineup = game_data.get(lineup_type, [])
                team = game_data.get('away_team' if 'away' in lineup_type else 'home_team')

                for player in lineup:
                    player_key = f"{player.get('name')}_{team}"
                    if player_key not in seen and player.get('name'):
                        seen.add(player_key)
                        players.append({
                            'name': player['name'],
                            'team': team,
                            'position': player.get('position', ''),
                            'player_type': 'batter'
                        })

            # Pitchers
            for pitcher_type in ['away_pitcher', 'home_pitcher']:
                pitcher = game_data.get(pitcher_type, {})
                team = game_data.get('away_team' if 'away' in pitcher_type else 'home_team')

                if pitcher.get('name') and pitcher['name'] != 'TBD':
                    pitcher_key = f"{pitcher['name']}_{team}"
                    if pitcher_key not in seen:
                        seen.add(pitcher_key)
                        players.append({
                            'name': pitcher['name'],
                            'team': team,
                            'position': 'P',
                            'player_type': 'pitcher'
                        })

        return players

    def _lookup_player_id(self, player_name: str) -> Optional[int]:
        """
        Lookup player ID using pybaseball
        """
        try:
            # Clean name
            parts = player_name.split()
            if len(parts) >= 2:
                lookup_data = pyb.playerid_lookup(parts[-1], parts[0])
                if lookup_data is not None and not lookup_data.empty:
                    return int(lookup_data.iloc[0]['key_mlbam'])
        except:
            pass
        return None

    def _integrate_metrics_into_lineups(self, lineup_data: Dict, player_metrics: Dict) -> Dict:
        """
        Integrate blended metrics into lineup structure
        """
        enriched = lineup_data.copy()

        enriched['enrichment_info'] = {
            'date_enriched': datetime.now().isoformat(),
            'metrics_type': 'rbi_focused',
            'players_processed': len(player_metrics)
        }

        games = enriched.get('games', {})

        for game_key, game_data in games.items():
            # Enrich batters
            for lineup_type in ['away_lineup', 'home_lineup']:
                lineup = game_data.get(lineup_type, [])
                for i, player in enumerate(lineup):
                    player_name = player.get('name', '')
                    if player_name in player_metrics:
                        blended = player_metrics[player_name].get('blended_metrics', {})
                        enriched['games'][game_key][lineup_type][i]['blended_metrics'] = blended
                        enriched['games'][game_key][lineup_type][i]['sample_sizes'] = \
                            player_metrics[player_name].get('sample_sizes', {})

            # Enrich pitchers
            for pitcher_type in ['away_pitcher', 'home_pitcher']:
                pitcher = game_data.get(pitcher_type, {})
                pitcher_name = pitcher.get('name', '')
                if pitcher_name and pitcher_name in player_metrics:
                    blended = player_metrics[pitcher_name].get('blended_metrics', {})
                    enriched['games'][game_key][pitcher_type]['blended_metrics'] = blended
                    enriched['games'][game_key][pitcher_type]['sample_sizes'] = \
                        player_metrics[pitcher_name].get('sample_sizes', {})

        return enriched

    def run(self):
        """
        Main execution method
        """
        print("\n" + "="*80)
        print("RBI METRICS FETCHER - STARTING")
        print("="*80)

        # Enrich lineups with RBI metrics
        enriched_data = self.enrich_lineup_with_metrics(
            lineup_file="mlb_lineups.json",
            output_file="lineups_with_rbi_metrics.json"
        )

        if enriched_data:
            games_count = len(enriched_data.get('games', {}))
            print("\n" + "="*80)
            print("[SUCCESS] RBI METRICS COLLECTION COMPLETE")
            print(f"[INFO] Processed {games_count} games")
            print(f"[INFO] Output: lineups_with_rbi_metrics.json")
            print("="*80)


def main():
    """
    Main function
    """
    fetcher = RBIMetricsFetcher()
    fetcher.run()


if __name__ == "__main__":
    main()
