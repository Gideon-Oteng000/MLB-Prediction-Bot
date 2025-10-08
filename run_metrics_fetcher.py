#!/usr/bin/env python3
"""
Run Metrics Fetcher
Fetches and processes MLB hitter, pitcher, lineup, and environment metrics
for Runs Scored probability prediction.

Pipeline Stage 2 of 4:
1. mlb_lineups_fetcher.py
2. run_metrics_fetcher.py  ← THIS SCRIPT
3. weather_ballpark_run_fetcher.py
4. master_run_predictor.py
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


class RunMetricsFetcher:
    """
    Fetches and processes MLB metrics for Run Scored prediction
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

        # League average defaults for missing values
        self.league_defaults = {
            'avg': 0.248,
            'obp': 0.315,
            'slg': 0.411,
            'iso': 0.163,
            'bb_pct': 8.2,
            'k_pct': 22.0,
            'sb': 3,
            'cs': 1,
            'wrc_plus': 100,
            'bsr': 0.0,
            'babip': 0.295,
            'woba': 0.315,
            'hard_pct': 37.5,
            'contact_pct': 76.0,
            'ld_pct': 21.0,
            'gb_pct': 44.0,
            'fb_pct': 35.0,
            'hr_fb': 12.5,
            'pull_pct': 38.0,
            'cent_pct': 35.0,
            'oppo_pct': 27.0,
            # Pitcher defaults
            'whip': 1.30,
            'xfip': 4.00,
            'era': 4.20,
            'lob_pct': 72.0,
            'pitcher_k_pct': 22.0,
            'pitcher_bb_pct': 8.5,
            'pitcher_gb_pct': 45.0,
            'pitcher_fb_pct': 35.0,
            'babip_allowed': 0.295
        }

        # Cache database
        self.cache_db = "run_metrics_cache.db"
        self._init_cache()

        print("[INFO] Run Metrics Fetcher initialized")
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
        Fetch Run-relevant batter metrics across all time windows
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
        Calculate Run-relevant metrics from Statcast data
        """
        metrics = {}

        try:
            # Batted ball metrics
            batted_balls = statcast_data[statcast_data['type'] == 'X']

            if not batted_balls.empty:
                # Launch metrics
                metrics['avg_launch_speed'] = float(batted_balls['launch_speed'].mean())
                metrics['avg_launch_angle'] = float(batted_balls['launch_angle'].mean())

                # Hard-hit, barrel
                hard_hits = batted_balls['launch_speed'] >= 95
                metrics['hard_pct'] = float(hard_hits.mean() * 100)

                barrels = (batted_balls['launch_speed'] >= 98) & \
                         (batted_balls['launch_angle'].between(26, 30))
                metrics['barrel_pct'] = float(barrels.mean() * 100)

                # Batted ball types
                if 'bb_type' in batted_balls.columns:
                    bb_counts = batted_balls['bb_type'].value_counts()
                    total_bb = len(batted_balls)

                    metrics['gb_pct'] = float((bb_counts.get('ground_ball', 0) / total_bb) * 100)
                    metrics['fb_pct'] = float((bb_counts.get('fly_ball', 0) / total_bb) * 100)
                    metrics['ld_pct'] = float((bb_counts.get('line_drive', 0) / total_bb) * 100)

                # Pull/Center/Oppo
                if 'hit_location' in batted_balls.columns:
                    # Simplified directional spray
                    metrics['pull_pct'] = 38.0  # Would need detailed spray chart data
                    metrics['cent_pct'] = 35.0
                    metrics['oppo_pct'] = 27.0

            # Plate discipline & outcomes
            plate_appearances = statcast_data.groupby(['game_date', 'at_bat_number']).size()
            pa_count = len(plate_appearances)

            if pa_count > 0:
                # K and BB rates
                strikeouts = statcast_data['events'].isin(['strikeout', 'strikeout_double_play']).sum()
                walks = statcast_data['events'].eq('walk').sum()

                metrics['k_pct'] = float((strikeouts / pa_count) * 100)
                metrics['bb_pct'] = float((walks / pa_count) * 100)

                # Contact rate
                swings = statcast_data['description'].isin(['foul', 'hit_into_play',
                                                            'swinging_strike', 'foul_tip']).sum()
                contact = statcast_data['description'].isin(['foul', 'hit_into_play',
                                                             'foul_tip']).sum()
                if swings > 0:
                    metrics['contact_pct'] = float((contact / swings) * 100)

                # Batting outcomes
                hits = statcast_data['events'].isin(['single', 'double', 'triple', 'home_run']).sum()
                abs_count = statcast_data['events'].isin(['single', 'double', 'triple', 'home_run',
                                                    'field_out', 'grounded_into_double_play',
                                                    'fielders_choice_out']).sum()

                if abs_count > 0:
                    metrics['avg'] = float(hits / abs_count)

                    # Total bases for SLG
                    total_bases = (
                        statcast_data['events'].eq('single').sum() +
                        statcast_data['events'].eq('double').sum() * 2 +
                        statcast_data['events'].eq('triple').sum() * 3 +
                        statcast_data['events'].eq('home_run').sum() * 4
                    )
                    metrics['slg'] = float(total_bases / abs_count)
                    metrics['iso'] = float(metrics['slg'] - metrics['avg'])

                # OBP
                if pa_count > 0:
                    on_base = hits + walks
                    metrics['obp'] = float(on_base / pa_count)

                # wOBA
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

                # Stolen bases (approximation - Statcast doesn't have direct SB data)
                # Would need to join with game logs for actual SB/CS
                metrics['sb'] = 3  # Default
                metrics['cs'] = 1  # Default

                # wRC+ and BSR would need FanGraphs data
                metrics['wrc_plus'] = 100  # Default
                metrics['bsr'] = 0.0  # Default

        except Exception as e:
            print(f"    [ERROR] Metric calculation error: {e}")

        return metrics

    def fetch_pitcher_metrics(self, player_id: int, player_name: str) -> Dict:
        """
        Fetch Run-prevention pitcher metrics across all time windows
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
        Calculate Run-prevention metrics from pitcher Statcast data
        """
        metrics = {}

        try:
            # Batted balls allowed
            batted_balls = statcast_data[statcast_data['type'] == 'X']

            if not batted_balls.empty:
                # Hard-hit allowed
                hard_hits = batted_balls['launch_speed'] >= 95
                metrics['hard_pct_allowed'] = float(hard_hits.mean() * 100)

                # Batted ball types
                if 'bb_type' in batted_balls.columns:
                    bb_counts = batted_balls['bb_type'].value_counts()
                    total_bb = len(batted_balls)

                    metrics['gb_pct'] = float((bb_counts.get('ground_ball', 0) / total_bb) * 100)
                    metrics['fb_pct'] = float((bb_counts.get('fly_ball', 0) / total_bb) * 100)

                    # HR/FB ratio
                    fb_count = bb_counts.get('fly_ball', 0)
                    hrs_allowed = statcast_data['events'].eq('home_run').sum()
                    if fb_count > 0:
                        metrics['hr_fb'] = float((hrs_allowed / fb_count) * 100)

            # Batters faced
            batters_faced = statcast_data.groupby(['game_date', 'at_bat_number']).size()
            bf_count = len(batters_faced)

            if bf_count > 0:
                # K and BB rates
                strikeouts = statcast_data['events'].isin(['strikeout', 'strikeout_double_play']).sum()
                walks = statcast_data['events'].eq('walk').sum()

                metrics['k_pct'] = float((strikeouts / bf_count) * 100)
                metrics['bb_pct'] = float((walks / bf_count) * 100)

                # Hits allowed
                hits_allowed = statcast_data['events'].isin(['single', 'double', 'triple', 'home_run']).sum()

                # WHIP proxy
                metrics['whip'] = float((hits_allowed + walks) / bf_count * 3)

                # LOB% (strand rate estimate)
                runs_allowed = hits_allowed  # Simplified
                baserunners = hits_allowed + walks
                if baserunners > 0:
                    strand_rate_proxy = 1 - (runs_allowed / (baserunners + 1))
                    metrics['lob_pct'] = float(max(0.5, min(0.9, strand_rate_proxy)) * 100)
                else:
                    metrics['lob_pct'] = 72.0

                # BABIP allowed
                balls_in_play = statcast_data['events'].isin(['single', 'double', 'triple',
                                                              'field_out', 'grounded_into_double_play']).sum()
                babip_hits = statcast_data['events'].isin(['single', 'double', 'triple']).sum()
                if balls_in_play > 0:
                    metrics['babip_allowed'] = float(babip_hits / balls_in_play)

                # xFIP calculation
                if 'fb_pct' in metrics and batted_balls is not None and not batted_balls.empty:
                    fb_count = batted_balls['bb_type'].value_counts().get('fly_ball', 0)
                    lg_hr_fb = 0.135
                    expected_hrs = fb_count * lg_hr_fb
                    ip_estimate = bf_count / 3.0

                    if ip_estimate > 0:
                        xfip_numerator = (13 * expected_hrs) + (3 * walks) - (2 * strikeouts)
                        xfip_raw = (xfip_numerator / ip_estimate) + 3.20
                        metrics['xfip'] = float(max(2.0, min(6.0, xfip_raw)))
                    else:
                        metrics['xfip'] = 4.00
                else:
                    metrics['xfip'] = 4.00

                # ERA estimate (simplified)
                metrics['era'] = metrics.get('xfip', 4.00) * 1.05

            # Platoon splits (wOBA vs LHH/RHH)
            if 'stand' in statcast_data.columns:
                vs_lhh = statcast_data[statcast_data['stand'] == 'L']
                vs_rhh = statcast_data[statcast_data['stand'] == 'R']

                if not vs_lhh.empty and 'woba_value' in vs_lhh.columns:
                    woba_lhh = vs_lhh[vs_lhh['woba_value'].notna()]
                    if not woba_lhh.empty:
                        metrics['woba_vs_lhh'] = float(woba_lhh['woba_value'].mean())

                if not vs_rhh.empty and 'woba_value' in vs_rhh.columns:
                    woba_rhh = vs_rhh[vs_rhh['woba_value'].notna()]
                    if not woba_rhh.empty:
                        metrics['woba_vs_rhh'] = float(woba_rhh['woba_value'].mean())

        except Exception as e:
            print(f"    [ERROR] Pitcher metric calculation error: {e}")

        return metrics

    def blend_batter_metrics(self, season: Dict, last_30: Dict, last_7: Dict) -> Dict:
        """
        Blend batter metrics using weighted average
        """
        blended = {}

        # All Run-relevant metrics to blend
        metrics_to_blend = [
            'avg', 'obp', 'slg', 'iso', 'woba', 'babip',
            'bb_pct', 'k_pct', 'sb', 'cs', 'wrc_plus', 'bsr',
            'hard_pct', 'contact_pct', 'ld_pct', 'gb_pct', 'fb_pct',
            'hr_fb', 'pull_pct', 'cent_pct', 'oppo_pct', 'barrel_pct'
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

        # Compute derived metrics
        obp = blended.get('obp', 0.315)
        slg = blended.get('slg', 0.411)
        bsr = blended.get('bsr', 0.0)
        sb = blended.get('sb', 3)
        cs = blended.get('cs', 1)

        # Run opportunity index
        blended['run_opportunity_index'] = round(obp * (1 + slg), 3)

        # Speed advancement score
        blended['speed_advancement_score'] = round(bsr + (sb - cs) * 0.2, 3)

        return blended

    def blend_pitcher_metrics(self, season: Dict, last_30: Dict, last_7: Dict) -> Dict:
        """
        Blend pitcher metrics using weighted average
        """
        blended = {}

        metrics_to_blend = [
            'k_pct', 'bb_pct', 'whip', 'xfip', 'era', 'lob_pct',
            'gb_pct', 'fb_pct', 'hr_fb', 'babip_allowed', 'hard_pct_allowed',
            'woba_vs_lhh', 'woba_vs_rhh'
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
                blended[metric] = self.league_defaults.get(metric, 0.0)

        # Compute pitcher opponent factor
        whip = blended.get('whip', 1.30)
        blended['pitcher_opponent_factor'] = round(1 + (whip - 1.30) * 0.2, 3)

        return blended

    def calculate_lineup_support_factor(self, lineup: List[Dict], current_index: int) -> float:
        """
        Calculate lineup support factor based on next two hitters
        """
        try:
            next_batters = []
            lineup_size = len(lineup)

            for offset in [1, 2]:
                next_idx = (current_index + offset) % lineup_size
                next_batter = lineup[next_idx]
                next_obp = next_batter.get('blended_metrics', {}).get('obp', 0.315)
                next_batters.append(next_obp)

            if next_batters:
                avg_next_obp = sum(next_batters) / len(next_batters)
                lineup_support_factor = avg_next_obp / 0.320
                return round(lineup_support_factor, 3)
        except:
            pass

        return 1.0

    def enrich_lineup_with_metrics(self, lineup_file: str = "mlb_lineups.json",
                                   output_file: str = "run_metrics_output.json") -> Dict:
        """
        Load lineups and enrich with Run metrics
        """
        print("[INFO] Enriching lineups with Run metrics")
        print("="*80)

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
                # Get player ID
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

        # Calculate lineup support factors
        enriched_data = self._add_lineup_support_factors(enriched_data)

        # Save output
        try:
            enriched_data['date'] = self.today_str
            enriched_data['total_players'] = len(player_metrics)

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

    def _add_lineup_support_factors(self, enriched_data: Dict) -> Dict:
        """
        Add lineup support factors for each batter
        """
        games = enriched_data.get('games', {})

        for game_key, game_data in games.items():
            for lineup_type in ['away_lineup', 'home_lineup']:
                lineup = game_data.get(lineup_type, [])

                for i, player in enumerate(lineup):
                    if player.get('blended_metrics'):
                        support_factor = self.calculate_lineup_support_factor(lineup, i)
                        enriched_data['games'][game_key][lineup_type][i]['blended_metrics']['lineup_support_factor'] = support_factor

        return enriched_data

    def run(self):
        """
        Main execution method
        """
        print("\n" + "="*80)
        print("RUN METRICS FETCHER - STARTING")
        print("="*80)

        # Enrich lineups with Run metrics
        enriched_data = self.enrich_lineup_with_metrics(
            lineup_file="mlb_lineups.json",
            output_file="run_metrics_output.json"
        )

        if enriched_data:
            games_count = len(enriched_data.get('games', {}))
            print("\n" + "="*80)
            print("[SUCCESS] RUN METRICS COLLECTION COMPLETE")
            print(f"[INFO] Processed {games_count} games")
            print(f"[INFO] Output: run_metrics_output.json")
            print("="*80)


def main():
    """
    Main function
    """
    fetcher = RunMetricsFetcher()
    fetcher.run()


if __name__ == "__main__":
    main()
