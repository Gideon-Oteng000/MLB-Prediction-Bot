# Run Metrics Fetcher - Documentation

## Overview
`run_metrics_fetcher.py` is **Stage 2** of the Run Scored prediction pipeline.

## Pipeline Flow
```
1. mlb_lineups_fetcher.py        → mlb_lineups.json
2. run_metrics_fetcher.py         → run_metrics_output.json  ← THIS SCRIPT
3. weather_ballpark_run_fetcher.py → final_integrated_run_data.json
4. master_run_predictor.py        → run_predictions.json
```

## Key Features

### ✅ Hitter Metrics
Fetches and blends the following metrics from **Statcast**:
- **Offensive**: AVG, OBP, SLG, ISO, wOBA, BABIP
- **Discipline**: BB%, K%, Contact%
- **Batted Ball**: LD%, GB%, FB%, Hard%, Barrel%
- **Speed**: SB, CS, BSR (BaseRunning)
- **Advanced**: wRC+, HR/FB
- **Spray**: Pull%, Center%, Oppo%

### ✅ Pitcher Metrics
- **Run Prevention**: K%, BB%, WHIP, xFIP, ERA, LOB%
- **Batted Ball**: GB%, FB%, HR/FB, Hard% allowed, BABIP allowed
- **Platoon Splits**: wOBA vs LHH, wOBA vs RHH
- **Derived**: `pitcher_opponent_factor = 1 + (WHIP - 1.30) × 0.2`

### ✅ Computed Features
1. **Run Opportunity Index**
   ```python
   run_opportunity_index = OBP × (1 + SLG)
   ```
   - Measures batter's ability to get on base AND advance runners

2. **Speed Advancement Score**
   ```python
   speed_advancement_score = BSR + (SB - CS) × 0.2
   ```
   - Captures baserunning and stolen base contribution

3. **Lineup Support Factor**
   ```python
   lineup_support_factor = avg(OBP_next_two_hitters) / 0.320
   ```
   - Higher values = better protection in lineup
   - Example: Batting ahead of high-OBP hitters = more run opportunities

## Time Window Blending
Uses **weighted averages** across 3 timeframes:
- **Season** (60% weight): Full 2025 season
- **Last 30 days** (30% weight): Recent form
- **Last 7 days** (10% weight): Hot/cold streaks

## League Defaults
When data is missing, uses MLB league averages:
```python
AVG=0.248, OBP=0.315, SLG=0.411, ISO=0.163
BB%=8.2, K%=22.0, SB=3, CS=1, wRC+=100
WHIP=1.30, xFIP=4.00, LOB%=72.0, HR/FB=12.5
```

## Caching System
- Uses SQLite database (`run_metrics_cache.db`)
- Caches metrics by player, date, and timeframe
- Significantly speeds up repeated runs
- Automatically clears stale data

## Output Format
```json
{
  "date": "2025-03-12",
  "total_players": 236,
  "games": {
    "BOS_vs_NYY": {
      "home_team": "BOS",
      "away_team": "NYY",
      "home_pitcher": {
        "name": "Chris Sale",
        "throws": "L",
        "blended_metrics": {
          "k_pct": 28.5,
          "bb_pct": 7.2,
          "whip": 1.15,
          "xfip": 3.45,
          "pitcher_opponent_factor": 0.97
        }
      },
      "home_lineup": [
        {
          "name": "Rafael Devers",
          "bat_side": "L",
          "batting_order": 3,
          "blended_metrics": {
            "avg": 0.295,
            "obp": 0.370,
            "slg": 0.545,
            "iso": 0.250,
            "bb_pct": 10.0,
            "k_pct": 18.0,
            "run_opportunity_index": 0.571,
            "speed_advancement_score": 2.5,
            "lineup_support_factor": 1.12
          }
        }
      ]
    }
  }
}
```

## Usage
```bash
# Run the fetcher
python run_metrics_fetcher.py

# Input: mlb_lineups.json (from mlb_lineups_fetcher.py)
# Output: run_metrics_output.json
```

## Error Handling
- Graceful fallback to league defaults when API fails
- Caches successful fetches to minimize API calls
- Rate limiting (0.1s delay between requests)
- Progress logging for each player

## Dependencies
```bash
pip install pybaseball pandas numpy
```

## Next Steps
After running this script:
1. Run `weather_ballpark_run_fetcher.py` to add environmental factors
2. Run `master_run_predictor.py` to generate Run Scored predictions

## Key Differences from RBI Model
| Metric | RBI Model | Run Model |
|--------|-----------|-----------|
| **Primary Focus** | Hitting with RISP | Getting on base + advancing |
| **Speed Metrics** | Less important | Critical (SB, BSR) |
| **OBP Weight** | Moderate | Very high |
| **Lineup Context** | Previous hitters' OBP | **Next** hitters' OBP |
| **Pitcher Impact** | Strand rate (LOB%) | Run prevention (xFIP) |

## Validation
Check output file has:
- ✅ All batters have `run_opportunity_index`
- ✅ All batters have `speed_advancement_score`
- ✅ All batters have `lineup_support_factor`
- ✅ All pitchers have `pitcher_opponent_factor`
- ✅ Missing values replaced with league defaults
