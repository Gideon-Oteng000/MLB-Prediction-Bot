# Historical Data Collection for HR Prediction Model

## Overview
The `historical_data_collector.py` script collects historical MLB game data from 2018-2024 seasons to train your machine learning model.

## What Data is Collected

### Game-Level Data
- Game date, teams, stadium
- Home/away designation
- Final HR outcomes

### Player-Level Features (per batter, per game)
**Season Metrics (as-of game date):**
- Barrel %, hard-hit %, exit velocity
- HR rate, K%, BB%, ISO
- Plate appearances (sample size)

**Recent Form:**
- Last 30 days: barrel%, hard-hit%, HR rate
- Last 7 days: barrel%, HR rate

**Environmental Factors:**
- Stadium HR factor (e.g., Coors Field = 1.25)
- Elevation
- Home/away status

**Target Variable:**
- `hr_hit`: Binary (1 if player hit HR, 0 if not)
- `hr_count`: Number of HRs hit in that game

## Output Files

### 1. CSV Format (for ML training)
**File:** `historical_hr_training_data_2018_2024.csv`

**Structure:** One row per batter per game
```
date,player_name,team,stadium,season_barrel_pct,season_hr_rate,...,hr_hit
2018-04-01,Aaron Judge,NYY,Yankee Stadium,15.2,0.08,...,1
2018-04-01,Giancarlo Stanton,NYY,Yankee Stadium,12.8,0.06,...,0
...
```

**Estimated Rows:** ~500,000+ rows

### 2. JSON Format (for inspection)
**File:** `historical_hr_data_2018_2024.json`

## Usage

### Run the Collector
```bash
python historical_data_collector.py
```

### Progress Tracking
The script automatically saves progress:
- `collection_progress.json` - tracks completed years
- `checkpoint_data.json` - saves checkpoints every 50 games

**To resume after interruption:** Just run the script again.

## Important Notes

### Time Estimate
- **Full collection (2018-2024):** 6-12 hours
- Includes automatic rate limiting

### Testing Mode
Currently set to collect **10 games per year** for testing.

**To collect full data:**
Edit line 155 in `historical_data_collector.py`:
```python
# Change this:
for i, game_date in enumerate(game_dates[:10], 1):  # TESTING

# To this:
for i, game_date in enumerate(game_dates, 1):  # Full collection
```

## Data Schema (CSV Columns)

| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier |
| `date` | Game date (YYYY-MM-DD) |
| `player_name` | Batter name |
| `team` | Batter's team |
| `stadium` | Stadium name |
| `stadium_hr_factor` | Park factor (1.0 = neutral) |
| `elevation` | Stadium elevation (feet) |
| `bat_side` | 'L' or 'R' |
| `season_barrel_pct` | Season barrel % |
| `season_hard_hit_pct` | Season hard-hit % |
| `season_hr_rate` | Season HR rate |
| `season_k_pct` | Season K% |
| `season_iso` | Season ISO |
| `month_barrel_pct` | Last 30 days barrel % |
| `month_hr_rate` | Last 30 days HR rate |
| `week_barrel_pct` | Last 7 days barrel % |
| **`hr_hit`** | **Target: 1 if HR hit, 0 otherwise** |

## Next Steps

After data collection:

### 1. Train ML Model
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('historical_hr_training_data_2018_2024.csv')

X = df.drop(['hr_hit', 'hr_count', 'game_id', 'player_name', 'date'], axis=1)
y = df['hr_hit']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
```

### 2. Save Model
```python
import joblib
joblib.dump(model, 'production_mlb_model.pkl')
```
