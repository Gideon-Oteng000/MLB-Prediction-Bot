# MLB Historical Data Pipeline (2018-2024)

Expert-level data engineering pipeline for collecting comprehensive MLB historical data to train machine learning models for predicting:
- **Home Runs (HR)**
- **Hits**
- **RBIs**
- **Runs**
- **Total Bases**
- **Singles, Doubles, Triples**
- **Strikeouts**

## 🎯 Features

### Data Sources
- **Statcast (Baseball Savant)** → Advanced metrics (exit velocity, launch angle, barrel rate, etc.)
- **MLB StatsAPI** → Game schedules, lineups, player info, box scores
- **OpenWeather API** → Weather conditions for games
- **Park Factors** → Stadium-specific hitting multipliers

### Database Schema
Perfect SQLite schema with 6 tables optimized for ML training:
- `games` → Game metadata, park factors, weather
- `players` → Player information and characteristics
- `lineups` → Game lineups and batting order
- `batter_stats` → Advanced hitting metrics up to each game date
- `pitcher_stats` → Advanced pitching metrics up to each game date
- `game_logs` → Actual game outcomes (labels for ML)

### Pipeline Features
- ✅ **Restartable** - Skips already processed games
- ✅ **Error handling** - Continues on individual game failures
- ✅ **Rate limiting** - Respects API limits
- ✅ **Comprehensive logging** - Full audit trail
- ✅ **Modular design** - Easy to extend and modify

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test with Small Dataset
```python
python run_pipeline_example.py
# Choose option 1: Test single day
```

### 3. Run Full Historical Pipeline
```python
python run_pipeline_example.py
# Choose option 4: Run FULL HISTORICAL (2018-2024)
```

## 📊 Database Structure

### Core Tables

```sql
-- Game metadata with park factors and weather
CREATE TABLE games (
    game_id TEXT PRIMARY KEY,
    date DATE,
    home_team TEXT,
    away_team TEXT,
    venue TEXT,
    park_factor_hr REAL,        -- HR park factor
    park_factor_doubles REAL,   -- 2B park factor
    park_factor_triples REAL,   -- 3B park factor
    park_factor_runs REAL,      -- Runs park factor
    elevation REAL,             -- Stadium elevation
    game_time TEXT,
    temperature REAL,           -- Game temperature
    wind_speed REAL,            -- Wind speed
    wind_direction TEXT         -- Wind direction
);

-- Advanced batter metrics (features)
CREATE TABLE batter_stats (
    player_id TEXT,
    game_date DATE,
    season INTEGER,
    -- Power metrics
    barrel_rate REAL,           -- Barrel rate
    hard_hit_rate REAL,         -- Hard hit rate (95+ mph)
    avg_exit_velocity REAL,     -- Average exit velocity
    max_exit_velocity REAL,     -- Max exit velocity
    avg_launch_angle REAL,      -- Average launch angle
    sweet_spot_pct REAL,        -- Sweet spot percentage
    pull_pct REAL,              -- Pull percentage
    -- Expected stats
    iso REAL,                   -- Isolated power
    slg REAL,                   -- Slugging percentage
    xslg REAL,                  -- Expected slugging
    xiso REAL,                  -- Expected isolated power
    woba REAL,                  -- Weighted on-base average
    xwoba REAL,                 -- Expected wOBA
    avg REAL,                   -- Batting average
    xba REAL,                   -- Expected batting average
    -- Plate discipline
    k_pct REAL,                 -- Strikeout rate
    bb_pct REAL,                -- Walk rate
    whiff_pct REAL,             -- Whiff rate
    o_swing_pct REAL,           -- Out-of-zone swing rate
    z_contact_pct REAL,         -- Zone contact rate
    -- Speed
    sprint_speed REAL,          -- Sprint speed
    sample_at_bats INTEGER,     -- Sample size
    PRIMARY KEY (player_id, game_date)
);

-- Game outcomes (ML labels)
CREATE TABLE game_logs (
    game_id TEXT,
    player_id TEXT,
    hr INTEGER,                 -- Home runs (target)
    hit INTEGER,                -- Hits (target)
    rbi INTEGER,                -- RBIs (target)
    run INTEGER,                -- Runs (target)
    total_bases INTEGER,        -- Total bases (target)
    single INTEGER,             -- Singles (target)
    double INTEGER,             -- Doubles (target)
    triple INTEGER,             -- Triples (target)
    strikeouts INTEGER,         -- Strikeouts (target)
    PRIMARY KEY (game_id, player_id)
);
```

## 🎓 Usage Examples

### Basic Pipeline Run
```python
from mlb_historical_data_pipeline import MLBDataPipeline, PipelineConfig

# Configure for 2024 season
config = PipelineConfig(
    START_YEAR=2024,
    END_YEAR=2024,
    DB_PATH="mlb_2024.db"
)

# Run pipeline
pipeline = MLBDataPipeline(config)
pipeline.run_full_pipeline()
```

### Check Pipeline Status
```python
pipeline = MLBDataPipeline(config)
status = pipeline.get_pipeline_status()

print(f"Games: {status['games']:,}")
print(f"Players: {status['players']:,}")
print(f"Batter Stats: {status['batter_stats']:,}")
print(f"Game Logs: {status['game_logs']:,}")
```

### Query for ML Training
```sql
-- Create training dataset
SELECT
    g.park_factor_hr,
    g.temperature,
    p.position,
    l.batting_order,
    bs.barrel_rate,
    bs.avg_exit_velocity,
    bs.k_pct,
    ps.era as pitcher_era,
    ps.hr_per_9 as pitcher_hr_per_9,
    -- LABELS
    gl.hr,
    gl.hit,
    gl.rbi
FROM games g
JOIN lineups l ON g.game_id = l.game_id
JOIN players p ON l.player_id = p.player_id
JOIN batter_stats bs ON p.player_id = bs.player_id AND g.date = bs.game_date
JOIN pitcher_stats ps ON g.game_id = ps.game_id
JOIN game_logs gl ON g.game_id = gl.game_id AND p.player_id = gl.player_id
WHERE g.date >= '2018-03-01';
```

## ⚡ Performance

### Expected Runtime
- **Single day**: ~2-3 minutes
- **Full season**: ~2-4 hours
- **Full historical (2018-2024)**: ~6-12 hours

### Database Size
- **Full historical dataset**: ~2-5 GB
- **Records expected**:
  - Games: ~16,000
  - Players: ~3,000
  - Batter Stats: ~500,000
  - Game Logs: ~200,000

## 🔧 Configuration

### PipelineConfig Options
```python
@dataclass
class PipelineConfig:
    OPENWEATHER_API_KEY: str = "your_api_key_here"
    DB_PATH: str = "mlb_training_data.db"
    START_YEAR: int = 2018
    END_YEAR: int = 2024
    RATE_LIMIT_DELAY: float = 1.0  # Seconds between API calls
    BATCH_SIZE: int = 50           # Games per batch
```

## 📈 ML Training Ready

The pipeline creates a perfect dataset for training ML models:

### Features (X)
- **Player metrics**: Barrel rate, exit velocity, launch angle, etc.
- **Pitcher metrics**: ERA, HR/9, strikeout rate, etc.
- **Environmental**: Park factors, weather, elevation
- **Situational**: Batting order, home/away, etc.

### Labels (y)
- **HR**: Home runs hit
- **Hit**: Hits
- **RBI**: RBIs
- **Run**: Runs scored
- **Total_bases**: Total bases
- **Single/Double/Triple**: Specific hit types
- **Strikeouts**: Strikeouts

### Sample ML Code
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_sql("""
    SELECT bs.barrel_rate, bs.avg_exit_velocity, bs.k_pct,
           g.park_factor_hr, gl.hr
    FROM games g
    JOIN batter_stats bs ON g.date = bs.game_date
    JOIN game_logs gl ON g.game_id = gl.game_id
    WHERE gl.hr IS NOT NULL
""", conn)

# Train model
X = df[['barrel_rate', 'avg_exit_velocity', 'k_pct', 'park_factor_hr']]
y = df['hr']

model = RandomForestRegressor()
model.fit(X, y)
```

## 🚨 Important Notes

1. **API Rate Limits**: The pipeline includes delays to respect API limits
2. **Restart Capability**: If interrupted, restart will skip completed games
3. **Error Handling**: Individual game failures won't stop the entire pipeline
4. **Data Quality**: Pipeline includes validation and default values for missing data
5. **Log Monitoring**: Check `data_pipeline.log` for detailed progress

## 📁 Output Files

- `mlb_training_data.db` - Main SQLite database
- `data_pipeline.log` - Detailed processing log
- `sample_ml_query.sql` - Example query for ML training

## 🏆 Ready for Production

This pipeline is enterprise-ready with:
- Comprehensive error handling
- Restart capability
- Rate limiting
- Detailed logging
- Modular design
- Complete documentation

Perfect foundation for building ML models to predict baseball outcomes!