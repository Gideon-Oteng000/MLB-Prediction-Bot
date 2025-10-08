# Weather & Ballpark Run Fetcher - Documentation

## Overview
`weather_ballpark_run_fetcher.py` is **Stage 3** of the Run Scored prediction pipeline.

## Pipeline Flow
```
1. mlb_lineups_fetcher.py         → mlb_lineups.json
2. run_metrics_fetcher.py          → run_metrics_output.json
3. weather_ballpark_run_fetcher.py → final_integrated_run_data.json  ← THIS SCRIPT
4. master_run_predictor.py         → run_predictions.json
```

## Purpose
Calculates **Run Environment Multipliers** using:
- ✅ Real-time weather data (temperature, wind, humidity, pressure)
- ✅ Park factors (offensive friendliness)
- ✅ Stadium orientation (wind direction alignment)
- ✅ Elevation effects (altitude/air density)

## Total Run Multiplier Formula

### Components:

#### 1️⃣ **Temperature Factor**
Warm air carries the ball further → more runs
```python
if temp > 75:
    temp_factor = 1 + ((temp - 75) / 10) × 0.03
elif temp < 55:
    temp_factor = 1 - ((55 - temp) / 10) × 0.02
else:
    temp_factor = 1.0
```
**Example**: 85°F → `1 + (10/10) × 0.03 = 1.03` (3% boost)

#### 2️⃣ **Humidity Factor**
High humidity = denser air = more ball carry
```python
if humidity < 40:
    humidity_factor = 0.97     # Dry air penalty
elif humidity > 70:
    humidity_factor = 1.03     # Humid air boost
else:
    humidity_factor = 1.0      # Normal
```

#### 3️⃣ **Air Pressure / Altitude Factor**
Lower pressure = thinner air = more offense
```python
pressure_factor = 1 + ((1013 - pressure) / 20) × 0.02
```
**Example**: Coors Field (5,200 ft) → very low pressure → significant boost

#### 4️⃣ **Wind Factor** (with Stadium Orientation)
```python
angle_diff = (wind_direction - stadium_orientation + 180) % 360 - 180

if -45° ≤ angle_diff ≤ 45°:
    # Wind blowing out to center
    wind_factor = 1 + (wind_speed × 0.004)
elif 45° < angle_diff ≤ 90° or -90° ≤ angle_diff < -45°:
    # Wind to power alleys
    wind_factor = 1 + (wind_speed × 0.002)
else:
    # Wind blowing in
    wind_factor = 1 - (wind_speed × 0.003)
```

**Example Orientations**:
- Wrigley Field: 30° (NE winds blow out)
- Oracle Park: 90° (cross winds from bay)
- Yankee Stadium: 75° (prevailing winds help RF)

#### 5️⃣ **Park Run Factor**
Inherent offensive friendliness (100 = league average)
```python
park_run_factor = stadium['run_factor'] / 100
```

**Park Examples**:
- **Coors Field**: 122 (extreme hitter's park)
- **Wrigley Field**: 109 (hitter-friendly)
- **Oracle Park**: 87 (pitcher's park)
- **Petco Park**: 90 (pitcher-friendly)

### **Combined Multiplier**:
```python
total_run_multiplier = (
    temp_factor ×
    humidity_factor ×
    pressure_factor ×
    wind_factor ×
    park_run_factor
)

# Capped between 0.75 and 1.35
total_run_multiplier = max(0.75, min(1.35, total_run_multiplier))
```

## Example Calculations

### Example 1: Coors Field (Hot Day)
```
Temperature: 85°F → temp_factor = 1.03
Humidity: 30% → humidity_factor = 0.97
Pressure: 850mb (altitude) → pressure_factor = 1.16
Wind: 10mph out to center → wind_factor = 1.04
Park Factor: 122 → park_run_factor = 1.22

total_run_multiplier = 1.03 × 0.97 × 1.16 × 1.04 × 1.22 = 1.35 (capped)
```
**Result**: Maximum offensive environment (+35% runs)

### Example 2: Oracle Park (Cold, Windy)
```
Temperature: 55°F → temp_factor = 1.00
Humidity: 75% → humidity_factor = 1.03
Pressure: 1020mb → pressure_factor = 0.99
Wind: 15mph blowing in → wind_factor = 0.955
Park Factor: 87 → park_run_factor = 0.87

total_run_multiplier = 1.00 × 1.03 × 0.99 × 0.955 × 0.87 = 0.83
```
**Result**: Pitcher's paradise (-17% runs)

### Example 3: Neutral Conditions
```
Temperature: 72°F → temp_factor = 1.00
Humidity: 55% → humidity_factor = 1.00
Pressure: 1013mb → pressure_factor = 1.00
Wind: 5mph neutral → wind_factor = 1.00
Park Factor: 100 → park_run_factor = 1.00

total_run_multiplier = 1.00 × 1.00 × 1.00 × 1.00 × 1.00 = 1.00
```
**Result**: League average conditions

## Output Structure

```json
{
  "games": {
    "NYY_vs_BOS": {
      "home_team": "NYY",
      "away_team": "BOS",
      "weather_ballpark": {
        "stadium": {
          "name": "Yankee Stadium",
          "city": "Bronx",
          "run_factor": 105,
          "elevation": 55,
          "orientation": 75
        },
        "weather": {
          "temperature": 82,
          "humidity": 62,
          "pressure": 1009,
          "wind_speed": 10,
          "wind_direction": 270
        },
        "temperature": 82,
        "wind_speed": 10,
        "wind_direction": 270,
        "humidity": 62,
        "pressure": 1009,
        "park_run_factor": 1.05,
        "total_run_multiplier": 1.18
      }
    }
  }
}
```

## Weather Fallback
When OpenWeatherMap API fails, uses neutral defaults:
```python
temperature: 75°F
humidity: 55%
pressure: 1013mb
wind_speed: 5mph
wind_direction: 0°
```

## Usage
```bash
# Run the fetcher
python weather_ballpark_run_fetcher.py

# Input: run_metrics_output.json
# Output: final_integrated_run_data.json
```

## Stadium Orientation Details

All 30 MLB stadiums have precise orientation data:

| Stadium | Orientation | Wind Effect |
|---------|-------------|-------------|
| Wrigley Field | 30° | NE winds blow out (famous) |
| Oracle Park | 90° | SF Bay cross-winds |
| Yankee Stadium | 75° | Helps RF short porch |
| Coors Field | 0° | Neutral (elevation matters) |
| Fenway Park | 45° | Green Monster effects |

## Integration with Run Predictor

The `total_run_multiplier` is applied in `master_run_predictor.py`:

```python
# Base run probability calculation
base_run_prob = league_run_rate × batter_factors × pitcher_factors × lineup_factors

# Apply weather/park multiplier
final_run_prob = base_run_prob × total_run_multiplier
```

## Key Differences from RBI Multiplier

| Factor | RBI Model | Run Model |
|--------|-----------|-----------|
| **Temperature** | Moderate effect | **Stronger effect** |
| **Wind** | Dampened 50% | **Full effect** |
| **Park Factor** | AVG + XBH focused | **Run-scoring focused** |
| **Cap Range** | 0.8 - 1.3 | **0.75 - 1.35** (wider) |

## Validation Checklist
✅ All games have `weather_ballpark` data
✅ `total_run_multiplier` between 0.75 and 1.35
✅ Stadium orientation used in wind calculations
✅ Fallback weather when API unavailable
✅ Park factors for all 30 MLB stadiums

## Next Step
Run `master_run_predictor.py` to generate Run Scored predictions!
