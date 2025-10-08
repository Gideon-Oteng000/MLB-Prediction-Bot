# Master Run Predictor - Documentation

## Overview
`master_run_predictor.py` is the **final stage (4 of 4)** in the Run Scored prediction pipeline.

## Pipeline Flow
```
1. mlb_lineups_fetcher.py         → mlb_lineups.json
2. run_metrics_fetcher.py          → run_metrics_output.json
3. weather_ballpark_run_fetcher.py → final_integrated_run_data.json
4. master_run_predictor.py         → run_predictions.json  ← THIS SCRIPT
```

## Run Probability Formula

### Complete Formula:
```
Run Probability = League Baseline × Batter Factors × Pitcher Factors ×
                  Lineup Context × Lineup Position × Park/Weather × Platoon Multiplier
```

### League Baseline:
```python
league_run_rate = 0.15  # ~15% of PAs result in a run scored
```

## Batter Factors (7 Components)

### 1️⃣ **On-Base Factor** (Most Critical)
Can't score if you don't get on base!
```python
obp_factor = 1 + (OBP - 0.315) × 1.8
```
**Weight**: 1.8 (highest)
- OBP = 0.400 → factor = 1.153 (+15.3%)
- OBP = 0.280 → factor = 0.937 (-6.3%)

### 2️⃣ **Baserunning Skill Factor**
Good baserunners convert singles into runs
```python
bsr_factor = 1 + (BSR × 0.05)
```
- BSR = +5 (elite) → factor = 1.25
- BSR = -3 (poor) → factor = 0.85

### 3️⃣ **Stolen Base Success Rate**
Speed helps score from 1st on doubles
```python
sb_rate = SB / (SB + CS)
sb_factor = 1 + (sb_rate - 0.70) × 0.4
```
- 90% success → factor = 1.08
- 50% success → factor = 0.92

### 4️⃣ **Contact Ability**
Can't score if you strike out
```python
contact_factor = 1 + (contact_pct - 0.76) × 1.0
```
- 85% contact → factor = 1.09
- 70% contact → factor = 0.94

### 5️⃣ **Power Factor**
Extra base hits = easier to score
```python
power_factor = 1 + (SLG - 0.411) × 0.8
```
- SLG = 0.550 → factor = 1.111
- SLG = 0.350 → factor = 0.951

### 6️⃣ **Line Drive Factor**
Line drives find gaps
```python
ld_factor = 1 + (LD% - 0.21) × 0.5
```

### 7️⃣ **Walk Rate Factor**
Walks get you on base
```python
walk_factor = 1 + (BB% - 0.082) × 0.3
```

### **Combined Batter Factor**:
```python
batter_factor = (obp_factor × bsr_factor × sb_factor × contact_factor ×
                power_factor × ld_factor × walk_factor)
```

## Pitcher Factors (5 Components)

### 1️⃣ **WHIP Factor**
More baserunners = more run opportunities
```python
whip_factor = 1 + (WHIP - 1.30) × 0.25
```
**Weight**: 0.25 (strong)
- WHIP = 1.50 → factor = 1.05 (+5% runs)
- WHIP = 1.10 → factor = 0.95 (-5% runs)

### 2️⃣ **Strikeout Factor**
High K% = fewer balls in play
```python
k_factor = 1 - (K% - 0.22) × 0.6
```
**Weight**: 0.6 (very strong)
- K% = 30% → factor = 0.952 (-4.8% runs)
- K% = 15% → factor = 1.042 (+4.2% runs)

### 3️⃣ **LOB% (Strand Rate) Factor**
Pitcher strands more = fewer runs
```python
lob_factor = 1 - (LOB% - 0.72) × 0.7
```
**Weight**: 0.7 (strongest)
- LOB% = 80% → factor = 0.944 (-5.6% runs)
- LOB% = 65% → factor = 1.049 (+4.9% runs)

### 4️⃣ **Walk Rate Factor**
More walks = more baserunners
```python
bb_factor = 1 + (BB% - 0.085) × 0.4
```
- BB% = 12% → factor = 1.014
- BB% = 5% → factor = 0.986

### 5️⃣ **xFIP Factor**
Overall run prevention ability
```python
xfip_factor = 1 - (xFIP - 4.00) × 0.05
```
- xFIP = 3.00 → factor = 1.05
- xFIP = 5.00 → factor = 0.95

### **Combined Pitcher Factor**:
```python
pitcher_factor = (whip_factor × k_factor × lob_factor × bb_factor × xfip_factor)
```

## Lineup Context Factor

### Lineup Support (from next 2 batters)
Having good hitters behind you = they drive you in
```python
lineup_support_factor = avg(OBP_next_two) / 0.320  # computed in run_metrics_fetcher
lineup_factor = 1 + (lineup_support_factor - 1.0) × 0.6
```

**Example**:
- Next 2 batters: OBP = 0.360, 0.380
- avg = 0.370
- lineup_support_factor = 0.370 / 0.320 = 1.156
- lineup_factor = 1 + (1.156 - 1.0) × 0.6 = 1.094 (+9.4%)

## Lineup Position Factor

Leadoff hitters get more PAs = more opportunities

| Position | Weight | Rationale |
|----------|--------|-----------|
| **1 (Leadoff)** | 1.10 | Most PAs, sets table |
| **2** | 1.05 | 2nd most PAs |
| **3-7** | 0.95-1.00 | Balanced |
| **8-9** | 1.05 | Turns lineup over |

```python
position_factor = lineup_position_weights[batting_order]
```

## Platoon Multiplier

Favorable matchups boost run scoring

```python
if batter_side == pitcher_throws:
    platoon_multiplier = 0.95  # Same-handed penalty
elif batter_side == 'S':
    platoon_multiplier = 1.02  # Switch hitter advantage
else:
    platoon_multiplier = 1.06  # Opposite-handed boost
```

**Examples**:
- LHB vs RHP: 1.06 (+6%)
- RHB vs LHP: 1.06 (+6%)
- LHB vs LHP: 0.95 (-5%)
- Switch: 1.02 (+2%)

## Park/Weather Multiplier

Applied from `weather_ballpark_run_fetcher.py`:
```python
run_multiplier = total_run_multiplier  # 0.75 to 1.35
```

Includes temperature, wind, humidity, pressure, park factor

## Final Calculation

```python
base_run_prob = (league_run_rate ×
                 batter_factor ×
                 pitcher_factor ×
                 lineup_factor ×
                 position_factor ×
                 run_multiplier)

final_run_prob = base_run_prob × platoon_multiplier

# Capped: 1% to 50%
run_prob = max(0.01, min(0.50, run_prob))
```

## Example Calculation

### Scenario: Leadoff Speedster
**Player**: Speedy McFast (Leadoff, Switch)
- OBP: 0.380 → obp_factor = 1.117
- BSR: +5.0 → bsr_factor = 1.25
- SB Rate: 85% → sb_factor = 1.06
- Contact: 82% → contact_factor = 1.06
- SLG: 0.420 → power_factor = 1.007
- LD%: 23% → ld_factor = 1.01
- BB%: 10% → walk_factor = 1.005
- **Lineup Support**: 1.12 → lineup_factor = 1.072
- **Position**: #1 → position_factor = 1.10

**Pitcher**: Jacob deGrom (RHP)
- WHIP: 1.05 → whip_factor = 0.9375
- K%: 32% → k_factor = 0.94
- LOB%: 78% → lob_factor = 0.958
- BB%: 5% → bb_factor = 0.986
- xFIP: 2.80 → xfip_factor = 1.06

**Environment**:
- Park/Weather: 1.08 (warm, Coors-like)
- Platoon: Switch vs RHP = 1.02

**Calculation**:
```python
batter_factor = 1.117 × 1.25 × 1.06 × 1.06 × 1.007 × 1.01 × 1.005 = 1.68
pitcher_factor = 0.9375 × 0.94 × 0.958 × 0.986 × 1.06 = 0.89
lineup_factor = 1.072
position_factor = 1.10
run_multiplier = 1.08
platoon_multiplier = 1.02

base_run_prob = 0.15 × 1.68 × 0.89 × 1.072 × 1.10 × 1.08 = 0.328
final_run_prob = 0.328 × 1.02 = 0.335 (33.5%)
```

**Result**: 33.5% chance to score a run

## Output Format

### Console Display:
```
==========================================================================================
TOP 10 RUN SCORED PREDICTIONS
==========================================================================================
Player Name          | Team | Order | vs Pitcher           | Opp  | Run Prob
------------------------------------------------------------------------------------------
Speedy McFast        | COL  | #1    | Jacob deGrom         | NYM  |   33.5%
Mookie Betts         | LAD  | #1    | Blake Snell          | SD   |   31.2%
...
```

### JSON Output (`run_predictions.json`):
```json
{
  "date": "2025-05-12T14:30:00",
  "total_predictions": 216,
  "predictions": [
    {
      "player_name": "Speedy McFast",
      "team": "COL",
      "batting_order": 1,
      "opposing_pitcher": "Jacob deGrom",
      "opposing_team": "NYM",
      "run_probability": 33.5,
      "game_key": "COL_vs_NYM"
    }
  ]
}
```

## Usage

```bash
# Run the predictor
python master_run_predictor.py

# Input: final_integrated_run_data.json
# Output: run_predictions.json + console display
```

## Key Differences: Runs vs RBIs

| Metric | RBI Model | Run Model |
|--------|-----------|-----------|
| **Most Important** | RISP AVG, Power | **OBP, Speed** |
| **Speed Weight** | Low | **Very High** |
| **Lineup Context** | Previous hitters' OBP | **Next hitters' OBP** |
| **Position Weight** | Cleanup = 1.20x | **Leadoff = 1.10x** |
| **Baserunning** | Minor | **Critical (BSR, SB%)** |
| **Pitcher Focus** | LOB%, Barrel% | **WHIP, K%, LOB%** |

## Validation Checklist

✅ All players have run probabilities (1-50%)
✅ Leadoff hitters generally higher than cleanup
✅ High-OBP speedsters ranked highest
✅ Poor baserunners penalized appropriately
✅ Lineup support factor applied
✅ Platoon multipliers correct
✅ Park/weather multiplier integrated
✅ JSON output saved successfully

## Expected Top Performers

High run probability players typically have:
- ✅ **High OBP** (0.380+)
- ✅ **Elite speed** (BSR +3 or higher)
- ✅ **Good SB success** (80%+)
- ✅ **Leadoff position** (#1 or #2)
- ✅ **Strong lineup support** (good hitters behind)
- ✅ **Favorable park** (Coors, Wrigley)

## Next Steps

Results can be used for:
- Daily fantasy sports (DFS) lineup optimization
- Betting props (player to score a run)
- ML model training data
- Game simulation inputs
