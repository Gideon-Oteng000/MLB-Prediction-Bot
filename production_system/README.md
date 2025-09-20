# MLB Production Betting System

## Folder Structure

```
production_system/
├── production_system.py          # Main production system
├── models/
│   └── production_mlb_model.pkl  # Trained RandomForest model
├── data/
│   └── historical_mlb_games.csv  # Historical games database
├── predictions/
│   └── predictions_YYYYMMDD.csv  # Daily prediction outputs
├── cache/
│   └── *.db                      # Cache and database files
├── config/
│   ├── .env                      # API keys
│   └── quick_start.bat           # Windows startup script
├── v4_advanced/
│   ├── RBI_v4.py                 # Advanced RBI system
│   ├── RBI_v4_lite.py            # Lite version
│   ├── dashboard_v4.py           # Interactive dashboard
│   └── ...                       # Other v4 files
└── docs/
    ├── README.md                 # This file
    ├── RBI_v4_README.md          # v4 documentation
    └── TROUBLESHOOTING.md        # Troubleshooting guide
```

## Quick Start

### Run Production System
```bash
cd production_system
python production_system.py
```

### Run v4 Advanced System
```bash
cd production_system/v4_advanced
python RBI_v4_lite.py
```

### Run Interactive Dashboard
```bash
cd production_system/v4_advanced
streamlit run dashboard_v4.py
```

## What Each System Does

### Production System (production_system.py)
- 52.6% accuracy on historical data
- Auto-updates with recent completed games
- Daily predictions for scheduled games
- Model persistence with .pkl files
- Simple, reliable betting recommendations

**Output Files:**
- models/production_mlb_model.pkl - Trained model
- data/historical_mlb_games.csv - Updated historical data
- predictions/predictions_YYYYMMDD.csv - Daily predictions

### v4 Advanced System (v4_advanced/)
- Real weather data integration
- Real betting odds with vig analysis
- Multiple ML models (XGBoost, LightGBM, etc.)
- Kelly Criterion bankroll management
- Monte Carlo simulation
- Interactive dashboard with visualizations

## Setup Requirements

### For Production System:
```bash
pip install pandas numpy requests scikit-learn joblib
```

### For v4 Advanced System:
```bash
pip install -r v4_advanced/requirements_v4.txt
```

Or use the installer:
```bash
cd v4_advanced
python install_v4.py
```

## Usage Examples

### Daily Betting Workflow
1. Morning: Run production system for today's games
2. Research: Check v4 system for detailed analysis
3. Betting: Use Kelly Criterion for bet sizing
4. Evening: Update with completed games

### Files Generated Daily
- predictions_20241219.csv - Today's game predictions
- Updated historical_mlb_games.csv - Fresh historical data
- Updated production_mlb_model.pkl - Retrained model (if needed)

## Performance Tracking

### Production System Accuracy
- Training Accuracy: ~52.6%
- Sample Size: 1500+ historical games
- Model Type: RandomForest (50 estimators)
- Features: Team win rates, run differentials, home field advantage

### v4 System Capabilities
- Enhanced Features: Weather, odds, player splits
- Multiple Models: Ensemble predictions
- Risk Management: Kelly Criterion optimization
- ROI Simulation: Monte Carlo analysis

## File Dependencies

### Required for Production System:
- historical_mlb_games.csv - Historical games data
- production_mlb_model.pkl - Trained model (auto-generated)

### Auto-Generated Files:
- predictions_YYYYMMDD.csv - Daily predictions
- production_mlb_model.pkl - Model file (when retrained)

### v4 System Dependencies:
- .env - API keys for weather/odds
- mlb_cache_v4.db - API response cache
- rbi_predictions_v4.db - Enhanced predictions database

## Troubleshooting

### Common Issues:
1. Missing historical data: Download from original source
2. API rate limits: Use cache and respect delays
3. Model retraining: Automatically triggered by new data
4. Package errors: See docs/TROUBLESHOOTING.md

### Quick Fixes:
```bash
# Reinstall packages
pip install --upgrade pandas numpy scikit-learn

# Reset model (forces retrain)
rm models/production_mlb_model.pkl

# Reset cache
rm cache/*.db
```

## Support

- Production System: Basic, reliable betting predictions
- v4 System: Advanced analytics and risk management
- Documentation: See docs/ folder
- Installation Help: Run v4_advanced/install_v4.py

---

Last Updated: 2025-09-19 18:03:31
System Status: Operational
