#!/usr/bin/env python3
"""
Simple File Organization Script for MLB Production System
Moves all related files to organized folder structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def organize_production_files():
    """Organize all production system files into proper structure"""

    print("Organizing MLB Production System Files")
    print("=" * 50)

    # Current directory
    current_dir = Path.cwd()

    # Files that production_system.py creates or uses
    production_files = [
        # Main system files
        'production_system.py',

        # Data files (CSV)
        'historical_mlb_games.csv',

        # Model files (PKL)
        'production_mlb_model.pkl',

        # Related v4 files we created
        'RBI_v4.py',
        'RBI_v4_lite.py',
        'dashboard_v4.py',
        'database_schema_v4.py',
        'install_v4.py',
        'requirements_v4.txt',
        'RBI_v4_README.md',
        'TROUBLESHOOTING.md',

        # Configuration files
        '.env',
        'quick_start.bat',

        # Previous versions
        'RBI.py',
        'test_real_data_pipeline.py'
    ]

    # Check which files exist
    existing_files = []
    missing_files = []

    print("\nChecking for production system files...")

    for file_name in production_files:
        if (current_dir / file_name).exists():
            existing_files.append(file_name)
            print(f"  Found: {file_name}")
        else:
            missing_files.append(file_name)
            print(f"  Missing: {file_name}")

    # Check for pattern files
    pattern_files = {
        'predictions_*.csv': 'Daily prediction files',
        '*.db': 'Database files'
    }

    for pattern, description in pattern_files.items():
        matching_files = list(current_dir.glob(pattern))
        if matching_files:
            for file in matching_files:
                existing_files.append(file.name)
                print(f"  Found: {file.name} ({description})")

    print(f"\nSummary:")
    print(f"  Found: {len(existing_files)} files")
    print(f"  Missing: {len(missing_files)} files")

    # Create organized folder structure
    print(f"\nCreating organized folder structure...")

    folders = [
        'production_system',
        'production_system/models',
        'production_system/data',
        'production_system/predictions',
        'production_system/config',
        'production_system/v4_advanced',
        'production_system/cache',
        'production_system/docs'
    ]

    for folder_path in folders:
        folder = current_dir / folder_path
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {folder_path}")

    # File organization mapping
    file_mapping = {
        # Main system files
        'production_system.py': 'production_system/',

        # Model files
        'production_mlb_model.pkl': 'production_system/models/',

        # Data files
        'historical_mlb_games.csv': 'production_system/data/',

        # v4 System files
        'RBI_v4.py': 'production_system/v4_advanced/',
        'RBI_v4_lite.py': 'production_system/v4_advanced/',
        'dashboard_v4.py': 'production_system/v4_advanced/',
        'database_schema_v4.py': 'production_system/v4_advanced/',
        'install_v4.py': 'production_system/v4_advanced/',
        'requirements_v4.txt': 'production_system/v4_advanced/',
        'test_real_data_pipeline.py': 'production_system/v4_advanced/',
        'RBI.py': 'production_system/v4_advanced/',

        # Documentation
        'RBI_v4_README.md': 'production_system/docs/',
        'TROUBLESHOOTING.md': 'production_system/docs/',

        # Configuration
        '.env': 'production_system/config/',
        'quick_start.bat': 'production_system/config/',
    }

    # Move/copy files to organized structure
    print(f"\nOrganizing files into folder structure...")

    moved_count = 0
    for file_name, target_folder in file_mapping.items():
        source_file = current_dir / file_name
        target_path = current_dir / target_folder

        if source_file.exists():
            target_file = target_path / source_file.name
            try:
                shutil.copy2(source_file, target_file)
                print(f"  Copied: {file_name} -> {target_folder}")
                moved_count += 1
            except Exception as e:
                print(f"  Error copying {file_name}: {e}")

    # Handle pattern files
    print(f"\nCopying pattern files...")

    # Prediction files
    prediction_files = list(current_dir.glob('predictions_*.csv'))
    for file in prediction_files:
        target_file = current_dir / 'production_system/predictions' / file.name
        try:
            shutil.copy2(file, target_file)
            print(f"  Copied: {file.name} -> production_system/predictions/")
            moved_count += 1
        except Exception as e:
            print(f"  Error copying {file.name}: {e}")

    # Database files
    db_files = list(current_dir.glob('*.db'))
    for file in db_files:
        target_file = current_dir / 'production_system/cache' / file.name
        try:
            shutil.copy2(file, target_file)
            print(f"  Copied: {file.name} -> production_system/cache/")
            moved_count += 1
        except Exception as e:
            print(f"  Error copying {file.name}: {e}")

    # Create a master README for the production system
    create_production_readme(current_dir / 'production_system')

    # Create a run script for the production system
    create_run_script(current_dir / 'production_system')

    print(f"\nOrganization complete!")
    print(f"  Files organized: {moved_count}")
    print(f"  Main folder: ./production_system/")
    print(f"  Run production system: cd production_system && python production_system.py")

def create_production_readme(production_folder):
    """Create a comprehensive README for the production system"""

    readme_content = f"""# MLB Production Betting System

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

Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System Status: Operational
"""

    readme_path = production_folder / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"  Created: README.md")

def create_run_script(production_folder):
    """Create a convenient run script"""

    script_content = """#!/usr/bin/env python3
# Quick run script for MLB Production System

import os
import sys
from pathlib import Path

def main():
    print("MLB Production System Launcher")
    print("=" * 40)
    print()
    print("Choose an option:")
    print("1. Run Production System (Daily Predictions)")
    print("2. Run v4 Lite System (Advanced Analytics)")
    print("3. Run v4 Dashboard (Interactive Interface)")
    print("4. Install v4 Dependencies")
    print("5. Exit")
    print()

    choice = input("Enter choice (1-5): ").strip()

    if choice == '1':
        print("\\nRunning Production System...")
        os.system('python production_system.py')

    elif choice == '2':
        print("\\nRunning v4 Lite System...")
        os.chdir('v4_advanced')
        os.system('python RBI_v4_lite.py')

    elif choice == '3':
        print("\\nStarting v4 Dashboard...")
        os.chdir('v4_advanced')
        os.system('streamlit run dashboard_v4.py')

    elif choice == '4':
        print("\\nInstalling v4 Dependencies...")
        os.chdir('v4_advanced')
        os.system('python install_v4.py')

    elif choice == '5':
        print("\\nGoodbye!")
        sys.exit(0)

    else:
        print("\\nInvalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()
"""

    script_path = production_folder / 'run.py'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"  Created: run.py (launcher script)")

if __name__ == "__main__":
    organize_production_files()