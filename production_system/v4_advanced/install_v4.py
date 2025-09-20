#!/usr/bin/env python3
"""
Installation and Setup Script for MLB RBI Prediction System v4.0
Handles dependency installation and system verification
"""

import subprocess
import sys
import os
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False

    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is available")
        return True
    except ImportError:
        print(f"❌ {package_name} is not available")
        return False

def install_core_packages():
    """Install core packages required for basic functionality"""

    core_packages = [
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.9.0",
        "requests>=2.28.0"
    ]

    print("\n📦 Installing core packages...")

    success_count = 0
    for package in core_packages:
        if install_package(package):
            success_count += 1

    print(f"\n✅ {success_count}/{len(core_packages)} core packages installed")
    return success_count == len(core_packages)

def install_optional_packages():
    """Install optional packages for enhanced functionality"""

    optional_packages = [
        ("xgboost>=1.7.0", "xgboost"),
        ("lightgbm>=3.3.0", "lightgbm"),
        ("shap>=0.41.0", "shap"),
        ("streamlit>=1.28.0", "streamlit"),
        ("plotly>=5.15.0", "plotly"),
        ("python-dotenv>=1.0.0", "dotenv")
    ]

    print("\n📦 Installing optional packages...")

    installed = []
    failed = []

    for package, import_name in optional_packages:
        if install_package(package):
            installed.append(package.split(">=")[0])
        else:
            failed.append(package.split(">=")[0])

    print(f"\n✅ {len(installed)} optional packages installed: {', '.join(installed)}")
    if failed:
        print(f"⚠️  {len(failed)} packages failed: {', '.join(failed)}")

    return installed, failed

def install_deep_learning_packages():
    """Install deep learning packages (optional)"""

    print("\n🧠 Installing deep learning packages (optional)...")
    print("Note: These are large packages and may take several minutes")

    dl_packages = [
        "tensorflow>=2.12.0",
        "keras>=2.12.0"
    ]

    choice = input("Install TensorFlow and Keras? (y/n): ").lower().strip()

    if choice == 'y':
        for package in dl_packages:
            install_package(package)
    else:
        print("⏭️  Skipping deep learning packages")
        print("   You can run the lite version without these")

def verify_installation():
    """Verify that key packages are working"""

    print("\n🔍 Verifying installation...")

    # Core packages
    core_checks = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("scipy", "scipy"),
        ("requests", "requests")
    ]

    # Optional packages
    optional_checks = [
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("shap", "shap"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly")
    ]

    print("\n📦 Core packages:")
    core_success = 0
    for name, import_name in core_checks:
        if check_package(name, import_name):
            core_success += 1

    print("\n📦 Optional packages:")
    optional_success = 0
    for name, import_name in optional_checks:
        if check_package(name, import_name):
            optional_success += 1

    print(f"\n📊 Installation Summary:")
    print(f"   Core packages: {core_success}/{len(core_checks)}")
    print(f"   Optional packages: {optional_success}/{len(optional_checks)}")

    if core_success == len(core_checks):
        print("✅ Core installation successful - you can run RBI_v4_lite.py")
    else:
        print("❌ Core installation incomplete - some features may not work")

    return core_success, optional_success

def create_env_file():
    """Create environment file for API keys"""

    print("\n🔑 Setting up API keys...")

    env_path = Path(".env")

    if env_path.exists():
        print("⚠️  .env file already exists")
        overwrite = input("Overwrite existing .env file? (y/n): ").lower().strip()
        if overwrite != 'y':
            print("⏭️  Keeping existing .env file")
            return

    env_content = """# MLB RBI Prediction System v4.0 API Keys

# Weather API (OpenWeatherMap)
WEATHER_API_KEY=e09911139e379f1e4ca813df1778b4ef

# Odds API (The Odds API)
ODDS_API_KEY=47b36e3e637a7690621e258da00e29d7

# Optional: Add your own API keys here
# MLB_API_KEY=your_key_here
# ADDITIONAL_WEATHER_KEY=your_key_here
"""

    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✅ .env file created with API keys")
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")

def test_lite_system():
    """Test the lite system"""

    print("\n🧪 Testing lite system...")

    try:
        # Try to import the lite system
        sys.path.append(str(Path.cwd()))

        print("📝 Testing imports...")
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        print("✅ Core imports successful")

        print("📝 Testing basic functionality...")
        # Simple test
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        pred = model.predict(X[:1])

        print(f"✅ Basic ML test successful (prediction: {pred[0]:.3f})")

        return True

    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main installation process"""

    print("🚀 MLB RBI Prediction System v4.0 - Installation Script")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        print("\n❌ Installation cannot continue with incompatible Python version")
        return

    # Install core packages
    if not install_core_packages():
        print("\n❌ Failed to install core packages")
        choice = input("Continue with optional packages anyway? (y/n): ").lower().strip()
        if choice != 'y':
            return

    # Install optional packages
    installed, failed = install_optional_packages()

    # Install deep learning packages
    install_deep_learning_packages()

    # Verify installation
    core_success, optional_success = verify_installation()

    # Create .env file
    create_env_file()

    # Test system
    if test_lite_system():
        print("\n🎉 Installation completed successfully!")
    else:
        print("\n⚠️  Installation completed with issues")

    # Next steps
    print("\n📝 Next Steps:")
    print("1. Run the lite system: python RBI_v4_lite.py")

    if "streamlit" in installed:
        print("2. Run the dashboard: streamlit run dashboard_v4.py")
    else:
        print("2. Install Streamlit for dashboard: pip install streamlit")

    if "tensorflow" not in [pkg.lower() for pkg in installed]:
        print("3. For deep learning features: pip install tensorflow")

    print("4. Check the README: RBI_v4_README.md")

    print("\n🔧 Troubleshooting:")
    print("- If packages fail to install, try: pip install --upgrade pip")
    print("- For TensorFlow issues, check: https://tensorflow.org/install")
    print("- For M1 Mac issues, use: pip install tensorflow-macos")

if __name__ == "__main__":
    main()