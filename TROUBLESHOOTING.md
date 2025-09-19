# 🔧 Troubleshooting Guide - MLB RBI Prediction System v4.0

## 🚨 Common Issues & Solutions

### **Issue: `ModuleNotFoundError: No module named 'tensorflow'`**

**Solution Options:**

1. **Use the Lite Version (Recommended)**
   ```bash
   python RBI_v4_lite.py
   ```
   - Runs without TensorFlow
   - Includes all core features except deep learning

2. **Install TensorFlow**
   ```bash
   pip install tensorflow>=2.12.0
   ```

3. **Use Installation Script**
   ```bash
   python install_v4.py
   ```

---

### **Issue: Package Installation Failures**

**Solutions:**

1. **Upgrade pip first**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install packages individually**
   ```bash
   pip install numpy pandas scikit-learn
   pip install xgboost lightgbm
   pip install streamlit plotly
   ```

3. **Use conda (alternative)**
   ```bash
   conda install numpy pandas scikit-learn
   conda install -c conda-forge xgboost lightgbm
   ```

---

### **Issue: `ImportError` for specific packages**

**Check what's missing:**
```bash
python -c "import numpy; print('NumPy OK')"
python -c "import pandas; print('Pandas OK')"
python -c "import sklearn; print('Scikit-learn OK')"
```

**Install missing packages:**
```bash
pip install [missing_package_name]
```

---

### **Issue: API Key Errors**

**Symptoms:**
- Weather data not loading
- Odds data unavailable

**Solutions:**

1. **Check .env file exists**
   ```bash
   # Create .env file with:
   WEATHER_API_KEY=e09911139e379f1e4ca813df1778b4ef
   ODDS_API_KEY=47b36e3e637a7690621e258da00e29d7
   ```

2. **Verify API keys are working**
   ```python
   from RBI_v4_lite import EnhancedMLBDataFetcher
   fetcher = EnhancedMLBDataFetcher()
   # Test weather fetch
   ```

---

### **Issue: Database Errors**

**Symptoms:**
- SQLite errors
- Permission denied

**Solutions:**

1. **Check file permissions**
   ```bash
   # Make sure directory is writable
   ```

2. **Initialize database manually**
   ```python
   from database_schema_v4 import DatabaseManagerV4
   db = DatabaseManagerV4()
   ```

3. **Delete and recreate database**
   ```bash
   # Delete *.db files and restart
   ```

---

### **Issue: Streamlit Dashboard Won't Start**

**Error:** `streamlit: command not found`

**Solution:**
```bash
pip install streamlit
streamlit run dashboard_v4.py
```

**Alternative:** Use Python module
```bash
python -m streamlit run dashboard_v4.py
```

---

### **Issue: Performance/Memory Issues**

**Symptoms:**
- Slow predictions
- High memory usage

**Solutions:**

1. **Reduce model complexity**
   ```python
   # In RBI_v4_lite.py, reduce n_estimators
   rf_model = RandomForestRegressor(n_estimators=50)  # Instead of 100
   ```

2. **Use smaller datasets**
   ```python
   # Reduce sample size for testing
   n_samples = 500  # Instead of 1000
   ```

3. **Clear cache regularly**
   ```python
   # Delete cache database periodically
   ```

---

## 🛠️ **System Requirements Check**

### **Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 1GB disk space
- Internet connection (for APIs)

### **Recommended:**
- Python 3.9+
- 8GB RAM
- 2GB disk space
- Stable internet

### **Check Your System:**
```python
import sys
import platform
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")
```

---

## 📦 **Package Version Compatibility**

### **Known Working Combinations:**

**Minimal Setup:**
```
numpy==1.21.6
pandas==1.5.3
scikit-learn==1.2.2
```

**Full Setup:**
```
numpy==1.21.6
pandas==1.5.3
scikit-learn==1.2.2
xgboost==1.7.4
lightgbm==3.3.5
streamlit==1.28.1
plotly==5.17.0
```

**With Deep Learning:**
```
[Above packages] +
tensorflow==2.12.0
keras==2.12.0
```

---

## 🔍 **Debugging Steps**

### **1. Test Core Functionality**
```bash
python -c "
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
print('✅ Core packages working')
"
```

### **2. Test Weather API**
```python
from RBI_v4_lite import EnhancedMLBDataFetcher
from datetime import datetime

fetcher = EnhancedMLBDataFetcher()
weather = fetcher.fetch_enhanced_weather(33.8003, -117.8827, datetime.now())
print(f"Temperature: {weather.temp_f}°F")
```

### **3. Test ML Models**
```python
from RBI_v4_lite import LiteMLModels
import numpy as np

models = LiteMLModels()
X = np.random.rand(100, 10)
y = np.random.rand(100)
models.train_models(X, y, [f'feature_{i}' for i in range(10)])
print("✅ ML models working")
```

---

## 🆘 **Getting Help**

### **Before Reporting Issues:**

1. ✅ Run the installation script: `python install_v4.py`
2. ✅ Try the lite version: `python RBI_v4_lite.py`
3. ✅ Check this troubleshooting guide
4. ✅ Verify your Python version: `python --version`

### **When Reporting Issues:**

Include this information:
```python
import sys
import platform
import pkg_resources

print("System Info:")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")

print("\nInstalled Packages:")
for pkg in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm']:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"{pkg}: {version}")
    except:
        print(f"{pkg}: NOT INSTALLED")
```

---

## 🎯 **Quick Fixes Summary**

| **Problem** | **Quick Fix** |
|-------------|---------------|
| TensorFlow missing | Use `RBI_v4_lite.py` |
| Package errors | Run `python install_v4.py` |
| API errors | Check `.env` file |
| Database errors | Delete `.db` files |
| Streamlit issues | `pip install streamlit` |
| Memory issues | Reduce model size |

---

## ✅ **Success Checklist**

- [ ] Python 3.8+ installed
- [ ] Core packages installed (numpy, pandas, sklearn)
- [ ] RBI_v4_lite.py runs without errors
- [ ] Weather API returns data
- [ ] ML models can train and predict
- [ ] Database initializes successfully

**If all boxes are checked, your system is ready! 🚀**