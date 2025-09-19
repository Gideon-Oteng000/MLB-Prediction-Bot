@echo off
echo.
echo ========================================
echo   MLB RBI Prediction System v4.0
echo   Quick Start Script
echo ========================================
echo.

echo Step 1: Installing core dependencies...
python install_v4.py

echo.
echo Step 2: Running lite system test...
python RBI_v4_lite.py

echo.
echo ========================================
echo   Quick Start Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Check the output above for any errors
echo 2. If successful, you can run the full system
echo 3. For dashboard: streamlit run dashboard_v4.py
echo.
pause