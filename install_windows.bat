@echo off
echo ========================================
echo FitPulse 2.1.0 - Windows Installation
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)
echo.

echo Installing core dependencies...
pip install streamlit==1.31.0 pandas==2.2.0 numpy==1.26.3 plotly==5.18.0 scipy==1.11.4 pytz==2024.1 scikit-learn==1.4.0
if errorlevel 1 (
    echo ERROR: Failed to install core dependencies
    pause
    exit /b 1
)
echo.

echo Installing TSFresh (this may take a few minutes)...
pip install tsfresh==0.20.2
if errorlevel 1 (
    echo WARNING: TSFresh installation failed
    echo You can continue without it, but feature extraction won't work
)
echo.

echo Installing Prophet...
pip install prophet==1.1.5
if errorlevel 1 (
    echo WARNING: Prophet installation failed
    echo Try: conda install -c conda-forge prophet
)
echo.

echo Attempting to install HDBSCAN...
pip install hdbscan --prefer-binary
if errorlevel 1 (
    echo.
    echo ========================================
    echo HDBSCAN installation failed (expected on Windows)
    echo ========================================
    echo.
    echo This is OK! You can still use:
    echo   - All data preprocessing
    echo   - All visualizations
    echo   - TSFresh feature extraction
    echo   - Prophet forecasting
    echo   - K-Means clustering
    echo.
    echo To install HDBSCAN, you have 3 options:
    echo   1. Use Conda: conda install -c conda-forge hdbscan
    echo   2. Install C++ Build Tools (see WINDOWS_INSTALL_GUIDE.md)
    echo   3. Continue without HDBSCAN (recommended)
    echo.
)
echo.

echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To run FitPulse:
echo   streamlit run app.py
echo.
echo For troubleshooting, see:
echo   WINDOWS_INSTALL_GUIDE.md
echo.
pause
