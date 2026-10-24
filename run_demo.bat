@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  acousticsim — AlienAnimal Demo
echo ============================================================
echo.

:: ── 1. Verify Python is available ───────────────────────────────────────
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found. Please install Python 3.8+ and
    echo         add it to your PATH, then re-run this script.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set PYVER=%%V
echo [OK] Python %PYVER% found.
echo.

:: ── 2. Check / install required packages ────────────────────────────────
echo Checking required packages...
echo.

set MISSING=0

call :check_pkg numpy        "numpy>=1.24.0"
call :check_pkg scipy        "scipy>=1.10.0"
call :check_pkg matplotlib   "matplotlib>=3.7.0"
call :check_pkg trimesh      "trimesh[easy]>=3.23.0"
call :check_pkg sklearn      "scikit-learn>=1.3.0"
call :check_pkg torch        "torch>=2.0.0"

if %MISSING% EQU 0 (
    echo All required packages are present.
) else (
    echo.
    echo [INFO] Installing missing packages via pip...
    python -m pip install --upgrade pip --quiet
    python -m pip install "numpy>=1.24.0" "scipy>=1.10.0" "matplotlib>=3.7.0" ^
        "trimesh[easy]>=3.23.0" "scikit-learn>=1.3.0" "torch>=2.0.0"
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] pip install failed. Check your internet connection or
        echo         run this script as Administrator.
        pause
        exit /b 1
    )
    echo.
    echo [OK] All packages installed successfully.
)

echo.
echo ============================================================
echo  Running simulation...
echo ============================================================
echo.

python alien_demo.py

echo.
if %ERRORLEVEL% EQU 0 (
    echo ============================================================
    echo  Done. Output images saved:
    echo    alien_pointcloud_scene.png
    echo    alien_pointcloud_surface.png
    echo ============================================================
) else (
    echo [ERROR] Demo failed with exit code %ERRORLEVEL%.
)

echo.
pause
exit /b %ERRORLEVEL%

:: ── Helper: check if a Python module imports successfully ────────────────
:check_pkg
set PKG_MOD=%~1
set PKG_INSTALL=%~2
python -c "import %PKG_MOD%" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [ MISSING ] %PKG_INSTALL%
    set MISSING=1
) else (
    echo   [   OK    ] %PKG_MOD%
)
exit /b 0
