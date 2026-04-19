@echo off
:: ─── tasking_helper install script ───────────────────────────────────────────
:: Creates a virtual environment, installs all dependencies, builds a wheel,
:: and installs it into the venv.
::
:: Usage:  install.bat
:: Optional: install.bat --no-gui   (skip PyQt6 / matplotlib)

setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
:: Strip trailing backslash
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

set "VENV_DIR=%PROJECT_DIR%\.venv"
set "DIST_DIR=%PROJECT_DIR%\dist"
set "NO_GUI=0"
if /i "%1"=="--no-gui" set "NO_GUI=1"

echo.
echo ============================================================
echo  tasking_helper installer
echo  Project : %PROJECT_DIR%
echo  Venv    : %VENV_DIR%
echo ============================================================
echo.

:: ── 1. Locate Python ─────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found on PATH.
    echo        Install Python 3.9+ and ensure it is on PATH.
    goto :fail
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo Using: %PY_VER%
echo.

:: ── 2. Create virtual environment ────────────────────────────────────────────
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment already exists — skipping creation.
) else (
    echo Creating virtual environment ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        goto :fail
    )
    echo Done.
)
echo.

:: ── 3. Activate and upgrade pip ──────────────────────────────────────────────
call "%VENV_DIR%\Scripts\activate.bat"
echo Upgrading pip ...
python -m pip install --upgrade pip --quiet
echo.

:: ── 4. Install core runtime dependencies ─────────────────────────────────────
echo Installing core dependencies (numpy) ...
pip install numpy --quiet
if errorlevel 1 goto :fail

:: ── 5. Install optional extras ───────────────────────────────────────────────
if "%NO_GUI%"=="0" (
    echo Installing GUI dependencies (PyQt6, matplotlib) ...
    pip install PyQt6 matplotlib --quiet
    if errorlevel 1 (
        echo WARNING: GUI dependencies failed. The GUI will not be available.
    )
)

echo Installing validation dependencies (sgp4, astropy) ...
pip install sgp4 astropy --quiet

echo Installing viz dependencies (plotly) ...
pip install plotly --quiet
echo.

:: ── 6. Install build tools ───────────────────────────────────────────────────
echo Installing build tools (hatchling, build) ...
pip install hatchling build --quiet
if errorlevel 1 goto :fail
echo.

:: ── 7. Build wheel ───────────────────────────────────────────────────────────
echo Building wheel ...
if exist "%DIST_DIR%" rd /s /q "%DIST_DIR%"
python -m build --wheel --outdir "%DIST_DIR%" "%PROJECT_DIR%"
if errorlevel 1 (
    echo ERROR: Wheel build failed.
    goto :fail
)
echo.

:: ── 8. Install wheel ─────────────────────────────────────────────────────────
echo Installing wheel into venv ...
for %%f in ("%DIST_DIR%\*.whl") do (
    pip install "%%f" --force-reinstall --quiet
    if errorlevel 1 (
        echo ERROR: Failed to install %%f
        goto :fail
    )
    echo Installed: %%~nxf
)
echo.

:: ── 9. Verify ────────────────────────────────────────────────────────────────
echo Verifying installation ...
python -c "import tasking_helper; print('  tasking_helper imported OK')"
if errorlevel 1 (
    echo ERROR: Import check failed.
    goto :fail
)
echo.
echo ============================================================
echo  Installation complete!
echo  Run the GUI with:  run_analysis_gui.bat
echo ============================================================
echo.
goto :end

:fail
echo.
echo ============================================================
echo  Installation FAILED. See errors above.
echo ============================================================
echo.
pause
exit /b 1

:end
pause
endlocal
