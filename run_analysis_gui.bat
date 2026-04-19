@echo off
:: ─── EO Ground Station Analysis GUI launcher ────────────────────────────────
:: Place this file anywhere (e.g. Desktop).
:: Edit PROJECT_DIR below if you move the project.

set "PROJECT_DIR=d:\work\projects\tasking_util\code\py_helpers"
set "VENV_DIR=%PROJECT_DIR%\.venv"
set "ACTIVATE=%VENV_DIR%\Scripts\activate.bat"

if not exist "%ACTIVATE%" (
    echo ERROR: Virtual environment not found at %VENV_DIR%
    echo        Run install.bat first to set up the environment.
    pause
    exit /b 1
)

call "%ACTIVATE%"
python -m tasking_helper.gui.analysis_gui
if errorlevel 1 pause
