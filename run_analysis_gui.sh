#!/usr/bin/env bash
# ─── EO Ground Station Analysis GUI launcher ─────────────────────────────────
# Place this file anywhere (e.g. ~/Desktop).
# Edit PROJECT_DIR below if you move the project.

PROJECT_DIR="$HOME/work/projects/tasking_util/code/py_helpers"
VENV_DIR="$PROJECT_DIR/.venv"
ACTIVATE="$VENV_DIR/bin/activate"

if [ ! -f "$ACTIVATE" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "       Run install.sh first to set up the environment."
    exit 1
fi

source "$ACTIVATE"
exec python -m tasking_helper.gui.analysis_gui "$@"
