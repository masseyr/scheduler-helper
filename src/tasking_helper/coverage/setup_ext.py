"""
Build the optional C extension _gap_ext for tasking_helper.coverage.

Usage (run from this directory or the repo root):
    python src/tasking_helper/coverage/setup_ext.py build_ext --inplace

The resulting .pyd / .so is placed next to this file and picked up
automatically when tasking_helper.coverage is imported.
"""

import sys
from pathlib import Path

import numpy as np
from setuptools import Extension, setup

HERE = Path(__file__).parent

ext = Extension(
    name="tasking_helper.coverage._gap_ext",
    sources=[str(HERE / "_gap_ext.c")],
    include_dirs=[np.get_include()],
    extra_compile_args=(
        ["/O2", "/W3"] if sys.platform == "win32" else ["-O3", "-Wall", "-Wextra"]
    ),
    language="c",
)

setup(
    name="tasking-helper-gap-ext",
    version="0.1.0",
    description="Fast gap-analysis C extension for tasking_helper.coverage",
    ext_modules=[ext],
)
