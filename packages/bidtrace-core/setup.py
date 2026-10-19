# SPDX-License-Identifier: MPL-2.0
"""Build the canonical selected modules; do not copy/fork the analysis engine."""
from pathlib import Path
import runpy
from setuptools import setup

ROOT = Path(__file__).resolve().parents[2]
manifest = runpy.run_path(str(ROOT / 'bidtrace_core/package_manifest.py'))
setup(py_modules=manifest['MODULES'], package_dir={'': '../..'}, include_package_data=False)
