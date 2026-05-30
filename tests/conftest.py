"""Pytest configuration for adding project root to PYTHONPATH.

Some test runners execute with a different working directory which can
prevent local packages (like `src`) from being importable. This helper
ensures the repository root is on `sys.path` during tests.
"""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
