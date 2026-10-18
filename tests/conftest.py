"""Cấu hình pytest: gốc repo (package `src`, `app`) và tests/helpers (shell_env, colmap_writer) vào sys.path.

Chạy từ gốc repo: `python -m pytest` (mọi test) hoặc `python -m pytest tests/test_pipeline.py`.
"""

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
for path in (TESTS.parent, TESTS / "helpers"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
