"""Cho phép chạy `python -m app.cli ...` từ thư mục gốc của repo."""

import sys

from app.cli.main import main

sys.exit(main())
