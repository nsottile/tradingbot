"""Compatibility entrypoint for the new Streamlit UI package."""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]

if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from polymarket_alpha.ui.app import render_app

render_app()
