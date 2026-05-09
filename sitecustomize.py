"""Make the src-layout package importable from the repository root.

This keeps commands such as ``python -m mathvision.app.gradio_app`` working
after installing requirements, even before running ``pip install -e .``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
