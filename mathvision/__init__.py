"""Import shim for running the src-layout package from the repo root.

The real package lives under ``src/mathvision``. This shim makes commands like
``python -m mathvision.app.gradio_app`` work before editable installation.
"""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "mathvision"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))  # type: ignore[name-defined]

__version__ = "0.1.0"
