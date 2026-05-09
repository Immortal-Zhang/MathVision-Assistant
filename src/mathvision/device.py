"""Device selection helpers."""

from __future__ import annotations


def get_best_device() -> str:
    """Return the best available torch device name.

    Priority is CUDA > Apple MPS > CPU. Torch is imported lazily so default
    command-line tools can still show helpful errors if torch is missing.
    """

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
