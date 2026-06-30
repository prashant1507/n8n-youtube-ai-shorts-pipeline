"""Release PyTorch / MPS memory between pipeline stages."""

from __future__ import annotations

import gc
import logging
import os

logger = logging.getLogger(__name__)


def release_gpu_memory() -> None:
    """Best-effort free of cached GPU/unified memory after unloading models."""
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug("GPU cache release skipped: %s", exc)
