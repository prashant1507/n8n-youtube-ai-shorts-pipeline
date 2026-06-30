"""Background music generation via Transformers MusicGen (MPS/CUDA/CPU)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import torch

from .config import PipelineConfig

logger = logging.getLogger(__name__)

_model = None
_processor = None


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(config: PipelineConfig):
    global _model, _processor
    if _model is not None:
        return _model, _processor

    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    model_id = config.music.model
    logger.info("Loading music model %s", model_id)
    _processor = AutoProcessor.from_pretrained(model_id)
    _model = MusicgenForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation="eager",
    )
    _model = _model.to(_device())
    return _model, _processor


def generate_music(
    prompt: str,
    duration_sec: float,
    config: PipelineConfig,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, processor = _load_model(config)
    device = _device()

    # MusicGen max token length ~512 tokens ≈ 30s at 50Hz; cap per pass
    max_chunk = 30.0
    chunks: list[np.ndarray] = []
    remaining = duration_sec
    sr = model.config.audio_encoder.sampling_rate

    while remaining > 0:
        chunk_dur = min(remaining, max_chunk)
        max_new_tokens = int(chunk_dur * 50)
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            audio = model.generate(**inputs, max_new_tokens=max_new_tokens)
        arr = audio[0, 0].cpu().numpy()
        chunks.append(arr)
        remaining -= chunk_dur

    combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    target_samples = int(duration_sec * sr)
    combined = combined[:target_samples]

    wavfile.write(str(output_path), sr, combined)
    logger.info("Saved music to %s (%.1fs)", output_path, len(combined) / sr)
    return output_path


def unload_model() -> None:
    global _model, _processor
    if _model is not None:
        del _model
    if _processor is not None:
        del _processor
    _model = _processor = None
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
