"""Text-to-speech using indic-parler-tts (Mary for English, Rani for Hindi)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# parler-tts 0.2.2 requires transformers 4.46.x; shim if a newer transformers is installed
try:
    from transformers.pytorch_utils import isin_mps_friendly  # noqa: F401
except ImportError:
    import transformers.pytorch_utils as _ptu

    def _isin_mps_friendly(elements, test_elements):
        if elements.device.type == "mps" and test_elements.device.type != "mps":
            test_elements = test_elements.to(elements.device)
        return torch.isin(elements, test_elements)

    _ptu.isin_mps_friendly = _isin_mps_friendly  # type: ignore[attr-defined]

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

from .config import PipelineConfig

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_description_tokenizer = None


def _device(config: PipelineConfig) -> str:
    preferred = config.voice.device
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(config: PipelineConfig):
    global _model, _tokenizer, _description_tokenizer
    if _model is not None:
        return _model, _tokenizer, _description_tokenizer

    device = _device(config)
    logger.info("Loading TTS model %s on %s", config.voice.model, device)
    _model = ParlerTTSForConditionalGeneration.from_pretrained(config.voice.model)
    _model = _model.to(device)
    _tokenizer = AutoTokenizer.from_pretrained(config.voice.model)
    _description_tokenizer = AutoTokenizer.from_pretrained(
        _model.config.text_encoder._name_or_path
    )
    return _model, _tokenizer, _description_tokenizer


def _generate_chunk(
    text: str,
    description: str,
    config: PipelineConfig,
) -> np.ndarray:
    model, tokenizer, description_tokenizer = _load_model(config)
    device = _device(config)

    desc = description_tokenizer(description, return_tensors="pt").to(device)
    prompt = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        generation = model.generate(
            input_ids=desc.input_ids,
            attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )

    audio = generation.cpu().numpy().squeeze()
    return audio.astype(np.float32)


def _concat_wavs(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        raise ValueError("No audio chunks to concatenate")
    return np.concatenate(chunks)


def generate_voice(
    narration: str,
    config: PipelineConfig,
    output_path: Path,
    scene_segments: list[str] | None = None,
    description: str | None = None,
) -> Path:
    voice_desc = (description or config.voice.description).strip()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = scene_segments if scene_segments else [narration]
    segments = [s.strip() for s in segments if s.strip()]
    if not segments:
        segments = [narration]

    logger.info("Generating voice for %d segment(s)", len(segments))
    chunks = [_generate_chunk(seg, voice_desc, config) for seg in segments]
    audio = _concat_wavs(chunks)

    model, _, _ = _load_model(config)
    sf.write(str(output_path), audio, model.config.sampling_rate)
    logger.info("Saved voice to %s (%.1fs)", output_path, len(audio) / model.config.sampling_rate)
    return output_path


def unload_model() -> None:
    global _model, _tokenizer, _description_tokenizer
    if _model is not None:
        del _model
    if _tokenizer is not None:
        del _tokenizer
    if _description_tokenizer is not None:
        del _description_tokenizer
    _model = _tokenizer = _description_tokenizer = None
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
