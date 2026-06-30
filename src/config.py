"""Load and validate pipeline configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "default.yaml"


def pick_random_theme(config: PipelineConfig) -> str:
    import random

    pool = [t.strip().lower() for t in config.themes if t and str(t).strip()]
    if not pool:
        raise ValueError(
            "No themes configured. Add a themes: list in default.yaml or pass --themes-csv."
        )
    return random.choice(pool)


def resolve_theme(theme: str | None, config: PipelineConfig) -> str:
    """Use explicit theme, or pick one at random when empty / 'auto'."""
    key = (theme or "").strip().lower()
    if not key or key == "auto":
        return pick_random_theme(config)
    return key


class VoiceConfig(BaseModel):
    description: str
    model: str
    device: str


class MusicConfig(BaseModel):
    model: str
    prompt: str
    volume: float


class VideoConfig(BaseModel):
    tier: str
    aspect_ratio: str
    resolution: str
    scenes_count: int
    style_anchor: str
    style_suffix: str
    flux_model: str
    flux_steps: int
    flux_guidance: float
    flux_max_sequence_length: int
    flux_gen_width: int
    flux_gen_height: int
    flux_python: str
    flux_local_files_only: bool
    flux_reference_mode: str
    flux_cpu_offload: bool
    motion_zoom: float
    motion_supersample: float
    ken_burns_fps: int
    wan_model: str


class QualityConfig(BaseModel):
    enabled: bool = True
    min_score: int = 7
    max_revisions: int = 2
    temperature: float = 0.3


class LLMConfig(BaseModel):
    base_url: str
    model: str
    temperature: float
    auto_select_model: bool
    quality: QualityConfig = Field(default_factory=QualityConfig)


class PipelineConfig(BaseModel):
    theme: str
    themes: list[str]
    language: str
    duration_sec: int
    tone: str
    voice: VoiceConfig
    music: MusicConfig
    video: VideoConfig
    llm: LLMConfig
    subtitles: bool

    @property
    def width(self) -> int:
        return int(self.video.resolution.split("x")[0])

    @property
    def height(self) -> int:
        return int(self.video.resolution.split("x")[1])


def require_theme(config: PipelineConfig) -> str:
    """Return content type; raise if not set (must be passed via --theme or random pick)."""
    theme = config.theme.strip()
    if not theme:
        hint = ", ".join(config.themes[:4]) if config.themes else "story, joke, bedtime, …"
        raise ValueError(
            "Content type is required. Pass --theme on CLI, omit for random, "
            f"or add themes in default.yaml (e.g. {hint})."
        )
    return theme


def is_hindi_language(language: str) -> bool:
    lang = language.lower().strip()
    return lang.startswith("hi") or lang == "hindi"


def apply_subtitle_policy(config: PipelineConfig, script: dict | None = None) -> None:
    """English: burn subtitles. Hindi: no on-screen subtitles."""
    if not config.subtitles:
        return
    if is_hindi_language(config.language):
        config.subtitles = False
        return
    if script and is_hindi_language(str(script.get("language", ""))):
        config.subtitles = False


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, **overrides: Any) -> PipelineConfig:
    default_path = DEFAULT_CONFIG
    if not default_path.exists():
        raise FileNotFoundError(f"Config not found: {default_path}")

    with open(default_path) as f:
        base: dict[str, Any] = yaml.safe_load(f) or {}

    path = Path(path) if path else default_path
    if path.resolve() != default_path.resolve():
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            overlay = yaml.safe_load(f) or {}
        base = _deep_merge(base, overlay)

    if overrides:
        base = _deep_merge(base, {k: v for k, v in overrides.items() if v is not None})

    return PipelineConfig.model_validate(base)
