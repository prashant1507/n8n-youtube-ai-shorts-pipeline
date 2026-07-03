"""Load and validate pipeline configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "default.yaml"


def resolve_theme(theme: str | None, config: "PipelineConfig") -> str:
    """Use explicit theme, or the next theme in serial rotation when empty / 'auto'."""
    key = (theme or "").strip().lower()
    if not key or key == "auto":
        from .theme_rotation import allocate_serial_theme

        chosen, _, _ = allocate_serial_theme(config.themes)
        return chosen
    from .theme_rotation import clear_pending

    clear_pending()
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
    min_hook_score: int = 7  # first-sentence hook is scored separately; below this triggers a revision


class SeoConfig(BaseModel):
    """YouTube packaging: SEO title variants, description, tags for reach/CTR."""

    enabled: bool = True
    title_variants: int = 5
    max_title_chars: int = 90
    title_hashtag_count: int = 2
    youtube_title_max_chars: int = 100
    max_tags: int = 15
    temperature: float = 0.7
    title_ab: bool = True  # rotate title styles per run so analytics can compare them
    title_styles: list[str] = Field(
        default_factory=lambda: ["question", "cliffhanger", "curiosity", "character", "emotional"]
    )


class AnalyticsConfig(BaseModel):
    """YouTube performance feedback: uploads registry + stats sync + SEO prompt feedback."""

    enabled: bool = True
    min_videos_for_feedback: int = 4       # need this many videos with stats before feeding the LLM
    min_stat_age_hours: int = 48           # only judge videos at least this old (early stats are noise)
    feedback_top_n: int = 3                # top/bottom titles shown to the SEO prompt
    engagement_comment_en: str = (
        "Which part did you like the most? Tell us in the comments!"
    )
    engagement_comment_hi: str = (
        "आपको सबसे अच्छा कौन सा हिस्सा लगा? कमेंट में बताइए!"
    )
    engagement_comment_pool_en: list[str] = Field(
        default_factory=lambda: [
            "Which part did you like the most? Tell us in the comments!",
            "What was your favorite moment in this video?",
            "Did this story surprise you? Let us know below!",
            "Which character or moment stood out to you?",
            "Would you watch a part 2? Comment yes or no!",
            "What should we cover next? Drop your ideas!",
        ]
    )
    engagement_comment_pool_hi: list[str] = Field(
        default_factory=lambda: [
            "आपको सबसे अच्छा कौन सा हिस्सा लगा? कमेंट में बताइए!",
            "इस वीडियो का सबसे पसंदीदा पल कौन सा था?",
            "क्या इस कहानी ने आपको हैरान किया? नीचे बताइए!",
            "कौन सा पल या किरदार सबसे यादगार लगा?",
            "क्या आप पार्ट 2 देखना चाहेंगे? हाँ या ना लिखिए!",
            "अगली वीडियो में क्या देखना चाहते हैं? सुझाव दीजिए!",
        ]
    )


class LLMConfig(BaseModel):
    base_url: str
    model: str
    temperature: float
    auto_select_model: bool
    quality: QualityConfig = Field(default_factory=QualityConfig)
    seo: SeoConfig = Field(default_factory=SeoConfig)


class ShortsConfig(BaseModel):
    """Retention touches for Shorts: on-screen hook, punchy captions, loop transition."""

    hook_overlay: bool = True
    hook_overlay_sec: float = 2.5
    caption_chunk_words: int = 4      # 0 = show the full narration segment (legacy behavior)
    caption_font_scale: float = 1.35  # relative to the base subtitle font
    caption_y_ratio: float = 0.72     # vertical center of captions (0 = top, 1 = bottom)
    loop_transition: bool = True
    loop_transition_sec: float = 0.4


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
    shorts: ShortsConfig = Field(default_factory=ShortsConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)

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
            "Content type is required. Pass --theme on CLI, omit for serial rotation, "
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
