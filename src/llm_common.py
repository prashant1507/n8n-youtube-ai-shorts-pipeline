"""Shared helpers for script_generator and quality_agent."""

from __future__ import annotations

import json
import re
from typing import Any

from .config import PipelineConfig


def language_label(config: PipelineConfig) -> str:
    return "hindi" if config.language.lower().startswith("hi") else "english"


def theme_key(theme: str) -> str:
    return theme.lower().strip().replace(" ", "_")


def word_target(duration_sec: int) -> tuple[int, int]:
    """~90–110 wpm for slow Divya narration."""
    lo = max(8, int(duration_sec * 90 / 60))
    hi = max(lo + 5, int(duration_sec * 110 / 60))
    return lo, hi


def sec_per_scene_for_tier(tier: str) -> int:
    return 4 if tier.lower() == "wan" else 8


def target_scene_count(config: PipelineConfig) -> int:
    n = config.video.scenes_count or 6
    by_duration = max(
        2,
        min(10, round(config.duration_sec / sec_per_scene_for_tier(config.video.tier))),
    )
    return max(2, min(10, n, by_duration))


def parse_llm_json(raw: str) -> dict[str, Any] | None:
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\n?|```$", "", clean, flags=re.MULTILINE).strip()
        clean = clean.replace("\n", " ").replace("\r", "")
        result = json.loads(clean)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None
