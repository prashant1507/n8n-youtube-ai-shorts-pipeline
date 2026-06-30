"""Build SD image prompts from n8n-style image_prompts (physical anchors, no names)."""

from __future__ import annotations

import logging
import re

from .config import PipelineConfig

logger = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 280

# LLM scene description style (n8n llm_server image_prompts step)
LLM_IMAGE_STYLE = (
    "Whimsical children's book illustration, soft textures, vibrant colors, no text, high detail"
)


def _scene_description(scene: dict) -> str:
    """Raw scene text from LLM — strip JSON-style quotes if present."""
    raw = scene.get("image_prompt", "").strip()
    if not raw:
        raw = scene.get("video_prompt", "").strip() or scene.get("narration_segment", "").strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    return re.sub(r"\s+", " ", raw).strip()


def build_image_prompt(scene: dict, script: dict, config: PipelineConfig) -> str:
    """Prepend Pixar style anchor (n8n FLUX pattern) + LLM scene description."""
    anchor = (config.video.style_anchor or config.video.style_suffix).strip().rstrip(".")
    image_prompt = _scene_description(scene)
    if not image_prompt:
        return anchor

    full = f"{anchor}. {image_prompt}"
    full = re.sub(r"\s+", " ", full).strip()
    if len(full) > MAX_PROMPT_CHARS:
        budget = MAX_PROMPT_CHARS - len(anchor) - 2
        if budget > 40:
            full = f"{anchor}. {image_prompt[:budget].rsplit(' ', 1)[0]}"
        else:
            full = anchor
    return full


def prepare_scene_prompts(script: dict, config: PipelineConfig, enrich: bool = False) -> tuple[dict, list[dict]]:
    """Build final SD prompts from script scenes (enrich disabled — prompts come from script_generator step 2)."""
    if enrich:
        logger.warning("Visual enrich is deprecated; image prompts come from script_generator step 2")

    prompts = []
    for i, scene in enumerate(script.get("scenes", [])):
        prompt = build_image_prompt(scene, script, config)
        prompts.append({
            "index": scene.get("index", i + 1),
            "narration_segment": scene.get("narration_segment", ""),
            "image_prompt": scene.get("image_prompt", ""),
            "final_prompt": prompt,
        })
    return script, prompts
