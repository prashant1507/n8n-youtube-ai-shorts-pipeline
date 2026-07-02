"""LLM quality gate: validate story + image prompts and revise when score is low."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from .config import PipelineConfig
from .llm_common import language_label, parse_llm_json, target_scene_count, theme_key, word_target
from .story_registry import recent_protagonist_names, recent_titles
from .theme_profiles import resolve_theme_profile

logger = logging.getLogger(__name__)

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "score": {"type": "integer"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["pass", "score", "issues", "summary"],
    "additionalProperties": False,
}

STORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "script": {"type": "string"}},
    "required": ["title", "script"],
    "additionalProperties": False,
}

IMAGE_PROMPTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "image_prompts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 10,
        }
    },
    "required": ["image_prompts"],
    "additionalProperties": False,
}


def _extract_content(message: Any) -> str:
    content = (getattr(message, "content", None) or "").strip()
    if content:
        return content
    extra = getattr(message, "model_extra", None) or {}
    rc = extra.get("reasoning_content", "")
    return str(rc).strip() if rc else ""


def _llm_json(
    client: OpenAI,
    model: str,
    config: PipelineConfig,
    system: str,
    user: str,
    schema: dict[str, Any],
    *,
    temperature: float | None = None,
) -> dict[str, Any]:
    temp = temperature if temperature is not None else config.llm.quality.temperature
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temp,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
            raw = _extract_content(response.choices[0].message)
            parsed = parse_llm_json(raw)
            if parsed:
                return parsed
        except Exception as exc:
            last_error = exc
            logger.warning("Quality LLM attempt %d failed: %s", attempt, exc)

    response = client.chat.completions.create(
        model=model,
        temperature=temp,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    parsed = parse_llm_json(_extract_content(response.choices[0].message))
    if parsed:
        return parsed
    raise RuntimeError(f"Quality agent LLM failed: {last_error}")


def _story_context(config: PipelineConfig) -> str:
    profile = resolve_theme_profile(config.theme)
    lang = language_label(config)
    lo, hi = word_target(config.duration_sec)
    instruction = profile["instruction_hi"] if lang == "hindi" else profile["instruction_en"]
    structure = profile.get("structure_hi" if lang == "hindi" else "structure_en", "")
    avoid_names = recent_protagonist_names(lang, limit=12)
    avoid_titles = recent_titles(theme_key(config.theme), lang, limit=8)
    lines = [
        f"Content type: {theme_key(config.theme)}",
        f"Language: {lang}",
        f"Target length: {lo}-{hi} words (~{config.duration_sec}s narration)",
        f"Tone: {config.tone}",
        f"Theme instruction: {instruction}",
        f"Structure: {structure}",
    ]
    if avoid_names:
        lines.append(f"Banned protagonist names (already used): {', '.join(avoid_names)}")
    if avoid_titles:
        lines.append(f"Past titles to differ from: {', '.join(avoid_titles)}")
    return "\n".join(lines)


def _validate_story_system() -> str:
    return (
        "You are a strict quality reviewer for children's narrated YouTube Shorts scripts.\n"
        "Score 1-10 and list concrete issues. Set pass=true only if score >= 8 with no critical issues.\n\n"
        "Check:\n"
        "1. Fits the content type and theme instruction\n"
        "2. Strong hook in the first 1-2 sentences (not generic 'In a village' / 'Once upon a time')\n"
        "3. TTS-ready: no markdown, emojis, stage directions, speaker labels\n"
        "4. Word count within target range\n"
        "5. Fresh protagonist name (not in banned list)\n"
        "6. Clear beginning, middle, ending — one main arc\n"
        "7. Child-friendly, read-aloud rhythm\n"
        "8. Original — not a famous fairy tale retread\n\n"
        "Return JSON: pass (bool), score (1-10), issues (string array), summary (one line)."
    )


def _validate_story(client: OpenAI, model: str, config: PipelineConfig, title: str, script: str) -> dict[str, Any]:
    user = (
        f"{_story_context(config)}\n\n"
        f"TITLE:\n{title}\n\n"
        f"SCRIPT:\n{script}\n\n"
        "Review this script."
    )
    return _llm_json(client, model, config, _validate_story_system(), user, VERDICT_SCHEMA)


def _revise_story(
    client: OpenAI,
    model: str,
    config: PipelineConfig,
    title: str,
    script: str,
    verdict: dict[str, Any],
) -> tuple[str, str]:
    lang = language_label(config)
    lo, hi = word_target(config.duration_sec)
    issues = verdict.get("issues") or []
    system = (
        "You are a children's script editor. Fix the script based on the review issues.\n"
        "Keep the same language, content type, and approximate length.\n"
        f"Target {lo}-{hi} words. Language: {lang}.\n"
        "Return ONLY one-line JSON with keys title and script.\n"
        "Use \\n\\n between paragraphs inside script."
    )
    user = (
        f"{_story_context(config)}\n\n"
        f"REVIEW SUMMARY: {verdict.get('summary', '')}\n"
        f"ISSUES TO FIX:\n- " + "\n- ".join(str(i) for i in issues) + "\n\n"
        f"CURRENT TITLE:\n{title}\n\n"
        f"CURRENT SCRIPT:\n{script}\n\n"
        "Output improved title and script."
    )
    result = _llm_json(client, model, config, system, user, STORY_SCHEMA)
    new_title = str(result.get("title", title)).strip() or title
    new_script = str(result.get("script", script)).replace("\\n\\n", "\n\n").replace("\\n", "\n").strip()
    return new_title, new_script or script


def refine_story(
    client: OpenAI,
    model: str,
    config: PipelineConfig,
    title: str,
    script: str,
) -> tuple[str, str]:
    """Validate story; revise up to max_revisions times if score is below min_score."""
    q = config.llm.quality
    if not q.enabled:
        return title, script

    for attempt in range(q.max_revisions + 1):
        verdict = _validate_story(client, model, config, title, script)
        score = int(verdict.get("score", 0))
        passed = bool(verdict.get("pass")) or score >= q.min_score
        logger.info(
            "Story quality review: score=%d pass=%s (%s)",
            score,
            passed,
            verdict.get("summary", ""),
        )
        if passed:
            return title, script
        issues = verdict.get("issues") or []
        logger.warning("Story quality issues: %s", "; ".join(str(i) for i in issues[:5]))
        if attempt >= q.max_revisions:
            logger.warning("Accepting story after %d quality revision(s)", q.max_revisions)
            return title, script
        title, script = _revise_story(client, model, config, title, script, verdict)
        logger.info("Story revised (quality pass %d/%d)", attempt + 1, q.max_revisions)

    return title, script


def _validate_images_system() -> str:
    return (
        "You are a storyboard quality reviewer for a vertical 9:16 children's animated short.\n"
        "Score 1-10. Set pass=true only if score >= 8 with no critical issues.\n\n"
        "Check image_prompts against the story:\n"
        "1. Correct count\n"
        "2. Chronological story order\n"
        "3. Scene 1 is a clear establishing shot with full character physical descriptions\n"
        "4. Scenes 2+ repeat the EXACT same character anchor text (no names)\n"
        "5. Each scene differs in camera angle, location, or action\n"
        "6. English only, 25-35 words each, no text-in-image requests\n"
        "7. Matches story mood and content type visual style\n\n"
        "Return JSON: pass, score, issues, summary."
    )


def _validate_image_prompts(
    client: OpenAI,
    model: str,
    config: PipelineConfig,
    script: str,
    image_prompts: list[str],
) -> dict[str, Any]:
    profile = resolve_theme_profile(config.theme)
    expected = target_scene_count(config)
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(image_prompts))
    user = (
        f"Content type: {theme_key(config.theme)}\n"
        f"Visual style: {profile.get('visual_style', '')}\n"
        f"Expected prompt count: {expected}\n\n"
        f"STORY:\n{script}\n\n"
        f"IMAGE PROMPTS ({len(image_prompts)}):\n{numbered}\n\n"
        "Review these image prompts."
    )
    return _llm_json(client, model, config, _validate_images_system(), user, VERDICT_SCHEMA)


def _revise_image_prompts(
    client: OpenAI,
    model: str,
    config: PipelineConfig,
    script: str,
    image_prompts: list[str],
    verdict: dict[str, Any],
) -> list[str]:
    expected = target_scene_count(config)
    profile = resolve_theme_profile(config.theme)
    issues = verdict.get("issues") or []
    system = (
        "You are a storyboard editor. Fix image prompts based on the review.\n"
        f"Output exactly {expected} prompts in chronological order.\n"
        "Scene 1: full character anchors. Later scenes: repeat anchors word-for-word.\n"
        "NO character names — physical descriptions only. English only.\n"
        "Return one-line JSON: {\"image_prompts\": [\"...\", ...]}"
    )
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(image_prompts))
    user = (
        f"Visual style: {profile.get('visual_style', '')}\n\n"
        f"REVIEW: {verdict.get('summary', '')}\n"
        f"ISSUES:\n- " + "\n- ".join(str(i) for i in issues) + "\n\n"
        f"STORY:\n{script}\n\n"
        f"CURRENT PROMPTS:\n{numbered}\n\n"
        f"Output exactly {expected} improved image_prompts."
    )
    result = _llm_json(client, model, config, system, user, IMAGE_PROMPTS_SCHEMA)
    prompts = result.get("image_prompts") or []
    return [str(p).strip() for p in prompts if str(p).strip()]


def refine_image_prompts(
    client: OpenAI,
    model: str,
    config: PipelineConfig,
    script: str,
    image_prompts: list[str],
) -> list[str]:
    """Validate image prompts; revise if score is below min_score."""
    q = config.llm.quality
    if not q.enabled:
        return image_prompts

    prompts = list(image_prompts)
    for attempt in range(q.max_revisions + 1):
        verdict = _validate_image_prompts(client, model, config, script, prompts)
        score = int(verdict.get("score", 0))
        passed = bool(verdict.get("pass")) or score >= q.min_score
        logger.info(
            "Image prompt quality review: score=%d pass=%s (%s)",
            score,
            passed,
            verdict.get("summary", ""),
        )
        if passed:
            return prompts
        issues = verdict.get("issues") or []
        logger.warning("Image prompt issues: %s", "; ".join(str(i) for i in issues[:5]))
        if attempt >= q.max_revisions:
            logger.warning("Accepting image prompts after %d quality revision(s)", q.max_revisions)
            return prompts
        revised = _revise_image_prompts(client, model, config, script, prompts, verdict)
        if len(revised) >= target_scene_count(config):
            prompts = revised
        else:
            logger.warning("Revision returned too few prompts; keeping previous set")
        logger.info("Image prompts revised (quality pass %d/%d)", attempt + 1, q.max_revisions)

    return prompts
