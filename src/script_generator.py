"""Generate story + image prompts via LM Studio (n8n-video-generator pattern).

Two sequential LLM calls:
  1. {title, script} in target language
  2. {image_prompts} in English with physical anchor descriptions (no names)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from .config import PipelineConfig
from .llm_common import (
    language_label,
    parse_llm_json,
    target_scene_count,
    theme_key,
    word_target,
)
from .theme_profiles import resolve_theme_profile, resolve_voice_description
from .visual_prompts import LLM_IMAGE_STYLE
from .quality_agent import refine_image_prompts, refine_story
from .story_registry import (
    MAX_UNIQUE_RETRIES,
    is_duplicate,
    recent_protagonist_names,
    recent_titles,
)

logger = logging.getLogger(__name__)


def _theme_profile(config: PipelineConfig) -> dict[str, str]:
    return resolve_theme_profile(config.theme)


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
            "minItems": 5,
            "maxItems": 10,
        }
    },
    "required": ["image_prompts"],
    "additionalProperties": False,
}

YOUTUBE_META_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "titles": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 8,
        },
        "title_variants": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 8,
        },
        "description": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 25,
        },
        "hashtags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 6,
        },
        "hook_text": {"type": "string"},
    },
    "required": ["titles", "title_variants", "description", "tags", "hashtags", "hook_text"],
    "additionalProperties": False,
}


def _extract_message_content(message: Any) -> str:
    """Support reasoning models that put JSON in reasoning_content."""
    content = (getattr(message, "content", None) or "").strip()
    if content:
        return content
    if hasattr(message, "model_extra") and message.model_extra:
        rc = message.model_extra.get("reasoning_content", "")
        if rc:
            return str(rc).strip()
    return ""


def _resolve_model(client: OpenAI, config: PipelineConfig) -> str:
    if not config.llm.auto_select_model:
        return config.llm.model
    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
        if config.llm.model in ids:
            return config.llm.model
        if ids:
            chosen = ids[0]
            logger.warning("Model %s not loaded; using %s instead", config.llm.model, chosen)
            return chosen
    except Exception as exc:
        logger.warning("Could not list LM Studio models: %s", exc)
    return config.llm.model


def _protagonist_name_rules(config: PipelineConfig) -> str:
    lang = language_label(config)
    avoid = recent_protagonist_names(lang, limit=12)
    if lang == "hindi":
        lines = [
            "### PROTAGONIST NAME ###",
            "Give the main character a new name you invent for this story only.",
            "Do not reuse protagonist names from past videos in this series.",
            "Avoid overused defaults (Luna, Mia, Max, Leo, Emma, Noah, etc.).",
        ]
        if avoid:
            lines.append(f"Do NOT reuse these names from past videos: {', '.join(avoid)}")
        return "\n".join(lines)

    lines = [
        "### PROTAGONIST NAME ###",
        "Give the main protagonist a fresh, distinctive name invented for this story only.",
        "Use a unique protagonist name each time — never reuse names from past videos.",
        "Avoid overused kid-story defaults (Luna, Mia, Max, Leo, Emma, Noah, etc.).",
    ]
    if avoid:
        lines.append(f"Do NOT reuse these protagonist names from past videos: {', '.join(avoid)}")
    return "\n".join(lines)


def _story_system_prompt(config: PipelineConfig) -> str:
    lang = language_label(config)
    lo, hi = word_target(config.duration_sec)
    profile = _theme_profile(config)
    role = profile["role_hi"] if lang == "hindi" else profile["role_en"]
    content_type = theme_key(config.theme)
    structure = profile.get("structure_hi" if lang == "hindi" else "structure_en", "")

    lang_rules = (
        "5. Write ALL of 'title' and 'script' in Hindi using Devanagari script.\n"
        "6. Use simple spoken Hindi suitable for children — natural sentences, not Hinglish.\n"
        "7. Avoid English words unless commonly used in Hindi speech."
        if lang == "hindi"
        else "5. Write ALL of 'title' and 'script' in English.\n"
        "6. Use simple, warm, child-friendly vocabulary."
    )

    task_key = "task_hi" if lang == "hindi" else "task_en"
    default_task = (
        "Write a script that will be READ ALOUD by a TTS narrator (Mary for English, Rani for Hindi). "
        "Invent something completely original — your own characters, setting, and plot. "
        "Do NOT retell famous stories, fairy tales, movies, or existing IP."
    )
    task_block = profile.get(task_key) or profile.get("task_en") or default_task

    return (
        f"You are a {role}.\n"
        f"Content type: {content_type}. Tone: {config.tone}.\n\n"
        "### YOUR TASK ###\n"
        f"{task_block}\n\n"
        "### HOOK — FIRST 1-2 SENTENCES (most important for YouTube Shorts) ###\n"
        "The opening decides whether viewers keep watching. Make the first 1-2 sentences a strong hook:\n"
        "- Open with a curiosity gap, a surprising claim, a vivid problem, a question, or a mini-cliffhanger.\n"
        "- Create an instant reason to keep watching within the first 3 seconds.\n"
        "- Hint at the payoff to come, but do NOT reveal the ending.\n"
        "- BANNED generic openings: 'Once upon a time', 'In a village', 'There was a', 'Long ago', "
        "'One day' as the very first words.\n"
        "- Keep the hook concrete and easy to say aloud — no meta narration like 'In this story'.\n\n"
        "### ENDING — LOOP-FRIENDLY ###\n"
        "End on a warm, satisfying note whose mood or image echoes the opening, so the short feels good to replay.\n\n"
        f"### NARRATION RULES (critical for TTS) ###\n"
        "1. Write for the ear, not the page — short sentences, gentle rhythm.\n"
        "2. Use simple punctuation only: periods, commas, question marks. No emojis, bullets, or stage directions.\n"
        "3. Limit dialogue; if used, keep quotes short.\n"
        "4. No markdown, headers, or speaker labels.\n"
        f"5. Target {lo}-{hi} words total (~{config.duration_sec}s at slow storytelling pace).\n"
        f"6. Structure for this type: {structure}\n\n"
        f"{_protagonist_name_rules(config)}\n\n"
        f"### LANGUAGE ###\n{lang_rules}\n\n"
        "### OUTPUT PROTOCOL ###\n"
        "Return ONLY raw JSON. No markdown, no code blocks, no extra text.\n"
        "The output must begin with '{' and end with '}'.\n\n"
        "### JSON FORMATTING (STRICT) ###\n"
        "1. ONE LINE ONLY — no physical line breaks inside the JSON.\n"
        "2. Paragraph breaks in script: use literal '\\n\\n'.\n"
        "3. Quotes in speech: escape as \\\"\n"
        '4. Keys: exactly "title" and "script".\n\n'
        'Schema: {"title": "string", "script": "string"}\n\n'
        "Output the single-line JSON now:"
    )


def _story_user_prompt(config: PipelineConfig) -> str:
    profile = _theme_profile(config)
    content_type = theme_key(config.theme)
    lang = language_label(config)
    instruction = profile["instruction_hi"] if lang == "hindi" else profile["instruction_en"]

    return (
        f"Content type: {content_type}\n"
        f"Language: {lang}\n"
        f"Duration: ~{config.duration_sec} seconds\n\n"
        f"{instruction}\n\n"
        "The script is narration only — one continuous piece to be read aloud. "
        "Make each sentence vivid enough to inspire a matching illustration later."
    )


def _image_prompts_system(config: PipelineConfig) -> str:
    scene_count = target_scene_count(config)
    profile = _theme_profile(config)
    visual_style = profile.get("visual_style", LLM_IMAGE_STYLE)
    return (
        "You are an expert storyboard artist for a narrated children's animated short.\n"
        "Each frame is a different shot, but every recurring character must look like the SAME person in every frame.\n\n"
        "### CHARACTER CONSISTENCY (highest priority) ###\n"
        "1. Before writing prompts, define a fixed 'Character Anchor' for each recurring character:\n"
        "   age, skin tone, build, hair, face, clothing colors, and one distinctive detail.\n"
        "2. Scene 1 MUST be a clear establishing shot showing all main characters with their full anchors.\n"
        "3. Scenes 2 onward: paste the EXACT same anchor text for each character — word for word, do not paraphrase.\n"
        "4. NO character names — physical descriptions only.\n\n"
        "### VISUAL VARIETY (while keeping same characters) ###\n"
        "Change camera, location, action, and lighting each scene — but never change how characters look.\n"
        "Each scene MUST differ in at least TWO of:\n"
        "- Camera: wide / medium / close-up / two-shot / low or high angle\n"
        "- Location: follow the narration as the story moves\n"
        "- Action: what happens in THIS beat\n"
        "- Mood/lighting: match the emotion\n\n"
        "### ART DIRECTION ###\n"
        f"Visual world: {visual_style}.\n"
        "Describe scenes only — do NOT add Pixar, 3D, or render-style tags (added later).\n\n"
        "### SCENE RULES ###\n"
        f"1. Produce exactly {scene_count} prompts in strict chronological story order.\n"
        "2. One distinct story moment per prompt — subject, action, setting, shot type, lighting.\n"
        "3. 25–35 words each — complete phrases; do not cut off mid-sentence.\n"
        "4. All image_prompts must be in ENGLISH.\n"
        "5. No text, words, letters, or signs in the image.\n\n"
        "### OUTPUT RULES (STRICT) ###\n"
        "1. Return ONLY a valid JSON object. No markdown, no code blocks.\n"
        "2. ONE LINE ONLY: The entire JSON must be a single physical line of text. Do not press Enter.\n"
        "3. Use exactly the key 'image_prompts'.\n\n"
        "### JSON SCHEMA ###\n"
        "{\n"
        '  "image_prompts": [\n'
        '    "Wide establishing shot: [full character anchor], village at dawn, warm light...",\n'
        '    "Close-up: [exact same character anchor], hands on ruined crops, rain, gray mood..."\n'
        "  ]\n"
        "}\n\n"
        "FINAL CHECK: Same character anchors repeated exactly? Different shots/locations? Output now:"
    )


def _image_prompts_user(config: PipelineConfig, script: str) -> str:
    scene_count = target_scene_count(config)
    return (
        f"Storyboard exactly {scene_count} illustrations for this narration.\n"
        "Scene 1: clear establishing shot — define every main character with a detailed fixed physical description.\n"
        "Scenes 2–N: reuse those EXACT descriptions word-for-word in every prompt so characters never change appearance.\n"
        "Vary camera angle, location, and action each scene — but characters must look identical throughout.\n"
        "NO NAMES — physical descriptions only.\n\n"
        f"STORY:\n{script}"
    )


def _llm_json_call(
    client: OpenAI,
    model: str,
    config: PipelineConfig,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_retries: int = 6,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=config.llm.temperature,
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
            message = response.choices[0].message
            raw = _extract_message_content(message)
            parsed = parse_llm_json(raw)
            if parsed:
                logger.info("LLM returned valid JSON on attempt %d", attempt)
                return parsed
            logger.warning("Attempt %d/%d: response did not parse as JSON", attempt, max_retries)
        except Exception as exc:
            last_error = exc
            logger.warning("Attempt %d/%d: %s", attempt, max_retries, exc)

    # Fallback without json_schema (older LM Studio builds)
    logger.warning("json_schema failed, retrying without strict schema")
    response = client.chat.completions.create(
        model=model,
        temperature=config.llm.temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = _extract_message_content(response.choices[0].message)
    parsed = parse_llm_json(raw)
    if parsed:
        return parsed
    raise RuntimeError(f"LLM failed to return valid JSON: {last_error}")


def _normalize_image_prompts(prompts: list[str], target: int) -> list[str]:
    """Trim or pad to exact scene count; strip style bloat LLM may add."""
    style_noise = re.compile(
        r"(?i)(pixar|3d animation|cinematic lighting|high detail|whimsical|storybook illustration|soft textures|vibrant colors|no text)[,\s]*",
    )
    cleaned = []
    for p in prompts:
        if not p or not p.strip():
            continue
        p = p.strip().strip('"')
        p = style_noise.sub("", p)
        p = re.sub(r"\s+", " ", p).strip(" ,")
        if p:
            cleaned.append(p)
    if len(cleaned) > target:
        cleaned = cleaned[:target]
    while len(cleaned) < target and cleaned:
        cleaned.append(cleaned[-1])
    result = []
    for p in cleaned:
        words = p.split()
        if len(words) > 40:
            p = " ".join(words[:40]).rstrip(",;:")
        result.append(p)
    return result


def _normalize_script_text(script: str) -> str:
    return script.replace("\\n\\n", "\n\n").replace("\\n", "\n").strip()


def _split_narration_for_scenes(text: str, n: int) -> list[str]:
    """Split narration into n contiguous segments aligned with image prompts in order."""
    text = text.replace("\n\n", " ").strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]
    if not sentences:
        return [text] * n
    if len(sentences) <= n:
        segments = list(sentences)
        while len(segments) < n:
            segments.append(segments[-1])
        return segments[:n]

    segments: list[str] = []
    for i in range(n):
        start = int(i * len(sentences) / n)
        end = int((i + 1) * len(sentences) / n) if i < n - 1 else len(sentences)
        segments.append(" ".join(sentences[start:end]))
    return segments


def _build_scenes(script_text: str, image_prompts: list[str], duration_sec: int) -> list[dict]:
    n = len(image_prompts)
    segments = _split_narration_for_scenes(script_text, n)
    per_scene = duration_sec / n
    scenes = []
    for i, (segment, image_prompt) in enumerate(zip(segments, image_prompts)):
        scenes.append({
            "index": i + 1,
            "narration_segment": segment,
            "image_prompt": image_prompt,
            "video_prompt": image_prompt,
            "duration_sec": round(per_scene, 1),
        })
    return scenes


def _default_music_prompt(config: PipelineConfig) -> str:
    return _theme_profile(config)["music"]


def _default_voice_description(config: PipelineConfig) -> str:
    return resolve_voice_description(
        config.theme,
        language_label(config),
        config.voice.description,
    )


def _story_retry_addon(config: PipelineConfig, rejected_titles: list[str]) -> str:
    avoid = recent_titles(theme_key(config.theme), language_label(config), limit=8)
    avoid_names = recent_protagonist_names(language_label(config), limit=12)
    lines = [
        "\n\n### REJECTED — already generated ###",
        "The story you just wrote matches one we already produced.",
        "Invent something COMPLETELY different: new characters, setting, and plot.",
        "Use a new protagonist name — not one from past videos.",
    ]
    if rejected_titles:
        lines.append(f"Rejected this run: {', '.join(rejected_titles)}")
    if avoid:
        lines.append(f"Do NOT reuse these past titles: {', '.join(avoid)}")
    if avoid_names:
        lines.append(f"Do NOT reuse these protagonist names: {', '.join(avoid_names)}")
    return "\n".join(lines)


def _generate_unique_story(
    client: OpenAI,
    model: str,
    config: PipelineConfig,
) -> tuple[str, str]:
    """Step 1 with dedup — retry LLM if story already exists in records/."""
    content_type = theme_key(config.theme)
    lang = language_label(config)
    rejected: list[str] = []

    for attempt in range(1, MAX_UNIQUE_RETRIES + 1):
        user = _story_user_prompt(config)
        if rejected:
            user += _story_retry_addon(config, rejected)

        story_result = _llm_json_call(
            client, model, config,
            _story_system_prompt(config),
            user,
            STORY_SCHEMA,
        )
        title = story_result.get("title", "").strip()
        script_text = _normalize_script_text(story_result.get("script", ""))
        if not title or not script_text:
            raise ValueError("Story generation returned empty title or script")

        if is_duplicate(script_text, title, content_type, lang):
            logger.warning(
                "Duplicate story rejected: %s (attempt %d/%d)",
                title, attempt, MAX_UNIQUE_RETRIES,
            )
            rejected.append(title)
            continue

        if attempt > 1:
            logger.info("Unique story accepted on attempt %d: %s", attempt, title)
        return title, script_text

    raise RuntimeError(
        f"Could not generate a unique story after {MAX_UNIQUE_RETRIES} attempts. "
        f"Rejected titles: {', '.join(rejected)}"
    )


def _youtube_meta_system(config: PipelineConfig) -> str:
    lang = language_label(config)
    seo = config.llm.seo
    content_type = theme_key(config.theme)
    if lang == "hindi":
        lang_rule = (
            "Write titles (upload), description and hook_text in natural Hindi (Devanagari). "
            "Write title_variants, tags and hashtags in English only (no Hindi script). "
            "title_variants are English alternate YouTube titles for SEO — same story, English wording."
        )
    else:
        lang_rule = "Write titles, title_variants, description, hook_text, tags and hashtags in English."

    return (
        "You are a YouTube Shorts growth strategist who writes packaging (titles, "
        "descriptions, tags) that maximizes click-through rate and reach.\n"
        f"Content type: {content_type}. Audience: family/kids-friendly viewers.\n\n"
        "### TITLES (upload) ###\n"
        f"Propose {seo.title_variants} DISTINCT upload title candidates in the story language, strongest first.\n"
        f"- Each title <= {seo.max_title_chars} characters.\n"
        "- Use curiosity gaps, specificity, emotion, or a light question — but never mislead "
        "(the title must match the actual story).\n"
        "- No ALL CAPS words, no clickbait lies, no emojis, family-safe.\n"
        "- Do not put the word 'Shorts' or hashtags inside the titles.\n\n"
        "### TITLE VARIANTS (English alternates) ###\n"
        f"Propose {seo.title_variants} DISTINCT English alternate titles (same story, English only). "
        "For Hindi stories these must be English; for English stories match the upload titles.\n"
        f"- Each <= {seo.max_title_chars} characters, no hashtags, family-safe.\n\n"
        "### DESCRIPTION ###\n"
        "2-4 short sentences: lead with the hook/benefit, tease the story, end with a soft "
        "call to action (like/subscribe/watch till end). No hashtags in the description body.\n\n"
        "### TAGS ###\n"
        f"Give up to {seo.max_tags} lowercase English SEO keywords/phrases relevant to the story, theme, "
        "and audience (mix broad + specific). No '#', no duplicates, no Hindi script.\n\n"
        "### HASHTAGS ###\n"
        "4-6 English hashtags WITHOUT the '#' symbol (added later). Always include 'shorts'. "
        "ASCII English only — no Hindi/Devanagari. "
        "Put the strongest discovery tags first — the first 2 are appended to the YouTube title. "
        "Theme-specific tags are welcome.\n\n"
        "### HOOK_TEXT ###\n"
        "A punchy on-screen hook of at most 8 words for the first frame — bold and curiosity-driving.\n\n"
        f"### LANGUAGE ###\n{lang_rule}\n\n"
        "### OUTPUT PROTOCOL ###\n"
        "Return ONLY raw JSON on a single line. It must begin with '{' and end with '}'.\n"
        'Keys: "titles" (array), "title_variants" (array), "description" (string), "tags" (array), '
        '"hashtags" (array), "hook_text" (string).'
    )


def _youtube_meta_user(config: PipelineConfig, title: str, script_text: str) -> str:
    return (
        f"STORY TITLE (working): {title}\n\n"
        f"STORY NARRATION:\n{script_text}\n\n"
        "Write the YouTube packaging for this Short now."
    )


_HASHTAG_CLEAN = re.compile(r"[^0-9A-Za-z]+")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]+")


def _clean_tag(tag: str) -> str:
    return re.sub(r"\s+", " ", str(tag).replace("#", "").strip()).strip(" ,")


def _english_seo_tag(tag: str) -> str:
    """YouTube upload tag — English ASCII only (strip Hindi script)."""
    text = _DEVANAGARI.sub("", _clean_tag(tag))
    text = re.sub(r"[^0-9A-Za-z ]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _english_title_variant(title: str) -> str:
    """English-only title variant — strip Devanagari, keep readable English."""
    text = _DEVANAGARI.sub("", str(title).strip().strip('"'))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


def _clean_hashtag(tag: str) -> str:
    return _HASHTAG_CLEAN.sub("", str(tag).replace("#", "").strip()).lower()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _choose_youtube_title(candidates: list[str], fallback: str, max_chars: int) -> str:
    for cand in candidates:
        cand = cand.strip().strip('"')
        if cand and len(cand) <= max_chars:
            return cand
    best = (candidates[0].strip().strip('"') if candidates else "") or fallback
    if len(best) > max_chars:
        best = best[: max_chars - 1].rstrip() + "\u2026"
    return best


def _pick_title_hashtags(hashtags: list[str], count: int) -> list[str]:
    """Take the first N hashtags from the LLM list for the upload title."""
    if count <= 0 or not hashtags:
        return []
    return hashtags[:count]


def _append_hashtags_to_title(
    base_title: str,
    hashtags: list[str],
    *,
    max_chars: int,
    hashtag_count: int,
) -> str:
    """Build upload title like 'The Great Pastry Disaster #cartoon #animation'."""
    base = base_title.strip().strip('"')
    picked = _pick_title_hashtags(hashtags, hashtag_count)
    if not picked:
        return base[:max_chars]

    suffix = " " + " ".join(f"#{tag}" for tag in picked)
    if len(suffix) >= max_chars:
        return base[:max_chars]

    max_base = max_chars - len(suffix)
    if len(base) > max_base:
        if max_base <= 1:
            return base[:max_chars]
        base = base[: max_base - 1].rstrip() + "\u2026"
    return base + suffix


def generate_youtube_metadata(
    client: OpenAI,
    model: str,
    config: PipelineConfig,
    title: str,
    script_text: str,
) -> dict[str, Any] | None:
    """Generate CTR/SEO packaging: title variants, description, tags, hashtags, hook text.

    Returns None (and the pipeline falls back to legacy metadata) on any failure.
    """
    seo = config.llm.seo
    if not seo.enabled:
        return None
    try:
        result = _llm_json_call(
            client,
            model,
            config,
            _youtube_meta_system(config),
            _youtube_meta_user(config, title, script_text),
            YOUTUBE_META_SCHEMA,
            max_retries=3,
        )
    except Exception as exc:
        logger.warning("YouTube metadata generation failed, using fallback: %s", exc)
        return None

    raw_titles = [str(t).strip().strip('"') for t in (result.get("titles") or []) if str(t).strip()]
    raw_titles = _dedupe_keep_order(raw_titles)

    raw_variants = [
        str(t).strip().strip('"') for t in (result.get("title_variants") or []) if str(t).strip()
    ]
    lang = language_label(config)
    if lang == "hindi":
        title_variants = _dedupe_keep_order(
            [v for v in (_english_title_variant(x) for x in raw_variants) if v and not _has_devanagari(v)]
        )
        upload_candidates = raw_titles
    else:
        title_variants = _dedupe_keep_order(
            [v for v in (_english_title_variant(x) for x in (raw_variants or raw_titles)) if v]
        )
        upload_candidates = raw_titles or title_variants

    tags = _dedupe_keep_order(
        [t for t in (_english_seo_tag(x) for x in (result.get("tags") or [])) if t]
    )[: seo.max_tags]

    hashtags = _dedupe_keep_order(
        [h for h in (_clean_hashtag(x) for x in (result.get("hashtags") or [])) if h]
    )
    if not any(h.lower() == "shorts" for h in hashtags):
        hashtags.append("shorts")
    hashtags = hashtags[:6]

    description = str(result.get("description") or "").strip()
    hook_text = str(result.get("hook_text") or "").strip().strip('"')

    base_title = _choose_youtube_title(upload_candidates, title, seo.max_title_chars)
    youtube_title = _append_hashtags_to_title(
        base_title,
        hashtags,
        max_chars=seo.youtube_title_max_chars,
        hashtag_count=seo.title_hashtag_count,
    )

    logger.info(
        "YouTube SEO title: %s (%d upload titles, %d EN variants, %d tags, title hashtags: %d)",
        youtube_title,
        len(upload_candidates),
        len(title_variants),
        len(tags),
        seo.title_hashtag_count,
    )

    return {
        "youtube_title": youtube_title,
        "youtube_title_base": base_title,
        "title_variants": title_variants,
        "youtube_description": description,
        "youtube_tags": tags,
        "hashtags": hashtags,
        "hook_text": hook_text,
    }


def generate_script(config: PipelineConfig) -> dict[str, Any]:
    """Two-step generation: LLM invents content from content-type theme."""
    client = OpenAI(base_url=config.llm.base_url, api_key="lm-studio")
    model = _resolve_model(client, config)
    lang = language_label(config)
    content_type = theme_key(config.theme)

    logger.info("Generating content type: %s (%d scenes)", content_type, target_scene_count(config))

    # Step 1: title + script (deduplicated via records/)
    title, script_text = _generate_unique_story(client, model, config)
    title, script_text = refine_story(client, model, config, title, script_text)
    logger.info("Generated [%s]: %s (%d chars)", content_type, title, len(script_text))

    # Step 2: image prompts (English, no names, physical anchors, chronological)
    scene_count = target_scene_count(config)
    prompts_result = _llm_json_call(
        client, model, config,
        _image_prompts_system(config),
        _image_prompts_user(config, script_text),
        IMAGE_PROMPTS_SCHEMA,
    )
    raw_prompts = prompts_result.get("image_prompts") or []
    if len(raw_prompts) < scene_count:
        raise ValueError(f"Expected at least {scene_count} image prompts, got {len(raw_prompts)}")
    image_prompts = _normalize_image_prompts(raw_prompts, scene_count)
    if len(image_prompts) != scene_count:
        raise ValueError(f"Expected {scene_count} image prompts, got {len(image_prompts)}")

    image_prompts = refine_image_prompts(client, model, config, script_text, image_prompts)
    image_prompts = _normalize_image_prompts(image_prompts, scene_count)

    scenes = _build_scenes(script_text, image_prompts, config.duration_sec)

    result: dict[str, Any] = {
        "title": title,
        "script": script_text,
        "narration": script_text,
        "image_prompts": image_prompts,
        "language": lang,
        "content_type": content_type,
        "theme": content_type,
        "scenes": scenes,
        "music_prompt": _default_music_prompt(config),
        "voice_description": _default_voice_description(config),
    }

    metadata = generate_youtube_metadata(client, model, config, title, script_text)
    if metadata:
        result.update(metadata)

    return result
