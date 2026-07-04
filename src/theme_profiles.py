"""Load content-type profiles from profiles/*.yaml at project root."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

from .config import ROOT

logger = logging.getLogger(__name__)

PROFILES_DIR = ROOT / "profiles"

REQUIRED_FIELDS = frozenset({
    "role_en",
    "role_hi",
    "instruction_en",
    "instruction_hi",
    "structure_en",
    "structure_hi",
    "visual_style",
    "music",
})


def _theme_key(theme: str) -> str:
    return theme.lower().strip().replace(" ", "_")


def _is_hindi(language: str) -> bool:
    lang = language.lower().strip()
    return lang in ("hi", "hindi")


def resolve_voice_description(theme: str, language: str, fallback: str) -> str:
    """Parler-TTS voice description from profiles/{theme}.yaml, else default.yaml fallback.

    Hindi uses voice_hi; English uses voice_en.
    """
    profile = resolve_theme_profile(theme)
    hindi = _is_hindi(language)

    if hindi:
        if profile.get("voice_hi"):
            return profile["voice_hi"]
    elif profile.get("voice_en"):
        return profile["voice_en"]

    if profile.get("voice"):
        return profile["voice"]

    return fallback.strip()


def resolve_voice_from_script(script: dict, fallback: str) -> str:
    """Voice description from script.json, or resolve from theme if missing (older runs)."""
    existing = str(script.get("voice_description", "")).strip()
    if existing:
        return existing
    theme = str(script.get("content_type") or script.get("theme") or "")
    language = str(script.get("language") or "english")
    if not theme:
        return fallback.strip()
    return resolve_voice_description(theme, language, fallback)


@lru_cache(maxsize=1)
def load_theme_profiles() -> dict[str, dict[str, str]]:
    """Load all profiles/*.yaml (skips files starting with _)."""
    profiles: dict[str, dict[str, str]] = {}
    if not PROFILES_DIR.is_dir():
        logger.warning("Profiles directory missing: %s", PROFILES_DIR)
        return profiles

    for path in sorted(PROFILES_DIR.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        file_key = path.stem.lower()
        try:
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            if not isinstance(raw, dict):
                logger.warning("Skipping profile (not a mapping): %s", path.name)
                continue

            profile: dict[str, str] = {}
            for key, value in raw.items():
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    profile[key] = text

            theme_key = _theme_key(profile.pop("id", file_key))
            missing = REQUIRED_FIELDS - profile.keys()
            if missing:
                logger.warning("Profile %s missing fields: %s", path.name, sorted(missing))

            profiles[theme_key] = profile
            logger.debug("Loaded theme profile: %s from %s", theme_key, path.name)
        except Exception as exc:
            logger.warning("Failed to load profile %s: %s", path.name, exc)

    return profiles


def get_theme_profile(theme: str) -> dict[str, str] | None:
    key = _theme_key(theme)
    return load_theme_profiles().get(key)


def default_theme_profile(theme: str) -> dict[str, str]:
    """Fallback when no profiles/{theme}.yaml exists."""
    label = theme.strip() or "story"
    return {
        "role_en": "creative short-form video scriptwriter",
        "role_hi": "creative short-form video scriptwriter",
        "instruction_en": (
            f"Invent completely NEW original content in the style/genre: '{label}'. "
            "Choose the subject, characters (if any), and shape yourself. "
            "Clear beginning and satisfying ending that fit the genre."
        ),
        "instruction_hi": (
            f"'{label}' शैली में बिल्कुल नई original content लिखो। "
            "विषय, पात्र (अगर हों) और रूप तुम खुद चुनो।"
        ),
        "structure_en": "Clear beginning, middle, and ending matched to the genre.",
        "structure_hi": "शैली के अनुसार साफ शुरुआत, बीच और अंत।",
        "visual_style": "Colorful storybook illustration, expressive, high detail, no text",
        "music": f"Soft background music matching a {label} mood, no vocals",
        "voice_en": (
            f"Divya speaks with a light, clear, expressive voice matching a {label} mood, "
            "bright and friendly, never deep or heavy. "
            "Natural flowing pace with short pauses. "
            "Very clear audio, close recording, no background noise. "
            "RULE: Full text should be in audio."
        ),
        "voice_hi": (
            f"Rani speaks in clear, natural Hindi with a light, expressive tone matching a {label} mood, "
            "bright and friendly, never deep or heavy. "
            "Natural flowing pace with short pauses. "
            "Very clear audio, close recording, no background noise. "
            "RULE: Full text should be in audio."
        ),
    }


def resolve_theme_profile(theme: str) -> dict[str, str]:
    return get_theme_profile(theme) or default_theme_profile(theme)
