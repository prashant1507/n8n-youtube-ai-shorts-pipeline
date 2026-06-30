"""Track generated stories in records/ to avoid LLM duplicates."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT

logger = logging.getLogger(__name__)

RECORDS_DIR = ROOT / "records"
REGISTRY_PATH = RECORDS_DIR / "stories.json"
OUTPUT_DIR = ROOT / "output"

MAX_UNIQUE_RETRIES = 5


def normalize_script(text: str) -> str:
    text = text.replace("\\n", "\n").replace("\r", "")
    text = re.sub(r"\s+", " ", text.strip())
    return text


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip())


def script_hash(text: str) -> str:
    return hashlib.sha256(normalize_script(text).encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_registry() -> list[dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []
    raw = REGISTRY_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else data.get("stories", [])
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", REGISTRY_PATH, exc)
        return []


def _save_registry(records: list[dict[str, Any]]) -> None:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def rebuild_from_output() -> int:
    """Import stories from output/*/script.json into records/. Returns count added."""
    records = _load_registry()
    known_hashes = {r["hash"] for r in records if r.get("hash")}
    added = 0

    if not OUTPUT_DIR.exists():
        return 0

    for script_path in sorted(OUTPUT_DIR.glob("*/script.json")):
        try:
            script = json.loads(script_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        narration = script.get("narration") or script.get("script") or ""
        if not narration.strip():
            continue

        h = script_hash(narration)
        if h in known_hashes:
            continue

        run_id = script_path.parent.name
        records.append({
            "hash": h,
            "title": script.get("title", ""),
            "title_norm": normalize_title(script.get("title", "")),
            "content_type": script.get("content_type") or script.get("theme", ""),
            "language": script.get("language", ""),
            "run_id": run_id,
            "created_at": _now_iso(),
            "script_preview": normalize_script(narration)[:160],
        })
        known_hashes.add(h)
        added += 1

    if added:
        _save_registry(records)
        logger.info("Imported %d stories from output/ into records/", added)
    return added


def ensure_registry() -> None:
    """Create records/ and backfill from output/ if registry is empty."""
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists() or not _load_registry():
        rebuild_from_output()


def is_duplicate(
    script: str,
    title: str,
    content_type: str,
    language: str,
) -> bool:
    """True if script hash or title already exists for this content_type + language."""
    ensure_registry()
    records = _load_registry()
    h = script_hash(script)
    title_norm = normalize_title(title)
    ctype = content_type.lower().strip()
    lang = language.lower().strip()

    for rec in records:
        if rec.get("content_type", "").lower() != ctype:
            continue
        if rec.get("language", "").lower() != lang:
            continue
        if rec.get("hash") == h:
            return True
        if title_norm and rec.get("title_norm") == title_norm:
            return True
    return False


def _protagonist_name_from_title(title: str) -> str | None:
    t = title.strip()
    if not t:
        return None
    m = re.match(r"^([A-Z][a-zA-Z'-]+)'s\b", t)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Z][a-zA-Z'-]+)\s+and\s+the\b", t, re.IGNORECASE)
    if m:
        name = m.group(1)
        return name[0].upper() + name[1:] if name else name
    return None


def _protagonist_names_from_preview(preview: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"\bnamed\s+([A-Z][a-zA-Z'-]+)\b", preview)]


def recent_protagonist_names(language: str, limit: int = 12) -> list[str]:
    """Protagonist names from past stories (for LLM avoid list). Language-wide, all themes."""
    ensure_registry()
    records = _load_registry()
    lang = language.lower().strip()
    names: list[str] = []
    seen: set[str] = set()

    for rec in reversed(records):
        if rec.get("language", "").lower() != lang:
            continue
        candidates: list[str] = []
        title_name = _protagonist_name_from_title(rec.get("title", ""))
        if title_name:
            candidates.append(title_name)
        candidates.extend(_protagonist_names_from_preview(rec.get("script_preview", "")))
        for name in candidates:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                names.append(name)
            if len(names) >= limit:
                return names
    return names


def recent_titles(content_type: str, language: str, limit: int = 8) -> list[str]:
    ensure_registry()
    records = _load_registry()
    ctype = content_type.lower().strip()
    lang = language.lower().strip()
    titles: list[str] = []
    for rec in reversed(records):
        if rec.get("content_type", "").lower() != ctype:
            continue
        if rec.get("language", "").lower() != lang:
            continue
        title = rec.get("title", "").strip()
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def register_story(
    script: str,
    title: str,
    content_type: str,
    language: str,
    run_id: str,
) -> None:
    """Append accepted story to records/stories.json."""
    ensure_registry()
    records = _load_registry()
    h = script_hash(script)
    if any(r.get("hash") == h for r in records):
        return

    records.append({
        "hash": h,
        "title": title,
        "title_norm": normalize_title(title),
        "content_type": content_type,
        "language": language,
        "run_id": run_id,
        "created_at": _now_iso(),
        "script_preview": normalize_script(script)[:160],
    })
    _save_registry(records)
    logger.info("Recorded story in records/: %s (%s)", title, run_id)
