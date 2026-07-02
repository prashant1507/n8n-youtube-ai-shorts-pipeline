"""Serial theme rotation with persisted cursor in records/."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT

logger = logging.getLogger(__name__)

RECORDS_DIR = ROOT / "records"
STATE_PATH = RECORDS_DIR / "theme_rotation.json"


def normalize_theme_pool(themes: list[str]) -> list[str]:
    pool = [t.strip().lower() for t in themes if t and str(t).strip()]
    if not pool:
        raise ValueError(
            "No themes configured. Add a themes: list in default.yaml or pass --themes-csv."
        )
    return pool


def pool_key(pool: list[str]) -> str:
    return "|".join(pool)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"pools": {}, "pending": None}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"pools": {}, "pending": None}
        data.setdefault("pools", {})
        data.setdefault("pending", None)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s — resetting rotation state", STATE_PATH, exc)
        return {"pools": {}, "pending": None}


def _save_state(state: dict[str, Any]) -> None:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _pool_entry(state: dict[str, Any], key: str, pool: list[str]) -> dict[str, Any]:
    entry = state["pools"].get(key) or {}
    if entry.get("themes") != pool:
        entry = {"themes": pool, "next_index": 0}
        state["pools"][key] = entry
    return entry


def allocate_serial_theme(pool: list[str]) -> tuple[str, str, int]:
    """
    Pick the next theme in serial order without advancing the cursor.

    Returns (theme, pool_key, index). Call commit_serial_theme() after script succeeds.
    Retries reuse the same pending allocation until committed.
    """
    normalized = normalize_theme_pool(pool)
    key = pool_key(normalized)
    state = _load_state()
    pending = state.get("pending")

    if pending and pending.get("pool_key") == key:
        idx = int(pending["index"])
        theme = normalized[idx % len(normalized)]
        logger.info("Theme rotation (retry): %s [%d/%d]", theme, idx + 1, len(normalized))
        return theme, key, idx

    entry = _pool_entry(state, key, normalized)
    idx = int(entry.get("next_index", 0)) % len(normalized)
    theme = normalized[idx]
    state["pending"] = {
        "pool_key": key,
        "index": idx,
        "theme": theme,
        "allocated_at": _now_iso(),
    }
    _save_state(state)
    logger.info("Theme rotation: %s [%d/%d]", theme, idx + 1, len(normalized))
    return theme, key, idx


def clear_pending() -> None:
    """Drop an uncommitted allocation (e.g. user switched to an explicit theme)."""
    state = _load_state()
    if state.get("pending"):
        state["pending"] = None
        _save_state(state)


def commit_serial_theme(
    pool: list[str],
    *,
    pool_key_override: str | None = None,
    expected_theme: str | None = None,
) -> str | None:
    """Advance cursor after a successful script stage. No-op if nothing pending."""
    normalized = normalize_theme_pool(pool)
    key = pool_key_override or pool_key(normalized)
    state = _load_state()
    pending = state.get("pending")

    if not pending or pending.get("pool_key") != key:
        return None
    if expected_theme and str(pending.get("theme", "")).lower() != expected_theme.strip().lower():
        return None

    idx = int(pending["index"])
    theme = str(pending.get("theme") or normalized[idx % len(normalized)])
    entry = _pool_entry(state, key, normalized)
    entry["next_index"] = (idx + 1) % len(normalized)
    entry["last_theme"] = theme
    entry["updated_at"] = _now_iso()
    state["pools"][key] = entry
    state["pending"] = None
    _save_state(state)
    logger.info(
        "Theme rotation advanced — next up: %s",
        normalized[entry["next_index"] % len(normalized)],
    )
    return theme


def rotation_status(pool: list[str] | None = None) -> dict[str, Any]:
    """Inspect rotation state (for debugging)."""
    state = _load_state()
    if pool is None:
        return state
    normalized = normalize_theme_pool(pool)
    key = pool_key(normalized)
    entry = state["pools"].get(key, {})
    next_idx = int(entry.get("next_index", 0)) % len(normalized)
    return {
        "pool_key": key,
        "themes": normalized,
        "next_index": next_idx,
        "next_theme": normalized[next_idx],
        "last_theme": entry.get("last_theme"),
        "pending": state.get("pending"),
    }
