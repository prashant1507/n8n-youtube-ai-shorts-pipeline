"""Title style A/B rotation: assign one packaging style per run.

Each upload gets a title style (question, cliffhanger, ...) via serial rotation
stored in records/. The style is saved into script.json and the uploads
registry, so youtube_analytics can compare view performance per style.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import ROOT

logger = logging.getLogger(__name__)

RECORDS_DIR = ROOT / "records"
STATE_PATH = RECORDS_DIR / "title_style_rotation.json"

# Style key -> instruction the SEO LLM sees when writing that style's title.
TITLE_STYLE_DEFINITIONS: dict[str, str] = {
    "question": (
        "A direct question the viewer can only answer by watching "
        "(e.g. 'What Would You Do With a Cloud in a Jar?')."
    ),
    "cliffhanger": (
        "Freeze the moment right before the turning point; never reveal what happens next "
        "(e.g. 'She Opened the Door and Everything Changed')."
    ),
    "curiosity": (
        "A bold, specific claim with a curiosity gap "
        "(e.g. 'This Tiny Robot Saved an Entire City')."
    ),
    "character": (
        "Lead with the protagonist and their unusual trait or problem "
        "(e.g. 'The Dragon Who Was Afraid of Fire')."
    ),
    "emotional": (
        "Tease the warm, feel-good payoff without spoiling it "
        "(e.g. 'The Kindest Thing a Pirate Ever Did')."
    ),
}


def style_definition(style: str) -> str:
    return TITLE_STYLE_DEFINITIONS.get(
        style.lower().strip(),
        "A strong, honest, curiosity-driving title in that spirit.",
    )


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", STATE_PATH, exc)
        return {}


def _save_state(state: dict[str, Any]) -> None:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def allocate_title_style(styles: list[str]) -> str:
    """Pick the next style in serial rotation and advance the counter."""
    pool = [s.lower().strip() for s in styles if s and s.strip()]
    if not pool:
        return ""
    state = _load_state()
    index = int(state.get("index", 0))
    style = pool[index % len(pool)]
    _save_state({"index": (index + 1) % len(pool), "last_style": style})
    logger.info("Title A/B style for this run: %s", style)
    return style
