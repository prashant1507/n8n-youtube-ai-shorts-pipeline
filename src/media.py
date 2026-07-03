"""Shared ffprobe / scene timing helpers and encode quality settings."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Frames pass through several x264 generations (Ken Burns, overlays, concat, mux, loop);
# defaults (CRF 23) compound visibly, so intermediate encodes use a higher quality.
X264_QUALITY: tuple[str, ...] = ("-crf", "18", "-preset", "medium")
AAC_BITRATE: tuple[str, ...] = ("-b:a", "192k")

VOICE_TIMINGS_NAME = "voice_timings.json"


def probe_duration(path: Path) -> float:
    """Return media duration in seconds."""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def _voice_segment_durations(voice_path: Path) -> list[float] | None:
    """Exact per-segment TTS durations written by generate_voice, if present."""
    timings_path = voice_path.parent / VOICE_TIMINGS_NAME
    if not timings_path.is_file():
        return None
    try:
        data = json.loads(timings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    durations = data.get("segment_durations") or []
    if not durations or not all(
        isinstance(d, (int, float)) and d > 0 for d in durations
    ):
        return None
    return [float(d) for d in durations]


def align_scene_durations(scenes: list[dict], voice_path: Path) -> list[dict]:
    """Set per-scene duration_sec from exact TTS segment timings, else estimate.

    Voice is synthesized per scene segment, so voice_timings.json gives the true
    per-scene durations. Fallback (older runs, single-segment narration): scale by
    narration character count to match total voice length.
    """
    exact = _voice_segment_durations(voice_path)
    if exact and len(exact) == len(scenes):
        return [
            {**scene, "duration_sec": round(dur, 2)}
            for scene, dur in zip(scenes, exact)
        ]

    total_voice = probe_duration(voice_path)
    weights = [max(1, len(s.get("narration_segment", ""))) for s in scenes]
    weight_sum = sum(weights) or len(scenes)
    aligned: list[dict] = []
    for scene, weight in zip(scenes, weights):
        s = dict(scene)
        s["duration_sec"] = round(total_voice * weight / weight_sum, 1)
        aligned.append(s)
    return aligned
