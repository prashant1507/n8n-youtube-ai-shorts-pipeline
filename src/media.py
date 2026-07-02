"""Shared ffprobe / scene timing helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


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


def align_scene_durations(scenes: list[dict], voice_path: Path) -> list[dict]:
    """Scale per-scene duration_sec to match actual voice length."""
    total_voice = probe_duration(voice_path)
    weights = [max(1, len(s.get("narration_segment", ""))) for s in scenes]
    weight_sum = sum(weights) or len(scenes)
    aligned: list[dict] = []
    for scene, weight in zip(scenes, weights):
        s = dict(scene)
        s["duration_sec"] = round(total_voice * weight / weight_sum, 1)
        aligned.append(s)
    return aligned
