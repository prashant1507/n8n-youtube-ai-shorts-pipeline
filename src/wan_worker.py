"""Isolated Wan video generation — run in a fresh process to avoid OOM after TTS/music."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_config
from .video_wan import generate_wan_video

logger = logging.getLogger(__name__)


def _probe_duration(path: Path) -> float:
    import subprocess

    return float(
        subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            text=True,
        ).strip()
    )


def _align_scene_durations(scenes: list[dict], voice_path: Path) -> list[dict]:
    total_voice = _probe_duration(voice_path)
    weights = [max(1, len(s.get("narration_segment", ""))) for s in scenes]
    weight_sum = sum(weights) or len(scenes)
    aligned = []
    for scene, weight in zip(scenes, weights):
        s = dict(scene)
        s["duration_sec"] = round(total_voice * weight / weight_sum, 1)
        aligned.append(s)
    return aligned


def run_wan_job(
    output_dir: Path,
    config_path: str | None = None,
    scene_index: int | None = None,
) -> list[Path]:
    out = output_dir.resolve()
    script_path = out / "script.json"
    voice_path = out / "voice.wav"
    if not script_path.exists():
        raise FileNotFoundError(f"No script.json in {out}")
    if not voice_path.exists():
        raise FileNotFoundError(f"No voice.wav in {out}")

    script = json.loads(script_path.read_text(encoding="utf-8"))
    overrides: dict = {"video": {"tier": "wan"}}
    if script.get("content_type") or script.get("theme"):
        overrides["theme"] = script.get("content_type") or script.get("theme")
    if script.get("language"):
        lang = str(script["language"]).lower()
        overrides["language"] = "hi" if lang.startswith("hi") else "en"

    config = load_config(config_path, **overrides)
    scenes = _align_scene_durations(script["scenes"], voice_path)

    if scene_index is not None:
        if scene_index < 0 or scene_index >= len(scenes):
            raise IndexError(f"scene_index {scene_index} out of range (0-{len(scenes) - 1})")
        scene = scenes[scene_index]
        clips_dir = out / "clips"
        prompt = scene.get("video_prompt") or scene.get("image_prompt", "")
        duration = float(scene.get("duration_sec", 5.0))
        clip_path = clips_dir / f"scene_{scene_index + 1:02d}.mp4"
        from .video_wan import generate_clip

        generate_clip(prompt, config, clip_path, duration_hint=duration)
        return [clip_path]

    return generate_wan_video(scenes, config, out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Wan2.2 clip generation (isolated subprocess)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--scene-index", type=int, default=None, help="Generate only this scene (0-based)")
    args = parser.parse_args()

    clips = run_wan_job(Path(args.output_dir), args.config, scene_index=args.scene_index)
    print(json.dumps({"clips": [str(c) for c in clips]}, indent=2))


if __name__ == "__main__":
    main()
