"""Isolated Wan video generation — run in a fresh process to avoid OOM after TTS/music."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .run_io import prepare_worker_job
from .video_wan import generate_clip, generate_wan_video

logger = logging.getLogger(__name__)


def run_wan_job(
    output_dir: Path,
    config_path: str | None = None,
    scene_index: int | None = None,
) -> list[Path]:
    out, _script, config, scenes = prepare_worker_job(output_dir, "wan", config_path)

    if scene_index is not None:
        if scene_index < 0 or scene_index >= len(scenes):
            raise IndexError(f"scene_index {scene_index} out of range (0-{len(scenes) - 1})")
        scene = scenes[scene_index]
        clip_path = out / "clips" / f"scene_{scene_index + 1:02d}.mp4"
        prompt = scene.get("video_prompt") or scene.get("image_prompt", "")
        duration = float(scene.get("duration_sec", 5.0))
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
