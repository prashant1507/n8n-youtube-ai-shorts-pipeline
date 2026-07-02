"""Generate scene clips — tier routing shared by pipeline stages."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import PipelineConfig
from .memory import release_gpu_memory
from .video_flux import generate_clips_from_images
from .workers import run_flux_worker, run_wan_worker

logger = logging.getLogger(__name__)


def generate_clips(
    output_dir: Path,
    config: PipelineConfig,
    scenes: list[dict],
    *,
    flux_subprocess: bool = False,
    config_path: str | None = None,
) -> list[Path]:
    """
    Generate scene MP4s for the configured video tier.

    flux_subprocess: when True, run full FLUX slideshow worker (images + Ken Burns).
    When False, build Ken Burns clips from existing images/ (images stage already ran).
    """
    tier = config.video.tier.lower()
    if tier == "wan":
        try:
            return run_wan_worker(output_dir, config, config_path)
        except Exception as exc:
            logger.warning("Wan2.2 failed (%s), falling back to FLUX slideshow", exc)
            return run_flux_worker(output_dir, config, config_path)

    if flux_subprocess:
        return run_flux_worker(output_dir, config, config_path)

    if not any((output_dir / "images").glob("scene_*.png")):
        raise FileNotFoundError("No scene images — run images stage first")
    release_gpu_memory()
    return generate_clips_from_images(scenes, config, output_dir)
