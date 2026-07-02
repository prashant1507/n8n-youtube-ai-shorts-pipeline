"""Subprocess helpers for isolated FLUX / Wan workers."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .config import ROOT, PipelineConfig
from .memory import release_gpu_memory
from .run_io import load_script
from .venv_paths import flux_python, wan_python

logger = logging.getLogger(__name__)


def _run_module(
    python: Path,
    module: str,
    output_dir: Path,
    config_path: str | None,
    extra_args: list[str] | None = None,
) -> dict:
    cmd = [str(python), "-m", module, "--output-dir", str(output_dir)]
    if config_path:
        cmd.extend(["--config", config_path])
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(msg)
    return json.loads(result.stdout)


def run_flux_worker(output_dir: Path, config: PipelineConfig, config_path: str | None = None) -> list[Path]:
    release_gpu_memory()
    python = flux_python(config)
    logger.info("Starting FLUX worker in image-venv...")
    payload = _run_module(python, "src.flux_worker", output_dir, config_path)
    return [Path(p) for p in payload["clips"]]


def run_wan_worker(output_dir: Path, config: PipelineConfig, config_path: str | None = None) -> list[Path]:
    release_gpu_memory()
    python = wan_python(config)
    script = load_script(output_dir)
    scene_count = len(script.get("scenes") or [])
    if scene_count == 0:
        raise ValueError(f"No scenes in {output_dir / 'script.json'}")

    clips: list[Path] = []
    for i in range(scene_count):
        release_gpu_memory()
        logger.info("Wan scene %d/%d in wan-venv subprocess...", i + 1, scene_count)
        payload = _run_module(
            python,
            "src.wan_worker",
            output_dir,
            config_path,
            extra_args=["--scene-index", str(i)],
        )
        clips.extend(Path(p) for p in payload["clips"])
    return clips
