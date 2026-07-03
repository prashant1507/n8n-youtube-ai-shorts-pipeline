"""Wan2.2 text-to-video via mlx-gen CLI (Apple Silicon)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from .config import ROOT, PipelineConfig
from .media import X264_QUALITY, probe_duration
from .venv_paths import wan_python

logger = logging.getLogger(__name__)

DEFAULT_WAN_MODEL = ROOT / "models" / "wan2.2-ti2v-5b"
WAN_FPS = 8
WAN_GEN_FRAMES = 13  # ~1.6s clip; extended to scene duration after generation


def _wan_python(config: PipelineConfig) -> Path:
    return wan_python(config)


def _extend_clip_to_duration(clip_path: Path, target_sec: float) -> Path:
    """Loop/slow a short Wan clip to match scene narration duration."""
    if target_sec <= 0:
        return clip_path
    src_dur = probe_duration(clip_path)
    if src_dur >= target_sec * 0.95:
        return clip_path
    # Slow playback to fill target duration (smoother than hard loop)
    factor = target_sec / max(src_dur, 0.1)
    tmp = clip_path.with_suffix(".extended.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path),
        "-filter:v", f"setpts={factor:.4f}*PTS",
        "-an",
        "-t", str(target_sec),
        "-c:v", "libx264", *X264_QUALITY,
        "-pix_fmt", "yuv420p",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg extend failed: {(result.stderr or result.stdout).strip()}")
    tmp.replace(clip_path)
    return clip_path


def _mlxgen_bin() -> Path | None:
    """Prefer project wan-venv mlxgen, then PATH."""
    local = ROOT / "wan-venv" / "bin" / "mlxgen"
    if local.is_file():
        return local
    found = shutil.which("mlxgen")
    return Path(found) if found else None


def _resolve_wan_model(config: PipelineConfig) -> str:
    configured = (config.video.wan_model or "").strip()
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            p = ROOT / configured
        if p.exists():
            return str(p.resolve())
        if configured.startswith(("Wan-AI/", "AbstractFramework/")):
            return configured
    if DEFAULT_WAN_MODEL.exists():
        return str(DEFAULT_WAN_MODEL.resolve())
    return "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def _mlxgen_available() -> bool:
    return _mlxgen_bin() is not None


def generate_clip(
    prompt: str,
    config: PipelineConfig,
    output_path: Path,
    duration_hint: float = 5.0,
) -> Path:
    """Generate one Wan2.2 clip via mlxgen."""
    mlxgen = _mlxgen_bin()
    if mlxgen is None:
        raise RuntimeError(
            "mlxgen not found. Install in wan-venv: pip install mlx-gen\n"
            f"Expected: {ROOT / 'wan-venv' / 'bin' / 'mlxgen'}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anchor = (config.video.style_anchor or config.video.style_suffix).strip().rstrip(".")
    scene = prompt.strip()
    full_prompt = f"{anchor}. {scene}" if scene else anchor

    model = _resolve_wan_model(config)
    model_path = Path(model)
    if not model_path.exists() and model.startswith("Wan-AI/"):
        raise RuntimeError(
            f"Wan model not prepared: {model}\n"
            f"Run: {mlxgen} prepare --model {model} "
            f"--path {DEFAULT_WAN_MODEL} -q 8"
        )

    frames = WAN_GEN_FRAMES
    width = min(config.width, 640)
    height = min(config.height, 360)

    cmd = [
        str(mlxgen), "generate",
        "--model", model,
        "--task", "t2v",
        "--prompt", full_prompt,
        "--output", str(output_path),
        "--frames", str(frames),
        "--fps", str(WAN_FPS),
        "--width", str(width),
        "--height", str(height),
        "--steps", "12",
        "--no-progress",
    ]
    logger.info("Running Wan2.2 (%s frames → %.1fs target): %s", frames, duration_hint, scene[:80])
    env = os.environ.copy()
    env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"mlxgen failed: {msg}")

    if not output_path.exists():
        candidates = sorted(output_path.parent.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if candidates:
            candidates[-1].rename(output_path)
        else:
            raise FileNotFoundError(f"Wan clip not created at {output_path}")

    _extend_clip_to_duration(output_path, duration_hint)
    logger.info("Wan clip ready: %s (%.1fs)", output_path.name, probe_duration(output_path))
    return output_path


def generate_wan_video(
    scenes: list[dict],
    config: PipelineConfig,
    output_dir: Path,
) -> list[Path]:
    clips_dir = output_dir / "clips"
    clips: list[Path] = []
    for i, scene in enumerate(scenes):
        prompt = scene.get("video_prompt") or scene.get("image_prompt", "")
        duration = float(scene.get("duration_sec", 5.0))
        clip_path = clips_dir / f"scene_{i + 1:02d}.mp4"
        generate_clip(prompt, config, clip_path, duration_hint=duration)
        clips.append(clip_path)
    return clips
