"""Storybook images via FLUX.2 Klein (n8n pattern) + Ken Burns slideshow."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .config import ROOT, PipelineConfig
from .venv_paths import flux_python
from .visual_prompts import prepare_scene_prompts

logger = logging.getLogger(__name__)


def _flux_python(config: PipelineConfig) -> Path:
    return flux_python(config)


def _scene_prompts_for_flux(scenes: list[dict]) -> list[str]:
    """Raw LLM scene descriptions — style anchor applied in flux_klein."""
    prompts: list[str] = []
    for scene in scenes:
        raw = (scene.get("image_prompt") or scene.get("video_prompt") or "").strip()
        if not raw:
            raw = scene.get("narration_segment", "").strip()
        prompts.append(raw)
    return prompts


def _flux_in_process() -> bool:
    """True only inside image-venv where transformers 5 + FLUX diffusers stack match."""
    import sys

    image_py = ROOT / "image-venv" / "bin" / "python"
    if not image_py.is_file():
        return False
    if Path(sys.executable).resolve() != image_py.resolve():
        return False
    try:
        from diffusers import Flux2KleinPipeline  # noqa: F401

        return True
    except (ImportError, RuntimeError, OSError):
        return False


def _run_flux_klein_batch(
    prompts: list[str],
    images_dir: Path,
    config: PipelineConfig,
    title: str,
) -> None:
    """Load FLUX once in-process (image-venv) or via one subprocess (flux-venv caller)."""
    images_dir.mkdir(parents=True, exist_ok=True)
    log_path = images_dir / "flux.log"

    spec = {
        "title": title,
        "all_prompts": prompts,
        "prompts": prompts,
        "scene_index": 0,
        "style_anchor": (config.video.style_anchor or config.video.style_suffix or "").strip(),
        "flux_model": config.video.flux_model,
        "flux_gen_width": config.video.flux_gen_width,
        "flux_gen_height": config.video.flux_gen_height,
        "flux_steps": config.video.flux_steps,
        "flux_guidance": config.video.flux_guidance,
        "flux_max_sequence_length": config.video.flux_max_sequence_length,
        "local_files_only": config.video.flux_local_files_only,
        "flux_reference_mode": config.video.flux_reference_mode,
        "flux_cpu_offload": config.video.flux_cpu_offload,
    }
    spec_path = images_dir / "_flux_batch.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")

    if _flux_in_process():
        from .flux_klein import generate_from_spec

        logger.info("FLUX batch: %d scene(s), in-process (single Python, single model load)", len(prompts))
        generate_from_spec(spec, images_dir)
    else:
        python = _flux_python(config)
        env = {
            **dict(__import__("os").environ),
            "PYTHONPATH": str(ROOT),
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
        }
        cmd = [
            str(python),
            "-m", "src.flux_klein",
            "--prompts-json", str(spec_path),
            "--output-dir", str(images_dir),
        ]
        logger.info("FLUX batch: %d scene(s), image-venv subprocess", len(prompts))
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            tail = log_path.read_text(encoding="utf-8")[-4000:] if log_path.exists() else ""
            raise RuntimeError(f"FLUX.2 Klein batch failed (see {log_path}):\n{tail}")

    for i in range(len(prompts)):
        out_img = images_dir / f"scene_{i + 1:02d}.png"
        if not out_img.exists():
            raise FileNotFoundError(f"FLUX did not write {out_img}")


def _ken_burns_clip(
    image_path: Path,
    output_path: Path,
    duration_sec: float,
    width: int,
    height: int,
    preset_idx: int = 0,
    motion_zoom: float = 0.18,
    supersample: float = 2.0,
    fps: int = 30,
) -> Path:
    """Ken Burns with supersampling (n8n Video._image_filter pattern)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    d = max(2, int(round(duration_sec * fps)))
    zf = motion_zoom
    sw = (int(round(width * supersample)) // 2) * 2
    sh = (int(round(height * supersample)) // 2) * 2
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    preset = preset_idx % 4
    if preset == 0:
        z, x, y = f"1.0+{zf}*on/{d}", cx, cy
    elif preset == 1:
        z, x, y = f"1.0+{zf}", f"(iw-iw/zoom)*on/{d}", cy
    elif preset == 2:
        z, x, y = f"1.0+{zf}", cx, f"(ih-ih/zoom)*on/{d}"
    else:
        z, x, y = f"1.0+{zf}-{zf}*on/{d}", cx, cy

    vf = (
        f"scale={sw}:{sh}:flags=lanczos,setsar=1,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={d}:s={sw}x{sh}:fps={fps},"
        f"scale={width}:{height}:flags=lanczos,format=yuv420p,setsar=1"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(image_path),
        "-vf", vf,
        "-t", str(duration_sec),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def _prepare_flux_script(
    scenes: list[dict],
    config: PipelineConfig,
    output_dir: Path,
    script: dict | None = None,
) -> tuple[dict, list[str]]:
    script = script or {"scenes": scenes, "title": config.theme}
    script, prompt_log = prepare_scene_prompts(script, config, enrich=False)
    (output_dir / "image_prompts.json").write_text(
        json.dumps({"prompts": prompt_log}, indent=2, ensure_ascii=False)
    )
    (output_dir / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False))
    return script, _scene_prompts_for_flux(scenes)


def generate_images(
    scenes: list[dict],
    config: PipelineConfig,
    output_dir: Path,
    script: dict | None = None,
) -> list[Path]:
    """FLUX.2 Klein scene images only (no Ken Burns clips)."""
    images_dir = output_dir / "images"
    script, flux_prompts = _prepare_flux_script(scenes, config, output_dir, script)
    _run_flux_klein_batch(
        flux_prompts,
        images_dir,
        config,
        script.get("title", config.theme),
    )
    return [images_dir / f"scene_{i + 1:02d}.png" for i in range(len(scenes))]


def generate_clips_from_images(
    scenes: list[dict],
    config: PipelineConfig,
    output_dir: Path,
) -> list[Path]:
    """Ken Burns clips from existing FLUX images."""
    images_dir = output_dir / "images"
    clips_dir = output_dir / "clips"
    clips: list[Path] = []
    motion_zoom = config.video.motion_zoom
    supersample = config.video.motion_supersample
    fps = config.video.ken_burns_fps

    for i, scene in enumerate(scenes):
        duration = float(scene.get("duration_sec", config.duration_sec / len(scenes)))
        img_path = images_dir / f"scene_{i + 1:02d}.png"
        clip_path = clips_dir / f"scene_{i + 1:02d}.mp4"
        if not img_path.exists():
            raise FileNotFoundError(f"FLUX image missing: {img_path}")
        logger.info("Ken Burns scene %d (%.1fs)", i + 1, duration)
        _ken_burns_clip(
            img_path, clip_path, duration,
            config.width, config.height,
            preset_idx=i,
            motion_zoom=motion_zoom,
            supersample=supersample,
            fps=fps,
        )
        clips.append(clip_path)

    return clips


def generate_slideshow(
    scenes: list[dict],
    config: PipelineConfig,
    output_dir: Path,
    script: dict | None = None,
) -> list[Path]:
    script, _ = _prepare_flux_script(scenes, config, output_dir, script)
    generate_images(scenes, config, output_dir, script=script)
    return generate_clips_from_images(scenes, config, output_dir)


def unload_model() -> None:
    """No in-process model; FLUX runs in a subprocess."""
    pass
