"""FLUX.2 Klein image generation — runs in image-venv (transformers 5 + diffusers 0.38+).

Not compatible with flux-venv (parler-tts pins transformers 4.46).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

DEFAULT_STYLE_ANCHOR = "3D Pixar-style animation, vibrant colors, cinematic lighting, high detail. "


def _device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def _script_seed(title: str, prompts: list[str]) -> int:
    payload = json.dumps({"title": title, "image_prompts": prompts}, sort_keys=True)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _prompt_for_model(style_anchor: str, scene_prompt: str) -> str:
    anchor = (style_anchor or DEFAULT_STYLE_ANCHOR).strip()
    if anchor and not anchor.endswith((". ", ".")):
        anchor = anchor.rstrip(".") + ". "
    return anchor + json.dumps(scene_prompt)


def _load_reference_image(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    from PIL import Image

    return Image.open(p).convert("RGB")


def _load_pipeline(
    model_id: str,
    local_files_only: bool,
    device: str,
    torch_dtype: torch.dtype,
    cpu_offload: bool = False,
):
    from diffusers import Flux2KleinPipeline

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    logger.info("Loading FLUX.2 Klein: %s on %s", model_id, device)
    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
    except Exception:
        if not local_files_only:
            raise
        logger.warning("Local cache miss for %s; downloading from Hub", model_id)
        pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)

    if cpu_offload:
        offload_device = device if device in ("cuda", "mps") else "cpu"
        logger.info("FLUX CPU offload enabled (device=%s)", offload_device)
        pipe.enable_model_cpu_offload(device=offload_device)
    else:
        pipe.to(device=device)
        if device == "cuda":
            pipe.enable_model_cpu_offload()
    return pipe


def _unload_pipeline(pipe, device: str) -> None:
    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def generate_from_spec(spec: dict, output_dir: Path) -> list[Path]:
    """Generate scenes with one FLUX load — reuses pipeline for all prompts in spec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts: list[str] = spec.get("prompts") or []
    if not prompts:
        raise ValueError("No prompts in spec")

    title = spec.get("title", "")
    style_anchor = spec.get("style_anchor", DEFAULT_STYLE_ANCHOR)
    model_id = spec.get("flux_model", "black-forest-labs/FLUX.2-klein-4B")
    width = int(spec.get("flux_gen_width", 512))
    height = int(spec.get("flux_gen_height", 912))
    steps = int(spec.get("flux_steps", 25))
    guidance_scale = float(spec.get("flux_guidance", 3.5))
    max_sequence_length = int(spec.get("flux_max_sequence_length", 256))
    local_files_only = bool(spec.get("local_files_only", True))
    reference_mode = str(spec.get("flux_reference_mode", "first")).lower().strip()
    cpu_offload = bool(spec.get("flux_cpu_offload", False))
    scene_start = int(spec.get("scene_index", 0))
    all_prompts: list[str] = spec.get("all_prompts") or prompts

    device, torch_dtype = _device_and_dtype()
    pipe = _load_pipeline(model_id, local_files_only, device, torch_dtype, cpu_offload=cpu_offload)
    seed = _script_seed(title, all_prompts)
    gen_device = "cpu" if device == "mps" else device

    reference_image = _load_reference_image(spec.get("reference_image"))
    paths: list[Path] = []
    anchor_path = output_dir / f"scene_{scene_start + 1:02d}.png"

    try:
        for offset, scene in enumerate(prompts):
            scene_num = scene_start + offset + 1
            prompt_for_model = _prompt_for_model(style_anchor, scene)
            logger.info("Scene %d prompt: %s", scene_num, prompt_for_model[:160])

            scene_seed = (seed + offset * 7919) % (2**32)
            pipe_kw: dict = dict(
                prompt=prompt_for_model,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator(device=gen_device).manual_seed(scene_seed),
            )

            ref_for_scene = None
            ref_path: Path | None = None
            if reference_mode == "chain" and reference_image is not None:
                ref_for_scene = reference_image
            elif reference_mode == "first" and offset > 0 and anchor_path.is_file():
                ref_path = anchor_path
                ref_for_scene = _load_reference_image(str(ref_path))
            elif reference_mode == "chain" and offset > 0:
                prev = output_dir / f"scene_{scene_num - 1:02d}.png"
                if prev.is_file():
                    ref_path = prev
                    ref_for_scene = _load_reference_image(str(ref_path))

            if ref_for_scene is not None:
                pipe_kw["image"] = ref_for_scene
                logger.info("Using reference image for scene %d (mode=%s)", scene_num, reference_mode)

            image = pipe(**pipe_kw).images[0]
            out_path = output_dir / f"scene_{scene_num:02d}.png"
            image.save(str(out_path), format="PNG")
            paths.append(out_path)

            if reference_mode == "chain" and reference_image is not None and reference_image is not image:
                del reference_image
            if reference_mode == "chain":
                reference_image = image
            if ref_for_scene is not None and ref_for_scene is not image:
                del ref_for_scene

            del image
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            logger.info("Saved %s", out_path)
    finally:
        _unload_pipeline(pipe, device)

    return paths


def main() -> None:
    from .log_format import configure_logging

    configure_logging(level=logging.INFO)
    parser = argparse.ArgumentParser(description="FLUX.2 Klein image generation")
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    spec = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    generate_from_spec(spec, Path(args.output_dir))


if __name__ == "__main__":
    main()
