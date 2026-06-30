"""End-to-end narration video pipeline orchestrator."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from .assembler import (
    assemble_pipeline_output,
    audio_mix_stage,
    concat_video_stage,
    final_mux_stage,
    list_scene_clips,
    subtitles_on_clips_stage,
)
from .config import (
    ROOT,
    PipelineConfig,
    apply_subtitle_policy,
    is_hindi_language,
    load_config,
    require_theme,
    resolve_theme,
)
from .story_registry import register_story
from .memory import release_gpu_memory
from .script_generator import generate_script

logger = logging.getLogger(__name__)
OUTPUT_DIR = ROOT / "output"


def _run_wan_subprocess(
    output_dir: Path,
    config: PipelineConfig,
    config_path: str | None = None,
) -> list[Path]:
    """Run Wan one scene per wan-venv subprocess to limit peak unified memory."""
    import subprocess

    from .video_wan import _wan_python

    release_gpu_memory()
    python = _wan_python(config)
    script = json.loads((output_dir / "script.json").read_text(encoding="utf-8"))
    scene_count = len(script.get("scenes") or [])
    if scene_count == 0:
        raise ValueError(f"No scenes in {output_dir / 'script.json'}")

    clips: list[Path] = []
    for i in range(scene_count):
        release_gpu_memory()
        cmd = [
            str(python),
            "-m", "src.wan_worker",
            "--output-dir", str(output_dir),
            "--scene-index", str(i),
        ]
        if config_path:
            cmd.extend(["--config", config_path])
        logger.info("Wan scene %d/%d in wan-venv subprocess...", i + 1, scene_count)
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"Wan worker scene {i + 1} failed: {msg}")
        payload = json.loads(result.stdout)
        clips.extend(Path(p) for p in payload["clips"])
    return clips


def _run_flux_subprocess(
    output_dir: Path,
    config: PipelineConfig,
    config_path: str | None = None,
) -> list[Path]:
    """Run FLUX + Ken Burns in image-venv so only one heavy Python process holds the model."""
    import subprocess

    from .video_flux import _flux_python

    release_gpu_memory()
    python = _flux_python(config)
    cmd = [str(python), "-m", "src.flux_worker", "--output-dir", str(output_dir)]
    if config_path:
        cmd.extend(["--config", config_path])
    logger.info("Starting FLUX worker in image-venv (frees flux-venv TTS/music memory)...")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"FLUX worker failed: {msg}")
    payload = json.loads(result.stdout)
    return [Path(p) for p in payload["clips"]]


ProgressCallback = Callable[[str, float, str], None]

PIPELINE_STAGES = (
    "script",
    "voice",
    "music",
    "images",
    "clips",
    "video",
    "subtitles",
    "audio_mix",
    "final",
)


def _resolve_output_dir(run_id: str | None, output_dir: Path | None) -> Path:
    if output_dir is not None:
        out = output_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out
    if not run_id:
        raise ValueError("run_id or output_dir is required for pipeline steps after script")
    out = (OUTPUT_DIR / Path(run_id).name).resolve()
    if not str(out).startswith(str(OUTPUT_DIR.resolve())):
        raise ValueError("invalid run_id")
    if not out.is_dir():
        raise FileNotFoundError(f"Run folder not found: {out}")
    return out


def _load_script(out: Path) -> dict:
    script_path = out / "script.json"
    if not script_path.exists():
        raise FileNotFoundError(f"No script.json in {out}")
    return json.loads(script_path.read_text(encoding="utf-8"))


def _sync_config_from_script(config: PipelineConfig, script: dict) -> None:
    if script.get("content_type") or script.get("theme"):
        config.theme = str(script.get("content_type") or script.get("theme"))
    lang = str(script.get("language", config.language)).lower()
    if lang.startswith("hi"):
        config.language = "hi"
    elif lang.startswith("en"):
        config.language = "en"
    apply_subtitle_policy(config, script)


def _stage_result(
    stage: str,
    out: Path,
    *,
    skipped: bool = False,
    extra: dict | None = None,
) -> dict:
    payload: dict = {
        "stage": stage,
        "run_id": out.name,
        "output_dir": str(out),
        "skipped": skipped,
    }
    script_path = out / "script.json"
    if script_path.exists():
        script = json.loads(script_path.read_text(encoding="utf-8"))
        payload["script"] = script
        payload["lang"] = script.get("language", "")
        payload["theme"] = script.get("content_type") or script.get("theme", "")
    for name in (
        "voice.wav",
        "music.wav",
        "video_raw.mp4",
        "video_subtitled.mp4",
        "audio_mixed.wav",
        "final.mp4",
        "final_subtitled.mp4",
    ):
        path = out / name
        if path.is_file():
            key = name.replace(".", "_").replace("-", "_")
            payload[key] = str(path)
    final = out / "final_subtitled.mp4"
    if not final.is_file():
        final = out / "final.mp4"
    if final.is_file():
        payload["final_video"] = str(final)
    if extra:
        payload.update(extra)
    return payload


def run_stage(
    stage: str,
    config: PipelineConfig,
    *,
    output_dir: Path | None = None,
    run_id: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Run a single pipeline stage. Returns metadata including run_id."""
    progress = on_progress or _default_progress
    stage = stage.strip().lower()
    if stage not in PIPELINE_STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {', '.join(PIPELINE_STAGES)}")

    if stage == "script":
        if output_dir is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            out = OUTPUT_DIR / run_id
        else:
            out = _resolve_output_dir(run_id, output_dir)
        out.mkdir(parents=True, exist_ok=True)

        progress("script", 0.05, "Generating narration script...")
        theme = require_theme(config)
        logger.info("Content type: %s", theme)
        (out / "theme.txt").write_text(config.theme, encoding="utf-8")
        script = generate_script(config)
        (out / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False))
        register_story(
            script.get("narration") or script.get("script", ""),
            script.get("title", ""),
            script.get("content_type") or script.get("theme", config.theme),
            script.get("language", config.language),
            out.name,
        )
        apply_subtitle_policy(config, script)
        return _stage_result("script", out)

    out = _resolve_output_dir(run_id, output_dir)
    script = _load_script(out)
    _sync_config_from_script(config, script)
    scenes = script["scenes"]
    voice_path = out / "voice.wav"
    music_path = out / "music.wav"
    narration = script["narration"]

    if stage == "voice":
        from .tts import generate_voice, unload_model as unload_tts

        progress("voice", 0.20, "Generating voice...")
        segments = [s.get("narration_segment", "") for s in scenes]
        generate_voice(
            narration,
            config,
            voice_path,
            scene_segments=segments if len(segments) > 1 else None,
        )
        unload_tts()
        release_gpu_memory()
        return _stage_result("voice", out)

    if stage == "music":
        if not voice_path.exists():
            raise FileNotFoundError(f"Missing {voice_path} — run voice stage first")
        from .music import generate_music, unload_model as unload_music

        progress("music", 0.40, "Generating background music...")
        import subprocess

        voice_dur = float(
            subprocess.check_output(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(voice_path)],
                text=True,
            ).strip()
        )
        music_prompt = script.get("music_prompt", config.music.prompt)
        generate_music(music_prompt, voice_dur + 2, config, music_path)
        unload_music()
        release_gpu_memory()
        return _stage_result("music", out)

    scenes = _align_scene_durations(scenes, voice_path)
    (out / "script.json").write_text(json.dumps({**script, "scenes": scenes}, indent=2, ensure_ascii=False))

    if stage == "images":
        if not voice_path.exists():
            raise FileNotFoundError(f"Missing {voice_path} — run voice stage first")
        tier = config.video.tier.lower()
        if tier == "wan":
            logger.info("Wan tier: images stage skipped (Wan generates clips directly)")
            return _stage_result("images", out, skipped=True, extra={"reason": "wan tier"})
        from .video_flux import generate_images

        progress("images", 0.50, "Generating FLUX scene images...")
        images = generate_images(scenes, config, out, script=script)
        release_gpu_memory()
        return _stage_result("images", out, extra={"images": [str(p) for p in images]})

    if stage == "clips":
        if not voice_path.exists():
            raise FileNotFoundError(f"Missing {voice_path} — run voice stage first")
        tier = config.video.tier.lower()
        progress("clips", 0.60, f"Generating scene clips ({tier})...")
        if tier == "wan":
            try:
                clip_paths = _run_wan_subprocess(out, config)
            except Exception as exc:
                logger.warning("Wan2.2 failed (%s), falling back to FLUX slideshow", exc)
                clip_paths = _run_flux_subprocess(out, config)
        else:
            from .video_flux import generate_clips_from_images

            if not any((out / "images").glob("scene_*.png")):
                raise FileNotFoundError("No scene images — run images stage first")
            clip_paths = generate_clips_from_images(scenes, config, out)
        release_gpu_memory()
        return _stage_result("clips", out, extra={"clips": [str(p) for p in clip_paths]})

    if stage == "video":
        progress("video", 0.72, "Concatenating scene clips...")
        video_raw = concat_video_stage(out)
        return _stage_result("video", out, extra={"video_raw": str(video_raw)})

    if stage == "subtitles":
        if is_hindi_language(str(script.get("language", config.language))):
            logger.info("Hindi language: subtitles stage skipped")
            return _stage_result("subtitles", out, skipped=True, extra={"reason": "hindi"})
        if not config.subtitles:
            logger.info("Subtitles disabled in config")
            return _stage_result("subtitles", out, skipped=True, extra={"reason": "disabled"})
        progress("subtitles", 0.78, "Adding subtitles to video...")
        from .assembler import concat_clips
        from .subtitles import write_srt

        clips = list_scene_clips(out)
        subtitled_clips = subtitles_on_clips_stage(clips, scenes, config, out)
        subtitled_video = out / "video_subtitled.mp4"
        concat_clips(subtitled_clips, subtitled_video)
        write_srt(scenes, out / "subtitles.srt")
        return _stage_result(
            "subtitles",
            out,
            extra={
                "video_subtitled": str(subtitled_video),
                "clips_subtitled": [str(p) for p in subtitled_clips],
            },
        )

    if stage == "audio_mix":
        if not voice_path.exists() or not music_path.exists():
            raise FileNotFoundError("Missing voice.wav or music.wav — run voice and music stages first")
        progress("audio_mix", 0.85, "Mixing voice and music...")
        audio_mixed = audio_mix_stage(voice_path, music_path, out, config.music.volume)
        return _stage_result("audio_mix", out, extra={"audio_mixed": str(audio_mixed)})

    if stage == "final":
        if not (out / "audio_mixed.wav").exists():
            raise FileNotFoundError("Missing audio_mixed.wav — run audio_mix stage first")
        video_path = out / "video_subtitled.mp4"
        if not video_path.is_file():
            video_path = out / "video_raw.mp4"
        if not video_path.is_file():
            raise FileNotFoundError("Missing video_raw.mp4 — run video stage first")
        progress("final", 0.95, "Muxing final video...")
        voice_dur = _probe_duration(voice_path)
        final_path = final_mux_stage(video_path, out / "audio_mixed.wav", out, voice_duration=voice_dur)
        if config.subtitles and not is_hindi_language(str(script.get("language", config.language))):
            subtitled = out / "final_subtitled.mp4"
            import subprocess

            subprocess.run(
                ["ffmpeg", "-y", "-i", str(final_path), "-c", "copy", str(subtitled)],
                check=True,
                capture_output=True,
            )
            final_path = subtitled
        progress("done", 1.0, f"Complete: {final_path}")
        return _stage_result("final", out, extra={"final_video": str(final_path)})

    raise ValueError(f"Unhandled stage: {stage}")


def _probe_duration(path: Path) -> float:
    import subprocess

    return float(
        subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            text=True,
        ).strip()
    )


def _align_scene_durations(scenes: list[dict], voice_path: Path) -> list[dict]:
    """Scale per-scene duration_sec to match actual voice length."""
    total_voice = _probe_duration(voice_path)
    weights = [max(1, len(s.get("narration_segment", ""))) for s in scenes]
    weight_sum = sum(weights) or len(scenes)
    aligned = []
    for scene, weight in zip(scenes, weights):
        s = dict(scene)
        s["duration_sec"] = round(total_voice * weight / weight_sum, 1)
        aligned.append(s)
    return aligned


def _default_progress(stage: str, pct: float, msg: str) -> None:
    logger.info("[%s %.0f%%] %s", stage, pct * 100, msg)


def run_pipeline(
    config: PipelineConfig,
    output_dir: Path | None = None,
    on_progress: ProgressCallback | None = None,
    skip_llm: bool = False,
    script_override: dict | None = None,
    video_only: bool = False,
) -> dict:
    """Run full pipeline. Returns metadata dict with paths."""
    progress = on_progress or _default_progress
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out = output_dir or (OUTPUT_DIR / run_id)
    out.mkdir(parents=True, exist_ok=True)

    voice_path = out / "voice.wav"
    music_path = out / "music.wav"

    if video_only:
        if not (out / "script.json").exists():
            raise FileNotFoundError(f"No script.json in {out}")
        if not voice_path.exists() or not music_path.exists():
            raise FileNotFoundError(f"Missing voice.wav or music.wav in {out}")
        script = json.loads((out / "script.json").read_text())
        scenes = script["scenes"]
        if not config.theme:
            config.theme = script.get("content_type") or script.get("theme", "story")
    else:
        # Stage 1: Script
        progress("script", 0.05, "Generating narration script...")
        if script_override:
            script = script_override
        elif skip_llm and (out / "script.json").exists():
            script = json.loads((out / "script.json").read_text())
        else:
            theme = require_theme(config)
            logger.info("Content type: %s", theme)
            progress("script", 0.05, f"Content type: {theme} — LLM inventing story...")
            (out / "theme.txt").write_text(config.theme, encoding="utf-8")
            script = generate_script(config)
        (out / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False))
        if not script_override and not skip_llm:
            register_story(
                script.get("narration") or script.get("script", ""),
                script.get("title", ""),
                script.get("content_type") or script.get("theme", config.theme),
                script.get("language", config.language),
                out.name,
            )

        narration = script["narration"]
        scenes = script["scenes"]
        music_prompt = script.get("music_prompt", config.music.prompt)

        # Stage 2: Voice (Divya profile from config)
        from .tts import generate_voice, unload_model as unload_tts

        progress("voice", 0.20, "Generating Divya voice...")
        segments = [s.get("narration_segment", "") for s in scenes]
        generate_voice(narration, config, voice_path, scene_segments=segments if len(segments) > 1 else None)
        unload_tts()

        # Stage 3: Music
        from .music import generate_music, unload_model as unload_music

        progress("music", 0.40, "Generating background music...")
        import subprocess

        voice_dur = float(
            subprocess.check_output(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(voice_path)],
                text=True,
            ).strip()
        )
        generate_music(music_prompt, voice_dur + 2, config, music_path)
        unload_music()
        release_gpu_memory()

    apply_subtitle_policy(config, script)
    if not config.subtitles and is_hindi_language(
        str(script.get("language", config.language))
    ):
        logger.info("Hindi language: subtitles disabled")

    scenes = _align_scene_durations(scenes, voice_path)
    release_gpu_memory()

    # Stage 4: Video
    tier = config.video.tier.lower()
    progress("video", 0.55, f"Generating story images + clips ({tier})...")
    if tier == "wan":
        try:
            clips = _run_wan_subprocess(out, config)
        except Exception as exc:
            logger.warning("Wan2.2 failed (%s), falling back to FLUX slideshow", exc)
            clips = _run_flux_subprocess(out, config)
    else:
        clips = _run_flux_subprocess(out, config)

    # Stage 5: Assembly
    progress("assembly", 0.85, "Assembling final video...")
    final = assemble_pipeline_output(clips, voice_path, music_path, scenes, config, out)

    progress("done", 1.0, f"Complete: {final}")
    return {
        "run_id": out.name if output_dir else run_id,
        "output_dir": str(out),
        "final_video": str(final),
        "theme": config.theme,
        "script": script,
        "voice": str(voice_path),
        "music": str(music_path),
        "clips": [str(c) for c in clips],
    }


def main() -> None:
    import argparse
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Narration video pipeline — pass --theme, --lang, --duration, --tier"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (default: default.yaml)")
    parser.add_argument(
        "--theme",
        type=str,
        required=False,
        help="Content type (story, joke, …). Omit or use 'auto' for random pick.",
    )
    parser.add_argument("--lang", type=str, default=None, choices=["en", "hi"], help="Narration language")
    parser.add_argument("--duration", type=int, default=None, help="Video duration in seconds (10-120)")
    parser.add_argument("--tier", type=str, default=None, choices=["flux", "wan"], help="Video tier")
    parser.add_argument("--themes-csv", type=str, default=None, help="Comma-separated themes for random pick")
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument("--video-only", action="store_true", help="Regenerate images/video from existing run folder")
    parser.add_argument("--from-run", type=str, default=None, help="Existing output folder for --video-only or --stage")
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=list(PIPELINE_STAGES),
        help="Run a single pipeline stage (for n8n step workflow)",
    )
    args = parser.parse_args()

    overrides = {}
    if args.lang:
        overrides["language"] = args.lang
    if args.duration:
        overrides["duration_sec"] = args.duration
    if args.theme:
        overrides["theme"] = args.theme
    if args.themes_csv:
        pool = [t.strip().lower() for t in args.themes_csv.split(",") if t.strip()]
        if pool:
            overrides["themes"] = pool
    if args.no_subtitles:
        overrides["subtitles"] = False

    config = load_config(args.config, **overrides)
    if args.tier:
        config.video.tier = args.tier

    if config.video.tier.lower() == "wan" and "wan-venv" in sys.prefix:
        logger.error(
            "wan-venv breaks parler-tts. Run the full pipeline from flux-venv with --tier wan:\n"
            "  source flux-venv/bin/activate\n"
            "  python -m src.pipeline --lang hi --duration 10 --tier wan --theme bedtime"
        )
        raise SystemExit(1)

    if args.video_only:
        if not args.from_run:
            parser.error("--video-only requires --from-run PATH")
        result = run_pipeline(config, output_dir=Path(args.from_run), video_only=True)
    elif args.stage:
        run_dir = Path(args.from_run) if args.from_run else None
        run_id = run_dir.name if run_dir else None
        if args.stage != "script" and not run_dir:
            parser.error(f"--stage {args.stage} requires --from-run PATH")
        if args.stage == "script":
            prev = config.theme.strip().lower()
            config.theme = resolve_theme(config.theme or None, config)
            if not prev or prev == "auto":
                logger.info("Random theme selected: %s", config.theme)
        result = run_stage(
            args.stage,
            config,
            output_dir=run_dir,
            run_id=run_id,
        )
    else:
        prev = config.theme.strip().lower()
        config.theme = resolve_theme(config.theme or None, config)
        if not prev or prev == "auto":
            logger.info("Random theme selected: %s", config.theme)
        result = run_pipeline(config)
    payload = json.dumps(result, indent=2)
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()
    # PyTorch MPS can SIGSEGV during interpreter shutdown after TTS/music; skip teardown for n8n.
    os._exit(0)


if __name__ == "__main__":
    main()
