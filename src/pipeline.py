"""End-to-end narration video pipeline orchestrator."""

from __future__ import annotations

import json
import logging
import os
import sys
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
from .clips import generate_clips
from .config import (
    PipelineConfig,
    apply_subtitle_policy,
    is_hindi_language,
    load_config,
    require_theme,
    resolve_theme,
)
from .media import probe_duration
from .memory import release_gpu_memory
from .run_io import (
    OUTPUT_DIR,
    load_script,
    narration_segments,
    persist_aligned_scenes,
    resolve_run_dir,
    save_script,
    sync_config_from_script,
)
from .log_format import configure_logging
from .script_generator import generate_script
from .story_registry import register_story
from .theme_profiles import resolve_voice_from_script
from .theme_rotation import commit_serial_theme

logger = logging.getLogger(__name__)

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


def _script_meta(script: dict) -> dict[str, str]:
    return {
        "title": str(script.get("title", "")),
        "youtube_title": str(script.get("youtube_title", "")),
        "youtube_description": str(script.get("youtube_description", "")),
        "youtube_comment": str(script.get("youtube_comment", "")),
        "hook_text": str(script.get("hook_text", "")),
        "language": str(script.get("language", "")),
        "content_type": str(script.get("content_type") or script.get("theme", "")),
        "title_style": str(script.get("title_style", "")),
    }


def _stage_result(
    stage: str,
    out: Path,
    *,
    skipped: bool = False,
    extra: dict | None = None,
    slim: bool = False,
) -> dict:
    payload: dict = {
        "stage": stage,
        "run_id": out.name,
        "output_dir": str(out),
        "skipped": skipped,
    }
    if (out / "script.json").is_file():
        script = load_script(out)
        if slim:
            payload["script_meta"] = _script_meta(script)
        else:
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
            payload[name.replace(".", "_").replace("-", "_")] = str(path)
    final = out / "final_subtitled.mp4"
    if not final.is_file():
        final = out / "final.mp4"
    if final.is_file():
        payload["final_video"] = str(final)
    if extra:
        payload.update(extra)
    return payload


def _n8n_slim() -> bool:
    return os.environ.get("N8N_STEP") == "1"


def _finish_result(result: dict) -> dict:
    """Persist full result to disk; return a compact payload for n8n stdout."""
    out_dir = Path(str(result.get("output_dir", "")))
    if out_dir.is_dir():
        (out_dir / ".step_result.json").write_text(
            json.dumps(result, ensure_ascii=False),
            encoding="utf-8",
        )

    if not _n8n_slim():
        return result

    slim = dict(result)
    script = slim.pop("script", None)
    if isinstance(script, dict):
        slim["script_meta"] = _script_meta(script)
        slim.setdefault("lang", script.get("language", ""))
        slim.setdefault("theme", script.get("content_type") or script.get("theme", ""))
    return slim


def _emit_pipeline_result(result: dict) -> None:
    payload = _finish_result(result)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    os._exit(0)


def _default_progress(stage: str, pct: float, msg: str) -> None:
    logger.info("[%s %.0f%%] %s", stage, pct * 100, msg)


def _register_script(script: dict, config: PipelineConfig, run_id: str) -> None:
    register_story(
        script.get("narration") or script.get("script", ""),
        script.get("title", ""),
        script.get("content_type") or script.get("theme", config.theme),
        script.get("language", config.language),
        run_id,
    )


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
            out = resolve_run_dir(run_id, output_dir)
        out.mkdir(parents=True, exist_ok=True)

        progress("script", 0.05, "Generating narration script...")
        theme = require_theme(config)
        logger.info("Content type: %s", theme)
        (out / "theme.txt").write_text(config.theme, encoding="utf-8")
        script = generate_script(config)
        save_script(out, script)
        commit_serial_theme(config.themes, expected_theme=config.theme)
        _register_script(script, config, out.name)
        apply_subtitle_policy(config, script)
        return _stage_result("script", out)

    out = resolve_run_dir(run_id, output_dir)
    script = load_script(out)
    sync_config_from_script(config, script)
    scenes = script["scenes"]
    voice_path = out / "voice.wav"
    music_path = out / "music.wav"
    narration = script["narration"]

    if stage == "voice":
        from .tts import generate_voice, unload_model as unload_tts

        progress("voice", 0.20, "Generating voice...")
        voice_description = resolve_voice_from_script(script, config.voice.description)
        generate_voice(
            narration,
            config,
            voice_path,
            scene_segments=narration_segments(scenes),
            description=voice_description,
        )
        unload_tts()
        release_gpu_memory()
        return _stage_result("voice", out)

    if stage == "music":
        if not voice_path.exists():
            raise FileNotFoundError(f"Missing {voice_path} — run voice stage first")
        from .music import generate_music, unload_model as unload_music

        progress("music", 0.40, "Generating background music...")
        music_prompt = script.get("music_prompt", config.music.prompt)
        generate_music(music_prompt, probe_duration(voice_path) + 2, config, music_path)
        unload_music()
        release_gpu_memory()
        return _stage_result("music", out)

    scenes = persist_aligned_scenes(out, script, voice_path)

    if stage == "images":
        if not voice_path.exists():
            raise FileNotFoundError(f"Missing {voice_path} — run voice stage first")
        if config.video.tier.lower() == "wan":
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
        clip_paths = generate_clips(out, config, scenes, flux_subprocess=False)
        release_gpu_memory()
        return _stage_result("clips", out, extra={"clips": [str(p) for p in clip_paths]})

    if stage == "video":
        progress("video", 0.72, "Concatenating scene clips...")
        video_raw = concat_video_stage(out)
        return _stage_result("video", out, extra={"video_raw": str(video_raw)})

    if stage == "subtitles":
        # config.subtitles is already policy-applied (Hindi and --no-subtitles turn it off),
        # but the first-clip hook overlay is burned regardless of captions.
        captions = bool(config.subtitles)
        hook_text = str(script.get("hook_text") or "").strip()
        want_hook = bool(hook_text and config.shorts.hook_overlay)
        if not captions and not want_hook:
            logger.info("Subtitles stage skipped: captions disabled and no hook to burn")
            return _stage_result("subtitles", out, skipped=True, extra={"reason": "disabled"})
        progress(
            "subtitles",
            0.78,
            "Adding subtitles to video..." if captions else "Burning hook overlay...",
        )
        from .assembler import concat_clips
        from .subtitles import write_srt

        clips = list_scene_clips(out)
        subtitled_clips = subtitles_on_clips_stage(
            clips, scenes, config, out,
            hook_text=hook_text or None, captions=captions,
        )
        subtitled_video = out / "video_subtitled.mp4"
        concat_clips(subtitled_clips, subtitled_video)
        if captions:
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
        final_path = final_mux_stage(
            video_path,
            out / "audio_mixed.wav",
            out,
            voice_duration=probe_duration(voice_path),
        )
        if config.subtitles and not is_hindi_language(str(script.get("language", config.language))):
            subtitled = out / "final_subtitled.mp4"
            final_path.replace(subtitled)
            final_path = subtitled
        if config.shorts.loop_transition:
            from .assembler import apply_loop_transition

            try:
                apply_loop_transition(final_path, config.shorts.loop_transition_sec)
            except Exception as exc:
                logger.warning("Loop transition skipped: %s", exc)
        progress("done", 1.0, f"Complete: {final_path}")
        return _stage_result("final", out, extra={"final_video": str(final_path)})

    raise ValueError(f"Unhandled stage: {stage}")


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
        script = load_script(out)
        scenes = script["scenes"]
        if not config.theme:
            config.theme = script.get("content_type") or script.get("theme", "story")
    else:
        progress("script", 0.05, "Generating narration script...")
        if script_override:
            script = script_override
        elif skip_llm and (out / "script.json").exists():
            script = load_script(out)
        else:
            theme = require_theme(config)
            logger.info("Content type: %s", theme)
            progress("script", 0.05, f"Content type: {theme} — LLM inventing story...")
            (out / "theme.txt").write_text(config.theme, encoding="utf-8")
            script = generate_script(config)
        save_script(out, script)
        if not script_override and not skip_llm:
            commit_serial_theme(config.themes, expected_theme=config.theme)
            _register_script(script, config, out.name)

        narration = script["narration"]
        scenes = script["scenes"]
        music_prompt = script.get("music_prompt", config.music.prompt)
        voice_description = resolve_voice_from_script(script, config.voice.description)

        from .tts import generate_voice, unload_model as unload_tts

        progress("voice", 0.20, "Generating voice (Divya/Rani)...")
        generate_voice(
            narration,
            config,
            voice_path,
            scene_segments=narration_segments(scenes),
            description=voice_description,
        )
        unload_tts()

        from .music import generate_music, unload_model as unload_music

        progress("music", 0.40, "Generating background music...")
        generate_music(music_prompt, probe_duration(voice_path) + 2, config, music_path)
        unload_music()
        release_gpu_memory()

    apply_subtitle_policy(config, script)
    if not config.subtitles and is_hindi_language(str(script.get("language", config.language))):
        logger.info("Hindi language: subtitles disabled")

    scenes = persist_aligned_scenes(out, script, voice_path)
    release_gpu_memory()

    tier = config.video.tier.lower()
    progress("video", 0.55, f"Generating story images + clips ({tier})...")
    clips = generate_clips(out, config, scenes, flux_subprocess=True)

    progress("assembly", 0.85, "Assembling final video...")
    final = assemble_pipeline_output(
        clips, voice_path, music_path, scenes, config, out,
        hook_text=script.get("hook_text"),
    )

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

    configure_logging(level=logging.INFO)

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
                logger.info("Serial theme selected: %s", config.theme)
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
            logger.info("Serial theme selected: %s", config.theme)
        result = run_pipeline(config)
    _emit_pipeline_result(result)


if __name__ == "__main__":
    main()
