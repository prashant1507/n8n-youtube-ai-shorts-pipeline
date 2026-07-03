"""ffmpeg video/audio assembly."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .config import PipelineConfig
from .media import AAC_BITRATE, X264_QUALITY, probe_duration
from .subtitles import (
    chunk_text,
    render_caption_overlay,
    render_hook_overlay,
    save_overlay_png,
    save_subtitle_overlay_png,
    write_srt,
)

logger = logging.getLogger(__name__)


def _run(cmd: list[str], cwd: str | None = None) -> None:
    logger.debug("ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def concat_clips(clips: list[Path], output_path: Path, crossfade_sec: float = 0.5) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(clips) == 1:
        _run(["ffmpeg", "-y", "-i", str(clips[0]), "-c", "copy", str(output_path)])
        return output_path

    # Simple concat via concat demuxer (no crossfade for reliability)
    list_file = output_path.parent / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{c.resolve()}'" for c in clips))
    _run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264", *X264_QUALITY, "-pix_fmt", "yuv420p",
        str(output_path),
    ])
    return output_path


def mix_audio(
    voice_path: Path,
    music_path: Path,
    output_path: Path,
    music_volume: float = 0.20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    voice_dur = probe_duration(voice_path)
    _run([
        "ffmpeg", "-y",
        "-i", str(voice_path),
        "-i", str(music_path),
        "-filter_complex",
        f"[1:a]volume={music_volume},afade=t=in:st=0:d=2,afade=t=out:st={max(0, voice_dur - 3)}:d=3[music];"
        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "[aout]",
        str(output_path),
    ])
    return output_path


def assemble_final(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    voice_duration: float | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dur = voice_duration or probe_duration(audio_path)
    _run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-t", str(dur),
        "-c:v", "libx264", *X264_QUALITY,
        "-c:a", "aac", *AAC_BITRATE,
        "-shortest",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ])
    return output_path


def _has_subtitles_filter() -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True,
            text=True,
            check=True,
        )
        return " subtitles " in out.stdout or "\nsubtitles " in out.stdout
    except Exception:
        return False


def embed_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
) -> Path:
    """Embed SRT as a soft subtitle track (works without libass)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y",
        "-i", str(video_path.resolve()),
        "-i", str(srt_path.resolve()),
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        str(output_path.resolve()),
    ])
    return output_path


def _subtitle_overlay_y() -> str:
    """ffmpeg overlay y: vertical center at 75% of frame (middle of lower half)."""
    return "H*3/4-h/2"


def overlay_subtitle_on_clip(
    clip_path: Path,
    text: str,
    output_path: Path,
    video_width: int,
    video_height: int,
    font_size: int = 28,
) -> Path:
    """Burn subtitle text on a clip using PIL + ffmpeg overlay."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub_png = output_path.parent / f".{output_path.stem}_sub.png"
    save_subtitle_overlay_png(text, video_width, sub_png, font_size=font_size)

    _run([
        "ffmpeg", "-y",
        "-i", str(clip_path.resolve()),
        "-i", str(sub_png.resolve()),
        "-filter_complex", f"[0:v][1:v]overlay=0:{_subtitle_overlay_y()}",
        "-c:v", "libx264", *X264_QUALITY,
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        str(output_path.resolve()),
    ])
    sub_png.unlink(missing_ok=True)
    return output_path


def _overlay_timed_pngs(
    clip_path: Path,
    overlays: list[dict],
    output_path: Path,
) -> Path:
    """Overlay several full-width PNGs on a clip, each time-gated to [start, end]."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not overlays:
        _run(["ffmpeg", "-y", "-i", str(clip_path.resolve()), "-c", "copy", str(output_path.resolve())])
        return output_path

    inputs: list[str] = ["-i", str(clip_path.resolve())]
    for ov in overlays:
        inputs += ["-i", str(Path(ov["png"]).resolve())]

    parts: list[str] = []
    label = "0:v"
    for idx, ov in enumerate(overlays, start=1):
        out_label = f"v{idx}"
        parts.append(
            f"[{label}][{idx}:v]overlay=0:{ov['y']}:"
            f"enable='between(t,{ov['start']:.3f},{ov['end']:.3f})'[{out_label}]"
        )
        label = out_label
    filt = ";".join(parts)

    _run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filt,
        "-map", f"[{label}]",
        "-map", "0:a?",
        "-c:v", "libx264", *X264_QUALITY,
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        str(output_path.resolve()),
    ])
    return output_path


def burn_dynamic_on_clip(
    clip: Path,
    scene: dict,
    output: Path,
    config: PipelineConfig,
    *,
    hook_text: str | None = None,
    is_first: bool = False,
    captions: bool = True,
) -> Path:
    """Burn punchy chunked captions (and, on the first clip, the on-screen hook).

    captions=False burns only the hook — used for Hindi, where on-screen
    captions are disabled but the retention hook should still appear.
    """
    shorts = config.shorts
    width, height = config.width, config.height
    scale = height / 1920.0
    dur = probe_duration(clip)
    overlays: list[dict] = []
    tmp_pngs: list[Path] = []

    text = scene.get("narration_segment", "").strip()
    chunks = chunk_text(text, shorts.caption_chunk_words) if (text and captions) else []
    if chunks:
        cap_font = max(20, int(round(28 * shorts.caption_font_scale * scale)))
        cap_y = f"H*{shorts.caption_y_ratio:.3f}-h/2"
        weights = [max(1, len(c)) for c in chunks]
        wsum = sum(weights)
        t = 0.0
        for j, (chunk, weight) in enumerate(zip(chunks, weights)):
            seg = dur * weight / wsum
            png = output.parent / f".{output.stem}_cap{j}.png"
            save_overlay_png(render_caption_overlay(chunk, width, font_size=cap_font), png)
            tmp_pngs.append(png)
            end = dur if j == len(chunks) - 1 else t + seg
            overlays.append({"png": png, "y": cap_y, "start": round(t, 3), "end": round(end, 3)})
            t += seg

    if is_first and hook_text and shorts.hook_overlay:
        hook_font = max(34, int(round(58 * scale)))
        png = output.parent / f".{output.stem}_hook.png"
        save_overlay_png(render_hook_overlay(hook_text, width, font_size=hook_font), png)
        tmp_pngs.append(png)
        overlays.append({
            "png": png,
            "y": "H*0.16",
            "start": 0.0,
            "end": round(min(shorts.hook_overlay_sec, dur), 3),
        })

    if not overlays:
        _run(["ffmpeg", "-y", "-i", str(clip.resolve()), "-c", "copy", str(output.resolve())])
    else:
        _overlay_timed_pngs(clip, overlays, output)

    for png in tmp_pngs:
        png.unlink(missing_ok=True)
    return output


def burn_subtitles_on_clips(
    clips: list[Path],
    scenes: list[dict],
    output_dir: Path,
    video_width: int,
    video_height: int,
    font_size: int = 28,
    config: PipelineConfig | None = None,
    hook_text: str | None = None,
    captions: bool = True,
) -> list[Path]:
    """Add captions to each clip. Uses dynamic chunked captions + hook when config is given."""
    subtitled_dir = output_dir / "clips_subtitled"
    subtitled_dir.mkdir(parents=True, exist_ok=True)
    out_clips: list[Path] = []

    use_dynamic = config is not None

    for i, clip in enumerate(clips):
        scene = scenes[i] if i < len(scenes) else {}
        text = scene.get("narration_segment", "").strip()
        out = subtitled_dir / clip.name
        if use_dynamic:
            burn_dynamic_on_clip(
                clip, scene, out, config,
                hook_text=hook_text, is_first=(i == 0), captions=captions,
            )
        elif text:
            overlay_subtitle_on_clip(clip, text, out, video_width, video_height, font_size)
        else:
            _run(["ffmpeg", "-y", "-i", str(clip), "-c", "copy", str(out)])
        out_clips.append(out)

    return out_clips


def apply_loop_transition(video_path: Path, fade_sec: float = 0.4) -> Path:
    """Cross-dissolve the ending back into the opening so the Short replays seamlessly.

    Edits in place (same filename), preserving audio and total duration.
    """
    dur = probe_duration(video_path)
    fade = max(0.15, min(fade_sec, dur * 0.3))
    offset = max(0.0, dur - fade)
    open_clip = video_path.parent / f".{video_path.stem}_open.mp4"
    looped = video_path.parent / f".{video_path.stem}_looped.mp4"

    _run([
        "ffmpeg", "-y",
        "-i", str(video_path.resolve()),
        "-t", f"{fade:.3f}",
        "-an", "-c:v", "libx264", *X264_QUALITY, "-pix_fmt", "yuv420p",
        str(open_clip.resolve()),
    ])
    _run([
        "ffmpeg", "-y",
        "-i", str(video_path.resolve()),
        "-i", str(open_clip.resolve()),
        "-filter_complex",
        f"[0:v][1:v]xfade=transition=fade:duration={fade:.3f}:offset={offset:.3f},format=yuv420p[v]",
        "-map", "[v]",
        "-map", "0:a?",
        "-c:a", "copy",
        "-c:v", "libx264", *X264_QUALITY,
        "-pix_fmt", "yuv420p",
        str(looped.resolve()),
    ])
    open_clip.unlink(missing_ok=True)
    looped.replace(video_path)
    return video_path


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    scenes: list[dict] | None = None,
    config: PipelineConfig | None = None,
) -> Path:
    """Burn bottom subtitles; uses per-scene overlay when libass is unavailable."""
    if _has_subtitles_filter():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        margin_v = 30
        if config and config.height:
            # Bottom-anchored ASS text; center ~75% down the frame (middle of lower half)
            margin_v = max(30, int(config.height * 0.25 - 28))
        style = (
            "FontSize=22,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,"
            f"BorderStyle=3,Alignment=2,MarginV={margin_v}"
        )
        _run([
            "ffmpeg", "-y",
            "-i", str(video_path.resolve()),
            "-vf", f"subtitles={srt_path.name}:force_style='{style}'",
            "-c:v", "libx264", *X264_QUALITY,
            "-c:a", "copy",
            str(output_path.resolve()),
        ], cwd=str(srt_path.parent))
        return output_path

    # Re-burn via clip overlays when we have scene data
    if scenes and config:
        logger.info("Burning bottom subtitles via PIL overlay")
        clips_dir = video_path.parent / "clips"
        clips = sorted(clips_dir.glob("scene_*.mp4"))
        if clips:
            sub_clips = burn_subtitles_on_clips(
                clips, scenes, video_path.parent,
                config.width, config.height,
            )
            temp_video = video_path.parent / "video_subtitled_raw.mp4"
            concat_clips(sub_clips, temp_video)
            _run([
                "ffmpeg", "-y",
                "-i", str(temp_video.resolve()),
                "-c:v", "libx264", *X264_QUALITY,
                "-pix_fmt", "yuv420p",
                str(output_path.resolve()),
            ])
            return output_path

    logger.warning("Cannot burn subtitles — falling back to soft embed")
    return embed_subtitles(video_path, srt_path, output_path)


def list_scene_clips(output_dir: Path, subtitled: bool = False) -> list[Path]:
    clips_dir = output_dir / ("clips_subtitled" if subtitled else "clips")
    clips = sorted(clips_dir.glob("scene_*.mp4"))
    if not clips:
        clips = sorted((output_dir / "clips").glob("scene_*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No scene clips in {output_dir / 'clips'}")
    return clips


def concat_video_stage(
    output_dir: Path,
    prefer_subtitled_clips: bool = False,
) -> Path:
    """Concat scene clips into video_raw.mp4."""
    clips = list_scene_clips(output_dir, subtitled=prefer_subtitled_clips)
    video_raw = output_dir / "video_raw.mp4"
    concat_clips(clips, video_raw)
    return video_raw


def subtitles_on_clips_stage(
    clips: list[Path],
    scenes: list[dict],
    config: PipelineConfig,
    output_dir: Path,
    hook_text: str | None = None,
    captions: bool = True,
) -> list[Path]:
    """Burn captions on each clip (dynamic chunked captions + first-clip hook)."""
    return burn_subtitles_on_clips(
        clips, scenes, output_dir, config.width, config.height,
        config=config, hook_text=hook_text, captions=captions,
    )


def subtitles_on_video_stage(
    video_path: Path,
    scenes: list[dict],
    output_dir: Path,
    config: PipelineConfig,
) -> Path:
    """Burn or embed subtitles on concatenated video."""
    srt_path = output_dir / "subtitles.srt"
    write_srt(scenes, srt_path)
    subtitled = output_dir / "video_subtitled.mp4"
    burn_subtitles(video_path, srt_path, subtitled, scenes=scenes, config=config)
    return subtitled


def audio_mix_stage(
    voice_path: Path,
    music_path: Path,
    output_dir: Path,
    music_volume: float,
) -> Path:
    audio_mixed = output_dir / "audio_mixed.wav"
    mix_audio(voice_path, music_path, audio_mixed, music_volume)
    return audio_mixed


def final_mux_stage(
    video_path: Path,
    audio_path: Path,
    output_dir: Path,
    voice_duration: float | None = None,
) -> Path:
    final_path = output_dir / "final.mp4"
    assemble_final(video_path, audio_path, final_path, voice_duration=voice_duration)
    return final_path


def assemble_pipeline_output(
    clips: list[Path],
    voice_path: Path,
    music_path: Path,
    scenes: list[dict],
    config: PipelineConfig,
    output_dir: Path,
    hook_text: str | None = None,
) -> Path:
    video_raw = output_dir / "video_raw.mp4"
    audio_mixed = output_dir / "audio_mixed.wav"
    final_path = output_dir / "final.mp4"

    want_captions = bool(config.subtitles and scenes)
    # Hindi disables captions, but the first-clip hook should still be burned
    want_hook = bool(scenes and hook_text and config.shorts.hook_overlay)
    if want_captions or want_hook:
        clips = burn_subtitles_on_clips(
            clips, scenes, output_dir, config.width, config.height,
            config=config, hook_text=hook_text, captions=want_captions,
        )

    video_raw = concat_video_stage(output_dir, prefer_subtitled_clips=want_captions or want_hook)
    audio_mixed = audio_mix_stage(voice_path, music_path, output_dir, config.music.volume)
    final_path = final_mux_stage(video_raw, audio_mixed, output_dir)

    if want_captions:
        srt_path = output_dir / "subtitles.srt"
        write_srt(scenes, srt_path)
        subtitled = output_dir / "final_subtitled.mp4"
        final_path.replace(subtitled)
        final_path = subtitled

    if config.shorts.loop_transition:
        try:
            apply_loop_transition(final_path, config.shorts.loop_transition_sec)
        except Exception as exc:  # loop is a nicety; never fail the whole render for it
            logger.warning("Loop transition skipped: %s", exc)

    return final_path
