"""ffmpeg video/audio assembly."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .config import PipelineConfig
from .media import probe_duration
from .subtitles import save_subtitle_overlay_png, write_srt

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
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
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
        "-c:v", "libx264",
        "-c:a", "aac",
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
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        str(output_path.resolve()),
    ])
    sub_png.unlink(missing_ok=True)
    return output_path


def burn_subtitles_on_clips(
    clips: list[Path],
    scenes: list[dict],
    output_dir: Path,
    video_width: int,
    video_height: int,
    font_size: int = 28,
) -> list[Path]:
    """Add subtitles to each clip matching scene narration."""
    subtitled_dir = output_dir / "clips_subtitled"
    subtitled_dir.mkdir(parents=True, exist_ok=True)
    out_clips: list[Path] = []

    for i, clip in enumerate(clips):
        scene = scenes[i] if i < len(scenes) else {}
        text = scene.get("narration_segment", "").strip()
        out = subtitled_dir / clip.name
        if text:
            overlay_subtitle_on_clip(clip, text, out, video_width, video_height, font_size)
        else:
            _run(["ffmpeg", "-y", "-i", str(clip), "-c", "copy", str(out)])
        out_clips.append(out)

    return out_clips


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
                "-c:v", "libx264",
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
) -> list[Path]:
    """Burn bottom subtitles on each clip."""
    return burn_subtitles_on_clips(
        clips, scenes, output_dir, config.width, config.height,
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
) -> Path:
    video_raw = output_dir / "video_raw.mp4"
    audio_mixed = output_dir / "audio_mixed.wav"
    final_path = output_dir / "final.mp4"

    if config.subtitles and scenes:
        clips = burn_subtitles_on_clips(
            clips, scenes, output_dir, config.width, config.height,
        )

    video_raw = concat_video_stage(output_dir, prefer_subtitled_clips=bool(config.subtitles and scenes))
    audio_mixed = audio_mix_stage(voice_path, music_path, output_dir, config.music.volume)
    final_path = final_mux_stage(video_raw, audio_mixed, output_dir)

    if config.subtitles and scenes:
        srt_path = output_dir / "subtitles.srt"
        write_srt(scenes, srt_path)
        subtitled = output_dir / "final_subtitled.mp4"
        _run(["ffmpeg", "-y", "-i", str(final_path), "-c", "copy", str(subtitled)])
        return subtitled

    return final_path
