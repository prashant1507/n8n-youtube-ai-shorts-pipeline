"""Build SRT subtitles and render burn-in overlays."""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(scenes: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    t = 0.0
    for i, scene in enumerate(scenes, start=1):
        text = scene.get("narration_segment", "").strip()
        if not text:
            continue
        dur = float(scene.get("duration_sec", 5.0))
        start = _format_ts(t)
        end = _format_ts(t + dur)
        lines.extend([str(i), f"{start} --> {end}", text, ""])
        t += dur

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_subtitle_overlay(
    text: str,
    video_width: int,
    font_size: int = 28,
    margin_x: int = 40,
    bar_padding: int = 16,
) -> Image.Image:
    """Render a subtitle bar as RGBA image (full video width)."""
    font = _load_font(font_size)
    max_chars = max(24, int((video_width - margin_x * 2) / (font_size * 0.45)))
    wrapped = textwrap.fill(text.strip(), width=max_chars)

    # Measure text block
    probe = Image.new("RGBA", (video_width, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=6)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    bar_h = text_h + bar_padding * 2
    img = Image.new("RGBA", (video_width, bar_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Semi-transparent bar across bottom
    draw.rectangle([0, 0, video_width, bar_h], fill=(0, 0, 0, 170))

    text_x = (video_width - text_w) // 2
    text_y = bar_padding

    # Outline for readability
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
        draw.multiline_text(
            (text_x + dx, text_y + dy),
            wrapped,
            font=font,
            fill=(0, 0, 0, 255),
            spacing=6,
            align="center",
        )
    draw.multiline_text(
        (text_x, text_y),
        wrapped,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=6,
        align="center",
    )
    return img


def save_subtitle_overlay_png(
    text: str,
    video_width: int,
    output_path: Path,
    font_size: int = 28,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = render_subtitle_overlay(text, video_width, font_size=font_size)
    img.save(output_path)
    return output_path
