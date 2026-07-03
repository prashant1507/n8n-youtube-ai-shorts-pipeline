"""Build SRT subtitles and render burn-in overlays."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# (path, regular face index, bold face index) — Arial has no Devanagari glyphs
_DEVANAGARI_FONTS = (
    ("/System/Library/Fonts/Supplemental/ITFDevanagari.ttc", 0, 1),
    ("/System/Library/Fonts/Supplemental/DevanagariMT.ttc", 0, 1),
    ("/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc", 0, 0),
)


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


def _load_font(
    size: int,
    black: bool = False,
    text: str = "",
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if text and _DEVANAGARI_RE.search(text):
        for path, regular_idx, bold_idx in _DEVANAGARI_FONTS:
            try:
                return ImageFont.truetype(path, size, index=bold_idx if black else regular_idx)
            except OSError:
                continue

    black_paths = (
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/Library/Fonts/Arial Black.ttf",
    )
    regular_paths = (
        "/System/Library/Fonts/supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    )
    for path in (black_paths + regular_paths) if black else regular_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def chunk_text(text: str, max_words: int) -> list[str]:
    """Split narration into short caption chunks (punchy Shorts style).

    max_words <= 0 keeps the full segment as a single caption (legacy behavior).
    """
    words = text.split()
    if not words:
        return []
    if max_words <= 0 or len(words) <= max_words:
        return [text.strip()]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _draw_outlined(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    *,
    spacing: int,
    outline: int,
) -> None:
    x, y = xy
    for dx in range(-outline, outline + 1):
        for dy in range(-outline, outline + 1):
            if dx or dy:
                draw.multiline_text(
                    (x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255),
                    spacing=spacing, align="center",
                )
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="center")


def render_caption_overlay(
    text: str,
    video_width: int,
    font_size: int = 44,
    margin_x: int = 48,
    pad: int = 18,
    pill_alpha: int = 150,
) -> Image.Image:
    """Punchy caption: bold white text on a tight rounded pill (full-width strip)."""
    font = _load_font(font_size, black=False, text=text)
    max_chars = max(8, int((video_width - margin_x * 2) / (font_size * 0.58)))
    wrapped = textwrap.fill(text.strip(), width=max_chars)
    spacing = 8

    probe = Image.new("RGBA", (video_width, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(probe)
    bbox = d.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    bar_h = th + pad * 2
    img = Image.new("RGBA", (video_width, bar_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    box_w = min(video_width, tw + pad * 2)
    x0 = (video_width - box_w) // 2
    d.rounded_rectangle([x0, 0, x0 + box_w, bar_h], radius=min(pad, bar_h // 2), fill=(0, 0, 0, pill_alpha))

    tx = (video_width - tw) // 2 - bbox[0]
    ty = pad - bbox[1]
    _draw_outlined(d, (tx, ty), wrapped, font, (255, 255, 255, 255), spacing=spacing, outline=2)
    return img


def render_hook_overlay(
    text: str,
    video_width: int,
    font_size: int = 60,
    margin_x: int = 44,
    pad: int = 22,
    pill_alpha: int = 120,
) -> Image.Image:
    """Big attention-grabbing hook card: heavy accent-yellow text with a thick outline."""
    font = _load_font(font_size, black=True, text=text)
    max_chars = max(6, int((video_width - margin_x * 2) / (font_size * 0.62)))
    wrapped = textwrap.fill(text.strip(), width=max_chars)
    spacing = 10

    probe = Image.new("RGBA", (video_width, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(probe)
    bbox = d.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    bar_h = th + pad * 2
    img = Image.new("RGBA", (video_width, bar_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    box_w = min(video_width, tw + pad * 2)
    x0 = (video_width - box_w) // 2
    d.rounded_rectangle([x0, 0, x0 + box_w, bar_h], radius=min(pad, bar_h // 2), fill=(0, 0, 0, pill_alpha))

    tx = (video_width - tw) // 2 - bbox[0]
    ty = pad - bbox[1]
    _draw_outlined(d, (tx, ty), wrapped, font, (255, 214, 0, 255), spacing=spacing, outline=3)
    return img


def save_overlay_png(image: Image.Image, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def render_subtitle_overlay(
    text: str,
    video_width: int,
    font_size: int = 28,
    margin_x: int = 40,
    bar_padding: int = 16,
) -> Image.Image:
    """Render a subtitle bar as RGBA image (full video width)."""
    font = _load_font(font_size, text=text)
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
