"""Run folder paths, script.json I/O, and config sync from a run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ROOT, PipelineConfig, apply_subtitle_policy, load_config
from .media import align_scene_durations

OUTPUT_DIR = ROOT / "output"


def resolve_run_dir(run_id: str | None, output_dir: Path | None) -> Path:
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


def load_script(run_dir: Path) -> dict[str, Any]:
    script_path = run_dir / "script.json"
    if not script_path.exists():
        raise FileNotFoundError(f"No script.json in {run_dir}")
    return json.loads(script_path.read_text(encoding="utf-8"))


def save_script(run_dir: Path, script: dict[str, Any]) -> None:
    (run_dir / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sync_config_from_script(config: PipelineConfig, script: dict[str, Any]) -> None:
    if script.get("content_type") or script.get("theme"):
        config.theme = str(script.get("content_type") or script.get("theme"))
    lang = str(script.get("language", config.language)).lower()
    if lang.startswith("hi"):
        config.language = "hi"
    elif lang.startswith("en"):
        config.language = "en"
    apply_subtitle_policy(config, script)


def config_overrides_from_script(script: dict[str, Any], *, tier: str | None = None) -> dict[str, Any]:
    """Build load_config() overrides from script.json (isolated workers)."""
    overrides: dict[str, Any] = {}
    if tier:
        overrides["video"] = {"tier": tier}
    if script.get("content_type") or script.get("theme"):
        overrides["theme"] = script.get("content_type") or script.get("theme")
    if script.get("language"):
        lang = str(script["language"]).lower()
        overrides["language"] = "hi" if lang.startswith("hi") else "en"
    return overrides


def prepare_worker_job(
    output_dir: Path,
    tier: str,
    config_path: str | None = None,
) -> tuple[Path, dict[str, Any], PipelineConfig, list[dict]]:
    """Load script + voice, apply overrides, return aligned scenes."""
    out = output_dir.resolve()
    voice_path = out / "voice.wav"
    script = load_script(out)
    if not voice_path.is_file():
        raise FileNotFoundError(f"No voice.wav in {out}")

    config = load_config(config_path, **config_overrides_from_script(script, tier=tier))
    scenes = align_scene_durations(script["scenes"], voice_path)
    return out, script, config, scenes


def persist_aligned_scenes(run_dir: Path, script: dict[str, Any], voice_path: Path) -> list[dict]:
    """Align scene durations to voice.wav and write script.json."""
    scenes = align_scene_durations(script["scenes"], voice_path)
    save_script(run_dir, {**script, "scenes": scenes})
    return scenes


def narration_segments(scenes: list[dict]) -> list[str] | None:
    segments = [s.get("narration_segment", "") for s in scenes]
    return segments if len(segments) > 1 else None
