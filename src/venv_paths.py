"""Resolve Python interpreters for isolated worker venvs."""

from __future__ import annotations

from pathlib import Path

from .config import ROOT, PipelineConfig


def resolve_python(
    *,
    configured: str,
    default_relative: str,
    purpose: str,
) -> Path:
    """Return a venv python from config override or project default."""
    if configured.strip():
        p = Path(configured).expanduser()
        if p.is_file():
            return p
        py = p / "bin" / "python"
        if py.is_file():
            return py
        raise RuntimeError(f"{purpose} python not found: {configured}")

    local = ROOT / default_relative
    if local.is_file():
        return local
    raise RuntimeError(f"{purpose} not found. Expected {local}")


def flux_python(config: PipelineConfig) -> Path:
    return resolve_python(
        configured=config.video.flux_python,
        default_relative="image-venv/bin/python",
        purpose="FLUX.2 Klein (create image-venv with requirements-image.txt)",
    )


def wan_python(_config: PipelineConfig) -> Path:
    return resolve_python(
        configured="",
        default_relative="wan-venv/bin/python",
        purpose="Wan (create wan-venv with mlx-gen)",
    )
