"""Isolated FLUX slideshow generation — fresh process after TTS/music to avoid OOM."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .log_format import configure_logging
from .run_io import prepare_worker_job
from .video_flux import generate_slideshow

logger = logging.getLogger(__name__)


def run_flux_job(output_dir: Path, config_path: str | None = None) -> list[Path]:
    out, script, config, scenes = prepare_worker_job(output_dir, "flux", config_path)
    return generate_slideshow(scenes, config, out, script=script)


def main() -> None:
    configure_logging(level=logging.INFO)
    parser = argparse.ArgumentParser(description="FLUX slideshow generation (isolated subprocess)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    clips = run_flux_job(Path(args.output_dir), args.config)
    print(json.dumps({"clips": [str(c) for c in clips]}, indent=2))


if __name__ == "__main__":
    main()
