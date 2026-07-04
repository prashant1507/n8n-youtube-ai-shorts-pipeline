"""Local HTTP API so n8n can trigger video generation without Execute Command."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .log_format import configure_logging
from .run_io import OUTPUT_DIR

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run-pipeline-for-n8n.sh"
STEP_SCRIPT = ROOT / "scripts" / "run-pipeline-step.sh"
HOST = os.environ.get("N8N_API_HOST", "0.0.0.0")
PORT = int(os.environ.get("N8N_API_PORT", "8765"))
PUBLIC_BASE = os.environ.get("N8N_API_PUBLIC_URL", f"http://host.docker.internal:{PORT}")
STEP_MAX_TRIES = max(1, int(os.environ.get("N8N_STEP_MAX_TRIES", "3")))
STEP_RETRY_WAIT_SEC = max(0, int(os.environ.get("N8N_STEP_RETRY_WAIT_SEC", "1800")))

logger = logging.getLogger(__name__)


def _public_base(handler: BaseHTTPRequestHandler | None = None) -> str:
    if handler is not None:
        host = (handler.headers.get("Host") or "").strip()
        if host:
            return f"http://{host}".rstrip("/")
    return PUBLIC_BASE.rstrip("/")


def _final_video_for_run(run_id: str) -> Path:
    safe_id = Path(run_id).name
    run_dir = (OUTPUT_DIR / safe_id).resolve()
    output_root = OUTPUT_DIR.resolve()
    if not str(run_dir).startswith(str(output_root)):
        raise ValueError("invalid run_id")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {safe_id}")
    for name in ("final_subtitled.mp4", "final.mp4"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No final video in {run_dir}")


def _enrich_result(result: dict[str, Any], handler: BaseHTTPRequestHandler | None = None) -> dict[str, Any]:
    run_id = str(result.get("run_id") or "").strip()
    if run_id:
        base = _public_base(handler)
        result["video_url"] = f"{base}/video/{run_id}"
    return result


def _extract_json_from(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the best complete JSON object in text (prefer payloads with run_id)."""
    best: dict[str, Any] | None = None
    best_raw: str | None = None
    start = 0
    while True:
        start = text.find("{", start)
        if start < 0:
            break
        raw = _extract_json_from(text, start)
        start += 1
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("run_id"):
            return data
        if best is None or len(raw) > len(best_raw or ""):
            best = data
            best_raw = raw
    return best if best and best.get("run_id") else None


def _recover_step_result(stdout: str, stderr: str) -> dict[str, Any] | None:
    """Load step result from disk when stdout JSON is missing or truncated."""
    import re

    haystack = f"{stdout}\n{stderr}"
    run_id = ""
    match = re.search(r'"run_id"\s*:\s*"([^"]+)"', haystack)
    if match:
        run_id = match.group(1).strip()

    candidates: list[Path] = []
    if run_id:
        candidates.append(OUTPUT_DIR / run_id / ".step_result.json")
        candidates.append(OUTPUT_DIR / run_id / "script.json")

    for path in sorted(OUTPUT_DIR.glob("*/.step_result.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
        candidates.append(path)

    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if path.name == "script.json":
            run_dir = path.parent
            return {
                "stage": "script",
                "run_id": run_dir.name,
                "output_dir": str(run_dir),
                "skipped": False,
                "script_meta": {
                    "title": str(data.get("title", "")),
                    "youtube_title": str(data.get("youtube_title", "")),
                    "youtube_description": str(data.get("youtube_description", "")),
                    "youtube_comment": str(data.get("youtube_comment", "")),
                    "hook_text": str(data.get("hook_text", "")),
                    "language": str(data.get("language", "")),
                    "content_type": str(data.get("content_type") or data.get("theme", "")),
                    "title_style": str(data.get("title_style", "")),
                },
                "lang": data.get("language", ""),
                "theme": data.get("content_type") or data.get("theme", ""),
                "recovered": True,
            }
        if isinstance(data, dict) and data.get("run_id"):
            data["recovered"] = True
            return data
    return None


def _parse_json_stdout(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    for chunk in _json_chunks(stdout):
        try:
            data = json.loads(chunk)
            if isinstance(data, dict) and data.get("run_id"):
                return data
        except json.JSONDecodeError:
            continue
    extracted = _extract_json_object(stdout)
    if extracted is not None:
        return extracted
    recovered = _recover_step_result(stdout, stderr)
    if recovered is not None:
        logger.warning(
            "Recovered pipeline result from disk for run_id=%s",
            recovered.get("run_id", ""),
        )
        return recovered
    if proc.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed (exit {proc.returncode}): {(stderr or stdout)[-2000:]}"
        )
    raise RuntimeError(
        f"Pipeline did not return JSON (stdout {len(stdout)} bytes): {stdout[-2000:]}"
    )


def _json_chunks(text: str):
    """Yield likely JSON payloads from stdout (logs may precede the final object)."""
    text = text.strip()
    if not text:
        return
    yield text
    last_line_break = text.rfind("\n{")
    if last_line_break >= 0:
        yield text[last_line_break + 1 :]
    first_brace = text.find("{")
    if first_brace > 0:
        yield text[first_brace:]


def _run_subprocess_with_retry(cmd: list[str], label: str) -> dict[str, Any]:
    """Run a pipeline subprocess; retry transient failures with a long wait."""
    last_error: Exception | None = None
    for attempt in range(1, STEP_MAX_TRIES + 1):
        logger.info("Running %s (attempt %d/%d): %s", label, attempt, STEP_MAX_TRIES, " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            return _parse_json_stdout(proc)
        except RuntimeError as exc:
            last_error = exc
            if attempt >= STEP_MAX_TRIES:
                raise
            logger.warning("%s failed (attempt %d/%d): %s", label, attempt, STEP_MAX_TRIES, exc)
            if STEP_RETRY_WAIT_SEC:
                logger.info("Waiting %ds before retry...", STEP_RETRY_WAIT_SEC)
                time.sleep(STEP_RETRY_WAIT_SEC)
    raise last_error or RuntimeError(f"{label} failed")


def _run_pipeline(body: dict[str, Any]) -> dict[str, Any]:
    lang = str(body.get("lang", "")).strip().lower()
    theme = str(body.get("theme", "")).strip().lower()
    themes_csv = str(body.get("themesCsv", "")).strip()
    duration = int(body.get("duration", 45))
    tier = str(body.get("tier", "flux")).strip().lower()

    if lang not in {"en", "hi"}:
        raise ValueError('lang is required and must be "en" or "hi"')
    if not SCRIPT.is_file():
        raise FileNotFoundError(f"Pipeline script not found: {SCRIPT}")

    cmd = [str(SCRIPT), lang, theme, str(duration), tier]
    if themes_csv:
        cmd.append(themes_csv)
    return _run_subprocess_with_retry(cmd, "full pipeline")


def _run_step(body: dict[str, Any]) -> dict[str, Any]:
    from .pipeline import PIPELINE_STAGES

    stage = str(body.get("stage", "")).strip().lower()
    if stage not in PIPELINE_STAGES:
        raise ValueError(f'stage must be one of: {", ".join(PIPELINE_STAGES)}')
    if not STEP_SCRIPT.is_file():
        raise FileNotFoundError(f"Step script not found: {STEP_SCRIPT}")

    run_id = str(body.get("run_id", "")).strip()
    if stage != "script" and not run_id:
        raise ValueError("run_id is required for all stages except script")

    lang = str(body.get("lang", "")).strip().lower()
    theme = str(body.get("theme", "")).strip().lower()
    themes_csv = str(body.get("themesCsv", "")).strip()
    duration = int(body.get("duration", 45))
    tier = str(body.get("tier", "flux")).strip().lower()

    if stage == "script":
        if lang not in {"en", "hi"}:
            raise ValueError('lang is required and must be "en" or "hi"')
        cmd = [str(STEP_SCRIPT), stage, "", lang, theme, str(duration), tier]
        if themes_csv:
            cmd.append(themes_csv)
    else:
        cmd = [str(STEP_SCRIPT), stage, run_id]

    return _run_subprocess_with_retry(cmd, f"step {stage}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            self._send(200, {"ok": True, "service": "video-pipeline-api"})
            return
        if path == "/youtube/report":
            from .youtube_analytics import performance_report

            try:
                self._send(200, performance_report())
            except Exception as exc:
                logger.exception("Analytics report error")
                self._send(500, {"error": str(exc)})
            return
        if path == "/youtube/uploads":
            from .youtube_analytics import list_uploads

            self._send(200, list_uploads())
            return
        if path == "/youtube/pending-comments":
            from .youtube_analytics import list_pending_comments

            self._send(200, list_pending_comments())
            return
        if path.startswith("/video/"):
            run_id = path[len("/video/") :].strip("/")
            try:
                video_path = _final_video_for_run(run_id)
                data = video_path.read_bytes()
                self._send_bytes(200, data, "video/mp4")
            except FileNotFoundError as exc:
                self._send(404, {"error": str(exc)})
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in {
            "/generate",
            "/step",
            "/youtube/uploaded",
            "/youtube/sync-stats",
            "/youtube/push-stats",
            "/youtube/comment-posted",
        }:
            self._send(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
            if path == "/youtube/uploaded":
                from .youtube_analytics import record_upload

                rec = record_upload(
                    run_id=str(body.get("run_id", "")).strip(),
                    video_id=str(body.get("video_id", "")).strip(),
                    title=str(body.get("title", "")).strip(),
                )
                self._send(200, {"ok": True, "upload": rec})
                return
            if path == "/youtube/sync-stats":
                from .youtube_analytics import performance_report, sync_stats

                result = sync_stats()
                result["report"] = performance_report()
                self._send(200, result)
                return
            if path == "/youtube/push-stats":
                from .youtube_analytics import performance_report, push_stats

                raw_stats = body.get("stats")
                if not isinstance(raw_stats, list):
                    raise ValueError('stats must be an array of {video_id, views, likes, comments}')
                result = push_stats(raw_stats)
                result["report"] = performance_report()
                self._send(200, result)
                return
            if path == "/youtube/comment-posted":
                from .youtube_analytics import mark_comment_posted

                rec = mark_comment_posted(str(body.get("video_id", "")).strip())
                self._send(200, {"ok": True, **rec})
                return
            if path == "/step":
                result = _run_step(body)
            else:
                result = _run_pipeline(body)
            self._send(200, _enrich_result(result, self))
        except (ValueError, FileNotFoundError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Pipeline error")
            self._send(500, {"error": str(exc)})


def main() -> None:
    configure_logging(level=logging.INFO)
    if not SCRIPT.is_file():
        raise SystemExit(f"Missing script: {SCRIPT}")
    if not STEP_SCRIPT.is_file():
        raise SystemExit(f"Missing script: {STEP_SCRIPT}")

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    logger.info("Video pipeline API listening on http://%s:%s", HOST, PORT)
    logger.info(
        "POST /generate  POST /step  POST /youtube/uploaded  POST /youtube/sync-stats  "
        "POST /youtube/push-stats  POST /youtube/comment-posted  GET /health  GET /video/{run_id}  "
        "GET /youtube/report  GET /youtube/uploads  GET /youtube/pending-comments"
    )
    logger.info(
        "Step retries: max_tries=%d wait_sec=%d (override with N8N_STEP_MAX_TRIES / N8N_STEP_RETRY_WAIT_SEC)",
        STEP_MAX_TRIES,
        STEP_RETRY_WAIT_SEC,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
