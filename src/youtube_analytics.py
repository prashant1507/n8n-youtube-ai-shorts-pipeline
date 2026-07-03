"""YouTube performance feedback loop.

records/uploads.json links each run to its uploaded YouTube video and stores
stats snapshots (views/likes/comments from the Data API v3). The SEO metadata
prompt reads this registry so title writing learns from real view data, and
title styles assigned by title_ab can be compared against each other.

Stats sync needs a YouTube Data API v3 key in the YOUTUBE_API_KEY env var
(public statistics only: no OAuth required).

CLI:
    python -m src.youtube_analytics sync     # fetch fresh stats for all uploads
    python -m src.youtube_analytics report   # print the performance report
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from .config import ROOT

logger = logging.getLogger(__name__)

RECORDS_DIR = ROOT / "records"
UPLOADS_PATH = RECORDS_DIR / "uploads.json"
OUTPUT_DIR = ROOT / "output"

API_KEY_ENV = "YOUTUBE_API_KEY"
VIDEOS_API = "https://www.googleapis.com/youtube/v3/videos"
MAX_SNAPSHOTS = 30  # per video; oldest are dropped


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().replace(microsecond=0).isoformat()


def _load_uploads() -> list[dict[str, Any]]:
    if not UPLOADS_PATH.exists():
        return []
    try:
        data = json.loads(UPLOADS_PATH.read_text(encoding="utf-8") or "[]")
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", UPLOADS_PATH, exc)
        return []


def _save_uploads(records: list[dict[str, Any]]) -> None:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _script_for_run(run_id: str) -> dict[str, Any]:
    path = OUTPUT_DIR / run_id / "script.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def record_upload(run_id: str, video_id: str, title: str = "") -> dict[str, Any]:
    """Register an uploaded video; enrich from the run's script.json when available."""
    if not video_id:
        raise ValueError("video_id is required")

    records = _load_uploads()
    for rec in records:
        if rec.get("video_id") == video_id:
            return rec

    script = _script_for_run(run_id) if run_id else {}
    rec = {
        "run_id": run_id,
        "video_id": video_id,
        "title": title or script.get("youtube_title") or script.get("title", ""),
        "title_style": script.get("title_style", ""),
        "hook_text": script.get("hook_text", ""),
        "language": script.get("language", ""),
        "content_type": script.get("content_type") or script.get("theme", ""),
        "uploaded_at": _now_iso(),
        "stats": [],
    }
    records.append(rec)
    _save_uploads(records)
    logger.info("Recorded upload %s (%s)", video_id, rec["title"])
    return rec


def _fetch_stats_batch(video_ids: list[str], api_key: str) -> dict[str, dict[str, int]]:
    query = urllib.parse.urlencode({
        "part": "statistics",
        "id": ",".join(video_ids),
        "key": api_key,
    })
    with urllib.request.urlopen(f"{VIDEOS_API}?{query}", timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    stats: dict[str, dict[str, int]] = {}
    for item in payload.get("items", []):
        s = item.get("statistics", {})
        stats[item["id"]] = {
            "views": int(s.get("viewCount", 0)),
            "likes": int(s.get("likeCount", 0)),
            "comments": int(s.get("commentCount", 0)),
        }
    return stats


def sync_stats(api_key: str | None = None) -> dict[str, Any]:
    """Fetch current public statistics for every recorded upload."""
    key = (api_key or os.environ.get(API_KEY_ENV, "")).strip()
    if not key:
        raise ValueError(
            f"No YouTube API key: set the {API_KEY_ENV} env var "
            "(Data API v3 key from Google Cloud Console)"
        )

    records = _load_uploads()
    ids = [r["video_id"] for r in records if r.get("video_id")]
    if not ids:
        return {"synced": 0, "missing": []}

    fetched: dict[str, dict[str, int]] = {}
    for i in range(0, len(ids), 50):
        fetched.update(_fetch_stats_batch(ids[i : i + 50], key))

    now = _now_iso()
    synced = 0
    missing: list[str] = []
    for rec in records:
        vid = rec.get("video_id", "")
        stats = fetched.get(vid)
        if stats is None:
            missing.append(vid)  # deleted, private, or not yet indexed
            continue
        rec.setdefault("stats", []).append({"fetched_at": now, **stats})
        rec["stats"] = rec["stats"][-MAX_SNAPSHOTS:]
        synced += 1

    _save_uploads(records)
    logger.info("Synced stats for %d/%d videos", synced, len(ids))
    return {"synced": synced, "missing": missing}


def _latest_stats(rec: dict[str, Any]) -> dict[str, Any] | None:
    stats = rec.get("stats") or []
    return stats[-1] if stats else None


def _age_hours(rec: dict[str, Any]) -> float | None:
    raw = rec.get("uploaded_at", "")
    try:
        uploaded = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if uploaded.tzinfo is None:
        uploaded = uploaded.replace(tzinfo=timezone.utc)
    return (_now() - uploaded).total_seconds() / 3600


def _views_per_day(rec: dict[str, Any]) -> float | None:
    latest = _latest_stats(rec)
    age = _age_hours(rec)
    if latest is None or age is None or age <= 0:
        return None
    return latest["views"] / max(age / 24, 0.25)


def _measured(records: list[dict[str, Any]], min_age_hours: float) -> list[dict[str, Any]]:
    """Uploads old enough to judge, with at least one stats snapshot."""
    out = []
    for rec in records:
        age = _age_hours(rec)
        if age is None or age < min_age_hours:
            continue
        if _views_per_day(rec) is None:
            continue
        out.append(rec)
    return out


def performance_report(min_age_hours: float = 48) -> dict[str, Any]:
    """Per-video views/day ranking and per-title-style aggregates."""
    records = _measured(_load_uploads(), min_age_hours)
    videos = []
    for rec in records:
        latest = _latest_stats(rec) or {}
        videos.append({
            "video_id": rec.get("video_id", ""),
            "title": rec.get("title", ""),
            "title_style": rec.get("title_style", ""),
            "language": rec.get("language", ""),
            "content_type": rec.get("content_type", ""),
            "views": latest.get("views", 0),
            "likes": latest.get("likes", 0),
            "comments": latest.get("comments", 0),
            "views_per_day": round(_views_per_day(rec) or 0, 1),
        })
    videos.sort(key=lambda v: v["views_per_day"], reverse=True)

    styles: dict[str, dict[str, Any]] = {}
    for v in videos:
        style = v["title_style"] or "unknown"
        agg = styles.setdefault(style, {"videos": 0, "total_views_per_day": 0.0})
        agg["videos"] += 1
        agg["total_views_per_day"] += v["views_per_day"]
    for agg in styles.values():
        agg["avg_views_per_day"] = round(agg.pop("total_views_per_day") / agg["videos"], 1)

    return {"videos": videos, "title_styles": styles, "measured": len(videos)}


def seo_feedback_lines(
    language: str,
    *,
    min_videos: int = 4,
    min_age_hours: float = 48,
    top_n: int = 3,
) -> str:
    """Prompt block with real performance data, or '' until there is enough signal."""
    lang = language.lower().strip()
    records = [
        r for r in _measured(_load_uploads(), min_age_hours)
        if not lang or (r.get("language", "").lower() or lang) == lang
    ]
    if len(records) < min_videos:
        return ""

    records.sort(key=lambda r: _views_per_day(r) or 0, reverse=True)

    def fmt(rec: dict[str, Any]) -> str:
        style = rec.get("title_style") or "unknown"
        vpd = _views_per_day(rec) or 0
        return f'- "{rec.get("title", "")}" [{style}]: {vpd:.0f} views/day'

    bottom = records[top_n:][-top_n:]
    lines = [
        "### REAL YOUTUBE PERFORMANCE DATA (from this channel: learn from it) ###",
        "Top performing titles:",
        *[fmt(r) for r in records[:top_n]],
    ]
    if bottom:
        lines.append("Under-performing titles:")
        lines.extend(fmt(r) for r in bottom)

    styles: dict[str, list[float]] = {}
    for rec in records:
        style = rec.get("title_style") or ""
        if style:
            styles.setdefault(style, []).append(_views_per_day(rec) or 0)
    if styles:
        avg = {s: sum(v) / len(v) for s, v in styles.items()}
        ranked = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)
        lines.append(
            "Title style averages: "
            + ", ".join(f"{s}: {v:.0f} views/day ({len(styles[s])} videos)" for s, v in ranked)
        )

    lines.append(
        "Write titles that share what makes the top performers work; "
        "avoid the patterns of the under-performers. Never copy a past title."
    )
    return "\n".join(lines)


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "sync":
        result = sync_stats()
    elif cmd == "report":
        result = performance_report()
    else:
        raise SystemExit(f"Unknown command {cmd!r}: expected 'sync' or 'report'")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
