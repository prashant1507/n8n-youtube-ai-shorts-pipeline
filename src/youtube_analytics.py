"""YouTube performance feedback loop.

records/uploads.csv links each run to its uploaded YouTube video; records/upload_stats.csv
stores views/likes/comments snapshots over time. The SEO metadata prompt reads this
registry so title writing learns from real view data, and title styles assigned by
title_ab can be compared against each other.

Stats sync options (pick one):
  - n8n OAuth (recommended): daily workflow calls YouTube API with the same
    youTubeOAuth2Api credential as upload, then POST /youtube/push-stats.
  - API key fallback: set YOUTUBE_API_KEY and POST /youtube/sync-stats, or
    `python -m src.youtube_analytics sync`.

CLI:
    python -m src.youtube_analytics sync      # fetch fresh stats for all uploads
    python -m src.youtube_analytics report    # print the performance report
    python -m src.youtube_analytics backfill  # build uploads.csv from output/ + channel
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT

logger = logging.getLogger(__name__)

RECORDS_DIR = ROOT / "records"
UPLOADS_PATH = RECORDS_DIR / "uploads.csv"
UPLOAD_STATS_PATH = RECORDS_DIR / "upload_stats.csv"
LEGACY_UPLOADS_JSON = RECORDS_DIR / "uploads.json"
OUTPUT_DIR = ROOT / "output"

UPLOAD_FIELDS = (
    "run_id",
    "video_id",
    "title",
    "title_style",
    "hook_text",
    "language",
    "content_type",
    "uploaded_at",
    "views",
    "likes",
    "comments",
    "stats_fetched_at",
    "comment_posted_at",
)
STATS_FIELDS = ("video_id", "fetched_at", "views", "likes", "comments")

API_KEY_ENV = "YOUTUBE_API_KEY"
CHANNEL_URL_ENV = "YOUTUBE_CHANNEL_URL"
DEFAULT_CHANNEL_URL = "https://www.youtube.com/@ShortSpark123a/shorts"
VIDEOS_API = "https://www.googleapis.com/youtube/v3/videos"
MAX_SNAPSHOTS = 30  # per video; oldest are dropped


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().replace(microsecond=0).isoformat()


def _load_stats_by_video() -> dict[str, list[dict[str, Any]]]:
    if not UPLOAD_STATS_PATH.exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    try:
        with UPLOAD_STATS_PATH.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                vid = (row.get("video_id") or "").strip()
                if not vid:
                    continue
                grouped.setdefault(vid, []).append({
                    "fetched_at": row.get("fetched_at", ""),
                    "views": int(row.get("views") or 0),
                    "likes": int(row.get("likes") or 0),
                    "comments": int(row.get("comments") or 0),
                })
    except OSError as exc:
        logger.warning("Could not read %s: %s", UPLOAD_STATS_PATH, exc)
        return {}
    for snapshots in grouped.values():
        snapshots.sort(key=lambda s: s.get("fetched_at", ""))
        if len(snapshots) > MAX_SNAPSHOTS:
            del snapshots[:-MAX_SNAPSHOTS]
    return grouped


def _load_uploads_from_json() -> list[dict[str, Any]]:
    if not LEGACY_UPLOADS_JSON.exists():
        return []
    try:
        data = json.loads(LEGACY_UPLOADS_JSON.read_text(encoding="utf-8") or "[]")
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", LEGACY_UPLOADS_JSON, exc)
        return []


def _snapshot_from_csv_row(row: dict[str, str]) -> dict[str, Any] | None:
    fetched_at = (row.get("stats_fetched_at") or "").strip()
    if not fetched_at:
        return None
    return {
        "fetched_at": fetched_at,
        "views": int(row.get("views") or 0),
        "likes": int(row.get("likes") or 0),
        "comments": int(row.get("comments") or 0),
    }


def _merge_record_stats(
    history: list[dict[str, Any]],
    row: dict[str, str],
) -> list[dict[str, Any]]:
    """Combine upload_stats.csv history with the latest row in uploads.csv."""
    merged = list(history)
    inline = _snapshot_from_csv_row(row)
    if inline is None:
        return merged[-MAX_SNAPSHOTS:]

    if not merged:
        return [inline]

    last_at = merged[-1].get("fetched_at", "")
    inline_at = inline["fetched_at"]
    if inline_at > last_at:
        merged.append(inline)
    elif inline_at == last_at:
        merged[-1] = inline
    merged.sort(key=lambda s: s.get("fetched_at", ""))
    return merged[-MAX_SNAPSHOTS:]


def _latest_stats(rec: dict[str, Any]) -> dict[str, Any] | None:
    stats = rec.get("stats") or []
    return stats[-1] if stats else None


def _atomic_write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _load_uploads() -> list[dict[str, Any]]:
    if UPLOADS_PATH.exists():
        stats_by_video = _load_stats_by_video()
        records: list[dict[str, Any]] = []
        try:
            with UPLOADS_PATH.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    vid = (row.get("video_id") or "").strip()
                    if not vid:
                        continue
                    rec = {
                        "run_id": row.get("run_id", ""),
                        "video_id": vid,
                        "title": row.get("title", ""),
                        "title_style": row.get("title_style", ""),
                        "hook_text": row.get("hook_text", ""),
                        "language": row.get("language", ""),
                        "content_type": row.get("content_type", ""),
                        "uploaded_at": row.get("uploaded_at", ""),
                        "comment_posted_at": row.get("comment_posted_at", ""),
                        "stats": _merge_record_stats(stats_by_video.get(vid, []), row),
                    }
                    records.append(rec)
        except OSError as exc:
            logger.warning("Could not read %s: %s", UPLOADS_PATH, exc)
            return []
        return records

    legacy = _load_uploads_from_json()
    if legacy:
        _save_uploads(legacy)
        try:
            LEGACY_UPLOADS_JSON.unlink()
            logger.info("Migrated %s -> %s", LEGACY_UPLOADS_JSON.name, UPLOADS_PATH.name)
        except OSError:
            pass
        return legacy
    return []


def _save_uploads(records: list[dict[str, Any]]) -> None:
    upload_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []

    for rec in records:
        latest = _latest_stats(rec) or {}
        upload_rows.append({
            "run_id": rec.get("run_id", ""),
            "video_id": rec.get("video_id", ""),
            "title": rec.get("title", ""),
            "title_style": rec.get("title_style", ""),
            "hook_text": rec.get("hook_text", ""),
            "language": rec.get("language", ""),
            "content_type": rec.get("content_type", ""),
            "uploaded_at": rec.get("uploaded_at", ""),
            "views": latest.get("views", 0),
            "likes": latest.get("likes", 0),
            "comments": latest.get("comments", 0),
            "stats_fetched_at": latest.get("fetched_at", ""),
            "comment_posted_at": rec.get("comment_posted_at", ""),
        })
        vid = rec.get("video_id", "")
        for snap in rec.get("stats") or []:
            stat_rows.append({
                "video_id": vid,
                "fetched_at": snap.get("fetched_at", ""),
                "views": snap.get("views", 0),
                "likes": snap.get("likes", 0),
                "comments": snap.get("comments", 0),
            })

    _atomic_write_csv(UPLOADS_PATH, UPLOAD_FIELDS, upload_rows)
    _atomic_write_csv(UPLOAD_STATS_PATH, STATS_FIELDS, stat_rows)
    logger.info(
        "Wrote %d upload rows to %s and %d stat snapshots to %s",
        len(upload_rows),
        UPLOADS_PATH.name,
        len(stat_rows),
        UPLOAD_STATS_PATH.name,
    )


def _script_for_run(run_id: str) -> dict[str, Any]:
    path = OUTPUT_DIR / run_id / "script.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _valid_video_id(video_id: str) -> bool:
    vid = str(video_id or "").strip()
    return bool(vid) and vid.lower() not in {"none", "null", "undefined"}


def record_upload(run_id: str, video_id: str, title: str = "") -> dict[str, Any]:
    """Register an uploaded video; enrich from the run's script.json when available."""
    if not _valid_video_id(video_id):
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
        "comment_posted_at": "",
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
    """Fetch current public statistics for every recorded upload (needs YOUTUBE_API_KEY)."""
    key = (api_key or os.environ.get(API_KEY_ENV, "")).strip()
    if not key:
        raise ValueError(
            f"No YouTube API key: set the {API_KEY_ENV} env var "
            "(Data API v3 key from Google Cloud Console), or sync via n8n OAuth "
            "(POST /youtube/push-stats)."
        )

    records = _load_uploads()
    ids = [r["video_id"] for r in records if r.get("video_id")]
    if not ids:
        return {"synced": 0, "missing": []}

    fetched: dict[str, dict[str, int]] = {}
    for i in range(0, len(ids), 50):
        fetched.update(_fetch_stats_batch(ids[i : i + 50], key))

    return _apply_fetched_stats(records, fetched, report_missing=True)


def _apply_fetched_stats(
    records: list[dict[str, Any]],
    fetched: dict[str, dict[str, int]],
    *,
    report_missing: bool = True,
) -> dict[str, Any]:
    now = _now_iso()
    synced = 0
    updated = 0
    missing: list[str] = []
    updates: list[dict[str, Any]] = []

    for rec in records:
        vid = rec.get("video_id", "")
        stats = fetched.get(vid)
        if stats is None:
            if report_missing:
                missing.append(vid)
            continue

        snap = {"fetched_at": now, **stats}
        bucket = rec.setdefault("stats", [])
        prev = bucket[-1] if bucket else None

        if prev and prev.get("fetched_at", "")[:16] == now[:16]:
            bucket[-1] = snap
        else:
            bucket.append(snap)
        rec["stats"] = bucket[-MAX_SNAPSHOTS:]

        if (
            prev is None
            or prev.get("views") != snap["views"]
            or prev.get("likes") != snap["likes"]
            or prev.get("comments") != snap["comments"]
            or prev.get("fetched_at") != snap["fetched_at"]
        ):
            updated += 1
            updates.append({
                "video_id": vid,
                "title": rec.get("title", ""),
                "views": snap["views"],
                "likes": snap["likes"],
                "comments": snap["comments"],
                "stats_fetched_at": snap["fetched_at"],
            })
        synced += 1

    _save_uploads(records)
    logger.info(
        "Synced stats for %d/%d videos (%d rows changed in CSV)",
        synced,
        len(records),
        updated,
    )
    return {
        "synced": synced,
        "updated": updated,
        "missing": missing,
        "updates": updates[:20],
    }


def push_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge stats pushed from n8n (YouTube OAuth HTTP Request — no API key on the Mac)."""
    fetched: dict[str, dict[str, int]] = {}
    for entry in entries:
        vid = str(entry.get("video_id", "")).strip()
        if not vid:
            continue
        fetched[vid] = {
            "views": int(entry.get("views", 0)),
            "likes": int(entry.get("likes", 0)),
            "comments": int(entry.get("comments", 0)),
        }

    records = _load_uploads()
    if not records:
        return {"synced": 0, "missing": [], "unknown": list(fetched.keys())}

    known = {r.get("video_id") for r in records}
    unknown = [vid for vid in fetched if vid not in known]
    result = _apply_fetched_stats(records, fetched, report_missing=False)
    if unknown:
        result["unknown"] = unknown
    return result


def list_uploads() -> dict[str, Any]:
    """Video IDs registered after upload (for n8n OAuth stats fetch)."""
    records = _load_uploads()
    uploads = []
    for rec in records:
        latest = _latest_stats(rec) or {}
        uploads.append({
            **rec,
            "views": latest.get("views", 0),
            "likes": latest.get("likes", 0),
            "comments": latest.get("comments", 0),
            "stats_fetched_at": latest.get("fetched_at", ""),
        })
    return {
        "count": len(records),
        "video_ids": [r["video_id"] for r in records if r.get("video_id")],
        "uploads": uploads,
    }


def _engagement_comment_for_run(run_id: str) -> str:
    """Reuse youtube_comment from script generation when output/{run_id}/script.json exists."""
    script = _script_for_run(run_id)
    return str(script.get("youtube_comment") or "").strip()



def _pool_engagement_comment(language: str, video_id: str) -> str:
    """Pick a comment from the language pool (varies per video and per run day)."""
    from .config import load_config

    cfg = load_config()
    lang = (language or "").lower().strip()
    if lang.startswith("hi") or lang == "hindi":
        pool = [t.strip() for t in cfg.analytics.engagement_comment_pool_hi if t.strip()]
        fallback = cfg.analytics.engagement_comment_hi
    else:
        pool = [t.strip() for t in cfg.analytics.engagement_comment_pool_en if t.strip()]
        fallback = cfg.analytics.engagement_comment_en

    if not pool:
        return fallback

    key = f"{video_id}:{date.today().isoformat()}"
    idx = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(pool)
    return pool[idx]


def _comment_for_upload(rec: dict[str, Any]) -> tuple[str, str]:
    """Return (comment_text, source) — script youtube_comment or pool fallback."""
    stored = _engagement_comment_for_run(rec.get("run_id", ""))
    if stored:
        return stored, "script"
    return _pool_engagement_comment(rec.get("language", ""), rec.get("video_id", "")), "pool"


def list_pending_comments() -> dict[str, Any]:
    """All uploads with a valid video_id — script comment or pool pick when no output script."""
    items: list[dict[str, str]] = []
    already_commented = 0
    from_script = 0
    from_pool = 0

    for rec in _load_uploads():
        vid = rec.get("video_id", "")
        if not _valid_video_id(vid):
            continue
        if rec.get("comment_posted_at"):
            already_commented += 1
        comment_text, source = _comment_for_upload(rec)
        if source == "script":
            from_script += 1
        else:
            from_pool += 1
        items.append({
            "video_id": vid,
            "run_id": rec.get("run_id", ""),
            "title": rec.get("title", ""),
            "comment_text": comment_text,
            "comment_source": source,
            "comment_posted_at": rec.get("comment_posted_at", ""),
        })

    return {
        "count": len(items),
        "items": items,
        "already_commented": already_commented,
        "from_script": from_script,
        "from_pool": from_pool,
    }


def mark_comment_posted(video_id: str) -> dict[str, Any]:
    """Record that the engagement comment was posted (upload or catch-up workflow)."""
    if not _valid_video_id(video_id):
        raise ValueError("video_id is required")

    records = _load_uploads()
    for rec in records:
        if rec.get("video_id") == video_id:
            when = _now_iso()
            rec["comment_posted_at"] = when
            _save_uploads(records)
            logger.info("Marked comment posted for %s", video_id)
            return {"video_id": video_id, "comment_posted_at": when}

    raise ValueError(f"video_id not in uploads registry: {video_id}")


def _norm_title(title: str) -> str:
    s = re.sub(r"#\S+", "", title or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)


def _run_id_timestamp(run_id: str) -> str:
    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", run_id)
    if not m:
        return ""
    dt = datetime(
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
        int(m.group(5)),
        int(m.group(6)),
        tzinfo=timezone.utc,
    )
    return dt.replace(microsecond=0).isoformat()


def _load_stories_by_run() -> dict[str, dict[str, Any]]:
    path = RECORDS_DIR / "stories.json"
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8") or "[]")
    except (json.JSONDecodeError, OSError):
        return {}
    return {row.get("run_id", ""): row for row in rows if row.get("run_id")}


def _fetch_channel_titles(channel_url: str) -> list[dict[str, str]]:
    """List channel videos via yt-dlp (no API key) or YouTube Data API."""
    ytdlp = shutil.which("yt-dlp")
    if ytdlp:
        proc = subprocess.run(
            [ytdlp, "--flat-playlist", "--print", "%(id)s\t%(title)s", channel_url],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "yt-dlp failed")
        videos: list[dict[str, str]] = []
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            video_id, title = line.split("\t", 1)
            videos.append({
                "video_id": video_id,
                "title": title,
                "norm": _norm_title(title),
            })
        return videos

    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise ValueError(
            "Install yt-dlp (brew install yt-dlp) or set YOUTUBE_API_KEY for channel listing"
        )

    handle = channel_url.rstrip("/").split("@")[-1].split("/")[0]
    query = urllib.parse.urlencode({
        "part": "contentDetails",
        "forHandle": handle,
        "key": key,
    })
    with urllib.request.urlopen(
        f"https://www.googleapis.com/youtube/v3/channels?{query}",
        timeout=30,
    ) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    items = payload.get("items") or []
    if not items:
        raise ValueError(f"No channel found for {channel_url!r}")
    playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    videos = []
    token = ""
    while True:
        params = {
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": "50",
            "key": key,
        }
        if token:
            params["pageToken"] = token
        q = urllib.parse.urlencode(params)
        with urllib.request.urlopen(
            f"https://www.googleapis.com/youtube/v3/playlistItems?{q}",
            timeout=30,
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for item in data.get("items", []):
            title = item["snippet"]["title"]
            videos.append({
                "video_id": item["contentDetails"]["videoId"],
                "title": title,
                "norm": _norm_title(title),
            })
        token = data.get("nextPageToken") or ""
        if not token:
            break
    return videos


def _match_channel_video(
    script: dict[str, Any],
    channel_videos: list[dict[str, str]],
) -> dict[str, str] | None:
    candidates = [
        script.get("youtube_title", ""),
        script.get("youtube_title_base", ""),
        script.get("title", ""),
    ]
    for cand in candidates:
        norm = _norm_title(str(cand))
        if not norm:
            continue
        for video in channel_videos:
            yt_norm = video["norm"]
            if norm == yt_norm or norm in yt_norm or yt_norm in norm:
                return video
    return None


def _run_has_final_video(run_dir: Path) -> bool:
    """True when the pipeline produced an uploadable final (subtitled or plain)."""
    return (run_dir / "final_subtitled.mp4").is_file() or (run_dir / "final.mp4").is_file()


def backfill_uploads(
    *,
    channel_url: str | None = None,
    require_final: bool = True,
    merge: bool = True,
) -> dict[str, Any]:
    """Build records/uploads.csv by matching output/*/script.json to channel titles."""
    channel = (channel_url or os.environ.get(CHANNEL_URL_ENV) or DEFAULT_CHANNEL_URL).strip()
    channel_videos = _fetch_channel_titles(channel)
    stories = _load_stories_by_run()

    existing = _load_uploads() if merge else []
    by_video = {r["video_id"]: r for r in existing if r.get("video_id")}
    matched = 0
    skipped_no_final = 0
    skipped_no_match = 0

    for script_path in sorted(OUTPUT_DIR.glob("*/script.json")):
        run_dir = script_path.parent
        run_id = run_dir.name
        if require_final and not _run_has_final_video(run_dir):
            skipped_no_final += 1
            continue

        script = _script_for_run(run_id)
        if not script:
            continue

        video = _match_channel_video(script, channel_videos)
        if not video:
            skipped_no_match += 1
            continue

        vid = video["video_id"]
        if not _valid_video_id(vid) or vid in by_video:
            continue

        story = stories.get(run_id, {})
        rec = {
            "run_id": run_id,
            "video_id": vid,
            "title": script.get("youtube_title") or script.get("title", video["title"]),
            "title_style": script.get("title_style", ""),
            "hook_text": script.get("hook_text", ""),
            "language": script.get("language", story.get("language", "")),
            "content_type": script.get("content_type") or script.get("theme", story.get("content_type", "")),
            "uploaded_at": _run_id_timestamp(run_id) or story.get("created_at", _now_iso()),
            "comment_posted_at": "",
            "stats": [],
        }
        by_video[vid] = rec
        matched += 1

    records = sorted(by_video.values(), key=lambda r: r.get("uploaded_at", ""))
    _save_uploads(records)
    logger.info("Backfilled %d uploads (%d total in registry)", matched, len(records))
    return {
        "matched": matched,
        "total": len(records),
        "skipped_no_final": skipped_no_final,
        "skipped_no_match": skipped_no_match,
        "channel_videos": len(channel_videos),
        "channel_url": channel,
    }


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
    elif cmd == "backfill":
        result = backfill_uploads()
    else:
        raise SystemExit(f"Unknown command {cmd!r}: expected 'sync', 'report', or 'backfill'")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
