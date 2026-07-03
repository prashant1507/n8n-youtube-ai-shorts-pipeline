# Change log

## 2026-07-03

- Moved the upload registry from JSON to CSV: records/uploads.csv is the main table and records/upload_stats.csv stores historical view/like/comment snapshots over time.
- Added comment_posted_at column to uploads.csv so you can see when an engagement comment was last marked as posted.
- Added python -m src.youtube_analytics backfill to rebuild uploads.csv from output folders and your YouTube channel via yt-dlp; backfill now accepts both final.mp4 and final_subtitled.mp4.
- YouTube stats sync now uses n8n OAuth instead of requiring YOUTUBE_API_KEY on the Mac: Get Uploads calls the local API, Fetch YouTube Stats uses the same YouTube OAuth credential as upload, then Push Stats writes into the CSV.
- Added n8n workflow YouTube Analytics _ 22.json (daily at 22:00) for the OAuth stats sync path; YOUTUBE_API_KEY plus POST /youtube/sync-stats remains as an optional fallback.
- Fixed upload workflows so video_id is taken from the Upload to YouTube node in an Upload Complete code step; null video_id had been breaking comment posting and registry rows.
- Upload workflows now wait 90 seconds after upload before posting the engagement comment, with up to 3 retries 30 seconds apart, then call POST /youtube/comment-posted to mark the row.
- Added engagement comment catch-up workflow YouTube Engagement Comments _ 4 Days.json (every 2 days at 16:00): GET /youtube/pending-comments, post via YouTube OAuth, then mark comment posted.
- Pending-comments API returns all uploads with a valid video_id, not only rows with an empty comment_posted_at, so rerunning the job still posts comments even after one was already marked.
- When output/{run_id}/script.json is missing or has no youtube_comment, the API picks a comment from engagement_comment_pool_en or engagement_comment_pool_hi in default.yaml; the pick varies by video and run day.
- Added engagement_comment_pool_en and engagement_comment_pool_hi lists in default.yaml and src/config.py; single engagement_comment_en and engagement_comment_hi remain as fallbacks when a pool is empty.
- New API routes: GET /youtube/uploads, GET /youtube/pending-comments, POST /youtube/push-stats, POST /youtube/comment-posted; pipeline API listens on port 8765.
- Per-theme TTS voice descriptions: resolve_voice_description in theme_profiles.py reads voice_en and voice_hi from profiles when set, otherwise uses default.yaml buckets wired through script generation, TTS, and the pipeline.
- Renamed and split n8n workflows: YouTube English _ 7,10,13,16,19.json and YouTube Hindi _ 9,12,15,18,21.json replace the older single upload workflow; updated README.md and how-to-run.md for CSV registry, analytics OAuth, and engagement comments.

## Earlier releases

- v13: requirements.txt updates.
- v12: audio improvements.
- v11: SEO and audio improvements.
- v10 and below: theme rotation in serial order, quality-check agent, code refactoring, README and pipeline iterations.
