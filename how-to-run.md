# How to Run

## First-time setup (flux-venv + image-venv)

From the project root:

```bash
cd /path/to/n8n-youtube

python3.12 -m venv flux-venv
source flux-venv/bin/activate
pip install -r requirements.txt
deactivate

python3.12 -m venv image-venv
source image-venv/bin/activate
pip install -r requirements-image.txt
deactivate
```

**Wan tier only** — separate env for `mlxgen` (not flux-venv):

```bash
python3.12 -m venv wan-venv
source wan-venv/bin/activate
pip install mlx-gen
deactivate
```

## Hugging Face login and model downloads

Models are pulled from the [Hugging Face Hub](https://huggingface.co/). Do this once before your first pipeline run.

### 1. Create an account and token

1. Sign up at [huggingface.co](https://huggingface.co/join).
2. Open **Settings → [Access Tokens](https://huggingface.co/settings/tokens)**.
3. Create a token with **Read** access (fine-grained or classic both work).

### 2. Log in with the Hugging Face CLI

From **flux-venv** (includes `huggingface-hub`):

```bash
source flux-venv/bin/activate
pip install -U "huggingface_hub[cli]"

hf auth login
# Paste your token when prompted, or:
# hf auth login --token hf_xxxxxxxxxxxxxxxx

hf auth whoami
deactivate
```

You only need to log in once per machine (token is saved under `~/.cache/huggingface/`).

Alternatively, set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in your shell before running the pipeline.

### 3. Request access to gated model repos

Some models require accepting a license on the Hub **before** download works. Log into the website, open each repo, and
click **Agree and access repository** (wording may vary):

| Model                    | Hub repo                                                                                      | Gated?          |
|--------------------------|-----------------------------------------------------------------------------------------------|-----------------|
| Parler-TTS (voice)       | [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)               | No              |
| MusicGen (music)         | [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small)                     | No              |
| FLUX.2 Klein (images)    | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | **Yes**         |
| Wan 2.2 (optional video) | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)     | Check repo page |

Wait until access is approved (usually instant for FLUX). If `hf download` returns **403 Forbidden**, you are not logged
in or have not accepted the license yet.

### 4. Download models

Pre-downloading avoids long waits on the first run. Cache location: `~/.cache/huggingface/hub/` (~15–20 GB for flux
tier; Wan adds more).

**Voice + music** (flux-venv):

```bash
source flux-venv/bin/activate
hf download ai4bharat/indic-parler-tts
hf download facebook/musicgen-small
deactivate
```

**FLUX.2 Klein images** (image-venv) — **required** when `flux_local_files_only: true` in `default.yaml` (default):

```bash
source image-venv/bin/activate
pip install -U "huggingface_hub[cli]"
hf auth login   # skip if already logged in from flux-venv

hf download black-forest-labs/FLUX.2-klein-4B
deactivate
```

**Wan tier** (optional) — `mlxgen prepare` downloads after Hub login:

```bash
source wan-venv/bin/activate
hf auth login   # skip if already logged in

mlxgen prepare --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --path models/wan2.2-ti2v-5b -q 8
deactivate
```

If you skip pre-download, TTS and MusicGen still fetch on first use. FLUX **will not** auto-download while
`flux_local_files_only` is `true` — run `hf download black-forest-labs/FLUX.2-klein-4B` first, or set
`flux_local_files_only: false` in `default.yaml` to allow online fetch (slower first run).

Start the n8n API (uses `flux-venv/bin/python`):

```bash
./scripts/start-n8n-api.sh
```

## flux tier (`flux-venv` + `image-venv`)

The **full pipeline** runs from `flux-venv` (TTS, music, orchestration).  
**FLUX.2 Klein** images run in a separate `image-venv` subprocess (transformers 5 — incompatible with parler-tts in
flux-venv). If `image-venv` is missing, create it with the steps above.

Run the pipeline:

```bash
source flux-venv/bin/activate
export TOKENIZERS_PARALLELISM=false

python -m src.pipeline --theme story --lang en --duration 45 --tier flux
```

Optional in `default.yaml`: `flux_python: ""` (empty = auto-detect `image-venv` in project root).

## wan tier (`flux-venv` + prepared model)

Wan video uses `wan-venv/bin/mlxgen` automatically. Run the **full pipeline from flux-venv** (parler-tts breaks in
wan-venv).

**Memory:** Wan 2.2 5B needs several GB of free unified memory. If the process dies with `zsh: killed`, macOS ran out of
RAM — the pipeline now runs Wan in an **isolated subprocess** after TTS/music unload. Close other apps, or resume only
the video stage:

```bash
python -m src.pipeline --video-only --from-run output/YYYYMMDD_HHMMSS_xxx --tier wan --lang hi
```

One-time model prep (requires Hugging Face login — see **Hugging Face login and model downloads** above):

```bash
source wan-venv/bin/activate
mlxgen prepare --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --path models/wan2.2-ti2v-5b -q 8
```

```bash
source flux-venv/bin/activate
export TOKENIZERS_PARALLELISM=false

python -m src.pipeline --theme story --lang en --duration 45 --tier wan
```

## CLI flags

| Flag             | Required                         | Description                                                                                                   |
|------------------|----------------------------------|---------------------------------------------------------------------------------------------------------------|
| `--theme`        | no                               | Content type from `themes` in `default.yaml`; omit or `auto` = random                                         |
| `--themes-csv`   | no                               | Comma-separated themes for random pick (overrides `default.yaml` list)                                        |
| `--lang`         | no                               | `en` or `hi` (default from config)                                                                            |
| `--duration`     | no                               | Video duration in seconds, 10–120 (default from config)                                                       |
| `--tier`         | no                               | `flux` or `wan` (default from config)                                                                         |
| `--config`       | no                               | Path to YAML config (default: `default.yaml`)                                                                 |
| `--no-subtitles` | no                               | Skip bottom subtitles                                                                                         |
| `--video-only`   | no                               | Regenerate video only; requires `--from-run`                                                                  |
| `--from-run`     | with `--video-only` or `--stage` | Existing output folder                                                                                        |
| `--stage`        | no                               | Run one stage only: `script`, `voice`, `music`, `images`, `clips`, `video`, `subtitles`, `audio_mix`, `final` |

```bash
python -m src.pipeline --video-only --from-run output/YYYYMMDD_HHMMSS_xxx --tier flux
```

## n8n automation (YouTube upload, 5× daily)

1. Start the API: `./scripts/start-n8n-api.sh` (keep running)
2. Import `n8n/video-pipeline-youtube.json` into self-hosted n8n
3. Edit **Configure Job** (see table below), attach YouTube OAuth on **Upload to YouTube**
4. Re-import the workflow JSON after any workflow changes in this repo

After each successful upload, **Record Upload** calls `POST /youtube/uploaded` and appends a row to
`records/uploads.csv`. See [YouTube analytics](#youtube-analytics--uploads-registry) below for stats sync.

The workflow runs **one HTTP step per pipeline stage** via `POST /step`:

| n8n node        | API stage   | Output                             |
|-----------------|-------------|------------------------------------|
| Generate Script | `script`    | `run_id`, `script.json`            |
| Generate Voice  | `voice`     | `voice.wav`                        |
| Generate Music  | `music`     | `music.wav`                        |
| Generate Images | `images`    | FLUX PNGs (skipped for `tier=wan`) |
| Generate Clips  | `clips`     | Ken Burns or Wan scene MP4s        |
| Assemble Video  | `video`     | `video_raw.mp4`                    |
| Add Subtitles   | `subtitles` | burned subs (skipped for Hindi)    |
| Mix Audio       | `audio_mix` | `audio_mixed.wav`                  |
| Final Video     | `final`     | `final.mp4`                        |

Legacy one-shot: `POST /generate` still works (full pipeline in one call).

### Step retries (30 min gap)

n8n **Retry On Fail** is capped at **5000 ms** (5 seconds) — it cannot wait 30 minutes between tries.

Retries are handled by the **pipeline API** instead. On failure, each `POST /step` is retried up to 3 times with a
30-minute wait (defaults in `scripts/start-n8n-api.sh`):

| Env var                   | Default | Meaning                                     |
|---------------------------|---------|---------------------------------------------|
| `N8N_STEP_MAX_TRIES`      | `3`     | Total attempts per step (1 run + 2 retries) |
| `N8N_STEP_RETRY_WAIT_SEC` | `1800`  | Seconds to wait between retries (30 min)    |

Leave n8n node **Retry On Fail** off (or at 5000 ms for quick network blips only — do not combine with API retries
unless you want multiplied attempts).

Restart the API after changing env vars: `kill $(lsof -ti :8765) && ./scripts/start-n8n-api.sh`

### Docker n8n + file access

n8n **does not** allow Read/Write File on all of `/home/node/.n8n`. By default only:

`/home/node/.n8n-files`

So mounting `n8n_data:/home/node/.n8n` is correct for n8n settings/credentials, but **videos under `/pipeline` are still
blocked** unless you do one of:

| Approach                                  | Setup                                                                                                                                                                           |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Recommended — HTTP (current workflow)** | No extra mount. **Fetch Video** downloads from `GET http://host.docker.internal:8765/video/{run_id}`.                                                                           |
| **Mount into `.n8n-files`**               | Add to `docker-compose.yml`: `- /path/to/n8n-youtube/output:/home/node/.n8n-files/youtube_videos:ro` and set `pipelineContainerPath` to `/home/node/.n8n-files/youtube_videos`. |
| **Allow `/pipeline`**                     | Add env `N8N_RESTRICT_FILE_ACCESS: /home/node/.n8n-files:/pipeline` (less strict).                                                                                              |

See `n8n/docker-compose.example.yml`. Remove the unused `./out:/out` mount — this project writes to `output/`, not
`out/`.

Keep `- /path/to/n8n-youtube:/pipeline:ro` only if you need to browse the whole project from n8n; upload still uses *
*Fetch Video** + API.

### Configure Job inputs

Set these in the **Configure Job** node before each run (or leave defaults for scheduled runs).

| Variable                | Required | Default                                | Description                                                                                                                                       |
|-------------------------|----------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `lang`                  | yes      | `en`                                   | Narration language: `en` (English) or `hi` (Hindi). Hindi disables on-screen subtitles.                                                           |
| `theme`                 | no       | *(empty)*                              | Fixed content type (e.g. `bedtime`). Leave empty or `auto` for **serial rotation** through the theme pool.                                        |
| `themesCsv`             | no       | *(empty)*                              | Comma-separated rotation pool when `theme` is empty. Overrides `themes:` in `default.yaml` for that workflow.                                     |
| `duration`              | no       | `60`                                   | Target video length in seconds (10–120). Actual voice length may vary slightly.                                                                   |
| `tier`                  | no       | `flux`                                 | Visual tier: `flux` (FLUX.2 Klein images + Ken Burns) or `wan` (Wan2.2 short clips via mlxgen; needs more RAM).                                   |
| `youtubePrivacy`        | no       | `public`                               | YouTube upload visibility: `private`, `unlisted`, or `public`.                                                                                    |
| `pipelineApiBase`       | no       | `http://host.docker.internal:8765`     | Pipeline API base URL (no `/generate` suffix). Steps call `POST {base}/step`. Use `http://127.0.0.1:8765` if n8n runs on the same Mac as the API. |
| `pipelineHostPath`      | no       | *(project root)*                       | Host project root (path mapping in Parse Pipeline Result). Set to your clone path, e.g. `/Users/user/Desktop/workspace/n8n-youtube`.              |
| `pipelineContainerPath` | no       | `/home/node/.n8n-files/youtube_videos` | Container path for `output/` runs when using the optional youtube_videos mount. Upload uses **Fetch Video** (HTTP), not disk read.                |

### Theme selection priority

1. **`theme` set** (e.g. `bedtime`) → always use that theme
2. **`theme` empty / `auto` + `themesCsv` set** → serial rotation through the CSV list
3. **`theme` empty / `auto` + no `themesCsv`** → serial rotation through `themes:` in `default.yaml`

Rotation state is saved in `records/theme_rotation.json`. After the last theme, the next run wraps to the first. If a
script step fails and retries, the **same** theme is reused until the script succeeds.

Example **Configure Job** for rotating kids content in English:

```
lang: en
theme: (empty)
themesCsv: story,joke,bedtime,friendship,fantasy,dragons
duration: 60
tier: flux
youtubePrivacy: public
```

### You do manually

1. **Stop API and rename folder:**
   ```bash
   kill $(lsof -ti :8765) 2>/dev/null
   cd /Users/user/Downloads/n8n-youtube
   ```

2. **Recreate venvs** (required after moving the project folder — old `activate` scripts embed the old path):

   ```bash
   rm -rf flux-venv image-venv
   python3.12 -m venv flux-venv && source flux-venv/bin/activate && pip install -r requirements.txt && deactivate
   python3.12 -m venv image-venv && source image-venv/bin/activate && pip install -r requirements-image.txt && deactivate
   ```

   Pipeline scripts use `flux-venv/bin/python` directly (not `source activate`), so they keep working after a move once
   the venv is recreated.

3. **Docker n8n** — update `docker-compose.yml` mounts:
   ```yaml
   - /Users/user/Downloads/n8n-youtube/output:/home/node/.n8n-files/youtube_videos:ro
   ```

4. **n8n** — re-import `n8n/video-pipeline-youtube.json` (or set **Configure Job** → `pipelineHostPath`).

5. **Cursor** — open folder `/Users/user/Downloads/n8n-youtube`.

6. **Restart API:** `./scripts/start-n8n-api.sh`

## YouTube analytics & uploads registry

The pipeline tracks which Shorts were uploaded and learns from view data for better SEO titles.

### Files (under `records/`)

| File | Purpose |
|------|---------|
| `uploads.csv` | One row per uploaded video — open in Excel/Numbers. Columns: `run_id`, `video_id`, `title`, `title_style`, `hook_text`, `language`, `content_type`, `uploaded_at`, `views`, `likes`, `comments`, `stats_fetched_at`, `comment_posted_at` |
| `upload_stats.csv` | Historical stat snapshots (one row per sync) |
| `stories.json` | Story dedup (separate from uploads) |

New uploads are recorded automatically when the upload workflow's **Record Upload** node succeeds.

### Stats sync workflow

1. Import `n8n/youtube-analytics-sync.json` (scheduled daily at **22:00** / 10 PM, or use **Manual Test**).
2. Set **Configure** → `pipelineApiBase` to `http://host.docker.internal:8765` (same port as the pipeline API).
3. Node auth:

| Node | URL | Authentication |
|------|-----|----------------|
| **Get Uploads** | `{pipelineApiBase}/youtube/uploads` | **None** (local Mac API) |
| **Fetch YouTube Stats** | `youtube/v3/videos?part=statistics&…` | **YouTube OAuth2 API** (same credential as upload) |
| **Push Stats** | `{pipelineApiBase}/youtube/push-stats` | **None** |

Do **not** put YouTube OAuth on **Get Uploads** — that node only reads `records/uploads.csv` on your Mac.

### CLI (from `flux-venv`)

```bash
source flux-venv/bin/activate

# Print views/day ranking + per-title-style averages
python -m src.youtube_analytics report

# Fetch fresh stats for all rows in uploads.csv (needs YOUTUBE_API_KEY on the Mac)
python -m src.youtube_analytics sync

# Rebuild uploads.csv from existing output/ runs + live channel titles
python -m src.youtube_analytics backfill
```

**Backfill** is useful if you uploaded videos before **Record Upload** was wired up. It:

- Scans `output/*/script.json` where `final.mp4` exists
- Matches titles against your channel via `yt-dlp` (`brew install yt-dlp`)
- Writes `records/uploads.csv` (default channel: `@ShortSpark123a/shorts`; override with `YOUTUBE_CHANNEL_URL`)

Verify the registry:

```bash
curl http://127.0.0.1:8765/youtube/uploads
# → {"count": 42, "video_ids": ["...", ...], "uploads": [...]}
```

Once enough videos have stats (`analytics.min_videos_for_feedback` in `default.yaml`, default 4), the SEO prompt
automatically receives top/bottom title performance data on the next script generation run.

### Engagement comment catch-up (every 2 days, 4 PM)

Import `n8n/youtube-engagement-comments.json`. It:

1. **Get Pending Comments** — `GET /youtube/pending-comments` (all uploads: `youtube_comment` from `script.json` when present, else a pool pick from `engagement_comment_pool_en` / `engagement_comment_pool_hi` in `default.yaml`)
2. **Post Comment** — YouTube OAuth, text from API response (no new LLM)
3. **Mark Comment Posted** — `POST /youtube/comment-posted` updates `comment_posted_at` in `uploads.csv` (informational; reruns still comment)