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

Start the n8n API (uses `flux-venv/bin/python`):

```bash
./scripts/start-n8n-api.sh
```

## flux tier (`flux-venv` + `image-venv`)

The **full pipeline** runs from `flux-venv` (TTS, music, orchestration).  
**FLUX.2 Klein** images run in a separate `image-venv` subprocess (transformers 5 — incompatible with parler-tts in flux-venv). If `image-venv` is missing, create it with the steps above.

Run the pipeline:

```bash
source flux-venv/bin/activate
export TOKENIZERS_PARALLELISM=false

python -m src.pipeline --theme story --lang en --duration 45 --tier flux
```

Optional in `default.yaml`: `flux_python: ""` (empty = auto-detect `image-venv` in project root).

## wan tier (`flux-venv` + prepared model)

Wan video uses `wan-venv/bin/mlxgen` automatically. Run the **full pipeline from flux-venv** (parler-tts breaks in wan-venv).

**Memory:** Wan 2.2 5B needs several GB of free unified memory. If the process dies with `zsh: killed`, macOS ran out of RAM — the pipeline now runs Wan in an **isolated subprocess** after TTS/music unload. Close other apps, or resume only the video stage:

```bash
python -m src.pipeline --video-only --from-run output/YYYYMMDD_HHMMSS_xxx --tier wan --lang hi
```

One-time model prep:

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

| Flag | Required | Description |
|------|----------|-------------|
| `--theme` | no | Content type from `themes` in `default.yaml`; omit or `auto` = random |
| `--themes-csv` | no | Comma-separated themes for random pick (overrides `default.yaml` list) |
| `--lang` | no | `en` or `hi` (default from config) |
| `--duration` | no | Video duration in seconds, 10–120 (default from config) |
| `--tier` | no | `flux` or `wan` (default from config) |
| `--config` | no | Path to YAML config (default: `default.yaml`) |
| `--no-subtitles` | no | Skip bottom subtitles |
| `--video-only` | no | Regenerate video only; requires `--from-run` |
| `--from-run` | with `--video-only` or `--stage` | Existing output folder |
| `--stage` | no | Run one stage only: `script`, `voice`, `music`, `images`, `clips`, `video`, `subtitles`, `audio_mix`, `final` |

```bash
python -m src.pipeline --video-only --from-run output/YYYYMMDD_HHMMSS_xxx --tier flux
```

## n8n automation (YouTube upload, 5× daily)

1. Start the API: `./scripts/start-n8n-api.sh` (keep running)
2. Import `n8n/video-pipeline-youtube.json` into self-hosted n8n
3. Edit **Configure Job** (see table below), attach YouTube OAuth on **Upload to YouTube**
4. Re-import the workflow JSON after any workflow changes in this repo

The workflow runs **one HTTP step per pipeline stage** via `POST /step`:

| n8n node | API stage | Output |
|----------|-----------|--------|
| Generate Script | `script` | `run_id`, `script.json` |
| Generate Voice | `voice` | `voice.wav` |
| Generate Music | `music` | `music.wav` |
| Generate Images | `images` | FLUX PNGs (skipped for `tier=wan`) |
| Generate Clips | `clips` | Ken Burns or Wan scene MP4s |
| Assemble Video | `video` | `video_raw.mp4` |
| Add Subtitles | `subtitles` | burned subs (skipped for Hindi) |
| Mix Audio | `audio_mix` | `audio_mixed.wav` |
| Final Video | `final` | `final.mp4` |

Legacy one-shot: `POST /generate` still works (full pipeline in one call).

### Step retries (30 min gap)

n8n **Retry On Fail** is capped at **5000 ms** (5 seconds) — it cannot wait 30 minutes between tries.

Retries are handled by the **pipeline API** instead. On failure, each `POST /step` is retried up to 3 times with a 30-minute wait (defaults in `scripts/start-n8n-api.sh`):

| Env var | Default | Meaning |
|---------|---------|---------|
| `N8N_STEP_MAX_TRIES` | `3` | Total attempts per step (1 run + 2 retries) |
| `N8N_STEP_RETRY_WAIT_SEC` | `1800` | Seconds to wait between retries (30 min) |

Leave n8n node **Retry On Fail** off (or at 5000 ms for quick network blips only — do not combine with API retries unless you want multiplied attempts).

Restart the API after changing env vars: `kill $(lsof -ti :8765) && ./scripts/start-n8n-api.sh`

### Docker n8n + file access

n8n **does not** allow Read/Write File on all of `/home/node/.n8n`. By default only:

`/home/node/.n8n-files`

So mounting `n8n_data:/home/node/.n8n` is correct for n8n settings/credentials, but **videos under `/pipeline` are still blocked** unless you do one of:

| Approach | Setup |
|----------|--------|
| **Recommended — HTTP (current workflow)** | No extra mount. **Fetch Video** downloads from `GET http://host.docker.internal:8765/video/{run_id}`. |
| **Mount into `.n8n-files`** | Add to `docker-compose.yml`: `- /path/to/n8n-youtube/output:/home/node/.n8n-files/youtube_videos:ro` and set `pipelineContainerPath` to `/home/node/.n8n-files/youtube_videos`. |
| **Allow `/pipeline`** | Add env `N8N_RESTRICT_FILE_ACCESS: /home/node/.n8n-files:/pipeline` (less strict). |

See `n8n/docker-compose.example.yml`. Remove the unused `./out:/out` mount — this project writes to `output/`, not `out/`.

Keep `- /path/to/n8n-youtube:/pipeline:ro` only if you need to browse the whole project from n8n; upload still uses **Fetch Video** + API.

### Configure Job inputs

Set these in the **Configure Job** node before each run (or leave defaults for scheduled runs).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `lang` | yes | `en` | Narration language: `en` (English) or `hi` (Hindi). Hindi disables on-screen subtitles. |
| `theme` | no | *(empty)* | Fixed content type for this run (e.g. `bedtime`, `joke`, `fantasy`). Leave empty or set `auto` to pick at random. Ignored when you want random from `themesCsv` only — leave `theme` empty. |
| `themesCsv` | no | *(empty)* | Comma-separated list used **only when `theme` is empty or `auto`**. Example: `story,joke,bedtime,friendship,fantasy`. Overrides the `themes:` list in `default.yaml` for that run. Useful when n8n cannot read your project files or you want a custom subset per workflow. |
| `duration` | no | `60` | Target video length in seconds (10–120). Actual voice length may vary slightly. |
| `tier` | no | `flux` | Visual tier: `flux` (FLUX.2 Klein images + Ken Burns) or `wan` (Wan2.2 short clips via mlxgen; needs more RAM). |
| `youtubePrivacy` | no | `public` | YouTube upload visibility: `private`, `unlisted`, or `public`. |
| `pipelineApiBase` | no | `http://host.docker.internal:8765` | Pipeline API base URL (no `/generate` suffix). Steps call `POST {base}/step`. Use `http://127.0.0.1:8765` if n8n runs on the same Mac as the API. |
| `pipelineHostPath` | no | *(project root)* | Host project root (path mapping in Parse Pipeline Result). Set to your clone path, e.g. `/Users/user/Desktop/workspace/n8n-youtube`. |
| `pipelineContainerPath` | no | `/home/node/.n8n-files/youtube_videos` | Container path for `output/` runs when using the optional youtube_videos mount. Upload uses **Fetch Video** (HTTP), not disk read. |

### Theme selection priority

1. **`theme` set** (e.g. `bedtime`) → always use that theme
2. **`theme` empty / `auto` + `themesCsv` set** → random pick from the CSV list
3. **`theme` empty / `auto` + no `themesCsv`** → random pick from `themes:` in `default.yaml`

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

   Pipeline scripts use `flux-venv/bin/python` directly (not `source activate`), so they keep working after a move once the venv is recreated.

3. **Docker n8n** — update `docker-compose.yml` mounts:
   ```yaml
   - /Users/user/Downloads/n8n-youtube/output:/home/node/.n8n-files/youtube_videos:ro
   ```

4. **n8n** — re-import `n8n/video-pipeline-youtube.json` (or set **Configure Job** → `pipelineHostPath`).

5. **Cursor** — open folder `/Users/user/Downloads/n8n-youtube`.

6. **Restart API:** `./scripts/start-n8n-api.sh`