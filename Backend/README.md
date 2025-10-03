# Local Whisper Server (Backend)

Run a local FastAPI server using faster-whisper (GPU-friendly) to transcribe audio for the React Native app with improved Tamil and English accuracy, plus optional denoising.

## Quick Start

- Python: 3.10+ recommended (3.11–3.13 OK)
- Install dependencies

```powershell
# From repo root or Backend
pip install -r Backend/requirements.txt
```

- GPU (optional, recommended for large-v3):
  - Install NVIDIA drivers
  - Install PyTorch CUDA build matching your CUDA version (see https://pytorch.org/get-started/locally/)
  - faster-whisper will automatically use CUDA if available

- FFmpeg (recommended for MP3/M4A/AAC support):
  - PowerShell (winget):

```powershell
winget install --id Gyan.FFmpeg -e
```

  - Or Chocolatey:

```powershell
choco install ffmpeg -y
```

The server attempts to auto-detect FFmpeg on Windows by merging PATH from the registry and scanning common install locations (including Winget shims under `%LOCALAPPDATA%\Microsoft\WinGet\Links`). If FFmpeg is not found, WAV files are still supported via a built-in decoder; other formats will require FFmpeg.

## Features

- Local Whisper transcription (faster-whisper backend)
- Optional Tamil -> English translation (literal + high-level summary modes)
- Intent & entity extraction for surveillance playback:
  - Relative months (last/this/next month) with specific day refinement
  - Relative weeks (last/this/next week)
  - Weekdays (English & Tamil) with relative modifiers
  - Multi-day spans (e.g., "10th to 12th last month")
  - Tamil month/day phrases and spans
  - Time ranges (from 10am to 11:30am; 14:00 to 15:45)
  - Automatic 12h -> 24h normalization
- Camera-aware playback commands ("camera 2 yesterday 5pm to 6pm")
- Auto-generated RTSP playback URL for downstream player
- Incremental self-learning: buffers high-confidence novel examples and supports manual correction append/retrain

## Camera Playback

When a command includes a camera reference, date (or resolvable relative date), and start/end times, the `/transcribe` response now contains `playback_url`.

Supported camera patterns:
```
camera 1
cam 2
channel 3
camera ten (word numbers one..twelve)
```

Channel mapping heuristic: `camera N -> NN01` (e.g., camera 1 => 0101). Adjust inside `server.py` if your NVR uses a different convention.

Playback RTSP URL format (Hikvision-style example):
```
rtsp://USER:PASS@HOST:554/Streaming/Channels/<channel>?starttime=YYYYMMDDTHHMMSSZ&endtime=YYYYMMDDTHHMMSSZ
```

Currently multi-day ranges pick the start day for URL composition (extend if your player supports cross-day playback via params).

## Example Commands

| Spoken Command | Key Parsed Entities | Result |
|----------------|---------------------|--------|
| last month on 10th from 9am to 10am | date=YYYY-08-10, start_time=09:00, end_time=10:00 | Single-day playback |
| 10th to 12th last month 3pm to 5pm | date_range_start, date_range_end, start/end times | Range metadata (URL uses start day) |
| on Monday 10am to 11am | date=(this week Monday), start/end times | Weekly day resolution |
| camera 3 last week Tuesday 14:00 to 15:30 | camera=3, date=(resolved) | Playback URL returned |
| cam two yesterday 5:15 pm to 5:45 pm | camera=2, date=(yesterday) | Playback URL returned |

## Relative Periods Supported

- this / last / next week
- this / last / next month (with specific day refinement)
- Weekdays (English & Tamil) optionally with last/this/next
- Multi-day spans (e.g., "10th to 12th last month")
- Tamil month and day phrases, including spans

## Integration Notes

Frontend auto-navigates to the CCTV screen when a playback URL is returned (React Navigation param: `playbackUrl`). The CCTV component detects playback vs live mode via presence of this param.

Environment variables (optional) influencing playback URL construction:
```
CCTV_USER=admin
CCTV_PASS=password
CCTV_HOST=192.168.1.64
```

Adjust the mapping logic or add authentication management as needed for production.

## Run the server

```powershell
cd "I:\Code Languages\AI\Speech\Backend"
python server.py
```

By default it binds to `http://192.168.10.8:8000`. Adjust the `host` in `server.py` if needed (e.g., `0.0.0.0` to listen on all interfaces).

## Endpoints

- `GET /health` — readiness and model status
- `POST /transcribe` — multipart form with fields:
  - `file` (audio file: wav, mp3, m4a, aac)
  - `language` (optional; default `auto`, e.g., `en`, `ta`). If you know it’s Tamil, pass `ta` for best results.

## Notes

- Model size is configured in `server.py` via `MODEL_SIZE` (default: `large-v3` for best TA/EN accuracy).
- Without FFmpeg, only `.wav` uploads are accepted; with FFmpeg installed, common formats work.
- Noise reduction runs when `noisereduce` is installed (already in requirements). You can disable by uninstalling the package.
- Logs will show where FFmpeg was resolved from, if available.

## Optional NLP Backends

### Transformer Seq2Seq (Experimental)

You can enable a prompt-based sequence-to-sequence extractor (default model: `t5-small`) to produce structured JSON entities (intent, date/date_range, start_time, end_time, camera, part_of_day, direction).

Setup:
```powershell
pip install -r Backend/requirements.txt
$env:NLP_BACKEND = "transformer"
# (Optional) choose another model, e.g.
$env:TRANSFORMER_MODEL = "google/flan-t5-small"
python Backend/server.py
```

Environment variables:
```
NLP_BACKEND=transformer
TRANSFORMER_MODEL=t5-small
TRANSFORMER_MAX_NEW_TOKENS=128
TRANSFORMER_DEVICE=cuda   # or cpu
```

Merging strategy: transformer output is used first; heuristic rules fill any missing keys (never overwriting model-provided ones).

### spaCy (Alternative)

If you prefer rule/pattern driven extraction:
```powershell
python -m spacy download en_core_web_sm
$env:NLP_BACKEND = "spacy"
python Backend/server.py
```
If the model is missing a blank English pipeline is used and regex heuristics still apply.
