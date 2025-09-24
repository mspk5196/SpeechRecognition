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
