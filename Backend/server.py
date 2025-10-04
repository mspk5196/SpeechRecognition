from concurrent.futures import process
from datetime import datetime, timedelta
import logging
import json
import os
import shutil
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import the SpeechToText service from our new module
from SpeechToText import WhisperSpeechToText
try:
    from nlp_commands import LocalNLP
except Exception as e:
    LocalNLP = None  # type: ignore
    logging.warning(f"LocalNLP import failed: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Local Whisper Server", version="1.0.0")

# Add CORS middleware to allow React Native app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's origin explicitly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load .env if present
load_dotenv()

# Define the Whisper model size for the server (can override via .env)
# Use large-v3 for best EN/Tamil accuracy; GPU recommended
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
SERVER_HOST = os.getenv("WHISPER_HOST", "10.150.249.75")
SERVER_PORT = int(os.getenv("WHISPER_PORT", "8000"))

# On Windows, winget often places shims at %LOCALAPPDATA%\Microsoft\WinGet\Links.
# Ensure ffmpeg is discoverable before loading Whisper to avoid WinError 2.
def ensure_ffmpeg_in_path() -> str | None:
    candidates = []
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        candidates.append(os.path.join(local_appdata, "Microsoft", "WinGet", "Links"))
    # Common manual installs
    candidates += [
        os.path.join(os.environ.get("ProgramFiles", r"C:\\Program Files"), "ffmpeg", "bin"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)"), "ffmpeg", "bin"),
        r"C:\\ffmpeg\\bin",
    ]

    ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if ffmpeg_path:
        return ffmpeg_path

    original_path = os.environ.get("PATH", "")
    for p in candidates:
        if p and os.path.isdir(p) and p not in original_path:
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
            if ffmpeg_path:
                return ffmpeg_path

    return None

# Initialize the Whisper SpeechToText service once when the server starts
try:
    detected_ffmpeg = ensure_ffmpeg_in_path()
    logger.info(f"FFmpeg in PATH resolves to: {detected_ffmpeg if detected_ffmpeg else 'None'}")
    if not detected_ffmpeg:
        logger.warning(
            "FFmpeg not detected at startup. Non-WAV formats may require FFmpeg; WAV will still be handled via in-process decoder."
        )
    stt_service = WhisperSpeechToText(model_size=MODEL_SIZE)
except Exception as e:
    logger.critical(f"Failed to load Whisper model at startup: {e}")
    stt_service = None

# Initialize Local NLP model (optional)
nlp_engine = None
if LocalNLP is not None:
    try:
        nlp_engine = LocalNLP()
    except Exception as nlp_err:
        logger.warning(f"LocalNLP initialization failed: {nlp_err}")
else:
    logger.info("LocalNLP class not available; NLP intents disabled.")

# -------------------------------------------------------------
# Playback success memory (adaptive ranking)
# -------------------------------------------------------------
SUCCESS_MEMORY_FILE = os.path.join(os.path.dirname(__file__), 'playback_success.json')

def load_success_memory():
    try:
        if os.path.isfile(SUCCESS_MEMORY_FILE):
            with open(SUCCESS_MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_success_memory(data: dict):
    try:
        with open(SUCCESS_MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed saving success memory: {e}")

success_memory = load_success_memory()

@app.post('/playback/mark_success')
async def mark_playback_success(pattern: str = Form(...), channel: str = Form(...)):
    """Endpoint the client can call once a candidate URL is confirmed to play recorded video.
    Stores last successful pattern/channel so future generations prioritize it."""
    success_memory['last'] = {'pattern': pattern, 'channel': channel, 'ts': datetime.utcnow().isoformat()+'Z'}
    save_success_memory(success_memory)
    return {'ok': True, 'stored': success_memory['last']}

@app.get("/")
async def root():
    """Returns a welcome message and current model status."""
    model_status = stt_service.model_size if stt_service else "Not Loaded"
    return {"message": "Local Whisper Server is running!", "model": model_status}

@app.get("/health")
async def health_check():
    """Returns the health status of the server and Whisper model."""
    if stt_service and stt_service.model: # Check if model was successfully loaded
        return {"status": "healthy", "model": stt_service.model_size, "ready": True}
    else:
        # If model failed to load, the service is not fully ready for transcription
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "model": MODEL_SIZE,
                "ready": False,
                "ffmpeg": shutil.which('ffmpeg'),
                "error": "Whisper model not loaded"
            }
        )

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe (e.g., WAV, MP3, M4A)."),
    language: str = Form("auto", description="Language hint for transcription (e.g., 'en', 'ta', or 'auto' for detection). Use 'ta' for Tamil when known."),
    translate_if_tamil: bool = Form(False, description="If true and the detected/forced language is Tamil ('ta'), also return an English translation using Whisper's translate task."),
    translation_mode: str = Form("none", description="Extended translation mode: none | literal | high_level. literal uses IndicTrans2 if installed; high_level adds summarization."),
    run_nlp: bool = Form(True, description="Run local NLP intent/entity extraction on English text (translation/summary if available)."),
):
    """
    Receives an audio file and transcribes it using the local Whisper AI model.
    """
    if not stt_service or not stt_service.model:
        logger.error("Transcription requested but Whisper model is not loaded.")
        raise HTTPException(
            status_code=503,
            detail={"error": "Whisper model is not available. Server might have failed to load it.", "success": False}
        )

    try:
        logger.info(f"Received transcription request; language hint: {language}")
        logger.info(f"File info - name: {file.filename}, content_type: {file.content_type}")
        
        # Read the entire file content asynchronously
        content = await file.read()
        logger.info(f"Read {len(content)} bytes from uploaded file")

        # Call the transcription method from the service instance
        transcription_result = stt_service.transcribe_audio_file(
            content,
            file.filename or "",
            language,
            translate_if_tamil=translate_if_tamil,
            translation_mode=translation_mode,
        )

        # Prefer explicitly supplied NLP text (English) from STT service
        english_candidate = transcription_result.get("nlp_text")
        if not english_candidate:
            # Backward-compatible fallback selection
            english_candidate = (
                transcription_result.get("high_level_translation")
                or transcription_result.get("literal_translation")
                or transcription_result.get("translation_text")
            )
            if not english_candidate:
                lang_code = (transcription_result.get("language") or "").lower()
                if lang_code.startswith("en"):
                    english_candidate = transcription_result.get("text")
        nlp_payload = None
        command_text = None
        playback_url = None
        playback_alternates = []
        playback_primary_pattern = None
        playback_patterns_tried: list[dict] = []
        playback_debug: list[str] = []
        if run_nlp and nlp_engine and english_candidate:
            try:
                nlp_payload = nlp_engine.predict_intent(english_candidate)
                if nlp_payload:
                    intent = nlp_payload.get('intent')
                    entities = nlp_payload.get('entities') or {}
                    if intent == 'playback':
                        start_t = entities.get('start_time') or entities.get('from') or 'unknown'
                        end_t = entities.get('end_time') or entities.get('to') or 'unknown'
                        base_cmd = f"PLAYBACK RANGE {start_t} {end_t}".strip()
                        date_val = entities.get('date')
                        date_range_start = entities.get('date_range_start')
                        date_range_end = entities.get('date_range_end')
                        camera = entities.get('camera')
                        assumed_today = False
                        assumed_camera = False
                        # Fallbacks: if we have times but no date at all, assume today for playback URL building
                        if (not date_val and not date_range_start) and start_t != 'unknown' and end_t != 'unknown':
                            from datetime import date as _date
                            date_val = _date.today().strftime('%Y-%m-%d')
                            assumed_today = True
                        # If no camera specified, use DEFAULT_CAMERA env or 1 for building URL (still report assumption)
                        if not camera:
                            camera = os.getenv('DEFAULT_CAMERA', '1')
                            assumed_camera = True
                        if date_val:
                            command_text = f"{base_cmd} DATE {date_val}".strip()
                        elif date_range_start and date_range_end:
                            command_text = f"{base_cmd} DATE_RANGE {date_range_start} {date_range_end}".strip()
                        else:
                            command_text = base_cmd
                        if camera:
                            command_text += f" CAMERA {camera}"
                        # Build playback URL if we have times and at least a start date
                        if camera and (date_val or date_range_start) and start_t != 'unknown' and end_t != 'unknown':
                            # If a date range exists and end date different + end_time < start_time, treat as overnight span
                            overnight = False
                            chosen_date_start = date_val or date_range_start
                            chosen_date_end = None
                            if date_range_start and date_range_end and date_range_end != date_range_start:
                                chosen_date_end = date_range_end
                            playback_debug.append(f"Attempting playback build cam={camera} date_start={chosen_date_start} date_end={chosen_date_end or chosen_date_start} start={start_t} end={end_t}")
                            chosen_date = chosen_date_start
                            def norm_time(t: str) -> str:
                                parts = t.split(':')
                                h = parts[0].zfill(2)
                                m = (parts[1] if len(parts) > 1 else '00').zfill(2)
                                s = (parts[2] if len(parts) > 2 else '00').zfill(2)
                                return h + m + s
                            try:
                                base_day_start = datetime.strptime(chosen_date_start, "%Y-%m-%d")
                                start_compact = base_day_start.strftime("%Y%m%d") + 'T' + norm_time(start_t)
                                # Decide end date
                                end_date_for_build = chosen_date_start
                                if chosen_date_end:
                                    end_date_for_build = chosen_date_end
                                base_day_end = datetime.strptime(end_date_for_build, "%Y-%m-%d")
                                end_compact = base_day_end.strftime("%Y%m%d") + 'T' + norm_time(end_t)
                                # If no explicit end date but end earlier than start (e.g., 23:00 to 05:00) assume next day
                                if not chosen_date_end:
                                    try:
                                        sh, sm, *_ = (start_t+':00').split(':')
                                        eh, em, *_ = (end_t+':00').split(':')
                                        if (int(eh), int(em)) < (int(sh), int(sm)):
                                            base_day_end = base_day_start + timedelta(days=1)
                                            end_compact = base_day_end.strftime("%Y%m%d") + 'T' + norm_time(end_t)
                                            overnight = True
                                    except Exception:
                                        pass
                                rtsp_user = os.getenv('CCTV_USER', 'admin')
                                rtsp_pass = os.getenv('CCTV_PASS', os.getenv('CCTV_PASSWORD', 'password'))
                                rtsp_host = os.getenv('CCTV_HOST', '192.168.1.64')
                                rtsp_port = os.getenv('CCTV_RTSP_PORT', '554')
                                cam_int = int(camera)

                                # Channel variants (refined ordering: proven working -> others)
                                channel_variants: list[str] = []
                                base101 = f"{cam_int}01"  # e.g. 1 -> 101
                                if base101 not in channel_variants:
                                    channel_variants.append(base101)
                                zp = f"{cam_int}01"    
                                # zp = f"{cam_int:02d}01"    # zero padded e.g. 0101
                                if zp not in channel_variants:
                                    channel_variants.append(base101)
                                    # channel_variants.append(zp)
                                if str(cam_int) not in channel_variants:  # plain camera number
                                    # channel_variants.append(str(cam_int)) 
                                    channel_variants.append(base101)   
  
                                # Vendor specific pattern selection
                                vendor = os.getenv('CCTV_VENDOR', '').lower().strip()
                                # Normalize times per vendor specification downstream
                                # We'll produce st/et in multiple shapes as needed
                                # base_day_* already include 'T' between date/time. We'll derive variants:
                                #  - compact_tz (YYYYMMDDtHHMMSSz) for Hikvision (lowercase t/z)
                                #  - compact_TZ (YYYYMMDDTHHMMSSZ) generic/unv
                                #  - compact_plain (YYYYMMDDHHMMSS) Dahua/CPPlus/generic alt
                                def derive_time_forms(dt_str: str) -> dict:
                                    # dt_str like YYYYMMDDTHHMMSS
                                    if 'T' not in dt_str:
                                        base = dt_str[:8] + 'T' + dt_str[8:]
                                    else:
                                        base = dt_str
                                    ymd = base[:8]; timepart = base[9:]
                                    return {
                                        'TZ': base + 'Z',                # YYYYMMDDTHHMMSSZ
                                        'tz': (ymd + 't' + timepart + 'z'), # lowercase
                                        'plain': ymd + timepart           # YYYYMMDDHHMMSS
                                    }
                                # start_compact / end_compact currently like YYYYMMDDTHHMMSS or with T (constructed above)
                                # Ensure they follow that format for forms derivation
                                start_forms = derive_time_forms(start_compact)
                                end_forms = derive_time_forms(end_compact)

                                if vendor == 'hikvision':
                                    # Must use /Streaming/tracks/{ch}?starttime=YYYYMMDDtHHMMSSZ (lowercase t/z) per spec
                                    pattern_templates = {
                                        'hik_tracks': '/Streaming/tracks/{ch}?starttime={st_lc}&endtime={et_lc}'
                                    }
                                    # Override time vars to lowercase variant
                                    start_token = start_forms['tz']
                                    end_token = end_forms['tz']
                                elif vendor in {'dahua','cpplus','cp_plus'}:
                                    # Dahua / CP Plus playback format: /cam/playback?channel=1&starttime=YYYYMMDDHHMMSS
                                    pattern_templates = {
                                        'dahua_playback': '/cam/playback?channel={cam}&starttime={st_plain}&endtime={et_plain}'
                                    }
                                    start_token = start_forms['plain']
                                    end_token = end_forms['plain']
                                elif vendor in {'unv','uniview'}:
                                    # Uniview playback: /playback?channel=1&starttime=YYYYMMDDTHHMMSSZ
                                    pattern_templates = {
                                        'unv_playback': '/playback?channel={cam}&starttime={st_TZ}&endtime={et_TZ}'
                                    }
                                    start_token = start_forms['TZ']
                                    end_token = end_forms['TZ']
                                elif vendor == 'axis':
                                    # Axis typically not RTSP playback; keep generic track + note
                                    pattern_templates = {
                                        'axis_tracks': '/Streaming/tracks/{ch}?starttime={st}Z&endtime={et}Z'
                                    }
                                    start_token = start_forms['TZ']
                                    end_token = end_forms['TZ']
                                    playback_debug.append('Axis vendor selected: direct RTSP playback may not be supported on some devices.')
                                else:
                                    # Generic / existing fallback set including tracks
                                    pattern_templates = {
                                        'tracks':        '/Streaming/tracks/{ch}?starttime={st}Z&endtime={et}Z',
                                        'playflag':      '/Streaming/Channels/{ch}?Playback=1&starttime={st}Z&endtime={et}Z',
                                        'base':          '/Streaming/Channels/{ch}?starttime={st}Z&endtime={et}Z',
                                        'base_noz':      '/Streaming/Channels/{ch}?starttime={st}&endtime={et}',
                                        'isapi_tracks':  '/ISAPI/Streaming/tracks/{ch}?starttime={st}Z&endtime={et}Z'
                                    }
                                    start_token = start_forms['TZ']
                                    end_token = end_forms['TZ']

                                # Optional explicit ordering override: CCTV_PATTERN_ORDER=tracks,playflag,base
                                order_env = os.getenv('CCTV_PATTERN_ORDER')
                                if order_env:
                                    desired = [o.strip() for o in order_env.split(',') if o.strip()]
                                    reordered = {}
                                    for key in desired:
                                        if key in pattern_templates:
                                            reordered[key] = pattern_templates[key]
                                    for k,v in pattern_templates.items():
                                        if k not in reordered:
                                            reordered[k] = v
                                    pattern_templates = reordered

                                # Promote remembered success (adaptive ranking)
                                remembered = success_memory.get('last')
                                if remembered:
                                    remb_pat = remembered.get('pattern')
                                    remb_ch = remembered.get('channel')
                                    if remb_pat in pattern_templates:
                                        pt_copy = {remb_pat: pattern_templates[remb_pat]}
                                        for k,v in pattern_templates.items():
                                            if k != remb_pat:
                                                pt_copy[k] = v
                                        pattern_templates = pt_copy
                                    if remb_ch in channel_variants:
                                        channel_variants = [remb_ch] + [c for c in channel_variants if c != remb_ch]
                                    playback_debug.append(f"Promoted remembered success pattern={remb_pat} channel={remb_ch}")

                                # Time style handling (todo 4) using CCTV_TIME_STYLE (only for generic style patterns)
                                time_style = os.getenv('CCTV_TIME_STYLE', 'utc_z').lower()
                                offset_suffix = ''
                                if time_style.startswith('offset='):
                                    # offset=+05:30 or offset=-04:00
                                    offset_suffix = time_style.split('=',1)[1].strip()
                                    if not offset_suffix or len(offset_suffix) < 3:
                                        offset_suffix = ''  # fallback
                                # Adjust pattern templates for time style (remove Z or add offset)
                                if time_style != 'utc_z':
                                    adjusted = {}
                                    for k,v in pattern_templates.items():
                                        v2 = v.replace('{st}Z','{st}').replace('{et}Z','{et}')
                                        if offset_suffix:
                                            v2 = v2.replace('{st}', '{st}'+offset_suffix).replace('{et}','{et}'+offset_suffix)
                                        adjusted[k] = v2
                                    pattern_templates = adjusted

                                # dynamic ensure_time_tokens respects time style
                                def ensure_time_tokens(pat: str) -> str:
                                    need_suffix = 'Z' if time_style == 'utc_z' else offset_suffix
                                    if '{st}' in pat and '{et}' in pat:
                                        return pat
                                    if 'starttime=' in pat and 'endtime=' in pat:
                                        return pat
                                    joiner = '&' if '?' in pat else '?'
                                    return pat + f"{joiner}starttime={{st}}{need_suffix}&endtime={{et}}{need_suffix}"

                                # Environment pattern parsing + candidate generation (restored)
                                raw_env = os.getenv('CCTV_PLAYBACK_PATTERNS')
                                custom_items: list[dict] = []  # each: {name, pattern}
                                filtered_keys = None
                                if raw_env:
                                    raw_env_str = raw_env.strip()
                                    if raw_env_str.startswith('['):
                                        try:
                                            arr = json.loads(raw_env_str)
                                            if isinstance(arr, list):
                                                for idx, obj in enumerate(arr):
                                                    if isinstance(obj, dict) and 'pattern' in obj:
                                                        nm = obj.get('name') or f'custom_{idx}'
                                                        custom_items.append({'name': nm, 'pattern': obj['pattern']})
                                            else:
                                                playback_debug.append('CCTV_PLAYBACK_PATTERNS JSON root not list – ignored')
                                        except Exception as je:
                                            playback_debug.append(f'Failed JSON parse CCTV_PLAYBACK_PATTERNS: {je}')
                                    else:
                                        filtered_keys = {p.strip() for p in raw_env_str.replace(';', ',').split(',') if p.strip()}
                                        pattern_templates = {k: v for k, v in pattern_templates.items() if k in filtered_keys}

                                if custom_items:
                                    pattern_items = custom_items
                                else:
                                    pattern_items = [{'name': k, 'pattern': v} for k, v in pattern_templates.items()]
                                if not pattern_items:
                                    playback_debug.append('No playback patterns available after env filtering')

                                from urllib.parse import quote
                                enc_user = quote(rtsp_user, safe='')
                                enc_pass = quote(rtsp_pass, safe='')
                                for ch in channel_variants:
                                    for item in pattern_items:
                                        name = item['name']
                                        tmpl = item['pattern']
                                        # Insert vendor-specific tokens
                                        tmpl = tmpl.replace('{channel}', ch)
                                        # Provide camera number placeholder for vendor patterns using {cam}
                                        tmpl = tmpl.replace('{cam}', str(cam_int))
                                        # Provide time placeholders for vendor variants
                                        tmpl_prepared = tmpl
                                        if '{st_lc}' in tmpl_prepared or '{et_lc}' in tmpl_prepared:
                                            tmpl_prepared = tmpl_prepared.replace('{st_lc}', start_forms['tz']).replace('{et_lc}', end_forms['tz'])
                                        if '{st_TZ}' in tmpl_prepared or '{et_TZ}' in tmpl_prepared:
                                            tmpl_prepared = tmpl_prepared.replace('{st_TZ}', start_forms['TZ']).replace('{et_TZ}', end_forms['TZ'])
                                        if '{st_plain}' in tmpl_prepared or '{et_plain}' in tmpl_prepared:
                                            tmpl_prepared = tmpl_prepared.replace('{st_plain}', start_forms['plain']).replace('{et_plain}', end_forms['plain'])
                                        # Generic placeholders {st}/{et}
                                        tmpl_prepared = tmpl_prepared.replace('{st}', start_token).replace('{et}', end_token)
                                        # Ensure tokens if generic style missing explicit query params
                                        tmpl_prepared = ensure_time_tokens(tmpl_prepared) if vendor in {'','generic'} else tmpl_prepared
                                        if '{ch}' in tmpl_prepared:
                                            formatted = tmpl_prepared.format(ch=ch, user=rtsp_user, password=rtsp_pass, host=rtsp_host, port=rtsp_port)
                                        else:
                                            formatted = tmpl_prepared.format(user=rtsp_user, password=rtsp_pass, host=rtsp_host, port=rtsp_port)
                                        if not formatted.startswith('rtsp://'):
                                            full_url = f"rtsp://{rtsp_user}:{rtsp_pass}@{rtsp_host}:{rtsp_port}{formatted}"
                                        else:
                                            full_url = formatted
                                        from urllib.parse import quote as _q
                                        full_url = full_url.replace(rtsp_user, _q(rtsp_user,'')).replace(rtsp_pass, _q(rtsp_pass,''))
                                        playback_alternates.append(full_url)
                                        playback_patterns_tried.append({'pattern': name, 'channel': ch, 'url': full_url})

                                if playback_alternates:
                                    playback_url = playback_alternates[0]
                                    playback_primary_pattern = playback_patterns_tried[0]['pattern'] if playback_patterns_tried else None
                                    note = ''
                                    if overnight:
                                        note = ' (overnight span)'
                                    elif chosen_date_end and chosen_date_end != chosen_date_start:
                                        note = ' (multi-date span)'
                                    playback_debug.append(f"Generated {len(playback_alternates)} candidate URLs; primary pattern={playback_primary_pattern}{note}")
                                    playback_type = 'playback' if 'starttime=' in playback_url.lower() else 'live'
                                    # Provide expected start/end and duration for client-side auto-stop
                                    try:
                                        # Parse start/end components to datetime
                                        def _parse_hms(t: str):
                                            t3 = (t+':00:00').split(':')  # ensure at least 3
                                            hh = int(t3[0]); mm = int(t3[1]); ss = int(t3[2])
                                            return hh, mm, ss
                                        sh, sm, ss = _parse_hms(start_t)
                                        eh, em, es = _parse_hms(end_t)
                                        start_dt_local = base_day_start.replace(hour=sh, minute=sm, second=ss, microsecond=0)
                                        end_dt_local = base_day_end.replace(hour=eh, minute=em, second=es, microsecond=0)
                                        # Guard: if end earlier and no explicit end date, overnight already adjusted above
                                        duration_ms = int((end_dt_local - start_dt_local).total_seconds() * 1000)
                                        if duration_ms < 0:
                                            duration_ms = 0
                                        playback_expected = {
                                            'start_local_iso': start_dt_local.isoformat(),
                                            'end_local_iso': end_dt_local.isoformat(),
                                            'duration_ms': duration_ms
                                        }
                                    except Exception as _dt_err:  # pragma: no cover
                                        playback_expected = None
                                        playback_debug.append(f"Failed to compute expected duration: {_dt_err}")
                                else:
                                    playback_debug.append('No playback_alternates generated after pattern loop')
                                    playback_type = 'unknown'
                            except Exception as p_err:
                                logger.debug(f"Failed building playback URL: {p_err}")
                                playback_debug.append(f"Exception building playback URLs: {p_err}")
                        else:
                            playback_debug.append("Playback build conditions not met: camera or date/times missing or unknown")
                        # Attach assumption flags back into NLP payload entities for frontend transparency
                        try:
                            if assumed_today or assumed_camera:
                                if nlp_payload.get('entities') is not None:
                                    if assumed_today:
                                        nlp_payload['entities']['assumed_date_today'] = 'true'
                                        nlp_payload['entities']['date'] = date_val
                                    if assumed_camera:
                                        nlp_payload['entities']['assumed_camera_default'] = 'true'
                                        nlp_payload['entities']['camera'] = camera
                        except Exception:
                            pass
                    elif intent != 'playback':
                        # Heuristic: if clear time range present treat as playback (user might omit 'playback')
                        if ('start_time' in entities and 'end_time' in entities) and intent != 'playback':
                            intent = 'playback'
                            command_text = f"PLAYBACK RANGE {entities.get('start_time')} {entities.get('end_time')}".strip()
                    elif intent == 'ptz':
                        direction = entities.get('direction', 'center')
                        command_text = f"PTZ MOVE {direction.upper()}"
                    elif intent == 'motion_check':
                        loc = entities.get('location', 'all')
                        command_text = f"MOTION CHECK {loc.upper()}"
                    else:
                        command_text = f"INTENT {intent.upper()}" if intent else None
            except Exception as ie:
                logger.warning(f"NLP processing failed: {ie}")

        return JSONResponse({
            "text": transcription_result["text"],
            "language": transcription_result["language"],
            "segments": transcription_result["segments"],
            "translation_text": transcription_result.get("translation_text", ""),
            "translation_segments": transcription_result.get("translation_segments", 0),
            "translation_performed": transcription_result.get("translation_performed", False),
            "literal_translation": transcription_result.get("literal_translation", ""),
            "high_level_translation": transcription_result.get("high_level_translation", ""),
            "translation_mode": transcription_result.get("translation_mode", "none"),
            "nlp": nlp_payload,
            "command_text": command_text,
            "playback_url": playback_url,
            "playback_alternates": playback_alternates,
            "playback_primary_pattern": playback_primary_pattern,
            "playback_patterns_tried": playback_patterns_tried,
            "playback_debug": playback_debug,
            "playback_type": locals().get('playback_type', None),
            "playback_expected": locals().get('playback_expected', None),
            "success": True,
        })
            
    except Exception as e:
        logger.error(f"Transcription endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "text": "",
                "success": False
            }
        )

@app.get("/models")
async def list_models():
    """Returns information about the currently loaded model and available options."""
    if stt_service:
        return {
            "current_model": stt_service.model_size,
            "available_models": stt_service.get_available_models(),
            "languages": stt_service.get_supported_languages()
        }
    else:
        return {
            "current_model": "Not Loaded",
            "available_models": ["tiny", "base", "small", "medium", "large"],
            "languages": ["auto"]
        }

if __name__ == "__main__":
    print("🎤 Starting Local Whisper Server...")
    print("📝 Ensure requirements are installed:")
    print("    pip install -r Backend/requirements.txt")
    print("    (for GPU, install PyTorch CUDA build and ensure NVIDIA drivers)")
    print(f"🌐 Server will attempt to run on: http://{SERVER_HOST}:{SERVER_PORT}")
    print("📱 Use this URL in your React Native app")
    print("⚡ Press Ctrl+C to stop the server")
  
    x = torch.rand(10000, 10000).to("cuda")
    print("Tensor on:", x.device)
    print("Allocated memory (MB):", torch.cuda.memory_allocated(0) / 1024**2)

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")