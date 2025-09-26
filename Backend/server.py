from concurrent.futures import process
from datetime import datetime
import logging
import os
import shutil
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

        # Determine best English candidate for NLP
        english_candidate = (
            transcription_result.get("high_level_translation")
            or transcription_result.get("literal_translation")
            or transcription_result.get("translation_text")
        )
        # Fallback: if none of the English transformation fields are present and original is already English
        if not english_candidate:
            lang_code = (transcription_result.get("language") or "").lower()
            if lang_code.startswith("en"):
                english_candidate = transcription_result.get("text")
        nlp_payload = None
        command_text = None
        playback_url = None
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
                        if date_val:
                            command_text = f"{base_cmd} DATE {date_val}".strip()
                        elif date_range_start and date_range_end:
                            command_text = f"{base_cmd} DATE_RANGE {date_range_start} {date_range_end}".strip()
                        else:
                            command_text = base_cmd
                        if camera:
                            command_text += f" CAMERA {camera}"
                        # Build playback URL if we have a concrete single date (or range start) and both times
                        if camera and (date_val or date_range_start) and start_t != 'unknown' and end_t != 'unknown':
                            chosen_date = date_val or date_range_start
                            def norm_time(t: str) -> str:
                                parts = t.split(':')
                                h = parts[0].zfill(2)
                                m = (parts[1] if len(parts) > 1 else '00').zfill(2)
                                s = (parts[2] if len(parts) > 2 else '00').zfill(2)
                                return h + m + s
                            try:
                                base_day = datetime.strptime(chosen_date, "%Y-%m-%d")
                                start_compact = base_day.strftime("%Y%m%d") + 'T' + norm_time(start_t)
                                end_compact = base_day.strftime("%Y%m%d") + 'T' + norm_time(end_t)
                                # Camera channel mapping heuristic: camera N -> N padded 2 digits + '01'
                                cam_channel = str(camera).zfill(2) + '01'
                                rtsp_user = os.getenv('CCTV_USER', 'admin')
                                rtsp_pass = os.getenv('CCTV_PASS', 'password')
                                rtsp_host = os.getenv('CCTV_HOST', '192.168.1.64')
                                playback_url = (
                                    f"rtsp://{rtsp_user}:{rtsp_pass}@{rtsp_host}:554/Streaming/Channels/{cam_channel}?starttime={start_compact}Z&endtime={end_compact}Z"
                                )
                            except Exception as p_err:
                                logger.debug(f"Failed building playback URL: {p_err}")
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
            "success": True,
        })
            
    except Exception as e:
        logger.error(f"Transcription endpoint error: {str(e)}", exc_info=True)
        # Return a 500 Internal Server Error with details
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
            "available_models": ["tiny", "base", "small", "medium", "large"], # Provide defaults even if service not ready
            "languages": ["auto"] # Provide default if service not ready
        }


if __name__ == "__main__":
    print("🎤 Starting Local Whisper Server...")
    print("📝 Ensure requirements are installed:")
    print("    pip install -r Backend/requirements.txt")
    print("    (for GPU, install PyTorch CUDA build and ensure NVIDIA drivers)")
    print(f"🌐 Server will attempt to run on: http://{SERVER_HOST}:{SERVER_PORT}")
    print("📱 Use this URL in your React Native app")
    print("⚡ Press Ctrl+C to stop the server")
    
    # Run the FastAPI application using uvicorn
    uvicorn.run(
        app, 
        host=SERVER_HOST,  # Listen on a specific IP or "0.0.0.0" for all interfaces
        port=SERVER_PORT,
        log_level="info"
    )