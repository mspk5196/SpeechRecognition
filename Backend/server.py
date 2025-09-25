from concurrent.futures import process
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
    # Try to ensure ffmpeg is on PATH and log what we find, but do not fail hard.
    detected_ffmpeg = ensure_ffmpeg_in_path()
    logger.info(f"FFmpeg in PATH resolves to: {detected_ffmpeg if detected_ffmpeg else 'None'}")
    if not detected_ffmpeg:
        logger.warning(
            "FFmpeg not detected at startup. Non-WAV formats may require FFmpeg; WAV will still be handled via in-process decoder."
        )
    stt_service = WhisperSpeechToText(model_size=MODEL_SIZE)
except Exception as e:
    logger.critical(f"Failed to load Whisper model at startup: {e}")
    # Depending on your deployment, you might want to exit or provide a degraded service.
    # For now, we'll let the app start but transcription requests will fail.
    stt_service = None # Set to None if initialization fails

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