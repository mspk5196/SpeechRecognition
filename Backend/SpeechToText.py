import os
import tempfile
import uuid
import logging
import time
import shutil
import io
import wave
import numpy as np

from typing import Optional, Tuple

try:
    # GPU detection is best-effort; torch is optional
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

try:
    # High-accuracy, GPU-friendly Whisper implementation
    from faster_whisper import WhisperModel  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "faster-whisper is required. Install with: pip install faster-whisper"
    ) from e

try:
    # Lightweight spectral noise reduction
    import noisereduce as nr  # type: ignore
except Exception:
    nr = None  # Optional; we will fall back gracefully

logger = logging.getLogger(__name__)

class WhisperSpeechToText:
    """
    A service class to handle Whisper model loading and audio transcription using faster-whisper.
    """
    def __init__(self, model_size: str = "large-v3"):
        """
        Initializes the service by loading the specified Whisper model (ctranslate2 format).

        Args:
            model_size (str): The model size or path (e.g., "tiny", "base", "small", "medium", "large-v3").
        """
        self.model_size = model_size

        self._ensure_ffmpeg_in_path()

        ffmpeg_path = shutil.which("ffmpeg")
        logger.info(f"FFmpeg resolved to: {ffmpeg_path if ffmpeg_path else 'NOT FOUND'}")

        device, compute_type = self._detect_device_and_precision()
        logger.info(f"Loading faster-whisper model '{self.model_size}' on {device} ({compute_type})…")
        t0 = time.time()
        self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def _ensure_ffmpeg_in_path(self) -> None:
        """Ensure FFmpeg is discoverable for this Python process.
        - Preserves existing PATH and augments with likely Windows locations.
        - Attempts to merge PATH from registry if current process hasn't picked up updates.
        """
        if shutil.which("ffmpeg"):
            return

        original_path = os.environ.get("PATH", "")

        # 1) On Windows, merge PATH from registry (Machine + User)
        try:
            import platform
            if platform.system().lower() == "windows":
                try:
                    import winreg
                    def _read_reg_path(root, subkey, name):
                        try:
                            with winreg.OpenKey(root, subkey) as k:
                                return winreg.QueryValueEx(k, name)[0]
                        except Exception:
                            return ""
                    machine_path = _read_reg_path(
                        winreg.HKEY_LOCAL_MACHINE,
                        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                        "Path",
                    )
                    user_path = _read_reg_path(
                        winreg.HKEY_CURRENT_USER,
                        r"Environment",
                        "Path",
                    )
                    # Expand env variables in each PATH segment from registry values
                    pieces = []
                    for raw in [machine_path, user_path, original_path]:
                        if not raw:
                            continue
                        for seg in raw.split(os.pathsep):
                            seg = seg.strip()
                            if not seg:
                                continue
                            expanded = os.path.expandvars(seg)
                            pieces.append(expanded)
                    merged = os.pathsep.join(pieces)
                    if merged:
                        os.environ["PATH"] = merged
                        logger.info("Merged + expanded PATH from Windows registry for current process.")
                except Exception as e:
                    logger.debug(f"Failed merging PATH from registry: {e}")
        except Exception:
            pass

        if shutil.which("ffmpeg"):
            return

        # 2) Probe common install locations and add if found
        candidates = []
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        if local_appdata:
            candidates.extend([
                os.path.join(local_appdata, "Microsoft", "WinGet", "Links"),
                os.path.join(local_appdata, "Microsoft", "WindowsApps"),
                os.path.join(local_appdata, "Packages"),
                os.path.join(local_appdata, "Microsoft", "WinGet", "Packages"),
            ])

        candidates.extend([
            r"C:\\Program Files\\FFmpeg\\bin",
            r"C:\\Program Files (x86)\\FFmpeg\\bin",
        ])

        def _dir_contains_ffmpeg(d: str) -> str | None:
            if not d or not os.path.isdir(d):
                return None
            exe = os.path.join(d, "ffmpeg.exe")
            return exe if os.path.isfile(exe) else None

        found_dir = None
        for base in candidates:
            if not base or not os.path.exists(base):
                continue
            # Check base and common subdir 'bin'
            for d in [base, os.path.join(base, "bin")]:
                exe = _dir_contains_ffmpeg(d)
                if exe:
                    found_dir = d
                    break
            if found_dir:
                break

        # 3) If still not found, perform a shallow scan inside winget Packages dirs
        if not found_dir and local_appdata:
            scan_roots = [
                os.path.join(local_appdata, "Packages"),
                os.path.join(local_appdata, "Microsoft", "WinGet", "Packages"),
            ]
            for root in scan_roots:
                if not os.path.isdir(root):
                    continue
                try:
                    for name in os.listdir(root):
                        p = os.path.join(root, name)
                        if not os.path.isdir(p):
                            continue
                        # Check immediate and bin
                        for d in [p, os.path.join(p, "bin")]:
                            exe = _dir_contains_ffmpeg(d)
                            if exe:
                                found_dir = d
                                break
                        if found_dir:
                            break
                        # Walk a limited depth to find ffmpeg.exe
                        max_depth = 3
                        base_depth = p.rstrip(os.sep).count(os.sep)
                        for dirpath, dirnames, filenames in os.walk(p):
                            depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
                            if depth > max_depth:
                                # Prevent deep traversal
                                dirnames[:] = []
                                continue
                            if "ffmpeg.exe" in filenames:
                                found_dir = dirpath
                                break
                        if found_dir:
                            break
                except Exception:
                    continue

        # 4) As a last attempt, scan common Program Files roots
        if not found_dir:
            program_files = [
                os.environ.get("ProgramFiles"),
                os.environ.get("ProgramFiles(x86)"),
                r"C:\\Program Files",
                r"C:\\Program Files (x86)",
            ]
            for root in program_files:
                if not root or not os.path.isdir(root):
                    continue
                try:
                    base_depth = root.rstrip(os.sep).count(os.sep)
                    for dirpath, dirnames, filenames in os.walk(root):
                        depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
                        if depth > 3:
                            dirnames[:] = []
                            continue
                        if "ffmpeg.exe" in filenames:
                            found_dir = dirpath
                            break
                    if found_dir:
                        break
                except Exception:
                    continue

        if found_dir:
            os.environ["PATH"] = found_dir + os.pathsep + os.environ.get("PATH", "")
            logger.info(f"Added FFmpeg directory to PATH at runtime: {found_dir}")

        if not shutil.which("ffmpeg"):
            logger.warning(
                "FFmpeg not found for this Python process. WAV files will be handled via a built-in decoder; "
                "other formats require FFmpeg."
            )

    def _detect_device_and_precision(self) -> Tuple[str, str]:
        """Detect whether to use CUDA GPU or CPU and select a suitable compute type.
        Returns (device, compute_type).
        """
        # Forced CPU override for troubleshooting
        if os.environ.get("FWH_FORCE_CPU"):
            return "cpu", "int8"

        # Prefer CUDA if available
        try:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda", "float16"
        except Exception:
            pass

        # Fallback: detect presence of NVIDIA driver
        if shutil.which("nvidia-smi"):
            # Assume CUDA runtime is available
            return "cuda", "float16"

        # CPU fallback: use int8 for speed/accuracy balance
        return "cpu", "int8"

    def _decode_wav_to_16k_mono(self, content: bytes) -> np.ndarray:
        with wave.open(io.BytesIO(content), 'rb') as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            frames = wf.readframes(nframes)

        if sampwidth != 2:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth*8} bits. Require 16-bit PCM.")

        audio_i16 = np.frombuffer(frames, dtype=np.int16)
        if nchannels > 1:
            audio_i16 = audio_i16.reshape(-1, nchannels).mean(axis=1).astype(np.int16)

        audio_f32 = (audio_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

        target_sr = 16000
        if framerate != target_sr:
            # Linear interpolation resampling to 16k
            factor = target_sr / float(framerate)
            new_length = int(np.round(len(audio_f32) * factor))
            xp = np.linspace(0, len(audio_f32) - 1, num=len(audio_f32), dtype=np.float64)
            x_new = np.linspace(0, len(audio_f32) - 1, num=new_length, dtype=np.float64)
            audio_f32 = np.interp(x_new, xp, audio_f32).astype(np.float32)

        return audio_f32

    def _maybe_denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Apply noise reduction if the optional dependency is available.
        Keeps output in float32 [-1, 1].
        """
        if nr is None:
            return audio
        try:
            cleaned = nr.reduce_noise(y=audio.astype(np.float32), sr=sr)
            # Guard against any NaNs or out-of-range values
            if not np.any(np.isnan(cleaned)) and cleaned.size:
                return np.clip(cleaned.astype(np.float32), -1.0, 1.0)
        except Exception as e:
            logger.debug(f"Noise reduction skipped due to error: {e}")
        return audio

    def transcribe_audio_file(
        self,
        file_content: bytes,
        filename: str,
        language: str = "auto",
        translate_if_tamil: bool = False,
        translation_mode: str = "none",  # 'none' | 'literal' | 'high_level'
    ) -> dict:
        """
        Transcribes audio content from a byte stream using the loaded Whisper model.

        Args:
            file_content (bytes): The raw byte content of the audio file.
            filename (str): The original filename, used to infer the file extension.
            language (str): The language to transcribe in (e.g., "en", "es", "auto").

        Returns:
            dict: A dictionary containing:
                - text: original transcription (Tamil if source was Tamil)
                - language: detected (or forced) source language code
                - segments: number of source segments
                - translation_text: English translation if performed, else ""
                - translation_segments: number of translation segments if performed
                - translation_performed: bool flag (whisper translate pass)
                - literal_translation: Tamil->English via external MT model (if enabled and mode)
                - high_level_translation: Summarized / high-level English paraphrase
                - translation_mode: echo of mode used

        Raises:
            Exception: If an error occurs during file processing or transcription.
        """
        temp_file_path: Optional[str] = None
        try:
            # Determine file extension
            file_extension = ".wav"
            if filename:
                name, ext = os.path.splitext(filename)
                if ext.lower() in ('.mp3', '.m4a', '.aac', '.wav'):
                    file_extension = ext
            
            # Create a unique temporary file path
            temp_dir = tempfile.gettempdir()
            unique_id = str(uuid.uuid4())
            temp_file_path = os.path.join(temp_dir, f"whisper_audio_{unique_id}{file_extension}")
            
            # Write content to temporary file
            with open(temp_file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Saved temporary file for transcription: {temp_file_path}")

            # Validate temporary file
            if not os.path.exists(temp_file_path):
                raise Exception(f"Temporary file not created: {temp_file_path}")
            
            file_size = os.path.getsize(temp_file_path)
            if file_size == 0:
                raise Exception("Uploaded file content is empty")

            # Small delay to ensure file is fully written and accessible
            time.sleep(0.1) 

            logger.info(f"Starting transcription for: {temp_file_path}")

            use_ffmpeg = shutil.which("ffmpeg") is not None
            _, ext = os.path.splitext(temp_file_path)
            ext = ext.lower()

            # Build transcription params tuned for EN/TA
            beam_size = 5
            best_of = 2
            temperature = [0.0, 0.2, 0.4]
            vad_filter = False
            if torch is not None:
                try:
                    # Enable VAD only if torch is present to avoid extra deps
                    vad_filter = True
                except Exception:
                    vad_filter = False

            language_arg = None if language == "auto" else language

            text = ""
            detected_language = None
            segments_count = 0
            translation_text = ""
            translation_segments = 0
            translation_performed = False
            literal_translation = ""
            high_level_translation = ""

            if use_ffmpeg and ext != ".wav":
                # Let faster-whisper handle decoding via FFmpeg
                segments, info = self.model.transcribe(
                    temp_file_path,
                    language=language_arg,
                    beam_size=beam_size,
                    best_of=best_of,
                    temperature=temperature,
                    vad_filter=vad_filter,
                )
            else:
                if ext != ".wav" and not use_ffmpeg:
                    raise RuntimeError("FFmpeg not available; please upload a WAV file or install FFmpeg.")
                # Decode WAV manually, denoise, and pass ndarray
                audio_arr = self._decode_wav_to_16k_mono(file_content)
                audio_arr = self._maybe_denoise(audio_arr, sr=16000)
                segments, info = self.model.transcribe(
                    audio_arr,
                    language=language_arg,
                    beam_size=beam_size,
                    best_of=best_of,
                    temperature=temperature,
                    vad_filter=vad_filter,
                )

            for seg in segments:
                txt = getattr(seg, "text", "")
                if txt:
                    text += (txt if text == "" else (" " + txt.strip()))
                segments_count += 1

            detected_language = getattr(info, "language", None)
            final_language = detected_language or (language_arg or "unknown")
            logger.info(
                f"Transcription completed: {len(text)} chars; language={final_language}; segments={segments_count}"
            )

            # Optional translation path: if Tamil (explicit or detected) and user requested
            is_tamil = (final_language == "ta") or (language_arg == "ta")
            if translate_if_tamil and is_tamil:
                try:
                    logger.info("Tamil detected (or forced) and translation requested; running English translation task…")
                    # Reuse decoding path: if we already produced a numpy array, reuse; else read from file
                    translation_source = None
                    if use_ffmpeg and ext != ".wav":
                        # Use path for efficiency
                        translation_source = temp_file_path
                    else:
                        # Need an array (re-decode) to avoid side-effects
                        translation_source = self._decode_wav_to_16k_mono(file_content)
                        translation_source = self._maybe_denoise(translation_source, sr=16000)

                    t_segments, t_info = self.model.transcribe(
                        translation_source,
                        task="translate",
                        beam_size=beam_size,
                        best_of=best_of,
                        temperature=temperature,
                        vad_filter=vad_filter,
                    )
                    for seg in t_segments:
                        t_txt = getattr(seg, "text", "")
                        if t_txt:
                            translation_text += (
                                t_txt if translation_text == "" else (" " + t_txt.strip())
                            )
                        translation_segments += 1
                    translation_performed = True
                    logger.info(
                        f"Translation completed: {len(translation_text)} chars; segments={translation_segments}"
                    )
                except Exception as te:
                    logger.error(f"Translation failed: {te}")

            # Extended translation modes (literal / high_level)
            # We intentionally lazy-import optional heavy deps so base STT keeps working.
            if is_tamil and translation_mode in {"literal", "high_level"}:
                try:
                    # Lazy load external translation model (IndicTrans2) if available
                    # Users must install: indictrans2, sentencepiece, sacremoses
                    if 'IndicTransModel' not in globals():
                        try:
                            from IndicTrans2.inference.engine import IndicTransModel  # type: ignore
                            globals()['IndicTransModel'] = IndicTransModel
                        except Exception as imp_err:
                            logger.warning(
                                "IndicTrans2 not installed; literal/high_level translation skipped. "
                                "Install with: pip install indictrans2 sentencepiece sacremoses"
                            )
                            IndicTransModel = None  # type: ignore
                    IndicModelClass = globals().get('IndicTransModel')
                    if IndicModelClass:
                        if not hasattr(self, '_indic_mt'):
                            device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
                            # Use generic multi pair model; users can swap via config
                            try:
                                self._indic_mt = IndicModelClass(
                                    model_name_or_path="ai4bharat/indictrans2-en-ta-all-gpu",
                                    device=device,
                                )
                            except Exception:
                                # Fallback to CPU model naming if GPU fails
                                try:
                                    self._indic_mt = IndicModelClass(
                                        model_name_or_path="ai4bharat/indictrans2-en-ta-all-cpu",
                                        device=device,
                                    )
                                except Exception as model_err:
                                    logger.warning(f"Failed loading IndicTrans2 model: {model_err}")
                        if hasattr(self, '_indic_mt'):
                            try:
                                literal_translation = self._indic_mt.translate_sentence(  # type: ignore
                                    text.strip(), src_lang="ta", tgt_lang="en"
                                )
                            except Exception as tr_err:
                                logger.warning(f"IndicTrans2 translation error: {tr_err}")
                    # High-level summarization / abstraction
                    if translation_mode == 'high_level':
                        # Provide a heuristic summarization using transformers summarization pipeline if available
                        if 'pipeline' not in globals():
                            try:
                                from transformers import pipeline  # type: ignore
                                globals()['pipeline'] = pipeline
                            except Exception as t_imp:
                                logger.warning(
                                    "transformers not installed; high-level summarization skipped. "
                                    "Install with: pip install transformers sentencepiece"
                                )
                                pipeline = None  # type: ignore
                        hf_pipeline = globals().get('pipeline')
                        if hf_pipeline:
                            try:
                                if not hasattr(self, '_summary_pipe'):
                                    # Use a lightweight multilingual summarization-capable model
                                    # (Users may replace with a better fine-tuned model)
                                    self._summary_pipe = hf_pipeline(
                                        'summarization',
                                        model='facebook/bart-large-cnn'
                                    )
                                # Choose source: prefer literal English if available else whisper translation
                                source_for_summary = literal_translation or translation_text or text
                                # Truncate excessively long input for summarization model
                                if len(source_for_summary) > 4000:
                                    source_for_summary = source_for_summary[:4000]
                                summary_chunks = self._summary_pipe(
                                    source_for_summary,
                                    max_length=180,
                                    min_length=40,
                                    do_sample=False,
                                )
                                if summary_chunks and isinstance(summary_chunks, list):
                                    high_level_translation = summary_chunks[0].get('summary_text', '')
                            except Exception as sum_err:
                                logger.warning(f"High-level summarization failed: {sum_err}")
                except Exception as outer_tr_err:
                    logger.warning(f"Extended translation pipeline error: {outer_tr_err}")

            return {
                "text": text.strip(),
                "language": final_language,
                "segments": segments_count,
                "translation_text": translation_text.strip(),
                "translation_segments": translation_segments,
                "translation_performed": translation_performed,
                "literal_translation": literal_translation.strip(),
                "high_level_translation": high_level_translation.strip(),
                "translation_mode": translation_mode,
            }
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            raise # Re-raise to be caught by the server's error handling
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file {temp_file_path}: {cleanup_error}")

    def get_available_models(self) -> list[str]:
        """Returns a list of popular faster-whisper model sizes."""
        return [
            "tiny", "base", "small", "medium", "large-v2", "large-v3"
        ]

    def get_supported_languages(self) -> list[str]:
        """Returns a list of languages supported by Whisper for transcription."""
        return [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", 
            "zh", "ar", "hi", "ta", "auto" # "auto" is a special value for language detection
        ]

if __name__ == "__main__":
    # Example usage for direct testing of the WhisperSpeechToText class
    print("--- Running a simple test for WhisperSpeechToText ---")
    try:
        # Note: This block won't run when imported by server.py
        stt_service = WhisperSpeechToText("tiny")
        print(f"Current model size: {stt_service.model_size}")
        print(f"Available models: {stt_service.get_available_models()}")
        print(f"Supported languages: {stt_service.get_supported_languages()}")
        
        # To actually test transcription, you'd need a sample audio file:
        # with open("path/to/your/audio.wav", "rb") as audio_file:
        #     content = audio_file.read()
        #     result = stt_service.transcribe_audio_file(content, "audio.wav", "en")
        #     print(f"Test Transcription Result: {result}")
 
    except Exception as e:
        print(f"An error occurred during test: {e}") 