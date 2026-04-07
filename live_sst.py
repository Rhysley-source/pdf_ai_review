import os
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI, APIStatusError, APIConnectionError
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("whisper_api")

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

# Directory where uploaded/recorded audio files are permanently saved
AUDIO_SAVE_DIR: Path = Path(os.getenv("AUDIO_SAVE_DIR", "saved_audio"))
AUDIO_SAVE_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS: set[str] = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a",
    ".wav", ".webm", ".ogg", ".flac",
}

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    logger.info("AsyncOpenAI client initialised")
    yield
    await app.state.openai_client.close()
    logger.info("AsyncOpenAI client closed")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Whisper Transcription API",
    description="Capture audio, save to disk, and transcribe using OpenAI Whisper",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved audio files over HTTP  →  GET /audio-files/<filename>
app.mount("/audio-files", StaticFiles(directory=str(AUDIO_SAVE_DIR)), name="audio-files")

# ── Schemas ───────────────────────────────────────────────────────────────────
class TranscriptResponse(BaseModel):
    request_id: str
    transcript: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None
    original_filename: str
    saved_filename: str
    saved_path: str
    file_size_bytes: int
    recorded_at: str

class HealthResponse(BaseModel):
    status: str
    version: str

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_openai_client(request: Request) -> AsyncOpenAI:
    return request.app.state.openai_client


async def read_and_validate(file: UploadFile) -> bytes:
    """Read file bytes and validate extension + size."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported extension '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    data = await file.read()

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed: {MAX_FILE_SIZE_MB} MB.",
        )

    return data


def build_save_filename(original_filename: str, request_id: str) -> str:
    """
    Build a collision-proof filename:
        YYYYMMDD_HHMMSS_<first-8-chars-of-uuid><ext>
    e.g.  20240315_143022_c3f1a2b3.wav
    """
    ext = Path(original_filename).suffix.lower() or ".audio"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{request_id[:8]}{ext}"


def save_audio(data: bytes, filename: str) -> Path:
    """Write raw bytes to AUDIO_SAVE_DIR and return the full path."""
    dest = AUDIO_SAVE_DIR / filename
    dest.write_bytes(data)
    logger.info("Audio saved → %s  (%d bytes)", dest, len(data))
    return dest


async def call_whisper(
    client: AsyncOpenAI,
    data: bytes,
    filename: str,
    language: Optional[str],
    prompt: Optional[str],
    request_id: str,
):
    """Send audio bytes to Whisper and return the verbose_json result."""
    ext = Path(filename).suffix.lower().lstrip(".") or "mp3"
    file_tuple = (filename, data, f"audio/{ext}")

    kwargs: dict = dict(
        model="whisper-1",
        file=file_tuple,
        response_format="verbose_json",   # returns language + duration
    )
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt

    try:
        return await client.audio.transcriptions.create(**kwargs)
    except APIConnectionError as exc:
        logger.error("[%s] OpenAI connection error: %s", request_id, exc)
        raise HTTPException(status_code=502, detail="Could not reach OpenAI. Try again later.")
    except APIStatusError as exc:
        logger.error("[%s] OpenAI API error %s: %s", request_id, exc.status_code, exc.message)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception:
        logger.exception("[%s] Unexpected transcription error", request_id)
        raise HTTPException(status_code=500, detail="Transcription failed.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post(
    "/record-and-transcribe",
    response_model=TranscriptResponse,
    summary="Save audio to disk AND return transcript",
    tags=["Transcription"],
)
@limiter.limit("20/minute")
async def record_and_transcribe(
    request: Request,
    file: UploadFile = File(..., description="Audio captured from mic or any audio file"),
    language: Optional[str] = Query(
        default=None,
        description="ISO-639-1 language code (e.g. 'en', 'hi'). Auto-detected if omitted.",
        min_length=2,
        max_length=10,
    ),
    prompt: Optional[str] = Query(
        default=None,
        max_length=500,
        description="Optional hint (domain terms / proper nouns) to improve accuracy.",
    ),
    client: AsyncOpenAI = Depends(get_openai_client),
):
    request_id = str(uuid.uuid4())
    recorded_at = datetime.now().isoformat()
    logger.info("[%s] record-and-transcribe | file=%s", request_id, file.filename)

    # 1. Read & validate
    audio_bytes = await read_and_validate(file)

    # 2. Save permanently to disk
    save_name = build_save_filename(file.filename or "recording.wav", request_id)
    saved_path = save_audio(audio_bytes, save_name)

    # 3. Transcribe via Whisper
    result = await call_whisper(
        client, audio_bytes, file.filename or save_name, language, prompt, request_id
    )

    logger.info(
        "[%s] Complete | chars=%d | lang=%s | file=%s",
        request_id, len(result.text), getattr(result, "language", "?"), saved_path,
    )

    return TranscriptResponse(
        request_id=request_id,
        transcript=result.text,
        language=getattr(result, "language", None),
        duration_seconds=getattr(result, "duration", None),
        original_filename=file.filename or "unknown",
        saved_filename=save_name,
        saved_path=str(saved_path),
        file_size_bytes=len(audio_bytes),
        recorded_at=recorded_at,
    )


@app.post(
    "/transcribe",
    response_model=TranscriptResponse,
    summary="Transcribe audio without saving to disk",
    tags=["Transcription"],
)
@limiter.limit("20/minute")
async def transcribe_only(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Query(default=None, min_length=2, max_length=10),
    prompt: Optional[str] = Query(default=None, max_length=500),
    client: AsyncOpenAI = Depends(get_openai_client),
):
    request_id = str(uuid.uuid4())
    recorded_at = datetime.now().isoformat()
    logger.info("[%s] transcribe-only | file=%s", request_id, file.filename)

    audio_bytes = await read_and_validate(file)
    result = await call_whisper(
        client, audio_bytes, file.filename or "audio.mp3", language, prompt, request_id
    )

    return TranscriptResponse(
        request_id=request_id,
        transcript=result.text,
        language=getattr(result, "language", None),
        duration_seconds=getattr(result, "duration", None),
        original_filename=file.filename or "unknown",
        saved_filename="",           # not saved
        saved_path="",
        file_size_bytes=len(audio_bytes),
        recorded_at=recorded_at,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})