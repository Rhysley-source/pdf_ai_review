import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from routes.route import router
from routes.convert_route import router as convert_router
from document_generation.document_generator import router as document_generate_router
from document_validation.validation_router import router as validation_router # NEW IMPORT
from db_files.db import init_db, close_pool

# ---------------------------------------------------------------------------
# Logging 123
# ---------------------------------------------------------------------------
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "formatter": "standard",
            "level":     "INFO",
            "stream":    "ext://sys.stdout",
        },
        "file": {
            "class":     "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level":     "DEBUG",
            "filename":  "app.log",
            "maxBytes":  10 * 1024 * 1024,
            "backupCount": 5,
            "encoding":  "utf-8",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
})

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    await init_db()
    yield
    await close_pool()
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Error classifier — maps raw exceptions to user-friendly messages
# ---------------------------------------------------------------------------

def _classify_error(exc: Exception) -> tuple[int, str, str]:
    """
    Returns (http_status, error_code, user_message) for any exception.

    Recognises common error families so the client always gets a meaningful
    message instead of a raw Python traceback or database error string.
    """
    msg = str(exc).lower()

    # ── Database charset / encoding (MySQL error 1366) ────────────────────
    if "1366" in msg or "incorrect string value" in msg or "incorrect integer value" in msg:
        return 422, "unsupported_characters", (
            "Your request contains special characters (e.g. ₹, €, £, ©) that "
            "are not supported by the database. Please ask your administrator to "
            "set the database column charset to utf8mb4."
        )

    # ── General database / connection errors ──────────────────────────────
    if any(k in msg for k in ("operationalerror", "connection refused", "could not connect",
                               "no such table", "relation does not exist",
                               "database", "asyncpg", "pymysql", "psycopg")):
        return 503, "database_error", (
            "A database error occurred. The service may be temporarily unavailable. "
            "Please try again in a moment."
        )

    # ── Unicode / encoding errors ─────────────────────────────────────────
    if any(k in msg for k in ("unicodedecodeerror", "unicodeencodeerror",
                               "codec can't", "charmap", "utf-8", "utf8")):
        return 422, "encoding_error", (
            "The request contains characters that could not be processed. "
            "Please ensure all text is valid UTF-8."
        )

    # ── OpenAI / LLM API errors ───────────────────────────────────────────
    if any(k in msg for k in ("openai", "rate limit", "ratelimit", "quota",
                               "insufficient_quota", "model_not_found",
                               "context_length_exceeded")):
        return 503, "ai_service_error", (
            "The AI service is temporarily unavailable or has reached its limit. "
            "Please try again in a moment."
        )

    # ── Timeout ───────────────────────────────────────────────────────────
    if any(k in msg for k in ("timeout", "timed out", "deadline")):
        return 504, "timeout", (
            "The request took too long to process. Please try again with a "
            "shorter or simpler request."
        )

    # ── File / PDF errors ─────────────────────────────────────────────────
    if any(k in msg for k in ("pdf", "weasyprint", "no extractable text",
                               "blank page", "file not found")):
        return 422, "file_error", (
            "There was a problem processing the uploaded file. "
            "Please ensure it is a valid, non-encrypted PDF."
        )

    # ── Fallback ──────────────────────────────────────────────────────────
    return 500, "internal_error", (
        "An unexpected error occurred. Please try again. "
        "If the problem persists, contact support."
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PDF AI Review API",
    description="""
## PDF AI Review + Document Generation API

### PDF Analysis
- **POST /analyze** — Analyse a PDF, return overview, summary, highlights
- **POST /analyze/stream** — Same but streams results via SSE
- **POST /key-clause-extraction** — Extract key clauses by document type
- **POST /detect-risks** — Detect legal/financial risks in a document
- **POST /red-flag-scanner** — AI red flag scan: identifies dangerous/unusual contract language with ⚠ warnings
- **POST /convert/pdf-to-docx** — Convert PDF to DOCX
- **POST /validate-document** — Validate a PDF document for issues like missing fields, wrong clauses, or harmful terms

### Document Generation (HTML)
- **POST /generate-html** — Generate an HTML document of any type from a text prompt
- **POST /regenerate-html** — Modify an existing HTML document by document_id
""",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(convert_router)
app.include_router(document_generate_router)
app.include_router(validation_router) # NEW INCLUDE


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

def _build_error_response(detail) -> dict:
    """
    Converts an HTTPException detail (any type) into a structured dict
    suitable for JSONResponse. Never raises — always returns a valid dict.
    """
    if isinstance(detail, dict):
        return {
            "detail": {
                "error":   str(detail.get("error", "request_error")),
                "message": str(detail.get("message", "An error occurred.")),
            }
        }
    if isinstance(detail, str):
        return {"detail": {"error": "request_error", "message": detail}}
    # Fallback for any other type (None, list, etc.)
    return {"detail": {"error": "request_error", "message": str(detail) if detail else "An error occurred."}}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles all FastAPI / Starlette HTTPExceptions.
    Converts any detail format into a consistent flat JSON response:
      { "status": "error", "error": "...", "message": "..." }
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=_build_error_response(exc.detail),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic request validation errors — return the first meaningful message."""
    errors = exc.errors()
    first  = errors[0] if errors else {}
    field  = " → ".join(str(loc) for loc in first.get("loc", []) if loc != "body")
    msg    = first.get("msg", "Invalid request")
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error":   "validation_error",
                "message": f"{field}: {msg}" if field else msg,
            }
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for any unhandled exception.
    HTTPExceptions are delegated to http_exception_handler so they are never
    double-wrapped. All other exceptions are classified and returned as a clean
    user-friendly JSON response. Raw details are logged but never exposed.
    """
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)

    status_code, error_code, message = _classify_error(exc)
    logger.exception(
        f"[{error_code}] Unhandled exception on {request.method} {request.url.path}: {exc}"
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": {
                "error":   error_code,
                "message": message,
            }
        },
    )
