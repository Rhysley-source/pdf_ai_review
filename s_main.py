import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI

from routes.route import router
from routes.convert_route import router as convert_router
from document_generation.document_generator import router as document_generate_router
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

app = FastAPI(
    title="PDF AI Review API",
    description="""
## PDF AI Review + Document Generation API

### PDF Analysis
- **POST /analyze** — Analyse a PDF, return overview, summary, highlights
- **POST /analyze/stream** — Same but streams results via SSE
- **POST /key-clause-extraction** — Extract key clauses by document type
- **POST /detect-risks** — Detect legal/financial risks in a document
- **POST /convert/pdf-to-docx** — Convert PDF to DOCX

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
