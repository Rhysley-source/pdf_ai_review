import uuid
import logging
import asyncio
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel, Field

from auth import verify_api_key
from utils.pdf_utils import extract_text_from_pdf
from feature_modules.document_chat import ingest_document, run_chat, collection_exists

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Document Chat (RAG)"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    collection_id: str = Field(..., description="ID returned by /chat/ingest")
    question:      str = Field(..., min_length=3, description="Question to ask about the document")


class IngestResponse(BaseModel):
    collection_id: str
    filename:      str
    chunks_stored: int
    message:       str


class QueryResponse(BaseModel):
    collection_id: str
    question:      str
    question_used: str
    answer:        str
    sources_used:  int
    rewrites_done: int


# ---------------------------------------------------------------------------
# POST /chat/ingest
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload a PDF and index it for RAG chat",
    description=(
        "Extracts text from the uploaded PDF, chunks it, embeds it with OpenAI, "
        "and stores the vectors in Qdrant. Returns a `collection_id` to use in `/chat/query`."
    ),
)
async def ingest(
    file: UploadFile = File(..., description="PDF file to ingest"),
    _auth = Depends(verify_api_key),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_file", "message": "Only PDF files are supported."},
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_file", "message": "Uploaded file is empty."},
        )

    # Extract text (runs OCR if needed — runs in thread pool inside extract_text_from_pdf)
    try:
        text, _pages = await asyncio.get_event_loop().run_in_executor(
            None, _extract_text_sync, pdf_bytes
        )
    except Exception as exc:
        logger.exception(f"[ingest] text extraction failed: {exc}")
        raise HTTPException(
            status_code=422,
            detail={"error": "extraction_error", "message": f"Could not extract text: {exc}"},
        )

    if not text or not text.strip():
        raise HTTPException(
            status_code=422,
            detail={"error": "no_text", "message": "No extractable text found in the PDF."},
        )

    collection_id = str(uuid.uuid4())

    try:
        chunks_stored = await asyncio.get_event_loop().run_in_executor(
            None, ingest_document, text, collection_id
        )
    except Exception as exc:
        logger.exception(f"[ingest] Qdrant ingestion failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail={"error": "ingest_error", "message": f"Failed to index document: {exc}"},
        )

    logger.info(f"[ingest] '{file.filename}' → collection '{collection_id}' ({chunks_stored} chunks)")
    return IngestResponse(
        collection_id=collection_id,
        filename=file.filename,
        chunks_stored=chunks_stored,
        message="Document indexed successfully. Use collection_id to query.",
    )


def _extract_text_sync(pdf_bytes: bytes):
    """Synchronous wrapper for extract_text_from_pdf used in executor."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        from utils.pdf_utils import extract_text_from_pdf as _extract
        text, pages = _extract(tmp_path)
        return text, pages
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# POST /chat/query
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about an ingested document",
    description=(
        "Runs a LangGraph RAG pipeline: retrieve relevant chunks → grade relevance → "
        "generate answer (with automatic query rewriting if needed)."
    ),
)
async def query(
    body: QueryRequest,
    _auth = Depends(verify_api_key),
):
    if not collection_exists(body.collection_id):
        raise HTTPException(
            status_code=404,
            detail={
                "error":   "collection_not_found",
                "message": f"No indexed document found for collection_id '{body.collection_id}'. "
                           "Please ingest the document first via POST /chat/ingest.",
            },
        )

    try:
        result = await run_chat(
            collection_id=body.collection_id,
            question=body.question,
        )
    except Exception as exc:
        logger.exception(f"[query] RAG pipeline failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail={"error": "rag_error", "message": f"Failed to generate answer: {exc}"},
        )

    return QueryResponse(
        collection_id=body.collection_id,
        question=body.question,
        question_used=result["question_used"],
        answer=result["answer"],
        sources_used=result["sources_used"],
        rewrites_done=result["rewrites_done"],
    )
