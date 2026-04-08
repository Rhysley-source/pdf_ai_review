"""
Document Q&A Routes
===================
POST /document-qa/index       — Upload PDF → returns session_id (indexes in FAISS)
POST /document-qa/ask         — Ask a question about the indexed document (RAG)
POST /document-qa/analyze     — Run all 3 AI tools on the indexed document
DELETE /document-qa/session/{session_id} — Delete an indexed session
GET  /document-qa/sessions    — List active sessions (debug)
"""

import json
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from feature_modules.key_clause_extraction import extract_text_from_upload
from feature_modules.rag_qa import (
    answer_question,
    delete_session,
    get_session,
    index_document,
    list_sessions,
)
from feature_modules.tools import (
    detect_risks_tool,
    key_clause_extraction_tool,
    red_flag_scanner_tool,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document-qa", tags=["Document Q&A (RAG)"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    session_id: str
    question: str


class AnalyzeRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# POST /document-qa/index
# ---------------------------------------------------------------------------

@router.post("/index")
async def index_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and index it in FAISS for Q&A.

    Returns a `session_id` that must be passed to `/ask` and `/analyze`.
    The index lives in memory — restart the server and sessions are lost.
    """
    text, pages_to_read, total_pages, request_id, t_start, file_path = (
        await extract_text_from_upload(file, endpoint="/document-qa/index")
    )

    session_id = await index_document(text)

    logger.info(
        f"[rag_route/index] request_id={request_id} session_id={session_id} "
        f"pages={total_pages} chars={len(text)}"
    )

    return {
        "status": "success",
        "session_id": session_id,
        "filename": file.filename,
        "total_pages": total_pages,
        "pages_indexed": pages_to_read,
        "message": "Document indexed successfully. Use session_id for /ask and /analyze.",
    }


# ---------------------------------------------------------------------------
# POST /document-qa/ask
# ---------------------------------------------------------------------------

@router.post("/ask")
async def ask_question(body: AskRequest):
    """
    Ask a natural-language question about a previously indexed PDF.

    The RAG pipeline (LangGraph):
      1. retrieve  — FAISS similarity search (top-5 chunks)
      2. generate  — LLM answers using ONLY retrieved context

    Returns the answer plus the source excerpts used to generate it.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = await answer_question(body.session_id, body.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"[rag_route/ask] session={body.session_id} error: {e}")
        raise HTTPException(status_code=500, detail="RAG pipeline failed.")

    return {"status": "success", **result}


# ---------------------------------------------------------------------------
# POST /document-qa/analyze
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze_document(body: AnalyzeRequest):
    """
    Run all three AI analysis tools on a previously indexed document:
      - key_clause_extraction  — structured clause extraction by document type
      - detect_risks           — risk score + detected risks with mitigation
      - red_flag_scanner       — checklist-based dangerous / unusual / missing flags

    Calls each tool in sequence using the full document text stored in the session.
    """
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{body.session_id}' not found. Upload the PDF first via /document-qa/index.",
        )

    text = session["full_text"]

    try:
        logger.info(f"[rag_route/analyze] session={body.session_id} running 3 tools")

        key_clauses_raw = await key_clause_extraction_tool.ainvoke({"text": text})
        risks_raw       = await detect_risks_tool.ainvoke({"text": text})
        red_flags_raw   = await red_flag_scanner_tool.ainvoke({"text": text})

        key_clauses = json.loads(key_clauses_raw)
        risks       = json.loads(risks_raw)
        red_flags   = json.loads(red_flags_raw)

    except Exception as e:
        logger.exception(f"[rag_route/analyze] session={body.session_id} error: {e}")
        raise HTTPException(status_code=500, detail="Document analysis failed.")

    return {
        "status": "success",
        "session_id": body.session_id,
        "key_clause_extraction": key_clauses,
        "risk_detection": risks,
        "red_flag_scan": red_flags,
    }


# ---------------------------------------------------------------------------
# DELETE /document-qa/session/{session_id}
# ---------------------------------------------------------------------------

@router.delete("/session/{session_id}")
async def delete_qa_session(session_id: str):
    """Delete an indexed document session to free memory."""
    removed = delete_session(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "success", "message": f"Session '{session_id}' deleted."}


# ---------------------------------------------------------------------------
# GET /document-qa/sessions
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def get_active_sessions():
    """List all active indexed sessions (session_id + chunk count)."""
    sessions = list_sessions()
    return {
        "status": "success",
        "active_sessions": len(sessions),
        "sessions": sessions,
    }
