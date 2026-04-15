"""
chat_route.py

RAG-based document chat API.

Endpoints:
  POST   /chat/upload           — Upload PDF → build FAISS index → return session_id
  POST   /chat/message          — Send a message → RAG answer
  GET    /chat/{session_id}     — Get session metadata
  DELETE /chat/{session_id}/history  — Clear conversation history (keep index)
  DELETE /chat/{session_id}     — Delete session entirely

Usage flow:
  1. POST /chat/upload  { file: <pdf> }
     → { session_id, pages_indexed, total_pages, processing_time_s }

  2. POST /chat/message  { session_id: "...", message: "What are the payment terms?" }
     → { session_id, answer, history_length }

  3. Repeat step 2 — history is maintained server-side per session.

  4. DELETE /chat/{session_id}  to free memory when done.
"""

import logging
import os
import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from feature_modules.key_clause_extraction import extract_text_from_upload
from feature_modules.rag_chat import (
    build_index,
    chat,
    clear_history,
    delete_session,
    get_session_info,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Document Chat (RAG)"])


# ---------------------------------------------------------------------------
# POST /chat/upload
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF and build a FAISS vector index for it.

    - Accepts native, scanned (OCR), and mixed PDFs.
    - Text extraction and OCR run in a thread pool — event loop is not blocked.
    - Embeddings are created with OpenAI text-embedding-3-small.
    - Returns a `session_id` to use in /chat/message calls.
    """
    t0 = time.perf_counter()

    # Extract text — reuses the same OCR pipeline as other routes
    text, pages_read, total_pages, request_id, _, file_path = (
        await extract_text_from_upload(file, endpoint="/chat/upload")
    )

    try:
        session_id = await build_index(text)
    except Exception as e:
        logger.exception(f"[{request_id}] FAISS index build failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to index document.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[{request_id}] /chat/upload — session={session_id[:8]} "
        f"pages={pages_read} {elapsed:.2f}s"
    )

    return {
        "session_id":        session_id,
        "pages_indexed":     pages_read,
        "total_pages":       total_pages,
        "processing_time_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# POST /chat/message
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


@router.post("/message")
async def send_message(body: ChatRequest):
    """
    Send a message to the indexed document and receive a grounded answer.

    The LangGraph pipeline:
      1. Embeds the question with OpenAI text-embedding-3-small.
      2. Retrieves the top-5 most relevant document chunks via FAISS cosine search.
      3. Calls the LLM (configured MODEL_NAME) with the retrieved context and
         the last 6 turns of conversation history.
      4. Returns the answer and updates server-side history.

    The LLM is instructed to answer ONLY from document context — it will
    clearly say so if the answer is not found.
    """
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = await chat(body.session_id, body.message.strip())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"[chat] message error: {e}")
        raise HTTPException(status_code=500, detail="Chat request failed. Please try again.")

    return {
        "session_id":     body.session_id,
        "answer":         result["answer"],
        "history_length": result["history_length"],
    }


# ---------------------------------------------------------------------------
# GET /chat/{session_id}  — session metadata
# ---------------------------------------------------------------------------

@router.get("/{session_id}")
async def get_session(session_id: str):
    """
    Return metadata for an active session.
    """
    info = get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"session_id": session_id, **info}


# ---------------------------------------------------------------------------
# DELETE /chat/{session_id}/history  — clear history, keep index
# ---------------------------------------------------------------------------

@router.delete("/{session_id}/history")
async def clear_chat_history(session_id: str):
    """
    Clear the conversation history for a session.
    The FAISS index is preserved — you can continue chatting from a clean slate.
    """
    cleared = clear_history(session_id)
    if not cleared:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"session_id": session_id, "status": "history_cleared"}


# ---------------------------------------------------------------------------
# DELETE /chat/{session_id}  — delete session + free memory
# ---------------------------------------------------------------------------

@router.delete("/{session_id}")
async def delete_chat_session(session_id: str):
    """
    Delete a session and free its FAISS index from memory.
    Call this when the user is done with the document.
    """
    removed = delete_session(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"session_id": session_id, "status": "deleted"}
