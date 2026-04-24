import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client.models import Filter, FieldCondition, MatchValue

from auth import verify_api_key
from rag.agent import chat, stream_chat
from rag.ingestion import delete_document, embed_and_store
from rag.memory import SessionMemory
from rag.qdrant_client_setup import COLLECTION_DOCUMENTS, get_async_client

router = APIRouter(prefix="/rag", tags=["RAG"])


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_stored: int
    total_chars: int
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    document_id: Optional[str] = Form(None),
    _: str = Depends(verify_api_key),
):
    """Upload a PDF and ingest it into Qdrant for RAG Q&A."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50 MB.")

    try:
        result = await embed_and_store(
            file_bytes=file_bytes,
            filename=file.filename,
            user_id=user_id,
            document_id=document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return UploadResponse(**result, message="Document ingested successfully.")


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: str
    document_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    document_id: Optional[str]


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    _: str = Depends(verify_api_key),
):
    """Single-turn agent chat. Pass session_id to continue a conversation."""
    session_id = req.session_id or str(uuid.uuid4())
    result = await chat(
        user_message=req.message,
        session_id=session_id,
        user_id=req.user_id,
        document_id=req.document_id,
    )
    return ChatResponse(**result)


@router.post("/chat/stream")
async def stream_chat_endpoint(
    req: ChatRequest,
    _: str = Depends(verify_api_key),
):
    """Streaming SSE chat — tokens arrive in real time."""
    session_id = req.session_id or str(uuid.uuid4())

    async def generate():
        yield f"data: {{\"session_id\": \"{session_id}\"}}\n\n"
        async for token in stream_chat(
            user_message=req.message,
            session_id=session_id,
            user_id=req.user_id,
            document_id=req.document_id,
        ):
            safe_token = token.replace("\n", "\\n")
            yield f"data: {safe_token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Direct Tool Endpoints ─────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    document_id: str
    document_type: Optional[str] = "contract"


class CompareRequest(BaseModel):
    document_id_1: str
    document_id_2: str


@router.post("/analyze/risks")
async def analyze_risks(req: AnalyzeRequest, _: str = Depends(verify_api_key)):
    """Run risk analysis directly without the chat agent."""
    from rag.tools import risk_analysis_tool
    result = await risk_analysis_tool.ainvoke(
        {"document_id": req.document_id, "document_type": req.document_type or "contract"}
    )
    return {"result": result}


@router.post("/analyze/key-clauses")
async def analyze_key_clauses(req: AnalyzeRequest, _: str = Depends(verify_api_key)):
    """Extract key clauses directly."""
    from rag.tools import key_clause_extraction_tool
    result = await key_clause_extraction_tool.ainvoke(
        {"document_id": req.document_id, "document_type": req.document_type or ""}
    )
    return {"result": result}


@router.post("/analyze/red-flags")
async def analyze_red_flags(req: AnalyzeRequest, _: str = Depends(verify_api_key)):
    """Scan for red flags directly."""
    from rag.tools import red_flag_scanner_tool
    result = await red_flag_scanner_tool.ainvoke({"document_id": req.document_id})
    return {"result": result}


@router.post("/analyze/compare")
async def analyze_compare(req: CompareRequest, _: str = Depends(verify_api_key)):
    """Compare two documents directly."""
    from rag.tools import document_compare_tool
    result = await document_compare_tool.ainvoke(
        {"document_id_1": req.document_id_1, "document_id_2": req.document_id_2}
    )
    return {"result": result}


# ── Document Management ───────────────────────────────────────────────────────

@router.get("/documents/{user_id}")
async def list_documents(user_id: str, _: str = Depends(verify_api_key)):
    """List all documents uploaded by a user."""
    client = get_async_client()
    try:
        results, _ = await client.scroll(
            collection_name=COLLECTION_DOCUMENTS,
            scroll_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=500,
            with_payload=["document_id", "filename", "uploaded_at", "chunk_index"],
            with_vectors=False,
        )
    finally:
        await client.close()

    seen: dict = {}
    for point in results:
        p = point.payload
        did = p.get("document_id")
        if did and did not in seen and p.get("chunk_index", 0) == 0:
            seen[did] = {
                "document_id": did,
                "filename": p.get("filename"),
                "uploaded_at": p.get("uploaded_at"),
            }
    return {"documents": list(seen.values())}


@router.delete("/document/{document_id}")
async def remove_document(document_id: str, _: str = Depends(verify_api_key)):
    """Delete a document and all its vectors from Qdrant."""
    await delete_document(document_id)
    return {"message": f"Document '{document_id}' deleted successfully."}


# ── Session Management ────────────────────────────────────────────────────────

@router.delete("/session/{session_id}")
async def clear_session(session_id: str, _: str = Depends(verify_api_key)):
    """Clear all chat history for a session."""
    memory = SessionMemory(session_id)
    await memory.clear()
    return {"message": f"Session '{session_id}' cleared successfully."}
