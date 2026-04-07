"""
t_document_route.py

FastAPI router for document generation.
All requests are logged to the document_requests table in PostgreSQL.

Endpoints:
  GET  /documents/types                 List all supported document types
  GET  /documents/stats                 Aggregate stats from DB
  GET  /documents/recent                Recent document requests from DB
  POST /documents/generate              Generate → JSON (text + base64 docx)
  POST /documents/generate/download     Generate → DOCX file download
  POST /documents/generate/stream       Generate → SSE token stream
  POST /documents/extract-fields        Extract fields only (no generation)
"""

import json
import time
import uuid
import base64
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

# ── Database ─────────────────────────────────────────────────────────────────
from db_files.db import (
    log_document_request,
    get_document_stats,
    get_recent_documents,
)

# ── Core generation logic ─────────────────────────────────────────────────────
from feature_modules.document_generator import (
    generate_document,
    SUPPORTED_DOCUMENT_TYPES,
    _SCHEMAS,
    _extract_fields,
    _get_missing_fields,
    _render_docx,
    _api_kwargs,
    _get_client,
)

# ── Prompts (moved to t_prompts.py) ──────────────────────────────────────────
from feature_modules.prompts import GENERATION_PROMPTS

# ── Intent classification (moved to t_intent.py) ─────────────────────────────
from feature_modules.intent import classify_intent, resolve_document_type

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Document Generation"])


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class DocumentGenerateRequest(BaseModel):
    document_type: Optional[str] = Field(
        default=None,
        description=(
            "Type of document to generate. "
            "If omitted, automatically detected from user_query. "
            "Supported: nda, job_offer, freelancer_agreement, service_agreement, "
            "consulting_agreement, lease_agreement, employment_contract"
        ),
        examples=["nda", "employment_contract", None],
    )
    user_query: str = Field(
        ...,
        min_length=20,
        description=(
            "Free-text description of the document you need under Indian law. "
            "Include party names, dates, amounts in INR, state jurisdiction, "
            "and any special clauses."
        ),
        examples=[
            "NDA between TechCorp Pvt Ltd (Mumbai) and Rajan Shah (contractor). "
            "Purpose: sharing AI product roadmap. Duration 2 years. "
            "Governed by Maharashtra law. Effective 1st April 2026.",
        ],
    )


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# GET /documents/types
# ---------------------------------------------------------------------------

@router.get("/types", summary="List all supported document types")
async def list_document_types() -> dict:
    types = []
    for slug, display_name in SUPPORTED_DOCUMENT_TYPES.items():
        schema = _SCHEMAS.get(slug, {"required": [], "optional": []})
        types.append({
            "slug":            slug,
            "display_name":    display_name,
            "required_fields": schema.get("required", []),
            "optional_fields": schema.get("optional", []),
        })
    return {"status": "success", "total": len(types), "document_types": types}


# ---------------------------------------------------------------------------
# GET /documents/stats
# ---------------------------------------------------------------------------

@router.get("/stats", summary="Aggregate document generation statistics")
async def document_stats() -> dict:
    stats = await get_document_stats()
    return {"status": "success", **stats}


# ---------------------------------------------------------------------------
# GET /documents/recent
# ---------------------------------------------------------------------------

@router.get("/recent", summary="Recent document generation requests")
async def recent_documents(limit: int = 20) -> dict:
    limit = min(limit, 100)
    rows  = await get_recent_documents(limit)
    return {"status": "success", "total": len(rows), "documents": rows}


# ---------------------------------------------------------------------------
# POST /documents/extract-fields
# ---------------------------------------------------------------------------

@router.post("/extract-fields", summary="Extract fields without generating document")
async def extract_document_fields(request: DocumentGenerateRequest) -> dict:
    request_id = str(uuid.uuid4())[:8]

    if request.document_type:
        doc_type = resolve_document_type(request.document_type)
        if not doc_type:
            raise HTTPException(status_code=400, detail={
                "error":           "Unsupported document type",
                "provided":        request.document_type,
                "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
            })
        detected = False
    else:
        doc_type = await classify_intent(request.user_query)
        detected = True
        if not doc_type:
            raise HTTPException(status_code=422, detail={
                "error": "Could not determine document type.",
                "hint":  "Please specify document_type explicitly.",
                "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
            })

    try:
        fields  = await _extract_fields(doc_type, request.user_query)
        missing = _get_missing_fields(doc_type, fields)
    except Exception as e:
        logger.exception(f"[{request_id}] field extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Field extraction failed: {str(e)}")

    return {
        "status":            "missing_fields" if missing else "success",
        "document_type":     doc_type,
        "document_name":     SUPPORTED_DOCUMENT_TYPES[doc_type],
        "type_was_detected": detected,
        "fields":            fields,
        "missing_fields":    missing,
        "message": (
            "All required fields found. Ready to generate."
            if not missing else
            f"Missing {len(missing)} required field(s): {', '.join(missing)}. "
            "Add these details to your query for a complete document."
        ),
    }


# ---------------------------------------------------------------------------
# POST /documents/generate — JSON response + DB log
# ---------------------------------------------------------------------------

@router.post("/generate", summary="Generate document — returns JSON with text and DOCX")
async def generate_document_endpoint(request: DocumentGenerateRequest) -> dict:
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()
    error_msg  = None

    logger.info(f"[{request_id}] ── DOC GENERATE — type='{request.document_type or 'auto'}'")

    try:
        result = await generate_document(request.document_type, request.user_query)
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"[{request_id}] error: {e}")
        await log_document_request(
            request_id        = request_id,
            document_type     = request.document_type or "unknown",
            document_name     = "",
            user_query        = request.user_query,
            status            = "error",
            completion_time_s = time.perf_counter() - t_start,
            endpoint          = "/documents/generate",
            error_message     = error_msg,
        )
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(e)}")

    if result["status"] == "unknown_type":
        await log_document_request(
            request_id        = request_id,
            document_type     = request.document_type or "unknown",
            document_name     = "",
            user_query        = request.user_query,
            status            = "unknown_type",
            completion_time_s = time.perf_counter() - t_start,
            endpoint          = "/documents/generate",
            error_message     = result.get("message"),
        )
        raise HTTPException(status_code=422, detail={
            "error":           result.get("message"),
            "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
        })

    elapsed    = time.perf_counter() - t_start
    docx_bytes = result.pop("docx_bytes", b"")
    docx_b64   = base64.b64encode(docx_bytes).decode("utf-8") if docx_bytes else ""
    detected   = request.document_type is None

    await log_document_request(
        request_id        = request_id,
        document_type     = result["document_type"],
        document_name     = result["document_name"],
        user_query        = request.user_query,
        status            = result["status"],
        type_was_detected = detected,
        missing_fields    = result.get("missing_fields", []),
        word_count        = result.get("word_count", 0),
        docx_size_bytes   = len(docx_bytes),
        input_tokens      = result.get("input_tokens", 0),
        output_tokens     = result.get("output_tokens", 0),
        completion_time_s = elapsed,
        endpoint          = "/documents/generate",
        fields            = result.get("fields", {}),
    )

    logger.info(
        f"[{request_id}] ── COMPLETE — {elapsed:.2f}s "
        f"type={result['document_type']} words={result.get('word_count', 0)}"
    )

    return {
        "status":             result["status"],
        "document_type":      result["document_type"],
        "document_name":      result["document_name"],
        "type_was_detected":  detected,
        "fields":             result["fields"],
        "missing_fields":     result["missing_fields"],
        "document":           result["document"],
        "docx_base64":        docx_b64,
        "word_count":         result["word_count"],
        "input_tokens":       result.get("input_tokens", 0),
        "output_tokens":      result.get("output_tokens", 0),
        "generation_time_s":  round(elapsed, 2),
        "request_id":         request_id,
    }


# ---------------------------------------------------------------------------
# POST /documents/generate/download — DOCX file + DB log
# ---------------------------------------------------------------------------

@router.post(
    "/generate/download",
    summary="Generate document — returns DOCX file download",
    response_class=Response,
)
async def generate_document_download(request: DocumentGenerateRequest):
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── DOC DOWNLOAD — type='{request.document_type or 'auto'}'")

    try:
        result = await generate_document(request.document_type, request.user_query)
    except Exception as e:
        logger.exception(f"[{request_id}] error: {e}")
        await log_document_request(
            request_id        = request_id,
            document_type     = request.document_type or "unknown",
            document_name     = "",
            user_query        = request.user_query,
            status            = "error",
            completion_time_s = time.perf_counter() - t_start,
            endpoint          = "/documents/generate/download",
            error_message     = str(e),
        )
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(e)}")

    if result["status"] == "unknown_type":
        await log_document_request(
            request_id        = request_id,
            document_type     = request.document_type or "unknown",
            document_name     = "",
            user_query        = request.user_query,
            status            = "unknown_type",
            completion_time_s = time.perf_counter() - t_start,
            endpoint          = "/documents/generate/download",
            error_message     = result.get("message"),
        )
        raise HTTPException(status_code=422, detail={
            "error":           result.get("message"),
            "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
        })

    elapsed    = time.perf_counter() - t_start
    docx_bytes = result.get("docx_bytes", b"")
    doc_type   = result["document_type"]
    missing    = result.get("missing_fields", [])
    detected   = request.document_type is None
    filename   = f"{doc_type}_{request_id}.docx"

    await log_document_request(
        request_id        = request_id,
        document_type     = doc_type,
        document_name     = result["document_name"],
        user_query        = request.user_query,
        status            = result["status"],
        type_was_detected = detected,
        missing_fields    = missing,
        word_count        = result.get("word_count", 0),
        docx_size_bytes   = len(docx_bytes),
        input_tokens      = result.get("input_tokens", 0),
        output_tokens     = result.get("output_tokens", 0),
        completion_time_s = elapsed,
        endpoint          = "/documents/generate/download",
        fields            = result.get("fields", {}),
    )

    logger.info(
        f"[{request_id}] ── DOWNLOAD COMPLETE — {elapsed:.2f}s "
        f"type={doc_type} words={result.get('word_count', 0)} "
        f"docx={len(docx_bytes):,} bytes"
    )

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Document-Type":     doc_type,
            "X-Document-Name":     result["document_name"],
            "X-Type-Detected":     str(detected).lower(),
            "X-Word-Count":        str(result.get("word_count", 0)),
            "X-Missing-Fields":    ",".join(missing) if missing else "",
            "X-Generation-Time":   str(round(elapsed, 2)),
            "X-Status":            result["status"],
            "X-Request-Id":        request_id,
        }
    )


# ---------------------------------------------------------------------------
# POST /documents/generate/stream — SSE stream + DB log on completion
# ---------------------------------------------------------------------------

@router.post("/generate/stream", summary="Generate document with real-time token streaming")
async def generate_document_stream(request: DocumentGenerateRequest):
    """
    Streams document generation via SSE.
    Logs to document_requests table when generation completes.

    Events: status, detected, fields, token, done, error
    """
    import os

    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── DOC STREAM — type='{request.document_type or 'auto'}'")

    async def _generate():
        doc_type_final = request.document_type or "unknown"
        doc_name_final = ""
        status_final   = "error"
        missing_final: list = []
        fields_final:  dict = {}
        word_count     = 0
        docx_size      = 0
        in_tok         = 0
        out_tok        = 0
        detected       = False
        error_msg      = None

        try:
            # Step 0: Resolve or detect type
            if request.document_type:
                doc_type = resolve_document_type(request.document_type)
                if not doc_type:
                    yield _sse("error", {
                        "message":         f"Unsupported document type: '{request.document_type}'",
                        "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
                    })
                    return
            else:
                yield _sse("status", {
                    "step":    "classifying",
                    "message": "Detecting document type...",
                })
                doc_type = await classify_intent(request.user_query)
                detected = True
                if not doc_type:
                    yield _sse("error", {
                        "message":         "Could not determine document type.",
                        "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
                    })
                    return
                yield _sse("detected", {
                    "document_type": doc_type,
                    "document_name": SUPPORTED_DOCUMENT_TYPES[doc_type],
                    "message":       f"Detected: {SUPPORTED_DOCUMENT_TYPES[doc_type]}",
                })

            doc_type_final = doc_type
            doc_name_final = SUPPORTED_DOCUMENT_TYPES[doc_type]

            # Step 1: Extract fields
            yield _sse("status", {
                "step":    "extracting_fields",
                "message": f"Extracting details for {doc_name_final}...",
            })
            fields  = await _extract_fields(doc_type, request.user_query)
            missing = _get_missing_fields(doc_type, fields)
            fields_final  = fields
            missing_final = missing

            yield _sse("fields", {
                "fields":            fields,
                "missing_fields":    missing,
                "document_type":     doc_type,
                "document_name":     doc_name_final,
                "type_was_detected": detected,
            })

            # Step 2: Stream document generation
            gen_prompt  = GENERATION_PROMPTS[doc_type]   # ← from t_prompts.py
            field_lines = "\n".join(
                f"  {k.replace('_', ' ').title()}: {v}"
                for k, v in fields.items()
                if v and v != "Not Specified"
            )

            system = gen_prompt
            user   = f"""Use these details to generate the complete {doc_name_final} under Indian law:

{field_lines if field_lines else "(Use standard Indian law template values)"}

Additional context:
\"\"\"{request.user_query}\"\"\"

Write the complete {doc_name_final} now. Start directly with the document title."""

            yield _sse("status", {
                "step":    "generating_document",
                "message": f"Drafting {doc_name_final}...",
            })

            client = _get_client()
            kwargs = _api_kwargs(max_tokens=32000, use_json=False)
            kwargs["messages"]       = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
            kwargs["stream"]         = True
            kwargs["stream_options"] = {"include_usage": True}

            full_document = ""

            async with await client.chat.completions.create(**kwargs) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta          = chunk.choices[0].delta.content
                        full_document += delta
                        yield _sse("token", {"delta": delta})
                    if chunk.usage:
                        in_tok  = chunk.usage.prompt_tokens
                        out_tok = chunk.usage.completion_tokens

            word_count = len(full_document.split())

            # Step 3: Render DOCX
            yield _sse("status", {"step": "rendering_docx", "message": "Rendering DOCX..."})
            docx_bytes = _render_docx(full_document, doc_name_final)
            docx_b64   = base64.b64encode(docx_bytes).decode("utf-8")
            docx_size  = len(docx_bytes)

            status_final = "missing_fields" if missing else "success"
            elapsed      = time.perf_counter() - t_start

            logger.info(
                f"[{request_id}] ── STREAM COMPLETE — {elapsed:.2f}s "
                f"type={doc_type} words={word_count} "
                f"tokens={in_tok}in/{out_tok}out"
            )

            yield _sse("done", {
                "status":             status_final,
                "document_type":      doc_type,
                "document_name":      doc_name_final,
                "type_was_detected":  detected,
                "missing_fields":     missing,
                "word_count":         word_count,
                "docx_base64":        docx_b64,
                "docx_size_bytes":    docx_size,
                "generation_time_s":  round(elapsed, 2),
                "input_tokens":       in_tok,
                "output_tokens":      out_tok,
                "total_tokens":       in_tok + out_tok,
                "request_id":         request_id,
            })

        except Exception as e:
            error_msg    = str(e)
            status_final = "error"
            logger.exception(f"[{request_id}] stream error: {e}")
            yield _sse("error", {"message": f"Document generation failed: {str(e)}"})

        finally:
            elapsed = time.perf_counter() - t_start
            await log_document_request(
                request_id        = request_id,
                document_type     = doc_type_final,
                document_name     = doc_name_final,
                user_query        = request.user_query,
                status            = status_final,
                type_was_detected = detected,
                missing_fields    = missing_final,
                word_count        = word_count,
                docx_size_bytes   = docx_size,
                input_tokens      = in_tok,
                output_tokens     = out_tok,
                completion_time_s = elapsed,
                endpoint          = "/documents/generate/stream",
                error_message     = error_msg,
                fields            = fields_final,
            )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )