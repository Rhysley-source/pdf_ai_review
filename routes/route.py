from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Depends
from fastapi.responses import StreamingResponse
import os
import io
import uuid
import json
import time
import asyncio
import logging
from functools import partial

from utils.pdf_utils import load_pdf, get_page_count, all_pages_blank
from llm_model.ai_model import generate_analysis, generate_analysis_stream, transcribe_audio
from utils.json_utils import extract_json
from db_files.db import log_request, log_comparison_request
from feature_modules.key_clause_extraction import _classify_document, extract_text_from_upload
from db_files.db import log_request
from feature_modules.key_clause_extraction import extract_key_clauses, extract_text_from_upload
from feature_modules.risk_detection import analyze_document_risks
from feature_modules.red_flag_scanner import scan_red_flags
from auth import verify_api_key

# for comparison endpoint
from feature_modules.clause_extraction import extract_clauses
from feature_modules.document_comparison import compare_documents

logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER     = os.path.join(BASE_DIR, "temp")
MAX_PDF_PAGES     = None
_BLANK_PDF_RESULT = {"overview": "", "summary": "", "highlights": []}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Async wrapper for load_pdf
#
# PaddleOCR-VL's .predict() is a blocking synchronous call that can take
# 7-20s per page on GPU. Running it directly on the asyncio event loop
# freezes the entire server for that duration — no other requests are
# served, no timeouts fire, health checks fail.
#
# Solution: run load_pdf in the default ThreadPoolExecutor so the event
# loop remains free to handle other work while OCR runs in a thread.
# ---------------------------------------------------------------------------
async def _load_pdf_async(file_path: str, max_pages: int | None = None):
    """
    Non-blocking wrapper around load_pdf.
    Runs the synchronous PDF extraction + OCR in a thread pool so the
    asyncio event loop is never blocked.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,                              # default ThreadPoolExecutor
        partial(load_pdf, file_path, max_pages)
    )


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# POST /analyze — full JSON response
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights"),
):
    """
    Analyse a PDF and return the full result as a single JSON response.
    Every request is logged to PostgreSQL (pdf_requests table).
    PDF extraction (including OCR) runs in a thread pool — event loop never blocked.
    """
    request_id    = str(uuid.uuid4())[:8]
    t_start       = time.perf_counter()
    total_in_tok  = 0
    total_out_tok = 0
    total_pages   = 0
    pages_to_read = 0
    was_truncated = False
    pdf_size      = 0
    status        = "success"
    error_msg     = None

    logger.info(f"[{request_id}] ── NEW REQUEST ──────────────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        content  = await file.read()
        pdf_size = len(content)
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[{request_id}] Step 1/5 — saved {pdf_size:,} bytes → '{safe_name}'")

        total_pages   = get_page_count(file_path)
        pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
        was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES
        logger.info(
            f"[{request_id}] Step 2/5 — pages={total_pages} analysing={pages_to_read} "
            f"{'(TRUNCATED)' if was_truncated else '(all pages)'}"
        )

        try:
            t_extract = time.perf_counter()
            # ── Non-blocking: OCR runs in thread pool ──────────────────────
            pages = await _load_pdf_async(file_path, pages_to_read)
            logger.info(
                f"[{request_id}] Step 3/5 — extracted {len(pages)} page(s) "
                f"({time.perf_counter()-t_extract:.2f}s)"
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        if all_pages_blank(pages):
            logger.warning(f"[{request_id}] blank PDF — skipping inference")
            status = "blank_pdf"
            result = dict(_BLANK_PDF_RESULT)
            result["blank_pdf"] = True
            if was_truncated:
                result.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)
            if analysis_type == 1: return {"overview": ""}
            if analysis_type == 2: return {"summary": ""}
            if analysis_type == 3: return {"highlights": []}
            return result

        merged_text = "\n\n".join(p.page_content for p in pages)
        logger.info(f"[{request_id}] Step 4/5 — merged {len(merged_text):,} chars")

        logger.info(f"[{request_id}] Step 5/5 — running inference")
        t_infer = time.perf_counter()
        final_output, total_in_tok, total_out_tok = await generate_analysis(merged_text)
        logger.info(f"[{request_id}] Step 5/5 — done ({time.perf_counter()-t_infer:.2f}s)")

    except HTTPException:
        status    = "error"
        error_msg = "HTTP error"
        raise
    except Exception as e:
        status    = "error"
        error_msg = str(e)
        logger.exception(f"[{request_id}] unhandled error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during PDF analysis.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] temp file deleted")

        elapsed = time.perf_counter() - t_start
        await log_request(
            request_id        = request_id,
            pdf_name          = file.filename or "unknown",
            pdf_size_bytes    = pdf_size,
            total_pages       = total_pages,
            pages_analysed    = pages_to_read,
            input_tokens      = total_in_tok,
            output_tokens     = total_out_tok,
            completion_time_s = elapsed,
            endpoint          = "/analyze",
            status            = status,
            error_message     = error_msg,
        )

    elapsed = time.perf_counter() - t_start
    logger.info(f"[{request_id}] ── COMPLETE — {elapsed:.2f}s ──────")

    if was_truncated:
        final_output.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)

    if analysis_type == 1: return {"overview":   final_output.get("overview", "")}
    if analysis_type == 2: return {"summary":    final_output.get("summary", "")}
    if analysis_type == 3: return {"highlights": final_output.get("highlights", [])}
    return final_output


# ---------------------------------------------------------------------------
# POST /key-clause-extraction
# ---------------------------------------------------------------------------

@router.post("/key-clause-extraction")
async def key_clause_extraction(
    file: UploadFile = File(...),
):
    text, pages_to_read, total_pages, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/key-clause-extraction"
    )

    status    = "success"
    error_msg = None

    try:
        logger.info(f"[{request_id}] Step 3 — extracting key clauses...")
        result = await extract_key_clauses(text)
        logger.info(
            f"[{request_id}] ── REQUEST COMPLETE — "
            f"clauses={result.get('total_clauses', 0)} "
            f"total time: {time.perf_counter() - t_start:.2f}s ──────"
        )
        return result

    except HTTPException:
        status    = "error"
        error_msg = "HTTP error"
        raise
    except Exception as e:
        status    = "error"
        error_msg = str(e)
        logger.exception(f"[{request_id}] Unhandled error during processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during key clause extraction.")
    finally:
        elapsed = time.perf_counter() - t_start
        await log_request(
            request_id        = request_id,
            pdf_name          = file.filename or "unknown",
            total_pages       = total_pages,
            pages_analysed    = pages_to_read,
            completion_time_s = elapsed,
            endpoint          = "/key-clause-extraction",
            status            = status,
            error_message     = error_msg,
        )
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] Temp file deleted: '{file_path}'")


# ---------------------------------------------------------------------------
# POST /detect-risks
# ---------------------------------------------------------------------------

@router.post("/detect-risks")
async def detect_risks(
    file: UploadFile = File(...),
):
    """AI Risk Detection Endpoint."""
    text, pages_to_read, total_pages, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/detect-risks"
    )

    status    = "success"
    error_msg = None

    try:
        logger.info(f"[{request_id}] Starting Risk Detection...")
        result = await analyze_document_risks(text)
        return result

    except Exception as e:
        status    = "error"
        error_msg = str(e)
        logger.exception(f"[{request_id}] Risk Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during risk detection.")
    finally:
        elapsed = time.perf_counter() - t_start
        await log_request(
            request_id        = request_id,
            pdf_name          = file.filename or "unknown",
            total_pages       = total_pages,
            pages_analysed    = pages_to_read,
            completion_time_s = elapsed,
            endpoint          = "/detect-risks",
            status            = status,
            error_message     = error_msg,
        )
        if os.path.exists(file_path):
            os.remove(file_path)


# ---------------------------------------------------------------------------
# POST /red-flag-scanner
# ---------------------------------------------------------------------------

@router.post("/red-flag-scanner")
async def red_flag_scanner(
    file: UploadFile = File(...),
):
    """
    AI Red Flag Scanner — identifies dangerous or unusual contract language.

    Returns structured warnings such as:
      ⚠ Unlimited liability clause detected
      ⚠ This contract does not include a termination clause

    Each flag includes severity, the exact clause excerpt, why it's dangerous,
    and a concrete recommendation for negotiation or fix.
    """
    text, pages_to_read, total_pages, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/red-flag-scanner"
    )

    status    = "success"
    error_msg = None

    try:
        logger.info(f"[{request_id}] Starting Red Flag Scan...")
        result = await scan_red_flags(text)

        flags    = result.get("detected_flags", [])
        elapsed  = time.perf_counter() - t_start
        logger.info(
            f"[{request_id}] ── RED FLAG SCAN COMPLETE — {elapsed:.2f}s "
            f"flags={len(flags)} risk={result.get('overall_risk_level', 'Unknown')}"
        )
        return {
            "status":             "success",
            "overall_risk_level": result.get("overall_risk_level", "Low"),
            "summary":            result.get("summary", ""),
            "counts": {
                "total":     len(flags),
                "dangerous": sum(1 for f in flags if f.get("category") == "Dangerous"),
                "unusual":   sum(1 for f in flags if f.get("category") == "Unusual"),
                "missing":   sum(1 for f in flags if f.get("category") == "Missing"),
                "critical":  sum(1 for f in flags if f.get("severity") == "Critical"),
                "high":      sum(1 for f in flags if f.get("severity") == "High"),
                "medium":    sum(1 for f in flags if f.get("severity") == "Medium"),
            },
            "detected_flags": flags,
        }

    except Exception as e:
        status    = "error"
        error_msg = str(e)
        logger.exception(f"[{request_id}] Red flag scan failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during red flag scan.")
    finally:
        elapsed = time.perf_counter() - t_start
        await log_request(
            request_id        = request_id,
            pdf_name          = file.filename or "unknown",
            total_pages       = total_pages,
            pages_analysed    = pages_to_read,
            completion_time_s = elapsed,
            endpoint          = "/red-flag-scanner",
            status            = status,
            error_message     = error_msg,
        )
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] Temp file deleted: '{file_path}'")


# ---------------------------------------------------------------------------
# POST /analyze/stream — Server-Sent Events
# ---------------------------------------------------------------------------

@router.post("/analyze/stream")
async def analyze_pdf_stream(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights"),
):
    """
    Analyse a PDF and stream results in real time using Server-Sent Events.
    PDF extraction runs in a thread pool — event loop never blocked during OCR.
    Token usage and timing are logged to PostgreSQL after the stream completes.
    nginx: add proxy_buffering off to the location block for this route.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── NEW STREAM REQUEST ───────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        async def _err():
            yield _sse("error", {"message": "Only PDF files are accepted."})
        return StreamingResponse(_err(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    async def _generate():
        was_truncated    = False
        pages_to_read    = 0
        total_pages      = 0
        final_highlights = []
        final_overview   = ""
        final_summary    = ""
        total_in_tok     = 0
        total_out_tok    = 0
        pdf_size         = 0
        status           = "success"
        error_msg        = None

        try:
            yield _sse("status", {"step": "saving", "message": "Saving uploaded file..."})
            content  = await file.read()
            pdf_size = len(content)
            with open(file_path, "wb") as f:
                f.write(content)
            logger.info(f"[{request_id}] saved {pdf_size:,} bytes")

            total_pages   = get_page_count(file_path)
            pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
            was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES

            yield _sse("status", {
                "step":    "extracting",
                "message": f"Extracting text from {pages_to_read} page(s)...",
                "pages":   pages_to_read,
            })

            try:
                # ── Non-blocking: OCR runs in thread pool ──────────────────
                pages = await _load_pdf_async(file_path, pages_to_read)
            except ValueError as e:
                status    = "error"
                error_msg = str(e)
                yield _sse("error", {"message": str(e)})
                return

            if all_pages_blank(pages):
                status = "blank_pdf"
                logger.warning(f"[{request_id}] blank PDF")
                yield _sse("status", {"step": "blank", "message": "PDF has no extractable content."})
                if analysis_type in (0, 1): yield _sse("overview", {"text": ""})
                if analysis_type in (0, 2): yield _sse("summary",  {"text": ""})
                yield _sse("done", {
                    "total_time":       round(time.perf_counter() - t_start, 2),
                    "total_highlights": 0,
                    "pages":            total_pages,
                    "blank_pdf":        True,
                })
                return

            merged_text = "\n\n".join(p.page_content for p in pages)
            logger.info(f"[{request_id}] merged {len(merged_text):,} chars")

            overview_sent   = False
            highlight_index = 0

            async for event_type, payload in generate_analysis_stream(merged_text):

                if event_type == "chunk_start":
                    yield _sse("status", {
                        "step":    "inference",
                        "message": f"Analysing chunk {payload['chunk']} of {payload['total']}...",
                        "chunk":   payload["chunk"],
                        "total":   payload["total"],
                    })

                elif event_type == "token":
                    # Live token delta — forward immediately so the client
                    # can render the LLM output as it is being generated.
                    yield _sse("token", {
                        "chunk": payload["chunk"],
                        "delta": payload["delta"],
                    })

                elif event_type == "chunk_done":
                    if analysis_type in (0, 3):
                        for h in payload.get("new_highlights", []):
                            yield _sse("highlight", {"text": h, "index": highlight_index})
                            highlight_index += 1

                    final_highlights = payload.get("all_highlights_so_far", final_highlights)
                    yield _sse("status", {
                        "step":              "chunk_done",
                        "chunk":             payload["chunk"],
                        "total":             payload["total"],
                        "highlights_so_far": len(final_highlights),
                    })

                elif event_type == "synthesis_start":
                    yield _sse("status", {
                        "step": "synthesis", "message": "Writing overview and summary...",
                    })

                elif event_type == "synthesis_done":
                    # Fired for both single- and multi-chunk paths
                    final_overview = payload.get("overview", final_overview)
                    final_summary  = payload.get("summary", final_summary)
                    if analysis_type in (0, 1):
                        yield _sse("overview", {"text": final_overview})
                        overview_sent = True
                    if analysis_type in (0, 2):
                        yield _sse("summary",  {"text": final_summary})

                elif event_type == "done":
                    total_in_tok  = payload.get("input_tokens",  0)
                    total_out_tok = payload.get("output_tokens", 0)

        except Exception as e:
            status    = "error"
            error_msg = str(e)
            logger.exception(f"[{request_id}] stream error: {e}")
            yield _sse("error", {"message": "Internal server error during PDF analysis."})
            return
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"[{request_id}] temp file deleted")

            elapsed = time.perf_counter() - t_start
            await log_request(
                request_id        = request_id,
                pdf_name          = file.filename or "unknown",
                pdf_size_bytes    = pdf_size,
                total_pages       = total_pages,
                pages_analysed    = pages_to_read,
                input_tokens      = total_in_tok,
                output_tokens     = total_out_tok,
                completion_time_s = elapsed,
                endpoint          = "/analyze/stream",
                status            = status,
                error_message     = error_msg,
            )

        elapsed = time.perf_counter() - t_start
        logger.info(f"[{request_id}] ── STREAM COMPLETE — {elapsed:.2f}s ──────")

        done_payload = {
            "total_time":       round(elapsed, 2),
            "total_highlights": len(final_highlights),
            "pages":            total_pages,
            "input_tokens":     total_in_tok,
            "output_tokens":    total_out_tok,
            "total_tokens":     total_in_tok + total_out_tok,
        }
        if was_truncated:
            done_payload.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)
        yield _sse("done", done_payload)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# POST /speech-to-text
# ---------------------------------------------------------------------------

_ALLOWED_AUDIO_EXTENSIONS = {
    ".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga",
    ".oga", ".ogg", ".wav", ".webm",
}

@router.post("/speech-to-text")
async def speech_to_text(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
):
    """
    Transcribe an uploaded audio file to text using OpenAI Whisper.
    Supported formats: flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm.
    Returns: {"text": "<transcription>"}
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in _ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_AUDIO_EXTENSIONS))}",
        )

    try:
        audio_bytes = await file.read()
        transcript, detected_language = await transcribe_audio(audio_bytes, file.filename)
    except Exception as e:
        logger.exception(f"[speech-to-text] transcription failed: {e}")
        raise HTTPException(status_code=500, detail="Audio transcription failed.")

    return {"text": transcript, "detected_language": detected_language}


# ---------------------------------------------------------------------------
# POST /convert/pdf-to-docx
# ---------------------------------------------------------------------------

@router.post("/convert/pdf-to-docx")
async def convert_pdf_to_docx(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
):
    """
    Convert an uploaded PDF to a downloadable DOCX file.
    Handles native, scanned, and mixed PDFs. No AI inference.
    PDF extraction runs in a thread pool — event loop never blocked during OCR.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── PDF TO DOCX ───────────────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}'")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[{request_id}] saved {len(content):,} bytes")

        try:
            # ── Non-blocking: OCR runs in thread pool ──────────────────────
            pages = await _load_pdf_async(file_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        logger.info(f"[{request_id}] extracted {len(pages)} page(s)")

        def _do_convert():
            from s_pdf_to_docx import _extract_blocks_ocr, _build_docx
            all_blocks = []
            for page in pages:
                blocks = _extract_blocks_ocr(page.page_content)
                all_blocks.append(blocks)
            return _build_docx(all_blocks, file.filename)

        loop = asyncio.get_running_loop()
        docx = await loop.run_in_executor(None, _do_convert)

        buffer = io.BytesIO()
        docx.save(buffer)
        buffer.seek(0)

        output_filename = file.filename.replace(".pdf", ".docx")
        elapsed = time.perf_counter() - t_start
        logger.info(f"[{request_id}] ── DOCX COMPLETE — {elapsed:.2f}s → '{output_filename}'")

        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Pages":             str(len(pages)),
                "X-Processing-Time":   str(round(elapsed, 2)),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] conversion failed: {e}")
        raise HTTPException(status_code=500, detail="PDF to DOCX conversion failed.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] temp file deleted")

# POST /compare-documents
@router.post("/compare-documents")
async def compare_documents_api(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
):
    """
    Compare two PDF documents and return a structured diff matching the
    target JSON schema.
 
    Steps:
      1. Extract text from both PDFs in parallel.
      2. Classify document types in parallel.
      3. Extract clauses from both documents in parallel (LLM-first).
      4. Run compare_documents() → produces target-shaped JSON.
      5. Log the result to the DB and inject log_id into the response.
    """
    request_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    t_start    = time.perf_counter()
 
    status              = "success"
    error_msg           = None
    duration_ms         = 0
    total_changes       = 0
    high_risk_changes   = 0
    overall_risk_level  = "low"
    similarity_percent  = ""
    input_tokens        = 0
    output_tokens       = 0
    compare_result      = {}
    path1 = path2       = None
 
    logger.info(f"[{request_id}] ── COMPARE START ── files=({file1.filename}, {file2.filename})")
 
    try:
        # ── Step 1: Extract text (parallel) ──────────────────────────────
        (text1, pages1, total1, _, _, path1), \
        (text2, pages2, total2, _, _, path2) = await asyncio.gather(
            extract_text_from_upload(file1, endpoint="/compare-documents"),
            extract_text_from_upload(file2, endpoint="/compare-documents"),
        )
 
        # ── Step 2: Validate length ───────────────────────────────────────
        if len(text1.split()) < 20 or len(text2.split()) < 20:
            raise HTTPException(
                status_code=400,
                detail="One or both documents are too short for comparison.",
            )
 
        # ── Step 3a: Classify both documents (parallel) ───────────────────
        doc_type1, doc_type2 = await asyncio.gather(
            _classify_document(text1),
            _classify_document(text2),
        )

        # ── Step 3b: Extract clauses using doc_type-aware prompts (parallel)
        (clauses1, in_tok1, out_tok1), \
        (clauses2, in_tok2, out_tok2) = await asyncio.gather(
            extract_clauses(text1, doc_type=doc_type1),
            extract_clauses(text2, doc_type=doc_type2),
        )

        input_tokens  = in_tok1 + in_tok2
        output_tokens = out_tok1 + out_tok2

        logger.info(
            f"[{request_id}] types=({doc_type1}, {doc_type2}) "
            f"clauses=({len(clauses1)}, {len(clauses2)}) "
            f"tokens={input_tokens}in/{output_tokens}out"
        )
 
        # ── Step 4: Compare ───────────────────────────────────────────────
        compare_result = await compare_documents(
            text1, text2, clauses1, clauses2,
            doc1_filename=file1.filename or "document_1.pdf",
            doc2_filename=file2.filename or "document_2.pdf",
            session_id=session_id,
            log_id=None,    # will be backfilled after DB insert
        )
 
        # Extract metrics for DB logging
        comp  = compare_result.get("comparison", {})
        total_changes      = comp.get("total_changes", 0)
        high_risk_changes  = comp.get("high_risk_changes", 0)
        overall_risk_level = comp.get("overall_risk_level", "low")
        similarity_percent = comp.get("text_diff_stats", {}).get("similarity_percent", "")
 
    except HTTPException:
        status    = "failed"
        error_msg = "HTTP error"
        raise
 
    except Exception as e:
        status    = "failed"
        error_msg = str(e)
        logger.exception(f"[{request_id}] comparison error: {e}")
        raise HTTPException(status_code=500, detail="Document comparison failed.")
    # finally:
    #     duration_ms = int((time.perf_counter() - t_start) * 1000)
 
    #     # ── Step 5: Log to DB ─────────────────────────────────────────────
    #     log_id = await log_comparison_request(
    #         session_id=session_id,
    #         request_id=request_id,
    #         doc1_filename=file1.filename or "unknown",
    #         doc2_filename=file2.filename or "unknown",
    #         status=status,
    #         duration_ms=duration_ms,
    #         total_changes=total_changes,
    #         high_risk_changes=high_risk_changes,
    #         overall_risk_level=overall_risk_level,
    #         similarity_percent=similarity_percent,
    #         result_json=compare_result if status == "success" else None,
    #         input_tokens=input_tokens,
    #         output_tokens=output_tokens,
    #         error_message=error_msg,
    #     )
 
    #     # Cleanup temp files
    #     for p in [path1, path2]:
    #         if p and os.path.exists(p):
    #             os.remove(p)
 
    #     logger.info(f"[{request_id}] ── COMPLETE — {duration_ms}ms ──")
 
    # # Inject log_id into response (DB layer returns the inserted row id)
    # if compare_result and log_id:
    #     compare_result["log_id"] = log_id
 
    finally:
        # Cleanup temp files regardless of success or failure
        for p in [path1, path2]:
            if p and os.path.exists(p):
                os.remove(p)
                logger.debug(f"[{request_id}] temp file deleted: {p}")

        duration_ms = int((time.perf_counter() - t_start) * 1000)
        logger.info(f"[{request_id}] ── COMPLETE — {duration_ms}ms status={status} ──")

    return compare_result