"""
t_main.py
=========
FastAPI application for PDF summarization — supports small, large,
text-based, image-only, scanned, and mixed PDFs.

Architecture
------------
POST /analyze          → streams file to disk, returns job_id (202 Accepted)
GET  /status/{job_id}  → poll progress  {stage, current, total}
GET  /result/{job_id}  → retrieve final JSON when done

Pipeline (3-level hierarchical summarization)
---------------------------------------------
Level 0  extract_text_from_pdf()     parallel batch OCR + native text
Level 1  generate_analysis()         per-chunk AI analysis (GPU-semaphore-gated)
Level 2  synthesize_section()        every SECTION_SIZE chunks → 1 section summary
Level 3  synthesize_final()          all section summaries → 1 final output

Edge cases handled
------------------
A) Image-only PDF, OCR unavailable
   → clear "failed" status with install instructions for PaddleOCR-VL

B) PDF with < MIN_TEXT_CHARS total text after extraction
   → clear "failed" status with exact character count

C) Corrupt / encrypted PDF
   → ValueError from extract_text_from_pdf → "failed" with reason

All fixes from original review retained
----------------------------------------
- Streaming upload  (no full-file RAM read)
- UUID temp filename  (no path traversal)
- clean_text imported from t_pdf_utils  (no local duplicate)
- load_model() called once at startup via lifespan
- asyncio.get_running_loop()  (not deprecated get_event_loop)
- GPU semaphore + retry inside t_ai_model
- Highlight deduplication + cap at MAX_HIGHLIGHTS
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import fitz
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from t_key_clause_extraction import  classify_document, DOCUMENT_HANDLERS


from backupfiles.t_ai_model import (
    generate_analysis,
    load_model,
    synthesize_final,
    synthesize_section,
)
from backupfiles.t_pdf_utils import chunk_text, clean_text, extract_text_from_pdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER    = "temp"
UPLOAD_CHUNK_MB  = 1                        # streaming read size
MAX_UPLOAD_BYTES = 500 * 1024 * 1024        # 500 MB hard upload limit
SECTION_SIZE     = 20                       # chunks per Level-2 section
MAX_HIGHLIGHTS   = 10                       # cap on final highlight list
MIN_TEXT_CHARS   = 20                       # minimum useful extracted text

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory job store
# Replace with Redis + a persistent DB for multi-worker deployments.
# ---------------------------------------------------------------------------
JobStatus = Literal[
    "pending", "extracting", "analyzing",
    "synthesizing", "done", "failed",
]

_jobs: dict[str, dict] = {}


def _new_job() -> str:
    jid        = str(uuid.uuid4())
    _jobs[jid] = {
        "status":   "pending",
        "progress": {"stage": "queued", "current": 0, "total": 0},
        "result":   None,
        "error":    None,
    }
    return jid


def _set_progress(
    jid: str,
    stage: str,
    current: int = 0,
    total: int   = 0,
) -> None:
    if jid not in _jobs:
        return
    _jobs[jid]["progress"] = {"stage": stage, "current": current, "total": total}
    _jobs[jid]["status"]   = (
        "extracting"   if stage == "extraction"                          else
        "analyzing"    if stage == "chunk_analysis"                      else
        "synthesizing" if stage in ("section_synthesis", "final_synthesis") else
        _jobs[jid]["status"]
    )


def _fail(jid: str, message: str) -> None:
    _jobs[jid]["status"] = "failed"
    _jobs[jid]["error"]  = message
    logger.error(f"[{jid}] FAILED: {message}")


# ---------------------------------------------------------------------------
# Application lifespan  (model loaded once at startup)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="PDF Summarizer — Large Document Edition",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# JSON extraction  (3-strategy parser)
# ---------------------------------------------------------------------------

def _fix_json_string(raw: str) -> str:
    """Clean common Mistral output issues before JSON parsing."""
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Fix illegal backslash escapes (keep valid: \" \\ \/ \b \f \n \r \t \uXXXX)
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    # Strip ASCII control characters except \n \r \t
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    return raw


def _extract_field_by_regex(text: str, field: str) -> str:
    m = re.search(rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_highlights_by_regex(text: str) -> list[str]:
    m = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not m:
        return []
    items = re.findall(r'"(.*?)"(?=\s*[,\]])', m.group(1), re.DOTALL)
    return [i.strip() for i in items if i.strip()]


def _postprocess_highlights(data: dict) -> dict:
    """Flatten comma-packed highlight items and strip stray quote characters."""
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            parts = item.split('", "') if '", "' in item else [item]
            for p in parts:
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
    return data


def extract_json(text: str) -> dict:
    """
    Parse JSON from raw model output using three fallback strategies:

    Strategy 1 — direct parse after cleaning
    Strategy 2 — isolate outermost { … } block, then parse;
                 on failure attempt a conservative quote-escape repair
    Strategy 3 — per-field regex extraction (last resort)
    """
    empty = {"overview": "", "summary": "", "highlights": []}

    if not text or not text.strip():
        return empty

    cleaned = _fix_json_string(text)

    # Strategy 1
    try:
        return _postprocess_highlights(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # Strategy 2
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        candidate = m.group()
        try:
            return _postprocess_highlights(json.loads(candidate))
        except json.JSONDecodeError as exc:
            logger.warning(f"JSON block parse failed ({exc}); trying repair")
            repaired = re.sub(
                r'(?<=[^\\])"(?=[^,\]}\n:}{\[])', r'\\"', candidate
            )
            try:
                return _postprocess_highlights(json.loads(repaired))
            except json.JSONDecodeError:
                pass

    # Strategy 3
    logger.error("All JSON parse strategies failed — falling back to regex")
    overview   = _extract_field_by_regex(cleaned, "overview")
    summary    = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)
    if overview or summary or highlights:
        return {"overview": overview, "summary": summary, "highlights": highlights}

    return empty


# ---------------------------------------------------------------------------
# Highlight deduplication
# ---------------------------------------------------------------------------

def _dedup_highlights(
    highlights: list[str],
    limit: int = MAX_HIGHLIGHTS,
) -> list[str]:
    """
    Normalise → deduplicate → cap.

    Normalisation: lowercase, strip punctuation, collapse whitespace.
    This catches near-duplicates like "Revenue was $2M" and "Revenue: $2M".
    """
    seen:   set[str]  = set()
    unique: list[str] = []
    for h in highlights:
        key = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', h.lower())).strip()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique[:limit]


# ---------------------------------------------------------------------------
# Hierarchical summarization pipeline
# ---------------------------------------------------------------------------

async def _run_pipeline(
    jid:           str,
    file_path:     str,
    analysis_type: int,
) -> None:
    """
    Full async pipeline — runs as a FastAPI BackgroundTask.

    Stage 0  Text extraction      (parallel page batches, adaptive OCR)
    Stage 1  Chunk analysis       (Level 0 — per chunk, GPU-semaphore-gated)
    Stage 2  Section synthesis    (Level 1 — every SECTION_SIZE chunks)
    Stage 3  Final synthesis      (Level 2 — all section summaries → 1 output)
    """
    try:

        # ── Stage 0: extraction ─────────────────────────────────────────────
        _set_progress(jid, "extraction", 0, 1)
        logger.info(f"[{jid}] Extraction started")

        try:
            text = await extract_text_from_pdf(file_path)
        except RuntimeError as exc:
            # Edge case A: image-only PDF or blank/corrupt PDF
            _fail(jid, str(exc))
            return
        except ValueError as exc:
            # Corrupt / encrypted file
            _fail(jid, f"Could not open PDF: {exc}")
            return

        text = clean_text(text)

        # Edge case B: total extracted text is too short to summarise
        if len(text) < MIN_TEXT_CHARS:
            _fail(
                jid,
                f"Extracted text is too short to summarise "
                f"({len(text)} characters after cleaning). "
                f"The PDF may contain only images, decorative elements, "
                f"or a single very short line of text. "
                f"Minimum required: {MIN_TEXT_CHARS} characters.",
            )
            return

        logger.info(f"[{jid}] Extracted {len(text):,} characters")
        _set_progress(jid, "extraction", 1, 1)

        # ── Stage 1: chunk analysis — Level 0 ───────────────────────────────
        chunks       = chunk_text(text)
        total_chunks = len(chunks)

        # Belt-and-suspenders: chunk_text should never return [] after the
        # MIN_TEXT_CHARS guard above, but handle it cleanly just in case.
        if not chunks:
            _fail(jid, "Text could not be split into processable chunks.")
            return

        logger.info(f"[{jid}] Analysing {total_chunks} chunks")
        _set_progress(jid, "chunk_analysis", 0, total_chunks)

        async def _analyse_chunk(idx: int, chunk: str) -> dict:
            raw    = await generate_analysis(chunk)
            parsed = extract_json(raw)
            _set_progress(jid, "chunk_analysis", idx + 1, total_chunks)
            logger.info(f"[{jid}] Chunk {idx+1}/{total_chunks} done")
            return parsed

        # All chunks run concurrently; GPU semaphore inside t_ai_model
        # limits actual simultaneous model.generate() calls.
        chunk_results: list[dict] = list(
            await asyncio.gather(
                *[_analyse_chunk(i, c) for i, c in enumerate(chunks)]
            )
        )

        # ── Stage 2: section synthesis — Level 1 ────────────────────────────
        sections       = [
            chunk_results[i : i + SECTION_SIZE]
            for i in range(0, total_chunks, SECTION_SIZE)
        ]
        total_sections = len(sections)
        logger.info(f"[{jid}] Synthesizing {total_sections} sections")
        _set_progress(jid, "section_synthesis", 0, total_sections)

        async def _synthesize_sec(idx: int, sec: list[dict]) -> dict:
            label  = f"section {idx+1} of {total_sections}"
            raw    = await synthesize_section(sec, label)
            parsed = extract_json(raw)
            _set_progress(jid, "section_synthesis", idx + 1, total_sections)
            logger.info(f"[{jid}] Section {idx+1}/{total_sections} done")
            return parsed

        section_results: list[dict] = list(
            await asyncio.gather(
                *[_synthesize_sec(i, s) for i, s in enumerate(sections)]
            )
        )

        # ── Stage 3: final synthesis — Level 2 ──────────────────────────────
        _set_progress(jid, "final_synthesis", 0, 1)
        logger.info(f"[{jid}] Final synthesis across {total_sections} sections")

        # Detect page count for the final prompt (best-effort)
        try:
            with fitz.open(file_path) as doc:
                total_pages = len(doc)
        except Exception:
            total_pages = total_chunks   # reasonable fallback

        final_raw    = await synthesize_final(section_results, total_pages)
        final_parsed = extract_json(final_raw)

        # Deduplicate and cap highlights
        final_parsed["highlights"] = _dedup_highlights(
            final_parsed.get("highlights", [])
        )

        # Filter output by analysis_type
        if analysis_type == 1:
            result = {"overview":   final_parsed.get("overview",   "")}
        elif analysis_type == 2:
            result = {"summary":    final_parsed.get("summary",    "")}
        elif analysis_type == 3:
            result = {"highlights": final_parsed.get("highlights", [])}
        else:
            result = final_parsed

        _jobs[jid]["status"] = "done"
        _jobs[jid]["result"] = result
        _set_progress(jid, "final_synthesis", 1, 1)
        logger.info(f"[{jid}] Pipeline complete ✓")

    except Exception as exc:
        logger.exception(f"[{jid}] Unexpected pipeline error")
        _fail(jid, f"Unexpected error: {exc}")

    finally:
        # Always delete the temp file, even on failure
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[{jid}] Temp file deleted")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze", status_code=202)
async def analyze_pdf(
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    analysis_type:    int        = Query(
        0, ge=0, le=3,
        description=(
            "What to return: "
            "0=all fields  "
            "1=overview only  "
            "2=summary only  "
            "3=highlights only"
        ),
    ),
):
    """
    Upload a PDF and start async analysis.

    The file is streamed to disk chunk-by-chunk — no full-file RAM spike.
    Returns HTTP 202 with a job_id immediately.

    Poll GET /status/{job_id} to track progress.
    Retrieve the result with GET /result/{job_id}.

    Supports text-based, scanned, image-only, and mixed PDFs up to 500 MB.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name  = f"{uuid.uuid4()}.pdf"
    file_path  = os.path.join(UPLOAD_FOLDER, safe_name)
    total_read = 0
    chunk_size = UPLOAD_CHUNK_MB * 1024 * 1024

    try:
        with open(file_path, "wb") as out:
            while True:
                data = await file.read(chunk_size)
                if not data:
                    break
                total_read += len(data)
                if total_read > MAX_UPLOAD_BYTES:
                    out.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                        ),
                    )
                out.write(data)
    except HTTPException:
        raise
    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    jid = _new_job()
    logger.info(
        f"[{jid}] Accepted {safe_name} "
        f"({total_read / 1024 / 1024:.1f} MB), "
        f"analysis_type={analysis_type}"
    )

    background_tasks.add_task(_run_pipeline, jid, file_path, analysis_type)

    return JSONResponse(
        status_code=202,
        content={
            "job_id":  jid,
            "message": "File accepted. Poll /status/{job_id} for progress.",
        },
    )


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Poll job progress.

    Response
    --------
    job_id    : str
    status    : pending | extracting | analyzing | synthesizing | done | failed
    progress  : {stage: str, current: int, total: int}
    error     : str  (only present when status == "failed")

    Common error messages
    ----------------------
    Image-only PDF, no OCR:
      "This PDF is entirely image-based … Install PaddleOCR-VL …"
    Too little text:
      "Extracted text is too short to summarise (N characters after cleaning)."
    Corrupt / encrypted:
      "Could not open PDF: …"
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    resp: dict = {
        "job_id":   job_id,
        "status":   job["status"],
        "progress": job["progress"],
    }
    if job["status"] == "failed":
        resp["error"] = job["error"]
    return resp


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """
    Retrieve the final analysis result.

    HTTP 200  — processing done; result JSON returned.
    HTTP 202  — still processing; check /status for progress.
    HTTP 404  — unknown job_id.
    HTTP 500  — job failed; detail contains the human-readable reason.

    Result JSON (analysis_type=0)
    ------------------------------
    {
      "overview":   "1-2 sentence document description",
      "summary":    "5-7 sentence executive summary",
      "highlights": ["fact 1", "fact 2", … up to 10]
    }
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=job.get("error", "Unknown error. Check /status for details."),
        )

    if job["status"] != "done":
        return JSONResponse(
            status_code=202,
            content={
                "job_id":   job_id,
                "status":   job["status"],
                "progress": job["progress"],
                "message":  "Processing in progress. Try again shortly.",
            },
        )

    return job["result"]

@app.post("/key-clause-extraction")
async def key_clause_extraction(file: UploadFile = File(...)):

    # ==============================
    # Step 0: Validate file
    # ==============================
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # ==============================
    # Step 1: Save file safely
    # ==============================
    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        with open(file_path, "wb") as f:
            f.write(raw_bytes)

        del raw_bytes  # free memory

        # ==============================
        # Step 2: Extract text (FIXED)
        # ==============================
        text = extract_text_from_pdf(file_path)

        if not text:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found"
            )

        # Optional extra cleaning
        text = clean_text(text)

        # ==============================
        # Step 3: Classification
        # ==============================
        doc_type = await classify_document(text)
        doc_type = doc_type.lower().strip()

        # ==============================
        # Step 4: Route
        # ==============================
        handler = DOCUMENT_HANDLERS.get(doc_type)

        if handler:
            return await handler(text)

        # ==============================
        # Step 5: Fallback
        # ==============================
        return {
            "status": "unsupported",
            "document_type": doc_type,
            "message": "Unsupported document type."
        }

    finally:
        # ==============================
        # Step 6: Cleanup
        # ==============================
        if os.path.exists(file_path):
            os.remove(file_path)

