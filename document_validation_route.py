import os
import uuid
import time
import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from document_validation.document_validation import validate_document, validate_for_signing
from feature_modules.key_clause_extraction import extract_text_from_upload
from db_files.db import log_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document-validation", tags=["Document Validation"])

from document_validation.document_validation import validate_document, validate_for_signing, auto_validate_document

# ---------------------------------------------------------------------------
# POST /validate
# ---------------------------------------------------------------------------

@router.post("/validate")
async def validate_doc_endpoint(
    file: UploadFile = File(...),
    attachments: Optional[List[str]] = Form(None, description="Optional list of attachment names to cross-check")
):
    """
    Validate a document. 
    The AI will automatically detect the document type and apply the appropriate validation logic.
    Accepts PDF or DOCX files.
    """
    text, pages_to_read, total_pages, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/validate"
    )

    status    = "success"
    error_msg = None
    total_in_tok = 0
    total_out_tok = 0

    try:
        logger.info(f"[{request_id}] Starting Automated Document Validation...")

        # Use auto_validate_document which handles classification and validation
        result, in_tok, out_tok = await auto_validate_document(text, attachments)

        total_in_tok = in_tok
        total_out_tok = out_tok
        return result

    except Exception as e:
        status    = "error"
        error_msg = str(e)
        logger.exception(f"[{request_id}] Document Validation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during document validation.")
    finally:
        elapsed = time.perf_counter() - t_start
        await log_request(
            request_id        = request_id,
            pdf_name          = file.filename or "unknown",
            total_pages       = total_pages,
            pages_analysed    = pages_to_read,
            input_tokens      = total_in_tok,
            output_tokens     = total_out_tok,
            completion_time_s = elapsed,
            endpoint          = "/validate",
            status            = status,
            error_message     = error_msg,
        )
        if os.path.exists(file_path):
            os.remove(file_path)

