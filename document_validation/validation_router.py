from fastapi import APIRouter, UploadFile, File, Query, HTTPException, status
from typing import Optional

from document_validation.validation_logic import validate_document

router = APIRouter()

@router.post("/validate-document", tags=["Document Validation"])
async def validate_document_endpoint(
    file: UploadFile = File(..., description="The PDF document to validate."),
):
    """
    Endpoint to validate a PDF document using an LLM.

    Upload a PDF file and optionally specify a validation type and specific checks.
    The LLM will analyze the document for issues like missing fields, wrong clauses,
    or harmful terms and conditions, returning a structured JSON response.
    """
    try:
        result = await validate_document(file)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")
