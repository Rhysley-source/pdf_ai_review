import logging
import os
import tempfile
from typing import Optional, Dict, Any

from fastapi import UploadFile

from utils.pdf_utils import load_pdf, merge_pages
from llm_model.ai_model import run_llm
from utils.json_utils import extract_json
from .prompts import prompts

logger = logging.getLogger(__name__)


# ✅ Prompt Generator Function
def get_validation_prompt(document_type: str, specific_checks: Optional[str] = None) -> str:
    base_prompt = "You are an expert document validator."

    
    selected_prompt = prompts.get(document_type.lower(), "Perform a general document validation.")

    final_prompt = f"""{base_prompt}

Document Type: {document_type}

{selected_prompt}
"""

    if specific_checks:
        final_prompt += f"\nAlso specifically check for: {specific_checks}\n"

    final_prompt += """
Carefully analyze the document for issues, especially:
- **Missing fields:** Identify any expected fields that are absent.
- **Placeholders:** Note any variables or entries that seem to be generic placeholders (e.g., "[Client Name]", "MM/DD/YYYY").
- **Incorrect/Unprovided Entries:** Point out any entries that are clearly incorrect or have no value provided (e.g., "N/A", "------").
- Risky or harmful content.
- Structural or formatting issues.

Provide your analysis in STRICT JSON format with the following keys:
{
  "overview": "Overall assessment of the document's completeness and quality.",
  "summary": "A detailed summary of the document's validation status. **Crucially, list all identified missing fields, placeholders, and incorrect/unprovided entries here.** Group them logically and provide context.",
  "highlights": [
    "Specific instance of a missing field or placeholder identified.",
    "Another specific detail of an incorrect entry."
  ]
}
Each item in 'highlights' should be a concise, specific finding of a missing field, placeholder, or incorrect entry.
"""

    return final_prompt


# ✅ Main Validation Function
async def validate_document(
    file: UploadFile,
    specific_checks: Optional[str] = None
) -> Dict[str, Any]:

    if not file.filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported for document validation.")

    temp_file_path = None

    try:
        # ✅ Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_file_path = tmp.name

        logger.info(f"Processing PDF file: {file.filename}")

        # ✅ Extract text
        pdf_document = load_pdf(temp_file_path)
        extracted_text = merge_pages(pdf_document)[0].page_content

        if not extracted_text or len(extracted_text.strip()) < 20:
            return {"status": "error", "message": "Could not extract sufficient text from the PDF."}

        logger.info(f"Extracted {len(extracted_text):,} characters")

        # =========================
        # ✅ STEP 1: CLASSIFICATION
        # =========================
        classification_prompt = """
You are a document classification expert.

Classify the document into one of these:
- resume
- contract
- invoice
- report
- letter
- other

Return STRICT JSON:
{
  "document_type": "..."
}
"""

        document_check = await run_llm(
            text=extracted_text[:4000],  # limit for safety
            system_prompt=classification_prompt
        )

        classification_json = extract_json(document_check)

        document_type = "other"
        if classification_json and isinstance(classification_json, dict):
            document_type = classification_json.get("document_type", "other")

        logger.info(f"Detected document type: {document_type}")

        # =========================
        # ✅ STEP 2: PROMPT GENERATION
        # =========================
        system_prompt = get_validation_prompt(document_type, specific_checks)

        # =========================
        # ✅ STEP 3: VALIDATION
        # =========================
        logger.info("Sending document for validation...")

        llm_response_text = await run_llm(
            text=extracted_text[:12000],  # limit to avoid token overflow
            system_prompt=system_prompt
        )

        validation_result = extract_json(llm_response_text)

        if not validation_result:
            logger.warning("Invalid JSON from LLM")

            return {
                "status": "warning",
                "document_type": document_type,
                "message": "Validation completed but JSON parsing failed.",
                "raw_response": llm_response_text
            }

        return {
            "status": "success",
            "document_type": document_type,
            "result": validation_result
        }

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return {"status": "error", "message": str(ve)}

    except Exception as e:
        logger.exception("Unexpected error during validation")
        return {"status": "error", "message": str(e)}

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Removed temp file: {temp_file_path}")