import logging
import os
import tempfile
from typing import Optional, Dict, Any

from fastapi import UploadFile

from utils.pdf_utils import load_pdf, merge_pages
from llm_model.ai_model import run_llm
from utils.json_utils import extract_json

logger = logging.getLogger(__name__)

async def validate_document(
    file: UploadFile,
    validation_type: Optional[str] = "general",
    specific_checks: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validates a document by extracting text and sending it to an LLM for analysis.

    Args:
        file: The uploaded PDF file.
        validation_type: The general type of validation to perform (e.g., "general", "legal", "financial").
        specific_checks: A comma-separated string of specific items to check for.

    Returns:
        A dictionary containing the validation results from the LLM.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported for document validation.")

    temp_file_path = None
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_file_path = tmp.name

        logger.info(f"Processing PDF file: {file.filename}")

        # Load PDF and extract text
        pdf_document = load_pdf(temp_file_path)
        extracted_text = merge_pages(pdf_document)[0].page_content

        if not extracted_text:
            return {"status": "error", "message": "Could not extract text from the PDF."}

        logger.info(f"Extracted {len(extracted_text):,} characters from PDF.")

        # Prepare prompt for LLM
        prompt_parts = [
            "You are an expert document validator. Analyze the provided document text for any issues.",
            f"Document type for validation: {validation_type}.",
        ]
        if specific_checks:
            prompt_parts.append(f"Specifically check for: {specific_checks}.")
        
        prompt_parts.append(
            "Look for missing fields, incorrect clauses, harmful terms and conditions, "
            "or any other discrepancies based on the validation type and specific checks.",
        )
        prompt_parts.append(
            "Provide your analysis in a JSON format with the following keys: "
            "'summary' (overall assessment), "
            "'issues' (a list of detected issues, each with 'type', 'description', 'severity'), "
            "'recommendations' (suggestions for improvement). "
            "If no issues are found, the 'issues' list should be empty. "
            "If there are no recommendations, 'recommendations' should be an empty list."
        )
        prompt_parts.append("""--- DOCUMENT START ---""")
        prompt_parts.append(extracted_text)
        prompt_parts.append("""--- DOCUMENT END ---""")

        llm_prompt = "\n".join(prompt_parts)

        # Call the LLM
        logger.info("Sending document text to LLM for validation...")
        llm_response_text = await run_llm(llm_prompt)
        logger.info("Received response from LLM.")

        # Extract JSON from LLM response
        validation_result = extract_json(llm_response_text)

        if not validation_result:
            logger.warning(f"LLM response did not contain valid JSON: {llm_response_text[:500]}...")
            return {
                "status": "warning",
                "message": "LLM analysis complete, but structured JSON output could not be parsed.",
                "raw_llm_response": llm_response_text
            }

        return {"status": "success", "result": validation_result}

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return {"status": "error", "message": str(ve)}
    except Exception as e:
        logger.exception("An unexpected error occurred during document validation.")
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Removed temporary file: {temp_file_path}")
