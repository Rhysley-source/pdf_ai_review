import logging
from typing import Dict, Any, Tuple
from .prompt_templates import PromptTemplates
from llm_model.ai_model import _run_inference_text
from utils.json_utils import extract_json

logger = logging.getLogger(__name__)

async def validate_document(document_text: str, doc_type: str = "general") -> Tuple[Dict[str, Any], int, int]:
    """
    Validate a document using OpenAI based on its type.
    Returns (parsed_json, input_tokens, output_tokens).
    """
    if not document_text or not document_text.strip():
        return {"issues": ["Empty document text"], "suggestions": [], "summary": "No content to analyze."}, 0, 0

    template = PromptTemplates.get_template(doc_type)
    system_prompt = PromptTemplates.SYSTEM
    
    # Fill the template
    user_prompt = template.substitute(document_text=document_text)
    
    logger.info(f"[validate_document] Validating as type: {doc_type}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    
    # Using _run_inference_text to get tokens as well
    raw_content, in_tokens, out_tokens = await _run_inference_text(
        messages=messages,
        label=f"validate_{doc_type}"
    )
    
    parsed = extract_json(raw_content)
    return parsed, in_tokens, out_tokens

async def validate_for_signing(document_text: str, attachments: list[str] = None) -> Tuple[Dict[str, Any], int, int]:
    """
    Specifically validate a document for pre-signing requirements.
    """
    if not document_text or not document_text.strip():
        return {
            "missing_signature_fields": False,
            "missing_dates": False,
            "blank_fields": ["Empty document text"],
            "missing_attachments": [],
            "alerts": ["No content to analyze."]
        }, 0, 0

    attachments_str = ", ".join(attachments) if attachments else "None provided"
    
    template = PromptTemplates.get_template("signing_validation")
    system_prompt = PromptTemplates.SYSTEM
    
    user_prompt = template.substitute(
        document_text=document_text,
        attachments_list=attachments_str
    )
    
    logger.info("[validate_for_signing] Starting pre-signing validation")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    
    raw_content, in_tokens, out_tokens = await _run_inference_text(
        messages=messages,
        label="signing_validation"
    )
    
    parsed = extract_json(raw_content)
    
    # Ensure all required keys are present in output
    required_keys = {
        "missing_signature_fields": False,
        "missing_dates": False,
        "blank_fields": [],
        "missing_attachments": [],
        "alerts": []
    }
    
    for key, default in required_keys.items():
        if key not in parsed:
            parsed[key] = default
            
    return parsed, in_tokens, out_tokens

async def auto_validate_document(document_text: str, attachments: list[str] = None) -> Tuple[Dict[str, Any], int, int]:
    """
    1. Classify the document type.
    2. Validate based on the detected type.
    Returns (result, total_in_tokens, total_out_tokens)
    """
    from feature_modules.key_clause_extraction import classify_document
    
    # Call 1: Classify
    doc_type = await classify_document(document_text)
    logger.info(f"[auto_validate] Detected document type: {doc_type}")
    
    # For classification, run_llm tokens aren't explicitly returned to this level, 
    # but we'll track the validation tokens.
    
    # Call 2: Validate
    # Mapping classifier types to validation templates
    # classifier returns: contract, resume, invoice, report, other
    
    if doc_type == "contract":
        # For contracts, we default to signing validation as it's the most common pre-signing requirement
        result, in_tok, out_tok = await validate_for_signing(document_text, attachments)
    elif doc_type == "resume":
        result, in_tok, out_tok = await validate_document(document_text, "resume")
    else:
        result, in_tok, out_tok = await validate_document(document_text, "general")
        
    # Inject the detected type into the result for clarity
    if isinstance(result, dict):
        result["detected_document_type"] = doc_type
        
    return result, in_tok, out_tok
