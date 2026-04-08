import json
import logging
from langchain_core.tools import tool

from feature_modules.key_clause_extraction import classify_document, DOCUMENT_HANDLERS
from feature_modules.risk_detection import analyze_document_risks
from feature_modules.red_flag_scanner import scan_red_flags

logger = logging.getLogger(__name__)


@tool
async def key_clause_extraction_tool(text: str) -> str:
    """
    Extract key clauses from a legal document.

    Classifies the document type (contract, resume, invoice) then extracts
    structured key clauses such as Agreement Type, Effective Date, Payment Terms,
    Governing Law, and more. Returns a JSON string with the extracted data.
    """
    try:
        doc_type = await classify_document(text)
        handler = DOCUMENT_HANDLERS.get(doc_type)
        if handler:
            result = await handler(text)
        else:
            result = {
                "status": "unsupported",
                "document_type": doc_type,
                "message": f"No handler available for document type: {doc_type}",
            }
        logger.info(f"[key_clause_tool] doc_type={doc_type} status={result.get('status')}")
        return json.dumps(result)
    except Exception as e:
        logger.error(f"[key_clause_tool] Error: {e}")
        return json.dumps({"error": str(e), "status": "error"})


@tool
async def detect_risks_tool(text: str) -> str:
    """
    Detect legal and financial risks in a document.

    Automatically detects the document type (contract, employment, NDA, lease,
    invoice, resume) and runs a type-specific risk analysis. Returns a JSON string
    containing a risk score, list of detected risks with severity and mitigation
    advice, missing required fields, and an overall assessment.
    """
    try:
        result = await analyze_document_risks(text)
        logger.info(f"[detect_risks_tool] doc_type={result.get('document_type')} status={result.get('status')}")
        return json.dumps(result)
    except Exception as e:
        logger.error(f"[detect_risks_tool] Error: {e}")
        return json.dumps({"error": str(e), "status": "error"})


@tool
async def red_flag_scanner_tool(text: str) -> str:
    """
    Scan a document for dangerous, unusual, or missing contract language.

    Uses a fixed per-document-type checklist to evaluate the contract for red flags
    such as unlimited liability, one-sided indemnity, missing termination clauses, etc.
    Severity is assigned from a hardcoded map for consistency. Returns a JSON string
    with detected flags, overall risk level (Critical/High/Medium/Low), and a summary.
    """
    try:
        result = await scan_red_flags(text)
        flags = result.get("detected_flags", [])
        logger.info(
            f"[red_flag_scanner_tool] doc_type={result.get('document_type')} "
            f"flags={len(flags)} risk={result.get('overall_risk_level')}"
        )
        return json.dumps(result)
    except Exception as e:
        logger.error(f"[red_flag_scanner_tool] Error: {e}")
        return json.dumps({"error": str(e), "status": "error"})


# Expose all tools as a list for easy registration in a LangGraph agent
ALL_TOOLS = [
    key_clause_extraction_tool,
    detect_risks_tool,
    red_flag_scanner_tool,
]
