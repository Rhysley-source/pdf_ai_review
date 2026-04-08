import logging
from typing import Dict
from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json
import re

logger = logging.getLogger(__name__)

CLAUSE_HINTS = [
    "payment", "pricing", "cost",
    "liability", "termination",
    "timeline", "delivery",
    "scope", "penalty", "renewal",
    "confidentiality", "warranty"
]

# ============================================================
# SECTION EXTRACTION (Rule-Based)
# ============================================================

def _is_heading(line: str) -> bool:
    """
    Checks if the line is a heading (e.g., Section, 1. Clause).
    """
    return bool(re.match(r"^\d+\.\s+.+", line) or re.match(r"^\d+\)\s+.+", line) or line.isupper())

def _extract_rule_based(text: str) -> Dict[str, str]:
    """
    Extract clauses based on document structure.
    """
    clauses = {}
    current_heading = "introduction"
    buffer = []

    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if _is_heading(line):
            if buffer:
                clauses[current_heading] = " ".join(buffer).strip()
            current_heading = line
            buffer = []
        else:
            buffer.append(line)
    
    if buffer:
        clauses[current_heading] = " ".join(buffer).strip()
    
    return clauses

# ============================================================
# LLM-BASED EXTRACTION (Advanced)
# ============================================================

def _build_prompt(text: str) -> str:
    return f"""
You are a legal document parser.

Extract key clauses from the document.

Return STRICT JSON ONLY (no explanation, no text outside JSON):

{{
  "clauses": {{
    "payment_terms": "...",
    "liability": "...",
    "termination": "...",
    "scope": "...",
    "timeline": "...",
    "other": "..."
  }}
}}

Rules:
- Fill ONLY clauses that exist
- If a clause is missing → return ""
- Do NOT invent content
- Keep each clause under 2-3 lines
- Output MUST be valid JSON

Document:
{text[:4000]}
"""

async def _extract_with_llm(text: str) -> Dict[str, str]:
    """
    Uses LLM for extraction of relevant clauses from the document.
    """
    raw = await run_llm("", _build_prompt(text))
    result = extract_json(raw)

    if not result:
        return {}

    return result.get("clauses", {})

# ============================================================
# FINAL CLAUSE EXTRACTION (Hybrid: Rule-Based + LLM)
# ============================================================

async def extract_clauses(text: str) -> Dict[str, str]:
    """
    Fully hybrid clause extraction:
    - Rule-based extraction for document structure
    - LLM-based extraction for semantic content
    """
    try:
        # Step 1: Extract with rules
        rule_based_clauses = _extract_rule_based(text)
        
        # Step 2: Extract with LLM if no clauses are found by rules
        if not rule_based_clauses:
            logger.info("[clause_extraction] No clauses found by rule-based extraction. Using LLM.")
            llm_clauses = await _extract_with_llm(text)
            return llm_clauses if llm_clauses else {}

        return rule_based_clauses

    except Exception as e:
        logger.exception(f"[clause_extraction] error: {e}")
        return {"error": "Clause extraction failed"}