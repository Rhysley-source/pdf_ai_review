"""
clause_extraction.py

Extracts named legal clauses from document text.

Strategy:
  1. LLM-first extraction with a rich, fixed schema of clause keys.
  2. Rule-based fallback ONLY if LLM returns nothing — not the other
     way around. The old approach ran rule-based first and fell back to
     LLM only on total failure; this produced raw heading strings as
     keys (e.g. "1. Payment Terms: The client shall pay...") instead of
     normalised keys ("payment_terms"), which broke the clause-matching
     similarity scores and triggered the full_text fallback in
     compare_documents.

Schema — the LLM is asked for exactly these keys. Absent clauses get "".
That gives compare_documents stable, matchable keys across both documents.
"""

import logging
from typing import Dict

from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target clause schema — fixed keys the LLM must use.
# Extend this list to add more clause types; the prompt rebuilds itself.
# ---------------------------------------------------------------------------
CLAUSE_SCHEMA: list[str] = [
    "payment_terms",
    "contract_term",
    "renewal_conditions",
    "termination",
    "liability",
    "indemnification",
    "governing_law",
    "dispute_resolution",
    "force_majeure",
    "confidentiality",
    "intellectual_property",
    "non_compete",
    "scope_of_work",
    "warranties",
    "penalties",
    "pricing",
    "delivery_timeline",
    "data_protection",
    "assignment",
    "entire_agreement",
]

# Map each key to its display name (used in prompts and UI)
CLAUSE_LABELS: dict[str, str] = {
    "payment_terms":      "Payment Terms",
    "contract_term":      "Contract Term",
    "renewal_conditions": "Renewal Conditions",
    "termination":        "Termination",
    "liability":          "Liability",
    "indemnification":    "Indemnification",
    "governing_law":      "Governing Law",
    "dispute_resolution": "Dispute Resolution",
    "force_majeure":      "Force Majeure",
    "confidentiality":    "Confidentiality",
    "intellectual_property": "Intellectual Property",
    "non_compete":        "Non-Compete",
    "scope_of_work":      "Scope of Work",
    "warranties":         "Warranties",
    "penalties":          "Penalties",
    "pricing":            "Pricing",
    "delivery_timeline":  "Delivery Timeline",
    "data_protection":    "Data Protection",
    "assignment":         "Assignment",
    "entire_agreement":   "Entire Agreement",
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_extraction_prompt(text: str) -> str:
    schema_lines = "\n".join(f'    "{k}": ""' for k in CLAUSE_SCHEMA)
    return f"""You are a legal contract analyst. Extract the verbatim or closely paraphrased text
of each clause from the document below and return ONLY a JSON object with these exact keys.

Rules:
- Use ONLY the keys listed. Do not add extra keys.
- Value = the actual clause text found in the document (quote it or closely paraphrase it).
- If a clause is absent from the document → return "" for that key.
- Do NOT summarise or shorten clause text — preserve the original wording faithfully.
- Values must be plain strings. No nested objects or arrays.
- Return ONLY valid JSON. No preamble, no explanation.

JSON schema (return all keys):
{{
{schema_lines}
}}

Document:
---
{text[:6000]}
---"""


def _build_retry_prompt(text: str) -> str:
    """Simpler retry prompt used when first attempt fails."""
    keys = ", ".join(f'"{k}"' for k in CLAUSE_SCHEMA[:10])
    return f"""Extract legal clauses. Return ONLY JSON. Keys: {keys} (plus others if present).
Empty string for absent clauses. No explanation.

Document (first 3000 chars):
{text[:3000]}"""


# ---------------------------------------------------------------------------
# LLM extraction (primary)
# ---------------------------------------------------------------------------

async def _extract_with_llm(text: str, retry: bool = False) -> Dict[str, str]:
    prompt = _build_retry_prompt(text) if retry else _build_extraction_prompt(text)
    raw = await run_llm("", prompt)
    result = extract_json(raw)
    if not result:
        return {}
    # Keep only recognised schema keys with non-empty string values
    return {
        k: v.strip()
        for k, v in result.items()
        if k in CLAUSE_SCHEMA and isinstance(v, str) and v.strip()
    }


# ---------------------------------------------------------------------------
# Rule-based fallback (used ONLY when LLM returns nothing)
# ---------------------------------------------------------------------------

import re

def _is_clause_heading(line: str) -> bool:
    """Detect numbered headings: '1. Payment Terms' / '1) ...' / ALL CAPS."""
    return bool(
        re.match(r"^\d+[\.\)]\s+.{3,}", line)
        or (line.isupper() and len(line) > 4)
    )


def _normalise_heading_to_key(heading: str) -> str:
    """
    Map a raw heading like '1. Payment Terms' to the nearest schema key,
    e.g. 'payment_terms'. Falls back to a slugified version of the heading.
    """
    h = heading.lower()
    h = re.sub(r"^\d+[\.\)]\s*", "", h)   # strip leading '1. '
    h = re.sub(r"[^a-z0-9]+", "_", h).strip("_")

    # Try exact match first
    if h in CLAUSE_SCHEMA:
        return h

    # Partial match against schema keys
    for key in CLAUSE_SCHEMA:
        if key in h or any(part in h for part in key.split("_") if len(part) > 3):
            return key

    return h  # unknown heading — will be filtered out later


def _extract_rule_based(text: str) -> Dict[str, str]:
    clauses: Dict[str, str] = {}
    current_key = None
    buffer: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_clause_heading(line):
            if current_key and buffer:
                clauses[current_key] = " ".join(buffer).strip()
            current_key = _normalise_heading_to_key(line)
            buffer = []
        elif current_key:
            buffer.append(line)

    if current_key and buffer:
        clauses[current_key] = " ".join(buffer).strip()

    # Keep only schema keys
    return {k: v for k, v in clauses.items() if k in CLAUSE_SCHEMA and v}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_clauses(text: str) -> Dict[str, str]:
    """
    Extract legal clauses from document text.

    Returns a dict mapping schema keys (e.g. "payment_terms") to the
    clause text found in the document. Absent clauses are omitted (not
    returned as empty strings) so callers can detect what's actually present.

    Strategy: LLM-first → retry → rule-based fallback.
    Rule-based is a last resort because it produces heading-string keys
    that break the similarity matching in document_comparison.py.
    """
    try:
        # Attempt 1 — full LLM extraction
        clauses = await _extract_with_llm(text, retry=False)
        if clauses:
            logger.info(
                f"[clause_extraction] LLM extracted {len(clauses)} clause(s): "
                + ", ".join(clauses.keys())
            )
            return clauses

        # Attempt 2 — simplified retry prompt
        logger.warning("[clause_extraction] LLM attempt 1 returned nothing — retrying")
        clauses = await _extract_with_llm(text, retry=True)
        if clauses:
            logger.info(
                f"[clause_extraction] LLM retry extracted {len(clauses)} clause(s)"
            )
            return clauses

        # Attempt 3 — rule-based fallback
        logger.warning("[clause_extraction] Both LLM attempts failed — rule-based fallback")
        clauses = _extract_rule_based(text)
        if clauses:
            logger.info(
                f"[clause_extraction] Rule-based extracted {len(clauses)} clause(s)"
            )
            return clauses

        logger.error("[clause_extraction] All strategies failed — returning empty")
        return {}

    except Exception as e:
        logger.exception(f"[clause_extraction] Unhandled error: {e}")
        return {}