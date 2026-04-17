"""
clause_extraction.py

Extracts named legal clauses from document text.

Key improvements vs. previous version:
  - Doc-type-aware: selects the correct prompt from clause_extraction_prompts.py
    based on the document type (contract, nda, employment, lease, invoice, other)
  - Role-persona prompts per document type (borrowed from prompts.py pattern)
  - Required / optional key separation in every prompt
  - "Not Specified" default — LLM never omits a key
  - Focus hint injected per doc type (borrowed from obligation_detection.py pattern)
  - Token tracking returned to caller
  - Retry with strengthened language if extraction is weak
  - Rule-based is a last-resort fallback only
"""

import logging
import re
from typing import Dict, Tuple

from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json
from feature_modules.clause_extraction_prompts import (
    build_extraction_prompt,
    DOC_TYPE_TO_PROMPT_KEY,
    CLAUSE_EXTRACTION_PROMPTS,
    CLAUSE_FOCUS_HINTS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema: all valid clause keys across all document types
# ---------------------------------------------------------------------------

ALL_CLAUSE_KEYS: frozenset = frozenset({
    "payment_terms", "contract_term", "renewal_conditions", "termination",
    "liability", "indemnification", "governing_law", "dispute_resolution",
    "force_majeure", "confidentiality", "intellectual_property", "non_compete",
    "non_solicitation", "scope_of_work", "warranties", "penalties", "pricing",
    "delivery_timeline", "data_protection", "assignment", "entire_agreement",
    "permitted_disclosures", "return_of_materials", "residuals",
    "probation_period", "notice_period", "benefits", "leave_policy",
    "security_deposit", "rent_escalation", "maintenance", "subletting",
    "permitted_use", "entry_by_landlord",
    "payment_due_date", "late_payment_penalty", "tax",
})

MIN_CLAUSE_VALUE_LENGTH = 80  # filters sentence fragments from LLM output

# Words that legitimately start a clause value (first word lowercase is OK)
_OK_CLAUSE_STARTERS = {
    "the", "a", "an", "all", "any", "each", "either", "no", "in", "on",
    "upon", "during", "for", "if", "when", "where", "this", "such",
    "subject", "notwithstanding", "provided", "unless", "except",
}


def _is_fragment(v: str) -> bool:
    """
    Return True if the value looks like a mid-sentence fragment rather than a
    complete clause. Fragments typically start with a lowercase word that cannot
    begin a sentence (e.g. 'connection with this Agreement...',
    'otherwise communicated...', 'policies, and codes of...').
    """
    first_char = v[0] if v else ""
    if not first_char.islower():
        return False  # starts with uppercase, digit, or punctuation — looks fine
    first_word = re.split(r"[\s,;]", v)[0].lower()
    return first_word not in _OK_CLAUSE_STARTERS


# ---------------------------------------------------------------------------
# LLM extraction — primary strategy
# ---------------------------------------------------------------------------

async def _extract_with_llm(
    text: str,
    doc_type: str,
    retry: bool = False,
) -> Tuple[Dict[str, str], int, int]:
    if retry:
        prompt_key = DOC_TYPE_TO_PROMPT_KEY.get(doc_type.lower(), "other")
        base  = CLAUSE_EXTRACTION_PROMPTS.get(prompt_key, CLAUSE_EXTRACTION_PROMPTS["other"])
        focus = CLAUSE_FOCUS_HINTS.get(prompt_key, CLAUSE_FOCUS_HINTS["other"])
        prompt = (
            f"{base}\n\n"
            "IMPORTANT: Be EXTREMELY thorough. Every clause that exists in the document "
            "MUST be extracted. Do not return 'Not Specified' unless the clause is "
            "genuinely absent. Re-read the document carefully before responding.\n\n"
            f"EXTRACTION FOCUS:\n{focus}\n\n"
            f"DOCUMENT TEXT:\n---\n{text[:6000]}\n---"
        )
    else:
        prompt = build_extraction_prompt(text, doc_type)

    result = await run_llm("", prompt)

    # Handle both (content, in, out) tuple and plain str signatures
    if isinstance(result, tuple):
        raw, in_tok, out_tok = result
    else:
        raw, in_tok, out_tok = result, 0, 0

    parsed = extract_json(raw)
    if not parsed:
        return {}, in_tok, out_tok

    clean = {
        k: v.strip()
        for k, v in parsed.items()
        if (
            k in ALL_CLAUSE_KEYS
            and isinstance(v, str)
            and v.strip()
            and v.strip().lower() not in ("not specified", "n/a", "none", "")
            and len(v.strip()) >= MIN_CLAUSE_VALUE_LENGTH
            and not _is_fragment(v.strip())
        )
    }
    return clean, in_tok, out_tok


# ---------------------------------------------------------------------------
# Rule-based fallback — LAST RESORT only
# ---------------------------------------------------------------------------

def _is_clause_heading(line: str) -> bool:
    return bool(
        re.match(r"^\d+[\.\)]\s+.{3,}", line)
        or (line.isupper() and len(line.strip()) > 4)
    )


def _normalise_heading(heading: str):
    h = re.sub(r"^\d+[\.\)]\s*", "", heading).lower()
    h = re.sub(r"[^a-z0-9]+", "_", h).strip("_")
    if h in ALL_CLAUSE_KEYS:
        return h
    for key in ALL_CLAUSE_KEYS:
        parts = [p for p in key.split("_") if len(p) > 3]
        if any(p in h for p in parts):
            return key
    return None


def _extract_rule_based(text: str) -> Dict[str, str]:
    clauses: Dict[str, str] = {}
    current_key = None
    buffer: list = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_clause_heading(line):
            if current_key and buffer:
                value = " ".join(buffer).strip()
                if len(value) >= MIN_CLAUSE_VALUE_LENGTH:
                    clauses[current_key] = value
            current_key = _normalise_heading(line)
            buffer = []
        elif current_key:
            buffer.append(line)

    if current_key and buffer:
        value = " ".join(buffer).strip()
        if len(value) >= MIN_CLAUSE_VALUE_LENGTH:
            clauses[current_key] = value

    return clauses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_clauses(
    text: str,
    doc_type: str = "other",
) -> Tuple[Dict[str, str], int, int]:
    """
    Extract legal clauses from document text using a doc-type-aware LLM prompt.

    Parameters
    ----------
    text     : raw document text
    doc_type : from classify_document() — drives prompt selection

    Returns
    -------
    (clauses_dict, input_tokens, output_tokens)

    Strategy: LLM attempt 1 → LLM retry (stronger) → rule-based fallback
    """
    total_in = total_out = 0

    try:
        # Attempt 1: doc-type-aware LLM
        clauses, in1, out1 = await _extract_with_llm(text, doc_type, retry=False)
        total_in += in1; total_out += out1

        if len(clauses) >= 2:
            logger.info(
                f"[clause_extraction] LLM ({doc_type}) → {len(clauses)} clauses: "
                + ", ".join(clauses.keys())
            )
            return clauses, total_in, total_out

        # Attempt 2: retry with stronger language
        logger.warning(
            f"[clause_extraction] Attempt 1 returned {len(clauses)} clause(s) "
            f"(doc_type='{doc_type}') — retrying"
        )
        clauses_r, in2, out2 = await _extract_with_llm(text, doc_type, retry=True)
        total_in += in2; total_out += out2

        if len(clauses_r) >= len(clauses):
            clauses = clauses_r

        if clauses:
            logger.info(f"[clause_extraction] Retry → {len(clauses)} clauses")
            return clauses, total_in, total_out

        # Attempt 3: rule-based fallback
        logger.warning("[clause_extraction] LLM failed — rule-based fallback")
        clauses = _extract_rule_based(text)
        logger.info(f"[clause_extraction] Rule-based → {len(clauses)} clauses")
        return clauses, total_in, total_out

    except Exception as e:
        logger.exception(f"[clause_extraction] Error: {e}")
        return {}, total_in, total_out