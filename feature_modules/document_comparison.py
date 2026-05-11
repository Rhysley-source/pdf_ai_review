"""
document_comparison.py — v3

Direct side-by-side comparison pipeline:
    compare_documents(extraction1, extraction2, text1, text2)
        → sends both raw document texts to the LLM simultaneously
        → LLM returns every difference with exact Doc1 and Doc2 text side by side
        → no clause extraction, no fuzzy matching, no risk scoring

Response shape:
    {
        "differences": [
            {
                "section":     "Payment Terms",
                "doc1_text":   "<verbatim text from Document 1>",
                "doc2_text":   "<verbatim text from Document 2>",
                "change_type": "modified | added | removed",
                "summary":     "<one-sentence description of the change>"
            }
        ],
        "total_differences": N,
        "insights": {"semantic_insights": [...], "recommendation": "..."}
    }
"""

import logging
import re
import time
from datetime import datetime, timezone

from llm_model.ai_model import run_llm_comparison
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)

_VALID_CHANGE_TYPES = {"modified", "added", "removed"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


_COMPARE_SYSTEM = """You are a document comparison expert.

Compare DOCUMENT 1 and DOCUMENT 2 and find EVERY difference between them.
Include all changes — wording, values, names, dates, terms, additions, removals.

Return ONLY this JSON:

{
  "differences": [
    {
      "section": "<topic or section name — plain text, e.g. 'Salary', 'Notice Period', 'Skills'>",
      "doc1_text": "<verbatim quote from Document 1 for this difference — empty string if absent in Doc1>",
      "doc2_text": "<verbatim quote from Document 2 for this difference — empty string if absent in Doc2>",
      "change_type": "modified | added | removed",
      "summary": "<one sentence: exactly what changed and why it matters>"
    }
  ],
  "semantic_insights": [
    "<key observation with exact quoted values from the documents>",
    "<another observation on a different topic>",
    "<which party benefits from these changes and why — name specific sections>"
  ],
  "recommendation": "<2-3 actionable sentences naming specific sections to address>"
}

Rules:
- Find EVERY difference — do not skip any change, even minor wording or single-word changes
- doc1_text and doc2_text must be verbatim quotes from the documents, not paraphrases or summaries
- "modified": both doc1_text and doc2_text must be non-empty
- "added":    doc1_text = "" (empty string); doc2_text = the new content verbatim from Document 2
- "removed":  doc1_text = the removed content verbatim from Document 1; doc2_text = "" (empty string)
- Each entry must cover exactly one topic/section — do NOT merge unrelated differences into one entry
- Do NOT include sections that are identical in both documents
- Return ONLY valid JSON — no markdown, no explanation"""


async def compare_documents(
    extraction1: dict,
    extraction2: dict,
    text1: str,
    text2: str,
    *,
    doc1_filename: str = "document_1.pdf",
    doc2_filename: str = "document_2.pdf",
    session_id: str = "",
) -> dict:
    """
    Compare two documents by sending both full texts to the LLM simultaneously.
    Returns a flat list of differences, each with verbatim Doc1 and Doc2 text
    for direct side-by-side display.
    """
    t_start = time.perf_counter()
    logger.info(f"[comparison] Starting — doc1={doc1_filename} | doc2={doc2_filename}")

    prompt = (
        f'DOCUMENT 1 — "{doc1_filename}":\n\n'
        f"{text1[:20000]}\n\n"
        f"{'─' * 60}\n\n"
        f'DOCUMENT 2 — "{doc2_filename}":\n\n'
        f"{text2[:20000]}\n\n"
        "Find and return ALL differences between Document 1 and Document 2."
    )

    result: dict = {}
    for attempt in range(1, 3):
        try:
            raw    = await run_llm_comparison(prompt, _COMPARE_SYSTEM, max_output_tokens=16000)
            result = extract_json_from_text(raw) or {}
            if isinstance(result.get("differences"), list):
                logger.info(
                    f"[comparison] LLM OK — "
                    f"{len(result['differences'])} difference(s) on attempt {attempt}"
                )
                break
            logger.warning(
                f"[comparison] attempt {attempt} — no differences list; "
                f"raw[:200]={raw[:200]}"
            )
        except Exception as e:
            logger.error(f"[comparison] attempt {attempt} error: {e}")

    differences: list[dict] = []
    for item in (result.get("differences") or []):
        if not isinstance(item, dict):
            continue
        section = str(item.get("section") or "").strip()
        if not section:
            continue
        change_type = str(item.get("change_type") or "modified").lower().strip()
        if change_type not in _VALID_CHANGE_TYPES:
            change_type = "modified"
        differences.append({
            "section":     section,
            "doc1_text":   str(item.get("doc1_text") or ""),
            "doc2_text":   str(item.get("doc2_text") or ""),
            "change_type": change_type,
            "summary":     str(item.get("summary") or ""),
        })

    doc1_type   = extraction1.get("document_type", "")
    doc2_type   = extraction2.get("document_type", "")
    types_match = _norm(doc1_type) == _norm(doc2_type)
    comparison_notice = (
        f"Both documents are of the same type ({doc1_type}). Comparison results are reliable."
        if types_match else
        f"Warning: Document types differ — '{doc1_type}' vs '{doc2_type}'. "
        "Results may be incomplete."
    )

    duration_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(f"[comparison] Done — {duration_ms}ms | {len(differences)} difference(s)")

    return {
        "status":            "success",
        "duration_ms":       duration_ms,
        "compared_at":       datetime.now(timezone.utc).isoformat(),
        "document_1_type":   doc1_type,
        "document_2_type":   doc2_type,
        "comparison_notice": comparison_notice,
        "total_differences": len(differences),
        "differences":       differences,
        "insights": {
            "semantic_insights": result.get("semantic_insights") or [],
            "recommendation":    result.get("recommendation") or "",
        },
    }
