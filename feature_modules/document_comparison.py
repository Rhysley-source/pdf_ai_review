"""
document_comparison.py — v4

Side-by-side diff pipeline:
    compare_documents(extraction1, extraction2, text1, text2)
        → word-level diff via difflib.SequenceMatcher  (no LLM, instant)
        → LLM semantic insights on just the changed portions
        → returns diff_blocks covering the COMPLETE content of both documents

diff_blocks shape (one block per contiguous equal/changed span):
    {"type": "equal",   "text": "..."}                          unchanged in both
    {"type": "replace", "doc1_text": "...", "doc2_text": "..."}  modified
    {"type": "insert",  "text": "..."}                          added only in Doc2
    {"type": "delete",  "text": "..."}                          removed from Doc1

Frontend rendering:
  Doc1 column → render "equal" + "delete" + replace.doc1_text   (red highlight on delete/replace)
  Doc2 column → render "equal" + "insert" + replace.doc2_text   (green highlight on insert/replace)
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher

from llm_model.ai_model import run_llm_comparison
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


# ---------------------------------------------------------------------------
# Word-level diff — returns diff_blocks covering complete content
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Split text into word + whitespace tokens so join() reproduces the original."""
    return re.findall(r"\S+|\s+", text or "")


def _build_diff_blocks(text1: str, text2: str) -> tuple[list[dict], dict]:
    """
    Word-level diff of text1 vs text2.

    Returns:
        diff_blocks — list of {"type": ..., ...} dicts (see module docstring)
        stats       — {"added_words", "removed_words", "similarity_percent"}
    """
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)

    sm = SequenceMatcher(None, tokens1, tokens2, autojunk=False)
    blocks: list[dict] = []
    added_words = removed_words = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            blocks.append({"type": "equal", "text": "".join(tokens1[i1:i2])})
        elif tag == "delete":
            t = "".join(tokens1[i1:i2])
            blocks.append({"type": "delete", "text": t})
            removed_words += sum(1 for tok in tokens1[i1:i2] if tok.strip())
        elif tag == "insert":
            t = "".join(tokens2[j1:j2])
            blocks.append({"type": "insert", "text": t})
            added_words += sum(1 for tok in tokens2[j1:j2] if tok.strip())
        elif tag == "replace":
            d1 = "".join(tokens1[i1:i2])
            d2 = "".join(tokens2[j1:j2])
            blocks.append({"type": "replace", "doc1_text": d1, "doc2_text": d2})
            removed_words += sum(1 for tok in tokens1[i1:i2] if tok.strip())
            added_words   += sum(1 for tok in tokens2[j1:j2] if tok.strip())

    ratio = sm.ratio()
    stats = {
        "added_words":        added_words,
        "removed_words":      removed_words,
        "similarity_percent": f"{round(ratio * 100, 1)}%",
        "similarity_score":   round(ratio, 4),
    }
    return blocks, stats


# ---------------------------------------------------------------------------
# LLM semantic insights — only on the changed spans, not full documents
# ---------------------------------------------------------------------------

_INSIGHTS_SYSTEM = (
    "You are a senior document analyst and legal reviewer. "
    "Analyse the changes between two versions of the same document type. "
    "Be specific — quote exact values (amounts, dates, durations, party names) that changed. "
    "Return ONLY valid JSON — no markdown, no explanation."
)

_MAX_CHANGE_CHARS = 8000


async def _get_insights(
    diff_blocks: list[dict],
    doc1_filename: str,
    doc2_filename: str,
) -> dict:
    empty = {"semantic_insights": [], "recommendation": ""}

    changed = [b for b in diff_blocks if b["type"] != "equal"]
    if not changed:
        return empty

    lines: list[str] = []
    total_chars = 0
    for i, b in enumerate(changed, 1):
        if b["type"] == "replace":
            line = f"{i}. MODIFIED — Doc1: {b['doc1_text'][:300].strip()} | Doc2: {b['doc2_text'][:300].strip()}"
        elif b["type"] == "insert":
            line = f"{i}. ADDED in Doc2 — {b['text'][:300].strip()}"
        else:
            line = f"{i}. REMOVED from Doc1 — {b['text'][:300].strip()}"
        lines.append(line)
        total_chars += len(line)
        if total_chars >= _MAX_CHANGE_CHARS:
            lines.append(f"... ({len(changed) - i} more changes not shown)")
            break

    prompt = (
        f'Changes between "{doc1_filename}" (Doc1) and "{doc2_filename}" (Doc2):\n\n'
        + "\n".join(lines)
        + '\n\nAnalyse these changes thoroughly and return ONLY this JSON:\n'
        '{\n'
        '  "semantic_insights": [\n'
        '    "<financial changes — exact amounts, rates, fees, or payment terms that changed>",\n'
        '    "<timeline changes — dates, durations, notice periods, or deadlines that changed>",\n'
        '    "<obligation changes — new duties added or existing duties removed for either party>",\n'
        '    "<risk changes — clauses that increase or decrease legal or financial risk>",\n'
        '    "<protection changes — clauses added or removed that protect one party>",\n'
        '    "<which party benefits most from Doc2 changes and why>",\n'
        '    "<any missing standard clauses or red flags introduced by the changes>"\n'
        '  ],\n'
        '  "recommendation": "<3-4 actionable sentences: what to negotiate, what to accept, what to reject>"\n'
        '}'
    )

    try:
        raw    = await run_llm_comparison(prompt, _INSIGHTS_SYSTEM, max_output_tokens=2000)
        result = extract_json_from_text(raw) or {}
        if result.get("semantic_insights"):
            return {
                "semantic_insights": result.get("semantic_insights") or [],
                "recommendation":    result.get("recommendation") or "",
            }
    except Exception as e:
        logger.error(f"[comparison] insights LLM error: {e}")

    return empty


# ---------------------------------------------------------------------------
# Incompatibility analysis — description + structured insights for mismatched types
# ---------------------------------------------------------------------------

_INCOMPATIBILITY_SYSTEM = (
    "You are a senior document analyst. "
    "Analyse two documents of different types and explain why they cannot be compared. "
    "Return ONLY valid JSON — no markdown, no explanation."
)


async def get_incompatibility_insights(
    doc1_type: str,
    doc2_type: str,
    doc1_filename: str,
    doc2_filename: str,
    clauses1: list,
    clauses2: list,
) -> tuple[str, dict]:
    """
    Returns (description: str, insights: dict) for incompatible document pairs.
    insights has the same shape as compatible-comparison insights:
      {"semantic_insights": [...], "recommendation": "..."}
    Both are generated in a single LLM call.
    """
    clauses1_text = "\n".join(
        f"- {c['clause_name']}: {c['excerpt'][:120]}" for c in clauses1[:8]
    )
    clauses2_text = "\n".join(
        f"- {c['clause_name']}: {c['excerpt'][:120]}" for c in clauses2[:8]
    )

    prompt = (
        f'Document 1: "{doc1_filename}" — type: {doc1_type}\n'
        f'Key clauses:\n{clauses1_text}\n\n'
        f'Document 2: "{doc2_filename}" — type: {doc2_type}\n'
        f'Key clauses:\n{clauses2_text}\n\n'
        'Return ONLY this JSON:\n'
        '{\n'
        '  "description": "<2-3 sentences: what each document is, why they cannot be compared>",\n'
        '  "semantic_insights": [\n'
        '    "<what Document 1 covers and its key obligations>",\n'
        '    "<what Document 2 covers and its key obligations>",\n'
        '    "<key clauses present in Doc 1 that are absent in Doc 2>",\n'
        '    "<key clauses present in Doc 2 that are absent in Doc 1>",\n'
        '    "<overall risk or concern from mixing these document types>"\n'
        '  ],\n'
        '  "recommendation": "<2-3 actionable sentences on what the user should do instead>"\n'
        '}'
    )

    empty_insights: dict = {"semantic_insights": [], "recommendation": ""}
    try:
        raw    = await run_llm_comparison(prompt, _INCOMPATIBILITY_SYSTEM, max_output_tokens=800)
        result = extract_json_from_text(raw) or {}
        description = (result.get("description") or "").strip()
        insights    = {
            "semantic_insights": result.get("semantic_insights") or [],
            "recommendation":    result.get("recommendation") or "",
        }
        if not description:
            description = (
                f"'{doc1_type}' and '{doc2_type}' serve entirely different legal purposes. "
                f"Their clauses, obligations, and structure are unrelated, making a direct "
                f"comparison unreliable and potentially misleading."
            )
        return description, insights
    except Exception as e:
        logger.error(f"[comparison] incompatibility insights failed: {e}")
        fallback = (
            f"'{doc1_type}' and '{doc2_type}' serve entirely different legal purposes. "
            f"Their clauses, obligations, and structure are unrelated, making a direct "
            f"comparison unreliable and potentially misleading."
        )
        return fallback, empty_insights


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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
    Side-by-side diff of two documents.

    Returns diff_blocks covering the COMPLETE content of both documents so the
    frontend can render a full highlighted view:
      - Doc1 column: equal + delete + replace.doc1_text  (red on delete/replace)
      - Doc2 column: equal + insert + replace.doc2_text  (green on insert/replace)
    """
    t_start = time.perf_counter()
    logger.info(f"[comparison] Starting — doc1={doc1_filename} | doc2={doc2_filename}")

    # 1. Word-level diff (pure Python, no LLM)
    diff_blocks, stats = await asyncio.to_thread(_build_diff_blocks, text1, text2)

    changed_count = sum(1 for b in diff_blocks if b["type"] != "equal")
    logger.info(
        f"[comparison] Diff done — "
        f"{len(diff_blocks)} blocks | {changed_count} changed | "
        f"similarity={stats['similarity_percent']}"
    )

    # 2. LLM insights only on the changed spans
    insights = await _get_insights(diff_blocks, doc1_filename, doc2_filename)

    # 3. Document type / compatibility notice
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
    logger.info(f"[comparison] Done — {duration_ms}ms")

    return {
        "status":            "success",
        "duration_ms":       duration_ms,
        "compared_at":       datetime.now(timezone.utc).isoformat(),
        "document_1_type":   doc1_type,
        "document_2_type":   doc2_type,
        "comparison_notice": comparison_notice,
        "stats":             stats,
        "diff_blocks":       diff_blocks,
        "insights":          insights,
    }
