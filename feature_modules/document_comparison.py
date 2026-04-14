"""
document_comparison.py  — v2

Simplified pipeline:
    extract_key_clauses(text1) + extract_key_clauses(text2)   [parallel, called from route]
        → compare_documents(result1, result2)
            → fuzzy clause matching
            → word-level diff per clause
            → severity × change_type risk scoring
            → LLM enrichment (summaries + insights + recommendation)

Response shape per clause:
    {
        "clause_name": "Payment Terms",
        "status":      "modified",          # added | removed | modified
        "severity":    "high",
        "doc1":        {"excerpt": "...", "significance": "..."},
        "doc2":        {"excerpt": "...", "significance": "..."},
        "word_diff":   [{"text": "word", "tag": "equal|insert|delete"}, ...],
        "summary":     "LLM one-liner about what changed and why it matters."
    }
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher

from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk tables
# ---------------------------------------------------------------------------

HIGH_RISK_NAMES = {
    "payment", "liability", "termination", "indemnif", "penalt",
    "pricing", "intellectual property", "ip rights", "ip ownership",
    "liquidated damages", "damages",
}
MEDIUM_RISK_NAMES = {
    "term", "renewal", "scope", "deliverable", "timeline", "deadline",
    "non-compete", "non compete", "warranty", "warranties", "force majeure",
    "data protection", "privacy", "confidential",
}

# Weight matrix: severity × change_type → risk points
_RISK_WEIGHTS = {
    ("high",   "removed"):  40,
    ("high",   "modified"): 30,
    ("high",   "added"):    20,
    ("medium", "removed"):  20,
    ("medium", "modified"): 15,
    ("medium", "added"):    10,
    ("low",    "removed"):  10,
    ("low",    "modified"):  5,
    ("low",    "added"):     3,
}


def _severity(clause_name: str) -> str:
    n = clause_name.lower()
    if any(k in n for k in HIGH_RISK_NAMES):
        return "high"
    if any(k in n for k in MEDIUM_RISK_NAMES):
        return "medium"
    return "low"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# ---------------------------------------------------------------------------
# Clause matching
# Matches clauses from doc2 → doc1 by name similarity + excerpt similarity.
# Returns list of (clause1_or_None, clause2_or_None, match_score).
# ---------------------------------------------------------------------------

_NAME_WEIGHT    = 0.6
_EXCERPT_WEIGHT = 0.4
_MATCH_THRESHOLD = 0.45


def _match_clauses(
    clauses1: list[dict],
    clauses2: list[dict],
) -> list[tuple[dict | None, dict | None]]:
    """
    Greedy best-match pairing between two clause lists.
    Each clause from doc1 is paired with at most one clause from doc2.
    Unmatched clauses from either side are appended as added/removed.
    """
    used2 = set()
    pairs: list[tuple[dict | None, dict | None]] = []

    for c1 in clauses1:
        best_idx, best_score = -1, 0.0
        for i, c2 in enumerate(clauses2):
            if i in used2:
                continue
            score = (
                _NAME_WEIGHT    * _sim(c1["clause_name"], c2["clause_name"]) +
                _EXCERPT_WEIGHT * _sim(c1.get("excerpt", ""), c2.get("excerpt", ""))
            )
            if score > best_score:
                best_score, best_idx = score, i

        if best_idx >= 0 and best_score >= _MATCH_THRESHOLD:
            used2.add(best_idx)
            pairs.append((c1, clauses2[best_idx]))
        else:
            pairs.append((c1, None))  # removed in doc2

    for i, c2 in enumerate(clauses2):
        if i not in used2:
            pairs.append((None, c2))  # added in doc2

    return pairs


# ---------------------------------------------------------------------------
# Word-level diff
# Operates on the excerpt fields of the paired clauses.
# Returns a list of {"text": str, "tag": "equal"|"insert"|"delete"}.
# "insert" = present in doc2 only, "delete" = present in doc1 only.
# ---------------------------------------------------------------------------

def _word_diff(text1: str, text2: str) -> list[dict]:
    words1 = (text1 or "").split()
    words2 = (text2 or "").split()

    sm  = SequenceMatcher(None, words1, words2, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for w in words1[i1:i2]:
                out.append({"text": w, "tag": "equal"})
        elif tag == "insert":
            for w in words2[j1:j2]:
                out.append({"text": w, "tag": "insert"})
        elif tag == "delete":
            for w in words1[i1:i2]:
                out.append({"text": w, "tag": "delete"})
        elif tag == "replace":
            for w in words1[i1:i2]:
                out.append({"text": w, "tag": "delete"})
            for w in words2[j1:j2]:
                out.append({"text": w, "tag": "insert"})
    return out


# ---------------------------------------------------------------------------
# Build raw clause changes from matched pairs
# ---------------------------------------------------------------------------

def _build_raw_changes(
    pairs: list[tuple[dict | None, dict | None]],
) -> list[dict]:
    changes = []
    for c1, c2 in pairs:
        if c1 is None and c2 is None:
            continue

        name     = (c1 or c2)["clause_name"]
        severity = _severity(name)

        if c1 is None:
            status = "added"
        elif c2 is None:
            status = "removed"
        else:
            # Both present — check if content actually changed
            ratio = _sim(c1.get("excerpt", ""), c2.get("excerpt", ""))
            if ratio > 0.97:
                continue  # identical — skip
            status = "modified"

        excerpt1 = (c1 or {}).get("excerpt", "")
        excerpt2 = (c2 or {}).get("excerpt", "")

        changes.append({
            "clause_name": name,
            "status":      status,
            "severity":    severity,
            "doc1": {
                "excerpt":      excerpt1,
                "significance": (c1 or {}).get("significance", ""),
            },
            "doc2": {
                "excerpt":      excerpt2,
                "significance": (c2 or {}).get("significance", ""),
            },
            "word_diff": _word_diff(excerpt1, excerpt2),
            "summary":   "",  # filled by LLM enrichment
        })
    return changes


# ---------------------------------------------------------------------------
# Risk scoring (severity × change_type weighted)
# ---------------------------------------------------------------------------

def _risk_score(changes: list[dict]) -> dict:
    score = 0
    high_count = 0
    for c in changes:
        pts = _RISK_WEIGHTS.get((c["severity"], c["status"]), 0)
        score += pts
        if c["severity"] == "high":
            high_count += 1

    score = min(score, 100)
    level = "high" if score >= 70 else "medium" if score >= 30 else "low"
    return {
        "risk_score":        score,
        "overall_risk_level": level,
        "high_risk_changes": high_count,
    }


# ---------------------------------------------------------------------------
# Text-level diff stats (line-based, for the summary block)
# ---------------------------------------------------------------------------

def _text_diff_stats(text1: str, text2: str) -> dict:
    lines1, lines2 = text1.splitlines(), text2.splitlines()
    sm = SequenceMatcher(None, lines1, lines2)
    added = removed = changed = unchanged = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            unchanged += i2 - i1
        elif tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "replace":
            ch = min(i2 - i1, j2 - j1)
            changed += ch
            removed += (i2 - i1) - ch
            added   += (j2 - j1) - ch
    ratio = round(sm.ratio(), 4)
    return {
        "lines_added":       added,
        "lines_removed":     removed,
        "lines_changed":     changed,
        "lines_unchanged":   unchanged,
        "similarity_score":  ratio,
        "similarity_percent": f"{round(ratio * 100, 1)}%",
    }


# ---------------------------------------------------------------------------
# LLM enrichment — per-clause summaries + insights + recommendation
# ---------------------------------------------------------------------------

def _build_enrichment_prompt(changes: list[dict], doc1_text: str, doc2_text: str) -> str:
    lines = []
    for i, c in enumerate(changes[:20], 1):
        e1 = (c["doc1"].get("excerpt") or "")[:300].replace("\n", " ")
        e2 = (c["doc2"].get("excerpt") or "")[:300].replace("\n", " ")
        lines.append(
            f"{i}. CLAUSE: {c['clause_name'].upper()}\n"
            f"   Status: {c['status']} | Severity: {c['severity']}\n"
            f"   Doc1 (original) : {e1 or '[absent]'}\n"
            f"   Doc2 (revised)  : {e2 or '[absent]'}"
        )
    changes_block = "\n\n".join(lines) or "No changes detected."

    return f"""You are a senior legal analyst specialising in contract risk review.

DETECTED CLAUSE CHANGES ({len(changes)} total):
{changes_block}

DOCUMENT 1 EXCERPT (original):
{doc1_text[:1200]}

DOCUMENT 2 EXCERPT (revised):
{doc2_text[:1200]}

Return ONLY this JSON — no markdown, no explanation:

{{
  "clause_summaries": {{
    "<clause_name as written above>": "<One sentence: what changed and the business impact>",
    "...one entry per clause listed above..."
  }},
  "semantic_insights": [
    "<Specific, quantified insight naming exact values/dates/%. E.g. Payment window cut from Net 30 to Net 15, doubling cash-flow pressure.>",
    "<Another insight on a different clause>",
    "<Minimum 3 insights — more if there are more changes>",
    "<FINAL insight: which party does the revised contract favour and why — name 2-3 specific reasons.>"
  ],
  "recommendation": "<2-3 actionable sentences naming specific clauses to push back on and the target outcome.>"
}}

Rules:
- clause_summaries key must exactly match the clause_name from the change list
- Each semantic_insight must reference specific values or clause text — no vague generalities
- Last insight must name the favoured party with evidence
- recommendation must name at least 2 specific clauses"""


async def _llm_enrichment(changes: list[dict], text1: str, text2: str) -> dict:
    empty = {"clause_summaries": {}, "semantic_insights": [], "recommendation": ""}
    if not changes:
        return empty

    try:
        prompt = _build_enrichment_prompt(changes, text1, text2)
        raw = await run_llm("", prompt)
        if isinstance(raw, tuple):
            raw = raw[0]

        result = extract_json_from_text(raw)
        if result and (result.get("clause_summaries") or result.get("semantic_insights")):
            logger.info(
                f"[comparison] LLM OK — "
                f"summaries={len(result.get('clause_summaries', {}))} "
                f"insights={len(result.get('semantic_insights', []))}"
            )
            return result

        # Retry with a worked example to guide format
        logger.warning("[comparison] LLM attempt 1 weak — retrying with example")
        ex      = changes[0]
        ex_name = ex["clause_name"]
        ex_e1   = (ex["doc1"].get("excerpt") or "original text")[:60]
        ex_e2   = (ex["doc2"].get("excerpt") or "revised text")[:60]

        retry_prompt = (
            "You are a senior legal analyst. Return ONLY valid JSON — no markdown.\n\n"
            "Required format example:\n"
            "{\n"
            f'  "clause_summaries": {{"{ex_name}": '
            f'"{ex_name} changed from \\"{ex_e1[:40]}...\\" to \\"{ex_e2[:40]}...\\" — increases risk for recipient."}},\n'
            '  "semantic_insights": [\n'
            f'    "{ex_name} shifted — original: {ex_e1} / revised: {ex_e2}. Impact: increases financial exposure.",\n'
            '    "State which party benefits overall and provide 2 specific reasons."\n'
            '  ],\n'
            '  "recommendation": "Negotiate to restore [clause] and [clause] before signing."\n'
            "}\n\n"
            "Now produce the real analysis:\n\n"
            + prompt
        )

        raw2 = await run_llm("", retry_prompt)
        if isinstance(raw2, tuple):
            raw2 = raw2[0]
        result2 = extract_json_from_text(raw2)
        if result2:
            logger.info("[comparison] Retry succeeded")
            return result2

        logger.error("[comparison] Both LLM enrichment attempts failed — returning empty")
        return empty

    except Exception as e:
        logger.exception(f"[comparison] Enrichment error: {e}")
        return empty


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
    Compare two documents given their extract_key_clauses() results.

    Parameters
    ----------
    extraction1 : output of extract_key_clauses(text1)
    extraction2 : output of extract_key_clauses(text2)
    text1       : raw text of document 1 (for text-diff stats + LLM context)
    text2       : raw text of document 2
    doc1_filename, doc2_filename : original filenames for display
    session_id  : caller-supplied session identifier

    Returns
    -------
    Structured comparison dict — see module docstring for shape.
    """
    t_start = time.perf_counter()

    clauses1: list[dict] = extraction1.get("key_clauses", [])
    clauses2: list[dict] = extraction2.get("key_clauses", [])

    logger.info(
        f"[comparison] Starting — "
        f"doc1={doc1_filename} ({len(clauses1)} clauses) | "
        f"doc2={doc2_filename} ({len(clauses2)} clauses)"
    )

    # 1. Match clauses across the two documents
    pairs = _match_clauses(clauses1, clauses2)

    # 2. Build raw change list + text diff stats (CPU-bound, run in thread)
    raw_changes, diff_stats = await asyncio.gather(
        asyncio.to_thread(_build_raw_changes, pairs),
        asyncio.to_thread(_text_diff_stats, text1, text2),
    )

    # 3. Risk score
    risk = _risk_score(raw_changes)

    # 4. LLM enrichment (per-clause summaries, insights, recommendation)
    llm_data = await _llm_enrichment(raw_changes, text1, text2)

    summaries = llm_data.get("clause_summaries", {})
    insights  = llm_data.get("semantic_insights", [])
    rec       = llm_data.get("recommendation", "")

    # 5. Attach LLM summaries to each clause change
    clause_changes = []
    for c in raw_changes:
        name    = c["clause_name"]
        summary = summaries.get(name, "").strip()
        if not summary:
            # Fallback: deterministic summary from structured fields
            if c["status"] == "added":
                summary = f"{name} is a new clause added in the revised document."
            elif c["status"] == "removed":
                summary = f"{name} has been removed from the revised document."
            else:
                summary = f"{name} has been modified in the revised document."

        clause_changes.append({
            "clause_name": name,
            "status":      c["status"],
            "severity":    c["severity"],
            "doc1":        c["doc1"],
            "doc2":        c["doc2"],
            "word_diff":   c["word_diff"],
            "summary":     summary,
        })

    duration_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        f"[comparison] Done — {duration_ms}ms | "
        f"changes={len(clause_changes)} | "
        f"risk={risk['overall_risk_level']} (score={risk['risk_score']})"
    )

    return {
        "status":      "success",
        "duration_ms": duration_ms,
        "comparison": {
            "session_id":          session_id,
            "doc1_filename":       doc1_filename,
            "doc2_filename":       doc2_filename,
            "doc1_document_type":  extraction1.get("document_type", ""),
            "doc2_document_type":  extraction2.get("document_type", ""),
            "compared_at":         datetime.now(timezone.utc).isoformat(),
            "total_changes":       len(clause_changes),
            "high_risk_changes":   risk["high_risk_changes"],
            "risk_score":          risk["risk_score"],
            "overall_risk_level":  risk["overall_risk_level"],
            "recommendation":      rec,
            "semantic_insights":   insights,
            "text_diff_stats":     diff_stats,
            "clause_changes":      clause_changes,
        },
    }