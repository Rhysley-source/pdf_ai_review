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

from llm_model.ai_model import run_llm_mini
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

        e1 = (c1 or {}).get("excerpt", "")
        e2 = (c2 or {}).get("excerpt", "")

        changes.append({
            "clause_name":      name,
            "status":           status,
            "severity":         severity,
            "doc1": {
                "excerpt":      e1,
                "significance": (c1 or {}).get("significance", ""),
            },
            "doc2": {
                "excerpt":      e2,
                "significance": (c2 or {}).get("significance", ""),
            },
            "word_diff":        _word_diff(e1, e2) if status == "modified" else [],
            "summary":          "",   # filled by LLM enrichment
            "difference_points": [],  # filled by LLM enrichment
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

_ENRICHMENT_SYSTEM = (
    "You are a senior legal analyst specialising in contract risk review. "
    "Analyse the clause changes provided and return a JSON object with exact difference points per clause. "
    "Return ONLY valid JSON — no markdown, no explanation."
)


def _build_enrichment_prompt(
    changes: list[dict],
    doc1_text: str,
    doc2_text: str,
    include_insights: bool = True,
) -> str:
    sorted_changes = sorted(
        changes[:20],
        key=lambda c: {"high": 0, "medium": 1, "low": 2}.get(c["severity"], 3),
    )
    lines = []
    for i, c in enumerate(sorted_changes, 1):
        e1 = (c["doc1"].get("excerpt") or "")[:500].replace("\n", " ")
        e2 = (c["doc2"].get("excerpt") or "")[:500].replace("\n", " ")
        lines.append(
            f"{i}. CLAUSE: {c['clause_name'].upper()}\n"
            f"   Status: {c['status']} | Severity: {c['severity']}\n"
            f"   Doc1 (original) : {e1 or '[absent]'}\n"
            f"   Doc2 (revised)  : {e2 or '[absent]'}"
        )

    if len(changes) > 20:
        logger.warning(f"[comparison] {len(changes)} changes — enriching top 20 by severity")

    changes_block = "\n\n".join(lines) or "No changes detected."

    if include_insights:
        json_shape = """\
{
  "clause_details": {
    "<clause_name exactly as written above>": {
      "summary": "<one sentence: what changed and the business impact>",
      "difference_points": [
        "<specific point 1 — name exact value/term that changed, e.g. 'Payment period changed from Net 30 to Net 15'>",
        "<specific point 2 — another concrete difference>",
        "<specific point 3 — add more if needed>"
      ]
    }
  },
  "semantic_insights": [
    "<quantified insight referencing exact values — e.g. 'Liability cap reduced from $500K to $100K'>",
    "<another insight on a different clause>",
    "<which party benefits from these changes and why — name 2-3 specific reasons>"
  ],
  "recommendation": "<2-3 actionable sentences naming specific clauses to negotiate and the target outcome>"
}"""
        doc_excerpt = (
            f"\nDOCUMENT 1 EXCERPT (original):\n{doc1_text[:1500]}\n\n"
            f"DOCUMENT 2 EXCERPT (revised):\n{doc2_text[:1500]}\n"
        )
    else:
        # Clause-only prompt — no insights/recommendation needed (faster, fewer tokens)
        json_shape = """\
{
  "clause_details": {
    "<clause_name exactly as written above>": {
      "summary": "<one sentence: what changed and the business impact>",
      "difference_points": [
        "<specific point 1 — name exact value/term that changed>",
        "<specific point 2 — another concrete difference>"
      ]
    }
  }
}"""
        doc_excerpt = ""

    return (
        f"DETECTED CLAUSE CHANGES ({len(sorted_changes)} shown, sorted by severity):\n"
        f"{changes_block}\n"
        f"{doc_excerpt}\n"
        f"Return ONLY this JSON:\n\n{json_shape}\n\n"
        "Rules:\n"
        "- clause_details key must exactly match the clause_name from the change list\n"
        "- difference_points must be specific and concrete — name exact values, dates, amounts, percentages\n"
        "- Every modified clause must have at least 2 difference_points\n"
        "- Added clauses: difference_points should describe what the new clause introduces\n"
        "- Removed clauses: difference_points should describe what protection is lost"
    )


async def _enrich_group(
    changes: list[dict],
    text1: str,
    text2: str,
    include_insights: bool = True,
    label: str = "",
) -> dict:
    """Run one LLM enrichment call for a subset of clause changes."""
    empty = {"clause_details": {}, "semantic_insights": [], "recommendation": ""}
    if not changes:
        return empty

    try:
        prompt = _build_enrichment_prompt(changes, text1, text2, include_insights)
        max_tokens = 4000 if include_insights else 2000

        # Attempt 1
        raw    = await run_llm_mini(prompt, _ENRICHMENT_SYSTEM, max_output_tokens=max_tokens)
        result = extract_json_from_text(raw)
        if result and result.get("clause_details"):
            logger.info(
                f"[comparison{label}] LLM OK — "
                f"clause_details={len(result.get('clause_details', {}))} "
                f"insights={len(result.get('semantic_insights', []))}"
            )
            return result

        # Attempt 2 — retry with worked example
        logger.warning(f"[comparison{label}] LLM attempt 1 weak — retrying with example")
        ex      = changes[0]
        ex_name = ex["clause_name"]
        ex_e1   = (ex["doc1"].get("excerpt") or "original text")[:60]
        ex_e2   = (ex["doc2"].get("excerpt") or "revised text")[:60]

        retry_prompt = (
            "Return ONLY valid JSON — no markdown.\n\n"
            "Required format example:\n"
            "{\n"
            f'  "clause_details": {{"{ex_name}": {{\n'
            f'    "summary": "{ex_name} changed — increases risk for signer.",\n'
            f'    "difference_points": [\n'
            f'      "Original: {ex_e1[:40]} | Revised: {ex_e2[:40]}",\n'
            f'      "Impact: financial exposure increased"\n'
            f'    ]\n'
            f'  }}}}'
        )
        if include_insights:
            retry_prompt += (
                ',\n'
                '  "semantic_insights": ["Specific change with exact values.", "Which party benefits and why."],\n'
                '  "recommendation": "Negotiate to restore [clause] before signing."\n'
            )
        retry_prompt += "}\n\nNow produce the real analysis:\n\n" + prompt

        raw2    = await run_llm_mini(retry_prompt, _ENRICHMENT_SYSTEM, max_output_tokens=max_tokens)
        result2 = extract_json_from_text(raw2)
        if result2 and result2.get("clause_details"):
            logger.info(f"[comparison{label}] Retry succeeded")
            return result2

        logger.error(f"[comparison{label}] Both LLM enrichment attempts failed — returning empty")
        return empty

    except Exception as e:
        logger.exception(f"[comparison{label}] Enrichment error: {e}")
        return empty


async def _llm_enrichment(changes: list[dict], text1: str, text2: str) -> dict:
    """
    Enrich clause changes with LLM summaries, difference_points, insights,
    and a negotiation recommendation.

    Splits changes by severity and runs two parallel LLM calls:
      - high/medium group → clause_details + semantic_insights + recommendation
      - low group         → clause_details only (smaller prompt, fewer tokens)

    Falls back to a single call when only one severity group is present.
    """
    if not changes:
        return {"clause_details": {}, "semantic_insights": [], "recommendation": ""}

    high_med = [c for c in changes if c["severity"] in ("high", "medium")]
    low      = [c for c in changes if c["severity"] == "low"]

    # Only one group present — no benefit from splitting
    if not high_med:
        return await _enrich_group(low, text1, text2, include_insights=True, label="/low-only")
    if not low:
        return await _enrich_group(high_med, text1, text2, include_insights=True, label="/high-med-only")

    # Two groups — run in parallel
    logger.info(
        f"[comparison] Parallel enrichment — "
        f"high/med={len(high_med)} clauses | low={len(low)} clauses"
    )
    high_med_result, low_result = await asyncio.gather(
        _enrich_group(high_med, text1, text2, include_insights=True,  label="/high-med"),
        _enrich_group(low,      text1, text2, include_insights=False, label="/low"),
    )

    # Merge clause_details from both groups; insights/recommendation come from high/med call
    merged_details = {
        **low_result.get("clause_details", {}),
        **high_med_result.get("clause_details", {}),  # high/med wins on key collision
    }
    return {
        "clause_details":    merged_details,
        "semantic_insights": high_med_result.get("semantic_insights", []),
        "recommendation":    high_med_result.get("recommendation",    ""),
    }


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

    # 2. Build raw change list (CPU-bound, run in thread)
    # text_diff_stats and risk_score are disabled for v1
    raw_changes = await asyncio.to_thread(_build_raw_changes, pairs)

    # 3. LLM enrichment — summaries, difference_points, insights, recommendation
    llm_data  = await _llm_enrichment(raw_changes, text1, text2)
    details   = llm_data.get("clause_details", {})
    insights  = llm_data.get("semantic_insights", [])
    rec       = llm_data.get("recommendation", "")

    # 4. Attach LLM enrichment + build side_by_side row per clause
    clause_changes = []
    for c in raw_changes:
        name        = c["clause_name"]
        llm_entry   = details.get(name, {})
        summary     = (llm_entry.get("summary") or "").strip()
        diff_points = llm_entry.get("difference_points") or []

        if not summary:
            if c["status"] == "added":
                summary = f"{name} is a new clause added in the revised document."
            elif c["status"] == "removed":
                summary = f"{name} has been removed from the revised document."
            else:
                summary = f"{name} has been modified in the revised document."

        e1      = c["doc1"].get("excerpt") or None
        e2      = c["doc2"].get("excerpt") or None
        wdiff   = c.get("word_diff", [])

        # Split word_diff into left (doc1) and right (doc2) token lists
        # Left  pane: equal + delete  (what was in doc1)
        # Right pane: equal + insert  (what is  in doc2)
        left_tokens  = [{"text": w["text"], "tag": w["tag"]} for w in wdiff if w["tag"] in ("equal", "delete")]
        right_tokens = [{"text": w["text"], "tag": w["tag"]} for w in wdiff if w["tag"] in ("equal", "insert")]

        clause_changes.append({
            "clause_name":  name,
            "status":       c["status"],
            "severity":     c["severity"],
            "summary":      summary,
            "side_by_side": {
                "left": {
                    "filename":    doc1_filename,
                    "present":     e1 is not None,
                    "excerpt":     e1,
                    "significance": c["doc1"].get("significance", ""),
                    "tokens":      left_tokens,   # equal + delete — render delete in red strikethrough
                },
                "right": {
                    "filename":    doc2_filename,
                    "present":     e2 is not None,
                    "excerpt":     e2,
                    "significance": c["doc2"].get("significance", ""),
                    "tokens":      right_tokens,  # equal + insert — render insert in green
                },
                "difference_points": diff_points,
            },
        })

    # 5. Document type compatibility message
    doc1_type = extraction1.get("document_type", "")
    doc2_type = extraction2.get("document_type", "")
    types_match = _norm(doc1_type) == _norm(doc2_type)
    if types_match:
        comparison_notice = (
            f"Both documents are of the same type ({doc1_type}). "
            "Comparison results are reliable."
        )
    else:
        comparison_notice = (
            f"Warning: Document types differ — '{doc1_type}' vs '{doc2_type}'. "
            "Comparing different document types may produce incomplete or inaccurate results."
        )

    duration_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        f"[comparison] Done — {duration_ms}ms | "
        f"changes={len(clause_changes)} | "
        f"types_match={types_match}"
    )

    high_count   = sum(1 for c in clause_changes if c["severity"] == "high")
    medium_count = sum(1 for c in clause_changes if c["severity"] == "medium")
    low_count    = sum(1 for c in clause_changes if c["severity"] == "low")
    added_count  = sum(1 for c in clause_changes if c["status"] == "added")
    removed_count= sum(1 for c in clause_changes if c["status"] == "removed")
    modified_count=sum(1 for c in clause_changes if c["status"] == "modified")

    return {
        "status":      "success",
        "duration_ms": duration_ms,
        "comparison": {
            "session_id":       session_id,
            "compared_at":      datetime.now(timezone.utc).isoformat(),
            "comparison_notice": comparison_notice,

            # Header block — drives the top bar of the UI
            "header": {
                "document_1": {
                    "filename":      doc1_filename,
                    "document_type": doc1_type,
                },
                "document_2": {
                    "filename":      doc2_filename,
                    "document_type": doc2_type,
                },
                "total_changes":  len(clause_changes),
                "by_severity": {
                    "high":   high_count,
                    "medium": medium_count,
                    "low":    low_count,
                },
                "by_status": {
                    "modified": modified_count,
                    "added":    added_count,
                    "removed":  removed_count,
                },
            },

            # Insights + recommendation — top section above clause rows
            "insights": {
                "semantic_insights": insights,
                "recommendation":    rec,
            },

            # Clause rows — each has side_by_side block ready for UI rendering
            "clause_changes": clause_changes,
        },
    }