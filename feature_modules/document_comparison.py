"""
document_comparison.py

Compares two documents and produces a structured JSON result that matches
the target schema:

{
  "status": "success",
  "log_id": <int from DB>,
  "duration_ms": <int>,
  "comparison": {
    "session_id": "...",
    "doc1_filename": "...",
    "doc2_filename": "...",
    "compared_at": "...",
    "total_changes": <int>,
    "high_risk_changes": <int>,
    "overall_risk_level": "high|medium|low",
    "recommendation": "...",
    "clause_changes": [
      {
        "clause": "payment_terms",
        "change_type": "modified|added|removed",
        "original_value": "...",
        "revised_value": "...",      # null when removed
        "severity": "high|medium|low",
        "summary": "..."
      }
    ],
    "semantic_insights": ["...", ...],
    "text_diff_stats": {
      "lines_added": <int>,
      "lines_removed": <int>,
      "lines_changed": <int>,
      "lines_unchanged": <int>,
      "similarity_score": <float>,
      "similarity_percent": "94.2%"
    }
  }
}

Key design decisions vs. the old version:
  - Risk analysis of doc2 in isolation (analyze_document_risks) has been
    removed — it's not in the target schema and costs an extra LLM call.
  - grouped_changes and redline arrays are removed from the top-level output
    (not in target schema).
  - clause_changes uses original_value / revised_value (target field names),
    not original / revised.
  - severity is lowercase ("high") not title-case ("High").
  - text_diff_stats includes lines_added / lines_removed / lines_changed /
    lines_unchanged computed from difflib.
  - The LLM prompt is substantially richer: it receives the full clause
    change list and is asked to produce (a) a per-clause summary and
    (b) semantic_insights and (c) an overall recommendation.
  - overall_risk_level is lowercase.
  - The function signature now accepts doc1_filename / doc2_filename /
    session_id so the caller (route.py) can pass them through.
"""

import logging
import difflib
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher

from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Clause keys whose changes are always "high" severity
HIGH_RISK_CLAUSES = {
    "payment_terms", "liability", "termination", "indemnification",
    "penalties", "pricing", "intellectual_property",
}

# Clause keys whose changes are "medium" severity
MEDIUM_RISK_CLAUSES = {
    "contract_term", "renewal_conditions", "scope_of_work",
    "delivery_timeline", "non_compete", "warranties",
    "force_majeure", "data_protection",
}

# Minimum character length for a clause value to be considered usable
MIN_CLAUSE_LENGTH = 30

# Minimum similarity score between two clause values to call it a "match"
MATCH_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _key_sim(k1: str, k2: str) -> float:
    """Normalised key similarity — underscores treated as spaces."""
    a = k1.replace("_", " ")
    b = k2.replace("_", " ")
    return SequenceMatcher(None, a, b).ratio()


def _combined_sim(k1: str, v1: str, k2: str, v2: str) -> float:
    """
    Weighted hybrid:
      50% key similarity  (ensures "payment_terms" → "payment_terms")
      50% value similarity (helps when keys are slightly different)
    """
    ks = _key_sim(k1, k2)
    vs = _sim(v1[:400], v2[:400])
    return 0.50 * ks + 0.50 * vs


def _severity(clause_key: str) -> str:
    """Return lowercase severity string for a clause key."""
    if clause_key in HIGH_RISK_CLAUSES:
        return "high"
    if clause_key in MEDIUM_RISK_CLAUSES:
        return "medium"
    return "low"


def _clean_clauses(clauses: Dict[str, str]) -> Dict[str, str]:
    return {
        k: v.strip()
        for k, v in (clauses or {}).items()
        if isinstance(v, str) and len(v.strip()) >= MIN_CLAUSE_LENGTH
    }


# ---------------------------------------------------------------------------
# Clause matching
# ---------------------------------------------------------------------------

def _match_clauses(c1: Dict[str, str], c2: Dict[str, str]) -> Dict[str, Optional[str]]:
    """
    For every clause in doc1, find the best-matching clause in doc2.
    Returns {c1_key: c2_key | None}.

    Preference order:
      1. Exact key match (score=1.0 shortcut)
      2. Combined key+value similarity above threshold
    """
    matched: Dict[str, Optional[str]] = {}

    for k1, v1 in c1.items():
        # Shortcut: exact key exists in doc2
        if k1 in c2:
            matched[k1] = k1
            logger.debug(f"[match] {k1} → {k1} (exact)")
            continue

        best_key: Optional[str] = None
        best_score: float = 0.0

        for k2, v2 in c2.items():
            score = _combined_sim(k1, v1, k2, v2)
            if score > best_score:
                best_score = score
                best_key = k2

        matched[k1] = best_key if best_score >= MATCH_THRESHOLD else None
        logger.debug(f"[match] {k1} → {best_key} (score={best_score:.2f})")

    return matched


# ---------------------------------------------------------------------------
# Text diff stats
# ---------------------------------------------------------------------------

def _compute_text_diff_stats(t1: str, t2: str) -> Dict[str, Any]:
    """
    Compute line-level diff statistics.

    Returns:
      lines_added, lines_removed, lines_changed, lines_unchanged,
      similarity_score, similarity_percent
    """
    lines1 = t1.splitlines()
    lines2 = t2.splitlines()

    sm = SequenceMatcher(None, lines1, lines2)
    similarity = round(sm.ratio(), 4)

    lines_added = 0
    lines_removed = 0
    lines_changed = 0     # lines that appear in both but differ
    lines_unchanged = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            lines_unchanged += i2 - i1
        elif tag == "insert":
            lines_added += j2 - j1
        elif tag == "delete":
            lines_removed += i2 - i1
        elif tag == "replace":
            # Count as both a removal and an addition; the smaller count is "changed"
            n_removed = i2 - i1
            n_added   = j2 - j1
            changed   = min(n_removed, n_added)
            lines_changed  += changed
            lines_removed  += n_removed - changed
            lines_added    += n_added   - changed

    return {
        "lines_added":     lines_added,
        "lines_removed":   lines_removed,
        "lines_changed":   lines_changed,
        "lines_unchanged": lines_unchanged,
        "similarity_score":   similarity,
        "similarity_percent": f"{round(similarity * 100, 1)}%",
    }


# ---------------------------------------------------------------------------
# Clause comparison (produces raw changes list with original/revised values)
# ---------------------------------------------------------------------------

def _compare_clauses(c1: Dict[str, str], c2: Dict[str, str]) -> List[Dict]:
    """
    Compare clauses from doc1 vs doc2.

    Returns list of change dicts:
      clause, change_type, original_value, revised_value, severity
    (No 'summary' yet — that is added by the LLM in _run_llm.)
    """
    matches = _match_clauses(c1, c2)
    changes: List[Dict] = []

    for k1, k2 in matches.items():
        v1 = c1.get(k1)
        v2 = c2.get(k2) if k2 else None

        # Skip identical clauses
        if v1 and v2 and _sim(v1, v2) > 0.97:
            continue

        change_type = (
            "added"    if not v1 and v2  else
            "removed"  if v1 and not v2  else
            "modified"
        )

        changes.append({
            "clause":         k1,
            "change_type":    change_type,
            "original_value": v1,
            "revised_value":  v2,
            "severity":       _severity(k1),
            "summary":        "",   # filled by LLM
        })

    # Clauses in doc2 that have NO match in doc1 → "added"
    matched_c2_keys = set(v for v in matches.values() if v)
    for k2, v2 in c2.items():
        if k2 not in matched_c2_keys and k2 not in c1:
            changes.append({
                "clause":         k2,
                "change_type":    "added",
                "original_value": None,
                "revised_value":  v2,
                "severity":       _severity(k2),
                "summary":        "",
            })

    return changes


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def _risk_score(changes: List[Dict]) -> Dict[str, Any]:
    h = sum(1 for c in changes if c["severity"] == "high")
    m = sum(1 for c in changes if c["severity"] == "medium")
    l = sum(1 for c in changes if c["severity"] == "low")

    score = min((h * 30) + (m * 15) + (l * 5), 100)

    return {
        "risk_score":        score,
        "overall_risk_level": "high" if score >= 70 else "medium" if score >= 30 else "low",
        "high_risk_changes": h,
    }


# ---------------------------------------------------------------------------
# LLM enrichment — per-clause summaries + semantic insights + recommendation
# ---------------------------------------------------------------------------

def _build_llm_prompt(
    text1: str,
    text2: str,
    changes: List[Dict],
) -> str:
    change_lines = []
    for c in changes[:20]:
        orig = (c["original_value"] or "")[:300]
        rev  = (c["revised_value"]  or "")[:300]
        change_lines.append(
            f"Clause: {c['clause']} | change_type: {c['change_type']} | severity: {c['severity']}\n"
            f"  original_value: {orig or '[absent]'}\n"
            f"  revised_value:  {rev  or '[absent]'}"
        )
    changes_block = "\n\n".join(change_lines) or "No clause-level changes detected."

    return f"""You are a senior legal analyst reviewing two versions of a contract.

DETECTED CLAUSE CHANGES:
{changes_block}

DOCUMENT 1 (original):
---
{text1[:2500]}
---

DOCUMENT 2 (revised):
---
{text2[:2500]}
---

Return ONLY a valid JSON object with exactly these fields:

{{
  "clause_summaries": {{
    "<clause_key>": "<one clear sentence describing what changed and why it matters>",
    ...
  }},
  "semantic_insights": [
    "<specific, quantified insight about a change — e.g. 'Payment terms tightened from Net 30 to Net 15, increasing cash-flow pressure on the service provider.'>",
    "<insight>",
    "<overall conclusion about which party the revised document favours and why>"
  ],
  "recommendation": "<2–3 sentence executive summary recommendation for the recipient>"
}}

Rules:
- clause_summaries: one entry per clause that changed. Key must match the clause key exactly.
- semantic_insights: 4–7 items. Be specific — name the values, percentages, durations changed.
  Last item must be an overall conclusion about which party benefits from the revisions.
- recommendation: actionable guidance (not just 'seek legal advice').
- Return ONLY the JSON object. No markdown, no explanation outside it."""


async def _run_llm_enrichment(
    text1: str,
    text2: str,
    changes: List[Dict],
) -> Dict[str, Any]:
    """
    Ask the LLM to:
      1. Write a one-sentence summary for each changed clause.
      2. Produce semantic_insights (specific, quantified).
      3. Write an overall recommendation.

    Returns dict with keys: clause_summaries, semantic_insights, recommendation.
    """
    if not changes:
        return {"clause_summaries": {}, "semantic_insights": [], "recommendation": ""}

    try:
        prompt = _build_llm_prompt(text1, text2, changes)
        raw    = await run_llm("", prompt)
        result = extract_json_from_text(raw)

        if result and (result.get("semantic_insights") or result.get("recommendation")):
            return result

        # Single retry with a stripped-down prompt
        logger.warning("[comparison] LLM enrichment attempt 1 returned no usable data — retrying")
        retry_prompt = (
            "Return ONLY valid JSON with keys: clause_summaries (object), "
            "semantic_insights (array of strings), recommendation (string).\n\n"
            + prompt[:2000]
        )
        raw    = await run_llm("", retry_prompt)
        result = extract_json_from_text(raw)
        return result or {"clause_summaries": {}, "semantic_insights": [], "recommendation": ""}

    except Exception as e:
        logger.exception(f"[comparison] LLM enrichment failed: {e}")
        return {"clause_summaries": {}, "semantic_insights": [], "recommendation": ""}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def compare_documents(
    text1: str,
    text2: str,
    clauses1: Dict[str, str],
    clauses2: Dict[str, str],
    *,
    doc1_filename: str = "document_1.pdf",
    doc2_filename: str = "document_2.pdf",
    session_id: str = "",
    log_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare two documents and return a structured result matching the target schema.

    Parameters
    ----------
    text1, text2        : raw document text for each document
    clauses1, clauses2  : extracted clause dicts from clause_extraction.extract_clauses()
    doc1_filename       : original filename (for metadata)
    doc2_filename       : revised filename (for metadata)
    session_id          : UUID string from the route layer
    log_id              : DB row id populated after DB insert (passed back by route)

    Returns
    -------
    Dict matching the target JSON schema.
    """
    t_start = time.perf_counter()

    if len(text1.split()) < 20 or len(text2.split()) < 20:
        raise ValueError("Documents too short for meaningful comparison.")

    # ── 1. Clean clauses ────────────────────────────────────────────────
    c1 = _clean_clauses(clauses1)
    c2 = _clean_clauses(clauses2)

    # Fallback: if either side has no usable clauses, use first 4000 chars
    # as a single blob so we can still produce text diff stats and LLM insights
    if not c1 or not c2:
        logger.warning(
            "[comparison] One or both clause dicts are empty after cleaning — "
            "falling back to full-text blob. This produces lower-quality clause diff."
        )
        c1 = c1 or {"full_text": text1[:4000]}
        c2 = c2 or {"full_text": text2[:4000]}

    logger.info(f"[comparison] doc1 clauses={list(c1.keys())}  doc2 clauses={list(c2.keys())}")

    # ── 2. Parallel: clause diff + text diff stats ───────────────────────
    clause_task = asyncio.to_thread(_compare_clauses, c1, c2)
    diff_task   = asyncio.to_thread(_compute_text_diff_stats, text1, text2)

    raw_changes, text_diff_stats = await asyncio.gather(clause_task, diff_task)

    # ── 3. Risk scoring ──────────────────────────────────────────────────
    risk = _risk_score(raw_changes)

    # ── 4. LLM enrichment (per-clause summaries + insights + recommendation)
    llm_data = await _run_llm_enrichment(text1, text2, raw_changes)

    clause_summaries    = llm_data.get("clause_summaries", {})
    semantic_insights   = llm_data.get("semantic_insights", [])
    recommendation      = llm_data.get("recommendation", "")

    # ── 5. Merge LLM summaries into clause_changes ───────────────────────
    clause_changes: List[Dict] = []
    for c in raw_changes:
        key     = c["clause"]
        summary = clause_summaries.get(key, "")
        if not summary:
            # Auto-generate a fallback summary so the field is never blank
            ct = c["change_type"]
            summary = (
                f"{key.replace('_', ' ').capitalize()} clause has been {ct}."
            )
        clause_changes.append({
            "clause":         key,
            "change_type":    c["change_type"],
            "original_value": c["original_value"],
            "revised_value":  c["revised_value"],
            "severity":       c["severity"],
            "summary":        summary,
        })

    duration_ms = int((time.perf_counter() - t_start) * 1000)

    logger.info(
        f"[comparison] done — {duration_ms}ms "
        f"changes={len(clause_changes)} "
        f"risk={risk['overall_risk_level']} "
        f"insights={len(semantic_insights)}"
    )

    return {
        "status":      "success",
        "log_id":      log_id,
        "duration_ms": duration_ms,
        "comparison": {
            "session_id":        session_id,
            "doc1_filename":     doc1_filename,
            "doc2_filename":     doc2_filename,
            "compared_at":       datetime.now(timezone.utc).isoformat(),
            "total_changes":     len(clause_changes),
            "high_risk_changes": risk["high_risk_changes"],
            "overall_risk_level": risk["overall_risk_level"],
            "recommendation":    recommendation,
            "clause_changes":    clause_changes,
            "semantic_insights": semantic_insights,
            "text_diff_stats":   text_diff_stats,
        },
    }