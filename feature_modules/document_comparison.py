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

from llm_model.ai_model import run_llm_mini, run_llm_comparison
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


def _normalize_clause_name(name: str) -> str:
    """
    Strip leading numbering and structural words before similarity scoring.
    'Clause 4.2 – Payment Terms' → 'payment terms'
    'Section 3: Non-Compete'     → 'non compete'
    'Article IV. Termination'    → 'termination'
    """
    s = (name or "").lower()
    # Remove leading clause/section/article with number and separator
    s = re.sub(r'^(clause|section|article|part|schedule|exhibit|annex)\s*[\divxlc\d.]*[\s\-–—:\.]*', '', s)
    # Remove a bare leading number like "4." or "4.2.1 "
    s = re.sub(r'^\d[\d.]*[\s\-–—:\.]*', '', s)
    # Strip remaining punctuation, normalize whitespace
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _name_sim(a: str, b: str) -> float:
    """Similarity on normalized clause names — strips numbering before comparing."""
    return SequenceMatcher(None, _normalize_clause_name(a), _normalize_clause_name(b)).ratio()


# ---------------------------------------------------------------------------
# Regex-based value extraction — improvement 4
# Pulls out concrete values (amounts, dates, durations, percentages) that
# are guaranteed to appear in difference_points even if the LLM misses them.
# ---------------------------------------------------------------------------

_VALUE_PATTERN = re.compile(
    r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|k|thousand))?' # $500,000 / $1.5 million
    r'|\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:USD|EUR|GBP|INR)'  # 10,000 USD
    r'|\b\d+(?:\.\d+)?\s*%'                                     # 15%
    r'|\b\d+\s*(?:calendar\s+)?days?\b'                         # 30 days
    r'|\b\d+\s*(?:business\s+)?days?\b'                         # 5 business days
    r'|\b\d+\s*months?\b'                                        # 6 months
    r'|\b\d+\s*years?\b'                                         # 2 years
    r'|\b\d+\s*weeks?\b'                                         # 4 weeks
    r'|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?'
    r'|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    r'\s+\d{1,2},?\s+\d{4}\b'                                   # January 1, 2025
    r'|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',                     # 01/15/2025
    re.IGNORECASE,
)


def _extract_value_diffs(excerpt1: str, excerpt2: str) -> list[str]:
    """
    Compare concrete values extracted from both excerpts via regex.
    Returns a list of "Doc1: <val> → Doc2: <val>" strings for values that differ.
    Values present in one doc but absent in the other are also flagged.
    """
    vals1 = {v.lower().strip() for v in _VALUE_PATTERN.findall(excerpt1 or "")}
    vals2 = {v.lower().strip() for v in _VALUE_PATTERN.findall(excerpt2 or "")}

    diffs: list[str] = []

    only_in_1 = vals1 - vals2
    only_in_2 = vals2 - vals1

    # Try to pair removed/added values for a cleaner "X → Y" message
    paired: set[str] = set()
    for v1 in sorted(only_in_1):
        # Find the closest value in doc2 by type heuristic (same unit suffix)
        suffix1 = re.sub(r'[\d,.$]', '', v1).strip()
        match = next(
            (v2 for v2 in sorted(only_in_2)
             if re.sub(r'[\d,.$]', '', v2).strip() == suffix1 and v2 not in paired),
            None,
        )
        if match:
            diffs.append(f"Doc1: \"{v1}\" → Doc2: \"{match}\"")
            paired.add(match)
        else:
            diffs.append(f"Doc1 only: \"{v1}\" (removed or changed)")

    for v2 in sorted(only_in_2):
        if v2 not in paired:
            diffs.append(f"Doc2 only: \"{v2}\" (added or changed)")

    return diffs


# ---------------------------------------------------------------------------
# Clause matching
# Matches clauses from doc2 → doc1 by name similarity + excerpt similarity.
# Returns list of (clause1_or_None, clause2_or_None, match_score).
# ---------------------------------------------------------------------------

_NAME_WEIGHT     = 0.65
_EXCERPT_WEIGHT  = 0.35
_MATCH_THRESHOLD = 0.62   # raised from 0.45 — prevents wrong-clause pairings
_MIN_NAME_SIM    = 0.28   # clause names must have some overlap before pairing


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
            # Use normalized names (strips "Clause 4 –" prefixes) for matching
            name_sim = _name_sim(c1["clause_name"], c2["clause_name"])
            # Reject pairs where clause names have no meaningful overlap
            if name_sim < _MIN_NAME_SIM:
                continue
            score = (
                _NAME_WEIGHT    * name_sim +
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
            if ratio >= 1.0:
                continue  # byte-identical — skip; even 1-word changes are flagged
            status = "modified"

        e1 = (c1 or {}).get("excerpt", "")
        e2 = (c2 or {}).get("excerpt", "")

        # Regex-detected value changes — guaranteed presence regardless of LLM quality
        value_diffs = _extract_value_diffs(e1, e2) if status == "modified" else []

        changes.append({
            "clause_name":      name,
            "status":           status,
            "severity":         severity,
            "doc1_excerpt":     e1,
            "doc2_excerpt":     e2,
            "value_diffs":      value_diffs,   # regex-extracted; prepended to LLM difference_points
            "difference_points": [],           # filled by LLM enrichment
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
    "Analyse the clause changes provided and identify EXACT, SPECIFIC differences — "
    "quote the actual values, amounts, dates, percentages, or terms that changed. "
    "Vague statements like 'terms were modified' are not acceptable. "
    "Return ONLY valid JSON — no markdown, no explanation."
)


def _build_enrichment_prompt(changes: list[dict], doc1_text: str, doc2_text: str) -> str:
    sorted_changes = sorted(
        changes[:20],
        key=lambda c: {"high": 0, "medium": 1, "low": 2}.get(c["severity"], 3),
    )
    lines = []
    for i, c in enumerate(sorted_changes, 1):
        e1 = (c.get("doc1_excerpt") or "")[:500].replace("\n", " ")
        e2 = (c.get("doc2_excerpt") or "")[:500].replace("\n", " ")
        lines.append(
            f"{i}. CLAUSE: {c['clause_name'].upper()}\n"
            f"   Status: {c['status']} | Severity: {c['severity']}\n"
            f"   Doc1 (original) : {e1 or '[absent]'}\n"
            f"   Doc2 (revised)  : {e2 or '[absent]'}"
        )

    if len(changes) > 20:
        logger.warning(f"[comparison] {len(changes)} changes — enriching top 20 by severity")

    changes_block = "\n\n".join(lines) or "No changes detected."

    return f"""DETECTED CLAUSE CHANGES ({len(sorted_changes)} shown, sorted by severity):
{changes_block}

DOCUMENT 1 EXCERPT (original):
{doc1_text[:1500]}

DOCUMENT 2 EXCERPT (revised):
{doc2_text[:1500]}

Return ONLY this JSON:

{{
  "clause_details": {{
    "<clause_name exactly as written above>": {{
      "summary": "<one sentence: what changed and the business impact>",
      "difference_points": [
        "<specific point 1 — name exact value/term that changed, e.g. 'Payment period changed from Net 30 to Net 15'>",
        "<specific point 2 — another concrete difference>",
        "<specific point 3 — add more if needed>"
      ]
    }}
  }},
  "semantic_insights": [
    "<quantified insight referencing exact values — e.g. 'Liability cap reduced from $500K to $100K'>",
    "<another insight on a different clause>",
    "<which party benefits from these changes and why — name 2-3 specific reasons>"
  ],
  "recommendation": "<2-3 actionable sentences naming specific clauses to negotiate and the target outcome>"
}}

Rules:
- clause_details key must exactly match the clause_name from the change list
- difference_points MUST quote exact text: "Doc1 says X, Doc2 says Y" — never say "was changed" without citing both values
- Every modified clause must have at least 2 difference_points
- Added clauses: quote exact new terms introduced
- Removed clauses: quote exact text that was removed and name the protection lost
- If a numeric value, date, party name, or dollar amount changed, it MUST appear in difference_points"""


async def _llm_enrichment(changes: list[dict], text1: str, text2: str) -> dict:
    empty = {"clause_details": {}, "semantic_insights": [], "recommendation": ""}
    if not changes:
        return empty

    try:
        prompt = _build_enrichment_prompt(changes, text1, text2)

        # Attempt 1 — uses COMPARISON_MODEL (default: gpt-4.1) for accurate legal analysis
        raw = await run_llm_comparison(prompt, _ENRICHMENT_SYSTEM, max_output_tokens=8000)
        result = extract_json_from_text(raw)
        if result and result.get("clause_details"):
            logger.info(
                f"[comparison] LLM OK — "
                f"clause_details={len(result.get('clause_details', {}))} "
                f"insights={len(result.get('semantic_insights', []))}"
            )
            return result

        # Attempt 2 — retry with worked example
        logger.warning("[comparison] LLM attempt 1 weak — retrying with example")
        ex      = changes[0]
        ex_name = ex["clause_name"]
        ex_e1   = (ex.get("doc1_excerpt") or "original text")[:60]
        ex_e2   = (ex.get("doc2_excerpt") or "revised text")[:60]

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
            f'  }}}},\n'
            '  "semantic_insights": ["Specific change with exact values.", "Which party benefits and why."],\n'
            '  "recommendation": "Negotiate to restore [clause] before signing."\n'
            "}\n\n"
            "Now produce the real analysis:\n\n"
            + prompt
        )

        raw2   = await run_llm_comparison(retry_prompt, _ENRICHMENT_SYSTEM, max_output_tokens=8000)
        result2 = extract_json_from_text(raw2)
        if result2 and result2.get("clause_details"):
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

    # Sort both clause lists alphabetically before matching so the greedy
    # algorithm always processes clauses in the same order regardless of how
    # the LLM returned them — this makes the comparison count deterministic.
    clauses1: list[dict] = sorted(
        extraction1.get("key_clauses", []),
        key=lambda c: _normalize_clause_name(c.get("clause_name", "")),
    )
    clauses2: list[dict] = sorted(
        extraction2.get("key_clauses", []),
        key=lambda c: _normalize_clause_name(c.get("clause_name", "")),
    )

    logger.info(
        f"[comparison] Starting — "
        f"doc1={doc1_filename} ({len(clauses1)} clauses) | "
        f"doc2={doc2_filename} ({len(clauses2)} clauses)"
    )

    # 1. Match clauses across the two documents
    pairs = _match_clauses(clauses1, clauses2)

    # 2. Build raw change list (CPU-bound, run in thread)
    raw_changes = await asyncio.to_thread(_build_raw_changes, pairs)

    # 3. LLM enrichment — difference_points, insights, recommendation
    llm_data  = await _llm_enrichment(raw_changes, text1, text2)
    details   = llm_data.get("clause_details", {})
    insights  = llm_data.get("semantic_insights", [])
    rec       = llm_data.get("recommendation", "")

    # 4. Assemble final clause_changes — only real differences are included.
    # "modified" entries with zero difference_points are discarded: they are
    # near-identical clauses that passed the 1.0 ratio threshold but have no
    # concrete change the LLM or regex could detect.
    clause_changes = []
    for c in raw_changes:
        name         = c["clause_name"]
        llm_entry    = details.get(name, {})
        # Regex value diffs first (factual), then LLM points (deduplicated)
        value_diffs  = c.get("value_diffs", [])
        llm_points   = llm_entry.get("difference_points") or []
        diff_points  = value_diffs + [p for p in llm_points if p not in value_diffs]

        # Drop "modified" clauses where no concrete difference was found
        if c["status"] == "modified" and not diff_points:
            logger.debug(f"[comparison] Dropping '{name}' — modified but no difference_points detected")
            continue

        clause_changes.append({
            "clause_name":       name,
            "status":            c["status"],
            "severity":          c["severity"],
            "doc1_excerpt":      c.get("doc1_excerpt") or "",
            "doc2_excerpt":      c.get("doc2_excerpt") or "",
            "difference_points": diff_points,
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

            # Insights + recommendation
            "insights": {
                "semantic_insights": insights,
                "recommendation":    rec,
            },

            # One entry per clause that actually changed.
            # Each entry contains only the concrete difference_points — no
            # token arrays or side_by_side metadata.
            "clause_changes": clause_changes,
        },
    }