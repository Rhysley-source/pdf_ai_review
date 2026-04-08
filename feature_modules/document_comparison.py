import logging
import difflib
import asyncio
import time
from typing import Dict, Any, List, Tuple
from difflib import SequenceMatcher

from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json_from_text
from feature_modules.risk_detection import analyze_document_risks

logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

HIGH_RISK = {"payment", "liability", "termination", "cost", "penalty"}
MEDIUM_RISK = {"timeline", "delivery", "scope", "renewal"}

GROUPS = {
    "financial": {"payment", "cost", "pricing"},
    "legal": {"liability", "termination", "law"},
    "operational": {"timeline", "delivery", "scope"},
}

MIN_CLAUSE_LENGTH = 40
MATCH_THRESHOLD = 0.65


# ============================================================
# HELPERS
# ============================================================

def _normalize(text: str) -> str:
    return (text or "").lower().strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _combined_similarity(k1: str, v1: str, k2: str, v2: str) -> float:
    """
    Hybrid similarity:
    - clause key similarity
    - clause content similarity
    """
    key_score = _similarity(k1, k2)
    value_score = _similarity(v1[:500], v2[:500])  # limit size
    return (0.4 * key_score) + (0.6 * value_score)


def _get_severity(clause: str) -> str:
    if any(k in clause for k in HIGH_RISK):
        return "High"
    if any(k in clause for k in MEDIUM_RISK):
        return "Medium"
    return "Low"


def _get_group(clause: str) -> str:
    for g, keys in GROUPS.items():
        if any(k in clause for k in keys):
            return g
    return "general"


def _clean_clauses(clauses: Dict[str, str]) -> Dict[str, str]:
    """
    Remove noisy or too-small clauses.
    """
    return {
        k: v for k, v in clauses.items()
        if v and len(v) >= MIN_CLAUSE_LENGTH
    }


# ============================================================
# CLAUSE MATCHING (IMPROVED)
# ============================================================

def _match_clauses(c1: dict, c2: dict) -> Dict[str, str | None]:
    matched = {}

    for k1, v1 in c1.items():
        best_match = None
        best_score = 0

        for k2, v2 in c2.items():
            score = _combined_similarity(k1, v1, k2, v2)

            if score > best_score:
                best_score = score
                best_match = k2

        matched[k1] = best_match if best_score >= MATCH_THRESHOLD else None

        logger.debug(f"[match] {k1} → {best_match} (score={best_score:.2f})")

    return matched


# ============================================================
# WORD DIFF
# ============================================================

def _word_diff_structured(a: str, b: str) -> List[Dict]:
    diff = difflib.ndiff((a or "").split(), (b or "").split())

    result = []
    for token in diff:
        if token.startswith("- "):
            result.append({"type": "removed", "text": token[2:]})
        elif token.startswith("+ "):
            result.append({"type": "added", "text": token[2:]})
        else:
            result.append({"type": "same", "text": token[2:]})

    return result


# ============================================================
# CLAUSE COMPARISON
# ============================================================

def _compare_clauses(c1: dict, c2: dict) -> List[Dict]:
    matches = _match_clauses(c1, c2)
    changes = []

    for k1, k2 in matches.items():
        v1 = c1.get(k1)
        v2 = c2.get(k2) if k2 else None

        if v1 == v2:
            continue

        change_type = "added" if not v1 else "removed" if not v2 else "modified"

        changes.append({
            "clause": k1,
            "matched_with": k2,
            "group": _get_group(k1),
            "severity": _get_severity(k1),
            "change_type": change_type,
            "original": v1,
            "revised": v2,
            "redline": _word_diff_structured(v1, v2)
        })

    return changes


# ============================================================
# TEXT DIFF
# ============================================================

def _text_diff(t1: str, t2: str):
    """
    Compute text-level diff using both diffing (difflib) and semantic similarity.
    """
    sim = SequenceMatcher(None, t1, t2).ratio()

    return {
        "similarity_score": round(sim, 4),
        "similarity_percent": f"{round(sim * 100, 1)}%",
        "detailed_diff": _generate_detailed_diff(t1, t2)  # Adds the diff as a list of changes
    }

def _generate_detailed_diff(t1: str, t2: str) -> list:
    """
    Generate a list of detailed changes between two texts.
    """
    diff = difflib.ndiff(t1.splitlines(), t2.splitlines())
    return [{"type": "added" if line.startswith('+') else "removed" if line.startswith('-') else "same", "line": line[2:]} for line in diff]


# ============================================================
# RISK ENGINE
# ============================================================

def _risk_score(changes: List[Dict]):
    h = sum(1 for c in changes if c["severity"] == "High")
    m = sum(1 for c in changes if c["severity"] == "Medium")
    l = sum(1 for c in changes if c["severity"] == "Low")

    score = min((h * 30) + (m * 15) + (l * 5), 100)

    return {
        "risk_score": score,
        "overall_risk": "High" if score >= 70 else "Medium" if score >= 30 else "Low",
        "high_risk_changes": h
    }


# ============================================================
# LLM INSIGHTS
# ============================================================

def _build_prompt(doc1: str, doc2: str, changes: List[Dict]):

    summary = "\n".join(
        f"- {c['clause']} ({c['change_type']}, {c['severity']})"
        for c in changes[:15]
    ) or "No major changes."

    return f"""
Analyze document differences.

Changes:
{summary}

Return STRICT JSON:
{{
  "insights": [],
  "clause_risks": [],
  "negotiation_suggestions": [],
  "recommendation": ""
}}

Doc1:
{doc1[:3000]}

Doc2:
{doc2[:3000]}
"""


async def _run_llm(doc1, doc2, changes):
    try:
        raw = await run_llm("", _build_prompt(doc1, doc2, changes))
        result = extract_json_from_text(raw)

        if result:
            return result

        logger.warning("[comparison] retry LLM")
        raw = await run_llm("", "Return ONLY valid JSON")
        return extract_json_from_text(raw) or {}

    except Exception as e:
        logger.exception(f"[comparison] LLM failed: {e}")
        return {}


# ============================================================
# MAIN FUNCTION (FINAL)
# ============================================================

async def compare_documents(
    text1: str,
    text2: str,
    clauses1: Dict[str, str],
    clauses2: Dict[str, str],
) -> Dict[str, Any]:

    start = time.time()

    if len(text1.split()) < 20 or len(text2.split()) < 20:
        raise ValueError("Documents too short")

    # Clean clauses
    clauses1 = _clean_clauses(clauses1 or {})
    clauses2 = _clean_clauses(clauses2 or {})

    # Fallback safety (CRITICAL)
    if not clauses1 or not clauses2:
        logger.warning("[comparison] empty clauses → fallback to full text chunking")
        clauses1 = {"full_text": text1[:4000]}
        clauses2 = {"full_text": text2[:4000]}

    # Clean up empty clauses before comparison
    clauses1 = {k: v for k, v in clauses1.items() if v}
    clauses2 = {k: v for k, v in clauses2.items() if v}

    # Log the cleaned clauses for debugging
    logger.info(f"[clauses1] {clauses1}")
    logger.info(f"[clauses2] {clauses2}")
    # Parallel execution
    clause_task = asyncio.to_thread(_compare_clauses, clauses1, clauses2)
    diff_task = asyncio.to_thread(_text_diff, text1, text2)
    risk_task = analyze_document_risks(text2)

    clause_changes, text_diff, risk_ai = await asyncio.gather(
        clause_task, diff_task, risk_task
    )

    risk = _risk_score(clause_changes)

    llm_data = await _run_llm(text1, text2, clause_changes)

    duration = int((time.time() - start) * 1000)

    return {
        "status": "success",
        "duration_ms": duration,

        "summary": {
            "total_changes": len(clause_changes),
            "risk_score": risk["risk_score"],
            "overall_risk": risk["overall_risk"],
            "high_risk_changes": risk["high_risk_changes"],
            "recommendation": llm_data.get("recommendation", "")
        },

        "grouped_changes": {
            g: [c for c in clause_changes if c["group"] == g]
            for g in ["financial", "legal", "operational", "general"]
        },

        "clause_changes": clause_changes,

        "semantic_insights": llm_data.get("insights", []),
        "clause_risk_analysis": llm_data.get("clause_risks", []),
        "negotiation_suggestions": llm_data.get("negotiation_suggestions", []),

        "text_diff": text_diff,
        "risk_analysis": risk_ai
    }