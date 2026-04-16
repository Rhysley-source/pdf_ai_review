import logging
import re
from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response normalizer
# ---------------------------------------------------------------------------

def _normalize_result(result: dict) -> dict:
    _VALID_SEVERITY   = {"High", "Medium", "Low"}
    _VALID_IMPORTANCE = {"Critical", "Important", "Optional"}

    normalized_risks: list[dict] = []
    for item in result.get("detected_risks") or []:
        if not isinstance(item, dict):
            continue
        severity = item.get("severity") or item.get("risk_level") or item.get("level") or "Medium"
        if severity not in _VALID_SEVERITY:
            severity = "Medium"
        normalized_risks.append({
            "risk_name":       str(item.get("risk_name")       or item.get("name")          or item.get("risk")           or "Unknown Risk"),
            "severity":        severity,
            "severity_reason": str(item.get("severity_reason") or item.get("reason_for_severity") or ""),
            "clause_found":    str(item.get("clause_found")    or item.get("clause")         or item.get("quote")          or "Not found"),
            "impact":          str(item.get("impact")          or item.get("danger")         or item.get("effect")         or ""),
            "mitigation":      str(item.get("mitigation")      or item.get("fix")            or item.get("recommendation") or ""),
        })

    normalized_fields: list[dict] = []
    for item in result.get("missing_fields") or []:
        if not isinstance(item, dict):
            continue
        importance = item.get("importance") or item.get("priority") or item.get("criticality") or "Important"
        if importance not in _VALID_IMPORTANCE:
            importance = "Important"
        normalized_fields.append({
            "field_name": str(item.get("field_name") or item.get("field") or item.get("name")   or "Unknown Field"),
            "importance": importance,
            "reason":     str(item.get("reason")     or item.get("why")   or item.get("detail") or ""),
        })

    return {
        "document_type":      str(result.get("document_type")      or "other"),
        "document_label":     str(result.get("document_label")     or ""),
        "detected_risks":     normalized_risks,
        "missing_fields":     normalized_fields,
        "overall_assessment": str(result.get("overall_assessment") or ""),
    }


# ---------------------------------------------------------------------------
# Public entry point — single LLM call
# ---------------------------------------------------------------------------

_MAX_SINGLE_CALL_CHARS = 100_000

_SINGLE_CALL_SYSTEM = """You are a senior legal and financial risk analyst.

Analyze the document and return a single JSON object with EXACTLY this structure:

{
  "document_type": "<one of: contract, employment, nda, lease, invoice, resume, other>",
  "document_label": "<specific document name — max 5 words>",
  "detected_risks": [
    {
      "risk_name": "<risk name — max 6 words>",
      "severity": "High | Medium | Low",
      "severity_reason": "<why this severity — max 15 words>",
      "clause_found": "<exact quote or description — max 25 words>",
      "impact": "<why dangerous — max 20 words>",
      "mitigation": "<how to fix — max 20 words>"
    }
  ],
  "missing_fields": [
    {
      "field_name": "<missing field — max 5 words>",
      "importance": "Critical | Important | Optional",
      "reason": "<why needed — max 15 words>"
    }
  ],
  "unfilled_placeholders": ["<placeholder verbatim>"],
  "overall_assessment": "<executive summary — max 40 words>"
}

Rules:
- ONLY flag a risk if you can cite specific language from the document that supports it
- Do NOT flag absence of protective language as a risk — put it in missing_fields
- Check for unfilled placeholders: [FIELD], _____, <NAME>, TBD, TBA
- detected_risks: [] if none found
- missing_fields: [] if none found
- unfilled_placeholders: [] if none found
- Return ONLY valid JSON — no markdown, no explanation"""


async def analyze_document_risks(text: str) -> dict:
    """Single LLM call: classify + risk analysis + placeholder check in one shot."""
    document = text[:_MAX_SINGLE_CALL_CHARS]

    logger.info(f"[risk_detection] Single-call analysis — {len(document):,} chars")
    raw    = await run_llm(document, _SINGLE_CALL_SYSTEM, max_output_tokens=3000)
    result = extract_json_from_text(raw)

    if not result:
        logger.warning("[risk_detection] JSON parse failed — returning empty result")
        result = {
            "document_type":         "other",
            "document_label":        "General Document",
            "detected_risks":        [],
            "missing_fields":        [],
            "unfilled_placeholders": [],
            "overall_assessment":    "",
        }

    doc_label = result.get("document_label") or result.get("document_type") or "General Document"

    # Promote unfilled placeholders to a High-severity risk entry
    placeholders = result.pop("unfilled_placeholders", []) or []
    if placeholders:
        count = len(placeholders)
        display_list = ", ".join(f'"{p}"' for p in placeholders)
        result.setdefault("detected_risks", []).insert(0, {
            "risk_name":       "Unfilled Placeholders Detected",
            "severity":        "High",
            "severity_reason": f"{count} unfilled placeholder(s) found — document is incomplete.",
            "clause_found":    display_list,
            "impact":          "Agreement with blank fields may be legally unenforceable.",
            "mitigation":      "Replace every placeholder with a real value before signing.",
        })

    analysis = _normalize_result(result)

    # Promote missing fields to risk entries
    _IMPORTANCE_TO_SEVERITY = {"Critical": "High", "Important": "Medium", "Optional": "Low"}
    missing_as_risks = [
        {
            "risk_name":       f"Missing Required Field: {f['field_name']}",
            "severity":        _IMPORTANCE_TO_SEVERITY.get(f["importance"], "Medium"),
            "severity_reason": (
                f"'{f['field_name']}' is a {f['importance'].lower()} field for a "
                f"{doc_label} and is absent from the document."
            ),
            "clause_found":    "Not found",
            "impact":          f["reason"],
            "mitigation":      f"Add '{f['field_name']}' with a real value before signing.",
        }
        for f in analysis.get("missing_fields", [])
    ]
    if missing_as_risks:
        analysis["detected_risks"] = analysis.get("detected_risks", []) + missing_as_risks

    detected_count = len(analysis.get("detected_risks", []))
    analysis.pop("missing_fields", None)

    logger.info(f"[risk_detection] Done — {detected_count} risk(s) found")

    return {
        "status":        "success",
        "analysis_type": "risk_detection",
        "document_type": doc_label,
        "risk_count": {
            "detected_risks": detected_count,
            "total":          detected_count,
        },
        "data": analysis,
    }
