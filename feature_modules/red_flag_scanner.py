import logging
from llm_model.ai_model import _run_inference_json
from utils.json_utils import extract_json_raw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WHY THE OLD APPROACH GAVE INCONSISTENT COUNTS
#
# Old approach: "find all dangerous clauses in this contract" → free-form list
#   - The model decides *how many* to return each call → 8, 10, 12 vary
#   - run_llm uses plain text inference (no response_format) → JSON parsing fails
#     sometimes, causing different items to survive extraction
#
# New approach: fixed checklist — model answers present/absent for every item
#   - Same 29 items evaluated every single call → count is bounded and stable
#   - _run_inference_json uses response_format=json_object → clean, parseable JSON
#   - Severity is hardcoded here (not decided by the model) → fully deterministic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fixed checklist — 29 items, evaluated on every call
# category: dangerous | unusual | missing
# ---------------------------------------------------------------------------

_CHECKLIST = [
    # ── Dangerous (clause IS present and harms the signer) ───────────────
    {"id": "unlimited_liability",      "category": "dangerous", "label": "Unlimited liability",           "check": "Is there a clause that places unlimited or uncapped financial liability on one party?"},
    {"id": "one_sided_indemnity",      "category": "dangerous", "label": "One-sided indemnification",    "check": "Does only one party bear all legal costs and losses, with no mutual indemnification?"},
    {"id": "unilateral_amendment",     "category": "dangerous", "label": "Unilateral amendment",         "check": "Can one party change contract terms without the other party's written consent?"},
    {"id": "mandatory_arbitration",    "category": "dangerous", "label": "Mandatory arbitration",        "check": "Is there a clause that forces disputes into private arbitration and waives the right to sue in court?"},
    {"id": "auto_renewal_trap",        "category": "dangerous", "label": "Auto-renewal trap",            "check": "Does the contract auto-renew with a very short, buried, or unclear opt-out window?"},
    {"id": "broad_ip_assignment",      "category": "dangerous", "label": "Broad IP assignment",          "check": "Does the contract transfer all IP ownership — including prior or personal work — to the other party?"},
    {"id": "personal_guarantee",       "category": "dangerous", "label": "Personal guarantee",           "check": "Is an individual personally liable for a company's obligations under this contract?"},
    {"id": "unilateral_price_change",  "category": "dangerous", "label": "Unilateral price change",      "check": "Can one party increase fees or prices without the other party's agreement?"},
    {"id": "assignment_no_consent",    "category": "dangerous", "label": "Assignment without consent",   "check": "Can one party transfer or assign the contract to a third party without consent?"},
    {"id": "disproportionate_penalty", "category": "dangerous", "label": "Disproportionate penalty",     "check": "Are financial penalties included that far exceed the actual value of the breach?"},
    {"id": "perpetual_confidentiality","category": "dangerous", "label": "Perpetual confidentiality",    "check": "Does the NDA or confidentiality obligation last forever with no expiry date?"},
    {"id": "excessive_non_compete",    "category": "dangerous", "label": "Excessive non-compete",        "check": "Is there a non-compete clause with unreasonably wide scope, geography, or duration?"},

    # ── Unusual (non-standard language a reasonable party would not expect) ─
    {"id": "vague_termination_trigger","category": "unusual",   "label": "Vague termination trigger",    "check": "Can the contract be terminated for vague, subjective, or undefined reasons?"},
    {"id": "deemed_acceptance",        "category": "unusual",   "label": "Deemed acceptance",            "check": "Is silence or inaction treated as acceptance of changes or new terms?"},
    {"id": "unilateral_audit_rights",  "category": "unusual",   "label": "Unilateral audit rights",      "check": "Can one party audit the other at any time with minimal or no notice?"},
    {"id": "broad_data_sharing",       "category": "unusual",   "label": "Broad data sharing rights",    "check": "Can one party share, sell, or monetize the other party's data without restriction?"},
    {"id": "indefinite_term",          "category": "unusual",   "label": "Indefinite contract term",     "check": "Does the contract have no fixed end date and can run indefinitely?"},
    {"id": "full_warranty_disclaimer", "category": "unusual",   "label": "Full warranty disclaimer",     "check": "Are all warranties — including implied ones — fully disclaimed?"},
    {"id": "vague_scope",              "category": "unusual",   "label": "Vague scope of work",          "check": "Are deliverables, services, or obligations defined so vaguely that disputes about performance are likely?"},

    # ── Missing (standard protections absent from the contract) ──────────
    {"id": "no_liability_cap",         "category": "missing",   "label": "No limitation of liability",   "check": "Is there NO clause that caps the maximum financial exposure of either party?"},
    {"id": "no_termination_clause",    "category": "missing",   "label": "No termination clause",        "check": "Is there NO clause that allows either party to exit the agreement?"},
    {"id": "no_force_majeure",         "category": "missing",   "label": "No force majeure clause",      "check": "Is there NO force majeure clause protecting parties from events outside their control?"},
    {"id": "no_dispute_resolution",    "category": "missing",   "label": "No dispute resolution process","check": "Is there NO mediation or escalation process defined before going to litigation?"},
    {"id": "no_payment_terms",         "category": "missing",   "label": "No clear payment terms",       "check": "Are payment due dates, late fees, or accepted currencies NOT specified?"},
    {"id": "no_governing_law",         "category": "missing",   "label": "No governing law clause",      "check": "Is the governing law and jurisdiction for disputes NOT stated anywhere in the contract?"},
    {"id": "no_ip_ownership",          "category": "missing",   "label": "No IP ownership clause",       "check": "Is intellectual property ownership NOT clearly stated anywhere in the contract?"},
    {"id": "no_confidentiality",       "category": "missing",   "label": "No confidentiality clause",    "check": "Is there NO confidentiality or NDA clause protecting either party's information?"},
    {"id": "no_warranty",              "category": "missing",   "label": "No warranty or SLA",           "check": "Is there NO warranty, quality standard, or service level agreement (SLA) specified?"},
    {"id": "no_renewal_notice",        "category": "missing",   "label": "No renewal or expiry notice",  "check": "Is there NO requirement to notify either party before the contract expires or renews?"},
]

# ---------------------------------------------------------------------------
# Severity map — hardcoded, not decided by the model.
# Same document = same severity, every time.
# ---------------------------------------------------------------------------

_SEVERITY = {
    "unlimited_liability":      "Critical",
    "one_sided_indemnity":      "Critical",
    "personal_guarantee":       "Critical",
    "no_liability_cap":         "Critical",
    "broad_ip_assignment":      "High",
    "unilateral_amendment":     "High",
    "mandatory_arbitration":    "High",
    "auto_renewal_trap":        "High",
    "excessive_non_compete":    "High",
    "unilateral_price_change":  "High",
    "assignment_no_consent":    "High",
    "disproportionate_penalty": "High",
    "broad_data_sharing":       "High",
    "no_termination_clause":    "High",
    "no_payment_terms":         "High",
    "no_confidentiality":       "High",
    "perpetual_confidentiality":"Medium",
    "vague_termination_trigger":"Medium",
    "deemed_acceptance":        "Medium",
    "unilateral_audit_rights":  "Medium",
    "indefinite_term":          "Medium",
    "full_warranty_disclaimer": "Medium",
    "vague_scope":              "Medium",
    "no_force_majeure":         "Medium",
    "no_dispute_resolution":    "Medium",
    "no_governing_law":         "Medium",
    "no_ip_ownership":          "Medium",
    "no_warranty":              "Medium",
    "no_renewal_notice":        "Medium",
}

_LABEL = {item["id"]: item["label"]    for item in _CHECKLIST}
_CAT   = {item["id"]: item["category"] for item in _CHECKLIST}

_SORT_ORDER = {"Critical": 0, "High": 1, "Medium": 2}

_CAT_ICON = {"dangerous": "🔴", "unusual": "🟠", "missing": "🟡"}


# ---------------------------------------------------------------------------
# Build the checklist prompt
#
# "json" appears in _SYSTEM so _run_inference_json (response_format=json_object)
# is safe to use here.
# ---------------------------------------------------------------------------

def _build_messages(text: str) -> list[dict]:
    _SYSTEM = (
        "You are a contract risk lawyer reviewing contracts for dangerous, unusual, "
        "and missing clauses. For each checklist item you receive, answer with "
        "exactly one of these status values: present, absent, not_applicable. "
        "Return ONLY a JSON object — no markdown, no backticks, no explanation."
    )

    checklist_lines = "\n".join(
        f'{i+1}. id="{item["id"]}" — {item["check"]}'
        for i, item in enumerate(_CHECKLIST)
    )

    user = f"""Evaluate every item in the checklist against the contract below.

For each item return:
  "id"             : the item id exactly as given
  "status"         : "present" if found in the contract, "absent" if not found, "not_applicable" if irrelevant to this document type
  "clause_excerpt" : exact quote from the contract (max 150 chars) if status=present, otherwise null
  "why_dangerous"  : one sentence on the specific risk to the signer if status=present, otherwise null
  "recommendation" : one concrete fix or negotiation step if status=present, otherwise null

Return ONLY this JSON structure:
{{
  "results": [
    {{
      "id": "<item id>",
      "status": "present | absent | not_applicable",
      "clause_excerpt": "<quote or null>",
      "why_dangerous": "<risk or null>",
      "recommendation": "<action or null>"
    }}
  ]
}}

Checklist ({len(_CHECKLIST)} items — you MUST evaluate ALL {len(_CHECKLIST)}):
{checklist_lines}

Contract:
---
{text[:15000]}
---"""

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user},
    ]


# ---------------------------------------------------------------------------
# Build detected_flags from raw AI results
# Called separately so it can also be used on retry results
# ---------------------------------------------------------------------------

def _build_flags(results: list) -> list[dict]:
    flags = []
    for item in results:
        if not isinstance(item, dict) or item.get("status") != "present":
            continue

        item_id  = item.get("id", "")
        category = _CAT.get(item_id, "unusual")
        severity = _SEVERITY.get(item_id, "Medium")
        label    = _LABEL.get(item_id, item_id.replace("_", " ").title())

        flags.append({
            "warning":        f"⚠ {label}",
            "indicator":      _CAT_ICON.get(category, "⚠"),
            "category":       category,
            "severity":       severity,
            "clause_excerpt": item.get("clause_excerpt") or "Not present in this contract",
            "why_dangerous":  item.get("why_dangerous")  or "",
            "recommendation": item.get("recommendation") or "",
        })

    flags.sort(key=lambda f: _SORT_ORDER.get(f["severity"], 3))
    return flags


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def scan_red_flags(text: str) -> dict:
    """
    Evaluate all 29 checklist items against the contract text.

    Consistent count guarantee:
      - Same fixed checklist on every call
      - AI only answers present / absent / not_applicable per item
      - Severity assigned from hardcoded map, not by the model
      - _run_inference_json forces response_format=json_object for clean parsing
    """
    logger.info(f"[red_flag_scanner] Evaluating {len(_CHECKLIST)} checklist items...")

    messages = _build_messages(text)
    raw, in_tok, out_tok = await _run_inference_json(messages, "red_flag_scanner")
    logger.info(f"[red_flag_scanner] tokens in={in_tok} out={out_tok}")

    parsed  = extract_json_raw(raw)
    results = parsed.get("results", []) if isinstance(parsed, dict) else []

    # Retry if model did not return all items
    if len(results) < len(_CHECKLIST) // 2:
        logger.warning(
            f"[red_flag_scanner] Only {len(results)}/{len(_CHECKLIST)} items returned — retrying"
        )
        raw, in_tok2, out_tok2 = await _run_inference_json(messages, "red_flag_scanner-retry")
        parsed  = extract_json_raw(raw)
        results = parsed.get("results", []) if isinstance(parsed, dict) else []
        in_tok  += in_tok2
        out_tok += out_tok2

    if not results:
        logger.error("[red_flag_scanner] No results returned — safe default")
        return {
            "detected_flags":    [],
            "overall_risk_level": "Low",
            "summary":           "Red flag scan could not be completed for this document.",
        }

    flags = _build_flags(results)

    # Compute overall risk from hardcoded severity values
    severities = {f["severity"] for f in flags}
    if "Critical" in severities:  overall = "Critical"
    elif "High"   in severities:  overall = "High"
    elif "Medium" in severities:  overall = "Medium"
    else:                         overall = "Low"

    dangerous = sum(1 for f in flags if f["category"] == "dangerous")
    unusual   = sum(1 for f in flags if f["category"] == "unusual")
    missing   = sum(1 for f in flags if f["category"] == "missing")
    critical  = sum(1 for f in flags if f["severity"] == "Critical")
    high      = sum(1 for f in flags if f["severity"] == "High")

    logger.info(
        f"[red_flag_scanner] {len(flags)}/{len(_CHECKLIST)} flags present | "
        f"dangerous={dangerous} unusual={unusual} missing={missing} | "
        f"critical={critical} high={high} | risk={overall}"
    )

    return {
        "detected_flags":    flags,
        "overall_risk_level": overall,
        "summary":           (
            f"Scanned {len(_CHECKLIST)} checklist items. "
            f"Found {len(flags)} flag(s): "
            f"{dangerous} dangerous, {unusual} unusual, {missing} missing protection(s). "
            f"Overall risk: {overall}."
        ),
    }
