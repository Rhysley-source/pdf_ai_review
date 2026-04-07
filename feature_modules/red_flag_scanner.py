import logging
from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Red flag categories the AI scans for
# ---------------------------------------------------------------------------

_RED_FLAGS = [
    "Unlimited liability: no cap on financial damages owed by one party",
    "Missing termination clause: no way to exit the agreement",
    "Unilateral amendment: one party can change terms without consent",
    "Auto-renewal trap: contract renews automatically with very short opt-out window",
    "Broad IP assignment: all intellectual property — including personal projects — transferred to the other party",
    "One-sided indemnification: only one party bears all legal costs and damages",
    "Mandatory arbitration: waives the right to sue in court",
    "Excessive non-compete: overly wide scope, geography, or duration restricting future work",
    "No governing law specified: jurisdiction for disputes is unclear",
    "Missing payment terms: no due date, interest on late payments, or currency",
    "Disproportionate penalty clause: penalties far exceed the value of the breach",
    "No force majeure clause: no protection for events outside a party's control",
    "Perpetual confidentiality: NDA obligations never expire",
    "Personal guarantee: individual is personally liable for a company's obligations",
    "Unilateral price change: one party can raise prices without consent",
    "Vague scope of work: deliverables not clearly defined, enabling scope creep disputes",
    "No dispute resolution process: no mediation or escalation path before litigation",
    "Unreasonable notice period: termination requires excessive advance notice",
    "Assignment without consent: contract can be transferred to a third party without agreement",
]

_SYSTEM_PROMPT = """You are a contract risk lawyer specializing in identifying dangerous and unusual contract language.

Scan the contract for the following red flag categories:
{flags_list}

For each red flag found, return a structured entry.
If a red flag category is NOT present in the document, do NOT include it.
If no red flags are found at all, return an empty detected_flags array.

Return ONLY valid JSON in exactly this structure:
{{
  "detected_flags": [
    {{
      "warning": "⚠ <short one-line warning — what was found>",
      "severity": "Critical | High | Medium",
      "clause_excerpt": "<exact quote or closest paraphrase from the contract, max 120 chars, or 'Not explicitly stated'>",
      "why_dangerous": "<one sentence explaining the business risk to the signer>",
      "recommendation": "<one concrete action to fix or negotiate this>"
    }}
  ],
  "missing_protections": [
    "<name of a standard clause that is absent and should be added>"
  ],
  "overall_risk_level": "Critical | High | Medium | Low",
  "summary": "<2-3 sentence executive summary of the contract's risk profile>"
}}

Rules:
- Only flag issues actually present (or absent) in THIS document
- warning must start with ⚠
- severity: Critical = could cause major financial/legal harm; High = significant risk; Medium = worth noting
- overall_risk_level is based on the worst severity found; if no flags → Low
- missing_protections lists clauses that are absent but should be present (e.g. limitation of liability, termination for convenience)
- Return raw JSON only — no markdown, no backticks, no explanation
"""


async def scan_red_flags(text: str) -> dict:
    """
    Scan contract text for dangerous or unusual language.
    Returns structured red flags with warnings, severity, and recommendations.
    """
    flags_list = "\n".join(f"  - {f}" for f in _RED_FLAGS)
    system = _SYSTEM_PROMPT.format(flags_list=flags_list)

    prompt = f"""Contract text:
---
{text[:15000]}
---

Scan the above contract and return the JSON red flag report."""

    logger.info("[red_flag_scanner] Starting scan...")
    raw = await run_llm("", system + "\n\n" + prompt)
    logger.info(f"[red_flag_scanner] Raw output: {len(raw)} chars")

    result = extract_json_raw(raw)

    # Retry with stricter prompt if parse failed or result is empty
    if not result or not isinstance(result.get("detected_flags"), list):
        logger.warning("[red_flag_scanner] Parse failed — retrying with stricter prompt")
        retry_prompt = f"""Return ONLY a raw JSON object. No markdown, no backticks.

{{
  "detected_flags": [],
  "missing_protections": [],
  "overall_risk_level": "Low",
  "summary": ""
}}

Fill the above JSON by scanning this contract for dangerous clauses:
---
{text[:10000]}
---"""
        raw = await run_llm("", retry_prompt)
        result = extract_json_raw(raw)

    if not result:
        logger.error("[red_flag_scanner] Both attempts failed — returning safe default")
        return {
            "detected_flags": [],
            "missing_protections": [],
            "overall_risk_level": "Low",
            "summary": "Red flag scan could not be completed for this document.",
        }

    # Ensure required keys exist
    result.setdefault("detected_flags", [])
    result.setdefault("missing_protections", [])
    result.setdefault("overall_risk_level", "Low")
    result.setdefault("summary", "")

    logger.info(
        f"[red_flag_scanner] Done — "
        f"{len(result['detected_flags'])} flag(s) | "
        f"{len(result['missing_protections'])} missing protection(s) | "
        f"risk={result['overall_risk_level']}"
    )
    return result
