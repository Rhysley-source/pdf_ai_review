import logging
from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Three categories the scanner checks:
#
#  1. DANGEROUS clauses  — clauses that are present and harmful to the signer
#  2. UNUSUAL clauses    — non-standard language a reasonable party would not expect
#  3. MISSING clauses    — standard protections that are absent from the document
# ---------------------------------------------------------------------------

_DANGEROUS_CLAUSES = [
    "Unlimited liability — no cap on financial damages owed by one party",
    "One-sided indemnification — only one party bears all legal costs and losses",
    "Unilateral amendment — one party can change contract terms without the other's consent",
    "Mandatory arbitration — waives the right to go to court; forces private arbitration",
    "Auto-renewal trap — contract renews automatically with a very short or buried opt-out window",
    "Broad IP assignment — all intellectual property, including personal/prior work, transferred to the other party",
    "Personal guarantee — an individual is personally liable for a company's obligations",
    "Unilateral price change — one party can increase fees or rates without the other's agreement",
    "Assignment without consent — contract can be transferred to a third party without agreement",
    "Disproportionate penalty — financial penalties far exceed the actual value of the breach",
    "Perpetual confidentiality — NDA or confidentiality obligation never expires",
    "Excessive non-compete — scope, geography, or duration is unreasonably wide",
    "Non-solicitation clause — prevents hiring or working with the other party's staff or clients",
]

_UNUSUAL_CLAUSES = [
    "Unusual termination trigger — contract can be terminated for vague or subjective reasons",
    "Deemed acceptance — silence or inaction is treated as acceptance of changes or new terms",
    "Unusual intellectual property ownership — IP ownership structure deviates from industry norms",
    "Unilateral audit rights — one party can audit the other at any time with minimal notice",
    "Unusual payment structure — payment terms, clawbacks, or holdbacks that are non-standard",
    "Broad data sharing rights — one party can share or monetize the other's data without restriction",
    "Non-standard governing law — contract is governed by a jurisdiction unusual for this type of deal",
    "Rolling or indefinite term — contract has no fixed end date and can run indefinitely",
    "Unconscionable warranty disclaimer — all warranties, even implied ones, disclaimed entirely",
    "Vague scope of work — deliverables are undefined, enabling scope creep or performance disputes",
]

_STANDARD_PROTECTIONS = [
    "Limitation of liability clause",
    "Termination for convenience clause",
    "Force majeure clause",
    "Dispute resolution / mediation step before litigation",
    "Clearly defined payment terms and due dates",
    "Governing law and jurisdiction clause",
    "Confidentiality / NDA with a defined expiry",
    "Intellectual property ownership clearly stated",
    "Warranty or service level agreement (SLA)",
    "Contract renewal / expiry notice requirement",
]


def _build_prompt(text: str) -> str:
    dangerous = "\n".join(f"  - {c}" for c in _DANGEROUS_CLAUSES)
    unusual   = "\n".join(f"  - {c}" for c in _UNUSUAL_CLAUSES)
    standard  = "\n".join(f"  - {c}" for c in _STANDARD_PROTECTIONS)

    return f"""You are a contract risk lawyer. Analyse the contract below for three things:

1. DANGEROUS CLAUSES — clauses present in the document that are harmful to the signer:
{dangerous}

2. UNUSUAL CLAUSES — non-standard language a reasonable party would not expect to find:
{unusual}

3. MISSING STANDARD PROTECTIONS — common clauses that are absent from this contract:
{standard}

Return ONLY valid JSON in exactly this structure (no markdown, no backticks):
{{
  "detected_flags": [
    {{
      "warning": "⚠ <short one-line warning describing exactly what was found>",
      "category": "Dangerous | Unusual | Missing",
      "severity": "Critical | High | Medium",
      "clause_excerpt": "<exact quote or closest paraphrase from the contract, max 150 chars — or 'Not present' if it is a missing clause>",
      "why_dangerous": "<one sentence: the specific business or legal risk this creates for the signer>",
      "recommendation": "<one concrete action: how to negotiate, add, or remove this clause>"
    }}
  ],
  "overall_risk_level": "Critical | High | Medium | Low",
  "summary": "<2–3 sentence executive summary of the contract's overall risk profile>"
}}

Rules:
- Only include flags that apply to THIS specific document
- For missing clauses, set clause_excerpt to "Not present in this contract"
- warning must always start with ⚠
- overall_risk_level: Critical if any Critical flag exists, High if any High, else Medium or Low
- If the document is not a contract, return detected_flags as [] and overall_risk_level as "Low"

Contract:
---
{text[:15000]}
---"""


def _build_retry_prompt(text: str) -> str:
    return f"""Return ONLY a raw JSON object. No markdown, no backticks, no explanation.

{{
  "detected_flags": [],
  "overall_risk_level": "Low",
  "summary": ""
}}

Fill the above JSON by finding dangerous, unusual, and missing clauses in this contract:
---
{text[:10000]}
---"""


async def scan_red_flags(text: str) -> dict:
    """
    Scan contract text for:
      - Dangerous clauses (harmful to the signer)
      - Unusual clauses (non-standard language)
      - Missing standard protections (absent clauses)

    Returns structured red flags with ⚠ warnings, severity, and recommendations.
    """
    logger.info("[red_flag_scanner] Starting scan...")

    raw = await run_llm("", _build_prompt(text))
    logger.info(f"[red_flag_scanner] Raw output: {len(raw)} chars")

    result = extract_json_raw(raw)

    if not result or not isinstance(result.get("detected_flags"), list):
        logger.warning("[red_flag_scanner] Parse failed — retrying with stricter prompt")
        raw    = await run_llm("", _build_retry_prompt(text))
        result = extract_json_raw(raw)

    if not result:
        logger.error("[red_flag_scanner] Both attempts failed — returning safe default")
        return {
            "detected_flags":    [],
            "overall_risk_level": "Low",
            "summary":           "Red flag scan could not be completed for this document.",
        }

    result.setdefault("detected_flags",    [])
    result.setdefault("overall_risk_level", "Low")
    result.setdefault("summary",           "")

    total     = len(result["detected_flags"])
    critical  = sum(1 for f in result["detected_flags"] if f.get("severity") == "Critical")
    high      = sum(1 for f in result["detected_flags"] if f.get("severity") == "High")
    dangerous = sum(1 for f in result["detected_flags"] if f.get("category") == "Dangerous")
    unusual   = sum(1 for f in result["detected_flags"] if f.get("category") == "Unusual")
    missing   = sum(1 for f in result["detected_flags"] if f.get("category") == "Missing")

    logger.info(
        f"[red_flag_scanner] Done — {total} flag(s) | "
        f"critical={critical} high={high} | "
        f"dangerous={dangerous} unusual={unusual} missing={missing} | "
        f"risk={result['overall_risk_level']}"
    )
    return result
