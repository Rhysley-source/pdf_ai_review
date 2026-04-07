import logging
from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document-type risk profiles
# Each entry defines:
#   risks        — specific risk categories to look for
#   required_fields — data that MUST be present in the document; missing ones
#                     are flagged as a risk themselves
# ---------------------------------------------------------------------------

_RISK_PROFILES = {
    "contract": {
        "label": "Contract / Legal Agreement",
        "risks": [
            "Auto-renewal: clauses that silently commit to another term",
            "Indemnity: broad financial liability placed on one party",
            "Termination penalties: excessive exit costs or lock-in periods",
            "Non-compete / Non-solicitation: restrictions on future business or hiring",
            "Missing liability caps: no upper limit on damages owed",
            "Jurisdiction / Governing law: disputes forced in unfavorable location",
            "Unilateral amendment: one party can change terms without consent",
            "Intellectual property assignment: broad IP ownership transfer",
            "Force majeure gaps: events that excuse performance not clearly defined",
            "Dispute resolution: mandatory arbitration removing right to sue",
        ],
        "required_fields": [
            "Effective Date", "Contract Term", "Termination Notice Period",
            "Governing Law", "Payment Terms", "Renewal Conditions",
        ],
    },
    "employment": {
        "label": "Employment Agreement / Offer Letter",
        "risks": [
            "At-will termination: employer can fire without cause or notice",
            "Non-compete clause: restricts working for competitors after leaving",
            "Non-solicitation: prevents hiring former colleagues",
            "Broad IP assignment: all inventions owned by employer including personal projects",
            "Clawback provisions: bonuses/equity can be taken back",
            "Arbitration clause: waives right to sue in court",
            "Probation period risks: reduced protections during probation",
            "Vague performance metrics: subjective criteria for termination",
            "Relocation clauses: forced relocation without compensation",
            "Salary/equity not clearly defined",
        ],
        "required_fields": [
            "Start Date", "Salary / Compensation", "Notice Period",
            "Equity / Bonus Terms", "Job Title / Role", "Governing Law",
        ],
    },
    "nda": {
        "label": "Non-Disclosure Agreement",
        "risks": [
            "Overly broad definition of confidential information",
            "No expiry on confidentiality obligations: lasts forever",
            "One-sided obligations: only one party bound",
            "No carve-outs for publicly available information",
            "Excessive remedies: injunctions and unlimited damages",
            "Residuals clause: allows use of retained memory of confidential info",
            "No clear permitted disclosure exceptions (legal, regulatory)",
            "Missing return/destruction of information clause",
        ],
        "required_fields": [
            "Effective Date", "Duration of Confidentiality", "Governing Law",
            "Definition of Confidential Information", "Permitted Disclosures",
        ],
    },
    "lease": {
        "label": "Lease / Rental Agreement",
        "risks": [
            "Automatic rent escalation without cap",
            "Landlord entry without notice or with very short notice",
            "Broad tenant liability for all damages including normal wear and tear",
            "Early termination penalty: excessive fees to exit lease",
            "Restriction on subletting or assignment",
            "Security deposit terms: unclear or hard to recover",
            "Maintenance responsibility shifted entirely to tenant",
            "No clear dispute resolution process",
            "Renewal at landlord's discretion only",
        ],
        "required_fields": [
            "Lease Start Date", "Lease End Date", "Monthly Rent",
            "Security Deposit", "Notice Period to Vacate", "Renewal Terms",
        ],
    },
    "invoice": {
        "label": "Invoice / Billing Document",
        "risks": [
            "Missing payment due date: no clear deadline",
            "No late payment penalty terms defined",
            "Vague description of goods/services delivered",
            "Missing tax breakdown (VAT, GST, etc.)",
            "No dispute resolution window for billing errors",
            "Currency not specified for international transactions",
            "No PO number or reference for tracking",
        ],
        "required_fields": [
            "Invoice Number", "Invoice Date", "Due Date",
            "Vendor Name", "Total Amount", "Tax Amount",
            "Payment Method / Bank Details",
        ],
    },
    "resume": {
        "label": "Resume / CV",
        "risks": [
            "Employment gap: unexplained periods of inactivity",
            "Missing contact information",
            "No quantified achievements (only responsibilities listed)",
            "Vague job titles that don't reflect actual role",
            "Missing education details or dates",
            "No skills section",
            "Inconsistent date formats or chronology",
        ],
        "required_fields": [
            "Full Name", "Contact Information (email/phone)",
            "Work Experience with Dates", "Education", "Skills",
        ],
    },
    "other": {
        "label": "General Document",
        "risks": [
            "Ambiguous obligations: unclear who is responsible for what",
            "Missing dates or validity period",
            "No signatures or authorization section",
            "Undefined terms or jargon without explanation",
            "Inconsistent data or figures within the document",
            "Missing governing law or jurisdiction",
            "No dispute resolution clause",
        ],
        "required_fields": [
            "Document Date", "Parties Involved", "Purpose / Subject Matter",
        ],
    },
}


# ---------------------------------------------------------------------------
# Step 1 — Detect document type
# ---------------------------------------------------------------------------

async def _detect_document_type(text: str) -> str:
    prompt = f"""Classify the document into EXACTLY ONE of these categories:
contract, employment, nda, lease, invoice, resume, other

Definitions:
- contract   → service agreements, vendor agreements, terms & conditions
- employment → offer letters, employment agreements, HR documents
- nda        → non-disclosure agreements, confidentiality agreements
- lease      → rental agreements, property leases, tenancy agreements
- invoice    → billing documents, receipts, payment summaries
- resume     → CV, job profiles, candidate details
- other      → anything else

Return ONLY one word. No explanation. No punctuation.

Document:
\"\"\"{text[:1500]}\"\"\""""

    # Pass empty string as text — document is already in the prompt
    result = await run_llm("", prompt)
    result = result.lower().strip()

    for label in _RISK_PROFILES:
        if label in result:
            return label
    return "other"


# ---------------------------------------------------------------------------
# Step 2 — Run type-specific risk analysis
# ---------------------------------------------------------------------------

async def _analyze_risks_for_type(text: str, doc_type: str) -> dict:
    profile = _RISK_PROFILES[doc_type]
    risks_list = "\n".join(f"    - {r}" for r in profile["risks"])
    fields_list = "\n".join(f"    - {f}" for f in profile["required_fields"])

    prompt = f"""
You are a legal and financial risk analyst specializing in {profile["label"]} documents.

TASK 1 — RISK DETECTION:
Scan the document for the following risk categories and flag any that are present:
{risks_list}

TASK 2 — MISSING REQUIRED FIELDS:
Check if the following required fields are present in the document.
Flag any that are missing or unclear as an additional risk:
{fields_list}

OUTPUT FORMAT — return ONLY valid JSON in exactly this structure:
{{
  "document_type": "{doc_type}",
  "document_label": "{profile["label"]}",
  "risk_score": <integer calculated EXACTLY as: (High_count * 30) + (Medium_count * 15) + (Low_count * 5), capped at 100>,
  "detected_risks": [
    {{
      "risk_name": "<risk category name>",
      "severity": "High | Medium | Low",
      "clause_found": "<exact quote or short description of the clause, or 'Not found'>",
      "impact": "<why this is dangerous to the signer>",
      "mitigation": "<how to negotiate or fix this>"
    }}
  ],
  "missing_fields": [
    {{
      "field_name": "<field that is missing>",
      "importance": "Critical | Important | Optional",
      "reason": "<why this field matters>"
    }}
  ],
  "overall_assessment": "<executive summary of the document risk profile in 2-3 sentences>"
}}

Rules:
- Only include risks that are actually present in the document
- Only include fields that are actually missing
- If no risks found, return detected_risks as []
- If no fields missing, return missing_fields as []
- risk_score MUST be calculated as: (High_count * 30) + (Medium_count * 15) + (Low_count * 5), capped at 100
- Example: 2 High + 1 Medium = (2*30) + (1*15) = 75

Document Text:
---
{text[:12000]}
---
"""

    logger.info(f"[risk_detection] Running {doc_type} risk analysis...")

    # Pass empty string as text — document is already embedded in the prompt
    # to avoid sending the document twice and hitting token limits
    raw_output = await run_llm("", prompt)
    logger.info(f"[risk_detection] Raw output length: {len(raw_output)} chars")
    logger.debug(f"[risk_detection] Raw LLM output: {raw_output[:500]}")

    result = extract_json_from_text(raw_output)

    # Retry once if we got blank/empty data
    if not result or not result.get("detected_risks") and not result.get("overall_assessment"):
        logger.warning("[risk_detection] Empty parse on attempt 1 — retrying with stricter prompt")
        retry_prompt = f"""Return ONLY a raw JSON object. No markdown, no backticks, no explanation.

{{
  "document_type": "{doc_type}",
  "document_label": "{profile["label"]}",
  "risk_score": 0,
  "detected_risks": [],
  "missing_fields": [],
  "overall_assessment": ""
}}

Now fill in the above JSON by analysing this document for risks:
{risks_list}

Document:
---
{text[:8000]}
---"""
        raw_output = await run_llm("", retry_prompt)
        logger.info(f"[risk_detection] Retry raw output length: {len(raw_output)} chars")
        result = extract_json_from_text(raw_output)

    if not result:
        logger.error("[risk_detection] Both attempts returned empty — returning safe default")
        result = {
            "document_type": doc_type,
            "document_label": profile["label"],
            "risk_score": 0,
            "detected_risks": [],
            "missing_fields": [],
            "overall_assessment": "Risk analysis could not be completed for this document.",
        }

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def analyze_document_risks(text: str) -> dict:
    """
    Step 1: detect document type
    Step 2: run type-specific risk analysis with relevant risk categories
            and check for required fields that are missing
    """
    doc_type = await _detect_document_type(text)
    logger.info(f"[risk_detection] Detected document type: {doc_type}")

    analysis = await _analyze_risks_for_type(text, doc_type)

    return {
        "status": "success",
        "analysis_type": "risk_detection",
        "document_type": doc_type,
        "data": analysis,
    }
