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
# Returns (slug, human_label)
#   slug       — one of the 7 known profile keys, used to select risk profile
#   human_label — specific document name the LLM identified (e.g. "Partnership Deed")
# ---------------------------------------------------------------------------

async def _detect_document_type(text: str) -> tuple[str, str]:
    prompt = f"""Analyse the document and return a JSON object with exactly two fields:

"slug"  — classify into EXACTLY ONE of: contract, employment, nda, lease, invoice, resume, other
"label" — the specific document type as a short human-readable name (2-5 words)

Slug definitions:
- contract   → service agreements, vendor agreements, terms & conditions, MOU, partnership deeds
- employment → offer letters, employment agreements, appointment letters, HR documents
- nda        → non-disclosure agreements, confidentiality agreements
- lease      → rental agreements, property leases, tenancy agreements, leave and licence
- invoice    → billing documents, receipts, payment summaries, purchase orders
- resume     → CV, job profiles, candidate profiles
- other      → anything not covered above (legal notices, affidavits, power of attorney,
               financial statements, medical reports, insurance policies, wills, etc.)

Label rules:
- Be specific — never return "Other Document" or "Unknown Document"
- Use the actual document name as it would appear on the document itself
- Examples by slug:
    contract   → "Service Agreement", "Vendor Contract", "Memorandum of Understanding"
    employment → "Job Offer Letter", "Appointment Letter", "Employment Contract"
    nda        → "Non-Disclosure Agreement", "Mutual Confidentiality Agreement"
    lease      → "Residential Lease Agreement", "Commercial Lease Deed"
    invoice    → "Tax Invoice", "Proforma Invoice", "Purchase Order"
    resume     → "Curriculum Vitae", "Resume"
    other      → "Partnership Deed", "Power of Attorney", "Affidavit",
                 "Insurance Policy", "Will and Testament", "Legal Notice",
                 "Financial Statement", "Medical Report", "Loan Agreement", etc.

Return ONLY this JSON object — no explanation, no markdown:
{{"slug": "<one of 7 slugs>", "label": "<specific document name>"}}

Document (first 1500 chars):
\"\"\"{text[:1500]}\"\"\""""

    raw = await run_llm("", prompt)

    # Try to parse JSON response
    parsed = extract_json_from_text(raw)
    slug  = (parsed.get("slug")  or "").lower().strip()
    label = (parsed.get("label") or "").strip()

    # Validate slug — fall back to word-match then "other"
    if slug not in _RISK_PROFILES:
        for known in _RISK_PROFILES:
            if known in raw.lower():
                slug = known
                break
        else:
            slug = "other"

    # If LLM did not give a specific label, derive a readable fallback
    if not label or label.lower() in ("other document", "unknown document", "unknown", "other"):
        _SLUG_LABELS = {
            "contract":   "Contract / Legal Agreement",
            "employment": "Employment Agreement",
            "nda":        "Non-Disclosure Agreement",
            "lease":      "Lease Agreement",
            "invoice":    "Invoice / Billing Document",
            "resume":     "Resume / CV",
            "other":      "General Document",
        }
        label = _SLUG_LABELS[slug]

    return slug, label


# ---------------------------------------------------------------------------
# Step 2a — Dynamic risk analysis for unknown document types (slug == "other")
#
# Instead of a fixed checklist, the LLM:
#   1. Identifies what risk categories are relevant to THIS specific document type
#   2. Scans the document against those categories in the same pass
#   3. Identifies required fields that should be present but are missing
#
# This means ANY document — Power of Attorney, Insurance Policy, Loan Agreement,
# Will, Affidavit, Medical Report, etc. — gets a fully relevant risk analysis.
# ---------------------------------------------------------------------------

async def _analyze_risks_dynamic(text: str, doc_label: str) -> dict:
    prompt = f"""You are a senior legal and financial risk analyst.

A document identified as "{doc_label}" has been uploaded for risk analysis.

YOUR TASK — perform a complete risk analysis in a single pass:

PART 1 — RISK DETECTION:
Based on your expert knowledge of "{doc_label}" documents, identify and flag any risks
present in this document. Think about:
- Clauses that are dangerous or unfair to one party
- Unusual or non-standard language
- Excessive liability, penalties, or obligations
- Rights that are waived or restricted
- Ambiguous language that could be exploited

PART 2 — MISSING REQUIRED FIELDS:
Identify fields or sections that are typically required in a "{doc_label}" but are
missing or unclear in this document.

OUTPUT FORMAT — return ONLY valid JSON in exactly this structure:
{{
  "document_type": "other",
  "document_label": "{doc_label}",
  "risk_score": <integer calculated as described in Rules below, capped at 100>,
  "detected_risks": [
    {{
      "risk_name": "<specific risk name relevant to {doc_label}>",
      "severity": "High | Medium | Low",
      "clause_found": "<exact quote or short description from the document, or 'Not found'>",
      "impact": "<why this is dangerous or problematic>",
      "mitigation": "<how to fix, negotiate, or protect against this>"
    }}
  ],
  "missing_fields": [
    {{
      "field_name": "<field or section missing from this {doc_label}>",
      "importance": "Critical | Important | Optional",
      "reason": "<why this field is needed in a {doc_label}>"
    }}
  ],
  "overall_assessment": "<2-3 sentence executive summary of the overall risk profile>"
}}

Rules:
- Risks must be SPECIFIC to "{doc_label}" — not generic boilerplate
- Only include risks actually present in the document
- Only include fields actually missing
- If no risks found, return detected_risks as []
- If no fields missing, return missing_fields as []
- risk_score = min(
    (High*30 + Medium*15 + Low*5) + (Critical*25 + Important*10 + Optional*5),
    100
  )

Document:
---
{text[:12000]}
---"""

    logger.info(f"[risk_detection] Running dynamic risk analysis for '{doc_label}'...")

    raw_output = await run_llm("", prompt)
    logger.info(f"[risk_detection] Dynamic raw output length: {len(raw_output)} chars")

    result = extract_json_from_text(raw_output)

    # Retry with shorter document if parse failed
    if not result or (not result.get("detected_risks") and not result.get("overall_assessment")):
        logger.warning("[risk_detection] Dynamic: empty parse — retrying")
        retry_prompt = f"""Return ONLY a raw JSON object. No markdown, no backticks.

Analyse the following "{doc_label}" document for risks and missing fields.
Return this exact structure filled in:

{{
  "document_type": "other",
  "document_label": "{doc_label}",
  "risk_score": 0,
  "detected_risks": [],
  "missing_fields": [],
  "overall_assessment": ""
}}

Document:
---
{text[:8000]}
---"""
        raw_output = await run_llm("", retry_prompt)
        result = extract_json_from_text(raw_output)

    if not result:
        logger.error(f"[risk_detection] Dynamic: both attempts failed for '{doc_label}'")
        result = {
            "document_type": "other",
            "document_label": doc_label,
            "risk_score": 0,
            "detected_risks": [],
            "missing_fields": [],
            "overall_assessment": "Risk analysis could not be completed for this document.",
        }

    return result


# ---------------------------------------------------------------------------
# Step 2b — Fixed-profile risk analysis for the 6 known document types
# ---------------------------------------------------------------------------

async def _analyze_risks_for_type(text: str, doc_type: str, doc_label: str) -> dict:
    profile = _RISK_PROFILES[doc_type]
    risks_list = "\n".join(f"    - {r}" for r in profile["risks"])
    fields_list = "\n".join(f"    - {f}" for f in profile["required_fields"])

    prompt = f"""
You are a legal and financial risk analyst specializing in {doc_label} documents.

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
  "document_label": "{doc_label}",
  "risk_score": <integer calculated EXACTLY as described below, capped at 100>,
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
- risk_score MUST be calculated by adding BOTH components below, then capping at 100:
    Component 1 — detected_risks severity:
      High * 30 + Medium * 15 + Low * 5
    Component 2 — missing_fields importance:
      Critical * 25 + Important * 10 + Optional * 5
    Final: min(Component1 + Component2, 100)
- Example: 1 High risk + 1 Critical missing field = (1*30) + (1*25) = 55

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
  "document_label": "{doc_label}",
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
            "document_label": doc_label,
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
    Step 1: detect document type slug + specific human-readable label

    Step 2: branch based on slug
      - Known type (contract / employment / nda / lease / invoice / resume)
        → _analyze_risks_for_type  — fixed profile checklist, deterministic
      - Unknown type (slug == "other", e.g. Power of Attorney, Loan Agreement)
        → _analyze_risks_dynamic   — LLM generates relevant risks for that specific doc
    """
    doc_type, doc_label = await _detect_document_type(text)
    logger.info(f"[risk_detection] Detected document type: {doc_type} ('{doc_label}')")

    _KNOWN_TYPES = {"contract", "employment", "nda", "lease", "invoice", "resume"}

    if doc_type in _KNOWN_TYPES:
        logger.info(f"[risk_detection] Using fixed profile for '{doc_type}'")
        analysis = await _analyze_risks_for_type(text, doc_type, doc_label)
    else:
        logger.info(f"[risk_detection] Using dynamic analysis for '{doc_label}'")
        analysis = await _analyze_risks_dynamic(text, doc_label)

    return {
        "status": "success",
        "analysis_type": "risk_detection",
        "document_type": doc_label,
        "data": analysis,
    }
