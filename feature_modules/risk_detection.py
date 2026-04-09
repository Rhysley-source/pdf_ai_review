import asyncio
import logging
import re
from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document-type risk profiles (fixed-profile pass for known types)
# ---------------------------------------------------------------------------

_RISK_PROFILES = {
    "contract": {
        "label": "Contract / Legal Agreement",
        "risks": [
            "Auto-renewal: clauses that silently commit to another term without notice",
            "Indemnity: broad financial liability placed on one party",
            "Termination penalties: excessive exit costs or lock-in periods",
            "Non-compete / Non-solicitation: restrictions on future business or hiring",
            "Missing liability caps: no upper limit on damages owed",
            "Jurisdiction / Governing law: disputes forced in unfavorable location",
            "Unilateral amendment: one party can change terms without consent",
            "Intellectual property assignment: broad IP ownership transfer including prior works",
            "Force majeure gaps: events that excuse performance not clearly defined",
            "Dispute resolution: mandatory arbitration removing right to sue in court",
            "Payment terms: vague or one-sided payment obligations",
            "Assignment clause: contract can be transferred to unknown third party",
            "Warranty disclaimers: all warranties disclaimed leaving no recourse",
            "Liquidated damages: pre-set penalties that may be disproportionate",
        ],
        "required_fields": [
            "Effective Date", "Contract Term / Duration", "Termination Notice Period",
            "Governing Law / Jurisdiction", "Payment Terms and Schedule",
            "Renewal Conditions", "Parties' Full Legal Names", "Signatures",
        ],
    },
    "employment": {
        "label": "Employment Agreement / Offer Letter",
        "risks": [
            "At-will termination: employer can fire without cause or notice",
            "Non-compete clause: restricts working for competitors after leaving",
            "Non-solicitation: prevents hiring former colleagues or approaching clients",
            "Broad IP assignment: all inventions owned by employer including personal projects",
            "Clawback provisions: bonuses or equity can be taken back",
            "Arbitration clause: waives right to sue in court",
            "Probation period risks: reduced protections during probation",
            "Vague performance metrics: subjective criteria that could justify termination",
            "Relocation clauses: forced relocation without sufficient compensation",
            "Salary or equity not clearly defined or subject to change",
            "Garden leave: paid but barred from working during notice period",
            "Moonlighting restriction: barred from any outside work or freelancing",
        ],
        "required_fields": [
            "Start Date", "Salary / Base Compensation", "Notice Period (both parties)",
            "Equity / Bonus Terms", "Job Title / Role", "Governing Law",
            "Probation Period Terms", "Benefits Summary",
        ],
    },
    "nda": {
        "label": "Non-Disclosure Agreement",
        "risks": [
            "Overly broad definition of confidential information covering everything",
            "No expiry on confidentiality obligations: lasts forever",
            "One-sided obligations: only one party is bound",
            "No carve-outs for publicly available or independently developed information",
            "Excessive remedies: injunctions and unlimited damages for any breach",
            "Residuals clause: allows use of retained memory of confidential info",
            "No clear permitted disclosure exceptions (legal, regulatory)",
            "Missing return or destruction of information clause",
        ],
        "required_fields": [
            "Effective Date", "Duration of Confidentiality", "Governing Law",
            "Definition of Confidential Information", "Permitted Disclosures",
            "Parties' Full Legal Names", "Signatures",
        ],
    },
    "lease": {
        "label": "Lease / Rental Agreement",
        "risks": [
            "Automatic rent escalation without a cap or notice requirement",
            "Landlord entry without notice or with very short notice period",
            "Broad tenant liability for all damages including normal wear and tear",
            "Early termination penalty: excessive fees to exit the lease",
            "Restriction on subletting or assignment without landlord's consent",
            "Security deposit terms: conditions for withholding are unclear or overly broad",
            "Maintenance responsibility shifted entirely to tenant",
            "No clear dispute resolution process",
            "Renewal at landlord's discretion only",
            "Holding over penalty: excessive charges if tenant stays past end date",
        ],
        "required_fields": [
            "Lease Start Date", "Lease End Date", "Monthly Rent",
            "Security Deposit Amount and Return Conditions", "Notice Period to Vacate",
            "Renewal Terms", "Property Address", "Permitted Use",
        ],
    },
    "invoice": {
        "label": "Invoice / Billing Document",
        "risks": [
            "Missing payment due date: no clear deadline for payment",
            "No late payment penalty terms defined",
            "Vague or incomplete description of goods or services delivered",
            "Missing tax breakdown (VAT, GST, sales tax, etc.)",
            "No dispute resolution window for billing errors",
            "Currency not specified for international transactions",
            "No PO number or reference number for tracking",
        ],
        "required_fields": [
            "Invoice Number", "Invoice Date", "Payment Due Date",
            "Vendor Name and Address", "Client Name and Address",
            "Total Amount Due", "Tax Amount", "Payment Method / Bank Details",
        ],
    },
    "resume": {
        "label": "Resume / CV",
        "risks": [
            "Unfilled template placeholders: [brackets] used instead of real data",
            "Employment gap: unexplained periods of inactivity",
            "Missing contact information: no email, phone, or location",
            "No quantified achievements: only responsibilities listed, no metrics",
            "Vague or generic job titles that don't reflect actual role",
            "Missing or incomplete education details",
            "Skills section missing or outdated",
            "Inconsistent date formats or chronology issues",
            "No professional summary or objective statement",
            "LinkedIn, GitHub, or portfolio links missing for technical roles",
        ],
        "required_fields": [
            "Full Name", "Email Address", "Phone Number", "Location",
            "Professional Summary or Objective",
            "Work Experience with Company Names and Dates",
            "Education with Institution and Year", "Skills Section",
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
            "Unfilled template placeholders: [brackets] used instead of real data",
        ],
        "required_fields": [
            "Document Date", "Parties Involved",
            "Purpose / Subject Matter", "Signatures or Authorization",
        ],
    },
}

_KNOWN_TYPES = {"contract", "employment", "nda", "lease", "invoice", "resume"}

_SEVERITY_RANK   = {"High": 3, "Medium": 2, "Low": 1}
_IMPORTANCE_RANK = {"Critical": 3, "Important": 2, "Optional": 1}

_CHUNK_SIZE    = 20_000
_CHUNK_OVERLAP = 1_000

# Matches [placeholder text] but NOT pure numeric citations like [1], [42]
_PLACEHOLDER_RE = re.compile(r'\[([^\]\[]{2,60})\]')
_NUMERIC_RE     = re.compile(r'^\s*\d+\s*$')


# ===========================================================================
# MANAGER FUNCTION 1 — Preprocessing
# Sanitizes raw document text before any LLM call.
# Fixes: openai.BadRequestError 400 caused by null bytes and control chars
#        in PDF-extracted text breaking JSON serialization.
# ===========================================================================

def manage_preprocessing(text: str) -> str:
    """
    Cleans document text so it can be safely sent to the OpenAI API.

    Removes:
    - Null bytes (\\x00) — primary cause of 400 Bad Request
    - Non-printable control characters (\\x01–\\x1f except \\t, \\n, \\r)
    - Surrogate unicode characters that cause JSON encode errors

    Normalizes:
    - Repeated spaces / tabs → single space
    - More than 3 consecutive blank lines → 2 blank lines
    """
    if not text:
        return ""

    # Replace null bytes
    text = text.replace('\x00', ' ')

    # Remove control characters except tab (0x09), newline (0x0a), carriage return (0x0d)
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Fix broken unicode (replace invalid sequences instead of crashing)
    text = text.encode('utf-8', errors='replace').decode('utf-8')

    # Normalize whitespace — collapse spaces/tabs, cap blank lines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text.strip()


# ===========================================================================
# MANAGER FUNCTION 2 — Document Type Detection
# Identifies what kind of document was uploaded.
# ===========================================================================

async def manage_type_detection(text: str) -> tuple[str, str]:
    """
    Classifies the document into one of 7 slugs and returns a human-readable label.
    Uses the first 4000 chars for fast, accurate classification.

    Returns: (slug, label)
      slug  — one of: contract, employment, nda, lease, invoice, resume, other
      label — specific name like "Job Offer Letter", "Tax Invoice", "Resume", etc.
    """
    system_prompt = """Analyse the document and return a JSON object with exactly two fields:

"slug"  — classify into EXACTLY ONE of: contract, employment, nda, lease, invoice, resume, other
"label" — the specific document type as a short human-readable name (2-5 words)

Slug definitions:
- contract   → service agreements, vendor agreements, terms & conditions, MOU, partnership deeds
- employment → offer letters, employment agreements, appointment letters, HR documents
- nda        → non-disclosure agreements, confidentiality agreements
- lease      → rental agreements, property leases, tenancy agreements, leave and licence
- invoice    → billing documents, receipts, payment summaries, purchase orders
- resume     → CV, job profiles, candidate profiles
- other      → anything else (legal notices, affidavits, power of attorney, financial statements,
               medical reports, insurance policies, wills, loan agreements, etc.)

Label examples:
  contract   → "Service Agreement", "Vendor Contract", "MOU"
  employment → "Job Offer Letter", "Appointment Letter", "Employment Contract"
  nda        → "Non-Disclosure Agreement", "Mutual Confidentiality Agreement"
  lease      → "Residential Lease Agreement", "Commercial Lease Deed"
  invoice    → "Tax Invoice", "Proforma Invoice", "Purchase Order"
  resume     → "Curriculum Vitae", "Resume"
  other      → "Partnership Deed", "Power of Attorney", "Affidavit",
               "Insurance Policy", "Will and Testament", "Loan Agreement", etc.

Return ONLY this JSON — no explanation, no markdown:
{"slug": "<one of 7 slugs>", "label": "<specific document name>"}"""

    raw    = await run_llm(text[:4_000], system_prompt)
    parsed = extract_json_from_text(raw)
    slug   = (parsed.get("slug")  or "").lower().strip()
    label  = (parsed.get("label") or "").strip()

    if slug not in _RISK_PROFILES:
        for known in _RISK_PROFILES:
            if known in raw.lower():
                slug = known
                break
        else:
            slug = "other"

    if not label or label.lower() in ("other document", "unknown document", "unknown", "other"):
        _FALLBACK_LABELS = {
            "contract":   "Contract / Legal Agreement",
            "employment": "Employment Agreement",
            "nda":        "Non-Disclosure Agreement",
            "lease":      "Lease Agreement",
            "invoice":    "Invoice / Billing Document",
            "resume":     "Resume / CV",
            "other":      "General Document",
        }
        label = _FALLBACK_LABELS[slug]

    logger.info(f"[risk_detection] Document type: {slug} ('{label}')")
    return slug, label


# ===========================================================================
# MANAGER FUNCTION 3 — Deep Analysis
# Runs 3 specialized parallel passes on the document for comprehensive coverage.
#
# Pass 1 (Structural) — missing fields, placeholders, inconsistencies, vague entries
# Pass 2 (Legal)      — unfair clauses, liability, rights waived, one-sided terms
# Pass 3 (Financial)  — payment risks, hidden costs, penalties, financial exposure
#
# For known document types, also runs a Fixed Profile pass (curated checklist).
# All passes run in parallel via asyncio.gather.
# ===========================================================================

def _empty_result() -> dict:
    return {"detected_risks": [], "missing_fields": []}


async def _pass_structural(chunk: str, doc_label: str, label: str) -> dict:
    """Pass 1: Completeness and structural integrity."""
    system_prompt = f"""You are a document quality analyst reviewing a "{doc_label}".

FOCUS: Find everything that is missing, incomplete, or uses placeholder text.

Check for ALL of the following:

1. UNFILLED PLACEHOLDERS
   Text like [Name], [Date], [Company], [Email], [Phone], [Address], [Amount], [Value],
   [X%], [TBD], [Insert here] — these are unfilled templates, NOT real data.
   List EACH placeholder as a separate missing_fields entry.

2. MISSING REQUIRED INFORMATION
   Fields or sections this type of document must have but lacks entirely.

3. INTERNAL INCONSISTENCIES
   Dates that contradict each other, amounts that don't add up, names that change
   across sections, references to clauses or exhibits that don't exist.

4. VAGUE OR UNDEFINED ENTRIES
   Values like "TBD", "To be agreed", "As discussed", "Reasonable", "Mutually agreed"
   that leave critical terms undefined.

5. MISSING PARTIES OR IDENTIFICATION
   Required names, titles, company names, or registration numbers not provided.

6. MISSING SIGNATURES OR AUTHORIZATION
   Signature blocks, witness, notarization, or approval fields absent.

7. DATE AND TIMELINE GAPS
   Effective date, expiry, notice periods, or milestone deadlines absent.

OUTPUT FORMAT — return ONLY valid JSON:
{{
  "detected_risks": [
    {{
      "risk_name": "<specific risk name>",
      "severity": "High | Medium | Low",
      "severity_reason": "<one sentence explaining severity>",
      "clause_found": "<exact text found, or 'Not found'>",
      "impact": "<why this is a problem>",
      "mitigation": "<how to fix>"
    }}
  ],
  "missing_fields": [
    {{
      "field_name": "<missing field or exact placeholder text>",
      "importance": "Critical | Important | Optional",
      "reason": "<why this field is needed>"
    }}
  ]
}}

Rules:
- List every [placeholder] as its own missing_fields entry with the placeholder text as field_name
- Be exhaustive — if something looks incomplete or vague, flag it
- Return ONLY the JSON — no markdown, no explanation"""

    logger.info(f"[risk_detection] Structural pass — {label}")
    try:
        raw    = await run_llm(chunk, system_prompt)
        result = extract_json_from_text(raw)
        return result if result else _empty_result()
    except Exception as e:
        logger.warning(f"[risk_detection] Structural pass failed ({label}): {e}")
        return _empty_result()


async def _pass_legal(chunk: str, doc_label: str, label: str) -> dict:
    """Pass 2: Legal and rights analysis."""
    system_prompt = f"""You are a senior legal counsel reviewing a "{doc_label}" to protect your client.

FOCUS: Find every legal risk, unfair clause, and dangerous term.

Read adversarially. Check for ALL of the following:

1. LIABILITY & INDEMNIFICATION
   Unlimited liability, broad indemnity, one-sided risk allocation.

2. RIGHTS WAIVED OR RESTRICTED
   Right to sue waived, right to work elsewhere restricted, IP rights transferred,
   privacy rights surrendered, right to appeal removed.

3. UNFAIR OR ONE-SIDED TERMS
   Obligations or penalties that apply to only one party, asymmetric rights.

4. TERMINATION RISKS
   Termination without cause or notice, excessive exit penalties, lock-in periods.

5. NON-COMPETE & NON-SOLICITATION
   Duration too long, scope too broad, geography unreasonably wide.

6. IP AND OWNERSHIP ASSIGNMENT
   Broad assignment of IP or work product, includes personal projects.

7. CONFIDENTIALITY OBLIGATIONS
   Overly broad definition, perpetual duration, no carve-outs.

8. DISPUTE RESOLUTION
   Mandatory arbitration, unfavorable jurisdiction, class-action waiver.

9. UNILATERAL CHANGE RIGHTS
   One party can modify terms without the other's consent.

10. ASSIGNMENT WITHOUT CONSENT
    Contract can be transferred to third parties without approval.

11. FORCE MAJEURE GAPS
    Events excusing performance poorly defined or too broad.

12. PENALTY AND LIQUIDATED DAMAGES
    Pre-set penalties that are disproportionate or punitive.

13. WARRANTIES DISCLAIMED
    All warranties disclaimed, no recourse for defective performance.

14. GOVERNING LAW
    Jurisdiction is inconvenient or legally unfavorable.

OUTPUT FORMAT — return ONLY valid JSON:
{{
  "detected_risks": [
    {{
      "risk_name": "<specific risk name>",
      "severity": "High | Medium | Low",
      "severity_reason": "<one sentence explaining severity>",
      "clause_found": "<exact quote from document, or 'Not found'>",
      "impact": "<how this harms the affected party>",
      "mitigation": "<how to negotiate or fix>"
    }}
  ],
  "missing_fields": []
}}

Rules:
- Quote the actual clause text wherever possible
- Only flag risks genuinely present — not hypothetical
- High = significant financial or legal harm; Medium = unfair but manageable; Low = minor concern
- Return ONLY the JSON — no markdown, no explanation"""

    logger.info(f"[risk_detection] Legal pass — {label}")
    try:
        raw    = await run_llm(chunk, system_prompt)
        result = extract_json_from_text(raw)
        return result if result else _empty_result()
    except Exception as e:
        logger.warning(f"[risk_detection] Legal pass failed ({label}): {e}")
        return _empty_result()


async def _pass_financial(chunk: str, doc_label: str, label: str) -> dict:
    """Pass 3: Financial and obligations analysis."""
    system_prompt = f"""You are a financial risk analyst reviewing a "{doc_label}".

FOCUS: Find every financial risk, hidden cost, and unclear monetary obligation.

Check for ALL of the following:

1. PAYMENT TERMS
   Vague due dates, unclear amounts, missing payment schedules.

2. PENALTIES AND LATE FEES
   Excessive late payment penalties, above-market interest rates.

3. FINANCIAL CAPS AND EXPOSURE
   No liability caps, unlimited financial exposure, uncapped indemnification.

4. AUTO-RENEWAL AND LOCK-IN COSTS
   Automatic renewals creating unexpected ongoing costs, minimum commitments.

5. PRICE ESCALATION
   Automatic price increases without sufficient notice or a cap.

6. HIDDEN OR UNCLEAR FEES
   Charges not itemized, costs buried in definitions, unexpected fees.

7. CURRENCY AND TAX
   Currency not specified, tax obligations unclear or unassigned.

8. REFUND AND CANCELLATION
   Unclear refund policies, excessive cancellation or termination fees.

9. PERFORMANCE PENALTIES
   Vague metrics that trigger financial penalties.

10. COMPENSATION AND EQUITY
    Vesting cliffs, clawback provisions, bonus at employer discretion, equity dilution.

11. SECURITY DEPOSITS AND RETAINERS
    Hard to recover, broad forfeiture triggers, no return timeline.

12. COST ALLOCATION
    Maintenance, insurance, utilities, or tax obligations not clearly assigned.

OUTPUT FORMAT — return ONLY valid JSON:
{{
  "detected_risks": [
    {{
      "risk_name": "<specific risk name>",
      "severity": "High | Medium | Low",
      "severity_reason": "<one sentence explaining severity>",
      "clause_found": "<exact quote or description, or 'Not found'>",
      "impact": "<financial or operational impact>",
      "mitigation": "<how to address or negotiate>"
    }}
  ],
  "missing_fields": []
}}

Rules:
- Evaluate monetary amounts for fairness and clarity where present
- Only flag risks actually present — not generic warnings
- Return ONLY the JSON — no markdown, no explanation"""

    logger.info(f"[risk_detection] Financial pass — {label}")
    try:
        raw    = await run_llm(chunk, system_prompt)
        result = extract_json_from_text(raw)
        return result if result else _empty_result()
    except Exception as e:
        logger.warning(f"[risk_detection] Financial pass failed ({label}): {e}")
        return _empty_result()


async def _pass_fixed_profile(chunk: str, doc_type: str, doc_label: str, label: str) -> dict:
    """Fixed-profile pass: checks the curated risk checklist for known document types."""
    profile     = _RISK_PROFILES[doc_type]
    risks_list  = "\n".join(f"    - {r}" for r in profile["risks"])
    fields_list = "\n".join(f"    - {f}" for f in profile["required_fields"])

    system_prompt = f"""You are a legal and financial risk analyst specializing in {doc_label} documents.

TASK 1 — RISK DETECTION:
Scan the document for the following risk categories and flag any that are present:
{risks_list}

TASK 2 — MISSING REQUIRED FIELDS:
Check if the following required fields are present with REAL values (not placeholders):
{fields_list}

OUTPUT FORMAT — return ONLY valid JSON:
{{
  "detected_risks": [
    {{
      "risk_name": "<risk category name>",
      "severity": "High | Medium | Low",
      "severity_reason": "<one sentence explaining severity>",
      "clause_found": "<exact quote or description, or 'Not found'>",
      "impact": "<why this is dangerous>",
      "mitigation": "<how to fix or negotiate>"
    }}
  ],
  "missing_fields": [
    {{
      "field_name": "<field that is missing>",
      "importance": "Critical | Important | Optional",
      "reason": "<why this field matters>"
    }}
  ]
}}

Rules:
- Only include risks actually present in the document
- A field with only placeholder text like [value] is missing — real content required
- Justify every severity rating in severity_reason
- Return ONLY the JSON — no markdown, no explanation"""

    logger.info(f"[risk_detection] Fixed profile ({doc_type}) — {label}")
    try:
        raw    = await run_llm(chunk, system_prompt)
        result = extract_json_from_text(raw)
        return result if result else _empty_result()
    except Exception as e:
        logger.warning(f"[risk_detection] Fixed profile failed ({label}): {e}")
        return _empty_result()


async def manage_deep_analysis(text: str, doc_type: str, doc_label: str) -> tuple[list[dict], list[dict]]:
    """
    Runs all analysis passes across all chunks in parallel.

    For known document types: Fixed Profile + Structural + Legal + Financial
    For unknown document types: Structural + Legal + Financial

    All passes × all chunks run in a single asyncio.gather — maximum parallelism.
    Returns (merged_risks, merged_fields) — raw, not yet consolidated.
    """
    chunks = _chunk_text(text)
    tasks  = []

    for i, chunk in enumerate(chunks):
        cl = f"chunk {i+1}/{len(chunks)}"
        # Run 3 deep passes on every chunk regardless of document type
        tasks.append(_pass_structural(chunk, doc_label, cl))
        tasks.append(_pass_legal(chunk, doc_label, cl))
        tasks.append(_pass_financial(chunk, doc_label, cl))
        # Add fixed-profile pass for known types
        if doc_type in _KNOWN_TYPES:
            tasks.append(_pass_fixed_profile(chunk, doc_type, doc_label, cl))

    logger.info(
        f"[risk_detection] Running {len(tasks)} analysis tasks in parallel "
        f"({'fixed+3 passes' if doc_type in _KNOWN_TYPES else '3 passes'} × {len(chunks)} chunk(s))"
    )

    results = await asyncio.gather(*tasks)

    all_risk_lists  = [r.get("detected_risks", []) for r in results]
    all_field_lists = [r.get("missing_fields",  []) for r in results]

    return _merge_risks(all_risk_lists), _merge_missing_fields(all_field_lists)


# ===========================================================================
# MANAGER FUNCTION 4 — Post-Processing
# Consolidates missing fields against the full document and synthesizes
# the overall assessment.
# ===========================================================================

async def manage_consolidation(
    full_text: str,
    candidate_fields: list[dict],
    doc_label: str,
) -> list[dict]:
    """
    Verifies each candidate missing field against the FULL document.
    Removes false positives: fields that appear in one chunk but are present elsewhere.
    Keeps fields that are completely absent OR only filled with placeholder text.
    Falls back to original list on any failure.
    """
    if not candidate_fields:
        return []

    field_names = [f.get("field_name", "") for f in candidate_fields if isinstance(f, dict)]
    if not field_names:
        return candidate_fields

    fields_list = "\n".join(f"  - {name}" for name in field_names)

    system_prompt = f"""You are a document reviewer checking whether specific fields are present in a "{doc_label}".

For each field listed, determine if it is present with a REAL value.

Fields to check:
{fields_list}

Return ONLY valid JSON:
{{
  "present": ["<field name>", ...],
  "absent":  ["<field name>", ...]
}}

Rules:
- "present" = field contains a real, actual value
- "absent"  = field is completely missing OR only contains placeholder text like [Email], [Value], [TBD]
- A placeholder is NOT a real value
- Return field names exactly as they appear in the list above
- Return ONLY the JSON — no explanation, no markdown"""

    try:
        raw    = await run_llm(full_text[:20_000], system_prompt)
        parsed = extract_json_from_text(raw)
        absent_raw = parsed.get("absent") or []
        if not absent_raw and not parsed.get("present"):
            raise ValueError("empty response")

        absent_keys = {_field_key(a) for a in absent_raw if isinstance(a, str)}
        result = [
            f for f in candidate_fields
            if isinstance(f, dict) and _field_key(f.get("field_name", "")) in absent_keys
        ]
        logger.info(
            f"[risk_detection] Field consolidation: "
            f"{len(candidate_fields)} candidates → {len(result)} confirmed absent"
        )
        return result if result else candidate_fields
    except Exception as e:
        logger.warning(f"[risk_detection] Consolidation failed — keeping original: {e}")
        return candidate_fields


async def manage_synthesis(
    merged_risks: list[dict],
    missing_fields: list[dict],
    doc_label: str,
    full_text: str,
) -> str:
    """
    Produces the final overall_assessment by synthesizing all merged findings
    and the full document in a single LLM pass.
    """
    if not merged_risks and not missing_fields:
        return f"No significant risks or missing fields were identified in this {doc_label}."

    high_count   = sum(1 for r in merged_risks if r.get("severity") == "High")
    medium_count = sum(1 for r in merged_risks if r.get("severity") == "Medium")

    risks_summary  = "\n".join(
        f"  - [{r.get('severity','?')}] {r.get('risk_name','?')}: {r.get('clause_found','')[:100]}"
        for r in merged_risks[:20]
    )
    fields_summary = "\n".join(
        f"  - [{f.get('importance','?')}] {f.get('field_name','?')}"
        for f in missing_fields[:15]
    )

    system_prompt = f"""You are a senior risk analyst writing an executive summary.

Full risk analysis of a "{doc_label}" document:

Detected Risks ({len(merged_risks)} total — {high_count} High, {medium_count} Medium):
{risks_summary or "  None"}

Missing Required Fields ({len(missing_fields)} total):
{fields_summary or "  None"}

Write a concise 3-4 sentence executive summary covering:
1. Overall risk level (High / Medium / Low) and why
2. The most critical risks or red flags
3. Most important missing fields and their impact
4. A clear recommendation (e.g., do not sign, fill placeholders first, negotiate specific clauses)

Return ONLY the summary text — no JSON, no bullet points, no headers."""

    try:
        raw = await run_llm(full_text[:5_000], system_prompt)
        return raw.strip()
    except Exception as e:
        logger.warning(f"[risk_detection] Synthesis failed: {e}")
        return ""


# ===========================================================================
# MANAGER FUNCTION 5 — Placeholder Detection
# Programmatic scan (LLM-independent) for [bracketed] template values.
# ===========================================================================

def manage_placeholder_detection(
    text: str,
    analysis: dict,
    doc_label: str,
) -> dict:
    """
    Scans the raw document text for unfilled [placeholder] patterns.
    This is programmatic and always reliable — catches what the LLM misses
    when the document structure looks syntactically complete.

    Merges found placeholders into the analysis dict in-place.
    Returns the updated analysis dict.
    """
    matches = _PLACEHOLDER_RE.findall(text)
    if not matches:
        return analysis

    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        if _NUMERIC_RE.match(m):
            continue
        key = re.sub(r'\s+', ' ', m.strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(m.strip())

    if not unique:
        return analysis

    logger.info(f"[risk_detection] {len(unique)} unfilled placeholder(s) found in '{doc_label}'")

    # Build risk entry
    examples  = ", ".join(f"[{p}]" for p in unique[:4])
    ph_risk   = {
        "risk_name": "Unfilled Template Placeholders",
        "severity": "High",
        "severity_reason": (
            f"{len(unique)} template placeholder(s) were not replaced with real values, "
            "making the document incomplete and potentially invalid."
        ),
        "clause_found": examples,
        "impact": (
            "The document is incomplete — contact details, dates, names, and other "
            "critical fields still contain template text instead of real data."
        ),
        "mitigation": "Replace every [bracketed placeholder] with the correct real-world value before using this document.",
    }

    # Build missing field entries
    ph_fields = [
        {
            "field_name": ph,
            "importance": "Critical",
            "reason": f"Template placeholder '[{ph}]' has not been filled in.",
        }
        for ph in unique
    ]

    # Merge — avoid duplicates
    existing_risk_keys  = {_risk_key(r.get("risk_name", "")) for r in analysis["detected_risks"]}
    existing_field_keys = {_field_key(f.get("field_name", "")) for f in analysis["missing_fields"]}

    if _risk_key("Unfilled Template Placeholders") not in existing_risk_keys:
        analysis["detected_risks"] = [ph_risk] + analysis["detected_risks"]

    new_fields = [
        f for f in ph_fields
        if _field_key(f.get("field_name", "")) not in existing_field_keys
    ]
    analysis["missing_fields"] = new_fields + analysis["missing_fields"]

    return analysis


# ===========================================================================
# MANAGER FUNCTION 6 — Response Building
# Formats the final normalized analysis into the API response structure.
# ===========================================================================

def manage_response_building(analysis: dict, doc_label: str) -> dict:
    """
    Wraps the normalized analysis into the standard API response envelope.
    """
    detected_count = len(analysis.get("detected_risks", []))
    missing_count  = len(analysis.get("missing_fields", []))

    return {
        "status":        "success",
        "analysis_type": "risk_detection",
        "document_type": doc_label,
        "risk_count": {
            "detected_risks": detected_count,
            "missing_fields": missing_count,
            "total":          detected_count + missing_count,
        },
        "data": analysis,
    }


# ===========================================================================
# Shared utilities
# ===========================================================================

def _field_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _risk_key(risk_name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", risk_name.lower())


def _chunk_text(text: str) -> list[str]:
    if len(text) <= _CHUNK_SIZE:
        return [text]
    chunks = []
    start  = 0
    while start < len(text):
        end = min(start + _CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    logger.info(f"[risk_detection] Document split into {len(chunks)} chunk(s)")
    return chunks


def _merge_risks(lists: list[list]) -> list[dict]:
    """
    Merges detected_risks from multiple sources. Deduplicates by risk_name,
    keeps highest severity and best clause excerpt. Sorted High → Medium → Low.
    """
    seen: dict[str, dict] = {}
    for risk_list in lists:
        if not isinstance(risk_list, list):
            continue
        for risk in risk_list:
            if not isinstance(risk, dict):
                continue
            key = _risk_key(risk.get("risk_name", ""))
            if not key:
                continue
            if key not in seen:
                seen[key] = dict(risk)
            else:
                existing_rank = _SEVERITY_RANK.get(seen[key].get("severity", "Medium"), 2)
                incoming_rank = _SEVERITY_RANK.get(risk.get("severity", "Medium"), 2)
                if incoming_rank > existing_rank:
                    seen[key] = dict(risk)
                if seen[key].get("clause_found", "Not found") in ("Not found", "") and \
                   risk.get("clause_found", "Not found") not in ("Not found", ""):
                    seen[key]["clause_found"] = risk["clause_found"]
    return sorted(
        seen.values(),
        key=lambda r: _SEVERITY_RANK.get(r.get("severity", "Medium"), 2),
        reverse=True,
    )


def _merge_missing_fields(lists: list[list]) -> list[dict]:
    """
    Merges missing_fields from multiple sources. Deduplicates by field_name,
    keeps highest importance. Sorted Critical → Important → Optional.
    """
    seen: dict[str, dict] = {}
    for field_list in lists:
        if not isinstance(field_list, list):
            continue
        for field in field_list:
            if not isinstance(field, dict):
                continue
            key = _field_key(field.get("field_name", ""))
            if not key:
                continue
            if key not in seen:
                seen[key] = dict(field)
            else:
                existing_rank = _IMPORTANCE_RANK.get(seen[key].get("importance", "Important"), 2)
                incoming_rank = _IMPORTANCE_RANK.get(field.get("importance", "Important"), 2)
                if incoming_rank > existing_rank:
                    seen[key] = dict(field)
    return sorted(
        seen.values(),
        key=lambda f: _IMPORTANCE_RANK.get(f.get("importance", "Important"), 2),
        reverse=True,
    )


def _normalize_result(result: dict) -> dict:
    """Enforces exact fixed field names on the analysis dict."""
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
            "risk_name":       str(item.get("risk_name")       or item.get("name")           or item.get("risk")            or "Unknown Risk"),
            "severity":        severity,
            "severity_reason": str(item.get("severity_reason") or item.get("severity_justification") or item.get("reason_for_severity") or ""),
            "clause_found":    str(item.get("clause_found")    or item.get("clause")          or item.get("quote")           or "Not found"),
            "impact":          str(item.get("impact")          or item.get("danger")          or item.get("effect")          or ""),
            "mitigation":      str(item.get("mitigation")      or item.get("fix")             or item.get("recommendation")  or ""),
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


# ===========================================================================
# PUBLIC ENTRY POINT
# Orchestrates all manager functions in sequence.
# ===========================================================================

async def analyze_document_risks(text: str) -> dict:
    """
    Orchestrates the full risk analysis pipeline via 6 manager functions:

    1. manage_preprocessing      — sanitize text (fixes 400 Bad Request from PDF chars)
    2. manage_type_detection     — classify document type
    3. manage_deep_analysis      — run all analysis passes in parallel
                                   (Structural + Legal + Financial + Fixed Profile)
    4. manage_consolidation      — verify missing fields against full document
    5. manage_synthesis          — generate executive summary
    6. manage_placeholder_detection — programmatic [placeholder] scan
    7. manage_response_building  — format final API response
    """

    # Step 1 — Sanitize (fixes 400 Bad Request from null bytes / control chars)
    text = manage_preprocessing(text)

    # Step 2 — Detect document type
    doc_type, doc_label = await manage_type_detection(text)

    # Step 3 — Run all analysis passes in parallel
    raw_risks, raw_fields = await manage_deep_analysis(text, doc_type, doc_label)

    # Step 4 — Consolidate: verify missing fields against full document
    confirmed_fields = await manage_consolidation(text, raw_fields, doc_label)

    # Step 5 — Synthesize overall assessment
    overall = await manage_synthesis(raw_risks, confirmed_fields, doc_label, text)

    # Assemble normalized analysis dict
    analysis = _normalize_result({
        "document_type":      doc_type,
        "document_label":     doc_label,
        "detected_risks":     raw_risks,
        "missing_fields":     confirmed_fields,
        "overall_assessment": overall,
    })

    # Step 6 — Placeholder detection (programmatic, always reliable)
    analysis = manage_placeholder_detection(text, analysis, doc_label)

    # Re-synthesize if placeholders were found (update the executive summary)
    if any(
        _risk_key(r.get("risk_name", "")) == _risk_key("Unfilled Template Placeholders")
        for r in analysis["detected_risks"]
    ):
        analysis["overall_assessment"] = await manage_synthesis(
            analysis["detected_risks"],
            analysis["missing_fields"],
            doc_label,
            text,
        )

    # Step 7 — Build response
    return manage_response_building(analysis, doc_label)
