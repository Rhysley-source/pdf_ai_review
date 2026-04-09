import asyncio
import logging
import re
from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document-type risk profiles (used by the fixed-profile pass for known types)
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
            "Unilateral amendment: one party can change terms without the other's consent",
            "Intellectual property assignment: broad IP ownership transfer including prior works",
            "Force majeure gaps: events that excuse performance not clearly defined",
            "Dispute resolution: mandatory arbitration removing right to sue in court",
            "Payment terms: vague or one-sided payment obligations",
            "Confidentiality obligations: overly broad or indefinite duration",
            "Assignment clause: contract can be transferred to unknown third party",
            "Warranty disclaimers: all warranties disclaimed leaving no recourse",
            "Liquidated damages: pre-set penalties that may be disproportionate",
        ],
        "required_fields": [
            "Effective Date",
            "Contract Term / Duration",
            "Termination Notice Period",
            "Governing Law / Jurisdiction",
            "Payment Terms and Schedule",
            "Renewal Conditions",
            "Parties' Full Legal Names and Addresses",
            "Signatures of All Parties",
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
            "Probation period risks: reduced protections or easier termination during probation",
            "Vague performance metrics: subjective criteria that could justify termination",
            "Relocation clauses: forced relocation without sufficient compensation",
            "Salary or equity not clearly defined or subject to change",
            "Excessive working hours or on-call requirements not compensated",
            "Garden leave: paid but barred from working during notice period",
            "Moonlighting restriction: barred from any outside work or freelancing",
        ],
        "required_fields": [
            "Start Date",
            "Salary / Base Compensation",
            "Notice Period (both parties)",
            "Equity / Bonus Terms and Vesting Schedule",
            "Job Title and Reporting Structure",
            "Governing Law / Jurisdiction",
            "Probation Period Terms",
            "Benefits Summary (health, PTO, etc.)",
        ],
    },
    "nda": {
        "label": "Non-Disclosure Agreement",
        "risks": [
            "Overly broad definition of confidential information covering everything",
            "No expiry on confidentiality obligations: perpetual / lasts forever",
            "One-sided obligations: only one party is bound",
            "No carve-outs for publicly available or independently developed information",
            "Excessive remedies: injunctions and unlimited damages for any breach",
            "Residuals clause: allows use of retained memory of confidential info",
            "No clear permitted disclosure exceptions (legal compulsion, regulatory)",
            "Missing return or destruction of information clause",
            "No definition of who is a permitted recipient within each party",
            "No limitation on the purpose for which confidential info can be used",
        ],
        "required_fields": [
            "Effective Date",
            "Duration of Confidentiality Obligations",
            "Governing Law / Jurisdiction",
            "Definition of Confidential Information",
            "Permitted Disclosures / Exceptions",
            "Parties' Full Legal Names",
            "Signatures",
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
            "Maintenance and repair responsibility shifted entirely to tenant",
            "No clear dispute resolution or escalation process",
            "Renewal at landlord's sole discretion with no tenant rights",
            "Holding over penalty: excessive charges if tenant stays past end date",
            "Utilities responsibility not clearly assigned",
        ],
        "required_fields": [
            "Lease Start Date",
            "Lease End Date",
            "Monthly Rent Amount",
            "Security Deposit Amount and Return Conditions",
            "Notice Period to Vacate",
            "Renewal Terms",
            "Landlord's Full Name and Contact Information",
            "Property Address",
            "Permitted Use of Property",
        ],
    },
    "invoice": {
        "label": "Invoice / Billing Document",
        "risks": [
            "Missing payment due date: no clear deadline for payment",
            "No late payment penalty terms defined",
            "Vague or incomplete description of goods or services delivered",
            "Missing tax breakdown (VAT, GST, sales tax, etc.)",
            "No dispute resolution window or process for billing errors",
            "Currency not specified for international transactions",
            "No PO number or reference number for tracking and reconciliation",
            "Missing remittance instructions: how and where to pay",
            "No itemized breakdown for multi-line invoices",
        ],
        "required_fields": [
            "Invoice Number",
            "Invoice Date",
            "Payment Due Date",
            "Vendor / Supplier Name and Address",
            "Client / Buyer Name and Address",
            "Total Amount Due",
            "Tax Amount and Rate",
            "Payment Method / Bank Details / Account Information",
            "Description of Goods or Services",
        ],
    },
    "resume": {
        "label": "Resume / CV",
        "risks": [
            "Unfilled template placeholders: [brackets] used instead of real data",
            "Employment gap: unexplained periods of inactivity between jobs",
            "Missing contact information: no email, phone number, or location",
            "No quantified achievements: only job duties listed, no metrics or outcomes",
            "Vague or generic job titles that don't reflect actual role or seniority",
            "Missing or incomplete education details: degree, institution, or graduation year absent",
            "Skills section missing, outdated, or does not match the stated experience",
            "Inconsistent date formats or reverse-chronological order not followed",
            "No professional summary or objective statement",
            "LinkedIn, GitHub, or portfolio links missing for technical or creative roles",
            "Certifications section absent for a technical or regulated profession",
        ],
        "required_fields": [
            "Full Name",
            "Email Address",
            "Phone Number",
            "Location (City / Country)",
            "Professional Summary or Objective Statement",
            "Work Experience with Company Names, Titles, and Dates",
            "Education with Institution Name and Graduation Year",
            "Skills Section",
        ],
    },
    "other": {
        "label": "General Document",
        "risks": [
            "Ambiguous obligations: unclear who is responsible for what action",
            "Missing dates or validity period",
            "No signatures or authorization section",
            "Undefined terms or jargon without explanation",
            "Inconsistent data or figures within the document",
            "Missing governing law or jurisdiction",
            "No dispute resolution clause",
            "Unfilled template placeholders: [brackets] used instead of real data",
            "Missing party identification: who the document applies to is unclear",
        ],
        "required_fields": [
            "Document Date",
            "Parties Involved",
            "Purpose / Subject Matter",
            "Signatures or Authorization",
        ],
    },
}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_CHUNK_SIZE    = 20_000
_CHUNK_OVERLAP = 1_000


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


# ---------------------------------------------------------------------------
# Placeholder detector (programmatic, LLM-independent)
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r'\[([^\]\[]{2,60})\]')
_NUMERIC_RE     = re.compile(r'^\s*\d+\s*$')


def _detect_placeholders(text: str, doc_label: str) -> tuple[list[dict], list[dict]]:
    """
    Returns (risk_entries, missing_field_entries) for any [bracketed placeholder]
    patterns found in the document text. Excludes pure numeric citations like [1].
    """
    matches = _PLACEHOLDER_RE.findall(text)
    if not matches:
        return [], []

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
        return [], []

    logger.info(f"[risk_detection] {len(unique)} unfilled placeholder(s) found in '{doc_label}'")

    examples = ", ".join(f"[{p}]" for p in unique[:4])
    risk = {
        "risk_name": "Unfilled Template Placeholders",
        "severity": "High",
        "severity_reason": (
            f"{len(unique)} template placeholder(s) were not replaced with actual values, "
            "making the document incomplete and potentially invalid."
        ),
        "clause_found": examples,
        "impact": (
            "The document is incomplete — placeholder values such as contact details, "
            "dates, and names still show template text instead of real data."
        ),
        "mitigation": "Replace every [bracketed placeholder] with the correct real-world value before using this document.",
    }
    fields = [
        {
            "field_name": ph,
            "importance": "Critical",
            "reason": f"Template placeholder '[{ph}]' has not been filled in with an actual value.",
        }
        for ph in unique
    ]
    return [risk], fields


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


# ---------------------------------------------------------------------------
# Deduplication and merge helpers
# ---------------------------------------------------------------------------

def _field_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _risk_key(risk_name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", risk_name.lower())


_SEVERITY_RANK   = {"High": 3, "Medium": 2, "Low": 1}
_IMPORTANCE_RANK = {"Critical": 3, "Important": 2, "Optional": 1}


def _merge_risks(lists: list[list]) -> list[dict]:
    """
    Merges multiple detected_risks lists. Deduplicates by normalised risk_name,
    keeping highest severity and best clause excerpt. Sorted High → Medium → Low.
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
    Merges multiple missing_fields lists. Deduplicates by field_name,
    keeping highest importance. Sorted Critical → Important → Optional.
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


# ---------------------------------------------------------------------------
# Consolidation helpers
# ---------------------------------------------------------------------------

async def _consolidate_missing_fields(
    full_text: str,
    candidate_fields: list[dict],
    doc_label: str,
) -> list[dict]:
    """
    Verifies each candidate missing field against the FULL document.
    Treats placeholder-filled fields as still absent.
    Falls back to original list on failure.
    """
    if not candidate_fields:
        return []

    field_names = [f.get("field_name", "") for f in candidate_fields if isinstance(f, dict)]
    if not field_names:
        return candidate_fields

    fields_list = "\n".join(f"  - {name}" for name in field_names)

    system_prompt = f"""You are a document reviewer checking whether specific fields are present in a "{doc_label}" document.

For each field listed below, determine if it is present with a REAL value.

Fields to check:
{fields_list}

Return ONLY a valid JSON object:
{{
  "present":  ["<field name>", ...],
  "absent":   ["<field name>", ...]
}}

Rules:
- "present" = the field contains a real, actual value
- "absent"  = the field is completely missing OR only contains placeholder text like [Email], [Value], [TBD]
- A placeholder is NOT a real value — fields with only [bracket] text are absent
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
        logger.warning(f"[risk_detection] Field consolidation failed — keeping original: {e}")
        return candidate_fields


async def _synthesize_overall_assessment(
    merged_risks: list[dict],
    missing_fields: list[dict],
    doc_label: str,
    full_text: str,
) -> str:
    """Single LLM synthesis pass over the complete merged result."""
    if not merged_risks and not missing_fields:
        return f"No significant risks or missing fields were identified in this {doc_label}."

    high_count   = sum(1 for r in merged_risks if r.get("severity") == "High")
    medium_count = sum(1 for r in merged_risks if r.get("severity") == "Medium")

    risks_summary = "\n".join(
        f"  - [{r.get('severity','?')}] {r.get('risk_name','?')}: {r.get('clause_found','')[:100]}"
        for r in merged_risks[:20]
    )
    fields_summary = "\n".join(
        f"  - [{f.get('importance','?')}] {f.get('field_name','?')}"
        for f in missing_fields[:15]
    )

    system_prompt = f"""You are a senior risk analyst writing an executive summary.

You have completed a full risk analysis of a "{doc_label}" document.

Detected Risks ({len(merged_risks)} total — {high_count} High, {medium_count} Medium):
{risks_summary or "  None"}

Missing Required Fields ({len(missing_fields)} total):
{fields_summary or "  None"}

Write a concise 3-4 sentence executive summary covering:
1. Overall risk level (High / Medium / Low) and the primary reason
2. The most critical risks or red flags found
3. The most important missing fields and their practical impact
4. A clear recommendation (e.g., do not sign, fill placeholders first, negotiate specific clauses)

Return ONLY the summary text — no JSON, no bullet points, no headers."""

    try:
        raw = await run_llm(full_text[:5_000], system_prompt)
        return raw.strip()
    except Exception as e:
        logger.warning(f"[risk_detection] Synthesis failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Step 1 — Detect document type
# ---------------------------------------------------------------------------

async def _detect_document_type(text: str) -> tuple[str, str]:
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

Label rules:
- Be specific — never return "Other Document" or "Unknown Document"
- Use the actual document name as it would appear on the document itself

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
# Step 2 — Three focused analysis passes
#
# Each pass runs in parallel and looks at the document through a different lens:
#
#   Pass 1 — Structural & Completeness
#     What is missing, incomplete, placeholder-filled, or inconsistent?
#
#   Pass 2 — Legal & Rights
#     What unfair, one-sided, or dangerous legal clauses are present?
#
#   Pass 3 — Financial & Obligations
#     What financial exposure, hidden costs, or unclear obligations exist?
#
# Running three focused passes in parallel gives the LLM more mental space
# per dimension and catches risks a single all-purpose pass would miss.
# ---------------------------------------------------------------------------

def _empty_chunk_result() -> dict:
    return {"detected_risks": [], "missing_fields": []}


async def _analyze_chunk_structural(chunk: str, doc_label: str, chunk_label: str) -> dict:
    """
    Pass 1: Completeness & structural integrity.
    Focused on: missing fields, placeholders, inconsistencies, vague entries.
    """
    system_prompt = f"""You are a document quality analyst reviewing a "{doc_label}".

FOCUS: Find everything that is missing, incomplete, or uses placeholder text.

Examine for ALL of the following:

1. UNFILLED PLACEHOLDERS
   Any text like [Name], [Date], [Company], [Email], [Phone], [Address], [Amount],
   [Value], [X%], [TBD], [Insert here], [Specify] — these are templates, NOT real data.
   List EACH placeholder as a separate missing field.

2. MISSING REQUIRED INFORMATION
   Fields or sections that this type of document must have but are completely absent.

3. INTERNAL INCONSISTENCIES
   Dates that contradict each other, amounts that don't add up, names that change
   across sections, references to clauses or exhibits that don't exist.

4. VAGUE ENTRIES
   Values like "TBD", "To be agreed", "As discussed", "Reasonable", "Mutually agreed"
   that leave critical terms undefined.

5. MISSING PARTIES OR IDENTIFICATION
   Required names, titles, company names, or registration numbers not provided.

6. MISSING SIGNATURES OR AUTHORIZATION
   Document lacks signature blocks, witness, notarization, or approval fields.

7. DATE AND TIMELINE GAPS
   Effective date, expiry, notice periods, or deadline milestones are absent.

OUTPUT FORMAT — return ONLY valid JSON:
{{
  "detected_risks": [
    {{
      "risk_name": "<specific risk name>",
      "severity": "High | Medium | Low",
      "severity_reason": "<one sentence explaining severity>",
      "clause_found": "<exact text or phrase found, or 'Not found'>",
      "impact": "<why this is a problem>",
      "mitigation": "<how to fix>"
    }}
  ],
  "missing_fields": [
    {{
      "field_name": "<name of the missing field or placeholder text>",
      "importance": "Critical | Important | Optional",
      "reason": "<why this field is needed>"
    }}
  ]
}}

Rules:
- List every [placeholder] as its own missing_fields entry using the exact placeholder text as field_name
- Be exhaustive — if something looks incomplete or vague, flag it
- Return ONLY the JSON — no markdown, no explanation"""

    logger.info(f"[risk_detection] Structural pass — {chunk_label}")
    try:
        raw    = await run_llm(chunk, system_prompt)
        result = extract_json_from_text(raw)
        return result if result else _empty_chunk_result()
    except Exception as e:
        logger.warning(f"[risk_detection] Structural pass failed ({chunk_label}): {e}")
        return _empty_chunk_result()


async def _analyze_chunk_legal(chunk: str, doc_label: str, chunk_label: str) -> dict:
    """
    Pass 2: Legal & rights analysis.
    Focused on: unfair clauses, liability, rights waived, one-sided terms.
    """
    system_prompt = f"""You are a senior legal counsel reviewing a "{doc_label}" to protect your client.

FOCUS: Find every legal risk, unfair clause, and dangerous term in the document.

Read adversarially. Look for ALL of the following:

1. LIABILITY & INDEMNIFICATION
   Unlimited liability, broad indemnity obligations, one-sided risk transfer.

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
   Broad assignment of intellectual property, works created outside work scope included.

7. CONFIDENTIALITY OBLIGATIONS
   Overly broad definition, perpetual duration, no carve-outs for public information.

8. DISPUTE RESOLUTION
   Mandatory arbitration, unfavorable or distant jurisdiction, class-action waiver.

9. AMENDMENT AND UNILATERAL CHANGE
   One party can modify terms without the other party's consent.

10. ASSIGNMENT WITHOUT CONSENT
    Contract can be transferred to unknown third parties.

11. FORCE MAJEURE GAPS
    Events excusing performance not clearly defined, too broad or too narrow.

12. PENALTY AND LIQUIDATED DAMAGES
    Pre-set penalties that are disproportionate or punitive.

13. WARRANTIES AND DISCLAIMERS
    All warranties disclaimed, no recourse for defective performance.

14. GOVERNING LAW
    Jurisdiction forces disputes in a location inconvenient or unfavorable to one party.

OUTPUT FORMAT — return ONLY valid JSON:
{{
  "detected_risks": [
    {{
      "risk_name": "<specific risk name>",
      "severity": "High | Medium | Low",
      "severity_reason": "<one sentence explaining severity>",
      "clause_found": "<exact quote from the document, or 'Not found'>",
      "impact": "<how this harms the affected party>",
      "mitigation": "<how to negotiate or fix>"
    }}
  ],
  "missing_fields": []
}}

Rules:
- Quote the actual problematic clause text wherever possible
- Only flag risks genuinely present in the document — not hypothetical ones
- High = could cause significant financial or legal harm
- Medium = unfair but manageable with negotiation
- Low = minor concern worth noting
- Return ONLY the JSON — no markdown, no explanation"""

    logger.info(f"[risk_detection] Legal pass — {chunk_label}")
    try:
        raw    = await run_llm(chunk, system_prompt)
        result = extract_json_from_text(raw)
        return result if result else _empty_chunk_result()
    except Exception as e:
        logger.warning(f"[risk_detection] Legal pass failed ({chunk_label}): {e}")
        return _empty_chunk_result()


async def _analyze_chunk_financial(chunk: str, doc_label: str, chunk_label: str) -> dict:
    """
    Pass 3: Financial & obligations analysis.
    Focused on: monetary exposure, hidden costs, payment risks, unclear obligations.
    """
    system_prompt = f"""You are a financial risk analyst reviewing a "{doc_label}".

FOCUS: Find every financial risk, hidden cost, and unclear monetary obligation.

Examine for ALL of the following:

1. PAYMENT TERMS
   Vague due dates, unclear amounts, missing payment schedules, no milestone payments.

2. PENALTIES AND LATE FEES
   Excessive late payment penalties, interest rates above market rate.

3. FINANCIAL CAPS AND EXPOSURE
   Missing liability caps, unlimited financial exposure, uncapped indemnification.

4. AUTO-RENEWAL AND LOCK-IN COSTS
   Automatic renewals creating unexpected ongoing costs, minimum commitment periods.

5. PRICE ESCALATION
   Automatic price increases without sufficient notice or a cap on the increase.

6. HIDDEN OR UNCLEAR FEES
   Fees not itemized, charges buried in definitions, costs that may apply unexpectedly.

7. CURRENCY AND TAX TREATMENT
   Currency not specified for cross-border transactions, tax obligations unclear.

8. REFUND AND CANCELLATION COSTS
   Unclear or one-sided refund policies, excessive cancellation or early termination fees.

9. PERFORMANCE PENALTIES
   Vague performance metrics that trigger financial penalties, SLA penalties.

10. COMPENSATION AND EQUITY RISKS
    Vesting cliffs, clawback provisions, bonuses at employer discretion, equity dilution.

11. SECURITY DEPOSITS AND RETAINERS
    Security deposits with unclear return conditions, broad forfeiture triggers.

12. COST ALLOCATION
    Maintenance, insurance, utilities, or tax obligations not clearly assigned to a party.

13. FINANCIAL REPORTING OBLIGATIONS
    Audit rights, expense reporting, or disclosure requirements that create burden or risk.

OUTPUT FORMAT — return ONLY valid JSON:
{{
  "detected_risks": [
    {{
      "risk_name": "<specific risk name>",
      "severity": "High | Medium | Low",
      "severity_reason": "<one sentence explaining severity>",
      "clause_found": "<exact quote or description from the document, or 'Not found'>",
      "impact": "<the financial or operational impact>",
      "mitigation": "<how to address or negotiate>"
    }}
  ],
  "missing_fields": []
}}

Rules:
- If monetary amounts are present, evaluate whether they are fair, clear, and capped
- Only flag risks actually present — not generic warnings
- Return ONLY the JSON — no markdown, no explanation"""

    logger.info(f"[risk_detection] Financial pass — {chunk_label}")
    try:
        raw    = await run_llm(chunk, system_prompt)
        result = extract_json_from_text(raw)
        return result if result else _empty_chunk_result()
    except Exception as e:
        logger.warning(f"[risk_detection] Financial pass failed ({chunk_label}): {e}")
        return _empty_chunk_result()


async def _analyze_chunk_fixed(chunk: str, doc_type: str, doc_label: str,
                                risks_list: str, fields_list: str,
                                chunk_label: str) -> dict:
    """
    Fixed-profile pass for known document types.
    Checks the curated risk checklist and required fields specific to that type.
    """
    system_prompt = f"""You are a legal and financial risk analyst specializing in {doc_label} documents.

TASK 1 — RISK DETECTION:
Scan the document for the following risk categories and flag any that are present:
{risks_list}

TASK 2 — MISSING REQUIRED FIELDS:
Check if the following required fields are present in the document with REAL values.
Flag any that are missing or only contain placeholder text like [value]:
{fields_list}

OUTPUT FORMAT — return ONLY valid JSON in exactly this structure:
{{
  "document_type": "{doc_type}",
  "document_label": "{doc_label}",
  "detected_risks": [
    {{
      "risk_name": "<risk category name>",
      "severity": "High | Medium | Low",
      "severity_reason": "<one sentence explaining why this severity level was assigned>",
      "clause_found": "<exact quote or short description of the clause, or 'Not found'>",
      "impact": "<why this is dangerous to the affected party>",
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
  "overall_assessment": ""
}}

Rules:
- Only include risks actually present in the document
- A field with only placeholder text like [value] is missing — real content is required
- Justify every severity rating in severity_reason
- If no risks found, return detected_risks as []
- If no fields missing, return missing_fields as []
- Return ONLY the JSON — no markdown, no explanation"""

    logger.info(f"[risk_detection] Fixed profile ({doc_type}) — {chunk_label}")
    try:
        raw    = await run_llm(chunk, system_prompt)
        result = extract_json_from_text(raw)
        return result if result else _empty_chunk_result()
    except Exception as e:
        logger.warning(f"[risk_detection] Fixed profile failed ({chunk_label}): {e}")
        return _empty_chunk_result()


# ---------------------------------------------------------------------------
# Step 3 — Run all passes across all chunks
#
# _analyze_all_dimensions   : runs the 3 focused passes across all chunks in parallel
# _analyze_all_fixed        : runs the fixed-profile pass across all chunks in parallel
# ---------------------------------------------------------------------------

async def _analyze_all_dimensions(text: str, doc_label: str) -> tuple[list[dict], list[dict]]:
    """
    Runs all 3 analysis passes (structural, legal, financial) across every chunk
    in a single asyncio.gather — all passes × all chunks run in parallel.
    Returns (merged_risks, merged_fields) — raw, not yet consolidated.
    """
    chunks = _chunk_text(text)

    # Build one flat list of tasks: 3 passes × N chunks
    tasks = []
    for i, chunk in enumerate(chunks):
        cl = f"chunk {i+1}/{len(chunks)}"
        tasks.append(_analyze_chunk_structural(chunk, doc_label, cl))
        tasks.append(_analyze_chunk_legal(chunk, doc_label, cl))
        tasks.append(_analyze_chunk_financial(chunk, doc_label, cl))

    results = await asyncio.gather(*tasks)

    all_risk_lists  = [r.get("detected_risks", []) for r in results]
    all_field_lists = [r.get("missing_fields",  []) for r in results]

    return _merge_risks(all_risk_lists), _merge_missing_fields(all_field_lists)


async def _analyze_all_fixed(text: str, doc_type: str, doc_label: str) -> tuple[list[dict], list[dict]]:
    """
    Runs the fixed-profile pass across all chunks in parallel.
    Returns (merged_risks, merged_fields) — raw, not yet consolidated.
    """
    profile     = _RISK_PROFILES[doc_type]
    risks_list  = "\n".join(f"    - {r}" for r in profile["risks"])
    fields_list = "\n".join(f"    - {f}" for f in profile["required_fields"])
    chunks      = _chunk_text(text)

    results = await asyncio.gather(*[
        _analyze_chunk_fixed(chunk, doc_type, doc_label, risks_list, fields_list,
                             f"chunk {i+1}/{len(chunks)}")
        for i, chunk in enumerate(chunks)
    ])

    return (
        _merge_risks([r.get("detected_risks", []) for r in results]),
        _merge_missing_fields([r.get("missing_fields", []) for r in results]),
    )


# ---------------------------------------------------------------------------
# Step 4 — Full analysis pipelines
#
# _analyze_risks_hybrid    : for known document types
#   → fixed profile + 3 focused passes run in parallel
#   → single consolidation + synthesis on the merged result
#
# _analyze_risks_dynamic   : for unknown document types
#   → 3 focused passes run in parallel
#   → single consolidation + synthesis
# ---------------------------------------------------------------------------

async def _analyze_risks_hybrid(text: str, doc_type: str, doc_label: str) -> dict:
    """
    Known document type: fixed-profile checklist + 3 deep-analysis passes.
    All run in parallel. Single consolidation + synthesis on the merged result.
    """
    logger.info(
        f"[risk_detection] Hybrid analysis — fixed profile + 3 passes in parallel for '{doc_label}'"
    )

    (fixed_risks, fixed_fields), (dim_risks, dim_fields) = await asyncio.gather(
        _analyze_all_fixed(text, doc_type, doc_label),
        _analyze_all_dimensions(text, doc_label),
    )

    merged_risks  = _merge_risks([fixed_risks, dim_risks])
    merged_fields = await _consolidate_missing_fields(
        text,
        _merge_missing_fields([fixed_fields, dim_fields]),
        doc_label,
    )
    overall = await _synthesize_overall_assessment(merged_risks, merged_fields, doc_label, text)

    return {
        "document_type":      doc_type,
        "document_label":     doc_label,
        "detected_risks":     merged_risks,
        "missing_fields":     merged_fields,
        "overall_assessment": overall,
    }


async def _analyze_risks_dynamic(text: str, doc_label: str) -> dict:
    """
    Unknown document type: 3 focused passes only.
    All run in parallel. Single consolidation + synthesis.
    """
    logger.info(f"[risk_detection] Dynamic analysis — 3 passes in parallel for '{doc_label}'")

    dim_risks, dim_fields = await _analyze_all_dimensions(text, doc_label)

    merged_fields = await _consolidate_missing_fields(text, dim_fields, doc_label)
    overall       = await _synthesize_overall_assessment(dim_risks, merged_fields, doc_label, text)

    return {
        "document_type":      "other",
        "document_label":     doc_label,
        "detected_risks":     dim_risks,
        "missing_fields":     merged_fields,
        "overall_assessment": overall,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def analyze_document_risks(text: str) -> dict:
    """
    Step 1 — Detect document type (4000-char classification window)

    Step 2 — Branch:
      Known type  → hybrid  (fixed-profile checklist + 3 parallel deep passes)
      Other type  → dynamic (3 parallel deep passes only)

    Step 3 — Programmatic placeholder scan
      Catches [bracketed] template values the LLM may miss because the
      document structure looks syntactically complete.

    Step 4 — Normalize, sort, and return fixed-structure response.
    """
    doc_type, doc_label = await _detect_document_type(text)
    logger.info(f"[risk_detection] Detected: {doc_type} ('{doc_label}')")

    _KNOWN_TYPES = {"contract", "employment", "nda", "lease", "invoice", "resume"}

    if doc_type in _KNOWN_TYPES:
        analysis = await _analyze_risks_hybrid(text, doc_type, doc_label)
    else:
        analysis = await _analyze_risks_dynamic(text, doc_label)

    analysis = _normalize_result(analysis)

    # Programmatic placeholder detection — independent of LLM, always reliable
    ph_risks, ph_fields = _detect_placeholders(text, doc_label)

    if ph_risks or ph_fields:
        existing_risk_keys = {_risk_key(r.get("risk_name", "")) for r in analysis["detected_risks"]}
        if _risk_key("Unfilled Template Placeholders") not in existing_risk_keys:
            analysis["detected_risks"] = ph_risks + analysis["detected_risks"]

        existing_field_keys = {_field_key(f.get("field_name", "")) for f in analysis["missing_fields"]}
        new_ph_fields = [
            f for f in ph_fields
            if _field_key(f.get("field_name", "")) not in existing_field_keys
        ]
        analysis["missing_fields"] = new_ph_fields + analysis["missing_fields"]

        # Re-synthesize to include placeholder findings in the executive summary
        if ph_risks:
            analysis["overall_assessment"] = await _synthesize_overall_assessment(
                analysis["detected_risks"],
                analysis["missing_fields"],
                doc_label,
                text,
            )

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
