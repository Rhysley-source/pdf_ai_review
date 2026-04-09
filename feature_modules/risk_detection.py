import asyncio
import logging
import re
from llm_model.ai_model import run_llm
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document-type risk profiles
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
            "Pet or guest restrictions that are excessively broad",
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
            "Skills section missing, outdated, or does not match the stated job experience",
            "Inconsistent date formats or reverse-chronological order not followed",
            "No professional summary or objective statement",
            "LinkedIn, GitHub, or portfolio links missing for technical or creative roles",
            "Only generic responsibilities listed with no evidence of impact",
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
# Splits long documents into overlapping windows so no clause is missed.
# ---------------------------------------------------------------------------

_CHUNK_SIZE    = 20_000   # chars per chunk — larger window reduces fragmentation
_CHUNK_OVERLAP = 1_000    # overlap between consecutive chunks


def _chunk_text(text: str) -> list[str]:
    """
    Splits text into overlapping chunks of _CHUNK_SIZE chars.
    Returns a single-element list when the text fits in one chunk.
    """
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
# Placeholder detector
# Scans for unfilled template values like [Email], [Company Name], [Start Date].
# Excludes pure numeric citations like [1], [42], [123].
# This is a programmatic pass — independent of LLM — so it is always reliable.
# ---------------------------------------------------------------------------

# Matches [text] but NOT pure numeric citations like [1] or [42]
_PLACEHOLDER_RE = re.compile(r'\[([^\]\[]{2,60})\]')
_NUMERIC_RE     = re.compile(r'^\s*\d+\s*$')


def _detect_placeholders(text: str, doc_label: str) -> tuple[list[dict], list[dict]]:
    """
    Returns (risk_entries, missing_field_entries) for any [bracketed placeholder]
    patterns found in the document text.
    """
    matches = _PLACEHOLDER_RE.findall(text)
    if not matches:
        return [], []

    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        # Skip pure numeric references like [1], [42]
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
            "dates, and names are still showing template text instead of real data."
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
# Enforces exact, fixed field names on the LLM output regardless of any
# variation the model introduces.
# ---------------------------------------------------------------------------

def _normalize_result(result: dict) -> dict:
    """
    Guarantees that every detected_risk and missing_field item always has
    exactly these keys:

    detected_risks items:
        risk_name       (str)
        severity        (str) — "High" | "Medium" | "Low"
        severity_reason (str) — why this severity was assigned
        clause_found    (str)
        impact          (str)
        mitigation      (str)

    missing_fields items:
        field_name  (str)
        importance  (str) — "Critical" | "Important" | "Optional"
        reason      (str)
    """
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
# Deduplication helpers
# ---------------------------------------------------------------------------

def _field_key(name: str) -> str:
    """Normalised key — lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _risk_key(risk_name: str) -> str:
    """Normalised key for deduplication — lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "", risk_name.lower())


_SEVERITY_RANK = {"High": 3, "Medium": 2, "Low": 1}


def _merge_risks(lists: list[list]) -> list[dict]:
    """
    Merges multiple detected_risks lists.
    Deduplicates by normalised risk_name.
    When the same risk appears in multiple sources, keeps the highest severity
    and the best clause_found excerpt.
    Output is sorted High → Medium → Low.
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
                # Prefer a real clause excerpt over "Not found"
                if seen[key].get("clause_found", "Not found") in ("Not found", "") and \
                   risk.get("clause_found", "Not found") not in ("Not found", ""):
                    seen[key]["clause_found"] = risk["clause_found"]

    # Sort High → Medium → Low
    return sorted(
        seen.values(),
        key=lambda r: _SEVERITY_RANK.get(r.get("severity", "Medium"), 2),
        reverse=True,
    )


def _merge_missing_fields(lists: list[list]) -> list[dict]:
    """
    Merges multiple missing_fields lists, deduplicating by field_name.
    Keeps Critical > Important > Optional when same field appears multiple times.
    """
    _IMPORTANCE_RANK = {"Critical": 3, "Important": 2, "Optional": 1}
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

    # Sort Critical → Important → Optional
    return sorted(
        seen.values(),
        key=lambda f: _IMPORTANCE_RANK.get(f.get("importance", "Important"), 2),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Consolidation helpers
# Run after all chunks are merged:
#   1. _consolidate_missing_fields — verifies missing fields against full document,
#      treats placeholder-filled fields as still absent
#   2. _synthesize_overall_assessment — single LLM pass over the entire merged
#      risk list to produce an accurate cross-document executive summary
# ---------------------------------------------------------------------------

async def _consolidate_missing_fields(
    full_text: str,
    candidate_fields: list[dict],
    doc_label: str,
) -> list[dict]:
    """
    Verifies each candidate missing field against the FULL document text.
    Returns only fields genuinely absent (including those only filled with placeholders).
    Falls back to the original list on any LLM/parse failure.
    """
    if not candidate_fields:
        return []

    field_names = [f.get("field_name", "") for f in candidate_fields if isinstance(f, dict)]
    if not field_names:
        return candidate_fields

    fields_list = "\n".join(f"  - {name}" for name in field_names)

    system_prompt = f"""You are a document reviewer checking whether specific fields are present in a "{doc_label}" document.

For each field listed below, determine whether it is present in the document with real content.

Fields to check:
{fields_list}

Return ONLY a valid JSON object:
{{
  "present":  ["<field name>", ...],
  "absent":   ["<field name>", ...]
}}

Rules:
- A field is "present" ONLY if it contains a real, actual value
- A field is "absent" if it is completely missing from the document
- A field is also "absent" if it exists but contains only a placeholder like [Email], [Company Name], [Value] — these are templates, not real data
- Return field names exactly as they appear in the list above
- Return ONLY the JSON — no explanation, no markdown"""

    try:
        raw    = await run_llm(full_text[:20_000], system_prompt)
        parsed = extract_json_from_text(raw)
        absent_raw = parsed.get("absent") or []
        if not absent_raw and not parsed.get("present"):
            raise ValueError("empty response")

        # Normalise to handle LLM returning slightly different casing/punctuation
        absent_keys = {_field_key(a) for a in absent_raw if isinstance(a, str)}
        result = [
            f for f in candidate_fields
            if isinstance(f, dict) and _field_key(f.get("field_name", "")) in absent_keys
        ]
        logger.info(
            f"[risk_detection] Missing field consolidation: "
            f"{len(candidate_fields)} candidates → {len(result)} confirmed absent"
        )
        return result if result else candidate_fields
    except Exception as e:
        logger.warning(f"[risk_detection] Missing field consolidation failed — keeping original list: {e}")
        return candidate_fields


async def _synthesize_overall_assessment(
    merged_risks: list[dict],
    missing_fields: list[dict],
    doc_label: str,
    full_text: str,
) -> str:
    """
    Produces a single accurate overall_assessment by synthesizing
    the full merged risk list and the entire document.
    """
    if not merged_risks and not missing_fields:
        return f"No significant risks or missing fields were identified in this {doc_label}."

    risks_summary = "\n".join(
        f"  - [{r.get('severity','?')}] {r.get('risk_name','?')}: {r.get('clause_found','')[:100]}"
        for r in merged_risks[:20]
    )
    fields_summary = "\n".join(
        f"  - [{f.get('importance','?')}] {f.get('field_name','?')}"
        for f in missing_fields[:15]
    )

    high_count   = sum(1 for r in merged_risks if r.get("severity") == "High")
    medium_count = sum(1 for r in merged_risks if r.get("severity") == "Medium")

    system_prompt = f"""You are a senior legal and financial risk analyst writing an executive summary.

You have completed a full risk analysis of a "{doc_label}" document.

Detected Risks ({len(merged_risks)} total — {high_count} High, {medium_count} Medium):
{risks_summary or "  None"}

Missing Required Fields ({len(missing_fields)} total):
{fields_summary or "  None"}

Write a concise 3-4 sentence executive summary covering:
1. Overall risk level (High / Medium / Low) and the primary reason
2. The most critical risks or red flags found
3. The most important missing fields and their practical impact
4. A clear recommendation (e.g., do not sign, fill in placeholders first, negotiate specific clauses)

Return ONLY the summary text — no JSON, no bullet points, no headers."""

    try:
        raw = await run_llm(full_text[:5_000], system_prompt)
        return raw.strip()
    except Exception as e:
        logger.warning(f"[risk_detection] Overall assessment synthesis failed: {e}")
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
# Step 2a — Dynamic risk analysis (slug == "other" OR called from hybrid)
# _analyze_chunk_dynamic  : single chunk, raw LLM result
# _raw_analyze_risks_dynamic : chunked merge, NO consolidation/synthesis
#                              (used by hybrid — hybrid does its own final pass)
# _analyze_risks_dynamic  : full pipeline including consolidation + synthesis
#                              (used as standalone for "other" document types)
# ---------------------------------------------------------------------------

async def _analyze_chunk_dynamic(chunk: str, doc_label: str, chunk_label: str) -> dict:
    """Runs dynamic risk analysis on a single text chunk."""
    system_prompt = f"""You are a senior legal and financial risk analyst.

The document provided is a "{doc_label}".

YOUR TASK — perform a complete risk analysis on the document content:

PART 1 — RISK DETECTION:
Identify and flag any risks present. Think about:
- Clauses that are dangerous or unfair to one party
- Unusual or non-standard language
- Excessive liability, penalties, or obligations
- Rights that are waived or restricted
- Ambiguous language that could be exploited

PART 2 — MISSING REQUIRED FIELDS:
Identify fields or sections typically required in a "{doc_label}" that are missing or unclear.

OUTPUT FORMAT — return ONLY valid JSON in exactly this structure:
{{
  "document_type": "other",
  "document_label": "{doc_label}",
  "detected_risks": [
    {{
      "risk_name": "<specific risk name>",
      "severity": "High | Medium | Low",
      "severity_reason": "<one sentence explaining why this severity level was assigned>",
      "clause_found": "<exact quote or short description from the document, or 'Not found'>",
      "impact": "<why this is dangerous or problematic>",
      "mitigation": "<how to fix, negotiate, or protect against this>"
    }}
  ],
  "missing_fields": [
    {{
      "field_name": "<field or section missing>",
      "importance": "Critical | Important | Optional",
      "reason": "<why this field is needed>"
    }}
  ],
  "overall_assessment": "<2-3 sentence executive summary of the overall risk profile>"
}}

Rules:
- Risks must be SPECIFIC to "{doc_label}" — not generic boilerplate
- Only include risks actually present in the document
- Treat any [bracketed text] (e.g. [Email], [Company Name], [Start Date]) as an unfilled placeholder — flag as missing field
- A field is absent if it is completely missing OR only contains placeholder text like [value]
- Justify every severity rating in severity_reason
- If no risks found, return detected_risks as []
- If no fields missing, return missing_fields as []"""

    logger.info(f"[risk_detection] Dynamic analysis — {chunk_label}")
    raw    = await run_llm(chunk, system_prompt)
    result = extract_json_from_text(raw)

    if not result or (not result.get("detected_risks") and not result.get("overall_assessment")):
        logger.warning(f"[risk_detection] Dynamic {chunk_label}: empty parse — retrying")
        retry_system = f"""Return ONLY a raw JSON object. No markdown, no backticks.

Analyse the "{doc_label}" document for risks and missing fields.
Fill in this exact structure:

{{
  "document_type": "other",
  "document_label": "{doc_label}",
  "detected_risks": [],
  "missing_fields": [],
  "overall_assessment": ""
}}"""
        raw    = await run_llm(chunk, retry_system)
        result = extract_json_from_text(raw)

    if not result:
        result = {
            "document_type": "other",
            "document_label": doc_label,
            "detected_risks": [],
            "missing_fields": [],
            "overall_assessment": "",
        }

    return result


async def _raw_analyze_risks_dynamic(text: str, doc_label: str) -> dict:
    """
    Dynamic risk analysis — chunk + merge only, NO consolidation or synthesis.
    Called from _analyze_risks_hybrid so the hybrid can do its own single final pass.
    """
    chunks = _chunk_text(text)
    chunk_results = await asyncio.gather(*[
        _analyze_chunk_dynamic(chunk, doc_label, f"chunk {i+1}/{len(chunks)}")
        for i, chunk in enumerate(chunks)
    ])
    return {
        "document_type":      "other",
        "document_label":     doc_label,
        "detected_risks":     _merge_risks([r.get("detected_risks", []) for r in chunk_results]),
        "missing_fields":     _merge_missing_fields([r.get("missing_fields", []) for r in chunk_results]),
        "overall_assessment": "",
    }


async def _analyze_risks_dynamic(text: str, doc_label: str) -> dict:
    """
    Full dynamic pipeline: chunk → merge → consolidate → synthesize.
    Used as the standalone path for "other" document types.
    """
    raw = await _raw_analyze_risks_dynamic(text, doc_label)

    merged_risks  = raw["detected_risks"]
    merged_fields = await _consolidate_missing_fields(text, raw["missing_fields"], doc_label)
    overall       = await _synthesize_overall_assessment(merged_risks, merged_fields, doc_label, text)

    return {
        "document_type":      "other",
        "document_label":     doc_label,
        "detected_risks":     merged_risks,
        "missing_fields":     merged_fields,
        "overall_assessment": overall,
    }


# ---------------------------------------------------------------------------
# Step 2b — Fixed-profile risk analysis for the 6 known document types
# _analyze_chunk_fixed       : single chunk, raw LLM result
# _analyze_risks_for_type    : chunked merge only (no consolidation/synthesis)
#                              — always called from hybrid
# ---------------------------------------------------------------------------

async def _analyze_chunk_fixed(chunk: str, doc_type: str, doc_label: str,
                                risks_list: str, fields_list: str,
                                chunk_label: str) -> dict:
    """Runs fixed-profile risk analysis on a single text chunk."""
    system_prompt = f"""You are a legal and financial risk analyst specializing in {doc_label} documents.

TASK 1 — RISK DETECTION:
Scan the document for the following risk categories and flag any that are present:
{risks_list}

TASK 2 — MISSING REQUIRED FIELDS:
Check if the following required fields are present in the document with REAL values.
Flag any that are missing or only contain placeholder text:
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
- Treat any [bracketed text] (e.g. [Email], [Company Name], [Start Date]) as an unfilled placeholder — flag as missing field
- A field with only placeholder text like [value] is missing — real content is required
- Justify every severity rating in severity_reason
- If no risks found, return detected_risks as []
- If no fields missing, return missing_fields as []"""

    logger.info(f"[risk_detection] Fixed profile ({doc_type}) — {chunk_label}")
    raw    = await run_llm(chunk, system_prompt)
    result = extract_json_from_text(raw)

    if not result or (not result.get("detected_risks") and not result.get("overall_assessment")):
        logger.warning(f"[risk_detection] Fixed {chunk_label}: empty parse — retrying")
        retry_system = f"""Return ONLY a raw JSON object. No markdown, no backticks, no explanation.

Analyse the {doc_label} document for these risks:
{risks_list}

Fill in this exact structure:
{{
  "document_type": "{doc_type}",
  "document_label": "{doc_label}",
  "detected_risks": [],
  "missing_fields": [],
  "overall_assessment": ""
}}"""
        raw    = await run_llm(chunk, retry_system)
        result = extract_json_from_text(raw)

    if not result:
        result = {
            "document_type": doc_type,
            "document_label": doc_label,
            "detected_risks": [],
            "missing_fields": [],
            "overall_assessment": "",
        }

    return result


async def _analyze_risks_for_type(text: str, doc_type: str, doc_label: str) -> dict:
    """
    Fixed-profile chunked analysis — chunk + merge only (no consolidation/synthesis).
    Always called from _analyze_risks_hybrid which handles the final pass.
    """
    profile     = _RISK_PROFILES[doc_type]
    risks_list  = "\n".join(f"    - {r}" for r in profile["risks"])
    fields_list = "\n".join(f"    - {f}" for f in profile["required_fields"])
    chunks      = _chunk_text(text)

    chunk_results = await asyncio.gather(*[
        _analyze_chunk_fixed(chunk, doc_type, doc_label, risks_list, fields_list,
                             f"chunk {i+1}/{len(chunks)}")
        for i, chunk in enumerate(chunks)
    ])

    return {
        "document_type":      doc_type,
        "document_label":     doc_label,
        "detected_risks":     _merge_risks([r.get("detected_risks", []) for r in chunk_results]),
        "missing_fields":     _merge_missing_fields([r.get("missing_fields", []) for r in chunk_results]),
        "overall_assessment": "",
    }


# ---------------------------------------------------------------------------
# Step 2 — Hybrid analysis for known types
# Runs fixed-profile + dynamic in parallel, merges, then single consolidation
# and synthesis pass on the combined result.
# ---------------------------------------------------------------------------

async def _analyze_risks_hybrid(text: str, doc_type: str, doc_label: str) -> dict:
    """
    Runs fixed-profile and raw dynamic analysis in parallel then merges.
    A single consolidation + synthesis pass runs on the final merged result.
    """
    logger.info(f"[risk_detection] Hybrid analysis — running fixed + dynamic in parallel for '{doc_label}'")

    fixed_result, dynamic_result = await asyncio.gather(
        _analyze_risks_for_type(text, doc_type, doc_label),
        _raw_analyze_risks_dynamic(text, doc_label),
    )

    merged_risks  = _merge_risks([
        fixed_result.get("detected_risks", []),
        dynamic_result.get("detected_risks", []),
    ])
    merged_fields = await _consolidate_missing_fields(
        text,
        _merge_missing_fields([
            fixed_result.get("missing_fields", []),
            dynamic_result.get("missing_fields", []),
        ]),
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def analyze_document_risks(text: str) -> dict:
    """
    Step 1 — detect document type (classification window: 4000 chars)

    Step 2 — branch based on slug:
      - Known type  → _analyze_risks_hybrid  (fixed profile + dynamic in parallel,
                       single consolidation + synthesis on the merged result)
      - Other type  → _analyze_risks_dynamic (fully dynamic, chunked, consolidated)

    Step 3 — programmatic placeholder scan (always runs, independent of LLM)
      Catches [bracketed] template values the LLM may have missed because the
      document structure looked complete.

    All detected_risks are sorted High → Medium → Low.
    All missing_fields are sorted Critical → Important → Optional.
    Response is normalized to a guaranteed fixed structure before returning.
    """
    doc_type, doc_label = await _detect_document_type(text)
    logger.info(f"[risk_detection] Detected: {doc_type} ('{doc_label}')")

    _KNOWN_TYPES = {"contract", "employment", "nda", "lease", "invoice", "resume"}

    if doc_type in _KNOWN_TYPES:
        analysis = await _analyze_risks_hybrid(text, doc_type, doc_label)
    else:
        analysis = await _analyze_risks_dynamic(text, doc_label)

    analysis = _normalize_result(analysis)

    # Programmatic placeholder detection — catches [Email], [Company Name], etc.
    # that LLMs may overlook because the document structure looks syntactically complete.
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

        # Re-synthesize to reflect placeholder findings in the executive summary
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
