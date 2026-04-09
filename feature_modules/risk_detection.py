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
# Chunking — improvement #3
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
# This is a programmatic pass — independent of LLM — so it is always reliable.
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r'\[([^\]\[]{2,60})\]')


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
        key = re.sub(r'\s+', ' ', m.strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(m.strip())

    if not unique:
        return [], []

    logger.info(f"[risk_detection] {len(unique)} unfilled placeholder(s) found in '{doc_label}'")

    examples = ", ".join(f"[{p}]" for p in unique[:3])
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
# Deduplication helpers — improvement #2 + #3
# ---------------------------------------------------------------------------

def _risk_key(risk_name: str) -> str:
    """Normalised key for deduplication — lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "", risk_name.lower())


_SEVERITY_RANK = {"High": 3, "Medium": 2, "Low": 1}


def _merge_risks(lists: list[list]) -> list[dict]:
    """
    Merges multiple detected_risks lists.
    Deduplicates by normalised risk_name.
    When the same risk appears in multiple sources, keeps the highest severity.
    Skips any item that is not a dict (guards against LLM returning strings).
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
                seen[key] = risk
            else:
                existing_rank = _SEVERITY_RANK.get(seen[key].get("severity", "Medium"), 2)
                incoming_rank = _SEVERITY_RANK.get(risk.get("severity", "Medium"), 2)
                if incoming_rank > existing_rank:
                    seen[key] = risk
                if seen[key].get("clause_found", "Not found") == "Not found" and \
                   risk.get("clause_found", "Not found") != "Not found":
                    seen[key]["clause_found"] = risk["clause_found"]
    return list(seen.values())


def _merge_missing_fields(lists: list[list]) -> list[dict]:
    """
    Merges multiple missing_fields lists, deduplicating by field_name.
    Skips any item that is not a dict (guards against LLM returning strings).
    """
    seen: dict[str, dict] = {}
    for field_list in lists:
        if not isinstance(field_list, list):
            continue
        for field in field_list:
            if not isinstance(field, dict):
                continue
            key = re.sub(r"[^a-z0-9]", "", field.get("field_name", "").lower())
            if key and key not in seen:
                seen[key] = field
    return list(seen.values())


# ---------------------------------------------------------------------------
# Consolidation helpers
# Run after all chunks are merged:
#   1. _consolidate_missing_fields — verifies missing fields against full document
#      (chunks may falsely flag fields as missing when they appear in another chunk)
#   2. _synthesize_overall_assessment — single LLM pass over the entire merged
#      risk list to produce an accurate cross-document executive summary
# ---------------------------------------------------------------------------

async def _consolidate_missing_fields(
    full_text: str,
    candidate_fields: list[dict],
    doc_label: str,
) -> list[dict]:
    """
    Takes the merged candidate missing_fields list from chunk analysis and
    verifies each field against the FULL document text.
    Returns only fields that are genuinely absent from the entire document.
    Silently returns the original list on any LLM/parse failure.
    """
    if not candidate_fields:
        return []

    field_names = [f.get("field_name", "") for f in candidate_fields if isinstance(f, dict)]
    if not field_names:
        return candidate_fields

    fields_list = "\n".join(f"  - {name}" for name in field_names)

    system_prompt = f"""You are a document reviewer checking whether specific fields are present in a "{doc_label}" document.

For each field listed below, determine whether it is present anywhere in the document (even partially or implicitly).

Fields to check:
{fields_list}

Return ONLY a valid JSON object:
{{
  "present":  ["<field name>", ...],
  "absent":   ["<field name>", ...]
}}

Rules:
- A field is "present" if it appears anywhere in the document — even if incomplete or informal
- A field is "absent" only if it is completely missing from the entire document
- Return ONLY the JSON — no explanation, no markdown"""

    try:
        raw    = await run_llm(full_text[:20000], system_prompt)
        parsed = extract_json_from_text(raw)
        absent = set(parsed.get("absent") or [])
        if not absent and not parsed.get("present"):
            raise ValueError("empty response")
        # Keep only fields confirmed absent
        result = [f for f in candidate_fields
                  if isinstance(f, dict) and f.get("field_name", "") in absent]
        logger.info(f"[risk_detection] Missing field consolidation: {len(candidate_fields)} candidates → {len(result)} confirmed absent")
        return result
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
        f"  - [{r.get('severity','?')}] {r.get('risk_name','?')}: {r.get('clause_found','')[:80]}"
        for r in merged_risks[:20]
    )
    fields_summary = "\n".join(
        f"  - [{f.get('importance','?')}] {f.get('field_name','?')}"
        for f in missing_fields[:10]
    )

    system_prompt = f"""You are a senior legal and financial risk analyst writing an executive summary.

You have completed a full risk analysis of a "{doc_label}" document.

Detected Risks ({len(merged_risks)} total):
{risks_summary or "  None"}

Missing Required Fields ({len(missing_fields)} total):
{fields_summary or "  None"}

Write a concise 3-4 sentence executive summary covering:
1. Overall risk level (High / Medium / Low) and why
2. The most critical risks found
3. Key missing fields and their impact
4. Overall recommendation

Return ONLY the summary text — no JSON, no bullet points, no headers."""

    try:
        raw = await run_llm(full_text[:5000], system_prompt)
        return raw.strip()
    except Exception as e:
        logger.warning(f"[risk_detection] Overall assessment synthesis failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Step 1 — Detect document type
# Improvement #1: classification window 1500 → 4000 chars
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

    raw    = await run_llm(text[:4000], system_prompt)
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
# Step 2a — Dynamic risk analysis (slug == "other" OR hybrid second pass)
# Improvement #3: chunked analysis
# Improvement #4: severity justification in prompt
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
- Only include fields actually missing or containing placeholder text
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


async def _analyze_risks_dynamic(text: str, doc_label: str) -> dict:
    """
    Dynamic risk analysis for unknown document types.
    Improvement #3: splits long documents into chunks, analyses each in parallel,
    then merges and deduplicates results.
    """
    chunks = _chunk_text(text)

    chunk_results = await asyncio.gather(*[
        _analyze_chunk_dynamic(chunk, doc_label, f"chunk {i+1}/{len(chunks)}")
        for i, chunk in enumerate(chunks)
    ])

    merged_risks  = _merge_risks([r.get("detected_risks", []) for r in chunk_results])
    merged_fields = await _consolidate_missing_fields(
        text,
        _merge_missing_fields([r.get("missing_fields", []) for r in chunk_results]),
        doc_label,
    )
    overall = await _synthesize_overall_assessment(merged_risks, merged_fields, doc_label, text)

    return {
        "document_type":      "other",
        "document_label":     doc_label,
        "detected_risks":     merged_risks,
        "missing_fields":     merged_fields,
        "overall_assessment": overall,
    }


# ---------------------------------------------------------------------------
# Step 2b — Fixed-profile risk analysis for the 6 known document types
# Improvement #3: chunked analysis
# Improvement #4: severity justification in prompt
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
Check if the following required fields are present in the document.
Flag any that are missing or unclear:
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
- Only include fields that are actually missing or contain placeholders
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
    Fixed-profile risk analysis for known document types.
    Improvement #3: chunked parallel analysis across full document.
    """
    profile    = _RISK_PROFILES[doc_type]
    risks_list = "\n".join(f"    - {r}" for r in profile["risks"])
    fields_list = "\n".join(f"    - {f}" for f in profile["required_fields"])
    chunks     = _chunk_text(text)

    chunk_results = await asyncio.gather(*[
        _analyze_chunk_fixed(chunk, doc_type, doc_label, risks_list, fields_list,
                             f"chunk {i+1}/{len(chunks)}")
        for i, chunk in enumerate(chunks)
    ])

    overall = max(
        (r.get("overall_assessment", "") for r in chunk_results),
        key=len,
        default="",
    )

    return {
        "document_type":      doc_type,
        "document_label":     doc_label,
        "detected_risks":     _merge_risks([r.get("detected_risks", []) for r in chunk_results]),
        "missing_fields":     _merge_missing_fields([r.get("missing_fields", []) for r in chunk_results]),
        "overall_assessment": overall,
    }


# ---------------------------------------------------------------------------
# Step 2 — Hybrid analysis for known types (improvement #2)
# Runs fixed-profile + dynamic in parallel, merges results.
# Fixed profile catches standard checklist risks reliably.
# Dynamic pass catches document-specific unusual clauses the checklist misses.
# ---------------------------------------------------------------------------

async def _analyze_risks_hybrid(text: str, doc_type: str, doc_label: str) -> dict:
    """
    Runs fixed-profile and dynamic analysis in parallel then merges.
    Used for all known document types.
    """
    logger.info(f"[risk_detection] Hybrid analysis — running fixed + dynamic in parallel for '{doc_label}'")

    fixed_result, dynamic_result = await asyncio.gather(
        _analyze_risks_for_type(text, doc_type, doc_label),
        _analyze_risks_dynamic(text, doc_label),
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
      - Known type  → _analyze_risks_hybrid  (fixed profile + dynamic in parallel)
      - Other type  → _analyze_risks_dynamic (fully dynamic, chunked)

    Both paths chunk the full document so no clause is ever missed due to truncation.
    All detected_risks include severity_reason for transparent, consistent ratings.
    Results are normalized to a guaranteed fixed structure before returning.
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
    # that LLMs may overlook because the document structure looks complete.
    ph_risks, ph_fields = _detect_placeholders(text, doc_label)

    if ph_risks or ph_fields:
        existing_risk_keys = {_risk_key(r.get("risk_name", "")) for r in analysis["detected_risks"]}
        if _risk_key("Unfilled Template Placeholders") not in existing_risk_keys:
            analysis["detected_risks"] = ph_risks + analysis["detected_risks"]

        existing_field_keys = {
            re.sub(r"[^a-z0-9]", "", f.get("field_name", "").lower())
            for f in analysis["missing_fields"]
        }
        new_ph_fields = [
            f for f in ph_fields
            if re.sub(r"[^a-z0-9]", "", f.get("field_name", "").lower()) not in existing_field_keys
        ]
        analysis["missing_fields"] = new_ph_fields + analysis["missing_fields"]

        # Re-synthesize overall_assessment to reflect placeholder findings
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
