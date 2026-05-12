"""
clause_extraction_prompts.py

Professional, intent-oriented LLM prompts for clause extraction.

Design principles (learned from prompts.py / obligation_detection.py):
  1. Role persona scoped to domain + document type (not generic "legal parser")
  2. Required keys + optional keys separated — LLM knows what MUST be found
  3. Each key has a plain-English description so the LLM doesn't guess meaning
  4. "Not Specified" as the default (never null, never missing key)
  5. Document-type-aware focus block (what clauses matter most for this type)
  6. Jurisdiction anchored where relevant
  7. Strict output rules at the end of every prompt

Import:
    from feature_modules.clause_extraction_prompts import (
        CLAUSE_EXTRACTION_PROMPTS,
        CLAUSE_FOCUS_HINTS,
        CLAUSE_KEY_DESCRIPTIONS,
        build_extraction_prompt,
    )
"""

# ---------------------------------------------------------------------------
# Canonical clause key → human-readable description
# Used in prompts AND in UI display labels
# ---------------------------------------------------------------------------

CLAUSE_KEY_DESCRIPTIONS: dict[str, str] = {
    "payment_terms":         "When and how payment is made — amount, due date, method, late fees",
    "contract_term":         "Duration of the agreement — start date, end date, or rolling term",
    "renewal_conditions":    "How and when the contract renews — auto-renewal, notice required",
    "termination":           "How either party can end the agreement — notice period, grounds, process",
    "liability":             "Limits on financial liability — cap amount, exclusions, carve-outs",
    "indemnification":       "Who indemnifies whom, for what events, and to what extent",
    "governing_law":         "Which jurisdiction's laws govern — state/country, court location",
    "dispute_resolution":    "How disputes are resolved — arbitration, mediation, litigation",
    "force_majeure":         "Events that excuse performance — pandemic, natural disaster, war",
    "confidentiality":       "What information must stay secret, for how long, and by whom",
    "intellectual_property": "Who owns work product, inventions, and pre-existing IP",
    "non_compete":           "Restrictions on working for competitors — duration, geography, scope",
    "non_solicitation":      "Restrictions on poaching employees or clients after engagement ends",
    "scope_of_work":         "What services or deliverables are included and excluded",
    "warranties":            "Representations and guarantees each party makes",
    "penalties":             "Fines, liquidated damages, or interest for breach or late payment",
    "pricing":               "Fee structure — fixed, hourly, milestone-based, escalation clauses",
    "delivery_timeline":     "Milestones, deadlines, and delivery schedule",
    "data_protection":       "Data handling, storage, and privacy obligations",
    "assignment":            "Whether rights can be transferred to a third party",
    "entire_agreement":      "Supersession clause — this document replaces all prior agreements",
    # NDA-specific
    "permitted_disclosures": "Disclosures allowed despite confidentiality — legal compulsion, regulators",
    "return_of_materials":   "Obligation to return or destroy confidential materials on termination",
    "residuals":             "Whether retained knowledge in unaided memory is excluded",
    # Employment / HR specific
    "probation_period":      "Trial period duration and confirmation process",
    "notice_period":         "Notice each party must give to end the relationship",
    "benefits":              "PF, health insurance, gratuity, leave, and other perks",
    "leave_policy":          "Casual, sick, earned leave entitlements and carry-forward rules",
    # Lease-specific
    "security_deposit":      "Refundable deposit amount and conditions for deduction or return",
    "rent_escalation":       "Annual or periodic rent increase clause — percentage or fixed amount",
    "maintenance":           "Who is responsible for repairs, upkeep, and society charges",
    "subletting":            "Whether tenant may sublet or assign the premises",
    # Invoice / payment doc
    "payment_due_date":      "Exact date by which payment must be made",
    "late_payment_penalty":  "Interest or fee charged on overdue amounts",
    "tax":                   "GST, TDS, VAT, or other applicable tax details",
}


# ---------------------------------------------------------------------------
# Document-type focus hints
# Injected into the prompt so the LLM prioritises the most important clauses
# for each document type — same pattern as obligation_detection.py's type_focus dict
# ---------------------------------------------------------------------------

CLAUSE_FOCUS_HINTS: dict[str, str] = {
    "contract": (
        "Focus especially on: payment_terms, liability, termination, renewal_conditions, "
        "governing_law, dispute_resolution, indemnification, force_majeure. "
        "For payment_terms extract the exact amount, due date, payment method, and any "
        "late-payment penalty. For liability extract the exact cap figure."
    ),
    "nda": (
        "Focus especially on: confidentiality, permitted_disclosures, return_of_materials, "
        "contract_term, governing_law, penalties, residuals. "
        "For confidentiality extract the exact scope of what information is covered. "
        "For contract_term extract whether confidentiality survives termination and for how long."
    ),
    "employment": (
        "Focus especially on: scope_of_work, probation_period, notice_period, benefits, "
        "non_compete, non_solicitation, intellectual_property, governing_law, leave_policy. "
        "For non_compete extract the exact duration and geographic restriction. "
        "For benefits extract PF, gratuity, health insurance details explicitly."
    ),
    "lease": (
        "Focus especially on: payment_terms, security_deposit, contract_term, termination, "
        "rent_escalation, maintenance, subletting, renewal_conditions, governing_law. "
        "For security_deposit extract the amount, deduction conditions, and return timeline. "
        "For rent_escalation extract the percentage and trigger date."
    ),
    "invoice": (
        "Focus especially on: payment_terms, payment_due_date, late_payment_penalty, "
        "tax, pricing, penalties. "
        "Extract the exact invoice number, due date, and any dispute window."
    ),
    "resume": (
        "This is not a contract — extract professional information sections only: "
        "scope_of_work (as job responsibilities), benefits (compensation/salary if mentioned), "
        "contract_term (employment period), governing_law (work location/jurisdiction)."
    ),
    "other": (
        "Extract any clause-like statement that defines an obligation, right, restriction, "
        "timeline, or financial term. Prefer extracting more clauses over fewer."
    ),
}


# ---------------------------------------------------------------------------
# Per-document-type extraction prompts
# Pattern from prompts.py:
#   - Role persona scoped to doc type
#   - Required keys (must be found if present)
#   - Optional keys (extract if present)
#   - "Not Specified" default
#   - Output rules
# ---------------------------------------------------------------------------

CLAUSE_EXTRACTION_PROMPTS: dict[str, str] = {

    # ── Contract (generic / service agreement) ────────────────────────────
    "contract": """You are a senior contracts attorney extracting clause text from a legal agreement.
Your task: find and copy the verbatim or closely paraphrased text for each clause below.

REQUIRED clauses — extract these if they exist anywhere in the document:
  payment_terms        : When and how payment is made — amount, due date, method, late fees
  contract_term        : Duration — start date, end date, or rolling term
  termination          : How either party can end the agreement — notice period, grounds
  liability            : Limit on financial liability — cap amount, exclusions
  governing_law        : Which jurisdiction's laws govern and which courts have jurisdiction
  scope_of_work        : What services or deliverables are included

OPTIONAL clauses — extract these if present:
  renewal_conditions   : Auto-renewal terms, notice required to prevent renewal
  indemnification      : Who indemnifies whom and for what events
  dispute_resolution   : Arbitration, mediation, or litigation clause
  force_majeure        : Events excusing performance — pandemic, disaster, war
  confidentiality      : Non-disclosure obligations between the parties
  intellectual_property: Ownership of work product and pre-existing IP
  non_compete          : Restrictions on working for competitors after contract ends
  non_solicitation     : Restrictions on poaching employees or clients
  warranties           : Representations and guarantees each party makes
  penalties            : Fines, liquidated damages, or interest for breach
  pricing              : Fee structure — fixed, hourly, milestone-based
  delivery_timeline    : Milestones, deadlines, and delivery schedule
  data_protection      : Data handling, storage, and privacy obligations
  assignment           : Whether rights can be transferred to a third party
  entire_agreement     : Supersession clause

Rules:
- Value = the actual clause text from the document (verbatim quote or close paraphrase)
- Do NOT summarise or shorten — preserve the original wording
- If a clause is absent → return "Not Specified" for that key
- Return ONLY a valid JSON object with the exact keys above
- No extra keys, no nested objects, no arrays — values are plain strings only
- No preamble, no explanation outside the JSON""",

    # ── NDA ───────────────────────────────────────────────────────────────
    "nda": """You are a senior IP and confidentiality law specialist extracting clause text from a Non-Disclosure Agreement.
Your task: find and copy the verbatim or closely paraphrased text for each clause below.

REQUIRED clauses — extract these if they exist anywhere in the document:
  confidentiality      : Exact scope of what information is covered as confidential
  contract_term        : Duration of the NDA — and whether confidentiality survives termination
  governing_law        : Which jurisdiction's laws govern; which courts have jurisdiction
  permitted_disclosures: Disclosures allowed despite confidentiality (legal compulsion, regulators)
  penalties            : Consequences and remedies for breach of confidentiality

OPTIONAL clauses — extract these if present:
  return_of_materials  : Obligation to return or destroy confidential materials on termination
  residuals            : Whether retained knowledge in unaided memory is excluded
  indemnification      : Indemnification for breach of confidentiality
  non_compete          : Any post-NDA competitive restrictions
  assignment           : Whether NDA obligations can be transferred
  entire_agreement     : Supersession clause
  dispute_resolution   : How disputes are resolved
  intellectual_property: Any IP ownership provisions in the NDA

Rules:
- Value = the actual clause text from the document (verbatim quote or close paraphrase)
- Do NOT summarise — preserve original wording
- If a clause is absent → return "Not Specified" for that key
- Return ONLY a valid JSON object with the exact keys above
- No extra keys, no nested objects, no arrays — values are plain strings only
- No preamble, no explanation outside the JSON""",

    # ── Employment ────────────────────────────────────────────────────────
    "employment": """You are an employment law specialist extracting clause text from an Employment Contract or Appointment Letter.
Your task: find and copy the verbatim or closely paraphrased text for each clause below.

REQUIRED clauses — extract these if they exist anywhere in the document:
  scope_of_work        : Job duties, responsibilities, role description
  contract_term        : Commencement date and whether permanent or fixed-term
  probation_period     : Trial period duration and confirmation process
  notice_period        : Notice each party must give to terminate employment
  benefits             : PF, health insurance, gratuity, and other perks
  governing_law        : Which jurisdiction's laws govern — state, applicable labour acts

OPTIONAL clauses — extract these if present:
  leave_policy         : Casual, sick, earned leave entitlements and carry-forward rules
  non_compete          : Post-employment competitive restrictions — duration and geography
  non_solicitation     : Restrictions on poaching employees or clients after leaving
  intellectual_property: Who owns inventions and work product created during employment
  confidentiality      : Non-disclosure obligations during and after employment
  termination          : Grounds and process for termination by either party
  data_protection      : Employee data handling obligations
  penalties            : Clawback, bond, or penalty clauses
  dispute_resolution   : How employment disputes are resolved
  entire_agreement     : Supersession clause

Rules:
- Value = the actual clause text from the document (verbatim quote or close paraphrase)
- Do NOT summarise — preserve original wording
- If a clause is absent → return "Not Specified" for that key
- Return ONLY a valid JSON object with the exact keys above
- No extra keys, no nested objects, no arrays — values are plain strings only
- No preamble, no explanation outside the JSON""",

    # ── Lease ─────────────────────────────────────────────────────────────
    "lease": """You are a real estate attorney extracting clause text from a Lease or Leave and Licence Agreement.
Your task: find and copy the verbatim or closely paraphrased text for each clause below.

REQUIRED clauses — extract these if they exist anywhere in the document:
  payment_terms        : Monthly rent amount, due date, and payment method
  security_deposit     : Deposit amount, deduction conditions, and return timeline
  contract_term        : Lease start date, end date, and lock-in period
  termination          : Notice required to vacate and early-termination conditions
  governing_law        : Which state's laws govern; applicable Rent Control Act

OPTIONAL clauses — extract these if present:
  rent_escalation      : Annual rent increase — percentage and trigger date
  maintenance          : Who handles repairs, maintenance, and society charges
  subletting           : Whether tenant may sublet or assign the premises
  renewal_conditions   : How and when the lease can be renewed
  penalties            : Late payment interest, penalties for breach
  permitted_use        : What the premises may be used for
  entry_by_landlord    : Landlord's right of entry — notice required
  entire_agreement     : Supersession clause
  dispute_resolution   : How disputes are resolved

Rules:
- Value = the actual clause text from the document (verbatim quote or close paraphrase)
- Do NOT summarise — preserve original wording
- If a clause is absent → return "Not Specified" for that key
- Return ONLY a valid JSON object with the exact keys above
- No extra keys, no nested objects, no arrays — values are plain strings only
- No preamble, no explanation outside the JSON""",

    # ── Invoice ───────────────────────────────────────────────────────────
    "invoice": """You are a financial and billing analyst extracting payment terms from an invoice or billing document.
Your task: find and copy the verbatim or closely paraphrased text for each clause below.

REQUIRED clauses — extract these if they exist anywhere in the document:
  payment_terms        : Payment method, bank details, and how to pay
  payment_due_date     : Exact date by which payment must be made
  pricing              : Line items, unit prices, quantities, and subtotal
  tax                  : GST, TDS, VAT, or other applicable taxes with rates and amounts

OPTIONAL clauses — extract these if present:
  late_payment_penalty : Interest or fee charged on overdue amounts
  penalties            : Any other penalties for non-payment or dispute
  dispute_resolution   : Window and process for raising billing disputes
  entire_agreement     : Any terms and conditions reference

Rules:
- Value = the actual clause text from the document (verbatim quote or close paraphrase)
- Do NOT summarise — preserve original wording
- If a clause is absent → return "Not Specified" for that key
- Return ONLY a valid JSON object with the exact keys above
- No extra keys, no nested objects, no arrays — values are plain strings only
- No preamble, no explanation outside the JSON""",

    # ── General / other ───────────────────────────────────────────────────
    "other": """You are a legal analyst extracting clause text from a document.
Your task: identify and extract any clause-like statements that define an obligation,
right, restriction, financial term, or timeline.

Extract all of the following clause types that exist in the document:
  payment_terms        : Payment amounts, due dates, and methods
  contract_term        : Duration of the agreement or commitment
  termination          : How the arrangement can be ended
  liability            : Limits on financial responsibility
  confidentiality      : Non-disclosure obligations
  governing_law        : Applicable law and jurisdiction
  dispute_resolution   : How disputes are resolved
  force_majeure        : Events excusing performance
  penalties            : Fines or consequences for breach
  intellectual_property: Ownership of created work
  non_compete          : Competitive restrictions
  warranties           : Guarantees made by any party
  scope_of_work        : What is covered by the arrangement
  data_protection      : Data handling obligations
  entire_agreement     : Supersession clause

Rules:
- Value = the actual clause text from the document (verbatim quote or close paraphrase)
- Do NOT summarise — preserve original wording
- If a clause is absent → return "Not Specified" for that key
- Return ONLY a valid JSON object with the exact keys above
- No extra keys, no nested objects, no arrays — values are plain strings only
- No preamble, no explanation outside the JSON""",
}

# Map classify_document output → prompt key
# Handles synonyms and variant labels from key_clause_extraction.classify_document
DOC_TYPE_TO_PROMPT_KEY: dict[str, str] = {
    "contract":             "contract",
    "service_agreement":    "contract",
    "general_contract":     "contract",
    "freelancer_agreement": "contract",
    "consulting_agreement": "contract",
    "nda":                  "nda",
    "non_disclosure":       "nda",
    "employment":           "employment",
    "employment_contract":  "employment",
    "lease":                "lease",
    "lease_agreement":      "lease",
    "rent_agreement":       "lease",
    "invoice":              "invoice",
    "billing":              "invoice",
    "resume":               "other",   # resume has no clauses
    "other":                "other",
}


def build_extraction_prompt(text: str, doc_type: str) -> str:
    """
    Build a complete clause extraction prompt for a given document type.

    Combines:
      1. The per-doc-type role + key description block
      2. The document-type focus hint (which clauses matter most)
      3. The document text

    Parameters
    ----------
    text     : document text (will be truncated to 6000 chars)
    doc_type : raw output from classify_document() — mapped to prompt key

    Returns
    -------
    Complete prompt string ready to send to run_llm().
    """
    prompt_key  = DOC_TYPE_TO_PROMPT_KEY.get(doc_type.lower(), "other")
    base_prompt = CLAUSE_EXTRACTION_PROMPTS[prompt_key]
    focus_hint  = CLAUSE_FOCUS_HINTS.get(prompt_key, CLAUSE_FOCUS_HINTS["other"])

    return f"""{base_prompt}

EXTRACTION FOCUS FOR THIS DOCUMENT TYPE ({doc_type.upper()}):
{focus_hint}

DOCUMENT TEXT:
---
{text[:12000]}
---"""