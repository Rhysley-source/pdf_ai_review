import logging
from llm_model.ai_model import _run_inference_json_mini as _run_inference_json
from utils.json_utils import extract_json_raw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-document-type checklists
#
# Each document type has its own fixed list of checklist items.
# The AI answers present / absent / not_applicable for every item.
# Count is stable because the question set is fixed per type.
#
# category:
#   dangerous — clause IS present and harms the signer
#   unusual   — non-standard language a reasonable party would not expect
#   missing   — standard protection absent from this document
# ---------------------------------------------------------------------------

_CHECKLISTS: dict[str, list[dict]] = {

    "nda": [
        {"id": "perpetual_confidentiality",    "cat": "dangerous", "sev": "High",     "label": "Perpetual confidentiality obligation",      "check": "Does the confidentiality obligation last forever with no expiry date?"},
        {"id": "one_sided_obligations",        "cat": "dangerous", "sev": "High",     "label": "One-sided confidentiality",                 "check": "Is only one party bound by confidentiality while the other has no obligation?"},
        {"id": "overly_broad_definition",      "cat": "dangerous", "sev": "High",     "label": "Overly broad definition of confidential info","check": "Is confidential information defined so broadly that almost anything qualifies?"},
        {"id": "residuals_clause",             "cat": "dangerous", "sev": "High",     "label": "Residuals clause",                          "check": "Is there a residuals clause allowing use of retained memory of confidential information?"},
        {"id": "unlimited_remedies",           "cat": "dangerous", "sev": "Critical", "label": "Unlimited remedies / injunctions",          "check": "Can one party seek unlimited damages or automatic injunctions without proving actual harm?"},
        {"id": "no_return_destruction",        "cat": "missing",   "sev": "Medium",   "label": "No return or destruction clause",           "check": "Is there NO clause requiring return or destruction of confidential information after the NDA ends?"},
        {"id": "no_carve_outs",                "cat": "missing",   "sev": "Medium",   "label": "No carve-outs for public information",      "check": "Are there NO exceptions for information that is already public, independently developed, or disclosed by law?"},
        {"id": "no_permitted_disclosures",     "cat": "missing",   "sev": "Medium",   "label": "No permitted disclosure exceptions",        "check": "Is there NO exception allowing disclosure when required by law, court order, or regulators?"},
        {"id": "no_governing_law",             "cat": "missing",   "sev": "Medium",   "label": "No governing law clause",                   "check": "Is the governing law and jurisdiction NOT stated?"},
        {"id": "no_term_defined",              "cat": "missing",   "sev": "High",     "label": "No defined NDA term or duration",           "check": "Is the duration of the NDA not clearly specified?"},
        {"id": "assignment_no_consent",        "cat": "unusual",   "sev": "Medium",   "label": "Assignment without consent",                "check": "Can the NDA be transferred to a third party without the other party's consent?"},
        {"id": "deemed_acceptance",            "cat": "unusual",   "sev": "Medium",   "label": "Deemed acceptance of changes",              "check": "Is silence or inaction treated as acceptance of changes to the NDA terms?"},
    ],

    "job_offer": [
        {"id": "at_will_no_notice",            "cat": "dangerous", "sev": "High",     "label": "At-will termination without notice",        "check": "Can the employer terminate employment immediately without any notice period or reason?"},
        {"id": "broad_ip_assignment",          "cat": "dangerous", "sev": "Critical", "label": "Broad IP assignment including personal work","check": "Does the offer letter assign all inventions — including personal projects made outside work hours — to the employer?"},
        {"id": "clawback_provisions",          "cat": "dangerous", "sev": "High",     "label": "Clawback of bonus or equity",               "check": "Can the employer take back bonuses or equity already paid or vested?"},
        {"id": "mandatory_arbitration",        "cat": "dangerous", "sev": "High",     "label": "Mandatory arbitration waiving court rights", "check": "Is the employee forced into arbitration and prevented from suing in court?"},
        {"id": "excessive_non_compete",        "cat": "dangerous", "sev": "High",     "label": "Excessive non-compete clause",              "check": "Is there a non-compete clause with unreasonably wide scope, geography, or duration?"},
        {"id": "non_solicitation",             "cat": "dangerous", "sev": "Medium",   "label": "Non-solicitation clause",                   "check": "Does it prevent the employee from hiring former colleagues or working with former clients after leaving?"},
        {"id": "vague_performance_metrics",    "cat": "unusual",   "sev": "Medium",   "label": "Vague or subjective performance metrics",   "check": "Are performance criteria for appraisal or termination subjective and undefined?"},
        {"id": "forced_relocation",            "cat": "unusual",   "sev": "Medium",   "label": "Forced relocation clause",                  "check": "Can the employer relocate the employee to any location without consent or compensation?"},
        {"id": "no_salary_clearly_stated",     "cat": "missing",   "sev": "Critical", "label": "Salary or compensation not clearly stated", "check": "Is the salary, CTC, or compensation NOT clearly specified in the offer?"},
        {"id": "no_notice_period",             "cat": "missing",   "sev": "High",     "label": "No notice period defined",                  "check": "Is the notice period for resignation or termination NOT stated?"},
        {"id": "no_equity_terms",              "cat": "missing",   "sev": "Medium",   "label": "No equity or bonus terms",                  "check": "Are equity grants, vesting schedules, or bonus structures mentioned but not clearly defined?"},
        {"id": "no_governing_law",             "cat": "missing",   "sev": "Medium",   "label": "No governing law clause",                   "check": "Is the governing law and jurisdiction NOT stated in the offer letter?"},
        {"id": "no_probation_terms",           "cat": "missing",   "sev": "Medium",   "label": "No probation period terms",                 "check": "Is there NO mention of probation period, its duration, or the rights during probation?"},
    ],

    "freelancer_agreement": [
        {"id": "unlimited_liability",          "cat": "dangerous", "sev": "Critical", "label": "Unlimited liability on freelancer",         "check": "Is the freelancer exposed to unlimited financial liability with no cap on damages?"},
        {"id": "broad_ip_assignment",          "cat": "dangerous", "sev": "High",     "label": "Broad IP assignment including prior work",  "check": "Does the agreement transfer all IP — including prior or independently created work — to the client?"},
        {"id": "unilateral_scope_change",      "cat": "dangerous", "sev": "High",     "label": "Unilateral scope change without extra pay",  "check": "Can the client change or expand the scope of work without paying additional fees?"},
        {"id": "delayed_payment_terms",        "cat": "dangerous", "sev": "High",     "label": "Unreasonably delayed payment terms",        "check": "Are payment terms structured so the freelancer is paid only after very long delays or on conditions they cannot control?"},
        {"id": "non_compete_post_project",     "cat": "dangerous", "sev": "High",     "label": "Post-project non-compete restriction",      "check": "Is the freelancer restricted from working with competitors or in the same industry after the project ends?"},
        {"id": "unilateral_termination",       "cat": "dangerous", "sev": "High",     "label": "Unilateral termination without payment",    "check": "Can the client terminate the contract at any time without paying for work already completed?"},
        {"id": "vague_deliverables",           "cat": "unusual",   "sev": "High",     "label": "Vague or undefined deliverables",           "check": "Are the deliverables, milestones, or acceptance criteria so vague that disputes are likely?"},
        {"id": "no_kill_fee",                  "cat": "missing",   "sev": "High",     "label": "No kill fee or cancellation payment",       "check": "Is there NO kill fee or partial payment clause if the client cancels mid-project?"},
        {"id": "no_revision_limit",            "cat": "missing",   "sev": "Medium",   "label": "No limit on revisions",                     "check": "Is there NO clause limiting the number of revisions or change requests?"},
        {"id": "no_payment_schedule",          "cat": "missing",   "sev": "High",     "label": "No clear payment schedule or milestones",   "check": "Are payment dates, milestone payments, or invoicing procedures NOT defined?"},
        {"id": "no_ip_licence_back",           "cat": "missing",   "sev": "Medium",   "label": "No IP licence back to freelancer",          "check": "Is there NO clause allowing the freelancer to use the work in their portfolio?"},
        {"id": "no_confidentiality",           "cat": "missing",   "sev": "Medium",   "label": "No confidentiality clause",                 "check": "Is there NO confidentiality obligation protecting either party's information?"},
        {"id": "no_dispute_resolution",        "cat": "missing",   "sev": "Medium",   "label": "No dispute resolution process",             "check": "Is there NO mediation or escalation path defined before going to litigation?"},
        {"id": "no_governing_law",             "cat": "missing",   "sev": "Medium",   "label": "No governing law clause",                   "check": "Is the governing law and jurisdiction NOT specified?"},
    ],

    "service_agreement": [
        {"id": "unlimited_liability",          "cat": "dangerous", "sev": "Critical", "label": "Unlimited liability",                       "check": "Is there no cap on financial damages owed by the service provider?"},
        {"id": "one_sided_indemnity",          "cat": "dangerous", "sev": "Critical", "label": "One-sided indemnification",                 "check": "Does only the service provider bear all legal costs and losses with no mutual indemnification?"},
        {"id": "unilateral_amendment",         "cat": "dangerous", "sev": "High",     "label": "Unilateral amendment of terms",             "check": "Can the client change service terms, scope, or pricing without the provider's consent?"},
        {"id": "unilateral_price_change",      "cat": "dangerous", "sev": "High",     "label": "Unilateral price change",                   "check": "Can the client reduce payment or the provider increase fees without agreement?"},
        {"id": "auto_renewal_trap",            "cat": "dangerous", "sev": "High",     "label": "Auto-renewal trap",                         "check": "Does the agreement auto-renew with a very short or buried opt-out window?"},
        {"id": "assignment_no_consent",        "cat": "dangerous", "sev": "High",     "label": "Assignment without consent",                "check": "Can the agreement be transferred to a third party without consent?"},
        {"id": "vague_sla",                    "cat": "unusual",   "sev": "High",     "label": "Vague or unmeasurable SLA",                 "check": "Are the service level targets (uptime, response time, etc.) defined so vaguely they cannot be enforced?"},
        {"id": "no_sla_penalty",               "cat": "missing",   "sev": "High",     "label": "No SLA breach penalty or credit",           "check": "Is there NO penalty, service credit, or remedy if the provider misses SLA targets?"},
        {"id": "no_liability_cap",             "cat": "missing",   "sev": "Critical", "label": "No limitation of liability",                "check": "Is there NO clause capping maximum financial exposure for either party?"},
        {"id": "no_termination_clause",        "cat": "missing",   "sev": "High",     "label": "No termination for convenience clause",     "check": "Is there NO clause allowing either party to exit the agreement with reasonable notice?"},
        {"id": "no_force_majeure",             "cat": "missing",   "sev": "Medium",   "label": "No force majeure clause",                   "check": "Is there NO protection for events outside a party's control?"},
        {"id": "no_payment_terms",             "cat": "missing",   "sev": "High",     "label": "No clear payment terms",                    "check": "Are payment due dates, late payment interest, or currency NOT specified?"},
        {"id": "no_data_protection",           "cat": "missing",   "sev": "High",     "label": "No data protection clause",                 "check": "Is there NO clause governing how customer data is handled, stored, or protected?"},
        {"id": "no_dispute_resolution",        "cat": "missing",   "sev": "Medium",   "label": "No dispute resolution process",             "check": "Is there NO mediation step before litigation?"},
        {"id": "no_governing_law",             "cat": "missing",   "sev": "Medium",   "label": "No governing law clause",                   "check": "Is the governing law and jurisdiction NOT stated?"},
    ],

    "consulting_agreement": [
        {"id": "unlimited_liability",          "cat": "dangerous", "sev": "Critical", "label": "Unlimited liability on consultant",         "check": "Is the consultant exposed to unlimited financial liability with no cap on damages?"},
        {"id": "broad_ip_assignment",          "cat": "dangerous", "sev": "High",     "label": "Broad IP assignment",                       "check": "Does the agreement transfer all IP — including pre-existing knowledge and methodologies — to the client?"},
        {"id": "excessive_non_compete",        "cat": "dangerous", "sev": "High",     "label": "Excessive non-compete clause",              "check": "Is there a non-compete clause that prevents the consultant from working in their industry after the engagement ends?"},
        {"id": "non_solicitation",             "cat": "dangerous", "sev": "Medium",   "label": "Non-solicitation clause",                   "check": "Does the contract prevent the consultant from working with the client's employees or competitors after the engagement?"},
        {"id": "unilateral_scope_change",      "cat": "dangerous", "sev": "High",     "label": "Unilateral scope change",                   "check": "Can the client expand the scope of consulting work without paying additional fees?"},
        {"id": "no_cap_on_liability",          "cat": "missing",   "sev": "Critical", "label": "No limitation of liability",                "check": "Is there NO clause capping the consultant's maximum financial exposure?"},
        {"id": "no_payment_terms",             "cat": "missing",   "sev": "High",     "label": "No clear payment or retainer terms",        "check": "Are payment amounts, retainer fees, invoicing procedures, or due dates NOT defined?"},
        {"id": "no_ip_ownership",              "cat": "missing",   "sev": "High",     "label": "No IP ownership clause",                    "check": "Is it unclear who owns the deliverables, reports, or work product created during the engagement?"},
        {"id": "no_termination_clause",        "cat": "missing",   "sev": "High",     "label": "No termination clause",                     "check": "Is there NO clause allowing either party to end the engagement with reasonable notice?"},
        {"id": "no_confidentiality",           "cat": "missing",   "sev": "High",     "label": "No confidentiality clause",                 "check": "Is there NO confidentiality obligation protecting the client's sensitive business information?"},
        {"id": "no_dispute_resolution",        "cat": "missing",   "sev": "Medium",   "label": "No dispute resolution process",             "check": "Is there NO mediation or escalation step before litigation?"},
        {"id": "no_governing_law",             "cat": "missing",   "sev": "Medium",   "label": "No governing law clause",                   "check": "Is the governing law and jurisdiction NOT specified?"},
        {"id": "deemed_acceptance",            "cat": "unusual",   "sev": "Medium",   "label": "Deemed acceptance of deliverables",         "check": "Are deliverables automatically deemed accepted after a period of silence even if not reviewed?"},
        {"id": "no_expense_policy",            "cat": "missing",   "sev": "Medium",   "label": "No expense reimbursement policy",           "check": "Is there NO clause covering travel, accommodation, or out-of-pocket expense reimbursement?"},
    ],

    "lease_agreement": [
        {"id": "rent_escalation_no_cap",       "cat": "dangerous", "sev": "High",     "label": "Rent escalation without cap",               "check": "Can the landlord increase rent by any amount without a defined cap or notice requirement?"},
        {"id": "landlord_entry_no_notice",     "cat": "dangerous", "sev": "High",     "label": "Landlord entry without notice",             "check": "Can the landlord enter the property without prior notice or with very short notice?"},
        {"id": "excessive_early_exit_penalty", "cat": "dangerous", "sev": "High",     "label": "Excessive early termination penalty",       "check": "Are there penalties for early exit that are disproportionately high compared to remaining rent?"},
        {"id": "broad_tenant_liability",       "cat": "dangerous", "sev": "High",     "label": "Broad tenant liability for all damages",    "check": "Is the tenant liable for all damages including normal wear and tear?"},
        {"id": "no_subletting_allowed",        "cat": "dangerous", "sev": "Medium",   "label": "No subletting or assignment allowed",       "check": "Is subletting or assigning the lease completely prohibited with no exceptions?"},
        {"id": "unilateral_lease_changes",     "cat": "dangerous", "sev": "High",     "label": "Landlord can change terms unilaterally",    "check": "Can the landlord modify lease terms — such as rules or fees — without the tenant's consent?"},
        {"id": "security_deposit_unclear",     "cat": "unusual",   "sev": "High",     "label": "Unclear security deposit terms",            "check": "Are the conditions for refunding or deducting from the security deposit vague or one-sided?"},
        {"id": "maintenance_all_on_tenant",    "cat": "unusual",   "sev": "Medium",   "label": "All maintenance responsibility on tenant",  "check": "Is the tenant responsible for all maintenance and repairs, including structural or major repairs?"},
        {"id": "renewal_at_landlord_discretion","cat":"unusual",   "sev": "Medium",   "label": "Renewal at landlord's discretion only",     "check": "Can only the landlord decide whether to renew the lease, with no tenant right to renew?"},
        {"id": "no_notice_to_vacate",          "cat": "missing",   "sev": "High",     "label": "No notice period to vacate",                "check": "Is the required notice period for the tenant to vacate NOT defined?"},
        {"id": "no_dispute_resolution",        "cat": "missing",   "sev": "Medium",   "label": "No dispute resolution process",             "check": "Is there NO process to resolve disputes between landlord and tenant before going to court?"},
        {"id": "no_governing_law",             "cat": "missing",   "sev": "Medium",   "label": "No governing law clause",                   "check": "Is the governing law and jurisdiction NOT stated in the lease?"},
        {"id": "no_renewal_terms",             "cat": "missing",   "sev": "Medium",   "label": "No renewal terms defined",                  "check": "Are renewal conditions, rent on renewal, or notice period for renewal NOT specified?"},
    ],

    "employment_contract": [
        {"id": "at_will_no_notice",            "cat": "dangerous", "sev": "High",     "label": "At-will termination without notice",        "check": "Can the employer terminate employment immediately without any notice or reason?"},
        {"id": "broad_ip_assignment",          "cat": "dangerous", "sev": "Critical", "label": "Broad IP assignment including personal work","check": "Does the contract assign all inventions — including personal projects done outside work — to the employer?"},
        {"id": "clawback_provisions",          "cat": "dangerous", "sev": "High",     "label": "Clawback of bonus or equity",               "check": "Can the employer claw back bonuses or equity already paid or vested?"},
        {"id": "mandatory_arbitration",        "cat": "dangerous", "sev": "High",     "label": "Mandatory arbitration",                     "check": "Is the employee forced into arbitration and prevented from suing in court?"},
        {"id": "excessive_non_compete",        "cat": "dangerous", "sev": "High",     "label": "Excessive non-compete clause",              "check": "Is there a non-compete with unreasonably wide scope, geography, or duration?"},
        {"id": "non_solicitation",             "cat": "dangerous", "sev": "Medium",   "label": "Non-solicitation clause",                   "check": "Does it prevent the employee from hiring former colleagues or working with former clients after leaving?"},
        {"id": "forced_relocation",            "cat": "unusual",   "sev": "Medium",   "label": "Forced relocation without consent",         "check": "Can the employer relocate the employee to any location without consent or additional compensation?"},
        {"id": "vague_performance_metrics",    "cat": "unusual",   "sev": "Medium",   "label": "Vague termination criteria",                "check": "Can the employer terminate based on subjective or undefined performance criteria?"},
        {"id": "no_salary_defined",            "cat": "missing",   "sev": "Critical", "label": "Salary or CTC not clearly defined",         "check": "Is the salary, total CTC, or compensation structure NOT clearly stated?"},
        {"id": "no_notice_period",             "cat": "missing",   "sev": "High",     "label": "No notice period defined",                  "check": "Is the notice period for resignation or termination NOT stated?"},
        {"id": "no_leave_policy",              "cat": "missing",   "sev": "Medium",   "label": "No leave or holiday policy",                "check": "Are annual leave, sick leave, or public holiday entitlements NOT mentioned?"},
        {"id": "no_equity_terms",              "cat": "missing",   "sev": "Medium",   "label": "Equity or bonus terms not defined",         "check": "Are equity grants, vesting schedules, or bonus structures vaguely defined or missing?"},
        {"id": "no_governing_law",             "cat": "missing",   "sev": "Medium",   "label": "No governing law clause",                   "check": "Is the governing law and jurisdiction NOT stated?"},
        {"id": "no_probation_terms",           "cat": "missing",   "sev": "Medium",   "label": "No probation period terms",                 "check": "Is the probation period, its duration, or the terms during probation NOT mentioned?"},
    ],

    # ── Fallback — used for any document type not in the list above ──────
    "general": [
        {"id": "unlimited_liability",          "cat": "dangerous", "sev": "Critical", "label": "Unlimited liability",                       "check": "Is there no cap on financial damages owed by one party?"},
        {"id": "one_sided_indemnity",          "cat": "dangerous", "sev": "Critical", "label": "One-sided indemnification",                 "check": "Does only one party bear all legal costs and losses?"},
        {"id": "unilateral_amendment",         "cat": "dangerous", "sev": "High",     "label": "Unilateral amendment",                      "check": "Can one party change terms without the other's consent?"},
        {"id": "mandatory_arbitration",        "cat": "dangerous", "sev": "High",     "label": "Mandatory arbitration",                     "check": "Is there a clause that waives the right to go to court and forces private arbitration?"},
        {"id": "auto_renewal_trap",            "cat": "dangerous", "sev": "High",     "label": "Auto-renewal trap",                         "check": "Does the contract auto-renew with a very short or buried opt-out window?"},
        {"id": "broad_ip_assignment",          "cat": "dangerous", "sev": "High",     "label": "Broad IP assignment",                       "check": "Does the contract transfer all IP — including prior work — to the other party?"},
        {"id": "personal_guarantee",           "cat": "dangerous", "sev": "Critical", "label": "Personal guarantee",                        "check": "Is an individual personally liable for a company's obligations?"},
        {"id": "assignment_no_consent",        "cat": "dangerous", "sev": "High",     "label": "Assignment without consent",                "check": "Can the contract be transferred to a third party without agreement?"},
        {"id": "disproportionate_penalty",     "cat": "dangerous", "sev": "High",     "label": "Disproportionate penalty",                  "check": "Are financial penalties far in excess of the actual value of the breach?"},
        {"id": "deemed_acceptance",            "cat": "unusual",   "sev": "Medium",   "label": "Deemed acceptance",                         "check": "Is silence or inaction treated as acceptance of changes?"},
        {"id": "indefinite_term",              "cat": "unusual",   "sev": "Medium",   "label": "Indefinite contract term",                  "check": "Does the contract have no fixed end date and run indefinitely?"},
        {"id": "vague_scope",                  "cat": "unusual",   "sev": "Medium",   "label": "Vague scope of work",                       "check": "Are obligations or deliverables defined so vaguely that disputes are likely?"},
        {"id": "no_liability_cap",             "cat": "missing",   "sev": "Critical", "label": "No limitation of liability",                "check": "Is there NO clause capping the maximum financial exposure of either party?"},
        {"id": "no_termination_clause",        "cat": "missing",   "sev": "High",     "label": "No termination clause",                     "check": "Is there NO way to exit the agreement?"},
        {"id": "no_force_majeure",             "cat": "missing",   "sev": "Medium",   "label": "No force majeure clause",                   "check": "Is there NO protection for events outside a party's control?"},
        {"id": "no_dispute_resolution",        "cat": "missing",   "sev": "Medium",   "label": "No dispute resolution process",             "check": "Is there NO mediation step before litigation?"},
        {"id": "no_payment_terms",             "cat": "missing",   "sev": "High",     "label": "No clear payment terms",                    "check": "Are payment due dates, amounts, or currency NOT specified?"},
        {"id": "no_governing_law",             "cat": "missing",   "sev": "Medium",   "label": "No governing law clause",                   "check": "Is the governing law and jurisdiction NOT stated?"},
        {"id": "no_confidentiality",           "cat": "missing",   "sev": "High",     "label": "No confidentiality clause",                 "check": "Is there NO confidentiality obligation between the parties?"},
    ],
}

# Map intent.py slugs → checklist keys
_SLUG_TO_CHECKLIST: dict[str, str] = {
    "nda":                   "nda",
    "job_offer":             "job_offer",
    "freelancer_agreement":  "freelancer_agreement",
    "service_agreement":     "service_agreement",
    "consulting_agreement":  "consulting_agreement",
    "lease_agreement":       "lease_agreement",
    "employment_contract":   "employment_contract",
}

# ---------------------------------------------------------------------------
# Step 1 — detect document type from text
# Uses _run_inference_json for clean, stable output
# "json" appears in the system prompt → response_format=json_object is safe
# ---------------------------------------------------------------------------

_DETECT_SYSTEM = (
    "You are a legal document classifier. "
    "Read the contract and return a JSON object identifying the document type. "
    "Supported slugs: nda, job_offer, freelancer_agreement, service_agreement, "
    "consulting_agreement, lease_agreement, employment_contract, general. "
    "Return ONLY this JSON: {\"doc_type\": \"<slug>\"}"
)

_DETECT_EXAMPLES = """
Examples:
- Non-Disclosure Agreement between two companies → nda
- Job offer letter for a software engineer → job_offer
- Freelance design project contract → freelancer_agreement
- Cloud hosting SLA or vendor agreement → service_agreement
- Strategy advisor retainer contract → consulting_agreement
- Apartment rent or lease deed → lease_agreement
- Full-time employment contract → employment_contract
- Any other document → general
"""

async def _detect_doc_type(text: str) -> str:
    messages = [
        {"role": "system", "content": _DETECT_SYSTEM},
        {"role": "user",   "content": f"{_DETECT_EXAMPLES}\n\nDocument (first 1500 chars):\n---\n{text[:1500]}\n---"},
    ]
    try:
        raw, _, _ = await _run_inference_json(messages, "red_flag_detect_type")
        parsed    = extract_json_raw(raw)
        slug      = (parsed.get("doc_type") or "general").lower().strip()
        checklist_key = _SLUG_TO_CHECKLIST.get(slug, "general")
        logger.info(f"[red_flag_scanner] detected doc_type='{slug}' → checklist='{checklist_key}'")
        return checklist_key
    except Exception as e:
        logger.warning(f"[red_flag_scanner] type detection failed ({e}) — using general")
        return "general"


# ---------------------------------------------------------------------------
# Step 2 — evaluate checklist for the detected document type
# ---------------------------------------------------------------------------

_EVAL_SYSTEM = (
    "You are a contract risk lawyer. "
    "For each checklist item evaluate the contract and answer with exactly one status: "
    "present (found in document), absent (not found), or not_applicable (irrelevant to this document type). "
    "You MUST evaluate every single item. "
    "Return ONLY the JSON object — no markdown, no backticks, no explanation."
)


def _build_eval_messages(checklist: list[dict], text: str) -> list[dict]:
    checklist_lines = "\n".join(
        f'{i+1}. id="{item["id"]}" — {item["check"]}'
        for i, item in enumerate(checklist)
    )

    user = f"""Evaluate every item below against the contract.

For each item return:
  "id"             : item id exactly as given
  "status"         : "present" | "absent" | "not_applicable"
  "clause_excerpt" : exact quote from the contract (max 150 chars) if status=present, else null
  "why_dangerous"  : one sentence on the risk to the signer if status=present, else null
  "recommendation" : one concrete fix or negotiation step if status=present, else null

Return ONLY this JSON:
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

Checklist ({len(checklist)} items — evaluate ALL {len(checklist)}):
{checklist_lines}

Contract:
---
{text[:80_000]}
---"""

    return [
        {"role": "system", "content": _EVAL_SYSTEM},
        {"role": "user",   "content": user},
    ]


# ---------------------------------------------------------------------------
# Step 3 — build flags from AI results (severity from hardcoded map)
# ---------------------------------------------------------------------------

_CAT_ICON = {"dangerous": "🔴", "unusual": "🟠", "missing": "🟡"}
_SORT_ORDER = {"Critical": 0, "High": 1, "Medium": 2}


def _build_flags(results: list, checklist: list[dict]) -> list[dict]:
    sev_map   = {item["id"]: item["sev"] for item in checklist}
    cat_map   = {item["id"]: item["cat"] for item in checklist}
    label_map = {item["id"]: item["label"] for item in checklist}

    flags = []
    for item in results:
        if not isinstance(item, dict) or item.get("status") != "present":
            continue

        item_id  = item.get("id", "")
        category = cat_map.get(item_id, "unusual")
        severity = sev_map.get(item_id, "Medium")
        label    = label_map.get(item_id, item_id.replace("_", " ").title())

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
    Step 1 — detect document type (nda / job_offer / freelancer_agreement /
             service_agreement / consulting_agreement / lease_agreement /
             employment_contract / general)
    Step 2 — evaluate the type-specific checklist (present / absent / not_applicable)
    Step 3 — build flags; severity assigned from hardcoded map (not the model)

    Consistent count guarantee:
    - Fixed checklist per document type → same questions every call
    - AI only answers present/absent/not_applicable → no free-form list
    - Severity comes from hardcoded map → deterministic
    - _run_inference_json → response_format=json_object → clean JSON every time
    """
    # Step 1 — detect type
    doc_type  = await _detect_doc_type(text)
    checklist = _CHECKLISTS[doc_type]

    logger.info(f"[red_flag_scanner] doc_type={doc_type} | checklist={len(checklist)} items")

    # Step 2 — evaluate checklist
    messages = _build_eval_messages(checklist, text)
    raw, in_tok, out_tok = await _run_inference_json(messages, "red_flag_eval")
    logger.info(f"[red_flag_scanner] tokens in={in_tok} out={out_tok}")

    parsed  = extract_json_raw(raw)
    results = parsed.get("results", []) if isinstance(parsed, dict) else []

    # Retry if fewer than half the items came back
    if len(results) < len(checklist) // 2:
        logger.warning(
            f"[red_flag_scanner] Only {len(results)}/{len(checklist)} items — retrying"
        )
        raw, _, _ = await _run_inference_json(messages, "red_flag_eval_retry")
        parsed    = extract_json_raw(raw)
        results   = parsed.get("results", []) if isinstance(parsed, dict) else []

    if not results:
        logger.error("[red_flag_scanner] No results — returning safe default")
        return {
            "document_type":     doc_type,
            "detected_flags":    [],
            "overall_risk_level": "Low",
            "summary":           "Red flag scan could not be completed for this document.",
        }

    # Step 3 — build flags
    flags = _build_flags(results, checklist)

    severities = {f["severity"] for f in flags}
    if "Critical" in severities:  overall = "Critical"
    elif "High"   in severities:  overall = "High"
    elif "Medium" in severities:  overall = "Medium"
    else:                         overall = "Low"

    dangerous = sum(1 for f in flags if f["category"] == "dangerous")
    unusual   = sum(1 for f in flags if f["category"] == "unusual")
    missing   = sum(1 for f in flags if f["category"] == "missing")

    logger.info(
        f"[red_flag_scanner] Done — {len(flags)}/{len(checklist)} flags present | "
        f"dangerous={dangerous} unusual={unusual} missing={missing} | risk={overall}"
    )

    return {
        "document_type":     doc_type,
        "detected_flags":    flags,
        "overall_risk_level": overall,
        "summary": (
            f"Scanned {len(checklist)} checklist items for {doc_type.replace('_', ' ')}. "
            f"Found {len(flags)} flag(s): "
            f"{dangerous} dangerous, {unusual} unusual, {missing} missing protection(s). "
            f"Overall risk: {overall}."
        ),
    }
