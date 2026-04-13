# document_generation/prompt_templates.py


class SimulatedPromptTemplate:
    def __init__(self, template: str, input_variables: list[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        formatted_template = self.template
        for var in self.input_variables:
            if var in kwargs:
                formatted_template = formatted_template.replace(f"{{{var}}}", str(kwargs[var]))
        return formatted_template


# ---------------------------------------------------------------------------
# STEP 1 — Query Analysis Prompt
# Detects document type and extracts key field values from the user request.
# Returns a structured JSON object for use in Step 2 template building.
# ---------------------------------------------------------------------------

QUERY_ANALYSIS_PROMPT = SimulatedPromptTemplate(
    template="""You are a document analysis assistant. Read the user's request and return ONLY a valid JSON object — no markdown, no explanation, no backticks.

JSON keys you must return:
  "is_document_request" : true if the user is asking to generate, create, or draft any kind of
                          document. false for anything else (questions, calculations, greetings,
                          general knowledge, weather, coding help, etc.).
  "doc_type"  : classify into EXACTLY one of:
                invoice, contract, employment, nda, lease, resume,
                certificate, report, proposal, purchase_order, letter, other
                (set to "other" when is_document_request is false)
  "doc_label" : short human-readable name for the specific document (max 6 words).
                Examples: "Tax Invoice", "Service Agreement", "Job Offer Letter",
                "Non-Disclosure Agreement", "Software Engineer Resume"
                (set to "" when is_document_request is false)
  "fields"    : a flat JSON object of ALL details extracted from the user's request.
                Use snake_case keys. Set value to null for any detail not mentioned.
                Extract: names, company names, dates, amounts, roles, addresses,
                durations, quantities, descriptions, and any other document-specific data.
                (set to {} when is_document_request is false)

Mapping guidance:
  invoice        → billing / payment document between vendor and client
  contract       → service, vendor, freelancer, consulting, or general agreement
  employment     → job offer letter, appointment letter, employment contract
  nda            → non-disclosure or confidentiality agreement
  lease          → property lease, rent agreement, tenancy, leave and licence
  resume         → CV, curriculum vitae, resume, candidate profile
  certificate    → certificate of completion, appreciation, training, achievement
  report         → analytical, financial, status, or summary report
  proposal       → business proposal, project proposal, RFP response
  purchase_order → purchase order, PO, procurement document
  letter         → formal letter, cover letter, recommendation letter, notice
  other          → anything not covered above

Return ONLY the raw JSON. No wrapper text, no markdown fences.

Example — document request:
{
  "is_document_request": true,
  "doc_type": "invoice",
  "doc_label": "Web Development Invoice",
  "fields": {
    "vendor_name": "Acme Corp",
    "client_name": "Beta Ltd",
    "amount": "2000 USD",
    "due_date": null
  }
}

Example — non-document request ("what is the capital of France?"):
{
  "is_document_request": false,
  "doc_type": "other",
  "doc_label": "",
  "fields": {}
}""",
    input_variables=[],
)


# ---------------------------------------------------------------------------
# STEP 2 — Section Templates (pure Python, no LLM)
# Defines the required sections and field labels for each document type.
# Used to build the enriched context passed to the Step 3 generation prompt.
# ---------------------------------------------------------------------------

SECTION_TEMPLATES: dict[str, dict] = {
    "invoice": {
        "required_sections": [
            "Invoice Header (invoice number, invoice date, due date)",
            "Bill From (vendor/seller name, address, contact)",
            "Bill To (client/buyer name, address, contact)",
            "Line Items Table (description, quantity, unit price, amount)",
            "Subtotal, Tax/GST breakdown, Grand Total",
            "Payment Instructions (bank details, payment method)",
            "Notes / Terms and Conditions",
        ],
        "field_hints": {
            "vendor_name": "Vendor / Seller Name",
            "vendor_address": "Vendor Address",
            "client_name": "Client / Buyer Name",
            "client_address": "Client Address",
            "invoice_number": "Invoice Number",
            "invoice_date": "Invoice Date",
            "due_date": "Due Date",
            "line_items": "Line Items",
            "subtotal": "Subtotal",
            "tax_rate": "Tax Rate",
            "total_amount": "Total Amount",
            "payment_method": "Payment Method / Bank Details",
        },
    },
    "contract": {
        "required_sections": [
            "Parties (full legal names, addresses, effective date)",
            "Recitals / Background",
            "Scope of Work / Services",
            "Term and Renewal",
            "Fees and Payment Terms",
            "Intellectual Property",
            "Confidentiality",
            "Limitation of Liability",
            "Termination",
            "Governing Law and Dispute Resolution",
            "General Provisions (severability, entire agreement, amendments)",
            "Signature Block (both parties, date)",
        ],
        "field_hints": {
            "party_a_name": "Party A Name",
            "party_b_name": "Party B Name",
            "effective_date": "Effective Date",
            "services_description": "Services Description",
            "contract_value": "Contract Value",
            "payment_terms": "Payment Terms",
            "governing_law": "Governing Law / Jurisdiction",
        },
    },
    "employment": {
        "required_sections": [
            "Date and Addressee",
            "Offer of Employment",
            "Job Title and Department",
            "Compensation and Benefits",
            "Start Date and Work Location",
            "Probation Period",
            "Notice Period",
            "Confidentiality and IP Assignment",
            "Code of Conduct",
            "Acceptance Deadline",
            "Signature Block",
        ],
        "field_hints": {
            "employer_name": "Employer / Company Name",
            "candidate_name": "Candidate Full Name",
            "job_title": "Job Title",
            "department": "Department",
            "start_date": "Start Date",
            "compensation": "Annual Compensation / Salary",
            "work_location": "Work Location",
            "probation_period": "Probation Period",
            "notice_period": "Notice Period",
            "reporting_to": "Reporting Manager",
        },
    },
    "nda": {
        "required_sections": [
            "Parties and Recitals",
            "Definitions — Confidential Information",
            "Exclusions from Confidential Information",
            "Obligations of Receiving Party",
            "Permitted Disclosures",
            "Term and Termination",
            "Return or Destruction of Materials",
            "Remedies",
            "Governing Law and Dispute Resolution",
            "Signature Block",
        ],
        "field_hints": {
            "disclosing_party": "Disclosing Party",
            "receiving_party": "Receiving Party",
            "effective_date": "Effective Date",
            "purpose": "Purpose of Disclosure",
            "duration": "NDA Duration",
            "governing_law": "Governing Law / Jurisdiction",
        },
    },
    "lease": {
        "required_sections": [
            "Parties — Landlord and Tenant",
            "Property Description",
            "Lease Term (start date, end date)",
            "Monthly Rent and Due Date",
            "Security Deposit",
            "Utilities and Maintenance Responsibilities",
            "Permitted Use and Restrictions (subletting, pets, alterations)",
            "Termination and Notice Period",
            "Move-out Conditions",
            "Governing Law",
            "Signature Block with Witness Lines",
        ],
        "field_hints": {
            "landlord_name": "Landlord Name",
            "tenant_name": "Tenant Name",
            "property_address": "Property Address",
            "lease_start": "Lease Start Date",
            "lease_end": "Lease End Date",
            "monthly_rent": "Monthly Rent",
            "security_deposit": "Security Deposit",
            "notice_period": "Notice Period to Vacate",
        },
    },
    "resume": {
        "required_sections": [
            "Header (full name, email, phone, location, LinkedIn/portfolio URL)",
            "Professional Summary",
            "Work Experience (reverse chronological — company, title, dates, bullet points)",
            "Education (degree, institution, graduation year)",
            "Skills",
            "Certifications (if applicable)",
            "Projects (if applicable)",
        ],
        "field_hints": {
            "full_name": "Full Name",
            "email": "Email Address",
            "phone": "Phone Number",
            "location": "City / Country",
            "job_title": "Target Job Title",
            "years_experience": "Years of Experience",
            "skills": "Key Skills",
            "education": "Education Details",
        },
    },
    "certificate": {
        "required_sections": [
            "Certificate Title (centered, prominent)",
            "Awarded To (recipient name, large font)",
            "Body Text (achievement / course / event description)",
            "Date of Award",
            "Issuer Name and Title",
            "Signature Line",
        ],
        "field_hints": {
            "recipient_name": "Recipient Name",
            "certificate_type": "Certificate Type",
            "achievement": "Achievement / Course / Event",
            "issue_date": "Date of Award",
            "issuer_name": "Issuing Organization / Person",
            "issuer_title": "Issuer Title / Designation",
        },
    },
    "report": {
        "required_sections": [
            "Report Title and Metadata (date, author, version)",
            "Executive Summary",
            "Introduction / Background",
            "Methodology (if applicable)",
            "Findings / Analysis",
            "Conclusions",
            "Recommendations",
            "Appendices (if applicable)",
        ],
        "field_hints": {
            "report_title": "Report Title",
            "author_name": "Author Name",
            "report_date": "Report Date",
            "subject": "Report Subject / Topic",
            "audience": "Target Audience",
            "period_covered": "Time Period Covered",
        },
    },
    "proposal": {
        "required_sections": [
            "Cover Page (title, submitted by, submitted to, date)",
            "Executive Summary",
            "Problem Statement / Opportunity",
            "Proposed Solution / Approach",
            "Scope of Work",
            "Timeline and Milestones",
            "Pricing / Budget",
            "About Us / Team",
            "Terms and Conditions",
            "Call to Action / Next Steps",
        ],
        "field_hints": {
            "proposing_party": "Proposing Party / Company",
            "client_name": "Client / Prospect Name",
            "proposal_date": "Proposal Date",
            "project_title": "Project Title",
            "total_value": "Total Proposed Value",
            "validity_period": "Proposal Valid Until",
        },
    },
    "purchase_order": {
        "required_sections": [
            "PO Header (PO number, date, buyer details)",
            "Vendor / Supplier Details",
            "Line Items Table (item description, quantity, unit price, total)",
            "Delivery Details (address, required date)",
            "Payment Terms",
            "Special Instructions / Notes",
            "Authorized Signature",
        ],
        "field_hints": {
            "buyer_name": "Buyer / Company Name",
            "buyer_address": "Buyer Address",
            "vendor_name": "Vendor / Supplier Name",
            "vendor_address": "Vendor Address",
            "po_number": "PO Number",
            "po_date": "PO Date",
            "delivery_date": "Required Delivery Date",
            "total_amount": "Total PO Amount",
            "payment_terms": "Payment Terms",
        },
    },
    "letter": {
        "required_sections": [
            "Sender Details and Date",
            "Recipient Name and Address",
            "Subject Line",
            "Salutation",
            "Body (opening paragraph, main content, closing paragraph)",
            "Complimentary Close",
            "Signature Block",
        ],
        "field_hints": {
            "sender_name": "Sender Name",
            "sender_title": "Sender Title / Designation",
            "recipient_name": "Recipient Name",
            "recipient_title": "Recipient Title",
            "letter_date": "Letter Date",
            "subject": "Subject",
        },
    },
    "other": {
        "required_sections": [
            "Document Title",
            "Parties / Participants (if applicable)",
            "Introduction / Purpose",
            "Main Content",
            "Terms and Conditions (if applicable)",
            "Closing / Conclusion",
            "Signature Block (if applicable)",
        ],
        "field_hints": {
            "document_title": "Document Title",
            "primary_party": "Primary Party / Author",
            "date": "Document Date",
        },
    },
}


# ---------------------------------------------------------------------------
# STEP 2 helper — builds the enriched context dict for the Step 3 prompt
# Pure Python, no LLM. Merges extracted fields with the section template.
# ---------------------------------------------------------------------------

def build_generation_context(analysis: dict, section_template: dict) -> dict:
    """
    Merges the Step 1 LLM analysis with the Step 2 section template into
    a context dict ready to be formatted into DOCUMENT_GENERATION_V2_PROMPT.

    Returns:
        {
            "doc_type":          str,
            "doc_label":         str,
            "required_sections": str,   # numbered list
            "extracted_fields":  str,   # label: value block
        }
    """
    import json as _json

    doc_type  = analysis.get("doc_type", "other")
    doc_label = analysis.get("doc_label") or doc_type.replace("_", " ").title()

    # Build numbered section list
    sections      = section_template.get("required_sections", [])
    sections_str  = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sections))

    # Build extracted fields block
    field_hints  = section_template.get("field_hints", {})
    raw_fields   = analysis.get("fields") or {}
    if not isinstance(raw_fields, dict):
        raw_fields = {}

    field_lines: list[str] = []
    for key, value in raw_fields.items():
        label = field_hints.get(key, key.replace("_", " ").title())
        if isinstance(value, list):
            display = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            display = _json.dumps(value)
        elif value is None:
            display = "[Not provided]"
        else:
            display = str(value)
        field_lines.append(f"  {label}: {display}")

    extracted_fields_str = "\n".join(field_lines) if field_lines else "  [No specific details extracted]"

    return {
        "doc_type":          doc_type,
        "doc_label":         doc_label,
        "required_sections": sections_str,
        "extracted_fields":  extracted_fields_str,
    }


# ---------------------------------------------------------------------------
# STEP 2 — Template Build Prompt (LLM)
# Takes the Step 1 analysis and produces a tailored document blueprint:
#   - sections refined/expanded for this specific document instance
#   - each section pre-filled with extracted field values
#   - tone and layout notes for the generator
# Returns a compact JSON object consumed by Step 3.
# ---------------------------------------------------------------------------

TEMPLATE_BUILD_PROMPT = SimulatedPromptTemplate(
    template="""You are a document blueprint specialist. Your job is to produce a detailed, fully pre-filled section plan for a "{doc_label}" document so that a document generator can produce the complete document without any guesswork.

Document Type  : {doc_label} ({doc_type})

Fields extracted from the request:
{extracted_fields}

Standard sections for this document type:
{required_sections}

Full user request (use this to extract every detail, clause, or requirement the user mentioned):
{user_request}

YOUR TASK — return ONLY a valid JSON object with exactly these keys:

{{
  "document_title": "<exact title to display at the top of the document, e.g. 'RENT AGREEMENT', 'TAX INVOICE', 'SERVICE AGREEMENT'>",
  "sections": [
    {{
      "title": "<section heading>",
      "content_hint": "<complete, detailed description of exactly what to write in this section. Embed ALL known values directly — names, amounts, dates, addresses, durations. Mark every missing required value as [Field Name]. Be specific enough that no further instructions are needed.>",
      "missing_fields": ["<name of each required field not found in the request>"]
    }}
  ],
  "tone": "<formal | professional | friendly | technical — pick the best fit for this document type>",
  "layout_notes": "<specific layout instruction — e.g. 'Two-column header table with landlord left, tenant right. Numbered clauses for all terms. Signature table at the bottom with two columns.'>"
}}

Rules:
1. Include EVERY section needed for a complete, legally sound {doc_label} — do not omit any standard section.
2. Also include any EXTRA sections the user specifically requested (e.g. penalty clauses, witness sections, special terms).
3. content_hint must be fully pre-filled with actual values — e.g. write "Monthly Rent: ₹18,000 (Rupees Eighteen Thousand)" not "monthly rent goes here".
4. For every field that is missing from the request, add it to missing_fields and use a placeholder whose text is the ACTUAL field label in brackets — e.g. [Email Address], [Phone Number], [Job Title], [Company Name]. NEVER use [Client Name], [Field Name], or any other generic label for a field that has its own name.
5. layout_notes must describe the exact table/structure needed (not just "standard layout").
6. Return ONLY raw JSON. No markdown, no backticks, no explanation.""",
    input_variables=["doc_type", "doc_label", "extracted_fields", "required_sections", "user_request"],
)


# ---------------------------------------------------------------------------
# STEP 3 — Final Document Generation Prompt
# Uses the enriched context from Steps 1 & 2 for precise HTML generation.
# ---------------------------------------------------------------------------

DOCUMENT_GENERATION_V2_PROMPT = SimulatedPromptTemplate(
    template="""You are an expert document generator. Produce a single complete HTML document that looks identical when viewed in a browser and when converted to PDF.

Document Type : {doc_label} ({doc_type})
Tone          : {tone}
Layout Notes  : {layout_notes}

Document Blueprint — generate each section in this exact order:
{sections_block}

Original User Request:
{user_request}

━━━ INSTRUCTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Return ONLY the complete HTML, starting with <html> and ending with </html>.
- Include <html>, <head> with ONE embedded <style> block, and <body>.
- Add contenteditable="true" to the outermost content div inside <body>.
- Render every blueprint section in order using its content_hint as the source.
- Wherever the blueprint shows a [Placeholder], render it as a styled span using the EXACT placeholder text from the blueprint — e.g. if the blueprint says [Email Address] write <span style="color:#cc0000;">[Email Address]</span>, if it says [Phone Number] write <span style="color:#cc0000;">[Phone Number]</span>. NEVER replace every placeholder with [Client Name] — each placeholder must show its own specific field name.
- Do NOT include markdown backticks, explanations, or any text outside the HTML.

━━━ DESIGN RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Flat design only — no box-shadow, text-shadow, drop-shadow, or 3D effects.
- No gradients, no background-image, no border-radius.
- Body background: plain white (#ffffff). No colored section backgrounds.
- Font: Arial, sans-serif. Body text 11pt. Section headings 12pt–14pt.
- Content must fill the full page width — no narrow centered containers.
- Section dividers: <hr style="border:none; border-top:1px solid #cccccc; margin:10px 0;"> only.
- The result must look like a clean printed document, not a web widget.

━━━ PDF-SAFE RULES (WeasyPrint — every rule is mandatory) ━━━━━━━━
LAYOUT
- NEVER use display:flex or display:grid.
- For ALL side-by-side or multi-column content use <table> — never floats.
  Two-column pattern:
  <table width="100%" style="border-collapse:collapse; margin-bottom:10px;"><tr>
    <td style="width:50%; vertical-align:top; padding:0;">LEFT CONTENT</td>
    <td style="width:50%; vertical-align:top; padding:0; text-align:right;">RIGHT CONTENT</td>
  </tr></table>
- NEVER use position:fixed, position:sticky, or position:absolute.
- NEVER use margin:auto — center text with text-align:center on the element itself.
- NEVER use negative margin values.

HEADINGS
- Every h1–h6 MUST have ALL of these as inline style attributes (not in <style>):
  font-size, font-weight, text-align, margin-top, margin-bottom, color.
  Example: <h2 style="font-size:13pt; font-weight:bold; text-align:left; margin-top:14px; margin-bottom:6px; color:#000000;">

SIZING & SPACING
- font-size: use pt or px only — never em, rem, %, or vw.
- Set line-height:1.5 on the body style and on every <p> element.
- NEVER set a fixed height on any element — use padding-top and padding-bottom for spacing.
- NEVER use min-height — use padding instead.
- NEVER use overflow:hidden or overflow:scroll.

TABLES
- Every data table MUST have: style="width:100%; border-collapse:collapse; table-layout:fixed;"
- Every <th> and <td> MUST have: explicit padding (e.g. padding:6px 8px) and border (e.g. border:1px solid #cccccc).
- Add style="word-wrap:break-word; overflow-wrap:break-word;" to <td> cells that may contain long text, URLs, or amounts.
- Add style="page-break-inside:avoid;" to any table that must not be split across PDF pages.

PAGE BREAKS
- Add style="page-break-inside:avoid;" to signature blocks and sections that must stay together.
- Do NOT add page-break rules to normal paragraph sections — let content flow naturally.

━━━ REQUIRED HTML SKELETON ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start every document from this base — fill in <style> and body content:
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body { font-family: Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #000000; background: #ffffff; margin: 0; padding: 24px; }
    p    { margin: 0 0 8px 0; line-height: 1.5; }
    /* Add document-specific class styles here */
  </style>
</head>
<body>
  <div contenteditable="true">
    <!-- ALL document content here -->
  </div>
</body>
</html>""",
    input_variables=["doc_type", "doc_label", "tone", "layout_notes", "sections_block", "user_request"],
)


# ---------------------------------------------------------------------------
# Regeneration prompt — modifies an existing HTML document.
# Unchanged from original.
# ---------------------------------------------------------------------------

REGENERATE_PROMPT = SimulatedPromptTemplate(
    template="""You are an expert HTML editor. Apply the user's modification to the existing HTML document while keeping the design and layout identical everywhere that was not changed.

RULES:
1. Return ONLY the complete updated HTML — starting with <html>, ending with </html>.
2. Keep contenteditable="true" on the outermost content div inside <body>.
3. No markdown backticks, explanations, or text outside the HTML.
4. Preserve all original <style> blocks and CSS — only change what the user requested.
5. Keep the design flat and print-ready: no box-shadow, text-shadow, gradients, border-radius, or colored section backgrounds.
6. Do NOT use display:flex or display:grid. For any side-by-side layout use <table> only.
7. Every h1–h6 must keep all inline style attributes (font-size, font-weight, text-align, margin-top, margin-bottom, color) set directly on the element — do not move them to <style>.
8. Do NOT set fixed height or min-height on any container — use padding only.
9. Do NOT use overflow:hidden, overflow:scroll, negative margins, or position:fixed/absolute/sticky.
10. Every data table must have style="width:100%; border-collapse:collapse; table-layout:fixed;" and all <td>/<th> must have explicit padding and border.
11. Add style="word-wrap:break-word; overflow-wrap:break-word;" to <td> cells containing long text, URLs, or amounts.
12. Add style="page-break-inside:avoid;" to signature blocks and any section that must not split across PDF pages.

Existing HTML:
{existing_html}

User Modification Request:
{modification_query}
""",
    input_variables=["existing_html", "modification_query"],
)


# ---------------------------------------------------------------------------
# Regeneration intent check prompt
# Classifies whether a modification query is trying to switch to a completely
# different document type vs. modifying the existing document.
# Returns a single JSON with "intent": "modify" | "new_document" and "reason".
# ---------------------------------------------------------------------------

REGENERATION_INTENT_PROMPT = SimulatedPromptTemplate(
    template="""You are a strict intent classifier for document modification requests.

Current document type: {current_doc_type}

User's modification query: "{modification_query}"

Decide the user's intent:

- "modify"       → the user wants to change, improve, update, or restyle the EXISTING {current_doc_type}
                   (e.g. change a value, improve layout, add a row, rename a field, make it look better)

- "new_document" → the user wants to generate a COMPLETELY DIFFERENT type of document
                   (e.g. convert an invoice into a contract, turn a resume into an NDA,
                   make this a purchase order instead, generate a lease from this invoice)

Return ONLY this JSON — no markdown, no explanation:
{{"intent": "modify" | "new_document", "reason": "<one short sentence>"}}""",
    input_variables=["current_doc_type", "modification_query"],
)


# ---------------------------------------------------------------------------
# Intent check prompt — kept for backward compatibility (currently unused).
# ---------------------------------------------------------------------------

INTENT_CHECK_PROMPT = SimulatedPromptTemplate(
    template="""You are a strict query classifier.

Decide whether the user's query is asking to generate, create, or draft any kind of document
(e.g. invoice, contract, resume, report, certificate, letter, agreement, proposal, form, etc.).

Reply with ONLY one word — exactly "YES" or "NO". No explanation, no punctuation.

User query: {user_request}""",
    input_variables=["user_request"],
)
