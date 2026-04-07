from fastapi import HTTPException, UploadFile
import time
import uuid
import os
import logging

from llm_model.ai_model import run_llm
from utils.pdf_utils import load_pdf, get_page_count, all_pages_blank
from utils.json_utils import extract_json_raw as extract_json_from_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# CLASSIFICATION
# ==============================
async def classify_document(text: str):
    prompt = f"""
Classify the document into EXACTLY ONE of the following categories:

contract
resume
invoice
report
other

Definitions:
- contract → legal agreements (NDA, Service Agreement, Lease, Terms & Conditions)
- resume → CV, job profile, candidate details
- invoice → billing documents, receipts, payment summaries
- report → business reports, analysis documents, research papers
- other → anything else (stories, books, random text)

Rules:
- Return ONLY one word from the list above
- Do NOT add explanation
- Do NOT return multiple categories
- If unsure, return "other"

Examples:
"Non Disclosure Agreement between two parties..." → contract
"John Doe, Software Engineer, Skills: Python..." → resume
"Invoice No: 12345, Total: $500..." → invoice
"Quarterly sales analysis shows growth..." → report
"Once upon a time..." → other

Text:
\"\"\"{text[:1500]}\"\"\"
"""

    result = await run_llm(text[:1500], prompt)

    # Safety cleanup
    result = result.lower().strip()

    valid_labels = {"contract", "resume", "invoice", "report", "other"}

    # Handle messy LLM outputs like "contract\n" or "contract document"
    for label in valid_labels:
        if label in result:
            return label

    return "other"

# ==============================
# CONTRACT HANDLER
# ==============================
async def handle_contract(text: str):
    prompt = f"""
Extract key contract clauses from the given text and return in STRICT JSON format.

Fields:
- Agreement Type (Service Agreement, NDA, Lease, etc.)
- Effective Date
- Contract Term
- Renewal Conditions
- Governing Law
- Payment Terms
- Termination Notice Period

Rules:
- Return ONLY valid JSON (no explanation, no extra text)
- Use EXACT field names as listed above
- Do NOT rename keys
- Do NOT add extra fields
- If any field is missing, return "Not Found"
- Keep values short and precise

Example Output:
{{
  "Agreement Type": "Service Agreement",
  "Effective Date": "Jan 1, 2026",
  "Contract Term": "3 years",
  "Renewal Conditions": "Auto-renew yearly",
  "Governing Law": "Alberta",
  "Payment Terms": "Net 30",
  "Termination Notice Period": "60 days"
}}

Text:
\"\"\"{text}\"\"\"
"""

    result = await run_llm(text, prompt)
    print("Raw LLM Output:", result)  # Debugging: see the unprocessed output
    parsed_data = extract_json_from_text(result)
    return {
        "status": "success",
        "document_type": "contract",
        "data": parsed_data
    }


# ==============================
# RESUME HANDLER
# ==============================
async def handle_resume(text: str):
    prompt = f"""
Extract structured resume information from the given text and return in STRICT JSON format.

Fields:
- Name
- Skills (list)
- Experience (list of roles or summary)
- Education (list)
- Missing Sections (list of missing sections like Skills, Experience, Education, Summary, Projects)

Rules:
- Return ONLY valid JSON (no explanation, no extra text)
- Use EXACT field names as listed above
- Do NOT rename keys
- Do NOT add extra fields
- If a field is not found, return "Not Found"
- Lists should be arrays ([])
- Keep values concise

Example Output:
{{
  "Name": "John Doe",
  "Skills": ["Python", "Machine Learning", "FastAPI"],
  "Experience": ["Software Engineer at ABC Corp (2022-2025)"],
  "Education": ["B.Tech in Computer Science"],
  "Missing Sections": ["Projects", "Summary"]
}}

Text:
\"\"\"{text}\"\"\"
"""

    result = await run_llm(text, prompt)
    print("Raw LLM Output:", result)  # Debugging: see the unprocessed output
    parsed_data = extract_json_from_text(result)
    return {
        "status": "success",
        "document_type": "resume",
        "data": parsed_data
    }


# ==============================
# INVOICE HANDLER
# ==============================
async def handle_invoice(text: str):
    prompt = f"""
Extract structured invoice information from the given text and return in STRICT JSON format.

Fields:
- Invoice Number
- Date
- Vendor Name
- Total Amount
- Tax
- Line Items (list of objects with: Description, Quantity, Unit Price, Amount)

Rules:
- Return ONLY valid JSON (no explanation, no extra text)
- Use EXACT field names as listed above
- Do NOT rename keys
- Do NOT add extra fields
- If any field is missing, return "Not Found"
- Line Items must be an array of objects
- If line items are not clearly available, return []

Example Output:
{{
  "Invoice Number": "INV-1023",
  "Date": "2026-01-15",
  "Vendor Name": "ABC Pvt Ltd",
  "Total Amount": "1500 USD",
  "Tax": "150 USD",
  "Line Items": [
    {{
      "Description": "Software License",
      "Quantity": "1",
      "Unit Price": "1000 USD",
      "Amount": "1000 USD"
    }},
    {{
      "Description": "Support Fee",
      "Quantity": "1",
      "Unit Price": "500 USD",
      "Amount": "500 USD"
    }}
  ]
}}

Text:
\"\"\"{text}\"\"\"
"""

    result = await run_llm(text, prompt)
    print("Raw LLM Output:", result)  # Debugging: see the unprocessed output
    parsed_data = extract_json_from_text(result)
    return {
        "status": "success",
        "document_type": "invoice",
        "data": parsed_data
    }

# ==============================
# ROUTER MAP
# ==============================
DOCUMENT_HANDLERS = {
    "contract": handle_contract,
    "resume": handle_resume,
    "invoice": handle_invoice
}

async def extract_text_from_upload(
    file: UploadFile,
    *,
    endpoint: str = "",
    max_pages: int | None = None,
) -> tuple[str, int, int, str, float, str]:
    logger.info(f"Received file: {file.filename} for endpoint: {endpoint}")
    request_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    total_pages = get_page_count(file_path)
    pages_to_read = total_pages if max_pages is None else min(total_pages, max_pages)

    try:
        pages = load_pdf(file_path, max_pages=pages_to_read)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if all_pages_blank(pages):
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    text = "\n\n".join(p.page_content for p in pages)

    return text, pages_to_read, total_pages, request_id, t_start, file_path





    # ONLY showing UPDATED / IMPORTANT PARTS (drop-in replace)

# ---------------------------
# ADD THIS HELPER
# ---------------------------

def _apply_defaults(document_type: str, fields: dict) -> dict:
    DEFAULTS = {
        "nda": {
            "governing_law": "India",
            "duration_years": "2",
            "purpose": "Business evaluation"
        }
    }

    defaults = DEFAULTS.get(document_type, {})
    for k, v in defaults.items():
        if fields.get(k) == "Not Specified":
            fields[k] = v
    return fields


# ---------------------------
# IMPROVED EXTRACTION
# ---------------------------

async def _extract_fields(document_type: str, user_query: str) -> dict:
    fields = await _extract_fields_once(document_type, user_query)

    # Retry if weak
    weak_count = sum(1 for v in fields.values() if v == "Not Specified")
    if weak_count >= len(fields) * 0.7:
        logger.warning("[doc_gen] weak extraction → retrying")
        fields = await _extract_fields_retry(document_type, user_query)

    return fields


async def _extract_fields_once(document_type: str, user_query: str) -> dict:
    schema = _SCHEMAS[document_type]
    doc_name = SUPPORTED_DOCUMENT_TYPES[document_type]

    system = f"""
Extract structured data for {doc_name}.
Return ONLY JSON.

Fields:
{schema["required"] + schema["optional"]}

Rules:
- Use exact keys
- Missing → "Not Specified"
"""

    client = _get_client()
    kwargs = _get_model_kwargs(1000)
    kwargs["messages"] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]
    kwargs["response_format"] = {"type": "json_object"}

    res = await client.chat.completions.create(**kwargs)
    raw = res.choices[0].message.content or "{}"

    return _extract_fields_from_json(raw)


async def _extract_fields_retry(document_type: str, user_query: str) -> dict:
    doc_name = SUPPORTED_DOCUMENT_TYPES[document_type]

    system = f"""
You are a legal expert.

Extract ALL possible structured fields for {doc_name}.
Infer intelligently if possible.

Return JSON only.
"""

    client = _get_client()
    kwargs = _get_model_kwargs(1000)
    kwargs["messages"] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]
    kwargs["response_format"] = {"type": "json_object"}

    res = await client.chat.completions.create(**kwargs)
    raw = res.choices[0].message.content or "{}"

    return _extract_fields_from_json(raw)


# ---------------------------
# DOCUMENT GENERATION FIX
# ---------------------------

async def _generate_document_text(document_type, fields, user_query):

    for attempt in range(2):  # retry logic
        doc_name = SUPPORTED_DOCUMENT_TYPES[document_type]

        system = f"""
Generate a complete {doc_name}.

Use legal structure.
Use placeholders if needed.
Plain text only.
"""

        client = _get_client()
        kwargs = _get_model_kwargs(max_tokens=2000)  # FIXED

        kwargs["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(fields)},
        ]

        res = await client.chat.completions.create(**kwargs)

        text = res.choices[0].message.content or ""

        if text.strip():
            return text, res.usage.prompt_tokens, res.usage.completion_tokens

        logger.warning(f"[doc_gen] empty response retry {attempt+1}")

    raise ValueError("LLM failed to generate document")


# ---------------------------
# MAIN FUNCTION FIX
# ---------------------------

async def generate_document(document_type: str, user_query: str):

    if len(user_query.split()) < 5:
        raise ValueError("User query too short")

    fields = await _extract_fields(document_type, user_query)

    logger.info(f"[doc_gen] fields:\n{json.dumps(fields, indent=2)}")

    fields = _apply_defaults(document_type, fields)

    missing = _get_missing_fields(document_type, fields)

    document_text, in_tok, out_tok = await _generate_document_text(
        document_type, fields, user_query
    )

    if not document_text.strip():
        raise ValueError("Empty document generated")

    pdf_bytes = _render_pdf(document_text, SUPPORTED_DOCUMENT_TYPES[document_type])

    return {
        "status": "missing_fields" if missing else "success",
        "document_type": document_type,
        "document_name": SUPPORTED_DOCUMENT_TYPES[document_type],
        "fields": fields,
        "missing_fields": missing,
        "document": document_text,
        "pdf_bytes": pdf_bytes,
        "word_count": len(document_text.split()),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }
