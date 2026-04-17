import asyncio
import logging
import os
import time
import uuid
from functools import partial

from fastapi import HTTPException, UploadFile

from llm_model.ai_model import run_llm, run_llm_mini
from utils.pdf_utils import load_pdf, get_page_count, all_pages_blank
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Document classification — used by /compare-documents and /key-clause-extraction
# ---------------------------------------------------------------------------

_KNOWN_SLUGS = {
    "contract", "employment", "nda", "lease",
    "invoice", "resume", "report", "other",
}

_SLUG_LABELS = {
    "contract":   "Contract / Legal Agreement",
    "employment": "Employment Agreement",
    "nda":        "Non-Disclosure Agreement",
    "lease":      "Lease Agreement",
    "invoice":    "Invoice / Billing Document",
    "resume":     "Resume / CV",
    "report":     "Report / Analysis",
    "other":      "General Document",
}

_CLASSIFY_PROMPT = """Analyse the document and return a JSON object with exactly two fields:

"slug"  — classify into EXACTLY ONE of:
          contract, employment, nda, lease, invoice, resume, report, other

"label" — the specific document type as a short human-readable name (2-5 words)

Slug definitions:
- contract   → service agreements, vendor agreements, terms & conditions, MOU, partnership deeds
- employment → offer letters, employment agreements, appointment letters, HR documents
- nda        → non-disclosure agreements, confidentiality agreements
- lease      → rental agreements, property leases, tenancy agreements
- invoice    → billing documents, receipts, payment summaries, purchase orders
- resume     → CV, job profiles, candidate profiles
- report     → analytical, financial, status, or summary reports
- other      → anything not covered above

Label: be specific (e.g. "Service Agreement", "Tax Invoice", "Job Offer Letter").
Never return "Other Document" or "Unknown Document" as a label.

Return ONLY this JSON — no markdown, no explanation:
{"slug": "<slug>", "label": "<label>"}"""


async def _classify_document(text: str) -> tuple[str, str]:
    raw    = await run_llm(text[:4000], _CLASSIFY_PROMPT)
    parsed = extract_json_from_text(raw)
    slug   = (parsed.get("slug") or "").lower().strip()
    label  = (parsed.get("label") or "").strip()

    if slug not in _KNOWN_SLUGS:
        for known in _KNOWN_SLUGS:
            if known in raw.lower():
                slug = known
                break
        else:
            slug = "other"

    if not label or label.lower() in ("other document", "unknown document", "unknown", "other"):
        label = _SLUG_LABELS[slug]

    return slug, label


async def classify_document(text: str) -> str:
    """Public wrapper — returns only the doc-type slug (used by compare-documents)."""
    slug, _ = await _classify_document(text)
    return slug


# ---------------------------------------------------------------------------
# Key clause extraction — single LLM call
# ---------------------------------------------------------------------------

_MAX_SINGLE_CALL_CHARS = 300_000

_SINGLE_CALL_SYSTEM = """You are a document analyst.

Analyze the document and return a single JSON object with EXACTLY this structure:

{
  "document_type": "<slug: contract|employment|nda|lease|invoice|resume|report|other>",
  "document_label": "<specific document name — max 5 words>",
  "key_clauses": [
    {
      "clause_name": "<clause or section name — max 5 words>",
      "excerpt": "<key text from document — max 30 words>",
      "significance": "<why this matters — max 20 words>"
    }
  ]
}

First identify the document type, then extract key clauses relevant to that type:

- contract:    payment terms, liability clauses, termination conditions, IP ownership, dispute resolution,
               indemnification, confidentiality, governing law, force majeure, warranties
- employment:  job title/role, salary and compensation, benefits, probation period, notice period,
               non-compete / non-solicitation, leave policy, working hours, termination conditions
- nda:         parties involved, definition of confidential information, duration, permitted disclosures,
               exclusions from confidentiality, breach consequences, jurisdiction
- lease:       rent amount and due date, lease duration, security deposit, maintenance responsibilities,
               renewal / termination terms, pet / subletting policy, late fees
- invoice:     line items and descriptions, unit prices, quantities, subtotal, tax, total amount due,
               payment due date, payment method, late payment penalties, billing parties
- resume:      professional summary, core skills and technologies, work experience (roles and achievements),
               education and qualifications, certifications and licenses, notable projects or accomplishments
- report:      key findings, main conclusions, critical metrics or data points, recommendations,
               methodology, data sources, risks or issues identified, action items
- other:       main topics covered, key decisions or outcomes, important figures or dates,
               parties or stakeholders involved, notable terms or conditions, action items

Rules:
- Extract ALL relevant clauses or sections actually present in the document
- Use real text from the document — do not fabricate or infer missing content
- If a section type is not present in the document, skip it
- Return ONLY valid JSON — no markdown, no explanation"""


async def extract_key_clauses(text: str) -> dict:
    """Single LLM call: classify + summarize + extract all key clauses in one shot."""
    document = text[:_MAX_SINGLE_CALL_CHARS]

    logger.info(f"[key_clause] Single-call extraction — {len(document):,} chars")
    raw    = await run_llm_mini(document, _SINGLE_CALL_SYSTEM, max_output_tokens=16000)
    result = extract_json_from_text(raw)

    if not result:
        logger.warning("[key_clause] JSON parse failed — returning empty result")
        result = {
            "document_type":  "other",
            "document_label": "General Document",
            "summary":        "",
            "key_clauses":    [],
        }

    key_clauses = result.get("key_clauses", [])
    if not isinstance(key_clauses, list):
        key_clauses = []

    cleaned = []
    for item in key_clauses:
        if not isinstance(item, dict):
            continue
        name = str(item.get("clause_name") or item.get("name") or "").strip()
        if not name:
            continue
        cleaned.append({
            "clause_name":  name,
            "excerpt":      str(item.get("excerpt") or item.get("text") or item.get("quote") or ""),
            "significance": str(item.get("significance") or item.get("importance") or item.get("reason") or ""),
        })

    doc_label = result.get("document_label") or result.get("document_type") or "General Document"
    logger.info(f"[key_clause] Done — {len(cleaned)} clause(s)")

    return {
        "status":        "success",
        "document_type": doc_label,
        "total_clauses": len(cleaned),
        "key_clauses":   cleaned,
    }


# ---------------------------------------------------------------------------
# Shared PDF ingestion helper (used by /key-clause-extraction and /detect-risks)
# OCR runs in thread pool — event loop is never blocked.
# ---------------------------------------------------------------------------

async def extract_text_from_upload(
    file: UploadFile,
    *,
    endpoint: str = "",
    max_pages: int | None = None,
) -> tuple[str, int, int, str, float, str]:
    logger.info(f"Received file: {file.filename} for endpoint: {endpoint}")
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    total_pages   = get_page_count(file_path)
    pages_to_read = total_pages if max_pages is None else min(total_pages, max_pages)

    try:
        loop  = asyncio.get_running_loop()
        pages = await loop.run_in_executor(
            None, partial(load_pdf, file_path, pages_to_read)
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if all_pages_blank(pages):
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    text = "\n\n".join(p.page_content for p in pages)
    return text, pages_to_read, total_pages, request_id, t_start, file_path
