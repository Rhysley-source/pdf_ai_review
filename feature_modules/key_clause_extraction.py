import asyncio
import json
import logging
import re
import time
import uuid
import os
from functools import partial

from fastapi import HTTPException, UploadFile

from llm_model.ai_model import run_llm
from utils.pdf_utils import load_pdf, get_page_count, all_pages_blank
from utils.json_utils import extract_json_raw as extract_json_from_text

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Chunking — same windows as risk_detection for consistency
# ---------------------------------------------------------------------------

_CHUNK_SIZE    = 10_000
_CHUNK_OVERLAP = 500


def _chunk_text(text: str) -> list[str]:
    if len(text) <= _CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + _CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    logger.info(f"[key_clause] Document split into {len(chunks)} chunk(s)")
    return chunks


# ---------------------------------------------------------------------------
# Step 1 — Document Classification
# Uses first 4000 chars; returns (slug, human-readable label).
# Aligned with risk_detection slugs so both routes classify consistently.
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


async def classify_document(text: str) -> str:
    """Public wrapper — returns only the doc-type slug (used by the comparison endpoint)."""
    slug, _label = await _classify_document(text)
    return slug


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


# ---------------------------------------------------------------------------
# Step 2 — Key Clause Extraction (chunked, parallel)
# ---------------------------------------------------------------------------

async def _extract_clauses_chunk(chunk: str, doc_label: str, chunk_label: str) -> list[dict]:
    """
    Asks the LLM to extract key clauses from one chunk.
    Returns a list of clause dicts: {clause_name, excerpt, significance}.
    """
    system_prompt = f"""You are a legal document analyst specialising in {doc_label} documents.

CRITICAL: Your response MUST be a single valid JSON object. No text before or after. No markdown. No code fences. No explanation. Start your response with {{ and end with }}. Any response that is not pure JSON will be rejected.

Extract ALL key clauses present in this section of the document.

A key clause is any provision, term, condition, or section that is important for understanding:
- Rights and obligations of the parties
- Financial terms, payment conditions, or amounts
- Time periods, deadlines, notice periods, or renewal terms
- Restrictions, limitations, permissions, or prohibitions
- Legal protections, liability, or risks
- Confidentiality, IP, or data obligations

For each key clause found, return these three fields:
  "clause_name"  — a short, specific name (e.g. "Payment Terms", "Termination Notice", "Non-Compete Restriction")
  "excerpt"      — the exact relevant text from the document, or a faithful paraphrase if the clause is very long
  "significance" — one sentence explaining why this clause is important

REQUIRED OUTPUT FORMAT — return exactly this JSON structure, nothing else:
{{
  "key_clauses": [
    {{
      "clause_name": "<name>",
      "excerpt": "<text from document>",
      "significance": "<why it matters>"
    }}
  ]
}}

STRICT RULES:
- Output MUST begin with {{ and end with }}
- All string values MUST use double quotes — never single quotes
- No trailing commas after the last item in any array or object
- Escape any double quotes inside string values as \\"
- Escape any backslashes inside string values as \\\\
- Do NOT use newlines inside string values — use a space instead
- Only include clauses that are ACTUALLY present in this section
- Be specific to "{doc_label}" — not generic observations
- If this section contains no key clauses, return {{"key_clauses": []}}"""

    logger.info(f"[key_clause] Extracting clauses — {chunk_label}")
    raw    = await run_llm(chunk, system_prompt, max_output_tokens=32000)
    result = extract_json_from_text(raw)

    clauses = result.get("key_clauses", [])
    if not isinstance(clauses, list):
        return []

    # Normalise each clause item
    cleaned = []
    for item in clauses:
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
    return cleaned


def _merge_clauses(lists: list[list[dict]]) -> list[dict]:
    """
    Merges clause lists from multiple chunks, deduplicating by normalised clause_name.
    When the same clause appears in multiple chunks, the version with the longer
    excerpt is kept (more complete quote).
    """
    seen: dict[str, dict] = {}
    for clause_list in lists:
        if not isinstance(clause_list, list):
            continue
        for clause in clause_list:
            if not isinstance(clause, dict):
                continue
            key = re.sub(r"[^a-z0-9]", "", clause.get("clause_name", "").lower())
            if not key:
                continue
            if key not in seen:
                seen[key] = clause
            else:
                # Keep the version with the more complete excerpt
                if len(clause.get("excerpt", "")) > len(seen[key].get("excerpt", "")):
                    seen[key]["excerpt"] = clause["excerpt"]
                if not seen[key].get("significance") and clause.get("significance"):
                    seen[key]["significance"] = clause["significance"]
    return list(seen.values())


# ---------------------------------------------------------------------------
# Step 3 — Document Summary (first 4000 chars, fast)
# ---------------------------------------------------------------------------

async def _generate_summary(text: str, doc_label: str) -> str:
    system_prompt = f"""You are reviewing a {doc_label}.
Write a concise 2-3 sentence summary of what this document is, who the parties are (if any),
and what its main purpose or key terms are.
Return ONLY the plain text summary — no JSON, no markdown, no headings."""
    return await run_llm(text[:4000], system_prompt, max_output_tokens=512)


# ---------------------------------------------------------------------------
# Public classify helper — returns just the slug (used by compare-documents)
# ---------------------------------------------------------------------------

async def classify_document(text: str) -> str:
    slug, _ = await _classify_document(text)
    return slug


# ---------------------------------------------------------------------------
# Public entry point — replaces all old per-type handlers
# ---------------------------------------------------------------------------

async def extract_key_clauses(text: str) -> dict:
    """
    Full key clause extraction pipeline:

    Step 1 — classify document type (first 4000 chars)
    Step 2 — extract key clauses from full document (chunked, parallel LLM calls)
    Step 3 — generate document summary (parallel with Step 2)
    Step 4 — merge + deduplicate clauses across chunks

    All document types are supported. No more "unsupported" responses.
    """
    # Step 1: classify
    doc_type, doc_label = await _classify_document(text)
    logger.info(f"[key_clause] Classified as: {doc_type} ('{doc_label}')")

    chunks = _chunk_text(text)

    # Step 2+3: clause extraction (all chunks) + summary — all in parallel
    clause_tasks  = [
        _extract_clauses_chunk(chunk, doc_label, f"chunk {i+1}/{len(chunks)}")
        for i, chunk in enumerate(chunks)
    ]
    summary_task  = _generate_summary(text, doc_label)

    *chunk_results, summary = await asyncio.gather(*clause_tasks, summary_task)

    # Step 4: merge
    key_clauses = _merge_clauses(chunk_results)

    logger.info(
        f"[key_clause] Done — {len(key_clauses)} unique clause(s) "
        f"from {len(chunks)} chunk(s)"
    )

    return {
        "status":        "success",
        "document_type": doc_label,
        "total_clauses": len(key_clauses),
        "summary":       summary.strip(),
        "key_clauses":   key_clauses,
    }


# ---------------------------------------------------------------------------
# Shared PDF ingestion helper (used by /key-clause-extraction and /detect-risks)
# OCR now runs in thread pool — event loop is never blocked.
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
        # Non-blocking: OCR runs in thread pool
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
