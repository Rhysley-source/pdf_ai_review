import base64
import io
import json
import os
import re
import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .prompt_templates import (
    QUERY_ANALYSIS_PROMPT,
    DOCUMENT_GENERATION_V2_PROMPT,
    SECTION_TEMPLATES,
    build_generation_context,
    REGENERATE_PROMPT,
)
from auth import verify_api_key

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MODEL      = os.environ.get("MODEL_NAME", "gpt-5-nano")
_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
_CLIENT     = AsyncOpenAI(api_key=_API_KEY)

# Models that do not support the temperature parameter
_FIXED_TEMPERATURE_MODELS = {
    "gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini",
    "o1", "o1-mini", "o3-mini", "o3",
}

# ---------------------------------------------------------------------------
# Storage — absolute path anchored to this file's directory
# ---------------------------------------------------------------------------

_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "html_db.json")


def get_storage() -> dict:
    if not os.path.exists(_DB_FILE):
        return {}
    try:
        with open(_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def update_storage(doc_id: str, html_content: str) -> None:
    db = get_storage()
    db[doc_id] = html_content
    with open(_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4)


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class DocumentGenerationRequest(BaseModel):
    document_id: str | None = None
    user_prompt: str


class DocumentRegenerationRequest(BaseModel):
    document_id: str
    modification_query: str


class HtmlToPdfRequest(BaseModel):
    document_id: str | None = None   # fetch HTML from html_db.json
    html: str | None = None          # or pass raw HTML directly


class Base64TextRequest(BaseModel):
    doc_id: str
    base64_data: str


# ---------------------------------------------------------------------------
# LLM caller — dedicated for HTML generation with higher output token limit
# ---------------------------------------------------------------------------

async def _call_llm(system_prompt: str, user_message: str) -> str:
    model = os.environ.get("MODEL_NAME", _MODEL)

    kwargs: dict = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }

    if model not in _FIXED_TEMPERATURE_MODELS:
        kwargs["temperature"] = 0.3

    try:
        response = await _CLIENT.chat.completions.create(**kwargs)
        choice   = response.choices[0]
        content  = choice.message.content or ""
        finish   = choice.finish_reason
        logger.info(
            f"[html-gen] in={response.usage.prompt_tokens} "
            f"out={response.usage.completion_tokens} "
            f"finish={finish} model={model}"
        )
        if finish == "length":
            logger.warning("[html-gen] Response was cut off by the model's context limit (finish_reason=length)")
        return content
    except Exception as e:
        logger.exception(f"[html-gen] OpenAI call failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Input validation — reject gibberish / meaningless prompts
# ---------------------------------------------------------------------------

_FORMAT_HINT = (
    "Please provide a clear document generation request. Examples:\n"
    "  • \"Generate a service agreement between Company A and Company B\"\n"
    "  • \"Create an employment offer letter for a software engineer role\"\n"
    "  • \"Draft a non-disclosure agreement between two parties\"\n"
    "  • \"Make an invoice for web development services worth $2,000\"\n"
    "  • \"Write a residential lease agreement for a 1-year term\""
)


def _is_gibberish(text: str) -> bool:
    """
    Returns True when the prompt looks like gibberish or has no meaningful content.

    Checks:
    1. Too short after stripping whitespace.
    2. Alphabetic characters make up less than 50 % of the text
       (catches strings like "123 @@@ !!!" or random symbols).
    3. Fewer than 2 words that are at least 3 alphabetic characters long
       (catches single-char spam like "a b c d e" or keyboard mashing).
    """
    stripped = text.strip()

    # Too short to mean anything
    if len(stripped) < 5:
        return True

    # Low alphabetic ratio — mostly numbers / symbols / spaces
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 0 and (alpha_count / len(stripped)) < 0.50:
        return True

    # Not enough real words (3+ consecutive alpha chars)
    real_words = re.findall(r"[A-Za-z]{3,}", stripped)
    if len(real_words) < 2:
        return True

    return False


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------

def _clean_html(raw: str) -> str:
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```html", "").replace("```", "").strip()
    start = cleaned.find("<html")
    end   = cleaned.rfind("</html>")
    if start != -1 and end != -1:
        # Full, well-formed response
        cleaned = cleaned[start : end + 7]
    elif start != -1:
        # Truncated — model stopped before </html>; keep everything from <html
        cleaned = cleaned[start:]
    return cleaned


# ---------------------------------------------------------------------------
# Pipeline helpers — Step 1, Step 2, Step 3
# ---------------------------------------------------------------------------

def _parse_analysis_json(raw: str) -> dict:
    """
    Robustly parses the Step 1 LLM response into a dict.
    Strips markdown fences if present, validates required keys,
    and clamps doc_type to known SECTION_TEMPLATES keys.
    """
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Step 1 returned non-JSON: {raw[:200]}") from exc

    if not isinstance(parsed, dict) or "doc_type" not in parsed:
        raise ValueError(f"Step 1 JSON missing 'doc_type': {raw[:200]}")

    # Clamp doc_type to known types
    if parsed["doc_type"] not in SECTION_TEMPLATES:
        parsed["doc_type"] = "other"

    # Ensure fields is always a dict
    if not isinstance(parsed.get("fields"), dict):
        parsed["fields"] = {}

    return parsed


async def _analyze_query(user_prompt: str) -> dict:
    """Step 1 — LLM call to detect document type and extract field values.
    Raises HTTP 422 immediately if the query is not a document generation request.
    """
    logger.info("[doc-gen] Step 1: analysing query...")
    raw = await _call_llm(QUERY_ANALYSIS_PROMPT.template, user_prompt)
    logger.info(f"[doc-gen] Step 1 raw output: {raw[:300]}")

    analysis = _parse_analysis_json(raw)

    if not analysis.get("is_document_request", True):
        raise HTTPException(
            status_code=422,
            detail=(
                "Your query is not related to document generation.\n\n"
                "Please provide a request to create a specific document. Examples:\n"
                "  • \"Generate a service agreement between Company A and Company B\"\n"
                "  • \"Create an invoice for web development services worth $2,000\"\n"
                "  • \"Draft an employment offer letter for a software engineer\"\n"
                "  • \"Make a non-disclosure agreement between two parties\"\n"
                "  • \"Write a residential lease agreement for a 1-year term\""
            ),
        )

    return analysis


def _build_template_context(analysis: dict) -> dict:
    """Step 2 — pure Python template merge. No LLM involved."""
    doc_type         = analysis["doc_type"]
    section_template = SECTION_TEMPLATES.get(doc_type, SECTION_TEMPLATES["other"])
    context          = build_generation_context(analysis, section_template)
    logger.info(f"[doc-gen] Step 2: built template context for doc_type='{doc_type}'")
    return context


async def _generate_html_from_context(context: dict, user_prompt: str) -> str:
    """Step 3 — final LLM call using the enriched type-specific context."""
    logger.info(f"[doc-gen] Step 3: generating HTML for '{context['doc_label']}'...")
    system_prompt = DOCUMENT_GENERATION_V2_PROMPT.format(
        **context,
        user_request=user_prompt,
    )
    return await _call_llm(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate-html", response_class=HTMLResponse)
async def generate_document_html(
    request: DocumentGenerationRequest,
    _: None = Depends(verify_api_key),
):
    """
    3-step pipeline:
      Step 1 (LLM)    — detect document type + extract field values from user_prompt
      Step 2 (Python) — select type-specific section template, merge extracted fields
      Step 3 (LLM)    — generate final HTML using the enriched context
    """
    if _is_gibberish(request.user_prompt):
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid request: your prompt appears to be gibberish or unclear.\n\n"
                + _FORMAT_HINT
            ),
        )

    doc_id = request.document_id or str(uuid.uuid4())

    try:
        # Step 1: query analysis
        analysis = await _analyze_query(request.user_prompt)

        # Step 2: template building (pure Python)
        context = _build_template_context(analysis)

        # Step 3: final HTML generation
        raw_html     = await _generate_html_from_context(context, request.user_prompt)
        cleaned_html = _clean_html(raw_html)

        if not cleaned_html.strip():
            raise HTTPException(
                status_code=500,
                detail=f"AI generated empty HTML. Raw output: {raw_html[:500]}"
            )

        update_storage(doc_id, cleaned_html)
        return HTMLResponse(content=cleaned_html, headers={"X-Document-Id": doc_id})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/regenerate-html", response_class=HTMLResponse)
async def regenerate_document_html(
    request: DocumentRegenerationRequest,
    _: None = Depends(verify_api_key),
):
    """
    Looks up HTML by document_id, applies user modifications,
    updates storage, and returns the modified HTML.
    """
    if _is_gibberish(request.modification_query):
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid request: your modification query appears to be gibberish or unclear.\n\n"
                "Please describe what you want to change. Examples:\n"
                "  • \"Change the vendor name to Acme Corp\"\n"
                "  • \"Update the due date to 30th April 2025\"\n"
                "  • \"Add a 10% GST row to the totals table\"\n"
                "  • \"Replace the client address with 123 Main Street, New York\""
            ),
        )

    db            = get_storage()
    existing_html = db.get(request.document_id)

    if not existing_html:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with ID '{request.document_id}'. Generate it first."
        )

    system_prompt = REGENERATE_PROMPT.format(
        existing_html=existing_html,
        modification_query=request.modification_query
    )

    try:
        raw_html     = await _call_llm(system_prompt, request.modification_query)
        cleaned_html = _clean_html(raw_html)

        update_storage(request.document_id, cleaned_html)
        return HTMLResponse(content=cleaned_html)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {str(e)}")


@router.get("/get-html/{document_id}", response_class=HTMLResponse)
async def get_document_html(
    document_id: str,
    _: None = Depends(verify_api_key),
):
    """
    Fetches previously generated HTML from html_db.json by document_id.
    """
    db = get_storage()
    html = db.get(document_id)
    if not html:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with ID '{document_id}'. Generate it first via /generate-html."
        )
    return HTMLResponse(content=html)


@router.post("/html-to-pdf")
async def html_to_pdf(
    request: HtmlToPdfRequest,
    _: None = Depends(verify_api_key),
):
    """
    Converts HTML to a PDF file.
    Provide either:
      - document_id  → fetches HTML from html_db.json
      - html         → uses the raw HTML string directly
    Returns a downloadable PDF.
    """
    try:
        from weasyprint import HTML as WeasyprintHTML
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="weasyprint is not installed. Run: pip install weasyprint"
        )

    if request.document_id:
        db = get_storage()
        html_content = db.get(request.document_id)
        if not html_content:
            raise HTTPException(
                status_code=404,
                detail=f"No document found with ID '{request.document_id}'."
            )
        filename = f"{request.document_id}.pdf"
    elif request.html:
        html_content = request.html
        filename = "document.pdf"
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'document_id' or 'html' in the request body."
        )

    a4_css = """
        @page {
            size: A4 portrait;
            margin: 15mm 15mm 15mm 15mm;
        }
        html, body {
            width: 100%;
            margin: 0;
            padding: 0;
            font-size: 11pt;
            font-family: Arial, sans-serif;
            color: #000;
            background: #fff;
            -webkit-print-color-adjust: exact;
        }
        * {
            box-sizing: border-box;
            max-width: 100%;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            word-wrap: break-word;
        }
        img {
            max-width: 100%;
            height: auto;
        }
    """

    try:
        from weasyprint import CSS
        pdf_bytes = WeasyprintHTML(string=html_content).write_pdf(
            stylesheets=[CSS(string=a4_css)]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/base64-text")
async def base64_to_text(
    request: Base64TextRequest,
    _: None = Depends(verify_api_key),
):
    """
    Decodes base64_data to plain text and updates html_db.json
    under the given doc_id (same store used by /generate-html).
    """
    try:
        text_content = base64.b64decode(request.base64_data).decode("utf-8")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64_data — could not decode to UTF-8 text."
        )

    update_storage(request.doc_id, text_content)
    logger.info(f"[base64-text] updated doc_id='{request.doc_id}' ({len(text_content)} chars)")

    return {"doc_id": request.doc_id, "char_count": len(text_content)}
