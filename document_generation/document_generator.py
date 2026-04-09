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
    TEMPLATE_BUILD_PROMPT,
    DOCUMENT_GENERATION_V2_PROMPT,
    REGENERATION_INTENT_PROMPT,
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


_MODIFICATION_KEYWORDS = {
    # direct action words
    "change", "update", "replace", "edit", "modify", "fix", "correct",
    "add", "remove", "delete", "insert", "rename", "set", "adjust",
    "rewrite", "move", "swap", "convert", "format", "increase", "decrease",
    "append", "clear", "shift", "put", "turn",
    # improvement / style words
    "improve", "better", "enhance", "refine", "redesign", "beautify",
    "nicer", "cleaner", "professional", "prettier", "modernize", "upgrade",
    "simplify", "bold", "resize", "align", "restyle", "revamp",
    # phrase fragments (checked as substrings)
    "make it", "make the", "use a", "use different", "look better",
    "look more", "more professional", "more formal", "more clean",
}

# Only phrases that clearly mean "create a brand new document"
_NEW_GENERATION_PHRASES = {
    "generate a", "generate an", "generate new",
    "create a", "create an", "create new",
    "draft a", "draft an",
    "write a new", "write an new",
    "build a", "build an",
    "produce a", "produce an",
    "give me a new", "give me an new",
    "new document", "new invoice", "new contract", "new resume",
}


def _is_modification_query(text: str) -> bool:
    """
    Returns True if the text looks like a modification/improvement request
    on an existing document.
    Returns False only if it clearly asks for a brand new document.
    """
    lower = text.lower()

    # Reject only if it explicitly asks for a brand new document
    for phrase in _NEW_GENERATION_PHRASES:
        if phrase in lower:
            return False

    # Accept if any modification/improvement keyword is present
    words = set(re.findall(r"[a-z]+", lower))
    if words & _MODIFICATION_KEYWORDS:
        return True

    # Also check multi-word phrases as substrings
    for phrase in _MODIFICATION_KEYWORDS:
        if " " in phrase and phrase in lower:
            return True

    return False


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


def _parse_blueprint_json(raw: str, analysis: dict) -> dict:
    """
    Parses the Step 2 LLM blueprint response.
    Falls back to the static Python template if the LLM returns malformed JSON.
    Returns a context dict ready for DOCUMENT_GENERATION_V2_PROMPT.
    """
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("[doc-gen] Step 2: blueprint parse failed — falling back to static template")
        return _static_template_context(analysis)

    sections = parsed.get("sections", [])
    if not isinstance(sections, list) or not sections:
        logger.warning("[doc-gen] Step 2: empty sections in blueprint — falling back to static template")
        return _static_template_context(analysis)

    # Build the sections_block string for Step 3
    lines = []
    for i, sec in enumerate(sections, 1):
        title        = sec.get("title", f"Section {i}")
        content_hint = sec.get("content_hint", "")
        lines.append(f"{i}. {title}\n   → {content_hint}")
    sections_block = "\n\n".join(lines)

    return {
        "doc_type":      analysis.get("doc_type", "other"),
        "doc_label":     analysis.get("doc_label", "Document"),
        "tone":          parsed.get("tone", "professional"),
        "layout_notes":  parsed.get("layout_notes", "Standard document layout."),
        "sections_block": sections_block,
    }


def _static_template_context(analysis: dict) -> dict:
    """Fallback: builds context from static SECTION_TEMPLATES (no LLM)."""
    doc_type         = analysis.get("doc_type", "other")
    section_template = SECTION_TEMPLATES.get(doc_type, SECTION_TEMPLATES["other"])
    base             = build_generation_context(analysis, section_template)

    # Convert required_sections + extracted_fields into the blueprint format
    sections_block = "\n\n".join(
        f"{line}"
        for line in base["required_sections"].splitlines()
    )
    return {
        "doc_type":       base["doc_type"],
        "doc_label":      base["doc_label"],
        "tone":           "professional",
        "layout_notes":   "Standard document layout.",
        "sections_block": sections_block,
    }


async def _build_template_context(analysis: dict) -> dict:
    """Step 2 — LLM call to build a tailored document blueprint.
    Falls back to the static Python template on parse failure.
    """
    doc_type  = analysis.get("doc_type", "other")
    doc_label = analysis.get("doc_label", "Document")

    # Build extracted_fields string to pass into the prompt
    section_template  = SECTION_TEMPLATES.get(doc_type, SECTION_TEMPLATES["other"])
    base_context      = build_generation_context(analysis, section_template)
    extracted_fields  = base_context["extracted_fields"]

    system_prompt = TEMPLATE_BUILD_PROMPT.format(
        doc_type=doc_type,
        doc_label=doc_label,
        extracted_fields=extracted_fields,
    )

    logger.info(f"[doc-gen] Step 2: building blueprint for '{doc_label}'...")
    raw = await _call_llm(system_prompt, f"Build the document blueprint for: {doc_label}")
    logger.info(f"[doc-gen] Step 2 raw output: {raw[:300]}")

    return _parse_blueprint_json(raw, analysis)


async def _generate_html_from_context(context: dict, user_prompt: str) -> str:
    """Step 3 — final LLM call using the enriched blueprint context."""
    logger.info(f"[doc-gen] Step 3: generating HTML for '{context['doc_label']}'...")
    system_prompt = DOCUMENT_GENERATION_V2_PROMPT.format(
        **context,
        user_request=user_prompt,
    )
    return await _call_llm(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Regeneration intent checker
# ---------------------------------------------------------------------------

def _extract_doc_type_from_html(html: str) -> str:
    """
    Best-effort extraction of the document type from stored HTML.
    Looks for a <title> or the first <h1>/<h2> tag as a readable label.
    Falls back to 'existing document' if nothing is found.
    """
    import re as _re
    title_match = _re.search(r"<title[^>]*>(.*?)</title>", html, _re.IGNORECASE | _re.DOTALL)
    if title_match:
        return title_match.group(1).strip()
    h_match = _re.search(r"<h[12][^>]*>(.*?)</h[12]>", html, _re.IGNORECASE | _re.DOTALL)
    if h_match:
        # Strip inner HTML tags
        return _re.sub(r"<[^>]+>", "", h_match.group(1)).strip()
    return "existing document"


async def _check_regeneration_intent(modification_query: str, existing_html: str) -> None:
    """
    Calls LLM to verify the modification query intends to modify the existing
    document — not generate a completely different document type.
    Raises HTTP 422 if the intent is 'new_document'.
    Silently passes on parse errors (fail open — let the LLM handle it).
    """
    current_doc_type = _extract_doc_type_from_html(existing_html)

    system_prompt = REGENERATION_INTENT_PROMPT.format(
        current_doc_type=current_doc_type,
        modification_query=modification_query,
    )

    try:
        raw = await _call_llm(system_prompt, modification_query)
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        parsed  = json.loads(cleaned)
        intent  = parsed.get("intent", "modify")
        reason  = parsed.get("reason", "")
    except Exception:
        # Fail open — if LLM or parse fails, don't block the user
        logger.warning("[doc-gen] Regeneration intent check failed — skipping")
        return

    if intent == "new_document":
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "wrong_endpoint",
                "message": f"Your request appears to be asking for a new type of document, not a modification of the existing one.",
                "reason":  reason,
                "hint":    "Use /generate-html to create a new document. Use /regenerate-html only to modify the existing document.",
            },
        )


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
        try:
            analysis = await _analyze_query(request.user_prompt)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[doc-gen] Step 1 failed")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "step1_failed",
                    "step": "Query Analysis",
                    "message": "Failed to analyse your request. The AI model did not respond correctly.",
                    "hint": "Try rephrasing your prompt.",
                    "detail": str(e),
                },
            )

        # Step 2: blueprint building (LLM)
        try:
            context = await _build_template_context(analysis)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[doc-gen] Step 2 failed")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "step2_failed",
                    "step": "Blueprint Building",
                    "message": "Failed to build the document blueprint. Falling back was not possible.",
                    "hint": "Try again — this is usually a transient model error.",
                    "detail": str(e),
                },
            )

        # Step 3: final HTML generation
        try:
            raw_html = await _generate_html_from_context(context, request.user_prompt)
        except Exception as e:
            logger.exception("[doc-gen] Step 3 failed")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "step3_failed",
                    "step": "HTML Generation",
                    "message": "The AI model failed to generate the document HTML.",
                    "hint": "Try again or simplify your prompt.",
                    "detail": str(e),
                },
            )

        cleaned_html = _clean_html(raw_html)

        if not cleaned_html.strip():
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "empty_output",
                    "step": "HTML Generation",
                    "message": "The AI model returned an empty document.",
                    "hint": "Try again or add more detail to your prompt.",
                    "raw_preview": raw_html[:300],
                },
            )

        try:
            update_storage(doc_id, cleaned_html)
        except Exception as e:
            logger.exception("[doc-gen] Storage write failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "storage_failed",
                    "step": "Save Document",
                    "message": "Document was generated but could not be saved to storage.",
                    "detail": str(e),
                },
            )

        return HTMLResponse(content=cleaned_html, headers={"X-Document-Id": doc_id})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[doc-gen] Unexpected error in /generate-html")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "unexpected_error",
                "message": "An unexpected error occurred during document generation.",
                "detail": str(e),
            },
        )


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

    if not _is_modification_query(request.modification_query):
        raise HTTPException(
            status_code=422,
            detail=(
                "This endpoint only accepts modification requests for an existing document.\n\n"
                "To generate a new document use /generate-html.\n\n"
                "To modify an existing document, describe what to change. Examples:\n"
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

    # LLM intent check — reject if user is trying to generate a different document type
    await _check_regeneration_intent(request.modification_query, existing_html)

    system_prompt = REGENERATE_PROMPT.format(
        existing_html=existing_html,
        modification_query=request.modification_query
    )

    try:
        raw_html = await _call_llm(system_prompt, request.modification_query)
    except Exception as e:
        logger.exception("[doc-gen] Regeneration LLM call failed")
        raise HTTPException(
            status_code=502,
            detail={
                "error": "llm_failed",
                "step": "Apply Modification",
                "message": "The AI model failed to apply your modification.",
                "hint": "Try again or rephrase what you want to change.",
                "detail": str(e),
            },
        )

    cleaned_html = _clean_html(raw_html)

    if not cleaned_html.strip():
        raise HTTPException(
            status_code=500,
            detail={
                "error": "empty_output",
                "step": "Apply Modification",
                "message": "The AI model returned an empty document after modification.",
                "hint": "Try again — the original document is still saved.",
                "raw_preview": raw_html[:300],
            },
        )

    try:
        update_storage(request.document_id, cleaned_html)
    except Exception as e:
        logger.exception("[doc-gen] Storage write failed on regeneration")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "storage_failed",
                "step": "Save Modified Document",
                "message": "Modification was applied but could not be saved to storage.",
                "detail": str(e),
            },
        )

    return HTMLResponse(content=cleaned_html)


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
