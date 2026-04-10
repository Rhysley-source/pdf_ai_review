import asyncio
import base64
import hashlib
import io
import json
import os
import re
import uuid
import logging
from collections import OrderedDict
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

# Step 3 (HTML generation) — full model, best output quality
_MODEL           = os.environ.get("MODEL_NAME", "gpt-5-nano")
# Steps 1+2 (JSON classification) — faster/lighter model, no quality impact
_FAST_MODEL      = os.environ.get("FAST_MODEL_NAME", "gpt-4.1-nano")
_API_KEY         = os.environ.get("OPENAI_API_KEY", "")
_CLIENT          = AsyncOpenAI(api_key=_API_KEY)

# Models that do not support the temperature parameter
_FIXED_TEMPERATURE_MODELS = {
    "gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini",
    "o1", "o1-mini", "o3-mini", "o3",
}

# ---------------------------------------------------------------------------
# LRU Cache for Steps 1+2 — keyed by prompt hash
# Same query → same seed (temperature=0) → result is deterministic,
# so caching is safe and has zero quality impact.
# Max 256 entries; oldest evicted first (OrderedDict-based LRU).
# ---------------------------------------------------------------------------

_PIPELINE_CACHE_MAX  = 256
_pipeline_cache: OrderedDict[int, dict] = OrderedDict()


def _cache_get(seed: int) -> dict | None:
    if seed not in _pipeline_cache:
        return None
    # Move to end (most-recently-used)
    _pipeline_cache.move_to_end(seed)
    return _pipeline_cache[seed]


def _cache_set(seed: int, value: dict) -> None:
    if seed in _pipeline_cache:
        _pipeline_cache.move_to_end(seed)
    else:
        if len(_pipeline_cache) >= _PIPELINE_CACHE_MAX:
            _pipeline_cache.popitem(last=False)  # evict LRU
    _pipeline_cache[seed] = value

# ---------------------------------------------------------------------------
# Storage — per-document files (fast O(1) read/write per doc)
#
# New writes go to  html_docs/<doc_id>.html  — one file per document.
# This avoids reading+writing the entire html_db.json on every request.
# Legacy html_db.json is kept as a read-only fallback for old documents.
# ---------------------------------------------------------------------------

_DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "html_docs")
_DB_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "html_db.json")
os.makedirs(_DOCS_DIR, exist_ok=True)


def _doc_path(doc_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", doc_id)
    return os.path.join(_DOCS_DIR, f"{safe_id}.html")


def _save_document(doc_id: str, html: str) -> None:
    """Write a single document file — no JSON serialization, no full-file rewrite."""
    with open(_doc_path(doc_id), "w", encoding="utf-8") as f:
        f.write(html)


def _load_document(doc_id: str) -> str | None:
    """Read a single document file. Falls back to legacy html_db.json if not found."""
    path = _doc_path(doc_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # Legacy fallback
    if os.path.exists(_DB_FILE):
        try:
            with open(_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get(doc_id)
        except (json.JSONDecodeError, OSError):
            pass
    return None


# Kept for backward compatibility — reads legacy store only
def get_storage() -> dict:
    if not os.path.exists(_DB_FILE):
        return {}
    try:
        with open(_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def update_storage(doc_id: str, html_content: str) -> None:
    """Legacy writer — new code should call _save_document instead."""
    _save_document(doc_id, html_content)


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

def _prompt_seed(system_prompt: str, user_message: str) -> int:
    """
    Derives a stable integer seed from the prompt content.
    Same prompt → same seed → deterministic LLM output (when model supports it).
    """
    digest = hashlib.sha256((system_prompt + user_message).encode()).hexdigest()
    return int(digest[:8], 16)


# Models that use max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {
    "gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini",
    "o1", "o1-mini", "o3-mini", "o3",
}

# Token budgets per step
_MAX_TOKENS_HTML  = 8192   # Step 3: full HTML document
_MAX_TOKENS_JSON  = 2048   # Steps 1+2: small JSON responses


async def _call_llm(
    system_prompt: str,
    user_message:  str,
    model:         str | None = None,
    max_tokens:    int = _MAX_TOKENS_HTML,
) -> str:
    """
    Core LLM caller. Uses the full generation model by default.
    Pass model=_FAST_MODEL for lightweight JSON classification steps.
    Pass max_tokens to cap output length per call.
    """
    model = model or os.environ.get("MODEL_NAME", _MODEL)

    kwargs: dict = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "seed": _prompt_seed(system_prompt, user_message),
    }

    # temperature=0 → greedy decoding, fully deterministic output
    if model not in _FIXED_TEMPERATURE_MODELS:
        kwargs["temperature"] = 0

    # Token limit — parameter name differs by model family
    if model in _MAX_COMPLETION_TOKENS_MODELS:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    try:
        response = await _CLIENT.chat.completions.create(**kwargs)
        choice   = response.choices[0]
        content  = choice.message.content or ""
        finish   = choice.finish_reason
        logger.info(
            f"[html-gen] model={model} in={response.usage.prompt_tokens} "
            f"out={response.usage.completion_tokens} finish={finish}"
        )
        if finish == "length":
            logger.warning("[html-gen] Response was cut off — increase max_tokens if HTML is incomplete")
        return content
    except Exception as e:
        logger.exception(f"[html-gen] OpenAI call failed: {e}")
        raise


async def _call_llm_fast(system_prompt: str, user_message: str) -> str:
    """Lightweight LLM call using the fast model — for JSON classification only."""
    return await _call_llm(system_prompt, user_message,
                           model=_FAST_MODEL, max_tokens=_MAX_TOKENS_JSON)


# ---------------------------------------------------------------------------
# User-facing error helpers
# ---------------------------------------------------------------------------

_GENERATE_EXAMPLES = [
    "Generate a service agreement between Acme Corp and Beta Ltd for consulting services worth $10,000",
    "Create an invoice for web development services — vendor: Sujit Studio, client: ABC Ltd, amount: $2,500",
    "Draft an NDA between two tech companies for a 2-year period, governed by California law",
    "Write an employment offer letter for a Senior Python Developer role at $90,000/year starting Jan 2026",
    "Make a residential lease agreement — landlord: Mr. Sharma, tenant: Rahul Verma, rent: $1,200/month",
    "Generate a business proposal for a mobile app project worth $50,000 for XYZ Corp",
    "Create a purchase order for 50 laptops from Dell at $800 each",
    "Write a recommendation letter for John Doe, Software Engineer at Google",
]

_MODIFY_EXAMPLES = [
    "Change the vendor name to Acme Corp",
    "Update the due date to 30th April 2026",
    "Add a 10% GST row to the totals table",
    "Replace the client address with 123 Main Street, New York",
    "Make the font size larger and the layout more professional",
    "Add a confidentiality clause at the end",
    "Change the payment terms from Net 30 to Net 15",
    "Remove the arbitration clause",
]


def _err_invalid_prompt(user_prompt: str) -> dict:
    """Returns a structured error for prompts that are too vague, short, or unclear."""
    return {
        "error":   "invalid_prompt",
        "message": (
            f"Your query \"{user_prompt[:80]}{'...' if len(user_prompt) > 80 else ''}\" is unclear. "
            "Please provide your request in the correct format."
        ),
    }


def _err_not_document_request(user_prompt: str) -> dict:
    """Returns a structured error when the prompt is not a document generation request."""
    return {
        "error":   "not_a_document_request",
        "message": (
            f"Your query \"{user_prompt[:80]}{'...' if len(user_prompt) > 80 else ''}\" is not a document generation request. "
            "Please provide your request in the correct format."
        ),
    }


def _err_model_failed(step: str, user_prompt: str, detail: str) -> dict:
    """Returns a structured error for transient AI model failures."""
    return {
        "error":   f"{step.lower().replace(' ', '_')}_failed",
        "message": f"An error occurred during {step}. Please try again.",
    }


def _err_empty_output(user_prompt: str) -> dict:
    """Returns a structured error when the model returns empty HTML."""
    return {
        "error":   "empty_output",
        "message": "The AI model returned an empty response. Please try again with more specific details.",
    }


def _err_invalid_modification(modification_query: str) -> dict:
    """Returns a structured error for modification prompts that are unclear."""
    return {
        "error":   "invalid_modification_query",
        "message": (
            f"Your query \"{modification_query[:80]}{'...' if len(modification_query) > 80 else ''}\" is unclear. "
            "Please provide your modification request in the correct format."
        ),
    }


def _err_document_not_found(document_id: str) -> dict:
    """Returns a structured error when the requested document ID does not exist."""
    return {
        "error":   "document_not_found",
        "message": f"No document found with ID '{document_id}'. Please generate a document first using /generate-html.",
    }


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
    """Step 1 — fast LLM call to detect document type and extract field values.
    Uses the fast model (JSON classification only).
    Raises HTTP 422 immediately if the query is not a document generation request.
    Caches result by prompt seed — same query skips the LLM call entirely.
    """
    seed   = _prompt_seed(QUERY_ANALYSIS_PROMPT.template, user_prompt)
    cached = _cache_get(seed)
    if cached and "analysis" in cached:
        logger.info("[doc-gen] Step 1: cache hit — skipping LLM call")
        analysis = cached["analysis"]
        analysis["_user_prompt"] = user_prompt
        analysis["_seed"]        = seed
    else:
        logger.info("[doc-gen] Step 1: analysing query (fast model)...")
        raw      = await _call_llm_fast(QUERY_ANALYSIS_PROMPT.template, user_prompt)
        logger.info(f"[doc-gen] Step 1 raw output: {raw[:300]}")
        analysis = _parse_analysis_json(raw)
        analysis["_user_prompt"] = user_prompt   # used by Step 2 for cache lookup
        analysis["_seed"]        = seed
        # Store partial cache entry — Step 2 will add "context" to it
        _cache_set(seed, {"analysis": analysis, "_seed": seed})

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
    """Step 2 — fast LLM call to build a tailored document blueprint.
    Uses the fast model (JSON output only).
    Caches result alongside Step 1 — same query skips both LLM calls.
    Falls back to the static Python template on parse failure.
    """
    seed   = _prompt_seed(QUERY_ANALYSIS_PROMPT.template, analysis.get("_user_prompt", ""))
    cached = _cache_get(seed)
    if cached and "context" in cached:
        logger.info("[doc-gen] Step 2: cache hit — skipping LLM call")
        return cached["context"]

    doc_type  = analysis.get("doc_type", "other")
    doc_label = analysis.get("doc_label", "Document")

    # Build extracted_fields string to pass into the prompt
    section_template = SECTION_TEMPLATES.get(doc_type, SECTION_TEMPLATES["other"])
    base_context     = build_generation_context(analysis, section_template)
    extracted_fields = base_context["extracted_fields"]

    system_prompt = TEMPLATE_BUILD_PROMPT.format(
        doc_type=doc_type,
        doc_label=doc_label,
        extracted_fields=extracted_fields,
    )

    logger.info(f"[doc-gen] Step 2: building blueprint for '{doc_label}' (fast model)...")
    raw = await _call_llm_fast(system_prompt, f"Build the document blueprint for: {doc_label}")
    logger.info(f"[doc-gen] Step 2 raw output: {raw[:300]}")

    context = _parse_blueprint_json(raw, analysis)

    # Update cache entry with the blueprint context
    if cached:
        cached["context"] = context
        _cache_set(seed, cached)

    return context


def _analysis_summary(analysis: dict) -> dict:
    """
    Extracts the safe, user-facing fields from a Step 1 analysis result.
    Included in every error response that occurs after Step 1 completes,
    so the client knows exactly what was detected from the query.
    """
    return {
        "detected_doc_type":  analysis.get("doc_type", "unknown"),
        "detected_doc_label": analysis.get("doc_label", "Unknown Document"),
        "extracted_fields":   {
            k: v for k, v in (analysis.get("fields") or {}).items()
            if not k.startswith("_")   # strip internal keys
        },
    }


async def _generate_html_from_context(context: dict, user_prompt: str) -> str:
    """
    Step 3 — final LLM call using the enriched blueprint context.
    Result is cached by prompt seed — same context + same prompt → instant return.
    """
    system_prompt = DOCUMENT_GENERATION_V2_PROMPT.format(
        **context,
        user_request=user_prompt,
    )
    seed   = _prompt_seed(system_prompt, user_prompt)
    cached = _cache_get(seed)
    if cached and "html" in cached:
        logger.info("[doc-gen] Step 3: cache hit — skipping LLM call")
        return cached["html"]

    logger.info(f"[doc-gen] Step 3: generating HTML for '{context['doc_label']}'...")
    html = await _call_llm(system_prompt, user_prompt)

    _cache_set(seed, {"html": html})
    return html


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


async def _check_regeneration_intent(modification_query: str, existing_html: str) -> str:
    """
    Calls LLM to determine whether the query wants to modify the existing document
    or generate a completely new one.

    Returns:
        "modify"       — user wants changes to the existing document
        "new_document" — user wants a brand new document of a different type
    Defaults to "modify" on any parse/LLM failure (fail open).
    """
    current_doc_type = _extract_doc_type_from_html(existing_html)

    system_prompt = REGENERATION_INTENT_PROMPT.format(
        current_doc_type=current_doc_type,
        modification_query=modification_query,
    )

    try:
        raw     = await _call_llm_fast(system_prompt, modification_query)
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        parsed  = json.loads(cleaned)
        intent  = parsed.get("intent", "modify")
        logger.info(f"[doc-gen] Regeneration intent: '{intent}' — {parsed.get('reason', '')}")
        return intent if intent in ("modify", "new_document") else "modify"
    except Exception:
        logger.warning("[doc-gen] Regeneration intent check failed — defaulting to 'modify'")
        return "modify"


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
        raise HTTPException(status_code=422, detail=_err_invalid_prompt(request.user_prompt))

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
                detail=_err_model_failed("Query Analysis", request.user_prompt, str(e)),
            )

        analysis_info = _analysis_summary(analysis)

        # Step 2: blueprint building (LLM)
        try:
            context = await _build_template_context(analysis)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[doc-gen] Step 2 failed")
            raise HTTPException(
                status_code=502,
                detail=_err_model_failed("Blueprint Building", request.user_prompt, str(e)),
            )

        # Step 3: final HTML generation
        try:
            raw_html = await _generate_html_from_context(context, request.user_prompt)
        except Exception as e:
            logger.exception("[doc-gen] Step 3 failed")
            raise HTTPException(
                status_code=502,
                detail=_err_model_failed("HTML Generation", request.user_prompt, str(e)),
            )

        cleaned_html = _clean_html(raw_html)

        if not cleaned_html.strip():
            raise HTTPException(
                status_code=500,
                detail=_err_empty_output(request.user_prompt),
            )

        try:
            await asyncio.to_thread(_save_document, doc_id, cleaned_html)
        except Exception as e:
            logger.exception("[doc-gen] Storage write failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "error":   "storage_failed",
                    "message": "Your document was generated but could not be saved. Please try again.",
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
                "error":   "unexpected_error",
                "message": "An unexpected error occurred. Please try again.",
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
            detail=_err_invalid_modification(request.modification_query),
        )

    # Fetch existing HTML — per-doc file read, wrapped in thread
    existing_html = await asyncio.to_thread(_load_document, request.document_id)

    if not existing_html:
        raise HTTPException(
            status_code=404,
            detail=_err_document_not_found(request.document_id),
        )

    # Intent check runs in parallel with nothing else here, but is isolated
    # so it uses the fast model and doesn't delay the modify path unnecessarily.
    # LLM intent check — branch based on whether user wants to modify or generate new
    intent = await _check_regeneration_intent(request.modification_query, existing_html)

    if intent == "new_document":
        # ── New document generation path (same as /generate-html) ──────────
        logger.info("[doc-gen] Regeneration intent=new_document — running generation pipeline")

        try:
            analysis = await _analyze_query(request.modification_query)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[doc-gen] Step 1 failed during regeneration→generate")
            raise HTTPException(
                status_code=502,
                detail=_err_model_failed("Query Analysis", request.modification_query, str(e)),
            )

        analysis_info = _analysis_summary(analysis)

        try:
            context = await _build_template_context(analysis)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[doc-gen] Step 2 failed during regeneration→generate")
            raise HTTPException(
                status_code=502,
                detail=_err_model_failed("Blueprint Building", request.modification_query, str(e)),
            )

        try:
            raw_html = await _generate_html_from_context(context, request.modification_query)
        except Exception as e:
            logger.exception("[doc-gen] Step 3 failed during regeneration→generate")
            raise HTTPException(
                status_code=502,
                detail=_err_model_failed("HTML Generation", request.modification_query, str(e)),
            )

        cleaned_html = _clean_html(raw_html)

        if not cleaned_html.strip():
            raise HTTPException(
                status_code=500,
                detail=_err_empty_output(request.modification_query),
            )

        doc_id = request.document_id or str(uuid.uuid4())
        await asyncio.to_thread(_save_document, doc_id, cleaned_html)
        return HTMLResponse(content=cleaned_html, headers={"X-Document-Id": doc_id})

    # ── Modify existing document path ───────────────────────────────────────
    system_prompt = REGENERATE_PROMPT.format(
        existing_html=existing_html,
        modification_query=request.modification_query,
    )

    try:
        raw_html = await _call_llm(system_prompt, request.modification_query)
    except Exception as e:
        logger.exception("[doc-gen] Regeneration LLM call failed")
        raise HTTPException(
            status_code=502,
            detail=_err_model_failed("Apply Modification", request.modification_query, str(e)),
        )

    cleaned_html = _clean_html(raw_html)

    if not cleaned_html.strip():
        raise HTTPException(
            status_code=500,
            detail=_err_empty_output(request.modification_query),
        )

    try:
        await asyncio.to_thread(_save_document, request.document_id, cleaned_html)
    except Exception as e:
        logger.exception("[doc-gen] Storage write failed on regeneration")
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "storage_failed",
                "message": "Modification was applied but could not be saved. Please try again.",
            },
        )

    return HTMLResponse(content=cleaned_html)


@router.get("/get-html/{document_id}", response_class=HTMLResponse)
async def get_document_html(
    document_id: str,
    _: None = Depends(verify_api_key),
):
    """
    Fetches previously generated HTML by document_id.
    """
    html = await asyncio.to_thread(_load_document, document_id)
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
        html_content = await asyncio.to_thread(_load_document, request.document_id)
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
            margin: 0;
            padding: 0;
            color: #000;
            background: #fff;
            -weasy-print-color-adjust: exact;
        }

        /* ── Heading alignment fix ──────────────────────────────────────────
           WeasyPrint's UA stylesheet uses margin-block-start / margin-block-end
           (CSS logical properties) for h1-h6. Their physical translations differ
           from Chrome's defaults and cause headings to appear shifted/indented.
           Resetting only the physical left/right margin+padding corrects this
           without touching text-align, font-size, or color set in the HTML.
           break-after:avoid keeps a heading attached to its following content
           so it never strands alone at the bottom of a page.
        ─────────────────────────────────────────────────────────────────── */
        h1, h2, h3, h4, h5, h6 {
            display: block;
            margin-left: 0;
            margin-right: 0;
            padding-left: 0;
            padding-right: 0;
            break-after: avoid;
            page-break-after: avoid;
        }

        table {
            border-collapse: collapse;
            word-wrap: break-word;
        }
        td, th {
            overflow: hidden;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        /* WeasyPrint does not support CSS Grid — fall back to block */
        [style*="display: grid"],
        [style*="display:grid"] {
            display: block !important;
        }
        /* WeasyPrint ignores fixed/sticky — make them static to avoid overlap */
        [style*="position: fixed"],
        [style*="position:fixed"],
        [style*="position: sticky"],
        [style*="position:sticky"] {
            position: static !important;
        }
    """

    try:
        from weasyprint import CSS
        pdf_bytes = WeasyprintHTML(
            string=html_content,
            base_url=".",
        ).write_pdf(
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

    await asyncio.to_thread(_save_document, request.doc_id, text_content)
    logger.info(f"[base64-text] updated doc_id='{request.doc_id}' ({len(text_content)} chars)")

    return {"doc_id": request.doc_id, "char_count": len(text_content)}
