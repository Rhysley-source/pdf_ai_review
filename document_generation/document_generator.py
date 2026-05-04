import asyncio
import base64
import html
import hashlib
import io
import json
import os
import re
import time
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
    COMBINED_ANALYSIS_BLUEPRINT_PROMPT,
    DOCUMENT_GENERATION_V2_PROMPT,
    DOCUMENT_GENERATION_TEXT_PROMPT,
    REGENERATION_INTENT_PROMPT,
    SECTION_TEMPLATES,
    build_generation_context,
    REGENERATE_PROMPT,
    REGENERATE_TEXT_PROMPT,
)
from auth import verify_api_key

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Step 3 (HTML generation) — full model, best output quality
_MODEL           = os.environ.get("MODEL_NAME", "gpt-4.1-nano")
# Steps 1+2 (JSON classification) — faster/lighter model, no quality impact
_FAST_MODEL      = os.environ.get("FAST_MODEL_NAME", "gpt-4.1-nano")
# Intent check — gpt-4o-mini for cheap semantic classification before full pipeline
_INTENT_MODEL    = os.environ.get("INTENT_MODEL_NAME", "gpt-4o-mini")
_API_KEY         = os.environ.get("OPENAI_API_KEY", "")
_CLIENT          = AsyncOpenAI(api_key=_API_KEY)

# Models that do not support the temperature parameter
_FIXED_TEMPERATURE_MODELS = {
    "gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini",
    "o1", "o1-mini", "o3-mini", "o3",
}

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


def _ascii_safe_html(content: str) -> str:
    """
    Convert non-ASCII characters to HTML entities so the output is
    7-bit ASCII-safe. Downstream MySQL latin1/utf8mb3 columns can store
    it without charset errors; browsers render the entities identically.
    e.g. ₹ → &#8377;  €  → &#8364;  © → &#169;
    """
    return content.encode("ascii", "xmlcharrefreplace").decode("ascii")


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


def _get_optional_int_env(name: str, default: int | None) -> int | None:
    """
    Parses an optional integer environment variable.
    Returns `default` when unset/invalid.
    Returns None when set to "none" or "0" (explicitly disabling the cap).
    """
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"none", "0"}:
        return None
    try:
        value = int(raw)
        if value <= 0:
            return default
        return value
    except ValueError:
        logger.warning(f"[doc-gen] Invalid {name}='{raw}' - using default={default}")
        return default


def _get_int_env(name: str, default: int) -> int:
    """Parses an integer environment variable with a safe fallback."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if value <= 0:
            return default
        return value
    except ValueError:
        logger.warning(f"[doc-gen] Invalid {name}='{raw}' - using default={default}")
        return default


# Token budgets per step
# Step 3 output cap: None = no cap (model outputs full response).
# Set MAX_TOKENS_HTML env var to a positive integer to re-enable a cap.
_MAX_TOKENS_HTML      = _get_optional_int_env("MAX_TOKENS_HTML", None)
_MAX_TOKENS_BLUEPRINT = _get_optional_int_env("MAX_TOKENS_BLUEPRINT", None)  # None = no cap; long docs need full output
_MAX_TOKENS_JSON      = 512   # Step 1: small JSON classification response — output is always compact
_MAX_TOKENS_COMBINED  = _get_optional_int_env("MAX_TOKENS_COMBINED", None)  # None = no cap; resumes/contracts need full blueprint
_HTML_GEN_RETRIES     = _get_int_env("HTML_GEN_RETRIES", 2)  # Step 3 retry attempts
_COMPACT_HTML_MAX_TOKENS = _get_int_env("COMPACT_HTML_MAX_TOKENS", 1800)
_COMPACT_HTML_RETRIES = _get_int_env("COMPACT_HTML_RETRIES", 2)


async def _call_llm(
    system_prompt: str,
    user_message:  str,
    model:         str | None = None,
    max_tokens:    int | None = _MAX_TOKENS_HTML,
    temperature:   float | None = None,
    use_seed:      bool = True,
) -> tuple[str, str]:
    """
    Core LLM caller. Uses the full generation model by default.
    Pass model=_FAST_MODEL for lightweight JSON classification steps.
    Pass max_tokens to cap output length per call.
    Pass temperature to control randomness (None = model default / steps 1+2 use 0).
    Pass use_seed=False to skip the deterministic seed (all steps now use True).
    """
    model = model or os.environ.get("MODEL_NAME", _MODEL)

    kwargs: dict = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }

    # Seed — used by all steps; pass use_seed=False to skip for non-deterministic calls
    if use_seed:
        kwargs["seed"] = _prompt_seed(system_prompt, user_message)

    # Temperature — models in _FIXED_TEMPERATURE_MODELS don't accept this param
    if model not in _FIXED_TEMPERATURE_MODELS:
        kwargs["temperature"] = temperature if temperature is not None else 0

    # Token limit — omitted when None so the model uses its own built-in maximum.
    # Steps 1+2 always pass explicit limits; Step 3 uses _MAX_TOKENS_HTML.
    if max_tokens is not None:
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
            logger.warning("[html-gen] finish=length — response was cut off mid-output")
        return content, finish
    except Exception as e:
        logger.exception(f"[html-gen] OpenAI call failed: {e}")
        raise


async def _call_llm_fast(system_prompt: str, user_message: str) -> str:
    """Lightweight LLM call using the fast model — for JSON classification only."""
    content, _ = await _call_llm(system_prompt, user_message,
                                 model=_FAST_MODEL, max_tokens=_MAX_TOKENS_JSON)
    return content


_INTENT_CHECK_SYSTEM_PROMPT = """\
You are a document intent classifier. Your job is to decide what the user wants.

There are exactly 3 possible intents:

━━━ 1. "request" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The user wants to CREATE or GENERATE a specific, identifiable document type.

BOTH conditions must be true to return "request":
  A) The query mentions a SPECIFIC document type (see list below)
  B) The query makes sense as a real document generation request

Recognised document types:
  resume, cv, curriculum vitae, invoice, bill, receipt, contract, agreement,
  offer letter, employment letter, appointment letter, nda, non-disclosure,
  lease, rent agreement, certificate, report, proposal, purchase order,
  letter, memo, quotation, payslip, salary slip, experience letter,
  relieving letter, joining letter, termination letter, internship letter

Valid examples — a document type alone is enough, extra details are optional:
  "resume"                                   → request
  "generate resume"                          → request
  "create invoice"                           → request
  "nda"                                      → request
  "resume sujeet python developer"           → request
  "create resume for John as Python dev"     → request
  "invoice 5000 to ABC Corp"                 → request
  "nda between Acme and Beta"                → request
  "offer letter priya manager 80k"           → request
  "make me a contract for freelance work"    → request
  "certificate of completion for rahul"      → request

━━━ 2. "raw_document" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The user has PASTED the actual text of an existing document — long structured
content with headings, clauses, dates, addresses, signature lines, tables, etc.
It looks like a real document, not a request to make one.

━━━ 3. "unrelated" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return "unrelated" for ANY of these cases:
  • No specific document type is mentioned
  • Query is gibberish, repeated words, or random text
  • Query uses action words (generate, create, make) WITHOUT a document type
  • General questions, greetings, math, coding help, weather, etc.

Invalid examples (return "unrelated"):
  "generate generate generate"   → unrelated  (repeated word, no document type)
  "create create"                → unrelated  (no document type)
  "make something"               → unrelated  (vague, no document type)
  "generate"                     → unrelated  (trigger word only)
  "what is python"               → unrelated
  "hello"                        → unrelated
  "2 + 2"                        → unrelated

━━━ RULE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A trigger word (generate, create, make) alone WITHOUT a document type → "unrelated".
A document type mentioned (with or without a trigger word) → "request".
Only return "unrelated" when NO document type is present or the query is gibberish.

Return ONLY: {"intent": "<request|raw_document|unrelated>"}"""


async def _check_document_intent(user_prompt: str) -> str:
    """
    Uses gpt-4o-mini to classify the user prompt as:
      "request"      — wants to generate a document
      "raw_document" — pasted an existing document
      "unrelated"    — nothing to do with document generation

    Falls back to "request" on any parse/API error so the main pipeline
    decides (and raises 422 if truly invalid).
    """
    try:
        content, _ = await _call_llm(
            system_prompt=_INTENT_CHECK_SYSTEM_PROMPT,
            user_message=user_prompt,
            model=_INTENT_MODEL,
            max_tokens=20,
            use_seed=False,
        )
        parsed = json.loads(content.strip())
        intent = parsed.get("intent", "request")
        if intent not in ("request", "raw_document", "unrelated"):
            intent = "request"
        return intent
    except Exception:
        logger.warning("[doc-gen] intent check failed, defaulting to 'request'")
        return "request"


_MODIFICATION_INTENT_SYSTEM_PROMPT = """\
You are a document modification classifier. The user has an existing document and is providing a query.
Determine if the query is asking to modify, update, change, or regenerate the document in any way.

Return ONLY: {"is_modification": true} or {"is_modification": false}

true  — any edit, update, change, add, remove, replace, rewrite, reformat request
false — unrelated queries (questions, general chat, math, greetings, etc.)"""


async def _check_modification_intent(query: str) -> bool:
    """
    Uses gpt-4o-mini to confirm the query is a modification/change request
    for an existing document. Returns False for unrelated queries.
    Falls back to True on any error so the pipeline continues normally.
    """
    try:
        content, _ = await _call_llm(
            system_prompt=_MODIFICATION_INTENT_SYSTEM_PROMPT,
            user_message=query,
            model=_INTENT_MODEL,
            max_tokens=10,
            use_seed=False,
        )
        parsed = json.loads(content.strip())
        return bool(parsed.get("is_modification", True))
    except Exception:
        logger.warning("[doc-gen] modification intent check failed, defaulting to True")
        return True


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


def _is_raw_document(text: str) -> bool:
    """
    Returns True when the input looks like a pasted raw/existing document
    rather than a user request to create one.

    Requires length > 500 chars AND at least 2 structural signals.
    """
    stripped = text.strip()
    if len(stripped) <= 500:
        return False

    signals = 0
    if re.search(r'\b[A-Z]{3,}[\s:]+[A-Z]{3,}', stripped):
        signals += 1
    if re.search(r'_{5,}', stripped):
        signals += 1
    if re.search(r'^\s*\d+[\.\)]\s+\w', stripped, re.MULTILINE):
        signals += 1
    if re.search(r'\b(WHEREAS|THEREFORE|HEREINAFTER|AGREEMENT|INVOICE|CERTIFICATE)\b', stripped):
        signals += 1
    if re.search(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', stripped):
        signals += 1
    if re.search(r'[\$₹£€]\s*\d+', stripped):
        signals += 1

    return signals >= 2


# ---------------------------------------------------------------------------
# Output cleaning + structural validation
# ---------------------------------------------------------------------------

def _clean_html(raw: str) -> str:
    """Strip markdown fences and extract the HTML block from raw LLM output."""
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```html", "").replace("```", "").strip()
    lower   = cleaned.lower()
    start   = lower.find("<html")
    end     = lower.rfind("</html>")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 7]
    elif start != -1:
        # Truncated — keep everything from <html onward
        cleaned = cleaned[start:]
    return cleaned


def _repair_truncated_html(html_text: str) -> str:
    """
    Best-effort repair for truncated model output.
    Adds missing closing tags so downstream renderers can still load the document.
    """
    repaired = html_text.strip()
    if not repaired:
        return repaired

    low = repaired.lower()
    if "<html" not in low:
        return repaired

    if "</body>" not in low and "<body" in low:
        repaired += "\n</body>"
    if "</html>" not in low:
        repaired += "\n</html>"
    return repaired


def _validate_html(html: str) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).

    Checks that the HTML has all the structural pieces needed to render as a
    complete, styled document.  Any missing piece triggers a retry so the model
    gets another chance to produce a properly structured response.

    Checks (in order):
      1. Non-empty after cleaning
      2. Has <html> opening tag
      3. Has <head> section
      4. Has embedded <style> block  — external <link> stylesheets are not
         acceptable because they won't resolve at PDF conversion time
      5. Has <body> section
      6. Has </html> closing tag     — missing = response was truncated
      7. Has meaningful text content — guards against a shell of empty tags
    """
    if not html.strip():
        return False, "empty response"

    low = html.lower()

    if "<html" not in low:
        return False, "missing <html> tag"
    if "<head" not in low:
        return False, "missing <head> section"
    if "<style" not in low:
        return False, "missing embedded <style> block — no CSS"
    if "<body" not in low:
        return False, "missing <body> section"
    if "</html>" not in low:
        return False, "response truncated — </html> not found"

    # Strip all tags and check for at least 100 chars of real text content
    text = re.sub(r"<[^>]+>", "", html)
    if len(text.strip()) < 100:
        return False, "insufficient text content"

    return True, "ok"


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
    Always calls the LLM — no caching.
    """
    logger.info("[doc-gen] Step 1: analysing query (fast model)...")
    raw      = await _call_llm_fast(QUERY_ANALYSIS_PROMPT.template, user_prompt)
    logger.info(f"[doc-gen] Step 1 raw output: {raw[:300]}")
    analysis = _parse_analysis_json(raw)
    analysis["_user_prompt"] = user_prompt

    if not analysis.get("is_document_request", False):
        raise HTTPException(
            status_code=422,
            detail=_err_not_document_request(analysis.get("_user_prompt", "")),
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

    # Build the sections_block string for Step 3 — include missing_fields so
    # Step 3 knows exactly which placeholders to render visibly in the document.
    lines = []
    for i, sec in enumerate(sections, 1):
        title          = sec.get("title", f"Section {i}")
        content_hint   = sec.get("content_hint", "")
        missing        = sec.get("missing_fields", [])
        entry          = f"{i}. {title}\n   → {content_hint}"
        if missing:
            entry += f"\n   ⚠ Missing fields (use visible placeholders): {', '.join(missing)}"
        lines.append(entry)
    sections_block = "\n\n".join(lines)

    # document_title from blueprint overrides the generic doc_label for the
    # actual heading shown in the document (e.g. "RENT AGREEMENT" vs "Lease Agreement")
    doc_label      = analysis.get("doc_label", "Document")
    document_title = (parsed.get("document_title") or "").strip() or doc_label

    return {
        "doc_type":       analysis.get("doc_type", "other"),
        "doc_label":      document_title,
        "tone":           parsed.get("tone", "professional"),
        "layout_notes":   parsed.get("layout_notes", "Standard document layout."),
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


async def _build_template_context(analysis: dict, user_prompt: str = "") -> dict:
    """Step 2 — builds a detailed, pre-filled document blueprint.

    Uses the fast model but with a higher token budget (_MAX_TOKENS_BLUEPRINT)
    so every section gets a complete content_hint with all field values embedded.

    Passes three inputs to the blueprint prompt:
      - extracted_fields  : structured key→value pairs from Step 1
      - required_sections : standard sections for this document type
      - user_request      : the original full user prompt, so any extra details,
                            clauses, or requirements the user mentioned are captured
    Falls back to the static Python template on parse failure.
    """
    doc_type  = analysis.get("doc_type", "other")
    doc_label = analysis.get("doc_label", "Document")

    section_template  = SECTION_TEMPLATES.get(doc_type, SECTION_TEMPLATES["other"])
    base_context      = build_generation_context(analysis, section_template)
    extracted_fields  = base_context["extracted_fields"]
    required_sections = base_context["required_sections"]

    system_prompt = TEMPLATE_BUILD_PROMPT.format(
        doc_type=doc_type,
        doc_label=doc_label,
        extracted_fields=extracted_fields,
        required_sections=required_sections,
        user_request=user_prompt,
    )

    logger.info(f"[doc-gen] Step 2: building blueprint for '{doc_label}'...")
    raw, _ = await _call_llm(
        system_prompt,
        f"Build the complete, pre-filled document blueprint for: {doc_label}",
        model=_FAST_MODEL,
        max_tokens=_MAX_TOKENS_BLUEPRINT,
        temperature=0,
        use_seed=True,
    )
    logger.info(f"[doc-gen] Step 2 raw output: {raw[:300]}")

    return _parse_blueprint_json(raw, analysis)


async def _analyze_and_build(user_prompt: str) -> dict:
    """
    Combined Step 1+2 — single LLM call that classifies the request, extracts
    fields, and builds the document blueprint in one round-trip.

    Returns a context dict ready for _generate_html_from_context (same shape as
    _build_template_context). Raises HTTP 422 for non-document requests.
    Falls back to static template on JSON parse failure.
    """
    logger.info("[doc-gen] Steps 1+2 (combined): analysing and building blueprint...")
    raw, finish = await _call_llm(
        COMBINED_ANALYSIS_BLUEPRINT_PROMPT.template,
        user_prompt,
        model=_FAST_MODEL,
        max_tokens=_MAX_TOKENS_COMBINED,
        temperature=0,
        use_seed=True,
    )
    logger.info(f"[doc-gen] Steps 1+2 combined raw output (finish={finish}, total_chars={len(raw)}): {raw[:120]}…")

    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        if finish == "length":
            logger.warning(
                f"[doc-gen] Steps 1+2 combined: output truncated at {_MAX_TOKENS_COMBINED} tokens "
                f"(prompt length={len(user_prompt)} chars) — falling back to two-step flow"
            )
        else:
            logger.warning("[doc-gen] Steps 1+2 combined: parse failed — falling back to two-step flow")
        analysis = await _analyze_query(user_prompt)
        return await _build_template_context(analysis, user_prompt=user_prompt)

    if not parsed.get("is_document_request", False):
        raise HTTPException(
            status_code=422,
            detail=_err_not_document_request(user_prompt),
        )

    sections = parsed.get("sections", [])
    if not isinstance(sections, list) or not sections:
        logger.warning("[doc-gen] Steps 1+2 combined: empty sections — falling back to static template")
        analysis = {
            "doc_type":  parsed.get("doc_type", "other"),
            "doc_label": parsed.get("doc_label", "Document"),
            "fields":    parsed.get("fields") or {},
        }
        return _static_template_context(analysis)

    lines = []
    for i, sec in enumerate(sections, 1):
        title        = sec.get("title", f"Section {i}")
        content_hint = sec.get("content_hint", "")
        missing      = sec.get("missing_fields", [])
        entry        = f"{i}. {title}\n   → {content_hint}"
        if missing:
            entry += f"\n   ⚠ Missing fields (use visible placeholders): {', '.join(missing)}"
        lines.append(entry)

    doc_label      = parsed.get("doc_label", "Document")
    document_title = (parsed.get("document_title") or "").strip() or doc_label

    return {
        "doc_type":       parsed.get("doc_type", "other"),
        "doc_label":      document_title,
        "tone":           parsed.get("tone", "professional"),
        "layout_notes":   parsed.get("layout_notes", "Standard document layout."),
        "sections_block": "\n\n".join(lines),
    }





def _extract_section_titles_from_block(sections_block: str) -> list[str]:
    """Extract readable section titles from the Step 2 sections_block string."""
    titles: list[str] = []
    for raw_line in sections_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("->") or line.startswith("â†’") or line.startswith("âš "):
            continue
        match = re.match(r"^\d+\.\s*(.+)$", line)
        if match:
            line = match.group(1).strip()
        if line and line not in titles:
            titles.append(line)
    return titles


def _render_static_html_from_context(context: dict, user_prompt: str) -> str:
    """
    Fast deterministic fallback HTML renderer.
    Used for short, low-detail prompts where LLM generation can be slow and unstable.
    """
    title = html.escape(context.get("doc_label", "Document"))
    doc_type = html.escape(context.get("doc_type", "document").replace("_", " ").title())
    prompt_preview = html.escape(user_prompt.strip()[:220])
    sections_block = context.get("sections_block", "")

    section_titles = _extract_section_titles_from_block(sections_block)
    if not section_titles:
        section_titles = ["Parties", "Terms", "Payment", "Signatures"]

    sections_html: list[str] = []
    for section_title in section_titles:
        safe_title = html.escape(section_title)
        placeholder = html.escape(f"[Provide {section_title} details]")
        sections_html.append(
            f"""
            <section style="margin-bottom: 12px; page-break-inside: avoid;">
              <h2 style="font-size:13pt; font-weight:bold; text-align:left; margin-top:12px; margin-bottom:6px; color:#000000;">{safe_title}</h2>
              <p style="margin:0 0 8px 0; line-height:1.5;">
                <span style="font-style:italic;">{placeholder}</span>
              </p>
            </section>
            """
        )

    return f"""
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #000000; background: #ffffff; margin: 0; padding: 24px; }}
    p    {{ margin: 0 0 8px 0; line-height: 1.5; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    td, th {{ border: 1px solid #cccccc; padding: 6px 8px; vertical-align: top; }}
  </style>
</head>
<body>
  <div contenteditable="true">
    <h1 style="font-size:16pt; font-weight:bold; text-align:center; margin-top:0; margin-bottom:10px; color:#000000;">{title}</h1>
    <p style="margin:0 0 10px 0; line-height:1.5;"><strong>Document Type:</strong> {doc_type}</p>
    <p style="margin:0 0 12px 0; line-height:1.5;"><strong>Request:</strong> {prompt_preview}</p>
    <hr style="border:none; border-top:1px solid #cccccc; margin:10px 0;">
    {''.join(sections_html)}
    <section style="margin-top: 18px; page-break-inside: avoid;">
      <h2 style="font-size:13pt; font-weight:bold; text-align:left; margin-top:12px; margin-bottom:6px; color:#000000;">Signatures</h2>
      <table>
        <tr>
          <td style="word-wrap:break-word; overflow-wrap:break-word;"><strong>Party 1 Signature:</strong><br><br>________________________</td>
          <td style="word-wrap:break-word; overflow-wrap:break-word;"><strong>Party 2 Signature:</strong><br><br>________________________</td>
        </tr>
      </table>
    </section>
  </div>
</body>
</html>
""".strip()


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



async def _generate_html_from_context(
    context: dict,
    user_prompt: str,
    compact_mode: bool = False,
) -> str:
    """
    Step 3 - final LLM call using the enriched blueprint context.
    Retries on truncated or invalid response.

    compact_mode is used for short, low-detail prompts to keep output concise
    and reduce truncation risk.
    """
    if compact_mode:
        sections_block = context.get("sections_block", "")
        if len(sections_block) > 1200:
            compact_titles = _extract_section_titles_from_block(sections_block)
            if compact_titles:
                sections_block = "\n".join(
                    f"{i}. {title}" for i, title in enumerate(compact_titles, 1)
                )

        # Short prompts without extracted fields can still trigger huge output
        # with the full prompt. Use a lighter prompt to keep responses complete.
        system_prompt = f"""You are an expert legal document HTML generator.
Generate one complete HTML document only.

Document Type: {context.get("doc_label", "Document")} ({context.get("doc_type", "other")})
Tone: {context.get("tone", "professional")}
Layout Notes: {context.get("layout_notes", "Standard document layout")}

Sections to include in this order:
{sections_block}

User request:
{user_prompt}

Rules:
- Return ONLY HTML from <html> to </html>.
- Include <head> with one embedded <style> block and <body>.
- Keep content inside one outer <div contenteditable="true">.
- Keep output concise and complete (about 500-800 words).
- If details are missing, use specific placeholders like [Landlord Name], [Property Address], [Start Date].
- Use clean print-friendly formatting (Arial, white background, simple tables where needed).
- Do not use markdown fences.
"""
        current_max_tokens = _COMPACT_HTML_MAX_TOKENS
        retries = max(1, _COMPACT_HTML_RETRIES)
        model_for_call = _FAST_MODEL
    else:
        system_prompt = DOCUMENT_GENERATION_V2_PROMPT.format(
            **context,
            user_request=user_prompt,
        )
        current_max_tokens = _MAX_TOKENS_HTML
        retries = _HTML_GEN_RETRIES
        model_for_call = None

    best_effort_html = ""

    for attempt in range(1, retries + 1):
        if attempt == 1:
            logger.info(
                f"[doc-gen] Step 3: generating HTML for '{context['doc_label']}'"
                f"{' (compact)' if compact_mode else ''}..."
            )
        else:
            logger.warning(f"[doc-gen] Step 3: retry {attempt}/{retries}")

        try:
            raw, finish = await _call_llm(
                system_prompt,
                user_prompt,
                model=model_for_call,
                max_tokens=current_max_tokens,
                temperature=0.1 if compact_mode else 0.2,
                use_seed=(attempt == 1),
            )
        except Exception:
            if attempt == retries:
                raise
            logger.warning(f"[doc-gen] Step 3: LLM call failed on attempt {attempt} - retrying")
            continue

        # If model reports truncation, repair and validate before retrying.
        if finish == "length":
            cleaned = _repair_truncated_html(_clean_html(raw))
            if len(cleaned) > len(best_effort_html):
                best_effort_html = cleaned
            valid, _ = _validate_html(cleaned)
            if valid:
                logger.warning(
                    "[doc-gen] Step 3: finish=length but HTML is complete and valid - accepting output"
                )
                return cleaned

            if attempt < retries:
                logger.warning(
                    f"[doc-gen] Step 3: response truncated (finish=length) on attempt "
                    f"{attempt}/{retries} - retrying"
                )
            else:
                logger.warning(
                    f"[doc-gen] Step 3: response truncated (finish=length) on final attempt "
                    f"{attempt}/{retries}"
                )
                if best_effort_html:
                    logger.warning(
                        "[doc-gen] Step 3: returning best-effort repaired HTML after truncation"
                    )
                    return best_effort_html
            continue

        cleaned = _repair_truncated_html(_clean_html(raw))
        if len(cleaned) > len(best_effort_html):
            best_effort_html = cleaned
        valid, reason = _validate_html(cleaned)
        if valid:
            return cleaned

        logger.warning(
            f"[doc-gen] Step 3: invalid HTML on attempt {attempt}/{retries} "
            f"- {reason} (finish={finish})"
        )

    if best_effort_html:
        logger.warning("[doc-gen] Step 3: returning best-effort repaired HTML after all attempts")
        return best_effort_html

    return ""  # all attempts exhausted - caller raises HTTPException


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
    intent = await _check_document_intent(request.user_prompt)
    logger.info(f"[doc-gen] /generate-html intent={intent!r}")
    if intent == "unrelated":
        raise HTTPException(status_code=422, detail=_err_invalid_prompt(request.user_prompt))
    elif intent == "raw_document":
        request = request.model_copy(update={
            "user_prompt": (
                "The following is a complete existing document. "
                "Analyze it, identify its type, extract all field values, "
                "and generate a new complete document of the same type:\n\n"
                + request.user_prompt
            )
        })

    doc_id = request.document_id or str(uuid.uuid4())
    request_started = time.perf_counter()
    logger.info(
        f"[doc-gen] /generate-html start doc_id={doc_id} "
        f"max_tokens_html={_MAX_TOKENS_HTML} retries={_HTML_GEN_RETRIES}"
    )

    try:
        # Steps 1+2: combined analysis + blueprint (single LLM call)
        step_started = time.perf_counter()
        try:
            context = await _analyze_and_build(request.user_prompt)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[doc-gen] Steps 1+2 combined failed")
            raise HTTPException(
                status_code=502,
                detail=_err_model_failed("Analysis + Blueprint", request.user_prompt, str(e)),
            )
        logger.info(
            f"[doc-gen] Steps 1+2 complete in {time.perf_counter() - step_started:.2f}s "
            f"doc_label='{context.get('doc_label', 'Document')}'"
        )

        # Step 3: generate full HTML (with retry + validation)
        step_started = time.perf_counter()
        try:
            raw_html = await _generate_html_from_context(context, request.user_prompt)
        except Exception as e:
            logger.exception("[doc-gen] Step 3 failed")
            raise HTTPException(
                status_code=502,
                detail=_err_model_failed("HTML Generation", request.user_prompt, str(e)),
            )
        logger.info(
            f"[doc-gen] Step 3 complete in {time.perf_counter() - step_started:.2f}s"
        )

        if not raw_html.strip():
            raise HTTPException(
                status_code=500,
                detail=_err_empty_output(request.user_prompt),
            )

        safe_html = _ascii_safe_html(raw_html)
        await asyncio.to_thread(_save_document, doc_id, safe_html)
        logger.info(
            f"[doc-gen] /generate-html done doc_id={doc_id} "
            f"total={time.perf_counter() - request_started:.2f}s"
        )
        return HTMLResponse(content=safe_html, headers={"X-Document-Id": doc_id})

    except HTTPException as exc:
        logger.warning(
            f"[doc-gen] /generate-html failed doc_id={doc_id} "
            f"status={exc.status_code} total={time.perf_counter() - request_started:.2f}s"
        )
        raise
    except Exception as e:
        logger.exception(
            f"[doc-gen] Unexpected error in /generate-html after "
            f"{time.perf_counter() - request_started:.2f}s"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "unexpected_error",
                "message": "An unexpected error occurred. Please try again.",
            },
        )


_DIRECT_TEXT_SYSTEM_PROMPT = """\
You are an expert legal document writer. Generate a COMPLETE, fully detailed, professionally formatted plain-text document based on the user's request.

CONTENT RULES:
- Every section must contain full legal/professional language — complete sentences, standard clauses, obligations, rights, and conditions.
- Do NOT write one-line summaries. Each section must be a proper paragraph or set of numbered clauses (3–6 sentences minimum).
- Use placeholder brackets ONLY for sensitive or user-specific data: [Party Name], [Address], [Amount], [Date], [Governing State], etc.
- All standard legal language, obligations, and boilerplate must be written out in full — never replaced with placeholders.

FORMATTING RULES:
- Output plain text only — no HTML tags, no markdown, no backticks.
- Document title: ALL CAPS, centered using spaces, on its own line.
- Section headings: ALL CAPS followed by a colon, on their own line.
- Separate major sections with: ----------------------------------------
- Tables: use plain ASCII with | and - characters.
- Signature blocks: use underscores: ____________________________
- Do not add any preamble, explanation, or closing note — output the document only.\
"""


@router.post("/generate-text/stream")
async def generate_document_text_stream(
    request: DocumentGenerationRequest,
    _: None = Depends(verify_api_key),
):
    """
    Streams a complete plain-text document directly from the user prompt.
    Single LLM call after intent check — no blueprint step.
    """
    intent = await _check_document_intent(request.user_prompt)
    logger.info(f"[doc-gen] /generate-text/stream intent={intent!r}")

    if intent == "unrelated":
        raise HTTPException(
            status_code=422,
            detail=_err_invalid_prompt(request.user_prompt),
        )

    user_message = request.user_prompt
    if intent == "raw_document":
        user_message = (
            "The following is a complete existing document. "
            "Analyze it, identify its type, and generate a new complete document of the same type:\n\n"
            + request.user_prompt
        )

    doc_id = request.document_id or str(uuid.uuid4())
    request_started = time.perf_counter()
    logger.info(f"[doc-gen] /generate-text/stream start doc_id={doc_id}")

    async def _stream_text():
        model = os.environ.get("MODEL_NAME", _MODEL)

        kwargs: dict = {
            "model":    model,
            "messages": [
                {"role": "system", "content": _DIRECT_TEXT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            "stream": True,
        }
        if model not in _FIXED_TEMPERATURE_MODELS:
            kwargs["temperature"] = 0.2
        if _MAX_TOKENS_HTML is not None:
            if model in _MAX_COMPLETION_TOKENS_MODELS:
                kwargs["max_completion_tokens"] = _MAX_TOKENS_HTML
            else:
                kwargs["max_tokens"] = _MAX_TOKENS_HTML

        accumulated: list[str] = []
        try:
            stream = await _CLIENT.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if not delta:
                    continue
                accumulated.append(delta)
                yield delta.encode("utf-8")
        except Exception:
            logger.exception("[doc-gen] /generate-text/stream Step 3 failed")
            return

        full_text = "".join(accumulated)
        if full_text.strip():
            try:
                await asyncio.to_thread(_save_document, doc_id, full_text)
                logger.info(
                    f"[doc-gen] /generate-text/stream saved doc_id={doc_id} "
                    f"total={time.perf_counter() - request_started:.2f}s"
                )
            except Exception:
                logger.exception("[doc-gen] /generate-text/stream storage write failed")

    return StreamingResponse(
        _stream_text(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Document-Id":     doc_id,
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
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
    is_modification = await _check_modification_intent(request.modification_query)
    logger.info(f"[doc-gen] /regenerate-html modification_intent={is_modification}")
    if not is_modification:
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

        try:
            context = await _build_template_context(analysis, user_prompt=request.modification_query)
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
        raw_html, _ = await _call_llm(system_prompt, request.modification_query)
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


@router.post("/regenerate-text/stream")
async def regenerate_document_html_stream(
    request: DocumentRegenerationRequest,
    _: None = Depends(verify_api_key),
):
    """
    Streaming version of /regenerate-html.

    Two paths — decided by an intent check before streaming starts:
      • modify       → streams the modified existing document HTML
      • new_document → runs Steps 1+2 (classify + blueprint) then streams fresh HTML

    HTML is streamed chunk-by-chunk as text/html. After the stream completes
    the full document is cleaned, validated, and saved to storage.
    X-Document-Id header carries the document ID.
    """
    if _is_gibberish(request.modification_query):
        raise HTTPException(
            status_code=422,
            detail=_err_invalid_modification(request.modification_query),
        )

    existing_html = await asyncio.to_thread(_load_document, request.document_id)
    if not existing_html:
        raise HTTPException(
            status_code=404,
            detail=_err_document_not_found(request.document_id),
        )

    # Intent check completes before streaming so we know which path to take
    intent = await _check_regeneration_intent(request.modification_query, existing_html)
    logger.info(f"[doc-gen] /regenerate-html/stream intent={intent} doc_id={request.document_id}")

    doc_id = request.document_id
    model  = os.environ.get("MODEL_NAME", _MODEL)

    # Both paths use the plain-text regeneration prompt
    system_prompt = REGENERATE_TEXT_PROMPT.format(
        existing_html=existing_html,
        modification_query=request.modification_query,
    )
    user_message = request.modification_query

    async def _stream():
        kwargs: dict = {
            "model":    model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "stream": True,
        }
        if model not in _FIXED_TEMPERATURE_MODELS:
            kwargs["temperature"] = 0.2
        if _MAX_TOKENS_HTML is not None:
            if model in _MAX_COMPLETION_TOKENS_MODELS:
                kwargs["max_completion_tokens"] = _MAX_TOKENS_HTML
            else:
                kwargs["max_tokens"] = _MAX_TOKENS_HTML

        try:
            stream = await _CLIENT.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if not delta:
                    continue
                yield delta.encode()
        except Exception:
            logger.exception("[doc-gen] /regenerate-html/stream Step 3 failed")
            return

    return StreamingResponse(
        _stream(),
        media_type="text/plain",
        headers={
            "X-Document-Id":     doc_id,
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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

    # Only @page (for paper size and margins) plus WeasyPrint compatibility
    # shims are injected here. Everything else — body padding, heading styles,
    # fonts, colors — is intentionally left to the HTML's own <style> block.
    #
    # Why: WeasyPrint applies passed stylesheets AFTER the document's own
    # <style> block in the CSS cascade. Rules with the same specificity that
    # appear later win. Any body/heading reset added here therefore silently
    # overrides the generated HTML's own layout styles, which is what caused
    # headings to appear shifted left relative to the surrounding content.
    a4_css = """
        @page {
            size: A4 portrait;
            margin: 15mm 15mm 15mm 15mm;
        }

        /* WeasyPrint does not support CSS Grid — fall back to block so
           content flows rather than disappearing entirely */
        [style*="display: grid"],
        [style*="display:grid"] {
            display: block !important;
        }

        /* WeasyPrint ignores position:fixed/sticky/absolute; make them static
           so they don't overlap page content */
        [style*="position: fixed"],
        [style*="position:fixed"],
        [style*="position: sticky"],
        [style*="position:sticky"],
        [style*="position: absolute"],
        [style*="position:absolute"] {
            position: static !important;
        }

        /* Fixed heights cause content to overflow and overlap the next block.
           Force all block containers to size themselves to their content.
           min-height is intentionally NOT reset — it is used for legitimate spacing
           (e.g. signature areas) and does not cause overflow in PDF. */
        div, section, article, aside, header, footer, main, li {
            height: auto !important;
            overflow: visible !important;
        }

        /* Negative margins pull elements into the previous block — zero them. */
        [style*="margin-top: -"],
        [style*="margin-top:-"] {
            margin-top: 0 !important;
        }
        [style*="margin-bottom: -"],
        [style*="margin-bottom:-"] {
            margin-bottom: 0 !important;
        }

        /* Keep major headings attached to their following paragraph across page breaks.
           Only h1/h2 — applying to h3-h6 forces too many blocks to stay together
           and leaves large whitespace gaps at the bottom of pages. */
        h1, h2 {
            break-after: avoid;
            page-break-after: avoid;
        }
    """

    def _render_pdf() -> bytes:
        from weasyprint import CSS
        return WeasyprintHTML(
            string=html_content,
            base_url=".",
        ).write_pdf(
            stylesheets=[CSS(string=a4_css)]
        )

    try:
        pdf_bytes = await asyncio.to_thread(_render_pdf)
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
