"""
t_intent.py

Intent classification for document type detection.

When document_type is not provided in the request payload,
this module uses an LLM call to detect the document type
from the user_query text.

Import:
  from t_intent import classify_intent, INTENT_EXAMPLES
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported slugs — must match SUPPORTED_DOCUMENT_TYPES in t_document_generator.py
# ---------------------------------------------------------------------------

SUPPORTED_SLUGS: list[str] = [
    "nda",
    "job_offer",
    "freelancer_agreement",
    "service_agreement",
    "consulting_agreement",
    "lease_agreement",
    "employment_contract",
]

# ---------------------------------------------------------------------------
# Type aliases for resolve_document_type fallback
# ---------------------------------------------------------------------------

TYPE_ALIASES: dict[str, str] = {
    "non disclosure agreement":  "nda",
    "non-disclosure agreement":  "nda",
    "non_disclosure_agreement":  "nda",
    "confidentiality agreement": "nda",
    "job offer":                 "job_offer",
    "offer letter":              "job_offer",
    "job offer letter":          "job_offer",
    "appointment letter":        "job_offer",
    "freelance agreement":       "freelancer_agreement",
    "freelancer":                "freelancer_agreement",
    "independent contractor":    "freelancer_agreement",
    "service":                   "service_agreement",
    "vendor agreement":          "service_agreement",
    "amc":                       "service_agreement",
    "consulting":                "consulting_agreement",
    "consultant agreement":      "consulting_agreement",
    "retainer":                  "consulting_agreement",
    "advisory":                  "consulting_agreement",
    "lease":                     "lease_agreement",
    "rental agreement":          "lease_agreement",
    "leave and licence":         "lease_agreement",
    "rent agreement":            "lease_agreement",
    "tenancy":                   "lease_agreement",
    "employment":                "employment_contract",
    "employment agreement":      "employment_contract",
    "service contract":          "employment_contract",
}

# ---------------------------------------------------------------------------
# Intent classification system prompt
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT: str = """You are a legal document classifier for Indian legal documents.
Read the user description and return ONLY the slug of the matching document type.

Supported document types and their slugs:
- nda                  : Non-Disclosure Agreement, confidentiality agreement, NDA
- job_offer            : Job offer letter, appointment letter, offer of employment
- freelancer_agreement : Freelancer contract, independent contractor agreement, project contract
- service_agreement    : Service agreement, vendor agreement, SLA, AMC, maintenance contract
- consulting_agreement : Consulting agreement, advisory agreement, retainer agreement
- lease_agreement      : Lease agreement, rent agreement, leave and licence, rental deed, tenancy agreement
- employment_contract  : Employment contract, employment agreement, permanent appointment, work contract

Classification rules:
- Return ONLY the slug (e.g. "nda") — nothing else
- Do NOT add explanation, punctuation, or extra words
- If a freelancer is hired for a SHORT project → freelancer_agreement
- If a person is hired as PERMANENT/FULL-TIME employee → employment_contract
- If a company provides ONGOING services → service_agreement
- If an expert is engaged for ADVISORY/STRATEGIC work → consulting_agreement
- If none match confidently → return "unknown"

Examples:
"NDA between two companies" → nda
"Hire John as full-time software engineer" → employment_contract
"Hire John as freelance designer for logo project" → freelancer_agreement
"Rent my apartment to tenant" → lease_agreement
"Engage strategy consultant for 3 months" → consulting_agreement
"Cloud hosting service agreement with SLA" → service_agreement
"Job offer letter for marketing manager" → job_offer"""

# ---------------------------------------------------------------------------
# Few-shot examples (used in the user message for better accuracy)
# ---------------------------------------------------------------------------

INTENT_EXAMPLES: list[dict[str, str]] = [
    {"query": "I need an NDA with my co-founder before sharing business plans",
     "slug":  "nda"},
    {"query": "Appointment letter for Priya joining as Senior Developer at 18 LPA CTC",
     "slug":  "job_offer"},
    {"query": "Freelance agreement with a UI/UX designer for 6-week mobile app redesign project",
     "slug":  "freelancer_agreement"},
    {"query": "Service agreement with cloud vendor for 99.9% uptime SLA",
     "slug":  "service_agreement"},
    {"query": "Consulting retainer with a CFO advisor for 6 months at 1.5L per month",
     "slug":  "consulting_agreement"},
    {"query": "Rent agreement for 2BHK flat in Bangalore, 25000/month, 3 months deposit",
     "slug":  "lease_agreement"},
    {"query": "Employment contract for Rahul as Operations Manager, 22 LPA, Delhi office",
     "slug":  "employment_contract"},
]


def _build_examples_text() -> str:
    lines = []
    for ex in INTENT_EXAMPLES:
        lines.append(f'"{ex["query"]}" → {ex["slug"]}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def classify_intent(user_query: str) -> str | None:
    """
    Classify user_query into a document type slug using an LLM call.

    Returns:
        str  — canonical slug (e.g. "nda", "job_offer")
        None — if classification fails or returns "unknown"
    """
    from openai import AsyncOpenAI

    model   = os.environ.get("MODEL_NAME", "gpt-5-nano")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    client  = AsyncOpenAI(api_key=api_key)

    # Model-specific kwargs
    _NO_TEMP = {"gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini", "o1", "o1-mini", "o3-mini", "o3"}
    _MCT     = {"gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini", "o1", "o1-mini", "o3-mini", "o3"}

    kwargs: dict = {"model": model}
    if model not in _NO_TEMP:
        kwargs["temperature"] = 0.0
    if model in _MCT:
        kwargs["max_completion_tokens"] = 20
    else:
        kwargs["max_tokens"] = 20

    examples_text = _build_examples_text()

    kwargs["messages"] = [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Additional examples for reference:\n{examples_text}\n\n"
            f"Now classify this:\n\"\"\"{user_query[:800]}\"\"\""
        )},
    ]

    try:
        response = await client.chat.completions.create(**kwargs)
        raw      = (response.choices[0].message.content or "").strip().lower()
        raw      = re.sub(r'[^a-z_]', '', raw)

        logger.info(f"[intent] raw classification result: '{raw}'")

        # Direct slug match
        if raw in SUPPORTED_SLUGS:
            logger.info(f"[intent] classified as: '{raw}'")
            return raw

        # Alias match
        resolved = _resolve_alias(raw)
        if resolved:
            logger.info(f"[intent] resolved via alias: '{raw}' → '{resolved}'")
            return resolved

        logger.warning(f"[intent] unknown classification: '{raw}'")
        return None

    except Exception as e:
        logger.error(f"[intent] classification failed: {e}")
        return None


def _resolve_alias(raw: str) -> str | None:
    """
    Try to resolve a raw string to a document type slug via aliases.
    """
    # Direct alias lookup
    if raw in TYPE_ALIASES:
        return TYPE_ALIASES[raw]

    # Partial alias match
    for alias, slug in TYPE_ALIASES.items():
        if alias in raw or raw in alias:
            return slug

    # Partial slug match
    for slug in SUPPORTED_SLUGS:
        if slug in raw or raw in slug:
            return slug

    return None


def resolve_document_type(raw: str) -> str | None:
    """
    Normalise a user-supplied document type string to a canonical slug.
    Used when document_type IS provided in the request.

    Returns None if unrecognised.
    """
    cleaned  = raw.lower().strip().replace("-", "_").replace(" ", "_")

    if cleaned in SUPPORTED_SLUGS:
        return cleaned

    readable = raw.lower().strip()
    if readable in TYPE_ALIASES:
        return TYPE_ALIASES[readable]

    resolved = _resolve_alias(cleaned)
    if resolved:
        return resolved

    resolved = _resolve_alias(readable)
    if resolved:
        return resolved

    return None