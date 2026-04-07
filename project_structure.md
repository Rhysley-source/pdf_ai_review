# Django Project Structure

```
├── ./
│   ├── .gitignore
│   ├── s_db.py
│   ├── s_json_utils.py
│   ├── t_document_generator.py
│   ├── t_risk_detection.py
│   ├── t_prompts.py
│   ├── t_intent.py
│   ├── t_key_clause_extraction.py
│   ├── s_pdf_utils.py
│   ├── t_document_route.py
│   ├── s_ai_model.py
│   ├── s_pdf_to_docx.py
│   ├── requirements.txt
│   ├── export.py
│   ├── access.log
│   ├── server.log
│   ├── s_main.py
│   ├── s_route.py
│   ├── document_analyzer/
│   ├── routes/
│   │   ├── s_route.py
│   ├── document_generation/
│   │   ├── prompt_templates.py
│   │   ├── document_generator.py
│   ├── temp/
│   │   ├── 9e3152b6-77b3-4672-b854-7f6198a2e29d.pdf
│   │   ├── 23a6a41c-5232-44c6-90ed-04cbb76c8ebb.pdf
│   │   ├── be700235-b224-4b25-9bc6-04e17c6c7505.pdf
│   ├── backupfiles/
│   │   ├── pdf_utils.py
│   │   ├── t_ai_model.py
│   │   ├── pdf_utils_backup
│   │   ├── main_backup
│   │   ├── t_utils.py
│   │   ├── t_main.py
│   │   ├── ai_model_backup.py
│   │   ├── t_pdf_utils.py
```

# File Contents

---
File: .gitignore
---

```text
ai_env/

__pycache__/

.env
/.env

.venv/
.venv


```

---
File: s_db.py
---

```py
"""
s_db.py

PostgreSQL database layer.

Tables:
  pdf_requests      — PDF analysis request logs
  document_requests — Legal document generation logs

Install:
  pip install asyncpg

.env:
  DATABASE_URL=postgresql://user:password@localhost:5432/dbname
"""

import os
import logging
import asyncpg

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

_pool: asyncpg.Pool | None = None


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set. "
                "Add it to your .env file."
            )
        _pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("[db] Connection pool created")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("[db] Connection pool closed")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CREATE_TABLES_SQL = """
-- PDF analysis requests
CREATE TABLE IF NOT EXISTS pdf_requests (
    id                SERIAL PRIMARY KEY,
    request_id        TEXT        NOT NULL,
    pdf_name          TEXT        NOT NULL,
    pdf_size_bytes    INTEGER,
    total_pages       INTEGER,
    pages_analysed    INTEGER,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    total_tokens      INTEGER,
    completion_time_s NUMERIC(10, 3),
    endpoint          TEXT,
    status            TEXT,
    error_message     TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pdf_requests_created_at
    ON pdf_requests (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_pdf_requests_request_id
    ON pdf_requests (request_id);

-- Legal document generation requests
CREATE TABLE IF NOT EXISTS document_requests (
    id                  SERIAL PRIMARY KEY,
    request_id          TEXT        NOT NULL,
    document_type       TEXT        NOT NULL,
    document_name       TEXT        NOT NULL,
    type_was_detected   BOOLEAN     DEFAULT FALSE,
    user_query          TEXT,
    status              TEXT,
    missing_fields      TEXT[],
    word_count          INTEGER,
    docx_size_bytes     INTEGER,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    total_tokens        INTEGER,
    completion_time_s   NUMERIC(10, 3),
    endpoint            TEXT,
    error_message       TEXT,
    fields              JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_requests_created_at
    ON document_requests (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_document_requests_request_id
    ON document_requests (request_id);

CREATE INDEX IF NOT EXISTS idx_document_requests_document_type
    ON document_requests (document_type);

CREATE INDEX IF NOT EXISTS idx_document_requests_status
    ON document_requests (status);
"""


async def init_db() -> None:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(CREATE_TABLES_SQL)
        logger.info("[db] Schema initialised (pdf_requests + document_requests tables ready)")
    except Exception as e:
        logger.error(f"[db] Schema init failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Log PDF analysis request
# ---------------------------------------------------------------------------

async def log_request(
    request_id:        str,
    pdf_name:          str,
    pdf_size_bytes:    int   = 0,
    total_pages:       int   = 0,
    pages_analysed:    int   = 0,
    input_tokens:      int   = 0,
    output_tokens:     int   = 0,
    completion_time_s: float = 0.0,
    endpoint:          str   = "/analyze",
    status:            str   = "success",
    error_message:     str | None = None,
) -> None:
    total_tokens = input_tokens + output_tokens
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO pdf_requests (
                    request_id, pdf_name, pdf_size_bytes,
                    total_pages, pages_analysed,
                    input_tokens, output_tokens, total_tokens,
                    completion_time_s, endpoint, status, error_message
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                )
                """,
                request_id, pdf_name, pdf_size_bytes,
                total_pages, pages_analysed,
                input_tokens, output_tokens, total_tokens,
                round(completion_time_s, 3),
                endpoint, status, error_message,
            )
        logger.info(
            f"[db] pdf_request logged — id={request_id} "
            f"pdf='{pdf_name}' tokens={total_tokens} "
            f"time={completion_time_s:.2f}s status={status}"
        )
    except Exception as e:
        logger.error(f"[db] log_request failed for {request_id}: {e}")


# ---------------------------------------------------------------------------
# Log document generation request
# ---------------------------------------------------------------------------

async def log_document_request(
    request_id:        str,
    document_type:     str,
    document_name:     str,
    user_query:        str        = "",
    status:            str        = "success",
    type_was_detected: bool       = False,
    missing_fields:    list[str]  | None = None,
    word_count:        int        = 0,
    docx_size_bytes:   int        = 0,
    input_tokens:      int        = 0,
    output_tokens:     int        = 0,
    completion_time_s: float      = 0.0,
    endpoint:          str        = "/documents/generate",
    error_message:     str | None = None,
    fields:            dict       | None = None,
) -> None:
    """
    Log a document generation request to the document_requests table.
    Never raises — DB errors are logged but do not break the API.
    """
    import json as _json

    total_tokens   = input_tokens + output_tokens
    missing        = missing_fields or []
    fields_json    = _json.dumps(fields) if fields else None

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO document_requests (
                    request_id, document_type, document_name,
                    type_was_detected, user_query, status,
                    missing_fields, word_count, docx_size_bytes,
                    input_tokens, output_tokens, total_tokens,
                    completion_time_s, endpoint, error_message, fields
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $9,
                    $10, $11, $12,
                    $13, $14, $15, $16
                )
                """,
                request_id,
                document_type,
                document_name,
                type_was_detected,
                user_query[:2000] if user_query else "",   # cap at 2000 chars
                status,
                missing,
                word_count,
                docx_size_bytes,
                input_tokens,
                output_tokens,
                total_tokens,
                round(completion_time_s, 3),
                endpoint,
                error_message,
                fields_json,
            )
        logger.info(
            f"[db] document_request logged — id={request_id} "
            f"type={document_type} words={word_count} "
            f"tokens={total_tokens} time={completion_time_s:.2f}s "
            f"status={status} missing={missing}"
        )
    except Exception as e:
        logger.error(f"[db] log_document_request failed for {request_id}: {e}")


# ---------------------------------------------------------------------------
# Query helpers (optional — for admin/stats endpoints)
# ---------------------------------------------------------------------------

async def get_document_stats() -> dict:
    """
    Returns aggregate stats for document generation.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    document_type,
                    COUNT(*)                          AS total,
                    COUNT(*) FILTER (WHERE status = 'success')        AS success_count,
                    COUNT(*) FILTER (WHERE status = 'missing_fields') AS missing_fields_count,
                    COUNT(*) FILTER (WHERE status = 'error')          AS error_count,
                    ROUND(AVG(word_count))            AS avg_words,
                    ROUND(AVG(completion_time_s), 2)  AS avg_time_s,
                    SUM(total_tokens)                 AS total_tokens
                FROM document_requests
                GROUP BY document_type
                ORDER BY total DESC
            """)
            total_row = await conn.fetchrow("""
                SELECT
                    COUNT(*)                         AS total_requests,
                    SUM(total_tokens)                AS total_tokens,
                    ROUND(AVG(completion_time_s), 2) AS avg_time_s
                FROM document_requests
            """)
        return {
            "by_type": [dict(r) for r in rows],
            "overall": dict(total_row) if total_row else {},
        }
    except Exception as e:
        logger.error(f"[db] get_document_stats failed: {e}")
        return {"by_type": [], "overall": {}}


async def get_recent_documents(limit: int = 20) -> list[dict]:
    """
    Returns recent document generation requests.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    request_id, document_type, document_name,
                    status, missing_fields, word_count,
                    input_tokens, output_tokens, total_tokens,
                    completion_time_s, type_was_detected, created_at
                FROM document_requests
                ORDER BY created_at DESC
                LIMIT $1
            """, limit)
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"[db] get_recent_documents failed: {e}")
        return []
```

---
File: s_json_utils.py
---

```py
"""
s_json_utils.py

All JSON extraction helpers used by s_ai_model.py, s_route.py, and any
other module that needs to parse LLM output.

Moved here so s_ai_model.py does NOT need to import from s_main/s_route,
which caused a circular-import / missing-symbol error after the router
refactor.
"""

import re
import json
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level string fixes
# ---------------------------------------------------------------------------

def _fix_json_string(raw: str) -> str:
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    return raw


def _extract_field_by_regex(text: str, field: str) -> str:
    pattern = rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_highlights_by_regex(text: str) -> list:
    match = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        match_open = re.search(r'"highlights"\s*:\s*\[(.*)', text, re.DOTALL)
        if not match_open:
            return []
        inner = match_open.group(1)
    else:
        inner = match.group(1)
    items = re.findall(r'"((?:[^"\\]|\\.)*)"', inner, re.DOTALL)
    return [i.strip() for i in items if i.strip()]


# ---------------------------------------------------------------------------
# Field flatteners — handle nested dicts/lists the LLM sometimes returns
# ---------------------------------------------------------------------------

def _flatten_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_flatten_field(v) for v in value]
        return ". ".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            flat = _flatten_field(v)
            if flat:
                label = str(k).replace("_", " ").capitalize()
                parts.append(f"{label}: {flat}")
        return ". ".join(parts)
    return str(value).strip()


def _flatten_highlights(value) -> list:
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            flat = _flatten_field(v)
            if flat:
                label = str(k).replace("_", " ").capitalize()
                items.append(f"{label}: {flat}")
        return items
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
            elif isinstance(item, dict):
                flat = _flatten_field(item)
                if flat:
                    result.append(flat)
            elif item is not None:
                s = str(item).strip()
                if s:
                    result.append(s)
        return result
    return []


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _postprocess_highlights(data: dict) -> dict:
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            if isinstance(item, dict):
                text = ""
                for key in ("fact", "highlight", "text", "value", "item", "point"):
                    if isinstance(item.get(key), str):
                        text = item[key]
                        break
                if not text:
                    text = next((v for v in item.values() if isinstance(v, str)), "")
                if text:
                    cleaned.append(text.replace('"', '').strip())
                continue
            if not isinstance(item, str):
                continue
            parts = item.split('", "') if '", "' in item else [item]
            for p in parts:
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
    return data


def _normalize_parsed(data, label: str = "") -> dict:
    empty = {"overview": "", "summary": "", "highlights": []}

    if isinstance(data, list):
        if not data:
            return empty
        if isinstance(data[0], dict):
            merged: dict = {"overview": "", "summary": "", "highlights": []}
            for item in data:
                if not isinstance(item, dict):
                    continue
                if not merged["overview"] and item.get("overview"):
                    merged["overview"] = item["overview"]
                if not merged["summary"] and item.get("summary"):
                    merged["summary"] = item["summary"]
                merged["highlights"].extend(item.get("highlights") or [])
            logger.warning(f"extract_json {label}: got list of dicts — merged {len(data)} item(s)")
            data = merged
        elif isinstance(data[0], str):
            return {"overview": "", "summary": "", "highlights": [s for s in data if s]}
        else:
            return empty

    if not isinstance(data, dict):
        logger.error(f"extract_json {label}: unexpected type {type(data).__name__}")
        return empty

    overview = data.get("overview", "")
    if not isinstance(overview, str):
        logger.warning(f"extract_json {label}: overview is {type(overview).__name__} — flattening")
        overview = _flatten_field(overview)

    summary = data.get("summary", "")
    if not isinstance(summary, str):
        logger.warning(f"extract_json {label}: summary is {type(summary).__name__} — flattening")
        summary = _flatten_field(summary)

    raw_highlights = data.get("highlights", [])
    if not isinstance(raw_highlights, list) or (
        raw_highlights and not isinstance(raw_highlights[0], str)
    ):
        logger.warning(f"extract_json {label}: highlights has non-string items — flattening")
        highlights = _flatten_highlights(raw_highlights)
    else:
        highlights = raw_highlights

    return {
        "overview":   overview.strip(),
        "summary":    summary.strip(),
        "highlights": highlights,
    }


def _recover_truncated_json(text: str) -> dict | None:
    result = {"overview": "", "summary": "", "highlights": []}
    ov_match = re.search(r'"overview"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if ov_match:
        result["overview"] = ov_match.group(1).strip()
    sm_match = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if sm_match:
        result["summary"] = sm_match.group(1).strip()
    result["highlights"] = _extract_highlights_by_regex(text)
    if result["overview"] or result["summary"] or result["highlights"]:
        logger.info(
            f"_recover_truncated_json: recovered "
            f"overview={'yes' if result['overview'] else 'no'} "
            f"summary={'yes' if result['summary'] else 'no'} "
            f"highlights={len(result['highlights'])}"
        )
        return result
    return None


# ---------------------------------------------------------------------------
# Public API — the one function everyone imports
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """
    Robustly extract JSON from LLM output.

    Strategy 1 — direct parse after cleaning
    Strategy 2 — isolate { } block + quote-escape repair
    Strategy 3 — truncation recovery
    Strategy 4 — per-field regex (last resort)
    """
    empty = {"overview": "", "summary": "", "highlights": []}
    if not text or not text.strip():
        logger.warning("extract_json: received empty text")
        return empty

    cleaned = _fix_json_string(text)

    try:
        data = json.loads(cleaned)
        logger.debug("extract_json: strategy 1 succeeded")
        return _postprocess_highlights(_normalize_parsed(data, "strategy-1"))
    except json.JSONDecodeError as e:
        logger.debug(f"extract_json: strategy 1 failed — {e}")

    brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            data = json.loads(candidate)
            return _postprocess_highlights(_normalize_parsed(data, "strategy-2"))
        except json.JSONDecodeError as e:
            logger.warning(f"extract_json: strategy 2 failed ({e}), trying repair")
            repaired = re.sub(r'(?<=[^\\])"(?=[^,\]}\n:}{\[])', r'\\"', candidate)
            try:
                data = json.loads(repaired)
                return _postprocess_highlights(_normalize_parsed(data, "strategy-2-repair"))
            except json.JSONDecodeError as e2:
                logger.warning(f"extract_json: strategy 2 repair failed — {e2}")
    else:
        logger.warning("extract_json: no brace block found in output")

    recovered = _recover_truncated_json(cleaned)
    if recovered:
        logger.info("extract_json: strategy 3 (truncation recovery) succeeded")
        return _postprocess_highlights(recovered)

    logger.error("extract_json: all strategies failed — falling back to regex")
    overview   = _extract_field_by_regex(cleaned, "overview")
    summary    = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)
    if overview or summary or highlights:
        return {"overview": overview, "summary": summary, "highlights": highlights}

    logger.error("extract_json: all strategies exhausted — returning empty")
    return empty
```

---
File: t_document_generator.py
---

```py
"""
t_document_generator.py

Business-level legal document generation engine — India jurisdiction.

This file contains ONLY the core generation logic.
Prompts   → t_prompts.py
Intent    → t_intent.py
DB        → s_db.py
Routes    → t_document_route.py
"""

import os
import re
import json
import logging
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI

# Import prompts and intent from separate modules
from t_prompts import EXTRACTION_PROMPTS, GENERATION_PROMPTS
from t_intent  import classify_intent, resolve_document_type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

_MODEL   = os.environ.get("MODEL_NAME", "gpt-5-nano")
_API_KEY = os.environ.get("OPENAI_API_KEY", "")

_NO_TEMP = {"gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini", "o1", "o1-mini", "o3-mini", "o3"}
_MCT     = {"gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini", "o1", "o1-mini", "o3-mini", "o3"}


def _get_client() -> AsyncOpenAI:
    key = os.environ.get("OPENAI_API_KEY", _API_KEY)
    return AsyncOpenAI(api_key=key)


def _api_kwargs(max_tokens: int, use_json: bool = False) -> dict:
    """
    Build OpenAI API kwargs.
    use_json=True  → field extraction   (response_format: json_object)
    use_json=False → document generation (plain text, NO response_format)

    IMPORTANT: Never set response_format for document generation —
    gpt-5-nano returns empty content when json_object is set on plain-text calls.
    """
    model  = os.environ.get("MODEL_NAME", _MODEL)
    kwargs: dict = {"model": model}
    if model not in _NO_TEMP:
        kwargs["temperature"] = 0.1
    if model in _MCT:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


# ---------------------------------------------------------------------------
# Supported document types
# ---------------------------------------------------------------------------

SUPPORTED_DOCUMENT_TYPES: dict[str, str] = {
    "nda":                   "Non-Disclosure Agreement",
    "job_offer":             "Job Offer Letter",
    "freelancer_agreement":  "Freelancer Agreement",
    "service_agreement":     "Service Agreement",
    "consulting_agreement":  "Consulting Agreement",
    "lease_agreement":       "Lease Agreement",
    "employment_contract":   "Employment Contract",
}

# ---------------------------------------------------------------------------
# Field schemas
# ---------------------------------------------------------------------------

_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "nda": {
        "required": ["disclosing_party", "receiving_party", "effective_date",
                     "purpose", "duration_years", "governing_state"],
        "optional": ["confidential_info_description", "exclusions",
                     "return_of_materials", "remedies", "signatory_names",
                     "stamp_duty_state"],
    },
    "job_offer": {
        "required": ["candidate_name", "company_name", "job_title",
                     "start_date", "ctc", "employment_type", "reporting_to"],
        "optional": ["work_location", "probation_period", "benefits",
                     "esop", "joining_bonus", "offer_expiry_date",
                     "hr_contact", "notice_period"],
    },
    "freelancer_agreement": {
        "required": ["client_name", "freelancer_name", "project_description",
                     "start_date", "end_date", "payment_amount",
                     "payment_schedule", "governing_state"],
        "optional": ["deliverables", "revision_rounds", "intellectual_property",
                     "confidentiality", "kill_fee", "late_payment_penalty",
                     "gst_applicable"],
    },
    "service_agreement": {
        "required": ["service_provider", "client", "services_description",
                     "start_date", "end_date", "fee", "payment_terms",
                     "governing_state"],
        "optional": ["service_levels", "termination_notice",
                     "limitation_of_liability", "indemnification",
                     "insurance_requirements", "dispute_resolution",
                     "gst_number"],
    },
    "consulting_agreement": {
        "required": ["consultant_name", "client_name", "scope_of_work",
                     "start_date", "end_date", "consulting_fee",
                     "payment_terms", "governing_state"],
        "optional": ["expenses_reimbursement", "non_compete_period",
                     "non_solicitation", "intellectual_property",
                     "confidentiality_period", "termination_clause",
                     "tds_applicable"],
    },
    "lease_agreement": {
        "required": ["landlord_name", "tenant_name", "property_address",
                     "lease_start_date", "lease_end_date", "monthly_rent",
                     "security_deposit", "governing_state"],
        "optional": ["maintenance_charges", "pet_policy",
                     "utilities_included", "lock_in_period",
                     "subletting_policy", "renewal_terms",
                     "notice_period", "stamp_duty_value"],
    },
    "employment_contract": {
        "required": ["employer_name", "employee_name", "job_title",
                     "department", "start_date", "ctc",
                     "work_hours", "governing_state"],
        "optional": ["probation_period", "notice_period", "non_compete",
                     "non_solicitation", "benefits", "leave_policy",
                     "termination_clause", "intellectual_property_assignment",
                     "pf_applicable", "gratuity_applicable"],
    },
}


# ---------------------------------------------------------------------------
# Own JSON extractor — no dependency on s_json_utils
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    text = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    logger.warning("[doc_gen] could not parse JSON from model response")
    return {}


# ---------------------------------------------------------------------------
# Step 1: Extract fields using EXTRACTION_PROMPTS from t_prompts.py
# ---------------------------------------------------------------------------

async def _extract_fields(document_type: str, user_query: str) -> dict[str, Any]:
    schema = _SCHEMAS[document_type]
    system = EXTRACTION_PROMPTS[document_type]   # ← from t_prompts.py

    client = _get_client()
    # use_json=True → sets response_format=json_object for reliable JSON output
    kwargs = _api_kwargs(max_tokens=4096, use_json=True)
    kwargs["messages"] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": (
            f"Extract all fields from this description as a JSON object.\n\n"
            f"Description:\n\"\"\"{user_query}\"\"\""
        )},
    ]

    response = await client.chat.completions.create(**kwargs)
    raw      = response.choices[0].message.content or "{}"

    logger.info(
        f"[doc_gen] field extraction "
        f"tokens={response.usage.prompt_tokens}in/{response.usage.completion_tokens}out "
        f"finish={response.choices[0].finish_reason}"
    )
    logger.debug(f"[doc_gen] extraction raw: {raw[:300]}")

    fields = _parse_json(raw)

    # Ensure all required fields exist
    for field in schema["required"]:
        if field not in fields or not fields[field]:
            fields[field] = "Not Specified"

    found = {k: v for k, v in fields.items() if v != "Not Specified"}
    logger.info(f"[doc_gen] extracted {len(found)} non-empty fields: {list(found.keys())}")

    return fields


# ---------------------------------------------------------------------------
# Step 2: Validate required fields
# ---------------------------------------------------------------------------

def _get_missing_fields(document_type: str, fields: dict) -> list[str]:
    return [
        f for f in _SCHEMAS[document_type]["required"]
        if not fields.get(f) or fields.get(f) == "Not Specified"
    ]


# ---------------------------------------------------------------------------
# Step 3: Generate document text using GENERATION_PROMPTS from t_prompts.py
# ---------------------------------------------------------------------------

async def _generate_document_text(
    document_type: str,
    fields: dict[str, Any],
    user_query: str,
) -> tuple[str, int, int]:
    doc_name   = SUPPORTED_DOCUMENT_TYPES[document_type]
    gen_prompt = GENERATION_PROMPTS[document_type]   # ← from t_prompts.py

    field_lines = "\n".join(
        f"  {k.replace('_', ' ').title()}: {v}"
        for k, v in fields.items()
        if v and v != "Not Specified"
    )

    system = gen_prompt

    user = f"""Use these details to generate the complete {doc_name} under Indian law:

{field_lines if field_lines else "(Use standard Indian law template values)"}

Additional context from user:
\"\"\"{user_query}\"\"\"

Write the complete {doc_name} now. Start directly with the document title."""

    client = _get_client()
    # CRITICAL: use_json=False — never set response_format for plain text generation
    kwargs = _api_kwargs(max_tokens=32000, use_json=False)
    kwargs["messages"] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    response      = await client.chat.completions.create(**kwargs)
    document_text = response.choices[0].message.content or ""
    in_tok        = response.usage.prompt_tokens
    out_tok       = response.usage.completion_tokens
    finish        = response.choices[0].finish_reason

    logger.info(
        f"[doc_gen] document generated — type={document_type} "
        f"length={len(document_text)} chars "
        f"tokens={in_tok}in/{out_tok}out finish={finish}"
    )

    if not document_text.strip():
        logger.error(
            f"[doc_gen] empty response — finish={finish} "
            f"model={os.environ.get('MODEL_NAME', _MODEL)}"
        )

    return document_text, in_tok, out_tok


# ---------------------------------------------------------------------------
# Step 4: Render DOCX
# ---------------------------------------------------------------------------

def _render_docx(document_text: str, document_name: str) -> bytes:
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from datetime import datetime

    doc = DocxDocument()

    for section in doc.sections:
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.25)
        section.right_margin  = Inches(1.25)

    BLUE  = RGBColor(0x2c, 0x4a, 0x8c)
    DARK  = RGBColor(0x1a, 0x1a, 0x1a)
    MUTED = RGBColor(0x55, 0x55, 0x55)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)

    def _set_cell_bg(cell, hex_color: str):
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd  = OxmlElement("w:shd")
        shd.set(qn("w:val"),   "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"),  hex_color)
        tcPr.append(shd)

    def _remove_table_borders(table):
        tbl   = table._tbl
        tblPr = tbl.find(qn("w:tblPr"))
        if tblPr is None:
            tblPr = OxmlElement("w:tblPr")
            tbl.insert(0, tblPr)
        tblBorders = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            b = OxmlElement(f"w:{side}")
            b.set(qn("w:val"), "none")
            tblBorders.append(b)
        tblPr.append(tblBorders)

    # Header banner
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell  = table.cell(0, 0)
    _set_cell_bg(cell, "1a2744")
    _remove_table_borders(table)
    para = cell.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run  = para.add_run(document_name.upper())
    run.bold           = True
    run.font.size      = Pt(20)
    run.font.color.rgb = WHITE

    para2 = cell.add_paragraph()
    para2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2  = para2.add_run("Governed by the Laws of India")
    run2.font.size      = Pt(9)
    run2.font.color.rgb = RGBColor(0xCC, 0xD6, 0xF0)

    doc.add_paragraph()

    dp  = doc.add_paragraph()
    dr  = dp.add_run(f"Generated: {datetime.now().strftime('%d %B %Y')}")
    dr.font.size      = Pt(9)
    dr.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    dp.paragraph_format.space_after = Pt(12)

    lines        = document_text.split("\n")
    in_signature = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            sp = doc.add_paragraph()
            sp.paragraph_format.space_before = Pt(0)
            sp.paragraph_format.space_after  = Pt(3)
            continue

        if re.match(r'^\d+\.\s+SIGNATURE', stripped.upper()):
            in_signature = True

        is_section = (
            re.match(r'^\d+\.\s+[A-Z][A-Z\s,/&().-]{3,}[:.]?\s*$', stripped)
            or re.match(r'^[A-Z][A-Z\s,/&().-]{3,}[:.]?\s*$', stripped)
            or (stripped.isupper() and 4 < len(stripped) < 80)
        )

        if is_section and not in_signature:
            p    = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after  = Pt(4)
            pPr  = p._p.get_or_add_pPr()
            pBdr = OxmlElement("w:pBdr")
            top  = OxmlElement("w:top")
            top.set(qn("w:val"),   "single")
            top.set(qn("w:sz"),    "4")
            top.set(qn("w:space"), "1")
            top.set(qn("w:color"), "CCCCCC")
            pBdr.append(top)
            pPr.append(pBdr)
            run = p.add_run(stripped)
            run.bold           = True
            run.font.size      = Pt(11)
            run.font.color.rgb = BLUE
            continue

        if re.match(r'^\d+\.\d+\.?\s+', stripped) and not in_signature:
            p   = doc.add_paragraph(style="Normal")
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(stripped)
            run.font.size      = Pt(10)
            run.font.color.rgb = DARK
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            continue

        if in_signature and re.match(
            r'^(Name|Title|Date|Company|Designation|Signature|Witness|Place|PAN|Aadhaar)\s*:',
            stripped, re.I
        ):
            label, _, rest = stripped.partition(":")
            lp  = doc.add_paragraph()
            lr  = lp.add_run(label.strip() + ":")
            lr.bold            = True
            lr.font.size       = Pt(9)
            lr.font.color.rgb  = MUTED
            lp.paragraph_format.space_after = Pt(2)
            vp  = doc.add_paragraph()
            vr  = vp.add_run(rest.strip() if rest.strip() else "________________________")
            vr.font.size      = Pt(10)
            vr.font.color.rgb = DARK
            vp.paragraph_format.space_after = Pt(12)
            continue

        p   = doc.add_paragraph(style="Normal")
        run = p.add_run(stripped)
        run.font.size      = Pt(10)
        run.font.color.rgb = DARK
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(4)

    # Footer
    for section in doc.sections:
        footer      = section.footer
        footer_para = footer.paragraphs[0]
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer_para.add_run(
            f"{document_name}  |  Governed by Laws of India  |  Confidential  |  Page "
        )
        run.font.size      = Pt(8)
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        fldChar1              = OxmlElement("w:fldChar")
        fldChar1.set(qn("w:fldCharType"), "begin")
        instrText             = OxmlElement("w:instrText")
        instrText.text        = "PAGE"
        fldChar2              = OxmlElement("w:fldChar")
        fldChar2.set(qn("w:fldCharType"), "end")
        run2 = footer_para.add_run()
        run2.font.size      = Pt(8)
        run2.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        run2._r.append(fldChar1)
        run2._r.append(instrText)
        run2._r.append(fldChar2)

    buffer     = BytesIO()
    doc.save(buffer)
    docx_bytes = buffer.getvalue()
    buffer.close()

    logger.info(f"[doc_gen] DOCX rendered — {len(docx_bytes):,} bytes")
    return docx_bytes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_document(
    document_type: str | None,
    user_query: str,
) -> dict:
    """
    Main entry point.

    document_type: canonical slug OR None (auto-detected via t_intent.classify_intent)
    user_query   : free-text description

    Returns dict with keys:
      status, document_type, document_name, fields, missing_fields,
      document, docx_bytes, word_count, input_tokens, output_tokens
    """
    total_in_tok  = 0
    total_out_tok = 0

    # ── Resolve or auto-detect document type ──────────────────────────────
    if document_type:
        doc_type = resolve_document_type(document_type)   # ← from t_intent.py
        if not doc_type:
            return {
                "status":         "unknown_type",
                "document_type":  document_type,
                "document_name":  "",
                "fields":         {},
                "missing_fields": [],
                "document":       "",
                "docx_bytes":     b"",
                "word_count":     0,
                "input_tokens":   0,
                "output_tokens":  0,
                "message": (
                    f"'{document_type}' is not supported. "
                    f"Supported: {', '.join(SUPPORTED_DOCUMENT_TYPES.keys())}"
                ),
            }
    else:
        logger.info("[doc_gen] document_type not provided — classifying intent via t_intent...")
        doc_type = await classify_intent(user_query)      # ← from t_intent.py
        if not doc_type:
            return {
                "status":         "unknown_type",
                "document_type":  None,
                "document_name":  "",
                "fields":         {},
                "missing_fields": [],
                "document":       "",
                "docx_bytes":     b"",
                "word_count":     0,
                "input_tokens":   total_in_tok,
                "output_tokens":  total_out_tok,
                "message": (
                    "Could not determine document type from description. "
                    "Please specify document_type explicitly. "
                    f"Supported: {', '.join(SUPPORTED_DOCUMENT_TYPES.keys())}"
                ),
            }

    doc_name = SUPPORTED_DOCUMENT_TYPES[doc_type]
    logger.info(f"[doc_gen] generating '{doc_name}' ({len(user_query)} chars)")

    # ── Step 1: Extract fields ─────────────────────────────────────────────
    fields = await _extract_fields(doc_type, user_query)

    # ── Step 2: Check missing ──────────────────────────────────────────────
    missing = _get_missing_fields(doc_type, fields)
    if missing:
        logger.warning(f"[doc_gen] missing required fields: {missing}")

    # ── Step 3: Generate document text ────────────────────────────────────
    document_text, in_tok, out_tok = await _generate_document_text(
        doc_type, fields, user_query
    )
    total_in_tok  += in_tok
    total_out_tok += out_tok

    # ── Step 4: Render DOCX ───────────────────────────────────────────────
    docx_bytes = _render_docx(document_text, doc_name)

    return {
        "status":         "missing_fields" if missing else "success",
        "document_type":  doc_type,
        "document_name":  doc_name,
        "fields":         fields,
        "missing_fields": missing,
        "document":       document_text,
        "docx_bytes":     docx_bytes,
        "word_count":     len(document_text.split()),
        "input_tokens":   total_in_tok,
        "output_tokens":  total_out_tok,
    }
```

---
File: t_risk_detection.py
---

```py
# u_risk_detection.py
import logging
from s_ai_model import run_llm
from backupfiles.t_utils import extract_json_from_text

logger = logging.getLogger(__name__)

async def analyze_document_risks(text: str):
    """
    Scans document for legal/financial risks and returns structured JSON.
    Reuses: s_ai_model.run_llm
    """
    
    prompt = """
    Analyze the provided document text for legal and financial risks. 
    Identify clauses that are unfavorable to the signer (e.g., founders or business owners).

    Specific Risks to detect:
    - Auto-renewal: Clauses that commit the user to another term automatically.
    - Indemnity: Clauses where the user takes on heavy financial liability.
    - Termination Penalties: Excessive costs for exiting the contract.
    - Non-compete: Restrictions on future business or career moves.
    - Missing Liability Caps: No limit on how much the user might owe in damages.
    - Jurisdiction: Legal disputes happening in a far-away or unfavorable location.

    Output ONLY valid JSON in this format:
    {
      "risk_score": 0-100,
      "detected_risks": [
        {
          "risk_name": "Name of category",
          "severity": "High/Medium/Low",
          "summary": "1 sentence describing the clause",
          "impact": "Why this is dangerous",
          "mitigation": "How to negotiate or change this"
        }
      ],
      "overall_assessment": "Executive summary of the document's risk profile."
    }

    Document Text:
    ---
    """ + text[:12000] + "\n---"

    logger.info("[risk_detection] Running AI analysis...")
    
    # Reusing your core LLM runner
    raw_output = await run_llm(text, prompt)
    print("Raw LLM Output:", raw_output)  # Debugging: see the unprocessed output
    # Using the new generic JSON parser
    analysis = extract_json_from_text(raw_output)
    
    return {
        "status": "success",
        "analysis_type": "risk_detection",
        "data": analysis
    }

```

---
File: t_prompts.py
---

```py
"""
t_prompts.py

All LLM prompts for document generation — India jurisdiction.

Contains:
  EXTRACTION_PROMPTS   — per-document-type field extraction prompts
  GENERATION_PROMPTS   — per-document-type document drafting prompts

Import:
  from t_prompts import EXTRACTION_PROMPTS, GENERATION_PROMPTS
"""


# ---------------------------------------------------------------------------
# Field extraction prompts
# One prompt per document type.
# Each prompt instructs the LLM to return ONLY a JSON object.
# response_format=json_object is set on the API call so no markdown fences.
# ---------------------------------------------------------------------------

EXTRACTION_PROMPTS: dict[str, str] = {

    # ------------------------------------------------------------------
    "nda": """You are a legal assistant specialising in Indian Non-Disclosure Agreements.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
disclosing_party    - name of company/person sharing confidential info
receiving_party     - name of company/person receiving confidential info
effective_date      - when the NDA takes effect (DD/MM/YYYY or Month DD, YYYY)
purpose             - why confidential information is being shared
duration_years      - how long the NDA lasts (e.g. "2 years")
governing_state     - Indian state whose laws govern (e.g. "Maharashtra", "Karnataka")

Optional JSON keys:
confidential_info_description - what specific info is covered
exclusions          - what is NOT confidential
return_of_materials - must materials be returned on termination
remedies            - remedies for breach
signatory_names     - names of signatories
stamp_duty_state    - state for stamp duty purposes

Set any field not mentioned to "Not Specified". Return ONLY the JSON object.""",

    # ------------------------------------------------------------------
    "job_offer": """You are an HR specialist drafting Indian Job Offer / Appointment Letters.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
candidate_name  - full name of the candidate
company_name    - name of the company
job_title       - position being offered
start_date      - proposed date of joining (DD/MM/YYYY)
ctc             - Cost to Company annual in INR (e.g. "12,00,000 INR per annum")
employment_type - full-time / part-time / contract / internship
reporting_to    - manager or supervisor name/designation

Optional JSON keys:
work_location    - city and office address or remote
probation_period - probation duration (e.g. "6 months")
benefits         - PF, gratuity, health insurance, etc.
esop             - ESOP or stock options
joining_bonus    - one-time joining bonus in INR
offer_expiry_date - deadline to accept offer
hr_contact       - HR contact name and email
notice_period    - notice period after probation

Set any field not mentioned to "Not Specified". Return ONLY the JSON object.""",

    # ------------------------------------------------------------------
    "freelancer_agreement": """You are a contracts specialist drafting Indian Freelancer Agreements.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
client_name         - name of company/person hiring the freelancer
freelancer_name     - name of the freelancer
project_description - what work is being done
start_date          - when work begins
end_date            - when work is due
payment_amount      - total payment in INR (e.g. "50,000 INR")
payment_schedule    - when/how payment is made (milestone/on-completion/monthly)
governing_state     - Indian state governing the agreement

Optional JSON keys:
deliverables         - specific outputs expected
revision_rounds      - number of revision rounds included
intellectual_property - who owns the work product
confidentiality      - confidentiality requirements
kill_fee             - cancellation fee
late_payment_penalty - late payment penalty
gst_applicable       - whether GST is applicable and GST numbers

Set any field not mentioned to "Not Specified". Return ONLY the JSON object.""",

    # ------------------------------------------------------------------
    "service_agreement": """You are a contracts specialist drafting Indian Service Agreements.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
service_provider     - name of company/person providing services
client               - name of company/person receiving services
services_description - description of the services
start_date           - service start date
end_date             - service end date
fee                  - total fee or rate in INR
payment_terms        - payment schedule (Net 30, monthly advance, etc.)
governing_state      - Indian state governing the agreement

Optional JSON keys:
service_levels         - SLA or performance standards
termination_notice     - notice period to terminate
limitation_of_liability - liability cap amount
indemnification        - indemnification terms
insurance_requirements - required insurance coverage
dispute_resolution     - arbitration or court
gst_number             - GST registration numbers of both parties

Set any field not mentioned to "Not Specified". Return ONLY the JSON object.""",

    # ------------------------------------------------------------------
    "consulting_agreement": """You are a contracts specialist drafting Indian Consulting Agreements.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
consultant_name  - name of consultant or firm
client_name      - name of client company
scope_of_work    - description of consulting services
start_date       - engagement start date
end_date         - engagement end date
consulting_fee   - fee in INR (hourly/daily/project basis)
payment_terms    - how and when payment is made
governing_state  - Indian state governing the agreement

Optional JSON keys:
expenses_reimbursement - expense reimbursement policy
non_compete_period     - non-compete duration after engagement
non_solicitation       - non-solicitation clause
intellectual_property  - ownership of work product
confidentiality_period - how long confidentiality lasts
termination_clause     - termination conditions and notice
tds_applicable         - whether TDS is deductible and at what rate

Set any field not mentioned to "Not Specified". Return ONLY the JSON object.""",

    # ------------------------------------------------------------------
    "lease_agreement": """You are a real estate attorney drafting Indian Lease / Leave and Licence Agreements.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
landlord_name    - full name of landlord/licensor
tenant_name      - full name of tenant(s)/licensee(s)
property_address - full address including city, state, PIN code
lease_start_date - when the lease/licence begins
lease_end_date   - when the lease/licence ends
monthly_rent     - monthly rent in INR
security_deposit - security deposit in INR (usually 2-6 months rent)
governing_state  - Indian state where property is located

Optional JSON keys:
maintenance_charges    - monthly maintenance amount
pet_policy             - whether pets are allowed
utilities_included     - electricity, water, gas included or not
lock_in_period         - minimum lock-in period
subletting_policy      - whether subletting is allowed
renewal_terms          - how the lease can be renewed
notice_period          - notice required to vacate
stamp_duty_value       - stamp duty value for registration

Set any field not mentioned to "Not Specified". Return ONLY the JSON object.""",

    # ------------------------------------------------------------------
    "employment_contract": """You are an employment law specialist drafting Indian Employment Contracts.
Extract fields from the user description. Return ONLY a valid JSON object.

Required JSON keys:
employer_name   - name of the employing company
employee_name   - full name of employee
job_title       - employee job title / designation
department      - department or team
start_date      - date of joining (DD/MM/YYYY)
ctc             - annual Cost to Company in INR
work_hours      - hours per day or week (e.g. "9 hours/day, 5 days/week")
governing_state - Indian state whose laws govern the contract

Optional JSON keys:
probation_period              - probationary period duration (typically 3-6 months)
notice_period                 - notice period required by both parties
non_compete                   - non-compete restrictions after leaving
non_solicitation              - non-solicitation clause
benefits                      - PF, ESI, health insurance, gratuity, etc.
leave_policy                  - casual leave, sick leave, earned leave entitlements
termination_clause            - grounds and process for termination
intellectual_property_assignment - IP ownership clause
pf_applicable                 - whether PF/EPF is applicable
gratuity_applicable           - whether gratuity is applicable

Set any field not mentioned to "Not Specified". Return ONLY the JSON object.""",
}


# ---------------------------------------------------------------------------
# Document generation prompts
# One prompt per document type.
# Each prompt is the SYSTEM message for the generation LLM call.
# NO response_format is set — plain text output only.
# ---------------------------------------------------------------------------

GENERATION_PROMPTS: dict[str, str] = {

    # ------------------------------------------------------------------
    "nda": """You are a senior advocate specialising in Indian contract and IP law.
Draft a complete, enforceable Non-Disclosure Agreement governed by Indian law.

Applicable Indian statutes:
- Indian Contract Act, 1872 (Sections 27, 73, 74)
- Information Technology Act, 2000
- Indian Stamp Act, 1899
- Copyright Act, 1957
- Specific Relief Act, 1963

Document structure:
1. PARTIES AND RECITALS
2. DEFINITIONS
   2.1 Confidential Information
   2.2 Exclusions from Confidential Information
3. OBLIGATIONS OF RECEIVING PARTY
   3.1 Non-Disclosure Obligation
   3.2 Standard of Care
   3.3 Permitted Disclosures (including disclosures required by SEBI, RBI, or courts)
4. TERM AND TERMINATION
5. RETURN OR DESTRUCTION OF MATERIALS
6. REMEDIES AND RELIEF
   6.1 Injunctive Relief under Specific Relief Act, 1963
   6.2 Damages under Indian Contract Act, 1872
7. GENERAL PROVISIONS
   7.1 Governing Law and Jurisdiction
   7.2 Dispute Resolution (Arbitration and Conciliation Act, 1996)
   7.3 Entire Agreement
   7.4 Amendments
   7.5 Stamp Duty
8. SIGNATURE BLOCK

Formatting rules:
- Section headings in ALL CAPS with number (e.g. "1. PARTIES AND RECITALS:")
- Sub-clauses numbered 1.1, 1.2 etc.
- Formal legal language compliant with Indian law
- Use INR for any monetary amounts
- Reference relevant Indian statutes where appropriate
- Plain text only — no markdown, no asterisks
- Use actual values provided; use [TO BE AGREED] for missing values""",

    # ------------------------------------------------------------------
    "job_offer": """You are a senior HR professional and employment law expert in India.
Draft a complete Job Offer / Appointment Letter compliant with Indian labour law.

Applicable Indian statutes:
- Industrial Employment (Standing Orders) Act, 1946
- Shops and Establishments Act (state-specific)
- Employees Provident Fund Act, 1952
- Payment of Gratuity Act, 1972
- Maternity Benefit Act, 1961
- Sexual Harassment of Women at Workplace (POSH) Act, 2013

Document structure:
1. DATE AND ADDRESSEE
2. OPENING — welcome and congratulate the candidate warmly
3. DESIGNATION AND DEPARTMENT
4. COMPENSATION STRUCTURE
   - Gross CTC breakdown (Basic, HRA, Special Allowance, LTA, etc.)
   - PF / EPF contribution (12% of Basic under EPF Act, 1952)
   - Gratuity eligibility (Payment of Gratuity Act, 1972)
5. BENEFITS
   - Medical / Health Insurance
   - Leave entitlements (CL, SL, PL as per Shops Act)
   - Other perks
6. DATE OF JOINING AND WORK LOCATION
7. PROBATION PERIOD AND CONFIRMATION
8. NOTICE PERIOD
9. CODE OF CONDUCT AND POLICIES
   - POSH policy reference (POSH Act, 2013)
   - Confidentiality obligation
10. CONDITIONS OF EMPLOYMENT
    - Background verification
    - Document submission
11. ACCEPTANCE INSTRUCTIONS
12. CLOSING
13. SIGNATURE BLOCK

Formatting rules:
- Warm professional tone for a letter
- Reference Indian statutes where appropriate
- All compensation in INR
- Plain text only — no markdown, no asterisks""",

    # ------------------------------------------------------------------
    "freelancer_agreement": """You are an Indian contracts attorney specialising in IT and creative services law.
Draft a complete Freelancer / Independent Contractor Agreement under Indian law.

Applicable Indian statutes:
- Indian Contract Act, 1872
- Copyright Act, 1957 (Section 17 — work for hire)
- Information Technology Act, 2000
- Goods and Services Tax Act, 2017
- Income Tax Act, 1961 (TDS under Section 194C / 194J)

Document structure:
1. PARTIES AND RECITALS
2. SCOPE OF WORK
   2.1 Project Description
   2.2 Deliverables
   2.3 Timeline and Milestones
3. COMPENSATION AND PAYMENT
   3.1 Project Fee (in INR)
   3.2 Payment Schedule
   3.3 GST (if applicable — GSTIN of both parties)
   3.4 TDS Deduction (Section 194C or 194J)
   3.5 Late Payment Interest
4. REVISIONS AND CHANGE ORDERS
5. INTELLECTUAL PROPERTY
   5.1 Assignment of Copyright under Copyright Act, 1957
   5.2 Moral Rights Waiver
   5.3 Freelancer Portfolio Rights
6. CONFIDENTIALITY
7. INDEPENDENT CONTRACTOR STATUS
   7.1 No Employer-Employee Relationship
   7.2 Tax Responsibility
   7.3 No PF / ESI / Gratuity Obligation
8. CANCELLATION AND KILL FEE
9. LIMITATION OF LIABILITY
10. DISPUTE RESOLUTION
    - Arbitration under Arbitration and Conciliation Act, 1996
11. GENERAL PROVISIONS
    - Governing Law (Indian law)
    - Stamp Duty
12. SIGNATURE BLOCK

Formatting rules:
- Section headings in ALL CAPS with number
- Sub-clauses numbered
- Formal legal language under Indian law
- All amounts in INR
- Plain text only — no markdown, no asterisks""",

    # ------------------------------------------------------------------
    "service_agreement": """You are an Indian commercial contracts attorney.
Draft a complete Service Agreement governed by Indian law.

Applicable Indian statutes:
- Indian Contract Act, 1872
- Specific Relief Act, 1963
- Information Technology Act, 2000
- GST Act, 2017
- Digital Personal Data Protection Act, 2023
- Arbitration and Conciliation Act, 1996

Document structure:
1. PARTIES
2. DEFINITIONS
3. SCOPE OF SERVICES
   3.1 Services Description
   3.2 Service Standards and SLAs
   3.3 Change in Scope
4. TERM AND RENEWAL
5. FEES AND PAYMENT
   5.1 Service Fees (in INR)
   5.2 Payment Terms
   5.3 GST (GSTIN details)
   5.4 TDS Deduction
   5.5 Late Payment Interest
6. INTELLECTUAL PROPERTY
7. CONFIDENTIALITY AND DATA PROTECTION
   - IT Act, 2000 and DPDP Act, 2023
8. REPRESENTATIONS AND WARRANTIES
9. LIMITATION OF LIABILITY
10. INDEMNIFICATION
11. TERMINATION
    11.1 Termination for Convenience
    11.2 Termination for Cause
    11.3 Effect of Termination
12. DISPUTE RESOLUTION
    - Arbitration under Arbitration and Conciliation Act, 1996
    - Jurisdiction: courts of [governing state]
13. GENERAL PROVISIONS
14. SIGNATURE BLOCK

Formatting rules:
- Section headings in ALL CAPS with number
- Sub-clauses numbered
- Formal legal language under Indian law
- All amounts in INR
- Plain text only — no markdown, no asterisks""",

    # ------------------------------------------------------------------
    "consulting_agreement": """You are an Indian commercial attorney specialising in professional services.
Draft a complete Consulting Agreement governed by Indian law.

Applicable Indian statutes:
- Indian Contract Act, 1872
- Copyright Act, 1957
- Income Tax Act, 1961 (TDS Section 194J)
- GST Act, 2017
- Arbitration and Conciliation Act, 1996

Document structure:
1. PARTIES
2. SCOPE OF CONSULTING SERVICES
   2.1 Scope of Work
   2.2 Deliverables
3. TERM
4. COMPENSATION AND EXPENSES
   4.1 Consulting Fees (in INR)
   4.2 Expense Reimbursement
   4.3 GST (if applicable)
   4.4 TDS under Section 194J
   4.5 Invoicing and Payment Timeline
5. INTELLECTUAL PROPERTY
   5.1 Work Product Ownership
   5.2 Background IP
   5.3 Licence Grant
6. CONFIDENTIALITY
7. NON-COMPETE AND NON-SOLICITATION
   7.1 Non-Competition
   7.2 Non-Solicitation of Employees
   7.3 Non-Solicitation of Clients
8. INDEPENDENT CONTRACTOR STATUS
   - No PF / ESI / Gratuity obligation
9. REPRESENTATIONS AND WARRANTIES
10. LIMITATION OF LIABILITY
11. TERMINATION
12. DISPUTE RESOLUTION
    - Arbitration under Arbitration and Conciliation Act, 1996
13. GENERAL PROVISIONS
14. SIGNATURE BLOCK

Formatting rules:
- Section headings in ALL CAPS with number
- Sub-clauses numbered
- Formal legal language under Indian law
- All amounts in INR
- Plain text only — no markdown, no asterisks""",

    # ------------------------------------------------------------------
    "lease_agreement": """You are an Indian property law attorney specialising in residential and commercial leases.
Draft a complete Leave and Licence / Lease Agreement compliant with Indian property law.

Applicable Indian statutes:
- Transfer of Property Act, 1882
- Registration Act, 1908 (mandatory registration if term exceeds 12 months)
- Indian Stamp Act, 1899 (state-specific stamp duty)
- Rent Control Act (state-specific)
- Maharashtra Rent Control Act / Delhi Rent Act (as applicable by state)

Document structure:
1. PARTIES AND PROPERTY
   - Full names, addresses, Aadhaar/PAN references
   - Complete property description with area in sq.ft.
2. NATURE OF AGREEMENT
   - Leave and Licence (preferred) or Lease
   - Distinction from tenancy under Rent Control Act
3. LICENCE / LEASE TERM
   - Start date, end date, lock-in period
4. LICENCE FEE / RENT
   4.1 Monthly Amount (in INR)
   4.2 Due Date (e.g. 1st of each month)
   4.3 Mode of Payment (bank transfer, cheque, UPI)
   4.4 Annual Escalation Clause (typically 5-10%)
5. REFUNDABLE SECURITY DEPOSIT
   5.1 Amount (in INR)
   5.2 Conditions for Deduction
   5.3 Return Timeline (typically 30-60 days after vacating)
6. MAINTENANCE AND SOCIETY CHARGES
7. UTILITIES AND SERVICES
8. PERMITTED USE OF PREMISES
9. RESTRICTION ON SUBLETTING AND ASSIGNMENT
10. ALTERATIONS AND REPAIRS
    10.1 Licensor / Landlord Obligations
    10.2 Licensee / Tenant Obligations
11. ENTRY BY LICENSOR / LANDLORD
12. TERMINATION AND NOTICE PERIOD
13. VACATION OF PREMISES AND MOVE-OUT
14. STAMP DUTY AND REGISTRATION
    - Applicable state stamp duty
    - Registration under Registration Act, 1908
15. GENERAL PROVISIONS
    - Governing Law and Jurisdiction
16. SCHEDULE A — PROPERTY DETAILS
17. SIGNATURE BLOCK WITH WITNESSES

Formatting rules:
- Section headings in ALL CAPS with number
- Sub-clauses numbered
- Formal legal language under Indian property law
- All amounts in INR
- Reference state-specific laws where appropriate
- Plain text only — no markdown, no asterisks""",

    # ------------------------------------------------------------------
    "employment_contract": """You are an Indian employment law attorney.
Draft a complete Employment Contract / Appointment Letter compliant with Indian labour law.

Applicable Indian statutes:
- Industrial Employment (Standing Orders) Act, 1946
- Employees Provident Fund and Miscellaneous Provisions Act, 1952
- Payment of Gratuity Act, 1972
- Payment of Bonus Act, 1965
- Maternity Benefit Act, 1961
- Sexual Harassment of Women at Workplace (POSH) Act, 2013
- Shops and Establishments Act (state-specific)
- Code on Wages, 2019
- Code on Social Security, 2020
- Income Tax Act, 1961 (TDS on salary)
- Digital Personal Data Protection Act, 2023
- Copyright Act, 1957 (IP assignment)

Document structure:
1. PARTIES
2. APPOINTMENT AND DESIGNATION
   2.1 Job Title and Grade
   2.2 Department and Reporting Structure
   2.3 Place of Work
3. DATE OF JOINING AND COMMENCEMENT
4. NATURE OF EMPLOYMENT
   - Permanent / Contract / Fixed-term
5. COMPENSATION AND BENEFITS
   5.1 Cost to Company (CTC) — annual in INR
   5.2 CTC Breakup (Basic, HRA, Special Allowance, LTA, etc.)
   5.3 Provident Fund (EPF) — 12% of Basic under EPF Act, 1952
   5.4 Gratuity — as per Payment of Gratuity Act, 1972
   5.5 Health and Medical Insurance
   5.6 Performance Bonus / Variable Pay
   5.7 Income Tax (TDS deduction)
6. WORKING HOURS
   - As per Shops and Establishments Act of [governing state]
7. LEAVE ENTITLEMENTS
   7.1 Earned / Privileged Leave
   7.2 Casual Leave
   7.3 Sick / Medical Leave
   7.4 Maternity / Paternity Leave
   7.5 National and Festival Holidays
8. PROBATIONARY PERIOD
   - Duration, confirmation process, extension conditions
9. NOTICE PERIOD
   - During probation and post-confirmation
10. CODE OF CONDUCT AND POLICIES
    - Company policies, POSH compliance
    - Prevention of Sexual Harassment (POSH Act, 2013)
11. CONFIDENTIALITY AND NON-DISCLOSURE
12. INTELLECTUAL PROPERTY ASSIGNMENT
    - All work product vests in employer
    - Reference Copyright Act, 1957
13. NON-COMPETE AND NON-SOLICITATION
    - Reasonable restrictions under Section 27, Indian Contract Act
14. TERMINATION OF EMPLOYMENT
    14.1 Resignation by Employee
    14.2 Termination by Employer
    14.3 Termination for Cause (misconduct, etc.)
    14.4 Retirement
    14.5 Full and Final Settlement
15. POST-TERMINATION OBLIGATIONS
16. DATA PROTECTION
    - Digital Personal Data Protection Act, 2023
17. GRIEVANCE REDRESSAL
    - Internal Complaints Committee (POSH)
    - HR escalation process
18. GOVERNING LAW AND DISPUTE RESOLUTION
    - Governing law: Laws of India
    - Jurisdiction: courts of [governing state]
    - Arbitration under Arbitration and Conciliation Act, 1996
19. GENERAL PROVISIONS
20. SIGNATURE BLOCK

Formatting rules:
- Section headings in ALL CAPS with number
- Sub-clauses numbered
- Formal legal language under Indian law
- All amounts in INR
- Reference specific Indian statutes throughout
- Plain text only — no markdown, no asterisks""",
}
```

---
File: t_intent.py
---

```py
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
```

---
File: t_key_clause_extraction.py
---

```py
from fastapi import HTTPException, UploadFile
import time
import uuid
import os
import logging

from s_ai_model import run_llm
from s_pdf_utils import load_pdf, get_page_count, all_pages_blank
from backupfiles.t_utils import extract_json_from_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "temp"                      # ← was missing, caused NameError
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

```

---
File: s_pdf_utils.py
---

```py
import re
import os
import time
import logging
import warnings
import numpy as np
import fitz  # PyMuPDF
import paddle
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from paddleocr import PaddleOCRVL
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Suppress noisy internal PaddleOCR tensor-copy warning — harmless, not our code
warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor",
    category=UserWarning,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking config
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 12000
CHUNK_OVERLAP = 200

# Pre-compiled regex patterns
_RE_HYPHEN    = re.compile(r"-\n")
_RE_MULTILINE = re.compile(r"\n{3,}")
_RE_SPACES    = re.compile(r" {2,}")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
NATIVE_TEXT_THRESHOLD = 0
OCR_RETRY_ATTEMPTS    = 2

# Parallel workers for native extraction (fitz is thread-safe for reads)
_NATIVE_EXTRACT_WORKERS = 4

# How many pages to sample when detecting PDF type
_PDF_TYPE_SAMPLE_PAGES = 5

# ---------------------------------------------------------------------------
# OCR timeout — if PaddleOCR takes longer than this per page, abort and
# insert a placeholder. Prevents a single bad page from freezing the server.
# ---------------------------------------------------------------------------
OCR_PAGE_TIMEOUT = 30   # seconds per page

# ---------------------------------------------------------------------------
# Stagger worker startup to prevent simultaneous GPU memory allocation.
#
# When gunicorn spawns multiple workers at the same time, each tries to load
# PaddleOCR-VL into GPU memory simultaneously. This causes:
#   - PyTorch/Paddle C++ dispatcher conflicts (deregisterImpl_ crash)
#   - CUDA out-of-memory spikes
#   - "No module named utils" errors from race conditions
#
# Solution: each worker waits (worker_slot * STAGGER_SECONDS) before loading.
# Worker slots are assigned by hashing the PID into 0..MAX_WORKERS-1.
# With 4 workers and 15s stagger: loads at 0s, 15s, 30s, 45s.
# ---------------------------------------------------------------------------
_MAX_WORKERS     = 4    # must match --workers in gunicorn service file
_STAGGER_SECONDS = 15   # seconds between each worker's model load

_worker_slot = os.getpid() % _MAX_WORKERS
_stagger_delay = _worker_slot * _STAGGER_SECONDS

if _stagger_delay > 0:
    logger.info(
        f"[pdf_utils] Worker PID={os.getpid()} slot={_worker_slot} "
        f"— waiting {_stagger_delay}s before loading PaddleOCR-VL "
        f"(prevents simultaneous GPU init crash)"
    )
    time.sleep(_stagger_delay)

# ---------------------------------------------------------------------------
# PaddleOCR-VL — loaded once per worker process at import time
# ---------------------------------------------------------------------------
logger.info(f"[pdf_utils] Loading PaddleOCR-VL 1.5 (PID={os.getpid()}) ...")
t0      = time.perf_counter()
_ocr_vl = PaddleOCRVL("v1.5")
logger.info(f"[pdf_utils] PaddleOCR-VL ready (PID={os.getpid()}, {time.perf_counter() - t0:.2f}s)")

# Dedicated single-thread executor for OCR — one GPU job at a time per worker.
# Using a single thread guarantees no concurrent GPU calls within this worker.
_OCR_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr_worker")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean PDF/OCR artifacts. Never returns None."""
    before = len(text)
    text = _RE_HYPHEN.sub("",        text)
    text = _RE_MULTILINE.sub("\n\n", text)
    text = _RE_SPACES.sub(" ",       text)
    text = text.strip()
    logger.debug(f"[pdf_utils] clean_text: {before} -> {len(text)} chars")
    return text


def _detect_pdf_type(doc: fitz.Document, pages_to_process: int) -> str:
    """
    Sample the first N pages to classify the PDF before extraction begins.

    Returns:
      'native'     - all sampled pages have text  -> text-based PDF
      'image_only' - NO sampled pages have text   -> fully scanned/image PDF
      'mixed'      - some have text, some do not  -> hybrid PDF
    """
    sample       = min(_PDF_TYPE_SAMPLE_PAGES, pages_to_process)
    native_count = 0

    for i in range(sample):
        text = doc[i].get_text("text").strip()
        if len(text) > 0:
            native_count += 1

    if native_count == sample:
        pdf_type = "native"
    elif native_count == 0:
        pdf_type = "image_only"
    else:
        pdf_type = "mixed"

    logger.info(
        f"[pdf_utils] PDF type: sampled {sample} page(s), "
        f"{native_count} had native text -> '{pdf_type}'"
    )
    return pdf_type


def _extract_native(page: fitz.Page) -> str | None:
    """Strategy 1 - PyMuPDF native. Returns text if any chars present, else None."""
    text = page.get_text("text").strip()
    if len(text) > NATIVE_TEXT_THRESHOLD:
        return clean_text(text)
    return None


def _page_to_image(page: fitz.Page, dpi: int = 150) -> np.ndarray:
    """Render a fitz page to a C-contiguous uint8 RGB numpy array."""
    pix = page.get_pixmap(dpi=dpi)
    img = np.ascontiguousarray(
        np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    )
    if pix.n == 4:
        img = img[:, :, :3]
    return img


# ---------------------------------------------------------------------------
# Blank-PDF detection
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(
    r"^\[Page \d+: (blank page|content could not be extracted)\]$"
)


def all_pages_blank(pages: list[Document]) -> bool:
    """
    Return True if every page in the list is a blank/placeholder page.
    """
    if not pages:
        return True
    for page in pages:
        text = page.page_content.strip()
        if text and not _PLACEHOLDER_RE.match(text):
            return False
    return True


# ---------------------------------------------------------------------------
# OCR result text extractor
# ---------------------------------------------------------------------------

_OCR_TEXT_KEYS = ("rec_text", "text")

_OCR_JUNK_MARKERS = (
    "numpy.ndarray",
    "layout_det",
    "rec_score",
    "table_res",
    "input_path",
    "model_settings",
    "parsing_res",
    "spotting_res",
    "page_id",
)


def _extract_ocr_text(res) -> str:
    """
    Safely extract the human-readable OCR text from a single PaddleOCR-VL
    result item, ignoring internal metadata/debug fields.
    Never falls back to str(res) to avoid serialising internal state.
    """
    if res is None:
        return ""

    if isinstance(res, str):
        s = res.strip()
        if any(marker in s for marker in _OCR_JUNK_MARKERS):
            logger.debug(f"_extract_ocr_text: discarding junk string ({s[:60]!r}…)")
            return ""
        return s

    if isinstance(res, dict):
        for key in _OCR_TEXT_KEYS:
            if key in res:
                val = res[key]
                return val.strip() if isinstance(val, str) else ""
        if "res" in res:
            return _extract_ocr_text(res["res"])
        return ""

    for key in _OCR_TEXT_KEYS:
        if hasattr(res, key):
            val = getattr(res, key)
            return val.strip() if isinstance(val, str) else ""

    if hasattr(res, "res"):
        return _extract_ocr_text(res.res)

    logger.debug(f"_extract_ocr_text: unrecognised result type {type(res).__name__!r} — skipping")
    return ""


# ---------------------------------------------------------------------------
# OCR worker — runs in _OCR_EXECUTOR thread
# ---------------------------------------------------------------------------

def _ocr_predict(img: np.ndarray) -> list:
    """
    Thin wrapper around _ocr_vl.predict so it can be submitted to the
    executor and cancelled via future.cancel() / timeout.
    Returns the raw result list from PaddleOCR-VL.
    """
    return _ocr_vl.predict(img)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pdf(file_path: str, max_pages: int | None = None) -> list[Document]:
    """
    Load a PDF and return one LangChain Document per page.

    Extraction pipeline per page:

      Tier 1 — PyMuPDF native  (parallel, ~0.01s/page, zero GPU cost)
      Tier 2 — PaddleOCR-VL   (serial GPU, ~7-10s/page, with timeout + retry)
      Tier 3 — Placeholder     ("[Page N: content could not be extracted]")

    OCR timeout: each page is given OCR_PAGE_TIMEOUT seconds (default 30s).
    If PaddleOCR hangs (e.g. corrupted image), the page gets a placeholder
    after the timeout rather than blocking the server for minutes.

    PDF type detection:
      'native'     -> Pass 1 only
      'image_only' -> Pass 1 skipped, all pages go to PaddleOCR-VL
      'mixed'      -> Pass 1 for text pages, PaddleOCR-VL for image pages

    Data integrity: every page is always returned. No page is silently dropped.
    """
    logger.info(
        f"[pdf_utils] load_pdf: '{file_path}'"
        + (f" max_pages={max_pages}" if max_pages else "")
    )
    t_load = time.perf_counter()

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"[pdf_utils] load_pdf: open failed -- {e}")
        raise ValueError(f"Could not open PDF: {e}")

    total_pages      = len(doc)
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)
    logger.info(
        f"[pdf_utils] {total_pages} total page(s), "
        f"processing {pages_to_process} "
        f"({'all' if pages_to_process == total_pages else f'first {pages_to_process}'})"
    )

    pdf_type   = _detect_pdf_type(doc, pages_to_process)
    fitz_pages = [doc[i] for i in range(pages_to_process)]

    # ── Pass 1: parallel native extraction ───────────────────────────────
    native_results: dict[int, str | None] = {}

    if pdf_type == "image_only":
        logger.info(
            "[pdf_utils] Image-only PDF -- skipping native pass, "
            "all pages going to PaddleOCR-VL"
        )
        native_results = {i: None for i in range(pages_to_process)}
    else:
        logger.info(
            f"[pdf_utils] Pass 1 -- parallel native extraction "
            f"({_NATIVE_EXTRACT_WORKERS} workers)"
        )
        t_pass1 = time.perf_counter()

        def _native_worker(idx_page):
            idx, page = idx_page
            return idx, _extract_native(page)

        with ThreadPoolExecutor(max_workers=_NATIVE_EXTRACT_WORKERS) as pool:
            futures = {
                pool.submit(_native_worker, (i, p)): i
                for i, p in enumerate(fitz_pages)
            }
            for future in as_completed(futures):
                try:
                    idx, text = future.result()
                    native_results[idx] = text
                except Exception as e:
                    logger.warning(f"[pdf_utils] Pass 1 worker failed: {e}")
                    native_results[futures[future]] = None

        native_hit = sum(1 for v in native_results.values() if v is not None)
        logger.info(
            f"[pdf_utils] Pass 1 done ({time.perf_counter() - t_pass1:.2f}s) -- "
            f"native={native_hit}, need_ocr={pages_to_process - native_hit}"
        )

    native_hit    = sum(1 for v in native_results.values() if v is not None)
    paddle_needed = [i for i, v in native_results.items() if v is None]

    if paddle_needed:
        logger.info(
            f"[pdf_utils] Pass 2 -- PaddleOCR-VL (GPU) for {len(paddle_needed)} page(s) "
            f"(timeout={OCR_PAGE_TIMEOUT}s/page)"
        )

    # ── Pass 2: pre-render images in parallel ─────────────────────────────
    pre_rendered: dict[int, np.ndarray] = {}

    if paddle_needed:
        logger.info(
            f"[pdf_utils] Pre-rendering {len(paddle_needed)} page image(s) "
            f"in parallel ({_NATIVE_EXTRACT_WORKERS} CPU workers) ..."
        )
        t_render = time.perf_counter()

        def _render_worker(idx: int) -> tuple[int, np.ndarray]:
            return idx, _page_to_image(fitz_pages[idx], dpi=150)

        with ThreadPoolExecutor(max_workers=_NATIVE_EXTRACT_WORKERS) as pool:
            futures = {pool.submit(_render_worker, i): i for i in paddle_needed}
            for future in as_completed(futures):
                try:
                    idx, img = future.result()
                    pre_rendered[idx] = img
                except Exception as e:
                    logger.warning(f"[pdf_utils] Pre-render failed for page {futures[future]+1}: {e}")

        logger.info(
            f"[pdf_utils] Pre-render done ({time.perf_counter()-t_render:.2f}s) — "
            f"{len(pre_rendered)}/{len(paddle_needed)} images ready"
        )

    # ── Pass 2: OCR with per-page timeout ─────────────────────────────────
    paddle_results: dict[int, str] = {}

    for idx in paddle_needed:
        fitz_page = fitz_pages[idx]
        page_num  = idx + 1
        pre_img   = pre_rendered.get(idx)

        for attempt in range(1, OCR_RETRY_ATTEMPTS + 1):
            if attempt == 1 and pre_img is not None:
                img = pre_img
                dpi = 150
            else:
                dpi = 150 + (attempt - 1) * 50
                img = _page_to_image(fitz_page, dpi=dpi)

            t_ocr = time.perf_counter()
            try:
                future  = _OCR_EXECUTOR.submit(_ocr_predict, img)
                results = future.result(timeout=OCR_PAGE_TIMEOUT)
                elapsed = time.perf_counter() - t_ocr

                page_parts = []
                for res in results:
                    extracted = _extract_ocr_text(res)
                    if extracted:
                        page_parts.append(extracted)

                del results
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()

                text = clean_text("\n".join(page_parts))

                if not text:
                    logger.info(
                        f"[pdf_utils] Page {page_num}: PaddleOCR-VL returned no text "
                        f"(attempt {attempt}, dpi={dpi}, {elapsed:.2f}s) — blank page"
                    )
                    paddle_results[idx] = f"[Page {page_num}: blank page]"
                else:
                    logger.info(
                        f"[pdf_utils] Page {page_num}: PaddleOCR-VL OK "
                        f"(attempt {attempt}, dpi={dpi}, {len(page_parts)} block(s), {elapsed:.2f}s)"
                    )
                    paddle_results[idx] = text
                break

            except FuturesTimeoutError:
                elapsed = time.perf_counter() - t_ocr
                logger.error(
                    f"[pdf_utils] Page {page_num}: PaddleOCR-VL TIMEOUT "
                    f"(attempt {attempt}, {elapsed:.1f}s > {OCR_PAGE_TIMEOUT}s limit) "
                    + ("— retrying at higher DPI" if attempt < OCR_RETRY_ATTEMPTS else "— giving up")
                )
                future.cancel()
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()

            except Exception as e:
                elapsed = time.perf_counter() - t_ocr
                logger.warning(
                    f"[pdf_utils] Page {page_num}: PaddleOCR-VL attempt {attempt} failed "
                    f"({elapsed:.2f}s, {e})"
                    + (" -- retrying" if attempt < OCR_RETRY_ATTEMPTS else " -- exhausted")
                )
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()

        else:
            logger.error(
                f"[pdf_utils] Page {page_num}: all OCR attempts failed/timed out — placeholder"
            )
            paddle_results[idx] = f"[Page {page_num}: content could not be extracted]"

        if idx not in paddle_results:
            paddle_results[idx] = f"[Page {page_num}: content could not be extracted]"

    # ── Assemble results in page order ────────────────────────────────────
    pages:             list[Document] = []
    paddle_count:      int = 0
    placeholder_count: int = 0
    blank_count:       int = 0

    for idx, fitz_page in enumerate(fitz_pages):
        page_num = idx + 1

        if native_results.get(idx) is not None:
            text = native_results[idx]
            logger.debug(f"[pdf_utils] Page {page_num}: native ({len(text)} chars)")

        elif idx in paddle_needed:
            text = paddle_results.get(idx, f"[Page {page_num}: content could not be extracted]")
            if text.startswith("[Page "):
                placeholder_count += 1
            else:
                paddle_count += 1

        else:
            logger.error(f"[pdf_utils] Page {page_num}: no result -- placeholder")
            text = f"[Page {page_num}: content could not be extracted]"
            placeholder_count += 1

        if not text:
            logger.warning(f"[pdf_utils] Page {page_num}: blank after extraction")
            text = f"[Page {page_num}: blank page]"
            blank_count += 1

        pages.append(Document(
            page_content=text,
            metadata={"page": page_num, "source": file_path},
        ))

    doc.close()
    elapsed      = time.perf_counter() - t_load
    total_loaded = len(pages)

    # ── Integrity report ──────────────────────────────────────────────────
    logger.info("[pdf_utils] -- EXTRACTION COMPLETE --------------------------")
    logger.info(f"[pdf_utils] PDF type         : {pdf_type}")
    logger.info(f"[pdf_utils] Pages processed  : {pages_to_process}/{total_pages}")
    logger.info(f"[pdf_utils] Native text       : {native_hit} page(s)")
    logger.info(f"[pdf_utils] PaddleOCR-VL      : {paddle_count} page(s)")
    logger.info(f"[pdf_utils] Placeholders      : {placeholder_count} page(s)")
    logger.info(f"[pdf_utils] Blank pages        : {blank_count} page(s)")
    logger.info(f"[pdf_utils] Total loaded      : {total_loaded} page(s)")
    logger.info(f"[pdf_utils] Time              : {elapsed:.2f}s")
    logger.info("[pdf_utils] ------------------------------------------------------")

    if placeholder_count > 0:
        logger.warning(
            f"[pdf_utils] {placeholder_count} page(s) could not be extracted -- "
            "included as placeholders"
        )

    if total_loaded != pages_to_process:
        logger.error(
            f"[pdf_utils] DATA INTEGRITY: expected {pages_to_process}, got {total_loaded}"
        )

    if total_loaded == 0:
        raise ValueError(
            "No content extracted from any page. "
            "File may be blank, encrypted, or corrupt."
        )

    return pages


def merge_pages(pages: list[Document]) -> list[Document]:
    """Concatenate all pages into one Document so the splitter produces full-size chunks."""
    if not pages:
        return pages

    total_chars_before = sum(len(p.page_content) for p in pages)
    logger.info(
        f"[pdf_utils] merge_pages: {len(pages)} page(s), "
        f"{total_chars_before:,} total chars"
    )

    combined_text = "\n\n".join(p.page_content for p in pages)

    if len(combined_text) < total_chars_before:
        logger.error(
            f"[pdf_utils] merge_pages: DATA LOSS -- "
            f"merged ({len(combined_text):,}) < sum of pages ({total_chars_before:,})"
        )

    merged = Document(
        page_content=combined_text,
        metadata={
            "page":   f"1-{pages[-1].metadata.get('page', len(pages))}",
            "source": pages[0].metadata.get("source", ""),
        },
    )
    logger.info(
        f"[pdf_utils] merge_pages: {len(combined_text):,} chars "
        f"(~{len(combined_text) // CHUNK_SIZE + 1} chunks)"
    )
    return [merged]


def split_documents(pages: list[Document]) -> list[Document]:
    """Merge all pages then split into CHUNK_SIZE chunks."""
    merged      = merge_pages(pages)
    total_chars = sum(len(p.page_content) for p in merged)

    logger.info(
        f"[pdf_utils] split_documents: {total_chars:,} chars, "
        f"chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    t_split = time.perf_counter()
    chunks  = splitter.split_documents(merged)
    avg     = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0

    logger.info(
        f"[pdf_utils] split_documents: {len(chunks)} chunk(s) in "
        f"{time.perf_counter() - t_split:.3f}s (avg {avg:.0f} chars/chunk)"
    )
    logger.info(
        f"[pdf_utils] Coverage: ALL {len(pages)} page(s) -> "
        f"{len(chunks)} inference call(s)"
    )
    return chunks


def get_page_count(file_path: str) -> int:
    """Return page count without fully loading the PDF."""
    try:
        with fitz.open(file_path) as doc:
            count = len(doc)
            logger.debug(f"[pdf_utils] get_page_count: {count} page(s)")
            return count
    except Exception as e:
        logger.error(f"[pdf_utils] get_page_count failed: {e}")
        return 0
```

---
File: t_document_route.py
---

```py
"""
t_document_route.py

FastAPI router for document generation.
All requests are logged to the document_requests table in PostgreSQL.

Endpoints:
  GET  /documents/types                 List all supported document types
  GET  /documents/stats                 Aggregate stats from DB
  GET  /documents/recent                Recent document requests from DB
  POST /documents/generate              Generate → JSON (text + base64 docx)
  POST /documents/generate/download     Generate → DOCX file download
  POST /documents/generate/stream       Generate → SSE token stream
  POST /documents/extract-fields        Extract fields only (no generation)
"""

import json
import time
import uuid
import base64
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

# ── Database ─────────────────────────────────────────────────────────────────
from s_db import (
    log_document_request,
    get_document_stats,
    get_recent_documents,
)

# ── Core generation logic ─────────────────────────────────────────────────────
from t_document_generator import (
    generate_document,
    SUPPORTED_DOCUMENT_TYPES,
    _SCHEMAS,
    _extract_fields,
    _get_missing_fields,
    _render_docx,
    _api_kwargs,
    _get_client,
)

# ── Prompts (moved to t_prompts.py) ──────────────────────────────────────────
from t_prompts import GENERATION_PROMPTS

# ── Intent classification (moved to t_intent.py) ─────────────────────────────
from t_intent import classify_intent, resolve_document_type

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Document Generation"])


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class DocumentGenerateRequest(BaseModel):
    document_type: Optional[str] = Field(
        default=None,
        description=(
            "Type of document to generate. "
            "If omitted, automatically detected from user_query. "
            "Supported: nda, job_offer, freelancer_agreement, service_agreement, "
            "consulting_agreement, lease_agreement, employment_contract"
        ),
        examples=["nda", "employment_contract", None],
    )
    user_query: str = Field(
        ...,
        min_length=20,
        description=(
            "Free-text description of the document you need under Indian law. "
            "Include party names, dates, amounts in INR, state jurisdiction, "
            "and any special clauses."
        ),
        examples=[
            "NDA between TechCorp Pvt Ltd (Mumbai) and Rajan Shah (contractor). "
            "Purpose: sharing AI product roadmap. Duration 2 years. "
            "Governed by Maharashtra law. Effective 1st April 2026.",
        ],
    )


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# GET /documents/types
# ---------------------------------------------------------------------------

@router.get("/types", summary="List all supported document types")
async def list_document_types() -> dict:
    types = []
    for slug, display_name in SUPPORTED_DOCUMENT_TYPES.items():
        schema = _SCHEMAS.get(slug, {"required": [], "optional": []})
        types.append({
            "slug":            slug,
            "display_name":    display_name,
            "required_fields": schema.get("required", []),
            "optional_fields": schema.get("optional", []),
        })
    return {"status": "success", "total": len(types), "document_types": types}


# ---------------------------------------------------------------------------
# GET /documents/stats
# ---------------------------------------------------------------------------

@router.get("/stats", summary="Aggregate document generation statistics")
async def document_stats() -> dict:
    stats = await get_document_stats()
    return {"status": "success", **stats}


# ---------------------------------------------------------------------------
# GET /documents/recent
# ---------------------------------------------------------------------------

@router.get("/recent", summary="Recent document generation requests")
async def recent_documents(limit: int = 20) -> dict:
    limit = min(limit, 100)
    rows  = await get_recent_documents(limit)
    return {"status": "success", "total": len(rows), "documents": rows}


# ---------------------------------------------------------------------------
# POST /documents/extract-fields
# ---------------------------------------------------------------------------

@router.post("/extract-fields", summary="Extract fields without generating document")
async def extract_document_fields(request: DocumentGenerateRequest) -> dict:
    request_id = str(uuid.uuid4())[:8]

    if request.document_type:
        doc_type = resolve_document_type(request.document_type)
        if not doc_type:
            raise HTTPException(status_code=400, detail={
                "error":           "Unsupported document type",
                "provided":        request.document_type,
                "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
            })
        detected = False
    else:
        doc_type = await classify_intent(request.user_query)
        detected = True
        if not doc_type:
            raise HTTPException(status_code=422, detail={
                "error": "Could not determine document type.",
                "hint":  "Please specify document_type explicitly.",
                "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
            })

    try:
        fields  = await _extract_fields(doc_type, request.user_query)
        missing = _get_missing_fields(doc_type, fields)
    except Exception as e:
        logger.exception(f"[{request_id}] field extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Field extraction failed: {str(e)}")

    return {
        "status":            "missing_fields" if missing else "success",
        "document_type":     doc_type,
        "document_name":     SUPPORTED_DOCUMENT_TYPES[doc_type],
        "type_was_detected": detected,
        "fields":            fields,
        "missing_fields":    missing,
        "message": (
            "All required fields found. Ready to generate."
            if not missing else
            f"Missing {len(missing)} required field(s): {', '.join(missing)}. "
            "Add these details to your query for a complete document."
        ),
    }


# ---------------------------------------------------------------------------
# POST /documents/generate — JSON response + DB log
# ---------------------------------------------------------------------------

@router.post("/generate", summary="Generate document — returns JSON with text and DOCX")
async def generate_document_endpoint(request: DocumentGenerateRequest) -> dict:
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()
    error_msg  = None

    logger.info(f"[{request_id}] ── DOC GENERATE — type='{request.document_type or 'auto'}'")

    try:
        result = await generate_document(request.document_type, request.user_query)
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"[{request_id}] error: {e}")
        await log_document_request(
            request_id        = request_id,
            document_type     = request.document_type or "unknown",
            document_name     = "",
            user_query        = request.user_query,
            status            = "error",
            completion_time_s = time.perf_counter() - t_start,
            endpoint          = "/documents/generate",
            error_message     = error_msg,
        )
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(e)}")

    if result["status"] == "unknown_type":
        await log_document_request(
            request_id        = request_id,
            document_type     = request.document_type or "unknown",
            document_name     = "",
            user_query        = request.user_query,
            status            = "unknown_type",
            completion_time_s = time.perf_counter() - t_start,
            endpoint          = "/documents/generate",
            error_message     = result.get("message"),
        )
        raise HTTPException(status_code=422, detail={
            "error":           result.get("message"),
            "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
        })

    elapsed    = time.perf_counter() - t_start
    docx_bytes = result.pop("docx_bytes", b"")
    docx_b64   = base64.b64encode(docx_bytes).decode("utf-8") if docx_bytes else ""
    detected   = request.document_type is None

    await log_document_request(
        request_id        = request_id,
        document_type     = result["document_type"],
        document_name     = result["document_name"],
        user_query        = request.user_query,
        status            = result["status"],
        type_was_detected = detected,
        missing_fields    = result.get("missing_fields", []),
        word_count        = result.get("word_count", 0),
        docx_size_bytes   = len(docx_bytes),
        input_tokens      = result.get("input_tokens", 0),
        output_tokens     = result.get("output_tokens", 0),
        completion_time_s = elapsed,
        endpoint          = "/documents/generate",
        fields            = result.get("fields", {}),
    )

    logger.info(
        f"[{request_id}] ── COMPLETE — {elapsed:.2f}s "
        f"type={result['document_type']} words={result.get('word_count', 0)}"
    )

    return {
        "status":             result["status"],
        "document_type":      result["document_type"],
        "document_name":      result["document_name"],
        "type_was_detected":  detected,
        "fields":             result["fields"],
        "missing_fields":     result["missing_fields"],
        "document":           result["document"],
        "docx_base64":        docx_b64,
        "word_count":         result["word_count"],
        "input_tokens":       result.get("input_tokens", 0),
        "output_tokens":      result.get("output_tokens", 0),
        "generation_time_s":  round(elapsed, 2),
        "request_id":         request_id,
    }


# ---------------------------------------------------------------------------
# POST /documents/generate/download — DOCX file + DB log
# ---------------------------------------------------------------------------

@router.post(
    "/generate/download",
    summary="Generate document — returns DOCX file download",
    response_class=Response,
)
async def generate_document_download(request: DocumentGenerateRequest):
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── DOC DOWNLOAD — type='{request.document_type or 'auto'}'")

    try:
        result = await generate_document(request.document_type, request.user_query)
    except Exception as e:
        logger.exception(f"[{request_id}] error: {e}")
        await log_document_request(
            request_id        = request_id,
            document_type     = request.document_type or "unknown",
            document_name     = "",
            user_query        = request.user_query,
            status            = "error",
            completion_time_s = time.perf_counter() - t_start,
            endpoint          = "/documents/generate/download",
            error_message     = str(e),
        )
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(e)}")

    if result["status"] == "unknown_type":
        await log_document_request(
            request_id        = request_id,
            document_type     = request.document_type or "unknown",
            document_name     = "",
            user_query        = request.user_query,
            status            = "unknown_type",
            completion_time_s = time.perf_counter() - t_start,
            endpoint          = "/documents/generate/download",
            error_message     = result.get("message"),
        )
        raise HTTPException(status_code=422, detail={
            "error":           result.get("message"),
            "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
        })

    elapsed    = time.perf_counter() - t_start
    docx_bytes = result.get("docx_bytes", b"")
    doc_type   = result["document_type"]
    missing    = result.get("missing_fields", [])
    detected   = request.document_type is None
    filename   = f"{doc_type}_{request_id}.docx"

    await log_document_request(
        request_id        = request_id,
        document_type     = doc_type,
        document_name     = result["document_name"],
        user_query        = request.user_query,
        status            = result["status"],
        type_was_detected = detected,
        missing_fields    = missing,
        word_count        = result.get("word_count", 0),
        docx_size_bytes   = len(docx_bytes),
        input_tokens      = result.get("input_tokens", 0),
        output_tokens     = result.get("output_tokens", 0),
        completion_time_s = elapsed,
        endpoint          = "/documents/generate/download",
        fields            = result.get("fields", {}),
    )

    logger.info(
        f"[{request_id}] ── DOWNLOAD COMPLETE — {elapsed:.2f}s "
        f"type={doc_type} words={result.get('word_count', 0)} "
        f"docx={len(docx_bytes):,} bytes"
    )

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Document-Type":     doc_type,
            "X-Document-Name":     result["document_name"],
            "X-Type-Detected":     str(detected).lower(),
            "X-Word-Count":        str(result.get("word_count", 0)),
            "X-Missing-Fields":    ",".join(missing) if missing else "",
            "X-Generation-Time":   str(round(elapsed, 2)),
            "X-Status":            result["status"],
            "X-Request-Id":        request_id,
        }
    )


# ---------------------------------------------------------------------------
# POST /documents/generate/stream — SSE stream + DB log on completion
# ---------------------------------------------------------------------------

@router.post("/generate/stream", summary="Generate document with real-time token streaming")
async def generate_document_stream(request: DocumentGenerateRequest):
    """
    Streams document generation via SSE.
    Logs to document_requests table when generation completes.

    Events: status, detected, fields, token, done, error
    """
    import os

    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── DOC STREAM — type='{request.document_type or 'auto'}'")

    async def _generate():
        doc_type_final = request.document_type or "unknown"
        doc_name_final = ""
        status_final   = "error"
        missing_final: list = []
        fields_final:  dict = {}
        word_count     = 0
        docx_size      = 0
        in_tok         = 0
        out_tok        = 0
        detected       = False
        error_msg      = None

        try:
            # Step 0: Resolve or detect type
            if request.document_type:
                doc_type = resolve_document_type(request.document_type)
                if not doc_type:
                    yield _sse("error", {
                        "message":         f"Unsupported document type: '{request.document_type}'",
                        "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
                    })
                    return
            else:
                yield _sse("status", {
                    "step":    "classifying",
                    "message": "Detecting document type...",
                })
                doc_type = await classify_intent(request.user_query)
                detected = True
                if not doc_type:
                    yield _sse("error", {
                        "message":         "Could not determine document type.",
                        "supported_types": list(SUPPORTED_DOCUMENT_TYPES.keys()),
                    })
                    return
                yield _sse("detected", {
                    "document_type": doc_type,
                    "document_name": SUPPORTED_DOCUMENT_TYPES[doc_type],
                    "message":       f"Detected: {SUPPORTED_DOCUMENT_TYPES[doc_type]}",
                })

            doc_type_final = doc_type
            doc_name_final = SUPPORTED_DOCUMENT_TYPES[doc_type]

            # Step 1: Extract fields
            yield _sse("status", {
                "step":    "extracting_fields",
                "message": f"Extracting details for {doc_name_final}...",
            })
            fields  = await _extract_fields(doc_type, request.user_query)
            missing = _get_missing_fields(doc_type, fields)
            fields_final  = fields
            missing_final = missing

            yield _sse("fields", {
                "fields":            fields,
                "missing_fields":    missing,
                "document_type":     doc_type,
                "document_name":     doc_name_final,
                "type_was_detected": detected,
            })

            # Step 2: Stream document generation
            gen_prompt  = GENERATION_PROMPTS[doc_type]   # ← from t_prompts.py
            field_lines = "\n".join(
                f"  {k.replace('_', ' ').title()}: {v}"
                for k, v in fields.items()
                if v and v != "Not Specified"
            )

            system = gen_prompt
            user   = f"""Use these details to generate the complete {doc_name_final} under Indian law:

{field_lines if field_lines else "(Use standard Indian law template values)"}

Additional context:
\"\"\"{request.user_query}\"\"\"

Write the complete {doc_name_final} now. Start directly with the document title."""

            yield _sse("status", {
                "step":    "generating_document",
                "message": f"Drafting {doc_name_final}...",
            })

            client = _get_client()
            kwargs = _api_kwargs(max_tokens=32000, use_json=False)
            kwargs["messages"]       = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
            kwargs["stream"]         = True
            kwargs["stream_options"] = {"include_usage": True}

            full_document = ""

            async with await client.chat.completions.create(**kwargs) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta          = chunk.choices[0].delta.content
                        full_document += delta
                        yield _sse("token", {"delta": delta})
                    if chunk.usage:
                        in_tok  = chunk.usage.prompt_tokens
                        out_tok = chunk.usage.completion_tokens

            word_count = len(full_document.split())

            # Step 3: Render DOCX
            yield _sse("status", {"step": "rendering_docx", "message": "Rendering DOCX..."})
            docx_bytes = _render_docx(full_document, doc_name_final)
            docx_b64   = base64.b64encode(docx_bytes).decode("utf-8")
            docx_size  = len(docx_bytes)

            status_final = "missing_fields" if missing else "success"
            elapsed      = time.perf_counter() - t_start

            logger.info(
                f"[{request_id}] ── STREAM COMPLETE — {elapsed:.2f}s "
                f"type={doc_type} words={word_count} "
                f"tokens={in_tok}in/{out_tok}out"
            )

            yield _sse("done", {
                "status":             status_final,
                "document_type":      doc_type,
                "document_name":      doc_name_final,
                "type_was_detected":  detected,
                "missing_fields":     missing,
                "word_count":         word_count,
                "docx_base64":        docx_b64,
                "docx_size_bytes":    docx_size,
                "generation_time_s":  round(elapsed, 2),
                "input_tokens":       in_tok,
                "output_tokens":      out_tok,
                "total_tokens":       in_tok + out_tok,
                "request_id":         request_id,
            })

        except Exception as e:
            error_msg    = str(e)
            status_final = "error"
            logger.exception(f"[{request_id}] stream error: {e}")
            yield _sse("error", {"message": f"Document generation failed: {str(e)}"})

        finally:
            elapsed = time.perf_counter() - t_start
            await log_document_request(
                request_id        = request_id,
                document_type     = doc_type_final,
                document_name     = doc_name_final,
                user_query        = request.user_query,
                status            = status_final,
                type_was_detected = detected,
                missing_fields    = missing_final,
                word_count        = word_count,
                docx_size_bytes   = docx_size,
                input_tokens      = in_tok,
                output_tokens     = out_tok,
                completion_time_s = elapsed,
                endpoint          = "/documents/generate/stream",
                error_message     = error_msg,
                fields            = fields_final,
            )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )
```

---
File: s_ai_model.py
---

```py
import asyncio
import os
import time
import logging
import tiktoken
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

from s_json_utils import extract_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME     = os.environ.get("MODEL_NAME", "gpt-5-nano")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment variables.")

# Models that do NOT support temperature parameter
_FIXED_TEMPERATURE_MODELS = {
    "gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini",
    "o1", "o1-mini", "o3-mini", "o3",
}

# Models that use max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {
    "gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini",
    "o1", "o1-mini", "o3-mini", "o3",
}

TOKEN_CHUNK_SIZE    = 800000
TOKEN_CHUNK_OVERLAP = 500
MAX_OUTPUT_TOKENS   = 4096
MAP_JSON_RETRY_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Module-level semaphore — shared across all requests on this worker
# ---------------------------------------------------------------------------
_MAP_CONCURRENCY  = 3
_MAP_SEMAPHORE: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _MAP_SEMAPHORE
    if _MAP_SEMAPHORE is None:
        _MAP_SEMAPHORE = asyncio.Semaphore(_MAP_CONCURRENCY)
        logger.info(f"[ai_model] MAP semaphore created (concurrency={_MAP_CONCURRENCY})")
    return _MAP_SEMAPHORE


_client   = AsyncOpenAI(api_key=OPENAI_API_KEY)
_encoding = tiktoken.encoding_for_model("gpt-4o")


# ---------------------------------------------------------------------------
# Token-accurate chunking
# ---------------------------------------------------------------------------

def split_by_tokens(text: str) -> list[str]:
    if not text:
        return []
    t0       = time.perf_counter()
    all_ids  = _encoding.encode(text)
    n_tokens = len(all_ids)
    logger.info(
        f"[split_by_tokens] {len(text):,} chars → {n_tokens:,} tokens "
        f"({len(text)/n_tokens:.2f} chars/token) in {time.perf_counter()-t0:.3f}s"
    )
    chunks = []
    start  = 0
    while start < n_tokens:
        end       = min(start + TOKEN_CHUNK_SIZE, n_tokens)
        chunk_ids = all_ids[start:end]
        chunks.append(_encoding.decode(chunk_ids))
        start    += TOKEN_CHUNK_SIZE - TOKEN_CHUNK_OVERLAP
    logger.info(f"[split_by_tokens] → {len(chunks)} chunk(s) of ≤{TOKEN_CHUNK_SIZE} tokens")
    return chunks


# ---------------------------------------------------------------------------
# Prompts for PDF analysis (used by generate_analysis / generate_analysis_stream)
# These prompts contain the word "json" — required by OpenAI when
# response_format=json_object is set.
# ---------------------------------------------------------------------------

_MAP_SYSTEM = (
    "You are a document analysis assistant. Analyse the document excerpt provided. "
    "Output ONLY a JSON object with exactly these fields:\n"
    '{"overview":"what this document is — type, subject, and purpose",'
    '"summary":"cover all key points in this excerpt — as long or short as the content requires",'
    '"highlights":["specific fact with number/name/date","fact","fact"]}\n'
    "CRITICAL: overview and summary MUST be plain strings. "
    "highlights MUST be a flat array of strings. "
    "NEVER use nested objects or nested arrays."
)

_MAP_RETRY_SYSTEM = (
    "Output ONLY this JSON object. Nothing else.\n"
    "overview and summary must be plain strings. highlights must be a flat array of strings.\n"
    '{"overview":"...","summary":"...","highlights":["...","...","..."]}'
)

_SYNTH_SYSTEM = (
    "You are given summaries of consecutive sections of a single document. "
    "Write a JSON object with exactly two fields:\n"
    '{"overview":"describe what the entire document is — its type, subject, and main purpose",'
    '"summary":"cover ALL major topics across the entire document. Work through from start to end. '
    "Be specific — include key subjects, people, figures, decisions, and conclusions. "
    'Do NOT repeat topics. Write as much as needed to accurately represent the full document."}\n'
    "CRITICAL: both fields must be plain strings — no nested objects, no arrays. "
    "No prose outside the JSON."
)


def _build_map_messages(text: str, retry: bool = False) -> list[dict]:
    system = _MAP_RETRY_SYSTEM if retry else _MAP_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Document:\n---\n{text}\n---"},
    ]


def _build_synth_messages(results: list[dict]) -> list[dict]:
    parts = []
    for i, r in enumerate(results, 1):
        overview = r.get("overview", "").strip()
        summary  = r.get("summary",  "").strip()
        if overview or summary:
            lines = [f"[Section {i}]"]
            if overview:
                lines.append(f"Type: {overview}")
            if summary:
                lines.append(f"Content: {summary}")
            parts.append("\n".join(lines))
    body = "\n\n".join(parts)
    return [
        {"role": "system", "content": _SYNTH_SYSTEM},
        {"role": "user",   "content": f"Section summaries (in document order):\n---\n{body}\n---"},
    ]


def _merge_highlights(results: list[dict]) -> list[str]:
    seen:       set  = set()
    highlights: list = []
    for r in results:
        for h in r.get("highlights", []):
            h = h.strip()
            if not h:
                continue
            key = " ".join(h.lower().split())
            if key not in seen:
                seen.add(key)
                highlights.append(h)
    logger.info(f"[_merge_highlights] {len(highlights)} distinct highlight(s) preserved")
    return highlights


# ---------------------------------------------------------------------------
# _build_api_kwargs — shared helper to build model-aware API kwargs
# ---------------------------------------------------------------------------

def _build_api_kwargs(
    messages:   list[dict],
    use_json:   bool = False,
    streaming:  bool = False,
) -> dict:
    """
    Build OpenAI API kwargs handling model differences.

    use_json=True  → sets response_format=json_object
                     ONLY use when messages contain the word "json"
                     (OpenAI requirement) — PDF analysis calls only.

    use_json=False → NO response_format
                     Required for key-clause-extraction, risk-detection,
                     and any plain-text call where prompt lacks "json".
    """
    model = os.environ.get("MODEL_NAME", MODEL_NAME)

    kwargs: dict = {
        "model":    model,
        "messages": messages,
    }

    # temperature: not supported by gpt-5-nano, o1, o3 family
    if model not in _FIXED_TEMPERATURE_MODELS:
        kwargs["temperature"] = 0.0

    # token limit parameter name differs by model
    if model in _MAX_COMPLETION_TOKENS_MODELS:
        kwargs["max_completion_tokens"] = MAX_OUTPUT_TOKENS
    else:
        kwargs["max_tokens"] = MAX_OUTPUT_TOKENS

    # JSON mode — ONLY when prompt contains word "json"
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}

    # streaming
    if streaming:
        kwargs["stream"]         = True
        kwargs["stream_options"] = {"include_usage": True}

    return kwargs


# ---------------------------------------------------------------------------
# _run_inference_json
#
# For: PDF analysis map + synthesis calls
# Routes: POST /analyze, POST /analyze/stream
#
# Sets response_format=json_object.
# Safe because _MAP_SYSTEM and _SYNTH_SYSTEM both contain the word "json".
# ---------------------------------------------------------------------------

async def _run_inference_json(
    messages: list[dict],
    label:    str = "",
) -> tuple[str, int, int]:
    """
    OpenAI call with response_format=json_object.
    Use ONLY when system prompt contains the word 'json'.
    Returns (content, input_tokens, output_tokens).
    """
    tag    = f"[{label}] " if label else ""
    t0     = time.perf_counter()
    kwargs = _build_api_kwargs(messages, use_json=True, streaming=False)

    try:
        response      = await _client.chat.completions.create(**kwargs)
        elapsed       = time.perf_counter() - t0
        content       = response.choices[0].message.content or ""
        input_tokens  = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        logger.info(f"{tag}in={input_tokens} out={output_tokens} in {elapsed:.2f}s")
        return content, input_tokens, output_tokens
    except Exception as e:
        logger.exception(f"{tag}OpenAI API call failed: {e}")
        raise


# ---------------------------------------------------------------------------
# _run_inference_text
#
# For: key-clause-extraction, risk-detection, intent classification
# Routes: POST /key-clause-extraction, POST /detect-risks
#
# NO response_format — plain text output.
# Required because those prompts do NOT contain the word "json" and
# setting json_object would cause a 400 error from OpenAI.
# ---------------------------------------------------------------------------

async def _run_inference_text(
    messages: list[dict],
    label:    str = "",
) -> tuple[str, int, int]:
    """
    OpenAI call WITHOUT response_format — plain text output.
    Use for key-clause-extraction, risk-detection, and any call
    where the prompt does NOT explicitly ask for JSON.
    Returns (content, input_tokens, output_tokens).
    """
    tag    = f"[{label}] " if label else ""
    t0     = time.perf_counter()
    kwargs = _build_api_kwargs(messages, use_json=False, streaming=False)

    try:
        response      = await _client.chat.completions.create(**kwargs)
        elapsed       = time.perf_counter() - t0
        content       = response.choices[0].message.content or ""
        input_tokens  = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        logger.info(f"{tag}in={input_tokens} out={output_tokens} in {elapsed:.2f}s")
        return content, input_tokens, output_tokens
    except Exception as e:
        logger.exception(f"{tag}OpenAI API call failed: {e}")
        raise


# ---------------------------------------------------------------------------
# _run_inference_stream
#
# For: generate_analysis_stream (streaming SSE)
# Route: POST /analyze/stream
#
# No response_format (incompatible with stream=True on some SDK versions).
# Yields: ("delta", str) | ("done", (input_tokens, output_tokens))
# ---------------------------------------------------------------------------

async def _run_inference_stream(messages: list[dict], label: str = ""):
    """
    Streaming OpenAI call — yields token deltas then final token counts.
    No response_format (not needed; extract_json handles parsing).
    """
    tag    = f"[{label}] " if label else ""
    t0     = time.perf_counter()
    kwargs = _build_api_kwargs(messages, use_json=False, streaming=True)

    input_tokens  = 0
    output_tokens = 0

    try:
        async with await _client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield ("delta", chunk.choices[0].delta.content)
                if chunk.usage:
                    input_tokens  = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

        logger.info(
            f"{tag}stream done — in={input_tokens} out={output_tokens} "
            f"({time.perf_counter()-t0:.2f}s)"
        )
    except Exception as e:
        logger.exception(f"{tag}streaming OpenAI call failed: {e}")
        raise

    yield ("done", (input_tokens, output_tokens))


# ---------------------------------------------------------------------------
# run_llm — generic plain-text runner
#
# Routes: POST /key-clause-extraction, POST /detect-risks
#
# Uses _run_inference_TEXT (no response_format) because:
#   - key-clause + risk-detection prompts do NOT contain the word "json"
#   - setting json_object without "json" in messages → 400 error from OpenAI
# ---------------------------------------------------------------------------

async def run_llm(
    text: str,
    system_prompt: str,
    max_input_tokens: int = 50000,
) -> str:
    """
    Generic plain-text LLM runner.
    Used by /key-clause-extraction and /detect-risks.
    Does NOT set response_format — plain text output only.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Document:\n----------------\n{text}\n----------------"},
    ]
    content, _, _ = await _run_inference_text(messages, "run_llm")
    return content


# ---------------------------------------------------------------------------
# _run_map_chunk — one map chunk with semaphore + retry
# Uses _run_inference_json (PDF analysis prompts contain "json")
# ---------------------------------------------------------------------------

async def _run_map_chunk(i: int, total: int, chunk_text: str) -> tuple[dict, int, int]:
    _EMPTY = {"overview": "", "summary": "", "highlights": []}
    sem    = _get_semaphore()
    lbl    = f"map {i+1}/{total}"

    async with sem:
        t_chunk    = time.perf_counter()
        parsed     = None
        in_tokens  = 0
        out_tokens = 0

        for attempt in range(1, MAP_JSON_RETRY_ATTEMPTS + 1):
            try:
                messages = _build_map_messages(chunk_text, retry=(attempt > 1))
                if attempt > 1:
                    logger.warning(f"[generate_analysis] [{lbl}] retry {attempt}")
                raw, i_tok, o_tok = await _run_inference_json(
                    messages, f"{lbl}-a{attempt}"
                )
                in_tokens  += i_tok
                out_tokens += o_tok
            except Exception as e:
                logger.error(f"[generate_analysis] [{lbl}] inference error: {e}")
                break

            candidate = extract_json(raw)
            if candidate.get("overview") or candidate.get("highlights"):
                parsed = candidate
                if attempt > 1:
                    logger.info(f"[generate_analysis] [{lbl}] retry {attempt} produced valid JSON")
                break
            else:
                logger.warning(
                    f"[generate_analysis] [{lbl}] attempt {attempt}: no usable JSON — "
                    + ("retrying" if attempt < MAP_JSON_RETRY_ATTEMPTS else "giving up")
                )

        if not isinstance(parsed, dict) or not (parsed.get("overview") or parsed.get("highlights")):
            parsed = dict(_EMPTY)

        logger.info(
            f"[generate_analysis] [{lbl}] done ({time.perf_counter()-t_chunk:.2f}s) "
            f"overview={'yes' if parsed.get('overview') else 'empty'} "
            f"highlights={len(parsed.get('highlights', []))}"
        )
        return parsed, in_tokens, out_tokens


# ---------------------------------------------------------------------------
# generate_analysis
# Route: POST /analyze
# ---------------------------------------------------------------------------

async def generate_analysis(merged_text: str) -> tuple[dict, int, int]:
    _EMPTY = {"overview": "", "summary": "", "highlights": []}
    if not merged_text or not merged_text.strip():
        return dict(_EMPTY), 0, 0

    t0     = time.perf_counter()
    chunks = split_by_tokens(merged_text)
    logger.info(f"[generate_analysis] {len(chunks)} chunk(s) in {time.perf_counter()-t0:.3f}s")

    t_pipeline    = time.perf_counter()
    total_in_tok  = 0
    total_out_tok = 0

    logger.info(
        f"[generate_analysis] MAP: {len(chunks)} chunk(s) "
        f"(parallel, concurrency={_MAP_CONCURRENCY})"
    )
    map_tasks   = [_run_map_chunk(i, len(chunks), ct) for i, ct in enumerate(chunks)]
    raw_results = list(await asyncio.gather(*map_tasks))

    map_results = []
    for parsed, i_tok, o_tok in raw_results:
        map_results.append(parsed)
        total_in_tok  += i_tok
        total_out_tok += o_tok

    valid_count = sum(1 for r in map_results if r.get("overview") or r.get("highlights"))
    logger.info(f"[generate_analysis] MAP complete — {len(map_results)} total, {valid_count} with content")

    if len(map_results) == 1:
        logger.info("[generate_analysis] single chunk — returning directly")
        result = map_results[0]

    else:
        all_highlights = _merge_highlights(map_results)

        logger.info(f"[generate_analysis] SYNTHESIS: overview+summary from {valid_count} chunk(s)")
        t_synth      = time.perf_counter()
        synth_result = {"overview": "", "summary": ""}
        try:
            synth_messages = _build_synth_messages(
                [r for r in map_results if r.get("overview") or r.get("summary")]
            )
            raw_synth, s_in, s_out = await _run_inference_json(synth_messages, "synthesis")
            total_in_tok  += s_in
            total_out_tok += s_out

            parsed_synth = extract_json(raw_synth)
            ov = parsed_synth.get("overview", "")
            sm = parsed_synth.get("summary",  "")
            synth_result["overview"] = (ov if isinstance(ov, str) else str(ov)).strip()
            synth_result["summary"]  = (sm if isinstance(sm, str) else str(sm)).strip()

            logger.info(
                f"[generate_analysis] SYNTHESIS done ({time.perf_counter()-t_synth:.2f}s) "
                f"overview={'yes' if synth_result['overview'] else 'empty'} "
                f"summary={'yes' if synth_result['summary'] else 'empty'}"
            )
        except Exception as e:
            logger.error(f"[generate_analysis] SYNTHESIS failed: {e} — fallback to first chunk")

        if not synth_result["overview"]:
            synth_result["overview"] = next(
                (r["overview"] for r in map_results if r.get("overview")), ""
            )
        if not synth_result["summary"]:
            synth_result["summary"]  = next(
                (r["summary"]  for r in map_results if r.get("summary")),  ""
            )

        result = {
            "overview":   synth_result["overview"],
            "summary":    synth_result["summary"],
            "highlights": all_highlights,
        }

    if not isinstance(result, dict):
        result = dict(_EMPTY)

    logger.info(
        f"[generate_analysis] total={time.perf_counter()-t_pipeline:.2f}s "
        f"tokens={total_in_tok}in/{total_out_tok}out "
        f"highlights={len(result.get('highlights', []))}"
    )
    return result, total_in_tok, total_out_tok


# ---------------------------------------------------------------------------
# _run_map_chunk_stream — stream one map chunk, yield deltas then result
# Uses _run_inference_stream (no response_format, streaming=True)
# ---------------------------------------------------------------------------

async def _run_map_chunk_stream(i: int, total: int, chunk_text: str):
    _EMPTY = {"overview": "", "summary": "", "highlights": []}
    sem    = _get_semaphore()
    lbl    = f"map {i+1}/{total}"

    async with sem:
        t_chunk    = time.perf_counter()
        in_tokens  = 0
        out_tokens = 0
        parsed     = None

        for attempt in range(1, MAP_JSON_RETRY_ATTEMPTS + 1):
            messages   = _build_map_messages(chunk_text, retry=(attempt > 1))
            raw_buffer = ""

            if attempt > 1:
                logger.warning(f"[stream] [{lbl}] retry {attempt}")

            try:
                async for event_type, payload in _run_inference_stream(
                    messages, f"{lbl}-a{attempt}"
                ):
                    if event_type == "delta":
                        raw_buffer += payload
                        yield ("delta", payload)
                    elif event_type == "done":
                        in_tokens  += payload[0]
                        out_tokens += payload[1]
            except Exception as e:
                logger.error(f"[stream] [{lbl}] inference error: {e}")
                break

            candidate = extract_json(raw_buffer)
            if candidate.get("overview") or candidate.get("highlights"):
                parsed = candidate
                break
            else:
                logger.warning(
                    f"[stream] [{lbl}] attempt {attempt}: no usable JSON — "
                    + ("retrying" if attempt < MAP_JSON_RETRY_ATTEMPTS else "giving up")
                )

        if not isinstance(parsed, dict) or not (parsed.get("overview") or parsed.get("highlights")):
            parsed = dict(_EMPTY)

        logger.info(
            f"[stream] [{lbl}] done ({time.perf_counter()-t_chunk:.2f}s) "
            f"overview={'yes' if parsed.get('overview') else 'empty'} "
            f"highlights={len(parsed.get('highlights', []))}"
        )
        yield ("chunk_done", {**parsed, "_in_tokens": in_tokens, "_out_tokens": out_tokens})


# ---------------------------------------------------------------------------
# generate_analysis_stream
# Route: POST /analyze/stream
# ---------------------------------------------------------------------------

async def generate_analysis_stream(merged_text: str):
    """
    Sequential SSE streaming — each chunk streams tokens live to client.

    Yields (event_type, payload):
      ("chunk_start",    {"chunk": N, "total": N})
      ("token",          {"chunk": N, "delta": str})
      ("chunk_done",     {"chunk": N, "total": N, "overview": str,
                          "new_highlights": [...], "all_highlights_so_far": [...]})
      ("synthesis_start", {})
      ("token",          {"chunk": "synthesis", "delta": str})
      ("synthesis_done", {"overview": str, "summary": str})
      ("done",           {"input_tokens": N, "output_tokens": N, "total_tokens": N})
    """
    if not merged_text or not merged_text.strip():
        yield ("done", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        return

    t0            = time.perf_counter()
    chunks        = split_by_tokens(merged_text)
    total_in_tok  = 0
    total_out_tok = 0
    map_results:    list = []
    all_highlights: list = []
    seen_keys:      set  = set()

    logger.info(f"[generate_analysis_stream] {len(chunks)} chunk(s) — sequential streaming")

    # ── MAP — sequential so tokens reach the client immediately ──────────
    for i, chunk_text in enumerate(chunks):
        yield ("chunk_start", {"chunk": i + 1, "total": len(chunks)})

        parsed = None
        async for event_type, payload in _run_map_chunk_stream(i, len(chunks), chunk_text):
            if event_type == "delta":
                yield ("token", {"chunk": i + 1, "delta": payload})

            elif event_type == "chunk_done":
                parsed         = payload
                total_in_tok  += payload.pop("_in_tokens",  0)
                total_out_tok += payload.pop("_out_tokens", 0)

                new_highlights = []
                for h in parsed.get("highlights", []):
                    h = h.strip()
                    if not h:
                        continue
                    key = " ".join(h.lower().split())
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_highlights.append(h)
                        new_highlights.append(h)

                map_results.append(parsed)

                yield ("chunk_done", {
                    "chunk":                 i + 1,
                    "total":                 len(chunks),
                    "overview":              parsed.get("overview", ""),
                    "new_highlights":        new_highlights,
                    "all_highlights_so_far": list(all_highlights),
                })
                logger.info(
                    f"[stream] [map {i+1}/{len(chunks)}] yielded — "
                    f"overview={'yes' if parsed.get('overview') else 'empty'} "
                    f"new_highlights={len(new_highlights)} total={len(all_highlights)}"
                )

    valid_count = sum(1 for r in map_results if r.get("overview") or r.get("highlights"))
    logger.info(f"[stream] MAP complete — {len(map_results)} total, {valid_count} with content")

    # ── SYNTHESIS ────────────────────────────────────────────────────────
    synth_overview = ""
    synth_summary  = ""

    if len(map_results) == 1:
        synth_overview = map_results[0].get("overview", "")
        synth_summary  = map_results[0].get("summary",  "")
        yield ("synthesis_done", {"overview": synth_overview, "summary": synth_summary})

    else:
        yield ("synthesis_start", {})
        synth_buffer = ""
        try:
            synth_messages = _build_synth_messages(
                [r for r in map_results if r.get("overview") or r.get("summary")]
            )
            async for event_type, payload in _run_inference_stream(synth_messages, "synthesis"):
                if event_type == "delta":
                    synth_buffer += payload
                    yield ("token", {"chunk": "synthesis", "delta": payload})
                elif event_type == "done":
                    total_in_tok  += payload[0]
                    total_out_tok += payload[1]

            parsed_synth = extract_json(synth_buffer)
            ov = parsed_synth.get("overview", "")
            sm = parsed_synth.get("summary",  "")
            synth_overview = (ov if isinstance(ov, str) else str(ov)).strip()
            synth_summary  = (sm if isinstance(sm, str) else str(sm)).strip()
            logger.info(
                f"[stream] SYNTHESIS done — "
                f"overview={'yes' if synth_overview else 'empty'} "
                f"summary={'yes' if synth_summary else 'empty'}"
            )
        except Exception as e:
            logger.error(f"[stream] SYNTHESIS failed: {e} — fallback to first chunk")

        if not synth_overview:
            synth_overview = next(
                (r["overview"] for r in map_results if r.get("overview")), ""
            )
        if not synth_summary:
            synth_summary  = next(
                (r["summary"]  for r in map_results if r.get("summary")),  ""
            )

        yield ("synthesis_done", {"overview": synth_overview, "summary": synth_summary})

    logger.info(
        f"[stream] total={time.perf_counter()-t0:.2f}s "
        f"tokens={total_in_tok}in/{total_out_tok}out "
        f"highlights={len(all_highlights)}"
    )
    yield ("done", {
        "input_tokens":  total_in_tok,
        "output_tokens": total_out_tok,
        "total_tokens":  total_in_tok + total_out_tok,
    })
```

---
File: s_pdf_to_docx.py
---

```py
import re
import fitz
import time
import logging
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structure detection helpers
# ---------------------------------------------------------------------------

def _detect_block_type(text: str, font_size: float = 12, is_bold: bool = False) -> str:
    """
    Classify a text block as heading, subheading, list item, or paragraph.
    """
    text = text.strip()
    if not text:
        return "empty"

    # Heading: short, bold or large font, no full stop at end
    if (is_bold or font_size >= 16) and len(text) < 120 and not text.endswith("."):
        return "heading1"

    if font_size >= 13 and len(text) < 150 and not text.endswith("."):
        return "heading2"

    # List item: starts with bullet, dash, number
    if re.match(r'^[\•\-\*\u2022\u2023\u25E6]\s+', text):
        return "list_bullet"
    if re.match(r'^\d+[\.\)]\s+', text):
        return "list_number"

    return "paragraph"


def _extract_blocks_native(page: fitz.Page) -> list[dict]:
    """
    Extract text blocks from a native PDF page with font metadata.
    Returns list of {text, font_size, is_bold, block_type}
    """
    blocks = []
    try:
        raw_blocks = page.get_text("dict")["blocks"]
        for block in raw_blocks:
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                line_text  = ""
                font_size  = 12
                is_bold    = False
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                    font_size  = span.get("size", 12)
                    flags      = span.get("flags", 0)
                    is_bold    = bool(flags & 2**4)  # bit 4 = bold

                line_text = line_text.strip()
                if not line_text:
                    continue

                blocks.append({
                    "text":       line_text,
                    "font_size":  font_size,
                    "is_bold":    is_bold,
                    "block_type": _detect_block_type(line_text, font_size, is_bold),
                })
    except Exception as e:
        logger.warning(f"_extract_blocks_native: {e}")

    return blocks


def _extract_blocks_ocr(ocr_text: str) -> list[dict]:
    """
    Convert flat OCR text into structured blocks.
    Without font metadata, use heuristics on line length and content.
    """
    blocks = []
    for line in ocr_text.splitlines():
        line = line.strip()
        if not line:
            continue
        blocks.append({
            "text":       line,
            "font_size":  12,
            "is_bold":    False,
            "block_type": _detect_block_type(line),
        })
    return blocks


# ---------------------------------------------------------------------------
# DOCX builder
# ---------------------------------------------------------------------------

def _build_docx(all_blocks: list[list[dict]], source_filename: str) -> Document:
    """
    Build a DOCX Document from extracted blocks.
    all_blocks is a list of pages, each page is a list of block dicts.
    """
    doc = Document()

    # Document title from filename
    title_para = doc.add_heading(
        source_filename.replace(".pdf", "").replace("_", " ").title(), level=0
    )
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for page_num, page_blocks in enumerate(all_blocks, 1):
        # Page separator (except before first page)
        if page_num > 1:
            doc.add_paragraph()  # spacing between pages

        for block in page_blocks:
            btype = block["block_type"]
            text  = block["text"]

            if btype == "empty":
                continue

            elif btype == "heading1":
                doc.add_heading(text, level=1)

            elif btype == "heading2":
                doc.add_heading(text, level=2)

            elif btype == "list_bullet":
                # Strip leading bullet character
                clean = re.sub(r'^[\•\-\*\u2022\u2023\u25E6]\s+', '', text)
                doc.add_paragraph(clean, style="List Bullet")

            elif btype == "list_number":
                clean = re.sub(r'^\d+[\.\)]\s+', '', text)
                doc.add_paragraph(clean, style="List Number")

            else:  # paragraph
                para = doc.add_paragraph(text)
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    return doc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pdf_to_docx(file_path: str, filename: str) -> Document:
    """
    Convert a PDF file to a DOCX Document object.

    Uses:
      - PyMuPDF native extraction with font metadata for text PDFs
      - Falls back to flat OCR text (passed in) for scanned PDFs

    Returns a python-docx Document ready to be saved.
    """
    t0 = time.perf_counter()
    logger.info(f"[pdf_to_docx] converting '{filename}'")

    try:
        doc_fitz = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    all_blocks = []

    for page_num in range(len(doc_fitz)):
        page   = doc_fitz[page_num]
        blocks = _extract_blocks_native(page)

        # If page has no native text, mark for OCR
        if not blocks:
            logger.info(f"[pdf_to_docx] page {page_num+1}: no native text — needs OCR")
            # OCR is handled in the endpoint using existing PaddleOCR-VL
            # Placeholder so page count stays accurate
            blocks = [{"text": f"[Page {page_num+1}: scanned content]",
                       "font_size": 12, "is_bold": False, "block_type": "paragraph"}]

        all_blocks.append(blocks)

    doc_fitz.close()

    docx = _build_docx(all_blocks, filename)
    logger.info(f"[pdf_to_docx] done ({time.perf_counter()-t0:.2f}s) — {len(all_blocks)} page(s)")
    return docx
```

---
File: requirements.txt
---

```txt
# ============================================================
# PDF AI Review Service — requirements.txt
# Python 3.10 | OpenAI API (gpt-4.1-nano)
# ============================================================

# ------------------------------------------------------------
# Web framework & serving
# ------------------------------------------------------------
fastapi==0.129.0
uvicorn==0.41.0
gunicorn==25.1.0
starlette==0.52.1
python-multipart==0.0.22
httpx==0.28.1
httpcore==1.0.9
anyio==4.12.1
sniffio==1.3.1
h11==0.16.0

# ------------------------------------------------------------
# OpenAI API
# ------------------------------------------------------------
openai>=1.30.0

# ------------------------------------------------------------
# PaddleOCR-VL
# NOTE: install PaddlePaddle GPU separately before pip install -r requirements.txt
#   pip install paddlepaddle-gpu==3.2.0 \
#     -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
# ------------------------------------------------------------
paddleocr==3.4.0
paddlex==3.4.2

# ------------------------------------------------------------
# PDF processing
# ------------------------------------------------------------
PyMuPDF==1.27.1
pypdfium2==5.6.0
pytesseract==0.3.13
python-docx

# ------------------------------------------------------------
# LangChain
# ------------------------------------------------------------
langchain==0.0.352
langchain-community==0.0.20
langchain-core==0.1.23
langsmith==0.0.87

# ------------------------------------------------------------
# Image / CV processing
# ------------------------------------------------------------
numpy==1.26.4
pillow==12.0.0
opencv-contrib-python==4.10.0.84
shapely==2.1.2

# ------------------------------------------------------------
# Data / ML utilities
# ------------------------------------------------------------
scikit-learn==1.7.2
scipy==1.15.3
pandas==2.3.3
tiktoken==0.12.0
sentencepiece==0.2.1
einops==0.8.2

# ------------------------------------------------------------
# Core utilities
# ------------------------------------------------------------
pydantic==2.12.5
pydantic_core==2.41.5
requests==2.32.5
urllib3==2.6.3
certifi==2026.1.4
charset-normalizer==3.4.4
chardet==7.1.0
packaging==23.2
typing_extensions==4.15.0
PyYAML==6.0.2
regex==2026.1.15
tqdm==4.67.3
filelock==3.20.0
fsspec==2025.12.0
psutil==7.2.2
sympy==1.13.1
networkx==3.4.2
Jinja2==3.1.6
MarkupSafe==2.1.5
click==8.3.1
six==1.17.0
rich==14.3.2

pdf2docx
python-docx
python-dotenv
reportlab
python-docx

```

---
File: export.py
---

```py
import os
from pathlib import Path

OUTPUT_FILE = "project_structure.md"

# Directories to ignore anywhere in project
EXCLUDED_DIRS = {
    ".git",
    ".vscode",
    "__pycache__",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "coverage",
}

# Files to ignore
EXCLUDED_FILES = {
    OUTPUT_FILE,
    "db.sqlite3",
    ".env",
    "README.md",
    ".htaccess",
    "app.log",
    "uvicorn.log",
    "gunicorn.log"
}

# Binary file extensions
BINARY_EXTENSIONS = {
    ".png",".jpg",".jpeg",".gif",".ico",".svg",".webp",
    ".woff",".woff2",".ttf",".eot",".otf",
    ".pdf",".zip",".gz",".tar",".rar",
    ".exe",".dll",".so",".a",".lib",".jar",".mp3"
}


def is_binary_file(file_path: Path):
    return file_path.suffix.lower() in BINARY_EXTENSIONS


def build_tree(start_path: Path):
    tree_lines = []

    for root, dirs, files in os.walk(start_path):

        # Remove excluded dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        level = root.replace(str(start_path), "").count(os.sep)
        indent = "│   " * level

        folder_name = os.path.basename(root) if level != 0 else "."
        tree_lines.append(f"{indent}├── {folder_name}/")

        sub_indent = "│   " * (level + 1)

        for f in files:
            if f in EXCLUDED_FILES:
                continue
            tree_lines.append(f"{sub_indent}├── {f}")

    return "\n".join(tree_lines)


def collect_files(start_path: Path):
    collected = []

    for root, dirs, files in os.walk(start_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for f in files:
            if f in EXCLUDED_FILES:
                continue

            file_path = Path(root) / f

            if is_binary_file(file_path):
                continue

            collected.append(file_path)

    return collected


def main():
    project_root = Path(".")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        out.write("# Django Project Structure\n\n")

        out.write("```\n")
        out.write(build_tree(project_root))
        out.write("\n```\n\n")

        out.write("# File Contents\n\n")

        for file_path in collect_files(project_root):

            rel_path = file_path.relative_to(project_root)
            extension = file_path.suffix.replace(".", "") or "text"

            out.write("---\n")
            out.write(f"File: {rel_path}\n")
            out.write("---\n\n")

            out.write(f"```{extension}\n")

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    out.write(f.read())
            except:
                out.write("<< Could not read file >>")

            out.write("\n```\n\n")

    print(f"✅ Django project exported to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()

```

---
File: access.log
---

```log
89.105.216.247:47950 - "POST /analyze HTTP/1.1" 200
89.105.216.247:35850 - "POST /analyze HTTP/1.1" 200
89.105.216.247:42988 - "POST /analyze HTTP/1.1" 200
89.105.216.247:40314 - "POST /analyze HTTP/1.1" 200
89.105.216.247:55478 - "POST /analyze HTTP/1.1" 200
89.105.216.247:49996 - "POST /analyze HTTP/1.1" 200
89.105.216.247:59144 - "POST /analyze HTTP/1.1" 200
89.105.216.247:51244 - "POST /analyze HTTP/1.1" 200
103.70.177.179:42310 - "GET /docs HTTP/1.1" 200
103.70.177.179:42310 - "GET /openapi.json HTTP/1.1" 200
103.70.177.179:42310 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
103.70.177.179:57578 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
103.70.177.179:28667 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
103.70.177.179:60663 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
23.106.66.131:52944 - "POST /analyze HTTP/1.1" 200
23.106.66.131:53124 - "POST /key-clause-extraction HTTP/1.1" 500
23.106.66.131:53970 - "POST /analyze HTTP/1.1" 200
23.106.66.131:55460 - "POST /analyze HTTP/1.1" 200
23.106.66.131:57749 - "POST /analyze HTTP/1.1" 200
23.106.66.131:58582 - "POST /analyze HTTP/1.1" 200
103.70.177.179:45416 - "GET /docs HTTP/1.1" 200
103.70.177.179:45416 - "GET /openapi.json HTTP/1.1" 200
103.70.177.179:45416 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
103.70.177.179:45416 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
91.230.168.67:45633 - "GET / HTTP/1.1" 404
91.230.168.38:57113 - "GET /favicon.ico HTTP/1.1" 404
103.70.177.179:28659 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
23.106.66.131:54522 - "POST /analyze HTTP/1.1" 200
64.227.183.181:43834 - "POST /analyze HTTP/1.1" 200
23.106.66.131:60827 - "POST /analyze HTTP/1.1" 200
23.106.66.131:61357 - "POST /analyze HTTP/1.1" 200
23.106.66.131:61418 - "POST /analyze HTTP/1.1" 200
23.106.66.131:65413 - "POST /analyze HTTP/1.1" 200
103.70.177.179:26987 - "POST /analyze?analysis_type=0 HTTP/1.1" 500
103.70.177.179:34026 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
23.106.66.131:52284 - "POST /analyze HTTP/1.1" 200
66.132.224.89:39898 - "GET / HTTP/1.1" 404
66.132.224.89:39900 - "PRI %2A HTTP/2.0" 404
66.132.224.89:48606 - "GET /.well-known/security.txt HTTP/1.1" 404
89.105.216.247:60092 - "POST /analyze HTTP/1.1" 200
64.227.183.181:33140 - "POST /analyze HTTP/1.1" 200
64.227.183.181:45014 - "POST /analyze HTTP/1.1" 200
209.38.132.35:52716 - "GET / HTTP/1.1" 404
209.38.132.35:52722 - "GET /favicon.ico HTTP/1.1" 404
199.45.154.124:12360 - "GET / HTTP/1.1" 404
199.45.154.124:46658 - "GET / HTTP/1.1" 404
199.45.154.124:46664 - "PRI %2A HTTP/2.0" 404
199.45.154.124:47978 - "GET /.well-known/security.txt HTTP/1.1" 404
65.49.1.10:33154 - "GET / HTTP/1.1" 404
65.49.1.18:6753 - "GET /favicon.ico HTTP/1.1" 404
65.49.1.17:22633 - "GET http%3A//api.ipify.org/?format=json HTTP/1.1" 404
65.49.1.11:24447 - "CONNECT www.shadowserver.org%3A443 HTTP/1.1" 404
85.11.183.21:52632 - "GET / HTTP/1.1" 404
16.58.56.214:54786 - "GET / HTTP/1.1" 404
16.58.56.214:56118 - "GET / HTTP/1.1" 404
89.105.216.247:44588 - "POST /analyze HTTP/1.1" 200
89.105.216.247:45100 - "POST /analyze HTTP/1.1" 200
103.70.177.179:8585 - "GET /docs HTTP/1.1" 200
103.70.177.179:8585 - "GET /openapi.json HTTP/1.1" 200
103.70.177.179:8585 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
23.106.66.131:54759 - "POST /analyze HTTP/1.1" 200
23.106.66.131:64877 - "GET /docs HTTP/1.1" 200
23.106.66.175:50582 - "GET / HTTP/1.1" 404
103.70.177.179:18110 - "GET /docs HTTP/1.1" 200
23.106.66.175:50582 - "GET /favicon.ico HTTP/1.1" 404
103.70.177.179:18110 - "GET /openapi.json HTTP/1.1" 200
23.106.66.131:64877 - "GET /openapi.json HTTP/1.1" 200
103.70.177.179:18110 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
23.106.66.131:64877 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
89.105.216.247:50476 - "POST /analyze HTTP/1.1" 200
23.106.66.131:49760 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
3.130.168.2:43764 - "GET / HTTP/1.1" 404
3.130.168.2:49696 - "GET / HTTP/1.1" 404
23.106.66.131:51105 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
23.106.66.131:57521 - "GET /docs HTTP/1.1" 200
23.106.66.131:57521 - "GET /openapi.json HTTP/1.1" 200
89.105.216.247:58112 - "POST /analyze HTTP/1.1" 200
103.70.177.179:7899 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
23.106.66.131:59514 - "POST /analyze HTTP/1.1" 200
23.106.66.131:59749 - "POST /analyze HTTP/1.1" 200
23.106.66.131:59933 - "POST /analyze HTTP/1.1" 200
23.106.66.131:60191 - "POST /analyze HTTP/1.1" 200
103.70.177.179:43056 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
103.70.177.179:55047 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
89.105.216.247:36356 - "POST /analyze HTTP/1.1" 200
103.70.177.179:55047 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
23.106.66.131:63036 - "POST /analyze?analysis_type=0 HTTP/1.1" 200
64.227.183.181:39090 - "POST /analyze HTTP/1.1" 200
23.106.66.131:58832 - "GET /analyze HTTP/1.1" 405
23.106.66.131:58832 - "GET /favicon.ico HTTP/1.1" 404
23.106.66.131:58832 - "GET /docs HTTP/1.1" 200
23.106.66.131:58832 - "GET /openapi.json HTTP/1.1" 200
23.106.66.131:58832 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
103.70.177.179:34193 - "GET /docs HTTP/1.1" 200
103.70.177.179:34193 - "GET /openapi.json HTTP/1.1" 200
103.70.177.179:34193 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
103.70.177.179:54101 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
103.70.177.179:54101 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
103.70.177.179:54101 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
103.70.177.179:24265 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
23.106.66.131:50450 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
23.106.66.131:57637 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
64.227.183.181:50600 - "POST /analyze HTTP/1.1" 200
23.106.66.131:62577 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
23.106.66.131:52558 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
23.106.66.131:54343 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
23.106.66.131:54536 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
23.106.66.131:54694 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
23.106.66.131:56411 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
23.106.66.131:56617 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
23.106.66.131:55744 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
23.106.66.131:56722 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
23.106.66.131:56909 - "POST /analyze/stream?analysis_type=0 HTTP/1.1" 200
64.227.183.181:34660 - "POST /analyze HTTP/1.1" 200
66.132.186.185:37874 - "GET / HTTP/1.1" 404
66.132.186.185:37920 - "PRI %2A HTTP/2.0" 404
66.132.186.185:40448 - "GET /login HTTP/1.1" 404
85.11.183.21:33630 - "GET / HTTP/1.1" 404
194.164.107.5:33994 - "GET / HTTP/1.1" 404
194.164.107.5:52648 - "GET /favicon.ico HTTP/1.1" 404
64.227.183.181:57752 - "POST /analyze HTTP/1.1" 200
64.62.156.192:58724 - "GET / HTTP/1.1" 404
64.62.156.194:58857 - "GET /favicon.ico HTTP/1.1" 404
64.62.156.197:60219 - "GET http%3A//api.ipify.org/?format=json HTTP/1.1" 404
64.62.156.193:9527 - "CONNECT www.shadowserver.org%3A443 HTTP/1.1" 404
64.227.183.181:47176 - "POST /analyze HTTP/1.1" 200
64.227.183.181:45382 - "POST /analyze HTTP/1.1" 200
89.105.216.247:37004 - "POST /analyze HTTP/1.1" 200
64.227.183.181:40564 - "POST /analyze HTTP/1.1" 200
103.70.177.179:47982 - "GET /docs HTTP/1.1" 200
103.70.177.179:47982 - "GET /openapi.json HTTP/1.1" 200
103.70.177.179:7751 - "GET /docs HTTP/1.1" 200
103.70.177.179:7751 - "GET /openapi.json HTTP/1.1" 200
103.70.177.179:7751 - "POST /documents/generate HTTP/1.1" 200
103.70.177.179:11564 - "POST /documents/generate HTTP/1.1" 500
103.70.177.179:30565 - "POST /documents/generate HTTP/1.1" 200
103.70.177.179:33799 - "POST /documents/generate HTTP/1.1" 500
103.70.177.179:31433 - "POST /documents/generate HTTP/1.1" 500
64.227.183.181:53566 - "POST /analyze HTTP/1.1" 200
64.227.183.181:51304 - "POST /analyze HTTP/1.1" 200
64.227.183.181:35004 - "POST /analyze HTTP/1.1" 200
134.209.202.96:34578 - "GET / HTTP/1.1" 404
134.209.202.96:34584 - "GET /favicon.ico HTTP/1.1" 404
66.132.186.188:36172 - "GET / HTTP/1.1" 404
66.132.186.188:57232 - "PRI %2A HTTP/2.0" 404
66.132.186.188:23046 - "GET /security.txt HTTP/1.1" 404
64.227.183.181:54880 - "POST /analyze HTTP/1.1" 200
185.224.128.16:38340 - "CONNECT www.example.com%3A443 HTTP/1.1" 404
103.70.177.179:31434 - "GET /docs HTTP/1.1" 200
103.70.177.179:31434 - "GET /openapi.json HTTP/1.1" 200
103.70.177.179:31434 - "POST /documents/generate HTTP/1.1" 500
103.70.177.179:1819 - "POST /documents/generate HTTP/1.1" 500
103.70.177.179:59111 - "POST /documents/generate HTTP/1.1" 500
3.129.187.38:45844 - "GET / HTTP/1.1" 404
3.129.187.38:50234 - "GET / HTTP/1.1" 404

```

---
File: server.log
---

```log
nohup: ignoring input
`torch_dtype` is deprecated! Use `dtype` instead!

Loading weights:   0%|          | 0/291 [00:00<?, ?it/s]
Loading weights:   0%|          | 1/291 [00:00<00:00, 1482.61it/s, Materializing param=lm_head.weight]
Loading weights:   0%|          | 1/291 [00:00<00:00, 1281.09it/s, Materializing param=lm_head.weight]
Loading weights:   1%|          | 2/291 [00:00<00:28, 10.26it/s, Materializing param=lm_head.weight]  
Loading weights:   1%|          | 2/291 [00:00<00:28, 10.26it/s, Materializing param=model.embed_tokens.weight]
Loading weights:   1%|          | 2/291 [00:00<00:28, 10.26it/s, Materializing param=model.embed_tokens.weight]
Loading weights:   1%|          | 3/291 [00:00<00:28, 10.26it/s, Materializing param=model.layers.0.input_layernorm.weight]
Loading weights:   1%|          | 3/291 [00:00<00:28, 10.26it/s, Materializing param=model.layers.0.input_layernorm.weight]
Loading weights:   1%|▏         | 4/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.mlp.down_proj.weight]  
Loading weights:   1%|▏         | 4/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.mlp.down_proj.weight]
Loading weights:   2%|▏         | 5/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.mlp.gate_proj.weight]
Loading weights:   2%|▏         | 5/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.mlp.gate_proj.weight]
Loading weights:   2%|▏         | 6/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.mlp.up_proj.weight]  
Loading weights:   2%|▏         | 6/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.mlp.up_proj.weight]
Loading weights:   2%|▏         | 7/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.post_attention_layernorm.weight]
Loading weights:   2%|▏         | 7/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.post_attention_layernorm.weight]
Loading weights:   3%|▎         | 8/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.self_attn.k_proj.weight]        
Loading weights:   3%|▎         | 8/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.self_attn.k_proj.weight]
Loading weights:   3%|▎         | 9/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.self_attn.o_proj.weight]
Loading weights:   3%|▎         | 9/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.self_attn.o_proj.weight]
Loading weights:   3%|▎         | 10/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.self_attn.q_proj.weight]
Loading weights:   3%|▎         | 10/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.self_attn.q_proj.weight]
Loading weights:   4%|▍         | 11/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.self_attn.v_proj.weight]
Loading weights:   4%|▍         | 11/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.0.self_attn.v_proj.weight]
Loading weights:   4%|▍         | 12/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.1.input_layernorm.weight] 
Loading weights:   4%|▍         | 12/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.1.input_layernorm.weight]
Loading weights:   4%|▍         | 13/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.1.mlp.down_proj.weight]  
Loading weights:   4%|▍         | 13/291 [00:00<00:27, 10.26it/s, Materializing param=model.layers.1.mlp.down_proj.weight]
Loading weights:   5%|▍         | 14/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.mlp.down_proj.weight]
Loading weights:   5%|▍         | 14/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.mlp.gate_proj.weight]
Loading weights:   5%|▍         | 14/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.mlp.gate_proj.weight]
Loading weights:   5%|▌         | 15/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.mlp.up_proj.weight]  
Loading weights:   5%|▌         | 15/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.mlp.up_proj.weight]
Loading weights:   5%|▌         | 16/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.post_attention_layernorm.weight]
Loading weights:   5%|▌         | 16/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.post_attention_layernorm.weight]
Loading weights:   6%|▌         | 17/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.self_attn.k_proj.weight]        
Loading weights:   6%|▌         | 17/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.self_attn.k_proj.weight]
Loading weights:   6%|▌         | 18/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.self_attn.o_proj.weight]
Loading weights:   6%|▌         | 18/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.self_attn.o_proj.weight]
Loading weights:   7%|▋         | 19/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.self_attn.q_proj.weight]
Loading weights:   7%|▋         | 19/291 [00:00<00:07, 38.78it/s, Materializing param=model.layers.1.self_attn.q_proj.weight]
Loading weights:   7%|▋         | 20/291 [00:00<00:06, 38.78it/s, Materializing param=model.layers.1.self_attn.v_proj.weight]
Loading weights:   7%|▋         | 20/291 [00:00<00:06, 38.78it/s, Materializing param=model.layers.1.self_attn.v_proj.weight]
Loading weights:   7%|▋         | 21/291 [00:00<00:06, 38.78it/s, Materializing param=model.layers.2.input_layernorm.weight] 
Loading weights:   7%|▋         | 21/291 [00:00<00:06, 38.78it/s, Materializing param=model.layers.2.input_layernorm.weight]
Loading weights:   8%|▊         | 22/291 [00:00<00:06, 38.78it/s, Materializing param=model.layers.2.mlp.down_proj.weight]  
Loading weights:   8%|▊         | 22/291 [00:00<00:06, 38.78it/s, Materializing param=model.layers.2.mlp.down_proj.weight]
Loading weights:   8%|▊         | 23/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.mlp.down_proj.weight]
Loading weights:   8%|▊         | 23/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.mlp.gate_proj.weight]
Loading weights:   8%|▊         | 23/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.mlp.gate_proj.weight]
Loading weights:   8%|▊         | 24/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.mlp.up_proj.weight]  
Loading weights:   8%|▊         | 24/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.mlp.up_proj.weight]
Loading weights:   9%|▊         | 25/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.post_attention_layernorm.weight]
Loading weights:   9%|▊         | 25/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.post_attention_layernorm.weight]
Loading weights:   9%|▉         | 26/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.self_attn.k_proj.weight]        
Loading weights:   9%|▉         | 26/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.self_attn.k_proj.weight]
Loading weights:   9%|▉         | 27/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.self_attn.o_proj.weight]
Loading weights:   9%|▉         | 27/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.self_attn.o_proj.weight]
Loading weights:  10%|▉         | 28/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.self_attn.q_proj.weight]
Loading weights:  10%|▉         | 28/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.self_attn.q_proj.weight]
Loading weights:  10%|▉         | 29/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.self_attn.v_proj.weight]
Loading weights:  10%|▉         | 29/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.2.self_attn.v_proj.weight]
Loading weights:  10%|█         | 30/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.3.input_layernorm.weight] 
Loading weights:  10%|█         | 30/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.3.input_layernorm.weight]
Loading weights:  11%|█         | 31/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.3.mlp.down_proj.weight]  
Loading weights:  11%|█         | 31/291 [00:00<00:04, 53.73it/s, Materializing param=model.layers.3.mlp.down_proj.weight]
Loading weights:  11%|█         | 32/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.mlp.down_proj.weight]
Loading weights:  11%|█         | 32/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.mlp.gate_proj.weight]
Loading weights:  11%|█         | 32/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.mlp.gate_proj.weight]
Loading weights:  11%|█▏        | 33/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.mlp.up_proj.weight]  
Loading weights:  11%|█▏        | 33/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.mlp.up_proj.weight]
Loading weights:  12%|█▏        | 34/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.post_attention_layernorm.weight]
Loading weights:  12%|█▏        | 34/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.post_attention_layernorm.weight]
Loading weights:  12%|█▏        | 35/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.self_attn.k_proj.weight]        
Loading weights:  12%|█▏        | 35/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.self_attn.k_proj.weight]
Loading weights:  12%|█▏        | 36/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.self_attn.o_proj.weight]
Loading weights:  12%|█▏        | 36/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.self_attn.o_proj.weight]
Loading weights:  13%|█▎        | 37/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.self_attn.q_proj.weight]
Loading weights:  13%|█▎        | 37/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.self_attn.q_proj.weight]
Loading weights:  13%|█▎        | 38/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.self_attn.v_proj.weight]
Loading weights:  13%|█▎        | 38/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.3.self_attn.v_proj.weight]
Loading weights:  13%|█▎        | 39/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.4.input_layernorm.weight] 
Loading weights:  13%|█▎        | 39/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.4.input_layernorm.weight]
Loading weights:  14%|█▎        | 40/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.4.mlp.down_proj.weight]  
Loading weights:  14%|█▎        | 40/291 [00:00<00:04, 52.24it/s, Materializing param=model.layers.4.mlp.down_proj.weight]
Loading weights:  14%|█▍        | 41/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.mlp.down_proj.weight]
Loading weights:  14%|█▍        | 41/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.mlp.gate_proj.weight]
Loading weights:  14%|█▍        | 41/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.mlp.gate_proj.weight]
Loading weights:  14%|█▍        | 42/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.mlp.up_proj.weight]  
Loading weights:  14%|█▍        | 42/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.mlp.up_proj.weight]
Loading weights:  15%|█▍        | 43/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.post_attention_layernorm.weight]
Loading weights:  15%|█▍        | 43/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.post_attention_layernorm.weight]
Loading weights:  15%|█▌        | 44/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.self_attn.k_proj.weight]        
Loading weights:  15%|█▌        | 44/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.self_attn.k_proj.weight]
Loading weights:  15%|█▌        | 45/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.self_attn.o_proj.weight]
Loading weights:  15%|█▌        | 45/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.self_attn.o_proj.weight]
Loading weights:  16%|█▌        | 46/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.self_attn.q_proj.weight]
Loading weights:  16%|█▌        | 46/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.self_attn.q_proj.weight]
Loading weights:  16%|█▌        | 47/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.self_attn.v_proj.weight]
Loading weights:  16%|█▌        | 47/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.4.self_attn.v_proj.weight]
Loading weights:  16%|█▋        | 48/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.5.input_layernorm.weight] 
Loading weights:  16%|█▋        | 48/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.5.input_layernorm.weight]
Loading weights:  17%|█▋        | 49/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.5.mlp.down_proj.weight]  
Loading weights:  17%|█▋        | 49/291 [00:00<00:04, 54.94it/s, Materializing param=model.layers.5.mlp.down_proj.weight]
Loading weights:  17%|█▋        | 50/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.mlp.down_proj.weight]
Loading weights:  17%|█▋        | 50/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.mlp.gate_proj.weight]
Loading weights:  17%|█▋        | 50/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.mlp.gate_proj.weight]
Loading weights:  18%|█▊        | 51/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.mlp.up_proj.weight]  
Loading weights:  18%|█▊        | 51/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.mlp.up_proj.weight]
Loading weights:  18%|█▊        | 52/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.post_attention_layernorm.weight]
Loading weights:  18%|█▊        | 52/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.post_attention_layernorm.weight]
Loading weights:  18%|█▊        | 53/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.self_attn.k_proj.weight]        
Loading weights:  18%|█▊        | 53/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.self_attn.k_proj.weight]
Loading weights:  19%|█▊        | 54/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.self_attn.o_proj.weight]
Loading weights:  19%|█▊        | 54/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.self_attn.o_proj.weight]
Loading weights:  19%|█▉        | 55/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.self_attn.q_proj.weight]
Loading weights:  19%|█▉        | 55/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.self_attn.q_proj.weight]
Loading weights:  19%|█▉        | 56/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.self_attn.v_proj.weight]
Loading weights:  19%|█▉        | 56/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.5.self_attn.v_proj.weight]
Loading weights:  20%|█▉        | 57/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.6.input_layernorm.weight] 
Loading weights:  20%|█▉        | 57/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.6.input_layernorm.weight]
Loading weights:  20%|█▉        | 58/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.6.mlp.down_proj.weight]  
Loading weights:  20%|█▉        | 58/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.6.mlp.down_proj.weight]
Loading weights:  20%|██        | 59/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.6.mlp.gate_proj.weight]
Loading weights:  20%|██        | 59/291 [00:01<00:04, 54.46it/s, Materializing param=model.layers.6.mlp.gate_proj.weight]
Loading weights:  21%|██        | 60/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.mlp.gate_proj.weight]
Loading weights:  21%|██        | 60/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.mlp.up_proj.weight]  
Loading weights:  21%|██        | 60/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.mlp.up_proj.weight]
Loading weights:  21%|██        | 61/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.post_attention_layernorm.weight]
Loading weights:  21%|██        | 61/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.post_attention_layernorm.weight]
Loading weights:  21%|██▏       | 62/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.self_attn.k_proj.weight]        
Loading weights:  21%|██▏       | 62/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.self_attn.k_proj.weight]
Loading weights:  22%|██▏       | 63/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.self_attn.o_proj.weight]
Loading weights:  22%|██▏       | 63/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.self_attn.o_proj.weight]
Loading weights:  22%|██▏       | 64/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.self_attn.q_proj.weight]
Loading weights:  22%|██▏       | 64/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.self_attn.q_proj.weight]
Loading weights:  22%|██▏       | 65/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.self_attn.v_proj.weight]
Loading weights:  22%|██▏       | 65/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.6.self_attn.v_proj.weight]
Loading weights:  23%|██▎       | 66/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.7.input_layernorm.weight] 
Loading weights:  23%|██▎       | 66/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.7.input_layernorm.weight]
Loading weights:  23%|██▎       | 67/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.7.mlp.down_proj.weight]  
Loading weights:  23%|██▎       | 67/291 [00:01<00:03, 60.30it/s, Materializing param=model.layers.7.mlp.down_proj.weight]
Loading weights:  23%|██▎       | 68/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.mlp.down_proj.weight]
Loading weights:  23%|██▎       | 68/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.mlp.gate_proj.weight]
Loading weights:  23%|██▎       | 68/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.mlp.gate_proj.weight]
Loading weights:  24%|██▎       | 69/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.mlp.up_proj.weight]  
Loading weights:  24%|██▎       | 69/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.mlp.up_proj.weight]
Loading weights:  24%|██▍       | 70/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.post_attention_layernorm.weight]
Loading weights:  24%|██▍       | 70/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.post_attention_layernorm.weight]
Loading weights:  24%|██▍       | 71/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.self_attn.k_proj.weight]        
Loading weights:  24%|██▍       | 71/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.self_attn.k_proj.weight]
Loading weights:  25%|██▍       | 72/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.self_attn.o_proj.weight]
Loading weights:  25%|██▍       | 72/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.self_attn.o_proj.weight]
Loading weights:  25%|██▌       | 73/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.self_attn.q_proj.weight]
Loading weights:  25%|██▌       | 73/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.self_attn.q_proj.weight]
Loading weights:  25%|██▌       | 74/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.self_attn.v_proj.weight]
Loading weights:  25%|██▌       | 74/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.7.self_attn.v_proj.weight]
Loading weights:  26%|██▌       | 75/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.8.input_layernorm.weight] 
Loading weights:  26%|██▌       | 75/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.8.input_layernorm.weight]
Loading weights:  26%|██▌       | 76/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.8.mlp.down_proj.weight]  
Loading weights:  26%|██▌       | 76/291 [00:01<00:03, 58.77it/s, Materializing param=model.layers.8.mlp.down_proj.weight]
Loading weights:  26%|██▋       | 77/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.mlp.down_proj.weight]
Loading weights:  26%|██▋       | 77/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.mlp.gate_proj.weight]
Loading weights:  26%|██▋       | 77/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.mlp.gate_proj.weight]
Loading weights:  27%|██▋       | 78/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.mlp.up_proj.weight]  
Loading weights:  27%|██▋       | 78/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.mlp.up_proj.weight]
Loading weights:  27%|██▋       | 79/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.post_attention_layernorm.weight]
Loading weights:  27%|██▋       | 79/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.post_attention_layernorm.weight]
Loading weights:  27%|██▋       | 80/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.self_attn.k_proj.weight]        
Loading weights:  27%|██▋       | 80/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.self_attn.k_proj.weight]
Loading weights:  28%|██▊       | 81/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.self_attn.o_proj.weight]
Loading weights:  28%|██▊       | 81/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.self_attn.o_proj.weight]
Loading weights:  28%|██▊       | 82/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.self_attn.q_proj.weight]
Loading weights:  28%|██▊       | 82/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.self_attn.q_proj.weight]
Loading weights:  29%|██▊       | 83/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.self_attn.v_proj.weight]
Loading weights:  29%|██▊       | 83/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.8.self_attn.v_proj.weight]
Loading weights:  29%|██▉       | 84/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.9.input_layernorm.weight] 
Loading weights:  29%|██▉       | 84/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.9.input_layernorm.weight]
Loading weights:  29%|██▉       | 85/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.9.mlp.down_proj.weight]  
Loading weights:  29%|██▉       | 85/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.9.mlp.down_proj.weight]
Loading weights:  30%|██▉       | 86/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.9.mlp.gate_proj.weight]
Loading weights:  30%|██▉       | 86/291 [00:01<00:03, 56.88it/s, Materializing param=model.layers.9.mlp.gate_proj.weight]
Loading weights:  30%|██▉       | 87/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.mlp.gate_proj.weight]
Loading weights:  30%|██▉       | 87/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.mlp.up_proj.weight]  
Loading weights:  30%|██▉       | 87/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.mlp.up_proj.weight]
Loading weights:  30%|███       | 88/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.post_attention_layernorm.weight]
Loading weights:  30%|███       | 88/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.post_attention_layernorm.weight]
Loading weights:  31%|███       | 89/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.self_attn.k_proj.weight]        
Loading weights:  31%|███       | 89/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.self_attn.k_proj.weight]
Loading weights:  31%|███       | 90/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.self_attn.o_proj.weight]
Loading weights:  31%|███       | 90/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.self_attn.o_proj.weight]
Loading weights:  31%|███▏      | 91/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.self_attn.q_proj.weight]
Loading weights:  31%|███▏      | 91/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.self_attn.q_proj.weight]
Loading weights:  32%|███▏      | 92/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.self_attn.v_proj.weight]
Loading weights:  32%|███▏      | 92/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.9.self_attn.v_proj.weight]
Loading weights:  32%|███▏      | 93/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.10.input_layernorm.weight]
Loading weights:  32%|███▏      | 93/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.10.input_layernorm.weight]
Loading weights:  32%|███▏      | 94/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.10.mlp.down_proj.weight]  
Loading weights:  32%|███▏      | 94/291 [00:01<00:03, 57.05it/s, Materializing param=model.layers.10.mlp.down_proj.weight]
Loading weights:  33%|███▎      | 95/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.mlp.down_proj.weight]
Loading weights:  33%|███▎      | 95/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.mlp.gate_proj.weight]
Loading weights:  33%|███▎      | 95/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.mlp.gate_proj.weight]
Loading weights:  33%|███▎      | 96/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.mlp.up_proj.weight]  
Loading weights:  33%|███▎      | 96/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.mlp.up_proj.weight]
Loading weights:  33%|███▎      | 97/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.post_attention_layernorm.weight]
Loading weights:  33%|███▎      | 97/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.post_attention_layernorm.weight]
Loading weights:  34%|███▎      | 98/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.self_attn.k_proj.weight]        
Loading weights:  34%|███▎      | 98/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.self_attn.k_proj.weight]
Loading weights:  34%|███▍      | 99/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.self_attn.o_proj.weight]
Loading weights:  34%|███▍      | 99/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.self_attn.o_proj.weight]
Loading weights:  34%|███▍      | 100/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.self_attn.q_proj.weight]
Loading weights:  34%|███▍      | 100/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.self_attn.q_proj.weight]
Loading weights:  35%|███▍      | 101/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.self_attn.v_proj.weight]
Loading weights:  35%|███▍      | 101/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.10.self_attn.v_proj.weight]
Loading weights:  35%|███▌      | 102/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.11.input_layernorm.weight] 
Loading weights:  35%|███▌      | 102/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.11.input_layernorm.weight]
Loading weights:  35%|███▌      | 103/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.11.mlp.down_proj.weight]  
Loading weights:  35%|███▌      | 103/291 [00:01<00:03, 56.93it/s, Materializing param=model.layers.11.mlp.down_proj.weight]
Loading weights:  36%|███▌      | 104/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.mlp.down_proj.weight]
Loading weights:  36%|███▌      | 104/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.mlp.gate_proj.weight]
Loading weights:  36%|███▌      | 104/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.mlp.gate_proj.weight]
Loading weights:  36%|███▌      | 105/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.mlp.up_proj.weight]  
Loading weights:  36%|███▌      | 105/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.mlp.up_proj.weight]
Loading weights:  36%|███▋      | 106/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.post_attention_layernorm.weight]
Loading weights:  36%|███▋      | 106/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.post_attention_layernorm.weight]
Loading weights:  37%|███▋      | 107/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.self_attn.k_proj.weight]        
Loading weights:  37%|███▋      | 107/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.self_attn.k_proj.weight]
Loading weights:  37%|███▋      | 108/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.self_attn.o_proj.weight]
Loading weights:  37%|███▋      | 108/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.self_attn.o_proj.weight]
Loading weights:  37%|███▋      | 109/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.self_attn.q_proj.weight]
Loading weights:  37%|███▋      | 109/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.self_attn.q_proj.weight]
Loading weights:  38%|███▊      | 110/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.self_attn.v_proj.weight]
Loading weights:  38%|███▊      | 110/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.11.self_attn.v_proj.weight]
Loading weights:  38%|███▊      | 111/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.12.input_layernorm.weight] 
Loading weights:  38%|███▊      | 111/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.12.input_layernorm.weight]
Loading weights:  38%|███▊      | 112/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.12.mlp.down_proj.weight]  
Loading weights:  38%|███▊      | 112/291 [00:01<00:03, 59.14it/s, Materializing param=model.layers.12.mlp.down_proj.weight]
Loading weights:  39%|███▉      | 113/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.mlp.down_proj.weight]
Loading weights:  39%|███▉      | 113/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.mlp.gate_proj.weight]
Loading weights:  39%|███▉      | 113/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.mlp.gate_proj.weight]
Loading weights:  39%|███▉      | 114/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.mlp.up_proj.weight]  
Loading weights:  39%|███▉      | 114/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.mlp.up_proj.weight]
Loading weights:  40%|███▉      | 115/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.post_attention_layernorm.weight]
Loading weights:  40%|███▉      | 115/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.post_attention_layernorm.weight]
Loading weights:  40%|███▉      | 116/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.self_attn.k_proj.weight]        
Loading weights:  40%|███▉      | 116/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.self_attn.k_proj.weight]
Loading weights:  40%|████      | 117/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.self_attn.o_proj.weight]
Loading weights:  40%|████      | 117/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.self_attn.o_proj.weight]
Loading weights:  41%|████      | 118/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.self_attn.q_proj.weight]
Loading weights:  41%|████      | 118/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.self_attn.q_proj.weight]
Loading weights:  41%|████      | 119/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.self_attn.v_proj.weight]
Loading weights:  41%|████      | 119/291 [00:02<00:03, 57.16it/s, Materializing param=model.layers.12.self_attn.v_proj.weight]
Loading weights:  41%|████      | 120/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.13.input_layernorm.weight] 
Loading weights:  41%|████      | 120/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.13.input_layernorm.weight]
Loading weights:  42%|████▏     | 121/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.13.mlp.down_proj.weight]  
Loading weights:  42%|████▏     | 121/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.13.mlp.down_proj.weight]
Loading weights:  42%|████▏     | 122/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.mlp.down_proj.weight]
Loading weights:  42%|████▏     | 122/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.mlp.gate_proj.weight]
Loading weights:  42%|████▏     | 122/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.mlp.gate_proj.weight]
Loading weights:  42%|████▏     | 123/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.mlp.up_proj.weight]  
Loading weights:  42%|████▏     | 123/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.mlp.up_proj.weight]
Loading weights:  43%|████▎     | 124/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.post_attention_layernorm.weight]
Loading weights:  43%|████▎     | 124/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.post_attention_layernorm.weight]
Loading weights:  43%|████▎     | 125/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.self_attn.k_proj.weight]        
Loading weights:  43%|████▎     | 125/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.self_attn.k_proj.weight]
Loading weights:  43%|████▎     | 126/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.self_attn.o_proj.weight]
Loading weights:  43%|████▎     | 126/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.self_attn.o_proj.weight]
Loading weights:  44%|████▎     | 127/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.self_attn.q_proj.weight]
Loading weights:  44%|████▎     | 127/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.self_attn.q_proj.weight]
Loading weights:  44%|████▍     | 128/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.self_attn.v_proj.weight]
Loading weights:  44%|████▍     | 128/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.13.self_attn.v_proj.weight]
Loading weights:  44%|████▍     | 129/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.14.input_layernorm.weight] 
Loading weights:  44%|████▍     | 129/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.14.input_layernorm.weight]
Loading weights:  45%|████▍     | 130/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.14.mlp.down_proj.weight]  
Loading weights:  45%|████▍     | 130/291 [00:02<00:02, 64.12it/s, Materializing param=model.layers.14.mlp.down_proj.weight]
Loading weights:  45%|████▌     | 131/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.mlp.down_proj.weight]
Loading weights:  45%|████▌     | 131/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.mlp.gate_proj.weight]
Loading weights:  45%|████▌     | 131/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.mlp.gate_proj.weight]
Loading weights:  45%|████▌     | 132/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.mlp.up_proj.weight]  
Loading weights:  45%|████▌     | 132/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.mlp.up_proj.weight]
Loading weights:  46%|████▌     | 133/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.post_attention_layernorm.weight]
Loading weights:  46%|████▌     | 133/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.post_attention_layernorm.weight]
Loading weights:  46%|████▌     | 134/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.self_attn.k_proj.weight]        
Loading weights:  46%|████▌     | 134/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.self_attn.k_proj.weight]
Loading weights:  46%|████▋     | 135/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.self_attn.o_proj.weight]
Loading weights:  46%|████▋     | 135/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.self_attn.o_proj.weight]
Loading weights:  47%|████▋     | 136/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.self_attn.q_proj.weight]
Loading weights:  47%|████▋     | 136/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.self_attn.q_proj.weight]
Loading weights:  47%|████▋     | 137/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.self_attn.v_proj.weight]
Loading weights:  47%|████▋     | 137/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.14.self_attn.v_proj.weight]
Loading weights:  47%|████▋     | 138/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.15.input_layernorm.weight] 
Loading weights:  47%|████▋     | 138/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.15.input_layernorm.weight]
Loading weights:  48%|████▊     | 139/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.15.mlp.down_proj.weight]  
Loading weights:  48%|████▊     | 139/291 [00:02<00:02, 56.65it/s, Materializing param=model.layers.15.mlp.down_proj.weight]
Loading weights:  48%|████▊     | 140/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.mlp.down_proj.weight]
Loading weights:  48%|████▊     | 140/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.mlp.gate_proj.weight]
Loading weights:  48%|████▊     | 140/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.mlp.gate_proj.weight]
Loading weights:  48%|████▊     | 141/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.mlp.up_proj.weight]  
Loading weights:  48%|████▊     | 141/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.mlp.up_proj.weight]
Loading weights:  49%|████▉     | 142/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.post_attention_layernorm.weight]
Loading weights:  49%|████▉     | 142/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.post_attention_layernorm.weight]
Loading weights:  49%|████▉     | 143/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.self_attn.k_proj.weight]        
Loading weights:  49%|████▉     | 143/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.self_attn.k_proj.weight]
Loading weights:  49%|████▉     | 144/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.self_attn.o_proj.weight]
Loading weights:  49%|████▉     | 144/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.self_attn.o_proj.weight]
Loading weights:  50%|████▉     | 145/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.self_attn.q_proj.weight]
Loading weights:  50%|████▉     | 145/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.self_attn.q_proj.weight]
Loading weights:  50%|█████     | 146/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.self_attn.v_proj.weight]
Loading weights:  50%|█████     | 146/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.15.self_attn.v_proj.weight]
Loading weights:  51%|█████     | 147/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.16.input_layernorm.weight] 
Loading weights:  51%|█████     | 147/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.16.input_layernorm.weight]
Loading weights:  51%|█████     | 148/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.16.mlp.down_proj.weight]  
Loading weights:  51%|█████     | 148/291 [00:02<00:02, 57.59it/s, Materializing param=model.layers.16.mlp.down_proj.weight]
Loading weights:  51%|█████     | 149/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.mlp.down_proj.weight]
Loading weights:  51%|█████     | 149/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.mlp.gate_proj.weight]
Loading weights:  51%|█████     | 149/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.mlp.gate_proj.weight]
Loading weights:  52%|█████▏    | 150/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.mlp.up_proj.weight]  
Loading weights:  52%|█████▏    | 150/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.mlp.up_proj.weight]
Loading weights:  52%|█████▏    | 151/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.post_attention_layernorm.weight]
Loading weights:  52%|█████▏    | 151/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.post_attention_layernorm.weight]
Loading weights:  52%|█████▏    | 152/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.self_attn.k_proj.weight]        
Loading weights:  52%|█████▏    | 152/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.self_attn.k_proj.weight]
Loading weights:  53%|█████▎    | 153/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.self_attn.o_proj.weight]
Loading weights:  53%|█████▎    | 153/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.self_attn.o_proj.weight]
Loading weights:  53%|█████▎    | 154/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.self_attn.q_proj.weight]
Loading weights:  53%|█████▎    | 154/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.self_attn.q_proj.weight]
Loading weights:  53%|█████▎    | 155/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.self_attn.v_proj.weight]
Loading weights:  53%|█████▎    | 155/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.16.self_attn.v_proj.weight]
Loading weights:  54%|█████▎    | 156/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.17.input_layernorm.weight] 
Loading weights:  54%|█████▎    | 156/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.17.input_layernorm.weight]
Loading weights:  54%|█████▍    | 157/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.17.mlp.down_proj.weight]  
Loading weights:  54%|█████▍    | 157/291 [00:02<00:02, 58.81it/s, Materializing param=model.layers.17.mlp.down_proj.weight]
Loading weights:  54%|█████▍    | 158/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.mlp.down_proj.weight]
Loading weights:  54%|█████▍    | 158/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.mlp.gate_proj.weight]
Loading weights:  54%|█████▍    | 158/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.mlp.gate_proj.weight]
Loading weights:  55%|█████▍    | 159/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.mlp.up_proj.weight]  
Loading weights:  55%|█████▍    | 159/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.mlp.up_proj.weight]
Loading weights:  55%|█████▍    | 160/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.post_attention_layernorm.weight]
Loading weights:  55%|█████▍    | 160/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.post_attention_layernorm.weight]
Loading weights:  55%|█████▌    | 161/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.self_attn.k_proj.weight]        
Loading weights:  55%|█████▌    | 161/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.self_attn.k_proj.weight]
Loading weights:  56%|█████▌    | 162/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.self_attn.o_proj.weight]
Loading weights:  56%|█████▌    | 162/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.self_attn.o_proj.weight]
Loading weights:  56%|█████▌    | 163/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.self_attn.q_proj.weight]
Loading weights:  56%|█████▌    | 163/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.self_attn.q_proj.weight]
Loading weights:  56%|█████▋    | 164/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.self_attn.v_proj.weight]
Loading weights:  56%|█████▋    | 164/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.17.self_attn.v_proj.weight]
Loading weights:  57%|█████▋    | 165/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.18.input_layernorm.weight] 
Loading weights:  57%|█████▋    | 165/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.18.input_layernorm.weight]
Loading weights:  57%|█████▋    | 166/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.18.mlp.down_proj.weight]  
Loading weights:  57%|█████▋    | 166/291 [00:02<00:02, 60.05it/s, Materializing param=model.layers.18.mlp.down_proj.weight]
Loading weights:  57%|█████▋    | 167/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.mlp.down_proj.weight]
Loading weights:  57%|█████▋    | 167/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.mlp.gate_proj.weight]
Loading weights:  57%|█████▋    | 167/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.mlp.gate_proj.weight]
Loading weights:  58%|█████▊    | 168/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.mlp.up_proj.weight]  
Loading weights:  58%|█████▊    | 168/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.mlp.up_proj.weight]
Loading weights:  58%|█████▊    | 169/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.post_attention_layernorm.weight]
Loading weights:  58%|█████▊    | 169/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.post_attention_layernorm.weight]
Loading weights:  58%|█████▊    | 170/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.self_attn.k_proj.weight]        
Loading weights:  58%|█████▊    | 170/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.self_attn.k_proj.weight]
Loading weights:  59%|█████▉    | 171/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.self_attn.o_proj.weight]
Loading weights:  59%|█████▉    | 171/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.self_attn.o_proj.weight]
Loading weights:  59%|█████▉    | 172/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.self_attn.q_proj.weight]
Loading weights:  59%|█████▉    | 172/291 [00:02<00:02, 57.16it/s, Materializing param=model.layers.18.self_attn.q_proj.weight]
Loading weights:  59%|█████▉    | 173/291 [00:03<00:02, 57.16it/s, Materializing param=model.layers.18.self_attn.v_proj.weight]
Loading weights:  59%|█████▉    | 173/291 [00:03<00:02, 57.16it/s, Materializing param=model.layers.18.self_attn.v_proj.weight]
Loading weights:  60%|█████▉    | 174/291 [00:03<00:02, 57.16it/s, Materializing param=model.layers.19.input_layernorm.weight] 
Loading weights:  60%|█████▉    | 174/291 [00:03<00:02, 57.16it/s, Materializing param=model.layers.19.input_layernorm.weight]
Loading weights:  60%|██████    | 175/291 [00:03<00:02, 57.16it/s, Materializing param=model.layers.19.mlp.down_proj.weight]  
Loading weights:  60%|██████    | 175/291 [00:03<00:02, 57.16it/s, Materializing param=model.layers.19.mlp.down_proj.weight]
Loading weights:  60%|██████    | 176/291 [00:03<00:02, 57.16it/s, Materializing param=model.layers.19.mlp.gate_proj.weight]
Loading weights:  60%|██████    | 176/291 [00:03<00:02, 57.16it/s, Materializing param=model.layers.19.mlp.gate_proj.weight]
Loading weights:  61%|██████    | 177/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.mlp.gate_proj.weight]
Loading weights:  61%|██████    | 177/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.mlp.up_proj.weight]  
Loading weights:  61%|██████    | 177/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.mlp.up_proj.weight]
Loading weights:  61%|██████    | 178/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.post_attention_layernorm.weight]
Loading weights:  61%|██████    | 178/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.post_attention_layernorm.weight]
Loading weights:  62%|██████▏   | 179/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.self_attn.k_proj.weight]        
Loading weights:  62%|██████▏   | 179/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.self_attn.k_proj.weight]
Loading weights:  62%|██████▏   | 180/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.self_attn.o_proj.weight]
Loading weights:  62%|██████▏   | 180/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.self_attn.o_proj.weight]
Loading weights:  62%|██████▏   | 181/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.self_attn.q_proj.weight]
Loading weights:  62%|██████▏   | 181/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.self_attn.q_proj.weight]
Loading weights:  63%|██████▎   | 182/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.self_attn.v_proj.weight]
Loading weights:  63%|██████▎   | 182/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.19.self_attn.v_proj.weight]
Loading weights:  63%|██████▎   | 183/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.20.input_layernorm.weight] 
Loading weights:  63%|██████▎   | 183/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.20.input_layernorm.weight]
Loading weights:  63%|██████▎   | 184/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.20.mlp.down_proj.weight]  
Loading weights:  63%|██████▎   | 184/291 [00:03<00:01, 62.61it/s, Materializing param=model.layers.20.mlp.down_proj.weight]
Loading weights:  64%|██████▎   | 185/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.mlp.down_proj.weight]
Loading weights:  64%|██████▎   | 185/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.mlp.gate_proj.weight]
Loading weights:  64%|██████▎   | 185/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.mlp.gate_proj.weight]
Loading weights:  64%|██████▍   | 186/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.mlp.up_proj.weight]  
Loading weights:  64%|██████▍   | 186/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.mlp.up_proj.weight]
Loading weights:  64%|██████▍   | 187/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.post_attention_layernorm.weight]
Loading weights:  64%|██████▍   | 187/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.post_attention_layernorm.weight]
Loading weights:  65%|██████▍   | 188/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.self_attn.k_proj.weight]        
Loading weights:  65%|██████▍   | 188/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.self_attn.k_proj.weight]
Loading weights:  65%|██████▍   | 189/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.self_attn.o_proj.weight]
Loading weights:  65%|██████▍   | 189/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.self_attn.o_proj.weight]
Loading weights:  65%|██████▌   | 190/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.self_attn.q_proj.weight]
Loading weights:  65%|██████▌   | 190/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.self_attn.q_proj.weight]
Loading weights:  66%|██████▌   | 191/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.self_attn.v_proj.weight]
Loading weights:  66%|██████▌   | 191/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.20.self_attn.v_proj.weight]
Loading weights:  66%|██████▌   | 192/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.21.input_layernorm.weight] 
Loading weights:  66%|██████▌   | 192/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.21.input_layernorm.weight]
Loading weights:  66%|██████▋   | 193/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.21.mlp.down_proj.weight]  
Loading weights:  66%|██████▋   | 193/291 [00:03<00:01, 56.80it/s, Materializing param=model.layers.21.mlp.down_proj.weight]
Loading weights:  67%|██████▋   | 194/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.mlp.down_proj.weight]
Loading weights:  67%|██████▋   | 194/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.mlp.gate_proj.weight]
Loading weights:  67%|██████▋   | 194/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.mlp.gate_proj.weight]
Loading weights:  67%|██████▋   | 195/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.mlp.up_proj.weight]  
Loading weights:  67%|██████▋   | 195/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.mlp.up_proj.weight]
Loading weights:  67%|██████▋   | 196/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.post_attention_layernorm.weight]
Loading weights:  67%|██████▋   | 196/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.post_attention_layernorm.weight]
Loading weights:  68%|██████▊   | 197/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.self_attn.k_proj.weight]        
Loading weights:  68%|██████▊   | 197/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.self_attn.k_proj.weight]
Loading weights:  68%|██████▊   | 198/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.self_attn.o_proj.weight]
Loading weights:  68%|██████▊   | 198/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.self_attn.o_proj.weight]
Loading weights:  68%|██████▊   | 199/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.self_attn.q_proj.weight]
Loading weights:  68%|██████▊   | 199/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.self_attn.q_proj.weight]
Loading weights:  69%|██████▊   | 200/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.self_attn.v_proj.weight]
Loading weights:  69%|██████▊   | 200/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.21.self_attn.v_proj.weight]
Loading weights:  69%|██████▉   | 201/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.22.input_layernorm.weight] 
Loading weights:  69%|██████▉   | 201/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.22.input_layernorm.weight]
Loading weights:  69%|██████▉   | 202/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.22.mlp.down_proj.weight]  
Loading weights:  69%|██████▉   | 202/291 [00:03<00:01, 58.89it/s, Materializing param=model.layers.22.mlp.down_proj.weight]
Loading weights:  70%|██████▉   | 203/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.mlp.down_proj.weight]
Loading weights:  70%|██████▉   | 203/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.mlp.gate_proj.weight]
Loading weights:  70%|██████▉   | 203/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.mlp.gate_proj.weight]
Loading weights:  70%|███████   | 204/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.mlp.up_proj.weight]  
Loading weights:  70%|███████   | 204/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.mlp.up_proj.weight]
Loading weights:  70%|███████   | 205/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.post_attention_layernorm.weight]
Loading weights:  70%|███████   | 205/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.post_attention_layernorm.weight]
Loading weights:  71%|███████   | 206/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.self_attn.k_proj.weight]        
Loading weights:  71%|███████   | 206/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.self_attn.k_proj.weight]
Loading weights:  71%|███████   | 207/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.self_attn.o_proj.weight]
Loading weights:  71%|███████   | 207/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.self_attn.o_proj.weight]
Loading weights:  71%|███████▏  | 208/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.self_attn.q_proj.weight]
Loading weights:  71%|███████▏  | 208/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.self_attn.q_proj.weight]
Loading weights:  72%|███████▏  | 209/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.self_attn.v_proj.weight]
Loading weights:  72%|███████▏  | 209/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.22.self_attn.v_proj.weight]
Loading weights:  72%|███████▏  | 210/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.23.input_layernorm.weight] 
Loading weights:  72%|███████▏  | 210/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.23.input_layernorm.weight]
Loading weights:  73%|███████▎  | 211/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.23.mlp.down_proj.weight]  
Loading weights:  73%|███████▎  | 211/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.23.mlp.down_proj.weight]
Loading weights:  73%|███████▎  | 212/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.23.mlp.gate_proj.weight]
Loading weights:  73%|███████▎  | 212/291 [00:03<00:01, 56.75it/s, Materializing param=model.layers.23.mlp.gate_proj.weight]
Loading weights:  73%|███████▎  | 213/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.mlp.gate_proj.weight]
Loading weights:  73%|███████▎  | 213/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.mlp.up_proj.weight]  
Loading weights:  73%|███████▎  | 213/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.mlp.up_proj.weight]
Loading weights:  74%|███████▎  | 214/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.post_attention_layernorm.weight]
Loading weights:  74%|███████▎  | 214/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.post_attention_layernorm.weight]
Loading weights:  74%|███████▍  | 215/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.self_attn.k_proj.weight]        
Loading weights:  74%|███████▍  | 215/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.self_attn.k_proj.weight]
Loading weights:  74%|███████▍  | 216/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.self_attn.o_proj.weight]
Loading weights:  74%|███████▍  | 216/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.self_attn.o_proj.weight]
Loading weights:  75%|███████▍  | 217/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.self_attn.q_proj.weight]
Loading weights:  75%|███████▍  | 217/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.self_attn.q_proj.weight]
Loading weights:  75%|███████▍  | 218/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.self_attn.v_proj.weight]
Loading weights:  75%|███████▍  | 218/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.23.self_attn.v_proj.weight]
Loading weights:  75%|███████▌  | 219/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.24.input_layernorm.weight] 
Loading weights:  75%|███████▌  | 219/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.24.input_layernorm.weight]
Loading weights:  76%|███████▌  | 220/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.24.mlp.down_proj.weight]  
Loading weights:  76%|███████▌  | 220/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.24.mlp.down_proj.weight]
Loading weights:  76%|███████▌  | 221/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.24.mlp.gate_proj.weight]
Loading weights:  76%|███████▌  | 221/291 [00:03<00:01, 62.72it/s, Materializing param=model.layers.24.mlp.gate_proj.weight]
Loading weights:  76%|███████▋  | 222/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.mlp.gate_proj.weight]
Loading weights:  76%|███████▋  | 222/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.mlp.up_proj.weight]  
Loading weights:  76%|███████▋  | 222/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.mlp.up_proj.weight]
Loading weights:  77%|███████▋  | 223/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.post_attention_layernorm.weight]
Loading weights:  77%|███████▋  | 223/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.post_attention_layernorm.weight]
Loading weights:  77%|███████▋  | 224/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.self_attn.k_proj.weight]        
Loading weights:  77%|███████▋  | 224/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.self_attn.k_proj.weight]
Loading weights:  77%|███████▋  | 225/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.self_attn.o_proj.weight]
Loading weights:  77%|███████▋  | 225/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.self_attn.o_proj.weight]
Loading weights:  78%|███████▊  | 226/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.self_attn.q_proj.weight]
Loading weights:  78%|███████▊  | 226/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.self_attn.q_proj.weight]
Loading weights:  78%|███████▊  | 227/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.self_attn.v_proj.weight]
Loading weights:  78%|███████▊  | 227/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.24.self_attn.v_proj.weight]
Loading weights:  78%|███████▊  | 228/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.25.input_layernorm.weight] 
Loading weights:  78%|███████▊  | 228/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.25.input_layernorm.weight]
Loading weights:  79%|███████▊  | 229/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.25.mlp.down_proj.weight]  
Loading weights:  79%|███████▊  | 229/291 [00:03<00:01, 59.10it/s, Materializing param=model.layers.25.mlp.down_proj.weight]
Loading weights:  79%|███████▉  | 230/291 [00:04<00:01, 56.00it/s, Materializing param=model.layers.25.mlp.down_proj.weight]
Loading weights:  79%|███████▉  | 230/291 [00:04<00:01, 56.00it/s, Materializing param=model.layers.25.mlp.gate_proj.weight]
Loading weights:  79%|███████▉  | 230/291 [00:04<00:01, 56.00it/s, Materializing param=model.layers.25.mlp.gate_proj.weight]
Loading weights:  79%|███████▉  | 231/291 [00:04<00:01, 56.00it/s, Materializing param=model.layers.25.mlp.up_proj.weight]  
Loading weights:  79%|███████▉  | 231/291 [00:04<00:01, 56.00it/s, Materializing param=model.layers.25.mlp.up_proj.weight]
Loading weights:  80%|███████▉  | 232/291 [00:04<00:01, 56.00it/s, Materializing param=model.layers.25.post_attention_layernorm.weight]
Loading weights:  80%|███████▉  | 232/291 [00:04<00:01, 56.00it/s, Materializing param=model.layers.25.post_attention_layernorm.weight]
Loading weights:  80%|████████  | 233/291 [00:04<00:01, 56.00it/s, Materializing param=model.layers.25.self_attn.k_proj.weight]        
Loading weights:  80%|████████  | 233/291 [00:04<00:01, 56.00it/s, Materializing param=model.layers.25.self_attn.k_proj.weight]
Loading weights:  80%|████████  | 234/291 [00:04<00:01, 56.00it/s, Materializing param=model.layers.25.self_attn.o_proj.weight]
Loading weights:  80%|████████  | 234/291 [00:04<00:01, 56.00it/s, Materializing param=model.layers.25.self_attn.o_proj.weight]
Loading weights:  81%|████████  | 235/291 [00:04<00:00, 56.00it/s, Materializing param=model.layers.25.self_attn.q_proj.weight]
Loading weights:  81%|████████  | 235/291 [00:04<00:00, 56.00it/s, Materializing param=model.layers.25.self_attn.q_proj.weight]
Loading weights:  81%|████████  | 236/291 [00:04<00:00, 56.00it/s, Materializing param=model.layers.25.self_attn.v_proj.weight]
Loading weights:  81%|████████  | 236/291 [00:04<00:00, 56.00it/s, Materializing param=model.layers.25.self_attn.v_proj.weight]
Loading weights:  81%|████████▏ | 237/291 [00:04<00:00, 56.00it/s, Materializing param=model.layers.26.input_layernorm.weight] 
Loading weights:  81%|████████▏ | 237/291 [00:04<00:00, 56.00it/s, Materializing param=model.layers.26.input_layernorm.weight]
Loading weights:  82%|████████▏ | 238/291 [00:04<00:00, 56.00it/s, Materializing param=model.layers.26.mlp.down_proj.weight]  
Loading weights:  82%|████████▏ | 238/291 [00:04<00:00, 56.00it/s, Materializing param=model.layers.26.mlp.down_proj.weight]
Loading weights:  82%|████████▏ | 239/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.mlp.down_proj.weight]
Loading weights:  82%|████████▏ | 239/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.mlp.gate_proj.weight]
Loading weights:  82%|████████▏ | 239/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.mlp.gate_proj.weight]
Loading weights:  82%|████████▏ | 240/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.mlp.up_proj.weight]  
Loading weights:  82%|████████▏ | 240/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.mlp.up_proj.weight]
Loading weights:  83%|████████▎ | 241/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.post_attention_layernorm.weight]
Loading weights:  83%|████████▎ | 241/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.post_attention_layernorm.weight]
Loading weights:  83%|████████▎ | 242/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.self_attn.k_proj.weight]        
Loading weights:  83%|████████▎ | 242/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.self_attn.k_proj.weight]
Loading weights:  84%|████████▎ | 243/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.self_attn.o_proj.weight]
Loading weights:  84%|████████▎ | 243/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.self_attn.o_proj.weight]
Loading weights:  84%|████████▍ | 244/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.self_attn.q_proj.weight]
Loading weights:  84%|████████▍ | 244/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.self_attn.q_proj.weight]
Loading weights:  84%|████████▍ | 245/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.self_attn.v_proj.weight]
Loading weights:  84%|████████▍ | 245/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.26.self_attn.v_proj.weight]
Loading weights:  85%|████████▍ | 246/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.27.input_layernorm.weight] 
Loading weights:  85%|████████▍ | 246/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.27.input_layernorm.weight]
Loading weights:  85%|████████▍ | 247/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.27.mlp.down_proj.weight]  
Loading weights:  85%|████████▍ | 247/291 [00:04<00:00, 57.41it/s, Materializing param=model.layers.27.mlp.down_proj.weight]
Loading weights:  85%|████████▌ | 248/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.mlp.down_proj.weight]
Loading weights:  85%|████████▌ | 248/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.mlp.gate_proj.weight]
Loading weights:  85%|████████▌ | 248/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.mlp.gate_proj.weight]
Loading weights:  86%|████████▌ | 249/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.mlp.up_proj.weight]  
Loading weights:  86%|████████▌ | 249/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.mlp.up_proj.weight]
Loading weights:  86%|████████▌ | 250/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.post_attention_layernorm.weight]
Loading weights:  86%|████████▌ | 250/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.post_attention_layernorm.weight]
Loading weights:  86%|████████▋ | 251/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.self_attn.k_proj.weight]        
Loading weights:  86%|████████▋ | 251/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.self_attn.k_proj.weight]
Loading weights:  87%|████████▋ | 252/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.self_attn.o_proj.weight]
Loading weights:  87%|████████▋ | 252/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.self_attn.o_proj.weight]
Loading weights:  87%|████████▋ | 253/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.self_attn.q_proj.weight]
Loading weights:  87%|████████▋ | 253/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.self_attn.q_proj.weight]
Loading weights:  87%|████████▋ | 254/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.self_attn.v_proj.weight]
Loading weights:  87%|████████▋ | 254/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.27.self_attn.v_proj.weight]
Loading weights:  88%|████████▊ | 255/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.28.input_layernorm.weight] 
Loading weights:  88%|████████▊ | 255/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.28.input_layernorm.weight]
Loading weights:  88%|████████▊ | 256/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.28.mlp.down_proj.weight]  
Loading weights:  88%|████████▊ | 256/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.28.mlp.down_proj.weight]
Loading weights:  88%|████████▊ | 257/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.28.mlp.gate_proj.weight]
Loading weights:  88%|████████▊ | 257/291 [00:04<00:00, 59.06it/s, Materializing param=model.layers.28.mlp.gate_proj.weight]
Loading weights:  89%|████████▊ | 258/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.mlp.gate_proj.weight]
Loading weights:  89%|████████▊ | 258/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.mlp.up_proj.weight]  
Loading weights:  89%|████████▊ | 258/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.mlp.up_proj.weight]
Loading weights:  89%|████████▉ | 259/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.post_attention_layernorm.weight]
Loading weights:  89%|████████▉ | 259/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.post_attention_layernorm.weight]
Loading weights:  89%|████████▉ | 260/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.self_attn.k_proj.weight]        
Loading weights:  89%|████████▉ | 260/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.self_attn.k_proj.weight]
Loading weights:  90%|████████▉ | 261/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.self_attn.o_proj.weight]
Loading weights:  90%|████████▉ | 261/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.self_attn.o_proj.weight]
Loading weights:  90%|█████████ | 262/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.self_attn.q_proj.weight]
Loading weights:  90%|█████████ | 262/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.self_attn.q_proj.weight]
Loading weights:  90%|█████████ | 263/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.self_attn.v_proj.weight]
Loading weights:  90%|█████████ | 263/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.28.self_attn.v_proj.weight]
Loading weights:  91%|█████████ | 264/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.29.input_layernorm.weight] 
Loading weights:  91%|█████████ | 264/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.29.input_layernorm.weight]
Loading weights:  91%|█████████ | 265/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.29.mlp.down_proj.weight]  
Loading weights:  91%|█████████ | 265/291 [00:04<00:00, 59.28it/s, Materializing param=model.layers.29.mlp.down_proj.weight]
Loading weights:  91%|█████████▏| 266/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.mlp.down_proj.weight]
Loading weights:  91%|█████████▏| 266/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.mlp.gate_proj.weight]
Loading weights:  91%|█████████▏| 266/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.mlp.gate_proj.weight]
Loading weights:  92%|█████████▏| 267/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.mlp.up_proj.weight]  
Loading weights:  92%|█████████▏| 267/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.mlp.up_proj.weight]
Loading weights:  92%|█████████▏| 268/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.post_attention_layernorm.weight]
Loading weights:  92%|█████████▏| 268/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.post_attention_layernorm.weight]
Loading weights:  92%|█████████▏| 269/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.self_attn.k_proj.weight]        
Loading weights:  92%|█████████▏| 269/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.self_attn.k_proj.weight]
Loading weights:  93%|█████████▎| 270/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.self_attn.o_proj.weight]
Loading weights:  93%|█████████▎| 270/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.self_attn.o_proj.weight]
Loading weights:  93%|█████████▎| 271/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.self_attn.q_proj.weight]
Loading weights:  93%|█████████▎| 271/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.self_attn.q_proj.weight]
Loading weights:  93%|█████████▎| 272/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.self_attn.v_proj.weight]
Loading weights:  93%|█████████▎| 272/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.29.self_attn.v_proj.weight]
Loading weights:  94%|█████████▍| 273/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.30.input_layernorm.weight] 
Loading weights:  94%|█████████▍| 273/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.30.input_layernorm.weight]
Loading weights:  94%|█████████▍| 274/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.30.mlp.down_proj.weight]  
Loading weights:  94%|█████████▍| 274/291 [00:04<00:00, 60.47it/s, Materializing param=model.layers.30.mlp.down_proj.weight]
Loading weights:  95%|█████████▍| 275/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.mlp.down_proj.weight]
Loading weights:  95%|█████████▍| 275/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.mlp.gate_proj.weight]
Loading weights:  95%|█████████▍| 275/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.mlp.gate_proj.weight]
Loading weights:  95%|█████████▍| 276/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.mlp.up_proj.weight]  
Loading weights:  95%|█████████▍| 276/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.mlp.up_proj.weight]
Loading weights:  95%|█████████▌| 277/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.post_attention_layernorm.weight]
Loading weights:  95%|█████████▌| 277/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.post_attention_layernorm.weight]
Loading weights:  96%|█████████▌| 278/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.self_attn.k_proj.weight]        
Loading weights:  96%|█████████▌| 278/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.self_attn.k_proj.weight]
Loading weights:  96%|█████████▌| 279/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.self_attn.o_proj.weight]
Loading weights:  96%|█████████▌| 279/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.self_attn.o_proj.weight]
Loading weights:  96%|█████████▌| 280/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.self_attn.q_proj.weight]
Loading weights:  96%|█████████▌| 280/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.self_attn.q_proj.weight]
Loading weights:  97%|█████████▋| 281/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.self_attn.v_proj.weight]
Loading weights:  97%|█████████▋| 281/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.30.self_attn.v_proj.weight]
Loading weights:  97%|█████████▋| 282/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.31.input_layernorm.weight] 
Loading weights:  97%|█████████▋| 282/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.31.input_layernorm.weight]
Loading weights:  97%|█████████▋| 283/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.31.mlp.down_proj.weight]  
Loading weights:  97%|█████████▋| 283/291 [00:04<00:00, 60.18it/s, Materializing param=model.layers.31.mlp.down_proj.weight]
Loading weights:  98%|█████████▊| 284/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.mlp.down_proj.weight]
Loading weights:  98%|█████████▊| 284/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.mlp.gate_proj.weight]
Loading weights:  98%|█████████▊| 284/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.mlp.gate_proj.weight]
Loading weights:  98%|█████████▊| 285/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.mlp.up_proj.weight]  
Loading weights:  98%|█████████▊| 285/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.mlp.up_proj.weight]
Loading weights:  98%|█████████▊| 286/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.post_attention_layernorm.weight]
Loading weights:  98%|█████████▊| 286/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.post_attention_layernorm.weight]
Loading weights:  99%|█████████▊| 287/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.self_attn.k_proj.weight]        
Loading weights:  99%|█████████▊| 287/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.self_attn.k_proj.weight]
Loading weights:  99%|█████████▉| 288/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.self_attn.o_proj.weight]
Loading weights:  99%|█████████▉| 288/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.self_attn.o_proj.weight]
Loading weights:  99%|█████████▉| 289/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.self_attn.q_proj.weight]
Loading weights:  99%|█████████▉| 289/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.self_attn.q_proj.weight]
Loading weights: 100%|█████████▉| 290/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.self_attn.v_proj.weight]
Loading weights: 100%|█████████▉| 290/291 [00:04<00:00, 59.77it/s, Materializing param=model.layers.31.self_attn.v_proj.weight]
Loading weights: 100%|██████████| 291/291 [00:04<00:00, 59.77it/s, Materializing param=model.norm.weight]                      
Loading weights: 100%|██████████| 291/291 [00:04<00:00, 59.77it/s, Materializing param=model.norm.weight]
Loading weights: 100%|██████████| 291/291 [00:04<00:00, 58.33it/s, Materializing param=model.norm.weight]
INFO:     Started server process [2105782]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
INFO:     89.105.216.247:57252 - "POST /analyze HTTP/1.1" 200 OK
INFO:     23.106.66.175:58496 - "GET /favicon.ico HTTP/1.1" 404 Not Found
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
INFO:     34.14.55.7:33798 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:57432 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34228 - "GET / HTTP/1.1" 404 Not Found
INFO:     195.3.221.86:36444 - "GET /dispatch.asp HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:43582 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:45360 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:55766 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:43368 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:42180 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:43922 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:49462 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:54490 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:40788 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:40326 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:51128 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:36532 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:43610 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:47734 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:49322 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:48280 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:46210 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:57194 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:57688 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:46266 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:38880 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35790 - "GET / HTTP/1.1" 404 Not Found
INFO:     40.76.250.51:35470 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:49836 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:41932 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:59284 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:50382 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:59406 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:38068 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35762 - "GET / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:60132 - "GET / HTTP/1.1" 404 Not Found
INFO:     167.94.138.126:44528 - "GET / HTTP/1.1" 404 Not Found
WARNING:  Invalid HTTP request received.
INFO:     167.94.138.126:20062 - "GET / HTTP/1.1" 404 Not Found
INFO:     167.94.138.126:20068 - "PRI %2A HTTP/2.0" 404 Not Found
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
INFO:     167.94.138.126:19252 - "GET /sitemap.xml HTTP/1.1" 404 Not Found
WARNING:  Invalid HTTP request received.
INFO:     167.94.138.58:17888 - "GET / HTTP/1.1" 404 Not Found
WARNING:  Invalid HTTP request received.
INFO:     167.94.138.58:44528 - "GET / HTTP/1.1" 404 Not Found
INFO:     167.94.138.58:19684 - "PRI %2A HTTP/2.0" 404 Not Found
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
INFO:     167.94.138.58:30638 - "GET /security.txt HTTP/1.1" 404 Not Found
INFO:     71.6.134.235:57220 - "GET / HTTP/1.1" 404 Not Found
INFO:     71.6.134.235:45190 - "GET /favicon.ico HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34494 - "GET //.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34500 - "GET //.env.backup HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34514 - "GET //.env.bak HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34518 - "GET //.env.config HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34528 - "GET //.env.debug HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34540 - "GET //.env.dev HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34554 - "GET //.env.dev.local HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34568 - "GET //.env.development HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34576 - "GET //.env.development.local HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34584 - "GET //.env.dist HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34596 - "GET //.env.docker.dev HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34612 - "GET //.env.example HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34616 - "GET //.env.example.local HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34624 - "GET //.env.live HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34640 - "GET //.env.local HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34650 - "GET //.env.old HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34662 - "GET //.env.override HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34674 - "GET //.env.php HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34676 - "GET //.env.private HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34686 - "GET //.env.prod HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34696 - "GET //.env.prod.local HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34702 - "GET //.env.production HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34712 - "GET //.env.production.local HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34716 - "GET //.env.save HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34722 - "GET //.env.secure HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34736 - "GET //.env.settings HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34738 - "GET //.env.stage HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34742 - "GET //.env.stage.local HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34758 - "GET //.env.staging HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34766 - "GET //.env.template HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34770 - "GET //.env.test HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34776 - "GET //.env_sample HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34784 - "GET //.env~ HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34800 - "GET //?pp=env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34814 - "GET //.aws/config HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34830 - "GET //.aws/credentials HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34836 - "GET //_profiler/phpinfo HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34848 - "GET //.gitlab-ci.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34864 - "GET //.travis.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34876 - "GET //?phpinfo=1 HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34884 - "GET //1board/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34900 - "GET //application/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34916 - "GET //aws.yaml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34926 - "GET //aws.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34936 - "GET //aws_config.csv HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34940 - "GET //aws_config.json HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34944 - "GET //aws_logs.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34946 - "GET //aws_report.csv HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34956 - "GET //aws_secrets.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34960 - "GET //awsConfig.csv HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34966 - "GET //awsconfig.js HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34982 - "GET //awsconfig.json HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34988 - "GET //aws-config.json HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:34996 - "GET //awsConfig.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35012 - "GET //aws-secret.yaml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35028 - "GET //awsSettings.js HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35036 - "GET //cms/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35042 - "GET //credentials.bak HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35046 - "GET //credentials.csv HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35056 - "GET //credentials.js HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35072 - "GET //credentials.json HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35074 - "GET //env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35080 - "GET //env.json HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35092 - "GET //git/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35102 - "GET //gitlab-ci.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35104 - "GET //docker_compose.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35118 - "GET //docker_secrets.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35124 - "GET //docker_usage.csv HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35132 - "GET //docker-compose.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35134 - "GET //env.template HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35148 - "GET //help/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35152 - "GET //helpdesk/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35158 - "GET //infophp.php HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35170 - "GET //php.ini HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35182 - "GET //php.php HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35184 - "GET //php_info.php HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35198 - "GET //php_version.php HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35200 - "GET //phpinfo HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35208 - "GET //phpinfo.php HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35214 - "GET //php-info.php HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35228 - "GET //phpinfo.php4 HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35236 - "GET //phpinfo.php5 HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35242 - "GET //phpinfo/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35254 - "GET //phpinfo/info.php HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35270 - "GET //s3.js HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35282 - "GET //s3_credentials.csv HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35288 - "GET //s3_keys.json HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35290 - "GET //s3config.json HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35302 - "GET //secrets.json HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35308 - "GET //secrets.yaml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35316 - "GET //secrets.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35326 - "GET //server.log HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35342 - "GET //services.yaml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35354 - "GET //ses_config.js HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35366 - "GET //ses_config.json HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35378 - "GET //settings.js HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35380 - "GET //sms.py HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35394 - "GET //staging.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35404 - "GET //symfony/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35416 - "GET //production.local HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35430 - "GET //production.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35446 - "GET //production/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35450 - "GET //servidor.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35464 - "GET //backend/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35478 - "GET //back/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35490 - "GET //bitbucket-pipelines.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35498 - "GET //debug.php HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35512 - "GET //database.yml HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35524 - "GET //old/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35528 - "GET //platform/.env HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35538 - "POST / HTTP/1.1" 404 Not Found
INFO:     178.128.48.59:35552 - "GET / HTTP/1.1" 404 Not Found
INFO:     64.62.156.108:53834 - "GET / HTTP/1.1" 404 Not Found
INFO:     64.62.156.119:5073 - "GET /favicon.ico HTTP/1.1" 404 Not Found
INFO:     64.62.156.108:5825 - "GET http%3A//api.ipify.org/?format=json HTTP/1.1" 404 Not Found
INFO:     64.62.156.113:4419 - "CONNECT www.shadowserver.org%3A443 HTTP/1.1" 404 Not Found
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
INFO:     3.129.187.38:33348 - "GET / HTTP/1.1" 404 Not Found
INFO:     3.129.187.38:43934 - "GET / HTTP/1.1" 404 Not Found
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
INFO:     89.105.216.247:34290 - "POST /analyze HTTP/1.1" 200 OK
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
INFO:     89.105.216.247:41330 - "POST /analyze HTTP/1.1" 200 OK
INFO:     23.106.66.175:62795 - "GET /favicon.ico HTTP/1.1" 404 Not Found
WARNING:  Invalid HTTP request received.
INFO:     91.230.168.179:35851 - "GET / HTTP/1.1" 404 Not Found
INFO:     195.184.76.164:52601 - "GET /favicon.ico HTTP/1.1" 404 Not Found
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
INFO:     89.105.216.247:57408 - "POST /analyze HTTP/1.1" 200 OK
WARNING:  Invalid HTTP request received.
INFO:     167.94.138.196:22958 - "GET / HTTP/1.1" 404 Not Found
INFO:     167.94.138.196:14910 - "PRI %2A HTTP/2.0" 404 Not Found
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
INFO:     167.94.138.196:28598 - "GET /wiki HTTP/1.1" 404 Not Found
INFO:     45.79.102.191:48370 - "GET / HTTP/1.0" 404 Not Found
INFO:     45.79.102.191:48514 - "GET /nice%20ports%2C/Trinity.txt.bak HTTP/1.0" 404 Not Found
WARNING:  Invalid HTTP request received.
INFO:     45.79.102.191:48552 - "OPTIONS / HTTP/1.0" 404 Not Found
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
INFO:     45.79.102.191:43086 - "GET /health HTTP/1.1" 404 Not Found
INFO:     45.79.102.191:43076 - "GET / HTTP/1.1" 404 Not Found
INFO:     45.79.102.191:60076 - "GET / HTTP/1.1" 404 Not Found
INFO:     45.79.102.191:60106 - "GET /dashboard/ HTTP/1.1" 404 Not Found
INFO:     45.79.102.191:60130 - "GET /webui/index.html HTTP/1.1" 404 Not Found
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
INFO:     23.106.66.175:63988 - "GET /favicon.ico HTTP/1.1" 404 Not Found
INFO:     192.241.153.43:42160 - "GET / HTTP/1.1" 404 Not Found
INFO:     192.241.153.43:42172 - "GET /favicon.ico HTTP/1.1" 404 Not Found
INFO:     65.49.1.94:54366 - "GET / HTTP/1.1" 404 Not Found
INFO:     65.49.1.104:16843 - "GET /favicon.ico HTTP/1.1" 404 Not Found
INFO:     65.49.1.103:10779 - "GET http%3A//api.ipify.org/?format=json HTTP/1.1" 404 Not Found
INFO:     65.49.1.95:18095 - "CONNECT www.shadowserver.org%3A443 HTTP/1.1" 404 Not Found
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
INFO:     89.105.216.247:60666 - "POST /analyze HTTP/1.1" 200 OK
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
INFO:     89.105.216.247:44926 - "POST /analyze HTTP/1.1" 200 OK
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
INFO:     89.105.216.247:34734 - "POST /analyze HTTP/1.1" 200 OK

```

---
File: s_main.py
---

```py
import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI

from s_route import router
from t_document_route import router as document_router
from document_generation.document_generator import router as document_generate_router
from s_db import init_db, close_pool

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "formatter": "standard",
            "level":     "INFO",
            "stream":    "ext://sys.stdout",
        },
        "file": {
            "class":     "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level":     "DEBUG",
            "filename":  "app.log",
            "maxBytes":  10 * 1024 * 1024,
            "backupCount": 5,
            "encoding":  "utf-8",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
})

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    await init_db()
    yield
    await close_pool()
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PDF AI Review API",
    description="""
## PDF AI Review + Legal Document Generation API

### PDF Analysis
- **POST /analyze** — Analyse a PDF, return overview, summary, highlights
- **POST /analyze/stream** — Same but streams results via SSE
- **POST /key-clause-extraction** — Extract key clauses by document type
- **POST /detect-risks** — Detect legal/financial risks in a document
- **POST /convert/pdf-to-docx** — Convert PDF to DOCX

### Document Generation
- **GET /documents/types** — List all supported document types
- **POST /documents/extract-fields** — Extract fields from description (no generation)
- **POST /documents/generate** — Generate a complete legal document
- **POST /documents/generate/stream** — Generate with real-time token streaming
""",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(document_router)
app.include_router(document_generate_router)

```

---
File: s_route.py
---

```py
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
import os
import io
import uuid
import json
import time
import asyncio
import logging
from functools import partial

from s_pdf_utils import load_pdf, get_page_count, all_pages_blank
from s_ai_model import generate_analysis, generate_analysis_stream
from s_json_utils import extract_json
from s_db import log_request
from t_key_clause_extraction import classify_document, DOCUMENT_HANDLERS, extract_text_from_upload
from t_risk_detection import analyze_document_risks

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_FOLDER     = "temp"
MAX_PDF_PAGES     = None
_BLANK_PDF_RESULT = {"overview": "", "summary": "", "highlights": []}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Async wrapper for load_pdf
#
# PaddleOCR-VL's .predict() is a blocking synchronous call that can take
# 7-20s per page on GPU. Running it directly on the asyncio event loop
# freezes the entire server for that duration — no other requests are
# served, no timeouts fire, health checks fail.
#
# Solution: run load_pdf in the default ThreadPoolExecutor so the event
# loop remains free to handle other work while OCR runs in a thread.
# ---------------------------------------------------------------------------
async def _load_pdf_async(file_path: str, max_pages: int | None = None):
    """
    Non-blocking wrapper around load_pdf.
    Runs the synchronous PDF extraction + OCR in a thread pool so the
    asyncio event loop is never blocked.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,                              # default ThreadPoolExecutor
        partial(load_pdf, file_path, max_pages)
    )


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# POST /analyze — full JSON response
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights")
):
    """
    Analyse a PDF and return the full result as a single JSON response.
    Every request is logged to PostgreSQL (pdf_requests table).
    PDF extraction (including OCR) runs in a thread pool — event loop never blocked.
    """
    request_id    = str(uuid.uuid4())[:8]
    t_start       = time.perf_counter()
    total_in_tok  = 0
    total_out_tok = 0
    total_pages   = 0
    pages_to_read = 0
    was_truncated = False
    pdf_size      = 0
    status        = "success"
    error_msg     = None

    logger.info(f"[{request_id}] ── NEW REQUEST ──────────────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        content  = await file.read()
        pdf_size = len(content)
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[{request_id}] Step 1/5 — saved {pdf_size:,} bytes → '{safe_name}'")

        total_pages   = get_page_count(file_path)
        pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
        was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES
        logger.info(
            f"[{request_id}] Step 2/5 — pages={total_pages} analysing={pages_to_read} "
            f"{'(TRUNCATED)' if was_truncated else '(all pages)'}"
        )

        try:
            t_extract = time.perf_counter()
            # ── Non-blocking: OCR runs in thread pool ──────────────────────
            pages = await _load_pdf_async(file_path, pages_to_read)
            logger.info(
                f"[{request_id}] Step 3/5 — extracted {len(pages)} page(s) "
                f"({time.perf_counter()-t_extract:.2f}s)"
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        if all_pages_blank(pages):
            logger.warning(f"[{request_id}] blank PDF — skipping inference")
            status = "blank_pdf"
            result = dict(_BLANK_PDF_RESULT)
            result["blank_pdf"] = True
            if was_truncated:
                result.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)
            if analysis_type == 1: return {"overview": ""}
            if analysis_type == 2: return {"summary": ""}
            if analysis_type == 3: return {"highlights": []}
            return result

        merged_text = "\n\n".join(p.page_content for p in pages)
        logger.info(f"[{request_id}] Step 4/5 — merged {len(merged_text):,} chars")

        logger.info(f"[{request_id}] Step 5/5 — running inference")
        t_infer = time.perf_counter()
        final_output, total_in_tok, total_out_tok = await generate_analysis(merged_text)
        logger.info(f"[{request_id}] Step 5/5 — done ({time.perf_counter()-t_infer:.2f}s)")

    except HTTPException:
        status    = "error"
        error_msg = "HTTP error"
        raise
    except Exception as e:
        status    = "error"
        error_msg = str(e)
        logger.exception(f"[{request_id}] unhandled error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during PDF analysis.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] temp file deleted")

        elapsed = time.perf_counter() - t_start
        await log_request(
            request_id        = request_id,
            pdf_name          = file.filename or "unknown",
            pdf_size_bytes    = pdf_size,
            total_pages       = total_pages,
            pages_analysed    = pages_to_read,
            input_tokens      = total_in_tok,
            output_tokens     = total_out_tok,
            completion_time_s = elapsed,
            endpoint          = "/analyze",
            status            = status,
            error_message     = error_msg,
        )

    elapsed = time.perf_counter() - t_start
    logger.info(f"[{request_id}] ── COMPLETE — {elapsed:.2f}s ──────")

    if was_truncated:
        final_output.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)

    if analysis_type == 1: return {"overview":   final_output.get("overview", "")}
    if analysis_type == 2: return {"summary":    final_output.get("summary", "")}
    if analysis_type == 3: return {"highlights": final_output.get("highlights", [])}
    return final_output


# ---------------------------------------------------------------------------
# POST /key-clause-extraction
# ---------------------------------------------------------------------------

@router.post("/key-clause-extraction")
async def key_clause_extraction(file: UploadFile = File(...)):

    text, _, _, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/key-clause-extraction"
    )

    try:
        doc_type = await classify_document(text)
        doc_type = doc_type.lower().strip()
        logger.info(f"[{request_id}] Step 3 — classified as: '{doc_type}'")

        handler = DOCUMENT_HANDLERS.get(doc_type)

        if handler:
            result = await handler(text)
            logger.info(
                f"[{request_id}] ── REQUEST COMPLETE — "
                f"total time: {time.perf_counter() - t_start:.2f}s ──────"
            )
            return result

        logger.warning(f"[{request_id}] No handler found for doc_type='{doc_type}'")
        return {
            "status": "unsupported",
            "document_type": doc_type,
            "message": "Unsupported document type."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error during processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during key clause extraction.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] Temp file deleted: '{file_path}'")


# ---------------------------------------------------------------------------
# POST /detect-risks
# ---------------------------------------------------------------------------

@router.post("/detect-risks")
async def detect_risks(file: UploadFile = File(...)):
    """AI Risk Detection Endpoint."""
    text, _, _, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/detect-risks"
    )

    try:
        logger.info(f"[{request_id}] Starting Risk Detection...")
        result = await analyze_document_risks(text)
        return result

    except Exception as e:
        logger.exception(f"[{request_id}] Risk Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during risk detection.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ---------------------------------------------------------------------------
# POST /analyze/stream — Server-Sent Events
# ---------------------------------------------------------------------------

@router.post("/analyze/stream")
async def analyze_pdf_stream(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights")
):
    """
    Analyse a PDF and stream results in real time using Server-Sent Events.
    PDF extraction runs in a thread pool — event loop never blocked during OCR.
    Token usage and timing are logged to PostgreSQL after the stream completes.
    nginx: add proxy_buffering off to the location block for this route.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── NEW STREAM REQUEST ───────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        async def _err():
            yield _sse("error", {"message": "Only PDF files are accepted."})
        return StreamingResponse(_err(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    async def _generate():
        was_truncated    = False
        pages_to_read    = 0
        total_pages      = 0
        final_highlights = []
        final_overview   = ""
        final_summary    = ""
        total_in_tok     = 0
        total_out_tok    = 0
        pdf_size         = 0
        status           = "success"
        error_msg        = None

        try:
            yield _sse("status", {"step": "saving", "message": "Saving uploaded file..."})
            content  = await file.read()
            pdf_size = len(content)
            with open(file_path, "wb") as f:
                f.write(content)
            logger.info(f"[{request_id}] saved {pdf_size:,} bytes")

            total_pages   = get_page_count(file_path)
            pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
            was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES

            yield _sse("status", {
                "step":    "extracting",
                "message": f"Extracting text from {pages_to_read} page(s)...",
                "pages":   pages_to_read,
            })

            try:
                # ── Non-blocking: OCR runs in thread pool ──────────────────
                pages = await _load_pdf_async(file_path, pages_to_read)
            except ValueError as e:
                status    = "error"
                error_msg = str(e)
                yield _sse("error", {"message": str(e)})
                return

            if all_pages_blank(pages):
                status = "blank_pdf"
                logger.warning(f"[{request_id}] blank PDF")
                yield _sse("status", {"step": "blank", "message": "PDF has no extractable content."})
                if analysis_type in (0, 1): yield _sse("overview", {"text": ""})
                if analysis_type in (0, 2): yield _sse("summary",  {"text": ""})
                yield _sse("done", {
                    "total_time":       round(time.perf_counter() - t_start, 2),
                    "total_highlights": 0,
                    "pages":            total_pages,
                    "blank_pdf":        True,
                })
                return

            merged_text = "\n\n".join(p.page_content for p in pages)
            logger.info(f"[{request_id}] merged {len(merged_text):,} chars")

            overview_sent   = False
            highlight_index = 0

            async for event_type, payload in generate_analysis_stream(merged_text):

                if event_type == "chunk_start":
                    yield _sse("status", {
                        "step":    "inference",
                        "message": f"Analysing chunk {payload['chunk']} of {payload['total']}...",
                        "chunk":   payload["chunk"],
                        "total":   payload["total"],
                    })

                elif event_type == "token":
                    # Live token delta — forward immediately so the client
                    # can render the LLM output as it is being generated.
                    yield _sse("token", {
                        "chunk": payload["chunk"],
                        "delta": payload["delta"],
                    })

                elif event_type == "chunk_done":
                    if analysis_type in (0, 3):
                        for h in payload.get("new_highlights", []):
                            yield _sse("highlight", {"text": h, "index": highlight_index})
                            highlight_index += 1

                    final_highlights = payload.get("all_highlights_so_far", final_highlights)
                    yield _sse("status", {
                        "step":              "chunk_done",
                        "chunk":             payload["chunk"],
                        "total":             payload["total"],
                        "highlights_so_far": len(final_highlights),
                    })

                elif event_type == "synthesis_start":
                    yield _sse("status", {
                        "step": "synthesis", "message": "Writing overview and summary...",
                    })

                elif event_type == "synthesis_done":
                    # Fired for both single- and multi-chunk paths
                    final_overview = payload.get("overview", final_overview)
                    final_summary  = payload.get("summary", "")
                    if analysis_type in (0, 1):
                        yield _sse("overview", {"text": final_overview})
                        overview_sent = True
                    if analysis_type in (0, 2):
                        yield _sse("summary",  {"text": final_summary})

                elif event_type == "done":
                    total_in_tok  = payload.get("input_tokens",  0)
                    total_out_tok = payload.get("output_tokens", 0)

        except Exception as e:
            status    = "error"
            error_msg = str(e)
            logger.exception(f"[{request_id}] stream error: {e}")
            yield _sse("error", {"message": "Internal server error during PDF analysis."})
            return
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"[{request_id}] temp file deleted")

            elapsed = time.perf_counter() - t_start
            await log_request(
                request_id        = request_id,
                pdf_name          = file.filename or "unknown",
                pdf_size_bytes    = pdf_size,
                total_pages       = total_pages,
                pages_analysed    = pages_to_read,
                input_tokens      = total_in_tok,
                output_tokens     = total_out_tok,
                completion_time_s = elapsed,
                endpoint          = "/analyze/stream",
                status            = status,
                error_message     = error_msg,
            )

        elapsed = time.perf_counter() - t_start
        logger.info(f"[{request_id}] ── STREAM COMPLETE — {elapsed:.2f}s ──────")

        done_payload = {
            "total_time":       round(elapsed, 2),
            "total_highlights": len(final_highlights),
            "pages":            total_pages,
            "input_tokens":     total_in_tok,
            "output_tokens":    total_out_tok,
            "total_tokens":     total_in_tok + total_out_tok,
        }
        if was_truncated:
            done_payload.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)
        yield _sse("done", done_payload)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# POST /convert/pdf-to-docx
# ---------------------------------------------------------------------------

@router.post("/convert/pdf-to-docx")
async def convert_pdf_to_docx(file: UploadFile = File(...)):
    """
    Convert an uploaded PDF to a downloadable DOCX file.
    Handles native, scanned, and mixed PDFs. No AI inference.
    PDF extraction runs in a thread pool — event loop never blocked during OCR.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── PDF TO DOCX ───────────────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}'")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[{request_id}] saved {len(content):,} bytes")

        try:
            # ── Non-blocking: OCR runs in thread pool ──────────────────────
            pages = await _load_pdf_async(file_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        logger.info(f"[{request_id}] extracted {len(pages)} page(s)")

        def _do_convert():
            from s_pdf_to_docx import _extract_blocks_ocr, _build_docx
            all_blocks = []
            for page in pages:
                blocks = _extract_blocks_ocr(page.page_content)
                all_blocks.append(blocks)
            return _build_docx(all_blocks, file.filename)

        loop = asyncio.get_running_loop()
        docx = await loop.run_in_executor(None, _do_convert)

        buffer = io.BytesIO()
        docx.save(buffer)
        buffer.seek(0)

        output_filename = file.filename.replace(".pdf", ".docx")
        elapsed = time.perf_counter() - t_start
        logger.info(f"[{request_id}] ── DOCX COMPLETE — {elapsed:.2f}s → '{output_filename}'")

        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Pages":             str(len(pages)),
                "X-Processing-Time":   str(round(elapsed, 2)),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] conversion failed: {e}")
        raise HTTPException(status_code=500, detail="PDF to DOCX conversion failed.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] temp file deleted")
```

---
File: routes/s_route.py
---

```py
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
import os
import io
import uuid
import json
import time
import asyncio
import logging
from functools import partial

from utils.s_pdf_utils import load_pdf, get_page_count, all_pages_blank
from llm_model.s_ai_model import generate_analysis, generate_analysis_stream
from utils.s_json_utils import extract_json
from db_files.s_db import log_request
from feature_modules.key_clause_extraction import classify_document, DOCUMENT_HANDLERS, extract_text_from_upload
from feature_modules.risk_detection import analyze_document_risks
from utils.s_pdf_to_docx import _extract_blocks_ocr, _build_docx

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_FOLDER     = "temp"
MAX_PDF_PAGES     = None
_BLANK_PDF_RESULT = {"overview": "", "summary": "", "highlights": []}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Async wrapper for load_pdf
#
# PaddleOCR-VL's .predict() is a blocking synchronous call that can take
# 7-20s per page on GPU. Running it directly on the asyncio event loop
# freezes the entire server for that duration — no other requests are
# served, no timeouts fire, health checks fail.
#
# Solution: run load_pdf in the default ThreadPoolExecutor so the event
# loop remains free to handle other work while OCR runs in a thread.
# ---------------------------------------------------------------------------
async def _load_pdf_async(file_path: str, max_pages: int | None = None):
    """
    Non-blocking wrapper around load_pdf.
    Runs the synchronous PDF extraction + OCR in a thread pool so the
    asyncio event loop is never blocked.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,                              # default ThreadPoolExecutor
        partial(load_pdf, file_path, max_pages)
    )


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# POST /analyze — full JSON response
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights")
):
    """
    Analyse a PDF and return the full result as a single JSON response.
    Every request is logged to PostgreSQL (pdf_requests table).
    PDF extraction (including OCR) runs in a thread pool — event loop never blocked.
    """
    request_id    = str(uuid.uuid4())[:8]
    t_start       = time.perf_counter()
    total_in_tok  = 0
    total_out_tok = 0
    total_pages   = 0
    pages_to_read = 0
    was_truncated = False
    pdf_size      = 0
    status        = "success"
    error_msg     = None

    logger.info(f"[{request_id}] ── NEW REQUEST ──────────────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        content  = await file.read()
        pdf_size = len(content)
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[{request_id}] Step 1/5 — saved {pdf_size:,} bytes → '{safe_name}'")

        total_pages   = get_page_count(file_path)
        pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
        was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES
        logger.info(
            f"[{request_id}] Step 2/5 — pages={total_pages} analysing={pages_to_read} "
            f"{'(TRUNCATED)' if was_truncated else '(all pages)'}"
        )

        try:
            t_extract = time.perf_counter()
            # ── Non-blocking: OCR runs in thread pool ──────────────────────
            pages = await _load_pdf_async(file_path, pages_to_read)
            logger.info(
                f"[{request_id}] Step 3/5 — extracted {len(pages)} page(s) "
                f"({time.perf_counter()-t_extract:.2f}s)"
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        if all_pages_blank(pages):
            logger.warning(f"[{request_id}] blank PDF — skipping inference")
            status = "blank_pdf"
            result = dict(_BLANK_PDF_RESULT)
            result["blank_pdf"] = True
            if was_truncated:
                result.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)
            if analysis_type == 1: return {"overview": ""}
            if analysis_type == 2: return {"summary": ""}
            if analysis_type == 3: return {"highlights": []}
            return result

        merged_text = "\n\n".join(p.page_content for p in pages)
        logger.info(f"[{request_id}] Step 4/5 — merged {len(merged_text):,} chars")

        logger.info(f"[{request_id}] Step 5/5 — running inference")
        t_infer = time.perf_counter()
        final_output, total_in_tok, total_out_tok = await generate_analysis(merged_text)
        logger.info(f"[{request_id}] Step 5/5 — done ({time.perf_counter()-t_infer:.2f}s)")

    except HTTPException:
        status    = "error"
        error_msg = "HTTP error"
        raise
    except Exception as e:
        status    = "error"
        error_msg = str(e)
        logger.exception(f"[{request_id}] unhandled error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during PDF analysis.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] temp file deleted")

        elapsed = time.perf_counter() - t_start
        await log_request(
            request_id        = request_id,
            pdf_name          = file.filename or "unknown",
            pdf_size_bytes    = pdf_size,
            total_pages       = total_pages,
            pages_analysed    = pages_to_read,
            input_tokens      = total_in_tok,
            output_tokens     = total_out_tok,
            completion_time_s = elapsed,
            endpoint          = "/analyze",
            status            = status,
            error_message     = error_msg,
        )

    elapsed = time.perf_counter() - t_start
    logger.info(f"[{request_id}] ── COMPLETE — {elapsed:.2f}s ──────")

    if was_truncated:
        final_output.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)

    if analysis_type == 1: return {"overview":   final_output.get("overview", "")}
    if analysis_type == 2: return {"summary":    final_output.get("summary", "")}
    if analysis_type == 3: return {"highlights": final_output.get("highlights", [])}
    return final_output


# ---------------------------------------------------------------------------
# POST /key-clause-extraction
# ---------------------------------------------------------------------------

@router.post("/key-clause-extraction")
async def key_clause_extraction(file: UploadFile = File(...)):

    text, _, _, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/key-clause-extraction"
    )

    try:
        doc_type = await classify_document(text)
        doc_type = doc_type.lower().strip()
        logger.info(f"[{request_id}] Step 3 — classified as: '{doc_type}'")

        handler = DOCUMENT_HANDLERS.get(doc_type)

        if handler:
            result = await handler(text)
            logger.info(
                f"[{request_id}] ── REQUEST COMPLETE — "
                f"total time: {time.perf_counter() - t_start:.2f}s ──────"
            )
            return result

        logger.warning(f"[{request_id}] No handler found for doc_type='{doc_type}'")
        return {
            "status": "unsupported",
            "document_type": doc_type,
            "message": "Unsupported document type."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error during processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during key clause extraction.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] Temp file deleted: '{file_path}'")


# ---------------------------------------------------------------------------
# POST /detect-risks
# ---------------------------------------------------------------------------

@router.post("/detect-risks")
async def detect_risks(file: UploadFile = File(...)):
    """AI Risk Detection Endpoint."""
    text, _, _, request_id, t_start, file_path = await extract_text_from_upload(
        file,
        endpoint="/detect-risks"
    )

    try:
        logger.info(f"[{request_id}] Starting Risk Detection...")
        result = await analyze_document_risks(text)
        return result

    except Exception as e:
        logger.exception(f"[{request_id}] Risk Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during risk detection.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ---------------------------------------------------------------------------
# POST /analyze/stream — Server-Sent Events
# ---------------------------------------------------------------------------

@router.post("/analyze/stream")
async def analyze_pdf_stream(
    file: UploadFile = File(...),
    analysis_type: int = Query(0, ge=0, le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights")
):
    """
    Analyse a PDF and stream results in real time using Server-Sent Events.
    PDF extraction runs in a thread pool — event loop never blocked during OCR.
    Token usage and timing are logged to PostgreSQL after the stream completes.
    nginx: add proxy_buffering off to the location block for this route.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── NEW STREAM REQUEST ───────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}' analysis_type={analysis_type}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        async def _err():
            yield _sse("error", {"message": "Only PDF files are accepted."})
        return StreamingResponse(_err(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    async def _generate():
        was_truncated    = False
        pages_to_read    = 0
        total_pages      = 0
        final_highlights = []
        final_overview   = ""
        final_summary    = ""
        total_in_tok     = 0
        total_out_tok    = 0
        pdf_size         = 0
        status           = "success"
        error_msg        = None

        try:
            yield _sse("status", {"step": "saving", "message": "Saving uploaded file..."})
            content  = await file.read()
            pdf_size = len(content)
            with open(file_path, "wb") as f:
                f.write(content)
            logger.info(f"[{request_id}] saved {pdf_size:,} bytes")

            total_pages   = get_page_count(file_path)
            pages_to_read = total_pages if MAX_PDF_PAGES is None else min(total_pages, MAX_PDF_PAGES)
            was_truncated = MAX_PDF_PAGES is not None and total_pages > MAX_PDF_PAGES

            yield _sse("status", {
                "step":    "extracting",
                "message": f"Extracting text from {pages_to_read} page(s)...",
                "pages":   pages_to_read,
            })

            try:
                # ── Non-blocking: OCR runs in thread pool ──────────────────
                pages = await _load_pdf_async(file_path, pages_to_read)
            except ValueError as e:
                status    = "error"
                error_msg = str(e)
                yield _sse("error", {"message": str(e)})
                return

            if all_pages_blank(pages):
                status = "blank_pdf"
                logger.warning(f"[{request_id}] blank PDF")
                yield _sse("status", {"step": "blank", "message": "PDF has no extractable content."})
                if analysis_type in (0, 1): yield _sse("overview", {"text": ""})
                if analysis_type in (0, 2): yield _sse("summary",  {"text": ""})
                yield _sse("done", {
                    "total_time":       round(time.perf_counter() - t_start, 2),
                    "total_highlights": 0,
                    "pages":            total_pages,
                    "blank_pdf":        True,
                })
                return

            merged_text = "\n\n".join(p.page_content for p in pages)
            logger.info(f"[{request_id}] merged {len(merged_text):,} chars")

            overview_sent   = False
            highlight_index = 0

            async for event_type, payload in generate_analysis_stream(merged_text):

                if event_type == "chunk_start":
                    yield _sse("status", {
                        "step":    "inference",
                        "message": f"Analysing chunk {payload['chunk']} of {payload['total']}...",
                        "chunk":   payload["chunk"],
                        "total":   payload["total"],
                    })

                elif event_type == "token":
                    # Live token delta — forward immediately so the client
                    # can render the LLM output as it is being generated.
                    yield _sse("token", {
                        "chunk": payload["chunk"],
                        "delta": payload["delta"],
                    })

                elif event_type == "chunk_done":
                    if analysis_type in (0, 3):
                        for h in payload.get("new_highlights", []):
                            yield _sse("highlight", {"text": h, "index": highlight_index})
                            highlight_index += 1

                    final_highlights = payload.get("all_highlights_so_far", final_highlights)
                    yield _sse("status", {
                        "step":              "chunk_done",
                        "chunk":             payload["chunk"],
                        "total":             payload["total"],
                        "highlights_so_far": len(final_highlights),
                    })

                elif event_type == "synthesis_start":
                    yield _sse("status", {
                        "step": "synthesis", "message": "Writing overview and summary...",
                    })

                elif event_type == "synthesis_done":
                    # Fired for both single- and multi-chunk paths
                    final_overview = payload.get("overview", final_overview)
                    final_summary  = payload.get("summary", "")
                    if analysis_type in (0, 1):
                        yield _sse("overview", {"text": final_overview})
                        overview_sent = True
                    if analysis_type in (0, 2):
                        yield _sse("summary",  {"text": final_summary})

                elif event_type == "done":
                    total_in_tok  = payload.get("input_tokens",  0)
                    total_out_tok = payload.get("output_tokens", 0)

        except Exception as e:
            status    = "error"
            error_msg = str(e)
            logger.exception(f"[{request_id}] stream error: {e}")
            yield _sse("error", {"message": "Internal server error during PDF analysis."})
            return
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"[{request_id}] temp file deleted")

            elapsed = time.perf_counter() - t_start
            await log_request(
                request_id        = request_id,
                pdf_name          = file.filename or "unknown",
                pdf_size_bytes    = pdf_size,
                total_pages       = total_pages,
                pages_analysed    = pages_to_read,
                input_tokens      = total_in_tok,
                output_tokens     = total_out_tok,
                completion_time_s = elapsed,
                endpoint          = "/analyze/stream",
                status            = status,
                error_message     = error_msg,
            )

        elapsed = time.perf_counter() - t_start
        logger.info(f"[{request_id}] ── STREAM COMPLETE — {elapsed:.2f}s ──────")

        done_payload = {
            "total_time":       round(elapsed, 2),
            "total_highlights": len(final_highlights),
            "pages":            total_pages,
            "input_tokens":     total_in_tok,
            "output_tokens":    total_out_tok,
            "total_tokens":     total_in_tok + total_out_tok,
        }
        if was_truncated:
            done_payload.update(truncated=True, pages_analysed=pages_to_read, total_pages=total_pages)
        yield _sse("done", done_payload)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# POST /convert/pdf-to-docx
# ---------------------------------------------------------------------------

@router.post("/convert/pdf-to-docx")
async def convert_pdf_to_docx(file: UploadFile = File(...)):
    """
    Convert an uploaded PDF to a downloadable DOCX file.
    Handles native, scanned, and mixed PDFs. No AI inference.
    PDF extraction runs in a thread pool — event loop never blocked during OCR.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    logger.info(f"[{request_id}] ── PDF TO DOCX ───────────────────────────────")
    logger.info(f"[{request_id}] filename='{file.filename}'")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[{request_id}] saved {len(content):,} bytes")

        try:
            # ── Non-blocking: OCR runs in thread pool ──────────────────────
            pages = await _load_pdf_async(file_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        logger.info(f"[{request_id}] extracted {len(pages)} page(s)")

        def _do_convert():
            
            all_blocks = []
            for page in pages:
                blocks = _extract_blocks_ocr(page.page_content)
                all_blocks.append(blocks)
            return _build_docx(all_blocks, file.filename)

        loop = asyncio.get_running_loop()
        docx = await loop.run_in_executor(None, _do_convert)

        buffer = io.BytesIO()
        docx.save(buffer)
        buffer.seek(0)

        output_filename = file.filename.replace(".pdf", ".docx")
        elapsed = time.perf_counter() - t_start
        logger.info(f"[{request_id}] ── DOCX COMPLETE — {elapsed:.2f}s → '{output_filename}'")

        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Pages":             str(len(pages)),
                "X-Processing-Time":   str(round(elapsed, 2)),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] conversion failed: {e}")
        raise HTTPException(status_code=500, detail="PDF to DOCX conversion failed.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"[{request_id}] temp file deleted")
```

---
File: document_generation/prompt_templates.py
---

```py
# document_generation/prompt_templates.py

# Simulate LangChain's PromptTemplate for this example
# In a real scenario, you'd install and import from langchain_core.prompts
class SimulatedPromptTemplate:
    def __init__(self, template: str, input_variables: list[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        formatted_template = self.template
        for var in self.input_variables:
            if var in kwargs:
                # Replace placeholders with actual values, ensuring proper escaping if necessary
                formatted_template = formatted_template.replace(f"{{{var}}}", str(kwargs[var]))
        return formatted_template

# Define prompt templates for different document types
# These could be loaded from external files or a database in a real application
prompt_templates = {
    "offer_letter": SimulatedPromptTemplate(
        template="""You are an AI assistant that generates professional offer letters in HTML format.
Generate a complete, well-structured HTML document for an offer letter based on the following details:

User Request: {user_request}

Ensure the HTML includes `<html>`, `<head>`, `<style>`, and `<body>`. Make the entire `<body>` tag (or its direct primary content container, like a main `div`) editable by adding the `contenteditable='true'` attribute. All text content inside the body should then be editable.
Do NOT include any markdown, backticks, or extra text outside the HTML structure.""",
        input_variables=["user_request"]
    ),
    "invoice": SimulatedPromptTemplate(
        template="""You are an AI assistant that generates invoices in HTML format.
Generate a complete, well-structured HTML document for an invoice based on the following details:

User Request: {user_request}

Ensure the HTML includes `<html>`, `<head>`, `<style>`, and `<body>`. Make the entire `<body>` tag (or its direct primary content container, like a main `div`) editable by adding the `contenteditable='true'` attribute. All text content inside the body should then be editable.
Do NOT include any markdown, backticks, or extra text outside the HTML structure.""",
        input_variables=["user_request"]
    ),
    "example_type": SimulatedPromptTemplate(
        template="""You are an AI assistant that generates example HTML documents.
Generate a complete HTML document based on the user's prompt and document ID.

User Prompt: {user_prompt}
Document ID: {document_id}

Ensure the HTML includes `<html>`, `<head>`, `<style>`, and `<body>`. Make the entire `<body>` tag (or its direct primary content container, like a main `div`) editable by adding the `contenteditable='true'` attribute. All text content inside the body should then be editable.
Do NOT include any markdown or extra text outside the HTML.""",
        input_variables=["user_prompt", "document_id"]
    )
}

# Add this to your existing prompt_templates.py

REGENERATE_PROMPT = SimulatedPromptTemplate(
    template="""You are an expert HTML editor. 
You will be provided with an existing HTML document and a user's request for modification.

GOAL:
Update the existing HTML content based on the user's instructions while maintaining the same design, layout, and CSS styles.

STRICT RULES:
1. Return ONLY the complete, updated HTML document.
2. Ensure the `<body>` (or main container) remains `contenteditable='true'`.
3. Do NOT include markdown backticks (```html), explanations, or notes.
4. Preserve all original <style> blocks unless the user explicitly asks to change colors/layout.

Existing HTML:
{existing_html}

Modification Request:
{user_query}
""",
    input_variables=["existing_html", "user_query"]
)

REGENERATE_PROMPT = SimulatedPromptTemplate(
    template="""You are an expert HTML editor. 
You will be provided with an existing HTML document and a user's request for modification.

GOAL:
Update the existing HTML content based on the user's instructions while maintaining the exact same CSS styles, design, and layout.

STRICT RULES:
1. Return ONLY the complete, updated HTML document starting with <html> and ending with </html>.
2. Ensure the `<body>` (or its primary content container) remains `contenteditable='true'`.
3. Do NOT include markdown backticks (```html), explanations, or notes.
4. Preserve all original <style> blocks and CSS logic.
5. Apply the changes requested by the user accurately.

Existing HTML:
{existing_html}

User Modification Request:
{modification_query}
""",
    input_variables=["existing_html", "modification_query"]
)
```

---
File: document_generation/document_generator.py
---

```py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import HTMLResponse
from s_ai_model import run_llm
from .prompt_templates import SimulatedPromptTemplate, prompt_templates, REGENERATE_PROMPT 

router = APIRouter()

class DocumentGenerationRequest(BaseModel):
    document_id: str
    document_type: str
    user_prompt: str

@router.post("/generate-html", response_class=HTMLResponse)
async def generate_document_html(request: DocumentGenerationRequest):
    """
    Generates HTML for a document based on the provided document type and user prompt using prompt templates.
    """
    selected_template = prompt_templates.get(request.document_type)

    if not selected_template:
        raise HTTPException(status_code=404, detail=f"Prompt template not found for document type: {request.document_type}")

    # Format the system prompt using the selected template and request data
    if request.document_type == "example_type":
        system_prompt = selected_template.format(
            user_prompt=request.user_prompt,
            document_id=request.document_id
        )
    else:
        system_prompt = selected_template.format(user_request=request.user_prompt)

    # Call the AI model
    try:
        generated_html = await run_llm(
            text=request.user_prompt, # The user's actual prompt goes here for the LLM to process
            system_prompt=system_prompt,
        )
        return generated_html
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI model generation failed: {str(e)}")
    


class DocumentRegenerationRequest(BaseModel):
    existing_html: str
    modification_query: str


def clean_html_output(raw_html: str) -> str:
    """Removes common LLM artifacts like markdown code blocks."""
    cleaned = raw_html.replace("```html", "").replace("```", "").strip()
    return cleaned

@router.post("/regenerate-html", response_class=HTMLResponse)
async def regenerate_document_html(request: DocumentRegenerationRequest):
    """
    Updates an existing HTML response based on a modification query.
    """
    if not request.existing_html.strip():
        raise HTTPException(status_code=400, detail="Existing HTML content is empty.")

    # Format the specialized regeneration prompt
    system_prompt = REGENERATE_PROMPT.format(
        existing_html=request.existing_html,
        modification_query=request.modification_query
    )

    try:
        # Call AI with the modification request
        updated_raw = await run_llm(
            text=f"Modify this document as follows: {request.modification_query}",
            system_prompt=system_prompt,
        )

        # Clean backticks and return as HTML
        cleaned_html = clean_html_output(updated_raw)
        return HTMLResponse(content=cleaned_html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {str(e)}")
```

---
File: backupfiles/pdf_utils.py
---

```py
import fitz  # PyMuPDF


# def extract_text_from_pdf(file_path: str) -> str:
#     """
#     Extract ONLY real text from PDF.
#     Image-only pages are skipped.
#     """
#     text_content = []
#     with fitz.open(file_path) as doc:
#         for page in doc:
#             text = page.get_text("text")
#             if not text or not text.strip():
#                 continue
#             text_content.append(text.strip())
#     return "\n".join(text_content)


# def chunk_text(text: str, chunk_size: int = 10000):
#     """
#     Split large text into smaller chunks
#     """
#     if not text:
#         return []
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# import re
# import fitz  # PyMuPDF
# import numpy as np
# from paddleocr import PaddleOCR

# # Initialize OCR only once (important for performance)
# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang="en",
#     use_gpu=True  # set False if no GPU
# )


# def clean_text(text: str) -> str:
#     """
#     Normalize and clean extracted text for better LLM results.
#     """
#     text = re.sub(r"\r", "\n", text)
#     text = re.sub(r"\n{2,}", "\n", text)
#     text = re.sub(r"[ \t]+", " ", text)
#     text = re.sub(r"[^\x00-\x7F]+", " ", text)
#     return text.strip()


# def extract_text_from_pdf(file_path: str) -> str:
#     """
#     Hybrid extraction:
#     - Try embedded text first
#     - If page has little/no text → use OCR only for that page
#     - Works for normal, scanned, and mixed PDFs
#     """

#     final_text = []

#     with fitz.open(file_path) as doc:
#         for page_index, page in enumerate(doc):

#             # 1️⃣ Try extracting embedded text
#             text = page.get_text("text")

#             if text and len(text.strip()) > 50:
#                 final_text.append(text.strip())
#                 continue

#             # 2️⃣ If not enough text → use OCR for this page
#             pix = page.get_pixmap(dpi=300)

#             # Convert to numpy array (no temp file needed)
#             img = np.frombuffer(pix.samples, dtype=np.uint8)
#             img = img.reshape(pix.height, pix.width, pix.n)

#             result = ocr.ocr(img, cls=True)

#             page_text = []
#             for line in result:
#                 for word_info in line:
#                     page_text.append(word_info[-1][0])

#             final_text.append(" ".join(page_text))

#     return clean_text("\n".join(final_text))


# def chunk_text(text: str, chunk_size: int = 2500):
#     """
#     Smart sentence-based chunking.
#     Prevents breaking context mid-sentence.
#     """

#     if not text:
#         return []

#     sentences = re.split(r'(?<=[.!?]) +', text)

#     chunks = []
#     current_chunk = ""

#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) <= chunk_size:
#             current_chunk += sentence + " "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + " "

#     if current_chunk:
#         chunks.append(current_chunk.strip())

#     return chunks





# ==============OCR-VL-1.5-0.9B VERSION (PaddleOCRVL)================

import re
import fitz  # PyMuPDF
import numpy as np
import paddle # Added for memory management
from paddleocr import PaddleOCRVL

# Initialize the 0.9B VLM once
# "v1.5" is the correct shorthand for PaddleOCR-VL-1.5-0.9B
ocr_vl = PaddleOCRVL("v1.5")

# def clean_text(text: str) -> str:
#     """Normalize and clean extracted text for better LLM results."""
#     text = re.sub(r"\r", "\n", text)
#     text = re.sub(r"\n{2,}", "\n", text)
#     text = re.sub(r"[ \t]+", " ", text)
#     text = re.sub(r"[^\x00-\x7F]+", " ", text)
#     return text.strip()

# def chunk_text(text: str, chunk_size: int = 10000):
#     """Split large text into smaller chunks."""
#     if not text:
#         return []
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def extract_text_from_pdf(file_path: str) -> str:
    final_text = []

    with fitz.open(file_path) as doc:
        for page in doc:
            # 1. Native Extraction (Fastest)
            text = page.get_text("text")
            if len(text.strip()) > 50:
                final_text.append(text.strip())
                continue

            # 2. VLM Extraction (For scanned pages)
            # Dropping DPI to 150 helps prevent the "MemoryError" you faced
            pix = page.get_pixmap(dpi=150) 
            
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: 
                img = img[:, :, :3]

            # Run prediction
            results = ocr_vl.predict(img)
            
            # --- FIX: Extract text using the correct attribute ---
            # Most PaddleX/OCR-VL objects use .res for the final string
            page_parts = []
            for res in results:
                if hasattr(res, 'res'):
                    page_parts.append(res.res)
                elif isinstance(res, dict) and 'res' in res:
                    page_parts.append(res['res'])
                else:
                    page_parts.append(str(res))
            
            final_text.append("\n".join(page_parts))

            # --- MEMORY CLEANUP ---
            # Critical for preventing OOM crashes on long PDFs
            del results
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()

    return clean_text("\n".join(final_text))




# updated code



# def extract_text_from_pdf(file_path: str) -> str:
#     """
#     Extract ONLY real text from PDF.
#     Image-only pages are skipped.
#     Raises ValueError if the PDF cannot be opened (e.g. encrypted or corrupt).
#     """
#     try:
#         text_content = []
#         with fitz.open(file_path) as doc:
#             for page in doc:
#                 text = page.get_text("text")
#                 if not text or not text.strip():
#                     continue
#                 text_content.append(text.strip())
#         return "\n".join(text_content)
#     except Exception as e:
#         raise ValueError(f"Could not open PDF: {e}")


def clean_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts:
    - Rejoin hyphenated line breaks
    - Collapse excessive blank lines
    - Collapse repeated spaces
    """
    text = re.sub(r'-\n', '', text)        # rejoin hyphenated words
    text = re.sub(r'\n{3,}', '\n\n', text) # collapse 3+ newlines to double
    text = re.sub(r' {2,}', ' ', text)     # collapse repeated spaces
    return text.strip()


def chunk_text(text: str, max_chars: int = 10000) -> list:
    """
    Split large text into smaller chunks on paragraph boundaries
    to avoid cutting mid-sentence.
    """
    if not text:
        return []

    paragraphs = text.split("\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = para
        else:
            current += "\n" + para

    if current.strip():
        chunks.append(current.strip())

    return chunks


```

---
File: backupfiles/t_ai_model.py
---

```py
"""
t_ai_model.py
=============
Mistral-7B-Instruct inference for hierarchical PDF summarization.

Features
--------
- Lazy model loading  (call load_model() once from FastAPI lifespan)
- GPU semaphore       (max MAX_CONCURRENT_GPU simultaneous generate() calls)
- Retry + backoff     (up to MAX_RETRY attempts per call)
- Three prompt modes:
    generate_analysis()   — analyse one text chunk   (Level 0)
    synthesize_section()  — condense N chunk results  (Level 1)
    synthesize_final()    — condense all sections     (Level 2)
- Correct device handling (device chosen before model load; no mismatch)
- Single tokenize pass with truncation (no double-tokenize bug)
- Correct decode slice  (uses truncated input length, not pre-truncation)
- asyncio.get_running_loop() instead of deprecated get_event_loop()
"""

import asyncio
import logging

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME         = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_INPUT_TOKENS   = 6_000   # hard token limit per inference call
MAX_NEW_TOKENS     = 600     # max tokens the model may generate per call
MAX_CONCURRENT_GPU = 3       # GPU semaphore width  (prevents OOM)
MAX_RETRY          = 3       # retry attempts for transient failures

# ---------------------------------------------------------------------------
# Module-level state  (populated by load_model())
# ---------------------------------------------------------------------------
_tokenizer  = None
_model      = None
_device     = None
_gpu_sem: asyncio.Semaphore | None = None   # created lazily on first use


def _get_sem() -> asyncio.Semaphore:
    """Return (and lazily create) the GPU semaphore on the running event loop."""
    global _gpu_sem
    if _gpu_sem is None:
        _gpu_sem = asyncio.Semaphore(MAX_CONCURRENT_GPU)
    return _gpu_sem


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> None:
    """
    Load tokenizer and model weights into memory.

    - Device is chosen BEFORE the model is loaded (no device_map="auto" mismatch).
    - CUDA → float16 with cudnn.benchmark and allow_tf32.
    - CPU  → float32 (float16 not supported on CPU).
    - Safe to call multiple times — subsequent calls are no-ops.
    """
    global _tokenizer, _model, _device

    if _model is not None:
        return  # already loaded

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading {MODEL_NAME} onto {_device} …")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if _device == "cuda":
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        ).to(_device)
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )

    _model.eval()
    logger.info("Model ready.")


# ---------------------------------------------------------------------------
# Core synchronous inference  (runs in thread pool via run_in_executor)
# ---------------------------------------------------------------------------

def _run_inference(prompt: str) -> str:
    """
    Tokenize → generate → decode.  Entirely synchronous.

    Correctness fixes vs original code
    -----------------------------------
    - Single tokenize call (was tokenized twice — once to count, once to truncate)
    - input_length taken from the truncated tensor (not pre-truncation encoding)
    - inputs.to(_device) consistent with model device (no device_map mismatch)
    - GPU memory freed in finally block regardless of success or failure
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    # Single tokenize pass with truncation
    inputs       = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    input_length = inputs["input_ids"].shape[1]   # length AFTER truncation

    if input_length == MAX_INPUT_TOKENS:
        logger.warning(f"Prompt truncated to {MAX_INPUT_TOKENS} tokens.")

    inputs = inputs.to(_device)

    try:
        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=_tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU out of memory. Reduce chunk size or MAX_INPUT_TOKENS."
        )
    finally:
        del inputs
        if _device == "cuda":
            torch.cuda.empty_cache()

    # Decode only newly generated tokens (slice from truncated input_length)
    new_tokens = output[0][input_length:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Async wrapper with semaphore + retry
# ---------------------------------------------------------------------------

async def _call_model(prompt: str) -> str:
    """
    Run inference asynchronously with GPU semaphore and retry.

    - Acquires semaphore before submitting to the thread pool.
    - Retries up to MAX_RETRY times with exponential back-off (1 s, 2 s, 4 s).
    - Returns "" on permanent failure (caller handles empty result gracefully).
    """
    loop = asyncio.get_running_loop()
    sem  = _get_sem()

    for attempt in range(MAX_RETRY):
        try:
            async with sem:
                result = await loop.run_in_executor(None, _run_inference, prompt)
            return result
        except RuntimeError as exc:
            if attempt == MAX_RETRY - 1:
                logger.error(
                    f"Inference failed after {MAX_RETRY} attempts: {exc}"
                )
                return ""
            wait = 2 ** attempt   # 1 s → 2 s → 4 s
            logger.warning(
                f"Inference attempt {attempt+1} failed ({exc}); "
                f"retrying in {wait}s …"
            )
            await asyncio.sleep(wait)

    return ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _chunk_prompt(text: str) -> str:
    """Level 0 prompt: analyse a single content chunk."""
    return f"""<s>[INST]
You are an expert document analyst with deep knowledge across legal, medical,
financial, technical, academic, and business domains.

Read the document EXCERPT below and return ONLY a JSON object with these three fields:

{{
  "overview": "1-2 sentences: what TYPE of document this is, its subject, purpose, and intended audience.",
  "summary": "4-6 sentences covering the most important content in this excerpt.",
  "highlights": [
    "One specific factual sentence containing a real number, date, name, percentage, or clause.",
    "One specific factual sentence containing a real number, date, name, percentage, or clause.",
    "One specific factual sentence containing a real number, date, name, percentage, or clause.",
    "One specific factual sentence containing a real number, date, name, percentage, or clause."
  ]
}}

STRICT RULES:
- Output ONLY the JSON object. No markdown, no backticks, no commentary.
- highlights: 4-6 items. Each must be a complete factual sentence with real values.
- Do NOT write vague statements like "The document contains important information."
- If a section has insufficient information, write "Not enough information available."
- Escape all special characters in strings properly.

Document excerpt:
----------------
{text}
----------------
[/INST]"""


def _section_prompt(chunk_summaries: list[str], section_label: str) -> str:
    """Level 1 prompt: synthesize N chunk summaries into one section summary."""
    numbered = "\n\n".join(
        f"[Chunk {i+1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    return f"""<s>[INST]
You are an expert document analyst. You are given {len(chunk_summaries)} sequential
chunk summaries from {section_label} of a large document.

Synthesize them into ONE coherent JSON object:

{{
  "overview": "1-2 sentences describing what this section covers.",
  "summary": "4-6 sentences summarising the key content across all chunks in this section.",
  "highlights": [
    "Most important specific fact, figure, or finding from this section.",
    "Second most important fact with a real number, date, name, or clause.",
    "Third most important fact.",
    "Fourth most important fact."
  ]
}}

STRICT RULES:
- Output ONLY the JSON. No markdown, no backticks.
- Do NOT simply list each chunk — synthesize them into a unified narrative.
- Eliminate any repeated points across chunks.
- highlights: 4-6 items, each a complete factual sentence with real values.

Chunk summaries:
----------------
{numbered}
----------------
[/INST]"""


def _final_prompt(section_summaries: list[str], total_pages: int) -> str:
    """Level 2 prompt: synthesize all section summaries into the final output."""
    numbered = "\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(section_summaries)
    )
    return f"""<s>[INST]
You are an expert document analyst. You have been given {len(section_summaries)} section-level
summaries that together cover a {total_pages}-page document in full.

Produce ONE authoritative document-level JSON object:

{{
  "overview": "1-2 sentences: document type, overall subject, purpose, and intended audience.",
  "summary": "5-7 sentences covering the document's purpose, key arguments or findings,
              important data or obligations, conclusions, and overall significance.",
  "highlights": [
    "The single most critical fact, figure, deadline, or obligation in the entire document.",
    "Second most important fact with a real number, date, name, percentage, or clause.",
    "Third most important fact.",
    "Fourth most important fact.",
    "Fifth most important fact.",
    "Sixth most important fact.",
    "Seventh most important fact (if applicable).",
    "Eighth most important fact (if applicable).",
    "Ninth most important fact (if applicable).",
    "Tenth most important fact (if applicable)."
  ]
}}

STRICT RULES:
- Output ONLY the JSON. No markdown, no backticks, no commentary.
- highlights: 6-10 items ranked by importance. Each must be a complete factual sentence.
- Do NOT repeat information already in the overview or summary.
- Eliminate all repetition across sections.
- Write in a professional, neutral tone.

Section summaries:
----------------
{numbered}
----------------
[/INST]"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_analysis(text: str) -> str:
    """
    Level 0 — analyse a single text chunk.
    Returns the raw model output string (caller parses with extract_json()).
    """
    return await _call_model(_chunk_prompt(text))


async def synthesize_section(
    chunk_results: list[dict],
    section_label: str = "a section",
) -> str:
    """
    Level 1 — condense the parsed results of N chunks into one section summary.

    Parameters
    ----------
    chunk_results : list of dicts with keys overview / summary / highlights
    section_label : human-readable label used in the prompt

    Returns raw model output string (caller parses with extract_json()).
    """
    chunk_summaries = []
    for r in chunk_results:
        parts = []
        if r.get("overview"):
            parts.append(f"Overview: {r['overview']}")
        if r.get("summary"):
            parts.append(f"Summary: {r['summary']}")
        if r.get("highlights"):
            hl = "\n".join(f"  - {h}" for h in r["highlights"])
            parts.append(f"Highlights:\n{hl}")
        chunk_summaries.append("\n".join(parts))

    return await _call_model(_section_prompt(chunk_summaries, section_label))


async def synthesize_final(
    section_results: list[dict],
    total_pages: int,
) -> str:
    """
    Level 2 — condense all section-level summaries into the final document output.

    Parameters
    ----------
    section_results : list of dicts with keys overview / summary / highlights
    total_pages     : original PDF page count (used in the prompt for context)

    Returns raw model output string (caller parses with extract_json()).
    """
    section_summaries = []
    for r in section_results:
        parts = []
        if r.get("overview"):
            parts.append(f"Overview: {r['overview']}")
        if r.get("summary"):
            parts.append(f"Summary: {r['summary']}")
        if r.get("highlights"):
            hl = "\n".join(f"  - {h}" for h in r["highlights"])
            parts.append(f"Highlights:\n{hl}")
        section_summaries.append("\n".join(parts))

    return await _call_model(_final_prompt(section_summaries, total_pages))

async def run_llm(
    text: str,
    system_prompt: str,
    max_input_tokens: int = MAX_INPUT_TOKENS
) -> str:
    """
    Generic reusable LLM runner.
    Accepts:
    - text (document content)
    - system_prompt (custom instruction per use-case)

    Returns:
    - model output (string)
    """

    prompt = f"""<s>[INST]
{system_prompt}

Document:
----------------
{text[:4000]}
----------------
[/INST]
"""

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_inference, prompt)

    return result


```

---
File: backupfiles/pdf_utils_backup
---

```text
import fitz  # PyMuPDF


# def extract_text_from_pdf(file_path: str) -> str:
#     """
#     Extract ONLY real text from PDF.
#     Image-only pages are skipped.
#     """
#     text_content = []
#     with fitz.open(file_path) as doc:
#         for page in doc:
#             text = page.get_text("text")
#             if not text or not text.strip():
#                 continue
#             text_content.append(text.strip())
#     return "\n".join(text_content)


# def chunk_text(text: str, chunk_size: int = 10000):
#     """
#     Split large text into smaller chunks
#     """
#     if not text:
#         return []
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# import re
# import fitz  # PyMuPDF
# import numpy as np
# from paddleocr import PaddleOCR

# # Initialize OCR only once (important for performance)
# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang="en",
#     use_gpu=True  # set False if no GPU
# )


# def clean_text(text: str) -> str:
#     """
#     Normalize and clean extracted text for better LLM results.
#     """
#     text = re.sub(r"\r", "\n", text)
#     text = re.sub(r"\n{2,}", "\n", text)
#     text = re.sub(r"[ \t]+", " ", text)
#     text = re.sub(r"[^\x00-\x7F]+", " ", text)
#     return text.strip()


# def extract_text_from_pdf(file_path: str) -> str:
#     """
#     Hybrid extraction:
#     - Try embedded text first
#     - If page has little/no text → use OCR only for that page
#     - Works for normal, scanned, and mixed PDFs
#     """

#     final_text = []

#     with fitz.open(file_path) as doc:
#         for page_index, page in enumerate(doc):

#             # 1️⃣ Try extracting embedded text
#             text = page.get_text("text")

#             if text and len(text.strip()) > 50:
#                 final_text.append(text.strip())
#                 continue

#             # 2️⃣ If not enough text → use OCR for this page
#             pix = page.get_pixmap(dpi=300)

#             # Convert to numpy array (no temp file needed)
#             img = np.frombuffer(pix.samples, dtype=np.uint8)
#             img = img.reshape(pix.height, pix.width, pix.n)

#             result = ocr.ocr(img, cls=True)

#             page_text = []
#             for line in result:
#                 for word_info in line:
#                     page_text.append(word_info[-1][0])

#             final_text.append(" ".join(page_text))

#     return clean_text("\n".join(final_text))


# def chunk_text(text: str, chunk_size: int = 2500):
#     """
#     Smart sentence-based chunking.
#     Prevents breaking context mid-sentence.
#     """

#     if not text:
#         return []

#     sentences = re.split(r'(?<=[.!?]) +', text)

#     chunks = []
#     current_chunk = ""

#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) <= chunk_size:
#             current_chunk += sentence + " "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + " "

#     if current_chunk:
#         chunks.append(current_chunk.strip())

#     return chunks





# ==============OCR-VL-1.5-0.9B VERSION (PaddleOCRVL)================

import re
import fitz  # PyMuPDF
import numpy as np
import paddle # Added for memory management
from paddleocr import PaddleOCRVL

# Initialize the 0.9B VLM once
# "v1.5" is the correct shorthand for PaddleOCR-VL-1.5-0.9B
ocr_vl = PaddleOCRVL("v1.5")

# def clean_text(text: str) -> str:
#     """Normalize and clean extracted text for better LLM results."""
#     text = re.sub(r"\r", "\n", text)
#     text = re.sub(r"\n{2,}", "\n", text)
#     text = re.sub(r"[ \t]+", " ", text)
#     text = re.sub(r"[^\x00-\x7F]+", " ", text)
#     return text.strip()

# def chunk_text(text: str, chunk_size: int = 10000):
#     """Split large text into smaller chunks."""
#     if not text:
#         return []
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def extract_text_from_pdf(file_path: str) -> str:
    final_text = []

    with fitz.open(file_path) as doc:
        for page in doc:
            # 1. Native Extraction (Fastest)
            text = page.get_text("text")
            if len(text.strip()) > 50:
                final_text.append(text.strip())
                continue

            # 2. VLM Extraction (For scanned pages)
            # Dropping DPI to 150 helps prevent the "MemoryError" you faced
            pix = page.get_pixmap(dpi=150) 
            
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: 
                img = img[:, :, :3]

            # Run prediction
            results = ocr_vl.predict(img)
            
            # --- FIX: Extract text using the correct attribute ---
            # Most PaddleX/OCR-VL objects use .res for the final string
            page_parts = []
            for res in results:
                if hasattr(res, 'res'):
                    page_parts.append(res.res)
                elif isinstance(res, dict) and 'res' in res:
                    page_parts.append(res['res'])
                else:
                    page_parts.append(str(res))
            
            final_text.append("\n".join(page_parts))

            # --- MEMORY CLEANUP ---
            # Critical for preventing OOM crashes on long PDFs
            del results
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()

    return clean_text("\n".join(final_text))




# updated code



# def extract_text_from_pdf(file_path: str) -> str:
#     """
#     Extract ONLY real text from PDF.
#     Image-only pages are skipped.
#     Raises ValueError if the PDF cannot be opened (e.g. encrypted or corrupt).
#     """
#     try:
#         text_content = []
#         with fitz.open(file_path) as doc:
#             for page in doc:
#                 text = page.get_text("text")
#                 if not text or not text.strip():
#                     continue
#                 text_content.append(text.strip())
#         return "\n".join(text_content)
#     except Exception as e:
#         raise ValueError(f"Could not open PDF: {e}")


def clean_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts:
    - Rejoin hyphenated line breaks
    - Collapse excessive blank lines
    - Collapse repeated spaces
    """
    text = re.sub(r'-\n', '', text)        # rejoin hyphenated words
    text = re.sub(r'\n{3,}', '\n\n', text) # collapse 3+ newlines to double
    text = re.sub(r' {2,}', ' ', text)     # collapse repeated spaces
    return text.strip()


def chunk_text(text: str, max_chars: int = 10000) -> list:
    """
    Split large text into smaller chunks on paragraph boundaries
    to avoid cutting mid-sentence.
    """
    if not text:
        return []

    paragraphs = text.split("\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = para
        else:
            current += "\n" + para

    if current.strip():
        chunks.append(current.strip())

    return chunks


```

---
File: backupfiles/main_backup
---

```text
# from fastapi import FastAPI, UploadFile, File, Query
# import os, json, re
# from pdf_utils import extract_text_from_pdf, chunk_text
# from ai_model import generate_analysis

# app = FastAPI()

# UPLOAD_FOLDER = "temp"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# def clean_text(text: str) -> str:
#     """Clean PDF text for better model analysis"""
#     text = text.replace("\n\n", "\n")  # remove extra blank lines
#     text = text.strip()
#     return text


# def extract_json(text):
#     try:
#         # Try JSON first
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if match:
#             return json.loads(match.group())
#     except:
#         pass

#     # Fallback: parse text format
#     summary = ""
#     positive = []
#     negative = []
#     avoid = []

#     sections = text.split("\n")

#     current = None
#     for line in sections:
#         line = line.strip()

#         if "Summary" in line:
#             current = "summary"
#             continue
#         elif "Positive" in line:
#             current = "positive"
#             continue
#         elif "Negative" in line:
#             current = "negative"
#             continue
#         elif "Avoid" in line:
#             current = "avoid"
#             continue

#         if current == "summary":
#             summary += line + " "

#         elif current == "positive" and line:
#             positive.append(line)

#         elif current == "negative" and line:
#             negative.append(line)

#         elif current == "avoid" and line:
#             avoid.append(line)

#     return {
#         "summary": summary.strip(),
#         "positive_points": positive,
#         "negative_points": negative,
#         "how_to_avoid": avoid
#     }


# @app.post("/analyze")
# async def analyze_pdf(
#     file: UploadFile = File(...),
#     analysis_type: int = Query(
#         0, ge=0, le=4,
#         description="0=all, 1=summary, 2=positive, 3=negative, 4=how_to_avoid"
#     )
# ):
#     """
#     Advanced PDF analysis with chunking and selective section output.
#     """

#     # Save uploaded PDF
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     # Extract text from PDF
#     text = extract_text_from_pdf(file_path)
#     text = clean_text(text)

#     # Chunk the text
#     chunks = chunk_text(text)

#     results = []

#     for chunk in chunks:
#         ai_response = generate_analysis(chunk)
#         print(ai_response)   # DEBUG
#         parsed = extract_json(ai_response)
#         results.append(parsed)

#     # Merge results
#     final_output = {
#         "summary": " ".join([r.get("summary", "") for r in results]),
#         "positive_points": sum([r.get("positive_points", []) for r in results], []),
#         "negative_points": sum([r.get("negative_points", []) for r in results], []),
#         "how_to_avoid": sum([r.get("how_to_avoid", []) for r in results], [])
#     }

#     # Return based on analysis_type
#     if analysis_type == 1:
#         return {"summary": final_output["summary"]}
#     elif analysis_type == 2:
#         return {"positive_points": final_output["positive_points"]}
#     elif analysis_type == 3:
#         return {"negative_points": final_output["negative_points"]}
#     elif analysis_type == 4:
#         return {"how_to_avoid": final_output["how_to_avoid"]}

#     return final_output






# def extract_json(text: str):
#     """Extract strict JSON from model output"""
#     try:
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if match:
#             return json.loads(match.group())
#     except Exception as e:
#         print("JSON parse error:", e)

#     # Safe fallback
#     return {
#         "overview": "",
#         "summary": "",
#         "highlights": []
#     }


# def extract_json(text: str):
#     """Extract and repair JSON from model output"""
#     try:
#         # 1. Find the first '{' and last '}' to strip any preamble or rambling
#         start = text.find('{')
#         end = text.rfind('}')
#         if start != -1 and end != -1:
#             json_str = text[start:end+1]
            
#             # 2. Fix common "escape" errors (replaces \ with \\ unless it's a valid escape)
#             # This fixes the "Invalid \escape" error you saw
#             json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
            
#             return json.loads(json_str)
#     except Exception as e:
#         print(f"JSON repair failed: {e}")

#     # Fallback to manual regex if JSON is totally mangled
#     return {
#         "overview": re.search(r'"overview":\s*"(.*?)"', text).group(1) if '"overview"' in text else "",
#         "summary": re.search(r'"summary":\s*"(.*?)"', text).group(1) if '"summary"' in text else "",
#         "highlights": re.findall(r'"highlights":\s*\[(.*?)\]', text, re.DOTALL) or []
#     }



#==================this is final code ===========================

# from fastapi import FastAPI, UploadFile, File, Query
# import os
# import json
# import re
# from pdf_utils import extract_text_from_pdf, chunk_text
# from ai_model import generate_analysis

# app = FastAPI()

# UPLOAD_FOLDER = "temp"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# def clean_text(text: str) -> str:
#     """Clean extracted PDF text"""
#     text = text.replace("\n\n", "\n")
#     return text.strip()

# def extract_json(text: str):
#     """Cleanly extract and structure JSON from Mistral output"""
#     try:
#         # 1. Clean up "Markdown-isms" like ```json ... ```
#         text = text.replace("```json", "").replace("```", "").strip()
        
#         # 2. Extract content between first { and last }
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if match:
#             clean_str = match.group()
#             # Fix illegal backslashes often found in model outputs
#             clean_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', clean_str)
#             data = json.loads(clean_str)

#             # 3. Post-Process 'highlights' to remove double-quotes if they exist
#             if "highlights" in data and isinstance(data["highlights"], list):
#                 # Flatten the list and strip inner escaped quotes
#                 cleaned_highlights = []
#                 for item in data["highlights"]:
#                     # Split if the model put everything in one string separated by commas
#                     parts = item.split('", "') if '", "' in item else [item]
#                     for p in parts:
#                         cleaned_highlights.append(p.replace('"', '').strip())
#                 data["highlights"] = cleaned_highlights
            
#             return data
#     except Exception as e:
#         print(f"Extraction Error: {e}")

#     return {"overview": "", "summary": "", "highlights": []}




# @app.post("/analyze")
# async def analyze_pdf(
#     file: UploadFile = File(...),
#     analysis_type: int = Query(
#         0,
#         ge=0,
#         le=3,
#         description="0=all, 1=overview, 2=summary, 3=highlights"
#     )
# ):
#     """
#     Analyze uploaded PDF and return:
#     - overview
#     - summary
#     - highlights
#     """

#     # Save uploaded file
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)

#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     # Extract text from PDF
#     text = extract_text_from_pdf(file_path)
#     text = clean_text(text)

#     # Split into chunks (for large PDFs)
#     chunks = chunk_text(text)

#     results = []

#     for chunk in chunks:
#         ai_response = generate_analysis(chunk)
#         parsed = extract_json(ai_response)
#         results.append(parsed)

#     # Merge results from all chunks
#     combined_overview = " ".join(
#         [r.get("overview", "") for r in results]
#     ).strip()

#     combined_summary = " ".join(
#         [r.get("summary", "") for r in results]
#     ).strip()

#     combined_highlights = list(set(
#         sum([r.get("highlights", []) for r in results], [])
#     ))

#     final_output = {
#         "overview": combined_overview,
#         "summary": combined_summary,
#         "highlights": combined_highlights
#     }

#     # Return selected section if requested
#     if analysis_type == 1:
#         return {"overview": final_output["overview"]}
#     elif analysis_type == 2:
#         return {"summary": final_output["summary"]}
#     elif analysis_type == 3:
#         return {"highlights": final_output["highlights"]}

#     return final_output




# modified code 


# from fastapi import FastAPI, UploadFile, File, Query, HTTPException
# import os
# import uuid
# import json
# import re
# import logging
# from pdf_utils import extract_text_from_pdf, chunk_text
# from ai_model import generate_analysis

# logger = logging.getLogger(__name__)

# app = FastAPI()

# UPLOAD_FOLDER = "temp"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# def clean_text(text: str) -> str:
#     """Clean extracted PDF text"""
#     text = text.replace("\n\n", "\n")
#     return text.strip()

# def extract_json(text: str):
#     """Cleanly extract and structure JSON from Mistral output"""
#     try:
#         # 1. Clean up "Markdown-isms" like ```json ... ```
#         text = text.replace("```json", "").replace("```", "").strip()
        
#         # 2. Extract content between first { and last }
#         match = re.search(r"\{.*\}", text, re.DOTALL)
#         if match:
#             clean_str = match.group()
#             # Fix illegal backslashes often found in model outputs
#             clean_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', clean_str)
#             data = json.loads(clean_str)

#             # 3. Post-Process 'highlights' to remove double-quotes if they exist
#             if "highlights" in data and isinstance(data["highlights"], list):
#                 # Flatten the list and strip inner escaped quotes
#                 cleaned_highlights = []
#                 for item in data["highlights"]:
#                     # Split if the model put everything in one string separated by commas
#                     parts = item.split('", "') if '", "' in item else [item]
#                     for p in parts:
#                         cleaned_highlights.append(p.replace('"', '').strip())
#                 data["highlights"] = cleaned_highlights
            
#             return data
#     except Exception as e:
#         logger.error(f"Extraction Error: {e}")

#     return {"overview": "", "summary": "", "highlights": []}


# @app.post("/analyze")
# async def analyze_pdf(
#     file: UploadFile = File(...),
#     analysis_type: int = Query(
#         0,
#         ge=0,
#         le=3,
#         description="0=all, 1=overview, 2=summary, 3=highlights"
#     )
# ):
#     """
#     Analyze uploaded PDF and return:
#     - overview
#     - summary
#     - highlights
#     """

#     # Validate file type
#     if not file.filename or not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

#     # Use a random UUID filename to prevent path traversal attacks
#     safe_name = f"{uuid.uuid4()}.pdf"
#     file_path = os.path.join(UPLOAD_FOLDER, safe_name)

#     try:
#         # Save uploaded file
#         with open(file_path, "wb") as f:
#             f.write(await file.read())

#         # Extract text from PDF
#         text = extract_text_from_pdf(file_path)
#         text = clean_text(text)

#         # Reject image-only or empty PDFs early
#         if not text:
#             raise HTTPException(
#                 status_code=422,
#                 detail="No extractable text found in PDF. The file may be image-only or empty."
#             )

#         # Split into chunks (for large PDFs)
#         chunks = chunk_text(text)

#         results = []

#         for chunk in chunks:
#             ai_response = await generate_analysis(chunk)
#             parsed = extract_json(ai_response)
#             results.append(parsed)

#     finally:
#         # Always clean up the uploaded file from disk
#         if os.path.exists(file_path):
#             os.remove(file_path)

#     # Merge results from all chunks
#     combined_overview = " ".join(
#         [r.get("overview", "") for r in results]
#     ).strip()

#     combined_summary = " ".join(
#         [r.get("summary", "") for r in results]
#     ).strip()

#     combined_highlights = list(set(
#         sum([r.get("highlights", []) for r in results], [])
#     ))

#     final_output = {
#         "overview": combined_overview,
#         "summary": combined_summary,
#         "highlights": combined_highlights
#     }

#     # Return selected section if requested
#     if analysis_type == 1:
#         return {"overview": final_output["overview"]}
#     elif analysis_type == 2:
#         return {"summary": final_output["summary"]}
#     elif analysis_type == 3:
#         return {"highlights": final_output["highlights"]}

#     return final_output




from fastapi import FastAPI, UploadFile, File, Query, HTTPException
import os
import uuid
import json
import re
import logging
from pdf_utils import extract_text_from_pdf, chunk_text
from ai_model import generate_analysis

logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def clean_text(text: str) -> str:
    """Clean extracted PDF text"""
    text = text.replace("\n\n", "\n")
    return text.strip()

def _fix_json_string(raw: str) -> str:
    """
    Apply a sequence of targeted repairs to common Mistral JSON output problems,
    without altering the structure or values of well-formed output.
    """
    # Remove markdown code fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Normalise Windows line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Fix illegal (non-JSON) backslash escapes, e.g. \' \, \: \. etc.
    # Keep valid escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)

    # Remove ASCII control characters (0x00-0x1F) except \n \r \t which are valid in JSON strings
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)

    return raw


def _extract_field_by_regex(text: str, field: str) -> str:
    """
    Fallback: extract a single string field value directly with regex
    when the whole JSON block can't be parsed.
    """
    pattern = rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_highlights_by_regex(text: str) -> list:
    """
    Fallback: extract highlights array items directly with regex.
    """
    match = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []
    inner = match.group(1)
    items = re.findall(r'"(.*?)"(?=\s*[,\]])', inner, re.DOTALL)
    return [i.strip() for i in items if i.strip()]


def _postprocess_highlights(data: dict) -> dict:
    """
    Post-process highlights list: flatten items the model packed into a single
    comma-separated string, and strip stray quote characters.
    """
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            parts = item.split('", "') if '", "' in item else [item]
            for p in parts:
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
    return data


def extract_json(text: str) -> dict:
    """
    Robustly extract and structure JSON from Mistral output using three
    fallback strategies so a single malformed character never silently
    returns an empty result.

    Strategy 1 — direct parse after cleaning
    Strategy 2 — isolate outermost { ... } block, then parse; on failure
                 attempt a conservative in-string quote-escape repair
    Strategy 3 — per-field regex extraction as last resort
    """
    empty = {"overview": "", "summary": "", "highlights": []}

    if not text or not text.strip():
        return empty

    cleaned = _fix_json_string(text)

    # --- Strategy 1: parse the whole cleaned string directly ---
    try:
        data = json.loads(cleaned)
        return _postprocess_highlights(data)
    except json.JSONDecodeError:
        pass

    # --- Strategy 2: isolate the outermost { ... } block ---
    brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            data = json.loads(candidate)
            return _postprocess_highlights(data)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON block parse failed ({e}), trying character-level repair")

            # Sub-strategy: attempt to repair unescaped double-quotes inside string values.
            # Replace any " that is NOT preceded by \ and NOT a structural delimiter
            # (i.e. not after : [ , { or before ] } ,) with \".
            # This is intentionally conservative — only fixes the most common model error.
            repaired = re.sub(
                r'(?<=[^\\])"(?=[^,\]}\n:}{\[])',
                r'\\"',
                candidate
            )
            try:
                data = json.loads(repaired)
                return _postprocess_highlights(data)
            except json.JSONDecodeError:
                pass

    # --- Strategy 3: regex field extraction (last resort) ---
    logger.error("Extraction Error: all JSON parse strategies failed, falling back to regex extraction")
    overview = _extract_field_by_regex(cleaned, "overview")
    summary = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)

    if overview or summary or highlights:
        return {"overview": overview, "summary": summary, "highlights": highlights}

    return empty


@app.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    analysis_type: int = Query(
        0,
        ge=0,
        le=3,
        description="0=all, 1=overview, 2=summary, 3=highlights"
    )
):
    """
    Analyze uploaded PDF and return:
    - overview
    - summary
    - highlights
    """

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Use a random UUID filename to prevent path traversal attacks
    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        text = clean_text(text)

        # Reject image-only or empty PDFs early
        if not text:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in PDF. The file may be image-only or empty."
            )

        # Split into chunks (for large PDFs)
        chunks = chunk_text(text)

        results = []

        for chunk in chunks:
            ai_response = await generate_analysis(chunk)
            parsed = extract_json(ai_response)
            results.append(parsed)

    finally:
        # Always clean up the uploaded file from disk
        if os.path.exists(file_path):
            os.remove(file_path)

    # Merge results from all chunks
    combined_overview = " ".join(
        [r.get("overview", "") for r in results]
    ).strip()

    combined_summary = " ".join(
        [r.get("summary", "") for r in results]
    ).strip()

    combined_highlights = list(set(
        sum([r.get("highlights", []) for r in results], [])
    ))

    final_output = {
        "overview": combined_overview,
        "summary": combined_summary,
        "highlights": combined_highlights
    }

    # Return selected section if requested
    if analysis_type == 1:
        return {"overview": final_output["overview"]}
    elif analysis_type == 2:
        return {"summary": final_output["summary"]}
    elif analysis_type == 3:
        return {"highlights": final_output["highlights"]}

    return final_output
```

---
File: backupfiles/t_utils.py
---

```py
# t_utils.py
import json
import re
import logging

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> dict:
    """
    Extracts a JSON object from a string.
    It looks for the first '{' and the last '}' to bound the JSON content.
    """
    logger.debug("Attempting to extract JSON from text.")

    # Find the start and end of the JSON block
    start_index = text.find('{')
    end_index = text.rfind('}')

    if start_index == -1 or end_index == -1 or end_index < start_index:
        logger.warning("Could not find a JSON block in the text.")
        return {}

    json_str = text[start_index : end_index + 1]

    # Clean up the string
    json_str = json_str.strip()
    # It might be wrapped in markdown
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON: {e}")
        logger.debug(f"Problematic JSON string: {json_str}")
        # As a fallback, try to be more robust with escapes
        json_str_repaired = re.sub(r'\(?!["\/bfnrtu])', r'', json_str)
        try:
            return json.loads(json_str_repaired)
        except json.JSONDecodeError as e2:
            logger.error(f"Failed to decode JSON even after repair: {e2}")
            return {}

```

---
File: backupfiles/t_main.py
---

```py
"""
t_main.py
=========
FastAPI application for PDF summarization — supports small, large,
text-based, image-only, scanned, and mixed PDFs.

Architecture
------------
POST /analyze          → streams file to disk, returns job_id (202 Accepted)
GET  /status/{job_id}  → poll progress  {stage, current, total}
GET  /result/{job_id}  → retrieve final JSON when done

Pipeline (3-level hierarchical summarization)
---------------------------------------------
Level 0  extract_text_from_pdf()     parallel batch OCR + native text
Level 1  generate_analysis()         per-chunk AI analysis (GPU-semaphore-gated)
Level 2  synthesize_section()        every SECTION_SIZE chunks → 1 section summary
Level 3  synthesize_final()          all section summaries → 1 final output

Edge cases handled
------------------
A) Image-only PDF, OCR unavailable
   → clear "failed" status with install instructions for PaddleOCR-VL

B) PDF with < MIN_TEXT_CHARS total text after extraction
   → clear "failed" status with exact character count

C) Corrupt / encrypted PDF
   → ValueError from extract_text_from_pdf → "failed" with reason

All fixes from original review retained
----------------------------------------
- Streaming upload  (no full-file RAM read)
- UUID temp filename  (no path traversal)
- clean_text imported from t_pdf_utils  (no local duplicate)
- load_model() called once at startup via lifespan
- asyncio.get_running_loop()  (not deprecated get_event_loop)
- GPU semaphore + retry inside t_ai_model
- Highlight deduplication + cap at MAX_HIGHLIGHTS
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import fitz
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from t_key_clause_extraction import  classify_document, DOCUMENT_HANDLERS


from backupfiles.t_ai_model import (
    generate_analysis,
    load_model,
    synthesize_final,
    synthesize_section,
)
from backupfiles.t_pdf_utils import chunk_text, clean_text, extract_text_from_pdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER    = "temp"
UPLOAD_CHUNK_MB  = 1                        # streaming read size
MAX_UPLOAD_BYTES = 500 * 1024 * 1024        # 500 MB hard upload limit
SECTION_SIZE     = 20                       # chunks per Level-2 section
MAX_HIGHLIGHTS   = 10                       # cap on final highlight list
MIN_TEXT_CHARS   = 20                       # minimum useful extracted text

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory job store
# Replace with Redis + a persistent DB for multi-worker deployments.
# ---------------------------------------------------------------------------
JobStatus = Literal[
    "pending", "extracting", "analyzing",
    "synthesizing", "done", "failed",
]

_jobs: dict[str, dict] = {}


def _new_job() -> str:
    jid        = str(uuid.uuid4())
    _jobs[jid] = {
        "status":   "pending",
        "progress": {"stage": "queued", "current": 0, "total": 0},
        "result":   None,
        "error":    None,
    }
    return jid


def _set_progress(
    jid: str,
    stage: str,
    current: int = 0,
    total: int   = 0,
) -> None:
    if jid not in _jobs:
        return
    _jobs[jid]["progress"] = {"stage": stage, "current": current, "total": total}
    _jobs[jid]["status"]   = (
        "extracting"   if stage == "extraction"                          else
        "analyzing"    if stage == "chunk_analysis"                      else
        "synthesizing" if stage in ("section_synthesis", "final_synthesis") else
        _jobs[jid]["status"]
    )


def _fail(jid: str, message: str) -> None:
    _jobs[jid]["status"] = "failed"
    _jobs[jid]["error"]  = message
    logger.error(f"[{jid}] FAILED: {message}")


# ---------------------------------------------------------------------------
# Application lifespan  (model loaded once at startup)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="PDF Summarizer — Large Document Edition",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# JSON extraction  (3-strategy parser)
# ---------------------------------------------------------------------------

def _fix_json_string(raw: str) -> str:
    """Clean common Mistral output issues before JSON parsing."""
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Fix illegal backslash escapes (keep valid: \" \\ \/ \b \f \n \r \t \uXXXX)
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    # Strip ASCII control characters except \n \r \t
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    return raw


def _extract_field_by_regex(text: str, field: str) -> str:
    m = re.search(rf'"{field}"\s*:\s*"(.*?)"(?=\s*[,}}])', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_highlights_by_regex(text: str) -> list[str]:
    m = re.search(r'"highlights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not m:
        return []
    items = re.findall(r'"(.*?)"(?=\s*[,\]])', m.group(1), re.DOTALL)
    return [i.strip() for i in items if i.strip()]


def _postprocess_highlights(data: dict) -> dict:
    """Flatten comma-packed highlight items and strip stray quote characters."""
    if "highlights" in data and isinstance(data["highlights"], list):
        cleaned = []
        for item in data["highlights"]:
            parts = item.split('", "') if '", "' in item else [item]
            for p in parts:
                cleaned.append(p.replace('"', '').strip())
        data["highlights"] = [h for h in cleaned if h]
    return data


def extract_json(text: str) -> dict:
    """
    Parse JSON from raw model output using three fallback strategies:

    Strategy 1 — direct parse after cleaning
    Strategy 2 — isolate outermost { … } block, then parse;
                 on failure attempt a conservative quote-escape repair
    Strategy 3 — per-field regex extraction (last resort)
    """
    empty = {"overview": "", "summary": "", "highlights": []}

    if not text or not text.strip():
        return empty

    cleaned = _fix_json_string(text)

    # Strategy 1
    try:
        return _postprocess_highlights(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # Strategy 2
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        candidate = m.group()
        try:
            return _postprocess_highlights(json.loads(candidate))
        except json.JSONDecodeError as exc:
            logger.warning(f"JSON block parse failed ({exc}); trying repair")
            repaired = re.sub(
                r'(?<=[^\\])"(?=[^,\]}\n:}{\[])', r'\\"', candidate
            )
            try:
                return _postprocess_highlights(json.loads(repaired))
            except json.JSONDecodeError:
                pass

    # Strategy 3
    logger.error("All JSON parse strategies failed — falling back to regex")
    overview   = _extract_field_by_regex(cleaned, "overview")
    summary    = _extract_field_by_regex(cleaned, "summary")
    highlights = _extract_highlights_by_regex(cleaned)
    if overview or summary or highlights:
        return {"overview": overview, "summary": summary, "highlights": highlights}

    return empty


# ---------------------------------------------------------------------------
# Highlight deduplication
# ---------------------------------------------------------------------------

def _dedup_highlights(
    highlights: list[str],
    limit: int = MAX_HIGHLIGHTS,
) -> list[str]:
    """
    Normalise → deduplicate → cap.

    Normalisation: lowercase, strip punctuation, collapse whitespace.
    This catches near-duplicates like "Revenue was $2M" and "Revenue: $2M".
    """
    seen:   set[str]  = set()
    unique: list[str] = []
    for h in highlights:
        key = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', h.lower())).strip()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique[:limit]


# ---------------------------------------------------------------------------
# Hierarchical summarization pipeline
# ---------------------------------------------------------------------------

async def _run_pipeline(
    jid:           str,
    file_path:     str,
    analysis_type: int,
) -> None:
    """
    Full async pipeline — runs as a FastAPI BackgroundTask.

    Stage 0  Text extraction      (parallel page batches, adaptive OCR)
    Stage 1  Chunk analysis       (Level 0 — per chunk, GPU-semaphore-gated)
    Stage 2  Section synthesis    (Level 1 — every SECTION_SIZE chunks)
    Stage 3  Final synthesis      (Level 2 — all section summaries → 1 output)
    """
    try:

        # ── Stage 0: extraction ─────────────────────────────────────────────
        _set_progress(jid, "extraction", 0, 1)
        logger.info(f"[{jid}] Extraction started")

        try:
            text = await extract_text_from_pdf(file_path)
        except RuntimeError as exc:
            # Edge case A: image-only PDF or blank/corrupt PDF
            _fail(jid, str(exc))
            return
        except ValueError as exc:
            # Corrupt / encrypted file
            _fail(jid, f"Could not open PDF: {exc}")
            return

        text = clean_text(text)

        # Edge case B: total extracted text is too short to summarise
        if len(text) < MIN_TEXT_CHARS:
            _fail(
                jid,
                f"Extracted text is too short to summarise "
                f"({len(text)} characters after cleaning). "
                f"The PDF may contain only images, decorative elements, "
                f"or a single very short line of text. "
                f"Minimum required: {MIN_TEXT_CHARS} characters.",
            )
            return

        logger.info(f"[{jid}] Extracted {len(text):,} characters")
        _set_progress(jid, "extraction", 1, 1)

        # ── Stage 1: chunk analysis — Level 0 ───────────────────────────────
        chunks       = chunk_text(text)
        total_chunks = len(chunks)

        # Belt-and-suspenders: chunk_text should never return [] after the
        # MIN_TEXT_CHARS guard above, but handle it cleanly just in case.
        if not chunks:
            _fail(jid, "Text could not be split into processable chunks.")
            return

        logger.info(f"[{jid}] Analysing {total_chunks} chunks")
        _set_progress(jid, "chunk_analysis", 0, total_chunks)

        async def _analyse_chunk(idx: int, chunk: str) -> dict:
            raw    = await generate_analysis(chunk)
            parsed = extract_json(raw)
            _set_progress(jid, "chunk_analysis", idx + 1, total_chunks)
            logger.info(f"[{jid}] Chunk {idx+1}/{total_chunks} done")
            return parsed

        # All chunks run concurrently; GPU semaphore inside t_ai_model
        # limits actual simultaneous model.generate() calls.
        chunk_results: list[dict] = list(
            await asyncio.gather(
                *[_analyse_chunk(i, c) for i, c in enumerate(chunks)]
            )
        )

        # ── Stage 2: section synthesis — Level 1 ────────────────────────────
        sections       = [
            chunk_results[i : i + SECTION_SIZE]
            for i in range(0, total_chunks, SECTION_SIZE)
        ]
        total_sections = len(sections)
        logger.info(f"[{jid}] Synthesizing {total_sections} sections")
        _set_progress(jid, "section_synthesis", 0, total_sections)

        async def _synthesize_sec(idx: int, sec: list[dict]) -> dict:
            label  = f"section {idx+1} of {total_sections}"
            raw    = await synthesize_section(sec, label)
            parsed = extract_json(raw)
            _set_progress(jid, "section_synthesis", idx + 1, total_sections)
            logger.info(f"[{jid}] Section {idx+1}/{total_sections} done")
            return parsed

        section_results: list[dict] = list(
            await asyncio.gather(
                *[_synthesize_sec(i, s) for i, s in enumerate(sections)]
            )
        )

        # ── Stage 3: final synthesis — Level 2 ──────────────────────────────
        _set_progress(jid, "final_synthesis", 0, 1)
        logger.info(f"[{jid}] Final synthesis across {total_sections} sections")

        # Detect page count for the final prompt (best-effort)
        try:
            with fitz.open(file_path) as doc:
                total_pages = len(doc)
        except Exception:
            total_pages = total_chunks   # reasonable fallback

        final_raw    = await synthesize_final(section_results, total_pages)
        final_parsed = extract_json(final_raw)

        # Deduplicate and cap highlights
        final_parsed["highlights"] = _dedup_highlights(
            final_parsed.get("highlights", [])
        )

        # Filter output by analysis_type
        if analysis_type == 1:
            result = {"overview":   final_parsed.get("overview",   "")}
        elif analysis_type == 2:
            result = {"summary":    final_parsed.get("summary",    "")}
        elif analysis_type == 3:
            result = {"highlights": final_parsed.get("highlights", [])}
        else:
            result = final_parsed

        _jobs[jid]["status"] = "done"
        _jobs[jid]["result"] = result
        _set_progress(jid, "final_synthesis", 1, 1)
        logger.info(f"[{jid}] Pipeline complete ✓")

    except Exception as exc:
        logger.exception(f"[{jid}] Unexpected pipeline error")
        _fail(jid, f"Unexpected error: {exc}")

    finally:
        # Always delete the temp file, even on failure
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[{jid}] Temp file deleted")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze", status_code=202)
async def analyze_pdf(
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    analysis_type:    int        = Query(
        0, ge=0, le=3,
        description=(
            "What to return: "
            "0=all fields  "
            "1=overview only  "
            "2=summary only  "
            "3=highlights only"
        ),
    ),
):
    """
    Upload a PDF and start async analysis.

    The file is streamed to disk chunk-by-chunk — no full-file RAM spike.
    Returns HTTP 202 with a job_id immediately.

    Poll GET /status/{job_id} to track progress.
    Retrieve the result with GET /result/{job_id}.

    Supports text-based, scanned, image-only, and mixed PDFs up to 500 MB.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name  = f"{uuid.uuid4()}.pdf"
    file_path  = os.path.join(UPLOAD_FOLDER, safe_name)
    total_read = 0
    chunk_size = UPLOAD_CHUNK_MB * 1024 * 1024

    try:
        with open(file_path, "wb") as out:
            while True:
                data = await file.read(chunk_size)
                if not data:
                    break
                total_read += len(data)
                if total_read > MAX_UPLOAD_BYTES:
                    out.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                        ),
                    )
                out.write(data)
    except HTTPException:
        raise
    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    jid = _new_job()
    logger.info(
        f"[{jid}] Accepted {safe_name} "
        f"({total_read / 1024 / 1024:.1f} MB), "
        f"analysis_type={analysis_type}"
    )

    background_tasks.add_task(_run_pipeline, jid, file_path, analysis_type)

    return JSONResponse(
        status_code=202,
        content={
            "job_id":  jid,
            "message": "File accepted. Poll /status/{job_id} for progress.",
        },
    )


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Poll job progress.

    Response
    --------
    job_id    : str
    status    : pending | extracting | analyzing | synthesizing | done | failed
    progress  : {stage: str, current: int, total: int}
    error     : str  (only present when status == "failed")

    Common error messages
    ----------------------
    Image-only PDF, no OCR:
      "This PDF is entirely image-based … Install PaddleOCR-VL …"
    Too little text:
      "Extracted text is too short to summarise (N characters after cleaning)."
    Corrupt / encrypted:
      "Could not open PDF: …"
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    resp: dict = {
        "job_id":   job_id,
        "status":   job["status"],
        "progress": job["progress"],
    }
    if job["status"] == "failed":
        resp["error"] = job["error"]
    return resp


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """
    Retrieve the final analysis result.

    HTTP 200  — processing done; result JSON returned.
    HTTP 202  — still processing; check /status for progress.
    HTTP 404  — unknown job_id.
    HTTP 500  — job failed; detail contains the human-readable reason.

    Result JSON (analysis_type=0)
    ------------------------------
    {
      "overview":   "1-2 sentence document description",
      "summary":    "5-7 sentence executive summary",
      "highlights": ["fact 1", "fact 2", … up to 10]
    }
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=job.get("error", "Unknown error. Check /status for details."),
        )

    if job["status"] != "done":
        return JSONResponse(
            status_code=202,
            content={
                "job_id":   job_id,
                "status":   job["status"],
                "progress": job["progress"],
                "message":  "Processing in progress. Try again shortly.",
            },
        )

    return job["result"]

@app.post("/key-clause-extraction")
async def key_clause_extraction(file: UploadFile = File(...)):

    # ==============================
    # Step 0: Validate file
    # ==============================
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # ==============================
    # Step 1: Save file safely
    # ==============================
    safe_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        with open(file_path, "wb") as f:
            f.write(raw_bytes)

        del raw_bytes  # free memory

        # ==============================
        # Step 2: Extract text (FIXED)
        # ==============================
        text = extract_text_from_pdf(file_path)

        if not text:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found"
            )

        # Optional extra cleaning
        text = clean_text(text)

        # ==============================
        # Step 3: Classification
        # ==============================
        doc_type = await classify_document(text)
        doc_type = doc_type.lower().strip()

        # ==============================
        # Step 4: Route
        # ==============================
        handler = DOCUMENT_HANDLERS.get(doc_type)

        if handler:
            return await handler(text)

        # ==============================
        # Step 5: Fallback
        # ==============================
        return {
            "status": "unsupported",
            "document_type": doc_type,
            "message": "Unsupported document type."
        }

    finally:
        # ==============================
        # Step 6: Cleanup
        # ==============================
        if os.path.exists(file_path):
            os.remove(file_path)


```

---
File: backupfiles/ai_model_backup.py
---

```py
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Load GPU model (Mistral 7B)
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True

# def generate_analysis(text):
#     """Generate advanced, professional 4-section PDF analysis using LLM"""

    


#     prompt = f"""
#     You are a professional document analyst capable of reviewing contracts, reports, policies, financial documents, academic papers, and general business PDFs.

#     Analyze the document and return ONLY valid JSON.

#     STRICT RULES:
#     - Do NOT explain anything.
#     - Do NOT add extra text.
#     - Output must be valid JSON only.

#     Return EXACTLY this format:

#     {{
#     "document_type": "Type of document (e.g., Contract, Report, Invoice, Policy, Research Paper, Other)",
#     "summary": "Concise executive summary",
#     "key_highlights": [
#     "Important point 1",
#     "Important point 2",
#     "Important point 3",
#     "Important point 4"
#     ],
#     "risk_analysis": [
#     "Risk or concern 1 (if applicable)",
#     "Risk or concern 2",
#     "Risk or concern 3",
#     "Risk or concern 4"
#     ]
#     }}

#     Document Content:
#     ----------------
#     {text}
#     ----------------

#     JSON:
#     """


#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     output = model.generate(
#         **inputs,
#         max_new_tokens=500,   # increased for detailed JSON
#         temperature=0.3,       # low for deterministic output
#         do_sample=False
#     )

# #    result = tokenizer.decode(output[0], skip_special_tokens=True)
#     result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
#     return result


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


# def generate_analysis(text: str) -> str:

#     prompt = f"""<s>[INST]
# You are an expert document analyst with deep knowledge across legal, medical,
# financial, technical, academic, and business domains.

# Your task is to read any type of document and produce a structured JSON briefing
# that is accurate, professional, and immediately useful to the reader.

# EXAMPLE OUTPUT (follow this structure exactly):
# {{
#   "overview": "This is a [document type] about [main subject], intended for [audience/purpose].",
#   "summary": "4-6 sentence executive summary covering the document purpose, key content,
#                important findings or obligations, and conclusions or outcomes.",
#   "highlights": [
#     "Specific important fact, figure, date, obligation, finding, or condition from the document.",
#     "Specific important fact, figure, date, obligation, finding, or condition from the document.",
#     "Specific important fact, figure, date, obligation, finding, or condition from the document.",
#     "Specific important fact, figure, date, obligation, finding, or condition from the document."
#   ]
# }}

# FIELD RULES:

# "overview" — 1-2 sentences only.
#   - Identify what TYPE of document this is.
#     Examples of types: employment contract, research paper, medical report,
#     user manual, financial statement, privacy policy, invoice, academic thesis,
#     insurance policy, government notice, product specification, meeting minutes.
#   - State the subject, purpose, and intended audience or parties.

# "summary" — 4-6 sentences in professional, neutral tone.
#   - For CONTRACTS/LEGAL: cover parties, obligations, terms, penalties, duration.
#   - For MEDICAL/HEALTH: cover diagnosis, findings, recommendations, medications, follow-up.
#   - For FINANCIAL: cover revenue, expenses, profit/loss, forecasts, risks.
#   - For TECHNICAL/MANUAL: cover product purpose, key features, requirements, warnings.
#   - For RESEARCH/ACADEMIC: cover objective, methodology, key findings, conclusions.
#   - For REPORTS/BUSINESS: cover context, analysis, recommendations, outcomes.
#   - If document type is unclear, summarize the most important information a reader needs.

# "highlights" — 4 to 6 items.
#   - Each must be ONE complete, specific, factual sentence.
#   - Always include real values from the document: numbers, dates, names,
#     percentages, durations, prices, dosages, deadlines, versions, scores, or clauses.
#   - Do NOT write vague statements like "The document contains important information."
#   - Do NOT write opinions or analysis — only facts extracted directly from the document.
#   - Prioritize: critical obligations, risks, key figures, deadlines, warnings, or outcomes.

# STRICT OUTPUT RULES:
#   - Output ONLY the JSON object.
#   - Do NOT add any explanation, greeting, or commentary.
#   - Do NOT use markdown, code blocks, or backticks.
#   - Do NOT repeat these instructions in your output.
#   - All string values must be properly escaped.
#   - If a section has insufficient information, write "Not enough information available."

# Document:
# ----------------
# {text}
# ----------------
# [/INST]
# """

#     # Tokenize
#     encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
#     token_count = encoded["input_ids"].shape[1]

#     # Warn if truncation will occur
#     if token_count > 6000:
#         print(f"Warning: input is {token_count} tokens, truncating to 6000.")

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=6000
#     ).to("cuda")

#     # Generate
#     try:
#         output = model.generate(
#             **inputs,
#             max_new_tokens=600,
#             do_sample=False,
#             repetition_penalty=1.15,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     except torch.cuda.OutOfMemoryError:
#         torch.cuda.empty_cache()
#         raise RuntimeError("GPU out of memory — reduce chunk size or document length.")

#     # Decode only new tokens
#     result = tokenizer.decode(
#         output[0][inputs["input_ids"].shape[1]:],
#         skip_special_tokens=True
#     )

#     return result




# updated code 

import torch
import asyncio
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Use CUDA if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


def _run_inference(prompt: str) -> str:
    """Synchronous inference — runs in a thread pool to avoid blocking the event loop."""

    # Tokenize
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    token_count = encoded["input_ids"].shape[1]

    # Warn if truncation will occur
    if token_count > 6000:
        logger.warning(f"Input is {token_count} tokens, truncating to 6000.")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=6000
    ).to(device)

    # Generate
    try:
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("GPU out of memory — reduce chunk size or document length.")
    finally:
        # Free input tensors from GPU memory
        del inputs
        if device == "cuda":
            torch.cuda.empty_cache()

    # Decode only new tokens
    result = tokenizer.decode(
        output[0][encoded["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return result


async def generate_analysis(text: str) -> str:

    prompt = f"""<s>[INST]
You are an expert document analyst with deep knowledge across legal, medical,
financial, technical, academic, and business domains.

Your task is to read any type of document and produce a structured JSON briefing
that is accurate, professional, and immediately useful to the reader.

EXAMPLE OUTPUT (follow this structure exactly):
{{
  "overview": "This is a [document type] about [main subject], intended for [audience/purpose].",
  "summary": "4-6 sentence executive summary covering the document purpose, key content,
               important findings or obligations, and conclusions or outcomes.",
  "highlights": [
    "Specific important fact, figure, date, obligation, finding, or condition from the document.",
    "Specific important fact, figure, date, obligation, finding, or condition from the document.",
    "Specific important fact, figure, date, obligation, finding, or condition from the document.",
    "Specific important fact, figure, date, obligation, finding, or condition from the document."
  ]
}}

FIELD RULES:

"overview" — 1-2 sentences only.
  - Identify what TYPE of document this is.
    Examples of types: employment contract, research paper, medical report,
    user manual, financial statement, privacy policy, invoice, academic thesis,
    insurance policy, government notice, product specification, meeting minutes.
  - State the subject, purpose, and intended audience or parties.

"summary" — 4-6 sentences in professional, neutral tone.
  - For CONTRACTS/LEGAL: cover parties, obligations, terms, penalties, duration.
  - For MEDICAL/HEALTH: cover diagnosis, findings, recommendations, medications, follow-up.
  - For FINANCIAL: cover revenue, expenses, profit/loss, forecasts, risks.
  - For TECHNICAL/MANUAL: cover product purpose, key features, requirements, warnings.
  - For RESEARCH/ACADEMIC: cover objective, methodology, key findings, conclusions.
  - For REPORTS/BUSINESS: cover context, analysis, recommendations, outcomes.
  - If document type is unclear, summarize the most important information a reader needs.

"highlights" — 4 to 6 items.
  - Each must be ONE complete, specific, factual sentence.
  - Always include real values from the document: numbers, dates, names,
    percentages, durations, prices, dosages, deadlines, versions, scores, or clauses.
  - Do NOT write vague statements like "The document contains important information."
  - Do NOT write opinions or analysis — only facts extracted directly from the document.
  - Prioritize: critical obligations, risks, key figures, deadlines, warnings, or outcomes.

STRICT OUTPUT RULES:
  - Output ONLY the JSON object.
  - Do NOT add any explanation, greeting, or commentary.
  - Do NOT use markdown, code blocks, or backticks.
  - Do NOT repeat these instructions in your output.
  - All string values must be properly escaped.
  - If a section has insufficient information, write "Not enough information available."

Document:
----------------
{text}
----------------
[/INST]
"""

    # Run blocking inference in a thread pool to avoid blocking the async event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_inference, prompt)
    return result
```

---
File: backupfiles/t_pdf_utils.py
---

```py
"""
t_pdf_utils.py
==============
PDF text extraction — supports small PDFs, large PDFs (500+ pages),
text-based PDFs, image-only / scanned PDFs, and mixed PDFs.

Features
--------
- Parallel batch page extraction  (BATCH_SIZE pages concurrently)
- No hard page limit              (handles 500+ pages)
- Smart page classification       (native text vs OCR vs blank)
- Adaptive chunk sizing           (dense / normal / sparse content)
- Grayscale + RGBA → RGB          (all pixmap formats handled)
- Safe PaddleOCR result unpacking (str or dict res.res)
- GPU memory freed after every OCR call
- Clear RuntimeError for image-only PDFs when OCR is unavailable
- Native text preserved as fallback when OCR returns nothing

Edge cases handled
------------------
A) Image-only PDF with OCR unavailable
   → RuntimeError with actionable install message (caught by t_main.py)

B) Pages with < 50 native chars  (old hard threshold removed)
   → _classify_page() keeps short-but-real text (title pages, headers)
   → Sends noisy/symbol pages to OCR with native text as fallback
"""

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import fitz          # PyMuPDF
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE       = 20      # pages extracted concurrently per batch
OCR_DPI          = 150     # render resolution for scanned pages
MIN_NATIVE_CHARS = 10      # native chars below this → always try OCR
WORD_RATIO_MIN   = 0.40    # min fraction of tokens that look like real words

# Adaptive chunk sizing (characters)
CHUNK_DENSE      = 5_000   # dense legal / technical prose
CHUNK_NORMAL     = 10_000  # standard mixed content
CHUNK_SPARSE     = 15_000  # tables, indices, appendices

# Shared thread pool — one per process, sized to batch width
_PAGE_EXECUTOR   = ThreadPoolExecutor(max_workers=BATCH_SIZE)

# ---------------------------------------------------------------------------
# Optional PaddleOCR-VL  (graceful degradation if not installed)
# ---------------------------------------------------------------------------
try:
    import paddle
    from paddleocr import PaddleOCRVL
    _ocr_vl           = PaddleOCRVL("v1.5")
    _PADDLE_AVAILABLE  = True
    logger.info("PaddleOCR-VL loaded successfully.")
except Exception as _exc:
    logger.warning(f"PaddleOCR-VL not available ({_exc}). OCR disabled.")
    _ocr_vl           = None
    _PADDLE_AVAILABLE  = False


# ---------------------------------------------------------------------------
# Page classification
# ---------------------------------------------------------------------------

def _classify_page(native: str) -> str:
    """
    Decide how to process a page based on its native PyMuPDF text.

    Returns
    -------
    "native"     — enough real text; skip OCR entirely
    "ocr_needed" — insufficient or noisy native text; attempt OCR
                   (native text kept as fallback if OCR returns "")
    "blank"      — completely empty; attempt OCR (may be image page)

    Rules
    -----
    - "" or None                           → "blank"
    - < MIN_NATIVE_CHARS chars             → "ocr_needed"
    - >= MIN_NATIVE_CHARS, high word ratio → "native"
    - >= MIN_NATIVE_CHARS, low word ratio  → "ocr_needed"  (encoding noise)
    """
    stripped = (native or "").strip()

    if not stripped:
        return "blank"

    char_count = len(stripped)

    if char_count < MIN_NATIVE_CHARS:
        return "ocr_needed"

    tokens     = stripped.split()
    real_words = sum(1 for t in tokens if sum(c.isalpha() for c in t) >= 2)
    ratio      = real_words / max(len(tokens), 1)

    if ratio >= WORD_RATIO_MIN:
        return "native"

    return "ocr_needed"


# ---------------------------------------------------------------------------
# Low-level helpers (synchronous — run inside thread pool)
# ---------------------------------------------------------------------------

def _page_to_rgb(pix: fitz.Pixmap) -> np.ndarray:
    """Convert a PyMuPDF Pixmap to an HxWx3 uint8 RGB numpy array.
    Handles n=1 (grayscale), n=3 (RGB), and n=4 (RGBA)."""
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 1:       # grayscale → RGB
        img = np.repeat(img, 3, axis=2)
    elif pix.n == 4:     # RGBA → RGB
        img = img[:, :, :3]
    return img


def _run_ocr_sync(img: np.ndarray) -> str:
    """
    Run PaddleOCR-VL on a single RGB page image.

    Safely unpacks res.res whether it is a plain string or a dict
    (PaddleOCR-VL returns dicts like {"rec_text": "...", ...}).
    Always frees GPU memory before returning.
    """
    if _ocr_vl is None:
        return ""

    results = None
    try:
        results    = _ocr_vl.predict(img)
        page_parts = []

        for res in results:
            # Unpack result object
            if hasattr(res, "res"):
                raw = res.res
            elif isinstance(res, dict) and "res" in res:
                raw = res["res"]
            else:
                raw = str(res)

            # res.res can be a dict → extract the text field
            if isinstance(raw, dict):
                text = (
                    raw.get("rec_text")
                    or raw.get("text")
                    or " ".join(str(v) for v in raw.values() if isinstance(v, str))
                )
            else:
                text = str(raw)

            if text and text.strip():
                page_parts.append(text.strip())

        return "\n".join(page_parts)

    except Exception as exc:
        logger.error(f"OCR predict error: {exc}")
        return ""

    finally:
        # Always free GPU memory
        try:
            del results
        except Exception:
            pass
        if _PADDLE_AVAILABLE:
            try:
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()
            except Exception:
                pass


def _extract_page_sync(file_path: str, page_num: int) -> tuple[int, str]:
    """
    Extract text from ONE page. Runs inside a thread-pool worker.

    Each call opens its own fitz.Document handle so threads never
    share document state. The pixmap is deleted immediately after OCR
    to prevent RAM accumulation across hundreds of pages.

    Returns (page_num, text) so results can be sorted back into order.

    Strategy
    --------
    1. Native text extraction via PyMuPDF.
    2. _classify_page() decides whether native text is sufficient.
    3. If OCR is needed and available → OCR; keep native as fallback.
    4. If OCR is needed but unavailable → return whatever native text exists.
    """
    try:
        with fitz.open(file_path) as doc:
            page   = doc[page_num]
            native = page.get_text("text") or ""
            kind   = _classify_page(native)

            # ── Case 1: native text is good ──────────────────────────────────
            if kind == "native":
                logger.debug(f"Page {page_num+1}: native ({len(native.strip())} chars)")
                return page_num, native.strip()

            # ── Case 2 & 3: OCR needed ───────────────────────────────────────
            if not _PADDLE_AVAILABLE:
                fallback = native.strip()
                if fallback:
                    logger.warning(
                        f"Page {page_num+1}: OCR unavailable, "
                        f"keeping {len(fallback)}-char native text as fallback."
                    )
                else:
                    logger.warning(
                        f"Page {page_num+1}: image page and OCR unavailable — skipped."
                    )
                return page_num, fallback

            # Run OCR
            try:
                pix      = page.get_pixmap(dpi=OCR_DPI)
                img      = _page_to_rgb(pix)
                del pix                          # free pixmap RAM immediately
                ocr_text = _run_ocr_sync(img)

                if ocr_text:
                    logger.debug(
                        f"Page {page_num+1}: OCR → {len(ocr_text)} chars"
                    )
                    return page_num, ocr_text

                # OCR returned nothing → fall back to native text (never discard)
                fallback = native.strip()
                if fallback:
                    logger.warning(
                        f"Page {page_num+1}: OCR returned empty; "
                        f"keeping {len(fallback)}-char native fallback."
                    )
                else:
                    logger.warning(f"Page {page_num+1}: OCR empty, no native text.")
                return page_num, fallback

            except MemoryError:
                logger.error(
                    f"Page {page_num+1}: MemoryError during OCR — using native fallback."
                )
                return page_num, native.strip()
            except Exception as exc:
                logger.error(
                    f"Page {page_num+1}: OCR error ({exc}) — using native fallback."
                )
                return page_num, native.strip()

    except Exception as exc:
        logger.error(f"Page {page_num+1}: could not open document ({exc}).")
        return page_num, ""


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------

async def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF of any size using async parallel page batches.

    Algorithm
    ---------
    1. Open the PDF once to count pages, then close it.
    2. Divide all page numbers into batches of BATCH_SIZE.
    3. For each batch, run every page in _PAGE_EXECUTOR concurrently
       via asyncio.gather (non-blocking for the event loop).
    4. Sort (page_num, text) results, join, and clean.

    Edge cases
    ----------
    - All pages empty + OCR unavailable  → RuntimeError (actionable message)
    - All pages empty + OCR available    → RuntimeError (blank/corrupt PDF)
    - Some pages empty                   → warning logged, others used

    Raises
    ------
    ValueError   — file cannot be opened (corrupt / encrypted)
    RuntimeError — no text could be extracted from any page
    """
    try:
        with fitz.open(file_path) as probe:
            total_pages = len(probe)
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}")

    logger.info(
        f"Extracting {total_pages} pages — "
        f"{-(-total_pages // BATCH_SIZE)} batch(es) of {BATCH_SIZE}"
    )

    batches = [
        list(range(i, min(i + BATCH_SIZE, total_pages)))
        for i in range(0, total_pages, BATCH_SIZE)
    ]

    loop         = asyncio.get_running_loop()
    page_results: list[tuple[int, str]] = []

    for b_idx, batch in enumerate(batches):
        logger.info(
            f"Batch {b_idx+1}/{len(batches)}: "
            f"pages {batch[0]+1}–{batch[-1]+1}"
        )
        futures    = [
            loop.run_in_executor(_PAGE_EXECUTOR, _extract_page_sync, file_path, pn)
            for pn in batch
        ]
        batch_out  = await asyncio.gather(*futures)
        page_results.extend(batch_out)

    page_results.sort(key=lambda t: t[0])

    # ── Edge case A: detect image-only PDF ──────────────────────────────────
    empty_count = sum(1 for _, t in page_results if not t.strip())

    if empty_count == total_pages:
        if not _PADDLE_AVAILABLE:
            raise RuntimeError(
                "This PDF is entirely image-based (scanned or photo PDF) and "
                "contains no extractable native text. "
                "PaddleOCR-VL is not installed, so OCR cannot be performed. "
                "Install it with:  pip install paddlepaddle paddleocr  "
                "then restart the server."
            )
        else:
            raise RuntimeError(
                "No text could be extracted from this PDF even with OCR. "
                "The file may be blank, password-protected, or corrupt."
            )

    if empty_count > 0:
        logger.warning(
            f"{empty_count}/{total_pages} pages yielded no text "
            f"(image pages skipped or blank pages)."
        )

    joined = "\n".join(text for _, text in page_results if text.strip())
    return clean_text(joined)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean common PDF extraction artefacts:
      - Rejoin hyphenated line-breaks  (word-\nbreak → wordbreak)
      - Collapse 3+ blank lines to double newline
      - Collapse repeated spaces
    """
    text = re.sub(r"-\n",    "",      text)
    text = re.sub(r"\n{3,}", "\n\n",  text)
    text = re.sub(r" {2,}",  " ",     text)
    return text.strip()


def adaptive_chunk_size(text: str) -> int:
    """
    Choose a chunk character limit based on word density.

    Density  > 0.15 words/char  →  CHUNK_DENSE  (5 000)  — legal / technical prose
    Density  > 0.08 words/char  →  CHUNK_NORMAL (10 000) — standard mixed content
    Density ≤ 0.08 words/char  →  CHUNK_SPARSE (15 000) — tables / indices
    """
    if not text:
        return CHUNK_NORMAL
    density = len(text.split()) / max(len(text), 1)
    if density > 0.15:
        return CHUNK_DENSE
    elif density > 0.08:
        return CHUNK_NORMAL
    return CHUNK_SPARSE


def chunk_text(text: str, max_chars: int = 0) -> list[str]:
    """
    Split text into chunks on paragraph boundaries (never mid-sentence).

    Parameters
    ----------
    text      : cleaned extracted text
    max_chars : 0 = choose automatically via adaptive_chunk_size()
                >0 = use this fixed size

    Returns
    -------
    List of non-empty string chunks.
    """
    if not text:
        return []

    size       = max_chars if max_chars > 0 else adaptive_chunk_size(text)
    paragraphs = text.split("\n")
    chunks: list[str] = []
    current    = ""

    for para in paragraphs:
        if len(current) + len(para) > size:
            if current.strip():
                chunks.append(current.strip())
            current = para
        else:
            current += "\n" + para

    if current.strip():
        chunks.append(current.strip())

    logger.info(
        f"chunk_text: {len(text):,} chars → {len(chunks)} chunks "
        f"(target {size:,} chars each)"
    )
    return chunks
```

