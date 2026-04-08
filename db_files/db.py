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

-- Comparison Logs
CREATE TABLE IF NOT EXISTS comparison_logs (
    id                  SERIAL PRIMARY KEY,

    # Session & request identity
    session_id          UUID NOT NULL,
    request_id          UUID NOT NULL DEFAULT gen_random_uuid(),

    # -- Documents compared
    doc1_filename       TEXT NOT NULL,
    doc2_filename       TEXT NOT NULL,

    # -- Timing
    requested_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    duration_ms         INTEGER,

    # -- Outcome
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'success', 'failed')),
    error_message       TEXT,

    # -- Change summary (fast querying)
    total_changes       INTEGER,
    high_risk_changes   INTEGER,
    overall_risk_level  TEXT CHECK (overall_risk_level IN ('high', 'medium', 'low')),
    similarity_percent  TEXT,

    # -- Full result payload
    result_json         JSONB,

    # -- Token + performance tracking (ADDED for consistency)
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    total_tokens        INTEGER,

    # -- Client metadata
    client_ip           TEXT,
    user_agent          TEXT
);
CREATE INDEX IF NOT EXISTS idx_comparison_logs_session_id
    ON comparison_logs (session_id);

CREATE INDEX IF NOT EXISTS idx_comparison_logs_requested_at
    ON comparison_logs (requested_at DESC);

CREATE INDEX IF NOT EXISTS idx_comparison_logs_status
    ON comparison_logs (status);

CREATE INDEX IF NOT EXISTS idx_comparison_logs_risk_level
    ON comparison_logs (overall_risk_level);

CREATE INDEX IF NOT EXISTS idx_comparison_logs_request_id
    ON comparison_logs (request_id);
"""

async def init_db() -> None:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(CREATE_TABLES_SQL)
        logger.info("[db] Schema initialised (pdf_requests + document_requests + comparison_logs tables ready)")
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

# Log Comparison Request
async def log_comparison_request(
    session_id: str,
    request_id: str,
    doc1_filename: str,
    doc2_filename: str,
    status: str = "success",
    duration_ms: int = 0,
    total_changes: int = 0,
    high_risk_changes: int = 0,
    overall_risk_level: str = "low",
    similarity_percent: str = "",
    result_json: dict | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    error_message: str | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> None:
    """
    Logs a contract comparison request.
    Never raises — safe for production pipelines.
    """
    import json as _json

    total_tokens = input_tokens + output_tokens
    result_json_str = _json.dumps(result_json) if result_json else None

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO comparison_logs (
                    session_id, request_id,
                    doc1_filename, doc2_filename,
                    completed_at, duration_ms,
                    status, error_message,
                    total_changes, high_risk_changes,
                    overall_risk_level, similarity_percent,
                    result_json,
                    input_tokens, output_tokens, total_tokens,
                    client_ip, user_agent
                ) VALUES (
                    $1, $2,
                    $3, $4,
                    NOW(), $5,
                    $6, $7,
                    $8, $9,
                    $10, $11,
                    $12,
                    $13, $14, $15,
                    $16, $17
                )
                """,
                session_id,
                request_id,
                doc1_filename,
                doc2_filename,
                duration_ms,
                status,
                error_message,
                total_changes,
                high_risk_changes,
                overall_risk_level,
                similarity_percent,
                result_json_str,
                input_tokens,
                output_tokens,
                total_tokens,
                client_ip,
                user_agent,
            )

        logger.info(
            f"[db] comparison logged — session={session_id} "
            f"docs=({doc1_filename}, {doc2_filename}) "
            f"changes={total_changes} risk={overall_risk_level} "
            f"time={duration_ms}ms status={status}"
        )

    except Exception as e:
        logger.error(f"[db] log_comparison_request failed: {e}")

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
    
async def get_comparison_stats() -> dict:
    """
    Returns aggregate stats for comparison logs.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:

            # 🔹 Overall stats
            overall_row = await conn.fetchrow("""
                SELECT
                    COUNT(*)                                  AS total_requests,
                    COUNT(*) FILTER (WHERE status = 'success') AS success_count,
                    COUNT(*) FILTER (WHERE status = 'failed')  AS failed_count,
                    ROUND(AVG(duration_ms))                    AS avg_duration_ms,
                    ROUND(AVG(total_changes))                  AS avg_changes,
                    SUM(total_changes)                         AS total_changes,
                    SUM(high_risk_changes)                     AS total_high_risk_changes,
                    SUM(total_tokens)                          AS total_tokens
                FROM comparison_logs
            """)

            # 🔹 Risk distribution
            risk_rows = await conn.fetch("""
                SELECT
                    overall_risk_level,
                    COUNT(*) AS count
                FROM comparison_logs
                WHERE overall_risk_level IS NOT NULL
                GROUP BY overall_risk_level
                ORDER BY count DESC
            """)

            # 🔹 Recent activity (last 24h)
            recent_row = await conn.fetchrow("""
                SELECT
                    COUNT(*) AS requests_last_24h
                FROM comparison_logs
                WHERE requested_at >= NOW() - INTERVAL '24 hours'
            """)

            # 🔹 Avg similarity
            similarity_row = await conn.fetchrow("""
                SELECT
                    ROUND(AVG(
                        NULLIF(REPLACE(similarity_percent, '%', ''), '')::numeric
                    ), 2) AS avg_similarity_percent
                FROM comparison_logs
                WHERE similarity_percent IS NOT NULL
            """)

        return {
            "overall": dict(overall_row) if overall_row else {},
            "risk_distribution": [dict(r) for r in risk_rows],
            "recent": dict(recent_row) if recent_row else {},
            "similarity": dict(similarity_row) if similarity_row else {},
        }

    except Exception as e:
        logger.error(f"[db] get_comparison_stats failed: {e}")
        return {
            "overall": {},
            "risk_distribution": [],
            "recent": {},
            "similarity": {}
        }

async def get_recent_comparisons(limit: int = 20) -> list[dict]:
    """
    Returns recent comparison requests.
    Ordered by most recent first.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    session_id,
                    request_id,
                    doc1_filename,
                    doc2_filename,
                    status,
                    total_changes,
                    high_risk_changes,
                    overall_risk_level,
                    similarity_percent,
                    duration_ms,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    requested_at,
                    completed_at
                FROM comparison_logs
                ORDER BY requested_at DESC
                LIMIT $1
            """, limit)

        return [dict(r) for r in rows]

    except Exception as e:
        logger.error(f"[db] get_recent_comparisons failed: {e}")
        return []