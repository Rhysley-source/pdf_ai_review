"""
In-memory session store for caching extracted PDF text.

Sessions are created by POST /analyze and consumed by
POST /key-clause-extraction and POST /detect-risks to skip
re-extraction / re-OCR on the same document.

TTL: SESSION_TTL_SECONDS (default 1 hour).
Cleanup is lazy (on access) — no background task required.
"""

import time
import uuid
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS: int = 3600  # 1 hour

# session_id → {text, filename, total_pages, pages_analysed, created_at}
_store: dict[str, dict] = {}


def create_session(
    text: str,
    filename: str,
    total_pages: int,
    pages_analysed: int,
) -> str:
    """
    Store extracted PDF text under a new UUID session ID.
    Returns the session_id string.
    """
    session_id = str(uuid.uuid4())
    _store[session_id] = {
        "text":           text,
        "filename":       filename,
        "total_pages":    total_pages,
        "pages_analysed": pages_analysed,
        "created_at":     time.time(),
    }
    logger.info(
        f"[session_store] created session={session_id[:8]} "
        f"filename='{filename}' pages={pages_analysed}/{total_pages} "
        f"text_len={len(text):,} chars ttl={SESSION_TTL_SECONDS}s"
    )
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """
    Retrieve a session by ID.
    Returns None if the session does not exist or has expired.
    Expired entries are deleted on access (lazy cleanup).
    """
    entry = _store.get(session_id)
    if entry is None:
        return None
    if time.time() - entry["created_at"] > SESSION_TTL_SECONDS:
        del _store[session_id]
        logger.info(f"[session_store] session={session_id[:8]} expired — removed")
        return None
    return entry


def cleanup_expired() -> int:
    """
    Delete all expired sessions.
    Returns the count of sessions removed.
    Suitable for periodic calls from a lifespan background task if needed.
    """
    now     = time.time()
    expired = [
        sid for sid, entry in _store.items()
        if now - entry["created_at"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _store[sid]
    if expired:
        logger.info(f"[session_store] cleanup removed {len(expired)} expired session(s)")
    return len(expired)
