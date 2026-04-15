"""
rag_chat.py

RAG-based document chat using FAISS vector store + LangGraph pipeline.

Flow:
  1. build_index(text)  → embed all chunks → FAISS IndexFlatIP → returns session_id
  2. chat(session_id, question)
       → embed question
       → LangGraph: [retrieve_node] → [generate_node]
       → retrieve_node: FAISS top-k search → builds context string
       → generate_node: OpenAI chat with system context + history
       → persists turn in session history

Session store is in-process memory (dict). Each worker process holds its own
sessions — suitable for single-process deployments (uvicorn --workers 1) or
behind a sticky-session load balancer.
"""

import logging
import time
import uuid
import os
from typing import TypedDict

import faiss
import numpy as np
from openai import AsyncOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_EMBED_MODEL    = "text-embedding-3-small"
_EMBED_DIM      = 1536
_TOP_K          = 5
_CHUNK_SIZE     = 800
_CHUNK_OVERLAP  = 150
_MAX_HISTORY    = 6      # turns kept in context (each turn = 2 messages)

_embed_client = AsyncOpenAI(api_key=_OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# In-memory session store
# key: session_id (str uuid4)
# value: {"index": faiss.Index, "chunks": list[str], "history": list[dict]}
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Text chunking — character-level windows (fast, no tokeniser dependency)
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + _CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# OpenAI Embeddings
# ---------------------------------------------------------------------------

async def _embed_texts(texts: list[str]) -> list[list[float]]:
    response = await _embed_client.embeddings.create(model=_EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


async def _embed_query(query: str) -> list[float]:
    response = await _embed_client.embeddings.create(model=_EMBED_MODEL, input=[query])
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Build FAISS index from full document text
# ---------------------------------------------------------------------------

async def build_index(text: str) -> str:
    """
    Chunk document text, embed all chunks using OpenAI text-embedding-3-small,
    build a FAISS IndexFlatIP (cosine similarity via L2-normalised inner product),
    and store the session in memory.

    Returns the new session_id.
    """
    session_id = str(uuid.uuid4())
    t0         = time.perf_counter()

    chunks = _chunk_text(text)
    logger.info(f"[rag] [{session_id[:8]}] {len(chunks)} chunk(s) — embedding...")

    all_embeddings: list[list[float]] = []
    for i in range(0, len(chunks), 100):       # OpenAI allows up to 2048 inputs per call
        batch = await _embed_texts(chunks[i : i + 100])
        all_embeddings.extend(batch)

    vectors = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)                # normalise → inner product == cosine similarity

    index = faiss.IndexFlatIP(_EMBED_DIM)
    index.add(vectors)

    _sessions[session_id] = {
        "index":   index,
        "chunks":  chunks,
        "history": [],
    }

    logger.info(
        f"[rag] [{session_id[:8]}] index ready — "
        f"{len(chunks)} chunks | {time.perf_counter() - t0:.2f}s"
    )
    return session_id


# ---------------------------------------------------------------------------
# LangGraph — two-node RAG pipeline
#   retrieve_node  : FAISS top-k search → builds context string
#   generate_node  : OpenAI chat with context + conversation history
# ---------------------------------------------------------------------------

class _ChatState(TypedDict):
    session_id:   str
    question:     str
    query_vector: list[float]   # pre-computed before graph invocation
    history:      list[dict]    # snapshot of session history passed in
    context:      str           # filled by retrieve_node
    answer:       str           # filled by generate_node


def _node_retrieve(state: _ChatState) -> dict:
    """
    Sync node — safe because FAISS search is CPU-bound and very fast.
    Retrieves the top-k most relevant chunks from the session FAISS index.
    """
    session = _sessions.get(state["session_id"])
    if not session:
        return {"context": ""}

    q_vec = np.array([state["query_vector"]], dtype=np.float32)
    faiss.normalize_L2(q_vec)

    k = min(_TOP_K, len(session["chunks"]))
    _, indices = session["index"].search(q_vec, k)

    retrieved = [
        session["chunks"][idx]
        for idx in indices[0]
        if idx >= 0
    ]

    context = "\n\n---\n\n".join(retrieved)
    logger.debug(f"[rag] [{state['session_id'][:8]}] retrieved {len(retrieved)} chunk(s)")
    return {"context": context}


async def _node_generate(state: _ChatState) -> dict:
    """
    Async node — calls OpenAI with the retrieved context + conversation history.
    Reuses the existing _client and _build_api_kwargs from ai_model.py so model
    selection and parameter handling stay consistent across the project.
    """
    from llm_model.ai_model import _client as _llm, _build_api_kwargs

    system = (
        "You are a helpful document assistant. Answer the user's question using ONLY "
        "the document context provided below.\n"
        "If the answer cannot be found in the context, clearly say so — do NOT invent information.\n"
        "Be concise and precise. Quote relevant document text when it helps.\n\n"
        f"Document context:\n{state['context']}"
    )

    messages: list[dict] = [{"role": "system", "content": system}]

    # Include the last N turns from conversation history
    for turn in state["history"][-(_MAX_HISTORY * 2):]:
        messages.append(turn)

    messages.append({"role": "user", "content": state["question"]})

    kwargs   = _build_api_kwargs(messages, use_json=False, streaming=False)
    response = await _llm.chat.completions.create(**kwargs)
    answer   = response.choices[0].message.content or ""

    logger.debug(f"[rag] [{state['session_id'][:8]}] answer generated ({len(answer)} chars)")
    return {"answer": answer}


def _build_rag_graph():
    g = StateGraph(_ChatState)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("generate", _node_generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


_graph = _build_rag_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def chat(session_id: str, question: str) -> dict:
    """
    Run one RAG chat turn.

    1. Embeds the question.
    2. Invokes the LangGraph pipeline (retrieve → generate).
    3. Appends the turn to session history.

    Returns {"answer": str, "history_length": int (number of completed turns)}.
    Raises ValueError if session_id is unknown.
    """
    session = _sessions.get(session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found or has expired.")

    t0 = time.perf_counter()

    query_vector = await _embed_query(question)

    state = _ChatState(
        session_id   = session_id,
        question     = question,
        query_vector = query_vector,
        history      = list(session["history"]),
        context      = "",
        answer       = "",
    )

    result = await _graph.ainvoke(state)
    answer = result["answer"]

    # Persist turn in session
    session["history"].append({"role": "user",      "content": question})
    session["history"].append({"role": "assistant",  "content": answer})

    logger.info(
        f"[rag] [{session_id[:8]}] turn complete — "
        f"{len(answer)} chars | {time.perf_counter() - t0:.2f}s"
    )
    return {
        "answer":         answer,
        "history_length": len(session["history"]) // 2,
    }


def clear_history(session_id: str) -> bool:
    """Clear conversation history for a session (keep the index)."""
    session = _sessions.get(session_id)
    if not session:
        return False
    session["history"] = []
    return True


def delete_session(session_id: str) -> bool:
    """Remove the session and its FAISS index from memory."""
    return _sessions.pop(session_id, None) is not None


def get_session_info(session_id: str) -> dict | None:
    """Return metadata for a session, or None if it doesn't exist."""
    session = _sessions.get(session_id)
    if not session:
        return None
    return {
        "chunks_indexed": len(session["chunks"]),
        "turns":          len(session["history"]) // 2,
    }
