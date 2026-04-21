import os
import logging
from typing import TypedDict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import AsyncOpenAI, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END, START

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536
CHUNK_SIZE      = 1000   # characters
CHUNK_OVERLAP   = 150
TOP_K           = 6
MAX_REWRITES    = 2

_OPENAI_KEY   = os.environ.get("OPENAI_API_KEY", "")
_MODEL        = os.environ.get("MODEL_NAME", "gpt-4o")
_FAST_MODEL   = os.environ.get("FAST_MODEL_NAME", "gpt-4.1-nano")
_QDRANT_URL   = os.environ.get("QDRANT_URL")          # optional remote server
_QDRANT_PATH  = os.path.join(os.path.dirname(__file__), "..", "qdrant_storage")

_sync_client  = OpenAI(api_key=_OPENAI_KEY)
_async_client = AsyncOpenAI(api_key=_OPENAI_KEY)

# ---------------------------------------------------------------------------
# Qdrant client — singleton per process
# ---------------------------------------------------------------------------
_qdrant: QdrantClient | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        if _QDRANT_URL:
            _qdrant = QdrantClient(url=_QDRANT_URL)
            logger.info(f"Qdrant connected to server: {_QDRANT_URL}")
        else:
            os.makedirs(_QDRANT_PATH, exist_ok=True)
            _qdrant = QdrantClient(path=_QDRANT_PATH)
            logger.info(f"Qdrant using local storage: {_QDRANT_PATH}")
    return _qdrant


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_batch(texts: List[str]) -> List[List[float]]:
    response = _sync_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


async def _embed_query(text: str) -> List[float]:
    response = await _async_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_document(text: str, collection_id: str) -> int:
    """
    Chunk → embed → store in Qdrant.
    Returns number of chunks stored.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("Document produced no text chunks after splitting.")

    embeddings = _embed_batch(chunks)

    client = _get_qdrant()

    # Drop existing collection if it exists (re-ingest scenario)
    existing = [c.name for c in client.get_collections().collections]
    if collection_id in existing:
        client.delete_collection(collection_id)

    client.create_collection(
        collection_name=collection_id,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=emb,
            payload={"text": chunk, "chunk_index": i},
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name=collection_id, points=points)
    logger.info(f"[ingest] {len(chunks)} chunks stored → collection '{collection_id}'")
    return len(chunks)


def collection_exists(collection_id: str) -> bool:
    client = _get_qdrant()
    names = [c.name for c in client.get_collections().collections]
    return collection_id in names


# ---------------------------------------------------------------------------
# LangGraph RAG state
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question:      str
    collection_id: str
    documents:     List[str]
    answer:        str
    rewrite_count: int


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

async def _retrieve(state: RAGState) -> dict:
    query_vec = await _embed_query(state["question"])
    hits = _get_qdrant().search(
        collection_name=state["collection_id"],
        query_vector=query_vec,
        limit=TOP_K,
    )
    docs = [hit.payload["text"] for hit in hits]
    logger.debug(f"[retrieve] {len(docs)} chunks retrieved for: '{state['question']}'")
    return {"documents": docs}


async def _grade_documents(state: RAGState) -> dict:
    """Filter retrieved chunks to only those relevant to the question."""
    if not state["documents"]:
        return {"documents": []}

    numbered = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(state["documents"])
    )
    prompt = (
        "You are a relevance grader. Given a user question and numbered document chunks, "
        "return a comma-separated list of the chunk numbers that are useful for answering the question. "
        "If none are relevant, return the word 'none'.\n\n"
        f"Question: {state['question']}\n\n"
        f"Chunks:\n{numbered}\n\n"
        "Relevant chunk numbers:"
    )
    resp = await _async_client.chat.completions.create(
        model=_FAST_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip().lower()

    if raw == "none":
        return {"documents": []}

    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()]
        relevant = [state["documents"][i] for i in indices if 0 <= i < len(state["documents"])]
    except Exception:
        relevant = state["documents"]

    logger.debug(f"[grade] {len(relevant)}/{len(state['documents'])} chunks kept")
    return {"documents": relevant}


async def _generate(state: RAGState) -> dict:
    if state["documents"]:
        context = "\n\n---\n\n".join(state["documents"])
    else:
        context = "No relevant context was found in the document."

    prompt = (
        "You are a document assistant. Answer the user's question using only the document "
        "context provided below. If the context is insufficient, say so clearly and briefly.\n\n"
        f"Document Context:\n{context}\n\n"
        f"Question: {state['question']}\n\n"
        "Answer:"
    )
    resp = await _async_client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content.strip()
    logger.debug(f"[generate] answer length={len(answer)}")
    return {"answer": answer}


async def _rewrite_query(state: RAGState) -> dict:
    prompt = (
        "Rewrite the following question to improve document retrieval. "
        "Use different phrasing but keep the same intent. Return only the rewritten question.\n\n"
        f"Original question: {state['question']}\n\n"
        "Rewritten question:"
    )
    resp = await _async_client.chat.completions.create(
        model=_FAST_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.4,
    )
    new_q = resp.choices[0].message.content.strip()
    logger.info(f"[rewrite] '{state['question']}' → '{new_q}'")
    return {
        "question":      new_q,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def _route_after_grading(state: RAGState) -> str:
    # Generate if we have relevant docs OR have already retried enough
    if state["documents"] or state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "generate"
    return "rewrite_query"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

_builder = StateGraph(RAGState)
_builder.add_node("retrieve",        _retrieve)
_builder.add_node("grade_documents", _grade_documents)
_builder.add_node("generate",        _generate)
_builder.add_node("rewrite_query",   _rewrite_query)

_builder.add_edge(START,              "retrieve")
_builder.add_edge("retrieve",         "grade_documents")
_builder.add_conditional_edges(
    "grade_documents",
    _route_after_grading,
    {"generate": "generate", "rewrite_query": "rewrite_query"},
)
_builder.add_edge("rewrite_query",   "retrieve")
_builder.add_edge("generate",        END)

rag_graph = _builder.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_chat(collection_id: str, question: str) -> dict:
    """
    Execute the RAG graph and return the answer with metadata.
    """
    initial: RAGState = {
        "question":      question,
        "collection_id": collection_id,
        "documents":     [],
        "answer":        "",
        "rewrite_count": 0,
    }
    result = await rag_graph.ainvoke(initial)
    return {
        "answer":          result["answer"],
        "sources_used":    len(result["documents"]),
        "question_used":   result["question"],   # may differ if query was rewritten
        "rewrites_done":   result.get("rewrite_count", 0),
    }
