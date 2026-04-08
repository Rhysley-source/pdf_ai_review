"""
RAG-based Document Q&A
======================
Architecture:
  1. index_document(text) — splits text into chunks, embeds with OpenAI,
     stores in an in-memory FAISS index, returns a session_id.
  2. answer_question(session_id, question) — runs a two-node LangGraph pipeline:
       retrieve  → similarity search FAISS for top-k relevant chunks
       generate  → LLM answers using retrieved context only
  3. Session store is in-memory (dict).  In production replace with Redis + disk FAISS.
"""

import asyncio
import logging
import os
import uuid
from functools import partial
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5-nano")

# ---------------------------------------------------------------------------
# Shared singletons
# ---------------------------------------------------------------------------

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-small",
)

# ---------------------------------------------------------------------------
# In-memory session store
# session_id → {"vectorstore": FAISS, "full_text": str, "chunk_count": int}
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

async def index_document(text: str) -> str:
    """
    Split text → embed → store in FAISS.
    Returns a session_id the caller uses for all subsequent Q&A calls.
    CPU-bound FAISS operations are offloaded to a thread pool so the
    asyncio event loop is never blocked.
    """
    session_id = str(uuid.uuid4())[:12]
    chunks = _splitter.split_text(text)
    logger.info(f"[rag_qa] Indexing session={session_id}  chunks={len(chunks)}")

    loop = asyncio.get_running_loop()
    # FAISS.from_texts is synchronous; run it in the thread pool
    vectorstore = await loop.run_in_executor(
        None,
        partial(FAISS.from_texts, chunks, _embeddings),
    )

    _SESSIONS[session_id] = {
        "vectorstore": vectorstore,
        "full_text": text,
        "chunk_count": len(chunks),
    }

    logger.info(f"[rag_qa] Indexed {len(chunks)} chunks → session_id={session_id}")
    return session_id


def get_session(session_id: str) -> dict | None:
    return _SESSIONS.get(session_id)


def delete_session(session_id: str) -> bool:
    if session_id in _SESSIONS:
        del _SESSIONS[session_id]
        return True
    return False


def list_sessions() -> list[dict]:
    return [
        {"session_id": sid, "chunk_count": data["chunk_count"]}
        for sid, data in _SESSIONS.items()
    ]


# ---------------------------------------------------------------------------
# LangGraph RAG pipeline factory
# ---------------------------------------------------------------------------

def _build_rag_graph(vectorstore: FAISS):
    """
    Returns a compiled LangGraph that runs:
      START → retrieve → generate → END
    """

    async def retrieve(state: RAGState) -> dict:
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(
            None,
            partial(vectorstore.similarity_search, state["question"], k=5),
        )
        context = [doc.page_content for doc in docs]
        logger.debug(f"[rag_qa:retrieve] retrieved {len(context)} chunks")
        return {"context": context}

    async def generate(state: RAGState) -> dict:
        context_text = "\n\n---\n\n".join(state["context"])

        system_prompt = (
            "You are a legal document assistant. "
            "Answer the user's question based ONLY on the provided document context. "
            "Be precise and cite the relevant clause or section when possible. "
            "If the answer is not found in the context, say clearly: "
            "'This information is not present in the document.'"
        )
        user_prompt = (
            f"Document Context:\n{context_text}\n\n"
            f"Question: {state['question']}"
        )

        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL_NAME)
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return {"answer": response.content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public Q&A entry point
# ---------------------------------------------------------------------------

async def answer_question(session_id: str, question: str) -> dict:
    """
    Run the RAG pipeline to answer `question` about the indexed document.
    Returns: {answer, sources, session_id, chunk_count}
    Raises ValueError if session_id is not found.
    """
    session = get_session(session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found or has been deleted.")

    vectorstore: FAISS = session["vectorstore"]
    rag_graph = _build_rag_graph(vectorstore)

    initial_state: RAGState = {
        "question": question,
        "context": [],
        "answer": "",
    }
    result = await rag_graph.ainvoke(initial_state)

    # Retrieve sources for transparency
    loop = asyncio.get_running_loop()
    source_docs = await loop.run_in_executor(
        None,
        partial(vectorstore.similarity_search, question, k=4),
    )
    sources = [
        {"excerpt": doc.page_content[:300]}
        for doc in source_docs
    ]

    logger.info(f"[rag_qa] answer_question session={session_id} q_len={len(question)}")

    return {
        "session_id": session_id,
        "question": question,
        "answer": result["answer"],
        "sources": sources,
        "chunks_searched": session["chunk_count"],
    }
