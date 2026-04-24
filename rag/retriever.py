import logging
import os
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue

from rag.qdrant_client_setup import (
    COLLECTION_DOCUMENTS,
    QDRANT_API_KEY,
    QDRANT_URL,
    get_async_client,
    get_sync_client,
)

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 5))


def build_retriever(
    document_id: Optional[str] = None,
    user_id: Optional[str] = None,
):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    client = get_sync_client()

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_DOCUMENTS,
        embedding=embeddings,
    )

    conditions = []
    if document_id:
        conditions.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
    elif user_id:
        conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))

    search_kwargs: dict = {"k": TOP_K}
    if conditions:
        search_kwargs["filter"] = Filter(must=conditions)

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


async def retrieve_document_text(document_id: str) -> str:
    """Fetch all chunks for a document ordered by chunk_index — used by analysis tools."""
    client = get_async_client()
    try:
        results, _ = await client.scroll(
            collection_name=COLLECTION_DOCUMENTS,
            scroll_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
    finally:
        await client.close()

    if not results:
        return ""

    chunks = sorted(results, key=lambda p: p.payload.get("chunk_index", 0))
    return "\n\n".join(p.payload["text"] for p in chunks)
