import os
import logging

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PayloadSchemaType,
    OptimizersConfigDiff,
)

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_DOCUMENTS = os.getenv("QDRANT_COLLECTION_DOCUMENTS", "documents")
COLLECTION_SESSIONS = os.getenv("QDRANT_COLLECTION_SESSIONS", "chat_sessions")
EMBEDDING_DIM = 1536  # text-embedding-ada-002


def get_sync_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_async_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


async def initialize_collections():
    client = get_async_client()
    try:
        existing = await client.get_collections()
        existing_names = [c.name for c in existing.collections]

        if COLLECTION_DOCUMENTS not in existing_names:
            await client.create_collection(
                collection_name=COLLECTION_DOCUMENTS,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=10_000),
            )
            await client.create_payload_index(
                collection_name=COLLECTION_DOCUMENTS,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=COLLECTION_DOCUMENTS,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_DOCUMENTS}")

        if COLLECTION_SESSIONS not in existing_names:
            await client.create_collection(
                collection_name=COLLECTION_SESSIONS,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            await client.create_payload_index(
                collection_name=COLLECTION_SESSIONS,
                field_name="session_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_SESSIONS}")
    finally:
        await client.close()
