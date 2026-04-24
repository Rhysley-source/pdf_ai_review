import hashlib
import logging
import os

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from qdrant_client.models import PointStruct

from rag.qdrant_client_setup import COLLECTION_SESSIONS, EMBEDDING_DIM, get_async_client

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 20))


def _session_id_to_uuid(session_id: str) -> str:
    h = hashlib.md5(session_id.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


class SessionMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._point_id = _session_id_to_uuid(session_id)

    async def load(self) -> list[BaseMessage]:
        client = get_async_client()
        try:
            results = await client.retrieve(
                collection_name=COLLECTION_SESSIONS,
                ids=[self._point_id],
                with_payload=True,
                with_vectors=False,
            )
        finally:
            await client.close()

        if not results:
            return []

        raw = results[0].payload.get("messages", [])
        try:
            return messages_from_dict(raw)
        except Exception:
            return []

    async def save(self, messages: list[BaseMessage]):
        trimmed = messages[-(MAX_HISTORY_TURNS * 2):]
        serialized = messages_to_dict(trimmed)

        client = get_async_client()
        try:
            await client.upsert(
                collection_name=COLLECTION_SESSIONS,
                points=[
                    PointStruct(
                        id=self._point_id,
                        vector=[0.0] * EMBEDDING_DIM,
                        payload={"session_id": self.session_id, "messages": serialized},
                    )
                ],
            )
        finally:
            await client.close()

    async def clear(self):
        client = get_async_client()
        try:
            await client.delete(
                collection_name=COLLECTION_SESSIONS,
                points_selector=[self._point_id],
            )
        finally:
            await client.close()
