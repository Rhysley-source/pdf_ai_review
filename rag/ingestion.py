import hashlib
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import fitz  # PyMuPDF — already in project
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from rag.qdrant_client_setup import COLLECTION_DOCUMENTS, get_async_client

logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n\n".join(pages)


def chunk_text(text: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents([text])
    return [{"text": d.page_content, "chunk_index": i} for i, d in enumerate(docs)]


async def embed_and_store(
    file_bytes: bytes,
    filename: str,
    user_id: str,
    document_id: Optional[str] = None,
) -> dict:
    if document_id is None:
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
        document_id = f"{user_id}_{file_hash}"

    raw_text = extract_text_from_pdf(file_bytes)
    if not raw_text.strip():
        raise ValueError("Could not extract text from the uploaded PDF.")

    chunks = chunk_text(raw_text)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectors = await embedder.aembed_documents([c["text"] for c in chunks])

    uploaded_at = datetime.utcnow().isoformat()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "document_id": document_id,
                "user_id": user_id,
                "filename": filename,
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"],
                "uploaded_at": uploaded_at,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client = get_async_client()
    try:
        await client.upsert(collection_name=COLLECTION_DOCUMENTS, points=points)
    finally:
        await client.close()

    logger.info(f"Ingested {len(points)} chunks for document_id={document_id}")
    return {
        "document_id": document_id,
        "filename": filename,
        "chunks_stored": len(points),
        "total_chars": len(raw_text),
        "user_id": user_id,
    }


async def delete_document(document_id: str):
    client = get_async_client()
    try:
        await client.delete(
            collection_name=COLLECTION_DOCUMENTS,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
    finally:
        await client.close()
    logger.info(f"Deleted document_id={document_id} from Qdrant")
