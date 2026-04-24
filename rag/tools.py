import asyncio
import json
import logging

from langchain_core.tools import tool

from rag.retriever import build_retriever, retrieve_document_text

logger = logging.getLogger(__name__)


@tool
async def rag_search_tool(query: str, document_id: str = "") -> str:
    """
    Search the uploaded document using semantic similarity to answer user questions.
    Use this for any factual question about document content.
    Args:
        query: The user's question or search phrase.
        document_id: Scope search to this specific document (recommended).
    """
    retriever = build_retriever(document_id=document_id if document_id else None)
    docs = await retriever.ainvoke(query)

    if not docs:
        return "No relevant content found in the document for this query."

    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(
            f"[Chunk {i}] (doc={meta.get('document_id', '?')}, "
            f"chunk={meta.get('chunk_index', '?')}):\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


@tool
async def risk_analysis_tool(document_id: str, document_type: str = "contract") -> str:
    """
    Perform comprehensive risk analysis on a document.
    Detects financial, legal, and compliance risks with severity ratings.
    Args:
        document_id: The ID of the uploaded document to analyze.
        document_type: Type of document (contract, nda, lease, etc.).
    """
    from feature_modules.risk_detection import analyze_document_risks

    text = await retrieve_document_text(document_id)
    if not text:
        return json.dumps({"error": f"Document '{document_id}' not found or empty."})

    try:
        result = await analyze_document_risks(text)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.exception(f"[risk_analysis_tool] Failed for {document_id}")
        return json.dumps({"error": str(exc)})


@tool
async def key_clause_extraction_tool(document_id: str, document_type: str = "") -> str:
    """
    Extract key clauses from a legal document.
    Identifies critical sections like termination, payment, liability, IP ownership, etc.
    Args:
        document_id: The ID of the uploaded document.
        document_type: Optional hint for the document type.
    """
    from feature_modules.key_clause_extraction import extract_key_clauses

    text = await retrieve_document_text(document_id)
    if not text:
        return json.dumps({"error": f"Document '{document_id}' not found or empty."})

    try:
        result = await extract_key_clauses(text)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.exception(f"[key_clause_extraction_tool] Failed for {document_id}")
        return json.dumps({"error": str(exc)})


@tool
async def red_flag_scanner_tool(document_id: str) -> str:
    """
    Scan a document for red flags: dangerous terms, one-sided clauses,
    missing protections, or legally unusual language.
    Args:
        document_id: The ID of the uploaded document.
    """
    from feature_modules.red_flag_scanner import scan_red_flags

    text = await retrieve_document_text(document_id)
    if not text:
        return json.dumps({"error": f"Document '{document_id}' not found or empty."})

    try:
        result = await scan_red_flags(text)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.exception(f"[red_flag_scanner_tool] Failed for {document_id}")
        return json.dumps({"error": str(exc)})


@tool
async def document_compare_tool(document_id_1: str, document_id_2: str) -> str:
    """
    Compare two uploaded documents side-by-side.
    Identifies differences in clauses, terms, obligations, and risk levels.
    Args:
        document_id_1: ID of the first document.
        document_id_2: ID of the second document.
    """
    from feature_modules.key_clause_extraction import extract_key_clauses
    from feature_modules.document_comparison import compare_documents

    text1, text2 = await asyncio.gather(
        retrieve_document_text(document_id_1),
        retrieve_document_text(document_id_2),
    )

    if not text1:
        return json.dumps({"error": f"Document '{document_id_1}' not found or empty."})
    if not text2:
        return json.dumps({"error": f"Document '{document_id_2}' not found or empty."})

    try:
        extraction1, extraction2 = await asyncio.gather(
            extract_key_clauses(text1),
            extract_key_clauses(text2),
        )
        result = await compare_documents(
            extraction1=extraction1,
            extraction2=extraction2,
            text1=text1,
            text2=text2,
            doc1_filename=document_id_1,
            doc2_filename=document_id_2,
        )
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.exception(f"[document_compare_tool] Failed for {document_id_1} vs {document_id_2}")
        return json.dumps({"error": str(exc)})


ALL_TOOLS = [
    rag_search_tool,
    risk_analysis_tool,
    key_clause_extraction_tool,
    red_flag_scanner_tool,
    document_compare_tool,
]
