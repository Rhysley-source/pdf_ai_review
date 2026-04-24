# RAG-Based Document Q&A — Full Implementation Guide

> No code. Pure steps, flows, decisions, and data movements.
> Stack: FastAPI · Qdrant · OpenAI · LangGraph · Memory Management
> Tools: Risk Analysis · Key Clause Extraction · Red Flag Scanner · Document Compare

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Dependencies & Environment Setup](#2-dependencies--environment-setup)
3. [Qdrant Database Setup](#3-qdrant-database-setup)
4. [Document Upload & Ingestion Pipeline](#4-document-upload--ingestion-pipeline)
5. [RAG Core — Retriever & Embeddings](#5-rag-core--retriever--embeddings)
6. [Memory Management](#6-memory-management)
7. [Tool Definitions — All 5 Tools](#7-tool-definitions--all-5-tools)
8. [LangGraph Agent with Memory](#8-langgraph-agent-with-memory)
9. [FastAPI Routes](#9-fastapi-routes)
10. [Complete File Structure](#10-complete-file-structure)
11. [Implementation Order (Phase by Phase)](#11-implementation-order-phase-by-phase)
12. [End-to-End Data Flow Summary](#12-end-to-end-data-flow-summary)

---

## 1. Architecture Overview

### 1.1 — Big Picture System Map

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Application                              │
│                                                                            │
│   POST /rag/upload     POST /rag/chat     POST /rag/analyze/{tool}        │
│   GET  /rag/documents  POST /rag/chat/stream  DELETE /rag/document        │
└──────┬─────────────────────────┬──────────────────────┬───────────────────┘
       │                         │                      │
       ▼                         ▼                      ▼
┌─────────────────┐   ┌──────────────────────┐  ┌──────────────────────┐
│  Ingestion       │   │   LangGraph Agent    │  │  Direct Tool         │
│  Pipeline        │   │   (with Memory)      │  │  Endpoints           │
│                  │   │                      │  │  (no agent loop)     │
│  Step 1: Extract │   │  Agent Node          │  │                      │
│  Step 2: Chunk   │   │    ↕                 │  │  /analyze/risks      │
│  Step 3: Embed   │   │  Tool Router Node    │  │  /analyze/key-clauses│
│  Step 4: Store   │   │    ↕                 │  │  /analyze/red-flags  │
│                  │   │  Tool Execution Node │  │  /analyze/compare    │
└────────┬─────────┘   └──────────┬───────────┘  └──────────┬───────────┘
         │                        │                          │
         ▼                        ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Qdrant Vector DB                              │
│                                                                           │
│   Collection: "documents"              Collection: "chat_sessions"        │
│   ┌──────────────────────────┐         ┌──────────────────────────┐      │
│   │ id: UUID                 │         │ id: UUID (from session)  │      │
│   │ vector: [1536 floats]    │         │ vector: [zeros]          │      │
│   │ payload:                 │         │ payload:                 │      │
│   │  • document_id (index)   │         │  • session_id (index)   │      │
│   │  • user_id (index)       │         │  • messages: [JSON]     │      │
│   │  • text (chunk content)  │         └──────────────────────────┘      │
│   │  • chunk_index           │                                            │
│   │  • filename              │                                            │
│   │  • uploaded_at           │                                            │
│   └──────────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              OpenAI                                      │
│                                                                           │
│   text-embedding-ada-002 → converts text chunks into 1536-dim vectors   │
│   gpt-4o               → reads context + history, reasons, calls tools  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 — Three Entry Modes

```
MODE 1: UPLOAD
  Client → POST /rag/upload → Ingestion Pipeline → Qdrant
  Result: document is stored and becomes searchable

MODE 2: AGENT CHAT
  Client → POST /rag/chat → Memory Load → LangGraph Agent
         → Agent chooses tool(s) → Tool(s) query Qdrant + call feature_modules
         → Agent composes answer → Memory Save → Client
  Result: natural language answer grounded in the document

MODE 3: DIRECT TOOL
  Client → POST /rag/analyze/risks → Tool runs directly (no agent loop)
         → Fetches full text from Qdrant → Calls feature_module → Client
  Result: structured JSON analysis report (faster, no LLM reasoning step)
```

---

## 2. Dependencies & Environment Setup

### 2.1 — New Packages to Add

```
PACKAGE                 PURPOSE
──────────────────────────────────────────────────────────────────
qdrant-client           Connect to Qdrant, store & search vectors
langgraph               Build the stateful agent graph
langchain               Core LangChain abstractions (retrievers, tools)
langchain-openai        OpenAI embeddings + chat model wrappers
langchain-qdrant        LangChain ↔ Qdrant vector store integration
langchain-community     Extra LangChain utilities
aiofiles                Async file operations (for upload handling)
python-multipart        Parse multipart/form-data (file upload forms)
```

### 2.2 — Environment Variables to Add

```
VARIABLE                    DEFAULT                 PURPOSE
──────────────────────────────────────────────────────────────────────────
QDRANT_URL                  http://localhost:6333   Where Qdrant is running
QDRANT_API_KEY              (empty)                 API key if using Qdrant Cloud
QDRANT_COLLECTION_DOCUMENTS documents               Name of the documents collection
QDRANT_COLLECTION_SESSIONS  chat_sessions           Name of the sessions collection
EMBEDDING_MODEL             text-embedding-ada-002  OpenAI embedding model
CHAT_MODEL                  gpt-4o                  OpenAI chat model for agent
CHUNK_SIZE                  1000                    Characters per chunk
CHUNK_OVERLAP               200                     Overlapping characters between chunks
TOP_K_RETRIEVAL             5                       Number of chunks returned per search
MAX_HISTORY_TURNS           20                      Max Q&A turns kept in memory
```

### 2.3 — Start Qdrant (Local Docker)

```
STEP 1: Pull and run Qdrant container
  Command: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

STEP 2: Verify Qdrant is running
  Open browser: http://localhost:6333/dashboard
  You should see the Qdrant Web UI

STEP 3: Qdrant Cloud alternative
  • Create account at cloud.qdrant.io
  • Create a cluster → copy URL and API Key
  • Set QDRANT_URL and QDRANT_API_KEY in .env
```

---

## 3. Qdrant Database Setup

### 3.1 — What initialize_collections() Does

```
ON APPLICATION STARTUP (lifespan hook in s_main.py):

  STEP 1: Connect to Qdrant
           → Read QDRANT_URL + QDRANT_API_KEY from env

  STEP 2: List existing collections
           → Ask Qdrant: "what collections exist?"

  STEP 3: Create "documents" collection (if missing)
           → Vector size: 1536 (matches text-embedding-ada-002 output)
           → Distance metric: COSINE (standard for text similarity)
           → Indexing threshold: 10,000 points
             (Qdrant builds HNSW index after 10K vectors for fast search)
           → Create payload index on "document_id" field (type: keyword)
             (enables fast filter: "only search within this document")
           → Create payload index on "user_id" field (type: keyword)
             (enables fast filter: "only search this user's documents")

  STEP 4: Create "chat_sessions" collection (if missing)
           → Vector size: 1536 (placeholder — we never search by vector here)
           → Distance metric: COSINE
           → Create payload index on "session_id" field (type: keyword)
             (enables fast fetch by exact session_id)

  STEP 5: Close connection
```

### 3.2 — Why Two Collections?

```
"documents" collection
  → Purpose: semantic search over document content
  → Searched by vector similarity + filtered by document_id
  → Many points per document (one per chunk)
  → Queried by: rag_search_tool, retrieve_document_text()

"chat_sessions" collection
  → Purpose: store conversation history (like a key-value store)
  → Never searched by vector — only fetched by exact session_id
  → One point per session (upserted/overwritten each turn)
  → Queried by: SessionMemory.load(), SessionMemory.save()
```

---

## 4. Document Upload & Ingestion Pipeline

### 4.1 — Full Upload Flow

```
CLIENT                      FASTAPI ROUTE              INGESTION MODULE
  │                               │                           │
  │── POST /rag/upload ──────────▶│                           │
  │   Content-Type: multipart     │                           │
  │   Body:                       │                           │
  │     file = contract.pdf       │                           │
  │     user_id = "user_123"      │                           │
  │                               │                           │
  │                               │── VALIDATE ──────────────▶│
  │                               │   • filename ends in .pdf  │
  │                               │   • file size ≤ 50 MB      │
  │                               │   • user_id not empty      │
  │                               │                           │
  │                               │── GENERATE document_id ──▶│
  │                               │   SHA256(file_bytes)[:12]  │
  │                               │   + user_id prefix         │
  │                               │   → "user_123_a1b2c3d4e5f6"│
  │                               │                           │
  │                               │── EXTRACT TEXT ──────────▶│
  │                               │                           │ PyMuPDF opens PDF
  │                               │                           │ Reads each page
  │                               │                           │ Joins with \n\n
  │                               │                           │ → raw_text string
  │                               │                           │
  │                               │── CHUNK TEXT ────────────▶│
  │                               │                           │ RecursiveTextSplitter
  │                               │                           │ chunk_size=1000
  │                               │                           │ chunk_overlap=200
  │                               │                           │ separators priority:
  │                               │                           │   \n\n → \n → ". "
  │                               │                           │   → " " → ""
  │                               │                           │ → chunks[] list
  │                               │                           │
  │                               │── EMBED CHUNKS ──────────▶│
  │                               │                           │ OpenAI ada-002
  │                               │                           │ Batch all chunk texts
  │                               │                           │ → vectors[] list
  │                               │                           │ (1536 floats each)
  │                               │                           │
  │                               │── BUILD POINTS ──────────▶│
  │                               │                           │ For each chunk+vector:
  │                               │                           │ PointStruct {
  │                               │                           │   id: random UUID
  │                               │                           │   vector: [1536 floats]
  │                               │                           │   payload: {
  │                               │                           │     document_id,
  │                               │                           │     user_id,
  │                               │                           │     filename,
  │                               │                           │     text: chunk text,
  │                               │                           │     chunk_index: 0,1,2…
  │                               │                           │     uploaded_at: ISO ts
  │                               │                           │   }
  │                               │                           │ }
  │                               │                           │
  │                               │── UPSERT TO QDRANT ──────▶│ → Qdrant "documents"
  │                               │                           │   collection
  │                               │                           │
  │◀── 200 OK ────────────────────│                           │
  │  {                            │                           │
  │    document_id: "user_123_…", │                           │
  │    filename: "contract.pdf",  │                           │
  │    chunks_stored: 42,         │                           │
  │    total_chars: 41200         │                           │
  │  }                            │                           │
```

### 4.2 — Chunking Strategy Visualized

```
WHY OVERLAP? — Clauses span chunk boundaries

  Original contract text (simplified):
  "…payment shall be due within 30 days of invoice. Late payments
  attract a 2% monthly penalty as per clause 8.3 of this agreement…"

  WITHOUT overlap (chunk_size=30, overlap=0):
  Chunk A: "…payment shall be due within 30 days"
  Chunk B: "of invoice. Late payments attract a"
  Chunk C: "2% monthly penalty as per clause 8.3"
  → A user asking "what is the late payment penalty?"
    may only get Chunk C (missing the "30 days" context from A)

  WITH overlap (chunk_size=30, overlap=10):
  Chunk A: "…payment shall be due within 30 days"
  Chunk B: "within 30 days of invoice. Late payments"  ← overlaps A
  Chunk C: "Late payments attract a 2% monthly penalty" ← overlaps B
  → The 30-day and penalty info always appear together in at least one chunk

SEPARATOR PRIORITY (RecursiveCharacterTextSplitter):
  1st try: split on "\n\n"  (paragraph breaks — best semantic split)
  2nd try: split on "\n"    (line breaks)
  3rd try: split on ". "   (sentence ends)
  4th try: split on " "    (word boundaries)
  5th try: split on ""     (character — last resort)
  → Always tries to preserve meaning at the highest possible level
```

### 4.3 — Document ID Generation

```
INPUT:  file_bytes (raw PDF binary), user_id

STEP 1: Hash the file bytes with SHA256
        → always same hash for same file
        → different hash for even 1 byte difference

STEP 2: Take first 12 hex characters of hash
        e.g. "a1b2c3d4e5f6"

STEP 3: Prefix with user_id
        → "user_123_a1b2c3d4e5f6"

RESULT: document_id is deterministic
        Upload same file twice → same document_id → Qdrant upserts (no duplicates)
        Upload different file → different document_id → separate set of chunks
```

---

## 5. RAG Core — Retriever & Embeddings

### 5.1 — How Semantic Search Works

```
INDEXING TIME (during upload):
  "The payment shall be due within 30 days of invoice."
         │
         ▼  OpenAI ada-002
  [0.12, -0.34, 0.88, 0.45, -0.21, …]  (1536 numbers)
         │
         ▼  Stored in Qdrant with payload { document_id, text, chunk_index }

QUERY TIME (during chat):
  User asks: "When are payments due?"
         │
         ▼  OpenAI ada-002 (same model)
  [0.11, -0.33, 0.86, 0.44, -0.20, …]  (similar vector — similar meaning)
         │
         ▼  Qdrant cosine similarity search
  Compare query vector against all stored vectors in "documents" collection
  Filter: only vectors where payload.document_id = "user_123_a1b2c3d4"
         │
         ▼  Return top-5 most similar chunks
  Chunk at index 4 → similarity 0.97  ✓ returned
  Chunk at index 12 → similarity 0.94 ✓ returned
  Chunk at index 7 → similarity 0.91  ✓ returned
  Chunk at index 31 → similarity 0.23 ✗ too dissimilar, skipped
```

### 5.2 — build_retriever() Scoping

```
PURPOSE: Return only chunks from the RIGHT document/user

INPUTS:
  document_id (optional) — scope to one specific document
  user_id (optional)     — scope to all documents from one user

FILTER LOGIC:
  If document_id provided:
    → Filter: payload.document_id == "user_123_a1b2c3d4"
    → User gets answers only from their uploaded file

  If only user_id provided:
    → Filter: payload.user_id == "user_123"
    → User can ask questions across all their documents

  If neither provided:
    → No filter — searches entire collection (admin use only)

WHY THIS MATTERS:
  Without filtering, user A's question could retrieve
  text from user B's confidential contract. The filter
  makes retrieval both secure and accurate.
```

### 5.3 — retrieve_document_text() — Full Document Fetch

```
PURPOSE: Get ALL text from a document (for analysis tools, not Q&A)

WHY DIFFERENT FROM RETRIEVER:
  RAG retriever → returns top-K most relevant chunks (semantic search)
  retrieve_document_text → returns ALL chunks in order (full document)

  Analysis tools (risk, clauses, red flags) need the FULL document
  to give accurate results — you can't detect risks from 5 random chunks.

FLOW:
  STEP 1: Qdrant "scroll" operation (not search)
           → Filter: payload.document_id == requested_id
           → Fetch all matching points (up to 500)
           → No vector comparison needed (just payload retrieval)

  STEP 2: Sort results by chunk_index (0, 1, 2, 3 …)
           → Guarantees document text is in original order

  STEP 3: Join all chunk texts with "\n\n"
           → Returns one continuous string of the full document

  STEP 4: Return to calling tool
           → Tool passes this to the appropriate feature_module
```

---

## 6. Memory Management

### 6.1 — SessionMemory Class Responsibilities

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SessionMemory                                 │
│                                                                       │
│  CONSTRUCTOR: SessionMemory(session_id)                              │
│    → Converts session_id string to a deterministic UUID              │
│      using MD5 hash (same session_id → always same UUID)            │
│    → This UUID is the Qdrant point ID for this session              │
│                                                                       │
│  METHOD: load()                                                       │
│    → Fetch point from Qdrant "chat_sessions" by UUID                │
│    → Deserialize payload.messages → list of BaseMessage objects     │
│    → If point not found → return empty list []                      │
│                                                                       │
│  METHOD: save(messages)                                               │
│    → Trim: keep only last (MAX_HISTORY_TURNS × 2) messages          │
│    → Serialize messages → JSON-compatible list                       │
│    → Upsert point to Qdrant (create if new, overwrite if exists)    │
│    → Uses placeholder zero-vector (not needed for searching)        │
│                                                                       │
│  METHOD: clear()                                                      │
│    → Delete the point from Qdrant by UUID                           │
│    → Session history is gone permanently                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 — Memory Load → Agent → Memory Save Flow

```
EVERY CHAT TURN:

  ① Client sends: { message, session_id, user_id, document_id }
          │
          ▼
  ② SessionMemory.load(session_id)
     → Qdrant fetch by session UUID
     → Returns: [H1, A1, H2, A2, H3, A3]  (past messages)
          │
          ▼
  ③ Build agent state:
     messages = [H1, A1, H2, A2, H3, A3, H4(new)]
          │
          ▼
  ④ LangGraph agent runs (may call tools)
     → Appends: A4 (agent's answer)
     messages = [H1, A1, H2, A2, H3, A3, H4, A4]
          │
          ▼
  ⑤ SessionMemory.save([H1,A1,H2,A2,H3,A3,H4,A4])
     → Trim if > MAX_HISTORY_TURNS × 2
     → Upsert to Qdrant
          │
          ▼
  ⑥ Return A4.content to client
```

### 6.3 — Memory Trim Visualization

```
Setting: MAX_HISTORY_TURNS = 5 → keeps last 10 messages (5 pairs)

Turn 1 saved:  [H1, A1]                                          2 msgs
Turn 2 saved:  [H1, A1, H2, A2]                                  4 msgs
Turn 3 saved:  [H1, A1, H2, A2, H3, A3]                          6 msgs
Turn 4 saved:  [H1, A1, H2, A2, H3, A3, H4, A4]                  8 msgs
Turn 5 saved:  [H1, A1, H2, A2, H3, A3, H4, A4, H5, A5]         10 msgs
Turn 6 saved:  [H2, A2, H3, A3, H4, A4, H5, A5, H6, A6]         10 msgs ← H1,A1 dropped
Turn 7 saved:  [H3, A3, H4, A4, H5, A5, H6, A6, H7, A7]         10 msgs ← H2,A2 dropped

H = HumanMessage (user question)
A = AIMessage (agent response, may include tool calls)

WHY TRIM?
  gpt-4o has a context window limit (128K tokens).
  Keeping 20 turns of legal document analysis = thousands of tokens.
  Trimming to the most recent turns keeps the agent fast and cheap
  while preserving the most relevant conversational context.
```

### 6.4 — Session ID → Qdrant Point ID

```
WHY DETERMINISTIC UUID?
  session_id "abc-123" must always map to the same Qdrant point ID.
  If it were random, each save() would create a NEW point instead of
  overwriting the existing one → history would never accumulate.

HOW:
  session_id string → MD5 hash → 32 hex chars
  → Formatted as UUID: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

  "abc-123" → MD5 → "e99a18c428cb38d5f260853678922e03"
           → UUID → "e99a18c4-28cb-38d5-f260-853678922e03"

  Same session_id → always same UUID → Qdrant upsert overwrites correctly
```

---

## 7. Tool Definitions — All 5 Tools

### How Tools Connect to LangGraph

```
┌────────────────────────────────────────────────────────────────────┐
│                   Tool Registration Flow                            │
│                                                                      │
│  Each tool is decorated with @tool                                  │
│  → LangGraph reads the function name + docstring + parameters       │
│  → Generates a JSON schema for gpt-4o to understand                │
│  → gpt-4o decides when to call each tool based on user intent      │
│                                                                      │
│  Tool call lifecycle:                                                │
│  gpt-4o outputs:  { tool: "risk_analysis_tool",                    │
│                     args: { document_id: "user_123_abc" } }        │
│         ↓                                                            │
│  ToolNode receives this → calls the function with those args        │
│         ↓                                                            │
│  Function runs → returns string result                              │
│         ↓                                                            │
│  Result added as ToolMessage to agent state                         │
│         ↓                                                            │
│  gpt-4o reads ToolMessage → formulates final answer                │
└────────────────────────────────────────────────────────────────────┘
```

### Tool 1 — rag_search_tool

```
NAME:     rag_search_tool
TRIGGER:  User asks a question about document content
          e.g. "What are the payment terms?" / "Who are the parties?"

INPUT PARAMETERS:
  query        — the user's question or search phrase
  document_id  — (optional) limit search to one specific document

INTERNAL FLOW:
  STEP 1: build_retriever(document_id)
           → Creates a LangChain retriever scoped to the document
           → Connects to Qdrant "documents" collection

  STEP 2: retriever.ainvoke(query)
           → Converts query → embedding vector via OpenAI ada-002
           → Runs Qdrant similarity search filtered by document_id
           → Returns top-K (default: 5) most relevant chunks

  STEP 3: Format results
           → For each chunk: show chunk number, document_id, chunk_index, text
           → Join with separator lines

OUTPUT:  Plain text with K passages from the document
         Example:
           [Chunk 1] (doc=user_123_abc, chunk=4):
           "The payment shall be due within 30 days of invoice..."
           ---
           [Chunk 2] (doc=user_123_abc, chunk=17):
           "Late payments shall attract a 2% monthly fee..."

WHEN AGENT USES THIS:  For any factual question about document content
```

### Tool 2 — risk_analysis_tool

```
NAME:     risk_analysis_tool
TRIGGER:  User asks about risks, dangers, problems in the document
          e.g. "What are the risks?" / "Is this contract risky?"

INPUT PARAMETERS:
  document_id    — which document to analyze
  document_type  — hint: "contract", "nda", "lease", etc. (optional)

INTERNAL FLOW:
  STEP 1: retrieve_document_text(document_id)
           → Fetch ALL chunks from Qdrant in order
           → Join into full document text string
           → (Not semantic search — we need the ENTIRE document)

  STEP 2: Pass full text to existing feature module
           → analyze_document_risks(text)
           → This module uses your existing LLM prompts + OpenAI

  STEP 3: Serialize result to JSON string
           → Return to LangGraph as ToolMessage

OUTPUT:  JSON risk report
  {
    detected_risks: [
      {
        risk_name: "Unlimited Liability Clause",
        severity: "High",
        severity_reason: "No cap on damages…",
        clause_found: "Party A shall be liable for all losses…",
        impact: "Company could face unlimited financial exposure",
        mitigation: "Negotiate a liability cap equal to contract value"
      }
    ],
    document_type: "contract",
    overall_risk_level: "High"
  }

WHEN AGENT USES THIS:  When user asks about risks or "is this safe to sign?"
```

### Tool 3 — key_clause_extraction_tool

```
NAME:     key_clause_extraction_tool
TRIGGER:  User asks about specific clauses or what the document contains
          e.g. "What are the key clauses?" / "Show me the termination clause"

INPUT PARAMETERS:
  document_id    — which document to analyze
  document_type  — optional hint for the type of document

INTERNAL FLOW:
  STEP 1: retrieve_document_text(document_id)
           → Full document text from Qdrant

  STEP 2: extract_key_clauses(text)
           → Auto-detects document type (contract/NDA/lease/invoice/etc.)
           → Extracts clauses relevant to that document type
           → Each clause has: name, excerpt, significance, status

  STEP 3: Return as JSON string

OUTPUT:  JSON clause extraction
  {
    document_type: "nda",
    clauses: [
      {
        clause_name: "Confidentiality Obligation",
        excerpt: "Party A agrees to keep all disclosed information…",
        significance: "Core obligation — defines what must be kept secret",
        status: "present"
      },
      {
        clause_name: "Termination",
        excerpt: "Either party may terminate with 30 days notice…",
        significance: "Exit mechanism — how the NDA ends",
        status: "present"
      }
    ]
  }

WHEN AGENT USES THIS:  When user wants to understand document structure
```

### Tool 4 — red_flag_scanner_tool

```
NAME:     red_flag_scanner_tool
TRIGGER:  User asks about warnings, dangers, unusual terms
          e.g. "Any red flags?" / "Is anything unusual in this NDA?"

INPUT PARAMETERS:
  document_id — which document to scan

INTERNAL FLOW:
  STEP 1: retrieve_document_text(document_id)
           → Full document text from Qdrant

  STEP 2: scan_red_flags(text)
           → Auto-detects document type
           → Loads the per-type checklist (e.g. NDA checklist has ~15 items)
           → Runs LLM evaluation: for each checklist item:
               "Is this dangerous clause PRESENT in the document?"
               "Is this protective clause MISSING from the document?"
           → Builds list of flagged items with severity

  STEP 3: Return as JSON string

OUTPUT:  JSON flag report
  {
    doc_type: "nda",
    total_checks: 15,
    flagged_count: 4,
    flags: [
      {
        label: "Perpetual confidentiality obligation",
        category: "dangerous",
        severity: "High",
        finding: "Clause found: 'obligations shall survive indefinitely'"
      },
      {
        label: "No return or destruction clause",
        category: "missing",
        severity: "Medium",
        finding: "No clause requiring return/destruction of confidential info"
      }
    ]
  }

WHEN AGENT USES THIS:  Proactively when user seems concerned about contract terms
```

### Tool 5 — document_compare_tool

```
NAME:     document_compare_tool
TRIGGER:  User uploads two documents and asks to compare them
          e.g. "Compare these two contracts" / "What changed in version 2?"

INPUT PARAMETERS:
  document_id_1 — first document
  document_id_2 — second document

INTERNAL FLOW:
  STEP 1: retrieve_document_text(document_id_1)
           → Full text of document 1 from Qdrant

  STEP 2: retrieve_document_text(document_id_2)
           → Full text of document 2 from Qdrant

  STEP 3: extract_key_clauses(text1)  [parallel with Step 4]
           → Get structured clause list for document 1

  STEP 4: extract_key_clauses(text2)  [parallel with Step 3]
           → Get structured clause list for document 2

  STEP 5: compare_documents(extraction1, extraction2, text1, text2)
           → Fuzzy clause matching: find the same clause in both docs
           → Word-level diff: show exact words added/removed/changed
           → Risk scoring: how severe is the change?
           → LLM enrichment: one-line human-readable summary per clause change

  STEP 6: Return as JSON string

OUTPUT:  JSON comparison report (per clause)
  {
    summary: { total_changes: 8, high_severity: 2, … },
    changes: [
      {
        clause_name: "Payment Terms",
        status: "modified",
        severity: "high",
        doc1: { excerpt: "payment within 30 days…" },
        doc2: { excerpt: "payment within 7 days…" },
        word_diff: [
          { text: "within ", tag: "equal" },
          { text: "30", tag: "delete" },
          { text: "7", tag: "insert" },
          { text: " days", tag: "equal" }
        ],
        summary: "Payment window shortened from 30 to 7 days — significant cash flow impact"
      }
    ]
  }

WHEN AGENT USES THIS:  When user has two documents and asks about differences
```

---

## 8. LangGraph Agent with Memory

### 8.1 — AgentState (The Shared Memory of One Turn)

```
AgentState is a TypedDict — a Python dictionary with typed fields.
It flows through every node in the graph during one conversation turn.

FIELDS:
  messages      → List of all messages so far (history + new + tool results)
                  Uses add_messages reducer — new messages are APPENDED
                  (not replaced) automatically by LangGraph

  session_id    → Which session this turn belongs to
                  Passed through unchanged

  document_id   → Which document is active for this conversation
                  Tools use this to scope their Qdrant queries

  user_id       → Who the user is
                  Used for multi-tenant document scoping

HOW STATE FLOWS:
  Initial state (built before graph.invoke):
    messages = [H1, A1, H2, A2, H3(new)]  ← history + new message
    session_id = "uuid-abc"
    document_id = "user_123_abc"
    user_id = "user_123"

  After agent_node runs:
    messages = [H1, A1, H2, A2, H3, A3]  ← A3 appended automatically

  After tool_node runs (if tool called):
    messages = [H1, A1, H2, A2, H3, A3, T3]  ← ToolMessage T3 appended

  After agent_node runs again:
    messages = [H1, A1, H2, A2, H3, A3, T3, A3_final]  ← final answer
```

### 8.2 — Graph Structure

```
                         ┌──────────┐
                         │  START   │
                         └────┬─────┘
                              │  Initial AgentState
                              ▼
                    ┌─────────────────────┐
                    │    agent_node        │
                    │                     │
                    │  gpt-4o receives:   │
                    │  • system prompt    │
                    │  • all messages     │
                    │  • tool schemas     │
                    │                     │
                    │  Outputs either:    │
                    │  • AIMessage with   │
                    │    tool_calls       │
                    │  • AIMessage with   │
                    │    plain text       │
                    └─────────┬───────────┘
                              │
                  ┌───────────▼────────────┐
                  │   should_continue()     │
                  │   conditional router    │
                  └───┬───────────────┬────┘
          tool_calls? │               │ no tool_calls
                YES   │               │ NO
                      ▼               ▼
           ┌────────────────┐    ┌─────────┐
           │   tools_node   │    │   END   │
           │                │    │         │
           │ ToolNode runs  │    │ Return  │
           │ the requested  │    │ final   │
           │ tool function  │    │ answer  │
           │                │    └─────────┘
           │ Appends result │
           │ as ToolMessage │
           └───────┬────────┘
                   │
                   └──────────────────▶ agent_node (loop back)
                     ToolMessage added to state, gpt-4o reads it next
```

### 8.3 — System Prompt Design

```
The system prompt is injected at the top of every agent call.
It tells gpt-4o:

  ROLE:
    You are an expert legal document analyst assistant.

  AVAILABLE TOOLS (and when to use each):
    rag_search          → for factual questions about document content
    risk_analysis       → when user asks about risks
    key_clause_extraction → when user asks about key clauses or structure
    red_flag_scanner    → when user asks about warnings or unusual terms
    document_compare    → when user has two documents and asks about differences

  BEHAVIOR RULES:
    • Always use rag_search first for content questions
    • Cite specific clauses/sections in answers
    • Be precise about legal terminology
    • Ask for clarification if intent is unclear

  CONTEXT INJECTION:
    session_id and document_id are injected into the prompt dynamically
    so the agent knows which document to focus on
```

### 8.4 — Multi-Tool Turn Example

```
USER: "What are the risks and are there any red flags I should worry about?"

TURN EXECUTION:

  ① agent_node (first pass)
     gpt-4o decides: this needs BOTH risk_analysis AND red_flag_scanner
     Outputs AIMessage with two tool_calls:
       tool_call_1: risk_analysis_tool(document_id="user_123_abc")
       tool_call_2: red_flag_scanner_tool(document_id="user_123_abc")

  ② tools_node
     Runs both tools (can run in parallel within ToolNode)
     Appends ToolMessage_1 (risk JSON result)
     Appends ToolMessage_2 (red flags JSON result)

  ③ agent_node (second pass)
     gpt-4o reads:
       • original question (H)
       • tool_call AIMessage (A with tool_calls)
       • ToolMessage_1 (risk results)
       • ToolMessage_2 (red flag results)
     No more tool_calls needed
     Outputs final AIMessage with combined natural-language answer

  ④ should_continue → "end" (no tool_calls in final AIMessage)

  ⑤ Memory saved, response returned to client
```

### 8.5 — Streaming Chat

```
DIFFERENCE FROM REGULAR CHAT:
  Regular:  graph.ainvoke(state) → waits for ALL processing → returns
  Streaming: graph.astream_events(state) → yields events AS they happen

EVENT TYPES DURING STREAMING:
  on_chat_model_stream → gpt-4o is generating a token → yield to client
  on_tool_start        → a tool is starting → (optional: notify client)
  on_tool_end          → a tool finished → (optional: notify client)
  on_chain_end         → graph finished → save memory

CLIENT-SIDE EXPERIENCE:
  → Tokens appear word by word in real time
  → User sees: "The main risks in this contract are..."
               then "...firstly, the unlimited liability clause..."
               (not waiting 10 seconds for full response)

SSE FORMAT (Server-Sent Events):
  Each token sent as:    data: The\n\n
                         data:  main\n\n
                         data:  risks\n\n
                         ...
                         data: [DONE]\n\n

  session_id sent first: data: {"session_id": "uuid-..."}\n\n
  → Client captures this to continue the conversation later
```

---

## 9. FastAPI Routes

### 9.1 — Upload Route

```
ENDPOINT:  POST /rag/upload
AUTH:      verify_api_key dependency (existing auth system)
CONTENT-TYPE: multipart/form-data

FORM FIELDS:
  file         → the PDF file (UploadFile)
  user_id      → string identifier for the user
  document_id  → (optional) override auto-generated ID

VALIDATION CHECKS:
  1. filename must end in ".pdf"
  2. file size must be ≤ 50 MB
  3. If fails → HTTP 400 or 413

SUCCESS RESPONSE (200):
  {
    "document_id": "user_123_a1b2c3d4e5f6",
    "filename": "contract.pdf",
    "chunks_stored": 42,
    "total_chars": 41200,
    "message": "Document ingested successfully."
  }

ERROR RESPONSE (422):
  If PyMuPDF extracts empty text (scanned image PDF without OCR):
  { "detail": { "error": "...", "message": "Could not extract text…" } }
```

### 9.2 — Chat Route

```
ENDPOINT:  POST /rag/chat
AUTH:      verify_api_key

REQUEST BODY (JSON):
  {
    "message": "What are the payment terms?",
    "session_id": "uuid-abc",      ← omit to start new session
    "user_id": "user_123",
    "document_id": "user_123_a1b2c3d4"  ← omit for multi-doc chat
  }

PROCESSING:
  1. If session_id missing → generate new UUID
  2. Load history from Qdrant sessions collection
  3. Run LangGraph agent (may call 1+ tools)
  4. Save updated history to Qdrant
  5. Return final AI response

SUCCESS RESPONSE (200):
  {
    "response": "The payment is due within 30 days of invoice…",
    "session_id": "uuid-abc",
    "document_id": "user_123_a1b2c3d4"
  }
```

### 9.3 — Streaming Chat Route

```
ENDPOINT:  POST /rag/chat/stream
AUTH:      verify_api_key

REQUEST BODY: same as /rag/chat

RESPONSE TYPE: text/event-stream (SSE)

EVENT STREAM FORMAT:
  data: {"session_id": "uuid-abc"}       ← first event: session info

  data: The                               ← subsequent events: tokens
  data:  payment
  data:  terms
  data:  require
  data:  30
  data:  days...
  ...
  data: [DONE]                            ← final event: stream complete

CLIENT HANDLING:
  → Parse first event to get session_id for follow-up requests
  → Concatenate all token events to reconstruct full response
  → Show [DONE] to know when to stop listening
```

### 9.4 — Direct Tool Routes

```
ENDPOINT:  POST /rag/analyze/risks
REQUEST:   { "document_id": "...", "document_type": "contract" }
RESPONSE:  { "result": "<JSON string from risk_analysis_tool>" }

ENDPOINT:  POST /rag/analyze/key-clauses
REQUEST:   { "document_id": "...", "document_type": "nda" }
RESPONSE:  { "result": "<JSON string from key_clause_extraction_tool>" }

ENDPOINT:  POST /rag/analyze/red-flags
REQUEST:   { "document_id": "..." }
RESPONSE:  { "result": "<JSON string from red_flag_scanner_tool>" }

ENDPOINT:  POST /rag/analyze/compare
REQUEST:   { "document_id_1": "...", "document_id_2": "..." }
RESPONSE:  { "result": "<JSON string from document_compare_tool>" }

WHY DIRECT ROUTES EXIST:
  The agent chat uses LLM reasoning to DECIDE which tool to call.
  These routes bypass that reasoning and call the tool DIRECTLY.

  USE AGENT CHAT WHEN:
    → User is having a conversation and may ask anything
    → You don't know in advance what analysis is needed

  USE DIRECT ROUTES WHEN:
    → Automated pipeline that always needs risk analysis on upload
    → Batch processing many documents
    → Faster (no LLM reasoning step — just tool → result)
```

### 9.5 — Document & Session Management Routes

```
ENDPOINT:  GET /rag/documents/{user_id}
PURPOSE:   List all documents a user has uploaded
FLOW:
  1. Qdrant scroll with filter: payload.user_id == user_id
  2. Deduplicate by document_id (keep only chunk_index=0 per document)
  3. Return: { "documents": [ {document_id, filename, uploaded_at} ] }

ENDPOINT:  DELETE /rag/document/{document_id}
PURPOSE:   Remove a document and all its vectors
FLOW:
  1. Qdrant delete with filter: payload.document_id == document_id
  2. All chunks for this document are removed
  3. Return: { "message": "Document deleted." }

ENDPOINT:  DELETE /rag/session/{session_id}
PURPOSE:   Clear chat history for a session
FLOW:
  1. Convert session_id → deterministic UUID
  2. Qdrant delete by UUID from "chat_sessions" collection
  3. Return: { "message": "Session cleared." }
```

---

## 10. Complete File Structure

```
pdf_ai_review/
│
├── rag/                              ← NEW MODULE
│   ├── __init__.py                   empty init file
│   ├── qdrant_client_setup.py        Qdrant connection + initialize_collections()
│   ├── ingestion.py                  PDF → chunks → embeddings → Qdrant
│   ├── retriever.py                  build_retriever() + retrieve_document_text()
│   ├── memory.py                     SessionMemory class (Qdrant-backed)
│   ├── tools.py                      5 @tool functions for LangGraph
│   └── agent.py                      LangGraph graph + chat() + stream_chat()
│
├── routes/
│   ├── route.py                      existing routes (unchanged)
│   ├── convert_route.py              existing routes (unchanged)
│   └── rag_route.py                  NEW: all /rag/* endpoints
│
├── feature_modules/                  existing (called by tools, unchanged)
│   ├── risk_detection.py             → analyze_document_risks(text)
│   ├── key_clause_extraction.py      → extract_key_clauses(text)
│   ├── red_flag_scanner.py           → scan_red_flags(text)
│   └── document_comparison.py        → compare_documents(ext1, ext2, t1, t2)
│
├── s_main.py                         UPDATED: add rag_router + initialize_collections()
├── .env                              UPDATED: add Qdrant + RAG env vars
└── requirements.txt                  UPDATED: add new dependencies
```

### What Calls What

```
rag_route.py
  → rag/ingestion.py        (for /upload)
  → rag/agent.py            (for /chat and /chat/stream)
  → rag/tools.py            (for /analyze/* direct endpoints)
  → rag/memory.py           (for /session/delete)
  → rag/qdrant_client_setup.py (for /documents list and /document delete)

rag/agent.py
  → rag/tools.py            (registers all 5 tools)
  → rag/memory.py           (load + save per turn)

rag/tools.py
  → rag/retriever.py        (rag_search_tool, retrieve_document_text)
  → feature_modules/risk_detection.py        (risk_analysis_tool)
  → feature_modules/key_clause_extraction.py (key_clause_extraction_tool)
  → feature_modules/red_flag_scanner.py      (red_flag_scanner_tool)
  → feature_modules/document_comparison.py   (document_compare_tool)

rag/retriever.py
  → rag/qdrant_client_setup.py  (get_sync_client)

rag/ingestion.py
  → rag/qdrant_client_setup.py  (get_async_client)

rag/memory.py
  → rag/qdrant_client_setup.py  (get_async_client)
```

---

## 11. Implementation Order (Phase by Phase)

```
PHASE 1 — INFRASTRUCTURE
─────────────────────────────────────────────────────────────────────
  Step 1 │ Add new packages to requirements.txt
         │  qdrant-client, langgraph, langchain, langchain-openai,
         │  langchain-qdrant, langchain-community, aiofiles,
         │  python-multipart

  Step 2 │ pip install -r requirements.txt

  Step 3 │ Add new env vars to .env
         │  QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL, CHAT_MODEL,
         │  CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL, MAX_HISTORY_TURNS

  Step 4 │ Start Qdrant
         │  docker run -p 6333:6333 qdrant/qdrant

  Step 5 │ Create rag/ directory with empty __init__.py


PHASE 2 — QDRANT CONNECTION
─────────────────────────────────────────────────────────────────────
  Step 6 │ Create rag/qdrant_client_setup.py
         │  → get_sync_client() function
         │  → get_async_client() function
         │  → initialize_collections() async function


PHASE 3 — DOCUMENT INGESTION
─────────────────────────────────────────────────────────────────────
  Step 7 │ Create rag/ingestion.py
         │  → extract_text_from_pdf(file_bytes) → str
         │  → chunk_text(text) → list of dicts
         │  → embed_and_store(file_bytes, filename, user_id) → dict
         │  → delete_document(document_id)

  Step 8 │ Create rag/retriever.py
         │  → build_retriever(document_id, user_id) → LangChain retriever
         │  → retrieve_document_text(document_id) → full text string


PHASE 4 — MEMORY
─────────────────────────────────────────────────────────────────────
  Step 9 │ Create rag/memory.py
         │  → SessionMemory class with load(), save(), clear()
         │  → _session_id_to_uuid() helper


PHASE 5 — TOOLS
─────────────────────────────────────────────────────────────────────
  Step 10│ Create rag/tools.py
         │  → rag_search_tool
         │  → risk_analysis_tool
         │  → key_clause_extraction_tool
         │  → red_flag_scanner_tool
         │  → document_compare_tool


PHASE 6 — LANGGRAPH AGENT
─────────────────────────────────────────────────────────────────────
  Step 11│ Create rag/agent.py
         │  → AgentState TypedDict
         │  → agent_node function
         │  → should_continue function
         │  → build_agent() → compiled LangGraph graph
         │  → chat() async function
         │  → stream_chat() async generator


PHASE 7 — API ROUTES
─────────────────────────────────────────────────────────────────────
  Step 12│ Create routes/rag_route.py
         │  → All 9 endpoints from Part 9

  Step 13│ Update s_main.py
         │  → Import rag_router from routes.rag_route
         │  → Import initialize_collections from rag.qdrant_client_setup
         │  → Add initialize_collections() call inside lifespan startup
         │  → Add app.include_router(rag_router)


PHASE 8 — TESTING
─────────────────────────────────────────────────────────────────────
  Step 14│ Test upload
         │  POST /rag/upload with a real PDF
         │  Verify: chunks_stored > 0, document_id returned

  Step 15│ Test basic chat
         │  POST /rag/chat with a factual question about the document
         │  Verify: response is grounded in document content

  Step 16│ Test session memory
         │  POST /rag/chat again with same session_id
         │  Ask a follow-up question referencing turn 1
         │  Verify: agent remembers context

  Step 17│ Test risk analysis trigger
         │  POST /rag/chat: "What are the risks in this contract?"
         │  Verify: agent calls risk_analysis_tool (check logs)

  Step 18│ Test red flag trigger
         │  POST /rag/chat: "Any red flags I should know about?"
         │  Verify: agent calls red_flag_scanner_tool

  Step 19│ Test document compare
         │  Upload second PDF → POST /rag/upload
         │  POST /rag/chat: "Compare document 1 and document 2"
         │  Verify: agent calls document_compare_tool

  Step 20│ Test streaming
         │  POST /rag/chat/stream
         │  Verify: tokens arrive progressively, [DONE] at end

  Step 21│ Test direct tool endpoints
         │  POST /rag/analyze/risks → verify JSON result without chat
         │  POST /rag/analyze/compare → verify comparison JSON
```

---

## 12. End-to-End Data Flow Summary

```
═══════════════════════════════════════════════════════════════════
UPLOAD FLOW (runs once per document)
═══════════════════════════════════════════════════════════════════

  PDF file bytes
      │
      ├─[PyMuPDF]──────────────────────────────▶ raw text string
      │
      ├─[RecursiveTextSplitter]────────────────▶ chunks list
      │    chunk_size=1000, overlap=200
      │
      ├─[OpenAI ada-002]───────────────────────▶ vector list
      │    1536 floats per chunk
      │
      └─[Qdrant upsert]────────────────────────▶ "documents" collection
           each chunk = one PointStruct


═══════════════════════════════════════════════════════════════════
CHAT FLOW (runs every message)
═══════════════════════════════════════════════════════════════════

  { message, session_id, document_id, user_id }
      │
      ├─[SessionMemory.load]───────────────────▶ past messages from Qdrant
      │
      ├─[LangGraph StateGraph.ainvoke]
      │     │
      │     ├─[agent_node / gpt-4o]
      │     │    reads: system + history + new message
      │     │    decides which tool(s) to call
      │     │
      │     ├─[ToolNode] ─ if tool called:
      │     │    ├─ rag_search_tool
      │     │    │     → Qdrant similarity search (top-K chunks)
      │     │    │
      │     │    ├─ risk_analysis_tool
      │     │    │     → Qdrant scroll (full text) → analyze_document_risks()
      │     │    │
      │     │    ├─ key_clause_extraction_tool
      │     │    │     → Qdrant scroll (full text) → extract_key_clauses()
      │     │    │
      │     │    ├─ red_flag_scanner_tool
      │     │    │     → Qdrant scroll (full text) → scan_red_flags()
      │     │    │
      │     │    └─ document_compare_tool
      │     │          → Qdrant scroll × 2 (both documents)
      │     │          → extract_key_clauses() × 2 (parallel)
      │     │          → compare_documents(ext1, ext2, text1, text2)
      │     │
      │     └─[agent_node / gpt-4o again]
      │          reads tool result → writes final AIMessage
      │
      ├─[SessionMemory.save]───────────────────▶ updated history to Qdrant
      │
      └─ return { response, session_id, document_id }


═══════════════════════════════════════════════════════════════════
STREAMING FLOW (same as CHAT, different delivery)
═══════════════════════════════════════════════════════════════════

  Same as CHAT FLOW but:
  astream_events() instead of ainvoke()
      │
      ├─ on_chat_model_stream events → yield token → SSE to client
      ├─ on_chain_end → save memory
      └─ final event: data: [DONE]
```

---

*Document generated for EaseSign PDF AI Review — 2026-04-23*
